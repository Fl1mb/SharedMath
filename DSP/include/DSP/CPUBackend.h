#pragma once

#include "FFTBackend.h"
#include "FFTConfig.h"

#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// Internal implementation details — not part of the public API.
/// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

constexpr double DSP_PI = 3.14159265358979323846;

/// ── Bit manipulation helpers ─────────────────────────────────────────────────

inline bool isPow2(size_t n) noexcept { return n > 0 && (n & (n - 1)) == 0; }

inline size_t nextPow2(size_t n) noexcept {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/// ── Bit-reversal permutation table ───────────────────────────────────────────

inline std::vector<size_t> makeBitrev(size_t n) {
    size_t bits = 0;
    while ((size_t{1} << bits) < n) ++bits;
    std::vector<size_t> rev(n);
    for (size_t i = 0; i < n; ++i) {
        size_t r = 0, x = i;
        for (size_t b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
        rev[i] = r;
    }
    return rev;
}

/// ── Twiddle-factor table ─────────────────────────────────────────────────────
/// Returns n/2 entries: tw[k] = exp(sign·2πi·k/n), where sign = -1 (forward)
/// or +1 (inverse).  Stored for the butterfly: x[i+k] ± tw[k*stride]*x[i+k+half].

inline std::vector<std::complex<double>> makeTwiddles(size_t n, bool inverse) {
    std::vector<std::complex<double>> tw(n / 2);
    double sign = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < n / 2; ++k) {
        double angle = sign * 2.0 * DSP_PI * static_cast<double>(k)
                                           / static_cast<double>(n);
        tw[k] = {std::cos(angle), std::sin(angle)};
    }
    return tw;
}

// ── Cooley-Tukey radix-2 DIT (in-place, power-of-2 only) ────────────────────
//
// Algorithm (Decimation In Time):
//   1. Bit-reversal permutation rearranges input into DIT order.
//   2. log2(n) butterfly stages, each combining pairs with twiddle factors.
//
// Twiddle index stride: for stage with block length `len`, the stride through
// the precomputed table is n/len, so tw[k*(n/len)] = exp(−2πi·k/len).

inline void cooleyTukeyDIT(std::complex<double>* x, size_t n,
                            const std::vector<size_t>&               bitrev,
                            const std::vector<std::complex<double>>& twiddles)
{
    // Step 1: Bit-reversal permutation
    for (size_t i = 0; i < n; ++i)
        if (i < bitrev[i]) std::swap(x[i], x[bitrev[i]]);

    // Step 2: Butterfly stages
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half   = len >> 1;
        size_t stride = n / len;
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < half; ++k) {
                std::complex<double> t = twiddles[k * stride] * x[i + k + half];
                std::complex<double> u = x[i + k];
                x[i + k]        = u + t;
                x[i + k + half] = u - t;
            }
        }
    }
}

// ── Bluestein's chirp-z transform (arbitrary N) ───────────────────────────────
//
// Converts an arbitrary-length DFT into a convolution of length M (power of 2,
// M ≥ 2N−1), then evaluates the convolution via two radix-2 FFTs.
//
// For the forward DFT  X[k] = Σ x[n]·exp(−2πi·nk/N):
//   Using nk = (n² + k² − (k−n)²)/2:
//     X[k] = chirp[k] · Σ_n (x[n]·chirp[n]) · conj(chirp[k−n])
//   where chirp[n] = exp(−πi·n²/N)
//
// The sum is a linear convolution of a[n]=x[n]·chirp[n] with b[n]=conj(chirp[n]).
//
// bitrevM / twiddlesM are precomputed for size M (always forward / inverse
// internally uses the conj trick).

inline void bluestein(std::complex<double>* x, size_t n, bool inverse,
                      const std::vector<size_t>&               bitrevM,
                      const std::vector<std::complex<double>>& twiddlesM)
{
    double sign = inverse ? 1.0 : -1.0;
    double piOverN = DSP_PI / static_cast<double>(n);

    /// Chirp sequence: chirp[k] = exp(sign·πi·k²/N)
    std::vector<std::complex<double>> chirp(n);
    for (size_t k = 0; k < n; ++k) {
        double ang = sign * piOverN * static_cast<double>(k) * static_cast<double>(k);
        chirp[k] = {std::cos(ang), std::sin(ang)};
    }

    size_t M = nextPow2(2 * n - 1);

    // a[k] = x[k] · chirp[k], zero-padded to M
    std::vector<std::complex<double>> a(M, {0.0, 0.0});
    for (size_t k = 0; k < n; ++k)
        a[k] = x[k] * chirp[k];

    // b = conj(chirp), stored for circular convolution:
    //   b[0..N-1]       = conj(chirp[0..N-1])
    //   b[M-N+1..M-1]   = conj(chirp[N-1..1])   (wrap-around)
    std::vector<std::complex<double>> b(M, {0.0, 0.0});
    for (size_t k = 0; k < n; ++k) {
        b[k] = std::conj(chirp[k]);
        if (k > 0) b[M - k] = std::conj(chirp[k]);
    }

    /// Forward FFT of a and b (size M, always forward)
    cooleyTukeyDIT(a.data(), M, bitrevM, twiddlesM);
    cooleyTukeyDIT(b.data(), M, bitrevM, twiddlesM);

    // Pointwise multiply in frequency domain (= convolution in time)
    for (size_t k = 0; k < M; ++k) a[k] *= b[k];

    // IFFT via the conj trick: IFFT(x) = conj(FFT(conj(x))) / M
    for (size_t k = 0; k < M; ++k) a[k] = std::conj(a[k]);
    cooleyTukeyDIT(a.data(), M, bitrevM, twiddlesM);
    double invM = 1.0 / static_cast<double>(M);
    for (size_t k = 0; k < M; ++k) a[k] = std::conj(a[k]) * invM;

    // X[k] = chirp[k] · a[k]  for k = 0..N-1
    for (size_t k = 0; k < n; ++k)
        x[k] = chirp[k] * a[k];
}

} // namespace detail


/// ─────────────────────────────────────────────────────────────────────────────
/// CPUBackend
///
/// Pure-C++ FFT backend.  No external dependencies.
///
/// Algorithms:
///   • Cooley-Tukey radix-2 DIT  — O(N log N), requires N = 2^k
///   • Bluestein chirp-z          — O(N log N), works for any N
///
/// All expensive table computation (bit-reversal, twiddle factors) is done
/// once in prepare(); execute() is then a pure arithmetic operation.
/// ─────────────────────────────────────────────────────────────────────────────
class CPUBackend final : public IFFTBackend {
public:
    CPUBackend() = default;

    /// ── IFFTBackend::prepare ─────────────────────────────────────────────
    void prepare(size_t n, const FFTConfig& cfg) override {
        if (n == 0) throw std::invalid_argument("CPUBackend: n must be > 0");

        n_       = n;
        inverse_ = (cfg.direction == FFTDirection::Inverse);
        norm_    = cfg.norm;

        if (cfg.algorithm == FFTAlgorithm::CooleyTukey && !detail::isPow2(n))
            throw std::invalid_argument(
                "CPUBackend: CooleyTukey requires a power-of-2 transform size; "
                "use FFTAlgorithm::Auto or FFTAlgorithm::Bluestein for N=" + std::to_string(n));

        usePow2_ = detail::isPow2(n) && (cfg.algorithm != FFTAlgorithm::Bluestein);

        if (usePow2_) {
            // Precompute radix-2 tables for N
            bitrev_   = detail::makeBitrev(n);
            twiddles_ = detail::makeTwiddles(n, inverse_);
        } else {
            // Precompute radix-2 tables for M (Bluestein's internal FFT)
            M_         = detail::nextPow2(2 * n - 1);
            bitrevM_   = detail::makeBitrev(M_);
            twiddlesM_ = detail::makeTwiddles(M_, /*inverse=*/false);
        }
    }

    /// ── IFFTBackend::execute ─────────────────────────────────────────────
    void execute(std::complex<double>* data) const override {
        if (usePow2_)
            detail::cooleyTukeyDIT(data, n_, bitrev_, twiddles_);
        else
            detail::bluestein(data, n_, inverse_, bitrevM_, twiddlesM_);

        applyNorm(data);
    }

    /// ── IFFTBackend::name ────────────────────────────────────────────────
    const char* name() const noexcept override {
        return usePow2_ ? "CPU – Cooley-Tukey radix-2 DIT"
                        : "CPU – Bluestein chirp-z";
    }

private:
    size_t   n_       = 0;
    bool     inverse_ = false;
    bool     usePow2_ = false;
    size_t   M_       = 0;        // Bluestein padding length
    FFTNorm  norm_    = FFTNorm::None;

    // Radix-2 tables for the main transform (power-of-2 path)
    std::vector<size_t>               bitrev_;
    std::vector<std::complex<double>> twiddles_;

    // Radix-2 tables for Bluestein's internal convolution FFT
    std::vector<size_t>               bitrevM_;
    std::vector<std::complex<double>> twiddlesM_;

    void applyNorm(std::complex<double>* data) const {
        double scale = 1.0;
        if      (norm_ == FFTNorm::ByN)    scale = 1.0 / static_cast<double>(n_);
        else if (norm_ == FFTNorm::BySqrtN) scale = 1.0 / std::sqrt(static_cast<double>(n_));
        else return;

        for (size_t i = 0; i < n_; ++i) data[i] *= scale;
    }
};

} // namespace SharedMath::DSP
