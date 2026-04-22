#pragma once

// SharedMath::DSP — Convolution and Correlation
//
// Free functions for:
//   • Linear convolution  (FFT-based and direct)
//   • Circular (cyclic) convolution
//   • Streaming convolution: Overlap-Add and Overlap-Save
//   • Cross-correlation and autocorrelation (FFT-based)
//   • Normalized variants of correlation
//
// All functions operate on std::vector<double> and are header-only.

#include "FFTPlan.h"
#include "FFTConfig.h"

#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Output-length mode  (same semantics as NumPy / SciPy)
// ─────────────────────────────────────────────────────────────────────────────

enum class ConvolutionMode {
    Full,   // Full linear output: length = len(a) + len(b) − 1
    Same,   // Central part with length = max(len(a), len(b))
    Valid   // Only fully-overlapping part: length = max(la,lb) − min(la,lb) + 1
            // (empty when la == lb for correlate; 1 when la == lb for convolve)
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

inline size_t nextPow2C(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Build complex buffer of length fftN from a real vector (zero-padded)
inline std::vector<std::complex<double>>
toComplexPadded(const std::vector<double>& v, size_t fftN)
{
    std::vector<std::complex<double>> c(fftN, {0.0, 0.0});
    for (size_t i = 0; i < v.size() && i < fftN; ++i) c[i] = v[i];
    return c;
}

// Trim a Full-length result (la + lb − 1) to Same or Valid.
// Uses the same trimming rule for both convolution and correlation.
inline std::vector<double> trimOutput(
    const std::vector<double>& full,
    size_t la, size_t lb,
    ConvolutionMode mode)
{
    if (mode == ConvolutionMode::Full) return full;

    size_t fullLen = la + lb - 1;

    if (mode == ConvolutionMode::Same) {
        size_t sameLen = std::max(la, lb);
        size_t start   = (fullLen - sameLen) / 2;
        return {full.begin() + static_cast<std::ptrdiff_t>(start),
                full.begin() + static_cast<std::ptrdiff_t>(start + sameLen)};
    }

    // Valid
    if (la == 0 || lb == 0) return {};
    size_t minL = std::min(la, lb);
    if (fullLen < 2 * (minL - 1) + 1) return {};
    size_t validLen = fullLen - 2 * (minL - 1);  // = |la − lb| + 1
    size_t start    = minL - 1;
    return {full.begin() + static_cast<std::ptrdiff_t>(start),
            full.begin() + static_cast<std::ptrdiff_t>(start + validLen)};
}

} // namespace detail


// ═════════════════════════════════════════════════════════════════════════════
// LINEAR CONVOLUTION
// ═════════════════════════════════════════════════════════════════════════════

// ── FFT-based linear convolution — O(N log N) ────────────────────────────────
//
// Equivalent to the existing convolve() in FFT.h, but additionally supports
// Same and Valid output modes and is the reference implementation used by
// the streaming methods below.
inline std::vector<double> convolveLinear(
    const std::vector<double>& a,
    const std::vector<double>& b,
    ConvolutionMode mode = ConvolutionMode::Full)
{
    if (a.empty() || b.empty()) return {};
    size_t la = a.size(), lb = b.size();
    size_t outLen = la + lb - 1;
    size_t fftN   = detail::nextPow2C(outLen);

    auto ca = detail::toComplexPadded(a, fftN);
    auto cb = detail::toComplexPadded(b, fftN);

    auto fwdPlan = FFTPlan::create(fftN);
    fwdPlan.execute(ca);
    fwdPlan.execute(cb);

    for (size_t i = 0; i < fftN; ++i) ca[i] *= cb[i];

    FFTPlan::create(fftN, {FFTDirection::Inverse, FFTNorm::ByN}).execute(ca);

    std::vector<double> full(outLen);
    for (size_t i = 0; i < outLen; ++i) full[i] = ca[i].real();

    return detail::trimOutput(full, la, lb, mode);
}


// ── Direct (time-domain) convolution — O(N·M) ────────────────────────────────
//
// More efficient than FFT-based for very short kernels (M ≲ 16).
// Results are numerically identical to convolveLinear() up to floating-point
// rounding (no FFT round-trip error).
inline std::vector<double> convolveLinearDirect(
    const std::vector<double>& a,
    const std::vector<double>& b,
    ConvolutionMode mode = ConvolutionMode::Full)
{
    if (a.empty() || b.empty()) return {};
    size_t la = a.size(), lb = b.size();
    size_t fullLen = la + lb - 1;
    std::vector<double> full(fullLen, 0.0);

    for (size_t i = 0; i < la; ++i)
        for (size_t j = 0; j < lb; ++j)
            full[i + j] += a[i] * b[j];

    return detail::trimOutput(full, la, lb, mode);
}


// ── Circular (cyclic) convolution of length n ─────────────────────────────────
//
// circ[k] = Σ_{m=0}^{n-1} a[m] · b[(k − m) mod n]
//
// Both a and b are zero-padded (or folded via modular indexing) to length n.
// When n ≥ la + lb − 1, circular convolution equals linear convolution.
inline std::vector<double> convolveCircular(
    const std::vector<double>& a,
    const std::vector<double>& b,
    size_t n)
{
    if (n == 0 || a.empty() || b.empty()) return {};

    // Fold each input into a length-n buffer (handles truncation and aliasing)
    std::vector<std::complex<double>> ca(n, {0.0, 0.0});
    std::vector<std::complex<double>> cb(n, {0.0, 0.0});
    for (size_t i = 0; i < a.size(); ++i) ca[i % n] += a[i];
    for (size_t i = 0; i < b.size(); ++i) cb[i % n] += b[i];

    auto fwdPlan = FFTPlan::create(n);
    fwdPlan.execute(ca);
    fwdPlan.execute(cb);

    for (size_t i = 0; i < n; ++i) ca[i] *= cb[i];

    FFTPlan::create(n, {FFTDirection::Inverse, FFTNorm::ByN}).execute(ca);

    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = ca[i].real();
    return out;
}


// ── Overlap-Add — streaming linear convolution ────────────────────────────────
//
// Splits `signal` into non-overlapping blocks of `blockSize` samples, convolves
// each block with `kernel` via FFT, and accumulates with appropriate overlap.
//
// Produces the same result as convolveLinear(signal, kernel, Full).
// Advantage over a single FFT: the per-block FFT size depends only on the
// kernel length, not on the (potentially huge) signal length.
//
// blockSize: new input samples per block.
//   0 (default) → auto-selected as next power-of-2 above 4 × kernel length.
//   Choosing blockSize ≈ 4…8 × kernel.size() gives good FFT efficiency.
inline std::vector<double> convolveOverlapAdd(
    const std::vector<double>& signal,
    const std::vector<double>& kernel,
    size_t blockSize = 0)
{
    if (signal.empty() || kernel.empty()) return {};

    size_t kLen = kernel.size();
    if (blockSize == 0)
        blockSize = detail::nextPow2C(4 * kLen + 1);

    // FFT size must hold one block + the kernel overlap without circular aliasing
    size_t fftN = detail::nextPow2C(blockSize + kLen - 1);

    size_t outLen = signal.size() + kLen - 1;
    std::vector<double> out(outLen, 0.0);

    // Pre-transform the kernel once
    auto K = detail::toComplexPadded(kernel, fftN);
    auto fwdPlan = FFTPlan::create(fftN);
    auto invPlan = FFTPlan::create(fftN, {FFTDirection::Inverse, FFTNorm::ByN});
    fwdPlan.execute(K);

    for (size_t start = 0; start < signal.size(); start += blockSize) {
        size_t len = std::min(blockSize, signal.size() - start);

        // Zero-padded block
        std::vector<std::complex<double>> block(fftN, {0.0, 0.0});
        for (size_t i = 0; i < len; ++i) block[i] = signal[start + i];

        fwdPlan.execute(block);
        for (size_t i = 0; i < fftN; ++i) block[i] *= K[i];
        invPlan.execute(block);

        // Overlap-add: block result has length len + kLen − 1
        size_t blockOutLen = len + kLen - 1;
        for (size_t i = 0; i < blockOutLen && (start + i) < outLen; ++i)
            out[start + i] += block[i].real();
    }

    return out;
}


// ── Overlap-Save (Overlap-Discard) — streaming linear convolution ─────────────
//
// Alternative streaming method: uses a sliding input buffer of length fftN.
// The first (kLen − 1) output samples of each block contain circular-aliasing
// artefacts and are discarded; the remaining `step` samples are the valid output.
//
// Produces the same result as convolveLinear(signal, kernel, Full).
//
// blockSize: valid output samples per block (= fftN − kLen + 1).
//   0 (default) → fftN is chosen as next power-of-2 above 8 × kernel length.
inline std::vector<double> convolveOverlapSave(
    const std::vector<double>& signal,
    const std::vector<double>& kernel,
    size_t blockSize = 0)
{
    if (signal.empty() || kernel.empty()) return {};

    size_t kLen = kernel.size();

    // Derive fftN from blockSize or choose automatically
    size_t fftN, step;
    if (blockSize == 0) {
        fftN = detail::nextPow2C(8 * kLen + 1);
        step = fftN - (kLen - 1);
    } else {
        fftN = detail::nextPow2C(blockSize + kLen - 1);
        step = fftN - (kLen - 1);  // step may differ slightly from blockSize
    }
    if (step == 0) step = 1;

    // Pre-transform the kernel (zero-padded to fftN)
    auto K = detail::toComplexPadded(kernel, fftN);
    auto fwdPlan = FFTPlan::create(fftN);
    auto invPlan = FFTPlan::create(fftN, {FFTDirection::Inverse, FFTNorm::ByN});
    fwdPlan.execute(K);

    size_t outLen   = signal.size() + kLen - 1;
    size_t pad      = kLen - 1;           // zeros prepended
    size_t numBlocks = (outLen + step - 1) / step;
    size_t totalPad  = pad + numBlocks * step;  // total padded signal length

    // Padded input: (kLen−1) leading zeros, signal, trailing zeros
    std::vector<double> padded(totalPad, 0.0);
    for (size_t i = 0; i < signal.size(); ++i) padded[pad + i] = signal[i];

    std::vector<double> out;
    out.reserve(outLen);

    for (size_t b = 0; b < numBlocks; ++b) {
        size_t pos = b * step;

        std::vector<std::complex<double>> block(fftN, {0.0, 0.0});
        for (size_t i = 0; i < fftN; ++i)
            block[i] = (pos + i < totalPad) ? padded[pos + i] : 0.0;

        fwdPlan.execute(block);
        for (size_t i = 0; i < fftN; ++i) block[i] *= K[i];
        invPlan.execute(block);

        // Discard first (kLen−1) samples; keep the rest
        for (size_t i = pad; i < fftN && out.size() < outLen; ++i)
            out.push_back(block[i].real());
    }

    out.resize(outLen, 0.0);
    return out;
}


// ═════════════════════════════════════════════════════════════════════════════
// CROSS-CORRELATION
// ═════════════════════════════════════════════════════════════════════════════

// ── FFT-based cross-correlation — O(N log N) ─────────────────────────────────
//
// corr[lag] = Σ_n a[n] · b[n + lag]       (real inputs, so conj(a) = a)
//
// Output layout (Full mode, length = la + lb − 1):
//   index 0           → lag = −(la − 1)   [most negative]
//   index la − 1      → lag = 0            [zero lag]
//   index la + lb − 2 → lag = lb − 1       [most positive]
//
// This matches the SciPy / NumPy convention for real 1-D inputs.
inline std::vector<double> crossCorrelate(
    const std::vector<double>& a,
    const std::vector<double>& b,
    ConvolutionMode mode = ConvolutionMode::Full)
{
    if (a.empty() || b.empty()) return {};
    size_t la = a.size(), lb = b.size();
    size_t outLen = la + lb - 1;
    size_t fftN   = detail::nextPow2C(outLen);

    auto ca = detail::toComplexPadded(a, fftN);
    auto cb = detail::toComplexPadded(b, fftN);

    auto fwdPlan = FFTPlan::create(fftN);
    fwdPlan.execute(ca);
    fwdPlan.execute(cb);

    // Correlation theorem: IFFT(conj(A) · B)
    for (size_t i = 0; i < fftN; ++i) ca[i] = std::conj(ca[i]) * cb[i];

    FFTPlan::create(fftN, {FFTDirection::Inverse, FFTNorm::ByN}).execute(ca);

    // After IFFT, ca[k] = corr at lag k (positive), ca[fftN-k] = corr at lag -k.
    // Rearrange so that index 0 = most-negative lag -(la-1).
    std::vector<double> full(outLen);
    //   Negative lags -(la-1)..−1 live in ca[fftN-(la-1)..fftN-1]
    size_t negStart = fftN - (la - 1);
    for (size_t i = 0; i + 1 < la; ++i)          // la-1 negative-lag samples
        full[i] = ca[negStart + i].real();
    //   Zero lag + positive lags live in ca[0..lb-1]
    for (size_t i = 0; i < lb; ++i)
        full[la - 1 + i] = ca[i].real();

    return detail::trimOutput(full, la, lb, mode);
}


// ── Normalized cross-correlation ──────────────────────────────────────────────
//
// R_norm[lag] = corr(a, b)[lag] / sqrt(energy(a) · energy(b))
//
// For equal-length inputs the peak value is in [−1, +1].
// Output length and layout are identical to crossCorrelate(a, b, Full).
inline std::vector<double> normalizedCrossCorrelate(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    auto raw = crossCorrelate(a, b, ConvolutionMode::Full);
    if (raw.empty()) return {};

    double ea = 0.0, eb = 0.0;
    for (double v : a) ea += v * v;
    for (double v : b) eb += v * v;

    double norm = std::sqrt(ea * eb);
    if (norm < 1e-300)
        return std::vector<double>(raw.size(), 0.0);

    for (double& v : raw) v /= norm;
    return raw;
}


// ═════════════════════════════════════════════════════════════════════════════
// AUTOCORRELATION
// ═════════════════════════════════════════════════════════════════════════════

// ── Autocorrelation — R[lag] = Σ_n x[n] · x[n + lag] ─────────────────────────
//
// Full output length = 2·N − 1.
// The zero-lag value (maximum for a non-zero signal) is at index N − 1.
// The result is always symmetric: R[k] == R[−k].
inline std::vector<double> autoCorrelate(
    const std::vector<double>& x,
    ConvolutionMode mode = ConvolutionMode::Full)
{
    return crossCorrelate(x, x, mode);
}


// ── Normalized autocorrelation — R_norm[lag] = R[lag] / R[0] ─────────────────
//
// The zero-lag element equals exactly 1.0.
// Full output length = 2·N − 1, centre index = N − 1.
inline std::vector<double> normalizedAutoCorrelate(const std::vector<double>& x)
{
    auto raw = autoCorrelate(x, ConvolutionMode::Full);
    if (raw.empty()) return {};

    // Zero-lag is at index N-1
    double r0 = raw[x.size() - 1];
    if (std::abs(r0) < 1e-300)
        return std::vector<double>(raw.size(), 0.0);

    for (double& v : raw) v /= r0;
    return raw;
}

} // namespace SharedMath::DSP
