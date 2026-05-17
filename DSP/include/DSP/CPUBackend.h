#pragma once

#include "FFTBackend.h"
#include "FFTConfig.h"

#include <complex>
#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

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
    void prepare(size_t n, const FFTConfig& cfg) override;

    /// ── IFFTBackend::execute ─────────────────────────────────────────────
    void execute(std::complex<double>* data) const override;

    /// ── IFFTBackend::name ────────────────────────────────────────────────
    const char* name() const noexcept override;

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

    void applyNorm(std::complex<double>* data) const;
};

} // namespace SharedMath::DSP