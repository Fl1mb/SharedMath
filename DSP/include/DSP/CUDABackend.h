#pragma once

// SharedMath::DSP — CUDA acceleration stub
//
// When SHAREDMATH_CUDA_DSP is defined (set by CMake when CUDA is available
// and the option SharedMath_ENABLE_CUDA_DSP is ON), this header exposes
// GPU-accelerated DSP primitives backed by cuFFT / cuBLAS.
//
// Without CUDA the stubs below fall back to the CPU implementations so that
// user code compiles unchanged on platforms without a GPU.
//
// Current GPU-accelerated routines (when SHAREDMATH_CUDA_DSP is defined):
//   rfftCUDA(signal)       → same result as rfft(), computed on the GPU
//   irfftCUDA(bins, n)     → same result as irfft()
//
// All other DSP functions remain CPU-only.

#include "FFT.h"

#ifdef SHAREDMATH_CUDA_DSP

// ── Forward declarations for the CUDA translation unit ────────────────────────
namespace SharedMath::DSP::CUDA {

// GPU rfft:  transfers signal to device, runs cuFFT, transfers result back.
std::vector<std::complex<double>> rfftCUDA(const std::vector<double>& signal);

// GPU irfft: transfers bins to device, runs inverse cuFFT, transfers back.
// n: expected output length (needed to disambiguate even/odd transforms).
std::vector<double> irfftCUDA(
    const std::vector<std::complex<double>>& bins,
    size_t n);

} // namespace SharedMath::DSP::CUDA

#else // ── CPU fallback ──────────────────────────────────────────────────────

#include <vector>
#include <complex>

namespace SharedMath::DSP::CUDA {

inline std::vector<std::complex<double>> rfftCUDA(
    const std::vector<double>& signal)
{
    return SharedMath::DSP::rfft(signal);
}

inline std::vector<double> irfftCUDA(
    const std::vector<std::complex<double>>& bins,
    size_t n)
{
    return SharedMath::DSP::irfft(bins, n);
}

} // namespace SharedMath::DSP::CUDA

#endif // SHAREDMATH_CUDA_DSP
