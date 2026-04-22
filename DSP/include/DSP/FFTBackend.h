#pragma once

#include "FFTConfig.h"
#include <complex>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// IFFTBackend — abstract execution backend
//
// Implement this interface to add a new compute target without changing any
// of the plan or user-API code.
//
// Minimal CUDA backend stub (future):
// ─────────────────────────────────────────────────────────────────────────────
//
//   class CUDABackend : public IFFTBackend {
//   public:
//       void prepare(size_t n, const FFTConfig& cfg) override {
//           n_ = n; inv_ = (cfg.direction == FFTDirection::Inverse);
//           // cufftPlan1d(&plan_, n, CUFFT_Z2Z, 1);
//       }
//       void execute(std::complex<double>* data) const override {
//           // cudaMemcpy device←host
//           // cufftExecZ2Z(plan_, d_data, d_data, inv_ ? CUFFT_INVERSE : CUFFT_FORWARD)
//           // cudaMemcpy host←device
//       }
//       const char* name() const noexcept override { return "CUDA (cuFFT)"; }
//   private:
//       // cufftHandle plan_;
//       size_t n_; bool inv_;
//   };
//
// ─────────────────────────────────────────────────────────────────────────────
class IFFTBackend {
public:
    virtual ~IFFTBackend() = default;

    // Called once when FFTPlan is built.
    // n   — transform length (same value that will appear in every execute() call)
    // cfg — full configuration: direction, normalization, algorithm hint
    virtual void prepare(size_t n, const FFTConfig& cfg) = 0;

    // Execute the transform **in-place** on `data` (length = n from prepare()).
    // On entry:  data contains the input signal / spectrum.
    // On exit:   data contains the transform result.
    virtual void execute(std::complex<double>* data) const = 0;

    // Human-readable identifier for logging and diagnostics.
    virtual const char* name() const noexcept = 0;
};

} // namespace SharedMath::DSP
