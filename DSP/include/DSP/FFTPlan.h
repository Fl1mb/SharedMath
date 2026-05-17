#pragma once

#include "FFTConfig.h"
#include "FFTBackend.h"
#include "CPUBackend.h"

#include <complex>
#include <vector>
#include <memory>
#include <cstddef>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// FFTPlan — FFTW-inspired plan/execute model
///
/// The plan precomputes all tables (twiddle factors, bit-reversal) once.
/// The resulting object can be executed on any number of different buffers of
/// the same length without recomputation.
///
/// ── Quick-start ──────────────────────────────────────────────────────────────
///
///   // Forward FFT (no normalization):
///   auto plan = FFTPlan::create(1024);
///   plan.execute(signal);
///
///   // Inverse FFT with 1/N normalization:
///   auto iplan = FFTPlan::create(1024, {FFTDirection::Inverse, FFTNorm::ByN});
///   iplan.execute(spectrum);
///
///   // Any-size FFT via Bluestein (e.g., prime length):
///   auto p = FFTPlan::create(997, {FFTDirection::Forward, FFTNorm::None,
///                                   FFTAlgorithm::Bluestein});
///
///   // Custom backend (e.g., future CUDA):
///   auto cuda = FFTPlan::create(1024, {}, std::make_unique<CUDABackend>());
///
/// ─────────────────────────────────────────────────────────────────────────────
class FFTPlan {
public:
    // ── Factory ───────────────────────────────────────────────────────────

    // Create with the default CPU backend.
    static FFTPlan create(size_t n, FFTConfig cfg = {});

    // Create with a custom backend (e.g. CUDABackend, OpenCLBackend …).
    // The plan takes ownership of the backend.
    static FFTPlan create(size_t n, FFTConfig cfg,
                          std::unique_ptr<IFFTBackend> backend);

    /// ── Execution ─────────────────────────────────────────────────────────

    /// In-place transform on a raw pointer (length must equal size()).
    void execute(std::complex<double>* data) const;

    /// In-place transform on a vector (throws on size mismatch).
    void execute(std::vector<std::complex<double>>& data) const;

    // Execute and return a copy (non-destructive, allocates a new vector).
    std::vector<std::complex<double>>
    executeConst(std::vector<std::complex<double>> data) const;

    /// ── Paired forward / inverse factory helpers ──────────────────────────

    /// Returns a matching inverse plan (same size, direction flipped, ByN norm).
    FFTPlan inversePlan(FFTNorm norm = FFTNorm::ByN) const;

    /// ── Metadata ──────────────────────────────────────────────────────────

    size_t           size()        const noexcept { return n_; }
    bool             isInverse()   const noexcept { return cfg_.direction == FFTDirection::Inverse; }
    const FFTConfig& config()      const noexcept { return cfg_; }
    const char*      backendName() const noexcept;

    /// Raw backend access — use to call device-specific methods on custom backends.
    IFFTBackend*       backend()       noexcept { return backend_.get(); }
    const IFFTBackend* backend() const noexcept { return backend_.get(); }

private:
    FFTPlan() = default;

    size_t                       n_ = 0;
    FFTConfig                    cfg_;
    std::unique_ptr<IFFTBackend> backend_;
};

} // namespace SharedMath::DSP