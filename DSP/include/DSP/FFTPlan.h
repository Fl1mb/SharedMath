#pragma once

#include "FFTConfig.h"
#include "FFTBackend.h"
#include "CPUBackend.h"

#include <complex>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstddef>
#include <string>

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
    static FFTPlan create(size_t n, FFTConfig cfg = {}) {
        return create(n, cfg, std::make_unique<CPUBackend>());
    }

    // Create with a custom backend (e.g. CUDABackend, OpenCLBackend …).
    // The plan takes ownership of the backend.
    static FFTPlan create(size_t n, FFTConfig cfg,
                          std::unique_ptr<IFFTBackend> backend) {
        if (n == 0)
            throw std::invalid_argument("FFTPlan: transform size must be > 0");
        if (!backend)
            throw std::invalid_argument("FFTPlan: backend must not be null");

        FFTPlan p;
        p.n_       = n;
        p.cfg_     = cfg;
        p.backend_ = std::move(backend);
        p.backend_->prepare(n, cfg);
        return p;
    }

    /// ── Execution ─────────────────────────────────────────────────────────

    /// In-place transform on a raw pointer (length must equal size()).
    void execute(std::complex<double>* data) const {
        backend_->execute(data);
    }

    /// In-place transform on a vector (throws on size mismatch).
    void execute(std::vector<std::complex<double>>& data) const {
        if (data.size() != n_)
            throw std::invalid_argument(
                "FFTPlan::execute: data size (" + std::to_string(data.size()) +
                ") does not match plan size (" + std::to_string(n_) + ")");
        backend_->execute(data.data());
    }

    // Execute and return a copy (non-destructive, allocates a new vector).
    std::vector<std::complex<double>>
    executeConst(std::vector<std::complex<double>> data) const {
        execute(data);
        return data;
    }

    /// ── Paired forward / inverse factory helpers ──────────────────────────

    /// Returns a matching inverse plan (same size, direction flipped, ByN norm).
    FFTPlan inversePlan(FFTNorm norm = FFTNorm::ByN) const {
        FFTConfig icfg = cfg_;
        icfg.direction = FFTDirection::Inverse;
        icfg.norm      = norm;
        return create(n_, icfg);
    }

    /// ── Metadata ──────────────────────────────────────────────────────────

    size_t           size()        const noexcept { return n_; }
    bool             isInverse()   const noexcept { return cfg_.direction == FFTDirection::Inverse; }
    const FFTConfig& config()      const noexcept { return cfg_; }
    const char*      backendName() const noexcept { return backend_->name(); }

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
