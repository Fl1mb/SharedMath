/**
 * @file FFTPlan.cpp
 * @brief Implementation of FFTPlan class.
 */

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "FFTBackend.h"
#include "CPUBackend.h"

#include <stdexcept>
#include <string>

namespace SharedMath::DSP {

// ── Factory ───────────────────────────────────────────────────────────

FFTPlan FFTPlan::create(size_t n, FFTConfig cfg)
{
    return create(n, cfg, std::make_unique<CPUBackend>());
}

FFTPlan FFTPlan::create(size_t n, FFTConfig cfg,
                        std::unique_ptr<IFFTBackend> backend)
{
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

void FFTPlan::execute(std::complex<double>* data) const
{
    backend_->execute(data);
}

void FFTPlan::execute(std::vector<std::complex<double>>& data) const
{
    if (data.size() != n_)
        throw std::invalid_argument(
            "FFTPlan::execute: data size (" + std::to_string(data.size()) +
            ") does not match plan size (" + std::to_string(n_) + ")");
    backend_->execute(data.data());
}

std::vector<std::complex<double>>
FFTPlan::executeConst(std::vector<std::complex<double>> data) const
{
    execute(data);
    return data;
}

/// ── Paired forward / inverse factory helpers ──────────────────────────

FFTPlan FFTPlan::inversePlan(FFTNorm norm) const
{
    FFTConfig icfg = cfg_;
    icfg.direction = FFTDirection::Inverse;
    icfg.norm      = norm;
    return create(n_, icfg);
}

/// ── Metadata ──────────────────────────────────────────────────────────

const char* FFTPlan::backendName() const noexcept
{
    return backend_->name();
}

} // namespace SharedMath::DSP