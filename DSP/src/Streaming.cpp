/**
 * @file Streaming.cpp
 * @brief Implementation of stateful streaming filters.
 */

#include "DSP/Streaming.h"
#include "DSP/FIR.h"
#include "DSP/IIR.h"

#include <vector>
#include <cstddef>
#include <algorithm>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// FIRFilter Implementation
// ─────────────────────────────────────────────────────────────────────────────
FIRFilter::FIRFilter(const std::vector<double>& h)
{
    setCoefficients(h);
}

void FIRFilter::setCoefficients(const std::vector<double>& h)
{
    h_ = h;
    delay_.assign(h_.size(), 0.0);
    pos_ = 0;
}

void FIRFilter::reset()
{
    std::fill(delay_.begin(), delay_.end(), 0.0);
    pos_ = 0;
}

double FIRFilter::processSample(double x)
{
    if (h_.empty()) return x;

    const size_t M = h_.size();
    delay_[pos_] = x;

    double y = 0.0;
    size_t idx = pos_;
    for (size_t k = 0; k < M; ++k) {
        y += h_[k] * delay_[idx];
        idx = (idx == 0) ? M - 1 : idx - 1;
    }

    pos_ = (pos_ + 1 == M) ? 0 : pos_ + 1;
    return y;
}

std::vector<double> FIRFilter::processBlock(const std::vector<double>& input)
{
    std::vector<double> out(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        out[i] = processSample(input[i]);
    return out;
}

void FIRFilter::processInPlace(std::vector<double>& buffer)
{
    for (double& v : buffer)
        v = processSample(v);
}

// ─────────────────────────────────────────────────────────────────────────────
// IIRFilter Implementation
// ─────────────────────────────────────────────────────────────────────────────
IIRFilter::IIRFilter(const std::vector<BiquadCoeffs>& sections)
    : cascade_(sections) {}

void IIRFilter::setSections(const std::vector<BiquadCoeffs>& sections)
{
    cascade_ = BiquadCascade(sections);
}

const std::vector<BiquadCoeffs>& IIRFilter::sections() const noexcept
{
    return cascade_.getSections();
}

void IIRFilter::reset()
{
    cascade_.reset();
}

double IIRFilter::processSample(double x)
{
    return cascade_.process(x);
}

std::vector<double> IIRFilter::processBlock(const std::vector<double>& input)
{
    return cascade_.process(input);
}

void IIRFilter::processInPlace(std::vector<double>& buffer)
{
    for (double& v : buffer)
        v = cascade_.process(v);
}

} // namespace SharedMath::DSP
