#pragma once

/// SharedMath::DSP — Stateful streaming filters
///
/// FIRFilter  — direct-form FIR with shift-register delay line
///   processSample(x)             → single output sample
///   processBlock(input)          → new vector
///   processInPlace(buffer)       → modifies buffer in place
///   reset()                      → zero delay-line state
///   setCoefficients(h)           → replace taps (resets state)
///   coefficients()               → const reference to taps
///
/// IIRFilter  — cascaded biquad sections (Transposed Direct Form II)
///   same interface, but setSections / sections instead of setCoefficients
///   processSample / processBlock / processInPlace / reset

#include "FIR.h"
#include "IIR.h"

#include <vector>
#include <cstddef>
#include <stdexcept>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// FIRFilter
/// ─────────────────────────────────────────────────────────────────────────────
class FIRFilter {
public:
    FIRFilter() = default;

    explicit FIRFilter(const std::vector<double>& h) { setCoefficients(h); }

    /// Replace taps and reset internal state.
    void setCoefficients(const std::vector<double>& h) {
        h_ = h;
        delay_.assign(h_.size(), 0.0);
        pos_ = 0;
    }

    const std::vector<double>& coefficients() const noexcept { return h_; }

    /// Zero the delay line without changing coefficients.
    void reset() {
        std::fill(delay_.begin(), delay_.end(), 0.0);
        pos_ = 0;
    }

    /// Process a single sample through the FIR delay line.
    double processSample(double x) {
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

    std::vector<double> processBlock(const std::vector<double>& input) {
        std::vector<double> out(input.size());
        for (size_t i = 0; i < input.size(); ++i)
            out[i] = processSample(input[i]);
        return out;
    }

    void processInPlace(std::vector<double>& buffer) {
        for (double& v : buffer)
            v = processSample(v);
    }

private:
    std::vector<double> h_;
    std::vector<double> delay_;  // circular delay line
    size_t              pos_ = 0;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// IIRFilter
/// ─────────────────────────────────────────────────────────────────────────────
class IIRFilter {
public:
    IIRFilter() = default;

    explicit IIRFilter(const std::vector<BiquadCoeffs>& sections)
        : cascade_(sections) {}

    /// Replace biquad sections and reset state.
    void setSections(const std::vector<BiquadCoeffs>& sections) {
        cascade_ = BiquadCascade(sections);
    }

    const std::vector<BiquadCoeffs>& sections() const noexcept {
        return cascade_.getSections();
    }

    /// Zero all biquad states.
    void reset() { cascade_.reset(); }

    double processSample(double x) { return cascade_.process(x); }

    std::vector<double> processBlock(const std::vector<double>& input) {
        return cascade_.process(input);
    }

    void processInPlace(std::vector<double>& buffer) {
        for (double& v : buffer)
            v = cascade_.process(v);
    }

private:
    BiquadCascade cascade_;
};

} // namespace SharedMath::DSP
