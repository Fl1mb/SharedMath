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

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// FIRFilter
/// ─────────────────────────────────────────────────────────────────────────────
class FIRFilter {
public:
    FIRFilter() = default;
    explicit FIRFilter(const std::vector<double>& h);

    /// Replace taps and reset internal state.
    void setCoefficients(const std::vector<double>& h);

    const std::vector<double>& coefficients() const noexcept { return h_; }

    /// Zero the delay line without changing coefficients.
    void reset();

    /// Process a single sample through the FIR delay line.
    double processSample(double x);

    std::vector<double> processBlock(const std::vector<double>& input);
    void processInPlace(std::vector<double>& buffer);

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
    explicit IIRFilter(const std::vector<BiquadCoeffs>& sections);

    /// Replace biquad sections and reset state.
    void setSections(const std::vector<BiquadCoeffs>& sections);

    const std::vector<BiquadCoeffs>& sections() const noexcept;

    /// Zero all biquad states.
    void reset();

    double processSample(double x);

    std::vector<double> processBlock(const std::vector<double>& input);
    void processInPlace(std::vector<double>& buffer);

private:
    BiquadCascade cascade_;
};

} // namespace SharedMath::DSP