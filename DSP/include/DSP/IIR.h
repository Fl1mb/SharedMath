#pragma once

#include <vector>
#include <complex>
#include <cstddef>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// BiquadCoeffs — second-order section (SOS), Transposed Direct Form II
/// Difference equation:
///   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
/// (a0 is assumed normalized to 1)
/// ─────────────────────────────────────────────────────────────────────────────
struct BiquadCoeffs {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;

    /// Stability: poles are roots of z^2 + a1*z + a2 = 0; require |p| < 1.
    bool isStable() const;

    /// Transposed Direct Form II single-sample processing.
    inline double process(double x, double& s1, double& s2) const {
        const double y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        return y;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth low-pass design.
// `cutoff` is normalized frequency in (0, 1), where 1 is Nyquist.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BiquadCoeffs> designButterworthLowPass(
    size_t order,
    double cutoff);

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth high-pass design.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BiquadCoeffs> designButterworthHighPass(
    size_t order,
    double cutoff);

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth band-pass design.
// `cutoffLow`, `cutoffHigh` are normalized in (0, 1).
// Resulting filter has order = 2 * `order`.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BiquadCoeffs> designButterworthBandPass(
    size_t order,
    double cutoffLow,
    double cutoffHigh);

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth band-stop (notch) design.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BiquadCoeffs> designButterworthBandStop(
    size_t order,
    double cutoffLow,
    double cutoffHigh);

// ─────────────────────────────────────────────────────────────────────────────
// RBJ Audio-EQ-Cookbook biquads.
// `frequency` is normalized to Nyquist, i.e. lies in (0, 1),
// which maps to ω₀ = π · frequency (so frequency=1 is Nyquist).
// ─────────────────────────────────────────────────────────────────────────────
BiquadCoeffs designPeakingEQ(
    double centerFreq,
    double gainDB,
    double q = 1.0);

BiquadCoeffs designLowShelf(
    double cutoff,
    double gainDB,
    double q = 0.707);

BiquadCoeffs designHighShelf(
    double cutoff,
    double gainDB,
    double q = 0.707);

/// ─────────────────────────────────────────────────────────────────────────────
/// BiquadCascade — with per-section state.
/// ─────────────────────────────────────────────────────────────────────────────
class BiquadCascade {
public:
    BiquadCascade() = default;
    explicit BiquadCascade(const std::vector<BiquadCoeffs>& sections);

    void setCoefficients(const std::vector<BiquadCoeffs>& sections);

    inline double process(double x) {
        double y = x;
        for (size_t i = 0; i < sections_.size(); ++i) {
            y = sections_[i].process(y, states_[i].s1, states_[i].s2);
        }
        return y;
    }

    void process(std::vector<double>& signal);
    std::vector<double> process(const std::vector<double>& signal);

    void reset();
    const std::vector<BiquadCoeffs>& getSections() const { return sections_; }

private:
    struct State { double s1 = 0.0, s2 = 0.0; };
    std::vector<BiquadCoeffs> sections_;
    std::vector<State> states_;
};

// ─────────────────────────────────────────────────────────────────────────────
// One-shot helpers.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> applyIIR(
    const std::vector<double>& signal,
    const std::vector<BiquadCoeffs>& sections);

// Zero-phase forward-backward filtering.
std::vector<double> filtfiltIIR(
    const std::vector<double>& signal,
    const std::vector<BiquadCoeffs>& sections);

// ─────────────────────────────────────────────────────────────────────────────
// Frequency-response helpers.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> frequencyResponseIIR(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft);

std::vector<double> magnitudeResponseIIR(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft);

std::vector<double> magnitudeResponseDB(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft);

std::vector<double> frequencyAxis(size_t nfft, double sampleRate = 1.0);

} // namespace SharedMath::DSP