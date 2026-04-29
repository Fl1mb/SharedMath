#pragma once

// SharedMath::DSP — Hz-first filter design API
//
// Convenient wrappers around the normalized-frequency FIR/IIR design functions.
// All functions accept frequencies in Hz and the sampling rate, validate inputs,
// and internally convert to the normalized-to-Nyquist convention used by the
// underlying design functions.
//
// Normalization: normalized = hz / (sampleRate / 2)  ← Nyquist = 1
//
// FIR (windowed-sinc):
//   designFIRLowPassHz  designFIRHighPassHz
//   designFIRBandPassHz designFIRBandStopHz
//
// IIR (Butterworth biquad sections):
//   designButterworthLowPassHz   designButterworthHighPassHz
//   designButterworthBandPassHz  designButterworthBandStopHz
//
// Notch biquad (RBJ cookbook):
//   designNotchHz

#include "FIR.h"
#include "IIR.h"

#include <vector>
#include <cmath>
#include <stdexcept>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (not part of the public API)
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

inline double hzToNyquistNorm(double hz, double sampleRate) noexcept {
    return hz / (sampleRate * 0.5);
}

inline void checkSampleRate(double sampleRate, const char* fn) {
    if (sampleRate <= 0.0)
        throw std::invalid_argument(
            std::string(fn) + ": sampleRate must be > 0");
}

inline void checkCutoffHz(double hz, double sampleRate, const char* fn) {
    checkSampleRate(sampleRate, fn);
    if (hz <= 0.0 || hz >= sampleRate * 0.5)
        throw std::invalid_argument(
            std::string(fn) + ": cutoffHz must be in (0, sampleRate/2)");
}

inline void checkBandHz(double lowHz, double highHz,
                         double sampleRate, const char* fn) {
    checkSampleRate(sampleRate, fn);
    if (lowHz <= 0.0 || highHz >= sampleRate * 0.5 || lowHz >= highHz)
        throw std::invalid_argument(
            std::string(fn) +
            ": require 0 < lowHz < highHz < sampleRate/2");
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// FIR Hz wrappers
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<double> designFIRLowPassHz(
    size_t order, double cutoffHz, double sampleRate,
    const WindowParams& wp = {})
{
    detail::checkCutoffHz(cutoffHz, sampleRate, "designFIRLowPassHz");
    return designFIRLowPass(order,
        detail::hzToNyquistNorm(cutoffHz, sampleRate), wp);
}

inline std::vector<double> designFIRHighPassHz(
    size_t order, double cutoffHz, double sampleRate,
    const WindowParams& wp = {})
{
    detail::checkCutoffHz(cutoffHz, sampleRate, "designFIRHighPassHz");
    return designFIRHighPass(order,
        detail::hzToNyquistNorm(cutoffHz, sampleRate), wp);
}

inline std::vector<double> designFIRBandPassHz(
    size_t order, double lowHz, double highHz, double sampleRate,
    const WindowParams& wp = {})
{
    detail::checkBandHz(lowHz, highHz, sampleRate, "designFIRBandPassHz");
    return designFIRBandPass(order,
        detail::hzToNyquistNorm(lowHz,  sampleRate),
        detail::hzToNyquistNorm(highHz, sampleRate), wp);
}

inline std::vector<double> designFIRBandStopHz(
    size_t order, double lowHz, double highHz, double sampleRate,
    const WindowParams& wp = {})
{
    detail::checkBandHz(lowHz, highHz, sampleRate, "designFIRBandStopHz");
    return designFIRBandStop(order,
        detail::hzToNyquistNorm(lowHz,  sampleRate),
        detail::hzToNyquistNorm(highHz, sampleRate), wp);
}

// ─────────────────────────────────────────────────────────────────────────────
// IIR (Butterworth) Hz wrappers
// ─────────────────────────────────────────────────────────────────────────────

inline std::vector<BiquadCoeffs> designButterworthLowPassHz(
    size_t order, double cutoffHz, double sampleRate)
{
    detail::checkCutoffHz(cutoffHz, sampleRate, "designButterworthLowPassHz");
    return designButterworthLowPass(order,
        detail::hzToNyquistNorm(cutoffHz, sampleRate));
}

inline std::vector<BiquadCoeffs> designButterworthHighPassHz(
    size_t order, double cutoffHz, double sampleRate)
{
    detail::checkCutoffHz(cutoffHz, sampleRate, "designButterworthHighPassHz");
    return designButterworthHighPass(order,
        detail::hzToNyquistNorm(cutoffHz, sampleRate));
}

inline std::vector<BiquadCoeffs> designButterworthBandPassHz(
    size_t order, double lowHz, double highHz, double sampleRate)
{
    detail::checkBandHz(lowHz, highHz, sampleRate, "designButterworthBandPassHz");
    return designButterworthBandPass(order,
        detail::hzToNyquistNorm(lowHz,  sampleRate),
        detail::hzToNyquistNorm(highHz, sampleRate));
}

inline std::vector<BiquadCoeffs> designButterworthBandStopHz(
    size_t order, double lowHz, double highHz, double sampleRate)
{
    detail::checkBandHz(lowHz, highHz, sampleRate, "designButterworthBandStopHz");
    return designButterworthBandStop(order,
        detail::hzToNyquistNorm(lowHz,  sampleRate),
        detail::hzToNyquistNorm(highHz, sampleRate));
}

// ─────────────────────────────────────────────────────────────────────────────
// designNotchHz — RBJ cookbook notch biquad
//
// freqHz:     notch center frequency in Hz
// sampleRate: samples per second
// q:          quality factor; higher Q → narrower notch (BW ≈ freqHz / Q)
//
// Returns a single BiquadCoeffs; apply with BiquadCascade or applyIIR.
// ─────────────────────────────────────────────────────────────────────────────
inline BiquadCoeffs designNotchHz(
    double freqHz,
    double sampleRate,
    double q = 1.0)
{
    detail::checkCutoffHz(freqHz, sampleRate, "designNotchHz");
    if (q <= 0.0)
        throw std::invalid_argument("designNotchHz: q must be > 0");

    // ω₀ = 2π f₀ / fs  (digital frequency in radians/sample)
    const double omega0 = 2.0 * detail::IIR_PI * freqHz / sampleRate;
    const double cos_w  = std::cos(omega0);
    const double alpha  = std::sin(omega0) / (2.0 * q);

    // RBJ notch:  b0=1, b1=-2cos(ω₀), b2=1
    //             a0=1+α, a1=-2cos(ω₀), a2=1-α
    const double a0 = 1.0 + alpha;
    const double b0 = 1.0;
    const double b1 = -2.0 * cos_w;
    const double b2 = 1.0;
    const double a1 = -2.0 * cos_w;
    const double a2 = 1.0 - alpha;

    return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
}

} // namespace SharedMath::DSP
