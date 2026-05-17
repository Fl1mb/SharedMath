#pragma once

/// SharedMath::DSP — Hz-first filter design API
///
/// Convenient wrappers around the normalized-frequency FIR/IIR design functions.
/// All functions accept frequencies in Hz and the sampling rate, validate inputs,
/// and internally convert to the normalized-to-Nyquist convention used by the
/// underlying design functions.
///
/// Normalization: normalized = hz / (sampleRate / 2)  ← Nyquist = 1
///
/// FIR (windowed-sinc):
///   designFIRLowPassHz  designFIRHighPassHz
///   designFIRBandPassHz designFIRBandStopHz
///
/// IIR (Butterworth biquad sections):
///   designButterworthLowPassHz   designButterworthHighPassHz
///   designButterworthBandPassHz  designButterworthBandStopHz
///
/// Notch biquad (RBJ cookbook):
///   designNotchHz
///
/// RBJ audio-EQ cookbook biquads:
///   designRBJLowPassHz    designRBJHighPassHz
///   designRBJBandPassHz   designRBJNotchHz
///   designRBJAllPassHz
///   designRBJLowShelfHz   designRBJHighShelfHz
///   designRBJPeakingEQHz

#include "FIR.h"
#include "IIR.h"

#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// FIR Hz wrappers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<double> designFIRLowPassHz(
    size_t order, double cutoffHz, double sampleRate,
    const WindowParams& wp = {});

std::vector<double> designFIRHighPassHz(
    size_t order, double cutoffHz, double sampleRate,
    const WindowParams& wp = {});

std::vector<double> designFIRBandPassHz(
    size_t order, double lowHz, double highHz, double sampleRate,
    const WindowParams& wp = {});

std::vector<double> designFIRBandStopHz(
    size_t order, double lowHz, double highHz, double sampleRate,
    const WindowParams& wp = {});

// ─────────────────────────────────────────────────────────────────────────────
// IIR (Butterworth) Hz wrappers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<BiquadCoeffs> designButterworthLowPassHz(
    size_t order, double cutoffHz, double sampleRate);

std::vector<BiquadCoeffs> designButterworthHighPassHz(
    size_t order, double cutoffHz, double sampleRate);

std::vector<BiquadCoeffs> designButterworthBandPassHz(
    size_t order, double lowHz, double highHz, double sampleRate);

std::vector<BiquadCoeffs> designButterworthBandStopHz(
    size_t order, double lowHz, double highHz, double sampleRate);

// ─────────────────────────────────────────────────────────────────────────────
// designNotchHz — RBJ cookbook notch biquad
//
// freqHz:     notch center frequency in Hz
// sampleRate: samples per second
// q:          quality factor; higher Q → narrower notch (BW ≈ freqHz / Q)
//
// Returns a single BiquadCoeffs; apply with BiquadCascade or applyIIR.
// ─────────────────────────────────────────────────────────────────────────────
BiquadCoeffs designNotchHz(
    double freqHz,
    double sampleRate,
    double q = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// RBJ cookbook filters, Hz-first variants.
//
// All return a single normalized biquad section. Use BiquadCascade, IIRFilter,
// or applyIIR(signal, {section}) to process samples.
// ─────────────────────────────────────────────────────────────────────────────

BiquadCoeffs designRBJLowPassHz(
    double cutoffHz,
    double sampleRate,
    double q = 0.7071067811865476);

BiquadCoeffs designRBJHighPassHz(
    double cutoffHz,
    double sampleRate,
    double q = 0.7071067811865476);

BiquadCoeffs designRBJBandPassHz(
    double centerHz,
    double sampleRate,
    double q = 1.0);

BiquadCoeffs designRBJNotchHz(
    double freqHz,
    double sampleRate,
    double q = 1.0);

BiquadCoeffs designRBJAllPassHz(
    double centerHz,
    double sampleRate,
    double q = 1.0);

BiquadCoeffs designRBJPeakingEQHz(
    double centerHz,
    double sampleRate,
    double gainDB,
    double q = 1.0);

BiquadCoeffs designRBJLowShelfHz(
    double cutoffHz,
    double sampleRate,
    double gainDB,
    double q = 0.7071067811865476);

BiquadCoeffs designRBJHighShelfHz(
    double cutoffHz,
    double sampleRate,
    double gainDB,
    double q = 0.7071067811865476);

} // namespace SharedMath::DSP