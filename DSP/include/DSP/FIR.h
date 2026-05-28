#pragma once

/// SharedMath::DSP — FIR filter design and application
///
/// Windowed-sinc design:  designFIRLowPass / HighPass / BandPass / BandStop
/// Auto-sized Kaiser FIR: designKaiserFIR
/// Filter application:    applyFIR  (Same-mode convolution)
/// Zero-phase filtering:  filtfilt  (forward + backward pass)

#include "Window.h"
#include "Convolution.h"

#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

enum class FIRType { LowPass, HighPass, BandPass, BandStop };

// ─────────────────────────────────────────────────────────────────────────────
// designFIRLowPass
//
// order: filter order (taps = order+1); forced to even (Type I).
// fc:    normalized cutoff ∈ (0, 1), where 1.0 = Nyquist (fs/2).
// wp:    window parameters (default: Hann, symmetric).
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRLowPass(
    size_t order,
    double fc,
    const WindowParams& wp = {});

// ─────────────────────────────────────────────────────────────────────────────
// designFIRHighPass
//
// Spectral inversion of a low-pass: h_hp = δ[center] − h_lp.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRHighPass(
    size_t order,
    double fc,
    const WindowParams& wp = {});

// ─────────────────────────────────────────────────────────────────────────────
// designFIRBandPass
//
// h_bp = (h_lp at fcHigh) − (h_lp at fcLow), multiplied by the same window.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRBandPass(
    size_t order,
    double fcLow,
    double fcHigh,
    const WindowParams& wp = {});

// ─────────────────────────────────────────────────────────────────────────────
// designFIRBandStop
//
// Spectral inversion of band-pass: h_bs = δ[center] − h_bp.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRBandStop(
    size_t order,
    double fcLow,
    double fcHigh,
    const WindowParams& wp = {});

// ─────────────────────────────────────────────────────────────────────────────
// designKaiserFIR — automatic order sizing from attenuation spec
//
// fc:              normalized cutoff (center of transition band) ∈ (0, 1)
// transitionWidth: full transition band width, normalized ∈ (0, 1)
// attenuationDB:   minimum stopband attenuation in dB (> 0)
// type:            LowPass or HighPass only; use two-cutoff variants for band
//
// Order formula (Harris, 1978):  M = (A − 7.95) / (2.285 · 2π · Δf)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designKaiserFIR(
    double fc,
    double transitionWidth,
    double attenuationDB,
    FIRType type = FIRType::LowPass);

// ─────────────────────────────────────────────────────────────────────────────
// applyFIR — apply FIR filter to a signal
//
// Computes linear convolution in "Same" mode so that the output length equals
// the input length and the group delay is centred out.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> applyFIR(
    const std::vector<double>& signal,
    const std::vector<double>& h);

// ─────────────────────────────────────────────────────────────────────────────
// filtfilt — zero-phase forward-backward FIR filtering
//
// Applies the filter forward then backward, cancelling group delay and
// yielding a squared magnitude response |H(f)|².
// Output length equals input length.
//
// Edge effects are present at the signal boundaries (zero-padding assumption).
// For minimal edge artefacts, ensure the signal is long relative to the
// filter order.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> filtfilt(
    const std::vector<double>& signal,
    const std::vector<double>& h);

} // namespace SharedMath::DSP