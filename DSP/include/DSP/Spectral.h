#pragma once

/// SharedMath::DSP — Power Spectral Density helpers
///
/// periodogram            — single-segment windowed-FFT PSD estimate
/// welchPSD               — Welch's averaged periodogram method
/// powerSpectralDensityDB — convert linear PSD to dB
/// crossPowerSpectralDensity — cross-PSD via Welch averaging

#include "Window.h" // makeWindow, WindowParams

#include <vector>
#include <complex>
#include <cstddef>

namespace SharedMath::DSP {

/// ── Scaling convention ────────────────────────────────────────────────────────
enum class PSDScaling {
    Density,   // V²/Hz — divide by (fs · Σw²)
    Spectrum   // V²    — divide by (Σw)²
};

/// ── Return types ──────────────────────────────────────────────────────────────
struct SpectralResult {
    std::vector<double> frequencies;  // Hz, one-sided (0..fs/2)
    std::vector<double> psd;          // power spectral density / spectrum
};

struct CrossSpectralResult {
    std::vector<double>               frequencies;  // Hz, one-sided
    std::vector<std::complex<double>> cpsd;          // complex CPSD
};

// ─────────────────────────────────────────────────────────────────────────────
// periodogram — single-segment PSD estimate via windowed FFT
//
// signal:     real input samples
// sampleRate: sampling rate in Hz
// wp:         window parameters (default: Hann)
// scaling:    Density [V²/Hz] or Spectrum [V²]
//
// The signal is windowed, zero-padded to the next power of 2, and transformed.
// Returns a one-sided (DC..Nyquist) estimate with size nextPow2(N)/2+1.
// ─────────────────────────────────────────────────────────────────────────────
SpectralResult periodogram(
    const std::vector<double>& signal,
    double sampleRate,
    const WindowParams& wp = {},
    PSDScaling scaling = PSDScaling::Density);

// ─────────────────────────────────────────────────────────────────────────────
// welchPSD — Welch's averaged periodogram method
//
// signal:    input samples
// sampleRate: Hz
// frameSize: segment length in samples
// hopSize:   step between frames; 0 → frameSize/2 (50 % overlap)
// wp:        window (default: Hann)
// scaling:   Density or Spectrum
//
// Frequency resolution ≈ sampleRate / nextPow2(frameSize).
// ─────────────────────────────────────────────────────────────────────────────
SpectralResult welchPSD(
    const std::vector<double>& signal,
    double sampleRate,
    size_t frameSize,
    size_t hopSize = 0,
    const WindowParams& wp = {},
    PSDScaling scaling = PSDScaling::Density);

// ─────────────────────────────────────────────────────────────────────────────
// powerSpectralDensityDB — convert a linear PSD array to decibels
//
// psd:      result from periodogram() or welchPSD()
// refPower: reference power level (default 1.0)
// Returns:  10 · log10(psd[k] / refPower)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> powerSpectralDensityDB(
    const std::vector<double>& psd,
    double refPower = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// crossPowerSpectralDensity — Welch cross-PSD estimate
//
// Estimates Pxy(f) = E[conj(X(f)) · Y(f)] averaged over overlapping frames.
// When x == y this equals the auto-PSD (Welch PSD).
// x and y must have the same length.
// ─────────────────────────────────────────────────────────────────────────────
CrossSpectralResult crossPowerSpectralDensity(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double sampleRate,
    size_t frameSize,
    size_t hopSize = 0,
    const WindowParams& wp = {},
    PSDScaling scaling = PSDScaling::Density);

} // namespace SharedMath::DSP