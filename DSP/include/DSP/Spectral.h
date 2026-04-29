#pragma once

// SharedMath::DSP — Power Spectral Density helpers
//
// periodogram            — single-segment windowed-FFT PSD estimate
// welchPSD               — Welch's averaged periodogram method
// powerSpectralDensityDB — convert linear PSD to dB
// crossPowerSpectralDensity — cross-PSD via Welch averaging

#include "FFT.h"    // rfftFrequencies, FFTPlan, FFTDirection, FFTNorm
#include "Window.h" // makeWindow, WindowParams

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

namespace SharedMath::DSP {

// ── Scaling convention ────────────────────────────────────────────────────────
enum class PSDScaling {
    Density,   // V²/Hz — divide by (fs · Σw²)
    Spectrum   // V²    — divide by (Σw)²
};

// ── Return types ──────────────────────────────────────────────────────────────
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
inline SpectralResult periodogram(
    const std::vector<double>& signal,
    double sampleRate,
    const WindowParams& wp = {},
    PSDScaling scaling = PSDScaling::Density)
{
    if (signal.empty())
        throw std::invalid_argument("periodogram: signal is empty");
    if (sampleRate <= 0.0)
        throw std::invalid_argument("periodogram: sampleRate must be > 0");

    size_t N    = signal.size();
    size_t Nfft = 1;
    while (Nfft < N) Nfft <<= 1;

    auto win = makeWindow(N, wp);
    double winSumSq = 0.0, winSum = 0.0;
    for (double w : win) { winSumSq += w * w; winSum += w; }

    // Zero-pad windowed signal
    std::vector<std::complex<double>> cx(Nfft, {0.0, 0.0});
    for (size_t i = 0; i < N; ++i) cx[i] = signal[i] * win[i];

    FFTPlan::create(Nfft, {FFTDirection::Forward, FFTNorm::None}).execute(cx);

    size_t m = Nfft / 2 + 1;
    double scale = (scaling == PSDScaling::Density)
        ? 1.0 / (sampleRate * winSumSq)
        : 1.0 / (winSum * winSum);

    std::vector<double> psd(m);
    for (size_t k = 0; k < m; ++k) {
        psd[k] = std::norm(cx[k]) * scale;
        // One-sided: double interior bins; DC (k=0) and Nyquist (k=m-1 for even Nfft) are not doubled
        if (k > 0 && k < m - 1) psd[k] *= 2.0;
    }

    return {rfftFrequencies(Nfft, sampleRate), std::move(psd)};
}

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
inline SpectralResult welchPSD(
    const std::vector<double>& signal,
    double sampleRate,
    size_t frameSize,
    size_t hopSize = 0,
    const WindowParams& wp = {},
    PSDScaling scaling = PSDScaling::Density)
{
    if (signal.empty())
        throw std::invalid_argument("welchPSD: signal is empty");
    if (sampleRate <= 0.0)
        throw std::invalid_argument("welchPSD: sampleRate must be > 0");
    if (frameSize < 2)
        throw std::invalid_argument("welchPSD: frameSize must be >= 2");
    if (frameSize > signal.size())
        throw std::invalid_argument("welchPSD: frameSize exceeds signal length");

    if (hopSize == 0) hopSize = frameSize / 2;

    size_t Nfft = 1;
    while (Nfft < frameSize) Nfft <<= 1;

    auto win = makeWindow(frameSize, wp);
    double winSumSq = 0.0, winSum = 0.0;
    for (double w : win) { winSumSq += w * w; winSum += w; }

    size_t m = Nfft / 2 + 1;
    std::vector<double> psdAccum(m, 0.0);
    size_t numFrames = 0;

    for (size_t start = 0; start + frameSize <= signal.size(); start += hopSize) {
        std::vector<std::complex<double>> cx(Nfft, {0.0, 0.0});
        for (size_t i = 0; i < frameSize; ++i)
            cx[i] = signal[start + i] * win[i];
        FFTPlan::create(Nfft, {FFTDirection::Forward, FFTNorm::None}).execute(cx);
        for (size_t k = 0; k < m; ++k)
            psdAccum[k] += std::norm(cx[k]);
        ++numFrames;
    }

    if (numFrames == 0)
        throw std::invalid_argument("welchPSD: no complete frames in signal");

    double nf = static_cast<double>(numFrames);
    double scale = (scaling == PSDScaling::Density)
        ? 1.0 / (sampleRate * winSumSq * nf)
        : 1.0 / (winSum * winSum * nf);

    std::vector<double> psd(m);
    for (size_t k = 0; k < m; ++k) {
        psd[k] = psdAccum[k] * scale;
        if (k > 0 && k < m - 1) psd[k] *= 2.0;
    }

    return {rfftFrequencies(Nfft, sampleRate), std::move(psd)};
}

// ─────────────────────────────────────────────────────────────────────────────
// powerSpectralDensityDB — convert a linear PSD array to decibels
//
// psd:      result from periodogram() or welchPSD()
// refPower: reference power level (default 1.0)
// Returns:  10 · log10(psd[k] / refPower)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> powerSpectralDensityDB(
    const std::vector<double>& psd,
    double refPower = 1.0)
{
    std::vector<double> out(psd.size());
    for (size_t k = 0; k < psd.size(); ++k)
        out[k] = 10.0 * std::log10(std::max(psd[k] / refPower, 1e-300));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// crossPowerSpectralDensity — Welch cross-PSD estimate
//
// Estimates Pxy(f) = E[conj(X(f)) · Y(f)] averaged over overlapping frames.
// When x == y this equals the auto-PSD (Welch PSD).
// x and y must have the same length.
// ─────────────────────────────────────────────────────────────────────────────
inline CrossSpectralResult crossPowerSpectralDensity(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double sampleRate,
    size_t frameSize,
    size_t hopSize = 0,
    const WindowParams& wp = {},
    PSDScaling scaling = PSDScaling::Density)
{
    if (x.empty() || y.empty())
        throw std::invalid_argument("crossPowerSpectralDensity: signals are empty");
    if (x.size() != y.size())
        throw std::invalid_argument(
            "crossPowerSpectralDensity: x and y must have the same length");
    if (sampleRate <= 0.0)
        throw std::invalid_argument(
            "crossPowerSpectralDensity: sampleRate must be > 0");
    if (frameSize < 2)
        throw std::invalid_argument(
            "crossPowerSpectralDensity: frameSize must be >= 2");
    if (frameSize > x.size())
        throw std::invalid_argument(
            "crossPowerSpectralDensity: frameSize exceeds signal length");

    if (hopSize == 0) hopSize = frameSize / 2;

    size_t Nfft = 1;
    while (Nfft < frameSize) Nfft <<= 1;

    auto win = makeWindow(frameSize, wp);
    double winSumSq = 0.0, winSum = 0.0;
    for (double w : win) { winSumSq += w * w; winSum += w; }

    size_t m = Nfft / 2 + 1;
    std::vector<std::complex<double>> cpsdAccum(m, {0.0, 0.0});
    size_t numFrames = 0;

    for (size_t start = 0; start + frameSize <= x.size(); start += hopSize) {
        std::vector<std::complex<double>> cx(Nfft, {0.0, 0.0});
        std::vector<std::complex<double>> cy(Nfft, {0.0, 0.0});
        for (size_t i = 0; i < frameSize; ++i) {
            cx[i] = x[start + i] * win[i];
            cy[i] = y[start + i] * win[i];
        }
        auto plan = FFTPlan::create(Nfft, {FFTDirection::Forward, FFTNorm::None});
        plan.execute(cx);
        plan.execute(cy);
        for (size_t k = 0; k < m; ++k)
            cpsdAccum[k] += std::conj(cx[k]) * cy[k];
        ++numFrames;
    }

    if (numFrames == 0)
        throw std::invalid_argument(
            "crossPowerSpectralDensity: no complete frames in signal");

    double nf    = static_cast<double>(numFrames);
    double scale = (scaling == PSDScaling::Density)
        ? 1.0 / (sampleRate * winSumSq * nf)
        : 1.0 / (winSum * winSum * nf);

    std::vector<std::complex<double>> cpsd(m);
    for (size_t k = 0; k < m; ++k) {
        cpsd[k] = cpsdAccum[k] * scale;
        if (k > 0 && k < m - 1) cpsd[k] *= 2.0;
    }

    return {rfftFrequencies(Nfft, sampleRate), std::move(cpsd)};
}

} // namespace SharedMath::DSP
