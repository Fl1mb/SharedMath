#pragma once

/// SharedMath::DSP — Short-Time Fourier Transform
///
/// stft()                 — analysis: real signal → complex time-frequency matrix
/// istft()                — synthesis: OLA reconstruction from STFTResult
/// magnitudeSpectrogram() — convenience: |X[frame][bin]|
/// powerSpectrogram()     — convenience: |X[frame][bin]|²

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "FFT.h"
#include "Window.h"

#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// STFTResult — analysis output container
///
/// frames[i] holds the rfft output (fftSize/2+1 complex bins) for frame i.
/// The analysis window and signal metadata are stored for ISTFT reconstruction.
/// ─────────────────────────────────────────────────────────────────────────────
struct STFTResult {
    std::vector<std::vector<std::complex<double>>> frames;
    std::vector<double> window;
    size_t fftSize      = 0;
    size_t hopSize      = 0;
    size_t signalLength = 0;
    double sampleRate   = 1.0;

    size_t numFrames() const noexcept { return frames.size(); }
    size_t numBins()   const noexcept { return fftSize / 2 + 1; }

    /// Start time (seconds) of each frame
    std::vector<double> timeAxis() const {
        std::vector<double> t(frames.size());
        double dt = static_cast<double>(hopSize) / sampleRate;
        for (size_t i = 0; i < frames.size(); ++i)
            t[i] = static_cast<double>(i) * dt;
        return t;
    }

    /// Frequency (Hz) for each bin: 0 .. sampleRate/2
    std::vector<double> freqAxis() const {
        return rfftFrequencies(fftSize, sampleRate);
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// stft — Short-Time Fourier Transform (analysis)
//
// signal:     real input signal
// fftSize:    FFT length = window length (must be >= 2)
// hopSize:    step between successive frames in samples (must be > 0)
// window:     analysis window of length fftSize (use makeWindow())
// sampleRate: Hz, stored in result metadata only
//
// Only complete frames (no zero-padding at boundaries) are computed:
//   numFrames = max(0,  1 + (N − fftSize) / hopSize)
//
// Each frame stores rfft output: fftSize/2+1 complex bins.
// A single FFTPlan is pre-created and reused across all frames.
// ─────────────────────────────────────────────────────────────────────────────
inline STFTResult stft(
    const std::vector<double>& signal,
    size_t fftSize,
    size_t hopSize,
    const std::vector<double>& window,
    double sampleRate = 1.0)
{
    if (fftSize < 2)
        throw std::invalid_argument("stft: fftSize must be >= 2");
    if (hopSize == 0)
        throw std::invalid_argument("stft: hopSize must be > 0");
    if (window.size() != fftSize)
        throw std::invalid_argument("stft: window length must equal fftSize");

    STFTResult result;
    result.fftSize      = fftSize;
    result.hopSize      = hopSize;
    result.signalLength = signal.size();
    result.sampleRate   = sampleRate;
    result.window       = window;

    if (signal.size() < fftSize) return result;

    size_t numFrames = 1 + (signal.size() - fftSize) / hopSize;
    result.frames.resize(numFrames);

    auto fwdPlan   = FFTPlan::create(fftSize);
    size_t halfBins = fftSize / 2 + 1;

    std::vector<std::complex<double>> frame(fftSize);

    for (size_t i = 0; i < numFrames; ++i) {
        size_t start = i * hopSize;
        for (size_t k = 0; k < fftSize; ++k)
            frame[k] = signal[start + k] * window[k];

        fwdPlan.execute(frame);
        result.frames[i].assign(frame.begin(), frame.begin() + halfBins);
    }

    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
// stft — convenience overload with WindowParams
//
// hopSize = 0  →  auto-set to fftSize/4 (75% overlap)
// ─────────────────────────────────────────────────────────────────────────────
inline STFTResult stft(
    const std::vector<double>& signal,
    size_t fftSize,
    size_t hopSize             = 0,
    const WindowParams& wp     = {},
    double sampleRate          = 1.0)
{
    size_t hop = (hopSize == 0) ? fftSize / 4 : hopSize;
    if (hop == 0) hop = 1;
    return stft(signal, fftSize, hop, makeWindow(fftSize, wp), sampleRate);
}


/// ─────────────────────────────────────────────────────────────────────────────
/// istft — Inverse STFT via overlap-add (OLA)
///
/// Reconstructs the time-domain signal from an STFTResult.
///
/// Each frame is processed as:
///   y_i = IDFT(frame_i)   [= analysis_window * original_frame for unmodified STFT]
///   output[i*hop .. i*hop+fftSize-1] += y_i
///   norm  [i*hop .. i*hop+fftSize-1] += window
///
/// The output is normalized by the accumulated window sum, giving
/// perfect reconstruction for any COLA window (e.g. Hann at 50% / 75% overlap).
///
/// Output length: (numFrames-1)*hopSize + fftSize, then trimmed to
/// result.signalLength if it was recorded by stft().
/// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> istft(const STFTResult& result) {
    if (result.frames.empty()) return {};

    size_t fftSize   = result.fftSize;
    size_t hopSize   = result.hopSize;
    size_t numFrames = result.frames.size();
    const auto& win  = result.window;

    size_t outLen  = (numFrames - 1) * hopSize + fftSize;
    size_t lastPair = (fftSize % 2 == 0) ? (fftSize / 2 - 1)
                                          : ((fftSize - 1) / 2);
    size_t halfBins = fftSize / 2 + 1;

    std::vector<double> output(outLen, 0.0);
    std::vector<double> norm(outLen, 0.0);

    auto invPlan = FFTPlan::create(fftSize, {FFTDirection::Inverse, FFTNorm::ByN});
    std::vector<std::complex<double>> full(fftSize);

    for (size_t i = 0; i < numFrames; ++i) {
        const auto& half = result.frames[i];

        // Reconstruct full Hermitian spectrum
        std::fill(full.begin(), full.end(), std::complex<double>{0.0, 0.0});
        for (size_t k = 0; k < halfBins && k < half.size(); ++k)
            full[k] = half[k];
        for (size_t k = 1; k <= lastPair; ++k)
            full[fftSize - k] = std::conj(half[k]);

        invPlan.execute(full);

        size_t start = i * hopSize;
        for (size_t k = 0; k < fftSize; ++k) {
            output[start + k] += full[k].real();
            norm[start + k]   += win[k];
        }
    }

    for (size_t n = 0; n < outLen; ++n)
        if (norm[n] > 1e-12) output[n] /= norm[n];

    if (result.signalLength > 0 && result.signalLength < outLen)
        output.resize(result.signalLength);

    return output;
}


// ─────────────────────────────────────────────────────────────────────────────
// magnitudeSpectrogram — |X[frame][bin]|
// Returns a 2D array [numFrames][numBins].
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<std::vector<double>> magnitudeSpectrogram(
    const STFTResult& result)
{
    std::vector<std::vector<double>> spec(result.numFrames());
    for (size_t i = 0; i < result.numFrames(); ++i) {
        const auto& f = result.frames[i];
        spec[i].resize(f.size());
        for (size_t k = 0; k < f.size(); ++k)
            spec[i][k] = std::abs(f[k]);
    }
    return spec;
}


// ─────────────────────────────────────────────────────────────────────────────
// powerSpectrogram — |X[frame][bin]|²
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<std::vector<double>> powerSpectrogram(
    const STFTResult& result)
{
    std::vector<std::vector<double>> spec(result.numFrames());
    for (size_t i = 0; i < result.numFrames(); ++i) {
        const auto& f = result.frames[i];
        spec[i].resize(f.size());
        for (size_t k = 0; k < f.size(); ++k)
            spec[i][k] = std::norm(f[k]);
    }
    return spec;
}

} // namespace SharedMath::DSP
