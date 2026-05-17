#pragma once

/// SharedMath::DSP — Short-Time Fourier Transform
///
/// stft()                 — analysis: real signal → complex time-frequency matrix
/// istft()                — synthesis: OLA reconstruction from STFTResult
/// magnitudeSpectrogram() — convenience: |X[frame][bin]|
/// powerSpectrogram()     — convenience: |X[frame][bin]|²

#include "Window.h"

#include <complex>
#include <vector>
#include <cstddef>

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
    std::vector<double> timeAxis() const;

    /// Frequency (Hz) for each bin: 0 .. sampleRate/2
    std::vector<double> freqAxis() const;
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
STFTResult stft(
    const std::vector<double>& signal,
    size_t fftSize,
    size_t hopSize,
    const std::vector<double>& window,
    double sampleRate = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// stft — convenience overload with WindowParams
//
// hopSize = 0  →  auto-set to fftSize/4 (75% overlap)
// ─────────────────────────────────────────────────────────────────────────────
STFTResult stft(
    const std::vector<double>& signal,
    size_t fftSize,
    size_t hopSize             = 0,
    const WindowParams& wp     = {},
    double sampleRate          = 1.0);

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
std::vector<double> istft(const STFTResult& result);

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeSpectrogram — |X[frame][bin]|
// Returns a 2D array [numFrames][numBins].
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<double>> magnitudeSpectrogram(
    const STFTResult& result);

// ─────────────────────────────────────────────────────────────────────────────
// powerSpectrogram — |X[frame][bin]|²
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<double>> powerSpectrogram(
    const STFTResult& result);

} // namespace SharedMath::DSP