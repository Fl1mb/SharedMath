/**
 * @file STFT.cpp
 * @brief Implementation of Short-Time Fourier Transform functions.
 */

#include "STFT.h"
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

// ─────────────────────────────────────────────────────────────────────────────
// STFTResult member functions
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> STFTResult::timeAxis() const {
    std::vector<double> t(frames.size());
    double dt = static_cast<double>(hopSize) / sampleRate;
    for (size_t i = 0; i < frames.size(); ++i)
        t[i] = static_cast<double>(i) * dt;
    return t;
}

std::vector<double> STFTResult::freqAxis() const {
    return rfftFrequencies(fftSize, sampleRate);
}

// ─────────────────────────────────────────────────────────────────────────────
// stft (full version with explicit window)
// ─────────────────────────────────────────────────────────────────────────────
STFTResult stft(
    const std::vector<double>& signal,
    size_t fftSize,
    size_t hopSize,
    const std::vector<double>& window,
    double sampleRate)
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
// stft (convenience overload with WindowParams)
// ─────────────────────────────────────────────────────────────────────────────
STFTResult stft(
    const std::vector<double>& signal,
    size_t fftSize,
    size_t hopSize,
    const WindowParams& wp,
    double sampleRate)
{
    size_t hop = (hopSize == 0) ? fftSize / 4 : hopSize;
    if (hop == 0) hop = 1;
    return stft(signal, fftSize, hop, makeWindow(fftSize, wp), sampleRate);
}

// ─────────────────────────────────────────────────────────────────────────────
// istft
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> istft(const STFTResult& result) {
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
// magnitudeSpectrogram
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<double>> magnitudeSpectrogram(
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
// powerSpectrogram
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::vector<double>> powerSpectrogram(
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