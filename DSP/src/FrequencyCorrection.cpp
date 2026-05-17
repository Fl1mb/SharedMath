/**
 * @file FrequencyCorrection.cpp
 * @brief Implementation of frequency shift and carrier offset correction.
 */

#include "FrequencyCorrection.h"
#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// frequencyShift
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> frequencyShift(
    const std::vector<std::complex<double>>& iq,
    double shiftHz,
    double sampleRate,
    double initialPhaseRad)
{
    if (sampleRate <= 0.0)
        throw std::invalid_argument("frequencyShift: sampleRate must be > 0");
    if (iq.empty()) return {};

    const size_t N        = iq.size();
    const double phaseInc = 2.0 * M_PI * shiftHz / sampleRate;
    std::vector<std::complex<double>> out(N);
    for (size_t n = 0; n < N; ++n)
        out[n] = iq[n] *
            std::polar(1.0, initialPhaseRad + phaseInc * static_cast<double>(n));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// estimateFrequencyOffsetFromPeak
// ─────────────────────────────────────────────────────────────────────────────
double estimateFrequencyOffsetFromPeak(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize)
{
    if (sampleRate <= 0.0)
        throw std::invalid_argument(
            "estimateFrequencyOffsetFromPeak: sampleRate must be > 0");
    if (fftSize == 0)
        throw std::invalid_argument(
            "estimateFrequencyOffsetFromPeak: fftSize must be > 0");
    if (iq.empty()) return 0.0;

    const size_t M     = fftSize;
    const size_t N     = iq.size();
    const double binHz = sampleRate / static_cast<double>(M);

    const size_t winLen = std::min(N, M);
    auto win = windowHann(winLen, /*symmetric=*/false);

    std::vector<std::complex<double>> frame(M, {0.0, 0.0});
    for (size_t i = 0; i < winLen; ++i)
        frame[i] = iq[i] * win[i];

    FFTPlan::create(M, {FFTDirection::Forward, FFTNorm::None}).execute(frame);

    // Find peak bin index (unshifted FFT output)
    size_t peakBin = 0;
    double peakMag = -1.0;
    for (size_t k = 0; k < M; ++k) {
        const double m = std::norm(frame[k]);
        if (m > peakMag) { peakMag = m; peakBin = k; }
    }

    // Map unshifted bin to signed frequency:
    //   bin < M/2  → positive frequency  (peakBin · binHz)
    //   bin ≥ M/2  → negative frequency  ((peakBin − M) · binHz)
    return (peakBin < M / 2)
        ? static_cast<double>(peakBin) * binHz
        : (static_cast<double>(peakBin) - static_cast<double>(M)) * binHz;
}

// ─────────────────────────────────────────────────────────────────────────────
// correctFrequencyOffset
// ─────────────────────────────────────────────────────────────────────────────
FrequencyCorrectionResult correctFrequencyOffset(
    const std::vector<std::complex<double>>& iq,
    const FrequencyCorrectionParams&         params)
{
    if (params.sampleRate <= 0.0)
        throw std::invalid_argument(
            "correctFrequencyOffset: sampleRate must be > 0");

    FrequencyCorrectionResult res;
    res.appliedFrequencyOffsetHz = params.frequencyOffsetHz;

    if (iq.empty()) {
        res.finalPhaseRad = params.initialPhaseRad;
        return res;
    }

    const size_t N        = iq.size();
    const double phaseInc = -2.0 * M_PI * params.frequencyOffsetHz / params.sampleRate;
    res.iq.resize(N);
    for (size_t n = 0; n < N; ++n)
        res.iq[n] = iq[n] *
            std::polar(1.0, params.initialPhaseRad + phaseInc * static_cast<double>(n));

    // Wrap final phase to keep it numerically bounded
    res.finalPhaseRad = std::fmod(
        params.initialPhaseRad + phaseInc * static_cast<double>(N),
        2.0 * M_PI);
    return res;
}

} // namespace SharedMath::DSP

/// @} // DSP_FrequencyCorrection