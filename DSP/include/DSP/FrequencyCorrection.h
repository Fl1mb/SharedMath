#pragma once

/**
 * @file FrequencyCorrection.h
 * @brief Frequency shift and carrier offset correction for IQ signals.
 *
 * @defgroup DSP_FrequencyCorrection Frequency Correction
 * @ingroup DSP
 * @{
 *
 * ### Example: shift a tone to baseband
 * @code{.cpp}
 * double offset = SharedMath::DSP::estimateFrequencyOffsetFromPeak(iq, 2e6);
 * auto corrected = SharedMath::DSP::frequencyShift(iq, -offset, 2e6);
 * @endcode
 *
 * @}
 */

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Configuration for correctFrequencyOffset().
 * @ingroup DSP_FrequencyCorrection
 */
struct FrequencyCorrectionParams {
    double sampleRate        = 1.0; ///< Sample rate in Hz.  Must be > 0.
    double frequencyOffsetHz = 0.0; ///< Carrier offset to remove in Hz.
    double initialPhaseRad   = 0.0; ///< Initial NCO phase in radians.
};

/**
 * @brief Output of correctFrequencyOffset().
 * @ingroup DSP_FrequencyCorrection
 */
struct FrequencyCorrectionResult {
    std::vector<std::complex<double>> iq;                    ///< Corrected IQ samples.
    double appliedFrequencyOffsetHz = 0.0; ///< The offset that was removed.
    double finalPhaseRad            = 0.0; ///< NCO phase at the last sample.
};

// ─────────────────────────────────────────────────────────────────────────────
// frequencyShift
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Multiply IQ by a complex exponential to shift the spectrum.
 *
 * Each output sample is:
 * @code
 *   out[n] = in[n] · exp(j · (initialPhaseRad + 2π · shiftHz · n / sampleRate))
 * @endcode
 *
 * A positive `shiftHz` moves the spectrum toward higher frequencies.
 * To bring a tone at `f0` to DC, pass `shiftHz = -f0`.
 *
 * @param iq             Input IQ samples.  Empty → returns empty.
 * @param shiftHz        Frequency shift in Hz.
 * @param sampleRate     Sample rate in Hz.  Must be > 0.
 * @param initialPhaseRad Starting NCO phase in radians.  Default 0.
 * @return Frequency-shifted IQ.
 * @throws std::invalid_argument if `sampleRate ≤ 0`.
 * @ingroup DSP_FrequencyCorrection
 */
inline std::vector<std::complex<double>> frequencyShift(
    const std::vector<std::complex<double>>& iq,
    double shiftHz,
    double sampleRate,
    double initialPhaseRad = 0.0)
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

/**
 * @brief Estimate the dominant frequency in an IQ block by locating the spectral peak.
 *
 * Applies a Hann window to the first `fftSize` samples, computes the FFT,
 * and maps the peak bin to a signed frequency in [−fs/2, +fs/2).
 *
 * @param iq         Complex IQ samples.  Empty → returns 0.
 * @param sampleRate Sample rate in Hz.  Must be > 0.
 * @param fftSize    FFT length.  Must be > 0.
 * @return Estimated dominant frequency in Hz.
 * @throws std::invalid_argument if `sampleRate ≤ 0` or `fftSize == 0`.
 * @ingroup DSP_FrequencyCorrection
 */
inline double estimateFrequencyOffsetFromPeak(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize = 1024)
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

/**
 * @brief Remove a known carrier frequency offset from an IQ stream.
 *
 * Equivalent to `frequencyShift(iq, -params.frequencyOffsetHz, params.sampleRate,
 * params.initialPhaseRad)` but also returns the applied offset and the final
 * NCO phase for seamless continuation of a streaming correction loop.
 *
 * @param iq     Input IQ samples.  Empty → returns empty result.
 * @param params Correction configuration.
 * @return FrequencyCorrectionResult with corrected IQ and final NCO state.
 * @throws std::invalid_argument if `params.sampleRate ≤ 0`.
 * @ingroup DSP_FrequencyCorrection
 */
inline FrequencyCorrectionResult correctFrequencyOffset(
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
