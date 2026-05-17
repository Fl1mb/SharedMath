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

#include <complex>
#include <vector>
#include <cstddef>

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
std::vector<std::complex<double>> frequencyShift(
    const std::vector<std::complex<double>>& iq,
    double shiftHz,
    double sampleRate,
    double initialPhaseRad = 0.0);

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
double estimateFrequencyOffsetFromPeak(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize = 1024);

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
FrequencyCorrectionResult correctFrequencyOffset(
    const std::vector<std::complex<double>>& iq,
    const FrequencyCorrectionParams&         params);

} // namespace SharedMath::DSP

/// @} // DSP_FrequencyCorrection