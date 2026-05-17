#pragma once

/**
 * @file Waterfall.h
 * @brief Short-time power spectrum matrix for spectrogram / waterfall display.
 *
 * @defgroup DSP_Waterfall Waterfall
 * @ingroup DSP
 * @{
 *
 * computeWaterfall() slices the IQ stream into overlapping Hann-windowed frames,
 * FFTs each frame, and assembles the result into a 2-D power-in-dBFS matrix
 * with associated time and frequency axes.
 *
 * ### Example
 * @code{.cpp}
 * SharedMath::DSP::WaterfallParams p;
 * p.sampleRate = 2e6;
 * p.fftSize    = 1024;
 * p.overlap    = 0.75;
 * p.centered   = true;   // frequency axis: -1 MHz … +1 MHz
 *
 * auto wf = SharedMath::DSP::computeWaterfall(iq, p);
 * // wf.powerDb[timeIdx][freqIdx]
 * @endcode
 *
 * @}
 */

#include <cstddef>
#include <complex>
#include <vector>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Data types
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Configuration for computeWaterfall().
 * @ingroup DSP_Waterfall
 */
struct WaterfallParams {
    double sampleRate = 1.0;    ///< Sample rate in Hz.  Must be > 0.
    size_t fftSize    = 1024;   ///< Frame length / FFT size.  Must be > 0.
    double overlap    = 0.5;    ///< Frame overlap in [0, 1).
    bool   centered   = true;   ///< If true, apply fftshift (axis from −fs/2 to +fs/2).
};

/**
 * @brief Output of computeWaterfall().
 * @ingroup DSP_Waterfall
 */
struct WaterfallResult {
    std::vector<double>              frequencyAxisHz; ///< Frequency axis (length = fftSize).
    std::vector<double>              timeAxisSec;     ///< Centre time of each frame in seconds.
    std::vector<std::vector<double>> powerDb;         ///< [timeIndex][frequencyIndex] in dBFS.
};

// ─────────────────────────────────────────────────────────────────────────────
// computeWaterfall
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute a short-time power spectrum (waterfall / spectrogram) matrix.
 *
 * Each frame is extracted with a hop of
 * `round(fftSize · (1 − overlap))` samples, multiplied by a periodic Hann
 * window, and forward-FFT'd.  The squared magnitude is normalised by the
 * window energy and converted to dBFS.
 *
 * When `params.centered` is `true` the spectrum is FFT-shifted: DC moves to
 * the centre column and the frequency axis runs from −fs/2 to just below +fs/2.
 * When `false` the axis runs from 0 to just below fs.
 *
 * @param iq     Complex IQ samples.  Empty → returns empty WaterfallResult.
 * @param params Waterfall configuration.
 * @return WaterfallResult with:
 *   - `frequencyAxisHz` of length `fftSize`,
 *   - `timeAxisSec` of length (number of complete frames),
 *   - `powerDb[t][f]` matching `timeAxisSec` × `frequencyAxisHz`.
 *
 * @throws std::invalid_argument if `sampleRate ≤ 0`, `fftSize == 0`,
 *         or `overlap` is outside [0, 1).
 *
 * @ingroup DSP_Waterfall
 */
WaterfallResult computeWaterfall(
    const std::vector<std::complex<double>>& iq,
    const WaterfallParams&                   params);

} // namespace SharedMath::DSP

/// @} // DSP_Waterfall
