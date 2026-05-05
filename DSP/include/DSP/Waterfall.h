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
inline WaterfallResult computeWaterfall(
    const std::vector<std::complex<double>>& iq,
    const WaterfallParams&                   params)
{
    if (params.sampleRate <= 0.0)
        throw std::invalid_argument("computeWaterfall: sampleRate must be > 0");
    if (params.fftSize == 0)
        throw std::invalid_argument("computeWaterfall: fftSize must be > 0");
    if (params.overlap < 0.0 || params.overlap >= 1.0)
        throw std::invalid_argument("computeWaterfall: overlap must be in [0, 1)");

    WaterfallResult result;
    if (iq.empty()) return result;

    const size_t M    = params.fftSize;
    const size_t N    = iq.size();
    const double fs   = params.sampleRate;
    const size_t step = std::max<size_t>(1,
        static_cast<size_t>(std::round(static_cast<double>(M) * (1.0 - params.overlap))));

    // ── Hann window (periodic) ────────────────────────────────────────────────
    const auto win = windowHann(M, /*symmetric=*/false);
    double winSumSq = 0.0;
    for (double w : win) winSumSq += w * w;
    const double scale = 1.0 / std::max(winSumSq, 1e-300);

    // ── Frequency axis ────────────────────────────────────────────────────────
    const double binHz = fs / static_cast<double>(M);
    result.frequencyAxisHz.resize(M);
    if (params.centered) {
        for (size_t k = 0; k < M; ++k)
            result.frequencyAxisHz[k] =
                (static_cast<double>(k) - static_cast<double>(M / 2)) * binHz;
    } else {
        for (size_t k = 0; k < M; ++k)
            result.frequencyAxisHz[k] = static_cast<double>(k) * binHz;
    }

    // ── Process each frame ────────────────────────────────────────────────────
    const size_t numFrames = (N >= M) ? (N - M) / step + 1 : 0;
    result.powerDb.reserve(numFrames);
    result.timeAxisSec.reserve(numFrames);

    for (size_t s = 0; s + M <= N; s += step) {
        std::vector<std::complex<double>> frame(M);
        for (size_t i = 0; i < M; ++i)
            frame[i] = iq[s + i] * win[i];

        FFTPlan::create(M, {FFTDirection::Forward, FFTNorm::None}).execute(frame);

        std::vector<double> row(M);
        if (params.centered) {
            for (size_t k = 0; k < M; ++k)
                row[(k + M / 2) % M] =
                    10.0 * std::log10(std::max(std::norm(frame[k]) * scale, 1e-300));
        } else {
            for (size_t k = 0; k < M; ++k)
                row[k] = 10.0 * std::log10(std::max(std::norm(frame[k]) * scale, 1e-300));
        }
        result.powerDb.push_back(std::move(row));

        // Centre time of this frame
        result.timeAxisSec.push_back(static_cast<double>(s + M / 2) / fs);
    }

    return result;
}

} // namespace SharedMath::DSP

/// @} // DSP_Waterfall
