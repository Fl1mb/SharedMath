/**
 * @file Waterfall.cpp
 * @brief Implementation of short-time power spectrum (waterfall) computation.
 */

#include "Waterfall.h"
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
// computeWaterfall
// ─────────────────────────────────────────────────────────────────────────────
WaterfallResult computeWaterfall(
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