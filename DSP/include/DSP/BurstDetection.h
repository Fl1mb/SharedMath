#pragma once

/**
 * @file BurstDetection.h
 * @brief Time-domain burst / packet detection for IQ signals.
 *
 * @defgroup DSP_BurstDetection Burst Detection
 * @ingroup DSP
 * @{
 *
 * detectBursts() slides a power-estimation window over the IQ stream and
 * returns a list of on-air transmission events, with optional gap merging and
 * minimum-duration filtering.
 *
 * ### Example
 * @code{.cpp}
 * SharedMath::DSP::BurstDetectionParams p;
 * p.sampleRate    = 2e6;
 * p.windowSize    = 512;
 * p.thresholdDb   = 12.0;
 * p.maxGapSec     = 50e-6;   // merge gaps < 50 µs
 * p.minDurationSec = 1e-4;   // discard bursts shorter than 100 µs
 *
 * auto bursts = SharedMath::DSP::detectBursts(iq, p);
 * @endcode
 *
 * @}
 */

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
 * @brief Configuration for detectBursts().
 * @ingroup DSP_BurstDetection
 */
struct BurstDetectionParams {
    double sampleRate     = 1.0;  ///< Sample rate in Hz.  Must be > 0.
    size_t windowSize     = 256;  ///< Power-estimation window length.  Must be > 0.
    double overlap        = 0.5;  ///< Window overlap in [0, 1).
    double thresholdDb    = 10.0; ///< Detection threshold above noise floor in dB.
    double minDurationSec = 0.0;  ///< Discard bursts shorter than this (seconds).
    double maxGapSec      = 0.0;  ///< Merge adjacent bursts separated by less than this (seconds; 0 = no merging).
};

/**
 * @brief Describes a single detected burst event.
 * @ingroup DSP_BurstDetection
 */
struct Burst {
    size_t startSample    = 0;  ///< First sample index.
    size_t endSample      = 0;  ///< Last sample index (inclusive).
    double startTimeSec   = 0.0; ///< Start time in seconds.
    double endTimeSec     = 0.0; ///< End time in seconds.
    double durationSec    = 0.0; ///< Duration in seconds.
    double peakPowerDb    = 0.0; ///< Peak per-window power in dBFS.
    double averagePowerDb = 0.0; ///< Average per-window power in dBFS.
    double snrDb          = 0.0; ///< Estimated SNR (peak power minus noise floor) in dB.
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

inline double medianBD(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 1) return v[n / 2];
    const double hi = v[n / 2];
    std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
    return 0.5 * (v[n / 2 - 1] + hi);
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// detectBursts
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Detect on/off-keyed bursts in an IQ stream.
 *
 * The algorithm:
 *  -# Divide IQ into overlapping windows of length `params.windowSize`.
 *  -# Compute mean instantaneous power (in dBFS) for each window.
 *  -# Estimate the noise floor as the median of all per-window powers.
 *  -# Merge consecutive above-threshold windows into raw burst records.
 *  -# If `maxGapSec > 0`, merge adjacent bursts separated by ≤ maxGapSec.
 *  -# Discard bursts with `durationSec < minDurationSec`.
 *
 * @param iq     Complex IQ samples.  Empty → returns empty vector.
 * @param params Burst detection configuration.
 * @return Vector of Burst records, ordered by `startSample`.
 *
 * @throws std::invalid_argument if `sampleRate ≤ 0`, `windowSize == 0`,
 *         or `overlap` is outside [0, 1).
 *
 * @ingroup DSP_BurstDetection
 */
inline std::vector<Burst> detectBursts(
    const std::vector<std::complex<double>>& iq,
    const BurstDetectionParams&              params)
{
    if (params.sampleRate <= 0.0)
        throw std::invalid_argument("detectBursts: sampleRate must be > 0");
    if (params.windowSize == 0)
        throw std::invalid_argument("detectBursts: windowSize must be > 0");
    if (params.overlap < 0.0 || params.overlap >= 1.0)
        throw std::invalid_argument("detectBursts: overlap must be in [0, 1)");

    if (iq.empty()) return {};

    const size_t N    = iq.size();
    const size_t wLen = params.windowSize;
    const size_t step = std::max<size_t>(1,
        static_cast<size_t>(std::round(static_cast<double>(wLen) * (1.0 - params.overlap))));
    const double fs   = params.sampleRate;

    // ── Per-window power in dBFS ──────────────────────────────────────────────
    std::vector<double> winPower;
    std::vector<size_t> winStart;
    winPower.reserve((N + step - 1) / step);
    winStart.reserve(winPower.capacity());

    for (size_t s = 0; s + wLen <= N; s += step) {
        double p = 0.0;
        for (size_t i = 0; i < wLen; ++i) p += std::norm(iq[s + i]);
        p /= static_cast<double>(wLen);
        winPower.push_back(10.0 * std::log10(std::max(p, 1e-300)));
        winStart.push_back(s);
    }
    if (winPower.empty()) return {};

    const double noiseFloor = detail::medianBD(winPower);
    const double threshold  = noiseFloor + params.thresholdDb;

    // ── Merge consecutive above-threshold windows into raw bursts ─────────────
    std::vector<Burst> raw;
    bool   inBurst = false;
    double peak    = 0.0;
    double sumPwr  = 0.0;
    size_t count   = 0;
    size_t bStart  = 0, bEnd = 0;

    auto pushBurst = [&]() {
        Burst b;
        b.startSample    = bStart;
        b.endSample      = bEnd;
        b.startTimeSec   = static_cast<double>(bStart) / fs;
        b.endTimeSec     = static_cast<double>(bEnd)   / fs;
        b.durationSec    = b.endTimeSec - b.startTimeSec;
        b.peakPowerDb    = peak;
        b.averagePowerDb = (count > 0) ? sumPwr / static_cast<double>(count) : peak;
        b.snrDb          = peak - noiseFloor;
        raw.push_back(b);
    };

    for (size_t wi = 0; wi < winPower.size(); ++wi) {
        const bool   above = (winPower[wi] >= threshold);
        const size_t wEnd  = winStart[wi] + wLen - 1;
        if (above) {
            if (!inBurst) {
                inBurst = true;
                bStart  = winStart[wi];
                peak    = winPower[wi];
                sumPwr  = 0.0;
                count   = 0;
            }
            bEnd = wEnd;
            if (winPower[wi] > peak) peak = winPower[wi];
            sumPwr += winPower[wi];
            ++count;
        } else {
            if (inBurst) { pushBurst(); inBurst = false; }
        }
    }
    if (inBurst) pushBurst();

    // ── Gap merging ───────────────────────────────────────────────────────────
    const size_t maxGapSamples =
        (params.maxGapSec > 0.0)
        ? static_cast<size_t>(std::ceil(params.maxGapSec * fs))
        : 0;

    std::vector<Burst> merged;
    for (auto& b : raw) {
        if (merged.empty()) {
            merged.push_back(b);
        } else {
            Burst& last = merged.back();
            const size_t gap =
                (b.startSample > last.endSample) ? b.startSample - last.endSample : 0;
            if (maxGapSamples > 0 && gap <= maxGapSamples) {
                last.endSample    = b.endSample;
                last.endTimeSec   = b.endTimeSec;
                last.durationSec  = last.endTimeSec - last.startTimeSec;
                last.peakPowerDb  = std::max(last.peakPowerDb, b.peakPowerDb);
                last.averagePowerDb =
                    0.5 * (last.averagePowerDb + b.averagePowerDb);  // approximate
                last.snrDb = last.peakPowerDb - noiseFloor;
            } else {
                merged.push_back(b);
            }
        }
    }

    // ── Filter by minimum duration ────────────────────────────────────────────
    std::vector<Burst> result;
    for (const auto& b : merged)
        if (b.durationSec >= params.minDurationSec)
            result.push_back(b);

    return result;
}

} // namespace SharedMath::DSP

/// @} // DSP_BurstDetection
