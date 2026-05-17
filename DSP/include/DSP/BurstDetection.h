#pragma once

/**
 * @file BurstDetection.h
 * @brief Time-domain burst / packet detection for IQ signals (header only).
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

#include <complex>
#include <cstddef>
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
// Public API
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
std::vector<Burst> detectBursts(
    const std::vector<std::complex<double>>& iq,
    const BurstDetectionParams&              params);

} // namespace SharedMath::DSP

/// @} // DSP_BurstDetection