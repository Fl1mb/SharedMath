#pragma once

/**
 * @file SignalEstimation.h
 * @brief Parameter estimation for IQ signals: centre frequency, bandwidth, SNR.
 *
 * @defgroup DSP_SignalEstimation Signal Estimation
 * @ingroup DSP
 * @{
 *
 * All functions operate on `std::vector<std::complex<double>>` IQ data.
 *
 * ### Example
 * @code{.cpp}
 * auto iq = ...;
 * SharedMath::DSP::SignalEstimationParams p;
 * p.sampleRate = 2e6;
 * p.fftSize    = 4096;
 * auto est = SharedMath::DSP::estimateSignal(iq, p);
 * std::cout << "cf=" << est.centerFrequencyHz << " SNR=" << est.snrDb << " dB\n";
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
 * @brief Estimated parameters of a captured IQ signal.
 * @ingroup DSP_SignalEstimation
 */
struct SignalEstimate {
    double centerFrequencyHz   = 0.0; ///< Power-centroid centre frequency in Hz.
    double bandwidthHz         = 0.0; ///< Alias for occupiedBandwidthHz.
    double occupiedBandwidthHz = 0.0; ///< Minimum bandwidth containing `occupiedPowerRatio` of power.
    double snrDb               = 0.0; ///< Estimated SNR in dB (peak spectral bin vs noise floor).
    double noiseFloorDb        = 0.0; ///< Estimated noise floor in dBFS (spectral median).
    double peakPowerDb         = 0.0; ///< Instantaneous peak power in dBFS.
    double averagePowerDb      = 0.0; ///< Mean instantaneous power in dBFS.
    double durationSec         = 0.0; ///< Duration of the IQ block in seconds.
};

/**
 * @brief Configuration for estimateSignal().
 * @ingroup DSP_SignalEstimation
 */
struct SignalEstimationParams {
    double sampleRate         = 1.0;   ///< Sample rate in Hz.  Must be > 0.
    size_t fftSize            = 1024;  ///< FFT size.  Must be > 0.
    double occupiedPowerRatio = 0.99;  ///< Power fraction for occupied-bandwidth (0, 1].
    double thresholdDb        = 10.0;  ///< Threshold above noise floor for signal region.
};

// ─────────────────────────────────────────────────────────────────────────────
// Parameter validation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Validate SignalEstimationParams; throws on invalid values.
 * @param p Parameters to check.
 * @throws std::invalid_argument if any field is out of range.
 * @ingroup DSP_SignalEstimation
 */
void validateEstimationParams(const SignalEstimationParams& p);

// ─────────────────────────────────────────────────────────────────────────────
// estimateCenterFrequencyHz
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Estimate the centre frequency of an IQ signal using the power centroid.
 *
 * Applies a Hann window to the first `fftSize` samples, computes the two-sided
 * power spectrum, and returns Σ(f·P(f)) / Σ(P(f)) over all spectral bins.
 * Falls back to the peak-bin frequency if the total power is negligible.
 *
 * @param iq         Complex IQ samples.  Empty → returns 0.
 * @param sampleRate Sample rate in Hz.  Must be > 0.
 * @param fftSize    FFT length.  Must be > 0.
 * @return Estimated centre frequency in Hz.
 * @throws std::invalid_argument if `sampleRate ≤ 0` or `fftSize == 0`.
 * @ingroup DSP_SignalEstimation
 */
double estimateCenterFrequencyHz(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize = 1024);

// ─────────────────────────────────────────────────────────────────────────────
// estimateOccupiedBandwidthHz
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Estimate the occupied bandwidth containing a given fraction of total power.
 *
 * Bins are sorted by descending power and accumulated until the running sum
 * reaches `occupiedPowerRatio` of the total.  The occupied bandwidth is the
 * span (in Hz) from the lowest to the highest selected bin.
 *
 * @param iq                  Complex IQ samples.  Empty → returns 0.
 * @param sampleRate          Sample rate in Hz.  Must be > 0.
 * @param occupiedPowerRatio  Power fraction in (0, 1].  Default 0.99.
 * @param fftSize             FFT length.  Must be > 0.
 * @return Occupied bandwidth in Hz.
 * @throws std::invalid_argument on invalid arguments.
 * @ingroup DSP_SignalEstimation
 */
double estimateOccupiedBandwidthHz(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    double occupiedPowerRatio = 0.99,
    size_t fftSize            = 1024);

// ─────────────────────────────────────────────────────────────────────────────
// estimateSignal
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute a full set of signal parameters from an IQ block.
 *
 * Combines instantaneous power statistics, spectral centre-frequency and
 * occupied-bandwidth estimates, and a noise-floor / SNR estimate derived from
 * the spectral median.
 *
 * @param iq     Complex IQ samples.  Empty → returns a zeroed estimate.
 * @param params Estimation configuration.
 * @return Filled SignalEstimate.
 * @throws std::invalid_argument if any parameter is out of range.
 * @ingroup DSP_SignalEstimation
 */
SignalEstimate estimateSignal(
    const std::vector<std::complex<double>>& iq,
    const SignalEstimationParams&            params);

} // namespace SharedMath::DSP

/// @} // DSP_SignalEstimation