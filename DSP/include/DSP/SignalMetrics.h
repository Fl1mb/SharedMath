#pragma once

/**
 * @file SignalMetrics.h
 * @brief Signal quality and power metrics for IQ signals.
 *
 * @defgroup DSP_SignalMetrics Signal Metrics
 * @ingroup DSP
 * @{
 *
 * Scalar quality measurements: average power, peak power, PAPR, EVM, and SNR.
 * All functions operate on `std::vector<std::complex<double>>` IQ data.
 *
 * @}
 */

#include <complex>
#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Power measurements
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute the mean instantaneous power in dBFS.
 *
 * Returns `10·log10(mean(|x[n]|²))`.
 *
 * @param iq Input IQ samples.  Empty → returns −∞.
 * @return Average power in dBFS.
 * @ingroup DSP_SignalMetrics
 */
double averagePowerDb(const std::vector<std::complex<double>>& iq);

/**
 * @brief Compute the peak instantaneous power in dBFS.
 *
 * Returns `10·log10(max(|x[n]|²))`.
 *
 * @param iq Input IQ samples.  Empty → returns −∞.
 * @return Peak power in dBFS.
 * @ingroup DSP_SignalMetrics
 */
double peakPowerDb(const std::vector<std::complex<double>>& iq);

/**
 * @brief Compute Peak-to-Average Power Ratio (PAPR) in dB.
 *
 * `paprDb = peakPowerDb(iq) − averagePowerDb(iq)`.
 * For a constant-amplitude signal PAPR ≈ 0 dB.
 *
 * @param iq Input IQ samples.  Empty → returns 0.
 * @return PAPR in dB.
 * @ingroup DSP_SignalMetrics
 */
double paprDb(const std::vector<std::complex<double>>& iq);

// ─────────────────────────────────────────────────────────────────────────────
// Error Vector Magnitude (EVM)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Compute the RMS Error Vector Magnitude as a percentage.
 *
 * @f[
 *   \text{EVM}_{\%} = 100 \cdot \sqrt{
 *       \frac{\sum_n |e[n]|^2}{\sum_n |r[n]|^2}
 *   }, \quad e[n] = \text{measured}[n] - \text{reference}[n]
 * @f]
 *
 * Returns 0 if the reference energy is negligible.
 *
 * @param measured   Received IQ symbols.  Must not be empty.
 * @param reference  Ideal / reference IQ symbols (same length as `measured`).
 * @return EVM in percent in [0, ∞).
 * @throws std::invalid_argument if either vector is empty or lengths differ.
 * @ingroup DSP_SignalMetrics
 */
double evmRmsPercent(
    const std::vector<std::complex<double>>& measured,
    const std::vector<std::complex<double>>& reference);

/**
 * @brief Compute the RMS Error Vector Magnitude in dB.
 *
 * `evmRmsDb = 20·log10(evmRmsPercent / 100)`.
 * A perfect match returns −∞.
 *
 * @param measured   Received IQ symbols.  Must not be empty.
 * @param reference  Ideal IQ symbols (same length).
 * @return EVM in dB (≤ 0 for a reasonable link budget).
 * @throws std::invalid_argument if either vector is empty or lengths differ.
 * @ingroup DSP_SignalMetrics
 */
double evmRmsDb(
    const std::vector<std::complex<double>>& measured,
    const std::vector<std::complex<double>>& reference);

// ─────────────────────────────────────────────────────────────────────────────
// SNR estimation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Estimate the signal-to-noise ratio from an IQ block's power spectrum.
 *
 * Applies a Hann window to the first `fftSize` samples and computes the power
 * spectrum.  The noise floor is estimated as the spectral median and the signal
 * power as the peak bin.  Returns `peak_dBFS − median_dBFS`.
 *
 * @param iq         Input IQ.  Empty → returns 0.
 * @param sampleRate Sample rate in Hz.  Must be > 0.
 * @param fftSize    FFT length.  Must be > 0.
 * @return Estimated SNR in dB.
 * @throws std::invalid_argument if `sampleRate ≤ 0` or `fftSize == 0`.
 * @ingroup DSP_SignalMetrics
 */
double estimateSnrDb(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize = 1024);

} // namespace SharedMath::DSP

/// @} // DSP_SignalMetrics