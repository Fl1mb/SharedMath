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

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

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
inline double averagePowerDb(const std::vector<std::complex<double>>& iq)
{
    if (iq.empty()) return -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (const auto& s : iq) sum += std::norm(s);
    sum /= static_cast<double>(iq.size());
    return 10.0 * std::log10(std::max(sum, 1e-300));
}

/**
 * @brief Compute the peak instantaneous power in dBFS.
 *
 * Returns `10·log10(max(|x[n]|²))`.
 *
 * @param iq Input IQ samples.  Empty → returns −∞.
 * @return Peak power in dBFS.
 * @ingroup DSP_SignalMetrics
 */
inline double peakPowerDb(const std::vector<std::complex<double>>& iq)
{
    if (iq.empty()) return -std::numeric_limits<double>::infinity();
    double mx = 0.0;
    for (const auto& s : iq) {
        const double p = std::norm(s);
        if (p > mx) mx = p;
    }
    return 10.0 * std::log10(std::max(mx, 1e-300));
}

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
inline double paprDb(const std::vector<std::complex<double>>& iq)
{
    if (iq.empty()) return 0.0;
    return peakPowerDb(iq) - averagePowerDb(iq);
}

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
inline double evmRmsPercent(
    const std::vector<std::complex<double>>& measured,
    const std::vector<std::complex<double>>& reference)
{
    if (measured.empty() || reference.empty())
        throw std::invalid_argument("evmRmsPercent: inputs must not be empty");
    if (measured.size() != reference.size())
        throw std::invalid_argument(
            "evmRmsPercent: measured and reference must have the same length");

    const size_t N = measured.size();
    double errPwr = 0.0, refPwr = 0.0;
    for (size_t i = 0; i < N; ++i) {
        errPwr += std::norm(measured[i] - reference[i]);
        refPwr += std::norm(reference[i]);
    }
    if (refPwr < 1e-300) return 0.0;
    return 100.0 * std::sqrt(errPwr / refPwr);
}

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
inline double evmRmsDb(
    const std::vector<std::complex<double>>& measured,
    const std::vector<std::complex<double>>& reference)
{
    const double pct = evmRmsPercent(measured, reference);
    if (pct < 1e-15) return -std::numeric_limits<double>::infinity();
    return 20.0 * std::log10(pct / 100.0);
}

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
inline double estimateSnrDb(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize = 1024)
{
    if (sampleRate <= 0.0)
        throw std::invalid_argument("estimateSnrDb: sampleRate must be > 0");
    if (fftSize == 0)
        throw std::invalid_argument("estimateSnrDb: fftSize must be > 0");
    if (iq.empty()) return 0.0;

    const size_t M      = fftSize;
    const size_t N      = iq.size();
    const size_t winLen = std::min(N, M);

    auto win = windowHann(winLen, /*symmetric=*/false);
    double winSumSq = 0.0;
    for (double w : win) winSumSq += w * w;

    std::vector<std::complex<double>> frame(M, {0.0, 0.0});
    for (size_t i = 0; i < winLen; ++i)
        frame[i] = iq[i] * win[i];

    FFTPlan::create(M, {FFTDirection::Forward, FFTNorm::None}).execute(frame);

    const double scale = 1.0 / std::max(winSumSq, 1e-300);
    std::vector<double> pwrDb(M);
    for (size_t k = 0; k < M; ++k)
        pwrDb[k] = 10.0 * std::log10(std::max(std::norm(frame[k]) * scale, 1e-300));

    // Noise floor via median
    std::vector<double> sorted = pwrDb;
    std::nth_element(sorted.begin(), sorted.begin() + M / 2, sorted.end());
    const double noiseFloor = sorted[M / 2];

    const double peak = *std::max_element(pwrDb.begin(), pwrDb.end());
    return peak - noiseFloor;
}

} // namespace SharedMath::DSP

/// @} // DSP_SignalMetrics
