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

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

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
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// Compute the two-sided (FFT-shifted) power spectrum of a single IQ frame.
/// @returns {frequencyAxisHz, powerLinear}, each of length @p fftSize.
inline std::pair<std::vector<double>, std::vector<double>>
twoSidedSpectrum(const std::vector<std::complex<double>>& iq,
                 double sampleRate, size_t fftSize)
{
    const size_t M     = fftSize;
    const size_t N     = iq.size();
    const double binHz = sampleRate / static_cast<double>(M);

    const size_t winLen = std::min(N, M);
    auto win = windowHann(winLen, /*symmetric=*/false);
    double winSumSq = 0.0;
    for (double w : win) winSumSq += w * w;

    std::vector<std::complex<double>> frame(M, {0.0, 0.0});
    for (size_t i = 0; i < winLen; ++i)
        frame[i] = iq[i] * win[i];

    FFTPlan::create(M, {FFTDirection::Forward, FFTNorm::None}).execute(frame);

    const double scale = 1.0 / std::max(winSumSq, 1e-300);
    std::vector<double> pwr(M), freqs(M);
    for (size_t k = 0; k < M; ++k) {
        pwr[(k + M / 2) % M]   = std::norm(frame[k]) * scale;
        freqs[k] = (static_cast<double>(k) - static_cast<double>(M / 2)) * binHz;
    }
    return {std::move(freqs), std::move(pwr)};
}

inline double medianSE(std::vector<double> v)
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
// Parameter validation
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Validate SignalEstimationParams; throws on invalid values.
 * @param p Parameters to check.
 * @throws std::invalid_argument if any field is out of range.
 * @ingroup DSP_SignalEstimation
 */
inline void validateEstimationParams(const SignalEstimationParams& p)
{
    if (p.sampleRate <= 0.0)
        throw std::invalid_argument("SignalEstimation: sampleRate must be > 0");
    if (p.fftSize == 0)
        throw std::invalid_argument("SignalEstimation: fftSize must be > 0");
    if (p.occupiedPowerRatio <= 0.0 || p.occupiedPowerRatio > 1.0)
        throw std::invalid_argument(
            "SignalEstimation: occupiedPowerRatio must be in (0, 1]");
}

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
inline double estimateCenterFrequencyHz(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize = 1024)
{
    if (sampleRate <= 0.0)
        throw std::invalid_argument(
            "estimateCenterFrequencyHz: sampleRate must be > 0");
    if (fftSize == 0)
        throw std::invalid_argument(
            "estimateCenterFrequencyHz: fftSize must be > 0");
    if (iq.empty()) return 0.0;

    auto [freqs, pwr] = detail::twoSidedSpectrum(iq, sampleRate, fftSize);

    double sumPwr = 0.0, sumFP = 0.0;
    for (size_t k = 0; k < freqs.size(); ++k) {
        sumPwr += pwr[k];
        sumFP  += freqs[k] * pwr[k];
    }
    if (sumPwr < 1e-300) {
        // Fallback: peak bin
        size_t pk = static_cast<size_t>(
            std::max_element(pwr.begin(), pwr.end()) - pwr.begin());
        return freqs[pk];
    }
    return sumFP / sumPwr;
}

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
inline double estimateOccupiedBandwidthHz(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    double occupiedPowerRatio = 0.99,
    size_t fftSize            = 1024)
{
    if (sampleRate <= 0.0)
        throw std::invalid_argument(
            "estimateOccupiedBandwidthHz: sampleRate must be > 0");
    if (fftSize == 0)
        throw std::invalid_argument(
            "estimateOccupiedBandwidthHz: fftSize must be > 0");
    if (occupiedPowerRatio <= 0.0 || occupiedPowerRatio > 1.0)
        throw std::invalid_argument(
            "estimateOccupiedBandwidthHz: occupiedPowerRatio must be in (0, 1]");
    if (iq.empty()) return 0.0;

    auto [freqs, pwr] = detail::twoSidedSpectrum(iq, sampleRate, fftSize);
    const size_t M     = pwr.size();
    const double binHz = sampleRate / static_cast<double>(M);
    const double total = std::accumulate(pwr.begin(), pwr.end(), 0.0);
    if (total < 1e-300) return 0.0;

    // Sort bins by descending power
    std::vector<size_t> idx(M);
    std::iota(idx.begin(), idx.end(), 0u);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return pwr[a] > pwr[b]; });

    const double target = total * occupiedPowerRatio;
    double accum = 0.0;
    size_t minBin = M, maxBin = 0;
    for (size_t i = 0; i < M; ++i) {
        accum += pwr[idx[i]];
        if (idx[i] < minBin) minBin = idx[i];
        if (idx[i] > maxBin) maxBin = idx[i];
        if (accum >= target) break;
    }
    if (minBin > maxBin) return 0.0;
    return (static_cast<double>(maxBin - minBin) + 1.0) * binHz;
}

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
inline SignalEstimate estimateSignal(
    const std::vector<std::complex<double>>& iq,
    const SignalEstimationParams&            params)
{
    validateEstimationParams(params);
    SignalEstimate est;
    if (iq.empty()) {
        est.noiseFloorDb = -std::numeric_limits<double>::infinity();
        return est;
    }

    const double fs = params.sampleRate;
    const size_t N  = iq.size();

    // ── Instantaneous power ───────────────────────────────────────────────────
    double sumPwr = 0.0, maxPwr = 0.0;
    for (const auto& s : iq) {
        const double p = std::norm(s);
        sumPwr += p;
        if (p > maxPwr) maxPwr = p;
    }
    const double avgPwr    = sumPwr / static_cast<double>(N);
    est.averagePowerDb     = 10.0 * std::log10(std::max(avgPwr,   1e-300));
    est.peakPowerDb        = 10.0 * std::log10(std::max(maxPwr,   1e-300));
    est.durationSec        = static_cast<double>(N) / fs;

    // ── Spectral analysis ─────────────────────────────────────────────────────
    auto [freqs, pwr] = detail::twoSidedSpectrum(iq, fs, params.fftSize);

    std::vector<double> pwrDb(pwr.size());
    for (size_t k = 0; k < pwr.size(); ++k)
        pwrDb[k] = 10.0 * std::log10(std::max(pwr[k], 1e-300));

    est.noiseFloorDb = detail::medianSE(pwrDb);

    // Centre frequency: power centroid
    {
        double sP = 0.0, sFP = 0.0;
        for (size_t k = 0; k < freqs.size(); ++k) { sP += pwr[k]; sFP += freqs[k] * pwr[k]; }
        est.centerFrequencyHz = (sP > 1e-300) ? sFP / sP : 0.0;
    }

    // Occupied bandwidth
    est.occupiedBandwidthHz =
        estimateOccupiedBandwidthHz(iq, fs, params.occupiedPowerRatio, params.fftSize);
    est.bandwidthHz = est.occupiedBandwidthHz;

    // SNR: peak spectral bin vs noise floor
    {
        const double peak = *std::max_element(pwrDb.begin(), pwrDb.end());
        est.snrDb = peak - est.noiseFloorDb;
    }

    return est;
}

} // namespace SharedMath::DSP

/// @} // DSP_SignalEstimation
