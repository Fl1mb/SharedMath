/**
 * @file SignalMetrics.cpp
 * @brief Implementation of signal quality and power metrics.
 */

#include "SignalMetrics.h"
#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Power measurements
// ─────────────────────────────────────────────────────────────────────────────
double averagePowerDb(const std::vector<std::complex<double>>& iq)
{
    if (iq.empty()) return -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (const auto& s : iq) sum += std::norm(s);
    sum /= static_cast<double>(iq.size());
    return 10.0 * std::log10(std::max(sum, 1e-300));
}

double peakPowerDb(const std::vector<std::complex<double>>& iq)
{
    if (iq.empty()) return -std::numeric_limits<double>::infinity();
    double mx = 0.0;
    for (const auto& s : iq) {
        const double p = std::norm(s);
        if (p > mx) mx = p;
    }
    return 10.0 * std::log10(std::max(mx, 1e-300));
}

double paprDb(const std::vector<std::complex<double>>& iq)
{
    if (iq.empty()) return 0.0;
    return peakPowerDb(iq) - averagePowerDb(iq);
}

// ─────────────────────────────────────────────────────────────────────────────
// Error Vector Magnitude (EVM)
// ─────────────────────────────────────────────────────────────────────────────
double evmRmsPercent(
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

double evmRmsDb(
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
double estimateSnrDb(
    const std::vector<std::complex<double>>& iq,
    double sampleRate,
    size_t fftSize)
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