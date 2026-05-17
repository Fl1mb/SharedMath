/**
 * @file FilterResponse.cpp
 * @brief Implementation of FIR frequency response analysis functions.
 */

#include "DSP/FilterResponse.h"
#include "DSP/FFT.h"

#include <cmath>
#include <complex>
#include <vector>
#include <cstddef>
#include <algorithm>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// frequencyResponseFIR
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> frequencyResponseFIR(
    const std::vector<double>& h,
    size_t nfft,
    double /*sampleRate*/)
{
    if (h.empty()) return {};

    size_t n = 1;
    while (n < nfft || n < h.size()) n <<= 1;

    std::vector<std::complex<double>> H(n, {0.0, 0.0});
    for (size_t i = 0; i < h.size(); ++i) H[i] = h[i];

    FFTPlan::create(n, {FFTDirection::Forward, FFTNorm::None}).execute(H);
    H.resize(n / 2 + 1);
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// firResponseFrequencies
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> firResponseFrequencies(
    const std::vector<double>& h,
    size_t nfft,
    double sampleRate)
{
    size_t n = 1;
    while (n < nfft || n < h.size()) n <<= 1;
    return rfftFrequencies(n, sampleRate);
}

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIR
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> magnitudeResponseFIR(
    const std::vector<double>& h,
    size_t nfft,
    double sampleRate)
{
    return magnitude(frequencyResponseFIR(h, nfft, sampleRate));
}

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIRDB
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> magnitudeResponseFIRDB(
    const std::vector<double>& h,
    size_t nfft,
    double sampleRate,
    double refMag)
{
    return magnitudeDB(frequencyResponseFIR(h, nfft, sampleRate), refMag);
}

// ─────────────────────────────────────────────────────────────────────────────
// phaseResponseFIR
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> phaseResponseFIR(
    const std::vector<double>& h,
    size_t nfft,
    double sampleRate)
{
    return phase(frequencyResponseFIR(h, nfft, sampleRate));
}

// ─────────────────────────────────────────────────────────────────────────────
// groupDelayFIR
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> groupDelayFIR(
    const std::vector<double>& h,
    size_t nfft,
    double /*sampleRate*/)
{
    if (h.empty()) return {};

    size_t n = 1;
    while (n < nfft || n < h.size()) n <<= 1;

    std::vector<std::complex<double>> H(n, {0.0, 0.0});
    std::vector<std::complex<double>> Z(n, {0.0, 0.0});  // DFT of {i·h[i]}
    for (size_t i = 0; i < h.size(); ++i) {
        H[i] = h[i];
        Z[i] = static_cast<double>(i) * h[i];
    }

    auto plan = FFTPlan::create(n, {FFTDirection::Forward, FFTNorm::None});
    plan.execute(H);
    plan.execute(Z);

    size_t m = n / 2 + 1;
    std::vector<double> gd(m);
    for (size_t k = 0; k < m; ++k) {
        double denom = std::norm(H[k]);
        gd[k] = (denom < 1e-30) ? 0.0
                                 : std::real(Z[k] * std::conj(H[k])) / denom;
    }
    return gd;
}

} // namespace SharedMath::DSP
