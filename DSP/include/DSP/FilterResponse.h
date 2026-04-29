#pragma once

// SharedMath::DSP — FIR frequency response analysis
//
// frequencyResponseFIR    — complex H(f) at nfft/2+1 one-sided bins
// firResponseFrequencies  — matching frequency axis in Hz
// magnitudeResponseFIR    — |H(f)|
// magnitudeResponseFIRDB  — 20·log10(|H(f)|)
// phaseResponseFIR        — arg(H(f)) in radians
// groupDelayFIR           — group delay in samples

#include "FFT.h"   // magnitude, phase, magnitudeDB, rfftFrequencies, FFTPlan

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// frequencyResponseFIR
//
// Computes H(f) by zero-padding h to the next power-of-2 >= max(nfft, h.size())
// and performing an FFT.  Returns the one-sided half-spectrum: n/2+1 bins.
// Use firResponseFrequencies() to get the matching frequency axis.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<std::complex<double>> frequencyResponseFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double /*sampleRate*/ = 1.0)   // not used in FFT; kept for API symmetry
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

// Frequency axis (Hz) matching frequencyResponseFIR(h, nfft, sampleRate).
inline std::vector<double> firResponseFrequencies(
    const std::vector<double>& h,
    size_t nfft,
    double sampleRate)
{
    size_t n = 1;
    while (n < nfft || n < h.size()) n <<= 1;
    return rfftFrequencies(n, sampleRate);
}

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIR — |H(f)|
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> magnitudeResponseFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0)
{
    return magnitude(frequencyResponseFIR(h, nfft, sampleRate));
}

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIRDB — 20·log10(|H(f)| / refMag)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> magnitudeResponseFIRDB(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0,
    double refMag = 1.0)
{
    return magnitudeDB(frequencyResponseFIR(h, nfft, sampleRate), refMag);
}

// ─────────────────────────────────────────────────────────────────────────────
// phaseResponseFIR — arg(H(f)) in radians ∈ (−π, π]
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> phaseResponseFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0)
{
    return phase(frequencyResponseFIR(h, nfft, sampleRate));
}

// ─────────────────────────────────────────────────────────────────────────────
// groupDelayFIR — group delay in samples
//
// Uses the standard formula:  τ(ω) = Re{ Z(ω) / H(ω) }
// where Z is the DTFT of { n·h[n] }.
//
// At zeros of H(ω) (where |H|² < 1e-30) the value is set to 0.
// For a symmetric linear-phase FIR of order M, τ ≈ M/2 samples in the passband.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> groupDelayFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double /*sampleRate*/ = 1.0)   // not used; kept for API symmetry
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
