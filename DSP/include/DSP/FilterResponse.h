#pragma once

/// SharedMath::DSP — FIR frequency response analysis
///
/// frequencyResponseFIR    — complex H(f) at nfft/2+1 one-sided bins
/// firResponseFrequencies  — matching frequency axis in Hz
/// magnitudeResponseFIR    — |H(f)|
/// magnitudeResponseFIRDB  — 20·log10(|H(f)|)
/// phaseResponseFIR        — arg(H(f)) in radians
/// groupDelayFIR           — group delay in samples

#include <vector>
#include <complex>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// frequencyResponseFIR
//
// Computes H(f) by zero-padding h to the next power-of-2 >= max(nfft, h.size())
// and performing an FFT.  Returns the one-sided half-spectrum: n/2+1 bins.
// Use firResponseFrequencies() to get the matching frequency axis.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> frequencyResponseFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0);   // kept for API symmetry

// Frequency axis (Hz) matching frequencyResponseFIR(h, nfft, sampleRate).
std::vector<double> firResponseFrequencies(
    const std::vector<double>& h,
    size_t nfft,
    double sampleRate);

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIR — |H(f)|
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> magnitudeResponseFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// magnitudeResponseFIRDB — 20·log10(|H(f)| / refMag)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> magnitudeResponseFIRDB(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0,
    double refMag = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// phaseResponseFIR — arg(H(f)) in radians ∈ (−π, π]
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> phaseResponseFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// groupDelayFIR — group delay in samples
//
// Uses the standard formula:  τ(ω) = Re{ Z(ω) / H(ω) }
// where Z is the DTFT of { n·h[n] }.
//
// At zeros of H(ω) (where |H|² < 1e-30) the value is set to 0.
// For a symmetric linear-phase FIR of order M, τ ≈ M/2 samples in the passband.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> groupDelayFIR(
    const std::vector<double>& h,
    size_t nfft = 512,
    double sampleRate = 1.0);   // kept for API symmetry

} // namespace SharedMath::DSP