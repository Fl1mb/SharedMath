#pragma once

/// SharedMath::DSP — Hilbert transform and instantaneous signal characteristics
///
/// analyticSignal()          — complex analytic signal  z = x + j·H{x}
/// hilbert()                 — Hilbert transform of a real signal
/// instantaneousAmplitude()  — envelope: |z[n]|
/// instantaneousPhase()      — arg(z[n]), optionally unwrapped
/// instantaneousFrequency()  — dφ/dt / (2π), in Hz

#include <complex>
#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// analyticSignal — compute the analytic signal via FFT
//
// Algorithm:
//   1. X = FFT(x)
//   2. Zero negative-frequency bins; double positive-frequency bins.
//      DC (k=0) and Nyquist (k=N/2 for even N) are left unchanged.
//   3. z = IFFT(modified X)
//
// z[n].real() == x[n];  z[n].imag() == H{x}[n]  (Hilbert transform of x).
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> analyticSignal(
    const std::vector<double>& x);

/// ─────────────────────────────────────────────────────────────────────────────
/// hilbert — imaginary part of the analytic signal (= quadrature component)
/// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> hilbert(const std::vector<double>& x);

/// ─────────────────────────────────────────────────────────────────────────────
/// instantaneousAmplitude — signal envelope: |z[n]| = sqrt(x²+H{x}²)
/// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> instantaneousAmplitude(const std::vector<double>& x);

// ─────────────────────────────────────────────────────────────────────────────
// instantaneousPhase — arg(z[n]) in radians
//
// unwrap = false  →  wrapped output ∈ (−π, π]
// unwrap = true   →  continuous (unwrapped) phase
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> instantaneousPhase(
    const std::vector<double>& x,
    bool unwrap = false);

// ─────────────────────────────────────────────────────────────────────────────
// instantaneousFrequency — dφ/dt / (2π), in Hz
//
// Computed as the first finite difference of the unwrapped phase.
// Output length is N−1.  Multiply by sampleRate to get Hz; pass
// sampleRate = 1.0 (default) to get normalised frequency ∈ [0, 0.5].
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> instantaneousFrequency(
    const std::vector<double>& x,
    double sampleRate = 1.0);

} // namespace SharedMath::DSP