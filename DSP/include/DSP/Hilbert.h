#pragma once

// SharedMath::DSP — Hilbert transform and instantaneous signal characteristics
//
// analyticSignal()          — complex analytic signal  z = x + j·H{x}
// hilbert()                 — Hilbert transform of a real signal
// instantaneousAmplitude()  — envelope: |z[n]|
// instantaneousPhase()      — arg(z[n]), optionally unwrapped
// instantaneousFrequency()  — dφ/dt / (2π), in Hz

#include "FFTPlan.h"
#include "FFTConfig.h"

#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>

namespace SharedMath::DSP {

namespace detail {

constexpr double HILBERT_2PI = 6.28318530717958647692;

// Standard phase unwrapping: accumulate corrected increments.
inline std::vector<double> unwrapPhase(const std::vector<double>& phi) {
    if (phi.empty()) return {};
    std::vector<double> out(phi.size());
    out[0] = phi[0];
    for (size_t i = 1; i < phi.size(); ++i) {
        double d = phi[i] - phi[i - 1];
        d -= HILBERT_2PI * std::round(d / HILBERT_2PI);
        out[i] = out[i - 1] + d;
    }
    return out;
}

} // namespace detail


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
inline std::vector<std::complex<double>> analyticSignal(
    const std::vector<double>& x)
{
    if (x.empty()) return {};
    size_t N = x.size();

    std::vector<std::complex<double>> X(N);
    for (size_t i = 0; i < N; ++i) X[i] = x[i];

    FFTPlan::create(N).execute(X);

    if (N > 1) {
        if (N % 2 == 0) {
            // Even N: DC at 0, Nyquist at N/2 — keep both; double 1..N/2-1; zero N/2+1..N-1
            for (size_t k = 1;         k < N / 2;      ++k) X[k] *= 2.0;
            for (size_t k = N / 2 + 1; k < N;          ++k) X[k]  = 0.0;
        } else {
            // Odd N: DC at 0 — keep; double 1..(N-1)/2; zero (N+1)/2..N-1
            for (size_t k = 1;           k <= (N - 1) / 2; ++k) X[k] *= 2.0;
            for (size_t k = (N + 1) / 2; k < N;            ++k) X[k]  = 0.0;
        }
    }

    FFTPlan::create(N, {FFTDirection::Inverse, FFTNorm::ByN}).execute(X);
    return X;
}


// ─────────────────────────────────────────────────────────────────────────────
// hilbert — imaginary part of the analytic signal (= quadrature component)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> hilbert(const std::vector<double>& x) {
    auto z = analyticSignal(x);
    std::vector<double> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = z[i].imag();
    return out;
}


// ─────────────────────────────────────────────────────────────────────────────
// instantaneousAmplitude — signal envelope: |z[n]| = sqrt(x²+H{x}²)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> instantaneousAmplitude(const std::vector<double>& x) {
    auto z = analyticSignal(x);
    std::vector<double> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = std::abs(z[i]);
    return out;
}


// ─────────────────────────────────────────────────────────────────────────────
// instantaneousPhase — arg(z[n]) in radians
//
// unwrap = false  →  wrapped output ∈ (−π, π]
// unwrap = true   →  continuous (unwrapped) phase
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> instantaneousPhase(
    const std::vector<double>& x,
    bool unwrap = false)
{
    auto z = analyticSignal(x);
    std::vector<double> phi(z.size());
    for (size_t i = 0; i < z.size(); ++i) phi[i] = std::arg(z[i]);
    return unwrap ? detail::unwrapPhase(phi) : phi;
}


// ─────────────────────────────────────────────────────────────────────────────
// instantaneousFrequency — dφ/dt / (2π), in Hz
//
// Computed as the first finite difference of the unwrapped phase.
// Output length is N−1.  Multiply by sampleRate to get Hz; pass
// sampleRate = 1.0 (default) to get normalised frequency ∈ [0, 0.5].
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> instantaneousFrequency(
    const std::vector<double>& x,
    double sampleRate = 1.0)
{
    if (x.size() < 2) return {};

    auto phi = instantaneousPhase(x, true);
    size_t N = phi.size();
    std::vector<double> freq(N - 1);
    for (size_t i = 0; i < N - 1; ++i)
        freq[i] = (phi[i + 1] - phi[i]) * sampleRate / detail::HILBERT_2PI;
    return freq;
}

} // namespace SharedMath::DSP
