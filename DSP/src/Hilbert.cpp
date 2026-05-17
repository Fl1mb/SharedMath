/**
 * @file Hilbert.cpp
 * @brief Implementation of Hilbert transform and instantaneous characteristics.
 */

#include "Hilbert.h"
#include "FFTPlan.h"
#include "FFTConfig.h"

#include <cmath>
#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

namespace detail {

constexpr double HILBERT_2PI = 6.28318530717958647692;

/// Standard phase unwrapping: accumulate corrected increments.
std::vector<double> unwrapPhase(const std::vector<double>& phi) {
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
// analyticSignal
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> analyticSignal(const std::vector<double>& x)
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
// hilbert
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> hilbert(const std::vector<double>& x)
{
    auto z = analyticSignal(x);
    std::vector<double> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = z[i].imag();
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// instantaneousAmplitude
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> instantaneousAmplitude(const std::vector<double>& x)
{
    auto z = analyticSignal(x);
    std::vector<double> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) out[i] = std::abs(z[i]);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// instantaneousPhase
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> instantaneousPhase(
    const std::vector<double>& x,
    bool unwrap)
{
    auto z = analyticSignal(x);
    std::vector<double> phi(z.size());
    for (size_t i = 0; i < z.size(); ++i) phi[i] = std::arg(z[i]);
    return unwrap ? detail::unwrapPhase(phi) : phi;
}

// ─────────────────────────────────────────────────────────────────────────────
// instantaneousFrequency
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> instantaneousFrequency(
    const std::vector<double>& x,
    double sampleRate)
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