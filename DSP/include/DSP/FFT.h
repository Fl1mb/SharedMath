#pragma once

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <functional>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// High-level FFT convenience API
//
// These free functions create a temporary FFTPlan, execute it, and return.
// For repeated transforms of the same size, create an FFTPlan directly and
// reuse it — that avoids re-allocating twiddle-factor tables.
// ─────────────────────────────────────────────────────────────────────────────

// ── In-place forward FFT ─────────────────────────────────────────────────────
// Transforms x into its DFT in-place.  Default: no normalization.
inline void fft(std::vector<std::complex<double>>& x,
                FFTNorm norm = FFTNorm::None)
{
    if (x.empty()) return;
    FFTPlan::create(x.size(), {FFTDirection::Forward, norm}).execute(x);
}

// ── In-place inverse FFT ─────────────────────────────────────────────────────
// Transforms X into its IDFT in-place.  Default: 1/N normalization.
inline void ifft(std::vector<std::complex<double>>& X,
                 FFTNorm norm = FFTNorm::ByN)
{
    if (X.empty()) return;
    FFTPlan::create(X.size(), {FFTDirection::Inverse, norm}).execute(X);
}

// ── Real FFT (half-spectrum) ─────────────────────────────────────────────────
// Input: N real samples.
// Output: N/2+1 complex DFT bins (non-redundant half due to Hermitian symmetry).
//
// To get the full spectrum, mirror: X[N-k] = conj(X[k]) for k=1..N/2-1.
inline std::vector<std::complex<double>> rfft(const std::vector<double>& x,
                                               FFTNorm norm = FFTNorm::None)
{
    if (x.empty()) return {};
    size_t n = x.size();
    std::vector<std::complex<double>> cx(n);
    for (size_t i = 0; i < n; ++i) cx[i] = x[i];

    FFTPlan::create(n, {FFTDirection::Forward, norm}).execute(cx);

    cx.resize(n / 2 + 1);
    return cx;
}

// ── Inverse real FFT ─────────────────────────────────────────────────────────
// Input:  N/2+1 complex bins (from rfft).
// Output: N real samples.
// n must match the original transform length.
inline std::vector<double> irfft(const std::vector<std::complex<double>>& X,
                                  size_t n,
                                  FFTNorm norm = FFTNorm::ByN)
{
    if (X.empty() || n == 0) return {};

    // Reconstruct full Hermitian spectrum
    std::vector<std::complex<double>> full(n, {0.0, 0.0});
    size_t half = n / 2 + 1;
    for (size_t k = 0; k < half && k < X.size(); ++k)
        full[k] = X[k];
    for (size_t k = 1; k + 1 < half; ++k)
        full[n - k] = std::conj(X[k]);

    FFTPlan::create(n, {FFTDirection::Inverse, norm}).execute(full);

    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = full[i].real();
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral analysis helpers
// ─────────────────────────────────────────────────────────────────────────────

// Magnitude spectrum: |X[k]|
inline std::vector<double> magnitude(const std::vector<std::complex<double>>& X) {
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [](const std::complex<double>& c) { return std::abs(c); });
    return out;
}

// Phase spectrum: arg(X[k]) in radians ∈ (−π, π]
inline std::vector<double> phase(const std::vector<std::complex<double>>& X) {
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [](const std::complex<double>& c) { return std::arg(c); });
    return out;
}

// Power spectrum: |X[k]|²
inline std::vector<double> powerSpectrum(const std::vector<std::complex<double>>& X) {
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [](const std::complex<double>& c) { return std::norm(c); });
    return out;
}

// Power spectrum in dB: 10·log₁₀(|X[k]|² / refPower)
// refPower defaults to 1.0 (0 dBFS convention).
inline std::vector<double> powerSpectrumDB(const std::vector<std::complex<double>>& X,
                                            double refPower = 1.0)
{
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [refPower](const std::complex<double>& c) {
                       return 10.0 * std::log10(
                           std::max(std::norm(c) / refPower, 1e-300));
                   });
    return out;
}

// Magnitude in dB: 20·log₁₀(|X[k]| / refMag)
inline std::vector<double> magnitudeDB(const std::vector<std::complex<double>>& X,
                                        double refMag = 1.0)
{
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [refMag](const std::complex<double>& c) {
                       return 20.0 * std::log10(
                           std::max(std::abs(c) / refMag, 1e-150));
                   });
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Frequency axis
// ─────────────────────────────────────────────────────────────────────────────

// Returns the frequency (Hz) for each of the n FFT output bins.
// Bins 0..N/2 are positive frequencies; bins N/2+1..N-1 are negative
// (matching NumPy fftfreq convention).
inline std::vector<double> fftFrequencies(size_t n, double sampleRate) {
    if (n == 0) return {};
    std::vector<double> f(n);
    double step = sampleRate / static_cast<double>(n);
    for (size_t k = 0; k < n; ++k)
        f[k] = (k <= n / 2) ? static_cast<double>(k) * step
                             : (static_cast<double>(k) - static_cast<double>(n)) * step;
    return f;
}

// Frequency axis for rfft output (N/2+1 bins, 0..sampleRate/2).
inline std::vector<double> rfftFrequencies(size_t n, double sampleRate) {
    size_t m = n / 2 + 1;
    std::vector<double> f(m);
    double step = sampleRate / static_cast<double>(n);
    for (size_t k = 0; k < m; ++k) f[k] = static_cast<double>(k) * step;
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// fftShift / ifftShift  (like numpy.fft.fftshift)
// ─────────────────────────────────────────────────────────────────────────────

// Shift zero-frequency component to the centre of the spectrum.
template<typename T>
inline std::vector<T> fftShift(const std::vector<T>& x) {
    if (x.empty()) return {};
    size_t n    = x.size();
    size_t half = n / 2;
    std::vector<T> out(n);
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(half), x.end(), out.begin());
    std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(half),
              out.begin() + static_cast<std::ptrdiff_t>(n - half));
    return out;
}

// Inverse of fftShift.
template<typename T>
inline std::vector<T> ifftShift(const std::vector<T>& x) {
    if (x.empty()) return {};
    size_t n    = x.size();
    size_t half = (n + 1) / 2;
    std::vector<T> out(n);
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(half), x.end(), out.begin());
    std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(half),
              out.begin() + static_cast<std::ptrdiff_t>(n - half));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Window-FFT pipeline helpers
// ─────────────────────────────────────────────────────────────────────────────

// Multiply signal by window element-wise.
inline std::vector<double> applyWindow(const std::vector<double>& signal,
                                        const std::vector<double>& window)
{
    if (signal.size() != window.size())
        throw std::invalid_argument(
            "applyWindow: signal length (" + std::to_string(signal.size()) +
            ") != window length (" + std::to_string(window.size()) + ")");
    std::vector<double> out(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) out[i] = signal[i] * window[i];
    return out;
}

// Windowed FFT in one call:
//   1. Multiply signal by window
//   2. Forward FFT
//   3. Return complex spectrum
inline std::vector<std::complex<double>>
windowedFFT(const std::vector<double>& signal,
            const std::vector<double>& window,
            FFTNorm norm = FFTNorm::None)
{
    auto windowed = applyWindow(signal, window);
    std::vector<std::complex<double>> cx(windowed.size());
    for (size_t i = 0; i < windowed.size(); ++i) cx[i] = windowed[i];
    FFTPlan::create(cx.size(), {FFTDirection::Forward, norm}).execute(cx);
    return cx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolution via FFT — O(N log N)
// ─────────────────────────────────────────────────────────────────────────────

// Linear convolution of a and b.
// Output length = a.size() + b.size() - 1.
inline std::vector<double> convolve(const std::vector<double>& a,
                                     const std::vector<double>& b)
{
    if (a.empty() || b.empty()) return {};
    size_t outLen = a.size() + b.size() - 1;

    // Next power of 2 ≥ outLen for zero-padding
    size_t n = 1;
    while (n < outLen) n <<= 1;

    auto toComplex = [&](const std::vector<double>& v) {
        std::vector<std::complex<double>> c(n, {0.0, 0.0});
        for (size_t i = 0; i < v.size(); ++i) c[i] = v[i];
        return c;
    };

    auto ca = toComplex(a);
    auto cb = toComplex(b);

    auto fwdPlan = FFTPlan::create(n);
    fwdPlan.execute(ca);
    fwdPlan.execute(cb);

    for (size_t i = 0; i < n; ++i) ca[i] *= cb[i];

    FFTPlan::create(n, {FFTDirection::Inverse, FFTNorm::ByN}).execute(ca);

    std::vector<double> out(outLen);
    for (size_t i = 0; i < outLen; ++i) out[i] = ca[i].real();
    return out;
}

// Cross-correlation of a and b: corr[k] = Σ conj(a[n]) · b[n+k]
// Output length = a.size() + b.size() - 1.
inline std::vector<double> correlate(const std::vector<double>& a,
                                      const std::vector<double>& b)
{
    if (a.empty() || b.empty()) return {};
    size_t outLen = a.size() + b.size() - 1;
    size_t n = 1;
    while (n < outLen) n <<= 1;

    auto toComplex = [&](const std::vector<double>& v) {
        std::vector<std::complex<double>> c(n, {0.0, 0.0});
        for (size_t i = 0; i < v.size(); ++i) c[i] = v[i];
        return c;
    };

    auto ca = toComplex(a);
    auto cb = toComplex(b);

    auto fwdPlan = FFTPlan::create(n);
    fwdPlan.execute(ca);
    fwdPlan.execute(cb);

    // Correlation theorem: IFFT(conj(A) · B)
    for (size_t i = 0; i < n; ++i) ca[i] = std::conj(ca[i]) * cb[i];

    FFTPlan::create(n, {FFTDirection::Inverse, FFTNorm::ByN}).execute(ca);

    // Rearrange: negative lags first, then positive lags
    std::vector<double> out(outLen);
    size_t negStart = n - (a.size() - 1);
    for (size_t i = 0; i < a.size() - 1; ++i)
        out[i] = ca[negStart + i].real();
    for (size_t i = 0; i < b.size(); ++i)
        out[a.size() - 1 + i] = ca[i].real();
    return out;
}

} // namespace SharedMath::DSP
