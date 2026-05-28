/**
 * @file FFT.cpp
 * @brief Implementation of high-level FFT convenience functions.
 */

#include "FFT.h"
#include "FFTPlan.h"
#include "FFTConfig.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace SharedMath::DSP {

// ── In-place forward FFT ─────────────────────────────────────────────────────
void fft(std::vector<std::complex<double>>& x, FFTNorm norm)
{
    if (x.empty()) return;
    FFTPlan::create(x.size(), {FFTDirection::Forward, norm}).execute(x);
}

// ── In-place inverse FFT ─────────────────────────────────────────────────────
void ifft(std::vector<std::complex<double>>& X, FFTNorm norm)
{
    if (X.empty()) return;
    FFTPlan::create(X.size(), {FFTDirection::Inverse, norm}).execute(X);
}

// ── Real FFT (half-spectrum) ─────────────────────────────────────────────────
std::vector<std::complex<double>> rfft(const std::vector<double>& x, FFTNorm norm)
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
std::vector<double> irfft(const std::vector<std::complex<double>>& X,
                           size_t n,
                           FFTNorm norm)
{
    if (X.empty() || n == 0) return {};

    std::vector<std::complex<double>> full(n, {0.0, 0.0});
    const size_t half = n / 2 + 1;
    
    for (size_t k = 0; k < half && k < X.size(); ++k)
        full[k] = X[k];

    const size_t last_pair = (n % 2 == 0) ? (n / 2 - 1) : ((n - 1) / 2);
    
    for (size_t k = 1; k <= last_pair; ++k) {
        full[n - k] = std::conj(X[k]);
    }

    FFTPlan::create(n, {FFTDirection::Inverse, norm}).execute(full);

    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = full[i].real();
    return out;
}

/// ─────────────────────────────────────────────────────────────────────────────
/// Spectral analysis helpers
/// ─────────────────────────────────────────────────────────────────────────────

std::vector<double> magnitude(const std::vector<std::complex<double>>& X)
{
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [](const std::complex<double>& c) { return std::abs(c); });
    return out;
}

std::vector<double> phase(const std::vector<std::complex<double>>& X)
{
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [](const std::complex<double>& c) { return std::arg(c); });
    return out;
}

std::vector<double> powerSpectrum(const std::vector<std::complex<double>>& X)
{
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [](const std::complex<double>& c) { return std::norm(c); });
    return out;
}

std::vector<double> powerSpectrumDB(const std::vector<std::complex<double>>& X,
                                    double refPower)
{
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [refPower](const std::complex<double>& c) {
                       return 10.0 * std::log10(
                           std::max(std::norm(c) / refPower, 1e-300));
                   });
    return out;
}

std::vector<double> magnitudeDB(const std::vector<std::complex<double>>& X,
                                double refMag)
{
    std::vector<double> out(X.size());
    std::transform(X.begin(), X.end(), out.begin(),
                   [refMag](const std::complex<double>& c) {
                       return 20.0 * std::log10(
                           std::max(std::abs(c) / refMag, 1e-150));
                   });
    return out;
}

/// ─────────────────────────────────────────────────────────────────────────────
/// Frequency axis
/// ─────────────────────────────────────────────────────────────────────────────

std::vector<double> fftFrequencies(size_t n, double sampleRate)
{
    if (n == 0) return {};
    std::vector<double> f(n);
    double step = sampleRate / static_cast<double>(n);
    for (size_t k = 0; k < n; ++k)
        f[k] = (k <= n / 2) ? static_cast<double>(k) * step
                             : (static_cast<double>(k) - static_cast<double>(n)) * step;
    return f;
}

std::vector<double> rfftFrequencies(size_t n, double sampleRate)
{
    size_t m = n / 2 + 1;
    std::vector<double> f(m);
    double step = sampleRate / static_cast<double>(n);
    for (size_t k = 0; k < m; ++k) f[k] = static_cast<double>(k) * step;
    return f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Window-FFT pipeline helpers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<double> applyWindow(const std::vector<double>& signal,
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

std::vector<std::complex<double>>
windowedFFT(const std::vector<double>& signal,
            const std::vector<double>& window,
            FFTNorm norm)
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

std::vector<double> convolve(const std::vector<double>& a,
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

std::vector<double> correlate(const std::vector<double>& a,
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

    /// Rearrange: negative lags first, then positive lags
    std::vector<double> out(outLen);
    size_t negStart = n - (a.size() - 1);
    for (size_t i = 0; i < a.size() - 1; ++i)
        out[i] = ca[negStart + i].real();
    for (size_t i = 0; i < b.size(); ++i)
        out[a.size() - 1 + i] = ca[i].real();
    return out;
}

} // namespace SharedMath::DSP
