#pragma once

// SharedMath::DSP — Polyphase resampling
//
// upfirdn(signal, h, L, M)
//   Core primitive: upsample by L, FIR filter h, downsample by M.
//   If h is empty an identity (single 1.0 tap) is used.
//   Output length = ceil((N*L + P - 1) / M),  P = len(h).
//
// interpolate(signal, L, h = {})
//   Upsample by integer factor L. Uses a Kaiser low-pass if h is empty.
//   FIR gain is scaled by L so amplitude is preserved.
//
// decimate(signal, M, h = {})
//   Downsample by integer factor M. Uses a Kaiser low-pass if h is empty.
//
// resamplePolyphase(signal, L, M, h = {})
//   Rational L/M resampling built on upfirdn. h defaults to a
//   Kaiser low-pass at fc = 1/max(L,M).
//
// resampleTo(signal, inputRate, outputRate, h = {})
//   Convenience wrapper that reduces the rational ratio inputRate→outputRate,
//   applies anti-alias filtering, compensates the FIR group delay, and returns
//   roughly round(N * outputRate / inputRate) samples.

#include "FIR.h"

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace SharedMath::DSP {

namespace detail {

inline std::vector<double> designDefaultResampleFIR(size_t L, size_t M) {
    const size_t maxLM = std::max(L, M);
    const double fc = 1.0 / static_cast<double>(maxLM);
    const double tw = std::max(fc * 0.1, 1e-6);
    auto fir = designKaiserFIR(fc, tw, 70.0, FIRType::LowPass);
    const double gain = static_cast<double>(L);
    for (double& c : fir) c *= gain;
    return fir;
}

inline size_t resampledLength(size_t n, size_t L, size_t M) {
    return (n * L + M - 1) / M;
}

inline std::vector<double> trimResamplingDelay(
    const std::vector<double>& y,
    size_t expectedLen,
    size_t filterLen,
    size_t downFactor)
{
    if (expectedLen == 0 || y.empty()) return {};

    const size_t groupDelay = (filterLen > 0) ? (filterLen - 1) / 2 : 0;
    const size_t start = (groupDelay + downFactor - 1) / downFactor;

    std::vector<double> out;
    out.reserve(expectedLen);
    for (size_t i = 0; i < expectedLen; ++i) {
        const size_t src = start + i;
        out.push_back(src < y.size() ? y[src] : 0.0);
    }
    return out;
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// upfirdn — upsample by L, filter, downsample by M
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> upfirdn(
    const std::vector<double>& signal,
    const std::vector<double>& h,
    size_t L,
    size_t M)
{
    if (L == 0) throw std::invalid_argument("upfirdn: L must be >= 1");
    if (M == 0) throw std::invalid_argument("upfirdn: M must be >= 1");
    if (signal.empty()) return {};

    const std::vector<double> identity{1.0};
    const std::vector<double>* pFilt = h.empty() ? &identity : &h;

    const size_t N = signal.size();
    const size_t P = pFilt->size();

    // Upsampled length before filtering: N*L  (with L-1 zeros between each)
    // After filtering (full convolution): N*L + P - 1
    // After downsampling by M: ceil((N*L + P - 1) / M)
    const size_t upLen  = N * L;
    const size_t convLen = upLen + P - 1;
    const size_t outLen  = (convLen + M - 1) / M;

    std::vector<double> out(outLen, 0.0);

    // Compute only the output samples we keep (every M-th of the convolution).
    // For output index i, convolution index is i*M.
    // conv[n] = sum_{k=0..P-1} h[k] * up[n-k]
    // up[j] = signal[j/L] if j%L==0, else 0
    // So only terms where (n-k) % L == 0 contribute.
    const auto& coeff = *pFilt;
    for (size_t i = 0; i < outLen; ++i) {
        const size_t n = i * M;   // convolution index
        double y = 0.0;
        // k: tap index; up[n-k] nonzero iff (n-k) % L == 0 and 0 <= (n-k)/L < N
        // k must satisfy: k <= n, k < P, (n-k) % L == 0
        // The valid k values are k = n % L, n % L + L, n % L + 2*L, ...
        const size_t k0 = n % L;
        for (size_t k = k0; k < P && k <= n; k += L) {
            const size_t sigIdx = (n - k) / L;
            if (sigIdx < N)
                y += coeff[k] * signal[sigIdx];
        }
        out[i] = y;
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// interpolate — upsample by integer factor L
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> interpolate(
    const std::vector<double>& signal,
    size_t L,
    const std::vector<double>& h = {})
{
    if (L == 0) throw std::invalid_argument("interpolate: L must be >= 1");
    if (signal.empty()) return {};
    if (L == 1) return signal;

    std::vector<double> fir;
    if (h.empty()) {
        // Auto-design: fc = 1/L (normalized), transition width = fc*0.1
        const double fc = 1.0 / static_cast<double>(L);
        const double tw = fc * 0.1;
        fir = designKaiserFIR(fc, tw, 60.0, FIRType::LowPass);
        // Scale by L to compensate for the gain reduction from upsampling
        const double Ld = static_cast<double>(L);
        for (double& c : fir) c *= Ld;
    }

    return upfirdn(signal, h.empty() ? fir : h, L, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// decimate — downsample by integer factor M (with anti-alias filter)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> decimate(
    const std::vector<double>& signal,
    size_t M,
    const std::vector<double>& h = {})
{
    if (M == 0) throw std::invalid_argument("decimate: M must be >= 1");
    if (signal.empty()) return {};
    if (M == 1) return signal;

    std::vector<double> fir;
    if (h.empty()) {
        const double fc = 1.0 / static_cast<double>(M);
        const double tw = fc * 0.1;
        fir = designKaiserFIR(fc, tw, 60.0, FIRType::LowPass);
    }

    return upfirdn(signal, h.empty() ? fir : h, 1, M);
}

// ─────────────────────────────────────────────────────────────────────────────
// resamplePolyphase — rational L/M resampling
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> resamplePolyphase(
    const std::vector<double>& signal,
    size_t L,
    size_t M,
    const std::vector<double>& h = {})
{
    if (L == 0) throw std::invalid_argument("resamplePolyphase: L must be >= 1");
    if (M == 0) throw std::invalid_argument("resamplePolyphase: M must be >= 1");
    if (signal.empty()) return {};

    std::vector<double> fir;
    if (h.empty()) {
        fir = detail::designDefaultResampleFIR(L, M);
    }

    return upfirdn(signal, h.empty() ? fir : h, L, M);
}

// ─────────────────────────────────────────────────────────────────────────────
// resamplePolyphaseAligned — rational resampling with FIR delay compensation
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> resamplePolyphaseAligned(
    const std::vector<double>& signal,
    size_t L,
    size_t M,
    const std::vector<double>& h = {})
{
    if (L == 0)
        throw std::invalid_argument("resamplePolyphaseAligned: L must be >= 1");
    if (M == 0)
        throw std::invalid_argument("resamplePolyphaseAligned: M must be >= 1");
    if (signal.empty()) return {};

    std::vector<double> fir = h.empty()
        ? detail::designDefaultResampleFIR(L, M)
        : h;

    const auto full = upfirdn(signal, fir, L, M);
    const size_t expectedLen = detail::resampledLength(signal.size(), L, M);
    return detail::trimResamplingDelay(full, expectedLen, fir.size(), M);
}

// ─────────────────────────────────────────────────────────────────────────────
// resampleTo — sample-rate based convenience wrapper
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> resampleTo(
    const std::vector<double>& signal,
    size_t inputRate,
    size_t outputRate,
    const std::vector<double>& h = {})
{
    if (inputRate == 0)
        throw std::invalid_argument("resampleTo: inputRate must be >= 1");
    if (outputRate == 0)
        throw std::invalid_argument("resampleTo: outputRate must be >= 1");
    if (signal.empty()) return {};
    if (inputRate == outputRate) return signal;

    const size_t g = std::gcd(inputRate, outputRate);
    const size_t L = outputRate / g;
    const size_t M = inputRate / g;
    return resamplePolyphaseAligned(signal, L, M, h);
}

} // namespace SharedMath::DSP
