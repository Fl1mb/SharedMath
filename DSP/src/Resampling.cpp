/**
 * @file Resampling.cpp
 * @brief Implementation of polyphase resampling functions.
 */

#include "DSP/Resampling.h"
#include "DSP/FIR.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace SharedMath::DSP {

namespace detail {

std::vector<double> designDefaultResampleFIR(size_t L, size_t M) {
    const size_t maxLM = std::max(L, M);
    const double fc = 1.0 / static_cast<double>(maxLM);
    const double tw = std::max(fc * 0.1, 1e-6);
    auto fir = designKaiserFIR(fc, tw, 70.0, FIRType::LowPass);
    const double gain = static_cast<double>(L);
    for (double& c : fir) c *= gain;
    return fir;
}

size_t resampledLength(size_t n, size_t L, size_t M) {
    return (n * L + M - 1) / M;
}

std::vector<double> trimResamplingDelay(
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
// upfirdn
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> upfirdn(
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

    const size_t upLen  = N * L;
    const size_t convLen = upLen + P - 1;
    const size_t outLen  = (convLen + M - 1) / M;

    std::vector<double> out(outLen, 0.0);

    const auto& coeff = *pFilt;
    for (size_t i = 0; i < outLen; ++i) {
        const size_t n = i * M;
        double y = 0.0;
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
// interpolate
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> interpolate(
    const std::vector<double>& signal,
    size_t L,
    const std::vector<double>& h)
{
    if (L == 0) throw std::invalid_argument("interpolate: L must be >= 1");
    if (signal.empty()) return {};
    if (L == 1) return signal;

    std::vector<double> fir;
    if (h.empty()) {
        const double fc = 1.0 / static_cast<double>(L);
        const double tw = fc * 0.1;
        fir = designKaiserFIR(fc, tw, 60.0, FIRType::LowPass);
        const double Ld = static_cast<double>(L);
        for (double& c : fir) c *= Ld;
    }

    return upfirdn(signal, h.empty() ? fir : h, L, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// decimate
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> decimate(
    const std::vector<double>& signal,
    size_t M,
    const std::vector<double>& h)
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
// resamplePolyphase
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> resamplePolyphase(
    const std::vector<double>& signal,
    size_t L,
    size_t M,
    const std::vector<double>& h)
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
// resamplePolyphaseAligned
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> resamplePolyphaseAligned(
    const std::vector<double>& signal,
    size_t L,
    size_t M,
    const std::vector<double>& h)
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
// resampleTo
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> resampleTo(
    const std::vector<double>& signal,
    size_t inputRate,
    size_t outputRate,
    const std::vector<double>& h)
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
