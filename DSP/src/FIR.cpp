/**
 * @file FIR.cpp
 * @brief Implementation of FIR filter design and application functions.
 */

#include "FIR.h"
#include "Window.h"
#include "Convolution.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>

namespace SharedMath::DSP {

namespace detail {

constexpr double FIR_PI = 3.14159265358979323846;

/// Ideal (unwindowed) low-pass sinc coefficients, Type I (even M).
/// Normalized cutoff fc ∈ (0,1); returns M+1 coefficients.
std::vector<double> idealLPCoeffs(size_t M, double fc) {
    std::vector<double> h(M + 1);
    double center = static_cast<double>(M) * 0.5;
    for (size_t i = 0; i <= M; ++i) {
        double n = static_cast<double>(i) - center;
        h[i] = (n == 0.0) ? fc : std::sin(FIR_PI * fc * n) / (FIR_PI * n);
    }
    return h;
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// designFIRLowPass
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRLowPass(
    size_t order,
    double fc,
    const WindowParams& wp)
{
    if (fc <= 0.0 || fc >= 1.0)
        throw std::invalid_argument("designFIRLowPass: fc must be in (0, 1)");
    if (order < 2) order = 2;
    if (order % 2 != 0) ++order;

    auto h = detail::idealLPCoeffs(order, fc);
    auto w = makeWindow(order + 1, wp);
    for (size_t i = 0; i <= order; ++i) h[i] *= w[i];
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// designFIRHighPass
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRHighPass(
    size_t order,
    double fc,
    const WindowParams& wp)
{
    if (fc <= 0.0 || fc >= 1.0)
        throw std::invalid_argument("designFIRHighPass: fc must be in (0, 1)");
    if (order < 2) order = 2;
    if (order % 2 != 0) ++order;

    auto h = detail::idealLPCoeffs(order, fc);
    auto w = makeWindow(order + 1, wp);
    for (size_t i = 0; i <= order; ++i) h[i] *= w[i];

    for (auto& v : h) v = -v;
    h[order / 2] += 1.0;
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// designFIRBandPass
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRBandPass(
    size_t order,
    double fcLow,
    double fcHigh,
    const WindowParams& wp)
{
    if (fcLow <= 0.0 || fcHigh >= 1.0 || fcLow >= fcHigh)
        throw std::invalid_argument(
            "designFIRBandPass: require 0 < fcLow < fcHigh < 1");
    if (order < 2) order = 2;
    if (order % 2 != 0) ++order;

    auto hH = detail::idealLPCoeffs(order, fcHigh);
    auto hL = detail::idealLPCoeffs(order, fcLow);
    auto w  = makeWindow(order + 1, wp);

    std::vector<double> h(order + 1);
    for (size_t i = 0; i <= order; ++i)
        h[i] = (hH[i] - hL[i]) * w[i];
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// designFIRBandStop
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designFIRBandStop(
    size_t order,
    double fcLow,
    double fcHigh,
    const WindowParams& wp)
{
    auto h = designFIRBandPass(order, fcLow, fcHigh, wp);
    size_t M = h.size() - 1;
    for (auto& v : h) v = -v;
    h[M / 2] += 1.0;
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// designKaiserFIR
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> designKaiserFIR(
    double fc,
    double transitionWidth,
    double attenuationDB,
    FIRType type)
{
    if (transitionWidth <= 0.0 || transitionWidth >= 1.0)
        throw std::invalid_argument(
            "designKaiserFIR: transitionWidth must be in (0, 1)");
    if (attenuationDB <= 0.0)
        throw std::invalid_argument(
            "designKaiserFIR: attenuationDB must be > 0");
    if (type == FIRType::BandPass || type == FIRType::BandStop)
        throw std::invalid_argument(
            "designKaiserFIR: use two-cutoff variants for BandPass/BandStop");

    const double A    = attenuationDB;
    const double beta = kaiserBeta(A);
    const double orderD =
        (A - 7.95) / (2.285 * 2.0 * detail::FIR_PI * transitionWidth);
    size_t order = (orderD < 2.0) ? size_t{2}
                                  : static_cast<size_t>(std::ceil(orderD));
    if (order % 2 != 0) ++order;

    WindowParams wp{};
    wp.type = WindowType::Kaiser;
    wp.beta = beta;

    return (type == FIRType::HighPass)
               ? designFIRHighPass(order, fc, wp)
               : designFIRLowPass(order, fc, wp);
}

// ─────────────────────────────────────────────────────────────────────────────
// applyFIR
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> applyFIR(
    const std::vector<double>& signal,
    const std::vector<double>& h)
{
    if (signal.empty() || h.empty()) return signal;
    return convolveLinear(signal, h, ConvolutionMode::Same);
}

// ─────────────────────────────────────────────────────────────────────────────
// filtfilt
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> filtfilt(
    const std::vector<double>& signal,
    const std::vector<double>& h)
{
    if (signal.empty() || h.empty()) return signal;

    auto y = convolveLinear(signal, h, ConvolutionMode::Same);
    std::reverse(y.begin(), y.end());
    y = convolveLinear(y, h, ConvolutionMode::Same);
    std::reverse(y.begin(), y.end());
    return y;
}

} // namespace SharedMath::DSP