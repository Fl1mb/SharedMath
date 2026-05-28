/**
 * @file Utils.cpp
 * @brief Implementation of practical signal-processing helpers.
 */

#include "SignalProcessing.h"

#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────
double mean(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    return std::accumulate(x.begin(), x.end(), 0.0) /
           static_cast<double>(x.size());
}

double rms(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double sum = 0.0;
    for (double v : x) sum += v * v;
    return std::sqrt(sum / static_cast<double>(x.size()));
}

double peakAbs(const std::vector<double>& x) {
    if (x.empty()) return 0.0;
    double pk = 0.0;
    for (double v : x) pk = std::max(pk, std::abs(v));
    return pk;
}

// ─────────────────────────────────────────────────────────────────────────────
// DC / normalization
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> removeDC(const std::vector<double>& x) {
    if (x.empty()) return {};
    const double mu = mean(x);
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] - mu;
    return out;
}

std::vector<double> normalizePeak(
    const std::vector<double>& x,
    double targetPeak)
{
    if (x.empty()) return {};
    if (targetPeak <= 0.0)
        throw std::invalid_argument("normalizePeak: targetPeak must be > 0");
    const double pk = peakAbs(x);
    if (pk == 0.0)
        throw std::invalid_argument("normalizePeak: signal is all-zero");
    const double scale = targetPeak / pk;
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * scale;
    return out;
}

std::vector<double> normalizeRMS(
    const std::vector<double>& x,
    double targetRMS)
{
    if (x.empty()) return {};
    if (targetRMS <= 0.0)
        throw std::invalid_argument("normalizeRMS: targetRMS must be > 0");
    const double r = rms(x);
    if (r == 0.0)
        throw std::invalid_argument("normalizeRMS: signal RMS is zero");
    const double scale = targetRMS / r;
    std::vector<double> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) out[i] = x[i] * scale;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Smoothing
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> movingAverage(
    const std::vector<double>& x,
    size_t windowSize)
{
    if (x.empty()) return {};
    if (windowSize == 0)
        throw std::invalid_argument("movingAverage: windowSize must be > 0");

    const size_t n = x.size();
    const int    r = static_cast<int>(windowSize) / 2;

    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) {
        const int lo = std::max(0, static_cast<int>(i) - r);
        const int hi = std::min(static_cast<int>(n) - 1,
                                static_cast<int>(i) + r);
        double s = 0.0;
        for (int j = lo; j <= hi; ++j) s += x[j];
        out[i] = s / static_cast<double>(hi - lo + 1);
    }
    return out;
}

std::vector<double> exponentialSmoothing(
    const std::vector<double>& x,
    double alpha)
{
    if (x.empty()) return {};
    if (alpha <= 0.0 || alpha > 1.0)
        throw std::invalid_argument(
            "exponentialSmoothing: alpha must be in (0, 1]");

    std::vector<double> out(x.size());
    out[0] = x[0];
    const double beta = 1.0 - alpha;
    for (size_t i = 1; i < x.size(); ++i)
        out[i] = alpha * x[i] + beta * out[i - 1];
    return out;
}

std::vector<double> medianFilter(
    const std::vector<double>& x,
    size_t kernelSize)
{
    if (x.empty()) return {};
    if (kernelSize == 0)
        throw std::invalid_argument("medianFilter: kernelSize must be > 0");

    // Force odd kernel
    if (kernelSize % 2 == 0) ++kernelSize;

    const size_t n = x.size();
    const int    r = static_cast<int>(kernelSize) / 2;

    std::vector<double> out(n);
    std::vector<double> win;
    win.reserve(kernelSize);

    for (size_t i = 0; i < n; ++i) {
        win.clear();
        const int lo = std::max(0, static_cast<int>(i) - r);
        const int hi = std::min(static_cast<int>(n) - 1,
                                static_cast<int>(i) + r);
        for (int j = lo; j <= hi; ++j) win.push_back(x[j]);
        std::sort(win.begin(), win.end());
        out[i] = win[win.size() / 2];
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Trend removal
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> detrendLinear(const std::vector<double>& x) {
    if (x.size() < 2) return x;

    const size_t n = x.size();
    const double nd = static_cast<double>(n);

    const double sum_t  = nd * (nd - 1.0) * 0.5;
    const double sum_t2 = nd * (nd - 1.0) * (2.0 * nd - 1.0) / 6.0;

    double sum_x = 0.0, sum_xt = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_x  += x[i];
        sum_xt += static_cast<double>(i) * x[i];
    }

    const double denom = nd * sum_t2 - sum_t * sum_t;
    const double b     = (nd * sum_xt - sum_t * sum_x) / denom;
    const double a     = (sum_x - b * sum_t) / nd;

    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = x[i] - (a + b * static_cast<double>(i));
    return out;
}

} // namespace SharedMath::DSP