/**
 * @file Window.cpp
 * @brief Implementation of window functions for spectral analysis.
 */

#include "Window.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace SharedMath::DSP {

namespace detail {

constexpr double WIN_PI = 3.14159265358979323846;

/// Effective denominator for symmetric vs periodic windows
inline size_t wM(size_t n, bool sym) { return sym ? n - 1 : n; }

/// Modified Bessel function of the first kind, order 0: I₀(x)
/// Converges to double precision in <50 iterations for all practical x.
double besselI0(double x) {
    double result = 1.0, term = 1.0, xh = x * x * 0.25;
    for (int k = 1; k <= 60; ++k) {
        term *= xh / static_cast<double>(k * k);
        result += term;
        if (term < 1e-17 * result) break;
    }
    return result;
}

} // namespace detail

// ── Rectangular ─────────────────────────────────────────────────────────────
std::vector<double> windowRectangular(size_t n) {
    return std::vector<double>(n, 1.0);
}

// ── Bartlett ────────────────────────────────────────────────────────────────
std::vector<double> windowBartlett(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    double h = M / 2.0;
    for (size_t i = 0; i < n; ++i)
        w[i] = 1.0 - std::abs((static_cast<double>(i) - h) / h);
    return w;
}

// ── Hann ────────────────────────────────────────────────────────────────────
std::vector<double> windowHann(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    for (size_t i = 0; i < n; ++i)
        w[i] = 0.5 * (1.0 - std::cos(2.0 * detail::WIN_PI * static_cast<double>(i) / M));
    return w;
}

// ── Hamming ─────────────────────────────────────────────────────────────────
std::vector<double> windowHamming(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    for (size_t i = 0; i < n; ++i)
        w[i] = 0.54 - 0.46 * std::cos(2.0 * detail::WIN_PI * static_cast<double>(i) / M);
    return w;
}

// ── Blackman ────────────────────────────────────────────────────────────────
std::vector<double> windowBlackman(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    for (size_t i = 0; i < n; ++i) {
        double x = 2.0 * detail::WIN_PI * static_cast<double>(i) / M;
        w[i] = 0.42 - 0.50 * std::cos(x) + 0.08 * std::cos(2.0 * x);
    }
    return w;
}

// ── Blackman-Harris ─────────────────────────────────────────────────────────
std::vector<double> windowBlackmanHarris(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    constexpr double a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168;
    for (size_t i = 0; i < n; ++i) {
        double x = 2.0 * detail::WIN_PI * static_cast<double>(i) / M;
        w[i] = a0 - a1 * std::cos(x) + a2 * std::cos(2.0 * x) - a3 * std::cos(3.0 * x);
    }
    return w;
}

// ── Nuttall ─────────────────────────────────────────────────────────────────
std::vector<double> windowNuttall(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    constexpr double a0 = 0.3635819, a1 = 0.4891775, a2 = 0.1365995, a3 = 0.0106411;
    for (size_t i = 0; i < n; ++i) {
        double x = 2.0 * detail::WIN_PI * static_cast<double>(i) / M;
        w[i] = a0 - a1 * std::cos(x) + a2 * std::cos(2.0 * x) - a3 * std::cos(3.0 * x);
    }
    return w;
}

// ── Flat-top ────────────────────────────────────────────────────────────────
std::vector<double> windowFlatTop(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    constexpr double a0 = 0.21557895, a1 = 0.41663158,
                     a2 = 0.27726316, a3 = 0.08357895, a4 = 0.00694737;
    for (size_t i = 0; i < n; ++i) {
        double x = 2.0 * detail::WIN_PI * static_cast<double>(i) / M;
        w[i] = a0 - a1 * std::cos(x)       + a2 * std::cos(2.0 * x)
                  - a3 * std::cos(3.0 * x) + a4 * std::cos(4.0 * x);
    }
    return w;
}

// ── Kaiser ──────────────────────────────────────────────────────────────────
std::vector<double> windowKaiser(size_t n, double beta, bool symmetric) {
    if (beta < 0.0)
        throw std::invalid_argument("windowKaiser: beta must be >= 0");
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M   = static_cast<double>(detail::wM(n, symmetric));
    double inv = 1.0 / detail::besselI0(beta);
    for (size_t i = 0; i < n; ++i) {
        double arg = 2.0 * static_cast<double>(i) / M - 1.0;
        w[i] = detail::besselI0(beta * std::sqrt(std::max(0.0, 1.0 - arg * arg))) * inv;
    }
    return w;
}

double kaiserBeta(double attenuationDB) {
    if (attenuationDB >= 50.0)
        return 0.1102 * (attenuationDB - 8.7);
    if (attenuationDB >= 21.0)
        return 0.5842 * std::pow(attenuationDB - 21.0, 0.4)
             + 0.07886 * (attenuationDB - 21.0);
    return 0.0;
}

// ── Gaussian ────────────────────────────────────────────────────────────────
std::vector<double> windowGaussian(size_t n, double sigma, bool symmetric) {
    if (sigma <= 0.0)
        throw std::invalid_argument("windowGaussian: sigma must be > 0");
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M  = static_cast<double>(detail::wM(n, symmetric));
    double h  = M / 2.0;
    for (size_t i = 0; i < n; ++i) {
        double x = (static_cast<double>(i) - h) / (sigma * h);
        w[i] = std::exp(-0.5 * x * x);
    }
    return w;
}

// ── Tukey ───────────────────────────────────────────────────────────────────
std::vector<double> windowTukey(size_t n, double alpha, bool symmetric) {
    if (alpha < 0.0 || alpha > 1.0)
        throw std::invalid_argument("windowTukey: alpha must be in [0, 1]");
    if (n == 0) return {};
    if (n == 1) return {1.0};

    std::vector<double> w(n);
    double M     = static_cast<double>(detail::wM(n, symmetric));
    double width = alpha * M / 2.0;

    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        if (width < 1.0 || (x >= width && x <= M - width)) {
            w[i] = 1.0;
        } else if (x < width) {
            w[i] = 0.5 * (1.0 - std::cos(detail::WIN_PI * x / width));
        } else {
            w[i] = 0.5 * (1.0 - std::cos(detail::WIN_PI * (M - x) / width));
        }
    }
    return w;
}

// ── Bartlett-Hann ───────────────────────────────────────────────────────────
std::vector<double> windowBartlettHann(size_t n, bool symmetric) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(i) / M;
        w[i] = 0.62 - 0.48 * std::abs(x - 0.5)
                    - 0.38 * std::cos(2.0 * detail::WIN_PI * x);
    }
    return w;
}

// ── Planck-taper ────────────────────────────────────────────────────────────
std::vector<double> windowPlanck(size_t n, double epsilon, bool symmetric) {
    if (epsilon <= 0.0 || epsilon >= 0.5)
        throw std::invalid_argument("windowPlanck: epsilon must be in (0, 0.5)");
    if (n == 0) return {};
    if (n == 1) return {1.0};

    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    double e = epsilon * M;

    auto taper = [](double t, double ep) -> double {
        if (t <= 0.0) return 0.0;
        if (t >= ep)  return 1.0;
        return 1.0 / (std::exp(ep / t - ep / (ep - t)) + 1.0);
    };

    for (size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i);
        w[i] = taper(t, e) * taper(M - t, e);
    }
    return w;
}

// ── Enum-based factory ──────────────────────────────────────────────────────
std::vector<double> makeWindow(size_t n, const WindowParams& p) {
    switch (p.type) {
        case WindowType::Rectangular:    return windowRectangular(n);
        case WindowType::Bartlett:       return windowBartlett(n, p.symmetric);
        case WindowType::Hann:           return windowHann(n, p.symmetric);
        case WindowType::Hamming:        return windowHamming(n, p.symmetric);
        case WindowType::Blackman:       return windowBlackman(n, p.symmetric);
        case WindowType::BlackmanHarris: return windowBlackmanHarris(n, p.symmetric);
        case WindowType::Nuttall:        return windowNuttall(n, p.symmetric);
        case WindowType::FlatTop:        return windowFlatTop(n, p.symmetric);
        case WindowType::Kaiser:         return windowKaiser(n, p.beta, p.symmetric);
        case WindowType::Gaussian:       return windowGaussian(n, p.sigma, p.symmetric);
        case WindowType::Tukey:          return windowTukey(n, p.alpha, p.symmetric);
        case WindowType::BartlettHann:   return windowBartlettHann(n, p.symmetric);
        case WindowType::Planck:         return windowPlanck(n, p.epsilon, p.symmetric);
        default:                         return windowRectangular(n);
    }
}

// ── Window metrics ──────────────────────────────────────────────────────────
double windowCoherentGain(const std::vector<double>& w) {
    if (w.empty()) return 0.0;
    double sum = 0.0;
    for (double v : w) sum += v;
    return sum / static_cast<double>(w.size());
}

double windowENBW(const std::vector<double>& w) {
    if (w.empty()) return 0.0;
    double s1 = 0.0, s2 = 0.0;
    for (double v : w) { s1 += v; s2 += v * v; }
    if (s1 == 0.0) return 0.0;
    return static_cast<double>(w.size()) * s2 / (s1 * s1);
}

double windowProcessingGain(const std::vector<double>& w) {
    double cg = windowCoherentGain(w);
    if (cg <= 0.0) return 0.0;
    return -20.0 * std::log10(cg);
}

} // namespace SharedMath::DSP