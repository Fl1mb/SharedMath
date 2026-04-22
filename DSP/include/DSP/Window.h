#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// Window functions for spectral analysis and filter design
//
// All functions return a std::vector<double> of length n.
//
// symmetric = true  (default) — suited for FIR filter design (Nyquist criterion)
// symmetric = false           — suited for spectral analysis (periodic, N+1 samples
//                               but only the first N are returned)
//
// Reference:
//   Harris, F.J. (1978). On the use of windows for harmonic analysis with
//   the discrete Fourier transform. Proc. IEEE, 66(1), 51–83.
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

constexpr double WIN_PI = 3.14159265358979323846;

// Effective denominator for symmetric vs periodic windows
inline size_t wM(size_t n, bool sym) { return sym ? n - 1 : n; }

// Modified Bessel function of the first kind, order 0: I₀(x)
// Converges to double precision in <50 iterations for all practical x.
inline double besselI0(double x) {
    double result = 1.0, term = 1.0, xh = x * x * 0.25;
    for (int k = 1; k <= 60; ++k) {
        term *= xh / static_cast<double>(k * k);
        result += term;
        if (term < 1e-17 * result) break;
    }
    return result;
}

} // namespace detail


// ── Rectangular (boxcar) ─────────────────────────────────────────────────────
// All-ones window. No sidelobe suppression; maximum frequency resolution.
inline std::vector<double> windowRectangular(size_t n) {
    return std::vector<double>(n, 1.0);
}


// ── Bartlett (triangular) ────────────────────────────────────────────────────
// Linear taper reaching zero at both ends.
// Peak sidelobe: −26 dB.
inline std::vector<double> windowBartlett(size_t n, bool symmetric = true) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    double h = M / 2.0;
    for (size_t i = 0; i < n; ++i)
        w[i] = 1.0 - std::abs((static_cast<double>(i) - h) / h);
    return w;
}


// ── Hann ─────────────────────────────────────────────────────────────────────
// Raised-cosine window. Good general-purpose choice.
// Peak sidelobe: −31 dB. Roll-off: −18 dB/octave.
inline std::vector<double> windowHann(size_t n, bool symmetric = true) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    for (size_t i = 0; i < n; ++i)
        w[i] = 0.5 * (1.0 - std::cos(2.0 * detail::WIN_PI * static_cast<double>(i) / M));
    return w;
}


// ── Hamming ──────────────────────────────────────────────────────────────────
// Optimised raised-cosine; minimises first sidelobe.
// Peak sidelobe: −42 dB. Does NOT reach zero at endpoints.
inline std::vector<double> windowHamming(size_t n, bool symmetric = true) {
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M = static_cast<double>(detail::wM(n, symmetric));
    for (size_t i = 0; i < n; ++i)
        w[i] = 0.54 - 0.46 * std::cos(2.0 * detail::WIN_PI * static_cast<double>(i) / M);
    return w;
}


// ── Blackman ─────────────────────────────────────────────────────────────────
// 3-term cosine sum. Good sidelobe suppression.
// Peak sidelobe: −58 dB. Roll-off: −18 dB/octave.
inline std::vector<double> windowBlackman(size_t n, bool symmetric = true) {
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


// ── Blackman-Harris ──────────────────────────────────────────────────────────
// 4-term cosine sum. Excellent sidelobe suppression.
// Peak sidelobe: −92 dB.
inline std::vector<double> windowBlackmanHarris(size_t n, bool symmetric = true) {
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


// ── Nuttall ───────────────────────────────────────────────────────────────────
// 4-term cosine sum with continuous first derivative.
// Peak sidelobe: −93 dB.
inline std::vector<double> windowNuttall(size_t n, bool symmetric = true) {
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


// ── Flat-top ─────────────────────────────────────────────────────────────────
// 5-term cosine sum. Very small amplitude error (< 0.01 dB) for sinusoids.
// Poor frequency resolution. Best for amplitude calibration measurements.
inline std::vector<double> windowFlatTop(size_t n, bool symmetric = true) {
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


// ── Kaiser ────────────────────────────────────────────────────────────────────
// Near-optimal window with adjustable sidelobe-vs-resolution trade-off.
//
//   beta ≈ 0    → rectangular
//   beta ≈ 5    → similar to Hamming   (−53 dB)
//   beta ≈ 8.6  → similar to Blackman  (−74 dB)
//   beta ≈ 14   → very high suppression (−120 dB)
//
// Rule of thumb for filter design:
//   beta = 0.1102 * (A - 8.7)          if A >= 50 dB
//   beta = 0.5842*(A-21)^0.4 + 0.07886*(A-21)  if 21 <= A < 50 dB
//   beta = 0                            if A < 21 dB
inline std::vector<double> windowKaiser(size_t n, double beta,
                                         bool symmetric = true) {
    if (beta < 0.0)
        throw std::invalid_argument("windowKaiser: beta must be >= 0");
    if (n == 0) return {};
    if (n == 1) return {1.0};
    std::vector<double> w(n);
    double M   = static_cast<double>(detail::wM(n, symmetric));
    double inv = 1.0 / detail::besselI0(beta);
    for (size_t i = 0; i < n; ++i) {
        double arg = 2.0 * static_cast<double>(i) / M - 1.0;  // ∈ [-1, 1]
        w[i] = detail::besselI0(beta * std::sqrt(std::max(0.0, 1.0 - arg * arg))) * inv;
    }
    return w;
}

// Helper: compute Kaiser beta from desired peak sidelobe attenuation (dB)
inline double kaiserBeta(double attenuationDB) {
    if (attenuationDB >= 50.0)
        return 0.1102 * (attenuationDB - 8.7);
    if (attenuationDB >= 21.0)
        return 0.5842 * std::pow(attenuationDB - 21.0, 0.4)
             + 0.07886 * (attenuationDB - 21.0);
    return 0.0;
}


// ── Gaussian ─────────────────────────────────────────────────────────────────
// Gaussian bell-curve window.
// sigma = 0.5 → −55 dB peak sidelobe (typical); smaller sigma → narrower.
inline std::vector<double> windowGaussian(size_t n, double sigma = 0.4,
                                           bool symmetric = true) {
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


// ── Tukey (tapered cosine) ────────────────────────────────────────────────────
// Flat in the middle, cosine-tapered at the edges.
//   alpha = 0 → rectangular
//   alpha = 1 → Hann
// Good for signals with known start/end times.
inline std::vector<double> windowTukey(size_t n, double alpha = 0.5,
                                        bool symmetric = true) {
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


// ── Bartlett-Hann ────────────────────────────────────────────────────────────
// Hybrid: combines Bartlett and Hann characteristics.
// Peak sidelobe: −35.9 dB.
inline std::vector<double> windowBartlettHann(size_t n, bool symmetric = true) {
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


// ── Planck-taper ─────────────────────────────────────────────────────────────
// Infinitely differentiable taper; used in gravitational-wave data analysis.
// epsilon ∈ (0, 0.5): fraction of window used for tapering on each side.
inline std::vector<double> windowPlanck(size_t n, double epsilon = 0.1,
                                         bool symmetric = true) {
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


// ─────────────────────────────────────────────────────────────────────────────
// Enum-based factory (for configuration-driven code)
// ─────────────────────────────────────────────────────────────────────────────

enum class WindowType {
    Rectangular,
    Bartlett,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Nuttall,
    FlatTop,
    Kaiser,
    Gaussian,
    Tukey,
    BartlettHann,
    Planck
};

struct WindowParams {
    WindowType type      = WindowType::Hann;
    bool       symmetric = true;
    // Type-specific parameters (only used by the relevant window):
    double beta          = 8.6;   // Kaiser: sidelobe attenuation control
    double sigma         = 0.4;   // Gaussian: width
    double alpha         = 0.5;   // Tukey: taper fraction
    double epsilon       = 0.1;   // Planck: taper fraction per side
};

inline std::vector<double> makeWindow(size_t n, const WindowParams& p = {}) {
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


// ─────────────────────────────────────────────────────────────────────────────
// Window metrics
// ─────────────────────────────────────────────────────────────────────────────

// Coherent power gain: CG = Σw[n] / N
// Divide spectral amplitudes by this to obtain correctly-scaled amplitudes.
inline double windowCoherentGain(const std::vector<double>& w) {
    if (w.empty()) return 0.0;
    double sum = 0.0;
    for (double v : w) sum += v;
    return sum / static_cast<double>(w.size());
}

// Equivalent Noise Bandwidth (in bins):  ENBW = N · Σw²[n] / (Σw[n])²
// Multiply noise floor by this to get the effective noise bandwidth in Hz
// when multiplied by the frequency resolution (fs/N).
inline double windowENBW(const std::vector<double>& w) {
    if (w.empty()) return 0.0;
    double s1 = 0.0, s2 = 0.0;
    for (double v : w) { s1 += v; s2 += v * v; }
    if (s1 == 0.0) return 0.0;
    return static_cast<double>(w.size()) * s2 / (s1 * s1);
}

// Processing gain (dB relative to rectangular):  PG = −20·log10(CG)
inline double windowProcessingGain(const std::vector<double>& w) {
    double cg = windowCoherentGain(w);
    if (cg <= 0.0) return 0.0;
    return -20.0 * std::log10(cg);
}

} // namespace SharedMath::DSP
