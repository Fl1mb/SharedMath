#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace SharedMath::DSP {

namespace detail {
    constexpr double IIR_PI = 3.14159265358979323846;
}

// ─────────────────────────────────────────────────────────────────────────────
// BiquadCoeffs — second-order section (SOS), Transposed Direct Form II
// Difference equation:
//   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
// (a0 is assumed normalized to 1)
// ─────────────────────────────────────────────────────────────────────────────
struct BiquadCoeffs {
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;

    // Stability: poles are roots of z^2 + a1*z + a2 = 0; require |p| < 1.
    bool isStable() const {
        const double disc = a1 * a1 - 4.0 * a2;
        if (disc >= 0.0) {
            const double s = std::sqrt(disc);
            const double p1 = (-a1 + s) * 0.5;
            const double p2 = (-a1 - s) * 0.5;
            return std::abs(p1) < 1.0 && std::abs(p2) < 1.0;
        }
        // Complex-conjugate poles: |p|^2 = a2, and we also need a2 >= 0 here.
        return a2 < 1.0 && a2 >= 0.0;
    }

    // Transposed Direct Form II single-sample processing.
    inline double process(double x, double& s1, double& s2) const {
        const double y = b0 * x + s1;
        s1 = b1 * x - a1 * y + s2;
        s2 = b2 * x - a2 * y;
        return y;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: design analog-prototype Butterworth poles (normalized, ωc=1)
// Returns pairs of complex-conjugate poles as (realPart, imagPart) with imag>0,
// and a bool indicating whether there is a real pole at s = -1 (odd order).
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {
    struct ButterPrototype {
        std::vector<std::pair<double, double>> cplxPairs; // (sigma, omega), omega > 0
        bool hasRealPole = false;                         // at s = -1 if true
    };

    inline ButterPrototype butterworthAnalogPrototype(size_t order) {
        ButterPrototype proto;
        const size_t nPairs = order / 2;
        proto.cplxPairs.reserve(nPairs);
        for (size_t k = 0; k < nPairs; ++k) {
            const double theta = IIR_PI * (2.0 * static_cast<double>(k) + 1.0) /
                                 (2.0 * static_cast<double>(order));
            const double sigma = -std::sin(theta); // < 0 for stability
            const double omega =  std::cos(theta); // > 0 for k < order/2
            proto.cplxPairs.emplace_back(sigma, omega);
        }
        proto.hasRealPole = (order % 2 == 1);
        return proto;
    }

    // Bilinear transform of a single 2nd-order analog section
    //   H(s) = 1 / (s^2 - 2σ s + (σ^2 + ω^2)),  pre-warped by K = tan(π fc / 2)
    // into a digital biquad for LOW-PASS.
    inline BiquadCoeffs bilinearLP_pair(double sigma, double omega, double K) {
        const double K2    = K * K;
        const double polR2 = sigma * sigma + omega * omega; // |p|^2
        // Analog denom: s^2 + (-2σ) s + polR2, after s = (1-z^-1)/(1+z^-1) / K
        // and clearing (1+z^-1)^2:
        //   num_digital   = K^2 * (1 + 2 z^-1 + z^-2)
        //   denom_digital = (1 - 2σK + polR2 K^2) + 2(polR2 K^2 - 1) z^-1
        //                 + (1 + 2σK + polR2 K^2) z^-2
        const double a0 = 1.0 - 2.0 * sigma * K + polR2 * K2;
        const double a1 = 2.0 * (polR2 * K2 - 1.0);
        const double a2 = 1.0 + 2.0 * sigma * K + polR2 * K2;

        const double b0 = K2;
        const double b1 = 2.0 * K2;
        const double b2 = K2;

        return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
    }

    // Bilinear transform for HIGH-PASS analog prototype 1/(s^2 - 2σs + polR2)
    // with LP→HP substitution s → 1/s, then pre-warp and bilinear.
    // Final digital biquad:
    inline BiquadCoeffs bilinearHP_pair(double sigma, double omega, double K) {
        const double K2    = K * K;
        const double polR2 = sigma * sigma + omega * omega;
        // After LP→HP (s → 1/s) the analog denom becomes polR2*s^2 - 2σ s + 1.
        // Applying bilinear s = (1-z^-1)/(K(1+z^-1)):
        //   num_digital   = 1 - 2 z^-1 + z^-2   (numerator of HP is s^2, i.e. (1-z^-1)^2 / K^2,
        //                   but after scaling by K^2 * polR2 we absorb constants)
        // Multiplying numerator and denominator by K^2 (1+z^-1)^2:
        //   denom = (polR2 - 2σK + K^2) + 2(K^2 - polR2) z^-1 + (polR2 + 2σK + K^2) z^-2
        //   num   = 1 - 2 z^-1 + z^-2     (times K^2 which we keep as gain distribution below)
        //
        // Since the analog HP prototype has unity gain at s→∞ (after LP→HP at s→0 → DC=0, s→∞ → 1),
        // we normalize so that the digital HP has gain 1 at Nyquist (z=-1):
        //   H(z=-1) = (1 - 2(-1) + 1) / denom(z=-1)
        //           = 4 / ((polR2 - 2σK + K^2) - 2(K^2 - polR2) + (polR2 + 2σK + K^2))
        //           = 4 / (4 polR2) = 1/polR2
        // So we need to multiply numerator by polR2 to get unity gain at Nyquist.
        const double a0 = polR2 - 2.0 * sigma * K + K2;
        const double a1 = 2.0 * (K2 - polR2);
        const double a2 = polR2 + 2.0 * sigma * K + K2;

        const double b0 =  polR2;
        const double b1 = -2.0 * polR2;
        const double b2 =  polR2;

        return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
    }

    // First-order LP section from analog pole at s = -1, pre-warped.
    inline BiquadCoeffs bilinearLP_real(double K) {
        // H(s) = 1/(s+1) → s = (1-z^-1)/(K(1+z^-1))
        // digital: (K + K z^-1) / ((1+K) + (K-1) z^-1)
        const double a0 = 1.0 + K;
        const double a1 = K - 1.0;
        const double b0 = K;
        const double b1 = K;
        return { b0 / a0, b1 / a0, 0.0, a1 / a0, 0.0 };
    }

    // First-order HP section from analog pole at s = -1 after LP→HP (pole still at s = -1).
    // H(s) = s/(s+1) → digital:
    inline BiquadCoeffs bilinearHP_real(double K) {
        const double a0 =  1.0 + K;
        const double a1 =  K - 1.0;
        const double b0 =  1.0;
        const double b1 = -1.0;
        return { b0 / a0, b1 / a0, 0.0, a1 / a0, 0.0 };
    }
} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth low-pass design.
// `cutoff` is normalized frequency in (0, 1), where 1 is Nyquist.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<BiquadCoeffs> designButterworthLowPass(
    size_t order,
    double cutoff)
{
    if (cutoff <= 0.0 || cutoff >= 1.0)
        throw std::invalid_argument("designButterworthLowPass: cutoff must be in (0, 1)");
    if (order == 0) order = 1;

    const double K = std::tan(detail::IIR_PI * cutoff * 0.5); // pre-warp
    const auto proto = detail::butterworthAnalogPrototype(order);

    std::vector<BiquadCoeffs> sections;
    sections.reserve(order / 2 + (order % 2));

    for (const auto& p : proto.cplxPairs) {
        sections.push_back(detail::bilinearLP_pair(p.first, p.second, K));
    }
    if (proto.hasRealPole) {
        sections.push_back(detail::bilinearLP_real(K));
    }
    return sections;
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth high-pass design.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<BiquadCoeffs> designButterworthHighPass(
    size_t order,
    double cutoff)
{
    if (cutoff <= 0.0 || cutoff >= 1.0)
        throw std::invalid_argument("designButterworthHighPass: cutoff must be in (0, 1)");
    if (order == 0) order = 1;

    const double K = std::tan(detail::IIR_PI * cutoff * 0.5);
    const auto proto = detail::butterworthAnalogPrototype(order);

    std::vector<BiquadCoeffs> sections;
    sections.reserve(order / 2 + (order % 2));

    for (const auto& p : proto.cplxPairs) {
        sections.push_back(detail::bilinearHP_pair(p.first, p.second, K));
    }
    if (proto.hasRealPole) {
        sections.push_back(detail::bilinearHP_real(K));
    }
    return sections;
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth band-pass design via LP→BP frequency transformation on poles.
// Each analog LP pole p₀ transforms to two poles that are roots of
//   p^2 - (BW * p₀) p + ω₀² = 0,
// where ω₀ = sqrt(ωL*ωH) (in pre-warped analog domain) and BW = ωH - ωL.
// Then each resulting complex-conjugate pole pair is bilinear-transformed
// to one digital biquad. Resulting filter has order = 2 * `order`.
// `cutoffLow`, `cutoffHigh` are normalized in (0, 1).
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<BiquadCoeffs> designButterworthBandPass(
    size_t order,
    double cutoffLow,
    double cutoffHigh)
{
    if (cutoffLow <= 0.0 || cutoffHigh >= 1.0 || cutoffLow >= cutoffHigh)
        throw std::invalid_argument(
            "designButterworthBandPass: require 0 < cutoffLow < cutoffHigh < 1");
    if (order == 0) order = 1;

    // Pre-warped analog edge frequencies.
    const double wL = std::tan(detail::IIR_PI * cutoffLow  * 0.5);
    const double wH = std::tan(detail::IIR_PI * cutoffHigh * 0.5);
    const double w0 = std::sqrt(wL * wH);     // analog center
    const double BW = wH - wL;                // analog bandwidth
    const double w0sq = w0 * w0;

    const auto proto = detail::butterworthAnalogPrototype(order);
    std::vector<BiquadCoeffs> sections;
    sections.reserve(order);

    // For each analog LP pole p₀ (one from each complex pair — we take both),
    // solve p^2 - BW*p₀*p + w0^2 = 0 → p = (BW*p₀ ± sqrt((BW*p₀)^2 - 4 w0^2)) / 2.
    auto transformPoleBP = [&](std::complex<double> p0,
                               std::complex<double>& pa,
                               std::complex<double>& pb) {
        const std::complex<double> bp = BW * p0;
        const std::complex<double> disc = bp * bp - 4.0 * w0sq;
        const std::complex<double> sq = std::sqrt(disc);
        pa = 0.5 * (bp + sq);
        pb = 0.5 * (bp - sq);
    };

    // Turn a single analog complex pole `p` (with its conjugate implicit)
    // into a digital biquad via bilinear s = (1 - z^-1)/(1 + z^-1).
    // BP has two zeros at s=0 and two zeros at s=∞ per pole-pair, yielding
    // digital zeros at z=1 and z=-1 (numerator = 1 - z^-2 per biquad, scaled).
    auto bilinearBPBiquad = [&](std::complex<double> pole) {
        // Conjugate-pair denom in s-domain: (s - p)(s - p*) = s^2 - 2 Re(p) s + |p|^2.
        const double sigma = pole.real();
        const double r2    = std::norm(pole); // |p|^2
        // Bilinear s → (1 - z^-1)/(1 + z^-1). Multiply num/denom by (1+z^-1)^2.
        //   denom_s = s^2 - 2σ s + r2
        //   → (1-z^-1)^2 - 2σ(1-z^-1)(1+z^-1) + r2(1+z^-1)^2
        //   = (1 - 2σ + r2) + 2(r2 - 1) z^-1 + (1 + 2σ + r2) z^-2
        // BP numerator per pole-pair is proportional to s (not s^2), so
        //   num_s ∝ (1 - z^-2). Normalization gain chosen so that peak is 1.
        const double a0 = 1.0 - 2.0 * sigma + r2;
        const double a1 = 2.0 * (r2 - 1.0);
        const double a2 = 1.0 + 2.0 * sigma + r2;

        // For unity gain at digital center frequency ω_c (z = e^{jω_c}),
        // with ω_c corresponding to analog w0: ω_c = 2*atan(w0).
        // Numerator 1 - z^-2 has magnitude |1 - e^{-j2ω}| = 2|sin ω|.
        // Denominator at z = e^{jω_c} is evaluated below for normalization.
        const double wc_digital = 2.0 * std::atan(w0);
        const double cw = std::cos(wc_digital);
        const double sw = std::sin(wc_digital);
        // num(e^{jωc}) = 1 - e^{-j2ωc}
        const std::complex<double> num_c(1.0 - std::cos(2.0 * wc_digital),
                                         std::sin(2.0 * wc_digital));
        // denom(e^{jωc}) = a0 + a1 e^{-jωc} + a2 e^{-j2ωc}, with a0-normalized later.
        const std::complex<double> den_c =
            std::complex<double>(a0, 0.0)
            + std::complex<double>(a1 * cw, -a1 * sw)
            + std::complex<double>(a2 * std::cos(2.0 * wc_digital),
                                  -a2 * std::sin(2.0 * wc_digital));
        const double gain = std::abs(den_c) / std::abs(num_c);

        const double b0 =  gain;
        const double b1 =  0.0;
        const double b2 = -gain;

        return BiquadCoeffs{ b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
    };

    for (const auto& pp : proto.cplxPairs) {
        // Analog pole p0 (we take the one with positive ω; its conjugate is handled
        // by producing a real biquad with the conjugate pole pair at the end).
        const std::complex<double> p0(pp.first, pp.second);
        std::complex<double> pa, pb;
        transformPoleBP(p0, pa, pb);
        // Each of pa, pb brings its own conjugate, giving two biquads.
        sections.push_back(bilinearBPBiquad(pa));
        sections.push_back(bilinearBPBiquad(pb));
    }
    if (proto.hasRealPole) {
        // Real pole at s = -1 becomes a complex-conjugate pair:
        //   p^2 + BW p + w0^2 = 0 → one biquad.
        const std::complex<double> p0(-1.0, 0.0);
        std::complex<double> pa, pb;
        transformPoleBP(p0, pa, pb);
        sections.push_back(bilinearBPBiquad(pa)); // pa and pb are conjugates here
    }
    return sections;
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth band-stop (notch) design via LP→BS frequency transformation.
// Analog LP pole p₀ maps to roots of  p^2 - (BW/p₀) p + ω₀² = 0  (after the
// substitution s → BW·s / (s² + ω₀²) for the prototype with cutoff 1).
// Equivalently: p^2 * p₀ - BW p + ω₀² p₀ = 0.
// Each pole-pair becomes two biquads with zeros at s = ± j ω₀ (i.e. at the
// notch frequency).
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<BiquadCoeffs> designButterworthBandStop(
    size_t order,
    double cutoffLow,
    double cutoffHigh)
{
    if (cutoffLow <= 0.0 || cutoffHigh >= 1.0 || cutoffLow >= cutoffHigh)
        throw std::invalid_argument(
            "designButterworthBandStop: require 0 < cutoffLow < cutoffHigh < 1");
    if (order == 0) order = 1;

    const double wL = std::tan(detail::IIR_PI * cutoffLow  * 0.5);
    const double wH = std::tan(detail::IIR_PI * cutoffHigh * 0.5);
    const double w0 = std::sqrt(wL * wH);
    const double BW = wH - wL;
    const double w0sq = w0 * w0;

    const auto proto = detail::butterworthAnalogPrototype(order);
    std::vector<BiquadCoeffs> sections;
    sections.reserve(order);

    // LP→BS on a single analog pole p₀: new poles satisfy
    //   p^2 - (BW/p₀) p + ω₀² = 0
    auto transformPoleBS = [&](std::complex<double> p0,
                               std::complex<double>& pa,
                               std::complex<double>& pb) {
        const std::complex<double> bp = BW / p0;
        const std::complex<double> disc = bp * bp - 4.0 * w0sq;
        const std::complex<double> sq = std::sqrt(disc);
        pa = 0.5 * (bp + sq);
        pb = 0.5 * (bp - sq);
    };

    // Digital biquad: analog zeros at ±jω₀ and the given complex-pole pair.
    //   denom_s = s^2 - 2σ s + r2
    //   num_s   = s^2 + ω₀²
    // Bilinear s = (1 - z^-1)/(1 + z^-1):
    //   num_digital = (1 - z^-1)^2 + ω₀²(1 + z^-1)^2
    //               = (1 + ω₀²) - 2(1 - ω₀²) z^-1 + (1 + ω₀²) z^-2
    auto bilinearBSBiquad = [&](std::complex<double> pole) {
        const double sigma = pole.real();
        const double r2    = std::norm(pole);

        const double a0 = 1.0 - 2.0 * sigma + r2;
        const double a1 = 2.0 * (r2 - 1.0);
        const double a2 = 1.0 + 2.0 * sigma + r2;

        const double nb0 = 1.0 + w0sq;
        const double nb1 = -2.0 * (1.0 - w0sq);
        const double nb2 = 1.0 + w0sq;

        // Normalize for unity gain at DC (z = 1):
        //   num(1) = nb0 + nb1 + nb2 = 4 ω₀²
        //   den(1) = a0 + a1 + a2   = 4 r2 (with a0-unnormalized)
        // For a band-stop we want unity gain at DC AND at Nyquist.
        // At Nyquist (z = -1): num(-1) = nb0 - nb1 + nb2 = 4; den(-1) = 4.
        // With our a0 normalization we divide everything by a0; DC gain becomes
        // 4 ω₀² / (4 r2 / a0 ... ) — instead we pick gain so H(1) = 1.
        const double gain_dc_num = nb0 + nb1 + nb2;           // = 4 ω₀²
        const double gain_dc_den = (a0 + a1 + a2);            // = 4 r2
        const double gain = gain_dc_den / gain_dc_num;

        const double b0 = gain * nb0;
        const double b1 = gain * nb1;
        const double b2 = gain * nb2;

        return BiquadCoeffs{ b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
    };

    for (const auto& pp : proto.cplxPairs) {
        const std::complex<double> p0(pp.first, pp.second);
        std::complex<double> pa, pb;
        transformPoleBS(p0, pa, pb);
        sections.push_back(bilinearBSBiquad(pa));
        sections.push_back(bilinearBSBiquad(pb));
    }
    if (proto.hasRealPole) {
        const std::complex<double> p0(-1.0, 0.0);
        std::complex<double> pa, pb;
        transformPoleBS(p0, pa, pb);
        sections.push_back(bilinearBSBiquad(pa));
    }
    return sections;
}

// ─────────────────────────────────────────────────────────────────────────────
// RBJ Audio-EQ-Cookbook biquads.
// `frequency` is normalized to Nyquist, i.e. lies in (0, 1),
// which maps to ω₀ = π · frequency (so frequency=1 is Nyquist).
// ─────────────────────────────────────────────────────────────────────────────
inline BiquadCoeffs designPeakingEQ(
    double centerFreq,
    double gainDB,
    double q = 1.0)
{
    if (centerFreq <= 0.0 || centerFreq >= 1.0)
        throw std::invalid_argument("designPeakingEQ: centerFreq must be in (0, 1)");
    if (q <= 0.0)
        throw std::invalid_argument("designPeakingEQ: q must be > 0");

    const double A      = std::pow(10.0, gainDB / 40.0);
    const double omega0 = detail::IIR_PI * centerFreq;      // normalized to Nyquist
    const double sin_w  = std::sin(omega0);
    const double cos_w  = std::cos(omega0);
    const double alpha  = sin_w / (2.0 * q);

    const double b0 =  1.0 + alpha * A;
    const double b1 = -2.0 * cos_w;
    const double b2 =  1.0 - alpha * A;
    const double a0 =  1.0 + alpha / A;
    const double a1 = -2.0 * cos_w;
    const double a2 =  1.0 - alpha / A;

    return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
}

inline BiquadCoeffs designLowShelf(
    double cutoff,
    double gainDB,
    double q = 0.707)
{
    if (cutoff <= 0.0 || cutoff >= 1.0)
        throw std::invalid_argument("designLowShelf: cutoff must be in (0, 1)");
    if (q <= 0.0)
        throw std::invalid_argument("designLowShelf: q must be > 0");

    const double A      = std::pow(10.0, gainDB / 40.0);
    const double omega0 = detail::IIR_PI * cutoff;
    const double sin_w  = std::sin(omega0);
    const double cos_w  = std::cos(omega0);
    const double alpha  = sin_w / (2.0 * q);
    const double sqA    = std::sqrt(A);

    const double b0 =       A * ((A + 1.0) - (A - 1.0) * cos_w + 2.0 * sqA * alpha);
    const double b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w);
    const double b2 =       A * ((A + 1.0) - (A - 1.0) * cos_w - 2.0 * sqA * alpha);
    const double a0 =            (A + 1.0) + (A - 1.0) * cos_w + 2.0 * sqA * alpha;
    const double a1 =    -2.0 * ((A - 1.0) + (A + 1.0) * cos_w);
    const double a2 =            (A + 1.0) + (A - 1.0) * cos_w - 2.0 * sqA * alpha;

    return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
}

inline BiquadCoeffs designHighShelf(
    double cutoff,
    double gainDB,
    double q = 0.707)
{
    if (cutoff <= 0.0 || cutoff >= 1.0)
        throw std::invalid_argument("designHighShelf: cutoff must be in (0, 1)");
    if (q <= 0.0)
        throw std::invalid_argument("designHighShelf: q must be > 0");

    const double A      = std::pow(10.0, gainDB / 40.0);
    const double omega0 = detail::IIR_PI * cutoff;
    const double sin_w  = std::sin(omega0);
    const double cos_w  = std::cos(omega0);
    const double alpha  = sin_w / (2.0 * q);
    const double sqA    = std::sqrt(A);

    const double b0 =        A * ((A + 1.0) + (A - 1.0) * cos_w + 2.0 * sqA * alpha);
    const double b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w);
    const double b2 =        A * ((A + 1.0) + (A - 1.0) * cos_w - 2.0 * sqA * alpha);
    const double a0 =             (A + 1.0) - (A - 1.0) * cos_w + 2.0 * sqA * alpha;
    const double a1 =      2.0 * ((A - 1.0) - (A + 1.0) * cos_w);
    const double a2 =             (A + 1.0) - (A - 1.0) * cos_w - 2.0 * sqA * alpha;

    return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
}

// ─────────────────────────────────────────────────────────────────────────────
// BiquadCascade — as before, with per-section state.
// ─────────────────────────────────────────────────────────────────────────────
class BiquadCascade {
public:
    BiquadCascade() = default;
    explicit BiquadCascade(const std::vector<BiquadCoeffs>& sections)
        : sections_(sections), states_(sections.size()) {}

    void setCoefficients(const std::vector<BiquadCoeffs>& sections) {
        sections_ = sections;
        states_.assign(sections.size(), State{});
    }

    inline double process(double x) {
        double y = x;
        for (size_t i = 0; i < sections_.size(); ++i) {
            y = sections_[i].process(y, states_[i].s1, states_[i].s2);
        }
        return y;
    }

    void process(std::vector<double>& signal) {
        for (auto& x : signal) x = process(x);
    }

    std::vector<double> process(const std::vector<double>& signal) {
        std::vector<double> out(signal.size());
        for (size_t i = 0; i < signal.size(); ++i) out[i] = process(signal[i]);
        return out;
    }

    void reset() {
        for (auto& s : states_) { s.s1 = 0.0; s.s2 = 0.0; }
    }

    const std::vector<BiquadCoeffs>& getSections() const { return sections_; }

private:
    struct State { double s1 = 0.0, s2 = 0.0; };
    std::vector<BiquadCoeffs> sections_;
    std::vector<State> states_;
};

// ─────────────────────────────────────────────────────────────────────────────
// One-shot helpers.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> applyIIR(
    const std::vector<double>& signal,
    const std::vector<BiquadCoeffs>& sections)
{
    BiquadCascade filter(sections);
    return filter.process(signal);
}

// Zero-phase forward-backward filtering.
// Note: simple version without edge-reflection padding. For best edge behavior
// pad the signal by a few filter lengths before calling.
inline std::vector<double> filtfiltIIR(
    const std::vector<double>& signal,
    const std::vector<BiquadCoeffs>& sections)
{
    if (signal.empty()) return signal;

    auto y = applyIIR(signal, sections);
    std::reverse(y.begin(), y.end());
    y = applyIIR(y, sections);
    std::reverse(y.begin(), y.end());
    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Frequency-response helpers.
// `sampleRate` is only used by frequencyAxis(); the filter itself is already
// defined on the normalized frequency grid (0 … π).
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<std::complex<double>> frequencyResponseIIR(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft)
{
    if (nfft < 2) nfft = 2;
    const size_t nBins = nfft / 2 + 1;

    std::vector<std::complex<double>> response(nBins);
    for (size_t k = 0; k < nBins; ++k) {
        const double omega = detail::IIR_PI * static_cast<double>(k) /
                             static_cast<double>(nfft / 2);
        const std::complex<double> z_inv = std::exp(std::complex<double>(0.0, -omega));
        const std::complex<double> z_inv2 = z_inv * z_inv;

        std::complex<double> num{1.0, 0.0};
        std::complex<double> den{1.0, 0.0};
        for (const auto& sec : sections) {
            num *= (sec.b0 + sec.b1 * z_inv + sec.b2 * z_inv2);
            den *= (1.0   + sec.a1 * z_inv + sec.a2 * z_inv2);
        }
        response[k] = num / den;
    }
    return response;
}

inline std::vector<double> magnitudeResponseIIR(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft)
{
    const auto resp = frequencyResponseIIR(sections, nfft);
    std::vector<double> mag(resp.size());
    std::transform(resp.begin(), resp.end(), mag.begin(),
                   [](const std::complex<double>& c) { return std::abs(c); });
    return mag;
}

inline std::vector<double> magnitudeResponseDB(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft)
{
    auto mag = magnitudeResponseIIR(sections, nfft);
    for (auto& m : mag) m = 20.0 * std::log10(std::max(m, 1e-12));
    return mag;
}

inline std::vector<double> frequencyAxis(size_t nfft, double sampleRate = 1.0) {
    if (nfft < 2) nfft = 2;
    const size_t nBins = nfft / 2 + 1;
    std::vector<double> f(nBins);
    const double step = sampleRate / (2.0 * static_cast<double>(nfft / 2));
    for (size_t i = 0; i < nBins; ++i) {
        f[i] = static_cast<double>(i) * step;
    }
    return f;
}

} // namespace SharedMath::DSP