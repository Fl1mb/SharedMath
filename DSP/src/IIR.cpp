/**
 * @file IIR.cpp
 * @brief Implementation of IIR filter design and processing functions.
 */

#include "IIR.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>
#include <complex>

namespace SharedMath::DSP {

namespace detail {
    constexpr double IIR_PI = 3.14159265358979323846;

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

    inline BiquadCoeffs bilinearLP_pair(double sigma, double omega, double K) {
        const double K2    = K * K;
        const double polR2 = sigma * sigma + omega * omega;
        const double a0 = 1.0 - 2.0 * sigma * K + polR2 * K2;
        const double a1 = 2.0 * (polR2 * K2 - 1.0);
        const double a2 = 1.0 + 2.0 * sigma * K + polR2 * K2;
        const double b0 = K2;
        const double b1 = 2.0 * K2;
        const double b2 = K2;
        return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
    }

    inline BiquadCoeffs bilinearHP_pair(double sigma, double omega, double K) {
        const double K2    = K * K;
        const double polR2 = sigma * sigma + omega * omega;
        const double a0 = polR2 - 2.0 * sigma * K + K2;
        const double a1 = 2.0 * (K2 - polR2);
        const double a2 = polR2 + 2.0 * sigma * K + K2;
        const double b0 =  polR2;
        const double b1 = -2.0 * polR2;
        const double b2 =  polR2;
        return { b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0 };
    }

    inline BiquadCoeffs bilinearLP_real(double K) {
        const double a0 = 1.0 + K;
        const double a1 = K - 1.0;
        const double b0 = K;
        const double b1 = K;
        return { b0 / a0, b1 / a0, 0.0, a1 / a0, 0.0 };
    }

    inline BiquadCoeffs bilinearHP_real(double K) {
        const double a0 =  1.0 + K;
        const double a1 =  K - 1.0;
        const double b0 =  1.0;
        const double b1 = -1.0;
        return { b0 / a0, b1 / a0, 0.0, a1 / a0, 0.0 };
    }
} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// BiquadCoeffs
// ─────────────────────────────────────────────────────────────────────────────
bool BiquadCoeffs::isStable() const {
    const double disc = a1 * a1 - 4.0 * a2;
    if (disc >= 0.0) {
        const double s = std::sqrt(disc);
        const double p1 = (-a1 + s) * 0.5;
        const double p2 = (-a1 - s) * 0.5;
        return std::abs(p1) < 1.0 && std::abs(p2) < 1.0;
    }
    return a2 < 1.0 && a2 >= 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth low-pass design.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BiquadCoeffs> designButterworthLowPass(
    size_t order,
    double cutoff)
{
    if (cutoff <= 0.0 || cutoff >= 1.0)
        throw std::invalid_argument("designButterworthLowPass: cutoff must be in (0, 1)");
    if (order == 0) order = 1;

    const double K = std::tan(detail::IIR_PI * cutoff * 0.5);
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
std::vector<BiquadCoeffs> designButterworthHighPass(
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
// Butterworth band-pass design.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BiquadCoeffs> designButterworthBandPass(
    size_t order,
    double cutoffLow,
    double cutoffHigh)
{
    if (cutoffLow <= 0.0 || cutoffHigh >= 1.0 || cutoffLow >= cutoffHigh)
        throw std::invalid_argument(
            "designButterworthBandPass: require 0 < cutoffLow < cutoffHigh < 1");
    if (order == 0) order = 1;

    const double wL = std::tan(detail::IIR_PI * cutoffLow  * 0.5);
    const double wH = std::tan(detail::IIR_PI * cutoffHigh * 0.5);
    const double w0 = std::sqrt(wL * wH);
    const double BW = wH - wL;
    const double w0sq = w0 * w0;

    const auto proto = detail::butterworthAnalogPrototype(order);
    std::vector<BiquadCoeffs> sections;
    sections.reserve(order);

    auto transformPoleBP = [&](std::complex<double> p0,
                               std::complex<double>& pa,
                               std::complex<double>& pb) {
        const std::complex<double> bp = BW * p0;
        const std::complex<double> disc = bp * bp - 4.0 * w0sq;
        const std::complex<double> sq = std::sqrt(disc);
        pa = 0.5 * (bp + sq);
        pb = 0.5 * (bp - sq);
    };

    auto bilinearBPBiquad = [&](std::complex<double> pole) {
        const double sigma = pole.real();
        const double r2    = std::norm(pole);
        const double a0 = 1.0 - 2.0 * sigma + r2;
        const double a1 = 2.0 * (r2 - 1.0);
        const double a2 = 1.0 + 2.0 * sigma + r2;
        const double wc_digital = 2.0 * std::atan(w0);
        const double cw = std::cos(wc_digital);
        const double sw = std::sin(wc_digital);
        const std::complex<double> num_c(1.0 - std::cos(2.0 * wc_digital),
                                         std::sin(2.0 * wc_digital));
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
        const std::complex<double> p0(pp.first, pp.second);
        std::complex<double> pa, pb;
        transformPoleBP(p0, pa, pb);
        sections.push_back(bilinearBPBiquad(pa));
        sections.push_back(bilinearBPBiquad(pb));
    }
    if (proto.hasRealPole) {
        const std::complex<double> p0(-1.0, 0.0);
        std::complex<double> pa, pb;
        transformPoleBP(p0, pa, pb);
        sections.push_back(bilinearBPBiquad(pa));
    }
    return sections;
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterworth band-stop (notch) design.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BiquadCoeffs> designButterworthBandStop(
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

    auto transformPoleBS = [&](std::complex<double> p0,
                               std::complex<double>& pa,
                               std::complex<double>& pb) {
        const std::complex<double> bp = BW / p0;
        const std::complex<double> disc = bp * bp - 4.0 * w0sq;
        const std::complex<double> sq = std::sqrt(disc);
        pa = 0.5 * (bp + sq);
        pb = 0.5 * (bp - sq);
    };

    auto bilinearBSBiquad = [&](std::complex<double> pole) {
        const double sigma = pole.real();
        const double r2    = std::norm(pole);
        const double a0 = 1.0 - 2.0 * sigma + r2;
        const double a1 = 2.0 * (r2 - 1.0);
        const double a2 = 1.0 + 2.0 * sigma + r2;
        const double nb0 = 1.0 + w0sq;
        const double nb1 = -2.0 * (1.0 - w0sq);
        const double nb2 = 1.0 + w0sq;
        const double gain_dc_num = nb0 + nb1 + nb2;
        const double gain_dc_den = (a0 + a1 + a2);
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
// ─────────────────────────────────────────────────────────────────────────────
BiquadCoeffs designPeakingEQ(double centerFreq, double gainDB, double q)
{
    if (centerFreq <= 0.0 || centerFreq >= 1.0)
        throw std::invalid_argument("designPeakingEQ: centerFreq must be in (0, 1)");
    if (q <= 0.0)
        throw std::invalid_argument("designPeakingEQ: q must be > 0");

    const double A      = std::pow(10.0, gainDB / 40.0);
    const double omega0 = detail::IIR_PI * centerFreq;
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

BiquadCoeffs designLowShelf(double cutoff, double gainDB, double q)
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

BiquadCoeffs designHighShelf(double cutoff, double gainDB, double q)
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
// BiquadCascade
// ─────────────────────────────────────────────────────────────────────────────
BiquadCascade::BiquadCascade(const std::vector<BiquadCoeffs>& sections)
    : sections_(sections), states_(sections.size()) {}

void BiquadCascade::setCoefficients(const std::vector<BiquadCoeffs>& sections) {
    sections_ = sections;
    states_.assign(sections.size(), State{});
}

void BiquadCascade::process(std::vector<double>& signal) {
    for (auto& x : signal) x = process(x);
}

std::vector<double> BiquadCascade::process(const std::vector<double>& signal) {
    std::vector<double> out(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) out[i] = process(signal[i]);
    return out;
}

void BiquadCascade::reset() {
    for (auto& s : states_) { s.s1 = 0.0; s.s2 = 0.0; }
}

// ─────────────────────────────────────────────────────────────────────────────
// One-shot helpers.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> applyIIR(
    const std::vector<double>& signal,
    const std::vector<BiquadCoeffs>& sections)
{
    BiquadCascade filter(sections);
    return filter.process(signal);
}

std::vector<double> filtfiltIIR(
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
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> frequencyResponseIIR(
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

std::vector<double> magnitudeResponseIIR(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft)
{
    const auto resp = frequencyResponseIIR(sections, nfft);
    std::vector<double> mag(resp.size());
    std::transform(resp.begin(), resp.end(), mag.begin(),
                   [](const std::complex<double>& c) { return std::abs(c); });
    return mag;
}

std::vector<double> magnitudeResponseDB(
    const std::vector<BiquadCoeffs>& sections,
    size_t nfft)
{
    auto mag = magnitudeResponseIIR(sections, nfft);
    for (auto& m : mag) m = 20.0 * std::log10(std::max(m, 1e-12));
    return mag;
}

std::vector<double> frequencyAxis(size_t nfft, double sampleRate)
{
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