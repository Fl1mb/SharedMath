#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace SharedMath::DSP {

enum class PulseShape {
    Rectangular,
    RaisedCosine,
    RootRaisedCosine,
    Gaussian
};

struct PulseShapingParams {
    size_t samplesPerSymbol = 4;
    size_t spanSymbols = 8;
    double rolloff = 0.35;
    double bt = 0.3;
};

namespace detail {
inline constexpr double kPi = 3.141592653589793238462643383279502884;

inline void validatePulseShapeCommon(size_t samplesPerSymbol, size_t spanSymbols)
{
    if (samplesPerSymbol == 0)
        throw std::invalid_argument("PulseShaping: samplesPerSymbol must be > 0");
    if (spanSymbols == 0)
        throw std::invalid_argument("PulseShaping: spanSymbols must be > 0");
}

inline void validateRolloff(double rolloff)
{
    if (rolloff < 0.0 || rolloff > 1.0)
        throw std::invalid_argument("PulseShaping: rolloff must be in [0, 1]");
}
} // namespace detail

inline std::vector<double> normalizeTapsEnergy(const std::vector<double>& taps)
{
    double energy = 0.0;
    for (double v : taps) energy += v * v;
    if (energy <= 0.0) return taps;

    std::vector<double> out(taps.size());
    const double scale = 1.0 / std::sqrt(energy);
    for (size_t i = 0; i < taps.size(); ++i) out[i] = taps[i] * scale;
    return out;
}

inline std::vector<double> rectangularPulse(size_t samplesPerSymbol)
{
    if (samplesPerSymbol == 0)
        throw std::invalid_argument("rectangularPulse: samplesPerSymbol must be > 0");
    return normalizeTapsEnergy(std::vector<double>(samplesPerSymbol, 1.0));
}

inline std::vector<double> raisedCosineTaps(
    size_t samplesPerSymbol,
    size_t spanSymbols,
    double rolloff)
{
    detail::validatePulseShapeCommon(samplesPerSymbol, spanSymbols);
    detail::validateRolloff(rolloff);

    const size_t nTaps = spanSymbols * samplesPerSymbol + 1;
    const double mid = static_cast<double>(nTaps - 1) * 0.5;
    std::vector<double> h(nTaps);

    for (size_t n = 0; n < nTaps; ++n) {
        const double t = (static_cast<double>(n) - mid) /
                         static_cast<double>(samplesPerSymbol);
        if (std::abs(t) < 1e-12) {
            h[n] = 1.0;
        } else if (rolloff > 0.0 &&
                   std::abs(std::abs(2.0 * rolloff * t) - 1.0) < 1e-10) {
            h[n] = 0.5 * rolloff * std::sin(detail::kPi / (2.0 * rolloff));
        } else {
            const double x = detail::kPi * t;
            const double sinc = std::sin(x) / x;
            const double denom = 1.0 - std::pow(2.0 * rolloff * t, 2.0);
            h[n] = sinc * std::cos(detail::kPi * rolloff * t) / denom;
        }
    }
    return normalizeTapsEnergy(h);
}

inline std::vector<double> rootRaisedCosineTaps(
    size_t samplesPerSymbol,
    size_t spanSymbols,
    double rolloff)
{
    detail::validatePulseShapeCommon(samplesPerSymbol, spanSymbols);
    detail::validateRolloff(rolloff);

    if (rolloff == 0.0)
        return raisedCosineTaps(samplesPerSymbol, spanSymbols, 0.0);

    const size_t nTaps = spanSymbols * samplesPerSymbol + 1;
    const double mid = static_cast<double>(nTaps - 1) * 0.5;
    std::vector<double> h(nTaps);

    for (size_t n = 0; n < nTaps; ++n) {
        const double t = (static_cast<double>(n) - mid) /
                         static_cast<double>(samplesPerSymbol);
        if (std::abs(t) < 1e-12) {
            h[n] = 1.0 + rolloff * (4.0 / detail::kPi - 1.0);
        } else if (std::abs(std::abs(t) - 1.0 / (4.0 * rolloff)) < 1e-10) {
            const double a = detail::kPi / (4.0 * rolloff);
            h[n] = (rolloff / std::sqrt(2.0)) *
                   ((1.0 + 2.0 / detail::kPi) * std::sin(a) +
                    (1.0 - 2.0 / detail::kPi) * std::cos(a));
        } else {
            const double num =
                std::sin(detail::kPi * t * (1.0 - rolloff)) +
                4.0 * rolloff * t *
                    std::cos(detail::kPi * t * (1.0 + rolloff));
            const double den =
                detail::kPi * t * (1.0 - std::pow(4.0 * rolloff * t, 2.0));
            h[n] = num / den;
        }
    }
    return normalizeTapsEnergy(h);
}

inline std::vector<double> gaussianPulseTaps(
    size_t samplesPerSymbol,
    size_t spanSymbols,
    double bt)
{
    detail::validatePulseShapeCommon(samplesPerSymbol, spanSymbols);
    if (bt <= 0.0)
        throw std::invalid_argument("gaussianPulseTaps: bt must be > 0");

    const size_t nTaps = spanSymbols * samplesPerSymbol + 1;
    const double mid = static_cast<double>(nTaps - 1) * 0.5;
    const double alpha = std::sqrt(std::log(2.0)) / bt;
    std::vector<double> h(nTaps);

    for (size_t n = 0; n < nTaps; ++n) {
        const double t = (static_cast<double>(n) - mid) /
                         static_cast<double>(samplesPerSymbol);
        h[n] = std::exp(-2.0 * std::pow(detail::kPi * t / alpha, 2.0));
    }
    return normalizeTapsEnergy(h);
}

inline std::vector<std::complex<double>> upsampleSymbols(
    const std::vector<std::complex<double>>& symbols,
    size_t samplesPerSymbol)
{
    if (samplesPerSymbol == 0)
        throw std::invalid_argument("upsampleSymbols: samplesPerSymbol must be > 0");
    if (symbols.empty()) return {};

    std::vector<std::complex<double>> out(symbols.size() * samplesPerSymbol,
                                          {0.0, 0.0});
    for (size_t i = 0; i < symbols.size(); ++i)
        out[i * samplesPerSymbol] = symbols[i];
    return out;
}

inline std::vector<std::complex<double>> pulseShape(
    const std::vector<std::complex<double>>& symbols,
    const std::vector<double>& taps,
    size_t samplesPerSymbol)
{
    if (samplesPerSymbol == 0)
        throw std::invalid_argument("pulseShape: samplesPerSymbol must be > 0");
    if (symbols.empty() || taps.empty()) return {};

    auto up = upsampleSymbols(symbols, samplesPerSymbol);
    std::vector<std::complex<double>> y(up.size() + taps.size() - 1, {0.0, 0.0});
    for (size_t n = 0; n < up.size(); ++n)
        for (size_t k = 0; k < taps.size(); ++k)
            y[n + k] += up[n] * taps[k];
    return y;
}

} // namespace SharedMath::DSP
