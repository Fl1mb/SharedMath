#pragma once

#include <complex>
#include <vector>
#include <cstddef>

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

// ─────────────────────────────────────────────────────────────────────────────
// Normalization
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> normalizeTapsEnergy(const std::vector<double>& taps);

// ─────────────────────────────────────────────────────────────────────────────
// Pulse shape tap generators
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> rectangularPulse(size_t samplesPerSymbol);

std::vector<double> raisedCosineTaps(
    size_t samplesPerSymbol,
    size_t spanSymbols,
    double rolloff);

std::vector<double> rootRaisedCosineTaps(
    size_t samplesPerSymbol,
    size_t spanSymbols,
    double rolloff);

std::vector<double> gaussianPulseTaps(
    size_t samplesPerSymbol,
    size_t spanSymbols,
    double bt);

// ─────────────────────────────────────────────────────────────────────────────
// Pulse shaping application
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::complex<double>> upsampleSymbols(
    const std::vector<std::complex<double>>& symbols,
    size_t samplesPerSymbol);

std::vector<std::complex<double>> pulseShape(
    const std::vector<std::complex<double>>& symbols,
    const std::vector<double>& taps,
    size_t samplesPerSymbol);

} // namespace SharedMath::DSP