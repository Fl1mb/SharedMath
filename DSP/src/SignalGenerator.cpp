/**
 * @file TestSignals.cpp
 * @brief Implementation of test signal generators.
 */

#include "SignalGenerator.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace SharedMath::DSP {

namespace detail {
constexpr double GEN_PI = 3.14159265358979323846;
} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// sineWave
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> sineWave(
    double freq,
    double sampleRate,
    size_t numSamples,
    double amplitude,
    double phaseRad)
{
    if (numSamples == 0) return {};
    if (sampleRate <= 0.0)
        throw std::invalid_argument("sineWave: sampleRate must be > 0");

    std::vector<double> out(numSamples);
    double dt = 1.0 / sampleRate;
    for (size_t i = 0; i < numSamples; ++i)
        out[i] = amplitude *
                 std::sin(2.0 * detail::GEN_PI * freq * (static_cast<double>(i) * dt) +
                          phaseRad);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// chirp
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> chirp(
    double f0,
    double f1,
    double sampleRate,
    size_t numSamples,
    double amplitude)
{
    if (numSamples == 0) return {};
    if (sampleRate <= 0.0)
        throw std::invalid_argument("chirp: sampleRate must be > 0");

    double T = static_cast<double>(numSamples - 1) / sampleRate;
    double k = (T > 0.0) ? (f1 - f0) / T : 0.0;

    std::vector<double> out(numSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        double t = static_cast<double>(i) / sampleRate;
        out[i] = amplitude *
                 std::cos(2.0 * detail::GEN_PI * (f0 * t + 0.5 * k * t * t));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// whiteNoise
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> whiteNoise(
    size_t numSamples,
    double amplitude,
    uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-amplitude, amplitude);
    std::vector<double> out(numSamples);
    for (auto& v : out) v = dist(rng);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// impulse
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> impulse(
    size_t numSamples,
    size_t position,
    double amplitude)
{
    std::vector<double> out(numSamples, 0.0);
    if (position < numSamples) out[position] = amplitude;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// stepSignal
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> stepSignal(
    size_t numSamples,
    size_t position,
    double amplitude)
{
    std::vector<double> out(numSamples, 0.0);
    for (size_t i = position; i < numSamples; ++i)
        out[i] = amplitude;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// squareWave
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> squareWave(
    double freq,
    double sampleRate,
    size_t numSamples,
    double amplitude,
    double dutyCycle)
{
    if (numSamples == 0) return {};
    if (sampleRate <= 0.0)
        throw std::invalid_argument("squareWave: sampleRate must be > 0");
    if (freq <= 0.0)
        throw std::invalid_argument("squareWave: freq must be > 0");
    if (dutyCycle <= 0.0 || dutyCycle >= 1.0)
        throw std::invalid_argument("squareWave: dutyCycle must be in (0, 1)");

    double period = sampleRate / freq;
    std::vector<double> out(numSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        double phase = std::fmod(static_cast<double>(i), period) / period;
        out[i] = (phase < dutyCycle) ? amplitude : -amplitude;
    }
    return out;
}

} // namespace SharedMath::DSP