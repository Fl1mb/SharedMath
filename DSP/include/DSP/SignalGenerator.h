#pragma once

// SharedMath::DSP — Test signal generators
//
// sineWave    — A·sin(2π·f·t + φ)
// chirp       — linear frequency sweep from f0 to f1
// whiteNoise  — uniform white noise, reproducible via seed
// impulse     — unit impulse at a given sample position
// stepSignal  — unit step starting at a given sample position
// squareWave  — square wave with configurable duty cycle

#include <vector>
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
// sineWave — A·sin(2π·f·(n/sampleRate) + phaseRad)
//
// freq:       frequency in Hz
// sampleRate: samples per second
// numSamples: output length
// amplitude:  peak amplitude (default 1.0)
// phaseRad:   initial phase in radians (default 0.0)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> sineWave(
    double freq,
    double sampleRate,
    size_t numSamples,
    double amplitude = 1.0,
    double phaseRad  = 0.0)
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
// chirp — linear frequency sweep
//
// Instantaneous frequency increases linearly from f0 at t=0 to f1 at t=T.
// Phase: φ(t) = 2π·(f0·t + (f1−f0)/(2T)·t²)
//
// f0, f1:     start/end frequencies in Hz
// sampleRate: samples per second
// numSamples: output length
// amplitude:  peak amplitude (default 1.0)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> chirp(
    double f0,
    double f1,
    double sampleRate,
    size_t numSamples,
    double amplitude = 1.0)
{
    if (numSamples == 0) return {};
    if (sampleRate <= 0.0)
        throw std::invalid_argument("chirp: sampleRate must be > 0");

    double T = static_cast<double>(numSamples - 1) / sampleRate;
    double k = (T > 0.0) ? (f1 - f0) / T : 0.0;   // Hz/s chirp rate

    std::vector<double> out(numSamples);
    for (size_t i = 0; i < numSamples; ++i) {
        double t = static_cast<double>(i) / sampleRate;
        out[i] = amplitude *
                 std::cos(2.0 * detail::GEN_PI * (f0 * t + 0.5 * k * t * t));
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// whiteNoise — uniform white noise in [−amplitude, +amplitude]
//
// Deterministic: the same seed always produces the same sequence.
// Uses std::mt19937_64 (64-bit Mersenne Twister).
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> whiteNoise(
    size_t numSamples,
    double amplitude = 1.0,
    uint64_t seed    = 0)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-amplitude, amplitude);
    std::vector<double> out(numSamples);
    for (auto& v : out) v = dist(rng);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// impulse — unit impulse (Kronecker delta)
//
// All samples are 0 except at 'position', where the value is 'amplitude'.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> impulse(
    size_t numSamples,
    size_t position  = 0,
    double amplitude = 1.0)
{
    std::vector<double> out(numSamples, 0.0);
    if (position < numSamples) out[position] = amplitude;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// stepSignal — Heaviside step function
//
// Samples before 'position' are 0; at and after 'position' they equal 'amplitude'.
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> stepSignal(
    size_t numSamples,
    size_t position  = 0,
    double amplitude = 1.0)
{
    std::vector<double> out(numSamples, 0.0);
    for (size_t i = position; i < numSamples; ++i)
        out[i] = amplitude;
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// squareWave
//
// Generates a square wave with the given duty cycle.
// Each period of length sampleRate/freq samples is split:
//   first (dutyCycle × period) samples → +amplitude
//   remaining samples                  → −amplitude
//
// freq:      frequency in Hz
// sampleRate: samples per second
// numSamples: output length
// amplitude:  peak amplitude (default 1.0)
// dutyCycle:  fraction of period at high level ∈ (0, 1) (default 0.5)
// ─────────────────────────────────────────────────────────────────────────────
inline std::vector<double> squareWave(
    double freq,
    double sampleRate,
    size_t numSamples,
    double amplitude = 1.0,
    double dutyCycle = 0.5)
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
