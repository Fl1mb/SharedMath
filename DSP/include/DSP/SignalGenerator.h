#pragma once

/// SharedMath::DSP — Test signal generators
///
/// sineWave    — A·sin(2π·f·t + φ)
/// chirp       — linear frequency sweep from f0 to f1
/// whiteNoise  — uniform white noise, reproducible via seed
/// impulse     — unit impulse at a given sample position
/// stepSignal  — unit step starting at a given sample position
/// squareWave  — square wave with configurable duty cycle

#include <vector>
#include <cstddef>
#include <cstdint>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// sineWave — A·sin(2π·f·(n/sampleRate) + phaseRad)
//
// freq:       frequency in Hz
// sampleRate: samples per second
// numSamples: output length
// amplitude:  peak amplitude (default 1.0)
// phaseRad:   initial phase in radians (default 0.0)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> sineWave(
    double freq,
    double sampleRate,
    size_t numSamples,
    double amplitude = 1.0,
    double phaseRad  = 0.0);

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
std::vector<double> chirp(
    double f0,
    double f1,
    double sampleRate,
    size_t numSamples,
    double amplitude = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// whiteNoise — uniform white noise in [−amplitude, +amplitude]
//
// Deterministic: the same seed always produces the same sequence.
// Uses std::mt19937_64 (64-bit Mersenne Twister).
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> whiteNoise(
    size_t numSamples,
    double amplitude = 1.0,
    uint64_t seed    = 0);

// ─────────────────────────────────────────────────────────────────────────────
// impulse — unit impulse (Kronecker delta)
//
// All samples are 0 except at 'position', where the value is 'amplitude'.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> impulse(
    size_t numSamples,
    size_t position  = 0,
    double amplitude = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// stepSignal — Heaviside step function
//
// Samples before 'position' are 0; at and after 'position' they equal 'amplitude'.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> stepSignal(
    size_t numSamples,
    size_t position  = 0,
    double amplitude = 1.0);

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
std::vector<double> squareWave(
    double freq,
    double sampleRate,
    size_t numSamples,
    double amplitude = 1.0,
    double dutyCycle = 0.5);

} // namespace SharedMath::DSP