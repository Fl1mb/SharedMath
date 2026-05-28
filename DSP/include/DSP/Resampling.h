#pragma once

/// SharedMath::DSP — Polyphase resampling
///
/// upfirdn(signal, h, L, M)
///   Core primitive: upsample by L, FIR filter h, downsample by M.
///   If h is empty an identity (single 1.0 tap) is used.
///   Output length = ceil((N*L + P - 1) / M),  P = len(h).
///
/// interpolate(signal, L, h = {})
///   Upsample by integer factor L. Uses a Kaiser low-pass if h is empty.
///   FIR gain is scaled by L so amplitude is preserved.
///
/// decimate(signal, M, h = {})
///   Downsample by integer factor M. Uses a Kaiser low-pass if h is empty.
///
/// resamplePolyphase(signal, L, M, h = {})
///   Rational L/M resampling built on upfirdn. h defaults to a
///   Kaiser low-pass at fc = 1/max(L,M).
///
/// resampleTo(signal, inputRate, outputRate, h = {})
///   Convenience wrapper that reduces the rational ratio inputRate→outputRate,
///   applies anti-alias filtering, compensates the FIR group delay, and returns
///   roughly round(N * outputRate / inputRate) samples.

#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// upfirdn — upsample by L, filter, downsample by M
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> upfirdn(
    const std::vector<double>& signal,
    const std::vector<double>& h,
    size_t L,
    size_t M);

// ─────────────────────────────────────────────────────────────────────────────
// interpolate — upsample by integer factor L
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> interpolate(
    const std::vector<double>& signal,
    size_t L,
    const std::vector<double>& h = {});

// ─────────────────────────────────────────────────────────────────────────────
// decimate — downsample by integer factor M (with anti-alias filter)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> decimate(
    const std::vector<double>& signal,
    size_t M,
    const std::vector<double>& h = {});

// ─────────────────────────────────────────────────────────────────────────────
// resamplePolyphase — rational L/M resampling
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> resamplePolyphase(
    const std::vector<double>& signal,
    size_t L,
    size_t M,
    const std::vector<double>& h = {});

// ─────────────────────────────────────────────────────────────────────────────
// resamplePolyphaseAligned — rational resampling with FIR delay compensation
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> resamplePolyphaseAligned(
    const std::vector<double>& signal,
    size_t L,
    size_t M,
    const std::vector<double>& h = {});

// ─────────────────────────────────────────────────────────────────────────────
// resampleTo — sample-rate based convenience wrapper
// ─────────────────────────────────────────────────────────────────────────────
std::vector<double> resampleTo(
    const std::vector<double>& signal,
    size_t inputRate,
    size_t outputRate,
    const std::vector<double>& h = {});

} // namespace SharedMath::DSP