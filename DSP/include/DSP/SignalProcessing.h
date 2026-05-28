#pragma once

/// SharedMath::DSP — Practical signal-processing helpers
///
/// Statistics:
///   mean, rms, peakAbs
///
/// DC / normalization:
///   removeDC, normalizePeak, normalizeRMS
///
/// Smoothing / filtering:
///   movingAverage, exponentialSmoothing, medianFilter
///
/// Trend removal:
///   detrendLinear

#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// Statistics
/// ─────────────────────────────────────────────────────────────────────────────

/// Arithmetic mean.  Returns 0 for empty signal.
double mean(const std::vector<double>& x);

/// Root-mean-square.  Returns 0 for empty signal.
double rms(const std::vector<double>& x);

/// Maximum absolute value.  Returns 0 for empty signal.
double peakAbs(const std::vector<double>& x);

/// ─────────────────────────────────────────────────────────────────────────────
/// DC / normalization
/// ─────────────────────────────────────────────────────────────────────────────

/// Subtract the signal mean (remove DC offset).
std::vector<double> removeDC(const std::vector<double>& x);

// Scale so that max(|x|) == targetPeak.
// Throws if the signal is all-zero or empty.
std::vector<double> normalizePeak(
    const std::vector<double>& x,
    double targetPeak = 1.0);

// Scale so that rms(x) == targetRMS.
// Throws if the signal is all-zero or empty.
std::vector<double> normalizeRMS(
    const std::vector<double>& x,
    double targetRMS = 1.0);

// ─────────────────────────────────────────────────────────────────────────────
// Smoothing
// ─────────────────────────────────────────────────────────────────────────────

// Centred moving average with edge replication.
// y[i] = mean(x[max(0, i-r) .. min(n-1, i+r)]),  r = windowSize / 2.
// windowSize == 0 or > signal length: throws std::invalid_argument.
std::vector<double> movingAverage(
    const std::vector<double>& x,
    size_t windowSize);

// Exponential (single-pole IIR) smoothing:  y[n] = α·x[n] + (1-α)·y[n-1]
// alpha ∈ (0, 1]:  α→0 heavy smoothing, α=1 passthrough.
std::vector<double> exponentialSmoothing(
    const std::vector<double>& x,
    double alpha);

// Running-median filter with edge replication.
// kernelSize is forced to the next odd number >= kernelSize if it is even.
// Each output sample is the median of the kernelSize nearest input samples
// (centred, edges clamped to the signal boundary).
std::vector<double> medianFilter(
    const std::vector<double>& x,
    size_t kernelSize);

/// ─────────────────────────────────────────────────────────────────────────────
/// Trend removal
/// ─────────────────────────────────────────────────────────────────────────────

/// Subtract the best-fit straight line (least-squares) from the signal.
/// Uses time index t[n] = n (0, 1, ..., N-1) as the independent variable.
/// A signal with a single sample is returned as-is (trend is undefined).
std::vector<double> detrendLinear(const std::vector<double>& x);

} // namespace SharedMath::DSP