#pragma once

/// SharedMath::DSP — Convolution and Correlation
///
/// Free functions for:
///   • Linear convolution  (FFT-based and direct)
///   • Circular (cyclic) convolution
///   • Streaming convolution: Overlap-Add and Overlap-Save
///   • Cross-correlation and autocorrelation (FFT-based)
///   • Normalized variants of correlation
///
/// All functions operate on std::vector<double>.

#include <complex>
#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// Output-length mode  (same semantics as NumPy / SciPy)
/// ─────────────────────────────────────────────────────────────────────────────

enum class ConvolutionMode {
    Full,   // Full linear output: length = len(a) + len(b) − 1
    Same,   // Central part with length = max(len(a), len(b))
    Valid   // Only fully-overlapping part: length = max(la,lb) − min(la,lb) + 1
            // (empty when la == lb for correlate; 1 when la == lb for convolve)
};

// ═════════════════════════════════════════════════════════════════════════════
// LINEAR CONVOLUTION
// ═════════════════════════════════════════════════════════════════════════════

// ── FFT-based linear convolution — O(N log N) ────────────────────────────────
//
// Equivalent to the existing convolve() in FFT.h, but additionally supports
// Same and Valid output modes and is the reference implementation used by
// the streaming methods below.
std::vector<double> convolveLinear(
    const std::vector<double>& a,
    const std::vector<double>& b,
    ConvolutionMode mode = ConvolutionMode::Full);

// ── Direct (time-domain) convolution — O(N·M) ────────────────────────────────
//
// More efficient than FFT-based for very short kernels (M ≲ 16).
// Results are numerically identical to convolveLinear() up to floating-point
// rounding (no FFT round-trip error).
std::vector<double> convolveLinearDirect(
    const std::vector<double>& a,
    const std::vector<double>& b,
    ConvolutionMode mode = ConvolutionMode::Full);

// ── Circular (cyclic) convolution of length n ─────────────────────────────────
//
// circ[k] = Σ_{m=0}^{n-1} a[m] · b[(k − m) mod n]
//
// Both a and b are zero-padded (or folded via modular indexing) to length n.
// When n ≥ la + lb − 1, circular convolution equals linear convolution.
std::vector<double> convolveCircular(
    const std::vector<double>& a,
    const std::vector<double>& b,
    size_t n);

// ── Overlap-Add — streaming linear convolution ────────────────────────────────
//
// Splits `signal` into non-overlapping blocks of `blockSize` samples, convolves
// each block with `kernel` via FFT, and accumulates with appropriate overlap.
//
// Produces the same result as convolveLinear(signal, kernel, Full).
// Advantage over a single FFT: the per-block FFT size depends only on the
// kernel length, not on the (potentially huge) signal length.
//
// blockSize: new input samples per block.
//   0 (default) → auto-selected as next power-of-2 above 4 × kernel length.
//   Choosing blockSize ≈ 4…8 × kernel.size() gives good FFT efficiency.
std::vector<double> convolveOverlapAdd(
    const std::vector<double>& signal,
    const std::vector<double>& kernel,
    size_t blockSize = 0);

// ── Overlap-Save (Overlap-Discard) — streaming linear convolution ─────────────
//
// Alternative streaming method: uses a sliding input buffer of length fftN.
// The first (kLen − 1) output samples of each block contain circular-aliasing
// artefacts and are discarded; the remaining `step` samples are the valid output.
//
// Produces the same result as convolveLinear(signal, kernel, Full).
//
// blockSize: valid output samples per block (= fftN − kLen + 1).
//   0 (default) → fftN is chosen as next power-of-2 above 8 × kernel length.
std::vector<double> convolveOverlapSave(
    const std::vector<double>& signal,
    const std::vector<double>& kernel,
    size_t blockSize = 0);

// ═════════════════════════════════════════════════════════════════════════════
// CROSS-CORRELATION
// ═════════════════════════════════════════════════════════════════════════════

// ── FFT-based cross-correlation — O(N log N) ─────────────────────────────────
//
// corr[lag] = Σ_n a[n] · b[n + lag]       (real inputs, so conj(a) = a)
//
// Output layout (Full mode, length = la + lb − 1):
//   index 0           → lag = −(la − 1)   [most negative]
//   index la − 1      → lag = 0            [zero lag]
//   index la + lb − 2 → lag = lb − 1       [most positive]
//
// This matches the SciPy / NumPy convention for real 1-D inputs.
std::vector<double> crossCorrelate(
    const std::vector<double>& a,
    const std::vector<double>& b,
    ConvolutionMode mode = ConvolutionMode::Full);

// ── Normalized cross-correlation ──────────────────────────────────────────────
//
// R_norm[lag] = corr(a, b)[lag] / sqrt(energy(a) · energy(b))
//
// For equal-length inputs the peak value is in [−1, +1].
// Output length and layout are identical to crossCorrelate(a, b, Full).
std::vector<double> normalizedCrossCorrelate(
    const std::vector<double>& a,
    const std::vector<double>& b);

// ═════════════════════════════════════════════════════════════════════════════
// AUTOCORRELATION
// ═════════════════════════════════════════════════════════════════════════════

// ── Autocorrelation — R[lag] = Σ_n x[n] · x[n + lag] ─────────────────────────
//
// Full output length = 2·N − 1.
// The zero-lag value (maximum for a non-zero signal) is at index N − 1.
// The result is always symmetric: R[k] == R[−k].
std::vector<double> autoCorrelate(
    const std::vector<double>& x,
    ConvolutionMode mode = ConvolutionMode::Full);

// ── Normalized autocorrelation — R_norm[lag] = R[lag] / R[0] ─────────────────
//
// The zero-lag element equals exactly 1.0.
// Full output length = 2·N − 1, centre index = N − 1.
std::vector<double> normalizedAutoCorrelate(const std::vector<double>& x);

} // namespace SharedMath::DSP