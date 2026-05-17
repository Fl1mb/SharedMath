#pragma once

#include "FFTPlan.h"
#include "FFTConfig.h"
#include "Window.h"

#include <complex>
#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

// ─────────────────────────────────────────────────────────────────────────────
// High-level FFT convenience API
//
// These free functions create a temporary FFTPlan, execute it, and return.
// For repeated transforms of the same size, create an FFTPlan directly and
// reuse it — that avoids re-allocating twiddle-factor tables.
// ─────────────────────────────────────────────────────────────────────────────

// ── In-place forward FFT ─────────────────────────────────────────────────────
// Transforms x into its DFT in-place.  Default: no normalization.
void fft(std::vector<std::complex<double>>& x,
         FFTNorm norm = FFTNorm::None);

// ── In-place inverse FFT ─────────────────────────────────────────────────────
// Transforms X into its IDFT in-place.  Default: 1/N normalization.
void ifft(std::vector<std::complex<double>>& X,
          FFTNorm norm = FFTNorm::ByN);

// ── Real FFT (half-spectrum) ─────────────────────────────────────────────────
// Input: N real samples.
// Output: N/2+1 complex DFT bins (non-redundant half due to Hermitian symmetry).
//
// To get the full spectrum, mirror: X[N-k] = conj(X[k]) for k=1..N/2-1.
std::vector<std::complex<double>> rfft(const std::vector<double>& x,
                                        FFTNorm norm = FFTNorm::None);

// ── Inverse real FFT ─────────────────────────────────────────────────────────
// Input:  N/2+1 complex bins (from rfft).
// Output: N real samples.
// n must match the original transform length.
std::vector<double> irfft(const std::vector<std::complex<double>>& X,
                           size_t n,
                           FFTNorm norm = FFTNorm::ByN);

/// ─────────────────────────────────────────────────────────────────────────────
/// Spectral analysis helpers
/// ─────────────────────────────────────────────────────────────────────────────

/// Magnitude spectrum: |X[k]|
std::vector<double> magnitude(const std::vector<std::complex<double>>& X);

/// Phase spectrum: arg(X[k]) in radians ∈ (−π, π]
std::vector<double> phase(const std::vector<std::complex<double>>& X);

/// Power spectrum: |X[k]|²
std::vector<double> powerSpectrum(const std::vector<std::complex<double>>& X);

// Power spectrum in dB: 10·log₁₀(|X[k]|² / refPower)
// refPower defaults to 1.0 (0 dBFS convention).
std::vector<double> powerSpectrumDB(const std::vector<std::complex<double>>& X,
                                    double refPower = 1.0);

// Magnitude in dB: 20·log₁₀(|X[k]| / refMag)
std::vector<double> magnitudeDB(const std::vector<std::complex<double>>& X,
                                double refMag = 1.0);

/// ─────────────────────────────────────────────────────────────────────────────
/// Frequency axis
/// ─────────────────────────────────────────────────────────────────────────────

/// Returns the frequency (Hz) for each of the n FFT output bins.
/// Bins 0..N/2 are positive frequencies; bins N/2+1..N-1 are negative
/// (matching NumPy fftfreq convention).
std::vector<double> fftFrequencies(size_t n, double sampleRate);

/// Frequency axis for rfft output (N/2+1 bins, 0..sampleRate/2).
std::vector<double> rfftFrequencies(size_t n, double sampleRate);

// ─────────────────────────────────────────────────────────────────────────────
// fftShift / ifftShift  (like numpy.fft.fftshift)
// ─────────────────────────────────────────────────────────────────────────────

// Shift zero-frequency component to the centre of the spectrum.
template<typename T>
std::vector<T> fftShift(const std::vector<T>& x);

// Inverse of fftShift.
template<typename T>
std::vector<T> ifftShift(const std::vector<T>& x);

// ─────────────────────────────────────────────────────────────────────────────
// Window-FFT pipeline helpers
// ─────────────────────────────────────────────────────────────────────────────

// Multiply signal by window element-wise.
std::vector<double> applyWindow(const std::vector<double>& signal,
                                const std::vector<double>& window);

// Windowed FFT in one call:
//   1. Multiply signal by window
//   2. Forward FFT
//   3. Return complex spectrum
std::vector<std::complex<double>>
windowedFFT(const std::vector<double>& signal,
            const std::vector<double>& window,
            FFTNorm norm = FFTNorm::None);

// ─────────────────────────────────────────────────────────────────────────────
// Convolution via FFT — O(N log N)
// ─────────────────────────────────────────────────────────────────────────────

// Linear convolution of a and b.
// Output length = a.size() + b.size() - 1.
std::vector<double> convolve(const std::vector<double>& a,
                              const std::vector<double>& b);

// Cross-correlation of a and b: corr[k] = Σ conj(a[n]) · b[n+k]
// Output length = a.size() + b.size() - 1.
std::vector<double> correlate(const std::vector<double>& a,
                               const std::vector<double>& b);

} // namespace SharedMath::DSP

// Include template implementations (must be in header)
#include "FFT_impl.h"