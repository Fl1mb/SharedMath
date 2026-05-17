#pragma once

#include <vector>
#include <cstddef>

namespace SharedMath::DSP {

/// ─────────────────────────────────────────────────────────────────────────────
/// Window functions for spectral analysis and filter design
///
/// All functions return a std::vector<double> of length n.
///
/// symmetric = true  (default) — suited for FIR filter design (Nyquist criterion)
/// symmetric = false           — suited for spectral analysis (periodic, N+1 samples
///                               but only the first N are returned)
///
/// Reference:
///   Harris, F.J. (1978). On the use of windows for harmonic analysis with
///   the discrete Fourier transform. Proc. IEEE, 66(1), 51–83.
/// ─────────────────────────────────────────────────────────────────────────────

/// ── Rectangular (boxcar) ─────────────────────────────────────────────────────
std::vector<double> windowRectangular(size_t n);

/// ── Bartlett (triangular) ────────────────────────────────────────────────────
std::vector<double> windowBartlett(size_t n, bool symmetric = true);

/// ── Hann ─────────────────────────────────────────────────────────────────────
std::vector<double> windowHann(size_t n, bool symmetric = true);

/// ── Hamming ──────────────────────────────────────────────────────────────────
std::vector<double> windowHamming(size_t n, bool symmetric = true);

/// ── Blackman ─────────────────────────────────────────────────────────────────
std::vector<double> windowBlackman(size_t n, bool symmetric = true);

/// ── Blackman-Harris ──────────────────────────────────────────────────────────
std::vector<double> windowBlackmanHarris(size_t n, bool symmetric = true);

/// ── Nuttall ───────────────────────────────────────────────────────────────────
std::vector<double> windowNuttall(size_t n, bool symmetric = true);

/// ── Flat-top ─────────────────────────────────────────────────────────────────
std::vector<double> windowFlatTop(size_t n, bool symmetric = true);

// ── Kaiser ────────────────────────────────────────────────────────────────────
std::vector<double> windowKaiser(size_t n, double beta, bool symmetric = true);

/// Helper: compute Kaiser beta from desired peak sidelobe attenuation (dB)
double kaiserBeta(double attenuationDB);

// ── Gaussian ─────────────────────────────────────────────────────────────────
std::vector<double> windowGaussian(size_t n, double sigma = 0.4, bool symmetric = true);

// ── Tukey (tapered cosine) ────────────────────────────────────────────────────
std::vector<double> windowTukey(size_t n, double alpha = 0.5, bool symmetric = true);

/// ── Bartlett-Hann ────────────────────────────────────────────────────────────
std::vector<double> windowBartlettHann(size_t n, bool symmetric = true);

// ── Planck-taper ─────────────────────────────────────────────────────────────
std::vector<double> windowPlanck(size_t n, double epsilon = 0.1, bool symmetric = true);

/// ─────────────────────────────────────────────────────────────────────────────
/// Enum-based factory (for configuration-driven code)
/// ─────────────────────────────────────────────────────────────────────────────

enum class WindowType {
    Rectangular,
    Bartlett,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Nuttall,
    FlatTop,
    Kaiser,
    Gaussian,
    Tukey,
    BartlettHann,
    Planck
};

struct WindowParams {
    WindowType type      = WindowType::Hann;
    bool       symmetric = true;
    double     beta      = 8.6;   // Kaiser: sidelobe attenuation control
    double     sigma     = 0.4;   // Gaussian: width
    double     alpha     = 0.5;   // Tukey: taper fraction
    double     epsilon   = 0.1;   // Planck: taper fraction per side
};

std::vector<double> makeWindow(size_t n, const WindowParams& p = {});

/// ─────────────────────────────────────────────────────────────────────────────
/// Window metrics
/// ─────────────────────────────────────────────────────────────────────────────

/// Coherent power gain: CG = Σw[n] / N
double windowCoherentGain(const std::vector<double>& w);

/// Equivalent Noise Bandwidth (in bins): ENBW = N · Σw²[n] / (Σw[n])²
double windowENBW(const std::vector<double>& w);

/// Processing gain (dB relative to rectangular): PG = −20·log10(CG)
double windowProcessingGain(const std::vector<double>& w);

} // namespace SharedMath::DSP