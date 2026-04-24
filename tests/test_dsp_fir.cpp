#include <gtest/gtest.h>
#include "DSP/FIR.h"
#include "DSP/FFT.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace SharedMath::DSP;

static constexpr double kPi  = 3.14159265358979323846;
static constexpr double kTol = 1e-9;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static double maxErr(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e30;
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// Peak magnitude of a real vector over the given index range [lo, hi).
static double peakAbs(const std::vector<double>& v, size_t lo, size_t hi) {
    double p = 0.0;
    for (size_t i = lo; i < hi && i < v.size(); ++i)
        p = std::max(p, std::abs(v[i]));
    return p;
}

// Cosine of given normalized frequency (1 = Nyquist), length n.
static std::vector<double> cosine(size_t n, double normFreq) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = std::cos(2.0 * kPi * normFreq * 0.5 * static_cast<double>(i));
    return x;
}

// ═════════════════════════════════════════════════════════════════════════════
// FIRDesign — structural properties of designed coefficients
// ═════════════════════════════════════════════════════════════════════════════

TEST(FIRDesign, LowPassLength) {
    auto h = designFIRLowPass(64, 0.3);
    EXPECT_EQ(h.size(), 65u);  // order 64 → 65 taps
}

TEST(FIRDesign, LowPassLengthOddOrderRounded) {
    // odd order must be bumped to even; 65 → 66 taps
    auto h = designFIRLowPass(65, 0.3);
    EXPECT_EQ(h.size() % 2, 1u);  // odd number of taps = Type I
    EXPECT_GE(h.size(), 66u);
}

TEST(FIRDesign, LowPassSymmetry) {
    auto h = designFIRLowPass(64, 0.25);
    size_t M = h.size() - 1;
    for (size_t k = 0; k <= M / 2; ++k)
        EXPECT_NEAR(h[k], h[M - k], 1e-15);
}

TEST(FIRDesign, HighPassSymmetry) {
    auto h = designFIRHighPass(64, 0.35);
    size_t M = h.size() - 1;
    for (size_t k = 0; k <= M / 2; ++k)
        EXPECT_NEAR(h[k], h[M - k], 1e-15);
}

TEST(FIRDesign, BandPassSymmetry) {
    auto h = designFIRBandPass(64, 0.2, 0.5);
    size_t M = h.size() - 1;
    for (size_t k = 0; k <= M / 2; ++k)
        EXPECT_NEAR(h[k], h[M - k], 1e-15);
}

TEST(FIRDesign, BandStopSymmetry) {
    auto h = designFIRBandStop(64, 0.2, 0.5);
    size_t M = h.size() - 1;
    for (size_t k = 0; k <= M / 2; ++k)
        EXPECT_NEAR(h[k], h[M - k], 1e-15);
}

TEST(FIRDesign, LowPassDCGainNearUnity) {
    auto h = designFIRLowPass(64, 0.5);  // fc at Nyquist/2 → all-pass half
    double dcGain = 0.0;
    for (double v : h) dcGain += v;
    EXPECT_NEAR(dcGain, 1.0, 0.01);
}

TEST(FIRDesign, HighPassNyquistGainNearUnity) {
    // Evaluate H(z) at z = exp(jπ) = -1 (Nyquist): sum h[k]*(-1)^k
    auto h = designFIRHighPass(64, 0.5);
    double gain = 0.0;
    for (size_t k = 0; k < h.size(); ++k)
        gain += h[k] * (k % 2 == 0 ? 1.0 : -1.0);
    EXPECT_NEAR(std::abs(gain), 1.0, 0.01);
}

TEST(FIRDesign, KaiserFIR_ReturnsNonEmpty) {
    auto h = designKaiserFIR(0.25, 0.05, 60.0);
    EXPECT_GT(h.size(), 1u);
}

TEST(FIRDesign, KaiserFIR_HigherAttenuationGivesMoreTaps) {
    auto h40  = designKaiserFIR(0.25, 0.05, 40.0);
    auto h80  = designKaiserFIR(0.25, 0.05, 80.0);
    EXPECT_LT(h40.size(), h80.size());
}

TEST(FIRDesign, KaiserFIR_NarrowerTransitionMoreTaps) {
    auto hWide   = designKaiserFIR(0.25, 0.1, 60.0);
    auto hNarrow = designKaiserFIR(0.25, 0.02, 60.0);
    EXPECT_LT(hWide.size(), hNarrow.size());
}

TEST(FIRDesign, InvalidCutoff_Throws) {
    EXPECT_THROW(designFIRLowPass(64, 0.0), std::invalid_argument);
    EXPECT_THROW(designFIRLowPass(64, 1.0), std::invalid_argument);
    EXPECT_THROW(designFIRHighPass(64, 1.1), std::invalid_argument);
}

TEST(FIRDesign, BandPass_InvalidCutoffs_Throws) {
    EXPECT_THROW(designFIRBandPass(64, 0.5, 0.3), std::invalid_argument);  // reversed
    EXPECT_THROW(designFIRBandPass(64, 0.0, 0.5), std::invalid_argument);  // zero low
    EXPECT_THROW(designFIRBandPass(64, 0.3, 1.0), std::invalid_argument);  // one at Nyquist
}

// ═════════════════════════════════════════════════════════════════════════════
// FIRFilter — frequency-domain behaviour via filtfilt on pure sinusoids
//
// Signal: 512 samples, test region: samples 128..383 (middle half).
// This avoids edge transients while containing enough cycles to hit amplitude 1.
// Passband criterion: max |y| in [0.9, 1.1].
// Stopband criterion: max |y| < 0.05.
// ═════════════════════════════════════════════════════════════════════════════

static const size_t kN = 512;

TEST(FIRFilter, LowPass_Passband) {
    auto h = designFIRLowPass(64, 0.4);
    auto y = filtfilt(cosine(kN, 0.2), h);  // 0.2 well below cutoff 0.4
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.1);
}

TEST(FIRFilter, LowPass_Stopband) {
    auto h = designFIRLowPass(64, 0.25);
    auto y = filtfilt(cosine(kN, 0.6), h);  // 0.6 well above cutoff 0.25
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.05);
}

TEST(FIRFilter, HighPass_Passband) {
    auto h = designFIRHighPass(64, 0.3);
    auto y = filtfilt(cosine(kN, 0.7), h);  // 0.7 well above cutoff 0.3
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.1);
}

TEST(FIRFilter, HighPass_Stopband) {
    auto h = designFIRHighPass(64, 0.4);
    auto y = filtfilt(cosine(kN, 0.1), h);  // 0.1 well below cutoff 0.4
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.05);
}

TEST(FIRFilter, BandPass_Center) {
    auto h = designFIRBandPass(64, 0.2, 0.6);
    auto y = filtfilt(cosine(kN, 0.4), h);  // center of [0.2, 0.6]
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.1);
}

TEST(FIRFilter, BandPass_BelowBand) {
    auto h = designFIRBandPass(64, 0.3, 0.6);
    auto y = filtfilt(cosine(kN, 0.1), h);
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.05);
}

TEST(FIRFilter, BandPass_AboveBand) {
    auto h = designFIRBandPass(64, 0.2, 0.5);
    auto y = filtfilt(cosine(kN, 0.8), h);
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.05);
}

TEST(FIRFilter, BandStop_Center) {
    auto h = designFIRBandStop(64, 0.2, 0.6);
    auto y = filtfilt(cosine(kN, 0.4), h);  // center of notch
    EXPECT_LT(peakAbs(y, kN/4, 3*kN/4), 0.05);
}

TEST(FIRFilter, BandStop_Passband) {
    auto h = designFIRBandStop(64, 0.3, 0.6);
    auto y = filtfilt(cosine(kN, 0.1), h);  // below notch
    EXPECT_NEAR(peakAbs(y, kN/4, 3*kN/4), 1.0, 0.1);
}

// ═════════════════════════════════════════════════════════════════════════════
// FIRApply — applyFIR basic behaviour
// ═════════════════════════════════════════════════════════════════════════════

TEST(FIRApply, OutputLengthEqualInput) {
    auto h = designFIRLowPass(32, 0.3);
    std::vector<double> sig(200, 1.0);
    EXPECT_EQ(applyFIR(sig, h).size(), sig.size());
}

TEST(FIRApply, EmptySignalReturnsEmpty) {
    auto h = designFIRLowPass(32, 0.3);
    EXPECT_TRUE(applyFIR({}, h).empty());
}

// ═════════════════════════════════════════════════════════════════════════════
// FIRFiltfilt — zero-phase and symmetry properties
// ═════════════════════════════════════════════════════════════════════════════

TEST(FIRFiltfilt, OutputLengthEqualInput) {
    auto h = designFIRLowPass(32, 0.3);
    std::vector<double> sig(300, 0.0);
    std::iota(sig.begin(), sig.end(), 0.0);
    EXPECT_EQ(filtfilt(sig, h).size(), sig.size());
}

TEST(FIRFiltfilt, EmptySignalReturnsEmpty) {
    auto h = designFIRLowPass(32, 0.3);
    EXPECT_TRUE(filtfilt({}, h).empty());
}

TEST(FIRFiltfilt, ZeroPhase_SymmetricInputSymmetricOutput) {
    // A palindromic (even-symmetric) signal filtered by a symmetric FIR
    // via filtfilt must yield a palindromic output (zero-phase property).
    const size_t N = 128;
    std::vector<double> sym(N, 0.0);
    for (size_t i = 0; i < N / 2; ++i) {
        double v = std::sin(2.0 * kPi * 0.08 * static_cast<double>(i));
        sym[i]       = v;
        sym[N - 1 - i] = v;
    }

    auto h = designFIRLowPass(32, 0.25);
    auto y = filtfilt(sym, h);

    ASSERT_EQ(y.size(), N);
    // Interior samples (away from edges where the filter has less context)
    for (size_t i = 8; i < N - 8; ++i)
        EXPECT_NEAR(y[i], y[N - 1 - i], kTol);
}

TEST(FIRFiltfilt, SquaredMagnitudeResponse) {
    // filtfilt = two passes → |H_ff(f)|² = |H(f)|⁴.
    // At DC a LP filter has gain ≈ 1, so filtfilt gain ≈ 1 at DC too.
    auto h = designFIRLowPass(64, 0.4);
    std::vector<double> dc(256, 1.0);
    auto y = filtfilt(dc, h);
    // Interior should be very close to 1 (DC passes through LP unchanged)
    for (size_t i = 32; i < 224; ++i)
        EXPECT_NEAR(y[i], 1.0, 0.01);
}
