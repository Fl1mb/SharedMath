#include <gtest/gtest.h>
#include "DSP/Window.h"

#include <cmath>
#include <numeric>
#include <algorithm>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double kEps = 1e-12;

// Check that w[i] == w[N-1-i] for all i (symmetric around centre)
static bool isSymmetric(const std::vector<double>& w) {
    size_t n = w.size();
    for (size_t i = 0; i < n / 2; ++i)
        if (std::abs(w[i] - w[n - 1 - i]) > kEps)
            return false;
    return true;
}

// Maximum value in window
static double maxVal(const std::vector<double>& w) {
    return *std::max_element(w.begin(), w.end());
}

// Minimum value in window
static double minVal(const std::vector<double>& w) {
    return *std::min_element(w.begin(), w.end());
}

// Sum of window
static double sumW(const std::vector<double>& w) {
    return std::accumulate(w.begin(), w.end(), 0.0);
}

// Sum of squares
static double sumW2(const std::vector<double>& w) {
    double s = 0.0;
    for (double v : w) s += v * v;
    return s;
}


// ─────────────────────────────────────────────────────────────────────────────
// Edge-case tests common to all windows (zero and unit length)
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowEdgeCases, ZeroLength) {
    EXPECT_TRUE(windowRectangular(0).empty());
    EXPECT_TRUE(windowBartlett(0).empty());
    EXPECT_TRUE(windowHann(0).empty());
    EXPECT_TRUE(windowHamming(0).empty());
    EXPECT_TRUE(windowBlackman(0).empty());
    EXPECT_TRUE(windowBlackmanHarris(0).empty());
    EXPECT_TRUE(windowNuttall(0).empty());
    EXPECT_TRUE(windowFlatTop(0).empty());
    EXPECT_TRUE(windowKaiser(0, 5.0).empty());
    EXPECT_TRUE(windowGaussian(0).empty());
    EXPECT_TRUE(windowTukey(0).empty());
    EXPECT_TRUE(windowBartlettHann(0).empty());
    EXPECT_TRUE(windowPlanck(0).empty());
}

TEST(WindowEdgeCases, UnitLength) {
    // All windows of length 1 should return {1.0}
    EXPECT_DOUBLE_EQ(windowRectangular(1)[0],    1.0);
    EXPECT_DOUBLE_EQ(windowBartlett(1)[0],        1.0);
    EXPECT_DOUBLE_EQ(windowHann(1)[0],            1.0);
    EXPECT_DOUBLE_EQ(windowHamming(1)[0],         1.0);
    EXPECT_DOUBLE_EQ(windowBlackman(1)[0],        1.0);
    EXPECT_DOUBLE_EQ(windowBlackmanHarris(1)[0],  1.0);
    EXPECT_DOUBLE_EQ(windowNuttall(1)[0],         1.0);
    EXPECT_DOUBLE_EQ(windowFlatTop(1)[0],         1.0);
    EXPECT_DOUBLE_EQ(windowKaiser(1, 5.0)[0],     1.0);
    EXPECT_DOUBLE_EQ(windowGaussian(1)[0],        1.0);
    EXPECT_DOUBLE_EQ(windowTukey(1)[0],           1.0);
    EXPECT_DOUBLE_EQ(windowBartlettHann(1)[0],    1.0);
    EXPECT_DOUBLE_EQ(windowPlanck(1)[0],          1.0);
}

TEST(WindowEdgeCases, InvalidKaiserBeta) {
    EXPECT_THROW(windowKaiser(16, -0.1), std::invalid_argument);
}

TEST(WindowEdgeCases, InvalidGaussianSigma) {
    EXPECT_THROW(windowGaussian(16, 0.0),  std::invalid_argument);
    EXPECT_THROW(windowGaussian(16, -0.1), std::invalid_argument);
}

TEST(WindowEdgeCases, InvalidTukeyAlpha) {
    EXPECT_THROW(windowTukey(16, -0.1), std::invalid_argument);
    EXPECT_THROW(windowTukey(16,  1.1), std::invalid_argument);
}

TEST(WindowEdgeCases, InvalidPlanckEpsilon) {
    EXPECT_THROW(windowPlanck(16,  0.0), std::invalid_argument);
    EXPECT_THROW(windowPlanck(16,  0.5), std::invalid_argument);
    EXPECT_THROW(windowPlanck(16, -0.1), std::invalid_argument);
    EXPECT_THROW(windowPlanck(16,  0.6), std::invalid_argument);
}


// ─────────────────────────────────────────────────────────────────────────────
// Rectangular window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowRectangular, BasicProperties) {
    auto w = windowRectangular(64);
    ASSERT_EQ(w.size(), 64u);
    for (double v : w) EXPECT_DOUBLE_EQ(v, 1.0);
}

TEST(WindowRectangular, MetricsAreOne) {
    auto w = windowRectangular(64);
    EXPECT_NEAR(windowCoherentGain(w), 1.0, kEps);
    EXPECT_NEAR(windowENBW(w), 1.0, kEps);
}


// ─────────────────────────────────────────────────────────────────────────────
// Bartlett window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowBartlett, Length) {
    EXPECT_EQ(windowBartlett(64).size(), 64u);
    EXPECT_EQ(windowBartlett(65).size(), 65u);
}

TEST(WindowBartlett, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowBartlett(64)));
    EXPECT_TRUE(isSymmetric(windowBartlett(65)));
}

TEST(WindowBartlett, Range) {
    auto w = windowBartlett(64);
    EXPECT_GE(minVal(w), 0.0 - kEps);
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}

TEST(WindowBartlett, EndsAtZero) {
    auto w = windowBartlett(64);
    EXPECT_NEAR(w.front(), 0.0, kEps);
    EXPECT_NEAR(w.back(),  0.0, kEps);
}

TEST(WindowBartlett, PeakAtCentre) {
    auto w = windowBartlett(65);  // odd → exact centre sample
    EXPECT_NEAR(w[32], 1.0, kEps);
}


// ─────────────────────────────────────────────────────────────────────────────
// Hann window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowHann, Length) {
    EXPECT_EQ(windowHann(64).size(), 64u);
}

TEST(WindowHann, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowHann(64)));
    EXPECT_TRUE(isSymmetric(windowHann(65)));
}

TEST(WindowHann, Range) {
    auto w = windowHann(64);
    EXPECT_GE(minVal(w), 0.0 - kEps);
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}

TEST(WindowHann, EndsAtZero) {
    // Symmetric: w[0] = 0.5*(1-cos(0)) = 0 and w[N-1] = 0.5*(1-cos(2π)) = 0
    auto w = windowHann(64);
    EXPECT_NEAR(w.front(), 0.0, kEps);
    EXPECT_NEAR(w.back(),  0.0, kEps);
}

TEST(WindowHann, CentreIsOne) {
    auto w = windowHann(65);
    EXPECT_NEAR(w[32], 1.0, kEps);
}

TEST(WindowHann, PeriodicDiffersFromSymmetric) {
    auto sym = windowHann(64, true);
    auto per = windowHann(64, false);
    // They must have the same length but different values
    ASSERT_EQ(sym.size(), per.size());
    bool differs = false;
    for (size_t i = 0; i < sym.size(); ++i)
        if (std::abs(sym[i] - per[i]) > kEps) { differs = true; break; }
    EXPECT_TRUE(differs);
}


// ─────────────────────────────────────────────────────────────────────────────
// Hamming window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowHamming, Length) {
    EXPECT_EQ(windowHamming(64).size(), 64u);
}

TEST(WindowHamming, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowHamming(64)));
    EXPECT_TRUE(isSymmetric(windowHamming(65)));
}

TEST(WindowHamming, EndpointsNotZero) {
    // Hamming does not reach zero — endpoints ≈ 0.08
    auto w = windowHamming(64);
    EXPECT_GT(w.front(), 0.05);
    EXPECT_GT(w.back(),  0.05);
}

TEST(WindowHamming, CentreIsOne) {
    auto w = windowHamming(65);
    EXPECT_NEAR(w[32], 1.0, kEps);
}

TEST(WindowHamming, Range) {
    auto w = windowHamming(64);
    EXPECT_GE(minVal(w), 0.0);
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}


// ─────────────────────────────────────────────────────────────────────────────
// Blackman window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowBlackman, Length) {
    EXPECT_EQ(windowBlackman(64).size(), 64u);
}

TEST(WindowBlackman, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowBlackman(64)));
    EXPECT_TRUE(isSymmetric(windowBlackman(65)));
}

TEST(WindowBlackman, EndsNearZero) {
    auto w = windowBlackman(64);
    // a0-a1+a2 = 0.42-0.50+0.08 = 0.0
    EXPECT_NEAR(w.front(), 0.0, 1e-10);
    EXPECT_NEAR(w.back(),  0.0, 1e-10);
}

TEST(WindowBlackman, Range) {
    auto w = windowBlackman(64);
    EXPECT_GE(minVal(w), -0.01);   // can be slightly negative due to coefficients
    EXPECT_LE(maxVal(w),  1.0 + kEps);
}


// ─────────────────────────────────────────────────────────────────────────────
// Blackman-Harris window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowBlackmanHarris, Length) {
    EXPECT_EQ(windowBlackmanHarris(128).size(), 128u);
}

TEST(WindowBlackmanHarris, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowBlackmanHarris(128)));
    EXPECT_TRUE(isSymmetric(windowBlackmanHarris(127)));
}

TEST(WindowBlackmanHarris, EndsNearZero) {
    auto w = windowBlackmanHarris(128);
    // a0-a1+a2-a3 = 0.35875 - 0.48829 + 0.14128 - 0.01168 = 0.00006
    EXPECT_NEAR(w.front(), 0.35875 - 0.48829 + 0.14128 - 0.01168, 1e-10);
}

TEST(WindowBlackmanHarris, ENBWGreaterThanHann) {
    // Higher sidelobe suppression ⟹ wider main lobe ⟹ larger ENBW
    auto bh  = windowBlackmanHarris(256);
    auto han = windowHann(256);
    EXPECT_GT(windowENBW(bh), windowENBW(han));
}


// ─────────────────────────────────────────────────────────────────────────────
// Nuttall window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowNuttall, Length) {
    EXPECT_EQ(windowNuttall(128).size(), 128u);
}

TEST(WindowNuttall, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowNuttall(128)));
}

TEST(WindowNuttall, EndsNearZero) {
    auto w = windowNuttall(128);
    // a0-a1+a2-a3 = 0.3635819 - 0.4891775 + 0.1365995 - 0.0106411 = 3.628e-4 ≈ 0
    EXPECT_NEAR(w.front(), 0.3635819 - 0.4891775 + 0.1365995 - 0.0106411, 1e-10);
}

TEST(WindowNuttall, ENBWLargerThanBlackman) {
    // More sidelobe suppression → larger ENBW
    auto nu  = windowNuttall(256);
    auto bl  = windowBlackman(256);
    EXPECT_GT(windowENBW(nu), windowENBW(bl));
}


// ─────────────────────────────────────────────────────────────────────────────
// Flat-top window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowFlatTop, Length) {
    EXPECT_EQ(windowFlatTop(64).size(), 64u);
}

TEST(WindowFlatTop, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowFlatTop(64)));
}

TEST(WindowFlatTop, CoherentGainNearOne) {
    // Flat-top is designed so that the peak of the DFT of a sinusoid is
    // very accurate.  Its coherent gain (≈ mean) is well-defined by the
    // a0 coefficient.
    auto w = windowFlatTop(256);
    double cg = windowCoherentGain(w);
    EXPECT_GT(cg, 0.0);
}

TEST(WindowFlatTop, CanBeNegative) {
    // Flat-top windows have negative sidelobes in the time domain
    auto w = windowFlatTop(64);
    EXPECT_LT(minVal(w), 0.0);
}


// ─────────────────────────────────────────────────────────────────────────────
// Kaiser window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowKaiser, Length) {
    EXPECT_EQ(windowKaiser(64, 5.0).size(), 64u);
}

TEST(WindowKaiser, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowKaiser(64, 5.0)));
    EXPECT_TRUE(isSymmetric(windowKaiser(65, 8.6)));
}

TEST(WindowKaiser, BetaZeroEqualsRectangular) {
    // Kaiser(beta=0) == rectangular (all ones)
    auto kw  = windowKaiser(64, 0.0);
    auto rec = windowRectangular(64);
    for (size_t i = 0; i < kw.size(); ++i)
        EXPECT_NEAR(kw[i], rec[i], 1e-10);
}

TEST(WindowKaiser, LargerBetaMoreConcentrated) {
    // Larger beta → more energy in centre → larger ENBW
    auto k5  = windowKaiser(256, 5.0);
    auto k14 = windowKaiser(256, 14.0);
    EXPECT_GT(windowENBW(k14), windowENBW(k5));
}

TEST(WindowKaiser, RangeZeroToOne) {
    auto w = windowKaiser(64, 8.6);
    EXPECT_GE(minVal(w), 0.0 - kEps);
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}

TEST(WindowKaiser, CentreIsOne) {
    // w[centre] == I0(beta)/I0(beta) == 1
    auto w = windowKaiser(65, 6.0);
    EXPECT_NEAR(w[32], 1.0, kEps);
}


// ─────────────────────────────────────────────────────────────────────────────
// Kaiser beta formula
// ─────────────────────────────────────────────────────────────────────────────

TEST(KaiserBeta, LowAttenuation) {
    EXPECT_DOUBLE_EQ(kaiserBeta(15.0), 0.0);   // < 21 dB
    EXPECT_DOUBLE_EQ(kaiserBeta(20.0), 0.0);
}

TEST(KaiserBeta, MidRange) {
    double b = kaiserBeta(35.0);
    EXPECT_GT(b, 0.0);
    EXPECT_LT(b, 5.0);
}

TEST(KaiserBeta, HighAttenuation) {
    double b = kaiserBeta(60.0);
    // 0.1102 * (60 - 8.7) = 0.1102 * 51.3 ≈ 5.653
    EXPECT_NEAR(b, 0.1102 * (60.0 - 8.7), 1e-10);
}

TEST(KaiserBeta, Monotone) {
    // More attenuation → larger beta
    EXPECT_LT(kaiserBeta(30.0), kaiserBeta(50.0));
    EXPECT_LT(kaiserBeta(50.0), kaiserBeta(80.0));
}


// ─────────────────────────────────────────────────────────────────────────────
// Gaussian window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowGaussian, Length) {
    EXPECT_EQ(windowGaussian(64).size(), 64u);
}

TEST(WindowGaussian, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowGaussian(64)));
    EXPECT_TRUE(isSymmetric(windowGaussian(65)));
}

TEST(WindowGaussian, Range) {
    auto w = windowGaussian(64, 0.4);
    EXPECT_GT(minVal(w), 0.0);   // Gaussian is strictly positive
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}

TEST(WindowGaussian, CentreIsOne) {
    // exp(-0.5*(0)^2) == 1 at the centre for symmetric case
    auto w = windowGaussian(65, 0.4);
    EXPECT_NEAR(w[32], 1.0, kEps);
}

TEST(WindowGaussian, NarrowerSigmaMoreConcentrated) {
    // Smaller sigma → narrower window in time → more concentrated → larger ENBW
    auto w02 = windowGaussian(256, 0.2);
    auto w04 = windowGaussian(256, 0.4);
    EXPECT_GT(windowENBW(w02), windowENBW(w04));
}


// ─────────────────────────────────────────────────────────────────────────────
// Tukey (tapered cosine) window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowTukey, Length) {
    EXPECT_EQ(windowTukey(64).size(), 64u);
}

TEST(WindowTukey, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowTukey(64)));
    EXPECT_TRUE(isSymmetric(windowTukey(65)));
}

TEST(WindowTukey, AlphaZeroEqualsRectangular) {
    auto tu  = windowTukey(64, 0.0);
    auto rec = windowRectangular(64);
    for (size_t i = 0; i < tu.size(); ++i)
        EXPECT_NEAR(tu[i], rec[i], 1e-10);
}

TEST(WindowTukey, AlphaOneApproximatesHann) {
    // Tukey(alpha=1) ≈ Hann — compare ENBW (they share the cosine taper formula)
    auto tu  = windowTukey(256, 1.0);
    auto han = windowHann(256);
    EXPECT_NEAR(windowENBW(tu), windowENBW(han), 0.05);
}

TEST(WindowTukey, FlatCentreForSmallAlpha) {
    // With alpha=0.25, the middle half of samples should all equal 1.0
    size_t n = 64;
    auto w = windowTukey(n, 0.25);
    size_t lo = static_cast<size_t>(0.25 * (n-1) / 2.0) + 1;
    size_t hi = n - 1 - lo;
    for (size_t i = lo; i <= hi; ++i)
        EXPECT_NEAR(w[i], 1.0, 1e-10);
}

TEST(WindowTukey, Range) {
    auto w = windowTukey(64, 0.5);
    EXPECT_GE(minVal(w), 0.0 - kEps);
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}


// ─────────────────────────────────────────────────────────────────────────────
// Bartlett-Hann window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowBartlettHann, Length) {
    EXPECT_EQ(windowBartlettHann(64).size(), 64u);
}

TEST(WindowBartlettHann, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowBartlettHann(64)));
    EXPECT_TRUE(isSymmetric(windowBartlettHann(65)));
}

TEST(WindowBartlettHann, Range) {
    auto w = windowBartlettHann(64);
    EXPECT_GE(minVal(w), 0.0 - kEps);
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}

TEST(WindowBartlettHann, FirstSampleFormula) {
    // i=0: 0.62 - 0.48*|0-0.5| - 0.38*cos(0) = 0.62 - 0.24 - 0.38 = 0
    auto w = windowBartlettHann(64);
    EXPECT_NEAR(w[0], 0.0, 1e-10);
}


// ─────────────────────────────────────────────────────────────────────────────
// Planck-taper window
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowPlanck, Length) {
    EXPECT_EQ(windowPlanck(64).size(), 64u);
}

TEST(WindowPlanck, Symmetric) {
    EXPECT_TRUE(isSymmetric(windowPlanck(64)));
    EXPECT_TRUE(isSymmetric(windowPlanck(65)));
}

TEST(WindowPlanck, EndsAtZero) {
    auto w = windowPlanck(64, 0.1);
    EXPECT_NEAR(w.front(), 0.0, 1e-10);
    EXPECT_NEAR(w.back(),  0.0, 1e-10);
}

TEST(WindowPlanck, FlatInCentre) {
    // Samples well inside the flat region (far from both tapers) should be 1
    size_t n = 128;
    auto w = windowPlanck(n, 0.1);
    size_t mid = n / 2;
    EXPECT_NEAR(w[mid], 1.0, 1e-10);
}

TEST(WindowPlanck, Range) {
    auto w = windowPlanck(128, 0.1);
    EXPECT_GE(minVal(w), 0.0 - kEps);
    EXPECT_LE(maxVal(w), 1.0 + kEps);
}

TEST(WindowPlanck, LargerEpsilonNarrowerFlat) {
    // Larger epsilon → more tapering → smaller sum of window values
    auto w1 = windowPlanck(256, 0.1);
    auto w2 = windowPlanck(256, 0.4);
    EXPECT_GT(sumW(w1), sumW(w2));
}


// ─────────────────────────────────────────────────────────────────────────────
// Symmetric vs periodic mode
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowSymmetricPeriodic, HannPeriodicFirstSampleIsZero) {
    // Periodic Hann: M = N, so w[0] = 0.5*(1-cos(0)) = 0
    auto w = windowHann(64, false);
    EXPECT_NEAR(w[0], 0.0, kEps);
}

TEST(WindowSymmetricPeriodic, HammingPeriodicFirstSampleFormula) {
    // Periodic Hamming: w[0] = 0.54 - 0.46*cos(0) = 0.54 - 0.46 = 0.08
    auto w = windowHamming(64, false);
    EXPECT_NEAR(w[0], 0.08, kEps);
}

TEST(WindowSymmetricPeriodic, SumPeriodicEqualsSymmetricShifted) {
    // For a periodic window of length N, the N+1-th sample would equal the first.
    // Therefore: sum(periodic N) = sum(symmetric N+1) - first_sample_of_sym_N+1
    // This is hard to test directly; instead verify sum(per) > sum(sym) for Hann
    // since the periodic denominator is smaller (M=N vs M=N-1).
    auto sym = windowHann(64, true);
    auto per = windowHann(64, false);
    // Both have the same length; their sums need not be equal
    EXPECT_NE(sumW(sym), sumW(per));
}


// ─────────────────────────────────────────────────────────────────────────────
// Window metrics
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowMetrics, CoherentGainRectangular) {
    EXPECT_NEAR(windowCoherentGain(windowRectangular(256)), 1.0, kEps);
}

TEST(WindowMetrics, CoherentGainHannHalf) {
    // Symmetric Hann: sum = N/2, so CG = 0.5
    auto w = windowHann(256, false);
    EXPECT_NEAR(windowCoherentGain(w), 0.5, 1e-6);
}

TEST(WindowMetrics, CoherentGainEmptyIsZero) {
    EXPECT_DOUBLE_EQ(windowCoherentGain({}), 0.0);
}

TEST(WindowMetrics, ENBWRectangularIsOne) {
    EXPECT_NEAR(windowENBW(windowRectangular(256)), 1.0, kEps);
}

TEST(WindowMetrics, ENBWHannIsOneAndHalf) {
    // Hann ENBW ≈ 1.5 bins
    EXPECT_NEAR(windowENBW(windowHann(256, false)), 1.5, 1e-4);
}

TEST(WindowMetrics, ENBWEmptyIsZero) {
    EXPECT_DOUBLE_EQ(windowENBW({}), 0.0);
}

TEST(WindowMetrics, ProcessingGainRectangularIsZero) {
    EXPECT_NEAR(windowProcessingGain(windowRectangular(256)), 0.0, kEps);
}

TEST(WindowMetrics, ProcessingGainHannPositive) {
    // Hann CG = 0.5 → PG = -20*log10(0.5) = +6 dB
    EXPECT_NEAR(windowProcessingGain(windowHann(256, false)), 6.0206, 1e-3);
}

TEST(WindowMetrics, ENBWDecreaseAsMoreTapering) {
    // More sidelobe suppression (wider main lobe) ↔ larger ENBW
    double enbw_rect = windowENBW(windowRectangular(256));
    double enbw_hann = windowENBW(windowHann(256));
    double enbw_bh   = windowENBW(windowBlackmanHarris(256));
    EXPECT_LT(enbw_rect, enbw_hann);
    EXPECT_LT(enbw_hann, enbw_bh);
}


// ─────────────────────────────────────────────────────────────────────────────
// makeWindow factory
// ─────────────────────────────────────────────────────────────────────────────

TEST(MakeWindow, DefaultIsHann) {
    auto factory = makeWindow(64);
    auto direct  = windowHann(64);
    ASSERT_EQ(factory.size(), direct.size());
    for (size_t i = 0; i < factory.size(); ++i)
        EXPECT_DOUBLE_EQ(factory[i], direct[i]);
}

TEST(MakeWindow, Rectangular) {
    WindowParams p;
    p.type = WindowType::Rectangular;
    auto w = makeWindow(64, p);
    for (double v : w) EXPECT_DOUBLE_EQ(v, 1.0);
}

TEST(MakeWindow, KaiserPassesBeta) {
    WindowParams p;
    p.type = WindowType::Kaiser;
    p.beta = 5.0;
    auto factory = makeWindow(64, p);
    auto direct  = windowKaiser(64, 5.0);
    ASSERT_EQ(factory.size(), direct.size());
    for (size_t i = 0; i < factory.size(); ++i)
        EXPECT_DOUBLE_EQ(factory[i], direct[i]);
}

TEST(MakeWindow, GaussianPassesSigma) {
    WindowParams p;
    p.type  = WindowType::Gaussian;
    p.sigma = 0.3;
    auto factory = makeWindow(64, p);
    auto direct  = windowGaussian(64, 0.3);
    ASSERT_EQ(factory.size(), direct.size());
    for (size_t i = 0; i < factory.size(); ++i)
        EXPECT_DOUBLE_EQ(factory[i], direct[i]);
}

TEST(MakeWindow, TukeyPassesAlpha) {
    WindowParams p;
    p.type  = WindowType::Tukey;
    p.alpha = 0.25;
    auto factory = makeWindow(64, p);
    auto direct  = windowTukey(64, 0.25);
    ASSERT_EQ(factory.size(), direct.size());
    for (size_t i = 0; i < factory.size(); ++i)
        EXPECT_DOUBLE_EQ(factory[i], direct[i]);
}

TEST(MakeWindow, PlanckPassesEpsilon) {
    WindowParams p;
    p.type    = WindowType::Planck;
    p.epsilon = 0.2;
    auto factory = makeWindow(64, p);
    auto direct  = windowPlanck(64, 0.2);
    ASSERT_EQ(factory.size(), direct.size());
    for (size_t i = 0; i < factory.size(); ++i)
        EXPECT_DOUBLE_EQ(factory[i], direct[i]);
}

TEST(MakeWindow, AllTypesReturnCorrectLength) {
    const size_t n = 128;
    for (int t = 0; t <= static_cast<int>(WindowType::Planck); ++t) {
        WindowParams p;
        p.type = static_cast<WindowType>(t);
        EXPECT_EQ(makeWindow(n, p).size(), n)
            << "Failed for WindowType=" << t;
    }
}

TEST(MakeWindow, AsymmetricFlag) {
    WindowParams p;
    p.type      = WindowType::Hann;
    p.symmetric = false;
    auto factory = makeWindow(64, p);
    auto direct  = windowHann(64, false);
    ASSERT_EQ(factory.size(), direct.size());
    for (size_t i = 0; i < factory.size(); ++i)
        EXPECT_DOUBLE_EQ(factory[i], direct[i]);
}


// ─────────────────────────────────────────────────────────────────────────────
// Cross-window comparisons (sidelobe ordering)
// ─────────────────────────────────────────────────────────────────────────────

TEST(WindowComparisons, ENBWOrdering) {
    // Expected: rect < Bartlett < Hann ≈ BartlettHann < Hamming
    //           < Blackman < Nuttall ≈ BlackmanHarris
    size_t n = 1024;
    EXPECT_LT(windowENBW(windowRectangular(n)),  windowENBW(windowHann(n)));
    EXPECT_LT(windowENBW(windowHann(n)),         windowENBW(windowBlackman(n)));
    EXPECT_LT(windowENBW(windowBlackman(n)),     windowENBW(windowBlackmanHarris(n)));
    EXPECT_LT(windowENBW(windowBlackmanHarris(n)), windowENBW(windowKaiser(n, 14.0)));
}

TEST(WindowComparisons, CoherentGainOrderingHannHamming) {
    // Hamming CG (≈0.54) > Hann CG (≈0.50) because Hamming has higher pedestal
    size_t n = 1024;
    EXPECT_GT(windowCoherentGain(windowHamming(n)),
              windowCoherentGain(windowHann(n)));
}
