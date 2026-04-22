#include <gtest/gtest.h>
#include "DSP/Convolution.h"

#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr double kTol = 1e-9;   // tolerance for FFT-based methods
static constexpr double kDir = 1e-12;  // tolerance for direct (exact) arithmetic

// Maximum absolute element-wise error between two equal-length vectors
static double maxErr(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 1e30;
    double e = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        e = std::max(e, std::abs(a[i] - b[i]));
    return e;
}

// Direct O(N²) linear convolution — independent reference implementation
static std::vector<double> naiveConv(const std::vector<double>& a,
                                      const std::vector<double>& b)
{
    if (a.empty() || b.empty()) return {};
    size_t la = a.size(), lb = b.size();
    std::vector<double> out(la + lb - 1, 0.0);
    for (size_t i = 0; i < la; ++i)
        for (size_t j = 0; j < lb; ++j)
            out[i + j] += a[i] * b[j];
    return out;
}

// Direct O(N²) circular convolution
static std::vector<double> naiveCirc(const std::vector<double>& a,
                                      const std::vector<double>& b,
                                      size_t n)
{
    std::vector<double> out(n, 0.0);
    for (size_t k = 0; k < n; ++k)
        for (size_t m = 0; m < n; ++m) {
            double av = (m < a.size()) ? a[m] : 0.0;
            double bv = ((k >= m ? k - m : k + n - m) < b.size())
                            ? b[k >= m ? k - m : k + n - m] : 0.0;
            out[k] += av * bv;
        }
    return out;
}

// Direct O(N²) cross-correlation: full output, negative lags first
// corr[k] = Σ_n a[n]*b[n + lag],  lag = k - (la-1)
static std::vector<double> naiveCross(const std::vector<double>& a,
                                       const std::vector<double>& b)
{
    if (a.empty() || b.empty()) return {};
    size_t la = a.size(), lb = b.size();
    size_t outLen = la + lb - 1;
    std::vector<double> out(outLen, 0.0);
    for (size_t k = 0; k < outLen; ++k) {
        int lag = static_cast<int>(k) - static_cast<int>(la - 1);
        for (size_t n = 0; n < la; ++n) {
            int bIdx = static_cast<int>(n) + lag;
            if (bIdx >= 0 && bIdx < static_cast<int>(lb))
                out[k] += a[n] * b[static_cast<size_t>(bIdx)];
        }
    }
    return out;
}

// Simple ramps / pulses for test signals
static std::vector<double> ramp(size_t n) {
    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), 1.0);
    return v;
}

static std::vector<double> impulse(size_t n, size_t pos = 0, double val = 1.0) {
    std::vector<double> v(n, 0.0);
    if (pos < n) v[pos] = val;
    return v;
}


// ═════════════════════════════════════════════════════════════════════════════
// convolveLinear  (FFT-based)
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConvolveLinear, EmptyInputs) {
    EXPECT_TRUE(convolveLinear({}, {1.0}).empty());
    EXPECT_TRUE(convolveLinear({1.0}, {}).empty());
    EXPECT_TRUE(convolveLinear({}, {}).empty());
}

TEST(ConvolveLinear, FullLength) {
    auto a = ramp(8);
    auto b = ramp(5);
    auto out = convolveLinear(a, b);
    EXPECT_EQ(out.size(), a.size() + b.size() - 1);
}

TEST(ConvolveLinear, MatchesNaive) {
    auto a = ramp(16);
    auto b = ramp(7);
    EXPECT_LT(maxErr(convolveLinear(a, b), naiveConv(a, b)), kTol);
}

TEST(ConvolveLinear, Commutative) {
    auto a = ramp(13);
    auto b = ramp(9);
    EXPECT_LT(maxErr(convolveLinear(a, b), convolveLinear(b, a)), kTol);
}

TEST(ConvolveLinear, ImpulseIsIdentity) {
    auto a = ramp(32);
    auto d = impulse(1);   // [1.0]
    auto out = convolveLinear(a, d);
    // convolve with unit impulse at 0 → same signal
    ASSERT_EQ(out.size(), a.size());
    EXPECT_LT(maxErr(out, a), kTol);
}

TEST(ConvolveLinear, DelayedImpulse) {
    auto a   = ramp(10);
    auto del = impulse(4, 2);   // impulse at position 2
    auto out = convolveLinear(a, del, ConvolutionMode::Full);
    // a convolved with δ[n-2] → a shifted right by 2, padded
    ASSERT_EQ(out.size(), a.size() + 3);
    for (size_t i = 0; i < 2; ++i)      EXPECT_NEAR(out[i], 0.0, kTol);
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(out[i + 2], a[i], kTol);
}

TEST(ConvolveLinear, SameMode) {
    auto a = ramp(16);
    auto b = ramp(5);
    auto full = convolveLinear(a, b, ConvolutionMode::Full);
    auto same = convolveLinear(a, b, ConvolutionMode::Same);
    EXPECT_EQ(same.size(), std::max(a.size(), b.size()));
    // Same is the centred part of Full
    size_t start = (full.size() - same.size()) / 2;
    for (size_t i = 0; i < same.size(); ++i)
        EXPECT_NEAR(same[i], full[start + i], kTol);
}

TEST(ConvolveLinear, ValidMode) {
    auto a = ramp(16);
    auto b = ramp(5);
    auto full  = convolveLinear(a, b, ConvolutionMode::Full);
    auto valid = convolveLinear(a, b, ConvolutionMode::Valid);
    size_t expectedLen = a.size() - b.size() + 1;
    EXPECT_EQ(valid.size(), expectedLen);
    // Valid is the inner part of Full (start at lb-1)
    for (size_t i = 0; i < valid.size(); ++i)
        EXPECT_NEAR(valid[i], full[b.size() - 1 + i], kTol);
}

TEST(ConvolveLinear, EqualLengthValidIsOneElement) {
    auto a = ramp(8);
    auto b = ramp(8);
    auto valid = convolveLinear(a, b, ConvolutionMode::Valid);
    auto full  = convolveLinear(a, b, ConvolutionMode::Full);
    ASSERT_EQ(valid.size(), 1u);
    // conv[k] = Σ a[i]*b[k-i], so the centre element (k = N-1) is
    // dot(a, reversed_b), NOT dot(a, b).  Verify against Full directly.
    EXPECT_NEAR(valid[0], full[a.size() - 1], kTol);
}

TEST(ConvolveLinear, SingleElementInputs) {
    std::vector<double> a = {3.0};
    std::vector<double> b = {7.0};
    auto out = convolveLinear(a, b);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(out[0], 21.0, kTol);
}

TEST(ConvolveLinear, DifferentSizes) {
    auto a = ramp(100);
    auto b = ramp(3);
    EXPECT_LT(maxErr(convolveLinear(a, b), naiveConv(a, b)), kTol);
}


// ═════════════════════════════════════════════════════════════════════════════
// convolveLinearDirect
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConvolveLinearDirect, EmptyInputs) {
    EXPECT_TRUE(convolveLinearDirect({}, {1.0}).empty());
    EXPECT_TRUE(convolveLinearDirect({1.0}, {}).empty());
}

TEST(ConvolveLinearDirect, MatchesNaive) {
    auto a = ramp(12);
    auto b = ramp(6);
    EXPECT_LT(maxErr(convolveLinearDirect(a, b), naiveConv(a, b)), kDir);
}

TEST(ConvolveLinearDirect, MatchesFFT) {
    auto a = ramp(20);
    auto b = ramp(8);
    EXPECT_LT(maxErr(convolveLinearDirect(a, b), convolveLinear(a, b)), kTol);
}

TEST(ConvolveLinearDirect, SameModeMatchesFFT) {
    auto a = ramp(20);
    auto b = ramp(8);
    EXPECT_LT(maxErr(
        convolveLinearDirect(a, b, ConvolutionMode::Same),
        convolveLinear(a, b, ConvolutionMode::Same)), kTol);
}

TEST(ConvolveLinearDirect, ValidModeMatchesFFT) {
    auto a = ramp(20);
    auto b = ramp(8);
    EXPECT_LT(maxErr(
        convolveLinearDirect(a, b, ConvolutionMode::Valid),
        convolveLinear(a, b, ConvolutionMode::Valid)), kTol);
}

TEST(ConvolveLinearDirect, Commutative) {
    auto a = ramp(7);
    auto b = ramp(5);
    EXPECT_LT(maxErr(convolveLinearDirect(a, b), convolveLinearDirect(b, a)), kDir);
}


// ═════════════════════════════════════════════════════════════════════════════
// convolveCircular
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConvolveCircular, EmptyInputs) {
    EXPECT_TRUE(convolveCircular({}, {1.0}, 4).empty());
    EXPECT_TRUE(convolveCircular({1.0}, {}, 4).empty());
    EXPECT_TRUE(convolveCircular({1.0}, {1.0}, 0).empty());
}

TEST(ConvolveCircular, OutputLength) {
    EXPECT_EQ(convolveCircular(ramp(4), ramp(4), 8).size(), 8u);
    EXPECT_EQ(convolveCircular(ramp(4), ramp(4), 5).size(), 5u);
}

TEST(ConvolveCircular, MatchesNaive) {
    auto a = ramp(5);
    auto b = ramp(5);
    size_t n = 8;
    EXPECT_LT(maxErr(convolveCircular(a, b, n), naiveCirc(a, b, n)), kTol);
}

TEST(ConvolveCircular, LargeLengthEqualsLinear) {
    // Circular with n >= la+lb-1 produces the same result as linear convolution
    auto a = ramp(8);
    auto b = ramp(5);
    size_t n = a.size() + b.size() - 1;  // = 12, which is >= la+lb-1
    // nextPow2 needed: convolveCircular works for any n
    auto circ  = convolveCircular(a, b, n);
    auto linear = convolveLinear(a, b);
    ASSERT_EQ(circ.size(), linear.size());
    EXPECT_LT(maxErr(circ, linear), kTol);
}

TEST(ConvolveCircular, ImpulseIsIdentity) {
    auto a = ramp(8);
    auto d = impulse(8);   // [1, 0, 0, …, 0]
    auto out = convolveCircular(a, d, 8);
    EXPECT_LT(maxErr(out, a), kTol);
}

TEST(ConvolveCircular, CircularShift) {
    // Convolving with a cyclic impulse at position k yields a cyclic shift of k
    size_t n = 8;
    auto a = ramp(n);
    auto shifted_impulse = impulse(n, 2);  // δ[n-2] cyclic
    auto out = convolveCircular(a, shifted_impulse, n);
    // Expected: a shifted right by 2 (cyclically)
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(out[i], a[(i + n - 2) % n], kTol);
}

TEST(ConvolveCircular, Commutative) {
    auto a = ramp(6);
    auto b = ramp(4);
    size_t n = 8;
    EXPECT_LT(maxErr(convolveCircular(a, b, n), convolveCircular(b, a, n)), kTol);
}


// ═════════════════════════════════════════════════════════════════════════════
// convolveOverlapAdd
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConvolveOverlapAdd, EmptyInputs) {
    EXPECT_TRUE(convolveOverlapAdd({}, {1.0}).empty());
    EXPECT_TRUE(convolveOverlapAdd({1.0}, {}).empty());
}

TEST(ConvolveOverlapAdd, OutputLength) {
    auto s = ramp(100);
    auto k = ramp(16);
    EXPECT_EQ(convolveOverlapAdd(s, k).size(), s.size() + k.size() - 1);
}

TEST(ConvolveOverlapAdd, MatchesLinear_DefaultBlock) {
    auto s = ramp(64);
    auto k = ramp(11);
    EXPECT_LT(maxErr(convolveOverlapAdd(s, k), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapAdd, MatchesLinear_SmallBlock) {
    auto s = ramp(50);
    auto k = ramp(7);
    // Force small blockSize to exercise multi-block path
    EXPECT_LT(maxErr(convolveOverlapAdd(s, k, 8), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapAdd, MatchesLinear_BlockSmallerThanKernel) {
    auto s = ramp(40);
    auto k = ramp(15);
    EXPECT_LT(maxErr(convolveOverlapAdd(s, k, 4), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapAdd, MatchesLinear_SignalSmallerThanBlock) {
    // Signal length < blockSize: single-block degenerate case
    auto s = ramp(8);
    auto k = ramp(4);
    EXPECT_LT(maxErr(convolveOverlapAdd(s, k, 64), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapAdd, MatchesLinear_LongSignal) {
    auto s = ramp(512);
    auto k = ramp(32);
    EXPECT_LT(maxErr(convolveOverlapAdd(s, k), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapAdd, KernelOfOne) {
    // Convolve with a scalar: should multiply every element
    auto s = ramp(20);
    std::vector<double> k = {3.0};
    auto out = convolveOverlapAdd(s, k);
    ASSERT_EQ(out.size(), s.size());
    for (size_t i = 0; i < s.size(); ++i)
        EXPECT_NEAR(out[i], s[i] * 3.0, kTol);
}


// ═════════════════════════════════════════════════════════════════════════════
// convolveOverlapSave
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConvolveOverlapSave, EmptyInputs) {
    EXPECT_TRUE(convolveOverlapSave({}, {1.0}).empty());
    EXPECT_TRUE(convolveOverlapSave({1.0}, {}).empty());
}

TEST(ConvolveOverlapSave, OutputLength) {
    auto s = ramp(100);
    auto k = ramp(16);
    EXPECT_EQ(convolveOverlapSave(s, k).size(), s.size() + k.size() - 1);
}

TEST(ConvolveOverlapSave, MatchesLinear_DefaultBlock) {
    auto s = ramp(64);
    auto k = ramp(11);
    EXPECT_LT(maxErr(convolveOverlapSave(s, k), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapSave, MatchesLinear_SmallBlock) {
    auto s = ramp(50);
    auto k = ramp(7);
    EXPECT_LT(maxErr(convolveOverlapSave(s, k, 8), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapSave, MatchesLinear_LargeKernel) {
    auto s = ramp(40);
    auto k = ramp(15);
    EXPECT_LT(maxErr(convolveOverlapSave(s, k, 32), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapSave, MatchesLinear_LongSignal) {
    auto s = ramp(512);
    auto k = ramp(32);
    EXPECT_LT(maxErr(convolveOverlapSave(s, k), convolveLinear(s, k)), kTol);
}

TEST(ConvolveOverlapSave, AgreesWithOverlapAdd) {
    auto s = ramp(128);
    auto k = ramp(21);
    EXPECT_LT(maxErr(convolveOverlapSave(s, k), convolveOverlapAdd(s, k)), kTol);
}

TEST(ConvolveOverlapSave, KernelOfOne) {
    auto s = ramp(20);
    std::vector<double> k = {5.0};
    auto out = convolveOverlapSave(s, k);
    ASSERT_EQ(out.size(), s.size());
    for (size_t i = 0; i < s.size(); ++i)
        EXPECT_NEAR(out[i], s[i] * 5.0, kTol);
}


// ═════════════════════════════════════════════════════════════════════════════
// crossCorrelate
// ═════════════════════════════════════════════════════════════════════════════

TEST(CrossCorrelate, EmptyInputs) {
    EXPECT_TRUE(crossCorrelate({}, {1.0}).empty());
    EXPECT_TRUE(crossCorrelate({1.0}, {}).empty());
}

TEST(CrossCorrelate, FullLength) {
    auto a = ramp(8);
    auto b = ramp(5);
    EXPECT_EQ(crossCorrelate(a, b).size(), a.size() + b.size() - 1);
}

TEST(CrossCorrelate, MatchesNaive) {
    auto a = ramp(10);
    auto b = ramp(6);
    EXPECT_LT(maxErr(crossCorrelate(a, b), naiveCross(a, b)), kTol);
}

TEST(CrossCorrelate, AutoCorrelationIsPeakAtCentre) {
    auto a   = ramp(16);
    auto r   = crossCorrelate(a, a);
    size_t centre = a.size() - 1;
    // The zero-lag value (index = N-1) must be the maximum
    double peak = r[centre];
    for (double v : r)
        EXPECT_LE(v, peak + kTol);
}

TEST(CrossCorrelate, ZeroLagEqualsEnergyForSelf) {
    auto a = ramp(12);
    auto r = crossCorrelate(a, a);
    double energy = 0.0;
    for (double v : a) energy += v * v;
    EXPECT_NEAR(r[a.size() - 1], energy, kTol);
}

TEST(CrossCorrelate, SameModeLength) {
    auto a = ramp(16);
    auto b = ramp(5);
    auto same = crossCorrelate(a, b, ConvolutionMode::Same);
    EXPECT_EQ(same.size(), std::max(a.size(), b.size()));
}

TEST(CrossCorrelate, ValidModeLength) {
    auto a = ramp(16);
    auto b = ramp(5);
    auto valid = crossCorrelate(a, b, ConvolutionMode::Valid);
    size_t expectedLen = std::max(a.size(), b.size())
                       - std::min(a.size(), b.size()) + 1;
    EXPECT_EQ(valid.size(), expectedLen);
}

TEST(CrossCorrelate, EqualLengthValidLength) {
    auto a = ramp(8);
    auto b = ramp(8);
    // Valid of equal-length: length = 1
    auto valid = crossCorrelate(a, b, ConvolutionMode::Valid);
    EXPECT_EQ(valid.size(), 1u);
}

TEST(CrossCorrelate, ImpulseAtZeroPicksOutElement) {
    // corr(a, delta_at_0)[lag=0] = a[0]
    auto a = ramp(10);
    auto d = impulse(10);
    auto r = crossCorrelate(a, d);
    // zero-lag index = a.size()-1 = 9
    EXPECT_NEAR(r[a.size() - 1], a[0], kTol);
}

TEST(CrossCorrelate, ShiftedImpulse) {
    // corr(a, delta_at_k) picks out a shifted version of a
    auto a = ramp(10);
    size_t shift = 3;
    auto d = impulse(10, shift);
    auto r = crossCorrelate(a, d);
    // corr[lag=k] = a[n]*b[n+k] = a[n]*delta[n+k-shift] = a[shift-k] at n=shift-k
    // At zero-lag index (a.size()-1), corr = a[shift] (b has index shift at n=shift)
    // More precisely: corr at zero-lag = sum_n a[n]*d[n] = a[shift]*d[shift] = a[shift]*1
    EXPECT_NEAR(r[a.size() - 1], a[shift], kTol);
}

TEST(CrossCorrelate, SymmetryRelation) {
    // corr(a,b)[lag] == corr(b,a)[-lag]
    // In index terms: crossCorr(a,b)[k] == crossCorr(b,a)[outLen-1-k]
    auto a = ramp(8);
    auto b = ramp(5);
    auto rab = crossCorrelate(a, b);
    auto rba = crossCorrelate(b, a);
    // outLen for (a,b) = 12, for (b,a) = 12 — same sizes because la+lb-1 is symmetric
    ASSERT_EQ(rab.size(), rba.size());
    for (size_t k = 0; k < rab.size(); ++k)
        EXPECT_NEAR(rab[k], rba[rab.size() - 1 - k], kTol);
}


// ═════════════════════════════════════════════════════════════════════════════
// normalizedCrossCorrelate
// ═════════════════════════════════════════════════════════════════════════════

TEST(NormalizedCrossCorrelate, EmptyInputs) {
    EXPECT_TRUE(normalizedCrossCorrelate({}, {1.0}).empty());
    EXPECT_TRUE(normalizedCrossCorrelate({1.0}, {}).empty());
}

TEST(NormalizedCrossCorrelate, SelfCorrelationPeakIsOne) {
    auto a = ramp(16);
    auto r = normalizedCrossCorrelate(a, a);
    size_t centre = a.size() - 1;
    EXPECT_NEAR(r[centre], 1.0, kTol);
}

TEST(NormalizedCrossCorrelate, AllValuesInMinusOneToOne) {
    auto a = ramp(16);
    auto b = {5.0, -1.0, 3.0, 2.0, -4.0, 0.0, 7.0, 1.0,
              -2.0, 3.0, 0.5, -0.5, 6.0, -3.0, 2.0, 1.0};
    auto r = normalizedCrossCorrelate(a, b);
    for (double v : r)
        EXPECT_LE(std::abs(v), 1.0 + kTol);
}

TEST(NormalizedCrossCorrelate, ZeroSignalGivesZeros) {
    std::vector<double> zeros(8, 0.0);
    auto a = ramp(8);
    auto r = normalizedCrossCorrelate(zeros, a);
    for (double v : r) EXPECT_NEAR(v, 0.0, kTol);
}

TEST(NormalizedCrossCorrelate, OutputLength) {
    auto a = ramp(10);
    auto b = ramp(6);
    EXPECT_EQ(normalizedCrossCorrelate(a, b).size(), a.size() + b.size() - 1);
}


// ═════════════════════════════════════════════════════════════════════════════
// autoCorrelate
// ═════════════════════════════════════════════════════════════════════════════

TEST(AutoCorrelate, EmptyInput) {
    EXPECT_TRUE(autoCorrelate({}).empty());
}

TEST(AutoCorrelate, OutputLength) {
    auto x = ramp(16);
    EXPECT_EQ(autoCorrelate(x).size(), 2 * x.size() - 1);
}

TEST(AutoCorrelate, Symmetry) {
    auto x = ramp(16);
    auto r = autoCorrelate(x);
    // R[k] == R[2N-2-k]  (symmetric around centre)
    for (size_t k = 0; k < r.size(); ++k)
        EXPECT_NEAR(r[k], r[r.size() - 1 - k], kTol);
}

TEST(AutoCorrelate, PeakAtZeroLag) {
    auto x = ramp(16);
    auto r = autoCorrelate(x);
    double peak = r[x.size() - 1];  // zero-lag index
    for (double v : r)
        EXPECT_LE(v, peak + kTol);
}

TEST(AutoCorrelate, ZeroLagEqualsEnergy) {
    auto x = ramp(12);
    double energy = 0.0;
    for (double v : x) energy += v * v;
    auto r = autoCorrelate(x);
    EXPECT_NEAR(r[x.size() - 1], energy, kTol);
}

TEST(AutoCorrelate, SameModeLength) {
    auto x = ramp(16);
    auto r = autoCorrelate(x, ConvolutionMode::Same);
    EXPECT_EQ(r.size(), x.size());
}

TEST(AutoCorrelate, ValidModeIsOneElement) {
    auto x = ramp(16);
    auto r = autoCorrelate(x, ConvolutionMode::Valid);
    // Valid of (N,N) → length 1 → the zero-lag value
    ASSERT_EQ(r.size(), 1u);
    double energy = 0.0;
    for (double v : x) energy += v * v;
    EXPECT_NEAR(r[0], energy, kTol);
}

TEST(AutoCorrelate, MatchesCrossCorrelateWithSelf) {
    auto x = ramp(20);
    auto ac  = autoCorrelate(x);
    auto cc  = crossCorrelate(x, x);
    EXPECT_LT(maxErr(ac, cc), kTol);
}

TEST(AutoCorrelate, ConstantSignal) {
    // Autocorr of all-ones: R[lag] = N - |lag|
    size_t n = 8;
    std::vector<double> x(n, 1.0);
    auto r = autoCorrelate(x);
    for (size_t k = 0; k < r.size(); ++k) {
        // lag = k - (n-1),  expected = n - |lag|
        int lag     = static_cast<int>(k) - static_cast<int>(n - 1);
        double expected = static_cast<double>(n - static_cast<size_t>(std::abs(lag)));
        EXPECT_NEAR(r[k], expected, kTol);
    }
}


// ═════════════════════════════════════════════════════════════════════════════
// normalizedAutoCorrelate
// ═════════════════════════════════════════════════════════════════════════════

TEST(NormalizedAutoCorrelate, EmptyInput) {
    EXPECT_TRUE(normalizedAutoCorrelate({}).empty());
}

TEST(NormalizedAutoCorrelate, ZeroLagIsOne) {
    auto x = ramp(16);
    auto r = normalizedAutoCorrelate(x);
    EXPECT_NEAR(r[x.size() - 1], 1.0, kTol);
}

TEST(NormalizedAutoCorrelate, AllValuesInMinusOneToOne) {
    // White noise–like signal
    std::vector<double> x = {1.0, -2.0, 3.5, -1.5, 2.0,
                              -3.0, 0.5, 4.0, -2.5, 1.0};
    auto r = normalizedAutoCorrelate(x);
    for (double v : r)
        EXPECT_LE(std::abs(v), 1.0 + kTol);
}

TEST(NormalizedAutoCorrelate, Symmetry) {
    auto x = ramp(16);
    auto r = normalizedAutoCorrelate(x);
    for (size_t k = 0; k < r.size(); ++k)
        EXPECT_NEAR(r[k], r[r.size() - 1 - k], kTol);
}

TEST(NormalizedAutoCorrelate, OutputLength) {
    auto x = ramp(12);
    EXPECT_EQ(normalizedAutoCorrelate(x).size(), 2 * x.size() - 1);
}

TEST(NormalizedAutoCorrelate, ZeroSignalAllZeros) {
    std::vector<double> zeros(8, 0.0);
    auto r = normalizedAutoCorrelate(zeros);
    for (double v : r) EXPECT_NEAR(v, 0.0, kTol);
}

TEST(NormalizedAutoCorrelate, IsScaledVersionOfAutoCorrelate) {
    auto x  = ramp(16);
    auto raw = autoCorrelate(x);
    auto nor = normalizedAutoCorrelate(x);
    double r0 = raw[x.size() - 1];
    ASSERT_GT(r0, 0.0);
    for (size_t k = 0; k < raw.size(); ++k)
        EXPECT_NEAR(nor[k], raw[k] / r0, kTol);
}


// ═════════════════════════════════════════════════════════════════════════════
// Cross-method consistency checks
// ═════════════════════════════════════════════════════════════════════════════

TEST(ConvolutionConsistency, OverlapAddVsOverlapSave) {
    auto s = ramp(200);
    auto k = ramp(17);
    EXPECT_LT(maxErr(convolveOverlapAdd(s, k), convolveOverlapSave(s, k)), kTol);
}

TEST(ConvolutionConsistency, DirectVsFFT_Various) {
    for (size_t la : {1u, 2u, 7u, 16u, 31u}) {
        for (size_t lb : {1u, 3u, 8u, 15u}) {
            auto a = ramp(la);
            auto b = ramp(lb);
            EXPECT_LT(maxErr(convolveLinearDirect(a, b), convolveLinear(a, b)), kTol)
                << "la=" << la << " lb=" << lb;
        }
    }
}

TEST(ConvolutionConsistency, CircularEqualLinearWhenSufficientLength) {
    auto a = ramp(8);
    auto b = ramp(6);
    size_t n = a.size() + b.size() - 1;  // = 13  (no aliasing)
    auto circ   = convolveCircular(a, b, n);
    auto linear = convolveLinear(a, b);
    EXPECT_LT(maxErr(circ, linear), kTol);
}

TEST(ConvolutionConsistency, Parseval_ViaAutoCorrelation) {
    // Parseval's theorem: Σ x[n]² == R_xx[0]
    auto x = ramp(32);
    double sumSq = 0.0;
    for (double v : x) sumSq += v * v;
    auto r = autoCorrelate(x);
    EXPECT_NEAR(r[x.size() - 1], sumSq, kTol);
}
