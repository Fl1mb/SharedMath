#include <gtest/gtest.h>
#include "DSP/SignalProcessing.h"

#include <cmath>
#include <numeric>
#include <vector>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Statistics

TEST(Mean, EmptyReturnsZero) {
    EXPECT_DOUBLE_EQ(mean({}), 0.0);
}

TEST(Mean, KnownValues) {
    EXPECT_NEAR(mean({1.0, 2.0, 3.0, 4.0}), 2.5, 1e-12);
}

TEST(RMS, EmptyReturnsZero) {
    EXPECT_DOUBLE_EQ(rms({}), 0.0);
}

TEST(RMS, UnitAmplitudeSine) {
    // RMS of {1, 0, -1, 0} → sqrt((1+0+1+0)/4) = sqrt(0.5)
    std::vector<double> x = {1.0, 0.0, -1.0, 0.0};
    EXPECT_NEAR(rms(x), std::sqrt(0.5), 1e-12);
}

TEST(PeakAbs, EmptyReturnsZero) {
    EXPECT_DOUBLE_EQ(peakAbs({}), 0.0);
}

TEST(PeakAbs, FindsMaxAbsolute) {
    EXPECT_NEAR(peakAbs({-3.0, 1.0, 2.0}), 3.0, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// removeDC

TEST(RemoveDC, EmptyReturnsEmpty) {
    EXPECT_TRUE(removeDC({}).empty());
}

TEST(RemoveDC, MeanBecomesZero) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto y = removeDC(x);
    double m = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    EXPECT_NEAR(m, 0.0, 1e-12);
}

TEST(RemoveDC, PreservesVariance) {
    std::vector<double> x = {10.0, 11.0, 9.0, 10.5};
    auto y = removeDC(x);
    EXPECT_NEAR(rms(y), rms(removeDC(x)), 1e-12);
    EXPECT_EQ(y.size(), x.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// normalizePeak

TEST(NormalizePeak, ScalesToTargetPeak) {
    std::vector<double> x = {0.5, -2.0, 1.0};
    auto y = normalizePeak(x, 1.0);
    EXPECT_NEAR(peakAbs(y), 1.0, 1e-12);
}

TEST(NormalizePeak, CustomTarget) {
    std::vector<double> x = {1.0, -3.0, 2.0};
    auto y = normalizePeak(x, 5.0);
    EXPECT_NEAR(peakAbs(y), 5.0, 1e-12);
}

TEST(NormalizePeak, AllZeroThrows) {
    EXPECT_THROW(normalizePeak({0.0, 0.0, 0.0}), std::invalid_argument);
}

TEST(NormalizePeak, BadTargetThrows) {
    EXPECT_THROW(normalizePeak({1.0, 2.0}, 0.0),  std::invalid_argument);
    EXPECT_THROW(normalizePeak({1.0, 2.0}, -1.0), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// normalizeRMS

TEST(NormalizeRMS, ScalesToTargetRMS) {
    std::vector<double> x = {1.0, -1.0, 1.0, -1.0};
    auto y = normalizeRMS(x, 2.0);
    EXPECT_NEAR(rms(y), 2.0, 1e-12);
}

TEST(NormalizeRMS, ZeroRMSThrows) {
    EXPECT_THROW(normalizeRMS({0.0, 0.0}), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// movingAverage

TEST(MovingAverage, ConstantSignalUnchanged) {
    std::vector<double> x(20, 3.0);
    auto y = movingAverage(x, 5);
    for (double v : y) EXPECT_NEAR(v, 3.0, 1e-12);
}

TEST(MovingAverage, WindowSizeOneIsIdentity) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto y = movingAverage(x, 1);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], x[i], 1e-12);
}

TEST(MovingAverage, ZeroWindowThrows) {
    EXPECT_THROW(movingAverage({1.0, 2.0}, 0), std::invalid_argument);
}

TEST(MovingAverage, SpikeIsSmoothed) {
    std::vector<double> x(50, 0.0);
    x[25] = 100.0;
    auto y = movingAverage(x, 11);
    EXPECT_LT(y[25], 100.0);
    EXPECT_GT(y[25], 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// exponentialSmoothing

TEST(ExponentialSmoothing, AlphaOneIsIdentity) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    auto y = exponentialSmoothing(x, 1.0);
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], x[i], 1e-12);
}

TEST(ExponentialSmoothing, InvalidAlphaThrows) {
    EXPECT_THROW(exponentialSmoothing({1.0}, 0.0),  std::invalid_argument);
    EXPECT_THROW(exponentialSmoothing({1.0}, 1.01), std::invalid_argument);
    EXPECT_THROW(exponentialSmoothing({1.0}, -0.1), std::invalid_argument);
}

TEST(ExponentialSmoothing, SmoothingReducesPeak) {
    std::vector<double> x(20, 0.0);
    x[10] = 1.0;
    auto y = exponentialSmoothing(x, 0.1);
    EXPECT_LT(y[10], 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// medianFilter

TEST(MedianFilter, RemovesImpulseSpike) {
    std::vector<double> x(30, 1.0);
    x[15] = 1000.0;
    auto y = medianFilter(x, 5);
    EXPECT_NEAR(y[15], 1.0, 1e-9);
}

TEST(MedianFilter, ConstantSignalUnchanged) {
    std::vector<double> x(20, 5.0);
    auto y = medianFilter(x, 7);
    for (double v : y) EXPECT_NEAR(v, 5.0, 1e-12);
}

TEST(MedianFilter, ZeroKernelThrows) {
    EXPECT_THROW(medianFilter({1.0, 2.0}, 0), std::invalid_argument);
}

TEST(MedianFilter, EvenKernelForcedOdd) {
    // Even kernel 4 → forced to 5; should not throw and output same length
    std::vector<double> x(10, 1.0);
    EXPECT_NO_THROW({
        auto y = medianFilter(x, 4);
        EXPECT_EQ(y.size(), x.size());
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// detrendLinear

TEST(DetrendLinear, RemovesLinearTrend) {
    // x[n] = n + noise-free, so trend == x itself; result should be ~0
    std::vector<double> x(50);
    for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(i);
    auto y = detrendLinear(x);
    for (double v : y) EXPECT_NEAR(v, 0.0, 1e-9);
}

TEST(DetrendLinear, PreservesACComponent) {
    // x[n] = sin(2π·f·n) + linear trend; after detrend, rms of sine should remain
    const size_t N = 256;
    const double f = 0.05;  // normalized frequency
    const double pi = 3.14159265358979323846;
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i)
        x[i] = std::sin(2.0 * pi * f * i) + 0.1 * static_cast<double>(i);
    auto y = detrendLinear(x);

    double rmsY = rms(y);
    EXPECT_NEAR(rmsY, 1.0 / std::sqrt(2.0), 0.05);
}

TEST(DetrendLinear, SingleSampleReturnedUnchanged) {
    std::vector<double> x = {7.0};
    auto y = detrendLinear(x);
    ASSERT_EQ(y.size(), 1u);
    EXPECT_NEAR(y[0], 7.0, 1e-12);
}
