#include <gtest/gtest.h>
#include "DSP/SignalGenerator.h"
#include "DSP/FFT.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace SharedMath::DSP;

static constexpr double kPI = 3.14159265358979323846;

// ─────────────────────────────────────────────────────────────────────────────
// sineWave

TEST(SineWave, CorrectLength) {
    EXPECT_EQ(sineWave(100.0, 1000.0, 256).size(), 256u);
    EXPECT_EQ(sineWave(100.0, 1000.0, 0).size(),   0u);
}

TEST(SineWave, AmplitudeRespected) {
    // 250 Hz at 1000 Hz → period = 4 samples; sample n=1 lands exactly on sin(π/2)=1
    auto s = sineWave(250.0, 1000.0, 1000, 3.7);
    double mx = *std::max_element(s.begin(), s.end());
    double mn = *std::min_element(s.begin(), s.end());
    EXPECT_LE(mx,  3.7 + 1e-9);
    EXPECT_GE(mn, -3.7 - 1e-9);
    EXPECT_NEAR(mx, 3.7, 1e-9);
}

TEST(SineWave, ZeroFrequencyIsZero) {
    // sin(0) = 0 everywhere except phase offset
    auto s = sineWave(0.0, 1000.0, 100, 1.0, 0.0);
    for (double v : s) EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(SineWave, PhaseOffset) {
    // sin(0 + π/2) = 1.0 → first sample should be 1.0
    auto s = sineWave(0.0, 1000.0, 10, 1.0, kPI / 2.0);
    EXPECT_NEAR(s[0], 1.0, 1e-12);
}

TEST(SineWave, KnownSamples) {
    // 250 Hz sine at 1000 Hz sample rate: period = 4 samples
    // n=0: sin(0)=0, n=1: sin(π/2)=1, n=2: sin(π)=0, n=3: sin(3π/2)=-1
    auto s = sineWave(250.0, 1000.0, 4);
    EXPECT_NEAR(s[0],  0.0, 1e-12);
    EXPECT_NEAR(s[1],  1.0, 1e-12);
    EXPECT_NEAR(s[2],  0.0, 1e-12);
    EXPECT_NEAR(s[3], -1.0, 1e-12);
}

TEST(SineWave, FrequencyPeakInSpectrum) {
    double fs = 1000.0, f = 100.0;
    size_t N = 1024;
    auto s = sineWave(f, fs, N);
    auto X = rfft(s);
    auto mag = magnitude(X);
    size_t pk = static_cast<size_t>(
        std::max_element(mag.begin(), mag.end()) - mag.begin());
    // Expected bin ≈ f/fs * N = 100/1000 * 1024 = 102.4 → bin 102
    EXPECT_NEAR(static_cast<double>(pk) * fs / N, f, 2.0 * fs / N);
}

// ─────────────────────────────────────────────────────────────────────────────
// chirp

TEST(Chirp, CorrectLength) {
    EXPECT_EQ(chirp(10.0, 200.0, 1000.0, 512).size(), 512u);
    EXPECT_EQ(chirp(10.0, 200.0, 1000.0, 0).size(),   0u);
}

TEST(Chirp, AmplitudeRespected) {
    auto s = chirp(10.0, 200.0, 1000.0, 1024, 2.5);
    for (double v : s) EXPECT_LE(std::abs(v), 2.5 + 1e-9);
}

TEST(Chirp, StartsAtF0) {
    // At t=0: cos(2π·f0·0) = 1.0
    auto s = chirp(100.0, 400.0, 1000.0, 1024, 1.0);
    EXPECT_NEAR(s[0], 1.0, 1e-12);
}

TEST(Chirp, DifferentFromSineWave) {
    // Chirp is not a constant-frequency sine wave
    auto ch = chirp(50.0, 200.0, 1000.0, 512);
    auto sw = sineWave(50.0, 1000.0, 512);
    double diff = 0.0;
    for (size_t i = 0; i < ch.size(); ++i)
        diff = std::max(diff, std::abs(ch[i] - sw[i]));
    EXPECT_GT(diff, 0.1);   // They diverge quickly as frequency sweeps
}

// ─────────────────────────────────────────────────────────────────────────────
// whiteNoise

TEST(WhiteNoise, CorrectLength) {
    EXPECT_EQ(whiteNoise(256).size(), 256u);
    EXPECT_EQ(whiteNoise(0).size(),   0u);
}

TEST(WhiteNoise, Reproducible) {
    auto a = whiteNoise(1000, 1.0, 42u);
    auto b = whiteNoise(1000, 1.0, 42u);
    EXPECT_EQ(a, b);
}

TEST(WhiteNoise, DifferentSeedsDifferentOutput) {
    auto a = whiteNoise(1000, 1.0, 0u);
    auto b = whiteNoise(1000, 1.0, 1u);
    EXPECT_NE(a, b);
}

TEST(WhiteNoise, AmplitudeBounded) {
    auto s = whiteNoise(10000, 2.0, 7u);
    for (double v : s) {
        EXPECT_LE(v,  2.0);
        EXPECT_GE(v, -2.0);
    }
}

TEST(WhiteNoise, ZeroMeanApproximately) {
    auto s = whiteNoise(100000, 1.0, 99u);
    double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
    EXPECT_NEAR(mean, 0.0, 0.01);   // very loose tolerance — statistical
}

// ─────────────────────────────────────────────────────────────────────────────
// impulse

TEST(Impulse, CorrectLength) {
    EXPECT_EQ(impulse(256).size(), 256u);
    EXPECT_EQ(impulse(0).size(),   0u);
}

TEST(Impulse, DefaultPositionIsZero) {
    auto s = impulse(10);
    EXPECT_NEAR(s[0], 1.0, 1e-12);
    for (size_t i = 1; i < s.size(); ++i) EXPECT_NEAR(s[i], 0.0, 1e-12);
}

TEST(Impulse, PositionRespected) {
    auto s = impulse(16, 5, 3.0);
    for (size_t i = 0; i < s.size(); ++i) {
        if (i == 5) EXPECT_NEAR(s[i], 3.0, 1e-12);
        else        EXPECT_NEAR(s[i], 0.0, 1e-12);
    }
}

TEST(Impulse, OutOfBoundsPositionAllZeros) {
    auto s = impulse(8, 100, 1.0);
    for (double v : s) EXPECT_NEAR(v, 0.0, 1e-12);
}

TEST(Impulse, FFTOfImpulseIsFlat) {
    // DFT of unit impulse at n=0 is the all-ones vector
    auto s = impulse(16, 0, 1.0);
    auto X = rfft(s);
    auto mag = magnitude(X);
    for (double m : mag) EXPECT_NEAR(m, 1.0, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// stepSignal

TEST(StepSignal, CorrectLength) {
    EXPECT_EQ(stepSignal(128).size(), 128u);
    EXPECT_EQ(stepSignal(0).size(),   0u);
}

TEST(StepSignal, BeforePositionIsZero) {
    auto s = stepSignal(20, 8, 1.0);
    for (size_t i = 0; i < 8; ++i) EXPECT_NEAR(s[i], 0.0, 1e-12);
}

TEST(StepSignal, AtAndAfterPositionIsAmplitude) {
    auto s = stepSignal(20, 8, 2.5);
    for (size_t i = 8; i < s.size(); ++i) EXPECT_NEAR(s[i], 2.5, 1e-12);
}

TEST(StepSignal, PositionZeroAllAmplitude) {
    auto s = stepSignal(10, 0, 1.0);
    for (double v : s) EXPECT_NEAR(v, 1.0, 1e-12);
}

TEST(StepSignal, StepIsImpulseCumSum) {
    // Cumulative sum of impulse at position p should equal step at position p
    size_t N = 16, pos = 5;
    auto imp  = impulse(N, pos, 1.0);
    auto step = stepSignal(N, pos, 1.0);
    double acc = 0.0;
    for (size_t i = 0; i < N; ++i) {
        acc += imp[i];
        EXPECT_NEAR(acc, step[i], 1e-12);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// squareWave

TEST(SquareWave, CorrectLength) {
    EXPECT_EQ(squareWave(100.0, 1000.0, 256).size(), 256u);
    EXPECT_EQ(squareWave(100.0, 1000.0, 0).size(),   0u);
}

TEST(SquareWave, OnlyTwoValues) {
    auto s = squareWave(100.0, 1000.0, 1024, 1.0, 0.5);
    for (double v : s) {
        bool ok = (std::abs(v - 1.0) < 1e-9) || (std::abs(v + 1.0) < 1e-9);
        EXPECT_TRUE(ok);
    }
}

TEST(SquareWave, DutyCycleRespected) {
    // 50% duty cycle on 100 Hz at 1000 Hz ≈ 5 high samples per 10-sample period
    auto s = squareWave(100.0, 1000.0, 1000, 1.0, 0.5);
    size_t high = 0, low = 0;
    for (double v : s) {
        if (v > 0.0) ++high;
        else         ++low;
    }
    EXPECT_NEAR(static_cast<double>(high) / s.size(), 0.5, 0.02);
}

TEST(SquareWave, MeanIsZeroForFiftyPercentDutyCycle) {
    auto s = squareWave(100.0, 1000.0, 1000, 1.0, 0.5);
    double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
    EXPECT_NEAR(mean, 0.0, 0.1);
}
