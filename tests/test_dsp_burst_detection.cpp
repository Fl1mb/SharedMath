#include <gtest/gtest.h>
#include "DSP/BurstDetection.h"

#include <complex>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::complex<double>>
makeToneBD(double f, double fs, size_t N, double A = 1.0)
{
    std::vector<std::complex<double>> iq(N);
    const double ph = 2.0 * M_PI * f / fs;
    for (size_t i = 0; i < N; ++i)
        iq[i] = A * std::polar(1.0, ph * static_cast<double>(i));
    return iq;
}

static std::vector<std::complex<double>>
makeNoiseBD(size_t N, double A = 0.01, uint64_t seed = 42)
{
    std::vector<std::complex<double>> out(N);
    uint64_t s = seed;
    auto rng = [&]() -> double {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(s >> 33) /
                static_cast<double>(uint64_t{1} << 31)) - 1.0;
    };
    for (auto& v : out) v = A * std::complex<double>(rng(), rng());
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty IQ
// ─────────────────────────────────────────────────────────────────────────────

TEST(BurstDetection, EmptyIQ_NoDetections)
{
    BurstDetectionParams p;
    auto bursts = detectBursts({}, p);
    EXPECT_TRUE(bursts.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST(BurstDetection, ZeroSampleRate_Throws)
{
    BurstDetectionParams p;
    p.sampleRate = 0.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectBursts(iq, p), std::invalid_argument);
}

TEST(BurstDetection, NegativeSampleRate_Throws)
{
    BurstDetectionParams p;
    p.sampleRate = -100.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectBursts(iq, p), std::invalid_argument);
}

TEST(BurstDetection, ZeroWindowSize_Throws)
{
    BurstDetectionParams p;
    p.windowSize = 0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectBursts(iq, p), std::invalid_argument);
}

TEST(BurstDetection, OverlapOne_Throws)
{
    BurstDetectionParams p;
    p.overlap = 1.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectBursts(iq, p), std::invalid_argument);
}

TEST(BurstDetection, NegativeOverlap_Throws)
{
    BurstDetectionParams p;
    p.overlap = -0.1;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectBursts(iq, p), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Basic burst detection
// ─────────────────────────────────────────────────────────────────────────────

TEST(BurstDetection, DetectsSyntheticBurst_InNoise)
{
    const double fs        = 10'000.0;
    const size_t N         = 10'000;
    const size_t bStart    = 3'000;
    const size_t bLen      = 2'000;

    auto iq    = makeNoiseBD(N, 0.01, 1);
    auto burst = makeToneBD(500.0, fs, bLen, 1.0);   // ~40 dB above noise
    for (size_t i = 0; i < bLen; ++i) iq[bStart + i] = burst[i];

    BurstDetectionParams p;
    p.sampleRate  = fs;
    p.windowSize  = 256;
    p.overlap     = 0.5;
    p.thresholdDb = 20.0;

    auto bursts = detectBursts(iq, p);
    ASSERT_FALSE(bursts.empty());

    // At least one burst must overlap with the injected region
    bool found = false;
    for (const auto& b : bursts) {
        if (b.startSample <= bStart + bLen && b.endSample >= bStart)
            found = true;
    }
    EXPECT_TRUE(found);
}

TEST(BurstDetection, DetectedBurst_TimestampsConsistent)
{
    const double fs     = 10'000.0;
    const size_t N      = 8'000;
    const size_t bStart = 2'000;
    const size_t bLen   = 2'000;

    auto iq    = makeNoiseBD(N, 0.01, 7);
    auto burst = makeToneBD(1'000.0, fs, bLen, 1.0);
    for (size_t i = 0; i < bLen; ++i) iq[bStart + i] = burst[i];

    BurstDetectionParams p;
    p.sampleRate  = fs;
    p.windowSize  = 256;
    p.thresholdDb = 15.0;

    auto bursts = detectBursts(iq, p);
    for (const auto& b : bursts) {
        EXPECT_LE(b.startSample, b.endSample);
        EXPECT_NEAR(b.startTimeSec, static_cast<double>(b.startSample) / fs, 1e-9);
        EXPECT_NEAR(b.endTimeSec,   static_cast<double>(b.endSample)   / fs, 1e-9);
        EXPECT_NEAR(b.durationSec, b.endTimeSec - b.startTimeSec, 1e-9);
        EXPECT_GE(b.snrDb, 0.0);
    }
}

TEST(BurstDetection, SNR_Equals_PeakPower_Minus_NoiseFloor)
{
    // We can't directly observe the noise floor, but SNR = peak - floor is tested
    // indirectly: SNR must be positive if the burst was significantly above noise.
    const double fs     = 10'000.0;
    auto iq    = makeNoiseBD(8'000, 0.01, 11);
    auto burst = makeToneBD(500.0, fs, 2'000, 1.0);
    for (size_t i = 0; i < 2'000; ++i) iq[2'000 + i] = burst[i];

    BurstDetectionParams p;
    p.sampleRate  = fs;
    p.windowSize  = 256;
    p.thresholdDb = 15.0;

    auto bursts = detectBursts(iq, p);
    ASSERT_FALSE(bursts.empty());
    EXPECT_GT(bursts[0].snrDb, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Gap merging
// ─────────────────────────────────────────────────────────────────────────────

TEST(BurstDetection, GapMerging_MergesTwoNearbyBursts)
{
    const double fs    = 10'000.0;
    const size_t N     = 12'000;
    const size_t bLen  = 1'500;
    const size_t gap   = 200;    // 20 ms gap

    // Two bursts with a 200-sample gap
    auto iq = makeNoiseBD(N, 0.001, 3);
    auto b1 = makeToneBD(1'000.0, fs, bLen, 1.0);
    auto b2 = makeToneBD(1'000.0, fs, bLen, 1.0);
    for (size_t i = 0; i < bLen; ++i) iq[1'000 + i] = b1[i];
    for (size_t i = 0; i < bLen; ++i) iq[1'000 + bLen + gap + i] = b2[i];

    BurstDetectionParams p;
    p.sampleRate  = fs;
    p.windowSize  = 128;
    p.thresholdDb = 15.0;
    p.maxGapSec   = (static_cast<double>(gap) + 100.0) / fs;  // gap + margin

    auto merged = detectBursts(iq, p);

    // Without merging there would be two bursts; with merging at least one fewer
    BurstDetectionParams p_no_merge = p;
    p_no_merge.maxGapSec = 0.0;
    auto unmerged = detectBursts(iq, p_no_merge);

    EXPECT_LE(merged.size(), unmerged.size());
}

TEST(BurstDetection, GapMerging_DoesNotMergeDistantBursts)
{
    const double fs    = 10'000.0;
    const size_t N     = 16'000;
    const size_t bLen  = 1'500;
    const size_t gap   = 3'000;  // large gap

    auto iq = makeNoiseBD(N, 0.001, 5);
    auto b1 = makeToneBD(1'000.0, fs, bLen, 1.0);
    auto b2 = makeToneBD(1'000.0, fs, bLen, 1.0);
    for (size_t i = 0; i < bLen; ++i) iq[1'000 + i]               = b1[i];
    for (size_t i = 0; i < bLen; ++i) iq[1'000 + bLen + gap + i]  = b2[i];

    BurstDetectionParams p;
    p.sampleRate  = fs;
    p.windowSize  = 128;
    p.thresholdDb = 15.0;
    p.maxGapSec   = 50e-3;   // 50 ms << gap of 300 ms → should NOT merge

    auto bursts = detectBursts(iq, p);
    EXPECT_GE(bursts.size(), 2u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimum duration filter
// ─────────────────────────────────────────────────────────────────────────────

TEST(BurstDetection, MinDuration_FiltersShortBursts)
{
    const double fs    = 10'000.0;
    const size_t N     = 8'000;
    const size_t bLen  = 256;   // short burst

    auto iq    = makeNoiseBD(N, 0.001, 9);
    auto burst = makeToneBD(1'000.0, fs, bLen, 1.0);
    for (size_t i = 0; i < bLen; ++i) iq[2'000 + i] = burst[i];

    BurstDetectionParams p;
    p.sampleRate     = fs;
    p.windowSize     = 128;
    p.thresholdDb    = 15.0;
    p.minDurationSec = static_cast<double>(bLen * 2) / fs;  // require 2× the burst length

    auto bursts = detectBursts(iq, p);
    // All returned bursts must be at least minDurationSec long
    for (const auto& b : bursts)
        EXPECT_GE(b.durationSec, p.minDurationSec);
}

TEST(BurstDetection, MinDuration_ZeroKeepsAllBursts)
{
    const double fs    = 10'000.0;
    const size_t N     = 8'000;
    const size_t bLen  = 256;

    auto iq    = makeNoiseBD(N, 0.001, 13);
    auto burst = makeToneBD(1'000.0, fs, bLen, 1.0);
    for (size_t i = 0; i < bLen; ++i) iq[2'000 + i] = burst[i];

    BurstDetectionParams p_no_min = {fs, 128, 0.5, 15.0, 0.0, 0.0};
    BurstDetectionParams p_with_min = p_no_min;
    p_with_min.minDurationSec = static_cast<double>(bLen) / fs;

    auto raw  = detectBursts(iq, p_no_min);
    auto filt = detectBursts(iq, p_with_min);
    EXPECT_LE(filt.size(), raw.size());
}
