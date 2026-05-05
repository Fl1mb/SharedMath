#include <gtest/gtest.h>
#include "DSP/Waterfall.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<std::complex<double>>
makeToneWF(double f, double fs, size_t N, double A = 1.0)
{
    std::vector<std::complex<double>> iq(N);
    const double ph = 2.0 * M_PI * f / fs;
    for (size_t i = 0; i < N; ++i)
        iq[i] = A * std::polar(1.0, ph * static_cast<double>(i));
    return iq;
}

/// Expected number of frames for a given IQ length, fftSize and overlap.
static size_t expectedFrames(size_t N, size_t M, double overlap)
{
    const size_t step = std::max<size_t>(1,
        static_cast<size_t>(std::round(static_cast<double>(M) * (1.0 - overlap))));
    if (N < M) return 0;
    return (N - M) / step + 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty IQ
// ─────────────────────────────────────────────────────────────────────────────

TEST(Waterfall, EmptyIQ_ReturnsEmptyResult)
{
    WaterfallParams p;
    auto wf = computeWaterfall({}, p);
    EXPECT_TRUE(wf.powerDb.empty());
    EXPECT_TRUE(wf.timeAxisSec.empty());
    // Frequency axis is not filled for empty IQ (no frames processed)
    EXPECT_TRUE(wf.frequencyAxisHz.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST(Waterfall, ZeroSampleRate_Throws)
{
    WaterfallParams p;
    p.sampleRate = 0.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(computeWaterfall(iq, p), std::invalid_argument);
}

TEST(Waterfall, NegativeSampleRate_Throws)
{
    WaterfallParams p;
    p.sampleRate = -1000.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(computeWaterfall(iq, p), std::invalid_argument);
}

TEST(Waterfall, ZeroFftSize_Throws)
{
    WaterfallParams p;
    p.fftSize = 0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(computeWaterfall(iq, p), std::invalid_argument);
}

TEST(Waterfall, OverlapOne_Throws)
{
    WaterfallParams p;
    p.overlap = 1.0;
    std::vector<std::complex<double>> iq(1024, {1.0, 0.0});
    EXPECT_THROW(computeWaterfall(iq, p), std::invalid_argument);
}

TEST(Waterfall, NegativeOverlap_Throws)
{
    WaterfallParams p;
    p.overlap = -0.1;
    std::vector<std::complex<double>> iq(1024, {1.0, 0.0});
    EXPECT_THROW(computeWaterfall(iq, p), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Dimensions
// ─────────────────────────────────────────────────────────────────────────────

TEST(Waterfall, FrequencyAxisLength_EqualsFftSize)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 64;
    p.overlap    = 0.5;
    auto iq = makeToneWF(100.0, p.sampleRate, 4096);
    auto wf = computeWaterfall(iq, p);
    EXPECT_EQ(wf.frequencyAxisHz.size(), p.fftSize);
}

TEST(Waterfall, TimeAxisLength_MatchesNumberOfFrames)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 64;
    p.overlap    = 0.5;
    const size_t N = 4096;
    auto iq = makeToneWF(100.0, p.sampleRate, N);
    auto wf = computeWaterfall(iq, p);

    const size_t expected = expectedFrames(N, p.fftSize, p.overlap);
    EXPECT_EQ(wf.timeAxisSec.size(), expected);
}

TEST(Waterfall, PowerDbMatrix_Rows_MatchTimeAxis)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 64;
    p.overlap    = 0.5;
    auto iq = makeToneWF(100.0, p.sampleRate, 4096);
    auto wf = computeWaterfall(iq, p);

    ASSERT_EQ(wf.powerDb.size(), wf.timeAxisSec.size());
    for (const auto& row : wf.powerDb)
        EXPECT_EQ(row.size(), p.fftSize);
}

// ─────────────────────────────────────────────────────────────────────────────
// Frequency axis
// ─────────────────────────────────────────────────────────────────────────────

TEST(Waterfall, CenteredFreqAxis_StartsNearNegativeNyquist)
{
    WaterfallParams p;
    p.sampleRate = 2000.0;
    p.fftSize    = 64;
    p.overlap    = 0.0;
    p.centered   = true;

    auto iq = makeToneWF(0.0, p.sampleRate, 4096);
    auto wf = computeWaterfall(iq, p);

    const double binHz = p.sampleRate / static_cast<double>(p.fftSize);
    EXPECT_NEAR(wf.frequencyAxisHz.front(),
                -(static_cast<double>(p.fftSize / 2)) * binHz, 1e-9);
}

TEST(Waterfall, CenteredFreqAxis_MonotonicallyIncreasing)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 32;
    p.centered   = true;
    auto iq = makeToneWF(100.0, p.sampleRate, 2048);
    auto wf = computeWaterfall(iq, p);

    for (size_t k = 1; k < wf.frequencyAxisHz.size(); ++k)
        EXPECT_GT(wf.frequencyAxisHz[k], wf.frequencyAxisHz[k - 1]);
}

TEST(Waterfall, UncenteredFreqAxis_StartsAtZero)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 64;
    p.centered   = false;
    auto iq = makeToneWF(100.0, p.sampleRate, 4096);
    auto wf = computeWaterfall(iq, p);

    EXPECT_DOUBLE_EQ(wf.frequencyAxisHz.front(), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Time axis
// ─────────────────────────────────────────────────────────────────────────────

TEST(Waterfall, TimeAxis_MonotonicallyIncreasing)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 64;
    p.overlap    = 0.5;
    auto iq = makeToneWF(100.0, p.sampleRate, 4096);
    auto wf = computeWaterfall(iq, p);

    ASSERT_GT(wf.timeAxisSec.size(), 1u);
    for (size_t i = 1; i < wf.timeAxisSec.size(); ++i)
        EXPECT_GT(wf.timeAxisSec[i], wf.timeAxisSec[i - 1]);
}

TEST(Waterfall, TimeAxis_FirstEntry_IsFrameCenter)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 64;
    p.overlap    = 0.0;
    auto iq = makeToneWF(100.0, p.sampleRate, 4096);
    auto wf = computeWaterfall(iq, p);

    ASSERT_FALSE(wf.timeAxisSec.empty());
    const double expectedFirst =
        static_cast<double>(p.fftSize / 2) / p.sampleRate;
    EXPECT_NEAR(wf.timeAxisSec.front(), expectedFirst, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral content
// ─────────────────────────────────────────────────────────────────────────────

TEST(Waterfall, ToneAppearsNearExpectedFrequencyBin_Centered)
{
    const double fs   = 10'000.0;
    const double tone = 2'000.0;
    const size_t M    = 256;
    const size_t N    = 8'192;

    WaterfallParams p;
    p.sampleRate = fs;
    p.fftSize    = M;
    p.overlap    = 0.0;
    p.centered   = true;

    auto iq = makeToneWF(tone, fs, N);
    auto wf = computeWaterfall(iq, p);

    ASSERT_FALSE(wf.powerDb.empty());

    // Find the peak frequency bin across all frames
    const double binHz = fs / static_cast<double>(M);
    size_t peakBin = 0;
    double peakVal = -1e300;
    for (size_t t = 0; t < wf.powerDb.size(); ++t)
        for (size_t k = 0; k < M; ++k)
            if (wf.powerDb[t][k] > peakVal) { peakVal = wf.powerDb[t][k]; peakBin = k; }

    EXPECT_NEAR(wf.frequencyAxisHz[peakBin], tone, 3.0 * binHz)
        << "Peak bin frequency: " << wf.frequencyAxisHz[peakBin]
        << " Hz, expected near " << tone << " Hz";
}

TEST(Waterfall, AllPowerDb_AreFinite)
{
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 64;
    p.overlap    = 0.5;
    auto iq = makeToneWF(100.0, p.sampleRate, 2048);
    auto wf = computeWaterfall(iq, p);
    for (const auto& row : wf.powerDb)
        for (double v : row)
            EXPECT_TRUE(std::isfinite(v));
}

TEST(Waterfall, OverlapZeroVsHalf_MoreFramesWithOverlap)
{
    const size_t N = 4096;
    WaterfallParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 128;

    p.overlap = 0.0;
    auto wf0 = computeWaterfall(makeToneWF(100.0, p.sampleRate, N), p);

    p.overlap = 0.5;
    auto wf05 = computeWaterfall(makeToneWF(100.0, p.sampleRate, N), p);

    EXPECT_GT(wf05.powerDb.size(), wf0.powerDb.size());
}
