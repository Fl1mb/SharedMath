#include <gtest/gtest.h>
#include "DSP/FrequencyCorrection.h"
#include "DSP/FFTPlan.h"
#include "DSP/FFTConfig.h"
#include "DSP/Window.h"

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
makeToneFC(double f, double fs, size_t N, double A = 1.0)
{
    std::vector<std::complex<double>> iq(N);
    const double ph = 2.0 * M_PI * f / fs;
    for (size_t i = 0; i < N; ++i)
        iq[i] = A * std::polar(1.0, ph * static_cast<double>(i));
    return iq;
}

/// Return the frequency (Hz) of the peak FFT bin, mapped to (−fs/2, +fs/2].
static double peakFrequencyHz(const std::vector<std::complex<double>>& iq,
                               double fs, size_t M = 1024)
{
    const size_t N = iq.size();
    auto win = windowHann(std::min(N, M), false);
    std::vector<std::complex<double>> frame(M, {0.0, 0.0});
    for (size_t i = 0; i < std::min(N, M); ++i)
        frame[i] = iq[i] * win[i];
    FFTPlan::create(M, {FFTDirection::Forward, FFTNorm::None}).execute(frame);
    size_t pk = 0; double mx = -1.0;
    for (size_t k = 0; k < M; ++k) {
        double m = std::norm(frame[k]);
        if (m > mx) { mx = m; pk = k; }
    }
    const double binHz = fs / static_cast<double>(M);
    return (pk < M / 2) ? static_cast<double>(pk) * binHz
                        : (static_cast<double>(pk) - static_cast<double>(M)) * binHz;
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty IQ
// ─────────────────────────────────────────────────────────────────────────────

TEST(FrequencyCorrection, EmptyIQ_FrequencyShift_ReturnsEmpty)
{
    auto out = frequencyShift({}, 100.0, 1000.0);
    EXPECT_TRUE(out.empty());
}

TEST(FrequencyCorrection, EmptyIQ_CorrectOffset_ReturnsEmpty)
{
    FrequencyCorrectionParams p;
    p.sampleRate = 1000.0;
    auto res = correctFrequencyOffset({}, p);
    EXPECT_TRUE(res.iq.empty());
    EXPECT_DOUBLE_EQ(res.appliedFrequencyOffsetHz, 0.0);
}

TEST(FrequencyCorrection, EmptyIQ_EstimatePeak_ReturnsZero)
{
    EXPECT_DOUBLE_EQ(estimateFrequencyOffsetFromPeak({}, 1000.0), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST(FrequencyCorrection, ZeroSampleRate_FrequencyShift_Throws)
{
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(frequencyShift(iq, 100.0, 0.0), std::invalid_argument);
}

TEST(FrequencyCorrection, NegativeSampleRate_FrequencyShift_Throws)
{
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(frequencyShift(iq, 100.0, -1.0), std::invalid_argument);
}

TEST(FrequencyCorrection, ZeroSampleRate_CorrectOffset_Throws)
{
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    FrequencyCorrectionParams p;
    p.sampleRate = 0.0;
    EXPECT_THROW(correctFrequencyOffset(iq, p), std::invalid_argument);
}

TEST(FrequencyCorrection, ZeroSampleRate_EstimatePeak_Throws)
{
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(estimateFrequencyOffsetFromPeak(iq, 0.0), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// frequencyShift
// ─────────────────────────────────────────────────────────────────────────────

TEST(FrequencyCorrection, FrequencyShift_OutputSameLength)
{
    auto iq  = makeToneFC(100.0, 1000.0, 512);
    auto out = frequencyShift(iq, 50.0, 1000.0);
    EXPECT_EQ(out.size(), iq.size());
}

TEST(FrequencyCorrection, FrequencyShift_PreservesAmplitude)
{
    // Multiplying by a unit complex exponential must not change |x|
    const double A = 2.0;
    auto iq  = makeToneFC(100.0, 1000.0, 256, A);
    auto out = frequencyShift(iq, 50.0, 1000.0);
    for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(std::abs(out[i]), std::abs(iq[i]), 1e-10);
}

TEST(FrequencyCorrection, FrequencyShift_MovesToneByExpectedAmount)
{
    // Tone at +500 Hz shifted by +200 Hz should appear near +700 Hz
    const double fs   = 4'000.0;
    const double tone = 500.0;
    const double shift = 200.0;
    const size_t N    = 4096;
    const size_t M    = 1024;

    auto iq  = makeToneFC(tone, fs, N);
    auto out = frequencyShift(iq, shift, fs);

    const double binHz = fs / static_cast<double>(M);
    EXPECT_NEAR(peakFrequencyHz(out, fs, M), tone + shift, 2.0 * binHz);
}

TEST(FrequencyCorrection, FrequencyShift_NegativeShift_MovesToDC)
{
    // Shifting a tone at f by -f should bring it near DC
    const double fs   = 8'000.0;
    const double tone = 1'500.0;
    const size_t N    = 8192;
    const size_t M    = 2048;

    auto iq  = makeToneFC(tone, fs, N);
    auto out = frequencyShift(iq, -tone, fs);

    const double binHz = fs / static_cast<double>(M);
    EXPECT_NEAR(peakFrequencyHz(out, fs, M), 0.0, 2.0 * binHz);
}

// ─────────────────────────────────────────────────────────────────────────────
// estimateFrequencyOffsetFromPeak
// ─────────────────────────────────────────────────────────────────────────────

TEST(FrequencyCorrection, EstimatePeak_PositiveTone)
{
    const double fs   = 10'000.0;
    const double tone = 2'000.0;
    auto iq = makeToneFC(tone, fs, 4096);

    const double est   = estimateFrequencyOffsetFromPeak(iq, fs, 1024);
    const double binHz = fs / 1024.0;
    EXPECT_NEAR(est, tone, 2.0 * binHz);
}

TEST(FrequencyCorrection, EstimatePeak_NegativeTone)
{
    const double fs   = 10'000.0;
    const double tone = -2'000.0;
    auto iq = makeToneFC(tone, fs, 4096);

    const double est   = estimateFrequencyOffsetFromPeak(iq, fs, 1024);
    const double binHz = fs / 1024.0;
    EXPECT_NEAR(est, tone, 2.0 * binHz);
}

// ─────────────────────────────────────────────────────────────────────────────
// correctFrequencyOffset
// ─────────────────────────────────────────────────────────────────────────────

TEST(FrequencyCorrection, CorrectOffset_RemovesTone)
{
    const double fs   = 8'000.0;
    const double tone = 1'000.0;
    const size_t N    = 8192;
    const size_t M    = 2048;

    auto iq = makeToneFC(tone, fs, N);

    FrequencyCorrectionParams p;
    p.sampleRate        = fs;
    p.frequencyOffsetHz = tone;   // remove the offset

    auto res = correctFrequencyOffset(iq, p);
    ASSERT_EQ(res.iq.size(), N);

    const double binHz = fs / static_cast<double>(M);
    EXPECT_NEAR(peakFrequencyHz(res.iq, fs, M), 0.0, 2.0 * binHz);
}

TEST(FrequencyCorrection, CorrectOffset_AppliedFieldMatchesParam)
{
    FrequencyCorrectionParams p;
    p.sampleRate        = 1000.0;
    p.frequencyOffsetHz = 123.4;
    std::vector<std::complex<double>> iq(64, {1.0, 0.0});
    auto res = correctFrequencyOffset(iq, p);
    EXPECT_DOUBLE_EQ(res.appliedFrequencyOffsetHz, 123.4);
}

TEST(FrequencyCorrection, CorrectOffset_FinalPhaseIsFinite)
{
    FrequencyCorrectionParams p;
    p.sampleRate        = 1000.0;
    p.frequencyOffsetHz = 100.0;
    std::vector<std::complex<double>> iq(256, {1.0, 0.0});
    auto res = correctFrequencyOffset(iq, p);
    EXPECT_TRUE(std::isfinite(res.finalPhaseRad));
}
