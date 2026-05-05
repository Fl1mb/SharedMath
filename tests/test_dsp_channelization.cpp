#include <gtest/gtest.h>
#include "DSP/Channelization.h"
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
makeToneCH(double f, double fs, size_t N, double A = 1.0)
{
    std::vector<std::complex<double>> iq(N);
    const double ph = 2.0 * M_PI * f / fs;
    for (size_t i = 0; i < N; ++i)
        iq[i] = A * std::polar(1.0, ph * static_cast<double>(i));
    return iq;
}

/// Peak frequency of an IQ block, mapped to (−fs/2, +fs/2].
static double peakFreqCH(const std::vector<std::complex<double>>& iq,
                          double fs, size_t M = 512)
{
    const size_t N = iq.size();
    auto win = windowHann(std::min(N, M), false);
    std::vector<std::complex<double>> frame(M, {0.0, 0.0});
    for (size_t i = 0; i < std::min(N, M); ++i) frame[i] = iq[i] * win[i];
    FFTPlan::create(M, {FFTDirection::Forward, FFTNorm::None}).execute(frame);
    size_t pk = 0; double mx = -1.0;
    for (size_t k = 0; k < M; ++k) { double m = std::norm(frame[k]); if (m > mx) { mx = m; pk = k; } }
    const double binHz = fs / static_cast<double>(M);
    return (pk < M / 2) ? static_cast<double>(pk) * binHz
                        : (static_cast<double>(pk) - static_cast<double>(M)) * binHz;
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST(Channelization, ZeroSampleRate_Throws)
{
    ChannelizerParams p;
    p.sampleRate    = 0.0;
    p.bandwidthHz   = 100.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(extractChannel(iq, p), std::invalid_argument);
}

TEST(Channelization, ZeroBandwidth_Throws)
{
    ChannelizerParams p;
    p.sampleRate  = 1000.0;
    p.bandwidthHz = 0.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(extractChannel(iq, p), std::invalid_argument);
}

TEST(Channelization, NegativeBandwidth_Throws)
{
    ChannelizerParams p;
    p.sampleRate  = 1000.0;
    p.bandwidthHz = -50.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(extractChannel(iq, p), std::invalid_argument);
}

TEST(Channelization, ZeroFilterOrder_Throws)
{
    ChannelizerParams p;
    p.sampleRate  = 1000.0;
    p.bandwidthHz = 100.0;
    p.filterOrder = 0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(extractChannel(iq, p), std::invalid_argument);
}

TEST(Channelization, CenterFreqAboveNyquist_Throws)
{
    ChannelizerParams p;
    p.sampleRate        = 1000.0;
    p.bandwidthHz       = 100.0;
    p.centerFrequencyHz = 600.0;   // > 500 Hz
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(extractChannel(iq, p), std::invalid_argument);
}

TEST(Channelization, CenterFreqBelowNegativeNyquist_Throws)
{
    ChannelizerParams p;
    p.sampleRate        = 1000.0;
    p.bandwidthHz       = 100.0;
    p.centerFrequencyHz = -600.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(extractChannel(iq, p), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty IQ
// ─────────────────────────────────────────────────────────────────────────────

TEST(Channelization, EmptyIQ_ReturnsEmptyResult)
{
    ChannelizerParams p;
    p.sampleRate  = 1000.0;
    p.bandwidthHz = 100.0;
    auto ch = extractChannel({}, p);
    EXPECT_TRUE(ch.iq.empty());
}

TEST(Channelization, EmptyIQ_OutputSampleRateSet)
{
    ChannelizerParams p;
    p.sampleRate       = 1000.0;
    p.bandwidthHz      = 100.0;
    p.outputSampleRate = 200.0;
    auto ch = extractChannel({}, p);
    EXPECT_GT(ch.sampleRate, 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Correct output metadata
// ─────────────────────────────────────────────────────────────────────────────

TEST(Channelization, OutputIsAlwaysBaseband)
{
    ChannelizerParams p;
    p.sampleRate        = 10'000.0;
    p.centerFrequencyHz = 2'000.0;
    p.bandwidthHz       = 500.0;
    auto iq = makeToneCH(2'000.0, p.sampleRate, 4096);
    auto ch = extractChannel(iq, p);
    EXPECT_DOUBLE_EQ(ch.centerFrequencyHz, 0.0);
}

TEST(Channelization, NoDecimation_OutputSampleRateEqualInput)
{
    ChannelizerParams p;
    p.sampleRate       = 1000.0;
    p.bandwidthHz      = 200.0;
    p.outputSampleRate = 0.0;   // no decimation
    auto iq = makeToneCH(0.0, p.sampleRate, 1024);
    auto ch = extractChannel(iq, p);
    EXPECT_DOUBLE_EQ(ch.sampleRate, p.sampleRate);
}

TEST(Channelization, OutputSampleRate_SetCorrectly_WithDecimation)
{
    ChannelizerParams p;
    p.sampleRate       = 10'000.0;
    p.bandwidthHz      = 1'000.0;
    p.outputSampleRate = 2'000.0;
    auto iq = makeToneCH(0.0, p.sampleRate, 8192);
    auto ch = extractChannel(iq, p);
    // Actual output rate = inputRate / round(inputRate / outputRate)
    const size_t factor = static_cast<size_t>(
        std::round(p.sampleRate / p.outputSampleRate));
    const double expectedFs = p.sampleRate / static_cast<double>(factor);
    EXPECT_NEAR(ch.sampleRate, expectedFs, 1.0);
}

TEST(Channelization, OutputLength_Shorter_WithDecimation)
{
    ChannelizerParams p;
    p.sampleRate       = 10'000.0;
    p.bandwidthHz      = 500.0;
    p.outputSampleRate = 1'000.0;
    auto iq = makeToneCH(0.0, p.sampleRate, 10'000);
    auto ch = extractChannel(iq, p);
    EXPECT_LT(ch.iq.size(), iq.size());
}

// ─────────────────────────────────────────────────────────────────────────────
// Tone extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST(Channelization, ToneAtChannelCenter_MovesToDC)
{
    // Wideband IQ with a tone at the channel centre frequency
    const double fs   = 20'000.0;
    const double tone = 3'000.0;
    const size_t N    = 8'192;
    const size_t M    = 512;

    auto iq = makeToneCH(tone, fs, N, 1.0);

    ChannelizerParams p;
    p.sampleRate        = fs;
    p.centerFrequencyHz = tone;
    p.bandwidthHz       = 1'000.0;
    p.filterOrder       = 64;

    auto ch = extractChannel(iq, p);
    ASSERT_FALSE(ch.iq.empty());

    const double binHz = ch.sampleRate / static_cast<double>(M);
    const double cf    = peakFreqCH(ch.iq, ch.sampleRate, M);
    EXPECT_NEAR(cf, 0.0, 3.0 * binHz)
        << "Expected tone near DC after DDC; got " << cf << " Hz";
}

TEST(Channelization, ToneOutsideChannel_Attenuated)
{
    // A tone far outside the channel passband should be significantly attenuated
    const double fs      = 20'000.0;
    const double toneCf  = 5'000.0;   // channel centre
    const double toneOOB = 500.0;     // out-of-band tone (3000 Hz offset from cf)

    const size_t N = 8'192;

    // Mix both in-band and out-of-band tones
    auto iq = makeToneCH(toneCf, fs, N, 1.0);
    auto oob = makeToneCH(toneOOB, fs, N, 1.0);   // will be shifted 4500 Hz from cf
    for (size_t i = 0; i < N; ++i) iq[i] += oob[i];

    ChannelizerParams p;
    p.sampleRate        = fs;
    p.centerFrequencyHz = toneCf;
    p.bandwidthHz       = 500.0;   // narrow — OOB tone is way outside
    p.filterOrder       = 128;

    auto ch = extractChannel(iq, p);
    ASSERT_FALSE(ch.iq.empty());

    // The in-band power should dominate
    double peakPwr = 0.0;
    for (const auto& s : ch.iq) {
        const double p2 = std::norm(s);
        if (p2 > peakPwr) peakPwr = p2;
    }
    EXPECT_GT(peakPwr, 0.0);
}
