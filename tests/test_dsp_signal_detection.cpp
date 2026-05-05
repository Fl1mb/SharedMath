#include <gtest/gtest.h>
#include "DSP/SignalDetection.h"

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a complex tone: A·exp(j·2π·f·n / fs) for n = 0 … N−1.
static std::vector<std::complex<double>>
makeTone(double freqHz, double sampleRate, size_t N, double amplitude = 1.0)
{
    std::vector<std::complex<double>> iq(N);
    const double phaseInc = 2.0 * M_PI * freqHz / sampleRate;
    for (size_t i = 0; i < N; ++i)
        iq[i] = amplitude * std::polar(1.0, phaseInc * static_cast<double>(i));
    return iq;
}

/// Generate deterministic complex uniform noise via a simple LCG.
static std::vector<std::complex<double>>
makeNoise(size_t N, double amplitude = 0.01, uint64_t seed = 42)
{
    std::vector<std::complex<double>> out(N);
    uint64_t s = seed;
    auto rng = [&]() -> double {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (static_cast<double>(s >> 33) /
                static_cast<double>(uint64_t{1} << 31)) - 1.0;
    };
    for (auto& v : out) v = amplitude * std::complex<double>(rng(), rng());
    return out;
}

/// Generate a low-ambiguity complex reference sequence for matched-filter tests.
static std::vector<std::complex<double>>
makeReference(size_t N, uint64_t seed = 123)
{
    std::vector<std::complex<double>> ref(N);
    uint64_t s = seed;
    for (auto& v : ref) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const double re = ((s >> 63) != 0) ? 1.0 : -1.0;
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const double im = ((s >> 63) != 0) ? 1.0 : -1.0;
        v = std::complex<double>(re, im) / std::sqrt(2.0);
    }
    return ref;
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty IQ: all methods must return no detections without throwing
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, EmptyIQ_TimeDomain_NoDetections)
{
    SignalDetectionParams p;
    auto result = detectSignals({}, p, DetectionMethod::EnergyTimeDomain);
    EXPECT_TRUE(result.detections.empty());
}

TEST(SignalDetection, EmptyIQ_Spectral_NoDetections)
{
    SignalDetectionParams p;
    auto result = detectSignals({}, p, DetectionMethod::EnergySpectral);
    EXPECT_TRUE(result.detections.empty());
    EXPECT_TRUE(result.frequencyAxisHz.empty());
    EXPECT_TRUE(result.spectrumDb.empty());
}

TEST(SignalDetection, EmptyIQ_MatchedFilter_NoDetections)
{
    std::vector<std::complex<double>> ref = {{1.0, 0.0}};
    SignalDetectionParams p;
    auto result = detectMatchedFilter({}, ref, p);
    EXPECT_TRUE(result.detections.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Invalid parameters must throw std::invalid_argument
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, InvalidSampleRate_Throws)
{
    SignalDetectionParams p;
    p.sampleRate = 0.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergyTimeDomain),
                 std::invalid_argument);
}

TEST(SignalDetection, NegativeSampleRate_Throws)
{
    SignalDetectionParams p;
    p.sampleRate = -1000.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergySpectral),
                 std::invalid_argument);
}

TEST(SignalDetection, ZeroFftSize_Throws)
{
    SignalDetectionParams p;
    p.fftSize = 0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergyTimeDomain),
                 std::invalid_argument);
}

TEST(SignalDetection, OverlapEqualOne_Throws)
{
    SignalDetectionParams p;
    p.overlap = 1.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergyTimeDomain),
                 std::invalid_argument);
}

TEST(SignalDetection, NegativeOverlap_Throws)
{
    SignalDetectionParams p;
    p.overlap = -0.1;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergyTimeDomain),
                 std::invalid_argument);
}

TEST(SignalDetection, NegativeBandwidth_Throws)
{
    SignalDetectionParams p;
    p.bandwidthHz = -1.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergyTimeDomain),
                 std::invalid_argument);
}

TEST(SignalDetection, CenterFreqAboveNyquist_Throws)
{
    SignalDetectionParams p;
    p.sampleRate         = 1000.0;
    p.centerFrequencyHz  = 600.0;  // > 500 Hz (Nyquist)
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergyTimeDomain),
                 std::invalid_argument);
}

TEST(SignalDetection, CenterFreqBelowNegativeNyquist_Throws)
{
    SignalDetectionParams p;
    p.sampleRate        = 1000.0;
    p.centerFrequencyHz = -600.0;
    std::vector<std::complex<double>> iq = {{1.0, 0.0}};
    EXPECT_THROW(detectSignals(iq, p, DetectionMethod::EnergySpectral),
                 std::invalid_argument);
}

TEST(SignalDetection, MatchedFilter_EmptyReference_Throws)
{
    SignalDetectionParams p;
    std::vector<std::complex<double>> iq(100, {1.0, 0.0});
    EXPECT_THROW(detectMatchedFilter(iq, {}, p), std::invalid_argument);
}

TEST(SignalDetection, MatchedFilter_RefLongerThanIQ_Throws)
{
    SignalDetectionParams p;
    std::vector<std::complex<double>> iq(10,  {1.0, 0.0});
    std::vector<std::complex<double>> ref(20, {1.0, 0.0});
    EXPECT_THROW(detectMatchedFilter(iq, ref, p), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral detector: frequency axis
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, SpectralFrequencyAxisSize)
{
    SignalDetectionParams p;
    p.sampleRate  = 2000.0;
    p.fftSize     = 64;
    p.thresholdDb = 200.0;  // suppress detections; still builds the axis

    auto iq     = makeTone(100.0, p.sampleRate, 4096);
    auto result = detectSpectral(iq, p);

    ASSERT_EQ(result.frequencyAxisHz.size(), p.fftSize);
    ASSERT_EQ(result.spectrumDb.size(),      p.fftSize);
}

TEST(SignalDetection, SpectralFrequencyAxisEndpoints)
{
    SignalDetectionParams p;
    p.sampleRate  = 2000.0;
    p.fftSize     = 64;
    p.thresholdDb = 200.0;

    auto iq     = makeTone(100.0, p.sampleRate, 4096);
    auto result = detectSpectral(iq, p);

    const double binHz = p.sampleRate / static_cast<double>(p.fftSize);

    // Leftmost bin: -(M/2)*binHz
    EXPECT_NEAR(result.frequencyAxisHz.front(),
                -(static_cast<double>(p.fftSize / 2)) * binHz, 1e-9);
    // Rightmost bin: +(M/2 - 1)*binHz  (symmetric two-sided spectrum)
    EXPECT_NEAR(result.frequencyAxisHz.back(),
                (static_cast<double>(p.fftSize / 2) - 1.0) * binHz, 1e-9);
}

TEST(SignalDetection, SpectralFrequencyAxis_MonotonicallyIncreasing)
{
    SignalDetectionParams p;
    p.sampleRate  = 1000.0;
    p.fftSize     = 32;
    p.thresholdDb = 200.0;

    auto iq     = makeTone(100.0, p.sampleRate, 2048);
    auto result = detectSpectral(iq, p);

    for (size_t k = 1; k < result.frequencyAxisHz.size(); ++k)
        EXPECT_GT(result.frequencyAxisHz[k], result.frequencyAxisHz[k - 1]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Time-domain: constant IQ power
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, ConstantIQ_PowerDb_Correct)
{
    // Constant IQ = (A, 0): mean power = A², powerDb ≈ 10·log10(A²)
    const double A = 2.0;
    const size_t N = 2048;
    std::vector<std::complex<double>> iq(N, {A, 0.0});

    SignalDetectionParams p;
    p.fftSize            = 256;
    p.thresholdDb        = -200.0;  // detect everything
    p.estimateNoiseFloor = false;

    auto result = detectEnergyTimeDomain(iq, p);
    ASSERT_FALSE(result.detections.empty());

    const double expectedDb = 10.0 * std::log10(A * A);
    EXPECT_NEAR(result.detections[0].powerDb, expectedDb, 0.5);
}

// ─────────────────────────────────────────────────────────────────────────────
// Time-domain energy detector: burst in noise
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, TimeDomain_DetectsBurstInNoise)
{
    const double fs        = 10'000.0;
    const size_t N         = 10'000;
    const size_t burstStart = 3'000;
    const size_t burstLen   = 2'000;

    auto iq    = makeNoise(N, 0.01, 1);
    auto burst = makeTone(500.0, fs, burstLen, 1.0);   // 40 dB above noise
    for (size_t i = 0; i < burstLen; ++i)
        iq[burstStart + i] = burst[i];

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.fftSize     = 256;
    p.overlap     = 0.5;
    p.thresholdDb = 20.0;

    auto result = detectSignals(iq, p, DetectionMethod::EnergyTimeDomain);

    ASSERT_FALSE(result.detections.empty());
    const auto& d = result.detections[0];

    // The detection must overlap with the burst
    EXPECT_LE(d.startSample, burstStart + p.fftSize);
    EXPECT_GE(d.endSample,   burstStart);
    EXPECT_GT(d.snrDb,   0.0);
    EXPECT_GE(d.confidence, 0.0);
    EXPECT_LE(d.confidence, 1.0);
    EXPECT_GT(d.endTimeSec,  d.startTimeSec);
}

TEST(SignalDetection, TimeDomain_SNRAndNoiseFloorConsistent)
{
    const double fs      = 10'000.0;
    const size_t N       = 8'000;
    const size_t bStart  = 2'000;
    const size_t bLen    = 2'000;

    auto iq    = makeNoise(N, 0.01, 7);
    auto burst = makeTone(1000.0, fs, bLen, 1.0);
    for (size_t i = 0; i < bLen; ++i) iq[bStart + i] = burst[i];

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.fftSize     = 256;
    p.thresholdDb = 15.0;

    auto result = detectEnergyTimeDomain(iq, p);
    ASSERT_FALSE(result.detections.empty());

    for (const auto& d : result.detections)
        EXPECT_NEAR(d.snrDb, d.powerDb - d.noiseFloorDb, 1e-9);
}

TEST(SignalDetection, TimeDomain_MinDurationFiltersShortBurst)
{
    const double fs         = 10'000.0;
    const size_t N          = 6'000;
    const size_t burstStart = 2'000;
    const size_t burstLen   = 300;

    auto iq    = makeNoise(N, 0.01, 17);
    auto burst = makeTone(750.0, fs, burstLen, 1.0);
    for (size_t i = 0; i < burstLen; ++i) iq[burstStart + i] = burst[i];

    SignalDetectionParams p;
    p.sampleRate     = fs;
    p.fftSize        = 128;
    p.thresholdDb    = 15.0;
    p.minDurationSec = 0.1;

    auto result = detectEnergyTimeDomain(iq, p);
    EXPECT_TRUE(result.detections.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectral detector: single complex tone
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, Spectral_FindsComplexTone)
{
    const double fs   = 20'000.0;
    const double tone = 3'000.0;
    const size_t N    = 16'384;

    auto iq    = makeTone(tone, fs, N, 1.0);
    auto noise = makeNoise(N, 1e-4, 99);
    for (size_t i = 0; i < N; ++i) iq[i] += noise[i];

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.fftSize     = 1024;
    p.thresholdDb = 15.0;

    auto result = detectSignals(iq, p, DetectionMethod::EnergySpectral);

    ASSERT_FALSE(result.detections.empty());
    EXPECT_EQ(result.frequencyAxisHz.size(), p.fftSize);
    EXPECT_EQ(result.spectrumDb.size(),      p.fftSize);
    EXPECT_LT(result.noiseFloorDb, result.detections[0].powerDb);
}

TEST(SignalDetection, Spectral_CenterFrequencyNearTone)
{
    const double fs   = 20'000.0;
    const double tone = 3'000.0;
    const size_t N    = 16'384;

    auto iq = makeTone(tone, fs, N, 1.0);

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.fftSize     = 1024;
    p.thresholdDb = 10.0;

    auto result = detectSignals(iq, p, DetectionMethod::EnergySpectral);

    ASSERT_FALSE(result.detections.empty());

    const double binHz = fs / static_cast<double>(p.fftSize);
    bool anyClose = false;
    for (const auto& d : result.detections)
        if (std::abs(d.centerFrequencyHz - tone) < 5.0 * binHz)
            anyClose = true;

    EXPECT_TRUE(anyClose)
        << "Expected a detection within 5 bins of " << tone << " Hz; "
        << "first detection cf=" << result.detections[0].centerFrequencyHz << " Hz";
}

TEST(SignalDetection, Spectral_NegativeFreqTone)
{
    // A tone at -2000 Hz is a complex exponential rotating in the opposite direction
    const double fs   = 20'000.0;
    const double tone = -2'000.0;
    const size_t N    = 16'384;

    auto iq = makeTone(tone, fs, N, 1.0);

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.fftSize     = 1024;
    p.thresholdDb = 10.0;

    auto result = detectSpectral(iq, p);

    ASSERT_FALSE(result.detections.empty());
    const double binHz = fs / static_cast<double>(p.fftSize);
    bool anyClose = false;
    for (const auto& d : result.detections)
        if (std::abs(d.centerFrequencyHz - tone) < 5.0 * binHz)
            anyClose = true;

    EXPECT_TRUE(anyClose)
        << "Expected detection near " << tone << " Hz";
}

TEST(SignalDetection, Spectral_SpectrumAndFreqAxisSameSize)
{
    SignalDetectionParams p;
    p.sampleRate = 1000.0;
    p.fftSize    = 128;

    auto iq     = makeTone(100.0, p.sampleRate, 2048);
    auto result = detectSpectral(iq, p);

    EXPECT_EQ(result.frequencyAxisHz.size(), result.spectrumDb.size());
    EXPECT_EQ(result.frequencyAxisHz.size(), p.fftSize);
}

TEST(SignalDetection, Spectral_BandwidthAndCenterRestrictSearchBand)
{
    const double fs = 20'000.0;
    const size_t N  = 16'384;

    auto iq = makeTone(1'000.0, fs, N, 0.5);
    auto outOfBand = makeTone(4'000.0, fs, N, 1.0);
    for (size_t i = 0; i < N; ++i) iq[i] += outOfBand[i];

    SignalDetectionParams p;
    p.sampleRate        = fs;
    p.fftSize           = 1024;
    p.thresholdDb       = 10.0;
    p.centerFrequencyHz = 1'000.0;
    p.bandwidthHz       = 500.0;
    p.guardBins         = 0;

    auto result = detectSpectral(iq, p);

    ASSERT_FALSE(result.detections.empty());
    const double binHz = fs / static_cast<double>(p.fftSize);
    for (const auto& d : result.detections) {
        EXPECT_NEAR(d.centerFrequencyHz, p.centerFrequencyHz, 3.0 * binHz);
        EXPECT_GT(std::abs(d.centerFrequencyHz - 4'000.0), 2'000.0);
    }
}

TEST(SignalDetection, Spectral_GuardBinsExpandReportedBandwidth)
{
    const double fs   = 20'000.0;
    const double tone = 2'000.0;
    const size_t N    = 16'384;

    auto iq = makeTone(tone, fs, N, 1.0);

    SignalDetectionParams noGuard;
    noGuard.sampleRate  = fs;
    noGuard.fftSize     = 1024;
    noGuard.thresholdDb = 10.0;
    noGuard.guardBins   = 0;

    SignalDetectionParams guarded = noGuard;
    guarded.guardBins = 3;

    auto a = detectSpectral(iq, noGuard);
    auto b = detectSpectral(iq, guarded);

    ASSERT_FALSE(a.detections.empty());
    ASSERT_FALSE(b.detections.empty());
    EXPECT_GT(b.detections[0].bandwidthHz, a.detections[0].bandwidthHz);
}

// ─────────────────────────────────────────────────────────────────────────────
// Matched filter
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, MatchedFilter_DetectsInsertedReference)
{
    const double fs       = 10'000.0;
    const size_t N        = 8'192;
    const size_t refLen   = 256;
    const size_t insertAt = 3'000;

    auto ref = makeReference(refLen, 77);
    auto iq  = makeNoise(N, 0.001, 77);
    for (size_t i = 0; i < refLen; ++i) iq[insertAt + i] = ref[i];

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.thresholdDb = 10.0;

    auto result = detectSignals(iq, p, DetectionMethod::MatchedFilter, ref);

    ASSERT_FALSE(result.detections.empty());

    // At least one detection must describe the inserted reference interval.
    bool found = false;
    for (const auto& d : result.detections) {
        const auto startDiff =
            std::abs(static_cast<long long>(d.startSample) -
                     static_cast<long long>(insertAt));
        const auto endDiff =
            std::abs(static_cast<long long>(d.endSample) -
                     static_cast<long long>(insertAt + refLen - 1));
        if (startDiff <= 2 && endDiff <= 2)
            found = true;
    }
    EXPECT_TRUE(found)
        << "No detection covering [" << insertAt << ", "
        << (insertAt + refLen - 1) << "]; first detection is ["
        << result.detections[0].startSample << ", "
        << result.detections[0].endSample << "]";
}

TEST(SignalDetection, MatchedFilter_ConfidenceInUnitRange)
{
    const double fs     = 10'000.0;
    const size_t refLen = 128;

    auto ref = makeReference(refLen, 1);
    auto iq  = makeNoise(512, 1e-4, 1);
    for (size_t i = 0; i < refLen; ++i) iq[i] = ref[i];   // perfect match at sample 0

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.thresholdDb = -100.0;   // detect everything

    auto result = detectMatchedFilter(iq, ref, p);

    double maxConfidence = 0.0;
    for (const auto& d : result.detections) {
        EXPECT_GE(d.confidence, 0.0);
        EXPECT_LE(d.confidence, 1.0);
        maxConfidence = std::max(maxConfidence, d.confidence);
    }
    EXPECT_NEAR(maxConfidence, 1.0, 1e-9);
}

TEST(SignalDetection, MatchedFilter_TimestampsConsistent)
{
    const double fs     = 8'000.0;
    const size_t refLen = 64;
    const size_t N      = 1'024;

    auto ref = makeReference(refLen, 3);
    auto iq  = makeNoise(N, 0.001, 3);
    for (size_t i = 0; i < refLen; ++i) iq[100 + i] = ref[i];

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.thresholdDb = 5.0;

    auto result = detectMatchedFilter(iq, ref, p);

    for (const auto& d : result.detections) {
        EXPECT_LE(d.startSample, d.endSample);
        EXPECT_DOUBLE_EQ(d.startTimeSec,
                         static_cast<double>(d.startSample) / fs);
        EXPECT_DOUBLE_EQ(d.endTimeSec,
                         static_cast<double>(d.endSample) / fs);
        EXPECT_GE(d.startTimeSec, 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SNR consistency across all methods
// ─────────────────────────────────────────────────────────────────────────────

TEST(SignalDetection, AllMethods_SNR_Equals_Power_Minus_NoiseFloor)
{
    const double fs   = 10'000.0;
    const size_t N    = 4'096;
    const size_t rLen = 64;

    auto ref = makeTone(1'000.0, fs, rLen, 1.0);
    auto iq  = makeNoise(N, 0.001, 55);
    for (size_t i = 0; i < rLen; ++i) iq[500 + i] = ref[i];

    SignalDetectionParams p;
    p.sampleRate  = fs;
    p.fftSize     = 256;
    p.thresholdDb = 10.0;

    for (auto method : {DetectionMethod::EnergyTimeDomain,
                        DetectionMethod::EnergySpectral,
                        DetectionMethod::MatchedFilter})
    {
        auto result = detectSignals(iq, p, method, ref);
        for (const auto& d : result.detections)
            EXPECT_NEAR(d.snrDb, d.powerDb - d.noiseFloorDb, 1e-9);
    }
}
