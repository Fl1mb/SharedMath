#include <gtest/gtest.h>
#include "DSP/STFT.h"
#include "DSP/Window.h"

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

// Build a test signal of length n covering kFrames full STFT frames exactly:
//   n = fftSize + (kFrames - 1) * hopSize
static std::vector<double> makeSignal(size_t n) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = std::sin(2.0 * kPi * 5.0 * static_cast<double>(i) / static_cast<double>(n))
             + 0.5 * std::cos(2.0 * kPi * 13.0 * static_cast<double>(i) / static_cast<double>(n));
    return x;
}

// Hamming window (periodic = symmetric=false) satisfies the COLA condition
// with hopSize = fftSize/2: sum of two adjacent half-shifted windows = 1.08 everywhere.
static std::vector<double> colaWindow(size_t fftSize) {
    return windowHamming(fftSize, /*symmetric=*/false);
}

// ═════════════════════════════════════════════════════════════════════════════
// STFTAnalysis — frame counts, bin counts, metadata
// ═════════════════════════════════════════════════════════════════════════════

TEST(STFTAnalysis, NumFrames) {
    size_t fftSize = 256, hopSize = 128;
    // Choose N so exactly 16 full frames fit: N = fftSize + 15*hopSize
    size_t N = fftSize + 15 * hopSize;  // = 2176
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win);
    EXPECT_EQ(res.numFrames(), 16u);
}

TEST(STFTAnalysis, NumBins) {
    size_t fftSize = 256, hopSize = 64;
    size_t N = fftSize + 7 * hopSize;
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win);
    EXPECT_EQ(res.numBins(), fftSize / 2 + 1);
    for (size_t i = 0; i < res.numFrames(); ++i)
        EXPECT_EQ(res.frames[i].size(), fftSize / 2 + 1);
}

TEST(STFTAnalysis, MetadataStored) {
    size_t fftSize = 128, hopSize = 32;
    size_t N = fftSize + 5 * hopSize;
    double sr = 44100.0;
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win, sr);
    EXPECT_EQ(res.fftSize,      fftSize);
    EXPECT_EQ(res.hopSize,      hopSize);
    EXPECT_EQ(res.signalLength, N);
    EXPECT_DOUBLE_EQ(res.sampleRate, sr);
}

TEST(STFTAnalysis, TimeAxisSpacing) {
    size_t fftSize = 256, hopSize = 128;
    size_t N = fftSize + 9 * hopSize;
    double sr = 8000.0;
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win, sr);
    auto t = res.timeAxis();
    ASSERT_EQ(t.size(), res.numFrames());
    double expectedDt = static_cast<double>(hopSize) / sr;
    EXPECT_DOUBLE_EQ(t[0], 0.0);
    for (size_t i = 1; i < t.size(); ++i)
        EXPECT_NEAR(t[i] - t[i - 1], expectedDt, 1e-12);
}

TEST(STFTAnalysis, FreqAxisRange) {
    size_t fftSize = 512, hopSize = 128;
    size_t N = fftSize + 3 * hopSize;
    double sr = 16000.0;
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win, sr);
    auto f = res.freqAxis();
    ASSERT_EQ(f.size(), fftSize / 2 + 1);
    EXPECT_DOUBLE_EQ(f.front(), 0.0);
    EXPECT_DOUBLE_EQ(f.back(), sr / 2.0);
}

TEST(STFTAnalysis, ShortSignalReturnsEmpty) {
    // Signal shorter than fftSize → no complete frames
    size_t fftSize = 256;
    auto win = colaWindow(fftSize);
    auto res = stft(std::vector<double>(100, 1.0), fftSize, 64, win);
    EXPECT_EQ(res.numFrames(), 0u);
}

// ═════════════════════════════════════════════════════════════════════════════
// STFTSpectral — correct frequency localisation
// ═════════════════════════════════════════════════════════════════════════════

TEST(STFTSpectral, PureToneAppearInCorrectBin) {
    // Bin-aligned tone: k0 = 20 out of fftSize = 256.
    // Frequency = 20/256 (normalised to [0, sampleRate/2]).
    size_t fftSize = 256, hopSize = 128;
    size_t N = fftSize + 15 * hopSize;
    double sr = 1.0;
    size_t k0 = 20;
    double freq = static_cast<double>(k0) / static_cast<double>(fftSize);

    std::vector<double> sig(N);
    for (size_t i = 0; i < N; ++i)
        sig[i] = std::cos(2.0 * kPi * freq * static_cast<double>(i));

    auto win = colaWindow(fftSize);
    auto res = stft(sig, fftSize, hopSize, win, sr);

    // Middle frame: find the bin with maximum magnitude
    size_t midFrame = res.numFrames() / 2;
    size_t maxBin = 0;
    double maxMag = 0.0;
    for (size_t k = 0; k < res.numBins(); ++k) {
        double m = std::abs(res.frames[midFrame][k]);
        if (m > maxMag) { maxMag = m; maxBin = k; }
    }
    EXPECT_EQ(maxBin, k0);
}

TEST(STFTSpectral, TwoTonesAtDifferentBins) {
    size_t fftSize = 256, hopSize = 128;
    size_t N = fftSize + 15 * hopSize;
    size_t k1 = 10, k2 = 50;
    double f1 = static_cast<double>(k1) / fftSize;
    double f2 = static_cast<double>(k2) / fftSize;

    std::vector<double> sig(N);
    for (size_t i = 0; i < N; ++i)
        sig[i] = std::cos(2.0 * kPi * f1 * i) + std::cos(2.0 * kPi * f2 * i);

    auto win = colaWindow(fftSize);
    auto res = stft(sig, fftSize, hopSize, win);
    size_t midFrame = res.numFrames() / 2;

    // Find two largest bins
    std::vector<std::pair<double, size_t>> bins;
    for (size_t k = 0; k < res.numBins(); ++k)
        bins.push_back({std::abs(res.frames[midFrame][k]), k});
    std::sort(bins.rbegin(), bins.rend());

    // Both k1 and k2 must be among the two top bins
    EXPECT_TRUE(bins[0].second == k1 || bins[0].second == k2);
    EXPECT_TRUE(bins[1].second == k1 || bins[1].second == k2);
}

// ═════════════════════════════════════════════════════════════════════════════
// STFTSynthesis — ISTFT reconstruction quality
// ═════════════════════════════════════════════════════════════════════════════

TEST(STFTSynthesis, PerfectReconstruction_Hamming50pct) {
    // Hamming periodic window + 50% overlap satisfies COLA (sum = 1.08 constant).
    // All interior samples of the unmodified STFT/ISTFT roundtrip must match
    // the original signal to floating-point precision.
    size_t fftSize = 256, hopSize = fftSize / 2;
    // Exact N so no tail is discarded
    size_t N = fftSize + 15 * hopSize;
    auto sig = makeSignal(N);
    auto win = colaWindow(fftSize);

    auto res  = stft(sig, fftSize, hopSize, win);
    auto recon = istft(res);

    ASSERT_EQ(recon.size(), N);
    // Skip the very first and last samples where the window taper hits zero;
    // one sample on each side is affected by the Hamming endpoint w[0]=0.08>0
    // so reconstruction is correct everywhere — test interior conservatively.
    for (size_t i = 1; i + 1 < N; ++i)
        EXPECT_NEAR(recon[i], sig[i], kTol);
}

TEST(STFTSynthesis, ReconstructionWithImpulse) {
    // A signal that is mostly zero except for a central impulse should
    // reconstruct exactly in the interior.
    size_t fftSize = 128, hopSize = 64;
    size_t N = fftSize + 7 * hopSize;
    std::vector<double> sig(N, 0.0);
    sig[N / 2] = 1.0;

    auto win  = colaWindow(fftSize);
    auto res  = stft(sig, fftSize, hopSize, win);
    auto recon = istft(res);

    ASSERT_EQ(recon.size(), N);
    for (size_t i = 1; i + 1 < N; ++i)
        EXPECT_NEAR(recon[i], sig[i], kTol);
}

TEST(STFTSynthesis, SignalLengthIsPreserved) {
    size_t fftSize = 128, hopSize = 32;
    size_t N = fftSize + 11 * hopSize;
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win);
    EXPECT_EQ(res.signalLength, N);
}

// ═════════════════════════════════════════════════════════════════════════════
// STFTSpectrograms — magnitudeSpectrogram and powerSpectrogram
// ═════════════════════════════════════════════════════════════════════════════

TEST(STFTSpectrograms, MagnitudeShape) {
    size_t fftSize = 128, hopSize = 64;
    size_t N = fftSize + 7 * hopSize;
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win);
    auto mag = magnitudeSpectrogram(res);
    ASSERT_EQ(mag.size(), res.numFrames());
    for (const auto& row : mag)
        EXPECT_EQ(row.size(), res.numBins());
}

TEST(STFTSpectrograms, PowerEqualsMagnitudeSquared) {
    size_t fftSize = 128, hopSize = 64;
    size_t N = fftSize + 5 * hopSize;
    auto win = colaWindow(fftSize);
    auto res = stft(makeSignal(N), fftSize, hopSize, win);
    auto mag = magnitudeSpectrogram(res);
    auto pwr = powerSpectrogram(res);
    for (size_t i = 0; i < res.numFrames(); ++i)
        for (size_t k = 0; k < res.numBins(); ++k)
            EXPECT_NEAR(pwr[i][k], mag[i][k] * mag[i][k], 1e-12);
}

// ═════════════════════════════════════════════════════════════════════════════
// STFTEdge — invalid arguments
// ═════════════════════════════════════════════════════════════════════════════

TEST(STFTEdge, ZeroHopSizeThrows) {
    auto win = colaWindow(64);
    EXPECT_THROW(stft(makeSignal(256), 64, 0, win), std::invalid_argument);
}

TEST(STFTEdge, WindowSizeMismatchThrows) {
    std::vector<double> win(32, 1.0);  // wrong size for fftSize=64
    EXPECT_THROW(stft(makeSignal(256), 64, 16, win), std::invalid_argument);
}

TEST(STFTEdge, FftSizeTooSmallThrows) {
    std::vector<double> win(1, 1.0);
    EXPECT_THROW(stft(makeSignal(256), 1, 1, win), std::invalid_argument);
}
