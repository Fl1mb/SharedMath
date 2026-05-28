#include <gtest/gtest.h>
#include "DSP/Signal.h"
#include "DSP/SignalGenerator.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace SharedMath::DSP;

namespace {

std::vector<std::complex<double>> makeTone(
    double freq, double sampleRate, size_t n, double amplitude = 1.0)
{
    std::vector<std::complex<double>> iq(n);
    const double phaseStep = 2.0 * M_PI * freq / sampleRate;
    for (size_t i = 0; i < n; ++i)
        iq[i] = amplitude * std::polar(1.0, phaseStep * static_cast<double>(i));
    return iq;
}

} // namespace

TEST(SignalContainer, RealSignal_ComputesBasicCharacteristics)
{
    Signal signal(std::vector<double>{1.0, 2.0, 3.0, 4.0}, 1000.0);

    EXPECT_TRUE(signal.isReal());
    EXPECT_FALSE(signal.isComplex());
    EXPECT_EQ(signal.size(), 4u);
    EXPECT_NEAR(signal.sampleRate(), 1000.0, 1e-12);
    EXPECT_NEAR(signal.durationSec(), 0.004, 1e-12);

    const auto& c = signal.characteristics();
    EXPECT_NEAR(c.meanValue.real(), 2.5, 1e-12);
    EXPECT_NEAR(c.rms, std::sqrt(7.5), 1e-12);
    EXPECT_NEAR(c.peakAmplitude, 4.0, 1e-12);
    EXPECT_NEAR(c.averagePower, 7.5, 1e-12);
    EXPECT_NEAR(c.peakPower, 16.0, 1e-12);
}

TEST(SignalContainer, ComplexSignal_DetectsDominantFrequency)
{
    const double fs   = 8000.0;
    const double tone = 1000.0;
    const size_t fftSize = 1024;

    Signal signal(makeTone(tone, fs, 4096), fs,
                  /*nominalCenterFrequencyHz=*/1.0e6,
                  {fftSize, 0.99});

    const double binHz = fs / static_cast<double>(fftSize);
    const auto& c = signal.characteristics();

    EXPECT_TRUE(signal.isComplex());
    EXPECT_NEAR(c.dominantFrequencyHz, tone, 2.0 * binHz);
    EXPECT_NEAR(c.absoluteDominantFrequencyHz, 1.0e6 + tone, 2.0 * binHz);
    EXPECT_GT(c.snrDb, 0.0);
}

TEST(SignalContainer, RemoveDC_ZeroesMeanForRealSignal)
{
    Signal signal(std::vector<double>{10.0, 11.0, 9.0, 10.0}, 100.0);
    auto centered = signal.removeDC();

    EXPECT_NEAR(centered.characteristics().meanValue.real(), 0.0, 1e-12);
}

TEST(SignalContainer, NormalizePeak_ScalesComplexSignal)
{
    Signal signal(makeTone(200.0, 4000.0, 512, 0.5), 4000.0);
    auto normalized = signal.normalizePeak(2.0);

    EXPECT_NEAR(normalized.characteristics().peakAmplitude, 2.0, 1e-9);
}

TEST(SignalContainer, NormalizeRMS_ScalesRealSignal)
{
    Signal signal(std::vector<double>{1.0, -1.0, 1.0, -1.0}, 100.0);
    auto normalized = signal.normalizeRMS(3.0);

    EXPECT_NEAR(normalized.characteristics().rms, 3.0, 1e-9);
}

TEST(SignalContainer, Slice_PreservesSampleRateAndCount)
{
    auto samples = sineWave(100.0, 1000.0, 64);
    Signal signal(samples, 1000.0);
    auto part = signal.slice(10, 20);

    EXPECT_EQ(part.size(), 20u);
    EXPECT_NEAR(part.sampleRate(), 1000.0, 1e-12);
    EXPECT_NEAR(part.durationSec(), 20.0 / 1000.0, 1e-12);
}

TEST(SignalContainer, Resample_UpdatesLengthAndSampleRate)
{
    auto samples = sineWave(100.0, 48000.0, 480);
    Signal signal(samples, 48000.0);
    auto resampled = signal.resample(24000.0);

    EXPECT_NEAR(resampled.sampleRate(), 24000.0, 1e-12);
    EXPECT_NEAR(static_cast<double>(resampled.size()), 240.0, 2.0);
}

TEST(SignalContainer, FrequencyShift_MovesToneToBaseband)
{
    const double fs   = 12000.0;
    const double tone = 1500.0;
    const size_t fftSize = 1024;

    Signal signal(makeTone(tone, fs, 4096), fs, 0.0, {fftSize, 0.99});
    auto shifted = signal.frequencyShift(-tone, {fftSize, 0.99});

    const double binHz = fs / static_cast<double>(fftSize);
    EXPECT_NEAR(shifted.characteristics().dominantFrequencyHz, 0.0, 2.0 * binHz);
}

TEST(SignalContainer, FrequencyShift_OnRealSignal_Throws)
{
    Signal signal(std::vector<double>{1.0, 2.0, 3.0}, 1000.0);
    EXPECT_THROW(signal.frequencyShift(100.0), std::invalid_argument);
}

TEST(SignalContainer, FileBackedRealSignal_ReadsFromRawByteFile)
{
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "sharedmath_signal_real_u8.raw";

    {
        std::ofstream out(path, std::ios::binary);
        const std::uint8_t bytes[] = {128, 129, 127, 130, 126};
        out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }

    Signal signal(
        SignalFileParams{
            path.string(),
            SignalFileFormat::RealU8,
            0,
            0,
            1.0,
            -128.0
        },
        1000.0,
        2.4e6);

    EXPECT_TRUE(signal.isFileBacked());
    EXPECT_TRUE(signal.isReal());
    EXPECT_EQ(signal.size(), 5u);
    EXPECT_THROW(signal.realSamples(), std::invalid_argument);
    EXPECT_THROW(signal.removeDC(), std::invalid_argument);

    auto block = signal.load(1, 3);
    EXPECT_FALSE(block.isFileBacked());
    ASSERT_TRUE(block.isReal());
    ASSERT_EQ(block.realSamples().size(), 3u);
    EXPECT_DOUBLE_EQ(block.realSamples()[0], 1.0);
    EXPECT_DOUBLE_EQ(block.realSamples()[1], -1.0);
    EXPECT_DOUBLE_EQ(block.realSamples()[2], 2.0);
    EXPECT_NEAR(block.nominalCenterFrequencyHz(), 2.4e6, 1e-12);

    fs::remove(path);
}

TEST(SignalContainer, FileBackedComplexSignal_ReadsInterleavedIQ)
{
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "sharedmath_signal_complex_i8.raw";

    {
        std::ofstream out(path, std::ios::binary);
        const std::int8_t bytes[] = {1, -2, 3, -4, 5, -6};
        out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }

    Signal signal(
        SignalFileParams{
            path.string(),
            SignalFileFormat::ComplexI8Interleaved,
            0,
            0,
            1.0,
            0.0
        },
        2000.0);

    EXPECT_TRUE(signal.isFileBacked());
    EXPECT_TRUE(signal.isComplex());
    EXPECT_EQ(signal.size(), 3u);

    auto block = signal.load(0, 2);
    const auto& iq = block.complexSamples();
    ASSERT_EQ(iq.size(), 2u);
    EXPECT_EQ(iq[0], std::complex<double>(1.0, -2.0));
    EXPECT_EQ(iq[1], std::complex<double>(3.0, -4.0));

    fs::remove(path);
}

TEST(SignalContainer, FileBackedSignal_SliceRemainsLazy)
{
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() / "sharedmath_signal_slice_u8.raw";

    {
        std::ofstream out(path, std::ios::binary);
        const std::uint8_t bytes[] = {1, 2, 3, 4, 5, 6};
        out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    }

    Signal signal(
        SignalFileParams{
            path.string(),
            SignalFileFormat::RealU8
        },
        100.0);

    auto slice = signal.slice(2, 3);
    EXPECT_TRUE(slice.isFileBacked());
    EXPECT_EQ(slice.size(), 3u);

    auto loaded = slice.load();
    const auto& x = loaded.realSamples();
    ASSERT_EQ(x.size(), 3u);
    EXPECT_DOUBLE_EQ(x[0], 3.0);
    EXPECT_DOUBLE_EQ(x[1], 4.0);
    EXPECT_DOUBLE_EQ(x[2], 5.0);

    fs::remove(path);
}
