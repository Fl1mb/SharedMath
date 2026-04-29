#include <gtest/gtest.h>

#include "DSP/Resampling.h"
#include "DSP/SignalGenerator.h"
#include "DSP/Spectral.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace SharedMath::DSP;

TEST(ResampleTo, SameRateReturnsInput) {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    auto y = resampleTo(x, 48000, 48000);
    ASSERT_EQ(y.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(y[i], x[i], 1e-12);
}

TEST(ResampleTo, DownsampleLengthMatchesRateRatio) {
    auto x = sineWave(440.0, 48000.0, 4800);
    auto y = resampleTo(x, 48000, 16000);
    EXPECT_EQ(y.size(), 1600u);
}

TEST(ResampleTo, UpsampleLengthMatchesRateRatio) {
    auto x = sineWave(440.0, 16000.0, 1600);
    auto y = resampleTo(x, 16000, 48000);
    EXPECT_EQ(y.size(), 4800u);
}

TEST(ResampleTo, PreservesToneFrequencyReasonably) {
    constexpr double f0 = 1000.0;
    auto x = sineWave(f0, 48000.0, 4800);
    auto y = resampleTo(x, 48000, 16000);

    auto psd = welchPSD(y, 16000.0, 1024);
    auto peak = std::max_element(psd.psd.begin() + 1, psd.psd.end());
    const size_t bin = static_cast<size_t>(peak - psd.psd.begin());

    EXPECT_NEAR(psd.frequencies[bin], f0, 40.0);
}

TEST(ResamplePolyphaseAligned, ReducesSimpleRatioLength) {
    std::vector<double> x(100, 1.0);
    auto y = resamplePolyphaseAligned(x, 2, 5);
    EXPECT_EQ(y.size(), 40u);
}

TEST(Resampling, InvalidParametersThrow) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    EXPECT_THROW(upfirdn(x, {}, 0, 1), std::invalid_argument);
    EXPECT_THROW(upfirdn(x, {}, 1, 0), std::invalid_argument);
    EXPECT_THROW(resampleTo(x, 0, 48000), std::invalid_argument);
    EXPECT_THROW(resampleTo(x, 48000, 0), std::invalid_argument);
    EXPECT_THROW(resamplePolyphaseAligned(x, 0, 1), std::invalid_argument);
    EXPECT_THROW(resamplePolyphaseAligned(x, 1, 0), std::invalid_argument);
}
