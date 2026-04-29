#include <gtest/gtest.h>

#include "DSP/FilterDesign.h"
#include "DSP/IIR.h"

#include <algorithm>
#include <cmath>

using namespace SharedMath::DSP;

namespace {

constexpr double kFs = 48000.0;
constexpr size_t kNFFT = 8192;

size_t nearestBin(double hz) {
    auto f = frequencyAxis(kNFFT, kFs);
    return static_cast<size_t>(
        std::min_element(f.begin(), f.end(),
            [hz](double a, double b) {
                return std::abs(a - hz) < std::abs(b - hz);
            }) - f.begin());
}

double magAt(const BiquadCoeffs& bq, double hz) {
    auto mag = magnitudeResponseIIR({bq}, kNFFT);
    return mag[nearestBin(hz)];
}

} // namespace

TEST(RBJLowPassHz, PassesDCAndAttenuatesHighFrequency) {
    auto bq = designRBJLowPassHz(2000.0, kFs);
    EXPECT_GT(magAt(bq, 100.0), 0.95);
    EXPECT_LT(magAt(bq, 12000.0), 0.05);
    EXPECT_TRUE(bq.isStable());
}

TEST(RBJHighPassHz, AttenuatesDCAndPassesHighFrequency) {
    auto bq = designRBJHighPassHz(2000.0, kFs);
    EXPECT_LT(magAt(bq, 100.0), 0.05);
    EXPECT_GT(magAt(bq, 12000.0), 0.95);
    EXPECT_TRUE(bq.isStable());
}

TEST(RBJBandPassHz, PeaksNearCenter) {
    auto bq = designRBJBandPassHz(4000.0, kFs, 4.0);
    EXPECT_GT(magAt(bq, 4000.0), magAt(bq, 1000.0));
    EXPECT_GT(magAt(bq, 4000.0), magAt(bq, 12000.0));
    EXPECT_TRUE(bq.isStable());
}

TEST(RBJNotchHz, SuppressesCenter) {
    auto bq = designRBJNotchHz(1000.0, kFs, 20.0);
    EXPECT_LT(magAt(bq, 1000.0), 0.1);
    EXPECT_GT(magAt(bq, 3000.0), 0.9);
    EXPECT_TRUE(bq.isStable());
}

TEST(RBJAllPassHz, MagnitudeIsUnity) {
    auto bq = designRBJAllPassHz(3000.0, kFs, 0.8);
    EXPECT_NEAR(magAt(bq, 100.0), 1.0, 1e-9);
    EXPECT_NEAR(magAt(bq, 3000.0), 1.0, 1e-9);
    EXPECT_NEAR(magAt(bq, 12000.0), 1.0, 1e-9);
    EXPECT_TRUE(bq.isStable());
}

TEST(RBJPeakingEQHz, GainAtCenterMatchesDB) {
    auto bq = designRBJPeakingEQHz(2000.0, kFs, 6.0, 3.0);
    EXPECT_NEAR(20.0 * std::log10(magAt(bq, 2000.0)), 6.0, 0.25);
    EXPECT_TRUE(bq.isStable());
}

TEST(RBJShelvesHz, BoostExpectedBand) {
    auto low = designRBJLowShelfHz(500.0, kFs, 6.0);
    auto high = designRBJHighShelfHz(8000.0, kFs, 6.0);

    EXPECT_GT(magAt(low, 50.0), magAt(low, 5000.0));
    EXPECT_GT(magAt(high, 16000.0), magAt(high, 1000.0));
    EXPECT_TRUE(low.isStable());
    EXPECT_TRUE(high.isStable());
}

TEST(RBJHz, InvalidParametersThrow) {
    EXPECT_THROW(designRBJLowPassHz(0.0, kFs), std::invalid_argument);
    EXPECT_THROW(designRBJHighPassHz(kFs / 2.0, kFs), std::invalid_argument);
    EXPECT_THROW(designRBJBandPassHz(1000.0, kFs, 0.0), std::invalid_argument);
    EXPECT_THROW(designRBJAllPassHz(1000.0, kFs, -1.0), std::invalid_argument);
}
