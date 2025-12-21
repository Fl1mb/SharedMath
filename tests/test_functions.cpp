#include <gtest/gtest.h>
#include "../include/SharedMath.h"

using namespace SharedMath::Functions;

// TEST(GammaTest, EasyGammaTest){
//     EXPECT_NEAR(1, GammaFunction::value(0), 1e-6);
//     EXPECT_NEAR(1, GammaFunction::value(1), 1e-6);
//     EXPECT_NEAR(24, GammaFunction::value(4), 1e-6);
//     EXPECT_NEAR(sqrt(M_PI), GammaFunction::value(0.5), 1e-6);
//     EXPECT_NEAR(15.0 / 8.0 * sqrt(M_PI), GammaFunction::value(7.0 / 2.0), 1e-6);
// }