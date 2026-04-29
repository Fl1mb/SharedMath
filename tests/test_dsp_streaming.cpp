#include <gtest/gtest.h>
#include "DSP/Streaming.h"
#include "DSP/FIR.h"
#include "DSP/IIR.h"

#include <cmath>
#include <vector>
#include <numeric>

using namespace SharedMath::DSP;

// ─────────────────────────────────────────────────────────────────────────────
// FIRFilter

TEST(FIRFilter, SampleBysSampleMatchesBlock) {
    auto h = designFIRLowPass(32, 0.3);
    FIRFilter f1(h), f2(h);

    std::vector<double> input(128);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = std::sin(0.1 * static_cast<double>(i));

    auto blockOut = f1.processBlock(input);

    std::vector<double> sampleOut(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        sampleOut[i] = f2.processSample(input[i]);

    ASSERT_EQ(blockOut.size(), sampleOut.size());
    for (size_t i = 0; i < blockOut.size(); ++i)
        EXPECT_NEAR(blockOut[i], sampleOut[i], 1e-12);
}

TEST(FIRFilter, ProcessInPlaceMatchesBlock) {
    auto h = designFIRLowPass(16, 0.4);
    FIRFilter f1(h), f2(h);

    std::vector<double> buf1(64), buf2(64);
    for (size_t i = 0; i < buf1.size(); ++i)
        buf1[i] = buf2[i] = static_cast<double>(i) * 0.01;

    auto ref = f1.processBlock(buf1);
    f2.processInPlace(buf2);

    for (size_t i = 0; i < ref.size(); ++i)
        EXPECT_NEAR(ref[i], buf2[i], 1e-12);
}

TEST(FIRFilter, StatePreservedBetweenBlocks) {
    auto h = designFIRLowPass(16, 0.3);
    FIRFilter f1(h), f2(h);

    std::vector<double> full(100);
    for (size_t i = 0; i < full.size(); ++i)
        full[i] = std::cos(0.2 * static_cast<double>(i));

    auto fullOut = f1.processBlock(full);

    // Split into two halves
    std::vector<double> half1(full.begin(), full.begin() + 50);
    std::vector<double> half2(full.begin() + 50, full.end());
    auto out1 = f2.processBlock(half1);
    auto out2 = f2.processBlock(half2);

    for (size_t i = 0; i < 50; ++i)
        EXPECT_NEAR(fullOut[i],      out1[i], 1e-12);
    for (size_t i = 0; i < 50; ++i)
        EXPECT_NEAR(fullOut[50 + i], out2[i], 1e-12);
}

TEST(FIRFilter, ResetReproducesResult) {
    auto h = designFIRLowPass(16, 0.3);
    FIRFilter f(h);

    std::vector<double> input(64);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<double>(i % 7) * 0.1;

    auto first = f.processBlock(input);
    f.reset();
    auto second = f.processBlock(input);

    for (size_t i = 0; i < first.size(); ++i)
        EXPECT_NEAR(first[i], second[i], 1e-12);
}

TEST(FIRFilter, SetCoefficientsResetsState) {
    auto h = designFIRLowPass(16, 0.3);
    FIRFilter f(h);

    // Process some data to dirty the state
    std::vector<double> junk(32, 1.0);
    f.processBlock(junk);

    // Replace with same coefficients (equivalent to reset)
    f.setCoefficients(h);

    std::vector<double> input(32);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<double>(i) * 0.05;

    FIRFilter fresh(h);
    auto expected = fresh.processBlock(input);
    auto actual   = f.processBlock(input);

    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(expected[i], actual[i], 1e-12);
}

TEST(FIRFilter, EmptyCoefficientsPassthrough) {
    FIRFilter f;
    EXPECT_NEAR(f.processSample(3.14), 3.14, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// IIRFilter

TEST(IIRFilter, SampleBySampleMatchesBlock) {
    auto sections = designButterworthLowPass(4, 0.3);
    IIRFilter f1(sections), f2(sections);

    std::vector<double> input(128);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = std::sin(0.1 * static_cast<double>(i));

    auto blockOut = f1.processBlock(input);

    std::vector<double> sampleOut(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        sampleOut[i] = f2.processSample(input[i]);

    for (size_t i = 0; i < blockOut.size(); ++i)
        EXPECT_NEAR(blockOut[i], sampleOut[i], 1e-12);
}

TEST(IIRFilter, ProcessInPlaceMatchesBlock) {
    auto sections = designButterworthLowPass(2, 0.4);
    IIRFilter f1(sections), f2(sections);

    std::vector<double> buf1(64, 1.0), buf2(64, 1.0);
    auto ref = f1.processBlock(buf1);
    f2.processInPlace(buf2);

    for (size_t i = 0; i < ref.size(); ++i)
        EXPECT_NEAR(ref[i], buf2[i], 1e-12);
}

TEST(IIRFilter, StatePreservedBetweenBlocks) {
    auto sections = designButterworthLowPass(2, 0.3);
    IIRFilter f1(sections), f2(sections);

    std::vector<double> full(80);
    for (size_t i = 0; i < full.size(); ++i)
        full[i] = std::cos(0.15 * static_cast<double>(i));

    auto fullOut = f1.processBlock(full);

    std::vector<double> half1(full.begin(), full.begin() + 40);
    std::vector<double> half2(full.begin() + 40, full.end());
    auto out1 = f2.processBlock(half1);
    auto out2 = f2.processBlock(half2);

    for (size_t i = 0; i < 40; ++i)
        EXPECT_NEAR(fullOut[i],      out1[i], 1e-12);
    for (size_t i = 0; i < 40; ++i)
        EXPECT_NEAR(fullOut[40 + i], out2[i], 1e-12);
}

TEST(IIRFilter, ResetReproducesResult) {
    auto sections = designButterworthLowPass(2, 0.3);
    IIRFilter f(sections);

    std::vector<double> input(64, 1.0);
    auto first = f.processBlock(input);
    f.reset();
    auto second = f.processBlock(input);

    for (size_t i = 0; i < first.size(); ++i)
        EXPECT_NEAR(first[i], second[i], 1e-12);
}

TEST(IIRFilter, SetSectionsResetsState) {
    auto sections = designButterworthLowPass(2, 0.3);
    IIRFilter f(sections);

    std::vector<double> junk(20, 5.0);
    f.processBlock(junk);
    f.setSections(sections);

    std::vector<double> input(20, 1.0);
    IIRFilter fresh(sections);
    auto expected = fresh.processBlock(input);
    auto actual   = f.processBlock(input);

    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(expected[i], actual[i], 1e-12);
}
