#include <gtest/gtest.h>
#include <ML/ml.h>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// StandardScaler
// ─────────────────────────────────────────────────────────────────────────────

TEST(StandardScaler, FitTransformZeroMeanUnitStd) {
    // X has 2 features: col0 = {1,2,3}, col1 = {4,5,6}
    Tensor X = Tensor::zeros({3, 2});
    X(0,0)=1; X(1,0)=2; X(2,0)=3;
    X(0,1)=4; X(1,1)=5; X(2,1)=6;

    StandardScaler sc;
    Tensor Xt = sc.fit_transform(X);

    // After standardisation each column should have mean ≈ 0
    for (size_t d = 0; d < 2; ++d) {
        double m = 0.0;
        for (size_t i = 0; i < 3; ++i) m += Xt(i, d);
        EXPECT_NEAR(m / 3.0, 0.0, 1e-10) << "column " << d;
    }
    EXPECT_TRUE(sc.fitted());
}

TEST(StandardScaler, InverseTransformRoundTrip) {
    Tensor X = Tensor::zeros({4, 2});
    X(0,0)=2; X(1,0)=4; X(2,0)=6; X(3,0)=8;
    X(0,1)=1; X(1,1)=3; X(2,1)=5; X(3,1)=7;

    StandardScaler sc;
    Tensor Xt  = sc.fit_transform(X);
    Tensor Xr  = sc.inverse_transform(Xt);

    for (size_t i = 0; i < 4; ++i)
        for (size_t d = 0; d < 2; ++d)
            EXPECT_NEAR(Xr(i,d), X(i,d), 1e-9);
}

TEST(StandardScaler, NotFittedThrows) {
    StandardScaler sc;
    Tensor X = Tensor::zeros({2, 2});
    EXPECT_THROW(sc.transform(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// MinMaxScaler
// ─────────────────────────────────────────────────────────────────────────────

TEST(MinMaxScaler, DefaultRange01) {
    Tensor X = Tensor::zeros({3, 2});
    X(0,0)=0; X(1,0)=5; X(2,0)=10;
    X(0,1)=2; X(1,1)=6; X(2,1)=10;

    MinMaxScaler sc;
    Tensor Xt = sc.fit_transform(X);

    // Column 0: min=0, max=10 → Xt values = 0, 0.5, 1.0
    EXPECT_NEAR(Xt(0,0), 0.0, 1e-9);
    EXPECT_NEAR(Xt(1,0), 0.5, 1e-9);
    EXPECT_NEAR(Xt(2,0), 1.0, 1e-9);
}

TEST(MinMaxScaler, CustomRange) {
    Tensor X = Tensor::zeros({2, 1});
    X(0,0) = 0.0;
    X(1,0) = 10.0;

    MinMaxScaler sc(-1.0, 1.0);
    Tensor Xt = sc.fit_transform(X);

    EXPECT_NEAR(Xt(0,0), -1.0, 1e-9);
    EXPECT_NEAR(Xt(1,0),  1.0, 1e-9);
}

TEST(MinMaxScaler, InverseTransformRoundTrip) {
    Tensor X = Tensor::zeros({3, 2});
    X(0,0)=1; X(1,0)=3; X(2,0)=5;
    X(0,1)=2; X(1,1)=4; X(2,1)=6;

    MinMaxScaler sc;
    Tensor Xt = sc.fit_transform(X);
    Tensor Xr = sc.inverse_transform(Xt);

    for (size_t i = 0; i < 3; ++i)
        for (size_t d = 0; d < 2; ++d)
            EXPECT_NEAR(Xr(i,d), X(i,d), 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// train_test_split
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrainTestSplit, SizesCorrect) {
    const size_t N = 20;
    Tensor X = Tensor::zeros({N, 3});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) y.flat(i) = static_cast<double>(i);

    auto split = train_test_split(X, y, 0.2, /*shuffle=*/false);
    EXPECT_EQ(split.X_train.dim(0) + split.X_test.dim(0), N);
    EXPECT_EQ(split.y_train.size() + split.y_test.size(), N);
    EXPECT_EQ(split.X_test.dim(0), 4u);
}

TEST(TrainTestSplit, NoShuffle) {
    const size_t N = 10;
    Tensor X = Tensor::zeros({N, 2});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) y.flat(i) = static_cast<double>(i);

    auto split = train_test_split(X, y, 0.3, /*shuffle=*/false);
    // Without shuffle: first 7 are train, last 3 are test
    EXPECT_EQ(split.y_train.size(), 7u);
    for (size_t i = 0; i < 7; ++i)
        EXPECT_NEAR(split.y_train.flat(i), static_cast<double>(i), 1e-9);
}

TEST(TrainTestSplit, ShuffleReproducible) {
    const size_t N = 20;
    Tensor X = Tensor::zeros({N, 2});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) y.flat(i) = static_cast<double>(i);

    auto s1 = train_test_split(X, y, 0.2, true, 42);
    auto s2 = train_test_split(X, y, 0.2, true, 42);
    for (size_t i = 0; i < s1.y_test.size(); ++i)
        EXPECT_NEAR(s1.y_test.flat(i), s2.y_test.flat(i), 1e-9);
}

TEST(TrainTestSplit, InvalidTestSizeThrows) {
    Tensor X = Tensor::zeros({5, 2});
    Tensor y = Tensor::zeros({5});
    EXPECT_THROW(train_test_split(X, y, 0.0), std::invalid_argument);
    EXPECT_THROW(train_test_split(X, y, 1.0), std::invalid_argument);
}
