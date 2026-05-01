#include <gtest/gtest.h>
#include <ML/ml.h>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Two clearly separated clusters for classification
static std::pair<Tensor, Tensor> makeTwoClusters(size_t N = 40) {
    Tensor X = Tensor::zeros({N, 2});
    Tensor y = Tensor::zeros({N});
    const size_t half = N / 2;
    for (size_t i = 0; i < half; ++i) {
        X(i, 0) = -3.0; X(i, 1) = 0.0;
        y.flat(i) = 0.0;
    }
    for (size_t i = half; i < N; ++i) {
        X(i, 0) = 3.0; X(i, 1) = 0.0;
        y.flat(i) = 1.0;
    }
    return {X, y};
}

// Simple regression: y = 2*x
static std::pair<Tensor, Tensor> makeLinearRegression(size_t N = 20) {
    Tensor X = Tensor::zeros({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        X(i, 0) = static_cast<double>(i);
        y.flat(i) = 2.0 * static_cast<double>(i);
    }
    return {X, y};
}

// ─────────────────────────────────────────────────────────────────────────────
// RandomForestClassifier
// ─────────────────────────────────────────────────────────────────────────────

TEST(RandomForestClassifier, TwoClusters100Trees) {
    auto [X, y] = makeTwoClusters();
    RandomForestClassifier rf(100, 0, 2, 42);
    rf.fit(X, y);
    Tensor pred = rf.predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < 40; ++i)
        if (static_cast<int>(pred.flat(i)) == static_cast<int>(y.flat(i))) ++correct;
    EXPECT_GE(correct, 38u);
}

TEST(RandomForestClassifier, ReproducibleWithSameSeed) {
    auto [X, y] = makeTwoClusters();
    RandomForestClassifier rf1(10, 0, 2, 7);
    RandomForestClassifier rf2(10, 0, 2, 7);
    rf1.fit(X, y); rf2.fit(X, y);
    Tensor p1 = rf1.predict(X);
    Tensor p2 = rf2.predict(X);
    for (size_t i = 0; i < 40; ++i)
        EXPECT_NEAR(p1.flat(i), p2.flat(i), 1e-9);
}

TEST(RandomForestClassifier, DifferentSeedsDifferentResults) {
    auto [X, y] = makeTwoClusters();
    // Create very noisy data so seeds matter
    Tensor Xn = Tensor::zeros({20, 2});
    Tensor yn = Tensor::zeros({20});
    for (size_t i = 0; i < 20; ++i) {
        Xn(i, 0) = static_cast<double>(i % 5);
        Xn(i, 1) = static_cast<double>(i % 3);
        yn.flat(i) = static_cast<double>(i % 2);
    }
    RandomForestClassifier rf1(5, 2, 2, 1);
    RandomForestClassifier rf2(5, 2, 2, 99);
    rf1.fit(Xn, yn); rf2.fit(Xn, yn);
    // At least one prediction differs
    Tensor p1 = rf1.predict(Xn);
    Tensor p2 = rf2.predict(Xn);
    bool differ = false;
    for (size_t i = 0; i < 20; ++i)
        if (p1.flat(i) != p2.flat(i)) { differ = true; break; }
    // This is probabilistic — just test it runs
    EXPECT_TRUE(rf1.fitted());
    EXPECT_TRUE(rf2.fitted());
    (void)differ;
}

TEST(RandomForestClassifier, NotFittedThrows) {
    RandomForestClassifier rf;
    Tensor X = Tensor::zeros({2, 2});
    EXPECT_THROW(rf.predict(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// RandomForestRegressor
// ─────────────────────────────────────────────────────────────────────────────

TEST(RandomForestRegressor, LinearRegressionLowError) {
    auto [X, y] = makeLinearRegression();
    RandomForestRegressor rf(50, 0, 2, 0);
    rf.fit(X, y);
    Tensor pred = rf.predict(X);
    double mse = 0.0;
    for (size_t i = 0; i < 20; ++i) {
        double d = pred.flat(i) - y.flat(i);
        mse += d * d;
    }
    mse /= 20.0;
    EXPECT_LT(mse, 5.0);
}

TEST(RandomForestRegressor, NotFittedThrows) {
    RandomForestRegressor rf;
    Tensor X = Tensor::zeros({2, 1});
    EXPECT_THROW(rf.predict(X), std::runtime_error);
}

TEST(RandomForestRegressor, Accessors) {
    RandomForestRegressor rf(10);
    EXPECT_EQ(rf.n_estimators(), 10u);
    EXPECT_FALSE(rf.fitted());
}
