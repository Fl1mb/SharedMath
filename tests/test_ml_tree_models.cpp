#include <gtest/gtest.h>
#include <ML/ml.h>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// XOR dataset (linearly non-separable, needs depth >= 2)
static std::pair<Tensor, Tensor> makeXOR() {
    Tensor X = Tensor::zeros({4, 2});
    Tensor y = Tensor::zeros({4});
    X(0,0)=0; X(0,1)=0; y.flat(0)=0;
    X(1,0)=0; X(1,1)=1; y.flat(1)=1;
    X(2,0)=1; X(2,1)=0; y.flat(2)=1;
    X(3,0)=1; X(3,1)=1; y.flat(3)=0;
    return {X, y};
}

// Simple linearly separable dataset
static std::pair<Tensor, Tensor> makeLinear() {
    const size_t N = 20;
    Tensor X = Tensor::zeros({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        X(i,0) = static_cast<double>(i);
        y.flat(i) = (i < N/2) ? 0.0 : 1.0;
    }
    return {X, y};
}

// Regression: y = x^2
static std::pair<Tensor, Tensor> makeQuadratic() {
    const size_t N = 10;
    Tensor X = Tensor::zeros({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        double x = static_cast<double>(i);
        X(i,0) = x;
        y.flat(i) = x * x;
    }
    return {X, y};
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionTreeClassifier
// ─────────────────────────────────────────────────────────────────────────────

TEST(DecisionTreeClassifier, XORPerfectFitUnlimitedDepth) {
    auto [X, y] = makeXOR();
    DecisionTreeClassifier clf;
    clf.fit(X, y);
    Tensor pred = clf.predict(X);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(pred.flat(i), y.flat(i), 1e-9) << "sample " << i;
}

TEST(DecisionTreeClassifier, LinearDataHighAccuracy) {
    auto [X, y] = makeLinear();
    DecisionTreeClassifier clf;
    clf.fit(X, y);
    Tensor pred = clf.predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < 20; ++i)
        if (static_cast<int>(pred.flat(i)) == static_cast<int>(y.flat(i))) ++correct;
    EXPECT_GE(correct, 18u);
}

TEST(DecisionTreeClassifier, Entropy) {
    auto [X, y] = makeXOR();
    DecisionTreeClassifier clf(0, 2, DecisionTreeClassifier::Criterion::Entropy);
    clf.fit(X, y);
    EXPECT_TRUE(clf.fitted());
    Tensor pred = clf.predict(X);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(pred.flat(i), y.flat(i), 1e-9);
}

TEST(DecisionTreeClassifier, MaxDepthLimitsTree) {
    auto [X, y] = makeXOR();
    // With max_depth=1 (single split) XOR cannot be perfectly separated
    DecisionTreeClassifier clf(1);
    clf.fit(X, y);
    EXPECT_TRUE(clf.fitted());
    // Just ensure predict runs without errors
    EXPECT_NO_THROW(clf.predict(X));
}

TEST(DecisionTreeClassifier, NotFittedThrows) {
    DecisionTreeClassifier clf;
    Tensor X = Tensor::zeros({2, 2});
    EXPECT_THROW(clf.predict(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// DecisionTreeRegressor
// ─────────────────────────────────────────────────────────────────────────────

TEST(DecisionTreeRegressor, MemorisesTrainingData) {
    auto [X, y] = makeQuadratic();
    DecisionTreeRegressor reg;
    reg.fit(X, y);
    Tensor pred = reg.predict(X);
    double mse = 0.0;
    for (size_t i = 0; i < 10; ++i) {
        double d = pred.flat(i) - y.flat(i);
        mse += d*d;
    }
    mse /= 10.0;
    EXPECT_LT(mse, 1.0);   // should be very close to 0
}

TEST(DecisionTreeRegressor, MaxDepth1) {
    auto [X, y] = makeQuadratic();
    DecisionTreeRegressor reg(1);
    reg.fit(X, y);
    EXPECT_NO_THROW(reg.predict(X));
}

TEST(DecisionTreeRegressor, NotFittedThrows) {
    DecisionTreeRegressor reg;
    Tensor X = Tensor::zeros({2, 1});
    EXPECT_THROW(reg.predict(X), std::runtime_error);
}
