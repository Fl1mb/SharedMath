#include <gtest/gtest.h>
#include <ML/ml.h>
#include <cmath>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// y = 3*x0 + 2*x1 + 1  (noiseless)
static std::pair<Tensor, Tensor> makeRegData(size_t N = 30) {
    Tensor X = Tensor::zeros({N, 2});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        X(i,0) = static_cast<double>(i) / static_cast<double>(N);
        X(i,1) = static_cast<double>(i % 5) * 0.2;
        y.flat(i) = 3.0 * X(i,0) + 2.0 * X(i,1) + 1.0;
    }
    return {X, y};
}

// Binary classification: y=0 for x<0, y=1 for x>0
static std::pair<Tensor, Tensor> makeBinaryClass() {
    const size_t N = 20;
    Tensor X = Tensor::zeros({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        double x = static_cast<double>(i) - 10.0;
        X(i,0) = x;
        y.flat(i) = (x >= 0.0) ? 1.0 : 0.0;
    }
    return {X, y};
}

// ─────────────────────────────────────────────────────────────────────────────
// RidgeRegression
// ─────────────────────────────────────────────────────────────────────────────

TEST(RidgeRegression, LowAlphaCloseToOLS) {
    auto [X, y] = makeRegData();
    RidgeRegression ridge(0.0001, 0.01, 5000);
    ridge.fit(X, y);
    Tensor pred = ridge.predict(X);

    double mse = mean_squared_error(pred, y);
    EXPECT_LT(mse, 0.1);
}

TEST(RidgeRegression, HighAlphaShrinks) {
    auto [X, y] = makeRegData();
    RidgeRegression ridge_small(0.01,  0.01, 2000);
    RidgeRegression ridge_large(100.0, 0.01, 2000);
    ridge_small.fit(X, y);
    ridge_large.fit(X, y);

    // Coefficients of high-alpha ridge should be smaller in norm
    double norm_small = 0.0, norm_large = 0.0;
    for (size_t d = 0; d < 2; ++d) {
        norm_small += ridge_small.coef().flat(d) * ridge_small.coef().flat(d);
        norm_large += ridge_large.coef().flat(d) * ridge_large.coef().flat(d);
    }
    EXPECT_LT(norm_large, norm_small);
}

TEST(RidgeRegression, NotFittedThrows) {
    RidgeRegression r;
    Tensor X = Tensor::zeros({2, 2});
    EXPECT_THROW(r.predict(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// LassoRegression
// ─────────────────────────────────────────────────────────────────────────────

TEST(LassoRegression, LowAlphaFitsData) {
    auto [X, y] = makeRegData();
    LassoRegression lasso(0.0001, 2000);
    lasso.fit(X, y);
    Tensor pred = lasso.predict(X);
    EXPECT_LT(mean_squared_error(pred, y), 0.2);
}

TEST(LassoRegression, HighAlphaSparsity) {
    auto [X, y] = makeRegData();
    LassoRegression lasso(10.0, 2000);
    lasso.fit(X, y);
    // At very high alpha some (or all) coefficients should be near zero
    bool any_zero = false;
    for (size_t d = 0; d < 2; ++d)
        if (std::abs(lasso.coef().flat(d)) < 0.1) any_zero = true;
    EXPECT_TRUE(any_zero);
}

TEST(LassoRegression, NotFittedThrows) {
    LassoRegression l;
    Tensor X = Tensor::zeros({2, 2});
    EXPECT_THROW(l.predict(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// ElasticNet
// ─────────────────────────────────────────────────────────────────────────────

TEST(ElasticNet, MixedPenaltyFitsData) {
    auto [X, y] = makeRegData();
    ElasticNet en(0.0001, 0.5, 2000);
    en.fit(X, y);
    Tensor pred = en.predict(X);
    EXPECT_LT(mean_squared_error(pred, y), 0.3);
}

TEST(ElasticNet, PureL2SimilarToRidge) {
    auto [X, y] = makeRegData();
    ElasticNet en(0.01, 0.0, 2000);   // l1_ratio=0 → pure Ridge
    RidgeRegression ridge(0.01, 0.01, 2000);
    en.fit(X, y);
    ridge.fit(X, y);

    Tensor p_en    = en.predict(X);
    Tensor p_ridge = ridge.predict(X);
    // Both should have similar MSE
    EXPECT_LT(std::abs(mean_squared_error(p_en, y) - mean_squared_error(p_ridge, y)), 0.5);
}

TEST(ElasticNet, NotFittedThrows) {
    ElasticNet en;
    Tensor X = Tensor::zeros({2, 2});
    EXPECT_THROW(en.predict(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianNB
// ─────────────────────────────────────────────────────────────────────────────

TEST(GaussianNB, BinaryPerfect) {
    auto [X, y] = makeBinaryClass();
    GaussianNB nb;
    nb.fit(X, y);
    Tensor pred = nb.predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < 20; ++i)
        if (static_cast<int>(pred.flat(i)) == static_cast<int>(y.flat(i))) ++correct;
    EXPECT_GE(correct, 18u);
}

TEST(GaussianNB, ProbaSumsToOne) {
    auto [X, y] = makeBinaryClass();
    GaussianNB nb;
    nb.fit(X, y);
    Tensor proba = nb.predict_proba(X);
    for (size_t i = 0; i < 20; ++i) {
        double s = proba(i, 0) + proba(i, 1);
        EXPECT_NEAR(s, 1.0, 1e-9);
    }
}

TEST(GaussianNB, NotFittedThrows) {
    GaussianNB nb;
    Tensor X = Tensor::zeros({2, 1});
    EXPECT_THROW(nb.predict(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// LinearSVM
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearSVM, BinaryClassification) {
    auto [X, y] = makeBinaryClass();
    LinearSVM svm(1.0, 1e-3, 2000);
    svm.fit(X, y);
    Tensor pred = svm.predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < 20; ++i)
        if (static_cast<int>(pred.flat(i)) == static_cast<int>(y.flat(i))) ++correct;
    EXPECT_GE(correct, 16u);
}

TEST(LinearSVM, DecisionFunctionSign) {
    auto [X, y] = makeBinaryClass();
    LinearSVM svm(1.0, 1e-3, 2000);
    svm.fit(X, y);
    Tensor df = svm.decision_function(X);
    Tensor pred = svm.predict(X);
    for (size_t i = 0; i < 20; ++i) {
        // predict returns 1 iff decision_function >= 0
        bool df_pos = (df.flat(i) >= 0.0);
        bool pred_one = (pred.flat(i) == 1.0);
        EXPECT_EQ(df_pos, pred_one);
    }
}

TEST(LinearSVM, NotFittedThrows) {
    LinearSVM svm;
    Tensor X = Tensor::zeros({2, 1});
    EXPECT_THROW(svm.predict(X), std::runtime_error);
}
