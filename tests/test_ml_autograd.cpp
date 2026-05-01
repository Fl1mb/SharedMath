#include <gtest/gtest.h>
#include "ML/ml.h"

#include <cmath>
#include <vector>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Backward-compatibility: existing AutoTensor API still works
// ─────────────────────────────────────────────────────────────────────────────

TEST(AutoTensor, StoresDataAndRequiresGradFlag) {
    AutoTensor x = AutoTensor::from(Tensor::ones({2, 3}), true);
    EXPECT_TRUE(x.requires_grad());
    ASSERT_EQ(x.data().shape(), (Tensor::Shape{2, 3}));
    EXPECT_DOUBLE_EQ(x.data()(1, 2), 1.0);
}

TEST(AutoTensor, ZeroGradCreatesMatchingGradient) {
    AutoTensor x(Tensor::ones({2, 2}), true);
    EXPECT_FALSE(x.has_grad());
    x.zero_grad();
    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(x.grad().shape(), (Tensor::Shape{2, 2}));
    for (size_t i = 0; i < x.grad().size(); ++i)
        EXPECT_DOUBLE_EQ(x.grad().flat(i), 0.0);
}

TEST(AutoTensor, RejectsGradientShapeMismatch) {
    AutoTensor x(Tensor::ones({2, 2}), true);
    EXPECT_THROW(x.set_grad(Tensor::ones({4})), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// y = x * x  →  dy/dx = 2x
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, SquareGradient) {
    // x = [1, 2, 3]
    auto x = AutoTensor::from(Tensor::from_vector({1.0, 2.0, 3.0}), true);
    auto y = (x * x).sum();     // sum so we get a scalar
    y.backward();

    ASSERT_TRUE(x.has_grad());
    // dy/dx[i] = 2*x[i]
    EXPECT_NEAR(x.grad().flat(0), 2.0, 1e-9);
    EXPECT_NEAR(x.grad().flat(1), 4.0, 1e-9);
    EXPECT_NEAR(x.grad().flat(2), 6.0, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// y = mean(x * x)  →  dy/dx = 2x / n
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, MeanSquaredGradient) {
    // x = [1, 2, 3], n = 3
    auto x = AutoTensor::from(Tensor::from_vector({1.0, 2.0, 3.0}), true);
    auto y = (x * x).mean();
    y.backward();

    ASSERT_TRUE(x.has_grad());
    double n = 3.0;
    EXPECT_NEAR(x.grad().flat(0), 2.0 * 1.0 / n, 1e-9);
    EXPECT_NEAR(x.grad().flat(1), 2.0 * 2.0 / n, 1e-9);
    EXPECT_NEAR(x.grad().flat(2), 2.0 * 3.0 / n, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// y = relu(x)  →  dy/dx = 1 if x > 0, else 0
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, ReluGradient) {
    auto x = AutoTensor::from(Tensor::from_vector({-2.0, 0.0, 3.0}), true);
    auto y = x.relu().sum();
    y.backward();

    ASSERT_TRUE(x.has_grad());
    EXPECT_NEAR(x.grad().flat(0), 0.0, 1e-9);   // -2 → dead
    EXPECT_NEAR(x.grad().flat(1), 0.0, 1e-9);   //  0 → dead (subgradient = 0)
    EXPECT_NEAR(x.grad().flat(2), 1.0, 1e-9);   //  3 → alive
}

// ─────────────────────────────────────────────────────────────────────────────
// y = sigmoid(x)  →  dy/dx = σ(x)(1-σ(x))
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, SigmoidGradient) {
    auto x = AutoTensor::from(Tensor::from_vector({0.0, 1.0, -1.0}), true);
    auto y = x.sigmoid().sum();
    y.backward();

    ASSERT_TRUE(x.has_grad());
    auto sig = [](double v){ return 1.0 / (1.0 + std::exp(-v)); };
    for (size_t i = 0; i < 3; ++i) {
        double xi  = x.data().flat(i);
        double s   = sig(xi);
        double expected = s * (1.0 - s);
        EXPECT_NEAR(x.grad().flat(i), expected, 1e-9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// y = tanh(x)  →  dy/dx = 1 - tanh²(x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, TanhGradient) {
    auto x = AutoTensor::from(Tensor::from_vector({0.0, 0.5, -0.5}), true);
    auto y = x.tanh().sum();
    y.backward();

    for (size_t i = 0; i < 3; ++i) {
        double xi  = x.data().flat(i);
        double t   = std::tanh(xi);
        EXPECT_NEAR(x.grad().flat(i), 1.0 - t * t, 1e-9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// a.matmul(b).sum() — gradient w.r.t. both operands
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, MatmulSumGradient) {
    // A: [2,3]  B: [3,2]  →  C = A@B: [2,2]  →  y = sum(C): scalar
    // dL/dA = dL/dC @ B^T = ones(2,2) @ ones(3,2)^T = ones(2,3)*2
    // dL/dB = A^T @ dL/dC = ones(3,2) * 2

    auto a = AutoTensor::from(Tensor::ones({2, 3}), true);
    auto b = AutoTensor::from(Tensor::ones({3, 2}), true);
    auto y = a.matmul(b).sum();
    y.backward();

    ASSERT_TRUE(a.has_grad());
    ASSERT_TRUE(b.has_grad());

    // dA[i,j] = sum_k ones[i,k] = 2  (B has 2 columns)
    for (size_t i = 0; i < a.grad().size(); ++i)
        EXPECT_NEAR(a.grad().flat(i), 2.0, 1e-9);

    // dB[j,k] = sum_i A[i,j] = 2  (A has 2 rows)
    for (size_t i = 0; i < b.grad().size(); ++i)
        EXPECT_NEAR(b.grad().flat(i), 2.0, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// exp, log, pow
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, ExpGradient) {
    // d exp(x)/dx = exp(x)
    auto x = AutoTensor::from(Tensor::from_vector({0.0, 1.0, 2.0}), true);
    auto y = x.exp().sum();
    y.backward();
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(x.grad().flat(i), std::exp(x.data().flat(i)), 1e-9);
}

TEST(Autograd, LogGradient) {
    // d log(x)/dx = 1/x
    auto x = AutoTensor::from(Tensor::from_vector({1.0, 2.0, 4.0}), true);
    auto y = x.log().sum();
    y.backward();
    for (size_t i = 0; i < 3; ++i)
        EXPECT_NEAR(x.grad().flat(i), 1.0 / x.data().flat(i), 1e-9);
}

TEST(Autograd, PowGradient) {
    // d x^3/dx = 3x^2
    auto x = AutoTensor::from(Tensor::from_vector({1.0, 2.0, 3.0}), true);
    auto y = x.pow(3.0).sum();
    y.backward();
    for (size_t i = 0; i < 3; ++i) {
        double xi = x.data().flat(i);
        EXPECT_NEAR(x.grad().flat(i), 3.0 * xi * xi, 1e-9);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient accumulation (multiple backward calls)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, GradientsAccumulate) {
    auto x = AutoTensor::from(Tensor::from_vector({2.0}), true);
    // Two separate forward+backward pass without zero_grad → gradients should sum
    (x * x).sum().backward();   // grad = 4
    (x * x).sum().backward();   // grad += 4 → 8
    EXPECT_NEAR(x.grad().flat(0), 8.0, 1e-9);
}

TEST(Autograd, ZeroGradClearsAccumulation) {
    auto x = AutoTensor::from(Tensor::from_vector({2.0}), true);
    (x * x).sum().backward();
    x.zero_grad();
    (x * x).sum().backward();
    EXPECT_NEAR(x.grad().flat(0), 4.0, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// backward() on non-scalar throws
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, BackwardNonScalarThrows) {
    auto x = AutoTensor::from(Tensor::ones({3}), true);
    auto y = x * x;   // still shape {3}
    EXPECT_THROW(y.backward(), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Transpose gradient
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, TransposeGradient) {
    // y = sum(X^T), grad_X = ones same shape as X
    auto x = AutoTensor::from(Tensor::ones({2, 3}), true);
    auto y = x.T().sum();
    y.backward();
    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(x.grad().shape(), (Tensor::Shape{2, 3}));
    for (size_t i = 0; i < x.grad().size(); ++i)
        EXPECT_NEAR(x.grad().flat(i), 1.0, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Chain rule: y = sigmoid(x * x)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Autograd, ChainRuleSigmoidSquare) {
    auto x = AutoTensor::from(Tensor::from_vector({1.0}), true);
    auto y = (x * x).sigmoid().sum();
    y.backward();

    // d/dx sigmoid(x²) = sigmoid(x²)(1-sigmoid(x²)) * 2x
    double x0  = 1.0;
    double s   = 1.0 / (1.0 + std::exp(-x0 * x0));
    double expected = s * (1.0 - s) * 2.0 * x0;
    EXPECT_NEAR(x.grad().flat(0), expected, 1e-9);
}
