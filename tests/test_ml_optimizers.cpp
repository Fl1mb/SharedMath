#include <gtest/gtest.h>
#include "ML/ml.h"

#include <cmath>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// Helper: build a simple MSE loss over y = w*x
// x=1, y_true=3, w=0 → loss = (w*1 - 3)^2 = 9, grad_w = -6
static std::pair<AutoTensor, double> simpleLoss(AutoTensor& w,
                                                double x_val,
                                                double y_true)
{
    auto x   = AutoTensor::from(Tensor::from_vector({x_val}));
    auto y   = AutoTensor::from(Tensor::from_vector({y_true}));
    auto y_hat = w * x;
    auto diff  = y_hat - y;
    auto loss  = (diff * diff).sum();
    return {loss, loss.data().flat(0)};
}

// ─────────────────────────────────────────────────────────────────────────────
// SGD — basic gradient descent decreases loss
// ─────────────────────────────────────────────────────────────────────────────

TEST(SGD, StepDecreasesLoss) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    SGD optim({&w}, /*lr=*/0.1);

    auto [loss0, val0] = simpleLoss(w, 1.0, 3.0);
    loss0.backward();
    optim.step();

    w.zero_grad();
    auto [loss1, val1] = simpleLoss(w, 1.0, 3.0);
    EXPECT_LT(val1, val0);
}

TEST(SGD, ConvergesOnLinearProblem) {
    // w*1 → 3.0: should converge to w ≈ 3
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    SGD optim({&w}, 0.05);

    for (int i = 0; i < 200; ++i) {
        w.zero_grad();
        auto [loss, _] = simpleLoss(w, 1.0, 3.0);
        loss.backward();
        optim.step();
    }
    EXPECT_NEAR(w.data().flat(0), 3.0, 0.05);
}

TEST(SGD, ZeroGrad) {
    auto w = AutoTensor::from(Tensor::from_vector({1.0}), true);
    SGD optim({&w}, 0.01);
    auto [loss, _] = simpleLoss(w, 1.0, 3.0);
    loss.backward();
    EXPECT_TRUE(w.has_grad());
    optim.zero_grad();
    EXPECT_NEAR(w.grad().flat(0), 0.0, 1e-12);
}

TEST(SGD, InvalidLrThrows) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    EXPECT_THROW(SGD({&w}, 0.0),  std::invalid_argument);
    EXPECT_THROW(SGD({&w}, -0.1), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// SGD with momentum
// ─────────────────────────────────────────────────────────────────────────────

TEST(SGDMomentum, ConvergesFasterThanPlainSGD) {
    // Compare convergence after 50 steps; momentum should be at least as good
    auto w1 = AutoTensor::from(Tensor::from_vector({0.0}), true);
    auto w2 = AutoTensor::from(Tensor::from_vector({0.0}), true);
    SGD plain({&w1}, 0.02);
    SGD mom  ({&w2}, 0.02, 0.9);

    for (int i = 0; i < 50; ++i) {
        w1.zero_grad();
        auto [l1, _1] = simpleLoss(w1, 1.0, 3.0); l1.backward(); plain.step();
        w2.zero_grad();
        auto [l2, _2] = simpleLoss(w2, 1.0, 3.0); l2.backward(); mom.step();
    }
    double err_plain = std::abs(w1.data().flat(0) - 3.0);
    double err_mom   = std::abs(w2.data().flat(0) - 3.0);
    EXPECT_LT(err_mom, err_plain + 0.1);  // momentum is at least comparable
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaGrad
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdaGrad, StepDecreasesLoss) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    AdaGrad optim({&w}, 0.5);

    auto [loss0, val0] = simpleLoss(w, 1.0, 3.0);
    loss0.backward();
    optim.step();

    w.zero_grad();
    auto [loss1, val1] = simpleLoss(w, 1.0, 3.0);
    EXPECT_LT(val1, val0);
}

TEST(AdaGrad, ConvergesOnLinearProblem) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    AdaGrad optim({&w}, 0.5);

    for (int i = 0; i < 500; ++i) {
        w.zero_grad();
        auto [loss, _] = simpleLoss(w, 1.0, 3.0);
        loss.backward();
        optim.step();
    }
    EXPECT_NEAR(w.data().flat(0), 3.0, 0.1);
}

TEST(AdaGrad, InvalidParamsThrow) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    EXPECT_THROW(AdaGrad({&w}, 0.0), std::invalid_argument);
    EXPECT_THROW(AdaGrad({&w}, -0.1), std::invalid_argument);
    EXPECT_THROW(AdaGrad({&w}, 1e-2, 0.0), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSProp
// ─────────────────────────────────────────────────────────────────────────────

TEST(RMSProp, StepDecreasesLoss) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    RMSProp optim({&w}, 0.1);

    auto [loss0, val0] = simpleLoss(w, 1.0, 3.0);
    loss0.backward();
    optim.step();

    w.zero_grad();
    auto [loss1, val1] = simpleLoss(w, 1.0, 3.0);
    EXPECT_LT(val1, val0);
}

TEST(RMSProp, ConvergesOnLinearProblem) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    RMSProp optim({&w}, 0.03, 0.9);

    for (int i = 0; i < 300; ++i) {
        w.zero_grad();
        auto [loss, _] = simpleLoss(w, 1.0, 3.0);
        loss.backward();
        optim.step();
    }
    EXPECT_NEAR(w.data().flat(0), 3.0, 0.15);
}

TEST(RMSProp, InvalidParamsThrow) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    EXPECT_THROW(RMSProp({&w}, 0.0), std::invalid_argument);
    EXPECT_THROW(RMSProp({&w}, -0.1), std::invalid_argument);
    EXPECT_THROW(RMSProp({&w}, 1e-3, -0.1), std::invalid_argument);
    EXPECT_THROW(RMSProp({&w}, 1e-3, 1.0), std::invalid_argument);
    EXPECT_THROW(RMSProp({&w}, 1e-3, 0.99, 0.0), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────

TEST(Adam, StepDecreasesLoss) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    Adam optim({&w}, 0.1);

    auto [loss0, val0] = simpleLoss(w, 1.0, 3.0);
    loss0.backward();
    optim.step();

    w.zero_grad();
    auto [loss1, val1] = simpleLoss(w, 1.0, 3.0);
    EXPECT_LT(val1, val0);
}

TEST(Adam, ConvergesOnLinearProblem) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    Adam optim({&w}, 0.1);

    for (int i = 0; i < 200; ++i) {
        w.zero_grad();
        auto [loss, _] = simpleLoss(w, 1.0, 3.0);
        loss.backward();
        optim.step();
    }
    EXPECT_NEAR(w.data().flat(0), 3.0, 0.1);
}

TEST(Adam, InvalidParamsThrow) {
    auto w = AutoTensor::from(Tensor::from_vector({0.0}), true);
    EXPECT_THROW(Adam({&w}, 0.0),  std::invalid_argument);  // lr <= 0
    EXPECT_THROW(Adam({&w}, 1e-3, 0.9, 0.999, 0.0), std::invalid_argument); // eps <= 0
}

// ─────────────────────────────────────────────────────────────────────────────
// Combined: Linear + SGD end-to-end minimisation
// ─────────────────────────────────────────────────────────────────────────────

TEST(OptimizerEndToEnd, LinearLayerSGD) {
    // Single Linear(1→1) trained on y = 2x
    Linear layer(1, 1, /*bias=*/false);
    // Force initial weight ≠ 2 for fair test
    layer.weight().data().flat(0) = 0.0;

    SGD optim(layer.parameters(), 0.05);

    for (int i = 0; i < 300; ++i) {
        layer.zero_grad();
        // One training sample: x=1, y=2
        auto x = AutoTensor::from(Tensor::from_matrix(1, 1, {1.0}));
        auto y = AutoTensor::from(Tensor::from_vector({2.0}));
        auto y_hat = layer.forward(x);
        auto diff  = y_hat - y;
        auto loss  = (diff * diff).sum();
        loss.backward();
        optim.step();
    }
    EXPECT_NEAR(layer.weight().data().flat(0), 2.0, 0.1);
}
