#include <gtest/gtest.h>
#include "ML/ml.h"

#include <cmath>
#include <memory>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Linear layer
// ─────────────────────────────────────────────────────────────────────────────

TEST(Linear, ForwardOutputShape) {
    Linear layer(4, 8);
    auto x = AutoTensor::from(Tensor::ones({3, 4}));    // batch=3, in=4
    auto y = layer.forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{3, 8}));
}

TEST(Linear, ParameterCount) {
    Linear layer(4, 8, /*use_bias=*/true);
    auto params = layer.parameters();
    ASSERT_EQ(params.size(), 2u);                       // weight + bias
    EXPECT_EQ(params[0]->data().shape(), (Tensor::Shape{4, 8})); // weight
    EXPECT_EQ(params[1]->data().shape(), (Tensor::Shape{1, 8})); // bias
}

TEST(Linear, ParameterCountNoBias) {
    Linear layer(4, 8, /*use_bias=*/false);
    EXPECT_EQ(layer.parameters().size(), 1u);
}

TEST(Linear, ForwardGradientFlows) {
    // Single-layer network: y = (x @ W + b).sum()
    Linear layer(3, 2);
    auto x = AutoTensor::from(Tensor::ones({2, 3}));    // no grad on input
    auto y = layer.forward(x).sum();
    y.backward();

    EXPECT_TRUE(layer.weight().has_grad());
    EXPECT_TRUE(layer.bias().has_grad());
}

TEST(Linear, InvalidInputDimThrows) {
    Linear layer(4, 8);
    auto x1d = AutoTensor::from(Tensor::ones({4}));       // must be 2-D
    EXPECT_THROW(layer.forward(x1d), std::invalid_argument);

    auto x_wrong = AutoTensor::from(Tensor::ones({3, 5})); // wrong feature dim
    EXPECT_THROW(layer.forward(x_wrong), std::invalid_argument);
}

TEST(Linear, ZeroInputGivesOnlyBias) {
    // If x = 0 then output = bias broadcast
    Linear layer(3, 2, /*use_bias=*/true);
    // Set bias to known value
    for (size_t j = 0; j < 2; ++j)
        layer.bias().data()(0, j) = static_cast<double>(j + 1);

    auto x = AutoTensor::from(Tensor::zeros({4, 3}));
    auto y = layer.forward(x);

    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 2; ++j)
            EXPECT_NEAR(y.data()(i, j), static_cast<double>(j + 1), 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequential
// ─────────────────────────────────────────────────────────────────────────────

TEST(Sequential, ForwardChain) {
    // [4→8] → [8→2]
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(4, 8));
    seq->add(std::make_shared<Linear>(8, 2));

    auto x = AutoTensor::from(Tensor::ones({5, 4}));
    auto y = seq->forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{5, 2}));
}

TEST(Sequential, ParametersAggregated) {
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(4, 8));    // weight[4,8] + bias[1,8]
    seq->add(std::make_shared<Linear>(8, 2));    // weight[8,2] + bias[1,2]

    auto params = seq->parameters();
    EXPECT_EQ(params.size(), 4u);
}

TEST(Sequential, GradientFlowsThroughLayers) {
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(2, 4));
    seq->add(std::make_shared<Linear>(4, 1));

    auto x = AutoTensor::from(Tensor::ones({3, 2}));
    auto y = seq->forward(x).sum();
    y.backward();

    for (auto* p : seq->parameters())
        EXPECT_TRUE(p->has_grad());
}

TEST(Sequential, TrainEvalPropagates) {
    auto seq = std::make_shared<Sequential>();
    seq->add(std::make_shared<Linear>(2, 2));
    seq->eval();
    EXPECT_FALSE(seq->is_training());
    seq->train();
    EXPECT_TRUE(seq->is_training());
}

// ─────────────────────────────────────────────────────────────────────────────
// Dropout
// ─────────────────────────────────────────────────────────────────────────────

TEST(Dropout, IdentityInEvalMode) {
    Dropout drop(0.5);
    drop.eval();
    auto x = AutoTensor::from(Tensor::ones({100}));
    auto y = drop.forward(x);
    // In eval mode: output == input
    for (size_t i = 0; i < 100; ++i)
        EXPECT_DOUBLE_EQ(y.data().flat(i), 1.0);
}

TEST(Dropout, ZeroDropoutIsIdentity) {
    Dropout drop(0.0);   // p=0 → no drop
    auto x = AutoTensor::from(Tensor::ones({50}));
    auto y = drop.forward(x);
    for (size_t i = 0; i < 50; ++i)
        EXPECT_DOUBLE_EQ(y.data().flat(i), 1.0);
}

TEST(Dropout, TrainingDropsSomeValues) {
    Dropout drop(0.5);
    drop.train();
    auto x = AutoTensor::from(Tensor::ones({1000}));
    auto y = drop.forward(x);
    // Roughly 50% should be zero (inverted scaling keeps mean ≈ 1)
    size_t zeros = 0;
    for (size_t i = 0; i < 1000; ++i)
        if (y.data().flat(i) == 0.0) ++zeros;
    // Allow generous statistical tolerance
    EXPECT_GT(zeros,  300u);
    EXPECT_LT(zeros,  700u);
}

TEST(Dropout, InvalidProbabilityThrows) {
    EXPECT_THROW(Dropout(-0.1), std::invalid_argument);
    EXPECT_THROW(Dropout(1.0),  std::invalid_argument);
}

TEST(Dropout, ZeroGradModule) {
    // Module::zero_grad() works for layers without parameters (Dropout)
    Dropout drop(0.3);
    EXPECT_NO_THROW(drop.zero_grad());  // empty parameters → no-op
}
