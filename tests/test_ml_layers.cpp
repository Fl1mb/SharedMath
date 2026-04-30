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

TEST(Conv2d, ForwardSingleChannelNoBias) {
    Conv2d conv(1, 1, 2, 1, 0, false);
    for (size_t i = 0; i < conv.weight().data().size(); ++i)
        conv.weight().data().flat(i) = 1.0;

    auto x = AutoTensor::from(Tensor({1, 1, 3, 3},
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9}));

    auto y = conv.forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{1, 1, 2, 2}));
    EXPECT_DOUBLE_EQ(y.data()(0, 0, 0, 0), 12.0);
    EXPECT_DOUBLE_EQ(y.data()(0, 0, 0, 1), 16.0);
    EXPECT_DOUBLE_EQ(y.data()(0, 0, 1, 0), 24.0);
    EXPECT_DOUBLE_EQ(y.data()(0, 0, 1, 1), 28.0);
}

TEST(Conv2d, BackwardComputesInputAndWeightGradients) {
    Conv2d conv(1, 1, 2, 1, 0, false);
    for (size_t i = 0; i < conv.weight().data().size(); ++i)
        conv.weight().data().flat(i) = 1.0;

    auto x = AutoTensor::from(Tensor({1, 1, 3, 3},
        {1, 2, 3,
         4, 5, 6,
         7, 8, 9}), true);

    auto loss = conv.forward(x).sum();
    loss.backward();

    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(conv.weight().has_grad());
    EXPECT_DOUBLE_EQ(x.grad()(0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(x.grad()(0, 0, 1, 1), 4.0);
    EXPECT_DOUBLE_EQ(x.grad()(0, 0, 2, 2), 1.0);
    EXPECT_DOUBLE_EQ(conv.weight().grad()(0, 0, 0, 0), 12.0);
    EXPECT_DOUBLE_EQ(conv.weight().grad()(0, 0, 1, 1), 28.0);
}

TEST(MaxPool2d, ForwardAndBackward) {
    MaxPool2d pool(2);
    auto x = AutoTensor::from(Tensor({1, 1, 2, 2}, {1, 5, 2, 4}), true);

    auto y = pool.forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{1, 1, 1, 1}));
    EXPECT_DOUBLE_EQ(y.data().flat(0), 5.0);

    y.sum().backward();
    EXPECT_DOUBLE_EQ(x.grad().flat(0), 0.0);
    EXPECT_DOUBLE_EQ(x.grad().flat(1), 1.0);
    EXPECT_DOUBLE_EQ(x.grad().flat(2), 0.0);
    EXPECT_DOUBLE_EQ(x.grad().flat(3), 0.0);
}

TEST(AvgPool2d, ForwardAndBackward) {
    AvgPool2d pool(2);
    auto x = AutoTensor::from(Tensor({1, 1, 2, 2}, {1, 5, 2, 4}), true);

    auto y = pool.forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{1, 1, 1, 1}));
    EXPECT_DOUBLE_EQ(y.data().flat(0), 3.0);

    y.sum().backward();
    for (size_t i = 0; i < x.grad().size(); ++i)
        EXPECT_DOUBLE_EQ(x.grad().flat(i), 0.25);
}

TEST(BatchNorm1d, NormalisesAndBackpropagatesAffineParams) {
    BatchNorm1d bn(2, 1e-12);
    auto x = AutoTensor::from(Tensor({2, 2}, {1, 3, 3, 7}), true);

    auto y = bn.forward(x);
    EXPECT_NEAR(y.data()(0, 0), -1.0, 1e-6);
    EXPECT_NEAR(y.data()(1, 0),  1.0, 1e-6);
    EXPECT_NEAR(y.data()(0, 1), -1.0, 1e-6);
    EXPECT_NEAR(y.data()(1, 1),  1.0, 1e-6);

    y.sum().backward();
    ASSERT_TRUE(bn.gamma().has_grad());
    ASSERT_TRUE(bn.beta().has_grad());
    EXPECT_NEAR(bn.gamma().grad().flat(0), 0.0, 1e-6);
    EXPECT_NEAR(bn.gamma().grad().flat(1), 0.0, 1e-6);
    EXPECT_DOUBLE_EQ(bn.beta().grad().flat(0), 2.0);
    EXPECT_DOUBLE_EQ(bn.beta().grad().flat(1), 2.0);
}

TEST(BatchNorm2d, NormalisesPerChannel) {
    BatchNorm2d bn(1, 1e-12);
    auto x = AutoTensor::from(Tensor({1, 1, 2, 2}, {1, 2, 3, 4}), true);

    auto y = bn.forward(x);
    EXPECT_NEAR(y.data().mean(), 0.0, 1e-9);
    EXPECT_NEAR(y.data().var(), 1.0, 1e-9);

    y.sum().backward();
    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(bn.gamma().has_grad());
    ASSERT_TRUE(bn.beta().has_grad());
    EXPECT_DOUBLE_EQ(bn.beta().grad().flat(0), 4.0);
}

TEST(Flatten, FlattensFeatureDimensionsAndBackpropagatesShape) {
    Flatten flatten;
    auto x = AutoTensor::from(Tensor({2, 3, 2}, {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12
    }), true);

    auto y = flatten.forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{2, 6}));
    EXPECT_DOUBLE_EQ(y.data()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(y.data()(1, 5), 12.0);

    y.sum().backward();
    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(x.grad().shape(), (Tensor::Shape{2, 3, 2}));
    for (size_t i = 0; i < x.grad().size(); ++i)
        EXPECT_DOUBLE_EQ(x.grad().flat(i), 1.0);
}

TEST(ActivationLayers, ReLUSigmoidTanhForward) {
    auto x = AutoTensor::from(Tensor::from_vector({-1.0, 0.0, 1.0}), true);

    auto relu = ReLU().forward(x);
    EXPECT_DOUBLE_EQ(relu.data().flat(0), 0.0);
    EXPECT_DOUBLE_EQ(relu.data().flat(1), 0.0);
    EXPECT_DOUBLE_EQ(relu.data().flat(2), 1.0);

    auto sigmoid = Sigmoid().forward(x);
    EXPECT_NEAR(sigmoid.data().flat(0), 1.0 / (1.0 + std::exp(1.0)), 1e-9);
    EXPECT_NEAR(sigmoid.data().flat(1), 0.5, 1e-9);

    auto tanh = Tanh().forward(x);
    EXPECT_NEAR(tanh.data().flat(0), std::tanh(-1.0), 1e-9);
    EXPECT_NEAR(tanh.data().flat(2), std::tanh(1.0), 1e-9);
}

TEST(GELU, ForwardAndBackwardAtZero) {
    auto x = AutoTensor::from(Tensor::from_vector({0.0}), true);

    auto y = GELU().forward(x);
    EXPECT_NEAR(y.data().flat(0), 0.0, 1e-12);

    y.backward();
    ASSERT_TRUE(x.has_grad());
    EXPECT_NEAR(x.grad().flat(0), 0.5, 1e-12);
}

TEST(Softmax, ForwardNormalisesRows) {
    Softmax softmax(1);
    auto x = AutoTensor::from(Tensor({2, 3}, {
        1.0, 2.0, 3.0,
        1.0, 1.0, 1.0
    }));

    auto y = softmax.forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{2, 3}));
    EXPECT_NEAR(y.data()(0, 0) + y.data()(0, 1) + y.data()(0, 2), 1.0, 1e-12);
    EXPECT_NEAR(y.data()(1, 0), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(y.data()(1, 1), 1.0 / 3.0, 1e-12);
    EXPECT_NEAR(y.data()(1, 2), 1.0 / 3.0, 1e-12);
}

TEST(Softmax, BackwardMatchesTwoClassJacobian) {
    Softmax softmax(0);
    auto x = AutoTensor::from(Tensor::from_vector({0.0, 0.0}), true);

    auto y = softmax.forward(x);
    y.backward(Tensor::from_vector({1.0, 0.0}));

    ASSERT_TRUE(x.has_grad());
    EXPECT_NEAR(x.grad().flat(0), 0.25, 1e-12);
    EXPECT_NEAR(x.grad().flat(1), -0.25, 1e-12);
}

TEST(Sequential, ConvActivationFlattenLinearChain) {
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Conv2d>(1, 2, 3, 1, 1));
    model->add(std::make_shared<ReLU>());
    model->add(std::make_shared<MaxPool2d>(2));
    model->add(std::make_shared<Flatten>());
    model->add(std::make_shared<Linear>(8, 3));

    auto x = AutoTensor::from(Tensor::ones({1, 1, 4, 4}), true);
    auto y = model->forward(x);

    ASSERT_EQ(y.data().shape(), (Tensor::Shape{1, 3}));
    y.sum().backward();
    ASSERT_TRUE(x.has_grad());
    for (auto* p : model->parameters())
        EXPECT_TRUE(p->has_grad());
}

TEST(LayerNorm, NormalisesLastDimensionAndBackpropagatesParams) {
    LayerNorm ln(3, 1e-12);
    auto x = AutoTensor::from(Tensor({2, 3}, {1, 2, 3, 2, 4, 6}), true);

    auto y = ln.forward(x);
    EXPECT_NEAR(y.data()(0, 0) + y.data()(0, 1) + y.data()(0, 2), 0.0, 1e-9);
    EXPECT_NEAR(y.data()(1, 0) + y.data()(1, 1) + y.data()(1, 2), 0.0, 1e-9);

    y.sum().backward();
    ASSERT_TRUE(x.has_grad());
    ASSERT_TRUE(ln.gamma().has_grad());
    ASSERT_TRUE(ln.beta().has_grad());
    EXPECT_DOUBLE_EQ(ln.beta().grad().flat(0), 2.0);
}

TEST(Embedding, ForwardAndBackwardAccumulatesRepeatedTokenGradients) {
    Embedding emb(4, 2);
    for (size_t i = 0; i < emb.weight().data().size(); ++i)
        emb.weight().data().flat(i) = static_cast<double>(i + 1);

    auto ids = AutoTensor::from(Tensor::from_vector({1, 2, 1}));
    auto y = emb.forward(ids);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{3, 2}));
    EXPECT_DOUBLE_EQ(y.data()(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(y.data()(2, 1), 4.0);

    y.sum().backward();
    ASSERT_TRUE(emb.weight().has_grad());
    EXPECT_DOUBLE_EQ(emb.weight().grad()(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(emb.weight().grad()(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(emb.weight().grad()(2, 0), 1.0);
}

TEST(MultiHeadAttention, ForwardShapeAndAllProjectionGradients) {
    MultiHeadAttention mha(4, 2);
    auto x = AutoTensor::from(Tensor({2, 3, 4}, {
        1.0, 0.0, 0.5, -0.5,
        0.2, 1.0, -0.1, 0.3,
        -0.4, 0.7, 1.2, 0.1,
        0.3, -0.8, 0.9, 1.0,
        1.1, 0.4, -0.6, 0.2,
        -0.2, 0.5, 0.8, -1.0
    }), true);

    auto y = mha.forward(x);
    ASSERT_EQ(y.data().shape(), (Tensor::Shape{2, 3, 4}));
    y.sum().backward();

    auto params = mha.parameters();
    ASSERT_EQ(params.size(), 4u);
    EXPECT_TRUE(x.has_grad());
    for (auto* p : params) {
        EXPECT_TRUE(p->has_grad());
    }
}

TEST(ModuleSerialization, SaveAndLoadLinearParameters) {
    Linear src(2, 1);
    src.weight().data()(0, 0) = 3.0;
    src.weight().data()(1, 0) = 4.0;
    src.bias().data()(0, 0) = 5.0;

    const std::string path = "/tmp/sharedmath_linear_params.bin";
    src.save(path);

    Linear dst(2, 1);
    dst.load(path);

    EXPECT_DOUBLE_EQ(dst.weight().data()(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(dst.weight().data()(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(dst.bias().data()(0, 0), 5.0);
}
