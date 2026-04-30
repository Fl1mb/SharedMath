#include <gtest/gtest.h>
#include "ML/ml.h"

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

TEST(Losses, MSELossComputesGradient) {
    auto pred = AutoTensor::from(Tensor::from_vector({1.0, 3.0}), true);
    auto target = AutoTensor::from(Tensor::from_vector({0.0, 1.0}));

    MSELoss loss_fn;
    auto loss = loss_fn.forward(pred, target);
    EXPECT_DOUBLE_EQ(loss.data().flat(0), 2.5);
    loss.backward();
    EXPECT_DOUBLE_EQ(pred.grad().flat(0), 1.0);
    EXPECT_DOUBLE_EQ(pred.grad().flat(1), 2.0);
}

TEST(Losses, CrossEntropyLossComputesGradient) {
    auto logits = AutoTensor::from(Tensor({2, 2}, {2.0, 0.0, 0.0, 2.0}), true);
    Tensor labels = Tensor::from_vector({0.0, 1.0});

    CrossEntropyLoss loss_fn;
    auto loss = loss_fn.forward(logits, labels);
    EXPECT_LT(loss.data().flat(0), 0.2);
    loss.backward();
    ASSERT_TRUE(logits.has_grad());
    EXPECT_LT(logits.grad()(0, 0), 0.0);
    EXPECT_GT(logits.grad()(0, 1), 0.0);
}

TEST(Trainer, FitsSimpleLinearRegressionWithMSE) {
    Tensor X({4, 1}, {0.0, 1.0, 2.0, 3.0});
    Tensor y({4, 1}, {1.0, 3.0, 5.0, 7.0});
    TensorDataset dataset(X, y);
    DataLoader loader(dataset, 4, false);

    Linear model(1, 1);
    model.weight().data().flat(0) = 0.0;
    model.bias().data().flat(0) = 0.0;

    SGD opt(model.parameters(), 0.05);
    MSELoss loss;
    Trainer trainer(model, opt, loss);
    auto losses = trainer.fit(loader, 120);

    ASSERT_FALSE(losses.empty());
    EXPECT_LT(losses.back(), losses.front());
    EXPECT_NEAR(model.weight().data().flat(0), 2.0, 0.2);
    EXPECT_NEAR(model.bias().data().flat(0), 1.0, 0.3);
}
