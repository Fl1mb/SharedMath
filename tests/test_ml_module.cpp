#include <gtest/gtest.h>

#include "ML/ml.h"

using namespace SharedMath::LinearAlgebra;
using namespace SharedMath::ML;

TEST(MLModule, AutoTensorStoresDataAndRequiresGradFlag) {
    AutoTensor x = AutoTensor::from(Tensor::ones({2, 3}), true);

    EXPECT_TRUE(x.requires_grad());
    ASSERT_EQ(x.data().shape(), (Tensor::Shape{2, 3}));
    EXPECT_DOUBLE_EQ(x.data()(1, 2), 1.0);
}

TEST(MLModule, AutoTensorZeroGradCreatesMatchingGradient) {
    AutoTensor x(Tensor::ones({2, 2}), true);

    EXPECT_FALSE(x.has_grad());
    x.zero_grad();

    ASSERT_TRUE(x.has_grad());
    ASSERT_EQ(x.grad().shape(), (Tensor::Shape{2, 2}));
    for (size_t i = 0; i < x.grad().size(); ++i)
        EXPECT_DOUBLE_EQ(x.grad().flat(i), 0.0);
}

TEST(MLModule, AutoTensorRejectsGradientShapeMismatch) {
    AutoTensor x(Tensor::ones({2, 2}), true);
    EXPECT_THROW(x.set_grad(Tensor::ones({4})), std::invalid_argument);
}
