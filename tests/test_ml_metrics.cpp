#include <gtest/gtest.h>
#include "ML/ml.h"

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// Helper: build a 1-D label tensor from initializer list
static Tensor labels(std::initializer_list<double> vals) {
    std::vector<double> v(vals);
    return Tensor::from_vector(v);
}

// ─────────────────────────────────────────────────────────────────────────────
// accuracy
// ─────────────────────────────────────────────────────────────────────────────

TEST(Accuracy, AllCorrect) {
    auto pred = labels({0, 1, 0, 1});
    auto true_ = labels({0, 1, 0, 1});
    EXPECT_DOUBLE_EQ(accuracy(pred, true_), 1.0);
}

TEST(Accuracy, NoneCorrect) {
    auto pred  = labels({1, 1, 1});
    auto true_ = labels({0, 0, 0});
    EXPECT_DOUBLE_EQ(accuracy(pred, true_), 0.0);
}

TEST(Accuracy, Mixed) {
    auto pred  = labels({1, 0, 1, 1});
    auto true_ = labels({1, 1, 1, 0});
    EXPECT_DOUBLE_EQ(accuracy(pred, true_), 0.5);
}

TEST(Accuracy, ShapeMismatchThrows) {
    EXPECT_THROW(accuracy(labels({0, 1}), labels({0, 1, 0})),
                 std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// precision
// ─────────────────────────────────────────────────────────────────────────────

TEST(Precision, PerfectPrecision) {
    auto pred  = labels({1, 0, 1, 0});
    auto true_ = labels({1, 0, 1, 0});
    EXPECT_DOUBLE_EQ(precision(pred, true_, 1), 1.0);
}

TEST(Precision, ZeroPrecisionNoTP) {
    // Predicted all 0, positive class = 1 → TP=0, FP=0 → 0.0
    auto pred  = labels({0, 0, 0});
    auto true_ = labels({1, 1, 0});
    EXPECT_DOUBLE_EQ(precision(pred, true_, 1), 0.0);
}

TEST(Precision, KnownValue) {
    // pred=1: 3 times. TP=2, FP=1 → prec = 2/3
    auto pred  = labels({1, 1, 1, 0});
    auto true_ = labels({1, 0, 1, 0});
    EXPECT_NEAR(precision(pred, true_, 1), 2.0 / 3.0, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// recall
// ─────────────────────────────────────────────────────────────────────────────

TEST(Recall, PerfectRecall) {
    auto pred  = labels({1, 1, 1});
    auto true_ = labels({1, 1, 1});
    EXPECT_DOUBLE_EQ(recall(pred, true_, 1), 1.0);
}

TEST(Recall, KnownValue) {
    // true positives = 2, false negatives = 1 → recall = 2/3
    auto pred  = labels({1, 0, 1, 0});
    auto true_ = labels({1, 1, 1, 0});
    EXPECT_NEAR(recall(pred, true_, 1), 2.0 / 3.0, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// f1_score
// ─────────────────────────────────────────────────────────────────────────────

TEST(F1Score, PerfectF1) {
    auto pred  = labels({1, 0, 1, 0});
    auto true_ = labels({1, 0, 1, 0});
    EXPECT_DOUBLE_EQ(f1_score(pred, true_, 1), 1.0);
}

TEST(F1Score, ZeroF1) {
    // All predicted 0, positive class = 1
    auto pred  = labels({0, 0, 0});
    auto true_ = labels({1, 1, 0});
    EXPECT_DOUBLE_EQ(f1_score(pred, true_, 1), 0.0);
}

TEST(F1Score, KnownValue) {
    // prec = 2/3, recall = 2/3 → F1 = 2/3
    auto pred  = labels({1, 1, 1, 0});
    auto true_ = labels({1, 0, 1, 0});
    // prec = 2/3, recall = 2/2 = 1 → F1 = 2*(2/3*1)/(2/3+1) = (4/3)/(5/3) = 4/5
    double p = precision(pred, true_, 1);
    double r = recall(pred, true_, 1);
    double expected = 2.0 * p * r / (p + r);
    EXPECT_NEAR(f1_score(pred, true_, 1), expected, 1e-9);
}

// ─────────────────────────────────────────────────────────────────────────────
// confusion_matrix
// ─────────────────────────────────────────────────────────────────────────────

TEST(ConfusionMatrix, BinaryPerfect) {
    auto pred  = labels({0, 1, 0, 1});
    auto true_ = labels({0, 1, 0, 1});
    Tensor cm  = confusion_matrix(pred, true_, 2);

    ASSERT_EQ(cm.shape(), (Tensor::Shape{2, 2}));
    EXPECT_DOUBLE_EQ(cm(0, 0), 2.0);  // TN
    EXPECT_DOUBLE_EQ(cm(1, 1), 2.0);  // TP
    EXPECT_DOUBLE_EQ(cm(0, 1), 0.0);  // FP
    EXPECT_DOUBLE_EQ(cm(1, 0), 0.0);  // FN
}

TEST(ConfusionMatrix, BinaryWithErrors) {
    //               true  0  1  0  1
    auto pred  = labels({0, 0, 1, 1});
    auto true_ = labels({0, 1, 0, 1});
    // cm[true][pred]
    // TN=1, FN=1, FP=1, TP=1
    Tensor cm = confusion_matrix(pred, true_, 2);
    EXPECT_DOUBLE_EQ(cm(0, 0), 1.0);  // true=0, pred=0
    EXPECT_DOUBLE_EQ(cm(1, 0), 1.0);  // true=1, pred=0 (FN)
    EXPECT_DOUBLE_EQ(cm(0, 1), 1.0);  // true=0, pred=1 (FP)
    EXPECT_DOUBLE_EQ(cm(1, 1), 1.0);  // true=1, pred=1 (TP)
}

TEST(ConfusionMatrix, ThreeClass) {
    auto pred  = labels({0, 1, 2, 0, 1, 2});
    auto true_ = labels({0, 1, 2, 2, 0, 1});
    Tensor cm  = confusion_matrix(pred, true_, 3);
    ASSERT_EQ(cm.shape(), (Tensor::Shape{3, 3}));
    // Diagonal counts: class 0: 1, class 1: 1, class 2: 1
    EXPECT_DOUBLE_EQ(cm(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(cm(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(cm(2, 2), 1.0);
}

TEST(ConfusionMatrix, InferNumClasses) {
    auto pred  = labels({0, 1, 0, 1});
    auto true_ = labels({0, 1, 0, 1});
    // num_classes=0 → inferred as 2
    EXPECT_NO_THROW({
        Tensor cm = confusion_matrix(pred, true_);
        EXPECT_EQ(cm.shape(), (Tensor::Shape{2, 2}));
    });
}

TEST(ConfusionMatrix, ShapeMismatchThrows) {
    EXPECT_THROW(confusion_matrix(labels({0, 1}), labels({0, 1, 0})),
                 std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// Consistency checks across metrics
// ─────────────────────────────────────────────────────────────────────────────

TEST(MetricsConsistency, PrecisionRecallF1OnSameData) {
    auto pred  = labels({1, 1, 0, 1, 0, 0, 1, 1});
    auto true_ = labels({1, 0, 0, 1, 1, 0, 1, 0});

    double p = precision(pred, true_, 1);
    double r = recall   (pred, true_, 1);
    double f = f1_score (pred, true_, 1);

    // F1 = 2pr / (p+r)
    if (p + r > 0.0)
        EXPECT_NEAR(f, 2.0 * p * r / (p + r), 1e-9);
    else
        EXPECT_DOUBLE_EQ(f, 0.0);
}

TEST(MetricsConsistency, AccuracyFromConfusionMatrix) {
    auto pred  = labels({0, 1, 0, 1, 1});
    auto true_ = labels({0, 1, 1, 1, 0});

    Tensor cm   = confusion_matrix(pred, true_, 2);
    double diag = cm(0, 0) + cm(1, 1);
    double total = 0.0;
    for (size_t i = 0; i < cm.size(); ++i) total += cm.flat(i);

    EXPECT_NEAR(diag / total, accuracy(pred, true_), 1e-9);
}
