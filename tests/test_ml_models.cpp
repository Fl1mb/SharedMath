#include <gtest/gtest.h>
#include "ML/ml.h"

#include <cmath>
#include <vector>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// LinearRegression — y = 2x + 1
// ─────────────────────────────────────────────────────────────────────────────

TEST(LinearRegression, FitsLinearData) {
    // Training data: y = 2*x + 1
    const size_t N = 20;
    Tensor X = Tensor::zeros({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        double xi = static_cast<double>(i);
        X(i, 0) = xi;
        y.flat(i) = 2.0 * xi + 1.0;
    }

    LinearRegression model(0.001, 3000);
    model.fit(X, y);

    // Coefficients should be close to w=2, b=1
    EXPECT_NEAR(model.coef().flat(0), 2.0, 0.1);
    EXPECT_NEAR(model.intercept(),    1.0, 0.5);
}

TEST(LinearRegression, PredictApproximatesGroundTruth) {
    const size_t N = 10;
    Tensor X = Tensor::zeros({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        X(i, 0) = static_cast<double>(i);
        y.flat(i) = 3.0 * static_cast<double>(i) + 2.0;
    }

    LinearRegression model(0.001, 3000);
    model.fit(X, y);

    Tensor X_test = Tensor::zeros({3, 1});
    X_test(0, 0) = 10.0; X_test(1, 0) = 11.0; X_test(2, 0) = 12.0;

    Tensor pred = model.predict(X_test);
    EXPECT_NEAR(pred.flat(0), 32.0, 2.0);   // 3*10 + 2
    EXPECT_NEAR(pred.flat(1), 35.0, 2.0);
    EXPECT_NEAR(pred.flat(2), 38.0, 2.0);
}

TEST(LinearRegression, PredictBeforeFitThrows) {
    LinearRegression model;
    Tensor X = Tensor::ones({3, 2});
    EXPECT_THROW(model.predict(X), std::runtime_error);
}

TEST(LinearRegression, InvalidInputThrows) {
    Tensor X = Tensor::ones({5, 2});
    Tensor y = Tensor::ones({5});
    LinearRegression model;
    model.fit(X, y);

    Tensor X_bad = Tensor::ones({3, 3}); // wrong feature dim
    EXPECT_THROW(model.predict(X_bad), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// LogisticRegression — binary separable dataset
// ─────────────────────────────────────────────────────────────────────────────

TEST(LogisticRegression, BinarySeparableData) {
    // Class 0: x < 0,  Class 1: x > 0
    const size_t N = 20;
    Tensor X = Tensor::zeros({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N / 2; ++i) {
        X(i, 0) = -static_cast<double>(i + 1);
        y.flat(i) = 0.0;
    }
    for (size_t i = N / 2; i < N; ++i) {
        X(i, 0) = static_cast<double>(i - N / 2 + 1);
        y.flat(i) = 1.0;
    }

    LogisticRegression model(0.5, 500);
    model.fit(X, y);

    Tensor pred = model.predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < N; ++i)
        if (static_cast<int>(pred.flat(i)) == static_cast<int>(y.flat(i)))
            ++correct;
    EXPECT_GE(correct, 18u);   // should get > 90% on perfectly separable data
}

TEST(LogisticRegression, PredictProbaInZeroOne) {
    Tensor X = Tensor::zeros({4, 1});
    Tensor y = Tensor::zeros({4});
    X(0,0) = -2; X(1,0) = -1; X(2,0) = 1; X(3,0) = 2;
    y.flat(0) = 0; y.flat(1) = 0; y.flat(2) = 1; y.flat(3) = 1;

    LogisticRegression model(0.5, 300);
    model.fit(X, y);

    Tensor proba = model.predict_proba(X);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GT(proba.flat(i), 0.0);
        EXPECT_LT(proba.flat(i), 1.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KMeans — two clearly separated clusters
// ─────────────────────────────────────────────────────────────────────────────

TEST(KMeans, TwoClustersSeparated) {
    // 10 points near (0,0) and 10 points near (10,10)
    const size_t N = 20;
    Tensor X = Tensor::zeros({N, 2});
    for (size_t i = 0; i < 10; ++i) {
        X(i, 0) = static_cast<double>(i % 3) * 0.1;
        X(i, 1) = static_cast<double>(i % 3) * 0.1;
    }
    for (size_t i = 10; i < N; ++i) {
        X(i, 0) = 10.0 + static_cast<double>((i-10) % 3) * 0.1;
        X(i, 1) = 10.0 + static_cast<double>((i-10) % 3) * 0.1;
    }

    KMeans km(2, 100, 0);
    km.fit(X);

    Tensor labels = km.predict(X);

    // All points in cluster 0..9 should have the same label
    int lab_a = static_cast<int>(labels.flat(0));
    int lab_b = (lab_a == 0) ? 1 : 0;
    for (size_t i = 0; i < 10;  ++i)
        EXPECT_EQ(static_cast<int>(labels.flat(i)), lab_a);
    for (size_t i = 10; i < N; ++i)
        EXPECT_EQ(static_cast<int>(labels.flat(i)), lab_b);
}

TEST(KMeans, CentroidsShape) {
    Tensor X = Tensor::ones({10, 3});
    KMeans km(4, 10, 0);
    km.fit(X);
    EXPECT_EQ(km.centroids().shape(), (Tensor::Shape{4, 3}));
}

TEST(KMeans, InvalidKThrows) {
    EXPECT_THROW(KMeans(0), std::invalid_argument);
}

TEST(KMeans, PredictBeforeFitThrows) {
    KMeans km(2);
    Tensor X = Tensor::ones({5, 2});
    EXPECT_THROW(km.predict(X), std::runtime_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// KNNClassifier
// ─────────────────────────────────────────────────────────────────────────────

TEST(KNNClassifier, PerfectClassificationTwoClusters) {
    // Training: class 0 at x=0, class 1 at x=10
    Tensor X_train = Tensor::zeros({6, 1});
    Tensor y_train = Tensor::zeros({6});
    for (size_t i = 0; i < 3; ++i) {
        X_train(i, 0) = static_cast<double>(i) * 0.1;
        y_train.flat(i) = 0.0;
    }
    for (size_t i = 3; i < 6; ++i) {
        X_train(i, 0) = 10.0 + static_cast<double>(i-3) * 0.1;
        y_train.flat(i) = 1.0;
    }

    KNNClassifier knn(3);
    knn.fit(X_train, y_train);

    Tensor X_test = Tensor::zeros({2, 1});
    X_test(0, 0) = 0.2;    // close to class 0
    X_test(1, 0) = 10.2;   // close to class 1

    Tensor pred = knn.predict(X_test);
    EXPECT_EQ(static_cast<int>(pred.flat(0)), 0);
    EXPECT_EQ(static_cast<int>(pred.flat(1)), 1);
}

TEST(KNNClassifier, InvalidKThrows) {
    EXPECT_THROW(KNNClassifier(0), std::invalid_argument);
}

TEST(KNNClassifier, PredictBeforeFitThrows) {
    KNNClassifier knn(3);
    Tensor X = Tensor::ones({3, 2});
    EXPECT_THROW(knn.predict(X), std::runtime_error);
}

TEST(KNNClassifier, FeatureDimMismatchThrows) {
    Tensor X_train = Tensor::ones({4, 2});
    Tensor y_train = Tensor::zeros({4});
    KNNClassifier knn(2);
    knn.fit(X_train, y_train);

    Tensor X_test = Tensor::ones({2, 3}); // wrong dim
    EXPECT_THROW(knn.predict(X_test), std::invalid_argument);
}
