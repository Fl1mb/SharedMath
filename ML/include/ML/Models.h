#pragma once

// SharedMath::ML — Classical ML Models
//
// LinearRegression   — gradient-descent OLS
// LogisticRegression — binary logistic regression via gradient descent
// KMeans             — Lloyd's algorithm (random seed, configurable iters)
// KNNClassifier      — k-nearest neighbours (brute force, L2 distance)

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

class SHAREDMATH_ML_EXPORT LinearRegression {
public:
    explicit LinearRegression(double lr = 0.01, size_t max_iter = 1000);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    const Tensor& coef() const;
    double intercept() const;

private:
    double m_lr;
    size_t m_max_iter;
    bool   m_fitted = false;
    Tensor m_theta;
    double m_bias = 0.0;
};

class SHAREDMATH_ML_EXPORT LogisticRegression {
public:
    explicit LogisticRegression(double lr = 0.1, size_t max_iter = 500);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict_proba(const Tensor& X) const;
    Tensor predict(const Tensor& X, double threshold = 0.5) const;

    const Tensor& coef() const;
    double intercept() const;

private:
    double m_lr;
    size_t m_max_iter;
    bool   m_fitted = false;
    Tensor m_theta;
    double m_bias = 0.0;
};

class SHAREDMATH_ML_EXPORT KMeans {
public:
    explicit KMeans(size_t k, size_t max_iter = 300, uint64_t seed = 0);

    void fit(const Tensor& X);
    Tensor predict(const Tensor& X) const;

    const Tensor& centroids() const;
    const std::vector<size_t>& labels() const;
    size_t k() const noexcept;

private:
    size_t              m_k;
    size_t              m_max_iter;
    uint64_t            m_seed;
    bool                m_fitted = false;
    Tensor              m_centroids;
    std::vector<size_t> m_labels;
};

class SHAREDMATH_ML_EXPORT KNNClassifier {
public:
    explicit KNNClassifier(size_t k = 5);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    size_t k() const noexcept;

private:
    size_t m_k;
    bool   m_fitted = false;
    Tensor m_X_train;
    Tensor m_y_train;
};

} // namespace SharedMath::ML
