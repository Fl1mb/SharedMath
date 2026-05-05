#pragma once

/// SharedMath::ML — Regularized Linear Models
///
/// RidgeRegression — OLS with L2 penalty (closed-form via gradient descent)
/// LassoRegression — OLS with L1 penalty (coordinate descent)
/// ElasticNet      — convex combination of L1 and L2 penalties

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstddef>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

/// ─────────────────────────────────────────────────────────────────────────────
/// RidgeRegression  (L2 regularisation)
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT RidgeRegression {
public:
    explicit RidgeRegression(double alpha   = 1.0,
                             double lr      = 0.01,
                             size_t max_iter = 1000);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    const Tensor& coef()      const;
    double        intercept() const noexcept;
    bool          fitted()    const noexcept;

private:
    double m_alpha;
    double m_lr;
    size_t m_max_iter;
    bool   m_fitted = false;
    Tensor m_theta;
    double m_bias = 0.0;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// LassoRegression  (L1 regularisation via coordinate descent)
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT LassoRegression {
public:
    explicit LassoRegression(double alpha    = 1.0,
                             size_t max_iter = 1000,
                             double tol      = 1e-4);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    const Tensor& coef()      const;
    double        intercept() const noexcept;
    bool          fitted()    const noexcept;

private:
    double m_alpha;
    size_t m_max_iter;
    double m_tol;
    bool   m_fitted = false;
    Tensor m_theta;
    double m_bias = 0.0;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// ElasticNet  (l1_ratio * L1 + (1 - l1_ratio) * L2)
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT ElasticNet {
public:
    explicit ElasticNet(double alpha    = 1.0,
                        double l1_ratio = 0.5,
                        size_t max_iter = 1000,
                        double tol      = 1e-4);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    const Tensor& coef()      const;
    double        intercept() const noexcept;
    bool          fitted()    const noexcept;

private:
    double m_alpha;
    double m_l1_ratio;
    size_t m_max_iter;
    double m_tol;
    bool   m_fitted = false;
    Tensor m_theta;
    double m_bias = 0.0;
};

} // namespace SharedMath::ML
