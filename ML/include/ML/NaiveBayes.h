#pragma once

// SharedMath::ML — Gaussian Naive Bayes classifier

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstddef>
#include <vector>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// GaussianNB
//
// Assumes each feature ~ N(μ_c, σ²_c) per class c.
// Prediction: argmax_c [ log P(c) + Σ_d log N(x_d | μ_cd, σ²_cd) ]
// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT GaussianNB {
public:
    explicit GaussianNB(double var_smoothing = 1e-9);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;
    Tensor predict_proba(const Tensor& X) const;   // [N, C] softmax-normalised log-probs

    size_t n_classes()  const noexcept;
    size_t n_features() const noexcept;
    bool   fitted()     const noexcept;

    // Per-class statistics
    const Tensor& class_prior()   const noexcept;  // [C] log priors
    const Tensor& theta()         const noexcept;  // [C, D] means
    const Tensor& sigma()         const noexcept;  // [C, D] variances

private:
    double m_var_smoothing;
    bool   m_fitted     = false;
    size_t m_n_classes  = 0;
    size_t m_n_features = 0;

    Tensor m_log_prior;  // [C]
    Tensor m_theta;      // [C, D] means
    Tensor m_sigma;      // [C, D] variances (including smoothing)
};

} // namespace SharedMath::ML
