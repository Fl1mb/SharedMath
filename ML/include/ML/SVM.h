#pragma once

// SharedMath::ML — Linear Support Vector Machine
//
// LinearSVM — binary SVM trained via mini-batch subgradient (hinge loss + L2)
//             Labels must be {0, 1} or {-1, +1} (auto-detected).

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstddef>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

class SHAREDMATH_ML_EXPORT LinearSVM {
public:
    explicit LinearSVM(double C        = 1.0,
                       double lr       = 1e-3,
                       size_t max_iter = 1000);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;
    Tensor decision_function(const Tensor& X) const;

    const Tensor& coef()      const;
    double        intercept() const noexcept;
    bool          fitted()    const noexcept;

private:
    double m_C;
    double m_lr;
    size_t m_max_iter;
    bool   m_fitted = false;
    Tensor m_w;
    double m_b = 0.0;
};

} // namespace SharedMath::ML
