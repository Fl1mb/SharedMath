#pragma once

#include <sharedmath_ml_export.h>

#include "AutogradTensor.h"
#include "LinearAlgebra/Tensor.h"

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

class SHAREDMATH_ML_EXPORT Loss {
public:
    virtual ~Loss() = default;
    virtual AutoTensor forward(const AutoTensor& y_pred, const AutoTensor& y_true) = 0;
    AutoTensor operator()(const AutoTensor& y_pred, const AutoTensor& y_true);
};

class SHAREDMATH_ML_EXPORT MSELoss : public Loss {
public:
    AutoTensor forward(const AutoTensor& y_pred, const AutoTensor& y_true) override;
};

class SHAREDMATH_ML_EXPORT CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(double eps = 1e-12);

    // logits: [N, C], labels: [N] with integer class ids stored as doubles.
    AutoTensor forward(const AutoTensor& logits, const Tensor& labels);
    AutoTensor operator()(const AutoTensor& logits, const Tensor& labels);

private:
    double m_eps;
};

} // namespace SharedMath::ML
