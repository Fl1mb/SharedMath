#pragma once

/**
 * @file Loss.h
 * @brief Loss functions for training neural networks.
 *
 * @defgroup ML_Losses Loss Functions
 * @ingroup ML
 * @{
 *
 * | Class | Formula |
 * |-------|---------|
 * | MSELoss | mean((ŷ - y)²) |
 * | CrossEntropyLoss | softmax + NLL (logits in, integer labels) |
 * | BCELoss | −mean(y log p + (1-y) log(1-p)) |
 * | BCEWithLogitsLoss | numerically stable BCE from raw logits |
 * | HuberLoss | smooth L1 with configurable δ |
 *
 * @}
 */

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

    /// logits: [N, C], labels: [N] with integer class ids stored as doubles.
    AutoTensor forward(const AutoTensor& logits, const Tensor& labels);
    AutoTensor operator()(const AutoTensor& logits, const Tensor& labels);

private:
    double m_eps;
};

/// Huber loss: L2 for |err| <= delta, L1-like otherwise.
class SHAREDMATH_ML_EXPORT HuberLoss : public Loss {
public:
    explicit HuberLoss(double delta = 1.0);

    AutoTensor forward(const AutoTensor& y_pred, const AutoTensor& y_true) override;

    double delta() const noexcept;

private:
    double m_delta;
};

/// Binary Cross-Entropy: expects probabilities in (0,1).
class SHAREDMATH_ML_EXPORT BCELoss : public Loss {
public:
    explicit BCELoss(double eps = 1e-12);

    AutoTensor forward(const AutoTensor& y_pred, const AutoTensor& y_true) override;

    double eps() const noexcept;

private:
    double m_eps;
};

/// Binary Cross-Entropy with logits (numerically stable).
class SHAREDMATH_ML_EXPORT BCEWithLogitsLoss : public Loss {
public:
    AutoTensor forward(const AutoTensor& y_pred, const AutoTensor& y_true) override;
};

} // namespace SharedMath::ML
