#pragma once

/**
 * @file Preprocessing.h
 * @brief Data preprocessing utilities.
 *
 * @defgroup ML_Preprocessing Preprocessing
 * @ingroup ML
 * @{
 */

/// SharedMath::ML — Data Preprocessing
///
/// StandardScaler  — zero-mean / unit-variance normalisation
/// MinMaxScaler    — scale features to [feature_range_min, feature_range_max]
/// train_test_split — random train/test split of (X, y) datasets

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstdint>
#include <utility>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

/// ─────────────────────────────────────────────────────────────────────────────
/// StandardScaler — z-score normalisation per feature column
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT StandardScaler {
public:
    StandardScaler() = default;

    void fit(const Tensor& X);
    Tensor transform(const Tensor& X) const;
    Tensor fit_transform(const Tensor& X);
    Tensor inverse_transform(const Tensor& X) const;

    const Tensor& mean()  const noexcept;
    const Tensor& scale() const noexcept;   // std deviation per feature
    bool fitted() const noexcept;

private:
    Tensor m_mean;
    Tensor m_scale;
    bool   m_fitted = false;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// MinMaxScaler — scale features to a given range (default [0, 1])
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT MinMaxScaler {
public:
    explicit MinMaxScaler(double feature_min = 0.0, double feature_max = 1.0);

    void fit(const Tensor& X);
    Tensor transform(const Tensor& X) const;
    Tensor fit_transform(const Tensor& X);
    Tensor inverse_transform(const Tensor& X) const;

    const Tensor& data_min()   const noexcept;
    const Tensor& data_max()   const noexcept;
    const Tensor& data_range() const noexcept;   // data_max - data_min
    bool fitted() const noexcept;

private:
    double m_feature_min;
    double m_feature_max;
    Tensor m_data_min;
    Tensor m_data_max;
    Tensor m_data_range;
    bool   m_fitted = false;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// train_test_split — split (X, y) into (X_train, X_test, y_train, y_test)
///
/// test_size : fraction of samples to assign to the test set (e.g. 0.2 = 20%)
/// shuffle   : whether to shuffle before splitting
/// seed      : RNG seed used when shuffle == true
/// ─────────────────────────────────────────────────────────────────────────────
struct SHAREDMATH_ML_EXPORT TrainTestSplit {
    Tensor X_train;
    Tensor X_test;
    Tensor y_train;
    Tensor y_test;
};

SHAREDMATH_ML_EXPORT
TrainTestSplit train_test_split(const Tensor& X,
                                const Tensor& y,
                                double   test_size = 0.2,
                                bool     shuffle   = true,
                                uint64_t seed      = 0);

} // namespace SharedMath::ML

/// @} // ML_Preprocessing
