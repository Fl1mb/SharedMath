#pragma once

/**
 * @file ModelSelection.h
 * @brief Cross-validation and model selection utilities.
 *
 * @defgroup ML_ModelSelection Model Selection
 * @ingroup ML
 * @{
 *
 * ### Example — 5-fold CV with KNN
 * @code{.cpp}
 * KFold kf(5, true, 42);
 * auto folds = kf.split(X.dim(0));
 *
 * using ModelPtr = std::shared_ptr<KNNClassifier>;
 * auto scores = cross_val_score<ModelPtr>(
 *     X, y, folds,
 *     [](const Tensor& Xtr, const Tensor& ytr) {
 *         auto m = std::make_shared<KNNClassifier>(3);
 *         m->fit(Xtr, ytr); return m;
 *     },
 *     [](ModelPtr& m, const Tensor& Xv, const Tensor& yv) {
 *         return accuracy(m->predict(Xv), yv);
 *     });
 * @endcode
 *
 * @}
 */

// SharedMath::ML — Model Selection Utilities
//
// KFold           — stratification-agnostic k-fold split
// StratifiedKFold — class-balanced k-fold split
// cross_val_score — convenience function: trains and evaluates a model k times

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

/// A single fold: indices that belong to the validation (test) set.
/// The training set is everything else.
struct SHAREDMATH_ML_EXPORT Fold {
    std::vector<size_t> train_indices;
    std::vector<size_t> val_indices;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// KFold
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT KFold {
public:
    explicit KFold(size_t n_splits = 5, bool shuffle = false, uint64_t seed = 0);

    /// Split N samples into k folds.
    std::vector<Fold> split(size_t n_samples) const;

    size_t n_splits() const noexcept;

private:
    size_t   m_n_splits;
    bool     m_shuffle;
    uint64_t m_seed;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// StratifiedKFold — preserves class proportions in each fold
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT StratifiedKFold {
public:
    explicit StratifiedKFold(size_t n_splits = 5, bool shuffle = false, uint64_t seed = 0);

    /// labels: 1-D integer class labels (stored as doubles).
    std::vector<Fold> split(const Tensor& labels) const;

    size_t n_splits() const noexcept;

private:
    size_t   m_n_splits;
    bool     m_shuffle;
    uint64_t m_seed;
};

// ─────────────────────────────────────────────────────────────────────────────
// cross_val_score
//
// Calls `train_fn` with (X_train, y_train) and `score_fn` with
// (model fitted on train, X_val, y_val).  Returns one score per fold.
//
// Example usage:
//   auto scores = cross_val_score(
//       X, y, kfold,
//       [](const Tensor& Xtr, const Tensor& ytr) {
//           auto m = std::make_shared<KNNClassifier>(3);
//           m->fit(Xtr, ytr); return m;
//       },
//       [](auto& m, const Tensor& Xv, const Tensor& yv) {
//           return accuracy(m->predict(Xv), yv);
//       });
// ─────────────────────────────────────────────────────────────────────────────

template<typename Model>
std::vector<double> cross_val_score(
    const Tensor& X,
    const Tensor& y,
    const std::vector<Fold>& folds,
    std::function<Model(const Tensor&, const Tensor&)> train_fn,
    std::function<double(Model&, const Tensor&, const Tensor&)> score_fn)
{
    auto slice_rows = [](const Tensor& t, const std::vector<size_t>& idx) {
        if (t.ndim() == 1) {
            Tensor out = Tensor::zeros({idx.size()});
            for (size_t i = 0; i < idx.size(); ++i) out.flat(i) = t.flat(idx[i]);
            return out;
        }
        // 2-D
        const size_t D = t.dim(1);
        Tensor out = Tensor::zeros({idx.size(), D});
        for (size_t i = 0; i < idx.size(); ++i)
            for (size_t d = 0; d < D; ++d)
                out(i, d) = t(idx[i], d);
        return out;
    };

    std::vector<double> scores;
    scores.reserve(folds.size());
    for (const auto& fold : folds) {
        Tensor Xtr = slice_rows(X, fold.train_indices);
        Tensor ytr = slice_rows(y, fold.train_indices);
        Tensor Xv  = slice_rows(X, fold.val_indices);
        Tensor yv  = slice_rows(y, fold.val_indices);

        auto model = train_fn(Xtr, ytr);
        scores.push_back(score_fn(model, Xv, yv));
    }
    return scores;
}

} // namespace SharedMath::ML
