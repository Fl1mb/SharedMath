#pragma once

/**
 * @file TreeModels.h
 * @brief Decision trees and random forest ensembles.
 *
 * @defgroup ML_Trees Tree Models
 * @ingroup ML
 * @{
 */

/// SharedMath::ML — Decision Trees and Random Forests
///
/// DecisionTreeClassifier  — CART (gini / entropy) for multi-class problems
/// DecisionTreeRegressor   — CART (MSE) for continuous targets
/// RandomForestClassifier  — ensemble of DecisionTreeClassifiers
/// RandomForestRegressor   — ensemble of DecisionTreeRegressors

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

/// ─────────────────────────────────────────────────────────────────────────────
/// Internal tree node (opaque to the user)
/// ─────────────────────────────────────────────────────────────────────────────
namespace detail { struct TreeNode; }

/// ─────────────────────────────────────────────────────────────────────────────
/// DecisionTreeClassifier
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT DecisionTreeClassifier {
public:
    enum class Criterion { Gini, Entropy };

    explicit DecisionTreeClassifier(
        size_t    max_depth    = 0,       // 0 = unlimited
        size_t    min_samples  = 2,       // minimum samples to split a node
        Criterion criterion    = Criterion::Gini);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    size_t max_depth()   const noexcept;
    size_t min_samples() const noexcept;
    bool   fitted()      const noexcept;

private:
    size_t    m_max_depth;
    size_t    m_min_samples;
    Criterion m_criterion;
    bool      m_fitted = false;
    size_t    m_num_classes = 0;

    std::shared_ptr<detail::TreeNode> m_root;

    std::shared_ptr<detail::TreeNode>
    buildTree(const Tensor& X, const Tensor& y,
              const std::vector<size_t>& indices, size_t depth) const;

    double impurity(const std::vector<size_t>& indices,
                    const Tensor& y) const;

    double predictSample(const detail::TreeNode* node,
                         const Tensor& X, size_t row) const;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// DecisionTreeRegressor
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT DecisionTreeRegressor {
public:
    explicit DecisionTreeRegressor(
        size_t max_depth   = 0,
        size_t min_samples = 2);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    size_t max_depth()   const noexcept;
    size_t min_samples() const noexcept;
    bool   fitted()      const noexcept;

private:
    size_t m_max_depth;
    size_t m_min_samples;
    bool   m_fitted = false;

    std::shared_ptr<detail::TreeNode> m_root;

    std::shared_ptr<detail::TreeNode>
    buildTree(const Tensor& X, const Tensor& y,
              const std::vector<size_t>& indices, size_t depth) const;

    double predictSample(const detail::TreeNode* node,
                         const Tensor& X, size_t row) const;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// RandomForestClassifier
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT RandomForestClassifier {
public:
    explicit RandomForestClassifier(
        size_t   n_estimators = 100,
        size_t   max_depth    = 0,
        size_t   min_samples  = 2,
        uint64_t seed         = 0,
        DecisionTreeClassifier::Criterion criterion
            = DecisionTreeClassifier::Criterion::Gini);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    size_t n_estimators() const noexcept;
    bool   fitted()       const noexcept;

private:
    size_t   m_n_estimators;
    size_t   m_max_depth;
    size_t   m_min_samples;
    uint64_t m_seed;
    DecisionTreeClassifier::Criterion m_criterion;
    bool     m_fitted      = false;
    size_t   m_num_classes = 0;

    std::vector<DecisionTreeClassifier> m_trees;
};

/// ─────────────────────────────────────────────────────────────────────────────
/// RandomForestRegressor
/// ─────────────────────────────────────────────────────────────────────────────
class SHAREDMATH_ML_EXPORT RandomForestRegressor {
public:
    explicit RandomForestRegressor(
        size_t   n_estimators = 100,
        size_t   max_depth    = 0,
        size_t   min_samples  = 2,
        uint64_t seed         = 0);

    void fit(const Tensor& X, const Tensor& y);
    Tensor predict(const Tensor& X) const;

    size_t n_estimators() const noexcept;
    bool   fitted()       const noexcept;

private:
    size_t   m_n_estimators;
    size_t   m_max_depth;
    size_t   m_min_samples;
    uint64_t m_seed;
    bool     m_fitted = false;

    std::vector<DecisionTreeRegressor> m_trees;
};

} // namespace SharedMath::ML

/// @} // ML_Trees
