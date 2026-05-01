#include "TreeModels.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace SharedMath::ML {

// ─────────────────────────────────────────────────────────────────────────────
// Internal node
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

struct TreeNode {
    // Split info (for internal nodes)
    size_t feature_idx = 0;
    double threshold   = 0.0;
    // Leaf value
    double leaf_value  = 0.0;   // for regressor or majority class
    std::vector<double> class_counts; // for classifier leaf

    std::shared_ptr<TreeNode> left;
    std::shared_ptr<TreeNode> right;

    bool is_leaf() const noexcept { return !left && !right; }
};

} // namespace detail

namespace {

// Compute mean of y over given indices
double mean_y(const Tensor& y, const std::vector<size_t>& idx) {
    double s = 0.0;
    for (size_t i : idx) s += y.flat(i);
    return idx.empty() ? 0.0 : s / static_cast<double>(idx.size());
}

// MSE of y over given indices
double mse_y(const Tensor& y, const std::vector<size_t>& idx) {
    if (idx.empty()) return 0.0;
    double m = mean_y(y, idx);
    double s = 0.0;
    for (size_t i : idx) {
        double d = y.flat(i) - m;
        s += d * d;
    }
    return s / static_cast<double>(idx.size());
}

// Find majority class label
double majority_class(const Tensor& y, const std::vector<size_t>& idx,
                      size_t num_classes)
{
    std::vector<size_t> cnt(num_classes, 0);
    for (size_t i : idx) ++cnt[static_cast<size_t>(y.flat(i))];
    return static_cast<double>(
        std::max_element(cnt.begin(), cnt.end()) - cnt.begin());
}

bool all_same_y(const Tensor& y, const std::vector<size_t>& idx) {
    if (idx.size() < 2) return true;
    const double first = y.flat(idx.front());
    for (size_t i : idx) {
        if (y.flat(i) != first) return false;
    }
    return true;
}

// Gini impurity
double gini(const Tensor& y, const std::vector<size_t>& idx, size_t nc) {
    if (idx.empty()) return 0.0;
    double n = static_cast<double>(idx.size());
    std::vector<size_t> cnt(nc, 0);
    for (size_t i : idx) ++cnt[static_cast<size_t>(y.flat(i))];
    double imp = 1.0;
    for (size_t c = 0; c < nc; ++c) {
        double p = cnt[c] / n;
        imp -= p * p;
    }
    return imp;
}

// Entropy impurity
double entropy(const Tensor& y, const std::vector<size_t>& idx, size_t nc) {
    if (idx.empty()) return 0.0;
    double n = static_cast<double>(idx.size());
    std::vector<size_t> cnt(nc, 0);
    for (size_t i : idx) ++cnt[static_cast<size_t>(y.flat(i))];
    double ent = 0.0;
    for (size_t c = 0; c < nc; ++c) {
        if (cnt[c] == 0) continue;
        double p = cnt[c] / n;
        ent -= p * std::log2(p);
    }
    return ent;
}

// Generic best-split finder (returns best feature, threshold, left/right indices)
struct SplitResult {
    size_t feature;
    double threshold;
    std::vector<size_t> left;
    std::vector<size_t> right;
    bool valid = false;
};

// Regression split on MSE reduction
SplitResult bestSplitRegressor(const Tensor& X, const Tensor& y,
                                const std::vector<size_t>& idx)
{
    const size_t D = X.dim(1);
    double parent_mse = mse_y(y, idx) * static_cast<double>(idx.size());
    double best_gain  = 0.0;
    SplitResult best;

    for (size_t f = 0; f < D; ++f) {
        // Collect (value, index) pairs for this feature
        std::vector<std::pair<double, size_t>> vals;
        vals.reserve(idx.size());
        for (size_t i : idx) vals.push_back({X(i, f), i});
        std::sort(vals.begin(), vals.end());

        for (size_t t = 0; t + 1 < vals.size(); ++t) {
            if (vals[t].first == vals[t + 1].first) continue;
            double threshold = 0.5 * (vals[t].first + vals[t + 1].first);

            std::vector<size_t> left, right;
            for (auto& [v, i] : vals) {
                if (v <= threshold) left.push_back(i);
                else                right.push_back(i);
            }
            if (left.empty() || right.empty()) continue;

            double gain = parent_mse
                - mse_y(y, left)  * static_cast<double>(left.size())
                - mse_y(y, right) * static_cast<double>(right.size());

            if (gain > best_gain) {
                best_gain = gain;
                best = {f, threshold, std::move(left), std::move(right), true};
            }
        }
    }
    return best;
}

// Classification split on criterion
SplitResult bestSplitClassifier(const Tensor& X, const Tensor& y,
                                 const std::vector<size_t>& idx,
                                 size_t nc,
                                 DecisionTreeClassifier::Criterion crit)
{
    const size_t D = X.dim(1);
    double n = static_cast<double>(idx.size());

    std::function<double(const Tensor&, const std::vector<size_t>&)> imp_fn;
    if (crit == DecisionTreeClassifier::Criterion::Gini) {
        imp_fn = [nc](const Tensor& y2, const std::vector<size_t>& i2) {
            return gini(y2, i2, nc);
        };
    } else {
        imp_fn = [nc](const Tensor& y2, const std::vector<size_t>& i2) {
            return entropy(y2, i2, nc);
        };
    }

    double parent_imp = imp_fn(y, idx);
    double best_gain  = -std::numeric_limits<double>::infinity();
    SplitResult best;

    for (size_t f = 0; f < D; ++f) {
        std::vector<std::pair<double, size_t>> vals;
        vals.reserve(idx.size());
        for (size_t i : idx) vals.push_back({X(i, f), i});
        std::sort(vals.begin(), vals.end());

        for (size_t t = 0; t + 1 < vals.size(); ++t) {
            if (vals[t].first == vals[t + 1].first) continue;
            double threshold = 0.5 * (vals[t].first + vals[t + 1].first);

            std::vector<size_t> left, right;
            for (auto& [v, i] : vals) {
                if (v <= threshold) left.push_back(i);
                else                right.push_back(i);
            }
            if (left.empty() || right.empty()) continue;

            double nl = static_cast<double>(left.size());
            double nr = static_cast<double>(right.size());
            double gain = parent_imp - (nl / n) * imp_fn(y, left)
                                     - (nr / n) * imp_fn(y, right);

            if (gain > best_gain) {
                best_gain = gain;
                best = {f, threshold, std::move(left), std::move(right), true};
            }
        }
    }
    return best;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// DecisionTreeClassifier
// ─────────────────────────────────────────────────────────────────────────────

DecisionTreeClassifier::DecisionTreeClassifier(size_t max_depth,
                                               size_t min_samples,
                                               Criterion criterion)
    : m_max_depth(max_depth),
      m_min_samples(min_samples),
      m_criterion(criterion)
{
    if (min_samples < 1)
        throw std::invalid_argument("DecisionTreeClassifier: min_samples must be >= 1");
}

std::shared_ptr<detail::TreeNode>
DecisionTreeClassifier::buildTree(const Tensor& X, const Tensor& y,
                                  const std::vector<size_t>& indices,
                                  size_t depth) const
{
    auto node = std::make_shared<detail::TreeNode>();

    auto makeLeaf = [&]() {
        node->leaf_value = majority_class(y, indices, m_num_classes);
        return node;
    };

    if (all_same_y(y, indices) ||
        indices.size() < m_min_samples ||
        (m_max_depth > 0 && depth >= m_max_depth))
        return makeLeaf();

    auto split = bestSplitClassifier(X, y, indices, m_num_classes, m_criterion);
    if (!split.valid) return makeLeaf();

    node->feature_idx = split.feature;
    node->threshold   = split.threshold;
    node->left  = buildTree(X, y, split.left,  depth + 1);
    node->right = buildTree(X, y, split.right, depth + 1);
    return node;
}

double DecisionTreeClassifier::impurity(const std::vector<size_t>& indices,
                                         const Tensor& y) const
{
    if (m_criterion == Criterion::Gini)
        return gini(y, indices, m_num_classes);
    return entropy(y, indices, m_num_classes);
}

double DecisionTreeClassifier::predictSample(const detail::TreeNode* node,
                                              const Tensor& X, size_t row) const
{
    if (node->is_leaf()) return node->leaf_value;
    if (X(row, node->feature_idx) <= node->threshold)
        return predictSample(node->left.get(),  X, row);
    return predictSample(node->right.get(), X, row);
}

void DecisionTreeClassifier::fit(const Tensor& X, const Tensor& y) {
    if (X.ndim() != 2) throw std::invalid_argument("DecisionTreeClassifier::fit: X must be 2-D");
    if (y.ndim() != 1) throw std::invalid_argument("DecisionTreeClassifier::fit: y must be 1-D");
    if (X.dim(0) != y.dim(0)) throw std::invalid_argument("DecisionTreeClassifier::fit: row-count mismatch");

    m_num_classes = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        size_t c = static_cast<size_t>(y.flat(i)) + 1;
        if (c > m_num_classes) m_num_classes = c;
    }
    if (m_num_classes == 0)
        throw std::invalid_argument("DecisionTreeClassifier::fit: empty y");

    std::vector<size_t> idx(X.dim(0));
    std::iota(idx.begin(), idx.end(), 0);
    m_root   = buildTree(X, y, idx, 0);
    m_fitted = true;
}

Tensor DecisionTreeClassifier::predict(const Tensor& X) const {
    if (!m_fitted)
        throw std::runtime_error("DecisionTreeClassifier::predict: model not fitted");
    if (X.ndim() != 2)
        throw std::invalid_argument("DecisionTreeClassifier::predict: X must be 2-D");
    const size_t N = X.dim(0);
    Tensor result = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i)
        result.flat(i) = predictSample(m_root.get(), X, i);
    return result;
}

size_t DecisionTreeClassifier::max_depth()   const noexcept { return m_max_depth; }
size_t DecisionTreeClassifier::min_samples() const noexcept { return m_min_samples; }
bool   DecisionTreeClassifier::fitted()      const noexcept { return m_fitted; }

// ─────────────────────────────────────────────────────────────────────────────
// DecisionTreeRegressor
// ─────────────────────────────────────────────────────────────────────────────

DecisionTreeRegressor::DecisionTreeRegressor(size_t max_depth, size_t min_samples)
    : m_max_depth(max_depth),
      m_min_samples(min_samples)
{
    if (min_samples < 1)
        throw std::invalid_argument("DecisionTreeRegressor: min_samples must be >= 1");
}

std::shared_ptr<detail::TreeNode>
DecisionTreeRegressor::buildTree(const Tensor& X, const Tensor& y,
                                  const std::vector<size_t>& indices,
                                  size_t depth) const
{
    auto node = std::make_shared<detail::TreeNode>();

    auto makeLeaf = [&]() {
        node->leaf_value = mean_y(y, indices);
        return node;
    };

    if (indices.size() < m_min_samples ||
        (m_max_depth > 0 && depth >= m_max_depth))
        return makeLeaf();

    auto split = bestSplitRegressor(X, y, indices);
    if (!split.valid) return makeLeaf();

    node->feature_idx = split.feature;
    node->threshold   = split.threshold;
    node->left  = buildTree(X, y, split.left,  depth + 1);
    node->right = buildTree(X, y, split.right, depth + 1);
    return node;
}

double DecisionTreeRegressor::predictSample(const detail::TreeNode* node,
                                             const Tensor& X, size_t row) const
{
    if (node->is_leaf()) return node->leaf_value;
    if (X(row, node->feature_idx) <= node->threshold)
        return predictSample(node->left.get(),  X, row);
    return predictSample(node->right.get(), X, row);
}

void DecisionTreeRegressor::fit(const Tensor& X, const Tensor& y) {
    if (X.ndim() != 2) throw std::invalid_argument("DecisionTreeRegressor::fit: X must be 2-D");
    if (y.ndim() != 1) throw std::invalid_argument("DecisionTreeRegressor::fit: y must be 1-D");
    if (X.dim(0) != y.dim(0)) throw std::invalid_argument("DecisionTreeRegressor::fit: row-count mismatch");

    std::vector<size_t> idx(X.dim(0));
    std::iota(idx.begin(), idx.end(), 0);
    m_root   = buildTree(X, y, idx, 0);
    m_fitted = true;
}

Tensor DecisionTreeRegressor::predict(const Tensor& X) const {
    if (!m_fitted)
        throw std::runtime_error("DecisionTreeRegressor::predict: model not fitted");
    if (X.ndim() != 2)
        throw std::invalid_argument("DecisionTreeRegressor::predict: X must be 2-D");
    const size_t N = X.dim(0);
    Tensor result = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i)
        result.flat(i) = predictSample(m_root.get(), X, i);
    return result;
}

size_t DecisionTreeRegressor::max_depth()   const noexcept { return m_max_depth; }
size_t DecisionTreeRegressor::min_samples() const noexcept { return m_min_samples; }
bool   DecisionTreeRegressor::fitted()      const noexcept { return m_fitted; }

// ─────────────────────────────────────────────────────────────────────────────
// RandomForestClassifier
// ─────────────────────────────────────────────────────────────────────────────

RandomForestClassifier::RandomForestClassifier(
    size_t n_estimators, size_t max_depth, size_t min_samples,
    uint64_t seed, DecisionTreeClassifier::Criterion criterion)
    : m_n_estimators(n_estimators),
      m_max_depth(max_depth),
      m_min_samples(min_samples),
      m_seed(seed),
      m_criterion(criterion)
{
    if (n_estimators == 0)
        throw std::invalid_argument("RandomForestClassifier: n_estimators must be > 0");
}

void RandomForestClassifier::fit(const Tensor& X, const Tensor& y) {
    if (X.ndim() != 2) throw std::invalid_argument("RandomForestClassifier::fit: X must be 2-D");
    if (y.ndim() != 1) throw std::invalid_argument("RandomForestClassifier::fit: y must be 1-D");
    if (X.dim(0) != y.dim(0)) throw std::invalid_argument("RandomForestClassifier::fit: row-count mismatch");

    m_num_classes = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        size_t c = static_cast<size_t>(y.flat(i)) + 1;
        if (c > m_num_classes) m_num_classes = c;
    }

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    std::mt19937_64 rng(m_seed);
    std::uniform_int_distribution<size_t> row_dist(0, N - 1);

    m_trees.clear();
    m_trees.reserve(m_n_estimators);

    for (size_t t = 0; t < m_n_estimators; ++t) {
        // Bootstrap sample
        Tensor X_boot = Tensor::zeros({N, D});
        Tensor y_boot = Tensor::zeros({N});
        for (size_t i = 0; i < N; ++i) {
            size_t idx = row_dist(rng);
            y_boot.flat(i) = y.flat(idx);
            for (size_t d = 0; d < D; ++d)
                X_boot(i, d) = X(idx, d);
        }
        DecisionTreeClassifier tree(m_max_depth, m_min_samples, m_criterion);
        tree.fit(X_boot, y_boot);
        m_trees.push_back(std::move(tree));
    }
    m_fitted = true;
}

Tensor RandomForestClassifier::predict(const Tensor& X) const {
    if (!m_fitted)
        throw std::runtime_error("RandomForestClassifier::predict: model not fitted");
    const size_t N = X.dim(0);
    Tensor result = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        std::vector<size_t> votes(m_num_classes, 0);
        for (const auto& tree : m_trees)
            ++votes[static_cast<size_t>(tree.predict(X).flat(i))];
        result.flat(i) = static_cast<double>(
            std::max_element(votes.begin(), votes.end()) - votes.begin());
    }
    return result;
}

size_t RandomForestClassifier::n_estimators() const noexcept { return m_n_estimators; }
bool   RandomForestClassifier::fitted()       const noexcept { return m_fitted; }

// ─────────────────────────────────────────────────────────────────────────────
// RandomForestRegressor
// ─────────────────────────────────────────────────────────────────────────────

RandomForestRegressor::RandomForestRegressor(
    size_t n_estimators, size_t max_depth, size_t min_samples, uint64_t seed)
    : m_n_estimators(n_estimators),
      m_max_depth(max_depth),
      m_min_samples(min_samples),
      m_seed(seed)
{
    if (n_estimators == 0)
        throw std::invalid_argument("RandomForestRegressor: n_estimators must be > 0");
}

void RandomForestRegressor::fit(const Tensor& X, const Tensor& y) {
    if (X.ndim() != 2) throw std::invalid_argument("RandomForestRegressor::fit: X must be 2-D");
    if (y.ndim() != 1) throw std::invalid_argument("RandomForestRegressor::fit: y must be 1-D");
    if (X.dim(0) != y.dim(0)) throw std::invalid_argument("RandomForestRegressor::fit: row-count mismatch");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);
    std::mt19937_64 rng(m_seed);
    std::uniform_int_distribution<size_t> row_dist(0, N - 1);

    m_trees.clear();
    m_trees.reserve(m_n_estimators);

    for (size_t t = 0; t < m_n_estimators; ++t) {
        Tensor X_boot = Tensor::zeros({N, D});
        Tensor y_boot = Tensor::zeros({N});
        for (size_t i = 0; i < N; ++i) {
            size_t idx = row_dist(rng);
            y_boot.flat(i) = y.flat(idx);
            for (size_t d = 0; d < D; ++d)
                X_boot(i, d) = X(idx, d);
        }
        DecisionTreeRegressor tree(m_max_depth, m_min_samples);
        tree.fit(X_boot, y_boot);
        m_trees.push_back(std::move(tree));
    }
    m_fitted = true;
}

Tensor RandomForestRegressor::predict(const Tensor& X) const {
    if (!m_fitted)
        throw std::runtime_error("RandomForestRegressor::predict: model not fitted");
    const size_t N = X.dim(0);
    Tensor result = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (const auto& tree : m_trees)
            sum += tree.predict(X).flat(i);
        result.flat(i) = sum / static_cast<double>(m_trees.size());
    }
    return result;
}

size_t RandomForestRegressor::n_estimators() const noexcept { return m_n_estimators; }
bool   RandomForestRegressor::fitted()       const noexcept { return m_fitted; }

} // namespace SharedMath::ML
