#pragma once

// SharedMath::ML — Classification Metrics
//
// All functions accept 1-D integer label tensors (class labels stored as doubles).
//
// accuracy(y_pred, y_true)
// precision(y_pred, y_true, positive_class)
// recall(y_pred, y_true, positive_class)
// f1_score(y_pred, y_true, positive_class)
// confusion_matrix(y_pred, y_true, num_classes)  → [C, C] Tensor

#include "LinearAlgebra/Tensor.h"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

namespace detail {

inline void checkLabels(const Tensor& pred, const Tensor& true_,
                        const char* fn)
{
    if (pred.ndim() != 1 || true_.ndim() != 1)
        throw std::invalid_argument(std::string(fn) + ": labels must be 1-D");
    if (pred.size() != true_.size() || pred.size() == 0)
        throw std::invalid_argument(
            std::string(fn) + ": label vectors must be equal-length and non-empty");
}

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
// accuracy — fraction of correctly predicted labels
// ─────────────────────────────────────────────────────────────────────────────
inline double accuracy(const Tensor& y_pred, const Tensor& y_true) {
    detail::checkLabels(y_pred, y_true, "accuracy");
    const size_t N = y_pred.size();
    size_t correct = 0;
    for (size_t i = 0; i < N; ++i)
        if (static_cast<int>(y_pred.flat(i)) == static_cast<int>(y_true.flat(i)))
            ++correct;
    return static_cast<double>(correct) / static_cast<double>(N);
}

// ─────────────────────────────────────────────────────────────────────────────
// precision — TP / (TP + FP) for the given positive class
// ─────────────────────────────────────────────────────────────────────────────
inline double precision(const Tensor& y_pred, const Tensor& y_true,
                        int positive_class = 1)
{
    detail::checkLabels(y_pred, y_true, "precision");
    const size_t N = y_pred.size();
    size_t tp = 0, fp = 0;
    for (size_t i = 0; i < N; ++i) {
        bool pred_pos = (static_cast<int>(y_pred.flat(i)) == positive_class);
        bool true_pos = (static_cast<int>(y_true.flat(i)) == positive_class);
        if (pred_pos && true_pos)  ++tp;
        if (pred_pos && !true_pos) ++fp;
    }
    if (tp + fp == 0) return 0.0;
    return static_cast<double>(tp) / static_cast<double>(tp + fp);
}

// ─────────────────────────────────────────────────────────────────────────────
// recall — TP / (TP + FN) for the given positive class
// ─────────────────────────────────────────────────────────────────────────────
inline double recall(const Tensor& y_pred, const Tensor& y_true,
                     int positive_class = 1)
{
    detail::checkLabels(y_pred, y_true, "recall");
    const size_t N = y_pred.size();
    size_t tp = 0, fn = 0;
    for (size_t i = 0; i < N; ++i) {
        bool pred_pos = (static_cast<int>(y_pred.flat(i)) == positive_class);
        bool true_pos = (static_cast<int>(y_true.flat(i)) == positive_class);
        if (pred_pos  && true_pos)  ++tp;
        if (!pred_pos && true_pos)  ++fn;
    }
    if (tp + fn == 0) return 0.0;
    return static_cast<double>(tp) / static_cast<double>(tp + fn);
}

// ─────────────────────────────────────────────────────────────────────────────
// f1_score — harmonic mean of precision and recall
// ─────────────────────────────────────────────────────────────────────────────
inline double f1_score(const Tensor& y_pred, const Tensor& y_true,
                       int positive_class = 1)
{
    double p = precision(y_pred, y_true, positive_class);
    double r = recall   (y_pred, y_true, positive_class);
    if (p + r == 0.0) return 0.0;
    return 2.0 * p * r / (p + r);
}

// ─────────────────────────────────────────────────────────────────────────────
// confusion_matrix — C×C count matrix
//
// confusion_matrix[i][j] = number of samples with true label i predicted as j.
// num_classes: if 0, inferred as max(label)+1.
// ─────────────────────────────────────────────────────────────────────────────
inline Tensor confusion_matrix(const Tensor& y_pred, const Tensor& y_true,
                               size_t num_classes = 0)
{
    detail::checkLabels(y_pred, y_true, "confusion_matrix");
    const size_t N = y_pred.size();

    if (num_classes == 0) {
        for (size_t i = 0; i < N; ++i) {
            size_t c = static_cast<size_t>(y_pred.flat(i)) + 1;
            if (c > num_classes) num_classes = c;
            c = static_cast<size_t>(y_true.flat(i)) + 1;
            if (c > num_classes) num_classes = c;
        }
    }
    if (num_classes == 0)
        throw std::invalid_argument("confusion_matrix: cannot infer num_classes");

    Tensor cm = Tensor::zeros({num_classes, num_classes});
    for (size_t i = 0; i < N; ++i) {
        size_t t = static_cast<size_t>(y_true.flat(i));
        size_t p = static_cast<size_t>(y_pred.flat(i));
        if (t >= num_classes || p >= num_classes)
            throw std::out_of_range("confusion_matrix: label out of [0, num_classes)");
        cm(t, p) += 1.0;
    }
    return cm;
}

// ─────────────────────────────────────────────────────────────────────────────
// Regression metrics
// ─────────────────────────────────────────────────────────────────────────────

inline void checkRegression(const Tensor& pred, const Tensor& true_,
                            const char* fn)
{
    if (pred.ndim() != 1 || true_.ndim() != 1)
        throw std::invalid_argument(std::string(fn) + ": tensors must be 1-D");
    if (pred.size() != true_.size() || pred.size() == 0)
        throw std::invalid_argument(
            std::string(fn) + ": tensors must be equal-length and non-empty");
}

inline double mean_squared_error(const Tensor& y_pred, const Tensor& y_true) {
    checkRegression(y_pred, y_true, "mean_squared_error");
    const size_t N = y_pred.size();
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double d = y_pred.flat(i) - y_true.flat(i);
        sum += d * d;
    }
    return sum / static_cast<double>(N);
}

inline double mean_absolute_error(const Tensor& y_pred, const Tensor& y_true) {
    checkRegression(y_pred, y_true, "mean_absolute_error");
    const size_t N = y_pred.size();
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i)
        sum += std::abs(y_pred.flat(i) - y_true.flat(i));
    return sum / static_cast<double>(N);
}

inline double r2_score(const Tensor& y_pred, const Tensor& y_true) {
    checkRegression(y_pred, y_true, "r2_score");
    const size_t N = y_pred.size();
    double mean = 0.0;
    for (size_t i = 0; i < N; ++i) mean += y_true.flat(i);
    mean /= static_cast<double>(N);

    double ss_res = 0.0, ss_tot = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double r = y_pred.flat(i) - y_true.flat(i);
        double t = y_true.flat(i) - mean;
        ss_res += r * r;
        ss_tot += t * t;
    }
    if (ss_tot == 0.0) return 1.0;
    return 1.0 - ss_res / ss_tot;
}

// ─────────────────────────────────────────────────────────────────────────────
// Probabilistic metrics
// ─────────────────────────────────────────────────────────────────────────────

// Binary log loss.  y_pred: probabilities in (0,1), y_true: {0,1}.
inline double log_loss(const Tensor& y_pred, const Tensor& y_true,
                       double eps = 1e-15)
{
    checkRegression(y_pred, y_true, "log_loss");
    const size_t N = y_pred.size();
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double p  = std::max(eps, std::min(1.0 - eps, y_pred.flat(i)));
        double yi = y_true.flat(i);
        sum -= yi * std::log(p) + (1.0 - yi) * std::log(1.0 - p);
    }
    return sum / static_cast<double>(N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-class F1 (macro / micro)
// ─────────────────────────────────────────────────────────────────────────────

inline double macro_f1_score(const Tensor& y_pred, const Tensor& y_true,
                             size_t num_classes = 0)
{
    detail::checkLabels(y_pred, y_true, "macro_f1_score");
    const size_t N = y_pred.size();

    if (num_classes == 0) {
        for (size_t i = 0; i < N; ++i) {
            size_t c = static_cast<size_t>(y_pred.flat(i)) + 1;
            if (c > num_classes) num_classes = c;
            c = static_cast<size_t>(y_true.flat(i)) + 1;
            if (c > num_classes) num_classes = c;
        }
    }

    double f1_sum = 0.0;
    for (size_t c = 0; c < num_classes; ++c)
        f1_sum += f1_score(y_pred, y_true, static_cast<int>(c));
    return f1_sum / static_cast<double>(num_classes);
}

inline double micro_f1_score(const Tensor& y_pred, const Tensor& y_true,
                              size_t num_classes = 0)
{
    detail::checkLabels(y_pred, y_true, "micro_f1_score");
    const size_t N = y_pred.size();

    if (num_classes == 0) {
        for (size_t i = 0; i < N; ++i) {
            size_t c = static_cast<size_t>(y_pred.flat(i)) + 1;
            if (c > num_classes) num_classes = c;
            c = static_cast<size_t>(y_true.flat(i)) + 1;
            if (c > num_classes) num_classes = c;
        }
    }

    size_t tp_total = 0, fp_total = 0, fn_total = 0;
    for (size_t c = 0; c < num_classes; ++c) {
        size_t tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < N; ++i) {
            bool pp = (static_cast<size_t>(y_pred.flat(i)) == c);
            bool pt = (static_cast<size_t>(y_true.flat(i)) == c);
            if (pp && pt) ++tp;
            else if (pp)  ++fp;
            else if (pt)  ++fn;
        }
        tp_total += tp;
        fp_total += fp;
        fn_total += fn;
    }
    double denom = static_cast<double>(2 * tp_total + fp_total + fn_total);
    if (denom == 0.0) return 0.0;
    return 2.0 * static_cast<double>(tp_total) / denom;
}

// ─────────────────────────────────────────────────────────────────────────────
// silhouette_score — clustering quality metric in [-1, 1]
//
// X       : [N, D] data matrix
// labels  : [N] cluster assignment for each sample
// ─────────────────────────────────────────────────────────────────────────────
inline double silhouette_score(const Tensor& X, const Tensor& labels) {
    if (X.ndim() != 2)
        throw std::invalid_argument("silhouette_score: X must be 2-D");
    if (labels.ndim() != 1 || labels.size() != X.dim(0))
        throw std::invalid_argument("silhouette_score: labels size must match X rows");

    const size_t N = X.dim(0);
    const size_t D = X.dim(1);

    // Squared Euclidean distance helper
    auto dist2 = [&](size_t i, size_t j) {
        double s = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = X(i, d) - X(j, d);
            s += diff * diff;
        }
        return s;
    };

    // Identify clusters
    size_t num_clusters = 0;
    for (size_t i = 0; i < N; ++i) {
        size_t c = static_cast<size_t>(labels.flat(i)) + 1;
        if (c > num_clusters) num_clusters = c;
    }
    if (num_clusters < 2)
        throw std::invalid_argument("silhouette_score: at least 2 clusters required");

    double total_s = 0.0;
    for (size_t i = 0; i < N; ++i) {
        size_t ci = static_cast<size_t>(labels.flat(i));

        // a(i) — mean intra-cluster distance
        double a = 0.0;
        size_t a_count = 0;
        for (size_t j = 0; j < N; ++j) {
            if (j != i && static_cast<size_t>(labels.flat(j)) == ci) {
                a += std::sqrt(dist2(i, j));
                ++a_count;
            }
        }
        a = (a_count > 0) ? a / static_cast<double>(a_count) : 0.0;

        // b(i) — minimum mean distance to any other cluster
        double b = std::numeric_limits<double>::infinity();
        for (size_t c = 0; c < num_clusters; ++c) {
            if (c == ci) continue;
            double bc = 0.0;
            size_t bc_count = 0;
            for (size_t j = 0; j < N; ++j) {
                if (static_cast<size_t>(labels.flat(j)) == c) {
                    bc += std::sqrt(dist2(i, j));
                    ++bc_count;
                }
            }
            if (bc_count > 0) {
                bc /= static_cast<double>(bc_count);
                if (bc < b) b = bc;
            }
        }
        if (b == std::numeric_limits<double>::infinity()) b = 0.0;

        double denom = std::max(a, b);
        total_s += (denom > 0.0) ? (b - a) / denom : 0.0;
    }
    return total_s / static_cast<double>(N);
}

} // namespace SharedMath::ML
