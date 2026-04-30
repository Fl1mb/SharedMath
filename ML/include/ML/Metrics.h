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

} // namespace SharedMath::ML
