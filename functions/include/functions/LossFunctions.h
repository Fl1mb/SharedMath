#pragma once

// SharedMath::Functions — Loss Functions for Machine Learning
//
// All functions operate on std::vector<T> (T = float or double).
// Scalar variants are also provided for element-wise use.
//
// Conventions:
//   y_pred — model output (raw scores or probabilities depending on loss)
//   y_true — ground-truth labels or one-hot vectors
//   All losses return the mean over the batch (reduction = "mean") unless noted.

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>

namespace SharedMath::Functions {

namespace detail {

template<typename T>
void checkSameSize(const std::vector<T>& a, const std::vector<T>& b,
                   const char* name) {
    if (a.size() != b.size() || a.empty())
        throw std::invalid_argument(
            std::string(name) + ": vectors must be non-empty and equal in length");
}

} // namespace detail


// ═════════════════════════════════════════════════════════════════════════════
// REGRESSION LOSSES
// ═════════════════════════════════════════════════════════════════════════════

// ── Mean Squared Error  MSE = Σ(y_pred − y_true)² / N ───────────────────────
template<typename T>
inline T mse(const std::vector<T>& y_pred, const std::vector<T>& y_true) {
    detail::checkSameSize(y_pred, y_true, "mse");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T d = y_pred[i] - y_true[i];
        sum += d * d;
    }
    return sum / static_cast<T>(y_pred.size());
}

// ── Mean Absolute Error  MAE = Σ|y_pred − y_true| / N ───────────────────────
template<typename T>
inline T mae(const std::vector<T>& y_pred, const std::vector<T>& y_true) {
    detail::checkSameSize(y_pred, y_true, "mae");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i)
        sum += std::abs(y_pred[i] - y_true[i]);
    return sum / static_cast<T>(y_pred.size());
}

// ── Root Mean Squared Error
template<typename T>
inline T rmse(const std::vector<T>& y_pred, const std::vector<T>& y_true) {
    return std::sqrt(mse(y_pred, y_true));
}

// ── Huber loss (smooth L1) ────────────────────────────────────────────────────
// L_δ(r) = r²/2                if |r| ≤ δ
//         = δ(|r| − δ/2)       if |r| > δ
// Behaves like MSE near 0 and like MAE far from 0.
template<typename T>
inline T huber(const std::vector<T>& y_pred, const std::vector<T>& y_true,
               T delta = T(1)) {
    detail::checkSameSize(y_pred, y_true, "huber");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T r = std::abs(y_pred[i] - y_true[i]);
        sum += (r <= delta) ? T(0.5) * r * r : delta * (r - T(0.5) * delta);
    }
    return sum / static_cast<T>(y_pred.size());
}

// ── Log-Cosh loss ─────────────────────────────────────────────────────────────
// L(r) = log(cosh(r)) ≈ r²/2 for small r, ≈ |r| − log2 for large r.
// Twice-differentiable; gentler than Huber on outliers.
template<typename T>
inline T logCosh(const std::vector<T>& y_pred, const std::vector<T>& y_true) {
    detail::checkSameSize(y_pred, y_true, "logCosh");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T r = y_pred[i] - y_true[i];
        // Numerically stable: log(cosh(r)) = |r| + log(1 + e^{-2|r|}) − log2
        T ar = std::abs(r);
        sum += ar + std::log(T(1) + std::exp(-T(2) * ar)) - static_cast<T>(0.6931471805599453);
    }
    return sum / static_cast<T>(y_pred.size());
}

// ── Mean Absolute Percentage Error  MAPE = 100/N · Σ|r_i / y_true_i| ─────────
template<typename T>
inline T mape(const std::vector<T>& y_pred, const std::vector<T>& y_true,
              T eps = T(1e-8)) {
    detail::checkSameSize(y_pred, y_true, "mape");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i)
        sum += std::abs((y_pred[i] - y_true[i]) / (std::abs(y_true[i]) + eps));
    return T(100) * sum / static_cast<T>(y_pred.size());
}


// ═════════════════════════════════════════════════════════════════════════════
// CLASSIFICATION LOSSES
// ═════════════════════════════════════════════════════════════════════════════

// ── Binary Cross-Entropy ──────────────────────────────────────────────────────
// L = −[y·log(p) + (1−y)·log(1−p)]
// y_pred: probabilities in (0,1);  y_true: binary labels in {0,1}.
template<typename T>
inline T binaryCrossEntropy(const std::vector<T>& y_pred,
                             const std::vector<T>& y_true,
                             T eps = T(1e-12)) {
    detail::checkSameSize(y_pred, y_true, "binaryCrossEntropy");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T p = std::max(eps, std::min(T(1) - eps, y_pred[i]));
        sum += -y_true[i] * std::log(p) - (T(1) - y_true[i]) * std::log(T(1) - p);
    }
    return sum / static_cast<T>(y_pred.size());
}

// ── Categorical Cross-Entropy ─────────────────────────────────────────────────
// L = −Σ_i y_i · log(p_i)
// y_pred: probability distribution (softmax output);  y_true: one-hot vector.
template<typename T>
inline T crossEntropy(const std::vector<T>& y_pred,
                       const std::vector<T>& y_true,
                       T eps = T(1e-12)) {
    detail::checkSameSize(y_pred, y_true, "crossEntropy");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i)
        sum += -y_true[i] * std::log(std::max(eps, y_pred[i]));
    return sum;
}

// ── Kullback–Leibler Divergence  D_KL(P ‖ Q) = Σ P_i log(P_i / Q_i) ─────────
// y_pred: Q (approximation);  y_true: P (reference).
template<typename T>
inline T klDivergence(const std::vector<T>& y_pred,
                       const std::vector<T>& y_true,
                       T eps = T(1e-12)) {
    detail::checkSameSize(y_pred, y_true, "klDivergence");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T p = y_true[i];
        if (p > T(0))
            sum += p * std::log(p / std::max(eps, y_pred[i]));
    }
    return sum;
}

// ── Hinge Loss  max(0, 1 − y·f(x)) — for binary SVM-style classification ─────
// y_true ∈ {−1, +1},  y_pred is the raw margin score.
template<typename T>
inline T hingeLoss(const std::vector<T>& y_pred, const std::vector<T>& y_true) {
    detail::checkSameSize(y_pred, y_true, "hingeLoss");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i)
        sum += std::max(T(0), T(1) - y_true[i] * y_pred[i]);
    return sum / static_cast<T>(y_pred.size());
}

// ── Squared Hinge  max(0, 1 − y·f(x))² ──────────────────────────────────────
template<typename T>
inline T squaredHinge(const std::vector<T>& y_pred, const std::vector<T>& y_true) {
    detail::checkSameSize(y_pred, y_true, "squaredHinge");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T h = std::max(T(0), T(1) - y_true[i] * y_pred[i]);
        sum += h * h;
    }
    return sum / static_cast<T>(y_pred.size());
}

// ── Focal Loss — down-weights easy examples (Lin et al., RetinaNet 2017) ──────
// FL(p) = −α(1−p)^γ log(p),  y_pred: probability,  y_true: binary label.
template<typename T>
inline T focalLoss(const std::vector<T>& y_pred,
                    const std::vector<T>& y_true,
                    T gamma = T(2),
                    T alpha = T(0.25),
                    T eps   = T(1e-12)) {
    detail::checkSameSize(y_pred, y_true, "focalLoss");
    T sum = T(0);
    for (size_t i = 0; i < y_pred.size(); ++i) {
        T p   = std::max(eps, std::min(T(1) - eps, y_pred[i]));
        T y   = y_true[i];
        T pt  = y * p + (T(1) - y) * (T(1) - p);
        T at  = y * alpha + (T(1) - y) * (T(1) - alpha);
        sum  += -at * std::pow(T(1) - pt, gamma) * std::log(pt);
    }
    return sum / static_cast<T>(y_pred.size());
}


// ═════════════════════════════════════════════════════════════════════════════
// REGULARISATION TERMS  (to be added to a training loss)
// ═════════════════════════════════════════════════════════════════════════════

// L1 regularisation: λ Σ |w_i|
template<typename T>
inline T l1Regularisation(const std::vector<T>& weights, T lambda = T(1e-4)) {
    T sum = T(0);
    for (const T& w : weights) sum += std::abs(w);
    return lambda * sum;
}

// L2 regularisation (weight decay): λ/2 Σ w_i²
template<typename T>
inline T l2Regularisation(const std::vector<T>& weights, T lambda = T(1e-4)) {
    T sum = T(0);
    for (const T& w : weights) sum += w * w;
    return lambda * T(0.5) * sum;
}

// Elastic-net: l1_ratio·L1 + (1−l1_ratio)·L2
template<typename T>
inline T elasticNet(const std::vector<T>& weights,
                     T lambda    = T(1e-4),
                     T l1_ratio  = T(0.5)) {
    return l1_ratio       * l1Regularisation(weights, lambda)
         + (T(1)-l1_ratio) * l2Regularisation(weights, lambda);
}


// ═════════════════════════════════════════════════════════════════════════════
// METRICS  (not differentiable — for evaluation, not training)
// ═════════════════════════════════════════════════════════════════════════════

// R² coefficient of determination
template<typename T>
inline T r2Score(const std::vector<T>& y_pred, const std::vector<T>& y_true) {
    detail::checkSameSize(y_pred, y_true, "r2Score");
    T mean = T(0);
    for (const T& v : y_true) mean += v;
    mean /= static_cast<T>(y_true.size());

    T ss_res = T(0), ss_tot = T(0);
    for (size_t i = 0; i < y_true.size(); ++i) {
        T dr = y_pred[i] - y_true[i];
        T dt = y_true[i] - mean;
        ss_res += dr * dr;
        ss_tot += dt * dt;
    }
    return (ss_tot < T(1e-300)) ? T(0) : T(1) - ss_res / ss_tot;
}

// Binary accuracy
template<typename T>
inline T binaryAccuracy(const std::vector<T>& y_pred,
                         const std::vector<T>& y_true,
                         T threshold = T(0.5)) {
    detail::checkSameSize(y_pred, y_true, "binaryAccuracy");
    size_t correct = 0;
    for (size_t i = 0; i < y_pred.size(); ++i)
        if ((y_pred[i] >= threshold ? T(1) : T(0)) == y_true[i]) ++correct;
    return static_cast<T>(correct) / static_cast<T>(y_pred.size());
}

} // namespace SharedMath::Functions
