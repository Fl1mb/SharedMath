#include "LinearAlgebra/PCA.h"
#include "LinearAlgebra/MatrixFunctions.h"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace SharedMath::LinearAlgebra {

PCA::PCA(size_t n_components, bool whiten, bool use_randomized)
    : n_components_req_(n_components)
    , whiten_(whiten)
    , use_randomized_(use_randomized)
{}

// ── fit ───────────────────────────────────────────────────────────────────────

PCA& PCA::fit(const DynamicMatrix& X) {
    size_t n = X.rows();  // samples
    size_t p = X.cols();  // features
    if (n < 2)
        throw std::invalid_argument("PCA::fit: need at least 2 samples");

    // 1. Compute per-feature mean and centre X
    mean_ = DynamicVector(p, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j)
            mean_[j] += X(i, j);
    for (size_t j = 0; j < p; ++j) mean_[j] /= static_cast<double>(n);

    DynamicMatrix Xc(n, p);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j)
            Xc(i, j) = X(i, j) - mean_[j];

    // 2. Determine number of components
    size_t k_max = std::min(n - 1, p);
    n_components_out_ = (n_components_req_ == 0 || n_components_req_ > k_max)
                        ? k_max : n_components_req_;

    // 3. SVD of centred data (thin)
    //    Xc = U * S * Vt   (U: n×k, S: k, Vt: k×p)
    auto [U, S, Vt] = (use_randomized_ && n_components_out_ < k_max / 2)
        ? rsvd(Xc, n_components_out_, 10, 2)
        : svd(Xc);

    // 4. Store results
    singular_values_.assign(S.begin(),
                             S.begin() + std::min(S.size(), n_components_out_));

    // Explained variance = S_i^2 / (n-1)
    double inv_nm1 = 1.0 / static_cast<double>(n - 1);
    double total_var = 0.0;
    for (double s : S) total_var += s * s * inv_nm1;

    explained_variance_.resize(n_components_out_);
    explained_variance_ratio_.resize(n_components_out_);
    for (size_t i = 0; i < n_components_out_; ++i) {
        double var = (i < S.size()) ? S[i] * S[i] * inv_nm1 : 0.0;
        explained_variance_[i]       = var;
        explained_variance_ratio_[i] = (total_var > 0) ? var / total_var : 0.0;
    }

    // Noise variance: average of discarded eigenvalues
    double discarded = 0.0;
    size_t n_discarded = 0;
    for (size_t i = n_components_out_; i < S.size(); ++i) {
        discarded += S[i] * S[i] * inv_nm1;
        ++n_discarded;
    }
    noise_variance_ = (n_discarded > 0) ? discarded / n_discarded : 0.0;

    // Components: first n_components_out_ rows of Vt
    components_ = DynamicMatrix(n_components_out_, p);
    for (size_t i = 0; i < n_components_out_; ++i)
        for (size_t j = 0; j < p; ++j)
            components_(i, j) = Vt(i, j);

    fitted_ = true;
    return *this;
}

// ── transform ─────────────────────────────────────────────────────────────────

DynamicMatrix PCA::transform(const DynamicMatrix& X) const {
    requireFit();
    size_t n = X.rows(), p = X.cols();
    if (p != components_.cols())
        throw std::invalid_argument(
            "PCA::transform: feature dimension mismatch (" +
            std::to_string(p) + " vs " + std::to_string(components_.cols()) + ")");

    // Z = (X - mean) * components_^T   → (n × k)
    DynamicMatrix Z(n, n_components_out_);
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n_components_out_; ++k) {
            double s = 0.0;
            for (size_t j = 0; j < p; ++j)
                s += (X(i, j) - mean_[j]) * components_(k, j);
            // Optionally whiten
            if (whiten_ && explained_variance_[k] > 0)
                s /= std::sqrt(explained_variance_[k]);
            Z(i, k) = s;
        }
    return Z;
}

// ── fit_transform ─────────────────────────────────────────────────────────────

DynamicMatrix PCA::fit_transform(const DynamicMatrix& X) {
    fit(X);
    return transform(X);
}

// ── inverse_transform ─────────────────────────────────────────────────────────

DynamicMatrix PCA::inverse_transform(const DynamicMatrix& Z) const {
    requireFit();
    size_t n = Z.rows();
    if (Z.cols() != n_components_out_)
        throw std::invalid_argument("PCA::inverse_transform: component count mismatch");
    size_t p = components_.cols();

    // X_rec = Z * components_ + mean
    DynamicMatrix X(n, p);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j) {
            double s = mean_[j];
            for (size_t k = 0; k < n_components_out_; ++k) {
                double z = Z(i, k);
                if (whiten_ && explained_variance_[k] > 0)
                    z *= std::sqrt(explained_variance_[k]);  // undo whitening
                s += z * components_(k, j);
            }
            X(i, j) = s;
        }
    return X;
}

// ── total_variance_explained ──────────────────────────────────────────────────

double PCA::total_variance_explained() const {
    requireFit();
    double s = 0.0;
    for (double r : explained_variance_ratio_) s += r;
    return s;
}

// ── whitening_matrix ──────────────────────────────────────────────────────────
// W such that Z_white = (X - mean) * W
// W = V * D^{-1/2}  where V = components_^T and D = diag(explained_variance_)

DynamicMatrix PCA::whitening_matrix() const {
    requireFit();
    size_t p = components_.cols(), k = n_components_out_;
    DynamicMatrix W(p, k);
    for (size_t j = 0; j < p; ++j)
        for (size_t i = 0; i < k; ++i) {
            double std_i = (explained_variance_[i] > 0)
                           ? std::sqrt(explained_variance_[i]) : 1.0;
            W(j, i) = components_(i, j) / std_i;
        }
    return W;
}

} // namespace SharedMath::LinearAlgebra
