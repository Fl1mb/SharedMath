#pragma once

#include "DynamicMatrix.h"
#include "DynamicVector.h"
#include <sharedmath_linearalgebra_export.h>

#include <vector>
#include <stdexcept>
#include <string>

namespace SharedMath::LinearAlgebra {

/// Principal Component Analysis (PCA).
///
/// Fits on a data matrix X of shape (n_samples × n_features), reduces to
/// n_components dimensions.  Backed by thin SVD of the mean-centred data.
///
/// Example:
///   PCA pca(50);
///   pca.fit(X_train);
///   DynamicMatrix Z  = pca.transform(X_test);
///   DynamicMatrix Xr = pca.inverse_transform(Z);
///
///   auto ratios = pca.explained_variance_ratio();   // per-component fraction
///   std::cout << "Total variance explained: " << pca.total_variance_explained();
///
class SHAREDMATH_LINEARALGEBRA_EXPORT PCA {
public:
    // n_components : number of principal components to keep
    //                (0 → keep all, capped at min(n_samples-1, n_features))
    // whiten       : divide projected coordinates by sqrt(variance) so that
    //                each component has unit variance (ZCA-style whitening)
    // use_randomized: use randomized SVD (faster for large matrices)
    explicit PCA(size_t n_components = 0,
                 bool   whiten        = false,
                 bool   use_randomized = false);

    /// ── Fit ───────────────────────────────────────────────────────────────

    /// Fit PCA on X.  X must be (n_samples × n_features) with n_samples > 1.
    PCA& fit(const DynamicMatrix& X);

    /// Fit and return projections in one call.
    DynamicMatrix fit_transform(const DynamicMatrix& X);

    /// ── Transform ─────────────────────────────────────────────────────────

    /// Project X to PC space: result is (n_samples × n_components).
    /// Requires fit() to have been called.
    DynamicMatrix transform(const DynamicMatrix& X) const;

    /// Reconstruct approximate X from PC coordinates Z (n_samples × n_components).
    DynamicMatrix inverse_transform(const DynamicMatrix& Z) const;

    /// ── Results ───────────────────────────────────────────────────────────

    /// Principal component directions, shape (n_components × n_features).
    /// Rows are sorted by descending explained variance.
    const DynamicMatrix& components() const { requireFit(); return components_; }

    /// Per-component variance (eigenvalues of the covariance matrix).
    const std::vector<double>& explained_variance() const {
        requireFit(); return explained_variance_;
    }

    /// Fraction of total variance explained by each component.
    const std::vector<double>& explained_variance_ratio() const {
        requireFit(); return explained_variance_ratio_;
    }

    /// Sum of explained_variance_ratio_ for kept components.
    double total_variance_explained() const;

    /// Per-feature mean estimated from the training data.
    const DynamicVector& mean() const { requireFit(); return mean_; }

    /// Singular values of the centred data matrix.
    const std::vector<double>& singular_values() const {
        requireFit(); return singular_values_;
    }

    /// Estimated noise variance (average of discarded eigenvalues).
    double noise_variance() const { requireFit(); return noise_variance_; }

    /// Whitening matrix W s.t. Z_white = Z * W  (or use whiten=true).
    DynamicMatrix whitening_matrix() const;

    /// Number of components actually kept.
    size_t n_components() const { return n_components_out_; }

    bool is_fitted() const noexcept { return fitted_; }

private:
    size_t n_components_req_;      // requested (0 = auto)
    size_t n_components_out_ = 0;  // actual after fit
    bool   whiten_;
    bool   use_randomized_;
    bool   fitted_ = false;

    DynamicMatrix      components_;              // (k × p)
    DynamicVector      mean_;                    // (p,)
    std::vector<double> explained_variance_;     // (k,)
    std::vector<double> explained_variance_ratio_; // (k,)
    std::vector<double> singular_values_;        // (k,)
    double             noise_variance_ = 0.0;

    void requireFit() const {
        if (!fitted_)
            throw std::runtime_error("PCA: call fit() before accessing results");
    }
};

} // namespace SharedMath::LinearAlgebra
