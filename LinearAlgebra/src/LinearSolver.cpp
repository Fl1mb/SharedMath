#include "LinearAlgebra/LinearSolver.h"
#include "LinearAlgebra/MatrixFunctions.h"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace SharedMath::LinearAlgebra {

// ── Named constructors ────────────────────────────────────────────────────────

LinearSolver LinearSolver::lu(const AbstractMatrix& A) {
    return LinearSolver(A, Method::LU);
}
LinearSolver LinearSolver::qr(const AbstractMatrix& A) {
    return LinearSolver(A, Method::QR);
}
LinearSolver LinearSolver::cholesky(const AbstractMatrix& A) {
    return LinearSolver(A, Method::Cholesky);
}

// ── Constructor / auto-detection ──────────────────────────────────────────────

LinearSolver::LinearSolver(const AbstractMatrix& A, Method method)
    : rows_(A.rows()), cols_(A.cols())
{
    if (method == Method::Auto) {
        // Choose: SPD → Cholesky, square → LU, rectangular → QR
        if (A.rows() == A.cols() && isSymmetric(A) && isPositiveDefinite(A))
            method = Method::Cholesky;
        else if (A.rows() == A.cols())
            method = Method::LU;
        else
            method = Method::QR;
    }
    method_ = method;

    switch (method_) {
    case Method::LU:       factorize_lu(A);       break;
    case Method::QR:       factorize_qr(A);       break;
    case Method::Cholesky: factorize_cholesky(A); break;
    default: break;
    }
}

// ── LU factorization (in-place, partial pivoting) ─────────────────────────────

void LinearSolver::factorize_lu(const AbstractMatrix& src) {
    if (src.rows() != src.cols())
        throw std::invalid_argument("LinearSolver (LU): matrix must be square");
    size_t n = src.rows();
    LU_ = DynamicMatrix(src);    // deep copy
    piv_.resize(n);
    std::iota(piv_.begin(), piv_.end(), 0);
    piv_sign_ = 1;

    for (size_t k = 0; k < n; ++k) {
        // Partial pivot: find max in column k below row k
        size_t maxRow = k;
        double maxVal = std::abs(LU_(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            double v = std::abs(LU_(i, k));
            if (v > maxVal) { maxVal = v; maxRow = i; }
        }
        if (maxRow != k) {
            // Swap rows k and maxRow
            for (size_t j = 0; j < n; ++j)
                std::swap(LU_(k, j), LU_(maxRow, j));
            std::swap(piv_[k], piv_[maxRow]);
            piv_sign_ = -piv_sign_;
        }
        if (std::abs(LU_(k, k)) < 1e-300)
            throw std::runtime_error("LinearSolver (LU): singular matrix");
        double inv_kk = 1.0 / LU_(k, k);
        for (size_t i = k + 1; i < n; ++i) {
            LU_(i, k) *= inv_kk;
            for (size_t j = k + 1; j < n; ++j)
                LU_(i, j) -= LU_(i, k) * LU_(k, j);
        }
    }
}

// ── QR factorization (Householder) ────────────────────────────────────────────

void LinearSolver::factorize_qr(const AbstractMatrix& src) {
    auto [Q, R] = ::SharedMath::LinearAlgebra::qr(src);
    Q_ = std::move(Q);
    R_ = std::move(R);
}

// ── Cholesky factorization ────────────────────────────────────────────────────

void LinearSolver::factorize_cholesky(const AbstractMatrix& src) {
    if (src.rows() != src.cols())
        throw std::invalid_argument("LinearSolver (Cholesky): matrix must be square");
    L_ = ::SharedMath::LinearAlgebra::cholesky(src);
}

// ── Triangular solvers ────────────────────────────────────────────────────────

// Solve L*x = b  (lower triangular)
std::vector<double> LinearSolver::forward_sub(
    const DynamicMatrix& L, const std::vector<double>& b, bool unit_diag)
{
    size_t n = L.rows();
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (size_t j = 0; j < i; ++j) s -= L(i, j) * x[j];
        x[i] = unit_diag ? s : s / L(i, i);
    }
    return x;
}

// Solve U*x = b  (upper triangular)
std::vector<double> LinearSolver::backward_sub(
    const DynamicMatrix& U, const std::vector<double>& b)
{
    size_t n = U.rows();
    std::vector<double> x(n);
    for (size_t i = n; i-- > 0; ) {
        double s = b[i];
        for (size_t j = i + 1; j < n; ++j) s -= U(i, j) * x[j];
        x[i] = s / U(i, i);
    }
    return x;
}

// ── solve_lu ──────────────────────────────────────────────────────────────────

std::vector<double> LinearSolver::solve_lu(const std::vector<double>& b) const {
    size_t n = LU_.rows();
    // Apply row permutation
    std::vector<double> pb(n);
    for (size_t i = 0; i < n; ++i) pb[i] = b[piv_[i]];
    // Forward substitution: L*y = pb  (unit lower triangular stored in LU_)
    auto y = forward_sub(LU_, pb, /*unit_diag=*/true);
    // Backward substitution: U*x = y
    return backward_sub(LU_, y);
}

// ── solve_qr ──────────────────────────────────────────────────────────────────

std::vector<double> LinearSolver::solve_qr(const std::vector<double>& b) const {
    // x = R^{-1} * Q^T * b  (least-squares for overdetermined)
    size_t m = Q_.rows(), n = R_.cols(), k = std::min(m, n);
    // y = Q^T * b  (only first k rows needed)
    std::vector<double> y(k, 0.0);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < m; ++j)
            y[i] += Q_(j, i) * b[j];
    // Back-substitute into R (k×n upper triangular, thin)
    DynamicMatrix Rk(k, k);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < k; ++j)
            Rk(i, j) = R_(i, j);
    return backward_sub(Rk, y);
}

// ── solve_cho ─────────────────────────────────────────────────────────────────

std::vector<double> LinearSolver::solve_cho(const std::vector<double>& b) const {
    // A = L * L^T  →  solve L*y = b, then L^T*x = y
    auto y = forward_sub(L_, b, /*unit_diag=*/false);
    // Solve L^T * x = y  using backward substitution on L^T
    size_t n = L_.rows();
    std::vector<double> x(n);
    for (size_t i = n; i-- > 0; ) {
        double s = y[i];
        for (size_t j = i + 1; j < n; ++j) s -= L_(j, i) * x[j];
        x[i] = s / L_(i, i);
    }
    return x;
}

// ── Public solve ──────────────────────────────────────────────────────────────

std::vector<double> LinearSolver::solve(const std::vector<double>& b) const {
    switch (method_) {
    case Method::LU:       return solve_lu(b);
    case Method::QR:       return solve_qr(b);
    case Method::Cholesky: return solve_cho(b);
    default:               return solve_lu(b);
    }
}

DynamicVector LinearSolver::solve(const DynamicVector& b) const {
    return DynamicVector(solve(b.vec()));
}

DynamicMatrix LinearSolver::solve(const DynamicMatrix& B) const {
    size_t nrhs = B.cols();
    DynamicMatrix X(cols_, nrhs);
    std::vector<double> col(B.rows());
    for (size_t j = 0; j < nrhs; ++j) {
        for (size_t i = 0; i < B.rows(); ++i) col[i] = B(i, j);
        auto x = solve(col);
        for (size_t i = 0; i < x.size(); ++i) X(i, j) = x[i];
    }
    return X;
}

// ── inverse ───────────────────────────────────────────────────────────────────

DynamicMatrix LinearSolver::inverse() const {
    if (rows_ != cols_)
        throw std::invalid_argument("LinearSolver::inverse: matrix must be square");
    size_t n = rows_;
    DynamicMatrix I = DynamicMatrix::eye(n);
    return solve(I);
}

// ── determinant ───────────────────────────────────────────────────────────────

double LinearSolver::determinant() const {
    if (method_ == Method::LU) {
        double d = static_cast<double>(piv_sign_);
        for (size_t i = 0; i < LU_.rows(); ++i) d *= LU_(i, i);
        return d;
    }
    if (method_ == Method::Cholesky) {
        // det(A) = det(L)^2 = (prod of L_ii)^2
        double d = 1.0;
        for (size_t i = 0; i < L_.rows(); ++i) d *= L_(i, i);
        return d * d;
    }
    // QR: not ideal for det — fall back to LU
    return LinearSolver(
        (method_ == Method::QR ? Q_ * R_ : DynamicMatrix(rows_, cols_)),
        Method::LU).determinant();
}

// ── rank ──────────────────────────────────────────────────────────────────────

size_t LinearSolver::rank(double tol) const {
    // Use QR with column pivoting (already computed if method==QR, else redo)
    auto [Q2, R2, piv2] = ::SharedMath::LinearAlgebra::qrp(
        (method_ == Method::LU       ? LU_          :
         method_ == Method::QR       ? Q_ * R_      :
         /* Cholesky */                L_ * L_.transposed()));
    if (tol < 0)
        tol = std::max(rows_, cols_) * std::abs(R2(0, 0)) * 2.2e-16;
    size_t r = 0;
    size_t k = std::min(R2.rows(), R2.cols());
    for (size_t i = 0; i < k; ++i)
        if (std::abs(R2(i, i)) > tol) ++r;
    return r;
}

} // namespace SharedMath::LinearAlgebra
