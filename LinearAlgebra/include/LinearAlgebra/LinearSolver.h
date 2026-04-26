#pragma once

#include "AbstractMatrix.h"
#include "DynamicMatrix.h"
#include "DynamicVector.h"
#include <sharedmath_linearalgebra_export.h>

#include <memory>
#include <vector>
#include <stdexcept>

namespace SharedMath::LinearAlgebra {

// Unified interface for solving linear systems and computing matrix properties.
//
// Usage (automatic method selection):
//   LinearSolver sol(A);                      // detects SPD/symmetric/general
//   DynamicVector x  = sol.solve(b);          // solve Ax = b
//   DynamicMatrix X  = sol.solve(B);          // solve AX = B (multiple rhs)
//   DynamicMatrix Ai = sol.inverse();         // A^{-1}
//   double        d  = sol.determinant();     // det(A)
//   size_t        r  = sol.rank();            // numerical rank
//
// Usage (explicit method):
//   auto sol = LinearSolver::cholesky(A);     // A must be SPD
//   auto sol = LinearSolver::qr(A);           // A may be rectangular (least-squares)
//   auto sol = LinearSolver::lu(A);           // A must be square
//
class SHAREDMATH_LINEARALGEBRA_EXPORT LinearSolver {
public:
    enum class Method {
        Auto,       // Detect: SPD → Cholesky, symmetric → LDLT, general → LU
        LU,         // PA = LU with partial pivoting (square only)
        QR,         // Householder QR (works for rectangular; gives least-squares)
        Cholesky,   // LL^T (A must be symmetric positive-definite)
    };

    // Factorize A using the chosen method.
    // Throws std::invalid_argument if the matrix is incompatible with the method.
    explicit LinearSolver(const AbstractMatrix& A, Method method = Method::Auto);

    // ── Solve ─────────────────────────────────────────────────────────────

    // Solve Ax = b.  For QR and overdetermined systems returns least-squares solution.
    DynamicVector solve(const DynamicVector& b) const;
    std::vector<double> solve(const std::vector<double>& b) const;

    // Solve AX = B (multiple right-hand sides).
    DynamicMatrix solve(const DynamicMatrix& B) const;

    // ── Matrix properties from factorization ──────────────────────────────

    // A^{-1} (square systems only; uses the stored factorization)
    DynamicMatrix inverse() const;

    // det(A) from the factorization (sign tracking for LU/Cholesky)
    double determinant() const;

    // Numerical rank via QR with column pivoting (independent of factorization method)
    size_t rank(double tol = -1.0) const;

    // ── Named constructors ────────────────────────────────────────────────
    static LinearSolver lu       (const AbstractMatrix& A);
    static LinearSolver qr       (const AbstractMatrix& A);
    static LinearSolver cholesky (const AbstractMatrix& A);

    Method method() const noexcept { return method_; }
    size_t rows()   const noexcept { return rows_; }
    size_t cols()   const noexcept { return cols_; }

private:
    Method method_;
    size_t rows_, cols_;

    // ── LU factorization storage ──────────────────────────────────────────
    DynamicMatrix LU_;           // combined L (below diag) + U (above) in-place
    std::vector<size_t> piv_;    // row permutation from partial pivoting
    int piv_sign_ = 1;           // +1 or -1 depending on number of row swaps

    // ── QR factorization storage ──────────────────────────────────────────
    DynamicMatrix Q_, R_;

    // ── Cholesky factorization storage ────────────────────────────────────
    DynamicMatrix L_;            // lower-triangular factor, A = L*L^T

    // ── Internal helpers ──────────────────────────────────────────────────
    void factorize_lu      (const AbstractMatrix& A);
    void factorize_qr      (const AbstractMatrix& A);
    void factorize_cholesky(const AbstractMatrix& A);

    // Back/forward substitution
    std::vector<double> solve_lu (const std::vector<double>& b) const;
    std::vector<double> solve_qr (const std::vector<double>& b) const;
    std::vector<double> solve_cho(const std::vector<double>& b) const;

    // Triangular solvers
    static std::vector<double> forward_sub (const DynamicMatrix& L,
                                             const std::vector<double>& b,
                                             bool unit_diag = false);
    static std::vector<double> backward_sub(const DynamicMatrix& U,
                                             const std::vector<double>& b);
};

} // namespace SharedMath::LinearAlgebra
