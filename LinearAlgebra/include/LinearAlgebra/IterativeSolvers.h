#pragma once

#include "AbstractMatrix.h"
#include <sharedmath_linearalgebra_export.h>
#include <vector>

namespace SharedMath::LinearAlgebra {

/// ── Conjugate Gradient ────────────────────────────────────────────────────────
/// Solves Ax = b where A must be symmetric positive definite.
/// x0     — initial guess; empty → zero vector
/// tol    — convergence threshold on ||r||_2
/// max_iter — maximum number of iterations
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> cg(const AbstractMatrix& A,
                        const std::vector<double>& b,
                        std::vector<double> x0 = {},
                        double tol = 1e-10,
                        size_t max_iter = 1000);

/// ── LSQR ─────────────────────────────────────────────────────────────────────
/// Solves min ||Ax - b||_2 for any m×n matrix A (overdetermined, underdetermined,
/// or rank-deficient). Based on Lanczos bidiagonalization.
/// tol    — convergence threshold on the relative residual
/// max_iter — maximum number of Lanczos steps
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> lsqr(const AbstractMatrix& A,
                          const std::vector<double>& b,
                          double tol = 1e-10,
                          size_t max_iter = 1000);

/// ── GMRES(m) ──────────────────────────────────────────────────────────────────
/// Generalised Minimal Residual — works for any square non-singular system,
/// including non-symmetric and indefinite matrices.
/// restart  — Krylov subspace dimension before restart (GMRES(restart))
/// x0       — initial guess; empty → zero vector
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> gmres(const AbstractMatrix& A,
                           const std::vector<double>& b,
                           std::vector<double> x0 = {},
                           double tol     = 1e-10,
                           size_t max_iter = 1000,
                           size_t restart  = 50);

/// ── BiCGSTAB ─────────────────────────────────────────────────────────────────
/// Biconjugate Gradient Stabilised — robust for non-symmetric systems,
/// short recurrence (constant memory unlike GMRES).
/// x0 — initial guess; empty → zero vector
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> bicgstab(const AbstractMatrix& A,
                              const std::vector<double>& b,
                              std::vector<double> x0 = {},
                              double tol     = 1e-10,
                              size_t max_iter = 1000);

/// ── MINRES ────────────────────────────────────────────────────────────────────
/// Minimum Residual method (Paige & Saunders) — for symmetric systems
/// (positive definite, indefinite, or singular).
/// x0 — initial guess; empty → zero vector
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> minres(const AbstractMatrix& A,
                            const std::vector<double>& b,
                            std::vector<double> x0 = {},
                            double tol     = 1e-10,
                            size_t max_iter = 1000);

} // namespace SharedMath::LinearAlgebra
