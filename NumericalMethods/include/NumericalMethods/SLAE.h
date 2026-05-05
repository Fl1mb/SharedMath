#pragma once

/**
 * @file SLAE.h
 * @brief Iterative solvers for systems of linear algebraic equations.
 *
 * @defgroup NumericalMethods_SLAE Linear System Solvers
 * @ingroup NumericalMethods
 */

#include <vector>
#include <cstddef>
#include "LinearAlgebra/AbstractMatrix.h"
#include <sharedmath_numericalmethods_export.h>

namespace SharedMath::NumericalMethods {

using SharedMath::LinearAlgebra::AbstractMatrix;

struct IterResult {
    std::vector<double> x;
    size_t iterations;
    double residual;   // ||Ax - b||_2
    bool   converged;
};

/// Jacobi iterative method (requires strictly diagonally dominant A)
SHAREDMATH_NUMERICALMETHODS_EXPORT
IterResult jacobi(const AbstractMatrix& A,
                   const std::vector<double>& b,
                   std::vector<double> x0 = {},
                   double tol = 1e-10, size_t max_iter = 10000);

/// Gauss-Seidel method (converges for SDD and SPD matrices)
SHAREDMATH_NUMERICALMETHODS_EXPORT
IterResult gauss_seidel(const AbstractMatrix& A,
                         const std::vector<double>& b,
                         std::vector<double> x0 = {},
                         double tol = 1e-10, size_t max_iter = 10000);

/// Successive Over-Relaxation; omega=1 → Gauss-Seidel, omega∈(1,2) accelerates
SHAREDMATH_NUMERICALMETHODS_EXPORT
IterResult sor(const AbstractMatrix& A,
                const std::vector<double>& b,
                double omega = 1.5,
                std::vector<double> x0 = {},
                double tol = 1e-10, size_t max_iter = 10000);

/// Conjugate Gradient (requires symmetric positive-definite A)
SHAREDMATH_NUMERICALMETHODS_EXPORT
IterResult conjugate_gradient(const AbstractMatrix& A,
                               const std::vector<double>& b,
                               std::vector<double> x0 = {},
                               double tol = 1e-10, size_t max_iter = 10000);

/// Restarted GMRES (works for any non-singular A)
SHAREDMATH_NUMERICALMETHODS_EXPORT
IterResult gmres(const AbstractMatrix& A,
                  const std::vector<double>& b,
                  size_t restart = 30,
                  std::vector<double> x0 = {},
                  double tol = 1e-10, size_t max_iter = 10000);

} // namespace SharedMath::NumericalMethods
