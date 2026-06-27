#pragma once

/**
 * @file NonlinearEquations.h
 * @brief Iterative solvers for nonlinear equations F(x) = 0.
 *
 * @defgroup NumericalMethods_NLE Nonlinear Equation Solvers
 * @ingroup NumericalMethods
 *
 * Provides Newton-Raphson (exact Jacobian) and Broyden's method
 * (secant-like Jacobian update — superlinear convergence without
 * analytic derivatives after the first step).
 */

#include <functional>
#include <vector>
#include <cstddef>
#include <sharedmath_numericalmethods_export.h>

namespace SharedMath::NumericalMethods {

struct NLEIterResult {
    std::vector<double> x;   /// Approximate root
    size_t iterations;       /// Number of iterations performed
    double residual;         /// ||F(x)|| — residual norm
    bool   converged;        /// true if the method converged within tolerance
};

using JacobianFunc = std::function<std::vector<std::vector<double>>(const std::vector<double>&)>;

/// Newton-Raphson for systems: F(x) = 0, with analytic Jacobian
SHAREDMATH_NUMERICALMETHODS_EXPORT
NLEIterResult newton_raphson_system(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    JacobianFunc J,
    std::vector<double> x0,
    double tol = 1e-10, size_t max_iter = 100);

/// Newton-Raphson for systems: F(x) = 0, with numerical Jacobian
SHAREDMATH_NUMERICALMETHODS_EXPORT
NLEIterResult newton_raphson_system(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    std::vector<double> x0,
    double tol = 1e-10, size_t max_iter = 100);

/// Broyden's method (bad Broyden update): F(x) = 0, analytic Jacobian on first step
SHAREDMATH_NUMERICALMETHODS_EXPORT
NLEIterResult broyden(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    JacobianFunc J,
    std::vector<double> x0,
    double tol = 1e-10, size_t max_iter = 100);

/// Broyden's method: F(x) = 0, numerical Jacobian on first step
SHAREDMATH_NUMERICALMETHODS_EXPORT
NLEIterResult broyden(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    std::vector<double> x0,
    double tol = 1e-10, size_t max_iter = 100);

} // namespace SharedMath::NumericalMethods
