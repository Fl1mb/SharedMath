#pragma once

/**
 * @file IntegralEquations.h
 * @brief Solvers for Fredholm and Volterra integral equations.
 *
 * @defgroup NumericalMethods_IE Integral Equations
 * @ingroup NumericalMethods
 */

#include <functional>
#include <vector>
#include <utility>
#include <sharedmath_numericalmethods_export.h>

namespace SharedMath::NumericalMethods {

/// Fredholm integral equation of the 2nd kind:
///
///   y(x) = f(x) + λ ∫_a^b K(x,t) y(t) dt
///
/// Discretises onto n equidistant nodes via the trapezoidal rule,
/// reducing the problem to a linear system (I - λhWK)y = f.
/// Returns {nodes, solution_values}.
SHAREDMATH_NUMERICALMETHODS_EXPORT
std::pair<std::vector<double>, std::vector<double>>
fredholm2(std::function<double(double)> f,
          std::function<double(double, double)> K,
          double a, double b, double lambda, size_t n = 100);

/// Volterra integral equation of the 2nd kind:
///
///   y(x) = f(x) + λ ∫_a^x K(x,t) y(t) dt
///
/// Solved step-by-step with the trapezoidal rule.
/// Returns {nodes, solution_values}.
SHAREDMATH_NUMERICALMETHODS_EXPORT
std::pair<std::vector<double>, std::vector<double>>
volterra2(std::function<double(double)> f,
          std::function<double(double, double)> K,
          double a, double b, double lambda, size_t n = 100);

} // namespace SharedMath::NumericalMethods
