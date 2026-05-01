#pragma once

/**
 * @file Integration.h
 * @brief Numerical integration and quadrature utilities.
 *
 * @defgroup NumericalMethods_Int Numerical Integration
 * @ingroup NumericalMethods
 */

#include <functional>
#include <cstddef>
#include <sharedmath_numericalmethods_export.h>

namespace SharedMath::NumericalMethods {

/// Midpoint (rectangle) rule
SHAREDMATH_NUMERICALMETHODS_EXPORT
double integrate_rect(std::function<double(double)> f,
                       double a, double b, size_t n = 1000);

/// Trapezoidal rule
SHAREDMATH_NUMERICALMETHODS_EXPORT
double integrate_trap(std::function<double(double)> f,
                       double a, double b, size_t n = 1000);

/// Simpson's 1/3 rule (n is rounded up to even if needed)
SHAREDMATH_NUMERICALMETHODS_EXPORT
double integrate_simpson(std::function<double(double)> f,
                          double a, double b, size_t n = 1000);

/// Gauss-Legendre quadrature, order 1..5
SHAREDMATH_NUMERICALMETHODS_EXPORT
double integrate_gauss(std::function<double(double)> f,
                        double a, double b, int order = 5);

/// Adaptive Simpson's quadrature to absolute tolerance tol
SHAREDMATH_NUMERICALMETHODS_EXPORT
double integrate_adaptive(std::function<double(double)> f,
                           double a, double b,
                           double tol = 1e-8, size_t max_depth = 50);

/// Double integral ∫_{ax}^{bx} ∫_{ay(x)}^{by(x)} f(x,y) dy dx
/// y-bounds ay and by may depend on x
SHAREDMATH_NUMERICALMETHODS_EXPORT
double integrate2d(std::function<double(double, double)> f,
                   double ax, double bx,
                   std::function<double(double)> ay,
                   std::function<double(double)> by,
                   size_t nx = 100, size_t ny = 100);

} // namespace SharedMath::NumericalMethods
