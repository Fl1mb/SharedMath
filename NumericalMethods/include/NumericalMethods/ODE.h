#pragma once

/**
 * @file ODE.h
 * @brief Scalar and system ordinary differential equation solvers.
 *
 * @defgroup NumericalMethods_ODE ODE Solvers
 * @ingroup NumericalMethods
 *
 * Includes explicit methods (Euler, RK4, RK45), implicit BDF (Gear),
 * and implicit Adams-Moulton families.
 *
 * @section sec_cuda_ode CUDA Acceleration Opportunities
 *
 * For large systems (n >> 100), the following stages benefit from GPU:
 *  - Jacobian computation (finite differences across n columns)
 *  - Linear system solve (cuSOLVER for n > ~200)
 *  - Batch ODE solving (many independent systems on GPU simultaneously)
 *  - Vector-valued f(t,y) evaluations when f is itself GPU-parallelizable
 */

#include <functional>
#include <vector>
#include <cstddef>
#include <sharedmath_numericalmethods_export.h>

namespace SharedMath::NumericalMethods {

/// Right-hand side types
using ScalarODE = std::function<double(double t, double y)>;
using SystemODE = std::function<std::vector<double>(double t, const std::vector<double>& y)>;

/// Jacobian type for implicit solvers: J[i][j] = df_i/dy_j
using JacobianODE = std::function<std::vector<std::vector<double>>(double t, const std::vector<double>& y)>;

struct ScalarODESolution {
    std::vector<double> t;
    std::vector<double> y;
};

struct SystemODESolution {
    std::vector<double> t;
    std::vector<std::vector<double>> y; // y[step][component]
};

/// ── Scalar ODE: dy/dt = f(t,y),  y(t0) = y0 ──────────────────────────────

SHAREDMATH_NUMERICALMETHODS_EXPORT
ScalarODESolution euler(ScalarODE f, double y0,
                         double t0, double t1, double h);

SHAREDMATH_NUMERICALMETHODS_EXPORT
ScalarODESolution rk4(ScalarODE f, double y0,
                       double t0, double t1, double h);

SHAREDMATH_NUMERICALMETHODS_EXPORT
ScalarODESolution rk45(ScalarODE f, double y0,
                        double t0, double t1,
                        double tol = 1e-6, double h0 = 1e-3);

/// ── ODE system: dy/dt = f(t,y),  y(t0) = y0 ──────────────────────────────

SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution euler_system(SystemODE f, std::vector<double> y0,
                                double t0, double t1, double h);

SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution rk4_system(SystemODE f, std::vector<double> y0,
                               double t0, double t1, double h);

SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution rk45_system(SystemODE f, std::vector<double> y0,
                                double t0, double t1,
                                double tol = 1e-6, double h0 = 1e-3);

/// ── BDF (Gear) method for stiff systems: dy/dt = f(t,y) ─────────────────
///
/// Backward Differentiation Formula of orders 1..6 with adaptive step size.
/// Requires the Jacobian df/dy (computed numerically if not provided).
///
/// @param f        Right-hand side f(t, y)
/// @param y0       Initial condition
/// @param t0       Start time
/// @param t1       End time
/// @param order    BDF order k ∈ [1, 6] (default: 1, auto-adjusted at runtime)
/// @param tol      Tolerance for step-size control
/// @param h0       Initial step size
SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution bdf(SystemODE f, std::vector<double> y0,
                       double t0, double t1,
                       size_t order = 1, double tol = 1e-6, double h0 = 1e-3);

/// BDF with user-supplied Jacobian
SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution bdf(SystemODE f, JacobianODE J, std::vector<double> y0,
                       double t0, double t1,
                       size_t order = 1, double tol = 1e-6, double h0 = 1e-3);

/// ── Adams-Moulton methods for non-stiff / mildly-stiff systems ──────────
///
/// Implicit Adams-Moulton formulas of orders 1..5 with adaptive step size.
/// Order 1 = Backward Euler, order 2 = Trapezoidal rule.
/// Higher orders require more function history; startup via RK4.
///
/// @param f        Right-hand side f(t, y)
/// @param y0       Initial condition
/// @param t0       Start time
/// @param t1       End time
/// @param order    AM order k ∈ [1, 5] (default: 2 = Trapezoidal)
/// @param tol      Tolerance for step-size control
/// @param h0       Initial step size
SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution adams_moulton(SystemODE f, std::vector<double> y0,
                                  double t0, double t1,
                                  size_t order = 2, double tol = 1e-6,
                                  double h0 = 1e-3);

/// Adams-Moulton with user-supplied Jacobian
SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution adams_moulton(SystemODE f, JacobianODE J,
                                  std::vector<double> y0,
                                  double t0, double t1,
                                  size_t order = 2, double tol = 1e-6,
                                  double h0 = 1e-3);

/// ── Adams-Bashforth-Moulton predictor-corrector ─────────────────────────
///
/// PECE (Predict-Evaluate-Correct-Evaluate) Adams pair.
/// Adams-Bashforth predicts y_{n+1}, then Adams-Moulton corrects it.
/// Orders 1..5; error estimated from predictor-corrector difference.
///
/// @param f        Right-hand side f(t, y)
/// @param y0       Initial condition
/// @param t0       Start time
/// @param t1       End time
/// @param order    Pair order k ∈ [1, 5] (default: 4)
/// @param tol      Tolerance for step-size control
/// @param h0       Initial step size
SHAREDMATH_NUMERICALMETHODS_EXPORT
SystemODESolution abm(SystemODE f, std::vector<double> y0,
                       double t0, double t1,
                       size_t order = 4, double tol = 1e-6,
                       double h0 = 1e-3);

} // namespace SharedMath::NumericalMethods
