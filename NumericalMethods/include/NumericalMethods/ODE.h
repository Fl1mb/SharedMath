#pragma once

#include <functional>
#include <vector>
#include <sharedmath_numericalmethods_export.h>

namespace SharedMath::NumericalMethods {

// Right-hand side types
using ScalarODE = std::function<double(double t, double y)>;
using SystemODE = std::function<std::vector<double>(double t, const std::vector<double>& y)>;

struct ScalarODESolution {
    std::vector<double> t;
    std::vector<double> y;
};

struct SystemODESolution {
    std::vector<double> t;
    std::vector<std::vector<double>> y; // y[step][component]
};

// ── Scalar ODE: dy/dt = f(t,y),  y(t0) = y0 ──────────────────────────────

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

// ── ODE system: dy/dt = f(t,y),  y(t0) = y0 ──────────────────────────────

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

} // namespace SharedMath::NumericalMethods
