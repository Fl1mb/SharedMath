#pragma once

#include <functional>
#include <vector>

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

// Explicit Euler, fixed step h
ScalarODESolution euler(ScalarODE f, double y0,
                         double t0, double t1, double h);

// Classic 4th-order Runge-Kutta, fixed step h
ScalarODESolution rk4(ScalarODE f, double y0,
                       double t0, double t1, double h);

// Dormand-Prince RK45, adaptive step, error tolerance tol, initial step h0
ScalarODESolution rk45(ScalarODE f, double y0,
                        double t0, double t1,
                        double tol = 1e-6, double h0 = 1e-3);

// ── ODE system: dy/dt = f(t,y),  y(t0) = y0 ──────────────────────────────

SystemODESolution euler_system(SystemODE f, std::vector<double> y0,
                                double t0, double t1, double h);

SystemODESolution rk4_system(SystemODE f, std::vector<double> y0,
                               double t0, double t1, double h);

SystemODESolution rk45_system(SystemODE f, std::vector<double> y0,
                                double t0, double t1,
                                double tol = 1e-6, double h0 = 1e-3);

} // namespace SharedMath::NumericalMethods
