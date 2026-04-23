#include "NumericalMethods/ODE.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace SharedMath::NumericalMethods {

// ── Dormand-Prince RK45 coefficients ────────────────────────────────────────
namespace {
    // Butcher tableau (a, c, b5, e = b5 - b4)
    constexpr double DP_A21 = 1.0/5.0;
    constexpr double DP_A31 = 3.0/40.0,     DP_A32 = 9.0/40.0;
    constexpr double DP_A41 = 44.0/45.0,    DP_A42 = -56.0/15.0,   DP_A43 = 32.0/9.0;
    constexpr double DP_A51 = 19372.0/6561, DP_A52 = -25360.0/2187,DP_A53 = 64448.0/6561, DP_A54 = -212.0/729;
    constexpr double DP_A61 = 9017.0/3168,  DP_A62 = -355.0/33,    DP_A63 = 46732.0/5247, DP_A64 = 49.0/176,   DP_A65 = -5103.0/18656;
    // 5th-order weights
    constexpr double DP_B1 = 35.0/384,  DP_B3 = 500.0/1113, DP_B4 = 125.0/192,
                     DP_B5 = -2187.0/6784, DP_B6 = 11.0/84;
    // Error (difference between 5th and 4th-order)
    constexpr double DP_E1 = 71.0/57600,  DP_E3 = -71.0/16695, DP_E4 = 71.0/1920,
                     DP_E5 = -17253.0/339200, DP_E6 = 22.0/525, DP_E7 = -1.0/40;
} // namespace

// ── Scalar methods ───────────────────────────────────────────────────────────

ScalarODESolution euler(ScalarODE f, double y0, double t0, double t1, double h) {
    ScalarODESolution sol;
    double t = t0, y = y0;
    sol.t.push_back(t); sol.y.push_back(y);
    while (t < t1 - 1e-12) {
        if (t + h > t1) h = t1 - t;
        y += h * f(t, y);
        t += h;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }
    return sol;
}

ScalarODESolution rk4(ScalarODE f, double y0, double t0, double t1, double h) {
    ScalarODESolution sol;
    double t = t0, y = y0;
    sol.t.push_back(t); sol.y.push_back(y);
    while (t < t1 - 1e-12) {
        if (t + h > t1) h = t1 - t;
        double k1 = f(t,           y);
        double k2 = f(t + 0.5*h,  y + 0.5*h*k1);
        double k3 = f(t + 0.5*h,  y + 0.5*h*k2);
        double k4 = f(t + h,       y + h*k3);
        y += h / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4);
        t += h;
        sol.t.push_back(t);
        sol.y.push_back(y);
    }
    return sol;
}

ScalarODESolution rk45(ScalarODE f, double y0,
                        double t0, double t1,
                        double tol, double h0)
{
    ScalarODESolution sol;
    double t = t0, y = y0, h = h0;
    sol.t.push_back(t); sol.y.push_back(y);
    const double hmin = 1e-12, hmax = (t1 - t0) * 0.1;

    while (t < t1 - 1e-12) {
        if (t + h > t1) h = t1 - t;

        double k1 = f(t, y);
        double k2 = f(t + h/5.0,          y + h*DP_A21*k1);
        double k3 = f(t + 3.0*h/10.0,     y + h*(DP_A31*k1 + DP_A32*k2));
        double k4 = f(t + 4.0*h/5.0,      y + h*(DP_A41*k1 + DP_A42*k2 + DP_A43*k3));
        double k5 = f(t + 8.0*h/9.0,      y + h*(DP_A51*k1 + DP_A52*k2 + DP_A53*k3 + DP_A54*k4));
        double k6 = f(t + h,               y + h*(DP_A61*k1 + DP_A62*k2 + DP_A63*k3 + DP_A64*k4 + DP_A65*k5));
        double y5 = y + h*(DP_B1*k1 + DP_B3*k3 + DP_B4*k4 + DP_B5*k5 + DP_B6*k6);
        double k7 = f(t + h, y5);

        double err = std::abs(h) * std::abs(DP_E1*k1 + DP_E3*k3 + DP_E4*k4 + DP_E5*k5 + DP_E6*k6 + DP_E7*k7);
        double scale = tol * (1.0 + std::abs(y));

        if (err <= scale) {
            t += h;
            y  = y5;
            sol.t.push_back(t);
            sol.y.push_back(y);
        }

        double factor = (err > 0.0) ? 0.9 * std::pow(scale / err, 0.2) : 5.0;
        factor = std::clamp(factor, 0.1, 5.0);
        h = std::clamp(h * factor, hmin, hmax);
    }
    return sol;
}

// ── System helpers ───────────────────────────────────────────────────────────

namespace {
    std::vector<double> vec_add(const std::vector<double>& a,
                                 const std::vector<double>& b) {
        std::vector<double> r(a.size());
        for (size_t i = 0; i < a.size(); ++i) r[i] = a[i] + b[i];
        return r;
    }
    std::vector<double> vec_axpy(double s, const std::vector<double>& x,
                                  const std::vector<double>& y) {
        std::vector<double> r(x.size());
        for (size_t i = 0; i < x.size(); ++i) r[i] = s * x[i] + y[i];
        return r;
    }
    std::vector<double> vec_scale(double s, const std::vector<double>& x) {
        std::vector<double> r(x.size());
        for (size_t i = 0; i < x.size(); ++i) r[i] = s * x[i];
        return r;
    }
    double vec_norm(const std::vector<double>& x) {
        double s = 0;
        for (double v : x) s += v * v;
        return std::sqrt(s);
    }
} // namespace

// ── System methods ───────────────────────────────────────────────────────────

SystemODESolution euler_system(SystemODE f, std::vector<double> y0,
                                double t0, double t1, double h)
{
    SystemODESolution sol;
    double t = t0;
    std::vector<double> y = y0;
    sol.t.push_back(t); sol.y.push_back(y);
    while (t < t1 - 1e-12) {
        if (t + h > t1) h = t1 - t;
        std::vector<double> k = f(t, y);
        y = vec_axpy(h, k, y);
        t += h;
        sol.t.push_back(t); sol.y.push_back(y);
    }
    return sol;
}

SystemODESolution rk4_system(SystemODE f, std::vector<double> y0,
                               double t0, double t1, double h)
{
    SystemODESolution sol;
    double t = t0;
    std::vector<double> y = y0;
    sol.t.push_back(t); sol.y.push_back(y);
    size_t n = y.size();
    while (t < t1 - 1e-12) {
        if (t + h > t1) h = t1 - t;
        auto k1 = f(t,          y);
        auto k2 = f(t + 0.5*h,  vec_axpy(0.5*h, k1, y));
        auto k3 = f(t + 0.5*h,  vec_axpy(0.5*h, k2, y));
        auto k4 = f(t + h,       vec_axpy(h,     k3, y));
        for (size_t i = 0; i < n; ++i)
            y[i] += h / 6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        t += h;
        sol.t.push_back(t); sol.y.push_back(y);
    }
    return sol;
}

SystemODESolution rk45_system(SystemODE f, std::vector<double> y0,
                                double t0, double t1,
                                double tol, double h0)
{
    SystemODESolution sol;
    double t = t0, h = h0;
    std::vector<double> y = y0;
    size_t n = y.size();
    sol.t.push_back(t); sol.y.push_back(y);
    const double hmin = 1e-12, hmax = (t1 - t0) * 0.1;

    while (t < t1 - 1e-12) {
        if (t + h > t1) h = t1 - t;

        auto k1 = f(t, y);
        auto k2 = f(t + h/5.0,     vec_axpy(h*DP_A21, k1, y));
        auto k3 = f(t + 3.*h/10.,  vec_add(vec_axpy(h*DP_A31, k1, y), vec_scale(h*DP_A32, k2)));
        auto k4 = f(t + 4.*h/5.,   vec_add(vec_add(vec_axpy(h*DP_A41, k1, y),
                                            vec_scale(h*DP_A42, k2)), vec_scale(h*DP_A43, k3)));
        auto k5 = f(t + 8.*h/9.,   vec_add(vec_add(vec_add(vec_axpy(h*DP_A51, k1, y),
                                            vec_scale(h*DP_A52, k2)), vec_scale(h*DP_A53, k3)),
                                            vec_scale(h*DP_A54, k4)));
        auto k6 = f(t + h,          vec_add(vec_add(vec_add(vec_add(vec_axpy(h*DP_A61, k1, y),
                                            vec_scale(h*DP_A62, k2)), vec_scale(h*DP_A63, k3)),
                                            vec_scale(h*DP_A64, k4)), vec_scale(h*DP_A65, k5)));

        std::vector<double> y5(n);
        for (size_t i = 0; i < n; ++i)
            y5[i] = y[i] + h*(DP_B1*k1[i] + DP_B3*k3[i] + DP_B4*k4[i] + DP_B5*k5[i] + DP_B6*k6[i]);

        auto k7 = f(t + h, y5);

        // Error estimate
        std::vector<double> err_vec(n);
        for (size_t i = 0; i < n; ++i)
            err_vec[i] = h*(DP_E1*k1[i] + DP_E3*k3[i] + DP_E4*k4[i] +
                            DP_E5*k5[i] + DP_E6*k6[i] + DP_E7*k7[i]);
        double err   = vec_norm(err_vec);
        double scale = tol * (1.0 + vec_norm(y));

        if (err <= scale) {
            t += h;
            y  = y5;
            sol.t.push_back(t); sol.y.push_back(y);
        }

        double factor = (err > 0.0) ? 0.9 * std::pow(scale / err, 0.2) : 5.0;
        factor = std::clamp(factor, 0.1, 5.0);
        h = std::clamp(h * factor, hmin, hmax);
    }
    return sol;
}

} // namespace SharedMath::NumericalMethods
