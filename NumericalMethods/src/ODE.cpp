#include "NumericalMethods/ODE.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <deque>

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

    // ── BDF alpha coefficients ──────────────────────────────────────────────
    // BDF-k: sum_{i=0}^{k} alpha[i] * y_{n+1-i} = h * f(x_{n+1}, y_{n+1})
    // alpha[0] is the coefficient of y_{n+1}
    struct BDFCoeffs {
        double alpha[7]; // alpha[0..k]
    };

    constexpr BDFCoeffs BDF_COEFFS[6] = {
        // k=1: y_{n+1} - y_n = h f_{n+1}
        {{  1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0}},
        // k=2: (3/2)y_{n+1} - 2y_n + (1/2)y_{n-1} = h f_{n+1}
        {{  1.5, -2.0,  0.5,  0.0,  0.0,  0.0,  0.0}},
        // k=3: (11/6)y_{n+1} - 3y_n + (3/2)y_{n-1} - (1/3)y_{n-2} = h f_{n+1}
        {{  11.0/6.0, -3.0,  1.5, -1.0/3.0,  0.0,  0.0,  0.0}},
        // k=4
        {{  25.0/12.0, -4.0,  3.0, -4.0/3.0,  0.25,  0.0,  0.0}},
        // k=5
        {{  137.0/60.0, -5.0,  5.0, -10.0/3.0,  1.25, -0.2,  0.0}},
        // k=6
        {{  49.0/20.0, -6.0,  7.5, -20.0/3.0,  3.75, -1.2,  1.0/6.0}}
    };

    // Error constants C_{k+1} for step-size estimation
    constexpr double BDF_ERR_CONST[6] = {
        -1.0/2.0, -2.0/3.0, -3.0/4.0, -4.0/5.0, -5.0/6.0, -6.0/7.0
    };
} // namespace

// ── Vector helpers (internal) ───────────────────────────────────────────────

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

// ── BDF (Gear) method for stiff systems ─────────────────────────────────────

namespace {
    // Solve linear system Ax = b via Gaussian elimination with partial pivoting
    std::vector<double> bdf_solve_linear(
        std::vector<std::vector<double>> A,
        std::vector<double> b)
    {
        size_t n = b.size();
        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            double max_val = std::abs(A[i][i]);
            for (size_t k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > max_val) {
                    max_val = std::abs(A[k][i]);
                    pivot = k;
                }
            }
            if (max_val < 1e-15)
                throw std::runtime_error("bdf: singular Jacobian");
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);
            for (size_t k = i + 1; k < n; ++k) {
                double factor = A[k][i] / A[i][i];
                for (size_t j = i; j < n; ++j)
                    A[k][j] -= factor * A[i][j];
                b[k] -= factor * b[i];
            }
        }
        std::vector<double> x(n);
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            x[i] = b[i];
            for (size_t j = i + 1; j < n; ++j)
                x[i] -= A[i][j] * x[j];
            x[i] /= A[i][i];
        }
        return x;
    }

    // Numerical Jacobian df/dy at (t, y)
    std::vector<std::vector<double>> bdf_jacobian(
        SystemODE f, double t, const std::vector<double>& y)
    {
        size_t n = y.size();
        auto ft = f(t, y);
        std::vector<std::vector<double>> J(n, std::vector<double>(n));
        for (size_t j = 0; j < n; ++j) {
            std::vector<double> yh = y;
            double h = 1e-7 * (1.0 + std::abs(y[j]));
            yh[j] += h;
            auto fth = f(t, yh);
            for (size_t i = 0; i < n; ++i)
                J[i][j] = (fth[i] - ft[i]) / h;
        }
        return J;
    }

    // One BDF step: solve alpha0*y_new - h*f(t_new, y_new) + sum_{i=1}^k alpha_i*y_{n+1-i} = 0
    // Returns y_new via Newton iteration. Returns empty vector on failure.
    std::vector<double> bdf_step(
        SystemODE f, JacobianODE J,
        const std::deque<std::vector<double>>& yhist,
        double t_new, double h, size_t k)
    {
        size_t n = yhist[0].size();
        const auto& alpha = BDF_COEFFS[k - 1].alpha;

        // Compute sum_{i=1}^{k} alpha_i * y_{n+1-i}
        std::vector<double> sum_alpha(n, 0.0);
        for (size_t i = 1; i <= k; ++i)
            for (size_t j = 0; j < n; ++j)
                sum_alpha[j] += alpha[i] * yhist[i - 1][j];

        // Initial guess: y_n (most recent value)
        std::vector<double> y_new = yhist[0];

        // Newton iteration
        bool converged = false;
        for (size_t iter = 0; iter < 50; ++iter) {
            auto fy = f(t_new, y_new);
            auto Jac = J(t_new, y_new);

            // F(y_new) = alpha0*y_new - h*f(t_new, y_new) + sum_alpha
            std::vector<double> F(n);
            for (size_t i = 0; i < n; ++i)
                F[i] = alpha[0] * y_new[i] - h * fy[i] + sum_alpha[i];

            double res = vec_norm(F);
            if (res < 1e-10) { converged = true; break; }

            // J_sys = alpha0*I - h*Jacobian
            std::vector<std::vector<double>> Jsys(n, std::vector<double>(n));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    Jsys[i][j] = -h * Jac[i][j];
                    if (i == j) Jsys[i][j] += alpha[0];
                }
            }

            auto dy = bdf_solve_linear(Jsys, F);
            for (size_t i = 0; i < n; ++i)
                y_new[i] -= dy[i];
        }

        if (!converged) return {};
        return y_new;
    }

    // Starting procedure: generate k initial values using RK4
    std::deque<std::vector<double>> bdf_start(
        SystemODE f, const std::vector<double>& y0,
        double t0, double h, size_t k)
    {
        std::deque<std::vector<double>> history;
        history.push_front(y0);

        std::vector<double> y = y0;
        double t = t0;
        size_t n = y0.size();

        for (size_t i = 1; i < k; ++i) {
            auto k1 = f(t, y);
            auto k2 = f(t + 0.5*h, vec_axpy(0.5*h, k1, y));
            auto k3 = f(t + 0.5*h, vec_axpy(0.5*h, k2, y));
            auto k4 = f(t + h, vec_axpy(h, k3, y));
            for (size_t j = 0; j < n; ++j)
                y[j] += h / 6.0 * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
            t += h;
            history.push_front(y);
        }

        return history;
    }

    // Error estimator: compare BDF-k with BDF-(k-1) using difference formula
    double bdf_error_estimate(const std::deque<std::vector<double>>& history,
                               size_t k)
    {
        if (k < 2 || history.size() < 2) return 0.0;
        // Error ~ |y_new - y_prev| scaled by error constant
        size_t n = history[0].size();
        double err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d = history[0][i] - history[1][i];
            err += d * d;
        }
        return std::sqrt(err) / std::abs(BDF_ERR_CONST[k - 1]);
    }
} // namespace

SystemODESolution bdf(SystemODE f, std::vector<double> y0,
                       double t0, double t1,
                       size_t order, double tol, double h0)
{
    // Default numerical Jacobian
    auto J = [f](double t, const std::vector<double>& y) {
        return bdf_jacobian(f, t, y);
    };
    return bdf(f, J, y0, t0, t1, order, tol, h0);
}

SystemODESolution bdf(SystemODE f, JacobianODE J, std::vector<double> y0,
                       double t0, double t1,
                       size_t order, double tol, double h0)
{
    if (order < 1 || order > 6)
        throw std::invalid_argument("bdf: order must be in [1, 6]");

    SystemODESolution sol;
    size_t n = y0.size();
    double t = t0;
    double h = h0;
    const double hmin = 1e-12, hmax = (t1 - t0) * 0.1;
    size_t k = order;

    sol.t.push_back(t);
    sol.y.push_back(y0);

    // Starting procedure: generate k-1 additional values with RK4
    auto history = bdf_start(f, y0, t0, h, k);
    double t_start = t0 + (k - 1) * h;
    for (int i = static_cast<int>(k) - 2; i >= 0; --i) {
        sol.t.push_back(t0 + (i + 1) * h);
        sol.y.push_back(history[i]);
    }
    t = t_start;

    // Main loop
    size_t max_steps = static_cast<size_t>((t1 - t0) / hmin) + 10000;
    size_t step_count = 0;
    while (t < t1 - 1e-12 && step_count < max_steps) {
        ++step_count;
        if (t + h > t1) h = t1 - t;

        double t_new = t + h;

        // BDF step
        auto y_new = bdf_step(f, J, history, t_new, h, k);

        if (y_new.empty()) {
            // Newton didn't converge — reduce step and retry
            h = std::max(h * 0.5, hmin);
            if (h <= hmin) {
                // Force step with current h
                y_new = history[0];
                auto fy = f(t_new, y_new);
                for (size_t i = 0; i < n; ++i)
                    y_new[i] = history[0][i] + h * fy[i];
            } else {
                continue;
            }
        }

        // Error estimation
        std::deque<std::vector<double>> new_hist = history;
        new_hist.push_front(y_new);
        if (new_hist.size() > k + 1) new_hist.pop_back();

        double err = bdf_error_estimate(new_hist, k);
        double scale = tol * (1.0 + vec_norm(y_new));

        if (err <= scale || h <= hmin) {
            // Accept step
            t = t_new;
            history.push_front(y_new);
            if (history.size() > k + 1) history.pop_back();
            sol.t.push_back(t);
            sol.y.push_back(y_new);
        } else {
            // Reject step — reduce h and retry
            h = std::max(h * 0.5, hmin);
            continue;
        }

        // Step-size adjustment
        if (err > 0.0 && err <= scale) {
            double factor = 0.9 * std::pow(scale / err, 1.0 / (k + 1));
            factor = std::clamp(factor, 0.2, 5.0);
            h = std::clamp(h * factor, hmin, hmax);
        } else if (err > scale) {
            h = std::max(h * 0.5, hmin);
        }
        // if err == 0 (e.g. k=1), keep h unchanged
    }

    return sol;
}

// ── Adams-Moulton methods ───────────────────────────────────────────────────

namespace {
    // Adams-Moulton beta coefficients for orders 1..5
    // AM-k: y_{n+1} = y_n + h * sum_{j=0}^{s} beta_j * f_{n+1-j}
    // where s = order - 1, and beta_0 is the coefficient of f(t_{n+1}, y_{n+1})
    struct AMCoeffs {
        double beta[6]; // beta[0..s], where s = order-1
    };

    constexpr AMCoeffs AM_COEFFS[5] = {
        // AM1 (Backward Euler): y_{n+1} = y_n + h*f_{n+1}
        {{  1.0,  0.0,  0.0,  0.0,  0.0,  0.0}},
        // AM2 (Trapezoidal): y_{n+1} = y_n + h/2*(f_{n+1} + f_n)
        {{  0.5,  0.5,  0.0,  0.0,  0.0,  0.0}},
        // AM3: y_{n+1} = y_n + h/12*(5*f_{n+1} + 8*f_n - f_{n-1})
        {{  5.0/12.0,  8.0/12.0, -1.0/12.0,  0.0,  0.0,  0.0}},
        // AM4: y_{n+1} = y_n + h/24*(9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})
        {{  9.0/24.0,  19.0/24.0, -5.0/24.0,  1.0/24.0,  0.0,  0.0}},
        // AM5: y_{n+1} = y_n + h/720*(251*f_{n+1} + 646*f_n - 264*f_{n-1} + 106*f_{n-2} - 19*f_{n-3})
        {{  251.0/720.0,  646.0/720.0, -264.0/720.0,  106.0/720.0, -19.0/720.0,  0.0}}
    };

    // Adams-Bashforth explicit predictor coefficients for orders 1..5
    // AB-k: y_{n+1} = y_n + h * sum_{j=0}^{k-1} beta_j * f_{n-j}
    struct ABCoeffs {
        double beta[5]; // beta[0..k-1]
    };

    constexpr ABCoeffs AB_COEFFS[5] = {
        // AB1 (Forward Euler): y_{n+1} = y_n + h*f_n
        {{  1.0,  0.0,  0.0,  0.0,  0.0}},
        // AB2: y_{n+1} = y_n + h/2*(3*f_n - f_{n-1})
        {{  1.5, -0.5,  0.0,  0.0,  0.0}},
        // AB3: y_{n+1} = y_n + h/12*(23*f_n - 16*f_{n-1} + 5*f_{n-2})
        {{  23.0/12.0, -16.0/12.0,  5.0/12.0,  0.0,  0.0}},
        // AB4: y_{n+1} = y_n + h/24*(55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})
        {{  55.0/24.0, -59.0/24.0,  37.0/24.0, -9.0/24.0,  0.0}},
        // AB5: y_{n+1} = y_n + h/720*(1901*f_n - 2774*f_{n-1} + 2616*f_{n-2} - 1274*f_{n-3} + 251*f_{n-4})
        {{  1901.0/720.0, -2774.0/720.0,  2616.0/720.0, -1274.0/720.0,  251.0/720.0}}
    };

    // One Adams-Moulton step via Newton iteration
    // yhist[0] = initial guess for y_{n+1}, yhist[1] = y_n, yhist[2] = y_{n-1}, ...
    // thist[0] = t_{n+1}, thist[1] = t_n, thist[2] = t_{n-1}, ...
    // F(y_new) = y_new - y_n - h * sum_{j=0}^{s} beta_j * f(t_{n+1-j}, y_{n+1-j}) = 0
    std::vector<double> am_step(
        SystemODE f, JacobianODE J,
        const std::deque<std::vector<double>>& yhist,
        const std::deque<double>& thist,
        double t_new, double h, size_t order)
    {
        size_t n = yhist[0].size();
        size_t s = order - 1;
        const auto& beta = AM_COEFFS[order - 1].beta;

        // Compute sum_{j=1}^{s} beta_j * f(t_{n+1-j}, y_{n+1-j})
        std::vector<double> sum_explicit(n, 0.0);
        for (size_t j = 1; j <= s; ++j)
            for (size_t i = 0; i < n; ++i)
                sum_explicit[i] += beta[j] * f(thist[j], yhist[j])[i];

        // Initial guess: y_n = yhist[1]
        std::vector<double> y_new = yhist[1];

        // Newton iteration: F(y) = y - y_n - h*beta_0*f(t_new, y) - h*sum_explicit = 0
        bool converged = false;
        for (size_t iter = 0; iter < 50; ++iter) {
            auto fy = f(t_new, y_new);
            auto Jac = J(t_new, y_new);

            // F = y_new - y_n - h*beta_0*f(t_new,y_new) - h*sum_explicit
            std::vector<double> F(n);
            for (size_t i = 0; i < n; ++i)
                F[i] = y_new[i] - yhist[1][i] - h * beta[0] * fy[i] - h * sum_explicit[i];

            double res = vec_norm(F);
            if (res < 1e-10) { converged = true; break; }

            // J_sys = I - h*beta_0*Jacobian
            std::vector<std::vector<double>> Jsys(n, std::vector<double>(n));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    Jsys[i][j] = -h * beta[0] * Jac[i][j];
                    if (i == j) Jsys[i][j] += 1.0;
                }
            }

            auto dy = bdf_solve_linear(Jsys, F);
            for (size_t i = 0; i < n; ++i)
                y_new[i] -= dy[i];
        }

        if (!converged) return {};
        return y_new;
    }

    // Adams-Bashforth-Moulton predictor-corrector step
    std::vector<double> abm_step(
        SystemODE f, JacobianODE J,
        const std::deque<std::vector<double>>& yhist,
        const std::deque<double>& thist,
        double t_new, double h, size_t order)
    {
        size_t n = yhist[0].size();
        const auto& ab_beta = AB_COEFFS[order - 1].beta;

        // Predict: y_pred = y_n + h * sum_{j=0}^{k-1} ab_beta_j * f_{n-j}
        std::vector<double> y_pred(n, 0.0);
        for (size_t j = 0; j < order; ++j)
            for (size_t i = 0; i < n; ++i)
                y_pred[i] += ab_beta[j] * f(thist[j], yhist[j])[i];
        for (size_t i = 0; i < n; ++i)
            y_pred[i] = yhist[0][i] + h * y_pred[i];

        // Correct: Newton iteration with AM formula
        std::deque<std::vector<double>> new_yhist = yhist;
        new_yhist.push_front(y_pred);
        std::deque<double> new_thist = thist;
        new_thist.push_front(t_new);

        return am_step(f, J, new_yhist, new_thist, t_new, h, order);
    }
} // namespace

SystemODESolution adams_moulton(SystemODE f, std::vector<double> y0,
                                  double t0, double t1,
                                  size_t order, double tol, double h0)
{
    auto J = [f](double t, const std::vector<double>& y) {
        return bdf_jacobian(f, t, y);
    };
    return adams_moulton(f, J, y0, t0, t1, order, tol, h0);
}

SystemODESolution adams_moulton(SystemODE f, JacobianODE J,
                                  std::vector<double> y0,
                                  double t0, double t1,
                                  size_t order, double tol, double h0)
{
    if (order < 1 || order > 5)
        throw std::invalid_argument("adams_moulton: order must be in [1, 5]");

    SystemODESolution sol;
    size_t n = y0.size();
    double t = t0;
    double h = h0;
    const double hmin = 1e-12, hmax = (t1 - t0) * 0.1;
    size_t s = order - 1; // number of known history points needed

    sol.t.push_back(t);
    sol.y.push_back(y0);

    // Starting procedure: generate s initial values with RK4
    auto yhist = bdf_start(f, y0, t0, h, s + 1);
    std::deque<double> thist;
    // yhist = [y_s, y_{s-1}, ..., y_1, y_0] with times [s*h, (s-1)*h, ..., h, 0]
    // We need thist in same order as yhist
    for (size_t i = 0; i <= s; ++i) {
        double ti = t0 + (s - i) * h;
        thist.push_back(ti);
    }
    // Add starting points to solution (skip y0 which is already there)
    for (int i = static_cast<int>(s) - 1; i >= 0; --i) {
        sol.t.push_back(t0 + (i + 1) * h);
        sol.y.push_back(yhist[i]);
    }
    t = t0 + s * h;

    // Main loop
    size_t max_steps = static_cast<size_t>((t1 - t0) / hmin) + 10000;
    size_t step_count = 0;
    while (t < t1 - 1e-12 && step_count < max_steps) {
        ++step_count;
        if (t + h > t1) h = t1 - t;

        double t_new = t + h;

        // Prepend initial guess y_n for am_step
        std::deque<std::vector<double>> am_yhist = yhist;
        am_yhist.push_front(yhist[0]); // duplicate y_n as initial guess
        std::deque<double> am_thist = thist;
        am_thist.push_front(t_new);

        auto y_new = am_step(f, J, am_yhist, am_thist, t_new, h, order);

        if (y_new.empty()) {
            h = std::max(h * 0.5, hmin);
            if (h <= hmin) {
                y_new = yhist[0];
                auto fy = f(t_new, y_new);
                for (size_t i = 0; i < n; ++i)
                    y_new[i] = yhist[0][i] + h * fy[i];
            } else {
                continue;
            }
        }

        // Error estimate from predictor-corrector difference
        double err = 0.0;
        if (order >= 2) {
            const auto& ab_beta = AB_COEFFS[order - 1].beta;
            std::vector<double> y_pred(n, 0.0);
            for (size_t j = 0; j < order; ++j)
                for (size_t i = 0; i < n; ++i)
                    y_pred[i] += ab_beta[j] * f(thist[j], yhist[j])[i];
            for (size_t i = 0; i < n; ++i)
                y_pred[i] = yhist[0][i] + h * y_pred[i];

            for (size_t i = 0; i < n; ++i) {
                double d = y_new[i] - y_pred[i];
                err += d * d;
            }
            err = std::sqrt(err);
        }

        double scale = tol * (1.0 + vec_norm(y_new));

        if (err <= scale || h <= hmin) {
            t = t_new;
            yhist.push_front(y_new);
            thist.push_front(t_new);
            while (yhist.size() > s + 1) { yhist.pop_back(); thist.pop_back(); }
            sol.t.push_back(t);
            sol.y.push_back(y_new);
        } else {
            h = std::max(h * 0.5, hmin);
            continue;
        }

        // Step-size adjustment
        if (err > 0.0 && err <= scale) {
            double factor = 0.9 * std::pow(scale / err, 1.0 / (order + 1));
            factor = std::clamp(factor, 0.2, 5.0);
            h = std::clamp(h * factor, hmin, hmax);
        }
    }

    return sol;
}

SystemODESolution abm(SystemODE f, std::vector<double> y0,
                       double t0, double t1,
                       size_t order, double tol, double h0)
{
    auto J = [f](double t, const std::vector<double>& y) {
        return bdf_jacobian(f, t, y);
    };

    if (order < 1 || order > 5)
        throw std::invalid_argument("abm: order must be in [1, 5]");

    SystemODESolution sol;
    size_t n = y0.size();
    double t = t0;
    double h = h0;
    const double hmin = 1e-12, hmax = (t1 - t0) * 0.1;
    size_t s = order - 1;

    sol.t.push_back(t);
    sol.y.push_back(y0);

    // Starting procedure
    auto yhist = bdf_start(f, y0, t0, h, s + 1);
    std::deque<double> thist;
    for (size_t i = 0; i <= s; ++i) {
        double ti = t0 + (s - i) * h;
        thist.push_back(ti);
    }
    for (int i = static_cast<int>(s) - 1; i >= 0; --i) {
        sol.t.push_back(t0 + (i + 1) * h);
        sol.y.push_back(yhist[i]);
    }
    t = t0 + s * h;

    // Main loop
    size_t max_steps = static_cast<size_t>((t1 - t0) / hmin) + 10000;
    size_t step_count = 0;
    while (t < t1 - 1e-12 && step_count < max_steps) {
        ++step_count;
        if (t + h > t1) h = t1 - t;

        double t_new = t + h;

        auto y_new = abm_step(f, J, yhist, thist, t_new, h, order);

        if (y_new.empty()) {
            h = std::max(h * 0.5, hmin);
            if (h <= hmin) {
                y_new = yhist[0];
                auto fy = f(t_new, y_new);
                for (size_t i = 0; i < n; ++i)
                    y_new[i] = yhist[0][i] + h * fy[i];
            } else {
                continue;
            }
        }

        // Error estimate: |y_corrected - y_predicted|
        const auto& ab_beta = AB_COEFFS[order - 1].beta;
        std::vector<double> y_pred(n, 0.0);
        for (size_t j = 0; j < order; ++j)
            for (size_t i = 0; i < n; ++i)
                y_pred[i] += ab_beta[j] * f(thist[j], yhist[j])[i];
        for (size_t i = 0; i < n; ++i)
            y_pred[i] = yhist[0][i] + h * y_pred[i];

        double err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double d = y_new[i] - y_pred[i];
            err += d * d;
        }
        err = std::sqrt(err);
        double scale = tol * (1.0 + vec_norm(y_new));

        if (err <= scale || h <= hmin) {
            t = t_new;
            yhist.push_front(y_new);
            thist.push_front(t_new);
            while (yhist.size() > s + 1) { yhist.pop_back(); thist.pop_back(); }
            sol.t.push_back(t);
            sol.y.push_back(y_new);
        } else {
            h = std::max(h * 0.5, hmin);
            continue;
        }

        // Step-size adjustment
        if (err > 0.0 && err <= scale) {
            double factor = 0.9 * std::pow(scale / err, 1.0 / (order + 1));
            factor = std::clamp(factor, 0.2, 5.0);
            h = std::clamp(h * factor, hmin, hmax);
        }
    }

    return sol;
}

} // namespace SharedMath::NumericalMethods
