#include "NumericalMethods/NonlinearEquations.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace SharedMath::NumericalMethods {

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {
    double vec_norm(const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return std::sqrt(s);
    }

    // Solve linear system Ax = b via Gaussian elimination with partial pivoting
    // Returns empty vector if singular (instead of throwing)
    std::vector<double> solve_linear_safe(
        std::vector<std::vector<double>> A,
        std::vector<double> b)
    {
        size_t n = b.size();
        for (size_t i = 0; i < n; ++i) {
            // Partial pivoting
            size_t pivot = i;
            double max_val = std::abs(A[i][i]);
            for (size_t k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > max_val) {
                    max_val = std::abs(A[k][i]);
                    pivot = k;
                }
            }
            if (max_val < 1e-15)
                return {};  // singular
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);

            // Elimination
            for (size_t k = i + 1; k < n; ++k) {
                double factor = A[k][i] / A[i][i];
                for (size_t j = i; j < n; ++j)
                    A[k][j] -= factor * A[i][j];
                b[k] -= factor * b[i];
            }
        }

        // Back substitution
        std::vector<double> x(n);
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            x[i] = b[i];
            for (size_t j = i + 1; j < n; ++j)
                x[i] -= A[i][j] * x[j];
            x[i] /= A[i][i];
        }
        return x;
    }

    // Solve linear system Ax = b (throws on singular)
    std::vector<double> solve_linear(
        std::vector<std::vector<double>> A,
        std::vector<double> b)
    {
        auto result = solve_linear_safe(std::move(A), std::move(b));
        if (result.empty())
            throw std::runtime_error("solve_linear: singular matrix");
        return result;
    }
    std::vector<std::vector<double>> numerical_jacobian(
        std::function<std::vector<double>(const std::vector<double>&)> F,
        const std::vector<double>& x)
    {
        size_t n = x.size();
        auto fx = F(x);
        std::vector<std::vector<double>> J(n, std::vector<double>(n));
        for (size_t j = 0; j < n; ++j) {
            std::vector<double> xh = x;
            double h = 1e-7 * (1.0 + std::abs(x[j]));
            xh[j] += h;
            auto fxh = F(xh);
            for (size_t i = 0; i < n; ++i)
                J[i][j] = (fxh[i] - fx[i]) / h;
        }
        return J;
    }
} // namespace

// ── Newton-Raphson for systems (analytic Jacobian) ──────────────────────────

NLEIterResult newton_raphson_system(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    JacobianFunc J,
    std::vector<double> x0,
    double tol, size_t max_iter)
{
    auto fx = F(x0);
    if (vec_norm(fx) < tol)
        return {x0, 0, vec_norm(fx), true};

    size_t n = x0.size();
    for (size_t iter = 0; iter < max_iter; ++iter) {
        auto Jac = J(x0);
        auto dx = solve_linear(Jac, fx);

        for (size_t i = 0; i < n; ++i)
            x0[i] -= dx[i];

        fx = F(x0);
        double res = vec_norm(fx);
        if (res < tol)
            return {x0, iter + 1, res, true};
    }

    return {x0, max_iter, vec_norm(fx), false};
}

// ── Newton-Raphson for systems (numerical Jacobian) ─────────────────────────

NLEIterResult newton_raphson_system(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    std::vector<double> x0,
    double tol, size_t max_iter)
{
    return newton_raphson_system(F,
        [F](const std::vector<double>& x) { return numerical_jacobian(F, x); },
        x0, tol, max_iter);
}

// ── Broyden's method ───────────────────────────────────────────────────────

namespace {
    // Matrix-vector product: C = A * v
    std::vector<double> matvec(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& v)
    {
        size_t n = v.size();
        std::vector<double> r(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                r[i] += A[i][j] * v[j];
        return r;
    }

    // Outer product: M = u * v^T (rank-1 matrix)
    std::vector<std::vector<double>> outer(
        const std::vector<double>& u,
        const std::vector<double>& v)
    {
        size_t n = u.size();
        std::vector<std::vector<double>> M(n, std::vector<double>(n));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                M[i][j] = u[i] * v[j];
        return M;
    }

    // Scalar dot product
    double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }

    // Matrix addition: C = A + s * M
    void mat_add_scaled(
        std::vector<std::vector<double>>& A,
        double s,
        const std::vector<std::vector<double>>& M)
    {
        size_t n = A.size();
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                A[i][j] += s * M[i][j];
    }
} // namespace

NLEIterResult broyden(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    JacobianFunc J,
    std::vector<double> x0,
    double tol, size_t max_iter)
{
    size_t n = x0.size();

    // Step 1: Initialize — A = J(x0), F_prev = F(x0)
    auto A = J(x0);
    auto fx = F(x0);
    double res = vec_norm(fx);
    if (res < tol)
        return {x0, 0, res, true};

    // Step 2: First iteration — Newton step
    auto dx = solve_linear_safe(A, fx);
    if (dx.empty()) {
        A = numerical_jacobian(F, x0);
        dx = solve_linear(A, fx);
    }
    for (size_t i = 0; i < n; ++i)
        x0[i] -= dx[i];

    auto fx_new = F(x0);
    res = vec_norm(fx_new);
    if (res < tol)
        return {x0, 1, res, true};

    // Step 3: Broyden iterations
    for (size_t iter = 1; iter < max_iter; ++iter) {
        auto f_curr = fx_new;

        // 3.2. Solve for new step
        dx = solve_linear_safe(A, f_curr);
        if (dx.empty()) {
            // Singular matrix — restart with numerical Jacobian
            A = numerical_jacobian(F, x0);
            dx = solve_linear(A, f_curr);
        }

        // 3.5. Check convergence before updating
        double dx_norm = vec_norm(dx);
        if (dx_norm < tol)
            return {x0, iter + 1, vec_norm(f_curr), true};

        // 3.3. Update variables
        std::vector<double> x_new(n);
        for (size_t i = 0; i < n; ++i)
            x_new[i] = x0[i] - dx[i];

        auto fx_new2 = F(x_new);
        double res_new = vec_norm(fx_new2);
        if (res_new < tol)
            return {x_new, iter + 1, res_new, true};

        // 3.4. Broyden update: A += (dF - A*dX) * dX^T / (dX^T * dX)
        std::vector<double> dX(n), dF(n);
        for (size_t i = 0; i < n; ++i) {
            dX[i] = x_new[i] - x0[i];
            dF[i] = fx_new2[i] - f_curr[i];
        }

        double dXt_dX = dot(dX, dX);
        if (dXt_dX > 1e-15) {
            auto AdX = matvec(A, dX);
            std::vector<double> numerator(n);
            for (size_t i = 0; i < n; ++i)
                numerator[i] = dF[i] - AdX[i];

            auto rank1 = outer(numerator, dX);
            mat_add_scaled(A, 1.0 / dXt_dX, rank1);
        } else {
            A = numerical_jacobian(F, x_new);
        }

        x0 = x_new;
        fx_new = fx_new2;
    }

    return {x0, max_iter, vec_norm(fx_new), false};
}

NLEIterResult broyden(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    std::vector<double> x0,
    double tol, size_t max_iter)
{
    return broyden(F,
        [F](const std::vector<double>& x) { return numerical_jacobian(F, x); },
        x0, tol, max_iter);
}

} // namespace SharedMath::NumericalMethods
