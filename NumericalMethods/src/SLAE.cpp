#include "NumericalMethods/SLAE.h"
#include <cmath>
#include <stdexcept>

namespace SharedMath::NumericalMethods {

// ── Helpers ──────────────────────────────────────────────────────────────────

namespace {
    // Compute residual norm ||Ax - b||_2
    double residual_norm(const AbstractMatrix& A,
                          const std::vector<double>& x,
                          const std::vector<double>& b)
    {
        size_t n = b.size();
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double row = 0.0;
            for (size_t j = 0; j < n; ++j) row += A.get(i, j) * x[j];
            double d = row - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    // Matrix-vector product
    std::vector<double> matvec(const AbstractMatrix& A,
                                const std::vector<double>& x)
    {
        size_t n = x.size();
        std::vector<double> r(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                r[i] += A.get(i, j) * x[j];
        return r;
    }

    double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }

    double norm2(const std::vector<double>& v) {
        return std::sqrt(dot(v, v));
    }

    std::vector<double> init_x0(const std::vector<double>& x0, size_t n) {
        return x0.empty() ? std::vector<double>(n, 0.0) : x0;
    }
} // namespace

// ── Jacobi ───────────────────────────────────────────────────────────────────

IterResult jacobi(const AbstractMatrix& A,
                   const std::vector<double>& b,
                   std::vector<double> x0,
                   double tol, size_t max_iter)
{
    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);
    std::vector<double> xnew(n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (size_t j = 0; j < n; ++j)
                if (j != i) sigma += A.get(i, j) * x[j];
            double diag = A.get(i, i);
            if (std::abs(diag) < 1e-15)
                throw std::runtime_error("jacobi: zero diagonal at row " + std::to_string(i));
            xnew[i] = (b[i] - sigma) / diag;
        }

        double res = 0.0;
        for (size_t i = 0; i < n; ++i) {
            res = std::max(res, std::abs(xnew[i] - x[i]));
        }
        x = xnew;

        if (res < tol)
            return {x, iter + 1, residual_norm(A, x, b), true};
    }
    return {x, max_iter, residual_norm(A, x, b), false};
}

// ── Gauss-Seidel ─────────────────────────────────────────────────────────────

IterResult gauss_seidel(const AbstractMatrix& A,
                         const std::vector<double>& b,
                         std::vector<double> x0,
                         double tol, size_t max_iter)
{
    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        double max_delta = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (size_t j = 0; j < n; ++j)
                if (j != i) sigma += A.get(i, j) * x[j];
            double diag = A.get(i, i);
            if (std::abs(diag) < 1e-15)
                throw std::runtime_error("gauss_seidel: zero diagonal at row " + std::to_string(i));
            double xnew = (b[i] - sigma) / diag;
            max_delta = std::max(max_delta, std::abs(xnew - x[i]));
            x[i] = xnew;
        }
        if (max_delta < tol)
            return {x, iter + 1, residual_norm(A, x, b), true};
    }
    return {x, max_iter, residual_norm(A, x, b), false};
}

// ── SOR ──────────────────────────────────────────────────────────────────────

IterResult sor(const AbstractMatrix& A,
                const std::vector<double>& b,
                double omega,
                std::vector<double> x0,
                double tol, size_t max_iter)
{
    if (omega <= 0.0 || omega >= 2.0)
        throw std::invalid_argument("sor: omega must be in (0, 2)");

    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        double max_delta = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (size_t j = 0; j < n; ++j)
                if (j != i) sigma += A.get(i, j) * x[j];
            double diag = A.get(i, i);
            if (std::abs(diag) < 1e-15)
                throw std::runtime_error("sor: zero diagonal at row " + std::to_string(i));
            double gs = (b[i] - sigma) / diag;
            double xnew = (1.0 - omega) * x[i] + omega * gs;
            max_delta = std::max(max_delta, std::abs(xnew - x[i]));
            x[i] = xnew;
        }
        if (max_delta < tol)
            return {x, iter + 1, residual_norm(A, x, b), true};
    }
    return {x, max_iter, residual_norm(A, x, b), false};
}

// ── Conjugate Gradient ───────────────────────────────────────────────────────

IterResult conjugate_gradient(const AbstractMatrix& A,
                               const std::vector<double>& b,
                               std::vector<double> x0,
                               double tol, size_t max_iter)
{
    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);

    // r = b - A*x
    auto Ax = matvec(A, x);
    std::vector<double> r(n), p(n);
    for (size_t i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
    p = r;
    double rr = dot(r, r);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (std::sqrt(rr) < tol)
            return {x, iter, std::sqrt(rr), true};

        auto Ap = matvec(A, p);
        double pAp = dot(p, Ap);
        if (std::abs(pAp) < 1e-300)
            return {x, iter, std::sqrt(rr), true};

        double alpha = rr / pAp;
        double rr_new = 0.0;
        std::vector<double> r_new(n);
        for (size_t i = 0; i < n; ++i) {
            x[i]    += alpha * p[i];
            r_new[i] = r[i] - alpha * Ap[i];
            rr_new  += r_new[i] * r_new[i];
        }

        double beta = rr_new / rr;
        for (size_t i = 0; i < n; ++i)
            p[i] = r_new[i] + beta * p[i];

        r  = r_new;
        rr = rr_new;
    }
    return {x, max_iter, std::sqrt(rr), false};
}

// ── GMRES (restarted) ────────────────────────────────────────────────────────

IterResult gmres(const AbstractMatrix& A,
                  const std::vector<double>& b,
                  size_t restart,
                  std::vector<double> x0,
                  double tol, size_t max_iter)
{
    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);
    size_t total_iter = 0;

    while (total_iter < max_iter) {
        // Compute initial residual
        auto Ax = matvec(A, x);
        std::vector<double> r(n);
        for (size_t i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
        double beta = norm2(r);
        if (beta < tol)
            return {x, total_iter, beta, true};

        size_t m = std::min(restart, max_iter - total_iter);

        // Krylov basis  V[0..m], upper Hessenberg H[m+1][m]
        std::vector<std::vector<double>> V(m + 1, std::vector<double>(n, 0.0));
        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));

        for (size_t i = 0; i < n; ++i) V[0][i] = r[i] / beta;

        // Givens rotations: cs, sn
        std::vector<double> cs(m), sn(m), g(m + 1, 0.0);
        g[0] = beta;

        size_t j = 0;
        for (; j < m; ++j) {
            // Arnoldi: w = A * V[j]
            std::vector<double> w = matvec(A, V[j]);

            // Modified Gram-Schmidt
            for (size_t k = 0; k <= j; ++k) {
                H[k][j] = dot(w, V[k]);
                for (size_t i = 0; i < n; ++i)
                    w[i] -= H[k][j] * V[k][i];
            }
            H[j + 1][j] = norm2(w);

            if (H[j + 1][j] > 1e-15) {
                for (size_t i = 0; i < n; ++i)
                    V[j + 1][i] = w[i] / H[j + 1][j];
            }

            // Apply previous Givens rotations to new column
            for (size_t k = 0; k < j; ++k) {
                double tmp =  cs[k] * H[k][j] + sn[k] * H[k + 1][j];
                H[k + 1][j] = -sn[k] * H[k][j] + cs[k] * H[k + 1][j];
                H[k][j]     = tmp;
            }

            // Compute new Givens rotation
            double denom = std::sqrt(H[j][j]*H[j][j] + H[j+1][j]*H[j+1][j]);
            if (denom < 1e-300) {
                cs[j] = 1.0; sn[j] = 0.0;
            } else {
                cs[j] = H[j][j]     / denom;
                sn[j] = H[j + 1][j] / denom;
            }

            H[j][j]     =  cs[j] * H[j][j] + sn[j] * H[j + 1][j];
            H[j + 1][j] = 0.0;
            g[j + 1]    = -sn[j] * g[j];
            g[j]        =  cs[j] * g[j];

            ++total_iter;
            if (std::abs(g[j + 1]) < tol) { ++j; break; }
        }

        // Solve upper triangular system H[0..j-1][0..j-1] * y = g[0..j-1]
        size_t jj = j;
        std::vector<double> y(jj, 0.0);
        for (int k = static_cast<int>(jj) - 1; k >= 0; --k) {
            y[k] = g[k];
            for (size_t l = k + 1; l < jj; ++l)
                y[k] -= H[k][l] * y[l];
            y[k] /= H[k][k];
        }

        // Update solution x += V[0..jj-1] * y
        for (size_t k = 0; k < jj; ++k)
            for (size_t i = 0; i < n; ++i)
                x[i] += y[k] * V[k][i];

        double res = std::abs(g[jj]);
        if (res < tol)
            return {x, total_iter, residual_norm(A, x, b), true};

        if (total_iter >= max_iter) break;
    }
    return {x, total_iter, residual_norm(A, x, b), false};
}

} // namespace SharedMath::NumericalMethods
