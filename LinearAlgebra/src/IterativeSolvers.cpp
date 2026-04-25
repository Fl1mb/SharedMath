#include "IterativeSolvers.h"
#include "DynamicMatrix.h"
#include "MatrixFunctions.h"   // for toDynamic / transposeMatrix (via friend / free functions)

#include <cmath>
#include <stdexcept>

namespace SharedMath::LinearAlgebra {

// ─── helpers local to this TU ─────────────────────────────────────────────────

static DynamicMatrix toDyn(const AbstractMatrix& A) {
    if (const auto* d = dynamic_cast<const DynamicMatrix*>(&A)) return *d;
    return DynamicMatrix(A);
}

// y = A * x  (dense)
static std::vector<double> matvec(const DynamicMatrix& A, const std::vector<double>& x) {
    size_t m = A.rows(), n = A.cols();
    std::vector<double> y(m, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            y[i] += A.get(i, j) * x[j];
    return y;
}

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static double nrm2(const std::vector<double>& v) { return std::sqrt(dot(v, v)); }

// ─── Conjugate Gradient ───────────────────────────────────────────────────────

std::vector<double> cg(const AbstractMatrix& A,
                        const std::vector<double>& b,
                        std::vector<double> x0,
                        double tol,
                        size_t max_iter)
{
    size_t n = b.size();
    if (A.rows() != n || A.cols() != n)
        throw std::invalid_argument("cg: A must be square n×n matching b");

    DynamicMatrix Ad = toDyn(A);
    std::vector<double> x = x0.empty() ? std::vector<double>(n, 0.0) : std::move(x0);

    // r = b - A*x
    std::vector<double> Ax = matvec(Ad, x);
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = b[i] - Ax[i];

    std::vector<double> p = r;
    double rs = dot(r, r);

    for (size_t iter = 0; iter < max_iter && std::sqrt(rs) > tol; ++iter) {
        std::vector<double> Ap = matvec(Ad, p);
        double pAp = dot(p, Ap);
        if (std::abs(pAp) < 1e-14) break;

        double alpha = rs / pAp;
        double rs_new = 0.0;
        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rs_new += r[i] * r[i];
        }
        double beta = rs_new / rs;
        for (size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rs = rs_new;
    }
    return x;
}

// ─── LSQR ─────────────────────────────────────────────────────────────────────
// Paige & Saunders (1982).  Solves min_x ||Ax - b||_2 for any A.

std::vector<double> lsqr(const AbstractMatrix& A,
                          const std::vector<double>& b,
                          double tol,
                          size_t max_iter)
{
    size_t m = A.rows(), n = A.cols();
    if (m != b.size())
        throw std::invalid_argument("lsqr: row count of A must equal size of b");

    DynamicMatrix Ad = toDyn(A);

    // Transpose once — used as A^T
    DynamicMatrix At(n, m);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            At.set(j, i, Ad.get(i, j));

    std::vector<double> x(n, 0.0);

    // Initialise bidiagonalization
    std::vector<double> u = b;   // u ← b − A*x0  (x0 = 0)
    double beta = nrm2(u);
    if (beta < 1e-14) return x;
    for (double& v : u) v /= beta;

    std::vector<double> v = matvec(At, u);
    double alpha = nrm2(v);
    if (alpha < 1e-14) return x;
    for (double& vi : v) vi /= alpha;

    std::vector<double> w = v;
    double phi_bar = beta;
    double rho_bar = alpha;

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // ① Bidiagonalize: β_{k+1} u_{k+1} = A v_k − α_k u_k
        std::vector<double> u_new = matvec(Ad, v);
        for (size_t i = 0; i < m; ++i) u_new[i] -= alpha * u[i];
        double beta_new = nrm2(u_new);

        // ② α_{k+1} v_{k+1} = Aᵀ u_{k+1} − β_{k+1} v_k
        //    (skipped when β_{k+1} = 0 — exact solution found)
        std::vector<double> v_new(n, 0.0);
        double alpha_new = 0.0;
        if (beta_new > 1e-14) {
            for (double& vi : u_new) vi /= beta_new;
            u = u_new;
            v_new = matvec(At, u);
            for (size_t i = 0; i < n; ++i) v_new[i] -= beta_new * v[i];
            alpha_new = nrm2(v_new);
            if (alpha_new > 1e-14)
                for (double& vi : v_new) vi /= alpha_new;
        }

        // ③ Givens rotation to eliminate β_{k+1} from the bidiagonal matrix
        double rho = std::sqrt(rho_bar * rho_bar + beta_new * beta_new);
        if (rho < 1e-14) break;
        double c         = rho_bar / rho;
        double s         = beta_new / rho;
        double phi       = c * phi_bar;
        phi_bar          = s * phi_bar;
        double theta_new = s * alpha_new;
        rho_bar          = -c * alpha_new;

        // ④ Update x and search direction w
        for (size_t i = 0; i < n; ++i) {
            x[i] += (phi / rho) * w[i];
            w[i]  = v_new[i] - (theta_new / rho) * w[i];
        }

        // Convergence: ‖r‖ ≈ |φ̄| → 0, or bidiagonalization terminated exactly
        if (beta_new < 1e-14 || alpha_new < 1e-14 ||
            std::abs(phi_bar) < tol) break;

        v     = v_new;
        alpha = alpha_new;
    }
    return x;
}

} // namespace SharedMath::LinearAlgebra
