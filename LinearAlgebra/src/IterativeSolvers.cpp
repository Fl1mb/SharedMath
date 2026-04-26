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

// ─── GMRES(m) ─────────────────────────────────────────────────────────────────
// Restarted Arnoldi GMRES with Givens rotations on the upper Hessenberg matrix.
// Reference: Saad & Schultz (1986), Saad (2003) "Iterative Methods for Sparse
// Linear Systems", Algorithm 6.9 (GMRES with restart).

std::vector<double> gmres(const AbstractMatrix& A,
                           const std::vector<double>& b,
                           std::vector<double> x0,
                           double tol,
                           size_t max_iter,
                           size_t restart)
{
    if (A.rows() != A.cols())
        throw std::invalid_argument("gmres: A must be square");
    size_t n = b.size();
    if (A.rows() != n)
        throw std::invalid_argument("gmres: A dimensions incompatible with b");

    DynamicMatrix Ad = toDyn(A);
    std::vector<double> x = x0.empty() ? std::vector<double>(n, 0.0) : std::move(x0);

    auto apply = [&](const std::vector<double>& v) { return matvec(Ad, v); };

    size_t m = std::min(restart, n);
    // Q stores the Krylov basis vectors as columns: Q[j] is a vector of size n
    std::vector<std::vector<double>> Q;   // Q[j] = j-th basis vector
    // H is the (m+1) × m upper Hessenberg matrix stored column-major
    std::vector<std::vector<double>> H;

    size_t total_iter = 0;
    while (total_iter < max_iter) {
        // Compute initial residual for this cycle
        std::vector<double> r = b;
        auto Ax = apply(x);
        for (size_t i = 0; i < n; ++i) r[i] -= Ax[i];
        double beta = nrm2(r);
        if (beta < tol) break;

        Q.clear(); H.clear();
        Q.push_back(std::vector<double>(n));
        for (size_t i = 0; i < n; ++i) Q[0][i] = r[i] / beta;

        // Givens rotation coefficients for QR of H
        std::vector<double> cs, sn, g(m + 1, 0.0);
        g[0] = beta;

        size_t j = 0;
        for (; j < m && total_iter < max_iter; ++j, ++total_iter) {
            // Arnoldi: w = A * Q[j]
            std::vector<double> w = apply(Q[j]);

            // Modified Gram-Schmidt orthogonalisation
            H.push_back(std::vector<double>(j + 2, 0.0));  // H[j] has j+2 rows
            for (size_t i = 0; i <= j; ++i) {
                H[j][i] = dot(w, Q[i]);
                for (size_t k = 0; k < n; ++k) w[k] -= H[j][i] * Q[i][k];
            }
            H[j][j + 1] = nrm2(w);

            // New basis vector
            if (H[j][j + 1] > 1e-300) {
                std::vector<double> qnew(n);
                double inv = 1.0 / H[j][j + 1];
                for (size_t k = 0; k < n; ++k) qnew[k] = w[k] * inv;
                Q.push_back(std::move(qnew));
            }

            // Apply previous Givens rotations to new column of H
            for (size_t i = 0; i < j; ++i) {
                double temp =  cs[i] * H[j][i] + sn[i] * H[j][i + 1];
                H[j][i + 1] = -sn[i] * H[j][i] + cs[i] * H[j][i + 1];
                H[j][i]     = temp;
            }

            // New Givens rotation to zero out H[j][j+1]
            double denom = std::sqrt(H[j][j] * H[j][j] + H[j][j + 1] * H[j][j + 1]);
            double cj = H[j][j]     / denom;
            double sj = H[j][j + 1] / denom;
            cs.push_back(cj); sn.push_back(sj);
            H[j][j]     = cj * H[j][j] + sj * H[j][j + 1];
            H[j][j + 1] = 0.0;

            // Update residual norm estimate
            g[j + 1] = -sj * g[j];
            g[j]     =  cj * g[j];

            if (std::abs(g[j + 1]) < tol) { ++j; break; }
        }

        // Solve the (j × j) upper-triangular system H_j * y = g_j
        size_t sz = j;
        std::vector<double> y(sz, 0.0);
        for (size_t i = sz; i-- > 0; ) {
            y[i] = g[i];
            for (size_t k = i + 1; k < sz; ++k) y[i] -= H[k][i] * y[k];
            y[i] /= H[i][i];
        }

        // Update solution: x = x + Q_j * y
        for (size_t i = 0; i < sz; ++i)
            for (size_t k = 0; k < n; ++k)
                x[k] += Q[i][k] * y[i];

        // Check true residual
        auto Ax2 = apply(x);
        double res = 0.0;
        for (size_t k = 0; k < n; ++k) { double d = b[k] - Ax2[k]; res += d*d; }
        if (std::sqrt(res) < tol) break;
    }
    return x;
}

// ─── BiCGSTAB ─────────────────────────────────────────────────────────────────
// Van der Vorst (1992).  Short recurrences, constant memory.

std::vector<double> bicgstab(const AbstractMatrix& A,
                              const std::vector<double>& b,
                              std::vector<double> x0,
                              double tol,
                              size_t max_iter)
{
    if (A.rows() != A.cols())
        throw std::invalid_argument("bicgstab: A must be square");
    size_t n = b.size();
    if (A.rows() != n)
        throw std::invalid_argument("bicgstab: A dimensions incompatible with b");

    DynamicMatrix Ad = toDyn(A);
    std::vector<double> x = x0.empty() ? std::vector<double>(n, 0.0) : std::move(x0);

    // r = b - A*x
    std::vector<double> r(n), r_hat(n);
    {
        auto Ax = matvec(Ad, x);
        for (size_t i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
    }
    r_hat = r;   // arbitrary shadow residual

    double rho = 1.0, alpha = 1.0, omega = 1.0;
    std::vector<double> v(n, 0.0), p(n, 0.0), s(n), t(n);

    double b_norm = nrm2(b);
    if (b_norm < 1e-300) b_norm = 1.0;

    for (size_t it = 0; it < max_iter; ++it) {
        double rho_new = dot(r_hat, r);
        if (std::abs(rho_new) < 1e-300) break;  // breakdown

        double beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta*(p - omega*v)
        for (size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        v = matvec(Ad, p);
        double denom = dot(r_hat, v);
        if (std::abs(denom) < 1e-300) break;
        alpha = rho / denom;

        // s = r - alpha*v
        for (size_t i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

        if (nrm2(s) < tol * b_norm) {
            for (size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
            break;
        }

        t = matvec(Ad, s);
        double tt = dot(t, t);
        omega = (tt > 0.0) ? dot(t, s) / tt : 0.0;

        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i] + omega * s[i];
            r[i]  = s[i] - omega * t[i];
        }

        if (nrm2(r) < tol * b_norm) break;
        if (std::abs(omega) < 1e-300) break;  // stagnation
    }
    return x;
}

// ─── MINRES ───────────────────────────────────────────────────────────────────
// Paige & Saunders (1975).  For symmetric systems (any definiteness).

std::vector<double> minres(const AbstractMatrix& A,
                            const std::vector<double>& b,
                            std::vector<double> x0,
                            double tol,
                            size_t max_iter)
{
    if (A.rows() != A.cols())
        throw std::invalid_argument("minres: A must be square (symmetric)");
    size_t n = b.size();
    if (A.rows() != n)
        throw std::invalid_argument("minres: A dimensions incompatible with b");

    DynamicMatrix Ad = toDyn(A);
    std::vector<double> x = x0.empty() ? std::vector<double>(n, 0.0) : std::move(x0);

    // Lanczos initialisation
    auto Ax0 = matvec(Ad, x);
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = b[i] - Ax0[i];

    double beta1 = nrm2(r);
    if (beta1 < tol) return x;

    std::vector<double> v_old(n, 0.0), v(n), v_new(n);
    for (size_t i = 0; i < n; ++i) v[i] = r[i] / beta1;

    // MINRES direction vectors
    std::vector<double> w(n, 0.0), w_bar(n), w_hat(n);
    for (size_t i = 0; i < n; ++i) w_bar[i] = v[i];

    double phi_bar = beta1;
    double c_old = 1.0, s_old = 0.0;
    double c = 1.0,     s = 0.0;
    double beta = beta1, beta_old = 0.0, alpha = 0.0;
    double eta = beta1;

    for (size_t it = 0; it < max_iter; ++it) {
        // Lanczos: z = A*v
        auto z = matvec(Ad, v);
        alpha = dot(v, z);

        // v_new = z - alpha*v - beta*v_old
        for (size_t i = 0; i < n; ++i)
            v_new[i] = z[i] - alpha * v[i] - beta * v_old[i];

        beta_old = beta;
        beta      = nrm2(v_new);

        // QR factorisation of the tridiagonal (Givens)
        double delta  =  c * alpha - s * beta_old;
        double eps_1  =  s * alpha + c * beta_old;
        double phi    =  std::sqrt(delta * delta + beta * beta);

        if (phi < 1e-300) break;

        double c_new = delta / phi;
        double s_new = beta  / phi;

        // Update w direction
        for (size_t i = 0; i < n; ++i) {
            w_hat[i] = w_bar[i];
            w_bar[i] = (beta > 1e-300) ? v_new[i] / beta : 0.0;
        }
        for (size_t i = 0; i < n; ++i)
            w[i] = (w_hat[i] - eps_1 * w[i]) / phi;

        // Update solution
        double phi_tilde = phi_bar * c_new;
        phi_bar = phi_bar * s_new;

        for (size_t i = 0; i < n; ++i)
            x[i] += phi_tilde * w[i];

        if (std::abs(phi_bar) < tol * beta1) break;

        // Advance Lanczos vectors
        v_old = v;
        v     = v_new;
        if (beta > 1e-300) {
            for (size_t i = 0; i < n; ++i) v[i] /= beta;
        }
        c_old = c; s_old = s;
        c = c_new; s = s_new;
    }
    return x;
}

} // namespace SharedMath::LinearAlgebra
