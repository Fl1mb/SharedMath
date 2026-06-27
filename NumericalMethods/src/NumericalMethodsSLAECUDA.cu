// NumericalMethodsSLAECUDA.cu — GPU-accelerated iterative solvers.
// CG, GMRES, Jacobi with matvec on GPU via GPUNumMat::gemv.

#ifdef SHAREDMATH_CUDA

#include "NumericalMethods/SLAE.h"
#include "NumericalMethods/NumericalMethodsGPU.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace SharedMath::NumericalMethods {

// ── Helpers ────────────────────────────────────────────────────────────────

namespace {
    // Flatten AbstractMatrix to row-major GPUNumMat
    GPUNumMat matrix_to_gpu(const AbstractMatrix& A) {
        size_t n = A.rows();
        std::vector<double> flat(n * n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                flat[i * n + j] = A.get(i, j);
        return GPUNumMat(n, n, flat);
    }

    double residual_norm_host(const AbstractMatrix& A,
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

    std::vector<double> init_x0(const std::vector<double>& x0, size_t n) {
        return x0.empty() ? std::vector<double>(n, 0.0) : x0;
    }
} // anonymous

// ── GPU Jacobi ─────────────────────────────────────────────────────────────

IterResult jacobi_cuda(const AbstractMatrix& A,
                        const std::vector<double>& b,
                        std::vector<double> x0,
                        double tol, size_t max_iter)
{
    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);

    // Extract diagonal and matrix on host
    std::vector<double> diag(n), row_data(n * n);
    for (size_t i = 0; i < n; ++i) {
        diag[i] = A.get(i, i);
        if (std::abs(diag[i]) < 1e-15)
            throw std::runtime_error("jacobi_cuda: zero diagonal at row " + std::to_string(i));
        for (size_t j = 0; j < n; ++j)
            row_data[i * n + j] = A.get(i, j);
    }

    GPUNumMat A_temp(n, n, row_data);
    GPUNumMat A_gpu = A_temp.toGPU();
    GPUNumVec b_temp(b);
    GPUNumVec b_gpu = b_temp.toGPU();
    GPUNumVec x_temp2(x);
    GPUNumVec x_gpu = x_temp2.toGPU();
    std::vector<double> xnew(n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // Compute A*x on GPU, bring back to host
        auto Ax = A_gpu.gemv(x_gpu).toCPU();
        const auto& Ax_h = Ax.host();

        // Jacobi update on CPU (per-element, needs diagonal)
        for (size_t i = 0; i < n; ++i)
            xnew[i] = (b[i] - (Ax_h[i] - diag[i] * x[i])) / diag[i];

        double res = 0.0;
        for (size_t i = 0; i < n; ++i)
            res = std::max(res, std::abs(xnew[i] - x[i]));
        x = xnew;
        GPUNumVec x_temp3(x);
        x_gpu = x_temp3.toGPU();

        if (res < tol)
            return {x, iter + 1, residual_norm_host(A, x, b), true};
    }
    return {x, max_iter, residual_norm_host(A, x, b), false};
}

// ── GPU Conjugate Gradient ─────────────────────────────────────────────────

IterResult conjugate_gradient_cuda(const AbstractMatrix& A,
                                    const std::vector<double>& b,
                                    std::vector<double> x0,
                                    double tol, size_t max_iter)
{
    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);

    GPUNumMat A_temp2 = matrix_to_gpu(A);
    GPUNumMat A_gpu = A_temp2.toGPU();
    GPUNumVec x_temp4(x);
    GPUNumVec x_gpu = x_temp4.toGPU();
    GPUNumVec b_temp2(b);
    GPUNumVec b_gpu = b_temp2.toGPU();

    // r = b - A*x
    auto Ax = A_gpu.gemv(x_gpu);
    auto r = GPUNumVec::axpy(-1.0, Ax, b_gpu);  // r = -Ax + b = b - Ax
    auto p = r;
    double rr = GPUNumVec::dot(r, r);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        if (std::sqrt(rr) < tol)
            return {x_gpu.toCPU().host(), iter, std::sqrt(rr), true};

        auto Ap = A_gpu.gemv(p);
        double pAp = GPUNumVec::dot(p, Ap);
        if (std::abs(pAp) < 1e-300)
            return {x_gpu.toCPU().host(), iter, std::sqrt(rr), true};

        double alpha = rr / pAp;

        // x = x + alpha * p
        x_gpu = GPUNumVec::axpy(alpha, p, x_gpu);
        // r = r - alpha * Ap
        r = GPUNumVec::axpy(-alpha, Ap, r);

        double rr_new = GPUNumVec::dot(r, r);
        double beta = rr_new / rr;
        // p = r + beta * p
        p = GPUNumVec::axpy(beta, p, r);

        rr = rr_new;
    }
    return {x_gpu.toCPU().host(), max_iter, std::sqrt(rr), false};
}

// ── GPU GMRES (restarted) ──────────────────────────────────────────────────

IterResult gmres_cuda(const AbstractMatrix& A,
                       const std::vector<double>& b,
                       size_t restart,
                       std::vector<double> x0,
                       double tol, size_t max_iter)
{
    size_t n = b.size();
    std::vector<double> x = init_x0(x0, n);
    size_t total_iter = 0;

    GPUNumMat A_temp3 = matrix_to_gpu(A);
    GPUNumMat A_gpu = A_temp3.toGPU();
    GPUNumVec b_temp3(b);
    GPUNumVec b_gpu = b_temp3.toGPU();

    while (total_iter < max_iter) {
        GPUNumVec x_temp(x);
        GPUNumVec x_gpu = x_temp.toGPU();
        auto Ax = A_gpu.gemv(x_gpu);
        auto r = GPUNumVec::axpy(-1.0, Ax, b_gpu);
        double beta = GPUNumVec::nrm2(r);
        if (beta < tol)
            return {x, total_iter, beta, true};

        size_t m = std::min(restart, max_iter - total_iter);

        // Krylov basis on GPU
        std::vector<GPUNumVec> V_gpu(m + 1);
        V_gpu[0] = GPUNumVec::scale(1.0 / beta, r);

        std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
        std::vector<double> cs(m), sn(m), g(m + 1, 0.0);
        g[0] = beta;

        size_t j = 0;
        for (; j < m; ++j) {
            // Arnoldi: w = A * V[j]
            auto w = A_gpu.gemv(V_gpu[j]);

            // Modified Gram-Schmidt
            for (size_t k = 0; k <= j; ++k) {
                H[k][j] = GPUNumVec::dot(w, V_gpu[k]);
                w = GPUNumVec::axpy(-H[k][j], V_gpu[k], w);
            }
            H[j + 1][j] = GPUNumVec::nrm2(w);

            if (H[j + 1][j] > 1e-15)
                V_gpu[j + 1] = GPUNumVec::scale(1.0 / H[j + 1][j], w);

            // Apply previous Givens rotations
            for (size_t k = 0; k < j; ++k) {
                double tmp =  cs[k] * H[k][j] + sn[k] * H[k + 1][j];
                H[k + 1][j] = -sn[k] * H[k][j] + cs[k] * H[k + 1][j];
                H[k][j]     = tmp;
            }

            // Compute new Givens rotation
            double denom = std::sqrt(H[j][j]*H[j][j] + H[j+1][j]*H[j+1][j]);
            if (denom < 1e-300) { cs[j] = 1.0; sn[j] = 0.0; }
            else { cs[j] = H[j][j] / denom; sn[j] = H[j+1][j] / denom; }

            H[j][j]     =  cs[j] * H[j][j] + sn[j] * H[j+1][j];
            H[j+1][j]   = 0.0;
            g[j + 1]    = -sn[j] * g[j];
            g[j]        =  cs[j] * g[j];

            ++total_iter;
            if (std::abs(g[j + 1]) < tol) { ++j; break; }
        }

        // Solve upper triangular H * y = g
        size_t jj = j;
        std::vector<double> y(jj, 0.0);
        for (int k = static_cast<int>(jj) - 1; k >= 0; --k) {
            y[k] = g[k];
            for (size_t l = k + 1; l < jj; ++l)
                y[k] -= H[k][l] * y[l];
            y[k] /= H[k][k];
        }

        // Update x += V * y  (on GPU)
        GPUNumVec update(n);
        update = update.toGPU();
        for (size_t k = 0; k < jj; ++k) {
            auto v_scaled = GPUNumVec::scale(y[k], V_gpu[k]);
            update = GPUNumVec::add(update, v_scaled);
        }
        x_gpu = GPUNumVec::add(x_gpu, update);
        x = x_gpu.toCPU().host();

        double res = std::abs(g[jj]);
        if (res < tol)
            return {x, total_iter, residual_norm_host(A, x, b), true};

        if (total_iter >= max_iter) break;
    }
    return {x, total_iter, residual_norm_host(A, x, b), false};
}

} // namespace SharedMath::NumericalMethods

#endif // SHAREDMATH_CUDA
