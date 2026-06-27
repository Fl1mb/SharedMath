// NumericalMethodsNLECUDA.cu — GPU-accelerated Broyden's method.
// matvec + rank-1 update on GPU via GPUNumMat, F callback on CPU.

#ifdef SHAREDMATH_CUDA

#include "NumericalMethods/NonlinearEquations.h"
#include "NumericalMethods/NumericalMethodsGPU.h"

#include <cmath>
#include <stdexcept>

namespace SharedMath::NumericalMethods {

namespace {
    double vec_norm(const std::vector<double>& v) {
        double s = 0.0;
        for (double x : v) s += x * x;
        return std::sqrt(s);
    }

    // LU solve on CPU (for fallback / first Newton step)
    std::vector<double> lu_solve(std::vector<std::vector<double>> A,
                                  std::vector<double> b)
    {
        size_t n = b.size();
        for (size_t i = 0; i < n; ++i) {
            size_t pivot = i;
            double max_val = std::abs(A[i][i]);
            for (size_t k = i + 1; k < n; ++k)
                if (std::abs(A[k][i]) > max_val) {
                    max_val = std::abs(A[k][i]);
                    pivot = k;
                }
            if (max_val < 1e-15) return {};
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);
            for (size_t k = i + 1; k < n; ++k) {
                double f = A[k][i] / A[i][i];
                for (size_t j = i; j < n; ++j) A[k][j] -= f * A[i][j];
                b[k] -= f * b[i];
            }
        }
        for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
            for (size_t j = i + 1; j < n; ++j) b[i] -= A[i][j] * b[j];
            b[i] /= A[i][i];
        }
        return b;
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

    // Flatten std::vector<std::vector<double>> to row-major std::vector<double>
    std::vector<double> flatten(const std::vector<std::vector<double>>& M) {
        size_t n = M.size();
        std::vector<double> flat(n * n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                flat[i * n + j] = M[i][j];
        return flat;
    }
} // anonymous

NLEIterResult broyden_cuda(
    std::function<std::vector<double>(const std::vector<double>&)> F,
    std::vector<double> x0,
    double tol, size_t max_iter)
{
    size_t n = x0.size();

    // Step 1: Initialize — A = J(x0), F_prev = F(x0)
    auto A_host = numerical_jacobian(F, x0);
    auto fx = F(x0);
    double res = vec_norm(fx);
    if (res < tol)
        return {x0, 0, res, true};

    // Transfer A to GPU
    GPUNumMat A_temp(n, n, flatten(A_host));
    GPUNumMat A_gpu = A_temp.toGPU();

    // Step 2: First iteration — Newton step on GPU
    GPUNumVec fx_temp(fx);
    GPUNumVec fx_gpu = fx_temp.toGPU();
    auto dx_gpu = A_gpu.solve(fx_gpu);
    auto dx = dx_gpu.toCPU().host();
    for (size_t i = 0; i < n; ++i) x0[i] -= dx[i];

    auto fx_new = F(x0);
    res = vec_norm(fx_new);
    if (res < tol)
        return {x0, 1, res, true};

    // Step 3: Broyden iterations
    for (size_t iter = 1; iter < max_iter; ++iter) {
        auto f_curr = fx_new;

        // Solve on GPU: A * dx = f_curr
        GPUNumVec f_temp(f_curr);
        GPUNumVec f_gpu = f_temp.toGPU();
        dx_gpu = A_gpu.solve(f_gpu);
        dx = dx_gpu.toCPU().host();

        double dx_norm = vec_norm(dx);
        if (dx_norm < tol)
            return {x0, iter + 1, vec_norm(f_curr), true};

        // Update x
        std::vector<double> x_new(n);
        for (size_t i = 0; i < n; ++i)
            x_new[i] = x0[i] - dx[i];

        auto fx_new2 = F(x_new);
        double res_new = vec_norm(fx_new2);
        if (res_new < tol)
            return {x_new, iter + 1, res_new, true };

        // Broyden update on GPU: A += (dF - A*dX) * dX^T / (dX^T * dX)
        std::vector<double> dX(n), dF(n);
        for (size_t i = 0; i < n; ++i) {
            dX[i] = x_new[i] - x0[i];
            dF[i] = fx_new2[i] - f_curr[i];
        }

        GPUNumVec dX_temp(dX);
        GPUNumVec dX_gpu = dX_temp.toGPU();
        GPUNumVec dF_temp(dF);
        GPUNumVec dF_gpu = dF_temp.toGPU();

        // A*dX on GPU
        auto AdX = A_gpu.gemv(dX_gpu);
        // numerator = dF - A*dX
        auto num = GPUNumVec::axpy(-1.0, AdX, dF_gpu);
        // dX^T * dX
        double dXdX = GPUNumVec::dot(dX_gpu, dX_gpu);

        if (dXdX > 1e-15) {
            // A += (1/dXdX) * num * dX^T  (rank-1 update on GPU)
            A_gpu.ger(1.0 / dXdX, num, dX_gpu);
        } else {
            // Restart with numerical Jacobian
            A_host = numerical_jacobian(F, x_new);
            GPUNumMat A_restart(n, n, flatten(A_host));
            A_gpu = A_restart.toGPU();
        }

        x0 = x_new;
        fx_new = fx_new2;
    }

    return {x0, max_iter, vec_norm(fx_new), false};
}

} // namespace SharedMath::NumericalMethods

#endif // SHAREDMATH_CUDA
