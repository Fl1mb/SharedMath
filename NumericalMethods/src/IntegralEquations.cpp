#include "NumericalMethods/IntegralEquations.h"
#include "LinearAlgebra/DynamicMatrix.h"
#include "LinearAlgebra/MatrixFunctions.h"
#include <stdexcept>

using SharedMath::LinearAlgebra::DynamicMatrix;
using SharedMath::LinearAlgebra::solve;

namespace SharedMath::NumericalMethods {

// ── Fredholm 2nd kind ─────────────────────────────────────────────────────
//
// Discretise [a,b] into n nodes: x_i = a + i*h, i=0..n-1
// Trapezoidal quadrature weights: w_0 = w_{n-1} = h/2, w_i = h otherwise.
// The equation y_i = f_i + λ * sum_j w_j * K(x_i, x_j) * y_j
// becomes the linear system  (I - λ * B) y = f
// where  B[i][j] = w_j * K(x_i, x_j).

std::pair<std::vector<double>, std::vector<double>>
fredholm2(std::function<double(double)> f,
          std::function<double(double, double)> K,
          double a, double b, double lambda, size_t n)
{
    if (n < 2) throw std::invalid_argument("fredholm2: n must be >= 2");

    double h = (b - a) / static_cast<double>(n - 1);

    // Build nodes and rhs
    std::vector<double> nodes(n), rhs(n);
    for (size_t i = 0; i < n; ++i) {
        nodes[i] = a + static_cast<double>(i) * h;
        rhs[i]   = f(nodes[i]);
    }

    // Trapezoidal weights
    std::vector<double> w(n, h);
    w[0] = w[n - 1] = 0.5 * h;

    // Build system matrix A = I - lambda * B
    DynamicMatrix A(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double Bij = w[j] * K(nodes[i], nodes[j]);
            A(i, j) = (i == j ? 1.0 : 0.0) - lambda * Bij;
        }
    }

    std::vector<double> sol = solve(A, rhs);
    return {nodes, sol};
}

// ── Volterra 2nd kind ─────────────────────────────────────────────────────
//
// Step from x_0 = a to x_{n-1} = b with step h = (b-a)/(n-1).
// At each step i:
//   y_i - 0.5*lambda*h*K(x_i,x_i)*y_i = f(x_i) + lambda*h*[0.5*K(x_i,x_0)*y_0
//                                           + sum_{j=1}^{i-1} K(x_i,x_j)*y_j]
// (trapezoidal rule, upper limit = x_i)

std::pair<std::vector<double>, std::vector<double>>
volterra2(std::function<double(double)> f,
          std::function<double(double, double)> K,
          double a, double b, double lambda, size_t n)
{
    if (n < 2) throw std::invalid_argument("volterra2: n must be >= 2");

    double h = (b - a) / static_cast<double>(n - 1);

    std::vector<double> nodes(n), y(n);
    for (size_t i = 0; i < n; ++i)
        nodes[i] = a + static_cast<double>(i) * h;

    // Initial value: y(x_0) = f(x_0)  (integral is zero at lower limit)
    y[0] = f(nodes[0]);

    for (size_t i = 1; i < n; ++i) {
        double xi = nodes[i];

        // Accumulated sum of interior points j = 1..i-1
        double interior = 0.0;
        for (size_t j = 1; j < i; ++j)
            interior += K(xi, nodes[j]) * y[j];

        double rhs = f(xi) + lambda * h * (0.5 * K(xi, nodes[0]) * y[0] + interior);
        double diag = 1.0 - 0.5 * lambda * h * K(xi, xi);

        if (std::abs(diag) < 1e-15)
            throw std::runtime_error("volterra2: near-singular diagonal at step " +
                                     std::to_string(i));
        y[i] = rhs / diag;
    }

    return {nodes, y};
}

} // namespace SharedMath::NumericalMethods
