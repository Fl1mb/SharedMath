#pragma once

#include <functional>
#include <vector>
#include "LinearAlgebra/DynamicMatrix.h"

namespace SharedMath::NumericalMethods {

using SharedMath::LinearAlgebra::DynamicMatrix;

// First derivative via central difference O(h²): df/dx at x
double derivative(std::function<double(double)> f, double x, double h = 1e-5);

// Second derivative via central difference O(h²): d²f/dx² at x
double derivative2(std::function<double(double)> f, double x, double h = 1e-5);

// n-th derivative via forward finite differences
double derivativeN(std::function<double(double)> f, double x, int n, double h = 1e-4);

// Partial derivative ∂f/∂x_idx at point x (central difference)
double partial(std::function<double(const std::vector<double>&)> f,
               const std::vector<double>& x, size_t idx, double h = 1e-5);

// Gradient ∇f at point x
std::vector<double> gradient(std::function<double(const std::vector<double>&)> f,
                              const std::vector<double>& x, double h = 1e-5);

// Jacobian J[i][j] = ∂f_i/∂x_j for f: R^n → R^m
DynamicMatrix jacobian(
    std::function<std::vector<double>(const std::vector<double>&)> f,
    const std::vector<double>& x, double h = 1e-5);

// Hessian H[i][j] = ∂²f/∂x_i ∂x_j for f: R^n → R
DynamicMatrix hessian(std::function<double(const std::vector<double>&)> f,
                       const std::vector<double>& x, double h = 1e-4);

} // namespace SharedMath::NumericalMethods
