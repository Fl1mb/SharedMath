#include "NumericalMethods/Differentiation.h"
#include <cmath>
#include <stdexcept>

namespace SharedMath::NumericalMethods {



double derivative(std::function<double(double)> f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

double derivative2(std::function<double(double)> f, double x, double h) {
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

double derivativeN(std::function<double(double)> f, double x, int n, double h) {
    if (n < 0) throw std::invalid_argument("derivative order must be >= 0");
    if (n == 0) return f(x);
    // n-th forward finite difference: Δ^n f(x) / h^n
    // Δ^n f(x) = sum_{k=0}^{n} (-1)^(n-k) * C(n,k) * f(x + k*h)
    double result = 0.0;
    long long binom = 1;
    for (int k = 0; k <= n; ++k) {
        double sign = ((n - k) % 2 == 0) ? 1.0 : -1.0;
        result += sign * static_cast<double>(binom) * f(x + k * h);
        if (k < n)
            binom = binom * (n - k) / (k + 1);
    }
    return result / std::pow(h, n);
}

double partial(std::function<double(const std::vector<double>&)> f,
               const std::vector<double>& x, size_t idx, double h) {
    std::vector<double> xp = x, xm = x;
    xp[idx] += h;
    xm[idx] -= h;
    return (f(xp) - f(xm)) / (2.0 * h);
}

std::vector<double> gradient(std::function<double(const std::vector<double>&)> f,
                              const std::vector<double>& x, double h) {
    std::vector<double> grad(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        grad[i] = partial(f, x, i, h);
    return grad;
}

DynamicMatrix jacobian(
    std::function<std::vector<double>(const std::vector<double>&)> f,
    const std::vector<double>& x, double h)
{
    std::vector<double> f0 = f(x);
    size_t m = f0.size(), n = x.size();
    DynamicMatrix J(m, n);
    for (size_t j = 0; j < n; ++j) {
        std::vector<double> xp = x, xm = x;
        xp[j] += h;
        xm[j] -= h;
        std::vector<double> fp = f(xp), fm = f(xm);
        for (size_t i = 0; i < m; ++i)
            J(i, j) = (fp[i] - fm[i]) / (2.0 * h);
    }
    return J;
}

DynamicMatrix hessian(std::function<double(const std::vector<double>&)> f,
                       const std::vector<double>& x, double h)
{
    size_t n = x.size();
    DynamicMatrix H(n, n);
    double f0 = f(x);
    for (size_t i = 0; i < n; ++i) {
        // Diagonal: standard second-difference
        std::vector<double> xp = x, xm = x;
        xp[i] += h;
        xm[i] -= h;
        H(i, i) = (f(xp) - 2.0 * f0 + f(xm)) / (h * h);

        // Off-diagonal: cross-difference, exploit symmetry
        for (size_t j = i + 1; j < n; ++j) {
            std::vector<double> xpp = x, xpm = x, xmp = x, xmm = x;
            xpp[i] += h; xpp[j] += h;
            xpm[i] += h; xpm[j] -= h;
            xmp[i] -= h; xmp[j] += h;
            xmm[i] -= h; xmm[j] -= h;
            double val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h * h);
            H(i, j) = H(j, i) = val;
        }
    }
    return H;
}

} // namespace SharedMath::NumericalMethods
