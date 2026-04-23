#include "NumericalMethods/Integration.h"
#include <cmath>
#include <stdexcept>

namespace SharedMath::NumericalMethods {

double integrate_rect(std::function<double(double)> f,
                       double a, double b, size_t n)
{
    double h = (b - a) / static_cast<double>(n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i)
        sum += f(a + (static_cast<double>(i) + 0.5) * h);
    return sum * h;
}

double integrate_trap(std::function<double(double)> f,
                       double a, double b, size_t n)
{
    double h = (b - a) / static_cast<double>(n);
    double sum = 0.5 * (f(a) + f(b));
    for (size_t i = 1; i < n; ++i)
        sum += f(a + static_cast<double>(i) * h);
    return sum * h;
}

double integrate_simpson(std::function<double(double)> f,
                          double a, double b, size_t n)
{
    if (n % 2 != 0) ++n;
    double h = (b - a) / static_cast<double>(n);
    double sum = f(a) + f(b);
    for (size_t i = 1; i < n; ++i)
        sum += f(a + static_cast<double>(i) * h) * (i % 2 == 0 ? 2.0 : 4.0);
    return sum * h / 3.0;
}

// Gauss-Legendre nodes and weights on [-1, 1] for orders 1..5
namespace {
    constexpr int GL_MAX = 5;
    const double GL_NODES[GL_MAX][GL_MAX] = {
        {0.0,                   0, 0, 0, 0},
        {-0.5773502691896257,   0.5773502691896257, 0, 0, 0},
        {-0.7745966692414834,   0.0, 0.7745966692414834, 0, 0},
        {-0.8611363115940526,  -0.3399810435848563,
          0.3399810435848563,   0.8611363115940526, 0},
        {-0.9061798459386640,  -0.5384693101056831,  0.0,
          0.5384693101056831,   0.9061798459386640}
    };
    const double GL_WEIGHTS[GL_MAX][GL_MAX] = {
        {2.0,                   0, 0, 0, 0},
        {1.0,                   1.0, 0, 0, 0},
        {0.5555555555555556,    0.8888888888888889, 0.5555555555555556, 0, 0},
        {0.3478548451374538,    0.6521451548625461,
         0.6521451548625461,    0.3478548451374538, 0},
        {0.2369268850561891,    0.4786286704993665,  0.5688888888888889,
         0.4786286704993665,    0.2369268850561891}
    };
} // namespace

double integrate_gauss(std::function<double(double)> f,
                        double a, double b, int order)
{
    if (order < 1 || order > GL_MAX)
        throw std::invalid_argument("Gauss-Legendre order must be in [1, 5]");
    double mid  = 0.5 * (a + b);
    double half = 0.5 * (b - a);
    double sum  = 0.0;
    for (int i = 0; i < order; ++i)
        sum += GL_WEIGHTS[order - 1][i] * f(mid + half * GL_NODES[order - 1][i]);
    return half * sum;
}

// Adaptive Simpson helper
namespace {
    double simpson_step(double a, double b, double fa, double fm, double fb) {
        return (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    }

    double adaptive_impl(std::function<double(double)>& f,
                          double a, double b,
                          double fa, double fb, double fmid,
                          double whole, double tol, size_t depth)
    {
        double mid  = 0.5 * (a + b);
        double lmid = 0.5 * (a + mid);
        double rmid = 0.5 * (mid + b);
        double flmid = f(lmid), frmid = f(rmid);
        double left  = simpson_step(a,   mid, fa,   flmid, fmid);
        double right = simpson_step(mid, b,   fmid, frmid, fb);
        double delta = left + right - whole;
        if (depth == 0 || std::abs(delta) <= 15.0 * tol)
            return left + right + delta / 15.0;
        return adaptive_impl(f, a,   mid, fa,   fmid, flmid, left,  tol * 0.5, depth - 1) +
               adaptive_impl(f, mid, b,   fmid, fb,   frmid, right, tol * 0.5, depth - 1);
    }
} // namespace

double integrate_adaptive(std::function<double(double)> f,
                           double a, double b,
                           double tol, size_t max_depth)
{
    double fa = f(a), fb = f(b), fmid = f(0.5 * (a + b));
    double whole = simpson_step(a, b, fa, fmid, fb);
    return adaptive_impl(f, a, b, fa, fb, fmid, whole, tol, max_depth);
}

double integrate2d(std::function<double(double, double)> f,
                   double ax, double bx,
                   std::function<double(double)> ay,
                   std::function<double(double)> by,
                   size_t nx, size_t ny)
{
    double hx  = (bx - ax) / static_cast<double>(nx);
    double sum = 0.0;
    for (size_t i = 0; i < nx; ++i) {
        double x  = ax + (static_cast<double>(i) + 0.5) * hx;
        double ya = ay(x), yb = by(x);
        double hy = (yb - ya) / static_cast<double>(ny);
        double inner = 0.0;
        for (size_t j = 0; j < ny; ++j)
            inner += f(x, ya + (static_cast<double>(j) + 0.5) * hy);
        sum += inner * hy;
    }
    return sum * hx;
}

} // namespace SharedMath::NumericalMethods
