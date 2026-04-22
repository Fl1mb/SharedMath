#pragma once

// SharedMath::Functions — Special Mathematical Functions
//
// Categories:
//   • Gamma family      — Γ, lnΓ, ψ (digamma), ψ' (trigamma), B, I_x(a,b), P(a,x)
//   • Error functions   — erf, erfc, erfinv
//   • Bessel (1st kind) — J₀, J₁, Jₙ
//   • Bessel (2nd kind) — Y₀, Y₁, Yₙ
//   • Modified Bessel   — I₀, I₁, Iₙ, K₀, K₁, Kₙ
//   • Orthogonal polynomials — Legendre Pₙ, Pₙᵐ;
//                              Chebyshev Tₙ, Uₙ;
//                              Hermite Hₙ, Heₙ; Laguerre Lₙ
//   • Elliptic integrals — K(k), E(k)
//   • Other             — sinc, Lambert W, Riemann ζ
//
// All implementations are header-only (inline).
// Algorithms: Abramowitz & Stegun, Numerical Recipes, DLMF.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SharedMath::Functions {

namespace detail {
    static constexpr double SF_PI    = 3.14159265358979323846;
    static constexpr double SF_SQRT2 = 1.41421356237309504880;
    static constexpr double SF_LN2   = 0.69314718055994530942;
}

// ═════════════════════════════════════════════════════════════════════════════
// GAMMA FAMILY
// ═════════════════════════════════════════════════════════════════════════════

// Γ(x) and ln|Γ(x)| are available as std::tgamma / std::lgamma from <cmath>.
// We do not re-declare them here to avoid clashes with POSIX identifiers
// (gamma, lgamma) that some platforms expose at global scope.

// ── Digamma (Psi) function: ψ(x) = d/dx ln Γ(x) ────────────────────────────
//
// Uses:
//   • Reflection formula  ψ(x) = ψ(1−x) − π cot(πx)      for x < 0.5
//   • Recurrence          ψ(x+1) = ψ(x) + 1/x             to bring x ≥ 6
//   • Asymptotic series   ψ(x) ~ ln x − Σ B_{2k}/(2k x^{2k})
inline double digamma(double x) {
    if (x <= 0.0 && x == std::floor(x))
        throw std::domain_error("digamma: argument is a non-positive integer");

    // Reflection for x < 0.5
    if (x < 0.5)
        return digamma(1.0 - x) - detail::SF_PI / std::tan(detail::SF_PI * x);

    // Recurse to x >= 6 for asymptotic validity
    double result = 0.0;
    double xx = x;
    while (xx < 6.0) { result -= 1.0 / xx; xx += 1.0; }

    // Asymptotic expansion (Stirling): ln(x) - 1/(2x) - Σ B_{2k}/(2k x^{2k})
    double z = 1.0 / (xx * xx);
    result += std::log(xx) - 0.5 / xx
           - z * (1.0/12.0 - z * (1.0/120.0 - z * (1.0/252.0
           - z * (1.0/240.0 - z * (1.0/132.0)))));
    return result;
}

// ── Trigamma function: ψ'(x) = d/dx ψ(x) = d²/dx² ln Γ(x) ─────────────────
//
// Reflection: ψ'(x) + ψ'(1−x) = π² / sin²(πx)
// Asymptotic:  ψ'(x) ~ 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + …
inline double trigamma(double x) {
    if (x <= 0.0 && x == std::floor(x))
        throw std::domain_error("trigamma: argument is a non-positive integer");

    if (x < 0.0) {
        double s = std::sin(detail::SF_PI * x);
        return detail::SF_PI * detail::SF_PI / (s * s) - trigamma(1.0 - x);
    }

    // Recurse to x >= 6
    double result = 0.0;
    double xx = x;
    while (xx < 6.0) { result += 1.0 / (xx * xx); xx += 1.0; }

    // Asymptotic: 1/x + 1/(2x²) + Σ B_{2k}/x^{2k+1}
    double z = 1.0 / (xx * xx);
    result += (1.0 / xx) * (1.0 + 0.5 / xx + z * (1.0/6.0
           - z * (1.0/30.0 - z * (1.0/42.0 - z / 30.0))));
    return result;
}

// ── Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b) ────────────────────────────────
inline double beta(double a, double b) {
    if (a <= 0.0 || b <= 0.0)
        throw std::domain_error("beta: arguments must be positive");
    return std::exp(std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b));
}

// Natural log of B(a, b)
inline double lbeta(double a, double b) {
    if (a <= 0.0 || b <= 0.0)
        throw std::domain_error("lbeta: arguments must be positive");
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

// ── Regularized incomplete beta I_x(a, b) ∈ [0, 1] ──────────────────────────
// Computed via continued fraction (Lentz algorithm, Numerical Recipes §6.4).
namespace detail {
inline double betacf(double a, double b, double x) {
    constexpr int    MAXIT = 300;
    constexpr double EPS   = std::numeric_limits<double>::epsilon();
    const     double FPMIN = std::numeric_limits<double>::min() / EPS;

    double qab = a + b, qap = a + 1.0, qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::abs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    double h = d;
    for (int m = 1; m <= MAXIT; ++m) {
        int m2 = 2 * m;
        // Even step
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d; h *= d * c;
        // Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d; if (std::abs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c; h *= del;
        if (std::abs(del - 1.0) < EPS) break;
    }
    return h;
}
} // namespace detail

// Regularized incomplete beta: I_x(a, b) = B_x(a,b) / B(a,b)
inline double betainc(double x, double a, double b) {
    if (x < 0.0 || x > 1.0) throw std::domain_error("betainc: x must be in [0,1]");
    if (a <= 0.0 || b <= 0.0) throw std::domain_error("betainc: a,b must be positive");
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;
    double bt = std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                       + a * std::log(x) + b * std::log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * detail::betacf(a, b, x) / a;
    return 1.0 - bt * detail::betacf(b, a, 1.0 - x) / b;
}

// ── Regularized lower incomplete gamma P(a, x) ───────────────────────────────
// Series expansion for x < a+1; continued fraction for x ≥ a+1.
namespace detail {
inline double gammap_series(double a, double x) {
    double ap = a, del = 1.0 / a, sum = del;
    for (int n = 1; n < 300; ++n) {
        del *= x / ++ap; sum += del;
        if (std::abs(del) < std::abs(sum) * 1e-15) break;
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}
inline double gammaq_cf(double a, double x) {
    constexpr double EPS   = 1e-15;
    const     double FPMIN = std::numeric_limits<double>::min() / EPS;
    double b = x + 1.0 - a, c = 1.0 / FPMIN, d = 1.0 / b, h = d;
    for (int i = 1; i < 300; ++i) {
        double an = -static_cast<double>(i) * (i - a);
        b += 2.0;
        d = an * d + b; if (std::abs(d) < FPMIN) d = FPMIN;
        c = b + an / c; if (std::abs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        double del = d * c; h *= del;
        if (std::abs(del - 1.0) < EPS) break;
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}
} // namespace detail

// P(a, x) = γ(a,x)/Γ(a)  ∈ [0,1].  Returns probability that a Gamma-distributed
// variable with shape a is ≤ x.
inline double gammainc(double a, double x) {
    if (a <= 0.0) throw std::domain_error("gammainc: a must be positive");
    if (x < 0.0)  throw std::domain_error("gammainc: x must be non-negative");
    if (x == 0.0) return 0.0;
    return (x < a + 1.0) ? detail::gammap_series(a, x)
                          : 1.0 - detail::gammaq_cf(a, x);
}


// ═════════════════════════════════════════════════════════════════════════════
// ERROR FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

// erf and erfc are available as std::erf / std::erfc from <cmath>.
// Not re-declared here to avoid POSIX name collision at global scope.

// ── Inverse error function: erfinv(y) such that erf(erfinv(y)) = y ───────────
// Uses Winitzki's rational approximation, then two Newton refinements.
inline double erfinv(double y) {
    if (y <= -1.0 || y >= 1.0) {
        if (y == -1.0) return -std::numeric_limits<double>::infinity();
        if (y ==  1.0) return  std::numeric_limits<double>::infinity();
        throw std::domain_error("erfinv: argument must be in (-1, 1)");
    }
    if (y == 0.0) return 0.0;

    // Winitzki's piecewise rational approximation (|error| < 4e-9)
    double w = -std::log((1.0 - y) * (1.0 + y));
    double p;
    if (w < 5.0) {
        w -= 2.5;
        p = 2.81022636e-08;   p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06   + p * w; p = -4.39150654e-06 + p * w;
        p = 0.00021858087    + p * w; p = -0.00125372503  + p * w;
        p = -0.00417768164   + p * w; p =  0.246640727    + p * w;
        p =  1.50140941      + p * w;
    } else {
        w = std::sqrt(w) - 3.0;
        p = -0.000200214257; p = 0.000100950558 + p * w;
        p =  0.00134934322  + p * w; p = -0.00367342844 + p * w;
        p =  0.00573950773  + p * w; p = -0.0076224613  + p * w;
        p =  0.00943887047  + p * w; p =  1.00167406    + p * w;
        p =  2.83297682     + p * w;
    }
    p *= y;

    // Two Newton refinements: Δ = (erf(p) − y) / (2/√π · exp(−p²))
    for (int i = 0; i < 2; ++i) {
        double e = std::erf(p) - y;
        p -= e * std::exp(p * p) * (detail::SF_SQRT2 / 2.0)
             * std::sqrt(detail::SF_PI);
    }
    return p;
}


// ═════════════════════════════════════════════════════════════════════════════
// BESSEL FUNCTIONS OF THE FIRST KIND  J₀, J₁, Jₙ
//
// Polynomial approximations from Abramowitz & Stegun / Numerical Recipes.
// ═════════════════════════════════════════════════════════════════════════════

// J₀(x) — maximum error < 1.6e-8
inline double besselJ0(double x) {
    double ax = std::abs(x);
    if (ax < 8.0) {
        double y = x * x;
        return (57568490574.0 + y * (-13362590354.0 + y * (651619640.7
             + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456))))))
             / (57568490411.0 + y * (1029532985.0 + y * (9494680.718
             + y * (59272.64853 + y * (267.8532712 + y)))));
    }
    double z = 8.0 / ax, y = z * z;
    double xx = ax - 0.785398163397448;   // ax − π/4
    double p = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
             + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
    double q = -0.1562499995e-1 + y * (0.1430488765e-3
             + y * (-0.6911147651e-5 + y * (0.7621095161e-6
             - y * 0.934945152e-7)));
    return std::sqrt(0.636619772338 / ax) * (std::cos(xx) * p - z * std::sin(xx) * q);
}

// J₁(x) — maximum error < 1.3e-8
inline double besselJ1(double x) {
    double ax = std::abs(x);
    double ans;
    
    if (ax < 8.0) {
        double y = x * x;  
        double p = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
                + y * (-2972611.439 + y * (15704.48260 + y * (-30.16116360))))));
        double q = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
                + y * (99447.43394 + y * (376.9991397 + y))));
        ans = p / q;
    } else {
        double z = 8.0 / ax;
        double y = z * z;
        double xx = ax - 2.356194491;  // ax - 3π/4
        double p = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
                + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        double q = 0.04687499995 + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6
                + y * 0.105787412e-6)));
        ans = std::sqrt(0.636619772338 / ax) * (std::cos(xx) * p - z * std::sin(xx) * q);

    }
    
    
    
    return ans;  
}

// Jₙ(x) for integer n — forward recurrence from J₀, J₁
inline double besselJn(int n, double x) {
    if (n < 0) return (n % 2 == 0) ? besselJn(-n, x) : -besselJn(-n, x);
    if (n == 0) return besselJ0(x);
    if (n == 1) return besselJ1(x);
    if (x == 0.0) return 0.0;
    
    // Для x >= n прямая рекурсия стабильна
    if (x >= n) {
        double j0 = besselJ0(x);
        double j1 = besselJ1(x);
        for (int k = 1; k < n; ++k) {
            double j_next = (2.0 * k / x) * j1 - j0;
            j0 = j1;
            j1 = j_next;
        }
        return j1;
    }
    
    // Miller's backward recurrence algorithm
    int m = n + 30;  // стартуем достаточно далеко
    if (x < 1e-6) {
        // Для малых x используем степенной ряд
        double term = 1.0;
        double sum = 0.0;
        for (int k = 0; k <= n + 10; ++k) {
            if (k == n) sum = term;
            term *= - (x * x) / (4.0 * (k + 1) * (k + 1));
        }
        return sum * std::pow(0.5 * x, n) / tgamma(n + 1);
    }
    
    std::vector<double> j(m + 2, 0.0);
    j[m] = 1e-30;  // малое начальное значение
    j[m-1] = 0.0;
    
    // Обратная рекурсия
    for (int k = m - 1; k > 0; --k) {
        j[k-1] = (2.0 * k / x) * j[k] - j[k+1];
        if (std::abs(j[k-1]) > 1e30) {
            // Нормализация для предотвращения переполнения
            for (int i = k-1; i <= m; ++i) j[i] *= 1e-30;
        }
    }
    
    // Нормализация: J_0(x) + 2*Σ_{k=1}∞ J_{2k}(x) = 1
    double sum = j[0];
    for (int k = 2; k <= m; k += 2) sum += 2.0 * j[k];
    
    // Нормируем все значения
    for (int k = 0; k <= m; ++k) j[k] /= sum;
    
    return j[n];
}


// ═════════════════════════════════════════════════════════════════════════════
// BESSEL FUNCTIONS OF THE SECOND KIND  Y₀, Y₁, Yₙ
// ═════════════════════════════════════════════════════════════════════════════

// Y₀(x) — maximum error < 1.4e-8; requires x > 0
inline double besselY0(double x) {
    if (x <= 0.0) throw std::domain_error("besselY0: x must be positive");
    if (x < 8.0) {
        double y = x * x;
        double p = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6
                + y * (10879881.29 + y * (-86327.92757 + y * 228.4622733))));
        double q = 40076544269.0 + y * (745249964.8 + y * (7189466.438
                + y * (47447.26752 + y * (226.1030244 + y))));
        return p / q + 0.636619772338 * besselJ0(x) * std::log(x);
    }
    double z = 8.0 / x, y = z * z;
    double xx = x - 0.785398163397448;
    double p = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
             + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
    double q = -0.1562499995e-1 + y * (0.1430488765e-3
             + y * (-0.6911147651e-5 + y * (0.7621095161e-6
             - y * 0.934945152e-7)));
    return std::sqrt(0.636619772338 / x) * (std::sin(xx) * p + z * std::cos(xx) * q);
}

// Y₁(x) — maximum error < 1.9e-8; requires x > 0
inline double besselY1(double x) {
    if (x <= 0.0) throw std::domain_error("besselY1: x must be positive");
    if (x < 8.0) {
        double y = x * x;
        double p = x * (-0.4900604943e13 + y * (0.1275274390e13
                + y * (-0.5153438139e11 + y * (0.7349264551e9
                + y * (-0.4237922726e7 + y * 0.8511937935e4)))));
        double q = 0.2499580570e14 + y * (0.4244419664e12
                + y * (0.3733650367e10 + y * (0.2245904002e8
                + y * (0.1020426050e6 + y * (0.3549632885e3 + y)))));
        return p / q + 0.636619772338 * (besselJ1(x) * std::log(x) - 1.0 / x);
    }
    double z = 8.0 / x, y = z * z;
    double xx = x - 2.356194491;
    double p = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
             + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
    double q = 0.04687499995 + y * (-0.2002690873e-3
             + y * (0.8449199096e-5 + y * (-0.88228987e-6
             + y * 0.105787412e-6)));
    return std::sqrt(0.636619772338 / x) * (std::sin(xx) * p + z * std::cos(xx) * q);
}

// Yₙ(x) — forward recurrence from Y₀, Y₁
inline double besselYn(int n, double x) {
    if (x <= 0.0) throw std::domain_error("besselYn: x must be positive");
    if (n < 0)  return (n % 2 == 0) ? besselYn(-n, x) : -besselYn(-n, x);
    if (n == 0) return besselY0(x);
    if (n == 1) return besselY1(x);
    double y0 = besselY0(x), y1 = besselY1(x);
    for (int k = 1; k < n; ++k) {
        double yk1 = (2.0 * k / x) * y1 - y0;
        y0 = y1; y1 = yk1;
    }
    return y1;
}


// ═════════════════════════════════════════════════════════════════════════════
// MODIFIED BESSEL FUNCTIONS  I₀, I₁, K₀, K₁  (A&S 9.8)
// ═════════════════════════════════════════════════════════════════════════════

// I₀(x) — modified Bessel of the first kind, order 0
inline double besselI0(double x) {
    double ax = std::abs(x);
    if (ax <= 3.75) {
        double t = ax / 3.75; t *= t;
        return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
             + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    }
    double t = 3.75 / ax;
    return (std::exp(ax) / std::sqrt(ax))
         * (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
         + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
         + t * (-0.01647633 + t * 0.00392377))))))));
}

// I₁(x) — modified Bessel of the first kind, order 1
inline double besselI1(double x) {
    double ax = std::abs(x);
    double ans;
    if (ax <= 3.75) {
        double t = ax / 3.75; t *= t;
        ans = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
            + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
    } else {
        double t = 3.75 / ax;
        ans = (std::exp(ax) / std::sqrt(ax))
            * (0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801
            + t * (-0.01031555 + t * (0.02282967 + t * (-0.02895312
            + t * (0.01787654 - t * 0.00420059))))))));
    }
    return (x < 0.0) ? -ans : ans;
}

// K₀(x) — modified Bessel of the second kind, order 0; x > 0
inline double besselK0(double x) {
    if (x <= 0.0) throw std::domain_error("besselK0: x must be positive");
    if (x <= 2.0) {
        double t = x / 2.0; t *= t;
        return -std::log(x / 2.0) * besselI0(x)
             + (-0.57721566 + t * (0.42278420 + t * (0.23069756 + t * (0.03488590
             + t * (0.00262698 + t * (0.00010750 + t * 0.0000074))))));
    }
    double t = 2.0 / x;
    return (std::exp(-x) / std::sqrt(x))
         * (1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446
         + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208))))));
}

// K₁(x) — modified Bessel of the second kind, order 1; x > 0
inline double besselK1(double x) {
    if (x <= 0.0) throw std::domain_error("besselK1: x must be positive");
    if (x <= 2.0) {
        double t = x / 2.0; t *= t;
        return std::log(x / 2.0) * besselI1(x)
             + (1.0 / x) * (1.0 + t * (0.15443144 + t * (-0.67278579 + t * (-0.18156897
             + t * (-0.01919402 + t * (-0.00110404 + t * (-0.00004686)))))));
    }
    double t = 2.0 / x;
    return (std::exp(-x) / std::sqrt(x))
         * (1.25331414 + t * (0.23498619 + t * (-0.03655620 + t * (0.01504268
         + t * (-0.00780353 + t * (0.00325614 - t * 0.00068245))))));
}

// Iₙ(x) — forward recurrence (stable for all x)
inline double besselIn(int n, double x) {
    if (n < 0)  return besselIn(-n, x);
    if (n == 0) return besselI0(x);
    if (n == 1) return besselI1(x);
    if (x == 0.0) return 0.0;
    double i0 = besselI0(x), i1 = besselI1(x);
    for (int k = 1; k < n; ++k) {
        double ik1 = i0 - (2.0 * k / x) * i1;
        i0 = i1; i1 = ik1;
    }
    return i1;
}

// Kₙ(x) — forward recurrence (stable, x > 0)
inline double besselKn(int n, double x) {
    if (x <= 0.0) throw std::domain_error("besselKn: x must be positive");
    if (n < 0)  return besselKn(-n, x);
    if (n == 0) return besselK0(x);
    if (n == 1) return besselK1(x);
    double k0 = besselK0(x), k1 = besselK1(x);
    for (int k = 1; k < n; ++k) {
        double kk1 = k0 + (2.0 * k / x) * k1;
        k0 = k1; k1 = kk1;
    }
    return k1;
}


// ═════════════════════════════════════════════════════════════════════════════
// ORTHOGONAL POLYNOMIALS  (all via 3-term recurrence)
// ═════════════════════════════════════════════════════════════════════════════

// ── Legendre polynomials Pₙ(x),  x ∈ [−1, 1] ────────────────────────────────
// Recurrence: (n+1)P_{n+1}(x) = (2n+1)x P_n(x) − n P_{n-1}(x)
inline double legendreP(int n, double x) {
    if (n < 0) throw std::domain_error("legendreP: n must be non-negative");
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double pm1 = 1.0, p = x;
    for (int k = 1; k < n; ++k) {
        double pp1 = ((2.0*k + 1.0) * x * p - k * pm1) / (k + 1.0);
        pm1 = p; p = pp1;
    }
    return p;
}

// Associated Legendre polynomial Pₙᵐ(x),  0 ≤ m ≤ n
// Uses Schmidt semi-normalised form (no Condon-Shortley phase).
inline double legendreP(int n, int m, double x) {
    if (m < 0 || m > n) throw std::domain_error("legendreP: require 0 <= m <= n");
    if (std::abs(x) > 1.0) throw std::domain_error("legendreP: x must be in [-1,1]");
    // Compute P_m^m first
    double pmm = 1.0;
    double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
    double fact = 1.0;
    for (int i = 1; i <= m; ++i) { pmm *= -fact * somx2; fact += 2.0; }
    if (n == m) return pmm;
    double pmmp1 = x * (2 * m + 1) * pmm;
    if (n == m + 1) return pmmp1;
    double pnm = 0.0;
    for (int ll = m + 2; ll <= n; ++ll) {
        pnm = (x * (2*ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1; pmmp1 = pnm;
    }
    return pnm;
}

// ── Chebyshev polynomials of the first kind Tₙ(x) ──────────────────────────
// Recurrence: T_{n+1}(x) = 2x T_n(x) − T_{n-1}(x)
inline double chebyshevT(int n, double x) {
    if (n < 0)  throw std::domain_error("chebyshevT: n must be non-negative");
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double tm1 = 1.0, t = x;
    for (int k = 1; k < n; ++k) { double tt = 2.0*x*t - tm1; tm1 = t; t = tt; }
    return t;
}

// ── Chebyshev polynomials of the second kind Uₙ(x) ─────────────────────────
// Recurrence: U_{n+1}(x) = 2x U_n(x) − U_{n-1}(x); U₀=1, U₁=2x
inline double chebyshevU(int n, double x) {
    if (n < 0)  throw std::domain_error("chebyshevU: n must be non-negative");
    if (n == 0) return 1.0;
    if (n == 1) return 2.0 * x;
    double um1 = 1.0, u = 2.0 * x;
    for (int k = 1; k < n; ++k) { double uu = 2.0*x*u - um1; um1 = u; u = uu; }
    return u;
}

// ── Physicists' Hermite polynomials Hₙ(x) ───────────────────────────────────
// Recurrence: H_{n+1}(x) = 2x H_n(x) − 2n H_{n-1}(x)
inline double hermiteH(int n, double x) {
    if (n < 0)  throw std::domain_error("hermiteH: n must be non-negative");
    if (n == 0) return 1.0;
    if (n == 1) return 2.0 * x;
    double hm1 = 1.0, h = 2.0 * x;
    for (int k = 1; k < n; ++k) { double hh = 2.0*x*h - 2.0*k*hm1; hm1 = h; h = hh; }
    return h;
}

// ── Probabilists' Hermite polynomials Heₙ(x) ────────────────────────────────
// Recurrence: He_{n+1}(x) = x He_n(x) − n He_{n-1}(x)
inline double hermiteHe(int n, double x) {
    if (n < 0)  throw std::domain_error("hermiteHe: n must be non-negative");
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double hm1 = 1.0, h = x;
    for (int k = 1; k < n; ++k) { double hh = x*h - k*hm1; hm1 = h; h = hh; }
    return h;
}

// ── Laguerre polynomials Lₙ(x) ──────────────────────────────────────────────
// Recurrence: (n+1)L_{n+1}(x) = (2n+1−x)L_n(x) − n L_{n-1}(x)
inline double laguerreL(int n, double x) {
    if (n < 0) throw std::domain_error("laguerreL: n must be non-negative");
    if (n == 0) return 1.0;
    if (n == 1) return 1.0 - x;
    double lm1 = 1.0, l = 1.0 - x;
    for (int k = 1; k < n; ++k) {
        double ll = ((2*k + 1 - x) * l - k * lm1) / (k + 1);
        lm1 = l; l = ll;
    }
    return l;
}

// ── Generalised Laguerre Lₙ^α(x) ────────────────────────────────────────────
// Recurrence: (n+1)L_{n+1}^α = (2n+1+α−x)L_n^α − (n+α)L_{n-1}^α
inline double laguerreL(int n, double alpha, double x) {
    if (n < 0) throw std::domain_error("laguerreL: n must be non-negative");
    if (n == 0) return 1.0;
    if (n == 1) return 1.0 + alpha - x;
    double lm1 = 1.0, l = 1.0 + alpha - x;
    for (int k = 1; k < n; ++k) {
        double ll = ((2*k + 1 + alpha - x) * l - (k + alpha) * lm1) / (k + 1);
        lm1 = l; l = ll;
    }
    return l;
}


// ═════════════════════════════════════════════════════════════════════════════
// COMPLETE ELLIPTIC INTEGRALS  (AGM algorithm — converges quadratically)
// ═════════════════════════════════════════════════════════════════════════════

// K(k) = ∫₀^{π/2} dθ / √(1 − k²sin²θ),   k ∈ [0, 1)
inline double ellipticK(double k) {
    if (k < 0.0 || k >= 1.0)
        throw std::domain_error("ellipticK: k must be in [0, 1)");
    double a = 1.0, b = std::sqrt(1.0 - k * k);
    for (int i = 0; i < 64; ++i) {
        double an = (a + b) * 0.5, bn = std::sqrt(a * b);
        a = an; b = bn;
        if (std::abs(a - b) < std::abs(a) * 1e-15) break;
    }
    return detail::SF_PI / (2.0 * a);
}

// E(k) = ∫₀^{π/2} √(1 − k²sin²θ) dθ,   k ∈ [0, 1]
// Uses the complementary AGM algorithm with accumulated power-of-two sum.
inline double ellipticE(double k) {
    if (k < 0.0 || k > 1.0)
        throw std::domain_error("ellipticE: k must be in [0, 1]");
    if (k == 1.0) return 1.0;
    if (k == 0.0) return detail::SF_PI / 2.0;
    
    double a[64], b[64]; 
    
    a[0] = 1.0;
    b[0] = std::sqrt(1.0 - k * k);
    
    int n = 0;
    for (; n < 63; ++n) {
        a[n+1] = (a[n] + b[n]) * 0.5;
        b[n+1] = std::sqrt(a[n] * b[n]);
        if (std::abs(a[n+1] - b[n+1]) < std::abs(a[n+1]) * 1e-15) break;
    }
    
    double sum = 0.0;
    double factor = 1.0;
    for (int i = 0; i <= n; ++i) {
        // K(k) = π/(2·a_∞)
        // E(k) = K(k) · (1 - Σ 2^{i-1}(a_i² - b_i²))
        double term = factor * (a[i] * a[i] - b[i] * b[i]);
        sum += term;
        factor *= 2.0;
    }
    
    double K = detail::SF_PI / (2.0 * a[n]);  
    double E = K * (1.0 - sum * 0.5);         
    
    return E;
}


// ═════════════════════════════════════════════════════════════════════════════
// OTHER USEFUL FUNCTIONS
// ═════════════════════════════════════════════════════════════════════════════

// ── Normalised sinc: sinc(x) = sin(πx) / (πx),  sinc(0) = 1 ────────────────
inline double sinc(double x) {
    if (x == 0.0) return 1.0;
    double px = detail::SF_PI * x;
    return std::sin(px) / px;
}

// Unnormalised sinc: sincu(x) = sin(x) / x
inline double sincu(double x) {
    if (x == 0.0) return 1.0;
    return std::sin(x) / x;
}

// ── Lambert W function W₀(x) — principal branch, x ≥ −1/e ──────────────────
// Iterative solution using Halley's method (≈ 4 iterations for 1e-15 accuracy).
inline double lambertW(double x) {
    const double e = 2.718281828459045;
    if (x < -1.0 / e)
        throw std::domain_error("lambertW: x must be >= -1/e");
    if (x == 0.0) return 0.0;
    // Initial estimate
    double w;
    if (x <= 0.0) {
        w = x * e / (1.0 + x * e);  // near branch point
    } else if (x < 1.0) {
        w = x * (1.0 - x * (1.0 - x));
    } else {
        w = std::log(x) - std::log(std::log(x));
    }
    // Halley's method
    for (int i = 0; i < 12; ++i) {
        double ew = std::exp(w);
        double wew = w * ew;
        double f   = wew - x;
        double df  = ew * (w + 1.0);
        double d2f = ew * (w + 2.0);
        double dw  = f / (df - f * d2f / (2.0 * df));
        w -= dw;
        if (std::abs(dw) < std::abs(w) * 1e-15) break;
    }
    return w;
}

// ── Riemann zeta function ζ(s) for real s > 1 ────────────────────────────────
// Uses Euler–Maclaurin summation with 20 correction terms; accurate to ~1e-12.
inline double riemannZeta(double s) {
    if (s <= 1.0) throw std::domain_error("riemannZeta: s must be > 1");
    
    // Для s=2 нужно больше членов, так как ряд сходится медленно
    int N = (s < 3) ? 50 : 20;
    if (s < 1.5) N = 100;
    
    double sum = 0.0;
    for (int n = 1; n <= N; ++n)
        sum += 1.0 / std::pow(static_cast<double>(n), s);
    
    double N_d = static_cast<double>(N);
    double N_pow_ms = std::pow(N_d, -s);
    
    // Интегральная аппроксимация хвоста
    sum += std::pow(N_d, 1.0 - s) / (s - 1.0);
    sum += 0.5 * N_pow_ms;
    
    // Поправки Эйлера-Маклорена
    double term = N_pow_ms / N_d;  // N^{-s-1}
    sum += (s / 12.0) * term;
    
    term /= (N_d * N_d);  // N^{-s-3}
    sum -= (s * (s+1.0) * (s+2.0) / 720.0) * term;
    
    term /= (N_d * N_d);  // N^{-s-5}
    sum += (s * (s+1.0) * (s+2.0) * (s+3.0) * (s+4.0) / 30240.0) * term;
    
    return sum;
}

// Convenience: ζ(s) for integer argument
inline double riemannZeta(int s) { return riemannZeta(static_cast<double>(s)); }

} // namespace SharedMath::Functions
