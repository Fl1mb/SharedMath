#ifndef CONSTANS_H
#define CONSTANS_H

/**
 * @file constans.h
 * @brief Mathematical and numerical constants used throughout SharedMath.
 * @ingroup Core
 *
 * All constants have been taken from:
 * https://en.wikipedia.org/wiki/List_of_mathematical_constants
 */

#include <array>


namespace SharedMath
{
    using scalar = double;

    /* All constants have been taken from: */
    /* https://en.wikipedia.org/wiki/List_of_mathematical_constants */

    constexpr scalar Pi = 3.14159265358979323846;             ///< π — ratio of a circle's circumference to its diameter
    constexpr scalar Exponent = 2.71828182845904523536;       ///< Euler's number e (base of the natural logarithm)
    constexpr scalar Tau = 2 * Pi;                            ///< τ = 2π — full turn in radians
    constexpr scalar PifagorConst = 1.414213562373095;        ///< Pythagoras' constant √2 (diagonal of a unit square)
    constexpr scalar TeodorConst = 1.732050807568877;         ///< Theodorus' constant √3 (height of a unit equilateral triangle × 2)
    constexpr scalar Phi = 1.618033988749894;                 ///< Golden ratio φ = (1 + √5) / 2
    constexpr scalar Sqrt2 = 1.414213562373095;               ///< Square root of 2
    constexpr scalar Sqrt3 = 1.732050807568877;               ///< Square root of 3
    constexpr scalar Ln2 = 0.693147180559945;                 ///< Natural logarithm of 2
    constexpr scalar SQRT_2PI = 2.50662827463;                ///< √(2π) — normalisation factor of the Gaussian distribution


    static constexpr double EULER_GAMMA = 0.577215664901532860606512090082402431042; ///< Euler–Mascheroni constant γ

    /// Lanczos approximation coefficients (g = 7, n = 9) for the Gamma function.
    static constexpr std::array<double, 9> LANCZOS_COEFFS = {
        0.99999999999980993227684700473478,
        676.520368121885098567009190444019,
        -1259.13921672240287047156078755283,
        771.3234287776530788486528258894,
        -176.61502916214059906584551354,
        12.507343278686904814458936853,
        -0.13857109526572011689554707,
        9.984369578019570859563e-6,
        1.50563273514931155834e-7
    };

    static constexpr double LANCZOS_G = 7.0; ///< Lanczos g parameter
    static constexpr int LANCZOS_N = 9;      ///< Number of Lanczos coefficients

    constexpr scalar Epsilon = 1e-10; ///< Default numerical tolerance for floating-point comparisons


} // namespace SharedMath




#endif //CONSTANS_H