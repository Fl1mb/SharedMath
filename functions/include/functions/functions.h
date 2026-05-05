/**
 * @file functions.h
 * @brief Umbrella header for the SharedMath Functions module.
 *
 * Include this single header to pull in the entire Functions module:
 *
 * | Sub-header            | Contents                                              |
 * |-----------------------|-------------------------------------------------------|
 * | SpecialFunctions.h    | Γ, ψ, β, Bessel, Legendre, elliptic integrals, sinc, W, ζ |
 * | Activations.h         | sigmoid, relu, gelu, swish, softmax, …                |
 * | LossFunctions.h       | mse, cross-entropy, huber, focal, regularisation, …   |
 *
 * Usage:
 * @code
 * #include <functions/functions.h>
 * @endcode
 *
 * @defgroup Functions Functions Module
 * @brief Collection of mathematical and ML functions for the SharedMath library.
 * @{
 *   @defgroup Functions_Special  Special Mathematical Functions
 *   @defgroup Functions_Activations Activation Functions
 *   @defgroup Functions_Losses   Loss Functions
 * @}
 *
 * @ingroup Functions
 */
#pragma once

#include "SpecialFunctions.h"   // Γ, ψ, β, Bessel, Legendre, elliptic, …
#include "Activations.h"        // sigmoid, relu, gelu, swish, softmax, …
#include "LossFunctions.h"      // mse, cross-entropy, huber, focal, …
