#pragma once

/**
 * @file SharedMath.h
 * @brief Top-level convenience header — includes all SharedMath sub-modules.
 * @ingroup Core
 *
 * Include this single header to pull in the complete SharedMath library:
 * geometry, linear algebra, mathematical functions, and numerical methods.
 */

/**
 * @defgroup Core Core
 * @brief Root module of the SharedMath library.
 *
 * The Core module provides the foundational types, constants, and CUDA
 * management infrastructure consumed by every other SharedMath sub-module.
 */

#include "geometry/geometry.h"
#include "LinearAlgebra/LinearAlgebra.h"
#include "functions/functions.h"
#include "NumericalMethods/NumericalMethods.h"