/**
 * @file NumericalMethods.h
 * @brief Convenience umbrella header that includes all NumericalMethods sub-modules.
 * @ingroup NumericalMethods
 *
 * @defgroup NumericalMethods Numerical Methods
 * @brief Collection of numerical algorithms provided by the SharedMath library.
 *
 * The NumericalMethods module is organised into the following sub-groups:
 *  - @ref NumericalMethods_Diff  — Numerical Differentiation
 *  - @ref NumericalMethods_Int   — Numerical Integration
 *  - @ref NumericalMethods_ODE   — ODE Solvers
 *  - @ref NumericalMethods_IE    — Integral Equations
 *  - @ref NumericalMethods_SLAE  — Linear System Solvers (SLAE)
 *
 * Include this header to pull in the entire module at once.
 */

#pragma once

#include "Differentiation.h"
#include "Integration.h"
#include "ODE.h"
#include "IntegralEquations.h"
#include "SLAE.h"
