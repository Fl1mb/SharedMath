#pragma once

/// SharedMath::ML — Dimensionality Reduction
///
/// Re-exports SharedMath::LinearAlgebra::PCA under the ML namespace,
/// making it accessible via <ML/ml.h> alongside other ML utilities.

#include "LinearAlgebra/PCA.h"

namespace SharedMath::ML {

/// Bring PCA into the ML namespace as a type alias for convenience.
using PCA = SharedMath::LinearAlgebra::PCA;

} // namespace SharedMath::ML
