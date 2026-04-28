// DynamicMatrixCUDA.h — internal CUDA accessor and dispatch declarations for
// DynamicMatrix.
//
// THIS HEADER IS NOT PART OF THE PUBLIC API.
// It is included only by DynamicMatrix.cpp and DynamicMatrixCUDA.cu.
// Do not install it or include it from user-facing code.
//
// Architecture mirrors TensorCUDA.h:
//   • DynamicMatrixCUDAImpl   — grants the .cu TU access to private fields
//                               via the friend declaration in DynamicMatrix.h
//   • dm_cuda_*               — free-function dispatch points called from
//                               DynamicMatrix.cpp under #ifdef SHAREDMATH_CUDA
#pragma once

#include "DynamicMatrix.h"   // resolved via the PRIVATE include path

namespace SharedMath::LinearAlgebra {
namespace detail {

// ── Dispatch functions (defined in DynamicMatrixCUDA.cu) ──────────────────────

// Matrix multiply: C = A * B  (both must be on GPU, row-major)
DynamicMatrix dm_cuda_matmul(const DynamicMatrix& A, const DynamicMatrix& B);

// Element-wise add / subtract (same shape, both on GPU)
DynamicMatrix dm_cuda_add(const DynamicMatrix& A, const DynamicMatrix& B);
DynamicMatrix dm_cuda_sub(const DynamicMatrix& A, const DynamicMatrix& B);

// Element-wise multiply by scalar
DynamicMatrix dm_cuda_scale(const DynamicMatrix& A, double scalar);

// DynamicMatrixCUDAImpl is defined in DynamicMatrixCUDA.cu
// (where CUDABuffer is complete).  Forward-declared here for the friend decl.
struct DynamicMatrixCUDAImpl;

} // namespace detail
} // namespace SharedMath::LinearAlgebra
