#pragma once

#include "AbstractMatrix.h"
#include "DynamicMatrix.h"
#include "LUDecomposition.h"
#include "Tensor.h"
#include <sharedmath_linearalgebra_export.h>

#include <vector>
#include <tuple>
#include <string>
#include <utility>

namespace SharedMath::LinearAlgebra {

// ── Matrix norms ──────────────────────────────────────────────────────────────

enum class NormType {
    Frobenius,  // sqrt(sum of squared elements)  — default
    One,        // max column absolute sum
    Inf,        // max row absolute sum
    Two         // spectral norm (largest singular value)
};

SHAREDMATH_LINEARALGEBRA_EXPORT
double norm(const AbstractMatrix& A, NormType type = NormType::Frobenius);

// p-norm of a vector (p=2 → Euclidean; p=inf → max abs)
SHAREDMATH_LINEARALGEBRA_EXPORT
double norm(const std::vector<double>& v, double p = 2.0);

// ── Factory functions ─────────────────────────────────────────────────────────

// n×n identity matrix
SHAREDMATH_LINEARALGEBRA_EXPORT DynamicMatrix eye(size_t n);

SHAREDMATH_LINEARALGEBRA_EXPORT DynamicMatrix zeros(size_t rows, size_t cols);
SHAREDMATH_LINEARALGEBRA_EXPORT DynamicMatrix ones(size_t rows, size_t cols);

// Create n×n diagonal matrix from vector
SHAREDMATH_LINEARALGEBRA_EXPORT DynamicMatrix diag(const std::vector<double>& v);

// Extract main diagonal of A as a vector
SHAREDMATH_LINEARALGEBRA_EXPORT std::vector<double> diag(const AbstractMatrix& A);

// ── Solve / Inverse ───────────────────────────────────────────────────────────

// Solve Ax = b via LU decomposition (A must be square and non-singular)
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> solve(const AbstractMatrix& A, const std::vector<double>& b);

// Compute A^{-1} via LU decomposition (A must be square and non-singular)
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix inv(const AbstractMatrix& A);

// Moore-Penrose pseudoinverse via thin SVD
// tol < 0  → machine-epsilon-based threshold
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix pinv(const AbstractMatrix& A, double tol = -1.0);

// Least-squares solution: minimise ||Ax - b||_2
// Works for overdetermined, underdetermined, and rank-deficient systems
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> lstsq(const AbstractMatrix& A, const std::vector<double>& b);

// ── Matrix properties ─────────────────────────────────────────────────────────

// Numerical rank via Gaussian elimination with partial pivoting
SHAREDMATH_LINEARALGEBRA_EXPORT
size_t rank(const AbstractMatrix& A, double tol = 1e-10);

// Determinant via LU decomposition (A must be square)
SHAREDMATH_LINEARALGEBRA_EXPORT
double det(const AbstractMatrix& A);

// Sum of diagonal elements
SHAREDMATH_LINEARALGEBRA_EXPORT
double trace(const AbstractMatrix& A);

// Condition number κ(A) = σ_max / σ_min via SVD
// Returns infinity if A is rank-deficient
SHAREDMATH_LINEARALGEBRA_EXPORT
double cond(const AbstractMatrix& A, double tol = -1.0);

// Returns true if A[i][j] ≈ A[j][i] within tol
SHAREDMATH_LINEARALGEBRA_EXPORT
bool isSymmetric(const AbstractMatrix& A, double tol = 1e-10);

// Returns true if A^T * A ≈ I within tol (A must be square)
SHAREDMATH_LINEARALGEBRA_EXPORT
bool isOrthogonal(const AbstractMatrix& A, double tol = 1e-10);

// Returns true if A is symmetric and passes Cholesky (i.e. SPD)
SHAREDMATH_LINEARALGEBRA_EXPORT
bool isPositiveDefinite(const AbstractMatrix& A);

// Kronecker product: result[i*p+k][j*q+l] = A[i][j] * B[k][l]
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix kron(const AbstractMatrix& A, const AbstractMatrix& B);

// ── Vector operations ─────────────────────────────────────────────────────────

// Outer product: result[i][j] = u[i] * v[j]
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix outer(const std::vector<double>& u, const std::vector<double>& v);

// Dot (inner) product of two vectors
SHAREDMATH_LINEARALGEBRA_EXPORT
double inner(const std::vector<double>& u, const std::vector<double>& v);

// Cross product (3-D vectors only)
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> cross(const std::vector<double>& u, const std::vector<double>& v);

// ── Matrix functions f(A) ─────────────────────────────────────────────────────

// Matrix exponential via Taylor series + scaling-and-squaring (A must be square)
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix expm(const AbstractMatrix& A);

// Matrix square root via eigendecomposition (A must be symmetric PSD)
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix sqrtm(const AbstractMatrix& A);

// Matrix logarithm via eigendecomposition (A must be symmetric positive definite)
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix logm(const AbstractMatrix& A);

// ── Decompositions ────────────────────────────────────────────────────────────

// LU decomposition with partial pivoting: P*A = L*U
// Returns {L (lower-triangular, unit diagonal), U (upper-triangular), P (permutation)}
// A must be square.
SHAREDMATH_LINEARALGEBRA_EXPORT
std::tuple<DynamicMatrix, DynamicMatrix, DynamicMatrix>
lu(const AbstractMatrix& A);

// QR decomposition via Householder reflections
// Returns {Q (m×m orthogonal), R (m×n upper-triangular)}
SHAREDMATH_LINEARALGEBRA_EXPORT
std::pair<DynamicMatrix, DynamicMatrix> qr(const AbstractMatrix& A);

// QR with column pivoting: A * P_mat = Q * R  (reveals numerical rank)
// Returns {Q (m×m), R (m×n), pivot} where pivot[k] is the original column index
SHAREDMATH_LINEARALGEBRA_EXPORT
std::tuple<DynamicMatrix, DynamicMatrix, std::vector<size_t>>
qrp(const AbstractMatrix& A);

// Polar decomposition: A = U * P  (A must be square)
// U is orthogonal, P is symmetric positive semidefinite
// Returns {U, P}
SHAREDMATH_LINEARALGEBRA_EXPORT
std::pair<DynamicMatrix, DynamicMatrix> polar(const AbstractMatrix& A);

// Schur decomposition for real symmetric matrices: A = Q * T * Q^T
// T is diagonal (eigenvalues descending), Q is orthogonal
// Returns {Q, T}
SHAREDMATH_LINEARALGEBRA_EXPORT
std::pair<DynamicMatrix, DynamicMatrix>
schur(const AbstractMatrix& A, size_t max_iter = 1000);

// Cholesky: returns L such that A = L * L^T
// A must be symmetric positive-definite
SHAREDMATH_LINEARALGEBRA_EXPORT
DynamicMatrix cholesky(const AbstractMatrix& A);

// ── Eigenvalues ───────────────────────────────────────────────────────────────

// Eigenvalues of a real symmetric matrix via QR iteration (sorted descending)
SHAREDMATH_LINEARALGEBRA_EXPORT
std::vector<double> eigvals(const AbstractMatrix& A, size_t max_iter = 1000);

// Full eigendecomposition of a real symmetric matrix via QR iteration
// Returns {eigenvalues (descending), V} where columns of V are eigenvectors
SHAREDMATH_LINEARALGEBRA_EXPORT
std::pair<std::vector<double>, DynamicMatrix>
eig(const AbstractMatrix& A, size_t max_iter = 1000);

// ── SVD ───────────────────────────────────────────────────────────────────────

// Thin SVD: A ≈ U * diag(S) * Vt, where k = min(rows, cols)
// Returns {U (m×k), S (k singular values, descending), Vt (k×n)}
SHAREDMATH_LINEARALGEBRA_EXPORT
std::tuple<DynamicMatrix, std::vector<double>, DynamicMatrix>
svd(const AbstractMatrix& A, size_t max_iter = 1000);

// ── Tensor operations ─────────────────────────────────────────────────────────

// Generalised tensor contraction over paired axes
// axes_a[i] of a is contracted with axes_b[i] of b; dimensions must match
SHAREDMATH_LINEARALGEBRA_EXPORT
Tensor tensordot(const Tensor& a, const Tensor& b,
                 const std::vector<size_t>& axes_a,
                 const std::vector<size_t>& axes_b);

// Einstein summation — single-operand form
// Supported patterns: "ij->ji", "ii->", "ij->i", "ij->j", "ij->", "ii->i"
SHAREDMATH_LINEARALGEBRA_EXPORT
Tensor einsum(const std::string& subscripts, const Tensor& a);

// Einstein summation — two-operand form
// Supported patterns: "ij,jk->ik", "i,i->", "i,j->ij", "ij,ij->", etc.
SHAREDMATH_LINEARALGEBRA_EXPORT
Tensor einsum(const std::string& subscripts, const Tensor& a, const Tensor& b);

// ── CUDA runtime query ────────────────────────────────────────────────────────

// Returns true if the library was built with CUDA support AND at least one
// CUDA-capable GPU is available at runtime.
// When false, Tensor::cuda() is a no-op and all ops run on CPU.
SHAREDMATH_LINEARALGEBRA_EXPORT
bool cuda_is_available() noexcept;

} // namespace SharedMath::LinearAlgebra
