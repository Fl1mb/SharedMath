#include "MatrixFunctions.h"

#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <sstream>
#include <random>

namespace SharedMath::LinearAlgebra {

// ─── Internal helpers ─────────────────────────────────────────────────────────

// Copy any AbstractMatrix into a DynamicMatrix (flat row-major storage).
// If A is already a DynamicMatrix we avoid the virtual-call loop entirely.
static DynamicMatrix toDynamic(const AbstractMatrix& A) {
    if (const auto* d = dynamic_cast<const DynamicMatrix*>(&A))
        return *d;               // already flat — just copy
    return DynamicMatrix(A);     // deep copy through virtual interface
}

static DynamicMatrix transposeMatrix(const AbstractMatrix& A) {
    // Fast path: DynamicMatrix has its own cache-friendly transposed()
    if (const auto* d = dynamic_cast<const DynamicMatrix*>(&A))
        return d->transposed();

    size_t m = A.rows(), n = A.cols();
    DynamicMatrix M(n, m);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j)
            M(j, i) = A.get(i, j);
    }
    return M;
}

// Forward/backward substitution given an already-decomposed LU and rhs b.
// pivot[i] contains the original row mapped to position i (partial-pivoting).
static std::vector<double> luSolve(const DynamicMatrix& L,
                                   const DynamicMatrix& U,
                                   const std::vector<size_t>& pivot,
                                   const std::vector<double>& b)
{
    size_t n = b.size();

    // Apply row permutation: pb[i] = b[pivot[i]]
    std::vector<double> pb(n);
    for (size_t i = 0; i < n; ++i) pb[i] = b[pivot[i]];

    // Forward substitution: Ly = pb  (L has 1s on diagonal)
    std::vector<double> y(n);
    for (size_t i = 0; i < n; ++i) {
        double s = pb[i];
        for (size_t j = 0; j < i; ++j) s -= L.get(i, j) * y[j];
        y[i] = s;
    }

    // Backward substitution: Ux = y
    std::vector<double> x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        double s = y[i];
        for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j)
            s -= U.get(i, j) * x[j];
        x[i] = s / U.get(i, i);
    }
    return x;
}

// Power iteration to find the largest eigenvalue of a symmetric PSD matrix.
static double largestEigenvalue(const DynamicMatrix& M, size_t max_iter = 1000) {
    size_t n = M.rows();
    std::vector<double> v(n, 1.0 / std::sqrt(static_cast<double>(n)));
    double lambda = 0.0;
    for (size_t iter = 0; iter < max_iter; ++iter) {
        std::vector<double> Mv(n, 0.0);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                Mv[i] += M.get(i, j) * v[j];
        double nrm = 0.0;
        for (double x : Mv) nrm += x * x;
        nrm = std::sqrt(nrm);
        if (nrm < 1e-14) break;
        double new_lambda = 0.0;
        for (size_t i = 0; i < n; ++i) {
            v[i] = Mv[i] / nrm;
            new_lambda += v[i] * Mv[i];
        }
        if (std::abs(new_lambda - lambda) < 1e-12) { lambda = new_lambda; break; }
        lambda = new_lambda;
    }
    return lambda;
}

// ─── norm ─────────────────────────────────────────────────────────────────────

double norm(const AbstractMatrix& A, NormType type) {
    size_t m = A.rows(), n = A.cols();
    switch (type) {
    case NormType::Frobenius: {
        double s = 0.0;
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                s += A.get(i, j) * A.get(i, j);
        return std::sqrt(s);
    }
    case NormType::One: {
        double mx = 0.0;
        for (size_t j = 0; j < n; ++j) {
            double col = 0.0;
            for (size_t i = 0; i < m; ++i) col += std::abs(A.get(i, j));
            mx = std::max(mx, col);
        }
        return mx;
    }
    case NormType::Inf: {
        double mx = 0.0;
        for (size_t i = 0; i < m; ++i) {
            double row = 0.0;
            for (size_t j = 0; j < n; ++j) row += std::abs(A.get(i, j));
            mx = std::max(mx, row);
        }
        return mx;
    }
    case NormType::Nuclear: {
        // Nuclear norm = sum of singular values
        auto [U, S, Vt] = svd(A);
        double s = 0.0;
        for (double sv : S) s += sv;
        return s;
    }
    case NormType::Two: {
        // Spectral norm = sqrt(largest eigenvalue of A^T * A)
        DynamicMatrix At = transposeMatrix(A);
        DynamicMatrix AtA = At * toDynamic(A);
        return std::sqrt(std::max(0.0, largestEigenvalue(AtA)));
    }
    }
    return 0.0;
}

double norm(const std::vector<double>& v, double p) {
    if (std::isinf(p) && p > 0) {
        double mx = 0.0;
        for (double x : v) mx = std::max(mx, std::abs(x));
        return mx;
    }
    double s = 0.0;
    for (double x : v) s += std::pow(std::abs(x), p);
    return std::pow(s, 1.0 / p);
}

// ─── factory functions ────────────────────────────────────────────────────────

DynamicMatrix eye(size_t n) {
    DynamicMatrix M(n, n);
    for (size_t i = 0; i < n; ++i) M.set(i, i, 1.0);
    return M;
}

DynamicMatrix zeros(size_t rows, size_t cols) { return DynamicMatrix(rows, cols); }

DynamicMatrix ones(size_t rows, size_t cols) {
    DynamicMatrix M(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            M.set(i, j, 1.0);
    return M;
}

DynamicMatrix diag(const std::vector<double>& v) {
    size_t n = v.size();
    DynamicMatrix M(n, n);
    for (size_t i = 0; i < n; ++i) M.set(i, i, v[i]);
    return M;
}

std::vector<double> diag(const AbstractMatrix& A) {
    size_t n = std::min(A.rows(), A.cols());
    std::vector<double> d(n);
    for (size_t i = 0; i < n; ++i) d[i] = A.get(i, i);
    return d;
}

// ─── solve / inverse ─────────────────────────────────────────────────────────

std::vector<double> solve(const AbstractMatrix& A, const std::vector<double>& b) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("solve: A must be square");
    if (A.rows() != b.size())
        throw std::invalid_argument("solve: dimension mismatch between A and b");
    DynamicMatrix Ad = toDynamic(A);
    LUDecomposition lu(Ad);
    lu.MakeDecomposition();
    return luSolve(lu.GetL(), lu.GetU(), lu.GetPivot(), b);
}

DynamicMatrix inv(const AbstractMatrix& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("inv: A must be square");
    size_t n = A.rows();
    DynamicMatrix Ad = toDynamic(A);
    LUDecomposition lu(Ad);
    lu.MakeDecomposition();

    DynamicMatrix result(n, n);
    std::vector<double> e(n, 0.0);
    for (size_t col = 0; col < n; ++col) {
        if (col > 0) e[col - 1] = 0.0;
        e[col] = 1.0;
        auto x = luSolve(lu.GetL(), lu.GetU(), lu.GetPivot(), e);
        for (size_t row = 0; row < n; ++row)
            result.set(row, col, x[row]);
    }
    return result;
}

// ─── LU decomposition (wraps LUDecomposition, returns P·A = L·U) ─────────────

std::tuple<DynamicMatrix, DynamicMatrix, DynamicMatrix>
lu(const AbstractMatrix& A)
{
    if (A.rows() != A.cols())
        throw std::invalid_argument("lu: requires square matrix");

    DynamicMatrix Ad = toDynamic(A);
    LUDecomposition dec(Ad);
    dec.MakeDecomposition();
    return {dec.GetL(), dec.GetU(), dec.GetPermutationMatrix()};
}

// ─── QR decomposition (Householder reflections) ───────────────────────────────

std::pair<DynamicMatrix, DynamicMatrix> qr(const AbstractMatrix& A) {
    size_t m = A.rows(), n = A.cols();

    // Working copies use DynamicMatrix (flat, cache-friendly) instead of
    // vector<vector<double>> to keep rows contiguous for the dot products.
    DynamicMatrix R = toDynamic(A);       // m × n
    DynamicMatrix Q = DynamicMatrix::eye(m); // m × m identity

    size_t k_max = std::min(m, n);
    for (size_t k = 0; k < k_max; ++k) {
        // x = column k of R starting at row k
        size_t len = m - k;
        std::vector<double> x(len);
        for (size_t i = 0; i < len; ++i) x[i] = R(k + i, k);

        double norm_x = 0.0;
        for (double v : x) norm_x += v * v;
        norm_x = std::sqrt(norm_x);
        if (norm_x < 1e-14) continue;

        // Householder reflector: u = x ± ‖x‖ e₁, normalised
        std::vector<double> u = x;
        u[0] += (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
        double norm_u = 0.0;
        for (double v : u) norm_u += v * v;
        norm_u = std::sqrt(norm_u);
        if (norm_u < 1e-14) continue;
        for (double& v : u) v /= norm_u;

        // Apply H = I − 2uuᵀ from the left to R[k:m, k:n]
        for (size_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (size_t i = 0; i < len; ++i) dot += u[i] * R(k + i, j);
            for (size_t i = 0; i < len; ++i) R(k + i, j) -= 2.0 * u[i] * dot;
        }

        // Apply H from the right to Q  (accumulate Q = H₀ H₁ …)
        for (size_t i = 0; i < m; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < len; ++j) dot += Q(i, k + j) * u[j];
            for (size_t j = 0; j < len; ++j) Q(i, k + j) -= 2.0 * dot * u[j];
        }
    }

    // Zero out below-diagonal entries of R (numerical noise)
    for (size_t i = 1; i < m; ++i)
        for (size_t j = 0; j < std::min(i, n); ++j)
            R(i, j) = 0.0;

    return {Q, R};
}

// ─── Cholesky decomposition ───────────────────────────────────────────────────

DynamicMatrix cholesky(const AbstractMatrix& A) {
    size_t n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("cholesky: requires square matrix");

    DynamicMatrix L(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = A.get(i, j);
            for (size_t k = 0; k < j; ++k)
                s -= L.get(i, k) * L.get(j, k);
            if (i == j) {
                if (s < 1e-14)
                    throw std::runtime_error(
                        "cholesky: matrix is not positive definite");
                L.set(i, j, std::sqrt(s));
            } else {
                L.set(i, j, s / L.get(j, j));
            }
        }
    }
    return L;
}

// ─── Eigenvalues / eigenvectors (real symmetric, QR iteration) ────────────────

std::pair<std::vector<double>, DynamicMatrix>
eig(const AbstractMatrix& A, size_t max_iter)
{
    if (A.rows() != A.cols())
        throw std::invalid_argument("eig: requires square matrix");
    size_t n = A.rows();

    DynamicMatrix Ak = toDynamic(A);

    // V accumulates the product of all Q factors: eigenvector matrix
    DynamicMatrix V = eye(n);

    for (size_t iter = 0; iter < max_iter; ++iter) {
        // ① Check convergence BEFORE touching the matrix.
        //   This avoids computing the Wilkinson shift when Ak is already
        //   diagonal (e.g. identity), which would produce 0/0 = NaN.
        {
            double off = 0.0;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    if (i != j) off += Ak(i, j) * Ak(i, j);
            if (std::sqrt(off) < 1e-10 * static_cast<double>(n)) break;
        }

        // ② Wilkinson shift from the bottom-right 2×2 submatrix.
        //    Guard: when the sub-diagonal entry b ≈ 0 the last eigenvalue has
        //    already separated; use Ak(n-1,n-1) as shift to avoid 0/0 = NaN.
        double shift = 0.0;
        if (n >= 2) {
            double b = Ak(n-1, n-2);
            if (std::abs(b) < 1e-14) {
                shift = Ak(n-1, n-1);
            } else {
                double d  = (Ak(n-2, n-2) - Ak(n-1, n-1)) / 2.0;
                double sq = std::sqrt(d * d + b * b);
                shift = Ak(n-1, n-1) - (b * b) / (d + (d >= 0.0 ? sq : -sq));
            }
        }

        // ③ A_shifted = Ak − shift·I
        for (size_t i = 0; i < n; ++i)
            Ak(i, i) -= shift;

        auto [Q, R] = qr(Ak);

        // ④ Ak = R·Q + shift·I
        Ak = R * Q;
        for (size_t i = 0; i < n; ++i)
            Ak(i, i) += shift;

        // ⑤ Accumulate eigenvectors: V = V·Q
        V = V * Q;
    }

    // Extract and sort eigenvalues descending
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return Ak.get(a, a) > Ak.get(b, b);
    });

    std::vector<double> evals(n);
    DynamicMatrix evecs(n, n);
    for (size_t i = 0; i < n; ++i) {
        evals[i] = Ak.get(order[i], order[i]);
        for (size_t row = 0; row < n; ++row)
            evecs.set(row, i, V.get(row, order[i]));
    }
    return {evals, evecs};
}

std::vector<double> eigvals(const AbstractMatrix& A, size_t max_iter) {
    return eig(A, max_iter).first;
}

// ─── Thin SVD ─────────────────────────────────────────────────────────────────
// Uses eigendecomposition of A^T * A to obtain right singular vectors V
// and computes U = A * V * diag(1/sigma) for nonzero singular values.

std::tuple<DynamicMatrix, std::vector<double>, DynamicMatrix>
svd(const AbstractMatrix& A, size_t max_iter)
{
    size_t m = A.rows(), n = A.cols();
    size_t k = std::min(m, n);

    // B = A^T * A  (n×n symmetric PSD)
    DynamicMatrix At = transposeMatrix(A);
    DynamicMatrix B  = At * toDynamic(A);

    // Eigendecompose B: B = V * D * V^T
    auto [evals, V] = eig(B, max_iter);

    // Singular values σ_i = sqrt(max(λ_i, 0)), keep first k
    std::vector<double> S(k);
    for (size_t i = 0; i < k; ++i)
        S[i] = (i < evals.size() && evals[i] > 0.0)
               ? std::sqrt(evals[i]) : 0.0;

    // V^T: first k rows of V^T  (k×n)
    DynamicMatrix Vt(k, n);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n; ++j)
            Vt.set(i, j, V.get(j, i));

    // U: left singular vectors (m×k), u_i = A * v_i / σ_i
    DynamicMatrix Ad  = toDynamic(A);
    DynamicMatrix U(m, k);
    for (size_t i = 0; i < k; ++i) {
        if (S[i] < 1e-12) continue;   // zero singular value: leave col as 0
        for (size_t row = 0; row < m; ++row) {
            double val = 0.0;
            for (size_t j = 0; j < n; ++j)
                val += Ad.get(row, j) * V.get(j, i);
            U.set(row, i, val / S[i]);
        }
    }

    return {U, S, Vt};
}

// ─── pseudoinverse ────────────────────────────────────────────────────────────

DynamicMatrix pinv(const AbstractMatrix& A, double tol) {
    size_t m = A.rows(), n = A.cols();
    auto [U, S, Vt] = svd(A);
    size_t k = S.size();

    if (tol < 0.0) {
        double max_s = k > 0 ? S[0] : 0.0;
        tol = std::numeric_limits<double>::epsilon() *
              static_cast<double>(std::max(m, n)) * max_s;
    }

    // Pinv = V * diag(1/S) * U^T   (n×m)
    DynamicMatrix result(n, m);
    for (size_t i = 0; i < k; ++i) {
        if (S[i] < tol) continue;
        double inv_s = 1.0 / S[i];
        // result += (v_i / σ_i) * u_i^T
        for (size_t r = 0; r < n; ++r)
            for (size_t c = 0; c < m; ++c)
                result.set(r, c, result.get(r, c) + Vt.get(i, r) * U.get(c, i) * inv_s);
    }
    return result;
}

// ─── least squares ───────────────────────────────────────────────────────────

std::vector<double> lstsq(const AbstractMatrix& A, const std::vector<double>& b) {
    // x = pinv(A) * b
    DynamicMatrix P = pinv(A);
    size_t n = P.rows();
    std::vector<double> x(n, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < b.size(); ++j)
            x[i] += P.get(i, j) * b[j];
    return x;
}

// ─── rank ────────────────────────────────────────────────────────────────────

size_t rank(const AbstractMatrix& A, double tol) {
    size_t m = A.rows(), n = A.cols();
    // Copy into 2-D vector for in-place elimination
    std::vector<std::vector<double>> M(m, std::vector<double>(n));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            M[i][j] = A.get(i, j);

    size_t r = 0, row = 0;
    for (size_t col = 0; col < n && row < m; ++col) {
        // Partial-pivot: find row with largest absolute value in column
        size_t pivot = row;
        for (size_t i = row + 1; i < m; ++i)
            if (std::abs(M[i][col]) > std::abs(M[pivot][col])) pivot = i;
        if (std::abs(M[pivot][col]) < tol) continue;

        std::swap(M[row], M[pivot]);
        double inv = 1.0 / M[row][col];
        for (size_t i = row + 1; i < m; ++i) {
            double f = M[i][col] * inv;
            for (size_t j = col; j < n; ++j) M[i][j] -= f * M[row][j];
        }
        ++r; ++row;
    }
    return r;
}

// ─── vector operations ────────────────────────────────────────────────────────

DynamicMatrix outer(const std::vector<double>& u, const std::vector<double>& v) {
    size_t m = u.size(), n = v.size();
    DynamicMatrix M(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            M.set(i, j, u[i] * v[j]);
    return M;
}

double inner(const std::vector<double>& u, const std::vector<double>& v) {
    if (u.size() != v.size())
        throw std::invalid_argument("inner: vectors must have the same size");
    double s = 0.0;
    for (size_t i = 0; i < u.size(); ++i) s += u[i] * v[i];
    return s;
}

std::vector<double> cross(const std::vector<double>& u,
                           const std::vector<double>& v)
{
    if (u.size() != 3 || v.size() != 3)
        throw std::invalid_argument("cross: both vectors must be 3-D");
    return {
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    };
}

// ─── tensordot ────────────────────────────────────────────────────────────────

Tensor tensordot(const Tensor& a, const Tensor& b,
                 const std::vector<size_t>& axes_a,
                 const std::vector<size_t>& axes_b)
{
    if (axes_a.size() != axes_b.size())
        throw std::invalid_argument("tensordot: axes vectors must have the same length");

    for (size_t i = 0; i < axes_a.size(); ++i)
        if (a.dim(axes_a[i]) != b.dim(axes_b[i]))
            throw std::invalid_argument("tensordot: contracted dimensions must match");

    // Collect free axes
    std::vector<size_t> free_a, free_b;
    for (size_t i = 0; i < a.ndim(); ++i) {
        if (std::find(axes_a.begin(), axes_a.end(), i) == axes_a.end())
            free_a.push_back(i);
    }
    for (size_t i = 0; i < b.ndim(); ++i) {
        if (std::find(axes_b.begin(), axes_b.end(), i) == axes_b.end())
            free_b.push_back(i);
    }

    // Result shape: free dims of a followed by free dims of b
    Tensor::Shape rshape;
    for (size_t i : free_a) rshape.push_back(a.dim(i));
    for (size_t i : free_b) rshape.push_back(b.dim(i));
    if (rshape.empty()) rshape.push_back(1);

    Tensor result(rshape, 0.0);

    // Contracted-index shape and strides
    Tensor::Shape c_shape;
    for (size_t ax : axes_a) c_shape.push_back(a.dim(ax));
    size_t c_size = 1;
    for (size_t d : c_shape) c_size *= d;

    std::vector<size_t> c_strides(c_shape.size(), 1);
    if (!c_shape.empty())
        for (int s = static_cast<int>(c_shape.size()) - 2; s >= 0; --s)
            c_strides[s] = c_strides[s + 1] * c_shape[s + 1];

    // Iterate over result elements
    for (size_t flat_r = 0; flat_r < result.size(); ++flat_r) {
        auto ridx = result.unravel(flat_r);

        // Map ridx → free indices for a and b
        std::vector<size_t> fa(a.ndim(), 0), fb(b.ndim(), 0);
        for (size_t i = 0; i < free_a.size(); ++i) fa[free_a[i]] = ridx[i];
        for (size_t i = 0; i < free_b.size(); ++i) fb[free_b[i]] = ridx[free_a.size() + i];

        // Sum over contracted indices
        double acc = 0.0;
        for (size_t flat_c = 0; flat_c < c_size; ++flat_c) {
            size_t tmp = flat_c;
            for (size_t k = 0; k < c_shape.size(); ++k) {
                size_t ci = tmp / c_strides[k];
                tmp       %= c_strides[k];
                fa[axes_a[k]] = ci;
                fb[axes_b[k]] = ci;
            }
            acc += a.at(fa) * b.at(fb);
        }
        result.flat(flat_r) = acc;
    }
    return result;
}

// ─── einsum ──────────────────────────────────────────────────────────────────

// Parse "lhs->rhs" subscript string.
static std::pair<std::string, std::string>
parseEinsum(const std::string& s, size_t expected_commas)
{
    auto arrow = s.find("->");
    if (arrow == std::string::npos)
        throw std::invalid_argument("einsum: subscripts must contain '->'");
    std::string lhs = s.substr(0, arrow);
    std::string rhs = s.substr(arrow + 2);
    size_t commas = std::count(lhs.begin(), lhs.end(), ',');
    if (commas != expected_commas)
        throw std::invalid_argument("einsum: wrong number of operands in subscripts");
    return {lhs, rhs};
}

// Build a map from index char → current multi-index value.
// Encode/decode a flat index over a given shape into the map.
static void decodeFlatIntoMap(size_t flat,
                               const std::string& labels,
                               const Tensor& t,
                               std::map<char, size_t>& idx_map)
{
    std::vector<size_t> idx = t.unravel(flat);
    for (size_t i = 0; i < labels.size(); ++i) idx_map[labels[i]] = idx[i];
}

Tensor einsum(const std::string& subscripts, const Tensor& a) {
    auto [lhs, out] = parseEinsum(subscripts, 0);
    if (lhs.size() != a.ndim())
        throw std::invalid_argument("einsum: subscript rank mismatch");

    std::map<char, size_t> sizes;
    for (size_t i = 0; i < lhs.size(); ++i) sizes[lhs[i]] = a.dim(i);

    Tensor::Shape out_shape;
    for (char c : out) out_shape.push_back(sizes.at(c));
    if (out_shape.empty()) out_shape.push_back(1);

    // Contracted indices: in lhs but not in out
    std::string contracted;
    for (char c : lhs)
        if (out.find(c) == std::string::npos && contracted.find(c) == std::string::npos)
            contracted += c;

    Tensor::Shape c_shape;
    for (char c : contracted) c_shape.push_back(sizes.at(c));
    size_t c_size = 1;
    for (size_t d : c_shape) c_size *= d;
    std::vector<size_t> c_strides(c_shape.size(), 1);
    if (!c_shape.empty())
        for (int i = static_cast<int>(c_shape.size()) - 2; i >= 0; --i)
            c_strides[i] = c_strides[i + 1] * c_shape[i + 1];

    Tensor result(out_shape, 0.0);
    for (size_t flat_r = 0; flat_r < result.size(); ++flat_r) {
        // Decode output multi-index
        auto ridx = result.unravel(flat_r);
        std::map<char, size_t> idx_map;
        for (size_t k = 0; k < out.size(); ++k) idx_map[out[k]] = ridx[k];

        double acc = 0.0;
        for (size_t flat_c = 0; flat_c < (c_size == 0 ? 1 : c_size); ++flat_c) {
            size_t tmp = flat_c;
            for (size_t k = 0; k < contracted.size(); ++k) {
                idx_map[contracted[k]] = tmp / c_strides[k];
                tmp %= c_strides[k];
            }
            std::vector<size_t> a_idx(lhs.size());
            for (size_t k = 0; k < lhs.size(); ++k) a_idx[k] = idx_map.at(lhs[k]);
            acc += a.at(a_idx);
        }
        result.flat(flat_r) = acc;
    }
    return result;
}

Tensor einsum(const std::string& subscripts, const Tensor& a, const Tensor& b) {
    auto [lhs_all, out] = parseEinsum(subscripts, 1);
    auto comma = lhs_all.find(',');
    std::string lhs_a = lhs_all.substr(0, comma);
    std::string lhs_b = lhs_all.substr(comma + 1);

    if (lhs_a.size() != a.ndim() || lhs_b.size() != b.ndim())
        throw std::invalid_argument("einsum: subscript rank mismatch");

    std::map<char, size_t> sizes;
    for (size_t i = 0; i < lhs_a.size(); ++i) sizes[lhs_a[i]] = a.dim(i);
    for (size_t i = 0; i < lhs_b.size(); ++i) {
        char c = lhs_b[i];
        if (sizes.count(c) && sizes[c] != b.dim(i))
            throw std::invalid_argument("einsum: index size mismatch");
        sizes[c] = b.dim(i);
    }

    Tensor::Shape out_shape;
    for (char c : out) out_shape.push_back(sizes.at(c));
    if (out_shape.empty()) out_shape.push_back(1);

    // Contracted indices: in lhs_a or lhs_b but not in out
    std::string contracted;
    for (char c : lhs_a + lhs_b)
        if (out.find(c) == std::string::npos && contracted.find(c) == std::string::npos)
            contracted += c;

    Tensor::Shape c_shape;
    for (char c : contracted) c_shape.push_back(sizes.at(c));
    size_t c_size = 1;
    for (size_t d : c_shape) c_size *= d;
    std::vector<size_t> c_strides(c_shape.size(), 1);
    if (!c_shape.empty())
        for (int i = static_cast<int>(c_shape.size()) - 2; i >= 0; --i)
            c_strides[i] = c_strides[i + 1] * c_shape[i + 1];

    Tensor result(out_shape, 0.0);
    for (size_t flat_r = 0; flat_r < result.size(); ++flat_r) {
        auto ridx = result.unravel(flat_r);
        std::map<char, size_t> idx_map;
        for (size_t k = 0; k < out.size(); ++k) idx_map[out[k]] = ridx[k];

        double acc = 0.0;
        for (size_t flat_c = 0; flat_c < (c_size == 0 ? 1 : c_size); ++flat_c) {
            size_t tmp = flat_c;
            for (size_t k = 0; k < contracted.size(); ++k) {
                idx_map[contracted[k]] = tmp / c_strides[k];
                tmp %= c_strides[k];
            }
            std::vector<size_t> ai(lhs_a.size()), bi(lhs_b.size());
            for (size_t k = 0; k < lhs_a.size(); ++k) ai[k] = idx_map.at(lhs_a[k]);
            for (size_t k = 0; k < lhs_b.size(); ++k) bi[k] = idx_map.at(lhs_b[k]);
            acc += a.at(ai) * b.at(bi);
        }
        result.flat(flat_r) = acc;
    }
    return result;
}

// ─── det ─────────────────────────────────────────────────────────────────────

double det(const AbstractMatrix& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("det: requires square matrix");
    DynamicMatrix Ad = toDynamic(A);
    LUDecomposition dec(Ad);
    dec.MakeDecomposition();
    return dec.Determinant();
}

// ─── trace ────────────────────────────────────────────────────────────────────

double trace(const AbstractMatrix& A) {
    size_t n = std::min(A.rows(), A.cols());
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += A.get(i, i);
    return s;
}

// ─── cond ─────────────────────────────────────────────────────────────────────

double cond(const AbstractMatrix& A, double tol) {
    auto [U, S, Vt] = svd(A);
    if (S.empty() || S[0] < 1e-14)
        return std::numeric_limits<double>::infinity();
    double s_min = S.back();
    double s_tol = (tol < 0.0)
                   ? std::numeric_limits<double>::epsilon() *
                     static_cast<double>(std::max(A.rows(), A.cols())) * S[0]
                   : tol;
    if (s_min < s_tol)
        return std::numeric_limits<double>::infinity();
    return S[0] / s_min;
}

// ─── isSymmetric / isOrthogonal / isPositiveDefinite ─────────────────────────

bool isSymmetric(const AbstractMatrix& A, double tol) {
    if (A.rows() != A.cols()) return false;
    size_t n = A.rows();
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (std::abs(A.get(i, j) - A.get(j, i)) > tol) return false;
    return true;
}

bool isOrthogonal(const AbstractMatrix& A, double tol) {
    if (A.rows() != A.cols()) return false;
    size_t n = A.rows();
    DynamicMatrix Ad  = toDynamic(A);
    DynamicMatrix AtA = transposeMatrix(A) * Ad;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (std::abs(AtA.get(i, j) - (i == j ? 1.0 : 0.0)) > tol) return false;
    return true;
}

bool isPositiveDefinite(const AbstractMatrix& A) {
    if (A.rows() != A.cols() || !isSymmetric(A)) return false;
    try { cholesky(A); return true; } catch (...) { return false; }
}

// ─── kron ─────────────────────────────────────────────────────────────────────

DynamicMatrix kron(const AbstractMatrix& A, const AbstractMatrix& B) {
    size_t m = A.rows(), n = A.cols();
    size_t p = B.rows(), q = B.cols();
    DynamicMatrix result(m * p, n * q);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double a = A.get(i, j);
            for (size_t k = 0; k < p; ++k)
                for (size_t l = 0; l < q; ++l)
                    result.set(i * p + k, j * q + l, a * B.get(k, l));
        }
    return result;
}

// ─── expm ─────────────────────────────────────────────────────────────────────
// Scaling-and-squaring with Taylor series: exp(A) = exp(A/2^s)^{2^s}

DynamicMatrix expm(const AbstractMatrix& A) {
    size_t n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("expm: requires square matrix");

    DynamicMatrix Ad = toDynamic(A);

    // Choose s so that ||A / 2^s||_1 <= 1
    double nrm = norm(Ad, NormType::One);
    int s = (nrm > 0.5) ? static_cast<int>(std::ceil(std::log2(nrm))) : 0;
    double scale = std::ldexp(1.0, -s);   // 2^{-s}

    DynamicMatrix As(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            As.set(i, j, Ad.get(i, j) * scale);

    // Taylor series: term_k = term_{k-1} * As / k, result += term_k
    DynamicMatrix result = eye(n);
    DynamicMatrix term   = eye(n);
    for (size_t k = 1; k <= 30; ++k) {
        term = term * As;
        double inv_k = 1.0 / static_cast<double>(k);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                term.set(i, j, term.get(i, j) * inv_k);

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                result.set(i, j, result.get(i, j) + term.get(i, j));

        double res_nrm = norm(result, NormType::One);
        if (norm(term, NormType::One) < 1e-15 * res_nrm) break;
    }

    // Square back 2^s times
    for (int i = 0; i < s; ++i)
        result = result * result;

    return result;
}

// ─── sqrtm ────────────────────────────────────────────────────────────────────
// Via eigendecomposition: sqrtm(A) = V * sqrt(D) * V^T  (symmetric PSD only)

DynamicMatrix sqrtm(const AbstractMatrix& A) {
    size_t n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("sqrtm: requires square matrix");
    if (!isSymmetric(A))
        throw std::invalid_argument("sqrtm: only symmetric positive semidefinite matrices are supported");

    auto [evals, V] = eig(A);

    for (double ev : evals)
        if (ev < -1e-10)
            throw std::runtime_error("sqrtm: matrix has negative eigenvalues");

    DynamicMatrix result(n, n);
    for (size_t k = 0; k < n; ++k) {
        double sq = (evals[k] > 0.0) ? std::sqrt(evals[k]) : 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                result.set(i, j, result.get(i, j) + sq * V.get(i, k) * V.get(j, k));
    }
    return result;
}

// ─── logm ─────────────────────────────────────────────────────────────────────
// Via eigendecomposition: logm(A) = V * log(D) * V^T  (symmetric PD only)

DynamicMatrix logm(const AbstractMatrix& A) {
    size_t n = A.rows();
    if (A.cols() != n)
        throw std::invalid_argument("logm: requires square matrix");
    if (!isSymmetric(A))
        throw std::invalid_argument("logm: only symmetric positive definite matrices are supported");

    auto [evals, V] = eig(A);

    for (double ev : evals)
        if (ev <= 1e-14)
            throw std::runtime_error("logm: matrix must be positive definite (all eigenvalues > 0)");

    DynamicMatrix result(n, n);
    for (size_t k = 0; k < n; ++k) {
        double lg = std::log(evals[k]);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                result.set(i, j, result.get(i, j) + lg * V.get(i, k) * V.get(j, k));
    }
    return result;
}

// ─── QR with column pivoting ─────────────────────────────────────────────────

std::tuple<DynamicMatrix, DynamicMatrix, std::vector<size_t>>
qrp(const AbstractMatrix& A)
{
    size_t m = A.rows(), n = A.cols();
    DynamicMatrix R = toDynamic(A);
    DynamicMatrix Q = eye(m);
    std::vector<size_t> piv(n);
    std::iota(piv.begin(), piv.end(), 0);

    size_t k_max = std::min(m, n);
    for (size_t k = 0; k < k_max; ++k) {
        // Find pivot: column k..n-1 with largest sub-column 2-norm
        double best_nrm = -1.0;
        size_t best_col = k;
        for (size_t j = k; j < n; ++j) {
            double nrm = 0.0;
            for (size_t i = k; i < m; ++i) nrm += R(i, j) * R(i, j);
            if (nrm > best_nrm) { best_nrm = nrm; best_col = j; }
        }

        // Swap columns k ↔ best_col in R and pivot vector
        if (best_col != k) {
            for (size_t i = 0; i < m; ++i) std::swap(R(i, k), R(i, best_col));
            std::swap(piv[k], piv[best_col]);
        }

        // Householder reflection for column k
        size_t len = m - k;
        std::vector<double> x(len);
        for (size_t i = 0; i < len; ++i) x[i] = R(k + i, k);

        double norm_x = 0.0;
        for (double v : x) norm_x += v * v;
        norm_x = std::sqrt(norm_x);
        if (norm_x < 1e-14) continue;

        std::vector<double> u = x;
        u[0] += (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
        double norm_u = 0.0;
        for (double v : u) norm_u += v * v;
        if (norm_u < 1e-14) continue;
        norm_u = std::sqrt(norm_u);
        for (double& v : u) v /= norm_u;

        // Apply H to R[k:m, k:n]
        for (size_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (size_t i = 0; i < len; ++i) dot += u[i] * R(k + i, j);
            for (size_t i = 0; i < len; ++i) R(k + i, j) -= 2.0 * u[i] * dot;
        }

        // Accumulate Q
        for (size_t i = 0; i < m; ++i) {
            double dot = 0.0;
            for (size_t j = 0; j < len; ++j) dot += Q(i, k + j) * u[j];
            for (size_t j = 0; j < len; ++j) Q(i, k + j) -= 2.0 * dot * u[j];
        }
    }

    // Zero sub-diagonal noise in R
    for (size_t i = 1; i < m; ++i)
        for (size_t j = 0; j < std::min(i, n); ++j)
            R(i, j) = 0.0;

    return {Q, R, piv};
}

// ─── Polar decomposition ──────────────────────────────────────────────────────
// A = U_p * P  where U_p orthogonal, P symmetric PSD  (A must be square)

std::pair<DynamicMatrix, DynamicMatrix> polar(const AbstractMatrix& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("polar: requires square matrix");

    // Thin SVD = full SVD for square A
    auto [U_s, S, Vt] = svd(A);
    size_t n = A.rows();
    size_t k = S.size();

    // U_polar = U_s * Vt
    DynamicMatrix U_polar = U_s * Vt;  // m×k * k×n = n×n for square

    // P_polar = V * diag(S) * Vt  (n×n symmetric PSD)
    DynamicMatrix V = transposeMatrix(Vt);          // n×k
    DynamicMatrix S_mat(k, k);
    for (size_t i = 0; i < k; ++i) S_mat.set(i, i, S[i]);
    DynamicMatrix P_polar = V * (S_mat * Vt);       // n×k * k×k * k×n

    // Force symmetry (remove floating-point asymmetry)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            double sym = 0.5 * (P_polar.get(i, j) + P_polar.get(j, i));
            P_polar.set(i, j, sym);
            P_polar.set(j, i, sym);
        }

    return {U_polar, P_polar};
}

// ─── Schur decomposition (real symmetric) ────────────────────────────────────
// A = Q * T * Q^T, T diagonal (eigenvalues), Q orthogonal (eigenvectors)

std::pair<DynamicMatrix, DynamicMatrix>
schur(const AbstractMatrix& A, size_t max_iter)
{
    if (A.rows() != A.cols())
        throw std::invalid_argument("schur: requires square matrix");
    if (!isSymmetric(A))
        throw std::invalid_argument("schur: only real symmetric matrices are supported");

    auto [evals, Q] = eig(A, max_iter);

    size_t n = evals.size();
    DynamicMatrix T(n, n);
    for (size_t i = 0; i < n; ++i) T.set(i, i, evals[i]);

    return {Q, T};   // A = Q * T * Q^T
}

// ─── rsvd ─────────────────────────────────────────────────────────────────────
// Randomized SVD (Halko–Martinsson–Tropp 2011, Algorithm 4.3 + 4.4)
// Step 1: form a random sketch  Y = A * Omega  (m × l, l = k + oversampling)
// Step 2: power iteration       Y = (A * A^T)^q * Y
// Step 3: orthonormalise        Q = orth(Y)
// Step 4: project               B = Q^T * A          (l × n)
// Step 5: thin SVD of B         B = Ub * S * Vt
// Step 6: recover left vectors  U = Q * Ub           (m × l)

std::tuple<DynamicMatrix, std::vector<double>, DynamicMatrix>
rsvd(const AbstractMatrix& A_abs,
     size_t k,
     size_t n_oversampling,
     size_t n_power_iter)
{
    size_t m = A_abs.rows(), n = A_abs.cols();
    if (k == 0 || k > std::min(m, n))
        throw std::invalid_argument("rsvd: k must be in [1, min(m,n)]");

    DynamicMatrix A = toDynamic(A_abs);
    size_t l = std::min(k + n_oversampling, std::min(m, n));

    // --- Step 1: random Gaussian test matrix Omega (n × l)
    std::mt19937_64 rng(20240101ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    DynamicMatrix Omega(n, l);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < l; ++j)
            Omega(i, j) = nd(rng);

    // Y = A * Omega  (m × l)
    DynamicMatrix Y = A * Omega;

    // --- Step 2: power iteration  Y = (A A^T)^q Y
    DynamicMatrix At = A.transposed();   // n × m
    for (size_t iter = 0; iter < n_power_iter; ++iter) {
        // Y = A * (A^T * Y)
        DynamicMatrix Z = At * Y;   // n × l
        Y = A * Z;                  // m × l
    }

    // --- Step 3: orthonormalise Y via QR
    auto [Q_big, R_big] = qr(Y);   // Q_big: m × m
    // Keep only first l columns
    DynamicMatrix Q(m, l);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < l; ++j)
            Q(i, j) = Q_big(i, j);

    // --- Step 4: B = Q^T * A  (l × n)
    DynamicMatrix Qt = Q.transposed();  // l × m
    DynamicMatrix B  = Qt * A;          // l × n

    // --- Step 5: thin SVD of B
    auto [Ub, S_full, Vt] = svd(B);    // Ub: l×l, S_full: l, Vt: l×n

    // --- Step 6: U = Q * Ub  (m × l), keep first k
    DynamicMatrix Ub_k(l, k);
    for (size_t i = 0; i < l; ++i)
        for (size_t j = 0; j < k; ++j)
            Ub_k(i, j) = Ub(i, j);

    DynamicMatrix U = Q * Ub_k;  // m × k

    std::vector<double> S(S_full.begin(), S_full.begin() + k);

    DynamicMatrix Vt_k(k, n);
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n; ++j)
            Vt_k(i, j) = Vt(i, j);

    return {U, S, Vt_k};
}

// ─── Structured matrix generators ────────────────────────────────────────────

DynamicMatrix toeplitz(const std::vector<double>& c, const std::vector<double>& r_in)
{
    const std::vector<double>& r = r_in.empty() ? c : r_in;
    size_t m = c.size(), n = r.size();
    DynamicMatrix M(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            if (i <= j) M(i, j) = r[j - i];       // upper: use first row
            else        M(i, j) = c[i - j];        // lower: use first col
        }
    return M;
}

DynamicMatrix circulant(const std::vector<double>& c)
{
    size_t n = c.size();
    DynamicMatrix M(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            M(i, j) = c[(j >= i) ? j - i : n - (i - j)];
    return M;
}

DynamicMatrix hankel(const std::vector<double>& c, const std::vector<double>& r_in)
{
    size_t m = c.size();
    std::vector<double> r = r_in.empty() ? std::vector<double>(m, 0.0) : r_in;
    size_t n = r.size();
    DynamicMatrix M(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            size_t idx = i + j;
            if (idx < m)      M(i, j) = c[idx];
            else if (idx < m + n - 1) M(i, j) = r[idx - m + 1];
            else              M(i, j) = 0.0;
        }
    return M;
}

DynamicMatrix vandermonde(const std::vector<double>& x, bool increasing)
{
    size_t m = x.size(), n = m;
    DynamicMatrix M(m, n);
    for (size_t i = 0; i < m; ++i) {
        double xi = x[i];
        for (size_t j = 0; j < n; ++j) {
            size_t exp = increasing ? j : (n - 1 - j);
            M(i, j) = std::pow(xi, static_cast<double>(exp));
        }
    }
    return M;
}

DynamicMatrix hilbert(size_t n)
{
    DynamicMatrix M(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            M(i, j) = 1.0 / static_cast<double>(i + j + 1);
    return M;
}

DynamicMatrix companion(const std::vector<double>& c)
{
    size_t n = c.size();
    if (n == 0)
        throw std::invalid_argument("companion: coefficient vector must be non-empty");
    DynamicMatrix M(n, n, 0.0);
    // Last column: negated polynomial coefficients (reversed)
    for (size_t i = 0; i < n; ++i)
        M(i, n - 1) = -c[n - 1 - i];
    // Sub-diagonal ones
    for (size_t i = 1; i < n; ++i)
        M(i, i - 1) = 1.0;
    return M;
}

// ─── random namespace ─────────────────────────────────────────────────────────

namespace random {

DynamicMatrix gaussian(size_t m, size_t n, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    DynamicMatrix M(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            M(i, j) = nd(rng);
    return M;
}

DynamicMatrix uniform(size_t m, size_t n, double lo, double hi, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ud(lo, hi);
    DynamicMatrix M(m, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            M(i, j) = ud(rng);
    return M;
}

DynamicMatrix orthogonal(size_t n, unsigned seed)
{
    DynamicMatrix G = gaussian(n, n, seed);
    auto [Q, R] = qr(G);
    // Make Q a proper orthogonal matrix: fix column signs so R has positive diagonal
    for (size_t j = 0; j < n; ++j)
        if (R(j, j) < 0.0)
            for (size_t i = 0; i < n; ++i) Q(i, j) = -Q(i, j);
    return Q;
}

DynamicMatrix spd(size_t n, double lo_ev, double hi_ev, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> evd(lo_ev, hi_ev);

    // Random orthogonal Q
    DynamicMatrix Q = orthogonal(n, seed);
    // Random eigenvalues
    std::vector<double> evals(n);
    for (double& v : evals) v = evd(rng);

    // A = Q * diag(evals) * Q^T
    DynamicMatrix D(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) D(i, i) = evals[i];
    DynamicMatrix Qt = Q.transposed();
    DynamicMatrix A  = Q * D * Qt;

    // Enforce exact symmetry
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            double s = 0.5 * (A(i, j) + A(j, i));
            A(i, j) = A(j, i) = s;
        }
    return A;
}

DynamicMatrix sparse(size_t m, size_t n, double density, unsigned seed)
{
    if (density <= 0.0 || density > 1.0)
        throw std::invalid_argument("random::sparse: density must be in (0,1]");
    std::mt19937_64 rng(seed);
    std::normal_distribution<double>   nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    DynamicMatrix M(m, n, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            if (ud(rng) < density) M(i, j) = nd(rng);
    return M;
}

} // namespace random

// ─── geig ─────────────────────────────────────────────────────────────────────
// Generalized symmetric-definite eigenproblem: A v = λ B v
// Reduction: L^{-1} A L^{-T} w = λ w  where B = L L^T (Cholesky)
// Then transform back: v = L^{-T} w

std::pair<std::vector<double>, DynamicMatrix>
geig(const AbstractMatrix& A, const AbstractMatrix& B, size_t max_iter)
{
    if (A.rows() != A.cols() || B.rows() != B.cols() || A.rows() != B.rows())
        throw std::invalid_argument("geig: A and B must be square and the same size");
    if (!isSymmetric(A) || !isSymmetric(B))
        throw std::invalid_argument("geig: A and B must be symmetric");

    size_t n = A.rows();
    // Cholesky of B = L * L^T
    DynamicMatrix L  = cholesky(B);   // lower triangular
    DynamicMatrix Lt = L.transposed();// upper triangular

    // Compute L^{-1} via forward substitution column by column
    // Solve L * X = I
    DynamicMatrix Linv(n, n, 0.0);
    for (size_t j = 0; j < n; ++j) {
        // Forward substitution for column j of identity
        for (size_t i = j; i < n; ++i) {
            double s = (i == j) ? 1.0 : 0.0;
            for (size_t k = j; k < i; ++k) s -= L(i, k) * Linv(k, j);
            Linv(i, j) = s / L(i, i);
        }
    }

    // C = L^{-1} * A * L^{-T}  (symmetric)
    DynamicMatrix LinvT = Linv.transposed();
    DynamicMatrix Ad    = toDynamic(A);
    DynamicMatrix C     = Linv * Ad * LinvT;

    // Force exact symmetry
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j) {
            double s = 0.5 * (C(i, j) + C(j, i));
            C(i, j) = C(j, i) = s;
        }

    // Standard symmetric eigenproblem on C
    auto [evals, W] = eig(C, max_iter);

    // Reverse to get ascending order
    std::reverse(evals.begin(), evals.end());

    DynamicMatrix W_asc(n, n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            W_asc(i, j) = W(i, n - 1 - j);

    // Transform back: V = L^{-T} * W_asc
    DynamicMatrix V = LinvT * W_asc;

    // B-orthonormalise columns of V: V[:,j] /= sqrt(V[:,j]^T B V[:,j])
    DynamicMatrix Bd = toDynamic(B);
    for (size_t j = 0; j < n; ++j) {
        // BV = B * V[:,j]
        double nrm2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double bvi = 0.0;
            for (size_t k = 0; k < n; ++k) bvi += Bd(i, k) * V(k, j);
            nrm2 += V(i, j) * bvi;
        }
        double nrm = std::sqrt(std::max(0.0, nrm2));
        if (nrm > 1e-14)
            for (size_t i = 0; i < n; ++i) V(i, j) /= nrm;
    }

    return {evals, V};
}

// ─── lyapunov ─────────────────────────────────────────────────────────────────
// Solve A X + X A^T + C = 0 via the Kronecker product reformulation:
// (I⊗A + A⊗I) vec(X) = -vec(C)
// This is O(n^6) for the direct solve — practical only for small n.
// For production use, implement Bartels–Stewart via Schur decomposition.

DynamicMatrix lyapunov(const AbstractMatrix& A_abs, const AbstractMatrix& C_abs,
                        double /*tol*/, size_t /*max_iter*/)
{
    if (A_abs.rows() != A_abs.cols())
        throw std::invalid_argument("lyapunov: A must be square");
    if (C_abs.rows() != A_abs.rows() || C_abs.cols() != A_abs.cols())
        throw std::invalid_argument("lyapunov: C must have the same shape as A");

    size_t n  = A_abs.rows();
    size_t n2 = n * n;
    DynamicMatrix A = toDynamic(A_abs);
    DynamicMatrix C = toDynamic(C_abs);

    // Build coefficient matrix M = I⊗A + A⊗I  (n²×n²)
    // (I⊗A)[row, col]: row = i*n+k, col = i*n+l → A[k,l]
    // (A⊗I)[row, col]: row = i*n+k, col = j*n+k → A[i,j]
    DynamicMatrix M(n2, n2, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < n; ++k) {
            size_t row = i * n + k;
            // I⊗A: column l in block i
            for (size_t l = 0; l < n; ++l)
                M(row, i * n + l) += A(k, l);
            // A⊗I: block j contributes A[i,j]*I
            for (size_t j = 0; j < n; ++j)
                M(row, j * n + k) += A(i, j);
        }

    // RHS: -vec(C)  (column-major vec)
    std::vector<double> rhs(n2);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            rhs[j * n + i] = -C(i, j);

    // Solve via LU
    std::vector<double> x = solve(M, rhs);

    // Reshape vec(X) back to X (column-major)
    DynamicMatrix X(n, n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            X(i, j) = x[j * n + i];

    return X;
}

// ─── sylvester ────────────────────────────────────────────────────────────────
// Solve A X + X B = C  via Kronecker product:
// (I⊗A + B^T⊗I) vec(X) = vec(C)

DynamicMatrix sylvester(const AbstractMatrix& A_abs,
                         const AbstractMatrix& B_abs,
                         const AbstractMatrix& C_abs)
{
    size_t m = A_abs.rows(), n = B_abs.cols();
    if (A_abs.cols() != m)
        throw std::invalid_argument("sylvester: A must be square");
    if (B_abs.rows() != n)
        throw std::invalid_argument("sylvester: B must be square");
    if (C_abs.rows() != m || C_abs.cols() != n)
        throw std::invalid_argument("sylvester: C must be m×n");

    DynamicMatrix A = toDynamic(A_abs);
    DynamicMatrix B = toDynamic(B_abs);
    DynamicMatrix C = toDynamic(C_abs);

    size_t mn = m * n;
    // M = I_n⊗A + B^T⊗I_m   (mn × mn)
    // (I_n⊗A)[r,c]: r=j*m+i, c=j*m+k → A[i,k]
    // (B^T⊗I_m)[r,c]: r=j*m+i, c=l*m+i → B^T[j,l] = B[l,j]
    DynamicMatrix M(mn, mn, 0.0);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i) {
            size_t row = j * m + i;
            // I_n⊗A block
            for (size_t k = 0; k < m; ++k)
                M(row, j * m + k) += A(i, k);
            // B^T⊗I_m block
            for (size_t l = 0; l < n; ++l)
                M(row, l * m + i) += B(l, j);  // B^T[j,l] = B[l,j]
        }

    // RHS: vec(C) in column-major order
    std::vector<double> rhs(mn);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            rhs[j * m + i] = C(i, j);

    std::vector<double> x = solve(M, rhs);

    DynamicMatrix X(m, n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            X(i, j) = x[j * m + i];
    return X;
}

// ─── gsvd ─────────────────────────────────────────────────────────────────────
// Generalized SVD of (A, B) via CS decomposition of the joint matrix [A; B].
// A: m×n, B: p×n  →  A = U * diag(alpha) * X^{-T}
//                     B = V * diag(beta)  * X^{-T}
// where alpha_i^2 + beta_i^2 = 1.
//
// Algorithm:
//   1. Stack K = [A; B]  ((m+p)×n)
//   2. Thin QR: K = Q * R  ((m+p)×n upper triangular, if m+p >= n)
//   3. Thin SVD of R: R = Ur * S_r * Vt  (X^{-T} = Vt, so X = Vt^T * diag(1/s))
//   4. Partition Q into Q_A (m×n) and Q_B (p×n)
//   5. Thin SVD of Q_A gives alpha, U; thin SVD of Q_B gives beta, V via
//      the coupling constraint alpha_i^2 + beta_i^2 = 1.
//
// For simplicity, use the direct CS approach:
//   thin SVD of Q_A = Ua * Sa * Wt
//   alpha = Sa (truncated to k = min(m,p,n) values)
//   U = Ua
//   beta = sqrt(1 - alpha^2)
//   V = Q_B * Wt^T / beta  (with care for zero beta)

GSVDResult gsvd(const AbstractMatrix& A_abs, const AbstractMatrix& B_abs)
{
    size_t m = A_abs.rows(), n = A_abs.cols();
    size_t p = B_abs.rows();
    if (B_abs.cols() != n)
        throw std::invalid_argument("gsvd: A and B must have the same number of columns");

    DynamicMatrix A = toDynamic(A_abs);
    DynamicMatrix B = toDynamic(B_abs);

    // Stack K = [A; B]
    DynamicMatrix K(m + p, n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) K(i, j)     = A(i, j);
    for (size_t i = 0; i < p; ++i)
        for (size_t j = 0; j < n; ++j) K(m + i, j) = B(i, j);

    // Thin QR of K: K = Q_full * R  (Q_full: (m+p)×(m+p), R: (m+p)×n)
    auto [Q_full, R_full] = qr(K);  // Q is (m+p)×(m+p)

    // Take thin part: Q_thin = Q_full[:, 0:n] if m+p > n, else full Q
    size_t qn = std::min(m + p, n);

    DynamicMatrix Q_A(m, qn), Q_B(p, qn);
    for (size_t i = 0; i < m;  ++i)
        for (size_t j = 0; j < qn; ++j) Q_A(i, j) = Q_full(i,     j);
    for (size_t i = 0; i < p;  ++i)
        for (size_t j = 0; j < qn; ++j) Q_B(i, j) = Q_full(m + i, j);

    // Thin SVD of R: R = Ur * Sr * Vt
    // X^{-T} = Vt → X = Vt^T * diag(1/Sr)
    DynamicMatrix R_thin(qn, n);
    for (size_t i = 0; i < qn; ++i)
        for (size_t j = 0; j < n; ++j) R_thin(i, j) = R_full(i, j);

    auto [Ur, Sr, Vt] = svd(R_thin);  // Sr: qn singular values

    // X = Vt^T * diag(1/Sr)   (n × qn)
    DynamicMatrix VtT = Vt.transposed();  // n × qn
    DynamicMatrix X(n, qn);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < qn; ++j)
            X(i, j) = (Sr[j] > 1e-14) ? VtT(i, j) / Sr[j] : 0.0;

    // SVD of Q_A * Ur  (m × qn)
    DynamicMatrix QAUr = Q_A * Ur;
    auto [U_cs, Sa, Wt] = svd(QAUr);   // Sa: cosines

    size_t k = std::min({m, p, qn});
    std::vector<double> alpha(k), beta(k);
    for (size_t i = 0; i < k; ++i) {
        alpha[i] = (i < Sa.size()) ? std::min(1.0, std::max(0.0, Sa[i])) : 0.0;
        beta[i]  = std::sqrt(std::max(0.0, 1.0 - alpha[i] * alpha[i]));
    }

    // U: first k columns of U_cs
    DynamicMatrix U(m, k);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < k; ++j) U(i, j) = U_cs(i, j);

    // V = Q_B * Ur * Wt^T, normalised by beta
    DynamicMatrix WtT = Wt.transposed();  // qn × k (taking first k cols)
    DynamicMatrix QBUr = Q_B * Ur;        // p × qn
    DynamicMatrix V_full = QBUr * WtT;    // p × qn → take first k cols

    DynamicMatrix V(p, k);
    for (size_t i = 0; i < p; ++i)
        for (size_t j = 0; j < k; ++j) {
            double b = beta[j];
            V(i, j) = (b > 1e-14) ? V_full(i, j) / b : V_full(i, j);
        }

    // X_out: first k columns of X
    DynamicMatrix X_out(n, k);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < k; ++j) X_out(i, j) = X(i, j);

    return {U, V, X_out, alpha, beta};
}

// ─── cuda_is_available ────────────────────────────────────────────────────────
// CPU-only build: always false.
// CUDA build: the real implementation is in TensorCUDA.cu and overrides this
// via weak linking on Linux/macOS or a separate definition on Windows.
// We define the CPU stub here so the linker always has at least one definition.
#ifndef SHAREDMATH_CUDA
bool cuda_is_available() noexcept { return false; }
#endif

} // namespace SharedMath::LinearAlgebra
