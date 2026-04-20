#include "MatrixFunctions.h"

#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include <map>
#include <string>
#include <sstream>

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
        // Wilkinson shift using the bottom-right 2×2 submatrix
        double shift = 0.0;
        if (n >= 2) {
            double d  = (Ak.get(n-2, n-2) - Ak.get(n-1, n-1)) / 2.0;
            double b  = Ak.get(n-1, n-2);
            double sq = std::sqrt(d * d + b * b);
            shift = Ak.get(n-1, n-1) - (b * b) / (d + (d >= 0 ? sq : -sq));
        }

        // A_shifted = Ak - shift * I
        for (size_t i = 0; i < n; ++i)
            Ak.set(i, i, Ak.get(i, i) - shift);

        auto [Q, R] = qr(Ak);

        // Ak = R * Q + shift * I
        Ak = R * Q;
        for (size_t i = 0; i < n; ++i)
            Ak.set(i, i, Ak.get(i, i) + shift);

        // Accumulate eigenvectors: V = V * Q
        V = V * Q;

        // Convergence: off-diagonal Frobenius norm
        double off = 0.0;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                if (i != j) off += Ak.get(i, j) * Ak.get(i, j);
        if (std::sqrt(off) < 1e-10 * n) break;
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

} // namespace SharedMath::LinearAlgebra
