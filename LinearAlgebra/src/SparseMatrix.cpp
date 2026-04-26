#include "LinearAlgebra/SparseMatrix.h"

#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <stdexcept>

namespace SharedMath::LinearAlgebra {

// ── Construction ──────────────────────────────────────────────────────────────

SparseMatrix::SparseMatrix(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0) {}

// ── from_triplets ─────────────────────────────────────────────────────────────

SparseMatrix SparseMatrix::from_triplets(
    size_t rows, size_t cols,
    const std::vector<size_t>& ri,
    const std::vector<size_t>& ci,
    const std::vector<double>& vals)
{
    if (ri.size() != ci.size() || ri.size() != vals.size())
        throw std::invalid_argument("SparseMatrix::from_triplets: inconsistent input sizes");

    // Sort entries row-major, then col-ascending
    size_t nnz = ri.size();
    std::vector<size_t> order(nnz);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return ri[a] != ri[b] ? ri[a] < ri[b] : ci[a] < ci[b];
    });

    SparseMatrix M(rows, cols);
    for (size_t k = 0; k < nnz; ++k) {
        size_t o = order[k];
        size_t r = ri[o], c = ci[o];
        double v = vals[o];
        if (r >= rows || c >= cols)
            throw std::out_of_range("SparseMatrix::from_triplets: index out of range");
        // Merge duplicate (r,c) entries
        if (!M.values_.empty() && M.col_idx_.back() == c &&
            M.row_ptr_[r] < M.values_.size()) {
            // Check if last entry is in the same row and same column
            // (possible because entries are sorted)
            if (M.values_.size() > M.row_ptr_[r] &&
                M.col_idx_[M.values_.size() - 1] == c) {
                M.values_.back() += v;
                continue;
            }
        }
        M.values_.push_back(v);
        M.col_idx_.push_back(c);
    }

    // Rebuild row pointers from scratch (simpler and correct)
    // Redo using a proper algorithm
    M.values_.clear();
    M.col_idx_.clear();
    M.row_ptr_.assign(rows + 1, 0);

    // Count entries per row (after dedup)
    // Build sorted triplets
    struct Entry { size_t r, c; double v; };
    std::vector<Entry> entries;
    entries.reserve(nnz);
    for (size_t k : order) {
        size_t r = ri[k], c = ci[k]; double v = vals[k];
        if (r >= rows || c >= cols)
            throw std::out_of_range("SparseMatrix::from_triplets: index out of range");
        if (!entries.empty() && entries.back().r == r && entries.back().c == c)
            entries.back().v += v;  // sum duplicates
        else
            entries.push_back({r, c, v});
    }

    // Fill CSR
    M.values_.resize(entries.size());
    M.col_idx_.resize(entries.size());
    for (size_t k = 0; k < entries.size(); ++k) {
        M.values_[k]  = entries[k].v;
        M.col_idx_[k] = entries[k].c;
        ++M.row_ptr_[entries[k].r + 1];
    }
    // Prefix sum for row_ptr
    for (size_t r = 0; r < rows; ++r)
        M.row_ptr_[r + 1] += M.row_ptr_[r];

    return M;
}

// ── from_dense ────────────────────────────────────────────────────────────────

SparseMatrix SparseMatrix::from_dense(const AbstractMatrix& A, double tol) {
    std::vector<size_t> ri, ci;
    std::vector<double> vals;
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j) {
            double v = A.get(i, j);
            if (std::abs(v) > tol) {
                ri.push_back(i); ci.push_back(j); vals.push_back(v);
            }
        }
    return from_triplets(A.rows(), A.cols(), ri, ci, vals);
}

// ── diag / eye ────────────────────────────────────────────────────────────────

SparseMatrix SparseMatrix::diag(const std::vector<double>& d) {
    size_t n = d.size();
    std::vector<size_t> ri(n), ci(n);
    std::iota(ri.begin(), ri.end(), 0);
    std::iota(ci.begin(), ci.end(), 0);
    return from_triplets(n, n, ri, ci, d);
}

SparseMatrix SparseMatrix::eye(size_t n) {
    return diag(std::vector<double>(n, 1.0));
}

// ── Element access ────────────────────────────────────────────────────────────

size_t SparseMatrix::find(size_t r, size_t c) const noexcept {
    size_t lo = row_ptr_[r], hi = row_ptr_[r + 1];
    // Binary search in sorted column indices
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (col_idx_[mid] == c) return mid;
        if (col_idx_[mid] < c)  lo = mid + 1;
        else                     hi = mid;
    }
    return values_.size(); // not found
}

double SparseMatrix::get(size_t r, size_t c) const {
    size_t idx = find(r, c);
    return (idx < values_.size()) ? values_[idx] : 0.0;
}

double& SparseMatrix::get(size_t r, size_t c) {
    size_t idx = find(r, c);
    if (idx >= values_.size())
        throw std::out_of_range(
            "SparseMatrix::get (mutable): structural zero at (" +
            std::to_string(r) + "," + std::to_string(c) + ")");
    return values_[idx];
}

void SparseMatrix::set(size_t r, size_t c, double v) {
    size_t idx = find(r, c);
    if (idx < values_.size()) {
        values_[idx] = v;
    } else {
        // Insert new structural nonzero — rebuild the row
        // Simple approach: go through row, insert in sorted position
        size_t pos = row_ptr_[r];
        while (pos < row_ptr_[r + 1] && col_idx_[pos] < c) ++pos;
        values_.insert(values_.begin() + pos, v);
        col_idx_.insert(col_idx_.begin() + pos, c);
        for (size_t rr = r + 1; rr <= rows_; ++rr) ++row_ptr_[rr];
    }
}

// ── SpMV ──────────────────────────────────────────────────────────────────────

DynamicVector SparseMatrix::operator*(const DynamicVector& x) const {
    if (x.size() != cols_)
        throw std::invalid_argument(
            "SparseMatrix * DynamicVector: dimension mismatch (" +
            std::to_string(cols_) + " vs " + std::to_string(x.size()) + ")");
    DynamicVector y(rows_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        double s = 0.0;
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
            s += values_[k] * x[col_idx_[k]];
        y[i] = s;
    }
    return y;
}

// ── Sparse × Dense ────────────────────────────────────────────────────────────

DynamicMatrix SparseMatrix::operator*(const DynamicMatrix& B) const {
    if (cols_ != B.rows())
        throw std::invalid_argument("SparseMatrix * DynamicMatrix: inner dim mismatch");
    DynamicMatrix C(rows_, B.cols(), 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        double* Ci = C.row_ptr(i);
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            double   aik = values_[k];
            size_t   col = col_idx_[k];
            const double* Bk = B.row_ptr(col);
            for (size_t j = 0; j < B.cols(); ++j) Ci[j] += aik * Bk[j];
        }
    }
    return C;
}

// ── Binary ops helper: merge two sorted rows ──────────────────────────────────

void SparseMatrix::merge_rows(
    const std::vector<double>& av, const std::vector<size_t>& ac,
    size_t a0, size_t a1,
    const std::vector<double>& bv, const std::vector<size_t>& bc,
    size_t b0, size_t b1,
    double alpha, double beta,
    std::vector<double>& rv, std::vector<size_t>& rc)
{
    size_t ai = a0, bi = b0;
    while (ai < a1 || bi < b1) {
        if (bi >= b1 || (ai < a1 && ac[ai] < bc[bi])) {
            rc.push_back(ac[ai]); rv.push_back(alpha * av[ai++]);
        } else if (ai >= a1 || (bi < b1 && bc[bi] < ac[ai])) {
            rc.push_back(bc[bi]); rv.push_back(beta  * bv[bi++]);
        } else { // same column
            double val = alpha * av[ai++] + beta * bv[bi++];
            if (val != 0.0) { rc.push_back(ac[ai - 1]); rv.push_back(val); }
        }
    }
}

// ── operator+ / operator- ─────────────────────────────────────────────────────

SparseMatrix SparseMatrix::operator+(const SparseMatrix& o) const {
    if (rows_ != o.rows_ || cols_ != o.cols_)
        throw std::invalid_argument("SparseMatrix::operator+: shape mismatch");
    SparseMatrix R(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        size_t a0 = row_ptr_[i],   a1 = row_ptr_[i + 1];
        size_t b0 = o.row_ptr_[i], b1 = o.row_ptr_[i + 1];
        merge_rows(values_, col_idx_, a0, a1,
                   o.values_, o.col_idx_, b0, b1,
                   1.0, 1.0, R.values_, R.col_idx_);
        R.row_ptr_[i + 1] = R.values_.size();
    }
    return R;
}

SparseMatrix SparseMatrix::operator-(const SparseMatrix& o) const {
    if (rows_ != o.rows_ || cols_ != o.cols_)
        throw std::invalid_argument("SparseMatrix::operator-: shape mismatch");
    SparseMatrix R(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        size_t a0 = row_ptr_[i],   a1 = row_ptr_[i + 1];
        size_t b0 = o.row_ptr_[i], b1 = o.row_ptr_[i + 1];
        merge_rows(values_, col_idx_, a0, a1,
                   o.values_, o.col_idx_, b0, b1,
                   1.0, -1.0, R.values_, R.col_idx_);
        R.row_ptr_[i + 1] = R.values_.size();
    }
    return R;
}

// ── Scalar ops ────────────────────────────────────────────────────────────────

SparseMatrix SparseMatrix::operator*(double s) const {
    SparseMatrix R = *this;
    for (double& v : R.values_) v *= s;
    return R;
}

SparseMatrix SparseMatrix::operator/(double s) const { return *this * (1.0 / s); }

// ── Sparse–sparse multiply ────────────────────────────────────────────────────

SparseMatrix SparseMatrix::matmul(const SparseMatrix& B) const {
    if (cols_ != B.rows_)
        throw std::invalid_argument("SparseMatrix::matmul: inner dim mismatch");
    // Row-by-row: C[i,:] = sum_k A[i,k] * B[k,:]
    std::vector<size_t> ri, ci;
    std::vector<double>  vals;
    std::vector<double>  acc(B.cols_, 0.0);  // dense accumulator for current row
    std::vector<size_t>  touched;

    for (size_t i = 0; i < rows_; ++i) {
        touched.clear();
        for (size_t ka = row_ptr_[i]; ka < row_ptr_[i + 1]; ++ka) {
            size_t k = col_idx_[ka];
            double aik = values_[ka];
            for (size_t kb = B.row_ptr_[k]; kb < B.row_ptr_[k + 1]; ++kb) {
                size_t j = B.col_idx_[kb];
                if (acc[j] == 0.0) touched.push_back(j);
                acc[j] += aik * B.values_[kb];
            }
        }
        std::sort(touched.begin(), touched.end());
        for (size_t j : touched) {
            if (acc[j] != 0.0) { ri.push_back(i); ci.push_back(j); vals.push_back(acc[j]); }
            acc[j] = 0.0;
        }
    }
    return from_triplets(rows_, B.cols_, ri, ci, vals);
}

// ── Transpose ─────────────────────────────────────────────────────────────────

SparseMatrix SparseMatrix::transposed() const {
    std::vector<size_t> ri, ci;
    std::vector<double>  vals;
    ri.reserve(values_.size());
    ci.reserve(values_.size());
    vals.reserve(values_.size());
    for (size_t i = 0; i < rows_; ++i)
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            ri.push_back(col_idx_[k]);
            ci.push_back(i);
            vals.push_back(values_[k]);
        }
    return from_triplets(cols_, rows_, ri, ci, vals);
}

// ── to_dense ──────────────────────────────────────────────────────────────────

DynamicMatrix SparseMatrix::to_dense() const {
    DynamicMatrix M(rows_, cols_, 0.0);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
            M(i, col_idx_[k]) = values_[k];
    return M;
}

// ── str ───────────────────────────────────────────────────────────────────────

std::string SparseMatrix::str() const {
    std::ostringstream oss;
    oss << "SparseMatrix(" << rows_ << "×" << cols_
        << ", nnz=" << nnz()
        << ", density=" << std::fixed << std::setprecision(4) << density() << ")\n";
    for (size_t i = 0; i < rows_; ++i)
        for (size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
            oss << "  (" << i << "," << col_idx_[k] << ") = " << values_[k] << "\n";
    return oss.str();
}

} // namespace SharedMath::LinearAlgebra
