#pragma once

#include "AbstractMatrix.h"
#include "DynamicMatrix.h"
#include "DynamicVector.h"
#include <sharedmath_linearalgebra_export.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace SharedMath::LinearAlgebra {

/// Sparse matrix in Compressed Sparse Row (CSR) format.
///
/// Memory layout:
///   values_[k]      — nonzero value of the k-th stored entry
///   col_idx_[k]     — column index of the k-th stored entry
///   row_ptr_[i]     — index into values_/col_idx_ of the first entry in row i
///   row_ptr_[rows_] — total number of stored entries (== nnz())
///
/// All entries within a row are stored in ascending column order.
/// Duplicate (i,j) entries in from_triplets() are summed automatically.
///
class SHAREDMATH_LINEARALGEBRA_EXPORT SparseMatrix : public AbstractMatrix {
public:
    /// ── Construction ──────────────────────────────────────────────────────

    /// Empty matrix with given dimensions (zero non-zeros)
    SparseMatrix(size_t rows, size_t cols);

    // Build from COO triplets.  Duplicate (i,j) entries are summed.
    // Entries are sorted row-major then column-ascending inside each row.
    static SparseMatrix from_triplets(size_t rows, size_t cols,
                                      const std::vector<size_t>& row_idx,
                                      const std::vector<size_t>& col_idx,
                                      const std::vector<double>& values);

    /// Convert dense matrix to sparse (drop entries with |v| <= tol).
    static SparseMatrix from_dense(const AbstractMatrix& A, double tol = 0.0);

    /// Diagonal sparse matrix from vector
    static SparseMatrix diag(const std::vector<double>& d);

    /// n×n identity
    static SparseMatrix eye(size_t n);

    SparseMatrix(const SparseMatrix&)            = default;
    SparseMatrix(SparseMatrix&&) noexcept        = default;
    SparseMatrix& operator=(const SparseMatrix&) = default;
    SparseMatrix& operator=(SparseMatrix&&) noexcept = default;
    ~SparseMatrix() override                     = default;

    /// ── AbstractMatrix interface ──────────────────────────────────────────
    size_t rows() const noexcept override { return rows_; }
    size_t cols() const noexcept override { return cols_; }

    /// O(log nnz_per_row) via binary search in row
    double  get(size_t r, size_t c) const override;
    double& get(size_t r, size_t c)       override;
    void    set(size_t r, size_t c, double v) override;

    double* toPtr()       noexcept override { return values_.data(); }
    const double* toPtr() const noexcept override { return values_.data(); }

    /// ── CSR storage accessors ─────────────────────────────────────────────
    const std::vector<double>& values()    const noexcept { return values_;  }
    const std::vector<size_t>& col_indices() const noexcept { return col_idx_; }
    const std::vector<size_t>& row_ptr()   const noexcept { return row_ptr_; }

    size_t nnz()     const noexcept { return values_.size(); }
    double density() const noexcept {
        size_t tot = rows_ * cols_;
        return (tot == 0) ? 0.0 : static_cast<double>(nnz()) / tot;
    }
    bool isSquare() const noexcept { return rows_ == cols_; }

    // ── Arithmetic ────────────────────────────────────────────────────────

    // Sparse matrix–vector multiply: y = A * x
    DynamicVector operator*(const DynamicVector& x) const;

    // Sparse matrix–dense matrix multiply: C = A * B (result is dense)
    DynamicMatrix operator*(const DynamicMatrix& B) const;

    // Element-wise (same sparsity pattern union)
    SparseMatrix operator+(const SparseMatrix& o) const;
    SparseMatrix operator-(const SparseMatrix& o) const;
    SparseMatrix operator*(double s) const;
    SparseMatrix operator/(double s) const;

    friend SparseMatrix operator*(double s, const SparseMatrix& m) { return m * s; }

    /// Sparse–sparse matrix multiply
    SparseMatrix matmul(const SparseMatrix& B) const;

    /// ── Transpose ─────────────────────────────────────────────────────────
    SparseMatrix transposed() const;

    /// ── Conversion ────────────────────────────────────────────────────────
    DynamicMatrix to_dense() const;

    /// ── Display ───────────────────────────────────────────────────────────
    std::string str() const;

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<double> values_;   // nonzero values
    std::vector<size_t> col_idx_;  // column indices  (size == nnz)
    std::vector<size_t> row_ptr_;  // row pointers    (size == rows+1)

    /// Find the index into values_/col_idx_ for entry (r, c).
    /// Returns values_.size() if not found (structural zero).
    size_t find(size_t r, size_t c) const noexcept;

    // Helper for binary operations: merge two sorted rows
    static void merge_rows(
        const std::vector<double>& av, const std::vector<size_t>& ac,
        size_t a0, size_t a1,
        const std::vector<double>& bv, const std::vector<size_t>& bc,
        size_t b0, size_t b1,
        double alpha, double beta,
        std::vector<double>& rv, std::vector<size_t>& rc);
};

} // namespace SharedMath::LinearAlgebra
