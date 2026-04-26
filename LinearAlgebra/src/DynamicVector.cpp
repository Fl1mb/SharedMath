#include "LinearAlgebra/DynamicVector.h"
#include "LinearAlgebra/DynamicMatrix.h"

#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace SharedMath::LinearAlgebra {

// ── to_column / to_row ────────────────────────────────────────────────────────

DynamicMatrix DynamicVector::to_column() const {
    // Returns an (n×1) DynamicMatrix
    DynamicMatrix M(data_.size(), 1);
    for (size_t i = 0; i < data_.size(); ++i) M(i, 0) = data_[i];
    return M;
}

DynamicMatrix DynamicVector::to_row() const {
    // Returns a (1×n) DynamicMatrix
    DynamicMatrix M(1, data_.size());
    for (size_t j = 0; j < data_.size(); ++j) M(0, j) = data_[j];
    return M;
}

DynamicVector DynamicVector::from_column(const DynamicMatrix& A) {
    if (A.cols() != 1)
        throw std::invalid_argument(
            "DynamicVector::from_column: matrix must have exactly 1 column, got " +
            std::to_string(A.cols()));
    DynamicVector v(A.rows());
    for (size_t i = 0; i < A.rows(); ++i) v[i] = A(i, 0);
    return v;
}

DynamicVector DynamicVector::from_row(const DynamicMatrix& A) {
    if (A.rows() != 1)
        throw std::invalid_argument(
            "DynamicVector::from_row: matrix must have exactly 1 row, got " +
            std::to_string(A.rows()));
    DynamicVector v(A.cols());
    for (size_t j = 0; j < A.cols(); ++j) v[j] = A(0, j);
    return v;
}

// ── Display ───────────────────────────────────────────────────────────────────

std::ostream& operator<<(std::ostream& os, const DynamicVector& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << v[i];
    }
    return os << "]";
}

// ── Matrix–vector free functions ──────────────────────────────────────────────

// y = A * x  (m = rows(A), n = cols(A))
DynamicVector matvec(const DynamicMatrix& A, const DynamicVector& x) {
    if (A.cols() != x.size())
        throw std::invalid_argument(
            "matvec: dimension mismatch — A.cols()=" + std::to_string(A.cols()) +
            " vs x.size()=" + std::to_string(x.size()));
    DynamicVector y(A.rows(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        const double* Ai = A.row_ptr(i);
        double s = 0.0;
        for (size_t j = 0; j < A.cols(); ++j) s += Ai[j] * x[j];
        y[i] = s;
    }
    return y;
}

// y = A^T * x  (n = cols(A))
DynamicVector rmatvec(const DynamicMatrix& A, const DynamicVector& x) {
    if (A.rows() != x.size())
        throw std::invalid_argument(
            "rmatvec: dimension mismatch — A.rows()=" + std::to_string(A.rows()) +
            " vs x.size()=" + std::to_string(x.size()));
    DynamicVector y(A.cols(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        const double* Ai = A.row_ptr(i);
        double xi = x[i];
        for (size_t j = 0; j < A.cols(); ++j) y[j] += Ai[j] * xi;
    }
    return y;
}

// Outer product: M[i][j] = u[i] * v[j]
DynamicMatrix outer(const DynamicVector& u, const DynamicVector& v) {
    DynamicMatrix M(u.size(), v.size());
    for (size_t i = 0; i < u.size(); ++i) {
        double* Mi = M.row_ptr(i);
        for (size_t j = 0; j < v.size(); ++j) Mi[j] = u[i] * v[j];
    }
    return M;
}

DynamicVector operator*(const DynamicMatrix& A, const DynamicVector& x) {
    return matvec(A, x);
}

DynamicVector operator*(const DynamicVector& x, const DynamicMatrix& A) {
    return rmatvec(A, x);
}

} // namespace SharedMath::LinearAlgebra
