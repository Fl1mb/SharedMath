#pragma once

#include "VectorN.h"
#include "AbstractMatrix.h"

#include <array>
#include <memory>
#include <stdexcept>

namespace SharedMath::LinearAlgebra {

// Compile-time fixed-size matrix.
//
// Storage: std::array<Vector<Cols>, Rows>.
// Since Vector<Cols> is standard-layout (sole member std::array<double,Cols>),
// the whole matrix is Rows*Cols doubles laid out contiguously, with no padding.
// toPtr() / rowPtr() return correct flat pointers.
//
// Prefer operator()(r, c) over operator[](r)[c] for direct access;
// the latter is kept for backward compatibility.
//
template<size_t Rows, size_t Cols>
class Matrix : public AbstractMatrix {
    static_assert(Rows > 0 && Cols > 0, "Matrix dimensions must be > 0");
public:
    /// ── Construction ──────────────────────────────────────────────────────
    Matrix() noexcept { for (auto& row : data_) row = Vector<Cols>(); }

    Matrix(std::initializer_list<std::initializer_list<double>> values) {
        if (values.size() != Rows)
            throw std::invalid_argument("Matrix: wrong number of rows in initializer list");
        size_t i = 0;
        for (const auto& row : values) {
            if (row.size() != Cols)
                throw std::invalid_argument("Matrix: wrong number of columns in initializer list");
            size_t j = 0;
            for (double v : row) data_[i][j++] = v;
            ++i;
        }
    }

    Matrix(const Matrix&)            = default;
    Matrix(Matrix&&) noexcept        = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix() override = default;

    /// ── Element access ────────────────────────────────────────────────────

    /// Preferred: direct (row, col) access
    double& operator()(size_t r, size_t c) {
        if (r >= Rows || c >= Cols)
            throw std::out_of_range("Matrix: index out of range");
        return data_[r][c];
    }
    double operator()(size_t r, size_t c) const {
        if (r >= Rows || c >= Cols)
            throw std::out_of_range("Matrix: index out of range");
        return data_[r][c];
    }

    // Row access (backward-compatible: matrix[r][c])
    Vector<Cols>& operator[](size_t r) noexcept { return data_[r]; }
    const Vector<Cols>& operator[](size_t r) const noexcept { return data_[r]; }

    /// AbstractMatrix interface
    double  get(size_t r, size_t c) const override { return (*this)(r, c); }
    double& get(size_t r, size_t c)       override { return (*this)(r, c); }
    void    set(size_t r, size_t c, double v) override { (*this)(r, c) = v; }

    /// Contiguous flat pointer to all Rows*Cols elements in row-major order.
    /// Safe: Vector<Cols> is standard-layout, so &data_[0][0] is the first double
    /// and the layout is identical to double[Rows][Cols].
    double*       toPtr()       override { return data_[0].data(); }
    const double* toPtr() const override { return data_[0].data(); }

    /// Pointer to the start of row r
    double*       rowPtr(size_t r) {
        if (r >= Rows) throw std::out_of_range("Matrix::rowPtr: row out of range");
        return data_[r].data();
    }
    const double* rowPtr(size_t r) const {
        if (r >= Rows) throw std::out_of_range("Matrix::rowPtr: row out of range");
        return data_[r].data();
    }

    /// ── Metadata ──────────────────────────────────────────────────────────
    size_t rows() const noexcept override { return Rows; }
    size_t cols() const noexcept override { return Cols; }
    static constexpr size_t totalElements() noexcept { return Rows * Cols; }
    static constexpr size_t dataSizeBytes() noexcept { return Rows * Cols * sizeof(double); }
    static constexpr bool   isContiguous()  noexcept { return true; }

    // ── Arithmetic ────────────────────────────────────────────────────────
    Matrix operator+(const Matrix& o) const noexcept {
        Matrix r;
        for (size_t i = 0; i < Rows; ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    Matrix operator-(const Matrix& o) const noexcept {
        Matrix r;
        for (size_t i = 0; i < Rows; ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }
    Matrix operator*(double s) const noexcept {
        Matrix r;
        for (size_t i = 0; i < Rows; ++i) r.data_[i] = data_[i] * s;
        return r;
    }
    friend Matrix operator*(double s, const Matrix& m) noexcept { return m * s; }

    Matrix& operator+=(const Matrix& o) noexcept {
        for (size_t i = 0; i < Rows; ++i) data_[i] += o.data_[i];
        return *this;
    }
    Matrix& operator-=(const Matrix& o) noexcept {
        for (size_t i = 0; i < Rows; ++i) data_[i] -= o.data_[i];
        return *this;
    }
    Matrix& operator*=(double s) noexcept {
        for (auto& row : data_) row *= s;
        return *this;
    }

    // Matrix * vector (M × N) * (N) = (M)
    Vector<Rows> operator*(const Vector<Cols>& v) const noexcept {
        Vector<Rows> result;
        for (size_t i = 0; i < Rows; ++i)
            result[i] = data_[i].dot(v);
        return result;
    }

    // Matrix * matrix: (Rows × Cols) * (Cols × Other) = (Rows × Other)
    // Cache-friendly i-k-j loop.
    template<size_t Other>
    Matrix<Rows, Other> mul(const Matrix<Cols, Other>& B) const noexcept {
        Matrix<Rows, Other> C;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t k = 0; k < Cols; ++k) {
                double a = data_[i][k];
                for (size_t j = 0; j < Other; ++j)
                    C[i][j] = C[i][j] + a * B[k][j];
            }
        }
        return C;
    }

    // ── Flat array conversions ────────────────────────────────────────────
    std::array<double, Rows * Cols> toRowMajorArray() const {
        std::array<double, Rows * Cols> out;
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                out[i * Cols + j] = data_[i][j];
        return out;
    }
    std::array<double, Rows * Cols> toColumnMajorArray() const {
        std::array<double, Rows * Cols> out;
        for (size_t j = 0; j < Cols; ++j)
            for (size_t i = 0; i < Rows; ++i)
                out[j * Rows + i] = data_[i][j];
        return out;
    }
    void fromRowMajorArray(const double* ptr) {
        for (size_t i = 0; i < Rows; ++i)
            for (size_t j = 0; j < Cols; ++j)
                data_[i][j] = ptr[i * Cols + j];
    }
    void fromColumnMajorArray(const double* ptr) {
        for (size_t j = 0; j < Cols; ++j)
            for (size_t i = 0; i < Rows; ++i)
                data_[i][j] = ptr[j * Rows + i];
    }

    /// ── Comma initializer  (mat << 1, 2, 3, ...) ─────────────────────────
    class CommaInitializer {
    public:
        explicit CommaInitializer(Matrix& mat) : mat_(mat) {
            for (auto& row : mat_.data_) row = Vector<Cols>();
        }

        CommaInitializer& operator,(double v) {
            if (row_ >= Rows)
                throw std::out_of_range("Matrix comma initializer: too many values");
            mat_.data_[row_][col_] = v;
            if (++col_ >= Cols) { col_ = 0; ++row_; }
            return *this;
        }

        Matrix& finalize() {
            if (row_ != Rows || col_ != 0)
                throw std::runtime_error(
                    "Matrix comma initializer: wrong number of values");
            return mat_;
        }
        operator Matrix&() { return finalize(); }

    private:
        Matrix& mat_;
        size_t  row_ = 0, col_ = 0;
    };

    CommaInitializer operator<<(double first) {
        CommaInitializer init(*this);
        init, first;
        return init;
    }

private:
    std::array<Vector<Cols>, Rows> data_;
};

} // namespace SharedMath::LinearAlgebra
