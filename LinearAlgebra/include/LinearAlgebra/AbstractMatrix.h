#pragma once

#include <cstddef>

namespace SharedMath::LinearAlgebra {

// Abstract interface for all matrix types in the library.
// Provides uniform element access and raw-pointer export so algorithms
// can operate on any concrete implementation (DynamicMatrix, Matrix<R,C>, …).
class AbstractMatrix {
public:
    virtual ~AbstractMatrix() = default;

    virtual size_t rows() const = 0;
    virtual size_t cols() const = 0;

    // Element access — bounds checking is the responsibility of the derived class
    virtual double  get(size_t row, size_t col) const = 0;
    virtual double& get(size_t row, size_t col)       = 0;
    virtual void    set(size_t row, size_t col, double val) = 0;

    // Raw pointer to the first element in row-major order.
    // Implementations MUST guarantee that all rows*cols elements are contiguous.
    virtual double*       toPtr()       = 0;
    virtual const double* toPtr() const = 0;
};

} // namespace SharedMath::LinearAlgebra
