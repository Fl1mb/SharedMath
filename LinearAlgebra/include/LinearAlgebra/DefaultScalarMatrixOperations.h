#pragma once

#include "MatrixOperationsStrategy.h"
#include "DynamicMatrix.h"
#include "LUDecomposition.h"

namespace SharedMath::LinearAlgebra {

class MatrixDeterminantStrategy : public ScalarMatrixOperationStrategy {
public:
    ~MatrixDeterminantStrategy() override = default;

    double execute(MatrixPtr A) override {
        if (!isSupported(*A))
            throw std::invalid_argument(
                "MatrixDeterminantStrategy: matrix must be square");
        // LUDecomposition requires DynamicMatrix; copy through virtual interface
        DynamicMatrix M(*A);
        LUDecomposition lu(M);
        try {
            lu.MakeDecomposition();
        } catch (const std::runtime_error&) {
            // MakeDecomposition throws when the matrix is singular.
            // A singular matrix has determinant 0 — return it, don't rethrow.
            return 0.0;
        }
        return lu.Determinant();
    }

    bool isSupported(const AbstractMatrix& A) const override {
        return A.rows() == A.cols();
    }
};

class MatrixTraceStrategy : public ScalarMatrixOperationStrategy {
public:
    ~MatrixTraceStrategy() override = default;

    double execute(MatrixPtr A) override {
        if (!isSupported(*A))
            throw std::invalid_argument(
                "MatrixTraceStrategy: matrix must be square");
        double result = 0.0;
        for (size_t i = 0; i < A->rows(); ++i)
            result += A->get(i, i);
        return result;
    }

    bool isSupported(const AbstractMatrix& A) const override {
        return A.rows() == A.cols();
    }
};

} // namespace SharedMath::LinearAlgebra
