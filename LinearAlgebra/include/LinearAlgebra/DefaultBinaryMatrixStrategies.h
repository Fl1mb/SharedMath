#pragma once

#include "MatrixOperationsStrategy.h"
#include "DynamicMatrix.h"

namespace SharedMath::LinearAlgebra {

class MatrixAdditionStrategy : public BinaryMatrixOperationStrategy {
public:
    ~MatrixAdditionStrategy() override = default;

    MatrixPtr execute(MatrixPtr A, MatrixPtr B) override {
        if (!isSupported(*A, *B))
            throw std::invalid_argument("MatrixAdditionStrategy: dimension mismatch");

        size_t m = A->rows(), n = A->cols();

        // Fast path: both operands are DynamicMatrix — use flat-vector ops
        auto* dA = dynamic_cast<DynamicMatrix*>(A.get());
        auto* dB = dynamic_cast<DynamicMatrix*>(B.get());
        if (dA && dB)
            return std::make_shared<DynamicMatrix>(*dA + *dB);

        // General path: go through virtual interface
        auto result = std::make_shared<DynamicMatrix>(m, n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*result)(i, j) = A->get(i, j) + B->get(i, j);
        return result;
    }

    bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const override {
        return A.rows() == B.rows() && A.cols() == B.cols();
    }
};

class MatrixSubstractionStrategy : public BinaryMatrixOperationStrategy {
public:
    ~MatrixSubstractionStrategy() override = default;

    MatrixPtr execute(MatrixPtr A, MatrixPtr B) override {
        if (!isSupported(*A, *B))
            throw std::invalid_argument("MatrixSubstractionStrategy: dimension mismatch");

        auto* dA = dynamic_cast<DynamicMatrix*>(A.get());
        auto* dB = dynamic_cast<DynamicMatrix*>(B.get());
        if (dA && dB)
            return std::make_shared<DynamicMatrix>(*dA - *dB);

        size_t m = A->rows(), n = A->cols();
        auto result = std::make_shared<DynamicMatrix>(m, n);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*result)(i, j) = A->get(i, j) - B->get(i, j);
        return result;
    }

    bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const override {
        return A.rows() == B.rows() && A.cols() == B.cols();
    }
};

class MatrixMultiplyStrategy : public BinaryMatrixOperationStrategy {
public:
    ~MatrixMultiplyStrategy() override = default;

    MatrixPtr execute(MatrixPtr A, MatrixPtr B) override {
        if (!isSupported(*A, *B))
            throw std::invalid_argument("MatrixMultiplyStrategy: inner dimensions mismatch");

        // Fast path: both DynamicMatrix — uses cache-friendly i-k-j operator*
        auto* dA = dynamic_cast<DynamicMatrix*>(A.get());
        auto* dB = dynamic_cast<DynamicMatrix*>(B.get());
        if (dA && dB)
            return std::make_shared<DynamicMatrix>(*dA * *dB);

        /// General path with cache-friendly i-k-j loop.
        /// Copies A into a DynamicMatrix so inner-loop rows are contiguous.
        DynamicMatrix cA(*A), cB(*B);
        return std::make_shared<DynamicMatrix>(cA * cB);
    }

    bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const override {
        return A.cols() == B.rows();
    }
};

class MatrixKronekerMultiplyStrategy : public BinaryMatrixOperationStrategy {
public:
    ~MatrixKronekerMultiplyStrategy() override = default;

    MatrixPtr execute(MatrixPtr A, MatrixPtr B) override {
        if (!A || !B)
            throw std::invalid_argument("MatrixKronekerMultiplyStrategy: null operand");

        size_t aR = A->rows(), aC = A->cols();
        size_t bR = B->rows(), bC = B->cols();
        auto result = std::make_shared<DynamicMatrix>(aR * bR, aC * bC);

        for (size_t p = 0; p < aR; ++p) {
            for (size_t q = 0; q < aC; ++q) {
                double a = A->get(p, q);
                for (size_t r = 0; r < bR; ++r) {
                    for (size_t s = 0; s < bC; ++s) {
                        (*result)(p * bR + r, q * bC + s) = a * B->get(r, s);
                    }
                }
            }
        }
        return result;
    }

    bool isSupported(const AbstractMatrix&, const AbstractMatrix&) const override {
        return true;
    }
};

} // namespace SharedMath::LinearAlgebra
