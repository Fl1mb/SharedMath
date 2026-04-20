#pragma once

#include "MatrixOperationsStrategy.h"
#include "DynamicMatrix.h"

namespace SharedMath::LinearAlgebra {

class MatrixTransposeStrategy : public UnaryMatrixOperationStrategy {
public:
    ~MatrixTransposeStrategy() override = default;

    MatrixPtr execute(MatrixPtr A) override {
        if (!A) throw std::invalid_argument("MatrixTransposeStrategy: null matrix");

        // Fast path: DynamicMatrix has cache-friendly transposed()
        if (auto* d = dynamic_cast<DynamicMatrix*>(A.get()))
            return std::make_shared<DynamicMatrix>(d->transposed());

        // General path
        size_t m = A->rows(), n = A->cols();
        auto result = std::make_shared<DynamicMatrix>(n, m);
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                (*result)(j, i) = A->get(i, j);
        return result;
    }

    bool isSupported(const AbstractMatrix&) const override { return true; }
};

} // namespace SharedMath::LinearAlgebra
