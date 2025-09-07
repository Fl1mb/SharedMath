#pragma once

#include "MatrixOperationsContext.h"
#include "MatrixOperationFactory.h"

namespace SharedMath::LinearAlgebra
{
    class MatrixOperations{
    public:
        using MatrixPtr = std::shared_ptr<AbstractMatrix>;

        MatrixPtr add(MatrixPtr A, MatrixPtr B);
        MatrixPtr substract(MatrixPtr A, MatrixPtr B);
        MatrixPtr multiply(MatrixPtr A, MatrixPtr B);
        MatrixPtr kroneckerProduct(MatrixPtr A, MatrixPtr B);

        MatrixPtr transpose(MatrixPtr A);

        void setCustomBinaryStrategy(std::unique_ptr<BinaryMatrixOperationStrategy> strategy);
        void setCustomUnaryStrategy(std::unique_ptr<UnaryMatrixOperationStrategy> strategy);

        bool canAdd(const AbstractMatrix& A, const AbstractMatrix& B) const;
        bool canMultiply(const AbstractMatrix& A, const AbstractMatrix& B) const;

    private:
        MatrixOperationsContext context_;
        MatrixStrategyFactory factory_;
    };
    
} // namespace SharedMath::LinearAlgebra
