#pragma once

#include "MatrixView.h"

namespace SharedMath
{
    namespace LinearAlgebra
    {

        class UnaryMatrixOperationStrategy{
        public:
            using MatrixPtr = std::shared_ptr<AbstractMatrix>;
    
            virtual ~UnaryMatrixOperationStrategy() = default;
    
            virtual MatrixPtr execute(MatrixPtr A) = 0;
            virtual bool isSupported(const AbstractMatrix& A) const = 0;
        };

        class BinaryMatrixOperationStrategy{
        public:
            using MatrixPtr = std::shared_ptr<AbstractMatrix>;    

            virtual ~BinaryMatrixOperationStrategy() = default;

            virtual MatrixPtr execute(MatrixPtr A, MatrixPtr B) = 0;
            virtual bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const = 0;
        };

    } // namespace LinearAlgebra
} // namespace SharedMath
