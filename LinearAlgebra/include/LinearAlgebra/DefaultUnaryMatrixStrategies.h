#pragma once

#include "MatrixOperationsStrategy.h"
#include "DynamicMatrix.h"

namespace SharedMath
{
    namespace LinearAlgebra
    {
        class MatrixTransposeStrategy : public UnaryMatrixOperationStrategy{
        public:
            virtual ~MatrixTransposeStrategy() override = default;

            MatrixPtr execute(MatrixPtr A) override{
                if(!A){
                    throw std::invalid_argument("Matrix is nullptr");
                }

                auto resultMatrix = std::make_shared<DynamicMatrix>(A->cols(), A->rows());

                for(size_t i = 0 ; i < A->rows(); ++i){
                    for(size_t j = 0; j < A->cols(); ++j){
                        resultMatrix->set(j, i, A->get(i, j));
                    }
                }
                return resultMatrix;
            }
                
            bool isSupported(const AbstractMatrix& A) const override{
                return true;
            }
        };
    } // namespace LinearAlgebra
} // namespace SharedMath
