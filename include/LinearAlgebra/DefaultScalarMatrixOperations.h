#pragma once

#include "MatrixOperationsStrategy.h"
#include "LUDecomposition.h"

namespace SharedMath::LinearAlgebra
{
    class MatrixDeterminantStrategy : public ScalarMatrixOperationStrategy{
    public:
        virtual ~MatrixDeterminantStrategy() override = default;

        double execute(MatrixPtr A) override{
            if(!isSupported(*A)){
                throw std::invalid_argument("Matrix is not square. Cannot calculate det");
            }
            LUDecomposition lu(A);
            try{
                lu.MakeDecomposition();
                return lu.Determinant();
            }catch(const std::exception& ex){
                throw std::runtime_error(std::string("Could not calculate determinant: ") + std::string(ex.what()));
            }
        }
        bool isSupported(const AbstractMatrix& A) const override{
            return A.cols() == A.rows();
        }
    };

    class MatrixTraceStrategy : public ScalarMatrixOperationStrategy{
    public:
        virtual ~MatrixTraceStrategy() override = default;

        double execute(MatrixPtr A) override{
            if(!isSupported(*A)){
                throw std::invalid_argument("Can't find trace: matrix is not squared");
            }

            double result = 0.0;

            for(size_t i = 0; i < A->rows(); ++i){
                result += A->get(i, i);
            }
            return result;
        }

        bool isSupported(const AbstractMatrix& A) const override{
            return A.cols() == A.rows();
        }
    };
    
} // namespace SharedMath::LinearAlgebra
