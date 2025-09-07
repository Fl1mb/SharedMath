#pragma once 

#include "MatrixOperationsStrategy.h"
#include "DynamicMatrix.h"

namespace SharedMath
{
    namespace LinearAlgebra
    {
        class MatrixAdditionStrategy : public BinaryMatrixOperationStrategy{
        public:
            virtual ~MatrixAdditionStrategy() override = default;
        
            MatrixPtr execute(MatrixPtr A, MatrixPtr B) override{
                if(!isSupported(*A, *B)){
                    throw std::invalid_argument("Matrices are not valid for addition");
                }

                auto matrixResult = std::make_shared<DynamicMatrix>(A->rows(), B->cols());

                for(size_t i = 0; i < matrixResult->rows(); ++i){
                    for(size_t j = 0; j < matrixResult->cols(); ++j){
                        double sum = A->get(i, j) + B->get(i, j);
                        matrixResult->set(i, j, sum);
                    }
                }

                return matrixResult;
            }

            bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const override{
                return A.rows() == B.rows() && A.cols() == B.cols();
            }

        };

        class MatrixSubstractionStrategy : public BinaryMatrixOperationStrategy{
        public:
            virtual ~MatrixSubstractionStrategy() override = default;

            MatrixPtr execute(MatrixPtr A, MatrixPtr B) override{
                if(!isSupported(*A, *B)){
                    throw std::invalid_argument("Matrices are not valid for substraction");
                }

                auto matrixResult = std::make_shared<DynamicMatrix>(A->rows(), B->cols());

                for(size_t i = 0; i < matrixResult->rows(); ++i){
                    for(size_t j = 0; j < matrixResult->cols(); ++j){
                        double sum = A->get(i, j) - B->get(i, j);
                        matrixResult->set(i, j, sum);
                    }
                }

                return matrixResult;
            }

            bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const override{
                return A.rows() == B.rows() && A.cols() == B.cols();
            }
        };

        class MatrixMultiplyStrategy : public BinaryMatrixOperationStrategy{
        public:
            virtual ~MatrixMultiplyStrategy() override = default;

            MatrixPtr execute(MatrixPtr A, MatrixPtr B) override{
                if(!isSupported(*A, *B)){
                    throw std::invalid_argument("Matrices are not supported for multiplication");
                }

                auto resultMatrix = std::make_shared<DynamicMatrix>(A->rows(), B->cols());

                for(size_t i = 0; i < A->rows(); i++){
                    for(size_t j = 0; j < B->cols(); ++j){
                        double sum = 0.0;

                        for(size_t r = 0; r < A->cols(); ++r){
                            double mul = A->get(i, r) * B->get(r, j); 
                            sum += mul;
                        }

                        resultMatrix->set(i, j, sum);
                    }
                }
                return resultMatrix;
            }

            bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const override{
                return A.cols() == B.rows();
            }
        };

        class MatrixKronekerMultiplyStrategy : public BinaryMatrixOperationStrategy{
        public:
            virtual ~MatrixKronekerMultiplyStrategy() override = default;

            MatrixPtr execute(MatrixPtr A, MatrixPtr B) override{
                if (!A || !B) {
                    throw std::invalid_argument("Matrices cannot be null");
                }

                size_t a_rows = A->rows();
                size_t a_cols = A->cols();
                size_t b_rows = B->rows();
                size_t b_cols = B->cols();

                size_t result_rows = a_rows * b_rows;
                size_t result_cols = a_cols * b_cols;

                auto resultMatrix = std::make_shared<DynamicMatrix>(result_rows, result_cols);

                for (size_t p = 0; p < a_rows; ++p) {
                    for (size_t q = 0; q < a_cols; ++q) {
                        double a_val = A->get(p, q);
                        
                        for (size_t r = 0; r < b_rows; ++r) {
                            for (size_t s = 0; s < b_cols; ++s) {
                                double b_val = B->get(r, s);
                                size_t i = p * b_rows + r;
                                size_t j = q * b_cols + s;
                                
                                resultMatrix->set(i, j, a_val * b_val);
                            }
                        }
                    }
                }
                return resultMatrix;
            }


            bool isSupported(const AbstractMatrix& A, const AbstractMatrix& B) const override{
                return true;
            }
        };
    } // namespace LinearAlgebra
} // namespace SharedMath
