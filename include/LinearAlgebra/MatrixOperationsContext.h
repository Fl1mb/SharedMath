#pragma once

#include "MatrixOperationFactory.h"

namespace SharedMath::LinearAlgebra
{

    class MatrixOperationsContext{
    public:
        using MatrixPtr = std::shared_ptr<AbstractMatrix>;

        void setBinaryStrategy(std::unique_ptr<BinaryMatrixOperationStrategy> strategy) {
            binaryStrategy_ = std::move(strategy);
        }

        void setUnaryStrategy(std::unique_ptr<UnaryMatrixOperationStrategy> strategy) {
            unaryStrategy_ = std::move(strategy);
        }

        void setScalarStrategy(std::unique_ptr<ScalarMatrixOperationStrategy> strategy){
            scalarStrategy_ = std::move(strategy);
        }

        MatrixPtr executeBinary(MatrixPtr A, MatrixPtr B) {
            if (!binaryStrategy_) {
                throw std::runtime_error("Binary strategy not set");
            }
            if (!A || !B) {
                throw std::invalid_argument("Matrices cannot be null");
            }
            if (!binaryStrategy_->isSupported(*A, *B)) {
                throw std::invalid_argument("Matrices not supported by current strategy");
            }
            return binaryStrategy_->execute(A, B);
        }
        
        MatrixPtr executeUnary(MatrixPtr A) {
            if (!unaryStrategy_) {
                throw std::runtime_error("Unary strategy not set");
            }
            if (!A) {
                throw std::invalid_argument("Matrix cannot be null");
            }
            if (!unaryStrategy_->isSupported(*A)) {
                throw std::invalid_argument("Matrix not supported by current strategy");
            }
            return unaryStrategy_->execute(A);
        }

        double executeScalar(MatrixPtr A){
            if(!scalarStrategy_){
                throw std::runtime_error("Scalar strategy not set");
            }
            if(!A){
                throw std::invalid_argument("Matrix cannot be null");
            }
            if(!scalarStrategy_->isSupported(*A)){
                throw std::invalid_argument("Matrix is not supported bu current strategy");
            }
            return scalarStrategy_->execute(A);
        }

        bool canExecuteBinary(const AbstractMatrix& A, const AbstractMatrix& B) const {
            return binaryStrategy_ && binaryStrategy_->isSupported(A, B);
        }

        bool canExecuteUnary(const AbstractMatrix& A) const {
            return unaryStrategy_ && unaryStrategy_->isSupported(A);
        }
    private:
        std::unique_ptr<BinaryMatrixOperationStrategy> binaryStrategy_;
        std::unique_ptr<UnaryMatrixOperationStrategy> unaryStrategy_;
        std::unique_ptr<ScalarMatrixOperationStrategy> scalarStrategy_;
    };
} // namespace SharedMath::LinearAlgebra
