#include "../../include/LinearAlgebra/MatrixOperations.h"

using namespace SharedMath::LinearAlgebra;

MatrixOperations::MatrixPtr 
MatrixOperations::add(MatrixPtr A, MatrixPtr B){
    context_.setBinaryStrategy(factory_.createBinaryStrategy(OperationType::ADDITION));
    return context_.executeBinary(A, B);
}

MatrixOperations::MatrixPtr 
MatrixOperations::substract(MatrixPtr A, MatrixPtr B){
    context_.setBinaryStrategy(factory_.createBinaryStrategy(OperationType::SUBSTRACTION));
    return context_.executeBinary(A, B);
}

MatrixOperations::MatrixPtr 
MatrixOperations::multiply(MatrixPtr A, MatrixPtr B){
    context_.setBinaryStrategy(factory_.createBinaryStrategy(OperationType::MULTIPLICATION));
    return context_.executeBinary(A, B);
}

MatrixOperations::MatrixPtr 
MatrixOperations::kroneckerProduct(MatrixPtr A, MatrixPtr B){
    context_.setBinaryStrategy(factory_.createBinaryStrategy(OperationType::KRONECKER_PRODUCT));
    return context_.executeBinary(A, B);
}

MatrixOperations::MatrixPtr 
MatrixOperations::transpose(MatrixPtr A){
    context_.setUnaryStrategy(factory_.createUnaryStrategy(OperationType::TRANSPOSE));
    return context_.executeUnary(A);
}

double MatrixOperations::trace(MatrixPtr A){
    context_.setScalarStrategy(factory_.createScalarStrategy(OperationType::TRACE));
    return context_.executeScalar(A);
}

void MatrixOperations::setCustomBinaryStrategy(std::unique_ptr<BinaryMatrixOperationStrategy> strategy){
    context_.setBinaryStrategy(std::move(strategy));
}

void MatrixOperations::setCustomUnaryStrategy(std::unique_ptr<UnaryMatrixOperationStrategy> strategy){
    context_.setUnaryStrategy(std::move(strategy));
}

bool MatrixOperations::canAdd(const AbstractMatrix& A, const AbstractMatrix& B) const{
    return MatrixAdditionStrategy().isSupported(A, B);
}

bool MatrixOperations::canMultiply(const AbstractMatrix& A, const AbstractMatrix& B) const{
    return MatrixMultiplyStrategy().isSupported(A, B);
}



