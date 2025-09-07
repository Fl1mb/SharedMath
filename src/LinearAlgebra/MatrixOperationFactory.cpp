#include "../../include/LinearAlgebra/MatrixOperationFactory.h"

using namespace SharedMath::LinearAlgebra;


MatrixStrategyFactory::MatrixStrategyFactory(){
    registerStrategies();
}

void MatrixStrategyFactory::registerStrategies(){
    binaryStrategies_[OperationType::ADDITION] = [](){
        return std::make_unique<MatrixAdditionStrategy>();
    };
    binaryStrategies_[OperationType::SUBSTRACTION] = [](){
        return std::make_unique<MatrixSubstractionStrategy>();
    };
    binaryStrategies_[OperationType::KRONECKER_PRODUCT] = [](){
        return std::make_unique<MatrixKronekerMultiplyStrategy>();
    };
    binaryStrategies_[OperationType::MULTIPLICATION] = [](){
        return std::make_unique<MatrixMultiplyStrategy>();
    };

    unaryStrategies_[OperationType::TRANSPOSE] = [](){
        return std::make_unique<MatrixTransposeStrategy>();
    };

    scalarStrategies_[OperationType::TRACE] = [](){
        return std::make_unique<MatrixTraceStrategy>();
    };
}

MatrixStrategyFactory::BinaryStrategy 
MatrixStrategyFactory::createBinaryStrategy(OperationType type){
    auto iter = binaryStrategies_.find(type);
    if(iter != binaryStrategies_.end()){
        return iter->second();
    }
    throw std::invalid_argument("Unknown type of binary operation");
}

MatrixStrategyFactory::UnaryStrategy 
MatrixStrategyFactory::createUnaryStrategy(OperationType type){
    auto iter = unaryStrategies_.find(type);
    if(iter != unaryStrategies_.end()){
        return iter->second();
    }
    throw std::invalid_argument("Unknows type of unary operation");
}

MatrixStrategyFactory::ScalarStrategy 
MatrixStrategyFactory::createScalarStrategy(OperationType type){
    auto iter = scalarStrategies_.find(type);
    if(iter != scalarStrategies_.end()){
        return iter->second();
    }
    throw std::invalid_argument("Uknowns type of scalar operation");
}