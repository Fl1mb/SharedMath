#pragma once

#include <unordered_map>
#include "DefaultBinaryMatrixStrategies.h"
#include "DefaultUnaryMatrixStrategies.h"
#include <memory>
#include <functional>

namespace SharedMath::LinearAlgebra
{
    enum class OperationType{
        ADDITION = 0,
        SUBSTRACTION,
        MULTIPLICATION,
        KRONECKER_PRODUCT,
        TRANSPOSE
    };

    class AbstractMatrixStrategyFactory{
    public:
        
        using BinaryStrategyCreator = std::function<std::unique_ptr<BinaryMatrixOperationStrategy>()>;
        using UnaryStrategyCreator = std::function<std::unique_ptr<UnaryMatrixOperationStrategy>()>;

        using BinaryStrategy = std::unique_ptr<BinaryMatrixOperationStrategy>;
        using UnaryStrategy = std::unique_ptr<UnaryMatrixOperationStrategy>;

        virtual ~AbstractMatrixStrategyFactory() = default;

        virtual BinaryStrategy createBinaryStrategy(OperationType type) = 0;
        virtual UnaryStrategy createUnaryStrategy(OperationType type) = 0;

    protected:
        std::unordered_map<OperationType, BinaryStrategyCreator> binaryStrategies_;
        std::unordered_map<OperationType, UnaryStrategyCreator> unaryStrategies_;
    };


    class MatrixStrategyFactory : public AbstractMatrixStrategyFactory{
    public:
        
        MatrixStrategyFactory();    
        virtual ~MatrixStrategyFactory() override = default;

        BinaryStrategy createBinaryStrategy(OperationType type) override;
        UnaryStrategy createUnaryStrategy(OperationType type) override;

    private:
        void registerStrategies();

    };
} // namespace SharedMath

