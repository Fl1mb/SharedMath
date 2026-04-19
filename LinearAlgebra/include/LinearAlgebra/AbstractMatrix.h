#pragma once

#include <cstdio>

namespace SharedMath    
{
    namespace LinearAlgebra
    {
        class AbstractMatrix{
        public:
            virtual ~AbstractMatrix() = default;

            virtual size_t rows() const = 0;
            virtual size_t cols() const = 0;

            virtual double get(size_t row, size_t col) const = 0;
            virtual double& get(size_t row, size_t col) = 0;
            virtual void set(size_t row, size_t col, double val) = 0;

            virtual double* toPtr() = 0;
            virtual const double* toPtr() const = 0;

        };
    } // namespace LinearAlgebra
    
    
} // namespace SharedMath   
