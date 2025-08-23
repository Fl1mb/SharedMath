#pragma once

#include "MatrixView.h"
#include <string>

namespace SharedMath
{
    namespace LinearAlgebra
    {
        template<size_t Rows, size_t Cols>
        class MatrixOperations{
        public:
            using matrix = Matrix<Rows, Cols>;
            using Tmatrix = Matrix<Cols, Rows>;

            virtual ~MatrixOperations() = default;

            virtual matrix add(const matrix& a, const matrix& b) const = 0;
            virtual matrix multiply(const matrix& a, const matrix& b) const = 0;
            virtual matrix multiply(const matrix& a, double scalar) const = 0;
            virtual matrix multiply(const matrix& a, const Vector<Cols>& vec) const = 0;

            virtual Tmatrix transpose(const matrix& mtrx) const;
            virtual double determinant(const matrix& mtrx) const = 0;
            virtual matrix inverse(const matrix& mtrx) const = 0;

            virtual void luDecomposition(const matrix& mtrx, matrix& L, matrix& U) const = 0;
            virtual std::pair<matrix, matrix> luDecomposition(const matrix& mtrx) const = 0;
            virtual void qrDecomposition(const matrix& mtrx, matrix& Q, matrix& Q) const = 0;
            virtual std::pair<matrix, matrix> qrDecomposition(const matrix& mtrx) const = 0;

            virtual std::string getBackendType() const = 0;
        };
    } // namespace LinearAlgebra 
} // namespace SharedMath
