#pragma once
#include "Matrix.h"
#include <memory>

namespace SharedMath
{
    namespace LinearAlgebra
    {
        class MatrixView {
        public:
            MatrixView() = default;

            MatrixView(AbstractMatrix* matrix);
            MatrixView(size_t startRow, size_t endRow, size_t startCol, size_t endCol, AbstractMatrix* matrix_);
            MatrixView(const MatrixView&) = default;
            MatrixView(MatrixView&&) noexcept = default;

            MatrixView& operator=(const MatrixView&) = default;
            MatrixView& operator=(MatrixView&&) noexcept = default;

            ~MatrixView() = default;

            size_t rows() const;
            size_t cols() const;

            double get(size_t row, size_t col) const;
            void set(size_t row, size_t col, double val); \

            double operator()(size_t row, size_t col) const;
            double& operator()(size_t row, size_t col);

            MatrixView subView(size_t startRow, size_t endRow, size_t startCol, size_t endCol) const;
            
            template<size_t Rows, size_t Cols>
            std::shared_ptr<Matrix<Rows, Cols>> toMatrix() const{
                if (rows() != Rows || cols() != Cols) {
                    throw std::invalid_argument("View size doesn't match template parameters");
                }
                auto ResultMatrix = std::make_shared<Matrix<Rows, Cols>>();
                for(size_t i = 0; i < Rows; ++i){
                    for(size_t j = 0; j < Cols; ++j){
                        (*ResultMatrix)[i][j] = get(i, j);
                    }
                }
                return ResultMatrix;
            }

        private:
            void checkIndices(size_t row, size_t col) const;

            size_t StartRowIdx;
            size_t StartColIdx;
            size_t EndRowIdx;
            size_t EndColIdx;  
            AbstractMatrix* matrix;
        }; 
    } // namespace LinearAlgebra
} // namespace SharedMath