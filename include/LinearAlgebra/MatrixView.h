#pragma once
#include "Matrix.h"

namespace SharedMath
{
    namespace LinearAlgebra
    {
        template<size_t ViewRows, size_t ViewCols, typename MatrixType>
        class MatrixView {
        public:
            MatrixView(MatrixType& matrix, 
                       size_t startRow_ = 0, 
                       size_t startCol_ = 0)
                : data(matrix),
                  startRow(startRow_),
                  startCol(startCol_)
            {
                validateIndices();
            }

            double& operator()(size_t row, size_t col) {
                checkBounds(row, col);
                return data[startRow + row][startCol + col];
            }

            const double& operator()(size_t row, size_t col) const {
                checkBounds(row, col);
                return data[startRow + row][startCol + col];
            }

            Vector<ViewCols> row(size_t rowIndex) const {
                checkBounds(rowIndex, 0);
                Vector<ViewCols> result;
                for(size_t col = 0; col < ViewCols; ++col){
                    result[col] = data[startRow + rowIndex][startCol + col];
                }
                return result;
            }

            Vector<ViewRows> column(size_t colIndex) const {
                checkBounds(0, colIndex);
                Vector<ViewRows> result;
                for(size_t row = 0; row < ViewRows; ++row){
                    result[row] = data[startRow + row][startCol + colIndex];
                }
                return result;
            }

            static constexpr size_t rows() { return ViewRows; }
            static constexpr size_t columns() { return ViewCols; }

            Matrix<ViewRows, ViewCols> toMatrix() const {
                Matrix<ViewRows, ViewCols> result;
                for (size_t i = 0; i < ViewRows; ++i) {
                    for (size_t j = 0; j < ViewCols; ++j) {
                        result[i][j] = data[startRow + i][startCol + j];
                    }
                }
                return result;
            }

        private:
            void validateIndices() const {
                if(startRow + ViewRows > data.rows() ||
                   startCol + ViewCols > data.cols()){
                    throw std::out_of_range("MatrixView indices out of range");
                }
            }

            void checkBounds(size_t row, size_t col) const {
                if(row >= ViewRows || col >= ViewCols){
                    throw std::out_of_range("MatrixView access out of bounds");
                }
            }

            MatrixType& data;
            size_t startRow;
            size_t startCol;
        };

        template<size_t ViewRows, size_t ViewCols, typename MatrixType>
        auto CreateMatrixView(MatrixType& matrix, 
                             size_t startRow = 0, 
                             size_t startCol = 0) {
            return MatrixView<ViewRows, ViewCols, MatrixType>(matrix, startRow, startCol);
        }
    } // namespace LinearAlgebra
} // namespace SharedMath