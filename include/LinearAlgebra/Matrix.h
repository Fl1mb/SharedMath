#pragma once

#include "VectorN.h"
#include <memory>

namespace SharedMath
{
    namespace LinearAlgebra
    {
        template<size_t Rows, size_t Cols>
        class Matrix{
        public:
            Matrix(){data.fill({});}
            Matrix(const Matrix&) = default;
            Matrix(Matrix&&) noexcept = default;
            Matrix& operator=(const Matrix&) = default;
            Matrix& operator=(Matrix&&) noexcept = default;
            ~Matrix() = default;

            Matrix(std::initializer_list<std::initializer_list<double>> values){
                if(values.size() != Rows){
                    throw std::invalid_argument("Invalid number of rows in initializer list");
                }
                size_t i = 0;
                for (const auto& row : values) {
                    if (row.size() != Cols) {
                        throw std::invalid_argument("Invalid number of columns in initializer list");
                    }
                    size_t j = 0;
                    for (double value : row) {
                        data[i][j] = value;
                        ++j;
                    }
                    ++i;
                }
            }

            Vector<Cols>& operator[](size_t index){return data[index];}
            const Vector<Cols>& operator[](size_t index) const{return data[index];}

            Vector<Rows> operator*(const Vector<Cols>& vec) const{
                Vector<Rows> result;
                for(size_t i = 0; i < Rows; ++i){
                    result[i] = data[i].dot(vec);
                }
                return result;
            }

            Matrix operator*(double scalar) const{
                Matrix result;
                for(auto i = 0; i < Rows; ++i){
                    result[i] = data[i] * scalar;
                }
                return result;
            }

            Matrix operator+(const Matrix& other) const{
                Matrix result;
                for(auto i = 0; i < Rows; i++){
                    result[i] = data[i] + other[i];
                }
                return result;
            }

            Matrix operator-(const Matrix& other) const{
                Matrix result;
                for(auto i = 0; i < Rows; i++){
                    result[i] = data[i] - other[i];
                }
                return result;
            }

            static constexpr size_t rows(){return Rows;}
            static constexpr size_t cols(){return Cols;}

            double* toPtr(){
                return &(data[0][0]);
            }

            const double* toPtr() const{
                return &(data[0][0]);
            }

            double* rowPtr(size_t row){
                if(row > Rows){
                    throw std::out_of_range("Row index is out of range");
                }
                return &(data[row][0]);
            }

            const double* rowPtr(size_t row) const{
                if(row > Rows){
                    throw std::out_of_range("Row index is out of range");
                }
                return &(data[row][0]);
            }

            std::array<double, Rows * Cols> toRowMajorArray() const{
                std::array<double, Rows * Cols> result;
                for (size_t i = 0; i < Rows; ++i) {
                    for (size_t j = 0; j < Cols; ++j) {
                        result[i * Cols + j] = data[i][j];
                    }
                }
                return result;
            }

            std::array<double, Rows * Cols> toColumnMajorArray() const {
                std::array<double, Rows * Cols> result;
                for (size_t j = 0; j < Cols; ++j) {
                    for (size_t i = 0; i < Rows; ++i) {
                        result[j * Rows + i] = data[i][j];
                    }
                }
                return result;
            }
            
            void fromRowMajorArray(const double* ptr) {
                for (size_t i = 0; i < Rows; ++i) {
                    for (size_t j = 0; j < Cols; ++j) {
                        data[i][j] = ptr[i * Cols + j];
                    }
                }
            }

            void fromColumnMajorArray(const double* ptr) {
                for (size_t j = 0; j < Cols; ++j) {
                    for (size_t i = 0; i < Rows; ++i) {
                        data[i][j] = ptr[j * Rows + i];
                    }
                }
            }


            static constexpr size_t dataSizeBytes() {
                return Rows * Cols * sizeof(double);
            }

            static constexpr size_t totalElements() {
                return Rows * Cols;
            }

            static constexpr bool isContiguous() {
                return true;
            }
        private:
            std::array<Vector<Cols>, Rows> data;
        };
    } // namespace LinearAlgebra
} // namespace SharedMath
