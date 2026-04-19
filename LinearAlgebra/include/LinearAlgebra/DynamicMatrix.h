#pragma once

#include "AbstractMatrix.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace SharedMath
{
    namespace LinearAlgebra
    {
        
        class DynamicMatrix : public AbstractMatrix{
        public:
            DynamicMatrix() = default;
            DynamicMatrix(size_t rows, size_t cols) : 
                rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, 0.0)) {}
            
            DynamicMatrix(const DynamicMatrix&) = default;
            DynamicMatrix(DynamicMatrix&&) noexcept = default;
            DynamicMatrix& operator=(const DynamicMatrix&) = default;
            DynamicMatrix& operator=(DynamicMatrix&&) noexcept = default;

            DynamicMatrix(std::shared_ptr<AbstractMatrix> otherMatrix){
                if(!otherMatrix)throw std::invalid_argument("Nullptr matrix");
                rows_ = otherMatrix->rows();
                cols_ = otherMatrix->cols();
                data_.resize(rows_, std::vector<double>(cols_, 0.0));

                for(size_t i = 0; i < rows_; ++i){
                    for(size_t j = 0; j < cols_; ++j){
                        data_[i][j] = otherMatrix->get(i, j);
                    }
                }
            }

            bool operator==(const DynamicMatrix& other) const{
                return data_ == other.data_ && (rows_ == other.rows_ && cols_ == other.cols_);
            }

            bool operator!=(const DynamicMatrix& other) const{
                return !(*this == other);
            }

            DynamicMatrix operator*(double scalar){
                DynamicMatrix result(rows(), cols());
                for(size_t i = 0; i < rows(); ++i){
                    for(size_t j = 0; j < cols(); ++j){
                        result.set(i, j, data_[i][j] * scalar);
                    }
                }
                return result;
            }

            DynamicMatrix operator*(const DynamicMatrix& other) const{
                if(cols() != other.rows()){
                    throw std::invalid_argument("Matrices are not supported for multiplication");
                }

                DynamicMatrix resultMatrix(rows(), other.cols());

                for(size_t i = 0; i < rows(); ++i){
                    for(size_t j = 0; j < other.cols(); ++j){
                        double sum = 0.0;

                        for(size_t r = 0; r < cols(); ++r){
                            double mul = get(i, r) * other.get(r, j);
                            sum += mul;
                        }

                        resultMatrix.set(i, j, sum);
                    }
                }

                return resultMatrix;
            }

            virtual ~DynamicMatrix() override = default;

            size_t rows() const override{return rows_;}
            size_t cols() const override{return cols_;}

            double get(size_t rows, size_t cols) const override{return data_[rows][cols];}
            double& get(size_t row, size_t col) override {return data_[row][col];}
            double* toPtr() override{return &data_[0][0];}
            const double* toPtr() const override{return &data_[0][0];}

            void set(size_t row, size_t col, double val) override {data_[row][col] = val;}
        
            void clear(){
                for(auto& el : data_){
                    for(auto& num : el)
                        num = 0.0;
                }
            }

        private:
            size_t rows_;
            size_t cols_;
            std::vector<std::vector<double>> data_;
        };

    } // namespace LinearAlgebra
    

} // namespace SharedMath
