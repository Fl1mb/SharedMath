#pragma once

#include "MatrixOperations.h"

namespace SharedMath
{
    namespace LinearAlgebra
    {
        template<size_t Rows, size_t Cols>
        class DefaultMatrixOperations : public MatrixOperations<Rows, Cols>{
        public:

            virtual ~DefaultMatrixOperations() override = default;

            matrix add(const matrix& a, const matrix& b) const override{
                Matrix<Rows, Cols> result;
                for(size_t i = 0; i < Rows; ++i){
                    result[i] = a[i] + b[i];
                }
                return result;
            }

            decltype(auto) multiply(const auto& a, const auto& b) const override{
                if constexpr (std::is_same_v<std::decay_t<decltype(a)>, matrix> && 
                             std::is_same_v<std::decay_t<decltype(b)>, double>) {
                    return a * b;
                }
                else if constexpr (std::is_same_v<std::decay_t<decltype(a)>, double> && 
                                  std::is_same_v<std::decay_t<decltype(b)>, matrix>) {
                    return b * a;
                }
                else if constexpr (std::is_same_v<std::decay_t<decltype(a)>, matrix> && 
                                  std::is_same_v<std::decay_t<decltype(b)>, Vector<Cols>>) {
                    return multiplyVector(a, b);
                }
                else if constexpr (std::is_same_v<std::decay_t<decltype(a)>, matrix> && 
                                  std::is_same_v<std::decay_t<decltype(b)>, matrix>) {
                    if constexpr (Cols == Rows) {
                        return multiplyMatrix(a, b);
                    } else {
                        static_assert(Cols == Rows, 
                            "Matrix multiplication requires compatible dimensions");
                    }
                }
                else {
                    static_assert(sizeof(a) == 0, 
                        "Unsupported types for matrix multiplication");
                }
            }
            
            matrix multiply(const matrix& a, const matrix& b) const override;
            matrix multiply(const matrix& a, double scalar) const override;
            matrix multiply(const matrix& a, const Vector<Cols>& vec) const override;

        private:

        };

    } // namespace LinearAlgebra
    

} // namespace SharedMath
