#pragma once

#include "../constans.h"
#include <math.h>
#include <array>
#include <stdexcept>

namespace SharedMath
{
    namespace LinearAlgebra
    {
        template<size_t N>
        class Vector{
        public:
            Vector(){data.fill(0.0);}
            Vector(std::initializer_list<double> values){
                std::copy(values.begin(), values.end(), data.begin());
            }
            Vector(const Vector<N>&) = default;
            Vector(Vector<N>&&) noexcept = default;
            Vector& operator=(const Vector<N>&) = default;
            Vector& operator=(Vector<N>&&) noexcept = default;
            ~Vector() = default;

            double& operator[](size_t index){return data[index];}
            const double& operator[](size_t index) const{return data[index];}

            bool operator==(const Vector<N>& other) const{
                for(auto i = 0; i < N; ++i){
                    if(data[i] != other.data[i])return false;
                }return true;
            }

            bool operator!=(const Vector<N>& other) const{
                return !(*this == other);
            }

            Vector operator+(const Vector& other) const{
                Vector result;
                for(size_t i = 0; i < N; ++i){
                    result[i] = data[i] + other.data[i];
                }
                return result;
            }

            Vector operator-(const Vector& other) const{
                Vector result;
                for(size_t i = 0; i < N; ++i){
                    result[i] = data[i] - other.data[i];
                }
                return result;
            }

            Vector operator*(double scalar) const{
                Vector result;
                for(size_t i = 0; i < N; ++i){
                    result[i] = data[i] * scalar;
                }
                return result;
            }

            double dot(const Vector& other) const{
                double result = 0.0;
                for(size_t i = 0; i < N; ++i){
                    result += data[i] * other[i];
                }
                return result;
            }

            double norm() const{
                return sqrt(dot(*this));
            }

            Vector normalized() const{
                double length = norm;
                if(length < Epsilon){
                    throw std::runtime_error("Cannot normalize zero vector");
                }
                return (*this) * (1.0 / length);
            }
            static constexpr size_t size() {return N;}
        private:
            std::array<double, N> data;
        };

    } // namespace LinearAlgebra

    
} // namespace SharedMath
