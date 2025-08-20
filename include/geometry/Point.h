#pragma once

#include <array>
#include <stdexcept>
#include <utility>
#include <cstring>
#include <math.h>

namespace SharedMath{
    namespace Geometry
    {
        template<size_t N>
        class Point{
        public:
            Point(){coords.fill(0.0);}
            Point(std::initializer_list<double> values){
                std::copy(values.begin(), values.end(), coords.begin());
            }
            Point(const Point<N>& other){
                coords = other.coords;
            }
            Point(Point<N>&& other) noexcept{
                coords = other.coords;
                other.coords.fill(0.0);
            }

            Point(double x, double y){
                coords[0] = x;
                coords[1] = y;
            }

            Point(double x, double y, double z){
                coords[0] = x;
                coords[1] = y;
                coords[2] = z;
            }

            Point& operator=(const Point<N>& other){
                if(*this == other)return *this;
                coords = other.coords;
                return *this;
            }
            Point& operator=(Point<N>&& other) noexcept{
                if(*this == other)return *this;
                coords = other.coords;
                other.coords.fill(0.0);
                return *this;
            }

            double operator[](size_t index)const{
                if(index >= N)throw std::out_of_range("Point::operator[] is out of range");
                return coords[index];
            }
            double& operator[](size_t index){
                if(index >= N)throw std::out_of_range("Point::operator[] is out of range");
                return coords[index];
            }

            bool operator==(const Point<N>& other)const{return coords == other.coords;}
            bool operator!=(const Point<N>& other)const{return !(*this == other);}

            ~Point() = default;

            void clearPoint(){
                coords.fill(0.0);
            }

        protected:
            std::array<double, N> coords;
        };
    } // namespace Geometry
    
}