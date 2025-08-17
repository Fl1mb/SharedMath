#pragma once
#include "Point.h"

namespace SharedMath
{
    namespace Geometry
    {
        template<size_t N>
        class Line{
        public:
            Line(): firstPoint(), secondPoint(){}
            Line(const Point<N>& firstPoint_, const Point<N>& secondPoint_):
                firstPoint(firstPoint_), secondPoint(secondPoint_){}
            Line(const Line<N>& other) = default;
            Line(Line<N>&& other) noexcept = default;

            Line& operator=(const Line<N>& other){
                if(*this == other)return *this;
                firstPoint = other.firstPoint;
                secondPoint = other.secondPoint;
                return *this;
            }

            Line& operator=(Line<N>&& other) noexcept{
                if(*this == other)return *this;
                firstPoint = other.firstPoint;
                secondPoint = other.secondPoint;
                other.firstPoint.clearPoint();
                other.secondPoint.clearPoint();
                return *this;
            }

            const Point<N>& getFirstPoint() const {return firstPoint;}
            const Point<N>& getSecondPoint() const {return secondPoint;}

            void setFirstPoint(const Point<N>& point){firstPoint = point;}
            void setSecondPoint(const Point<N>& point){secondPoint = point;}

            bool operator==(const Line<N>& other) const{
                return  firstPoint == other.firstPoint &&
                        secondPoint == other.secondPoint;
            }
            bool operator!=(const Line<N>& other) const{
                return !(*this == other);
            }
            
            double getLength() const{
                double sum = 0.0;
                for(auto i = 0; i < N; i++){
                    double diff = secondPoint[i] - firstPoint[i];
                    sum += diff * diff;
                }
                return sqrt(sum);
            }

        private:
            Point<N> firstPoint;
            Point<N> secondPoint;
        };
        
    } // namespace Geometry
    
} // namespace SharedMath

