#pragma once

#include "Point.h"
#include <array>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class PolygonInterface{
        public:
            virtual double area() const = 0;
            virtual double perimeter() const = 0;

        };
        
        template<size_t VertexCount>
        class Polygon : public PolygonInterface{
        public:
            static_assert(VertexCount > 2, "Polygon must have Vertex > 2");
            
            Polygon() = default;
            explicit Polygon(const std::array<Point<2>, VertexCount>& vertices) : VerticesPoints(vertices){}

            enum COORD{
                X_COORD = 0,
                Y_COORD = 1
            };

            Polygon(const Polygon&) = default;
            Polygon(Polygon&&) noexcept = default;
            Polygon& operator=(const Polygon&) = default;
            Polygon& operator=(Polygon&&) noexcept = default;
            virtual ~Polygon() = default;

            bool operator==(const Polygon& other)const {return VerticesPoints == other.VerticesPoints;}
            bool operator!=(const Polygon& other)const {return !(*this == other);}

            const Point<2>& vertex(size_t index) const{
                if(index >= VertexCount)throw std::invalid_argument("Polygon::vertex index out of range");
                return VerticesPoints[index];
            }
            Point<2>& vertex(size_t index){
                if(index >= VertexCount)throw std::invalid_argument("Polygon::vertex index out of range");
                return VerticesPoints[index];
            }

            virtual double area() const override{
                double sum = 0.0;
                for(size_t i = 0; i < VertexCount; ++i){
                    size_t j = (i + 1) % VertexCount;
                    sum += (VerticesPoints[i][X_COORD] * VerticesPoints[j][Y_COORD]) - 
                       (VerticesPoints[j][X_COORD] * VerticesPoints[i][Y_COORD]);
                }
                return std::abs(sum) / 2.0;
            }
            virtual double perimeter() const override{
                double sum = 0.0;
                for (size_t i = 0; i < VertexCount; ++i) {
                    size_t j = (i + 1) % VertexCount;
                    sum += distance(VerticesPoints[i], VerticesPoints[j]);
                }
                return sum;
            }

        protected:
            std::array<Point<2>, VertexCount> VerticesPoints;

            static double distance(const Point<2>& firstPoint, const Point<2>& secondPoint){
                double dx = secondPoint[X_COORD] - firstPoint[X_COORD];
                double dy = secondPoint[Y_COORD] - firstPoint[Y_COORD];
                return std::hypot(dx, dy); 
            }

            static double crossProduct(const Point<2>& first, const Point<2>& second, const Point<2>& third){
                return (second[X_COORD] - first[X_COORD] * (third[Y_COORD] - first[Y_COORD])-
                        second[Y_COORD] - first[Y_COORD] * (third[X_COORD] - first[X_COORD]));
            }
            
        };
    } // namespace Geometry
    
} // namespace SharedMath
