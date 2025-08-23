#pragma once

#include "../Point.h"
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
            explicit Polygon(const std::array<Point2D, VertexCount>& vertices) : VerticesPoints(vertices){}

            Polygon(const Polygon&) = default;
            Polygon(Polygon&&) noexcept = default;
            Polygon& operator=(const Polygon&) = default;
            Polygon& operator=(Polygon&&) noexcept = default;
            virtual ~Polygon() = default;

            bool operator==(const Polygon& other)const {return VerticesPoints == other.VerticesPoints;}
            bool operator!=(const Polygon& other)const {return !(*this == other);}

            const Point2D& vertex(size_t index) const{
                if(index >= VertexCount)throw std::invalid_argument("Polygon::vertex index out of range");
                return VerticesPoints[index];
            }
            Point2D& vertex(size_t index){
                if(index >= VertexCount)throw std::invalid_argument("Polygon::vertex index out of range");
                return VerticesPoints[index];
            }

            virtual double area() const override{
                double sum = 0.0;
                for(size_t i = 0; i < VertexCount; ++i){
                    size_t j = (i + 1) % VertexCount;
                    sum += (VerticesPoints[i].x() * VerticesPoints[j].y()) - 
                       (VerticesPoints[j].x() * VerticesPoints[i].y());
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

            void setVertices(const std::array<Point2D, VertexCount>& points){
                VerticesPoints = points;
            }
            decltype(auto) getVertices() const{return VerticesPoints;}

        protected:
            std::array<Point2D, VertexCount> VerticesPoints;

            static double distance(const Point2D& firstPoint, const Point2D& secondPoint){
                double dx = secondPoint.x() - firstPoint.x();
                double dy = secondPoint.y() - firstPoint.y();
                return std::hypot(dx, dy); 
            }

            static double crossProduct(const Point2D& first, const Point2D& second, const Point2D& third){
                return (second.x() - first.x() * (third.y() - first.y())-
                        second.y() - first.y() * (third.x() - first.x()));
            }
            
        };
    } // namespace Geometry
    
} // namespace SharedMath
