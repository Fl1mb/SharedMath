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

            void setFirstPoint(Point<N>&& point){
                firstPoint = point;
                point.clearPoint();
            }
            void setSecondPoint(Point<N>&& point){
                secondPoint = point;
                point.clearPoint();
            }

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
            
        protected:
            Point<N> firstPoint;
            Point<N> secondPoint;
        };

        class Line2D : public Line<2> {
        public:
            Line2D() : Line<2>() {}
            Line2D(const Point2D& firstPoint_, const Point2D& secondPoint_)
                : Line<2>(firstPoint_, secondPoint_) {}
            Line2D(double x1, double y1, double x2, double y2)
                : Line<2>(Point2D(x1, y1), Point2D(x2, y2)) {}
            
            Line2D(const Line2D& other) = default;
            Line2D(Line2D&& other) noexcept = default;

            Line2D& operator=(const Line2D& other) {
                Line<2>::operator=(other);
                return *this;
            }

            Line2D& operator=(Line2D&& other) noexcept {
                Line<2>::operator=(std::move(other));
                return *this;
            }

            double getSlope() const {
                double dx = secondPoint[0] - firstPoint[0];
                double dy = secondPoint[1] - firstPoint[1];
                
                if (dx == 0.0) {
                    return std::numeric_limits<double>::infinity();
                }
                return dy / dx;
            }

            double getYIntercept() const {
                double slope = getSlope();
                if (std::isinf(slope)) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                return firstPoint[1] - slope * firstPoint[0];
            }

            bool isVertical() const {
                return firstPoint[0] == secondPoint[0];
            }

            bool isHorizontal() const {
                return firstPoint[1] == secondPoint[1];
            }

            Point2D getFirstPoint2D() const {
                return Point2D(firstPoint[0], firstPoint[1]);
            }

            Point2D getSecondPoint2D() const {
                return Point2D(secondPoint[0], secondPoint[1]);
            }

            void setFirstPoint2D(const Point2D& point) {
                setFirstPoint(point);
            }

            void setSecondPoint2D(const Point2D& point) {
                setSecondPoint(point);
            }

            void setFirstPoint2D(double x, double y) {
                setFirstPoint(Point2D(x, y));
            }

            void setSecondPoint2D(double x, double y) {
                setSecondPoint(Point2D(x, y));
            }
        };

        class Line3D : public Line<3> {
        public:
            Line3D() : Line<3>() {}
            Line3D(const Point3D& firstPoint_, const Point3D& secondPoint_)
                : Line<3>(firstPoint_, secondPoint_) {}
            Line3D(double x1, double y1, double z1, double x2, double y2, double z2)
                : Line<3>(Point3D(x1, y1, z1), Point3D(x2, y2, z2)) {}
            
            Line3D(const Line3D& other) = default;
            Line3D(Line3D&& other) noexcept = default;

            Line3D& operator=(const Line3D& other) {
                Line<3>::operator=(other);
                return *this;
            }

            Line3D& operator=(Line3D&& other) noexcept {
                Line<3>::operator=(std::move(other));
                return *this;
            }

            Point3D getDirectionVector() const {
                return Point3D(
                    secondPoint[0] - firstPoint[0],
                    secondPoint[1] - firstPoint[1],
                    secondPoint[2] - firstPoint[2]
                );
            }

            Point3D getFirstPoint3D() const {
                return Point3D(firstPoint[0], firstPoint[1], firstPoint[2]);
            }

            Point3D getSecondPoint3D() const {
                return Point3D(secondPoint[0], secondPoint[1], secondPoint[2]);
            }

            void setFirstPoint3D(const Point3D& point) {
                setFirstPoint(point);
            }

            void setSecondPoint3D(const Point3D& point) {
                setSecondPoint(point);
            }

            void setFirstPoint3D(double x, double y, double z) {
                setFirstPoint(Point3D(x, y, z));
            }

            void setSecondPoint3D(double x, double y, double z) {
                setSecondPoint(Point3D(x, y, z));
            }

            std::array<double, 3> getParametricX(double t) const {
                return {
                    firstPoint[0] + t * (secondPoint[0] - firstPoint[0]),
                    firstPoint[1] + t * (secondPoint[1] - firstPoint[1]),
                    firstPoint[2] + t * (secondPoint[2] - firstPoint[2])
                };
            }
        };
        
    } // namespace Geometry
    
} // namespace SharedMath

