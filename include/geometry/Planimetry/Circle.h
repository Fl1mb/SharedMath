#pragma once

#include "../Point.h"
#include "../../constans.h"
#include "../Vectors.h"
#include <math.h>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class Circle{
        public:
            Circle() : center(0.0, 0.0), radius(1.0){}
            Circle(const Point2D& center_, double radius_) :
                center(center_), radius(radius_) {
                    if(radius_ < Epsilon)throw std::invalid_argument("Radius must be positive");
                }

            Circle(double x, double y, double radius_) :
                center(x, y), radius(radius_){
                    if(radius_ < Epsilon)throw std::invalid_argument("Radius must be positive");
                }

            Circle(const Circle&) = default;
            Circle(Circle&&) noexcept = default;
            Circle& operator=(const Circle&) = default;
            Circle& operator=(Circle&&) noexcept = default;

            ~Circle() = default;

            Point2D getCenter() const{return center;}
            double getRadius() const{return radius;}
            double getDiameter() const{return 2.0 * radius;}

            void setCenter(const Point2D& center_) {center = center_;}
            void setRadius(double rad){
                if(rad < Epsilon)throw std::invalid_argument("Radius must be positive");
                radius = rad;
            }

            double area() const {return Pi * radius * radius;}
            double length() const {return 2.0 * Pi * radius;}

            double contains(const Point2D& point){
                double dx = point.x() - center.x();
                double dy = point.y() - center.y();

                return (dx * dx + dy * dy) <= (radius * radius + Epsilon);
            }

            bool intersects(const Circle& other) const{
                double dx = center.x() - other.center.x();
                double dy = center.y() - other.center.y();
                double distanceSquared = dx * dx + dy * dy;
                double sumRadii = radius + other.radius;
                return distanceSquared <= (sumRadii * sumRadii + Epsilon);
            }

            bool isTangent(const Circle& other) const {
                double dx = center.x() - other.center.x();
                double dy = center.y() - other.center.y();
                double distance = std::sqrt(dx * dx + dy * dy);
                double sumRadii = radius + other.radius;
                double diffRadii = std::abs(radius - other.radius);
                return std::abs(distance - sumRadii) < Epsilon || 
                       std::abs(distance - diffRadii) < Epsilon;
            }

            void move(const Vector2D& offset) {
                center = Point2D(center.x() + offset.x(), center.y() + offset.y());
            }

            void scale(double factor){
                if(factor < Epsilon)throw std::invalid_argument("Factor must be positive");
                radius *= factor;
            }

            bool operator==(const Circle& other) const {
                return center == other.center && std::abs(radius - other.radius) < Epsilon;
            }

            bool operator!=(const Circle& other) const {
                return !(*this == other);
            }

             Circle operator+(const Vector2D& offset) const {
                return Circle(Point2D(center.x() + offset.x(), center.y() + offset.y()), radius);
            }

            Circle operator-(const Vector2D& offset) const {
                return Circle(Point2D(center.x() - offset.x(), center.y() - offset.y()), radius);
            }

            Circle& operator+=(const Vector2D& offset) {
                center = Point2D(center.x() + offset.x(), center.y() + offset.y());
                return *this;
            }

            Circle& operator-=(const Vector2D& offset) {
                center = Point2D(center.x() - offset.x(), center.y() - offset.y());
                return *this;
            }

            bool isConcentric(const Circle& other) const {
                return center == other.center;
            }

            double distanceTo(const Point2D& point) const {
                double dx = point.x() - center.x();
                double dy = point.y() - center.y();
                return std::max(0.0, std::sqrt(dx * dx + dy * dy) - radius);
            }

        private:
            Point2D center;
            double radius;

        };
        
    } // namespace Geometry
    
    
} // namespace SharedMath
