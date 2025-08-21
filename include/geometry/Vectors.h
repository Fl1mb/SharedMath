#pragma once

#include "Line.h"
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class Vector2D : public Line2D{
        public:
            Vector2D() : Line2D(){}
            Vector2D(const Point2D& start, const Point2D& end) : Line2D(start, end) {}
            Vector2D(double x, double y) : Line2D(Point2D(0,0), Point2D(x,y)) {}

            explicit Vector2D(const Point2D& point) : Line2D(Point2D(0,0), point) {}

            double length() const{
                return Line2D::getLength();
            }

            double x() const { return getSecondPoint()[0] - getFirstPoint()[0]; }
            double y() const { return getSecondPoint()[1] - getFirstPoint()[1]; }

             Vector2D normalized() const {
                double len = length();
                if (len < 1e-10) {
                    throw std::invalid_argument("Cannot normalize zero vector");
                }
                return Vector2D(x() / len, y() / len);
            }

            double dot(const Vector2D& other) const {
                return x() * other.x() + y() * other.y();
            }

            double cross(const Vector2D& other) const {
                return x() * other.y() - y() * other.x();
            }

            Vector2D operator+(const Vector2D& other) const {
                return Vector2D(x() + other.x(), y() + other.y());
            }

            Vector2D operator-(const Vector2D& other) const {
                return Vector2D(x() - other.x(), y() - other.y());
            }

            Vector2D operator*(double scalar) const {
                return Vector2D(x() * scalar, y() * scalar);
            }

            Vector2D operator/(double scalar) const {
                if (std::abs(scalar) < Epsilon) {
                    throw std::invalid_argument("Division by zero");
                }
                return Vector2D(x() / scalar, y() / scalar);
            }

            bool isZero() const {
                return std::abs(x()) < Epsilon && std::abs(y()) < Epsilon;
            }

            bool isParallel(const Vector2D& other) const {
                if (isZero() || other.isZero()) return false;
                return std::abs(cross(other)) < Epsilon;
            }

            bool isPerpendicular(const Vector2D& other) const {
                return std::abs(dot(other)) < Epsilon;
            }

            Vector2D rotate(double angle) const {
                double cosA = std::cos(angle);
                double sinA = std::sin(angle);
                return Vector2D(
                    x() * cosA - y() * sinA,
                    x() * sinA + y() * cosA
                );
            }

            Vector2D normal() const {
                return Vector2D(-y(), x());
            }

            static double getAngle(const Line2D& first, const Line2D& second){
                Vector2D vec1(first.getFirstPoint2D(), first.getSecondPoint2D());
                Vector2D vec2(second.getFirstPoint2D(), second.getSecondPoint2D());
                
                double dotProduct = vec1.dot(vec2);
                
                double length1 = vec1.length();
                double length2 = vec2.length();
  
                if (length1 < Epsilon || length2 < Epsilon) {
                    throw std::invalid_argument("One or both lines have zero length");
                }
                
                double cosAngle = dotProduct / (length1 * length2);
                
                cosAngle = std::max(-1.0, std::min(1.0, cosAngle));

                return std::acos(cosAngle);
            }
        };

        class Vector3D : public Line3D{
        public:
            Vector3D() : Line3D() {}
            Vector3D(const Point3D& start, const Point3D& end) : Line3D(start, end) {}
            Vector3D(double x, double y, double z) : Line3D(Point3D(0, 0, 0), Point3D(x, y, z)) {}

            explicit Vector3D(const Point3D& point) : Line3D(Point3D(0, 0, 0), point) {}

            double x() const { return getSecondPoint()[0] - getFirstPoint()[0]; }
            double y() const { return getSecondPoint()[1] - getFirstPoint()[1]; }
            double z() const { return getSecondPoint()[2] - getFirstPoint()[2]; }

            double length() const {
                return Line3D::getLength();
            }

            Vector3D normalized() const {
                double len = length();
                if (len < Epsilon) {
                    throw std::invalid_argument("Cannot normalize zero vector");
                }
                return Vector3D(x() / len, y() / len, z() / len);
            }

            double dot(const Vector3D& other) const {
                return x() * other.x() + y() * other.y() + z() * other.z();
            }

            Vector3D cross(const Vector3D& other) const {
                return Vector3D(
                    y() * other.z() - z() * other.y(),
                    z() * other.x() - x() * other.z(),
                    x() * other.y() - y() * other.x()
                );
            }

            Vector3D operator+(const Vector3D& other) const {
                return Vector3D(x() + other.x(), y() + other.y(), z() + other.z());
            }

            Vector3D operator-(const Vector3D& other) const {
                return Vector3D(x() - other.x(), y() - other.y(), z() - other.z());
            }

            Vector3D operator*(double scalar) const {
                return Vector3D(x() * scalar, y() * scalar, z() * scalar);
            }

            Vector3D operator/(double scalar) const {
                if (std::abs(scalar) < 1e-10) {
                    throw std::invalid_argument("Division by zero");
                }
                return Vector3D(x() / scalar, y() / scalar, z() / scalar);
            }

            bool isZero() const {
                return std::abs(x()) < 1e-10 && std::abs(y()) < 1e-10 && std::abs(z()) < 1e-10;
            }

            bool isParallel(const Vector3D& other) const {
                if (isZero() || other.isZero()) return false;
                
                Vector3D crossProduct = cross(other);
                return crossProduct.isZero();
            }

            bool isPerpendicular(const Vector3D& other) const {
                return std::abs(dot(other)) < 1e-10;
            }

            double tripleProduct(const Vector3D& b, const Vector3D& c) const {
                return dot(b.cross(c));
            }

            static double getAngle(const Line3D& first, const Line3D& second){
                Vector3D vec1(first.getFirstPoint3D(), first.getSecondPoint3D());
                Vector3D vec2(second.getFirstPoint3D(), second.getSecondPoint3D());
                
                double dotProduct = vec1.dot(vec2);

                double length1 = vec1.length();
                double length2 = vec2.length();
                
                if (length1 < Epsilon || length2 < Epsilon) {
                    throw std::invalid_argument("One or both lines have zero length");
                }
                
                double cosAngle = dotProduct / (length1 * length2);
                
                cosAngle = std::max(-1.0, std::min(1.0, cosAngle));
                
                return std::acos(cosAngle);
            }
        };


    } // namespace Geometry
        
} // namespace SharedMath
