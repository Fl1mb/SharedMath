#pragma once

#include "Line.h"
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class Vector2D : public Line<2>{
        public:
            Vector2D() : Line<2>(){}
            Vector2D(const Point<2>& start, const Point<2>& end) : Line<2>(start, end) {}
            Vector2D(double x, double y) : Line<2>(Point<2>(0,0), Point<2>(x,y)) {}

            explicit Vector2D(const Point<2>& point) : Line<2>(Point<2>(0,0), point) {}

            double length() const{
                return Line<2>::getLength();
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

            static double getAngle(const Line<2>& first, const Line<2>& second){
                Vector2D vec1(first.getFirstPoint(), first.getSecondPoint());
                Vector2D vec2(second.getFirstPoint(), second.getSecondPoint());
                
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

        class Vector3D : public Line<3>{
        public:
            Vector3D() : Line<3>() {}
            Vector3D(const Point<3>& start, const Point<3>& end) : Line<3>(start, end) {}
            Vector3D(double x, double y, double z) : Line<3>(Point<3>(0, 0, 0), Point<3>(x, y, z)) {}

            explicit Vector3D(const Point<3>& point) : Line<3>(Point<3>(0, 0, 0), point) {}

            double x() const { return getSecondPoint()[0] - getFirstPoint()[0]; }
            double y() const { return getSecondPoint()[1] - getFirstPoint()[1]; }
            double z() const { return getSecondPoint()[2] - getFirstPoint()[2]; }

            double length() const {
                return Line<3>::getLength();
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

            static double getAngle(const Line<3>& first, const Line<3>& second){
                Vector3D vec1(first.getFirstPoint(), first.getSecondPoint());
                Vector3D vec2(second.getFirstPoint(), second.getSecondPoint());
                
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
