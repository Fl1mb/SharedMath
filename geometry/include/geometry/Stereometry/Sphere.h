#pragma once
#include "../Point.h"
#include "../Vectors.h"
#include "constans.h"
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class Sphere {
        public:
            Sphere() : center_(0.0, 0.0, 0.0), radius_(1.0) {}

            Sphere(const Point3D& center, double radius)
                : center_(center), radius_(radius)
            {
                if (radius <= Epsilon)
                    throw std::invalid_argument("Sphere radius must be positive");
            }

            Sphere(double x, double y, double z, double radius)
                : Sphere(Point3D(x, y, z), radius) {}

            Sphere(const Sphere&) = default;
            Sphere(Sphere&&) noexcept = default;
            Sphere& operator=(const Sphere&) = default;
            Sphere& operator=(Sphere&&) noexcept = default;
            ~Sphere() = default;

            Point3D getCenter() const { return center_; }
            double getRadius() const { return radius_; }

            void setCenter(const Point3D& c) { center_ = c; }
            void setRadius(double r) {
                if (r <= Epsilon) throw std::invalid_argument("Sphere radius must be positive");
                radius_ = r;
            }

            double area() const {
                return 4.0 * Pi * radius_ * radius_;
            }

            double volume() const {
                return (4.0 / 3.0) * Pi * radius_ * radius_ * radius_;
            }

            bool contains(const Point3D& p) const {
                double dx = p.x() - center_.x();
                double dy = p.y() - center_.y();
                double dz = p.z() - center_.z();
                return (dx * dx + dy * dy + dz * dz) <= (radius_ * radius_ + Epsilon);
            }

            bool intersects(const Sphere& other) const {
                double dx = center_.x() - other.center_.x();
                double dy = center_.y() - other.center_.y();
                double dz = center_.z() - other.center_.z();
                double distSq = dx * dx + dy * dy + dz * dz;
                double sumR = radius_ + other.radius_;
                return distSq <= sumR * sumR + Epsilon;
            }

            double distanceTo(const Point3D& p) const {
                double dx = p.x() - center_.x();
                double dy = p.y() - center_.y();
                double dz = p.z() - center_.z();
                return std::max(0.0, std::sqrt(dx * dx + dy * dy + dz * dz) - radius_);
            }

            void move(const Vector3D& offset) {
                center_ = Point3D(center_.x() + offset.x(),
                                  center_.y() + offset.y(),
                                  center_.z() + offset.z());
            }

            void scale(double factor) {
                if (factor <= Epsilon)
                    throw std::invalid_argument("Scale factor must be positive");
                radius_ *= factor;
            }

            bool operator==(const Sphere& other) const {
                return center_ == other.center_ &&
                       std::abs(radius_ - other.radius_) < Epsilon;
            }

            bool operator!=(const Sphere& other) const {
                return !(*this == other);
            }

            Sphere operator+(const Vector3D& offset) const {
                Sphere result = *this;
                result.move(offset);
                return result;
            }

            Sphere operator-(const Vector3D& offset) const {
                Sphere result = *this;
                result.move(Vector3D(-offset.x(), -offset.y(), -offset.z()));
                return result;
            }

        private:
            Point3D center_;
            double radius_;
        };

    } // namespace Geometry
} // namespace SharedMath
