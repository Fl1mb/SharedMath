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
        class Cylinder {
        public:
            // Default: axis along Z, base at origin
            Cylinder()
                : baseCenter_(0.0, 0.0, 0.0),
                  axis_(0.0, 0.0, 1.0),
                  radius_(1.0),
                  height_(1.0) {}

            Cylinder(const Point3D& baseCenter, double radius, double height)
                : baseCenter_(baseCenter),
                  axis_(0.0, 0.0, 1.0),
                  radius_(radius),
                  height_(height)
            {
                validate();
            }

            Cylinder(const Point3D& baseCenter, const Point3D& topCenter, double radius)
                : baseCenter_(baseCenter), radius_(radius)
            {
                Vector3D ax(baseCenter, topCenter);
                height_ = ax.length();
                if (height_ < Epsilon)
                    throw std::invalid_argument("Cylinder: base and top centers cannot coincide");
                axis_ = ax.normalized();
                validate();
            }

            Cylinder(const Cylinder&) = default;
            Cylinder(Cylinder&&) noexcept = default;
            Cylinder& operator=(const Cylinder&) = default;
            Cylinder& operator=(Cylinder&&) noexcept = default;
            ~Cylinder() = default;

            Point3D getBaseCenter()  const { return baseCenter_; }
            Vector3D getAxis()       const { return axis_; }
            double getRadius()       const { return radius_; }
            double getHeight()       const { return height_; }

            Point3D getTopCenter() const {
                return Point3D(baseCenter_.x() + axis_.x() * height_,
                               baseCenter_.y() + axis_.y() * height_,
                               baseCenter_.z() + axis_.z() * height_);
            }

            Point3D getBottomCenter() const { return baseCenter_; }

            double volume() const {
                return Pi * radius_ * radius_ * height_;
            }

            double lateralArea() const {
                return 2.0 * Pi * radius_ * height_;
            }

            double totalArea() const {
                return 2.0 * Pi * radius_ * (radius_ + height_);
            }

            bool contains(const Point3D& p) const {
                // Project p onto axis, check height
                Vector3D bp(baseCenter_, p);
                double t = bp.x() * axis_.x() + bp.y() * axis_.y() + bp.z() * axis_.z();
                if (t < -Epsilon || t > height_ + Epsilon) return false;
                // Radial distance
                double rx = bp.x() - t * axis_.x();
                double ry = bp.y() - t * axis_.y();
                double rz = bp.z() - t * axis_.z();
                return (rx * rx + ry * ry + rz * rz) <= radius_ * radius_ + Epsilon;
            }

            void move(const Vector3D& offset) {
                baseCenter_ = Point3D(baseCenter_.x() + offset.x(),
                                      baseCenter_.y() + offset.y(),
                                      baseCenter_.z() + offset.z());
            }

            void scale(double factor) {
                if (factor <= Epsilon)
                    throw std::invalid_argument("Scale factor must be positive");
                radius_ *= factor;
                height_ *= factor;
            }

            bool operator==(const Cylinder& other) const {
                return baseCenter_ == other.baseCenter_ &&
                       std::abs(radius_ - other.radius_) < Epsilon &&
                       std::abs(height_ - other.height_) < Epsilon;
            }

            bool operator!=(const Cylinder& other) const {
                return !(*this == other);
            }

        private:
            Point3D  baseCenter_;
            Vector3D axis_;
            double   radius_;
            double   height_;

            void validate() const {
                if (radius_ <= Epsilon)
                    throw std::invalid_argument("Cylinder: radius must be positive");
                if (height_ <= Epsilon)
                    throw std::invalid_argument("Cylinder: height must be positive");
            }
        };

    } // namespace Geometry
} // namespace SharedMath
