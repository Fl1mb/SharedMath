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
        class Cone {
        public:
            // Default: apex at (0,0,1), base at origin, radius=1
            Cone()
                : apex_(0.0, 0.0, 1.0),
                  baseCenter_(0.0, 0.0, 0.0),
                  radius_(1.0) {}

            // (baseCenter, apex, radius)
            Cone(const Point3D& baseCenter, const Point3D& apex, double radius)
                : apex_(apex), baseCenter_(baseCenter), radius_(radius)
            {
                validate();
            }

            // (baseCenter, height, radius) — apex directly above
            Cone(const Point3D& baseCenter, double height, double radius)
                : baseCenter_(baseCenter), radius_(radius)
            {
                if (height <= Epsilon)
                    throw std::invalid_argument("Cone: height must be positive");
                apex_ = Point3D(baseCenter.x(), baseCenter.y(), baseCenter.z() + height);
                validate();
            }

            Cone(const Cone&) = default;
            Cone(Cone&&) noexcept = default;
            Cone& operator=(const Cone&) = default;
            Cone& operator=(Cone&&) noexcept = default;
            ~Cone() = default;

            Point3D getApex()       const { return apex_; }
            Point3D getBaseCenter() const { return baseCenter_; }
            double  getRadius()     const { return radius_; }

            double height() const {
                double dx = apex_.x() - baseCenter_.x();
                double dy = apex_.y() - baseCenter_.y();
                double dz = apex_.z() - baseCenter_.z();
                return std::sqrt(dx * dx + dy * dy + dz * dz);
            }

            double slantHeight() const {
                double h = height();
                return std::sqrt(h * h + radius_ * radius_);
            }

            double volume() const {
                return Pi * radius_ * radius_ * height() / 3.0;
            }

            double lateralArea() const {
                return Pi * radius_ * slantHeight();
            }

            double totalArea() const {
                return lateralArea() + Pi * radius_ * radius_;
            }

            Vector3D getAxis() const {
                Vector3D ax(baseCenter_, apex_);
                double len = ax.length();
                if (len < Epsilon) return Vector3D(0.0, 0.0, 1.0);
                return ax.normalized();
            }

            bool contains(const Point3D& p) const {
                // For a right circular cone: point inside if
                // distance from axis scaled by (h-t)/h >= radial distance
                Vector3D axis = getAxis();
                double h = height();
                if (h < Epsilon) return false;
                Vector3D bp(baseCenter_, p);
                double t = bp.x() * axis.x() + bp.y() * axis.y() + bp.z() * axis.z();
                if (t < -Epsilon || t > h + Epsilon) return false;
                double rx = bp.x() - t * axis.x();
                double ry = bp.y() - t * axis.y();
                double rz = bp.z() - t * axis.z();
                double radialDist2 = rx * rx + ry * ry + rz * rz;
                double maxR = radius_ * (1.0 - t / h);
                return radialDist2 <= maxR * maxR + Epsilon;
            }

            void move(const Vector3D& offset) {
                apex_       = Point3D(apex_.x()       + offset.x(), apex_.y()       + offset.y(), apex_.z()       + offset.z());
                baseCenter_ = Point3D(baseCenter_.x() + offset.x(), baseCenter_.y() + offset.y(), baseCenter_.z() + offset.z());
            }

            void scale(double factor) {
                if (factor <= Epsilon)
                    throw std::invalid_argument("Scale factor must be positive");
                radius_ *= factor;
                // Scale apex relative to base center
                apex_ = Point3D(
                    baseCenter_.x() + (apex_.x() - baseCenter_.x()) * factor,
                    baseCenter_.y() + (apex_.y() - baseCenter_.y()) * factor,
                    baseCenter_.z() + (apex_.z() - baseCenter_.z()) * factor
                );
            }

            bool operator==(const Cone& other) const {
                return apex_ == other.apex_ &&
                       baseCenter_ == other.baseCenter_ &&
                       std::abs(radius_ - other.radius_) < Epsilon;
            }

            bool operator!=(const Cone& other) const {
                return !(*this == other);
            }

        private:
            Point3D apex_;
            Point3D baseCenter_;
            double  radius_;

            void validate() const {
                if (radius_ <= Epsilon)
                    throw std::invalid_argument("Cone: radius must be positive");
                double dx = apex_.x() - baseCenter_.x();
                double dy = apex_.y() - baseCenter_.y();
                double dz = apex_.z() - baseCenter_.z();
                if (std::sqrt(dx * dx + dy * dy + dz * dz) < Epsilon)
                    throw std::invalid_argument("Cone: apex and base center cannot coincide");
            }
        };

    } // namespace Geometry
} // namespace SharedMath
