#pragma once

#include "Point.h"
#include "Vectors.h"
#include "constans.h"
#include <algorithm>
#include <array>
#include <tuple>
#include <utility>
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        /// Cartesian (Decart) coordinate system with optional rotation
        class CartesianCoordinateSystem {
        public:
            CartesianCoordinateSystem()
                : origin_(0.0, 0.0), angle_(0.0) {}

            CartesianCoordinateSystem(const Point2D& origin, double angle = 0.0)
                : origin_(origin), angle_(angle) {}

            CartesianCoordinateSystem(const CartesianCoordinateSystem&) = default;
            CartesianCoordinateSystem& operator=(const CartesianCoordinateSystem&) = default;
            ~CartesianCoordinateSystem() = default;

            Point2D getOrigin() const { return origin_; }
            double getAngle() const { return angle_; }

            void setOrigin(const Point2D& origin) { origin_ = origin; }
            void setAngle(double angle) { angle_ = angle; }

            /// Transform a local point to global coordinates
            Point2D toGlobal(const Point2D& local) const {
                double cosA = std::cos(angle_);
                double sinA = std::sin(angle_);
                return Point2D(
                    origin_.x() + local.x() * cosA - local.y() * sinA,
                    origin_.y() + local.x() * sinA + local.y() * cosA
                );
            }

            /// Transform a global point to local coordinates
            Point2D toLocal(const Point2D& global) const {
                double dx = global.x() - origin_.x();
                double dy = global.y() - origin_.y();
                double cosA = std::cos(angle_);
                double sinA = std::sin(angle_);
                return Point2D(
                     dx * cosA + dy * sinA,
                    -dx * sinA + dy * cosA
                );
            }

            /// Convert polar coordinates to Cartesian Point2D
            static Point2D fromPolar(double r, double theta) {
                return Point2D(r * std::cos(theta), r * std::sin(theta));
            }

        private:
            Point2D origin_;
            double angle_;
        };

        /// Keep old name for backward compatibility
        using DecartCoordinateSystem = CartesianCoordinateSystem;

        /// Polar coordinate system
        class PolarCoordinateSystem {
        public:
            PolarCoordinateSystem() : origin_(0.0, 0.0) {}
            explicit PolarCoordinateSystem(const Point2D& origin) : origin_(origin) {}

            PolarCoordinateSystem(const PolarCoordinateSystem&) = default;
            PolarCoordinateSystem& operator=(const PolarCoordinateSystem&) = default;
            ~PolarCoordinateSystem() = default;

            Point2D getOrigin() const { return origin_; }
            void setOrigin(const Point2D& o) { origin_ = o; }

            // Convert Cartesian point to polar (r, theta)
            static std::pair<double, double> toPolar(const Point2D& cartesian) {
                double r = std::sqrt(cartesian.x() * cartesian.x() + cartesian.y() * cartesian.y());
                double theta = std::atan2(cartesian.y(), cartesian.x());
                return { r, theta };
            }

            /// Convert polar to Cartesian
            static Point2D toCartesian(double r, double theta) {
                return Point2D(r * std::cos(theta), r * std::sin(theta));
            }

            // Convert global Cartesian point to local polar coords (r, theta) relative to origin
            std::pair<double, double> toLocal(const Point2D& globalCartesian) const {
                Point2D rel(globalCartesian.x() - origin_.x(),
                            globalCartesian.y() - origin_.y());
                return toPolar(rel);
            }

            /// Convert local polar coords to global Cartesian point
            Point2D toGlobal(double r, double theta) const {
                return Point2D(origin_.x() + r * std::cos(theta),
                               origin_.y() + r * std::sin(theta));
            }

        private:
            Point2D origin_;
        };

        /// Cylindrical coordinate system (r, theta, z)
        class CylyndricalCoordinateSystem {
        public:
            CylyndricalCoordinateSystem() = default;
            CylyndricalCoordinateSystem(const CylyndricalCoordinateSystem&) = default;
            CylyndricalCoordinateSystem& operator=(const CylyndricalCoordinateSystem&) = default;
            ~CylyndricalCoordinateSystem() = default;

            static Point3D toCartesian(double r, double theta, double z) {
                return Point3D(r * std::cos(theta), r * std::sin(theta), z);
            }

            static std::tuple<double, double, double> toCylindrical(const Point3D& p) {
                double r = std::sqrt(p.x() * p.x() + p.y() * p.y());
                double theta = std::atan2(p.y(), p.x());
                return { r, theta, p.z() };
            }
        };

        /// Spherical coordinate system (r, theta, phi) — physics convention:
        ///   theta = polar angle from +Z, phi = azimuthal angle in XY from +X
        class SphericalCoordinateSystem {
        public:
            SphericalCoordinateSystem() = default;
            SphericalCoordinateSystem(const SphericalCoordinateSystem&) = default;
            SphericalCoordinateSystem& operator=(const SphericalCoordinateSystem&) = default;
            ~SphericalCoordinateSystem() = default;

            static Point3D toCartesian(double r, double theta, double phi) {
                return Point3D(
                    r * std::sin(theta) * std::cos(phi),
                    r * std::sin(theta) * std::sin(phi),
                    r * std::cos(theta)
                );
            }

            static std::tuple<double, double, double> toSpherical(const Point3D& p) {
                double r = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
                double theta = (r > Epsilon) ? std::acos(p.z() / r) : 0.0;
                double phi = std::atan2(p.y(), p.x());
                return { r, theta, phi };
            }
        };

    } // namespace Geometry
} // namespace SharedMath
