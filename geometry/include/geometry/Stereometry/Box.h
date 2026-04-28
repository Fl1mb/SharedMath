#pragma once
#include "../Point.h"
#include "../Vectors.h"
#include "constans.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace SharedMath
{
    namespace Geometry
    {
        class Box {
        public:
            Box() : minCorner_(0.0, 0.0, 0.0), maxCorner_(1.0, 1.0, 1.0) {}

            Box(const Point3D& minCorner, const Point3D& maxCorner)
                : minCorner_(minCorner), maxCorner_(maxCorner)
            {
                // Ensure min <= max on each axis
                if (minCorner_.x() > maxCorner_.x() ||
                    minCorner_.y() > maxCorner_.y() ||
                    minCorner_.z() > maxCorner_.z())
                    throw std::invalid_argument("Box: minCorner must be <= maxCorner on all axes");
            }

            Box(const Point3D& center, double width, double height, double depth)
            {
                if (width <= 0 || height <= 0 || depth <= 0)
                    throw std::invalid_argument("Box: dimensions must be positive");
                minCorner_ = Point3D(center.x() - width / 2.0,
                                     center.y() - height / 2.0,
                                     center.z() - depth / 2.0);
                maxCorner_ = Point3D(center.x() + width / 2.0,
                                     center.y() + height / 2.0,
                                     center.z() + depth / 2.0);
            }

            Box(const Box&) = default;
            Box(Box&&) noexcept = default;
            Box& operator=(const Box&) = default;
            Box& operator=(Box&&) noexcept = default;
            ~Box() = default;

            Point3D getMinCorner() const { return minCorner_; }
            Point3D getMaxCorner() const { return maxCorner_; }

            double getWidth()  const { return maxCorner_.x() - minCorner_.x(); }
            double getHeight() const { return maxCorner_.y() - minCorner_.y(); }
            double getDepth()  const { return maxCorner_.z() - minCorner_.z(); }

            Point3D getCenter() const {
                return Point3D(
                    (minCorner_.x() + maxCorner_.x()) / 2.0,
                    (minCorner_.y() + maxCorner_.y()) / 2.0,
                    (minCorner_.z() + maxCorner_.z()) / 2.0
                );
            }

            double volume() const {
                return getWidth() * getHeight() * getDepth();
            }

            double surfaceArea() const {
                double w = getWidth(), h = getHeight(), d = getDepth();
                return 2.0 * (w * h + h * d + w * d);
            }

            bool contains(const Point3D& p) const {
                return p.x() >= minCorner_.x() - Epsilon && p.x() <= maxCorner_.x() + Epsilon &&
                       p.y() >= minCorner_.y() - Epsilon && p.y() <= maxCorner_.y() + Epsilon &&
                       p.z() >= minCorner_.z() - Epsilon && p.z() <= maxCorner_.z() + Epsilon;
            }

            bool intersects(const Box& other) const {
                return !(maxCorner_.x() < other.minCorner_.x() || minCorner_.x() > other.maxCorner_.x() ||
                         maxCorner_.y() < other.minCorner_.y() || minCorner_.y() > other.maxCorner_.y() ||
                         maxCorner_.z() < other.minCorner_.z() || minCorner_.z() > other.maxCorner_.z());
            }

            Box merge(const Box& other) const {
                return Box(
                    Point3D(std::min(minCorner_.x(), other.minCorner_.x()),
                            std::min(minCorner_.y(), other.minCorner_.y()),
                            std::min(minCorner_.z(), other.minCorner_.z())),
                    Point3D(std::max(maxCorner_.x(), other.maxCorner_.x()),
                            std::max(maxCorner_.y(), other.maxCorner_.y()),
                            std::max(maxCorner_.z(), other.maxCorner_.z()))
                );
            }

            void move(const Vector3D& offset) {
                minCorner_ = Point3D(minCorner_.x() + offset.x(),
                                     minCorner_.y() + offset.y(),
                                     minCorner_.z() + offset.z());
                maxCorner_ = Point3D(maxCorner_.x() + offset.x(),
                                     maxCorner_.y() + offset.y(),
                                     maxCorner_.z() + offset.z());
            }

            void scale(double factor) {
                if (factor <= Epsilon)
                    throw std::invalid_argument("Scale factor must be positive");
                Point3D c = getCenter();
                double hw = getWidth() * factor / 2.0;
                double hh = getHeight() * factor / 2.0;
                double hd = getDepth() * factor / 2.0;
                minCorner_ = Point3D(c.x() - hw, c.y() - hh, c.z() - hd);
                maxCorner_ = Point3D(c.x() + hw, c.y() + hh, c.z() + hd);
            }

            bool operator==(const Box& other) const {
                return minCorner_ == other.minCorner_ && maxCorner_ == other.maxCorner_;
            }

            bool operator!=(const Box& other) const {
                return !(*this == other);
            }

        private:
            Point3D minCorner_;
            Point3D maxCorner_;
        };

    } // namespace Geometry
} // namespace SharedMath
