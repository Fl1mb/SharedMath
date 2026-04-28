#pragma once
#include "../Point.h"
#include "../Vectors.h"
#include "Circle.h"
#include "constans.h"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace SharedMath
{
    namespace Geometry
    {
        class Ellipse {
        public:
            Ellipse() : center(0.0, 0.0), semiMajor(1.0), semiMinor(1.0) {}

            Ellipse(const Point2D& center_, double a, double b)
                : center(center_), semiMajor(a), semiMinor(b)
            {
                if (a <= Epsilon || b <= Epsilon)
                    throw std::invalid_argument("Semi-axes must be positive");
                if (semiMajor < semiMinor) std::swap(semiMajor, semiMinor);
            }

            Ellipse(double x, double y, double a, double b)
                : Ellipse(Point2D(x, y), a, b) {}

            Ellipse(const Ellipse&) = default;
            Ellipse(Ellipse&&) noexcept = default;
            Ellipse& operator=(const Ellipse&) = default;
            Ellipse& operator=(Ellipse&&) noexcept = default;
            ~Ellipse() = default;

            Point2D getCenter() const { return center; }
            double getSemiMajor() const { return semiMajor; }
            double getSemiMinor() const { return semiMinor; }

            double area() const {
                return Pi * semiMajor * semiMinor;
            }

            // Ramanujan approximation
            double perimeter() const {
                double h = (semiMajor - semiMinor) * (semiMajor - semiMinor) /
                           ((semiMajor + semiMinor) * (semiMajor + semiMinor));
                return Pi * (semiMajor + semiMinor) * (1.0 + 3.0 * h / (10.0 + std::sqrt(4.0 - 3.0 * h)));
            }

            double eccentricity() const {
                return std::sqrt(1.0 - (semiMinor * semiMinor) / (semiMajor * semiMajor));
            }

            double focalDistance() const {
                return std::sqrt(semiMajor * semiMajor - semiMinor * semiMinor);
            }

            std::pair<Point2D, Point2D> getFoci() const {
                double c = focalDistance();
                return { Point2D(center.x() - c, center.y()),
                         Point2D(center.x() + c, center.y()) };
            }

            bool contains(const Point2D& p) const {
                double dx = p.x() - center.x();
                double dy = p.y() - center.y();
                return (dx * dx) / (semiMajor * semiMajor) +
                       (dy * dy) / (semiMinor * semiMinor) <= 1.0 + Epsilon;
            }

            bool intersects(const Circle& circle) const {
                // Check if circle center is inside ellipse scaled by (radius) margin
                double dx = circle.getCenter().x() - center.x();
                double dy = circle.getCenter().y() - center.y();
                double r = circle.getRadius();
                double ea = semiMajor + r;
                double eb = semiMinor + r;
                // approximate: check if distance from ellipse to circle center <= radius
                // Use bounding approach
                double val = (dx * dx) / (ea * ea) + (dy * dy) / (eb * eb);
                // also check circle doesn't fully contain ellipse without intersection
                double val2 = (dx * dx) / (semiMajor * semiMajor) + (dy * dy) / (semiMinor * semiMinor);
                return val <= 1.0 + Epsilon || val2 <= 1.0 + Epsilon || circle.contains(center);
            }

            void move(const Vector2D& offset) {
                center = Point2D(center.x() + offset.x(), center.y() + offset.y());
            }

            void scale(double factor) {
                if (factor <= Epsilon)
                    throw std::invalid_argument("Scale factor must be positive");
                semiMajor *= factor;
                semiMinor *= factor;
            }

            bool operator==(const Ellipse& other) const {
                return center == other.center &&
                       std::abs(semiMajor - other.semiMajor) < Epsilon &&
                       std::abs(semiMinor - other.semiMinor) < Epsilon;
            }

            bool operator!=(const Ellipse& other) const {
                return !(*this == other);
            }

            Ellipse operator+(const Vector2D& offset) const {
                Ellipse result = *this;
                result.move(offset);
                return result;
            }

            Ellipse operator-(const Vector2D& offset) const {
                Ellipse result = *this;
                result.move(Vector2D(-offset.x(), -offset.y()));
                return result;
            }

        private:
            Point2D center;
            double semiMajor;
            double semiMinor;
        };

    } // namespace Geometry
} // namespace SharedMath
