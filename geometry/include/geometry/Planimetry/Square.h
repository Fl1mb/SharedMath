#pragma once
#include "Rectangle.h"
#include <array>
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class SHAREDMATH_GEOMETRY_EXPORT Square : public Rectangle {
        public:
            Square() = default;

            Square(const Point2D& bottomLeft, double side)
                : Rectangle(bottomLeft, Point2D(bottomLeft.x() + side, bottomLeft.y() + side))
            {
                if (side <= Epsilon)
                    throw std::invalid_argument("Square side must be positive");
            }

            explicit Square(const std::array<Point2D, 4>& points)
                : Rectangle(points)
            {
                if (!isSquare(points))
                    throw std::invalid_argument("Points do not form a square");
            }

            Square(const Square&) = default;
            Square(Square&&) noexcept = default;
            Square& operator=(const Square&) = default;
            Square& operator=(Square&&) noexcept = default;
            ~Square() override = default;

            double getSide() const {
                return Rectangle::getWidth();
            }

            static bool isSquare(const std::array<Point2D, 4>& points) {
                if (!Rectangle::isRectangle(points))
                    return false;
                // All sides equal
                auto dist = [](const Point2D& a, const Point2D& b) {
                    return std::hypot(b.x() - a.x(), b.y() - a.y());
                };
                double d01 = dist(points[0], points[1]);
                double d12 = dist(points[1], points[2]);
                return std::abs(d01 - d12) < Epsilon;
            }

            double area() const override {
                double s = getSide();
                return s * s;
            }

            double perimeter() const override {
                return 4.0 * getSide();
            }
        };

    } // namespace Geometry
} // namespace SharedMath
