#pragma once
#include "Parallelogram.h"

namespace SharedMath
{
    namespace Geometry
    {
        class Rectangle : public Parallelogram {
        public:
            Rectangle() = default;

            Rectangle(const std::array<Point2D, 4>& points);

            Rectangle(const Point2D& bottomLeft, const Point2D& topRight);

            Rectangle(const Point2D& position, double width, double height);

            Rectangle(const Rectangle&) = default;
            Rectangle(Rectangle&&) noexcept = default;
            Rectangle& operator=(const Rectangle&) = default;
            Rectangle& operator=(Rectangle&&) noexcept = default;
            ~Rectangle() override = default;

            static bool isRectangle(const std::array<Point2D, 4>& vertices);

            double getWidth() const;
            double getHeight() const;
            double getAspectRatio() const;
            
            Point2D getCenter() const;
            Point2D getBottomLeft() const;
            Point2D getBottomRight() const;
            Point2D getTopLeft() const;
            Point2D getTopRight() const;

            void setSize(double width, double height);
            void setPosition(const Point2D& position);
            void move(const Vector2D& offset);
            void scale(double factor);
            void scale(double widthFactor, double heightFactor);

            bool isSquare() const;
            bool contains(const Point2D& point) const;
            bool intersects(const Rectangle& other) const;

            bool operator==(const Rectangle& other) const;
            bool operator!=(const Rectangle& other) const;

            Rectangle operator+(const Vector2D& offset) const;
            Rectangle operator-(const Vector2D& offset) const;
            Rectangle& operator+=(const Vector2D& offset);
            Rectangle& operator-=(const Vector2D& offset);

        private:
            static std::array<Point2D, 4> orderRectanglePoints(const std::array<Point2D, 4>& points);
        };

    } // namespace Geometry
}