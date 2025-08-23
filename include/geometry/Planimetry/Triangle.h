#pragma once

#include "Polygon.h"
#include "../Vectors.h"
#include <algorithm>

namespace SharedMath
{
    namespace Geometry
    {
        class Triangle : public Polygon<3>{
        public:
            Triangle() = default;
            Triangle(const Triangle&) = default;
            Triangle(Triangle&&) noexcept = default;
            Triangle& operator=(const Triangle&) = default;
            Triangle& operator=(Triangle&&) noexcept = default;

            Triangle(const std::array<Point2D, 3>& points);
            Triangle(const Point2D& a, const Point2D& b, const Point2D& c);

            ~Triangle() override = default;

            static bool isValidTriangle(const std::array<Point2D, 3>& vertices);

            virtual double area() const override;
            virtual double perimeter() const override;

            double getSideLength(size_t sideIndex) const;
            double getAngle(size_t vertexIndex) const;
            double getAltitude(size_t sideIndex) const;
            
            Point2D getCentroid() const;
            Point2D getCircumcenter() const;
            Point2D getIncenter() const;
            Point2D getOrthocenter() const;

            bool isEquilateral() const;
            bool isIsosceles() const;
            bool isRight() const;
            bool isAcute() const;
            bool isObtuse() const;

            bool contains(const Point2D& point) const;
            bool isPointInside(const Point2D& point) const;

            void move(const Vector2D& offset);
            void scale(double factor);
            void rotate(double angle, const Point2D& center = Point2D(0, 0));

            bool operator==(const Triangle& other) const;
            bool operator!=(const Triangle& other) const;

            Triangle operator+(const Vector2D& offset) const;
            Triangle operator-(const Vector2D& offset) const;
            Triangle& operator+=(const Vector2D& offset);
            Triangle& operator-=(const Vector2D& offset);
        


        private:
            static bool arePointsCollinear(const Point2D& a, const Point2D& b, const Point2D& c);
            static double calculateTriangleArea(const Point2D& a, const Point2D& b, const Point2D& c);

        };

    } // namespace Geometry
    
    
} // namespace SharedMath
