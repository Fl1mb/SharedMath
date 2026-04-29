#pragma once
#include "../Point.h"
#include "../Vectors.h"
#include "Rectangle.h"
#include "constans.h"
#include <sharedmath_geometry_export.h>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace SharedMath
{
    namespace Geometry
    {
        class SHAREDMATH_GEOMETRY_EXPORT DynamicPolygon {
        public:
            DynamicPolygon() = default;

            explicit DynamicPolygon(const std::vector<Point2D>& points);

            DynamicPolygon(const DynamicPolygon&) = default;
            DynamicPolygon(DynamicPolygon&&) noexcept = default;
            DynamicPolygon& operator=(const DynamicPolygon&) = default;
            DynamicPolygon& operator=(DynamicPolygon&&) noexcept = default;
            ~DynamicPolygon() = default;

            double area() const;
            double perimeter() const;

            size_t vertexCount() const;
            const Point2D& vertex(size_t i) const;

            void addVertex(const Point2D& p);
            void removeVertex(size_t i);

            bool isConvex() const;
            bool contains(const Point2D& p) const;

            void move(const Vector2D& offset);

            Rectangle getBoundingBox() const;

            bool operator==(const DynamicPolygon& other) const;
            bool operator!=(const DynamicPolygon& other) const;

        private:
            std::vector<Point2D> vertices_;
        };

    } // namespace Geometry
} // namespace SharedMath
