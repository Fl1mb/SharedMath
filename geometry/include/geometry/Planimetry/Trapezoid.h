#pragma once
#include "Polygon.h"
#include "../Vectors.h"
#include "../Line.h"
#include "constans.h"
#include <sharedmath_geometry_export.h>
#include <array>
#include <stdexcept>
#include <cmath>

namespace SharedMath
{
    namespace Geometry
    {
        class SHAREDMATH_GEOMETRY_EXPORT Trapezoid : public Polygon<4> {
        public:
            Trapezoid() = default;

            explicit Trapezoid(const std::array<Point2D, 4>& points);

            Trapezoid(const Trapezoid&) = default;
            Trapezoid(Trapezoid&&) noexcept = default;
            Trapezoid& operator=(const Trapezoid&) = default;
            Trapezoid& operator=(Trapezoid&&) noexcept = default;
            ~Trapezoid() override = default;

            static bool isTrapezoid(const std::array<Point2D, 4>& points);

            /// Returns the two parallel sides as Line2D
            Line2D getBase1() const;
            Line2D getBase2() const;

            double getHeight() const;

            double area() const override;
            double perimeter() const override;

            bool isIsosceles() const;

            /// Returns midline as Line2D (connects midpoints of non-parallel sides)
            Line2D getMidline() const;

        private:
            // Indices of the first pair of parallel sides (base1: idx0->idx1, base2: idx2->idx3 or similar)
            int base1Start_ = 0;
            int base1End_   = 1;
            int base2Start_ = 2;
            int base2End_   = 3;

            static bool areParallel(const Point2D& a, const Point2D& b,
                                    const Point2D& c, const Point2D& d);

            void findBases(const std::array<Point2D, 4>& pts);
        };

    } // namespace Geometry
} // namespace SharedMath
