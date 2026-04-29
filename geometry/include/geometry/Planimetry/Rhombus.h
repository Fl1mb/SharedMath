#pragma once
#include "Parallelogram.h"
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
        class SHAREDMATH_GEOMETRY_EXPORT Rhombus : public Parallelogram {
        public:
            Rhombus() = default;

            explicit Rhombus(const std::array<Point2D, 4>& points);

            Rhombus(const Rhombus&) = default;
            Rhombus(Rhombus&&) noexcept = default;
            Rhombus& operator=(const Rhombus&) = default;
            Rhombus& operator=(Rhombus&&) noexcept = default;
            ~Rhombus() override = default;

            static bool isRhombus(const std::array<Point2D, 4>& points);

            // Returns diagonal from vertex 0 to vertex 2
            Line2D getDiagonal1() const;
            // Returns diagonal from vertex 1 to vertex 3
            Line2D getDiagonal2() const;

            // Returns the acute angle (in radians) of the rhombus
            double getAngle() const;

            // Area = d1 * d2 / 2
            double area() const override;

            // Inradius = area / (2 * side)
            double inradius() const;
        };

    } // namespace Geometry
} // namespace SharedMath
