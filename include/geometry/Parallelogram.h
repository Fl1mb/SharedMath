#pragma once
#include "Point.h"
#include "Polygon.h"
#include "Vectors.h"
#include "../constans.h"
#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace SharedMath
{
    namespace Geometry
    {
        class Parallelogram : public Polygon<4>{
        public:
            Parallelogram() = default;

            Parallelogram(const std::array<Point2D, 4>& points);

            Parallelogram(const Parallelogram&) = default;
            Parallelogram(Parallelogram&&) noexcept = default;
            Parallelogram& operator=(const Parallelogram&) = default;
            Parallelogram& operator=(Parallelogram&&) noexcept = default;
            ~Parallelogram() override = default;

            static bool isParallelogram(const std::array<Point2D, 4>& vertices);

            virtual double area() const override;
            virtual double perimeter() const override;

            void SetVertices(const std::array<Point2D, 4>& vertices);
            decltype(auto) GetVertices() const {return VerticesPoints;}

        protected:
            static Point2D findDownLeftPoint(const std::array<Point2D, 4>& points);
            static Point2D findDownRightPoint(const std::array<Point2D, 4>& points);
            static Point2D findUpLeftPoint(const std::array<Point2D, 4>& points);
            static Point2D findUpRightPoint(const std::array<Point2D, 4>& points);
        };

    } // namespace Geometry
    
} // namespace SharedMath