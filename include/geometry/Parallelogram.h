#pragma once
#include "Point.h"
#include "Polygon.h"

namespace SharedMath
{
    namespace Geometry
    {
        class Parallelogram : public Polygon<4>{
        public:
            Parallelogram() = default;

            Parallelogram(const std::array<Point<2>, 4>& points);

            Parallelogram(const Parallelogram&) = default;
            Parallelogram(Parallelogram&&) noexcept = default;
            Parallelogram& operator=(const Parallelogram&) = default;
            Parallelogram& operator=(Parallelogram&&) noexcept = default;
            ~Parallelogram() override = default;

            static bool isParallelogram(const std::array<Point<2>, 4>& vertices);

            //TODO
            virtual double area() const override;
            virtual double perimeter() const override;

        protected:
            
        };

    } // namespace Geometry
    
    
} // namespace SharedMath

