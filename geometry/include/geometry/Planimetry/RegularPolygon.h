#pragma once
#include "Polygon.h"
#include "constans.h"
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        template<size_t N>
        class RegularPolygon : public Polygon<N> {
            static_assert(N >= 3, "RegularPolygon requires at least 3 sides");
        public:
            RegularPolygon() : center_(0.0, 0.0), circumradius_(1.0) {
                computeVertices();
            }

            RegularPolygon(const Point2D& center, double circumradius)
                : center_(center), circumradius_(circumradius)
            {
                if (circumradius <= Epsilon)
                    throw std::invalid_argument("Circumradius must be positive");
                computeVertices();
            }

            RegularPolygon(const RegularPolygon&) = default;
            RegularPolygon(RegularPolygon&&) noexcept = default;
            RegularPolygon& operator=(const RegularPolygon&) = default;
            RegularPolygon& operator=(RegularPolygon&&) noexcept = default;
            ~RegularPolygon() override = default;

            double getSideLength() const {
                return 2.0 * circumradius_ * sin(Pi / static_cast<double>(N));
            }

            double getCircumradius() const { return circumradius_; }

            double getInradius() const {
                return circumradius_ * cos(Pi / static_cast<double>(N));
            }

            double getInteriorAngle() const {
                return (static_cast<double>(N) - 2.0) * Pi / static_cast<double>(N);
            }

            double getExteriorAngle() const {
                return 2.0 * Pi / static_cast<double>(N);
            }

            Point2D getCenter() const { return center_; }

            double area() const override {
                double s = getSideLength();
                double apothem = getInradius();
                return 0.5 * static_cast<double>(N) * s * apothem;
            }

            double perimeter() const override {
                return static_cast<double>(N) * getSideLength();
            }

        private:
            Point2D center_;
            double circumradius_;

            void computeVertices() {
                std::array<Point2D, N> pts;
                for (size_t i = 0; i < N; ++i) {
                    double angle = 2.0 * Pi * static_cast<double>(i) / static_cast<double>(N)
                                   - Pi / 2.0; // start at top
                    pts[i] = Point2D(
                        center_.x() + circumradius_ * cos(angle),
                        center_.y() + circumradius_ * sin(angle)
                    );
                }
                this->setVertices(pts);
            }
        };

    } // namespace Geometry
} // namespace SharedMath
