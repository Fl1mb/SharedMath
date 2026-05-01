#pragma once
#include "../Point.h"
#include "../Vectors.h"
#include <array>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        template<size_t Degree>
        class BezierCurve {
        public:
            BezierCurve() {
                controlPoints_.fill(Point2D(0.0, 0.0));
            }

            explicit BezierCurve(const std::array<Point2D, Degree + 1>& points)
                : controlPoints_(points) {}

            const std::array<Point2D, Degree + 1>& getControlPoints() const {
                return controlPoints_;
            }

            void setControlPoints(const std::array<Point2D, Degree + 1>& pts) {
                controlPoints_ = pts;
            }

            /// De Casteljau algorithm
            Point2D evaluate(double t) const {
                std::array<Point2D, Degree + 1> pts = controlPoints_;
                size_t n = Degree + 1;
                for (size_t r = 1; r < n; ++r) {
                    for (size_t i = 0; i < n - r; ++i) {
                        pts[i] = Point2D(
                            (1.0 - t) * pts[i].x() + t * pts[i + 1].x(),
                            (1.0 - t) * pts[i].y() + t * pts[i + 1].y()
                        );
                    }
                }
                return pts[0];
            }

            std::vector<Point2D> sample(size_t numPoints) const {
                if (numPoints < 2)
                    throw std::invalid_argument("BezierCurve::sample: numPoints must be >= 2");
                std::vector<Point2D> result;
                result.reserve(numPoints);
                for (size_t i = 0; i < numPoints; ++i) {
                    double t = static_cast<double>(i) / static_cast<double>(numPoints - 1);
                    result.push_back(evaluate(t));
                }
                return result;
            }

            double approximateLength(size_t steps = 100) const {
                if (steps < 1) steps = 1;
                double length = 0.0;
                Point2D prev = evaluate(0.0);
                for (size_t i = 1; i <= steps; ++i) {
                    double t = static_cast<double>(i) / static_cast<double>(steps);
                    Point2D curr = evaluate(t);
                    double dx = curr.x() - prev.x();
                    double dy = curr.y() - prev.y();
                    length += std::sqrt(dx * dx + dy * dy);
                    prev = curr;
                }
                return length;
            }

            BezierCurve<Degree> translate(const Vector2D& offset) const {
                std::array<Point2D, Degree + 1> pts = controlPoints_;
                for (auto& p : pts) {
                    p = Point2D(p.x() + offset.x(), p.y() + offset.y());
                }
                return BezierCurve<Degree>(pts);
            }

            // Degree elevation (only for Degree < 3 to avoid template recursion issues)
            // Returns a BezierCurve<Degree+1> with same shape
            BezierCurve<Degree + 1> elevate() const {
                // Degree elevation formula: P'_i = (i/(n+1)) * P_{i-1} + (1 - i/(n+1)) * P_i
                std::array<Point2D, Degree + 2> newPts;
                size_t n = Degree;
                newPts[0] = controlPoints_[0];
                for (size_t i = 1; i <= n; ++i) {
                    double alpha = static_cast<double>(i) / static_cast<double>(n + 1);
                    newPts[i] = Point2D(
                        alpha * controlPoints_[i - 1].x() + (1.0 - alpha) * controlPoints_[i].x(),
                        alpha * controlPoints_[i - 1].y() + (1.0 - alpha) * controlPoints_[i].y()
                    );
                }
                newPts[n + 1] = controlPoints_[n];
                return BezierCurve<Degree + 1>(newPts);
            }

        private:
            std::array<Point2D, Degree + 1> controlPoints_;
        };

        using QuadraticBezier = BezierCurve<2>;
        using CubicBezier     = BezierCurve<3>;

    } // namespace Geometry
} // namespace SharedMath
