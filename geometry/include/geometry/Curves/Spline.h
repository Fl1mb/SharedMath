#pragma once
#include "../Point.h"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        // Centripetal Catmull-Rom spline
        class CatmullRomSpline {
        public:
            CatmullRomSpline() : alpha_(0.5) {}

            CatmullRomSpline(std::vector<Point2D> points, double alpha = 0.5)
                : controlPoints_(std::move(points)), alpha_(alpha)
            {
                if (controlPoints_.size() < 2)
                    throw std::invalid_argument("CatmullRomSpline: need at least 2 control points");
            }

            void addPoint(const Point2D& p) {
                controlPoints_.push_back(p);
            }

            const std::vector<Point2D>& getControlPoints() const {
                return controlPoints_;
            }

            // Evaluate at parameter t in [0, n-3] where n = controlPoints_.size()
            // Each unit interval [i, i+1] spans one segment between p[i+1] and p[i+2]
            Point2D evaluate(double t) const
            {
                size_t n = controlPoints_.size();
                if (n < 4)
                    throw std::runtime_error("CatmullRomSpline::evaluate: need at least 4 control points");

                // Clamp t to valid range [0, n-3]
                double maxT = static_cast<double>(n - 3);
                t = std::max(0.0, std::min(t, maxT));

                size_t seg = static_cast<size_t>(t);
                if (seg >= n - 3) seg = n - 4;
                double localT = t - static_cast<double>(seg);

                const Point2D& p0 = controlPoints_[seg];
                const Point2D& p1 = controlPoints_[seg + 1];
                const Point2D& p2 = controlPoints_[seg + 2];
                const Point2D& p3 = controlPoints_[seg + 3];

                return catmullRom(p0, p1, p2, p3, localT);
            }

            // Sample n points uniformly over the full spline
            std::vector<Point2D> sample(size_t n) const
            {
                size_t np = controlPoints_.size();
                if (np < 4)
                    throw std::runtime_error("CatmullRomSpline::sample: need at least 4 control points");
                if (n < 2) throw std::invalid_argument("CatmullRomSpline::sample: n must be >= 2");

                double maxT = static_cast<double>(np - 3);
                std::vector<Point2D> result;
                result.reserve(n);
                for (size_t i = 0; i < n; ++i) {
                    double t = maxT * static_cast<double>(i) / static_cast<double>(n - 1);
                    result.push_back(evaluate(t));
                }
                return result;
            }

        private:
            std::vector<Point2D> controlPoints_;
            double alpha_;

            // Knot parameterization
            double tj(double ti, const Point2D& pi, const Point2D& pj) const {
                double dx = pj.x() - pi.x();
                double dy = pj.y() - pi.y();
                double dist = std::pow(dx * dx + dy * dy, 0.5 * alpha_);
                return ti + dist;
            }

            Point2D catmullRom(const Point2D& p0, const Point2D& p1,
                               const Point2D& p2, const Point2D& p3,
                               double t) const
            {
                double t0 = 0.0;
                double t1 = tj(t0, p0, p1);
                double t2 = tj(t1, p1, p2);
                double t3 = tj(t2, p2, p3);

                double tParam = t1 + t * (t2 - t1);

                auto interp = [](const Point2D& a, const Point2D& b, double ta, double tb, double tv) -> Point2D {
                    if (std::abs(tb - ta) < 1e-12) return a;
                    double f = (tv - ta) / (tb - ta);
                    return Point2D(a.x() + f * (b.x() - a.x()),
                                   a.y() + f * (b.y() - a.y()));
                };

                Point2D A1 = interp(p0, p1, t0, t1, tParam);
                Point2D A2 = interp(p1, p2, t1, t2, tParam);
                Point2D A3 = interp(p2, p3, t2, t3, tParam);

                Point2D B1 = interp(A1, A2, t0, t2, tParam);
                Point2D B2 = interp(A2, A3, t1, t3, tParam);

                return interp(B1, B2, t1, t2, tParam);
            }
        };

    } // namespace Geometry
} // namespace SharedMath
