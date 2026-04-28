#pragma once
#include "Circle.h"
#include "../Vectors.h"
#include "constans.h"
#include <cmath>

namespace SharedMath
{
    namespace Geometry
    {
        class Arc {
        public:
            Arc() : circle_(), startAngle_(0.0), endAngle_(Pi) {}

            Arc(const Circle& circle, double startAngle, double endAngle)
                : circle_(circle), startAngle_(startAngle), endAngle_(endAngle) {}

            Arc(const Arc&) = default;
            Arc(Arc&&) noexcept = default;
            Arc& operator=(const Arc&) = default;
            Arc& operator=(Arc&&) noexcept = default;
            ~Arc() = default;

            const Circle& getCircle() const { return circle_; }
            double getStartAngle() const { return startAngle_; }
            double getEndAngle() const { return endAngle_; }

            double subtendedAngle() const {
                double angle = endAngle_ - startAngle_;
                // Normalize to [0, 2*Pi]
                while (angle < 0.0) angle += 2.0 * Pi;
                while (angle > 2.0 * Pi) angle -= 2.0 * Pi;
                return angle;
            }

            double arcLength() const {
                return circle_.getRadius() * subtendedAngle();
            }

            double chordLength() const {
                double half = subtendedAngle() / 2.0;
                return 2.0 * circle_.getRadius() * sin(half);
            }

            double sectorArea() const {
                double r = circle_.getRadius();
                return 0.5 * r * r * subtendedAngle();
            }

            double segmentArea() const {
                double r = circle_.getRadius();
                double theta = subtendedAngle();
                return 0.5 * r * r * (theta - sin(theta));
            }

            Point2D getStartPoint() const {
                double r = circle_.getRadius();
                Point2D c = circle_.getCenter();
                return Point2D(c.x() + r * cos(startAngle_),
                               c.y() + r * sin(startAngle_));
            }

            Point2D getEndPoint() const {
                double r = circle_.getRadius();
                Point2D c = circle_.getCenter();
                return Point2D(c.x() + r * cos(endAngle_),
                               c.y() + r * sin(endAngle_));
            }

            bool contains(double angle) const {
                // Normalize angle to [0, 2*Pi)
                double a = angle;
                double s = startAngle_;
                double e = endAngle_;
                // Normalize all to [0, 2Pi)
                auto norm = [](double x) -> double {
                    x = std::fmod(x, 2.0 * Pi);
                    if (x < 0.0) x += 2.0 * Pi;
                    return x;
                };
                a = norm(a);
                s = norm(s);
                e = norm(e);
                if (s <= e) return a >= s && a <= e;
                return a >= s || a <= e; // wraps around 0
            }

            double midAngle() const {
                return startAngle_ + subtendedAngle() / 2.0;
            }

            void move(const Vector2D& offset) {
                circle_.move(offset);
            }

            bool operator==(const Arc& other) const {
                return circle_ == other.circle_ &&
                       abs(startAngle_ - other.startAngle_) < Epsilon &&
                       abs(endAngle_ - other.endAngle_) < Epsilon;
            }

            bool operator!=(const Arc& other) const {
                return !(*this == other);
            }

        private:
            Circle circle_;
            double startAngle_;
            double endAngle_;
        };

    } // namespace Geometry
} // namespace SharedMath
