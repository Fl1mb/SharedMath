#include "Planimetry/Trapezoid.h"
#include <algorithm>
#include <cmath>

using namespace SharedMath::Geometry;
using namespace SharedMath;

bool Trapezoid::areParallel(const Point2D& a, const Point2D& b,
                             const Point2D& c, const Point2D& d)
{
    // Direction vectors
    double dx1 = b.x() - a.x();
    double dy1 = b.y() - a.y();
    double dx2 = d.x() - c.x();
    double dy2 = d.y() - c.y();
    // Cross product == 0 means parallel
    return std::abs(dx1 * dy2 - dy1 * dx2) < Epsilon;
}

bool Trapezoid::isTrapezoid(const std::array<Point2D, 4>& pts)
{
    // Check uniqueness
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = i + 1; j < 4; ++j)
            if (pts[i] == pts[j]) return false;

    // Check that at least one pair of opposite sides is parallel
    // Pairs: (0-1 || 2-3) or (1-2 || 3-0)
    bool pair1 = areParallel(pts[0], pts[1], pts[3], pts[2]);
    bool pair2 = areParallel(pts[1], pts[2], pts[0], pts[3]);
    return pair1 || pair2;
}

void Trapezoid::findBases(const std::array<Point2D, 4>& pts)
{
    // Try pair 0-1 and 2-3 (sides 0->1 and 3->2, i.e., opposite sides of polygon)
    if (areParallel(pts[0], pts[1], pts[3], pts[2])) {
        base1Start_ = 0; base1End_ = 1;
        base2Start_ = 3; base2End_ = 2;
    } else {
        // pair 1-2 and 0-3
        base1Start_ = 1; base1End_ = 2;
        base2Start_ = 0; base2End_ = 3;
    }
}

Trapezoid::Trapezoid(const std::array<Point2D, 4>& points)
{
    if (!isTrapezoid(points))
        throw std::invalid_argument("Points do not form a trapezoid");
    setVertices(points);
    findBases(points);
}

Line2D Trapezoid::getBase1() const
{
    const auto& v = getVertices();
    return Line2D(v[base1Start_], v[base1End_]);
}

Line2D Trapezoid::getBase2() const
{
    const auto& v = getVertices();
    return Line2D(v[base2Start_], v[base2End_]);
}

double Trapezoid::getHeight() const
{
    // Height = distance from base2 line to base1 line
    Line2D b1 = getBase1();
    const auto& v = getVertices();
    Point2D p = v[base2Start_];

    // Line from b1: ax + by + c = 0
    double dx = b1.getSecondPoint2D().x() - b1.getFirstPoint2D().x();
    double dy = b1.getSecondPoint2D().y() - b1.getFirstPoint2D().y();
    // Normal (dy, -dx), line: dy*(x - x1) - dx*(y - y1) = 0
    double a = dy;
    double b = -dx;
    double c = -(a * b1.getFirstPoint2D().x() + b * b1.getFirstPoint2D().y());
    double len = std::sqrt(a * a + b * b);
    if (len < Epsilon) return 0.0;
    return std::abs(a * p.x() + b * p.y() + c) / len;
}

double Trapezoid::area() const
{
    double b1 = getBase1().getLength();
    double b2 = getBase2().getLength();
    double h  = getHeight();
    return 0.5 * (b1 + b2) * h;
}

double Trapezoid::perimeter() const
{
    const auto& v = getVertices();
    double sum = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        size_t j = (i + 1) % 4;
        double dx = v[j].x() - v[i].x();
        double dy = v[j].y() - v[i].y();
        sum += std::sqrt(dx * dx + dy * dy);
    }
    return sum;
}

bool Trapezoid::isIsosceles() const
{
    const auto& v = getVertices();
    // The non-parallel sides (legs) are the ones not being bases
    // base1: base1Start_ -> base1End_
    // base2: base2Start_ -> base2End_
    // legs connect: base1End_ -> base2Start_ and base2End_ -> base1Start_
    double leg1 = std::hypot(v[base2Start_].x() - v[base1End_].x(),
                             v[base2Start_].y() - v[base1End_].y());
    double leg2 = std::hypot(v[base1Start_].x() - v[base2End_].x(),
                             v[base1Start_].y() - v[base2End_].y());
    return std::abs(leg1 - leg2) < Epsilon;
}

Line2D Trapezoid::getMidline() const
{
    const auto& v = getVertices();
    // Midpoints of the two legs
    Point2D mid1(
        (v[base1End_].x() + v[base2Start_].x()) / 2.0,
        (v[base1End_].y() + v[base2Start_].y()) / 2.0
    );
    Point2D mid2(
        (v[base2End_].x() + v[base1Start_].x()) / 2.0,
        (v[base2End_].y() + v[base1Start_].y()) / 2.0
    );
    return Line2D(mid1, mid2);
}
