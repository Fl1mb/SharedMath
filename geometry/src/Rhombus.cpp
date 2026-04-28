#include "Planimetry/Rhombus.h"
#include <cmath>

using namespace SharedMath::Geometry;
using namespace SharedMath;

bool Rhombus::isRhombus(const std::array<Point2D, 4>& points)
{
    if (!Parallelogram::isParallelogram(points))
        return false;

    auto dist = [](const Point2D& a, const Point2D& b) {
        double dx = b.x() - a.x();
        double dy = b.y() - a.y();
        return std::sqrt(dx * dx + dy * dy);
    };

    double d01 = dist(points[0], points[1]);
    double d12 = dist(points[1], points[2]);
    double d23 = dist(points[2], points[3]);
    double d30 = dist(points[3], points[0]);

    return std::abs(d01 - d12) < Epsilon &&
           std::abs(d12 - d23) < Epsilon &&
           std::abs(d23 - d30) < Epsilon;
}

Rhombus::Rhombus(const std::array<Point2D, 4>& points)
    : Parallelogram(points)
{
    if (!isRhombus(points))
        throw std::invalid_argument("Points do not form a rhombus");
}

Line2D Rhombus::getDiagonal1() const
{
    const auto& v = getVertices();
    return Line2D(v[0], v[2]);
}

Line2D Rhombus::getDiagonal2() const
{
    const auto& v = getVertices();
    return Line2D(v[1], v[3]);
}

double Rhombus::getAngle() const
{
    const auto& v = getVertices();
    Vector2D side1(v[0], v[1]);
    Vector2D side2(v[0], v[3]);

    double cosA = side1.dot(side2) / (side1.length() * side2.length());
    cosA = std::max(-1.0, std::min(1.0, cosA));
    double angle = std::acos(cosA);

    // Return the acute angle
    if (angle > Pi / 2.0)
        angle = Pi - angle;
    return angle;
}

double Rhombus::area() const
{
    double d1 = getDiagonal1().getLength();
    double d2 = getDiagonal2().getLength();
    return d1 * d2 / 2.0;
}

double Rhombus::inradius() const
{
    const auto& v = getVertices();
    double side = Vector2D(v[0], v[1]).length();
    if (side < Epsilon) return 0.0;
    return area() / (2.0 * side);
}
