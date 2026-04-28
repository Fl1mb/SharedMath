#include "Planimetry/DynamicPolygon.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace SharedMath::Geometry;
using namespace SharedMath;

DynamicPolygon::DynamicPolygon(const std::vector<Point2D>& points)
    : vertices_(points)
{
    if (points.size() < 3)
        throw std::invalid_argument("DynamicPolygon requires at least 3 points");
}

double DynamicPolygon::area() const
{
    double sum = 0.0;
    size_t n = vertices_.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        sum += vertices_[i].x() * vertices_[j].y();
        sum -= vertices_[j].x() * vertices_[i].y();
    }
    return std::abs(sum) / 2.0;
}

double DynamicPolygon::perimeter() const
{
    double sum = 0.0;
    size_t n = vertices_.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        double dx = vertices_[j].x() - vertices_[i].x();
        double dy = vertices_[j].y() - vertices_[i].y();
        sum += std::sqrt(dx * dx + dy * dy);
    }
    return sum;
}

size_t DynamicPolygon::vertexCount() const
{
    return vertices_.size();
}

const Point2D& DynamicPolygon::vertex(size_t i) const
{
    if (i >= vertices_.size())
        throw std::out_of_range("DynamicPolygon::vertex index out of range");
    return vertices_[i];
}

void DynamicPolygon::addVertex(const Point2D& p)
{
    vertices_.push_back(p);
}

void DynamicPolygon::removeVertex(size_t i)
{
    if (i >= vertices_.size())
        throw std::out_of_range("DynamicPolygon::removeVertex index out of range");
    if (vertices_.size() <= 3)
        throw std::invalid_argument("Cannot remove vertex: polygon must have at least 3 vertices");
    vertices_.erase(vertices_.begin() + static_cast<std::ptrdiff_t>(i));
}

bool DynamicPolygon::isConvex() const
{
    size_t n = vertices_.size();
    if (n < 3) return false;
    int sign = 0;
    for (size_t i = 0; i < n; ++i) {
        const Point2D& a = vertices_[i];
        const Point2D& b = vertices_[(i + 1) % n];
        const Point2D& c = vertices_[(i + 2) % n];
        double cross = (b.x() - a.x()) * (c.y() - a.y()) -
                       (b.y() - a.y()) * (c.x() - a.x());
        if (std::abs(cross) > Epsilon) {
            int s = (cross > 0) ? 1 : -1;
            if (sign == 0) sign = s;
            else if (sign != s) return false;
        }
    }
    return true;
}

bool DynamicPolygon::contains(const Point2D& p) const
{
    // Ray casting algorithm
    size_t n = vertices_.size();
    bool inside = false;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const Point2D& vi = vertices_[i];
        const Point2D& vj = vertices_[j];
        if (((vi.y() > p.y()) != (vj.y() > p.y())) &&
            (p.x() < (vj.x() - vi.x()) * (p.y() - vi.y()) / (vj.y() - vi.y()) + vi.x()))
        {
            inside = !inside;
        }
    }
    return inside;
}

void DynamicPolygon::move(const Vector2D& offset)
{
    for (auto& v : vertices_) {
        v = Point2D(v.x() + offset.x(), v.y() + offset.y());
    }
}

Rectangle DynamicPolygon::getBoundingBox() const
{
    if (vertices_.empty())
        throw std::runtime_error("DynamicPolygon has no vertices");

    double minX = vertices_[0].x(), maxX = vertices_[0].x();
    double minY = vertices_[0].y(), maxY = vertices_[0].y();

    for (const auto& v : vertices_) {
        if (v.x() < minX) minX = v.x();
        if (v.x() > maxX) maxX = v.x();
        if (v.y() < minY) minY = v.y();
        if (v.y() > maxY) maxY = v.y();
    }

    return Rectangle(Point2D(minX, minY), Point2D(maxX, maxY));
}

bool DynamicPolygon::operator==(const DynamicPolygon& other) const
{
    return vertices_ == other.vertices_;
}

bool DynamicPolygon::operator!=(const DynamicPolygon& other) const
{
    return !(*this == other);
}
