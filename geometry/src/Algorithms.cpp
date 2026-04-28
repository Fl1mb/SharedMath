#include "Algorithms.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace SharedMath::Geometry;
using namespace SharedMath::Geometry::Algorithms;
using namespace SharedMath;

// ---- Convex Hull (Graham scan) ----

static double cross2D(const Point2D& O, const Point2D& A, const Point2D& B)
{
    return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
}

std::vector<Point2D> Algorithms::convexHull(std::vector<Point2D> points)
{
    size_t n = points.size();
    if (n < 3) return points;

    // Sort by x then y
    std::sort(points.begin(), points.end(), [](const Point2D& a, const Point2D& b) {
        return a.x() < b.x() || (abs(a.x() - b.x()) < Epsilon && a.y() < b.y());
    });

    std::vector<Point2D> hull;
    hull.reserve(2 * n);

    // Lower hull
    for (size_t i = 0; i < n; ++i) {
        while (hull.size() >= 2 && cross2D(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0)
            hull.pop_back();
        hull.push_back(points[i]);
    }

    // Upper hull
    size_t lower_size = hull.size();
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        while (hull.size() > lower_size && cross2D(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0)
            hull.pop_back();
        hull.push_back(points[i]);
    }

    hull.pop_back(); // remove duplicate last point
    return hull;
}

// ---- Point in polygon ----

bool Algorithms::pointInPolygon(const Point2D& p, const std::vector<Point2D>& polygon)
{
    size_t n = polygon.size();
    if (n < 3) return false;
    bool inside = false;
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        const Point2D& vi = polygon[i];
        const Point2D& vj = polygon[j];
        if (((vi.y() > p.y()) != (vj.y() > p.y())) &&
            (p.x() < (vj.x() - vi.x()) * (p.y() - vi.y()) / (vj.y() - vi.y()) + vi.x()))
        {
            inside = !inside;
        }
    }
    return inside;
}

// ---- Distance point to segment ----

double Algorithms::distancePointSegment(const Point2D& p, const Point2D& a, const Point2D& b)
{
    double dx = b.x() - a.x();
    double dy = b.y() - a.y();
    double lenSq = dx * dx + dy * dy;
    if (lenSq < Epsilon) {
        double ex = p.x() - a.x(), ey = p.y() - a.y();
        return sqrt(ex * ex + ey * ey);
    }
    double t = ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / lenSq;
    t = std::max(0.0, std::min(1.0, t));
    double px = a.x() + t * dx - p.x();
    double py = a.y() + t * dy - p.y();
    return sqrt(px * px + py * py);
}

// ---- Distance segment to segment ----

double Algorithms::distanceSegmentSegment(const Point2D& a1, const Point2D& b1,
                                           const Point2D& a2, const Point2D& b2)
{
    // Check if segments intersect first
    auto cross = [](double ax, double ay, double bx, double by) { return ax * by - ay * bx; };
    double d1x = b1.x() - a1.x(), d1y = b1.y() - a1.y();
    double d2x = b2.x() - a2.x(), d2y = b2.y() - a2.y();
    double denom = cross(d1x, d1y, d2x, d2y);

    if (abs(denom) > Epsilon) {
        double dx = a2.x() - a1.x(), dy = a2.y() - a1.y();
        double t1 = cross(dx, dy, d2x, d2y) / denom;
        double t2 = cross(dx, dy, d1x, d1y) / denom;
        if (t1 >= 0.0 && t1 <= 1.0 && t2 >= 0.0 && t2 <= 1.0)
            return 0.0;
    }

    // Minimum of 4 endpoint distances
    double d1 = distancePointSegment(a1, a2, b2);
    double d2 = distancePointSegment(b1, a2, b2);
    double d3 = distancePointSegment(a2, a1, b1);
    double d4 = distancePointSegment(b2, a1, b1);
    return std::min({d1, d2, d3, d4});
}

// ---- Distance point to plane ----

double Algorithms::distancePointPlane(const Point3D& p, const Plane3D& plane)
{
    return plane.distanceTo(p);
}

// ---- Triangulate (ear-clipping) ----

static bool isEar(const std::vector<Point2D>& poly,
                  const std::vector<size_t>& indices, size_t i)
{
    size_t n = indices.size();
    size_t prev = (i + n - 1) % n;
    size_t next = (i + 1) % n;

    const Point2D& A = poly[indices[prev]];
    const Point2D& B = poly[indices[i]];
    const Point2D& C = poly[indices[next]];

    // Check that triangle ABC is counter-clockwise (convex vertex)
    double cross = (B.x() - A.x()) * (C.y() - A.y()) - (B.y() - A.y()) * (C.x() - A.x());
    if (cross <= 0) return false;

    // Check no other vertex is inside triangle ABC
    for (size_t j = 0; j < n; ++j) {
        if (j == prev || j == i || j == next) continue;
        const Point2D& P = poly[indices[j]];
        // Barycentric test
        double d1 = (P.x() - A.x()) * (B.y() - A.y()) - (P.y() - A.y()) * (B.x() - A.x());
        double d2 = (P.x() - B.x()) * (C.y() - B.y()) - (P.y() - B.y()) * (C.x() - B.x());
        double d3 = (P.x() - C.x()) * (A.y() - C.y()) - (P.y() - C.y()) * (A.x() - C.x());
        bool allPos = d1 > 0 && d2 > 0 && d3 > 0;
        bool allNeg = d1 < 0 && d2 < 0 && d3 < 0;
        if (allPos || allNeg) return false;
    }
    return true;
}

std::vector<std::array<size_t, 3>> Algorithms::triangulate(const std::vector<Point2D>& polygon)
{
    size_t n = polygon.size();
    if (n < 3) throw std::invalid_argument("triangulate: need at least 3 vertices");

    std::vector<std::array<size_t, 3>> triangles;
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    // Ensure CCW winding
    double area = 0.0;
    for (size_t i = 0; i < n; ++i) {
        size_t j = (i + 1) % n;
        area += polygon[i].x() * polygon[j].y() - polygon[j].x() * polygon[i].y();
    }
    if (area < 0) std::reverse(indices.begin(), indices.end());

    size_t remaining = n;
    size_t attempts = 0;
    size_t i = 0;
    while (remaining > 3) {
        if (attempts > remaining) break; // no ear found — degenerate polygon
        if (isEar(polygon, indices, i % remaining)) {
            size_t prev = (i % remaining + remaining - 1) % remaining;
            size_t curr = i % remaining;
            size_t next = (i % remaining + 1) % remaining;
            triangles.push_back({ indices[prev], indices[curr], indices[next] });
            indices.erase(indices.begin() + static_cast<std::ptrdiff_t>(curr));
            --remaining;
            attempts = 0;
        } else {
            ++i;
            ++attempts;
        }
    }
    if (remaining == 3)
        triangles.push_back({ indices[0], indices[1], indices[2] });

    return triangles;
}

// ---- Bounding boxes ----

std::pair<Point2D, Point2D> Algorithms::boundingBox2D(const std::vector<Point2D>& points)
{
    if (points.empty()) throw std::invalid_argument("boundingBox2D: no points");
    double minX = points[0].x(), maxX = points[0].x();
    double minY = points[0].y(), maxY = points[0].y();
    for (const auto& p : points) {
        if (p.x() < minX) minX = p.x();
        if (p.x() > maxX) maxX = p.x();
        if (p.y() < minY) minY = p.y();
        if (p.y() > maxY) maxY = p.y();
    }
    return { Point2D(minX, minY), Point2D(maxX, maxY) };
}

std::pair<Point3D, Point3D> Algorithms::boundingBox3D(const std::vector<Point3D>& points)
{
    if (points.empty()) throw std::invalid_argument("boundingBox3D: no points");
    double minX = points[0].x(), maxX = points[0].x();
    double minY = points[0].y(), maxY = points[0].y();
    double minZ = points[0].z(), maxZ = points[0].z();
    for (const auto& p : points) {
        if (p.x() < minX) minX = p.x();
        if (p.x() > maxX) maxX = p.x();
        if (p.y() < minY) minY = p.y();
        if (p.y() > maxY) maxY = p.y();
        if (p.z() < minZ) minZ = p.z();
        if (p.z() > maxZ) maxZ = p.z();
    }
    return { Point3D(minX, minY, minZ), Point3D(maxX, maxY, maxZ) };
}
