#include "Intersection.h"
#include <cmath>
#include <algorithm>

using namespace SharedMath::Geometry;
using namespace SharedMath::Geometry::Intersection;
using namespace SharedMath;

// ---- 2D intersections ----

Result2D Intersection::lineLine(const Line2D& a, const Line2D& b)
{
    Result2D result;
    Point2D p1 = a.getFirstPoint2D();
    Point2D p2 = a.getSecondPoint2D();
    Point2D p3 = b.getFirstPoint2D();
    Point2D p4 = b.getSecondPoint2D();

    double d1x = p2.x() - p1.x(), d1y = p2.y() - p1.y();
    double d2x = p4.x() - p3.x(), d2y = p4.y() - p3.y();

    double denom = d1x * d2y - d1y * d2x;
    if (std::abs(denom) < Epsilon) {
        // Parallel (or coincident): no single intersection point
        result.hit = false;
        return result;
    }

    double t = ((p3.x() - p1.x()) * d2y - (p3.y() - p1.y()) * d2x) / denom;
    result.hit = true;
    result.points.push_back(Point2D(p1.x() + t * d1x, p1.y() + t * d1y));
    return result;
}

Result2D Intersection::lineCircle(const Line2D& line, const Circle& circle)
{
    Result2D result;
    Point2D p1 = line.getFirstPoint2D();
    Point2D p2 = line.getSecondPoint2D();
    Point2D c  = circle.getCenter();
    double  r  = circle.getRadius();

    double dx = p2.x() - p1.x();
    double dy = p2.y() - p1.y();
    double fx = p1.x() - c.x();
    double fy = p1.y() - c.y();

    double A = dx * dx + dy * dy;
    double B = 2.0 * (fx * dx + fy * dy);
    double C = fx * fx + fy * fy - r * r;

    double disc = B * B - 4.0 * A * C;
    if (disc < -Epsilon) return result;
    if (std::abs(A) < Epsilon) return result;

    disc = std::max(0.0, disc);
    double sqrtDisc = std::sqrt(disc);
    double t1 = (-B - sqrtDisc) / (2.0 * A);
    double t2 = (-B + sqrtDisc) / (2.0 * A);

    result.hit = true;
    result.points.push_back(Point2D(p1.x() + t1 * dx, p1.y() + t1 * dy));
    if (disc > Epsilon)
        result.points.push_back(Point2D(p1.x() + t2 * dx, p1.y() + t2 * dy));
    return result;
}

Result2D Intersection::lineEllipse(const Line2D& line, const Ellipse& ellipse)
{
    Result2D result;
    Point2D p1 = line.getFirstPoint2D();
    Point2D p2 = line.getSecondPoint2D();
    Point2D ce = ellipse.getCenter();
    double  a  = ellipse.getSemiMajor();
    double  b  = ellipse.getSemiMinor();

    // Translate so ellipse is at origin
    double x1 = p1.x() - ce.x(), y1 = p1.y() - ce.y();
    double dx = p2.x() - p1.x(), dy = p2.y() - p1.y();

    // Parametric: (x1 + t*dx)^2/a^2 + (y1 + t*dy)^2/b^2 = 1
    double A = (dx * dx) / (a * a) + (dy * dy) / (b * b);
    double B = 2.0 * ((x1 * dx) / (a * a) + (y1 * dy) / (b * b));
    double C = (x1 * x1) / (a * a) + (y1 * y1) / (b * b) - 1.0;

    double disc = B * B - 4.0 * A * C;
    if (disc < -Epsilon || std::abs(A) < Epsilon) return result;

    disc = std::max(0.0, disc);
    double sqrtDisc = std::sqrt(disc);
    double t1 = (-B - sqrtDisc) / (2.0 * A);
    double t2 = (-B + sqrtDisc) / (2.0 * A);

    result.hit = true;
    result.points.push_back(Point2D(p1.x() + t1 * dx, p1.y() + t1 * dy));
    if (disc > Epsilon)
        result.points.push_back(Point2D(p1.x() + t2 * dx, p1.y() + t2 * dy));
    return result;
}

Result2D Intersection::circleCircle(const Circle& c1, const Circle& c2)
{
    Result2D result;
    double dx = c2.getCenter().x() - c1.getCenter().x();
    double dy = c2.getCenter().y() - c1.getCenter().y();
    double d  = std::sqrt(dx * dx + dy * dy);
    double r1 = c1.getRadius(), r2 = c2.getRadius();

    if (d > r1 + r2 + Epsilon || d < std::abs(r1 - r2) - Epsilon || d < Epsilon)
        return result;

    double a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    double h2 = r1 * r1 - a * a;
    if (h2 < 0.0) h2 = 0.0;
    double h = std::sqrt(h2);

    double mx = c1.getCenter().x() + a * dx / d;
    double my = c1.getCenter().y() + a * dy / d;

    result.hit = true;
    result.points.push_back(Point2D(mx + h * dy / d, my - h * dx / d));
    if (h > Epsilon)
        result.points.push_back(Point2D(mx - h * dy / d, my + h * dx / d));
    return result;
}

// ---- 3D intersections ----

Result3D Intersection::linePlane(const Line3D& line, const Plane3D& plane)
{
    Result3D result;
    Point3D p0 = line.getFirstPoint3D();
    Point3D p1 = line.getSecondPoint3D();
    double dx = p1.x() - p0.x();
    double dy = p1.y() - p0.y();
    double dz = p1.z() - p0.z();

    double a = plane.getA(), b = plane.getB(), c = plane.getC(), d = plane.getD();
    double denom = a * dx + b * dy + c * dz;

    if (std::abs(denom) < Epsilon) return result; // parallel

    double t = -(a * p0.x() + b * p0.y() + c * p0.z() + d) / denom;
    result.hit = true;
    result.points.push_back(Point3D(p0.x() + t * dx,
                                    p0.y() + t * dy,
                                    p0.z() + t * dz));
    return result;
}

Result3D Intersection::planePlane(const Plane3D& p1, const Plane3D& p2)
{
    Result3D result;
    // Line of intersection: direction = n1 x n2
    double a1 = p1.getA(), b1 = p1.getB(), c1 = p1.getC(), d1 = p1.getD();
    double a2 = p2.getA(), b2 = p2.getB(), c2 = p2.getC(), d2 = p2.getD();

    double dx = b1 * c2 - c1 * b2;
    double dy = c1 * a2 - a1 * c2;
    double dz = a1 * b2 - b1 * a2;

    double len = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (len < Epsilon) return result; // parallel planes

    result.hit = true;

    // Find a point on the line (set z=0 and solve, or pick largest component)
    Point3D pt;
    if (std::abs(dz) >= std::abs(dx) && std::abs(dz) >= std::abs(dy)) {
        // Solve a1*x + b1*y = -d1, a2*x + b2*y = -d2
        double det = a1 * b2 - a2 * b1;
        if (std::abs(det) < Epsilon) { result.hit = false; return result; }
        double px = (-d1 * b2 + d2 * b1) / det;
        double py = (a1 * (-d2) - a2 * (-d1)) / det;
        pt = Point3D(px, py, 0.0);
    } else if (std::abs(dy) >= std::abs(dx)) {
        double det = a1 * c2 - a2 * c1;
        if (std::abs(det) < Epsilon) { result.hit = false; return result; }
        double px = (-d1 * c2 + d2 * c1) / det;
        double pz = (a1 * (-d2) - a2 * (-d1)) / det;
        pt = Point3D(px, 0.0, pz);
    } else {
        double det = b1 * c2 - b2 * c1;
        if (std::abs(det) < Epsilon) { result.hit = false; return result; }
        double py = (-d1 * c2 + d2 * c1) / det;
        double pz = (b1 * (-d2) - b2 * (-d1)) / det;
        pt = Point3D(0.0, py, pz);
    }

    // Return two points defining the line
    result.points.push_back(pt);
    result.points.push_back(Point3D(pt.x() + dx, pt.y() + dy, pt.z() + dz));
    return result;
}

Result3D Intersection::sphereSphere(const Sphere& s1, const Sphere& s2)
{
    Result3D result;
    double dx = s2.getCenter().x() - s1.getCenter().x();
    double dy = s2.getCenter().y() - s1.getCenter().y();
    double dz = s2.getCenter().z() - s1.getCenter().z();
    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    double r1 = s1.getRadius(), r2 = s2.getRadius();

    if (d > r1 + r2 + Epsilon || d < std::abs(r1 - r2) - Epsilon || d < Epsilon)
        return result;

    // Intersection is a circle in a plane
    double a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    // Center of intersection circle
    double cx = s1.getCenter().x() + a * dx / d;
    double cy = s1.getCenter().y() + a * dy / d;
    double cz = s1.getCenter().z() + a * dz / d;

    result.hit = true;
    result.points.push_back(Point3D(cx, cy, cz));
    return result;
}

Result3D Intersection::lineSphere(const Line3D& line, const Sphere& sphere)
{
    Result3D result;
    Point3D p0 = line.getFirstPoint3D();
    Point3D p1 = line.getSecondPoint3D();
    Point3D c  = sphere.getCenter();
    double  r  = sphere.getRadius();

    double dx = p1.x() - p0.x();
    double dy = p1.y() - p0.y();
    double dz = p1.z() - p0.z();
    double fx = p0.x() - c.x();
    double fy = p0.y() - c.y();
    double fz = p0.z() - c.z();

    double A = dx*dx + dy*dy + dz*dz;
    double B = 2.0 * (fx*dx + fy*dy + fz*dz);
    double C = fx*fx + fy*fy + fz*fz - r*r;

    double disc = B*B - 4.0*A*C;
    if (disc < -Epsilon || std::abs(A) < Epsilon) return result;

    disc = std::max(0.0, disc);
    double sqrtDisc = std::sqrt(disc);
    double t1 = (-B - sqrtDisc) / (2.0 * A);
    double t2 = (-B + sqrtDisc) / (2.0 * A);

    result.hit = true;
    result.points.push_back(Point3D(p0.x() + t1*dx, p0.y() + t1*dy, p0.z() + t1*dz));
    if (disc > Epsilon)
        result.points.push_back(Point3D(p0.x() + t2*dx, p0.y() + t2*dy, p0.z() + t2*dz));
    return result;
}
