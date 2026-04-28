#include "Stereometry/Plane3D.h"
#include <cmath>
#include <stdexcept>

using namespace SharedMath::Geometry;
using namespace SharedMath;

Plane3D::Plane3D() : a_(0.0), b_(0.0), c_(1.0), d_(0.0) {}

Plane3D::Plane3D(double a, double b, double c, double d)
    : a_(a), b_(b), c_(c), d_(d)
{
    double len = sqrt(a * a + b * b + c * c);
    if (len < Epsilon)
        throw std::invalid_argument("Plane3D: normal vector cannot be zero");
}

Plane3D::Plane3D(const Point3D& point, const Vector3D& normal)
{
    double len = normal.length();
    if (len < Epsilon)
        throw std::invalid_argument("Plane3D: normal vector cannot be zero");
    a_ = normal.x();
    b_ = normal.y();
    c_ = normal.z();
    d_ = -(a_ * point.x() + b_ * point.y() + c_ * point.z());
}

Plane3D::Plane3D(const Point3D& p1, const Point3D& p2, const Point3D& p3)
{
    Vector3D v1(p1, p2);
    Vector3D v2(p1, p3);
    Vector3D normal = v1.cross(v2);
    double len = normal.length();
    if (len < Epsilon)
        throw std::invalid_argument("Plane3D: three points are collinear");
    a_ = normal.x();
    b_ = normal.y();
    c_ = normal.z();
    d_ = -(a_ * p1.x() + b_ * p1.y() + c_ * p1.z());
}

Vector3D Plane3D::getNormal() const
{
    return Vector3D(a_, b_, c_);
}

double Plane3D::distanceTo(const Point3D& p) const
{
    double len = sqrt(a_ * a_ + b_ * b_ + c_ * c_);
    if (len < Epsilon) return 0.0;
    return abs(a_ * p.x() + b_ * p.y() + c_ * p.z() + d_) / len;
}

bool Plane3D::contains(const Point3D& p) const
{
    return distanceTo(p) < Epsilon;
}

bool Plane3D::isParallelTo(const Plane3D& other) const
{
    Vector3D n1 = getNormal();
    Vector3D n2 = other.getNormal();
    return n1.isParallel(n2);
}

bool Plane3D::isPerpendicularTo(const Plane3D& other) const
{
    Vector3D n1 = getNormal();
    Vector3D n2 = other.getNormal();
    return n1.isPerpendicular(n2);
}

Plane3D Plane3D::normalize() const
{
    double len = sqrt(a_ * a_ + b_ * b_ + c_ * c_);
    if (len < Epsilon) return *this;
    return Plane3D(a_ / len, b_ / len, c_ / len, d_ / len);
}

Point3D Plane3D::project(const Point3D& p) const
{
    double len2 = a_ * a_ + b_ * b_ + c_ * c_;
    if (len2 < Epsilon) return p;
    double t = (a_ * p.x() + b_ * p.y() + c_ * p.z() + d_) / len2;
    return Point3D(p.x() - a_ * t,
                   p.y() - b_ * t,
                   p.z() - c_ * t);
}

Point3D Plane3D::reflect(const Point3D& p) const
{
    Point3D proj = project(p);
    return Point3D(2.0 * proj.x() - p.x(),
                   2.0 * proj.y() - p.y(),
                   2.0 * proj.z() - p.z());
}

bool Plane3D::operator==(const Plane3D& other) const
{
    // Two planes are equal if their normalized equations are the same
    Plane3D n1 = normalize();
    Plane3D n2 = other.normalize();
    // Account for sign flip
    double scale = 1.0;
    if (abs(n1.a_) > Epsilon) scale = n2.a_ / n1.a_;
    else if (abs(n1.b_) > Epsilon) scale = n2.b_ / n1.b_;
    else scale = n2.c_ / n1.c_;

    return abs(n1.a_ * scale - n2.a_) < Epsilon &&
           abs(n1.b_ * scale - n2.b_) < Epsilon &&
           abs(n1.c_ * scale - n2.c_) < Epsilon &&
           abs(n1.d_ * scale - n2.d_) < Epsilon;
}

bool Plane3D::operator!=(const Plane3D& other) const
{
    return !(*this == other);
}
