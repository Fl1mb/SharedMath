#include "Transform2D.h"
#include <cmath>
#include <stdexcept>

using namespace SharedMath::Geometry;
using namespace SharedMath::LinearAlgebra;
using namespace SharedMath;

Transform2D::Transform2D()
    : matrix_(3, 3, 0.0)
{
    // Initialize as identity
    matrix_(0, 0) = 1.0;
    matrix_(1, 1) = 1.0;
    matrix_(2, 2) = 1.0;
}

Transform2D::Transform2D(const DynamicMatrix& matrix)
    : matrix_(matrix)
{
    if (matrix.rows() != 3 || matrix.cols() != 3)
        throw std::invalid_argument("Transform2D: matrix must be 3x3");
}

Transform2D Transform2D::identity()
{
    return Transform2D();
}

Transform2D Transform2D::translation(double dx, double dy)
{
    Transform2D t;
    t.matrix_(0, 2) = dx;
    t.matrix_(1, 2) = dy;
    return t;
}

Transform2D Transform2D::rotation(double angle)
{
    Transform2D t;
    double c = std::cos(angle);
    double s = std::sin(angle);
    t.matrix_(0, 0) =  c;
    t.matrix_(0, 1) = -s;
    t.matrix_(1, 0) =  s;
    t.matrix_(1, 1) =  c;
    return t;
}

Transform2D Transform2D::scale(double sx, double sy)
{
    Transform2D t;
    t.matrix_(0, 0) = sx;
    t.matrix_(1, 1) = sy;
    return t;
}

Transform2D Transform2D::shear(double shx, double shy)
{
    Transform2D t;
    t.matrix_(0, 1) = shx;
    t.matrix_(1, 0) = shy;
    return t;
}

Transform2D Transform2D::operator*(const Transform2D& other) const
{
    DynamicMatrix result = matrix_ * other.matrix_;
    return Transform2D(result);
}

Point2D Transform2D::operator*(const Point2D& p) const
{
    double x = matrix_(0, 0) * p.x() + matrix_(0, 1) * p.y() + matrix_(0, 2);
    double y = matrix_(1, 0) * p.x() + matrix_(1, 1) * p.y() + matrix_(1, 2);
    double w = matrix_(2, 0) * p.x() + matrix_(2, 1) * p.y() + matrix_(2, 2);
    if (abs(w) > Epsilon) { x /= w; y /= w; }
    return Point2D(x, y);
}

Vector2D Transform2D::operator*(const Vector2D& v) const
{
    // Vectors transform without translation (w=0)
    double x = matrix_(0, 0) * v.x() + matrix_(0, 1) * v.y();
    double y = matrix_(1, 0) * v.x() + matrix_(1, 1) * v.y();
    return Vector2D(x, y);
}

Transform2D Transform2D::inverse() const
{
    // 3x3 inverse via cofactors
    const auto& m = matrix_;
    double a = m(0,0), b = m(0,1), c = m(0,2);
    double d = m(1,0), e = m(1,1), f = m(1,2);
    double g = m(2,0), h = m(2,1), i = m(2,2);

    double det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
    if (abs(det) < Epsilon)
        throw std::runtime_error("Transform2D::inverse: matrix is singular");

    DynamicMatrix inv(3, 3, 0.0);
    inv(0,0) =  (e*i - f*h) / det;
    inv(0,1) = -(b*i - c*h) / det;
    inv(0,2) =  (b*f - c*e) / det;
    inv(1,0) = -(d*i - f*g) / det;
    inv(1,1) =  (a*i - c*g) / det;
    inv(1,2) = -(a*f - c*d) / det;
    inv(2,0) =  (d*h - e*g) / det;
    inv(2,1) = -(a*h - b*g) / det;
    inv(2,2) =  (a*e - b*d) / det;

    return Transform2D(inv);
}
