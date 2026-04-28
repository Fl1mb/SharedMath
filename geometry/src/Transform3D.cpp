#include "Transform3D.h"
#include <cmath>
#include <stdexcept>

using namespace SharedMath::Geometry;
using namespace SharedMath::LinearAlgebra;
using namespace SharedMath;

Transform3D::Transform3D()
    : matrix_(4, 4, 0.0)
{
    matrix_(0, 0) = 1.0;
    matrix_(1, 1) = 1.0;
    matrix_(2, 2) = 1.0;
    matrix_(3, 3) = 1.0;
}

Transform3D::Transform3D(const DynamicMatrix& matrix)
    : matrix_(matrix)
{
    if (matrix.rows() != 4 || matrix.cols() != 4)
        throw std::invalid_argument("Transform3D: matrix must be 4x4");
}

Transform3D Transform3D::identity()
{
    return Transform3D();
}

Transform3D Transform3D::translation(double dx, double dy, double dz)
{
    Transform3D t;
    t.matrix_(0, 3) = dx;
    t.matrix_(1, 3) = dy;
    t.matrix_(2, 3) = dz;
    return t;
}

Transform3D Transform3D::rotationX(double angle)
{
    Transform3D t;
    double c = std::cos(angle);
    double s = std::sin(angle);
    t.matrix_(1, 1) =  c;
    t.matrix_(1, 2) = -s;
    t.matrix_(2, 1) =  s;
    t.matrix_(2, 2) =  c;
    return t;
}

Transform3D Transform3D::rotationY(double angle)
{
    Transform3D t;
    double c = std::cos(angle);
    double s = std::sin(angle);
    t.matrix_(0, 0) =  c;
    t.matrix_(0, 2) =  s;
    t.matrix_(2, 0) = -s;
    t.matrix_(2, 2) =  c;
    return t;
}

Transform3D Transform3D::rotationZ(double angle)
{
    Transform3D t;
    double c = std::cos(angle);
    double s = std::sin(angle);
    t.matrix_(0, 0) =  c;
    t.matrix_(0, 1) = -s;
    t.matrix_(1, 0) =  s;
    t.matrix_(1, 1) =  c;
    return t;
}

Transform3D Transform3D::scale(double sx, double sy, double sz)
{
    Transform3D t;
    t.matrix_(0, 0) = sx;
    t.matrix_(1, 1) = sy;
    t.matrix_(2, 2) = sz;
    return t;
}

Transform3D Transform3D::lookAt(const Point3D& eye, const Point3D& target, const Vector3D& up)
{
    // Compute forward, right, up vectors
    Vector3D forward(eye, target);
    double flen = forward.length();
    if (flen < Epsilon)
        throw std::invalid_argument("Transform3D::lookAt: eye and target cannot be the same point");
    forward = forward.normalized();

    Vector3D right = forward.cross(up);
    double rlen = right.length();
    if (rlen < Epsilon)
        throw std::invalid_argument("Transform3D::lookAt: up vector is parallel to forward");
    right = right.normalized();

    Vector3D newUp = right.cross(forward);

    DynamicMatrix m(4, 4, 0.0);
    m(0, 0) = right.x();   m(0, 1) = right.y();   m(0, 2) = right.z();
    m(1, 0) = newUp.x();   m(1, 1) = newUp.y();   m(1, 2) = newUp.z();
    m(2, 0) = -forward.x(); m(2, 1) = -forward.y(); m(2, 2) = -forward.z();
    m(3, 3) = 1.0;

    m(0, 3) = -(right.x()    * eye.x() + right.y()    * eye.y() + right.z()    * eye.z());
    m(1, 3) = -(newUp.x()    * eye.x() + newUp.y()    * eye.y() + newUp.z()    * eye.z());
    m(2, 3) =  (forward.x()  * eye.x() + forward.y()  * eye.y() + forward.z()  * eye.z());

    return Transform3D(m);
}

Transform3D Transform3D::operator*(const Transform3D& other) const
{
    return Transform3D(matrix_ * other.matrix_);
}

Point3D Transform3D::operator*(const Point3D& p) const
{
    double x = matrix_(0,0)*p.x() + matrix_(0,1)*p.y() + matrix_(0,2)*p.z() + matrix_(0,3);
    double y = matrix_(1,0)*p.x() + matrix_(1,1)*p.y() + matrix_(1,2)*p.z() + matrix_(1,3);
    double z = matrix_(2,0)*p.x() + matrix_(2,1)*p.y() + matrix_(2,2)*p.z() + matrix_(2,3);
    double w = matrix_(3,0)*p.x() + matrix_(3,1)*p.y() + matrix_(3,2)*p.z() + matrix_(3,3);
    if (std::abs(w) > Epsilon) { x /= w; y /= w; z /= w; }
    return Point3D(x, y, z);
}

Vector3D Transform3D::operator*(const Vector3D& v) const
{
    double x = matrix_(0,0)*v.x() + matrix_(0,1)*v.y() + matrix_(0,2)*v.z();
    double y = matrix_(1,0)*v.x() + matrix_(1,1)*v.y() + matrix_(1,2)*v.z();
    double z = matrix_(2,0)*v.x() + matrix_(2,1)*v.y() + matrix_(2,2)*v.z();
    return Vector3D(x, y, z);
}

// Helper: compute determinant of 3x3 submatrix excluding row r and col c
static double minor3(const DynamicMatrix& m, size_t r, size_t c)
{
    double sub[3][3];
    size_t si = 0;
    for (size_t i = 0; i < 4; ++i) {
        if (i == r) continue;
        size_t sj = 0;
        for (size_t j = 0; j < 4; ++j) {
            if (j == c) continue;
            sub[si][sj] = m(i, j);
            ++sj;
        }
        ++si;
    }
    return sub[0][0] * (sub[1][1]*sub[2][2] - sub[1][2]*sub[2][1])
         - sub[0][1] * (sub[1][0]*sub[2][2] - sub[1][2]*sub[2][0])
         + sub[0][2] * (sub[1][0]*sub[2][1] - sub[1][1]*sub[2][0]);
}

Transform3D Transform3D::inverse() const
{
    // Compute inverse via cofactor matrix / determinant
    DynamicMatrix adj(4, 4, 0.0);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            double sign = ((i + j) % 2 == 0) ? 1.0 : -1.0;
            adj(j, i) = sign * minor3(matrix_, i, j); // transpose for adjugate
        }
    }

    double det = 0.0;
    for (size_t j = 0; j < 4; ++j)
        det += matrix_(0, j) * adj(j, 0);

    if (std::abs(det) < Epsilon)
        throw std::runtime_error("Transform3D::inverse: matrix is singular");

    DynamicMatrix inv(4, 4, 0.0);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            inv(i, j) = adj(i, j) / det;

    return Transform3D(inv);
}
