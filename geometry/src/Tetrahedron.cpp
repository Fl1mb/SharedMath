#include "Stereometry/Tetrahedron.h"
#include <cmath>
#include <stdexcept>

using namespace SharedMath::Geometry;
using namespace SharedMath;

double Tetrahedron::edgeLength(const Point3D& a, const Point3D& b)
{
    double dx = b.x() - a.x();
    double dy = b.y() - a.y();
    double dz = b.z() - a.z();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double Tetrahedron::triangleArea(const Point3D& a, const Point3D& b, const Point3D& c)
{
    Vector3D ab(a, b);
    Vector3D ac(a, c);
    Vector3D cross = ab.cross(ac);
    return cross.length() / 2.0;
}

Tetrahedron::Tetrahedron(const std::array<Point3D, 4>& vertices)
    : vertices_(vertices)
{
    // Validate non-coplanar: check determinant != 0
    Vector3D v1(vertices[0], vertices[1]);
    Vector3D v2(vertices[0], vertices[2]);
    Vector3D v3(vertices[0], vertices[3]);
    double det = v1.tripleProduct(v2, v3);
    if (std::abs(det) < Epsilon)
        throw std::invalid_argument("Tetrahedron: four vertices are coplanar");
}

const Point3D& Tetrahedron::getVertex(size_t i) const
{
    if (i >= 4) throw std::out_of_range("Tetrahedron::getVertex: index out of range");
    return vertices_[i];
}

double Tetrahedron::volume() const
{
    Vector3D v1(vertices_[0], vertices_[1]);
    Vector3D v2(vertices_[0], vertices_[2]);
    Vector3D v3(vertices_[0], vertices_[3]);
    return std::abs(v1.tripleProduct(v2, v3)) / 6.0;
}

double Tetrahedron::surfaceArea() const
{
    // 4 triangular faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    return triangleArea(vertices_[0], vertices_[1], vertices_[2]) +
           triangleArea(vertices_[0], vertices_[1], vertices_[3]) +
           triangleArea(vertices_[0], vertices_[2], vertices_[3]) +
           triangleArea(vertices_[1], vertices_[2], vertices_[3]);
}

Vector3D Tetrahedron::getFaceNormal(size_t faceIndex) const
{
    // Face indices: 0:(1,2,3), 1:(0,2,3), 2:(0,1,3), 3:(0,1,2)
    static const size_t faces[4][3] = {
        {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}
    };
    if (faceIndex >= 4) throw std::out_of_range("Tetrahedron::getFaceNormal: index out of range");
    const Point3D& a = vertices_[faces[faceIndex][0]];
    const Point3D& b = vertices_[faces[faceIndex][1]];
    const Point3D& c = vertices_[faces[faceIndex][2]];
    Vector3D ab(a, b);
    Vector3D ac(a, c);
    Vector3D n = ab.cross(ac);
    double len = n.length();
    if (len < Epsilon) return n;
    return n.normalized();
}

Point3D Tetrahedron::getCentroid() const
{
    return Point3D(
        (vertices_[0].x() + vertices_[1].x() + vertices_[2].x() + vertices_[3].x()) / 4.0,
        (vertices_[0].y() + vertices_[1].y() + vertices_[2].y() + vertices_[3].y()) / 4.0,
        (vertices_[0].z() + vertices_[1].z() + vertices_[2].z() + vertices_[3].z()) / 4.0
    );
}

bool Tetrahedron::isRegular() const
{
    // All 6 edges equal
    double edges[6] = {
        edgeLength(vertices_[0], vertices_[1]),
        edgeLength(vertices_[0], vertices_[2]),
        edgeLength(vertices_[0], vertices_[3]),
        edgeLength(vertices_[1], vertices_[2]),
        edgeLength(vertices_[1], vertices_[3]),
        edgeLength(vertices_[2], vertices_[3])
    };
    for (int i = 1; i < 6; ++i)
        if (std::abs(edges[i] - edges[0]) > Epsilon) return false;
    return true;
}

bool Tetrahedron::contains(const Point3D& p) const
{
    // Barycentric coordinates method
    // p = v0 + u*(v1-v0) + v*(v2-v0) + w*(v3-v0)
    // p is inside if u,v,w >= 0 and u+v+w <= 1
    Vector3D v1(vertices_[0], vertices_[1]);
    Vector3D v2(vertices_[0], vertices_[2]);
    Vector3D v3(vertices_[0], vertices_[3]);
    Vector3D vp(vertices_[0], p);

    double d11 = v1.dot(v1), d12 = v1.dot(v2), d13 = v1.dot(v3);
    double d22 = v2.dot(v2), d23 = v2.dot(v3), d33 = v3.dot(v3);
    double dp1 = vp.dot(v1), dp2 = vp.dot(v2), dp3 = vp.dot(v3);

    // Solve 3x3 system using Cramer's rule
    // [d11 d12 d13] [u]   [dp1]
    // [d12 d22 d23] [v] = [dp2]
    // [d13 d23 d33] [w]   [dp3]
    double det = d11 * (d22 * d33 - d23 * d23)
               - d12 * (d12 * d33 - d23 * d13)
               + d13 * (d12 * d23 - d22 * d13);

    if (std::abs(det) < Epsilon) return false;

    double u = (dp1 * (d22 * d33 - d23 * d23)
              - d12 * (dp2 * d33 - d23 * dp3)
              + d13 * (dp2 * d23 - d22 * dp3)) / det;

    double v = (d11 * (dp2 * d33 - d23 * dp3)
              - dp1 * (d12 * d33 - d23 * d13)
              + d13 * (d12 * dp3 - dp2 * d13)) / det;

    double w = (d11 * (d22 * dp3 - dp2 * d23)
              - d12 * (d12 * dp3 - dp2 * d13)
              + dp1 * (d12 * d23 - d22 * d13)) / det;

    return u >= -Epsilon && v >= -Epsilon && w >= -Epsilon && (u + v + w) <= 1.0 + Epsilon;
}

bool Tetrahedron::operator==(const Tetrahedron& other) const
{
    return vertices_ == other.vertices_;
}

bool Tetrahedron::operator!=(const Tetrahedron& other) const
{
    return !(*this == other);
}
