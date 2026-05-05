#pragma once
#include "../Point.h"
#include "../Vectors.h"
#include "constans.h"
#include <sharedmath_geometry_export.h>
#include <array>
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class SHAREDMATH_GEOMETRY_EXPORT Tetrahedron {
        public:
            Tetrahedron() = default;

            explicit Tetrahedron(const std::array<Point3D, 4>& vertices);

            Tetrahedron(const Tetrahedron&) = default;
            Tetrahedron(Tetrahedron&&) noexcept = default;
            Tetrahedron& operator=(const Tetrahedron&) = default;
            Tetrahedron& operator=(Tetrahedron&&) noexcept = default;
            ~Tetrahedron() = default;

            const Point3D& getVertex(size_t i) const;

            /// Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
            double volume() const;

            double surfaceArea() const;

            Vector3D getFaceNormal(size_t faceIndex) const;

            Point3D getCentroid() const;

            bool isRegular() const;

            bool contains(const Point3D& p) const;

            bool operator==(const Tetrahedron& other) const;
            bool operator!=(const Tetrahedron& other) const;

        private:
            std::array<Point3D, 4> vertices_;

            static double triangleArea(const Point3D& a, const Point3D& b, const Point3D& c);
            static double edgeLength(const Point3D& a, const Point3D& b);
        };

    } // namespace Geometry
} // namespace SharedMath
