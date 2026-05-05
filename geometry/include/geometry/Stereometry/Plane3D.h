#pragma once
#include "../Point.h"
#include "../Vectors.h"
#include "constans.h"
#include <sharedmath_geometry_export.h>
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        /// Plane equation: ax + by + cz + d = 0
        class SHAREDMATH_GEOMETRY_EXPORT Plane3D {
        public:
            /// Default: z = 0 plane (a=0, b=0, c=1, d=0)
            Plane3D();

            Plane3D(double a, double b, double c, double d);

            Plane3D(const Point3D& point, const Vector3D& normal);

            Plane3D(const Point3D& p1, const Point3D& p2, const Point3D& p3);

            Plane3D(const Plane3D&) = default;
            Plane3D(Plane3D&&) noexcept = default;
            Plane3D& operator=(const Plane3D&) = default;
            Plane3D& operator=(Plane3D&&) noexcept = default;
            ~Plane3D() = default;

            double getA() const { return a_; }
            double getB() const { return b_; }
            double getC() const { return c_; }
            double getD() const { return d_; }

            Vector3D getNormal() const;

            double distanceTo(const Point3D& p) const;
            bool contains(const Point3D& p) const;
            bool isParallelTo(const Plane3D& other) const;
            bool isPerpendicularTo(const Plane3D& other) const;

            /// Make unit normal (normalize coefficients)
            Plane3D normalize() const;

            /// Project point onto plane
            Point3D project(const Point3D& p) const;

            /// Reflect point through plane
            Point3D reflect(const Point3D& p) const;

            bool operator==(const Plane3D& other) const;
            bool operator!=(const Plane3D& other) const;

        private:
            double a_, b_, c_, d_;
        };

    } // namespace Geometry
} // namespace SharedMath
