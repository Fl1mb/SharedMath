#pragma once
#include "Point.h"
#include "Vectors.h"
#include "constans.h"
#include "LinearAlgebra/DynamicMatrix.h"
#include <sharedmath_geometry_export.h>
#include <cmath>
#include <stdexcept>

namespace SharedMath
{
    namespace Geometry
    {
        class SHAREDMATH_GEOMETRY_EXPORT Transform3D {
        public:
            Transform3D();

            explicit Transform3D(const LinearAlgebra::DynamicMatrix& matrix);

            Transform3D(const Transform3D&) = default;
            Transform3D(Transform3D&&) noexcept = default;
            Transform3D& operator=(const Transform3D&) = default;
            Transform3D& operator=(Transform3D&&) noexcept = default;
            ~Transform3D() = default;

            /// Static factories
            static Transform3D identity();
            static Transform3D translation(double dx, double dy, double dz);
            static Transform3D rotationX(double angle);
            static Transform3D rotationY(double angle);
            static Transform3D rotationZ(double angle);
            static Transform3D scale(double sx, double sy, double sz);

            /// View matrix
            static Transform3D lookAt(const Point3D& eye, const Point3D& target, const Vector3D& up);

            // Composition
            Transform3D operator*(const Transform3D& other) const;

            // Apply to point
            Point3D operator*(const Point3D& p) const;

            // Apply to vector (ignores translation)
            Vector3D operator*(const Vector3D& v) const;

            Transform3D inverse() const;

            const LinearAlgebra::DynamicMatrix& getMatrix() const { return matrix_; }

        private:
            LinearAlgebra::DynamicMatrix matrix_;
        };

    } // namespace Geometry
} // namespace SharedMath
