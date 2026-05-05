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
        class SHAREDMATH_GEOMETRY_EXPORT Transform2D {
        public:
            Transform2D();

            explicit Transform2D(const LinearAlgebra::DynamicMatrix& matrix);

            Transform2D(const Transform2D&) = default;
            Transform2D(Transform2D&&) noexcept = default;
            Transform2D& operator=(const Transform2D&) = default;
            Transform2D& operator=(Transform2D&&) noexcept = default;
            ~Transform2D() = default;

            /// Static factories
            static Transform2D identity();
            static Transform2D translation(double dx, double dy);
            static Transform2D rotation(double angle);
            static Transform2D scale(double sx, double sy);
            static Transform2D shear(double shx, double shy);

            // Composition
            Transform2D operator*(const Transform2D& other) const;

            // Apply to point (uses homogeneous coordinates)
            Point2D operator*(const Point2D& p) const;

            // Apply to vector (ignores translation)
            Vector2D operator*(const Vector2D& v) const;

            Transform2D inverse() const;

            const LinearAlgebra::DynamicMatrix& getMatrix() const { return matrix_; }

        private:
            LinearAlgebra::DynamicMatrix matrix_;
        };

    } // namespace Geometry
} // namespace SharedMath
