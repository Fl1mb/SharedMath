#pragma once
#include "Point.h"
#include "Line.h"
#include "Planimetry/Circle.h"
#include "Planimetry/Ellipse.h"
#include "Stereometry/Plane3D.h"
#include "Stereometry/Sphere.h"
#include "constans.h"
#include <sharedmath_geometry_export.h>
#include <vector>

namespace SharedMath
{
    namespace Geometry
    {
        namespace Intersection
        {
            struct SHAREDMATH_GEOMETRY_EXPORT Result2D {
                bool hit = false;
                std::vector<Point2D> points;
            };

            struct SHAREDMATH_GEOMETRY_EXPORT Result3D {
                bool hit = false;
                std::vector<Point3D> points;
            };

            SHAREDMATH_GEOMETRY_EXPORT Result2D lineLine(const Line2D& a, const Line2D& b);
            SHAREDMATH_GEOMETRY_EXPORT Result2D lineCircle(const Line2D& line, const Circle& circle);
            SHAREDMATH_GEOMETRY_EXPORT Result2D lineEllipse(const Line2D& line, const Ellipse& ellipse);
            SHAREDMATH_GEOMETRY_EXPORT Result2D circleCircle(const Circle& c1, const Circle& c2);

            SHAREDMATH_GEOMETRY_EXPORT Result3D linePlane(const Line3D& line, const Plane3D& plane);
            SHAREDMATH_GEOMETRY_EXPORT Result3D planePlane(const Plane3D& p1, const Plane3D& p2);
            SHAREDMATH_GEOMETRY_EXPORT Result3D sphereSphere(const Sphere& s1, const Sphere& s2);
            SHAREDMATH_GEOMETRY_EXPORT Result3D lineSphere(const Line3D& line, const Sphere& sphere);

        } // namespace Intersection
    } // namespace Geometry
} // namespace SharedMath
