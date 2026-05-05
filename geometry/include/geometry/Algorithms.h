#pragma once
#include "Point.h"
#include "Stereometry/Plane3D.h"
#include "constans.h"
#include <sharedmath_geometry_export.h>
#include <vector>
#include <array>
#include <utility>

namespace SharedMath
{
    namespace Geometry
    {
        namespace Algorithms
        {
            /// Convex hull (Graham scan) — returns vertices in CCW order
            SHAREDMATH_GEOMETRY_EXPORT
            std::vector<Point2D> convexHull(std::vector<Point2D> points);

            /// Point-in-polygon (ray casting)
            SHAREDMATH_GEOMETRY_EXPORT
            bool pointInPolygon(const Point2D& p, const std::vector<Point2D>& polygon);

            /// Distance from point to segment [a,b]
            SHAREDMATH_GEOMETRY_EXPORT
            double distancePointSegment(const Point2D& p, const Point2D& a, const Point2D& b);

            /// Distance between two segments
            SHAREDMATH_GEOMETRY_EXPORT
            double distanceSegmentSegment(const Point2D& a1, const Point2D& b1,
                                          const Point2D& a2, const Point2D& b2);

            /// Distance from 3D point to plane
            SHAREDMATH_GEOMETRY_EXPORT
            double distancePointPlane(const Point3D& p, const Plane3D& plane);

            /// Triangulate a simple polygon (ear-clipping) — returns triangles as index triples
            SHAREDMATH_GEOMETRY_EXPORT
            std::vector<std::array<size_t, 3>> triangulate(const std::vector<Point2D>& polygon);

            /// Bounding boxes
            SHAREDMATH_GEOMETRY_EXPORT
            std::pair<Point2D, Point2D> boundingBox2D(const std::vector<Point2D>& points);

            SHAREDMATH_GEOMETRY_EXPORT
            std::pair<Point3D, Point3D> boundingBox3D(const std::vector<Point3D>& points);

        } // namespace Algorithms
    } // namespace Geometry
} // namespace SharedMath
