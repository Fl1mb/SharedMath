#include <gtest/gtest.h>
#include "../include/SharedMath.h"

namespace SharedMath::Geometry {
    TEST(PolygonTest, TriangleCreation) {
        std::array<Point2D, 3> points = {
            Point2D{0.0, 0.0},
            Point2D{1.0, 0.0},
            Point2D{0.0, 1.0}
        };
        Polygon<3> triangle(points);
        
        EXPECT_EQ(triangle.vertex(0), Point2D({0.0, 0.0}));
        EXPECT_EQ(triangle.vertex(1), Point2D({1.0, 0.0}));
        EXPECT_EQ(triangle.vertex(2), Point2D({0.0, 1.0}));
    }

    TEST(PolygonTest, TriangleArea) {
        Polygon<3> triangle({
            Point2D{0.0, 0.0},
            Point2D{1.0, 0.0},
            Point2D{0.0, 1.0}
        });
        EXPECT_DOUBLE_EQ(triangle.area(), 0.5);
    }

    TEST(PolygonTest, SquareArea) {
        Polygon<4> square({
            Point2D{0.0, 0.0},
            Point2D{1.0, 0.0},
            Point2D{1.0, 1.0},
            Point2D{0.0, 1.0}
        });
        EXPECT_DOUBLE_EQ(square.area(), 1.0);
    }

    TEST(PolygonTest, PerimeterCalculation) {
        Polygon<4> rectangle({
            Point2D{0.0, 0.0},
            Point2D{2.0, 0.0},
            Point2D{2.0, 1.0},
            Point2D{0.0, 1.0}
        });
        EXPECT_DOUBLE_EQ(rectangle.perimeter(), 6.0);
    }

    TEST(PolygonTest, VertexAccessValidation) {
        Polygon<3> triangle({
            Point2D{0.0, 0.0},
            Point2D{1.0, 0.0},
            Point2D{0.0, 1.0}
        });
        
        EXPECT_NO_THROW(triangle.vertex(0));
        EXPECT_THROW(triangle.vertex(3), std::invalid_argument);
    }

    TEST(PolygonTest, DegeneratePolygon) {
        Polygon<3> degenerate({
            Point2D{1.0, 1.0},
            Point2D{1.0, 1.0},
            Point2D{1.0, 1.0}
        });
        
        EXPECT_DOUBLE_EQ(degenerate.area(), 0.0);
        EXPECT_DOUBLE_EQ(degenerate.perimeter(), 0.0);
    }


    TEST(PolygonTest, CopyConstructor) {
        Polygon<4> original({
            Point2D{0.0, 0.0},
            Point2D{1.0, 0.0},
            Point2D{1.0, 1.0},
            Point2D{0.0, 1.0}
        });
        
        Polygon<4> copy(original);
        EXPECT_EQ(original, copy);
    }

    TEST(PolygonTest, MoveConstructor) {
        Polygon<4> original({
            Point2D{0.0, 0.0},
            Point2D{1.0, 0.0},
            Point2D{1.0, 1.0},
            Point2D{0.0, 1.0}
        });
        
        Polygon<4> moved(std::move(original));
        EXPECT_EQ(moved.vertex(0), Point2D({0.0, 0.0}));
    }

    TEST(PolygonTest, EqualityOperator) {
        Polygon<3> poly1({
            Point2D({0.0, 0.0}),
            Point2D({1.0, 0.0}),
            Point2D({0.0, 1.0})
        });
        
        Polygon<3> poly2({
            Point2D({0.0, 0.0}),
            Point2D({1.0, 0.0}),
            Point2D({0.0, 1.0})
        });
        
        Polygon<3> poly3({
            Point2D({0.0, 0.0}),
            Point2D({1.0, 0.0}),
            Point2D({1.0, 1.0})
        });
        
        EXPECT_TRUE(poly1 == poly2);
        EXPECT_FALSE(poly1 == poly3);
    }

    TEST(PolygonTest, LargePolygon) {
        std::array<Point2D, 100> points;
        for (size_t i = 0; i < 100; ++i) {
            double angle = 2 * M_PI * i / 100;
            points[i] = Point2D{std::cos(angle), std::sin(angle)};
        }
        
        Polygon<100> circle(points);
        EXPECT_NEAR(circle.area(), M_PI, 0.01);
    }
}
