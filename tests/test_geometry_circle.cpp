#include <gtest/gtest.h>
#include "../include/SharedMath.h"

using namespace SharedMath::Geometry;

TEST(CircleTest, CircleCreation) {
    Circle circle(Point2D(1.0, 2.0), 3.0);
    
    EXPECT_EQ(circle.getCenter(), Point2D(1.0, 2.0));
    EXPECT_DOUBLE_EQ(circle.getRadius(), 3.0);
    EXPECT_DOUBLE_EQ(circle.getDiameter(), 6.0);
}

TEST(CircleTest, InvalidRadiusThrows) {
    EXPECT_THROW(Circle(Point2D(0, 0), 0.0), std::invalid_argument);
    EXPECT_THROW(Circle(Point2D(0, 0), -1.0), std::invalid_argument);
}

TEST(CircleTest, PointContainment) {
    Circle circle(Point2D(0, 0), 2.0);
    
    EXPECT_TRUE(circle.contains(Point2D(0, 0)));
    EXPECT_TRUE(circle.contains(Point2D(1, 1)));
    EXPECT_TRUE(circle.contains(Point2D(2, 0))); // На границе
    EXPECT_FALSE(circle.contains(Point2D(3, 0)));
}

TEST(CircleTest, CircleIntersection) {
    Circle circle1(Point2D(0, 0), 2.0);
    Circle circle2(Point2D(3, 0), 2.0);
    Circle circle3(Point2D(5, 0), 1.0);
    
    EXPECT_TRUE(circle1.intersects(circle2)); // Пересекаются
    EXPECT_FALSE(circle1.intersects(circle3)); // Не пересекаются
}

TEST(CircleTest, MoveOperation) {
    Circle circle(Point2D(1.0, 2.0), 3.0);
    circle.move(Vector2D(2.0, -1.0));
    
    EXPECT_EQ(circle.getCenter(), Point2D(3.0, 1.0));
    EXPECT_DOUBLE_EQ(circle.getRadius(), 3.0);
}

TEST(CircleTest, ScaleOperation) {
    Circle circle(Point2D(0, 0), 2.0);
    circle.scale(1.5);
    
    EXPECT_DOUBLE_EQ(circle.getRadius(), 3.0);
    EXPECT_NEAR(circle.area(), 9.0 * M_PI, 1e-10);
}