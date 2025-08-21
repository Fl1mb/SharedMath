#include <gtest/gtest.h>
#include "../include/SharedMath.h"

using namespace SharedMath::Geometry;

TEST(RectangleTest, RectangleCreationFromPoints) {
    std::array<Point2D, 4> points = {
        Point2D{0.0, 0.0},
        Point2D{5.0, 0.0},
        Point2D{5.0, 3.0},
        Point2D{0.0, 3.0}
    };
    
    Rectangle rect(points);
    
    EXPECT_EQ(rect.getBottomLeft(), Point2D(0.0, 0.0));
    EXPECT_EQ(rect.getBottomRight(), Point2D(5.0, 0.0));
    EXPECT_EQ(rect.getTopRight(), Point2D(5.0, 3.0));
    EXPECT_EQ(rect.getTopLeft(), Point2D(0.0, 3.0));
}

TEST(RectangleTest, RectangleCreationFromCorners) {
    Point2D bottomLeft(1.0, 2.0);
    Point2D topRight(6.0, 5.0);
    
    Rectangle rect(bottomLeft, topRight);
    
    EXPECT_EQ(rect.getBottomLeft(), bottomLeft);
    EXPECT_EQ(rect.getTopRight(), topRight);
    EXPECT_EQ(rect.getWidth(), 5.0);
    EXPECT_EQ(rect.getHeight(), 3.0);
}

TEST(RectangleTest, RectangleCreationFromPositionAndSize) {
    Point2D position(2.0, 3.0);
    double width = 4.0;
    double height = 2.0;
    
    Rectangle rect(position, width, height);
    
    EXPECT_EQ(rect.getBottomLeft(), position);
    EXPECT_EQ(rect.getWidth(), width);
    EXPECT_EQ(rect.getHeight(), height);
    EXPECT_EQ(rect.getTopRight(), Point2D(6.0, 5.0));
}

TEST(RectangleTest, InvalidRectangleThrowsException) {
    std::array<Point2D, 4> invalidPoints = {
        Point2D{0.0, 0.0},
        Point2D{1.0, 0.0},
        Point2D{2.0, 1.0}, // Не прямоугольник
        Point2D{0.0, 1.0}
    };
    
    EXPECT_THROW(Rectangle rect(invalidPoints), std::invalid_argument);
}

TEST(RectangleTest, IsRectangleValidation) {
    std::array<Point2D, 4> validPoints = {
        Point2D{0.0, 0.0},
        Point2D{4.0, 0.0},
        Point2D{4.0, 2.0},
        Point2D{0.0, 2.0}
    };
    
    std::array<Point2D, 4> invalidPoints = {
        Point2D{0.0, 0.0},
        Point2D{3.0, 0.0},
        Point2D{4.0, 2.0}, // Не прямоугольник
        Point2D{0.0, 2.0}
    };
    
    EXPECT_TRUE(Rectangle::isRectangle(validPoints));
    EXPECT_FALSE(Rectangle::isRectangle(invalidPoints));
}

TEST(RectangleTest, AreaCalculation) {
    Rectangle rect(Point2D(1.0, 2.0), Point2D(5.0, 6.0));
    EXPECT_DOUBLE_EQ(rect.area(), 16.0);
}

TEST(RectangleTest, PerimeterCalculation) {
    Rectangle rect(Point2D(0.0, 0.0), Point2D(3.0, 4.0));
    EXPECT_DOUBLE_EQ(rect.perimeter(), 14.0);
}

TEST(RectangleTest, CenterPoint) {
    Rectangle rect(Point2D(1.0, 1.0), Point2D(5.0, 5.0));
    Point2D center = rect.getCenter();
    
    EXPECT_DOUBLE_EQ(center.x(), 3.0);
    EXPECT_DOUBLE_EQ(center.y(), 3.0);
}

TEST(RectangleTest, IsSquareDetection) {
    Rectangle square(Point2D(0.0, 0.0), Point2D(4.0, 4.0));
    Rectangle rect(Point2D(0.0, 0.0), Point2D(5.0, 3.0));
    
    EXPECT_TRUE(square.isSquare());
    EXPECT_FALSE(rect.isSquare());
}

TEST(RectangleTest, PointContainment) {
    Rectangle rect(Point2D(1.0, 1.0), Point2D(4.0, 4.0));
    
    EXPECT_TRUE(rect.contains(Point2D(2.0, 2.0)));
    EXPECT_TRUE(rect.contains(Point2D(1.0, 1.0))); // Граница
    EXPECT_FALSE(rect.contains(Point2D(0.0, 0.0)));
    EXPECT_FALSE(rect.contains(Point2D(5.0, 5.0)));
}

TEST(RectangleTest, RectangleIntersection) {
    Rectangle rect1(Point2D(0.0, 0.0), Point2D(3.0, 3.0));
    Rectangle rect2(Point2D(2.0, 2.0), Point2D(5.0, 5.0));
    Rectangle rect3(Point2D(4.0, 4.0), Point2D(6.0, 6.0));
    
    EXPECT_TRUE(rect1.intersects(rect2));
    EXPECT_TRUE(rect2.intersects(rect1));
    EXPECT_FALSE(rect1.intersects(rect3));
    EXPECT_FALSE(rect3.intersects(rect1));
}

TEST(RectangleTest, MoveOperation) {
    Rectangle rect(Point2D(1.0, 2.0), Point2D(4.0, 5.0));
    Vector2D offset(2.0, 3.0);
    
    rect.move(offset);
    
    EXPECT_EQ(rect.getBottomLeft(), Point2D(3.0, 5.0));
    EXPECT_EQ(rect.getTopRight(), Point2D(6.0, 8.0));
}

TEST(RectangleTest, ScaleOperation) {
    Rectangle rect(Point2D(0.0, 0.0), Point2D(2.0, 2.0));
    
    rect.scale(2.0);
    
    EXPECT_DOUBLE_EQ(rect.getWidth(), 4.0);
    EXPECT_DOUBLE_EQ(rect.getHeight(), 4.0);
    EXPECT_EQ(rect.getCenter(), Point2D(1.0, 1.0));
}

TEST(RectangleTest, SetSizeOperation) {
    Rectangle rect(Point2D(1.0, 1.0), Point2D(3.0, 3.0));
    
    rect.setSize(5.0, 2.0);
    
    EXPECT_DOUBLE_EQ(rect.getWidth(), 5.0);
    EXPECT_DOUBLE_EQ(rect.getHeight(), 2.0);
    EXPECT_EQ(rect.getBottomLeft(), Point2D(1.0, 1.0));
}

TEST(RectangleTest, SetPositionOperation) {
    Rectangle rect(Point2D(1.0, 1.0), Point2D(4.0, 4.0));
    
    rect.setPosition(Point2D(0.0, 0.0));
    
    EXPECT_EQ(rect.getBottomLeft(), Point2D(0.0, 0.0));
    EXPECT_DOUBLE_EQ(rect.getWidth(), 3.0);
    EXPECT_DOUBLE_EQ(rect.getHeight(), 3.0);
}

TEST(RectangleTest, EqualityOperators) {
    Rectangle rect1(Point2D(0.0, 0.0), Point2D(3.0, 3.0));
    Rectangle rect2(Point2D(0.0, 0.0), Point2D(3.0, 3.0));
    Rectangle rect3(Point2D(1.0, 1.0), Point2D(4.0, 4.0));
    
    EXPECT_TRUE(rect1 == rect2);
    EXPECT_FALSE(rect1 == rect3);
    EXPECT_TRUE(rect1 != rect3);
    EXPECT_FALSE(rect1 != rect2);
}

TEST(RectangleTest, VectorAdditionOperators) {
    Rectangle rect(Point2D(1.0, 2.0), Point2D(4.0, 5.0));
    Vector2D offset(1.0, -1.0);
    
    Rectangle result1 = rect + offset;
    Rectangle result2 = rect - offset;
    
    EXPECT_EQ(result1.getBottomLeft(), Point2D(2.0, 1.0));
    EXPECT_EQ(result2.getBottomLeft(), Point2D(0.0, 3.0));
}

TEST(RectangleTest, CompoundAssignmentOperators) {
    Rectangle rect(Point2D(1.0, 2.0), Point2D(4.0, 5.0));
    Vector2D offset(1.0, -1.0);
    
    rect += offset;
    EXPECT_EQ(rect.getBottomLeft(), Point2D(2.0, 1.0));
    
    rect -= offset;
    EXPECT_EQ(rect.getBottomLeft(), Point2D(1.0, 2.0));
}

TEST(RectangleTest, AspectRatioCalculation) {
    Rectangle rect1(Point2D(0.0, 0.0), Point2D(4.0, 2.0)); // 2:1
    Rectangle rect2(Point2D(0.0, 0.0), Point2D(3.0, 3.0)); // 1:1
    
    EXPECT_DOUBLE_EQ(rect1.getAspectRatio(), 2.0);
    EXPECT_DOUBLE_EQ(rect2.getAspectRatio(), 1.0);
}

TEST(RectangleTest, InvalidSizeThrowsException) {
    EXPECT_THROW(Rectangle(Point2D(0.0, 0.0), -1.0, 2.0), std::invalid_argument);
    EXPECT_THROW(Rectangle(Point2D(0.0, 0.0), 2.0, -1.0), std::invalid_argument);
    EXPECT_THROW(Rectangle(Point2D(0.0, 0.0), 0.0, 2.0), std::invalid_argument);
}