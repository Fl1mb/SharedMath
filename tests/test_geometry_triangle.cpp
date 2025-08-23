#include <gtest/gtest.h>
#include "../include/SharedMath.h"

using namespace SharedMath::Geometry;

TEST(TriangleTest, TriangleCreationFromPoints) {
    std::array<Point2D, 3> points = {
        Point2D{0.0, 0.0},
        Point2D{4.0, 0.0},
        Point2D{0.0, 3.0}
    };
    
    Triangle triangle(points);
    
    EXPECT_EQ(triangle.vertex(0), Point2D(0.0, 0.0));
    EXPECT_EQ(triangle.vertex(1), Point2D(4.0, 0.0));
    EXPECT_EQ(triangle.vertex(2), Point2D(0.0, 3.0));
}

TEST(TriangleTest, TriangleCreationFromVertices) {
    Point2D a(0.0, 0.0);
    Point2D b(5.0, 0.0);
    Point2D c(0.0, 4.0);
    
    Triangle triangle(a, b, c);
    
    EXPECT_EQ(triangle.vertex(0), a);
    EXPECT_EQ(triangle.vertex(1), b);
    EXPECT_EQ(triangle.vertex(2), c);
}

TEST(TriangleTest, InvalidTriangleThrowsException) {
    std::array<Point2D, 3> collinearPoints = {
        Point2D{0.0, 0.0},
        Point2D{1.0, 1.0},
        Point2D{2.0, 2.0}  // Коллинеарные точки
    };
    
    std::array<Point2D, 3> duplicatePoints = {
        Point2D{0.0, 0.0},
        Point2D{0.0, 0.0},  // Дубликат
        Point2D{3.0, 4.0}
    };
    
    EXPECT_THROW(Triangle triangle(collinearPoints), std::invalid_argument);
    EXPECT_THROW(Triangle triangle(duplicatePoints), std::invalid_argument);
}

TEST(TriangleTest, IsValidTriangleValidation) {
    std::array<Point2D, 3> validPoints = {
        Point2D{0.0, 0.0},
        Point2D{3.0, 0.0},
        Point2D{0.0, 4.0}
    };
    
    std::array<Point2D, 3> invalidPoints = {
        Point2D{0.0, 0.0},
        Point2D{1.0, 1.0},
        Point2D{2.0, 2.0}  // Коллинеарные
    };
    
    EXPECT_TRUE(Triangle::isValidTriangle(validPoints));
    EXPECT_FALSE(Triangle::isValidTriangle(invalidPoints));
}

TEST(TriangleTest, AreaCalculation) {
    Triangle triangle(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(0.0, 4.0));
    EXPECT_DOUBLE_EQ(triangle.area(), 6.0);
}

TEST(TriangleTest, PerimeterCalculation) {
    Triangle triangle(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(0.0, 4.0));
    EXPECT_DOUBLE_EQ(triangle.perimeter(), 12.0);
}

TEST(TriangleTest, SideLengthCalculation) {
    Triangle triangle(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(0.0, 4.0));
    
    EXPECT_DOUBLE_EQ(triangle.getSideLength(0), 3.0);  // AB
    EXPECT_DOUBLE_EQ(triangle.getSideLength(1), 5.0);  // BC
    EXPECT_DOUBLE_EQ(triangle.getSideLength(2), 4.0);  // CA
}

TEST(TriangleTest, AngleCalculation) {
    Triangle rightTriangle(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(0.0, 4.0));
    
    double angleA = rightTriangle.getAngle(0);
    double angleB = rightTriangle.getAngle(1);
    double angleC = rightTriangle.getAngle(2);
    
    EXPECT_NEAR(angleA, M_PI/2, 1e-10);  // Прямой угол
    EXPECT_NEAR(angleB, std::atan(4.0/3.0), 1e-10);
    EXPECT_NEAR(angleC, std::atan(3.0/4.0), 1e-10);
}

TEST(TriangleTest, CentroidCalculation) {
    Triangle triangle(Point2D(0.0, 0.0), Point2D(6.0, 0.0), Point2D(0.0, 9.0));
    Point2D centroid = triangle.getCentroid();
    
    EXPECT_DOUBLE_EQ(centroid.x(), 2.0);
    EXPECT_DOUBLE_EQ(centroid.y(), 3.0);
}

TEST(TriangleTest, TriangleTypeDetection) {
    Triangle equilateral(Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(1.0, std::sqrt(3.0)));
    Triangle isosceles(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(1.5, 4.0));
    Triangle right(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(0.0, 4.0));
    
    EXPECT_TRUE(equilateral.isEquilateral());
    EXPECT_TRUE(isosceles.isIsosceles());
    EXPECT_TRUE(right.isRight());
    
    EXPECT_FALSE(equilateral.isRight());
    EXPECT_FALSE(isosceles.isEquilateral());
    EXPECT_FALSE(right.isEquilateral());
}

TEST(TriangleTest, PointContainment) {
    Triangle triangle(Point2D(0.0, 0.0), Point2D(4.0, 0.0), Point2D(0.0, 3.0));
    
    EXPECT_TRUE(triangle.contains(Point2D(1.0, 1.0)));
    EXPECT_TRUE(triangle.contains(Point2D(2.0, 0.5)));  // На стороне
    EXPECT_FALSE(triangle.contains(Point2D(-1.0, -1.0)));
    EXPECT_FALSE(triangle.contains(Point2D(3.0, 3.0)));
}

TEST(TriangleTest, MoveOperation) {
    Triangle triangle(Point2D(1.0, 2.0), Point2D(4.0, 2.0), Point2D(1.0, 5.0));
    Vector2D offset(2.0, 3.0);
    
    triangle.move(offset);
    
    EXPECT_EQ(triangle.vertex(0), Point2D(3.0, 5.0));
    EXPECT_EQ(triangle.vertex(1), Point2D(6.0, 5.0));
    EXPECT_EQ(triangle.vertex(2), Point2D(3.0, 8.0));
}

TEST(TriangleTest, ScaleOperation) {
    Triangle triangle(Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(0.0, 2.0));
    
    triangle.scale(2.0);
    
    // После масштабирования 2x относительно (0,0)
    EXPECT_EQ(triangle.vertex(0), Point2D(0.0, 0.0));
    EXPECT_EQ(triangle.vertex(1), Point2D(4.0, 0.0));
    EXPECT_EQ(triangle.vertex(2), Point2D(0.0, 4.0));
    
    EXPECT_NEAR(triangle.getSideLength(0), 4.0, 1e-10);
    EXPECT_NEAR(triangle.getSideLength(1), 4.0 * std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(triangle.getSideLength(2), 4.0, 1e-10);
}

TEST(TriangleTest, EqualityOperators) {
    Triangle triangle1(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(0.0, 4.0));
    Triangle triangle2(Point2D(0.0, 0.0), Point2D(3.0, 0.0), Point2D(0.0, 4.0));
    Triangle triangle3(Point2D(1.0, 1.0), Point2D(4.0, 1.0), Point2D(1.0, 5.0));
    
    EXPECT_TRUE(triangle1 == triangle2);
    EXPECT_FALSE(triangle1 == triangle3);
    EXPECT_TRUE(triangle1 != triangle3);
    EXPECT_FALSE(triangle1 != triangle2);
}

TEST(TriangleTest, VectorAdditionOperators) {
    Triangle triangle(Point2D(1.0, 2.0), Point2D(4.0, 2.0), Point2D(1.0, 5.0));
    Vector2D offset(1.0, -1.0);
    
    Triangle result1 = triangle + offset;
    Triangle result2 = triangle - offset;
    
    EXPECT_EQ(result1.vertex(0), Point2D(2.0, 1.0));
    EXPECT_EQ(result2.vertex(0), Point2D(0.0, 3.0));
}

TEST(TriangleTest, CompoundAssignmentOperators) {
    Triangle triangle(Point2D(1.0, 2.0), Point2D(4.0, 2.0), Point2D(1.0, 5.0));
    Vector2D offset(1.0, -1.0);
    
    triangle += offset;
    EXPECT_EQ(triangle.vertex(0), Point2D(2.0, 1.0));
    
    triangle -= offset;
    EXPECT_EQ(triangle.vertex(0), Point2D(1.0, 2.0));
}

TEST(TriangleTest, AltitudeCalculation) {
    Triangle triangle(Point2D(0.0, 0.0), Point2D(6.0, 0.0), Point2D(0.0, 8.0));
    
    // Вершины: A(0,0), B(6,0), C(0,8)
    // Стороны: 
    // side0: AB (длина 6)
    // side1: BC (длина 10)
    // side2: CA (длина 8)
    
    double altitudeA = triangle.getAltitude(0);  // Высота из C к стороне AB
    double altitudeB = triangle.getAltitude(1);  // Высота из A к стороне BC  
    double altitudeC = triangle.getAltitude(2);  // Высота из B к стороне CA
    
    // Проверяем с допуском для плавающей точки
    EXPECT_NEAR(altitudeA, 8.0, 1e-10);    // Высота из C к AB = 8
    EXPECT_NEAR(altitudeB, 4.8, 1e-10);    // Высота из A к BC = (6*8)/10 = 4.8
    EXPECT_NEAR(altitudeC, 6.0, 1e-10);    // Высота из B к CA = (6*8)/8 = 6
}

TEST(TriangleTest, CircumcenterCalculation) {
    Triangle rightTriangle(Point2D(0.0, 0.0), Point2D(6.0, 0.0), Point2D(0.0, 8.0));
    Point2D circumcenter = rightTriangle.getCircumcenter();
    
    // Для прямоугольного треугольника центр описанной окружности - середина гипотенузы
    EXPECT_DOUBLE_EQ(circumcenter.x(), 3.0);
    EXPECT_DOUBLE_EQ(circumcenter.y(), 4.0);
}