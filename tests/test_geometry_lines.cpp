#include <gtest/gtest.h>
#include "../include/SharedMath.h"


namespace SharedMath::Geometry {
    TEST(PointTest, DefaultConstructor) {
        Point2D p;
        EXPECT_DOUBLE_EQ(p[0], 0.0);
        EXPECT_DOUBLE_EQ(p[1], 0.0);
    }

    TEST(PointTest, InitializerListConstructor) {
        Point3D p{1.1, 2.2, 3.3};
        EXPECT_DOUBLE_EQ(p[0], 1.1);
        EXPECT_DOUBLE_EQ(p[1], 2.2);
        EXPECT_DOUBLE_EQ(p[2], 3.3);
    }

    TEST(PointTest, CopyConstructor) {
        Point2D p1{5.0, 10.0};
        Point2D p2(p1);
        EXPECT_EQ(p1, p2);
    }

    TEST(PointTest, MoveConstructor) {
        Point2D p1{3.0, 4.0};
        Point2D p2(std::move(p1));
        EXPECT_EQ(p2[0], 3.0);
        EXPECT_EQ(p2[1], 4.0);
        EXPECT_EQ(p1[0], 0.0);  // Проверка, что p1 обнулился
    }

    TEST(PointTest, AssignmentOperator) {
        Point3D p1{1.0, 2.0, 3.0};
        Point3D p2;
        p2 = p1;
        EXPECT_EQ(p1, p2);
    }

    TEST(PointTest, MoveAssignmentOperator) {
        Point2D p1{7.0, 8.0};
        Point2D p2;
        p2 = std::move(p1);
        EXPECT_EQ(p2[0], 7.0);
        EXPECT_EQ(p2[1], 8.0);
        EXPECT_EQ(p1[0], 0.0);  // Проверка, что p1 обнулился
    }

    TEST(PointTest, EqualityOperator) {
        Point2D p1{1.0, 2.0};
        Point2D p2{1.0, 2.0};
        Point2D p3{3.0, 4.0};
        EXPECT_TRUE(p1 == p2);
        EXPECT_FALSE(p1 == p3);
    }

    TEST(PointTest, ClearPoint) {
        Point3D p{1.0, 2.0, 3.0};
        p.clearPoint();
        EXPECT_DOUBLE_EQ(p[0], 0.0);
        EXPECT_DOUBLE_EQ(p[1], 0.0);
        EXPECT_DOUBLE_EQ(p[2], 0.0);
    }

    TEST(PointTest, BracketOperatorOutOfBounds) {
        Point3D p{1.0, 2.0, 3.0};
        EXPECT_THROW(p[3], std::out_of_range); // Проверка выхода за границы
    }

    TEST(PointTest, ConstBracketOperator) {
        const Point2D p{1.5, 2.5};
        EXPECT_DOUBLE_EQ(p[0], 1.5); // Проверка const-версии operator[]
    }

    TEST(PointTest, InequalityOperator) {
        Point2D p1{1.0, 2.0};
        Point2D p2{1.0, 2.1};
        EXPECT_TRUE(p1 != p2);
    }

    TEST(PointTest, ZeroLength) {
        Point3D p;
        EXPECT_DOUBLE_EQ(p[0], 0.0);
        EXPECT_DOUBLE_EQ(p[1], 0.0);
        EXPECT_DOUBLE_EQ(p[2], 0.0);
    }

    TEST(LineTest, DefaultConstructor) {
        Line<2> line;
        EXPECT_EQ(line.getFirstPoint(), Point2D());
        EXPECT_EQ(line.getSecondPoint(), Point2D());
    }

    TEST(LineTest, ParameterizedConstructor) {
        Point2D p1{1.0, 2.0};
        Point2D p2{3.0, 4.0};
        Line<2> line(p1, p2);
        EXPECT_EQ(line.getFirstPoint(), p1);
        EXPECT_EQ(line.getSecondPoint(), p2);
    }

    TEST(LineTest, CopyConstructor) {
        Line<3> line1({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0});
        Line<3> line2(line1);
        EXPECT_EQ(line1, line2);
    }

    TEST(LineTest, MoveConstructor) {
        Point2D p1{1.0, 1.0};
        Point2D p2{2.0, 2.0};
        Line<2> line1(p1, p2);
        Line<2> line2(std::move(line1));
        
        EXPECT_EQ(line2.getFirstPoint(), p1);
        EXPECT_EQ(line2.getSecondPoint(), p2);
        EXPECT_EQ(line1.getFirstPoint(), Point2D());  // Проверка, что line1 обнулился
    }

    TEST(LineTest, AssignmentOperator) {
        Line<2> line1({0.0, 0.0}, {1.0, 1.0});
        Line<2> line2;
        line2 = line1;
        EXPECT_EQ(line1, line2);
    }

    TEST(LineTest, MoveAssignmentOperator) {
        Line<3> line1({1.0, 1.0, 1.0}, {2.0, 2.0, 2.0});
        Line<3> line2;
        line2 = std::move(line1);
        
        EXPECT_EQ(line2.getFirstPoint(), Point3D({1.0, 1.0, 1.0}));
        EXPECT_EQ(line2.getSecondPoint(), Point3D({2.0, 2.0, 2.0}));
        EXPECT_EQ(line1.getFirstPoint(), Point3D());  // Проверка, что line1 обнулился
    }

    TEST(LineTest, EqualityOperator) {
        Line<2> line1({1.0, 2.0}, {3.0, 4.0});
        Line<2> line2({1.0, 2.0}, {3.0, 4.0});
        Line<2> line3({5.0, 6.0}, {7.0, 8.0});
        EXPECT_TRUE(line1 == line2);
        EXPECT_FALSE(line1 == line3);
    }

    TEST(LineTest, GetLength) {
        Line<2> line({0.0, 0.0}, {3.0, 4.0});  // 3-4-5 triangle
        EXPECT_DOUBLE_EQ(line.getLength(), 5.0);
        
        Line<3> line3d({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
        EXPECT_DOUBLE_EQ(line3d.getLength(), std::sqrt(3.0));
    }

    TEST(LineTest, SetPoints) {
        Line<2> line;
        Point2D p1{1.0, 2.0};
        Point2D p2{3.0, 4.0};
        
        line.setFirstPoint(p1);
        line.setSecondPoint(p2);
        
        EXPECT_EQ(line.getFirstPoint(), p1);
        EXPECT_EQ(line.getSecondPoint(), p2);
    }

    TEST(LineTest, DegenerateLine) {
        Point2D p{1.0, 1.0};
        Line<2> line(p, p); // Линия с одинаковыми точками
        EXPECT_DOUBLE_EQ(line.getLength(), 0.0);
    }

    TEST(LineTest, VerticalLine) {
        Line<2> line({0.0, 0.0}, {0.0, 5.0}); // Вертикальная линия
        EXPECT_DOUBLE_EQ(line.getLength(), 5.0);
    }

    TEST(LineTest, HorizontalLine) {
        Line<2> line({0.0, 0.0}, {5.0, 0.0}); // Горизонтальная линия
        EXPECT_DOUBLE_EQ(line.getLength(), 5.0);
    }

    TEST(LineTest, 3DLineLength) {
        Line<3> line({1.0, 2.0, 3.0}, {4.0, 6.0, 9.0});
        double expected = std::sqrt(3*3 + 4*4 + 6*6);
        EXPECT_DOUBLE_EQ(line.getLength(), expected);
    }

    TEST(LineTest, SetPointsWithMove) {
        Line<2> line;
        Point2D p1{1.0, 1.0};
        Point2D p2{2.0, 2.0};
        
        line.setFirstPoint(std::move(p1));
        line.setSecondPoint(std::move(p2));
        
        EXPECT_EQ(line.getFirstPoint(), Point2D({1.0, 1.0}));
        EXPECT_EQ(line.getSecondPoint(), Point2D({2.0, 2.0}));
        EXPECT_EQ(p1, Point2D()); // Проверка, что p1 обнулился
    }

    TEST(LineEdgeCasesTest, ZeroLengthLine) {
        Point2D p{1.0, 1.0};
        Line<2> line(p, p);
        EXPECT_DOUBLE_EQ(line.getLength(), 0.0);
        EXPECT_TRUE(line.getFirstPoint() == line.getSecondPoint());
    }

    TEST(PointEdgeCasesTest, LargeCoordinates) {
        Point2D p{1e300, 1e300};
        EXPECT_DOUBLE_EQ(p[0], 1e300);
        EXPECT_DOUBLE_EQ(p[1], 1e300);
    }

    TEST(LineEdgeCasesTest, LargeCoordinatesLine) {
        Line<2> line({0.0, 0.0}, {1e200, 1e200});
        EXPECT_TRUE(std::isinf(line.getLength())); // Проверка на переполнение
    }

    TEST(Vector2D, DefaultConstructor) {
        Vector2D vec;
        EXPECT_NEAR(vec.x(), 0.0, Epsilon);
        EXPECT_NEAR(vec.y(), 0.0, Epsilon);
        EXPECT_TRUE(vec.isZero());
    }

    TEST(Vector2D, PointConstructor) {
        Point2D p(3.0, 4.0);
        Vector2D vec(p);
        EXPECT_NEAR(vec.x(), 3.0, Epsilon);
        EXPECT_NEAR(vec.y(), 4.0, Epsilon);
        EXPECT_NEAR(vec.length(), 5.0, Epsilon);
    }

    TEST(Vector2D, ComponentConstructor) {
        Vector2D vec(3.0, 4.0);
        EXPECT_NEAR(vec.x(), 3.0, Epsilon);
        EXPECT_NEAR(vec.y(), 4.0, Epsilon);
        EXPECT_NEAR(vec.length(), 5.0, Epsilon);
    }

    TEST(Vector2D, TwoPointConstructor) {
        Point2D start(1.0, 2.0);
        Point2D end(4.0, 6.0);
        Vector2D vec(start, end);
        EXPECT_NEAR(vec.x(), 3.0, Epsilon);
        EXPECT_NEAR(vec.y(), 4.0, Epsilon);
    }

    TEST(Vector2D, Normalization) {
        Vector2D vec(3.0, 4.0);
        Vector2D norm = vec.normalized();
        EXPECT_NEAR(norm.length(), 1.0, Epsilon);
        EXPECT_NEAR(norm.x(), 0.6, Epsilon);
        EXPECT_NEAR(norm.y(), 0.8, Epsilon);
    }

    TEST(Vector2D, DotProduct) {
        Vector2D a(1.0, 2.0);
        Vector2D b(3.0, 4.0);
        double dot = a.dot(b);
        EXPECT_NEAR(dot, 11.0, Epsilon); // 1*3 + 2*4 = 11
    }

    TEST(Vector2D, CrossProduct) {
        Vector2D a(1.0, 2.0);
        Vector2D b(3.0, 4.0);
        double cross = a.cross(b);
        EXPECT_NEAR(cross, -2.0, Epsilon); // 1*4 - 2*3 = -2
    }

    TEST(Vector2D, VectorAddition) {
        Vector2D a(1.0, 2.0);
        Vector2D b(3.0, 4.0);
        Vector2D result = a + b;
        EXPECT_NEAR(result.x(), 4.0, Epsilon);
        EXPECT_NEAR(result.y(), 6.0, Epsilon);
    }

    TEST(Vector2D, VectorSubtraction) {
        Vector2D a(5.0, 6.0);
        Vector2D b(2.0, 3.0);
        Vector2D result = a - b;
        EXPECT_NEAR(result.x(), 3.0, Epsilon);
        EXPECT_NEAR(result.y(), 3.0, Epsilon);
    }

    TEST(Vector2D, ScalarMultiplication) {
        Vector2D a(2.0, 3.0);
        Vector2D result = a * 2.5;
        EXPECT_NEAR(result.x(), 5.0, Epsilon);
        EXPECT_NEAR(result.y(), 7.5, Epsilon);
    }

    TEST(Vector2D, ScalarDivision) {
        Vector2D a(6.0, 9.0);
        Vector2D result = a / 3.0;
        EXPECT_NEAR(result.x(), 2.0, Epsilon);
        EXPECT_NEAR(result.y(), 3.0, Epsilon);
    }

    TEST(Vector2D, ParallelCheck) {
        Vector2D a(2.0, 4.0);
        Vector2D b(1.0, 2.0); // b = 0.5 * a
        EXPECT_TRUE(a.isParallel(b));
        
        Vector2D c(2.0, 3.0);
        EXPECT_FALSE(a.isParallel(c));
    }

    TEST(Vector2D, PerpendicularCheck) {
        Vector2D a(1.0, 0.0);
        Vector2D b(0.0, 1.0);
        EXPECT_TRUE(a.isPerpendicular(b));
        
        Vector2D c(1.0, 1.0);
        EXPECT_FALSE(a.isPerpendicular(c));
    }

    TEST(Vector2D, Rotation) {
        Vector2D vec(1.0, 0.0);
        Vector2D rotated = vec.rotate(M_PI / 2); // 90 degrees
        EXPECT_NEAR(rotated.x(), 0.0, Epsilon);
        EXPECT_NEAR(rotated.y(), 1.0, Epsilon);
    }

    TEST(Vector2D, NormalVector) {
        Vector2D vec(3.0, 4.0);
        Vector2D normal = vec.normal();
        EXPECT_NEAR(normal.x(), -4.0, Epsilon);
        EXPECT_NEAR(normal.y(), 3.0, Epsilon);
        EXPECT_TRUE(vec.isPerpendicular(normal));
    }

    TEST(Vector3D, DefaultConstructor) {
        Vector3D vec;
        EXPECT_NEAR(vec.x(), 0.0, Epsilon);
        EXPECT_NEAR(vec.y(), 0.0, Epsilon);
        EXPECT_NEAR(vec.z(), 0.0, Epsilon);
        EXPECT_TRUE(vec.isZero());
    }

    TEST(Vector3D, ComponentConstructor) {
        Vector3D vec(1.0, 2.0, 3.0);
        EXPECT_NEAR(vec.x(), 1.0, Epsilon);
        EXPECT_NEAR(vec.y(), 2.0, Epsilon);
        EXPECT_NEAR(vec.z(), 3.0, Epsilon);
        EXPECT_NEAR(vec.length(), std::sqrt(14.0), Epsilon);
    }

    TEST(Vector3D, TwoPointConstructor) {
        Point3D start(1.0, 2.0, 3.0);
        Point3D end(5.0, 7.0, 9.0);
        Vector3D vec(start, end);
        EXPECT_NEAR(vec.x(), 4.0, Epsilon);
        EXPECT_NEAR(vec.y(), 5.0, Epsilon);
        EXPECT_NEAR(vec.z(), 6.0, Epsilon);
    }

    TEST(Vector3D, DotProduct) {
        Vector3D a(1.0, 2.0, 3.0);
        Vector3D b(4.0, 5.0, 6.0);
        double dot = a.dot(b);
        EXPECT_NEAR(dot, 32.0, Epsilon); // 1*4 + 2*5 + 3*6 = 32
    }

    TEST(Vector3D, CrossProduct) {
        Vector3D a(1.0, 2.0, 3.0);
        Vector3D b(4.0, 5.0, 6.0);
        Vector3D cross = a.cross(b);
        EXPECT_NEAR(cross.x(), -3.0, Epsilon); // 2*6 - 3*5 = -3
        EXPECT_NEAR(cross.y(), 6.0, Epsilon);  // 3*4 - 1*6 = 6
        EXPECT_NEAR(cross.z(), -3.0, Epsilon); // 1*5 - 2*4 = -3
    }

    TEST(Vector3D, VectorOperations) {
        Vector3D a(1.0, 2.0, 3.0);
        Vector3D b(4.0, 5.0, 6.0);
        
        Vector3D sum = a + b;
        EXPECT_NEAR(sum.x(), 5.0, Epsilon);
        EXPECT_NEAR(sum.y(), 7.0, Epsilon);
        EXPECT_NEAR(sum.z(), 9.0, Epsilon);
        
        Vector3D diff = a - b;
        EXPECT_NEAR(diff.x(), -3.0, Epsilon);
        EXPECT_NEAR(diff.y(), -3.0, Epsilon);
        EXPECT_NEAR(diff.z(), -3.0, Epsilon);
        
        Vector3D scaled = a * 2.0;
        EXPECT_NEAR(scaled.x(), 2.0, Epsilon);
        EXPECT_NEAR(scaled.y(), 4.0, Epsilon);
        EXPECT_NEAR(scaled.z(), 6.0, Epsilon);
    }

    TEST(Vector3D, ParallelCheck) {
        Vector3D a(2.0, 4.0, 6.0);
        Vector3D b(1.0, 2.0, 3.0); // b = 0.5 * a
        EXPECT_TRUE(a.isParallel(b));
        
        Vector3D c(2.0, 3.0, 4.0);
        EXPECT_FALSE(a.isParallel(c));
    }

    TEST(Vector3D, PerpendicularCheck) {
        Vector3D a(1.0, 0.0, 0.0);
        Vector3D b(0.0, 1.0, 0.0);
        EXPECT_TRUE(a.isPerpendicular(b));
        
        Vector3D c(1.0, 1.0, 0.0);
        EXPECT_FALSE(a.isPerpendicular(c));
    }

    TEST(Vector3D, TripleProduct) {
        Vector3D a(1.0, 0.0, 0.0);
        Vector3D b(0.0, 1.0, 0.0);
        Vector3D c(0.0, 0.0, 1.0);
        
        double triple = a.tripleProduct(b, c);
        EXPECT_NEAR(triple, 1.0, Epsilon); // Volume of unit cube
    }

    TEST(Vector3D, Normalization) {
        Vector3D vec(2.0, 3.0, 6.0);
        Vector3D norm = vec.normalized();
        EXPECT_NEAR(norm.length(), 1.0, Epsilon);
        EXPECT_NEAR(norm.x(), 2.0/7.0, Epsilon); // 2/7
        EXPECT_NEAR(norm.y(), 3.0/7.0, Epsilon); // 3/7
        EXPECT_NEAR(norm.z(), 6.0/7.0, Epsilon); // 6/7
    }

    TEST(VectorExceptions, NormalizeZeroVector) {
        Vector2D zero2D;
        EXPECT_THROW(zero2D.normalized(), std::invalid_argument);
        
        Vector3D zero3D;
        EXPECT_THROW(zero3D.normalized(), std::invalid_argument);
    }

    TEST(VectorExceptions, DivisionByZero) {
        Vector2D vec2D(1.0, 1.0);
        EXPECT_THROW(vec2D / 0.0, std::invalid_argument);
        
        Vector3D vec3D(1.0, 1.0, 1.0);
        EXPECT_THROW(vec3D / 0.0, std::invalid_argument);
    }
    
}