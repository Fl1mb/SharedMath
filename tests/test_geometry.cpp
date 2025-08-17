#include <gtest/gtest.h>
#include "../include/SharedMath.h"


namespace SharedMath::Geometry {
    TEST(PointTest, DefaultConstructor) {
        Point<2> p;
        EXPECT_DOUBLE_EQ(p[0], 0.0);
        EXPECT_DOUBLE_EQ(p[1], 0.0);
    }

    TEST(PointTest, InitializerListConstructor) {
        Point<3> p{1.1, 2.2, 3.3};
        EXPECT_DOUBLE_EQ(p[0], 1.1);
        EXPECT_DOUBLE_EQ(p[1], 2.2);
        EXPECT_DOUBLE_EQ(p[2], 3.3);
    }

    TEST(PointTest, CopyConstructor) {
        Point<2> p1{5.0, 10.0};
        Point<2> p2(p1);
        EXPECT_EQ(p1, p2);
    }

    TEST(PointTest, MoveConstructor) {
        Point<2> p1{3.0, 4.0};
        Point<2> p2(std::move(p1));
        EXPECT_EQ(p2[0], 3.0);
        EXPECT_EQ(p2[1], 4.0);
        EXPECT_EQ(p1[0], 0.0);  // Проверка, что p1 обнулился
    }

    TEST(PointTest, AssignmentOperator) {
        Point<3> p1{1.0, 2.0, 3.0};
        Point<3> p2;
        p2 = p1;
        EXPECT_EQ(p1, p2);
    }

    TEST(PointTest, MoveAssignmentOperator) {
        Point<2> p1{7.0, 8.0};
        Point<2> p2;
        p2 = std::move(p1);
        EXPECT_EQ(p2[0], 7.0);
        EXPECT_EQ(p2[1], 8.0);
        EXPECT_EQ(p1[0], 0.0);  // Проверка, что p1 обнулился
    }

    TEST(PointTest, EqualityOperator) {
        Point<2> p1{1.0, 2.0};
        Point<2> p2{1.0, 2.0};
        Point<2> p3{3.0, 4.0};
        EXPECT_TRUE(p1 == p2);
        EXPECT_FALSE(p1 == p3);
    }

    TEST(PointTest, ClearPoint) {
        Point<3> p{1.0, 2.0, 3.0};
        p.clearPoint();
        EXPECT_DOUBLE_EQ(p[0], 0.0);
        EXPECT_DOUBLE_EQ(p[1], 0.0);
        EXPECT_DOUBLE_EQ(p[2], 0.0);
    }

    TEST(LineTest, DefaultConstructor) {
        Line<2> line;
        EXPECT_EQ(line.getFirstPoint(), Point<2>());
        EXPECT_EQ(line.getSecondPoint(), Point<2>());
    }

    TEST(LineTest, ParameterizedConstructor) {
        Point<2> p1{1.0, 2.0};
        Point<2> p2{3.0, 4.0};
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
        Point<2> p1{1.0, 1.0};
        Point<2> p2{2.0, 2.0};
        Line<2> line1(p1, p2);
        Line<2> line2(std::move(line1));
        
        EXPECT_EQ(line2.getFirstPoint(), p1);
        EXPECT_EQ(line2.getSecondPoint(), p2);
        EXPECT_EQ(line1.getFirstPoint(), Point<2>());  // Проверка, что line1 обнулился
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
        
        EXPECT_EQ(line2.getFirstPoint(), Point<3>({1.0, 1.0, 1.0}));
        EXPECT_EQ(line2.getSecondPoint(), Point<3>({2.0, 2.0, 2.0}));
        EXPECT_EQ(line1.getFirstPoint(), Point<3>());  // Проверка, что line1 обнулился
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
        Point<2> p1{1.0, 2.0};
        Point<2> p2{3.0, 4.0};
        
        line.setFirstPoint(p1);
        line.setSecondPoint(p2);
        
        EXPECT_EQ(line.getFirstPoint(), p1);
        EXPECT_EQ(line.getSecondPoint(), p2);
    }
}