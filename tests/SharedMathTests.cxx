#include <gtest/gtest.h>

TEST(ExampleTest, BasicAssertions){
    EXPECT_EQ(2 + 2, 4);
    EXPECT_NE(1, 0);
}