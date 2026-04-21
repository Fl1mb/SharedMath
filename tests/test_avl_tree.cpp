#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include "Graphs/AVLTree.h"

using namespace SharedMath::Graphs;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {
    // Returns true when inorder() == expected sorted sequence
    template<typename T>
    bool isSorted(const std::vector<T>& v) {
        return std::is_sorted(v.begin(), v.end());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, DefaultConstructorEmpty) {
    AVLTree<int> t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.height(), 0);
}

TEST(AVLTreeTest, MinMaxEmptyReturnsNullopt) {
    AVLTree<int> t;
    EXPECT_FALSE(t.min().has_value());
    EXPECT_FALSE(t.max().has_value());
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, InsertSingle) {
    AVLTree<int> t;
    t.insert(42);
    EXPECT_EQ(t.size(), 1u);
    EXPECT_FALSE(t.empty());
    EXPECT_TRUE(t.contains(42));
    EXPECT_EQ(t.min().value(), 42);
    EXPECT_EQ(t.max().value(), 42);
}

TEST(AVLTreeTest, InsertDuplicatesIgnored) {
    AVLTree<int> t;
    t.insert(10);
    t.insert(10);
    t.insert(10);
    EXPECT_EQ(t.size(), 1u);
}

TEST(AVLTreeTest, InsertPreservesInorderSort) {
    AVLTree<int> t;
    std::vector<int> values = {5, 3, 8, 1, 4, 7, 9, 2, 6};
    for (int v : values) t.insert(v);

    auto sorted = t.inorder();
    EXPECT_TRUE(isSorted(sorted));
    EXPECT_EQ(sorted.size(), values.size());
}

TEST(AVLTreeTest, InsertAscendingStillBalanced) {
    // Without balancing this would degenerate into a linked list (height = N)
    // AVL guarantees height <= 1.44 * log2(N+2)
    AVLTree<int> t;
    const int N = 1000;
    for (int i = 0; i < N; ++i) t.insert(i);

    EXPECT_EQ(t.size(), static_cast<size_t>(N));
    EXPECT_LE(t.height(), static_cast<int>(std::ceil(1.45 * std::log2(N + 2))));
}

TEST(AVLTreeTest, InsertDescendingStillBalanced) {
    AVLTree<int> t;
    const int N = 1000;
    for (int i = N - 1; i >= 0; --i) t.insert(i);

    EXPECT_LE(t.height(), static_cast<int>(std::ceil(1.45 * std::log2(N + 2))));
}

TEST(AVLTreeTest, InsertRandomOrderInorderCorrect) {
    AVLTree<int> t;
    std::vector<int> vals = {15, 6, 20, 3, 9, 17, 25, 2, 4, 7, 13};
    for (int v : vals) t.insert(v);

    auto result = t.inorder();
    auto expected = vals;
    std::sort(expected.begin(), expected.end());

    EXPECT_EQ(result, expected);
}

// ─────────────────────────────────────────────────────────────────────────────
// Min / Max
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, MinMaxCorrect) {
    AVLTree<int> t;
    for (int v : {50, 20, 80, 10, 30, 60, 90}) t.insert(v);
    EXPECT_EQ(t.min().value(), 10);
    EXPECT_EQ(t.max().value(), 90);
}

TEST(AVLTreeTest, MinMaxUpdateAfterInsert) {
    AVLTree<int> t;
    t.insert(5);
    EXPECT_EQ(t.min().value(), 5);
    t.insert(1);
    EXPECT_EQ(t.min().value(), 1);
    t.insert(10);
    EXPECT_EQ(t.max().value(), 10);
}

// ─────────────────────────────────────────────────────────────────────────────
// Contains
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, ContainsExisting) {
    AVLTree<int> t;
    for (int v : {3, 7, 1, 9, 5}) t.insert(v);
    for (int v : {3, 7, 1, 9, 5}) EXPECT_TRUE(t.contains(v));
}

TEST(AVLTreeTest, ContainsMissing) {
    AVLTree<int> t;
    t.insert(5);
    EXPECT_FALSE(t.contains(0));
    EXPECT_FALSE(t.contains(6));
    EXPECT_FALSE(t.contains(100));
}

// ─────────────────────────────────────────────────────────────────────────────
// Remove
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, RemoveLeaf) {
    AVLTree<int> t;
    for (int v : {5, 3, 7}) t.insert(v);
    t.remove(3);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_FALSE(t.contains(3));
    EXPECT_TRUE(t.contains(5));
    EXPECT_TRUE(t.contains(7));
}

TEST(AVLTreeTest, RemoveNodeWithOneChild) {
    AVLTree<int> t;
    for (int v : {5, 3, 7, 2}) t.insert(v);
    // 3 has only left child (2)
    t.remove(3);
    EXPECT_EQ(t.size(), 3u);
    EXPECT_FALSE(t.contains(3));
    EXPECT_TRUE(t.contains(2));
}

TEST(AVLTreeTest, RemoveNodeWithTwoChildren) {
    AVLTree<int> t;
    for (int v : {5, 3, 7, 2, 4, 6, 8}) t.insert(v);
    t.remove(5);
    EXPECT_EQ(t.size(), 6u);
    EXPECT_FALSE(t.contains(5));
    // Tree must still be a valid BST — inorder must be sorted
    EXPECT_TRUE(isSorted(t.inorder()));
}

TEST(AVLTreeTest, RemoveRoot) {
    AVLTree<int> t;
    t.insert(10);
    t.remove(10);
    EXPECT_TRUE(t.empty());
    EXPECT_FALSE(t.contains(10));
    EXPECT_FALSE(t.min().has_value());
}

TEST(AVLTreeTest, RemoveNonExistentIsNoop) {
    AVLTree<int> t;
    t.insert(5);
    t.insert(3);
    t.remove(99);  // should not throw or change size
    EXPECT_EQ(t.size(), 2u);
}

TEST(AVLTreeTest, RemoveAllElements) {
    AVLTree<int> t;
    std::vector<int> vals = {4, 2, 6, 1, 3, 5, 7};
    for (int v : vals) t.insert(v);
    for (int v : vals) t.remove(v);
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

TEST(AVLTreeTest, RemainsSortedAfterManyRemovals) {
    AVLTree<int> t;
    for (int i = 1; i <= 20; ++i) t.insert(i);
    // Remove odd numbers
    for (int i = 1; i <= 20; i += 2) t.remove(i);

    auto result = t.inorder();
    EXPECT_EQ(result.size(), 10u);
    EXPECT_TRUE(isSorted(result));
    for (int v : result) EXPECT_EQ(v % 2, 0);  // only evens remain
}

TEST(AVLTreeTest, RemoveKeepsBalance) {
    AVLTree<int> t;
    const int N = 500;
    for (int i = 0; i < N; ++i) t.insert(i);
    for (int i = 0; i < N; i += 2) t.remove(i);  // remove evens

    EXPECT_EQ(t.size(), static_cast<size_t>(N / 2));
    EXPECT_LE(t.height(), static_cast<int>(std::ceil(1.45 * std::log2(N / 2 + 2))));
}

// ─────────────────────────────────────────────────────────────────────────────
// Clear
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, ClearEmptyTree) {
    AVLTree<int> t;
    t.clear();
    EXPECT_TRUE(t.empty());
}

TEST(AVLTreeTest, ClearNonEmpty) {
    AVLTree<int> t;
    for (int v : {1, 2, 3, 4, 5}) t.insert(v);
    t.clear();
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.height(), 0);
}

TEST(AVLTreeTest, ClearAndReuse) {
    AVLTree<int> t;
    for (int v : {10, 20, 30}) t.insert(v);
    t.clear();
    for (int v : {1, 2, 3}) t.insert(v);
    EXPECT_EQ(t.size(), 3u);
    EXPECT_TRUE(t.contains(1));
}

// ─────────────────────────────────────────────────────────────────────────────
// forEach
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, ForEachVisitsInOrder) {
    AVLTree<int> t;
    for (int v : {5, 3, 7, 1, 4}) t.insert(v);

    std::vector<int> visited;
    t.forEach([&](const int& v) { visited.push_back(v); });

    EXPECT_TRUE(isSorted(visited));
    EXPECT_EQ(visited.size(), 5u);
}

// ─────────────────────────────────────────────────────────────────────────────
// String type
// ─────────────────────────────────────────────────────────────────────────────

TEST(AVLTreeTest, StringType) {
    AVLTree<std::string> t;
    t.insert("banana");
    t.insert("apple");
    t.insert("cherry");

    auto v = t.inorder();
    EXPECT_EQ(v[0], "apple");
    EXPECT_EQ(v[1], "banana");
    EXPECT_EQ(v[2], "cherry");
}
