#include <gtest/gtest.h>
#include <string>
#include "Graphs/UnionFind.h"

using namespace SharedMath::Graphs;

// ─────────────────────────────────────────────────────────────────────────────
// makeSet
// ─────────────────────────────────────────────────────────────────────────────

TEST(UnionFindTest, EmptyInitially) {
    UnionFind<int> uf;
    EXPECT_EQ(uf.size(), 0u);
    EXPECT_EQ(uf.numSets(), 0u);
}

TEST(UnionFindTest, MakeSetAddsElements) {
    UnionFind<int> uf;
    uf.makeSet(1);
    uf.makeSet(2);
    uf.makeSet(3);
    EXPECT_EQ(uf.size(), 3u);
    EXPECT_EQ(uf.numSets(), 3u);
}

TEST(UnionFindTest, MakeSetIdempotent) {
    UnionFind<int> uf;
    uf.makeSet(1);
    uf.makeSet(1);  // second call is a no-op
    uf.makeSet(1);
    EXPECT_EQ(uf.size(), 1u);
    EXPECT_EQ(uf.numSets(), 1u);
}

TEST(UnionFindTest, ContainsAfterMakeSet) {
    UnionFind<int> uf;
    EXPECT_FALSE(uf.contains(42));
    uf.makeSet(42);
    EXPECT_TRUE(uf.contains(42));
}

// ─────────────────────────────────────────────────────────────────────────────
// find
// ─────────────────────────────────────────────────────────────────────────────

TEST(UnionFindTest, FindSingletonIsSelf) {
    UnionFind<int> uf;
    uf.makeSet(7);
    EXPECT_EQ(uf.find(7), 7);
}

TEST(UnionFindTest, FindThrowsForUnregistered) {
    UnionFind<int> uf;
    EXPECT_THROW(uf.find(99), std::out_of_range);
}

// ─────────────────────────────────────────────────────────────────────────────
// unite
// ─────────────────────────────────────────────────────────────────────────────

TEST(UnionFindTest, UniteReturnsTrueForDifferentSets) {
    UnionFind<int> uf;
    uf.makeSet(1); uf.makeSet(2);
    EXPECT_TRUE(uf.unite(1, 2));
}

TEST(UnionFindTest, UniteReturnsFalseForSameSet) {
    UnionFind<int> uf;
    uf.makeSet(1); uf.makeSet(2);
    uf.unite(1, 2);
    EXPECT_FALSE(uf.unite(1, 2));  // already connected
}

TEST(UnionFindTest, UniteReducesNumSets) {
    UnionFind<int> uf;
    for (int i = 1; i <= 5; ++i) uf.makeSet(i);
    EXPECT_EQ(uf.numSets(), 5u);
    uf.unite(1, 2);
    EXPECT_EQ(uf.numSets(), 4u);
    uf.unite(3, 4);
    EXPECT_EQ(uf.numSets(), 3u);
    uf.unite(1, 3);
    EXPECT_EQ(uf.numSets(), 2u);
    uf.unite(5, 1);
    EXPECT_EQ(uf.numSets(), 1u);
}

TEST(UnionFindTest, UniteAlreadyConnectedNoSetsChange) {
    UnionFind<int> uf;
    uf.makeSet(1); uf.makeSet(2);
    uf.unite(1, 2);
    size_t before = uf.numSets();
    uf.unite(2, 1);  // no-op
    EXPECT_EQ(uf.numSets(), before);
}

// ─────────────────────────────────────────────────────────────────────────────
// connected
// ─────────────────────────────────────────────────────────────────────────────

TEST(UnionFindTest, NotConnectedInitially) {
    UnionFind<int> uf;
    uf.makeSet(1); uf.makeSet(2);
    EXPECT_FALSE(uf.connected(1, 2));
}

TEST(UnionFindTest, ConnectedAfterUnite) {
    UnionFind<int> uf;
    uf.makeSet(1); uf.makeSet(2);
    uf.unite(1, 2);
    EXPECT_TRUE(uf.connected(1, 2));
    EXPECT_TRUE(uf.connected(2, 1));  // symmetric
}

TEST(UnionFindTest, ConnectivityIsTransitive) {
    UnionFind<int> uf;
    for (int i = 1; i <= 4; ++i) uf.makeSet(i);
    uf.unite(1, 2);
    uf.unite(2, 3);
    uf.unite(3, 4);
    EXPECT_TRUE(uf.connected(1, 4));  // 1→2→3→4 transitive
    EXPECT_TRUE(uf.connected(4, 1));
}

TEST(UnionFindTest, TwoSeparateComponents) {
    UnionFind<int> uf;
    for (int i = 1; i <= 6; ++i) uf.makeSet(i);
    uf.unite(1, 2); uf.unite(2, 3);  // component {1,2,3}
    uf.unite(4, 5); uf.unite(5, 6);  // component {4,5,6}

    EXPECT_TRUE(uf.connected(1, 3));
    EXPECT_TRUE(uf.connected(4, 6));
    EXPECT_FALSE(uf.connected(1, 4));
    EXPECT_FALSE(uf.connected(3, 6));
}

// ─────────────────────────────────────────────────────────────────────────────
// Path compression (indirect check via large chain)
// ─────────────────────────────────────────────────────────────────────────────

TEST(UnionFindTest, LargeChainAllConnected) {
    UnionFind<int> uf;
    const int N = 1000;
    for (int i = 0; i < N; ++i) uf.makeSet(i);
    for (int i = 0; i < N - 1; ++i) uf.unite(i, i + 1);

    EXPECT_EQ(uf.numSets(), 1u);
    EXPECT_TRUE(uf.connected(0, N - 1));
    EXPECT_TRUE(uf.connected(N / 2, N - 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// clear
// ─────────────────────────────────────────────────────────────────────────────

TEST(UnionFindTest, ClearResetsState) {
    UnionFind<int> uf;
    uf.makeSet(1); uf.makeSet(2);
    uf.unite(1, 2);
    uf.clear();
    EXPECT_EQ(uf.size(), 0u);
    EXPECT_EQ(uf.numSets(), 0u);
    EXPECT_FALSE(uf.contains(1));
}

TEST(UnionFindTest, ClearAndReuse) {
    UnionFind<int> uf;
    uf.makeSet(1); uf.makeSet(2); uf.unite(1, 2);
    uf.clear();
    uf.makeSet(10); uf.makeSet(20);
    EXPECT_FALSE(uf.connected(10, 20));
    uf.unite(10, 20);
    EXPECT_TRUE(uf.connected(10, 20));
}

// ─────────────────────────────────────────────────────────────────────────────
// String key type
// ─────────────────────────────────────────────────────────────────────────────

TEST(UnionFindTest, StringKeys) {
    UnionFind<std::string> uf;
    uf.makeSet("a"); uf.makeSet("b"); uf.makeSet("c");
    uf.unite("a", "b");
    EXPECT_TRUE(uf.connected("a", "b"));
    EXPECT_FALSE(uf.connected("a", "c"));
}
