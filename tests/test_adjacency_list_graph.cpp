#include <gtest/gtest.h>
#include <string>
#include <algorithm>
#include "Graphs/AdjacencyListGraph.h"

using namespace SharedMath::Graphs;
using DirGraph   = AdjacencyListGraph<int, double>;
using UndirGraph = AdjacencyListGraph<int, double>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {
    UndirGraph makeUndirected() { return UndirGraph(false); }
    DirGraph   makeDirected()   { return DirGraph(true);    }

    std::vector<int> sortedNeighbors(const DirGraph& g, int v) {
        std::vector<int> ns;
        for (const auto& e : g.neighbors(v)) ns.push_back(e.to);
        std::sort(ns.begin(), ns.end());
        return ns;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vertex operations
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, EmptyGraph) {
    DirGraph g;
    EXPECT_EQ(g.vertexCount(), 0u);
    EXPECT_EQ(g.edgeCount(), 0u);
}

TEST(AdjacencyListGraphTest, AddVertex) {
    DirGraph g;
    g.addVertex(1);
    g.addVertex(2);
    EXPECT_EQ(g.vertexCount(), 2u);
    EXPECT_TRUE(g.hasVertex(1));
    EXPECT_TRUE(g.hasVertex(2));
    EXPECT_FALSE(g.hasVertex(3));
}

TEST(AdjacencyListGraphTest, AddVertexIdempotent) {
    DirGraph g;
    g.addVertex(5);
    g.addVertex(5);
    EXPECT_EQ(g.vertexCount(), 1u);
}

TEST(AdjacencyListGraphTest, AddEdgeCreatesVerticesImplicitly) {
    DirGraph g;
    g.addEdge(1, 2, 3.0);
    EXPECT_TRUE(g.hasVertex(1));
    EXPECT_TRUE(g.hasVertex(2));
}

TEST(AdjacencyListGraphTest, RemoveIsolatedVertex) {
    DirGraph g;
    g.addVertex(1);
    g.addVertex(2);
    g.removeVertex(1);
    EXPECT_FALSE(g.hasVertex(1));
    EXPECT_TRUE(g.hasVertex(2));
    EXPECT_EQ(g.vertexCount(), 1u);
}

TEST(AdjacencyListGraphTest, RemoveVertexAlsoRemovesIncidentEdges) {
    DirGraph g;
    g.addEdge(1, 2, 1.0);
    g.addEdge(2, 3, 1.0);
    g.addEdge(3, 1, 1.0);
    g.removeVertex(2);

    EXPECT_FALSE(g.hasEdge(1, 2));
    EXPECT_FALSE(g.hasEdge(2, 3));
    EXPECT_TRUE(g.hasEdge(3, 1));  // unaffected
}

TEST(AdjacencyListGraphTest, RemoveNonExistentVertexIsNoop) {
    DirGraph g;
    g.addVertex(1);
    g.removeVertex(99);  // should not throw
    EXPECT_EQ(g.vertexCount(), 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge operations — directed
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, DirectedAddEdge) {
    DirGraph g;
    g.addEdge(1, 2, 5.0);
    EXPECT_TRUE(g.hasEdge(1, 2));
    EXPECT_FALSE(g.hasEdge(2, 1));  // directed: reverse does not exist
}

TEST(AdjacencyListGraphTest, DirectedEdgeWeight) {
    DirGraph g;
    g.addEdge(1, 2, 3.14);
    auto w = g.weight(1, 2);
    ASSERT_TRUE(w.has_value());
    EXPECT_DOUBLE_EQ(w.value(), 3.14);
}

TEST(AdjacencyListGraphTest, WeightNulloptForMissingEdge) {
    DirGraph g;
    g.addEdge(1, 2, 1.0);
    EXPECT_FALSE(g.weight(2, 1).has_value());
    EXPECT_FALSE(g.weight(1, 3).has_value());
}

TEST(AdjacencyListGraphTest, RemoveDirectedEdge) {
    DirGraph g;
    g.addEdge(1, 2, 1.0);
    g.addEdge(1, 3, 2.0);
    g.removeEdge(1, 2);
    EXPECT_FALSE(g.hasEdge(1, 2));
    EXPECT_TRUE(g.hasEdge(1, 3));
}

TEST(AdjacencyListGraphTest, RemoveNonExistentEdgeIsNoop) {
    DirGraph g;
    g.addEdge(1, 2, 1.0);
    g.removeEdge(1, 99);  // no throw
    EXPECT_TRUE(g.hasEdge(1, 2));
}

TEST(AdjacencyListGraphTest, DirectedEdgeCount) {
    DirGraph g;
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 1);
    EXPECT_EQ(g.edgeCount(), 3u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Edge operations — undirected
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, UndirectedAddEdgeSymmetric) {
    auto g = makeUndirected();
    g.addEdge(1, 2, 7.0);
    EXPECT_TRUE(g.hasEdge(1, 2));
    EXPECT_TRUE(g.hasEdge(2, 1));  // symmetric
}

TEST(AdjacencyListGraphTest, UndirectedEdgeCountNotDoubled) {
    auto g = makeUndirected();
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 1);
    EXPECT_EQ(g.edgeCount(), 3u);  // each undirected edge counted once
}

TEST(AdjacencyListGraphTest, UndirectedRemoveEdgeBothDirections) {
    auto g = makeUndirected();
    g.addEdge(1, 2, 1.0);
    g.removeEdge(1, 2);
    EXPECT_FALSE(g.hasEdge(1, 2));
    EXPECT_FALSE(g.hasEdge(2, 1));
}

TEST(AdjacencyListGraphTest, UndirectedRemoveVertexCleansBothSides) {
    auto g = makeUndirected();
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 3);
    g.removeVertex(1);
    EXPECT_FALSE(g.hasEdge(2, 1));
    EXPECT_FALSE(g.hasEdge(3, 1));
    EXPECT_TRUE(g.hasEdge(2, 3));
}

// ─────────────────────────────────────────────────────────────────────────────
// Degrees
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, DirectedInOutDegree) {
    DirGraph g;
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(3, 2);
    EXPECT_EQ(g.outDegree(1), 2u);
    EXPECT_EQ(g.inDegree(1),  0u);
    EXPECT_EQ(g.outDegree(2), 0u);
    EXPECT_EQ(g.inDegree(2),  2u);
    EXPECT_EQ(g.outDegree(3), 1u);
    EXPECT_EQ(g.inDegree(3),  1u);
}

TEST(AdjacencyListGraphTest, DirectedInDegreeUpdatedOnRemove) {
    DirGraph g;
    g.addEdge(1, 2);
    g.addEdge(3, 2);
    EXPECT_EQ(g.inDegree(2), 2u);
    g.removeEdge(1, 2);
    EXPECT_EQ(g.inDegree(2), 1u);
}

TEST(AdjacencyListGraphTest, UndirectedDegreeSymmetric) {
    auto g = makeUndirected();
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    EXPECT_EQ(g.outDegree(1), g.inDegree(1));
    EXPECT_EQ(g.outDegree(1), 2u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Neighbors
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, NeighborsCorrect) {
    DirGraph g;
    g.addEdge(1, 2, 1.0);
    g.addEdge(1, 3, 2.0);
    g.addEdge(1, 4, 3.0);
    auto ns = sortedNeighbors(g, 1);
    EXPECT_EQ(ns, (std::vector<int>{2, 3, 4}));
}

TEST(AdjacencyListGraphTest, NeighborsThrowsForUnknownVertex) {
    DirGraph g;
    EXPECT_THROW(g.neighbors(99), std::out_of_range);
}

TEST(AdjacencyListGraphTest, NeighborsEmptyForIsolatedVertex) {
    DirGraph g;
    g.addVertex(5);
    EXPECT_TRUE(g.neighbors(5).empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// vertices()
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, VerticesContainsAll) {
    DirGraph g;
    g.addEdge(1, 2);
    g.addEdge(3, 4);
    auto vs = g.vertices();
    std::sort(vs.begin(), vs.end());
    EXPECT_EQ(vs, (std::vector<int>{1, 2, 3, 4}));
}

// ─────────────────────────────────────────────────────────────────────────────
// reversed()
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, ReversedDirectedGraph) {
    DirGraph g;
    g.addEdge(1, 2, 5.0);
    g.addEdge(2, 3, 3.0);
    g.addEdge(3, 1, 1.0);

    auto rev = g.reversed();
    EXPECT_TRUE(rev.hasEdge(2, 1));
    EXPECT_TRUE(rev.hasEdge(3, 2));
    EXPECT_TRUE(rev.hasEdge(1, 3));
    EXPECT_FALSE(rev.hasEdge(1, 2));
}

TEST(AdjacencyListGraphTest, ReversedPreservesWeights) {
    DirGraph g;
    g.addEdge(1, 2, 4.2);
    auto rev = g.reversed();
    ASSERT_TRUE(rev.weight(2, 1).has_value());
    EXPECT_DOUBLE_EQ(rev.weight(2, 1).value(), 4.2);
}

TEST(AdjacencyListGraphTest, ReversedUndirectedIsSame) {
    auto g = makeUndirected();
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    auto rev = g.reversed();
    // For undirected, reversed() returns a copy of the same graph
    EXPECT_TRUE(rev.hasEdge(1, 2));
    EXPECT_TRUE(rev.hasEdge(2, 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// isDirected()
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, IsDirectedFlag) {
    DirGraph   dg;
    auto       ug = makeUndirected();
    EXPECT_TRUE(dg.isDirected());
    EXPECT_FALSE(ug.isDirected());
}

// ─────────────────────────────────────────────────────────────────────────────
// Default weight
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, DefaultWeightIsOne) {
    DirGraph g;
    g.addEdge(1, 2);  // no explicit weight
    ASSERT_TRUE(g.weight(1, 2).has_value());
    EXPECT_DOUBLE_EQ(g.weight(1, 2).value(), 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// String vertex type
// ─────────────────────────────────────────────────────────────────────────────

TEST(AdjacencyListGraphTest, StringVertices) {
    AdjacencyListGraph<std::string, double> g;
    g.addEdge("A", "B", 2.0);
    g.addEdge("B", "C", 3.0);
    EXPECT_TRUE(g.hasEdge("A", "B"));
    EXPECT_FALSE(g.hasEdge("C", "A"));
    EXPECT_EQ(g.vertexCount(), 3u);
}
