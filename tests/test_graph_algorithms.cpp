#include <gtest/gtest.h>
#include <algorithm>
#include <set>
#include <unordered_set>
#include "Graphs/GraphAlgorithms.h"

using namespace SharedMath::Graphs;

using G  = AdjacencyListGraph<int, double>;
using SG = AdjacencyListGraph<std::string, double>;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {

G makeDiamond() {
    //  1
    // / \
    // 2  3
    //  \ /
    //   4
    G g;
    g.addEdge(1, 2, 1.0);
    g.addEdge(1, 3, 4.0);
    g.addEdge(2, 4, 2.0);
    g.addEdge(3, 4, 1.0);
    return g;
}

G makeLinear(int n) {
    // 1 -> 2 -> ... -> n
    G g;
    for (int i = 1; i < n; ++i) g.addEdge(i, i + 1, 1.0);
    return g;
}

G makeUndirected() {
    G g(false);
    g.addEdge(1, 2, 2.0);
    g.addEdge(2, 3, 3.0);
    g.addEdge(3, 4, 1.0);
    g.addEdge(4, 1, 4.0);
    g.addEdge(1, 3, 6.0);
    return g;
}

// Sort SCCs for deterministic comparison
std::vector<std::set<int>> normalizeSCCs(std::vector<std::vector<int>> sccs) {
    std::vector<std::set<int>> result;
    for (auto& scc : sccs) result.emplace_back(scc.begin(), scc.end());
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// BFS / DFS traversal
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, BFSTraversalLinear) {
    auto g = makeLinear(5);
    auto order = bfsTraversal(g, 1);
    ASSERT_EQ(order.size(), 5u);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 2);
    EXPECT_EQ(order[4], 5);
}

TEST(GraphAlgorithmsTest, BFSTraversalUnknownStartReturnsEmpty) {
    G g;
    g.addEdge(1, 2);
    EXPECT_TRUE(bfsTraversal(g, 99).empty());
}

TEST(GraphAlgorithmsTest, BFSTraversalVisitsAllReachable) {
    auto g = makeDiamond();
    auto order = bfsTraversal(g, 1);
    std::set<int> visited(order.begin(), order.end());
    EXPECT_EQ(visited, (std::set<int>{1, 2, 3, 4}));
}

TEST(GraphAlgorithmsTest, BFSTraversalDoesNotCrossComponents) {
    G g;
    g.addEdge(1, 2);
    g.addEdge(3, 4);  // disconnected component
    auto order = bfsTraversal(g, 1);
    std::set<int> visited(order.begin(), order.end());
    EXPECT_EQ(visited, (std::set<int>{1, 2}));
}

TEST(GraphAlgorithmsTest, DFSTraversalVisitsAll) {
    auto g = makeDiamond();
    std::set<int> visited;
    dfsTraversal<int,double>(g, 1, [&](const int& v) { visited.insert(v); });
    EXPECT_EQ(visited, (std::set<int>{1, 2, 3, 4}));
}

TEST(GraphAlgorithmsTest, DFSTraversalCallbackCount) {
    auto g = makeLinear(6);
    int count = 0;
    dfsTraversal<int, double>(g, 1, [&](const int&) { ++count; });
    EXPECT_EQ(count, 6);
}

TEST(GraphAlgorithmsTest, BFSDistancesCorrect) {
    auto g = makeDiamond();
    auto dist = bfsDistances(g, 1);
    EXPECT_EQ(dist[1], 0u);
    EXPECT_EQ(dist[2], 1u);
    EXPECT_EQ(dist[3], 1u);
    EXPECT_EQ(dist[4], 2u);
}

TEST(GraphAlgorithmsTest, BFSDistancesLinear) {
    auto g = makeLinear(5);
    auto dist = bfsDistances(g, 1);
    for (int i = 1; i <= 5; ++i)
        EXPECT_EQ(dist[i], static_cast<size_t>(i - 1));
}

// ═════════════════════════════════════════════════════════════════════════════
// Dijkstra
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, DijkstraSingleVertex) {
    G g;
    g.addVertex(1);
    auto res = dijkstra(g, 1);
    EXPECT_DOUBLE_EQ(res.dist[1], 0.0);
}

TEST(GraphAlgorithmsTest, DijkstraSimplePath) {
    G g;
    g.addEdge(1, 2, 3.0);
    g.addEdge(2, 3, 4.0);
    auto res = dijkstra(g, 1);
    EXPECT_DOUBLE_EQ(res.dist[1], 0.0);
    EXPECT_DOUBLE_EQ(res.dist[2], 3.0);
    EXPECT_DOUBLE_EQ(res.dist[3], 7.0);
}

TEST(GraphAlgorithmsTest, DijkstraChoseShorterAlternative) {
    // Diamond: 1→2→4 costs 3, 1→3→4 costs 5 → shortest to 4 is 3
    auto g = makeDiamond();
    auto res = dijkstra(g, 1);
    EXPECT_DOUBLE_EQ(res.dist[4], 3.0);
}

TEST(GraphAlgorithmsTest, DijkstraUnreachableVertex) {
    G g;
    g.addEdge(1, 2, 1.0);
    g.addVertex(3);  // isolated
    auto res = dijkstra(g, 1);
    EXPECT_FALSE(res.reachable(3));
}

TEST(GraphAlgorithmsTest, DijkstraPathReconstruction) {
    auto g = makeDiamond();
    auto res = dijkstra(g, 1);
    auto path = res.pathTo(4);
    ASSERT_EQ(path.front(), 1);
    ASSERT_EQ(path.back(), 4);
    // Optimal: 1→2→4
    EXPECT_EQ(path, (std::vector<int>{1, 2, 4}));
}

TEST(GraphAlgorithmsTest, DijkstraPathToUnreachable) {
    G g;
    g.addEdge(1, 2);
    g.addVertex(3);
    auto res = dijkstra(g, 1);
    EXPECT_TRUE(res.pathTo(3).empty());
}

// ═════════════════════════════════════════════════════════════════════════════
// Bellman-Ford
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, BellmanFordNonNegative) {
    auto g = makeDiamond();
    auto res = bellmanFord(g, 1);
    ASSERT_TRUE(res.has_value());
    EXPECT_DOUBLE_EQ(res->dist[4], 3.0);
}

TEST(GraphAlgorithmsTest, BellmanFordNegativeWeights) {
    G g;
    g.addEdge(1, 2,  5.0);
    g.addEdge(1, 3,  2.0);
    g.addEdge(3, 2, -4.0);  // negative edge; best path 1→3→2 costs -2
    auto res = bellmanFord(g, 1);
    ASSERT_TRUE(res.has_value());
    EXPECT_DOUBLE_EQ(res->dist[2], -2.0);
}

TEST(GraphAlgorithmsTest, BellmanFordNegativeCycleDetected) {
    G g;
    g.addEdge(1, 2,  1.0);
    g.addEdge(2, 3, -3.0);
    g.addEdge(3, 1,  1.0);  // cycle 1→2→3→1 with total weight -1
    auto res = bellmanFord(g, 1);
    EXPECT_FALSE(res.has_value());
}

TEST(GraphAlgorithmsTest, BellmanFordPathReconstruction) {
    G g;
    g.addEdge(1, 2, 1.0);
    g.addEdge(2, 3, 2.0);
    g.addEdge(1, 3, 10.0);
    auto res = bellmanFord(g, 1);
    ASSERT_TRUE(res.has_value());
    auto path = res->pathTo(3);
    EXPECT_EQ(path, (std::vector<int>{1, 2, 3}));
}

// ═════════════════════════════════════════════════════════════════════════════
// Floyd-Warshall
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, FloydWarshallBasic) {
    auto g = makeDiamond();
    auto res = floydWarshall(g);
    ASSERT_TRUE(res.has_value());
    EXPECT_DOUBLE_EQ(res->dist[res->indexOf(1)][res->indexOf(4)], 3.0);
    EXPECT_DOUBLE_EQ(res->dist[res->indexOf(1)][res->indexOf(1)], 0.0);
}

TEST(GraphAlgorithmsTest, FloydWarshallSymmetricUndirected) {
    G g(false);
    g.addEdge(1, 2, 5.0);
    g.addEdge(2, 3, 3.0);
    auto res = floydWarshall(g);
    ASSERT_TRUE(res.has_value());
    size_t i1 = res->indexOf(1), i2 = res->indexOf(2), i3 = res->indexOf(3);
    EXPECT_DOUBLE_EQ(res->dist[i1][i3], 8.0);
    EXPECT_DOUBLE_EQ(res->dist[i3][i1], 8.0);  // symmetric
}

TEST(GraphAlgorithmsTest, FloydWarshallNegativeCycle) {
    G g;
    g.addEdge(1, 2,  1.0);
    g.addEdge(2, 1, -2.0);  // negative cycle
    auto res = floydWarshall(g);
    EXPECT_FALSE(res.has_value());
}

TEST(GraphAlgorithmsTest, FloydWarshallPathReconstruction) {
    G g;
    g.addEdge(1, 2, 1.0);
    g.addEdge(2, 3, 1.0);
    g.addEdge(1, 3, 5.0);
    auto res = floydWarshall(g);
    ASSERT_TRUE(res.has_value());
    auto path = res->path(1, 3);
    EXPECT_EQ(path.front(), 1);
    EXPECT_EQ(path.back(), 3);
    EXPECT_EQ(path.size(), 3u);  // 1→2→3
}

TEST(GraphAlgorithmsTest, FloydWarshallUnreachable) {
    G g;
    g.addEdge(1, 2, 1.0);
    g.addVertex(3);  // isolated
    auto res = floydWarshall(g);
    ASSERT_TRUE(res.has_value());
    EXPECT_FALSE(res->reachable(1, 3));
    EXPECT_TRUE(res->path(1, 3).empty());
}

// ═════════════════════════════════════════════════════════════════════════════
// Prim
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, PrimMSTEdgeCount) {
    auto g = makeUndirected();
    auto mst = prim(g, 1);
    // MST of a connected graph with N vertices has N-1 edges
    EXPECT_EQ(mst.edges.size(), 3u);  // 4 vertices → 3 edges
}

TEST(GraphAlgorithmsTest, PrimMSTIsMinimum) {
    auto g = makeUndirected();
    auto mst = prim(g, 1);
    // Manually verified: min spanning tree of the test graph weighs 6.0
    // edges: (1,2,2), (3,4,1), (2,3,3) → total 6
    EXPECT_DOUBLE_EQ(mst.totalWeight, 6.0);
}

TEST(GraphAlgorithmsTest, PrimSingleVertex) {
    G g(false);
    g.addVertex(1);
    auto mst = prim(g, 1);
    EXPECT_EQ(mst.edges.size(), 0u);
    EXPECT_DOUBLE_EQ(mst.totalWeight, 0.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Kruskal
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, KruskalMSTEdgeCount) {
    auto g = makeUndirected();
    auto mst = kruskal(g);
    EXPECT_EQ(mst.edges.size(), 3u);
}

TEST(GraphAlgorithmsTest, KruskalMSTSameWeightAsPrim) {
    auto g   = makeUndirected();
    auto mk  = kruskal(g);
    auto mp  = prim(g, 1);
    EXPECT_DOUBLE_EQ(mk.totalWeight, mp.totalWeight);
}

TEST(GraphAlgorithmsTest, KruskalDisconnectedGraph) {
    G g(false);
    g.addEdge(1, 2, 1.0);
    g.addEdge(3, 4, 2.0);  // separate component
    auto mst = kruskal(g);
    // Spanning forest: 2 components → 2 edges
    EXPECT_EQ(mst.edges.size(), 2u);
    EXPECT_DOUBLE_EQ(mst.totalWeight, 3.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Topological sort
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, TopologicalSortSimpleDAG) {
    G g;
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 4);
    g.addEdge(3, 4);
    auto result = topologicalSort(g);
    ASSERT_TRUE(result.has_value());
    const auto& order = result.value();
    EXPECT_EQ(order.size(), 4u);
    // 1 must come before 2, 3, 4
    auto pos = [&](int v) {
        return std::find(order.begin(), order.end(), v) - order.begin();
    };
    EXPECT_LT(pos(1), pos(2));
    EXPECT_LT(pos(1), pos(3));
    EXPECT_LT(pos(2), pos(4));
    EXPECT_LT(pos(3), pos(4));
}

TEST(GraphAlgorithmsTest, TopologicalSortLinear) {
    auto g = makeLinear(5);
    auto result = topologicalSort(g);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(GraphAlgorithmsTest, TopologicalSortCycleDetected) {
    G g;
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 1);  // cycle
    auto result = topologicalSort(g);
    EXPECT_FALSE(result.has_value());
}

TEST(GraphAlgorithmsTest, TopologicalSortSingleNode) {
    G g;
    g.addVertex(42);
    auto result = topologicalSort(g);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), (std::vector<int>{42}));
}

// ═════════════════════════════════════════════════════════════════════════════
// hasCycle
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, DirectedCycleDetected) {
    G g;
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 1);
    EXPECT_TRUE(hasCycle(g));
}

TEST(GraphAlgorithmsTest, DirectedDAGNoCycle) {
    auto g = makeDiamond();
    EXPECT_FALSE(hasCycle(g));
}

TEST(GraphAlgorithmsTest, DirectedSelfLoop) {
    G g;
    g.addEdge(1, 1);
    EXPECT_TRUE(hasCycle(g));
}

TEST(GraphAlgorithmsTest, UndirectedTreeNoCycle) {
    G g(false);
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 4);
    EXPECT_FALSE(hasCycle(g));
}

TEST(GraphAlgorithmsTest, UndirectedCycleDetected) {
    G g(false);
    g.addEdge(1, 2);
    g.addEdge(2, 3);
    g.addEdge(3, 1);
    EXPECT_TRUE(hasCycle(g));
}

TEST(GraphAlgorithmsTest, SingleVertexNoCycle) {
    G g;
    g.addVertex(1);
    EXPECT_FALSE(hasCycle(g));
}

// ═════════════════════════════════════════════════════════════════════════════
// Connected components
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, ConnectedSingleComponent) {
    G g(false);
    g.addEdge(1, 2); g.addEdge(2, 3); g.addEdge(3, 4);
    auto comps = connectedComponents(g);
    EXPECT_EQ(comps.size(), 1u);
    EXPECT_EQ(comps[0].size(), 4u);
}

TEST(GraphAlgorithmsTest, TwoDisconnectedComponents) {
    G g(false);
    g.addEdge(1, 2); g.addEdge(3, 4);
    auto comps = connectedComponents(g);
    EXPECT_EQ(comps.size(), 2u);
}

TEST(GraphAlgorithmsTest, ThreeIsolatedVertices) {
    G g(false);
    g.addVertex(1); g.addVertex(2); g.addVertex(3);
    auto comps = connectedComponents(g);
    EXPECT_EQ(comps.size(), 3u);
}

TEST(GraphAlgorithmsTest, IsConnectedTrue) {
    G g(false);
    g.addEdge(1, 2); g.addEdge(2, 3); g.addEdge(3, 4);
    EXPECT_TRUE(isConnected(g));
}

TEST(GraphAlgorithmsTest, IsConnectedFalse) {
    G g(false);
    g.addEdge(1, 2); g.addVertex(3);
    EXPECT_FALSE(isConnected(g));
}

TEST(GraphAlgorithmsTest, IsConnectedEmptyGraphTrue) {
    G g;
    EXPECT_TRUE(isConnected(g));
}

// ═════════════════════════════════════════════════════════════════════════════
// Kosaraju SCC
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, KosarajuSingleSCC) {
    G g;
    g.addEdge(1, 2); g.addEdge(2, 3); g.addEdge(3, 1);  // complete cycle
    auto sccs = kosaraju(g);
    EXPECT_EQ(sccs.size(), 1u);
    auto norm = normalizeSCCs(sccs);
    EXPECT_EQ(norm[0], (std::set<int>{1, 2, 3}));
}

TEST(GraphAlgorithmsTest, KosarajuDAGEachNodeOwnSCC) {
    auto g = makeLinear(4);  // 1→2→3→4 DAG
    auto sccs = kosaraju(g);
    EXPECT_EQ(sccs.size(), 4u);  // each vertex is its own SCC
    for (const auto& scc : sccs) EXPECT_EQ(scc.size(), 1u);
}

TEST(GraphAlgorithamsTest_Kosaraju, TwoSCCs) {
    G g;
    // SCC1: {1,2,3}, SCC2: {4,5}
    g.addEdge(1, 2); g.addEdge(2, 3); g.addEdge(3, 1);
    g.addEdge(4, 5); g.addEdge(5, 4);
    g.addEdge(3, 4);  // bridge — not part of either SCC
    auto sccs  = kosaraju(g);
    auto norm  = normalizeSCCs(sccs);
    ASSERT_EQ(norm.size(), 2u);
    EXPECT_EQ(norm[0], (std::set<int>{1, 2, 3}));
    EXPECT_EQ(norm[1], (std::set<int>{4, 5}));
}

// ═════════════════════════════════════════════════════════════════════════════
// Tarjan SCC
// ═════════════════════════════════════════════════════════════════════════════

TEST(GraphAlgorithmsTest, TarjanSingleSCC) {
    G g;
    g.addEdge(1, 2); g.addEdge(2, 3); g.addEdge(3, 1);
    auto sccs = tarjanSCC(g);
    EXPECT_EQ(sccs.size(), 1u);
    EXPECT_EQ(normalizeSCCs(sccs)[0], (std::set<int>{1, 2, 3}));
}

TEST(GraphAlgorithmsTest, TarjanMatchesKosaraju) {
    G g;
    g.addEdge(1, 2); g.addEdge(2, 3); g.addEdge(3, 1);
    g.addEdge(4, 5); g.addEdge(5, 4);
    g.addEdge(3, 4);

    auto ks = normalizeSCCs(kosaraju(g));
    auto ts = normalizeSCCs(tarjanSCC(g));
    EXPECT_EQ(ks, ts);
}

TEST(GraphAlgorithmsTest, TarjanDAG) {
    auto g = makeLinear(5);
    auto sccs = tarjanSCC(g);
    EXPECT_EQ(sccs.size(), 5u);
}
