#pragma once

#include "AdjacencyListGraph.h"
#include "UnionFind.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <stack>
#include <optional>
#include <functional>
#include <algorithm>
#include <limits>
#include <tuple>
#include <stdexcept>
#include <type_traits>

namespace SharedMath::Graphs {

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace detail {

template<typename W>
W graphInf() noexcept {
    if constexpr (std::is_floating_point_v<W>)
        return std::numeric_limits<W>::infinity();
    else
        return std::numeric_limits<W>::max() / 2;
}

} // namespace detail


// ─────────────────────────────────────────────────────────────────────────────
//  Result types
// ─────────────────────────────────────────────────────────────────────────────

// Result of a single-source shortest-path algorithm (Dijkstra / Bellman-Ford).
template<typename V, typename W>
struct ShortestPathResult {
    // dist[v]  = shortest distance from source to v (INF if unreachable).
    std::unordered_map<V, W>                prev_opt_dist;  // distance
    std::unordered_map<V, W>                dist;
    std::unordered_map<V, std::optional<V>> prev;           // predecessor

    bool reachable(const V& v) const {
        auto it = dist.find(v);
        return it != dist.end() && it->second < detail::graphInf<W>();
    }

    // Reconstruct the path from source to target.
    // Returns {} if target is unreachable.
    std::vector<V> pathTo(const V& target) const {
        if (!reachable(target)) return {};
        std::vector<V> path;
        std::optional<V> cur = target;
        while (cur.has_value()) {
            path.push_back(*cur);
            auto it = prev.find(*cur);
            cur = (it != prev.end()) ? it->second : std::nullopt;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }
};

// Result of a Minimum Spanning Tree algorithm (Prim / Kruskal).
template<typename V, typename W>
struct MSTResult {
    std::vector<std::tuple<V, V, W>> edges;       // (from, to, weight)
    W                                totalWeight{};
};

// Result of Floyd-Warshall (all-pairs shortest paths).
template<typename V, typename W>
struct AllPairsResult {
    std::vector<V>              vertices;
    std::vector<std::vector<W>> dist;  // dist[i][j] = shortest distance
    std::vector<std::vector<int>> next; // next[i][j] = next hop index for path reconstruction

    size_t indexOf(const V& v) const {
        for (size_t i = 0; i < vertices.size(); ++i)
            if (vertices[i] == v) return i;
        throw std::out_of_range("AllPairsResult: vertex not found");
    }

    bool reachable(const V& from, const V& to) const {
        return dist[indexOf(from)][indexOf(to)] < detail::graphInf<W>();
    }

    // Reconstruct path from 'from' to 'to'. Returns {} if unreachable.
    std::vector<V> path(const V& from, const V& to) const {
        size_t i = indexOf(from), j = indexOf(to);
        if (next[i][j] < 0) return {};
        std::vector<V> p;
        size_t cur = i;
        while (cur != j) {
            p.push_back(vertices[cur]);
            cur = static_cast<size_t>(next[cur][j]);
        }
        p.push_back(vertices[j]);
        return p;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
//  BFS / DFS traversal
// ─────────────────────────────────────────────────────────────────────────────

// BFS from 'start'. Returns vertices in visit order. O(V+E).
template<typename V, typename W>
std::vector<V> bfsTraversal(const AdjacencyListGraph<V,W>& g, const V& start) {
    if (!g.hasVertex(start)) return {};

    std::vector<V>         order;
    std::unordered_set<V>  visited;
    std::queue<V>          q;

    q.push(start);
    visited.insert(start);

    while (!q.empty()) {
        V u = q.front(); q.pop();
        order.push_back(u);
        for (const auto& e : g.neighbors(u))
            if (!visited.count(e.to)) {
                visited.insert(e.to);
                q.push(e.to);
            }
    }
    return order;
}

// Iterative DFS from 'start'. Calls onVisit for each discovered vertex. O(V+E).
template<typename V, typename W>
void dfsTraversal(const AdjacencyListGraph<V,W>& g, const V& start,
                  std::function<void(const V&)> onVisit)
{
    if (!g.hasVertex(start) || !onVisit) return;

    std::unordered_set<V> visited;
    std::stack<V>         stk;

    stk.push(start);
    while (!stk.empty()) {
        V u = stk.top(); stk.pop();
        if (visited.count(u)) continue;
        visited.insert(u);
        onVisit(u);
        // Push neighbors in reverse so the first neighbor is visited first
        const auto& nbrs = g.neighbors(u);
        for (auto it = nbrs.rbegin(); it != nbrs.rend(); ++it)
            if (!visited.count(it->to)) stk.push(it->to);
    }
}

// BFS hop-count distances from 'src' (ignores edge weights). O(V+E).
template<typename V, typename W>
std::unordered_map<V, size_t> bfsDistances(const AdjacencyListGraph<V,W>& g, const V& src) {
    std::unordered_map<V, size_t> dist;
    if (!g.hasVertex(src)) return {};

    dist[src] = 0;
    std::queue<V> q;
    q.push(src);

    while (!q.empty()) {
        V u = q.front(); q.pop();
        for (const auto& e : g.neighbors(u))
            if (!dist.count(e.to)) {
                dist[e.to] = dist[u] + 1;
                q.push(e.to);
            }
    }
    return dist;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Dijkstra — single-source shortest paths
//  O((V + E) log V) — requires non-negative edge weights.
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
ShortestPathResult<V,W> dijkstra(const AdjacencyListGraph<V,W>& g, const V& src) {
    using Pair = std::pair<W, V>;
    const W INF = detail::graphInf<W>();

    ShortestPathResult<V,W> res;
    for (const auto& v : g.vertices()) {
        res.dist[v] = INF;
        res.prev[v] = std::nullopt;
    }
    res.dist[src] = W{};

    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pq;
    pq.push({W{}, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > res.dist[u]) continue;  // stale entry

        for (const auto& e : g.neighbors(u)) {
            W nd = d + e.weight;
            if (nd < res.dist[e.to]) {
                res.dist[e.to] = nd;
                res.prev[e.to] = u;
                pq.push({nd, e.to});
            }
        }
    }
    return res;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Bellman-Ford — single-source shortest paths
//  O(V·E) — handles negative weights.
//  Returns std::nullopt if a negative cycle reachable from src is detected.
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
std::optional<ShortestPathResult<V,W>> bellmanFord(
    const AdjacencyListGraph<V,W>& g, const V& src)
{
    const W INF = detail::graphInf<W>();

    ShortestPathResult<V,W> res;
    for (const auto& v : g.vertices()) {
        res.dist[v] = INF;
        res.prev[v] = std::nullopt;
    }
    res.dist[src] = W{};

    const size_t n = g.vertexCount();

    // Relax all edges V-1 times
    for (size_t iter = 0; iter < n - 1; ++iter) {
        bool updated = false;
        for (const auto& u : g.vertices()) {
            if (res.dist[u] == INF) continue;
            for (const auto& e : g.neighbors(u)) {
                W nd = res.dist[u] + e.weight;
                if (nd < res.dist[e.to]) {
                    res.dist[e.to] = nd;
                    res.prev[e.to] = u;
                    updated = true;
                }
            }
        }
        if (!updated) break;  // converged early
    }

    // V-th relaxation pass: if any distance decreases, there is a negative cycle
    for (const auto& u : g.vertices()) {
        if (res.dist[u] == INF) continue;
        for (const auto& e : g.neighbors(u))
            if (res.dist[u] + e.weight < res.dist[e.to])
                return std::nullopt;  // negative cycle detected
    }

    return res;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Floyd-Warshall — all-pairs shortest paths
//  O(V³) — handles negative weights.
//  Returns std::nullopt if a negative cycle is detected (negative diagonal).
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
std::optional<AllPairsResult<V,W>> floydWarshall(const AdjacencyListGraph<V,W>& g) {
    const W INF = detail::graphInf<W>();

    AllPairsResult<V,W> res;
    res.vertices = g.vertices();
    const size_t n = res.vertices.size();

    // Build index map
    std::unordered_map<V, size_t> idx;
    for (size_t i = 0; i < n; ++i) idx[res.vertices[i]] = i;

    // Initialise distance and next-hop matrices
    res.dist.assign(n, std::vector<W>(n, INF));
    res.next.assign(n, std::vector<int>(n, -1));

    for (size_t i = 0; i < n; ++i) res.dist[i][i] = W{};

    for (const auto& u : res.vertices) {
        size_t i = idx[u];
        for (const auto& e : g.neighbors(u)) {
            size_t j = idx[e.to];
            if (e.weight < res.dist[i][j]) {
                res.dist[i][j] = e.weight;
                res.next[i][j] = static_cast<int>(j);
            }
        }
    }

    // Main DP loop
    for (size_t k = 0; k < n; ++k)
        for (size_t i = 0; i < n; ++i) {
            if (res.dist[i][k] == INF) continue;
            for (size_t j = 0; j < n; ++j) {
                if (res.dist[k][j] == INF) continue;
                W nd = res.dist[i][k] + res.dist[k][j];
                if (nd < res.dist[i][j]) {
                    res.dist[i][j] = nd;
                    res.next[i][j] = res.next[i][k];
                }
            }
        }

    // Negative cycle check: negative value on the diagonal
    for (size_t i = 0; i < n; ++i)
        if (res.dist[i][i] < W{}) return std::nullopt;

    return res;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Prim's MST — O(E log V)
//  For undirected, connected graphs. Returns MST edges.
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
MSTResult<V,W> prim(const AdjacencyListGraph<V,W>& g, const V& start) {
    using Tri = std::tuple<W, V, V>;  // (weight, to, from)

    MSTResult<V,W>          mst;
    std::unordered_set<V>   inMST;
    std::priority_queue<Tri, std::vector<Tri>, std::greater<Tri>> pq;

    auto expand = [&](const V& u) {
        inMST.insert(u);
        for (const auto& e : g.neighbors(u))
            if (!inMST.count(e.to))
                pq.push({e.weight, e.to, u});
    };

    if (!g.hasVertex(start)) return mst;
    expand(start);

    while (!pq.empty()) {
        auto [w, to, from] = pq.top(); pq.pop();
        if (inMST.count(to)) continue;  // already in MST
        mst.edges.emplace_back(from, to, w);
        mst.totalWeight += w;
        expand(to);
    }
    return mst;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Kruskal's MST — O(E log E)
//  Works on both directed and undirected graphs; for undirected, each edge is
//  processed once. Returns MST (or minimum spanning forest if disconnected).
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
MSTResult<V,W> kruskal(const AdjacencyListGraph<V,W>& g) {
    using Edge = std::tuple<W, V, V>;

    MSTResult<V,W> mst;
    UnionFind<V>   uf;

    // Register all vertices
    for (const auto& v : g.vertices()) uf.makeSet(v);

    // Collect edges (for undirected: use index comparison to avoid duplicates)
    std::vector<V>              verts = g.vertices();
    std::unordered_map<V,size_t> vidx;
    for (size_t i = 0; i < verts.size(); ++i) vidx[verts[i]] = i;

    std::vector<Edge> edges;
    for (size_t ui = 0; ui < verts.size(); ++ui) {
        const V& u = verts[ui];
        for (const auto& e : g.neighbors(u)) {
            size_t vi = vidx[e.to];
            if (g.isDirected() || ui <= vi)
                edges.emplace_back(e.weight, u, e.to);
        }
    }

    std::sort(edges.begin(), edges.end());

    for (auto& [w, u, v] : edges) {
        if (uf.unite(u, v)) {
            mst.edges.emplace_back(u, v, w);
            mst.totalWeight += w;
        }
    }
    return mst;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Topological Sort — Kahn's algorithm O(V+E)
//  Directed graphs only.
//  Returns std::nullopt if the graph contains a cycle.
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
std::optional<std::vector<V>> topologicalSort(const AdjacencyListGraph<V,W>& g) {
    // Copy in-degree map (we'll modify it)
    std::unordered_map<V, size_t> inDeg;
    for (const auto& v : g.vertices())
        inDeg[v] = g.inDegree(v);

    std::queue<V> q;
    for (const auto& [v, d] : inDeg)
        if (d == 0) q.push(v);

    std::vector<V> order;
    order.reserve(g.vertexCount());

    while (!q.empty()) {
        V u = q.front(); q.pop();
        order.push_back(u);
        for (const auto& e : g.neighbors(u))
            if (--inDeg[e.to] == 0) q.push(e.to);
    }

    if (order.size() != g.vertexCount()) return std::nullopt;  // cycle
    return order;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Cycle Detection — O(V+E)
//
//  Directed:   DFS with three-colour marking (white / grey / black).
//  Undirected: DFS with parent tracking (assumes simple graph).
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
bool hasCycle(const AdjacencyListGraph<V,W>& g) {
    if (g.isDirected()) {
        // 0=white(unseen), 1=grey(in stack), 2=black(done)
        std::unordered_map<V,int> color;
        for (const auto& v : g.vertices()) color[v] = 0;

        std::function<bool(const V&)> dfs = [&](const V& u) -> bool {
            color[u] = 1;
            for (const auto& e : g.neighbors(u)) {
                if (color[e.to] == 1) return true;   // back edge → cycle
                if (color[e.to] == 0 && dfs(e.to))   return true;
            }
            color[u] = 2;
            return false;
        };

        for (const auto& v : g.vertices())
            if (color[v] == 0 && dfs(v)) return true;
        return false;

    } else {
        std::unordered_set<V> visited;

        std::function<bool(const V&, const V*)> dfs =
            [&](const V& u, const V* parent) -> bool {
                visited.insert(u);
                for (const auto& e : g.neighbors(u)) {
                    if (!visited.count(e.to)) {
                        if (dfs(e.to, &u)) return true;
                    } else if (!parent || e.to != *parent) {
                        return true;  // non-parent back edge → cycle
                    }
                }
                return false;
            };

        for (const auto& v : g.vertices())
            if (!visited.count(v) && dfs(v, nullptr)) return true;
        return false;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  Connected Components — O(V+E)
//  For undirected graphs. Returns a list of component vertex-sets.
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
std::vector<std::vector<V>> connectedComponents(const AdjacencyListGraph<V,W>& g) {
    std::unordered_set<V>      visited;
    std::vector<std::vector<V>> comps;

    for (const auto& start : g.vertices()) {
        if (visited.count(start)) continue;

        auto& comp = comps.emplace_back();
        std::queue<V> q;
        q.push(start);
        visited.insert(start);

        while (!q.empty()) {
            V u = q.front(); q.pop();
            comp.push_back(u);
            for (const auto& e : g.neighbors(u))
                if (!visited.count(e.to)) {
                    visited.insert(e.to);
                    q.push(e.to);
                }
        }
    }
    return comps;
}

// Returns true if all vertices are reachable from any start vertex. O(V+E).
template<typename V, typename W>
bool isConnected(const AdjacencyListGraph<V,W>& g) {
    if (g.vertexCount() == 0) return true;
    const auto verts = g.vertices();
    return bfsTraversal(g, verts.front()).size() == g.vertexCount();
}


// ─────────────────────────────────────────────────────────────────────────────
//  Kosaraju's SCC — Strongly Connected Components O(V+E)
//  For directed graphs.
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
std::vector<std::vector<V>> kosaraju(const AdjacencyListGraph<V,W>& g) {
    std::unordered_set<V> visited;
    std::stack<V>         finish;

    // Pass 1: DFS on original graph — record finish order
    std::function<void(const V&)> dfs1 = [&](const V& u) {
        visited.insert(u);
        for (const auto& e : g.neighbors(u))
            if (!visited.count(e.to)) dfs1(e.to);
        finish.push(u);
    };
    for (const auto& v : g.vertices())
        if (!visited.count(v)) dfs1(v);

    // Pass 2: DFS on reversed graph in reverse-finish order
    const AdjacencyListGraph<V,W> rev = g.reversed();
    visited.clear();
    std::vector<std::vector<V>> sccs;

    std::function<void(const V&, std::vector<V>&)> dfs2 =
        [&](const V& u, std::vector<V>& scc) {
            visited.insert(u);
            scc.push_back(u);
            for (const auto& e : rev.neighbors(u))
                if (!visited.count(e.to)) dfs2(e.to, scc);
        };

    while (!finish.empty()) {
        V v = finish.top(); finish.pop();
        if (!visited.count(v)) {
            auto& scc = sccs.emplace_back();
            dfs2(v, scc);
        }
    }
    return sccs;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Tarjan's SCC — alternative O(V+E) SCC algorithm (single DFS pass)
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W>
std::vector<std::vector<V>> tarjanSCC(const AdjacencyListGraph<V,W>& g) {
    std::unordered_map<V, int>  disc, low;
    std::unordered_map<V, bool> onStack;
    std::stack<V>               stk;
    std::vector<std::vector<V>> sccs;
    int timer = 0;

    std::function<void(const V&)> dfs = [&](const V& u) {
        disc[u] = low[u] = timer++;
        stk.push(u);
        onStack[u] = true;

        for (const auto& e : g.neighbors(u)) {
            if (!disc.count(e.to)) {
                dfs(e.to);
                low[u] = std::min(low[u], low[e.to]);
            } else if (onStack[e.to]) {
                low[u] = std::min(low[u], disc[e.to]);
            }
        }

        if (low[u] == disc[u]) {
            auto& scc = sccs.emplace_back();
            while (true) {
                V w = stk.top(); stk.pop();
                onStack[w] = false;
                scc.push_back(w);
                if (w == u) break;
            }
        }
    };

    for (const auto& v : g.vertices())
        if (!disc.count(v)) dfs(v);

    return sccs;
}

} // namespace SharedMath::Graphs
