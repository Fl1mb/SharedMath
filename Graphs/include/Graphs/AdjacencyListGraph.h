#pragma once

#include <unordered_map>
#include <vector>
#include <optional>
#include <stdexcept>
#include <algorithm>

namespace SharedMath::Graphs {

// ─────────────────────────────────────────────────────────────────────────────
// AdjacencyListGraph<V, W>
//
// General-purpose weighted graph backed by an adjacency list.
//
//   V  — vertex identifier type (must be hashable, e.g. int, std::string)
//   W  — edge weight type (default: double)
//
// Supports both directed and undirected modes (set at construction time).
//
// For directed graphs:
//   addEdge(u, v, w)  →  single arc u → v
//   inDegree(v)       →  number of arcs arriving at v  (O(1))
//   reversed()        →  new graph with all arcs flipped
//
// For undirected graphs:
//   addEdge(u, v, w)  →  symmetric edge stored as both u→v and v→u
//   inDegree(v)       ==  outDegree(v)
//
// Quick-start:
//   AdjacencyListGraph<int> g;            // directed, double weights
//   g.addEdge(1, 2, 3.5);
//   for (auto& e : g.neighbors(1))
//       std::cout << e.to << " " << e.weight << "\n";
// ─────────────────────────────────────────────────────────────────────────────
template<typename V, typename W = double>
class AdjacencyListGraph {
public:
    // ── Types ─────────────────────────────────────────────────────────────
    struct Edge {
        V to;
        W weight;
    };

    using EdgeList = std::vector<Edge>;
    using AdjMap   = std::unordered_map<V, EdgeList>;

    // ── Construction ──────────────────────────────────────────────────────
    explicit AdjacencyListGraph(bool directed = true)
        : directed_(directed) {}

    // ── Vertex operations ─────────────────────────────────────────────────

    // Add a vertex with no edges. Idempotent.
    void addVertex(const V& v) { ensureVertex(v); }

    // Remove a vertex and all incident edges.
    void removeVertex(const V& v) {
        if (!adj_.count(v)) return;

        // Directed: decrement inDeg_ of all vertices v points to
        if (directed_)
            for (const auto& e : adj_[v])
                if (inDeg_.count(e.to)) --inDeg_[e.to];

        // Remove all edges from other vertices that point to v
        for (auto& [u, edges] : adj_) {
            if (u == v) continue;
            edges.erase(
                std::remove_if(edges.begin(), edges.end(),
                               [&](const Edge& e) { return e.to == v; }),
                edges.end());
        }

        adj_.erase(v);
        if (directed_) inDeg_.erase(v);
    }

    bool hasVertex(const V& v) const { return adj_.count(v) > 0; }

    // ── Edge operations ───────────────────────────────────────────────────

    // Add a weighted edge. For undirected graphs the reverse arc is added too.
    // Duplicate edges ARE allowed (use hasEdge() to guard if needed).
    void addEdge(const V& from, const V& to, W w = W{1}) {
        ensureVertex(from);
        ensureVertex(to);
        adj_[from].push_back({to, w});
        if (directed_) {
            ++inDeg_[to];
        } else {
            // Undirected: add reverse arc only if not already present
            auto& back = adj_[to];
            bool found = std::any_of(back.begin(), back.end(),
                                     [&](const Edge& e) { return e.to == from; });
            if (!found) back.push_back({from, w});
        }
    }

    // Remove the first edge from→to (does nothing if not found).
    void removeEdge(const V& from, const V& to) {
        auto it = adj_.find(from);
        if (it == adj_.end()) return;

        auto& edges = it->second;
        auto  pos   = std::find_if(edges.begin(), edges.end(),
                                   [&](const Edge& e) { return e.to == to; });
        if (pos != edges.end()) {
            edges.erase(pos);
            if (directed_ && inDeg_.count(to)) --inDeg_[to];
        }

        if (!directed_) {
            auto& back = adj_[to];
            auto  pos2 = std::find_if(back.begin(), back.end(),
                                      [&](const Edge& e) { return e.to == from; });
            if (pos2 != back.end()) back.erase(pos2);
        }
    }

    bool hasEdge(const V& from, const V& to) const {
        auto it = adj_.find(from);
        if (it == adj_.end()) return false;
        return std::any_of(it->second.begin(), it->second.end(),
                           [&](const Edge& e) { return e.to == to; });
    }

    // Returns the weight of the first matching edge, or std::nullopt.
    std::optional<W> weight(const V& from, const V& to) const {
        auto it = adj_.find(from);
        if (it == adj_.end()) return std::nullopt;
        for (const auto& e : it->second)
            if (e.to == to) return e.weight;
        return std::nullopt;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    const EdgeList& neighbors(const V& v) const {
        auto it = adj_.find(v);
        if (it == adj_.end())
            throw std::out_of_range("AdjacencyListGraph: vertex not found");
        return it->second;
    }

    // Returns all vertex IDs (order not guaranteed for unordered_map).
    std::vector<V> vertices() const {
        std::vector<V> vs;
        vs.reserve(adj_.size());
        for (const auto& [v, _] : adj_) vs.push_back(v);
        return vs;
    }

    size_t vertexCount() const noexcept { return adj_.size(); }

    size_t edgeCount() const noexcept {
        size_t total = 0;
        for (const auto& [_, edges] : adj_) total += edges.size();
        return directed_ ? total : total / 2;
    }

    size_t outDegree(const V& v) const {
        auto it = adj_.find(v);
        return it == adj_.end() ? 0 : it->second.size();
    }

    // For undirected graphs this equals outDegree (O(1)).
    size_t inDegree(const V& v) const {
        if (!directed_) return outDegree(v);
        auto it = inDeg_.find(v);
        return it == inDeg_.end() ? 0 : it->second;
    }

    bool isDirected() const noexcept { return directed_; }

    // ── Graph transformations ─────────────────────────────────────────────

    // Returns a new directed graph with every arc reversed (no-op for undirected).
    AdjacencyListGraph reversed() const {
        if (!directed_) return *this;
        AdjacencyListGraph rev(true);
        for (const auto& [v, _] : adj_) rev.addVertex(v);
        for (const auto& [u, edges] : adj_)
            for (const auto& e : edges)
                rev.addEdge(e.to, u, e.weight);
        return rev;
    }

    // ── Raw access (for algorithms) ───────────────────────────────────────
    const AdjMap&                            adjacency()  const noexcept { return adj_;    }
    const std::unordered_map<V, size_t>&     inDegrees()  const noexcept { return inDeg_;  }

private:
    bool   directed_;
    AdjMap adj_;
    std::unordered_map<V, size_t> inDeg_;  // maintained only for directed graphs

    void ensureVertex(const V& v) {
        if (!adj_.count(v)) {
            adj_.emplace(v, EdgeList{});
            if (directed_) inDeg_.emplace(v, 0);
        }
    }
};

} // namespace SharedMath::Graphs
