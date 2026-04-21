#pragma once

#include <unordered_map>
#include <stdexcept>

namespace SharedMath::Graphs {

// ─────────────────────────────────────────────────────────────────────────────
// UnionFind<T>  —  Disjoint Set Union
//
// Supports any hashable type T (int, string, etc.).
// Uses path compression + union by rank for near-O(1) amortised operations.
//
// Typical usage (Kruskal's MST):
//   UnionFind<int> uf;
//   for (int v : vertices) uf.makeSet(v);
//   for (auto& [w, u, v] : sortedEdges)
//       if (uf.unite(u, v)) mst.push_back({u, v, w});
// ─────────────────────────────────────────────────────────────────────────────
template<typename T>
class UnionFind {
public:
    // Register x as a new singleton set. Idempotent: safe to call multiple times.
    void makeSet(const T& x) {
        if (!parent_.count(x)) {
            parent_.emplace(x, x);
            rank_.emplace(x, 0);
            ++numSets_;
        }
    }

    // Find the representative (root) of x's set. Applies path compression.
    // Throws std::out_of_range if x was never registered via makeSet().
    T find(const T& x) {
        auto it = parent_.find(x);
        if (it == parent_.end())
            throw std::out_of_range("UnionFind::find: element not registered");
        if (it->second != x)
            it->second = find(it->second);  // path compression
        return it->second;
    }

    // Merge the sets containing x and y.
    // Returns true if they were in different sets (a merge actually happened).
    bool unite(const T& x, const T& y) {
        T rx = find(x), ry = find(y);
        if (rx == ry) return false;

        // Union by rank — attach smaller tree under larger
        size_t rankX = rank_[rx], rankY = rank_[ry];
        if (rankX < rankY) std::swap(rx, ry);
        parent_[ry] = rx;
        if (rankX == rankY) ++rank_[rx];
        --numSets_;
        return true;
    }

    // Returns true if x and y belong to the same set.
    bool connected(const T& x, const T& y) { return find(x) == find(y); }

    // Number of distinct sets currently tracked.
    size_t numSets()  const noexcept { return numSets_; }

    // Total number of registered elements.
    size_t size()     const noexcept { return parent_.size(); }

    bool   contains(const T& x) const noexcept { return parent_.count(x) > 0; }

    // Reset to empty state.
    void clear() { parent_.clear(); rank_.clear(); numSets_ = 0; }

private:
    std::unordered_map<T, T>      parent_;
    std::unordered_map<T, size_t> rank_;
    size_t                        numSets_ = 0;
};

} // namespace SharedMath::Graphs
