// GraphsAlg.tpp — implementations for functions declared in GraphsAlg.h.
// Included at the end of GraphsAlg.h; do NOT include directly.

#pragma once

namespace SharedMath::Graphs {

// ── findFirst ────────────────────────────────────────────────────────────────
// BFS-based search. Returns the first node satisfying predicate, or nullptr.
template<typename T>
GraphNode<T>* findFirst(
    GraphNode<T>* start,
    std::function<bool(GraphNode<T>*)> predicate)
{
    if (!start || !predicate) return nullptr;

    std::queue<GraphNode<T>*> q;
    std::unordered_set<GraphNode<T>*> visited;

    q.push(start);
    visited.insert(start);

    while (!q.empty()) {
        auto* cur = q.front(); q.pop();
        if (predicate(cur)) return cur;

        for (auto* child : cur->childs) {
            if (child && !visited.count(child)) {
                visited.insert(child);
                q.push(child);
            }
        }
    }
    return nullptr;
}

// ── dfsRecursive ─────────────────────────────────────────────────────────────
// Pre-order and/or post-order callbacks; handles cycles via visited set.
template<typename T>
void dfsRecursive(
    GraphNode<T>* start,
    std::function<void(GraphNode<T>*)> preOrder,
    std::function<void(GraphNode<T>*)> postOrder)
{
    if (!start) return;

    std::unordered_set<GraphNode<T>*> visited;

    std::function<void(GraphNode<T>*)> impl = [&](GraphNode<T>* node) {
        if (!node || visited.count(node)) return;
        visited.insert(node);

        if (preOrder)  preOrder(node);
        for (auto* child : node->childs) impl(child);
        if (postOrder) postOrder(node);
    };

    impl(start);
}

// ── dfsIterative ─────────────────────────────────────────────────────────────
// Iterative DFS. onVisit returns false to stop traversal early.
template<typename T>
void dfsIterative(
    GraphNode<T>* start,
    std::function<bool(GraphNode<T>*)> onVisit)
{
    if (!start || !onVisit) return;

    std::stack<GraphNode<T>*>          stk;
    std::unordered_set<GraphNode<T>*>  visited;

    stk.push(start);

    while (!stk.empty()) {
        auto* cur = stk.top(); stk.pop();
        if (!cur || visited.count(cur)) continue;

        visited.insert(cur);
        if (!onVisit(cur)) return;  // early exit

        // Push in reverse so left-most child is processed first
        const auto& ch = cur->childs;
        for (auto it = ch.rbegin(); it != ch.rend(); ++it)
            if (*it && !visited.count(*it))
                stk.push(*it);
    }
}

// ── topologicalSort ───────────────────────────────────────────────────────────
// Kahn's algorithm on a collection of GraphNode<T>* forming a DAG.
// If the graph has a cycle, the result will be shorter than allNodes.size().
template<typename T>
std::vector<GraphNode<T>*> topologicalSort(
    const std::vector<GraphNode<T>*>& allNodes)
{
    // Count in-degrees
    std::unordered_map<GraphNode<T>*, size_t> inDeg;
    for (auto* n : allNodes) {
        inDeg.emplace(n, 0);  // ensure every node has an entry
    }
    for (auto* n : allNodes) {
        if (!n) continue;
        for (auto* child : n->childs)
            if (child) ++inDeg[child];
    }

    std::queue<GraphNode<T>*> q;
    for (auto& [node, deg] : inDeg)
        if (deg == 0) q.push(node);

    std::vector<GraphNode<T>*> result;
    result.reserve(allNodes.size());

    while (!q.empty()) {
        auto* cur = q.front(); q.pop();
        result.push_back(cur);
        for (auto* child : cur->childs)
            if (child && --inDeg[child] == 0) q.push(child);
    }

    return result;  // size < allNodes.size() ⟹ cycle exists
}

} // namespace SharedMath::Graphs
