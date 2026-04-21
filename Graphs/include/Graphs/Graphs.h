#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  SharedMath :: Graphs  —  umbrella header
//
//  Include this single header to pull in the entire Graphs module.
//
//  Contents:
//
//  Tree structures
//    BaseGraph<T>           – generic N-ary tree with raw-pointer nodes
//    BinaryTree<T>          – binary search tree (insert/remove/search)
//    AVLTree<T, Compare>    – self-balancing BST; O(log n) all operations
//
//  Graph representations
//    AdjacencyListGraph<V,W> – weighted directed/undirected graph
//                              (adjacency list, supports Dijkstra etc.)
//
//  Utilities
//    UnionFind<T>           – disjoint-set union (path compression + rank)
//
//  Tree traversal (on BaseGraph nodes)
//    bfs(start, target)                    – path via BFS
//    bfs(start, onVisit)                   – BFS with callback
//    dfsRecursive(start, preOrder, postOrder)
//    dfsIterative(start, onVisit)          – early-exit DFS
//    findFirst(start, predicate)           – BFS-based predicate search
//    topologicalSort(allNodes)             – Kahn's on GraphNode DAG
//
//  Graph algorithms (on AdjacencyListGraph)
//    bfsTraversal(g, start)                – visit order
//    dfsTraversal(g, start, onVisit)       – iterative DFS with callback
//    bfsDistances(g, src)                  – hop-count distances
//    dijkstra(g, src)                      – SSSP, non-negative weights O((V+E)logV)
//    bellmanFord(g, src)                   – SSSP, handles negatives O(VE)
//    floydWarshall(g)                      – all-pairs SP O(V³)
//    prim(g, start)                        – MST O(E log V)
//    kruskal(g)                            – MST O(E log E)
//    topologicalSort(g)                    – Kahn's on AdjacencyListGraph O(V+E)
//    hasCycle(g)                           – cycle detection O(V+E)
//    connectedComponents(g)                – undirected CC O(V+E)
//    isConnected(g)                        – connectivity check O(V+E)
//    kosaraju(g)                           – SCC O(V+E)
//    tarjanSCC(g)                          – SCC (single pass) O(V+E)
// ─────────────────────────────────────────────────────────────────────────────

#include "BaseGraph.h"
#include "BinaryTree.h"
#include "AVLTree.h"
#include "GraphsAlg.h"       // also pulls in GraphsAlg.tpp
#include "UnionFind.h"
#include "AdjacencyListGraph.h"
#include "GraphAlgorithms.h"
