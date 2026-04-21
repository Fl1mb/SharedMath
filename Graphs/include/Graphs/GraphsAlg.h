#ifndef SHARED_MATH_GRAPHS_ALGORITHMS
#define SHARED_MATH_GRAPHS_ALGORITHMS

#include "BaseGraph.h"

#include <functional>
#include <stack>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace SharedMath::Graphs
{

        
    // Returns the way to target by bfs algorithm
    template<typename T>
    std::vector<GraphNode<T>*> bfs(
        GraphNode<T>* start, 
        GraphNode<T>* target = nullptr)
    {
        if(!start)return {};
        if(target == start)return {start};

        std::queue<GraphNode<T>*> queue;
        std::unordered_set<GraphNode<T>*> visited;
        std::unordered_map<GraphNode<T>*, GraphNode<T>*> parent;

        queue.push(start);
        visited.insert(start);
        parent[start] = nullptr;

        while(!queue.empty()){

            auto* current = queue.front();
            queue.pop();

            if(target && current == target){
                std::vector<GraphNode<T>*> path;
                GraphNode<T>* node = target;

                while(node){
                    path.push_back(node);
                    node = parent[node];
                }
                std::reverse(path.begin(), path.end());
                return path;
            }


            for(auto* neighbor : current->childs){
                if(!neighbor)continue;

                if(visited.find(neighbor) == visited.end()){
                    visited.insert(neighbor);
                    parent[neighbor] = current;
                    queue.push(neighbor);
                }
            }
        }
        return {};
    }
    

    // Callback BFS
    template<typename T>
    void bfs(
        GraphNode<T>* start,
        std::function<void(GraphNode<T>*)> onVisit
    )
    {
        if(!start || !onVisit)return;


        std::queue<GraphNode<T>*> queue;
        std::unordered_set<GraphNode<T>*> visited;

        queue.push(start);
        visited.insert(start);
        
        while (!queue.empty()) {
            auto* current = queue.front();
            queue.pop();
            
            onVisit(current);
            
            for (auto* neighbor : current->childs) {
                if (neighbor && visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    queue.push(neighbor);
                }
            }
        }
    }

    // Returns first predicated node
    template<typename T>
    GraphNode<T>* findFirst(
        GraphNode<T>* start,
        std::function<bool(GraphNode<T>*)> predicate
    );

    // Recursive DFS with callback
    template<typename T>
    void dfsRecursive(
        GraphNode<T>* start,
        std::function<void(GraphNode<T>*)> preOreder = nullptr,
        std::function<void(GraphNode<T>*)> postOrder = nullptr
    );

    // DFS with iterative control
    template<typename T>
    void dfsIterative(
        GraphNode<T>* start,
        std::function<bool(GraphNode<T>*)> onVisit  // return false to stop
    );

    // Returns topological sorted vector from graph
    template<typename T>
    std::vector<GraphNode<T>*> topologicalSort(
        const std::vector<GraphNode<T>*>& allNodes
    );



} // namespace SharedMath::Graph


#include "GraphsAlg.tpp"


#endif // SHARED_MATH_GRAPHS_ALGORITHMS