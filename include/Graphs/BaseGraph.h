#pragma once
#include <unordered_map>
#include <vector>
#include <stack>
#include <queue>
#include <stdexcept>


namespace SharedMath::Graphs
{
    template<typename T>
    struct GraphNode{
        GraphNode(T Data) : 
            data(Data), 
            childs(std::vector<GraphNode*>(nullptr)),
            parent(nullptr),
            isRoot(false)
            {}
        ~GraphNode() = default;

        std::vector<GraphNode*> childs;
        T data;
        GraphNode* parent {nullptr};
        bool isRoot {nullptr};
    };

    template<typename T>
    class BaseGraph{
    public:
        BaseGraph(T rootData);
        virtual ~BaseGraph();

        GraphNode<T>* getRoot() const {return root};
        GraphNode<T>* findNode(T data) const;

        void clear();

        static std::vector<GraphNode<T>*> getAllChilds(const GraphNode<T>* root);
        static void addChild(GraphNode<T>* parent, T childData);
        static void deleteSubtree(GraphNode<T>* node);

    protected:
        GraphNode<T>* root;

    };

    template<typename T>
    inline BaseGraph<T>::BaseGraph(T rootData){
        this->root = new T();
        root->childs = std::vector<GraphNode<T>*>(nullptr);
        root->data = rootData;
        root->parent = nullptr;
        root->isRoot = true;
    }

    template<typename T> 
    inline BaseGraph<T>::~BaseGraph(){
        clear();
    }

    template<typename T>
    inline void BaseGraph<T>::clear(){
        std::queue<GraphNode<T>*> queue;

        queue.push(root);

        while(!queue.empty()){
            auto* node = queue.front();
            queue.pop();
            auto childs = getAllChilds(node);
            
            for(const auto* child : childs){
                if(child)
                    queue.push(child);
            }

            delete node;
        }
    }

    template<typename T>
    inline std::vector<GraphNode<T>*> BaseGraph<T>::getAllChilds(const GraphNode<T>* node){
        if(!node)return {};
        return node->childs;
    }

    template<typename T>
    inline void BaseGraph<T>::addChild(GraphNode<T>* parent, T childData){
        if(!parent){
            throw std::invalid_argument("Parent pointer is nullptr");
        }

        GraphNode<T>* node = new GraphNode<T>(childData);
        node->parent = parent;
        parent->childs.push_back(node);
    }

    template<typename T>
    inline GraphNode<T>* BaseGraph<T>::findNode(T data) const{
        std::queue<GraphNode<T>*> nodesQueue;
        nodesQueue.push(root);

        while(!nodesQueue.empty()){
            auto* node = nodesQueue.front();
            nodesQueue.pop();

            if(node){
                if(node->data == data)return node;

                auto childNodes = node->childs;
                for(const auto* childNode : childNodes){
                    if(!childNode)
                        nodesQueue.push(childNode);
                }
            }
        }
        return nullptr;
    }

    template<typename T>
    inline void BaseGraph<T>::deleteSubtree(GraphNode<T>* node){
        std::stack<GraphNode<T>*> stack;
        stack.push(node);

        while(!stack.empty()){
            auto* currentNode = stack.top();
            stack.pop();

            if(currentNode){
                for(const auto* child : currentNode->childs){
                    if(child){
                        stack.push(child);
                    }
                }
                delete currentNode;
            }
        }
    }
} // namespace SharedMath::Graphs
