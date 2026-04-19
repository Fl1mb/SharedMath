#pragma once

#include <functional>
#include <stack>
#include <queue>
#include <stdexcept>

namespace SharedMath::Graphs
{
    template<typename T>
    struct BinaryTreeNode{
        T data;
        BinaryTreeNode* right;
        BinaryTreeNode* left;

        BinaryTreeNode(T value) : 
            data(value),
            right(nullptr),
            left(nullptr)
        {}
    };

    template<typename T>
    class BinaryTree{
    public:
        BinaryTree() : root(nullptr){}
        BinaryTree(T rootValue): root(new BinaryTreeNode<T>(rootValue)) {} 
        
        ~BinaryTree() { clear(); } 

        BinaryTree(const BinaryTree&) = delete;
        BinaryTree& operator=(const BinaryTree&) = delete;

        void clear();
        void remove(T value);
        void insert(T value);
        BinaryTreeNode<T>* search(T value) const;

        bool isEmpty() const {return root == nullptr;}
        BinaryTreeNode<T>* getRoot() const {return root;}
        size_t size() const {return countNodes(root);}
        size_t height() const {return calculateHeight(root);}

    private:
        BinaryTreeNode<T>* root;

        size_t countNodes(BinaryTreeNode<T>* node) const;
        size_t calculateHeight(BinaryTreeNode<T>* node) const;
        BinaryTreeNode<T>* findMin(BinaryTreeNode<T>* node) const;
        BinaryTreeNode<T>* findMax(BinaryTreeNode<T>* node) const;
        
        void removeNodeWithZeroOrOneChild(BinaryTreeNode<T>* node, BinaryTreeNode<T>* parent);
        void removeNodeWithTwoChildren(BinaryTreeNode<T>* node);
    };

    template<typename T>
    inline void BinaryTree<T>::clear(){
        if (!root) return;
        
        std::stack<BinaryTreeNode<T>*> stack;
        stack.push(root);

        while(!stack.empty()){
            BinaryTreeNode<T>* node = stack.top();
            stack.pop();

            if(node){
                if (node->right) stack.push(node->right);
                if (node->left) stack.push(node->left);
                
                delete node;
            }
        }
        root = nullptr;
    }

    template<typename T>
    inline void BinaryTree<T>::insert(T value){
        if (!root) {
            root = new BinaryTreeNode<T>(value);
            return;
        }

        BinaryTreeNode<T>* current = root;
        BinaryTreeNode<T>* parent = nullptr;

        while (current) {
            parent = current;
            
            if (value < current->data) {
                current = current->left;
            } else if (value > current->data) {
                current = current->right;
            } else {
                return;
            }
        }

        BinaryTreeNode<T>* newNode = new BinaryTreeNode<T>(value);
        
        if (value < parent->data) {
            parent->left = newNode;
        } else {
            parent->right = newNode;
        }
    }

    template<typename T>
    inline BinaryTreeNode<T>* BinaryTree<T>::search(T value) const{
        BinaryTreeNode<T>* current = root;
        
        while (current) {
            if (current->data == value) {
                return current;
            } else if (value < current->data) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        
        return nullptr;
    }

    template<typename T>
    inline void BinaryTree<T>::remove(T value){
        if (!root) return;
        
        BinaryTreeNode<T>* current = root;
        BinaryTreeNode<T>* parent = nullptr;

        while (current && current->data != value) {
            parent = current;
            
            if (value < current->data) {
                current = current->left;
            } else {
                current = current->right;
            }
        }

        if (!current) {
            throw std::invalid_argument("Not found value in binary tree");
        }

        if (!current->left && !current->right) {
            if (!parent) {
                delete root;
                root = nullptr;
            } else if (parent->left == current) {
                delete parent->left;
                parent->left = nullptr;
            } else {
                delete parent->right;
                parent->right = nullptr;
            }
        }
        else if (!current->left || !current->right) {
            BinaryTreeNode<T>* child = current->left ? current->left : current->right;
            
            if (!parent) {
                delete root;
                root = child;
            } else if (parent->left == current) {
                delete parent->left;
                parent->left = child;
            } else {
                delete parent->right;
                parent->right = child;
            }
        }
        else {
            BinaryTreeNode<T>* successorParent = current;
            BinaryTreeNode<T>* successor = current->right;
            
            while (successor->left) {
                successorParent = successor;
                successor = successor->left;
            }
            
            current->data = successor->data;
            
            if (successorParent->left == successor) {
                successorParent->left = successor->right;
            } else {
                successorParent->right = successor->right;
            }
            
            delete successor;
        }
    }

    template<typename T>
    inline BinaryTreeNode<T>* BinaryTree<T>::findMin(BinaryTreeNode<T>* node) const{
        if (!node) return nullptr;
        
        while (node->left) {
            node = node->left;
        }
        return node;
    }

    template<typename T>
    inline BinaryTreeNode<T>* BinaryTree<T>::findMax(BinaryTreeNode<T>* node) const{
        if (!node) return nullptr;
        
        while (node->right) {
            node = node->right;
        }
        return node;
    }

    template<typename T>
    inline size_t BinaryTree<T>::countNodes(BinaryTreeNode<T>* node) const{
        if (!node) return 0;
        
        size_t count = 0;
        std::stack<BinaryTreeNode<T>*> stack;
        stack.push(node);
        
        while (!stack.empty()) {
            BinaryTreeNode<T>* current = stack.top();
            stack.pop();
            
            if (current) {
                count++;
                if (current->right) stack.push(current->right);
                if (current->left) stack.push(current->left);
            }
        }
        
        return count;
    }

    template<typename T>
    inline size_t BinaryTree<T>::calculateHeight(BinaryTreeNode<T>* node) const{
        if (!node) return 0;
        
        size_t maxHeight = 0;
        std::queue<std::pair<BinaryTreeNode<T>*, size_t>> q;
        q.push({node, 1});
        
        while (!q.empty()) {
            auto [current, level] = q.front();
            q.pop();
            
            maxHeight = std::max(maxHeight, level);
            
            if (current->left) {
                q.push({current->left, level + 1});
            }
            if (current->right) {
                q.push({current->right, level + 1});
            }
        }
        
        return maxHeight;
    }
} // namespace SharedMath::Graphs