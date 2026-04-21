#pragma once

#include <functional>
#include <optional>
#include <vector>
#include <memory>
#include <stdexcept>

namespace SharedMath::Graphs {

// ─────────────────────────────────────────────────────────────────────────────
// AVLTree<T, Compare>  —  Self-balancing Binary Search Tree
//
// Guarantees O(log n) insert / remove / search at all times.
// Duplicates are silently ignored (set semantics).
// Memory is managed by std::unique_ptr — no manual cleanup needed.
//
// Quick-start:
//   AVLTree<int> tree;
//   tree.insert(5); tree.insert(2); tree.insert(8);
//   tree.remove(2);
//   auto sorted = tree.inorder();   // {5, 8}
//   std::cout << tree.height();     // 2
// ─────────────────────────────────────────────────────────────────────────────
template<typename T, typename Compare = std::less<T>>
class AVLTree {
    // ── Internal node ─────────────────────────────────────────────────────
    struct Node {
        T data;
        int height = 1;
        std::unique_ptr<Node> left, right;

        explicit Node(T val) : data(std::move(val)) {}
    };

    // ── Helpers ───────────────────────────────────────────────────────────
    static int  nodeHeight(const Node* n) noexcept { return n ? n->height : 0; }
    static int  bf(const Node* n) noexcept {
        return n ? nodeHeight(n->left.get()) - nodeHeight(n->right.get()) : 0;
    }
    static void updateHeight(Node* n) noexcept {
        if (n)
            n->height = 1 + std::max(nodeHeight(n->left.get()),
                                     nodeHeight(n->right.get()));
    }

    // Right rotation (LL imbalance)
    static std::unique_ptr<Node> rotateRight(std::unique_ptr<Node> y) {
        auto x    = std::move(y->left);
        y->left   = std::move(x->right);
        updateHeight(y.get());
        x->right  = std::move(y);
        updateHeight(x.get());
        return x;
    }

    // Left rotation (RR imbalance)
    static std::unique_ptr<Node> rotateLeft(std::unique_ptr<Node> x) {
        auto y    = std::move(x->right);
        x->right  = std::move(y->left);
        updateHeight(x.get());
        y->left   = std::move(x);
        updateHeight(y.get());
        return y;
    }

    // Rebalance after insert/delete
    static std::unique_ptr<Node> balance(std::unique_ptr<Node> n) {
        updateHeight(n.get());
        int b = bf(n.get());

        if (b > 1) {                          // Left heavy
            if (bf(n->left.get()) < 0)        // LR case
                n->left = rotateLeft(std::move(n->left));
            return rotateRight(std::move(n));
        }
        if (b < -1) {                         // Right heavy
            if (bf(n->right.get()) > 0)       // RL case
                n->right = rotateRight(std::move(n->right));
            return rotateLeft(std::move(n));
        }
        return n;
    }

    // ── Recursive operations ──────────────────────────────────────────────
    std::unique_ptr<Node> insert(std::unique_ptr<Node> n, T val) {
        if (!n) { ++size_; return std::make_unique<Node>(std::move(val)); }

        if (cmp_(val, n->data))
            n->left  = insert(std::move(n->left),  std::move(val));
        else if (cmp_(n->data, val))
            n->right = insert(std::move(n->right), std::move(val));
        // else: duplicate — ignore

        return balance(std::move(n));
    }

    static const Node* findMin(const Node* n) noexcept {
        while (n->left) n = n->left.get();
        return n;
    }
    static const Node* findMax(const Node* n) noexcept {
        while (n->right) n = n->right.get();
        return n;
    }

    // Remove the minimum node in subtree n, return updated subtree
    static std::unique_ptr<Node> removeMin(std::unique_ptr<Node> n) {
        if (!n->left) return std::move(n->right);
        n->left = removeMin(std::move(n->left));
        return balance(std::move(n));
    }

    std::unique_ptr<Node> remove(std::unique_ptr<Node> n, const T& val) {
        if (!n) return nullptr;  // not found — no-op

        if (cmp_(val, n->data)) {
            n->left  = remove(std::move(n->left),  val);
        } else if (cmp_(n->data, val)) {
            n->right = remove(std::move(n->right), val);
        } else {
            // Found — remove this node
            --size_;
            if (!n->right) return std::move(n->left);   // no right child
            if (!n->left)  return std::move(n->right);  // no left child

            // Two children: replace with in-order successor (min of right)
            const Node* succ = findMin(n->right.get());
            n->data  = succ->data;
            n->right = removeMin(std::move(n->right));
        }
        return balance(std::move(n));
    }

    const Node* find(const Node* n, const T& val) const noexcept {
        while (n) {
            if      (cmp_(val,    n->data)) n = n->left.get();
            else if (cmp_(n->data, val))    n = n->right.get();
            else                            return n;
        }
        return nullptr;
    }

    void collectInorder(const Node* n, std::vector<T>& out) const {
        if (!n) return;
        collectInorder(n->left.get(), out);
        out.push_back(n->data);
        collectInorder(n->right.get(), out);
    }

    // ── Members ───────────────────────────────────────────────────────────
    std::unique_ptr<Node> root_;
    Compare               cmp_;
    size_t                size_ = 0;

public:
    AVLTree() = default;
    explicit AVLTree(Compare cmp) : cmp_(std::move(cmp)) {}

    // Disable copy — use clone() for explicit copies
    AVLTree(const AVLTree&)            = delete;
    AVLTree& operator=(const AVLTree&) = delete;

    AVLTree(AVLTree&&)            noexcept = default;
    AVLTree& operator=(AVLTree&&) noexcept = default;

    // ── Modification ──────────────────────────────────────────────────────

    // Insert val. Duplicates are ignored (set semantics).
    void insert(T val) { root_ = insert(std::move(root_), std::move(val)); }

    // Remove val. No-op if not present.
    void remove(const T& val) { root_ = remove(std::move(root_), val); }

    // Remove all elements.
    void clear() noexcept { root_.reset(); size_ = 0; }

    // ── Queries ───────────────────────────────────────────────────────────

    bool contains(const T& val) const noexcept { return find(root_.get(), val) != nullptr; }

    size_t size()   const noexcept { return size_; }
    bool   empty()  const noexcept { return size_ == 0; }
    int    height() const noexcept { return nodeHeight(root_.get()); }

    // Minimum / maximum element (std::nullopt if empty).
    std::optional<T> min() const {
        if (!root_) return std::nullopt;
        return findMin(root_.get())->data;
    }
    std::optional<T> max() const {
        if (!root_) return std::nullopt;
        return findMax(root_.get())->data;
    }

    // In-order traversal — returns elements in sorted order. O(n).
    std::vector<T> inorder() const {
        std::vector<T> out;
        out.reserve(size_);
        collectInorder(root_.get(), out);
        return out;
    }

    // In-order callback traversal — avoids building a vector.
    void forEach(std::function<void(const T&)> fn) const {
        std::function<void(const Node*)> impl = [&](const Node* n) {
            if (!n) return;
            impl(n->left.get());
            fn(n->data);
            impl(n->right.get());
        };
        impl(root_.get());
    }

    // Balance factor of the root node (useful for debugging).
    int rootBalanceFactor() const noexcept { return bf(root_.get()); }
};

} // namespace SharedMath::Graphs
