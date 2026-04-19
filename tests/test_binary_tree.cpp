#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <algorithm>
#include "Graphs/BinaryTree.h"

using namespace SharedMath::Graphs;

// ============================================
// Тесты для вставки (Insert)
// ============================================

TEST(BinaryTreeTest, InsertIntoEmptyTree) {
    BinaryTree<int> tree;
    tree.insert(42);
    
    ASSERT_FALSE(tree.isEmpty());
    ASSERT_EQ(tree.size(), 1);
    ASSERT_EQ(tree.getRoot()->data, 42);
}

TEST(BinaryTreeTest, InsertMultipleValues) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    tree.insert(20);
    tree.insert(40);
    tree.insert(60);
    tree.insert(80);
    
    ASSERT_EQ(tree.size(), 7);
    ASSERT_EQ(tree.getRoot()->data, 50);
    ASSERT_EQ(tree.getRoot()->left->data, 30);
    ASSERT_EQ(tree.getRoot()->right->data, 70);
    ASSERT_EQ(tree.getRoot()->left->left->data, 20);
    ASSERT_EQ(tree.getRoot()->left->right->data, 40);
    ASSERT_EQ(tree.getRoot()->right->left->data, 60);
    ASSERT_EQ(tree.getRoot()->right->right->data, 80);
}

TEST(BinaryTreeTest, InsertDuplicateValues) {
    BinaryTree<int> tree;
    tree.insert(10);
    tree.insert(10); // Дубликат
    tree.insert(10); // Дубликат
    
    ASSERT_EQ(tree.size(), 1); // Дубликаты игнорируются
    ASSERT_EQ(tree.getRoot()->data, 10);
}

TEST(BinaryTreeTest, InsertDoubleValues) {
    BinaryTree<double> tree;
    
    tree.insert(4.0);
    tree.insert(8.0);
    tree.insert(1.0);
    tree.insert(9.0);
    tree.insert(8.0); // Дубликат
    
    ASSERT_EQ(tree.size(), 4);
    ASSERT_EQ(tree.getRoot()->data, 4.0);
    ASSERT_EQ(tree.getRoot()->left->data, 1.0);
    ASSERT_EQ(tree.getRoot()->right->data, 8.0);
    ASSERT_EQ(tree.getRoot()->right->right->data, 9.0);
}

TEST(BinaryTreeTest, InsertStringValues) {
    BinaryTree<std::string> tree;
    
    tree.insert("apple");
    tree.insert("banana");
    tree.insert("cherry");
    tree.insert("date");
    
    ASSERT_EQ(tree.size(), 4);
    ASSERT_EQ(tree.getRoot()->data, "apple");
    ASSERT_EQ(tree.getRoot()->right->data, "banana");
    ASSERT_EQ(tree.getRoot()->right->right->data, "cherry");
}

// ============================================
// Тесты для поиска (Search)
// ============================================

TEST(BinaryTreeTest, SearchInEmptyTree) {
    BinaryTree<int> tree;
    auto* result = tree.search(42);
    
    ASSERT_EQ(result, nullptr);
}

TEST(BinaryTreeTest, SearchExistingValue) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    
    auto* result = tree.search(30);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->data, 30);
}

TEST(BinaryTreeTest, SearchNonExistingValue) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    
    auto* result = tree.search(99);
    ASSERT_EQ(result, nullptr);
}

TEST(BinaryTreeTest, SearchRootValue) {
    BinaryTree<int> tree;
    tree.insert(42);
    
    auto* result = tree.search(42);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->data, 42);
    ASSERT_EQ(result, tree.getRoot());
}

// ============================================
// Тесты для удаления (Remove)
// ============================================

TEST(BinaryTreeTest, RemoveNonExistingValue) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    
    ASSERT_THROW(tree.remove(99), std::invalid_argument);
    ASSERT_EQ(tree.size(), 3);
}

TEST(BinaryTreeTest, RemoveLeafNode) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    tree.insert(20);
    tree.insert(40);
    
    ASSERT_EQ(tree.size(), 5);
    tree.remove(20); // Лист
    
    ASSERT_EQ(tree.size(), 4);
    ASSERT_EQ(tree.search(20), nullptr);
    ASSERT_EQ(tree.getRoot()->left->left, nullptr);
}

TEST(BinaryTreeTest, RemoveNodeWithOneChild) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    tree.insert(20);
    // Структура:
    //     50
    //    /  \
    //   30   70
    //  /
    // 20
    
    tree.remove(30); // Узел с одним левым потомком
    
    ASSERT_EQ(tree.size(), 3);
    ASSERT_EQ(tree.search(30), nullptr);
    ASSERT_EQ(tree.getRoot()->left->data, 20);
}

TEST(BinaryTreeTest, RemoveNodeWithTwoChildren) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    tree.insert(20);
    tree.insert(40);
    tree.insert(60);
    tree.insert(80);
    // Структура:
    //       50
    //     /    \
    //    30     70
    //   /  \   /  \
    //  20  40 60  80
    
    tree.remove(50); // Корень с двумя потомками
    
    ASSERT_EQ(tree.size(), 6);
    ASSERT_EQ(tree.search(50), nullptr);
    // Преемник должен быть 60 (минимум в правом поддереве)
    ASSERT_EQ(tree.getRoot()->data, 60);
    ASSERT_EQ(tree.getRoot()->right->left, nullptr); // 60 удален из правого поддерева
}

TEST(BinaryTreeTest, RemoveRootWithOneChild) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    // Структура:
    //   50
    //  /
    // 30
    
    tree.remove(50);
    
    ASSERT_EQ(tree.size(), 1);
    ASSERT_EQ(tree.getRoot()->data, 30);
}

TEST(BinaryTreeTest, RemoveRootAlone) {
    BinaryTree<int> tree;
    tree.insert(42);
    
    tree.remove(42);
    
    ASSERT_TRUE(tree.isEmpty());
    ASSERT_EQ(tree.size(), 0);
    ASSERT_EQ(tree.getRoot(), nullptr);
}

// ============================================
// Тесты для очистки (Clear)
// ============================================

TEST(BinaryTreeTest, ClearEmptyTree) {
    BinaryTree<int> tree;
    
    tree.clear();
    
    ASSERT_TRUE(tree.isEmpty());
    ASSERT_EQ(tree.size(), 0);
}

TEST(BinaryTreeTest, ClearNonEmptyTree) {
    BinaryTree<int> tree;
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    
    ASSERT_EQ(tree.size(), 3);
    tree.clear();
    
    ASSERT_TRUE(tree.isEmpty());
    ASSERT_EQ(tree.size(), 0);
    ASSERT_EQ(tree.getRoot(), nullptr);
}

TEST(BinaryTreeTest, ClearAndReuse) {
    BinaryTree<int> tree;
    
    tree.insert(10);
    tree.insert(20);
    tree.insert(30);
    ASSERT_EQ(tree.size(), 3);
    
    tree.clear();
    ASSERT_TRUE(tree.isEmpty());
    
    tree.insert(100);
    tree.insert(200);
    ASSERT_EQ(tree.size(), 2);
    ASSERT_EQ(tree.getRoot()->data, 100);
}

// ============================================
// Тесты для размера и высоты (Size & Height)
// ============================================

TEST(BinaryTreeTest, SizeOfEmptyTree) {
    BinaryTree<int> tree;
    
    ASSERT_EQ(tree.size(), 0);
}

TEST(BinaryTreeTest, SizeAfterOperations) {
    BinaryTree<int> tree;
    
    ASSERT_EQ(tree.size(), 0);
    
    tree.insert(50);
    ASSERT_EQ(tree.size(), 1);
    
    tree.insert(30);
    ASSERT_EQ(tree.size(), 2);
    
    tree.insert(70);
    ASSERT_EQ(tree.size(), 3);
    
    tree.remove(30);
    ASSERT_EQ(tree.size(), 2);
    
    tree.clear();
    ASSERT_EQ(tree.size(), 0);
}

TEST(BinaryTreeTest, HeightOfEmptyTree) {
    BinaryTree<int> tree;
    
    ASSERT_EQ(tree.height(), 0);
}

TEST(BinaryTreeTest, HeightOfSingleNodeTree) {
    BinaryTree<int> tree;
    tree.insert(42);
    
    ASSERT_EQ(tree.height(), 1);
}

TEST(BinaryTreeTest, HeightOfBalancedTree) {
    BinaryTree<int> tree;
    // Построим сбалансированное дерево высоты 3
    //       50
    //     /    \
    //    30     70
    //   /  \   /  \
    //  20  40 60  80
    
    tree.insert(50);
    tree.insert(30);
    tree.insert(70);
    tree.insert(20);
    tree.insert(40);
    tree.insert(60);
    tree.insert(80);
    
    ASSERT_EQ(tree.height(), 3);
}

TEST(BinaryTreeTest, HeightOfUnbalancedTree) {
    BinaryTree<int> tree;
    // Построим вырожденное дерево (связанный список)
    // 10
    //  \
    //   20
    //    \
    //     30
    //      \
    //       40
    
    tree.insert(10);
    tree.insert(20);
    tree.insert(30);
    tree.insert(40);
    
    ASSERT_EQ(tree.height(), 4);
}

// ============================================
// Тесты для проверки пустоты (isEmpty)
// ============================================

TEST(BinaryTreeTest, IsEmptyOnNewTree) {
    BinaryTree<int> tree;
    
    ASSERT_TRUE(tree.isEmpty());
}

TEST(BinaryTreeTest, IsEmptyAfterInsert) {
    BinaryTree<int> tree;
    
    tree.insert(42);
    ASSERT_FALSE(tree.isEmpty());
    
    tree.remove(42);
    ASSERT_TRUE(tree.isEmpty());
}

TEST(BinaryTreeTest, IsEmptyAfterClear) {
    BinaryTree<int> tree;
    
    tree.insert(10);
    tree.insert(20);
    tree.insert(30);
    ASSERT_FALSE(tree.isEmpty());
    
    tree.clear();
    ASSERT_TRUE(tree.isEmpty());
}

// ============================================
// Тесты для комплексных сценариев
// ============================================

TEST(BinaryTreeTest, InsertRemoveSequence) {
    BinaryTree<int> tree;
    
    // Вставляем значения
    for (int i = 0; i < 10; ++i) {
        tree.insert(i * 10);
    }
    ASSERT_EQ(tree.size(), 10);
    
    // Удаляем каждое второе значение
    for (int i = 0; i < 10; i += 2) {
        tree.remove(i * 10);
    }
    ASSERT_EQ(tree.size(), 5);
    
    // Проверяем оставшиеся значения
    for (int i = 1; i < 10; i += 2) {
        ASSERT_NE(tree.search(i * 10), nullptr);
    }
}

// ============================================
// Тесты для конструкторов и деструкторов
// ============================================

TEST(BinaryTreeTest, ConstructorWithValue) {
    BinaryTree<int> tree(42);
    
    ASSERT_FALSE(tree.isEmpty());
    ASSERT_EQ(tree.size(), 1);
    ASSERT_EQ(tree.getRoot()->data, 42);
}

TEST(BinaryTreeTest, DestructorCleansUp) {
    // Этот тест проверяет, что деструктор корректно освобождает память
    // В реальности может понадобиться инструмент для обнаружения утечек памяти
    auto* tree = new BinaryTree<int>();
    
    for (int i = 0; i < 100; ++i) {
        tree->insert(i);
    }
    
    delete tree; // Деструктор должен очистить всю память
    SUCCEED(); // Если дошли сюда без краша - вероятно, все хорошо
}
