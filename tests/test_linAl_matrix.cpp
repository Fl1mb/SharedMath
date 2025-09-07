#include <gtest/gtest.h>
#include "../include/SharedMath.h"

using namespace SharedMath::LinearAlgebra;

TEST(MatrixViewTest, BasicCreation) {
    Matrix<4, 4> matrix;
    // Заполняем матрицу тестовыми данными
    double counter = 1.0;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            matrix[i][j] = counter++;
        }
    }
    
    // Создаем view на подматрицу 2x2
    MatrixView view(1, 3, 1, 3, &matrix);
    
    EXPECT_EQ(view.rows(), 2);
    EXPECT_EQ(view.cols(), 2);
}

TEST(MatrixViewTest, FullMatrixView) {
    Matrix<2, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    // View на всю матрицу
    MatrixView fullView(&matrix);
    
    EXPECT_EQ(fullView.rows(), 2);
    EXPECT_EQ(fullView.cols(), 3);
    
    // Проверка всех элементов
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(fullView.get(i, j), matrix[i][j]);
        }
    }
}

TEST(MatrixViewTest, ElementAccess) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    MatrixView view(0, 2, 1, 3, &matrix);
    
    // Проверка чтения элементов
    EXPECT_DOUBLE_EQ(view.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(view.get(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(view.get(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(view.get(1, 1), 6.0);
    
    // Проверка через оператор ()
    EXPECT_DOUBLE_EQ(view(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(view(0, 1), 3.0);
}

TEST(MatrixViewTest, ElementModification) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    MatrixView view(1, 3, 1, 3, &matrix);
    
    // Модификация через view
    view.set(0, 0, 100.0);
    view.set(0, 1, 200.0);
    view.set(1, 0, 300.0);
    view.set(1, 1, 400.0);
    
    // Проверка изменений в исходной матрице
    EXPECT_DOUBLE_EQ(matrix[1][1], 100.0);
    EXPECT_DOUBLE_EQ(matrix[1][2], 200.0);
    EXPECT_DOUBLE_EQ(matrix[2][1], 300.0);
    EXPECT_DOUBLE_EQ(matrix[2][2], 400.0);
    
    // Проверка изменений через view
    EXPECT_DOUBLE_EQ(view.get(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(view.get(0, 1), 200.0);
    EXPECT_DOUBLE_EQ(view.get(1, 0), 300.0);
    EXPECT_DOUBLE_EQ(view.get(1, 1), 400.0);
}

TEST(MatrixViewTest, SubViewCreation) {
    Matrix<4, 4> matrix;
    double counter = 1.0;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            matrix[i][j] = counter++;
        }
    }
    
    // Создаем основной view
    MatrixView mainView(0, 3, 0, 3, &matrix); // 3x3 view
    
    // Создаем subview внутри основного view
    MatrixView subView = mainView.subView(1, 3, 1, 3); // 2x2 subview
    
    EXPECT_EQ(subView.rows(), 2);
    EXPECT_EQ(subView.cols(), 2);
    
    // Проверка элементов subview
    EXPECT_DOUBLE_EQ(subView.get(0, 0), 6.0);  // matrix[1][1]
    EXPECT_DOUBLE_EQ(subView.get(0, 1), 7.0);  // matrix[1][2]
    EXPECT_DOUBLE_EQ(subView.get(1, 0), 10.0); // matrix[2][1]
    EXPECT_DOUBLE_EQ(subView.get(1, 1), 11.0); // matrix[2][2]
}

TEST(MatrixViewTest, ToMatrixConversionStatic) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    MatrixView view(0, 2, 1, 3, &matrix);
    auto converted = view.toMatrix<2, 2>();
    
    // Проверка преобразования
    EXPECT_DOUBLE_EQ((*converted)[0][0], 2.0);
    EXPECT_DOUBLE_EQ((*converted)[0][1], 3.0);
    EXPECT_DOUBLE_EQ((*converted)[1][0], 5.0);
    EXPECT_DOUBLE_EQ((*converted)[1][1], 6.0);
    
    // Проверка, что это копия, а не ссылка
    (*converted)[0][0] = 100.0;
    EXPECT_DOUBLE_EQ(view.get(0, 0), 2.0); // Исходный view не изменился
    EXPECT_DOUBLE_EQ((*converted)[0][0], 100.0); // Копия изменилась
}


TEST(MatrixViewTest, EdgeCaseSingleElement) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    // View на один элемент
    MatrixView singleView(1, 2, 1, 2, &matrix);
    
    EXPECT_EQ(singleView.rows(), 1);
    EXPECT_EQ(singleView.cols(), 1);
    EXPECT_DOUBLE_EQ(singleView.get(0, 0), 5.0);
}

TEST(MatrixViewTest, InvalidViewCreation) {
    Matrix<2, 2> matrix = {{1.0, 2.0}, {3.0, 4.0}};
    
    // Невалидные параметры
    EXPECT_THROW(MatrixView(0, 3, 0, 3, &matrix), std::invalid_argument); // Выход за границы
    EXPECT_THROW(MatrixView(1, 0, 0, 1, &matrix), std::invalid_argument); // Start > End
    EXPECT_THROW(MatrixView(0, 1, 1, 0, &matrix), std::invalid_argument); // Start > End
    EXPECT_THROW(MatrixView(0, 1, 0, 1, nullptr), std::invalid_argument); // Null matrix
}

TEST(MatrixViewTest, InvalidAccess) {
    Matrix<2, 2> matrix = {{1.0, 2.0}, {3.0, 4.0}};
    MatrixView view(0, 1, 0, 1, &matrix); // 1x1 view
    
    EXPECT_THROW(view.get(1, 0), std::out_of_range); // Выход за границы view
    EXPECT_THROW(view.get(0, 1), std::out_of_range); // Выход за границы view
    EXPECT_THROW(view.set(1, 0, 5.0), std::out_of_range); // Выход за границы view
}


TEST(MatrixViewTest, CopyAndMoveOperations) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    // Copy constructor
    MatrixView view1(0, 2, 0, 2, &matrix);
    MatrixView view2 = view1; // Copy
    
    EXPECT_DOUBLE_EQ(view2.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view2.get(1, 1), 5.0);
    
    // Move constructor
    MatrixView view3 = std::move(view1); // Move
    
    EXPECT_DOUBLE_EQ(view3.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view3.get(1, 1), 5.0);
}


TEST(MatrixViewTest, ConstMatrixAccess) {
    const Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    // Const view на const матрицу
    MatrixView view(0, 2, 0, 2, const_cast<AbstractMatrix*>(static_cast<const AbstractMatrix*>(&matrix)));
    
    // Можно читать
    EXPECT_DOUBLE_EQ(view.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view.get(1, 1), 5.0);
    
    // Но нельзя модифицировать через const матрицу
    // (это вызовет ошибку компиляции, если попытаться вызвать set на const матрице)
}