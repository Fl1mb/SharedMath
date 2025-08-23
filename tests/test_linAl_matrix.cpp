#include <gtest/gtest.h>
#include "../include/SharedMath.h"

using namespace SharedMath::LinearAlgebra;

TEST(MatrixTest, MatrixCreation) {
    Matrix<2, 3> matrix;
    
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 3);
    
    // Проверка инициализации нулями
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(matrix[i][j], 0.0);
        }
    }
}

TEST(MatrixTest, MatrixVectorMultiplication) {
    Matrix<2, 3> matrix;
    matrix[0] = Vector<3>{1.0, 2.0, 3.0};
    matrix[1] = Vector<3>{4.0, 5.0, 6.0};
    
    Vector<3> vec{2.0, 3.0, 4.0};
    Vector<2> result = matrix * vec;
    
    EXPECT_DOUBLE_EQ(result[0], 20.0); // 1*2 + 2*3 + 3*4 = 20
    EXPECT_DOUBLE_EQ(result[1], 47.0); // 4*2 + 5*3 + 6*4 = 47
}

TEST(MatrixTest, MatrixScalarMultiplication) {
    Matrix<2, 2> matrix;
    matrix[0] = Vector<2>{1.0, 2.0};
    matrix[1] = Vector<2>{3.0, 4.0};
    
    Matrix<2, 2> result = matrix * 2.0;
    
    EXPECT_DOUBLE_EQ(result[0][0], 2.0);
    EXPECT_DOUBLE_EQ(result[0][1], 4.0);
    EXPECT_DOUBLE_EQ(result[1][0], 6.0);
    EXPECT_DOUBLE_EQ(result[1][1], 8.0);
}

TEST(MatrixTest, MatrixAddition) {
    Matrix<2, 2> mat1;
    mat1[0] = Vector<2>{1.0, 2.0};
    mat1[1] = Vector<2>{3.0, 4.0};
    
    Matrix<2, 2> mat2;
    mat2[0] = Vector<2>{5.0, 6.0};
    mat2[1] = Vector<2>{7.0, 8.0};
    
    Matrix<2, 2> result = mat1 + mat2;
    
    EXPECT_DOUBLE_EQ(result[0][0], 6.0);
    EXPECT_DOUBLE_EQ(result[0][1], 8.0);
    EXPECT_DOUBLE_EQ(result[1][0], 10.0);
    EXPECT_DOUBLE_EQ(result[1][1], 12.0);
}

TEST(MatrixTest, MatrixSubtraction) {
    Matrix<2, 2> mat1;
    mat1[0] = Vector<2>{5.0, 6.0};
    mat1[1] = Vector<2>{7.0, 8.0};
    
    Matrix<2, 2> mat2;
    mat2[0] = Vector<2>{1.0, 2.0};
    mat2[1] = Vector<2>{3.0, 4.0};
    
    Matrix<2, 2> result = mat1 - mat2;
    
    EXPECT_DOUBLE_EQ(result[0][0], 4.0);
    EXPECT_DOUBLE_EQ(result[0][1], 4.0);
    EXPECT_DOUBLE_EQ(result[1][0], 4.0);
    EXPECT_DOUBLE_EQ(result[1][1], 4.0);
}

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
    auto view = CreateMatrixView<2, 2>(matrix, 1, 1);
    
    EXPECT_EQ(view.rows(), 2);
    EXPECT_EQ(view.columns(), 2);
}

TEST(MatrixViewTest, ElementAccess) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    auto view = CreateMatrixView<2, 2>(matrix, 0, 1);
    
    // Проверка чтения элементов
    EXPECT_DOUBLE_EQ(view(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(view(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(view(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(view(1, 1), 6.0);
}

TEST(MatrixViewTest, ElementModification) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    auto view = CreateMatrixView<2, 2>(matrix, 1, 1);
    
    // Модификация через view
    view(0, 0) = 100.0;
    view(0, 1) = 200.0;
    view(1, 0) = 300.0;
    view(1, 1) = 400.0;
    
    // Проверка изменений в исходной матрице
    EXPECT_DOUBLE_EQ(matrix[1][1], 100.0);
    EXPECT_DOUBLE_EQ(matrix[1][2], 200.0);
    EXPECT_DOUBLE_EQ(matrix[2][1], 300.0);
    EXPECT_DOUBLE_EQ(matrix[2][2], 400.0);
    
    // Проверка изменений через view
    EXPECT_DOUBLE_EQ(view(0, 0), 100.0);
    EXPECT_DOUBLE_EQ(view(0, 1), 200.0);
    EXPECT_DOUBLE_EQ(view(1, 0), 300.0);
    EXPECT_DOUBLE_EQ(view(1, 1), 400.0);
}

TEST(MatrixViewTest, RowAccess) {
    Matrix<3, 4> matrix = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0}
    };
    
    auto view = CreateMatrixView<2, 3>(matrix, 0, 1);
    Vector<3> row = view.row(1); // Вторая строка view
    
    EXPECT_DOUBLE_EQ(row[0], 6.0);
    EXPECT_DOUBLE_EQ(row[1], 7.0);
    EXPECT_DOUBLE_EQ(row[2], 8.0);
}

TEST(MatrixViewTest, ColumnAccess) {
    Matrix<4, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0},
        {10.0, 11.0, 12.0}
    };
    
    auto view = CreateMatrixView<3, 2>(matrix, 1, 0);
    Vector<3> column = view.column(1); // Второй столбец view
    
    EXPECT_DOUBLE_EQ(column[0], 5.0);
    EXPECT_DOUBLE_EQ(column[1], 8.0);
    EXPECT_DOUBLE_EQ(column[2], 11.0);
}

TEST(MatrixViewTest, ToMatrixConversion) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    auto view = CreateMatrixView<2, 2>(matrix, 0, 1);
    Matrix<2, 2> converted = view.toMatrix();
    
    // Проверка преобразования
    EXPECT_DOUBLE_EQ(converted[0][0], 2.0);
    EXPECT_DOUBLE_EQ(converted[0][1], 3.0);
    EXPECT_DOUBLE_EQ(converted[1][0], 5.0);
    EXPECT_DOUBLE_EQ(converted[1][1], 6.0);
    
    // Проверка, что это копия, а не ссылка
    converted[0][0] = 100.0;
    EXPECT_DOUBLE_EQ(view(0, 0), 2.0); // Исходный view не изменился
    EXPECT_DOUBLE_EQ(converted[0][0], 100.0); // Копия изменилась
}

TEST(MatrixViewTest, FullMatrixView) {
    Matrix<2, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    // View на всю матрицу
    auto fullView = CreateMatrixView<2, 3>(matrix);
    
    EXPECT_EQ(fullView.rows(), 2);
    EXPECT_EQ(fullView.columns(), 3);
    
    // Проверка всех элементов
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(fullView(i, j), matrix[i][j]);
        }
    }
}

TEST(MatrixViewTest, DifferentMatrixTypes) {
    // Тест с разными типами матриц
    Matrix<5, 5> largeMatrix;
    Matrix<2, 2> smallMatrix = {{1.0, 2.0}, {3.0, 4.0}};
    
    // View на большую матрицу
    auto largeView = CreateMatrixView<3, 3>(largeMatrix, 1, 1);
    EXPECT_NO_THROW(largeView(0, 0) = 5.0);
    
    // View на маленькую матрицу
    auto smallView = CreateMatrixView<1, 1>(smallMatrix, 0, 0);
    EXPECT_DOUBLE_EQ(smallView(0, 0), 1.0);
}

TEST(MatrixViewTest, ConstMatrixView) {
    const Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    // Const view на const матрицу
    const auto view = CreateMatrixView<2, 2>(matrix, 0, 0);
    
    // Можно читать, но нельзя модифицировать
    EXPECT_DOUBLE_EQ(view(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(view(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(view(1, 1), 5.0);
    
    // Не должно компилироваться: view(0, 0) = 10.0;
}

TEST(MatrixViewTest, MoveOperations) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    // Создание и перемещение view
    auto view1 = CreateMatrixView<2, 2>(matrix, 0, 0);
    auto view2 = std::move(view1); // Move constructor
    
    EXPECT_DOUBLE_EQ(view2(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(view2(0, 1), 2.0);
}

TEST(MatrixViewTest, EdgeCaseSingleElement) {
    Matrix<3, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    // View на один элемент
    auto singleView = CreateMatrixView<1, 1>(matrix, 1, 1);
    
    EXPECT_EQ(singleView.rows(), 1);
    EXPECT_EQ(singleView.columns(), 1);
    EXPECT_DOUBLE_EQ(singleView(0, 0), 5.0);
}

TEST(MatrixTest, ToPtrFunctionality) {
    Matrix<2, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    // Проверка указателя
    double* ptr = matrix.toPtr();
    EXPECT_DOUBLE_EQ(ptr[0], 1.0);
    EXPECT_DOUBLE_EQ(ptr[1], 2.0);
    EXPECT_DOUBLE_EQ(ptr[2], 3.0);
    EXPECT_DOUBLE_EQ(ptr[3], 4.0);
    EXPECT_DOUBLE_EQ(ptr[4], 5.0);
    EXPECT_DOUBLE_EQ(ptr[5], 6.0);

    // Модификация через указатель
    ptr[0] = 100.0;
    EXPECT_DOUBLE_EQ(matrix[0][0], 100.0);
}

TEST(MatrixTest, RowPtrFunctionality) {
    Matrix<3, 2> matrix = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    double* row1 = matrix.rowPtr(1);
    EXPECT_DOUBLE_EQ(row1[0], 3.0);
    EXPECT_DOUBLE_EQ(row1[1], 4.0);

    row1[0] = 300.0;
    EXPECT_DOUBLE_EQ(matrix[1][0], 300.0);
}

TEST(MatrixTest, RowMajorConversion) {
    Matrix<2, 2> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    auto rowMajor = matrix.toRowMajorArray();
    EXPECT_DOUBLE_EQ(rowMajor[0], 1.0);
    EXPECT_DOUBLE_EQ(rowMajor[1], 2.0);
    EXPECT_DOUBLE_EQ(rowMajor[2], 3.0);
    EXPECT_DOUBLE_EQ(rowMajor[3], 4.0);
}

TEST(MatrixTest, FromRowMajorArray) {
    Matrix<2, 2> matrix;
    std::array<double, 4> data = {10.0, 20.0, 30.0, 40.0};
    
    matrix.fromRowMajorArray(data.data());
    
    EXPECT_DOUBLE_EQ(matrix[0][0], 10.0);
    EXPECT_DOUBLE_EQ(matrix[0][1], 20.0);
    EXPECT_DOUBLE_EQ(matrix[1][0], 30.0);
    EXPECT_DOUBLE_EQ(matrix[1][1], 40.0);
}
