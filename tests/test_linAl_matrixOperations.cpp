#include <gtest/gtest.h>
#include "../include/SharedMath.h"

using namespace SharedMath::LinearAlgebra;

TEST(DynamicMatrixTest, BasicCreation) {
    DynamicMatrix matrix(3, 4);
    
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
    
    // Проверяем, что матрица инициализирована нулями
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(matrix.get(i, j), 0.0);
        }
    }
}

TEST(DynamicMatrixTest, SetAndGet) {
    DynamicMatrix matrix(2, 2);
    
    matrix.set(0, 0, 1.0);
    matrix.set(0, 1, 2.0);
    matrix.set(1, 0, 3.0);
    matrix.set(1, 1, 4.0);
    
    EXPECT_DOUBLE_EQ(matrix.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 1), 4.0);
}

TEST(DynamicMatrixTest, ScalarMultiplication) {
    DynamicMatrix matrix(2, 2);
    matrix.set(0, 0, 1.0);
    matrix.set(0, 1, 2.0);
    matrix.set(1, 0, 3.0);
    matrix.set(1, 1, 4.0);
    
    DynamicMatrix result = matrix * 2.0;
    
    EXPECT_DOUBLE_EQ(result.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result.get(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result.get(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(result.get(1, 1), 8.0);
}

TEST(DynamicMatrixTest, Clear) {
    DynamicMatrix matrix(2, 2);
    matrix.set(0, 0, 1.0);
    matrix.set(0, 1, 2.0);
    matrix.set(1, 0, 3.0);
    matrix.set(1, 1, 4.0);
    
    matrix.clear();
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(matrix.get(i, j), 0.0);
        }
    }
}

TEST(DynamicMatrixTest, Equality) {
    DynamicMatrix matrix1(2, 2);
    matrix1.set(0, 0, 1.0);
    matrix1.set(0, 1, 2.0);
    
    DynamicMatrix matrix2(2, 2);
    matrix2.set(0, 0, 1.0);
    matrix2.set(0, 1, 2.0);
    
    DynamicMatrix matrix3(2, 2);
    matrix3.set(0, 0, 5.0);
    matrix3.set(0, 1, 6.0);
    
    EXPECT_TRUE(matrix1 == matrix2);
    EXPECT_FALSE(matrix1 == matrix3);
    EXPECT_TRUE(matrix1 != matrix3);
}

TEST(DynamicMatrixTest, CopyConstructor) {
    DynamicMatrix original(2, 2);
    original.set(0, 0, 1.0);
    original.set(0, 1, 2.0);
    
    DynamicMatrix copy(original);
    
    EXPECT_EQ(copy.rows(), 2);
    EXPECT_EQ(copy.cols(), 2);
    EXPECT_DOUBLE_EQ(copy.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(copy.get(0, 1), 2.0);
}

// Тесты для Matrix (шаблонной)
TEST(MatrixTest, BasicCreation) {
    Matrix<3, 4> matrix;
    
    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
    
    // Проверяем инициализацию нулями
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_DOUBLE_EQ(matrix.get(i, j), 0.0);
        }
    }
}

TEST(MatrixTest, InitializerList) {
    Matrix<2, 2> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    EXPECT_DOUBLE_EQ(matrix.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 1), 4.0);
}

TEST(MatrixTest, ArrayAccess) {
    Matrix<2, 2> matrix;
    matrix[0][0] = 1.0;
    matrix[0][1] = 2.0;
    matrix[1][0] = 3.0;
    matrix[1][1] = 4.0;
    
    EXPECT_DOUBLE_EQ(matrix.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 1), 4.0);
}

TEST(MatrixTest, ScalarMultiplication) {
    Matrix<2, 2> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix<2, 2> result = matrix * 2.0;
    
    EXPECT_DOUBLE_EQ(result.get(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(result.get(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result.get(1, 0), 6.0);
    EXPECT_DOUBLE_EQ(result.get(1, 1), 8.0);
}

TEST(MatrixTest, MatrixAddition) {
    Matrix<2, 2> matrix1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix<2, 2> matrix2 = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    Matrix<2, 2> result = matrix1 + matrix2;
    
    EXPECT_DOUBLE_EQ(result.get(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result.get(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(result.get(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(result.get(1, 1), 12.0);
}

TEST(MatrixTest, MatrixSubtraction) {
    Matrix<2, 2> matrix1 = {
        {5.0, 6.0},
        {7.0, 8.0}
    };
    
    Matrix<2, 2> matrix2 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix<2, 2> result = matrix1 - matrix2;
    
    EXPECT_DOUBLE_EQ(result.get(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(result.get(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result.get(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(result.get(1, 1), 4.0);
}

TEST(MatrixTest, VectorMultiplication) {
    Matrix<2, 3> matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    Vector<3> vec = {2.0, 3.0, 4.0};
    Vector<2> result = matrix * vec;
    
    EXPECT_DOUBLE_EQ(result[0], 20.0); // 1*2 + 2*3 + 3*4
    EXPECT_DOUBLE_EQ(result[1], 47.0); // 4*2 + 5*3 + 6*4
}

TEST(MatrixTest, ToRowMajorArray) {
    Matrix<2, 2> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    auto array = matrix.toRowMajorArray();
    EXPECT_DOUBLE_EQ(array[0], 1.0);
    EXPECT_DOUBLE_EQ(array[1], 2.0);
    EXPECT_DOUBLE_EQ(array[2], 3.0);
    EXPECT_DOUBLE_EQ(array[3], 4.0);
}

TEST(MatrixTest, ToColumnMajorArray) {
    Matrix<2, 2> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    auto array = matrix.toColumnMajorArray();
    EXPECT_DOUBLE_EQ(array[0], 1.0);
    EXPECT_DOUBLE_EQ(array[1], 3.0);
    EXPECT_DOUBLE_EQ(array[2], 2.0);
    EXPECT_DOUBLE_EQ(array[3], 4.0);
}

TEST(MatrixTest, CommaInitializer) {
    Matrix<2, 2> matrix;
    matrix << 1.0, 2.0,
              3.0, 4.0;
    
    EXPECT_DOUBLE_EQ(matrix.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(matrix.get(1, 1), 4.0);
}

// Тесты для MatrixOperations
TEST(MatrixOperationsTest, Addition) {
    MatrixOperations ops;
    
    auto A = std::make_shared<Matrix<2, 2>>();
    *A << 1.0, 2.0,
           3.0, 4.0;
    
    auto B = std::make_shared<Matrix<2, 2>>();
    *B << 5.0, 6.0,
           7.0, 8.0;
    
    auto result = ops.add(A, B);
    
    EXPECT_DOUBLE_EQ(result->get(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result->get(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(result->get(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(result->get(1, 1), 12.0);
}

TEST(MatrixOperationsTest, Subtraction) {
    MatrixOperations ops;
    
    auto A = std::make_shared<Matrix<2, 2>>();
    *A << 5.0, 6.0,
           7.0, 8.0;
    
    auto B = std::make_shared<Matrix<2, 2>>();
    *B << 1.0, 2.0,
           3.0, 4.0;
    
    auto result = ops.substract(A, B);
    
    EXPECT_DOUBLE_EQ(result->get(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(result->get(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result->get(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(result->get(1, 1), 4.0);
}

TEST(MatrixOperationsTest, Multiplication) {
    MatrixOperations ops;
    
    auto A = std::make_shared<Matrix<2, 3>>();
    *A << 1.0, 2.0, 3.0,
           4.0, 5.0, 6.0;
    
    auto B = std::make_shared<Matrix<3, 2>>();
    *B << 7.0, 8.0,
           9.0, 10.0,
           11.0, 12.0;
    
    auto result = ops.multiply(A, B);
    
    EXPECT_DOUBLE_EQ(result->get(0, 0), 58.0);  // 1*7 + 2*9 + 3*11
    EXPECT_DOUBLE_EQ(result->get(0, 1), 64.0);  // 1*8 + 2*10 + 3*12
    EXPECT_DOUBLE_EQ(result->get(1, 0), 139.0); // 4*7 + 5*9 + 6*11
    EXPECT_DOUBLE_EQ(result->get(1, 1), 154.0); // 4*8 + 5*10 + 6*12
}

TEST(MatrixOperationsTest, Transpose) {
    MatrixOperations ops;
    
    auto A = std::make_shared<Matrix<2, 3>>();
    *A << 1.0, 2.0, 3.0,
           4.0, 5.0, 6.0;
    
    auto result = ops.transpose(A);
    
    EXPECT_EQ(result->rows(), 3);
    EXPECT_EQ(result->cols(), 2);
    EXPECT_DOUBLE_EQ(result->get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->get(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(result->get(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(result->get(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(result->get(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(result->get(2, 1), 6.0);
}

TEST(MatrixOperationsTest, KroneckerProduct) {
    MatrixOperations ops;
    
    auto A = std::make_shared<Matrix<2, 2>>();
    *A << 1.0, 2.0,
           3.0, 4.0;
    
    auto B = std::make_shared<Matrix<2, 2>>();
    *B << 5.0, 6.0,
           7.0, 8.0;
    
    auto result = ops.kroneckerProduct(A, B);
    
    EXPECT_EQ(result->rows(), 4);
    EXPECT_EQ(result->cols(), 4);
    
    // Проверяем блоки кронекерова произведения
    EXPECT_DOUBLE_EQ(result->get(0, 0), 5.0);   // 1*5
    EXPECT_DOUBLE_EQ(result->get(0, 1), 6.0);   // 1*6
    EXPECT_DOUBLE_EQ(result->get(0, 2), 10.0);  // 2*5
    EXPECT_DOUBLE_EQ(result->get(0, 3), 12.0);  // 2*6
    
    EXPECT_DOUBLE_EQ(result->get(1, 0), 7.0);   // 1*7
    EXPECT_DOUBLE_EQ(result->get(1, 1), 8.0);   // 1*8
    EXPECT_DOUBLE_EQ(result->get(1, 2), 14.0);  // 2*7
    EXPECT_DOUBLE_EQ(result->get(1, 3), 16.0);  // 2*8
    
    EXPECT_DOUBLE_EQ(result->get(2, 0), 15.0);  // 3*5
    EXPECT_DOUBLE_EQ(result->get(2, 1), 18.0);  // 3*6
    EXPECT_DOUBLE_EQ(result->get(2, 2), 20.0);  // 4*5
    EXPECT_DOUBLE_EQ(result->get(2, 3), 24.0);  // 4*6
    
    EXPECT_DOUBLE_EQ(result->get(3, 0), 21.0);  // 3*7
    EXPECT_DOUBLE_EQ(result->get(3, 1), 24.0);  // 3*8
    EXPECT_DOUBLE_EQ(result->get(3, 2), 28.0);  // 4*7
    EXPECT_DOUBLE_EQ(result->get(3, 3), 32.0);  // 4*8
}

TEST(MatrixOperationsTest, MixedMatrixTypes) {
    MatrixOperations ops;
    
    // Тестирование операций между разными типами матриц
    auto fixedMatrix = std::make_shared<Matrix<2, 2>>();
    *fixedMatrix << 1.0, 2.0,
                    3.0, 4.0;
    
    auto dynamicMatrix = std::make_shared<DynamicMatrix>(2, 2);
    dynamicMatrix->set(0, 0, 5.0);
    dynamicMatrix->set(0, 1, 6.0);
    dynamicMatrix->set(1, 0, 7.0);
    dynamicMatrix->set(1, 1, 8.0);
    
    auto result = ops.add(fixedMatrix, dynamicMatrix);
    
    EXPECT_DOUBLE_EQ(result->get(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(result->get(0, 1), 8.0);
    EXPECT_DOUBLE_EQ(result->get(1, 0), 10.0);
    EXPECT_DOUBLE_EQ(result->get(1, 1), 12.0);
}

TEST(MatrixOperationsTest, ValidationChecks) {
    MatrixOperations ops;
    
    auto A = std::make_shared<Matrix<2, 3>>();
    auto B = std::make_shared<Matrix<2, 2>>(); // Несовместимые размеры
    
    EXPECT_FALSE(ops.canAdd(*A, *B));
    EXPECT_FALSE(ops.canMultiply(*A, *B));
    
    // Должны выбрасывать исключения
    EXPECT_THROW(ops.add(A, B), std::invalid_argument);
    EXPECT_THROW(ops.multiply(A, B), std::invalid_argument);
}

TEST(MatrixOperationsTest, TraceChecks){
    MatrixOperations ops;

    auto matrix = std::make_shared<DynamicMatrix>(2, 2);

    matrix->set(0, 0, 1); matrix->set(0, 1, 2); 
    matrix->set(1, 0, 4); matrix->set(1, 1, 5);

    EXPECT_EQ(ops.trace(matrix), 6.0);

    auto invalid_matrix = std::make_shared<DynamicMatrix>(4, 6);

    EXPECT_ANY_THROW(ops.trace(invalid_matrix));
}

TEST(LUDecompositionTest, DecompositionTest) {
    DynamicMatrix A(3, 3);
    A.set(0, 0, 2); A.set(0, 1, -1); A.set(0, 2, -2);
    A.set(1, 0, -4); A.set(1, 1, 6); A.set(1, 2, 3);
    A.set(2, 0, -4); A.set(2, 1, -2); A.set(2, 2, 8);

    LUDecomposition lu(A);
    lu.MakeDecomposition();

    const auto& L = lu.GetL();
    const auto& U = lu.GetU();
    const auto& pivot = lu.GetPivot();

    EXPECT_EQ(L.rows(), 3);
    EXPECT_EQ(L.cols(), 3);
    EXPECT_EQ(U.rows(), 3);
    EXPECT_EQ(U.cols(), 3);

    EXPECT_NEAR(L.get(0, 0), 1.00, 1e-10);
    EXPECT_NEAR(L.get(0, 1), 0.00, 1e-10);
    EXPECT_NEAR(L.get(0, 2), 0.00, 1e-10);
    EXPECT_NEAR(L.get(1, 0), 1.00, 1e-10);
    EXPECT_NEAR(L.get(1, 1), 1.00, 1e-10);
    EXPECT_NEAR(L.get(1, 2), 0.00, 1e-10);
    EXPECT_NEAR(L.get(2, 0), -0.50, 1e-10);
    EXPECT_NEAR(L.get(2, 1), -0.25, 1e-10);
    EXPECT_NEAR(L.get(2, 2), 1.00, 1e-10);

    EXPECT_NEAR(U.get(0, 0), -4.00, 1e-10);
    EXPECT_NEAR(U.get(0, 1), 6.00, 1e-10);
    EXPECT_NEAR(U.get(0, 2), 3.00, 1e-10);
    EXPECT_NEAR(U.get(1, 0), 0.00, 1e-10);
    EXPECT_NEAR(U.get(1, 1), -8.00, 1e-10);
    EXPECT_NEAR(U.get(1, 2), 5.00, 1e-10);
    EXPECT_NEAR(U.get(2, 0), 0.00, 1e-10);
    EXPECT_NEAR(U.get(2, 1), 0.00, 1e-10);
    EXPECT_NEAR(U.get(2, 2), 0.75, 1e-10);

    EXPECT_EQ(pivot[0], 1);
    EXPECT_EQ(pivot[1], 2);
    EXPECT_EQ(pivot[2], 0);

    EXPECT_TRUE(lu.VerifyDecomposition());
}

TEST(DeterminantTest, DetTest){
    MatrixOperations ops;

    auto matrix = std::make_shared<DynamicMatrix>(1, 2);

    EXPECT_THROW(ops.determinant(matrix), std::invalid_argument);

    matrix.reset();
    matrix = std::make_shared<DynamicMatrix>(1, 1);
    matrix->set(0, 0, 5);

    EXPECT_EQ(ops.determinant(matrix), 5);

    matrix.reset();
    matrix = std::make_shared<DynamicMatrix>(3, 3);
    
    matrix->set(0, 0, 3); matrix->set(0, 1, 8); matrix->set(0, 2, 9);
    matrix->set(1, 0, 29); matrix->set(1, 1, 29); matrix->set(1, 2,12);
    matrix->set(2, 0, 27); matrix->set(2, 1, 22); matrix->set(2, 2, 221);

    EXPECT_NEAR(ops.determinant(matrix), -31550.0, 1e-6);
}

TEST(LUDecompositionTest, IdentityMatrix) {
    // Тест на единичную матрицу
    DynamicMatrix I(3, 3);
    I.set(0, 0, 1); I.set(0, 1, 0); I.set(0, 2, 0);
    I.set(1, 0, 0); I.set(1, 1, 1); I.set(1, 2, 0);
    I.set(2, 0, 0); I.set(2, 1, 0); I.set(2, 2, 1);

    LUDecomposition lu(I);
    lu.MakeDecomposition();

    const auto& L = lu.GetL();
    const auto& U = lu.GetU();

    // Для единичной матрицы L и U также должны быть единичными
    EXPECT_NEAR(L.get(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(L.get(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(L.get(2, 2), 1.0, 1e-10);
    
    EXPECT_NEAR(U.get(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(U.get(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(U.get(2, 2), 1.0, 1e-10);

    EXPECT_TRUE(lu.VerifyDecomposition());
    EXPECT_NEAR(lu.Determinant(), 1.0, 1e-10);
}

TEST(LUDecompositionTest, DiagonalMatrix) {
    // Тест на диагональную матрицу
    DynamicMatrix D(3, 3);
    D.set(0, 0, 2); D.set(0, 1, 0); D.set(0, 2, 0);
    D.set(1, 0, 0); D.set(1, 1, 3); D.set(1, 2, 0);
    D.set(2, 0, 0); D.set(2, 1, 0); D.set(2, 2, 5);

    LUDecomposition lu(D);
    lu.MakeDecomposition();

    const auto& L = lu.GetL();
    const auto& U = lu.GetU();

    // L должна быть единичной, U - исходной диагональной
    EXPECT_NEAR(L.get(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(L.get(1, 1), 1.0, 1e-10);
    EXPECT_NEAR(L.get(2, 2), 1.0, 1e-10);
    
    EXPECT_NEAR(U.get(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(U.get(1, 1), 3.0, 1e-10);
    EXPECT_NEAR(U.get(2, 2), 5.0, 1e-10);

    EXPECT_TRUE(lu.VerifyDecomposition());
    EXPECT_NEAR(lu.Determinant(), 30.0, 1e-10); // 2 * 3 * 5 = 30
}

TEST(LUDecompositionTest, SingularMatrix) {
    // Тест на вырожденную матрицу
    DynamicMatrix S(3, 3);
    S.set(0, 0, 1); S.set(0, 1, 2); S.set(0, 2, 3);
    S.set(1, 0, 2); S.set(1, 1, 4); S.set(1, 2, 6); // 2 * первая строка
    S.set(2, 0, 3); S.set(2, 1, 6); S.set(2, 2, 9); // 3 * первая строка

    LUDecomposition lu(S);
    lu.MakeDecomposition();
    
    // Вырожденная матрица должна выбрасывать исключение
    EXPECT_NEAR(lu.Determinant(), 0.0, 1.0e-10);
}

TEST(LUDecompositionTest, PermutationMatrix) {
    // Тест на матрицу перестановок
    DynamicMatrix A(3, 3);
    A.set(0, 0, 0); A.set(0, 1, 1); A.set(0, 2, 0);
    A.set(1, 0, 1); A.set(1, 1, 0); A.set(1, 2, 0);
    A.set(2, 0, 0); A.set(2, 1, 0); A.set(2, 2, 1);

    LUDecomposition lu(A);
    lu.MakeDecomposition();

    DynamicMatrix P = lu.GetPermutationMatrix();
    
    // Проверяем, что P - матрица перестановок
    double sum = 0;
    for (size_t i = 0; i < P.rows(); ++i) {
        for (size_t j = 0; j < P.cols(); ++j) {
            sum += P.get(i, j);
        }
    }
    EXPECT_NEAR(sum, 3.0, 1e-10); // Сумма всех элементов = размеру матрицы
    
    EXPECT_TRUE(lu.VerifyDecomposition());
}

TEST(DeterminantTest, ZeroDeterminant) {
    MatrixOperations ops;

    auto matrix = std::make_shared<DynamicMatrix>(3, 3);
    matrix->set(0, 0, 1); matrix->set(0, 1, 2); matrix->set(0, 2, 3);
    matrix->set(1, 0, 4); matrix->set(1, 1, 5); matrix->set(1, 2, 6);
    matrix->set(2, 0, 7); matrix->set(2, 1, 8); matrix->set(2, 2, 9);

    // Вырожденная матрица (строки линейно зависимы)
    EXPECT_NEAR(ops.determinant(matrix), 0.0, 1e-10);
}

TEST(DeterminantTest, NegativeDeterminant) {
    MatrixOperations ops;

    auto matrix = std::make_shared<DynamicMatrix>(2, 2);
    matrix->set(0, 0, 1); matrix->set(0, 1, 2);
    matrix->set(1, 0, 3); matrix->set(1, 1, 4);

    // 1*4 - 2*3 = 4 - 6 = -2
    EXPECT_NEAR(ops.determinant(matrix), -2.0, 1e-10);
}

TEST(DeterminantTest, LargeValues) {
    MatrixOperations ops;

    auto matrix = std::make_shared<DynamicMatrix>(2, 2);
    matrix->set(0, 0, 1e6); matrix->set(0, 1, 2e6);
    matrix->set(1, 0, 3e6); matrix->set(1, 1, 4e6);

    // (1e6 * 4e6) - (2e6 * 3e6) = 4e12 - 6e12 = -2e12
    EXPECT_NEAR(ops.determinant(matrix), -2e12, 1e-2);
}

TEST(DeterminantTest, SmallValues) {
    MatrixOperations ops;

    auto matrix = std::make_shared<DynamicMatrix>(2, 2);
    matrix->set(0, 0, 1e-6); matrix->set(0, 1, 2e-6);
    matrix->set(1, 0, 3e-6); matrix->set(1, 1, 4e-6);

    // (1e-6 * 4e-6) - (2e-6 * 3e-6) = 4e-12 - 6e-12 = -2e-12
    EXPECT_NEAR(ops.determinant(matrix), -2e-12, 1e-20);
}

TEST(LUDecompositionTest, ClearMethod) {
    DynamicMatrix A(2, 2);
    A.set(0, 0, 1); A.set(0, 1, 2);
    A.set(1, 0, 3); A.set(1, 1, 4);

    LUDecomposition lu(A);
    lu.MakeDecomposition();
    
    EXPECT_TRUE(lu.IsDecomposed());
    
    lu.Clear();
    
    EXPECT_FALSE(lu.IsDecomposed());
    EXPECT_THROW(lu.GetL(), std::runtime_error);
    EXPECT_THROW(lu.GetU(), std::runtime_error);
}

TEST(LUDecompositionTest, SetMatrixAfterClear) {
    DynamicMatrix A(2, 2);
    A.set(0, 0, 1); A.set(0, 1, 2);
    A.set(1, 0, 3); A.set(1, 1, 4);

    DynamicMatrix B(2, 2);
    B.set(0, 0, 5); B.set(0, 1, 6);
    B.set(1, 0, 7); B.set(1, 1, 8);

    LUDecomposition lu(A);
    lu.MakeDecomposition();
    
    lu.Clear();
    lu.SetMatrixToDecompose(B);
    lu.MakeDecomposition();

    EXPECT_NEAR(lu.Determinant(), 5*8 - 6*7, 1e-10); // 40 - 42 = -2
    EXPECT_TRUE(lu.VerifyDecomposition());
}

TEST(LUDecompositionTest, NonSquareMatrix) {
    DynamicMatrix A(2, 3); // Не квадратная матрица
    
    LUDecomposition lu;
    
    EXPECT_THROW(lu.SetMatrixToDecompose(A), std::invalid_argument);
}

TEST(DeterminantTest, VerifyDecompositionTolerance) {
    DynamicMatrix A(3, 3);
    A.set(0, 0, 1); A.set(0, 1, 2); A.set(0, 2, 3);
    A.set(1, 0, 4); A.set(1, 1, 5); A.set(1, 2, 6);
    A.set(2, 0, 7); A.set(2, 1, 8); A.set(2, 2, 10); // Изменено для невырожденности

    LUDecomposition lu(A);
    lu.MakeDecomposition();

    // Проверяем с разной толерантностью
    EXPECT_TRUE(lu.VerifyDecomposition(1e-10));
    EXPECT_TRUE(lu.VerifyDecomposition(1e-5));
    EXPECT_TRUE(lu.VerifyDecomposition(1e-15));
}

TEST(LUDecompositionTest, AccessWithoutDecomposition) {
    DynamicMatrix A(2, 2);
    A.set(0, 0, 1); A.set(0, 1, 2);
    A.set(1, 0, 3); A.set(1, 1, 4);

    LUDecomposition lu(A);
    
    // Попытка доступа до выполнения разложения
    EXPECT_THROW(lu.GetL(), std::runtime_error);
    EXPECT_THROW(lu.GetU(), std::runtime_error);
    EXPECT_THROW(lu.GetPivot(), std::runtime_error);
    EXPECT_THROW(lu.GetPermutationMatrix(), std::runtime_error);
    EXPECT_THROW(lu.Determinant(), std::runtime_error);
    EXPECT_EQ(lu.VerifyDecomposition(), false);
}