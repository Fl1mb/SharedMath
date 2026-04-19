#include <gtest/gtest.h>
#include "LinearAlgebra/Tensor.h"

#include <cmath>
#include <sstream>

using namespace SharedMath::LinearAlgebra;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

static constexpr double kEps = 1e-9;

static void expectNear(const Tensor& a, const Tensor& b, double tol = kEps) {
    ASSERT_EQ(a.shape(), b.shape()) << "shapes differ";
    for (size_t i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a.flat(i), b.flat(i), tol)
            << "mismatch at flat index " << i;
}

// ════════════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorConstruction, DefaultIsEmpty) {
    Tensor t;
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
    EXPECT_EQ(t.ndim(), 0u);
}

TEST(TensorConstruction, ShapeFill) {
    Tensor t({2, 3}, 7.0);
    EXPECT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.dim(0), 2u);
    EXPECT_EQ(t.dim(1), 3u);
    EXPECT_EQ(t.size(), 6u);
    for (size_t i = 0; i < t.size(); ++i)
        EXPECT_DOUBLE_EQ(t.flat(i), 7.0);
}

TEST(TensorConstruction, ShapeData) {
    std::vector<double> data = {1, 2, 3, 4, 5, 6};
    Tensor t({2, 3}, data);
    EXPECT_EQ(t.size(), 6u);
    EXPECT_DOUBLE_EQ(t(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(t(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(t(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(t(1, 2), 6.0);
}

TEST(TensorConstruction, SizeMismatchThrows) {
    EXPECT_THROW(Tensor({2, 3}, std::vector<double>(5)), std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════════
// Static factories
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorFactories, Zeros) {
    Tensor t = Tensor::zeros({3, 4});
    EXPECT_EQ(t.size(), 12u);
    for (size_t i = 0; i < t.size(); ++i)
        EXPECT_DOUBLE_EQ(t.flat(i), 0.0);
}

TEST(TensorFactories, Ones) {
    Tensor t = Tensor::ones({2, 2, 2});
    for (size_t i = 0; i < t.size(); ++i)
        EXPECT_DOUBLE_EQ(t.flat(i), 1.0);
}

TEST(TensorFactories, Eye) {
    Tensor t = Tensor::eye(3);
    ASSERT_EQ(t.shape(), (Tensor::Shape{3, 3}));
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_DOUBLE_EQ(t(i, j), i == j ? 1.0 : 0.0);
}

TEST(TensorFactories, Arange) {
    Tensor t = Tensor::arange(0.0, 5.0, 1.0);
    ASSERT_EQ(t.size(), 5u);
    for (size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(t(i), static_cast<double>(i));
}

TEST(TensorFactories, ArangeNegativeStep) {
    Tensor t = Tensor::arange(5.0, 0.0, -1.0);
    ASSERT_EQ(t.size(), 5u);
    EXPECT_DOUBLE_EQ(t(0), 5.0);
    EXPECT_DOUBLE_EQ(t(4), 1.0);
}

TEST(TensorFactories, Linspace) {
    Tensor t = Tensor::linspace(0.0, 1.0, 5);
    ASSERT_EQ(t.size(), 5u);
    EXPECT_NEAR(t(0), 0.00, kEps);
    EXPECT_NEAR(t(1), 0.25, kEps);
    EXPECT_NEAR(t(2), 0.50, kEps);
    EXPECT_NEAR(t(3), 0.75, kEps);
    EXPECT_NEAR(t(4), 1.00, kEps);
}

TEST(TensorFactories, FromVector) {
    Tensor t = Tensor::from_vector({3.0, 1.0, 4.0});
    ASSERT_EQ(t.ndim(), 1u);
    EXPECT_DOUBLE_EQ(t(0), 3.0);
    EXPECT_DOUBLE_EQ(t(2), 4.0);
}

TEST(TensorFactories, FromMatrix) {
    Tensor t = Tensor::from_matrix(2, 3, {1, 2, 3, 4, 5, 6});
    ASSERT_EQ(t.shape(), (Tensor::Shape{2, 3}));
    EXPECT_DOUBLE_EQ(t(1, 2), 6.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Element access
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorAccess, Variadic3D) {
    Tensor t({2, 3, 4}, 0.0);
    t(1, 2, 3) = 42.0;
    EXPECT_DOUBLE_EQ(t(1, 2, 3), 42.0);
    EXPECT_DOUBLE_EQ(t(0, 0, 0), 0.0);
}

TEST(TensorAccess, AtVector) {
    Tensor t({2, 2}, {1, 2, 3, 4});
    EXPECT_DOUBLE_EQ(t.at({0, 0}), 1.0);
    EXPECT_DOUBLE_EQ(t.at({1, 1}), 4.0);
}

TEST(TensorAccess, OutOfRangeThrows) {
    Tensor t({2, 2});
    EXPECT_THROW(t(2, 0), std::out_of_range);
    EXPECT_THROW(t(0, 2), std::out_of_range);
}

TEST(TensorAccess, WrongRankThrows) {
    Tensor t({3, 3});
    EXPECT_THROW(t.at({0, 0, 0}), std::invalid_argument);
}

TEST(TensorAccess, UnravelIndex) {
    Tensor t({2, 3, 4});
    auto idx = t.unravel(11);
    // Row-major: flat=11 in shape [2,3,4] → strides [12,4,1]
    // 11 / 12 = 0 rem 11; 11 / 4 = 2 rem 3; 3 / 1 = 3
    EXPECT_EQ(idx[0], 0u);
    EXPECT_EQ(idx[1], 2u);
    EXPECT_EQ(idx[2], 3u);
}

// ════════════════════════════════════════════════════════════════════════════
// Shape operations
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorShape, Reshape) {
    Tensor t = Tensor::arange(0.0, 6.0);
    Tensor r = t.reshape({2, 3});
    ASSERT_EQ(r.shape(), (Tensor::Shape{2, 3}));
    EXPECT_DOUBLE_EQ(r(1, 2), 5.0);
}

TEST(TensorShape, ReshapeWrongSizeThrows) {
    Tensor t({6});
    EXPECT_THROW(t.reshape({2, 4}), std::invalid_argument);
}

TEST(TensorShape, Flatten) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor f = t.flatten();
    ASSERT_EQ(f.ndim(), 1u);
    ASSERT_EQ(f.size(), 6u);
    EXPECT_DOUBLE_EQ(f(5), 6.0);
}

TEST(TensorShape, Squeeze) {
    Tensor t({1, 3, 1, 2});
    Tensor s = t.squeeze();
    ASSERT_EQ(s.shape(), (Tensor::Shape{3, 2}));
}

TEST(TensorShape, ExpandDims) {
    Tensor t({3, 4});
    Tensor e = t.expand_dims(1);
    ASSERT_EQ(e.shape(), (Tensor::Shape{3, 1, 4}));
}

TEST(TensorShape, Transpose2D) {
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tr = t.transpose();
    ASSERT_EQ(tr.shape(), (Tensor::Shape{3, 2}));
    EXPECT_DOUBLE_EQ(tr(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(tr(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(tr(2, 1), 6.0);
}

TEST(TensorShape, TransposeCustomAxes) {
    Tensor t({2, 3, 4});
    Tensor tr = t.transpose({1, 0, 2});
    ASSERT_EQ(tr.shape(), (Tensor::Shape{3, 2, 4}));
}

TEST(TensorShape, Slice) {
    Tensor t = Tensor::arange(0.0, 12.0).reshape({3, 4});
    Tensor s = t.slice(0, 1, 3);   // rows 1 and 2
    ASSERT_EQ(s.shape(), (Tensor::Shape{2, 4}));
    EXPECT_DOUBLE_EQ(s(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(s(1, 3), 11.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Arithmetic & broadcasting
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorArithmetic, ScalarAdd) {
    Tensor t = Tensor::ones({2, 2});
    Tensor r = t + 3.0;
    for (size_t i = 0; i < r.size(); ++i)
        EXPECT_DOUBLE_EQ(r.flat(i), 4.0);
}

TEST(TensorArithmetic, ScalarMul) {
    Tensor t = Tensor::ones({3});
    Tensor r = 5.0 * t;
    for (size_t i = 0; i < r.size(); ++i)
        EXPECT_DOUBLE_EQ(r.flat(i), 5.0);
}

TEST(TensorArithmetic, ScalarDiv) {
    Tensor t({3}, {2.0, 4.0, 8.0});
    Tensor r = t / 2.0;
    EXPECT_DOUBLE_EQ(r(0), 1.0);
    EXPECT_DOUBLE_EQ(r(1), 2.0);
    EXPECT_DOUBLE_EQ(r(2), 4.0);
}

TEST(TensorArithmetic, Negation) {
    Tensor t({2}, {1.0, -3.0});
    Tensor r = -t;
    EXPECT_DOUBLE_EQ(r(0), -1.0);
    EXPECT_DOUBLE_EQ(r(1),  3.0);
}

TEST(TensorArithmetic, ElementwiseAdd) {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    Tensor r = a + b;
    EXPECT_DOUBLE_EQ(r(0, 0), 6.0);
    EXPECT_DOUBLE_EQ(r(1, 1), 12.0);
}

TEST(TensorArithmetic, ElementwiseMul) {
    Tensor a({3}, {2.0, 3.0, 4.0});
    Tensor b({3}, {1.0, 2.0, 3.0});
    Tensor r = a * b;
    EXPECT_DOUBLE_EQ(r(0), 2.0);
    EXPECT_DOUBLE_EQ(r(1), 6.0);
    EXPECT_DOUBLE_EQ(r(2), 12.0);
}

TEST(TensorArithmetic, BroadcastRowVector) {
    // (3, 1) + (1, 4) → (3, 4)
    Tensor a({3, 1}, {1.0, 2.0, 3.0});
    Tensor b({1, 4}, {10.0, 20.0, 30.0, 40.0});
    Tensor r = a + b;
    ASSERT_EQ(r.shape(), (Tensor::Shape{3, 4}));
    EXPECT_DOUBLE_EQ(r(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(r(2, 3), 43.0);
}

TEST(TensorArithmetic, BroadcastScalarTensor) {
    // (1,) + (3, 3) → (3, 3)
    Tensor a({1}, {5.0});
    Tensor b = Tensor::ones({3, 3});
    Tensor r = a + b;
    ASSERT_EQ(r.shape(), (Tensor::Shape{3, 3}));
    for (size_t i = 0; i < r.size(); ++i)
        EXPECT_DOUBLE_EQ(r.flat(i), 6.0);
}

TEST(TensorArithmetic, IncompatibleBroadcastThrows) {
    Tensor a({3});
    Tensor b({4});
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TensorArithmetic, CompoundAssignment) {
    Tensor t = Tensor::ones({3});
    t += Tensor::ones({3});
    for (size_t i = 0; i < t.size(); ++i)
        EXPECT_DOUBLE_EQ(t.flat(i), 2.0);
    t *= 3.0;
    for (size_t i = 0; i < t.size(); ++i)
        EXPECT_DOUBLE_EQ(t.flat(i), 6.0);
}

TEST(TensorArithmetic, Equality) {
    Tensor a({2}, {1.0, 2.0});
    Tensor b({2}, {1.0, 2.0});
    Tensor c({2}, {1.0, 3.0});
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
}

// ════════════════════════════════════════════════════════════════════════════
// Global reductions
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorReductions, Sum) {
    Tensor t = Tensor::arange(1.0, 5.0);   // [1, 2, 3, 4]
    EXPECT_DOUBLE_EQ(t.sum(), 10.0);
}

TEST(TensorReductions, Product) {
    Tensor t({3}, {2.0, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(t.product(), 24.0);
}

TEST(TensorReductions, MinMax) {
    Tensor t({4}, {3.0, 1.0, 4.0, 2.0});
    EXPECT_DOUBLE_EQ(t.min(), 1.0);
    EXPECT_DOUBLE_EQ(t.max(), 4.0);
}

TEST(TensorReductions, Mean) {
    Tensor t = Tensor::arange(0.0, 5.0);   // [0, 1, 2, 3, 4]
    EXPECT_NEAR(t.mean(), 2.0, kEps);
}

TEST(TensorReductions, VariancePopulation) {
    Tensor t({4}, {2.0, 4.0, 4.0, 4.0});   // mean=3.5 would be wrong; let's use {2,4,4,4}
    // mean = 3.5 → no. Actual mean = 14/4 = 3.5
    // But let's use a simple case:
    Tensor u({4}, {0.0, 2.0, 4.0, 6.0});  // mean=3, var=5
    EXPECT_NEAR(u.var(),      5.0, kEps);
    EXPECT_NEAR(u.var(true),  20.0/3.0, kEps);
}

TEST(TensorReductions, StdDev) {
    Tensor t({4}, {0.0, 2.0, 4.0, 6.0});  // stddev = sqrt(5)
    EXPECT_NEAR(t.stddev(), std::sqrt(5.0), kEps);
}

TEST(TensorReductions, Argmin) {
    Tensor t({4}, {3.0, 1.0, 4.0, 2.0});
    EXPECT_EQ(t.argmin(), 1u);
}

TEST(TensorReductions, Argmax) {
    Tensor t({4}, {3.0, 1.0, 4.0, 2.0});
    EXPECT_EQ(t.argmax(), 2u);
}

// ════════════════════════════════════════════════════════════════════════════
// Axis reductions
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorAxisReductions, SumAxis0) {
    // [[1, 2, 3],  → sum axis 0 → [5, 7, 9]
    //  [4, 5, 6]]
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor s = t.sum(0);
    ASSERT_EQ(s.shape(), (Tensor::Shape{3}));
    EXPECT_DOUBLE_EQ(s(0), 5.0);
    EXPECT_DOUBLE_EQ(s(1), 7.0);
    EXPECT_DOUBLE_EQ(s(2), 9.0);
}

TEST(TensorAxisReductions, SumAxis1) {
    // [[1, 2, 3],  → sum axis 1 → [6, 15]
    //  [4, 5, 6]]
    Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor s = t.sum(1);
    ASSERT_EQ(s.shape(), (Tensor::Shape{2}));
    EXPECT_DOUBLE_EQ(s(0), 6.0);
    EXPECT_DOUBLE_EQ(s(1), 15.0);
}

TEST(TensorAxisReductions, MaxAxis0) {
    Tensor t({2, 3}, {1, 5, 3, 4, 2, 6});
    Tensor mx = t.max(0);
    ASSERT_EQ(mx.shape(), (Tensor::Shape{3}));
    EXPECT_DOUBLE_EQ(mx(0), 4.0);
    EXPECT_DOUBLE_EQ(mx(1), 5.0);
    EXPECT_DOUBLE_EQ(mx(2), 6.0);
}

TEST(TensorAxisReductions, MeanAxis1) {
    Tensor t({2, 4}, {0, 2, 4, 6,  1, 3, 5, 7});
    Tensor m = t.mean(1);
    EXPECT_NEAR(m(0), 3.0, kEps);
    EXPECT_NEAR(m(1), 4.0, kEps);
}

// ════════════════════════════════════════════════════════════════════════════
// Element-wise math
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorElemMath, Abs) {
    Tensor t({3}, {-1.0, 2.0, -3.0});
    Tensor r = t.abs();
    EXPECT_DOUBLE_EQ(r(0), 1.0);
    EXPECT_DOUBLE_EQ(r(1), 2.0);
    EXPECT_DOUBLE_EQ(r(2), 3.0);
}

TEST(TensorElemMath, Sqrt) {
    Tensor t({3}, {0.0, 1.0, 4.0});
    Tensor r = t.sqrt();
    EXPECT_NEAR(r(0), 0.0, kEps);
    EXPECT_NEAR(r(1), 1.0, kEps);
    EXPECT_NEAR(r(2), 2.0, kEps);
}

TEST(TensorElemMath, ExpLog) {
    Tensor t({3}, {0.0, 1.0, 2.0});
    Tensor e = t.exp();
    Tensor l = e.log();
    EXPECT_NEAR(l(0), 0.0, kEps);
    EXPECT_NEAR(l(1), 1.0, kEps);
    EXPECT_NEAR(l(2), 2.0, kEps);
}

TEST(TensorElemMath, Pow) {
    Tensor t({3}, {1.0, 2.0, 3.0});
    Tensor r = t.pow(2.0);
    EXPECT_NEAR(r(0), 1.0, kEps);
    EXPECT_NEAR(r(1), 4.0, kEps);
    EXPECT_NEAR(r(2), 9.0, kEps);
}

TEST(TensorElemMath, Clip) {
    Tensor t({5}, {-2.0, -1.0, 0.0, 1.0, 2.0});
    Tensor r = t.clip(-1.0, 1.0);
    EXPECT_DOUBLE_EQ(r(0), -1.0);
    EXPECT_DOUBLE_EQ(r(2),  0.0);
    EXPECT_DOUBLE_EQ(r(4),  1.0);
}

TEST(TensorElemMath, Sign) {
    Tensor t({3}, {-5.0, 0.0, 3.0});
    Tensor r = t.sign();
    EXPECT_DOUBLE_EQ(r(0), -1.0);
    EXPECT_DOUBLE_EQ(r(1),  0.0);
    EXPECT_DOUBLE_EQ(r(2),  1.0);
}

TEST(TensorElemMath, TrigFunctions) {
    Tensor t({1}, {0.0});
    EXPECT_NEAR(t.sin()(0), 0.0, kEps);
    EXPECT_NEAR(t.cos()(0), 1.0, kEps);
    EXPECT_NEAR(t.tanh()(0), 0.0, kEps);
}

TEST(TensorElemMath, FloorCeilRound) {
    Tensor t({3}, {1.2, 2.5, 3.8});
    Tensor fl = t.floor();
    Tensor ce = t.ceil();
    Tensor ro = t.round();
    EXPECT_DOUBLE_EQ(fl(0), 1.0);  EXPECT_DOUBLE_EQ(fl(2), 3.0);
    EXPECT_DOUBLE_EQ(ce(0), 2.0);  EXPECT_DOUBLE_EQ(ce(2), 4.0);
    EXPECT_DOUBLE_EQ(ro(0), 1.0);  EXPECT_DOUBLE_EQ(ro(2), 4.0);
}

TEST(TensorElemMath, Apply) {
    Tensor t({3}, {1.0, 2.0, 3.0});
    Tensor r = t.apply([](double x){ return x * x + 1.0; });
    EXPECT_DOUBLE_EQ(r(0), 2.0);
    EXPECT_DOUBLE_EQ(r(1), 5.0);
    EXPECT_DOUBLE_EQ(r(2), 10.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Linear algebra
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorLinAlg, Matmul) {
    // [1 2]   [5 6]   [19 22]
    // [3 4] × [7 8] = [43 50]
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor b({2, 2}, {5, 6, 7, 8});
    Tensor r = a.matmul(b);
    ASSERT_EQ(r.shape(), (Tensor::Shape{2, 2}));
    EXPECT_NEAR(r(0, 0), 19.0, kEps);
    EXPECT_NEAR(r(0, 1), 22.0, kEps);
    EXPECT_NEAR(r(1, 0), 43.0, kEps);
    EXPECT_NEAR(r(1, 1), 50.0, kEps);
}

TEST(TensorLinAlg, MatmulNonSquare) {
    // (2×3) × (3×2) = (2×2)
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor r = a.matmul(b);
    ASSERT_EQ(r.shape(), (Tensor::Shape{2, 2}));
    EXPECT_NEAR(r(0, 0), 58.0, kEps);
    EXPECT_NEAR(r(1, 1), 154.0, kEps);
}

TEST(TensorLinAlg, MatmulDimMismatchThrows) {
    Tensor a({2, 3});
    Tensor b({2, 3});
    EXPECT_THROW(a.matmul(b), std::invalid_argument);
}

TEST(TensorLinAlg, Trace) {
    Tensor t({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_DOUBLE_EQ(t.trace(), 15.0);   // 1 + 5 + 9
}

TEST(TensorLinAlg, TraceNonSquareThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.trace(), std::invalid_argument);
}

TEST(TensorLinAlg, DiagVectorToMatrix) {
    Tensor v({3}, {1.0, 2.0, 3.0});
    Tensor m = v.diag();
    ASSERT_EQ(m.shape(), (Tensor::Shape{3, 3}));
    EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(m(2, 2), 3.0);
    EXPECT_DOUBLE_EQ(m(0, 1), 0.0);
}

TEST(TensorLinAlg, DiagMatrixToVector) {
    Tensor m({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor d = m.diag();
    ASSERT_EQ(d.shape(), (Tensor::Shape{3}));
    EXPECT_DOUBLE_EQ(d(0), 1.0);
    EXPECT_DOUBLE_EQ(d(1), 5.0);
    EXPECT_DOUBLE_EQ(d(2), 9.0);
}

// ════════════════════════════════════════════════════════════════════════════
// Output
// ════════════════════════════════════════════════════════════════════════════

TEST(TensorOutput, StreamOperator) {
    Tensor t({2, 2}, {1, 2, 3, 4});
    std::ostringstream oss;
    oss << t;
    EXPECT_FALSE(oss.str().empty());
    EXPECT_NE(oss.str().find("Tensor"), std::string::npos);
}
