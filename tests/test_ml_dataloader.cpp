#include <gtest/gtest.h>
#include "ML/ml.h"

#include <vector>
#include <numeric>
#include <set>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// TensorDataset
// ─────────────────────────────────────────────────────────────────────────────

TEST(TensorDataset, SizeAndShapes) {
    Tensor X = Tensor::ones({10, 4});
    Tensor y = Tensor::zeros({10});
    TensorDataset ds(X, y);
    EXPECT_EQ(ds.size(), 10u);
    EXPECT_EQ(ds.X().shape(), (Tensor::Shape{10, 4}));
    EXPECT_EQ(ds.y().shape(), (Tensor::Shape{10}));
}

TEST(TensorDataset, GetReturnsRow) {
    // X[i, :] = i+1; y[i] = i
    const size_t N = 5, D = 3;
    Tensor X = Tensor::zeros({N, D});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) {
        y.flat(i) = static_cast<double>(i);
        for (size_t d = 0; d < D; ++d)
            X(i, d) = static_cast<double>(i + 1);
    }

    TensorDataset ds(X, y);
    auto [xi, yi] = ds.get(2);
    EXPECT_EQ(xi.shape(), (Tensor::Shape{1, D}));
    for (size_t d = 0; d < D; ++d)
        EXPECT_DOUBLE_EQ(xi(0, d), 3.0);
    EXPECT_DOUBLE_EQ(yi.flat(0), 2.0);
}

TEST(TensorDataset, GetOutOfRangeThrows) {
    TensorDataset ds(Tensor::ones({5, 2}), Tensor::zeros({5}));
    EXPECT_THROW(ds.get(5), std::out_of_range);
}

TEST(TensorDataset, ShapeMismatchThrows) {
    Tensor X = Tensor::ones({5, 2});
    Tensor y = Tensor::zeros({4});   // wrong N
    EXPECT_THROW(TensorDataset(X, y), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// DataLoader — basic batching
// ─────────────────────────────────────────────────────────────────────────────

TEST(DataLoader, CorrectNumberOfBatches) {
    TensorDataset ds(Tensor::ones({10, 2}), Tensor::zeros({10}));
    DataLoader loader(ds, 3);               // 10 / 3 = 4 batches (3+3+3+1)
    EXPECT_EQ(loader.numBatches(), 4u);
}

TEST(DataLoader, BatchSizes) {
    TensorDataset ds(Tensor::ones({7, 2}), Tensor::zeros({7}));
    DataLoader loader(ds, 3);

    std::vector<size_t> sizes;
    for (auto batch : loader)
        sizes.push_back(batch.X.dim(0));

    EXPECT_EQ(sizes.size(), 3u);
    EXPECT_EQ(sizes[0], 3u);
    EXPECT_EQ(sizes[1], 3u);
    EXPECT_EQ(sizes[2], 1u);
}

TEST(DataLoader, AllSamplesVisited) {
    // y[i] = i; verify sum over all batches equals N*(N-1)/2
    const size_t N = 12;
    Tensor X = Tensor::ones({N, 2});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) y.flat(i) = static_cast<double>(i);

    TensorDataset ds(X, y);
    DataLoader loader(ds, 4, /*shuffle=*/false);

    double total = 0.0;
    for (auto batch : loader)
        for (size_t i = 0; i < batch.y.size(); ++i)
            total += batch.y.flat(i);

    EXPECT_DOUBLE_EQ(total, static_cast<double>(N * (N - 1) / 2));
}

TEST(DataLoader, ShuffleChangesOrder) {
    const size_t N = 20;
    Tensor X = Tensor::ones({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) y.flat(i) = static_cast<double>(i);

    TensorDataset ds(X, y);
    DataLoader shuffled(ds, N, /*shuffle=*/true, /*seed=*/42);

    // First batch of size N with shuffle should be a permutation (not 0,1,...,N-1)
    auto batch = shuffled.getBatch(0);
    bool in_order = true;
    for (size_t i = 0; i < N; ++i)
        if (static_cast<size_t>(batch.y.flat(i)) != i) { in_order = false; break; }

    EXPECT_FALSE(in_order);  // with high probability, shuffled ≠ original order
}

TEST(DataLoader, ShuffleCoversAllSamples) {
    const size_t N = 15;
    Tensor X = Tensor::ones({N, 1});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N; ++i) y.flat(i) = static_cast<double>(i);

    TensorDataset ds(X, y);
    DataLoader loader(ds, 3, /*shuffle=*/true, 0);

    std::set<int> seen;
    for (auto batch : loader)
        for (size_t i = 0; i < batch.y.size(); ++i)
            seen.insert(static_cast<int>(batch.y.flat(i)));

    EXPECT_EQ(seen.size(), N);
}

TEST(DataLoader, ZeroBatchSizeThrows) {
    TensorDataset ds(Tensor::ones({5, 2}), Tensor::zeros({5}));
    EXPECT_THROW(DataLoader(ds, 0), std::invalid_argument);
}

TEST(DataLoader, BatchShapeConsistency) {
    const size_t N = 8, D = 3;
    TensorDataset ds(Tensor::ones({N, D}), Tensor::zeros({N}));
    DataLoader loader(ds, 4);
    for (auto batch : loader) {
        EXPECT_EQ(batch.X.ndim(), 2u);
        EXPECT_EQ(batch.X.dim(1), D);
        EXPECT_EQ(batch.y.dim(0), batch.X.dim(0));
    }
}
