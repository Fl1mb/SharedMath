#include <gtest/gtest.h>
#include <ML/ml.h>
#include <numeric>

using namespace SharedMath::ML;
using Tensor = SharedMath::LinearAlgebra::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// KFold
// ─────────────────────────────────────────────────────────────────────────────

TEST(KFold, FoldsHaveCorrectSizes) {
    KFold kf(5);
    auto folds = kf.split(20);
    EXPECT_EQ(folds.size(), 5u);
    for (const auto& f : folds) {
        EXPECT_EQ(f.train_indices.size() + f.val_indices.size(), 20u);
        EXPECT_EQ(f.val_indices.size(), 4u);
    }
}

TEST(KFold, NoOverlapBetweenTrainAndVal) {
    KFold kf(4);
    auto folds = kf.split(12);
    for (const auto& f : folds) {
        for (size_t v : f.val_indices) {
            bool in_train = false;
            for (size_t t : f.train_indices)
                if (t == v) { in_train = true; break; }
            EXPECT_FALSE(in_train);
        }
    }
}

TEST(KFold, AllIndicesCovered) {
    const size_t N = 15;
    KFold kf(3);
    auto folds = kf.split(N);
    // Union of all val_indices should cover [0, N)
    std::vector<bool> seen(N, false);
    for (const auto& f : folds)
        for (size_t v : f.val_indices) seen[v] = true;
    for (size_t i = 0; i < N; ++i)
        EXPECT_TRUE(seen[i]) << "index " << i << " never in val";
}

TEST(KFold, ShuffleReproducible) {
    KFold kf1(3, true, 42);
    KFold kf2(3, true, 42);
    auto f1 = kf1.split(9);
    auto f2 = kf2.split(9);
    for (size_t k = 0; k < 3; ++k)
        for (size_t i = 0; i < f1[k].val_indices.size(); ++i)
            EXPECT_EQ(f1[k].val_indices[i], f2[k].val_indices[i]);
}

TEST(KFold, TooFewSamplesThrows) {
    KFold kf(5);
    EXPECT_THROW(kf.split(3), std::invalid_argument);
}

// ─────────────────────────────────────────────────────────────────────────────
// StratifiedKFold
// ─────────────────────────────────────────────────────────────────────────────

TEST(StratifiedKFold, BalancedClasses) {
    // 20 samples: 10 class-0, 10 class-1
    Tensor y = Tensor::zeros({20});
    for (size_t i = 0; i < 10; ++i) y.flat(i) = 0.0;
    for (size_t i = 10; i < 20; ++i) y.flat(i) = 1.0;

    StratifiedKFold skf(5);
    auto folds = skf.split(y);
    EXPECT_EQ(folds.size(), 5u);

    // Each val fold should have roughly equal class proportions
    for (const auto& f : folds) {
        size_t c0 = 0, c1 = 0;
        for (size_t v : f.val_indices) {
            if (v < 10) ++c0; else ++c1;
        }
        // Should have ~2 from each class per fold
        EXPECT_GE(c0, 1u);
        EXPECT_GE(c1, 1u);
    }
}

TEST(StratifiedKFold, AllIndicesCovered) {
    Tensor y = Tensor::zeros({12});
    for (size_t i = 0; i < 6; ++i) y.flat(i) = 0.0;
    for (size_t i = 6; i < 12; ++i) y.flat(i) = 1.0;

    StratifiedKFold skf(3);
    auto folds = skf.split(y);
    std::vector<bool> seen(12, false);
    for (const auto& f : folds)
        for (size_t v : f.val_indices) seen[v] = true;
    for (size_t i = 0; i < 12; ++i)
        EXPECT_TRUE(seen[i]);
}

// ─────────────────────────────────────────────────────────────────────────────
// cross_val_score with KNN
// ─────────────────────────────────────────────────────────────────────────────

TEST(CrossValScore, KNNOnEasyDataset) {
    // Two easily separable clusters
    const size_t N = 40;
    Tensor X = Tensor::zeros({N, 2});
    Tensor y = Tensor::zeros({N});
    for (size_t i = 0; i < N/2; ++i) {
        X(i,0) = -5.0; X(i,1) = 0.0; y.flat(i) = 0.0;
    }
    for (size_t i = N/2; i < N; ++i) {
        X(i,0) =  5.0; X(i,1) = 0.0; y.flat(i) = 1.0;
    }

    KFold kf(4, true, 0);
    auto folds = kf.split(N);

    using ModelPtr = std::shared_ptr<KNNClassifier>;
    auto train_fn = [](const Tensor& Xtr, const Tensor& ytr) -> ModelPtr {
        auto m = std::make_shared<KNNClassifier>(3);
        m->fit(Xtr, ytr);
        return m;
    };
    auto score_fn = [](ModelPtr& m, const Tensor& Xv, const Tensor& yv) {
        return accuracy(m->predict(Xv), yv);
    };

    auto scores = cross_val_score<ModelPtr>(X, y, folds, train_fn, score_fn);
    EXPECT_EQ(scores.size(), 4u);
    for (double s : scores)
        EXPECT_GE(s, 0.9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Extended Metrics
// ─────────────────────────────────────────────────────────────────────────────

TEST(RegressionMetrics, MSE) {
    Tensor pred = Tensor::from_vector({1.0, 2.0, 3.0});
    Tensor true_ = Tensor::from_vector({1.0, 2.0, 3.0});
    EXPECT_NEAR(mean_squared_error(pred, true_), 0.0, 1e-9);

    Tensor pred2 = Tensor::from_vector({0.0, 0.0, 0.0});
    EXPECT_NEAR(mean_squared_error(pred2, true_), (1+4+9)/3.0, 1e-9);
}

TEST(RegressionMetrics, MAE) {
    Tensor pred = Tensor::from_vector({1.0, 3.0, 5.0});
    Tensor true_ = Tensor::from_vector({0.0, 2.0, 4.0});
    EXPECT_NEAR(mean_absolute_error(pred, true_), 1.0, 1e-9);
}

TEST(RegressionMetrics, R2PerfectFit) {
    Tensor pred = Tensor::from_vector({1.0, 2.0, 3.0, 4.0});
    Tensor true_ = Tensor::from_vector({1.0, 2.0, 3.0, 4.0});
    EXPECT_NEAR(r2_score(pred, true_), 1.0, 1e-9);
}

TEST(RegressionMetrics, R2ConstantPrediction) {
    Tensor true_ = Tensor::from_vector({1.0, 2.0, 3.0, 4.0});
    Tensor pred  = Tensor::from_vector({2.5, 2.5, 2.5, 2.5});  // mean
    // R² of predicting the mean is 0
    EXPECT_NEAR(r2_score(pred, true_), 0.0, 1e-9);
}

TEST(ClassificationMetrics, MacroF1) {
    Tensor pred = Tensor::from_vector({0.0, 1.0, 2.0, 0.0, 1.0, 2.0});
    Tensor true_ = Tensor::from_vector({0.0, 1.0, 2.0, 1.0, 2.0, 0.0});
    double mf1 = macro_f1_score(pred, true_, 3);
    EXPECT_GT(mf1, 0.0);
    EXPECT_LE(mf1, 1.0);
}

TEST(ClassificationMetrics, MicroF1PerfectPredictions) {
    Tensor pred = Tensor::from_vector({0.0, 1.0, 2.0});
    Tensor true_ = Tensor::from_vector({0.0, 1.0, 2.0});
    EXPECT_NEAR(micro_f1_score(pred, true_, 3), 1.0, 1e-9);
}

TEST(ClusteringMetrics, SilhouettePerfectClusters) {
    // Two well-separated clusters in 1-D
    Tensor X = Tensor::zeros({6, 1});
    Tensor lbl = Tensor::zeros({6});
    X(0,0)=-10; X(1,0)=-11; X(2,0)=-9;
    X(3,0)= 10; X(4,0)= 11; X(5,0)=  9;
    lbl.flat(0)=0; lbl.flat(1)=0; lbl.flat(2)=0;
    lbl.flat(3)=1; lbl.flat(4)=1; lbl.flat(5)=1;

    double s = silhouette_score(X, lbl);
    EXPECT_GT(s, 0.8);
}
