#include "ModelSelection.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace SharedMath::ML {

// ─────────────────────────────────────────────────────────────────────────────
// KFold
// ─────────────────────────────────────────────────────────────────────────────

KFold::KFold(size_t n_splits, bool shuffle, uint64_t seed)
    : m_n_splits(n_splits), m_shuffle(shuffle), m_seed(seed)
{
    if (n_splits < 2)
        throw std::invalid_argument("KFold: n_splits must be >= 2");
}

std::vector<Fold> KFold::split(size_t n_samples) const {
    if (n_samples < m_n_splits)
        throw std::invalid_argument("KFold::split: n_samples < n_splits");

    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    if (m_shuffle) {
        std::mt19937_64 rng(m_seed);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<Fold> folds(m_n_splits);
    for (size_t k = 0; k < m_n_splits; ++k) {
        // Distribute samples as evenly as possible
        size_t start = k * n_samples / m_n_splits;
        size_t end   = (k + 1) * n_samples / m_n_splits;

        folds[k].val_indices.reserve(end - start);
        for (size_t i = start; i < end; ++i)
            folds[k].val_indices.push_back(indices[i]);

        folds[k].train_indices.reserve(n_samples - (end - start));
        for (size_t i = 0; i < start; ++i)
            folds[k].train_indices.push_back(indices[i]);
        for (size_t i = end; i < n_samples; ++i)
            folds[k].train_indices.push_back(indices[i]);
    }
    return folds;
}

size_t KFold::n_splits() const noexcept { return m_n_splits; }

// ─────────────────────────────────────────────────────────────────────────────
// StratifiedKFold
// ─────────────────────────────────────────────────────────────────────────────

StratifiedKFold::StratifiedKFold(size_t n_splits, bool shuffle, uint64_t seed)
    : m_n_splits(n_splits), m_shuffle(shuffle), m_seed(seed)
{
    if (n_splits < 2)
        throw std::invalid_argument("StratifiedKFold: n_splits must be >= 2");
}

std::vector<Fold> StratifiedKFold::split(const Tensor& labels) const {
    if (labels.ndim() != 1)
        throw std::invalid_argument("StratifiedKFold::split: labels must be 1-D");

    const size_t N = labels.size();
    if (N < m_n_splits)
        throw std::invalid_argument("StratifiedKFold::split: n_samples < n_splits");

    // Group indices by class
    size_t n_classes = 0;
    for (size_t i = 0; i < N; ++i) {
        size_t c = static_cast<size_t>(labels.flat(i)) + 1;
        if (c > n_classes) n_classes = c;
    }

    std::vector<std::vector<size_t>> by_class(n_classes);
    for (size_t i = 0; i < N; ++i)
        by_class[static_cast<size_t>(labels.flat(i))].push_back(i);

    if (m_shuffle) {
        std::mt19937_64 rng(m_seed);
        for (auto& cls_idx : by_class)
            std::shuffle(cls_idx.begin(), cls_idx.end(), rng);
    }

    // Round-robin assignment into folds
    std::vector<std::vector<size_t>> fold_idx(m_n_splits);
    for (auto& cls_idx : by_class) {
        for (size_t i = 0; i < cls_idx.size(); ++i)
            fold_idx[i % m_n_splits].push_back(cls_idx[i]);
    }

    std::vector<Fold> folds(m_n_splits);
    for (size_t k = 0; k < m_n_splits; ++k) {
        folds[k].val_indices = fold_idx[k];
        for (size_t j = 0; j < m_n_splits; ++j) {
            if (j == k) continue;
            for (size_t idx : fold_idx[j])
                folds[k].train_indices.push_back(idx);
        }
    }
    return folds;
}

size_t StratifiedKFold::n_splits() const noexcept { return m_n_splits; }

} // namespace SharedMath::ML
