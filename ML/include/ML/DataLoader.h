#pragma once

/// SharedMath::ML — Dataset and DataLoader
///
/// TensorDataset  — pairs of (X, y) tensors split into individual samples
/// DataLoader     — iterates over a TensorDataset in (optionally shuffled) batches

#include <sharedmath_ml_export.h>

#include "LinearAlgebra/Tensor.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace SharedMath::ML {

using Tensor = SharedMath::LinearAlgebra::Tensor;

class SHAREDMATH_ML_EXPORT TensorDataset {
public:
    TensorDataset(Tensor X, Tensor y);

    size_t size() const noexcept;
    const Tensor& X() const noexcept;
    const Tensor& y() const noexcept;

    std::pair<Tensor, Tensor> get(size_t i) const;

private:
    Tensor m_X;
    Tensor m_y;
};

class SHAREDMATH_ML_EXPORT DataLoader {
public:
    struct Batch {
        Tensor X;
        Tensor y;
    };

    DataLoader(const TensorDataset& dataset,
               size_t batch_size,
               bool shuffle = false,
               uint64_t seed = 0);

    class SHAREDMATH_ML_EXPORT iterator {
    public:
        iterator(const DataLoader* dl, size_t batch_idx);

        bool operator!=(const iterator& o) const;
        iterator& operator++();
        Batch operator*() const;

    private:
        const DataLoader* m_dl;
        size_t m_batch_idx;
    };

    iterator begin() const;
    iterator end() const;

    size_t numBatches() const noexcept;
    size_t batch_size() const noexcept;

    void resetEpoch();
    Batch getBatch(size_t batch_idx) const;

private:
    const TensorDataset& m_dataset;
    size_t              m_batch_size;
    bool                m_shuffle;
    uint64_t            m_seed;
    std::vector<size_t> m_indices;

    void rebuildIndices();
};

} // namespace SharedMath::ML
