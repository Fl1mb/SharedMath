#include "DataLoader.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace SharedMath::ML {

TensorDataset::TensorDataset(Tensor X, Tensor y)
    : m_X(std::move(X)),
      m_y(std::move(y))
{
    if (m_X.ndim() < 1 || m_y.ndim() < 1)
        throw std::invalid_argument("TensorDataset: tensors must be non-empty");
    if (m_X.dim(0) != m_y.dim(0))
        throw std::invalid_argument(
            "TensorDataset: X and y must have the same number of samples (dim 0)");
}

size_t TensorDataset::size() const noexcept { return m_X.dim(0); }
const Tensor& TensorDataset::X() const noexcept { return m_X; }
const Tensor& TensorDataset::y() const noexcept { return m_y; }

std::pair<Tensor, Tensor> TensorDataset::get(size_t i) const {
    if (i >= size())
        throw std::out_of_range("TensorDataset::get: index out of range");
    return {m_X.slice(0, i, i + 1), m_y.slice(0, i, i + 1)};
}

DataLoader::DataLoader(const TensorDataset& dataset,
                       size_t batch_size,
                       bool shuffle,
                       uint64_t seed)
    : m_dataset(dataset),
      m_batch_size(batch_size),
      m_shuffle(shuffle),
      m_seed(seed)
{
    if (batch_size == 0)
        throw std::invalid_argument("DataLoader: batch_size must be > 0");
    rebuildIndices();
}

DataLoader::iterator::iterator(const DataLoader* dl, size_t batch_idx)
    : m_dl(dl),
      m_batch_idx(batch_idx)
{}

bool DataLoader::iterator::operator!=(const iterator& o) const {
    return m_batch_idx != o.m_batch_idx;
}

DataLoader::iterator& DataLoader::iterator::operator++() {
    ++m_batch_idx;
    return *this;
}

DataLoader::Batch DataLoader::iterator::operator*() const {
    return m_dl->getBatch(m_batch_idx);
}

DataLoader::iterator DataLoader::begin() const {
    return {this, 0};
}

DataLoader::iterator DataLoader::end() const {
    return {this, numBatches()};
}

size_t DataLoader::numBatches() const noexcept {
    return (m_dataset.size() + m_batch_size - 1) / m_batch_size;
}

size_t DataLoader::batch_size() const noexcept {
    return m_batch_size;
}

void DataLoader::resetEpoch() {
    rebuildIndices();
}

DataLoader::Batch DataLoader::getBatch(size_t batch_idx) const {
    const size_t N = m_dataset.size();
    const size_t start = batch_idx * m_batch_size;
    if (start >= N)
        throw std::out_of_range("DataLoader::getBatch: batch index out of range");

    const size_t end = std::min(start + m_batch_size, N);
    const size_t bsz = end - start;

    std::vector<Tensor> xrows;
    std::vector<Tensor> yrows;
    xrows.reserve(bsz);
    yrows.reserve(bsz);

    for (size_t pos = start; pos < end; ++pos) {
        auto [xi, yi] = m_dataset.get(m_indices[pos]);
        xrows.push_back(std::move(xi));
        yrows.push_back(std::move(yi));
    }

    return {Tensor::concat(xrows, 0), Tensor::concat(yrows, 0)};
}

void DataLoader::rebuildIndices() {
    m_indices.resize(m_dataset.size());
    std::iota(m_indices.begin(), m_indices.end(), 0);
    if (m_shuffle) {
        std::mt19937_64 rng(m_seed++);
        std::shuffle(m_indices.begin(), m_indices.end(), rng);
    }
}

} // namespace SharedMath::ML
