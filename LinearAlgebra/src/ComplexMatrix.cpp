#include "LinearAlgebra/ComplexMatrix.h"

namespace SharedMath::LinearAlgebra {

ComplexMatrix ComplexMatrix::operator+(const ComplexMatrix& o) const {
    checkSameShape(o);
    ComplexMatrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i)
        result.data_[i] = data_[i] + o.data_[i];
    return result;
}

ComplexMatrix ComplexMatrix::operator-(const ComplexMatrix& o) const {
    checkSameShape(o);
    ComplexMatrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i)
        result.data_[i] = data_[i] - o.data_[i];
    return result;
}

ComplexMatrix ComplexMatrix::operator*(Complex s) const {
    ComplexMatrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i)
        result.data_[i] = data_[i] * s;
    return result;
}

ComplexMatrix ComplexMatrix::operator*(const ComplexMatrix& o) const {
    if (cols_ != o.rows_)
        throw std::invalid_argument("ComplexMatrix: dimension mismatch for multiplication");

    ComplexMatrix result(rows_, o.cols_, Complex(0.0, 0.0));
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t k = 0; k < cols_; ++k) {
            Complex aik = at_unsafe(i, k);
            for (size_t j = 0; j < o.cols_; ++j)
                result.at_unsafe(i, j) += aik * o.at_unsafe(k, j);
        }
    }
    return result;
}

ComplexMatrix& ComplexMatrix::operator+=(const ComplexMatrix& o) {
    checkSameShape(o);
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] += o.data_[i];
    return *this;
}

ComplexMatrix& ComplexMatrix::operator-=(const ComplexMatrix& o) {
    checkSameShape(o);
    for (size_t i = 0; i < data_.size(); ++i)
        data_[i] -= o.data_[i];
    return *this;
}

ComplexMatrix& ComplexMatrix::operator*=(Complex s) {
    for (auto& v : data_) v *= s;
    return *this;
}

std::vector<double> real(const ComplexMatrix& m) {
    std::vector<double> result(m.size());
    for (size_t i = 0; i < m.size(); ++i)
        result[i] = m.data()[i].real();
    return result;
}

std::vector<double> imag(const ComplexMatrix& m) {
    std::vector<double> result(m.size());
    for (size_t i = 0; i < m.size(); ++i)
        result[i] = m.data()[i].imag();
    return result;
}

} // namespace SharedMath::LinearAlgebra
