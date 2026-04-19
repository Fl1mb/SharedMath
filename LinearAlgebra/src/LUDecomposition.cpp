#include "LUDecomposition.h"

using namespace SharedMath::LinearAlgebra;


void LUDecomposition::SetMatrixToDecompose(const DynamicMatrix& matrix){
    matrix_ = matrix;
    if(matrix.rows() != matrix.cols()){
        throw std::invalid_argument("Matrix must be squared");
    }
}

void LUDecomposition::MakeDecomposition() {
    if (matrix_.rows() == 0) {
        throw std::runtime_error("Matrix is not set for decomposition");
    }

    size_t n = matrix_.rows();
    L = DynamicMatrix(n, n);
    U = matrix_;
    pivot.resize(n);
    singular = false;

    for (size_t i = 0; i < n; ++i) {
        pivot[i] = i;
    }

    for (size_t k = 0; k < n; ++k) {
        double maxVal = 0.0;
        size_t maxIndex = k;
        
        for (size_t i = k; i < n; ++i) {
            double currentVal = std::abs(U.get(i, k));
            if (currentVal > maxVal) {
                maxVal = currentVal;
                maxIndex = i;
            }
        }

        if (maxVal < 1e-12) {
            singular = true;
            throw std::runtime_error("Matrix is singular or nearly singular");
        }

        if (maxIndex != k) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(U.get(k, j), U.get(maxIndex, j));
            }
            for (size_t j = 0; j < k; ++j) {
                std::swap(L.get(k, j), L.get(maxIndex, j));
            }

            std::swap(pivot[k], pivot[maxIndex]);
        }

        L.set(k, k, 1.0);

        for (size_t i = k + 1; i < n; ++i) {
            double factor = U.get(i, k) / U.get(k, k);
            L.set(i, k, factor);

            for (size_t j = k; j < n; ++j) {
                U.set(i, j, U.get(i, j) - factor * U.get(k, j));
            }
        }
    }

    decomposed = true;
}


const DynamicMatrix& LUDecomposition::GetL() const{
    if (!decomposed) {
        throw std::runtime_error("Decomposition has not been performed yet");
    }
    return L;
}

const DynamicMatrix& LUDecomposition::GetU() const{
    if (!decomposed) {
        throw std::runtime_error("Decomposition has not been performed yet");
    }
    return U;
}

const std::vector<size_t>& LUDecomposition::GetPivot() const{
    if (!decomposed) {
        throw std::runtime_error("Decomposition has not been performed yet");
    }
    return pivot;
}

DynamicMatrix LUDecomposition::GetPermutationMatrix() const{
    if (!decomposed) {
        throw std::runtime_error("Decomposition has not been performed yet");
    }

    size_t n = pivot.size();
    DynamicMatrix P(n, n);

    for(size_t i = 0; i < n; ++i){
        P.set(i, pivot[i], 1.0);
    }

    return P;
}

double LUDecomposition::Determinant() const{
    if(singular){
        return 0.0;
    }
    if (!decomposed) {
        throw std::runtime_error("Decomposition has not been performed yet");
    }

    size_t n = U.rows();
    double det = 1.0;

    for (size_t i = 0; i < n; ++i) {
        det *= U.get(i, i);
    }

    int inversionCount = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (pivot[i] > pivot[j]) {
                inversionCount++;
            }
        }
    }

    if (inversionCount % 2 == 1) {
        det = -det;
    }

    return det;
}

bool LUDecomposition::VerifyDecomposition(double tolerance) const{
    if (!decomposed) {
        return false;
    }

    DynamicMatrix P = GetPermutationMatrix();
    DynamicMatrix PA = P * matrix_;
    DynamicMatrix LU = L * U;

    for (size_t i = 0; i < PA.rows(); ++i) {
        for (size_t j = 0; j < PA.cols(); ++j) {
            if (std::abs(PA.get(i, j) - LU.get(i, j)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

bool LUDecomposition::IsDecomposed() const{
    return decomposed;
}

void LUDecomposition::Clear(){
    L.clear();
    U.clear();
    pivot.clear();
    decomposed = false;
    matrix_.clear();
}