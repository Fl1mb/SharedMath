#pragma once
#include "DynamicMatrix.h"
#include <sharedmath_export.h>

namespace SharedMath::LinearAlgebra
{

    class SHAREDMATH_EXPORT LUDecomposition
    {
    private:
        DynamicMatrix L;
        DynamicMatrix U;
        
        std::vector<size_t> pivot;
        DynamicMatrix matrix_;
        bool decomposed = false;
        bool singular = false;

    public:
        LUDecomposition() = default;
        ~LUDecomposition() = default;
        explicit LUDecomposition(const DynamicMatrix& matrix): matrix_(matrix){
            if(matrix.rows() != matrix.cols()){
                throw std::invalid_argument("Matrix must be squared");
            }
        }

        void SetMatrixToDecompose(const DynamicMatrix& matrix);
        void MakeDecomposition();

        const DynamicMatrix& GetL() const;
        const DynamicMatrix& GetU() const;
        const std::vector<size_t>& GetPivot() const;
        
        DynamicMatrix GetPermutationMatrix() const;

        double Determinant() const;

        bool VerifyDecomposition(double tolerance = 1e-10) const;
        bool IsDecomposed() const;

        void Clear();

    };


} // namespace SharedMath::LinearAlgebra
