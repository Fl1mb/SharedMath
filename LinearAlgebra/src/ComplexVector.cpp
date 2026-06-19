#include "LinearAlgebra/ComplexVector.h"
#include <iostream>

namespace SharedMath::LinearAlgebra {

std::ostream& operator<<(std::ostream& os, const ComplexVector& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) os << ", ";
        os << v[i];
    }
    os << "]";
    return os;
}

ComplexVector operator*(Complex s, const ComplexVector& v) {
    return v * s;
}

} // namespace SharedMath::LinearAlgebra
