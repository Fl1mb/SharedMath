#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace SharedMath::DSP {

template<typename T>
std::vector<T> fftShift(const std::vector<T>& x) {
    if (x.empty()) return {};
    size_t n    = x.size();
    size_t half = n / 2;
    std::vector<T> out(n);
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(half), x.end(), out.begin());
    std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(half),
              out.begin() + static_cast<std::ptrdiff_t>(n - half));
    return out;
}

template<typename T>
std::vector<T> ifftShift(const std::vector<T>& x) {
    if (x.empty()) return {};
    size_t n    = x.size();
    size_t half = (n + 1) / 2;
    std::vector<T> out(n);
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(half), x.end(), out.begin());
    std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(half),
              out.begin() + static_cast<std::ptrdiff_t>(n - half));
    return out;
}

} // namespace SharedMath::DSP
