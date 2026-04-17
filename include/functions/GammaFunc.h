#pragma once

#include "../constans.h"
#include <sharedmath_export.h>
#include <complex>

namespace SharedMath
{
    namespace Functions
    {
        class SHAREDMATH_EXPORT GammaFunction{
            public:
                static double value(double x);
                static std::complex<double> value(const std::complex<double>& z);

                static double ln_gamma(double x);
                static std::complex<double> ln_gamma(const std::complex<double>& z);

                static bool is_pole(double x);
                static bool is_pole(const std::complex<double>& z);

                static double digamma(double x);
                static std::complex<double> digamma(const std::complex<double>& z);

            private:
                static std::complex<double> gamma_lanczos(const std::complex<double>& z);
                static std::complex<double> gamma_stirling(const std::complex<double>& z);

                static std::complex<double> gamma_reflection(const std::complex<double>& z);
                static std::complex<double> lanczos_core(const std::complex<double>& z);

                static std::complex<double> stirling_with_recursion(const std::complex<double>& z, double threshold);
                static std::complex<double> stirling_core(const std::complex<double>& z);
                static std::complex<double> ln_gamma_stirling(const std::complex<double>& z);

                static std::complex<double> handle_pole(const std::complex<double>& z);

                static bool is_near_pole(const std::complex<double>& z);
                static std::complex<double> gamma_small_z(const std::complex<double>& z);
        };
        
    } // namespace Functions
    
} // namespace SharedMath
