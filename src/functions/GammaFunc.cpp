#include "../../include/functions/GammaFunc.h"
#include <cmath>

using namespace SharedMath::Functions;

double GammaFunction::value(double x){
    if(std::abs(x) < Epsilon)return 1.0;
    auto result = value({x, 0.0});
    return std::real(result);
}

std::complex<double> GammaFunction::value(const std::complex<double>& z){
    double re = std::real(z);
    double im = std::imag(z);
    double abs_z = std::abs(z);

    if (is_pole(z)) {
        return handle_pole(z);
    }

    if (abs_z < 1e-8) {
        return gamma_small_z(z);
    }
    
    if (re < 0.0 && !is_near_pole(z)) {
        if (re < -50.0) {
            return std::exp(ln_gamma_stirling(z));
        }
        return gamma_reflection(z);
    }

    if (re >= 0.0) {
        if (abs_z > 50.0) {
            return std::exp(ln_gamma_stirling(z));
        }

        if (abs_z > 10.0) {
            return gamma_stirling(z);
        }
        
        if (abs_z >= 5.0) {
            if (re < 0.5) {
                return gamma_reflection(z);
            }
            return gamma_lanczos(z);
        }
        
        if (abs_z >= 0.5) {
            if (re < 0.5) {
                return gamma_reflection(z);
            }
            return gamma_lanczos(z);
        }
        
        return gamma_small_z(z);
    }
    
    return gamma_lanczos(z);
}

double GammaFunction::ln_gamma(double x){
    return std::real(ln_gamma_stirling({x, 0.0}));
}

std::complex<double> GammaFunction::ln_gamma(const std::complex<double>& z){
    return ln_gamma_stirling(z);
}

double GammaFunction::digamma(double x){
    const double h = 1e-8;
    return (ln_gamma(x + h) - ln_gamma(x - h)) / (2.0 * h);

}

std::complex<double> 
GammaFunction::digamma(const std::complex<double>& z){
    const double h = 1e-8;
    return (ln_gamma(z + h) - ln_gamma(z - h)) / (2.0 * h);
}

bool GammaFunction::is_pole(double x){
    return is_pole({x, 0.0});
}

bool GammaFunction::is_pole(const std::complex<double>& z){
    double re = z.real();
    double im = z.imag();

    if(re <= 0.0 && abs(im) < Epsilon){
        double int_part = floor(re);
        if(abs(re - int_part) < Epsilon){
            return true;
        }
    }
    return false;
}


std::complex<double> 
GammaFunction::gamma_lanczos(const std::complex<double>& z){
    if(std::real(z) < 0.5){
        return gamma_reflection(z);
    }return lanczos_core(z - 1.0);
}

std::complex<double> 
GammaFunction::gamma_reflection(const std::complex<double>& z){
    std::complex<double> w = 1.0 - z;
    std::complex<double> sin_pi_z = std::sin(Pi * z);

    if (std::abs(sin_pi_z) < 1e-15) {
        if (std::abs(std::imag(z)) < 1e-12 && 
            std::abs(z - std::floor(std::real(z))) < 1e-12) {
            return std::complex<double>(
                std::numeric_limits<double>::infinity(),
                0.0
            );
        }
    }

    std::complex<double> gamma_1_minus_z = lanczos_core(w - 1.0);
    return Pi / (sin_pi_z * gamma_1_minus_z);
}



std::complex<double> 
GammaFunction::lanczos_core(const std::complex<double>& z){
    std::complex<double> agg = LANCZOS_COEFFS[0];
    
    for (int i = 1; i < LANCZOS_N; ++i) {
        agg += LANCZOS_COEFFS[i] / (z + static_cast<double>(i));
    }
    
    std::complex<double> t = z + LANCZOS_G + 0.5;
    
    std::complex<double> power_base = z + 0.5;
    std::complex<double> power_term = power_base * std::log(t);
    
    std::complex<double> result = SQRT_2PI * 
                                    std::exp(power_term - t) * 
                                    agg;
    
    return result;
}

std::complex<double> 
GammaFunction::gamma_stirling(const std::complex<double>& z){
    constexpr double STIRLING_THRESHOLD = 10.0;

    if (std::abs(z) < STIRLING_THRESHOLD) {
        return stirling_with_recursion(z, STIRLING_THRESHOLD);
    }

    return stirling_core(z);
}


std::complex<double> 
GammaFunction::stirling_with_recursion(const std::complex<double>& z, double threshold){
    int n = 0;
    std::complex<double> current_z = z;
    std::complex<double> product = 1.0;

    while (std::abs(current_z) < threshold) {
        product /= current_z;
        current_z += 1.0;
        n++;
    }

    std::complex<double> gamma_large = stirling_core(current_z);

    return gamma_large * product;
}


std::complex<double> 
GammaFunction::stirling_core(const std::complex<double>& z){
    static constexpr std::array<double, 8> STIRLING_COEFFS = {
        1.0 / 12.0,
        1.0 / 288.0,
        -139.0 / 51840.0,
        -571.0 / 2488320.0,
        163879.0 / 209018880.0,
        5246819.0 / 75246796800.0,
        -534703531.0 / 902961561600.0,
        -4483131259.0 / 86684309913600.0
    };

    std::complex<double> log_z = std::log(z);
    std::complex<double> power_term = (z - 0.5) * log_z;

    std::complex<double> series = 1.0;
    std::complex<double> z_inv = 1.0 / z;
    std::complex<double> z_power = z_inv;

    for (size_t i = 0; i < STIRLING_COEFFS.size(); ++i) {
        series += STIRLING_COEFFS[i] * z_power;
        z_power *= z_inv;
    }

    std::complex<double> result = SQRT_2PI * 
                                std::exp(power_term - z) * 
                                series;

    return result;
}

std::complex<double> 
GammaFunction::handle_pole(const std::complex<double>& z){
    double re = z.real();

    int n = static_cast<int>(std::floor(-re + 0.5));
    double sign = (n % 2 == 0) ? 1.0 : -1.0;

    return std::complex<double>(
        sign * std::numeric_limits<double>::infinity(),
        0.0
    );
}

bool GammaFunction::is_near_pole(const std::complex<double>& z){
    double re = std::real(z);
    double im = std::imag(z);

    if (re <= 0.0 && std::abs(im) < Epsilon) {
        double nearest_int = std::round(re);
        if (std::abs(re - nearest_int) < 1e-6) {
            return true;
        }
    }
    return false;
}

std::complex<double> 
GammaFunction::gamma_small_z(const std::complex<double>& z){
    static constexpr std::array<double, 10> SMALL_Z_COEFFS = {
        -EULER_GAMMA,                                    // коэффициент при z^1
        0.9890559953279725553953956515006347079392,     // z^2: γ²/2 + π²/12
        -0.9074790760808862890165601673562751149282,    // z^3
        0.9817280868344001873363802940218508503600,     // z^4
        -0.9819950689031452112162446005763604481908,    // z^5
        0.9931491146212761937612223841897123482056,     // z^6
        -0.9960017593664728863568677628423164271089,    // z^7
        0.9981056937831298796285618666855256744810,     // z^8
        -0.9990252676219490398845760724957689143878,    // z^9
        0.9995156560727770438190362998680147323444      // z^10
    };

    if (std::abs(z) < 1e-15) {
        return 1.0 / z - EULER_GAMMA;
    }

    std::complex<double> result = 1.0 / z - EULER_GAMMA;
    std::complex<double> z_power = z;
    
    for (size_t i = 0; i < SMALL_Z_COEFFS.size(); ++i) {
        result += SMALL_Z_COEFFS[i] * z_power;
        z_power *= z;
        if (std::abs(z_power) < 1e-20) {
            break;
        }
    }
    
    return result;
}

std::complex<double> GammaFunction::ln_gamma_stirling(const std::complex<double>& z) {
    constexpr double STIRLING_LN_THRESHOLD = 10.0;
    
    if (std::abs(z) < STIRLING_LN_THRESHOLD) {
        if (is_near_pole(z)) {
            return std::complex<double>(
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN()
            );
        }
        return std::log(value(z));
    }

    static constexpr std::array<double, 8> STIRLING_LN_COEFFS = {
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360360.0,
        1.0 / 156.0,
        -3617.0 / 122400.0
    };
    
    std::complex<double> log_z = std::log(z);
    std::complex<double> series = 0.0;
    std::complex<double> z_inv = 1.0 / z;
    std::complex<double> z_power = z_inv;
    
    for (size_t i = 0; i < STIRLING_LN_COEFFS.size(); ++i) {
        series += STIRLING_LN_COEFFS[i] * z_power;
        z_power *= z_inv;
    }
    std::complex<double> result = (z - 0.5) * log_z - z + 
                                  0.5 * std::log(2.0 * Pi) + 
                                  series;
    
    return result;
}