#pragma once

#include <algorithm>
#include <cmath>

namespace tmv::milbrandt_detail
{
// Cohard-Pinty (2000) continental CCN activation polynomial used by the
// operational Milbrandt-Yau double-moment scheme. Inputs are m/s, K, Pa;
// result is activated cloud-droplet concentration in m-3.
inline double continental_activated_ccn(double vertical_velocity, double temperature_k, double pressure_pa)
{
    if (!(vertical_velocity > 0.001) || !std::isfinite(vertical_velocity) || !std::isfinite(temperature_k) ||
        !std::isfinite(pressure_pa))
    {
        return 2.0e8;
    }
    const double x             = std::log10(vertical_velocity * 100.0);
    const double x2            = x * x;
    const double x3            = x2 * x;
    const double x4            = x2 * x2;
    const double t             = temperature_k - 273.15;
    const double t2            = t * t;
    const double t3            = t2 * t;
    const double t4            = t2 * t2;
    const double pressure_hpa  = pressure_pa * 0.01;
    const double pressure_hpa2 = pressure_hpa * pressure_hpa;

    const double c = -2.112e-9 * t4 + 3.9836e-8 * t3 + 2.3703e-6 * t2 - 1.4542e-4 * t - 0.0698;
    const double d = -4.210e-8 * t4 + 5.5745e-7 * t3 + 1.8460e-5 * t2 + 9.6078e-4 * t + 0.7120;
    const double e = 1.434e-7 * t4 - 1.6455e-6 * t3 - 4.3334e-5 * t2 - 7.6720e-3 * t + 1.0056;
    const double f = 1.340e-6 * pressure_hpa2 - 3.5114e-3 * pressure_hpa + 1.9453;
    const double g = 4.226e-3 * x4 - 5.6012e-3 * x3 - 8.7846e-2 * x2 + 2.7435e-2 * x + 0.9932;
    const double h = 5.811e-9 * t4 + 1.5589e-7 * t3 - 3.8623e-5 * t2 + 1.4471e-3 * t + 0.1496;
    const double y = c * x2 + d * x + e + f * g * h;
    return std::pow(10.0, std::max(0.0, y)) * 1.0e6;
}

} // namespace tmv::milbrandt_detail
