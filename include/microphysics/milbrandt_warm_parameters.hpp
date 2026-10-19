#pragma once

#include "core/infra/physical_constants.hpp"
#include <cmath>

namespace tmv::milbrandt_detail
{
struct WarmRainParameters
{
    double alpha_r, c_r, d_r, a_r, b_r;
    double inv_mu_c;
    double gc1;
    double gc2;
    double gc3;
    double gc4;
    double gr31;
    double gr34;
    double gr37;
    double inv_cloud_mass_factor;
    double inv_rain_mass_factor;
    double gamma_ventilation, gamma_fall_ventilation;
};

inline WarmRainParameters warm_rain_parameters(double alpha_r, double c_r, double d_r, double a_r, double b_r)
{
    WarmRainParameters result{};
    result.alpha_r               = alpha_r;
    result.c_r                   = c_r;
    result.d_r                   = d_r;
    result.a_r                   = a_r;
    result.b_r                   = b_r;
    result.inv_mu_c              = 1.0 / 3.0;
    result.gc1                   = std::tgamma(1.0 + 1.0);
    result.gc2                   = std::tgamma(1.0 + 1.0 + 3.0 * result.inv_mu_c);
    result.gc3                   = std::tgamma(1.0 + 1.0 + 6.0 * result.inv_mu_c);
    result.gc4                   = std::tgamma(1.0 + 1.0 + 9.0 * result.inv_mu_c);
    result.gr31                  = std::tgamma(1.0 + alpha_r);
    result.gr34                  = std::tgamma(4.0 + alpha_r);
    result.gr37                  = std::tgamma(7.0 + alpha_r);
    result.inv_cloud_mass_factor = 1.0 / (result.gc2 / result.gc1 * (3.14159265358979323846 / 6.0) *
                                          physical_constants::liquid_water_density_kgm3);
    result.inv_rain_mass_factor  = 1.0 / (c_r * result.gr34 / result.gr31);

    result.gamma_ventilation      = std::tgamma(2.0 + alpha_r);
    result.gamma_fall_ventilation = std::tgamma(2.5 + alpha_r + 0.5 * b_r);
    return result;
}

} // namespace tmv::milbrandt_detail
