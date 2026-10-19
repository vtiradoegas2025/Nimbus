#pragma once

#include "microphysics/base/thermodynamics.hpp"
#include <algorithm>
#include <cmath>

namespace tmv::milbrandt_detail
{
struct GammaDistribution
{
    double number_volume      = 0.0;
    double inverse_lambda     = 0.0;
    double intercept          = 0.0;
    double mean_mass_diameter = 0.0;
};

inline GammaDistribution diagnose_distribution(double density,
                                               double mixing_ratio,
                                               double number_mixing_ratio,
                                               double alpha,
                                               double mass_coefficient,
                                               double mass_exponent)
{
    GammaDistribution result;
    if (!(density > 0.0) || !(mixing_ratio > 0.0) || !(number_mixing_ratio > 0.0) || !(mass_coefficient > 0.0) ||
        !(mass_exponent > 0.0))
    {
        return result;
    }
    result.number_volume      = density * number_mixing_ratio;
    const double gamma_number = std::tgamma(1.0 + alpha);
    const double gamma_mass   = std::tgamma(1.0 + alpha + mass_exponent);
    result.inverse_lambda =
        std::pow(density * mixing_ratio * gamma_number / (mass_coefficient * result.number_volume * gamma_mass),
                 1.0 / mass_exponent);
    result.intercept = result.number_volume / (gamma_number * std::pow(result.inverse_lambda, 1.0 + alpha));
    result.mean_mass_diameter =
        std::pow(density * mixing_ratio / (mass_coefficient * result.number_volume), 1.0 / mass_exponent);
    if (!std::isfinite(result.inverse_lambda) || !std::isfinite(result.intercept) ||
        !std::isfinite(result.mean_mass_diameter))
    {
        return {};
    }
    return result;
}

struct AirTransport
{
    double vapor_diffusivity    = 0.0;
    double thermal_conductivity = 0.0;
    double kinematic_viscosity  = 0.0;
    double schmidt_one_third    = 0.0;
    double density_factor       = 0.0;
};

inline AirTransport diagnose_air_transport(double temperature_k, double pressure_pa, double density)
{
    const double temperature_c = temperature_k - microphysics_constants::T0;
    AirTransport result;
    result.vapor_diffusivity = (2.2157e-5 + 0.0155e-5 * temperature_c) * 1.0e5 / pressure_pa;
    const double dynamic_viscosity =
        1.72e-5 * (393.0 / (temperature_k + 120.0)) * std::pow(temperature_k / microphysics_constants::T0, 1.5);
    result.kinematic_viscosity  = dynamic_viscosity / density;
    result.schmidt_one_third    = std::cbrt(result.kinematic_viscosity / result.vapor_diffusivity);
    result.thermal_conductivity = 2.3971e-2 + 0.0078e-2 * temperature_c;
    result.density_factor       = std::sqrt(1.225 / density);
    return result;
}

inline double ventilation_integral(
    const GammaDistribution& distribution, double alpha, double fall_a, double fall_b, const AirTransport& air)
{
    if (!(distribution.inverse_lambda > 0.0) || !(distribution.intercept > 0.0))
    {
        return 0.0;
    }
    constexpr double ventilation_a  = 0.78;
    constexpr double ventilation_b  = 0.30;
    const double     inverse_lambda = distribution.inverse_lambda;
    return ventilation_a * std::tgamma(2.0 + alpha) * std::pow(inverse_lambda, 2.0 + alpha) +
           ventilation_b * air.schmidt_one_third * std::sqrt(air.density_factor * fall_a / air.kinematic_viscosity) *
               std::tgamma(2.5 + 0.5 * fall_b + alpha) * std::pow(inverse_lambda, 2.5 + 0.5 * fall_b + alpha);
}

inline double distribution_moment(const GammaDistribution& distribution, double alpha, double exponent)
{
    if (!(distribution.intercept > 0.0) || !(distribution.inverse_lambda > 0.0) || !(exponent > -1.0 - alpha))
    {
        return 0.0;
    }
    return distribution.intercept * std::tgamma(1.0 + alpha + exponent) *
           std::pow(distribution.inverse_lambda, 1.0 + alpha + exponent);
}

inline double mass_weighted_fall_speed(const GammaDistribution& distribution,
                                       double                   alpha,
                                       double                   mass_exponent,
                                       double                   fall_a,
                                       double                   fall_b,
                                       const AirTransport&      air)
{
    if (!(distribution.inverse_lambda > 0.0))
        return 0.0;
    return air.density_factor * fall_a * std::pow(distribution.inverse_lambda, fall_b) *
           std::tgamma(1.0 + alpha + mass_exponent + fall_b) / std::tgamma(1.0 + alpha + mass_exponent);
}

} // namespace tmv::milbrandt_detail
