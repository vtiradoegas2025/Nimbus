/**
 * @file thermodynamics.hpp
 * @brief Declarations for the microphysics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the microphysics runtime and scheme implementations.
 * This file is part of the src/microphysics subsystem.
 */

#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "physics/microphysics_base.hpp"
#include "core/field3d.hpp"

/**
 * @brief Thermodynamic conversions and moist-air utility functions.
 */
namespace thermodynamics 
{

/**
 * @brief Converts the potential temperature to the temperature.
 */
inline double theta_to_temperature(double theta, double p) 
{
    using namespace microphysics_constants;
    if (!std::isfinite(theta) || !std::isfinite(p))
    {
        return T0;
    }
    const double theta_safe = std::clamp(theta, 150.0, 500.0);
    const double p_safe = std::clamp(p, 100.0, 200000.0);
    double kappa = microphysics_constants::R_d / microphysics_constants::cp;
    const double T = theta_safe * std::pow(p_safe / microphysics_constants::p0, kappa);
    return std::isfinite(T) ? T : T0;
}


/**
 * @brief Converts the temperature to the potential temperature.
 */
inline double temperature_to_theta(double T, double p) 
{
    using namespace microphysics_constants;
    if (!std::isfinite(T) || !std::isfinite(p))
    {
        return 300.0;
    }
    const double T_safe = std::clamp(T, 150.0, 400.0);
    const double p_safe = std::clamp(p, 100.0, 200000.0);
    double kappa = microphysics_constants::R_d / microphysics_constants::cp;
    const double theta = T_safe * std::pow(microphysics_constants::p0 / p_safe, kappa);
    return std::isfinite(theta) ? theta : 300.0;
}

/**
 * @brief Computes the saturation vapor pressure over water.
 */
inline double saturation_vapor_pressure_water(double T) 
{
    using namespace microphysics_constants;
    if (!std::isfinite(T))
    {
        return 0.0;
    }
    T = std::clamp(T, 150.0, 400.0);
    double T_c = T - T0;

    if (T >= T0) 
    {
        const double value = 611.21 * std::exp((18.678 - T_c/234.5) * T_c / (257.14 + T_c));
        return std::isfinite(value) ? std::max(0.0, value) : 0.0;
    } 
    else 
    {
        const double value = 611.15 * std::exp((23.036 - T_c/333.7) * T_c / (279.82 + T_c));
        return std::isfinite(value) ? std::max(0.0, value) : 0.0;
    }
}

/**
 * @brief Computes the saturation vapor pressure over ice.
 */
inline double saturation_vapor_pressure_ice(double T) 
{
    using namespace microphysics_constants;
    if (!std::isfinite(T))
    {
        return 0.0;
    }
    T = std::clamp(T, 150.0, 400.0);
    double T_c = T - T0;
    const double value = 611.15 * std::exp((23.036 - T_c/333.7) * T_c / (279.82 + T_c));
    return std::isfinite(value) ? std::max(0.0, value) : 0.0;
}

/**
 * @brief Computes the saturation mixing ratio over water.
 */
inline double saturation_mixing_ratio_water(double T, double p) 
{
    using namespace microphysics_constants;
    double e_sat = saturation_vapor_pressure_water(T);
    if (!std::isfinite(e_sat) || !std::isfinite(p))
    {
        return 0.0;
    }
    const double p_safe = std::clamp(p, 100.0, 200000.0);
    const double denom = std::max(p_safe - e_sat, 1.0);
    const double qvs = 0.62198 * e_sat / denom;
    return std::clamp(qvs, 0.0, 0.2);
}


/**
 * @brief Computes the saturation mixing ratio over ice.
 */
inline double saturation_mixing_ratio_ice(double T, double p) 
{
    using namespace microphysics_constants;
    double e_sat = saturation_vapor_pressure_ice(T);
    if (!std::isfinite(e_sat) || !std::isfinite(p))
    {
        return 0.0;
    }
    const double p_safe = std::clamp(p, 100.0, 200000.0);
    const double denom = std::max(p_safe - e_sat, 1.0);
    const double qvs = 0.62198 * e_sat / denom;
    return std::clamp(qvs, 0.0, 0.2);
}

/**
 * @brief Computes the relative humidity.
 */
inline double relative_humidity(double qv, double qvs) 
{
    if (!std::isfinite(qv) || !std::isfinite(qvs) || qvs <= 0.0)
    {
        return 0.0;
    }
    return qv / qvs;
}

/**
 * @brief Computes the latent heat of vaporization.
 */
inline double latent_heat_vaporization(double T) 
{
    using namespace microphysics_constants;
    if (!std::isfinite(T))
    {
        return L_v;
    }
    return L_v - 2370.0 * (T - T0);
}

/**
 * @brief Computes the virtual temperature.
 */
inline double virtual_temperature(double T, double qv, double qc = 0.0, double qr = 0.0) 
{
    using namespace microphysics_constants;
    if (!std::isfinite(T))
    {
        return T0;
    }
    const double qv_safe = std::clamp(std::isfinite(qv) ? qv : 0.0, 0.0, 0.2);
    const double qc_safe = std::clamp(std::isfinite(qc) ? qc : 0.0, 0.0, 0.2);
    const double qr_safe = std::clamp(std::isfinite(qr) ? qr : 0.0, 0.0, 0.2);
    double q_total = qv_safe + qc_safe + qr_safe;
    const double Tv = T * (1.0 + 0.608 * qv_safe - q_total);
    return std::isfinite(Tv) ? Tv : T0;
}

/**
 * @brief Computes the air density.
 */
inline double air_density(double p, double T, double qv = 0.0) 
{
    using namespace microphysics_constants;
    double Tv = virtual_temperature(T, qv);
    if (!std::isfinite(p) || !std::isfinite(Tv))
    {
        return 1.0;
    }
    const double p_safe = std::clamp(p, 100.0, 200000.0);
    const double Tv_safe = std::max(Tv, 1.0);
    const double rho_val = p_safe / (microphysics_constants::R_d * Tv_safe);
    return std::isfinite(rho_val) ? std::max(0.0, rho_val) : 1.0;
}

/**
 * @brief Computes the temperature tendency to the potential temperature tendency.
 */
inline double temperature_tendency_to_theta(double dT_dt, double theta, double p) 
{
    using namespace microphysics_constants;
    if (!std::isfinite(dT_dt) || !std::isfinite(p))
    {
        return 0.0;
    }
    const double p_safe = std::clamp(p, 100.0, 200000.0);
    double kappa = microphysics_constants::R_d / microphysics_constants::cp;
    const double dtheta_dt = dT_dt * std::pow(microphysics_constants::p0 / p_safe, kappa);
    return std::isfinite(dtheta_dt) ? dtheta_dt : 0.0;
}

/**
 * @brief Computes the saturation adjustment.
 */
inline double saturation_adjustment(double T_in, double p, double& qv, double& qc) 
{
    using namespace microphysics_constants;

    double T = std::isfinite(T_in) ? std::clamp(T_in, 150.0, 400.0) : T0;
    if (!std::isfinite(qv)) qv = 0.0;
    if (!std::isfinite(qc)) qc = 0.0;
    double qvs = saturation_mixing_ratio_water(T, p);
    if (!std::isfinite(qvs) || qvs < 0.0)
    {
        qvs = 0.0;
    }

    if (qv > qvs) 
    {
        double excess = qv - qvs;
        qc += excess;
        qv = qvs;
    } 
    else if (qv < qvs && qc > 0.0) 
    {
        double deficit = qvs - qv;
        double evaporate = std::min(deficit, qc);
        qc -= evaporate;
        qv += evaporate;
    }

    qc = std::max(0.0, qc);
    qv = std::max(0.0, qv);

    return T;
}

/**
 * @brief Converts the potential temperature field to the temperature field.
 */
void convert_theta_to_temperature_field(
    const Field3D& theta,
    const Field3D& p,
    Field3D& temperature
);

/**
 * @brief Converts the temperature field to the potential temperature field.
 */
void convert_temperature_to_theta_field(
    const Field3D& temperature,
    const Field3D& p,
    Field3D& theta
);

}
