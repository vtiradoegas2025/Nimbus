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
#include <limits>
#include "microphysics/microphysics_base.hpp"
#include "core/field/field3d.hpp"

/**
 * @brief Thermodynamic conversions and moist-air utility functions.
 */
namespace tmv
{
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
    // Upper bound matches the prognostic state clamp (simulation.hpp
    // theta_max_k = 700); the old 500 K cap silently treated legitimate
    // stratospheric theta (~400-430 K near a 16 km top) inconsistently.
    const double theta_safe = std::clamp(theta, 150.0, 700.0);
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

    // Buck water equation at ALL temperatures - it is valid for
    // supercooled water. The old sub-freezing branch returned the Buck
    // ICE equation, collapsing the water/ice saturation split to zero
    // below 0 C: the Wegener-Bergeron-Findeisen band vanished and every
    // vapor-to-ice deposition guard (qv > qvs_ice while qv < qvs_water)
    // was unreachable (SCIENCE.md M7).
    const double value = 611.21 * std::exp((18.678 - T_c/234.5) * T_c / (257.14 + T_c));
    return std::isfinite(value) ? std::max(0.0, value) : 0.0;
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
 * @brief Computes virtual temperature from dry-air mixing ratios.
 *
 * Nimbus stores qv/qc/qr as mass of water per mass of dry air. For a physical
 * mixture pressure, p_mix = rho_d R_d T (1 + qv/epsilon), while the total
 * moist density is rho_d (1 + qv + qc + qr). Defining
 * p_mix = rho_m R_d Tv gives the exact dry-mixing-ratio conversion below;
 * condensates contribute mass but no gas pressure. This diagnostic identity
 * does not replace Nimbus's prognostic dry rho-theta EOS.
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
    const double epsilon = R_d / R_v;
    const double total_moist_mass_per_dry = 1.0 + qv_safe + qc_safe + qr_safe;
    const double Tv = T * (1.0 + qv_safe / epsilon) / total_moist_mass_per_dry;
    return std::isfinite(Tv) ? Tv : T0;
}

/**
 * @brief Computes prognostic dry-air density from Nimbus dry-EOS pressure.
 *
 * The active dynamics closure is p = rho_d R_d T and represents vapor
 * buoyancy separately. The result is therefore independent of moisture and
 * is the density that must multiply every prognostic q field to form a water
 * mass concentration.
 */
inline double dry_air_density(double p, double T)
{
    using namespace microphysics_constants;
    if (!std::isfinite(p) || !std::isfinite(T))
    {
        return 1.0;
    }
    const double p_safe = std::clamp(p, 100.0, 200000.0);
    const double T_safe = std::max(T, 1.0);
    const double rho_val = p_safe / (R_d * T_safe);
    return std::isfinite(rho_val) ? std::max(0.0, rho_val) : 1.0;
}

/**
 * @brief Compatibility name for the model's prognostic dry-air density.
 *
 * Prefer dry_air_density() in new code so the mass basis remains visible at
 * call sites.
 */
inline double air_density(double p, double T, double qv = 0.0)
{
    (void)qv;
    return dry_air_density(p, T);
}

/**
 * @brief Computes the temperature tendency to the potential temperature tendency.
 */
inline double temperature_tendency_to_theta(double dT_dt, double p)
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
 * @brief Computes density from the equation of state given pressure and
 *        potential temperature.
 *
 * Uses the ideal-gas EOS:
 *   T   = theta * (p / p0)^kappa
 *   rho = p / (R_d * T)
 *
 * This is the thermodynamic closure that ties rho to p and theta.
 * Without it, density-based buoyancy (-g (rho - rho0) / rho) cannot
 * respond to theta perturbations (e.g. warm bubble, latent heating).
 */
inline double density_from_eos(double p_val, double theta_val)
{
    if (!std::isfinite(p_val) || !std::isfinite(theta_val))
    {
        return 1.0;
    }
    const double p_safe = std::clamp(p_val, 100.0, 200000.0);
    // Upper bound matches simulation.hpp theta_max_k (700); was 500.
    const double theta_safe = std::clamp(theta_val, 150.0, 700.0);
    const double kappa = microphysics_constants::R_d / microphysics_constants::cp;
    const double T = theta_safe * std::pow(p_safe / microphysics_constants::p0, kappa);
    if (T < 1.0)
    {
        return 1.0;
    }
    const double rho_val = p_safe / (microphysics_constants::R_d * T);
    return std::isfinite(rho_val) ? std::clamp(rho_val, 0.01, 5.0) : 1.0;
}

/**
 * @brief Updates the density field from the equation of state.
 *
 * Applies density_from_eos pointwise to ensure rho is thermodynamically
 * consistent with p and theta. Call this after any process that modifies
 * theta (bubble initialization, microphysics, radiation, turbulence).
 */
void update_density_from_eos(
    const Field3D& p, const Field3D& theta, Field3D& rho);

/**
 * @brief Computes the saturation adjustment.
 */
inline double saturation_adjustment(double T_in, double p, double& qv, double& qc) 
{
    using namespace microphysics_constants;

    double T = std::isfinite(T_in) ? std::clamp(T_in, 150.0, 400.0) : T0;
    qv = std::isfinite(qv) ? std::max(0.0, qv) : 0.0;
    qc = std::isfinite(qc) ? std::max(0.0, qc) : 0.0;
    double qvs = saturation_mixing_ratio_water(T, p);
    if (!std::isfinite(qvs) || qvs < 0.0)
    {
        qvs = 0.0;
    }

    // Solve the coupled liquid saturation/latent-heat problem. The previous
    // implementation used the linearized 1/(1+gamma) mass correction but
    // returned the original temperature, so it removed or created latent
    // energy while downstream processes still saw the pre-adjustment state.
    // Bisection is monotone for both condensation and evaporation:
    //
    //   qv' = qvs(T'),  T' = T + (Lv/cp) * (qv - qv').
    //
    // If the finite cloud reservoir cannot reach saturation, evaporate all
    // available cloud and retain the corresponding cooling.
    constexpr int solve_iterations = 10;
    const double latent_temperature_factor = L_v / cp;
    const auto saturation_slope =
        [&](double temperature, double saturation)
        {
            if (!(temperature > 150.0 && temperature < 400.0)
                || !(saturation > 0.0 && saturation < 0.2))
            {
                return 0.0;
            }
            const double pressure =
                std::clamp(p, 100.0, 200000.0);
            const double vapor_pressure =
                saturation * pressure / (0.62198 + saturation);
            const double temperature_c = temperature - T0;
            const double denominator = 257.14 + temperature_c;
            const double numerator =
                (18.678 - temperature_c / 234.5)
                * temperature_c;
            const double numerator_slope =
                18.678 - 2.0 * temperature_c / 234.5;
            const double log_pressure_slope =
                (numerator_slope * denominator - numerator)
                / (denominator * denominator);
            return saturation
                * pressure
                / std::max(pressure - vapor_pressure, 1.0)
                * log_pressure_slope;
        };
    if (qv > qvs)
    {
        const double maximum_condensation = std::min(
            qv,
            std::max(0.0, (400.0 - T)
                     / latent_temperature_factor));
        const auto residual =
            [&](double condensed)
            {
                const double adjusted_temperature =
                    T + latent_temperature_factor * condensed;
                return qv - condensed
                    - saturation_mixing_ratio_water(
                        adjusted_temperature, p);
            };
        double condensed = maximum_condensation;
        if (residual(maximum_condensation) <= 0.0)
        {
            double lower = 0.0;
            double upper = maximum_condensation;
            condensed = std::clamp(
                (qv - qvs)
                    / (1.0
                       + latent_temperature_factor
                         * saturation_slope(T, qvs)),
                lower, upper);
            for (int iteration = 0;
                 iteration < solve_iterations; ++iteration)
            {
                const double adjusted_temperature =
                    T + latent_temperature_factor * condensed;
                const double adjusted_saturation =
                    saturation_mixing_ratio_water(
                        adjusted_temperature, p);
                const double value =
                    qv - condensed - adjusted_saturation;
                if (std::abs(value)
                    <= 8.0 * std::numeric_limits<double>::epsilon()
                       * std::max(
                           {1.0, std::abs(qv),
                            std::abs(adjusted_saturation)}))
                {
                    break;
                }
                if (value > 0.0)
                    lower = condensed;
                else
                    upper = condensed;
                const double derivative =
                    -1.0
                    - latent_temperature_factor
                      * saturation_slope(
                          adjusted_temperature,
                          adjusted_saturation);
                const double candidate = condensed - value / derivative;
                condensed =
                    candidate > lower && candidate < upper
                    ? candidate
                    : 0.5 * (lower + upper);
            }
        }
        qv -= condensed;
        qc += condensed;
        T += latent_temperature_factor * condensed;
    }
    else if (qv < qvs && qc > 0.0)
    {
        const double maximum_evaporation = std::min(
            qc,
            std::max(0.0, (T - 150.0)
                     / latent_temperature_factor));
        const auto residual =
            [&](double evaporated)
            {
                const double adjusted_temperature =
                    T - latent_temperature_factor * evaporated;
                return qv + evaporated
                    - saturation_mixing_ratio_water(
                        adjusted_temperature, p);
            };
        double evaporated = maximum_evaporation;
        if (residual(maximum_evaporation) >= 0.0)
        {
            double lower = 0.0;
            double upper = maximum_evaporation;
            evaporated = std::clamp(
                (qvs - qv)
                    / (1.0
                       + latent_temperature_factor
                         * saturation_slope(T, qvs)),
                lower, upper);
            for (int iteration = 0;
                 iteration < solve_iterations; ++iteration)
            {
                const double adjusted_temperature =
                    T - latent_temperature_factor * evaporated;
                const double adjusted_saturation =
                    saturation_mixing_ratio_water(
                        adjusted_temperature, p);
                const double value =
                    qv + evaporated - adjusted_saturation;
                if (std::abs(value)
                    <= 8.0 * std::numeric_limits<double>::epsilon()
                       * std::max(
                           {1.0, std::abs(qv),
                            std::abs(adjusted_saturation)}))
                {
                    break;
                }
                if (value < 0.0)
                    lower = evaporated;
                else
                    upper = evaporated;
                const double derivative =
                    1.0
                    + latent_temperature_factor
                      * saturation_slope(
                          adjusted_temperature,
                          adjusted_saturation);
                const double candidate = evaporated - value / derivative;
                evaporated =
                    candidate > lower && candidate < upper
                    ? candidate
                    : 0.5 * (lower + upper);
            }
        }
        qc -= evaporated;
        qv += evaporated;
        T -= latent_temperature_factor * evaporated;
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

} // namespace thermodynamics
} // namespace tmv

namespace thermodynamics = tmv::thermodynamics;
