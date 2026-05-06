/**
 * @file parametric_cape.cpp
 * @brief Procedural sounding source from a CAPE target. The math is the
 *        verbatim extraction of the previously-inline base-state build in
 *        `src/core/orchestration/dynamics/equations.cpp::initialize()` so
 *        that constructing this source with default parameters (matching the
 *        previously-hardcoded values) produces an identical column.
 */

#include "init/sounding/parametric_cape.hpp"
#include "core/infra/physical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace tmv::init
{

namespace
{

/**
 * @brief Magnus-formula saturation vapor pressure (Pa) over water/ice.
 *
 * Switches to ice formulation below 273.15 K. Coefficients match the
 * inline Magnus form used by the previous initialize() body so the
 * extracted source produces bit-identical qv at every level.
 */
double saturation_vapor_pressure_pa(double T_k)
{
    const double T_c = T_k - physical_constants::freezing_temperature_k;
    if (T_k >= physical_constants::freezing_temperature_k)
    {
        return 611.21 * std::exp((18.678 - T_c / 234.5) * T_c / (257.14 + T_c));
    }
    return 611.15 * std::exp((23.036 - T_c / 333.7) * T_c / (279.82 + T_c));
}

/**
 * @brief Saturation mixing ratio (kg/kg) at the supplied (T, p).
 */
double saturation_mixing_ratio(double T_k, double p_pa)
{
    const double e_sat = saturation_vapor_pressure_pa(T_k);
    return 0.622 * e_sat / std::max(p_pa - e_sat, 1.0);
}

}  // namespace

ParametricCAPESoundingSource::ParametricCAPESoundingSource(ParametricCAPEParams params)
    : params_(params)
{
}

Sounding ParametricCAPESoundingSource::build(const std::vector<double>& z_m, double dz) const
{
    if (z_m.empty())
    {
        throw std::invalid_argument("ParametricCAPESoundingSource::build: z_m is empty");
    }
    if (!(dz > 0.0))
    {
        throw std::invalid_argument("ParametricCAPESoundingSource::build: dz must be positive");
    }

    // Input safety clamps. Match the previously-inline floors so a config
    // with surface_theta_k = 240 silently rises to 250, etc.
    const double cape_scaling = params_.cape_target_jkg / params_.cape_reference_jkg;
    const double surface_theta = std::max(250.0, params_.surface_theta_k);
    const double surface_qv = std::max(1.0e-5, params_.surface_qv_kgkg);
    const double tropopause_z = std::max(8000.0, params_.tropopause_z_m);

    const double unstable_top_z = std::max(
        params_.unstable_top_floor_m,
        std::min(params_.unstable_top_ceil_m,
                 params_.unstable_top_factor * tropopause_z));
    const double unstable_lapse_rate =
        params_.unstable_lapse_base_kpm
        + params_.unstable_lapse_cape_kpm * cape_scaling;
    const double kappa = R_d / cp;

    // Five-layer T(z): mixed / unstable / upper-trop / tropopause-cap / strat.
    auto T_actual_at = [&](double z) -> double
    {
        if (z < params_.mixed_layer_top_m)
        {
            return surface_theta + params_.mixed_layer_dtheta_k;
        }
        if (z < unstable_top_z)
        {
            return surface_theta + params_.mixed_layer_dtheta_k
                   - unstable_lapse_rate * (z - params_.mixed_layer_top_m);
        }
        const double T_at_unstable_top =
            surface_theta + params_.mixed_layer_dtheta_k
            - unstable_lapse_rate * (unstable_top_z - params_.mixed_layer_top_m);
        if (z < tropopause_z)
        {
            return T_at_unstable_top
                   - params_.upper_trop_lapse_kpm * (z - unstable_top_z);
        }
        const double T_at_tropopause =
            T_at_unstable_top
            - params_.upper_trop_lapse_kpm * (tropopause_z - unstable_top_z);
        const double strat_base_z = tropopause_z + params_.tropopause_depth_m;
        if (z < strat_base_z)
        {
            return T_at_tropopause;
        }
        return T_at_tropopause + params_.strat_warming_kpm * (z - strat_base_z);
    };

    const std::size_t nz = z_m.size();
    Sounding s;
    s.z_m = z_m;
    s.T_k.resize(nz);
    s.theta_k.resize(nz);
    s.qv_kgkg.resize(nz);
    s.p_pa.resize(nz);
    s.rho_kgm3.resize(nz);

    // Hydrostatic integration. The previously-inline code used the literal
    // 0.0 for the surface temperature sample regardless of z_m[0]; we match
    // that here. In practice z_m[0] == 0 for every grid we run, so the two
    // are equivalent. The exponential form is exact for an isothermal layer
    // and second-order accurate using T_avg between adjacent levels.
    s.T_k[0] = T_actual_at(0.0);
    s.p_pa[0] = p0;
    s.rho_kgm3[0] = std::max(s.p_pa[0] / (R_d * s.T_k[0]), params_.rho_floor_kgm3);

    for (std::size_t k = 1; k < nz; ++k)
    {
        s.T_k[k] = T_actual_at(z_m[k]);
        const double T_avg = 0.5 * (s.T_k[k] + s.T_k[k - 1]);
        s.p_pa[k] = s.p_pa[k - 1] * std::exp(-g * dz / (R_d * T_avg));
        s.rho_kgm3[k] = std::max(s.p_pa[k] / (R_d * s.T_k[k]), params_.rho_floor_kgm3);
    }

    // Theta from definition: theta = T * (p0/p)^(R_d/cp).
    for (std::size_t k = 0; k < nz; ++k)
    {
        s.theta_k[k] = s.T_k[k] * std::pow(p0 / s.p_pa[k], kappa);
    }

    // Moisture column: surface mixing ratio (CAPE-modulated, clamped),
    // exponential decay above moisture_decay_z_m, capped at rh_cap of
    // saturation at every level.
    const double base_moisture = std::clamp(
        surface_qv * (params_.base_moisture_low_factor
                      + params_.base_moisture_cape_factor * cape_scaling),
        params_.base_moisture_min,
        params_.base_moisture_max);
    const double moisture_scale_height = std::max(
        params_.moisture_scale_height_floor_m,
        params_.moisture_scale_height_factor * tropopause_z);

    for (std::size_t k = 0; k < nz; ++k)
    {
        const double z = z_m[k];
        double qv_val;
        if (z < params_.moisture_decay_z_m)
        {
            qv_val = base_moisture;
        }
        else
        {
            qv_val = base_moisture * std::exp(
                -(z - params_.moisture_decay_z_m) / moisture_scale_height);
        }
        const double qvsat = saturation_mixing_ratio(s.T_k[k], s.p_pa[k]);
        s.qv_kgkg[k] = std::min(qv_val, qvsat * params_.rh_cap);
    }

    return s;
}

std::string ParametricCAPESoundingSource::describe() const
{
    return "parametric_cape";
}

}  // namespace tmv::init
