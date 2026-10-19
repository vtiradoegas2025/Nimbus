#pragma once

#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"

#include <array>
#include <cmath>

namespace tmv::pressure_energy_detail
{
// Reusable only across a density/pressure-only update on a validated flat C-grid.
struct CellMaterial
{
    double theta, mixture_factor, water_heat_capacity, phase_reference;
    double velocity_squared, tke, volume, height;
};

inline CellMaterial prepare(const SimulationStateView& state, int i, int j, int k)
{
    const auto& f = state.fields;
    const double qv = f.qv[i][j][k];
    const double liquid = static_cast<double>(f.qc[i][j][k]) + static_cast<double>(f.qr[i][j][k]);
    const double ice = static_cast<double>(f.qi[i][j][k]) + static_cast<double>(f.qs[i][j][k])
        + static_cast<double>(f.qg[i][j][k]) + static_cast<double>(f.qh[i][j][k]);
    const double total = qv + liquid + ice;
    const double u_w = f.u[i - 1][j][k], u_e = f.u[i][j][k];
    const double v_s = f.v[i][j - 1][k], v_n = f.v[i][j][k];
    const double w_b = k == 0 ? 0.0 : static_cast<double>(f.w[i][j][k - 1]);
    const double w_t = f.w[i][j][k];
    constexpr double cv_v = physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double vapor_reference = physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk * physical_constants::freezing_temperature_k;
    constexpr double ice_reference = -physical_constants::latent_heat_fusion_jkg;
    return {static_cast<double>(f.theta[i][j][k]), 1.0 + total,
        qv * cv_v + liquid * physical_constants::specific_heat_liquid_water_jkgk
            + ice * physical_constants::specific_heat_ice_jkgk,
        qv * vapor_reference + ice * ice_reference,
        u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n + w_b * w_b + w_t * w_t,
        static_cast<double>(f.tke[i][j][k]),
        physical_grid_metrics::cell_volume(state, i, j, k),
        physical_grid_metrics::scalar_height(state, i, j, k)};
}

// Retain the cell-result rounding boundary before attribution differences/sums.
#if defined(__GNUC__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
inline std::array<double, 6> reservoirs(const CellMaterial& material, double density, double pressure)
{
    constexpr double cv_d = physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    const double dry_mass = density * material.volume;
    const double mixture_mass = dry_mass * material.mixture_factor;
    const double temperature = material.theta * std::pow(pressure / physical_constants::reference_pressure_pa, kappa);
    const double delta_t = temperature - physical_constants::freezing_temperature_k;
    return {dry_mass * cv_d * delta_t,
        dry_mass * delta_t * material.water_heat_capacity,
        dry_mass * material.phase_reference,
        0.25 * mixture_mass * material.velocity_squared,
        mixture_mass * physical_constants::gravity_ms2 * material.height,
        dry_mass * material.tke};
}
}
