#pragma once

#include "core/infra/physical_constants.hpp"

#include <cmath>
#include <cstdint>

struct SimulationStateView;

namespace tmv::dynamics_eos
{

struct PressureProjectionEnergyReportV1
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t energy_basis_version = 2;

    double dry_internal_change_j = 0.0;
    double water_sensible_change_j = 0.0;
    double phase_reference_change_j = 0.0;
    double carrier_kinetic_change_j = 0.0;
    double gravitational_change_j = 0.0;
    double subgrid_tke_change_j = 0.0;
    double committed_energy_change_j = 0.0;
    double closure_residual_j = 0.0;
    bool complete = false;
};

/** Current scientific reference: evaluate the dry rho-theta EOS in fp64. */
inline double pressure_fp64(float rho_kgm3, float theta_k) noexcept
{
    const double base = static_cast<double>(rho_kgm3)
                      * physical_constants::gas_constant_dry_air_jkgk
                      * static_cast<double>(theta_k)
                      / physical_constants::reference_pressure_pa;
    return physical_constants::reference_pressure_pa
         * std::pow(base, physical_constants::dry_air_gamma);
}

/** CPU-028 experiment: evaluate the same stored-float EOS in fp32. */
inline float pressure_fp32(float rho_kgm3, float theta_k) noexcept
{
    constexpr float rd = static_cast<float>(
        physical_constants::gas_constant_dry_air_jkgk);
    constexpr float p0 = static_cast<float>(
        physical_constants::reference_pressure_pa);
    constexpr float gamma = static_cast<float>(
        physical_constants::dry_air_gamma);
    const float base = rho_kgm3 * rd * theta_k / p0;
    return p0 * std::pow(base, gamma);
}

/** Build-selected EOS evaluation; fp64 remains the default contract. */
inline double pressure_selected(float rho_kgm3, float theta_k) noexcept
{
#ifdef NIMBUS_EOS_FP32_EXPERIMENT
    return static_cast<double>(pressure_fp32(rho_kgm3, theta_k));
#else
    return pressure_fp64(rho_kgm3, theta_k);
#endif
}

/**
 * Diagnose stored pressure from the production FP64 dry rho-theta EOS and
 * return the exact basis-v2 thermodynamic inventory change caused by that
 * algebraic projection. The attribution is fused into the pressure writer.
 */
PressureProjectionEnergyReportV1 apply_pressure_projection(
    const SimulationStateView& state);

void accumulate_pressure_projection_energy(
    PressureProjectionEnergyReportV1& destination,
    const PressureProjectionEnergyReportV1& source);

} // namespace tmv::dynamics_eos
