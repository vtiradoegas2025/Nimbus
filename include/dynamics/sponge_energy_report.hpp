/**
 * @file sponge_energy_report.hpp
 * @brief Exact basis-v2 energy attribution for boundary sponge relaxation.
 *
 * Each pass owns the state change it actually commits. Momentum relaxation is
 * decomposed into requested linear work, the exact finite-step quadratic
 * correction, and FP32 storage rounding. The six reservoir fields then state
 * the complete moist-mixture basis-v2 inventory change. For a sponge pass the
 * carrier-kinetic reservoir is the committed momentum work; theta relaxation
 * changes dry internal and water sensible energy at fixed stored pressure.
 */

#pragma once

#include <cstdint>

namespace tmv
{

struct SpongePassEnergyReportV1
{
    double momentum_requested_linear_work_j = 0.0;
    double momentum_finite_step_correction_j = 0.0;
    double momentum_storage_rounding_j = 0.0;

    double dry_internal_change_j = 0.0;
    double water_sensible_change_j = 0.0;
    double phase_reference_change_j = 0.0;
    double carrier_kinetic_change_j = 0.0;
    double gravitational_change_j = 0.0;
    double subgrid_tke_change_j = 0.0;
    double committed_energy_change_j = 0.0;
    double closure_residual_j = 0.0;

    bool applied = false;
    bool complete = false;
};

struct SpongeEnergyReportV1
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t energy_basis_version = 2;

    SpongePassEnergyReportV1 upper;
    SpongePassEnergyReportV1 w_only;
    SpongePassEnergyReportV1 lateral;
    bool complete = false;
};

void accumulate_sponge_pass_energy(
    SpongePassEnergyReportV1& destination,
    const SpongePassEnergyReportV1& source);

void accumulate_sponge_energy(
    SpongeEnergyReportV1& destination,
    const SpongeEnergyReportV1& source);

} // namespace tmv
