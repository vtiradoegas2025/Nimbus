/**
 * @file slow_momentum_energy_report.hpp
 * @brief Exact basis-v2 attribution for the Cartesian slow dynamics commit.
 *
 * Momentum advection is presently evaluated in advective velocity form, not
 * as a retained conservative face flux.  Consequently this report measures
 * the work authored by that stencil at the actual velocity commit; it does not
 * invent a standalone boundary-energy flux.  Physical kinetic-energy export
 * is obtained later by composing this work with the operator-time continuity
 * mass flux and its carrier-kinetic change.
 *
 * The slow tendency also contains Coriolis, buoyancy, hydrometeor loading, and
 * optional PBL forcing.  The report separates the advective linear work from
 * that remainder and expands the exact quadratic finite-step term into both
 * self terms plus their cross term.  Storage rounding is the difference
 * between that double-precision update identity and the committed FP32 state.
 *
 * The same fused writer advances pressure before the first acoustic substep.
 * That pressure change alters dry and water-sensible internal energy and must
 * be owned here: the acoustic ledger begins from the already-committed pressure
 * and therefore cannot own it.  The two pressure reservoirs below are exact
 * before/after changes at the FP32 pressure commit and require no snapshots or
 * additional field traversal.
 */

#pragma once

#include "core/field/field3d.hpp"

#include <cstdint>

namespace tmv
{

struct SlowMomentumAdvectionTendencyOutputs
{
    Field3D* du_dt = nullptr;
    Field3D* dv_dt = nullptr;
    Field3D* dw_dt = nullptr;

    [[nodiscard]] bool requested() const noexcept
    {
        return du_dt != nullptr || dv_dt != nullptr || dw_dt != nullptr;
    }

    [[nodiscard]] bool complete() const noexcept
    {
        return du_dt != nullptr && dv_dt != nullptr && dw_dt != nullptr;
    }
};

struct CartesianSlowMomentumEnergyReportV1
{
    static constexpr std::uint32_t report_schema_version = 2;
    static constexpr std::uint32_t basis_version = 2;

    double advection_requested_linear_work_j = 0.0;
    double nonadvection_requested_linear_work_j = 0.0;
    double advection_finite_step_correction_j = 0.0;
    double nonadvection_finite_step_correction_j = 0.0;
    double coupled_finite_step_cross_work_j = 0.0;
    double storage_rounding_j = 0.0;
    double committed_kinetic_change_j = 0.0;
    double closure_residual_j = 0.0;
    double pressure_dry_internal_change_j = 0.0;
    double pressure_water_sensible_change_j = 0.0;
    double committed_pressure_energy_change_j = 0.0;
    bool complete = false;
};

} // namespace tmv

using tmv::CartesianSlowMomentumEnergyReportV1;
using tmv::SlowMomentumAdvectionTendencyOutputs;
