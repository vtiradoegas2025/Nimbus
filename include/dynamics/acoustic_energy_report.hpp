/**
 * @file acoustic_energy_report.hpp
 * @brief Operator-time energy attribution for one Cartesian acoustic step.
 *
 * The report follows the committed split operator rather than reconstructing
 * work from an endpoint snapshot.  Continuity changes dry carrier mass, the
 * EOS projection changes temperature, and the backward pressure-gradient
 * pass changes face velocity.  The first six fields measure the resulting
 * SCI-144 moist-mixture basis-v2 reservoir changes before the velocity
 * commit. Basis v2 is identical to v1 on a flat grid and adds the diagnosed
 * physical ground-face kinetic energy required by terrain coordinates. The
 * pressure-gradient fields then split that velocity commit into its requested
 * linear work, finite-step quadratic correction, and fp32/clamp residual.
 * On the collocated A-grid, the boundary fields also own the exact
 * mixture-weighted kinetic work of the interior directional null-mode
 * projection; halo closure itself has zero physical-inventory work.
 *
 * The acoustic boundary closure appends its exact committed work and KW78
 * equation decomposition to the same step-scoped transaction.
 */

#pragma once

#include <cstdint>

struct CartesianCGridAcousticEnergyReportV1
{
    static constexpr std::uint32_t report_schema_version = 1;
    static constexpr std::uint32_t basis_version = 2;

    double continuity_eos_dry_internal_change_j = 0.0;
    double continuity_eos_water_sensible_change_j = 0.0;
    double continuity_eos_phase_reference_change_j = 0.0;
    double continuity_carrier_kinetic_change_j = 0.0;
    double continuity_gravitational_change_j = 0.0;
    double continuity_subgrid_tke_change_j = 0.0;

    // SCI-150 separates the carrier-kinetic inventory change committed by
    // density storage from the continuum-linear change requested by the
    // continuity tendency. Positive boundary transport is outward. The
    // interior conversion is the discrete remainder in
    //
    //   requested change = -outward transport + interior conversion.
    //
    // Storage rounding closes committed = requested + storage. The final two
    // terms compose continuity with the SCI-149 advective momentum work; they
    // intentionally exclude Coriolis, buoyancy, loading, PBL, finite-step
    // quadratic terms, and fp32 storage.
    double continuity_carrier_requested_change_j = 0.0;
    double continuity_carrier_outward_boundary_transport_j = 0.0;
    double continuity_carrier_interior_conversion_j = 0.0;
    double continuity_carrier_storage_rounding_j = 0.0;
    double slow_momentum_advection_requested_linear_work_j = 0.0;
    double kinetic_transport_interior_conversion_j = 0.0;
    double kinetic_transport_closure_residual_j = 0.0;
    bool kinetic_transport_complete = false;

    double pressure_gradient_requested_linear_work_j = 0.0;
    double pressure_gradient_finite_step_correction_j = 0.0;
    double pressure_gradient_storage_rounding_j = 0.0;
    double pressure_gradient_committed_work_j = 0.0;

    // Exact SCI-144 basis-v2 kinetic-inventory change made by post-slow and
    // post-substep boundary closures.  The KW78 terms measure the update from
    // the saved radiative face history to the exact stored fp32 face.  The
    // constraint/history term is the remainder between that equation work and
    // the closure's live-face-to-final committed work, including east/north
    // pressure-update replacement and subsequent wall/lid constraints.
    double boundary_committed_work_j = 0.0;
    double boundary_radiative_requested_linear_work_j = 0.0;
    double boundary_radiative_finite_step_correction_j = 0.0;
    double boundary_radiative_storage_rounding_j = 0.0;
    double boundary_radiative_committed_work_j = 0.0;
    double boundary_constraint_history_committed_work_j = 0.0;
    std::uint64_t boundary_application_count = 0;
    bool boundary_complete = false;

    double committed_inventory_change_j = 0.0;
    std::uint64_t substep_count = 0;
    bool complete = false;
};

// Generic spelling for new Cartesian A-grid and C-grid call sites. Keep the
// historical C-grid spelling as the stored/checkpoint ABI name for schema v1.
using CartesianAcousticEnergyReportV1 =
    CartesianCGridAcousticEnergyReportV1;
