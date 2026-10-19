#pragma once

#include "core/field/field3d.hpp"
#include "core/infra/nested_grid.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tmv::domain
{

struct ConservedScalarProlongationReport
{
    double target_dry_mass_kg = 0.0;
    double committed_dry_mass_kg = 0.0;
    double dry_storage_residual_kg = 0.0;
    double target_carrier_inventory = 0.0;
    double committed_carrier_inventory = 0.0;
    double carrier_storage_residual = 0.0;
    std::size_t parent_cells = 0;
    std::size_t child_cells = 0;
};

struct ConservedFaceProlongationReport
{
    double target_inventory = 0.0;
    double committed_inventory = 0.0;
    double storage_residual = 0.0;
    std::size_t parent_owners = 0;
    std::size_t child_owners = 0;
};

struct ConservedMomentumProlongationReport
{
    double target_x_dual_dry_mass_kg = 0.0;
    double committed_x_dual_dry_mass_kg = 0.0;
    double x_dry_mass_residual_kg = 0.0;
    double target_x_momentum_kgms = 0.0;
    double committed_x_momentum_kgms = 0.0;
    double x_storage_residual_kgms = 0.0;
    double target_y_dual_dry_mass_kg = 0.0;
    double committed_y_dual_dry_mass_kg = 0.0;
    double y_dry_mass_residual_kg = 0.0;
    double target_y_momentum_kgms = 0.0;
    double committed_y_momentum_kgms = 0.0;
    double y_storage_residual_kgms = 0.0;
    double target_z_dual_dry_mass_kg = 0.0;
    double committed_z_dual_dry_mass_kg = 0.0;
    double z_dry_mass_residual_kg = 0.0;
    double target_z_momentum_kgms = 0.0;
    double committed_z_momentum_kgms = 0.0;
    double z_storage_residual_kgms = 0.0;
    std::size_t child_x_faces = 0;
    std::size_t child_y_faces = 0;
    std::size_t child_z_faces = 0;
};

struct PressureDiagnosisReport
{
    double target_pressure_volume_integral_pa_m3 = 0.0;
    double committed_pressure_volume_integral_pa_m3 = 0.0;
    double storage_residual_pa_m3 = 0.0;
    std::size_t diagnosed_cells = 0;
};

struct NestedRelaxationConfigV1
{
    static constexpr std::uint32_t schema_version = 1;
    double e_folding_time_s = 0.0;
};

struct ConservedScalarRelaxationReport
{
    double target_dry_mass_change_kg = 0.0;
    double committed_dry_mass_change_kg = 0.0;
    double dry_storage_residual_kg = 0.0;
    double target_carrier_change = 0.0;
    double committed_carrier_change = 0.0;
    double carrier_storage_residual = 0.0;
    std::size_t forced_cells = 0;
};

struct ConservedMomentumRelaxationReport
{
    double target_x_change_kgms = 0.0;
    double committed_x_change_kgms = 0.0;
    double x_storage_residual_kgms = 0.0;
    double target_y_change_kgms = 0.0;
    double committed_y_change_kgms = 0.0;
    double y_storage_residual_kgms = 0.0;
    double target_z_change_kgms = 0.0;
    double committed_z_change_kgms = 0.0;
    double z_storage_residual_kgms = 0.0;
};

double nested_scalar_relaxation_fraction(
    const OneWayNestRelationshipV1& relationship,
    const NestedRelaxationConfigV1& config,
    int child_i,
    int child_j,
    double child_step_s);

// Builds a first-order reference target in conserved variables. The source
// fields are conceptual physical scalar cells without storage halos. Output
// storage must already have the exact child shape, so successful execution
// performs no allocation.
void prolong_conserved_scalar_flat(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_dry_density,
    const Field3D& current_dry_density,
    const Field3D& previous_specific,
    const Field3D& current_specific,
    bool require_nonnegative_specific,
    Field3D& child_dry_density,
    Field3D& child_specific,
    ConservedScalarProlongationReport& report);

// Production-layout form of the scalar carrier reference. Parent and child
// fields include one lateral C-grid halo on every side; only physical child
// scalar cells are published and closure storage is left untouched.
void prolong_conserved_scalar_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_dry_density,
    const Field3D& current_dry_density,
    const Field3D& previous_specific,
    const Field3D& current_specific,
    bool require_nonnegative_specific,
    Field3D& child_dry_density,
    Field3D& child_specific,
    ConservedScalarProlongationReport& report);

// Remaps a conserved density that is genuinely piecewise constant over each
// native horizontal face Voronoi owner. The child footprint clips its two
// endpoint dual volumes to half width. This generic primitive is not a
// scalar-density-weighted momentum adapter because the two halves of such a
// face generally carry different scalar-cell densities.
void prolong_conserved_horizontal_face_flat(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    NestedHorizontalLocation owner,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_conserved_density,
    const Field3D& current_conserved_density,
    Field3D& child_conserved_density,
    ConservedFaceProlongationReport& report);

// Reconstructs native C-grid velocities from temporally blended parent
// momenta while dividing by the exact dual dry mass formed from the already
// committed child scalar density. All fields use the production flat C-grid
// storage layout: one lateral scalar halo on every side, u owns x faces, v
// owns y faces, and stored w owns the upper face of its scalar level.
void prolong_cgrid_momentum_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_parent_dry_density,
    const Field3D& current_parent_dry_density,
    const Field3D& previous_parent_u,
    const Field3D& current_parent_u,
    const Field3D& previous_parent_v,
    const Field3D& current_parent_v,
    const Field3D& previous_parent_w,
    const Field3D& current_parent_w,
    const Field3D& committed_child_dry_density,
    Field3D& child_u,
    Field3D& child_v,
    Field3D& child_w,
    ConservedMomentumProlongationReport& report);

// Diagnoses pressure over the physical child scalar domain from the committed
// rho/theta pair using the production FP64 dry-air EOS. The complete result is
// validated and its FP32 storage residual is reported before any mutation.
// Pressure is diagnostic; the volume integral is a rounding diagnostic, not a
// conserved pressure or energy ledger.
void diagnose_child_pressure_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const std::vector<double>& layer_thickness_m,
    const Field3D& committed_child_dry_density,
    const Field3D& committed_child_potential_temperature,
    Field3D& child_pressure,
    PressureDiagnosisReport& report);

// Applies the rate-based relaxation profile to one scalar carrier in
// conserved form. Source and destination use native halo-bearing storage but
// only physical child cells are considered. The outer ring is prescribed
// exactly; the interior taper is a timestep-independent exponential rate.
void relax_conserved_scalar_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const NestedRelaxationConfigV1& config,
    double child_step_s,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_parent_dry_density,
    const Field3D& current_parent_dry_density,
    const Field3D& previous_parent_specific,
    const Field3D& current_parent_specific,
    const Field3D& old_child_dry_density,
    const Field3D& old_child_specific,
    bool require_nonnegative_specific,
    Field3D& candidate_child_dry_density,
    Field3D& candidate_child_specific,
    ConservedScalarRelaxationReport& report);

void relax_cgrid_momentum_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const NestedRelaxationConfigV1& config,
    double child_step_s,
    const std::vector<double>& layer_thickness_m,
    const Field3D& old_dry_density,
    const Field3D& old_u,
    const Field3D& old_v,
    const Field3D& old_w,
    const Field3D& target_dry_density,
    const Field3D& target_u,
    const Field3D& target_v,
    const Field3D& target_w,
    const Field3D& committed_new_dry_density,
    Field3D& candidate_u,
    Field3D& candidate_v,
    Field3D& candidate_w,
    ConservedMomentumRelaxationReport& report);

} // namespace tmv::domain
