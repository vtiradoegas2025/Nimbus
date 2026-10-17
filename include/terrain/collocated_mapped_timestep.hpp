#pragma once

#include "core/field/field3d.hpp"
#include "terrain/terrain_base.hpp"

#include <cstdint>
#include <vector>

namespace tmv::terrain_collocated
{

struct CollocatedMappedState
{
    Field3D& u;
    Field3D& v;
    Field3D& w;
    Field3D& rho;
    Field3D& p;
    Field3D& theta;
    Field3D& qv;
    Field3D& tracer;
    Field3D& tke;
};

struct CollocatedMappedStateView
{
    const Field3D& u;
    const Field3D& v;
    const Field3D& w;
    const Field3D& rho;
    const Field3D& p;
    const Field3D& theta;
    const Field3D& qv;
    const Field3D& tracer;
    const Field3D& tke;
};

struct ConservationClosure
{
    double before = 0.0;
    double after = 0.0;
    double source = 0.0;
    double boundary_outflow = 0.0;
    double residual = 0.0;
};

struct CollocatedMappedStepReport
{
    static constexpr std::uint32_t report_schema_version = 1;

    std::uint64_t physical_cells = 0;
    std::uint64_t closure_cells_written = 0;
    std::uint64_t eos_projected_cells = 0;
    std::uint64_t internal_mass_faces = 0;
    double dt_s = 0.0;
    double max_outgoing_courant = 0.0;
    ConservationClosure dry_mass;
    ConservationClosure rho_theta;
    ConservationClosure water_vapor;
    ConservationClosure tracer;
    ConservationClosure subgrid_tke;
    ConservationClosure x_momentum;
    ConservationClosure y_momentum;
    ConservationClosure z_momentum;
    bool rigid_closed_boundary = false;
    bool final_pressure_uses_canonical_fp64 = false;
    bool complete = false;
};

using CollocatedMappedCompletedObserver = void (*)(
    void* context,
    const CollocatedMappedStateView& candidate,
    const TerrainContravariantFluxes& retained_mass_flux,
    const CollocatedMappedStepReport& report);

/**
 * Advance one conservative mapped Cartesian A-grid Euler stage.
 *
 * The exact foundation owns one closure cell on every side. Physical cells
 * are [1,nx-1) x [1,ny-1) x [1,nz-1). All coordinate-boundary mass fluxes
 * are zero; the lower closure velocity is reflected about the local terrain
 * tangent and the upper closure is reflected about a rigid horizontal lid.
 * Density, rho-theta, rho-qv, rho-tracer, rho-TKE, and all three
 * density-weighted momenta share one retained contravariant mass flux. The
 * pressure field is diagnostic and is projected from candidate rho/theta by
 * the canonical fp64 dry EOS before atomic publication.
 */
void advance_collocated_mapped_stage(
    CollocatedMappedState state,
    const TerrainMetrics3D& metrics,
    const std::vector<double>& reference_pressure,
    const std::vector<double>& reference_theta,
    double dt_s,
    CollocatedMappedStepReport& report,
    CollocatedMappedCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);

} // namespace tmv::terrain_collocated

