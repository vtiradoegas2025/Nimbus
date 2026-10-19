#pragma once

#include "core/field/field3d.hpp"
#include "terrain/terrain_base.hpp"

#include <array>

struct SimulationStateView;
class TerrainComputeKernels;

namespace tmv
{

struct MovingMomentumTransportLedger
{
    double initial_impulse = 0.0;
    double final_impulse = 0.0;
    double outward_impulse = 0.0;
    double impulse_storage_rounding = 0.0;
    double initial_kinetic = 0.0;
    double final_kinetic = 0.0;
    double outward_kinetic = 0.0;
    double spatial_conversion = 0.0;
    double finite_step_work = 0.0;
    double density_storage_work = 0.0;
    double velocity_storage_work = 0.0;
};

struct MovingCGridMomentumTransport
{
    Field3D u;
    Field3D v;
    Field3D w;
    std::array<MovingMomentumTransportLedger, 3> ledger;
};

// Donor-cell transport of dry momentum using half-sums of the retained primal
// mass transfers. Lateral normal faces, implicit ground and lid are owned by
// boundary closure; the ledgers cover only the advanced dual volumes.
MovingCGridMomentumTransport prepare_moving_cgrid_momentum_transport(
    const SimulationStateView& before,
    const TerrainMetrics3D& ending_geometry,
    const Field3D& ending_density,
    const TerrainContravariantFluxes& integrated_mass_flux,
    bool include_boundary_faces = false);

MovingCGridMomentumTransport prepare_moving_cgrid_momentum_transport(
    const SimulationStateView& before, const TerrainMetrics3D& ending_geometry,
    const Field3D& ending_density, const TerrainContravariantFluxes& integrated_mass_flux,
    TerrainComputeKernels& kernels, bool include_boundary_faces = false);

} // namespace tmv
