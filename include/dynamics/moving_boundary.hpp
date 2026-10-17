#pragma once
#include "dynamics/moving_cgrid_transport.hpp"

namespace tmv
{
struct MovingBoundaryLedger
{
    double initial_impulse   = 0;
    double requested_impulse = 0;
    double final_impulse     = 0;
    double storage_rounding  = 0;
    double initial_kinetic   = 0;
    double exact_kinetic     = 0;
    double final_kinetic     = 0;
};
struct MovingGroundTargets
{
    TerrainField3D impulse;
    double         initial_impulse = 0;
    double         outward_impulse = 0;
};
std::array<MovingBoundaryLedger, 2>
project_moving_top_momentum(const TerrainMetrics3D& geometry, const Field3D& density, Field3D& u, Field3D& v);
// Only the required boundary planes are read; all candidate columns are validated before publication.
std::array<MovingBoundaryLedger, 2> project_moving_top_momentum(
    const TerrainMetrics3D& geometry, const Field3D& density, Field3D& u, Field3D& v, TerrainComputeKernels& kernels);
MovingGroundTargets  prepare_moving_ground_targets(const TerrainMetrics3D&           geometry,
                                                   const Field3D&                    density,
                                                   const Field3D&                    u,
                                                   const Field3D&                    v,
                                                   const TerrainContravariantFluxes& mass_flux);
MovingGroundTargets  prepare_moving_ground_targets(const TerrainMetrics3D&           geometry,
                                                   const Field3D&                    density,
                                                   const Field3D&                    u,
                                                   const Field3D&                    v,
                                                   const TerrainContravariantFluxes& mass_flux,
                                                   TerrainComputeKernels&            kernels);
MovingBoundaryLedger correct_moving_ground_momentum(const TerrainMetrics3D&    geometry,
                                                    const Field3D&             density,
                                                    const Field3D&             u,
                                                    const Field3D&             v,
                                                    Field3D&                   w,
                                                    const MovingGroundTargets& targets);
MovingBoundaryLedger correct_moving_ground_momentum(const TerrainMetrics3D&    geometry,
                                                    const Field3D&             density,
                                                    const Field3D&             u,
                                                    const Field3D&             v,
                                                    Field3D&                   w,
                                                    const MovingGroundTargets& targets,
                                                    TerrainComputeKernels&     kernels);
} // namespace tmv
