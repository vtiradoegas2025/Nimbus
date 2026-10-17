#pragma once
#include "core/field/field3d.hpp"
struct SimulationStateView;
class TerrainComputeKernels;
namespace tmv
{
struct HeldSourceInputs
{
    const Field3D* u_force         = nullptr;
    const Field3D* v_force         = nullptr;
    const Field3D* theta_rate      = nullptr;
    const Field3D* vapor_rate      = nullptr;
    const Field3D* liquid_rate     = nullptr;
    const Field3D* ice_rate        = nullptr;
    const Field3D* radiation_power = nullptr;
};
struct ReconstructedHeldSources
{
    Field3D du, dv, dtheta_pbl, dqv, dqc, dqi, dtheta_rad;
};
ReconstructedHeldSources reconstruct_native_held_sources(const SimulationStateView& state,
                                                         const HeldSourceInputs&    inputs,
                                                         TerrainComputeKernels&     kernels);
void                     reconstruct_native_held_sources(SimulationStateView& state, TerrainComputeKernels& kernels);
} // namespace tmv
