#pragma once
struct SimulationStorageState;
struct SimulationStateView;
class TerrainComputeKernels;
class DynamicsComputeKernels;
struct CartesianCGridAcousticEnergyReportV1;
namespace tmv
{
struct FinalBoundaryEnergyReportV1;
// Full-step closure preserves radiation-owned faces; acoustic time advancement is separate.
void apply_native_cgrid_boundary(SimulationStorageState&      state,
                                 TerrainComputeKernels&       kernels,
                                 bool                         reseed_wave_history = false,
                                 FinalBoundaryEnergyReportV1* report              = nullptr);
void apply_native_cgrid_acoustic_boundary(SimulationStorageState&               state,
                                          DynamicsComputeKernels&               kernels,
                                          double                                dt_s,
                                          CartesianCGridAcousticEnergyReportV1* report = nullptr);
void apply_native_cgrid_boundary(const SimulationStateView&   state,
                                 TerrainComputeKernels&       kernels,
                                 bool                         reseed_wave_history = false,
                                 FinalBoundaryEnergyReportV1* report              = nullptr);
void apply_native_cgrid_acoustic_boundary(const SimulationStateView&            state,
                                          DynamicsComputeKernels&               kernels,
                                          double                                dt_s,
                                          CartesianCGridAcousticEnergyReportV1* report = nullptr);
} // namespace tmv
