#pragma once

#include "compute/backend_interface.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace nimbus::compute::cpu_internal
{

struct AcousticCgridContext
{
    int nx;
    int ny;
    int nz;
    double inv_dx;
    double inv_dy;
    double inv_dz;
    double dt_small;
    double cell_volume;
};

class CpuComputeBackend final : public ComputeBackend,
                                public AdvectionComputeKernels,
                                public DynamicsComputeKernels
{
public:
    std::string name() const override;

    bool initialize(std::string& error) override;
    void shutdown() override;

    AdvectionComputeKernels* advection_kernels() override;
    const AdvectionComputeKernels* advection_kernels() const override;

    DynamicsComputeKernels* dynamics_kernels() override;
    const DynamicsComputeKernels* dynamics_kernels() const override;

    std::uint64_t successful_kernel_dispatch_count(
        nimbus::compute::KernelId kernel) const override;

    bool supports(AdvectionKernelOp op) const override;
    bool supports_supplied_cartesian_cgrid_mass_flux() const override;
    bool dispatch_advect_scalars_conservative_cartesian_cgrid(
        const ConservativeCartesianCgridTransportDispatch& dispatch) override;

    bool supports(DynamicsKernelOp op) const override;
    bool dispatch_acoustic_substep_fused_cartesian_cgrid(
        const AcousticSubstepFusedCartesianCgridDispatch& dispatch) override;

private:
    bool dispatch_acoustic_fast_path(
        const AcousticSubstepFusedCartesianCgridDispatch& dispatch,
        const AcousticCgridContext& context);
    bool dispatch_acoustic_energy_path(
        const AcousticSubstepFusedCartesianCgridDispatch& dispatch,
        const AcousticCgridContext& context);

    std::uint64_t acoustic_fused_dispatches_ = 0;

    // One fixed-order accumulator per x slab. Capacity survives acoustic
    // substeps and model steps; only the diagnostic path touches these arrays.
    std::vector<std::array<double, 8>> continuity_energy_partials_;
    std::vector<std::array<double, 3>> pressure_work_partials_;
};

} // namespace nimbus::compute::cpu_internal
