#include "cpu_backend.hpp"

#include "compute/compute_backend_factory.hpp"

#include <memory>
#include <string>

namespace nimbus::compute::cpu_internal
{

std::string CpuComputeBackend::name() const
{
    return "cpu";
}

bool CpuComputeBackend::initialize(std::string& error)
{
    error.clear();
    acoustic_fused_dispatches_ = 0;
    return true;
}

void CpuComputeBackend::shutdown()
{
}

AdvectionComputeKernels* CpuComputeBackend::advection_kernels()
{
    return this;
}

const AdvectionComputeKernels* CpuComputeBackend::advection_kernels() const
{
    return this;
}

DynamicsComputeKernels* CpuComputeBackend::dynamics_kernels()
{
    return this;
}

const DynamicsComputeKernels* CpuComputeBackend::dynamics_kernels() const
{
    return this;
}

std::uint64_t CpuComputeBackend::successful_kernel_dispatch_count(
    nimbus::compute::KernelId kernel) const
{
    return kernel
            == nimbus::compute::KernelId::AcousticPressureCartesianCgrid
        ? acoustic_fused_dispatches_
        : 0;
}

bool CpuComputeBackend::supports(AdvectionKernelOp op) const
{
    return op == AdvectionKernelOp::ConservativeCartesianCgridTransport;
}

bool CpuComputeBackend::supports_supplied_cartesian_cgrid_mass_flux() const
{
    return true;
}

bool CpuComputeBackend::supports(DynamicsKernelOp op) const
{
    return op == DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid;
}

} // namespace nimbus::compute::cpu_internal

std::unique_ptr<ComputeBackend> create_cpu_compute_backend()
{
    return std::make_unique<
        nimbus::compute::cpu_internal::CpuComputeBackend>();
}


