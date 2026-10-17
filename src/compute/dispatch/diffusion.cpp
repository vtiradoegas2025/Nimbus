#include "compute/bridge/diffusion.hpp"

#include "compute/compute_backend.hpp"

namespace
{
DiffusionComputeKernels* active_diffusion_kernels(ComputeBackendRuntimeState& compute)
{
    ComputeBackend* backend = mutable_compute_backend(compute);
    return backend != nullptr ? backend->diffusion_kernels() : nullptr;
}
} // namespace

bool dispatch_diffusion_batch_backend(const DiffusionBatchDispatch& dispatch)
{
    return dispatch_diffusion_batch_backend(global_compute_backend_runtime_state, dispatch);
}

bool dispatch_diffusion_batch_backend(ComputeBackendRuntimeState& compute, const DiffusionBatchDispatch& dispatch)
{
    DiffusionComputeKernels* kernels = active_diffusion_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(DiffusionKernelOp::Diffusion))
    {
        return kernels->dispatch_diffusion_batch(dispatch);
    }
    return false;
}
