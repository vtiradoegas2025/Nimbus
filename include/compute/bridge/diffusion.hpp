#pragma once

struct ComputeBackendRuntimeState;

#include "compute/compute_kernel_interfaces.hpp"

/** Dispatches independent diffusion targets in one backend batch when supported. */
bool dispatch_diffusion_batch_backend(const DiffusionBatchDispatch& dispatch);
bool dispatch_diffusion_batch_backend(ComputeBackendRuntimeState& compute, const DiffusionBatchDispatch& dispatch);
