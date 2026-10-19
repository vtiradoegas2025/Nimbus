#pragma once

/**
 * @file boundary_layer.hpp
 * @brief Boundary-layer compute dispatch contracts.
 */

#include "compute/dispatch/slab_column.hpp"

enum class BoundaryLayerKernelOp
{
    MynnLevel25Stability,
    SlabColumn
};

// Independent grid points for the Nakanishi--Niino level-2.5 stability
// functions. All five outputs are one transaction: a backend must leave them
// untouched when validation, submission, or result validation fails.
struct MynnLevel25StabilityDispatch
{
    const float* gm_s2 = nullptr;
    const float* gh_s2 = nullptr;
    const float* mixing_length_m = nullptr;
    const float* q_squared_m2s2 = nullptr;
    float* sm = nullptr;
    float* sh = nullptr;
    float* gradient_richardson = nullptr;
    float* flux_richardson = nullptr;
    float* equilibrium_limiter = nullptr;
    int count = 0;
};

class BoundaryLayerComputeKernels
{
public:
    virtual ~BoundaryLayerComputeKernels() = default;

    virtual bool supports(BoundaryLayerKernelOp /*op*/) const { return false; }

    virtual bool dispatch_slab_column(const SlabColumnDispatch&)
    {
        return false;
    }
    virtual bool dispatch_mynn_level25_stability(
        const MynnLevel25StabilityDispatch& /*dispatch*/)
    {
        return false;
    }
};
