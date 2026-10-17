#pragma once
#include "compute/kernel_abi.hpp"
struct SlabColumnDispatch
{
    // 23 parameter pairs per column, then theta/qv/u/v/p/rho/interface-height pairs.
    const float* input = nullptr;
    // 11 diagnostic pairs per column, then four tendency pairs per layer.
    float*                                        output = nullptr;
    nimbus::compute::abi::SlabColumnPushConstants constants{};
};
