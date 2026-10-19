#pragma once
#include "compute/kernel_abi.hpp"
struct SimpleGreyDispatch
{
    // Per column: 17 configuration pairs, then seven physical-state pairs per interface.
    const float* input = nullptr;
    // Nine pairs per interface: four fluxes, three heating rates, and two column powers at interface zero.
    float*                                        output = nullptr;
    nimbus::compute::abi::SimpleGreyPushConstants constants{};
};
