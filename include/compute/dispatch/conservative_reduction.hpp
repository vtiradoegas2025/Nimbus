#pragma once
#include "compute/kernel_abi.hpp"
struct ConservativeReductionDispatch
{
    const float*                                             input  = nullptr;
    float*                                                   output = nullptr;
    nimbus::compute::abi::ConservativeReductionPushConstants constants{};
};
