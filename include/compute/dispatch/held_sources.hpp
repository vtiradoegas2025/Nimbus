#pragma once
#include "compute/kernel_abi.hpp"
#include <cstddef>
struct HeldSourcesDispatch
{
    static constexpr std::size_t input_stride = 11, output_stride = 7;
    // rho, pressure, physical-volume pair, then seven raw held source scalars.
    const float* input = nullptr;
    // du, dv, PBL dtheta/dqv/dqc/dqi, radiation dtheta.
    float*                                         output = nullptr;
    nimbus::compute::abi::HeldSourcesPushConstants constants{};
};
