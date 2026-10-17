#pragma once
#include "compute/kernel_abi.hpp"
#include <cstddef>
struct ReferenceAtmosphereDispatch
{
    // Five profile pairs per level (theta, pressure, vapor, u, v), then height pairs.
    static constexpr std::size_t profile_stride = 10;
    static constexpr std::size_t output_stride  = 6;
    const float*                 input          = nullptr;
    // Interpolated theta, pressure, vapor, u, v and EOS density for each requested height.
    float*                                                 output = nullptr;
    nimbus::compute::abi::ReferenceAtmospherePushConstants constants{};
};
