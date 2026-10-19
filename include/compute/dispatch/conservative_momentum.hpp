#pragma once

#include "compute/kernel_abi.hpp"
#include <cstddef>

struct ConservativeMomentumDispatch
{
    static constexpr std::size_t input_stride    = 21;
    static constexpr std::size_t tendency_stride = 8;
    // u/v/w, dry rho, p, rho0 pair, p0 pair, qv, stored qv0, loading pair, u0/v0 pairs, mapped positive-face Jx/Jy
    // pairs.
    const float* input = nullptr;
    // Eight tendencies per cell, followed by high/low x/y/z flux pairs on W/E/S/N strips.
    float*                                                  output = nullptr;
    nimbus::compute::abi::ConservativeMomentumPushConstants constants{};
};
