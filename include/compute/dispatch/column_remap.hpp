#pragma once
#include "compute/kernel_abi.hpp"
#include <cstddef>
struct ColumnRemapDispatch
{
    static constexpr std::size_t input_stride = 9, output_stride = 11;
    // Three normalized pairs followed by their individual binary exponents.
    const float* input = nullptr;
    // Three remapped pairs, FP32 value/residual, then three binary exponents.
    float*                                         output = nullptr;
    nimbus::compute::abi::ColumnRemapPushConstants constants{};
};
