#pragma once
#include "compute/kernel_abi.hpp"
#include <array>
struct CgridBoundaryDispatch
{
    std::array<const float*, 5>                       fields{};
    const float*                                      wave = nullptr;
    std::array<float*, 5>                             output{};
    float*                                            wave_output = nullptr;
    nimbus::compute::abi::AcousticBcFacePushConstants constants{};
};
