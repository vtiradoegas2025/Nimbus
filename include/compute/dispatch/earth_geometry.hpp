#pragma once
#include "compute/kernel_abi.hpp"
#include <array>

struct EarthGeometryDispatch
{
    // Padded horizontal planes use (nx + 4)*(ny + 4) samples. Metric planes are level-major.
    // Operations 0/1 return node heights or cell/x-face/y-face means in high/low pairs.
    // Operation 1 reads node pairs; scheme 0 samples a position and scheme 1 integrates the interval.
    // Operation 2 reads the three mean pairs and returns z/J/mx/my/zeta/z-face/Jx/Jy/Gx/Gy pairs.
    // Operation 3 reads Jx/Jy pairs and returns signed x/y swept-volume pairs.
    nimbus::compute::abi::EarthGeometryPushConstants constants{};
    std::array<const float*, 8>                      input{};
    std::array<float*, 20>                           output{};
};
