#pragma once
#include <array>

struct ConservativeRemapDispatch
{
    // Full-volume bindings: value high/low, beginning/end density,
    // old/new volume high/low, and west/east/south/north swept volume high/low.
    std::array<const float*, 16> input{};
    // High/low pairs: value, initial inventory, final inventory, boundary export, storage rounding.
    std::array<float*, 10> output{};
    int                    nx = 0, ny = 0, nz = 0;
    bool                   mixing_ratio = false;
    bool                   store_float  = true;
    int                    lower_bound  = 0; // 0: unrestricted, 1: nonnegative, 2: strictly positive.
};
