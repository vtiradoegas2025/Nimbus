#pragma once
#include "compute/kernel_abi.hpp"
#include <cstddef>

struct MovingMomentumDispatch
{
    static constexpr std::size_t input_stride  = 19;
    static constexpr std::size_t output_stride = 24;
    // Padded (nx+1)*(ny+1)*(nz+1) records: rho0/rho1, J0/J1 pairs, u/v/w,
    // three flux pairs and ground Gx/Gy pairs. Operation 1 reads horizontal swept volumes instead of fluxes.
    const float* input = nullptr;
    // Operation 0: stored wind, 11 high/low ledger pairs, then the velocity scaling exponent. Operation 1: x/y/z
    // mass-flux pairs.
    // Operations 2/4 return a stored boundary velocity, seven impulse/kinetic ledger pairs and the exponent.
    // Operation 3 returns scaled target/initial/outward impulse pairs and the exponent. Boundary grids have NZ=2.
    // Operation 4 reads a normalized target pair in slots 9/10 and its binary exponent in slot 11.
    float*                                            output = nullptr;
    nimbus::compute::abi::MovingMomentumPushConstants constants{};
};
