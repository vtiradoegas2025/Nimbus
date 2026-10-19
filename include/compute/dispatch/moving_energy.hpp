#pragma once
#include "compute/kernel_abi.hpp"
#include <cstddef>

struct MovingEnergyDispatch
{
    static constexpr std::size_t input_stride  = 24;
    static constexpr std::size_t output_stride = 14;
    // Cell records: rho/theta/p/u/v/w, qv/qc/qr/qi/qs/qg/qh/tke, height/Gx/Gy/target/volume pairs.
    const float* input = nullptr;
    // theta/p, then high/low pairs: heat capacity, nonthermal energy, total density,
    // requested inventory, stored inventory and storage rounding. Ghost inventories are zero.
    float*                                          output = nullptr;
    nimbus::compute::abi::MovingEnergyPushConstants constants{};
};
