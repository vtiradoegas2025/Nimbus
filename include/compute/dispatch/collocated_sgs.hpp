#pragma once
#include <array>

struct CollocatedSgsDispatch
{
    // All bindings are full-volume arrays.
    // u/v/w/theta/qv/TKE/rho/dz, then nu/Ktheta/Kq/Ktke/buoyancy/dissipation high/low.
    std::array<const float*, 15> input{};
    // Closure: nu/Ktheta/Kq/Ktke/buoyancy/dissipation high/Pr/dissipation low.
    // Tendency: du/dv/dw/dtheta/dqv/dTKE/shear/unused.
    std::array<float*, 8> output{};
    int                   nx = 0, ny = 0, nz = 0, phase = 0, filter_mode = 0;
    bool                  horizontal = true, vertical = true, periodic_x = false, periodic_y = false;
    bool                  tke = false, stability = false, moist = false, has_qv = false;
    float                 dx = 0, dy = 0, Cs = 0, Pr = 0, Sc = 0, nu_max = 0, K_max = 0;
};
