#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kMappedStateFluxMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct MappedStateUpdateConstants {
    int nx; int ny; int nz; uint mode;
    float dt; float pad0; float pad1; float pad2;
};

struct MappedFluxAccumulationConstants {
    int nx; int ny; int nz; int pad0;
    float time_weight; float pad1; float pad2; float pad3;
};

kernel void mapped_state_update(
    device const float* u_in [[buffer(0)]],
    device const float* v_in [[buffer(1)]],
    device const float* w_in [[buffer(2)]],
    device const float* rho_in [[buffer(3)]],
    device const float* p_in [[buffer(4)]],
    device const float* theta [[buffer(5)]],
    device const float* du [[buffer(6)]],
    device const float* dv [[buffer(7)]],
    device const float* dw [[buffer(8)]],
    device const float* drho [[buffer(9)]],
    device const float* dp [[buffer(10)]],
    device float* u_out [[buffer(11)]],
    device float* v_out [[buffer(12)]],
    device float* w_out [[buffer(13)]],
    device float* rho_out [[buffer(14)]],
    device float* p_out [[buffer(15)]],
    constant MappedStateUpdateConstants& c [[buffer(16)]],
    uint gid [[thread_position_in_grid]])
{
    const uint count = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= count) return;

    float next_u = u_in[gid];
    float next_v = v_in[gid];
    float next_w = w_in[gid];
    float next_rho = rho_in[gid];
    float next_p = p_in[gid];
    if (c.mode == 0u) {
        const float delta_u = du[gid] * c.dt;
        const float delta_v = dv[gid] * c.dt;
        const float delta_w = dw[gid] * c.dt;
        const float delta_p = dp[gid] * c.dt;
        next_u = u_in[gid] + delta_u;
        next_v = v_in[gid] + delta_v;
        next_w = w_in[gid] + delta_w;
        next_p = p_in[gid] + delta_p;
    } else if (c.mode == 1u) {
        const float delta_rho = drho[gid] * c.dt;
        next_rho = rho_in[gid] + delta_rho;
        next_p = 100000.0f * pow(next_rho * 287.0f * theta[gid] / 100000.0f,
                                1.40027894f);
    } else {
        const float delta_u = du[gid] * c.dt;
        const float delta_v = dv[gid] * c.dt;
        const float delta_w = dw[gid] * c.dt;
        next_u = u_in[gid] + delta_u;
        next_v = v_in[gid] + delta_v;
        next_w = w_in[gid] + delta_w;
    }
    u_out[gid] = next_u;
    v_out[gid] = next_v;
    w_out[gid] = next_w;
    rho_out[gid] = next_rho;
    p_out[gid] = next_p;
}

kernel void mapped_flux_accumulation(
    device const float* accumulated_x [[buffer(0)]],
    device const float* accumulated_y [[buffer(1)]],
    device const float* accumulated_zeta [[buffer(2)]],
    device const float* step_x [[buffer(3)]],
    device const float* step_y [[buffer(4)]],
    device const float* step_zeta [[buffer(5)]],
    device float* candidate_x [[buffer(6)]],
    device float* candidate_y [[buffer(7)]],
    device float* candidate_zeta [[buffer(8)]],
    constant MappedFluxAccumulationConstants& c [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    const uint count = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= count) return;
    const float delta_x = step_x[gid] * c.time_weight;
    const float delta_y = step_y[gid] * c.time_weight;
    const float delta_zeta = step_zeta[gid] * c.time_weight;
    candidate_x[gid] = accumulated_x[gid] + delta_x;
    candidate_y[gid] = accumulated_y[gid] + delta_y;
    candidate_zeta[gid] = accumulated_zeta[gid] + delta_zeta;
}
)msl";

} // namespace nimbus::metal::shaders
