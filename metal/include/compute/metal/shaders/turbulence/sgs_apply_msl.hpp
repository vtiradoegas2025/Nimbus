#pragma once

namespace nimbus::metal::shaders
{

// Runtime-compiled translation of
// vulkan/shaders/compute/turbulence/sgs_apply.comp.
inline constexpr const char* kSgsApplyMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nx;
    int ny;
    int nz;
    int has_qv;
    int has_tke;
    float dt_step;
    float theta_fallback;
    float theta_min;
    float theta_max;
    float qv_max;
    float wind_h_max;
    float wind_v_max;
    int sanitize;
    int collocated;
};

inline bool finite_f(float value)
{
    return !isnan(value) && !isinf(value);
}

inline float zero_if_nonfinite(float value)
{
    return finite_f(value) ? value : 0.0f;
}

kernel void sgs_apply(
    device const float* u_in        [[buffer(0)]],
    device const float* v_in        [[buffer(1)]],
    device const float* w_in        [[buffer(2)]],
    device const float* theta_in    [[buffer(3)]],
    device const float* qv_in       [[buffer(4)]],
    device const float* tke_in      [[buffer(5)]],
    device const float* dudt        [[buffer(6)]],
    device const float* dvdt        [[buffer(7)]],
    device const float* dwdt        [[buffer(8)]],
    device const float* dthetadt    [[buffer(9)]],
    device const float* dqvdt       [[buffer(10)]],
    device const float* dtkedt      [[buffer(11)]],
    device const float* dtkedt_pbl  [[buffer(12)]],
    device float* u_out             [[buffer(13)]],
    device float* v_out             [[buffer(14)]],
    device float* w_out             [[buffer(15)]],
    device float* theta_out         [[buffer(16)]],
    device float* qv_out            [[buffer(17)]],
    device float* tke_out           [[buffer(18)]],
    constant PushConstants& p       [[buffer(19)]],
    uint gid                        [[thread_position_in_grid]])
{
    uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total)
        return;

    int i = int(gid / uint(p.ny * p.nz));
    int j = int((gid / uint(p.nz)) % uint(p.ny));
    int k = int(gid % uint(p.nz));
    if (p.collocated != 0 && (i == 0 || i == p.nx - 1 || j == 0 || j == p.ny - 1 || k == 0 || k == p.nz - 1))
    {
        u_out[gid] = u_in[gid]; v_out[gid] = v_in[gid]; w_out[gid] = w_in[gid];
        theta_out[gid] = theta_in[gid]; qv_out[gid] = qv_in[gid]; tke_out[gid] = tke_in[gid];
        return;
    }

    float du = p.sanitize != 0 ? zero_if_nonfinite(dudt[gid]) : dudt[gid];
    float dv = p.sanitize != 0 ? zero_if_nonfinite(dvdt[gid]) : dvdt[gid];
    float dw = p.sanitize != 0 ? zero_if_nonfinite(dwdt[gid]) : dwdt[gid];
    float dth = p.sanitize != 0 ? zero_if_nonfinite(dthetadt[gid]) : dthetadt[gid];

    float u_new = u_in[gid] + du * p.dt_step;
    u_out[gid] = p.sanitize == 0 ? u_new
        : (finite_f(u_new) ? clamp(u_new, -p.wind_h_max, p.wind_h_max) : 0.0f);
    float v_new = v_in[gid] + dv * p.dt_step;
    v_out[gid] = p.sanitize == 0 ? v_new
        : (finite_f(v_new) ? clamp(v_new, -p.wind_h_max, p.wind_h_max) : 0.0f);
    float w_new = w_in[gid] + dw * p.dt_step;
    w_out[gid] = p.sanitize == 0 ? w_new
        : (finite_f(w_new) ? clamp(w_new, -p.wind_v_max, p.wind_v_max) : 0.0f);

    float theta_new = theta_in[gid] + dth * p.dt_step;
    if (p.sanitize != 0 && !finite_f(theta_new))
        theta_new = finite_f(theta_in[gid]) ? theta_in[gid] : p.theta_fallback;
    theta_out[gid] = p.sanitize == 0 ? theta_new : clamp(theta_new, p.theta_min, p.theta_max);

    if (p.has_qv != 0)
    {
        float dqv = p.sanitize != 0 ? zero_if_nonfinite(dqvdt[gid]) : dqvdt[gid];
        float qv_new = qv_in[gid] + dqv * p.dt_step;
        qv_out[gid] = p.sanitize == 0 ? qv_new
            : (finite_f(qv_new) ? clamp(qv_new, 0.0f, p.qv_max) : 0.0f);
    }
    else
    {
        qv_out[gid] = qv_in[gid];
    }

    if (p.has_tke != 0)
    {
        float dtke = p.sanitize != 0
            ? zero_if_nonfinite(dtkedt[gid]) + zero_if_nonfinite(dtkedt_pbl[gid])
            : dtkedt[gid] + dtkedt_pbl[gid];
        float tke_new = tke_in[gid] + dtke * p.dt_step;
        if (p.sanitize != 0 && !finite_f(tke_new))
            tke_new = finite_f(tke_in[gid]) ? tke_in[gid] : 0.0f;
        tke_out[gid] = p.sanitize == 0 ? tke_new : max(0.0f, tke_new);
    }
    else
    {
        tke_out[gid] = tke_in[gid];
    }
}
)msl";

} // namespace nimbus::metal::shaders
