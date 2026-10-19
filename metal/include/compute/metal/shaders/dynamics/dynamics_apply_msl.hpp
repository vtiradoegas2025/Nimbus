#pragma once

namespace nimbus::metal::shaders
{

// Native pointwise state mutation kernels for the split-explicit dynamics
// tail. Buffer indices intentionally mirror the canonical Vulkan bindings;
// constants retain the shared 48-byte and 32-byte ABIs.
inline constexpr const char* kDynamicsApplyMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct StateRelaxationConstants {
    int nx; int ny; int nz; int lateral_width;
    int reserved_zero; int do_upper; int do_w_only; int do_lateral;
    float dt_step; float reserved1; float reserved2; float reserved3;
};

struct SlowApplyConstants {
    int nx; int ny; int nz; int sanitize;
    float dt_large; float pad1; float pad2; float pad3;
};

kernel void state_relaxation_cartesian(
    device const float* u_in [[buffer(0)]],
    device const float* v_in [[buffer(1)]],
    device const float* w_in [[buffer(2)]],
    device const float* theta_in [[buffer(3)]],
    device const float* profiles [[buffer(4)]],
    device float* u_out [[buffer(5)]],
    device float* v_out [[buffer(6)]],
    device float* w_out [[buffer(7)]],
    device float* theta_out [[buffer(8)]],
    constant StateRelaxationConstants& p [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total) return;
    const int k = int(gid % uint(p.nz));
    const uint ij = gid / uint(p.nz);
    const int j = int(ij % uint(p.ny));
    const int i = int(ij / uint(p.ny));
    const float theta0 = profiles[uint(4 * p.nz + k)];
    const float u_ref = profiles[uint(2 * p.nz + k)];
    const float v_ref = profiles[uint(3 * p.nz + k)];
    const float dt_safe = max(p.dt_step, 1.0e-12f);
    float uv = u_in[gid], vv = v_in[gid], wv = w_in[gid], tv = theta_in[gid];

    if (p.do_upper != 0) {
        const float alpha = profiles[uint(k)];
        const float alpha_w = profiles[uint(6 * p.nz + k)];
        if (alpha > 0.0f || alpha_w > 0.0f) {
            const float factor = 1.0f - exp(-alpha * dt_safe);
            const float factor_w = 1.0f - exp(-alpha_w * dt_safe);
            uv += factor * (u_ref - uv);
            vv += factor * (v_ref - vv);
            wv += factor_w * -wv;
            tv += factor * (theta0 - tv);
        }
    }
    if (p.do_w_only != 0) {
        const float alpha_w = profiles[uint(p.nz + k)];
        if (alpha_w > 0.0f) {
            const float factor_w = 1.0f - exp(-alpha_w * dt_safe);
            wv += factor_w * -wv;
        }
    }
    if (p.do_lateral != 0 && p.lateral_width > 0) {
        const int distance = min(min(i, p.nx - 1 - i), min(j, p.ny - 1 - j));
        if (distance < p.lateral_width) {
            const float alpha = profiles[uint(5 * p.nz + distance)];
            const float factor = 1.0f - exp(-alpha * dt_safe);
            uv += factor * (u_ref - uv);
            vv += factor * (v_ref - vv);
            wv += factor * -wv;
            tv += factor * (theta0 - tv);
        }
    }
    u_out[gid] = uv; v_out[gid] = vv; w_out[gid] = wv; theta_out[gid] = tv;
}

inline float zero_if_nonfinite(float value)
{
    return isfinite(value) ? value : 0.0f;
}

kernel void slow_apply(
    device const float* u_in [[buffer(0)]],
    device const float* v_in [[buffer(1)]],
    device const float* w_in [[buffer(2)]],
    device const float* p_in [[buffer(3)]],
    device const float* du [[buffer(4)]],
    device const float* dv [[buffer(5)]],
    device const float* dw [[buffer(6)]],
    device const float* dp [[buffer(7)]],
    device const float* du_pbl [[buffer(8)]],
    device const float* dv_pbl [[buffer(9)]],
    device float* u_out [[buffer(10)]],
    device float* v_out [[buffer(11)]],
    device float* w_out [[buffer(12)]],
    device float* p_out [[buffer(13)]],
    constant SlowApplyConstants& c [[buffer(14)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= total) return;
    const float dus = (c.sanitize != 0 ? zero_if_nonfinite(du[gid] + du_pbl[gid]) : du[gid] + du_pbl[gid]);
    const float dvs = (c.sanitize != 0 ? zero_if_nonfinite(dv[gid] + dv_pbl[gid]) : dv[gid] + dv_pbl[gid]);
    u_out[gid] = u_in[gid] + dus * c.dt_large;
    v_out[gid] = v_in[gid] + dvs * c.dt_large;
    w_out[gid] = w_in[gid] + (c.sanitize != 0 ? zero_if_nonfinite(dw[gid]) : dw[gid]) * c.dt_large;
    p_out[gid] = p_in[gid] + (c.sanitize != 0 ? zero_if_nonfinite(dp[gid]) : dp[gid]) * c.dt_large;
}
)msl";

} // namespace nimbus::metal::shaders
