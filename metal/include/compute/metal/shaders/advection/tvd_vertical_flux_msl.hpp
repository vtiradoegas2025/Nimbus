#pragma once

namespace nimbus::metal::shaders
{

// Native translation of vulkan/shaders/compute/advection/tvd_vertical_flux.comp.
// One invocation evaluates one cell directly from neighboring reconstructed
// interface states. This preserves the canonical fp32 arithmetic without the
// Vulkan shader's fixed-size per-column scratch arrays.
inline constexpr const char* kTvdVerticalFluxMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nr; int nth; int nz; int limiter_id;
    int positivity; float positivity_dt; float cfl_target; float padding;
};

inline float tvd_limiter(float r, int limiter_id)
{
    if (limiter_id == 0)
        return max(0.0f, min(r, 1.0f));
    if (limiter_id == 1)
        return (r + abs(r)) / (1.0f + abs(r));
    if (limiter_id == 2)
        return max(0.0f, max(min(2.0f * r, 1.0f), min(r, 2.0f)));
    return max(0.0f, min(min(2.0f * r, 2.0f), (1.0f + r) * 0.5f));
}

inline float reconstructed_left(device const float* q, uint base, int nz, int k, int limiter_id)
{
    float center = q[base + uint(k)];
    if (k == 0 || k == nz - 1)
        return center;
    float next = q[base + uint(k + 1)];
    float r = (center - q[base + uint(k - 1)]) / (next - center + 1.0e-30f);
    float delta = tvd_limiter(r, limiter_id) * (next - center);
    return center - 0.5f * delta;
}

inline float reconstructed_right(device const float* q, uint base, int nz, int k, int limiter_id)
{
    float center = q[base + uint(k)];
    if (k == 0 || k == nz - 1)
        return center;
    float next = q[base + uint(k + 1)];
    float r = (center - q[base + uint(k - 1)]) / (next - center + 1.0e-30f);
    float delta = tvd_limiter(r, limiter_id) * (next - center);
    return center + 0.5f * delta;
}

kernel void tvd_vertical_flux(
    device const float* q       [[buffer(0)]],
    device const float* w       [[buffer(1)]],
    device const float* dz      [[buffer(2)]],
    device float* dqdt          [[buffer(3)]],
    constant PushConstants& p   [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    uint total = uint(p.nr) * uint(p.nth) * uint(p.nz);
    if (gid >= total) return;
    int k = int(gid % uint(p.nz));
    uint base = gid - uint(k);
    float tendency = 0.0f;

    if (k < p.nz - 1)
    {
        float velocity_right = w[gid];
        float flux_right = velocity_right >= 0.0f
            ? velocity_right * reconstructed_right(q, base, p.nz, k, p.limiter_id)
            : velocity_right * reconstructed_left(q, base, p.nz, k + 1, p.limiter_id);
        float flux_left = 0.0f;
        if (k > 0)
        {
            float velocity_left = w[gid - 1];
            flux_left = velocity_left >= 0.0f
                ? velocity_left * reconstructed_right(q, base, p.nz, k - 1, p.limiter_id)
                : velocity_left * reconstructed_left(q, base, p.nz, k, p.limiter_id);
        }
        tendency = -(flux_right - flux_left) / dz[uint(k)];
    }

    if (p.positivity != 0)
    {
        if (!isfinite(tendency)) tendency = 0.0f;
        float floor_tendency = -q[gid] / max(p.positivity_dt, 1.0e-12f);
        if (tendency < floor_tendency) tendency = floor_tendency;
    }
    dqdt[gid] = isfinite(tendency) ? tendency : 0.0f;
}
)msl";

} // namespace nimbus::metal::shaders
