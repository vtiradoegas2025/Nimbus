#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kMicrophysicsApplyMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nx; int ny; int nz; int pad0;
    int i_begin; int i_end; int j_begin; int j_end;
    float dt_step; float theta_fallback; float theta_min; float theta_max;
    float max_theta_step; float qv_max; float hydro_max; float pad1;
};

inline float guarded_tendency(float value, thread uint& nonfinite_count)
{
    if (!isfinite(value))
    {
        ++nonfinite_count;
        return 0.0f;
    }
    return value;
}

kernel void microphysics_apply(
    device float* theta               [[buffer(0)]],
    device float* qv                  [[buffer(1)]],
    device float* qc                  [[buffer(2)]],
    device float* qr                  [[buffer(3)]],
    device float* qi                  [[buffer(4)]],
    device float* qs                  [[buffer(5)]],
    device float* qg                  [[buffer(6)]],
    device float* qh                  [[buffer(7)]],
    device const float* dtheta        [[buffer(8)]],
    device const float* dqv           [[buffer(9)]],
    device const float* dqc           [[buffer(10)]],
    device const float* dqr           [[buffer(11)]],
    device const float* dqi           [[buffer(12)]],
    device const float* dqs           [[buffer(13)]],
    device const float* dqg           [[buffer(14)]],
    device const float* dqh           [[buffer(15)]],
    device const float* dtheta_rad    [[buffer(16)]],
    device const float* dtheta_pbl    [[buffer(17)]],
    device const float* dqv_pbl       [[buffer(18)]],
    device atomic_uint* stats         [[buffer(19)]],
    constant PushConstants& p         [[buffer(20)]],
    uint gid                          [[thread_position_in_grid]])
{
    uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total) return;
    int k = int(gid % uint(p.nz));
    int i = int(gid / (uint(p.ny) * uint(p.nz)));
    int j = int((gid / uint(p.nz)) % uint(p.ny));
    if (i < p.i_begin || i >= p.i_end || j < p.j_begin || j >= p.j_end) return;

    uint nonfinite_tendency = 0u;
    float dth = guarded_tendency(dtheta[gid], nonfinite_tendency);
    float dv = guarded_tendency(dqv[gid], nonfinite_tendency);
    float dc = guarded_tendency(dqc[gid], nonfinite_tendency);
    float dr = guarded_tendency(dqr[gid], nonfinite_tendency);
    float di = guarded_tendency(dqi[gid], nonfinite_tendency);
    float ds = guarded_tendency(dqs[gid], nonfinite_tendency);
    float dg = guarded_tendency(dqg[gid], nonfinite_tendency);
    float dh = guarded_tendency(dqh[gid], nonfinite_tendency);

    float dtheta_total = (dth + dtheta_rad[gid] + dtheta_pbl[gid]) * p.dt_step;
    bool nonfinite_total = !isfinite(dtheta_total);
    if (nonfinite_total) dtheta_total = 0.0f;
    float dtheta_limited = clamp(dtheta_total, -p.max_theta_step, p.max_theta_step);
    bool limited = dtheta_limited != dtheta_total;

    float theta_old = theta[gid];
    float theta_new = theta_old + dtheta_limited;
    float theta_safe = isfinite(theta_new) ? theta_new : (isfinite(theta_old) ? theta_old : p.theta_fallback);
    float theta_bounded = clamp(theta_safe, p.theta_min, p.theta_max);
    bool theta_clamped = theta_bounded != theta_new || !isfinite(theta_new);
    theta[gid] = theta_bounded;

    float qv_new = qv[gid] + (dv + dqv_pbl[gid]) * p.dt_step;
    qv[gid] = isfinite(qv_new) ? clamp(qv_new, 0.0f, p.qv_max) : 0.0f;
    float qc_new = qc[gid] + dc * p.dt_step;
    qc[gid] = isfinite(qc_new) ? clamp(qc_new, 0.0f, p.hydro_max) : 0.0f;
    float qr_new = qr[gid] + dr * p.dt_step;
    qr[gid] = isfinite(qr_new) ? clamp(qr_new, 0.0f, p.hydro_max) : 0.0f;
    float qi_new = qi[gid] + di * p.dt_step;
    qi[gid] = isfinite(qi_new) ? clamp(qi_new, 0.0f, p.hydro_max) : 0.0f;
    float qs_new = qs[gid] + ds * p.dt_step;
    qs[gid] = isfinite(qs_new) ? clamp(qs_new, 0.0f, p.hydro_max) : 0.0f;
    float qg_new = qg[gid] + dg * p.dt_step;
    qg[gid] = isfinite(qg_new) ? clamp(qg_new, 0.0f, p.hydro_max) : 0.0f;
    float qh_new = qh[gid] + dh * p.dt_step;
    qh[gid] = isfinite(qh_new) ? clamp(qh_new, 0.0f, p.hydro_max) : 0.0f;

    if (nonfinite_tendency > 0u)
        atomic_fetch_add_explicit(&stats[0], nonfinite_tendency, memory_order_relaxed);
    if (nonfinite_total)
        atomic_fetch_add_explicit(&stats[1], 1u, memory_order_relaxed);
    if (limited)
        atomic_fetch_add_explicit(&stats[2], 1u, memory_order_relaxed);
    if (theta_clamped)
    {
        atomic_fetch_add_explicit(&stats[3], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[10u + uint(k)], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[10u + uint(p.nz) + uint(i)], 1u, memory_order_relaxed);
        if (atomic_exchange_explicit(&stats[4], 1u, memory_order_relaxed) == 0u)
        {
            atomic_store_explicit(&stats[5], gid, memory_order_relaxed);
            atomic_store_explicit(&stats[6], as_type<uint>(theta_new), memory_order_relaxed);
            atomic_store_explicit(&stats[7], as_type<uint>(theta_bounded), memory_order_relaxed);
            atomic_store_explicit(&stats[8], as_type<uint>(dtheta_total), memory_order_relaxed);
            atomic_store_explicit(&stats[9], as_type<uint>(dtheta_limited), memory_order_relaxed);
        }
    }
}
)msl";

} // namespace nimbus::metal::shaders
