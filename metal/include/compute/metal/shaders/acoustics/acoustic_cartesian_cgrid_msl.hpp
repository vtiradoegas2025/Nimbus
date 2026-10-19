#pragma once

#include "compute/metal/shaders/acoustics/acoustic_eos_msl.hpp"

namespace nimbus::metal::shaders
{

// Runtime-compiled translation of the canonical Cartesian C-grid acoustic
// pressure and momentum shaders. Buffer indices preserve their descriptor
// order; constants follow the final resource binding.
// clang-format off
inline constexpr const char* kAcousticCartesianCgridMsl = R"MSL(

#include <metal_stdlib>
using namespace metal;
)MSL"
NIMBUS_ACOUSTIC_EOS_MSL
R"MSL(
struct PressureConstants
{
    int   nx;
    int   ny;
    int   nz;
    float dx;
    float dy;
    float dz_val;
    float gamma_val;
    float dt_small;
    float rho_min;
    float rho_max;
    float p_min;
    float p_max;
    float p_ref_fallback;
    float pad0;
    float frame_velocity_u_ms;
    float frame_velocity_v_ms;
    int   preserve_unrepaired;
    float gamma_low;
};

kernel void acoustic_pressure_cartesian_cgrid(device const float*         u [[buffer(0)]],
                                              device const float*         v [[buffer(1)]],
                                              device const float*         w [[buffer(2)]],
                                              device const float*         rho [[buffer(3)]],
                                              device const float*         p [[buffer(4)]],
                                              device const float*         rho0 [[buffer(5)]],
                                              device const float*         p0 [[buffer(6)]],
                                              device const float*         theta [[buffer(7)]],
                                              device float*               rho_out [[buffer(8)]],
                                              device float*               p_out [[buffer(9)]],
                                              constant PressureConstants& c [[buffer(10)]],
                                              uint                        gid [[thread_position_in_grid]])
{
    uint total = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= total)
        return;
    int   k    = int(gid % uint(c.nz));
    uint  ij   = gid / uint(c.nz);
    int   j    = int(ij % uint(c.ny));
    int   i    = int(ij / uint(c.ny));
    float drho = 0.0f;
    if (i >= 1 && i <= c.nx - 2 && j >= 1 && j <= c.ny - 2)
    {
        uint  ip = uint(((i + 1) * c.ny + j) * c.nz + k), im = uint(((i - 1) * c.ny + j) * c.nz + k);
        uint  jp = uint((i * c.ny + j + 1) * c.nz + k), jm = uint((i * c.ny + j - 1) * c.nz + k);
        float fxp = .5f * (rho[gid] + rho[ip]) * (u[gid] - c.frame_velocity_u_ms);
        float fxm = .5f * (rho[im] + rho[gid]) * (u[im] - c.frame_velocity_u_ms);
        float fyp = .5f * (rho[gid] + rho[jp]) * (v[gid] - c.frame_velocity_v_ms);
        float fym = .5f * (rho[jm] + rho[gid]) * (v[jm] - c.frame_velocity_v_ms);
        float fzp = 0.0f;
        if (k + 1 < c.nz)
        {
            uint kp = uint((i * c.ny + j) * c.nz + k + 1);
            fzp     = .5f * (rho[gid] + rho[kp]) * w[gid];
        }
        else
            fzp = rho[gid] * w[gid];
        float fzm = 0.0f;
        if (k > 0)
        {
            uint km = gid - 1;
            fzm     = .5f * (rho[km] + rho[gid]) * w[km];
        }
        drho = -((fxp - fxm) / c.dx + (fyp - fym) / c.dy + (fzp - fzm) / c.dz_val);
        if (c.preserve_unrepaired != 0)
        {
            float2 flux_z_high = k + 1 < c.nz
                ? acoustic_precise_mass_flux(rho[gid], rho[gid + 1], w[gid], 0.0f)
                : df_mul(float2(rho[gid], 0), float2(w[gid], 0));
            float2 flux_z_low = k > 0
                ? acoustic_precise_mass_flux(rho[gid - 1], rho[gid], w[gid - 1], 0.0f) : float2(0);
            drho = acoustic_precise_density_rate(
                acoustic_precise_mass_flux(rho[gid], rho[ip], u[gid], c.frame_velocity_u_ms),
                acoustic_precise_mass_flux(rho[im], rho[gid], u[im], c.frame_velocity_u_ms),
                acoustic_precise_mass_flux(rho[gid], rho[jp], v[gid], c.frame_velocity_v_ms),
                acoustic_precise_mass_flux(rho[jm], rho[gid], v[jm], c.frame_velocity_v_ms),
                flux_z_high, flux_z_low, c.dx, c.dy, c.dz_val);
        }
        if (c.preserve_unrepaired == 0 && !isfinite(drho))
            drho = 0.0f;
    }
    float rn = c.preserve_unrepaired != 0 ? acoustic_precise_update(rho[gid], drho, c.dt_small)
                                         : rho[gid] + drho * c.dt_small;
    if (c.preserve_unrepaired == 0 && (!isfinite(rn) || rn <= 0.0f))
        rn = max(0.1f, rho0[k]);
    if (c.preserve_unrepaired == 0)
        rn = clamp(rn, c.rho_min, c.rho_max);
    rho_out[gid] = rn;
    float pn = as_type<float>(0x7fc00000u), th = theta[gid];
    if (th > 0.0f)
    {
        if (rn > 0.0f && isfinite(rn) && isfinite(th))
        {
            pn = acoustic_pressure_eos(rn, th, float2(c.gamma_val, c.gamma_low));
        }
        else
            pn = 100000.0f * pow(rn * 287.0f * th / 100000.0f, c.gamma_val);
    }
    if (c.preserve_unrepaired == 0 && (!isfinite(pn) || pn <= 0.0f))
        pn = c.p_ref_fallback;
    p_out[gid] = c.preserve_unrepaired != 0 ? pn : clamp(pn, c.p_min, c.p_max);
    (void)p;
    (void)p0;
}

struct MomentumConstants
{
    int   nx;
    int   ny;
    int   nz;
    float dx;
    float dy;
    float dz_val;
    float dt_small;
    float wind_clamp_h;
    float wind_clamp_v;
    int   preserve_unrepaired;
    int paired_base_gradient;
};
inline float safe_rho(float x)
{
    return isfinite(x) && x > 1.0e-6f ? x : 1.0f;
}
inline float clean(float x, int preserve)
{
    return preserve != 0 || isfinite(x) ? x : 0.0f;
}
inline float clamp_wind(float x, float limit, int preserve)
{
    return preserve != 0 ? x : (isfinite(x) ? clamp(x, -limit, limit) : 0.0f);
}

kernel void acoustic_momentum_cartesian_cgrid(device const float*         rho [[buffer(0)]],
                                              device const float*         p [[buffer(1)]],
                                              device const float*         dp0 [[buffer(2)]],
                                              device const float*         u [[buffer(3)]],
                                              device const float*         v [[buffer(4)]],
                                              device const float*         w [[buffer(5)]],
                                              device float*               u_out [[buffer(6)]],
                                              device float*               v_out [[buffer(7)]],
                                              device float*               w_out [[buffer(8)]],
                                              constant MomentumConstants& c [[buffer(9)]],
                                              uint                        gid [[thread_position_in_grid]])
{
    uint total = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= total)
        return;
    int   k      = int(gid % uint(c.nz));
    uint  ij     = gid / uint(c.nz);
    int   j      = int(ij % uint(c.ny));
    int   i      = int(ij / uint(c.ny));
    bool  active = i >= 1 && i <= c.nx - 2 && j >= 1 && j <= c.ny - 2 && k <= c.nz - 2;
    float du = 0.0f, dv = 0.0f, dw = 0.0f;
    if (active)
    {
        uint ip = uint(((i + 1) * c.ny + j) * c.nz + k), jp = uint((i * c.ny + j + 1) * c.nz + k), kp = gid + 1;
        du = clean(-((p[ip] - p[gid]) / c.dx) / safe_rho(.5f * (rho[gid] + rho[ip])), c.preserve_unrepaired);
        dv = clean(-((p[jp] - p[gid]) / c.dy) / safe_rho(.5f * (rho[gid] + rho[jp])), c.preserve_unrepaired);
        dw = clean(-(((p[kp] - p[gid]) / c.dz_val) - dp0[k]) / safe_rho(.5f * (rho[gid] + rho[kp])),
                   c.preserve_unrepaired);
        if (c.preserve_unrepaired != 0)
        {
            du = acoustic_precise_momentum_rate(p[gid], p[ip], rho[gid], rho[ip], c.dx, float2(0));
            dv = acoustic_precise_momentum_rate(p[gid], p[jp], rho[gid], rho[jp], c.dy, float2(0));
            dw = acoustic_precise_momentum_rate(p[gid], p[kp], rho[gid], rho[kp], c.dz_val,
                 float2(dp0[k], c.paired_base_gradient != 0 ? dp0[k + c.nz] : 0.0f));
        }
    }
    u_out[gid] = c.preserve_unrepaired != 0 ? acoustic_precise_update(u[gid], du, c.dt_small)
                                          : clamp_wind(u[gid] + du * c.dt_small, c.wind_clamp_h, 0);
    v_out[gid] = c.preserve_unrepaired != 0 ? acoustic_precise_update(v[gid], dv, c.dt_small)
                                          : clamp_wind(v[gid] + dv * c.dt_small, c.wind_clamp_h, 0);
    w_out[gid] = c.preserve_unrepaired != 0 ? acoustic_precise_update(w[gid], dw, c.dt_small)
                                          : clamp_wind(w[gid] + dw * c.dt_small, c.wind_clamp_v, 0);
}
)MSL";
// clang-format on

} // namespace nimbus::metal::shaders
