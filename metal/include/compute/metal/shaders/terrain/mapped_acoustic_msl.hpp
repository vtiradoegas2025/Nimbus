#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kMappedAcousticMsl = R"msl(

#include <metal_stdlib>
using namespace metal;

struct MappedMassContinuityConstants
{
    int   nx;
    int   ny;
    int   nz;
    int   stage;
    float dx;
    float dy;
    float dzeta;
    float gamma_value;
    float frame_u_ms;
    float frame_v_ms;
    float pad0;
    float pad1;
    float pad2;
    float pad3;
    float pad4;
    float pad5;
};

struct MappedFastMomentumConstants
{
    int   nx;
    int   ny;
    int   nz;
    int   sampled_reference;
    float dx;
    float dy;
    float nominal_dz;
    float pad1;
};

inline uint mapped_index3(int i, int j, int k, int ny, int nz)
{
    return (uint(i) * uint(ny) + uint(j)) * uint(nz) + uint(k);
}

kernel void mapped_mass_continuity(device float*                           a0 [[buffer(0)]],
                                   device float*                           a1 [[buffer(1)]],
                                   device float*                           a2 [[buffer(2)]],
                                   device float*                           a3 [[buffer(3)]],
                                   device float*                           a4 [[buffer(4)]],
                                   device float*                           a5 [[buffer(5)]],
                                   device float*                           a6 [[buffer(6)]],
                                   device float*                           a7 [[buffer(7)]],
                                   device float*                           a8 [[buffer(8)]],
                                   device float*                           a9 [[buffer(9)]],
                                   device float*                           a10 [[buffer(10)]],
                                   device float*                           a11 [[buffer(11)]],
                                   device float*                           a12 [[buffer(12)]],
                                   device float*                           a13 [[buffer(13)]],
                                   device float*                           a14 [[buffer(14)]],
                                   device float*                           a15 [[buffer(15)]],
                                   device float*                           a16 [[buffer(16)]],
                                   device float*                           a17 [[buffer(17)]],
                                   constant MappedMassContinuityConstants& c [[buffer(18)]],
                                   uint                                    gid [[thread_position_in_grid]])
{
    const uint count = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= count)
        return;
    const int  k  = int(gid % uint(c.nz));
    const int  j  = int((gid / uint(c.nz)) % uint(c.ny));
    const int  i  = int(gid / (uint(c.ny) * uint(c.nz)));
    const auto at = [&](int ii, int jj, int kk) { return mapped_index3(ii, jj, kk, c.ny, c.nz); };

    if (c.stage == 0)
    {
        a5[gid] = 0.0f;
        if (k == 0 && i > 0 && j > 0)
        {
            const float ground_u = 0.5f * (a0[at(i - 1, j, 0)] + a0[gid]) - c.frame_u_ms;
            const float ground_v = 0.5f * (a1[at(i, j - 1, 0)] + a1[gid]) - c.frame_v_ms;
            a5[gid]              = ground_u * (-a3[gid]) + ground_v * (-a4[gid]);
        }
        return;
    }

    const auto rho_x_face = [&](int ii, int jj, int kk)
    { return 0.5f * (a0[at(ii, jj, kk)] + a0[at(ii + 1, jj, kk)]); };
    const auto rho_y_face = [&](int ii, int jj, int kk)
    { return 0.5f * (a0[at(ii, jj, kk)] + a0[at(ii, jj + 1, kk)]); };
    const auto rho_u_center = [&](int ii, int jj, int kk)
    {
        return 0.5f * (rho_x_face(ii - 1, jj, kk) * (a1[at(ii - 1, jj, kk)] - c.frame_u_ms) +
                       rho_x_face(ii, jj, kk) * (a1[at(ii, jj, kk)] - c.frame_u_ms));
    };
    const auto rho_v_center = [&](int ii, int jj, int kk)
    {
        return 0.5f * (rho_y_face(ii, jj - 1, kk) * (a2[at(ii, jj - 1, kk)] - c.frame_v_ms) +
                       rho_y_face(ii, jj, kk) * (a2[at(ii, jj, kk)] - c.frame_v_ms));
    };
    const auto u_center = [&](int ii, int jj, int kk) { return 0.5f * (a1[at(ii - 1, jj, kk)] + a1[at(ii, jj, kk)]); };
    const auto v_center = [&](int ii, int jj, int kk) { return 0.5f * (a2[at(ii, jj - 1, kk)] + a2[at(ii, jj, kk)]); };
    const auto mass_flux_x = [&](int ii, int jj, int kk)
    {
        const uint n = at(ii, jj, kk);
        return a6[n] * rho_x_face(ii, jj, kk) * (a1[n] - c.frame_u_ms);
    };
    const auto mass_flux_y = [&](int ii, int jj, int kk)
    {
        const uint n = at(ii, jj, kk);
        return a7[n] * rho_y_face(ii, jj, kk) * (a2[n] - c.frame_v_ms);
    };
    const auto mass_flux_zeta = [&](int ii, int jj, int kk)
    {
        const uint  n     = at(ii, jj, kk);
        const int   upper = min(kk + 1, c.nz - 1);
        const float rho_w = (kk + 1 < c.nz ? 0.5f * (a0[n] + a0[at(ii, jj, kk + 1)]) : a0[n]) * a3[n];
        const float rho_u =
            kk + 1 < c.nz ? 0.5f * (rho_u_center(ii, jj, kk) + rho_u_center(ii, jj, upper)) : rho_u_center(ii, jj, kk);
        const float rho_v =
            kk + 1 < c.nz ? 0.5f * (rho_v_center(ii, jj, kk) + rho_v_center(ii, jj, upper)) : rho_v_center(ii, jj, kk);
        return rho_w + a8[n] * rho_u + a9[n] * rho_v;
    };
    const auto zeta_velocity = [&](int ii, int jj, int kk)
    {
        const uint  n     = at(ii, jj, kk);
        const int   upper = min(kk + 1, c.nz - 1);
        const float uf = kk + 1 < c.nz ? 0.5f * (u_center(ii, jj, kk) + u_center(ii, jj, upper)) : u_center(ii, jj, kk);
        const float vf = kk + 1 < c.nz ? 0.5f * (v_center(ii, jj, kk) + v_center(ii, jj, upper)) : v_center(ii, jj, kk);
        return a3[n] + a8[n] * (uf - c.frame_u_ms) + a9[n] * (vf - c.frame_v_ms);
    };

    a10[gid] = 0.0f;
    a11[gid] = 0.0f;
    a12[gid] = 0.0f;
    a13[gid] = 0.0f;
    a14[gid] = 0.0f;
    if (i < c.nx - 1)
        a12[gid] = mass_flux_x(i, j, k);
    if (j < c.ny - 1)
        a13[gid] = mass_flux_y(i, j, k);
    if (i > 0 && i < c.nx - 1 && j > 0 && j < c.ny - 1)
        a14[gid] = mass_flux_zeta(i, j, k);
    if (i > 0 && i < c.nx - 1 && j > 0 && j < c.ny - 1)
    {
        const float transformed_mass =
            (mass_flux_x(i, j, k) - mass_flux_x(i - 1, j, k)) / c.dx +
            (mass_flux_y(i, j, k) - mass_flux_y(i, j - 1, k)) / c.dy +
            (mass_flux_zeta(i, j, k) - (k > 0 ? mass_flux_zeta(i, j, k - 1) : 0.0f)) / c.dzeta;
        a10[gid] = -transformed_mass / a5[gid];
        const float transformed_velocity =
            (a6[gid] * (a1[gid] - c.frame_u_ms) - a6[at(i - 1, j, k)] * (a1[at(i - 1, j, k)] - c.frame_u_ms)) / c.dx +
            (a7[gid] * (a2[gid] - c.frame_v_ms) - a7[at(i, j - 1, k)] * (a2[at(i, j - 1, k)] - c.frame_v_ms)) / c.dy +
            (zeta_velocity(i, j, k) - (k > 0 ? zeta_velocity(i, j, k - 1) : 0.0f)) / c.dzeta;
        a11[gid] = -c.gamma_value * a4[gid] * transformed_velocity / a5[gid];
    }
    (void)a15;
    (void)a16;
    (void)a17;
}

inline float
mapped_reference_at(device const float* reference_pressure, float z, constant MappedFastMomentumConstants& c)
{
    if (c.nz == 1)
        return reference_pressure[0];
    const float fractional = z / c.nominal_dz - 0.5f;
    const int   lower      = clamp(int(floor(fractional)), 0, c.nz - 2);
    const float weight     = fractional - float(lower);
    return reference_pressure[lower] + weight * (reference_pressure[lower + 1] - reference_pressure[lower]);
}

kernel void mapped_fast_momentum(device const float*                   rho [[buffer(0)]],
                                 device const float*                   pressure [[buffer(1)]],
                                 device const float*                   height [[buffer(2)]],
                                 device const float*                   reference_pressure [[buffer(3)]],
                                 device float*                         du_dt [[buffer(4)]],
                                 device float*                         dv_dt [[buffer(5)]],
                                 device float*                         dw_dt [[buffer(6)]],
                                 constant MappedFastMomentumConstants& c [[buffer(7)]],
                                 uint                                  gid [[thread_position_in_grid]])
{
    const uint count = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= count)
        return;
    const int  k            = int(gid % uint(c.nz));
    const int  j            = int((gid / uint(c.nz)) % uint(c.ny));
    const int  i            = int(gid / (uint(c.ny) * uint(c.nz)));
    const auto at           = [&](int ii, int jj, int kk) { return mapped_index3(ii, jj, kk, c.ny, c.nz); };
    const auto perturbation = [&](int ii, int jj, int kk)
    {
        const uint n = at(ii, jj, kk);
        return pressure[n] - (c.sampled_reference != 0 ? reference_pressure[uint(c.nz) + 3u * n]
                                                       : mapped_reference_at(reference_pressure, height[n], c));
    };
    const auto column_gradient = [&](int ii, int jj, int kk)
    {
        const int lower = kk == 0 ? 0 : kk - 1;
        const int upper = kk + 1 < c.nz ? kk + 1 : c.nz - 1;
        return (perturbation(ii, jj, upper) - perturbation(ii, jj, lower)) /
               (height[at(ii, jj, upper)] - height[at(ii, jj, lower)]);
    };
    du_dt[gid] = 0.0f;
    dv_dt[gid] = 0.0f;
    dw_dt[gid] = 0.0f;
    if (i <= 0 || i >= c.nx - 1 || j <= 0 || j >= c.ny - 1 || k >= c.nz - 1)
        return;
    const float gradient_x = (perturbation(i + 1, j, k) - perturbation(i, j, k)) / c.dx -
                             (height[at(i + 1, j, k)] - height[at(i, j, k)]) / c.dx * 0.5f *
                                 (column_gradient(i, j, k) + column_gradient(i + 1, j, k));
    const float gradient_y = (perturbation(i, j + 1, k) - perturbation(i, j, k)) / c.dy -
                             (height[at(i, j + 1, k)] - height[at(i, j, k)]) / c.dy * 0.5f *
                                 (column_gradient(i, j, k) + column_gradient(i, j + 1, k));
    const float gradient_z =
        (perturbation(i, j, k + 1) - perturbation(i, j, k)) / (height[at(i, j, k + 1)] - height[at(i, j, k)]);
    du_dt[gid] = -gradient_x / (0.5f * (rho[gid] + rho[at(i + 1, j, k)]));
    dv_dt[gid] = -gradient_y / (0.5f * (rho[gid] + rho[at(i, j + 1, k)]));
    dw_dt[gid] = -gradient_z / (0.5f * (rho[gid] + rho[at(i, j, k + 1)]));
}
)msl";

} // namespace nimbus::metal::shaders
