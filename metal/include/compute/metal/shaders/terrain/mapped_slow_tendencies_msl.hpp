#pragma once

namespace nimbus::metal::shaders
{

inline constexpr const char* kMappedSlowTendenciesMsl = R"msl(

#include <metal_stdlib>
using namespace metal;

struct MappedSlowConstants
{
    int   nx;
    int   ny;
    int   nz;
    int   has_moisture;
    int   has_loading;
    int   sampled_reference;
    int   pad_i1;
    int   pad_i2;
    float dx;
    float dy;
    float dzeta;
    float nominal_dz;
    float gravity;
    float pad_f0;
    float pad_f1;
    float pad_f2;
};

kernel void mapped_slow_tendencies(device const float*           u [[buffer(0)]],
                                   device const float*           v [[buffer(1)]],
                                   device const float*           w [[buffer(2)]],
                                   device const float*           rho [[buffer(3)]],
                                   device const float*           pressure [[buffer(4)]],
                                   device const float*           height [[buffer(5)]],
                                   device const float*           jacobian_x [[buffer(6)]],
                                   device const float*           jacobian_y [[buffer(7)]],
                                   device const float*           metric_x [[buffer(8)]],
                                   device const float*           metric_y [[buffer(9)]],
                                   device const float*           ground_w [[buffer(10)]],
                                   device const float*           reference_pressure [[buffer(11)]],
                                   device const float*           reference_theta [[buffer(12)]],
                                   device const float*           qv [[buffer(13)]],
                                   device const float*           reference_qv [[buffer(14)]],
                                   device const float*           loading [[buffer(15)]],
                                   device float*                 du_dt [[buffer(16)]],
                                   device float*                 dv_dt [[buffer(17)]],
                                   device float*                 dw_dt [[buffer(18)]],
                                   device float*                 drho_dt [[buffer(19)]],
                                   device float*                 dp_dt [[buffer(20)]],
                                   device float*                 adv_u_dt [[buffer(21)]],
                                   device float*                 adv_v_dt [[buffer(22)]],
                                   device float*                 adv_w_dt [[buffer(23)]],
                                   constant MappedSlowConstants& c [[buffer(24)]],
                                   uint                          gid [[thread_position_in_grid]])
{
    const uint count = uint(c.nx) * uint(c.ny) * uint(c.nz);
    if (gid >= count)
        return;
    const int  k  = int(gid % uint(c.nz));
    const int  j  = int((gid / uint(c.nz)) % uint(c.ny));
    const int  i  = int(gid / (uint(c.ny) * uint(c.nz)));
    const auto at = [&](int ii, int jj, int kk) { return (uint(ii) * uint(c.ny) + uint(jj)) * uint(c.nz) + uint(kk); };
    const auto field_value = [&](int field, int ii, int jj, int kk)
    {
        const uint n = at(ii, jj, kk);
        return field == 0 ? u[n] : (field == 1 ? v[n] : w[n]);
    };
    const auto fd1 = [&](float m1, float center, float p1, float a, float inv_d)
    { return a >= 0.0f ? (center - m1) * inv_d : (p1 - center) * inv_d; };
    const auto fd3 = [&](float m2, float m1, float center, float p1, float p2, float a, float inv_d)
    {
        return a >= 0.0f ? (2.0f * p1 + 3.0f * center - 6.0f * m1 + m2) * (inv_d / 6.0f)
                         : (-p2 + 6.0f * p1 - 3.0f * center - 2.0f * m1) * (inv_d / 6.0f);
    };
    const auto fd5 = [&](float m3, float m2, float m1, float center, float p1, float p2, float p3, float a, float inv_d)
    {
        return a >= 0.0f
                   ? (-2.0f * m3 + 15.0f * m2 - 60.0f * m1 + 20.0f * center + 30.0f * p1 - 3.0f * p2) * (inv_d / 60.0f)
                   : (2.0f * p3 - 15.0f * p2 + 60.0f * p1 - 20.0f * center - 30.0f * m1 + 3.0f * m2) * (inv_d / 60.0f);
    };
    const auto upwind_x = [&](int field, float a, int ii, int jj, int kk)
    {
        const int   half_width = min(ii, c.nx - 1 - ii);
        const float center     = field_value(field, ii, jj, kk);
        if (half_width <= 0)
            return 0.0f;
        const float m1 = field_value(field, ii - 1, jj, kk);
        const float p1 = field_value(field, ii + 1, jj, kk);
        if (half_width == 1)
            return fd1(m1, center, p1, a, 1.0f / c.dx);
        const float m2 = field_value(field, ii - 2, jj, kk);
        const float p2 = field_value(field, ii + 2, jj, kk);
        if (half_width == 2)
            return fd3(m2, m1, center, p1, p2, a, 1.0f / c.dx);
        return fd5(field_value(field, ii - 3, jj, kk),
                   m2,
                   m1,
                   center,
                   p1,
                   p2,
                   field_value(field, ii + 3, jj, kk),
                   a,
                   1.0f / c.dx);
    };
    const auto upwind_y = [&](int field, float a, int ii, int jj, int kk)
    {
        const int   half_width = min(jj, c.ny - 1 - jj);
        const float center     = field_value(field, ii, jj, kk);
        if (half_width <= 0)
            return 0.0f;
        const float m1 = field_value(field, ii, jj - 1, kk);
        const float p1 = field_value(field, ii, jj + 1, kk);
        if (half_width == 1)
            return fd1(m1, center, p1, a, 1.0f / c.dy);
        const float m2 = field_value(field, ii, jj - 2, kk);
        const float p2 = field_value(field, ii, jj + 2, kk);
        if (half_width == 2)
            return fd3(m2, m1, center, p1, p2, a, 1.0f / c.dy);
        return fd5(field_value(field, ii, jj - 3, kk),
                   m2,
                   m1,
                   center,
                   p1,
                   p2,
                   field_value(field, ii, jj + 3, kk),
                   a,
                   1.0f / c.dy);
    };
    const auto upwind_z = [&](int field, float a, int ii, int jj, int kk)
    {
        const int   half_width = min(kk, c.nz - 1 - kk);
        const float center     = field_value(field, ii, jj, kk);
        if (half_width <= 0)
            return 0.0f;
        const float m1 = field_value(field, ii, jj, kk - 1);
        const float p1 = field_value(field, ii, jj, kk + 1);
        if (half_width == 1)
            return fd1(m1, center, p1, a, 1.0f / c.nominal_dz);
        const float m2 = field_value(field, ii, jj, kk - 2);
        const float p2 = field_value(field, ii, jj, kk + 2);
        if (half_width == 2)
            return fd3(m2, m1, center, p1, p2, a, 1.0f / c.nominal_dz);
        return fd5(field_value(field, ii, jj, kk - 3),
                   m2,
                   m1,
                   center,
                   p1,
                   p2,
                   field_value(field, ii, jj, kk + 3),
                   a,
                   1.0f / c.nominal_dz);
    };
    const auto profile_at = [&](int profile, float z)
    {
        const float fractional = z / c.nominal_dz - 0.5f;
        const int   lower      = clamp(int(floor(fractional)), 0, c.nz - 2);
        const float weight     = fractional - float(lower);
        const float lower_value =
            profile == 0 ? reference_pressure[lower] : (profile == 1 ? reference_theta[lower] : reference_qv[lower]);
        const float upper_value = profile == 0 ? reference_pressure[lower + 1]
                                               : (profile == 1 ? reference_theta[lower + 1] : reference_qv[lower + 1]);
        return lower_value + weight * (upper_value - lower_value);
    };
    const auto reference_density = [&](uint n)
    {
        if (c.sampled_reference != 0)
            return reference_pressure[uint(c.nz) + 3u * n + 1u];
        const float z           = height[n];
        const float p0          = profile_at(0, z);
        const float theta0      = profile_at(1, z);
        const float temperature = theta0 * pow(p0 / 100000.0f, 287.0f / 1004.0f);
        return p0 / (287.0f * temperature);
    };
    const auto reference_vapor = [&](uint n)
    { return c.sampled_reference != 0 ? reference_pressure[uint(c.nz) + 3u * n + 2u] : profile_at(2, height[n]); };
    const auto pressure_prime = [&](int ii, int jj, int kk)
    {
        const uint n = at(ii, jj, kk);
        return pressure[n] -
               (c.sampled_reference != 0 ? reference_pressure[uint(c.nz) + 3u * n] : profile_at(0, height[n]));
    };
    const auto column_gradient_prime = [&](int ii, int jj, int kk)
    {
        const int lower = kk == 0 ? 0 : kk - 1;
        const int upper = kk + 1 < c.nz ? kk + 1 : c.nz - 1;
        return (pressure_prime(ii, jj, upper) - pressure_prime(ii, jj, lower)) /
               (height[at(ii, jj, upper)] - height[at(ii, jj, lower)]);
    };
    const auto gradient_prime_x = [&](int ii, int jj, int kk)
    {
        const float computational = (pressure_prime(ii + 1, jj, kk) - pressure_prime(ii, jj, kk)) / c.dx;
        const float slope         = (height[at(ii + 1, jj, kk)] - height[at(ii, jj, kk)]) / c.dx;
        return computational -
               slope * 0.5f * (column_gradient_prime(ii, jj, kk) + column_gradient_prime(ii + 1, jj, kk));
    };
    const auto gradient_prime_y = [&](int ii, int jj, int kk)
    {
        const float computational = (pressure_prime(ii, jj + 1, kk) - pressure_prime(ii, jj, kk)) / c.dy;
        const float slope         = (height[at(ii, jj + 1, kk)] - height[at(ii, jj, kk)]) / c.dy;
        return computational -
               slope * 0.5f * (column_gradient_prime(ii, jj, kk) + column_gradient_prime(ii, jj + 1, kk));
    };
    const auto zeta_x = [&](float ux, float uy, float wz, int ii, int jj, int kk)
    {
        const float slope_x = (height[at(ii + 1, jj, kk)] - height[at(ii, jj, kk)]) / c.dx;
        const float north   = 0.5f * (height[at(ii, jj + 1, kk)] + height[at(ii + 1, jj + 1, kk)]);
        const float south   = 0.5f * (height[at(ii, jj - 1, kk)] + height[at(ii + 1, jj - 1, kk)]);
        return (wz - ux * slope_x - uy * (north - south) / (2.0f * c.dy)) / jacobian_x[at(ii, jj, kk)];
    };
    const auto zeta_y = [&](float ux, float uy, float wz, int ii, int jj, int kk)
    {
        const float east    = 0.5f * (height[at(ii + 1, jj, kk)] + height[at(ii + 1, jj + 1, kk)]);
        const float west    = 0.5f * (height[at(ii - 1, jj, kk)] + height[at(ii - 1, jj + 1, kk)]);
        const float slope_y = (height[at(ii, jj + 1, kk)] - height[at(ii, jj, kk)]) / c.dy;
        return (wz - ux * (east - west) / (2.0f * c.dx) - uy * slope_y) / jacobian_y[at(ii, jj, kk)];
    };
    const auto zeta_z = [&](float ux, float uy, float wz, int ii, int jj, int kk)
    {
        const uint  n        = at(ii, jj, kk);
        const float jacobian = (height[at(ii, jj, kk + 1)] - height[n]) / c.dzeta;
        return (wz + metric_x[n] * ux + metric_y[n] * uy) / jacobian;
    };

    du_dt[gid]    = 0.0f;
    dv_dt[gid]    = 0.0f;
    dw_dt[gid]    = 0.0f;
    drho_dt[gid]  = 0.0f;
    dp_dt[gid]    = 0.0f;
    adv_u_dt[gid] = 0.0f;
    adv_v_dt[gid] = 0.0f;
    adv_w_dt[gid] = 0.0f;
    if (i <= 0 || i >= c.nx - 1 || j <= 0 || j >= c.ny - 1)
        return;

    if (k < c.nz - 1)
    {
        const float ux   = u[gid];
        const float uy_x = 0.25f * (v[at(i, j - 1, k)] + v[gid] + v[at(i + 1, j - 1, k)] + v[at(i + 1, j, k)]);
        const float wx =
            k == 0 ? 0.25f * (ground_w[at(i, j, 0)] + ground_w[at(i + 1, j, 0)] + w[at(i, j, 0)] + w[at(i + 1, j, 0)])
                   : 0.25f * (w[at(i, j, k - 1)] + w[gid] + w[at(i + 1, j, k - 1)] + w[at(i + 1, j, k)]);
        const float az_x = zeta_x(ux, uy_x, wx, i, j, k);
        const float duz =
            k == 0 ? (az_x >= 0.0f ? 0.0f : (u[at(i, j, 1)] - ux) / c.nominal_dz) : upwind_z(0, az_x, i, j, k);
        const float adv_x = -ux * upwind_x(0, ux, i, j, k) - uy_x * upwind_y(0, uy_x, i, j, k) - az_x * duz;
        du_dt[gid]        = adv_x;
        adv_u_dt[gid]     = adv_x;

        const float uy   = v[gid];
        const float ux_y = 0.25f * (u[at(i - 1, j, k)] + u[gid] + u[at(i - 1, j + 1, k)] + u[at(i, j + 1, k)]);
        const float wy =
            k == 0 ? 0.25f * (ground_w[at(i, j, 0)] + ground_w[at(i, j + 1, 0)] + w[at(i, j, 0)] + w[at(i, j + 1, 0)])
                   : 0.25f * (w[at(i, j, k - 1)] + w[gid] + w[at(i, j + 1, k - 1)] + w[at(i, j + 1, k)]);
        const float az_y = zeta_y(ux_y, uy, wy, i, j, k);
        const float dvz =
            k == 0 ? (az_y >= 0.0f ? 0.0f : (v[at(i, j, 1)] - uy) / c.nominal_dz) : upwind_z(1, az_y, i, j, k);
        const float adv_y = -ux_y * upwind_x(1, ux_y, i, j, k) - uy * upwind_y(1, uy, i, j, k) - az_y * dvz;
        dv_dt[gid]        = adv_y;
        adv_v_dt[gid]     = adv_y;

        const float ux_w      = 0.25f * (u[at(i - 1, j, k)] + u[gid] + u[at(i - 1, j, k + 1)] + u[at(i, j, k + 1)]);
        const float uy_w      = 0.25f * (v[at(i, j - 1, k)] + v[gid] + v[at(i, j - 1, k + 1)] + v[at(i, j, k + 1)]);
        const float az_w      = zeta_z(ux_w, uy_w, w[gid], i, j, k);
        const float dwz       = k == 0 ? (az_w >= 0.0f ? (w[gid] - ground_w[at(i, j, 0)]) / c.nominal_dz
                                                       : (w[at(i, j, 1)] - w[gid]) / c.nominal_dz)
                                       : upwind_z(2, az_w, i, j, k);
        const float adv_w     = -ux_w * upwind_x(2, ux_w, i, j, k) - uy_w * upwind_y(2, uy_w, i, j, k) - az_w * dwz;
        const float rho_face  = 0.5f * (rho[gid] + rho[at(i, j, k + 1)]);
        const float rho0_face = 0.5f * (reference_density(gid) + reference_density(at(i, j, k + 1)));
        const float density_departure = rho_face - rho0_face;
        const float vapor_departure   = c.has_moisture != 0
                                            ? 0.5f * (qv[gid] + qv[at(i, j, k + 1)]) -
                                                  0.5f * (reference_vapor(gid) + reference_vapor(at(i, j, k + 1)))
                                            : 0.0f;
        const float moisture          = c.gravity * 0.608f * vapor_departure;
        const float load = c.has_loading != 0 ? -c.gravity * 0.5f * (loading[gid] + loading[at(i, j, k + 1)]) : 0.0f;
        dw_dt[gid]       = adv_w - c.gravity * density_departure / rho_face + moisture + load;
        adv_w_dt[gid]    = adv_w;
    }
    if (k > 0 && k < c.nz - 1)
    {
        const float center_u = 0.5f * (u[gid] + u[at(i - 1, j, k)]);
        const float center_v = 0.5f * (v[gid] + v[at(i, j - 1, k)]);
        const float center_w = 0.5f * (w[gid] + w[at(i, j, k - 1)]);
        const float gx       = 0.5f * (gradient_prime_x(i - 1, j, k) + gradient_prime_x(i, j, k));
        const float gy       = 0.5f * (gradient_prime_y(i, j - 1, k) + gradient_prime_y(i, j, k));
        const float gz       = (pressure[at(i, j, k + 1)] - pressure[at(i, j, k - 1)]) /
                               (height[at(i, j, k + 1)] - height[at(i, j, k - 1)]);
        dp_dt[gid]           = -center_u * gx - center_v * gy - center_w * gz;
    }
}
)msl";

} // namespace nimbus::metal::shaders
