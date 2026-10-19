#pragma once

namespace nimbus::metal::shaders
{

// Runtime-compiled translation of
// vulkan/shaders/compute/diffusion/diffusion.comp.
inline constexpr const char* kDiffusionMsl = R"msl(
#include <metal_stdlib>
using namespace metal;

struct PushConstants
{
    int nx;
    int ny;
    int nz;
    int location;
    int c_grid;
    int periodic_x;
    int periodic_y;
    int use_variable_K;
    float inv_dx2;
    float inv_dy2;
    float K_h;
    float K_v;
    float padding0;
    float padding1;
    float padding2;
    float padding3;
};

inline uint index_of(int i, int j, int k, constant PushConstants& p)
{
    return uint((i * p.ny + j) * p.nz + k);
}

inline float scalar_density(device const float* rho, int i, int j, int k,
                            constant PushConstants& p)
{
    return rho[index_of(i, j, k, p)];
}

inline float component_density(device const float* rho, device const float* dz,
                               int i, int j, int k, constant PushConstants& p)
{
    if (p.c_grid == 0 || p.location == 0)
        return scalar_density(rho, i, j, k, p);
    if (p.location == 1)
    {
        int right = p.periodic_x != 0 && i == p.nx - 2 ? 1 : i + 1;
        return 0.5f * (scalar_density(rho, i, j, k, p) +
                       scalar_density(rho, right, j, k, p));
    }
    if (p.location == 2)
    {
        int north = p.periodic_y != 0 && j == p.ny - 2 ? 1 : j + 1;
        return 0.5f * (scalar_density(rho, i, j, k, p) +
                       scalar_density(rho, i, north, k, p));
    }
    return (scalar_density(rho, i, j, k, p) * dz[k] +
            scalar_density(rho, i, j, k + 1, p) * dz[k + 1]) /
           (dz[k] + dz[k + 1]);
}

inline float cell_vertical_K(device const float* vertical_K,
                             int i, int j, int k,
                             constant PushConstants& p)
{
    return p.use_variable_K != 0
        ? vertical_K[index_of(i, j, k, p)] : p.K_v;
}

inline float component_transport(device const float* rho,
                                 device const float* vertical_K,
                                 int i, int j, int k,
                                 constant PushConstants& p)
{
    if (p.c_grid == 0 || p.location == 0)
        return scalar_density(rho, i, j, k, p) *
            cell_vertical_K(vertical_K, i, j, k, p);
    if (p.location == 1)
    {
        int right = p.periodic_x != 0 && i == p.nx - 2 ? 1 : i + 1;
        return 0.5f * (
            scalar_density(rho, i, j, k, p) *
                cell_vertical_K(vertical_K, i, j, k, p) +
            scalar_density(rho, right, j, k, p) *
                cell_vertical_K(vertical_K, right, j, k, p));
    }
    if (p.location == 2)
    {
        int north = p.periodic_y != 0 && j == p.ny - 2 ? 1 : j + 1;
        return 0.5f * (
            scalar_density(rho, i, j, k, p) *
                cell_vertical_K(vertical_K, i, j, k, p) +
            scalar_density(rho, i, north, k, p) *
                cell_vertical_K(vertical_K, i, north, k, p));
    }
    return 0.5f * (
        scalar_density(rho, i, j, k, p) *
            cell_vertical_K(vertical_K, i, j, k, p) +
        scalar_density(rho, i, j, k + 1, p) *
            cell_vertical_K(vertical_K, i, j, k + 1, p));
}

kernel void diffusion(
    device const float* field    [[buffer(0)]],
    device const float* rho      [[buffer(1)]],
    device const float* dz       [[buffer(2)]],
    device float* tendency       [[buffer(3)]],
    device const float* vertical_K [[buffer(4)]],
    constant PushConstants& p    [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]])
{
    uint total = uint(p.nx) * uint(p.ny) * uint(p.nz);
    if (gid >= total)
        return;

    int k = int(gid % uint(p.nz));
    uint ij = gid / uint(p.nz);
    int j = int(ij % uint(p.ny));
    int i = int(ij / uint(p.ny));
    tendency[gid] = 0.0f;

    int i_begin = 1;
    int i_end = p.nx - 1;
    int j_begin = 1;
    int j_end = p.ny - 1;
    int k_begin = p.c_grid != 0 ? 0 : 1;
    int k_end = p.c_grid != 0 ? p.nz : p.nz - 1;
    if (p.c_grid != 0 && p.location == 1)
    {
        i_end = p.periodic_x != 0 ? p.nx - 1 : p.nx - 2;
        k_end = p.nz - 1;
    }
    else if (p.c_grid != 0 && p.location == 2)
    {
        j_end = p.periodic_y != 0 ? p.ny - 1 : p.ny - 2;
        k_end = p.nz - 1;
    }
    else if (p.c_grid != 0 && p.location == 3)
    {
        k_end = p.nz - 1;
    }
    if (i < i_begin || i >= i_end || j < j_begin || j >= j_end ||
        k < k_begin || k >= k_end)
        return;

    float rho_c = component_density(rho, dz, i, j, k, p);
    float value = field[gid];
    float result = 0.0f;
    if (p.K_v > 0.0f || p.use_variable_K != 0)
    {
        bool bottom_closed = k == k_begin;
        bool top_closed = k == k_end - 1;
        bool w_control = p.c_grid != 0 && p.location == 3;
        float control_mass = w_control
            ? 0.5f * (scalar_density(rho, i, j, k, p) * dz[k] +
                      scalar_density(rho, i, j, k + 1, p) * dz[k + 1])
            : rho_c * dz[k];
        float bottom_flux = 0.0f;
        float top_flux = 0.0f;
        if (!bottom_closed)
        {
            float point_dz = w_control ? dz[k] : 0.5f * (dz[k - 1] + dz[k]);
            float transport = p.use_variable_K != 0
                ? (w_control
                    ? scalar_density(rho, i, j, k, p) *
                        cell_vertical_K(vertical_K, i, j, k, p)
                    : 0.5f * (
                        component_transport(rho, vertical_K, i, j, k - 1, p) +
                        component_transport(rho, vertical_K, i, j, k, p)))
                : (w_control
                    ? p.K_v * scalar_density(rho, i, j, k, p)
                    : 0.5f * p.K_v *
                        (component_density(rho, dz, i, j, k - 1, p) + rho_c));
            bottom_flux = transport * (value - field[index_of(i, j, k - 1, p)]) / point_dz;
        }
        if (!top_closed)
        {
            float point_dz = w_control ? dz[k + 1] : 0.5f * (dz[k] + dz[k + 1]);
            float transport = p.use_variable_K != 0
                ? (w_control
                    ? scalar_density(rho, i, j, k + 1, p) *
                        cell_vertical_K(vertical_K, i, j, k + 1, p)
                    : 0.5f * (
                        component_transport(rho, vertical_K, i, j, k, p) +
                        component_transport(rho, vertical_K, i, j, k + 1, p)))
                : (w_control
                    ? p.K_v * scalar_density(rho, i, j, k + 1, p)
                    : 0.5f * p.K_v *
                        (rho_c + component_density(rho, dz, i, j, k + 1, p)));
            top_flux = transport * (field[index_of(i, j, k + 1, p)] - value) / point_dz;
        }
        result += (top_flux - bottom_flux) / control_mass;
    }

    if (p.K_h > 0.0f)
    {
        bool west_closed = i == i_begin && p.periodic_x == 0;
        bool east_closed = i == i_end - 1 && p.periodic_x == 0;
        bool south_closed = j == j_begin && p.periodic_y == 0;
        bool north_closed = j == j_end - 1 && p.periodic_y == 0;
        int iw = i == i_begin && p.periodic_x != 0 ? i_end - 1 : i - 1;
        int ie = i == i_end - 1 && p.periodic_x != 0 ? i_begin : i + 1;
        int js = j == j_begin && p.periodic_y != 0 ? j_end - 1 : j - 1;
        int jn = j == j_end - 1 && p.periodic_y != 0 ? j_begin : j + 1;
        float x_div = 0.0f;
        float y_div = 0.0f;
        if (!west_closed)
        {
            float rho_w = 0.5f * (rho_c + component_density(rho, dz, iw, j, k, p));
            x_div -= rho_w * (value - field[index_of(iw, j, k, p)]) * p.inv_dx2;
        }
        if (!east_closed)
        {
            float rho_e = 0.5f * (rho_c + component_density(rho, dz, ie, j, k, p));
            x_div += rho_e * (field[index_of(ie, j, k, p)] - value) * p.inv_dx2;
        }
        if (!south_closed)
        {
            float rho_s = 0.5f * (rho_c + component_density(rho, dz, i, js, k, p));
            y_div -= rho_s * (value - field[index_of(i, js, k, p)]) * p.inv_dy2;
        }
        if (!north_closed)
        {
            float rho_n = 0.5f * (rho_c + component_density(rho, dz, i, jn, k, p));
            y_div += rho_n * (field[index_of(i, jn, k, p)] - value) * p.inv_dy2;
        }
        result += p.K_h * (x_div + y_div) / rho_c;
    }
    tendency[gid] = result;
}
)msl";

} // namespace nimbus::metal::shaders
