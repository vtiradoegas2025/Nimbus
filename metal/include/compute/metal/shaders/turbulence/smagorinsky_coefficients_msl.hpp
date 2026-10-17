#pragma once

namespace nimbus::metal::shaders
{

// Runtime-compiled MSL translation of
// vulkan/shaders/compute/turbulence/smagorinsky_coefficients.comp.
// The constants structure is the Metal spelling of the canonical 60-byte
// SmagorinskyCoefficientsPushConstants ABI.
inline constexpr char kSmagorinskyCoefficientsMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct SmagorinskyCoefficientsPushConstants
{
    int nx;
    int ny;
    int nz;
    int filter_mode;
    int use_horizontal;
    int use_vertical;
    float dx;
    float dy;
    float Cs;
    float Pr_t;
    float Sc_t;
    float nu_t_max;
    float K_max;
    int periodic_y;
    int periodic_x;
};

inline bool finite_f(float value)
{
    return !isnan(value) && !isinf(value);
}

inline uint field_index(int i, int j, int k,
                        constant SmagorinskyCoefficientsPushConstants& pc)
{
    return uint((i * pc.ny + j) * pc.nz + k);
}

inline int periodic_prev(int index, int count)
{
    return index <= 1 ? count - 2 : min(index - 1, count - 2);
}

inline int periodic_next(int index, int count)
{
    return index >= count - 2 ? 1 : max(index + 1, 1);
}

inline float center_distance(device const float* dz_profile, int lower_k)
{
    return 0.5f * (dz_profile[lower_k] + dz_profile[lower_k + 1]);
}

kernel void smagorinsky_coefficients(
    device const float* u_in [[buffer(0)]],
    device const float* v_in [[buffer(1)]],
    device const float* w_in [[buffer(2)]],
    device const float* dz_profile [[buffer(3)]],
    device const float* dzc_profile [[buffer(4)]],
    device float* nu_t [[buffer(5)]],
    device float* K_theta [[buffer(6)]],
    device float* K_q [[buffer(7)]],
    constant SmagorinskyCoefficientsPushConstants& pc [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = uint(pc.nx) * uint(pc.ny) * uint(pc.nz);
    if (gid >= total)
        return;

    const int k = int(gid % uint(pc.nz));
    const uint ij = gid / uint(pc.nz);
    const int j = int(ij % uint(pc.ny));
    const int i = int(ij / uint(pc.ny));
    const uint center = gid;

    const bool wrap_x = pc.periodic_x != 0;
    const bool wrap_y = pc.periodic_y != 0;
    const int jm = wrap_y ? periodic_prev(j, pc.ny) : max(j - 1, 0);
    const int im = wrap_x ? periodic_prev(i, pc.nx) : max(i - 1, 0);

    const float w_center = w_in[center];
    const float w_down = k > 0
        ? w_in[field_index(i, j, k - 1, pc)] : 0.0f;
    const float S11 = (u_in[center] - u_in[field_index(im, j, k, pc)]) / pc.dx;
    const float S22 = (v_in[center] - v_in[field_index(i, jm, k, pc)]) / pc.dy;
    const float S33 = (w_center - w_down) / dz_profile[k];

    float s12[4];
    {
        int ie[2] = {im, i};
        int je[2] = {jm, j};
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                const int jj1 = wrap_y
                    ? periodic_next(je[b], pc.ny)
                    : min(je[b] + 1, pc.ny - 1);
                const int ii1 = wrap_x
                    ? periodic_next(ie[a], pc.nx)
                    : min(ie[a] + 1, pc.nx - 1);
                const float dudy =
                    (u_in[field_index(ie[a], jj1, k, pc)]
                     - u_in[field_index(ie[a], je[b], k, pc)]) / pc.dy;
                const float dvdx =
                    (v_in[field_index(ii1, je[b], k, pc)]
                     - v_in[field_index(ie[a], je[b], k, pc)]) / pc.dx;
                s12[a * 2 + b] = 0.5f * (dudy + dvdx);
            }
        }
    }

    float2 xz[4];
    {
        int ie[2] = {im, i};
        int ke[2] = {k - 1, k};
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                if (ke[b] < 0 || ke[b] >= pc.nz - 1)
                {
                    xz[a * 2 + b] = float2(0.0f);
                    continue;
                }
                const int ii1 = wrap_x
                    ? periodic_next(ie[a], pc.nx)
                    : min(ie[a] + 1, pc.nx - 1);
                const float dudz =
                    (u_in[field_index(ie[a], j, ke[b] + 1, pc)]
                     - u_in[field_index(ie[a], j, ke[b], pc)])
                    / center_distance(dz_profile, ke[b]);
                const float dwdx =
                    (w_in[field_index(ii1, j, ke[b], pc)]
                     - w_in[field_index(ie[a], j, ke[b], pc)]) / pc.dx;
                xz[a * 2 + b] = float2(dudz, dwdx);
            }
        }
    }

    float2 yz[4];
    {
        int je[2] = {jm, j};
        int ke[2] = {k - 1, k};
        for (int a = 0; a < 2; ++a)
        {
            for (int b = 0; b < 2; ++b)
            {
                if (ke[b] < 0 || ke[b] >= pc.nz - 1)
                {
                    yz[a * 2 + b] = float2(0.0f);
                    continue;
                }
                const int jj1 = wrap_y
                    ? periodic_next(je[a], pc.ny)
                    : min(je[a] + 1, pc.ny - 1);
                const float dvdz =
                    (v_in[field_index(i, je[a], ke[b] + 1, pc)]
                     - v_in[field_index(i, je[a], ke[b], pc)])
                    / center_distance(dz_profile, ke[b]);
                const float dwdy =
                    (w_in[field_index(i, jj1, ke[b], pc)]
                     - w_in[field_index(i, je[a], ke[b], pc)]) / pc.dy;
                yz[a * 2 + b] = float2(dvdz, dwdy);
            }
        }
    }

    const float S12_sq = 0.25f * (
        s12[0] * s12[0] + s12[1] * s12[1]
        + s12[2] * s12[2] + s12[3] * s12[3]);
    const float dudz_sq = 0.25f * (
        xz[0].x * xz[0].x + xz[1].x * xz[1].x
        + xz[2].x * xz[2].x + xz[3].x * xz[3].x);
    const float dwdx_sq = 0.25f * (
        xz[0].y * xz[0].y + xz[1].y * xz[1].y
        + xz[2].y * xz[2].y + xz[3].y * xz[3].y);
    const float dvdz_sq = 0.25f * (
        yz[0].x * yz[0].x + yz[1].x * yz[1].x
        + yz[2].x * yz[2].x + yz[3].x * yz[3].x);
    const float dwdy_sq = 0.25f * (
        yz[0].y * yz[0].y + yz[1].y * yz[1].y
        + yz[2].y * yz[2].y + yz[3].y * yz[3].y);

    float strain_sq = 0.0f;
    if (pc.use_horizontal != 0 && pc.use_vertical == 0)
    {
        strain_sq = 2.0f * (S11 * S11 + S22 * S22)
            + 4.0f * S12_sq + dwdx_sq + dwdy_sq;
    }
    else if (pc.use_horizontal == 0 && pc.use_vertical != 0)
    {
        strain_sq = dudz_sq + dvdz_sq + 2.0f * S33 * S33;
    }
    else
    {
        const float S13_sq = 0.25f * (
            0.25f * (xz[0].x + xz[0].y) * (xz[0].x + xz[0].y)
            + 0.25f * (xz[1].x + xz[1].y) * (xz[1].x + xz[1].y)
            + 0.25f * (xz[2].x + xz[2].y) * (xz[2].x + xz[2].y)
            + 0.25f * (xz[3].x + xz[3].y) * (xz[3].x + xz[3].y));
        const float S23_sq = 0.25f * (
            0.25f * (yz[0].x + yz[0].y) * (yz[0].x + yz[0].y)
            + 0.25f * (yz[1].x + yz[1].y) * (yz[1].x + yz[1].y)
            + 0.25f * (yz[2].x + yz[2].y) * (yz[2].x + yz[2].y)
            + 0.25f * (yz[3].x + yz[3].y) * (yz[3].x + yz[3].y));
        strain_sq = 2.0f * (S11 * S11 + S22 * S22 + S33 * S33)
            + 4.0f * (S12_sq + S13_sq + S23_sq);
    }

    const float strain_magnitude = finite_f(strain_sq) && strain_sq > 0.0f
        ? sqrt(strain_sq) : 0.0f;
    float filter_width = pc.filter_mode == 1
        ? sqrt(pc.dx * pc.dy)
        : pow(pc.dx * pc.dy * dz_profile[k], 1.0f / 3.0f);
    if (!finite_f(filter_width) || filter_width <= 0.0f)
        filter_width = 1.0f;

    float viscosity = pc.Cs * pc.Cs * filter_width * filter_width
        * strain_magnitude;
    viscosity = max(viscosity, 0.0f);
    if (!finite_f(viscosity))
        viscosity = 0.0f;
    viscosity = min(viscosity, pc.nu_t_max);
    nu_t[center] = viscosity;

    float theta_diffusivity = viscosity / pc.Pr_t;
    float moisture_diffusivity = viscosity / pc.Sc_t;
    if (!finite_f(theta_diffusivity) || theta_diffusivity < 0.0f)
        theta_diffusivity = 0.0f;
    if (!finite_f(moisture_diffusivity) || moisture_diffusivity < 0.0f)
        moisture_diffusivity = 0.0f;
    K_theta[center] = min(theta_diffusivity, pc.K_max);
    K_q[center] = min(moisture_diffusivity, pc.K_max);

    (void)dzc_profile;
}
)msl";

} // namespace nimbus::metal::shaders
