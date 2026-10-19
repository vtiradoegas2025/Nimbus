#pragma once

namespace nimbus::metal::shaders
{
inline constexpr char kRrtmgLongwaveTransferMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct RrtmgLongwaveTransferPushConstants
{
    int layers;
    int has_clouds;
    float surface_emissivity;
    float precipitable_water_cm;
    int padding0; int padding1; int padding2; int padding3;
};

constant int band_ends[16] = {10, 22, 38, 52, 68, 76, 88, 96,
                              108, 114, 122, 130, 134, 136, 138, 140};
constant float secant_a0[16] = {1.66f, 1.55f, 1.58f, 1.66f, 1.54f, 1.454f, 1.89f, 1.33f,
                                1.668f, 1.66f, 1.66f, 1.66f, 1.66f, 1.66f, 1.66f, 1.66f};
constant float secant_a1[16] = {0.0f, 0.25f, 0.22f, 0.0f, 0.13f, 0.446f, -0.10f, 0.40f,
                                -0.006f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
constant float secant_a2[16] = {0.0f, -12.0f, -11.7f, 0.0f, -0.72f, -0.243f, 0.19f, -0.062f,
                                0.414f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

inline int band_for_gpoint(uint g)
{
    for (int band = 0; band < 16; ++band) if (g < uint(band_ends[band])) return band;
    return 15;
}

inline float diffusivity_secant(int band, float water)
{
    if (band == 0 || band == 3 || band >= 9) return 1.66f;
    return clamp(secant_a0[band] + secant_a1[band] * exp(secant_a2[band] * water), 1.50f, 1.80f);
}

inline float2 clear_transmission(float depth)
{
    if (depth <= 0.06f) return float2(depth - 0.5f * depth * depth, depth / 6.0f);
    const float transmission = exp(-depth);
    return float2(1.0f - transmission,
                  1.0f - 2.0f * (1.0f / depth - transmission / (1.0f - transmission)));
}

inline float cloudy_lookup(float depth)
{
    const float coordinate = depth / (1.0f / 0.278f + depth);
    const uint index = uint(10000.0f * coordinate + 0.5f);
    if (index >= 10000u) return 1.0e10f;
    const float fraction = float(index) / 10000.0f;
    return (1.0f / 0.278f) * fraction / (1.0f - fraction);
}

inline float2 layer_transmission(float gas_depth, float cloud_depth)
{
    if (cloud_depth > 0.0f)
    {
        if (gas_depth > 0.06f) gas_depth = cloudy_lookup(gas_depth);
        float total = gas_depth + cloud_depth;
        if (total > 0.06f) total = cloudy_lookup(total);
        return clear_transmission(total);
    }
    return clear_transmission(gas_depth);
}

kernel void rrtmg_longwave_transfer(
    device const float* absorption [[buffer(0)]],
    device const float* planck_fraction [[buffer(1)]],
    device const float* layer_planck [[buffer(2)]],
    device const float* interface_planck [[buffer(3)]],
    device const float* surface_planck [[buffer(4)]],
    device const float* band_width_cm [[buffer(5)]],
    device const float* cloud_absorption [[buffer(6)]],
    device float* upward_partial [[buffer(7)]],
    device float* downward_partial [[buffer(8)]],
    constant RrtmgLongwaveTransferPushConstants& pc [[buffer(9)]],
    uint g [[thread_position_in_grid]])
{
    if (g >= 140u) return;
    const int band = band_for_gpoint(g);
    const float secant = diffusivity_secant(band, pc.precipitable_water_cm);
    const uint stride = uint(pc.layers + 1);
    const uint base = g * stride;
    float down = 0.0f;
    downward_partial[base + uint(pc.layers)] = 0.0f;
    for (int layer = pc.layers - 1; layer >= 0; --layer)
    {
        const uint index = uint(layer) * 140u + g;
        const float cloud = pc.has_clouds != 0 ? cloud_absorption[uint(layer) * 16u + uint(band)] : 0.0f;
        const float2 transfer = layer_transmission(secant * absorption[index], secant * cloud);
        const float planck = layer_planck[uint(layer) * 16u + uint(band)];
        const float difference = interface_planck[uint(layer) * 16u + uint(band)] - planck;
        const float source = planck_fraction[index] * (planck + transfer.y * difference);
        down += (source - down) * transfer.x;
        downward_partial[base + uint(layer)] = down;
    }
    float up = surface_planck[uint(band)] * planck_fraction[g]
             + (1.0f - pc.surface_emissivity) * down;
    upward_partial[base] = up;
    for (int layer = 0; layer < pc.layers; ++layer)
    {
        const uint index = uint(layer) * 140u + g;
        const float cloud = pc.has_clouds != 0 ? cloud_absorption[uint(layer) * 16u + uint(band)] : 0.0f;
        const float2 transfer = layer_transmission(secant * absorption[index], secant * cloud);
        const float planck = layer_planck[uint(layer) * 16u + uint(band)];
        const float difference = interface_planck[uint(layer + 1) * 16u + uint(band)] - planck;
        const float source = planck_fraction[index] * (planck + transfer.y * difference);
        up += (source - up) * transfer.x;
        upward_partial[base + uint(layer + 1)] = up;
    }
    const float scale = 3.14159265358979323846e4f * band_width_cm[uint(band)];
    for (int interface_index = 0; interface_index <= pc.layers; ++interface_index)
    {
        const uint index = base + uint(interface_index);
        upward_partial[index] *= scale;
        downward_partial[index] *= scale;
    }
}
)msl";
} // namespace nimbus::metal::shaders
