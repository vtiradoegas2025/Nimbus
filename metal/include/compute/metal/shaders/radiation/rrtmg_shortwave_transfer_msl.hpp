#pragma once

namespace nimbus::metal::shaders
{
inline constexpr char kRrtmgShortwaveTransferMsl[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct RrtmgShortwaveTransferPushConstants
{
    int layers;
    int has_clouds;
    float cosine_zenith;
    float solar_irradiance_wm2;
    float surface_albedo;
    float source_sum;
    int padding0;
    int padding1;
};

struct LayerTransfer
{
    float direct_reflectance;
    float diffuse_reflectance;
    float direct_transmittance;
    float diffuse_transmittance;
    float direct_beam;
};

constant int band_ends[14] = {6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100, 112};

inline float transmission(float optical_depth)
{
    const float bounded = min(optical_depth, 40.0f);
    return bounded <= 1.0e-4f
               ? 1.0f - bounded + 0.5f * bounded * bounded
               : exp(-bounded);
}

inline LayerTransfer clear_transfer()
{
    LayerTransfer result;
    result.direct_reflectance = 0.0f;
    result.diffuse_reflectance = 0.0f;
    result.direct_transmittance = 1.0f;
    result.diffuse_transmittance = 1.0f;
    result.direct_beam = 1.0f;
    return result;
}

inline LayerTransfer pifm_two_stream(float absorption,
                                     float scattering,
                                     float asymmetry,
                                     float cosine_zenith)
{
    LayerTransfer result = clear_transfer();
    const float optical_depth = absorption + scattering;
    if (optical_depth <= 0.0f) return result;

    const float omega = clamp(scattering / optical_depth, 0.0f, 1.0f);
    const float gamma1 = (8.0f - omega * (5.0f + 3.0f * asymmetry)) * 0.25f;
    const float gamma2 = 3.0f * omega * (1.0f - asymmetry) * 0.25f;
    const float gamma3 = (2.0f - 3.0f * asymmetry * cosine_zenith) * 0.25f;
    const float gamma4 = 1.0f - gamma3;
    result.direct_beam = transmission(optical_depth / cosine_zenith);
    if (omega >= 0.9999995f)
    {
        const float gamma_tau = gamma1 * optical_depth;
        result.direct_reflectance =
            (gamma_tau - (gamma1 * cosine_zenith - gamma3) * (1.0f - result.direct_beam)) /
            (1.0f + gamma_tau);
        result.direct_transmittance = 1.0f - result.direct_reflectance;
        result.diffuse_reflectance = gamma_tau / (1.0f + gamma_tau);
        result.diffuse_transmittance = 1.0f - result.diffuse_reflectance;
        return result.direct_beam == 1.0f ? clear_transfer() : result;
    }

    const float a1 = gamma1 * gamma4 + gamma2 * gamma3;
    const float a2 = gamma1 * gamma3 + gamma2 * gamma4;
    const float k = sqrt(max(0.0f, gamma1 * gamma1 - gamma2 * gamma2));
    const float kp = k * cosine_zenith;
    const float kp1 = 1.0f + kp;
    const float km1 = 1.0f - kp;
    const float two_k = 2.0f * k;
    const float one_kp2 = 1.0f - kp * kp;
    const float kg = k + gamma1;
    const float r1 = km1 * (a2 + k * gamma3);
    const float r2 = kp1 * (a2 - k * gamma3);
    const float r3 = two_k * (gamma3 - a2 * cosine_zenith);
    const float r4 = one_kp2 * kg;
    const float r5 = one_kp2 * (k - gamma1);
    const float t1 = kp1 * (a1 + k * gamma4);
    const float t2 = km1 * (a1 - k * gamma4);
    const float t3 = two_k * (gamma4 + a1 * cosine_zenith);
    const float beta = (gamma1 - k) / kg;
    const float exp_k = transmission(k * optical_depth);
    const float inv_exp_k = 1.0f / exp_k;
    const float denominator = r4 * inv_exp_k + r5 * exp_k;
    if (fabs(denominator) <= 1.0e-8f)
    {
        result.direct_reflectance = 1.0e-8f;
        result.direct_transmittance = result.direct_beam;
    }
    else
    {
        result.direct_reflectance =
            omega * (r1 * inv_exp_k - r2 * exp_k - r3 * result.direct_beam) / denominator;
        result.direct_transmittance =
            result.direct_beam - omega *
                (result.direct_beam * (t1 * inv_exp_k - t2 * exp_k) - t3) / denominator;
    }
    const float exp_k_squared = exp_k * exp_k;
    const float diffuse_denominator = 1.0f / ((1.0f - beta * exp_k_squared) * kg);
    result.diffuse_reflectance = gamma2 * (1.0f - exp_k_squared) * diffuse_denominator;
    result.diffuse_transmittance = two_k * exp_k * diffuse_denominator;
    return result;
}

inline int band_for_gpoint(uint g_point)
{
    for (int band = 0; band < 14; ++band)
        if (g_point < uint(band_ends[band])) return band;
    return 13;
}

kernel void rrtmg_shortwave_transfer(
    device const float* absorption [[buffer(0)]],
    device const float* rayleigh [[buffer(1)]],
    device const float* solar_source [[buffer(2)]],
    device const float* cloud_depth [[buffer(3)]],
    device const float* cloud_omega [[buffer(4)]],
    device const float* cloud_asymmetry [[buffer(5)]],
    device float* upward_partial [[buffer(6)]],
    device float* downward_partial [[buffer(7)]],
    device float* direct_reflectance [[buffer(8)]],
    device float* diffuse_reflectance [[buffer(9)]],
    device float* direct_transmittance [[buffer(10)]],
    device float* diffuse_transmittance [[buffer(11)]],
    device float* direct_beam [[buffer(12)]],
    device float* cumulative_direct [[buffer(13)]],
    device float* upward_direct [[buffer(14)]],
    device float* upward_diffuse [[buffer(15)]],
    device float* downward_total [[buffer(16)]],
    device float* downward_diffuse [[buffer(17)]],
    constant RrtmgShortwaveTransferPushConstants& pc [[buffer(18)]],
    uint g_point [[thread_position_in_grid]])
{
    if (g_point >= 112u) return;
    const uint stride = uint(pc.layers + 1);
    const uint base = g_point * stride;
    const int band = band_for_gpoint(g_point);
    cumulative_direct[base] = 1.0f;
    for (int top = 0; top < pc.layers; ++top)
    {
        const int layer = pc.layers - 1 - top;
        const uint input_index = uint(layer) * 112u + g_point;
        const uint spectral_index = base + uint(top);
        const uint cloud_index = uint(layer) * 14u + uint(band);
        const float depth = pc.has_clouds != 0 ? cloud_depth[cloud_index] : 0.0f;
        const float cloud_scattering =
            pc.has_clouds != 0 ? depth * cloud_omega[cloud_index] : 0.0f;
        const float scattering = rayleigh[input_index] + cloud_scattering;
        const float asymmetry = scattering > 0.0f && pc.has_clouds != 0
                                    ? cloud_scattering * cloud_asymmetry[cloud_index] / scattering
                                    : 0.0f;
        const LayerTransfer transfer = pifm_two_stream(
            absorption[input_index] + depth - cloud_scattering,
            scattering,
            asymmetry,
            pc.cosine_zenith);
        direct_reflectance[spectral_index] = transfer.direct_reflectance;
        diffuse_reflectance[spectral_index] = transfer.diffuse_reflectance;
        direct_transmittance[spectral_index] = transfer.direct_transmittance;
        diffuse_transmittance[spectral_index] = transfer.diffuse_transmittance;
        direct_beam[spectral_index] = transfer.direct_beam;
        cumulative_direct[base + uint(top + 1)] =
            cumulative_direct[base + uint(top)] * transfer.direct_beam;
    }

    upward_direct[base + uint(pc.layers)] = pc.surface_albedo;
    upward_diffuse[base + uint(pc.layers)] = pc.surface_albedo;
    const int lowest = pc.layers - 1;
    const uint lowest_index = base + uint(lowest);
    float coupling = 1.0f /
        (1.0f - pc.surface_albedo * diffuse_reflectance[lowest_index]);
    upward_direct[lowest_index] = direct_reflectance[lowest_index] +
        diffuse_transmittance[lowest_index] *
        ((direct_transmittance[lowest_index] - direct_beam[lowest_index]) * pc.surface_albedo +
         direct_beam[lowest_index] * pc.surface_albedo) * coupling;
    upward_diffuse[lowest_index] = diffuse_reflectance[lowest_index] +
        diffuse_transmittance[lowest_index] * diffuse_transmittance[lowest_index] *
        pc.surface_albedo * coupling;
    for (int top = lowest - 1; top >= 0; --top)
    {
        const uint index = base + uint(top);
        coupling = 1.0f /
            (1.0f - upward_diffuse[index + 1u] * diffuse_reflectance[index]);
        upward_direct[index] = direct_reflectance[index] + diffuse_transmittance[index] *
            ((direct_transmittance[index] - direct_beam[index]) * upward_diffuse[index + 1u] +
             direct_beam[index] * upward_direct[index + 1u]) * coupling;
        upward_diffuse[index] = diffuse_reflectance[index] +
            diffuse_transmittance[index] * diffuse_transmittance[index] *
            upward_diffuse[index + 1u] * coupling;
    }

    downward_total[base] = 1.0f;
    downward_diffuse[base] = 0.0f;
    downward_total[base + 1u] = direct_transmittance[base];
    downward_diffuse[base + 1u] = diffuse_reflectance[base];
    for (int top = 1; top < pc.layers; ++top)
    {
        const uint index = base + uint(top);
        coupling = 1.0f /
            (1.0f - diffuse_reflectance[index] * downward_diffuse[index]);
        downward_total[index + 1u] = cumulative_direct[index] * direct_transmittance[index] +
            diffuse_transmittance[index] *
            ((downward_total[index] - cumulative_direct[index]) +
             cumulative_direct[index] * direct_reflectance[index] * downward_diffuse[index]) * coupling;
        downward_diffuse[index + 1u] = diffuse_reflectance[index] +
            diffuse_transmittance[index] * diffuse_transmittance[index] *
            downward_diffuse[index] * coupling;
    }

    const float incident = pc.solar_irradiance_wm2 * pc.cosine_zenith *
                           solar_source[g_point] / pc.source_sum;
    for (int top = 0; top <= pc.layers; ++top)
    {
        const uint index = base + uint(top);
        coupling = 1.0f /
            (1.0f - downward_diffuse[index] * upward_diffuse[index]);
        const float upward =
            (cumulative_direct[index] * upward_direct[index] +
             (downward_total[index] - cumulative_direct[index]) * upward_diffuse[index]) * coupling;
        const float downward = cumulative_direct[index] +
            ((downward_total[index] - cumulative_direct[index]) +
             cumulative_direct[index] * upward_direct[index] * downward_diffuse[index]) * coupling;
        const uint output_index = base + uint(pc.layers - top);
        upward_partial[output_index] = incident * upward;
        downward_partial[output_index] = incident * downward;
    }
}
)msl";
} // namespace nimbus::metal::shaders
