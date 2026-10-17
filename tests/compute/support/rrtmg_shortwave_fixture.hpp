#pragma once

#include "radiation/schemes/rrtmg/rrtmg_transfer.hpp"

#include <cstddef>
#include <vector>

struct RrtmgShortwaveFixture
{
    int                        layers = 0;
    std::vector<float>         absorption;
    std::vector<float>         rayleigh;
    std::vector<float>         solar_source;
    std::vector<float>         cloud_depth;
    std::vector<float>         cloud_omega;
    std::vector<float>         cloud_asymmetry;
    tmv::rrtmg::SpectralFluxes expected_clear;
    tmv::rrtmg::SpectralFluxes expected_cloudy;
    float                      cosine_zenith        = 0.63f;
    float                      solar_irradiance_wm2 = 1361.0f;
    float                      surface_albedo       = 0.18f;
};

inline RrtmgShortwaveFixture make_rrtmg_shortwave_fixture(int layers, int variant = 0)
{
    constexpr std::size_t g_points = 112;
    constexpr std::size_t bands    = 14;
    RrtmgShortwaveFixture fixture;
    fixture.layers               = layers;
    const float variant_scale    = 1.0f + 0.07f * static_cast<float>(variant);
    fixture.cosine_zenith        = variant == 2 ? 0.0f : 0.42f + 0.13f * static_cast<float>(variant);
    fixture.solar_irradiance_wm2 = 1315.0f + 18.0f * static_cast<float>(variant);
    fixture.surface_albedo       = 0.12f + 0.09f * static_cast<float>(variant);
    fixture.absorption.resize(static_cast<std::size_t>(layers) * g_points);
    fixture.rayleigh.resize(static_cast<std::size_t>(layers) * g_points);
    fixture.solar_source.resize(g_points);
    fixture.cloud_depth.resize(static_cast<std::size_t>(layers) * bands);
    fixture.cloud_omega.resize(static_cast<std::size_t>(layers) * bands);
    fixture.cloud_asymmetry.resize(static_cast<std::size_t>(layers) * bands);
    for (std::size_t g = 0; g < g_points; ++g)
    {
        fixture.solar_source[g] = 0.8f + 0.003f * static_cast<float>(g);
        for (int layer = 0; layer < layers; ++layer)
        {
            const std::size_t index   = static_cast<std::size_t>(layer) * g_points + g;
            fixture.absorption[index] = variant == 3 ? 120.0f + 0.5f * static_cast<float>(layer)
                                                     : variant_scale * (0.003f + 0.0007f * static_cast<float>(layer) +
                                                                        0.000015f * static_cast<float>(g));
            fixture.rayleigh[index]   = variant_scale * (0.002f + 0.00001f * static_cast<float>(g));
        }
    }
    for (int layer = 0; layer < layers; ++layer)
        for (std::size_t band = 0; band < bands; ++band)
        {
            const std::size_t index = static_cast<std::size_t>(layer) * bands + band;
            fixture.cloud_depth[index] =
                ((layer + static_cast<int>(band)) % 3 == 0) ? 0.035f + 0.002f * static_cast<float>(band) : 0.0f;
            fixture.cloud_omega[index]     = 0.86f;
            fixture.cloud_asymmetry[index] = 0.72f;
        }

    tmv::rrtmg::ShortwaveCoefficients coefficients;
    coefficients.grid.reduced_g_point_offsets = {0, 6, 18, 26, 34, 44, 54, 56, 66, 74, 80, 86, 94, 100, 112};
    tmv::rrtmg::ShortwaveOptics optics;
    optics.layers   = static_cast<std::size_t>(layers);
    optics.g_points = g_points;
    optics.absorption_optical_depth.assign(fixture.absorption.begin(), fixture.absorption.end());
    optics.rayleigh_optical_depth.assign(fixture.rayleigh.begin(), fixture.rayleigh.end());
    optics.solar_source_fraction.assign(fixture.solar_source.begin(), fixture.solar_source.end());
    tmv::rrtmg::ShortwaveCloudOptics clouds;
    clouds.layers = static_cast<std::size_t>(layers);
    clouds.optical_depth.assign(fixture.cloud_depth.begin(), fixture.cloud_depth.end());
    clouds.single_scattering_albedo.assign(fixture.cloud_omega.begin(), fixture.cloud_omega.end());
    clouds.asymmetry.assign(fixture.cloud_asymmetry.begin(), fixture.cloud_asymmetry.end());
    fixture.expected_clear = tmv::rrtmg::solve_shortwave(
        coefficients, optics, fixture.cosine_zenith, fixture.solar_irradiance_wm2, fixture.surface_albedo, nullptr);
    fixture.expected_cloudy = tmv::rrtmg::solve_shortwave(
        coefficients, optics, fixture.cosine_zenith, fixture.solar_irradiance_wm2, fixture.surface_albedo, &clouds);
    return fixture;
}
