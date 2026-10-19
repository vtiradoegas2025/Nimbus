#pragma once

#include "radiation/schemes/rrtmg/rrtmg_transfer.hpp"

#include <cstddef>
#include <vector>

struct RrtmgLongwaveFixture
{
    int layers = 0;
    std::vector<float> absorption;
    std::vector<float> planck_fraction;
    std::vector<float> layer_planck;
    std::vector<float> interface_planck;
    std::vector<float> surface_planck;
    std::vector<float> band_width;
    std::vector<float> cloud_absorption;
    tmv::rrtmg::SpectralFluxes expected_clear;
    tmv::rrtmg::SpectralFluxes expected_cloudy;
    float emissivity = 0.92f;
    float precipitable_water_cm = 1.7f;
};

inline RrtmgLongwaveFixture make_rrtmg_longwave_fixture(int layers, int variant = 0)
{
    constexpr std::size_t g_points = 140;
    constexpr std::size_t bands = 16;
    RrtmgLongwaveFixture fixture;
    fixture.layers = layers;
    const float variant_scale = 1.0f + 0.08f * static_cast<float>(variant);
    fixture.emissivity = 0.88f + 0.025f * static_cast<float>(variant);
    fixture.precipitable_water_cm = 1.2f + 0.45f * static_cast<float>(variant);
    fixture.absorption.resize(static_cast<std::size_t>(layers) * g_points);
    fixture.planck_fraction.resize(static_cast<std::size_t>(layers) * g_points);
    fixture.layer_planck.resize(static_cast<std::size_t>(layers) * bands);
    fixture.interface_planck.resize(static_cast<std::size_t>(layers + 1) * bands);
    fixture.surface_planck.resize(bands);
    fixture.band_width.resize(bands);
    fixture.cloud_absorption.resize(static_cast<std::size_t>(layers) * bands);
    const std::vector<std::size_t> offsets{
        0, 10, 22, 38, 52, 68, 76, 88, 96, 108, 114, 122, 130, 134, 136, 138, 140};
    for (std::size_t band = 0; band < bands; ++band)
    {
        fixture.band_width[band] = 45.0f + static_cast<float>(band);
        fixture.surface_planck[band] = variant_scale *
            (8.0e-6f + static_cast<float>(band) * 1.0e-7f);
        for (int layer = 0; layer < layers; ++layer)
        {
            fixture.layer_planck[static_cast<std::size_t>(layer) * bands + band] =
                variant_scale * (5.0e-6f + static_cast<float>(layer) * 2.0e-7f +
                                 static_cast<float>(band) * 5.0e-8f);
            fixture.cloud_absorption[static_cast<std::size_t>(layer) * bands + band] =
                ((layer + static_cast<int>(band)) % 3 == 0) ? 0.025f : 0.0f;
        }
        for (int interface_index = 0; interface_index <= layers; ++interface_index)
            fixture.interface_planck[static_cast<std::size_t>(interface_index) * bands + band] =
                variant_scale * (4.8e-6f + static_cast<float>(interface_index) * 2.1e-7f +
                                 static_cast<float>(band) * 5.0e-8f);
        const float fraction = 1.0f / static_cast<float>(offsets[band + 1] - offsets[band]);
        for (std::size_t g = offsets[band]; g < offsets[band + 1]; ++g)
            for (int layer = 0; layer < layers; ++layer)
            {
                const std::size_t index = static_cast<std::size_t>(layer) * g_points + g;
                fixture.planck_fraction[index] = fraction;
                fixture.absorption[index] = variant_scale *
                    (0.004f + 0.001f * static_cast<float>(layer) +
                     0.00003f * static_cast<float>(g));
            }
    }

    tmv::rrtmg::LongwaveCoefficients coefficients;
    coefficients.grid.reduced_g_point_offsets = offsets;
    coefficients.grid.band_lower_wavenumber_cm.resize(bands);
    coefficients.grid.band_upper_wavenumber_cm.resize(bands);
    for (std::size_t band = 0; band < bands; ++band)
    {
        coefficients.grid.band_lower_wavenumber_cm[band] = 100.0 * static_cast<double>(band);
        coefficients.grid.band_upper_wavenumber_cm[band] =
            coefficients.grid.band_lower_wavenumber_cm[band] + fixture.band_width[band];
    }
    tmv::rrtmg::PreparedAtmosphere atmosphere;
    atmosphere.layer_pressure_pa.assign(static_cast<std::size_t>(layers), 70000.0);
    tmv::rrtmg::LongwavePreparation preparation;
    preparation.precipitable_water_cm = fixture.precipitable_water_cm;
    preparation.layer_planck.assign(fixture.layer_planck.begin(), fixture.layer_planck.end());
    preparation.interface_planck.assign(fixture.interface_planck.begin(), fixture.interface_planck.end());
    preparation.surface_planck.assign(fixture.surface_planck.begin(), fixture.surface_planck.end());
    tmv::rrtmg::LongwaveOptics optics;
    optics.layers = static_cast<std::size_t>(layers);
    optics.g_points = g_points;
    optics.absorption_optical_depth.assign(fixture.absorption.begin(), fixture.absorption.end());
    optics.planck_fraction.assign(fixture.planck_fraction.begin(), fixture.planck_fraction.end());
    tmv::rrtmg::LongwaveCloudOptics clouds;
    clouds.layers = static_cast<std::size_t>(layers);
    clouds.absorption_optical_depth.assign(
        fixture.cloud_absorption.begin(), fixture.cloud_absorption.end());
    fixture.expected_clear = tmv::rrtmg::solve_longwave(
        coefficients, atmosphere, preparation, optics, fixture.emissivity, nullptr);
    fixture.expected_cloudy = tmv::rrtmg::solve_longwave(
        coefficients, atmosphere, preparation, optics, fixture.emissivity, &clouds);
    return fixture;
}
