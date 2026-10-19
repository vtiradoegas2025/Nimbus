#pragma once

#include "compute/support/rrtmg_longwave_fixture.hpp"
#include "compute/support/rrtmg_shortwave_fixture.hpp"

#include <cstddef>
#include <utility>
#include <vector>

struct RrtmgLongwaveBatchFixture
{
    int columns = 0;
    int layers = 0;
    std::vector<RrtmgLongwaveFixture> columns_data;
    std::vector<float> absorption;
    std::vector<float> planck_fraction;
    std::vector<float> layer_planck;
    std::vector<float> interface_planck;
    std::vector<float> surface_planck;
    std::vector<float> cloud_absorption;
    std::vector<float> emissivity;
    std::vector<float> precipitable_water_cm;
    std::vector<float> band_width;
};

struct RrtmgShortwaveBatchFixture
{
    int columns = 0;
    int layers = 0;
    std::vector<RrtmgShortwaveFixture> columns_data;
    std::vector<float> absorption;
    std::vector<float> rayleigh;
    std::vector<float> solar_source;
    std::vector<float> cloud_depth;
    std::vector<float> cloud_omega;
    std::vector<float> cloud_asymmetry;
    std::vector<float> cosine_zenith;
    std::vector<float> solar_irradiance_wm2;
    std::vector<float> surface_albedo;
};

namespace rrtmg_batch_fixture_detail
{
template <typename T>
void append(std::vector<T>& destination, const std::vector<T>& source)
{
    destination.insert(destination.end(), source.begin(), source.end());
}
} // namespace rrtmg_batch_fixture_detail

inline RrtmgLongwaveBatchFixture make_rrtmg_longwave_batch_fixture(
    int columns, int layers)
{
    RrtmgLongwaveBatchFixture batch;
    batch.columns = columns;
    batch.layers = layers;
    batch.columns_data.reserve(static_cast<std::size_t>(columns));
    batch.emissivity.reserve(static_cast<std::size_t>(columns));
    batch.precipitable_water_cm.reserve(static_cast<std::size_t>(columns));
    for (int column = 0; column < columns; ++column)
    {
        auto fixture = make_rrtmg_longwave_fixture(layers, column);
        if (column == 0) batch.band_width = fixture.band_width;
        rrtmg_batch_fixture_detail::append(batch.absorption, fixture.absorption);
        rrtmg_batch_fixture_detail::append(batch.planck_fraction, fixture.planck_fraction);
        rrtmg_batch_fixture_detail::append(batch.layer_planck, fixture.layer_planck);
        rrtmg_batch_fixture_detail::append(batch.interface_planck, fixture.interface_planck);
        rrtmg_batch_fixture_detail::append(batch.surface_planck, fixture.surface_planck);
        rrtmg_batch_fixture_detail::append(batch.cloud_absorption, fixture.cloud_absorption);
        batch.emissivity.push_back(fixture.emissivity);
        batch.precipitable_water_cm.push_back(fixture.precipitable_water_cm);
        batch.columns_data.push_back(std::move(fixture));
    }
    return batch;
}

inline RrtmgShortwaveBatchFixture make_rrtmg_shortwave_batch_fixture(
    int columns, int layers)
{
    RrtmgShortwaveBatchFixture batch;
    batch.columns = columns;
    batch.layers = layers;
    batch.columns_data.reserve(static_cast<std::size_t>(columns));
    batch.cosine_zenith.reserve(static_cast<std::size_t>(columns));
    batch.solar_irradiance_wm2.reserve(static_cast<std::size_t>(columns));
    batch.surface_albedo.reserve(static_cast<std::size_t>(columns));
    for (int column = 0; column < columns; ++column)
    {
        auto fixture = make_rrtmg_shortwave_fixture(layers, column);
        rrtmg_batch_fixture_detail::append(batch.absorption, fixture.absorption);
        rrtmg_batch_fixture_detail::append(batch.rayleigh, fixture.rayleigh);
        rrtmg_batch_fixture_detail::append(batch.solar_source, fixture.solar_source);
        rrtmg_batch_fixture_detail::append(batch.cloud_depth, fixture.cloud_depth);
        rrtmg_batch_fixture_detail::append(batch.cloud_omega, fixture.cloud_omega);
        rrtmg_batch_fixture_detail::append(batch.cloud_asymmetry, fixture.cloud_asymmetry);
        batch.cosine_zenith.push_back(fixture.cosine_zenith);
        batch.solar_irradiance_wm2.push_back(fixture.solar_irradiance_wm2);
        batch.surface_albedo.push_back(fixture.surface_albedo);
        batch.columns_data.push_back(std::move(fixture));
    }
    return batch;
}
