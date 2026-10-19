#include "rrtmg.hpp"

#include "compute/dispatch/radiation.hpp"
#include "radiation/base/radiative_transfer.hpp"
#include "rrtmg_atmosphere.hpp"
#include "rrtmg_cloud_optics.hpp"
#include "rrtmg_coefficients.hpp"
#include "rrtmg_optics.hpp"
#include "rrtmg_preprocessing.hpp"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tmv
{
namespace
{

struct PreparedRrtmgBatchColumn
{
    rrtmg::PreparedAtmosphere   atmosphere;
    rrtmg::LongwavePreparation  longwave_preparation;
    rrtmg::LongwaveOptics       longwave_optics;
    rrtmg::LongwaveCloudOptics  longwave_clouds;
    rrtmg::ShortwaveOptics      shortwave_optics;
    rrtmg::ShortwaveCloudOptics shortwave_clouds;
};

void append_float(std::vector<float>& destination, const std::vector<double>& source)
{
    const std::size_t offset = destination.size();
    destination.resize(offset + source.size());
    std::transform(source.begin(),
                   source.end(),
                   destination.begin() + offset,
                   [](double value) { return static_cast<float>(value); });
}

void rethrow_first(const std::vector<std::exception_ptr>& failures)
{
    const auto failure = std::find_if(
        failures.begin(), failures.end(), [](const std::exception_ptr& value) { return value != nullptr; });
    if (failure != failures.end())
        std::rethrow_exception(*failure);
}

} // namespace

bool RrtmgScheme::supports_native_column_batch(const RadiationConfig&         config,
                                               const RadiationComputeKernels& kernels) const
{
    if (!initialized_ || coefficients_ == nullptr || coefficients_->longwave == nullptr)
        return false;
    if (config.do_lw && !kernels.supports(RadiationKernelOp::RrtmgLongwaveTransferBatch))
        return false;
    if (config.do_sw &&
        (coefficients_->shortwave == nullptr || !kernels.supports(RadiationKernelOp::RrtmgShortwaveTransferBatch)))
        return false;
    return config.do_lw || config.do_sw;
}

bool RrtmgScheme::compute_columns_native(const RadiationConfig&          config,
                                         const RadiationColumnStateView* columns,
                                         std::size_t                     column_count,
                                         RadiationColumnTendencies*      tendencies,
                                         RadiationComputeKernels&        kernels)
{
    return compute_columns_native_diagnosed(config, columns, column_count, tendencies, kernels, nullptr);
}

bool RrtmgScheme::compute_columns_native_diagnosed(const RadiationConfig&          config,
                                                   const RadiationColumnStateView* columns,
                                                   std::size_t                     column_count,
                                                   RadiationColumnTendencies*      tendencies,
                                                   RadiationComputeKernels&        kernels,
                                                   RadiationColumnFluxes*          fluxes)
{
    validate_compute_config(config);
    if (columns == nullptr || tendencies == nullptr || column_count == 0 ||
        column_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        !supports_native_column_batch(config, kernels))
        return false;

    const auto& first_pressure = columns[0].p;
    if (first_pressure == nullptr || first_pressure->empty() ||
        first_pressure->size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return false;
    const std::size_t layers = first_pressure->size();
    for (std::size_t column = 0; column < column_count; ++column)
    {
        if (columns[column].p == nullptr || columns[column].rho == nullptr || columns[column].dz == nullptr ||
            columns[column].p->size() != layers || columns[column].rho->size() != layers ||
            columns[column].dz->size() != layers)
            return false;
    }

    try
    {
        std::vector<PreparedRrtmgBatchColumn> prepared(column_count);
        std::vector<std::exception_ptr>       preparation_failures(column_count);

#pragma omp parallel for if (column_count >= 8) schedule(static)
        for (std::size_t column = 0; column < column_count; ++column)
        {
            try
            {
                auto& item      = prepared[column];
                item.atmosphere = rrtmg::prepare_atmosphere(config, columns[column]);
                item.longwave_preparation =
                    rrtmg::prepare_longwave(*coefficients_->longwave, item.atmosphere, columns[column].emissivity_lw);
                if (config.do_lw)
                {
                    item.longwave_optics = rrtmg::compute_longwave_gas_optics(
                        *coefficients_->longwave, item.atmosphere, item.longwave_preparation);
                    if (!config.clear_sky_only)
                        item.longwave_clouds = rrtmg::compute_longwave_cloud_optics(item.atmosphere);
                }
                if (config.do_sw)
                {
                    item.shortwave_optics = rrtmg::compute_shortwave_gas_optics(
                        *coefficients_->shortwave, item.atmosphere, item.longwave_preparation.interpolation);
                    if (!config.clear_sky_only)
                        item.shortwave_clouds =
                            rrtmg::compute_shortwave_cloud_optics(*coefficients_->shortwave, item.atmosphere);
                }
            }
            catch (...)
            {
                preparation_failures[column] = std::current_exception();
            }
        }
        rethrow_first(preparation_failures);

        const std::size_t  interface_count = column_count * (layers + 1);
        std::vector<float> longwave_up(interface_count, 0.0f);
        std::vector<float> longwave_down(interface_count, 0.0f);
        std::vector<float> shortwave_up(interface_count, 0.0f);
        std::vector<float> shortwave_down(interface_count, 0.0f);

        if (config.do_lw)
        {
            std::vector<float> absorption;
            std::vector<float> planck_fraction;
            std::vector<float> layer_planck;
            std::vector<float> interface_planck;
            std::vector<float> surface_planck;
            std::vector<float> cloud_absorption;
            std::vector<float> emissivity(column_count);
            std::vector<float> precipitable_water(column_count);
            std::vector<float> band_width;
            const std::size_t  band_count = coefficients_->longwave->grid.band_lower_wavenumber_cm.size();
            band_width.reserve(band_count);
            for (std::size_t band = 0; band < band_count; ++band)
                band_width.push_back(static_cast<float>(coefficients_->longwave->grid.band_upper_wavenumber_cm[band] -
                                                        coefficients_->longwave->grid.band_lower_wavenumber_cm[band]));

            for (std::size_t column = 0; column < column_count; ++column)
            {
                const auto& item = prepared[column];
                append_float(absorption, item.longwave_optics.absorption_optical_depth);
                append_float(planck_fraction, item.longwave_optics.planck_fraction);
                append_float(layer_planck, item.longwave_preparation.layer_planck);
                append_float(interface_planck, item.longwave_preparation.interface_planck);
                append_float(surface_planck, item.longwave_preparation.surface_planck);
                if (!config.clear_sky_only)
                    append_float(cloud_absorption, item.longwave_clouds.absorption_optical_depth);
                emissivity[column]         = static_cast<float>(columns[column].emissivity_lw);
                precipitable_water[column] = static_cast<float>(item.longwave_preparation.precipitable_water_cm);
            }
            const RrtmgLongwaveTransferBatchDispatch dispatch{absorption.data(),
                                                              planck_fraction.data(),
                                                              layer_planck.data(),
                                                              interface_planck.data(),
                                                              surface_planck.data(),
                                                              band_width.data(),
                                                              config.clear_sky_only ? nullptr : cloud_absorption.data(),
                                                              emissivity.data(),
                                                              precipitable_water.data(),
                                                              longwave_up.data(),
                                                              longwave_down.data(),
                                                              static_cast<int>(column_count),
                                                              static_cast<int>(layers)};
            if (!kernels.dispatch_rrtmg_longwave_transfer_batch(dispatch))
                return false;
        }

        if (config.do_sw)
        {
            std::vector<float> absorption;
            std::vector<float> rayleigh;
            std::vector<float> solar_source;
            std::vector<float> cloud_depth;
            std::vector<float> cloud_omega;
            std::vector<float> cloud_asymmetry;
            std::vector<float> cosine_zenith(column_count);
            std::vector<float> solar_irradiance(column_count);
            std::vector<float> surface_albedo(column_count);
            for (std::size_t column = 0; column < column_count; ++column)
            {
                const auto& item = prepared[column];
                append_float(absorption, item.shortwave_optics.absorption_optical_depth);
                append_float(rayleigh, item.shortwave_optics.rayleigh_optical_depth);
                append_float(solar_source, item.shortwave_optics.solar_source_fraction);
                if (!config.clear_sky_only)
                {
                    append_float(cloud_depth, item.shortwave_clouds.optical_depth);
                    append_float(cloud_omega, item.shortwave_clouds.single_scattering_albedo);
                    append_float(cloud_asymmetry, item.shortwave_clouds.asymmetry);
                }
                cosine_zenith[column]    = static_cast<float>(columns[column].mu0);
                solar_irradiance[column] = static_cast<float>(columns[column].S0);
                surface_albedo[column]   = static_cast<float>(columns[column].albedo_sw);
            }
            const RrtmgShortwaveTransferBatchDispatch dispatch{absorption.data(),
                                                               rayleigh.data(),
                                                               solar_source.data(),
                                                               config.clear_sky_only ? nullptr : cloud_depth.data(),
                                                               config.clear_sky_only ? nullptr : cloud_omega.data(),
                                                               config.clear_sky_only ? nullptr : cloud_asymmetry.data(),
                                                               cosine_zenith.data(),
                                                               solar_irradiance.data(),
                                                               surface_albedo.data(),
                                                               shortwave_up.data(),
                                                               shortwave_down.data(),
                                                               static_cast<int>(column_count),
                                                               static_cast<int>(layers)};
            if (!kernels.dispatch_rrtmg_shortwave_transfer_batch(dispatch))
                return false;
        }

        std::vector<RadiationColumnTendencies> candidates(column_count);
        std::vector<RadiationColumnFluxes>     flux_candidates(fluxes ? column_count : 0);
        std::vector<std::exception_ptr>        finalization_failures(column_count);
#pragma omp parallel for if (column_count >= 8) schedule(static)
        for (std::size_t column = 0; column < column_count; ++column)
        {
            try
            {
                auto& candidate = candidates[column];
                candidate.dTdt_rad.assign(layers, 0.0);
                candidate.dTdt_lw.assign(layers, 0.0);
                candidate.dTdt_sw.assign(layers, 0.0);
                std::vector<double> net_flux(layers + 1, 0.0);
                const std::size_t   offset = column * (layers + 1);
                if (config.do_lw)
                {
                    for (std::size_t interface = 0; interface <= layers; ++interface)
                        net_flux[interface] = static_cast<double>(longwave_up[offset + interface]) -
                                              static_cast<double>(longwave_down[offset + interface]);
                    radiative_transfer::heating_rate_from_flux_divergence(
                        *columns[column].rho, *columns[column].dz, net_flux, candidate.dTdt_lw);
                    candidate.atmospheric_power_lw_wm2 = -(net_flux.back() - net_flux.front());
                }
                if (config.do_sw)
                {
                    for (std::size_t interface = 0; interface <= layers; ++interface)
                        net_flux[interface] = static_cast<double>(shortwave_up[offset + interface]) -
                                              static_cast<double>(shortwave_down[offset + interface]);
                    radiative_transfer::heating_rate_from_flux_divergence(
                        *columns[column].rho, *columns[column].dz, net_flux, candidate.dTdt_sw);
                    candidate.atmospheric_power_sw_wm2 = -(net_flux.back() - net_flux.front());
                }
                if (fluxes)
                {
                    auto& f = flux_candidates[column];
                    for (auto* values :
                         {&f.Fup_lw, &f.Fdn_lw, &f.Fup_sw, &f.Fdn_sw, &f.Fnet_lw, &f.Fnet_sw, &f.Fnet_total})
                        values->resize(layers + 1);
                    for (std::size_t k = 0; k <= layers; ++k)
                    {
                        f.Fup_lw[k]     = config.do_lw ? longwave_up[offset + k] : 0;
                        f.Fdn_lw[k]     = config.do_lw ? longwave_down[offset + k] : 0;
                        f.Fup_sw[k]     = config.do_sw ? shortwave_up[offset + k] : 0;
                        f.Fdn_sw[k]     = config.do_sw ? shortwave_down[offset + k] : 0;
                        f.Fnet_lw[k]    = f.Fup_lw[k] - f.Fdn_lw[k];
                        f.Fnet_sw[k]    = f.Fup_sw[k] - f.Fdn_sw[k];
                        f.Fnet_total[k] = f.Fnet_lw[k] + f.Fnet_sw[k];
                    }
                }
                for (std::size_t layer = 0; layer < layers; ++layer)
                    candidate.dTdt_rad[layer] = candidate.dTdt_lw[layer] + candidate.dTdt_sw[layer];
                candidate.atmospheric_power_total_wm2 =
                    candidate.atmospheric_power_lw_wm2 + candidate.atmospheric_power_sw_wm2;
            }
            catch (...)
            {
                finalization_failures[column] = std::current_exception();
            }
        }
        rethrow_first(finalization_failures);
        for (std::size_t column = 0; column < column_count; ++column)
        {
            tendencies[column] = std::move(candidates[column]);
            if (fluxes)
                fluxes[column] = std::move(flux_candidates[column]);
        }
        return true;
    }
    catch (const std::bad_alloc&)
    {
        return false;
    }
    catch (const std::length_error&)
    {
        return false;
    }
}

} // namespace tmv
