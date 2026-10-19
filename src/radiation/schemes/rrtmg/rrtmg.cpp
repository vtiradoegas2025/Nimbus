#include "rrtmg.hpp"

#include "radiation/base/radiative_transfer.hpp"
#include "radiation/rrtmg_config_limits.hpp"
#include "rrtmg_atmosphere.hpp"
#include "rrtmg_cloud_optics.hpp"
#include "rrtmg_coefficients.hpp"
#include "rrtmg_optics.hpp"
#include "rrtmg_preprocessing.hpp"
#include "rrtmg_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace tmv
{
namespace
{

std::string coefficient_path(const char* environment, const char* fallback)
{
    const char* value = std::getenv(environment);
    return value != nullptr && value[0] != '\0' ? value : fallback;
}

void validate_config(const RadiationConfig& config)
{
    if (!config.do_lw && !config.do_sw)
        throw std::invalid_argument("RRTMG requires longwave or shortwave transfer");
    if (!std::isfinite(config.dt_radiation) || config.dt_radiation <= 0.0 || config.use_subcycling)
        throw std::invalid_argument("RRTMG cadence configuration is invalid");
    if (!rrtmg_config_limits::valid_liquid_effective_radius(config.rrtmg_liquid_effective_radius_um) ||
        !rrtmg_config_limits::valid_ice_effective_radius(config.rrtmg_ice_effective_radius_um))
        throw std::invalid_argument("RRTMG cloud effective radii are outside the native optics bounds");
    if (config.do_sw)
    {
        (void)idealized_solar_cosine_zenith(config, 12.0 * 3600.0);
        if (config.terrain_sw_cosine_nodes < 1 ||
            config.terrain_sw_cosine_nodes > 32 ||
            config.terrain_sw_azimuth_nodes < 4 ||
            config.terrain_sw_azimuth_nodes > 128 ||
            config.terrain_sw_azimuth_nodes % 4 != 0)
            throw std::invalid_argument(
                "RRTMG mapped shortwave ordinate configuration is invalid");
    }
}

bool same_science_config(const RadiationConfig& left, const RadiationConfig& right)
{
    return left.clear_sky_only == right.clear_sky_only &&
           left.rrtmg_ozone_vmr == right.rrtmg_ozone_vmr && left.rrtmg_co2_vmr == right.rrtmg_co2_vmr &&
           left.rrtmg_ch4_vmr == right.rrtmg_ch4_vmr && left.rrtmg_n2o_vmr == right.rrtmg_n2o_vmr &&
           left.rrtmg_co_vmr == right.rrtmg_co_vmr && left.rrtmg_o2_vmr == right.rrtmg_o2_vmr &&
           left.rrtmg_n2_vmr == right.rrtmg_n2_vmr && left.rrtmg_cfc11_vmr == right.rrtmg_cfc11_vmr &&
           left.rrtmg_cfc12_vmr == right.rrtmg_cfc12_vmr && left.rrtmg_cfc22_vmr == right.rrtmg_cfc22_vmr &&
           left.rrtmg_ccl4_vmr == right.rrtmg_ccl4_vmr &&
           left.rrtmg_liquid_effective_radius_um == right.rrtmg_liquid_effective_radius_um &&
           left.rrtmg_ice_effective_radius_um == right.rrtmg_ice_effective_radius_um;
}

void initialize_fluxes(RadiationColumnFluxes& fluxes, std::size_t layers)
{
    const auto clear = [layers](std::vector<double>& values) { values.assign(layers + 1, 0.0); };
    clear(fluxes.Fup_lw);
    clear(fluxes.Fdn_lw);
    clear(fluxes.Fup_sw);
    clear(fluxes.Fdn_sw);
    clear(fluxes.Fnet_lw);
    clear(fluxes.Fnet_sw);
    clear(fluxes.Fnet_total);
}

} // namespace

RrtmgScheme::RrtmgScheme() = default;

std::unique_ptr<RadiationSchemeBase> RrtmgScheme::clone() const
{
    return std::make_unique<RrtmgScheme>(*this);
}

std::string RrtmgScheme::name() const
{
    return "rrtmg";
}

int RrtmgScheme::required_fields() const
{
    return static_cast<int>(RadiationRequirements::TEMPERATURE) |
           static_cast<int>(RadiationRequirements::THETA_PRESSURE) |
           static_cast<int>(RadiationRequirements::SOLAR_GEOMETRY) |
           static_cast<int>(RadiationRequirements::INTERFACE_GEOMETRY);
}

void RrtmgScheme::initialize(const RadiationConfig& config)
{
    validate_config(config);
    auto coefficients = std::make_shared<Coefficients>();
    coefficients->longwave =
        rrtmg::shared_longwave_coefficients(coefficient_path("NIMBUS_RRTMG_LW_COEFFICIENTS", "data/rrtmg/rrtmg_lw.nc"));
    if (config.do_sw)
        coefficients->shortwave = rrtmg::shared_shortwave_coefficients(
            coefficient_path("NIMBUS_RRTMG_SW_COEFFICIENTS", "data/rrtmg/rrtmg_sw.nc"));
    coefficients_       = std::move(coefficients);
    initialized_config_ = config;
    initialized_        = true;
}

void RrtmgScheme::validate_compute_config(
    const RadiationConfig& config) const
{
    if (!initialized_ || coefficients_ == nullptr)
        throw std::logic_error(
            "RRTMG must be initialized before column transfer");
    validate_config(config);
    if ((config.do_lw && !initialized_config_.do_lw) ||
        (config.do_sw && !initialized_config_.do_sw) ||
        !same_science_config(config, initialized_config_))
        throw std::invalid_argument(
            "RRTMG compute configuration differs from initialized science state");
}

void RrtmgScheme::compute_column(const RadiationConfig&          config,
                                 const RadiationColumnStateView& column,
                                 RadiationColumnTendencies&      tendencies,
                                 RadiationColumnFluxes*          fluxes)
{
    validate_compute_config(config);
    if (column.rho == nullptr || column.dz == nullptr)
        throw std::invalid_argument("RRTMG requires density and layer thickness");

    const rrtmg::PreparedAtmosphere atmosphere = rrtmg::prepare_atmosphere(config, column);
    const std::size_t               layers     = atmosphere.layer_pressure_pa.size();
    RadiationColumnTendencies       candidate;
    candidate.dTdt_rad.assign(layers, 0.0);
    candidate.dTdt_lw.assign(layers, 0.0);
    candidate.dTdt_sw.assign(layers, 0.0);
    RadiationColumnFluxes candidate_fluxes;
    if (fluxes != nullptr)
        initialize_fluxes(candidate_fluxes, layers);
    std::vector<double> local_net_flux;
    auto                working_net_flux = [&]() -> std::vector<double>&
    {
        if (column.workspace != nullptr)
            return column.workspace->flux_net;
        return local_net_flux;
    };

    rrtmg::LongwavePreparation preparation;
    if (config.do_lw)
    {
        const auto& longwave = *coefficients_->longwave;
        preparation          = rrtmg::prepare_longwave(longwave, atmosphere, column.emissivity_lw);
        const auto optics    = rrtmg::compute_longwave_gas_optics(longwave, atmosphere, preparation);
        rrtmg::LongwaveCloudOptics cloud_optics;
        const rrtmg::LongwaveCloudOptics* clouds = nullptr;
        if (!config.clear_sky_only)
        {
            cloud_optics = rrtmg::compute_longwave_cloud_optics(atmosphere);
            clouds       = &cloud_optics;
        }
        auto spectral_fluxes =
            rrtmg::solve_longwave(longwave, atmosphere, preparation, optics, column.emissivity_lw, clouds);
        std::vector<double>& net_flux = fluxes != nullptr ? candidate_fluxes.Fnet_lw : working_net_flux();
        net_flux.resize(layers + 1);
        for (std::size_t interface = 0; interface <= layers; ++interface)
            net_flux[interface] = spectral_fluxes.upward_wm2[interface] - spectral_fluxes.downward_wm2[interface];
        radiative_transfer::heating_rate_from_flux_divergence(*column.rho, *column.dz, net_flux, candidate.dTdt_lw);
        candidate.atmospheric_power_lw_wm2 = -(net_flux.back() - net_flux.front());
        if (fluxes != nullptr)
        {
            candidate_fluxes.Fup_lw = std::move(spectral_fluxes.upward_wm2);
            candidate_fluxes.Fdn_lw = std::move(spectral_fluxes.downward_wm2);
        }
    }
    if (config.do_sw)
    {
        if (preparation.interpolation.empty())
        {
            const auto& longwave = *coefficients_->longwave;
            if (longwave.grid.reference_temperature_k.empty())
                throw std::logic_error("shortwave-only RRTMG preprocessing requires the shared reference grid");
            preparation = rrtmg::prepare_longwave(longwave, atmosphere, column.emissivity_lw);
        }
        if (coefficients_->shortwave == nullptr)
            throw std::logic_error("RRTMG shortwave coefficients are unavailable");
        const auto optics =
            rrtmg::compute_shortwave_gas_optics(*coefficients_->shortwave, atmosphere, preparation.interpolation);
        rrtmg::ShortwaveCloudOptics cloud_optics;
        const rrtmg::ShortwaveCloudOptics* clouds = nullptr;
        if (!config.clear_sky_only)
        {
            cloud_optics = rrtmg::compute_shortwave_cloud_optics(*coefficients_->shortwave, atmosphere);
            clouds       = &cloud_optics;
        }
        auto spectral_fluxes = rrtmg::solve_shortwave(
            *coefficients_->shortwave, optics, column.mu0, column.S0, column.albedo_sw, clouds);
        std::vector<double>& net_flux = fluxes != nullptr ? candidate_fluxes.Fnet_sw : working_net_flux();
        net_flux.resize(layers + 1);
        for (std::size_t interface = 0; interface <= layers; ++interface)
            net_flux[interface] = spectral_fluxes.upward_wm2[interface] - spectral_fluxes.downward_wm2[interface];
        radiative_transfer::heating_rate_from_flux_divergence(*column.rho, *column.dz, net_flux, candidate.dTdt_sw);
        candidate.atmospheric_power_sw_wm2 = -(net_flux.back() - net_flux.front());
        if (fluxes != nullptr)
        {
            candidate_fluxes.Fup_sw = std::move(spectral_fluxes.upward_wm2);
            candidate_fluxes.Fdn_sw = std::move(spectral_fluxes.downward_wm2);
        }
    }
    for (std::size_t layer = 0; layer < layers; ++layer)
        candidate.dTdt_rad[layer] = candidate.dTdt_lw[layer] + candidate.dTdt_sw[layer];
    candidate.atmospheric_power_total_wm2 = candidate.atmospheric_power_lw_wm2 + candidate.atmospheric_power_sw_wm2;
    if (fluxes != nullptr)
    {
        for (std::size_t interface = 0; interface <= layers; ++interface)
            candidate_fluxes.Fnet_total[interface] =
                candidate_fluxes.Fnet_lw[interface] + candidate_fluxes.Fnet_sw[interface];
    }

    tendencies = std::move(candidate);
    if (fluxes != nullptr)
        *fluxes = std::move(candidate_fluxes);
}

bool RrtmgScheme::compute_mapped_shortwave_optics(
    const RadiationConfig& config,
    const RadiationColumnStateView& column,
    MappedShortwaveColumnOptics& optics)
{
    validate_compute_config(config);
    if (!config.do_sw || coefficients_->shortwave == nullptr)
        return false;

    const rrtmg::PreparedAtmosphere atmosphere =
        rrtmg::prepare_atmosphere(config, column);
    const auto& longwave = *coefficients_->longwave;
    if (longwave.grid.reference_temperature_k.empty())
        throw std::logic_error(
            "mapped RRTMG shortwave requires the shared reference grid");
    const rrtmg::LongwavePreparation preparation =
        rrtmg::prepare_longwave(longwave, atmosphere, column.emissivity_lw);
    const auto& shortwave = *coefficients_->shortwave;
    const rrtmg::ShortwaveOptics gas =
        rrtmg::compute_shortwave_gas_optics(
            shortwave, atmosphere, preparation.interpolation);
    rrtmg::ShortwaveCloudOptics cloud;
    if (!config.clear_sky_only)
        cloud = rrtmg::compute_shortwave_cloud_optics(shortwave, atmosphere);

    MappedShortwaveColumnOptics candidate;
    candidate.layers = gas.layers;
    candidate.g_points = gas.g_points;
    candidate.absorption_optical_depth.resize(gas.layers * gas.g_points);
    candidate.scattering_optical_depth.resize(gas.layers * gas.g_points);
    candidate.scattering_asymmetry.resize(gas.layers * gas.g_points);
    candidate.solar_source_fraction = gas.solar_source_fraction;

    for (std::size_t g_point = 0; g_point < gas.g_points; ++g_point)
    {
        const auto upper = std::upper_bound(
            shortwave.grid.reduced_g_point_offsets.begin(),
            shortwave.grid.reduced_g_point_offsets.end(), g_point);
        if (upper == shortwave.grid.reduced_g_point_offsets.begin() ||
            upper == shortwave.grid.reduced_g_point_offsets.end())
            throw std::logic_error(
                "mapped RRTMG shortwave g-point has no spectral band");
        const std::size_t band = static_cast<std::size_t>(
            upper - shortwave.grid.reduced_g_point_offsets.begin() - 1);
        for (std::size_t layer = 0; layer < gas.layers; ++layer)
        {
            const std::size_t index = layer * gas.g_points + g_point;
            const std::size_t cloud_index = layer * 14 + band;
            const double cloud_depth = config.clear_sky_only
                ? 0.0 : cloud.optical_depth[cloud_index];
            const double cloud_omega = config.clear_sky_only
                ? 0.0 : cloud.single_scattering_albedo[cloud_index];
            const double cloud_scattering = cloud_depth * cloud_omega;
            const double scattering =
                gas.rayleigh_optical_depth[index] + cloud_scattering;
            const double absorption = gas.absorption_optical_depth[index] +
                cloud_depth - cloud_scattering;
            const double asymmetry = scattering > 0.0 && cloud_scattering > 0.0
                ? cloud_scattering * cloud.asymmetry[cloud_index] / scattering
                : 0.0;
            if (!std::isfinite(absorption) || absorption < 0.0 ||
                !std::isfinite(scattering) || scattering < 0.0 ||
                !std::isfinite(asymmetry) || asymmetry < 0.0 ||
                asymmetry > 1.0)
                throw std::runtime_error(
                    "mapped RRTMG shortwave produced invalid spectral optics");
            candidate.absorption_optical_depth[index] = absorption;
            candidate.scattering_optical_depth[index] = scattering;
            candidate.scattering_asymmetry[index] = asymmetry;
        }
    }
    if (candidate.layers == 0 || candidate.g_points != 112 ||
        candidate.solar_source_fraction.size() != candidate.g_points ||
        std::any_of(candidate.solar_source_fraction.begin(),
                    candidate.solar_source_fraction.end(),
                    [](double value)
                    { return !std::isfinite(value) || value < 0.0; }))
        throw std::runtime_error(
            "mapped RRTMG shortwave produced malformed spectral state");
    const double source_sum = std::accumulate(
        candidate.solar_source_fraction.begin(),
        candidate.solar_source_fraction.end(), 0.0);
    if (!std::isfinite(source_sum) || source_sum <= 0.0)
        throw std::runtime_error(
            "mapped RRTMG shortwave source weights do not normalize");
    for (double& source : candidate.solar_source_fraction)
        source /= source_sum;

    optics = std::move(candidate);
    return true;
}

} // namespace tmv
