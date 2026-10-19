#include "rrtmg_atmosphere.hpp"

#include "radiation/rrtmg_config_limits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv::rrtmg
{
namespace
{

constexpr double molar_mass_dry_air_gmol             = 28.966;
constexpr double molar_mass_water_gmol               = 18.01528;
constexpr double avogadro_molecules_per_mol          = 6.02214076e23;
constexpr double square_centimeters_per_square_meter = 1.0e4;

const std::vector<double>& require_profile(const std::vector<double>* profile, const char* name)
{
    if (profile == nullptr)
        throw std::invalid_argument(std::string("RRTMG requires column profile ") + name);
    return *profile;
}

void require_gas(double value, double maximum, const char* name)
{
    if (!std::isfinite(value) || value < 0.0 || value > maximum)
        throw std::invalid_argument(std::string("RRTMG gas mole fraction is invalid: ") + name);
}

double log_pressure_interpolate(double lower_pressure,
                                double upper_pressure,
                                double interface_pressure,
                                double lower_temperature,
                                double upper_temperature)
{
    const double denominator = std::log(upper_pressure) - std::log(lower_pressure);
    if (std::abs(denominator) < 1.0e-14)
        return 0.5 * (lower_temperature + upper_temperature);
    const double weight = (std::log(interface_pressure) - std::log(lower_pressure)) / denominator;
    return lower_temperature + std::clamp(weight, 0.0, 1.0) * (upper_temperature - lower_temperature);
}

void initialize_gas(GasProfile&                gas,
                    std::size_t                layers,
                    double                     fallback_volume_mixing_ratio,
                    const std::vector<double>* profile,
                    double                     maximum,
                    const char*                name)
{
    if (profile == nullptr)
    {
        gas.volume_mixing_ratio.assign(layers, fallback_volume_mixing_ratio);
    }
    else
    {
        if (profile->size() != layers)
            throw std::invalid_argument(std::string("RRTMG gas profile has inconsistent shape: ") + name);
        for (double value : *profile)
            require_gas(value, maximum, name);
        gas.volume_mixing_ratio = *profile;
    }
    gas.column_molecules_cm2.resize(layers);
}

} // namespace

PreparedAtmosphere prepare_atmosphere(const RadiationConfig& config, const RadiationColumnStateView& column)
{
    const auto&       pressure              = require_profile(column.p, "p");
    const auto&       interface_pressure    = require_profile(column.p_int, "p_int");
    const auto&       temperature           = require_profile(column.T, "T");
    const auto*       interface_temperature = column.T_int;
    const auto&       density               = require_profile(column.rho, "rho");
    const auto&       thickness             = require_profile(column.dz, "dz");
    const auto&       vapor                 = require_profile(column.qv, "qv");
    const std::size_t layers                = pressure.size();
    if (layers == 0 || interface_pressure.size() != layers + 1 || temperature.size() != layers ||
        density.size() != layers || thickness.size() != layers || vapor.size() != layers)
        throw std::invalid_argument("RRTMG column profiles have inconsistent shapes");
    if (interface_temperature != nullptr && interface_temperature->size() != layers + 1)
        throw std::invalid_argument("RRTMG interface-temperature profile has inconsistent shape");
    if (!std::isfinite(column.Tsfc) || column.Tsfc < 150.0 || column.Tsfc > 400.0)
        throw std::invalid_argument("RRTMG surface temperature must be in [150,400] K");

    using namespace tmv::rrtmg_config_limits;
    require_gas(config.rrtmg_ozone_vmr, ozone_vmr_max, "ozone");
    require_gas(config.rrtmg_co2_vmr, co2_vmr_max, "co2");
    require_gas(config.rrtmg_ch4_vmr, ch4_vmr_max, "ch4");
    require_gas(config.rrtmg_n2o_vmr, n2o_vmr_max, "n2o");
    require_gas(config.rrtmg_co_vmr, co_vmr_max, "co");
    require_gas(config.rrtmg_o2_vmr, o2_vmr_max, "o2");
    require_gas(config.rrtmg_n2_vmr, n2_vmr_max, "n2");
    require_gas(config.rrtmg_cfc11_vmr, halocarbon_vmr_max, "cfc11");
    require_gas(config.rrtmg_cfc12_vmr, halocarbon_vmr_max, "cfc12");
    require_gas(config.rrtmg_cfc22_vmr, halocarbon_vmr_max, "cfc22");
    require_gas(config.rrtmg_ccl4_vmr, halocarbon_vmr_max, "ccl4");
    if (!valid_liquid_effective_radius(config.rrtmg_liquid_effective_radius_um) ||
        !valid_ice_effective_radius(config.rrtmg_ice_effective_radius_um))
        throw std::invalid_argument("RRTMG cloud effective radius is outside supported bounds");

    const std::vector<double>* liquid = column.qc;
    const std::vector<double>* ice    = column.qi;
    if (!config.clear_sky_only &&
        (liquid == nullptr || ice == nullptr || liquid->size() != layers || ice->size() != layers))
        throw std::invalid_argument("cloudy RRTMG requires shape-matched qc and qi profiles");

    // Validate the entire caller-owned state before allocating or evaluating
    // logarithms. Preparation is therefore fail-closed even when a malformed
    // value appears late in the column.
    for (std::size_t k = 0; k < layers; ++k)
    {
        if (!std::isfinite(pressure[k]) || pressure[k] <= 0.0 || !std::isfinite(temperature[k]) ||
            temperature[k] <= 0.0 || !std::isfinite(density[k]) || density[k] <= 0.0 || !std::isfinite(thickness[k]) ||
            thickness[k] <= 0.0 || !std::isfinite(vapor[k]) || vapor[k] < 0.0 || vapor[k] > 0.5 ||
            !std::isfinite(interface_pressure[k]) || !std::isfinite(interface_pressure[k + 1]) ||
            interface_pressure[k] <= 0.0 || interface_pressure[k + 1] < 0.0 ||
            interface_pressure[k + 1] > interface_pressure[k] || pressure[k] > interface_pressure[k] ||
            pressure[k] < interface_pressure[k + 1])
            throw std::invalid_argument("RRTMG column contains unphysical state");
        if (!config.clear_sky_only &&
            (!std::isfinite((*liquid)[k]) || (*liquid)[k] < 0.0 || !std::isfinite((*ice)[k]) || (*ice)[k] < 0.0))
            throw std::invalid_argument("RRTMG condensate profiles must be finite and non-negative");
    }
    if (interface_temperature != nullptr)
    {
        for (double value : *interface_temperature)
            if (!std::isfinite(value) || value <= 0.0)
                throw std::invalid_argument("RRTMG interface-temperature profile contains invalid state");
    }

    PreparedAtmosphere result;
    result.surface_temperature_k = column.Tsfc;
    result.layer_pressure_pa     = pressure;
    result.interface_pressure_pa = interface_pressure;
    result.layer_temperature_k   = temperature;
    result.interface_temperature_k.resize(layers + 1);
    result.dry_air_column_molecules_cm2.resize(layers);
    result.gases.water_vapor.volume_mixing_ratio.resize(layers);
    result.gases.water_vapor.column_molecules_cm2.resize(layers);
    initialize_gas(result.gases.nitrogen, layers, config.rrtmg_n2_vmr, column.nitrogen_vmr, n2_vmr_max, "n2");
    initialize_gas(result.gases.ccl4, layers, config.rrtmg_ccl4_vmr, column.ccl4_vmr, halocarbon_vmr_max, "ccl4");
    initialize_gas(result.gases.cfc11, layers, config.rrtmg_cfc11_vmr, column.cfc11_vmr, halocarbon_vmr_max, "cfc11");
    initialize_gas(result.gases.cfc12, layers, config.rrtmg_cfc12_vmr, column.cfc12_vmr, halocarbon_vmr_max, "cfc12");
    initialize_gas(result.gases.cfc22, layers, config.rrtmg_cfc22_vmr, column.cfc22_vmr, halocarbon_vmr_max, "cfc22");
    initialize_gas(result.gases.carbon_dioxide, layers, config.rrtmg_co2_vmr, column.carbon_dioxide_vmr, co2_vmr_max, "co2");
    initialize_gas(result.gases.ozone, layers, config.rrtmg_ozone_vmr, column.ozone_vmr, ozone_vmr_max, "ozone");
    initialize_gas(result.gases.nitrous_oxide, layers, config.rrtmg_n2o_vmr, column.nitrous_oxide_vmr, n2o_vmr_max, "n2o");
    initialize_gas(result.gases.carbon_monoxide, layers, config.rrtmg_co_vmr, column.carbon_monoxide_vmr, co_vmr_max, "co");
    initialize_gas(result.gases.methane, layers, config.rrtmg_ch4_vmr, column.methane_vmr, ch4_vmr_max, "ch4");
    initialize_gas(result.gases.oxygen, layers, config.rrtmg_o2_vmr, column.oxygen_vmr, o2_vmr_max, "o2");

    for (std::size_t k = 0; k < layers; ++k)
    {
        const double declared_dry_gas_fraction =
            result.gases.nitrogen.volume_mixing_ratio[k] + result.gases.oxygen.volume_mixing_ratio[k] +
            result.gases.carbon_dioxide.volume_mixing_ratio[k] + result.gases.methane.volume_mixing_ratio[k] +
            result.gases.nitrous_oxide.volume_mixing_ratio[k] + result.gases.carbon_monoxide.volume_mixing_ratio[k] +
            result.gases.ozone.volume_mixing_ratio[k] + result.gases.cfc11.volume_mixing_ratio[k] +
            result.gases.cfc12.volume_mixing_ratio[k] + result.gases.cfc22.volume_mixing_ratio[k] +
            result.gases.ccl4.volume_mixing_ratio[k];
        if (!std::isfinite(declared_dry_gas_fraction) || declared_dry_gas_fraction > 1.0)
            throw std::invalid_argument("RRTMG declared dry-gas mole fractions exceed unity");
    }
    result.cloud_fraction.assign(layers, 0.0);
    result.liquid_water_path_gm2.assign(layers, 0.0);
    result.ice_water_path_gm2.assign(layers, 0.0);
    result.liquid_effective_radius_um.assign(layers, config.rrtmg_liquid_effective_radius_um);
    result.ice_effective_radius_um.assign(layers, config.rrtmg_ice_effective_radius_um);
    if (interface_temperature != nullptr)
    {
        result.interface_temperature_k = *interface_temperature;
    }
    else
    {
        result.interface_temperature_k.front() = column.Tsfc;
        for (std::size_t face = 1; face < layers; ++face)
            result.interface_temperature_k[face] = log_pressure_interpolate(
                pressure[face - 1], pressure[face], interface_pressure[face], temperature[face - 1], temperature[face]);
        result.interface_temperature_k.back() = temperature.back();
    }

    for (std::size_t k = 0; k < layers; ++k)
    {
        const double dry_air_column = density[k] * thickness[k] * avogadro_molecules_per_mol /
                                      (molar_mass_dry_air_gmol * 1.0e-3) / square_centimeters_per_square_meter;
        result.dry_air_column_molecules_cm2[k]          = dry_air_column;
        result.gases.water_vapor.volume_mixing_ratio[k] = vapor[k] * molar_mass_dry_air_gmol / molar_mass_water_gmol;
        result.gases.water_vapor.column_molecules_cm2[k] =
            dry_air_column * result.gases.water_vapor.volume_mixing_ratio[k];
        const auto set_column = [k, dry_air_column](GasProfile& gas)
        { gas.column_molecules_cm2[k] = dry_air_column * gas.volume_mixing_ratio[k]; };
        set_column(result.gases.nitrogen);
        set_column(result.gases.ccl4);
        set_column(result.gases.cfc11);
        set_column(result.gases.cfc12);
        set_column(result.gases.cfc22);
        set_column(result.gases.carbon_dioxide);
        set_column(result.gases.ozone);
        set_column(result.gases.nitrous_oxide);
        set_column(result.gases.carbon_monoxide);
        set_column(result.gases.methane);
        set_column(result.gases.oxygen);

        if (!config.clear_sky_only)
        {
            const double qc                 = (*liquid)[k];
            const double qi                 = (*ice)[k];
            result.liquid_water_path_gm2[k] = 1000.0 * density[k] * thickness[k] * qc;
            result.ice_water_path_gm2[k]    = 1000.0 * density[k] * thickness[k] * qi;
            result.cloud_fraction[k]        = (qc > 0.0 || qi > 0.0) ? 1.0 : 0.0;
        }
    }
    return result;
}

} // namespace tmv::rrtmg
