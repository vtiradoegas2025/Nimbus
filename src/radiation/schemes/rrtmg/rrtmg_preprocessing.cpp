#include "rrtmg_preprocessing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

namespace tmv::rrtmg
{
namespace
{

constexpr std::size_t longwave_band_count         = 16;
constexpr std::size_t planck_temperature_count    = 181;
constexpr double      planck_temperature_origin_k = 159.0;

bool has_shape(const CoefficientArray& table, std::initializer_list<std::size_t> expected) noexcept
{
    return table.shape.size() == expected.size() &&
           std::equal(table.shape.begin(), table.shape.end(), expected.begin());
}

void validate_gas_profile(const GasProfile& profile, std::size_t layers, const char* name)
{
    if (profile.volume_mixing_ratio.size() != layers || profile.column_molecules_cm2.size() != layers)
        throw std::invalid_argument(std::string("RRTMG gas profile has an invalid shape: ") + name);
    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        if (!std::isfinite(profile.volume_mixing_ratio[layer]) || profile.volume_mixing_ratio[layer] < 0.0 ||
            !std::isfinite(profile.column_molecules_cm2[layer]) || profile.column_molecules_cm2[layer] < 0.0)
            throw std::invalid_argument(std::string("RRTMG gas profile contains invalid data: ") + name);
    }
}

void validate_atmosphere(const PreparedAtmosphere& atmosphere)
{
    const std::size_t layers = atmosphere.layer_pressure_pa.size();
    if (!std::isfinite(atmosphere.surface_temperature_k) || atmosphere.surface_temperature_k <= 0.0 || layers == 0 ||
        atmosphere.layer_temperature_k.size() != layers || atmosphere.interface_pressure_pa.size() != layers + 1 ||
        atmosphere.interface_temperature_k.size() != layers + 1 ||
        atmosphere.dry_air_column_molecules_cm2.size() != layers)
        throw std::invalid_argument("RRTMG longwave preparation shapes are invalid");

    const std::array<std::pair<const GasProfile*, const char*>, 12> gases{{
        {&atmosphere.gases.nitrogen, "nitrogen"},
        {&atmosphere.gases.ccl4, "ccl4"},
        {&atmosphere.gases.cfc11, "cfc11"},
        {&atmosphere.gases.cfc12, "cfc12"},
        {&atmosphere.gases.cfc22, "cfc22"},
        {&atmosphere.gases.water_vapor, "water_vapor"},
        {&atmosphere.gases.carbon_dioxide, "carbon_dioxide"},
        {&atmosphere.gases.ozone, "ozone"},
        {&atmosphere.gases.nitrous_oxide, "nitrous_oxide"},
        {&atmosphere.gases.carbon_monoxide, "carbon_monoxide"},
        {&atmosphere.gases.methane, "methane"},
        {&atmosphere.gases.oxygen, "oxygen"},
    }};
    for (const auto& gas : gases)
        validate_gas_profile(*gas.first, layers, gas.second);

    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        if (!std::isfinite(atmosphere.layer_pressure_pa[layer]) || atmosphere.layer_pressure_pa[layer] <= 0.0 ||
            !std::isfinite(atmosphere.layer_temperature_k[layer]) || atmosphere.layer_temperature_k[layer] <= 0.0 ||
            !std::isfinite(atmosphere.dry_air_column_molecules_cm2[layer]) ||
            atmosphere.dry_air_column_molecules_cm2[layer] <= 0.0)
            throw std::invalid_argument("RRTMG longwave atmosphere contains invalid data");
    }
    for (double temperature : atmosphere.interface_temperature_k)
    {
        if (!std::isfinite(temperature) || temperature <= 0.0)
            throw std::invalid_argument("RRTMG interface temperature contains invalid data");
    }
}

double interpolate_planck(const LongwaveCoefficients& coefficients, std::size_t band, double temperature_k)
{
    if (!std::isfinite(temperature_k))
        throw std::invalid_argument("RRTMG Planck temperature must be finite");
    if (!has_shape(coefficients.integrated_planck_function, {longwave_band_count, planck_temperature_count}) ||
        coefficients.integrated_planck_function.values.size() != longwave_band_count * planck_temperature_count)
        throw std::invalid_argument("RRTMG Planck coefficient table is malformed");

    const int    one_based_index = std::clamp(static_cast<int>(temperature_k - planck_temperature_origin_k), 1, 180);
    const double fraction        = temperature_k - planck_temperature_origin_k - static_cast<double>(one_based_index);
    const std::size_t lower      = static_cast<std::size_t>(one_based_index - 1);
    const std::size_t offset     = band * planck_temperature_count + lower;
    const auto&       values     = coefficients.integrated_planck_function.values;
    return values[offset] + fraction * (values[offset + 1] - values[offset]);
}

double gas_fraction_sum_without_water(const GasComposition& gases, std::size_t layer)
{
    return gases.carbon_dioxide.volume_mixing_ratio[layer] + gases.ozone.volume_mixing_ratio[layer] +
           gases.nitrous_oxide.volume_mixing_ratio[layer] + gases.carbon_monoxide.volume_mixing_ratio[layer] +
           gases.methane.volume_mixing_ratio[layer] + gases.oxygen.volume_mixing_ratio[layer];
}

LayerInterpolation
prepare_interpolation(const LongwaveCoefficients& coefficients, const PreparedAtmosphere& atmosphere, std::size_t layer)
{
    const double pressure_hpa  = atmosphere.layer_pressure_pa[layer] * 0.01;
    const double temperature_k = atmosphere.layer_temperature_k[layer];
    const double log_pressure  = std::log(pressure_hpa);

    const int one_based_pressure_index    = std::clamp(static_cast<int>(36.0 - 5.0 * (log_pressure + 0.04)), 1, 58);
    const std::size_t pressure_index      = static_cast<std::size_t>(one_based_pressure_index - 1);
    const std::size_t next_pressure_index = pressure_index + 1;
    const double      pressure_fraction   = 5.0 * ((6.96 - 0.2 * static_cast<double>(pressure_index)) - log_pressure);

    const auto temperature_cell = [temperature_k](double reference_temperature)
    {
        const double normalized      = (temperature_k - reference_temperature) / 15.0;
        const int    one_based_index = std::clamp(static_cast<int>(3.0 + normalized), 1, 4);
        const double fraction        = normalized - static_cast<double>(one_based_index - 3);
        return std::pair<std::size_t, double>{static_cast<std::size_t>(one_based_index - 1), fraction};
    };
    const auto lower_temperature = temperature_cell(coefficients.grid.reference_temperature_k[pressure_index]);
    const auto upper_temperature = temperature_cell(coefficients.grid.reference_temperature_k[next_pressure_index]);

    LayerInterpolation result;
    result.pressure_index                      = pressure_index;
    result.temperature_index_at_lower_pressure = lower_temperature.first;
    result.temperature_index_at_upper_pressure = upper_temperature.first;
    const double lower_pressure_weight         = 1.0 - pressure_fraction;
    result.pressure_temperature_weights        = {
        lower_pressure_weight * (1.0 - lower_temperature.second),
        pressure_fraction * (1.0 - upper_temperature.second),
        lower_pressure_weight * lower_temperature.second,
        pressure_fraction * upper_temperature.second,
    };
    result.lower_atmosphere = log_pressure > 4.56;

    const double water               = atmosphere.gases.water_vapor.volume_mixing_ratio[layer];
    const double scale               = pressure_hpa * (296.0 / 1013.0) / temperature_k;
    const double foreign_base        = scale / (1.0 + water);
    const double scaled_water_column = 1.0e-20 * atmosphere.gases.water_vapor.column_molecules_cm2[layer];
    result.water_foreign_factor      = scaled_water_column * foreign_base;
    result.water_self_factor         = scaled_water_column * water * foreign_base;

    if (result.lower_atmosphere)
    {
        const double foreign_coordinate = (332.0 - temperature_k) / 36.0;
        const int    foreign_one_based  = std::clamp(static_cast<int>(foreign_coordinate), 1, 2);
        result.water_foreign_index      = static_cast<std::size_t>(foreign_one_based - 1);
        result.water_foreign_fraction   = foreign_coordinate - foreign_one_based;

        const double self_coordinate = (temperature_k - 188.0) / 7.2;
        const int    self_one_based  = std::clamp(static_cast<int>(self_coordinate) - 7, 1, 9);
        result.water_self_index      = static_cast<std::size_t>(self_one_based - 1);
        result.water_self_fraction   = self_coordinate - (self_one_based + 7);
    }
    else
    {
        result.water_foreign_index    = 2;
        result.water_foreign_fraction = (temperature_k - 188.0) / 36.0 - 1.0;
        result.water_self_index       = 0;
        result.water_self_fraction    = 0.0;
    }

    const double minor_coordinate     = (temperature_k - 180.8) / 7.2;
    const int    minor_one_based      = std::clamp(static_cast<int>(minor_coordinate), 1, 18);
    result.minor_temperature_index    = static_cast<std::size_t>(minor_one_based - 1);
    result.minor_temperature_fraction = minor_coordinate - minor_one_based;
    result.minor_scale                = pressure_hpa / temperature_k;
    const double broadening_fraction  = std::max(0.0, 1.0 - gas_fraction_sum_without_water(atmosphere.gases, layer));
    result.nitrogen_minor_scale       = result.minor_scale * broadening_fraction / (1.0 + water);
    return result;
}

} // namespace

LongwavePreparation prepare_longwave(const LongwaveCoefficients& coefficients,
                                     const PreparedAtmosphere&   atmosphere,
                                     double                      surface_emissivity)
{
    if (!std::isfinite(surface_emissivity) || surface_emissivity < 0.0 || surface_emissivity > 1.0)
        throw std::invalid_argument("RRTMG longwave emissivity must be in [0,1]");
    validate_atmosphere(atmosphere);
    const std::size_t layers = atmosphere.layer_pressure_pa.size();
    if (coefficients.grid.reference_temperature_k.size() != 59)
        throw std::invalid_argument("RRTMG longwave reference-temperature grid is malformed");

    LongwavePreparation result;
    result.interpolation.resize(layers);
    result.layer_planck.resize(layers * longwave_band_count);
    result.interface_planck.resize((layers + 1) * longwave_band_count);
    result.surface_planck.resize(longwave_band_count);
    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        result.interpolation[layer] = prepare_interpolation(coefficients, atmosphere, layer);
        result.precipitable_water_cm +=
            atmosphere.gases.water_vapor.column_molecules_cm2[layer] / 6.02214076e23 * 18.01528;
    }
    for (std::size_t band = 0; band < longwave_band_count; ++band)
    {
        result.surface_planck[band] =
            surface_emissivity * interpolate_planck(coefficients, band, atmosphere.surface_temperature_k);
        for (std::size_t layer = 0; layer < layers; ++layer)
            result.layer_planck[layer * longwave_band_count + band] =
                interpolate_planck(coefficients, band, atmosphere.layer_temperature_k[layer]);
        for (std::size_t interface = 0; interface <= layers; ++interface)
            result.interface_planck[interface * longwave_band_count + band] =
                interpolate_planck(coefficients, band, atmosphere.interface_temperature_k[interface]);
    }
    if (!std::isfinite(result.precipitable_water_cm) || result.precipitable_water_cm < 0.0)
        throw std::runtime_error("RRTMG precipitable-water calculation failed");
    return result;
}

} // namespace tmv::rrtmg
