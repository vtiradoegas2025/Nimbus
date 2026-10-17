#include "rrtmg_optics.hpp"

#include "rrtmg_optics_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv::rrtmg
{
namespace
{

std::string key_name(const std::vector<std::string>& names, std::size_t band, std::size_t species)
{
    constexpr std::size_t bands = 14;
    if (names.size() != 2 * bands || band >= bands || species >= 2)
        throw std::invalid_argument("RRTMG shortwave key-species table is malformed");
    return names[species * bands + band];
}

double key_species_ratio(std::size_t band)
{
    constexpr std::array<double, 14> ratios{
        252.131,
        0.364641,
        38.9589,
        5.49281,
        1.0,
        0.0045321,
        1.6 * 0.022708,
        1.0,
        0.124692,
        1.0,
        1.0,
        1.0,
        6.67029e-7,
        1.0,
    };
    return ratios[band];
}

bool uses_lower_water_continuum(std::size_t band) noexcept
{
    return band <= 8 || band == 13;
}

bool uses_upper_water_foreign_continuum(std::size_t band) noexcept
{
    return band == 1 || band == 4 || band == 5;
}

std::size_t solar_source_layer(std::size_t band, const std::vector<LayerInterpolation>& interpolation)
{
    constexpr std::array<std::size_t, 14> reference_pressure_index{
        18,
        30,
        6,
        3,
        3,
        8,
        2,
        6,
        1,
        2,
        59,
        32,
        42,
        49,
    };
    // Band 24's reference index is the first pressure cell. The reference
    // loop therefore retains its initialized last lower-atmosphere layer.
    if (band == 8)
    {
        for (std::size_t layer = interpolation.size(); layer-- > 0;)
            if (interpolation[layer].lower_atmosphere)
                return layer;
    }
    const std::size_t reference = reference_pressure_index[band];
    for (std::size_t layer = 1; layer < interpolation.size(); ++layer)
    {
        const std::size_t below = interpolation[layer - 1].pressure_index + 1;
        const std::size_t here  = interpolation[layer].pressure_index + 1;
        if (below < reference && here >= reference)
            return layer;
    }
    return interpolation.size() - 1;
}

double source_eta(const optics_internal::KeySpecies& species, std::size_t layer, double reference_ratio)
{
    if (species.first == nullptr || species.second == nullptr)
        return 0.0;
    const double first  = optics_internal::scaled_column(*species.first, layer);
    const double second = reference_ratio * optics_internal::scaled_column(*species.second, layer);
    return first + second > 0.0 ? first / (first + second) : 0.0;
}

} // namespace

ShortwaveOptics compute_shortwave_gas_optics(const ShortwaveCoefficients&           coefficients,
                                             const PreparedAtmosphere&              atmosphere,
                                             const std::vector<LayerInterpolation>& interpolation)
{
    constexpr std::size_t bands  = 14;
    const std::size_t     layers = atmosphere.layer_pressure_pa.size();
    if (interpolation.size() != layers || coefficients.grid.reduced_g_point_offsets.size() != bands + 1 ||
        coefficients.grid.reduced_g_point_offsets.back() != 112)
        throw std::invalid_argument("RRTMG shortwave optics shapes are invalid");

    ShortwaveOptics result;
    result.layers   = layers;
    result.g_points = 112;
    result.absorption_optical_depth.assign(layers * result.g_points, 0.0);
    result.rayleigh_optical_depth.assign(layers * result.g_points, 0.0);
    result.solar_source_fraction.assign(result.g_points, 0.0);

    for (std::size_t band = 0; band < bands; ++band)
    {
        const std::size_t count        = static_cast<std::size_t>(coefficients.grid.reduced_g_points_per_band[band]);
        const std::size_t offset       = coefficients.grid.reduced_g_point_offsets[band];
        const std::size_t source_layer = solar_source_layer(band, interpolation);
        for (std::size_t layer = 0; layer < layers; ++layer)
        {
            const auto& cell = interpolation[layer];
            const auto& names =
                cell.lower_atmosphere ? coefficients.lower_key_species_names : coefficients.upper_key_species_names;
            const auto species = optics_internal::key_species(
                coefficients.grid, atmosphere, key_name(names, band, 0), key_name(names, band, 1));
            const std::size_t       pressure_count = cell.lower_atmosphere ? 13 : 47;
            const std::size_t       ratio_count    = cell.lower_atmosphere ? 9 : 5;
            const CoefficientArray& key_table =
                cell.lower_atmosphere ? coefficients.key_absorption_lower : coefficients.key_absorption_upper;
            const CoefficientArray& quiet_sun_table =
                cell.lower_atmosphere ? coefficients.quiet_sun_lower : coefficients.quiet_sun_upper;
            const CoefficientArray& facular_table =
                cell.lower_atmosphere ? coefficients.facular_lower : coefficients.facular_upper;
            const CoefficientArray& sunspot_table =
                cell.lower_atmosphere ? coefficients.sunspot_lower : coefficients.sunspot_upper;
            const CoefficientArray& rayleigh_table =
                cell.lower_atmosphere ? coefficients.rayleigh_extinction_lower : coefficients.rayleigh_extinction_upper;
            const double reference_ratio = key_species_ratio(band);
            const double eta             = source_eta(species, layer, reference_ratio);

            for (std::size_t g_point = 0; g_point < count; ++g_point)
            {
                double absorption = optics_internal::interpolate_key_absorption(key_table,
                                                                                bands,
                                                                                pressure_count,
                                                                                ratio_count,
                                                                                band,
                                                                                g_point,
                                                                                cell,
                                                                                species,
                                                                                nullptr,
                                                                                layer,
                                                                                reference_ratio,
                                                                                false);
                if (band == 6 && !cell.lower_atmosphere)
                    absorption *= 1.6;
                if (band == 7 && cell.lower_atmosphere)
                    absorption *= 1.029;
                if (cell.lower_atmosphere && uses_lower_water_continuum(band))
                {
                    absorption += cell.water_self_factor *
                                  optics_internal::interpolate_continuum(coefficients.water_self_absorption,
                                                                         bands,
                                                                         10,
                                                                         band,
                                                                         g_point,
                                                                         cell.water_self_index,
                                                                         cell.water_self_fraction);
                    absorption += cell.water_foreign_factor *
                                  optics_internal::interpolate_continuum(coefficients.water_foreign_absorption_lower,
                                                                         bands,
                                                                         3,
                                                                         band,
                                                                         g_point,
                                                                         cell.water_foreign_index,
                                                                         cell.water_foreign_fraction);
                }
                else if (!cell.lower_atmosphere && uses_upper_water_foreign_continuum(band))
                {
                    absorption += cell.water_foreign_factor *
                                  optics_internal::interpolate_continuum(coefficients.water_foreign_absorption_upper,
                                                                         bands,
                                                                         2,
                                                                         band,
                                                                         g_point,
                                                                         0,
                                                                         cell.water_foreign_fraction);
                }
                if (band == 6)
                {
                    const double oxygen_column = optics_internal::scaled_column(atmosphere.gases.oxygen, layer);
                    absorption += 4.35e-4 * oxygen_column / 700.0;
                }
                if (band == 4 || band == 8 || band == 9 || band == 13)
                {
                    constexpr std::size_t   water_absorber          = 5;
                    constexpr std::size_t   carbon_dioxide_absorber = 6;
                    constexpr std::size_t   ozone_absorber          = 7;
                    constexpr std::size_t   methane_absorber        = 10;
                    const CoefficientArray& minor_table = cell.lower_atmosphere ? coefficients.minor_absorption_lower
                                                                                : coefficients.minor_absorption_upper;
                    std::size_t             minor_absorber = ozone_absorber;
                    const GasProfile*       minor_gas      = &atmosphere.gases.ozone;
                    if (band == 4)
                    {
                        minor_absorber = methane_absorber;
                        minor_gas      = &atmosphere.gases.methane;
                    }
                    else if (band == 13)
                    {
                        minor_absorber = cell.lower_atmosphere ? carbon_dioxide_absorber : water_absorber;
                        minor_gas =
                            cell.lower_atmosphere ? &atmosphere.gases.carbon_dioxide : &atmosphere.gases.water_vapor;
                    }
                    absorption += optics_internal::scaled_column(*minor_gas, layer) *
                                  optics_internal::minor_cross_section(
                                      minor_table, bands, band, minor_absorber, g_point, ratio_count);
                }
                const double rayleigh_coefficient =
                    optics_internal::interpolate_ratio_table(rayleigh_table, bands, ratio_count, band, g_point, eta);
                const double molecular_column = 1.0e-20 * (atmosphere.dry_air_column_molecules_cm2[layer] +
                                                           atmosphere.gases.water_vapor.column_molecules_cm2[layer]);
                const double rayleigh         = molecular_column * rayleigh_coefficient;
                // Nimbus uses the official RRTMG mean-cycle NRLSSI2 spectrum:
                // quiet-Sun irradiance plus the mean facular brightening and
                // sunspot-darkening terms. The caller's S0 scales the complete
                // spectrum in the transfer solve.
                const double source =
                    optics_internal::interpolate_ratio_table(quiet_sun_table, bands, ratio_count, band, g_point, eta) +
                    optics_internal::interpolate_ratio_table(facular_table, bands, ratio_count, band, g_point, eta) +
                    optics_internal::interpolate_ratio_table(sunspot_table, bands, ratio_count, band, g_point, eta);
                if (!std::isfinite(absorption) || absorption < 0.0)
                    throw std::runtime_error("RRTMG produced invalid shortwave absorption optical depth in band " +
                                             std::to_string(band + 16) + ", layer " + std::to_string(layer) +
                                             ", g-point " + std::to_string(g_point));
                optics_internal::require_finite_nonnegative(rayleigh, "shortwave Rayleigh optical depth");
                optics_internal::require_finite_nonnegative(source, "shortwave solar source fraction");
                const std::size_t index                = layer * result.g_points + offset + g_point;
                result.absorption_optical_depth[index] = absorption;
                result.rayleigh_optical_depth[index]   = rayleigh;
                if (layer == source_layer)
                    result.solar_source_fraction[offset + g_point] = source;
            }
        }
    }
    return result;
}

} // namespace tmv::rrtmg
