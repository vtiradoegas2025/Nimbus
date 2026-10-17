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

using optics_internal::KeySpecies;

struct BandSpecies
{
    const char* lower_first;
    const char* lower_second;
    const char* upper_first;
    const char* upper_second;
};

constexpr std::array<BandSpecies, 16> band_species{{
    {"H2O", "", "H2O", ""},
    {"H2O", "", "H2O", ""},
    {"H2O", "CO2", "H2O", "CO2"},
    {"H2O", "CO2", "O3", "CO2"},
    {"H2O", "CO2", "O3", "CO2"},
    {"H2O", "", "", ""},
    {"H2O", "O3", "O3", ""},
    {"H2O", "", "O3", ""},
    {"H2O", "CH4", "CH4", ""},
    {"H2O", "", "H2O", ""},
    {"H2O", "", "H2O", ""},
    {"H2O", "CO2", "", ""},
    {"H2O", "N2O", "", ""},
    {"CO2", "", "CO2", ""},
    {"N2O", "CO2", "", ""},
    {"H2O", "CH4", "CH4", ""},
}};

std::size_t planck_mapping_pressure(std::size_t band, bool lower_atmosphere)
{
    constexpr std::size_t                                unused = 59;
    constexpr std::array<std::array<std::size_t, 2>, 16> pressure{{
        {{unused, unused}},
        {{unused, unused}},
        {{8, 12}},
        {{10, 12}},
        {{4, 42}},
        {{unused, unused}},
        {{2, unused}},
        {{unused, unused}},
        {{8, unused}},
        {{unused, unused}},
        {{unused, unused}},
        {{9, unused}},
        {{4, unused}},
        {{unused, unused}},
        {{0, unused}},
        {{5, unused}},
    }};
    return pressure[band][lower_atmosphere ? 0 : 1];
}

double planck_eta(const LongwaveCoefficients& coefficients,
                  const KeySpecies&           species,
                  std::size_t                 band,
                  bool                        lower_atmosphere,
                  std::size_t                 layer)
{
    if (species.first == nullptr || species.second == nullptr)
        return 0.0;
    constexpr std::size_t absorber_count = 12;
    const std::size_t     pressure       = planck_mapping_pressure(band, lower_atmosphere);
    if (pressure >= 59 || coefficients.absorber_amount_mls.shape != std::vector<std::size_t>{59, absorber_count})
        throw std::invalid_argument("RRTMG Planck-fraction mapping is unavailable");
    const double second_reference =
        coefficients.absorber_amount_mls.values[pressure * absorber_count + species.second_absorber];
    if (!(second_reference > 0.0))
        throw std::invalid_argument("RRTMG Planck-fraction reference ratio is invalid");
    const double reference_ratio =
        coefficients.absorber_amount_mls.values[pressure * absorber_count + species.first_absorber] / second_reference;
    const double first  = optics_internal::scaled_column(*species.first, layer);
    const double second = reference_ratio * optics_internal::scaled_column(*species.second, layer);
    return first + second > 0.0 ? first / (first + second) : 0.0;
}

double optical_depth_correction(std::size_t band, std::size_t local_g_point, bool lower_atmosphere, double pressure_hpa)
{
    if (band == 0)
    {
        if (lower_atmosphere)
            return pressure_hpa < 250.0 ? 1.0 - 0.15 * (250.0 - pressure_hpa) / 154.4 : 1.0;
        return 1.0 - 0.15 * pressure_hpa / 95.6;
    }
    if (band == 1 && lower_atmosphere)
        return 1.0 - 0.05 * (pressure_hpa - 100.0) / 900.0;
    if (band == 3 && !lower_atmosphere)
    {
        constexpr std::array<double, 7> corrections{0.92, 0.88, 1.07, 1.10, 0.99, 0.88, 0.943};
        if (local_g_point >= 7 && local_g_point < 14)
            return corrections[local_g_point - 7];
    }
    if (band == 6 && !lower_atmosphere)
    {
        constexpr std::array<double, 6> corrections{0.92, 0.88, 1.07, 1.10, 0.99, 0.855};
        if (local_g_point >= 5 && local_g_point < 11)
            return corrections[local_g_point - 5];
    }
    return 1.0;
}

} // namespace

LongwaveOptics compute_longwave_gas_optics(const LongwaveCoefficients& coefficients,
                                           const PreparedAtmosphere&   atmosphere,
                                           const LongwavePreparation&  preparation)
{
    constexpr std::size_t bands  = 16;
    const std::size_t     layers = atmosphere.layer_pressure_pa.size();
    if (preparation.interpolation.size() != layers || coefficients.grid.reduced_g_point_offsets.size() != bands + 1 ||
        coefficients.grid.reduced_g_point_offsets.back() != 140)
        throw std::invalid_argument("RRTMG longwave optics shapes are invalid");

    LongwaveOptics result;
    result.layers   = layers;
    result.g_points = 140;
    result.absorption_optical_depth.assign(layers * result.g_points, 0.0);
    result.planck_fraction.assign(layers * result.g_points, 0.0);

    for (std::size_t band = 0; band < bands; ++band)
    {
        const std::size_t count  = static_cast<std::size_t>(coefficients.grid.reduced_g_points_per_band[band]);
        const std::size_t offset = coefficients.grid.reduced_g_point_offsets[band];
        for (std::size_t layer = 0; layer < layers; ++layer)
        {
            const auto&        interpolation = preparation.interpolation[layer];
            const BandSpecies& names         = band_species[band];
            const KeySpecies   species =
                optics_internal::key_species(coefficients.grid,
                                             atmosphere,
                                             interpolation.lower_atmosphere ? names.lower_first : names.upper_first,
                                             interpolation.lower_atmosphere ? names.lower_second : names.upper_second);
            const std::size_t       pressure_count = interpolation.lower_atmosphere ? 13 : 47;
            const std::size_t       ratio_count    = interpolation.lower_atmosphere ? 9 : 5;
            const CoefficientArray& key_table =
                interpolation.lower_atmosphere ? coefficients.key_absorption_lower : coefficients.key_absorption_upper;
            const CoefficientArray& fraction_table = interpolation.lower_atmosphere
                                                         ? coefficients.planck_fraction_lower
                                                         : coefficients.planck_fraction_upper;

            for (std::size_t g_point = 0; g_point < count; ++g_point)
            {
                const double key_optical_depth = optics_internal::interpolate_key_absorption(key_table,
                                                                                   bands,
                                                                                   pressure_count,
                                                                                   ratio_count,
                                                                                   band,
                                                                                   g_point,
                                                                                   interpolation,
                                                                                   species,
                                                                                   &coefficients.absorber_amount_mls,
                                                                                   layer);
                double self_continuum = 0.0;
                if (interpolation.lower_atmosphere)
                {
                    self_continuum = interpolation.water_self_factor *
                        optics_internal::interpolate_continuum(
                            coefficients.water_self_absorption,
                            bands, 10, band, g_point,
                            interpolation.water_self_index,
                            interpolation.water_self_fraction);
                }
                const double foreign_continuum =
                    interpolation.water_foreign_factor *
                    optics_internal::interpolate_continuum(
                        coefficients.water_foreign_absorption,
                        bands, 4, band, g_point,
                        interpolation.water_foreign_index,
                        interpolation.water_foreign_fraction);
                const double minor_optical_depth =
                    optics_internal::longwave_minor_optical_depth(
                    coefficients, atmosphere, interpolation, band, g_point, layer);
                const double correction = optical_depth_correction(
                    band, g_point, interpolation.lower_atmosphere, atmosphere.layer_pressure_pa[layer] * 0.01);
                const double optical_depth =
                    (key_optical_depth + self_continuum + foreign_continuum
                     + minor_optical_depth) * correction;
                const double fraction = optics_internal::interpolate_ratio_table(
                    fraction_table,
                    bands,
                    ratio_count,
                    band,
                    g_point,
                    planck_eta(coefficients, species, band, interpolation.lower_atmosphere, layer));
                if (!std::isfinite(optical_depth) || optical_depth < 0.0)
                    throw std::runtime_error(
                        "RRTMG produced invalid longwave optical depth at band "
                        + std::to_string(band) + ", g-point "
                        + std::to_string(g_point) + ", layer "
                        + std::to_string(layer) + ", value "
                        + std::to_string(optical_depth) + ", pressure_pa "
                        + std::to_string(
                            atmosphere.layer_pressure_pa[layer])
                        + ", temperature_k "
                        + std::to_string(
                            atmosphere.layer_temperature_k[layer])
                        + ", lower_atmosphere "
                        + (interpolation.lower_atmosphere ? "true" : "false")
                        + ", key " + std::to_string(key_optical_depth)
                        + ", self " + std::to_string(self_continuum)
                        + ", foreign " + std::to_string(foreign_continuum)
                        + ", minor " + std::to_string(minor_optical_depth)
                        + ", correction " + std::to_string(correction)
                        + ", pressure_index "
                        + std::to_string(interpolation.pressure_index)
                        + ", temperature_indices "
                        + std::to_string(
                            interpolation.temperature_index_at_lower_pressure)
                        + "/"
                        + std::to_string(
                            interpolation.temperature_index_at_upper_pressure)
                        + ", weights "
                        + std::to_string(
                            interpolation.pressure_temperature_weights[0])
                        + "/"
                        + std::to_string(
                            interpolation.pressure_temperature_weights[1])
                        + "/"
                        + std::to_string(
                            interpolation.pressure_temperature_weights[2])
                        + "/"
                        + std::to_string(
                            interpolation.pressure_temperature_weights[3]));
                optics_internal::require_finite_nonnegative(fraction, "longwave Planck fraction");
                const std::size_t index                = layer * result.g_points + offset + g_point;
                result.absorption_optical_depth[index] = optical_depth;
                result.planck_fraction[index]          = fraction;
            }
        }
    }
    return result;
}

} // namespace tmv::rrtmg
