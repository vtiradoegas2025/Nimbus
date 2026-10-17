#include "rrtmg_cloud_optics.hpp"

#include "generated/rrtmg_cloud_tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace tmv::rrtmg
{
namespace
{

constexpr std::size_t liquid_radius_count = 58;
constexpr double      cloud_minimum       = 1.0e-20;

struct RadiusCell
{
    std::size_t index    = 0;
    double      fraction = 0.0;
};

RadiusCell liquid_radius_cell(double radius_um)
{
    if (!std::isfinite(radius_um) || radius_um < 2.5 || radius_um > 60.0)
        throw std::invalid_argument("RRTMG liquid effective radius must be in [2.5,60] microns");
    const double      coordinate = radius_um - 1.5;
    std::size_t       index      = static_cast<std::size_t>(coordinate);
    if (index == 0)
        index = 1;
    if (index == 58)
        index = 57;
    return {index - 1, coordinate - static_cast<double>(index)};
}

double interpolate_liquid(const std::array<double, 16 * liquid_radius_count>& table,
                          std::size_t                                        band,
                          const RadiusCell&                                  cell)
{
    const std::size_t offset = band * liquid_radius_count + cell.index;
    return table[offset] + cell.fraction * (table[offset + 1] - table[offset]);
}

double interpolate_liquid(const std::array<double, 14 * liquid_radius_count>& table,
                          std::size_t                                        band,
                          const RadiusCell&                                  cell)
{
    const std::size_t offset = band * liquid_radius_count + cell.index;
    return table[offset] + cell.fraction * (table[offset + 1] - table[offset]);
}

std::size_t ebert_curry_region(double upper_wavenumber_cm)
{
    if (upper_wavenumber_cm > 1.43e4)
        return 0;
    if (upper_wavenumber_cm > 7.7e3)
        return 1;
    if (upper_wavenumber_cm > 5.3e3)
        return 2;
    if (upper_wavenumber_cm > 4.0e3)
        return 3;
    if (upper_wavenumber_cm >= 2.5e3)
        return 4;
    throw std::invalid_argument("RRTMG shortwave band is outside the Ebert-Curry cloud regions");
}

void require_optical_property(double value, double lower, double upper, const char* name)
{
    if (!std::isfinite(value) || value < lower || value > upper)
        throw std::runtime_error(
            std::string("RRTMG produced invalid cloud ") + name + "=" +
            std::to_string(value));
}

} // namespace

LongwaveCloudOptics compute_longwave_cloud_optics(const PreparedAtmosphere& atmosphere)
{
    constexpr std::array<std::size_t, 16> ice_region{0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
    const std::size_t layers = atmosphere.layer_pressure_pa.size();
    if (atmosphere.cloud_fraction.size() != layers || atmosphere.liquid_water_path_gm2.size() != layers ||
        atmosphere.ice_water_path_gm2.size() != layers || atmosphere.liquid_effective_radius_um.size() != layers ||
        atmosphere.ice_effective_radius_um.size() != layers)
        throw std::invalid_argument("RRTMG longwave cloud inputs have inconsistent shapes");

    LongwaveCloudOptics result;
    result.layers = layers;
    result.absorption_optical_depth.assign(layers * 16, 0.0);
    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        if (atmosphere.cloud_fraction[layer] == 0.0)
            continue;
        const double liquid_path = atmosphere.liquid_water_path_gm2[layer];
        const double ice_path    = atmosphere.ice_water_path_gm2[layer];
        const double ice_radius  = atmosphere.ice_effective_radius_um[layer];
        if (!std::isfinite(ice_radius) || ice_radius < 13.0 || ice_radius > 130.0)
            throw std::invalid_argument("RRTMG ice effective radius must be in [13,130] microns");
        const RadiusCell liquid_cell = liquid_radius_cell(atmosphere.liquid_effective_radius_um[layer]);
        for (std::size_t band = 0; band < 16; ++band)
        {
            const std::size_t region = ice_region[band];
            const double ice_absorption = cloud_tables::longwave_ice_intercept[region] +
                                          cloud_tables::longwave_ice_inverse_radius[region] / ice_radius;
            const double liquid_absorption =
                interpolate_liquid(cloud_tables::longwave_liquid_absorption, band, liquid_cell);
            const double optical_depth = liquid_path * liquid_absorption + ice_path * ice_absorption;
            require_optical_property(optical_depth, 0.0, std::numeric_limits<double>::max(), "longwave optical depth");
            result.absorption_optical_depth[layer * 16 + band] = optical_depth;
        }
    }
    return result;
}

ShortwaveCloudOptics compute_shortwave_cloud_optics(const ShortwaveCoefficients& coefficients,
                                                     const PreparedAtmosphere&   atmosphere)
{
    constexpr std::size_t bands  = 14;
    const std::size_t     layers = atmosphere.layer_pressure_pa.size();
    if (coefficients.grid.band_upper_wavenumber_cm.size() != bands || atmosphere.cloud_fraction.size() != layers ||
        atmosphere.liquid_water_path_gm2.size() != layers || atmosphere.ice_water_path_gm2.size() != layers ||
        atmosphere.liquid_effective_radius_um.size() != layers || atmosphere.ice_effective_radius_um.size() != layers)
        throw std::invalid_argument("RRTMG shortwave cloud inputs have inconsistent shapes");

    ShortwaveCloudOptics result;
    result.layers = layers;
    result.original_optical_depth.assign(layers * bands, 0.0);
    result.optical_depth.assign(layers * bands, 0.0);
    result.single_scattering_albedo.assign(layers * bands, 0.0);
    result.asymmetry.assign(layers * bands, 0.0);
    for (std::size_t layer = 0; layer < layers; ++layer)
    {
        if (atmosphere.cloud_fraction[layer] == 0.0)
            continue;
        const double liquid_path = atmosphere.liquid_water_path_gm2[layer];
        const double ice_path    = atmosphere.ice_water_path_gm2[layer];
        const double ice_radius  = atmosphere.ice_effective_radius_um[layer];
        if (!std::isfinite(ice_radius) || ice_radius < 13.0 || ice_radius > 130.0)
            throw std::invalid_argument("RRTMG ice effective radius must be in [13,130] microns");
        const RadiusCell liquid_cell = liquid_radius_cell(atmosphere.liquid_effective_radius_um[layer]);
        for (std::size_t band = 0; band < bands; ++band)
        {
            const std::size_t region = ebert_curry_region(coefficients.grid.band_upper_wavenumber_cm[band]);
            const double liquid_extinction =
                interpolate_liquid(cloud_tables::shortwave_liquid_extinction, band, liquid_cell);
            double liquid_ssa =
                interpolate_liquid(cloud_tables::shortwave_liquid_single_scattering_albedo, band, liquid_cell);
            const double liquid_g =
                interpolate_liquid(cloud_tables::shortwave_liquid_asymmetry, band, liquid_cell);
            const double ice_extinction = cloud_tables::shortwave_ice_extinction_intercept[region] +
                                          cloud_tables::shortwave_ice_extinction_inverse_radius[region] / ice_radius;
            const double ice_ssa = 1.0 - cloud_tables::shortwave_ice_absorption_intercept[region] -
                                   cloud_tables::shortwave_ice_absorption_radius[region] * ice_radius;
            const double ice_g = std::min(1.0 - 1.0e-6,
                                          cloud_tables::shortwave_ice_asymmetry_intercept[region] +
                                              cloud_tables::shortwave_ice_asymmetry_radius[region] * ice_radius);
            if (liquid_cell.fraction < 0.0 && liquid_ssa > 1.0)
                liquid_ssa = cloud_tables::shortwave_liquid_single_scattering_albedo
                    [band * liquid_radius_count + liquid_cell.index];
            require_optical_property(liquid_extinction, 0.0, std::numeric_limits<double>::max(), "liquid extinction");
            require_optical_property(liquid_ssa, 0.0, 1.0, "liquid single-scattering albedo");
            require_optical_property(liquid_g, 0.0, 1.0, "liquid asymmetry");
            require_optical_property(ice_extinction, 0.0, std::numeric_limits<double>::max(), "ice extinction");
            require_optical_property(ice_ssa, 0.0, 1.0, "ice single-scattering albedo");
            require_optical_property(ice_g, 0.0, 1.0, "ice asymmetry");

            const double liquid_original = liquid_path * liquid_extinction;
            const double ice_original    = ice_path * ice_extinction;
            const double liquid_forward  = liquid_g * liquid_g;
            const double ice_forward     = ice_g * ice_g;
            const double liquid_scale    = 1.0 - liquid_forward * liquid_ssa;
            const double ice_scale       = 1.0 - ice_forward * ice_ssa;
            const double liquid_tau      = liquid_scale * liquid_original;
            const double ice_tau         = ice_scale * ice_original;
            const double liquid_scaled_ssa = liquid_ssa * (1.0 - liquid_forward) / liquid_scale;
            const double ice_scaled_ssa    = ice_ssa * (1.0 - ice_forward) / ice_scale;
            const double liquid_scattering = liquid_scaled_ssa * liquid_tau;
            const double ice_scattering    = ice_scaled_ssa * ice_tau;
            const double total_tau          = liquid_tau + ice_tau;
            const double total_scattering = liquid_scattering + ice_scattering;
            const std::size_t index       = layer * bands + band;
            result.original_optical_depth[index] = liquid_original + ice_original;
            if (total_tau <= cloud_minimum || total_scattering <= cloud_minimum)
            {
                result.optical_depth[index] = 0.0;
                result.single_scattering_albedo[index] = 0.0;
                result.asymmetry[index] = 0.0;
                continue;
            }
            result.optical_depth[index] = total_tau;
            result.single_scattering_albedo[index] =
                total_scattering / total_tau;
            result.asymmetry[index] =
                (liquid_scattering * (liquid_g - liquid_forward) /
                     (1.0 - liquid_forward) +
                 ice_scattering * (ice_g - ice_forward) /
                     (1.0 - ice_forward)) / total_scattering;
            require_optical_property(result.optical_depth[index], 0.0, std::numeric_limits<double>::max(),
                                     "shortwave optical depth");
            require_optical_property(result.single_scattering_albedo[index], 0.0, 1.0,
                                     "shortwave single-scattering albedo");
            require_optical_property(result.asymmetry[index], 0.0, 1.0, "shortwave asymmetry");
        }
    }
    return result;
}

} // namespace tmv::rrtmg
