#include "rrtmg_transfer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tmv::rrtmg
{
namespace
{

double diffusivity_secant(std::size_t band, double precipitable_water_cm)
{
    constexpr std::array<double, 16> a0{
        1.66,
        1.55,
        1.58,
        1.66,
        1.54,
        1.454,
        1.89,
        1.33,
        1.668,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
        1.66,
    };
    constexpr std::array<double, 16> a1{
        0.0,
        0.25,
        0.22,
        0.0,
        0.13,
        0.446,
        -0.10,
        0.40,
        -0.006,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    constexpr std::array<double, 16> a2{
        0.0,
        -12.0,
        -11.7,
        0.0,
        -0.72,
        -0.243,
        0.19,
        -0.062,
        0.414,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    };
    if (band == 0 || band == 3 || band >= 9)
        return 1.66;
    return std::clamp(a0[band] + a1[band] * std::exp(a2[band] * precipitable_water_cm), 1.50, 1.80);
}

struct LayerTransmission
{
    double absorptivity  = 0.0;
    double source_factor = 0.0;
};

LayerTransmission clear_layer_transmission(double optical_depth)
{
    if (!std::isfinite(optical_depth) || optical_depth < 0.0)
        throw std::runtime_error("RRTMG longwave transfer received invalid optical depth");
    if (optical_depth <= 0.06)
        return {optical_depth - 0.5 * optical_depth * optical_depth, optical_depth / 6.0};

    const double transmission  = std::exp(-optical_depth);
    const double source_factor = 1.0 - 2.0 * (1.0 / optical_depth - transmission / (1.0 - transmission));
    return {1.0 - transmission, source_factor};
}

double cloudy_lookup_optical_depth(double optical_depth)
{
    constexpr double pade_inverse = 1.0 / 0.278;
    constexpr double table_scale  = 10000.0;
    const double     coordinate   = optical_depth / (pade_inverse + optical_depth);
    const auto       index        = static_cast<std::size_t>(table_scale * coordinate + 0.5);
    if (index >= static_cast<std::size_t>(table_scale))
        return 1.0e10;
    const double fraction = static_cast<double>(index) / table_scale;
    return pade_inverse * fraction / (1.0 - fraction);
}

LayerTransmission cloudy_layer_transmission(double gas_optical_depth, double cloud_optical_depth)
{
    if (!std::isfinite(gas_optical_depth) || gas_optical_depth < 0.0 ||
        !std::isfinite(cloud_optical_depth) || cloud_optical_depth < 0.0)
        throw std::runtime_error("RRTMG longwave transfer received invalid cloudy optical depth");
    if (gas_optical_depth > 0.06)
        gas_optical_depth = cloudy_lookup_optical_depth(gas_optical_depth);
    double total_optical_depth = gas_optical_depth + cloud_optical_depth;
    if (total_optical_depth > 0.06)
        total_optical_depth = cloudy_lookup_optical_depth(total_optical_depth);
    return clear_layer_transmission(total_optical_depth);
}

} // namespace

SpectralFluxes solve_longwave(const LongwaveCoefficients& coefficients,
                              const PreparedAtmosphere&   atmosphere,
                              const LongwavePreparation&  preparation,
                              const LongwaveOptics&       optics,
                              double                      surface_emissivity,
                              const LongwaveCloudOptics*  clouds)
{
    constexpr double  radiance_to_flux = 3.14159265358979323846e4;
    const std::size_t layers           = atmosphere.layer_pressure_pa.size();
    if (!std::isfinite(surface_emissivity) || surface_emissivity < 0.0 || surface_emissivity > 1.0 ||
        optics.layers != layers || optics.g_points != 140 || optics.absorption_optical_depth.size() != layers * 140 ||
        optics.planck_fraction.size() != layers * 140 || preparation.layer_planck.size() != layers * 16 ||
        preparation.interface_planck.size() != (layers + 1) * 16 || preparation.surface_planck.size() != 16 ||
        (clouds != nullptr &&
         (clouds->layers != layers || clouds->absorption_optical_depth.size() != layers * 16)))
        throw std::invalid_argument("RRTMG longwave transfer shapes are invalid");

    SpectralFluxes result;
    result.upward_wm2.assign(layers + 1, 0.0);
    result.downward_wm2.assign(layers + 1, 0.0);
    std::vector<double> downward(layers + 1, 0.0);
    std::vector<double> upward(layers + 1, 0.0);

    for (std::size_t band = 0; band < 16; ++band)
    {
        const double secant = diffusivity_secant(band, preparation.precipitable_water_cm);
        const double band_width =
            coefficients.grid.band_upper_wavenumber_cm[band] - coefficients.grid.band_lower_wavenumber_cm[band];
        const std::size_t begin = coefficients.grid.reduced_g_point_offsets[band];
        const std::size_t end   = coefficients.grid.reduced_g_point_offsets[band + 1];
        for (std::size_t g_point = begin; g_point < end; ++g_point)
        {
            downward.back() = 0.0;
            for (std::size_t layer = layers; layer-- > 0;)
            {
                const std::size_t index    = layer * optics.g_points + g_point;
                const double cloud_depth = clouds == nullptr ? 0.0 : clouds->absorption_optical_depth[layer * 16 + band];
                const double gas_depth = secant * optics.absorption_optical_depth[index];
                const auto   transfer  = cloud_depth > 0.0
                                             ? cloudy_layer_transmission(gas_depth, secant * cloud_depth)
                                             : clear_layer_transmission(gas_depth);
                const double      layer_planck     = preparation.layer_planck[layer * 16 + band];
                const double      lower_difference = preparation.interface_planck[layer * 16 + band] - layer_planck;
                const double      source =
                    optics.planck_fraction[index] * (layer_planck + transfer.source_factor * lower_difference);
                downward[layer] = downward[layer + 1] + (source - downward[layer + 1]) * transfer.absorptivity;
            }
            const double surface_fraction = optics.planck_fraction[g_point];
            upward.front() =
                preparation.surface_planck[band] * surface_fraction + (1.0 - surface_emissivity) * downward.front();
            for (std::size_t layer = 0; layer < layers; ++layer)
            {
                const std::size_t index    = layer * optics.g_points + g_point;
                const double cloud_depth = clouds == nullptr ? 0.0 : clouds->absorption_optical_depth[layer * 16 + band];
                const double gas_depth = secant * optics.absorption_optical_depth[index];
                const auto   transfer  = cloud_depth > 0.0
                                             ? cloudy_layer_transmission(gas_depth, secant * cloud_depth)
                                             : clear_layer_transmission(gas_depth);
                const double      layer_planck = preparation.layer_planck[layer * 16 + band];
                const double upper_difference  = preparation.interface_planck[(layer + 1) * 16 + band] - layer_planck;
                const double source =
                    optics.planck_fraction[index] * (layer_planck + transfer.source_factor * upper_difference);
                upward[layer + 1] = upward[layer] + (source - upward[layer]) * transfer.absorptivity;
            }
            const double scale = radiance_to_flux * band_width;
            for (std::size_t interface = 0; interface <= layers; ++interface)
            {
                result.upward_wm2[interface] += scale * upward[interface];
                result.downward_wm2[interface] += scale * downward[interface];
            }
        }
    }
    for (std::size_t interface = 0; interface <= layers; ++interface)
    {
        if (!std::isfinite(result.upward_wm2[interface]) || result.upward_wm2[interface] < 0.0 ||
            !std::isfinite(result.downward_wm2[interface]) || result.downward_wm2[interface] < 0.0)
            throw std::runtime_error("RRTMG longwave transfer produced invalid flux");
    }
    return result;
}

} // namespace tmv::rrtmg
