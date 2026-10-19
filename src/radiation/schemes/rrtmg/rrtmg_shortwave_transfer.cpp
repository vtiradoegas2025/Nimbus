#include "rrtmg_transfer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv::rrtmg
{
namespace
{

struct LayerTransfer
{
    double direct_reflectance    = 0.0;
    double diffuse_reflectance   = 0.0;
    double direct_transmittance  = 1.0;
    double diffuse_transmittance = 1.0;
    double direct_beam           = 1.0;
};

double exponential_transmission(double optical_depth)
{
    // Native transfer kernels use float arithmetic. Capping the exponent at
    // 40 keeps an optically opaque but non-zero transmission on every path,
    // avoiding 0 * infinity in the analytically equivalent PIFM terms.
    const double bounded = std::min(optical_depth, 40.0);
    if (bounded <= 1.0e-4)
        return 1.0 - bounded + 0.5 * bounded * bounded;
    return std::exp(-bounded);
}

LayerTransfer pifm_two_stream(double absorption, double scattering, double asymmetry, double cosine_zenith)
{
    const double optical_depth = absorption + scattering;
    if (optical_depth <= 0.0)
        return {};

    constexpr double epsilon            = 1.0e-8;
    constexpr double conservative_limit = 0.9999995;
    const double     omega              = std::clamp(scattering / optical_depth, 0.0, 1.0);
    const double     triple_asymmetry   = 3.0 * asymmetry;
    const double     gamma1             = (8.0 - omega * (5.0 + triple_asymmetry)) * 0.25;
    const double     gamma2             = 3.0 * omega * (1.0 - asymmetry) * 0.25;
    const double     gamma3             = (2.0 - triple_asymmetry * cosine_zenith) * 0.25;
    const double     gamma4             = 1.0 - gamma3;
    const double     direct_beam        = exponential_transmission(optical_depth / cosine_zenith);

    LayerTransfer result;
    result.direct_beam = direct_beam;
    if (omega >= conservative_limit)
    {
        const double a               = gamma1 * cosine_zenith;
        const double a1              = a - gamma3;
        const double gamma_tau       = gamma1 * optical_depth;
        result.direct_reflectance    = (gamma_tau - a1 * (1.0 - direct_beam)) / (1.0 + gamma_tau);
        result.direct_transmittance  = 1.0 - result.direct_reflectance;
        result.diffuse_reflectance   = gamma_tau / (1.0 + gamma_tau);
        result.diffuse_transmittance = 1.0 - result.diffuse_reflectance;
        if (direct_beam == 1.0)
            return {};
        return result;
    }

    const double a1            = gamma1 * gamma4 + gamma2 * gamma3;
    const double a2            = gamma1 * gamma3 + gamma2 * gamma4;
    const double k             = std::sqrt(std::max(0.0, gamma1 * gamma1 - gamma2 * gamma2));
    const double kp            = k * cosine_zenith;
    const double kp1           = 1.0 + kp;
    const double km1           = 1.0 - kp;
    const double two_k         = 2.0 * k;
    const double one_kp2       = 1.0 - kp * kp;
    const double kg            = k + gamma1;
    const double r1            = km1 * (a2 + k * gamma3);
    const double r2            = kp1 * (a2 - k * gamma3);
    const double r3            = two_k * (gamma3 - a2 * cosine_zenith);
    const double r4            = one_kp2 * kg;
    const double r5            = one_kp2 * (k - gamma1);
    const double t1            = kp1 * (a1 + k * gamma4);
    const double t2            = km1 * (a1 - k * gamma4);
    const double t3            = two_k * (gamma4 + a1 * cosine_zenith);
    const double beta          = (gamma1 - k) / kg;
    const double exp_k         = exponential_transmission(k * optical_depth);
    const double inv_exp_k     = 1.0 / exp_k;
    const double denominator_r = r4 * inv_exp_k + r5 * exp_k;
    const double denominator_t = r4 * inv_exp_k + r5 * exp_k;
    if (std::abs(denominator_r) <= epsilon)
    {
        result.direct_reflectance   = epsilon;
        result.direct_transmittance = direct_beam;
    }
    else
    {
        result.direct_reflectance = omega * (r1 * inv_exp_k - r2 * exp_k - r3 * direct_beam) / denominator_r;
        result.direct_transmittance =
            direct_beam - omega * (direct_beam * (t1 * inv_exp_k - t2 * exp_k) - t3) / denominator_t;
    }
    const double exp_k_squared       = exp_k * exp_k;
    const double diffuse_denominator = 1.0 / ((1.0 - beta * exp_k_squared) * kg);
    result.diffuse_reflectance       = gamma2 * (1.0 - exp_k_squared) * diffuse_denominator;
    result.diffuse_transmittance     = two_k * exp_k * diffuse_denominator;
    return result;
}

} // namespace

SpectralFluxes solve_shortwave(const ShortwaveCoefficients& coefficients,
                               const ShortwaveOptics&       optics,
                               double                       cosine_zenith,
                               double                       solar_irradiance_wm2,
                               double                       surface_albedo,
                               const ShortwaveCloudOptics*  clouds)
{
    if (!std::isfinite(cosine_zenith) || cosine_zenith < 0.0 || cosine_zenith > 1.0 ||
        !std::isfinite(solar_irradiance_wm2) || solar_irradiance_wm2 < 0.0 || !std::isfinite(surface_albedo) ||
        surface_albedo < 0.0 || surface_albedo > 1.0 || optics.layers == 0 || optics.g_points != 112 ||
        optics.absorption_optical_depth.size() != optics.layers * 112 ||
        optics.rayleigh_optical_depth.size() != optics.layers * 112 || optics.solar_source_fraction.size() != 112 ||
        coefficients.grid.reduced_g_point_offsets.size() != 15 ||
        (clouds != nullptr && (clouds->layers != optics.layers || clouds->optical_depth.size() != optics.layers * 14 ||
                               clouds->single_scattering_albedo.size() != optics.layers * 14 ||
                               clouds->asymmetry.size() != optics.layers * 14)))
        throw std::invalid_argument("RRTMG shortwave transfer shapes are invalid");

    SpectralFluxes result;
    result.upward_wm2.assign(optics.layers + 1, 0.0);
    result.downward_wm2.assign(optics.layers + 1, 0.0);
    if (cosine_zenith == 0.0 || solar_irradiance_wm2 == 0.0)
        return result;

    double source_sum = 0.0;
    for (std::size_t g_point = 0; g_point < optics.g_points; ++g_point)
        source_sum += optics.solar_source_fraction[g_point];
    if (!std::isfinite(source_sum) || source_sum <= 0.0)
        throw std::runtime_error("RRTMG solar source normalization is invalid");

    std::vector<LayerTransfer> transfer(optics.layers);
    std::vector<double>        cumulative_direct(optics.layers + 1);
    std::vector<double>        downward_total(optics.layers + 1);
    std::vector<double>        downward_diffuse_reflectance(optics.layers + 1);
    std::vector<double>        upward_direct_reflectance(optics.layers + 1);
    std::vector<double>        upward_diffuse_reflectance(optics.layers + 1);
    std::vector<double>        downward(optics.layers + 1);
    std::vector<double>        upward(optics.layers + 1);
    for (std::size_t g_point = 0; g_point < optics.g_points; ++g_point)
    {
        const auto        upper = std::upper_bound(coefficients.grid.reduced_g_point_offsets.begin(),
                                                   coefficients.grid.reduced_g_point_offsets.end(),
                                                   g_point);
        const std::size_t band =
            static_cast<std::size_t>(upper - coefficients.grid.reduced_g_point_offsets.begin() - 1);
        // The RRTMG two-stream quadrature proceeds top to bottom. Nimbus
        // atmospheric storage is bottom to top, so reverse only at this
        // narrow transfer boundary.
        cumulative_direct[0] = 1.0;
        for (std::size_t top_layer = 0; top_layer < optics.layers; ++top_layer)
        {
            const std::size_t layer       = optics.layers - 1 - top_layer;
            const std::size_t index       = layer * optics.g_points + g_point;
            const std::size_t cloud_index = layer * 14 + band;
            const double      cloud_depth = clouds == nullptr ? 0.0 : clouds->optical_depth[cloud_index];
            const double      cloud_scattering =
                clouds == nullptr ? 0.0 : cloud_depth * clouds->single_scattering_albedo[cloud_index];
            const double rayleigh   = optics.rayleigh_optical_depth[index];
            const double scattering = rayleigh + cloud_scattering;
            const double asymmetry  = scattering > 0.0 && clouds != nullptr
                                          ? cloud_scattering * clouds->asymmetry[cloud_index] / scattering
                                          : 0.0;
            transfer[top_layer] =
                pifm_two_stream(optics.absorption_optical_depth[index] + cloud_depth - cloud_scattering,
                                scattering,
                                asymmetry,
                                cosine_zenith);
            cumulative_direct[top_layer + 1] = cumulative_direct[top_layer] * transfer[top_layer].direct_beam;
        }

        upward_direct_reflectance.back()  = surface_albedo;
        upward_diffuse_reflectance.back() = surface_albedo;
        const std::size_t lowest          = optics.layers - 1;
        double            coupling        = 1.0 / (1.0 - surface_albedo * transfer[lowest].diffuse_reflectance);
        upward_direct_reflectance[lowest] =
            transfer[lowest].direct_reflectance +
            transfer[lowest].diffuse_transmittance *
                ((transfer[lowest].direct_transmittance - transfer[lowest].direct_beam) * surface_albedo +
                 transfer[lowest].direct_beam * surface_albedo) *
                coupling;
        upward_diffuse_reflectance[lowest] =
            transfer[lowest].diffuse_reflectance +
            transfer[lowest].diffuse_transmittance * transfer[lowest].diffuse_transmittance * surface_albedo * coupling;
        for (std::size_t top_layer = lowest; top_layer-- > 0;)
        {
            coupling =
                1.0 / (1.0 - upward_diffuse_reflectance[top_layer + 1] * transfer[top_layer].diffuse_reflectance);
            upward_direct_reflectance[top_layer] =
                transfer[top_layer].direct_reflectance +
                transfer[top_layer].diffuse_transmittance *
                    ((transfer[top_layer].direct_transmittance - transfer[top_layer].direct_beam) *
                         upward_diffuse_reflectance[top_layer + 1] +
                     transfer[top_layer].direct_beam * upward_direct_reflectance[top_layer + 1]) *
                    coupling;
            upward_diffuse_reflectance[top_layer] =
                transfer[top_layer].diffuse_reflectance + transfer[top_layer].diffuse_transmittance *
                                                              transfer[top_layer].diffuse_transmittance *
                                                              upward_diffuse_reflectance[top_layer + 1] * coupling;
        }

        downward_total[0]               = 1.0;
        downward_diffuse_reflectance[0] = 0.0;
        if (optics.layers > 0)
        {
            downward_total[1]               = transfer[0].direct_transmittance;
            downward_diffuse_reflectance[1] = transfer[0].diffuse_reflectance;
        }
        for (std::size_t top_layer = 1; top_layer < optics.layers; ++top_layer)
        {
            coupling = 1.0 / (1.0 - transfer[top_layer].diffuse_reflectance * downward_diffuse_reflectance[top_layer]);
            downward_total[top_layer + 1] = cumulative_direct[top_layer] * transfer[top_layer].direct_transmittance +
                                            transfer[top_layer].diffuse_transmittance *
                                                ((downward_total[top_layer] - cumulative_direct[top_layer]) +
                                                 cumulative_direct[top_layer] * transfer[top_layer].direct_reflectance *
                                                     downward_diffuse_reflectance[top_layer]) *
                                                coupling;
            downward_diffuse_reflectance[top_layer + 1] =
                transfer[top_layer].diffuse_reflectance + transfer[top_layer].diffuse_transmittance *
                                                              transfer[top_layer].diffuse_transmittance *
                                                              downward_diffuse_reflectance[top_layer] * coupling;
        }

        const double incident =
            solar_irradiance_wm2 * cosine_zenith * optics.solar_source_fraction[g_point] / source_sum;
        for (std::size_t top_interface = 0; top_interface <= optics.layers; ++top_interface)
        {
            coupling =
                1.0 / (1.0 - downward_diffuse_reflectance[top_interface] * upward_diffuse_reflectance[top_interface]);
            upward[top_interface]       = (cumulative_direct[top_interface] * upward_direct_reflectance[top_interface] +
                                           (downward_total[top_interface] - cumulative_direct[top_interface]) *
                                               upward_diffuse_reflectance[top_interface]) *
                                          coupling;
            downward[top_interface]     = cumulative_direct[top_interface] +
                                          ((downward_total[top_interface] - cumulative_direct[top_interface]) +
                                           cumulative_direct[top_interface] * upward_direct_reflectance[top_interface] *
                                               downward_diffuse_reflectance[top_interface]) *
                                              coupling;
            const std::size_t interface = optics.layers - top_interface;
            result.upward_wm2[interface] += incident * upward[top_interface];
            result.downward_wm2[interface] += incident * downward[top_interface];
        }
    }
    for (std::size_t interface = 0; interface <= optics.layers; ++interface)
    {
        if (!std::isfinite(result.upward_wm2[interface]) || result.upward_wm2[interface] < 0.0 ||
            !std::isfinite(result.downward_wm2[interface]) || result.downward_wm2[interface] < 0.0)
            throw std::runtime_error("RRTMG shortwave transfer produced invalid flux");
    }
    return result;
}

} // namespace tmv::rrtmg
