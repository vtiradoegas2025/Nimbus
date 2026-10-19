#include "microphysics/schemes/thompson/warm_rain_processes.hpp"

#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double water_density = 1000.0;
constexpr double rain_mass_coefficient = pi * water_density / 6.0;
constexpr double minimum_cloud_diameter = 1.0e-6;
constexpr double minimum_rain_diameter = 50.0e-6;
constexpr double rain_velocity_coefficient = 4854.0;
constexpr double rain_velocity_decay = 195.0;
constexpr double reference_air_density = 101325.0 / (287.05 * 298.0);
constexpr double water_vapor_gas_constant = 461.5;
constexpr double dry_heat_capacity = 1004.0;
constexpr double latent_heat_at_freezing = 2.5e6;
constexpr double schmidt_number = 0.632;

void require_nonnegative(double value, const char* message)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(message);
}

void require_positive(double value, const char* message)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(message);
}

std::array<double, 100> rain_bin_centers()
{
    std::array<double, 101> edges{};
    std::array<double, 100> centers{};
    edges.front() = minimum_rain_diameter;
    edges.back() = 0.005;
    for (std::size_t index = 1; index < 100; ++index)
        edges[index] = std::exp(
            static_cast<double>(index) / 100.0
                * std::log(edges.back() / edges.front())
            + std::log(edges.front()));
    for (std::size_t index = 0; index < centers.size(); ++index)
        centers[index] = std::sqrt(edges[index] * edges[index + 1]);
    return centers;
}

double rain_cloud_collection_efficiency(double rain_diameter_m,
                                        double cloud_diameter_m)
{
    if (rain_diameter_m < minimum_rain_diameter
        || cloud_diameter_m < 3.0e-6)
        return 0.0;
    const double ratio = cloud_diameter_m / rain_diameter_m;
    double efficiency = 0.0;
    if (ratio > 0.25)
    {
        const double x = cloud_diameter_m * 1.0e6;
        if (rain_diameter_m < 75.0e-6)
            efficiency = 0.026794 * x - 0.20604;
        else if (rain_diameter_m < 125.0e-6)
            efficiency = -0.00066842 * x * x + 0.061542 * x - 0.37089;
        else if (rain_diameter_m < 175.0e-6)
            efficiency = 4.091e-6 * std::pow(x, 4.0)
                - 0.00030908 * std::pow(x, 3.0)
                + 0.0066237 * x * x - 0.0013687 * x - 0.073022;
        else if (rain_diameter_m < 250.0e-6)
            efficiency = 9.6719e-5 * std::pow(x, 3.0)
                - 0.0068901 * x * x + 0.17305 * x - 0.65988;
        else if (rain_diameter_m < 350.0e-6)
            efficiency = 9.0488e-5 * std::pow(x, 3.0)
                - 0.006585 * x * x + 0.16606 * x - 0.56125;
        else
            efficiency = 0.00010721 * std::pow(x, 3.0)
                - 0.0072962 * x * x + 0.1704 * x - 0.46929;
    }
    else
    {
        const double velocity = -0.1021
            + 4.932e3 * rain_diameter_m
            - 0.9551e6 * rain_diameter_m * rain_diameter_m
            + 0.07934e9 * std::pow(rain_diameter_m, 3.0)
            - 0.002362e12 * std::pow(rain_diameter_m, 4.0);
        const double stokes = cloud_diameter_m * cloud_diameter_m
            * velocity * water_density
            / (9.0 * 1.718e-5 * rain_diameter_m);
        const double reynolds = 9.0 * stokes
            / (ratio * ratio * water_density);
        if (stokes > 0.0 && reynolds > 0.0)
        {
            const double f = std::log(reynolds);
            const double k0 = std::exp(
                -0.1007 - 0.358 * f + 0.0261 * f * f);
            const double z = std::log(stokes / (k0 + 1.0e-15));
            const double h = 0.1465 + 1.302 * z - 0.607 * z * z
                + 0.293 * z * z * z;
            const double y0 = 2.0 / pi * std::atan(h);
            efficiency = std::pow(y0 + ratio, 2.0)
                / std::pow(1.0 + ratio, 2.0);
        }
    }
    return std::clamp(efficiency, 0.0, 0.95);
}

double tabulated_rain_cloud_efficiency(double rain_mvd_m,
                                       double cloud_mvd_m)
{
    static const auto rain_bins = rain_bin_centers();
    int rain_index = 1 + static_cast<int>(
        100.0 * std::log(rain_mvd_m / rain_bins.front())
        / std::log(rain_bins.back() / rain_bins.front()));
    rain_index = std::clamp(rain_index, 1, 100);
    const int cloud_index = std::clamp(
        static_cast<int>(cloud_mvd_m * 1.0e6), 1, 100);
    return rain_cloud_collection_efficiency(
        rain_bins[static_cast<std::size_t>(rain_index - 1)],
        static_cast<double>(cloud_index) * 1.0e-6);
}

struct RainDistribution
{
    double slope_m_inv = 0.0;
    double inverse_slope_m = 0.0;
    double median_volume_diameter_m = 0.0;
    double intercept_m4 = 0.0;
};

RainDistribution rain_distribution(double rain_water_kg_m3,
                                   double rain_number_m3)
{
    RainDistribution result;
    if (rain_water_kg_m3 <= 1.0e-12) return result;
    require_positive(rain_number_m3, "invalid Thompson rain number");
    result.slope_m_inv = std::cbrt(
        rain_mass_coefficient * 6.0 * rain_number_m3
        / rain_water_kg_m3);
    result.inverse_slope_m = 1.0 / result.slope_m_inv;
    result.median_volume_diameter_m = 3.672 * result.inverse_slope_m;
    result.intercept_m4 = rain_number_m3 * result.slope_m_inv;
    return result;
}

void validate_conversion(const WarmRainConversionRates& rates)
{
    const double water = rates.cloud_water_kg_m3_s
        + rates.rain_water_kg_m3_s;
    if (!std::isfinite(rates.cloud_water_kg_m3_s) || !std::isfinite(rates.rain_water_kg_m3_s) ||
        !std::isfinite(rates.cloud_number_m3_s) || !std::isfinite(rates.rain_number_m3_s) || std::abs(water) > 1.0e-12)
        throw std::runtime_error("invalid Thompson warm-rain rates");
}

} // namespace

std::shared_ptr<const std::vector<float>> native_warm_rain_coefficients()
{
    static const auto coefficients = []()
    {
        auto values = std::make_shared<std::vector<float>>();
        values->reserve(20200);
        const auto bins   = rain_bin_centers();
        const auto append = [&](double value)
        {
            const float high = static_cast<float>(value);
            values->push_back(high);
            values->push_back(static_cast<float>(value - high));
        };
        for (double diameter : bins)
        {
            for (int cloud = 1; cloud <= 100; ++cloud)
            {
                append(rain_cloud_collection_efficiency(diameter, cloud * 1e-6));
            }
        }
        const double span   = std::log(bins.back() / bins.front());
        const auto   locate = [&](double diameter)
        { return std::clamp(static_cast<int>(100.0 * std::log(diameter / bins.front()) / span), 0, 99); };
        append(bins.front());
        for (int bin = 1; bin < 100; ++bin)
        {
            double edge  = bins.front() * std::exp(bin * span / 100.0);
            bool   found = false;
            for (int attempt = 0; attempt < 64; ++attempt)
            {
                if (locate(edge) < bin)
                {
                    edge = std::nextafter(edge, std::numeric_limits<double>::infinity());
                }
                else if (locate(std::nextafter(edge, 0.0)) < bin)
                {
                    found = true;
                    break;
                }
                else
                {
                    edge = std::nextafter(edge, 0.0);
                }
            }
            if (!found)
            {
                throw std::runtime_error("Thompson collection edge is not representable");
            }
            append(edge);
        }
        return std::shared_ptr<const std::vector<float>>(std::move(values));
    }();
    return coefficients;
}

WarmRainConversionRates warm_rain_conversion_rates(double temperature_k,
                                                   double dry_air_density_kg_m3,
                                                   double cloud_water_kg_m3,
                                                   double rain_water_kg_m3,
                                                   double cloud_number_m3,
                                                   double rain_number_m3,
                                                   double dt_s)
{
    require_positive(temperature_k, "invalid Thompson warm-rain temperature");
    require_positive(dry_air_density_kg_m3,
                     "invalid Thompson warm-rain density");
    require_nonnegative(cloud_water_kg_m3,
                        "invalid Thompson cloud water");
    require_nonnegative(rain_water_kg_m3,
                        "invalid Thompson rain water");
    require_nonnegative(cloud_number_m3,
                        "invalid Thompson cloud number");
    require_nonnegative(rain_number_m3,
                        "invalid Thompson rain number");
    require_positive(dt_s, "invalid Thompson warm-rain timestep");

    WarmRainConversionRates result;
    const RainDistribution rain = rain_distribution(
        rain_water_kg_m3, rain_number_m3);
    double self_collection = 0.0;
    if (rain_water_kg_m3 > 1.0e-12
        && rain.median_volume_diameter_m > minimum_rain_diameter)
    {
        const double efficiency = 1.0 - std::exp(
            2300.0 * (rain.median_volume_diameter_m - 1950.0e-6));
        self_collection = efficiency * 2.0
            * rain_number_m3 * rain_water_kg_m3;
    }

    double autoconversion = 0.0;
    double autoconversion_cloud_number = 0.0;
    double autoconversion_rain_number = 0.0;
    double accretion = 0.0;
    double accretion_cloud_number = 0.0;
    double cloud_mvd = minimum_cloud_diameter;
    if (cloud_water_kg_m3 > 1.0e-12)
    {
        require_positive(cloud_number_m3, "invalid Thompson cloud number");
        const double shape_coordinate = 1000.0e6 / cloud_number_m3;
        const int shape = shape_coordinate >= 13.0
            ? 15
            : std::clamp(
                static_cast<int>(std::lround(shape_coordinate)) + 2,
                2, 15);
        const double third_moment_ratio = (shape + 1.0) * (shape + 2.0) * (shape + 3.0);
        const double slope =
            std::cbrt(cloud_number_m3 * rain_mass_coefficient * third_moment_ratio / cloud_water_kg_m3);
        cloud_mvd = std::clamp(
            static_cast<double>(shape + 3.672) / slope,
            minimum_cloud_diameter, minimum_rain_diameter);
        if (cloud_water_kg_m3 > 0.01e-3)
        {
            const double characteristic = std::max(
                1.0,
                std::cbrt(cloud_water_kg_m3
                          / (rain_mass_coefficient * cloud_number_m3))
                    * 1.0e6);
            // Berry–Reinhardt needs M6/M3 here; M3/M0 repeats the mean-volume diameter and erases the breadth.
            const double sixth_to_third_moment_ratio = (shape + 4.0) * (shape + 5.0) * (shape + 6.0);
            const double gamma_diameter              = std::cbrt(sixth_to_third_moment_ratio) / slope * 1.0e6;
            const double breadth_sixth = std::pow(std::max(
                0.0,
                std::pow(characteristic, 3.0)
                    * std::pow(gamma_diameter, 3.0)
                - std::pow(characteristic, 6.0)), 1.0 / 6.0);
            const double zeta = 0.027 * cloud_water_kg_m3 * std::max(
                6.25e-6 * characteristic
                    * std::pow(breadth_sixth, 3.0) - 0.4,
                0.0);
            const double tau = 3.72 / (
                cloud_water_kg_m3
                * (std::max(0.5 * breadth_sixth - 7.5, 0.0) + 1.0e-12));
            autoconversion = std::min(
                cloud_water_kg_m3 / dt_s, zeta / tau);
            autoconversion_rain_number = autoconversion
                / (rain_mass_coefficient * static_cast<double>(shape)
                   * 10.0 * std::pow(minimum_rain_diameter, 3.0));
            autoconversion_cloud_number = std::min(
                cloud_number_m3 / dt_s,
                autoconversion
                    / (rain_mass_coefficient * std::pow(cloud_mvd, 3.0)));
        }
        if (rain_water_kg_m3 > 1.0e-12
            && rain.median_volume_diameter_m > minimum_rain_diameter
            && cloud_mvd > minimum_cloud_diameter)
        {
            const double density_factor = std::sqrt(
                reference_air_density / dry_air_density_kg_m3);
            const double efficiency = tabulated_rain_cloud_efficiency(
                rain.median_volume_diameter_m, cloud_mvd);
            const double coefficient = density_factor * pi * 0.25
                * rain_velocity_coefficient * std::tgamma(4.0)
                * efficiency * rain.intercept_m4
                * std::pow(rain.slope_m_inv + rain_velocity_decay, -4.0);
            accretion = std::min(
                cloud_water_kg_m3 / dt_s,
                coefficient * cloud_water_kg_m3);
            accretion_cloud_number = std::min(
                cloud_number_m3 / dt_s,
                coefficient * cloud_number_m3);
        }
    }

    const double cloud_sink = autoconversion + accretion;
    const double scale = cloud_sink > cloud_water_kg_m3 / dt_s
        ? cloud_water_kg_m3 / (cloud_sink * dt_s) : 1.0;
    autoconversion *= scale;
    accretion *= scale;
    autoconversion_cloud_number *= scale;
    accretion_cloud_number *= scale;
    autoconversion_rain_number *= scale;
    if (!std::isfinite(self_collection))
        throw std::runtime_error("invalid Thompson rain self-collection rate");
    self_collection = std::min(rain_number_m3 / dt_s, self_collection);

    result.cloud_water_kg_m3_s = -autoconversion - accretion;
    result.rain_water_kg_m3_s = -result.cloud_water_kg_m3_s;
    result.cloud_number_m3_s =
        -autoconversion_cloud_number - accretion_cloud_number;
    result.rain_number_m3_s =
        autoconversion_rain_number - self_collection;
    validate_conversion(result);
    return result;
}

RainEvaporationRates rain_evaporation_rates(
    double pressure_pa,
    double temperature_k,
    double dry_air_density_kg_m3,
    double vapor_kg_kg_dry,
    double rain_water_kg_m3,
    double rain_number_m3,
    double dt_s)
{
    require_positive(pressure_pa, "invalid Thompson evaporation pressure");
    require_positive(temperature_k,
                     "invalid Thompson evaporation temperature");
    require_positive(dry_air_density_kg_m3,
                     "invalid Thompson evaporation density");
    require_nonnegative(vapor_kg_kg_dry,
                        "invalid Thompson evaporation vapor");
    require_nonnegative(rain_water_kg_m3,
                        "invalid Thompson evaporation rain water");
    require_nonnegative(rain_number_m3,
                        "invalid Thompson evaporation rain number");
    require_positive(dt_s, "invalid Thompson evaporation timestep");

    RainEvaporationRates result;
    if (rain_water_kg_m3 <= 1.0e-12) return result;
    const double saturation = liquid_saturation_mixing_ratio(
        pressure_pa, temperature_k);
    const double supersaturation = vapor_kg_kg_dry / saturation - 1.0;
    if (supersaturation >= -1.0e-15) return result;

    const RainDistribution rain = rain_distribution(
        rain_water_kg_m3, rain_number_m3);
    const double celsius = temperature_k - 273.15;
    const double density_factor = std::sqrt(
        reference_air_density / dry_air_density_kg_m3);
    const double density_factor_root = std::sqrt(density_factor);
    const double diffusivity = 2.11e-5
        * std::pow(temperature_k / 273.15, 1.94)
        * (101325.0 / pressure_pa);
    const double viscosity = (celsius >= 0.0
        ? 1.718 + 0.0049 * celsius
        : 1.718 + 0.0049 * celsius - 1.2e-5 * celsius * celsius)
        * 1.0e-5;
    const double ventilation = std::sqrt(
        dry_air_density_kg_m3 / viscosity);
    const double latent_heat = latent_heat_at_freezing
        + (2106.0 - 4218.0) * celsius;
    const double conductivity = (5.69 + 0.0168 * celsius)
        * 1.0e-5 * 418.936;
    const double reciprocal_temperature = 1.0 / temperature_k;
    const double saturated_vapor_density =
        dry_air_density_kg_m3 * saturation;
    const double first_derivative = saturated_vapor_density
        * reciprocal_temperature
        * (latent_heat * reciprocal_temperature
           / water_vapor_gas_constant - 1.0);
    const double second_derivative = saturated_vapor_density * (
        std::pow(reciprocal_temperature
                     * (latent_heat * reciprocal_temperature
                        / water_vapor_gas_constant - 1.0),
                 2.0)
        - 2.0 * latent_heat * std::pow(reciprocal_temperature, 3.0)
            / water_vapor_gas_constant
        + reciprocal_temperature * reciprocal_temperature);
    const double gamma = latent_heat * diffusivity / conductivity
        * first_derivative;
    double alpha = 0.5 * std::pow(gamma / (1.0 + gamma), 2.0)
        * second_derivative / first_derivative
        * saturated_vapor_density / first_derivative;
    alpha = std::max(1.0e-9, alpha);
    const double saturation_deficit = std::min(-1.0e-9, supersaturation);
    const double evaporation_factor = 2.0 * pi * (
        1.0 - alpha * saturation_deficit
        + 2.0 * alpha * alpha * saturation_deficit * saturation_deficit
        - 5.0 * std::pow(alpha, 3.0)
            * std::pow(saturation_deficit, 3.0)) / (1.0 + gamma);

    double evaporation = 0.0;
    if (vapor_kg_kg_dry / saturation < 0.95
        && rain_water_kg_m3 / dry_air_density_kg_m3 <= 1.0e-8)
        evaporation = rain_water_kg_m3 / dt_s;
    else
    {
        const double first_ventilation = 0.78 * std::tgamma(2.0);
        const double second_ventilation = 0.308
            * std::cbrt(schmidt_number)
            * std::sqrt(rain_velocity_coefficient) * std::tgamma(3.0);
        const double raw = evaporation_factor * diffusivity
            * (-supersaturation) * rain.intercept_m4
            * saturated_vapor_density
            * (first_ventilation * std::pow(rain.inverse_slope_m, 2.0)
               + second_ventilation * ventilation * density_factor_root
                   * std::pow(rain.slope_m_inv
                                  + 0.5 * rain_velocity_decay,
                              -3.0));
        evaporation = std::min({
            rain_water_kg_m3 / dt_s,
            dry_air_density_kg_m3
                * (saturation - vapor_kg_kg_dry) / dt_s,
            raw});
    }
    if (!std::isfinite(evaporation) || evaporation < 0.0)
        throw std::runtime_error("invalid Thompson rain evaporation rate");
    const double number_return = std::min(
        0.99 * rain_number_m3 / dt_s,
        evaporation * rain_number_m3 / rain_water_kg_m3);
    result.vapor_kg_m3_s = evaporation;
    result.rain_water_kg_m3_s = -evaporation;
    result.rain_number_m3_s = -number_return;
    result.water_aerosol_m3_s = number_return;
    result.temperature_k_s = -latent_heat
        / (dry_heat_capacity * (1.0 + 0.887 * vapor_kg_kg_dry))
        * evaporation / dry_air_density_kg_m3;
    if (!std::isfinite(result.temperature_k_s)
        || !std::isfinite(result.rain_number_m3_s))
        throw std::runtime_error("invalid Thompson rain evaporation state");
    return result;
}

}
