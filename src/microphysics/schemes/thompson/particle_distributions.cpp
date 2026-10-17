#include "microphysics/schemes/thompson/particle_distributions.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double rho_reference = 101325.0 / (287.05 * 298.0);
constexpr double water_mass_coefficient = pi * 1000.0 / 6.0;
constexpr double ice_mass_coefficient = pi * 890.0 / 6.0;
constexpr double graupel_mass_coefficient = pi * 400.0 / 6.0;
constexpr double snow_mass_coefficient = 0.069;

constexpr std::array<double, 10> snow_a = {
    5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
    0.31255, 0.000204, 0.003199, 0.0, -0.015952};
constexpr std::array<double, 10> snow_b = {
    0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
    0.060366, 0.000079, 0.000594, 0.0, -0.003577};

void require_state(double content, double number, double density)
{
    if (!std::isfinite(content) || content < 0.0
        || !std::isfinite(number) || number < 0.0
        || !std::isfinite(density) || density <= 0.0)
        throw std::invalid_argument("invalid Thompson particle distribution state");
}

double density_factor(double density)
{
    return std::sqrt(rho_reference / density);
}

double snow_log_coefficient(const std::array<double, 10>& coefficients,
                            double temperature_c,
                            double moment)
{
    const double t2 = temperature_c * temperature_c;
    const double m2 = moment * moment;
    return coefficients[0] + coefficients[1] * temperature_c
        + coefficients[2] * moment
        + coefficients[3] * temperature_c * moment
        + coefficients[4] * t2 + coefficients[5] * m2
        + coefficients[6] * t2 * moment
        + coefficients[7] * temperature_c * m2
        + coefficients[8] * t2 * temperature_c
        + coefficients[9] * m2 * moment;
}

double snow_moment(double second_moment,
                   double temperature_c,
                   double order)
{
    const double coefficient = std::pow(
        10.0, snow_log_coefficient(snow_a, temperature_c, order));
    const double exponent = snow_log_coefficient(
        snow_b, temperature_c, order);
    return coefficient * std::pow(second_moment, exponent);
}

} // namespace

MassNumberFallSpeeds cloud_fall_speeds(double cloud_water_kg_m3,
                                       double cloud_number_m3,
                                       double air_density_kg_m3)
{
    require_state(cloud_water_kg_m3, cloud_number_m3, air_density_kg_m3);
    if (cloud_water_kg_m3 <= 1.0e-12 || cloud_number_m3 <= 2.0)
        return {};
    const int shape = std::min(
        15, static_cast<int>(std::floor(1.0e9 / cloud_number_m3 + 0.5)) + 2);
    const double slope = std::cbrt(
        cloud_number_m3 * water_mass_coefficient
        * std::tgamma(shape + 4.0) / std::tgamma(shape + 1.0)
        / cloud_water_kg_m3);
    const double inverse_slope = 1.0 / slope;
    const double factor = density_factor(air_density_kg_m3) * 0.316946e8
        * inverse_slope * inverse_slope;
    return {
        factor * std::tgamma(shape + 6.0) / std::tgamma(shape + 4.0),
        factor * std::tgamma(shape + 3.0) / std::tgamma(shape + 1.0)};
}

MassNumberFallSpeeds rain_fall_speeds(double rain_water_kg_m3,
                                      double rain_number_m3,
                                      double air_density_kg_m3)
{
    require_state(rain_water_kg_m3, rain_number_m3, air_density_kg_m3);
    if (rain_water_kg_m3 <= 1.0e-12 || rain_number_m3 <= 1.0e-6)
        return {};
    double slope = std::cbrt(
        water_mass_coefficient * 6.0 * rain_number_m3 / rain_water_kg_m3);
    const double median_volume_diameter = 3.672 / slope;
    if (median_volume_diameter > 2.5e-3) slope = 3.672 / 2.5e-3;
    else if (median_volume_diameter < 37.5e-6) slope = 3.672 / 37.5e-6;
    const double factor = density_factor(air_density_kg_m3) * 4854.0;
    return {
        factor * 4.0 * std::pow(slope, 4.0)
            / std::pow(slope + 195.0, 5.0),
        factor * std::tgamma(3.5) / std::tgamma(2.5)
            * std::pow(slope, 2.5) / std::pow(slope + 195.0, 3.5)};
}

MassNumberFallSpeeds ice_fall_speeds(double ice_water_kg_m3,
                                     double ice_number_m3,
                                     double air_density_kg_m3)
{
    require_state(ice_water_kg_m3, ice_number_m3, air_density_kg_m3);
    if (ice_water_kg_m3 <= 1.0e-12 || ice_number_m3 <= 1.0e-6)
        return {};
    double slope = std::cbrt(
        ice_mass_coefficient * 6.0 * ice_number_m3 / ice_water_kg_m3);
    const double mass_diameter = 4.0 / slope;
    if (mass_diameter < 5.0e-6) slope = 4.0 / 5.0e-6;
    else if (mass_diameter > 300.0e-6) slope = 4.0 / 300.0e-6;
    const double factor = density_factor(air_density_kg_m3) * 1493.9 / slope;
    return {factor * 4.0, factor * 2.5};
}

SnowDistribution snow_distribution(double snow_water_kg_m3,
                                   double temperature_k,
                                   double air_density_kg_m3)
{
    require_state(snow_water_kg_m3, 0.0, air_density_kg_m3);
    if (!std::isfinite(temperature_k) || temperature_k <= 0.0)
        throw std::invalid_argument("invalid Thompson snow temperature");
    if (snow_water_kg_m3 <= 1.0e-12) return {};
    const double temperature_c = std::min(-0.1, temperature_k - 273.15);
    const double second = snow_water_kg_m3 / snow_mass_coefficient;
    const double third = snow_moment(second, temperature_c, 3.0);
    const double inverse_size = second / third;
    const double number_numerator = second * inverse_size * inverse_size
        * inverse_size;
    const double number = number_numerator * 490.6
            / (inverse_size * 20.78)
        + number_numerator * 17.46 * std::pow(inverse_size, 0.6357)
            * std::tgamma(1.6357)
            / std::pow(inverse_size * 3.29, 1.6357);
    const double first = 1.0 / (inverse_size * 20.78 + 100.0);
    const double second_slope = 1.0 / (inverse_size * 3.29 + 100.0);
    const double numerator = 490.6 * std::tgamma(3.55)
            * std::pow(first, 3.55)
        + 17.46 * std::pow(inverse_size, 0.6357)
            * std::tgamma(4.1857) * std::pow(second_slope, 4.1857);
    const double denominator = 490.6 * std::tgamma(3.0)
            / std::pow(inverse_size * 20.78, 3.0)
        + 17.46 * std::pow(inverse_size, 0.6357)
            * std::tgamma(3.6357)
            / std::pow(inverse_size * 3.29, 3.6357);
    return {second, third, number,
            density_factor(air_density_kg_m3) * 40.0
                * numerator / denominator};
}

MassNumberFallSpeeds graupel_fall_speeds(double graupel_water_kg_m3,
                                         double air_density_kg_m3)
{
    require_state(graupel_water_kg_m3, 0.0, air_density_kg_m3);
    if (graupel_water_kg_m3 <= 1.0e-12) return {};
    const double logarithmic_content = std::log10(
        std::max(1.0e-9, graupel_water_kg_m3));
    const double intercept = std::pow(
        10.0, std::clamp(3.0 + 2.0 / 7.0
                         * (logarithmic_content + 8.0), 2.0, 6.0));
    const double exponential_slope = std::pow(
        intercept * graupel_mass_coefficient / graupel_water_kg_m3, 0.25);
    const double slope = exponential_slope * std::cbrt(6.0);
    const double factor = density_factor(air_density_kg_m3) * 442.0
        * std::pow(1.0 / slope, 0.89);
    return {factor * std::tgamma(4.89) / std::tgamma(4.0),
            factor * std::tgamma(1.89) / std::tgamma(1.0)};
}

} // namespace tmv::thompson
