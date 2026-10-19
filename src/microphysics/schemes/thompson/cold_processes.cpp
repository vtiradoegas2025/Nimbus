#include "microphysics/schemes/thompson/cold_processes.hpp"
#include "microphysics/schemes/thompson/cloud_phase.hpp"

#include "microphysics/base/thermodynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double freezing_temperature      = cloud_melting_temperature;
constexpr double minimum_rain_content = 1.0e-6;
constexpr double minimum_active_content = 1.0e-12;
constexpr double water_mass_coefficient = pi * 1000.0 / 6.0;
constexpr double minimum_rain_mvd = 0.75 * 50.0e-6;
constexpr double maximum_rain_mvd = 2.5e-3;
constexpr double rain_mvd_shape_offset = 3.672;
constexpr double maximum_cloud_number = 2000.0e6;
constexpr double minimum_cloud_diameter = 1.0e-6;
constexpr double maximum_cloud_diameter = 100.0e-6;
constexpr double ice_mass_coefficient = pi * 890.0 / 6.0;
constexpr double minimum_ice_diameter = 1.2898438095973233e-5;
constexpr double latent_sublimation = 2.834e6;
constexpr double vapor_gas_constant = 461.5;
constexpr double heat_capacity = 1004.0;
constexpr double latent_fusion = 3.34e5;
constexpr double snow_mass_coefficient = 0.069;
constexpr double snow_velocity_coefficient = 40.0;
constexpr double snow_velocity_exponent = 0.55;
constexpr double snow_velocity_decay = 100.0;
constexpr double reference_air_density = 101325.0 / (287.05 * 298.0);

constexpr std::array<double, 10> snow_a = {
    5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
    0.31255, 0.000204, 0.003199, 0.0, -0.015952};
constexpr std::array<double, 10> snow_b = {
    0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
    0.060366, 0.000079, 0.000594, 0.0, -0.003577};

void require_state(const RainFreezingState& state)
{
    if (!std::isfinite(state.temperature_k) || state.temperature_k <= 0.0
        || !std::isfinite(state.rain_kg_m3) || state.rain_kg_m3 < 0.0
        || !std::isfinite(state.rain_number_m3)
        || state.rain_number_m3 < 0.0
        || !std::isfinite(state.ice_nuclei_m3)
        || state.ice_nuclei_m3 < 0.0)
        throw std::invalid_argument("invalid Thompson rain-freezing state");
}

void require_state(const CloudFreezingState& state)
{
    if (!std::isfinite(state.temperature_k) || state.temperature_k <= 0.0
        || !std::isfinite(state.cloud_kg_m3) || state.cloud_kg_m3 < 0.0
        || !std::isfinite(state.cloud_number_m3)
        || state.cloud_number_m3 < 0.0
        || !std::isfinite(state.ice_nuclei_m3)
        || state.ice_nuclei_m3 < 0.0)
        throw std::invalid_argument("invalid Thompson cloud-freezing state");
}

void require_state(const IceDepositionState& state, double dt_s)
{
    if (!std::isfinite(state.pressure_pa) || state.pressure_pa <= 0.0
        || !std::isfinite(state.temperature_k) || state.temperature_k <= 0.0
        || !std::isfinite(state.dry_air_density_kg_m3)
        || state.dry_air_density_kg_m3 <= 0.0
        || !std::isfinite(state.vapor_kg_kg_dry)
        || state.vapor_kg_kg_dry < 0.0
        || !std::isfinite(state.ice_kg_m3) || state.ice_kg_m3 < 0.0
        || !std::isfinite(state.ice_number_m3)
        || state.ice_number_m3 < 0.0
        || !std::isfinite(state.snow_kg_m3) || state.snow_kg_m3 < 0.0
        || !std::isfinite(state.graupel_kg_m3)
        || state.graupel_kg_m3 < 0.0
        || !std::isfinite(dt_s) || dt_s <= 0.0)
        throw std::invalid_argument("invalid Thompson ice-deposition state");
}

void require_state(const SnowRimingState& state, double dt_s)
{
    if (!std::isfinite(state.temperature_k) || state.temperature_k <= 0.0
        || !std::isfinite(state.dry_air_density_kg_m3)
        || state.dry_air_density_kg_m3 <= 0.0
        || !std::isfinite(state.cloud_kg_m3) || state.cloud_kg_m3 < 0.0
        || !std::isfinite(state.cloud_number_m3)
        || state.cloud_number_m3 < 0.0
        || !std::isfinite(state.snow_kg_m3) || state.snow_kg_m3 < 0.0
        || !std::isfinite(state.graupel_kg_m3)
        || state.graupel_kg_m3 < 0.0
        || !std::isfinite(state.snow_deposition_kg_m3_s)
        || !std::isfinite(dt_s) || dt_s <= 0.0)
        throw std::invalid_argument("invalid Thompson snow-riming state");
}

void require_state(const IceSnowConversionState& state, double dt_s)
{
    if (!std::isfinite(state.ice_kg_m3) || state.ice_kg_m3 < 0.0
        || !std::isfinite(state.ice_number_m3)
        || state.ice_number_m3 < 0.0
        || !std::isfinite(dt_s) || dt_s <= 0.0)
        throw std::invalid_argument(
            "invalid Thompson ice-to-snow state");
}

std::size_t decade_index(double value, int first_exponent,
                         std::size_t count)
{
    if (!(value > 0.0) || !std::isfinite(value)) return 0;
    const int rounded = static_cast<int>(std::floor(std::log10(value) + 0.5));
    int exponent = rounded;
    for (int candidate = rounded - 1; candidate <= rounded + 1; ++candidate)
    {
        const double scaled = value / std::pow(10.0, candidate);
        if (scaled >= 1.0 && scaled < 10.0)
        {
            exponent = candidate;
            break;
        }
    }
    const int digit = static_cast<int>(value / std::pow(10.0, exponent));
    const int one_based = digit + 9 * (exponent - first_exponent);
    return static_cast<std::size_t>(
        std::clamp(one_based, 1, static_cast<int>(count)) - 1);
}

IceSnowIndex deposition_index(double mass, double sorted_number)
{
    return {decade_index(mass, -10, ColdProcessTableMetadata::ice_contents),
            decade_index(sorted_number, 0, ColdProcessTableMetadata::ice_numbers)};
}

double sorted_rain_number(const RainFreezingState& state)
{
    if (state.rain_kg_m3 <= minimum_active_content
        || state.rain_number_m3 <= minimum_active_content)
        return state.rain_number_m3;
    double lambda = std::cbrt(
        water_mass_coefficient * 6.0 * state.rain_number_m3
        / state.rain_kg_m3);
    const double mvd = rain_mvd_shape_offset / lambda;
    if (mvd > maximum_rain_mvd)
        lambda = rain_mvd_shape_offset / maximum_rain_mvd;
    else if (mvd < minimum_rain_mvd)
        lambda = rain_mvd_shape_offset / minimum_rain_mvd;
    else
        return state.rain_number_m3;
    return state.rain_kg_m3 * lambda * lambda * lambda
        / (6.0 * water_mass_coefficient);
}

double sorted_cloud_number(const CloudFreezingState& state)
{
    if (state.cloud_kg_m3 <= minimum_active_content)
        return state.cloud_number_m3;
    double number = std::clamp(
        state.cloud_number_m3, 2.0, maximum_cloud_number);
    const int shape = std::min(
        15, static_cast<int>(std::floor(1000.0e6 / number + 0.5)) + 2);
    const double moment_ratio = (shape + 1.0) * (shape + 2.0)
        * (shape + 3.0);
    double lambda = std::cbrt(
        number * water_mass_coefficient * moment_ratio
        / state.cloud_kg_m3);
    const double mean_diameter = (shape + 4.0) / lambda;
    if (mean_diameter < minimum_cloud_diameter)
        lambda = (shape + 4.0) / minimum_cloud_diameter;
    else if (mean_diameter > maximum_cloud_diameter)
        lambda = (shape + 4.0) / maximum_cloud_diameter;
    else
        return number;
    return std::min(
        maximum_cloud_number,
        state.cloud_kg_m3 * std::pow(lambda, 3.0)
            / (water_mass_coefficient * moment_ratio));
}

double sorted_ice_number(const IceDepositionState& state)
{
    if (state.ice_kg_m3 <= minimum_active_content)
        return state.ice_number_m3;
    double number = std::max(1.0e-6, state.ice_number_m3);
    if (number <= 1.0e-6)
    {
        const double lambda = 4.0 / 5.0e-6;
        return std::min(
            999.0e3,
            state.ice_kg_m3 * std::pow(lambda, 3.0)
                / (6.0 * ice_mass_coefficient));
    }
    double lambda = std::cbrt(
        ice_mass_coefficient * 6.0 * number / state.ice_kg_m3);
    const double mean_diameter = 4.0 / lambda;
    if (mean_diameter < 5.0e-6)
    {
        lambda = 4.0 / 5.0e-6;
        number = std::min(
            999.0e3,
            state.ice_kg_m3 * std::pow(lambda, 3.0)
                / (6.0 * ice_mass_coefficient));
    }
    else if (mean_diameter > 300.0e-6)
    {
        lambda = 4.0 / 300.0e-6;
        number = state.ice_kg_m3 * std::pow(lambda, 3.0)
            / (6.0 * ice_mass_coefficient);
    }
    return number;
}

double sorted_ice_number(double ice_kg_m3, double ice_number_m3)
{
    return sorted_ice_number(IceDepositionState{
        1.0, 1.0, 1.0, 0.0, ice_kg_m3, ice_number_m3});
}

struct SublimationThermodynamics
{
    double saturation_mixing_ratio_ice = 0.0;
    double vapor_density_saturation = 0.0;
    double supersaturation_ice = 0.0;
    double diffusion = 0.0;
    double factor = 0.0;
    double viscosity = 0.0;
};

SublimationThermodynamics sublimation_thermodynamics(
    const IceDepositionState& state)
{
    SublimationThermodynamics result;
    result.saturation_mixing_ratio_ice =
        thermodynamics::saturation_mixing_ratio_ice(
            static_cast<float>(state.temperature_k),
            static_cast<float>(state.pressure_pa));
    result.supersaturation_ice = state.vapor_kg_kg_dry
        / result.saturation_mixing_ratio_ice - 1.0;
    if (std::abs(result.supersaturation_ice) < 1.0e-9)
        result.supersaturation_ice = 0.0;
    result.diffusion = 2.11e-5
        * std::pow(state.temperature_k / freezing_temperature, 1.94)
        * (101325.0 / state.pressure_pa);
    const double celsius = state.temperature_k - freezing_temperature;
    result.viscosity = celsius >= 0.0
        ? (1.718 + 0.0049 * celsius) * 1.0e-5
        : (1.718 + 0.0049 * celsius - 1.2e-5 * celsius * celsius)
            * 1.0e-5;
    const double conductivity = (5.69 + 0.0168 * celsius)
        * 1.0e-5 * 418.936;
    result.vapor_density_saturation = state.dry_air_density_kg_m3
        * result.saturation_mixing_ratio_ice;
    const double inverse_temperature = 1.0 / state.temperature_k;
    const double derivative = result.vapor_density_saturation
        * inverse_temperature
        * (latent_sublimation * inverse_temperature / vapor_gas_constant
           - 1.0);
    const double second_derivative = result.vapor_density_saturation
        * (std::pow(
               inverse_temperature
                   * (latent_sublimation * inverse_temperature
                      / vapor_gas_constant - 1.0),
               2.0)
           - 2.0 * latent_sublimation
               * inverse_temperature * inverse_temperature
               * inverse_temperature / vapor_gas_constant
           + inverse_temperature * inverse_temperature);
    const double gamma = latent_sublimation * result.diffusion
        / conductivity * derivative;
    double alpha = 0.5 * std::pow(gamma / (1.0 + gamma), 2.0)
        * second_derivative / derivative
        * result.vapor_density_saturation / derivative;
    alpha = std::max(1.0e-9, alpha);
    const double supersaturation = result.supersaturation_ice;
    result.factor = 4.0 * pi
        * (1.0 - alpha * supersaturation
           + 2.0 * alpha * alpha * supersaturation * supersaturation
           - 5.0 * alpha * alpha * alpha * supersaturation
               * supersaturation * supersaturation)
        / (1.0 + gamma);
    if (!std::isfinite(result.factor) || result.factor < 0.0)
        throw std::invalid_argument("invalid Thompson ice-deposition thermodynamics");
    return result;
}

double snow_log_coefficient(const std::array<double, 10>& coefficients,
                            double temperature_c, double moment)
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

double snow_moment(double second_moment, double temperature_c,
                   double order)
{
    const double coefficient = std::pow(
        10.0, snow_log_coefficient(snow_a, temperature_c, order));
    const double exponent = snow_log_coefficient(
        snow_b, temperature_c, order);
    return coefficient * std::pow(second_moment, exponent);
}

double cloud_mvd(double cloud_kg_m3, double cloud_number_m3)
{
    const int shape = std::min(
        15, static_cast<int>(std::floor(1000.0e6 / cloud_number_m3 + 0.5))
            + 2);
    const double lambda = std::cbrt(
        cloud_number_m3 * water_mass_coefficient
        * std::tgamma(shape + 4.0) / std::tgamma(shape + 1.0)
        / cloud_kg_m3);
    return std::clamp(
        (shape + 3.672) / lambda,
        minimum_cloud_diameter, 50.0e-6);
}

double snow_cloud_collection_efficiency(double snow_diameter,
                                        double cloud_diameter)
{
    constexpr std::size_t bins = 100;
    const double first_edge = 300.0e-6;
    const double last_edge = 0.02;
    const double first_center = std::sqrt(
        first_edge * std::exp(std::log(last_edge / first_edge) / bins)
        * first_edge);
    const double last_lower = std::exp(
        99.0 / bins * std::log(last_edge / first_edge)
        + std::log(first_edge));
    const double last_center = std::sqrt(last_lower * last_edge);
    int snow_index = 1 + static_cast<int>(
        bins * std::log(snow_diameter / first_center)
        / std::log(last_center / first_center));
    snow_index = std::clamp(snow_index, 1, static_cast<int>(bins));
    const double lower = std::exp(
        static_cast<double>(snow_index - 1) / bins
            * std::log(last_edge / first_edge)
        + std::log(first_edge));
    const double upper = std::exp(
        static_cast<double>(snow_index) / bins
            * std::log(last_edge / first_edge)
        + std::log(first_edge));
    const double table_snow_diameter = std::sqrt(lower * upper);
    const int cloud_index = std::clamp(
        static_cast<int>(cloud_diameter * 1.0e6), 1, 100);
    const double table_cloud_diameter = cloud_index * 1.0e-6;
    const double cloud_velocity = 1.19e4
        * (1.0e4 * table_cloud_diameter * table_cloud_diameter * 0.25);
    const double snow_velocity = snow_velocity_coefficient
        * std::pow(table_snow_diameter, snow_velocity_exponent)
        * std::exp(-snow_velocity_decay * table_snow_diameter)
        - cloud_velocity;
    const double melted_diameter = std::cbrt(
        snow_mass_coefficient * table_snow_diameter
            * table_snow_diameter / water_mass_coefficient);
    const double ratio = table_cloud_diameter / melted_diameter;
    if (ratio > 0.25 || table_snow_diameter < 300.0e-6
        || table_cloud_diameter < 6.0e-6 || snow_velocity < 1.0e-3)
        return 0.0;
    const double stokes = table_cloud_diameter * table_cloud_diameter
        * snow_velocity * 1000.0
        / (9.0 * 1.718e-5 * melted_diameter);
    const double reynolds = 9.0 * stokes / (ratio * ratio * 1000.0);
    const double f = std::log(reynolds);
    const double k0 = std::exp(
        -0.1007 - 0.358 * f + 0.0261 * f * f);
    const double z = std::log(stokes / (k0 + 1.0e-15));
    const double h = 0.1465 + 1.302 * z - 0.607 * z * z
        + 0.293 * z * z * z;
    const double y = 2.0 / pi * std::atan(h);
    return std::clamp(
        static_cast<double>(static_cast<float>(
            (y + ratio) * (y + ratio)
            / ((1.0 + ratio) * (1.0 + ratio)))),
        0.0, 0.95);
}

struct GraupelRimingKernel
{
    double mass_rate = 0.0;
    double number_rate = 0.0;
};

GraupelRimingKernel graupel_riming_kernel(
    const SnowRimingState& state,
    double cloud_diameter)
{
    if (state.graupel_kg_m3 < 1.0e-6
        || cloud_diameter <= minimum_cloud_diameter)
        return {};
    constexpr double graupel_mass_coefficient = pi * 400.0 / 6.0;
    constexpr double velocity_coefficient = 442.0;
    constexpr double velocity_exponent = 0.89;
    const double intercept = std::pow(
        10.0, std::clamp(
            3.0 + 2.0 / 7.0
                * (std::log10(std::max(1.0e-9, state.graupel_kg_m3))
                   + 8.0),
            2.0, 6.0));
    const double lambda = std::pow(
        intercept * graupel_mass_coefficient * 6.0
            / state.graupel_kg_m3,
        0.25);
    const double inverse_lambda = 1.0 / lambda;
    const double number = state.graupel_kg_m3
        * lambda * lambda * lambda
        / (6.0 * graupel_mass_coefficient);
    const double distribution_intercept = number * lambda;
    const double graupel_diameter = 4.0 * inverse_lambda;
    const double density_factor = std::sqrt(
        reference_air_density / state.dry_air_density_kg_m3);
    const double velocity = density_factor * velocity_coefficient
        * std::tgamma(4.0 + velocity_exponent) / 6.0
        * std::pow(inverse_lambda, velocity_exponent);
    const double celsius = state.temperature_k - freezing_temperature;
    const double viscosity = celsius >= 0.0
        ? (1.718 + 0.0049 * celsius) * 1.0e-5
        : (1.718 + 0.0049 * celsius - 1.2e-5 * celsius * celsius)
            * 1.0e-5;
    const double stokes = cloud_diameter * cloud_diameter * velocity
        * 1000.0 / (9.0 * viscosity * graupel_diameter);
    const double efficiency = stokes < 0.4 ? 0.0
        : stokes <= 10.0 ? 0.55 * std::log10(2.51 * stokes)
                         : 0.77;
    const double coefficient = pi * 0.25 * velocity_coefficient
        * std::tgamma(3.0 + velocity_exponent);
    const double kernel = density_factor * coefficient * efficiency
        * distribution_intercept
        * std::pow(inverse_lambda, 3.0 + velocity_exponent);
    return {state.cloud_kg_m3 * kernel,
            state.cloud_number_m3 * kernel};
}

} // namespace

std::vector<double> native_riming_coefficients()
{
    std::vector<double> values;
    values.reserve(10002);
    constexpr double first = 300e-6, last = .02;
    values.push_back(std::sqrt(first * std::exp(std::log(last / first) / 100) * first));
    values.push_back(std::sqrt(std::exp(99.0 / 100 * std::log(last / first) + std::log(first)) * last));
    for (int bin = 0; bin < 100; ++bin)
    {
        const double lower    = std::exp(bin / 100.0 * std::log(last / first) + std::log(first));
        const double upper    = std::exp((bin + 1) / 100.0 * std::log(last / first) + std::log(first));
        const double diameter = std::sqrt(lower * upper);
        for (int cloud = 1; cloud <= 100; ++cloud)
            values.push_back(snow_cloud_collection_efficiency(diameter, (cloud + .25) * 1e-6));
    }
    return values;
}

RainFreezingIndex rain_freezing_lookup_index(const RainFreezingState& state)
{
    require_state(state);
    if (state.temperature_k >= freezing_temperature
        || state.rain_kg_m3 <= minimum_rain_content)
        return {};

    const double rain_number = sorted_rain_number(state);
    const double lambda = std::cbrt(
        water_mass_coefficient * 6.0 * rain_number / state.rain_kg_m3);
    const double exponential_intercept = state.rain_kg_m3
        * std::pow(lambda, 4.0) / (6.0 * water_mass_coefficient);
    const double supercooling = freezing_temperature - state.temperature_k;
    const auto temperature = static_cast<std::size_t>(std::clamp(
        static_cast<int>(std::floor(supercooling + 0.5)), 1, 45) - 1);
    return {
        decade_index(state.rain_kg_m3, -6,
                     ColdProcessTableMetadata::water_contents),
        decade_index(exponential_intercept, 6,
                     ColdProcessTableMetadata::rain_intercepts),
        temperature,
        decade_index(state.ice_nuclei_m3, 0,
                     ColdProcessTableMetadata::ice_nuclei)};
}

RainFreezingExchange rain_freezing_exchange(
    const RainFreezingCell& cell,
    const RainFreezingState& state)
{
    require_state(state);
    for (double value : {
             cell.ice_mass, cell.graupel_mass,
             cell.ice_number, cell.rain_number})
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument(
                "invalid Thompson rain-freezing table value");

    if (state.temperature_k >= freezing_temperature
        || state.rain_kg_m3 <= minimum_active_content)
        return {};
    if (state.rain_kg_m3 <= minimum_rain_content)
    {
        if (state.temperature_k >= homogeneous_freezing_temperature)
            return {};
        return {
            -state.rain_kg_m3, state.rain_kg_m3, 0.0,
            -state.rain_number_m3, state.rain_number_m3, 0.0};
    }

    double ice_mass = cell.ice_mass;
    double graupel_mass = cell.graupel_mass;
    double ice_number = cell.ice_number;
    double graupel_number = cell.rain_number;
    const double requested_mass = ice_mass + graupel_mass;
    const double requested_number = ice_number + graupel_number;
    const double factor = std::min({
        1.0,
        requested_mass > 0.0
            ? state.rain_kg_m3 / requested_mass : 1.0,
        requested_number > 0.0
            ? state.rain_number_m3 / requested_number : 1.0});
    ice_mass *= factor;
    graupel_mass *= factor;
    ice_number *= factor;
    graupel_number *= factor;
    return {
        -(ice_mass + graupel_mass), ice_mass, graupel_mass,
        -(ice_number + graupel_number), ice_number, graupel_number};
}

RainFreezingExchange rain_freezing_exchange(
    const ColdProcessTables& tables,
    const RainFreezingState& state)
{
    require_state(state);
    if (state.temperature_k >= freezing_temperature
        || state.rain_kg_m3 <= minimum_active_content)
        return {};
    if (state.rain_kg_m3 <= minimum_rain_content)
        return rain_freezing_exchange(RainFreezingCell{}, state);
    if (tables.rain_freezing.size() != checked_rain_freezing_size())
        throw std::invalid_argument("Thompson rain-freezing table dimension mismatch");
    return rain_freezing_exchange(
        tables.rain_freezing[flatten(rain_freezing_lookup_index(state))],
        state);
}

CloudFreezingIndex cloud_freezing_lookup_index(const CloudFreezingState& state)
{
    require_state(state);
    if (state.temperature_k >= freezing_temperature || state.cloud_kg_m3 <= minimum_active_content)
        return {};

    const double cloud_number       = sorted_cloud_number(state);
    std::size_t  cloud_number_index = 0, upper = ColdProcessTableMetadata::cloud_numbers;
    while (upper - cloud_number_index > 1)
    {
        const auto middle = (cloud_number_index + upper) / 2;
        if (cloud_number >= freezing_cloud_number(middle))
            cloud_number_index = middle;
        else
            upper = middle;
    }
    const double supercooling = freezing_temperature - state.temperature_k;
    const auto   temperature =
        static_cast<std::size_t>(std::clamp(static_cast<int>(std::floor(supercooling + 0.5)), 1, 45) - 1);
    return {decade_index(state.cloud_kg_m3, -6, ColdProcessTableMetadata::water_contents),
            cloud_number_index,
            temperature,
            decade_index(state.ice_nuclei_m3, 0, ColdProcessTableMetadata::ice_nuclei)};
}

CloudFreezingExchange cloud_freezing_exchange(const CloudFreezingCell& cell, const CloudFreezingState& state)
{
    require_state(state);
    for (double value : {cell.ice_mass, cell.ice_number})
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument("invalid Thompson cloud-freezing table value");

    if (state.temperature_k >= freezing_temperature || state.cloud_kg_m3 <= minimum_active_content)
        return {};
    if (state.cloud_kg_m3 <= minimum_rain_content && state.temperature_k < homogeneous_freezing_temperature)
    {
        return {-state.cloud_kg_m3, state.cloud_kg_m3, -state.cloud_number_m3, state.cloud_number_m3};
    }

    const double mass        = std::min(state.cloud_kg_m3, cell.ice_mass);
    const double mass_factor = cell.ice_mass > 0.0 ? mass / cell.ice_mass : 0.0;
    const double number      = std::min({state.cloud_number_m3, cell.ice_number * mass_factor, mass / (2.0e-12)});
    return {-mass, mass, -number, number};
}

CloudFreezingExchange cloud_freezing_exchange(const ColdProcessTables& tables, const CloudFreezingState& state)
{
    require_state(state);
    if (state.temperature_k >= freezing_temperature || state.cloud_kg_m3 <= minimum_active_content)
        return {};
    if (state.cloud_kg_m3 <= minimum_rain_content && state.temperature_k < homogeneous_freezing_temperature)
        return cloud_freezing_exchange(CloudFreezingCell{}, state);
    if (tables.cloud_freezing.size() != checked_cloud_freezing_size())
        throw std::invalid_argument("Thompson cloud-freezing table dimension mismatch");
    const auto index             = cloud_freezing_lookup_index(state);
    const auto mass_interpolated = [&](CloudFreezingIndex cell_index)
    {
        if (state.cloud_kg_m3 < minimum_rain_content)
            return generate_cloud_freezing_cell(cell_index, state.cloud_kg_m3);
        auto         cell          = tables.cloud_freezing[flatten(cell_index)];
        const double lower_content = freezing_water_content(cell_index.cloud_content);
        if (cell_index.cloud_content + 1 < ColdProcessTableMetadata::water_contents &&
            state.cloud_kg_m3 > lower_content)
        {
            const double upper_content = freezing_water_content(++cell_index.cloud_content);
            const auto   upper_cell    = tables.cloud_freezing[flatten(cell_index)];
            for (double value : {cell.ice_mass, cell.ice_number, upper_cell.ice_mass, upper_cell.ice_number})
                if (!std::isfinite(value) || value < 0)
                    throw std::invalid_argument("invalid Thompson cloud-freezing table value");
            const double weight =
                std::clamp((state.cloud_kg_m3 - lower_content) / (upper_content - lower_content), 0.0, 1.0);
            cell.ice_mass += weight * (upper_cell.ice_mass - cell.ice_mass);
            cell.ice_number += weight * (upper_cell.ice_number - cell.ice_number);
        }
        return cell;
    };
    auto         cell   = mass_interpolated(index);
    const double number = sorted_cloud_number(state), lower_number = freezing_cloud_number(index.cloud_number);
    if (index.cloud_number + 1 < ColdProcessTableMetadata::cloud_numbers && number > lower_number)
    {
        auto         upper_index  = index;
        const double upper_number = freezing_cloud_number(++upper_index.cloud_number);
        const auto   upper_cell   = mass_interpolated(upper_index);
        for (double value : {cell.ice_mass, cell.ice_number, upper_cell.ice_mass, upper_cell.ice_number})
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("invalid Thompson cloud-freezing table value");
        const double weight =
            std::clamp(std::log(number / lower_number) / std::log(upper_number / lower_number), 0.0, 1.0);
        cell.ice_mass += weight * (upper_cell.ice_mass - cell.ice_mass);
        cell.ice_number += weight * (upper_cell.ice_number - cell.ice_number);
    }
    return cloud_freezing_exchange(cell, state);
}

IceSnowIndex ice_deposition_lookup_index(const IceDepositionState& state)
{
    require_state(state, 1.0);
    if (state.ice_kg_m3 <= minimum_active_content)
        return {};
    return deposition_index(state.ice_kg_m3, sorted_ice_number(state));
}

IceDepositionExchange
ice_deposition_rates(const ColdProcessTables& tables, const IceDepositionState& state, double dt_s)
{
    require_state(state, dt_s);
    if (state.temperature_k >= freezing_temperature ||
        (state.ice_kg_m3 <= minimum_active_content && state.snow_kg_m3 <= minimum_active_content &&
         state.graupel_kg_m3 < 1.0e-6))
        return {};
    if (tables.ice_snow.size() != checked_ice_snow_size())
        throw std::invalid_argument("Thompson ice-deposition table dimension mismatch");

    const auto   thermodynamics  = sublimation_thermodynamics(state);
    double       ice_rate        = 0.0;
    double       snow_rate       = 0.0;
    double       snow_vapor_rate = 0.0;
    double       graupel_rate    = 0.0;
    double       number_rate     = 0.0;
    const double saturation_rate = (state.vapor_kg_kg_dry - thermodynamics.saturation_mixing_ratio_ice) *
                                   state.dry_air_density_kg_m3 / dt_s * 0.999;
    if (state.ice_kg_m3 > minimum_active_content)
    {
        const double ice_number     = sorted_ice_number(state);
        const double lambda         = std::cbrt(ice_mass_coefficient * 6.0 * ice_number / state.ice_kg_m3);
        const double inverse_lambda = 1.0 / lambda;
        const double mean_diameter  = std::max(minimum_ice_diameter, 4.0 * inverse_lambda);
        const double mean_mass      = ice_mass_coefficient * mean_diameter * mean_diameter * mean_diameter;
        ice_rate = 0.5 * thermodynamics.factor
            * thermodynamics.diffusion
            * thermodynamics.supersaturation_ice
            * thermodynamics.vapor_density_saturation
            * ice_number * inverse_lambda;
        if (ice_rate < 0.0)
        {
            ice_rate = std::max(-state.ice_kg_m3 / dt_s, ice_rate);
            number_rate = std::max(-std::min(ice_number, state.ice_number_m3) / dt_s, ice_rate / mean_mass);
        }
        else
        {
            const auto& partition = tables.ice_snow[flatten(deposition_index(state.ice_kg_m3, ice_number))];
            if (!std::isfinite(partition.ice_deposition_fraction)
                || partition.ice_deposition_fraction < 0.0
                || partition.ice_deposition_fraction > 1.0)
                throw std::invalid_argument(
                    "invalid Thompson ice-deposition table value");
            snow_rate =
                (1.0 - partition.ice_deposition_fraction) * ice_rate;
            ice_rate *= partition.ice_deposition_fraction;
        }
    }
    if (state.snow_kg_m3 > minimum_active_content)
    {
        const double actual_celsius =
            state.temperature_k - freezing_temperature;
        const double celsius = std::min(
            -0.1, state.temperature_k - freezing_temperature);
        const double second = state.snow_kg_m3 / snow_mass_coefficient;
        const double first = snow_moment(second, celsius, 1.0);
        const double ventilated = snow_moment(
            second, celsius, 1.0 + (snow_velocity_exponent + 1.0) / 2.0);
        const double capacitance = std::clamp(
            0.15 + (actual_celsius + 1.5)
                * (0.5 - 0.15) / (-30.0 + 1.5),
            0.15, 0.5);
        const double density_factor = std::sqrt(
            reference_air_density / state.dry_air_density_kg_m3);
        const double density_factor_sqrt = std::sqrt(density_factor);
        const double viscosity_factor = std::sqrt(
            state.dry_air_density_kg_m3 / thermodynamics.viscosity);
        const double ventilation = 0.28 * std::cbrt(0.632)
            * std::sqrt(snow_velocity_coefficient);
        double rate = capacitance * thermodynamics.factor
            * thermodynamics.diffusion
            * thermodynamics.supersaturation_ice
            * thermodynamics.vapor_density_saturation
            * (0.86 * first
               + ventilation * density_factor_sqrt
                   * viscosity_factor * ventilated);
        if (rate < 0.0)
            rate = std::max(-state.snow_kg_m3 / dt_s, rate);
        snow_vapor_rate = rate;
        snow_rate += snow_vapor_rate;
    }
    if (state.graupel_kg_m3 >= 1.0e-6
        && thermodynamics.supersaturation_ice < -1.0e-15)
    {
        constexpr double graupel_mass_coefficient = pi * 400.0 / 6.0;
        constexpr double velocity_coefficient = 442.0;
        constexpr double velocity_exponent = 0.89;
        const double intercept = std::pow(
            10.0, std::clamp(
                3.0 + 2.0 / 7.0
                    * (std::log10(std::max(
                           1.0e-9, state.graupel_kg_m3)) + 8.0),
                2.0, 6.0));
        const double lambda = std::pow(
            intercept * graupel_mass_coefficient * 6.0
                / state.graupel_kg_m3,
            0.25);
        const double inverse_lambda = 1.0 / lambda;
        const double density_factor = std::sqrt(
            reference_air_density / state.dry_air_density_kg_m3);
        const double density_factor_sqrt = std::sqrt(density_factor);
        const double viscosity_factor = std::sqrt(
            state.dry_air_density_kg_m3 / thermodynamics.viscosity);
        const double ventilation = 0.28 * std::cbrt(0.632)
            * std::sqrt(velocity_coefficient)
            * std::tgamma(0.5 * (velocity_exponent + 5.0));
        graupel_rate = 0.5 * thermodynamics.factor
            * thermodynamics.diffusion
            * thermodynamics.supersaturation_ice
            * thermodynamics.vapor_density_saturation * intercept
            * (0.86 * inverse_lambda * inverse_lambda
               + ventilation * viscosity_factor * density_factor_sqrt
                   * std::pow(
                       inverse_lambda,
                       0.5 * (velocity_exponent + 5.0)));
        graupel_rate = std::max(
            -state.graupel_kg_m3 / dt_s, graupel_rate);
    }
    double deposited_rate = ice_rate + snow_rate + graupel_rate;
    if ((deposited_rate > 1.0e-15 && deposited_rate > saturation_rate)
        || (deposited_rate < -1.0e-15
            && deposited_rate < saturation_rate))
    {
        const double factor = saturation_rate / deposited_rate;
        ice_rate *= factor;
        snow_rate *= factor;
        snow_vapor_rate *= factor;
        graupel_rate *= factor;
        number_rate *= factor;
        deposited_rate = saturation_rate;
    }
    IceDepositionExchange result{
        -deposited_rate,
        ice_rate,
        snow_rate,
        number_rate,
        latent_sublimation / heat_capacity
            * deposited_rate / state.dry_air_density_kg_m3,
        graupel_rate,
        snow_vapor_rate};
    const double water = result.vapor_kg_m3_s
        + result.ice_kg_m3_s + result.snow_kg_m3_s
        + result.graupel_kg_m3_s;
    if (!std::isfinite(water) || std::abs(water) > 2.0e-15
        || !std::isfinite(result.ice_number_m3_s)
        || !std::isfinite(result.temperature_k_s))
        throw std::runtime_error("invalid Thompson ice-deposition exchange");
    return result;
}

SnowRimingExchange snow_riming_rates(
    const SnowRimingState& state,
    double dt_s)
{
    require_state(state, dt_s);
    if (state.temperature_k >= freezing_temperature
        || state.cloud_kg_m3 <= minimum_active_content
        || state.cloud_number_m3 <= 2.0
        || (state.snow_kg_m3 <= minimum_active_content
            && state.graupel_kg_m3 < 1.0e-6))
        return {};

    const double cloud_diameter = cloud_mvd(
        state.cloud_kg_m3, state.cloud_number_m3);
    if (cloud_diameter <= minimum_cloud_diameter) return {};
    double snow_mass_rate = 0.0;
    double snow_number_rate = 0.0;
    double snow_diameter = 0.0;
    if (state.snow_kg_m3 > minimum_active_content)
    {
        const double temperature_c = std::min(
            -0.1, state.temperature_k - freezing_temperature);
        const double second = state.snow_kg_m3 / snow_mass_coefficient;
        const double third = snow_moment(second, temperature_c, 3.0);
        snow_diameter = third / second;
        if (snow_diameter > 300.0e-6)
        {
            const double efficiency = snow_cloud_collection_efficiency(
                snow_diameter, cloud_diameter);
            const double riming_moment = snow_moment(
                second, temperature_c, snow_velocity_exponent + 2.0);
            const double density_factor = std::sqrt(
                reference_air_density / state.dry_air_density_kg_m3);
            const double kernel = density_factor * pi * 0.25
                * snow_velocity_coefficient * efficiency * riming_moment;
            snow_mass_rate = state.cloud_kg_m3 * kernel;
            snow_number_rate = state.cloud_number_m3 * kernel;
        }
    }
    const auto graupel = graupel_riming_kernel(state, cloud_diameter);
    double graupel_mass_rate = graupel.mass_rate;
    double graupel_number_rate = graupel.number_rate;
    double mass_rate = snow_mass_rate + graupel_mass_rate;
    double number_rate = snow_number_rate + graupel_number_rate;
    const double donor_factor = std::min({
        1.0,
        mass_rate > 0.0
            ? state.cloud_kg_m3 / (dt_s * mass_rate) : 1.0,
        number_rate > 0.0
            ? state.cloud_number_m3 / (dt_s * number_rate) : 1.0});
    snow_mass_rate *= donor_factor;
    snow_number_rate *= donor_factor;
    graupel_mass_rate *= donor_factor;
    graupel_number_rate *= donor_factor;
    mass_rate = snow_mass_rate + graupel_mass_rate;
    number_rate = snow_number_rate + graupel_number_rate;
    double ice_mass_rate = 0.0;
    double ice_number_rate = 0.0;
    double snow_secondary_ice_rate = 0.0;
    double graupel_secondary_ice_rate = 0.0;
    double snow_fall_speed_multiplier = 1.0;
    if (graupel_mass_rate > 1.0e-15
        && state.temperature_k - freezing_temperature > -8.0)
    {
        const double celsius = state.temperature_k - freezing_temperature;
        double temperature_factor = 0.0;
        if (celsius >= -5.0 && celsius < -3.0)
            temperature_factor = 0.5 * (-3.0 - celsius);
        else if (celsius > -8.0 && celsius < -5.0)
            temperature_factor = (8.0 + celsius) / 3.0;
        ice_number_rate = 3.5e8 * temperature_factor * graupel_mass_rate;
        ice_mass_rate = 1.0e-12 * ice_number_rate;
        const double riming_total = snow_mass_rate + graupel_mass_rate;
        if (riming_total > 0.0)
        {
            snow_secondary_ice_rate =
                snow_mass_rate / riming_total * ice_mass_rate;
            graupel_secondary_ice_rate =
                graupel_mass_rate / riming_total * ice_mass_rate;
        }
    }
    if (snow_mass_rate > 2.0 * state.snow_deposition_kg_m3_s
        && state.snow_deposition_kg_m3_s > 1.0e-15
        && snow_diameter > 0.0)
    {
        const double ratio = std::min(
            30.0, snow_mass_rate / state.snow_deposition_kg_m3_s);
        snow_fall_speed_multiplier = std::min(
            1.5, 1.1 + (ratio - 2.0) * 0.014);
        double graupel_fraction = std::min(
            0.95, 0.15 + (ratio - 2.0) * 0.028);
        const double celsius = state.temperature_k - freezing_temperature;
        const double snow_velocity = snow_velocity_coefficient
            * std::pow(snow_diameter, snow_velocity_exponent)
            * std::exp(-snow_velocity_decay * snow_diameter);
        const double rime_coordinate = std::clamp(
            -(cloud_diameter * 0.5e6) * snow_velocity
                / std::min(-0.1, celsius),
            0.1, 10.0);
        const double rime_density =
            (0.051 + 0.114 * rime_coordinate
             - 0.0055 * rime_coordinate * rime_coordinate) * 1000.0;
        if (rime_density < 150.0) graupel_fraction = 0.0;
        const double converted = graupel_fraction * snow_mass_rate;
        snow_mass_rate -= converted;
        graupel_mass_rate += converted;
    }
    snow_mass_rate -= snow_secondary_ice_rate;
    graupel_mass_rate -= graupel_secondary_ice_rate;
    SnowRimingExchange result{
        -mass_rate,
        snow_mass_rate,
        -number_rate,
        latent_fusion / heat_capacity
            * mass_rate / state.dry_air_density_kg_m3,
        graupel_mass_rate,
        ice_mass_rate,
        ice_number_rate,
        snow_fall_speed_multiplier};
    if (!std::isfinite(result.cloud_kg_m3_s)
        || !std::isfinite(result.cloud_number_m3_s)
        || !std::isfinite(result.temperature_k_s)
        || !std::isfinite(result.ice_number_m3_s))
        throw std::runtime_error("invalid Thompson snow-riming exchange");
    return result;
}

IceSnowConversionExchange ice_to_snow_rates(
    const ColdProcessTables& tables,
    const IceSnowConversionState& state,
    double dt_s)
{
    require_state(state, dt_s);
    if (state.ice_kg_m3 <= minimum_active_content
        || state.ice_number_m3 <= 1.0e-6)
        return {};
    if (tables.ice_snow.size() != checked_ice_snow_size())
        throw std::invalid_argument(
            "Thompson ice-to-snow table dimension mismatch");
    const double ice_number = sorted_ice_number(
        state.ice_kg_m3, state.ice_number_m3);
    const double lambda        = std::cbrt(ice_mass_coefficient * 6.0 * ice_number / state.ice_kg_m3);
    const double mean_diameter = 4.0 / lambda;
    double       mass_rate     = 0.0;
    double       number_rate   = 0.0;
    const auto   ice_index     = decade_index(state.ice_kg_m3, -10, ColdProcessTableMetadata::ice_contents);
    const auto   number_index  = decade_index(ice_number, 0, ColdProcessTableMetadata::ice_numbers);
    if (ice_index + 1 == ColdProcessTableMetadata::ice_contents || mean_diameter > 5.0 * 300.0e-6)
    {
        mass_rate   = 0.99 * state.ice_kg_m3 / dt_s;
        number_rate = 0.95 * std::min(ice_number, state.ice_number_m3) / dt_s;
    }
    else if (mean_diameter >= 0.1 * 300.0e-6)
    {
        const double mass_weight =
            ice_index + 1 < ColdProcessTableMetadata::ice_contents
                ? std::clamp((state.ice_kg_m3 - ice_table_content(ice_index)) /
                                 (ice_table_content(ice_index + 1) - ice_table_content(ice_index)),
                             0.0,
                             1.0)
                : 0;
        const double number_weight =
            number_index + 1 < ColdProcessTableMetadata::ice_numbers
                ? std::clamp((ice_number - ice_table_number(number_index)) /
                                 (ice_table_number(number_index + 1) - ice_table_number(number_index)),
                             0.0,
                             1.0)
                : 0;
        IceSnowCell conversion;
        for (std::size_t n = 0; n < 2; ++n)
            for (std::size_t m = 0; m < 2; ++m)
            {
                const double weight = (n ? number_weight : 1 - number_weight) * (m ? mass_weight : 1 - mass_weight);
                if (weight == 0)
                    continue;
                const auto& cell = tables.ice_snow[flatten(IceSnowIndex{ice_index + m, number_index + n})];
                if (!std::isfinite(cell.snow_mass) || cell.snow_mass < 0 || !std::isfinite(cell.ice_number) ||
                    cell.ice_number < 0)
                    throw std::invalid_argument("invalid Thompson ice-to-snow table value");
                conversion.snow_mass += weight * cell.snow_mass;
                conversion.ice_number += weight * cell.ice_number;
            }
        mass_rate   = std::min(0.99 * state.ice_kg_m3 / dt_s, conversion.snow_mass / dt_s);
        number_rate = std::min(0.95 * std::min(ice_number, state.ice_number_m3) / dt_s, conversion.ice_number / dt_s);
    }
    return {-mass_rate, mass_rate, -number_rate};
}

} // namespace tmv::thompson
