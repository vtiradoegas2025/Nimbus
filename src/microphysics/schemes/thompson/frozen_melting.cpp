#include "microphysics/schemes/thompson/frozen_melting.hpp"

#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double freezing_temperature = 273.15;
constexpr double fusion_heat = 2.834e6 - 2.5e6;
constexpr double graupel_mass_coefficient = pi * 400.0 / 6.0;
constexpr double snow_mass_coefficient = 0.069;
constexpr double reference_density = 101325.0 / (287.05 * 298.0);
constexpr std::array<double, 10> snow_a = {
    5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
    0.31255, 0.000204, 0.003199, 0.0, -0.015952};
constexpr std::array<double, 10> snow_b = {
    0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
    0.060366, 0.000079, 0.000594, 0.0, -0.003577};

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

double snow_polynomial(const std::array<double, 10>& c,
                       double temperature_c,
                       double moment)
{
    const double t2 = temperature_c * temperature_c;
    const double m2 = moment * moment;
    return c[0] + c[1] * temperature_c + c[2] * moment
        + c[3] * temperature_c * moment + c[4] * t2 + c[5] * m2
        + c[6] * t2 * moment + c[7] * temperature_c * m2
        + c[8] * t2 * temperature_c + c[9] * m2 * moment;
}

double snow_moment(double reference_moment,
                   double temperature_c,
                   double order)
{
    return std::pow(10.0, snow_polynomial(snow_a, temperature_c, order))
        * std::pow(reference_moment,
                   snow_polynomial(snow_b, temperature_c, order));
}

}

FrozenMeltingRates frozen_melting_rates(
    double pressure_pa,
    double temperature_k,
    double dry_air_density_kg_m3,
    double vapor_kg_kg_dry,
    double snow_kg_m3,
    double graupel_kg_m3,
    double dt_s)
{
    require_positive(pressure_pa, "invalid Thompson melt pressure");
    require_positive(temperature_k, "invalid Thompson melt temperature");
    require_positive(dry_air_density_kg_m3,
                     "invalid Thompson melt density");
    require_nonnegative(vapor_kg_kg_dry, "invalid Thompson melt vapor");
    require_nonnegative(snow_kg_m3, "invalid Thompson snow content");
    require_nonnegative(graupel_kg_m3,
                        "invalid Thompson graupel content");
    require_positive(dt_s, "invalid Thompson melt timestep");

    FrozenMeltingRates result;
    const double celsius = temperature_k - freezing_temperature;
    if (celsius <= 0.0 || (snow_kg_m3 <= 1.0e-12
                           && graupel_kg_m3 <= 1.0e-12))
        return result;
    const double diffusivity = 2.11e-5
        * std::pow(temperature_k / freezing_temperature, 1.94)
        * (101325.0 / pressure_pa);
    const double viscosity = (1.718 + 0.0049 * celsius) * 1.0e-5;
    const double ventilation = std::sqrt(dry_air_density_kg_m3 / viscosity);
    const double density_factor_root = std::sqrt(std::sqrt(
        reference_density / dry_air_density_kg_m3));
    const double conductivity = (5.69 + 0.0168 * celsius)
        * 1.0e-5 * 418.936;
    const double saturation_at_freezing = liquid_saturation_mixing_ratio(
        pressure_pa, freezing_temperature);
    const double vapor_deficit = std::max(
        0.0, saturation_at_freezing - vapor_kg_kg_dry);
    const double wet_bulb = wet_bulb_temperature(
        pressure_pa, temperature_k, vapor_kg_kg_dry);
    const double thermal_driver = celsius * conductivity
        - 2.5e6 * diffusivity * vapor_deficit;

    if (snow_kg_m3 > 1.0e-12)
    {
        const double snow_temperature = std::min(-0.1, celsius);
        const double second = snow_kg_m3 / snow_mass_coefficient;
        const double zeroth = snow_moment(second, snow_temperature, 0.0);
        const double first = snow_moment(second, snow_temperature, 1.0);
        const double ventilation_moment = snow_moment(
            second, snow_temperature, 1.0 + (0.55 + 1.0) / 2.0);
        const double coefficient = pi * 4.0 * 0.15 / fusion_heat;
        double melt = thermal_driver * coefficient
            * (0.86 * first
               + 0.28 * std::cbrt(0.632) * std::sqrt(40.0)
                   * density_factor_root * ventilation
                   * ventilation_moment);
        melt = std::clamp(melt, 0.0, snow_kg_m3 / dt_s);
        result.snow_kg_m3_s = -melt;
        result.rain_kg_m3_s += melt;
        result.rain_number_m3_s += zeroth / snow_kg_m3 * melt
            * std::pow(10.0, -0.25 * (wet_bulb - freezing_temperature));
    }

    if (graupel_kg_m3 > 1.0e-12)
    {
        const double logarithmic_content = std::log10(
            std::max(1.0e-9, graupel_kg_m3));
        const double exponential_intercept = std::pow(
            10.0, std::clamp(
                3.0 + 2.0 / 7.0 * (logarithmic_content + 8.0),
                2.0, 6.0));
        const double exponential_slope = std::pow(
            exponential_intercept * graupel_mass_coefficient
                / graupel_kg_m3,
            0.25);
        const double slope = exponential_slope * std::cbrt(6.0);
        const double inverse_slope = 1.0 / slope;
        const double intercept = exponential_intercept
            / exponential_slope * slope;
        const double number = intercept / slope;
        double melt_intercept = intercept;
        if (graupel_kg_m3 * number < 1.0e-4)
            melt_intercept = 1.0e-4 / graupel_kg_m3 * slope;
        const double first = pi * 4.0 * 0.5 / fusion_heat
            * 0.86 * std::tgamma(2.0);
        const double second = pi * 4.0 * 0.5 / fusion_heat
            * 0.28 * std::cbrt(0.632) * std::sqrt(442.0)
            * std::tgamma(0.5 * (0.89 + 5.0));
        double melt = thermal_driver * melt_intercept
            * (first * std::pow(inverse_slope, 2.0)
               + second * density_factor_root * ventilation
                   * std::pow(inverse_slope,
                              0.5 * (0.89 + 5.0)));
        melt = std::clamp(melt, 0.0, graupel_kg_m3 / dt_s);
        result.graupel_kg_m3_s = -melt;
        result.rain_kg_m3_s += melt;
        result.rain_number_m3_s += melt * number / graupel_kg_m3
            * std::pow(10.0, -0.33 * (wet_bulb - freezing_temperature));
    }
    result.temperature_k_s = -fusion_heat
        / (1004.0 * (1.0 + 0.887 * vapor_kg_kg_dry))
        * result.rain_kg_m3_s / dry_air_density_kg_m3;
    const double water = result.rain_kg_m3_s + result.snow_kg_m3_s
        + result.graupel_kg_m3_s;
    if (!std::isfinite(result.rain_number_m3_s)
        || !std::isfinite(result.temperature_k_s)
        || std::abs(water) > 1.0e-12)
        throw std::runtime_error("invalid Thompson frozen melting rates");
    return result;
}

}
