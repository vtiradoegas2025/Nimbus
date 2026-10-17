#include "microphysics/schemes/thompson/collection_processes.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double rain_mass_coefficient = pi * 1000.0 / 6.0;

void require_nonnegative(double value, const char* message)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(message);
}

std::size_t decade_index(double value, int first_exponent)
{
    const double first = std::pow(10.0, first_exponent);
    if (value <= first) return 0;
    int exponent = static_cast<int>(std::floor(std::log10(value)));
    double scale = std::pow(10.0, exponent);
    double mantissa = value / scale;
    if (mantissa >= 10.0)
    {
        ++exponent;
        mantissa = 1.0;
    }
    const int digit = std::clamp(static_cast<int>(mantissa), 1, 9);
    const int index = 9 * (exponent - first_exponent) + digit - 1;
    return static_cast<std::size_t>(std::clamp(index, 0, 36));
}

double rain_intercept(double rain_water_kg_m3, double rain_number_m3)
{
    if (rain_water_kg_m3 <= 1.0e-6) return 1.0e10;
    const double slope = std::cbrt(
        rain_mass_coefficient * 6.0 * rain_number_m3 / rain_water_kg_m3);
    return rain_water_kg_m3 / (6.0 * rain_mass_coefficient)
        * std::pow(slope, 4.0);
}

double graupel_intercept(double graupel_water_kg_m3)
{
    const double logarithmic_content = std::log10(
        std::max(1.0e-9, graupel_water_kg_m3));
    return std::pow(10.0, std::clamp(
        3.0 + 2.0 / 7.0 * (logarithmic_content + 8.0), 2.0, 6.0));
}

}

RainSnowIndex rain_snow_lookup_index(double rain_water_kg_m3,
                                     double rain_number_m3,
                                     double snow_water_kg_m3,
                                     double wet_bulb_temperature_k)
{
    require_nonnegative(rain_water_kg_m3, "invalid Thompson rain content");
    require_nonnegative(rain_number_m3, "invalid Thompson rain number");
    require_nonnegative(snow_water_kg_m3, "invalid Thompson snow content");
    if (!std::isfinite(wet_bulb_temperature_k)
        || wet_bulb_temperature_k <= 0.0)
        throw std::invalid_argument("invalid Thompson wet-bulb temperature");
    const int truncated = static_cast<int>(
        (wet_bulb_temperature_k - 273.15 - 2.5) / 5.0) - 1;
    const std::size_t temperature = static_cast<std::size_t>(
        std::clamp(-truncated, 1, 9) - 1);
    return {
        decade_index(snow_water_kg_m3, -6),
        temperature,
        decade_index(rain_intercept(rain_water_kg_m3, rain_number_m3), 6),
        decade_index(rain_water_kg_m3, -6)};
}

RainGraupelIndex rain_graupel_lookup_index(double rain_water_kg_m3,
                                           double rain_number_m3,
                                           double graupel_water_kg_m3)
{
    require_nonnegative(rain_water_kg_m3, "invalid Thompson rain content");
    require_nonnegative(rain_number_m3, "invalid Thompson rain number");
    require_nonnegative(graupel_water_kg_m3,
                        "invalid Thompson graupel content");
    return {
        decade_index(graupel_intercept(graupel_water_kg_m3), 2),
        decade_index(graupel_water_kg_m3, -6),
        0,
        decade_index(rain_intercept(rain_water_kg_m3, rain_number_m3), 6),
        decade_index(rain_water_kg_m3, -6)};
}

CollectionProcessRates collection_process_rates(
    const CollectionTables& tables,
    double rain_water_kg_m3,
    double rain_number_m3,
    double snow_water_kg_m3,
    double graupel_water_kg_m3,
    double wet_bulb_temperature_k,
    double dt_s)
{
    require_nonnegative(rain_water_kg_m3, "invalid Thompson rain content");
    require_nonnegative(rain_number_m3, "invalid Thompson rain number");
    require_nonnegative(snow_water_kg_m3, "invalid Thompson snow content");
    require_nonnegative(graupel_water_kg_m3,
                        "invalid Thompson graupel content");
    if (!std::isfinite(dt_s) || dt_s <= 0.0)
        throw std::invalid_argument("invalid Thompson collection timestep");
    if (tables.rain_snow.size() != checked_rain_snow_size()
        || tables.rain_graupel.size() != checked_rain_graupel_size())
        throw std::invalid_argument("invalid Thompson collection tables");

    CollectionProcessRates result;
    if (rain_water_kg_m3 >= 1.0e-6 && snow_water_kg_m3 >= 1.0e-6)
    {
        const auto& v = tables.rain_snow[flatten(rain_snow_lookup_index(
            rain_water_kg_m3, rain_number_m3, snow_water_kg_m3,
            wet_bulb_temperature_k))].values;
        if (wet_bulb_temperature_k < 273.15)
        {
            result.rain_kg_m3_s -= v[3] + v[6] + v[1] + v[4];
            result.snow_kg_m3_s += v[3] + v[6] - v[0] - v[5];
            result.graupel_kg_m3_s += v[1] + v[4] + v[0] + v[5];
            result.rain_number_m3_s -= v[8] + v[9] + v[10] + v[11];
        }
        else
        {
            result.snow_kg_m3_s += -v[0] - v[5] + v[3] + v[6];
            result.rain_kg_m3_s -= result.snow_kg_m3_s;
        }
    }
    if (rain_water_kg_m3 >= 1.0e-6 && graupel_water_kg_m3 >= 1.0e-6)
    {
        const auto& v = tables.rain_graupel[flatten(
            rain_graupel_lookup_index(rain_water_kg_m3, rain_number_m3,
                                      graupel_water_kg_m3))];
        if (wet_bulb_temperature_k < 273.15)
        {
            const double frozen = std::min(
                rain_water_kg_m3 / dt_s,
                v.rain_mass_by_rain + v.rain_mass_by_graupel);
            result.rain_kg_m3_s -= frozen;
            result.graupel_kg_m3_s += frozen;
            result.rain_number_m3_s -= std::min(
                rain_number_m3 / dt_s,
                v.graupel_number_by_rain + v.rain_number_by_graupel);
        }
        else
        {
            const double melted = std::min(
                graupel_water_kg_m3 / dt_s, v.graupel_mass_by_rain);
            result.rain_kg_m3_s += melted;
            result.graupel_kg_m3_s -= melted;
            result.rain_number_m3_s += 1.5 * v.rain_number_by_graupel;
        }
    }
    double mass_scale = 1.0;
    if (result.rain_kg_m3_s < 0.0)
        mass_scale = std::min(
            mass_scale,
            rain_water_kg_m3 / (-result.rain_kg_m3_s * dt_s));
    if (result.snow_kg_m3_s < 0.0)
        mass_scale = std::min(
            mass_scale,
            snow_water_kg_m3 / (-result.snow_kg_m3_s * dt_s));
    if (result.graupel_kg_m3_s < 0.0)
        mass_scale = std::min(
            mass_scale,
            graupel_water_kg_m3 / (-result.graupel_kg_m3_s * dt_s));
    result.rain_kg_m3_s *= mass_scale;
    result.snow_kg_m3_s *= mass_scale;
    result.graupel_kg_m3_s *= mass_scale;
    result.rain_number_m3_s *= mass_scale;
    if (result.rain_number_m3_s < -rain_number_m3 / dt_s)
        result.rain_number_m3_s = -rain_number_m3 / dt_s;
    const double water_sum = result.rain_kg_m3_s
        + result.snow_kg_m3_s + result.graupel_kg_m3_s;
    if (!std::isfinite(result.rain_kg_m3_s)
        || !std::isfinite(result.snow_kg_m3_s)
        || !std::isfinite(result.graupel_kg_m3_s)
        || !std::isfinite(result.rain_number_m3_s)
        || !std::isfinite(water_sum) || std::abs(water_sum) > 1.0e-12)
        throw std::runtime_error("nonconservative Thompson collection rates");
    return result;
}

}
