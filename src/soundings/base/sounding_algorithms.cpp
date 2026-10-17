/**
 * @file sounding_algorithms.cpp
 * @brief Shared sounding QC, derivation, and interpolation helpers.
 *
 * Provides executable logic for the soundings runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/soundings subsystem.
 */

#include "soundings/base/sounding_algorithms.hpp"
#include "core/infra/physical_constants.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


namespace 
{

const double L_v = 2.5e6;
const double T0 = 273.15;

double quiet_nan()
{
    return std::numeric_limits<double>::quiet_NaN();
}

bool has_complete_optional_profile(const std::vector<double>& values, std::size_t expected_levels)
{
    return !values.empty() && values.size() == expected_levels;
}

std::pair<double, double> wind_components(double speed_ms, double direction_deg)
{
    const double direction_rad =
        direction_deg * physical_constants::pi / 180.0;
    return {
        -speed_ms * std::sin(direction_rad),
        -speed_ms * std::cos(direction_rad),
    };
}

std::pair<double, double> wind_polar(double u_ms, double v_ms)
{
    const double speed = std::hypot(u_ms, v_ms);
    if (speed == 0.0)
    {
        return {0.0, 0.0};
    }
    double direction = std::atan2(-u_ms, -v_ms) * 180.0
                     / physical_constants::pi;
    if (direction < 0.0)
    {
        direction += 360.0;
    }
    return {speed, direction};
}

/**
 * @brief Calculate potential temperature
 * @param temperature_k Temperature in Kelvin
 * @param pressure_hpa Pressure in hPa
 * @return Potential temperature in Kelvin
 */
double calculate_potential_temperature(double temperature_k, double pressure_hpa) {
    if (!std::isfinite(temperature_k) || !std::isfinite(pressure_hpa) || pressure_hpa <= 0.0)
    {
        return quiet_nan();
    }
    const double kappa = R_d / cp;
    const double p_pa = pressure_hpa * 100.0;
    if (!std::isfinite(p_pa) || p_pa <= 0.0)
    {
        return quiet_nan();
    }
    return temperature_k * std::pow(p0 / p_pa, kappa);
}

/**
 * @brief Calculate mixing ratio from dewpoint
 * @param dewpoint_k Dewpoint temperature in Kelvin
 * @param pressure_hpa Pressure in hPa
 * @return Mixing ratio in kg/kg
 */
double calculate_mixing_ratio(double dewpoint_k, double pressure_hpa) {
    if (!std::isfinite(dewpoint_k) || !std::isfinite(pressure_hpa) || pressure_hpa <= 0.0)
    {
        return quiet_nan();
    }

    const double e_s = 6.112 * std::exp(17.67 * (dewpoint_k - T0) / (dewpoint_k - 29.65));
    const double denom_hpa = pressure_hpa - e_s;
    if (!std::isfinite(e_s) || !std::isfinite(denom_hpa) || denom_hpa <= 0.0)
    {
        return quiet_nan();
    }

    return (0.62198 * e_s) / denom_hpa;
}

double calculate_equivalent_potential_temperature_from_mixing(
    double temperature_k,
    double mixing_ratio_kgkg,
    double pressure_hpa)
{
    if (!std::isfinite(temperature_k) || temperature_k <= 0.0
        || !std::isfinite(mixing_ratio_kgkg) || mixing_ratio_kgkg < 0.0
        || !std::isfinite(pressure_hpa) || pressure_hpa <= 0.0)
    {
        return quiet_nan();
    }
    const double pressure_pa = pressure_hpa * 100.0;
    const double denominator = cp + L_v * mixing_ratio_kgkg / temperature_k;
    if (!std::isfinite(denominator) || denominator <= 0.0)
    {
        return quiet_nan();
    }
    const double lifted_temperature =
        temperature_k * std::pow(p0 / pressure_pa, R_d / denominator);
    if (!std::isfinite(lifted_temperature) || lifted_temperature <= 0.0)
    {
        return quiet_nan();
    }
    return lifted_temperature
         * std::exp(L_v * mixing_ratio_kgkg / (cp * lifted_temperature));
}

}

/**
 * @brief Linear interpolation between two points
 * @param x1 First x value
 * @param y1 First y value
 * @param x2 Second x value
 * @param y2 Second y value
 * @param x_target Target x value
 * @return Interpolated y value
 */
double linear_interpolate(double x1, double y1, double x2, double y2, double x_target) {
    if (x1 == x2) return y1;
    return y1 + (y2 - y1) * (x_target - x1) / (x2 - x1);
}

/**
 * @brief Quality control for sounding data
 * @param data Sounding data to validate
 * @param config Configuration with QC thresholds
 * @return true if data passes QC
 */
bool quality_control_sounding(SoundingData& data, const SoundingConfig& config) 
{
    if (!data.is_valid()) 
    {
        tmv::log_error("Sounding data is invalid - missing basic fields");
        return false;
    }

    const std::size_t original_size = data.num_levels();
    if (!data.dewpoint_k.empty() && data.dewpoint_k.size() != original_size)
    {
        tmv::log_warn("Dewpoint profile length (", data.dewpoint_k.size(),
                      ") does not match required profile length (", original_size,
                      "); discarding dewpoint values.");
        data.dewpoint_k.clear();
    }

    if (!data.mixing_ratio_kgkg.empty()
        && data.mixing_ratio_kgkg.size() != original_size)
    {
        tmv::log_warn("Mixing-ratio profile length (",
                      data.mixing_ratio_kgkg.size(),
                      ") does not match required profile length (",
                      original_size, "); discarding mixing-ratio values.");
        data.mixing_ratio_kgkg.clear();
    }

    const bool have_any_polar_wind =
        !data.wind_speed_ms.empty() || !data.wind_direction_deg.empty();
    const bool have_complete_polar_wind =
        data.wind_speed_ms.size() == original_size &&
        data.wind_direction_deg.size() == original_size &&
        !data.wind_speed_ms.empty() &&
        !data.wind_direction_deg.empty();
    if (have_any_polar_wind && !have_complete_polar_wind)
    {
        tmv::log_warn("Polar wind profile lengths (speed=", data.wind_speed_ms.size(),
                      ", direction=", data.wind_direction_deg.size(),
                      ") do not match required profile length (", original_size,
                      "); discarding wind values.");
        data.wind_speed_ms.clear();
        data.wind_direction_deg.clear();
    }

    const bool have_any_component_wind =
        !data.wind_u_ms.empty() || !data.wind_v_ms.empty();
    const bool have_complete_component_wind =
        data.wind_u_ms.size() == original_size
        && data.wind_v_ms.size() == original_size
        && !data.wind_u_ms.empty() && !data.wind_v_ms.empty();
    if (have_any_component_wind && !have_complete_component_wind)
    {
        tmv::log_warn("Cartesian wind profile lengths (u=", data.wind_u_ms.size(),
                      ", v=", data.wind_v_ms.size(),
                      ") do not match required profile length (", original_size,
                      "); discarding component winds.");
        data.wind_u_ms.clear();
        data.wind_v_ms.clear();
    }

    const bool has_dewpoint_profile = has_complete_optional_profile(data.dewpoint_k, original_size);
    const bool has_mixing_profile =
        has_complete_optional_profile(data.mixing_ratio_kgkg, original_size);
    const bool has_polar_wind_profile =
        has_complete_optional_profile(data.wind_speed_ms, original_size) &&
        has_complete_optional_profile(data.wind_direction_deg, original_size);
    const bool has_component_wind_profile =
        has_complete_optional_profile(data.wind_u_ms, original_size)
        && has_complete_optional_profile(data.wind_v_ms, original_size);
    const bool has_primary_theta =
        data.potential_temperature_is_primary
        && has_complete_optional_profile(
            data.potential_temperature_k, original_size);
    if (data.potential_temperature_is_primary && !has_primary_theta)
    {
        tmv::log_error(
            "Primary potential-temperature profile is incomplete");
        return false;
    }

    std::vector<size_t> valid_indices;

    for (size_t i = 0; i < data.num_levels(); ++i) 
    {
        bool valid_level = true;

        if (!std::isfinite(data.height_m[i]) ||
            !std::isfinite(data.pressure_hpa[i]) ||
            !std::isfinite(data.temperature_k[i]))
        {
            valid_level = false;
        }

        if (data.pressure_hpa[i] < config.min_pressure_hpa ||
            data.pressure_hpa[i] > config.max_pressure_hpa) 
        {
            valid_level = false;
        }

        if (data.temperature_k[i] < config.min_temperature_k ||
            data.temperature_k[i] > config.max_temperature_k) 
        {
            valid_level = false;
        }

        if (has_mixing_profile
            && (!std::isfinite(data.mixing_ratio_kgkg[i])
                || data.mixing_ratio_kgkg[i] < 0.0))
        {
            valid_level = false;
        }
        if (has_dewpoint_profile && !std::isfinite(data.dewpoint_k[i]))
        {
            valid_level = false;
        }
        if (has_primary_theta
            && (!std::isfinite(data.potential_temperature_k[i])
                || data.potential_temperature_k[i] <= 0.0))
        {
            valid_level = false;
        }
        if (has_component_wind_profile
            && (!std::isfinite(data.wind_u_ms[i])
                || !std::isfinite(data.wind_v_ms[i])))
        {
            valid_level = false;
        }
        if (!has_component_wind_profile && has_polar_wind_profile
            && (!std::isfinite(data.wind_speed_ms[i])
                || data.wind_speed_ms[i] < 0.0
                || !std::isfinite(data.wind_direction_deg[i])))
        {
            valid_level = false;
        }

        if (i > 0 && data.height_m[i] <= data.height_m[i-1]) 
        {
            valid_level = false;
        }
        if (i > 0 && data.pressure_hpa[i] >= data.pressure_hpa[i - 1])
        {
            valid_level = false;
        }

        if (valid_level) 
        {
            valid_indices.push_back(i);
        }
    }

    if (valid_indices.size() < 5) 
    {
        tmv::log_error("Insufficient valid levels in sounding: ", valid_indices.size(),
                       "/", original_size);
        return false;
    }

    SoundingData filtered_data = data;
    filtered_data.clear();
    filtered_data.potential_temperature_is_primary = has_primary_theta;

    for (size_t idx : valid_indices) 
    {
        filtered_data.height_m.push_back(data.height_m[idx]);
        filtered_data.pressure_hpa.push_back(data.pressure_hpa[idx]);
        filtered_data.temperature_k.push_back(data.temperature_k[idx]);

        if (has_dewpoint_profile) 
        {
            filtered_data.dewpoint_k.push_back(data.dewpoint_k[idx]);
        }

        if (has_mixing_profile)
        {
            filtered_data.mixing_ratio_kgkg.push_back(
                data.mixing_ratio_kgkg[idx]);
        }
        if (has_primary_theta)
        {
            filtered_data.potential_temperature_k.push_back(
                data.potential_temperature_k[idx]);
        }

        if (has_polar_wind_profile)
        {
            filtered_data.wind_speed_ms.push_back(data.wind_speed_ms[idx]);
            filtered_data.wind_direction_deg.push_back(data.wind_direction_deg[idx]);
        }
        if (has_component_wind_profile)
        {
            filtered_data.wind_u_ms.push_back(data.wind_u_ms[idx]);
            filtered_data.wind_v_ms.push_back(data.wind_v_ms[idx]);
        }
    }

    data = filtered_data;

    tmv::log_info("Sounding quality control: ", data.num_levels(),
                  " valid levels retained from ", original_size);

    return true;
}

/**
 * @brief Calculate derived thermodynamic quantities
 * @param data Sounding data with basic fields
 */
void calculate_derived_quantities(SoundingData& data) 
{
    if (!data.is_valid())
    {
        data.potential_temperature_k.clear();
        data.equivalent_potential_temperature_k.clear();
        data.mixing_ratio_kgkg.clear();
        return;
    }

    const std::size_t levels = data.num_levels();
    const bool preserve_primary_theta =
        data.potential_temperature_is_primary
        && has_complete_optional_profile(data.potential_temperature_k, levels);
    if (!preserve_primary_theta)
    {
        data.potential_temperature_is_primary = false;
        data.potential_temperature_k.assign(levels, quiet_nan());
        for (size_t i = 0; i < levels; ++i)
        {
            data.potential_temperature_k[i] = calculate_potential_temperature(
                data.temperature_k[i], data.pressure_hpa[i]);
        }
    }

    const bool has_complete_dewpoint_profile = has_complete_optional_profile(data.dewpoint_k, levels);
    const bool has_complete_mixing_profile =
        has_complete_optional_profile(data.mixing_ratio_kgkg, levels);
    if (!has_complete_dewpoint_profile && !has_complete_mixing_profile)
    {
        data.mixing_ratio_kgkg.clear();
        data.equivalent_potential_temperature_k.clear();
        return;
    }

    data.equivalent_potential_temperature_k.assign(levels, quiet_nan());
    if (!has_complete_mixing_profile)
    {
        data.mixing_ratio_kgkg.assign(levels, quiet_nan());
    }

    for (size_t i = 0; i < levels; ++i)
    {
        if (!has_complete_mixing_profile)
        {
            data.mixing_ratio_kgkg[i] = calculate_mixing_ratio(
                data.dewpoint_k[i], data.pressure_hpa[i]);
        }
        data.equivalent_potential_temperature_k[i] =
            calculate_equivalent_potential_temperature_from_mixing(
                data.temperature_k[i],
                data.mixing_ratio_kgkg[i],
                data.pressure_hpa[i]);
        if (!std::isfinite(data.mixing_ratio_kgkg[i]))
        {
            data.equivalent_potential_temperature_k[i] = quiet_nan();
        }
    }
}

/**
 * @brief Interpolate sounding data to target heights using linear interpolation
 * @param source Source sounding data
 * @param target_heights Target heights in meters
 * @param config Sounding configuration
 * @return Interpolated sounding data
 */
SoundingData interpolate_sounding_linear(
    const SoundingData& source,
    const std::vector<double>& target_heights,
    const SoundingConfig& config) {

    if (!source.is_valid())
    {
        throw std::runtime_error("Source sounding is invalid for interpolation");
    }

    SoundingData result;
    result.height_m = target_heights;
    const std::size_t source_levels = source.height_m.size();
    const bool has_dewpoint_profile = has_complete_optional_profile(source.dewpoint_k, source_levels);
    const bool has_mixing_profile =
        has_complete_optional_profile(source.mixing_ratio_kgkg, source_levels);
    const bool has_primary_theta =
        source.potential_temperature_is_primary
        && has_complete_optional_profile(
            source.potential_temperature_k, source_levels);
    const bool has_component_wind_profile =
        has_complete_optional_profile(source.wind_u_ms, source_levels)
        && has_complete_optional_profile(source.wind_v_ms, source_levels);
    const bool has_polar_wind_profile =
        has_complete_optional_profile(source.wind_speed_ms, source_levels) &&
        has_complete_optional_profile(source.wind_direction_deg, source_levels);
    const bool has_wind_profile =
        has_component_wind_profile || has_polar_wind_profile;
    if (!source.dewpoint_k.empty() && !has_dewpoint_profile)
    {
        tmv::log_warn("Ignoring incomplete dewpoint profile during interpolation (size=",
                      source.dewpoint_k.size(), ", expected=", source_levels, ").");
    }
    if ((!source.wind_speed_ms.empty() || !source.wind_direction_deg.empty()
         || !source.wind_u_ms.empty() || !source.wind_v_ms.empty())
        && !has_wind_profile)
    {
        tmv::log_warn("Ignoring incomplete wind profile during interpolation (speed=",
                      source.wind_speed_ms.size(), ", direction=",
                      source.wind_direction_deg.size(), ", u=",
                      source.wind_u_ms.size(), ", v=", source.wind_v_ms.size(),
                      ", expected=", source_levels, ").");
    }

    result.pressure_hpa.resize(target_heights.size());
    result.temperature_k.resize(target_heights.size());
    result.potential_temperature_k.resize(target_heights.size());
    result.potential_temperature_is_primary = has_primary_theta;

    if (has_dewpoint_profile)
    {
        result.dewpoint_k.resize(target_heights.size());
    }
    if (has_mixing_profile)
    {
        result.mixing_ratio_kgkg.resize(target_heights.size());
    }

    if (has_wind_profile) {
        result.wind_u_ms.resize(target_heights.size());
        result.wind_v_ms.resize(target_heights.size());
        result.wind_speed_ms.resize(target_heights.size());
        result.wind_direction_deg.resize(target_heights.size());
    }

    for (size_t i = 0; i < target_heights.size(); ++i) 
    {
        double target_h = target_heights[i];

        auto it = std::lower_bound(source.height_m.begin(), source.height_m.end(), target_h);

        auto assign_optional_linear = [&](size_t idx1, size_t idx2, bool use_linear)
        {
            if (has_dewpoint_profile)
            {
                result.dewpoint_k[i] = use_linear
                    ? linear_interpolate(
                        source.height_m[idx1], source.dewpoint_k[idx1],
                        source.height_m[idx2], source.dewpoint_k[idx2], target_h)
                    : source.dewpoint_k[idx1];
            }

            if (has_mixing_profile)
            {
                result.mixing_ratio_kgkg[i] = use_linear
                    ? linear_interpolate(
                        source.height_m[idx1], source.mixing_ratio_kgkg[idx1],
                        source.height_m[idx2], source.mixing_ratio_kgkg[idx2],
                        target_h)
                    : source.mixing_ratio_kgkg[idx1];
            }
            if (has_primary_theta)
            {
                result.potential_temperature_k[i] = use_linear
                    ? linear_interpolate(
                        source.height_m[idx1],
                        source.potential_temperature_k[idx1],
                        source.height_m[idx2],
                        source.potential_temperature_k[idx2],
                        target_h)
                    : source.potential_temperature_k[idx1];
            }

            if (has_wind_profile)
            {
                const auto component_at = [&](std::size_t idx)
                {
                    if (has_component_wind_profile)
                    {
                        return std::pair<double, double>{
                            source.wind_u_ms[idx], source.wind_v_ms[idx]};
                    }
                    return wind_components(
                        source.wind_speed_ms[idx],
                        source.wind_direction_deg[idx]);
                };
                const auto first = component_at(idx1);
                const auto second = component_at(idx2);
                result.wind_u_ms[i] = use_linear
                    ? linear_interpolate(
                        source.height_m[idx1], first.first,
                        source.height_m[idx2], second.first, target_h)
                    : first.first;
                result.wind_v_ms[i] = use_linear
                    ? linear_interpolate(
                        source.height_m[idx1], first.second,
                        source.height_m[idx2], second.second, target_h)
                    : first.second;
                const auto polar =
                    wind_polar(result.wind_u_ms[i], result.wind_v_ms[i]);
                result.wind_speed_ms[i] = polar.first;
                result.wind_direction_deg[i] = polar.second;
            }
        };

        if (it == source.height_m.begin()) 
        {
            if (config.extrapolate_below_ground) 
            {
                if (source.height_m.size() > 1)
                {
                    const size_t idx1 = 0;
                    const size_t idx2 = 1;
                    result.pressure_hpa[i] = linear_interpolate(
                        source.height_m[idx1], source.pressure_hpa[idx1],
                        source.height_m[idx2], source.pressure_hpa[idx2], target_h);
                    result.temperature_k[i] = linear_interpolate(
                        source.height_m[idx1], source.temperature_k[idx1],
                        source.height_m[idx2], source.temperature_k[idx2], target_h);
                    assign_optional_linear(idx1, idx2, true);
                } 
                else 
                {
                    result.pressure_hpa[i] = source.pressure_hpa[0];
                    result.temperature_k[i] = source.temperature_k[0];
                    assign_optional_linear(0, 0, false);
                }
            } 
            else 
            {
                result.pressure_hpa[i] = source.pressure_hpa[0];
                result.temperature_k[i] = source.temperature_k[0];
                assign_optional_linear(0, 0, false);
            }
        } 
        else if (it == source.height_m.end()) 
        {
            if (config.extrapolate_above_top) 
            {
                if (source.height_m.size() > 1) 
                {
                    const size_t idx2 = source.height_m.size() - 1;
                    const size_t idx1 = idx2 - 1;
                    result.pressure_hpa[i] = linear_interpolate(
                        source.height_m[idx1], source.pressure_hpa[idx1],
                        source.height_m[idx2], source.pressure_hpa[idx2], target_h);
                    result.temperature_k[i] = linear_interpolate(
                        source.height_m[idx1], source.temperature_k[idx1],
                        source.height_m[idx2], source.temperature_k[idx2], target_h);
                    assign_optional_linear(idx1, idx2, true);
                } 
                else 
                {
                    result.pressure_hpa[i] = source.pressure_hpa.back();
                    result.temperature_k[i] = source.temperature_k.back();
                    assign_optional_linear(source.height_m.size() - 1, source.height_m.size() - 1, false);
                }
            } 
            else 
            {
                result.pressure_hpa[i] = source.pressure_hpa.back();
                result.temperature_k[i] = source.temperature_k.back();
                assign_optional_linear(source.height_m.size() - 1, source.height_m.size() - 1, false);
            }
        } 
        else 
        {
            size_t idx2 = it - source.height_m.begin();
            size_t idx1 = idx2 - 1;

            result.pressure_hpa[i] = linear_interpolate(
                source.height_m[idx1], source.pressure_hpa[idx1],
                source.height_m[idx2], source.pressure_hpa[idx2], target_h);
            result.temperature_k[i] = linear_interpolate(
                source.height_m[idx1], source.temperature_k[idx1],
                source.height_m[idx2], source.temperature_k[idx2], target_h);

            assign_optional_linear(idx1, idx2, true);
        }
    }

    calculate_derived_quantities(result);

    return result;
}
