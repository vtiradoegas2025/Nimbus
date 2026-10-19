#include "triple_cloud_adjustment.hpp"
#include "activation.hpp"
#include "microphysics/base/thermodynamics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr double latent_temperature_factor = microphysics_constants::L_v / microphysics_constants::cp;

bool fp32_range(double value) noexcept
{
    const double magnitude = std::abs(value);
    return std::isfinite(value) && magnitude <= std::numeric_limits<float>::max() &&
        (value == 0.0 || magnitude >= std::numeric_limits<float>::denorm_min());
}

bool same_bits(double first, double second) noexcept
{
    return std::memcmp(&first, &second, sizeof(double)) == 0;
}

template <std::size_t Size>
bool same_values(const std::array<double, Size>& first, const std::array<double, Size>& second) noexcept
{
    for (std::size_t index = 0; index < Size; ++index)
    {
        if (!same_bits(first[index], second[index]))
        {
            return false;
        }
    }
    return true;
}

std::array<double, 7> source_values(const TripleCloudAdjustmentCell& cell) noexcept
{
    return {cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
            cell.vapor_per_kg, cell.cloud_per_kg, cell.cloud_number_per_kg,
            cell.vertical_velocity_ms};
}

std::array<double, 7> endpoint_values(const TripleCloudAdjustmentEndpoints& endpoints) noexcept
{
    return {endpoints.vapor_per_kg, endpoints.cloud_per_kg, endpoints.cloud_number_per_kg,
            endpoints.temperature_increment_k, endpoints.temperature_after_k,
            endpoints.admitted_cloud_increment_per_kg, endpoints.activated_number_per_m3};
}

bool journal_close(double initial, double expected, double actual) noexcept
{
    const double scale = std::max({std::abs(initial), std::abs(expected), std::abs(actual), 1.0e-30});
    return std::isfinite(expected) && std::isfinite(actual) &&
        std::abs(expected - actual) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool source_valid(const TripleCloudAdjustmentCell& cell, double duration_s, std::string& error)
{
    if (!std::isfinite(duration_s) || duration_s <= 0.0 ||
        !std::isfinite(cell.temperature_k) || cell.temperature_k <= microphysics_constants::T0 ||
        cell.temperature_k >= 400.0 || !std::isfinite(cell.pressure_pa) ||
        cell.pressure_pa < 100.0 || cell.pressure_pa > 200000.0 ||
        !std::isfinite(cell.dry_density_kg_m3) || cell.dry_density_kg_m3 <= 0.0 ||
        !std::isfinite(cell.vapor_per_kg) || cell.vapor_per_kg < 0.0 ||
        !std::isfinite(cell.cloud_per_kg) || cell.cloud_per_kg < 0.0 ||
        !std::isfinite(cell.cloud_number_per_kg) || cell.cloud_number_per_kg < 0.0 ||
        !std::isfinite(cell.vertical_velocity_ms) ||
        (cell.cloud_per_kg > 0.0 && cell.cloud_number_per_kg <= 0.0))
    {
        error = "exact triple cloud adjustment source or duration is invalid";
        return false;
    }
    for (double value : source_values(cell))
    {
        if (!fp32_range(value))
        {
            error = "exact triple cloud adjustment source is not FP32 representable";
            return false;
        }
    }
    return true;
}

double saturation_residual(const TripleCloudAdjustmentCell& cell, double magnitude, bool condensation)
{
    const double increment = condensation ? magnitude : -magnitude;
    const double temperature = cell.temperature_k + latent_temperature_factor * increment;
    return cell.vapor_per_kg - increment -
        thermodynamics::saturation_mixing_ratio_water(temperature, cell.pressure_pa);
}

double bracketed_transfer(const TripleCloudAdjustmentCell& cell, double upper, bool condensation)
{
    double lower = 0.0;
    double lower_residual = saturation_residual(cell, lower, condensation);
    double upper_residual = saturation_residual(cell, upper, condensation);
    if (upper_residual == 0.0)
    {
        return upper;
    }
    for (int iteration = 0; iteration < 96; ++iteration)
    {
        if (std::nextafter(lower, upper) >= upper)
        {
            break;
        }
        const double middle = lower + 0.5 * (upper - lower);
        if (middle == lower || middle == upper)
        {
            break;
        }
        const double residual = saturation_residual(cell, middle, condensation);
        if (residual == 0.0)
        {
            return middle;
        }
        if ((condensation && residual > 0.0) || (!condensation && residual < 0.0))
        {
            lower = middle;
            lower_residual = residual;
        }
        else
        {
            upper = middle;
            upper_residual = residual;
        }
    }
    return std::abs(lower_residual) < std::abs(upper_residual) ? lower : upper;
}

bool admitted_increment(const TripleCloudAdjustmentCell& cell, double& increment, std::string& error)
{
    const double saturation = thermodynamics::saturation_mixing_ratio_water(cell.temperature_k, cell.pressure_pa);
    increment = 0.0;
    if (cell.vapor_per_kg > saturation)
    {
        const double maximum = std::min(cell.vapor_per_kg,
            (400.0 - cell.temperature_k) / latent_temperature_factor);
        if (saturation_residual(cell, maximum, true) > 0.0)
        {
            error = "exact triple cloud adjustment condensation cannot close below the thermal bound";
            return false;
        }
        increment = bracketed_transfer(cell, maximum, true);
    }
    else if (cell.vapor_per_kg < saturation && cell.cloud_per_kg > 0.0)
    {
        const double maximum = std::min(cell.cloud_per_kg,
            (cell.temperature_k - microphysics_constants::T0) / latent_temperature_factor);
        if (saturation_residual(cell, maximum, false) < 0.0)
        {
            if (maximum != cell.cloud_per_kg)
            {
                error = "exact triple cloud adjustment evaporation cannot close above the warm-phase bound";
                return false;
            }
            increment = -cell.cloud_per_kg;
        }
        else
        {
            increment = -bracketed_transfer(cell, maximum, false);
        }
    }
    return true;
}

bool endpoints_valid(const TripleCloudAdjustmentCell& cell, double duration_s,
                     const TripleCloudAdjustmentEndpoints& endpoints, std::string& error)
{
    for (double value : endpoint_values(endpoints))
    {
        if (!fp32_range(value))
        {
            error = "exact triple cloud adjustment endpoint is not FP32 representable";
            return false;
        }
    }
    const double increment = endpoints.admitted_cloud_increment_per_kg;
    const double source_saturation = thermodynamics::saturation_mixing_ratio_water(
        cell.temperature_k, cell.pressure_pa);
    const bool condensation = cell.vapor_per_kg > source_saturation;
    const bool evaporation = cell.vapor_per_kg < source_saturation && cell.cloud_per_kg > 0.0;
    if ((condensation && increment <= 0.0) || (evaporation && increment >= 0.0) ||
        (!condensation && !evaporation && increment != 0.0) ||
        increment > cell.vapor_per_kg || -increment > cell.cloud_per_kg ||
        endpoints.vapor_per_kg < 0.0 || endpoints.cloud_per_kg < 0.0 ||
        endpoints.cloud_number_per_kg < 0.0 || endpoints.activated_number_per_m3 < 0.0 ||
        (endpoints.cloud_per_kg > 0.0 && endpoints.cloud_number_per_kg <= 0.0) ||
        endpoints.temperature_after_k <= microphysics_constants::T0 || endpoints.temperature_after_k >= 400.0 ||
        endpoints.vapor_per_kg != cell.vapor_per_kg - increment ||
        endpoints.cloud_per_kg != cell.cloud_per_kg + increment ||
        endpoints.temperature_increment_k != latent_temperature_factor * increment ||
        endpoints.temperature_after_k != cell.temperature_k + endpoints.temperature_increment_k ||
        endpoints.cloud_donor_exhausted != (increment < 0.0 && -increment == cell.cloud_per_kg))
    {
        error = "exact triple cloud adjustment branch, budget or warm-phase endpoint differs";
        return false;
    }
    double expected_number = cell.cloud_number_per_kg;
    double activation_target = 0.0;
    if (condensation)
    {
        activation_target = continental_activated_ccn(
            cell.vertical_velocity_ms, endpoints.temperature_after_k, cell.pressure_pa);
        expected_number = std::max(cell.cloud_number_per_kg, activation_target / cell.dry_density_kg_m3);
    }
    else if (evaporation)
    {
        expected_number = cell.cloud_number_per_kg * (endpoints.cloud_per_kg / cell.cloud_per_kg);
    }
    if (!same_bits(endpoints.cloud_number_per_kg, expected_number) ||
        !same_bits(endpoints.activated_number_per_m3, activation_target) ||
        (endpoints.cloud_donor_exhausted &&
         (endpoints.cloud_per_kg != 0.0 || endpoints.cloud_number_per_kg != 0.0)))
    {
        error = "exact triple cloud adjustment activation or evaporative number closure differs";
        return false;
    }
    const double saturation_after = thermodynamics::saturation_mixing_ratio_water(
        endpoints.temperature_after_k, cell.pressure_pa);
    const double saturation_tolerance = 8.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(endpoints.vapor_per_kg), std::abs(saturation_after)});
    const bool exhausted_subsaturation = endpoints.cloud_donor_exhausted &&
        endpoints.vapor_per_kg < saturation_after;
    if ((condensation || evaporation) && !exhausted_subsaturation &&
        std::abs(endpoints.vapor_per_kg - saturation_after) > saturation_tolerance)
    {
        error = "exact triple cloud adjustment liquid saturation does not close";
        return false;
    }
    if (!journal_close(cell.vapor_per_kg + cell.cloud_per_kg,
                       cell.vapor_per_kg + cell.cloud_per_kg,
                       endpoints.vapor_per_kg + endpoints.cloud_per_kg) ||
        !journal_close(0.0, latent_temperature_factor * increment, endpoints.temperature_increment_k) ||
        !journal_close(cell.temperature_k, cell.temperature_k + latent_temperature_factor * increment,
                       endpoints.temperature_after_k))
    {
        error = "exact triple cloud adjustment water or latent heat journal does not close";
        return false;
    }
    for (double rate : {increment / duration_s, endpoints.temperature_increment_k / duration_s,
                        (endpoints.cloud_number_per_kg - cell.cloud_number_per_kg) / duration_s})
    {
        if (!fp32_range(rate))
        {
            error = "exact triple cloud adjustment diagnostic rate is not FP32 representable";
            return false;
        }
    }
    return true;
}

} // namespace

bool produce_exact_triple_cloud_adjustment(
    const TripleCloudAdjustmentCell& cell, double duration_s,
    TripleCloudAdjustmentResult& producer, std::string& error)
{
    if (!source_valid(cell, duration_s, error))
    {
        return false;
    }
    double increment = 0.0;
    if (!admitted_increment(cell, increment, error))
    {
        return false;
    }
    TripleCloudAdjustmentResult candidate;
    candidate.source_ = cell;
    candidate.duration_s_ = duration_s;
    auto& endpoints = candidate.endpoints_;
    endpoints.vapor_per_kg = cell.vapor_per_kg - increment;
    endpoints.cloud_per_kg = cell.cloud_per_kg + increment;
    endpoints.cloud_number_per_kg = cell.cloud_number_per_kg;
    endpoints.admitted_cloud_increment_per_kg = increment;
    endpoints.temperature_increment_k = latent_temperature_factor * increment;
    endpoints.temperature_after_k = cell.temperature_k + endpoints.temperature_increment_k;
    endpoints.cloud_donor_exhausted = increment < 0.0 && -increment == cell.cloud_per_kg;
    if (increment > 0.0)
    {
        endpoints.activated_number_per_m3 = continental_activated_ccn(
            cell.vertical_velocity_ms, endpoints.temperature_after_k, cell.pressure_pa);
        endpoints.cloud_number_per_kg = std::max(
            cell.cloud_number_per_kg, endpoints.activated_number_per_m3 / cell.dry_density_kg_m3);
    }
    else if (increment < 0.0)
    {
        endpoints.cloud_number_per_kg = cell.cloud_number_per_kg * (endpoints.cloud_per_kg / cell.cloud_per_kg);
    }
    if (!endpoints_valid(cell, duration_s, endpoints, error))
    {
        return false;
    }
    producer = candidate;
    error.clear();
    return true;
}

bool apply_exact_triple_cloud_adjustment(
    const TripleCloudAdjustmentCell& cell, double duration_s,
    const TripleCloudAdjustmentResult& producer,
    TripleCloudAdjustmentResult& result, std::string& error)
{
    if (!source_valid(cell, duration_s, error))
    {
        return false;
    }
    if (!same_values(source_values(cell), source_values(producer.source_cell())) ||
        !same_bits(duration_s, producer.duration_s()))
    {
        error = "exact triple cloud adjustment producer source or duration provenance differs";
        return false;
    }
    TripleCloudAdjustmentResult expected;
    if (!produce_exact_triple_cloud_adjustment(cell, duration_s, expected, error))
    {
        return false;
    }
    if (!same_values(endpoint_values(producer.endpoints()), endpoint_values(expected.endpoints())) ||
        producer.endpoints().cloud_donor_exhausted != expected.endpoints().cloud_donor_exhausted)
    {
        error = "exact triple cloud adjustment producer endpoint provenance differs";
        return false;
    }
    result = producer;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
