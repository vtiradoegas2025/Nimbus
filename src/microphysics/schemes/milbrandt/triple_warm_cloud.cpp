#include "triple_warm_cloud.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

template <typename Value>
bool same_bits(Value first, Value second) noexcept
{
    return std::memcmp(&first, &second, sizeof(Value)) == 0;
}

template <typename Value, std::size_t Size>
bool same_values(const std::array<Value, Size>& first,
                 const std::array<Value, Size>& second) noexcept
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

std::array<double, 17> parameter_values(const WarmRainParameters& p) noexcept
{
    return {p.alpha_r, p.c_r, p.d_r, p.a_r, p.b_r, p.inv_mu_c,
            p.gc1, p.gc2, p.gc3, p.gc4, p.gr31, p.gr34, p.gr37,
            p.inv_cloud_mass_factor, p.inv_rain_mass_factor,
            p.gamma_ventilation, p.gamma_fall_ventilation};
}

std::array<double, 11> endpoint_values(const WarmRainDoubleEndpoints& e) noexcept
{
    return {e.vapor_per_kg, e.cloud_per_kg, e.rain_per_kg,
            e.cloud_number_per_kg, e.rain_number_per_kg,
            e.temperature_increment_k, e.cloud_donor_budget_per_kg,
            e.evaporation.mass_before_per_kg, e.evaporation.number_before_per_m3,
            e.evaporation.admitted_mass_per_kg, e.evaporation.mass_after_per_kg};
}

bool same_endpoints(const WarmRainDoubleEndpoints& first,
                    const WarmRainDoubleEndpoints& second) noexcept
{
    return same_values(endpoint_values(first), endpoint_values(second)) &&
        same_bits(first.evaporation.number_after_per_m3, second.evaporation.number_after_per_m3) &&
        first.cloud_donor_exhausted == second.cloud_donor_exhausted &&
        first.evaporation.present == second.evaporation.present &&
        first.evaporation.donor_exhausted == second.evaporation.donor_exhausted &&
        first.reclassification.present == second.reclassification.present &&
        same_values(std::array<double, 4>{first.reclassification.rain_mass_before_per_kg,
            first.reclassification.rain_number_before_per_m3, first.reclassification.cloud_mass_before_per_kg,
            first.reclassification.cloud_number_before_per_m3},
            std::array<double, 4>{second.reclassification.rain_mass_before_per_kg,
            second.reclassification.rain_number_before_per_m3, second.reclassification.cloud_mass_before_per_kg,
            second.reclassification.cloud_number_before_per_m3});
}

template <typename Rain>
bool same_rain(const Rain& first, const Rain& second) noexcept
{
    return same_bits(first.mass_per_kg, second.mass_per_kg) &&
        same_bits(first.number_per_kg, second.number_per_kg) &&
        same_bits(first.reflectivity_per_kg, second.reflectivity_per_kg);
}

bool same_rates(const WarmRainTendencies& first, const WarmRainTendencies& second) noexcept
{
    return same_values(std::array<float, 6>{first.cloud, first.rain, first.vapor,
                         first.temperature, first.cloud_number, first.rain_number},
                       std::array<float, 6>{second.cloud, second.rain, second.vapor,
                         second.temperature, second.cloud_number, second.rain_number});
}

bool journal_close(double initial, double expected, double actual) noexcept
{
    const double scale = std::max({std::abs(initial), std::abs(expected), std::abs(actual), 1.0e-30});
    return std::isfinite(expected) && std::isfinite(actual) &&
        std::abs(expected - actual) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool warm_temperature(double temperature) noexcept
{
    return std::isfinite(temperature) && temperature > microphysics_constants::T0 && temperature < 400.0;
}

bool source_valid(const WarmRainParameters& parameters, const DoubleWarmRainCell& cell,
                  double dt, const RainTripleMomentState& initial_rain, std::string& error)
{
    try
    {
        validate_triple_warm_rain_source(parameters, cell, dt, {});
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
    if (!same_bits(cell.rain, static_cast<double>(initial_rain.mass_per_kg)) ||
        !same_bits(cell.rain_number, static_cast<double>(initial_rain.number_per_kg)))
    {
        error = "exact triple warm cloud source rain q/N provenance differs";
        return false;
    }
    return true;
}

TripleCloudAdjustmentCell cloud_source(const DoubleWarmRainCell& cell, double velocity) noexcept
{
    return {cell.temperature, cell.pressure, cell.density, cell.vapor,
            cell.cloud, cell.cloud_number, velocity};
}

DoubleWarmRainCell warm_source(const DoubleWarmRainCell& original,
                              const TripleCloudAdjustmentEndpoints& cloud) noexcept
{
    return {cloud.temperature_after_k, original.pressure, original.density,
            cloud.vapor_per_kg, cloud.cloud_per_kg, original.rain,
            cloud.cloud_number_per_kg, original.rain_number};
}

template <typename WarmResult, typename CloudResult>
bool finish_composition(const DoubleWarmRainCell& original,
                        const TripleCloudAdjustmentResult& cloud,
                        const WarmResult& warm,
                        CloudResult& candidate,
                        std::string& error)
{
    const auto& adjusted = cloud.endpoints();
    const auto& endpoints = warm.endpoints;
    const double total_heat = adjusted.temperature_increment_k + endpoints.temperature_increment_k;
    const double stage_final_temperature = adjusted.temperature_after_k + endpoints.temperature_increment_k;
    const double final_temperature = original.temperature + total_heat;
    const double initial_water = original.vapor + original.cloud + original.rain;
    const double final_water = endpoints.vapor_per_kg + endpoints.cloud_per_kg + endpoints.rain_per_kg;
    const double latent_factor = microphysics_constants::L_v / microphysics_constants::cp;
    if (!warm_temperature(original.temperature) || !warm_temperature(adjusted.temperature_after_k) ||
        !warm_temperature(stage_final_temperature) || !warm_temperature(final_temperature) ||
        !journal_close(initial_water, initial_water, final_water) ||
        !journal_close(original.temperature, stage_final_temperature, final_temperature) ||
        !journal_close(original.vapor * latent_factor,
                       latent_factor * (original.vapor - endpoints.vapor_per_kg), total_heat))
    {
        error = "exact triple warm cloud water or total heat closure is invalid";
        return false;
    }
    candidate.cloud = cloud;
    candidate.warm = warm;
    candidate.endpoints = endpoints;
    candidate.rain = warm.rain;
    candidate.rounded_rain = warm.rounded_rain;
    candidate.temperature_increment_k = total_heat;
    return true;
}

} // namespace

bool produce_exact_triple_warm_cloud_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell,
    double vertical_velocity, double dt, const RainTripleMomentState& initial_rain,
    ExactTripleWarmCloudCellResult& result, std::string& error,
    std::size_t observer_capacity)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    TripleCloudAdjustmentResult cloud;
    if (!produce_exact_triple_cloud_adjustment(cloud_source(cell, vertical_velocity), dt, cloud, error))
    {
        return false;
    }
    ExactTripleWarmDoubleSourceCellResult warm;
    if (!evaluate_exact_triple_warm_double_source_cell(
            parameters, warm_source(cell, cloud.endpoints()), dt, initial_rain,
            warm, error, observer_capacity))
    {
        return false;
    }
    ExactTripleWarmCloudCellResult candidate;
    if (!finish_composition(cell, cloud, warm, candidate, error))
    {
        return false;
    }
    candidate.original_rain_ = initial_rain;
    result = candidate;
    error.clear();
    return true;
}

bool apply_exact_triple_warm_cloud_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell,
    double vertical_velocity, double dt, const RainTripleMomentState& initial_rain,
    const ExactTripleWarmCloudCellResult& producer,
    ExactTripleWarmCloudCellResult& result, std::string& error)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    if (!same_rain(initial_rain, producer.original_rain_) ||
        !same_values(parameter_values(parameters), parameter_values(producer.warm.producer.parameters())))
    {
        error = "exact triple warm cloud original rain or parameter provenance differs";
        return false;
    }
    TripleCloudAdjustmentResult cloud;
    if (!apply_exact_triple_cloud_adjustment(
            cloud_source(cell, vertical_velocity), dt, producer.cloud, cloud, error))
    {
        return false;
    }
    ExactTripleWarmDoubleSourceCellResult warm;
    if (!apply_exact_triple_warm_double_source_cell(
            parameters, warm_source(cell, cloud.endpoints()), dt, initial_rain,
            producer.warm.producer, warm, error))
    {
        return false;
    }
    ExactTripleWarmCloudCellResult candidate;
    if (!finish_composition(cell, cloud, warm, candidate, error))
    {
        return false;
    }
    if (!same_endpoints(producer.warm.endpoints, warm.endpoints) ||
        !same_rain(producer.warm.rain, warm.rain) ||
        !same_rain(producer.warm.rounded_rain, warm.rounded_rain) ||
        !same_rates(producer.warm.rates, warm.rates) ||
        !same_endpoints(producer.endpoints, candidate.endpoints) ||
        !same_rain(producer.rain, candidate.rain) ||
        !same_rain(producer.rounded_rain, candidate.rounded_rain) ||
        !same_bits(producer.temperature_increment_k, candidate.temperature_increment_k))
    {
        error = "exact triple warm cloud captured endpoint provenance differs";
        return false;
    }
    candidate.original_rain_ = initial_rain;
    result = candidate;
    error.clear();
    return true;
}

bool produce_exact_triple_dynamic_warm_cloud_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell,
    double vertical_velocity, double dt, const RainTripleMomentState& initial_rain,
    ExactTripleDynamicWarmCloudCellResult& result, std::string& error, std::size_t observer_capacity)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    TripleCloudAdjustmentResult cloud;
    if (!produce_exact_triple_cloud_adjustment(cloud_source(cell, vertical_velocity), dt, cloud, error))
    {
        return false;
    }
    ExactTripleDynamicWarmCellResult warm;
    if (!evaluate_exact_triple_dynamic_warm_cell(parameters, warm_source(cell, cloud.endpoints()),
            dt, initial_rain, warm, error, observer_capacity))
    {
        return false;
    }
    ExactTripleDynamicWarmCloudCellResult candidate;
    if (!finish_composition(cell, cloud, warm, candidate, error))
    {
        return false;
    }
    candidate.original_rain_ = initial_rain;
    result = candidate;
    error.clear();
    return true;
}

bool apply_exact_triple_dynamic_warm_cloud_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell,
    double vertical_velocity, double dt, const RainTripleMomentState& initial_rain,
    const ExactTripleDynamicWarmCloudCellResult& producer,
    ExactTripleDynamicWarmCloudCellResult& result, std::string& error)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error) ||
        !same_rain(initial_rain, producer.original_rain_))
    {
        if (error.empty())
        {
            error = "dynamic warm cloud original rain provenance differs";
        }
        return false;
    }
    TripleCloudAdjustmentResult cloud;
    if (!apply_exact_triple_cloud_adjustment(cloud_source(cell, vertical_velocity), dt, producer.cloud, cloud, error))
    {
        return false;
    }
    ExactTripleDynamicWarmCellResult warm;
    if (!apply_exact_triple_dynamic_warm_cell(parameters, warm_source(cell, cloud.endpoints()),
            dt, initial_rain, producer.warm.producer, warm, error))
    {
        return false;
    }
    ExactTripleDynamicWarmCloudCellResult candidate;
    if (!finish_composition(cell, cloud, warm, candidate, error))
    {
        return false;
    }
    if (!same_endpoints(producer.warm.endpoints, warm.endpoints) ||
        !same_rain(producer.warm.rain, warm.rain) || !same_rain(producer.warm.rounded_rain, warm.rounded_rain) ||
        !same_rates(producer.warm.rates, warm.rates) || !same_endpoints(producer.endpoints, candidate.endpoints) ||
        !same_rain(producer.rain, candidate.rain) || !same_rain(producer.rounded_rain, candidate.rounded_rain) ||
        !same_bits(producer.temperature_increment_k, candidate.temperature_increment_k))
    {
        error = "dynamic warm cloud captured endpoint provenance differs";
        return false;
    }
    candidate.original_rain_ = initial_rain;
    result = candidate;
    error.clear();
    return true;
}

bool produce_exact_triple_subcycled_warm_cloud_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell,
    double vertical_velocity, double dt, const RainTripleMomentState& initial_rain,
    ExactTripleSubcycledWarmCloudCellResult& result, std::string& error, const WarmSubcycleOptions& options)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    TripleCloudAdjustmentResult cloud;
    if (!produce_exact_triple_cloud_adjustment(cloud_source(cell, vertical_velocity), dt, cloud, error))
    {
        return false;
    }
    WarmSubcycleCellResult warm;
    if (!evaluate_exact_triple_warm_subcycled_cell(parameters, warm_source(cell, cloud.endpoints()),
            dt, initial_rain, warm, error, options))
    {
        return false;
    }
    ExactTripleSubcycledWarmCloudCellResult candidate;
    if (!finish_composition(cell, cloud, warm, candidate, error))
    {
        return false;
    }
    candidate.original_rain_ = initial_rain;
    result = std::move(candidate);
    error.clear();
    return true;
}

bool apply_exact_triple_subcycled_warm_cloud_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell,
    double vertical_velocity, double dt, const RainTripleMomentState& initial_rain,
    const ExactTripleSubcycledWarmCloudCellResult& producer,
    ExactTripleSubcycledWarmCloudCellResult& result, std::string& error, const WarmSubcycleOptions& options)
{
    error.clear();
    if (!source_valid(parameters, cell, dt, initial_rain, error) || !same_rain(initial_rain, producer.original_rain_))
    {
        if (error.empty())
        {
            error = "subcycled warm cloud original rain provenance differs";
        }
        return false;
    }
    TripleCloudAdjustmentResult cloud;
    if (!apply_exact_triple_cloud_adjustment(cloud_source(cell, vertical_velocity), dt, producer.cloud, cloud, error))
    {
        return false;
    }
    WarmSubcycleCellResult warm;
    if (!apply_exact_triple_warm_subcycled_cell(parameters, warm_source(cell, cloud.endpoints()),
            dt, initial_rain, producer.warm.producer, warm, error, options))
    {
        return false;
    }
    ExactTripleSubcycledWarmCloudCellResult candidate;
    if (!finish_composition(cell, cloud, warm, candidate, error))
    {
        return false;
    }
    if (!same_endpoints(producer.warm.endpoints, warm.endpoints) ||
        !same_rain(producer.warm.rain, warm.rain) || !same_rain(producer.warm.rounded_rain, warm.rounded_rain) ||
        !same_rates(producer.warm.rates, warm.rates) || !same_endpoints(producer.endpoints, candidate.endpoints) ||
        !same_rain(producer.rain, candidate.rain) || !same_rain(producer.rounded_rain, candidate.rounded_rain) ||
        !same_bits(producer.temperature_increment_k, candidate.temperature_increment_k))
    {
        error = "subcycled warm cloud captured endpoint provenance differs";
        return false;
    }
    candidate.original_rain_ = initial_rain;
    result = std::move(candidate);
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
