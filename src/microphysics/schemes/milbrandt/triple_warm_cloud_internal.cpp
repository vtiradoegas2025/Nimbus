#include "milbrandt.hpp"
#include "triple_warm_cloud.hpp"

#include <cmath>
#include <limits>

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_warm_cloud_primary_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {&rain_number, nullptr, nullptr, nullptr, nullptr},
        {&rain_reflectivity, nullptr, nullptr, nullptr, nullptr},
        InternalTripleMode::WarmCloudPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::evaluate_internal_triple_warm_cloud_primary_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double velocity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::DoubleWarmRainCell source{
        cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], cell[10], cell[11]};
    const milbrandt_detail::RainTripleMomentState initial{
        static_cast<float>(cell[5]), static_cast<float>(cell[11]), reflectivity[0]};
    milbrandt_detail::ExactTripleWarmCloudCellResult warmed;
    if (!milbrandt_detail::produce_exact_triple_warm_cloud_cell(
            milbrandt_detail::warm_rain_parameters(alpha_r_, c_r_, d_r_, a_r_, b_r_),
            source, velocity, dt, initial, warmed, error))
    {
        return false;
    }
    InternalTriplePrimaryCellCandidate candidate;
    for (std::size_t species = 0; species < candidate.water_after.size(); ++species)
    {
        candidate.water_after[species] = cell[species + 3];
    }
    for (std::size_t species = 0; species < candidate.number_after.size(); ++species)
    {
        candidate.number_after[species] = static_cast<float>(cell[species + 11]);
    }
    candidate.reflectivity_after = reflectivity;
    candidate.water_after[0] = warmed.endpoints.vapor_per_kg;
    candidate.water_after[1] = warmed.endpoints.cloud_per_kg;
    candidate.water_after[2] = warmed.endpoints.rain_per_kg;
    candidate.cloud_number_after = warmed.endpoints.cloud_number_per_kg;
    candidate.number_after[0] = warmed.rounded_rain.number_per_kg;
    candidate.reflectivity_after[0] = warmed.rounded_rain.reflectivity_per_kg;
    candidate.temperature_increment = warmed.temperature_increment_k;
    for (std::size_t species = 0; species < 3; ++species)
    {
        const double rate = (candidate.water_after[species] - cell[species + 3]) / dt;
        if (!std::isfinite(rate) || std::abs(rate) > std::numeric_limits<float>::max())
        {
            error = "internal Milbrandt WarmCloudPrimaryOnly diagnostic water rate is invalid";
            return false;
        }
        candidate.rates[species + 1] = static_cast<float>(rate);
    }
    result = candidate;
    error.clear();
    return true;
}

bool MilbrandtScheme::enable_internal_triple_dynamic_warm_cloud_primary_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {&rain_number, nullptr, nullptr, nullptr, nullptr},
        {&rain_reflectivity, nullptr, nullptr, nullptr, nullptr},
        InternalTripleMode::DynamicWarmCloudPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::evaluate_internal_triple_dynamic_warm_cloud_primary_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double velocity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::DoubleWarmRainCell source{
        cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], cell[10], cell[11]};
    const milbrandt_detail::RainTripleMomentState initial{
        static_cast<float>(cell[5]), static_cast<float>(cell[11]), reflectivity[0]};
    milbrandt_detail::ExactTripleDynamicWarmCloudCellResult warmed;
    if (!milbrandt_detail::produce_exact_triple_dynamic_warm_cloud_cell(
            milbrandt_detail::warm_rain_parameters(alpha_r_, c_r_, d_r_, a_r_, b_r_),
            source, velocity, dt, initial, warmed, error))
    {
        return false;
    }
    InternalTriplePrimaryCellCandidate candidate;
    for (std::size_t species = 0; species < candidate.water_after.size(); ++species)
    {
        candidate.water_after[species] = cell[species + 3];
    }
    for (std::size_t species = 0; species < candidate.number_after.size(); ++species)
    {
        candidate.number_after[species] = static_cast<float>(cell[species + 11]);
    }
    candidate.reflectivity_after = reflectivity;
    candidate.water_after[0] = warmed.endpoints.vapor_per_kg;
    candidate.water_after[1] = warmed.endpoints.cloud_per_kg;
    candidate.water_after[2] = warmed.endpoints.rain_per_kg;
    candidate.cloud_number_after = warmed.endpoints.cloud_number_per_kg;
    candidate.number_after[0] = warmed.rounded_rain.number_per_kg;
    candidate.reflectivity_after[0] = warmed.rounded_rain.reflectivity_per_kg;
    candidate.temperature_increment = warmed.temperature_increment_k;
    for (std::size_t species = 0; species < 3; ++species)
    {
        const double rate = (candidate.water_after[species] - cell[species + 3]) / dt;
        if (!std::isfinite(rate) || std::abs(rate) > std::numeric_limits<float>::max())
        {
            error = "internal Milbrandt DynamicWarmCloudPrimaryOnly diagnostic water rate is invalid";
            return false;
        }
        candidate.rates[species + 1] = static_cast<float>(rate);
    }
    result = candidate;
    error.clear();
    return true;
}

bool MilbrandtScheme::enable_internal_triple_subcycled_warm_cloud_primary_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {&rain_number, nullptr, nullptr, nullptr, nullptr},
        {&rain_reflectivity, nullptr, nullptr, nullptr, nullptr},
        InternalTripleMode::SubcycledWarmCloudPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::evaluate_internal_triple_subcycled_warm_cloud_primary_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double velocity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::DoubleWarmRainCell source{
        cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], cell[10], cell[11]};
    const milbrandt_detail::RainTripleMomentState initial{
        static_cast<float>(cell[5]), static_cast<float>(cell[11]), reflectivity[0]};
    milbrandt_detail::ExactTripleSubcycledWarmCloudCellResult warmed;
    if (!milbrandt_detail::produce_exact_triple_subcycled_warm_cloud_cell(
            milbrandt_detail::warm_rain_parameters(alpha_r_, c_r_, d_r_, a_r_, b_r_),
            source, velocity, dt, initial, warmed, error))
    {
        return false;
    }
    InternalTriplePrimaryCellCandidate candidate;
    for (std::size_t species = 0; species < candidate.water_after.size(); ++species)
    {
        candidate.water_after[species] = cell[species + 3];
    }
    for (std::size_t species = 0; species < candidate.number_after.size(); ++species)
    {
        candidate.number_after[species] = static_cast<float>(cell[species + 11]);
    }
    candidate.reflectivity_after = reflectivity;
    candidate.water_after[0] = warmed.endpoints.vapor_per_kg;
    candidate.water_after[1] = warmed.endpoints.cloud_per_kg;
    candidate.water_after[2] = warmed.endpoints.rain_per_kg;
    candidate.cloud_number_after = warmed.endpoints.cloud_number_per_kg;
    candidate.number_after[0] = warmed.rounded_rain.number_per_kg;
    candidate.reflectivity_after[0] = warmed.rounded_rain.reflectivity_per_kg;
    candidate.temperature_increment = warmed.temperature_increment_k;
    for (std::size_t species = 0; species < 3; ++species)
    {
        const double rate = (candidate.water_after[species] - cell[species + 3]) / dt;
        if (!std::isfinite(rate) || std::abs(rate) > std::numeric_limits<float>::max())
        {
            error = "internal Milbrandt SubcycledWarmCloudPrimaryOnly diagnostic water rate is invalid";
            return false;
        }
        candidate.rates[species + 1] = static_cast<float>(rate);
    }
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
