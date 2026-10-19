#include "milbrandt.hpp"
#include "triple_warm_primary.hpp"

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_warm_primary_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {&rain_number, nullptr, nullptr, nullptr, nullptr},
        {&rain_reflectivity, nullptr, nullptr, nullptr, nullptr},
        InternalTripleMode::WarmPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::evaluate_internal_triple_warm_primary_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::WarmRainCell warm_cell{
        static_cast<float>(cell[0]), static_cast<float>(cell[1]), static_cast<float>(cell[2]),
        static_cast<float>(cell[3]), static_cast<float>(cell[4]), static_cast<float>(cell[5]),
        static_cast<float>(cell[10]), static_cast<float>(cell[11])};
    const milbrandt_detail::RainTripleMomentState initial{
        warm_cell.rain, warm_cell.rain_number, reflectivity[0]};
    milbrandt_detail::ExactTripleWarmCellResult warmed;
    if (!milbrandt_detail::evaluate_exact_triple_warm_cell(
            milbrandt_detail::warm_rain_parameters(alpha_r_, c_r_, d_r_, a_r_, b_r_),
            warm_cell, dt, initial, warmed, error))
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
    candidate.temperature_increment = warmed.endpoints.temperature_increment_k;
    candidate.rates[1] = warmed.rates.vapor;
    candidate.rates[2] = warmed.rates.cloud;
    candidate.rates[3] = warmed.rates.rain;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
