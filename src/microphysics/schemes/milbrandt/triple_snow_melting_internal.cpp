#include "milbrandt.hpp"
#include "triple_snow_melting.hpp"

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_snow_melting_for_test(
    const SimulationStateView& state,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    const Field3D& snow_number,
    const Field3D& snow_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {&rain_number, nullptr, &snow_number, nullptr, nullptr},
        {&rain_reflectivity, nullptr, &snow_reflectivity, nullptr, nullptr},
        InternalTripleMode::SnowMeltingOnly, error);
}

bool MilbrandtScheme::evaluate_internal_triple_snow_melting_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::RetainedSnowMeltingTripleState initial{
        {cell[5], cell[11], reflectivity[0]},
        {cell[7], cell[13], reflectivity[2]}};
    std::array<float, 11> rates{};
    milbrandt_detail::RetainedSnowMeltingTripleResult melted;
    if (!milbrandt_detail::evaluate_retained_triple_snow_melting_cell(
            cell, dt, {}, initial, rates, melted, error))
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
    candidate.water_after[2] = melted.unrounded.rain.mass_per_kg;
    candidate.water_after[4] = melted.unrounded.snow.mass_per_kg;
    candidate.number_after[0] = melted.rain.number_per_kg;
    candidate.number_after[2] = melted.snow.number_per_kg;
    candidate.reflectivity_after[0] = melted.rain.reflectivity_per_kg;
    candidate.reflectivity_after[2] = melted.snow.reflectivity_per_kg;
    candidate.temperature_increment = -microphysics_constants::L_f / microphysics_constants::cp
        * melted.event.admitted_mass_transfer_per_kg;
    candidate.rates[3] = rates[4];
    candidate.rates[5] = rates[1];
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
