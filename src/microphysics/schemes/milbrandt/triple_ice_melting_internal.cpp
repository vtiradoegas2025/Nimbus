#include "milbrandt.hpp"
#include "triple_ice_melting.hpp"

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_ice_melting_for_test(
    const SimulationStateView& state,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    const Field3D& ice_number,
    const Field3D& ice_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {&rain_number, &ice_number, nullptr, nullptr, nullptr},
        {&rain_reflectivity, &ice_reflectivity, nullptr, nullptr, nullptr},
        InternalTripleMode::IceMeltingOnly, error);
}

bool MilbrandtScheme::evaluate_internal_triple_ice_melting_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::RetainedIceMeltingTripleState initial{
        {cell[5], cell[11], reflectivity[0]},
        {cell[6], cell[12], reflectivity[1]}};
    std::array<float, 11> rates{};
    milbrandt_detail::RetainedIceMeltingTripleResult melted;
    if (!milbrandt_detail::evaluate_retained_triple_ice_melting_cell(
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
    candidate.water_after[3] = melted.unrounded.ice.mass_per_kg;
    candidate.number_after[0] = melted.rain.number_per_kg;
    candidate.number_after[1] = melted.ice.number_per_kg;
    candidate.reflectivity_after[0] = melted.rain.reflectivity_per_kg;
    candidate.reflectivity_after[1] = melted.ice.reflectivity_per_kg;
    candidate.temperature_increment = -microphysics_constants::L_f / microphysics_constants::cp
        * melted.event.admitted_mass_transfer_per_kg;
    candidate.rates[3] = rates[4];
    candidate.rates[4] = rates[0];
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
