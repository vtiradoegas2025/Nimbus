#include "milbrandt.hpp"
#include "triple_melting.hpp"

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_melting_for_test(
    const SimulationStateView& state,
    const Field3D& rain_number,
    const Field3D& rain_reflectivity,
    const Field3D& graupel_number,
    const Field3D& graupel_reflectivity,
    const Field3D& hail_number,
    const Field3D& hail_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_mode_for_test(
        state, &rain_number, &rain_reflectivity, graupel_number, graupel_reflectivity,
        hail_number, hail_reflectivity, InternalTripleMode::MeltingOnly, error);
}

bool MilbrandtScheme::evaluate_internal_triple_melting_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const std::array<double, 12> melting_cell{
        cell[0], cell[1], cell[2], cell[3],
        cell[6], cell[7], cell[8], cell[9],
        cell[12], cell[13], cell[14], cell[15]};
    const milbrandt_detail::MeltingTripleState initial{
        {cell[5], cell[11], reflectivity[0]},
        {cell[8], cell[14], reflectivity[3]},
        {cell[9], cell[15], reflectivity[4]}};
    std::array<float, 11> rates{};
    milbrandt_detail::MeltingTripleResult melted;
    if (!milbrandt_detail::evaluate_published_triple_melting_cell(
            melting_cell, dt, {}, initial, rates, melted, error))
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
    candidate.water_after[5] = melted.unrounded.graupel.mass_per_kg;
    candidate.water_after[6] = melted.unrounded.hail.mass_per_kg;
    const std::array<const milbrandt_detail::MeltingTripleMomentEndpoint*, 3> endpoints{
        &melted.rain, &melted.graupel, &melted.hail};
    const std::array<std::size_t, 3> species_index{0, 3, 4};
    for (std::size_t index = 0; index < endpoints.size(); ++index)
    {
        candidate.number_after[species_index[index]] = endpoints[index]->number_per_kg;
        candidate.reflectivity_after[species_index[index]] = endpoints[index]->reflectivity_per_kg;
    }
    const double mass_melted = melted.events[0].admitted_mass_transfer_per_kg
        + melted.events[1].admitted_mass_transfer_per_kg;
    candidate.temperature_increment = -microphysics_constants::L_f / microphysics_constants::cp * mass_melted;
    candidate.rates[3] = rates[4];
    candidate.rates[4] = rates[0];
    candidate.rates[5] = rates[1];
    candidate.rates[6] = rates[2];
    candidate.rates[7] = rates[3];
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
