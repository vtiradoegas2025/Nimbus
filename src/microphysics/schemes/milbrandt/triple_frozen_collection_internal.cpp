#include "milbrandt.hpp"
#include "cold.hpp"
#include "triple_frozen_collection.hpp"

#include <cmath>
#include <limits>

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_frozen_collection_for_test(
    const SimulationStateView& state,
    const Field3D& ice_number,
    const Field3D& ice_reflectivity,
    const Field3D& snow_number,
    const Field3D& snow_reflectivity,
    const Field3D& graupel_number,
    const Field3D& graupel_reflectivity,
    const Field3D& hail_number,
    const Field3D& hail_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, &ice_number, &snow_number, &graupel_number, &hail_number},
        {nullptr, &ice_reflectivity, &snow_reflectivity, &graupel_reflectivity, &hail_reflectivity},
        InternalTripleMode::FrozenCollectionPrimaryOnly, error);
}

bool MilbrandtScheme::evaluate_internal_triple_frozen_collection_primary_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::ColdParameters parameters{
        a_g_, a_h_, a_i_, a_r_, a_s_,
        alpha_g_, alpha_h_, alpha_i_, alpha_r_, alpha_s_,
        b_g_, b_h_, b_i_, b_r_, b_s_,
        c_g_, c_h_, c_i_, c_r_, c_s_,
        d_g_, d_h_, d_i_, d_r_, d_s_, hail_processes_};
    milbrandt_detail::FrozenCollectionCell source;
    source.temperature_k = cell[0];
    source.pressure_pa = cell[1];
    source.dry_density_kg_m3 = cell[2];
    for (std::size_t species = 0; species < source.initial.frozen.size(); ++species)
    {
        source.initial.frozen[species] = {cell[species + 6], cell[species + 12], reflectivity[species + 1]};
    }
    milbrandt_detail::FrozenCollectionResult collected;
    if (!milbrandt_detail::evaluate_exact_triple_frozen_collection_cell(
            parameters, source, dt, collected, error))
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
    for (std::size_t species = 0; species < collected.rounded.size(); ++species)
    {
        const double rate = collected.mass_increment_per_kg[species] / dt;
        if (!std::isfinite(rate) || std::abs(rate) > std::numeric_limits<float>::max() ||
            (rate != 0.0 && static_cast<float>(rate) == 0.0f))
        {
            error = "frozen collection mass rate is not FP32 representable";
            return false;
        }
        candidate.water_after[species + 3] = collected.unrounded.frozen[species].mass_per_kg;
        candidate.number_after[species + 1] = collected.rounded[species].number_per_kg;
        candidate.reflectivity_after[species + 1] = collected.rounded[species].reflectivity_per_kg;
        candidate.rates[species + 4] = static_cast<float>(rate);
    }
    candidate.temperature_increment = collected.temperature_increment_k;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
