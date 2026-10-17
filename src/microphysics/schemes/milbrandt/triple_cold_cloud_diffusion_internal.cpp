#include "milbrandt.hpp"
#include "cold.hpp"
#include "triple_cold_cloud_diffusion.hpp"

#include <cmath>
#include <limits>

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_cold_cloud_diffusion_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    const Field3D& snow_number,
    const Field3D& snow_reflectivity,
    const Field3D& graupel_number,
    const Field3D& graupel_reflectivity,
    const Field3D& hail_number,
    const Field3D& hail_reflectivity,
    std::string& error)
{
    return enable_internal_triple_primary_moments_for_test(state,
        {nullptr, nullptr, &snow_number, &graupel_number, &hail_number},
        {nullptr, nullptr, &snow_reflectivity, &graupel_reflectivity, &hail_reflectivity},
        InternalTripleMode::ColdCloudDiffusionPrimaryOnly, error, &cloud_number);
}

bool MilbrandtScheme::evaluate_internal_triple_cold_cloud_diffusion_primary_cell(
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
    milbrandt_detail::ColdCloudDiffusionCell source;
    source.temperature_k = cell[0];
    source.pressure_pa = cell[1];
    source.dry_density_kg_m3 = cell[2];
    source.vapor_mass_per_kg = cell[3];
    source.initial.cloud_mass_per_kg = cell[4];
    source.initial.cloud_number_per_kg = cell[10];
    for (std::size_t species = 0; species < source.initial.frozen.size(); ++species)
    {
        source.initial.frozen[species] = {cell[species + 7], cell[species + 13], reflectivity[species + 2]};
    }
    milbrandt_detail::ColdCloudDiffusionResult combined;
    if (!milbrandt_detail::evaluate_exact_triple_cold_cloud_diffusion_cell(
            parameters, source, dt, combined, error))
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
    candidate.water_after[0] = combined.vapor_mass_per_kg;
    candidate.water_after[1] = combined.unrounded.cloud_mass_per_kg;
    candidate.cloud_number_after = combined.unrounded.cloud_number_per_kg;
    candidate.temperature_increment = combined.temperature_increment_k;
    const auto store_rate = [&](double increment, std::size_t slot)
    {
        const double rate = increment / dt;
        if (!std::isfinite(rate) || std::abs(rate) > std::numeric_limits<float>::max() ||
            (rate != 0.0 && static_cast<float>(rate) == 0.0f))
        {
            error = "cold cloud diffusion mass rate is not FP32 representable";
            return false;
        }
        candidate.rates[slot] = static_cast<float>(rate);
        return true;
    };
    if (!store_rate(combined.vapor_mass_increment_per_kg, 1) ||
        !store_rate(combined.cloud_mass_increment_per_kg, 2))
    {
        return false;
    }
    for (std::size_t species = 0; species < combined.rounded.size(); ++species)
    {
        if (!store_rate(combined.mass_increment_per_kg[species], species + 5))
        {
            return false;
        }
        candidate.water_after[species + 4] = combined.unrounded.frozen[species].mass_per_kg;
        candidate.number_after[species + 2] = combined.rounded[species].number_per_kg;
        candidate.reflectivity_after[species + 2] = combined.rounded[species].reflectivity_per_kg;
    }
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
