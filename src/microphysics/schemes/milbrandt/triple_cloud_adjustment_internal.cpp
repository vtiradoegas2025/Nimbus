#include "milbrandt.hpp"
#include "triple_cloud_adjustment.hpp"
#include "core/runtime/simulation_state.hpp"

namespace tmv
{

bool MilbrandtScheme::enable_internal_triple_cloud_adjustment_for_test(
    const SimulationStateView& state,
    const Field3D& cloud_number,
    std::string& error)
{
    if (internal_triple_)
    {
        error = "internal Milbrandt triple modes are mutually exclusive";
        return false;
    }
    if (!validate_internal_triple_source_profile(
            state, false, error, std::nullopt, false, false, false, false, false, true))
    {
        return false;
    }
    const auto& density = state.fields.rho;
    Field3D zero(density.size_r(), density.size_th(), density.size_z(), 0.0f);
    return enable_internal_triple_primary_moments_for_test(state,
        {&zero, nullptr, nullptr, nullptr, nullptr},
        {&zero, nullptr, nullptr, nullptr, nullptr},
        InternalTripleMode::CloudAdjustmentOnly, error, &cloud_number);
}

bool MilbrandtScheme::evaluate_internal_triple_cloud_adjustment_cell(
    const std::array<double, 16>& cell,
    const std::array<float, 5>& reflectivity,
    double velocity,
    double dt,
    InternalTriplePrimaryCellCandidate& result,
    std::string& error) const
{
    const milbrandt_detail::TripleCloudAdjustmentCell source{
        cell[0], cell[1], cell[2], cell[3], cell[4], cell[10], velocity};
    milbrandt_detail::TripleCloudAdjustmentResult adjusted;
    if (!milbrandt_detail::produce_exact_triple_cloud_adjustment(source, dt, adjusted, error))
    {
        return false;
    }
    const auto& endpoints = adjusted.endpoints();
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
    candidate.water_after[0] = endpoints.vapor_per_kg;
    candidate.water_after[1] = endpoints.cloud_per_kg;
    candidate.cloud_number_after = endpoints.cloud_number_per_kg;
    candidate.temperature_increment = endpoints.temperature_increment_k;
    candidate.rates[1] = static_cast<float>(-endpoints.admitted_cloud_increment_per_kg / dt);
    candidate.rates[2] = static_cast<float>(endpoints.admitted_cloud_increment_per_kg / dt);
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv
