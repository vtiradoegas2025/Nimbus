#include "triple_ice_melting.hpp"
#include "triple_sedimentation.hpp"

#include <cmath>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

bool valid_state(const MeltingTripleMomentState& state,
                 TripleMomentSpecies species,
                 double& alpha,
                 std::string& error)
{
    for (double value : {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0 || value > std::numeric_limits<float>::max() ||
            (value != 0.0 && value < std::numeric_limits<float>::denorm_min()))
        {
            error = "retained triple ice melting q/N/Z is not FP32 representable";
            return false;
        }
    }
    if (state.mass_per_kg == 0.0 && state.number_per_kg == 0.0 && state.reflectivity_per_kg == 0.0)
    {
        alpha = 0.0;
        return true;
    }
    double lambda = 0.0;
    return diagnose_triple_gamma_shape(state.mass_per_kg, state.number_per_kg,
        state.reflectivity_per_kg, current_nimbus_triple_sedimentation_parameters(species).mass_law,
        alpha, lambda, error);
}

bool round_endpoint(const MeltingTripleMomentState& state,
                    TripleMomentSpecies species,
                    MeltingTripleMomentEndpoint& result,
                    std::string& error)
{
    double alpha = 0.0;
    if (!valid_state(state, species, alpha, error))
    {
        return false;
    }
    const MeltingTripleMomentEndpoint candidate{
        static_cast<float>(state.mass_per_kg), static_cast<float>(state.number_per_kg),
        static_cast<float>(state.reflectivity_per_kg)};
    if (!valid_state({candidate.mass_per_kg, candidate.number_per_kg, candidate.reflectivity_per_kg},
                     species, alpha, error))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool valid_source(const std::array<double, 16>& cell,
                  double duration,
                  const MeltingNoCollectionProfile& profile,
                  const RetainedIceMeltingTripleState& initial,
                  double& ice_alpha,
                  std::string& error)
{
    if (!std::isfinite(duration) || duration <= 0.0 || !std::isfinite(cell[0]) || cell[0] < 150.0 ||
        !std::isfinite(cell[1]) || cell[1] < 100.0 || cell[1] > 200000.0 ||
        !std::isfinite(cell[2]) || cell[2] <= 0.0)
    {
        error = "retained triple ice melting context is invalid";
        return false;
    }
    for (std::size_t index = 3; index < cell.size(); ++index)
    {
        if (!std::isfinite(cell[index]) || cell[index] < 0.0)
        {
            error = "retained triple ice melting water or number input is invalid";
            return false;
        }
    }
    if (!validate_melting_no_collection_profile(profile, error) ||
        !validate_retained_ice_melting_unowned_channels(
            {cell[4], cell[7], cell[8], cell[9], cell[10], cell[13], cell[14], cell[15]}, error))
    {
        return false;
    }
    if (cell[5] != initial.rain.mass_per_kg || cell[11] != initial.rain.number_per_kg ||
        cell[6] != initial.ice.mass_per_kg || cell[12] != initial.ice.number_per_kg)
    {
        error = "retained triple ice melting cell differs from exact source q/N";
        return false;
    }
    double rain_alpha = 0.0;
    return valid_state(initial.rain, TripleMomentSpecies::Rain, rain_alpha, error) &&
        valid_state(initial.ice, TripleMomentSpecies::Ice, ice_alpha, error);
}

RetainedIceMeltingAcceptedEvent accepted_event(
    const std::array<double, 16>& cell,
    double duration,
    const RetainedIceMeltingTripleState& initial,
    double alpha)
{
    const bool melts = cell[0] > microphysics_constants::T0;
    return {initial.ice.mass_per_kg, initial.ice.number_per_kg, initial.ice.reflectivity_per_kg,
            alpha, melts ? initial.ice.mass_per_kg : 0.0, melts ? initial.ice.number_per_kg : 0.0,
            melts && initial.ice.mass_per_kg > 0.0 ? 1.0 : 0.0,
            cell[0], cell[1], cell[2], cell[3], duration};
}

bool same_event(const RetainedIceMeltingAcceptedEvent& left,
                const RetainedIceMeltingAcceptedEvent& right) noexcept
{
    return left.entry_mass_per_kg == right.entry_mass_per_kg &&
        left.entry_number_per_kg == right.entry_number_per_kg &&
        left.entry_reflectivity_per_kg == right.entry_reflectivity_per_kg &&
        left.source_alpha == right.source_alpha &&
        left.admitted_mass_transfer_per_kg == right.admitted_mass_transfer_per_kg &&
        left.admitted_number_transfer_per_kg == right.admitted_number_transfer_per_kg &&
        left.admitted_fraction == right.admitted_fraction && left.temperature_k == right.temperature_k &&
        left.pressure_pa == right.pressure_pa && left.dry_density_kg_m3 == right.dry_density_kg_m3 &&
        left.vapor_per_kg == right.vapor_per_kg && left.duration_s == right.duration_s;
}

} // namespace

bool validate_retained_ice_melting_unowned_channels(
    const std::array<double, 8>& channels,
    std::string& error)
{
    for (std::size_t index = 0; index < channels.size(); ++index)
    {
        if (channels[index] != 0.0)
        {
            error = std::string("retained triple ice melting observed unowned channel: ")
                + retained_ice_melting_unowned_channel_names[index];
            return false;
        }
    }
    error.clear();
    return true;
}

bool apply_retained_triple_ice_melting(
    const std::array<double, 16>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const RetainedIceMeltingTripleState& initial,
    const RetainedIceMeltingAcceptedEvent& event,
    RetainedIceMeltingTripleResult& result,
    std::string& error)
{
    double alpha = 0.0;
    if (!valid_source(cell, duration_s, profile, initial, alpha, error))
    {
        return false;
    }
    if (!same_event(event, accepted_event(cell, duration_s, initial, alpha)))
    {
        error = "retained triple ice melting event differs from instantaneous source or context";
        return false;
    }
    const double reflectivity_transfer = event.admitted_fraction * initial.ice.reflectivity_per_kg;
    const auto ice_law = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Ice).mass_law;
    const auto rain_law = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law;
    const double coefficient_ratio = ice_law.coefficient / rain_law.coefficient;
    RetainedIceMeltingTripleResult candidate;
    candidate.event = event;
    candidate.unrounded = {
        {initial.rain.mass_per_kg + event.admitted_mass_transfer_per_kg,
         initial.rain.number_per_kg + event.admitted_number_transfer_per_kg,
         initial.rain.reflectivity_per_kg + coefficient_ratio * coefficient_ratio * reflectivity_transfer},
        {initial.ice.mass_per_kg - event.admitted_mass_transfer_per_kg,
         initial.ice.number_per_kg - event.admitted_number_transfer_per_kg,
         initial.ice.reflectivity_per_kg - reflectivity_transfer}};
    if (!round_endpoint(candidate.unrounded.rain, TripleMomentSpecies::Rain, candidate.rain, error) ||
        !round_endpoint(candidate.unrounded.ice, TripleMomentSpecies::Ice, candidate.ice, error))
    {
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

bool evaluate_retained_triple_ice_melting_cell(
    const std::array<double, 16>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const RetainedIceMeltingTripleState& initial,
    std::array<float, 11>& rates,
    RetainedIceMeltingTripleResult& result,
    std::string& error)
{
    double alpha = 0.0;
    if (!valid_source(cell, duration_s, profile, initial, alpha, error))
    {
        return false;
    }
    const auto event = accepted_event(cell, duration_s, initial, alpha);
    const double mass_rate = event.admitted_mass_transfer_per_kg / duration_s;
    const double number_rate = event.admitted_number_transfer_per_kg / duration_s;
    const double heat_rate = -microphysics_constants::L_f / microphysics_constants::cp
        * event.admitted_mass_transfer_per_kg / duration_s;
    if ((event.admitted_mass_transfer_per_kg != 0.0 && (mass_rate == 0.0 || heat_rate == 0.0)) ||
        (event.admitted_number_transfer_per_kg != 0.0 && number_rate == 0.0))
    {
        error = "retained triple ice melting nonzero rate underflowed";
        return false;
    }
    const std::array<double, 11> raw_rates{
        -mass_rate, 0.0, 0.0, 0.0, mass_rate, heat_rate, number_rate, -number_rate, 0.0, 0.0, 0.0};
    for (double rate : raw_rates)
    {
        if (!melting_rate_representable(rate))
        {
            error = "retained triple ice melting rate is not FP32 representable";
            return false;
        }
    }
    RetainedIceMeltingTripleResult candidate;
    if (!apply_retained_triple_ice_melting(cell, duration_s, profile, initial, event, candidate, error))
    {
        return false;
    }
    std::array<float, 11> candidate_rates{};
    for (std::size_t index = 0; index < candidate_rates.size(); ++index)
    {
        candidate_rates[index] = static_cast<float>(raw_rates[index]);
    }
    rates = candidate_rates;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
