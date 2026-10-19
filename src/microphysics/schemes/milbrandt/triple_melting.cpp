#include "triple_melting.hpp"
#include "melting.hpp"
#include "triple_sedimentation.hpp"
#include "microphysics/milbrandt_cold_parameters.hpp"

#include <cmath>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

bool fp32_range(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0 &&
        value <= std::numeric_limits<float>::max() &&
        (value == 0.0 || value >= std::numeric_limits<float>::denorm_min());
}

bool valid_state(const MeltingTripleMomentState& state,
                 TripleMomentSpecies species,
                 double& alpha,
                 std::string& error)
{
    for (double value : {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg})
    {
        if (!fp32_range(value))
        {
            error = "published triple melting q/N/Z is not FP32 representable";
            return false;
        }
    }
    if (state.mass_per_kg == 0.0 && state.number_per_kg == 0.0 &&
        state.reflectivity_per_kg == 0.0)
    {
        alpha = 0.0;
        return true;
    }
    double lambda = 0.0;
    return diagnose_triple_gamma_shape(state.mass_per_kg, state.number_per_kg,
        state.reflectivity_per_kg,
        current_nimbus_triple_sedimentation_parameters(species).mass_law,
        alpha, lambda, error);
}

bool round_endpoint(const MeltingTripleMomentState& unrounded,
                    TripleMomentSpecies species,
                    MeltingTripleMomentEndpoint& result,
                    std::string& error)
{
    double alpha = 0.0;
    if (!valid_state(unrounded, species, alpha, error))
    {
        return false;
    }
    const MeltingTripleMomentEndpoint candidate{
        static_cast<float>(unrounded.mass_per_kg),
        static_cast<float>(unrounded.number_per_kg),
        static_cast<float>(unrounded.reflectivity_per_kg)};
    if (!valid_state({candidate.mass_per_kg, candidate.number_per_kg,
                      candidate.reflectivity_per_kg}, species, alpha, error))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool source_matches(const std::array<double, 12>& cell,
                    const MeltingTripleState& initial,
                    std::string& error)
{
    if (cell[6] != initial.graupel.mass_per_kg || cell[10] != initial.graupel.number_per_kg ||
        cell[7] != initial.hail.mass_per_kg || cell[11] != initial.hail.number_per_kg)
    {
        error = "published triple melting cell differs from exact source q/N";
        return false;
    }
    return true;
}

} // namespace

bool apply_published_triple_melting(
    const std::array<double, 12>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const MeltingTripleState& initial,
    const std::array<MeltingAcceptedEvent, 2>& events,
    MeltingTripleResult& result,
    std::string& error)
{
    if (!validate_triple_melting_context(cell, duration_s, profile, error) ||
        !source_matches(cell, initial, error))
    {
        return false;
    }
    double rain_alpha = 0.0;
    double graupel_alpha = 0.0;
    double hail_alpha = 0.0;
    if (!valid_state(initial.rain, TripleMomentSpecies::Rain, rain_alpha, error) ||
        !valid_state(initial.graupel, TripleMomentSpecies::Graupel, graupel_alpha, error) ||
        !valid_state(initial.hail, TripleMomentSpecies::Hail, hail_alpha, error))
    {
        return false;
    }
    const std::array<const MeltingTripleMomentState*, 2> donors{&initial.graupel, &initial.hail};
    const std::array<double, 2> alphas{graupel_alpha, hail_alpha};
    for (std::size_t index = 0; index < events.size(); ++index)
    {
        const auto& event = events[index];
        const auto& donor = *donors[index];
        const auto species = index == 0 ? MeltingFrozenSpecies::Graupel : MeltingFrozenSpecies::Hail;
        if (event.species != species || event.entry_mass_per_kg != donor.mass_per_kg ||
            event.entry_number_per_kg != donor.number_per_kg || event.source_alpha != alphas[index] ||
            event.temperature_k != cell[0] || event.pressure_pa != cell[1] ||
            event.dry_density_kg_m3 != cell[2] || event.vapor_per_kg != cell[3] ||
            event.duration_s != duration_s)
        {
            error = "published triple melting event source or context differs";
            return false;
        }
        try
        {
            const auto checked = make_melting_accepted_event(event.species,
                event.entry_mass_per_kg, event.entry_number_per_kg, event.source_alpha,
                event.admitted_mass_transfer_per_kg, event.admitted_number_transfer_per_kg,
                event.temperature_k, event.pressure_pa, event.dry_density_kg_m3,
                event.vapor_per_kg, event.duration_s);
            if (checked.admitted_fraction != event.admitted_fraction)
            {
                error = "published triple melting event fraction differs";
                return false;
            }
        }
        catch (const std::runtime_error& failure)
        {
            error = failure.what();
            return false;
        }
    }
    MeltingTripleResult candidate;
    candidate.events = events;
    candidate.unrounded = initial;
    const auto rain_law = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law;
    const std::array<MeltingTripleMomentState*, 2> endpoints{
        &candidate.unrounded.graupel, &candidate.unrounded.hail};
    for (std::size_t index = 0; index < events.size(); ++index)
    {
        const auto& donor = *donors[index];
        const auto& event = events[index];
        const double reflectivity_transfer = donor.mass_per_kg > 0.0
            ? donor.reflectivity_per_kg *
                (2.0 * event.admitted_mass_transfer_per_kg / donor.mass_per_kg
                 - event.admitted_number_transfer_per_kg / donor.number_per_kg)
            : 0.0;
        const auto law = current_nimbus_triple_sedimentation_parameters(
            index == 0 ? TripleMomentSpecies::Graupel : TripleMomentSpecies::Hail).mass_law;
        const double coefficient_ratio = law.coefficient / rain_law.coefficient;
        *endpoints[index] = {
            donor.mass_per_kg - event.admitted_mass_transfer_per_kg,
            donor.number_per_kg - event.admitted_number_transfer_per_kg,
            donor.reflectivity_per_kg - reflectivity_transfer};
        candidate.unrounded.rain.mass_per_kg += event.admitted_mass_transfer_per_kg;
        candidate.unrounded.rain.number_per_kg += event.admitted_number_transfer_per_kg;
        candidate.unrounded.rain.reflectivity_per_kg +=
            coefficient_ratio * coefficient_ratio * reflectivity_transfer;
    }
    if (!round_endpoint(candidate.unrounded.rain, TripleMomentSpecies::Rain, candidate.rain, error) ||
        !round_endpoint(candidate.unrounded.graupel, TripleMomentSpecies::Graupel, candidate.graupel, error) ||
        !round_endpoint(candidate.unrounded.hail, TripleMomentSpecies::Hail, candidate.hail, error))
    {
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

bool evaluate_published_triple_melting_cell(
    const std::array<double, 12>& cell,
    double duration_s,
    const MeltingNoCollectionProfile& profile,
    const MeltingTripleState& initial,
    std::array<float, 11>& rates,
    MeltingTripleResult& result,
    std::string& error)
{
    if (!validate_triple_melting_context(cell, duration_s, profile, error) ||
        !source_matches(cell, initial, error))
    {
        return false;
    }
    double rain_alpha = 0.0;
    double graupel_alpha = 0.0;
    double hail_alpha = 0.0;
    if (!valid_state(initial.rain, TripleMomentSpecies::Rain, rain_alpha, error) ||
        !valid_state(initial.graupel, TripleMomentSpecies::Graupel, graupel_alpha, error) ||
        !valid_state(initial.hail, TripleMomentSpecies::Hail, hail_alpha, error))
    {
        return false;
    }
    const auto& p = default_cold_parameters;
    const std::array<double, 13> parameters{
        p[9], p[4], p[14], graupel_alpha, p[15], p[20], p[0], p[10],
        hail_alpha, p[16], p[21], p[1], p[11]};
    std::array<float, 11> candidate_rates{};
    MeltingAcceptedEventBuffer observer;
    try
    {
        published_triple_melting_cell(parameters, cell, duration_s, profile, candidate_rates, observer);
    }
    catch (const std::runtime_error& failure)
    {
        error = failure.what();
        return false;
    }
    MeltingTripleResult candidate;
    if (observer.size != 2 || !apply_published_triple_melting(
            cell, duration_s, profile, initial, observer.events, candidate, error))
    {
        if (error.empty())
        {
            error = "published triple melting has no paired admitted events";
        }
        return false;
    }
    rates = candidate_rates;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
