#include "triple_cold_diffusion.hpp"
#include "cold.hpp"
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

bool valid_state(const ColdFrozenDiffusionTripleState& state,
                 TripleMomentMassLaw law,
                 double& alpha,
                 std::string& error)
{
    for (double value : {state.mass_per_kg, state.number_per_kg,
                         state.reflectivity_per_kg})
    {
        if (!fp32_range(value))
        {
            error = "published frozen diffusion q/N/Z is not FP32 representable";
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
    return diagnose_triple_gamma_shape(
        state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg,
        law, alpha, lambda, error);
}

TripleMomentMassLaw species_law(ColdFrozenDiffusionSpecies species)
{
    return current_nimbus_triple_sedimentation_parameters(
        species == ColdFrozenDiffusionSpecies::Graupel
            ? TripleMomentSpecies::Graupel : TripleMomentSpecies::Hail).mass_law;
}

ColdFrozenDiffusionTripleState unrounded_diffusion_endpoint(
    const ColdFrozenDiffusionTripleState& initial,
    const ColdFrozenDiffusionAcceptedEvent& event) noexcept
{
    const double reflectivity_increment = initial.mass_per_kg > 0.0
        ? initial.reflectivity_per_kg *
            (2.0 * event.admitted_mass_increment_per_kg / initial.mass_per_kg
             - event.admitted_number_increment_per_kg / initial.number_per_kg)
        : 0.0;
    return {
        initial.mass_per_kg + event.admitted_mass_increment_per_kg,
        initial.number_per_kg + event.admitted_number_increment_per_kg,
        initial.reflectivity_per_kg + reflectivity_increment};
}

bool validate_and_round_joint_endpoint(
    const ColdFrozenDiffusionTripleState& unrounded,
    TripleMomentMassLaw law,
    ColdFrozenDiffusionTripleEndpoint& result,
    std::string& error)
{
    double alpha = 0.0;
    if (!valid_state(unrounded, law, alpha, error))
    {
        return false;
    }
    const ColdFrozenDiffusionTripleEndpoint candidate{
        static_cast<float>(unrounded.mass_per_kg),
        static_cast<float>(unrounded.number_per_kg),
        static_cast<float>(unrounded.reflectivity_per_kg)};
    if (!valid_state({candidate.mass_per_kg, candidate.number_per_kg,
                      candidate.reflectivity_per_kg}, law, alpha, error))
    {
        return false;
    }
    result = candidate;
    return true;
}

} // namespace

bool apply_published_frozen_diffusion(
    const ColdFrozenDiffusionTripleState& initial,
    double dry_density_kg_m3,
    double duration_s,
    const ColdFrozenDiffusionAcceptedEvent& event,
    TripleMomentMassLaw mass_law,
    ColdFrozenDiffusionTripleEndpoint& result,
    std::string& error)
{
    if ((event.species != ColdFrozenDiffusionSpecies::Graupel &&
         event.species != ColdFrozenDiffusionSpecies::Hail) ||
        mass_law.coefficient != species_law(event.species).coefficient ||
        mass_law.exponent != 3.0 ||
        !std::isfinite(dry_density_kg_m3) || dry_density_kg_m3 <= 0.0 ||
        !std::isfinite(duration_s) || duration_s <= 0.0 ||
        event.dry_density_kg_m3 != dry_density_kg_m3 ||
        event.duration_s != duration_s ||
        event.entry_mass_per_kg != initial.mass_per_kg ||
        event.entry_number_per_kg != initial.number_per_kg)
    {
        error = "published frozen diffusion event source, context or cubic law differs";
        return false;
    }
    double alpha = 0.0;
    if (!valid_state(initial, mass_law, alpha, error) ||
        event.source_alpha != alpha)
    {
        if (error.empty())
        {
            error = "published frozen diffusion event alpha differs from live q/N/Z";
        }
        return false;
    }
    try
    {
        (void)make_published_frozen_diffusion_event(
            event.species, event.entry_mass_per_kg, event.entry_number_per_kg,
            event.source_alpha, event.admitted_mass_increment_per_kg,
            event.admitted_number_increment_per_kg,
            event.dry_density_kg_m3, event.duration_s);
    }
    catch (const std::runtime_error& failure)
    {
        error = failure.what();
        return false;
    }
    double reflectivity_increment = 0.0;
    if (initial.mass_per_kg > 0.0)
    {
        reflectivity_increment = initial.reflectivity_per_kg *
            (2.0 * event.admitted_mass_increment_per_kg / initial.mass_per_kg
             - event.admitted_number_increment_per_kg / initial.number_per_kg);
    }
    const ColdFrozenDiffusionTripleState endpoint{
        initial.mass_per_kg + event.admitted_mass_increment_per_kg,
        initial.number_per_kg + event.admitted_number_increment_per_kg,
        initial.reflectivity_per_kg + reflectivity_increment};
    if (!valid_state(endpoint, mass_law, alpha, error))
    {
        return false;
    }
    const ColdFrozenDiffusionTripleEndpoint candidate{
        static_cast<float>(endpoint.mass_per_kg),
        static_cast<float>(endpoint.number_per_kg),
        static_cast<float>(endpoint.reflectivity_per_kg)};
    if (!valid_state({candidate.mass_per_kg, candidate.number_per_kg,
                      candidate.reflectivity_per_kg}, mass_law, alpha, error))
    {
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

bool evaluate_published_triple_frozen_diffusion_cell(
    const std::array<double, 16>& cell,
    double duration_s,
    ColdFrozenDiffusionSpecies species,
    const ColdFrozenDiffusionTripleState& initial,
    std::array<float, 14>& rates,
    ColdFrozenDiffusionTripleEndpoint& result,
    std::string& error)
{
    if (species != ColdFrozenDiffusionSpecies::Graupel &&
        species != ColdFrozenDiffusionSpecies::Hail)
    {
        error = "published frozen diffusion species is invalid";
        return false;
    }
    const std::size_t mass_index = species == ColdFrozenDiffusionSpecies::Graupel ? 8 : 9;
    const std::size_t number_index = species == ColdFrozenDiffusionSpecies::Graupel ? 14 : 15;
    if (cell[mass_index] != initial.mass_per_kg ||
        cell[number_index] != initial.number_per_kg)
    {
        error = "published frozen diffusion cell differs from exact source q/N";
        return false;
    }
    double alpha = 0.0;
    if (!valid_state(initial, species_law(species), alpha, error))
    {
        return false;
    }
    const auto& p = default_cold_parameters;
    ColdParameters parameters{
        p[0], p[1], p[2], p[3], p[4],
        p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14],
        p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], true};
    if (species == ColdFrozenDiffusionSpecies::Graupel)
    {
        parameters.alpha_g_ = alpha;
    }
    else
    {
        parameters.alpha_h_ = alpha;
    }
    std::array<float, 14> candidate_rates{};
    ColdFrozenDiffusionAcceptedEventBuffer observer;
    observer.species = species;
    ColdFrozenDiffusionTripleEndpoint candidate;
    try
    {
        published_frozen_diffusion_cold_cell(
            parameters, cell, duration_s, candidate_rates, observer);
    }
    catch (const std::runtime_error& failure)
    {
        error = failure.what();
        return false;
    }
    if (observer.size != 1 ||
        !apply_published_frozen_diffusion(initial, cell[2], duration_s,
            observer.events[0], species_law(species), candidate, error))
    {
        if (error.empty())
        {
            error = "published frozen diffusion has no admitted event";
        }
        return false;
    }
    rates = candidate_rates;
    result = candidate;
    error.clear();
    return true;
}

bool apply_published_joint_frozen_diffusion(
    const ColdJointFrozenDiffusionTripleState& initial,
    double dry_density_kg_m3,
    double duration_s,
    const std::array<ColdFrozenDiffusionAcceptedEvent, 2>& events,
    ColdJointFrozenDiffusionTripleResult& result,
    std::string& error)
{
    if (events[0].species != ColdFrozenDiffusionSpecies::Graupel ||
        events[1].species != ColdFrozenDiffusionSpecies::Hail)
    {
        error = "published joint frozen diffusion event species or order differs";
        return false;
    }
    const auto graupel_law = species_law(ColdFrozenDiffusionSpecies::Graupel);
    const auto hail_law = species_law(ColdFrozenDiffusionSpecies::Hail);
    std::array<ColdFrozenDiffusionTripleEndpoint, 2> diffusion_endpoints;
    if (!apply_published_frozen_diffusion(initial.graupel, dry_density_kg_m3,
            duration_s, events[0], graupel_law, diffusion_endpoints[0], error) ||
        !apply_published_frozen_diffusion(initial.hail, dry_density_kg_m3,
            duration_s, events[1], hail_law, diffusion_endpoints[1], error))
    {
        return false;
    }
    ColdJointFrozenDiffusionTripleResult candidate;
    candidate.diffusion_events = events;
    candidate.diffusion_endpoint = {
        unrounded_diffusion_endpoint(initial.graupel, events[0]),
        unrounded_diffusion_endpoint(initial.hail, events[1])};
    candidate.unrounded = candidate.diffusion_endpoint;
    const auto& hail = candidate.diffusion_endpoint.hail;
    double hail_alpha = 0.0;
    if (!valid_state(hail, hail_law, hail_alpha, error))
    {
        return false;
    }
    const double diameter = hail.mass_per_kg > 0.0
        ? std::cbrt(hail.mass_per_kg / (hail_law.coefficient * hail.number_per_kg))
        : 0.0;
    if (!std::isfinite(diameter))
    {
        error = "published joint frozen diffusion terminal hail diameter is invalid";
        return false;
    }
    candidate.terminal_conversion = {
        false, hail.mass_per_kg, hail.number_per_kg, hail.reflectivity_per_kg,
        hail_alpha, diameter, 0.0, dry_density_kg_m3, duration_s};
    if (hail.mass_per_kg > 0.0 && diameter < 1.0e-3)
    {
        const double coefficient_ratio = hail_law.coefficient / graupel_law.coefficient;
        const double reflectivity_transfer = coefficient_ratio * coefficient_ratio
            * hail.reflectivity_per_kg;
        candidate.terminal_conversion.transferred = true;
        candidate.terminal_conversion.graupel_reflectivity_increment_per_kg =
            reflectivity_transfer;
        candidate.unrounded.graupel.mass_per_kg += hail.mass_per_kg;
        candidate.unrounded.graupel.number_per_kg += hail.number_per_kg;
        candidate.unrounded.graupel.reflectivity_per_kg += reflectivity_transfer;
        candidate.unrounded.hail = {
            hail.mass_per_kg - hail.mass_per_kg,
            hail.number_per_kg - hail.number_per_kg,
            hail.reflectivity_per_kg - hail.reflectivity_per_kg};
    }
    if (!validate_and_round_joint_endpoint(candidate.unrounded.graupel,
            graupel_law, candidate.graupel, error) ||
        !validate_and_round_joint_endpoint(candidate.unrounded.hail,
            hail_law, candidate.hail, error))
    {
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

bool evaluate_published_joint_triple_frozen_diffusion_cell(
    const std::array<double, 16>& cell,
    double duration_s,
    const ColdJointFrozenDiffusionTripleState& initial,
    std::array<float, 14>& rates,
    ColdJointFrozenDiffusionTripleResult& result,
    std::string& error)
{
    if (cell[8] != initial.graupel.mass_per_kg ||
        cell[14] != initial.graupel.number_per_kg ||
        cell[9] != initial.hail.mass_per_kg ||
        cell[15] != initial.hail.number_per_kg)
    {
        error = "published joint frozen diffusion cell differs from exact source q/N";
        return false;
    }
    double graupel_alpha = 0.0;
    double hail_alpha = 0.0;
    if (!valid_state(initial.graupel, species_law(ColdFrozenDiffusionSpecies::Graupel),
                     graupel_alpha, error) ||
        !valid_state(initial.hail, species_law(ColdFrozenDiffusionSpecies::Hail),
                     hail_alpha, error))
    {
        return false;
    }
    const auto& p = default_cold_parameters;
    ColdParameters parameters{
        p[0], p[1], p[2], p[3], p[4],
        p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14],
        p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], true};
    parameters.alpha_g_ = graupel_alpha;
    parameters.alpha_h_ = hail_alpha;
    std::array<float, 14> candidate_rates{};
    ColdJointFrozenDiffusionAcceptedEventBuffer observer;
    try
    {
        published_joint_frozen_diffusion_cold_cell(
            parameters, cell, duration_s, candidate_rates, observer);
    }
    catch (const std::runtime_error& failure)
    {
        error = failure.what();
        return false;
    }
    ColdJointFrozenDiffusionTripleResult candidate;
    if (observer.size != 2 ||
        !apply_published_joint_frozen_diffusion(initial, cell[2], duration_s,
            observer.events, candidate, error))
    {
        if (error.empty())
        {
            error = "published joint frozen diffusion has no paired admitted events";
        }
        return false;
    }
    if (candidate.terminal_conversion.transferred)
    {
        const auto& transfer = candidate.terminal_conversion;
        const std::array<double, 4> raw_rates{
            (observer.events[0].admitted_mass_increment_per_kg
                + transfer.entry_mass_per_kg) / duration_s,
            -initial.hail.mass_per_kg / duration_s,
            (observer.events[0].admitted_number_increment_per_kg
                + transfer.entry_number_per_kg) / duration_s,
            -initial.hail.number_per_kg / duration_s};
        for (double rate : raw_rates)
        {
            if (!published_joint_frozen_diffusion_rate_representable(rate))
            {
                error = "published joint frozen diffusion terminal rate is not FP32 representable";
                return false;
            }
        }
        candidate_rates[4] = static_cast<float>(raw_rates[0]);
        candidate_rates[5] = static_cast<float>(raw_rates[1]);
        candidate_rates[12] = static_cast<float>(raw_rates[2]);
        candidate_rates[13] = static_cast<float>(raw_rates[3]);
    }
    rates = candidate_rates;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
