#include "triple_cold_freezing.hpp"
#include "triple_sedimentation.hpp"

#include <array>
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

bool valid_state(const ColdFreezingTripleMomentState& state,
                 TripleMomentMassLaw law,
                 const char* label,
                 std::string& error)
{
    const std::array<double, 3> values{
        state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg};
    for (double value : values)
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = std::string("rain-freezing ") + label + " is nonfinite or negative";
            return false;
        }
    }
    if (state.mass_per_kg == 0.0 && state.number_per_kg == 0.0 &&
        state.reflectivity_per_kg == 0.0)
    {
        return true;
    }
    double alpha = 0.0;
    double lambda = 0.0;
    if (!diagnose_triple_gamma_shape(
            state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg,
            law, alpha, lambda, error))
    {
        error = std::string("rain-freezing ") + label + ": " + error;
        return false;
    }
    return true;
}

bool round_endpoint(const ColdFreezingTripleMomentState& state,
                    TripleMomentMassLaw law,
                    const char* label,
                    ColdFreezingTripleMomentEndpoint& result,
                    std::string& error)
{
    const std::array<double, 3> values{
        state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg};
    for (double value : values)
    {
        if (!fp32_range(value))
        {
            error = std::string("rain-freezing ") + label + " is not FP32 representable";
            return false;
        }
    }
    const ColdFreezingTripleMomentEndpoint candidate{
        static_cast<float>(state.mass_per_kg),
        static_cast<float>(state.number_per_kg),
        static_cast<float>(state.reflectivity_per_kg)};
    if (!valid_state({candidate.mass_per_kg, candidate.number_per_kg,
                      candidate.reflectivity_per_kg}, law, label, error))
    {
        return false;
    }
    result = candidate;
    return true;
}

} // namespace

bool apply_equal_fraction_rain_freezing(
    const ColdFreezingTripleMomentState& rain,
    const ColdFreezingTripleMomentState& frozen,
    double dry_density_kg_m3,
    double duration_s,
    const ColdRainFreezingAcceptedEvent& event,
    TripleMomentMassLaw rain_mass_law,
    TripleMomentMassLaw frozen_mass_law,
    ColdFreezingTripleMomentResult& result,
    std::string& error)
{
    if (event.destination != ColdRainFreezingDestination::Graupel &&
        event.destination != ColdRainFreezingDestination::Hail)
    {
        error = "rain-freezing destination is invalid";
        return false;
    }
    const auto rain_parameters = current_nimbus_triple_sedimentation_parameters(
        TripleMomentSpecies::Rain);
    const auto frozen_parameters = current_nimbus_triple_sedimentation_parameters(
        event.destination == ColdRainFreezingDestination::Graupel ?
            TripleMomentSpecies::Graupel : TripleMomentSpecies::Hail);
    if (rain_mass_law.coefficient != rain_parameters.mass_law.coefficient ||
        rain_mass_law.exponent != 3.0 ||
        frozen_mass_law.coefficient != frozen_parameters.mass_law.coefficient ||
        frozen_mass_law.exponent != 3.0)
    {
        error = "rain-freezing requires the selected current cubic mass laws";
        return false;
    }
    if (!std::isfinite(dry_density_kg_m3) || !(dry_density_kg_m3 > 0.0) ||
        !std::isfinite(duration_s) || !(duration_s > 0.0) ||
        event.dry_density_kg_m3 != dry_density_kg_m3 ||
        event.duration_s != duration_s ||
        event.entry_rain_mass_per_kg != rain.mass_per_kg ||
        event.entry_rain_number_per_kg != rain.number_per_kg)
    {
        error = "rain-freezing event entry differs from the supplied source or context";
        return false;
    }
    if (!valid_state(rain, rain_mass_law, "source input", error) ||
        !valid_state(frozen, frozen_mass_law, "destination input", error))
    {
        return false;
    }
    for (const auto& state : {rain, frozen})
    {
        for (double value : {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg})
        {
            if (!fp32_range(value))
            {
                error = "rain-freezing input is outside the FP32 value range";
                return false;
            }
        }
    }
    if (event.admitted_mass_increment_per_kg == 0.0 &&
        event.admitted_number_increment_per_kg == 0.0)
    {
        if (event.admitted_fraction != 0.0)
        {
            error = "rain-freezing zero event has a nonzero fraction";
            return false;
        }
    }
    else
    {
        try
        {
            const auto checked = make_cold_rain_freezing_accepted_event(
                event.destination, event.entry_rain_mass_per_kg,
                event.entry_rain_number_per_kg,
                event.admitted_mass_increment_per_kg,
                event.admitted_number_increment_per_kg,
                event.dry_density_kg_m3, event.duration_s);
            if (checked.admitted_fraction != event.admitted_fraction)
            {
                error = "rain-freezing event fraction differs from admitted number";
                return false;
            }
        }
        catch (const std::runtime_error& failure)
        {
            error = failure.what();
            return false;
        }
    }
    const double source_reflectivity_increment =
        event.admitted_fraction * rain.reflectivity_per_kg;
    const double coefficient_ratio =
        rain_mass_law.coefficient / frozen_mass_law.coefficient;
    // Cubic-law diameter remapping preserves each particle's mass; its sixth
    // moment therefore scales by the squared mass-coefficient ratio.
    const double frozen_reflectivity_increment =
        source_reflectivity_increment * coefficient_ratio * coefficient_ratio;
    const ColdFreezingTripleMomentState next_rain{
        rain.mass_per_kg - event.admitted_mass_increment_per_kg,
        rain.number_per_kg - event.admitted_number_increment_per_kg,
        rain.reflectivity_per_kg - source_reflectivity_increment};
    const ColdFreezingTripleMomentState next_frozen{
        frozen.mass_per_kg + event.admitted_mass_increment_per_kg,
        frozen.number_per_kg + event.admitted_number_increment_per_kg,
        frozen.reflectivity_per_kg + frozen_reflectivity_increment};
    if (!valid_state(next_rain, rain_mass_law, "source endpoint", error) ||
        !valid_state(next_frozen, frozen_mass_law, "destination endpoint", error))
    {
        return false;
    }
    ColdFreezingTripleMomentResult candidate;
    if (!round_endpoint(next_rain, rain_mass_law, "rounded source endpoint", candidate.rain, error) ||
        !round_endpoint(next_frozen, frozen_mass_law, "rounded destination endpoint", candidate.frozen, error))
    {
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
