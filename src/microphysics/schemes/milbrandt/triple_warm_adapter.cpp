#include "triple_warm_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::milbrandt_detail
{
namespace
{

bool finite_event(const WarmRainAcceptedEvent& event, double density)
{
    return std::isfinite(event.cloud_mass_rate_per_kg_s) &&
        std::isfinite(event.rain_mass_rate_per_kg_s) &&
        std::isfinite(event.vapor_mass_rate_per_kg_s) &&
        std::isfinite(event.cloud_number_rate_per_m3_s) &&
        std::isfinite(event.rain_number_rate_per_m3_s) &&
        std::isfinite(event.temperature_rate_k_s) &&
        std::isfinite(event.cloud_donor_scale) &&
        std::isfinite(event.seed_diameter_m) &&
        event.dry_density_kg_m3 == density;
}

bool no_other_currency(const WarmRainAcceptedEvent& event)
{
    return event.vapor_mass_rate_per_kg_s == 0.0 &&
        event.temperature_rate_k_s == 0.0;
}

bool no_rain_currency(const WarmRainAcceptedEvent& event)
{
    return event.rain_mass_rate_per_kg_s == 0.0 &&
        event.rain_number_rate_per_m3_s == 0.0;
}

bool endpoint_matches(float source_input, float source_rate,
                      float incoming_rate, double dt, float candidate)
{
    const double expected = static_cast<double>(source_input) +
        dt * (static_cast<double>(source_rate) -
              static_cast<double>(incoming_rate));
    const double scale = std::max({std::abs(static_cast<double>(source_input)),
                                   std::abs(expected),
                                   std::abs(static_cast<double>(candidate)),
                                   1.0e-30});
    return std::isfinite(expected) &&
        std::abs(expected - static_cast<double>(candidate)) <=
            8.0 * std::numeric_limits<float>::epsilon() * scale;
}

bool map_event(const WarmRainAcceptedEvent& source,
               double dt,
               double expected_density,
               bool triple_producer,
               AcceptedRainPhaseBatch& batch,
               std::string& error)
{
    const double density = source.dry_density_kg_m3;
    if (!(std::isfinite(density) && density > 0.0) ||
        !finite_event(source, expected_density) ||
        (source.kind != WarmRainAcceptedEventKind::Evaporation &&
         !no_other_currency(source)))
    {
        error = "observed warm event has malformed units or currency";
        return false;
    }
    const bool scaled_accretion = triple_producer &&
        source.kind == WarmRainAcceptedEventKind::Accretion &&
        source.cloud_donor_scale > 0.0 &&
        source.cloud_donor_scale <= 1.0;
    if (source.kind != WarmRainAcceptedEventKind::Autoconversion &&
        ((!scaled_accretion && source.cloud_donor_scale != 1.0) ||
         source.seed_diameter_m != 0.0 ||
         !source.seed_mass_number_coherent))
    {
        error = "non-autoconversion warm event carries seed or donor metadata";
        return false;
    }
    AcceptedRainEvent event;
    RainEventPhase phase = RainEventPhase::SimultaneousPhysical;
    switch (source.kind)
    {
    case WarmRainAcceptedEventKind::Autoconversion:
    {
        if (!source.seed_mass_number_coherent ||
            !(source.cloud_donor_scale > 0.0 &&
              source.cloud_donor_scale <= 1.0) ||
            (!triple_producer && source.cloud_donor_scale != 1.0) ||
            !(source.seed_diameter_m > 0.0) ||
            !(source.rain_mass_rate_per_kg_s > 0.0) ||
            !(source.rain_number_rate_per_m3_s > 0.0) ||
            source.cloud_mass_rate_per_kg_s !=
                -source.rain_mass_rate_per_kg_s ||
            source.cloud_number_rate_per_m3_s != 0.0)
        {
            error = "incoherent admitted autoconversion mass/number seed";
            return false;
        }
        const double coherent_number =
            source.rain_mass_rate_per_kg_s * density /
            (523.598775598 * std::pow(source.seed_diameter_m, 3.0));
        const double scale = std::max(
            std::abs(coherent_number),
            std::abs(source.rain_number_rate_per_m3_s));
        if (!std::isfinite(coherent_number) ||
            std::abs(coherent_number -
                     source.rain_number_rate_per_m3_s) >
                32.0 * std::numeric_limits<double>::epsilon() * scale ||
            !make_admitted_autoconversion_event(
                source.rain_mass_rate_per_kg_s,
                source.rain_number_rate_per_m3_s,
                density, dt, event, error))
        {
            if (error.empty())
            {
                error = "autoconversion seed diameter and admitted number disagree";
            }
            return false;
        }
        break;
    }
    case WarmRainAcceptedEventKind::Accretion:
        if (!(source.rain_mass_rate_per_kg_s > 0.0) ||
            source.cloud_mass_rate_per_kg_s !=
                -source.rain_mass_rate_per_kg_s ||
            source.rain_number_rate_per_m3_s != 0.0 ||
            source.cloud_number_rate_per_m3_s > 0.0)
        {
            error = "admitted rain accretion currency is inconsistent";
            return false;
        }
        if (!make_admitted_same_category_event(
                AcceptedRainEventKind::Accretion,
                source.rain_mass_rate_per_kg_s, 0.0,
                dt, event, error))
        {
            return false;
        }
        break;
    case WarmRainAcceptedEventKind::RainSelfCollection:
        if (source.cloud_mass_rate_per_kg_s != 0.0 ||
            source.rain_mass_rate_per_kg_s != 0.0 ||
            source.cloud_number_rate_per_m3_s != 0.0 ||
            !(source.rain_number_rate_per_m3_s < 0.0) ||
            !make_admitted_same_category_event(
                AcceptedRainEventKind::SelfCollection, 0.0,
                source.rain_number_rate_per_m3_s / density,
                dt, event, error))
        {
            if (error.empty())
            {
                error = "admitted rain self-collection currency is inconsistent";
            }
            return false;
        }
        break;
    case WarmRainAcceptedEventKind::PreRainNumberRegularization:
    case WarmRainAcceptedEventKind::PostRainNumberRegularization:
    case WarmRainAcceptedEventKind::RainNumberClip:
    {
        if (triple_producer &&
            source.kind == WarmRainAcceptedEventKind::RainNumberClip)
        {
            error = "triple warm rain-number clip requires a reviewed Z correction";
            return false;
        }
        if (source.cloud_mass_rate_per_kg_s != 0.0 ||
            source.rain_mass_rate_per_kg_s != 0.0 ||
            source.cloud_number_rate_per_m3_s != 0.0)
        {
            error = "rain number correction carries other currency";
            return false;
        }
        const auto kind = source.kind ==
            WarmRainAcceptedEventKind::PreRainNumberRegularization
            ? AcceptedRainEventKind::PreRainNumberRegularization
            : source.kind ==
                WarmRainAcceptedEventKind::PostRainNumberRegularization
                ? AcceptedRainEventKind::PostRainNumberRegularization
                : AcceptedRainEventKind::RainNumberClip;
        phase = source.kind ==
            WarmRainAcceptedEventKind::PreRainNumberRegularization
            ? RainEventPhase::PreRegularization
            : RainEventPhase::NumericalCorrection;
        if (!make_admitted_rain_number_correction_event(
                kind, source.rain_number_rate_per_m3_s,
                density, dt, event, error))
        {
            return false;
        }
        break;
    }
    case WarmRainAcceptedEventKind::RainToCloud:
    case WarmRainAcceptedEventKind::RainReset:
    {
        if (triple_producer)
        {
            error = "triple warm rain reclassification or reset has no reviewed cloud-Z owner";
            return false;
        }
        phase = RainEventPhase::NumericalCorrection;
        const auto kind = source.kind == WarmRainAcceptedEventKind::RainToCloud
            ? AcceptedRainEventKind::RainToCloud
            : AcceptedRainEventKind::RainReset;
        if (!make_admitted_rain_removal_event(
                kind, source.cloud_mass_rate_per_kg_s,
                source.rain_mass_rate_per_kg_s,
                source.cloud_number_rate_per_m3_s,
                source.rain_number_rate_per_m3_s,
                density, dt, event, error))
        {
            return false;
        }
        break;
    }
    case WarmRainAcceptedEventKind::Evaporation:
        phase = RainEventPhase::Evaporation;
        {
            const double expected_temperature_rate =
                microphysics_constants::L_v /
                microphysics_constants::cp *
                source.rain_mass_rate_per_kg_s;
            const double temperature_scale = std::max(
                std::abs(expected_temperature_rate),
                std::abs(source.temperature_rate_k_s));
            if (!std::isfinite(expected_temperature_rate) ||
                std::abs(source.temperature_rate_k_s -
                         expected_temperature_rate) >
                    32.0 * std::numeric_limits<double>::epsilon() *
                        temperature_scale)
            {
                error = "admitted rain evaporation latent-cooling provenance is inconsistent";
                return false;
            }
        }
        if (source.cloud_mass_rate_per_kg_s != 0.0 ||
            source.cloud_number_rate_per_m3_s != 0.0 ||
            !(source.rain_mass_rate_per_kg_s < 0.0) ||
            source.vapor_mass_rate_per_kg_s !=
                -source.rain_mass_rate_per_kg_s ||
            source.rain_number_rate_per_m3_s > 0.0 ||
            !make_admitted_same_category_event(
                AcceptedRainEventKind::Evaporation,
                source.rain_mass_rate_per_kg_s,
                source.rain_number_rate_per_m3_s / density,
                dt, event, error))
        {
            if (error.empty())
            {
                error = "admitted rain evaporation currency is inconsistent";
            }
            return false;
        }
        break;
    case WarmRainAcceptedEventKind::CloudSelfCollection:
    case WarmRainAcceptedEventKind::CloudNumberClip:
        if (!no_rain_currency(source) ||
            source.cloud_mass_rate_per_kg_s != 0.0 ||
            source.vapor_mass_rate_per_kg_s != 0.0 ||
            source.temperature_rate_k_s != 0.0 ||
            (source.kind == WarmRainAcceptedEventKind::CloudSelfCollection &&
             !(source.cloud_number_rate_per_m3_s < 0.0)) ||
            (source.kind == WarmRainAcceptedEventKind::CloudNumberClip &&
             !(source.cloud_number_rate_per_m3_s > 0.0)))
        {
            error = "cloud-only warm event carries rain currency";
            return false;
        }
        return true;
    default:
        error = "unsupported observed warm event requires science review";
        return false;
    }
    return append_admitted_rain_phase_event(batch, phase, event, error);
}

bool finish_observed_rain_cell(
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const RainTripleMomentState& initial_rain,
    const WarmRainTendencies& pending_rates,
    const WarmRainAcceptedEventBuffer& observed,
    bool triple_producer,
    WarmRainTendencies& output_rates,
    RainTripleMomentState& output_rain,
    std::string& error)
{
    AcceptedRainPhaseBatch batch;
    batch.duration_s = dt;
    std::array<bool, 15> seen_source_kinds{};
    for (std::size_t index = 0; index < observed.size; ++index)
    {
        const auto kind_index = static_cast<std::size_t>(
            observed.events[index].kind);
        if (kind_index >= seen_source_kinds.size() ||
            seen_source_kinds[kind_index])
        {
            error = "observed warm cell has a duplicate or unknown channel";
            return false;
        }
        seen_source_kinds[kind_index] = true;
        if (!map_event(observed.events[index], dt, cell.density,
                       triple_producer, batch, error))
        {
            return false;
        }
    }
    RainTripleMomentState pending_rain;
    if (!apply_admitted_rain_phases(initial_rain, batch, pending_rain, error))
    {
        return false;
    }
    if (!endpoint_matches(cell.rain, pending_rates.rain,
                          incoming_rates.rain, dt,
                          pending_rain.mass_per_kg) ||
        !endpoint_matches(cell.rain_number,
                          pending_rates.rain_number,
                          incoming_rates.rain_number, dt,
                          pending_rain.number_per_kg))
    {
        error = "phase-aware rain q/N endpoint differs from warm-cell output";
        return false;
    }
    output_rates = pending_rates;
    output_rain = pending_rain;
    error.clear();
    return true;
}

} // namespace

bool map_observed_rain_event(
    const WarmRainAcceptedEvent& event,
    double dt,
    double expected_density,
    AcceptedRainPhaseBatch& batch,
    std::string& error)
{
    error.clear();
    return map_event(event, dt, expected_density, false, batch, error);
}

bool map_triple_produced_rain_event(
    const WarmRainAcceptedEvent& event,
    double dt,
    double expected_density,
    AcceptedRainPhaseBatch& batch,
    std::string& error)
{
    error.clear();
    return map_event(event, dt, expected_density, true, batch, error);
}

bool map_dynamic_triple_produced_rain_event(
    const WarmRainAcceptedEvent& source,
    double dt,
    double expected_density,
    AcceptedRainPhaseBatch& batch,
    std::string& error)
{
    error.clear();
    if (source.kind == WarmRainAcceptedEventKind::RainToCloud)
    {
        return map_event(source, dt, expected_density, false, batch, error);
    }
    if (source.kind != WarmRainAcceptedEventKind::RainBreakup)
    {
        return map_event(source, dt, expected_density, true, batch, error);
    }
    if (!std::isfinite(expected_density) || expected_density <= 0.0 ||
        !finite_event(source, expected_density) || !no_other_currency(source) ||
        source.cloud_mass_rate_per_kg_s != 0.0 || source.rain_mass_rate_per_kg_s != 0.0 ||
        source.cloud_number_rate_per_m3_s != 0.0 || source.rain_number_rate_per_m3_s <= 0.0 ||
        source.cloud_donor_scale != 1.0 || source.seed_diameter_m != 0.0 ||
        !source.seed_mass_number_coherent)
    {
        error = "dynamic rain breakup currency or provenance is inconsistent";
        return false;
    }
    AcceptedRainEvent event;
    if (!make_admitted_same_category_event(AcceptedRainEventKind::Breakup, 0.0,
            source.rain_number_rate_per_m3_s / expected_density, dt, event, error))
    {
        return false;
    }
    return append_admitted_rain_phase_event(batch, RainEventPhase::SimultaneousPhysical, event, error);
}

bool evaluate_observed_triple_warm_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const RainTripleMomentState& initial_rain,
    WarmRainTendencies& output_rates,
    RainTripleMomentState& output_rain,
    std::string& error)
{
    error.clear();
    if (!(std::isfinite(dt) && dt > 0.0) ||
        parameters.alpha_r != 0.0 ||
        parameters.c_r != 523.598775598 ||
        parameters.d_r != 3.0 ||
        initial_rain.mass_per_kg != cell.rain ||
        initial_rain.number_per_kg != cell.rain_number)
    {
        error = "initial triple rain q/N, default warm law or timestep is invalid";
        return false;
    }
    WarmRainTendencies pending_rates = incoming_rates;
    WarmRainAcceptedEventBuffer observed;
    try
    {
        warm_rain_cell(parameters, cell, dt, pending_rates, &observed);
    }
    catch (const std::exception& failure)
    {
        error = std::string("warm-event observation failed: ") +
            failure.what();
        return false;
    }
    return finish_observed_rain_cell(
        cell, dt, incoming_rates, initial_rain, pending_rates, observed,
        false, output_rates, output_rain, error);
}

bool evaluate_triple_only_warm_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const RainTripleMomentState& initial_rain,
    WarmRainTendencies& output_rates,
    RainTripleMomentState& output_rain,
    std::string& error)
{
    error.clear();
    if (!(std::isfinite(dt) && dt > 0.0) ||
        parameters.alpha_r != 0.0 ||
        parameters.c_r != 523.598775598 ||
        parameters.d_r != 3.0 ||
        initial_rain.mass_per_kg != cell.rain ||
        initial_rain.number_per_kg != cell.rain_number)
    {
        error = "initial triple rain q/N, default warm law or timestep is invalid";
        return false;
    }
    TripleWarmRainResult observed;
    try
    {
        triple_warm_rain_cell(
            parameters, cell, dt, incoming_rates, observed);
    }
    catch (const std::exception& failure)
    {
        error = std::string("triple warm-event production failed: ") +
            failure.what();
        return false;
    }
    if (observed.duration_s() != dt ||
        observed.dry_density_kg_m3() != cell.density)
    {
        error = "triple warm-event producer provenance differs";
        return false;
    }
    return finish_observed_rain_cell(
        cell, dt, incoming_rates, initial_rain,
        observed.rates(), observed.events(), true,
        output_rates, output_rain, error);
}

} // namespace tmv::milbrandt_detail
