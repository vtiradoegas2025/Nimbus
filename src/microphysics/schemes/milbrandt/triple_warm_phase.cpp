#include "triple_warm_phase.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr double kRainMassCoefficient = 523.598775598;
constexpr double kAutoconversionSeedG = 5.6;

struct DoubleRainState
{
    double mass = 0.0;
    double number = 0.0;
    double reflectivity = 0.0;
};

bool valid_state(const DoubleRainState& state)
{
    if (!std::isfinite(state.mass) || !std::isfinite(state.number) ||
        !std::isfinite(state.reflectivity) || state.mass < 0.0 ||
        state.number < 0.0 || state.reflectivity < 0.0)
    {
        return false;
    }
    if (state.mass == 0.0 || state.number == 0.0 ||
        state.reflectivity == 0.0)
    {
        return state.mass == 0.0 && state.number == 0.0 &&
            state.reflectivity == 0.0;
    }
    const double log_lower = 2.0 * std::log(state.mass) -
        std::log(state.number) - 2.0 * std::log(kRainMassCoefficient);
    return std::isfinite(log_lower) &&
        std::log(state.reflectivity) >= log_lower;
}

bool canonical_event(const AcceptedRainEvent& event,
                     std::string& error)
{
    AcceptedRainEvent expected;
    bool valid = false;
    switch (event.kind)
    {
    case AcceptedRainEventKind::Autoconversion:
        valid = make_admitted_autoconversion_event(
            event.admitted_mass_rate_per_kg_s,
            event.source_number_rate_per_m3_s,
            event.source_dry_density_kg_m3,
            event.duration_s, expected, error);
        break;
    case AcceptedRainEventKind::Evaporation:
    case AcceptedRainEventKind::SelfCollection:
    case AcceptedRainEventKind::Accretion:
    case AcceptedRainEventKind::Breakup:
        valid = make_admitted_same_category_event(
            event.kind, event.admitted_mass_rate_per_kg_s,
            event.admitted_number_rate_per_kg_s,
            event.duration_s, expected, error);
        break;
    case AcceptedRainEventKind::PreRainNumberRegularization:
    case AcceptedRainEventKind::RainNumberClip:
    case AcceptedRainEventKind::PostRainNumberRegularization:
        valid = make_admitted_rain_number_correction_event(
            event.kind, event.source_number_rate_per_m3_s,
            event.source_dry_density_kg_m3,
            event.duration_s, expected, error);
        break;
    case AcceptedRainEventKind::RainToCloud:
    case AcceptedRainEventKind::RainReset:
        valid = make_admitted_rain_removal_event(
            event.kind, event.source_cloud_mass_rate_per_kg_s,
            event.admitted_mass_rate_per_kg_s,
            event.source_cloud_number_rate_per_m3_s,
            event.source_number_rate_per_m3_s,
            event.source_dry_density_kg_m3,
            event.duration_s, expected, error);
        break;
    }
    if (!valid || expected.kind != event.kind ||
        expected.admitted_mass_rate_per_kg_s !=
            event.admitted_mass_rate_per_kg_s ||
        expected.admitted_number_rate_per_kg_s !=
            event.admitted_number_rate_per_kg_s ||
        expected.duration_s != event.duration_s ||
        expected.source_number_rate_per_m3_s !=
            event.source_number_rate_per_m3_s ||
        expected.source_dry_density_kg_m3 !=
            event.source_dry_density_kg_m3 ||
        expected.source_cloud_mass_rate_per_kg_s !=
            event.source_cloud_mass_rate_per_kg_s ||
        expected.source_cloud_number_rate_per_m3_s !=
            event.source_cloud_number_rate_per_m3_s)
    {
        if (valid)
        {
            error = "rain event differs from its typed admitted provenance";
        }
        return false;
    }
    return true;
}

bool same_duration(double requested, double actual)
{
    return std::isfinite(requested) && requested > 0.0 &&
        actual == requested;
}

bool matches_full_removal(double value, double rate, double dt)
{
    const double residual = value + dt * rate;
    const double scale = std::max(std::abs(value), std::abs(dt * rate));
    return std::isfinite(residual) &&
        std::abs(residual) <=
            8.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool apply_number_correction(DoubleRainState& state,
                             const AcceptedRainEvent& event,
                             std::string& error,
                             DynamicWarmStepFailure* failure = nullptr)
{
    if (!(state.mass > 0.0 && state.number > 0.0 &&
          state.reflectivity > 0.0))
    {
        error = "rain number correction has no represented rain population";
        return false;
    }
    const double next_number = state.number +
        event.duration_s * event.admitted_number_rate_per_kg_s;
    if (!(std::isfinite(next_number) && next_number > 0.0))
    {
        if (failure)
        {
            *failure = DynamicWarmStepFailure::IntervalRejected;
        }
        error = "rain number clip/reset grouping is unrepresentable";
        return false;
    }
    state.reflectivity *= state.number / next_number;
    state.number = next_number;
    if (!valid_state(state))
    {
        if (failure)
        {
            *failure = DynamicWarmStepFailure::IntervalRejected;
        }
        error = "rain number correction endpoint is unrealizable";
        return false;
    }
    return true;
}

bool apply_removal(DoubleRainState& state,
                   const AcceptedRainEvent& event,
                   std::string& error)
{
    if (!(state.mass > 0.0 && state.number > 0.0 &&
          state.reflectivity > 0.0) ||
        !matches_full_removal(state.mass,
                              event.admitted_mass_rate_per_kg_s,
                              event.duration_s) ||
        !matches_full_removal(state.number,
                              event.admitted_number_rate_per_kg_s,
                              event.duration_s))
    {
        error = "rain removal does not match the represented full endpoint";
        return false;
    }
    state = {};
    return true;
}

} // namespace

bool append_admitted_rain_phase_event(
    AcceptedRainPhaseBatch& batch,
    RainEventPhase phase,
    const AcceptedRainEvent& event,
    std::string& error)
{
    error.clear();
    if (batch.size >= batch.events.size() ||
        !same_duration(batch.duration_s, event.duration_s) ||
        (batch.size > 0 &&
         static_cast<int>(phase) <
             static_cast<int>(batch.events[batch.size - 1].phase)) ||
        !canonical_event(event, error))
    {
        if (error.empty())
        {
            error = "rain phase event is out of order or exceeds capacity";
        }
        return false;
    }
    batch.events[batch.size++] = {phase, event};
    error.clear();
    return true;
}

bool finalize_rain_phase_endpoint(
    double mass_per_kg,
    double number_per_kg,
    double reflectivity_per_kg,
    RainTripleMomentState& result,
    std::string& error)
{
    const DoubleRainState state{
        mass_per_kg, number_per_kg, reflectivity_per_kg};
    if (!valid_state(state) ||
        state.mass > std::numeric_limits<float>::max() ||
        state.number > std::numeric_limits<float>::max() ||
        state.reflectivity > std::numeric_limits<float>::max())
    {
        error = "rain phase endpoint is invalid or exceeds FP32 storage";
        return false;
    }
    const RainTripleMomentState candidate{
        static_cast<float>(state.mass),
        static_cast<float>(state.number),
        static_cast<float>(state.reflectivity)};
    if (!valid_state({candidate.mass_per_kg,
                      candidate.number_per_kg,
                      candidate.reflectivity_per_kg}))
    {
        error = "rounded rain phase endpoint is unrealizable";
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

template <typename Initial>
static bool apply_admitted_rain_phases_impl(
    const Initial& initial,
    const AcceptedRainPhaseBatch& batch,
    RainTripleMomentState& result,
    ExactRainTripleMomentState* exact,
    const AdmittedRainEvaporationEndpoint* evaporation_endpoint,
    std::string& error,
    bool publish_rounded = true,
    DynamicWarmStepFailure* failure = nullptr)
{
    error.clear();
    if (failure)
    {
        *failure = DynamicWarmStepFailure::InvalidSource;
    }
    if (batch.size > batch.events.size() ||
        !(std::isfinite(batch.duration_s) && batch.duration_s > 0.0))
    {
        error = "rain phase batch has an invalid resource bound or duration";
        return false;
    }
    DoubleRainState state{initial.mass_per_kg,
                          initial.number_per_kg,
                          initial.reflectivity_per_kg};
    if (!valid_state(state))
    {
        error = "initial rain phase q/N/Z state is unrealizable";
        return false;
    }
    std::array<bool, 10> seen_kinds{};
    std::size_t evaporation_count = 0;
    for (std::size_t event_index = 0; event_index < batch.size;
         ++event_index)
    {
        const auto kind_index = static_cast<std::size_t>(
            batch.events[event_index].event.kind);
        if (kind_index >= seen_kinds.size() ||
            seen_kinds[kind_index])
        {
            error = "rain phase batch contains a duplicate or unknown event kind";
            return false;
        }
        seen_kinds[kind_index] = true;
        if (batch.events[event_index].event.kind == AcceptedRainEventKind::Evaporation)
        {
            ++evaporation_count;
        }
    }
    if (evaporation_endpoint)
    {
        const auto& e = *evaporation_endpoint;
        if (e.present != (evaporation_count == 1) ||
            (!e.present && (e.mass_before_per_kg != 0.0 || e.number_before_per_kg != 0.0 ||
                           e.admitted_mass_per_kg != 0.0 || e.mass_after_per_kg != 0.0 ||
                           e.number_after_per_kg != 0.0 || e.donor_exhausted)))
        {
            error = "rain evaporation endpoint presence or empty metadata differs";
            return false;
        }
    }
    int previous_phase = -1;
    std::size_t index = 0;
    while (index < batch.size)
    {
        const RainEventPhase phase = batch.events[index].phase;
        if (static_cast<int>(phase) < previous_phase ||
            static_cast<int>(phase) > static_cast<int>(RainEventPhase::Evaporation))
        {
            error = "rain phase event order is invalid";
            return false;
        }
        previous_phase = static_cast<int>(phase);
        if (phase == RainEventPhase::SimultaneousPhysical)
        {
            const DoubleRainState pre_physical = state;
            double mass_rate = 0.0;
            double number_rate = 0.0;
            double reflectivity_rate = 0.0;
            do
            {
                const AcceptedRainEvent& event = batch.events[index].event;
                if (!same_duration(batch.duration_s, event.duration_s) ||
                    !canonical_event(event, error))
                {
                    return false;
                }
                if (event.kind == AcceptedRainEventKind::Autoconversion)
                {
                    const double rate =
                        (kAutoconversionSeedG /
                         (kRainMassCoefficient * kRainMassCoefficient)) *
                        event.admitted_mass_rate_per_kg_s *
                        event.admitted_mass_rate_per_kg_s /
                        event.admitted_number_rate_per_kg_s;
                    reflectivity_rate += rate;
                }
                else if (event.kind == AcceptedRainEventKind::Accretion ||
                         event.kind == AcceptedRainEventKind::SelfCollection ||
                         event.kind == AcceptedRainEventKind::Breakup)
                {
                    if (!(pre_physical.mass > 0.0 &&
                          pre_physical.number > 0.0 &&
                          pre_physical.reflectivity > 0.0))
                    {
                        error = "same-phase B3 source lacks pre-physical rain state";
                        return false;
                    }
                    reflectivity_rate += pre_physical.reflectivity *
                        (2.0 * event.admitted_mass_rate_per_kg_s /
                             pre_physical.mass -
                         event.admitted_number_rate_per_kg_s /
                             pre_physical.number);
                }
                else
                {
                    error = "unsupported event in simultaneous rain phase";
                    return false;
                }
                mass_rate += event.admitted_mass_rate_per_kg_s;
                number_rate += event.admitted_number_rate_per_kg_s;
                ++index;
            }
            while (index < batch.size &&
                   batch.events[index].phase == phase);
            state.mass += batch.duration_s * mass_rate;
            state.number += batch.duration_s * number_rate;
            state.reflectivity += batch.duration_s * reflectivity_rate;
            if (!valid_state(state))
            {
                if (failure)
                {
                    *failure = DynamicWarmStepFailure::IntervalRejected;
                }
                error = "simultaneous rain physical phase is unrealizable";
                return false;
            }
            continue;
        }
        const AcceptedRainEvent& event = batch.events[index].event;
        if (!same_duration(batch.duration_s, event.duration_s) ||
            !canonical_event(event, error))
        {
            return false;
        }
        if (phase == RainEventPhase::PreRegularization)
        {
            if (event.kind !=
                AcceptedRainEventKind::PreRainNumberRegularization ||
                !apply_number_correction(state, event, error, failure))
            {
                if (error.empty())
                {
                    error = "unsupported pre-physical rain event";
                }
                return false;
            }
        }
        else if (phase == RainEventPhase::NumericalCorrection)
        {
            if (event.kind == AcceptedRainEventKind::RainNumberClip ||
                event.kind == AcceptedRainEventKind::PostRainNumberRegularization)
            {
                if (!apply_number_correction(state, event, error, failure))
                {
                    return false;
                }
            }
            else if (event.kind == AcceptedRainEventKind::RainToCloud ||
                     event.kind == AcceptedRainEventKind::RainReset)
            {
                if (!apply_removal(state, event, error))
                {
                    return false;
                }
            }
            else
            {
                error = "unsupported numerical rain event";
                return false;
            }
        }
        else if (phase == RainEventPhase::Evaporation)
        {
            if (event.kind != AcceptedRainEventKind::Evaporation ||
                !(state.mass > 0.0 && state.number > 0.0 &&
                  state.reflectivity > 0.0))
            {
                error = "unsupported or unrepresented rain evaporation";
                return false;
            }
            if (evaporation_endpoint)
            {
                const auto& e = *evaporation_endpoint;
                const auto close = [](double initial, double expected, double actual)
                {
                    const double scale = std::max({std::abs(initial), std::abs(expected), std::abs(actual), 1.0e-30});
                    return std::isfinite(expected) && std::isfinite(actual) &&
                        std::abs(expected - actual) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
                };
                if (!e.present || !std::isfinite(e.mass_before_per_kg) || e.mass_before_per_kg <= 0.0 ||
                    !std::isfinite(e.number_before_per_kg) || e.number_before_per_kg <= 0.0 ||
                    !std::isfinite(e.admitted_mass_per_kg) || e.admitted_mass_per_kg <= 0.0 ||
                    e.admitted_mass_per_kg > e.mass_before_per_kg ||
                    !std::isfinite(e.mass_after_per_kg) || e.mass_after_per_kg < 0.0 ||
                    e.mass_after_per_kg != e.mass_before_per_kg - e.admitted_mass_per_kg ||
                    !std::isfinite(e.number_after_per_kg) || e.number_after_per_kg < 0.0 ||
                    e.donor_exhausted != (e.admitted_mass_per_kg == e.mass_before_per_kg) ||
                    !close(e.mass_before_per_kg, state.mass, e.mass_before_per_kg) ||
                    !close(e.number_before_per_kg, state.number, e.number_before_per_kg) ||
                    !close(e.number_before_per_kg,
                           e.number_before_per_kg * (e.mass_after_per_kg / e.mass_before_per_kg),
                           e.number_after_per_kg) ||
                    !close(e.mass_before_per_kg, event.duration_s * event.admitted_mass_rate_per_kg_s,
                           -e.admitted_mass_per_kg) ||
                    !close(e.number_before_per_kg, event.duration_s * event.admitted_number_rate_per_kg_s,
                           e.number_after_per_kg - e.number_before_per_kg))
                {
                    error = "rain evaporation endpoint source, budget or journal differs";
                    return false;
                }
                state.reflectivity *= e.mass_after_per_kg / e.mass_before_per_kg;
                state.mass = e.mass_after_per_kg;
                state.number = e.number_after_per_kg;
            }
            else
            {
                const double rate = state.reflectivity *
                    (2.0 * event.admitted_mass_rate_per_kg_s / state.mass -
                     event.admitted_number_rate_per_kg_s / state.number);
                state.mass += event.duration_s *
                    event.admitted_mass_rate_per_kg_s;
                state.number += event.duration_s *
                    event.admitted_number_rate_per_kg_s;
                state.reflectivity += event.duration_s * rate;
            }
            if (!valid_state(state))
            {
                if (failure)
                {
                    *failure = DynamicWarmStepFailure::IntervalRejected;
                }
                error = "rain evaporation endpoint is unrealizable";
                return false;
            }
        }
        ++index;
    }
    if (publish_rounded &&
        !finalize_rain_phase_endpoint(state.mass, state.number, state.reflectivity, result, error))
    {
        return false;
    }
    if (exact)
    {
        *exact = {state.mass, state.number, state.reflectivity};
    }
    if (failure)
    {
        *failure = DynamicWarmStepFailure::None;
    }
    return true;
}

bool apply_admitted_rain_phases(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    RainTripleMomentState& result,
    std::string& error)
{
    return apply_admitted_rain_phases_impl(initial, batch, result, nullptr, nullptr, error);
}

bool apply_admitted_rain_phases_exact(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    RainTripleMomentState& rounded,
    ExactRainTripleMomentState& exact,
    std::string& error)
{
    return apply_admitted_rain_phases_impl(initial, batch, rounded, &exact, nullptr, error);
}

bool apply_admitted_rain_phases_with_evaporation_endpoint(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    const AdmittedRainEvaporationEndpoint& endpoint,
    RainTripleMomentState& rounded,
    ExactRainTripleMomentState& exact,
    std::string& error)
{
    return apply_admitted_rain_phases_impl(initial, batch, rounded, &exact, &endpoint, error);
}

bool apply_admitted_rain_phase_prefix_exact(
    const RainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    ExactRainTripleMomentState& result,
    std::string& error)
{
    RainTripleMomentState unused;
    ExactRainTripleMomentState candidate;
    if (!apply_admitted_rain_phases_impl(initial, batch, unused, &candidate, nullptr, error, false))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool apply_admitted_rain_phase_prefix_fp64(
    const ExactRainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    ExactRainTripleMomentState& result,
    std::string& error,
    DynamicWarmStepFailure* failure)
{
    RainTripleMomentState unused;
    ExactRainTripleMomentState candidate;
    if (!apply_admitted_rain_phases_impl(initial, batch, unused, &candidate,
            nullptr, error, false, failure))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool apply_admitted_rain_phases_with_evaporation_endpoint_fp64(
    const ExactRainTripleMomentState& initial,
    const AcceptedRainPhaseBatch& batch,
    const AdmittedRainEvaporationEndpoint& endpoint,
    ExactRainTripleMomentState& result,
    std::string& error,
    DynamicWarmStepFailure* failure)
{
    RainTripleMomentState unused;
    ExactRainTripleMomentState candidate;
    if (!apply_admitted_rain_phases_impl(initial, batch, unused, &candidate,
            &endpoint, error, false, failure))
    {
        return false;
    }
    result = candidate;
    return true;
}

} // namespace tmv::milbrandt_detail
