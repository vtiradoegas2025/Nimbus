#include "triple_warm_events.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr double kRainMassCoefficient = 523.598775598;
constexpr double kAutoconversionSeedG = 5.6;

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

bool representable(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0 &&
           value <= std::numeric_limits<float>::max() &&
           (value == 0.0 ||
            value >= std::numeric_limits<float>::denorm_min());
}

bool number_correction_kind(AcceptedRainEventKind kind) noexcept
{
    return kind == AcceptedRainEventKind::PreRainNumberRegularization ||
           kind == AcceptedRainEventKind::RainNumberClip ||
           kind == AcceptedRainEventKind::PostRainNumberRegularization;
}

bool rain_removal_kind(AcceptedRainEventKind kind) noexcept
{
    return kind == AcceptedRainEventKind::RainToCloud ||
           kind == AcceptedRainEventKind::RainReset;
}

bool matches_complete_removal(double amount, double rate,
                              double duration) noexcept
{
    const double residual = amount + duration * rate;
    const double scale = std::max(std::abs(amount),
                                  std::abs(duration * rate));
    return std::isfinite(residual) &&
           std::abs(residual) <=
               8.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool realizable(double mass, double number, double reflectivity) noexcept
{
    if (!representable(mass) || !representable(number) ||
        !representable(reflectivity))
    {
        return false;
    }
    if (mass == 0.0 || number == 0.0 || reflectivity == 0.0)
    {
        return mass == 0.0 && number == 0.0 && reflectivity == 0.0;
    }
    const double log_lower = 2.0 * std::log(mass) -
        std::log(number) - 2.0 * std::log(kRainMassCoefficient);
    return std::isfinite(log_lower) &&
        std::log(reflectivity) >= log_lower;
}

bool event_reflectivity_rate(double mass, double number,
                             double reflectivity,
                             const AcceptedRainEvent& event,
                             double& rate,
                             std::string& error)
{
    if (!positive_finite(event.duration_s) ||
        !std::isfinite(event.admitted_mass_rate_per_kg_s) ||
        !std::isfinite(event.admitted_number_rate_per_kg_s))
    {
        error = "admitted rain event rate or duration is invalid";
        return false;
    }
    double candidate_rate = 0.0;
    if (event.kind == AcceptedRainEventKind::Autoconversion)
    {
        if (!positive_finite(event.admitted_mass_rate_per_kg_s) ||
            !positive_finite(event.admitted_number_rate_per_kg_s) ||
            !positive_finite(event.source_number_rate_per_m3_s) ||
            !positive_finite(event.source_dry_density_kg_m3) ||
            event.admitted_number_rate_per_kg_s !=
                event.source_number_rate_per_m3_s /
                    event.source_dry_density_kg_m3)
        {
            error = "admitted rain autoconversion provenance is invalid";
            return false;
        }
        candidate_rate =
            (kAutoconversionSeedG /
             (kRainMassCoefficient * kRainMassCoefficient)) *
            event.admitted_mass_rate_per_kg_s *
            event.admitted_mass_rate_per_kg_s /
            event.admitted_number_rate_per_kg_s;
    }
    else if (number_correction_kind(event.kind))
    {
        if (!(mass > 0.0 && number > 0.0 && reflectivity > 0.0) ||
            event.admitted_mass_rate_per_kg_s != 0.0 ||
            !positive_finite(event.source_dry_density_kg_m3) ||
            !std::isfinite(event.source_number_rate_per_m3_s) ||
            event.admitted_number_rate_per_kg_s !=
                event.source_number_rate_per_m3_s /
                    event.source_dry_density_kg_m3 ||
            (event.kind == AcceptedRainEventKind::RainNumberClip &&
             event.admitted_number_rate_per_kg_s == 0.0) ||
            (event.kind != AcceptedRainEventKind::RainNumberClip &&
             !positive_finite(event.admitted_number_rate_per_kg_s)) ||
            event.source_cloud_mass_rate_per_kg_s != 0.0 ||
            event.source_cloud_number_rate_per_m3_s != 0.0)
        {
            error = "admitted rain number correction provenance is invalid";
            return false;
        }
        const double next_number = number +
            event.duration_s * event.admitted_number_rate_per_kg_s;
        if (!positive_finite(next_number))
        {
            error = "admitted rain number correction exhausts rain number";
            return false;
        }
        // This is an explicit fixed-shape numerical policy, not a B3 process rate.
        const double next_reflectivity = reflectivity * number / next_number;
        if (!realizable(mass, next_number, next_reflectivity))
        {
            error = "admitted rain number correction endpoint is unrealizable";
            return false;
        }
        candidate_rate = (next_reflectivity - reflectivity) /
            event.duration_s;
    }
    else if (rain_removal_kind(event.kind))
    {
        if (!(mass > 0.0 && number > 0.0 && reflectivity > 0.0) ||
            !positive_finite(event.source_dry_density_kg_m3) ||
            !(event.admitted_mass_rate_per_kg_s < 0.0) ||
            !(event.admitted_number_rate_per_kg_s < 0.0) ||
            !std::isfinite(event.source_number_rate_per_m3_s) ||
            event.admitted_number_rate_per_kg_s !=
                event.source_number_rate_per_m3_s /
                    event.source_dry_density_kg_m3 ||
            !matches_complete_removal(
                mass, event.admitted_mass_rate_per_kg_s, event.duration_s) ||
            !matches_complete_removal(
                number, event.admitted_number_rate_per_kg_s, event.duration_s) ||
            (event.kind == AcceptedRainEventKind::RainToCloud &&
             (event.source_cloud_mass_rate_per_kg_s !=
                  -event.admitted_mass_rate_per_kg_s ||
              event.source_cloud_number_rate_per_m3_s !=
                  -event.source_number_rate_per_m3_s)) ||
            (event.kind == AcceptedRainEventKind::RainReset &&
             (event.source_cloud_mass_rate_per_kg_s != 0.0 ||
              event.source_cloud_number_rate_per_m3_s != 0.0)))
        {
            error = "admitted rain removal provenance is invalid";
            return false;
        }
        candidate_rate = -reflectivity / event.duration_s;
    }
    else
    {
        if (!(mass > 0.0 && number > 0.0 && reflectivity > 0.0) ||
            (event.kind == AcceptedRainEventKind::Evaporation &&
             (event.admitted_mass_rate_per_kg_s >= 0.0 ||
              event.admitted_number_rate_per_kg_s > 0.0)) ||
            (event.kind == AcceptedRainEventKind::SelfCollection &&
             (event.admitted_mass_rate_per_kg_s != 0.0 ||
              event.admitted_number_rate_per_kg_s >= 0.0)) ||
            (event.kind == AcceptedRainEventKind::Breakup &&
             (event.admitted_mass_rate_per_kg_s != 0.0 ||
              event.admitted_number_rate_per_kg_s <= 0.0)) ||
            (event.kind == AcceptedRainEventKind::Accretion &&
             (event.admitted_mass_rate_per_kg_s <= 0.0 ||
              event.admitted_number_rate_per_kg_s != 0.0)) ||
            (event.kind != AcceptedRainEventKind::Evaporation &&
             event.kind != AcceptedRainEventKind::SelfCollection &&
             event.kind != AcceptedRainEventKind::Accretion &&
             event.kind != AcceptedRainEventKind::Breakup))
        {
            error = "admitted same-category rain event provenance is invalid";
            return false;
        }
        candidate_rate = reflectivity *
            (2.0 * event.admitted_mass_rate_per_kg_s / mass -
             event.admitted_number_rate_per_kg_s / number);
    }
    if (!std::isfinite(candidate_rate))
    {
        error = "admitted rain reflectivity rate overflowed";
        return false;
    }
    rate = candidate_rate;
    error.clear();
    return true;
}

} // namespace

bool make_admitted_autoconversion_event(
    double admitted_mass_rate_per_kg_s,
    double admitted_number_rate_per_m3_s,
    double dry_density_kg_m3,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error)
{
    if (!positive_finite(admitted_mass_rate_per_kg_s) ||
        !positive_finite(admitted_number_rate_per_m3_s) ||
        !positive_finite(dry_density_kg_m3) ||
        !positive_finite(duration_s))
    {
        error = "admitted rain autoconversion event is invalid";
        return false;
    }
    const double number_rate_per_kg_s =
        admitted_number_rate_per_m3_s / dry_density_kg_m3;
    if (!positive_finite(number_rate_per_kg_s))
    {
        error = "admitted rain autoconversion number conversion is invalid";
        return false;
    }
    event = {AcceptedRainEventKind::Autoconversion,
             admitted_mass_rate_per_kg_s, number_rate_per_kg_s,
             duration_s, admitted_number_rate_per_m3_s,
             dry_density_kg_m3};
    error.clear();
    return true;
}

bool make_admitted_same_category_event(
    AcceptedRainEventKind kind,
    double admitted_mass_rate_per_kg_s,
    double admitted_number_rate_per_kg_s,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error)
{
    if (!std::isfinite(admitted_mass_rate_per_kg_s) ||
        !std::isfinite(admitted_number_rate_per_kg_s) ||
        !positive_finite(duration_s) ||
        (kind == AcceptedRainEventKind::Evaporation &&
         (admitted_mass_rate_per_kg_s >= 0.0 ||
          admitted_number_rate_per_kg_s > 0.0)) ||
        (kind == AcceptedRainEventKind::SelfCollection &&
         (admitted_mass_rate_per_kg_s != 0.0 ||
          admitted_number_rate_per_kg_s >= 0.0)) ||
        (kind == AcceptedRainEventKind::Breakup &&
         (admitted_mass_rate_per_kg_s != 0.0 ||
          admitted_number_rate_per_kg_s <= 0.0)) ||
        (kind == AcceptedRainEventKind::Accretion &&
         (admitted_mass_rate_per_kg_s <= 0.0 ||
          admitted_number_rate_per_kg_s != 0.0)) ||
        (kind != AcceptedRainEventKind::Evaporation &&
         kind != AcceptedRainEventKind::SelfCollection &&
         kind != AcceptedRainEventKind::Accretion &&
         kind != AcceptedRainEventKind::Breakup))
    {
        error = "admitted same-category rain event is invalid";
        return false;
    }
    event = {kind, admitted_mass_rate_per_kg_s,
             admitted_number_rate_per_kg_s, duration_s, 0.0, 0.0};
    error.clear();
    return true;
}

bool make_admitted_rain_number_correction_event(
    AcceptedRainEventKind kind,
    double admitted_rain_number_rate_per_m3_s,
    double dry_density_kg_m3,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error)
{
    if (!number_correction_kind(kind) ||
        !std::isfinite(admitted_rain_number_rate_per_m3_s) ||
        !positive_finite(dry_density_kg_m3) ||
        !positive_finite(duration_s) ||
        (kind == AcceptedRainEventKind::RainNumberClip &&
         admitted_rain_number_rate_per_m3_s == 0.0) ||
        (kind != AcceptedRainEventKind::RainNumberClip &&
         !positive_finite(admitted_rain_number_rate_per_m3_s)))
    {
        error = "admitted rain number correction event is invalid";
        return false;
    }
    const double number_rate_per_kg_s =
        admitted_rain_number_rate_per_m3_s / dry_density_kg_m3;
    if (!std::isfinite(number_rate_per_kg_s) ||
        (kind == AcceptedRainEventKind::RainNumberClip &&
         number_rate_per_kg_s == 0.0) ||
        (kind != AcceptedRainEventKind::RainNumberClip &&
         !positive_finite(number_rate_per_kg_s)))
    {
        error = "admitted rain number correction conversion is invalid";
        return false;
    }
    event = {kind, 0.0, number_rate_per_kg_s, duration_s,
             admitted_rain_number_rate_per_m3_s, dry_density_kg_m3};
    error.clear();
    return true;
}

bool make_admitted_rain_removal_event(
    AcceptedRainEventKind kind,
    double admitted_cloud_mass_rate_per_kg_s,
    double admitted_rain_mass_rate_per_kg_s,
    double admitted_cloud_number_rate_per_m3_s,
    double admitted_rain_number_rate_per_m3_s,
    double dry_density_kg_m3,
    double duration_s,
    AcceptedRainEvent& event,
    std::string& error)
{
    if (!rain_removal_kind(kind) ||
        !std::isfinite(admitted_cloud_mass_rate_per_kg_s) ||
        !std::isfinite(admitted_rain_mass_rate_per_kg_s) ||
        !std::isfinite(admitted_cloud_number_rate_per_m3_s) ||
        !std::isfinite(admitted_rain_number_rate_per_m3_s) ||
        !(admitted_rain_mass_rate_per_kg_s < 0.0) ||
        !(admitted_rain_number_rate_per_m3_s < 0.0) ||
        !positive_finite(dry_density_kg_m3) ||
        !positive_finite(duration_s) ||
        (kind == AcceptedRainEventKind::RainToCloud &&
         (admitted_cloud_mass_rate_per_kg_s !=
              -admitted_rain_mass_rate_per_kg_s ||
          admitted_cloud_number_rate_per_m3_s !=
              -admitted_rain_number_rate_per_m3_s)) ||
        (kind == AcceptedRainEventKind::RainReset &&
         (admitted_cloud_mass_rate_per_kg_s != 0.0 ||
          admitted_cloud_number_rate_per_m3_s != 0.0)))
    {
        error = "admitted rain removal event is invalid";
        return false;
    }
    const double number_rate_per_kg_s =
        admitted_rain_number_rate_per_m3_s / dry_density_kg_m3;
    if (!(number_rate_per_kg_s < 0.0))
    {
        error = "admitted rain removal number conversion is invalid";
        return false;
    }
    event = {kind, admitted_rain_mass_rate_per_kg_s,
             number_rate_per_kg_s, duration_s,
             admitted_rain_number_rate_per_m3_s, dry_density_kg_m3,
             admitted_cloud_mass_rate_per_kg_s,
             admitted_cloud_number_rate_per_m3_s};
    error.clear();
    return true;
}

bool admitted_rain_reflectivity_rate(
    const RainTripleMomentState& state,
    const AcceptedRainEvent& event,
    double& rate_per_kg_s,
    std::string& error)
{
    if (!realizable(state.mass_per_kg, state.number_per_kg,
                    state.reflectivity_per_kg))
    {
        error = "initial rain q/N/Z state is unrealizable";
        return false;
    }
    return event_reflectivity_rate(
        state.mass_per_kg, state.number_per_kg,
        state.reflectivity_per_kg, event, rate_per_kg_s, error);
}

bool apply_admitted_rain_events(
    const RainTripleMomentState& initial,
    const std::vector<AcceptedRainEvent>& events,
    RainTripleMomentState& result,
    std::string& error)
{
    if (events.size() > 64)
    {
        error = "admitted rain event sequence exceeds the 64-event resource budget";
        return false;
    }
    double mass = initial.mass_per_kg;
    double number = initial.number_per_kg;
    double reflectivity = initial.reflectivity_per_kg;
    if (!realizable(mass, number, reflectivity))
    {
        error = "initial rain q/N/Z state is unrealizable";
        return false;
    }
    const double duration_s = events.empty() ? 0.0 : events.front().duration_s;
    for (const AcceptedRainEvent& event : events)
    {
        if (event.duration_s != duration_s)
        {
            error = "admitted rain event durations differ";
            return false;
        }
        double reflectivity_rate = 0.0;
        if (!event_reflectivity_rate(mass, number, reflectivity,
                                     event, reflectivity_rate, error))
        {
            return false;
        }
        const bool removal = rain_removal_kind(event.kind);
        const double next_mass = removal ? 0.0 : mass +
            event.duration_s * event.admitted_mass_rate_per_kg_s;
        const double next_number = removal ? 0.0 : number +
            event.duration_s * event.admitted_number_rate_per_kg_s;
        const double next_reflectivity = removal ? 0.0 :
            number_correction_kind(event.kind) ?
                reflectivity * number / next_number :
                reflectivity + event.duration_s * reflectivity_rate;
        if (!realizable(next_mass, next_number, next_reflectivity))
        {
            error = "admitted rain event yields an unrealizable q/N/Z candidate";
            return false;
        }
        mass = next_mass;
        number = next_number;
        reflectivity = next_reflectivity;
    }
    const RainTripleMomentState candidate{
        static_cast<float>(mass), static_cast<float>(number),
        static_cast<float>(reflectivity)};
    if (!realizable(candidate.mass_per_kg, candidate.number_per_kg,
                    candidate.reflectivity_per_kg))
    {
        error = "admitted rain event rounded q/N/Z candidate is unrealizable";
        return false;
    }
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
