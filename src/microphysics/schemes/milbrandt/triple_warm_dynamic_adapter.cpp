#include "triple_warm_dynamic.hpp"
#include "triple_warm_adapter.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

template <typename Value, std::size_t Size>
bool same_values(const std::array<Value, Size>& first, const std::array<Value, Size>& second)
{
    for (std::size_t index = 0; index < Size; ++index)
    {
        if (std::memcmp(&first[index], &second[index], sizeof(Value)) != 0)
        {
            return false;
        }
    }
    return true;
}

template <typename Rain>
bool same_rain(const Rain& first, const Rain& second)
{
    using Value = decltype(first.mass_per_kg);
    return same_values(std::array<Value, 3>{first.mass_per_kg, first.number_per_kg, first.reflectivity_per_kg},
        std::array<Value, 3>{second.mass_per_kg, second.number_per_kg, second.reflectivity_per_kg});
}

bool same_spectrum(const TripleWarmRainSpectrum* first, const TripleWarmRainSpectrum* second)
{
    if (!first || !second)
    {
        return first == nullptr && second == nullptr;
    }
    const auto values = [](const TripleWarmRainSpectrum& s)
    {
        return std::array<double, 8>{s.alpha(), s.lambda_per_m(), s.mean_mass_diameter_m(),
            s.alpha_zero_equivalent_inverse_slope_m(), s.diameter_moment_1_m(),
            s.diameter_moment_3_m3(), s.diameter_moment_6_m6(), s.ventilation_diameter_moment()};
    };
    return same_rain(first->source(), second->source()) && same_values(values(*first), values(*second));
}

std::array<double, 16> endpoint_values(const WarmRainDoubleEndpoints& e)
{
    return {e.vapor_per_kg, e.cloud_per_kg, e.rain_per_kg, e.cloud_number_per_kg, e.rain_number_per_kg,
        e.temperature_increment_k, e.cloud_donor_budget_per_kg, e.evaporation.mass_before_per_kg,
        e.evaporation.number_before_per_m3, e.evaporation.admitted_mass_per_kg,
        e.evaporation.mass_after_per_kg, e.evaporation.number_after_per_m3,
        e.reclassification.rain_mass_before_per_kg, e.reclassification.rain_number_before_per_m3,
        e.reclassification.cloud_mass_before_per_kg, e.reclassification.cloud_number_before_per_m3};
}

template <typename Capture>
bool same_capture(const Capture& first, const Capture& second)
{
    if (!same_rain(first.original_rain(), second.original_rain()) ||
        !same_rain(first.pre_physical_rain(), second.pre_physical_rain()) ||
        !same_rain(first.pre_evaporation_rain(), second.pre_evaporation_rain()) ||
        !same_spectrum(first.original_spectrum(), second.original_spectrum()) ||
        !same_spectrum(first.pre_physical_spectrum(), second.pre_physical_spectrum()) ||
        !same_spectrum(first.pre_evaporation_spectrum(), second.pre_evaporation_spectrum()) ||
        first.pre_physical_event_count() != second.pre_physical_event_count() ||
        first.pre_evaporation_event_count() != second.pre_evaporation_event_count())
    {
        return false;
    }
    const auto& a = first.producer();
    const auto& b = second.producer();
    const auto& ae = a.endpoints();
    const auto& be = b.endpoints();
    const auto rate_values = [](const WarmRainTendencies& r)
    {
        return std::array<float, 6>{r.cloud, r.rain, r.vapor, r.temperature, r.cloud_number, r.rain_number};
    };
    if (!a.uses_dynamic_spectrum() || !b.uses_dynamic_spectrum() ||
        !same_values(rate_values(a.rates()), rate_values(b.rates())) ||
        !same_values(endpoint_values(ae), endpoint_values(be)) ||
        ae.cloud_donor_exhausted != be.cloud_donor_exhausted ||
        ae.evaporation.present != be.evaporation.present ||
        ae.evaporation.donor_exhausted != be.evaporation.donor_exhausted ||
        ae.reclassification.present != be.reclassification.present ||
        a.events().size != b.events().size || a.events().capacity != b.events().capacity)
    {
        return false;
    }
    for (std::size_t index = 0; index < a.events().size; ++index)
    {
        const auto& x = a.events().events[index];
        const auto& y = b.events().events[index];
        const auto values = [](const WarmRainAcceptedEvent& e)
        {
            return std::array<double, 10>{e.cloud_mass_rate_per_kg_s, e.rain_mass_rate_per_kg_s,
                e.vapor_mass_rate_per_kg_s, e.cloud_number_rate_per_m3_s, e.rain_number_rate_per_m3_s,
                e.temperature_rate_k_s, e.dry_density_kg_m3, e.cloud_donor_scale, e.seed_diameter_m,
                static_cast<double>(e.seed_mass_number_coherent)};
        };
        if (x.kind != y.kind || !same_values(values(x), values(y)))
        {
            return false;
        }
    }
    return true;
}

} // namespace

template <typename Initial>
bool apply_dynamic_rain_prefix_impl(
    const Initial& initial,
    const WarmRainAcceptedEventBuffer& events,
    double dt,
    double density,
    double owned_mass_per_kg,
    double owned_number_per_kg,
    ExactRainTripleMomentState& result,
    std::string& error,
    DynamicWarmStepFailure* failure = nullptr)
{
    if (failure)
    {
        *failure = DynamicWarmStepFailure::InvalidSource;
    }
    if (!std::isfinite(dt) || dt <= 0.0 || !std::isfinite(density) || density <= 0.0 ||
        !std::isfinite(owned_mass_per_kg) || owned_mass_per_kg < 0.0 ||
        !std::isfinite(owned_number_per_kg) || owned_number_per_kg < 0.0 ||
        events.capacity > events.events.size() || events.size > events.capacity)
    {
        error = "dynamic rain phase prefix source or resource bound is invalid";
        return false;
    }
    if constexpr (std::is_same_v<Initial, ExactRainTripleMomentState>)
    {
        if (initial.mass_per_kg != 0.0 || initial.number_per_kg != 0.0 ||
            initial.reflectivity_per_kg != 0.0)
        {
            TripleWarmRainSpectrum spectrum;
            if (!diagnose_triple_warm_rain_spectrum(initial, spectrum, error))
            {
                return false;
            }
            if (spectrum.alpha() > 200.0)
            {
                error = "dynamic FP64 original rain phase shape exceeds 200";
                return false;
            }
        }
    }
    AcceptedRainPhaseBatch batch;
    batch.duration_s = dt;
    for (std::size_t index = 0; index < events.size; ++index)
    {
        if (events.events[index].kind == WarmRainAcceptedEventKind::Evaporation ||
            !map_dynamic_triple_produced_rain_event(events.events[index], dt, density, batch, error))
        {
            if (error.empty())
            {
                error = "dynamic rain phase prefix includes evaporation";
            }
            return false;
        }
    }
    ExactRainTripleMomentState candidate;
    bool applied = false;
    if constexpr (std::is_same_v<Initial, ExactRainTripleMomentState>)
    {
        applied = apply_admitted_rain_phase_prefix_fp64(initial, batch, candidate, error, failure);
    }
    else
    {
        applied = apply_admitted_rain_phase_prefix_exact(initial, batch, candidate, error);
    }
    if (!applied)
    {
        return false;
    }
    const auto close = [](double initial_value, double expected, double actual)
    {
        const double scale = std::max({std::abs(initial_value), std::abs(expected), std::abs(actual), 1.0e-30});
        return std::isfinite(expected) && std::isfinite(actual) &&
            std::abs(expected - actual) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
    };
    if (!close(initial.mass_per_kg, candidate.mass_per_kg, owned_mass_per_kg) ||
        !close(initial.number_per_kg, candidate.number_per_kg, owned_number_per_kg))
    {
        if (failure)
        {
            *failure = DynamicWarmStepFailure::InvalidSource;
        }
        error = "dynamic rain phase prefix and admitted q/N differ";
        return false;
    }
    candidate.mass_per_kg = owned_mass_per_kg;
    candidate.number_per_kg = owned_number_per_kg;
    if (candidate.mass_per_kg == 0.0 && candidate.number_per_kg == 0.0)
    {
        if (candidate.reflectivity_per_kg != 0.0)
        {
            if (failure)
            {
                *failure = DynamicWarmStepFailure::InvalidSource;
            }
            error = "dynamic rain empty phase retains reflectivity";
            return false;
        }
    }
    else
    {
        TripleWarmRainSpectrum spectrum;
        if (!diagnose_triple_warm_rain_spectrum(candidate, spectrum, error))
        {
            if (failure)
            {
                *failure = DynamicWarmStepFailure::IntervalRejected;
            }
            return false;
        }
        if constexpr (std::is_same_v<Initial, ExactRainTripleMomentState>)
        {
            if (spectrum.alpha() > 200.0)
            {
                error = "dynamic FP64 rain phase shape exceeds 200";
                if (failure)
                {
                    *failure = DynamicWarmStepFailure::IntervalRejected;
                }
                return false;
            }
        }
    }
    result = candidate;
    error.clear();
    if (failure)
    {
        *failure = DynamicWarmStepFailure::None;
    }
    return true;
}

bool apply_dynamic_rain_prefix(
    const RainTripleMomentState& initial,
    const WarmRainAcceptedEventBuffer& events,
    double dt,
    double density,
    double owned_mass_per_kg,
    double owned_number_per_kg,
    ExactRainTripleMomentState& result,
    std::string& error)
{
    return apply_dynamic_rain_prefix_impl(initial, events, dt, density,
        owned_mass_per_kg, owned_number_per_kg, result, error);
}

bool apply_dynamic_rain_prefix_fp64(
    const ExactRainTripleMomentState& initial,
    const WarmRainAcceptedEventBuffer& events,
    double dt,
    double density,
    double owned_mass_per_kg,
    double owned_number_per_kg,
    ExactRainTripleMomentState& result,
    std::string& error,
    DynamicWarmStepFailure* failure)
{
    return apply_dynamic_rain_prefix_impl(initial, events, dt, density,
        owned_mass_per_kg, owned_number_per_kg, result, error, failure);
}

bool validate_dynamic_warm_producer(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const RainTripleMomentState& initial, const TripleWarmRainDynamicResult& producer,
    std::string& error)
{
    if (!same_rain(initial, producer.original_rain()) || !producer.producer().uses_dynamic_spectrum())
    {
        error = "dynamic warm original rain or rate policy provenance differs";
        return false;
    }
    TripleWarmRainDynamicResult expected;
    try
    {
        triple_warm_rain_cell_dynamic(parameters, cell, dt, {}, initial,
            expected, producer.producer().events().capacity);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
    if (!same_capture(producer, expected))
    {
        error = "dynamic warm spectrum, phase or admitted rate provenance differs";
        return false;
    }
    error.clear();
    return true;
}

bool validate_dynamic_warm_fp64_producer(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const ExactRainTripleMomentState& initial, const TripleWarmRainDynamicFP64Result& producer,
    const DynamicWarmFP64Policies& policies, std::string& error)
{
    if (!same_rain(initial, producer.original_rain()) || !producer.producer().uses_dynamic_spectrum() ||
        producer.policies().pre_number_regularization != policies.pre_number_regularization ||
        producer.policies().post_number_regularization != policies.post_number_regularization ||
        producer.policies().reclassification != policies.reclassification)
    {
        error = "dynamic FP64 warm original rain or rate policy provenance differs";
        return false;
    }
    TripleWarmRainDynamicFP64Result expected;
    try
    {
        triple_warm_rain_cell_dynamic_fp64(parameters, cell, dt, {}, initial,
            expected, producer.producer().events().capacity, policies);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
    if (!same_capture(producer, expected))
    {
        error = "dynamic FP64 warm spectrum, phase or admitted rate provenance differs";
        return false;
    }
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
