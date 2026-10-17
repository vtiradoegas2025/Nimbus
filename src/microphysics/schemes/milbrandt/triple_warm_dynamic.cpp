#include "triple_warm_dynamic.hpp"

#include <stdexcept>

namespace tmv::milbrandt_detail
{
namespace
{

bool empty_population(const ExactRainTripleMomentState& state) noexcept
{
    return state.mass_per_kg == 0.0 && state.number_per_kg == 0.0 &&
        state.reflectivity_per_kg == 0.0;
}

void diagnose_context(const ExactRainTripleMomentState& state,
                      TripleWarmRainSpectrum& spectrum,
                      bool& present)
{
    present = !empty_population(state);
    if (present)
    {
        std::string error;
        if (!diagnose_triple_warm_rain_spectrum(state, spectrum, error))
        {
            throw std::invalid_argument(error);
        }
    }
}

template <typename Initial>
class DynamicWarmRainSpectrumPolicy
{
public:
    static constexpr bool dynamic = true;
    static constexpr bool typed_resource = std::is_same_v<Initial, ExactRainTripleMomentState>;

    DynamicWarmRainSpectrumPolicy(const Initial& initial,
                                 double dt, double density,
                                 const DynamicWarmFP64Policies& policies = {})
        : initial_(initial), dt_(dt), density_(density), policies_(policies)
    {
    }

    bool pre_number_regularization() const noexcept
    {
        return policies_.pre_number_regularization;
    }

    bool post_number_regularization() const noexcept
    {
        return policies_.post_number_regularization;
    }

    bool reclassification() const noexcept
    {
        return policies_.reclassification;
    }

    void before_physical(
        const std::array<WarmRainAcceptedEvent, 16>& events,
        std::size_t count, double mass, double number)
    {
        bind_prefix(events, count, mass, number,
                    pre_physical_rain, pre_physical_spectrum,
                    pre_physical_spectrum_present);
        pre_physical_event_count = count;
        current_spectrum_ = pre_physical_spectrum_present
            ? &pre_physical_spectrum : nullptr;
    }

    void before_evaporation(
        const std::array<WarmRainAcceptedEvent, 16>& events,
        std::size_t count, double mass, double number)
    {
        bind_prefix(events, count, mass, number,
                    pre_evaporation_rain, pre_evaporation_spectrum,
                    pre_evaporation_spectrum_present);
        pre_evaporation_event_count = count;
        current_spectrum_ = pre_evaporation_spectrum_present
            ? &pre_evaporation_spectrum : nullptr;
    }

    double diameter_moment_1_m() const
    {
        return spectrum().diameter_moment_1_m();
    }

    double diameter_moment_3_m3() const
    {
        return spectrum().diameter_moment_3_m3();
    }

    double diameter_moment_6_m6() const
    {
        return spectrum().diameter_moment_6_m6();
    }

    double ventilation_diameter_moment() const
    {
        return spectrum().ventilation_diameter_moment();
    }

    double alpha_zero_equivalent_inverse_slope_m() const
    {
        return spectrum().alpha_zero_equivalent_inverse_slope_m();
    }

    ExactRainTripleMomentState pre_physical_rain{};
    ExactRainTripleMomentState pre_evaporation_rain{};
    TripleWarmRainSpectrum pre_physical_spectrum{};
    TripleWarmRainSpectrum pre_evaporation_spectrum{};
    std::size_t pre_physical_event_count = 0;
    std::size_t pre_evaporation_event_count = 0;
    bool pre_physical_spectrum_present = false;
    bool pre_evaporation_spectrum_present = false;

private:
    void bind_prefix(const std::array<WarmRainAcceptedEvent, 16>& events,
                     std::size_t count, double mass, double number,
                     ExactRainTripleMomentState& rain,
                     TripleWarmRainSpectrum& spectrum,
                     bool& present)
    {
        WarmRainAcceptedEventBuffer prefix;
        prefix.events = events;
        prefix.size = count;
        std::string error;
        bool accepted = false;
        if constexpr (std::is_same_v<Initial, ExactRainTripleMomentState>)
        {
            DynamicWarmStepFailure failure = DynamicWarmStepFailure::None;
            accepted = apply_dynamic_rain_prefix_fp64(initial_, prefix, dt_, density_,
                mass, number, rain, error, &failure);
            if (!accepted && failure == DynamicWarmStepFailure::IntervalRejected)
            {
                throw DynamicWarmIntervalError(error);
            }
        }
        else
        {
            accepted = apply_dynamic_rain_prefix(initial_, prefix, dt_, density_,
                mass, number, rain, error);
        }
        if (!accepted)
        {
            throw std::invalid_argument(error);
        }
        diagnose_context(rain, spectrum, present);
    }

    const TripleWarmRainSpectrum& spectrum() const
    {
        if (!current_spectrum_)
        {
            throw std::logic_error("dynamic warm rate has no represented phase spectrum");
        }
        return *current_spectrum_;
    }

    Initial initial_{};
    double dt_ = 0.0;
    double density_ = 0.0;
    DynamicWarmFP64Policies policies_{};
    const TripleWarmRainSpectrum* current_spectrum_ = nullptr;
};

} // namespace

void triple_warm_rain_cell_dynamic(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const RainTripleMomentState& initial_rain,
    TripleWarmRainDynamicResult& result,
    std::size_t observer_capacity)
{
    validate_triple_warm_rain_source(parameters, cell, dt, incoming_rates);
    if (initial_rain.mass_per_kg != cell.rain ||
        initial_rain.number_per_kg != cell.rain_number)
    {
        throw std::invalid_argument("dynamic warm original rain q/N differs from its source");
    }
    TripleWarmRainDynamicResult candidate;
    candidate.original_rain_ = initial_rain;
    diagnose_context({initial_rain.mass_per_kg, initial_rain.number_per_kg,
                      initial_rain.reflectivity_per_kg},
                     candidate.original_spectrum_,
                     candidate.original_spectrum_present_);
    auto& producer = candidate.producer_;
    producer.source_ = cell;
    producer.parameters_ = parameters;
    producer.incoming_ = incoming_rates;
    producer.rates_ = incoming_rates;
    producer.events_.capacity = observer_capacity;
    producer.uses_dynamic_spectrum_ = true;
    DynamicWarmRainSpectrumPolicy policy(initial_rain, dt, cell.density);
    warm_rain_cell_impl<true>(parameters, cell, dt, producer.rates_,
                             &producer.events_, &producer.endpoints_, &policy);
    producer.duration_s_ = dt;
    candidate.pre_physical_rain_ = policy.pre_physical_rain;
    candidate.pre_evaporation_rain_ = policy.pre_evaporation_rain;
    candidate.pre_physical_spectrum_ = policy.pre_physical_spectrum;
    candidate.pre_evaporation_spectrum_ = policy.pre_evaporation_spectrum;
    candidate.pre_physical_event_count_ = policy.pre_physical_event_count;
    candidate.pre_evaporation_event_count_ = policy.pre_evaporation_event_count;
    candidate.pre_physical_spectrum_present_ = policy.pre_physical_spectrum_present;
    candidate.pre_evaporation_spectrum_present_ = policy.pre_evaporation_spectrum_present;
    result = candidate;
}

void triple_warm_rain_cell_dynamic_fp64(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const ExactRainTripleMomentState& initial_rain,
    TripleWarmRainDynamicFP64Result& result,
    std::size_t observer_capacity,
    const DynamicWarmFP64Policies& policies)
{
    validate_triple_warm_rain_source_impl(parameters, cell, dt, incoming_rates);
    if (cell.temperature <= microphysics_constants::T0 || cell.temperature >= 400.0 ||
        cell.pressure < 100.0 || cell.pressure > 2.0e5 ||
        (cell.cloud > 0.0 && cell.cloud_number <= 0.0) ||
        initial_rain.mass_per_kg != cell.rain ||
        initial_rain.number_per_kg != cell.rain_number)
    {
        throw std::invalid_argument("dynamic FP64 warm source or original rain q/N is invalid");
    }
    if (observer_capacity > 16)
    {
        throw WarmRainObserverCapacityError();
    }
    TripleWarmRainDynamicFP64Result candidate;
    candidate.policies_ = policies;
    candidate.original_rain_ = initial_rain;
    diagnose_context(initial_rain, candidate.original_spectrum_,
        candidate.original_spectrum_present_);
    if (candidate.original_spectrum_present_ && candidate.original_spectrum_.alpha() > 200.0)
    {
        throw std::invalid_argument("dynamic FP64 warm original rain shape exceeds 200");
    }
    auto& producer = candidate.producer_;
    producer.source_ = cell;
    producer.parameters_ = parameters;
    producer.incoming_ = incoming_rates;
    producer.rates_ = incoming_rates;
    producer.events_.capacity = observer_capacity;
    producer.uses_dynamic_spectrum_ = true;
    DynamicWarmRainSpectrumPolicy policy(initial_rain, dt, cell.density, policies);
    warm_rain_cell_impl<true>(parameters, cell, dt, producer.rates_,
        &producer.events_, &producer.endpoints_, &policy);
    producer.duration_s_ = dt;
    candidate.pre_physical_rain_ = policy.pre_physical_rain;
    candidate.pre_evaporation_rain_ = policy.pre_evaporation_rain;
    candidate.pre_physical_spectrum_ = policy.pre_physical_spectrum;
    candidate.pre_evaporation_spectrum_ = policy.pre_evaporation_spectrum;
    candidate.pre_physical_event_count_ = policy.pre_physical_event_count;
    candidate.pre_evaporation_event_count_ = policy.pre_evaporation_event_count;
    candidate.pre_physical_spectrum_present_ = policy.pre_physical_spectrum_present;
    candidate.pre_evaporation_spectrum_present_ = policy.pre_evaporation_spectrum_present;
    result = candidate;
}

} // namespace tmv::milbrandt_detail
