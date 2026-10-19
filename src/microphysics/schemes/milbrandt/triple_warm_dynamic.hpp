#pragma once

#include "triple_warm_spectrum.hpp"
#include "warm_rain.hpp"

namespace tmv::milbrandt_detail
{

struct DynamicWarmFP64Policies
{
    bool pre_number_regularization = true;
    bool post_number_regularization = true;
    bool reclassification = true;
};

class TripleWarmRainDynamicResult
{
public:
    const TripleWarmRainDoubleSourceResult& producer() const noexcept
    {
        return producer_;
    }

    const RainTripleMomentState& original_rain() const noexcept
    {
        return original_rain_;
    }

    const ExactRainTripleMomentState& pre_physical_rain() const noexcept
    {
        return pre_physical_rain_;
    }

    const ExactRainTripleMomentState& pre_evaporation_rain() const noexcept
    {
        return pre_evaporation_rain_;
    }

    const TripleWarmRainSpectrum* original_spectrum() const noexcept
    {
        return original_spectrum_present_ ? &original_spectrum_ : nullptr;
    }

    const TripleWarmRainSpectrum* pre_physical_spectrum() const noexcept
    {
        return pre_physical_spectrum_present_ ? &pre_physical_spectrum_ : nullptr;
    }

    const TripleWarmRainSpectrum* pre_evaporation_spectrum() const noexcept
    {
        return pre_evaporation_spectrum_present_ ? &pre_evaporation_spectrum_ : nullptr;
    }

    std::size_t pre_physical_event_count() const noexcept
    {
        return pre_physical_event_count_;
    }

    std::size_t pre_evaporation_event_count() const noexcept
    {
        return pre_evaporation_event_count_;
    }

private:
    friend void triple_warm_rain_cell_dynamic(
        const WarmRainParameters&, const DoubleWarmRainCell&, double,
        const WarmRainTendencies&, const RainTripleMomentState&,
        TripleWarmRainDynamicResult&, std::size_t);

    TripleWarmRainDoubleSourceResult producer_{};
    RainTripleMomentState original_rain_{};
    ExactRainTripleMomentState pre_physical_rain_{};
    ExactRainTripleMomentState pre_evaporation_rain_{};
    TripleWarmRainSpectrum original_spectrum_{};
    TripleWarmRainSpectrum pre_physical_spectrum_{};
    TripleWarmRainSpectrum pre_evaporation_spectrum_{};
    std::size_t pre_physical_event_count_ = 0;
    std::size_t pre_evaporation_event_count_ = 0;
    bool original_spectrum_present_ = false;
    bool pre_physical_spectrum_present_ = false;
    bool pre_evaporation_spectrum_present_ = false;
};

class TripleWarmRainDynamicFP64Result
{
public:
    const DynamicWarmFP64Policies& policies() const noexcept
    {
        return policies_;
    }

    const TripleWarmRainDoubleSourceResult& producer() const noexcept
    {
        return producer_;
    }

    const ExactRainTripleMomentState& original_rain() const noexcept
    {
        return original_rain_;
    }

    const ExactRainTripleMomentState& pre_physical_rain() const noexcept
    {
        return pre_physical_rain_;
    }

    const ExactRainTripleMomentState& pre_evaporation_rain() const noexcept
    {
        return pre_evaporation_rain_;
    }

    const TripleWarmRainSpectrum* original_spectrum() const noexcept
    {
        return original_spectrum_present_ ? &original_spectrum_ : nullptr;
    }

    const TripleWarmRainSpectrum* pre_physical_spectrum() const noexcept
    {
        return pre_physical_spectrum_present_ ? &pre_physical_spectrum_ : nullptr;
    }

    const TripleWarmRainSpectrum* pre_evaporation_spectrum() const noexcept
    {
        return pre_evaporation_spectrum_present_ ? &pre_evaporation_spectrum_ : nullptr;
    }

    std::size_t pre_physical_event_count() const noexcept
    {
        return pre_physical_event_count_;
    }

    std::size_t pre_evaporation_event_count() const noexcept
    {
        return pre_evaporation_event_count_;
    }

private:
    friend void triple_warm_rain_cell_dynamic_fp64(
        const WarmRainParameters&, const DoubleWarmRainCell&, double,
        const WarmRainTendencies&, const ExactRainTripleMomentState&,
        TripleWarmRainDynamicFP64Result&, std::size_t, const DynamicWarmFP64Policies&);

    DynamicWarmFP64Policies policies_{};
    TripleWarmRainDoubleSourceResult producer_{};
    ExactRainTripleMomentState original_rain_{};
    ExactRainTripleMomentState pre_physical_rain_{};
    ExactRainTripleMomentState pre_evaporation_rain_{};
    TripleWarmRainSpectrum original_spectrum_{};
    TripleWarmRainSpectrum pre_physical_spectrum_{};
    TripleWarmRainSpectrum pre_evaporation_spectrum_{};
    std::size_t pre_physical_event_count_ = 0;
    std::size_t pre_evaporation_event_count_ = 0;
    bool original_spectrum_present_ = false;
    bool pre_physical_spectrum_present_ = false;
    bool pre_evaporation_spectrum_present_ = false;
};

class DynamicWarmIntervalError : public std::invalid_argument
{
public:
    explicit DynamicWarmIntervalError(const std::string& error)
        : std::invalid_argument(error)
    {
    }
};

bool apply_dynamic_rain_prefix(
    const RainTripleMomentState& initial,
    const WarmRainAcceptedEventBuffer& events,
    double dt,
    double density,
    double owned_mass_per_kg,
    double owned_number_per_kg,
    ExactRainTripleMomentState& result,
    std::string& error);

bool apply_dynamic_rain_prefix_fp64(
    const ExactRainTripleMomentState& initial,
    const WarmRainAcceptedEventBuffer& events,
    double dt,
    double density,
    double owned_mass_per_kg,
    double owned_number_per_kg,
    ExactRainTripleMomentState& result,
    std::string& error,
    DynamicWarmStepFailure* failure = nullptr);

void triple_warm_rain_cell_dynamic_fp64(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const ExactRainTripleMomentState& initial_rain,
    TripleWarmRainDynamicFP64Result& result,
    std::size_t observer_capacity = 16,
    const DynamicWarmFP64Policies& policies = {});

void triple_warm_rain_cell_dynamic(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const WarmRainTendencies& incoming_rates,
    const RainTripleMomentState& initial_rain,
    TripleWarmRainDynamicResult& result,
    std::size_t observer_capacity = 16);

} // namespace tmv::milbrandt_detail
