#include "triple_warm_subcycle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace tmv::milbrandt_detail
{
namespace
{

struct ProcessState
{
    DoubleWarmRainCell cell{};
    ExactRainTripleMomentState rain{};
    double heat = 0.0;
    double heat_compensation = 0.0;
};

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

std::array<double, 8> cell_values(const DoubleWarmRainCell& s)
{
    return {s.temperature, s.pressure, s.density, s.vapor, s.cloud, s.rain, s.cloud_number, s.rain_number};
}

template <typename Rain>
auto rain_values(const Rain& s)
{
    return std::array<decltype(s.mass_per_kg), 3>{s.mass_per_kg, s.number_per_kg, s.reflectivity_per_kg};
}

std::array<double, 17> parameter_values(const WarmRainParameters& p)
{
    return {p.alpha_r, p.c_r, p.d_r, p.a_r, p.b_r, p.inv_mu_c, p.gc1, p.gc2, p.gc3,
        p.gc4, p.gr31, p.gr34, p.gr37, p.inv_cloud_mass_factor, p.inv_rain_mass_factor,
        p.gamma_ventilation, p.gamma_fall_ventilation};
}

std::array<double, 5> option_values(const WarmSubcycleOptions& p)
{
    return {p.relative_tolerance, p.absolute_mass_tolerance, p.absolute_number_tolerance,
        p.absolute_reflectivity_tolerance, p.absolute_temperature_tolerance};
}

bool same_options(const WarmSubcycleOptions& a, const WarmSubcycleOptions& b)
{
    return same_values(option_values(a), option_values(b)) && a.max_accepted_steps == b.max_accepted_steps &&
        a.max_trials == b.max_trials && a.max_halvings == b.max_halvings && a.observer_capacity == b.observer_capacity;
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

bool same_endpoints(const WarmRainDoubleEndpoints& a, const WarmRainDoubleEndpoints& b)
{
    return same_values(endpoint_values(a), endpoint_values(b)) && a.cloud_donor_exhausted == b.cloud_donor_exhausted &&
        a.evaporation.present == b.evaporation.present && a.evaporation.donor_exhausted == b.evaporation.donor_exhausted &&
        a.reclassification.present == b.reclassification.present;
}

bool supported_rain(const ExactRainTripleMomentState& rain, std::string& error)
{
    if (rain.mass_per_kg == 0.0 && rain.number_per_kg == 0.0 && rain.reflectivity_per_kg == 0.0)
    {
        return true;
    }
    TripleWarmRainSpectrum spectrum;
    if (!diagnose_triple_warm_rain_spectrum(rain, spectrum, error) || spectrum.alpha() > 200.0)
    {
        if (error.empty())
        {
            error = "subcycled warm rain is outside its shape bound";
        }
        return false;
    }
    return true;
}

bool regularize_number(ProcessState& state, const WarmRainParameters& p, bool before, std::string& error)
{
    double number = state.cell.density * state.rain.number_per_kg;
    const bool represented = before ? state.rain.mass_per_kg > 1e-14 && number > 1e-3
        : state.rain.mass_per_kg > 0.0 && number > 0.0;
    if (represented)
    {
        const double diameter = std::cbrt(state.cell.density * state.rain.mass_per_kg / (number * p.c_r));
        if (diameter > 3e-3)
        {
            const double previous = number;
            const double excess = diameter - 3e-3;
            number *= std::max(1.0 + 2e4 * excess * excess, std::pow(diameter / 5e-3, 3.0));
            if (!std::isfinite(number) || number <= 0.0)
            {
                error = "subcycled warm numerical number correction overflowed";
                return false;
            }
            state.rain.reflectivity_per_kg *= previous / number;
            state.rain.number_per_kg = number / state.cell.density;
            state.cell.rain_number = state.rain.number_per_kg;
        }
    }
    return supported_rain(state.rain, error);
}

ProcessState advance(const ProcessState& from, const ExactTripleDynamicWarmFP64CellResult& step, double original_temperature)
{
    ProcessState next = from;
    const double increment = step.endpoints.temperature_increment_k - next.heat_compensation;
    const double sum = next.heat + increment;
    next.heat_compensation = (sum - next.heat) - increment;
    next.heat = sum;
    next.cell.temperature = original_temperature + next.heat;
    next.cell.vapor = step.endpoints.vapor_per_kg;
    next.cell.cloud = step.endpoints.cloud_per_kg;
    next.cell.cloud_number = step.endpoints.cloud_number_per_kg;
    next.cell.rain = step.rain.mass_per_kg;
    next.cell.rain_number = step.rain.number_per_kg;
    next.rain = step.rain;
    return next;
}

double error_norm(const ProcessState& start, const ProcessState& full, const ProcessState& fine,
                  const WarmSubcycleOptions& options)
{
    const auto quantities = [](const ProcessState& s)
    {
        return std::array<double, 7>{s.cell.vapor, s.cell.cloud, s.cell.rain,
            s.cell.cloud_number, s.cell.rain_number, s.rain.reflectivity_per_kg, s.cell.temperature};
    };
    const auto a = quantities(start), b = quantities(full), c = quantities(fine);
    const std::array<double, 7> absolute{options.absolute_mass_tolerance, options.absolute_mass_tolerance,
        options.absolute_mass_tolerance, options.absolute_number_tolerance, options.absolute_number_tolerance,
        options.absolute_reflectivity_tolerance, options.absolute_temperature_tolerance};
    double norm = 0.0;
    for (std::size_t index = 0; index < a.size(); ++index)
    {
        const double scale = absolute[index] + options.relative_tolerance *
            std::max({std::abs(a[index]), std::abs(b[index]), std::abs(c[index])});
        norm = std::max(norm, std::abs(b[index] - c[index]) / (3.0 * scale));
    }
    return norm;
}

bool close(double initial, double expected, double actual)
{
    const double scale = std::max({std::abs(initial), std::abs(expected), std::abs(actual), 1e-30});
    return std::isfinite(expected) && std::isfinite(actual) &&
        std::abs(expected - actual) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

bool corrected_state_valid(const ProcessState& before, const ProcessState& after, std::string& error)
{
    for (double value : {after.cell.vapor, after.cell.cloud, after.cell.rain,
                         after.cell.cloud_number, after.cell.rain_number})
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "subcycled extrapolated primary state is invalid";
            return false;
        }
    }
    if (!std::isfinite(after.cell.temperature) || after.cell.temperature <= microphysics_constants::T0 ||
        after.cell.temperature >= 400.0 || (after.cell.cloud > 0.0 && after.cell.cloud_number <= 0.0) ||
        !supported_rain(after.rain, error))
    {
        if (error.empty())
        {
            error = "subcycled extrapolated thermal or moment state is invalid";
        }
        return false;
    }
    const double water = before.cell.vapor + before.cell.cloud + before.cell.rain;
    const double next_water = after.cell.vapor + after.cell.cloud + after.cell.rain;
    const double latent = microphysics_constants::L_v / microphysics_constants::cp;
    if (!close(water, water, next_water) || !close(before.cell.vapor * latent,
            latent * (before.cell.vapor - after.cell.vapor), after.heat - before.heat))
    {
        error = "subcycled extrapolated water or latent heat closure differs";
        return false;
    }
    return true;
}

bool second_order_flow(const WarmRainParameters& parameters, const ProcessState& current, double duration,
                       double original_temperature, const WarmSubcycleOptions& options,
                       ProcessState& output, WarmSubcycleAcceptedStep& trace,
                       DynamicWarmStepFailure& failure, std::string& error)
{
    const DynamicWarmFP64Policies policies{false, false, true};
    const double half = duration / 2.0;
    ExactTripleDynamicWarmFP64CellResult full, first, second;
    if (!(half > 0.0))
    {
        failure = DynamicWarmStepFailure::Resource;
        error = "subcycled extrapolation has no representable half interval";
        return false;
    }
    if (!evaluate_exact_triple_dynamic_warm_fp64_cell(parameters, current.cell, duration,
            current.rain, full, error, &failure, options.observer_capacity, policies) ||
        !evaluate_exact_triple_dynamic_warm_fp64_cell(parameters, current.cell, half,
            current.rain, first, error, &failure, options.observer_capacity, policies))
    {
        return false;
    }
    const ProcessState coarse = advance(current, full, original_temperature);
    const ProcessState middle = advance(current, first, original_temperature);
    if (!evaluate_exact_triple_dynamic_warm_fp64_cell(parameters, middle.cell, half,
            middle.rain, second, error, &failure, options.observer_capacity, policies))
    {
        return false;
    }
    const ProcessState fine = advance(middle, second, original_temperature);
    const auto extrapolate = [](double initial, double high, double low)
    {
        return initial + (low - initial) + (low - high);
    };
    const double heat = (first.endpoints.temperature_increment_k + second.endpoints.temperature_increment_k) * 2.0
        - full.endpoints.temperature_increment_k;
    ExactTripleDynamicWarmFP64CellResult corrected;
    corrected.endpoints = {extrapolate(current.cell.vapor, coarse.cell.vapor, fine.cell.vapor),
        extrapolate(current.cell.cloud, coarse.cell.cloud, fine.cell.cloud),
        extrapolate(current.cell.rain, coarse.cell.rain, fine.cell.rain),
        extrapolate(current.cell.cloud_number, coarse.cell.cloud_number, fine.cell.cloud_number),
        extrapolate(current.cell.rain_number, coarse.cell.rain_number, fine.cell.rain_number), heat};
    corrected.rain = {corrected.endpoints.rain_per_kg, corrected.endpoints.rain_number_per_kg,
        extrapolate(current.rain.reflectivity_per_kg, coarse.rain.reflectivity_per_kg, fine.rain.reflectivity_per_kg)};
    ProcessState candidate = advance(current, corrected, original_temperature);
    if (!corrected_state_valid(current, candidate, error))
    {
        failure = DynamicWarmStepFailure::IntervalRejected;
        return false;
    }
    WarmSubcycleAcceptedStep captured;
    captured.duration_s = duration;
    captured.source_cell = current.cell;
    captured.initial_rain = current.rain;
    captured.endpoints = corrected.endpoints;
    captured.final_rain = corrected.rain;
    captured.euler_probes = {{{duration, current.cell, current.rain, full.endpoints, full.rain},
        {half, current.cell, current.rain, first.endpoints, first.rain},
        {half, middle.cell, middle.rain, second.endpoints, second.rain}}};
    output = candidate;
    trace = captured;
    failure = DynamicWarmStepFailure::None;
    error.clear();
    return true;
}

} // namespace

struct WarmSubcycleBuilder
{
    static bool build(const WarmRainParameters& parameters, const DoubleWarmRainCell& source, double dt,
                      const RainTripleMomentState& initial, const WarmSubcycleOptions& options,
                      WarmSubcycleCellResult& result, std::string& error)
    {
        error.clear();
        for (double value : option_values(options))
        {
            if (!std::isfinite(value) || value <= 0.0)
            {
                error = "subcycled warm tolerance is invalid";
                return false;
            }
        }
        if (options.relative_tolerance > 1e-2 || options.max_accepted_steps < 2 || options.max_accepted_steps > 4096 ||
            options.max_trials == 0 || options.max_trials > 8192 || options.max_halvings > 40 || options.observer_capacity > 16)
        {
            error = "subcycled warm controller resource options are invalid";
            return false;
        }
        try
        {
            validate_triple_warm_rain_source(parameters, source, dt, {});
        }
        catch (const std::exception& failure)
        {
            error = failure.what();
            return false;
        }
        if (source.rain != initial.mass_per_kg || source.rain_number != initial.number_per_kg ||
            (source.cloud > 0.0 && source.cloud_number <= 0.0))
        {
            error = "subcycled warm original q/N source is invalid";
            return false;
        }
        ProcessState current{source, {initial.mass_per_kg, initial.number_per_kg, initial.reflectivity_per_kg}};
        if (!supported_rain(current.rain, error) || !regularize_number(current, parameters, true, error))
        {
            return false;
        }
        WarmSubcycleCellResult candidate;
        auto& producer = candidate.producer;
        producer.source_cell_ = source;
        producer.original_rain_ = initial;
        producer.parameters_ = parameters;
        producer.options_ = options;
        producer.requested_duration_s_ = dt;
        producer.pre_regularized_rain_ = current.rain;
        double elapsed = 0.0;
        double proposed = dt;
        std::size_t halvings = 0;
        while (elapsed < dt)
        {
            if (producer.trials_ >= options.max_trials || producer.steps_.size() + 2 > options.max_accepted_steps)
            {
                error = "subcycled warm controller exhausted its bounded work budget: elapsed=" +
                    std::to_string(elapsed) + ", duration=" + std::to_string(dt) +
                    ", accepted=" + std::to_string(producer.steps_.size()) + ", trials=" + std::to_string(producer.trials_);
                return false;
            }
            const double remaining = dt - elapsed;
            const double duration = std::min(proposed, remaining);
            const double half = duration / 2.0;
            if (!(duration > 0.0 && half > 0.0 && elapsed + duration > elapsed))
            {
                error = "subcycled warm controller cannot make representable time progress";
                return false;
            }
            ++producer.trials_;
            DynamicWarmStepFailure failure = DynamicWarmStepFailure::None;
            WarmSubcycleAcceptedStep full, first, second;
            ProcessState middle, fine, coarse;
            bool accepted = second_order_flow(parameters, current, duration, source.temperature,
                options, coarse, full, failure, error);
            double norm = std::numeric_limits<double>::infinity();
            if (accepted)
            {
                accepted = second_order_flow(parameters, current, half, source.temperature,
                    options, middle, first, failure, error);
            }
            if (accepted)
            {
                accepted = second_order_flow(parameters, middle, half, source.temperature,
                    options, fine, second, failure, error);
            }
            if (accepted)
            {
                norm = error_norm(current, coarse, fine, options);
                accepted = std::isfinite(norm) && norm <= 1.0;
            }
            if (!accepted)
            {
                if (failure != DynamicWarmStepFailure::None && failure != DynamicWarmStepFailure::IntervalRejected)
                {
                    return false;
                }
                ++producer.rejected_trials_;
                if (halvings >= options.max_halvings)
                {
                    error = "subcycled warm controller exhausted its interval refinement budget";
                    return false;
                }
                ++halvings;
                proposed = half;
                continue;
            }
            first.error_norm = norm;
            second.error_norm = norm;
            producer.steps_.push_back(first);
            producer.steps_.push_back(second);
            current = fine;
            elapsed = duration == remaining ? dt : elapsed + duration;
            proposed = norm <= 0.25 && duration <= (dt - elapsed) / 2.0 ? duration * 2.0 : duration;
            halvings = 0;
        }
        producer.completed_duration_s_ = elapsed;
        producer.post_regularization_before_ = current.rain;
        if (!regularize_number(current, parameters, false, error))
        {
            return false;
        }
        producer.final_rain_ = current.rain;
        candidate.endpoints = {current.cell.vapor, current.cell.cloud, current.cell.rain,
            current.cell.cloud_number, current.cell.rain_number, current.heat};
        candidate.rain = current.rain;
        const double water = source.vapor + source.cloud + source.rain;
        const double final_water = current.cell.vapor + current.cell.cloud + current.cell.rain;
        const double latent = microphysics_constants::L_v / microphysics_constants::cp;
        if (!close(water, water, final_water) ||
            !close(source.vapor * latent, latent * (source.vapor - current.cell.vapor), current.heat) ||
            !finalize_rain_phase_endpoint(current.rain.mass_per_kg, current.rain.number_per_kg,
                current.rain.reflectivity_per_kg, candidate.rounded_rain, error) ||
            !supported_rain({candidate.rounded_rain.mass_per_kg, candidate.rounded_rain.number_per_kg,
                candidate.rounded_rain.reflectivity_per_kg}, error))
        {
            if (error.empty())
            {
                error = "subcycled warm final water or latent heat closure is invalid";
            }
            return false;
        }
        for (double value : {current.cell.vapor, current.cell.cloud, current.cell.cloud_number})
        {
            if (!std::isfinite(value) || value < 0.0 || value > std::numeric_limits<float>::max() ||
                (value > 0.0 && value < std::numeric_limits<float>::denorm_min()))
            {
                error = "subcycled warm final primary field is not FP32 representable";
                return false;
            }
        }
        const std::array<double, 6> rates{(current.cell.cloud - source.cloud) / dt,
            (current.cell.rain - source.rain) / dt, (current.cell.vapor - source.vapor) / dt,
            current.heat / dt, (current.cell.cloud_number - source.cloud_number) / dt,
            (current.cell.rain_number - source.rain_number) / dt};
        for (double value : rates)
        {
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
            {
                error = "subcycled warm diagnostic rate is not FP32 representable";
                return false;
            }
        }
        candidate.rates = {static_cast<float>(rates[0]), static_cast<float>(rates[1]), static_cast<float>(rates[2]),
            static_cast<float>(rates[3]), static_cast<float>(rates[4]), static_cast<float>(rates[5])};
        result = std::move(candidate);
        error.clear();
        return true;
    }

    static bool same_producer(const WarmSubcycleProducer& a, const WarmSubcycleProducer& b)
    {
        if (!same_values(cell_values(a.source_cell_), cell_values(b.source_cell_)) ||
            !same_values(rain_values(a.original_rain_), rain_values(b.original_rain_)) ||
            !same_values(parameter_values(a.parameters_), parameter_values(b.parameters_)) ||
            !same_options(a.options_, b.options_) || a.trials_ != b.trials_ || a.rejected_trials_ != b.rejected_trials_ ||
            !same_values(std::array<double, 2>{a.completed_duration_s_, a.requested_duration_s_},
                std::array<double, 2>{b.completed_duration_s_, b.requested_duration_s_}) ||
            !same_values(rain_values(a.pre_regularized_rain_), rain_values(b.pre_regularized_rain_)) ||
            !same_values(rain_values(a.post_regularization_before_), rain_values(b.post_regularization_before_)) ||
            !same_values(rain_values(a.final_rain_), rain_values(b.final_rain_)) || a.steps_.size() != b.steps_.size())
        {
            return false;
        }
        for (std::size_t index = 0; index < a.steps_.size(); ++index)
        {
            const auto& x = a.steps_[index];
            const auto& y = b.steps_[index];
            if (!same_values(std::array<double, 2>{x.duration_s, x.error_norm}, std::array<double, 2>{y.duration_s, y.error_norm}) ||
                !same_values(cell_values(x.source_cell), cell_values(y.source_cell)) ||
                !same_values(rain_values(x.initial_rain), rain_values(y.initial_rain)) ||
                !same_values(rain_values(x.final_rain), rain_values(y.final_rain)) || !same_endpoints(x.endpoints, y.endpoints))
            {
                return false;
            }
            for (std::size_t probe = 0; probe < x.euler_probes.size(); ++probe)
            {
                const auto& u = x.euler_probes[probe];
                const auto& v = y.euler_probes[probe];
                if (!same_values(std::array<double, 1>{u.duration_s}, std::array<double, 1>{v.duration_s}) ||
                    !same_values(cell_values(u.source_cell), cell_values(v.source_cell)) ||
                    !same_values(rain_values(u.initial_rain), rain_values(v.initial_rain)) ||
                    !same_values(rain_values(u.final_rain), rain_values(v.final_rain)) || !same_endpoints(u.endpoints, v.endpoints))
                {
                    return false;
                }
            }
        }
        return true;
    }
};

bool evaluate_exact_triple_warm_subcycled_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, WarmSubcycleCellResult& result,
    std::string& error, const WarmSubcycleOptions& options)
{
    return WarmSubcycleBuilder::build(parameters, cell, dt, initial_rain, options, result, error);
}

bool apply_exact_triple_warm_subcycled_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, const WarmSubcycleProducer& producer,
    WarmSubcycleCellResult& result, std::string& error, const WarmSubcycleOptions& options)
{
    WarmSubcycleCellResult candidate;
    if (!WarmSubcycleBuilder::build(parameters, cell, dt, initial_rain, options, candidate, error))
    {
        return false;
    }
    if (!WarmSubcycleBuilder::same_producer(producer, candidate.producer))
    {
        error = "subcycled warm source, options or admitted trace provenance differs";
        return false;
    }
    result = std::move(candidate);
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
