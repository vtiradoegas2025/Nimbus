#include "triple_warm_primary.hpp"
#include "triple_sedimentation.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
bool validate_dynamic_warm_producer(
    const WarmRainParameters&, const DoubleWarmRainCell&, double,
    const RainTripleMomentState&, const TripleWarmRainDynamicResult&, std::string&);
bool validate_dynamic_warm_fp64_producer(
    const WarmRainParameters&, const DoubleWarmRainCell&, double,
    const ExactRainTripleMomentState&, const TripleWarmRainDynamicFP64Result&,
    const DynamicWarmFP64Policies&, std::string&);

namespace
{

bool representable(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0 && value <= std::numeric_limits<float>::max() &&
        (value == 0.0 || value >= std::numeric_limits<float>::denorm_min());
}

bool valid_rain(const ExactRainTripleMomentState& state, std::string& error)
{
    for (double value : {state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg})
    {
        if (!representable(value))
        {
            error = "exact triple warm rain q/N/Z is not FP32 representable";
            return false;
        }
    }
    if (state.mass_per_kg == 0.0 && state.number_per_kg == 0.0 && state.reflectivity_per_kg == 0.0)
    {
        return true;
    }
    double alpha = 0.0;
    double lambda = 0.0;
    return diagnose_triple_gamma_shape(state.mass_per_kg, state.number_per_kg, state.reflectivity_per_kg,
        current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Rain).mass_law, alpha, lambda, error);
}

bool valid_fp64_rain(const ExactRainTripleMomentState& state, std::string& error)
{
    if (state.mass_per_kg == 0.0 && state.number_per_kg == 0.0 && state.reflectivity_per_kg == 0.0)
    {
        return true;
    }
    TripleWarmRainSpectrum spectrum;
    if (!diagnose_triple_warm_rain_spectrum(state, spectrum, error))
    {
        return false;
    }
    if (spectrum.alpha() > 200.0)
    {
        error = "dynamic FP64 warm rain shape exceeds 200";
        return false;
    }
    return true;
}

std::array<double, 17> parameter_values(const WarmRainParameters& p)
{
    return {p.alpha_r, p.c_r, p.d_r, p.a_r, p.b_r, p.inv_mu_c,
            p.gc1, p.gc2, p.gc3, p.gc4, p.gr31, p.gr34, p.gr37,
            p.inv_cloud_mass_factor, p.inv_rain_mass_factor, p.gamma_ventilation, p.gamma_fall_ventilation};
}

bool close(double initial, double expected, double actual) noexcept
{
    const double scale = std::max({std::abs(initial), std::abs(expected), std::abs(actual), 1.0e-30});
    return std::isfinite(expected) && std::isfinite(actual) &&
        std::abs(expected - actual) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
}

template <typename Cell, typename Initial>
bool source_valid(const WarmRainParameters& parameters, const Cell& cell, double dt,
                  const Initial& initial, std::string& error)
{
    try
    {
        if constexpr (std::is_same_v<Initial, ExactRainTripleMomentState>)
        {
            validate_triple_warm_rain_source_impl(parameters, cell, dt, {});
            if (cell.temperature <= microphysics_constants::T0 || cell.temperature >= 400.0 ||
                cell.pressure < 100.0 || cell.pressure > 2.0e5)
            {
                error = "dynamic FP64 warm source is outside its warm thermodynamic bounds";
                return false;
            }
        }
        else
        {
            validate_triple_warm_rain_source(parameters, cell, dt, {});
        }
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
    if (initial.mass_per_kg != cell.rain || initial.number_per_kg != cell.rain_number ||
        (cell.cloud > 0.0f && cell.cloud_number <= 0.0f))
    {
        error = "exact triple warm source q/N differs or cloud number is invalid";
        return false;
    }
    if constexpr (std::is_same_v<Initial, ExactRainTripleMomentState>)
    {
        return valid_fp64_rain(initial, error);
    }
    else
    {
        return valid_rain({initial.mass_per_kg, initial.number_per_kg, initial.reflectivity_per_kg}, error);
    }
}

template <typename Cell, typename Initial, typename Producer, typename Result>
bool apply_exact_triple_warm_cell_impl(
    const WarmRainParameters& parameters, const Cell& cell, double dt,
    const Initial& initial_rain, const Producer& producer,
    Result& result, std::string& error, bool dynamic_rates = false,
    DynamicWarmStepFailure* failure = nullptr)
{
    constexpr bool unrounded = std::is_same_v<Initial, ExactRainTripleMomentState>;
    if (failure)
    {
        *failure = DynamicWarmStepFailure::InvalidSource;
    }
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    const auto& incoming = producer.incoming_rates();
    const auto& rates = producer.rates();
    const auto& endpoints = producer.endpoints();
    const auto& events = producer.events();
    const auto valid_endpoint = [](double value)
    {
        if constexpr (unrounded)
        {
            return std::isfinite(value) && value >= 0.0;
        }
        else
        {
            return representable(value);
        }
    };
    if constexpr (std::is_same_v<Cell, DoubleWarmRainCell>)
    {
        if (producer.uses_dynamic_spectrum() != dynamic_rates)
        {
            error = "exact triple warm producer rate policy differs";
            return false;
        }
    }
    if (std::memcmp(&cell, &producer.source_cell(), sizeof(cell)) != 0 ||
        parameter_values(parameters) != parameter_values(producer.parameters()) ||
        producer.duration_s() != dt || producer.dry_density_kg_m3() != cell.density ||
        incoming.cloud != 0.0f || incoming.rain != 0.0f || incoming.vapor != 0.0f ||
        incoming.temperature != 0.0f || incoming.cloud_number != 0.0f || incoming.rain_number != 0.0f ||
        events.capacity > events.events.size() || events.size > events.capacity)
    {
        error = "exact triple warm producer source, context or incoming provenance differs";
        return false;
    }
    for (double value : {endpoints.vapor_per_kg, endpoints.cloud_per_kg, endpoints.rain_per_kg,
                         endpoints.cloud_number_per_kg, endpoints.rain_number_per_kg})
    {
        if (!valid_endpoint(value))
        {
            error = unrounded ? "dynamic FP64 warm endpoint is invalid" :
                "exact triple warm endpoint is not FP32 representable";
            return false;
        }
    }
    if (!std::isfinite(endpoints.temperature_increment_k) ||
        (endpoints.cloud_per_kg > 0.0 && endpoints.cloud_number_per_kg <= 0.0))
    {
        if constexpr (unrounded)
        {
            if (failure)
            {
                *failure = DynamicWarmStepFailure::IntervalRejected;
            }
        }
        error = "exact triple warm cloud or heat endpoint is invalid";
        return false;
    }
    for (float value : {rates.cloud, rates.rain, rates.vapor, rates.temperature, rates.cloud_number, rates.rain_number})
    {
        if (!std::isfinite(value))
        {
            error = "exact triple warm diagnostic rate is invalid";
            return false;
        }
    }
    AcceptedRainPhaseBatch batch;
    batch.duration_s = dt;
    std::array<bool, 16> seen{};
    std::array<double, 6> sums{};
    std::array<double, 6> pre_reclassification_sums{};
    double admitted_cloud_mass_rate = 0.0;
    bool limited_cloud_admission = false;
    for (std::size_t index = 0; index < events.size; ++index)
    {
        const auto& event = events.events[index];
        const auto kind = static_cast<std::size_t>(event.kind);
        if (kind >= (dynamic_rates ? seen.size() : 15) || seen[kind])
        {
            error = "exact triple warm producer has duplicate or unknown event";
            return false;
        }
        seen[kind] = true;
        if (event.kind == WarmRainAcceptedEventKind::RainToCloud)
        {
            pre_reclassification_sums = sums;
        }
        if (event.kind == WarmRainAcceptedEventKind::Autoconversion ||
            event.kind == WarmRainAcceptedEventKind::Accretion)
        {
            admitted_cloud_mass_rate -= event.cloud_mass_rate_per_kg_s;
            limited_cloud_admission = limited_cloud_admission || event.cloud_donor_scale < 1.0;
        }
        bool mapped = false;
        if constexpr (std::is_same_v<Cell, DoubleWarmRainCell>)
        {
            if (dynamic_rates)
            {
                mapped = map_dynamic_triple_produced_rain_event(event, dt, cell.density, batch, error);
            }
            else if (event.kind == WarmRainAcceptedEventKind::RainToCloud &&
                endpoints.reclassification.present)
            {
                // Published MY diagnoses cloud's fixed spectrum; it does not prognose cloud Z.
                mapped = map_observed_rain_event(event, dt, cell.density, batch, error);
            }
            else
            {
                mapped = map_triple_produced_rain_event(event, dt, cell.density, batch, error);
            }
        }
        else
        {
            mapped = map_triple_produced_rain_event(event, dt, cell.density, batch, error);
        }
        if (!mapped)
        {
            return false;
        }
        sums[0] += event.vapor_mass_rate_per_kg_s;
        sums[1] += event.cloud_mass_rate_per_kg_s;
        sums[2] += event.rain_mass_rate_per_kg_s;
        sums[3] += event.cloud_number_rate_per_m3_s;
        sums[4] += event.rain_number_rate_per_m3_s;
        sums[5] += event.temperature_rate_k_s;
    }
    const auto& reclassification = endpoints.reclassification;
    const bool seen_reclassification = seen[static_cast<std::size_t>(WarmRainAcceptedEventKind::RainToCloud)];
    if (reclassification.present != seen_reclassification ||
        (!reclassification.present &&
         (reclassification.rain_mass_before_per_kg != 0.0 ||
          reclassification.rain_number_before_per_m3 != 0.0 ||
          reclassification.cloud_mass_before_per_kg != 0.0 ||
          reclassification.cloud_number_before_per_m3 != 0.0)))
    {
        error = "exact triple warm reclassification presence or empty metadata differs";
        return false;
    }
    if (reclassification.present)
    {
        if (!std::is_same_v<Cell, DoubleWarmRainCell> ||
            !valid_endpoint(reclassification.rain_mass_before_per_kg) ||
            reclassification.rain_mass_before_per_kg <= 0.0 ||
            !std::isfinite(reclassification.rain_number_before_per_m3) ||
            reclassification.rain_number_before_per_m3 <= 0.0 ||
            !valid_endpoint(reclassification.cloud_mass_before_per_kg) ||
            !std::isfinite(reclassification.cloud_number_before_per_m3) ||
            reclassification.cloud_number_before_per_m3 < 0.0 ||
            !(std::cbrt(cell.density * reclassification.rain_mass_before_per_kg /
                (reclassification.rain_number_before_per_m3 * parameters.c_r)) < 82.0e-6) ||
            endpoints.cloud_per_kg != reclassification.cloud_mass_before_per_kg +
                reclassification.rain_mass_before_per_kg ||
            endpoints.cloud_number_per_kg != (reclassification.cloud_number_before_per_m3 +
                reclassification.rain_number_before_per_m3) / cell.density ||
            endpoints.rain_per_kg != 0.0 || endpoints.rain_number_per_kg != 0.0 ||
            endpoints.vapor_per_kg != cell.vapor || endpoints.temperature_increment_k != 0.0 ||
            endpoints.evaporation.present)
        {
            error = "exact triple warm reclassification source or authoritative endpoint differs";
            return false;
        }
        for (std::size_t index = 0; index < events.size; ++index)
        {
            const auto& event = events.events[index];
            if (event.kind != WarmRainAcceptedEventKind::RainToCloud)
            {
                continue;
            }
            if (event.cloud_mass_rate_per_kg_s != reclassification.rain_mass_before_per_kg / dt ||
                event.rain_mass_rate_per_kg_s != -reclassification.rain_mass_before_per_kg / dt ||
                event.cloud_number_rate_per_m3_s != reclassification.rain_number_before_per_m3 / dt ||
                event.rain_number_rate_per_m3_s != -reclassification.rain_number_before_per_m3 / dt ||
                !close(cell.cloud, endpoints.cloud_donor_exhausted
                    ? static_cast<double>(cell.cloud) - endpoints.cloud_donor_budget_per_kg
                    : static_cast<double>(cell.cloud) + dt * pre_reclassification_sums[1],
                    reclassification.cloud_mass_before_per_kg) ||
                !close(cell.density * cell.cloud_number, cell.density * cell.cloud_number +
                    dt * pre_reclassification_sums[3],
                    reclassification.cloud_number_before_per_m3))
            {
                error = "exact triple warm reclassification budget and journal disagree";
                return false;
            }
        }
    }
    const auto& evaporation = endpoints.evaporation;
    const bool seen_evaporation = seen[static_cast<std::size_t>(WarmRainAcceptedEventKind::Evaporation)];
    if (evaporation.present != seen_evaporation ||
        (!evaporation.present &&
         (evaporation.mass_before_per_kg != 0.0 || evaporation.number_before_per_m3 != 0.0 ||
          evaporation.admitted_mass_per_kg != 0.0 || evaporation.mass_after_per_kg != 0.0 ||
          evaporation.number_after_per_m3 != 0.0 || evaporation.donor_exhausted)))
    {
        error = "exact triple warm evaporation budget presence or empty metadata differs";
        return false;
    }
    AdmittedRainEvaporationEndpoint phase_evaporation;
    if (evaporation.present)
    {
        if (!std::isfinite(evaporation.mass_before_per_kg) || evaporation.mass_before_per_kg <= 0.0 ||
            !std::isfinite(evaporation.number_before_per_m3) || evaporation.number_before_per_m3 <= 0.0 ||
            !std::isfinite(evaporation.admitted_mass_per_kg) || evaporation.admitted_mass_per_kg <= 0.0 ||
            evaporation.admitted_mass_per_kg > evaporation.mass_before_per_kg ||
            !std::isfinite(evaporation.mass_after_per_kg) || evaporation.mass_after_per_kg < 0.0 ||
            evaporation.mass_after_per_kg != evaporation.mass_before_per_kg - evaporation.admitted_mass_per_kg ||
            !std::isfinite(evaporation.number_after_per_m3) || evaporation.number_after_per_m3 < 0.0 ||
            evaporation.donor_exhausted != (evaporation.admitted_mass_per_kg == evaporation.mass_before_per_kg) ||
            !close(evaporation.number_before_per_m3,
                   evaporation.number_before_per_m3 * (evaporation.mass_after_per_kg / evaporation.mass_before_per_kg),
                   evaporation.number_after_per_m3) ||
            endpoints.rain_per_kg != evaporation.mass_after_per_kg ||
            endpoints.rain_number_per_kg != evaporation.number_after_per_m3 / cell.density ||
            endpoints.vapor_per_kg != static_cast<double>(cell.vapor) + evaporation.admitted_mass_per_kg ||
            endpoints.temperature_increment_k != -microphysics_constants::L_v / microphysics_constants::cp
                * evaporation.admitted_mass_per_kg)
        {
            error = "exact triple warm evaporation budget and authoritative endpoints disagree";
            return false;
        }
        for (std::size_t index = 0; index < events.size; ++index)
        {
            const auto& event = events.events[index];
            if (event.kind != WarmRainAcceptedEventKind::Evaporation)
            {
                continue;
            }
            if (!close(evaporation.mass_before_per_kg, dt * event.rain_mass_rate_per_kg_s,
                       -evaporation.admitted_mass_per_kg) ||
                !close(evaporation.number_before_per_m3, dt * event.rain_number_rate_per_m3_s,
                       evaporation.number_after_per_m3 - evaporation.number_before_per_m3) ||
                !close(evaporation.admitted_mass_per_kg, dt * event.vapor_mass_rate_per_kg_s,
                       evaporation.admitted_mass_per_kg) ||
                !close(endpoints.temperature_increment_k, dt * event.temperature_rate_k_s,
                       endpoints.temperature_increment_k))
            {
                error = "exact triple warm evaporation budget and journal disagree";
                return false;
            }
        }
        phase_evaporation = {true, evaporation.mass_before_per_kg,
            evaporation.number_before_per_m3 / cell.density, evaporation.admitted_mass_per_kg,
            evaporation.mass_after_per_kg, evaporation.number_after_per_m3 / cell.density,
            evaporation.donor_exhausted};
    }
    bool full_cloud_admission = limited_cloud_admission;
    if constexpr (std::is_same_v<Cell, DoubleWarmRainCell>)
    {
        full_cloud_admission = full_cloud_admission ||
            (cell.cloud > 0.0 && admitted_cloud_mass_rate >= cell.cloud / dt);
    }
    if (endpoints.cloud_donor_exhausted != full_cloud_admission ||
        (endpoints.cloud_donor_exhausted
            ? endpoints.cloud_donor_budget_per_kg != cell.cloud ||
              (reclassification.present ? reclassification.cloud_mass_before_per_kg : endpoints.cloud_per_kg) != 0.0 ||
              !close(cell.cloud, dt * admitted_cloud_mass_rate, endpoints.cloud_donor_budget_per_kg)
            : endpoints.cloud_donor_budget_per_kg != 0.0))
    {
        error = "exact triple warm cloud donor budget provenance differs";
        return false;
    }
    if (!close(cell.vapor, static_cast<double>(cell.vapor) + dt * sums[0], endpoints.vapor_per_kg) ||
        !close(cell.cloud, static_cast<double>(cell.cloud) + dt * sums[1], endpoints.cloud_per_kg) ||
        !close(cell.rain, static_cast<double>(cell.rain) + dt * sums[2], endpoints.rain_per_kg) ||
        !close(static_cast<double>(cell.density) * cell.cloud_number,
               static_cast<double>(cell.density) * cell.cloud_number + dt * sums[3],
               endpoints.cloud_number_per_kg * cell.density) ||
        !close(static_cast<double>(cell.density) * cell.rain_number,
               static_cast<double>(cell.density) * cell.rain_number + dt * sums[4],
               endpoints.rain_number_per_kg * cell.density) ||
        !close(0.0, dt * sums[5], endpoints.temperature_increment_k))
    {
        error = "exact triple warm producer endpoint and event journal disagree";
        return false;
    }
    RainTripleMomentState rounded_phase;
    ExactRainTripleMomentState exact_phase;
    bool phase_applied = false;
    if constexpr (unrounded)
    {
        phase_applied = apply_admitted_rain_phases_with_evaporation_endpoint_fp64(
            initial_rain, batch, phase_evaporation, exact_phase, error, failure);
    }
    else
    {
        phase_applied = apply_admitted_rain_phases_with_evaporation_endpoint(
            initial_rain, batch, phase_evaporation, rounded_phase, exact_phase, error);
    }
    if (!phase_applied)
    {
        return false;
    }
    if (!close(cell.rain, exact_phase.mass_per_kg, endpoints.rain_per_kg) ||
        !close(cell.rain_number, exact_phase.number_per_kg, endpoints.rain_number_per_kg))
    {
        if (failure)
        {
            *failure = DynamicWarmStepFailure::InvalidSource;
        }
        error = "exact triple warm rain phase and authoritative producer q/N disagree";
        return false;
    }
    Result candidate;
    if constexpr (!unrounded)
    {
        candidate.producer = producer;
    }
    candidate.endpoints = endpoints;
    candidate.rates = rates;
    candidate.rain = {endpoints.rain_per_kg, endpoints.rain_number_per_kg, exact_phase.reflectivity_per_kg};
    if constexpr (unrounded)
    {
        if (!valid_fp64_rain(candidate.rain, error) ||
            !std::isfinite(cell.temperature + endpoints.temperature_increment_k) ||
            cell.temperature + endpoints.temperature_increment_k <= microphysics_constants::T0 ||
            cell.temperature + endpoints.temperature_increment_k >= 400.0)
        {
            if (error.empty())
            {
                error = "dynamic FP64 warm thermal endpoint is outside its warm bounds";
            }
            if (failure)
            {
                *failure = DynamicWarmStepFailure::IntervalRejected;
            }
            return false;
        }
    }
    else
    {
        candidate.rounded_rain = {static_cast<float>(candidate.rain.mass_per_kg),
                                  static_cast<float>(candidate.rain.number_per_kg),
                                  static_cast<float>(candidate.rain.reflectivity_per_kg)};
        if (!valid_rain(candidate.rain, error) ||
            !valid_rain({candidate.rounded_rain.mass_per_kg, candidate.rounded_rain.number_per_kg,
                         candidate.rounded_rain.reflectivity_per_kg}, error))
        {
            return false;
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

} // namespace

bool apply_exact_triple_warm_cell(
    const WarmRainParameters& parameters, const WarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, const TripleWarmRainExactResult& producer,
    ExactTripleWarmCellResult& result, std::string& error)
{
    return apply_exact_triple_warm_cell_impl(parameters, cell, dt, initial_rain, producer, result, error);
}

bool evaluate_exact_triple_warm_cell(
    const WarmRainParameters& parameters, const WarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, ExactTripleWarmCellResult& result,
    std::string& error, std::size_t observer_capacity)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    TripleWarmRainExactResult producer;
    try
    {
        triple_warm_rain_cell_exact(parameters, cell, dt, {}, producer, observer_capacity);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
    return apply_exact_triple_warm_cell(parameters, cell, dt, initial_rain, producer, result, error);
}

bool apply_exact_triple_warm_double_source_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, const TripleWarmRainDoubleSourceResult& producer,
    ExactTripleWarmDoubleSourceCellResult& result, std::string& error)
{
    return apply_exact_triple_warm_cell_impl(parameters, cell, dt, initial_rain, producer, result, error);
}

bool evaluate_exact_triple_warm_double_source_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, ExactTripleWarmDoubleSourceCellResult& result,
    std::string& error, std::size_t observer_capacity)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    TripleWarmRainDoubleSourceResult producer;
    try
    {
        triple_warm_rain_cell_exact_double_source(parameters, cell, dt, {}, producer, observer_capacity);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
    return apply_exact_triple_warm_double_source_cell(parameters, cell, dt, initial_rain, producer, result, error);
}

bool apply_exact_triple_dynamic_warm_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, const TripleWarmRainDynamicResult& producer,
    ExactTripleDynamicWarmCellResult& result, std::string& error)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error) ||
        !validate_dynamic_warm_producer(parameters, cell, dt, initial_rain, producer, error))
    {
        return false;
    }
    ExactTripleWarmDoubleSourceCellResult validated;
    if (!apply_exact_triple_warm_cell_impl(parameters, cell, dt, initial_rain,
            producer.producer(), validated, error, true))
    {
        return false;
    }
    ExactTripleDynamicWarmCellResult candidate;
    candidate.endpoints = validated.endpoints;
    candidate.rain = validated.rain;
    candidate.rounded_rain = validated.rounded_rain;
    candidate.rates = validated.rates;
    candidate.producer = producer;
    result = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_dynamic_warm_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const RainTripleMomentState& initial_rain, ExactTripleDynamicWarmCellResult& result,
    std::string& error, std::size_t observer_capacity)
{
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    TripleWarmRainDynamicResult producer;
    try
    {
        triple_warm_rain_cell_dynamic(parameters, cell, dt, {}, initial_rain, producer, observer_capacity);
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
    return apply_exact_triple_dynamic_warm_cell(parameters, cell, dt, initial_rain, producer, result, error);
}

bool apply_exact_triple_dynamic_warm_fp64_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const ExactRainTripleMomentState& initial_rain, const TripleWarmRainDynamicFP64Result& producer,
    ExactTripleDynamicWarmFP64CellResult& result, std::string& error,
    DynamicWarmStepFailure* failure, const DynamicWarmFP64Policies& policies)
{
    error.clear();
    if (failure)
    {
        *failure = DynamicWarmStepFailure::InvalidSource;
    }
    if (!source_valid(parameters, cell, dt, initial_rain, error) ||
        !validate_dynamic_warm_fp64_producer(parameters, cell, dt, initial_rain, producer, policies, error))
    {
        return false;
    }
    ExactTripleDynamicWarmFP64CellResult candidate;
    if (!apply_exact_triple_warm_cell_impl(parameters, cell, dt, initial_rain,
            producer.producer(), candidate, error, true, failure))
    {
        return false;
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_dynamic_warm_fp64_cell(
    const WarmRainParameters& parameters, const DoubleWarmRainCell& cell, double dt,
    const ExactRainTripleMomentState& initial_rain, ExactTripleDynamicWarmFP64CellResult& result,
    std::string& error, DynamicWarmStepFailure* failure, std::size_t observer_capacity,
    const DynamicWarmFP64Policies& policies)
{
    error.clear();
    if (failure)
    {
        *failure = DynamicWarmStepFailure::InvalidSource;
    }
    if (!source_valid(parameters, cell, dt, initial_rain, error))
    {
        return false;
    }
    TripleWarmRainDynamicFP64Result producer;
    try
    {
        triple_warm_rain_cell_dynamic_fp64(parameters, cell, dt, {}, initial_rain,
            producer, observer_capacity, policies);
    }
    catch (const WarmRainObserverCapacityError& rejection)
    {
        error = rejection.what();
        if (failure)
        {
            *failure = DynamicWarmStepFailure::Resource;
        }
        return false;
    }
    catch (const DynamicWarmIntervalError& rejection)
    {
        error = rejection.what();
        if (failure)
        {
            *failure = DynamicWarmStepFailure::IntervalRejected;
        }
        return false;
    }
    catch (const std::exception& rejection)
    {
        error = rejection.what();
        return false;
    }
    ExactTripleDynamicWarmFP64CellResult candidate;
    if (!apply_exact_triple_warm_cell_impl(parameters, cell, dt, initial_rain,
            producer.producer(), candidate, error, true, failure))
    {
        return false;
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
