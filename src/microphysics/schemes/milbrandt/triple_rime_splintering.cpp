#include "triple_rime_splintering.hpp"

#include "cold.hpp"
#include "triple_sedimentation.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr double embryo_mass_kg = 1.0e-12;
constexpr double ice_mass_coefficient = 440.0;
constexpr double initial_ice_shape_ratio = 20.0;
constexpr std::array<TripleMomentSpecies, 4> species{
    TripleMomentSpecies::Ice, TripleMomentSpecies::Snow,
    TripleMomentSpecies::Graupel, TripleMomentSpecies::Hail};

bool reject(RimeSplinteringFailure kind, const char* message,
            std::string& error, RimeSplinteringFailure* failure)
{
    error = message;
    if (failure)
        *failure = kind;
    return false;
}

std::array<double, 25> parameter_values(const ColdParameters& p) noexcept
{
    return {p.a_g_, p.a_h_, p.a_i_, p.a_r_, p.a_s_,
        p.alpha_g_, p.alpha_h_, p.alpha_i_, p.alpha_r_, p.alpha_s_,
        p.b_g_, p.b_h_, p.b_i_, p.b_r_, p.b_s_,
        p.c_g_, p.c_h_, p.c_i_, p.c_r_, p.c_s_,
        p.d_g_, p.d_h_, p.d_i_, p.d_r_, p.d_s_};
}

bool same_value(double a, double b) noexcept
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

template <std::size_t Count>
bool same_values(const std::array<double, Count>& a,
                 const std::array<double, Count>& b) noexcept
{
    for (std::size_t index = 0; index < a.size(); ++index)
    {
        if (!same_value(a[index], b[index]))
            return false;
    }
    return true;
}

bool same_source(const RimeSplinteringCell& a, const RimeSplinteringCell& b) noexcept
{
    if (!same_value(a.temperature_k, b.temperature_k) ||
        !same_value(a.pressure_pa, b.pressure_pa) ||
        !same_value(a.dry_density_kg_m3, b.dry_density_kg_m3) ||
        !same_value(a.cloud_mass_per_kg, b.cloud_mass_per_kg) ||
        !same_values(a.admitted_cloud_mass_per_kg, b.admitted_cloud_mass_per_kg))
        return false;
    for (std::size_t index = 0; index < a.initial.size(); ++index)
    {
        const auto& x = a.initial[index];
        const auto& y = b.initial[index];
        if (!same_value(x.mass_per_kg, y.mass_per_kg) ||
            !same_value(x.number_per_kg, y.number_per_kg) ||
            !same_value(x.reflectivity_per_kg, y.reflectivity_per_kg))
            return false;
    }
    return true;
}

double cubic_log_shape_ratio(double alpha) noexcept
{
    return std::log1p(3.0 / (alpha + 1.0)) +
        std::log1p(3.0 / (alpha + 2.0)) +
        std::log1p(3.0 / (alpha + 3.0));
}

bool cubic_shape(const FrozenCollectionPopulation& source, double coefficient,
                 double& alpha) noexcept
{
    int q_exponent = 0;
    int n_exponent = 0;
    int z_exponent = 0;
    int c_exponent = 0;
    const double q_fraction = std::frexp(source.mass_per_kg, &q_exponent);
    const double n_fraction = std::frexp(source.number_per_kg, &n_exponent);
    const double z_fraction = std::frexp(source.reflectivity_per_kg, &z_exponent);
    const double c_fraction = std::frexp(coefficient, &c_exponent);
    const double coefficient_over_mass = c_fraction / q_fraction;
    const double log_ratio = std::log(std::ldexp(
        z_fraction * n_fraction * coefficient_over_mass * coefficient_over_mass,
        z_exponent + n_exponent + 2 * (c_exponent - q_exponent)));
    if (!std::isfinite(log_ratio) || log_ratio <= 0.0)
        return false;
    double lower = -1.0 + 1.0e-9;
    double upper = 1.0;
    if (!(log_ratio < cubic_log_shape_ratio(lower)))
        return false;
    while (cubic_log_shape_ratio(upper) > log_ratio && upper < 1.0e6)
        upper *= 2.0;
    if (upper > 1.0e6 || cubic_log_shape_ratio(upper) > log_ratio)
        return false;
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const double middle = 0.5 * (lower + upper);
        if (cubic_log_shape_ratio(middle) > log_ratio)
            lower = middle;
        else
            upper = middle;
    }
    alpha = 0.5 * (lower + upper);
    return std::isfinite(alpha);
}

bool valid_population(const FrozenCollectionPopulation& population,
                      TripleMomentSpecies category, std::string& error, ColdSpectrumDomain spectrum_domain)
{
    for (double value : {population.mass_per_kg, population.number_per_kg,
                         population.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "rime splintering source or endpoint q/N/Z is nonfinite or negative";
            return false;
        }
    }
    if (population.mass_per_kg == 0.0 && population.number_per_kg == 0.0 &&
        population.reflectivity_per_kg == 0.0)
        return true;
    const auto law = current_nimbus_triple_sedimentation_parameters(category).mass_law;
    double alpha = 0.0;
    double lambda = 0.0;
    if (!diagnose_triple_gamma_shape(population.mass_per_kg, population.number_per_kg,
            population.reflectivity_per_kg, law, alpha, lambda, error))
        return false;
    if (law.exponent == 3.0 && !cubic_shape(population, law.coefficient, alpha))
    {
        error = "rime splintering cubic spectrum is outside its admitted bracket";
        return false;
    }
    if (alpha > cold_spectrum_shape_limit(spectrum_domain, category))
    {
        error = cold_spectrum_shape_limit(spectrum_domain, category) == 200.0
            ? "rime splintering gamma shape exceeds 200"
            : "rime splintering converted-graupel gamma shape exceeds 512";
        return false;
    }
    return true;
}

bool float_range(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0 &&
        value <= std::numeric_limits<float>::max() &&
        (value == 0.0 || value >= std::numeric_limits<float>::denorm_min());
}

bool same_capture(const RimeSplinteringProducer& a, const RimeSplinteringProducer& b) noexcept
{
    if (a.event_count() != b.event_count() || !same_source(a.source(), b.source()) ||
        !same_values(a.parameters(), b.parameters()) || a.hail_processes() != b.hail_processes() ||
        a.spectrum_domain() != b.spectrum_domain() ||
        !same_value(a.duration_s(), b.duration_s()) ||
        !same_value(a.splinters_per_kg_collected(), b.splinters_per_kg_collected()) ||
        !same_values(a.mass_increment_per_kg(), b.mass_increment_per_kg()) ||
        !same_values(a.number_increment_per_kg(), b.number_increment_per_kg()) ||
        !same_values(a.reflectivity_increment_per_kg(), b.reflectivity_increment_per_kg()))
        return false;
    for (std::size_t index = 0; index < a.channels().size(); ++index)
    {
        const auto& x = a.channels()[index];
        const auto& y = b.channels()[index];
        if (x.species != y.species || !same_value(x.embryo_mass_per_kg, y.embryo_mass_per_kg) ||
            !same_value(x.number_increment_per_kg, y.number_increment_per_kg) ||
            !same_value(x.reflectivity_increment_per_kg, y.reflectivity_increment_per_kg))
            return false;
    }
    return true;
}

bool apply_endpoint(const RimeSplinteringProducer& producer, RimeSplinteringResult& result,
                    std::string& error, RimeSplinteringFailure* failure)
{
    RimeSplinteringResult candidate;
    candidate.mass_increment_per_kg = producer.mass_increment_per_kg();
    candidate.number_increment_per_kg = producer.number_increment_per_kg();
    candidate.reflectivity_increment_per_kg = producer.reflectivity_increment_per_kg();
    long double water_change = 0.0L;
    long double water_scale = 0.0L;
    for (std::size_t index = 0; index < candidate.unrounded.size(); ++index)
    {
        const auto& initial = producer.source().initial[index];
        auto& endpoint = candidate.unrounded[index];
        endpoint = initial;
        if (candidate.mass_increment_per_kg[index] != 0.0)
            endpoint.mass_per_kg += candidate.mass_increment_per_kg[index];
        if (candidate.number_increment_per_kg[index] != 0.0)
            endpoint.number_per_kg += candidate.number_increment_per_kg[index];
        if (candidate.reflectivity_increment_per_kg[index] != 0.0)
            endpoint.reflectivity_per_kg += candidate.reflectivity_increment_per_kg[index];
        if (!valid_population(endpoint, species[index], error, producer.spectrum_domain()) ||
            !float_range(endpoint.mass_per_kg) || !float_range(endpoint.number_per_kg) ||
            !float_range(endpoint.reflectivity_per_kg))
        {
            if (error.empty())
                error = "rime splintering isolated endpoint is not FP32 representable";
            if (failure)
                *failure = RimeSplinteringFailure::IntervalRejected;
            return false;
        }
        auto& stored = candidate.rounded[index];
        stored = {static_cast<float>(endpoint.mass_per_kg), static_cast<float>(endpoint.number_per_kg),
            static_cast<float>(endpoint.reflectivity_per_kg)};
        if (!valid_population({stored.mass_per_kg, stored.number_per_kg, stored.reflectivity_per_kg},
                species[index], error, producer.spectrum_domain()))
        {
            if (failure)
                *failure = RimeSplinteringFailure::IntervalRejected;
            return false;
        }
        water_change += static_cast<long double>(endpoint.mass_per_kg) - initial.mass_per_kg;
        water_scale += initial.mass_per_kg;
    }
    if (std::abs(water_change) > 64.0L * std::numeric_limits<double>::epsilon() * water_scale)
        return reject(RimeSplinteringFailure::IntervalRejected,
            "rime splintering isolated endpoint violates its frozen-water budget", error, failure);
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure)
        *failure = RimeSplinteringFailure::None;
    return true;
}

} // namespace

bool produce_exact_rime_splintering_admission(
    const ColdParameters& parameters, const RimeSplinteringCell& cell,
    double duration_s, RimeSplinteringProducer& producer, std::string& error,
    std::size_t capacity, RimeSplinteringFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (failure)
        *failure = RimeSplinteringFailure::None;
    if (!valid_cold_spectrum_domain(spectrum_domain))
    {
        return reject(RimeSplinteringFailure::InvalidContext,
            "rime splintering spectrum domain is invalid", error, failure);
    }
    if (!std::isfinite(cell.temperature_k) || cell.temperature_k < 150.0 ||
        cell.temperature_k >= microphysics_constants::T0 ||
        !std::isfinite(cell.pressure_pa) || cell.pressure_pa < 100.0 || cell.pressure_pa > 200000.0 ||
        !std::isfinite(cell.dry_density_kg_m3) || cell.dry_density_kg_m3 <= 0.0 ||
        !std::isfinite(duration_s) || duration_s <= 0.0 ||
        parameter_values(parameters) != default_cold_parameters)
        return reject(RimeSplinteringFailure::InvalidContext,
            "rime splintering context or retained physical parameters differ", error, failure);
    if (capacity != 3)
        return reject(RimeSplinteringFailure::Capacity,
            "rime splintering requires exactly three bounded channel records", error, failure);
    if (!std::isfinite(cell.cloud_mass_per_kg) || cell.cloud_mass_per_kg < 0.0)
        return reject(RimeSplinteringFailure::InvalidSource,
            "rime splintering original cloud mass is invalid", error, failure);
    double total_collected = 0.0;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        if (!valid_population(cell.initial[index], species[index], error, spectrum_domain))
        {
            if (failure)
                *failure = RimeSplinteringFailure::InvalidSource;
            return false;
        }
        const double mass = cell.admitted_cloud_mass_per_kg[index];
        if (!std::isfinite(mass) || mass < 0.0 ||
            (mass > 0.0 && cell.initial[index].mass_per_kg == 0.0))
            return reject(RimeSplinteringFailure::InvalidSource,
                "rime splintering accepted cloud mass has no original collector or is invalid", error, failure);
        total_collected += mass;
    }
    if (!std::isfinite(total_collected) || total_collected > cell.cloud_mass_per_kg)
        return reject(RimeSplinteringFailure::InvalidSource,
            "rime splintering accepted cloud journal exceeds its original shared donor", error, failure);
    RimeSplinteringProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameter_values(parameters);
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.duration_s_ = duration_s;
    candidate.spectrum_domain_ = spectrum_domain;
    const double temperature_c = cell.temperature_k - microphysics_constants::T0;
    if (temperature_c >= -8.0 && temperature_c <= -5.0)
        candidate.splinters_per_kg_collected_ = 3.5e8 * (temperature_c + 8.0) / 3.0;
    else if (temperature_c > -5.0 && temperature_c < -3.0)
        candidate.splinters_per_kg_collected_ = 3.5e8 * (-3.0 - temperature_c) * 0.5;
    for (std::size_t index = 0; index < candidate.channels_.size(); ++index)
    {
        auto& channel = candidate.channels_[index];
        channel.species = static_cast<RimeSplinteringSpecies>(index);
        channel.number_increment_per_kg = candidate.splinters_per_kg_collected_ *
            cell.admitted_cloud_mass_per_kg[index];
        channel.embryo_mass_per_kg = embryo_mass_kg * channel.number_increment_per_kg;
        channel.reflectivity_increment_per_kg = initial_ice_shape_ratio * embryo_mass_kg *
            channel.embryo_mass_per_kg / (ice_mass_coefficient * ice_mass_coefficient);
        if (!std::isfinite(channel.number_increment_per_kg) ||
            !std::isfinite(channel.embryo_mass_per_kg) ||
            !std::isfinite(channel.reflectivity_increment_per_kg) ||
            (channel.number_increment_per_kg > 0.0 &&
             (channel.embryo_mass_per_kg <= 0.0 || channel.reflectivity_increment_per_kg <= 0.0)))
            return reject(RimeSplinteringFailure::IntervalRejected,
                "rime splintering nonzero B4 transfer is unrepresentable", error, failure);
        candidate.number_increment_per_kg_[0] += channel.number_increment_per_kg;
        candidate.reflectivity_increment_per_kg_[0] += channel.reflectivity_increment_per_kg;
        if (index != 0)
        {
            candidate.mass_increment_per_kg_[0] += channel.embryo_mass_per_kg;
            candidate.mass_increment_per_kg_[index] -= channel.embryo_mass_per_kg;
            candidate.reflectivity_increment_per_kg_[index] -= channel.reflectivity_increment_per_kg;
        }
    }
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        for (double value : {candidate.mass_increment_per_kg_[index], candidate.number_increment_per_kg_[index],
                             candidate.reflectivity_increment_per_kg_[index]})
        {
            if (!std::isfinite(value))
                return reject(RimeSplinteringFailure::IntervalRejected,
                    "rime splintering aggregate transfer is unrepresentable", error, failure);
        }
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_rime_splintering_cell(
    const ColdParameters& parameters, const RimeSplinteringCell& cell,
    double duration_s, RimeSplinteringResult& result, std::string& error,
    std::size_t capacity, RimeSplinteringFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    RimeSplinteringProducer producer;
    if (!produce_exact_rime_splintering_admission(parameters, cell, duration_s,
            producer, error, capacity, failure, spectrum_domain))
        return false;
    return apply_endpoint(producer, result, error, failure);
}

bool apply_exact_triple_rime_splintering_cell(
    const ColdParameters& parameters, const RimeSplinteringCell& cell,
    double duration_s, const RimeSplinteringProducer& producer,
    RimeSplinteringResult& result, std::string& error,
    std::size_t capacity, RimeSplinteringFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (!valid_cold_spectrum_domain(spectrum_domain))
    {
        return reject(RimeSplinteringFailure::InvalidContext,
            "rime splintering spectrum domain is invalid", error, failure);
    }
    if (!producer.valid_ || !same_source(producer.source(), cell) ||
        !same_values(producer.parameters(), parameter_values(parameters)) ||
        producer.hail_processes() != parameters.hail_processes || producer.spectrum_domain() != spectrum_domain ||
        !same_value(producer.duration_s(), duration_s))
        return reject(RimeSplinteringFailure::InvalidProducer,
            "rime splintering capture differs from its exact source or context", error, failure);
    RimeSplinteringProducer expected;
    if (!produce_exact_rime_splintering_admission(parameters, cell, duration_s,
            expected, error, capacity, failure, spectrum_domain))
        return false;
    if (!same_capture(producer, expected))
        return reject(RimeSplinteringFailure::InvalidProducer,
            "rime splintering capture differs from its original cloud journal or B4 transfer", error, failure);
    return apply_endpoint(expected, result, error, failure);
}

} // namespace tmv::milbrandt_detail
