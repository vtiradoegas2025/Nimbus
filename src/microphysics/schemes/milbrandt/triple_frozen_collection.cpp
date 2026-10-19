#include "triple_frozen_collection.hpp"
#include "triple_rain_collection.hpp"
#include "triple_rain_collision.hpp"
#include "triple_hail_wet_growth.hpp"

#include "cold.hpp"
#include "triple_sedimentation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr std::array<TripleMomentSpecies, 4> species{
    TripleMomentSpecies::Ice, TripleMomentSpecies::Snow,
    TripleMomentSpecies::Graupel, TripleMomentSpecies::Hail};

constexpr std::array<FrozenCollectionChannel, 5> channels{
    FrozenCollectionChannel::IceToSnow,
    FrozenCollectionChannel::IceToGraupel,
    FrozenCollectionChannel::IceToHail,
    FrozenCollectionChannel::SnowToHail,
    FrozenCollectionChannel::SnowSelfCollection};

struct Spectrum
{
    double alpha = 0.0;
    double lambda = 0.0;
    double speed = 0.0;
    double mass_order = 0.0;
    double normalized_mass = 0.0;
};

bool reject(FrozenCollectionFailure kind, const char* message,
            std::string& error, FrozenCollectionFailure* failure)
{
    error = message;
    if (failure)
    {
        *failure = kind;
    }
    return false;
}

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
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

bool same_parameters(const std::array<double, 25>& a,
                     const std::array<double, 25>& b) noexcept
{
    for (std::size_t index = 0; index < a.size(); ++index)
    {
        if (!same_value(a[index], b[index]))
        {
            return false;
        }
    }
    return true;
}

bool same_source(const FrozenCollectionCell& a,
                 const FrozenCollectionCell& b) noexcept
{
    if (!same_value(a.temperature_k, b.temperature_k) ||
        !same_value(a.pressure_pa, b.pressure_pa) ||
        !same_value(a.dry_density_kg_m3, b.dry_density_kg_m3))
    {
        return false;
    }
    for (std::size_t index = 0; index < a.initial.frozen.size(); ++index)
    {
        const auto& x = a.initial.frozen[index];
        const auto& y = b.initial.frozen[index];
        if (!same_value(x.mass_per_kg, y.mass_per_kg) ||
            !same_value(x.number_per_kg, y.number_per_kg) ||
            !same_value(x.reflectivity_per_kg, y.reflectivity_per_kg))
        {
            return false;
        }
    }
    return true;
}

double cubic_log_shape_ratio(double alpha) noexcept
{
    return std::log1p(3.0 / (alpha + 1.0)) +
        std::log1p(3.0 / (alpha + 2.0)) +
        std::log1p(3.0 / (alpha + 3.0));
}

bool refine_cubic(const FrozenCollectionPopulation& source,
                  double coefficient, double& alpha, double& lambda) noexcept
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
    if (!positive_finite(log_ratio))
    {
        return false;
    }
    double lower = -1.0 + 1.0e-9;
    double upper = 1.0;
    if (!(log_ratio < cubic_log_shape_ratio(lower)))
    {
        return false;
    }
    while (cubic_log_shape_ratio(upper) > log_ratio && upper < 1.0e6)
    {
        upper *= 2.0;
    }
    if (upper > 1.0e6 || cubic_log_shape_ratio(upper) > log_ratio)
    {
        return false;
    }
    for (int iteration = 0; iteration < 80; ++iteration)
    {
        const double middle = 0.5 * (lower + upper);
        if (cubic_log_shape_ratio(middle) > log_ratio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    alpha = 0.5 * (lower + upper);
    lambda = std::exp((std::log(coefficient) + std::log(source.number_per_kg) +
        std::log(alpha + 1.0) + std::log(alpha + 2.0) +
        std::log(alpha + 3.0) - std::log(source.mass_per_kg)) / 3.0);
    return std::isfinite(alpha) && positive_finite(lambda);
}

bool population_spectrum(const FrozenCollectionPopulation& source,
                         TripleMomentSpecies category, Spectrum& spectrum,
                         std::string& error, ColdSpectrumDomain spectrum_domain)
{
    for (double value : {source.mass_per_kg, source.number_per_kg,
                         source.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "frozen collection q/N/Z is nonfinite or negative";
            return false;
        }
    }
    if (source.mass_per_kg == 0.0 && source.number_per_kg == 0.0 &&
        source.reflectivity_per_kg == 0.0)
    {
        spectrum = {};
        return true;
    }
    const auto law = current_nimbus_triple_sedimentation_parameters(category).mass_law;
    Spectrum candidate;
    if (!diagnose_triple_gamma_shape(source.mass_per_kg, source.number_per_kg,
            source.reflectivity_per_kg, law, candidate.alpha, candidate.lambda,
            error))
    {
        return false;
    }
    if (law.exponent == 3.0 &&
        !refine_cubic(source, law.coefficient, candidate.alpha, candidate.lambda))
    {
        error = "frozen collection cubic spectrum is outside its admitted bracket";
        return false;
    }
    if (candidate.alpha > cold_spectrum_shape_limit(spectrum_domain, category))
    {
        error = "frozen collection gamma shape exceeds its selected spectrum domain";
        return false;
    }
    candidate.mass_order = law.exponent;
    candidate.normalized_mass = source.mass_per_kg / source.number_per_kg / law.coefficient;
    if (!positive_finite(candidate.normalized_mass))
    {
        error = "frozen collection normalized mass moment is unrepresentable";
        return false;
    }
    spectrum = candidate;
    return true;
}

bool moment(const Spectrum& spectrum, double order, double& result) noexcept
{
    double candidate = 1.0;
    if (order == spectrum.mass_order)
    {
        candidate = spectrum.normalized_mass;
    }
    else if (order == spectrum.mass_order + 1.0)
    {
        candidate = spectrum.normalized_mass *
            ((spectrum.alpha + spectrum.mass_order + 1.0) / spectrum.lambda);
    }
    else if (order == spectrum.mass_order + 2.0)
    {
        candidate = spectrum.normalized_mass *
            ((spectrum.alpha + spectrum.mass_order + 1.0) / spectrum.lambda) *
            ((spectrum.alpha + spectrum.mass_order + 2.0) / spectrum.lambda);
    }
    else if (order == std::floor(order))
    {
        for (int factor = 1; factor <= static_cast<int>(order); ++factor)
        {
            candidate *= (spectrum.alpha + factor) / spectrum.lambda;
        }
    }
    else
    {
        candidate = std::exp(std::lgamma(spectrum.alpha + order + 1.0) -
            std::lgamma(spectrum.alpha + 1.0) - order * std::log(spectrum.lambda));
    }
    if (!positive_finite(candidate))
    {
        return false;
    }
    result = candidate;
    return true;
}

bool capture_pair(const FrozenCollectionCell& cell, double duration,
                  const std::array<Spectrum, 4>& spectra,
                  std::size_t donor, std::size_t collector, double efficiency,
                  FrozenCollectionAcceptedChannel& capture)
{
    const auto& y = cell.initial.frozen[donor];
    const auto& x = cell.initial.frozen[collector];
    if (!(y.mass_per_kg > 1.0e-14) || !(x.mass_per_kg > 1.0e-14))
    {
        return true;
    }
    const auto law = current_nimbus_triple_sedimentation_parameters(species[donor]).mass_law;
    std::array<double, 2> x_moments{};
    std::array<double, 5> y_moments{};
    if (!moment(spectra[collector], 1.0, x_moments[0]) ||
        !moment(spectra[collector], 2.0, x_moments[1]) ||
        !moment(spectra[donor], 1.0, y_moments[0]) ||
        !moment(spectra[donor], 2.0, y_moments[1]) ||
        !moment(spectra[donor], law.exponent, y_moments[2]) ||
        !moment(spectra[donor], law.exponent + 1.0, y_moments[3]) ||
        !moment(spectra[donor], law.exponent + 2.0, y_moments[4]))
    {
        return false;
    }
    const double difference = spectra[collector].speed - spectra[donor].speed;
    const double relative_speed = std::sqrt(difference * difference +
        0.04 * spectra[collector].speed * spectra[donor].speed);
    constexpr double pi_over_four = 0.785398163397448309616;
    const double common = duration * pi_over_four * efficiency * relative_speed *
        cell.dry_density_kg_m3 * x.number_per_kg * y.number_per_kg;
    capture.raw_mass_increment_per_kg = common * law.coefficient *
        (x_moments[1] * y_moments[2] +
         2.0 * x_moments[0] * y_moments[3] + y_moments[4]);
    capture.raw_number_increment_per_kg = common *
        (x_moments[1] + 2.0 * x_moments[0] * y_moments[0] + y_moments[1]);
    return std::isfinite(capture.raw_mass_increment_per_kg) &&
        capture.raw_mass_increment_per_kg >= 0.0 &&
        std::isfinite(capture.raw_number_increment_per_kg) &&
        capture.raw_number_increment_per_kg >= 0.0;
}

double admission_factor(double mass, double number,
                        double mass_sink, double number_sink) noexcept
{
    double factor = 1.0;
    if (mass_sink > 0.0)
    {
        factor = std::min(factor, mass / mass_sink);
    }
    if (number_sink > 0.0)
    {
        factor = std::min(factor, number / number_sink);
    }
    return factor;
}

void admit(FrozenCollectionAcceptedChannel& channel, double factor) noexcept
{
    channel.donor_admission_factor = factor;
    channel.admitted_mass_increment_per_kg = channel.raw_mass_increment_per_kg * factor;
    channel.admitted_number_increment_per_kg = channel.raw_number_increment_per_kg * factor;
}

bool stored_population(const FrozenCollectionPopulation& source,
                       TripleMomentSpecies category,
                       FrozenCollectionEndpoint& endpoint, std::string& error,
                       ColdSpectrumDomain spectrum_domain)
{
    for (double value : {source.mass_per_kg, source.number_per_kg,
                         source.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0 ||
            value > std::numeric_limits<float>::max() ||
            (value > 0.0 && value < std::numeric_limits<float>::denorm_min()))
        {
            error = "frozen collection final q/N/Z is not FP32 representable";
            return false;
        }
    }
    const FrozenCollectionEndpoint candidate{
        static_cast<float>(source.mass_per_kg),
        static_cast<float>(source.number_per_kg),
        static_cast<float>(source.reflectivity_per_kg)};
    Spectrum spectrum;
    if (!population_spectrum({candidate.mass_per_kg, candidate.number_per_kg,
            candidate.reflectivity_per_kg}, category, spectrum, error, spectrum_domain))
    {
        return false;
    }
    endpoint = candidate;
    return true;
}

bool source_increments(const FrozenCollectionProducer& producer,
                       FrozenCollectionIncrements& increments, std::string& error,
                       FrozenCollectionFailure* failure)
{
    const auto& initial = producer.source().initial;
    const auto& events = producer.channels();
    FrozenCollectionIncrements candidate;
    for (std::size_t event = 0; event < 3; ++event)
    {
        const double mass = events[event].admitted_mass_increment_per_kg;
        const double number = events[event].admitted_number_increment_per_kg;
        candidate.mass_increment_per_kg[0] -= mass;
        candidate.number_increment_per_kg[0] -= number;
        candidate.mass_increment_per_kg[event + 1] += mass;
    }
    candidate.mass_increment_per_kg[1] -= events[3].admitted_mass_increment_per_kg;
    candidate.mass_increment_per_kg[3] += events[3].admitted_mass_increment_per_kg;
    candidate.number_increment_per_kg[1] -=
        events[3].admitted_number_increment_per_kg +
        events[4].admitted_number_increment_per_kg;
    for (std::size_t index = 0; index < initial.frozen.size(); ++index)
    {
        const auto& source = initial.frozen[index];
        const double exponent = current_nimbus_triple_sedimentation_parameters(species[index]).mass_law.exponent;
        if (source.mass_per_kg > 0.0)
        {
            const double power = 6.0 / exponent;
            candidate.reflectivity_increment_per_kg[index] = source.reflectivity_per_kg *
                (power * candidate.mass_increment_per_kg[index] / source.mass_per_kg +
                 (1.0 - power) * candidate.number_increment_per_kg[index] / source.number_per_kg);
        }
        for (double value : {candidate.mass_increment_per_kg[index], candidate.number_increment_per_kg[index],
                             candidate.reflectivity_increment_per_kg[index]})
        {
            if (!std::isfinite(value))
            {
                return reject(FrozenCollectionFailure::IntervalRejected,
                    "frozen collection source-moment increment is unrepresentable", error, failure);
            }
        }
    }
    increments = candidate;
    error.clear();
    if (failure)
    {
        *failure = FrozenCollectionFailure::None;
    }
    return true;
}

bool phase_endpoint(const FrozenCollectionProducer& producer,
                    FrozenCollectionResult& result, std::string& error,
                    FrozenCollectionFailure* failure)
{
    FrozenCollectionIncrements increments;
    if (!source_increments(producer, increments, error, failure))
    {
        return false;
    }
    const auto& initial = producer.source().initial;
    FrozenCollectionResult candidate;
    candidate.mass_increment_per_kg = increments.mass_increment_per_kg;
    candidate.number_increment_per_kg = increments.number_increment_per_kg;
    candidate.reflectivity_increment_per_kg = increments.reflectivity_increment_per_kg;
    for (std::size_t index = 0; index < initial.frozen.size(); ++index)
    {
        const auto& source = initial.frozen[index];
        auto& endpoint = candidate.unrounded.frozen[index];
        endpoint = {source.mass_per_kg + candidate.mass_increment_per_kg[index],
            source.number_per_kg + candidate.number_increment_per_kg[index],
            source.reflectivity_per_kg + candidate.reflectivity_increment_per_kg[index]};
        Spectrum spectrum;
        if (!population_spectrum(endpoint, species[index], spectrum, error, producer.spectrum_domain()) ||
            !stored_population(endpoint, species[index], candidate.rounded[index], error, producer.spectrum_domain()))
        {
            if (failure)
            {
                *failure = FrozenCollectionFailure::IntervalRejected;
            }
            return false;
        }
    }
    long double water_change = 0.0L;
    long double water_scale = 0.0L;
    for (std::size_t index = 0; index < initial.frozen.size(); ++index)
    {
        water_change += static_cast<long double>(candidate.unrounded.frozen[index].mass_per_kg) -
            initial.frozen[index].mass_per_kg;
        water_scale += initial.frozen[index].mass_per_kg;
    }
    if (std::abs(water_change) >
        64.0L * std::numeric_limits<double>::epsilon() * water_scale)
    {
        return reject(FrozenCollectionFailure::IntervalRejected,
            "frozen collection interval violates its exact water budget", error, failure);
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure)
    {
        *failure = FrozenCollectionFailure::None;
    }
    return true;
}

bool same_channels(const FrozenCollectionProducer& a,
                   const FrozenCollectionProducer& b) noexcept
{
    if (a.event_count() != b.event_count() ||
        !same_parameters(a.parameters(), b.parameters()) ||
        a.hail_processes() != b.hail_processes() ||
        a.wet_hail_collection() != b.wet_hail_collection() ||
        a.spectrum_domain() != b.spectrum_domain() ||
        !same_value(a.duration_s(), b.duration_s()) ||
        !same_source(a.source(), b.source()))
    {
        return false;
    }
    for (std::size_t index = 0; index < a.channels().size(); ++index)
    {
        const auto& x = a.channels()[index];
        const auto& y = b.channels()[index];
        if (x.channel != y.channel ||
            !same_value(x.raw_mass_increment_per_kg, y.raw_mass_increment_per_kg) ||
            !same_value(x.raw_number_increment_per_kg, y.raw_number_increment_per_kg) ||
            !same_value(x.admitted_mass_increment_per_kg, y.admitted_mass_increment_per_kg) ||
            !same_value(x.admitted_number_increment_per_kg, y.admitted_number_increment_per_kg) ||
            !same_value(x.donor_admission_factor, y.donor_admission_factor))
        {
            return false;
        }
    }
    for (std::size_t index = 0; index < a.source_alpha().size(); ++index)
    {
        if (!same_value(a.source_alpha()[index], b.source_alpha()[index]) ||
            !same_value(a.source_lambda_per_m()[index], b.source_lambda_per_m()[index]) ||
            !same_value(a.mass_weighted_speed_m_s()[index], b.mass_weighted_speed_m_s()[index]))
        {
            return false;
        }
    }
    return true;
}

} // namespace

bool produce_exact_frozen_collection_admission(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, FrozenCollectionProducer& producer, std::string& error,
    std::size_t capacity, FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    if (failure)
    {
        *failure = FrozenCollectionFailure::None;
    }
    if (!valid_cold_spectrum_domain(spectrum_domain) || !std::isfinite(duration_s) || duration_s <= 0.0 ||
        !std::isfinite(cell.temperature_k) || cell.temperature_k < 150.0 ||
        cell.temperature_k >= microphysics_constants::T0 ||
        !std::isfinite(cell.pressure_pa) || cell.pressure_pa < 100.0 ||
        cell.pressure_pa > 200000.0 || !positive_finite(cell.dry_density_kg_m3) ||
        parameter_values(parameters) != default_cold_parameters)
    {
        return reject(FrozenCollectionFailure::InvalidContext,
            "frozen collection context or retained physical parameters differ", error, failure);
    }
    if (capacity != channels.size())
    {
        return reject(FrozenCollectionFailure::Capacity,
            "frozen collection requires exactly five bounded channel records", error, failure);
    }
    FrozenCollectionProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameter_values(parameters);
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.wet_hail_collection_ = wet_hail_collection;
    candidate.spectrum_domain_ = spectrum_domain;
    candidate.duration_s_ = duration_s;
    std::array<Spectrum, 4> spectra{};
    const double density_factor = std::sqrt(1.225 / cell.dry_density_kg_m3);
    if (!positive_finite(density_factor))
    {
        return reject(FrozenCollectionFailure::InvalidContext,
            "frozen collection air-density correction is unrepresentable", error, failure);
    }
    for (std::size_t index = 0; index < cell.initial.frozen.size(); ++index)
    {
        if (!population_spectrum(cell.initial.frozen[index], species[index], spectra[index], error, spectrum_domain))
        {
            if (failure)
            {
                *failure = FrozenCollectionFailure::InvalidSource;
            }
            return false;
        }
        if (cell.initial.frozen[index].mass_per_kg > 0.0)
        {
            const auto defaults = current_nimbus_triple_sedimentation_parameters(species[index]);
            double mass_velocity_moment = 0.0;
            if (!moment(spectra[index], defaults.mass_law.exponent + defaults.fall_exponent,
                    mass_velocity_moment))
            {
                return reject(FrozenCollectionFailure::InvalidSource,
                    "frozen collection mass-weighted velocity moment is unrepresentable", error, failure);
            }
            spectra[index].speed = density_factor * defaults.fall_coefficient *
                mass_velocity_moment / spectra[index].normalized_mass;
            if (!positive_finite(spectra[index].speed))
            {
                return reject(FrozenCollectionFailure::InvalidSource,
                    "frozen collection mass-weighted velocity is unrepresentable", error, failure);
            }
        }
        candidate.source_alpha_[index] = spectra[index].alpha;
        candidate.source_lambda_per_m_[index] = spectra[index].lambda;
        candidate.mass_weighted_speed_m_s_[index] = spectra[index].speed;
    }
    const double temperature_c = cell.temperature_k - microphysics_constants::T0;
    const double ice_snow_efficiency = std::min(1.0, 0.05 * std::exp(0.1 * temperature_c));
    const double other_efficiency = std::min(1.0, 0.01 * std::exp(0.1 * temperature_c));
    for (std::size_t index = 0; index < candidate.channels_.size(); ++index)
    {
        candidate.channels_[index].channel = channels[index];
    }
    for (std::size_t index = 0; index < 3; ++index)
    {
        if (!capture_pair(cell, duration_s, spectra, 0, index + 1,
                index == 0 ? ice_snow_efficiency : index == 2 && wet_hail_collection ? 1.0 : other_efficiency,
                candidate.channels_[index]))
        {
            return reject(FrozenCollectionFailure::IntervalRejected,
                "frozen collection uncapped ice interval is unrepresentable", error, failure);
        }
    }
    if (!capture_pair(cell, duration_s, spectra, 1, 3, wet_hail_collection ? 1.0 : other_efficiency,
            candidate.channels_[3]))
    {
        return reject(FrozenCollectionFailure::IntervalRejected,
            "frozen collection uncapped snow interval is unrepresentable", error, failure);
    }
    const auto& snow = cell.initial.frozen[1];
    if (snow.mass_per_kg > 1.0e-14)
    {
        // The retained aggregation closure uses cubic scaling even for non-cubic snow.
        const double number = std::min(0.5 * cell.dry_density_kg_m3 * snow.number_per_kg,
            duration_s * 0.93952 * ice_snow_efficiency *
                std::pow(cell.dry_density_kg_m3 * snow.mass_per_kg,
                    (2.0 + parameters.b_s_) / 3.0) *
                std::pow(cell.dry_density_kg_m3 * snow.number_per_kg,
                    (4.0 - parameters.b_s_) / 3.0)) / cell.dry_density_kg_m3;
        if (!std::isfinite(number) || number < 0.0)
        {
            return reject(FrozenCollectionFailure::IntervalRejected,
                "frozen collection snow-aggregation interval is unrepresentable", error, failure);
        }
        candidate.channels_[4].raw_number_increment_per_kg = number;
    }
    const auto& ice = cell.initial.frozen[0];
    const double ice_mass_sink = candidate.channels_[0].raw_mass_increment_per_kg +
        candidate.channels_[1].raw_mass_increment_per_kg + candidate.channels_[2].raw_mass_increment_per_kg;
    const double ice_number_sink = candidate.channels_[0].raw_number_increment_per_kg +
        candidate.channels_[1].raw_number_increment_per_kg + candidate.channels_[2].raw_number_increment_per_kg;
    if (!std::isfinite(ice_mass_sink) || !std::isfinite(ice_number_sink))
    {
        return reject(FrozenCollectionFailure::IntervalRejected,
            "frozen collection aggregate ice interval is unrepresentable", error, failure);
    }
    const double ice_factor = admission_factor(ice.mass_per_kg, ice.number_per_kg,
        ice_mass_sink, ice_number_sink);
    for (std::size_t index = 0; index < 3; ++index)
    {
        admit(candidate.channels_[index], ice_factor);
    }
    const double snow_mass_available = snow.mass_per_kg + (wet_hail_collection ? 0.0 : candidate.channels_[0].admitted_mass_increment_per_kg);
    const double snow_number_sink = candidate.channels_[3].raw_number_increment_per_kg +
        candidate.channels_[4].raw_number_increment_per_kg;
    if (!std::isfinite(snow_mass_available) || !std::isfinite(snow_number_sink))
    {
        return reject(FrozenCollectionFailure::IntervalRejected,
            "frozen collection aggregate snow interval is unrepresentable", error, failure);
    }
    const double snow_factor = admission_factor(snow_mass_available, snow.number_per_kg,
        candidate.channels_[3].raw_mass_increment_per_kg, snow_number_sink);
    admit(candidate.channels_[3], snow_factor);
    admit(candidate.channels_[4], snow_factor);
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_frozen_collection_cell(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, FrozenCollectionResult& result, std::string& error,
    std::size_t capacity, FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    FrozenCollectionProducer producer;
    if (!produce_exact_frozen_collection_admission(parameters, cell, duration_s,
            producer, error, capacity, failure, spectrum_domain, wet_hail_collection))
    {
        return false;
    }
    return phase_endpoint(producer, result, error, failure);
}

bool validate_exact_frozen_collection_admission(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, const FrozenCollectionProducer& producer, std::string& error,
    std::size_t capacity, FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    if (producer.event_count() != channels.size() || !same_source(producer.source(), cell) ||
        !same_parameters(producer.parameters(), parameter_values(parameters)) ||
        producer.hail_processes() != parameters.hail_processes ||
        producer.spectrum_domain() != spectrum_domain ||
        producer.wet_hail_collection() != wet_hail_collection ||
        !same_value(producer.duration_s(), duration_s))
    {
        return reject(FrozenCollectionFailure::InvalidProducer,
            "frozen collection capture differs from its exact source or context", error, failure);
    }
    FrozenCollectionProducer expected;
    if (!produce_exact_frozen_collection_admission(parameters, cell, duration_s,
            expected, error, capacity, failure, spectrum_domain, wet_hail_collection))
    {
        return false;
    }
    if (!same_channels(producer, expected))
    {
        return reject(FrozenCollectionFailure::InvalidProducer,
            "frozen collection capture differs from its exact source, context or admission", error, failure);
    }
    error.clear();
    if (failure)
    {
        *failure = FrozenCollectionFailure::None;
    }
    return true;
}

bool evaluate_exact_frozen_collection_increments(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, const FrozenCollectionProducer& producer,
    FrozenCollectionIncrements& increments, std::string& error,
    std::size_t capacity, FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    return validate_exact_frozen_collection_admission(parameters, cell, duration_s, producer,
        error, capacity, failure, spectrum_domain, wet_hail_collection) && source_increments(producer, increments, error, failure);
}

bool apply_exact_triple_frozen_collection_cell(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, const FrozenCollectionProducer& producer,
    FrozenCollectionResult& result, std::string& error,
    std::size_t capacity, FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    return validate_exact_frozen_collection_admission(parameters, cell, duration_s, producer,
        error, capacity, failure, spectrum_domain, wet_hail_collection) && phase_endpoint(producer, result, error, failure);
}

bool produce_exact_rain_collection_admission(
    const ColdParameters& parameters, const RainCollectionCell& cell, double duration_s,
    RainCollectionProducer& producer, std::string& error, std::size_t capacity,
    FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    const auto& context = cell.frozen;
    if (!valid_cold_spectrum_domain(spectrum_domain) || !positive_finite(duration_s) ||
        !std::isfinite(context.temperature_k) || context.temperature_k < 150.0 ||
        context.temperature_k >= microphysics_constants::T0 ||
        !std::isfinite(context.pressure_pa) || context.pressure_pa < 100.0 || context.pressure_pa > 200000.0 ||
        !positive_finite(context.dry_density_kg_m3) || parameter_values(parameters) != default_cold_parameters)
    {
        return reject(FrozenCollectionFailure::InvalidContext,
            "rain collection context or retained physical parameters differ", error, failure);
    }
    if (capacity != 4)
    {
        return reject(FrozenCollectionFailure::Capacity,
            "rain collection requires four bounded channel records", error, failure);
    }
    RainCollectionProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameter_values(parameters);
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.spectrum_domain_ = spectrum_domain;
    candidate.duration_s_ = duration_s;
    std::array<Spectrum, 5> spectra{};
    const double density_factor = std::sqrt(1.225 / context.dry_density_kg_m3);
    if (!positive_finite(density_factor))
    {
        return reject(FrozenCollectionFailure::InvalidContext,
            "rain collection density correction is unrepresentable", error, failure);
    }
    for (std::size_t index = 0; index < spectra.size(); ++index)
    {
        const auto category = index == 0 ? TripleMomentSpecies::Rain : species[index - 1];
        const auto& population = index == 0 ? cell.rain : context.initial.frozen[index - 1];
        auto& spectrum = spectra[index];
        if (!population_spectrum(population, category, spectrum, error, spectrum_domain))
        {
            if (failure) *failure = FrozenCollectionFailure::InvalidSource;
            return false;
        }
        if (population.mass_per_kg > 0.0)
        {
            const auto law = current_nimbus_triple_sedimentation_parameters(category);
            double velocity_moment = 0.0;
            if (!moment(spectrum, law.mass_law.exponent + law.fall_exponent, velocity_moment))
            {
                return reject(FrozenCollectionFailure::InvalidSource,
                    "rain collection velocity moment is unrepresentable", error, failure);
            }
            spectrum.speed = density_factor * law.fall_coefficient * velocity_moment / spectrum.normalized_mass;
            if (!positive_finite(spectrum.speed))
            {
                return reject(FrozenCollectionFailure::InvalidSource,
                    "rain collection velocity is unrepresentable", error, failure);
            }
        }
        candidate.source_alpha_[index] = spectrum.alpha;
        candidate.source_lambda_per_m_[index] = spectrum.lambda;
        candidate.mass_weighted_speed_m_s_[index] = spectrum.speed;
    }
    double mass_sink = 0.0;
    double number_sink = 0.0;
    for (std::size_t index = 0; index < candidate.channels_.size(); ++index)
    {
        const auto& collector = context.initial.frozen[index];
        if (cell.rain.mass_per_kg <= 1e-14 || collector.mass_per_kg <= 1e-14) continue;
        std::array<double, 5> rain_moments{};
        double first = 0.0, second = 0.0;
        for (std::size_t order = 0; order < rain_moments.size(); ++order)
        {
            if (!moment(spectra[0], static_cast<double>(order + 1), rain_moments[order]))
            {
                return reject(FrozenCollectionFailure::IntervalRejected,
                    "rain collection donor moment is unrepresentable", error, failure);
            }
        }
        if (!moment(spectra[index + 1], 1.0, first) || !moment(spectra[index + 1], 2.0, second))
        {
            return reject(FrozenCollectionFailure::IntervalRejected,
                "rain collection collector moment is unrepresentable", error, failure);
        }
        const double difference = spectra[0].speed - spectra[index + 1].speed;
        const double speed = std::sqrt(difference * difference + .04 * spectra[0].speed * spectra[index + 1].speed);
        const double common = duration_s * .785398163397448309616 * speed * context.dry_density_kg_m3 *
            cell.rain.number_per_kg * collector.number_per_kg;
        auto& channel = candidate.channels_[index];
        channel.raw_mass_per_kg = common * parameters.c_r_ *
            (second * rain_moments[2] + 2.0 * first * rain_moments[3] + rain_moments[4]);
        channel.raw_number_per_kg = common * (second + 2.0 * first * rain_moments[0] + rain_moments[1]);
        if (!positive_finite(channel.raw_mass_per_kg) || !positive_finite(channel.raw_number_per_kg))
        {
            return reject(FrozenCollectionFailure::IntervalRejected,
                "rain collection active transfer is unrepresentable", error, failure);
        }
        mass_sink += channel.raw_mass_per_kg;
        number_sink += channel.raw_number_per_kg;
    }
    if (!std::isfinite(mass_sink) || !std::isfinite(number_sink) || mass_sink < 0.0 || number_sink < 0.0)
    {
        return reject(FrozenCollectionFailure::IntervalRejected,
            "rain collection aggregate original-donor budget is unrepresentable", error, failure);
    }
    candidate.admission_factor_ = admission_factor(cell.rain.mass_per_kg, cell.rain.number_per_kg, mass_sink, number_sink);
    if (mass_sink > cell.rain.mass_per_kg || number_sink > cell.rain.number_per_kg)
    {
        candidate.admission_factor_ *= cold_snow_riming_overdraw_factor;
    }
    double accepted_mass = 0.0, accepted_number = 0.0;
    for (auto& channel : candidate.channels_)
    {
        channel.mass_per_kg = candidate.admission_factor_ * channel.raw_mass_per_kg;
        channel.number_per_kg = candidate.admission_factor_ * channel.raw_number_per_kg;
        if ((channel.raw_mass_per_kg > 0.0 && candidate.admission_factor_ > 0.0 && channel.mass_per_kg == 0.0) ||
            (channel.raw_number_per_kg > 0.0 && candidate.admission_factor_ > 0.0 && channel.number_per_kg == 0.0))
        {
            return reject(FrozenCollectionFailure::IntervalRejected,
                "rain collection admitted transfer underflows", error, failure);
        }
        accepted_mass += channel.mass_per_kg;
        accepted_number += channel.number_per_kg;
    }
    if (accepted_mass > cell.rain.mass_per_kg || accepted_number > cell.rain.number_per_kg)
    {
        return reject(FrozenCollectionFailure::IntervalRejected,
            "rain collection paired original-donor admission did not close", error, failure);
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

bool validate_exact_rain_collection_admission(
    const ColdParameters& parameters, const RainCollectionCell& cell, double duration_s,
    const RainCollectionProducer& producer, std::string& error, std::size_t capacity,
    FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    const auto& rain = producer.source().rain;
    if (producer.event_count() != 4 || !same_source(producer.source().frozen, cell.frozen) ||
        !same_value(rain.mass_per_kg, cell.rain.mass_per_kg) ||
        !same_value(rain.number_per_kg, cell.rain.number_per_kg) ||
        !same_value(rain.reflectivity_per_kg, cell.rain.reflectivity_per_kg) ||
        !same_parameters(producer.parameters(), parameter_values(parameters)) ||
        producer.hail_processes() != parameters.hail_processes || producer.spectrum_domain() != spectrum_domain ||
        !same_value(producer.duration_s(), duration_s))
    {
        return reject(FrozenCollectionFailure::InvalidProducer,
            "rain collection journal differs from its original source, interval or policy", error, failure);
    }
    RainCollectionProducer expected;
    if (!produce_exact_rain_collection_admission(parameters, cell, duration_s, expected, error,
            capacity, failure, spectrum_domain)) return false;
    for (std::size_t index = 0; index < 4; ++index)
    {
        const auto& a = producer.channels()[index];
        const auto& b = expected.channels()[index];
        if (!same_value(a.raw_mass_per_kg, b.raw_mass_per_kg) || !same_value(a.raw_number_per_kg, b.raw_number_per_kg) ||
            !same_value(a.mass_per_kg, b.mass_per_kg) || !same_value(a.number_per_kg, b.number_per_kg))
        {
            return reject(FrozenCollectionFailure::InvalidProducer,
                "rain collection journal differs from regenerated original-donor admission", error, failure);
        }
    }
    if (!same_value(producer.admission_factor(), expected.admission_factor()) ||
        producer.source_alpha() != expected.source_alpha() ||
        producer.source_lambda_per_m() != expected.source_lambda_per_m() ||
        producer.mass_weighted_speed_m_s() != expected.mass_weighted_speed_m_s())
    {
        return reject(FrozenCollectionFailure::InvalidProducer,
            "rain collection captured spectrum differs", error, failure);
    }
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

bool produce_exact_rain_collision_admission(
    const ColdParameters& parameters, const RainCollectionCell& cell, double duration_s,
    RainCollisionProducer& producer, std::string& error, std::size_t capacity,
    FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (capacity != 7)
        return reject(FrozenCollectionFailure::Capacity,
            "rain collisions require four rain and three frozen collection records", error, failure);
    RainCollisionProducer candidate;
    if (!produce_exact_rain_collection_admission(parameters, cell, duration_s, candidate.rain_source_,
            error, 4, failure, spectrum_domain)) return false;
    const auto& source = candidate.rain_source_;
    Spectrum rain;
    if (!population_spectrum(cell.rain, TripleMomentSpecies::Rain, rain, error, spectrum_domain)) return false;
    double rain_first = 0.0, rain_second = 0.0;
    if (cell.rain.mass_per_kg > 1e-14 && (!moment(rain, 1.0, rain_first) || !moment(rain, 2.0, rain_second)))
        return reject(FrozenCollectionFailure::InvalidSource,
            "rain collision collector spectrum is unrepresentable", error, failure);
    for (std::size_t index = 0; index < 3; ++index)
    {
        auto& channel = candidate.channels_[index];
        channel.rain = source.channels()[index];
        const auto& frozen = cell.frozen.initial.frozen[index];
        if (channel.rain.raw_mass_per_kg == 0.0) continue;
        Spectrum spectrum;
        if (!population_spectrum(frozen, species[index], spectrum, error, spectrum_domain)) return false;
        const auto law = current_nimbus_triple_sedimentation_parameters(species[index]).mass_law;
        std::array<double, 3> moments{};
        for (std::size_t order = 0; order < moments.size(); ++order)
            if (!moment(spectrum, law.exponent + order, moments[order]))
                return reject(FrozenCollectionFailure::InvalidSource,
                    "rain collision frozen donor moment is unrepresentable", error, failure);
        const auto& speeds = source.mass_weighted_speed_m_s();
        const double difference = speeds[0] - speeds[index + 1];
        const double relative = std::sqrt(difference * difference + .04 * speeds[0] * speeds[index + 1]);
        const double common = duration_s * .785398163397448309616 * relative * cell.frozen.dry_density_kg_m3 *
            cell.rain.number_per_kg * frozen.number_per_kg;
        channel.frozen.raw_mass_per_kg = common * law.coefficient *
            (rain_second * moments[0] + 2.0 * rain_first * moments[1] + moments[2]);
        // Both sides count the same collision events; retain the captured evaluation order.
        channel.frozen.raw_number_per_kg = channel.rain.raw_number_per_kg;
        if (!positive_finite(channel.frozen.raw_mass_per_kg))
            return reject(FrozenCollectionFailure::IntervalRejected,
                "rain collision frozen mass transfer is unrepresentable", error, failure);
        double frozen_factor = admission_factor(frozen.mass_per_kg, frozen.number_per_kg,
            channel.frozen.raw_mass_per_kg, channel.frozen.raw_number_per_kg);
        if (channel.frozen.raw_mass_per_kg > frozen.mass_per_kg || channel.frozen.raw_number_per_kg > frozen.number_per_kg)
            frozen_factor *= cold_snow_riming_overdraw_factor;
        channel.admission_factor = std::min(source.admission_factor(), frozen_factor);
        channel.rain.mass_per_kg = channel.rain.raw_mass_per_kg * channel.admission_factor;
        channel.rain.number_per_kg = channel.rain.raw_number_per_kg * channel.admission_factor;
        channel.frozen.mass_per_kg = channel.frozen.raw_mass_per_kg * channel.admission_factor;
        channel.frozen.number_per_kg = channel.frozen.raw_number_per_kg * channel.admission_factor;
        channel.rain_mean_mass_diameter_m = std::pow(cell.frozen.dry_density_kg_m3 * cell.rain.mass_per_kg /
            (parameters.c_r_ * (cell.frozen.dry_density_kg_m3 * cell.rain.number_per_kg)), 1.0 / parameters.d_r_);
        channel.frozen_mean_mass_diameter_m = std::pow(cell.frozen.dry_density_kg_m3 * frozen.mass_per_kg /
            (law.coefficient * (cell.frozen.dry_density_kg_m3 * frozen.number_per_kg)),
            1.0 / law.exponent);
        const double diameter = std::max(channel.rain_mean_mass_diameter_m, channel.frozen_mean_mass_diameter_m);
        constexpr double pi_over_six = physical_constants::pi / 6.0;
        const double frozen_density = index == 0 ? 500.0 : index == 2 ? 400.0 :
            std::min(500.0, law.coefficient / pi_over_six * std::pow(channel.frozen_mean_mass_diameter_m, law.exponent - 3.0));
        channel.mixed_density_kg_m3 = (frozen_density * std::pow(channel.frozen_mean_mass_diameter_m, 3.0) +
            microphysics_constants::rho_w * std::pow(channel.rain_mean_mass_diameter_m, 3.0)) / std::pow(diameter, 3.0);
        if (index == 1 && channel.mixed_density_kg_m3 <= .5 * (frozen_density + 400.0))
            channel.destination = TripleMomentSpecies::Snow;
        else if (parameters.hail_processes && channel.rain_mean_mass_diameter_m > 2.5e-3 && channel.mixed_density_kg_m3 >= 650.0)
            channel.destination = TripleMomentSpecies::Hail;
        channel.destination_density_kg_m3 = channel.destination == TripleMomentSpecies::Snow ? frozen_density :
            channel.destination == TripleMomentSpecies::Hail ? 900.0 : 400.0;
        channel.product_mass_per_kg = channel.rain.mass_per_kg + channel.frozen.mass_per_kg;
        channel.product_number_per_kg = std::min(channel.product_mass_per_kg /
            (channel.destination_density_kg_m3 * pi_over_six * diameter * diameter * diameter),
            std::min(channel.rain.number_per_kg, channel.frozen.number_per_kg));
        const auto product_law = current_nimbus_triple_sedimentation_parameters(channel.destination).mass_law;
        const double lambda = std::exp((std::log(product_law.coefficient) + std::log(channel.product_number_per_kg) +
            std::lgamma(product_law.exponent + 1.0) - std::log(channel.product_mass_per_kg)) / product_law.exponent);
        channel.product_reflectivity_per_kg = channel.product_number_per_kg * 720.0 / std::pow(lambda, 6.0);
        for (double value : {channel.rain.mass_per_kg, channel.rain.number_per_kg, channel.frozen.mass_per_kg,
                channel.frozen.number_per_kg, channel.rain_mean_mass_diameter_m, channel.frozen_mean_mass_diameter_m,
                channel.mixed_density_kg_m3, channel.destination_density_kg_m3, channel.product_mass_per_kg,
                channel.product_number_per_kg, channel.product_reflectivity_per_kg})
            if (!positive_finite(value))
                return reject(FrozenCollectionFailure::IntervalRejected,
                    "rain collision admitted pair or routed product is unrepresentable", error, failure);
        if (channel.frozen.mass_per_kg > frozen.mass_per_kg || channel.frozen.number_per_kg > frozen.number_per_kg)
            return reject(FrozenCollectionFailure::IntervalRejected,
                "rain collision pair exceeds its original frozen donor", error, failure);
    }
    double rain_mass = candidate.hail_collection().mass_per_kg;
    double rain_number = candidate.hail_collection().number_per_kg;
    for (const auto& channel : candidate.channels_)
    {
        rain_mass += channel.rain.mass_per_kg;
        rain_number += channel.rain.number_per_kg;
    }
    if (!std::isfinite(rain_mass) || !std::isfinite(rain_number) || rain_mass > cell.rain.mass_per_kg || rain_number > cell.rain.number_per_kg)
        return reject(FrozenCollectionFailure::IntervalRejected,
            "rain collision pairs exceed their shared original rain budget", error, failure);
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

bool validate_exact_rain_collision_admission(
    const ColdParameters& parameters, const RainCollectionCell& cell, double duration_s,
    const RainCollisionProducer& producer, std::string& error, std::size_t capacity,
    FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (capacity != 7)
        return reject(FrozenCollectionFailure::Capacity,
            "rain collisions require four rain and three frozen collection records", error, failure);
    if (producer.event_count() != 7)
        return reject(FrozenCollectionFailure::InvalidProducer, "rain collision journal is absent", error, failure);
    if (!validate_exact_rain_collection_admission(parameters, cell, duration_s, producer.rain_source(),
            error, 4, failure, spectrum_domain)) return false;
    RainCollisionProducer expected;
    if (!produce_exact_rain_collision_admission(parameters, cell, duration_s, expected, error,
            capacity, failure, spectrum_domain)) return false;
    const auto values = [](const RainCollisionChannel& c)
    {
        return std::array<double, 17>{c.rain.raw_mass_per_kg, c.rain.raw_number_per_kg, c.rain.mass_per_kg, c.rain.number_per_kg,
            c.frozen.raw_mass_per_kg, c.frozen.raw_number_per_kg, c.frozen.mass_per_kg, c.frozen.number_per_kg,
            c.admission_factor, c.rain_mean_mass_diameter_m, c.frozen_mean_mass_diameter_m, c.mixed_density_kg_m3,
            c.destination_density_kg_m3, c.product_mass_per_kg, c.product_number_per_kg, c.product_reflectivity_per_kg,
            static_cast<double>(c.destination)};
    };
    for (std::size_t index = 0; index < 3; ++index)
    {
        const auto a = values(producer.channels()[index]);
        const auto b = values(expected.channels()[index]);
        for (std::size_t field = 0; field < a.size(); ++field)
            if (!same_value(a[field], b[field]))
                return reject(FrozenCollectionFailure::InvalidProducer,
                    "rain collision journal differs from its paired source admission or routing", error, failure);
    }
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

bool evaluate_exact_rain_collision_increments(
    const ColdParameters& parameters, const RainCollectionCell& cell, double duration_s,
    const RainCollisionProducer& producer, RainCollisionIncrements& increments,
    std::string& error, std::size_t capacity, FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (!validate_exact_rain_collision_admission(parameters, cell, duration_s, producer, error,
            capacity, failure, spectrum_domain)) return false;
    RainCollisionIncrements candidate;
    const auto& hail = cell.frozen.initial.frozen[3];
    const double hail_gain = producer.hail_collection().mass_per_kg;
    candidate.rain.mass_per_kg = -hail_gain;
    candidate.rain.number_per_kg = -producer.hail_collection().number_per_kg;
    candidate.frozen.mass_increment_per_kg[3] = hail_gain;
    candidate.frozen.reflectivity_increment_per_kg[3] = hail.mass_per_kg > 0.0 ?
        2.0 * hail.reflectivity_per_kg * hail_gain / hail.mass_per_kg : 0.0;
    for (std::size_t index = 0; index < 3; ++index)
    {
        const auto& channel = producer.channels()[index];
        const auto& donor = cell.frozen.initial.frozen[index];
        candidate.rain.mass_per_kg -= channel.rain.mass_per_kg;
        candidate.rain.number_per_kg -= channel.rain.number_per_kg;
        candidate.frozen.mass_increment_per_kg[index] -= channel.frozen.mass_per_kg;
        candidate.frozen.number_increment_per_kg[index] -= channel.frozen.number_per_kg;
        if (donor.mass_per_kg > 0.0)
        {
            const double power = 6.0 / current_nimbus_triple_sedimentation_parameters(species[index]).mass_law.exponent;
            candidate.frozen.reflectivity_increment_per_kg[index] += donor.reflectivity_per_kg *
                (-power * channel.frozen.mass_per_kg / donor.mass_per_kg +
                 (power - 1.0) * channel.frozen.number_per_kg / donor.number_per_kg);
        }
        const auto destination = static_cast<std::size_t>(channel.destination) - 1;
        candidate.frozen.mass_increment_per_kg[destination] += channel.product_mass_per_kg;
        candidate.frozen.number_increment_per_kg[destination] += channel.product_number_per_kg;
        candidate.frozen.reflectivity_increment_per_kg[destination] += channel.product_reflectivity_per_kg;
    }
    if (cell.rain.mass_per_kg > 0.0)
        candidate.rain.reflectivity_per_kg = cell.rain.reflectivity_per_kg *
            (2.0 * candidate.rain.mass_per_kg / cell.rain.mass_per_kg - candidate.rain.number_per_kg / cell.rain.number_per_kg);
    candidate.temperature_k = -microphysics_constants::L_f * candidate.rain.mass_per_kg / microphysics_constants::cp;
    for (std::size_t index = 0; index < 4; ++index)
        for (double value : {candidate.frozen.mass_increment_per_kg[index], candidate.frozen.number_increment_per_kg[index],
                candidate.frozen.reflectivity_increment_per_kg[index]})
            if (!std::isfinite(value))
                return reject(FrozenCollectionFailure::IntervalRejected,
                    "rain collision frozen increment is unrepresentable", error, failure);
    for (double value : {candidate.rain.mass_per_kg, candidate.rain.number_per_kg, candidate.rain.reflectivity_per_kg, candidate.temperature_k})
        if (!std::isfinite(value))
            return reject(FrozenCollectionFailure::IntervalRejected,
                "rain collision rain or heat increment is unrepresentable", error, failure);
    increments = candidate;
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

bool apply_exact_triple_rain_collision_cell(
    const ColdParameters& parameters, const RainCollectionCell& cell, double duration_s,
    const RainCollisionProducer& producer, RainCollisionResult& result, std::string& error,
    std::size_t capacity, FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    RainCollisionResult candidate;
    if (!evaluate_exact_rain_collision_increments(parameters, cell, duration_s, producer,
            candidate.increments, error, capacity, failure, spectrum_domain)) return false;
    candidate.unrounded = cell;
    candidate.unrounded.frozen.temperature_k += candidate.increments.temperature_k;
    if (!std::isfinite(candidate.unrounded.frozen.temperature_k) || candidate.unrounded.frozen.temperature_k >= microphysics_constants::T0)
        return reject(FrozenCollectionFailure::IntervalRejected,
            "rain collision heat leaves the cold domain", error, failure);
    long double water = 0.0L, scale = cell.rain.mass_per_kg;
    for (std::size_t index = 0; index < 5; ++index)
    {
        const auto category = index == 0 ? TripleMomentSpecies::Rain : species[index - 1];
        const auto& source = index == 0 ? cell.rain : cell.frozen.initial.frozen[index - 1];
        auto& endpoint = index == 0 ? candidate.unrounded.rain : candidate.unrounded.frozen.initial.frozen[index - 1];
        const auto& increments = candidate.increments;
        endpoint.mass_per_kg += index == 0 ? increments.rain.mass_per_kg : increments.frozen.mass_increment_per_kg[index - 1];
        endpoint.number_per_kg += index == 0 ? increments.rain.number_per_kg : increments.frozen.number_increment_per_kg[index - 1];
        endpoint.reflectivity_per_kg += index == 0 ? increments.rain.reflectivity_per_kg : increments.frozen.reflectivity_increment_per_kg[index - 1];
        Spectrum spectrum;
        auto& rounded = index == 0 ? candidate.rounded_rain : candidate.rounded_frozen[index - 1];
        if (!population_spectrum(endpoint, category, spectrum, error, spectrum_domain) ||
            !stored_population(endpoint, category, rounded, error, spectrum_domain))
        {
            if (failure) *failure = FrozenCollectionFailure::IntervalRejected;
            return false;
        }
        water += static_cast<long double>(endpoint.mass_per_kg) - source.mass_per_kg;
        if (index != 0) scale += source.mass_per_kg;
    }
    if (std::abs(water) > 64.0L * std::numeric_limits<double>::epsilon() * scale)
        return reject(FrozenCollectionFailure::IntervalRejected,
            "rain collision endpoint violates its water budget", error, failure);
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

bool evaluate_exact_triple_rain_collision_cell(
    const ColdParameters& parameters, const RainCollectionCell& cell, double duration_s,
    RainCollisionResult& result, std::string& error, std::size_t capacity,
    FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    RainCollisionProducer producer;
    return produce_exact_rain_collision_admission(parameters, cell, duration_s, producer, error,
        capacity, failure, spectrum_domain) && apply_exact_triple_rain_collision_cell(parameters, cell, duration_s,
        producer, result, error, capacity, failure, spectrum_domain);
}

bool produce_exact_hail_wet_growth_admission(
    const ColdParameters& parameters, const HailWetGrowthCell& cell, double duration_s,
    HailWetGrowthProducer& producer, std::string& error, std::size_t capacity,
    FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    const auto& source = cell.source;
    const auto& state = source.initial;
    if (capacity != 1)
        return reject(FrozenCollectionFailure::Capacity, "hail wet growth requires one bounded thermal record", error, failure);
    if (!valid_cold_spectrum_domain(spectrum_domain) || !positive_finite(duration_s) ||
        !std::isfinite(source.temperature_k) || source.temperature_k < 150.0 || source.temperature_k >= microphysics_constants::T0 ||
        !std::isfinite(source.pressure_pa) || source.pressure_pa < 100.0 || source.pressure_pa > 200000.0 ||
        !positive_finite(source.dry_density_kg_m3) || parameter_values(parameters) != default_cold_parameters)
        return reject(FrozenCollectionFailure::InvalidContext, "hail wet-growth context or parameters differ", error, failure);
    for (double value : {state.vapor_mass_per_kg, state.cloud_mass_per_kg, state.cloud_number_per_kg})
        if (!std::isfinite(value) || value < 0.0 || !std::isfinite(static_cast<float>(value)))
            return reject(FrozenCollectionFailure::InvalidSource, "hail wet-growth vapor or cloud source is invalid", error, failure);
    if (state.cloud_mass_per_kg > 0.0 && state.cloud_number_per_kg <= 0.0)
        return reject(FrozenCollectionFailure::InvalidSource, "hail wet-growth cloud has no number source", error, failure);
    std::array<Spectrum, 4> spectra{};
    Spectrum rain;
    if (!population_spectrum(state.rain, TripleMomentSpecies::Rain, rain, error, spectrum_domain))
    {
        if (failure) *failure = FrozenCollectionFailure::InvalidSource;
        return false;
    }
    for (std::size_t index = 0; index < 4; ++index)
        if (!population_spectrum(state.frozen[index], species[index], spectra[index], error, spectrum_domain))
        {
            if (failure) *failure = FrozenCollectionFailure::InvalidSource;
            return false;
        }
    const std::array<double, 4> donors{state.cloud_mass_per_kg, state.rain.mass_per_kg,
        state.frozen[0].mass_per_kg, state.frozen[1].mass_per_kg};
    double dry_total = 0.0, wet_total = 0.0;
    for (std::size_t index = 0; index < 4; ++index)
    {
        const double dry = cell.dry_collection_per_kg[index];
        const double wet = cell.wet_collection_per_kg[index];
        if (!std::isfinite(dry) || !std::isfinite(wet) || dry < 0.0 || wet < 0.0)
            return reject(FrozenCollectionFailure::InvalidSource, "hail wet-growth collection is nonfinite or negative", error, failure);
        if (dry > donors[index] || wet > donors[index])
            return reject(FrozenCollectionFailure::IntervalRejected, "hail wet-growth collection exceeds an original donor", error, failure);
        dry_total += dry;
        wet_total += wet;
    }
    if (!same_value(cell.dry_collection_per_kg[1], cell.wet_collection_per_kg[1]))
        return reject(FrozenCollectionFailure::InvalidSource, "wet growth cannot change its original rain collection admission", error, failure);
    const auto& hail = state.frozen[3];
    if (hail.mass_per_kg == 0.0 && (dry_total != 0.0 || wet_total != 0.0))
        return reject(FrozenCollectionFailure::InvalidSource, "hail wet growth cannot use newly created hail as a collector", error, failure);
    HailWetGrowthProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameter_values(parameters);
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.spectrum_domain_ = spectrum_domain;
    candidate.duration_s_ = duration_s;
    candidate.retained_liquid_per_kg_ = cell.dry_collection_per_kg[0] + cell.dry_collection_per_kg[1];
    const double tc = source.temperature_k - microphysics_constants::T0;
    if (hail.mass_per_kg == 0.0)
        candidate.regime_ = HailWetGrowthRegime::NoOriginalHail;
    else if (tc <= -40.0)
        candidate.regime_ = HailWetGrowthRegime::ColdInactive;
    else
    {
        const auto air = diagnose_air_transport(source.temperature_k, source.pressure_pa, source.dry_density_kg_m3);
        const double denominator = microphysics_constants::L_f + 4218.0 * tc;
        double first = 0.0, ventilated = 0.0;
        if (!positive_finite(denominator) || !positive_finite(air.vapor_diffusivity) ||
            !positive_finite(air.thermal_conductivity) || !positive_finite(air.kinematic_viscosity) ||
            !moment(spectra[3], 1.0, first) || !moment(spectra[3], 1.5 + .5 * parameters.b_h_, ventilated))
            return reject(FrozenCollectionFailure::InvalidSource, "hail wet-growth ventilation is unrepresentable", error, failure);
        candidate.normalized_ventilation_m_ = .78 * first + .30 * air.schmidt_one_third *
            std::sqrt(air.density_factor * parameters.a_h_ / air.kinematic_viscosity) * ventilated;
        const double saturation = thermodynamics::saturation_mixing_ratio_water(microphysics_constants::T0, source.pressure_pa);
        candidate.atmospheric_capacity_per_kg_ = duration_s * 2.0 * physical_constants::pi * hail.number_per_kg *
            candidate.normalized_ventilation_m_ / denominator *
            (source.dry_density_kg_m3 * microphysics_constants::L_v * air.vapor_diffusivity *
                (saturation - state.vapor_mass_per_kg) - air.thermal_conductivity * tc);
        const double frozen = cell.wet_collection_per_kg[2] + cell.wet_collection_per_kg[3];
        candidate.frozen_capacity_per_kg_ = frozen * (1.0 - 2115.3 * tc / denominator);
        candidate.growth_capacity_per_kg_ = std::max(0.0,
            candidate.atmospheric_capacity_per_kg_ + candidate.frozen_capacity_per_kg_);
        for (double value : {candidate.normalized_ventilation_m_, candidate.atmospheric_capacity_per_kg_,
                candidate.frozen_capacity_per_kg_, candidate.growth_capacity_per_kg_})
            if (!std::isfinite(value))
                return reject(FrozenCollectionFailure::IntervalRejected, "hail wet-growth capacity is unrepresentable", error, failure);
        candidate.regime_ = candidate.growth_capacity_per_kg_ < dry_total ? HailWetGrowthRegime::Wet : HailWetGrowthRegime::Dry;
        if (candidate.regime_ == HailWetGrowthRegime::Wet)
        {
            if (candidate.growth_capacity_per_kg_ < frozen)
                return reject(FrozenCollectionFailure::IntervalRejected,
                    "hail wet growth requires surface-condensation or melting coupling for this frozen load", error, failure);
            const double liquid = cell.wet_collection_per_kg[0] + cell.wet_collection_per_kg[1];
            candidate.retained_liquid_per_kg_ = std::min(liquid, candidate.growth_capacity_per_kg_ - frozen);
            candidate.shed_mass_per_kg_ = liquid - candidate.retained_liquid_per_kg_;
            constexpr double diameter = 1e-3;
            candidate.shed_number_per_kg_ = candidate.shed_mass_per_kg_ / (parameters.c_r_ * diameter * diameter * diameter);
            if (candidate.shed_mass_per_kg_ > 0.0)
            {
                candidate.shed_reflectivity_per_kg_ = 5.6 * candidate.shed_mass_per_kg_ *
                    (candidate.shed_mass_per_kg_ / candidate.shed_number_per_kg_) / (parameters.c_r_ * parameters.c_r_);
                if (!positive_finite(candidate.shed_number_per_kg_) || !positive_finite(candidate.shed_reflectivity_per_kg_))
                    return reject(FrozenCollectionFailure::IntervalRejected, "hail shedding moments are unrepresentable", error, failure);
            }
        }
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

bool validate_exact_hail_wet_growth_admission(
    const ColdParameters& parameters, const HailWetGrowthCell& cell, double duration_s,
    const HailWetGrowthProducer& producer, std::string& error, std::size_t capacity,
    FrozenCollectionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    const auto values = [](const HailWetGrowthCell& cell)
    {
        const auto& state = cell.source.initial;
        std::array<double, 29> result{cell.source.temperature_k, cell.source.pressure_pa, cell.source.dry_density_kg_m3,
            state.vapor_mass_per_kg, state.cloud_mass_per_kg, state.cloud_number_per_kg,
            state.rain.mass_per_kg, state.rain.number_per_kg, state.rain.reflectivity_per_kg};
        for (std::size_t index = 0; index < 4; ++index)
        {
            result[9 + 3 * index] = state.frozen[index].mass_per_kg;
            result[10 + 3 * index] = state.frozen[index].number_per_kg;
            result[11 + 3 * index] = state.frozen[index].reflectivity_per_kg;
            result[21 + index] = cell.dry_collection_per_kg[index];
            result[25 + index] = cell.wet_collection_per_kg[index];
        }
        return result;
    };
    const auto a = values(cell), b = values(producer.source());
    bool bound = producer.event_count() == 1 && same_parameters(parameter_values(parameters), producer.parameters()) &&
        parameters.hail_processes == producer.hail_processes() && same_value(duration_s, producer.duration_s()) &&
        spectrum_domain == producer.spectrum_domain();
    for (std::size_t index = 0; index < a.size(); ++index) bound &= same_value(a[index], b[index]);
    if (!bound)
        return reject(FrozenCollectionFailure::InvalidProducer, "hail wet-growth journal differs from its source or collection budgets", error, failure);
    HailWetGrowthProducer expected;
    if (!produce_exact_hail_wet_growth_admission(parameters, cell, duration_s, expected, error, capacity, failure, spectrum_domain)) return false;
    const auto diagnostics = [](const HailWetGrowthProducer& p)
    {
        return std::array<double, 8>{p.normalized_ventilation_m(), p.atmospheric_capacity_per_kg(), p.frozen_capacity_per_kg(),
            p.growth_capacity_per_kg(), p.retained_liquid_per_kg(), p.shed_mass_per_kg(), p.shed_number_per_kg(), p.shed_reflectivity_per_kg()};
    };
    const auto x = diagnostics(producer), y = diagnostics(expected);
    for (std::size_t index = 0; index < x.size(); ++index)
        if (!same_value(x[index], y[index]))
            return reject(FrozenCollectionFailure::InvalidProducer, "hail wet-growth thermal or shedding capture differs", error, failure);
    if (producer.regime() != expected.regime())
        return reject(FrozenCollectionFailure::InvalidProducer, "hail wet-growth regime differs", error, failure);
    error.clear();
    if (failure) *failure = FrozenCollectionFailure::None;
    return true;
}

} // namespace tmv::milbrandt_detail
