#include "triple_cold_cloud_collection.hpp"

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

constexpr std::array<TripleMomentSpecies, 3> species{
    TripleMomentSpecies::Snow, TripleMomentSpecies::Graupel,
    TripleMomentSpecies::Hail};

constexpr std::array<ColdCloudCollectionChannel, 3> channels{
    ColdCloudCollectionChannel::CloudToSnow,
    ColdCloudCollectionChannel::CloudToGraupel,
    ColdCloudCollectionChannel::CloudToHail};

struct Spectrum
{
    double alpha = 0.0;
    double lambda = 0.0;
    double speed = 0.0;
    double mean_mass_diameter = 0.0;
    double normalized_mass = 0.0;
};

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

bool reject(ColdCloudCollectionFailure kind, const char* message,
            std::string& error, ColdCloudCollectionFailure* failure)
{
    error = message;
    if (failure)
    {
        *failure = kind;
    }
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
        {
            return false;
        }
    }
    return true;
}

bool same_source(const ColdCloudCollectionCell& a,
                 const ColdCloudCollectionCell& b) noexcept
{
    if (!same_value(a.temperature_k, b.temperature_k) ||
        !same_value(a.pressure_pa, b.pressure_pa) ||
        !same_value(a.dry_density_kg_m3, b.dry_density_kg_m3) ||
        !same_value(a.initial.cloud_mass_per_kg, b.initial.cloud_mass_per_kg) ||
        !same_value(a.initial.cloud_number_per_kg, b.initial.cloud_number_per_kg))
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
            error = "cold cloud collection frozen q/N/Z is nonfinite or negative";
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
        error = "cold cloud collection cubic spectrum is outside its admitted bracket";
        return false;
    }
    if (candidate.alpha > cold_spectrum_shape_limit(spectrum_domain, category))
    {
        error = cold_spectrum_shape_limit(spectrum_domain, category) == 200.0
            ? "cold cloud collection gamma shape exceeds 200"
            : "cold cloud collection converted-graupel gamma shape exceeds 512";
        return false;
    }
    candidate.normalized_mass = source.mass_per_kg / source.number_per_kg / law.coefficient;
    candidate.mean_mass_diameter = std::pow(candidate.normalized_mass, 1.0 / law.exponent);
    if (!positive_finite(candidate.normalized_mass) ||
        !positive_finite(candidate.mean_mass_diameter))
    {
        error = "cold cloud collection mass moment or mean diameter is unrepresentable";
        return false;
    }
    spectrum = candidate;
    return true;
}

bool normalized_moment(const Spectrum& spectrum, double order,
                       double& result) noexcept
{
    double candidate = 1.0;
    if (order == std::floor(order))
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

bool valid_cloud(double mass, double number, std::string& error)
{
    if (!std::isfinite(mass) || mass < 0.0 || !std::isfinite(number) || number < 0.0 ||
        (mass > 0.0 && number == 0.0))
    {
        error = "cold cloud collection cloud q/N is invalid or number-exhausted";
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

bool stored_population(const FrozenCollectionPopulation& source,
                       TripleMomentSpecies category,
                       FrozenCollectionEndpoint& endpoint, std::string& error,
                       ColdSpectrumDomain spectrum_domain)
{
    for (double value : {source.mass_per_kg, source.number_per_kg,
                         source.reflectivity_per_kg})
    {
        if (!float_range(value))
        {
            error = "cold cloud collection final frozen q/N/Z is not FP32 representable";
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

bool phase_endpoint(const ColdCloudCollectionProducer& producer,
                    ColdCloudCollectionResult& result, std::string& error,
                    ColdCloudCollectionFailure* failure)
{
    const auto& source = producer.source();
    ColdCloudCollectionResult candidate;
    candidate.unrounded = source.initial;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& event = producer.channels()[index];
        const auto& initial = source.initial.frozen[index];
        const double mass = event.admitted_mass_increment_per_kg;
        candidate.mass_increment_per_kg[index] = mass;
        candidate.cloud_mass_increment_per_kg -= mass;
        candidate.cloud_number_increment_per_kg -= event.admitted_number_increment_per_kg;
        if (initial.mass_per_kg > 0.0)
        {
            const double exponent = current_nimbus_triple_sedimentation_parameters(species[index]).mass_law.exponent;
            candidate.reflectivity_increment_per_kg[index] = initial.reflectivity_per_kg *
                ((6.0 / exponent) * mass / initial.mass_per_kg);
        }
        auto& endpoint = candidate.unrounded.frozen[index];
        endpoint.mass_per_kg += mass;
        endpoint.reflectivity_per_kg += candidate.reflectivity_increment_per_kg[index];
        Spectrum spectrum;
        if (!population_spectrum(endpoint, species[index], spectrum, error, producer.spectrum_domain()) ||
            !stored_population(endpoint, species[index], candidate.rounded[index], error, producer.spectrum_domain()))
        {
            if (failure)
            {
                *failure = ColdCloudCollectionFailure::IntervalRejected;
            }
            return false;
        }
    }
    candidate.unrounded.cloud_mass_per_kg += candidate.cloud_mass_increment_per_kg;
    candidate.unrounded.cloud_number_per_kg += candidate.cloud_number_increment_per_kg;
    if (!valid_cloud(candidate.unrounded.cloud_mass_per_kg,
            candidate.unrounded.cloud_number_per_kg, error) ||
        !float_range(candidate.unrounded.cloud_mass_per_kg) ||
        !float_range(candidate.unrounded.cloud_number_per_kg))
    {
        if (error.empty())
        {
            error = "cold cloud collection final cloud q/N is not FP32 representable";
        }
        if (failure)
        {
            *failure = ColdCloudCollectionFailure::IntervalRejected;
        }
        return false;
    }
    candidate.rounded_cloud_mass_per_kg = static_cast<float>(candidate.unrounded.cloud_mass_per_kg);
    candidate.rounded_cloud_number_per_kg = static_cast<float>(candidate.unrounded.cloud_number_per_kg);
    if (!valid_cloud(candidate.rounded_cloud_mass_per_kg, candidate.rounded_cloud_number_per_kg, error))
    {
        if (failure)
        {
            *failure = ColdCloudCollectionFailure::IntervalRejected;
        }
        return false;
    }
    candidate.temperature_increment_k = -microphysics_constants::L_f / microphysics_constants::cp *
        candidate.cloud_mass_increment_per_kg;
    const double temperature = source.temperature_k + candidate.temperature_increment_k;
    if (!std::isfinite(temperature) || temperature < 150.0 ||
        temperature >= microphysics_constants::T0)
    {
        return reject(ColdCloudCollectionFailure::IntervalRejected,
            "cold cloud collection heating leaves its cold phase", error, failure);
    }
    long double water_change = static_cast<long double>(candidate.unrounded.cloud_mass_per_kg) -
        source.initial.cloud_mass_per_kg;
    long double water_scale = source.initial.cloud_mass_per_kg;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        water_change += static_cast<long double>(candidate.unrounded.frozen[index].mass_per_kg) -
            source.initial.frozen[index].mass_per_kg;
        water_scale += source.initial.frozen[index].mass_per_kg;
    }
    if (std::abs(water_change) >
        64.0L * std::numeric_limits<double>::epsilon() * water_scale)
    {
        return reject(ColdCloudCollectionFailure::IntervalRejected,
            "cold cloud collection interval violates its exact water budget", error, failure);
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure)
    {
        *failure = ColdCloudCollectionFailure::None;
    }
    return true;
}

bool same_capture(const ColdCloudCollectionProducer& a,
                  const ColdCloudCollectionProducer& b) noexcept
{
    if (a.event_count() != b.event_count() ||
        !same_source(a.source(), b.source()) ||
        !same_values(a.parameters(), b.parameters()) ||
        a.hail_processes() != b.hail_processes() ||
        a.spectrum_domain() != b.spectrum_domain() ||
        a.wet_hail_collection() != b.wet_hail_collection() ||
        !same_value(a.duration_s(), b.duration_s()) ||
        !same_value(a.cloud_inverse_lambda_m(), b.cloud_inverse_lambda_m()) ||
        !same_values(a.source_alpha(), b.source_alpha()) ||
        !same_values(a.source_lambda_per_m(), b.source_lambda_per_m()) ||
        !same_values(a.mass_weighted_speed_m_s(), b.mass_weighted_speed_m_s()) ||
        !same_values(a.efficiencies(), b.efficiencies()))
    {
        return false;
    }
    for (std::size_t index = 0; index < channels.size(); ++index)
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
    return true;
}

} // namespace

bool produce_exact_cold_cloud_collection_admission(
    const ColdParameters& parameters, const ColdCloudCollectionCell& cell,
    double duration_s, ColdCloudCollectionProducer& producer, std::string& error,
    std::size_t capacity, ColdCloudCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    if (failure)
    {
        *failure = ColdCloudCollectionFailure::None;
    }
    if (!valid_cold_spectrum_domain(spectrum_domain))
    {
        return reject(ColdCloudCollectionFailure::InvalidContext,
            "cold cloud collection spectrum domain is invalid", error, failure);
    }
    if (!std::isfinite(duration_s) || duration_s <= 0.0 ||
        !std::isfinite(cell.temperature_k) || cell.temperature_k < 150.0 ||
        cell.temperature_k >= microphysics_constants::T0 ||
        !std::isfinite(cell.pressure_pa) || cell.pressure_pa < 100.0 ||
        cell.pressure_pa > 200000.0 || !positive_finite(cell.dry_density_kg_m3) ||
        parameter_values(parameters) != default_cold_parameters)
    {
        return reject(ColdCloudCollectionFailure::InvalidContext,
            "cold cloud collection context or retained physical parameters differ", error, failure);
    }
    if (capacity != channels.size())
    {
        return reject(ColdCloudCollectionFailure::Capacity,
            "cold cloud collection requires exactly three bounded channel records", error, failure);
    }
    if (!valid_cloud(cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg, error))
    {
        if (failure)
        {
            *failure = ColdCloudCollectionFailure::InvalidSource;
        }
        return false;
    }
    const double density_factor = std::sqrt(1.225 / cell.dry_density_kg_m3);
    const double dynamic_viscosity = 1.72e-5 *
        (393.0 / (cell.temperature_k + 120.0)) *
        std::pow(cell.temperature_k / microphysics_constants::T0, 1.5);
    if (!positive_finite(density_factor) || !positive_finite(dynamic_viscosity))
    {
        return reject(ColdCloudCollectionFailure::InvalidContext,
            "cold cloud collection air-density factor or viscosity is unrepresentable", error, failure);
    }
    ColdCloudCollectionProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameter_values(parameters);
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.spectrum_domain_ = spectrum_domain;
    candidate.wet_hail_collection_ = wet_hail_collection;
    candidate.duration_s_ = duration_s;
    std::array<double, 6> cloud_moments{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double cloud_diameter = 0.0;
    if (cell.initial.cloud_mass_per_kg > 0.0)
    {
        const double mass_moment = cell.initial.cloud_mass_per_kg /
            cell.initial.cloud_number_per_kg / parameters.c_r_;
        cloud_diameter = std::cbrt(mass_moment);
        candidate.cloud_inverse_lambda_m_ = std::cbrt(0.5 * mass_moment);
        if (!positive_finite(mass_moment) || !positive_finite(cloud_diameter) ||
            !positive_finite(candidate.cloud_inverse_lambda_m_))
        {
            return reject(ColdCloudCollectionFailure::InvalidSource,
                "cold cloud collection generalized cloud spectrum is unrepresentable", error, failure);
        }
        double power = 1.0;
        for (std::size_t order = 1; order < cloud_moments.size(); ++order)
        {
            power *= candidate.cloud_inverse_lambda_m_;
            cloud_moments[order] = std::tgamma(2.0 + static_cast<double>(order) / 3.0) * power;
            if (!positive_finite(cloud_moments[order]))
            {
                return reject(ColdCloudCollectionFailure::InvalidSource,
                    "cold cloud collection normalized cloud moment is unrepresentable", error, failure);
            }
        }
        cloud_moments[3] = mass_moment;
    }
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& initial = cell.initial.frozen[index];
        Spectrum spectrum;
        if (!population_spectrum(initial, species[index], spectrum, error, spectrum_domain))
        {
            if (failure)
            {
                *failure = ColdCloudCollectionFailure::InvalidSource;
            }
            return false;
        }
        const auto defaults = current_nimbus_triple_sedimentation_parameters(species[index]);
        if (initial.mass_per_kg > 0.0)
        {
            double velocity_moment = 0.0;
            if (!normalized_moment(spectrum,
                    defaults.mass_law.exponent + defaults.fall_exponent, velocity_moment))
            {
                return reject(ColdCloudCollectionFailure::InvalidSource,
                    "cold cloud collection frozen velocity moment is unrepresentable", error, failure);
            }
            spectrum.speed = density_factor * defaults.fall_coefficient *
                velocity_moment / spectrum.normalized_mass;
            if (!positive_finite(spectrum.speed))
            {
                return reject(ColdCloudCollectionFailure::InvalidSource,
                    "cold cloud collection mass-weighted velocity is unrepresentable", error, failure);
            }
        }
        candidate.source_alpha_[index] = spectrum.alpha;
        candidate.source_lambda_per_m_[index] = spectrum.lambda;
        candidate.mass_weighted_speed_m_s_[index] = spectrum.speed;
        auto& event = candidate.channels_[index];
        event.channel = channels[index];
        if (!(cell.initial.cloud_mass_per_kg > 1.0e-14) || !(initial.mass_per_kg > 1.0e-14))
        {
            continue;
        }
        double efficiency = 0.0;
        if (index == 0)
        {
            efficiency = std::min(cloud_diameter, 30.0e-6) * 3.333e4 *
                std::sqrt(std::min(spectrum.mean_mass_diameter, 1.0e-3) * 1.0e3);
        }
        else if (index == 1)
        {
            const double stokes = microphysics_constants::rho_w * spectrum.speed *
                cloud_diameter * cloud_diameter /
                (9.0 * dynamic_viscosity * spectrum.mean_mass_diameter);
            efficiency = 0.55 * std::log10(2.51 * std::clamp(stokes, 1.5, 10.0));
        }
        else
        {
            efficiency = std::exp(-8.68e-7 * std::pow(cloud_diameter, -1.6) *
                spectrum.mean_mass_diameter);
        }
        if (!std::isfinite(efficiency))
        {
            return reject(ColdCloudCollectionFailure::InvalidSource,
                "cold cloud collection operational efficiency is unrepresentable", error, failure);
        }
        efficiency = wet_hail_collection && index == 2 ? 1.0 : std::clamp(efficiency, 0.0, 1.0);
        candidate.efficiencies_[index] = efficiency;
        std::array<double, 3> frozen_moments{};
        for (std::size_t order = 0; order < frozen_moments.size(); ++order)
        {
            if (!normalized_moment(spectrum,
                    defaults.fall_exponent + static_cast<double>(order), frozen_moments[order]))
            {
                return reject(ColdCloudCollectionFailure::InvalidSource,
                    "cold cloud collection normalized frozen moment is unrepresentable", error, failure);
            }
        }
        constexpr double pi_over_four = 0.785398163397448309616;
        const double common = duration_s * pi_over_four * efficiency * density_factor *
            defaults.fall_coefficient * cell.dry_density_kg_m3 *
            cell.initial.cloud_number_per_kg * initial.number_per_kg;
        const double area_factor = index == 0 ? 0.6366 : 1.0;
        event.raw_mass_increment_per_kg = common * area_factor * parameters.c_r_ *
            (frozen_moments[2] * cloud_moments[3] +
             2.0 * frozen_moments[1] * cloud_moments[4] +
             frozen_moments[0] * cloud_moments[5]);
        event.raw_number_increment_per_kg = common * area_factor *
            (frozen_moments[2] + 2.0 * frozen_moments[1] * cloud_moments[1] +
             frozen_moments[0] * cloud_moments[2]);
        if (!std::isfinite(event.raw_mass_increment_per_kg) || event.raw_mass_increment_per_kg < 0.0 ||
            !std::isfinite(event.raw_number_increment_per_kg) || event.raw_number_increment_per_kg < 0.0)
        {
            return reject(ColdCloudCollectionFailure::IntervalRejected,
                "cold cloud collection uncapped interval is unrepresentable", error, failure);
        }
    }
    double mass_sink = 0.0;
    double number_sink = 0.0;
    for (const auto& event : candidate.channels_)
    {
        mass_sink += event.raw_mass_increment_per_kg;
        number_sink += event.raw_number_increment_per_kg;
    }
    if (!std::isfinite(mass_sink) || !std::isfinite(number_sink))
    {
        return reject(ColdCloudCollectionFailure::IntervalRejected,
            "cold cloud collection aggregate interval is unrepresentable", error, failure);
    }
    double factor = 1.0;
    if (mass_sink > 0.0)
    {
        factor = std::min(factor, cell.initial.cloud_mass_per_kg / mass_sink);
    }
    if (number_sink > 0.0)
    {
        factor = std::min(factor, cell.initial.cloud_number_per_kg / number_sink);
    }
    for (auto& event : candidate.channels_)
    {
        event.donor_admission_factor = factor;
        event.admitted_mass_increment_per_kg = event.raw_mass_increment_per_kg * factor;
        event.admitted_number_increment_per_kg = event.raw_number_increment_per_kg * factor;
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_cold_cloud_collection_cell(
    const ColdParameters& parameters, const ColdCloudCollectionCell& cell,
    double duration_s, ColdCloudCollectionResult& result, std::string& error,
    std::size_t capacity, ColdCloudCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    ColdCloudCollectionProducer producer;
    if (!produce_exact_cold_cloud_collection_admission(parameters, cell, duration_s,
            producer, error, capacity, failure, spectrum_domain, wet_hail_collection))
    {
        return false;
    }
    return phase_endpoint(producer, result, error, failure);
}

bool apply_exact_triple_cold_cloud_collection_cell(
    const ColdParameters& parameters, const ColdCloudCollectionCell& cell,
    double duration_s, const ColdCloudCollectionProducer& producer,
    ColdCloudCollectionResult& result, std::string& error,
    std::size_t capacity, ColdCloudCollectionFailure* failure, ColdSpectrumDomain spectrum_domain, bool wet_hail_collection)
{
    if (!valid_cold_spectrum_domain(spectrum_domain))
    {
        return reject(ColdCloudCollectionFailure::InvalidContext,
            "cold cloud collection spectrum domain is invalid", error, failure);
    }
    if (!producer.valid_ || !same_source(producer.source(), cell) ||
        !same_values(producer.parameters(), parameter_values(parameters)) ||
        producer.hail_processes() != parameters.hail_processes ||
        producer.spectrum_domain() != spectrum_domain ||
        producer.wet_hail_collection() != wet_hail_collection ||
        !same_value(producer.duration_s(), duration_s))
    {
        return reject(ColdCloudCollectionFailure::InvalidProducer,
            "cold cloud collection capture differs from its exact source or context", error, failure);
    }
    ColdCloudCollectionProducer expected;
    if (!produce_exact_cold_cloud_collection_admission(parameters, cell, duration_s,
            expected, error, capacity, failure, spectrum_domain, wet_hail_collection))
    {
        return false;
    }
    if (!same_capture(producer, expected))
    {
        return reject(ColdCloudCollectionFailure::InvalidProducer,
            "cold cloud collection capture differs from its exact spectrum, rates or admission", error, failure);
    }
    return phase_endpoint(expected, result, error, failure);
}

} // namespace tmv::milbrandt_detail
