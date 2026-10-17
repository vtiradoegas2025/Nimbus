#include "triple_ice_cloud_collection.hpp"

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

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

bool reject(IceCloudCollectionFailure kind, const char* message,
            std::string& error, IceCloudCollectionFailure* failure)
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

bool same_source(const IceCloudCollectionCell& a,
                 const IceCloudCollectionCell& b) noexcept
{
    return same_value(a.temperature_k, b.temperature_k)
        && same_value(a.pressure_pa, b.pressure_pa)
        && same_value(a.dry_density_kg_m3, b.dry_density_kg_m3)
        && same_value(a.initial.cloud_mass_per_kg, b.initial.cloud_mass_per_kg)
        && same_value(a.initial.cloud_number_per_kg, b.initial.cloud_number_per_kg)
        && same_value(a.initial.ice.mass_per_kg, b.initial.ice.mass_per_kg)
        && same_value(a.initial.ice.number_per_kg, b.initial.ice.number_per_kg)
        && same_value(a.initial.ice.reflectivity_per_kg, b.initial.ice.reflectivity_per_kg);
}

double cubic_log_shape_ratio(double alpha) noexcept
{
    return std::log1p(3.0 / (alpha + 1.0))
        + std::log1p(3.0 / (alpha + 2.0))
        + std::log1p(3.0 / (alpha + 3.0));
}

bool ice_spectrum(const FrozenCollectionPopulation& source, double& alpha,
                  double& lambda, std::string& error)
{
    for (double value : {source.mass_per_kg, source.number_per_kg,
                         source.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "ice cloud collection q/N/Z is nonfinite or negative";
            return false;
        }
    }
    if (source.mass_per_kg == 0.0 && source.number_per_kg == 0.0
        && source.reflectivity_per_kg == 0.0)
    {
        alpha = 0.0;
        lambda = 0.0;
        return true;
    }
    if (!diagnose_triple_gamma_shape(source.mass_per_kg, source.number_per_kg,
            source.reflectivity_per_kg, {440.0, 3.0}, alpha, lambda, error))
    {
        return false;
    }
    int q_exponent = 0;
    int n_exponent = 0;
    int z_exponent = 0;
    int c_exponent = 0;
    const double q_fraction = std::frexp(source.mass_per_kg, &q_exponent);
    const double n_fraction = std::frexp(source.number_per_kg, &n_exponent);
    const double z_fraction = std::frexp(source.reflectivity_per_kg, &z_exponent);
    const double c_fraction = std::frexp(440.0, &c_exponent);
    const double coefficient_over_mass = c_fraction / q_fraction;
    const double log_ratio = std::log(std::ldexp(
        z_fraction * n_fraction * coefficient_over_mass * coefficient_over_mass,
        z_exponent + n_exponent + 2 * (c_exponent - q_exponent)));
    double lower = -1.0 + 1.0e-9;
    double upper = 1.0;
    if (!positive_finite(log_ratio) || !(log_ratio < cubic_log_shape_ratio(lower)))
    {
        error = "ice cloud collection cubic spectrum is outside its admitted bracket";
        return false;
    }
    while (cubic_log_shape_ratio(upper) > log_ratio && upper < 1.0e6)
    {
        upper *= 2.0;
    }
    if (upper > 1.0e6 || cubic_log_shape_ratio(upper) > log_ratio)
    {
        error = "ice cloud collection cubic spectrum is outside its admitted bracket";
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
    const double candidate_alpha = 0.5 * (lower + upper);
    const double candidate_lambda = std::exp((std::log(440.0)
        + std::log(source.number_per_kg) + std::log(candidate_alpha + 1.0)
        + std::log(candidate_alpha + 2.0) + std::log(candidate_alpha + 3.0)
        - std::log(source.mass_per_kg)) / 3.0);
    if (!std::isfinite(candidate_alpha) || candidate_alpha > 200.0
        || !positive_finite(candidate_lambda))
    {
        error = "ice cloud collection gamma shape or slope is outside its retained bounds";
        return false;
    }
    alpha = candidate_alpha;
    lambda = candidate_lambda;
    return true;
}

bool valid_cloud(double mass, double number) noexcept
{
    return std::isfinite(mass) && mass >= 0.0 && std::isfinite(number)
        && number >= 0.0 && (mass == 0.0 || number > 0.0);
}

bool float_range(double value) noexcept
{
    return std::isfinite(value) && value >= 0.0
        && value <= std::numeric_limits<float>::max()
        && (value == 0.0 || value >= std::numeric_limits<float>::denorm_min());
}

bool same_capture(const IceCloudCollectionProducer& a,
                  const IceCloudCollectionProducer& b) noexcept
{
    const auto& x = a.channel();
    const auto& y = b.channel();
    return a.event_count() == b.event_count() && same_source(a.source(), b.source())
        && same_parameters(a.parameters(), b.parameters())
        && a.hail_processes() == b.hail_processes()
        && same_value(a.duration_s(), b.duration_s())
        && same_value(a.cloud_inverse_lambda_m(), b.cloud_inverse_lambda_m())
        && same_value(a.source_alpha(), b.source_alpha())
        && same_value(a.source_lambda_per_m(), b.source_lambda_per_m())
        && same_value(x.raw_mass_increment_per_kg, y.raw_mass_increment_per_kg)
        && same_value(x.raw_number_increment_per_kg, y.raw_number_increment_per_kg)
        && same_value(x.admitted_mass_increment_per_kg, y.admitted_mass_increment_per_kg)
        && same_value(x.admitted_number_increment_per_kg, y.admitted_number_increment_per_kg)
        && same_value(x.donor_admission_factor, y.donor_admission_factor);
}

bool phase_endpoint(const IceCloudCollectionProducer& producer,
                    IceCloudCollectionResult& result, std::string& error,
                    IceCloudCollectionFailure* failure)
{
    const auto& source = producer.source();
    const auto& event = producer.channel();
    IceCloudCollectionResult candidate;
    candidate.unrounded = source.initial;
    candidate.ice_mass_increment_per_kg = event.admitted_mass_increment_per_kg;
    candidate.cloud_mass_increment_per_kg = -event.admitted_mass_increment_per_kg;
    candidate.cloud_number_increment_per_kg = -event.admitted_number_increment_per_kg;
    if (source.initial.ice.mass_per_kg > 0.0)
    {
        candidate.ice_reflectivity_increment_per_kg = source.initial.ice.reflectivity_per_kg
            * (2.0 * event.admitted_mass_increment_per_kg / source.initial.ice.mass_per_kg);
    }
    candidate.unrounded.cloud_mass_per_kg += candidate.cloud_mass_increment_per_kg;
    candidate.unrounded.cloud_number_per_kg += candidate.cloud_number_increment_per_kg;
    candidate.unrounded.ice.mass_per_kg += candidate.ice_mass_increment_per_kg;
    candidate.unrounded.ice.reflectivity_per_kg += candidate.ice_reflectivity_increment_per_kg;
    double alpha = 0.0;
    double lambda = 0.0;
    if (!valid_cloud(candidate.unrounded.cloud_mass_per_kg, candidate.unrounded.cloud_number_per_kg)
        || !float_range(candidate.unrounded.cloud_mass_per_kg)
        || !float_range(candidate.unrounded.cloud_number_per_kg)
        || !ice_spectrum(candidate.unrounded.ice, alpha, lambda, error)
        || !float_range(candidate.unrounded.ice.mass_per_kg)
        || !float_range(candidate.unrounded.ice.number_per_kg)
        || !float_range(candidate.unrounded.ice.reflectivity_per_kg))
    {
        return reject(IceCloudCollectionFailure::IntervalRejected,
            "ice cloud collection unrounded endpoint is unrealizable or not FP32 representable", error, failure);
    }
    candidate.rounded_cloud_mass_per_kg = static_cast<float>(candidate.unrounded.cloud_mass_per_kg);
    candidate.rounded_cloud_number_per_kg = static_cast<float>(candidate.unrounded.cloud_number_per_kg);
    candidate.rounded_ice = {static_cast<float>(candidate.unrounded.ice.mass_per_kg),
        static_cast<float>(candidate.unrounded.ice.number_per_kg),
        static_cast<float>(candidate.unrounded.ice.reflectivity_per_kg)};
    if (!valid_cloud(candidate.rounded_cloud_mass_per_kg, candidate.rounded_cloud_number_per_kg)
        || !ice_spectrum({candidate.rounded_ice.mass_per_kg, candidate.rounded_ice.number_per_kg,
                         candidate.rounded_ice.reflectivity_per_kg}, alpha, lambda, error))
    {
        return reject(IceCloudCollectionFailure::IntervalRejected,
            "ice cloud collection stored endpoint is unrealizable", error, failure);
    }
    candidate.temperature_increment_k = microphysics_constants::L_f / microphysics_constants::cp
        * event.admitted_mass_increment_per_kg;
    const double temperature = source.temperature_k + candidate.temperature_increment_k;
    if (!std::isfinite(temperature) || temperature < 150.0 || temperature >= microphysics_constants::T0)
    {
        return reject(IceCloudCollectionFailure::IntervalRejected,
            "ice cloud collection heating leaves its cold phase", error, failure);
    }
    const long double water_change = static_cast<long double>(candidate.unrounded.cloud_mass_per_kg)
        - source.initial.cloud_mass_per_kg + static_cast<long double>(candidate.unrounded.ice.mass_per_kg)
        - source.initial.ice.mass_per_kg;
    const long double water_scale = static_cast<long double>(source.initial.cloud_mass_per_kg)
        + source.initial.ice.mass_per_kg;
    if (std::abs(water_change) > 64.0L * std::numeric_limits<double>::epsilon() * water_scale)
    {
        return reject(IceCloudCollectionFailure::IntervalRejected,
            "ice cloud collection interval violates its exact water budget", error, failure);
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure)
    {
        *failure = IceCloudCollectionFailure::None;
    }
    return true;
}

} // namespace

bool produce_exact_ice_cloud_collection_admission(
    const ColdParameters& parameters, const IceCloudCollectionCell& cell,
    double duration_s, IceCloudCollectionProducer& producer, std::string& error,
    std::size_t capacity, IceCloudCollectionFailure* failure)
{
    if (failure)
    {
        *failure = IceCloudCollectionFailure::None;
    }
    if (!std::isfinite(duration_s) || duration_s <= 0.0
        || !std::isfinite(cell.temperature_k) || cell.temperature_k < 150.0
        || cell.temperature_k >= microphysics_constants::T0
        || !std::isfinite(cell.pressure_pa) || cell.pressure_pa < 100.0
        || cell.pressure_pa > 200000.0 || !positive_finite(cell.dry_density_kg_m3)
        || parameter_values(parameters) != default_cold_parameters)
    {
        return reject(IceCloudCollectionFailure::InvalidContext,
            "ice cloud collection context or retained physical parameters differ", error, failure);
    }
    if (capacity != 1)
    {
        return reject(IceCloudCollectionFailure::Capacity,
            "ice cloud collection requires exactly one bounded channel record", error, failure);
    }
    if (!valid_cloud(cell.initial.cloud_mass_per_kg, cell.initial.cloud_number_per_kg))
    {
        return reject(IceCloudCollectionFailure::InvalidSource,
            "ice cloud collection cloud q/N is invalid or number-exhausted", error, failure);
    }
    IceCloudCollectionProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = parameter_values(parameters);
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.duration_s_ = duration_s;
    if (!ice_spectrum(cell.initial.ice, candidate.source_alpha_, candidate.source_lambda_per_m_, error))
    {
        if (failure)
        {
            *failure = IceCloudCollectionFailure::InvalidSource;
        }
        return false;
    }
    const double cloud = cell.initial.cloud_mass_per_kg;
    const double cloud_number = cell.initial.cloud_number_per_kg;
    std::array<double, 6> cloud_moments{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (cloud > 0.0)
    {
        const double mass_moment = cloud / cloud_number / parameters.c_r_;
        candidate.cloud_inverse_lambda_m_ = std::cbrt(0.5 * mass_moment);
        if (!positive_finite(mass_moment) || !positive_finite(candidate.cloud_inverse_lambda_m_))
        {
            return reject(IceCloudCollectionFailure::InvalidSource,
                "ice cloud collection generalized cloud spectrum is unrepresentable", error, failure);
        }
        for (std::size_t order = 1; order < cloud_moments.size(); ++order)
        {
            cloud_moments[order] = order == 3 ? mass_moment
                : std::tgamma(2.0 + static_cast<double>(order) / 3.0)
                    * std::pow(candidate.cloud_inverse_lambda_m_, static_cast<double>(order));
            if (!positive_finite(cloud_moments[order]))
            {
                return reject(IceCloudCollectionFailure::InvalidSource,
                    "ice cloud collection normalized cloud moment is unrepresentable", error, failure);
            }
        }
    }
    auto& event = candidate.channel_;
    if (cloud > 1.0e-14 && cell.initial.ice.mass_per_kg > 1.0e-14)
    {
        std::array<double, 3> ice_moments{};
        for (std::size_t order = 0; order < ice_moments.size(); ++order)
        {
            const double exponent = parameters.b_i_ + static_cast<double>(order);
            ice_moments[order] = std::exp(std::lgamma(candidate.source_alpha_ + exponent + 1.0)
                - std::lgamma(candidate.source_alpha_ + 1.0)
                - exponent * std::log(candidate.source_lambda_per_m_));
            if (!positive_finite(ice_moments[order]))
            {
                return reject(IceCloudCollectionFailure::InvalidSource,
                    "ice cloud collection normalized ice moment is unrepresentable", error, failure);
            }
        }
        const double common = duration_s * 0.785398163397448309616
            * std::sqrt(1.225 / cell.dry_density_kg_m3) * parameters.a_i_
            * cell.dry_density_kg_m3 * cloud_number * cell.initial.ice.number_per_kg;
        event.raw_mass_increment_per_kg = common * parameters.c_r_
            * (ice_moments[2] * cloud_moments[3] + 2.0 * ice_moments[1] * cloud_moments[4]
               + ice_moments[0] * cloud_moments[5]);
        event.raw_number_increment_per_kg = common
            * (ice_moments[2] + 2.0 * ice_moments[1] * cloud_moments[1]
               + ice_moments[0] * cloud_moments[2]);
        if (!std::isfinite(event.raw_mass_increment_per_kg) || event.raw_mass_increment_per_kg < 0.0
            || !std::isfinite(event.raw_number_increment_per_kg) || event.raw_number_increment_per_kg < 0.0)
        {
            return reject(IceCloudCollectionFailure::IntervalRejected,
                "ice cloud collection uncapped interval is unrepresentable", error, failure);
        }
        if (event.raw_mass_increment_per_kg > cloud)
        {
            event.donor_admission_factor = cloud / event.raw_mass_increment_per_kg;
        }
        if (event.raw_number_increment_per_kg > cloud_number)
        {
            event.donor_admission_factor = std::min(event.donor_admission_factor,
                cloud_number / event.raw_number_increment_per_kg);
        }
    }
    event.admitted_mass_increment_per_kg = event.raw_mass_increment_per_kg * event.donor_admission_factor;
    event.admitted_number_increment_per_kg = event.raw_number_increment_per_kg * event.donor_admission_factor;
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_ice_cloud_collection_cell(
    const ColdParameters& parameters, const IceCloudCollectionCell& cell,
    double duration_s, IceCloudCollectionResult& result, std::string& error,
    std::size_t capacity, IceCloudCollectionFailure* failure)
{
    IceCloudCollectionProducer producer;
    if (!produce_exact_ice_cloud_collection_admission(parameters, cell, duration_s,
            producer, error, capacity, failure))
    {
        return false;
    }
    return phase_endpoint(producer, result, error, failure);
}

bool apply_exact_triple_ice_cloud_collection_cell(
    const ColdParameters& parameters, const IceCloudCollectionCell& cell,
    double duration_s, const IceCloudCollectionProducer& producer,
    IceCloudCollectionResult& result, std::string& error,
    std::size_t capacity, IceCloudCollectionFailure* failure)
{
    if (!producer.valid_ || !same_source(producer.source(), cell)
        || !same_parameters(producer.parameters(), parameter_values(parameters))
        || producer.hail_processes() != parameters.hail_processes
        || !same_value(producer.duration_s(), duration_s))
    {
        return reject(IceCloudCollectionFailure::InvalidProducer,
            "ice cloud collection capture differs from its exact source or context", error, failure);
    }
    IceCloudCollectionProducer expected;
    if (!produce_exact_ice_cloud_collection_admission(parameters, cell, duration_s,
            expected, error, capacity, failure))
    {
        return false;
    }
    if (!same_capture(producer, expected))
    {
        return reject(IceCloudCollectionFailure::InvalidProducer,
            "ice cloud collection capture differs from its exact admitted channel", error, failure);
    }
    return phase_endpoint(expected, result, error, failure);
}

} // namespace tmv::milbrandt_detail
