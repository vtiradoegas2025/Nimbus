#include "triple_cold_cloud_diffusion.hpp"

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

constexpr std::array<ColdCloudDiffusionSpecies, 3> diffusion_species{
    ColdCloudDiffusionSpecies::Snow, ColdCloudDiffusionSpecies::Graupel,
    ColdCloudDiffusionSpecies::Hail};

bool positive_finite(double value) noexcept
{
    return std::isfinite(value) && value > 0.0;
}

bool reject(ColdCloudDiffusionFailure kind, const char* message,
            std::string& error, ColdCloudDiffusionFailure* failure)
{
    error = message;
    if (failure)
    {
        *failure = kind;
    }
    return false;
}

ColdCloudDiffusionFailure collection_failure(ColdCloudCollectionFailure failure) noexcept
{
    switch (failure)
    {
    case ColdCloudCollectionFailure::None:
        return ColdCloudDiffusionFailure::None;
    case ColdCloudCollectionFailure::InvalidSource:
        return ColdCloudDiffusionFailure::InvalidSource;
    case ColdCloudCollectionFailure::InvalidContext:
        return ColdCloudDiffusionFailure::InvalidContext;
    case ColdCloudCollectionFailure::Capacity:
        return ColdCloudDiffusionFailure::Capacity;
    case ColdCloudCollectionFailure::InvalidProducer:
        return ColdCloudDiffusionFailure::InvalidProducer;
    case ColdCloudCollectionFailure::IntervalRejected:
        return ColdCloudDiffusionFailure::IntervalRejected;
    }
    return ColdCloudDiffusionFailure::InvalidSource;
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

bool same_collection_source(const ColdCloudCollectionCell& a,
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
    for (std::size_t index = 0; index < species.size(); ++index)
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

ColdCloudCollectionCell collection_cell(const ColdCloudDiffusionCell& source)
{
    return {source.temperature_k, source.pressure_pa, source.dry_density_kg_m3, source.initial};
}

bool same_source(const ColdCloudDiffusionCell& a,
                 const ColdCloudDiffusionCell& b) noexcept
{
    return same_value(a.vapor_mass_per_kg, b.vapor_mass_per_kg) &&
        same_collection_source(collection_cell(a), collection_cell(b));
}

bool same_collection(const ColdCloudCollectionProducer& a,
                     const ColdCloudCollectionProducer& b) noexcept
{
    if (a.event_count() != b.event_count() ||
        !same_collection_source(a.source(), b.source()) ||
        !same_values(a.parameters(), b.parameters()) ||
        a.hail_processes() != b.hail_processes() ||
        a.wet_hail_collection() != b.wet_hail_collection() ||
        a.spectrum_domain() != b.spectrum_domain() ||
        !same_value(a.duration_s(), b.duration_s()) ||
        !same_value(a.cloud_inverse_lambda_m(), b.cloud_inverse_lambda_m()) ||
        !same_values(a.source_alpha(), b.source_alpha()) ||
        !same_values(a.source_lambda_per_m(), b.source_lambda_per_m()) ||
        !same_values(a.mass_weighted_speed_m_s(), b.mass_weighted_speed_m_s()) ||
        !same_values(a.efficiencies(), b.efficiencies()))
    {
        return false;
    }
    for (std::size_t index = 0; index < species.size(); ++index)
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

bool same_thermodynamics(const ColdCloudDiffusionThermodynamics& a,
                         const ColdCloudDiffusionThermodynamics& b) noexcept
{
    return same_values(std::array<double, 11>{
        a.saturation_ice_per_kg, a.saturation_temperature_derivative_per_kg_k,
        a.saturation_ratio_ice, a.vapor_diffusivity_m2_s,
        a.thermal_conductivity_w_m_k, a.kinematic_viscosity_m2_s,
        a.schmidt_one_third, a.density_factor,
        a.diffusion_resistance_m_s_per_kg, a.riming_feedback, a.relaxation_number},
        std::array<double, 11>{
        b.saturation_ice_per_kg, b.saturation_temperature_derivative_per_kg_k,
        b.saturation_ratio_ice, b.vapor_diffusivity_m2_s,
        b.thermal_conductivity_w_m_k, b.kinematic_viscosity_m2_s,
        b.schmidt_one_third, b.density_factor,
        b.diffusion_resistance_m_s_per_kg, b.riming_feedback, b.relaxation_number}) &&
        same_values(a.thermal_lambda_per_m, b.thermal_lambda_per_m) &&
        same_values(a.ventilation_m, b.ventilation_m) &&
        same_values(a.ambient_rate_coefficient_per_s, b.ambient_rate_coefficient_per_s);
}

bool same_capture(const ColdCloudDiffusionProducer& a,
                  const ColdCloudDiffusionProducer& b) noexcept
{
    if (a.spectrum_domain() != b.spectrum_domain() || a.event_count() != b.event_count() || !same_source(a.source(), b.source()) ||
        !same_value(a.duration_s(), b.duration_s()) ||
        !same_collection(a.collection_producer(), b.collection_producer()) ||
        !same_thermodynamics(a.thermodynamics(), b.thermodynamics()))
    {
        return false;
    }
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& x = a.channels()[index];
        const auto& y = b.channels()[index];
        if (x.species != y.species ||
            !same_value(x.environmental_mass_increment_per_kg, y.environmental_mass_increment_per_kg) ||
            !same_value(x.riming_mass_increment_per_kg, y.riming_mass_increment_per_kg) ||
            !same_value(x.raw_mass_increment_per_kg, y.raw_mass_increment_per_kg) ||
            !same_value(x.admitted_mass_increment_per_kg, y.admitted_mass_increment_per_kg) ||
            !same_value(x.admitted_number_increment_per_kg, y.admitted_number_increment_per_kg))
        {
            return false;
        }
    }
    return true;
}

double normalized_moment(double alpha, double lambda, double order) noexcept
{
    if (order == 1.0)
    {
        return (alpha + 1.0) / lambda;
    }
    return std::exp(std::lgamma(alpha + order + 1.0) -
        std::lgamma(alpha + 1.0) - order * std::log(lambda));
}

double saturation_derivative(double temperature, double pressure) noexcept
{
    const double tc = temperature - microphysics_constants::T0;
    const double e = thermodynamics::saturation_vapor_pressure_ice(temperature);
    const double denominator = std::max(pressure - e, 1.0);
    const double qraw = 0.62198 * e / denominator;
    if (qraw >= 0.2)
    {
        return 0.0;
    }
    const double derivative = e *
        (23.036 * 279.82 - (2.0 * 279.82 * tc + tc * tc) / 333.7) /
        ((279.82 + tc) * (279.82 + tc));
    return pressure - e > 1.0
        ? 0.62198 * pressure * derivative / (denominator * denominator)
        : 0.62198 * derivative;
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
    return std::isfinite(alpha);
}

bool valid_population(const FrozenCollectionPopulation& source,
                      TripleMomentSpecies category, std::string& error, ColdSpectrumDomain spectrum_domain)
{
    for (double value : {source.mass_per_kg, source.number_per_kg,
                         source.reflectivity_per_kg})
    {
        if (!std::isfinite(value) || value < 0.0)
        {
            error = "coupled cold cloud diffusion q/N/Z is nonfinite or negative";
            return false;
        }
    }
    if (source.mass_per_kg == 0.0 && source.number_per_kg == 0.0 &&
        source.reflectivity_per_kg == 0.0)
    {
        return true;
    }
    double alpha = 0.0;
    double lambda = 0.0;
    const auto law = current_nimbus_triple_sedimentation_parameters(category).mass_law;
    if (!diagnose_triple_gamma_shape(source.mass_per_kg, source.number_per_kg,
            source.reflectivity_per_kg, law, alpha, lambda, error))
    {
        return false;
    }
    if (law.exponent == 3.0 && !cubic_shape(source, law.coefficient, alpha))
    {
        error = "coupled cold cloud diffusion cubic spectrum is outside its admitted bracket";
        return false;
    }
    if (alpha > cold_spectrum_shape_limit(spectrum_domain, category))
    {
        error = cold_spectrum_shape_limit(spectrum_domain, category) == 200.0
            ? "coupled cold cloud diffusion gamma shape exceeds 200"
            : "coupled cold cloud diffusion converted-graupel gamma shape exceeds 512";
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

bool valid_cloud(double mass, double number) noexcept
{
    return std::isfinite(mass) && mass >= 0.0 &&
        std::isfinite(number) && number >= 0.0 &&
        (mass == 0.0 || number > 0.0);
}

bool phase_endpoint(const ColdCloudDiffusionProducer& producer,
                    ColdCloudDiffusionResult& result, std::string& error,
                    ColdCloudDiffusionFailure* failure)
{
    const auto& source = producer.source();
    const auto& collection = producer.collection_producer();
    ColdCloudDiffusionResult candidate;
    candidate.unrounded = source.initial;
    double total_collection = 0.0;
    double total_diffusion = 0.0;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& initial = source.initial.frozen[index];
        const auto& collected = collection.channels()[index];
        const auto& diffused = producer.channels()[index];
        const double mass_increment = collected.admitted_mass_increment_per_kg +
            diffused.admitted_mass_increment_per_kg;
        candidate.mass_increment_per_kg[index] = mass_increment;
        candidate.number_increment_per_kg[index] = diffused.admitted_number_increment_per_kg;
        total_collection += collected.admitted_mass_increment_per_kg;
        total_diffusion += diffused.admitted_mass_increment_per_kg;
        candidate.cloud_number_increment_per_kg -= collected.admitted_number_increment_per_kg;
        if (initial.mass_per_kg > 0.0)
        {
            const double exponent = current_nimbus_triple_sedimentation_parameters(species[index]).mass_law.exponent;
            const double power = 6.0 / exponent;
            candidate.reflectivity_increment_per_kg[index] = initial.reflectivity_per_kg *
                (power * mass_increment / initial.mass_per_kg +
                 (1.0 - power) * diffused.admitted_number_increment_per_kg / initial.number_per_kg);
        }
        auto& endpoint = candidate.unrounded.frozen[index];
        endpoint = {initial.mass_per_kg + mass_increment,
            initial.number_per_kg + diffused.admitted_number_increment_per_kg,
            initial.reflectivity_per_kg + candidate.reflectivity_increment_per_kg[index]};
        if (!valid_population(endpoint, species[index], error, producer.spectrum_domain()) ||
            !float_range(endpoint.mass_per_kg) || !float_range(endpoint.number_per_kg) ||
            !float_range(endpoint.reflectivity_per_kg))
        {
            if (error.empty())
            {
                error = "coupled cold cloud diffusion frozen endpoint is not FP32 representable";
            }
            if (failure)
            {
                *failure = ColdCloudDiffusionFailure::IntervalRejected;
            }
            return false;
        }
        auto& rounded = candidate.rounded[index];
        rounded = {static_cast<float>(endpoint.mass_per_kg),
            static_cast<float>(endpoint.number_per_kg),
            static_cast<float>(endpoint.reflectivity_per_kg)};
        if (!valid_population({rounded.mass_per_kg, rounded.number_per_kg,
                rounded.reflectivity_per_kg}, species[index], error, producer.spectrum_domain()))
        {
            if (failure)
            {
                *failure = ColdCloudDiffusionFailure::IntervalRejected;
            }
            return false;
        }
    }
    candidate.cloud_mass_increment_per_kg = -total_collection;
    candidate.vapor_mass_increment_per_kg = -total_diffusion;
    candidate.unrounded.cloud_mass_per_kg += candidate.cloud_mass_increment_per_kg;
    candidate.unrounded.cloud_number_per_kg += candidate.cloud_number_increment_per_kg;
    candidate.vapor_mass_per_kg = source.vapor_mass_per_kg + candidate.vapor_mass_increment_per_kg;
    if (!valid_cloud(candidate.unrounded.cloud_mass_per_kg, candidate.unrounded.cloud_number_per_kg) ||
        !float_range(candidate.unrounded.cloud_mass_per_kg) ||
        !float_range(candidate.unrounded.cloud_number_per_kg) ||
        !float_range(candidate.vapor_mass_per_kg))
    {
        return reject(ColdCloudDiffusionFailure::IntervalRejected,
            "coupled cold cloud diffusion vapor/cloud endpoint is negative, exhausted or not FP32 representable",
            error, failure);
    }
    candidate.rounded_vapor_mass_per_kg = static_cast<float>(candidate.vapor_mass_per_kg);
    candidate.rounded_cloud_mass_per_kg = static_cast<float>(candidate.unrounded.cloud_mass_per_kg);
    candidate.rounded_cloud_number_per_kg = static_cast<float>(candidate.unrounded.cloud_number_per_kg);
    if (!valid_cloud(candidate.rounded_cloud_mass_per_kg, candidate.rounded_cloud_number_per_kg))
    {
        return reject(ColdCloudDiffusionFailure::IntervalRejected,
            "coupled cold cloud diffusion stored cloud endpoint is number-exhausted", error, failure);
    }
    candidate.temperature_increment_k =
        (microphysics_constants::L_f * total_collection + microphysics_constants::L_s * total_diffusion) /
        microphysics_constants::cp;
    const double temperature = source.temperature_k + candidate.temperature_increment_k;
    if (!std::isfinite(temperature) || temperature < 150.0 || temperature >= microphysics_constants::T0)
    {
        return reject(ColdCloudDiffusionFailure::IntervalRejected,
            "coupled cold cloud diffusion heating leaves its cold phase", error, failure);
    }
    long double water_change = static_cast<long double>(candidate.unrounded.cloud_mass_per_kg) -
        source.initial.cloud_mass_per_kg + static_cast<long double>(candidate.vapor_mass_per_kg) -
        source.vapor_mass_per_kg;
    long double water_scale = source.initial.cloud_mass_per_kg +
        static_cast<long double>(source.vapor_mass_per_kg);
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        water_change += static_cast<long double>(candidate.unrounded.frozen[index].mass_per_kg) -
            source.initial.frozen[index].mass_per_kg;
        water_scale += source.initial.frozen[index].mass_per_kg;
    }
    if (std::abs(water_change) >
        64.0L * std::numeric_limits<double>::epsilon() * water_scale)
    {
        return reject(ColdCloudDiffusionFailure::IntervalRejected,
            "coupled cold cloud diffusion interval violates its exact water budget", error, failure);
    }
    candidate.producer = producer;
    result = candidate;
    error.clear();
    if (failure)
    {
        *failure = ColdCloudDiffusionFailure::None;
    }
    return true;
}

} // namespace

bool produce_exact_cold_cloud_diffusion_admission(
    const ColdParameters& parameters, const ColdCloudDiffusionCell& cell,
    double duration_s, ColdCloudDiffusionProducer& producer, std::string& error,
    std::size_t capacity, ColdCloudDiffusionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (failure)
    {
        *failure = ColdCloudDiffusionFailure::None;
    }
    if (!valid_cold_spectrum_domain(spectrum_domain))
    {
        return reject(ColdCloudDiffusionFailure::InvalidContext,
            "coupled cold cloud diffusion spectrum domain is invalid", error, failure);
    }
    if (!std::isfinite(cell.vapor_mass_per_kg) || cell.vapor_mass_per_kg < 0.0)
    {
        return reject(ColdCloudDiffusionFailure::InvalidSource,
            "coupled cold cloud diffusion vapor source is nonfinite or negative", error, failure);
    }
    ColdCloudDiffusionProducer candidate;
    ColdCloudCollectionFailure collection_status = ColdCloudCollectionFailure::None;
    if (!produce_exact_cold_cloud_collection_admission(parameters, collection_cell(cell), duration_s,
            candidate.collection_producer_, error, capacity, &collection_status, spectrum_domain))
    {
        if (failure)
        {
            *failure = collection_failure(collection_status);
        }
        return false;
    }
    candidate.source_ = cell;
    candidate.spectrum_domain_ = spectrum_domain;
    candidate.duration_s_ = duration_s;
    auto& diagnostic = candidate.thermodynamics_;
    diagnostic.saturation_ice_per_kg = thermodynamics::saturation_mixing_ratio_ice(
        cell.temperature_k, cell.pressure_pa);
    diagnostic.saturation_temperature_derivative_per_kg_k = saturation_derivative(
        cell.temperature_k, cell.pressure_pa);
    diagnostic.saturation_ratio_ice = cell.vapor_mass_per_kg / diagnostic.saturation_ice_per_kg;
    const auto air = diagnose_air_transport(cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3);
    diagnostic.vapor_diffusivity_m2_s = air.vapor_diffusivity;
    diagnostic.thermal_conductivity_w_m_k = air.thermal_conductivity;
    diagnostic.kinematic_viscosity_m2_s = air.kinematic_viscosity;
    diagnostic.schmidt_one_third = air.schmidt_one_third;
    diagnostic.density_factor = air.density_factor;
    for (double value : {diagnostic.saturation_ice_per_kg, diagnostic.vapor_diffusivity_m2_s,
                         diagnostic.thermal_conductivity_w_m_k, diagnostic.kinematic_viscosity_m2_s,
                         diagnostic.schmidt_one_third, diagnostic.density_factor})
    {
        if (!positive_finite(value))
        {
            return reject(ColdCloudDiffusionFailure::InvalidSource,
                "coupled cold cloud diffusion air or saturation source is unrepresentable", error, failure);
        }
    }
    if (!std::isfinite(diagnostic.saturation_ratio_ice) || diagnostic.saturation_ratio_ice < 0.0 ||
        !std::isfinite(diagnostic.saturation_temperature_derivative_per_kg_k) ||
        diagnostic.saturation_temperature_derivative_per_kg_k < 0.0)
    {
        return reject(ColdCloudDiffusionFailure::InvalidSource,
            "coupled cold cloud diffusion saturation ratio or derivative is unrepresentable", error, failure);
    }
    const double thermal_denominator = air.thermal_conductivity * microphysics_constants::R_v *
        cell.temperature_k * cell.temperature_k;
    diagnostic.diffusion_resistance_m_s_per_kg =
        microphysics_constants::L_s * microphysics_constants::L_s / thermal_denominator +
        1.0 / (cell.dry_density_kg_m3 * diagnostic.saturation_ice_per_kg * air.vapor_diffusivity);
    diagnostic.riming_feedback = microphysics_constants::L_s * microphysics_constants::L_f /
        thermal_denominator / diagnostic.diffusion_resistance_m_s_per_kg;
    if (!positive_finite(diagnostic.diffusion_resistance_m_s_per_kg) ||
        !positive_finite(diagnostic.riming_feedback))
    {
        return reject(ColdCloudDiffusionFailure::InvalidSource,
            "coupled cold cloud diffusion resistance or riming feedback is unrepresentable", error, failure);
    }
    constexpr double pi_times_two = 6.283185307179586476925;
    constexpr double ventilation_a = 0.78;
    constexpr double ventilation_b = 0.30;
    double ambient_coefficient = 0.0;
    for (std::size_t index = 0; index < species.size(); ++index)
    {
        const auto& initial = cell.initial.frozen[index];
        auto& channel = candidate.channels_[index];
        channel.species = diffusion_species[index];
        if (initial.mass_per_kg == 0.0)
        {
            continue;
        }
        const auto defaults = current_nimbus_triple_sedimentation_parameters(species[index]);
        const double alpha = candidate.collection_producer_.source_alpha()[index];
        double lambda = candidate.collection_producer_.source_lambda_per_m()[index];
        if (index == 0)
        {
            // Snow's operational thermal closure uses its spherical 100 kg/m3 equivalent.
            constexpr double spherical_mass_coefficient = 3.14159265358979323846 / 6.0 * 100.0;
            lambda = std::exp((std::log(spherical_mass_coefficient) +
                std::log(initial.number_per_kg) + std::log(alpha + 1.0) +
                std::log(alpha + 2.0) + std::log(alpha + 3.0) -
                std::log(initial.mass_per_kg)) / 3.0);
        }
        if (!positive_finite(lambda))
        {
            return reject(ColdCloudDiffusionFailure::InvalidSource,
                "coupled cold cloud diffusion thermal slope is unrepresentable", error, failure);
        }
        diagnostic.thermal_lambda_per_m[index] = lambda;
        const double first = normalized_moment(alpha, lambda, 1.0);
        const double ventilated = normalized_moment(alpha, lambda, 1.5 + 0.5 * defaults.fall_exponent);
        diagnostic.ventilation_m[index] = ventilation_a * first +
            ventilation_b * air.schmidt_one_third *
                std::sqrt(air.density_factor * defaults.fall_coefficient / air.kinematic_viscosity) * ventilated;
        const double capacitance = index == 0 ? 0.5 : 1.0;
        diagnostic.ambient_rate_coefficient_per_s[index] = capacitance * pi_times_two * initial.number_per_kg *
            diagnostic.ventilation_m[index] /
            diagnostic.diffusion_resistance_m_s_per_kg / diagnostic.saturation_ice_per_kg;
        if (!positive_finite(first) || !positive_finite(ventilated) ||
            !positive_finite(diagnostic.ventilation_m[index]) ||
            !positive_finite(diagnostic.ambient_rate_coefficient_per_s[index]))
        {
            return reject(ColdCloudDiffusionFailure::InvalidSource,
                "coupled cold cloud diffusion normalized ventilation source is unrepresentable", error, failure);
        }
        ambient_coefficient += diagnostic.ambient_rate_coefficient_per_s[index];
        channel.environmental_mass_increment_per_kg = duration_s * capacitance * pi_times_two *
            (diagnostic.saturation_ratio_ice - 1.0) * initial.number_per_kg *
            diagnostic.ventilation_m[index] / diagnostic.diffusion_resistance_m_s_per_kg;
        // Eliminating particle surface temperature cancels capacitance in the riming term.
        channel.riming_mass_increment_per_kg = -diagnostic.riming_feedback *
            candidate.collection_producer_.channels()[index].admitted_mass_increment_per_kg;
        channel.raw_mass_increment_per_kg = channel.environmental_mass_increment_per_kg +
            channel.riming_mass_increment_per_kg;
        channel.admitted_mass_increment_per_kg = channel.raw_mass_increment_per_kg;
        if (channel.raw_mass_increment_per_kg < 0.0)
        {
            channel.admitted_number_increment_per_kg = initial.number_per_kg *
                (channel.raw_mass_increment_per_kg / initial.mass_per_kg);
        }
        if (!std::isfinite(channel.environmental_mass_increment_per_kg) ||
            !std::isfinite(channel.riming_mass_increment_per_kg) ||
            !std::isfinite(channel.raw_mass_increment_per_kg) ||
            !std::isfinite(channel.admitted_number_increment_per_kg))
        {
            return reject(ColdCloudDiffusionFailure::IntervalRejected,
                "coupled cold cloud diffusion uncapped interval is unrepresentable", error, failure);
        }
    }
    diagnostic.relaxation_number = duration_s * ambient_coefficient *
        (1.0 + diagnostic.saturation_temperature_derivative_per_kg_k *
            microphysics_constants::L_s / microphysics_constants::cp);
    if (!std::isfinite(diagnostic.relaxation_number) || diagnostic.relaxation_number < 0.0 ||
        diagnostic.relaxation_number > 1.0)
    {
        return reject(ColdCloudDiffusionFailure::IntervalRejected,
            "coupled cold cloud diffusion interval exceeds its linear relaxation bound", error, failure);
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

bool evaluate_exact_triple_cold_cloud_diffusion_cell(
    const ColdParameters& parameters, const ColdCloudDiffusionCell& cell,
    double duration_s, ColdCloudDiffusionResult& result, std::string& error,
    std::size_t capacity, ColdCloudDiffusionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    ColdCloudDiffusionProducer producer;
    if (!produce_exact_cold_cloud_diffusion_admission(parameters, cell, duration_s,
            producer, error, capacity, failure, spectrum_domain))
    {
        return false;
    }
    return phase_endpoint(producer, result, error, failure);
}

bool apply_exact_triple_cold_cloud_diffusion_cell(
    const ColdParameters& parameters, const ColdCloudDiffusionCell& cell,
    double duration_s, const ColdCloudDiffusionProducer& producer,
    ColdCloudDiffusionResult& result, std::string& error,
    std::size_t capacity, ColdCloudDiffusionFailure* failure, ColdSpectrumDomain spectrum_domain)
{
    if (!valid_cold_spectrum_domain(spectrum_domain))
    {
        return reject(ColdCloudDiffusionFailure::InvalidContext,
            "coupled cold cloud diffusion spectrum domain is invalid", error, failure);
    }
    if (!producer.valid_ || !same_source(producer.source(), cell) ||
        !same_values(producer.collection_producer().parameters(), parameter_values(parameters)) ||
        producer.collection_producer().hail_processes() != parameters.hail_processes ||
        producer.spectrum_domain() != spectrum_domain ||
        producer.collection_producer().spectrum_domain() != spectrum_domain ||
        !same_value(producer.duration_s(), duration_s))
    {
        return reject(ColdCloudDiffusionFailure::InvalidProducer,
            "coupled cold cloud diffusion capture differs from its exact source or context", error, failure);
    }
    ColdCloudDiffusionProducer expected;
    if (!produce_exact_cold_cloud_diffusion_admission(parameters, cell, duration_s,
            expected, error, capacity, failure, spectrum_domain))
    {
        return false;
    }
    if (!same_capture(producer, expected))
    {
        return reject(ColdCloudDiffusionFailure::InvalidProducer,
            "coupled cold cloud diffusion capture differs from its collection, thermal rates or admission",
            error, failure);
    }
    return phase_endpoint(expected, result, error, failure);
}

} // namespace tmv::milbrandt_detail
