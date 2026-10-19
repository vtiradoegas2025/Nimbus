#include "triple_snow_graupel_conversion.hpp"

#include "cold.hpp"
#include "triple_sedimentation.hpp"

#include <algorithm>
#include <cmath>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr double graupel_mass_coefficient = 209.439510239;

bool reject(SnowGraupelConversionFailure kind, const char* message,
            std::string& error, SnowGraupelConversionFailure* failure)
{
    error = message;
    if (failure)
        *failure = kind;
    return false;
}


ColdParameters parameters(const ColdRimingProducer& riming) noexcept
{
    const auto& p = riming.parameters();
    return {p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
        p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19],
        p[20], p[21], p[22], p[23], p[24], riming.hail_processes()};
}

} // namespace

bool produce_exact_snow_graupel_conversion_admission(
    const ColdRimingProducer& riming_source, SnowGraupelConversionProducer& producer,
    std::string& error, std::size_t capacity, SnowGraupelConversionFailure* failure)
{
    if (failure)
        *failure = SnowGraupelConversionFailure::None;
    if (capacity != 1)
        return reject(SnowGraupelConversionFailure::Capacity,
            "snow-to-graupel conversion requires exactly one channel record", error, failure);
    const auto source_capacity = (riming_source.homogeneous_freezing_enabled() ? 5u : 4u)
        + (riming_source.rain_collection_enabled() ? 4u : 0u) + (riming_source.rain_collision_enabled() ? 3u : 0u) + (riming_source.wet_growth_enabled() ? 1u : 0u);
    if (riming_source.event_count() != source_capacity)
        return reject(SnowGraupelConversionFailure::InvalidProducer,
            "snow-to-graupel conversion has no valid source-riming capture", error, failure);
    std::string regeneration_error;
    if (!validate_exact_cold_riming_admission(parameters(riming_source), riming_source.source(),
            riming_source.duration_s(), riming_source, regeneration_error, source_capacity, nullptr,
            riming_source.spectrum_domain(), riming_source.homogeneous_freezing_enabled(),
            riming_source.rain_collection_enabled(), riming_source.rain_collision_enabled(), riming_source.wet_hail_collection_enabled()))
        return reject(SnowGraupelConversionFailure::InvalidProducer,
            "snow-to-graupel conversion source-riming capture differs from its original context or transfers",
            error, failure);

    const auto& snow = riming_source.source().initial.frozen[1];
    const auto& growth = riming_source.channels()[1];
    const double mass = growth.collection_mass_per_kg > growth.diffusion_mass_per_kg
        ? (4.0 / 3.0) * growth.collection_mass_per_kg : 0.0;
    const double old_mass_loss = mass + std::max(-growth.diffusion_mass_per_kg, 0.0);
    if (!std::isfinite(mass) || !std::isfinite(old_mass_loss) || old_mass_loss > snow.mass_per_kg)
        return reject(SnowGraupelConversionFailure::IntervalRejected,
            "snow-to-graupel conversion and sublimation overdraw the original snow mass", error, failure);

    SnowGraupelConversionProducer candidate;
    candidate.riming_source_ = riming_source;
    candidate.raw_mass_transfer_per_kg_ = mass;
    candidate.admitted_mass_transfer_per_kg_ = mass;
    if (snow.mass_per_kg > 0.0)
    {
        const auto law = current_nimbus_triple_sedimentation_parameters(TripleMomentSpecies::Snow).mass_law;
        if (!diagnose_triple_gamma_shape(snow.mass_per_kg, snow.number_per_kg,
                snow.reflectivity_per_kg, law, candidate.source_alpha_,
                candidate.source_lambda_per_m_, error))
        {
            if (failure)
                *failure = SnowGraupelConversionFailure::InvalidProducer;
            return false;
        }
        const double a = candidate.source_alpha_ + 1.0;
        const double log_ratio = std::lgamma(a + 2.0 * law.exponent) + std::lgamma(a)
            - 2.0 * std::lgamma(a + law.exponent);
        candidate.source_mass_squared_per_kg_ = std::exp(
            2.0 * std::log(snow.mass_per_kg) - std::log(snow.number_per_kg) + log_ratio);
        if (!std::isfinite(candidate.source_mass_squared_per_kg_) ||
            candidate.source_mass_squared_per_kg_ <= 0.0)
            return reject(SnowGraupelConversionFailure::IntervalRejected,
                "snow-to-graupel source mass-squared moment is unrepresentable", error, failure);
    }
    if (mass > 0.0)
    {
        candidate.admitted_fraction_ = mass / snow.mass_per_kg;
        candidate.donor_number_loss_per_kg_ = snow.number_per_kg * candidate.admitted_fraction_;
        const double donor_z = snow.reflectivity_per_kg * candidate.admitted_fraction_;
        // Equal-mass particles map snow M(2d), not snow M(6), into cubic graupel reflectivity.
        const double recipient_z = candidate.admitted_fraction_ * candidate.source_mass_squared_per_kg_ /
            (graupel_mass_coefficient * graupel_mass_coefficient);
        for (double value : {candidate.admitted_fraction_, candidate.donor_number_loss_per_kg_,
                             donor_z, recipient_z})
        {
            if (!std::isfinite(value) || value <= 0.0)
                return reject(SnowGraupelConversionFailure::IntervalRejected,
                    "snow-to-graupel nonzero fractional-spectrum transfer is unrepresentable", error, failure);
        }
        candidate.mass_increment_per_kg_[1] = -mass;
        candidate.mass_increment_per_kg_[2] = mass;
        candidate.number_increment_per_kg_[1] = -candidate.donor_number_loss_per_kg_;
        candidate.number_increment_per_kg_[2] = candidate.donor_number_loss_per_kg_;
        candidate.reflectivity_increment_per_kg_[1] = -donor_z;
        candidate.reflectivity_increment_per_kg_[2] = recipient_z;
    }
    const double old_number_loss = candidate.donor_number_loss_per_kg_ +
        std::max(-growth.diffusion_number_per_kg, 0.0);
    if (!std::isfinite(old_number_loss) || old_number_loss > snow.number_per_kg)
        return reject(SnowGraupelConversionFailure::IntervalRejected,
            "snow-to-graupel conversion and sublimation overdraw the original snow number", error, failure);
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
