#include "triple_ice_graupel_conversion.hpp"

#include "cold.hpp"

#include <algorithm>
#include <cmath>

namespace tmv::milbrandt_detail
{
namespace
{

constexpr double graupel_embryo_mass_kg = 1.6e-10;
constexpr double graupel_mass_coefficient = 209.439510239;

bool reject(IceGraupelConversionFailure kind, const char* message,
            std::string& error, IceGraupelConversionFailure* failure)
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

bool produce_exact_ice_graupel_conversion_admission(
    const ColdRimingProducer& riming_source, IceGraupelConversionProducer& producer,
    std::string& error, std::size_t capacity, IceGraupelConversionFailure* failure)
{
    if (failure)
        *failure = IceGraupelConversionFailure::None;
    if (capacity != 1)
        return reject(IceGraupelConversionFailure::Capacity,
            "ice-to-graupel conversion requires exactly one channel record", error, failure);
    const auto source_capacity = (riming_source.homogeneous_freezing_enabled() ? 5u : 4u)
        + (riming_source.rain_collection_enabled() ? 4u : 0u) + (riming_source.rain_collision_enabled() ? 3u : 0u) + (riming_source.wet_growth_enabled() ? 1u : 0u);
    if (riming_source.event_count() != source_capacity)
        return reject(IceGraupelConversionFailure::InvalidProducer,
            "ice-to-graupel conversion has no valid source-riming capture", error, failure);
    std::string regeneration_error;
    if (!validate_exact_cold_riming_admission(parameters(riming_source), riming_source.source(),
            riming_source.duration_s(), riming_source, regeneration_error, source_capacity, nullptr,
            riming_source.spectrum_domain(), riming_source.homogeneous_freezing_enabled(),
            riming_source.rain_collection_enabled(), riming_source.rain_collision_enabled(), riming_source.wet_hail_collection_enabled()))
        return reject(IceGraupelConversionFailure::InvalidProducer,
            "ice-to-graupel conversion source-riming capture differs from its original context or transfers",
            error, failure);

    const auto& ice = riming_source.source().initial.frozen[0];
    const auto& growth = riming_source.channels()[0];
    const double mass = 2.0 * std::max(
        growth.collection_mass_per_kg - std::max(growth.diffusion_mass_per_kg, 0.0), 0.0);
    const double old_mass_loss = mass + std::max(-growth.diffusion_mass_per_kg, 0.0);
    if (!std::isfinite(mass) || !std::isfinite(old_mass_loss) || old_mass_loss > ice.mass_per_kg)
        return reject(IceGraupelConversionFailure::IntervalRejected,
            "ice-to-graupel conversion and sublimation overdraw the original ice mass", error, failure);

    IceGraupelConversionProducer candidate;
    candidate.riming_source_ = riming_source;
    candidate.raw_mass_transfer_per_kg_ = mass;
    candidate.admitted_mass_transfer_per_kg_ = mass;
    if (mass > 0.0)
    {
        const double fraction = mass / ice.mass_per_kg;
        candidate.donor_number_loss_per_kg_ = ice.number_per_kg * fraction;
        candidate.recipient_number_gain_per_kg_ = mass / graupel_embryo_mass_kg;
        const double donor_z = ice.reflectivity_per_kg * fraction;
        const double recipient_z = (20.0 * graupel_embryo_mass_kg) * mass /
            (graupel_mass_coefficient * graupel_mass_coefficient);
        for (double value : {fraction, candidate.donor_number_loss_per_kg_,
                             candidate.recipient_number_gain_per_kg_, donor_z, recipient_z})
        {
            if (!std::isfinite(value) || value <= 0.0)
                return reject(IceGraupelConversionFailure::IntervalRejected,
                    "ice-to-graupel conversion nonzero B1/B3/B4 transfer is unrepresentable", error, failure);
        }
        candidate.mass_increment_per_kg_[0] = -mass;
        candidate.mass_increment_per_kg_[2] = mass;
        candidate.number_increment_per_kg_[0] = -candidate.donor_number_loss_per_kg_;
        candidate.number_increment_per_kg_[2] = candidate.recipient_number_gain_per_kg_;
        candidate.reflectivity_increment_per_kg_[0] = -donor_z;
        candidate.reflectivity_increment_per_kg_[2] = recipient_z;
    }
    const double old_number_loss = candidate.donor_number_loss_per_kg_ +
        std::max(-growth.diffusion_number_per_kg, 0.0);
    if (!std::isfinite(old_number_loss) || old_number_loss > ice.number_per_kg)
        return reject(IceGraupelConversionFailure::IntervalRejected,
            "ice-to-graupel conversion and sublimation overdraw the original ice number", error, failure);
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
