#include "triple_homogeneous_cloud_freezing.hpp"

#include "cold.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace tmv::milbrandt_detail
{
namespace
{

bool reject(HomogeneousCloudFreezingFailure kind, const char* message,
            std::string& error, HomogeneousCloudFreezingFailure* failure)
{
    error = message;
    if (failure != nullptr)
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

bool same(double a, double b) noexcept
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

template <std::size_t Count>
bool same(const std::array<double, Count>& a, const std::array<double, Count>& b) noexcept
{
    for (std::size_t index = 0; index < Count; ++index)
    {
        if (!same(a[index], b[index]))
        {
            return false;
        }
    }
    return true;
}

std::array<double, 5> source_values(const HomogeneousCloudFreezingCell& cell) noexcept
{
    return {cell.temperature_k, cell.pressure_pa, cell.dry_density_kg_m3,
        cell.cloud_mass_per_kg, cell.cloud_number_per_kg};
}

std::array<double, 7> channel_values(const HomogeneousCloudFreezingChannel& channel) noexcept
{
    return {channel.fraction, channel.mean_volume_cm3, channel.log10_rate_per_cm3_s,
        channel.source_mass_squared_per_kg, channel.raw_mass_increment_per_kg,
        channel.raw_number_increment_per_kg, channel.raw_reflectivity_increment_per_kg};
}

bool same_capture(const HomogeneousCloudFreezingProducer& a,
                  const HomogeneousCloudFreezingProducer& b) noexcept
{
    return a.event_count() == b.event_count() && same(source_values(a.source()), source_values(b.source()))
        && same(a.parameters(), b.parameters()) && a.hail_processes() == b.hail_processes()
        && same(a.duration_s(), b.duration_s()) && same(channel_values(a.channel()), channel_values(b.channel()));
}

} // namespace

bool produce_exact_homogeneous_cloud_freezing_admission(
    const ColdParameters& parameters, const HomogeneousCloudFreezingCell& cell,
    double duration_s, HomogeneousCloudFreezingProducer& producer, std::string& error,
    std::size_t capacity, HomogeneousCloudFreezingFailure* failure)
{
    if (failure != nullptr)
    {
        *failure = HomogeneousCloudFreezingFailure::None;
    }
    if (capacity != 1)
    {
        return reject(HomogeneousCloudFreezingFailure::Capacity,
            "homogeneous cloud freezing requires exactly one bounded channel record", error, failure);
    }
    const auto captured_parameters = parameter_values(parameters);
    if (!std::isfinite(duration_s) || duration_s <= 0.0
        || !std::isfinite(cell.temperature_k) || cell.temperature_k < 150.0
        || cell.temperature_k >= microphysics_constants::T0
        || !std::isfinite(cell.pressure_pa) || cell.pressure_pa < 100.0 || cell.pressure_pa > 200000.0
        || !std::isfinite(cell.dry_density_kg_m3) || cell.dry_density_kg_m3 <= 0.0
        || captured_parameters != default_cold_parameters)
    {
        return reject(HomogeneousCloudFreezingFailure::InvalidContext,
            "homogeneous cloud freezing context or retained physical parameters differ", error, failure);
    }
    if (!std::isfinite(cell.cloud_mass_per_kg) || cell.cloud_mass_per_kg < 0.0
        || !std::isfinite(cell.cloud_number_per_kg) || cell.cloud_number_per_kg < 0.0
        || (cell.cloud_mass_per_kg > 0.0 && cell.cloud_number_per_kg <= 0.0))
    {
        return reject(HomogeneousCloudFreezingFailure::InvalidSource,
            "homogeneous cloud freezing cloud mass or number is invalid", error, failure);
    }

    HomogeneousCloudFreezingProducer candidate;
    candidate.source_ = cell;
    candidate.parameters_ = captured_parameters;
    candidate.hail_processes_ = parameters.hail_processes;
    candidate.duration_s_ = duration_s;
    auto& channel = candidate.channel_;
    const double tc = cell.temperature_k - microphysics_constants::T0;
    channel.log10_rate_per_cm3_s = std::max(-20.0,
        (((-1.536e-4 * tc - 0.0265) * tc - 1.7439) * tc - 52.6611) * tc - 606.3952);
    if (cell.cloud_mass_per_kg > 0.0)
    {
        int q_exponent = 0;
        int n_exponent = 0;
        const double q_fraction = std::frexp(cell.cloud_mass_per_kg, &q_exponent);
        const double n_fraction = std::frexp(cell.cloud_number_per_kg, &n_exponent);
        constexpr double pi = 3.141592653589793238462643383279502884;
        channel.mean_volume_cm3 = std::ldexp((pi / 6.0 * 1.0e6 / parameters.c_r_)
            * q_fraction / n_fraction, q_exponent - n_exponent);
        // The retained generalized cloud law has M(2m)=3/2 q²/N, not a chosen ice alpha.
        channel.source_mass_squared_per_kg = std::ldexp(1.5 * q_fraction * q_fraction / n_fraction,
            2 * q_exponent - n_exponent);
        if (!std::isfinite(channel.mean_volume_cm3) || channel.mean_volume_cm3 <= 0.0
            || !std::isfinite(channel.source_mass_squared_per_kg) || channel.source_mass_squared_per_kg <= 0.0)
        {
            return reject(HomogeneousCloudFreezingFailure::InvalidSource,
                "homogeneous cloud freezing source volume or mass-squared moment is unrepresentable", error, failure);
        }
    }

    // This source activity cutoff is retained from the operational cloud-freezing path.
    if (cell.cloud_mass_per_kg > 1.0e-14 && tc <= -30.0)
    {
        if (tc < -50.0)
        {
            channel.fraction = 1.0;
        }
        else
        {
            const double log_hazard = channel.log10_rate_per_cm3_s * std::log(10.0)
                + std::log(channel.mean_volume_cm3) + std::log(duration_s);
            if (log_hazard > std::log(std::numeric_limits<double>::max()))
            {
                channel.fraction = 1.0;
            }
            else
            {
                const double hazard = std::exp(log_hazard);
                if (hazard == 0.0)
                {
                    return reject(HomogeneousCloudFreezingFailure::IntervalRejected,
                        "homogeneous cloud freezing positive hazard is unrepresentable", error, failure);
                }
                channel.fraction = -std::expm1(-hazard);
            }
        }
        channel.raw_mass_increment_per_kg = channel.fraction * cell.cloud_mass_per_kg;
        channel.raw_number_increment_per_kg = channel.fraction * cell.cloud_number_per_kg;
        channel.raw_reflectivity_increment_per_kg = channel.fraction * channel.source_mass_squared_per_kg
            / (parameters.c_i_ * parameters.c_i_);
        for (double value : {channel.fraction, channel.raw_mass_increment_per_kg,
                             channel.raw_number_increment_per_kg, channel.raw_reflectivity_increment_per_kg})
        {
            if (!std::isfinite(value) || value <= 0.0)
            {
                return reject(HomogeneousCloudFreezingFailure::IntervalRejected,
                    "homogeneous cloud freezing nonzero q/N/Z transfer is unrepresentable", error, failure);
            }
        }
        if (channel.fraction > 1.0 || channel.raw_mass_increment_per_kg > cell.cloud_mass_per_kg
            || channel.raw_number_increment_per_kg > cell.cloud_number_per_kg)
        {
            return reject(HomogeneousCloudFreezingFailure::IntervalRejected,
                "homogeneous cloud freezing transfer exceeds its original cloud donor", error, failure);
        }
    }
    candidate.valid_ = true;
    producer = candidate;
    error.clear();
    return true;
}

bool validate_exact_homogeneous_cloud_freezing_admission(
    const ColdParameters& parameters, const HomogeneousCloudFreezingCell& cell,
    double duration_s, const HomogeneousCloudFreezingProducer& producer, std::string& error,
    std::size_t capacity, HomogeneousCloudFreezingFailure* failure)
{
    if (failure != nullptr)
    {
        *failure = HomogeneousCloudFreezingFailure::None;
    }
    if (capacity != 1)
    {
        return reject(HomogeneousCloudFreezingFailure::Capacity,
            "homogeneous cloud freezing requires exactly one bounded channel record", error, failure);
    }
    if (producer.event_count() != 1 || !same(producer.parameters(), parameter_values(parameters))
        || producer.hail_processes() != parameters.hail_processes || !same(producer.duration_s(), duration_s)
        || !same(source_values(producer.source()), source_values(cell)))
    {
        return reject(HomogeneousCloudFreezingFailure::InvalidProducer,
            "homogeneous cloud freezing capture differs from its original source context", error, failure);
    }
    HomogeneousCloudFreezingProducer regenerated;
    std::string regeneration_error;
    if (!produce_exact_homogeneous_cloud_freezing_admission(parameters, cell, duration_s,
            regenerated, regeneration_error, capacity, nullptr) || !same_capture(producer, regenerated))
    {
        return reject(HomogeneousCloudFreezingFailure::InvalidProducer,
            "homogeneous cloud freezing capture differs from its exact source transfers", error, failure);
    }
    error.clear();
    return true;
}

} // namespace tmv::milbrandt_detail
