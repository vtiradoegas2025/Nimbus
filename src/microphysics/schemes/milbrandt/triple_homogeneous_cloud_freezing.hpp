#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace tmv::milbrandt_detail
{

struct ColdParameters;

struct HomogeneousCloudFreezingCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double cloud_mass_per_kg = 0.0;
    double cloud_number_per_kg = 0.0;
};

enum class HomogeneousCloudFreezingFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

struct HomogeneousCloudFreezingChannel
{
    double fraction = 0.0;
    double mean_volume_cm3 = 0.0;
    double log10_rate_per_cm3_s = 0.0;
    double source_mass_squared_per_kg = 0.0;
    double raw_mass_increment_per_kg = 0.0;
    double raw_number_increment_per_kg = 0.0;
    double raw_reflectivity_increment_per_kg = 0.0;
};

class HomogeneousCloudFreezingProducer;

bool produce_exact_homogeneous_cloud_freezing_admission(
    const ColdParameters& parameters, const HomogeneousCloudFreezingCell& cell,
    double duration_s, HomogeneousCloudFreezingProducer& producer, std::string& error,
    std::size_t capacity = 1, HomogeneousCloudFreezingFailure* failure = nullptr);

bool validate_exact_homogeneous_cloud_freezing_admission(
    const ColdParameters& parameters, const HomogeneousCloudFreezingCell& cell,
    double duration_s, const HomogeneousCloudFreezingProducer& producer, std::string& error,
    std::size_t capacity = 1, HomogeneousCloudFreezingFailure* failure = nullptr);

class HomogeneousCloudFreezingProducer
{
public:
    const HomogeneousCloudFreezingCell& source() const noexcept
    {
        return source_;
    }

    const std::array<double, 25>& parameters() const noexcept
    {
        return parameters_;
    }

    bool hail_processes() const noexcept
    {
        return hail_processes_;
    }

    double duration_s() const noexcept
    {
        return duration_s_;
    }

    const HomogeneousCloudFreezingChannel& channel() const noexcept
    {
        return channel_;
    }

    std::size_t event_count() const noexcept
    {
        return valid_ ? 1 : 0;
    }

private:
    friend bool produce_exact_homogeneous_cloud_freezing_admission(
        const ColdParameters&, const HomogeneousCloudFreezingCell&, double,
        HomogeneousCloudFreezingProducer&, std::string&, std::size_t,
        HomogeneousCloudFreezingFailure*);

    HomogeneousCloudFreezingCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    double duration_s_ = 0.0;
    HomogeneousCloudFreezingChannel channel_{};
    bool valid_ = false;
};

} // namespace tmv::milbrandt_detail
