#pragma once

#include "triple_frozen_collection.hpp"

namespace tmv::milbrandt_detail
{

struct IceCloudCollectionState
{
    double cloud_mass_per_kg = 0.0;
    double cloud_number_per_kg = 0.0;
    FrozenCollectionPopulation ice;
};

struct IceCloudCollectionCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    IceCloudCollectionState initial;
};

enum class IceCloudCollectionFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

struct IceCloudCollectionAcceptedChannel
{
    double raw_mass_increment_per_kg = 0.0;
    double raw_number_increment_per_kg = 0.0;
    double admitted_mass_increment_per_kg = 0.0;
    double admitted_number_increment_per_kg = 0.0;
    double donor_admission_factor = 1.0;
};

class IceCloudCollectionProducer;
struct IceCloudCollectionResult;

bool produce_exact_ice_cloud_collection_admission(
    const ColdParameters& parameters, const IceCloudCollectionCell& cell,
    double duration_s, IceCloudCollectionProducer& producer, std::string& error,
    std::size_t capacity = 1, IceCloudCollectionFailure* failure = nullptr);

bool evaluate_exact_triple_ice_cloud_collection_cell(
    const ColdParameters& parameters, const IceCloudCollectionCell& cell,
    double duration_s, IceCloudCollectionResult& result, std::string& error,
    std::size_t capacity = 1, IceCloudCollectionFailure* failure = nullptr);

bool apply_exact_triple_ice_cloud_collection_cell(
    const ColdParameters& parameters, const IceCloudCollectionCell& cell,
    double duration_s, const IceCloudCollectionProducer& producer,
    IceCloudCollectionResult& result, std::string& error,
    std::size_t capacity = 1, IceCloudCollectionFailure* failure = nullptr);

class IceCloudCollectionProducer
{
public:
    const IceCloudCollectionCell& source() const noexcept
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

    double cloud_inverse_lambda_m() const noexcept
    {
        return cloud_inverse_lambda_m_;
    }

    double source_alpha() const noexcept
    {
        return source_alpha_;
    }

    double source_lambda_per_m() const noexcept
    {
        return source_lambda_per_m_;
    }

    const IceCloudCollectionAcceptedChannel& channel() const noexcept
    {
        return channel_;
    }

    std::size_t event_count() const noexcept
    {
        return valid_ ? 1 : 0;
    }

private:
    friend bool produce_exact_ice_cloud_collection_admission(
        const ColdParameters&, const IceCloudCollectionCell&, double,
        IceCloudCollectionProducer&, std::string&, std::size_t,
        IceCloudCollectionFailure*);
    friend bool apply_exact_triple_ice_cloud_collection_cell(
        const ColdParameters&, const IceCloudCollectionCell&, double,
        const IceCloudCollectionProducer&, IceCloudCollectionResult&,
        std::string&, std::size_t, IceCloudCollectionFailure*);

    IceCloudCollectionCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    double duration_s_ = 0.0;
    double cloud_inverse_lambda_m_ = 0.0;
    double source_alpha_ = 0.0;
    double source_lambda_per_m_ = 0.0;
    IceCloudCollectionAcceptedChannel channel_{};
    bool valid_ = false;
};

struct IceCloudCollectionResult
{
    IceCloudCollectionState unrounded;
    float rounded_cloud_mass_per_kg = 0.0f;
    float rounded_cloud_number_per_kg = 0.0f;
    FrozenCollectionEndpoint rounded_ice;
    double cloud_mass_increment_per_kg = 0.0;
    double cloud_number_increment_per_kg = 0.0;
    double ice_mass_increment_per_kg = 0.0;
    double ice_reflectivity_increment_per_kg = 0.0;
    double temperature_increment_k = 0.0;
    IceCloudCollectionProducer producer;
};

} // namespace tmv::milbrandt_detail
