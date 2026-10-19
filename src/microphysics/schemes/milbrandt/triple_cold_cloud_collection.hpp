#pragma once

#include "triple_frozen_collection.hpp"
#include "triple_cold_spectrum_domain.hpp"

namespace tmv::milbrandt_detail
{

struct ColdCloudCollectionState
{
    double cloud_mass_per_kg = 0.0;
    double cloud_number_per_kg = 0.0;
    std::array<FrozenCollectionPopulation, 3> frozen{};
};

struct ColdCloudCollectionCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    ColdCloudCollectionState initial;
};

enum class ColdCloudCollectionChannel
{
    CloudToSnow,
    CloudToGraupel,
    CloudToHail
};

struct ColdCloudCollectionAcceptedChannel
{
    ColdCloudCollectionChannel channel = ColdCloudCollectionChannel::CloudToSnow;
    double raw_mass_increment_per_kg = 0.0;
    double raw_number_increment_per_kg = 0.0;
    double admitted_mass_increment_per_kg = 0.0;
    double admitted_number_increment_per_kg = 0.0;
    double donor_admission_factor = 1.0;
};

enum class ColdCloudCollectionFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

class ColdCloudCollectionProducer;
struct ColdCloudCollectionResult;

// Admission captures the isolated collision operator, not its remaining cold couplings.
bool produce_exact_cold_cloud_collection_admission(
    const ColdParameters& parameters, const ColdCloudCollectionCell& cell,
    double duration_s, ColdCloudCollectionProducer& producer, std::string& error,
    std::size_t capacity = 3, ColdCloudCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

bool evaluate_exact_triple_cold_cloud_collection_cell(
    const ColdParameters& parameters, const ColdCloudCollectionCell& cell,
    double duration_s, ColdCloudCollectionResult& result, std::string& error,
    std::size_t capacity = 3, ColdCloudCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

bool apply_exact_triple_cold_cloud_collection_cell(
    const ColdParameters& parameters, const ColdCloudCollectionCell& cell,
    double duration_s, const ColdCloudCollectionProducer& producer,
    ColdCloudCollectionResult& result, std::string& error,
    std::size_t capacity = 3, ColdCloudCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

class ColdCloudCollectionProducer
{
public:
    ColdSpectrumDomain spectrum_domain() const noexcept
    {
        return spectrum_domain_;
    }
    const ColdCloudCollectionCell& source() const noexcept
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

    bool wet_hail_collection() const noexcept { return wet_hail_collection_; }

    double duration_s() const noexcept
    {
        return duration_s_;
    }

    double cloud_inverse_lambda_m() const noexcept
    {
        return cloud_inverse_lambda_m_;
    }

    const std::array<double, 3>& source_alpha() const noexcept
    {
        return source_alpha_;
    }

    const std::array<double, 3>& source_lambda_per_m() const noexcept
    {
        return source_lambda_per_m_;
    }

    const std::array<double, 3>& mass_weighted_speed_m_s() const noexcept
    {
        return mass_weighted_speed_m_s_;
    }

    const std::array<double, 3>& efficiencies() const noexcept
    {
        return efficiencies_;
    }

    const std::array<ColdCloudCollectionAcceptedChannel, 3>& channels() const noexcept
    {
        return channels_;
    }

    std::size_t event_count() const noexcept
    {
        return valid_ ? channels_.size() : 0;
    }

private:
    friend bool produce_exact_cold_cloud_collection_admission(
        const ColdParameters&, const ColdCloudCollectionCell&, double,
        ColdCloudCollectionProducer&, std::string&, std::size_t,
        ColdCloudCollectionFailure*, ColdSpectrumDomain, bool);
    friend bool evaluate_exact_triple_cold_cloud_collection_cell(
        const ColdParameters&, const ColdCloudCollectionCell&, double,
        ColdCloudCollectionResult&, std::string&, std::size_t,
        ColdCloudCollectionFailure*, ColdSpectrumDomain, bool);
    friend bool apply_exact_triple_cold_cloud_collection_cell(
        const ColdParameters&, const ColdCloudCollectionCell&, double,
        const ColdCloudCollectionProducer&, ColdCloudCollectionResult&,
        std::string&, std::size_t, ColdCloudCollectionFailure*, ColdSpectrumDomain, bool);

    ColdCloudCollectionCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    bool wet_hail_collection_ = false;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    double duration_s_ = 0.0;
    double cloud_inverse_lambda_m_ = 0.0;
    std::array<double, 3> source_alpha_{};
    std::array<double, 3> source_lambda_per_m_{};
    std::array<double, 3> mass_weighted_speed_m_s_{};
    std::array<double, 3> efficiencies_{};
    std::array<ColdCloudCollectionAcceptedChannel, 3> channels_{};
    bool valid_ = false;
};

struct ColdCloudCollectionResult
{
    ColdCloudCollectionState unrounded;
    float rounded_cloud_mass_per_kg = 0.0f;
    float rounded_cloud_number_per_kg = 0.0f;
    std::array<FrozenCollectionEndpoint, 3> rounded{};
    std::array<double, 3> mass_increment_per_kg{};
    std::array<double, 3> reflectivity_increment_per_kg{};
    double cloud_mass_increment_per_kg = 0.0;
    double cloud_number_increment_per_kg = 0.0;
    double temperature_increment_k = 0.0;
    ColdCloudCollectionProducer producer;
};

} // namespace tmv::milbrandt_detail
