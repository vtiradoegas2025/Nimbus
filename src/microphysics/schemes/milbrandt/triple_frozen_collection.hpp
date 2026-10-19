#pragma once

#include "triple_cold_spectrum_domain.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace tmv::milbrandt_detail
{

struct ColdParameters;

struct FrozenCollectionPopulation
{
    double mass_per_kg = 0.0;
    double number_per_kg = 0.0;
    double reflectivity_per_kg = 0.0;
};

struct FrozenCollectionState
{
    std::array<FrozenCollectionPopulation, 4> frozen{};
};

struct FrozenCollectionCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    FrozenCollectionState initial;
};

struct FrozenCollectionEndpoint
{
    float mass_per_kg = 0.0f;
    float number_per_kg = 0.0f;
    float reflectivity_per_kg = 0.0f;
};

struct FrozenCollectionIncrements
{
    std::array<double, 4> mass_increment_per_kg{};
    std::array<double, 4> number_increment_per_kg{};
    std::array<double, 4> reflectivity_increment_per_kg{};
};

enum class FrozenCollectionChannel
{
    IceToSnow,
    IceToGraupel,
    IceToHail,
    SnowToHail,
    SnowSelfCollection
};

struct FrozenCollectionAcceptedChannel
{
    FrozenCollectionChannel channel = FrozenCollectionChannel::IceToSnow;
    double raw_mass_increment_per_kg = 0.0;
    double raw_number_increment_per_kg = 0.0;
    double admitted_mass_increment_per_kg = 0.0;
    double admitted_number_increment_per_kg = 0.0;
    double donor_admission_factor = 1.0;
};

enum class FrozenCollectionFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

class FrozenCollectionProducer;
struct FrozenCollectionResult;

// Transfer admission alone does not assert realizability of the updated moments.
bool produce_exact_frozen_collection_admission(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, FrozenCollectionProducer& producer, std::string& error,
    std::size_t capacity = 5, FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

bool validate_exact_frozen_collection_admission(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, const FrozenCollectionProducer& producer, std::string& error,
    std::size_t capacity = 5, FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

bool evaluate_exact_frozen_collection_increments(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, const FrozenCollectionProducer& producer,
    FrozenCollectionIncrements& increments, std::string& error,
    std::size_t capacity = 5, FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

bool evaluate_exact_triple_frozen_collection_cell(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, FrozenCollectionResult& result, std::string& error,
    std::size_t capacity = 5, FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

bool apply_exact_triple_frozen_collection_cell(
    const ColdParameters& parameters, const FrozenCollectionCell& cell,
    double duration_s, const FrozenCollectionProducer& producer,
    FrozenCollectionResult& result, std::string& error,
    std::size_t capacity = 5, FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200, bool wet_hail_collection = false);

class FrozenCollectionProducer
{
public:
    const FrozenCollectionCell& source() const noexcept
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

    ColdSpectrumDomain spectrum_domain() const noexcept
    {
        return spectrum_domain_;
    }

    bool wet_hail_collection() const noexcept { return wet_hail_collection_; }

    double duration_s() const noexcept
    {
        return duration_s_;
    }

    const std::array<FrozenCollectionAcceptedChannel, 5>& channels() const noexcept
    {
        return channels_;
    }

    const std::array<double, 4>& source_alpha() const noexcept
    {
        return source_alpha_;
    }

    const std::array<double, 4>& source_lambda_per_m() const noexcept
    {
        return source_lambda_per_m_;
    }

    const std::array<double, 4>& mass_weighted_speed_m_s() const noexcept
    {
        return mass_weighted_speed_m_s_;
    }

    std::size_t event_count() const noexcept
    {
        return valid_ ? channels_.size() : 0;
    }

private:
    friend bool produce_exact_frozen_collection_admission(
        const ColdParameters&, const FrozenCollectionCell&, double,
        FrozenCollectionProducer&, std::string&, std::size_t,
        FrozenCollectionFailure*, ColdSpectrumDomain, bool);
    friend bool evaluate_exact_triple_frozen_collection_cell(
        const ColdParameters&, const FrozenCollectionCell&, double,
        FrozenCollectionResult&, std::string&, std::size_t,
        FrozenCollectionFailure*, ColdSpectrumDomain, bool);
    friend bool apply_exact_triple_frozen_collection_cell(
        const ColdParameters&, const FrozenCollectionCell&, double,
        const FrozenCollectionProducer&, FrozenCollectionResult&, std::string&,
        std::size_t, FrozenCollectionFailure*, ColdSpectrumDomain, bool);

    FrozenCollectionCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    bool wet_hail_collection_ = false;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    double duration_s_ = 0.0;
    std::array<FrozenCollectionAcceptedChannel, 5> channels_{};
    std::array<double, 4> source_alpha_{};
    std::array<double, 4> source_lambda_per_m_{};
    std::array<double, 4> mass_weighted_speed_m_s_{};
    bool valid_ = false;
};

struct FrozenCollectionResult
{
    FrozenCollectionState unrounded;
    std::array<FrozenCollectionEndpoint, 4> rounded{};
    std::array<double, 4> mass_increment_per_kg{};
    std::array<double, 4> number_increment_per_kg{};
    std::array<double, 4> reflectivity_increment_per_kg{};
    double temperature_increment_k = 0.0;
    FrozenCollectionProducer producer;
};

} // namespace tmv::milbrandt_detail
