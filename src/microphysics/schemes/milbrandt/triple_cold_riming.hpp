#pragma once

#include "triple_cold_cloud_diffusion.hpp"
#include "triple_cold_state.hpp"
#include "triple_hail_wet_growth.hpp"
#include "triple_ice_cloud_collection.hpp"
#include "triple_homogeneous_cloud_freezing.hpp"
#include "triple_rain_collection.hpp"
#include "triple_rain_collision.hpp"

namespace tmv::milbrandt_detail
{

bool validate_cold_riming_source_population(
    const FrozenCollectionPopulation& population, TripleMomentSpecies species,
    ColdSpectrumDomain spectrum_domain, std::string& error);

struct ColdRimingChannel
{
    double raw_collection_mass_per_kg = 0.0;
    double raw_collection_number_per_kg = 0.0;
    double collection_mass_per_kg = 0.0;
    double collection_number_per_kg = 0.0;
    double environmental_diffusion_mass_per_kg = 0.0;
    double riming_diffusion_mass_per_kg = 0.0;
    double diffusion_mass_per_kg = 0.0;
    double diffusion_number_per_kg = 0.0;
    double splinter_embryo_mass_per_kg = 0.0;
    double splinter_number_per_kg = 0.0;
    double splinter_reflectivity_per_kg = 0.0;
};

enum class ColdRimingFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

class ColdRimingProducer;
struct ColdRimingResult;

bool produce_exact_cold_riming_admission(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdRimingProducer&, std::string& error,
    std::size_t capacity = 4, ColdRimingFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200,
    bool homogeneous_freezing = false, bool rain_collection = false, bool rain_collision = false,
    bool wet_hail_collection = false);

bool validate_exact_cold_riming_admission(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    const ColdRimingProducer&, std::string& error,
    std::size_t capacity = 4, ColdRimingFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200,
    bool homogeneous_freezing = false, bool rain_collection = false, bool rain_collision = false,
    bool wet_hail_collection = false);

bool evaluate_exact_triple_cold_riming_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdRimingResult&, std::string& error,
    std::size_t capacity = 4, ColdRimingFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200,
    bool homogeneous_freezing = false, bool rain_collection = false, bool rain_collision = false,
    bool wet_hail_collection = false);

bool apply_exact_triple_cold_riming_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    const ColdRimingProducer&, ColdRimingResult&, std::string& error,
    std::size_t capacity = 4, ColdRimingFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200,
    bool homogeneous_freezing = false, bool rain_collection = false, bool rain_collision = false,
    bool wet_hail_collection = false);

bool produce_exact_wet_cold_riming_admission(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdRimingProducer&, FrozenCollectionProducer&, std::string& error,
    std::size_t capacity = 13, ColdRimingFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::GraupelAndHail512);

class ColdRimingProducer
{
public:
    const ColdRimingCell& source() const noexcept
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
    bool homogeneous_freezing_enabled() const noexcept
    {
        return homogeneous_freezing_enabled_;
    }
    bool rain_collection_enabled() const noexcept { return rain_collection_enabled_; }
    const RainCollectionProducer& rain_collection() const noexcept { return rain_collection_; }
    bool wet_hail_collection_enabled() const noexcept { return wet_hail_collection_enabled_; }
    bool wet_growth_enabled() const noexcept { return wet_growth_enabled_; }
    const HailWetGrowthProducer& wet_growth() const noexcept { return wet_growth_; }
    double riming_feedback() const noexcept { return riming_feedback_; }
    bool rain_collision_enabled() const noexcept { return rain_collision_enabled_; }
    const RainCollisionProducer& rain_collision() const noexcept { return rain_collision_; }
    const RainCollisionIncrements& rain_collision_increments() const noexcept { return rain_collision_increments_; }
    const RainCollectionChannel& admitted_rain_channel(std::size_t index) const noexcept
    {
        return rain_collision_enabled_ && index < 3 ? rain_collision_.channels()[index].rain : rain_collection_.channels()[index];
    }
    const HomogeneousCloudFreezingProducer& homogeneous_freezing() const noexcept
    {
        return homogeneous_freezing_;
    }
    const std::array<double, 3>& homogeneous_increments_per_kg() const noexcept
    {
        return homogeneous_increments_per_kg_;
    }
    double duration_s() const noexcept
    {
        return duration_s_;
    }
    double cloud_admission_factor() const noexcept
    {
        return cloud_admission_factor_;
    }
    double relaxation_number() const noexcept
    {
        return relaxation_number_;
    }
    const std::array<ColdRimingChannel, 4>& channels() const noexcept
    {
        return channels_;
    }
    std::size_t event_count() const noexcept
    {
        return valid_ ? channels_.size() + (homogeneous_freezing_enabled_ ? 1 : 0) + (rain_collection_enabled_ ? 4 : 0) + (rain_collision_enabled_ ? 3 : 0) + (wet_growth_enabled_ ? 1 : 0) : 0;
    }

private:
    friend bool produce_exact_cold_riming_admission(
        const ColdParameters&, const ColdRimingCell&, double,
        ColdRimingProducer&, std::string&, std::size_t, ColdRimingFailure*, ColdSpectrumDomain, bool, bool, bool, bool);

    friend bool produce_exact_wet_cold_riming_admission(
        const ColdParameters&, const ColdRimingCell&, double, ColdRimingProducer&, FrozenCollectionProducer&,
        std::string&, std::size_t, ColdRimingFailure*, ColdSpectrumDomain);
    bool wet_hail_collection_enabled_ = false;
    bool wet_growth_enabled_ = false;
    HailWetGrowthProducer wet_growth_;
    double riming_feedback_ = 0.0;
    ColdRimingCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    bool homogeneous_freezing_enabled_ = false;
    HomogeneousCloudFreezingProducer homogeneous_freezing_;
    bool rain_collection_enabled_ = false;
    RainCollectionProducer rain_collection_;
    bool rain_collision_enabled_ = false;
    RainCollisionProducer rain_collision_;
    RainCollisionIncrements rain_collision_increments_;
    std::array<double, 3> homogeneous_increments_per_kg_{};
    double duration_s_ = 0.0;
    double cloud_admission_factor_ = 1.0;
    double relaxation_number_ = 0.0;
    std::array<ColdRimingChannel, 4> channels_{};
    bool valid_ = false;
};

struct ColdRimingResult
{
    ColdRimingState unrounded;
    float rounded_vapor_mass_per_kg = 0.0f;
    float rounded_cloud_mass_per_kg = 0.0f;
    float rounded_cloud_number_per_kg = 0.0f;
    std::array<FrozenCollectionEndpoint, 4> rounded{};
    std::array<double, 4> mass_increment_per_kg{};
    std::array<double, 4> number_increment_per_kg{};
    std::array<double, 4> reflectivity_increment_per_kg{};
    double vapor_mass_increment_per_kg = 0.0;
    double cloud_mass_increment_per_kg = 0.0;
    double cloud_number_increment_per_kg = 0.0;
    double temperature_increment_k = 0.0;
    FrozenCollectionEndpoint rounded_rain;
    double rain_mass_increment_per_kg = 0.0;
    double rain_number_increment_per_kg = 0.0;
    double rain_reflectivity_increment_per_kg = 0.0;
    ColdRimingProducer producer;
};

} // namespace tmv::milbrandt_detail
