#pragma once

#include "triple_cold_cloud_collection.hpp"

namespace tmv::milbrandt_detail
{

struct ColdCloudDiffusionCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double vapor_mass_per_kg = 0.0;
    ColdCloudCollectionState initial;
};

enum class ColdCloudDiffusionSpecies
{
    Snow,
    Graupel,
    Hail
};

struct ColdCloudDiffusionAcceptedChannel
{
    ColdCloudDiffusionSpecies species = ColdCloudDiffusionSpecies::Snow;
    double environmental_mass_increment_per_kg = 0.0;
    double riming_mass_increment_per_kg = 0.0;
    double raw_mass_increment_per_kg = 0.0;
    double admitted_mass_increment_per_kg = 0.0;
    double admitted_number_increment_per_kg = 0.0;
};

struct ColdCloudDiffusionThermodynamics
{
    double saturation_ice_per_kg = 0.0;
    double saturation_temperature_derivative_per_kg_k = 0.0;
    double saturation_ratio_ice = 0.0;
    double vapor_diffusivity_m2_s = 0.0;
    double thermal_conductivity_w_m_k = 0.0;
    double kinematic_viscosity_m2_s = 0.0;
    double schmidt_one_third = 0.0;
    double density_factor = 0.0;
    double diffusion_resistance_m_s_per_kg = 0.0;
    double riming_feedback = 0.0;
    std::array<double, 3> thermal_lambda_per_m{};
    std::array<double, 3> ventilation_m{};
    std::array<double, 3> ambient_rate_coefficient_per_s{};
    double relaxation_number = 0.0;
};

enum class ColdCloudDiffusionFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

class ColdCloudDiffusionProducer;
struct ColdCloudDiffusionResult;

// Source admission does not assert realizability of the coupled primary endpoint.
bool produce_exact_cold_cloud_diffusion_admission(
    const ColdParameters& parameters, const ColdCloudDiffusionCell& cell,
    double duration_s, ColdCloudDiffusionProducer& producer, std::string& error,
    std::size_t capacity = 3, ColdCloudDiffusionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool evaluate_exact_triple_cold_cloud_diffusion_cell(
    const ColdParameters& parameters, const ColdCloudDiffusionCell& cell,
    double duration_s, ColdCloudDiffusionResult& result, std::string& error,
    std::size_t capacity = 3, ColdCloudDiffusionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool apply_exact_triple_cold_cloud_diffusion_cell(
    const ColdParameters& parameters, const ColdCloudDiffusionCell& cell,
    double duration_s, const ColdCloudDiffusionProducer& producer,
    ColdCloudDiffusionResult& result, std::string& error,
    std::size_t capacity = 3, ColdCloudDiffusionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

class ColdCloudDiffusionProducer
{
public:
    ColdSpectrumDomain spectrum_domain() const noexcept
    {
        return spectrum_domain_;
    }
    const ColdCloudDiffusionCell& source() const noexcept
    {
        return source_;
    }

    const ColdCloudCollectionProducer& collection_producer() const noexcept
    {
        return collection_producer_;
    }

    const ColdCloudDiffusionThermodynamics& thermodynamics() const noexcept
    {
        return thermodynamics_;
    }

    const std::array<ColdCloudDiffusionAcceptedChannel, 3>& channels() const noexcept
    {
        return channels_;
    }

    double duration_s() const noexcept
    {
        return duration_s_;
    }

    std::size_t event_count() const noexcept
    {
        return valid_ ? channels_.size() : 0;
    }

private:
    friend bool produce_exact_cold_cloud_diffusion_admission(
        const ColdParameters&, const ColdCloudDiffusionCell&, double,
        ColdCloudDiffusionProducer&, std::string&, std::size_t,
        ColdCloudDiffusionFailure*, ColdSpectrumDomain);
    friend bool evaluate_exact_triple_cold_cloud_diffusion_cell(
        const ColdParameters&, const ColdCloudDiffusionCell&, double,
        ColdCloudDiffusionResult&, std::string&, std::size_t,
        ColdCloudDiffusionFailure*, ColdSpectrumDomain);
    friend bool apply_exact_triple_cold_cloud_diffusion_cell(
        const ColdParameters&, const ColdCloudDiffusionCell&, double,
        const ColdCloudDiffusionProducer&, ColdCloudDiffusionResult&,
        std::string&, std::size_t, ColdCloudDiffusionFailure*, ColdSpectrumDomain);

    ColdCloudDiffusionCell source_{};
    ColdCloudCollectionProducer collection_producer_{};
    ColdCloudDiffusionThermodynamics thermodynamics_{};
    std::array<ColdCloudDiffusionAcceptedChannel, 3> channels_{};
    double duration_s_ = 0.0;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    bool valid_ = false;
};

struct ColdCloudDiffusionResult
{
    ColdCloudCollectionState unrounded;
    double vapor_mass_per_kg = 0.0;
    float rounded_vapor_mass_per_kg = 0.0f;
    float rounded_cloud_mass_per_kg = 0.0f;
    float rounded_cloud_number_per_kg = 0.0f;
    std::array<FrozenCollectionEndpoint, 3> rounded{};
    std::array<double, 3> mass_increment_per_kg{};
    std::array<double, 3> number_increment_per_kg{};
    std::array<double, 3> reflectivity_increment_per_kg{};
    double vapor_mass_increment_per_kg = 0.0;
    double cloud_mass_increment_per_kg = 0.0;
    double cloud_number_increment_per_kg = 0.0;
    double temperature_increment_k = 0.0;
    ColdCloudDiffusionProducer producer;
};

} // namespace tmv::milbrandt_detail
