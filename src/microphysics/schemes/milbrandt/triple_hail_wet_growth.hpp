#pragma once

#include "triple_cold_state.hpp"
#include "triple_cold_spectrum_domain.hpp"

namespace tmv::milbrandt_detail
{

struct HailWetGrowthCell
{
    ColdRimingCell source;
    // C/R/I/S are admitted collections by the original hail population.
    std::array<double, 4> dry_collection_per_kg{};
    std::array<double, 4> wet_collection_per_kg{};
};

enum class HailWetGrowthRegime
{
    NoOriginalHail,
    ColdInactive,
    Dry,
    Wet
};

class HailWetGrowthProducer;

bool produce_exact_hail_wet_growth_admission(
    const ColdParameters&, const HailWetGrowthCell&, double duration_s,
    HailWetGrowthProducer&, std::string& error, std::size_t capacity = 1,
    FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool validate_exact_hail_wet_growth_admission(
    const ColdParameters&, const HailWetGrowthCell&, double duration_s,
    const HailWetGrowthProducer&, std::string& error, std::size_t capacity = 1,
    FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

class HailWetGrowthProducer
{
public:
    const HailWetGrowthCell& source() const noexcept { return source_; }
    const std::array<double, 25>& parameters() const noexcept { return parameters_; }
    bool hail_processes() const noexcept { return hail_processes_; }
    ColdSpectrumDomain spectrum_domain() const noexcept { return spectrum_domain_; }
    double duration_s() const noexcept { return duration_s_; }
    HailWetGrowthRegime regime() const noexcept { return regime_; }
    double normalized_ventilation_m() const noexcept { return normalized_ventilation_m_; }
    double atmospheric_capacity_per_kg() const noexcept { return atmospheric_capacity_per_kg_; }
    double frozen_capacity_per_kg() const noexcept { return frozen_capacity_per_kg_; }
    double growth_capacity_per_kg() const noexcept { return growth_capacity_per_kg_; }
    double retained_liquid_per_kg() const noexcept { return retained_liquid_per_kg_; }
    double shed_mass_per_kg() const noexcept { return shed_mass_per_kg_; }
    double shed_number_per_kg() const noexcept { return shed_number_per_kg_; }
    double shed_reflectivity_per_kg() const noexcept { return shed_reflectivity_per_kg_; }
    std::size_t event_count() const noexcept { return valid_ ? 1 : 0; }

private:
    friend bool produce_exact_hail_wet_growth_admission(
        const ColdParameters&, const HailWetGrowthCell&, double, HailWetGrowthProducer&,
        std::string&, std::size_t, FrozenCollectionFailure*, ColdSpectrumDomain);
    HailWetGrowthCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    double duration_s_ = 0.0;
    HailWetGrowthRegime regime_ = HailWetGrowthRegime::NoOriginalHail;
    double normalized_ventilation_m_ = 0.0;
    double atmospheric_capacity_per_kg_ = 0.0;
    double frozen_capacity_per_kg_ = 0.0;
    double growth_capacity_per_kg_ = 0.0;
    double retained_liquid_per_kg_ = 0.0;
    double shed_mass_per_kg_ = 0.0;
    double shed_number_per_kg_ = 0.0;
    double shed_reflectivity_per_kg_ = 0.0;
    bool valid_ = false;
};

} // namespace tmv::milbrandt_detail
