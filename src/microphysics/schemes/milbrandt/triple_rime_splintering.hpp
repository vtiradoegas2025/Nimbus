#pragma once

#include "triple_frozen_collection.hpp"
#include "triple_cold_spectrum_domain.hpp"

namespace tmv::milbrandt_detail
{

struct RimeSplinteringCell
{
    double temperature_k = 0.0;
    double pressure_pa = 0.0;
    double dry_density_kg_m3 = 0.0;
    double cloud_mass_per_kg = 0.0;
    std::array<FrozenCollectionPopulation, 4> initial{};
    std::array<double, 4> admitted_cloud_mass_per_kg{};
};

enum class RimeSplinteringSpecies
{
    Ice,
    Snow,
    Graupel
};

struct RimeSplinteringAcceptedChannel
{
    RimeSplinteringSpecies species = RimeSplinteringSpecies::Ice;
    // Ice self-splintering uses this mass only in B4; its actual ice mass increment is zero.
    double embryo_mass_per_kg = 0.0;
    double number_increment_per_kg = 0.0;
    double reflectivity_increment_per_kg = 0.0;
};

enum class RimeSplinteringFailure
{
    None,
    InvalidSource,
    InvalidContext,
    Capacity,
    InvalidProducer,
    IntervalRejected
};

class RimeSplinteringProducer;
struct RimeSplinteringResult;

// Transfer capture remains usable before fresh riming and diffusion fund the combined endpoint.
bool produce_exact_rime_splintering_admission(
    const ColdParameters& parameters, const RimeSplinteringCell& cell,
    double duration_s, RimeSplinteringProducer& producer, std::string& error,
    std::size_t capacity = 3, RimeSplinteringFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool evaluate_exact_triple_rime_splintering_cell(
    const ColdParameters& parameters, const RimeSplinteringCell& cell,
    double duration_s, RimeSplinteringResult& result, std::string& error,
    std::size_t capacity = 3, RimeSplinteringFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool apply_exact_triple_rime_splintering_cell(
    const ColdParameters& parameters, const RimeSplinteringCell& cell,
    double duration_s, const RimeSplinteringProducer& producer,
    RimeSplinteringResult& result, std::string& error,
    std::size_t capacity = 3, RimeSplinteringFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

class RimeSplinteringProducer
{
public:
    ColdSpectrumDomain spectrum_domain() const noexcept
    {
        return spectrum_domain_;
    }
    const RimeSplinteringCell& source() const noexcept
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

    double splinters_per_kg_collected() const noexcept
    {
        return splinters_per_kg_collected_;
    }

    const std::array<RimeSplinteringAcceptedChannel, 3>& channels() const noexcept
    {
        return channels_;
    }

    const std::array<double, 4>& mass_increment_per_kg() const noexcept
    {
        return mass_increment_per_kg_;
    }

    const std::array<double, 4>& number_increment_per_kg() const noexcept
    {
        return number_increment_per_kg_;
    }

    const std::array<double, 4>& reflectivity_increment_per_kg() const noexcept
    {
        return reflectivity_increment_per_kg_;
    }

    std::size_t event_count() const noexcept
    {
        return valid_ ? channels_.size() : 0;
    }

private:
    friend bool produce_exact_rime_splintering_admission(
        const ColdParameters&, const RimeSplinteringCell&, double,
        RimeSplinteringProducer&, std::string&, std::size_t, RimeSplinteringFailure*, ColdSpectrumDomain);
    friend bool evaluate_exact_triple_rime_splintering_cell(
        const ColdParameters&, const RimeSplinteringCell&, double,
        RimeSplinteringResult&, std::string&, std::size_t, RimeSplinteringFailure*, ColdSpectrumDomain);
    friend bool apply_exact_triple_rime_splintering_cell(
        const ColdParameters&, const RimeSplinteringCell&, double,
        const RimeSplinteringProducer&, RimeSplinteringResult&,
        std::string&, std::size_t, RimeSplinteringFailure*, ColdSpectrumDomain);

    RimeSplinteringCell source_{};
    std::array<double, 25> parameters_{};
    bool hail_processes_ = false;
    double duration_s_ = 0.0;
    ColdSpectrumDomain spectrum_domain_ = ColdSpectrumDomain::Base200;
    double splinters_per_kg_collected_ = 0.0;
    std::array<RimeSplinteringAcceptedChannel, 3> channels_{};
    std::array<double, 4> mass_increment_per_kg_{};
    std::array<double, 4> number_increment_per_kg_{};
    std::array<double, 4> reflectivity_increment_per_kg_{};
    bool valid_ = false;
};

struct RimeSplinteringResult
{
    std::array<FrozenCollectionPopulation, 4> unrounded{};
    std::array<FrozenCollectionEndpoint, 4> rounded{};
    std::array<double, 4> mass_increment_per_kg{};
    std::array<double, 4> number_increment_per_kg{};
    std::array<double, 4> reflectivity_increment_per_kg{};
    double temperature_increment_k = 0.0;
    double vapor_mass_increment_per_kg = 0.0;
    double cloud_mass_increment_per_kg = 0.0;
    RimeSplinteringProducer producer;
};

} // namespace tmv::milbrandt_detail
