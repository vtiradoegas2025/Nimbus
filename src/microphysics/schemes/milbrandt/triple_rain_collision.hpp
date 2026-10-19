#pragma once

#include "triple_rain_collection.hpp"

namespace tmv::milbrandt_detail
{

struct RainCollisionChannel
{
    RainCollectionChannel rain;
    RainCollectionChannel frozen;
    double admission_factor = 1.0;
    TripleMomentSpecies destination = TripleMomentSpecies::Graupel;
    double rain_mean_mass_diameter_m = 0.0;
    double frozen_mean_mass_diameter_m = 0.0;
    double mixed_density_kg_m3 = 0.0;
    double destination_density_kg_m3 = 0.0;
    double product_mass_per_kg = 0.0;
    double product_number_per_kg = 0.0;
    double product_reflectivity_per_kg = 0.0;
};

class RainCollisionProducer;

bool produce_exact_rain_collision_admission(
    const ColdParameters&, const RainCollectionCell&, double duration_s,
    RainCollisionProducer&, std::string& error, std::size_t capacity = 7,
    FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool validate_exact_rain_collision_admission(
    const ColdParameters&, const RainCollectionCell&, double duration_s,
    const RainCollisionProducer&, std::string& error, std::size_t capacity = 7,
    FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

class RainCollisionProducer
{
public:
    const RainCollectionProducer& rain_source() const noexcept { return rain_source_; }
    const std::array<RainCollisionChannel, 3>& channels() const noexcept { return channels_; }
    const RainCollectionChannel& hail_collection() const noexcept { return rain_source_.channels()[3]; }
    std::size_t event_count() const noexcept { return valid_ ? 7 : 0; }

private:
    friend bool produce_exact_rain_collision_admission(
        const ColdParameters&, const RainCollectionCell&, double, RainCollisionProducer&,
        std::string&, std::size_t, FrozenCollectionFailure*, ColdSpectrumDomain);
    RainCollectionProducer rain_source_;
    std::array<RainCollisionChannel, 3> channels_{};
    bool valid_ = false;
};

struct RainCollisionIncrements
{
    FrozenCollectionPopulation rain;
    FrozenCollectionIncrements frozen;
    double temperature_k = 0.0;
};

struct RainCollisionResult
{
    RainCollectionCell unrounded;
    FrozenCollectionEndpoint rounded_rain;
    std::array<FrozenCollectionEndpoint, 4> rounded_frozen{};
    RainCollisionIncrements increments;
    RainCollisionProducer producer;
};

bool evaluate_exact_rain_collision_increments(
    const ColdParameters&, const RainCollectionCell&, double duration_s,
    const RainCollisionProducer&, RainCollisionIncrements&, std::string& error,
    std::size_t capacity = 7, FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool evaluate_exact_triple_rain_collision_cell(
    const ColdParameters&, const RainCollectionCell&, double duration_s,
    RainCollisionResult&, std::string& error, std::size_t capacity = 7,
    FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

bool apply_exact_triple_rain_collision_cell(
    const ColdParameters&, const RainCollectionCell&, double duration_s,
    const RainCollisionProducer&, RainCollisionResult&, std::string& error,
    std::size_t capacity = 7, FrozenCollectionFailure* failure = nullptr,
    ColdSpectrumDomain spectrum_domain = ColdSpectrumDomain::Base200);

} // namespace tmv::milbrandt_detail
