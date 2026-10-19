#pragma once

#include "triple_cold_sll.hpp"
#include "triple_frozen_collection.hpp"

namespace tmv::milbrandt_detail
{

struct ColdCollectedSllResult
{
    ColdRimingResult combined;
    FrozenCollectionProducer collection;
    IceGraupelConversionProducer ice_conversion;
    SnowGraupelConversionProducer snow_conversion;
    GraupelHailConversionProducer hail_conversion;
};

bool evaluate_exact_triple_cold_collected_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 8, ColdRimingFailure* failure = nullptr);

bool apply_exact_triple_cold_collected_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    const FrozenCollectionProducer&, const IceGraupelConversionProducer&,
    const SnowGraupelConversionProducer&, const GraupelHailConversionProducer&,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 8, ColdRimingFailure* failure = nullptr);

bool evaluate_exact_triple_cold_rain_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 12, ColdRimingFailure* failure = nullptr);

bool apply_exact_triple_cold_rain_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    const FrozenCollectionProducer&, const IceGraupelConversionProducer&,
    const SnowGraupelConversionProducer&, const GraupelHailConversionProducer&,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 12, ColdRimingFailure* failure = nullptr);

bool evaluate_exact_triple_cold_collision_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 15, ColdRimingFailure* failure = nullptr);

bool apply_exact_triple_cold_collision_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    const FrozenCollectionProducer&, const IceGraupelConversionProducer&,
    const SnowGraupelConversionProducer&, const GraupelHailConversionProducer&,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 15, ColdRimingFailure* failure = nullptr);

bool evaluate_exact_triple_cold_wet_growth_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 16, ColdRimingFailure* failure = nullptr);

bool apply_exact_triple_cold_wet_growth_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    const FrozenCollectionProducer&, const IceGraupelConversionProducer&,
    const SnowGraupelConversionProducer&, const GraupelHailConversionProducer&,
    ColdCollectedSllResult& result, std::string& error,
    std::size_t capacity = 16, ColdRimingFailure* failure = nullptr);

} // namespace tmv::milbrandt_detail
