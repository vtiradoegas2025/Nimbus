#pragma once

#include "triple_cold_graupel_conversions.hpp"
#include "triple_graupel_hail_conversion.hpp"

namespace tmv::milbrandt_detail
{

struct ColdSllResult
{
    ColdRimingResult combined;
    IceGraupelConversionProducer ice_conversion;
    SnowGraupelConversionProducer snow_conversion;
    GraupelHailConversionProducer hail_conversion;
};

bool evaluate_exact_triple_cold_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    ColdSllResult& result, std::string& error,
    std::size_t capacity = 3, ColdRimingFailure* failure = nullptr);

bool apply_exact_triple_cold_sll_cell(
    const ColdParameters&, const ColdRimingCell&, double duration_s,
    const IceGraupelConversionProducer&, const SnowGraupelConversionProducer&,
    const GraupelHailConversionProducer&, ColdSllResult& result, std::string& error,
    std::size_t capacity = 3, ColdRimingFailure* failure = nullptr);

} // namespace tmv::milbrandt_detail
