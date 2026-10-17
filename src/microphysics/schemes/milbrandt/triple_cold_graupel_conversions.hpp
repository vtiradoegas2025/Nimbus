#pragma once

#include "triple_cold_ice_graupel.hpp"
#include "triple_snow_graupel_conversion.hpp"

namespace tmv::milbrandt_detail
{

struct ColdGraupelConversionsResult
{
    ColdRimingResult combined;
    IceGraupelConversionProducer ice_conversion;
    SnowGraupelConversionProducer snow_conversion;
};

bool evaluate_exact_triple_cold_graupel_conversions_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdGraupelConversionsResult& result, std::string& error,
    std::size_t capacity = 2, ColdRimingFailure* failure = nullptr,
    bool homogeneous_freezing = false);

bool apply_exact_triple_cold_graupel_conversions_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const IceGraupelConversionProducer& ice_conversion,
    const SnowGraupelConversionProducer& snow_conversion,
    ColdGraupelConversionsResult& result, std::string& error,
    std::size_t capacity = 2, ColdRimingFailure* failure = nullptr,
    bool homogeneous_freezing = false);

} // namespace tmv::milbrandt_detail
