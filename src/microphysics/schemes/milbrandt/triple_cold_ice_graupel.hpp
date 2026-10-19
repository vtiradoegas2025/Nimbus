#pragma once

#include "triple_ice_graupel_conversion.hpp"

namespace tmv::milbrandt_detail
{

struct ColdIceGraupelResult
{
    ColdRimingResult combined;
    IceGraupelConversionProducer conversion;
};

bool evaluate_exact_triple_cold_ice_graupel_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    ColdIceGraupelResult& result, std::string& error,
    std::size_t capacity = 1, ColdRimingFailure* failure = nullptr);

bool apply_exact_triple_cold_ice_graupel_cell(
    const ColdParameters& parameters, const ColdRimingCell& cell, double duration_s,
    const IceGraupelConversionProducer& producer, ColdIceGraupelResult& result,
    std::string& error, std::size_t capacity = 1, ColdRimingFailure* failure = nullptr);

} // namespace tmv::milbrandt_detail
