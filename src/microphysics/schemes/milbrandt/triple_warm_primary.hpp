#pragma once

#include "triple_warm_adapter.hpp"
#include "triple_warm_dynamic.hpp"

namespace tmv::milbrandt_detail
{

struct ExactTripleWarmCellResult
{
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    RainTripleMomentState rounded_rain;
    WarmRainTendencies rates;
    TripleWarmRainExactResult producer;
};

struct ExactTripleWarmDoubleSourceCellResult
{
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    RainTripleMomentState rounded_rain;
    WarmRainTendencies rates;
    TripleWarmRainDoubleSourceResult producer;
};

struct ExactTripleDynamicWarmCellResult
{
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    RainTripleMomentState rounded_rain;
    WarmRainTendencies rates;
    TripleWarmRainDynamicResult producer;
};

struct ExactTripleDynamicWarmFP64CellResult
{
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    WarmRainTendencies rates;
    TripleWarmRainDynamicFP64Result producer;
};

bool apply_exact_triple_warm_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    const TripleWarmRainExactResult& producer,
    ExactTripleWarmCellResult& result,
    std::string& error);

bool evaluate_exact_triple_warm_cell(
    const WarmRainParameters& parameters,
    const WarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    ExactTripleWarmCellResult& result,
    std::string& error,
    std::size_t observer_capacity = 16);

bool apply_exact_triple_warm_double_source_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    const TripleWarmRainDoubleSourceResult& producer,
    ExactTripleWarmDoubleSourceCellResult& result,
    std::string& error);

bool evaluate_exact_triple_warm_double_source_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    ExactTripleWarmDoubleSourceCellResult& result,
    std::string& error,
    std::size_t observer_capacity = 16);

bool apply_exact_triple_dynamic_warm_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    const TripleWarmRainDynamicResult& producer,
    ExactTripleDynamicWarmCellResult& result,
    std::string& error);

bool evaluate_exact_triple_dynamic_warm_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    ExactTripleDynamicWarmCellResult& result,
    std::string& error,
    std::size_t observer_capacity = 16);

bool apply_exact_triple_dynamic_warm_fp64_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const ExactRainTripleMomentState& initial_rain,
    const TripleWarmRainDynamicFP64Result& producer,
    ExactTripleDynamicWarmFP64CellResult& result,
    std::string& error,
    DynamicWarmStepFailure* failure = nullptr,
    const DynamicWarmFP64Policies& policies = {});

bool evaluate_exact_triple_dynamic_warm_fp64_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const ExactRainTripleMomentState& initial_rain,
    ExactTripleDynamicWarmFP64CellResult& result,
    std::string& error,
    DynamicWarmStepFailure* failure = nullptr,
    std::size_t observer_capacity = 16,
    const DynamicWarmFP64Policies& policies = {});

} // namespace tmv::milbrandt_detail
