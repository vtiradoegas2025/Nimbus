#pragma once

#include "triple_cloud_adjustment.hpp"
#include "triple_warm_primary.hpp"
#include "triple_warm_subcycle.hpp"

namespace tmv::milbrandt_detail
{

struct ExactTripleWarmCloudCellResult
{
    TripleCloudAdjustmentResult cloud;
    ExactTripleWarmDoubleSourceCellResult warm;
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    RainTripleMomentState rounded_rain;
    double temperature_increment_k = 0.0;

private:
    friend bool produce_exact_triple_warm_cloud_cell(
        const WarmRainParameters&, const DoubleWarmRainCell&, double, double,
        const RainTripleMomentState&, ExactTripleWarmCloudCellResult&,
        std::string&, std::size_t);
    friend bool apply_exact_triple_warm_cloud_cell(
        const WarmRainParameters&, const DoubleWarmRainCell&, double, double,
        const RainTripleMomentState&, const ExactTripleWarmCloudCellResult&,
        ExactTripleWarmCloudCellResult&, std::string&);

    RainTripleMomentState original_rain_{};
};

bool produce_exact_triple_warm_cloud_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double vertical_velocity,
    double dt,
    const RainTripleMomentState& initial_rain,
    ExactTripleWarmCloudCellResult& result,
    std::string& error,
    std::size_t observer_capacity = 16);

bool apply_exact_triple_warm_cloud_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double vertical_velocity,
    double dt,
    const RainTripleMomentState& initial_rain,
    const ExactTripleWarmCloudCellResult& producer,
    ExactTripleWarmCloudCellResult& result,
    std::string& error);

struct ExactTripleDynamicWarmCloudCellResult
{
    TripleCloudAdjustmentResult cloud;
    ExactTripleDynamicWarmCellResult warm;
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    RainTripleMomentState rounded_rain;
    double temperature_increment_k = 0.0;

private:
    friend bool produce_exact_triple_dynamic_warm_cloud_cell(
        const WarmRainParameters&, const DoubleWarmRainCell&, double, double,
        const RainTripleMomentState&, ExactTripleDynamicWarmCloudCellResult&, std::string&, std::size_t);
    friend bool apply_exact_triple_dynamic_warm_cloud_cell(
        const WarmRainParameters&, const DoubleWarmRainCell&, double, double,
        const RainTripleMomentState&, const ExactTripleDynamicWarmCloudCellResult&,
        ExactTripleDynamicWarmCloudCellResult&, std::string&);
    RainTripleMomentState original_rain_{};
};

bool produce_exact_triple_dynamic_warm_cloud_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double vertical_velocity,
    double dt,
    const RainTripleMomentState& initial_rain,
    ExactTripleDynamicWarmCloudCellResult& result,
    std::string& error,
    std::size_t observer_capacity = 16);

bool apply_exact_triple_dynamic_warm_cloud_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double vertical_velocity,
    double dt,
    const RainTripleMomentState& initial_rain,
    const ExactTripleDynamicWarmCloudCellResult& producer,
    ExactTripleDynamicWarmCloudCellResult& result,
    std::string& error);

struct ExactTripleSubcycledWarmCloudCellResult
{
    TripleCloudAdjustmentResult cloud;
    WarmSubcycleCellResult warm;
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    RainTripleMomentState rounded_rain;
    double temperature_increment_k = 0.0;

private:
    friend bool produce_exact_triple_subcycled_warm_cloud_cell(
        const WarmRainParameters&, const DoubleWarmRainCell&, double, double,
        const RainTripleMomentState&, ExactTripleSubcycledWarmCloudCellResult&, std::string&, const WarmSubcycleOptions&);
    friend bool apply_exact_triple_subcycled_warm_cloud_cell(
        const WarmRainParameters&, const DoubleWarmRainCell&, double, double,
        const RainTripleMomentState&, const ExactTripleSubcycledWarmCloudCellResult&,
        ExactTripleSubcycledWarmCloudCellResult&, std::string&, const WarmSubcycleOptions&);
    RainTripleMomentState original_rain_{};
};

bool produce_exact_triple_subcycled_warm_cloud_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double vertical_velocity,
    double dt,
    const RainTripleMomentState& initial_rain,
    ExactTripleSubcycledWarmCloudCellResult& result,
    std::string& error,
    const WarmSubcycleOptions& options = {});

bool apply_exact_triple_subcycled_warm_cloud_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double vertical_velocity,
    double dt,
    const RainTripleMomentState& initial_rain,
    const ExactTripleSubcycledWarmCloudCellResult& producer,
    ExactTripleSubcycledWarmCloudCellResult& result,
    std::string& error,
    const WarmSubcycleOptions& options = {});

} // namespace tmv::milbrandt_detail
