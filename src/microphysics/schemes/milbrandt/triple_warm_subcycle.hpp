#pragma once

#include "triple_warm_primary.hpp"

#include <array>
#include <vector>

namespace tmv::milbrandt_detail
{

struct WarmSubcycleOptions
{
    double relative_tolerance = 1.0e-6;
    double absolute_mass_tolerance = 1.0e-14;
    double absolute_number_tolerance = 1.0e-6;
    double absolute_reflectivity_tolerance = 1.0e-30;
    double absolute_temperature_tolerance = 1.0e-8;
    std::size_t max_accepted_steps = 4096;
    std::size_t max_trials = 8192;
    std::size_t max_halvings = 40;
    std::size_t observer_capacity = 16;
};

struct WarmSubcycleEulerProbe
{
    double duration_s = 0.0;
    DoubleWarmRainCell source_cell{};
    ExactRainTripleMomentState initial_rain{};
    WarmRainDoubleEndpoints endpoints{};
    ExactRainTripleMomentState final_rain{};
};

struct WarmSubcycleAcceptedStep
{
    double duration_s = 0.0;
    DoubleWarmRainCell source_cell{};
    ExactRainTripleMomentState initial_rain{};
    WarmRainDoubleEndpoints endpoints{};
    ExactRainTripleMomentState final_rain{};
    double error_norm = 0.0;
    std::array<WarmSubcycleEulerProbe, 3> euler_probes{};
};

class WarmSubcycleProducer
{
public:
    const DoubleWarmRainCell& source_cell() const noexcept
    {
        return source_cell_;
    }
    const RainTripleMomentState& original_rain() const noexcept
    {
        return original_rain_;
    }
    const WarmSubcycleOptions& options() const noexcept
    {
        return options_;
    }
    const std::vector<WarmSubcycleAcceptedStep>& steps() const noexcept
    {
        return steps_;
    }
    std::size_t accepted_steps() const noexcept
    {
        return steps_.size();
    }
    std::size_t rejected_trials() const noexcept
    {
        return rejected_trials_;
    }
    std::size_t trials() const noexcept
    {
        return trials_;
    }
    double completed_duration_s() const noexcept
    {
        return completed_duration_s_;
    }
    unsigned method_version() const noexcept
    {
        return 2;
    }
    const ExactRainTripleMomentState& pre_regularized_rain() const noexcept
    {
        return pre_regularized_rain_;
    }
    const ExactRainTripleMomentState& post_regularization_before() const noexcept
    {
        return post_regularization_before_;
    }
    const ExactRainTripleMomentState& final_rain() const noexcept
    {
        return final_rain_;
    }

private:
    friend struct WarmSubcycleBuilder;
    DoubleWarmRainCell source_cell_{};
    RainTripleMomentState original_rain_{};
    WarmSubcycleOptions options_{};
    WarmRainParameters parameters_{};
    std::vector<WarmSubcycleAcceptedStep> steps_;
    std::size_t rejected_trials_ = 0;
    std::size_t trials_ = 0;
    double completed_duration_s_ = 0.0;
    double requested_duration_s_ = 0.0;
    ExactRainTripleMomentState pre_regularized_rain_{};
    ExactRainTripleMomentState post_regularization_before_{};
    ExactRainTripleMomentState final_rain_{};
};

struct WarmSubcycleCellResult
{
    WarmRainDoubleEndpoints endpoints;
    ExactRainTripleMomentState rain;
    RainTripleMomentState rounded_rain;
    WarmRainTendencies rates;
    WarmSubcycleProducer producer;
};

bool evaluate_exact_triple_warm_subcycled_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    WarmSubcycleCellResult& result,
    std::string& error,
    const WarmSubcycleOptions& options = {});

bool apply_exact_triple_warm_subcycled_cell(
    const WarmRainParameters& parameters,
    const DoubleWarmRainCell& cell,
    double dt,
    const RainTripleMomentState& initial_rain,
    const WarmSubcycleProducer& producer,
    WarmSubcycleCellResult& result,
    std::string& error,
    const WarmSubcycleOptions& options = {});

} // namespace tmv::milbrandt_detail
