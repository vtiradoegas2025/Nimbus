/**
 * @file full_stochastic.hpp
 * @brief Declarations for the chaos module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the chaos runtime and scheme implementations.
 * This file is part of the src/chaos subsystem.
 */

#pragma once
#include "chaos/chaos_base.hpp"
#include "chaos/base/random_generator.hpp"
#include "chaos/base/correlation_filter.hpp"

namespace chaos {

/**
 * @brief Full stochastic perturbation scheme
 *
 * Represents time-evolving model uncertainty (subgrid physics errors) in
 * convection-permitting ensembles. Uses SPPT-like multiplicative noise
 * applied to physics tendencies. Based on AMS research on stochastic
 * physics in CAM ensembles.
 */
class FullStochasticScheme : public ChaosScheme {
public:
    std::string name() const override { return "full_stochastic"; }

    void initialize(const ChaosConfig& cfg, const GridMetrics& grid) override;

    void apply_initial_conditions(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        SimulationState& state,
        ChaosDiagnostics* diag = nullptr
    ) override;

    void apply_tendencies(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        const ChaosStateView& state_view,
        ChaosTendencies& tendencies,
        ChaosDiagnostics* diag = nullptr
    ) override;

    void step_noise(
        const ChaosConfig& cfg,
        const GridMetrics& grid,
        double dt
    ) override;

private:
    ChaosRNG rng_;
    std::unique_ptr<CorrelationFilter> correlation_filter_;

    Field3D xi_microphysics_;
    Field3D xi_pbl_;
    Field3D xi_diffusion_;

    Field3D xi_micro_prev_;
    Field3D xi_pbl_prev_;
    Field3D xi_diff_prev_;

    std::vector<std::vector<double>> horizontal_slice_workspace_;

    uint64_t time_step_counter_ = 0;
};

}
