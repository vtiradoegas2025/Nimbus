/**
 * @file boundary_layer.hpp
 * @brief Declarations for the chaos module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the chaos runtime and scheme implementations.
 * This file is part of the src/chaos subsystem.
 */

#pragma once
#include "physics/chaos_base.hpp"
#include "chaos/base/random_generator.hpp"
#include "chaos/base/correlation_filter.hpp"

namespace chaos 
{

/**
 * @brief Boundary layer focused perturbation scheme
 *
 * Injects uncertainty specifically where supercells/tornadoes are sensitive:
 * near-surface thermodynamics, shear, fluxes, and PBL mixing.
 * Based on AMS research on physically-based stochastic PBL perturbations.
 */
class BoundaryLayerScheme : public ChaosScheme 
{
public:
    std::string name() const override { return "boundary_layer"; }

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

    Field3D xi_pbl_;
    Field3D xi_pbl_prev_;
    std::vector<std::vector<double>> horizontal_slice_workspace_;
    uint64_t time_step_counter_ = 0;
};

}
