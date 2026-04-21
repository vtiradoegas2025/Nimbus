/**
 * @file none.hpp
 * @brief Declarations for the chaos module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the chaos runtime and scheme implementations.
 * This file is part of the src/chaos subsystem.
 */

#pragma once
#include "chaos/chaos_base.hpp"

namespace chaos {

/**
 * @brief "None" chaos scheme - no perturbations applied
 *
 * This scheme serves as a baseline and does nothing. Useful for deterministic
 * runs or when chaos perturbations are disabled.
 */
class NoneScheme : public ChaosScheme {
public:
    std::string name() const override { return "none"; }

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
};

}
