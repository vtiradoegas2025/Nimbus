/**
 * @file none.cpp
 * @brief Implementation for the chaos module.
 *
 * Provides executable logic for the chaos runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/chaos subsystem.
 */

#include "none.hpp"
#include "util/log.hpp"

namespace tmv::chaos
{

/**
 * @brief Initializes a no-op chaos scheme.
 */
void NoneScheme::initialize(const ChaosConfig& cfg, const GridMetrics& grid) {
    (void)cfg;
    (void)grid;
    tmv::log_info("Initialized chaos scheme: none (no perturbations)");
}

/**
 * @brief Leaves initial conditions unchanged and reports no perturbations.
 */
void NoneScheme::apply_initial_conditions(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    SimulationState& state,
    ChaosDiagnostics* diag
) {
    if (diag) {
        diag->mean_perturbation = 0.0;
        diag->variance_perturbation = 0.0;
        diag->warnings.push_back("None scheme: no IC perturbations applied");
    }
}

/**
 * @brief Leaves tendencies unchanged and reports no perturbations.
 */
void NoneScheme::apply_tendencies(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    const ChaosStateView& state_view,
    ChaosTendencies& tendencies,
    ChaosDiagnostics* diag
) {
    if (diag) {
        diag->mean_perturbation = 0.0;
        diag->variance_perturbation = 0.0;
        diag->warnings.push_back("None scheme: no tendency perturbations applied");
    }
}

/**
 * @brief Advances noise state (no-op for none scheme).
 */
void NoneScheme::step_noise(
    const ChaosConfig& cfg,
    const GridMetrics& grid,
    double dt
) {
}

}
