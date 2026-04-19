/**
 * @file conservation_budget.hpp
 * @brief Domain-integrated mass/water/energy budget diagnostics.
 *
 * Provides conservation budget snapshots and per-stage transition reporting.
 * Used by the dynamics orchestrator to track conservation drift across
 * dynamics, microphysics, turbulence, and diffusion stages.
 *
 * Extracted from src/core/dynamics.cpp.
 */

#pragma once

/**
 * @brief Domain-integrated mass/water/energy budget snapshot.
 */
struct ConservationBudget
{
    double dry_mass = 0.0;
    double total_water = 0.0;
    double thermal_energy = 0.0;
};

/**
 * @brief Computes dry-mass, water, and thermal-energy integrals over the domain.
 *
 * Uses radial cell-volume weights (cylindrical geometry) to integrate
 * rho, moisture species, and thermal energy across all grid cells.
 *
 * @return Domain-integrated conservation budget snapshot.
 */
ConservationBudget compute_conservation_budget();

/**
 * @brief Logs per-stage conservation changes when thresholds are exceeded.
 *
 * Computes relative changes in dry mass, total water, and thermal energy
 * between two budget snapshots. Emits a warning when any relative change
 * exceeds the configured threshold; otherwise logs at debug level.
 *
 * @param stage    Label for the physics stage (e.g., "dynamics", "microphysics").
 * @param before   Budget snapshot before the stage.
 * @param after    Budget snapshot after the stage.
 * @param dt_stage Duration of the stage in seconds (for tendency reporting).
 */
void report_budget_transition(const char* stage,
                              const ConservationBudget& before,
                              const ConservationBudget& after,
                              double dt_stage);
