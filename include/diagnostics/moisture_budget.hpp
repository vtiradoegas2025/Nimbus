/**
 * @file moisture_budget.hpp
 * @brief Domain-integrated moisture budget diagnostics (F.5b).
 *
 * Companion to conservation_budget.hpp, momentum_budget.hpp, energy_budget.hpp.
 * Partitions the airborne water mass by phase so the F.6.C gate can assert
 * |dq/q| < 1e-4 over a 1 hr simulation (microphysics on):
 *
 *   vapor       = integral(rho * qv * dV)
 *   liquid      = integral(rho * (qc + qr) * dV)         cloud + rain
 *   ice         = integral(rho * (qi + qs + qg + qh) * dV) ice + snow + graupel + hail
 *   total_water = vapor + liquid + ice
 *
 * The active production convention is dry-air density rho with q fields as
 * dry-air mixing ratios, so rho*q is water mass per volume. That convention
 * is being made explicit model-wide under SCI-024.
 *
 * The phase split is the diagnostic of microphysics source/sink activity:
 * condensation/evaporation move mass vapor <-> liquid, freezing/melting move
 * liquid <-> ice, deposition/sublimation move vapor <-> ice. Every microphysics
 * process conserves total_water (it only redistributes among phases), so the
 * per-phase drifts (d_vapor, d_liquid, d_ice) ARE the net source/sink terms,
 * while total_water must stay invariant across a microphysics stage. Advection
 * and sedimentation can change total_water only through domain-boundary flux
 * (precipitation fallout), captured as a boundary loss in the inventory drift.
 */

#pragma once

#include <string>

struct SimulationStateView;

/**
 * @brief Domain-integrated water-mass inventory snapshot (kg).
 */
struct MoistureBudget
{
    double vapor       = 0.0;  // integral(rho * qv * dV)
    double liquid      = 0.0;  // integral(rho * (qc + qr) * dV)
    double ice         = 0.0;  // integral(rho * (qi + qs + qg + qh) * dV)
    double total_water = 0.0;  // vapor + liquid + ice
};

/**
 * @brief Per-stage moisture drift between two snapshots + warn flag.
 *
 * d_vapor / d_liquid / d_ice are the net per-phase source/sink across the
 * stage; rel_total_water is the conservation residual the gate keys on.
 * Relative drifts use std::max(1, |before|) (absolute fallback near zero).
 */
struct MoistureDrift
{
    double d_vapor       = 0.0;
    double d_liquid      = 0.0;
    double d_ice         = 0.0;
    double d_total_water = 0.0;

    double rel_vapor       = 0.0;
    double rel_liquid      = 0.0;
    double rel_ice         = 0.0;
    double rel_total_water = 0.0;

    bool warn = false;  // rel_total_water > BUDGET_THRESH_MOISTURE
};

// Per-step total-water closure threshold == the F.6.C 1-hr gate (|dq/q| <
// 1e-4). Keyed on total_water only: per-phase mass legitimately moves every
// step; total water must close (modulo precipitation fallout at the boundary).
constexpr double BUDGET_THRESH_MOISTURE = 1.0e-4;

/**
 * @brief Computes the domain-integrated moisture inventory by phase.
 *
 * Single pass over physical scalar cells with the uniform Cartesian cell
 * volume. The production C-grid's one-cell lateral scalar closure is excluded;
 * the collocated A-grid additionally excludes bottom/top closure planes.
 * Returns an all-zero snapshot only when every required field is unallocated.
 * Partial/mismatched allocation, non-finite samples, and nonpositive dry
 * density fail closed. Finite signed mixing ratios remain visible for the
 * field validator rather than being hidden by a diagnostic clamp.
 */
MoistureBudget compute_moisture_budget(const SimulationStateView& state);
MoistureBudget compute_moisture_budget();

/**
 * @brief Computes per-stage drift and the warn flag. Pure function.
 */
MoistureDrift evaluate_moisture_drift(const MoistureBudget& before,
                                      const MoistureBudget& after);

/**
 * @brief Logs the moisture transition (WARN when total-water drift exceeds
 *        threshold, debug otherwise) and appends a CSV row when open.
 */
void report_moisture_transition(const SimulationStateView& state,
                                const char* stage,
                                const MoistureBudget& before,
                                const MoistureBudget& after,
                                double dt_stage);
void report_moisture_transition(const char* stage,
                                const MoistureBudget& before,
                                const MoistureBudget& after,
                                double dt_stage);

/**
 * @brief Opens (truncating) the moisture-budget CSV and writes the header.
 */
void open_moisture_csv(const std::string& path);

/**
 * @brief Flushes and closes the moisture-budget CSV if open (safe when closed).
 */
void close_moisture_csv();
