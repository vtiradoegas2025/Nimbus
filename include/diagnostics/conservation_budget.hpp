/**
 * @file conservation_budget.hpp
 * @brief Domain-integrated mass/water/energy budget diagnostics.
 *
 * Provides conservation budget snapshots and per-stage transition reporting.
 * Used by the dynamics orchestrator to track conservation drift across
 * dynamics, microphysics, turbulence, and diffusion stages.
 *
 * Two groups of integrals are tracked:
 *
 *   1. TRANSPORT-CONSERVED quantities. These are the finite-volume
 *      inventories the production Cartesian C-grid kernels are formulated to
 *      advance. In a closed domain without explicit sources or diffusion they
 *      should close to the fp32 storage/recovery noise floor when scalar
 *      recovery uses the density from the matching continuity update. A drift
 *      can therefore expose either a stencil defect or a density/scalar
 *      time-level coupling defect:
 *
 *        total_mass     = integral(rho * dV)
 *          -- legacy member name for the prognostic dry-air mass implied by
 *             the active dry rho-theta EOS; conserved by mass continuity
 *             d(rho)/dt + div(rho * u) = 0
 *
 *        theta_volume   = integral(rho * theta * dV)
 *          -- conserved by the C-grid scalar advection kernel for theta:
 *             d(rho theta)/dt + div(rho u theta) = 0.
 *             The member name is retained for CSV/API compatibility; it is
 *             a density-weighted inventory, not an unweighted volume sum.
 *
 *        q_volume_total = integral(rho * (qv + qc + qr + qi + qs + qg + qh)
 *                                  * dV)
 *          -- conserved by the same rho-weighted flux form for each water
 *             dry-air mixing ratio. This legacy member name likewise denotes
 *             water mass (kg), not an unweighted mixing-ratio volume.
 *
 *   2. DERIVED inventories and compatibility aliases:
 *
 *        dry_mass       = integral(rho_d * dV)
 *          -- compatibility alias of total_mass; it is dry-air mass, not
 *             total moist-air mass and not a mass-fraction conversion
 *        total_water    = integral(rho * water * dV)
 *        moist_total_mass = dry_mass + total_water
 *        thermal_energy = integral(rho * cp * T * dV)
 *
 *      Under the declared dry-density / dry-air-mixing-ratio convention,
 *      total_water duplicates q_volume_total and moist_total_mass is their
 *      physical sum with dry mass. Thermal energy is still only a dry
 *      sensible-enthalpy proxy, not the model's complete moist/discrete
 *      energy invariant.
 *
 * Only physical scalar cells are integrated. The production C-grid stores a
 * one-cell lateral scalar closure layer, so i/j index 0 and the last i/j index
 * are excluded; every C-grid vertical scalar level is physical. The A-grid
 * uses the same lateral closure plus bottom/top closure planes, matching the
 * exact cells advanced by its dynamics and conservative scalar transport.
 *
 * Extracted from src/core/dynamics.cpp.
 */

#pragma once

#include <array>
#include <string>

struct SimulationStateView;
struct MoistureBudget;

/** Horizontal storage rectangle omitted from a physical-cell reduction. */
struct HorizontalBudgetExclusion
{
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
};

/**
 * @brief Domain-integrated mass/water/energy budget snapshot.
 *
 * See file-level docstring for the difference between the transport-conserved
 * group (total_mass, theta_volume, q_volume_total) and the derived-physical
 * group (dry_mass, total_water, moist_total_mass, thermal_energy).
 */
struct ConservationBudget
{
    // Transport-conserved integrals (tight thresholds in report_budget_transition).
    double total_mass     = 0.0;  // legacy name: prognostic dry-air mass integral(rho*dV)
    double theta_volume   = 0.0;  // legacy name: integral(rho * theta * dV)
    double q_volume_total = 0.0;  // legacy name: integral(rho * q_total * dV)

    // Derived inventories and compatibility aliases.
    double dry_mass        = 0.0;  // compatibility alias: same dry-air mass as total_mass
    double total_water     = 0.0;  // integral(rho_d * water * dV), same as q_volume_total
    // Density-weighted masses in qv,qc,qr,qi,qs,qg,qh order.
    std::array<double, 7> water_species_mass{};
    double moist_total_mass = 0.0; // dry_mass + total_water
    double thermal_energy  = 0.0;  // legacy name: dry sensible-enthalpy proxy rho_d*cp*T
};

/**
 * @brief Per-stage drift between two ConservationBudget snapshots.
 *
 * Returned by evaluate_budget_drift; consumed by report_budget_transition
 * (which emits a log line) and by tests (which assert on the warn flags).
 *
 * The relative drifts use std::max(1, |before|) in the denominator so
 * tiny inventories do not produce spurious infinities; near-zero before
 * values fall back to absolute drift.
 */
struct BudgetDrift
{
    double d_total_mass     = 0.0;
    double d_theta_volume   = 0.0;
    double d_q_volume       = 0.0;
    double d_dry_mass       = 0.0;
    double d_total_water    = 0.0;
    std::array<double, 7> d_water_species{};
    double d_moist_total_mass = 0.0;
    double d_thermal_energy = 0.0;

    double rel_total_mass     = 0.0;
    double rel_theta_volume   = 0.0;
    double rel_q_volume       = 0.0;
    double rel_dry_mass       = 0.0;
    double rel_total_water    = 0.0;
    double rel_moist_total_mass = 0.0;
    double rel_thermal_energy = 0.0;

    bool warn_kernel   = false;  // any transport-conserved drift > THRESH_KERNEL
    bool warn_physical = false;  // any derived-inventory drift > its threshold
};

// Thresholds in evaluate_budget_drift / report_budget_transition. Exposed
// here so tests can assert against the same constants the runtime uses.
//
// BUDGET_THRESH_KERNEL is set above the fp32 storage-of-rho noise floor
// (~1e-7 in practice on a 32x32x24 grid post-fix) so the WARN KERNEL line
// only fires on real form/discretization bugs, not float roundoff.
// Active-flow scenarios drift at ~1e-5/step from the same float-storage
// floor amplified by per-step accumulation; the integration harness uses
// looser per-scenario thresholds for those.
constexpr double BUDGET_THRESH_KERNEL = 1.0e-7;
constexpr double BUDGET_THRESH_DRY    = 5.0e-4;
constexpr double BUDGET_THRESH_WATER  = 5.0e-4;
constexpr double BUDGET_THRESH_ENERGY = 2.0e-3;

/**
 * @brief Computes domain-integrated conservation budgets across both groups.
 *
 * Single pass over physical scalar cells using exact physical cell volume
 * (dx * dy times local interface separation under terrain). C-grid lateral scalar closure is
 * excluded; collocated grids additionally exclude bottom/top closure planes.
 * Both groups are computed from the same loop so each quantity sees the same
 * field snapshot and domain partition. An entirely unallocated state returns
 * zero; partial/mismatched allocation, non-finite samples, nonpositive dry
 * density/pressure/theta, or a nonphysical temperature fail closed. Finite
 * signed mixing ratios remain visible for field validation and are never
 * clamped by the diagnostic.
 *
 * @return Domain-integrated conservation budget snapshot.
 */
ConservationBudget compute_conservation_budget(const SimulationStateView& state);
ConservationBudget compute_conservation_budget();

/**
 * Compute the ordinary conservation and phase-partitioned moisture snapshots
 * in one physical-cell traversal. Both outputs retain the standalone
 * per-cell arithmetic, fixed-x-slab partial order, and merge order.
 */
void compute_conservation_and_moisture_budgets(
    const SimulationStateView& state,
    ConservationBudget& conservation,
    MoistureBudget& moisture);

/**
 * Compute the identical production budget over physical scalar cells outside
 * one horizontal storage rectangle. Used by recycled-window retirement so its
 * mass/water/theta/thermal currencies cannot drift from the ordinary ledger.
 */
ConservationBudget compute_conservation_budget_excluding_horizontal_rectangle(
    const SimulationStateView& state,
    const HorizontalBudgetExclusion& exclusion);

/**
 * @brief Computes the per-stage drift and warn flags between two snapshots.
 *
 * Pure function -- no logging, no globals beyond what the snapshots already
 * carry. Used by report_budget_transition and by unit tests.
 */
BudgetDrift evaluate_budget_drift(const ConservationBudget& before,
                                  const ConservationBudget& after);

/**
 * @brief Logs per-stage conservation changes when thresholds are exceeded.
 *
 * Calls evaluate_budget_drift and emits a single log line. The line
 * includes both groups of integrals and uses tag "WARN KERNEL" when a
 * transport-conserved drift trips, "WARN" when only a physical drift trips,
 * and the unmodified "[PHYSICS BUDGET]" tag at debug level otherwise.
 *
 * When a budget CSV stream is open (see open_budget_csv), this function
 * also appends one row per call to the CSV file, unconditional on the
 * warn threshold — the CSV is the time-series record consumed by
 * plotting tools and needs every transition, not just warnings.
 *
 * @param stage    Label for the physics stage (e.g., "dynamics", "microphysics").
 * @param before   Budget snapshot before the stage.
 * @param after    Budget snapshot after the stage.
 * @param dt_stage Duration of the stage in seconds (for tendency reporting).
 */
void report_budget_transition(const SimulationStateView& state,
                              const char* stage,
                              const ConservationBudget& before,
                              const ConservationBudget& after,
                              double dt_stage);
void report_budget_transition(const char* stage,
                              const ConservationBudget& before,
                              const ConservationBudget& after,
                              double dt_stage);

/**
 * @brief Opens a CSV file for per-stage budget time-series output.
 *
 * The file is opened in truncate mode (overwrites any existing file at
 * the same path), and the column header is written immediately. Each
 * subsequent call to report_budget_transition appends one row.
 *
 * If the file cannot be opened, the function logs a warning and the
 * CSV stream stays closed — report_budget_transition becomes a no-op
 * for the CSV path while continuing to emit log lines as usual.
 *
 * Header columns:
 *   sim_time_s, stage, d_total_mass, d_theta_volume, d_q_volume,
 *   d_dry_mass, d_total_water, d_thermal_energy,
 *   rel_total_mass, rel_theta_volume, rel_q_volume,
 *   rel_dry_mass, rel_total_water, rel_thermal_energy,
 *   warn_kernel, warn_physical, d_moist_total_mass, rel_moist_total_mass
 *
 * The moist-total columns are appended after the historical 16-column
 * prefix so existing positional consumers remain compatible.
 *
 * Safe to call multiple times — re-opens the file, re-writes the header.
 */
void open_budget_csv(const std::string& path);

/**
 * @brief Flushes and closes the budget CSV file if open.
 *
 * Safe to call when no CSV is open (no-op). Should be called at
 * simulation shutdown so the file is properly flushed to disk.
 */
void close_budget_csv();

/**
 * @brief Step-level gate for the per-stage diagnostic reductions
 *        (GPU audit Phase 3.6).
 *
 * This gate owns scientific reductions: per-stage conservation, momentum,
 * energy, moisture, and the microphysics conservation check. Derived fields
 * needed for output/live consumers use the independent gate below so an
 * explicit "science diagnostics off" benchmark cannot make presentation
 * output stale. Defaults to true for standalone callers and tests.
 */
void set_step_diagnostics_enabled(bool enabled);
bool step_diagnostics_enabled();

/**
 * @brief Step-level gate for radar/vorticity and other derived fields.
 *
 * The runtime enables this whenever scientific diagnostics are due or a
 * concrete consumer (snapshot output, live derived field, or trajectories)
 * needs fresh derived state. Defaults to true for standalone callers/tests.
 */
void set_step_derived_diagnostics_enabled(bool enabled);
bool step_derived_diagnostics_enabled();

/**
 * @brief Selects the lazy diagnostic groups required on this step.
 *
 * The legacy umbrella setter above enables/disables both groups. The runtime
 * uses this narrower setter after resolving concrete output dependencies.
 */
void set_step_derived_diagnostic_requirements(bool radar,
                                              bool vorticity,
                                              bool angular_momentum,
                                              bool pressure);
bool step_radar_diagnostics_enabled();
bool step_dynamics_diagnostics_enabled();
bool step_vorticity_diagnostics_enabled();
bool step_angular_momentum_diagnostics_enabled();
bool step_pressure_diagnostics_enabled();
