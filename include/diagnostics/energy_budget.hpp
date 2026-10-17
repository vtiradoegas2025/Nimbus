/**
 * @file energy_budget.hpp
 * @brief Domain-integrated energy budget diagnostics (F.5b).
 *
 * Companion to conservation_budget.hpp (mass/water) and momentum_budget.hpp.
 * Tracks a domain-integrated energy inventory for drift monitoring:
 *
 *   kinetic   = integral(0.5 * rho_d * (u^2 + v^2 + w^2) * dV)
 *   internal  = integral(rho_d * c_v * T * dV)        c_v = c_p - R_d
 *   potential = integral(rho_d * g * z * dV)          z = physical scalar height
 *   latent    = integral(rho_d * L_v * qv * dV)       vapor latent heat
 *   total     = kinetic + internal + potential + latent
 *
 * with T = theta * (p / p0)^(R_d/c_p). This is not yet a proven discrete total-
 * energy invariant. It uses the prognostic dry density, dry-air c_v, vapor
 * latent energy only, reconstructed face winds, and no explicit boundary/
 * source ledger. Water sensible/kinetic/potential energy, frozen-phase
 * reference energy, and exact pairings to every operator are omitted. The
 * inventory is useful for bounded calm-case drift detection, but a passing
 * value must not be labeled complete moist-energy conservation (SCI-023/024).
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>

struct SimulationStateView;
class Field3D;

struct HorizontalEnergyBudgetExclusion
{
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
};

/**
 * @brief Domain-integrated energy inventory snapshot (Joules).
 */
struct EnergyBudget
{
    double kinetic   = 0.0;  // integral(0.5 * rho_d * |U|^2 * dV)
    double internal  = 0.0;  // integral(rho_d * c_v * T * dV)
    double potential = 0.0;  // integral(rho_d * g * z * dV)
    double latent    = 0.0;  // integral(rho_d * L_v * qv * dV)
    double total     = 0.0;  // sum of the four reservoirs
};

/**
 * Versioned physical moist-mixture energy inventory.
 *
 * This is a diagnostic basis, not yet a claimed invariant of every Nimbus
 * operator. All water variables are dry-air mixing ratios and rho is dry-air
 * density. With Tr=273.15 K:
 *
 *   e_d = cv_d (T-Tr)
 *   e_v = cv_v (T-Tr) + Lv - Rv Tr
 *   e_l = c_l  (T-Tr)
 *   e_i = c_i  (T-Tr) - Lf
 *
 * Vapor's reference converts the declared latent enthalpy at Tr to internal
 * energy; condensed-phase p/rho work is neglected. Cloud/rain are liquid and
 * ice/snow/graupel/hail are frozen. Resolved kinetic and gravitational energy
 * use total mixture mass rho_d(1+qt). C-grid kinetic energy uses the native
 * adjacent-half-cell component control volumes, not centered winds. TKE is a
 * Nimbus dry-carrier prognostic and therefore uses rho_d.
 */
struct MoistMixtureEnergyBudgetV1
{
    static constexpr std::uint32_t basis_version = 1;

    double dry_internal = 0.0;
    double water_sensible = 0.0;
    double phase_reference = 0.0;
    double resolved_kinetic = 0.0;
    double gravitational = 0.0;
    double subgrid_tke = 0.0;
    double total_water_mass = 0.0;
    double total = 0.0;
};

/**
 * Physical moist-mixture basis v2.
 *
 * V2 preserves every v1 reservoir/reference choice and corrects one C-grid
 * terrain omission: the implicit impermeable ground face has zero
 * contravariant flux, but its physical vertical velocity is generally
 * u*dh/dx + v*dh/dy. V2 assigns that diagnosed face the same owned scalar-cell
 * half-mass used by every other physical boundary component face.
 */
struct MoistMixtureEnergyBudgetV2
{
    static constexpr std::uint32_t basis_version = 2;

    double dry_internal = 0.0;
    double water_sensible = 0.0;
    double phase_reference = 0.0;
    double resolved_kinetic = 0.0;
    double gravitational = 0.0;
    double subgrid_tke = 0.0;
    double total_water_mass = 0.0;
    double total = 0.0;
};

/** One fused traversal result: compatibility proxy plus v1 and corrected v2. */
struct EnergyDiagnosticSnapshotV1
{
    EnergyBudget legacy_proxy;
    MoistMixtureEnergyBudgetV1 moist_mixture;
    MoistMixtureEnergyBudgetV2 moist_mixture_v2;
};

/**
 * @brief Per-stage energy drift between two snapshots + warn flag.
 *
 * Relative drifts use std::max(1, |before|) in the denominator (absolute
 * fallback near zero).
 */
struct EnergyDrift
{
    double d_kinetic   = 0.0;
    double d_internal  = 0.0;
    double d_potential = 0.0;
    double d_latent    = 0.0;
    double d_total     = 0.0;

    double rel_kinetic   = 0.0;
    double rel_internal  = 0.0;
    double rel_potential = 0.0;
    double rel_latent    = 0.0;
    double rel_total     = 0.0;

    bool warn = false;  // rel_total > BUDGET_THRESH_ENERGY_TOTAL
};

// Legacy per-step inventory-drift warning threshold. This is not by itself a
// one-hour initial/final or complete moist/discrete energy-conservation gate;
// see the file-level caveat.
constexpr double BUDGET_THRESH_ENERGY_TOTAL = 1.0e-3;

/**
 * @brief Computes the domain-integrated energy inventory.
 *
 * Single pass over physical scalar cells with exact physical cell volume and
 * height, including terrain-following columns. The production C-grid's one-cell lateral scalar closure is excluded;
 * the collocated A-grid additionally excludes bottom/top closure planes.
 * C-grid face velocities are reconstructed to scalar centers before kinetic
 * energy is formed. Returns an all-zero snapshot only when every required
 * field is unallocated. Partial/mismatched fields and any non-finite or
 * nonphysical physical-domain input fail closed; no invalid cell is skipped
 * from one reservoir while remaining in another.
 */
EnergyBudget compute_energy_budget(const SimulationStateView& state);
EnergyBudget compute_energy_budget();

/**
 * Compute the complete v1 moist-mixture diagnostic basis over physical cells.
 *
 * Every prognostic thermodynamic, water, velocity, and TKE field is required
 * with the exact grid shape. Non-finite/nonphysical density, pressure,
 * temperature, water, TKE, geometry, or velocity fails the complete snapshot.
 */
MoistMixtureEnergyBudgetV1 compute_moist_mixture_energy_v1(
    const SimulationStateView& state);
MoistMixtureEnergyBudgetV1 compute_moist_mixture_energy_v1();

/**
 * Compute basis v2. Flat and collocated states reduce exactly to v1. Active
 * C-grid terrain adds the diagnosed physical ground-face kinetic half-volume.
 */
MoistMixtureEnergyBudgetV2 compute_moist_mixture_energy_v2(
    const SimulationStateView& state);
MoistMixtureEnergyBudgetV2 compute_moist_mixture_energy_v2();

/** Compute basis v2 with fixed-slab, thread-count-independent reductions. */
MoistMixtureEnergyBudgetV2 compute_moist_mixture_energy_v2_deterministic(
    const SimulationStateView& state);

/**
 * Compute the identical v2 physical moist-mixture basis outside one
 * horizontal storage rectangle. Face kinetic half-volumes remain owned by
 * their contributing scalar cells, making retained/retiring subsets additive.
 */
MoistMixtureEnergyBudgetV2
compute_moist_mixture_energy_v2_excluding_horizontal_rectangle(
    const SimulationStateView& state,
    const HorizontalEnergyBudgetExclusion& exclusion);

MoistMixtureEnergyBudgetV2
compute_moist_mixture_energy_v2_excluding_horizontal_rectangle_deterministic(
    const SimulationStateView& state, const HorizontalEnergyBudgetExclusion& exclusion);

// Absolute cell contributions, including owned face halves, bound cancellation roundoff.
std::array<double, 7> moist_mixture_energy_v2_magnitudes(
    const SimulationStateView& state, const HorizontalEnergyBudgetExclusion* exclusion = nullptr);

/**
 * Compute the legacy proxy and v1 in one physical-domain traversal, then
 * derive v2 with at most one terrain-surface reduction.
 */
EnergyDiagnosticSnapshotV1 compute_energy_diagnostic_snapshot_v1(
    const SimulationStateView& state);
EnergyDiagnosticSnapshotV1 compute_energy_diagnostic_snapshot_v1();

EnergyDiagnosticSnapshotV1 compute_energy_diagnostic_snapshot_v1_deterministic(
    const SimulationStateView& state);

// Returns [before, current] resolved kinetic energies for shared thermodynamics
// and geometry. Inputs are unchanged; complete-inventory rejection is preserved.
std::array<double, 2> compute_flat_cgrid_velocity_energy_pair_deterministic(
    const SimulationStateView& state, Field3D& before_u, Field3D& before_v, Field3D& before_w);

// Complete a velocity-only update. The caller must preserve all shared
// thermodynamic fields and geometry from the validated complete snapshot.
double compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(
    const SimulationStateView& state, const EnergyDiagnosticSnapshotV1& before);

/**
 * @brief Computes per-stage drift and the warn flag. Pure function.
 */
EnergyDrift evaluate_energy_drift(const EnergyBudget& before,
                                  const EnergyBudget& after);

/**
 * @brief Logs the energy transition (WARN when total drift exceeds threshold,
 *        debug otherwise) and appends a CSV row when a stream is open.
 */
void report_energy_transition(const SimulationStateView& state,
                              const char* stage,
                              const EnergyBudget& before,
                              const EnergyBudget& after,
                              double dt_stage);
void report_energy_transition(const char* stage,
                              const EnergyBudget& before,
                              const EnergyBudget& after,
                              double dt_stage);
void report_energy_transition(
    const SimulationStateView& state,
    const char* stage,
    const EnergyDiagnosticSnapshotV1& before,
    const EnergyDiagnosticSnapshotV1& after,
    double dt_stage);

/**
 * @brief Opens (truncating) the energy-budget CSV and writes the header.
 */
void open_energy_csv(const std::string& path);

/**
 * @brief Flushes and closes the energy-budget CSV if open (safe when closed).
 */
void close_energy_csv();
