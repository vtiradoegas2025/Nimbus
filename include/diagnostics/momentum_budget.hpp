/**
 * @file momentum_budget.hpp
 * @brief Domain-integrated momentum budget diagnostics (F.5b).
 *
 * Companion to conservation_budget.hpp (mass/water/energy). Tracks the
 * domain-integrated momentum inventory so the F.6.C conservation gate can
 * assert |dp/p| < 1e-3 over a 1 hr simulation, and so per-stage drift can be
 * attributed to the physics step that caused it (the same before/after model
 * the mass budget uses).
 *
 * Inventory (single pass over all physical cells, uniform Cartesian cell volume
 * dx * dy * dz):
 *
 *   momentum_x = integral(rho * u * dV)
 *   momentum_y = integral(rho * v * dV)
 *     -- the Cartesian (frame-invariant) linear momentum components. In a
 *        closed, balanced domain these stay near zero; net horizontal force
 *        (Coriolis + pressure gradient + boundary stress) is what moves them.
 *
 *   momentum_z = integral(rho * w * dV)
 *     -- vertical momentum; changed by the buoyancy/weight residual and the
 *        vertical pressure-gradient force.
 *
 *   angular_momentum_z = integral(rho * (x*v - y*u) * dV)
 *     -- axial angular momentum about the domain-center vertical axis, where
 *        (x, y) are offsets from that center; the central invariant for vortex
 *        work (mesocyclone, tornadogenesis): conserved absent net vertical
 *        torque.
 *
 * Collocated velocities are read directly over i/j/k=[1,N-2]; its six edge
 * planes are boundary closure storage. The production C-grid carries the same
 * lateral scalar closure but all vertical scalar levels are physical. Face
 * velocities at those physical scalar centers are
 * reconstructed by averaging the two adjacent faces; the maintained west and
 * south boundary faces therefore still contribute to their adjacent physical
 * cells. The surface vertical velocity uses the rigid lower face w=0.
 *
 * Per-term forcing attribution (compute_momentum_forcing) records the momentum
 * sources that are *exactly* recoverable from a single snapshot:
 *   - weight_z   = -integral(rho * g * dV)              (gravity on w)
 *   - buoyancy_z = -integral(g * (rho - rho0(z)) * dV)  (buoyancy on w)
 *   - coriolis_x =  f * momentum_y, coriolis_y = -f * momentum_x
 *                                                       (f-plane Coriolis)
 * The pressure-gradient and advection contributions are boundary-flux terms
 * (their volume integrals reduce to surface integrals); for a closed domain
 * advection conserves momentum and the pressure-gradient net equals the
 * boundary pressure force, so both are captured implicitly by the inventory
 * drift rather than re-derived here.
 *
 * The legacy no-argument API reads the process-wide state through an adapter;
 * the explicit overloads accept a `SimulationStateView` so tests and future
 * runtime owners can supply state directly.
 */

#pragma once

#include <string>

struct SimulationStateView;

struct HorizontalMomentumBudgetExclusion
{
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;
};

/**
 * @brief Domain-integrated momentum inventory snapshot.
 *
 * `total_mass` and the two reference scales are carried so the drift can be
 * normalized by a *characteristic* momentum rather than by the instantaneous
 * inventory: net momentum passes through zero on a quiescent domain, where a
 * naive relative drift (divide by |inventory|) is ill-conditioned. See
 * evaluate_momentum_drift.
 */
struct MomentumBudget
{
    double momentum_x         = 0.0;  // integral(rho * u_cartesian_x * dV)
    double momentum_y         = 0.0;  // integral(rho * u_cartesian_y * dV)
    double momentum_z         = 0.0;  // integral(rho * w * dV)
    double angular_momentum_z = 0.0;  // integral(rho * (x*v - y*u) * dV)

    // Derived magnitudes (the F.6.C gate is on horizontal/total magnitude).
    double horizontal_magnitude = 0.0;  // sqrt(momentum_x^2 + momentum_y^2)
    double total_magnitude      = 0.0;  // sqrt(mx^2 + my^2 + mz^2)

    // Normalization support (see evaluate_momentum_drift).
    double total_mass         = 0.0;  // integral(rho * dV)  [kg]
    double reference_linear   = 0.0;  // total_mass * V_ref            [kg m/s]
    double reference_angular  = 0.0;  // total_mass * V_ref * R_domain [kg m^2/s]
};

/**
 * @brief Exactly-recoverable momentum forcing integrals (attribution aid).
 *
 * See file docstring. These are diagnostic decompositions, not used by the
 * closure gate, but verifiable analytically and useful for locating drift.
 */
struct MomentumForcing
{
    double weight_z   = 0.0;  // -integral(rho * g * dV)            [N]
    double buoyancy_z = 0.0;  // -integral(g * (rho - rho0) * dV)   [N]
    double coriolis_x = 0.0;  //  coriolis_f * momentum_y           [N]
    double coriolis_y = 0.0;  // -coriolis_f * momentum_x           [N]
};

/**
 * @brief Per-stage momentum drift between two snapshots + warn flag.
 *
 * Relative drifts are normalized by a characteristic momentum, not by the
 * instantaneous inventory: linear components by max(|before|, reference_linear)
 * and angular by max(|before|, reference_angular). This keeps the metric
 * well-conditioned when the net momentum is near zero (a quiescent domain),
 * while reducing to a true relative drift once the flow carries real momentum.
 */
struct MomentumDrift
{
    double d_momentum_x         = 0.0;
    double d_momentum_y         = 0.0;
    double d_momentum_z         = 0.0;
    double d_angular_momentum_z = 0.0;

    double rel_momentum_x         = 0.0;
    double rel_momentum_y         = 0.0;
    double rel_momentum_z         = 0.0;
    double rel_angular_momentum_z = 0.0;

    bool warn = false;  // any relative drift > BUDGET_THRESH_MOMENTUM
};

// Per-step momentum closure threshold. Set to the F.6.C 1-hr gate value
// (|dp/p| < 1e-3); per-step drift on a balanced fixture sits far below this,
// so the warn line only fires on a real conservation defect.
constexpr double BUDGET_THRESH_MOMENTUM = 1.0e-3;

// Characteristic velocity defining the "effectively calm" floor. Momentum
// drift is measured against max(|inventory|, total_mass * this), so a domain
// whose mean wind is below this carries a near-zero momentum that does not
// generate an ill-conditioned (divide-by-almost-zero) relative drift. Above
// it, the actual inventory dominates and the gate is a true relative measure.
constexpr double MOMENTUM_REF_VELOCITY_MS = 1.0;

/**
 * @brief Computes the domain-integrated momentum inventory.
 *
 * Single pass over physical cells with the uniform Cartesian cell volume.
 * C-grid velocities are reconstructed at scalar centers without scratch
 * allocation and the one-cell lateral scalar halo is excluded; collocated
 * velocities and their full indexed domain are unchanged. Returns an all-zero
 * snapshot only when all required fields are unallocated. Partial/mismatched
 * allocation and non-finite or nonphysical physical-domain samples fail
 * closed.
 */
MomentumBudget compute_momentum_budget(const SimulationStateView& state);
MomentumBudget compute_momentum_budget();

// Physical dry-air momentum includes the implicit terrain-ground half-volume.
// The historical budget above retains the stored-face convention for its ABI.
MomentumBudget compute_physical_momentum_budget_v2(const SimulationStateView& state);

/**
 * Compute the identical production inventory outside one horizontal storage
 * rectangle. C-grid face winds are still reconstructed at each contributing
 * scalar center. Angular momentum retains the full-domain-center reference so
 * subset inventories are additive within one domain-origin transaction.
 */
MomentumBudget compute_momentum_budget_excluding_horizontal_rectangle(
    const SimulationStateView& state,
    const HorizontalMomentumBudgetExclusion& exclusion);

/**
 * @brief Computes the exactly-recoverable momentum forcing integrals.
 */
MomentumForcing compute_momentum_forcing(const SimulationStateView& state);
MomentumForcing compute_momentum_forcing();

/**
 * @brief Computes per-stage drift and the warn flag. Pure function.
 */
MomentumDrift evaluate_momentum_drift(const MomentumBudget& before,
                                      const MomentumBudget& after);

/**
 * @brief Logs the momentum transition (WARN when drift exceeds threshold,
 *        debug otherwise) and appends a CSV row when a stream is open.
 *
 * @param stage    Physics-stage label (e.g., "dynamics", "total_step").
 * @param before   Snapshot before the stage.
 * @param after    Snapshot after the stage.
 * @param dt_stage Stage duration in seconds (ignored when <= 0).
 */
void report_momentum_transition(const SimulationStateView& state,
                                const char* stage,
                                const MomentumBudget& before,
                                const MomentumBudget& after,
                                double dt_stage);
void report_momentum_transition(const char* stage,
                                const MomentumBudget& before,
                                const MomentumBudget& after,
                                double dt_stage);

/**
 * @brief Opens (truncating) the momentum-budget CSV time series and writes the
 *        header. Safe to re-call. No-op-on-failure with a warning.
 */
void open_momentum_csv(const std::string& path);

/**
 * @brief Flushes and closes the momentum-budget CSV if open (safe when closed).
 */
void close_momentum_csv();
