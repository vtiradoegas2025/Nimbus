#pragma once

struct SimulationStateView;

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/physical_constants.hpp"
#include "dynamics/coriolis.hpp"
#include "dynamics/slow_momentum_energy_report.hpp"

/**
 * @file dynamics_base.hpp
 * @brief Abstract interface for dynamics schemes and shared constants.
 *
 * Defines momentum-tendency and diagnostic hooks used by runtime
 * dynamics modules.
 * Also exposes factory creation for concrete scheme selection.
 */

namespace tmv
{

class CartesianCGridMomentumFluxRegister;

class DynamicsScheme
{
public:
    virtual void bind_execution_context(const SimulationStateView&) {}

    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~DynamicsScheme() = default;

    /**
     * Deep-copy this scheme, including reusable execution storage. The
     * fail-closed default preserves source compatibility for external schemes
     * while making unsupported in-memory runtime snapshots explicit.
     */
    virtual std::unique_ptr<DynamicsScheme> clone() const { return nullptr; }

    /** Generation-local scheme scratch must explicitly opt in to recycling. */
    virtual bool supports_domain_recycling_invalidation() const noexcept
    {
        return false;
    }

    /** Clear state derived from the pre-recycle field generation in place. */
    virtual void invalidate_generation_state_after_domain_recycling() noexcept
    {
    }

    /** Install the Earth-referenced Coriolis profile for the next tendency
     * evaluation. External schemes retain f-plane source compatibility by
     * ignoring the profile; built-in Cartesian schemes implement it. */
    virtual void set_coriolis_profile(const CartesianCoriolisProfile&) {}

    /**
     * @brief Computes tendencies for momentum and mass-related fields.
     */
    virtual void compute_momentum_tendencies(const Field3D& u,
                                             const Field3D& v,
                                             const Field3D& w,
                                             const Field3D& rho,
                                             const Field3D& p,
                                             const Field3D& theta,
                                             double dt,
                                             Field3D& du_dt,
                                             Field3D& dv_dt,
                                             Field3D& dw_dt,
                                             Field3D& drho_dt,
                                             Field3D& dp_dt) = 0;

    // -----------------------------------------------------------------

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string get_scheme_name() const = 0;

    /**
     * @brief Returns the coordinate system used by the scheme.
     * @return Coordinate-system name.
     */
    virtual std::string get_coordinate_system() const = 0;

    /**
     * @brief Returns the number of prognostic variables.
     * @return Number of prognostic variables.
     */
    virtual int get_num_prognostic_vars() const = 0;
};

/**
 * @brief Mixin interface for dynamics schemes that support split-explicit
 *        acoustic time stepping (Klemp & Wilhelmson 1978).
 *
 * Schemes that inherit this provide the slow/fast tendency decomposition
 * needed by the split-explicit time stepping algorithm. Schemes that do
 * not support split-explicit (e.g., tornado axisymmetric) do not inherit
 * this and carry no dead virtual methods.
 *
 * Usage: `dynamic_cast<SplitExplicitDynamics*>(scheme)` to check capability.
 */
class SplitExplicitDynamics
{
public:
    virtual ~SplitExplicitDynamics() = default;

    /**
     * Optional accelerator residency hint for the two-evaluation slow-RK2
     * transaction. The orchestrator enables this before stage 1 and disables
     * it after stage 2 only when density, moisture species, hydrometeor
     * loading, and base-state profiles are unchanged between evaluations.
     *
     * Standalone tendency calls retain the safe default (disabled). CPU
     * implementations and schemes without reusable inputs may ignore it.
     */
    virtual void set_slow_rk2_invariant_input_residency(bool) {}

    virtual void set_momentum_interface_flux_register(
        CartesianCGridMomentumFluxRegister*) noexcept {}

    virtual void set_momentum_interface_flux_slow_weight(double weight)
    {
        if (weight != 1.0)
            throw std::runtime_error(
                "dynamics scheme does not support weighted momentum interface flux capture");
    }

    [[nodiscard]] virtual bool momentum_interface_flux_capture_active() const noexcept
    {
        return false;
    }

    /// Computes slow (advective + buoyancy) tendencies only.
    /// Called once per large time step.
    virtual void compute_slow_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p, const Field3D& theta,
        double dt,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
        Field3D& drho_dt, Field3D& dp_dt,
        SlowMomentumAdvectionTendencyOutputs* advection_outputs = nullptr) = 0;

    /// Computes fast pressure/density tendencies (forward phase).
    /// Called N times per large step: dp/dt = -γ p ∇·u, dρ/dt = -ρ ∇·u.
    virtual void compute_fast_pressure_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        double dt_small,
        Field3D& drho_dt, Field3D& dp_dt) = 0;

    /// Computes fast momentum tendencies (backward phase).
    /// Called N times per large step after p/ρ update: du/dt = -(1/ρ)∇p.
    virtual void compute_fast_momentum_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        double dt_small,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt) = 0;

    /**
     * Optional first-class CPU acoustic substep. Implementations may update
     * density/pressure and then momentum in place while keeping one worker
     * team alive across the forward/backward barrier. Returning false selects
     * the separate tendency callbacks above.
     */
    virtual bool apply_fast_substep_cpu_in_place(
        Field3D&, Field3D&, Field3D&,
        Field3D&, Field3D&, const Field3D&,
        double, Field3D&)
    {
        return false;
    }
};

extern std::unique_ptr<DynamicsScheme>& dynamics_scheme;

/**
 * @brief Initializes the dynamics subsystem.
 * @param scheme_name Dynamics scheme identifier.
 */
void initialize_dynamics(const std::string& scheme_name = "tornado");
void initialize_dynamics(const SimulationStateView& execution_state, const std::string& scheme_name = "tornado");

/**
 * @brief Advances dynamics using the active integration path.
 * @param current_time Current simulation time in seconds.
 */
void step_dynamics(double current_time = 0.0);
void step_dynamics(const SimulationStateView& execution_state, double current_time = 0.0);

/**
 * @brief Advances dynamics using the modular dynamics implementation.
 * @param dt_dynamics Dynamics step length in seconds.
 * @param current_time Current simulation time in seconds.
 */
void step_dynamics_new(double dt_dynamics, double current_time = 0.0);
void step_dynamics_new(const SimulationStateView& execution_state, double dt_dynamics, double current_time = 0.0);

/**
 * Reset/log timing-only ownership for the production dynamics orchestration.
 * These counters are inactive unless runtime performance timing is enabled.
 */
void reset_dynamics_perf_stats();
void log_dynamics_perf_summary(double model_step_total_s);

/**
 * @brief Computes dynamics diagnostic fields.
 */
void compute_dynamics_diagnostics(bool compute_vorticity = true,
                                  bool compute_angular_momentum = true,
                                  bool compute_pressure = true);
void compute_dynamics_diagnostics(const SimulationStateView& execution_state, bool compute_vorticity = true,
                                  bool compute_angular_momentum = true,
                                  bool compute_pressure = true);

namespace dynamics_constants
{
inline constexpr double g = physical_constants::gravity_ms2;
constexpr double f_coriolis = 1e-4;
inline constexpr double R_d = physical_constants::gas_constant_dry_air_jkgk;
inline constexpr double cp = physical_constants::specific_heat_cp_jkgk;
inline constexpr double p0 = physical_constants::reference_pressure_pa;
constexpr double theta0 = physical_constants::theta_reference_k;
constexpr double gamma = cp / (cp - R_d);
constexpr double L_v = physical_constants::latent_heat_vaporization_jkg;

constexpr double eps = 1e-8;
constexpr double nu_visc = 0.01;
constexpr double sponge_strength = 0.01;
} // namespace dynamics_constants

} // namespace tmv

#include "dynamics/factory.hpp"

using tmv::DynamicsScheme;
using tmv::SplitExplicitDynamics;
using tmv::create_dynamics_scheme;
using tmv::get_available_dynamics_schemes;
using tmv::dynamics_scheme;
using tmv::initialize_dynamics;
using tmv::step_dynamics;
using tmv::step_dynamics_new;
using tmv::reset_dynamics_perf_stats;
using tmv::log_dynamics_perf_summary;
using tmv::compute_dynamics_diagnostics;
namespace dynamics_constants = tmv::dynamics_constants;
