#pragma once

#include <memory>
#include <string>

#include "core/field/field3d.hpp"
#include "core/infra/physical_constants.hpp"

/**
 * @file dynamics_base.hpp
 * @brief Abstract interface for dynamics schemes and shared constants.
 *
 * Defines momentum-tendency and diagnostic hooks used by runtime
 * dynamics modules in cylindrical tornado simulations.
 * Also exposes factory creation for concrete scheme selection.
 */

class DynamicsScheme
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~DynamicsScheme() = default;

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

    /**
     * @brief Computes vorticity-budget diagnostics.
     */
    virtual void compute_vorticity_diagnostics(const Field3D& u,
                                               const Field3D& v,
                                               const Field3D& w,
                                               const Field3D& rho,
                                               const Field3D& p,
                                               Field3D& vorticity_r,
                                               Field3D& vorticity_theta,
                                               Field3D& vorticity_z,
                                               Field3D& stretching_term,
                                               Field3D& tilting_term,
                                               Field3D& baroclinic_term)
    {
    }

    /**
     * @brief Computes angular-momentum diagnostics.
     */
    virtual void compute_angular_momentum(const Field3D& u,
                                          const Field3D& v,
                                          Field3D& angular_momentum,
                                          Field3D& angular_momentum_tendency)
    {
    }

    /**
     * @brief Computes pressure-decomposition diagnostics.
     */
    virtual void compute_pressure_diagnostics(const Field3D& u,
                                              const Field3D& v,
                                              const Field3D& w,
                                              const Field3D& rho,
                                              const Field3D& theta,
                                              Field3D& p_prime,
                                              Field3D& dynamic_pressure,
                                              Field3D& buoyancy_pressure)
    {
    }

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

    /// Computes slow (advective + buoyancy) tendencies only.
    /// Called once per large time step.
    virtual void compute_slow_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p, const Field3D& theta,
        double dt,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt,
        Field3D& drho_dt, Field3D& dp_dt) = 0;

    /// Computes fast pressure/density tendencies (forward phase).
    /// Called N times per large step: dp/dt = -γ p ∇·u, dρ/dt = -ρ ∇·u.
    virtual void compute_fast_pressure_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        Field3D& drho_dt, Field3D& dp_dt) = 0;

    /// Computes fast momentum tendencies (backward phase).
    /// Called N times per large step after p/ρ update: du/dt = -(1/ρ)∇p.
    virtual void compute_fast_momentum_tendencies(
        const Field3D& u, const Field3D& v, const Field3D& w,
        const Field3D& rho, const Field3D& p,
        Field3D& du_dt, Field3D& dv_dt, Field3D& dw_dt) = 0;
};

/**
 * @brief Creates a dynamics scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<DynamicsScheme> create_dynamics_scheme(const std::string& scheme_name);

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
