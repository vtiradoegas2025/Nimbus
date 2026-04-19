/**
 * @file cartesian.hpp
 * @brief Declarations for the Cartesian dynamics scheme.
 *
 * Cartesian (x, y, z) implementation of the DynamicsScheme interface.
 * Selected when `coordinate_system: cartesian` is set in runtime config.
 *
 * Field reuse (parallel-fields-then-unify — see docs/CoordinateBackend_Plan.md
 * Phase A.1 / docs/Journey.md Bug 7):
 *   - The first field argument (formal name `u_r`) carries u_x.
 *   - The second field argument (formal name `u_theta`) carries u_y.
 *   - The third field argument (formal name `u_z`) carries u_z (unchanged).
 *
 * Phase B will rename the field globals (u -> u_x, v_theta -> u_y) and update
 * the base interface accordingly. Until then we keep the cylindrical-flavored
 * signature so the existing call sites in `src/core/dynamics.cpp` do not have
 * to change, and both backends can coexist.
 *
 * This file is part of the src/dynamics subsystem.
 */

#pragma once
#include "physics/dynamics_base.hpp"
#include <vector>

/**
 * @brief Cartesian-grid dynamics scheme (Phase A of the Coordinate Backend
 *        Plan). Computes momentum, mass, and pressure tendencies using pure
 *        (x, y, z) finite differences. Removes the cylindrical terms
 *        (centrifugal, 1/r factors, azimuthal periodic wrap, axis-singularity
 *        guards) that cannot be evaluated correctly for non-axisymmetric
 *        base states such as the WK2002 supercell hodograph.
 */
class CartesianScheme : public DynamicsScheme
{
public:
    /**
     * @brief Constructs the Cartesian dynamics scheme, caching grid dims.
     */
    CartesianScheme();

    /**
     * @brief Computes the momentum tendencies on a Cartesian grid.
     *        See the class-level note on field aliasing.
     */
    void compute_momentum_tendencies(const Field3D& u_r,
                                     const Field3D& u_theta,
                                     const Field3D& u_z,
                                     const Field3D& rho,
                                     const Field3D& p,
                                     const Field3D& theta,
                                     double dt,
                                     Field3D& du_r_dt,
                                     Field3D& du_theta_dt,
                                     Field3D& du_z_dt,
                                     Field3D& drho_dt,
                                     Field3D& dp_dt) override;

    /**
     * @brief Computes Cartesian vorticity diagnostics (ω = ∇ × u) along with
     *        stretching, tilting, and the vertical baroclinic term.
     */
    void compute_vorticity_diagnostics(const Field3D& u_r,
                                       const Field3D& u_theta,
                                       const Field3D& u_z,
                                       const Field3D& rho,
                                       const Field3D& p,
                                       Field3D& vorticity_r,
                                       Field3D& vorticity_theta,
                                       Field3D& vorticity_z,
                                       Field3D& stretching_term,
                                       Field3D& tilting_term,
                                       Field3D& baroclinic_term) override;

    /**
     * @brief Computes pressure-decomposition diagnostics on a Cartesian grid.
     */
    void compute_pressure_diagnostics(const Field3D& u_r,
                                      const Field3D& u_theta,
                                      const Field3D& u_z,
                                      const Field3D& rho,
                                      const Field3D& theta,
                                      Field3D& p_prime,
                                      Field3D& dynamic_pressure,
                                      Field3D& buoyancy_pressure) override;

    // Split-explicit acoustic substep support
    bool supports_split_acoustic() const override { return true; }
    void compute_slow_tendencies(
        const Field3D& u_r, const Field3D& u_theta, const Field3D& u_z,
        const Field3D& rho, const Field3D& p, const Field3D& theta, double dt,
        Field3D& du_r_dt, Field3D& du_theta_dt, Field3D& du_z_dt,
        Field3D& drho_dt, Field3D& dp_dt) override;
    void compute_fast_pressure_tendencies(
        const Field3D& u_r, const Field3D& u_theta, const Field3D& u_z,
        const Field3D& rho, const Field3D& p,
        Field3D& drho_dt, Field3D& dp_dt) override;
    void compute_fast_momentum_tendencies(
        const Field3D& u_r, const Field3D& u_theta, const Field3D& u_z,
        const Field3D& rho, const Field3D& p,
        Field3D& du_r_dt, Field3D& du_theta_dt, Field3D& du_z_dt) override;

    std::string get_scheme_name() const override { return "cartesian"; }
    std::string get_coordinate_system() const override { return "cartesian"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    /**
     * @brief Centered ∂/∂x derivative on the Cartesian grid.
     *
     * Phase A uses square horizontal cells: dx = dr_. A separate dy/dx
     * global will be introduced in Phase B (or when a non-square horizontal
     * grid becomes a requirement). The plan explicitly defers this.
     */
    double compute_dx(const Field3D& field, int i, int j, int k) const;

    /**
     * @brief Centered ∂/∂y derivative on the Cartesian grid.
     *        dy = dr_ in Phase A (square horizontal cells).
     */
    double compute_dy(const Field3D& field, int i, int j, int k) const;

    /**
     * @brief Centered ∂/∂z derivative on the Cartesian grid.
     */
    double compute_dz(const Field3D& field, int i, int j, int k) const;

    int NR_;
    int NTH_;
    int NZ_;
    double dr_;
    double dz_;
};
