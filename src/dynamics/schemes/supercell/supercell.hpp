/**
 * @file supercell.hpp
 * @brief Declarations for the dynamics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the dynamics runtime and scheme implementations.
 * This file is part of the src/dynamics subsystem.
 */

#pragma once
#include "physics/dynamics_base.hpp"
#include <vector>


/**
 * @brief Is the supercell scheme.
 */
class SupercellScheme : public DynamicsScheme
{
public:
    /**
     * @brief Constructs the supercell dynamics scheme.
     */
    SupercellScheme();

    /**
 * @brief Computes the momentum tendencies for the supercell scheme.
 */
    void compute_momentum_tendencies(
        const Field3D& u_r,
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
        Field3D& dp_dt
    ) override;

    /**
 * @brief Computes the vorticity diagnostics for the supercell scheme.
 */
    void compute_vorticity_diagnostics(
        const Field3D& u_r,
        const Field3D& u_theta,
        const Field3D& u_z,
        const Field3D& rho,
        const Field3D& p,
        Field3D& vorticity_r,
        Field3D& vorticity_theta,
        Field3D& vorticity_z,
        Field3D& stretching_term,
        Field3D& tilting_term,
        Field3D& baroclinic_term
    ) override;

    /**
 * @brief Computes the pressure diagnostics for the supercell scheme.
 */
    void compute_pressure_diagnostics(const Field3D& u_r,  const Field3D& u_theta,
        const Field3D& u_z, const Field3D& rho, const Field3D& theta,
        Field3D& p_prime, Field3D& dynamic_pressure, Field3D& buoyancy_pressure
    ) override;

    std::string get_scheme_name() const override { return "supercell"; }
    std::string get_coordinate_system() const override { return "cylindrical"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    /**
     * @brief Computes centered radial derivative at one grid point.
     */
    double compute_dr(const Field3D& field, int i, int j, int k) const;
    /**
     * @brief Computes centered azimuthal derivative at one grid point.
     */
    double compute_dtheta(const Field3D& field, int i, int j, int k) const;
    /**
     * @brief Computes centered vertical derivative at one grid point.
     */
    double compute_dz(const Field3D& field, int i, int j, int k) const;

    /**
     * @brief Computes radial vorticity from local derivative terms.
     */
    double compute_vorticity_r(double dtheta_u_z, double dz_u_theta) const;
    /**
     * @brief Computes azimuthal vorticity from local derivative terms.
     */
    double compute_vorticity_theta(double dz_u_r, double dr_u_z) const;
    /**
     * @brief Computes vertical vorticity from local derivative terms.
     */
    double compute_vorticity_z(double dr_u_theta, double dtheta_u_r, double r) const;

    /**
     * @brief Computes buoyancy acceleration from density/theta perturbation.
     */
    double compute_buoyancy(double theta_prime, double rho, double rho0) const;

    int NR_, NTH_, NZ_;
    double dr_, dtheta_, dz_;
};
