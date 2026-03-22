/**
 * @file tornado.hpp
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
 * @brief Cylindrical-coordinate dynamics scheme for tornado-scale flow.
 */
class TornadoScheme : public DynamicsScheme
{
public:
    /**
     * @brief Constructs a tornado dynamics scheme with default metrics.
     */
    TornadoScheme();

    /**
     * @brief Computes momentum, density, and pressure tendencies.
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
     * @brief Computes angular momentum and its local tendency.
     */
    void compute_angular_momentum(
        const Field3D& u_r,
        const Field3D& u_theta,
        Field3D& angular_momentum,
        Field3D& angular_momentum_tendency
    ) override;

    /**
     * @brief Computes vorticity components and budget diagnostics.
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

    std::string get_scheme_name() const override { return "tornado"; }
    std::string get_coordinate_system() const override { return "cylindrical"; }
    int get_num_prognostic_vars() const override { return 5; }

private:
    /**
     * @brief Computes centered radial derivative at one grid point.
     */
    double compute_dr(const Field3D& field, int i, int j, int k) const;
    /**
     * @brief Computes centered vertical derivative at one grid point.
     */
    double compute_dz(const Field3D& field, int i, int j, int k) const;

    /**
     * @brief Computes radial mass flux through an annular control volume.
     */
    double compute_radial_mass_flux(const Field3D& u_r,
                                   const Field3D& rho,
                                   int i, int k) const;

    int NR_, NTH_, NZ_;
    double dr_, dtheta_, dz_;
};
