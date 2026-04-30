/**
 * @file tornado.hpp
 * @brief Declarations for the dynamics module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the dynamics runtime and scheme implementations.
 * This file is part of the src/dynamics subsystem.
 */

#pragma once
#include "dynamics/dynamics_base.hpp"
#include "numerics/derivatives/derivative_operators.hpp"
#include <memory>
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
        const Field3D& u,
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
        Field3D& dp_dt
    ) override;

    /**
     * @brief Computes angular momentum and its local tendency.
     */
    void compute_angular_momentum(
        const Field3D& u,
        const Field3D& v,
        Field3D& angular_momentum,
        Field3D& angular_momentum_tendency
    ) override;

    /**
     * @brief Computes vorticity components and budget diagnostics.
     */
    void compute_vorticity_diagnostics(
        const Field3D& u,
        const Field3D& v,
        const Field3D& w,
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
    /// Computes radial mass flux through an annular control volume.
    double compute_radial_mass_flux(const Field3D& u,
                                   const Field3D& rho,
                                   int i, int k) const;

    int NR_, NTH_, NZ_;
    double dr_, dtheta_, dz_;
    const GridGeometry& geo_;
    std::unique_ptr<DerivativeOperators> deriv_;
};
