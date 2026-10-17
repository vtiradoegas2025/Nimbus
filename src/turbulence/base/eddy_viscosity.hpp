/**
 * @file eddy_viscosity.hpp
 * @brief Declarations for the turbulence module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the turbulence runtime and scheme implementations.
 * This file is part of the src/turbulence subsystem.
 */

#pragma once
#include <vector>
#include "turbulence/turbulence_base.hpp"
#include "core/field/field3d.hpp"

namespace tmv
{
namespace eddy_viscosity
{

/**
 * @brief Components and magnitude of the local 3D strain-rate tensor.
 */
struct StrainRate
{
    double S11, S12, S13;
    double S21, S22, S23;
    double S31, S32, S33;
    double magnitude;
};

/**
 * @brief True when the j (y) direction is topologically periodic: the
 *        Cartesian y axis is periodic only with Periodic y-faces.
 *        Single source of truth for the SGS stencils' j-neighbor rule
 *        (SCIENCE.md M16) - the strain, both diffusion Laplacians,
 *        and the GPU smagorinsky shaders (via push constant) follow it.
 */
bool periodic_y_topology();
inline bool periodic_y_topology(const TurbulenceStateView& state) noexcept { return state.periodic_y; }

/** True when the physical C-grid x rows wrap from 1 to NR-2. */
bool periodic_x_topology();
inline bool periodic_x_topology(const TurbulenceStateView& state) noexcept { return state.periodic_x; }

/**
 * @brief Validates the complete state/configuration/metric contract consumed
 *        by an active SGS scheme. Invalid science inputs are rejected before
 *        any scheme-owned output is changed.
 */
void validate_scheme_inputs(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    const TurbulenceStateView& state,
    bool require_tke
);

/**
 * @brief Computes the 3D strain rate tensor at a grid point.
 *
 * Collocated-read formulation retained for flat-grid reference oracles; the
 * production paths use
 * compute_strain_sq_centered (SCIENCE.md M14). Active terrain rejects
 * until the staggered physical tensor operator is implemented.
 */
StrainRate compute_strain_rate_3d(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k
);

/**
 * @brief |S|^2 = 2 S_ij S_ij at the cell center from STAGGERED-NATURAL
 *        components (SCIENCE.md M14): the diagonal terms are the
 *        natural face-to-center differences (du/dx from the cell's own
 *        x-faces, etc.); the off-diagonal terms live on cell edges and
 *        their SQUARES are averaged to the center (4 surrounding
 *        edges). Resolves single-cell (2dx) shear the collocated read
 *        was blind to, with no O(dx) component misregistration.
 *        j-edge neighbors follow periodic_y_topology(); the flat ground
 *        closure uses w(ground) = 0 and free-slip du/dz = dv/dz = 0.
 *        Active terrain uses physical component heights, the impermeable
 *        physical ground velocity, and Gx/Gy corrections at the natural
 *        diagonal/edge locations.
 */
double compute_strain_sq_centered(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k
);

/** Mode-consistent non-negative stress contraction used for eddy viscosity. */
double compute_directional_strain_sq_centered(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k,
    const std::string& mode
);

/**
 * @brief Computes the physical filter width (Smagorinsky mixing length).
 *        Terrain cubic-root mode uses dx*dy*J*dzeta, not a slope-inflated
 *        face-normal CFL distance.
 */
double compute_filter_width(
    const TurbulenceConfig& cfg,
    const GridMetrics& grid,
    int i, int j, int k
);

/**
 * @brief Computes the Smagorinsky viscosity.
 */
double compute_smagorinsky_viscosity(
    double Cs,
    double Delta,
    double strain_mag,
    double stability_factor = 1.0
);

/**
 * @brief Computes the eddy diffusivities from the eddy viscosity.
 */
void compute_eddy_diffusivities(
    double nu_t,
    double Pr_t,
    double Sc_t,
    double& K_theta,
    double& K_q,
    double& K_tke
);

/**
 * @brief Computes conservative flat or terrain-mapped C-grid momentum
 *        diffusion tendencies. Terrain uses contravariant tensor stresses,
 *        exact Jacobian-weighted dual masses, and zero mapped ground/lid
 *        traction.
 */
void compute_momentum_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& nu_t,
    Field3D& dudt_sgs,
    Field3D& dvdt_sgs,
    Field3D& dwdt_sgs,
    bool periodic_x,
    bool periodic_y,
    double* mapped_boundary_power_w = nullptr
);

/** Compatibility overload using the process boundary topology. */
inline void compute_momentum_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& nu_t,
    Field3D& dudt_sgs,
    Field3D& dvdt_sgs,
    Field3D& dwdt_sgs,
    double* mapped_boundary_power_w = nullptr)
{
    compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu_t, dudt_sgs, dvdt_sgs, dwdt_sgs,
        periodic_x_topology(state), periodic_y_topology(state),
        mapped_boundary_power_w);
}

/**
 * @brief Computes the scalar diffusion tendencies.
 */
void compute_scalar_diffusion_tendencies(
    const TurbulenceConfig& cfg,
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    const Field3D& K_field,
    const Field3D& phi,
    Field3D& dphi_dt_sgs
);

/**
 * @brief Computes the stability correction.
 */
double stability_correction_ri(
    double Ri,
    double Ri_crit = 0.25
);

double lilly_equilibrium_strain_rate(double strain_squared, double buoyancy_frequency_squared,
                                    double turbulent_prandtl);

/**
 * @brief Computes the Brunt-Väisälä frequency.
 */
double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k,
    bool moist = false
);

double compute_buoyancy_frequency_squared(
    const TurbulenceStateView& state, const GridMetrics& grid,
    int i, int j, int k, bool moist);

double cgrid_stability_factor_ri(
    const TurbulenceStateView& state,
    const GridMetrics& grid,
    int i, int j, int k,
    bool moist = false
);

/**
 * @brief Compatibility overload that fails closed because physical grid
 *        metrics are required.
 */
double compute_brunt_vaisala_frequency(
    const TurbulenceStateView& state,
    int i, int j, int k
);

/**
 * @brief Applies the positivity limits.
 */
void apply_positivity_limits(
    Field3D& field,
    double min_value = 0.0,
    double max_value = 1e6
);

} // namespace eddy_viscosity
} // namespace tmv

namespace eddy_viscosity = tmv::eddy_viscosity;
