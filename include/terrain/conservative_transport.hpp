#pragma once

#include "core/field/field3d.hpp"
#include "init/config/config_v1.hpp"
#include "terrain/terrain_base.hpp"

#include <vector>

namespace tmv::terrain_transport
{

namespace mass_flux_detail
{
struct CollocatedPressureCouplingView
{
    const Field3D* pressure = nullptr;
    const std::vector<double>* reference_pressure = nullptr;
    double dt = 0.0;
};
} // namespace mass_flux_detail

/** Cartesian vector used by terrain surface geometry helpers. */
struct PhysicalVector3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

/** Cartesian symmetric stress tensor in physical coordinates. */
struct SymmetricStressTensor
{
    double xx = 0.0;
    double yy = 0.0;
    double zz = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yz = 0.0;
};

/**
 * Geometry of the physical lower boundary at one scalar column.
 *
 * The upward unit normal is (-dh/dx,-dh/dy,1)/area_ratio.  The unnormalised
 * normal is also the physical area vector per unit horizontal area, so it is
 * the exact boundary counterpart of F^zeta = Fz + Gx*Fx + Gy*Fy.
 */
struct GroundSurfaceFrame
{
    double slope_x = 0.0;
    double slope_y = 0.0;
    double normal_x = 0.0;
    double normal_y = 0.0;
    double normal_z = 1.0;
    double area_ratio = 1.0;
};

/**
 * Convert derivatives at fixed computational coordinates to a Cartesian
 * physical gradient. The inputs are dphi/dx|zeta, dphi/dy|zeta, dphi/dzeta;
 * J=dz/dzeta and Gx=-dz/dx|zeta, Gy=-dz/dy|zeta.
 */
PhysicalVector3 physical_gradient_from_computational(
    double derivative_x_at_zeta,
    double derivative_y_at_zeta,
    double derivative_zeta,
    double jacobian,
    double metric_gx,
    double metric_gy);

/** Checked lower-boundary frame for an interior scalar column. */
GroundSurfaceFrame ground_surface_frame(const TerrainMetrics3D& metrics,
                                        int i, int j);

/** Physical ground velocity satisfying v dot n = 0 for supplied u and v. */
PhysicalVector3 ground_tangent_velocity(const GroundSurfaceFrame& frame,
                                        double physical_u,
                                        double physical_v);

/** Orthogonal projection of a Cartesian vector onto the terrain tangent. */
PhysicalVector3 project_to_ground_tangent(const GroundSurfaceFrame& frame,
                                          const PhysicalVector3& vector);

/**
 * Flux through the lower coordinate surface per unit horizontal area.
 * This is F dot (-dh/dx,-dh/dy,1), with positive orientation upward.
 */
double mapped_ground_normal_flux(const GroundSurfaceFrame& frame,
                                 const PhysicalVector3& physical_flux);

/** Convert physical traction per actual terrain area to horizontal area. */
PhysicalVector3 area_weighted_ground_traction(
    const GroundSurfaceFrame& frame,
    const PhysicalVector3& physical_traction);

/** Jx times the physical stress column crossing an x-coordinate face. */
PhysicalVector3 contravariant_stress_flux_x(
    double face_jacobian,
    const SymmetricStressTensor& physical_stress);

/** Jy times the physical stress column crossing a y-coordinate face. */
PhysicalVector3 contravariant_stress_flux_y(
    double face_jacobian,
    const SymmetricStressTensor& physical_stress);

/**
 * All three momentum fluxes through a zeta face:
 * T dot (Gx,Gy,1). This is the tensor form of
 * F^zeta = Fz + Gx*Fx + Gy*Fy.
 */
PhysicalVector3 contravariant_stress_flux_zeta(
    double metric_gx,
    double metric_gy,
    const SymmetricStressTensor& physical_stress);

/**
 * Physical stress flux through the upward ground coordinate surface, per unit
 * horizontal area.  This applies the same row-wise contravariant transform as
 * the finite-volume interior and preserves all three momentum components.
 */
PhysicalVector3 mapped_ground_stress_traction(
    const GroundSurfaceFrame& frame,
    const SymmetricStressTensor& physical_stress);

/**
 * @brief Validates the complete finite-volume terrain geometry.
 */
void validate_geometry(const TerrainMetrics3D& metrics);

/** Constant-time shape/spacing validation for already-qualified geometry. */
void validate_geometry_shape(const TerrainMetrics3D& metrics);

/**
 * @brief Transforms physical vector fluxes to computational face fluxes.
 *
 *   F^x    = J_x F_x
 *   F^y    = J_y F_y
 *   F^zeta = F_z + G_x F_x + G_y F_y
 *
 * where G_x=-dz/dx and G_y=-dz/dy on zeta faces.
 */
TerrainContravariantFluxes transform_fluxes(
    const TerrainMetrics3D& metrics,
    const TerrainPhysicalFluxes& physical);

/**
 * @brief Returns physical divergence from contravariant face fluxes.
 *
 * The returned cell tendency is (1/J) times the computational flux
 * divergence.  Multiplying it by J and computational cell volume therefore
 * telescopes exactly to the boundary flux.
 */
TerrainField3D conservative_divergence(
    const TerrainMetrics3D& metrics,
    const TerrainContravariantFluxes& fluxes);

/** Validate C-grid state/geometry before entering per-cell transport loops. */
void validate_cgrid_state(const TerrainMetrics3D& metrics,
                          const Field3D& rho,
                          const Field3D& u,
                          const Field3D& v,
                          const Field3D& w);

/** Constant-time C-grid field/geometry shape validation for hot loops. */
void validate_cgrid_shapes(const TerrainMetrics3D& metrics,
                           const Field3D& rho,
                           const Field3D& u,
                           const Field3D& v,
                           const Field3D& w);

/** Jacobian-weighted mass flux on the positive x face owned by cell i. */
double cgrid_mass_flux_x(const TerrainMetrics3D& metrics,
                         const Field3D& rho, const Field3D& u,
                         int i, int j, int k,
                         double frame_u_ms = 0.0);

/** Jacobian-weighted mass flux on the positive y face owned by cell j. */
double cgrid_mass_flux_y(const TerrainMetrics3D& metrics,
                         const Field3D& rho, const Field3D& v,
                         int i, int j, int k,
                         double frame_v_ms = 0.0);

/**
 * Jacobian-weighted contravariant mass flux on the positive zeta face owned
 * by cell k.  The horizontal mass flux is interpolated to the same face.
 */
double cgrid_mass_flux_zeta(const TerrainMetrics3D& metrics,
                            const Field3D& rho,
                            const Field3D& u,
                            const Field3D& v,
                            const Field3D& w,
                            int i, int j, int k,
                            double frame_u_ms = 0.0,
                            double frame_v_ms = 0.0);

/** Physical mass divergence at an interior-horizontal C-grid scalar cell. */
double cgrid_mass_divergence(const TerrainMetrics3D& metrics,
                             const Field3D& rho,
                             const Field3D& u,
                             const Field3D& v,
                             const Field3D& w,
                             int i, int j, int k,
                             double frame_u_ms = 0.0,
                             double frame_v_ms = 0.0);

/** Physical velocity divergence using the identical terrain face geometry. */
double cgrid_velocity_divergence(const TerrainMetrics3D& metrics,
                                 const Field3D& u,
                                 const Field3D& v,
                                 const Field3D& w,
                                 int i, int j, int k,
                                 double frame_u_ms = 0.0,
                                 double frame_v_ms = 0.0);

/** Mapped A-grid mass fluxes stored in the positive-face owner cell. */
double agrid_mass_flux_x(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& u,
    int i, int j, int k,
    tmv::init::config::LateralBC west,
    tmv::init::config::LateralBC east,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling = nullptr,
    double frame_u_ms = 0.0);
double agrid_mass_flux_y(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& v,
    int i, int j, int k,
    tmv::init::config::LateralBC south,
    tmv::init::config::LateralBC north,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling = nullptr,
    double frame_v_ms = 0.0);
double agrid_mass_flux_zeta(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    int i, int j, int k,
    tmv::init::config::BottomBC bottom,
    tmv::init::config::TopBC top,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling = nullptr,
    double frame_u_ms = 0.0,
    double frame_v_ms = 0.0);
double agrid_mass_divergence(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    int i, int j, int k,
    tmv::init::config::LateralBC west,
    tmv::init::config::LateralBC east,
    tmv::init::config::LateralBC south,
    tmv::init::config::LateralBC north,
    tmv::init::config::BottomBC bottom,
    tmv::init::config::TopBC top,
    const mass_flux_detail::CollocatedPressureCouplingView* pressure_coupling = nullptr,
    double frame_u_ms = 0.0,
    double frame_v_ms = 0.0);
double agrid_velocity_divergence(
    const TerrainMetrics3D& metrics,
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    int i, int j, int k,
    tmv::init::config::LateralBC west,
    tmv::init::config::LateralBC east,
    tmv::init::config::LateralBC south,
    tmv::init::config::LateralBC north,
    tmv::init::config::BottomBC bottom,
    tmv::init::config::TopBC top,
    double frame_u_ms = 0.0,
    double frame_v_ms = 0.0);

/**
 * Sample a one-dimensional scalar-centred reference profile at physical
 * height.  The returned value is rounded through fp32 so a Field3D initialized
 * from the same function subtracts to bit-exact zero.
 */
double reference_profile_value_fp32(const std::vector<double>& profile,
                                    double physical_height,
                                    double nominal_dz);

/** Dry-EOS reference density from locally sampled stored-float p and theta. */
double reference_density_fp32(const std::vector<double>& reference_pressure,
                              const std::vector<double>& reference_theta,
                              double physical_height,
                              double nominal_dz);

/** Physical gradient of a cell-centred scalar on terrain-following geometry. */
double cgrid_scalar_gradient_x(const TerrainMetrics3D& metrics,
                               const Field3D& scalar,
                               int i, int j, int k);
double cgrid_scalar_gradient_y(const TerrainMetrics3D& metrics,
                               const Field3D& scalar,
                               int i, int j, int k);
double cgrid_scalar_gradient_z(const TerrainMetrics3D& metrics,
                               const Field3D& scalar,
                               int i, int j, int k);
double cgrid_scalar_gradient_z_center(const TerrainMetrics3D& metrics,
                                      const Field3D& scalar,
                                      int i, int j, int k);

/**
 * Conservative anisotropic diffusion of a cell-centred scalar on the mapped
 * grid. The operator is the negative weighted adjoint of the physical
 * rho*K_h*(d/dx,d/dy) and rho*K_v*d/dz face gradients, so constants are exact
 * null states and the mass-weighted scalar variance cannot increase.
 * Lateral boundaries may be closed or periodic independently. Periodic seams
 * wrap only the physical cells (1..N-2), never closure storage. The output is
 * specific tendency and closure storage is deterministically zero.
 */
void cgrid_scalar_diffusion_tendency(
    const TerrainMetrics3D& metrics,
    const Field3D& scalar,
    const Field3D& rho,
    const Field3D* K_vertical,
    double K_h,
    double K_v,
    bool periodic_x,
    bool periodic_y,
    Field3D& tendency);

/** Mapped live-density scalar diffusion with cell-varying SGS coefficients. */
void cgrid_variable_scalar_diffusion_tendency(
    const TerrainMetrics3D& metrics,
    const Field3D& scalar,
    const Field3D& rho,
    const Field3D& K_horizontal,
    const Field3D& K_vertical,
    bool use_horizontal,
    bool use_vertical,
    bool periodic_x,
    bool periodic_y,
    Field3D& tendency);

/** Conservative sufficient explicit-Euler diffusion number for the operator. */
double cgrid_scalar_diffusion_number(
    const TerrainMetrics3D& metrics,
    const Field3D& rho,
    const Field3D* K_vertical,
    double K_h,
    double K_v,
    double dt,
    bool periodic_x,
    bool periodic_y);

/**
 * Computational vertical velocity d(zeta)/dt, expressed in metres of zeta
 * per second, at each C-grid location.  These implement
 * (w - u_rel dz/dx - v_rel dz/dy) / J using location-consistent geometry.
 * Callers pass physical velocities relative to the translating grid.
 */
double cgrid_zeta_velocity_at_x_face(const TerrainMetrics3D& metrics,
                                     double physical_u,
                                     double physical_v,
                                     double physical_w,
                                     int i, int j, int k);
double cgrid_zeta_velocity_at_y_face(const TerrainMetrics3D& metrics,
                                     double physical_u,
                                     double physical_v,
                                     double physical_w,
                                     int i, int j, int k);
double cgrid_zeta_velocity_at_z_face(const TerrainMetrics3D& metrics,
                                     double physical_u,
                                     double physical_v,
                                     double physical_w,
                                     int i, int j, int k);

/**
 * Physical vertical velocity of an impermeable terrain surface.  The stored
 * C-grid has an implicit lower zeta face, so no-penetration is represented by
 * zero contravariant flux while its physical velocity is
 * w_ground = (u_ground - U_frame) dh/dx
 *          + (v_ground - V_frame) dh/dy.
 * This defines terrain as grid-attached: a translating computational window
 * carries its terrain map with it.  Earth-fixed terrain remapping is a
 * different operation and is not implied by this interface.
 */
double cgrid_ground_vertical_velocity(const TerrainMetrics3D& metrics,
                                      const Field3D& u,
                                      const Field3D& v,
                                      int i, int j,
                                      double frame_u_ms = 0.0,
                                      double frame_v_ms = 0.0);

/**
 * Well-balanced physical pressure gradient.  The horizontally uniform base
 * profile is sampled at each column's physical height and subtracted before
 * differencing.  A state initialized from that same reference therefore has
 * exactly zero pressure acceleration on arbitrary qualified terrain.
 */
double cgrid_pressure_perturbation_gradient_x(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k);
double cgrid_pressure_perturbation_gradient_y(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k);
double cgrid_pressure_perturbation_gradient_z(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k);

/** Cell-centred physical gradient for mapped collocated momentum/transport. */
PhysicalVector3 agrid_pressure_perturbation_gradient_center(
    const TerrainMetrics3D& metrics,
    const Field3D& pressure,
    const std::vector<double>& reference_pressure,
    int i, int j, int k,
    bool periodic_x = false,
    bool periodic_y = false);

PhysicalVector3 agrid_scalar_gradient_center(
    const TerrainMetrics3D& metrics,
    const Field3D& scalar,
    int i, int j, int k,
    bool periodic_x = false,
    bool periodic_y = false);

} // namespace tmv::terrain_transport

namespace terrain_transport = tmv::terrain_transport;
