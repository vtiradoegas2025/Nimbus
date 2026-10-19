#pragma once

struct ComputeBackendRuntimeState;

#include "compute/compute_kernel_interfaces.hpp"
#include "numerics/advection/advection_base.hpp"

/**
 * @brief Dispatches the batched conservative WENO5/MC scalar transport on
 *        the Cartesian C-grid (GPU audit Phase 3.2): one submission,
 *        mass flux computed once and per-scalar flux + update on device.
 *        The retained dt_kappas ABI entries must all be zero.
 *
 * @return True if GPU dispatch succeeded. False means caller should use
 *         the per-scalar CPU path.
 */
bool dispatch_advect_scalars_conservative_cartesian_cgrid_backend(
    const float*                                    u,
    const float*                                    v,
    const float*                                    w,
    const float*                                    rho_data,
    const float* const*                             q_in,
    float* const*                                   q_out,
    const float*                                    dt_kappas,
    const int*                                      positive_definite,
    int                                             n_scalars,
    int                                             nx,
    int                                             ny,
    int                                             nz,
    float                                           inv_dx,
    float                                           inv_dy,
    float                                           inv_dz,
    float                                           inv_dx2,
    float                                           inv_dy2,
    float                                           inv_dz2,
    float                                           dt_step,
    float                                           rho_floor,
    const float*                                    density_begin             = nullptr,
    const float*                                    mass_flux_x               = nullptr,
    const float*                                    mass_flux_y               = nullptr,
    const float*                                    mass_flux_z               = nullptr,
    bool                                            use_supplied_mass_flux    = false,
    Field3D* const*                                 host_fields               = nullptr,
    const double*                                   host_kappas               = nullptr,
    const tmv::CartesianCGridTransportContext*      host_context              = nullptr,
    double                                          dt_step_host              = 0.0,
    tmv::ScalarBoundaryTransportReport* const*      host_boundary_reports     = nullptr,
    const tmv::ScalarMoistEnergyKind*               host_energy_kinds         = nullptr,
    tmv::ScalarMoistEnergyTransportReportV1* const* host_energy_reports       = nullptr,
    float                                           frame_velocity_u_ms       = 0.0f,
    float                                           frame_velocity_v_ms       = 0.0f,
    const SimulationStateView*                      execution_state           = nullptr,
    tmv::CartesianInterfaceFluxRegister* const*     host_interface_registers  = nullptr,
    bool                                            weno                      = true,
    bool                                            preserve_positive_support = false);
bool dispatch_advect_scalars_conservative_cartesian_cgrid_backend(
    ComputeBackendRuntimeState&                     compute,
    const float*                                    u,
    const float*                                    v,
    const float*                                    w,
    const float*                                    rho_data,
    const float* const*                             q_in,
    float* const*                                   q_out,
    const float*                                    dt_kappas,
    const int*                                      positive_definite,
    int                                             n_scalars,
    int                                             nx,
    int                                             ny,
    int                                             nz,
    float                                           inv_dx,
    float                                           inv_dy,
    float                                           inv_dz,
    float                                           inv_dx2,
    float                                           inv_dy2,
    float                                           inv_dz2,
    float                                           dt_step,
    float                                           rho_floor,
    const float*                                    density_begin             = nullptr,
    const float*                                    mass_flux_x               = nullptr,
    const float*                                    mass_flux_y               = nullptr,
    const float*                                    mass_flux_z               = nullptr,
    bool                                            use_supplied_mass_flux    = false,
    Field3D* const*                                 host_fields               = nullptr,
    const double*                                   host_kappas               = nullptr,
    const tmv::CartesianCGridTransportContext*      host_context              = nullptr,
    double                                          dt_step_host              = 0.0,
    tmv::ScalarBoundaryTransportReport* const*      host_boundary_reports     = nullptr,
    const tmv::ScalarMoistEnergyKind*               host_energy_kinds         = nullptr,
    tmv::ScalarMoistEnergyTransportReportV1* const* host_energy_reports       = nullptr,
    float                                           frame_velocity_u_ms       = 0.0f,
    float                                           frame_velocity_v_ms       = 0.0f,
    const SimulationStateView*                      execution_state           = nullptr,
    tmv::CartesianInterfaceFluxRegister* const*     host_interface_registers  = nullptr,
    bool                                            weno                      = true,
    bool                                            preserve_positive_support = false);

/**
 * @brief Dispatches Cartesian x-advection via GPU backend if available.
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_advection_x_backend(
    const float* src, const float* u_data, float* dst,
    int nr, int nth, int nz,
    float dx, float dt);
bool dispatch_advection_x_backend(ComputeBackendRuntimeState& compute,
    const float* src, const float* u_data, float* dst,
    int nr, int nth, int nz,
    float dx, float dt);

/**
 * @brief Dispatches Cartesian y-advection via GPU backend if available.
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_advection_y_backend(
    const float* src, const float* v_data, float* dst,
    int nr, int nth, int nz,
    float dy, float dt);
bool dispatch_advection_y_backend(ComputeBackendRuntimeState& compute,
    const float* src, const float* v_data, float* dst,
    int nr, int nth, int nz,
    float dy, float dt);
