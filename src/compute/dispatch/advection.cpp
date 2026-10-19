#include "compute/bridge/advection.hpp"

#include "compute/compute_backend.hpp"

namespace
{
AdvectionComputeKernels* active_advection_kernels(ComputeBackendRuntimeState& compute)
{
    ComputeBackend* backend = mutable_compute_backend(compute);
    return backend != nullptr ? backend->advection_kernels() : nullptr;
}
} // namespace

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
    const float*                                    density_begin,
    const float*                                    mass_flux_x,
    const float*                                    mass_flux_y,
    const float*                                    mass_flux_z,
    bool                                            use_supplied_mass_flux,
    Field3D* const*                                 host_fields,
    const double*                                   host_kappas,
    const tmv::CartesianCGridTransportContext*      host_context,
    double                                          dt_step_host,
    tmv::ScalarBoundaryTransportReport* const*      host_boundary_reports,
    const tmv::ScalarMoistEnergyKind*               host_energy_kinds,
    tmv::ScalarMoistEnergyTransportReportV1* const* host_energy_reports,
    float                                           frame_velocity_u_ms,
    float                                           frame_velocity_v_ms,
    const SimulationStateView*                      execution_state,
    tmv::CartesianInterfaceFluxRegister* const*     host_interface_registers,
    bool                                            weno,
    bool                                            preserve_positive_support)
{
    return dispatch_advect_scalars_conservative_cartesian_cgrid_backend(global_compute_backend_runtime_state,
                                                                        u,
                                                                        v,
                                                                        w,
                                                                        rho_data,
                                                                        q_in,
                                                                        q_out,
                                                                        dt_kappas,
                                                                        positive_definite,
                                                                        n_scalars,
                                                                        nx,
                                                                        ny,
                                                                        nz,
                                                                        inv_dx,
                                                                        inv_dy,
                                                                        inv_dz,
                                                                        inv_dx2,
                                                                        inv_dy2,
                                                                        inv_dz2,
                                                                        dt_step,
                                                                        rho_floor,
                                                                        density_begin,
                                                                        mass_flux_x,
                                                                        mass_flux_y,
                                                                        mass_flux_z,
                                                                        use_supplied_mass_flux,
                                                                        host_fields,
                                                                        host_kappas,
                                                                        host_context,
                                                                        dt_step_host,
                                                                        host_boundary_reports,
                                                                        host_energy_kinds,
                                                                        host_energy_reports,
                                                                        frame_velocity_u_ms,
                                                                        frame_velocity_v_ms,
                                                                        execution_state,
                                                                        host_interface_registers,
                                                                        weno,
                                                                        preserve_positive_support);
}

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
    const float*                                    density_begin,
    const float*                                    mass_flux_x,
    const float*                                    mass_flux_y,
    const float*                                    mass_flux_z,
    bool                                            use_supplied_mass_flux,
    Field3D* const*                                 host_fields,
    const double*                                   host_kappas,
    const tmv::CartesianCGridTransportContext*      host_context,
    double                                          dt_step_host,
    tmv::ScalarBoundaryTransportReport* const*      host_boundary_reports,
    const tmv::ScalarMoistEnergyKind*               host_energy_kinds,
    tmv::ScalarMoistEnergyTransportReportV1* const* host_energy_reports,
    float                                           frame_velocity_u_ms,
    float                                           frame_velocity_v_ms,
    const SimulationStateView*                      execution_state,
    tmv::CartesianInterfaceFluxRegister* const*     host_interface_registers,
    bool                                            weno,
    bool                                            preserve_positive_support)
{
    AdvectionComputeKernels* kernels = active_advection_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(AdvectionKernelOp::ConservativeCartesianCgridTransport))
    {
        if (use_supplied_mass_flux
            && (!density_begin || !mass_flux_x || !mass_flux_y || !mass_flux_z
                || !kernels->supports_supplied_cartesian_cgrid_mass_flux()))
        {
            return false;
        }
        const ConservativeCartesianCgridTransportDispatch dispatch{u,
                                                                   v,
                                                                   w,
                                                                   rho_data,
                                                                   q_in,
                                                                   q_out,
                                                                   dt_kappas,
                                                                   positive_definite,
                                                                   n_scalars,
                                                                   nx,
                                                                   ny,
                                                                   nz,
                                                                   inv_dx,
                                                                   inv_dy,
                                                                   inv_dz,
                                                                   inv_dx2,
                                                                   inv_dy2,
                                                                   inv_dz2,
                                                                   dt_step,
                                                                   rho_floor,
                                                                   density_begin,
                                                                   mass_flux_x,
                                                                   mass_flux_y,
                                                                   mass_flux_z,
                                                                   use_supplied_mass_flux,
                                                                   host_fields,
                                                                   host_kappas,
                                                                   host_context,
                                                                   dt_step_host,
                                                                   host_boundary_reports,
                                                                   host_energy_kinds,
                                                                   host_energy_reports,
                                                                   frame_velocity_u_ms,
                                                                   frame_velocity_v_ms,
                                                                   execution_state,
                                                                   host_interface_registers,
                                                                   weno,
                                                                   preserve_positive_support};
        return kernels->dispatch_advect_scalars_conservative_cartesian_cgrid(dispatch);
    }
    return false;
}

bool dispatch_advection_x_backend(
    const float* src, const float* u_data, float* dst,
    int nr, int nth, int nz,
    float dx, float dt)
{
    return dispatch_advection_x_backend(global_compute_backend_runtime_state, src, u_data, dst, nr, nth, nz, dx, dt);
}

bool dispatch_advection_x_backend(ComputeBackendRuntimeState& compute,
    const float* src, const float* u_data, float* dst,
    int nr, int nth, int nz,
    float dx, float dt)
{
    AdvectionComputeKernels* kernels = active_advection_kernels(compute);
    if (kernels != nullptr && kernels->supports(AdvectionKernelOp::AdvectionX))
    {
        const ScalarAdvectionDispatch dispatch{
            src, u_data, dst,
            nr, nth, nz,
            dx, 0.0f, dt
        };
        return kernels->dispatch_advection_x(dispatch);
    }
    return false;
}

bool dispatch_advection_y_backend(
    const float* src, const float* v_data, float* dst,
    int nr, int nth, int nz,
    float dy, float dt)
{
    return dispatch_advection_y_backend(global_compute_backend_runtime_state, src, v_data, dst, nr, nth, nz, dy, dt);
}

bool dispatch_advection_y_backend(ComputeBackendRuntimeState& compute,
    const float* src, const float* v_data, float* dst,
    int nr, int nth, int nz,
    float dy, float dt)
{
    AdvectionComputeKernels* kernels = active_advection_kernels(compute);
    if (kernels != nullptr && kernels->supports(AdvectionKernelOp::AdvectionY))
    {
        const ScalarAdvectionDispatch dispatch{
            src, v_data, dst,
            nr, nth, nz,
            dy, 0.0f, dt
        };
        return kernels->dispatch_advection_y(dispatch);
    }
    return false;
}
