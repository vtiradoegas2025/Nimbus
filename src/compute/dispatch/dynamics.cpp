#include "compute/bridge/dynamics.hpp"

#include "compute/compute_backend.hpp"

namespace
{
DynamicsComputeKernels* active_dynamics_kernels(ComputeBackendRuntimeState& compute)
{
    ComputeBackend* backend = mutable_compute_backend(compute);
    return backend != nullptr ? backend->dynamics_kernels() : nullptr;
}
} // namespace

bool dispatch_acoustic_continuity_mass_flux_cartesian_agrid_backend(
    const float* u, const float* v, const float* w,
    const float* rho, const float* p,
    const float* p0_base_data, int base_len,
    float* drho_dt,
    float* mass_flux_x, float* mass_flux_y, float* mass_flux_z,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float dt_small,
    int lateral_bc_west, int lateral_bc_east,
    int lateral_bc_south, int lateral_bc_north,
    int top_bc, bool accumulate_mass_flux, float mass_flux_weight,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms)
{
    return dispatch_acoustic_continuity_mass_flux_cartesian_agrid_backend(global_compute_backend_runtime_state, u, v, w, rho, p, p0_base_data, base_len, drho_dt, mass_flux_x, mass_flux_y, mass_flux_z, nx, ny, nz, dx, dy, dz, dt_small, lateral_bc_west, lateral_bc_east, lateral_bc_south, lateral_bc_north, top_bc, accumulate_mass_flux, mass_flux_weight, frame_velocity_u_ms, frame_velocity_v_ms);
}

bool dispatch_acoustic_continuity_mass_flux_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* u, const float* v, const float* w,
    const float* rho, const float* p,
    const float* p0_base_data, int base_len,
    float* drho_dt,
    float* mass_flux_x, float* mass_flux_y, float* mass_flux_z,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float dt_small,
    int lateral_bc_west, int lateral_bc_east,
    int lateral_bc_south, int lateral_bc_north,
    int top_bc, bool accumulate_mass_flux, float mass_flux_weight,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels == nullptr || !kernels->supports(
            DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid))
    {
        return false;
    }
    return kernels->dispatch_acoustic_continuity_mass_flux_cartesian_agrid({
        u, v, w, rho, p, p0_base_data, base_len,
        drho_dt, mass_flux_x, mass_flux_y, mass_flux_z,
        nx, ny, nz, dx, dy, dz, dt_small,
        lateral_bc_west, lateral_bc_east,
        lateral_bc_south, lateral_bc_north,
        top_bc, accumulate_mass_flux, mass_flux_weight,
        frame_velocity_u_ms, frame_velocity_v_ms});
}

bool dispatch_acoustic_pressure_eos_cartesian_agrid_backend(
    const float* rho_in, const float* theta, const float* drho_dt,
    float* rho_out, float* p_out,
    int nx, int ny, int nz,
    float dt_small, float reference_pressure,
    float gas_constant_dry_air, float gamma,
    float rho_min, float rho_max, float p_min, float p_max,
    TwoFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    return dispatch_acoustic_pressure_eos_cartesian_agrid_backend(global_compute_backend_runtime_state, rho_in, theta, drho_dt, rho_out, p_out, nx, ny, nz, dt_small, reference_pressure, gas_constant_dry_air, gamma, rho_min, rho_max, p_min, p_max, completed_observer, completed_observer_context);
}

bool dispatch_acoustic_pressure_eos_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho_in, const float* theta, const float* drho_dt,
    float* rho_out, float* p_out,
    int nx, int ny, int nz,
    float dt_small, float reference_pressure,
    float gas_constant_dry_air, float gamma,
    float rho_min, float rho_max, float p_min, float p_max,
    TwoFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    return kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::AcousticPressureEosCartesianAgrid) &&
        kernels->dispatch_acoustic_pressure_eos_cartesian_agrid({
            rho_in, theta, drho_dt, rho_out, p_out,
            nx, ny, nz, dt_small, reference_pressure,
            gas_constant_dry_air, gamma,
            rho_min, rho_max, p_min, p_max,
            completed_observer, completed_observer_context});
}

bool dispatch_acoustic_momentum_cartesian_agrid_backend(
    const float* rho, const float* p,
    const float* p0_base_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz, float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    ThreeFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    return dispatch_acoustic_momentum_cartesian_agrid_backend(global_compute_backend_runtime_state, rho, p, p0_base_data, base_len, u_in, v_in, w_in, u_out, v_out, w_out, nx, ny, nz, dx, dy, dz, dt_small, wind_clamp_h, wind_clamp_v, completed_observer, completed_observer_context);
}

bool dispatch_acoustic_momentum_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho, const float* p,
    const float* p0_base_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz, float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    ThreeFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    return kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::AcousticMomentumCartesianAgrid) &&
        kernels->dispatch_acoustic_momentum_cartesian_agrid({
            rho, p, p0_base_data, base_len,
            u_in, v_in, w_in, u_out, v_out, w_out,
            nx, ny, nz, dx, dy, dz, dt_small,
            wind_clamp_h, wind_clamp_v,
            completed_observer, completed_observer_context});
}

bool dispatch_acoustic_null_mode_constraint_cartesian_agrid_backend(
    const float* rho,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    ThreeFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    return dispatch_acoustic_null_mode_constraint_cartesian_agrid_backend(global_compute_backend_runtime_state, rho, qv, qc, qr, qi, qs, qg, qh, u_in, v_in, w_in, u_out, v_out, w_out, nx, ny, nz, completed_observer, completed_observer_context);
}

bool dispatch_acoustic_null_mode_constraint_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    ThreeFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    return kernels != nullptr && kernels->supports(
            DynamicsKernelOp::AcousticNullModeConstraintCartesianAgrid) &&
        kernels->dispatch_acoustic_null_mode_constraint_cartesian_agrid({
            rho, qv, qc, qr, qi, qs, qg, qh,
            u_in, v_in, w_in, u_out, v_out, w_out,
            nx, ny, nz, completed_observer, completed_observer_context});
}

bool dispatch_acoustic_boundary_cartesian_agrid_backend(
    const float* p0_base_data, const float* rho0_base_data,
    const float* u0_base_data, const float* v0_base_data, int base_len,
    float* u, float* v, float* w, float* rho, float* p,
    int nx, int ny, int nz,
    FiveFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    return dispatch_acoustic_boundary_cartesian_agrid_backend(global_compute_backend_runtime_state, p0_base_data, rho0_base_data, u0_base_data, v0_base_data, base_len, u, v, w, rho, p, nx, ny, nz, completed_observer, completed_observer_context);
}

bool dispatch_acoustic_boundary_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* p0_base_data, const float* rho0_base_data,
    const float* u0_base_data, const float* v0_base_data, int base_len,
    float* u, float* v, float* w, float* rho, float* p,
    int nx, int ny, int nz,
    FiveFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    return kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::AcousticBoundaryCartesianAgrid) &&
        kernels->dispatch_acoustic_boundary_cartesian_agrid({
            p0_base_data, rho0_base_data, u0_base_data, v0_base_data,
            base_len, u, v, w, rho, p, nx, ny, nz,
            completed_observer, completed_observer_context});
}

bool dispatch_acoustic_pressure_cartesian_cgrid_backend(
    const float* u, const float* v, const float* w,
    const float* rho_in, const float* p_in,
    const float* rho0_base_data, const float* p0_base_data, int base_len,
    const float* theta_data,
    float* rho_out, float* p_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    double gamma_val, float dt_small,
    float rho_min, float rho_max,
    float p_min, float p_max,
    float p_ref_fallback,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms,
    bool repair_primary_state)
{
    return dispatch_acoustic_pressure_cartesian_cgrid_backend(global_compute_backend_runtime_state, u, v, w, rho_in, p_in, rho0_base_data, p0_base_data, base_len, theta_data, rho_out, p_out, nx, ny, nz, dx, dy, dz, gamma_val, dt_small, rho_min, rho_max, p_min, p_max, p_ref_fallback, frame_velocity_u_ms, frame_velocity_v_ms, repair_primary_state);
}

bool dispatch_acoustic_pressure_cartesian_cgrid_backend(ComputeBackendRuntimeState& compute,
    const float* u, const float* v, const float* w,
    const float* rho_in, const float* p_in,
    const float* rho0_base_data, const float* p0_base_data, int base_len,
    const float* theta_data,
    float* rho_out, float* p_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    double gamma_val, float dt_small,
    float rho_min, float rho_max,
    float p_min, float p_max,
    float p_ref_fallback,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms,
    bool repair_primary_state)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::AcousticPressureCartesianCgrid))
    {
        const AcousticPressureCartesianCgridDispatch dispatch{
            u, v, w,
            rho_in, p_in,
            rho0_base_data, p0_base_data, base_len,
            theta_data,
            rho_out, p_out,
            nx, ny, nz,
            dx, dy, dz,
            gamma_val, dt_small,
            rho_min, rho_max,
            p_min, p_max,
            p_ref_fallback,
            frame_velocity_u_ms, frame_velocity_v_ms, repair_primary_state
        };
        return kernels->dispatch_acoustic_pressure_cartesian_cgrid(dispatch);
    }
    return false;
}

bool dispatch_acoustic_momentum_cartesian_cgrid_backend(
    const float* rho_data, const float* p_data,
    const float* dp0_dz_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    bool repair_primary_state,
    const float* dp0_dz_low_data)
{
    return dispatch_acoustic_momentum_cartesian_cgrid_backend(global_compute_backend_runtime_state, rho_data, p_data, dp0_dz_data, base_len, u_in, v_in, w_in, u_out, v_out, w_out, nx, ny, nz, dx, dy, dz, dt_small, wind_clamp_h, wind_clamp_v, repair_primary_state, dp0_dz_low_data);
}

bool dispatch_acoustic_momentum_cartesian_cgrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho_data, const float* p_data,
    const float* dp0_dz_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    bool repair_primary_state,
    const float* dp0_dz_low_data)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::AcousticMomentumCartesianCgrid))
    {
        const AcousticMomentumCartesianCgridDispatch dispatch{
            rho_data, p_data,
            dp0_dz_data, base_len,
            u_in, v_in, w_in,
            u_out, v_out, w_out,
            nx, ny, nz,
            dx, dy, dz,
            dt_small,
            wind_clamp_h, wind_clamp_v, repair_primary_state, dp0_dz_low_data
        };
        return kernels->dispatch_acoustic_momentum_cartesian_cgrid(dispatch);
    }
    return false;
}

bool dispatch_acoustic_substep_fused_cartesian_cgrid_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    const float* theta_data,
    const float* rho0_base_data, const float* p0_base_data,
    const float* dp0_dz_data, int base_len,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float gamma_val, float dt_small,
    float rho_min, float rho_max,
    float p_min, float p_max,
    float p_ref_fallback,
    float wind_clamp_h, float wind_clamp_v,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms)
{
    return dispatch_acoustic_substep_fused_cartesian_cgrid_backend(global_compute_backend_runtime_state, u, v, w, rho, p, theta_data, rho0_base_data, p0_base_data, dp0_dz_data, base_len, nx, ny, nz, dx, dy, dz, gamma_val, dt_small, rho_min, rho_max, p_min, p_max, p_ref_fallback, wind_clamp_h, wind_clamp_v, frame_velocity_u_ms, frame_velocity_v_ms);
}

bool dispatch_acoustic_substep_fused_cartesian_cgrid_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* rho, float* p,
    const float* theta_data,
    const float* rho0_base_data, const float* p0_base_data,
    const float* dp0_dz_data, int base_len,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float gamma_val, float dt_small,
    float rho_min, float rho_max,
    float p_min, float p_max,
    float p_ref_fallback,
    float wind_clamp_h, float wind_clamp_v,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid))
    {
        AcousticSubstepFusedCartesianCgridDispatch dispatch{
            u, v, w, rho, p,
            theta_data,
            rho0_base_data, p0_base_data, dp0_dz_data, base_len,
            nx, ny, nz,
            dx, dy, dz,
            gamma_val, dt_small,
            rho_min, rho_max,
            p_min, p_max,
            p_ref_fallback,
            wind_clamp_h, wind_clamp_v
        };
        dispatch.frame_velocity_u_ms = frame_velocity_u_ms;
        dispatch.frame_velocity_v_ms = frame_velocity_v_ms;
        return kernels->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch);
    }
    return false;
}

bool dispatch_acoustic_substeps_batched_cartesian_cgrid_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    const float* theta_data,
    const float* rho0_base_data, const float* p0_base_data,
    const float* dp0_dz_data, int base_len,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float gamma_val, float dt_small, int n_substeps,
    float rho_min, float rho_max,
    float p_min, float p_max,
    float p_ref_fallback,
    float wind_clamp_h, float wind_clamp_v,
    int bc_west, int bc_east, int bc_south, int bc_north,
    int bc_bottom, int bc_top,
    float* mass_flux_x,
    float* mass_flux_y,
    float* mass_flux_z,
    bool accumulate_mass_flux,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms)
{
    return dispatch_acoustic_substeps_batched_cartesian_cgrid_backend(global_compute_backend_runtime_state, u, v, w, rho, p, theta_data, rho0_base_data, p0_base_data, dp0_dz_data, base_len, nx, ny, nz, dx, dy, dz, gamma_val, dt_small, n_substeps, rho_min, rho_max, p_min, p_max, p_ref_fallback, wind_clamp_h, wind_clamp_v, bc_west, bc_east, bc_south, bc_north, bc_bottom, bc_top, mass_flux_x, mass_flux_y, mass_flux_z, accumulate_mass_flux, frame_velocity_u_ms, frame_velocity_v_ms);
}

bool dispatch_acoustic_substeps_batched_cartesian_cgrid_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* rho, float* p,
    const float* theta_data,
    const float* rho0_base_data, const float* p0_base_data,
    const float* dp0_dz_data, int base_len,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float gamma_val, float dt_small, int n_substeps,
    float rho_min, float rho_max,
    float p_min, float p_max,
    float p_ref_fallback,
    float wind_clamp_h, float wind_clamp_v,
    int bc_west, int bc_east, int bc_south, int bc_north,
    int bc_bottom, int bc_top,
    float* mass_flux_x,
    float* mass_flux_y,
    float* mass_flux_z,
    bool accumulate_mass_flux,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid))
    {
        if (accumulate_mass_flux
            && (!mass_flux_x || !mass_flux_y || !mass_flux_z
                || !kernels->supports_batched_cartesian_cgrid_mass_flux_export()))
        {
            return false;
        }
        const AcousticSubstepsBatchedCartesianCgridDispatch dispatch{
            u, v, w, rho, p,
            theta_data,
            rho0_base_data, p0_base_data, dp0_dz_data, base_len,
            nx, ny, nz, dx, dy, dz,
            gamma_val, dt_small, n_substeps,
            rho_min, rho_max, p_min, p_max,
            p_ref_fallback,
            wind_clamp_h, wind_clamp_v,
            bc_west, bc_east, bc_south, bc_north,
            bc_bottom, bc_top,
            mass_flux_x, mass_flux_y, mass_flux_z,
            accumulate_mass_flux,
            frame_velocity_u_ms, frame_velocity_v_ms
        };
        return kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(dispatch);
    }
    return false;
}

bool dispatch_state_relaxation_cartesian_backend(
    float* u, float* v, float* w,
    float* theta_data,
    const float* profiles, int profiles_len,
    int nx, int ny, int nz,
    int lateral_width,
    bool do_upper, bool do_w_only, bool do_lateral,
    float dt_step,
    FourFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    return dispatch_state_relaxation_cartesian_backend(global_compute_backend_runtime_state, u, v, w, theta_data, profiles, profiles_len, nx, ny, nz, lateral_width, do_upper, do_w_only, do_lateral, dt_step, completed_observer, completed_observer_context);
}

bool dispatch_state_relaxation_cartesian_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* theta_data,
    const float* profiles, int profiles_len,
    int nx, int ny, int nz,
    int lateral_width,
    bool do_upper, bool do_w_only, bool do_lateral,
    float dt_step,
    FourFieldCompletedObserver completed_observer,
    void* completed_observer_context)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(DynamicsKernelOp::StateRelaxationCartesian))
    {
        const StateRelaxationDispatch dispatch{
            u, v, w, theta_data,
            profiles, profiles_len,
            nx, ny, nz, lateral_width,
            do_upper, do_w_only, do_lateral,
            dt_step,
            completed_observer,
            completed_observer_context
        };
        return kernels->dispatch_state_relaxation_cartesian(dispatch);
    }
    return false;
}

bool dispatch_slow_apply_backend(
    float* u, float* v, float* w, float* p_data,
    const float* du_dt, const float* dv_dt, const float* dw_dt,
    const float* dp_dt,
    const float* du_dt_pbl, const float* dv_dt_pbl,
    int nx, int ny, int nz,
    float dt_large,
    SlowApplyDispatch::CompletedObserver completed_observer,
    void* completed_observer_context,
    bool sanitize)
{
    return dispatch_slow_apply_backend(global_compute_backend_runtime_state, u, v, w, p_data, du_dt, dv_dt, dw_dt, dp_dt, du_dt_pbl, dv_dt_pbl, nx, ny, nz, dt_large, completed_observer, completed_observer_context, sanitize);
}

bool dispatch_slow_apply_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w, float* p_data,
    const float* du_dt, const float* dv_dt, const float* dw_dt,
    const float* dp_dt,
    const float* du_dt_pbl, const float* dv_dt_pbl,
    int nx, int ny, int nz,
    float dt_large,
    SlowApplyDispatch::CompletedObserver completed_observer,
    void* completed_observer_context,
    bool sanitize)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr && kernels->supports(DynamicsKernelOp::SlowApply))
    {
        const SlowApplyDispatch dispatch{
            u, v, w, p_data,
            du_dt, dv_dt, dw_dt, dp_dt,
            du_dt_pbl, dv_dt_pbl,
            nx, ny, nz,
            dt_large,
            completed_observer,
            completed_observer_context,
            sanitize
        };
        return kernels->dispatch_slow_apply(dispatch);
    }
    return false;
}

bool dispatch_cartesian_cgrid_slow_tendencies_backend(
    const float* u_data, const float* v_data, const float* w_data,
    const float* rho_data, const float* p_data,
    const float* rho0_base_data, int rho0_base_len,
    const float* qv0_base_data, int qv0_base_len,
    const float* u0_base_data, const float* v0_base_data,
    int wind0_base_len,
    const float* qv_data,
    const float* loading_data,
    float* du_dt_data, float* dv_dt_data, float* dw_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dx, float dy, float dz,
    float g, float coriolis_f_val,
    bool has_moisture, bool has_loading,
    bool slow_rk2_invariant_input_residency,
    float* du_advection_dt_data,
    float* dv_advection_dt_data,
    float* dw_advection_dt_data,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms,
    float coriolis_beta)
{
    return dispatch_cartesian_cgrid_slow_tendencies_backend(global_compute_backend_runtime_state, u_data, v_data, w_data, rho_data, p_data, rho0_base_data, rho0_base_len, qv0_base_data, qv0_base_len, u0_base_data, v0_base_data, wind0_base_len, qv_data, loading_data, du_dt_data, dv_dt_data, dw_dt_data, drho_dt_data, dp_dt_data, nr, nth, nz, dx, dy, dz, g, coriolis_f_val, has_moisture, has_loading, slow_rk2_invariant_input_residency, du_advection_dt_data, dv_advection_dt_data, dw_advection_dt_data, frame_velocity_u_ms, frame_velocity_v_ms, coriolis_beta);
}

bool dispatch_cartesian_cgrid_slow_tendencies_backend(ComputeBackendRuntimeState& compute,
    const float* u_data, const float* v_data, const float* w_data,
    const float* rho_data, const float* p_data,
    const float* rho0_base_data, int rho0_base_len,
    const float* qv0_base_data, int qv0_base_len,
    const float* u0_base_data, const float* v0_base_data,
    int wind0_base_len,
    const float* qv_data,
    const float* loading_data,
    float* du_dt_data, float* dv_dt_data, float* dw_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dx, float dy, float dz,
    float g, float coriolis_f_val,
    bool has_moisture, bool has_loading,
    bool slow_rk2_invariant_input_residency,
    float* du_advection_dt_data,
    float* dv_advection_dt_data,
    float* dw_advection_dt_data,
    float frame_velocity_u_ms,
    float frame_velocity_v_ms,
    float coriolis_beta)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr && kernels->supports(DynamicsKernelOp::CartesianCgridSlowTendencies))
    {
        const CartesianCgridSlowTendenciesDispatch dispatch{
            u_data, v_data, w_data,
            rho_data, p_data,
            rho0_base_data, rho0_base_len,
            qv0_base_data, qv0_base_len,
            u0_base_data, v0_base_data, wind0_base_len,
            qv_data,
            loading_data,
            du_dt_data, dv_dt_data, dw_dt_data,
            drho_dt_data, dp_dt_data,
            nr, nth, nz,
            dx, dy, dz,
            g, coriolis_f_val,
            has_moisture, has_loading,
            slow_rk2_invariant_input_residency,
            du_advection_dt_data,
            dv_advection_dt_data,
            dw_advection_dt_data,
            frame_velocity_u_ms,
            frame_velocity_v_ms,
            coriolis_beta
        };
        return kernels->dispatch_cartesian_cgrid_slow_tendencies(dispatch);
    }
    return false;
}

bool dispatch_cartesian_tendencies_backend(
    const float* u_x_data, const float* u_y_data, const float* w_data,
    const float* rho_data, const float* p_data, const float* theta_data,
    const float* p0_base_data, const float* rho0_base_data,
    const float* loading_data,
    const float* u0_base_data, const float* v0_base_data,
    const float* qv_data, const float* qv0_base_data,
    float* du_x_dt_data, float* du_y_dt_data, float* dw_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dx, float dy, float dz,
    float g, float gamma_val, float coriolis_f_val,
    int lateral_bc_west, int lateral_bc_east,
    int lateral_bc_south, int lateral_bc_north,
    int top_bc, bool has_moisture,
    float coriolis_beta,
    bool split_slow_only)
{
    return dispatch_cartesian_tendencies_backend(global_compute_backend_runtime_state, u_x_data, u_y_data, w_data, rho_data, p_data, theta_data, p0_base_data, rho0_base_data, loading_data, u0_base_data, v0_base_data, qv_data, qv0_base_data, du_x_dt_data, du_y_dt_data, dw_dt_data, drho_dt_data, dp_dt_data, nr, nth, nz, dx, dy, dz, g, gamma_val, coriolis_f_val, lateral_bc_west, lateral_bc_east, lateral_bc_south, lateral_bc_north, top_bc, has_moisture, coriolis_beta, split_slow_only);
}

bool dispatch_cartesian_tendencies_backend(ComputeBackendRuntimeState& compute,
    const float* u_x_data, const float* u_y_data, const float* w_data,
    const float* rho_data, const float* p_data, const float* theta_data,
    const float* p0_base_data, const float* rho0_base_data,
    const float* loading_data,
    const float* u0_base_data, const float* v0_base_data,
    const float* qv_data, const float* qv0_base_data,
    float* du_x_dt_data, float* du_y_dt_data, float* dw_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dx, float dy, float dz,
    float g, float gamma_val, float coriolis_f_val,
    int lateral_bc_west, int lateral_bc_east,
    int lateral_bc_south, int lateral_bc_north,
    int top_bc, bool has_moisture,
    float coriolis_beta,
    bool split_slow_only)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr && kernels->supports(DynamicsKernelOp::CartesianTendencies))
    {
        const CartesianTendenciesDispatch dispatch{
            u_x_data, u_y_data, w_data,
            rho_data, p_data, theta_data,
            p0_base_data, rho0_base_data,
            loading_data,
            u0_base_data, v0_base_data,
            qv_data, qv0_base_data,
            du_x_dt_data, du_y_dt_data, dw_dt_data,
            drho_dt_data, dp_dt_data,
            nr, nth, nz,
            dx, dy, dz,
            g, gamma_val, coriolis_f_val,
            lateral_bc_west, lateral_bc_east,
            lateral_bc_south, lateral_bc_north,
            top_bc, has_moisture,
            coriolis_beta,
            split_slow_only
        };
        return kernels->dispatch_cartesian_tendencies(dispatch);
    }
    return false;
}

bool dispatch_acoustic_momentum_backend(
    const float* rho_data, const float* p_data,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v)
{
    return dispatch_acoustic_momentum_backend(global_compute_backend_runtime_state, rho_data, p_data, u_in, v_in, w_in, u_out, v_out, w_out, nr, nth, nz, dr, dtheta, dz, dt_small, wind_clamp_h, wind_clamp_v);
}

bool dispatch_acoustic_momentum_backend(ComputeBackendRuntimeState& compute,
    const float* rho_data, const float* p_data,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr && kernels->supports(DynamicsKernelOp::AcousticMomentum))
    {
        const AcousticMomentumDispatch dispatch{
            rho_data, p_data, u_in, v_in, w_in,
            u_out, v_out, w_out, nr, nth, nz,
            dr, dtheta, dz, dt_small,
            wind_clamp_h, wind_clamp_v
        };
        return kernels->dispatch_acoustic_momentum(dispatch);
    }
    return false;
}

bool dispatch_acoustic_substep_fused_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v)
{
    return dispatch_acoustic_substep_fused_backend(global_compute_backend_runtime_state, u, v, w, rho, p, nr, nth, nz, dr, dtheta, dz, gamma_val, dt_small, rho_floor, p_floor, wind_clamp_h, wind_clamp_v);
}

bool dispatch_acoustic_substep_fused_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr && kernels->supports(DynamicsKernelOp::AcousticSubstepFused))
    {
        const AcousticSubstepFusedDispatch dispatch{
            u, v, w, rho, p, nr, nth, nz,
            dr, dtheta, dz, gamma_val, dt_small,
            rho_floor, p_floor, wind_clamp_h, wind_clamp_v
        };
        return kernels->dispatch_acoustic_substep_fused(dispatch);
    }
    return false;
}

bool dispatch_acoustic_substeps_batched_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small, int n_substeps,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v)
{
    return dispatch_acoustic_substeps_batched_backend(global_compute_backend_runtime_state, u, v, w, rho, p, nr, nth, nz, dr, dtheta, dz, gamma_val, dt_small, n_substeps, rho_floor, p_floor, wind_clamp_h, wind_clamp_v);
}

bool dispatch_acoustic_substeps_batched_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small, int n_substeps,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v)
{
    DynamicsComputeKernels* kernels = active_dynamics_kernels(compute);
    if (kernels != nullptr && kernels->supports(DynamicsKernelOp::AcousticSubstepsBatched))
    {
        const AcousticSubstepsBatchedDispatch dispatch{
            u, v, w, rho, p, nr, nth, nz,
            dr, dtheta, dz, gamma_val, dt_small, n_substeps,
            rho_floor, p_floor, wind_clamp_h, wind_clamp_v
        };
        return kernels->dispatch_acoustic_substeps_batched(dispatch);
    }
    return false;
}
