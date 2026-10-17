#pragma once

struct ComputeBackendRuntimeState;

#include "compute/compute_kernel_interfaces.hpp"

/** Dispatch production A-grid continuity and its authoritative face fluxes. */
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
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f);
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
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f);

bool dispatch_acoustic_pressure_eos_cartesian_agrid_backend(
    const float* rho_in, const float* theta, const float* drho_dt,
    float* rho_out, float* p_out,
    int nx, int ny, int nz,
    float dt_small, float reference_pressure,
    float gas_constant_dry_air, float gamma,
    float rho_min, float rho_max, float p_min, float p_max,
    TwoFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);
bool dispatch_acoustic_pressure_eos_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho_in, const float* theta, const float* drho_dt,
    float* rho_out, float* p_out,
    int nx, int ny, int nz,
    float dt_small, float reference_pressure,
    float gas_constant_dry_air, float gamma,
    float rho_min, float rho_max, float p_min, float p_max,
    TwoFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);

bool dispatch_acoustic_momentum_cartesian_agrid_backend(
    const float* rho, const float* p,
    const float* p0_base_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz, float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    ThreeFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);
bool dispatch_acoustic_momentum_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho, const float* p,
    const float* p0_base_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz, float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    ThreeFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);

bool dispatch_acoustic_null_mode_constraint_cartesian_agrid_backend(
    const float* rho,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    ThreeFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);
bool dispatch_acoustic_null_mode_constraint_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    ThreeFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);

bool dispatch_acoustic_boundary_cartesian_agrid_backend(
    const float* p0_base_data, const float* rho0_base_data,
    const float* u0_base_data, const float* v0_base_data, int base_len,
    float* u, float* v, float* w, float* rho, float* p,
    int nx, int ny, int nz,
    FiveFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);
bool dispatch_acoustic_boundary_cartesian_agrid_backend(ComputeBackendRuntimeState& compute,
    const float* p0_base_data, const float* rho0_base_data,
    const float* u0_base_data, const float* v0_base_data, int base_len,
    float* u, float* v, float* w, float* rho, float* p,
    int nx, int ny, int nz,
    FiveFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);

/**
 * @brief Dispatches the Cartesian C-grid acoustic pressure substep
 *        (conservative continuity + rho integration + diagnostic p
 *        closure) via GPU backend if available (GPU audit Phase 3.1).
 *
 * rho0_base/p0_base are 1D profiles of length >= nz, broadcast 1D->3D
 * inside the backend.
 *
 * @return True if GPU dispatch succeeded. False means caller should use
 *         the CPU path.
 */
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
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f,
    bool repair_primary_state = true);
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
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f,
    bool repair_primary_state = true);

/**
 * @brief Dispatches the Cartesian C-grid acoustic momentum substep
 *        (perturbation-p face gradients + velocity integration + clamps)
 *        via GPU backend if available (GPU audit Phase 3.1).
 *
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_acoustic_momentum_cartesian_cgrid_backend(
    const float* rho_data, const float* p_data,
    const float* dp0_dz_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    bool repair_primary_state = true,
    const float* dp0_dz_low_data = nullptr);
bool dispatch_acoustic_momentum_cartesian_cgrid_backend(ComputeBackendRuntimeState& compute,
    const float* rho_data, const float* p_data,
    const float* dp0_dz_data, int base_len,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nx, int ny, int nz,
    float dx, float dy, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v,
    bool repair_primary_state = true,
    const float* dp0_dz_low_data = nullptr);

/**
 * @brief Dispatches the fused Cartesian C-grid acoustic substep
 *        (pressure + barrier + momentum) in one GPU submission
 *        (GPU audit Phase 3.1). In-place on the 5 state fields.
 *
 * @return True if GPU dispatch succeeded.
 */
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
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f);
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
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f);

/**
 * @brief Dispatches the whole Cartesian C-grid acoustic loop (all
 *        n_substeps, including the per-substep boundary conditions) in one
 *        GPU submission (GPU audit Phase 4.1).
 *
 * @return True if GPU dispatch succeeded. False means caller should run
 *         the per-substep path.
 */
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
    float* mass_flux_x = nullptr,
    float* mass_flux_y = nullptr,
    float* mass_flux_z = nullptr,
    bool accumulate_mass_flux = false,
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f);
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
    float* mass_flux_x = nullptr,
    float* mass_flux_y = nullptr,
    float* mass_flux_z = nullptr,
    bool accumulate_mass_flux = false,
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f);

/**
 * @brief Dispatches the fused upper/w-only/lateral sponge update via GPU
 *        backend if available. The production FP64 EOS projection is
 *        intentionally CPU-owned on every backend.
 *
 * @return True if GPU dispatch succeeded. False means caller should run
 *         the CPU sequence.
 */
bool dispatch_state_relaxation_cartesian_backend(
    float* u, float* v, float* w,
    float* theta_data,
    const float* profiles, int profiles_len,
    int nx, int ny, int nz,
    int lateral_width,
    bool do_upper, bool do_w_only, bool do_lateral,
    float dt_step,
    FourFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);
bool dispatch_state_relaxation_cartesian_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* theta_data,
    const float* profiles, int profiles_len,
    int nx, int ny, int nz,
    int lateral_width,
    bool do_upper, bool do_w_only, bool do_lateral,
    float dt_step,
    FourFieldCompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr);

/**
 * @brief Applies the split-explicit slow tendencies to u/v/w/p on device
 *        (no clamps, non-finite sums integrate as zero; GPU audit
 *        Phase 4.2).
 *
 * @return True if GPU dispatch succeeded. False means caller should run
 *         the CPU apply loop.
 */
bool dispatch_slow_apply_backend(
    float* u, float* v, float* w, float* p_data,
    const float* du_dt, const float* dv_dt, const float* dw_dt,
    const float* dp_dt,
    const float* du_dt_pbl, const float* dv_dt_pbl,
    int nx, int ny, int nz,
    float dt_large,
    SlowApplyDispatch::CompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr,
    bool sanitize = true);
bool dispatch_slow_apply_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w, float* p_data,
    const float* du_dt, const float* dv_dt, const float* dw_dt,
    const float* dp_dt,
    const float* du_dt_pbl, const float* dv_dt_pbl,
    int nx, int ny, int nz,
    float dt_large,
    SlowApplyDispatch::CompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr,
    bool sanitize = true);

/**
 * @brief Dispatches Cartesian C-grid slow tendencies via GPU backend.
 *
 * 14-buffer SSBO layout with 1D->3D base-state broadcast and
 * has_moisture/has_loading flags. dx/dy map to the dr/dtheta slots.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
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
    bool slow_rk2_invariant_input_residency = false,
    float* du_advection_dt_data = nullptr,
    float* dv_advection_dt_data = nullptr,
    float* dw_advection_dt_data = nullptr,
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f,
    float coriolis_beta = 0.0f);
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
    bool slow_rk2_invariant_input_residency = false,
    float* du_advection_dt_data = nullptr,
    float* dv_advection_dt_data = nullptr,
    float* dw_advection_dt_data = nullptr,
    float frame_velocity_u_ms = 0.0f,
    float frame_velocity_v_ms = 0.0f,
    float coriolis_beta = 0.0f);

/**
 * @brief Dispatches Cartesian momentum tendencies via GPU backend if available.
 *
 * Computes 5 tendency fields from 6 state fields + 2 reference-state profiles.
 * Uses reference-state subtraction in the vertical momentum equation.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
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
    float coriolis_beta = 0.0f,
    bool split_slow_only = false);
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
    float coriolis_beta = 0.0f,
    bool split_slow_only = false);

/**
 * @brief Dispatches fused acoustic momentum substep on GPU.
 *
 * Computes pressure gradient from updated p, integrates u,v,w
 * forward by dt_small, applies boundary conditions.
 *
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_acoustic_momentum_backend(
    const float* rho_data, const float* p_data,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v);
bool dispatch_acoustic_momentum_backend(ComputeBackendRuntimeState& compute,
    const float* rho_data, const float* p_data,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v);

/**
 * @brief Fused acoustic substep: pressure + momentum in one GPU submission.
 *
 * Performs both forward (divergence -> rho/p) and backward (pressure gradient
 * -> u/v/w) phases in a single command buffer with a compute-to-compute
 * barrier. All 5 fields are updated in-place.
 *
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_acoustic_substep_fused_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v);
bool dispatch_acoustic_substep_fused_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v);

/**
 * @brief Batched acoustic substeps: all N substeps in one GPU submission.
 *
 * Records N iterations of (pressure + barrier + momentum + barrier) in a
 * single command buffer. Fields stay GPU-resident across all substeps.
 * GPU shaders handle boundary conditions internally between substeps.
 *
 * @param n_substeps Number of acoustic substeps to execute.
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_acoustic_substeps_batched_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small, int n_substeps,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v);
bool dispatch_acoustic_substeps_batched_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small, int n_substeps,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v);
