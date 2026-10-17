#pragma once

struct ComputeBackendRuntimeState;

#include "compute/compute_kernel_interfaces.hpp"

/**
 * @brief Runs the fused Smagorinsky chain (eddy coefficients + SGS
 *        tendencies in one submission, GPU audit Phase 4.3c).
 *
 * @return True if GPU dispatch succeeded. False means caller should run
 *         the CPU path (or the split dispatches).
 */
bool dispatch_smagorinsky_fused_backend(
    const float* u_data, const float* v_data, const float* w_data,
    const float* theta_data, const float* qv_data,
    const float* dz_prof, const float* dzc_prof,
    const float* rho, int prof_len,
    float* nu_t, float* K_theta, float* K_q,
    float* dudt_sgs, float* dvdt_sgs, float* dwdt_sgs,
    float* dthetadt_sgs, float* dqvdt_sgs,
    int nx, int ny, int nz,
    int filter_mode, bool use_horizontal, bool use_vertical, bool has_qv,
    bool periodic_x, bool periodic_y,
    float dx_val, float dy_val,
    float Cs, float Pr_t, float Sc_t,
    float nu_t_max, float K_max,
    bool download_coefficients = true,
    std::array<bool, 4> open_lateral = {{true, true, true, true}});
bool dispatch_smagorinsky_fused_backend(ComputeBackendRuntimeState& compute,
    const float* u_data, const float* v_data, const float* w_data,
    const float* theta_data, const float* qv_data,
    const float* dz_prof, const float* dzc_prof,
    const float* rho, int prof_len,
    float* nu_t, float* K_theta, float* K_q,
    float* dudt_sgs, float* dvdt_sgs, float* dwdt_sgs,
    float* dthetadt_sgs, float* dqvdt_sgs,
    int nx, int ny, int nz,
    int filter_mode, bool use_horizontal, bool use_vertical, bool has_qv,
    bool periodic_x, bool periodic_y,
    float dx_val, float dy_val,
    float Cs, float Pr_t, float Sc_t,
    float nu_t_max, float K_max,
    bool download_coefficients = true,
    std::array<bool, 4> open_lateral = {{true, true, true, true}});

/**
 * @brief Applies the SGS turbulence tendencies to u/v/w/theta/qv/tke on
 *        device. sanitize selects guarded/clamped versus raw Warn/Strict
 *        candidates; the optional observer runs after the fence and before
 *        host copy-back (GPU audit Phase 4.2).
 *
 * @return True if GPU dispatch succeeded. False means caller should run
 *         the CPU integrate loop.
 */
bool dispatch_sgs_apply_backend(
    float* u, float* v, float* w,
    float* theta_data, float* qv_data, float* tke_data,
    const float* dudt, const float* dvdt, const float* dwdt,
    const float* dthetadt, const float* dqvdt, const float* dtkedt,
    const float* dtkedt_pbl,
    int nx, int ny, int nz,
    bool has_qv, bool has_tke,
    float dt_step, float theta_fallback,
    float theta_min, float theta_max, float qv_max_val,
    float wind_h_max, float wind_v_max,
    bool resident_tendencies_unchanged,
    bool sanitize = true,
    SgsApplyDispatch::CompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr,
    bool collocated = false);
bool dispatch_sgs_apply_backend(ComputeBackendRuntimeState& compute,
    float* u, float* v, float* w,
    float* theta_data, float* qv_data, float* tke_data,
    const float* dudt, const float* dvdt, const float* dwdt,
    const float* dthetadt, const float* dqvdt, const float* dtkedt,
    const float* dtkedt_pbl,
    int nx, int ny, int nz,
    bool has_qv, bool has_tke,
    float dt_step, float theta_fallback,
    float theta_min, float theta_max, float qv_max_val,
    float wind_h_max, float wind_v_max,
    bool resident_tendencies_unchanged,
    bool sanitize = true,
    SgsApplyDispatch::CompletedObserver completed_observer = nullptr,
    void* completed_observer_context = nullptr,
    bool collocated = false);

/**
 * @brief Dispatches the Smagorinsky eddy-coefficient pass via GPU backend
 *        if available (GPU audit Phase 3.4). Flat-terrain,
 *        stability "none" only (caller guards).
 *
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_smagorinsky_coefficients_backend(
    const float* u, const float* v, const float* w,
    const float* dz_prof, const float* dzc_prof, int prof_len,
    float* nu_t, float* K_theta, float* K_q,
    int nx, int ny, int nz,
    int filter_mode, bool use_horizontal, bool use_vertical,
    bool periodic_x, bool periodic_y,
    float dx, float dy,
    float Cs, float Pr_t, float Sc_t,
    float nu_t_max, float K_max);
bool dispatch_smagorinsky_coefficients_backend(ComputeBackendRuntimeState& compute,
    const float* u, const float* v, const float* w,
    const float* dz_prof, const float* dzc_prof, int prof_len,
    float* nu_t, float* K_theta, float* K_q,
    int nx, int ny, int nz,
    int filter_mode, bool use_horizontal, bool use_vertical,
    bool periodic_x, bool periodic_y,
    float dx, float dy,
    float Cs, float Pr_t, float Sc_t,
    float nu_t_max, float K_max);
