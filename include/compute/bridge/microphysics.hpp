#pragma once

struct ComputeBackendRuntimeState;

#include "compute/compute_kernel_interfaces.hpp"

/**
 * @brief Applies microphysics (+ radiation/PBL) tendencies to the eight
 *        thermodynamic state fields on device with the driver's exact
 *        guard semantics (GPU audit Phase 4.2). stats_out (length
 *        >= 10 + nz + nx) receives the guard counters, the per-k/per-i
 *        theta-clamp histograms and the first-clamp detail record.
 *
 * @return True if GPU dispatch succeeded. False means caller should run
 *         the CPU apply loop.
 */
bool dispatch_microphysics_apply_backend(
    float* theta_data, float* qv_data, float* qc_data, float* qr_data,
    float* qi_data, float* qs_data, float* qg_data, float* qh_data,
    const float* dtheta, const float* dqv, const float* dqc,
    const float* dqr, const float* dqi, const float* dqs,
    const float* dqg, const float* dqh,
    const float* dtheta_rad, const float* dtheta_pbl,
    const float* dqv_pbl,
    int nx, int ny, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    float dt_step, float theta_fallback,
    float theta_min, float theta_max, float max_theta_step,
    float qv_max_val, float hydro_max_val,
    unsigned int* stats_out, int stats_len);
bool dispatch_microphysics_apply_backend(ComputeBackendRuntimeState& compute,
    float* theta_data, float* qv_data, float* qc_data, float* qr_data,
    float* qi_data, float* qs_data, float* qg_data, float* qh_data,
    const float* dtheta, const float* dqv, const float* dqc,
    const float* dqr, const float* dqi, const float* dqs,
    const float* dqg, const float* dqh,
    const float* dtheta_rad, const float* dtheta_pbl,
    const float* dqv_pbl,
    int nx, int ny, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    float dt_step, float theta_fallback,
    float theta_min, float theta_max, float max_theta_step,
    float qv_max_val, float hydro_max_val,
    unsigned int* stats_out, int stats_len);

/**
 * @brief Dispatches the fused point-wise Lin microphysics via GPU backend
 *        if available (GPU audit Phase 3.3).
 *
 * @return True if GPU dispatch succeeded. False means caller should use
 *         the CPU path.
 */
bool dispatch_lin_pointwise_backend(
    const float* theta_data, const float* p_data,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh,
    float* dtheta_dt, float* dqv_dt, float* dqc_dt, float* dqr_dt,
    float* dqi_dt, float* dqs_dt, float* dqg_dt, float* dqh_dt,
    int nx, int ny, int nz,
    float dt_step, float inv_dt,
    float qc0, float c_auto, float c_accr, float c_evap,
    float c_ihom, float c_rime, float c_agg,
    float c_melt_s, float c_melt_g, float c_melt_h, float c_subl,
    float Lv_cp, float Lf_cp, float Ls_cp);
bool dispatch_lin_pointwise_backend(ComputeBackendRuntimeState& compute,
    const float* theta_data, const float* p_data,
    const float* qv, const float* qc, const float* qr,
    const float* qi, const float* qs, const float* qg, const float* qh,
    float* dtheta_dt, float* dqv_dt, float* dqc_dt, float* dqr_dt,
    float* dqi_dt, float* dqs_dt, float* dqg_dt, float* dqh_dt,
    int nx, int ny, int nz,
    float dt_step, float inv_dt,
    float qc0, float c_auto, float c_accr, float c_evap,
    float c_ihom, float c_rime, float c_agg,
    float c_melt_s, float c_melt_g, float c_melt_h, float c_subl,
    float Lv_cp, float Lf_cp, float Ls_cp);

/** Pointwise Lin physics + sedimentation in one device submission. */
bool dispatch_lin_fused_backend(
    const LinPointwiseDispatch& pointwise,
    const LinSedimentationDispatch& sedimentation);
bool dispatch_lin_fused_backend(ComputeBackendRuntimeState& compute,
    const LinPointwiseDispatch& pointwise,
    const LinSedimentationDispatch& sedimentation);

/**
 * @brief Dispatches Lin column sedimentation via GPU backend if available
 *        (GPU audit Phase 3.3). Accumulates onto the point-wise
 *        tendencies in place.
 *
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_lin_sedimentation_backend(
    const float* qr, const float* qs, const float* qg, const float* qh,
    const float* rho_d,
    float* dqr_dt, float* dqs_dt, float* dqg_dt, float* dqh_dt,
    int nx, int ny, int nz,
    float dz,
    float vt_cap);
bool dispatch_lin_sedimentation_backend(ComputeBackendRuntimeState& compute,
    const float* qr, const float* qs, const float* qg, const float* qh,
    const float* rho_d,
    float* dqr_dt, float* dqs_dt, float* dqg_dt, float* dqh_dt,
    int nx, int ny, int nz,
    float dz,
    float vt_cap);

/**
 * @brief Dispatches fused Kessler point-wise microphysics via GPU backend.
 *
 * Computes three-species warm-rain tendencies. Frozen-species ABI slots are
 * retained but inert.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_kessler_pointwise_backend(
    const float* temperature_data, const float* p_data,
    const float* qv_data,
    const float* qc_data, const float* qr_data,
    const float* rho_d_data, const float* qh_data,
    float* dtheta_dt_data, float* dqv_dt_data,
    float* dqc_dt_data, float* dqr_dt_data,
    float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    int k_begin, int k_end,
    float qc0, float c_auto, float c_accr, float evaporation_efficiency,
    float Lv_cp, float T0, float dt_step);
bool dispatch_kessler_pointwise_backend(ComputeBackendRuntimeState& compute,
    const float* temperature_data, const float* p_data,
    const float* qv_data,
    const float* qc_data, const float* qr_data,
    const float* rho_d_data, const float* qh_data,
    float* dtheta_dt_data, float* dqv_dt_data,
    float* dqc_dt_data, float* dqr_dt_data,
    float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    int k_begin, int k_end,
    float qc0, float c_auto, float c_accr, float evaporation_efficiency,
    float Lv_cp, float T0, float dt_step);

/**
 * @brief Dispatches Kessler sedimentation via GPU backend (column-wise).
 *
 * Reads existing tendency values and adds sedimentation contributions.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_kessler_sedimentation_backend(
    const float* qr_data, const float* qg_data, const float* qh_data,
    const float* rho_d_data,
    float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    int k_begin, int k_end,
    float dz_val,
    float Vt_max_rain, float vt_cap, float dt_step, const float* layer_thickness_m = nullptr);
bool dispatch_kessler_sedimentation_backend(ComputeBackendRuntimeState& compute,
    const float* qr_data, const float* qg_data, const float* qh_data,
    const float* rho_d_data,
    float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    int k_begin, int k_end,
    float dz_val,
    float Vt_max_rain, float vt_cap, float dt_step, const float* layer_thickness_m = nullptr);

struct LinPointwiseDispatch;
struct LinSedimentationDispatch;
bool dispatch_lin_pointwise_backend(ComputeBackendRuntimeState& compute, const LinPointwiseDispatch& dispatch);
bool dispatch_lin_sedimentation_backend(ComputeBackendRuntimeState& compute, const LinSedimentationDispatch& dispatch);
