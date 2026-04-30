#pragma once

#include <string>
#include <vector>

#include "numerics/advection/advection_base.hpp"

/**
 * @file compute_kernel_template.hpp
 * @brief Registry and dispatch surface for backend-managed compute kernel templates.
 *
 * The intent is to isolate backend plumbing from meteorology kernel logic.
 * Template authors define kernel-level numerical behavior, while this runtime
 * layer owns backend selection, validation/fallback, and dispatch routing.
 */

struct VerticalFluxTemplateDescriptor
{
    std::string id;
    std::string summary;
    bool backend_dispatch_ready = false;
};

/**
 * @brief Sets the active vertical-flux template id.
 *
 * Unknown ids are ignored to preserve a safe default template.
 */
void set_active_vertical_flux_template_id(const std::string& template_id);

/**
 * @brief Returns active vertical-flux template id.
 */
const std::string& active_vertical_flux_template_id();

/**
 * @brief Returns true if a vertical-flux template id is available.
 */
bool has_vertical_flux_template(const std::string& template_id);

/**
 * @brief Lists available vertical-flux templates.
 */
std::vector<VerticalFluxTemplateDescriptor> list_vertical_flux_templates();

/**
 * @brief Dispatches active vertical-flux template using backend-managed path.
 *
 * Returns false when the template cannot execute the backend dispatch path.
 */
bool dispatch_vertical_flux_template_backend(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt = nullptr);

/**
 * @brief Dispatches radial advection via GPU backend if available.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_radial_advection_backend(
    const float* src, const float* u_data, float* dst,
    int nr, int nth, int nz,
    float dr, float dt);

/**
 * @brief Dispatches azimuthal advection via GPU backend if available.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_azimuthal_advection_backend(
    const float* src, const float* v_data, float* dst,
    int nr, int nth, int nz,
    float dr, float dtheta, float dt);

/**
 * @brief Dispatches cylindrical diffusion via GPU backend if available.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_diffusion_backend(
    const float* src, float* dst,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt, float kappa);

/**
 * @brief Dispatches supercell momentum tendencies via GPU backend if available.
 *
 * Computes 5 tendency fields from 6 input state fields. Only dispatches when
 * terrain metrics are NOT active (uniform grid spacing assumption).
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_supercell_tendencies_backend(
    const float* u_r_data, const float* u_theta_data, const float* u_z_data,
    const float* rho_data, const float* p_data, const float* theta_data,
    const float* loading_data,
    float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float g, float gamma_val, float theta0);

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
    float* du_x_dt_data, float* du_y_dt_data, float* dw_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dx, float dy, float dz,
    float g, float gamma_val, float coriolis_f_val);

/**
 * @brief Dispatches Cartesian x-advection via GPU backend if available.
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_advection_x_backend(
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

/**
 * @brief Dispatches tornado momentum tendencies via GPU backend if available.
 *
 * Uses only radial and vertical derivatives (no azimuthal). Includes vortex
 * damping friction. Computes all (i,j,k) interior points on GPU.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_tornado_tendencies_backend(
    const float* u_r_data, const float* u_theta_data, const float* u_z_data,
    const float* rho_data, const float* p_data, const float* theta_data,
    const float* loading_data,
    float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dr, float dz,
    float g, float theta0, float eps, float friction_coeff);

/**
 * @brief Dispatches fused Kessler point-wise microphysics via GPU backend.
 *
 * Computes warm rain + ice + melting tendencies in a single GPU dispatch.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_kessler_pointwise_backend(
    const float* temperature_data, const float* p_data,
    const float* qv_data,
    const float* qc_data, const float* qr_data,
    const float* qg_data, const float* qh_data,
    float* dtheta_dt_data, float* dqv_dt_data,
    float* dqc_dt_data, float* dqr_dt_data,
    float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    float qc0, float c_auto, float c_accr, float c_evap,
    float c_freeze, float c_rime, float c_melt, float c_subl,
    float Lv_cp, float Lf_cp, float Ls_cp, float T0);

/**
 * @brief Dispatches Kessler sedimentation via GPU backend (column-wise).
 *
 * Reads existing tendency values and adds sedimentation contributions.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_kessler_sedimentation_backend(
    const float* qr_data, const float* qg_data, const float* qh_data,
    float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    float dz_val,
    float a_rain, float b_rain, float Vt_max_rain,
    float a_grau, float b_grau, float Vt_max_grau,
    float a_hail, float b_hail, float Vt_max_hail);

/**
 * @brief Dispatches Thompson point-wise microphysics via GPU backend.
 */
bool dispatch_thompson_pointwise_backend(
    const float* temperature_data, const float* p_data,
    const float* qv_data, const float* qc_data, const float* qr_data,
    const float* qi_data, const float* qs_data,
    const float* qg_data, const float* qh_data,
    float* dtheta_dt_data, float* dqv_dt_data,
    float* dqc_dt_data, float* dqr_dt_data,
    float* dqi_dt_data, float* dqs_dt_data,
    float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    float qc0, float c_auto, float c_evap,
    float c_dep, float c_subl, float c_melt,
    float Lv_cp, float Lf_cp, float Ls_cp, float T0,
    float ccn_conc, float in_conc);

/**
 * @brief Dispatches Thompson sedimentation via GPU backend (4 species).
 */
bool dispatch_thompson_sedimentation_backend(
    const float* qr_data, const float* qs_data,
    const float* qg_data, const float* qh_data,
    float* dqr_dt_data, float* dqs_dt_data,
    float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    float dz_val,
    float a_rain, float b_rain, float Vt_max_rain,
    float a_snow, float b_snow, float Vt_max_snow,
    float a_grau, float b_grau, float Vt_max_grau,
    float a_hail, float b_hail, float Vt_max_hail);

/**
 * @brief Returns true when batched advection dispatch is available.
 */
bool supports_batched_advection_dispatch();

/**
 * @brief Dispatches pre-vertical advection batch (radial+azimuthal).
 *
 * Records both steps in a single GPU command buffer submission.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_advection_batch_pre_vertical_backend(
    const float* scalar_in, float* result_out,
    const float* u_data, const float* v_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dt_half);

/**
 * @brief Dispatches post-vertical advection batch (azimuthal+radial+diffusion).
 *
 * Records all three steps in a single GPU command buffer submission.
 *
 * @return True if GPU dispatch succeeded. False means caller should use CPU path.
 */
bool dispatch_advection_batch_post_vertical_backend(
    const float* scalar_in, float* result_out,
    const float* u_data, const float* v_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_half, float dt_full, float kappa);

/**
 * @brief Dispatches fused acoustic pressure substep on GPU.
 *
 * Computes cylindrical divergence from u,v,w, integrates rho and p
 * forward by dt_small, applies boundary conditions.
 *
 * @return True if GPU dispatch succeeded.
 */
bool dispatch_acoustic_pressure_backend(
    const float* u_data, const float* v_data, const float* w_data,
    const float* rho_in, const float* p_in,
    float* rho_out, float* p_out,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small,
    float rho_floor, float p_floor);

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

