#include "compute/bridge/microphysics.hpp"

#include "compute/compute_backend.hpp"

namespace
{
MicrophysicsComputeKernels* active_microphysics_kernels(ComputeBackendRuntimeState& compute)
{
    ComputeBackend* backend = mutable_compute_backend(compute);
    return backend != nullptr ? backend->microphysics_kernels() : nullptr;
}
} // namespace

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
    unsigned int* stats_out, int stats_len)
{
    return dispatch_microphysics_apply_backend(global_compute_backend_runtime_state, theta_data, qv_data, qc_data, qr_data, qi_data, qs_data, qg_data, qh_data, dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh, dtheta_rad, dtheta_pbl, dqv_pbl, nx, ny, nz, i_begin, i_end, j_begin, j_end, dt_step, theta_fallback, theta_min, theta_max, max_theta_step, qv_max_val, hydro_max_val, stats_out, stats_len);
}

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
    unsigned int* stats_out, int stats_len)
{
    MicrophysicsComputeKernels* kernels = active_microphysics_kernels(compute);
    if (kernels != nullptr &&
        kernels->supports(MicrophysicsKernelOp::MicrophysicsApply))
    {
        const MicrophysicsApplyDispatch dispatch{
            theta_data, qv_data, qc_data, qr_data,
            qi_data, qs_data, qg_data, qh_data,
            dtheta, dqv, dqc, dqr, dqi, dqs, dqg, dqh,
            dtheta_rad, dtheta_pbl, dqv_pbl,
            nx, ny, nz,
            i_begin, i_end, j_begin, j_end,
            dt_step, theta_fallback,
            theta_min, theta_max, max_theta_step,
            qv_max_val, hydro_max_val,
            stats_out, stats_len
        };
        return kernels->dispatch_microphysics_apply(dispatch);
    }
    return false;
}

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
    float Lv_cp, float Lf_cp, float Ls_cp)
{
    return dispatch_lin_pointwise_backend(global_compute_backend_runtime_state, theta_data, p_data, qv, qc, qr, qi, qs, qg, qh, dtheta_dt, dqv_dt, dqc_dt, dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt, nx, ny, nz, dt_step, inv_dt, qc0, c_auto, c_accr, c_evap, c_ihom, c_rime, c_agg, c_melt_s, c_melt_g, c_melt_h, c_subl, Lv_cp, Lf_cp, Ls_cp);
}

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
    float Lv_cp, float Lf_cp, float Ls_cp)
{
    MicrophysicsComputeKernels* kernels = active_microphysics_kernels(compute);
    if (kernels != nullptr && kernels->supports(MicrophysicsKernelOp::LinPointwise))
    {
        const LinPointwiseDispatch dispatch{
            theta_data, p_data,
            qv, qc, qr, qi, qs, qg, qh,
            dtheta_dt, dqv_dt, dqc_dt, dqr_dt,
            dqi_dt, dqs_dt, dqg_dt, dqh_dt,
            nx, ny, nz,
            dt_step, inv_dt,
            qc0, c_auto, c_accr, c_evap,
            c_ihom, c_rime, c_agg, c_melt_s, c_melt_g, c_melt_h, c_subl,
            Lv_cp, Lf_cp, Ls_cp
        };
        return kernels->dispatch_lin_pointwise(dispatch);
    }
    return false;
}

bool dispatch_lin_fused_backend(
    const LinPointwiseDispatch& pointwise,
    const LinSedimentationDispatch& sedimentation)
{
    return dispatch_lin_fused_backend(global_compute_backend_runtime_state, pointwise, sedimentation);
}

bool dispatch_lin_fused_backend(ComputeBackendRuntimeState& compute,
    const LinPointwiseDispatch& pointwise,
    const LinSedimentationDispatch& sedimentation)
{
    MicrophysicsComputeKernels* kernels = active_microphysics_kernels(compute);
    if (kernels != nullptr && kernels->supports(MicrophysicsKernelOp::LinFused))
    {
        return kernels->dispatch_lin_fused({pointwise, sedimentation});
    }
    return false;
}

bool dispatch_lin_sedimentation_backend(
    const float* qr, const float* qs, const float* qg, const float* qh,
    const float* rho_d,
    float* dqr_dt, float* dqs_dt, float* dqg_dt, float* dqh_dt,
    int nx, int ny, int nz,
    float dz,
    float vt_cap)
{
    return dispatch_lin_sedimentation_backend(global_compute_backend_runtime_state, qr, qs, qg, qh, rho_d, dqr_dt, dqs_dt, dqg_dt, dqh_dt, nx, ny, nz, dz, vt_cap);
}

bool dispatch_lin_sedimentation_backend(ComputeBackendRuntimeState& compute,
    const float* qr, const float* qs, const float* qg, const float* qh,
    const float* rho_d,
    float* dqr_dt, float* dqs_dt, float* dqg_dt, float* dqh_dt,
    int nx, int ny, int nz,
    float dz,
    float vt_cap)
{
    MicrophysicsComputeKernels* kernels = active_microphysics_kernels(compute);
    if (kernels != nullptr && kernels->supports(MicrophysicsKernelOp::LinSedimentation))
    {
        const LinSedimentationDispatch dispatch{
            qr, qs, qg, qh,
            rho_d,
            dqr_dt, dqs_dt, dqg_dt, dqh_dt,
            nx, ny, nz,
            dz,
            vt_cap
        };
        return kernels->dispatch_lin_sedimentation(dispatch);
    }
    return false;
}

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
    float Lv_cp, float T0, float dt_step)
{
    return dispatch_kessler_pointwise_backend(global_compute_backend_runtime_state, temperature_data, p_data, qv_data, qc_data, qr_data, rho_d_data, qh_data, dtheta_dt_data, dqv_dt_data, dqc_dt_data, dqr_dt_data, dqg_dt_data, dqh_dt_data, nr, nth, nz, i_begin, i_end, j_begin, j_end, k_begin, k_end, qc0, c_auto, c_accr, evaporation_efficiency, Lv_cp, T0, dt_step);
}

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
    float Lv_cp, float T0, float dt_step)
{
    MicrophysicsComputeKernels* kernels = active_microphysics_kernels(compute);
    if (kernels != nullptr && kernels->supports(MicrophysicsKernelOp::KesslerPointwise))
    {
        const KesslerPointwiseDispatch dispatch{
            temperature_data, p_data,
            qv_data, qc_data, qr_data, rho_d_data, qh_data,
            dtheta_dt_data, dqv_dt_data, dqc_dt_data, dqr_dt_data,
            dqg_dt_data, dqh_dt_data,
            nr, nth, nz,
            i_begin, i_end, j_begin, j_end, k_begin, k_end,
            qc0, c_auto, c_accr, evaporation_efficiency,
            Lv_cp, T0, dt_step
        };
        return kernels->dispatch_kessler_pointwise(dispatch);
    }
    return false;
}

bool dispatch_kessler_sedimentation_backend(
    const float* qr_data, const float* qg_data, const float* qh_data,
    const float* rho_d_data,
    float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    int k_begin, int k_end,
    float dz_val,
    float Vt_max_rain, float vt_cap, float dt_step, const float* layer_thickness_m)
{
    return dispatch_kessler_sedimentation_backend(global_compute_backend_runtime_state, qr_data, qg_data, qh_data, rho_d_data, dqr_dt_data, dqg_dt_data, dqh_dt_data, nr, nth, nz, i_begin, i_end, j_begin, j_end, k_begin, k_end, dz_val, Vt_max_rain, vt_cap, dt_step, layer_thickness_m);
}

bool dispatch_kessler_sedimentation_backend(ComputeBackendRuntimeState& compute,
    const float* qr_data, const float* qg_data, const float* qh_data,
    const float* rho_d_data,
    float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    int i_begin, int i_end, int j_begin, int j_end,
    int k_begin, int k_end,
    float dz_val,
    float Vt_max_rain, float vt_cap, float dt_step, const float* layer_thickness_m)
{
    MicrophysicsComputeKernels* kernels = active_microphysics_kernels(compute);
    if (kernels != nullptr && kernels->supports(MicrophysicsKernelOp::KesslerSedimentation))
    {
        const KesslerSedimentationDispatch dispatch{
            qr_data, qg_data, qh_data,
            rho_d_data,
            dqr_dt_data, dqg_dt_data, dqh_dt_data,
            nr, nth, nz,
            i_begin, i_end, j_begin, j_end, k_begin, k_end,
            dz_val,
            Vt_max_rain, vt_cap, dt_step, layer_thickness_m
        };
        return kernels->dispatch_kessler_sedimentation(dispatch);
    }
    return false;
}

bool dispatch_lin_pointwise_backend(ComputeBackendRuntimeState& compute,const LinPointwiseDispatch& dispatch)
{
    auto* kernels=active_microphysics_kernels(compute);
    return kernels && kernels->supports(MicrophysicsKernelOp::LinPointwise) && kernels->dispatch_lin_pointwise(dispatch);
}
bool dispatch_lin_sedimentation_backend(ComputeBackendRuntimeState& compute,const LinSedimentationDispatch& dispatch)
{
    auto* kernels=active_microphysics_kernels(compute);
    return kernels && kernels->supports(MicrophysicsKernelOp::LinSedimentation) && kernels->dispatch_lin_sedimentation(dispatch);
}
