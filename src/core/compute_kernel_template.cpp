#include "numerics/compute_kernel_template.hpp"
#include "numerics/compute_backend.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "util/grid_metric_utils.hpp"

namespace
{

enum class TVDLimiterKind
{
    minmod,
    vanleer,
    superbee,
    mc,
    universal
};

std::string canonicalize_id(std::string value)
{
    std::string canonical;
    canonical.reserve(value.size());
    for (unsigned char c : value)
    {
        if (c == '_' || c == '-' || std::isspace(c))
        {
            continue;
        }
        canonical.push_back(static_cast<char>(std::tolower(c)));
    }
    return canonical;
}

std::string g_active_vertical_flux_template_id = "tvd_vertical_flux_v1";

bool parse_tvd_limiter_kind(const std::string& limiter_id, TVDLimiterKind& out_kind)
{
    const std::string canonical = canonicalize_id(limiter_id);
    if (canonical == "minmod")
    {
        out_kind = TVDLimiterKind::minmod;
        return true;
    }
    if (canonical == "vanleer")
    {
        out_kind = TVDLimiterKind::vanleer;
        return true;
    }
    if (canonical == "superbee")
    {
        out_kind = TVDLimiterKind::superbee;
        return true;
    }
    if (canonical == "mc" || canonical.empty())
    {
        out_kind = TVDLimiterKind::mc;
        return true;
    }
    if (canonical == "universal")
    {
        out_kind = TVDLimiterKind::universal;
        return true;
    }
    return false;
}

inline double tvd_limiter_value(double r, TVDLimiterKind kind)
{
    switch (kind)
    {
        case TVDLimiterKind::minmod:
            return std::max(0.0, std::min(1.0, r));
        case TVDLimiterKind::vanleer:
            return (std::abs(r) + r) / (1.0 + std::abs(r));
        case TVDLimiterKind::superbee:
            return std::max({0.0, std::min(1.0, 2.0 * r), std::min(2.0, r)});
        case TVDLimiterKind::universal:
            if (r >= 0.0 && r <= 1.0)
            {
                return std::min(2.0 * r, (1.0 + r) / 2.0);
            }
            if (r > 1.0)
            {
                return std::min(r, 2.0);
            }
            return 0.0;
        case TVDLimiterKind::mc:
        default:
            return std::max(0.0, std::min({(1.0 + r) / 2.0, 2.0, 2.0 * r}));
    }
}

bool dispatch_tvd_vertical_flux_v1(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt)
{
    if (!state.q || !state.w || !state.grid)
    {
        return false;
    }

    const int nr = state.q->size_r();
    const int nth = state.q->size_th();
    const int nz = state.q->size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        return false;
    }
    if (state.w->size_r() != nr || state.w->size_th() != nth || state.w->size_z() != nz)
    {
        return false;
    }

    TVDLimiterKind limiter = TVDLimiterKind::mc;
    if (!parse_tvd_limiter_kind(cfg.limiter_id, limiter))
    {
        return false;
    }

    tendencies.dqdt_adv.resize(nr, nth, nz, 0.0f);
    float* dqdt_data = tendencies.dqdt_adv.data();
    const float* q_data = state.q->data();
    const float* w_data = state.w->data();

    const bool terrain_metrics = grid_metric::has_terrain_metrics(*state.grid);
    const std::vector<double>& dz_levels = state.grid->dz;
    const bool use_level_dz = !terrain_metrics && static_cast<int>(dz_levels.size()) >= nz;
    const double dt_safe = std::max(std::abs(cfg.positivity_dt), 1.0e-12);

    double max_cfl = 0.0;

    #pragma omp parallel reduction(max:max_cfl)
    {
        std::vector<double> q_col(static_cast<std::size_t>(nz), 0.0);
        std::vector<double> w_col(static_cast<std::size_t>(nz), 0.0);
        std::vector<double> dz_col(static_cast<std::size_t>(nz), 1.0);
        std::vector<double> q_left(static_cast<std::size_t>(nz), 0.0);
        std::vector<double> q_right(static_cast<std::size_t>(nz), 0.0);
        std::vector<double> dqdt_col(static_cast<std::size_t>(nz), 0.0);

        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                const std::size_t base =
                    (static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) +
                     static_cast<std::size_t>(j)) *
                    static_cast<std::size_t>(nz);

                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t kk = static_cast<std::size_t>(k);
                    q_col[kk] = static_cast<double>(q_data[base + kk]);
                    w_col[kk] = static_cast<double>(w_data[base + kk]);

                    double local_dz = 1.0;
                    if (use_level_dz)
                    {
                        local_dz = dz_levels[kk];
                    }
                    else
                    {
                        local_dz = grid_metric::local_dz(*state.grid, i, j, k, nz);
                    }
                    local_dz = std::max(std::abs(local_dz), 1.0e-6);
                    dz_col[kk] = local_dz;

                    const double cfl = std::abs(w_col[kk]) / local_dz;
                    if (cfl > max_cfl)
                    {
                        max_cfl = cfl;
                    }
                }

                std::fill(dqdt_col.begin(), dqdt_col.end(), 0.0);
                if (nz == 1)
                {
                    dqdt_data[base] = 0.0f;
                    continue;
                }

                q_left[0] = q_col[0];
                q_right[0] = q_col[0];
                q_left[static_cast<std::size_t>(nz - 1)] = q_col[static_cast<std::size_t>(nz - 1)];
                q_right[static_cast<std::size_t>(nz - 1)] = q_col[static_cast<std::size_t>(nz - 1)];

                for (int k = 1; k < nz - 1; ++k)
                {
                    const std::size_t kk = static_cast<std::size_t>(k);
                    const std::size_t km = static_cast<std::size_t>(k - 1);
                    const std::size_t kp = static_cast<std::size_t>(k + 1);
                    const double denominator = q_col[kp] - q_col[kk] + numerics_constants::epsilon;
                    const double r = (q_col[kk] - q_col[km]) / denominator;
                    const double phi = tvd_limiter_value(r, limiter);
                    const double delta_q = phi * (q_col[kp] - q_col[kk]);
                    q_left[kk] = q_col[kk] - 0.5 * delta_q;
                    q_right[kk] = q_col[kk] + 0.5 * delta_q;
                }

                for (int k = 0; k < nz - 1; ++k)
                {
                    const std::size_t kk = static_cast<std::size_t>(k);
                    const std::size_t kp = static_cast<std::size_t>(k + 1);
                    const double vel_right = w_col[kk];
                    const double flux_right = (vel_right >= 0.0)
                        ? vel_right * q_right[kk]
                        : vel_right * q_left[kp];

                    double flux_left = 0.0;
                    if (k > 0)
                    {
                        const std::size_t km = static_cast<std::size_t>(k - 1);
                        const double vel_left = w_col[km];
                        flux_left = (vel_left >= 0.0)
                            ? vel_left * q_right[km]
                            : vel_left * q_left[kk];
                    }

                    const double dflux_dz = (flux_right - flux_left) / dz_col[kk];
                    dqdt_col[kk] -= dflux_dz;
                }

                if (cfg.positivity)
                {
                    for (int k = 0; k < nz; ++k)
                    {
                        const std::size_t kk = static_cast<std::size_t>(k);
                        double tendency = dqdt_col[kk];
                        if (!std::isfinite(tendency))
                        {
                            tendency = 0.0;
                        }
                        const double floor_tendency = (0.0 - q_col[kk]) / dt_safe;
                        if (tendency < floor_tendency)
                        {
                            tendency = floor_tendency;
                        }
                        dqdt_col[kk] = tendency;
                    }
                }

                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t kk = static_cast<std::size_t>(k);
                    double tendency = dqdt_col[kk];
                    if (!std::isfinite(tendency))
                    {
                        tendency = 0.0;
                    }
                    dqdt_data[base + kk] = static_cast<float>(tendency);
                }
            }
        }
    }

    if (diag_opt)
    {
        diag_opt->max_cfl_z = max_cfl;
        diag_opt->suggested_dt = (max_cfl > 1.0e-12)
            ? (cfg.cfl_target / max_cfl)
            : std::numeric_limits<double>::infinity();
    }

    return true;
}

} // namespace

void set_active_vertical_flux_template_id(const std::string& template_id)
{
    if (!has_vertical_flux_template(template_id))
    {
        return;
    }
    g_active_vertical_flux_template_id = template_id;
}

const std::string& active_vertical_flux_template_id()
{
    return g_active_vertical_flux_template_id;
}

bool has_vertical_flux_template(const std::string& template_id)
{
    const std::string id = canonicalize_id(template_id);
    return id == "tvdverticalfluxv1" || id == "tvddefaultv1";
}

std::vector<VerticalFluxTemplateDescriptor> list_vertical_flux_templates()
{
    const ComputeBackend* backend = active_compute_backend();
    const bool gpu_ready = backend != nullptr && backend->supports_vertical_flux_dispatch();
    return {
        {"tvd_vertical_flux_v1",
         "Default TVD vertical advection flux-divergence template (MUSCL + limiter + positivity).",
         gpu_ready},
    };
}

bool dispatch_vertical_flux_template_backend(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt)
{
    const std::string id = canonicalize_id(g_active_vertical_flux_template_id);
    if (id != "tvdverticalfluxv1" && id != "tvddefaultv1")
    {
        return false;
    }

    // Try GPU dispatch if backend supports it
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_vertical_flux_dispatch())
    {
        if (!state.q || !state.w || !state.grid)
        {
            return false;
        }
        const int nr = state.q->size_r();
        const int nth = state.q->size_th();
        const int nz = state.q->size_z();
        if (nr <= 0 || nth <= 0 || nz <= 0)
        {
            return false;
        }

        // Marshal limiter string to integer id
        TVDLimiterKind limiter_kind = TVDLimiterKind::mc;
        parse_tvd_limiter_kind(cfg.limiter_id, limiter_kind);
        const int limiter_int = static_cast<int>(limiter_kind);

        tendencies.dqdt_adv.resize(nr, nth, nz, 0.0f);

        const std::vector<double>& dz_levels = state.grid->dz;
        const double dt_safe = std::max(std::abs(cfg.positivity_dt), 1.0e-12);

        VerticalFluxDispatchResult result{};
        const bool ok = backend->dispatch_vertical_flux(
            state.q->data(), state.w->data(), tendencies.dqdt_adv.data(),
            nr, nth, nz,
            dz_levels.data(), static_cast<int>(dz_levels.size()),
            limiter_int, cfg.positivity, dt_safe,
            cfg.cfl_target, result);

        if (ok)
        {
            if (diag_opt)
            {
                diag_opt->max_cfl_z = result.max_cfl_z;
                diag_opt->suggested_dt = result.suggested_dt;
            }
            return true;
        }
        // GPU dispatch failed — fall through to CPU reference
    }

    // CPU reference path
    return dispatch_tvd_vertical_flux_v1(cfg, state, tendencies, diag_opt);
}

bool dispatch_radial_advection_backend(
    const float* src, const float* u_data, float* dst,
    int nr, int nth, int nz,
    float dr, float dt)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_radial_advection_dispatch())
    {
        return backend->dispatch_radial_advection(src, u_data, dst, nr, nth, nz, dr, dt);
    }
    return false;
}

bool dispatch_azimuthal_advection_backend(
    const float* src, const float* v_data, float* dst,
    int nr, int nth, int nz,
    float dr, float dtheta, float dt)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_azimuthal_advection_dispatch())
    {
        return backend->dispatch_azimuthal_advection(src, v_data, dst, nr, nth, nz, dr, dtheta, dt);
    }
    return false;
}

bool dispatch_diffusion_backend(
    const float* src, float* dst,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt, float kappa)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_diffusion_dispatch())
    {
        return backend->dispatch_diffusion(src, dst, nr, nth, nz, dr, dtheta, dz, dt, kappa);
    }
    return false;
}

bool dispatch_supercell_tendencies_backend(
    const float* u_r_data, const float* u_theta_data, const float* u_z_data,
    const float* rho_data, const float* p_data, const float* theta_data,
    float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float g, float gamma_val, float theta0)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_supercell_tendencies_dispatch())
    {
        return backend->dispatch_supercell_tendencies(
            u_r_data, u_theta_data, u_z_data,
            rho_data, p_data, theta_data,
            du_r_dt_data, du_theta_dt_data, du_z_dt_data,
            drho_dt_data, dp_dt_data,
            nr, nth, nz,
            dr, dtheta, dz,
            g, gamma_val, theta0);
    }
    return false;
}

bool dispatch_tornado_tendencies_backend(
    const float* u_r_data, const float* u_theta_data, const float* u_z_data,
    const float* rho_data, const float* p_data, const float* theta_data,
    float* du_r_dt_data, float* du_theta_dt_data, float* du_z_dt_data,
    float* drho_dt_data, float* dp_dt_data,
    int nr, int nth, int nz,
    float dr, float dz,
    float g, float theta0, float eps, float friction_coeff)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_tornado_tendencies_dispatch())
    {
        return backend->dispatch_tornado_tendencies(
            u_r_data, u_theta_data, u_z_data,
            rho_data, p_data, theta_data,
            du_r_dt_data, du_theta_dt_data, du_z_dt_data,
            drho_dt_data, dp_dt_data,
            nr, nth, nz,
            dr, dz,
            g, theta0, eps, friction_coeff);
    }
    return false;
}

bool dispatch_kessler_pointwise_backend(
    const float* temperature_data, const float* qv_data,
    const float* qc_data, const float* qr_data,
    const float* qg_data, const float* qh_data,
    float* dtheta_dt_data, float* dqv_dt_data,
    float* dqc_dt_data, float* dqr_dt_data,
    float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    float qc0, float c_auto, float c_accr, float c_evap,
    float c_freeze, float c_rime, float c_melt, float c_subl,
    float Lv_cp, float Lf_cp, float Ls_cp, float T0)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_kessler_pointwise_dispatch())
    {
        return backend->dispatch_kessler_pointwise(
            temperature_data, qv_data, qc_data, qr_data, qg_data, qh_data,
            dtheta_dt_data, dqv_dt_data, dqc_dt_data, dqr_dt_data,
            dqg_dt_data, dqh_dt_data,
            nr, nth, nz,
            qc0, c_auto, c_accr, c_evap,
            c_freeze, c_rime, c_melt, c_subl,
            Lv_cp, Lf_cp, Ls_cp, T0);
    }
    return false;
}

bool dispatch_kessler_sedimentation_backend(
    const float* qr_data, const float* qg_data, const float* qh_data,
    float* dqr_dt_data, float* dqg_dt_data, float* dqh_dt_data,
    int nr, int nth, int nz,
    float dz_val,
    float a_rain, float b_rain, float Vt_max_rain,
    float a_grau, float b_grau, float Vt_max_grau,
    float a_hail, float b_hail, float Vt_max_hail)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_kessler_sedimentation_dispatch())
    {
        return backend->dispatch_kessler_sedimentation(
            qr_data, qg_data, qh_data,
            dqr_dt_data, dqg_dt_data, dqh_dt_data,
            nr, nth, nz,
            dz_val,
            a_rain, b_rain, Vt_max_rain,
            a_grau, b_grau, Vt_max_grau,
            a_hail, b_hail, Vt_max_hail);
    }
    return false;
}

bool supports_batched_advection_dispatch()
{
    const ComputeBackend* backend = active_compute_backend();
    return backend != nullptr && backend->supports_batched_advection_dispatch();
}

bool dispatch_advection_batch_pre_vertical_backend(
    const float* scalar_in, float* result_out,
    const float* u_data, const float* v_theta_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dt_half)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_batched_advection_dispatch())
    {
        return backend->dispatch_advection_batch_pre_vertical(
            scalar_in, result_out,
            u_data, v_theta_data,
            nr, nth, nz,
            dr, dtheta, dt_half);
    }
    return false;
}

bool dispatch_advection_batch_post_vertical_backend(
    const float* scalar_in, float* result_out,
    const float* u_data, const float* v_theta_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_half, float dt_full, float kappa)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_batched_advection_dispatch())
    {
        return backend->dispatch_advection_batch_post_vertical(
            scalar_in, result_out,
            u_data, v_theta_data,
            nr, nth, nz,
            dr, dtheta, dz,
            dt_half, dt_full, kappa);
    }
    return false;
}

