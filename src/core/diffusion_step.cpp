/**
 * @file diffusion_step.cpp
 * @brief Runtime diffusion tendency driver.
 *
 * Computes and applies diffusion tendencies via the active diffusion scheme.
 * Manages its own tendency buffers and config checks internally.
 *
 * Extracted from src/core/dynamics.cpp.
 */

#include "core/diffusion_step.hpp"
#include "core/simulation.hpp"
#include "numerics/diffusion_base.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

DiffusionTendencies diffusion_tend_buf;
DiffusionDiagnostics diffusion_diag_buf;

inline bool field_matches_domain(const Field3D& f)
{
    return f.size_r() == NR && f.size_th() == NTH && f.size_z() == NZ;
}

void ensure_diffusion_tendency_buffers()
{
    auto ensure = [](Field3D& field)
    {
        if (field.size_r() != NR || field.size_th() != NTH || field.size_z() != NZ)
        {
            field.resize(NR, NTH, NZ, 0.0f);
        }
        else
        {
            field.fill(0.0f);
        }
    };

    ensure(diffusion_tend_buf.dudt_diff);
    ensure(diffusion_tend_buf.dvdt_diff);
    ensure(diffusion_tend_buf.dwdt_diff);
    ensure(diffusion_tend_buf.dthetadt_diff);
    ensure(diffusion_tend_buf.dqvdt_diff);
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool diffusion_applies_to_momentum()
{
    const std::string apply_to = lower_copy(global_diffusion_config.apply_to);
    return apply_to == "momentum" || apply_to == "all";
}

bool diffusion_applies_to_scalars()
{
    const std::string apply_to = lower_copy(global_diffusion_config.apply_to);
    return apply_to == "scalars" || apply_to == "all";
}

bool diffusion_runtime_enabled()
{
    if (!diffusion_scheme)
    {
        return false;
    }
    const double K_h = std::max(0.0, global_diffusion_config.K_h);
    const double K_v = std::max(0.0, global_diffusion_config.K_v);
    if (K_h <= 0.0 && K_v <= 0.0 && !global_diffusion_config.use_variable_K)
    {
        return false;
    }

    return diffusion_applies_to_momentum() || diffusion_applies_to_scalars();
}

void log_runtime_diffusion_path_once()
{
    static bool logged = false;
    if (logged)
    {
        return;
    }
    logged = true;

    if (!diffusion_scheme)
    {
        tmv::log_info("[DIFFUSION] Runtime path: disabled (no diffusion scheme).");
        return;
    }

    tmv::log_info("[DIFFUSION] Runtime path: src/numerics/diffusion/", diffusion_scheme->name(),
                  " (apply_to=", global_diffusion_config.apply_to,
                  ", K_h=", global_diffusion_config.K_h,
                  ", K_v=", global_diffusion_config.K_v, ")");
}

}  // namespace

void apply_runtime_diffusion(double dt_dynamics)
{
    if (!diffusion_runtime_enabled())
    {
        return;
    }

    ensure_diffusion_tendency_buffers();
    log_runtime_diffusion_path_once();

    const bool do_momentum = diffusion_applies_to_momentum();
    const bool do_scalars = diffusion_applies_to_scalars();

    DiffusionConfig runtime_cfg = global_diffusion_config;
    runtime_cfg.dt_diffusion = std::max(1.0e-6, dt_dynamics);

    DiffusionStateView state{};
    state.grid = &global_grid_metrics;
    state.rho = &rho;
    if (do_momentum)
    {
        state.u = &u;
        state.v = &v_theta;
        state.w = &w;
    }
    if (do_scalars)
    {
        state.theta = &theta;
        state.qv = &qv;
    }

    try
    {
        diffusion_scheme->compute_diffusion_tendencies(runtime_cfg, state, diffusion_tend_buf, &diffusion_diag_buf);
    }
    catch (const std::exception& e)
    {
        tmv::log_error("[DIFFUSION] tendency computation failed: ", e.what());
        return;
    }

    const bool shape_ok =
        field_matches_domain(diffusion_tend_buf.dudt_diff) &&
        field_matches_domain(diffusion_tend_buf.dvdt_diff) &&
        field_matches_domain(diffusion_tend_buf.dwdt_diff) &&
        field_matches_domain(diffusion_tend_buf.dthetadt_diff) &&
        field_matches_domain(diffusion_tend_buf.dqvdt_diff);
    if (!shape_ok)
    {
        tmv::log_error("[DIFFUSION] tendency buffers have invalid shape; skipping diffusion update.");
        return;
    }

    if (do_momentum)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    float dudt = diffusion_tend_buf.dudt_diff[i][j][k];
                    float dvdt = diffusion_tend_buf.dvdt_diff[i][j][k];
                    float dwdt = diffusion_tend_buf.dwdt_diff[i][j][k];
                    if (!std::isfinite(dudt)) dudt = 0.0f;
                    if (!std::isfinite(dvdt)) dvdt = 0.0f;
                    if (!std::isfinite(dwdt)) dwdt = 0.0f;

                    float u_new = u[i][j][k] + dudt * static_cast<float>(dt_dynamics);
                    float v_new = v_theta[i][j][k] + dvdt * static_cast<float>(dt_dynamics);
                    float w_new = w[i][j][k] + dwdt * static_cast<float>(dt_dynamics);

                    if (!std::isfinite(u_new)) u_new = 0.0f;
                    if (!std::isfinite(v_new)) v_new = 0.0f;
                    if (!std::isfinite(w_new)) w_new = 0.0f;

                    u[i][j][k] = clamp_wind_horizontal_ms(u_new);
                    v_theta[i][j][k] = clamp_wind_horizontal_ms(v_new);
                    w[i][j][k] = clamp_wind_vertical_ms(w_new);
                }
            }
        }
    }

    if (do_scalars)
    {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    float dthetadt = diffusion_tend_buf.dthetadt_diff[i][j][k];
                    float dqvdt = diffusion_tend_buf.dqvdt_diff[i][j][k];
                    if (!std::isfinite(dthetadt)) dthetadt = 0.0f;
                    if (!std::isfinite(dqvdt)) dqvdt = 0.0f;

                    float theta_new = theta[i][j][k] + dthetadt * static_cast<float>(dt_dynamics);
                    float qv_new = qv[i][j][k] + dqvdt * static_cast<float>(dt_dynamics);

                    if (!std::isfinite(theta_new)) theta_new = static_cast<float>(theta0);
                    if (!std::isfinite(qv_new)) qv_new = 0.0f;

                    theta[i][j][k] = clamp_theta_k(theta_new);
                    qv[i][j][k] = clamp_qv_kgkg(qv_new);
                }
            }
        }
    }

    if(lower_copy(global_diffusion_config.scheme_id) == "explicit" && diffusion_diag_buf.max_diffusion_number > 1.0)
    {
        tmv::log_warn("[DIFFUSION] max diffusion number exceeds 1 (",
                      diffusion_diag_buf.max_diffusion_number,
                      "); reduce dt or diffusivity for stronger explicit stability margin.");
    }
}
