/**
 * @file explicit.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "explicit.hpp"
#include "core/field_pool.hpp"
#include "numerics/compute_kernel_template.hpp"
#include "util/grid_metric_utils.hpp"
#include "util/simd_utils.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
inline bool matches_shape(const Field3D* field, int nr, int nth, int nz)
{
    return field != nullptr &&
           field->size_r() == nr &&
           field->size_th() == nth &&
           field->size_z() == nz;
}

inline void ensure_tendency_shape(Field3D& field, int nr, int nth, int nz)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
    {
        field.resize(nr, nth, nz, 0.0f);
    }
    else
    {
        field.fill(0.0f);
    }
}

inline double positive_or_fallback(double value, double fallback)
{
    return (std::isfinite(value) && value > 0.0) ? value : fallback;
}
}

/**
 * @brief Implements the explicit diffusion scheme.
 */

/**
 * @brief Initializes the explicit diffusion scheme.
 */

ExplicitDiffusionScheme::ExplicitDiffusionScheme()
{
}

/**
 * @brief Initializes the explicit diffusion scheme.
 */
void ExplicitDiffusionScheme::initialize()
{
    initialize(DiffusionConfig{});
}

/**
 * @brief Initializes the explicit diffusion scheme.
 */
void ExplicitDiffusionScheme::initialize(const DiffusionConfig& cfg)
{
    config_ = cfg;

    tmv::log_info("Initialized explicit diffusion scheme");
    tmv::log_info("  K_h = ", cfg.K_h, " m²/s, K_v = ", cfg.K_v, " m²/s");
}

/**
 * @brief Computes the diffusion tendencies.
 */
void ExplicitDiffusionScheme::compute_diffusion_tendencies(
    const DiffusionConfig& cfg,
    const DiffusionStateView& state,
    DiffusionTendencies& tendencies,
    DiffusionDiagnostics* diag_opt
)
{
    if (!state.grid)
    {
        throw std::runtime_error("Invalid grid for explicit diffusion");
    }

    const auto* ref_field = state.theta ? state.theta : (state.u ? state.u : state.qv);
    if (ref_field == nullptr || ref_field->empty())
    {
        return;
    }
    const int nr = ref_field->size_r();
    const int nth = ref_field->size_th();
    const int nz = ref_field->size_z();

    ensure_tendency_shape(tendencies.dudt_diff, nr, nth, nz);
    ensure_tendency_shape(tendencies.dvdt_diff, nr, nth, nz);
    ensure_tendency_shape(tendencies.dwdt_diff, nr, nth, nz);
    ensure_tendency_shape(tendencies.dthetadt_diff, nr, nth, nz);
    ensure_tendency_shape(tendencies.dqvdt_diff, nr, nth, nz);

    const bool has_variable_scalar_k = cfg.use_variable_K && matches_shape(state.K_scalar, nr, nth, nz);
    const bool has_variable_momentum_k = cfg.use_variable_K && matches_shape(state.K_momentum, nr, nth, nz);
    const Field3D* scalar_k = has_variable_scalar_k ? state.K_scalar : nullptr;
    const Field3D* momentum_k = has_variable_momentum_k ? state.K_momentum : nullptr;

    const double scalar_k_default = positive_or_fallback(cfg.K_v, positive_or_fallback(cfg.K_h, 0.0));
    const double momentum_k_default = positive_or_fallback(cfg.K_v, positive_or_fallback(cfg.K_h, 0.0));

    if (state.u && state.v && state.w &&
        matches_shape(state.u, nr, nth, nz) &&
        matches_shape(state.v, nr, nth, nz) &&
        matches_shape(state.w, nr, nth, nz))
    {
        compute_momentum_diffusion(*state.u, *state.v, *state.w, momentum_k,
                                   momentum_k_default, *state.grid,
                                   tendencies.dudt_diff, tendencies.dvdt_diff, tendencies.dwdt_diff);
    }

    if (state.theta && matches_shape(state.theta, nr, nth, nz))
    {
        compute_scalar_diffusion(*state.theta, scalar_k, scalar_k_default,
                                 *state.grid, tendencies.dthetadt_diff);
    }

    if (state.qv && matches_shape(state.qv, nr, nth, nz))
    {
        compute_scalar_diffusion(*state.qv, scalar_k, scalar_k_default,
                                 *state.grid, tendencies.dqvdt_diff);
    }

    if (diag_opt)
    {
        if (diag_opt->K_effective.size_r() != nr ||
            diag_opt->K_effective.size_th() != nth ||
            diag_opt->K_effective.size_z() != nz)
        {
            diag_opt->K_effective.resize(nr, nth, nz, 0.0f);
        }

        if (scalar_k != nullptr)
        {
            for (int i = 0; i < nr; ++i)
            {
                for (int j = 0; j < nth; ++j)
                {
                    for (int k = 0; k < nz; ++k)
                    {
                        diag_opt->K_effective[i][j][k] = static_cast<float>((*scalar_k)[i][j][k]);
                    }
                }
            }
        }
        else
        {
            diag_opt->K_effective.fill(static_cast<float>(scalar_k_default));
        }
        diag_opt->max_diffusion_number = check_stability(cfg, state);
    }
}

/**
 * @brief Checks the stability of the diffusion scheme.
 */
double ExplicitDiffusionScheme::check_stability(
    const DiffusionConfig& cfg,
    const DiffusionStateView& state
)
{
    if (!state.grid) return 0.0;

    double max_diff_num = 0.0;
    double K_max = std::max(std::max(0.0, cfg.K_h), std::max(0.0, cfg.K_v));

    const double dx_safe = std::max(state.grid->dx, 1.0e-6);
    const double dy_safe = std::max(state.grid->dy, 1.0e-6);
    const double dz0_safe = (!state.grid->dz.empty()) ? std::max(state.grid->dz[0], 1.0e-6) : 1.0;
    double diff_num_x = 2.0 * K_max * cfg.dt_diffusion / (dx_safe * dx_safe);
    double diff_num_y = 2.0 * K_max * cfg.dt_diffusion / (dy_safe * dy_safe);
    double diff_num_z = 2.0 * K_max * cfg.dt_diffusion / (dz0_safe * dz0_safe);

    max_diff_num = std::max({diff_num_x, diff_num_y, diff_num_z});

    return max_diff_num;
}

/**
 * @brief Computes the scalar diffusion.
 */
void ExplicitDiffusionScheme::compute_scalar_diffusion(
    const Field3D& field,
    const Field3D* K_field,
    double K_default,
    const GridMetrics& grid,
    Field3D& tendency
)
{
    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        return;
    }

    if (K_default <= 0.0 && K_field == nullptr)
    {
        tendency.fill(0.0f);
        return;
    }

    // GPU dispatch: uniform-K, no terrain metrics, uniform dz.
    // The shader computes dst = src + dt*kappa*laplacian. We pass very large
    // dr/dtheta values to suppress horizontal terms, leaving only vertical
    // diffusion (matching the CPU path). Then convert to tendency.
    if (K_field == nullptr && !grid_metric::has_terrain_metrics(grid))
    {
        // Use a unit timestep to get tendency = kappa * d²field/dz²
        const float dt_unit = 1.0f;
        const float kappa_f = static_cast<float>(K_default);
        // Large values suppress horizontal Laplacian terms: 1/dr² ≈ 0, 1/(r²dθ²) ≈ 0
        const float large_spacing = 1.0e+12f;
        const float dz_f = static_cast<float>(grid_metric::local_dz(grid, 0, 0, 1, nz));

        auto dst_guard = FieldPool::instance().scoped_acquire(nr, nth, nz);
        Field3D& dst = dst_guard.field;
        // Copy src into dst (shader writes interior only; boundaries need valid data)
        std::memcpy(dst.data(), field.data(), field.size() * sizeof(float));

        if (dispatch_diffusion_backend(
                field.data(), dst.data(),
                nr, nth, nz,
                large_spacing, large_spacing, dz_f,
                dt_unit, kappa_f))
        {
            // Convert updated field to tendency: tendency = dst - src
            tendency.resize(nr, nth, nz);
            const int total = static_cast<int>(field.size());
            simd_utils::subtract_vectors(dst.data(), field.data(), tendency.data(), total);
            return;
        }
        // GPU dispatch failed — fall through to CPU path
    }

    // SIMD-accelerated path: uniform K, uniform dz, no terrain.
    // diffuse_1d() computes dst[k] = src[k] + scale*(src[k-1] - 2*src[k] + src[k+1]).
    // We subtract src afterwards to get tendency = scale * laplacian.
    if (K_field == nullptr && !grid_metric::has_terrain_metrics(grid) && nz >= 3)
    {
        const double dz_uniform = grid_metric::local_dz(grid, 0, 0, 1, nz);
        const float scale = static_cast<float>(K_default / (dz_uniform * dz_uniform));

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                // diffuse_1d processes elements [0, nz_inner).
                // Caller must ensure src has valid data at index -1 and nz_inner.
                // Since field(i,j,*) starts at k=0 and ends at k=nz-1,
                // we pass &field(i,j,1) as src so src[-1]=field(i,j,0) and
                // src[nz_inner]=field(i,j,nz-1) are valid.
                const int nz_inner = nz - 2;
                const float* src = &field(i, j, 1);
                float* dst = &tendency(i, j, 1);
                simd_utils::diffuse_1d(src, dst, scale, nz_inner);
                // diffuse_1d gives: dst[k] = src[k] + scale*laplacian
                // We want tendency = scale * laplacian = dst[k] - src[k]
                simd_utils::subtract_vectors(dst, src, dst, nz_inner);
                tendency(i, j, 0) = 0.0f;
                tendency(i, j, nz - 1) = 0.0f;
            }
        }
        return;
    }

    // General CPU path: variable K, terrain-corrected dz, or non-uniform grid.
    auto dz_at = [&](int i, int j, int level) -> double
    {
        return grid_metric::local_dz(grid, i, j, level, nz);
    };

    auto sanitize_diffusivity = [K_default](double value) -> double
    {
        if (!std::isfinite(value))
        {
            return std::max(0.0, K_default);
        }
        return std::max(0.0, value);
    };

    // Cache-blocked j-tiling: keeps stencil neighborhood in L1/L2 cache.
    // At TILE_J=32, NZ=128: working set per tile ≈ 3×34×128×8 = 104 KB.
    static constexpr int TILE_J = 32;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i)
    {
        for (int jt = 0; jt < nth; jt += TILE_J)
        {
            const int j_end = std::min(jt + TILE_J, nth);
            for (int j = jt; j < j_end; ++j)
            {
                for (int k = 1; k < nz - 1; ++k)
                {
                    const double dz = dz_at(i, j, k);
                    const double dz_up = dz_at(i, j, k - 1);
                    const double dz_down = dz_at(i, j, k + 1);

                    const double phi_km = field(i, j, k - 1);
                    const double phi_k = field(i, j, k);
                    const double phi_kp = field(i, j, k + 1);

                    const double dphi_dz_up = (phi_k - phi_km) / dz_up;
                    const double dphi_dz_down = (phi_kp - phi_k) / dz_down;

                    const double k_here_raw = (K_field != nullptr) ? (*K_field)(i, j, k) : K_default;
                    const double k_up_raw = (K_field != nullptr) ? (*K_field)(i, j, k - 1) : K_default;
                    const double k_down_raw = (K_field != nullptr) ? (*K_field)(i, j, k + 1) : K_default;
                    const double k_here = sanitize_diffusivity(k_here_raw);
                    const double k_up = sanitize_diffusivity(k_up_raw);
                    const double k_down = sanitize_diffusivity(k_down_raw);

                    const double K_avg_up = 0.5 * (k_here + k_up);
                    const double K_avg_down = 0.5 * (k_here + k_down);

                    const double flux_up = -K_avg_up * dphi_dz_up;
                    const double flux_down = -K_avg_down * dphi_dz_down;
                    const double dflux_dz = (flux_down - flux_up) / dz;

                    tendency(i, j, k) = static_cast<float>(-dflux_dz);
                }
            }
        }
    }

    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            tendency(i, j, 0) = 0.0f;
            tendency(i, j, nz - 1) = 0.0f;
        }
    }
}

/**
 * @brief Computes the momentum diffusion.
 */
void ExplicitDiffusionScheme::compute_momentum_diffusion(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D* nu_t,
    double nu_default,
    const GridMetrics& grid,
    Field3D& du_dt,
    Field3D& dv_dt,
    Field3D& dw_dt
)
{
    const int nr = u.size_r();
    const int nth = u.size_th();
    const int nz = u.size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0)
    {
        return;
    }

    if (nu_default <= 0.0 && nu_t == nullptr)
    {
        du_dt.fill(0.0f);
        dv_dt.fill(0.0f);
        dw_dt.fill(0.0f);
        return;
    }

    auto dz_at = [&](int i, int j, int level) -> double
    {
        return grid_metric::local_dz(grid, i, j, level, nz);
    };

    auto sanitize_viscosity = [nu_default](double value) -> double
    {
        if (!std::isfinite(value))
        {
            return std::max(0.0, nu_default);
        }
        return std::max(0.0, value);
    };

    static constexpr int TILE_J_MOM = 32;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nr; ++i)
    {
        for (int jt = 0; jt < nth; jt += TILE_J_MOM)
        {
            const int j_end = std::min(jt + TILE_J_MOM, nth);
            for (int j = jt; j < j_end; ++j)
            {
            for (int k = 1; k < nz - 1; ++k)
            {
                const double dz = dz_at(i, j, k);

                const double nu_here_raw = (nu_t != nullptr) ? (*nu_t)(i, j, k) : nu_default;
                const double nu_up_raw = (nu_t != nullptr) ? (*nu_t)(i, j, k - 1) : nu_default;
                const double nu_down_raw = (nu_t != nullptr) ? (*nu_t)(i, j, k + 1) : nu_default;
                const double nu_here = sanitize_viscosity(nu_here_raw);
                const double nu_up = sanitize_viscosity(nu_up_raw);
                const double nu_down = sanitize_viscosity(nu_down_raw);
                const double nu_avg_up = 0.5 * (nu_here + nu_up);
                const double nu_avg_down = 0.5 * (nu_here + nu_down);

                const double du_dz_up = (u(i, j, k) - u(i, j, k - 1)) / dz_at(i, j, k - 1);
                const double du_dz_down = (u(i, j, k + 1) - u(i, j, k)) / dz_at(i, j, k + 1);
                du_dt(i, j, k) = static_cast<float>(-(nu_avg_down * du_dz_down - nu_avg_up * du_dz_up) / dz);

                const double dv_dz_up = (v(i, j, k) - v(i, j, k - 1)) / dz_at(i, j, k - 1);
                const double dv_dz_down = (v(i, j, k + 1) - v(i, j, k)) / dz_at(i, j, k + 1);
                dv_dt(i, j, k) = static_cast<float>(-(nu_avg_down * dv_dz_down - nu_avg_up * dv_dz_up) / dz);

                const double dw_dz_up = (w(i, j, k) - w(i, j, k - 1)) / dz_at(i, j, k - 1);
                const double dw_dz_down = (w(i, j, k + 1) - w(i, j, k)) / dz_at(i, j, k + 1);
                dw_dt(i, j, k) = static_cast<float>(-(nu_avg_down * dw_dz_down - nu_avg_up * dw_dz_up) / dz);
            }
            }
        }
    }

    for (int i = 0; i < nr; ++i)
    {
        for (int j = 0; j < nth; ++j)
        {
            du_dt(i, j, 0) = 0.0f;
            dv_dt(i, j, 0) = 0.0f;
            dw_dt(i, j, 0) = 0.0f;
            du_dt(i, j, nz - 1) = 0.0f;
            dv_dt(i, j, nz - 1) = 0.0f;
            dw_dt(i, j, nz - 1) = 0.0f;
        }
    }
}
