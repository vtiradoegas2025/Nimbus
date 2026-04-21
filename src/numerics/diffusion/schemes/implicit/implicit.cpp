/**
 * @file implicit.cpp
 * @brief Implementation for the numerics module.
 *
 * Provides executable logic for the numerics runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/numerics subsystem.
 */

#include "implicit.hpp"
#include "core/field/field_pool.hpp"
#include "util/grid_metric_utils.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

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
 * @brief Implements the implicit diffusion scheme.
 */
ImplicitDiffusionScheme::ImplicitDiffusionScheme() {}

/**
 * @brief Initializes the implicit diffusion scheme.
 */
void ImplicitDiffusionScheme::initialize()
{
    initialize(DiffusionConfig{});
}

/**
 * @brief Initializes the implicit diffusion scheme.
 */
void ImplicitDiffusionScheme::initialize(const DiffusionConfig& cfg)
{
    config_ = cfg;

    std::cout << "Initialized implicit diffusion scheme" << std::endl;
    std::cout << "  Vertical implicit diffusion enabled" << std::endl;
}

/**
 * @brief Computes the diffusion tendencies.
 */
void ImplicitDiffusionScheme::compute_diffusion_tendencies(
    const DiffusionConfig& cfg,
    const DiffusionStateView& state,
    DiffusionTendencies& tendencies,
    DiffusionDiagnostics* diag_opt
)
{
    if (!state.grid)
    {
        throw std::runtime_error("Invalid grid for implicit diffusion");
    }

    const auto* ref_field = state.theta ? state.theta : (state.u ? state.u : state.qv);
    if (ref_field == nullptr || ref_field->empty())
    {
        return;
    }
    const int nr = ref_field->size_r();
    const int nth = ref_field->size_th();
    const int nz = ref_field->size_z();
    const double dt_eff = std::max(cfg.dt_diffusion, 1.0e-6);

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

    auto& pool = FieldPool::instance();

    if (state.theta && matches_shape(state.theta, nr, nth, nz))
    {
        auto theta_guard = pool.scoped_acquire_copy(*state.theta);
        Field3D& theta_new = theta_guard.field;
        implicit_vertical_diffusion(*state.theta, scalar_k, scalar_k_default, *state.grid, dt_eff, theta_new);

        const double inv_dt = 1.0 / dt_eff;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                for (int k = 0; k < nz; ++k)
                {
                    tendencies.dthetadt_diff(i, j, k) =
                        static_cast<float>((theta_new(i, j, k) - (*state.theta)(i, j, k)) * inv_dt);
                }
            }
        }
    }

    if (state.qv && matches_shape(state.qv, nr, nth, nz))
    {
        auto qv_guard = pool.scoped_acquire_copy(*state.qv);
        Field3D& qv_new = qv_guard.field;
        implicit_vertical_diffusion(*state.qv, scalar_k, scalar_k_default, *state.grid, dt_eff, qv_new);

        const double inv_dt = 1.0 / dt_eff;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                for (int k = 0; k < nz; ++k)
                {
                    tendencies.dqvdt_diff(i, j, k) =
                        static_cast<float>((qv_new(i, j, k) - (*state.qv)(i, j, k)) * inv_dt);
                }
            }
        }
    }

    if (state.u && state.v && state.w &&
        matches_shape(state.u, nr, nth, nz) &&
        matches_shape(state.v, nr, nth, nz) &&
        matches_shape(state.w, nr, nth, nz))
    {
        auto u_guard = pool.scoped_acquire_copy(*state.u);
        auto v_guard = pool.scoped_acquire_copy(*state.v);
        auto w_guard = pool.scoped_acquire_copy(*state.w);
        Field3D& u_new = u_guard.field;
        Field3D& v_new = v_guard.field;
        Field3D& w_new = w_guard.field;

        implicit_vertical_momentum_diffusion(*state.u, *state.v, *state.w, momentum_k,
                                             momentum_k_default, *state.grid, dt_eff,
                                             u_new, v_new, w_new);

        const double inv_dt = 1.0 / dt_eff;
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                for (int k = 0; k < nz; ++k)
                {
                    tendencies.dudt_diff(i, j, k) =
                        static_cast<float>((u_new(i, j, k) - (*state.u)(i, j, k)) * inv_dt);
                    tendencies.dvdt_diff(i, j, k) =
                        static_cast<float>((v_new(i, j, k) - (*state.v)(i, j, k)) * inv_dt);
                    tendencies.dwdt_diff(i, j, k) =
                        static_cast<float>((w_new(i, j, k) - (*state.w)(i, j, k)) * inv_dt);
                }
            }
        }
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
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nr; ++i)
            {
                for (int j = 0; j < nth; ++j)
                {
                    for (int k = 0; k < nz; ++k)
                    {
                        diag_opt->K_effective(i, j, k) = static_cast<float>((*scalar_k)(i, j, k));
                    }
                }
            }
        }
        else
        {
            diag_opt->K_effective.fill(static_cast<float>(scalar_k_default));
        }
        diag_opt->max_diffusion_number = 1.0;
    }
}

/**
 * @brief Checks the stability of the diffusion scheme.
 */
double ImplicitDiffusionScheme::check_stability(
    const DiffusionConfig& cfg,
    const DiffusionStateView& state
)
{
    (void)cfg;
    (void)state;
    return 1.0;
}

/**
 * @brief Solves the tridiagonal system.
 */
void ImplicitDiffusionScheme::solve_tridiagonal(
    const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& rhs,
    std::vector<double>& x
)
{
    const int n = static_cast<int>(std::min(b.size(), rhs.size()));
    if (n <= 0)
    {
        x.clear();
        return;
    }

    x.assign(static_cast<size_t>(n), 0.0);
    if (n == 1)
    {
        const double b0 = (std::abs(b[0]) > 1.0e-12) ? b[0] : ((b[0] >= 0.0) ? 1.0e-12 : -1.0e-12);
        x[0] = rhs[0] / b0;
        return;
    }

    std::vector<double> cp(static_cast<size_t>(n), 0.0);
    std::vector<double> dp(static_cast<size_t>(n), 0.0);

    const double b0 = (std::abs(b[0]) > 1.0e-12) ? b[0] : ((b[0] >= 0.0) ? 1.0e-12 : -1.0e-12);
    cp[0] = (c.empty() ? 0.0 : c[0] / b0);
    dp[0] = rhs[0] / b0;

    for (int i = 1; i < n; ++i)
    {
        const double ai = (i < static_cast<int>(a.size())) ? a[i] : 0.0;
        const double bi = (i < static_cast<int>(b.size())) ? b[i] : 1.0;
        const double ci = (i < static_cast<int>(c.size())) ? c[i] : 0.0;

        double denom = bi - ai * cp[i - 1];
        if (std::abs(denom) <= 1.0e-12)
        {
            denom = (denom >= 0.0) ? 1.0e-12 : -1.0e-12;
        }

        cp[i] = (i < n - 1) ? (ci / denom) : 0.0;
        dp[i] = (rhs[i] - ai * dp[i - 1]) / denom;
    }

    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i)
    {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
}

/**
 * @brief Computes the vertical diffusion.
 */
void ImplicitDiffusionScheme::implicit_vertical_diffusion(
    const Field3D& field,
    const Field3D* K_field,
    double K_default,
    const GridMetrics& grid,
    double dt,
    Field3D& field_new
)
{
    const int nr = field.size_r();
    const int nth = field.size_th();
    const int nz = field.size_z();

    if (field_new.size_r() != nr || field_new.size_th() != nth || field_new.size_z() != nz)
    {
        field_new.resize(nr, nth, nz, 0.0f);
    }
    field_new = field;

    if (nr <= 0 || nth <= 0 || nz <= 2)
    {
        return;
    }
    if (K_default <= 0.0 && K_field == nullptr)
    {
        return;
    }

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

    #pragma omp parallel
    {
        std::vector<double> a(static_cast<size_t>(nz), 0.0);
        std::vector<double> b(static_cast<size_t>(nz), 1.0);
        std::vector<double> c(static_cast<size_t>(nz), 0.0);
        std::vector<double> rhs(static_cast<size_t>(nz), 0.0);
        std::vector<double> phi_new;

        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < nr; ++i)
        {
            for (int j = 0; j < nth; ++j)
            {
                std::fill(a.begin(), a.end(), 0.0);
                std::fill(b.begin(), b.end(), 1.0);
                std::fill(c.begin(), c.end(), 0.0);
                std::fill(rhs.begin(), rhs.end(), 0.0);

                for (int k = 1; k < nz - 1; ++k)
                {
                    const double dz_up = dz_at(i, j, k - 1);
                    const double dz_down = dz_at(i, j, k + 1);
                    const double dz = dz_at(i, j, k);

                    const double K_here_raw = (K_field != nullptr) ? (*K_field)(i, j, k) : K_default;
                    const double K_up_raw = (K_field != nullptr) ? (*K_field)(i, j, k - 1) : K_default;
                    const double K_down_raw = (K_field != nullptr) ? (*K_field)(i, j, k + 1) : K_default;
                    const double K_here = sanitize_diffusivity(K_here_raw);
                    const double K_up_val = sanitize_diffusivity(K_up_raw);
                    const double K_down_val = sanitize_diffusivity(K_down_raw);
                    const double K_up = 0.5 * (K_here + K_up_val);
                    const double K_down = 0.5 * (K_here + K_down_val);

                    const double coef_up = -0.5 * dt * K_up / (dz_up * dz);
                    const double coef_down = -0.5 * dt * K_down / (dz_down * dz);

                    a[k] = coef_up;
                    b[k] = 1.0 - coef_up - coef_down;
                    c[k] = coef_down;

                    const double explicit_up = 0.5 * dt * K_up / (dz_up * dz);
                    const double explicit_down = 0.5 * dt * K_down / (dz_down * dz);

                    rhs[k] = field(i, j, k) +
                             explicit_up * (field(i, j, k - 1) - field(i, j, k)) +
                             explicit_down * (field(i, j, k + 1) - field(i, j, k));
                }

                b[0] = 1.0;
                rhs[0] = field(i, j, 0);
                b[nz - 1] = 1.0;
                rhs[nz - 1] = field(i, j, nz - 1);

                solve_tridiagonal(a, b, c, rhs, phi_new);

                for (int k = 0; k < nz; ++k)
                {
                    field_new(i, j, k) = static_cast<float>(phi_new[k]);
                }
            }
        }
    }
}

/**
 * @brief Computes the vertical momentum diffusion.
 */
void ImplicitDiffusionScheme::implicit_vertical_momentum_diffusion(
    const Field3D& u,
    const Field3D& v,
    const Field3D& w,
    const Field3D* nu_t,
    double nu_default,
    const GridMetrics& grid,
    double dt,
    Field3D& u_new,
    Field3D& v_new,
    Field3D& w_new
)
{
    implicit_vertical_diffusion(u, nu_t, nu_default, grid, dt, u_new);
    implicit_vertical_diffusion(v, nu_t, nu_default, grid, dt, v_new);
    implicit_vertical_diffusion(w, nu_t, nu_default, grid, dt, w_new);
}
