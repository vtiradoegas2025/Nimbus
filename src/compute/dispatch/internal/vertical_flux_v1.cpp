#include "compute/dispatch/internal/vertical_flux.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "util/grid_metric_utils.hpp"

namespace nimbus::compute::dispatch_internal
{
namespace
{

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

} // namespace

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
    // This template advances a Cartesian vertical flux using physical w.
    // Terrain transport requires the retained C-grid contravariant mass-flux
    // operator and must never be approximated by pointwise terrain spacing.
    if (state.grid->terrain_metrics_active)
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

    const std::vector<double>& dz_levels = state.grid->dz;
    const bool use_level_dz = static_cast<int>(dz_levels.size()) >= nz;
    const double dt_safe = std::max(std::abs(cfg.positivity_dt), 1.0e-12);

    double max_cfl = 0.0;

    // Stack-allocate per-thread column buffers to avoid heap allocation overhead.
    // MAX_VERT_LEVELS covers all supported grid configurations (student=32,
    // research=64, production=128, max=512). Total: 6*512*8 = 24KB per thread.
    static constexpr int MAX_VERT_LEVELS = 512;

    #pragma omp parallel reduction(max:max_cfl)
    {
        double q_col[MAX_VERT_LEVELS];
        double w_col[MAX_VERT_LEVELS];
        double dz_col[MAX_VERT_LEVELS];
        double q_left[MAX_VERT_LEVELS];
        double q_right[MAX_VERT_LEVELS];
        double dqdt_col[MAX_VERT_LEVELS];

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
                        local_dz = grid_metric::vertical_cell_thickness(
                            *state.grid, i, j, k, nz);
                    }
                    local_dz = std::max(std::abs(local_dz), 1.0e-6);
                    dz_col[kk] = local_dz;

                    const double cfl = std::abs(w_col[kk]) / local_dz;
                    if (cfl > max_cfl)
                    {
                        max_cfl = cfl;
                    }
                }

                std::fill(dqdt_col, dqdt_col + nz, 0.0);
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

} // namespace nimbus::compute::dispatch_internal
