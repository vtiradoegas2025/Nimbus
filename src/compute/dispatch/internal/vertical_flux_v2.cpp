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

inline float tvd_limiter_f(float r, TVDLimiterKind kind)
{
    switch (kind)
    {
        case TVDLimiterKind::minmod:
            return std::max(0.0f, std::min(1.0f, r));
        case TVDLimiterKind::vanleer:
            return (std::abs(r) + r) / (1.0f + std::abs(r));
        case TVDLimiterKind::superbee:
            return std::max({0.0f, std::min(1.0f, 2.0f * r), std::min(2.0f, r)});
        case TVDLimiterKind::universal:
            if (r >= 0.0f && r <= 1.0f)
                return std::min(2.0f * r, (1.0f + r) * 0.5f);
            if (r > 1.0f)
                return std::min(r, 2.0f);
            return 0.0f;
        case TVDLimiterKind::mc:
        default:
            return std::max(0.0f, std::min({(1.0f + r) * 0.5f, 2.0f, 2.0f * r}));
    }
}

} // namespace

bool dispatch_tvd_vertical_flux_v2(
    const AdvectionConfig& cfg,
    const AdvectionStateView& state,
    AdvectionTendencies& tendencies,
    AdvectionDiagnostics* diag_opt)
{
    if (!state.q || !state.w || !state.grid)
    {
        return false;
    }
    if (state.grid->terrain_metrics_active)
    {
        return false;
    }

    const int nr = state.q->size_r();
    const int nth = state.q->size_th();
    const int nz = state.q->size_z();
    if (nr <= 0 || nth <= 0 || nz <= 2)
    {
        // Fall back to v1 for trivial grids
        return dispatch_tvd_vertical_flux_v1(cfg, state, tendencies, diag_opt);
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

    // Detect fused integration mode: write dst = src + dt*tendency directly,
    // eliminating the separate integration pass that otherwise re-reads dqdt
    // from memory. This halves the memory traffic for the integration step.
    const bool fused = (tendencies.fused_dst != nullptr &&
                        tendencies.fused_src != nullptr);
    float* __restrict__ fused_dst = tendencies.fused_dst;
    const float* __restrict__ fused_src = tendencies.fused_src;
    const float fused_dt = tendencies.fused_dt;
    tendencies.fused_completed = false;

    tendencies.dqdt_adv.resize(nr, nth, nz, 0.0f);
    float* __restrict__ dqdt_data = fused ? nullptr : tendencies.dqdt_adv.data();
    const float* __restrict__ q_data = state.q->data();
    const float* __restrict__ w_data = state.w->data();

    const std::vector<double>& dz_levels = state.grid->dz;
    const bool use_level_dz = static_cast<int>(dz_levels.size()) >= nz;
    const float dt_safe = static_cast<float>(std::max(std::abs(cfg.positivity_dt), 1.0e-12));
    const bool positivity = cfg.positivity;

    // Pre-convert dz to float for the common uniform-grid case.
    std::vector<float> dz_f(static_cast<std::size_t>(nz), 1.0f);
    if (use_level_dz)
    {
        for (int k = 0; k < nz; ++k)
        {
            dz_f[k] = static_cast<float>(
                std::max(std::abs(dz_levels[k]), 1.0e-6));
        }
    }
    const float* dz_fp = dz_f.data();

    // Pre-compute reciprocal dz for the hot loop (multiply is faster than divide).
    std::vector<float> inv_dz_f(static_cast<std::size_t>(nz));
    if (use_level_dz)
    {
        for (int k = 0; k < nz; ++k)
        {
            inv_dz_f[k] = 1.0f / dz_fp[k];
        }
    }
    const float* inv_dz_fp = inv_dz_f.data();

    constexpr float eps_f = 1.0e-12f;
    const int total_cols = nr * nth;

    // Batch size for multi-column processing. Each batch processes BATCH
    // columns at the same k-level simultaneously. The inner k-loops operate
    // on contiguous float arrays of length BATCH, which the compiler
    // auto-vectorizes to NEON/SSE/AVX instructions.
    constexpr int BATCH = 8;
    const int n_batches = total_cols / BATCH;

    float global_max_cfl = 0.0f;

    #pragma omp parallel reduction(max:global_max_cfl)
    {
        // Per-thread scratch arrays: [BATCH] elements at each k-level.
        // Layout: scratch[array_index * nz * BATCH + k * BATCH + b]
        // This gives contiguous BATCH-wide rows at each k for vectorization.
        const std::size_t nzs = static_cast<std::size_t>(nz);
        const std::size_t plane = nzs * BATCH;
        std::vector<float> scratch(plane * 6, 0.0f);
        float* q_bat    = scratch.data();            // [nz][BATCH]
        float* w_bat    = q_bat + plane;             // [nz][BATCH]
        float* ql_bat   = w_bat + plane;             // [nz][BATCH] left state
        float* qr_bat   = ql_bat + plane;            // [nz][BATCH] right state
        float* dqdt_bat = qr_bat + plane;            // [nz][BATCH] tendency
        float* dz_bat   = dqdt_bat + plane;          // [nz][BATCH] grid spacing

        float thread_max_cfl = 0.0f;

        // --- Process full batches ---
        #pragma omp for schedule(static)
        for (int batch = 0; batch < n_batches; ++batch)
        {
            const int col0 = batch * BATCH;

            // Load BATCH columns into interleaved [nz][BATCH] layout.
            for (int b = 0; b < BATCH; ++b)
            {
                const std::size_t base =
                    static_cast<std::size_t>(col0 + b) * nzs;
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t off = static_cast<std::size_t>(k) * BATCH + b;
                    q_bat[off] = q_data[base + k];
                    w_bat[off] = w_data[base + k];
                }
            }

            // Grid spacing: uniform case shares dz across all columns.
            if (use_level_dz)
            {
                for (int k = 0; k < nz; ++k)
                {
                    const float dz_k = dz_fp[k];
                    const std::size_t row = static_cast<std::size_t>(k) * BATCH;
                    for (int b = 0; b < BATCH; ++b) dz_bat[row + b] = dz_k;
                }
            }
            else
            {
                for (int b = 0; b < BATCH; ++b)
                {
                    const int col = col0 + b;
                    const int ci = col / nth, cj = col % nth;
                    for (int k = 0; k < nz; ++k)
                    {
                        float dz = static_cast<float>(
                            grid_metric::vertical_cell_thickness(
                                *state.grid, ci, cj, k, nz));
                        dz_bat[static_cast<std::size_t>(k) * BATCH + b] =
                            std::max(std::abs(dz), 1.0e-6f);
                    }
                }
            }

            // CFL diagnostic (vectorizable: BATCH independent abs+div per k)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t row = static_cast<std::size_t>(k) * BATCH;
                for (int b = 0; b < BATCH; ++b)
                {
                    float cfl = std::abs(w_bat[row + b]) / dz_bat[row + b];
                    if (cfl > thread_max_cfl) thread_max_cfl = cfl;
                }
            }

            // Initialize tendency
            for (std::size_t i = 0; i < plane; ++i) dqdt_bat[i] = 0.0f;

            // --- MUSCL reconstruction (vectorizable inner loop) ---
            // Boundary cells: piecewise constant
            for (int b = 0; b < BATCH; ++b)
            {
                ql_bat[b] = q_bat[b]; // k=0
                qr_bat[b] = q_bat[b];
                const std::size_t last = static_cast<std::size_t>(nz - 1) * BATCH + b;
                ql_bat[last] = q_bat[last]; // k=nz-1
                qr_bat[last] = q_bat[last];
            }

            // Interior cells: the inner b-loop is BATCH-wide and vectorizable.
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t row  = static_cast<std::size_t>(k) * BATCH;
                const std::size_t rowm = static_cast<std::size_t>(k - 1) * BATCH;
                const std::size_t rowp = static_cast<std::size_t>(k + 1) * BATCH;

                #pragma omp simd
                for (int b = 0; b < BATCH; ++b)
                {
                    const float qk  = q_bat[row + b];
                    const float qkm = q_bat[rowm + b];
                    const float qkp = q_bat[rowp + b];
                    const float denom = qkp - qk + eps_f;
                    const float r = (qk - qkm) / denom;
                    const float phi = tvd_limiter_f(r, limiter);
                    const float delta_q = phi * (qkp - qk);
                    ql_bat[row + b] = qk - 0.5f * delta_q;
                    qr_bat[row + b] = qk + 0.5f * delta_q;
                }
            }

            // --- Upwind flux divergence (vectorizable inner loop) ---
            // k=0: no left flux
            {
                const std::size_t row0 = 0;
                const std::size_t row1 = static_cast<std::size_t>(BATCH);
                const float inv_dz0 = use_level_dz ? inv_dz_fp[0] : 0.0f;
                for (int b = 0; b < BATCH; ++b)
                {
                    const float vel = w_bat[row0 + b];
                    // Branchless upwind: select via sign mask
                    const float sel = (vel >= 0.0f) ? 1.0f : 0.0f;
                    const float flux_r = vel * (sel * qr_bat[row0 + b] +
                                                (1.0f - sel) * ql_bat[row1 + b]);
                    const float idz = use_level_dz ? inv_dz0
                        : (1.0f / dz_bat[row0 + b]);
                    dqdt_bat[row0 + b] = -flux_r * idz;
                }
            }

            // k=1..nz-2: general case with left and right fluxes
            for (int k = 1; k < nz - 1; ++k)
            {
                const std::size_t row  = static_cast<std::size_t>(k) * BATCH;
                const std::size_t rowm = static_cast<std::size_t>(k - 1) * BATCH;
                const std::size_t rowp = static_cast<std::size_t>(k + 1) * BATCH;
                const float idz_k = use_level_dz ? inv_dz_fp[k] : 0.0f;

                #pragma omp simd
                for (int b = 0; b < BATCH; ++b)
                {
                    // Right flux at k+1/2
                    const float vel_r = w_bat[row + b];
                    const float sel_r = (vel_r >= 0.0f) ? 1.0f : 0.0f;
                    const float flux_r = vel_r * (sel_r * qr_bat[row + b] +
                                                  (1.0f - sel_r) * ql_bat[rowp + b]);

                    // Left flux at k-1/2
                    const float vel_l = w_bat[rowm + b];
                    const float sel_l = (vel_l >= 0.0f) ? 1.0f : 0.0f;
                    const float flux_l = vel_l * (sel_l * qr_bat[rowm + b] +
                                                  (1.0f - sel_l) * ql_bat[row + b]);

                    const float idz = use_level_dz ? idz_k
                        : (1.0f / dz_bat[row + b]);
                    dqdt_bat[row + b] = -(flux_r - flux_l) * idz;
                }
            }

            // --- Positivity limiting (vectorizable) ---
            if (positivity)
            {
                const float neg_inv_dt = -1.0f / dt_safe;
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t row = static_cast<std::size_t>(k) * BATCH;
                    for (int b = 0; b < BATCH; ++b)
                    {
                        float t = dqdt_bat[row + b];
                        const float floor_t = q_bat[row + b] * neg_inv_dt;
                        // Branchless: max(t, floor_t) with NaN -> 0
                        t = std::isfinite(t) ? t : 0.0f;
                        t = (t > floor_t) ? t : floor_t;
                        dqdt_bat[row + b] = t;
                    }
                }
            }

            // --- Store results back (scatter from [nz][BATCH] to field) ---
            if (fused)
            {
                // Fused: write dst = src + dt * tendency directly.
                // Reads src from fused_src (avoids re-reading q_data which
                // may differ from src when boundaries are pre-copied).
                for (int b = 0; b < BATCH; ++b)
                {
                    const std::size_t base =
                        static_cast<std::size_t>(col0 + b) * nzs;
                    for (int k = 0; k < nz; ++k)
                    {
                        float t = dqdt_bat[static_cast<std::size_t>(k) * BATCH + b];
                        t = std::isfinite(t) ? t : 0.0f;
                        fused_dst[base + k] = fused_src[base + k] + fused_dt * t;
                    }
                }
            }
            else
            {
                for (int b = 0; b < BATCH; ++b)
                {
                    const std::size_t base =
                        static_cast<std::size_t>(col0 + b) * nzs;
                    for (int k = 0; k < nz; ++k)
                    {
                        float t = dqdt_bat[static_cast<std::size_t>(k) * BATCH + b];
                        dqdt_data[base + k] = std::isfinite(t) ? t : 0.0f;
                    }
                }
            }
        }

        // --- Process remainder columns (< BATCH) with scalar v1 logic ---
        #pragma omp for schedule(static)
        for (int col = n_batches * BATCH; col < total_cols; ++col)
        {
            const std::size_t base = static_cast<std::size_t>(col) * nzs;
            float* q_col  = q_bat;
            float* w_col  = w_bat;
            float* q_l    = ql_bat;
            float* q_r    = qr_bat;
            float* dq_col = dqdt_bat;

            for (int k = 0; k < nz; ++k)
            {
                q_col[k] = q_data[base + k];
                w_col[k] = w_data[base + k];
            }
            for (int k = 0; k < nz; ++k)
            {
                float dz_k = use_level_dz ? dz_fp[k]
                    : static_cast<float>(std::max(std::abs(
                          grid_metric::vertical_cell_thickness(*state.grid,
                              col / nth, col % nth, k, nz)), 1.0e-6));
                float cfl = std::abs(w_col[k]) / dz_k;
                if (cfl > thread_max_cfl) thread_max_cfl = cfl;
            }
            for (int k = 0; k < nz; ++k) dq_col[k] = 0.0f;
            q_l[0] = q_r[0] = q_col[0];
            q_l[nz-1] = q_r[nz-1] = q_col[nz-1];
            for (int k = 1; k < nz - 1; ++k)
            {
                float denom = q_col[k+1] - q_col[k] + eps_f;
                float r = (q_col[k] - q_col[k-1]) / denom;
                float phi = tvd_limiter_f(r, limiter);
                float dq = phi * (q_col[k+1] - q_col[k]);
                q_l[k] = q_col[k] - 0.5f * dq;
                q_r[k] = q_col[k] + 0.5f * dq;
            }
            for (int k = 0; k < nz - 1; ++k)
            {
                float vel_r = w_col[k];
                float sel_r = (vel_r >= 0.0f) ? 1.0f : 0.0f;
                float fx_r = vel_r * (sel_r * q_r[k] + (1.0f - sel_r) * q_l[k+1]);
                float fx_l = 0.0f;
                if (k > 0)
                {
                    float vel_l = w_col[k-1];
                    float sel_l = (vel_l >= 0.0f) ? 1.0f : 0.0f;
                    fx_l = vel_l * (sel_l * q_r[k-1] + (1.0f - sel_l) * q_l[k]);
                }
                float idz = use_level_dz ? inv_dz_fp[k]
                    : (1.0f / (use_level_dz ? dz_fp[k] : static_cast<float>(
                        std::max(std::abs(grid_metric::vertical_cell_thickness(*state.grid,
                            col/nth, col%nth, k, nz)), 1.0e-6))));
                dq_col[k] = -(fx_r - fx_l) * idz;
            }
            if (positivity)
            {
                float neg_inv_dt = -1.0f / dt_safe;
                for (int k = 0; k < nz; ++k)
                {
                    float t = dq_col[k];
                    t = std::isfinite(t) ? t : 0.0f;
                    float fl = q_col[k] * neg_inv_dt;
                    dq_col[k] = (t > fl) ? t : fl;
                }
            }
            if (fused)
            {
                for (int k = 0; k < nz; ++k)
                {
                    float t = dq_col[k];
                    t = std::isfinite(t) ? t : 0.0f;
                    fused_dst[base + k] = fused_src[base + k] + fused_dt * t;
                }
            }
            else
            {
                for (int k = 0; k < nz; ++k)
                {
                    float t = dq_col[k];
                    dqdt_data[base + k] = std::isfinite(t) ? t : 0.0f;
                }
            }
        }

        global_max_cfl = std::max(global_max_cfl, thread_max_cfl);
    }

    if (fused)
    {
        tendencies.fused_completed = true;
    }

    if (diag_opt)
    {
        diag_opt->max_cfl_z = static_cast<double>(global_max_cfl);
        diag_opt->suggested_dt = (global_max_cfl > 1.0e-12f)
            ? (cfg.cfl_target / static_cast<double>(global_max_cfl))
            : std::numeric_limits<double>::infinity();
    }

    return true;
}

} // namespace nimbus::compute::dispatch_internal
