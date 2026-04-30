#include "compute/compute_kernel_template.hpp"
#include "compute/compute_backend.hpp"

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

std::string g_active_vertical_flux_template_id = "tvd_vertical_flux_v2";

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

// ---------------------------------------------------------------------------
// v2: Float-precision TVD vertical flux with multi-column processing.
//
// The v1 kernel processes one column at a time in double precision. This v2
// kernel processes columns in float precision (matching the validated Vulkan
// GPU shader) and groups work so the compiler can auto-vectorize across
// multiple columns sharing the same k-index. On ARM NEON this yields 4x
// throughput from float SIMD; on x86 AVX up to 8x.
//
// Algorithm is identical to v1: MUSCL reconstruction + TVD limiter + upwind
// flux divergence + positivity limiting. The only change is precision
// (double -> float) and loop structure (one-at-a-time -> batched).
// ---------------------------------------------------------------------------

/// Float-precision TVD limiter (branchless where possible).
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

    const bool terrain_metrics = grid_metric::has_terrain_metrics(*state.grid);
    const std::vector<double>& dz_levels = state.grid->dz;
    const bool use_level_dz = !terrain_metrics && static_cast<int>(dz_levels.size()) >= nz;
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
    const int remainder = total_cols % BATCH;

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
                            grid_metric::local_dz(*state.grid, ci, cj, k, nz));
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
                          grid_metric::local_dz(*state.grid,
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
                        std::max(std::abs(grid_metric::local_dz(*state.grid,
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
    return id == "tvdverticalfluxv1" || id == "tvddefaultv1" ||
           id == "tvdverticalfluxv2" || id == "tvddefaultv2";
}

std::vector<VerticalFluxTemplateDescriptor> list_vertical_flux_templates()
{
    const ComputeBackend* backend = active_compute_backend();
    const bool gpu_ready = backend != nullptr && backend->supports_vertical_flux_dispatch();
    return {
        {"tvd_vertical_flux_v2",
         "Float-precision TVD vertical flux (default). Same algorithm as v1 but "
         "operates in float for compiler auto-vectorization across columns.",
         gpu_ready},
        {"tvd_vertical_flux_v1",
         "Double-precision TVD vertical flux (reference). Higher precision, lower throughput.",
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
    const bool is_v1 = (id == "tvdverticalfluxv1" || id == "tvddefaultv1");
    const bool is_v2 = (id == "tvdverticalfluxv2" || id == "tvddefaultv2");
    if (!is_v1 && !is_v2)
    {
        return false;
    }

    // Try GPU dispatch if backend supports it (shared by v1 and v2)
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
        // GPU dispatch failed -- fall through to CPU path
    }

    // CPU path: v2 (float, auto-vectorizable) or v1 (double, reference)
    if (is_v2)
    {
        return dispatch_tvd_vertical_flux_v2(cfg, state, tendencies, diag_opt);
    }
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
    const float* loading_data,
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
            loading_data,
            du_r_dt_data, du_theta_dt_data, du_z_dt_data,
            drho_dt_data, dp_dt_data,
            nr, nth, nz,
            dr, dtheta, dz,
            g, gamma_val, theta0);
    }
    return false;
}

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
    float g, float gamma_val, float coriolis_f_val)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_cartesian_tendencies_dispatch())
    {
        return backend->dispatch_cartesian_tendencies(
            u_x_data, u_y_data, w_data,
            rho_data, p_data, theta_data,
            p0_base_data, rho0_base_data,
            loading_data,
            u0_base_data, v0_base_data,
            du_x_dt_data, du_y_dt_data, dw_dt_data,
            drho_dt_data, dp_dt_data,
            nr, nth, nz,
            dx, dy, dz,
            g, gamma_val, coriolis_f_val);
    }
    return false;
}

bool dispatch_advection_x_backend(
    const float* src, const float* u_data, float* dst,
    int nr, int nth, int nz,
    float dx, float dt)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_advection_x_dispatch())
    {
        return backend->dispatch_advection_x(src, u_data, dst, nr, nth, nz, dx, dt);
    }
    return false;
}

bool dispatch_advection_y_backend(
    const float* src, const float* v_data, float* dst,
    int nr, int nth, int nz,
    float dy, float dt)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_advection_y_dispatch())
    {
        return backend->dispatch_advection_y(src, v_data, dst, nr, nth, nz, dy, dt);
    }
    return false;
}

bool dispatch_tornado_tendencies_backend(
    const float* u_r_data, const float* u_theta_data, const float* u_z_data,
    const float* rho_data, const float* p_data, const float* theta_data,
    const float* loading_data,
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
            loading_data,
            du_r_dt_data, du_theta_dt_data, du_z_dt_data,
            drho_dt_data, dp_dt_data,
            nr, nth, nz,
            dr, dz,
            g, theta0, eps, friction_coeff);
    }
    return false;
}

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
    float Lv_cp, float Lf_cp, float Ls_cp, float T0)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_kessler_pointwise_dispatch())
    {
        return backend->dispatch_kessler_pointwise(
            temperature_data, p_data,
            qv_data, qc_data, qr_data, qg_data, qh_data,
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
    float ccn_conc, float in_conc)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_thompson_pointwise_dispatch())
    {
        return backend->dispatch_thompson_pointwise(
            temperature_data, p_data,
            qv_data, qc_data, qr_data, qi_data, qs_data,
            qg_data, qh_data,
            dtheta_dt_data, dqv_dt_data,
            dqc_dt_data, dqr_dt_data,
            dqi_dt_data, dqs_dt_data,
            dqg_dt_data, dqh_dt_data,
            nr, nth, nz,
            qc0, c_auto, c_evap,
            c_dep, c_subl, c_melt,
            Lv_cp, Lf_cp, Ls_cp, T0,
            ccn_conc, in_conc);
    }
    return false;
}

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
    float a_hail, float b_hail, float Vt_max_hail)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_thompson_sedimentation_dispatch())
    {
        return backend->dispatch_thompson_sedimentation(
            qr_data, qs_data, qg_data, qh_data,
            dqr_dt_data, dqs_dt_data, dqg_dt_data, dqh_dt_data,
            nr, nth, nz, dz_val,
            a_rain, b_rain, Vt_max_rain,
            a_snow, b_snow, Vt_max_snow,
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
    const float* u_data, const float* v_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dt_half)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_batched_advection_dispatch())
    {
        return backend->dispatch_advection_batch_pre_vertical(
            scalar_in, result_out,
            u_data, v_data,
            nr, nth, nz,
            dr, dtheta, dt_half);
    }
    return false;
}

bool dispatch_advection_batch_post_vertical_backend(
    const float* scalar_in, float* result_out,
    const float* u_data, const float* v_data,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_half, float dt_full, float kappa)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_batched_advection_dispatch())
    {
        return backend->dispatch_advection_batch_post_vertical(
            scalar_in, result_out,
            u_data, v_data,
            nr, nth, nz,
            dr, dtheta, dz,
            dt_half, dt_full, kappa);
    }
    return false;
}

bool dispatch_acoustic_pressure_backend(
    const float* u_data, const float* v_data, const float* w_data,
    const float* rho_in, const float* p_in,
    float* rho_out, float* p_out,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small,
    float rho_floor, float p_floor)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_acoustic_pressure_dispatch())
    {
        return backend->dispatch_acoustic_pressure(
            u_data, v_data, w_data, rho_in, p_in,
            rho_out, p_out, nr, nth, nz,
            dr, dtheta, dz, gamma_val, dt_small,
            rho_floor, p_floor);
    }
    return false;
}

bool dispatch_acoustic_momentum_backend(
    const float* rho_data, const float* p_data,
    const float* u_in, const float* v_in, const float* w_in,
    float* u_out, float* v_out, float* w_out,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float dt_small,
    float wind_clamp_h, float wind_clamp_v)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_acoustic_momentum_dispatch())
    {
        return backend->dispatch_acoustic_momentum(
            rho_data, p_data, u_in, v_in, w_in,
            u_out, v_out, w_out, nr, nth, nz,
            dr, dtheta, dz, dt_small,
            wind_clamp_h, wind_clamp_v);
    }
    return false;
}

bool dispatch_acoustic_substep_fused_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_acoustic_substep_fused_dispatch())
    {
        return backend->dispatch_acoustic_substep_fused(
            u, v, w, rho, p, nr, nth, nz,
            dr, dtheta, dz, gamma_val, dt_small,
            rho_floor, p_floor, wind_clamp_h, wind_clamp_v);
    }
    return false;
}

bool dispatch_acoustic_substeps_batched_backend(
    float* u, float* v, float* w,
    float* rho, float* p,
    int nr, int nth, int nz,
    float dr, float dtheta, float dz,
    float gamma_val, float dt_small, int n_substeps,
    float rho_floor, float p_floor,
    float wind_clamp_h, float wind_clamp_v)
{
    ComputeBackend* backend = mutable_compute_backend();
    if (backend != nullptr && backend->supports_acoustic_substeps_batched_dispatch())
    {
        return backend->dispatch_acoustic_substeps_batched(
            u, v, w, rho, p, nr, nth, nz,
            dr, dtheta, dz, gamma_val, dt_small, n_substeps,
            rho_floor, p_floor, wind_clamp_h, wind_clamp_v);
    }
    return false;
}

