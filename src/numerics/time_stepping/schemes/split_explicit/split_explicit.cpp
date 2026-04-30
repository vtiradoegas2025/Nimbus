/**
 * @file split_explicit.cpp
 * @brief Implementation of split-explicit acoustic time stepping.
 *
 * Implements the Klemp & Wilhelmson (1978) forward-backward acoustic
 * substep algorithm. The scheme receives field-specific callbacks from
 * the dynamics layer and orchestrates the timing and ordering.
 *
 * This file is part of the src/numerics subsystem.
 */

#include "split_explicit.hpp"
#include "dynamics/dynamics_base.hpp"
#include "util/log.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

SplitExplicitScheme::SplitExplicitScheme() = default;

void SplitExplicitScheme::initialize()
{
    initialize(TimeSteppingConfig{}, nullptr);
}

void SplitExplicitScheme::initialize(const TimeSteppingConfig& cfg, RHSFunction /*rhs_func*/)
{
    config_ = cfg;
    std::cout << "Initialized split-explicit (Klemp-Wilhelmson 1978) time stepping scheme" << std::endl;
    std::cout << "  Acoustic substeps: "
              << (cfg.n_acoustic_substeps > 0
                      ? std::to_string(cfg.n_acoustic_substeps)
                      : "auto")
              << std::endl;
}

void SplitExplicitScheme::step(
    const TimeSteppingConfig& /*cfg*/,
    TimeSteppingState& /*state*/,
    TimeSteppingDiagnostics* /*diag_opt*/)
{
    // The generic step() interface is not used for split-explicit.
    // The dynamics layer calls step_split_acoustic() directly.
    throw std::runtime_error(
        "SplitExplicitScheme::step() should not be called directly. "
        "Use step_split_acoustic() with callbacks from the dynamics layer.");
}

double SplitExplicitScheme::suggest_dt(
    const TimeSteppingConfig& /*cfg*/,
    const TimeSteppingState& /*state*/)
{
    // The split-explicit scheme does not constrain the large timestep
    // beyond the advective CFL, which is handled by the numerics layer.
    return 1.0;
}

int SplitExplicitScheme::resolve_substep_count(const TimeSteppingConfig& cfg, double dt_large)
{
    if (cfg.n_acoustic_substeps > 0)
        return cfg.n_acoustic_substeps;

    // Auto-compute: N = ceil(dt * c_s / dz) + 1.
    // Use a conservative c_s estimate of 350 m/s (sea-level standard atmosphere).
    constexpr double c_s_est = 350.0;
    extern double dz;
    const double min_dz = (std::isfinite(dz) && dz > 0.0) ? dz : 500.0;
    int N = static_cast<int>(std::ceil(dt_large * c_s_est / min_dz)) + 1;
    return std::max(N, 2);
}

void SplitExplicitScheme::ensure_buffers(int nr, int nth, int nz)
{
    if (buffers_allocated_
        && du_fast_buf_.size_r() == nr
        && du_fast_buf_.size_th() == nth
        && du_fast_buf_.size_z() == nz)
        return;

    du_fast_buf_.resize(nr, nth, nz, 0.0f);
    dv_fast_buf_.resize(nr, nth, nz, 0.0f);
    dw_fast_buf_.resize(nr, nth, nz, 0.0f);
    drho_fast_buf_.resize(nr, nth, nz, 0.0f);
    dp_fast_buf_.resize(nr, nth, nz, 0.0f);
    buffers_allocated_ = true;
}

void SplitExplicitScheme::step_split_acoustic(
    const TimeSteppingConfig& cfg,
    double dt_large,
    const SplitExplicitCallbacks& cb)
{
    const int N = resolve_substep_count(cfg, dt_large);
    const double dt_small = dt_large / N;

    // Step 1: Compute slow tendencies into caller-managed buffers.
    cb.compute_slow_tendencies();

    // Step 2: Apply slow tendencies with the large timestep.
    cb.apply_slow_tendencies(dt_large);

    // Step 3: Acoustic substep loop (forward-backward).
    // Try batched GPU path first (all N substeps in one submission).
    const bool batched = cb.apply_fast_batched &&
                         cb.apply_fast_batched(dt_small, N);

    if (!batched)
    {
        for (int n = 0; n < N; ++n)
        {
            // Try fused GPU path (pressure + momentum in one command buffer).
            const bool fused = cb.apply_fast_fused && cb.apply_fast_fused(dt_small);

            if (!fused)
            {
                // 3a. FORWARD: compute divergence -> update p, rho.
                cb.apply_fast_pressure(dt_small);

                // 3b. BACKWARD: compute pressure gradient using NEW p, rho -> update u, v, w.
                cb.apply_fast_momentum(dt_small);
            }

            // 3c. Lightweight BCs on acoustic fields.
            cb.acoustic_bcs();
        }
    }

    // Final BCs after batched path (GPU handles internal BCs but CPU
    // applies once at the end for full consistency).
    if (batched)
    {
        cb.acoustic_bcs();
    }
}
