#include "split_explicit.hpp"
#include "dynamics/dynamics_base.hpp"
#include "util/log.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

extern double& dz;
extern double& dr;
extern double& dtheta;
extern int& NR;
extern int& NTH;

namespace tmv
{

SplitExplicitScheme::SplitExplicitScheme() = default;

void SplitExplicitScheme::initialize()
{
    initialize(TimeSteppingConfig{}, nullptr);
}

void SplitExplicitScheme::initialize(const TimeSteppingConfig& cfg, RHSFunction /*rhs_func*/)
{
    config_ = cfg;
    last_warned_requested_substeps_ = -1;
    last_warned_effective_substeps_ = -1;
    tmv::log_info("Initialized split-explicit (Klemp-Wilhelmson 1978) time stepping scheme");
    tmv::log_info("  Acoustic substeps: ",
                  cfg.n_acoustic_substeps > 0
                      ? std::to_string(cfg.n_acoustic_substeps) + " minimum"
                      : "auto");
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

void SplitExplicitScheme::invalidate_generation_state_after_domain_recycling() noexcept
{
    du_fast_buf_.fill(0.0f);
    dv_fast_buf_.fill(0.0f);
    dw_fast_buf_.fill(0.0f);
    drho_fast_buf_.fill(0.0f);
    dp_fast_buf_.fill(0.0f);
}

int SplitExplicitScheme::resolve_substep_count(const TimeSteppingConfig& cfg, double dt_large)
{
    if (!std::isfinite(dt_large) || dt_large <= 0.0)
    {
        throw std::invalid_argument(
            "split-explicit stepping requires a positive finite large timestep");
    }

    if (!std::isfinite(cfg.max_acoustic_speed_ms) ||
        cfg.max_acoustic_speed_ms <= 0.0)
    {
        throw std::runtime_error(
            "split-explicit acoustic CFL cannot be evaluated because the "
            "runtime maximum sound speed is unavailable");
    }
    if (!std::isfinite(cfg.min_acoustic_spacing_m) ||
        cfg.min_acoustic_spacing_m < 0.0)
    {
        throw std::runtime_error(
            "split-explicit acoustic CFL received an invalid physical spacing");
    }

    // Compute N_required = ceil(dt * c_max / (C_target * dx_min)) + 1,
    // where dx_min is the
    // smallest acoustic grid spacing the substep needs to resolve.
    //
    // Take the min across both horizontal spacings (dr = dx, dtheta = dy on
    // Cartesian grids) and the vertical spacing (dz) so the acoustic CFL stays
    // bounded everywhere the acoustic mode propagates.
    //
    // A target of 0.5 also leaves enough margin for the combined 3-D wave
    // number (sqrt(3) * 0.5 < 1 on an isotropic grid).  Unlike the old fixed
    // 350 m/s estimate, c_max is sampled from the current model state.
    constexpr double cfl_target = 0.5;

    const double dz_safe = (std::isfinite(::dz) && ::dz > 0.0) ? ::dz : 500.0;
    double dx_min = (std::isfinite(cfg.min_acoustic_spacing_m) &&
                     cfg.min_acoustic_spacing_m > 0.0)
        ? cfg.min_acoustic_spacing_m : dz_safe;

    // A positive diagnosed spacing already includes Cartesian horizontal
    // distances and terrain face-area/volume geometry. The fallback retains
    // the direct-unit-test and pre-diagnosis Cartesian contract.
    if (!(std::isfinite(cfg.min_acoustic_spacing_m) &&
          cfg.min_acoustic_spacing_m > 0.0) && ::NR > 1)
    {
        const double dr_safe = (std::isfinite(::dr) && ::dr > 0.0) ? ::dr : dx_min;
        dx_min = std::min(dx_min, dr_safe);
    }

    if (!(std::isfinite(cfg.min_acoustic_spacing_m) &&
          cfg.min_acoustic_spacing_m > 0.0) && ::NTH > 1)
    {
        const double dy_safe = (std::isfinite(::dtheta) && ::dtheta > 0.0)
                                   ? ::dtheta
                                   : dx_min;
        dx_min = std::min(dx_min, dy_safe);
    }

    if (!(dx_min > 0.0)) dx_min = dz_safe;
    const double required_real =
        dt_large * cfg.max_acoustic_speed_ms / (cfl_target * dx_min);
    if (!std::isfinite(required_real) ||
        required_real >= static_cast<double>(std::numeric_limits<int>::max() - 1))
    {
        throw std::runtime_error(
            "split-explicit acoustic CFL requires an unrepresentable number "
            "of substeps; check dt, grid spacing, and thermodynamic state");
    }

    const int required = std::max(
        static_cast<int>(std::ceil(required_real)) + 1, 2);
    const int requested = cfg.n_acoustic_substeps;
    const int effective = requested > 0 ? std::max(requested, required) : required;

    if (requested > 0 && effective > requested &&
        (last_warned_requested_substeps_ != requested ||
         last_warned_effective_substeps_ != effective))
    {
        tmv::log_warn(
            "Configured acoustic_substeps=", requested,
            " under-resolves the runtime acoustic CFL (c_max=",
            cfg.max_acoustic_speed_ms, " m/s); promoting to ", effective,
            ". Positive acoustic_substeps is a requested minimum.");
        last_warned_requested_substeps_ = requested;
        last_warned_effective_substeps_ = effective;
    }

    return effective;
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

    // Step 2b: Seed any time-integrated open-boundary state (KW78 radiative
    // faces) from the POST-slow-tendency fields, so the radiation's "previous
    // face value" reflects this step's slow update. Must run after slow
    // tendencies and before the first acoustic substep (batched or per-substep),
    // matching the GPU batch-start seed.
    if (cb.seed_acoustic_radiation)
        cb.seed_acoustic_radiation();

    // Step 3: Acoustic substep loop (forward-backward). Backends may retain
    // immutable inputs across the observable per-substep submissions. Close
    // that lifetime on both success and failure so later physics can reuse
    // the same allocation high-water mark.
    bool batched = false;
    try
    {
        if (cb.begin_acoustic_loop_residency)
            cb.begin_acoustic_loop_residency();
        // Try a backend-batched path first (all N substeps in one dispatch).
        batched = cb.apply_fast_batched &&
                  cb.apply_fast_batched(dt_small, N);

        if (!batched)
        {
            for (int n = 0; n < N; ++n)
            {
                // Try a fused backend path (one CPU team or one GPU command buffer).
                const bool fused =
                    cb.apply_fast_fused && cb.apply_fast_fused(dt_small);

                if (!fused)
                {
                    // 3a. FORWARD: compute divergence -> update p, rho.
                    cb.apply_fast_pressure(dt_small);

                    // Pressure/rho halos needed by a centered backward operator
                    // must be moved to the new forward time level before that
                    // operator reads them. This is a zero-time closure; the
                    // time-dependent boundary advance remains exactly once below.
                    if (cb.close_fast_pressure_boundaries)
                        cb.close_fast_pressure_boundaries();

                    // 3b. BACKWARD: compute pressure gradient using NEW p, rho -> update u, v, w.
                    cb.apply_fast_momentum(dt_small);
                }

                // 3c. Lightweight BCs on acoustic fields.
                cb.acoustic_bcs(dt_small);
            }
        }

        // Final BCs after batched path (GPU handles internal BCs but CPU
        // applies once at the end for full consistency). dt = 0: this is a
        // ghost/tangential refresh only - the GPU already integrated the
        // KW78 radiative faces through all N substeps in-shader, and the
        // radiation update is NOT idempotent (a nonzero dt here would
        // advance the faces one extra substep from a stale stash every
        // batched step).
        if (batched && !cb.batched_acoustic_boundary_complete)
        {
            cb.acoustic_bcs(0.0);
        }
    }
    catch (...)
    {
        if (cb.release_acoustic_loop_residency)
            cb.release_acoustic_loop_residency();
        throw;
    }
    if (cb.release_acoustic_loop_residency)
        cb.release_acoustic_loop_residency();
}

} // namespace tmv
