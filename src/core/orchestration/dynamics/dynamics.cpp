#include "microphysics/grid_geometry.hpp"
#include "diagnostics/resolved_diagnostics.hpp"
#include "core/infra/nested_scalar_feedback.hpp"
#include "dynamics/native_cgrid_boundary.hpp"
#include "microphysics/prognostic_transport.hpp"
// Cross-module dynamics sequencing and timestep coupling.

#include "boundary_conditions/factory.hpp"
#include "boundary_conditions/runtime.hpp"
#include "chaos/runtime.hpp"
#include "core/field/field_bounds.hpp"
#include "core/orchestration/dynamics/sgs_apply.hpp"
#include "core/orchestration/dynamics/sgs_apply_internal.hpp"
#include "core/orchestration/dynamics/pressure_cell_energy.hpp"
#include "core/orchestration/physics/diffusion_step.hpp"
#include "core/field/field_sanitization.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/rayleigh_damping.hpp"
#ifdef NIMBUS_ALLOCATION_TELEMETRY
#include "core/infra/allocation_telemetry.hpp"
#endif
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/flat_native_tvd_step_transaction.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/level_budget_csv.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "dynamics/collocated_velocity_constraint.hpp"
#include "dynamics/device_acoustic_energy_attribution.hpp"
#include "dynamics/device_slow_energy_attribution.hpp"
#include "dynamics/dynamics_base.hpp"
#include "dynamics/eos_pressure.hpp"
#include "microphysics/runtime.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/runtime.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "numerics/time_stepping/rk_combination.hpp"
#include "turbulence/runtime.hpp"
#include "dynamics/runtime.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "core/field/field_pool.hpp"
#include "radar/runtime.hpp"
#include "terrain/runtime.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/native_mapped_transaction.hpp"
#include "util/parallel_copy.hpp"
#include "boundary_layer/runtime.hpp"
#include <cstring>

#include "diagnostics/field_contract.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <type_traits>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
struct DynamicsPerfTotals
{
    double total_s                  = 0.0;
    double core_s                   = 0.0;
    double acoustic_energy_snapshot_s = 0.0;
    double acoustic_state_copy_s    = 0.0;
    double acoustic_transport_copy_s = 0.0;
    double acoustic_transport_flux_s = 0.0;
    double acoustic_dispatch_s = 0.0;
    double acoustic_attribution_s = 0.0;
    double acoustic_initial_validation_s = 0.0;
    double acoustic_boundary_s = 0.0;
    double slow_compute_s = 0.0;
    double slow_apply_s = 0.0;
    double advection_s              = 0.0;
    double microphysics_transport_s = 0.0;
    double microphysics_s           = 0.0;
    double turbulence_s             = 0.0;
    double turbulence_refresh_s     = 0.0;
    double turbulence_chaos_s       = 0.0;
    double turbulence_stability_s   = 0.0;
    double turbulence_apply_s       = 0.0;
    double turbulence_commit_s      = 0.0;
    double diffusion_s              = 0.0;
    double eos_s                    = 0.0;
    double sponge_s                 = 0.0;
    double final_boundary_s         = 0.0;
    double science_conservation_s   = 0.0;
    double science_momentum_s       = 0.0;
    double science_energy_s         = 0.0;
    double science_moisture_s       = 0.0;
};

DynamicsPerfTotals dynamics_perf_totals;
using DynamicsPerfClock = std::chrono::steady_clock;

DynamicsPerfClock::time_point dynamics_perf_start()
{
    return global_perf_timing_enabled ? DynamicsPerfClock::now() : DynamicsPerfClock::time_point{};
}

void dynamics_perf_finish(double& accumulator, DynamicsPerfClock::time_point started)
{
    if (global_perf_timing_enabled)
    {
        accumulator += std::chrono::duration<double>(DynamicsPerfClock::now() - started).count();
    }
}

ConservationBudget measured_conservation_budget(const SimulationStateView& state, bool enabled)
{
    if (!enabled)
        return {};
    const auto         started = dynamics_perf_start();
    ConservationBudget result  = compute_conservation_budget(state);
    dynamics_perf_finish(dynamics_perf_totals.science_conservation_s, started);
    return result;
}

MomentumBudget measured_momentum_budget(const SimulationStateView& state, bool enabled)
{
    if (!enabled)
        return {};
    const auto     started = dynamics_perf_start();
    MomentumBudget result  = compute_momentum_budget(state);
    dynamics_perf_finish(dynamics_perf_totals.science_momentum_s, started);
    return result;
}

EnergyDiagnosticSnapshotV1 measured_energy_snapshot(const SimulationStateView& state, bool enabled)
{
    if (!enabled)
        return {};
    const auto                 started = dynamics_perf_start();
    EnergyDiagnosticSnapshotV1 result  = compute_energy_diagnostic_snapshot_v1(state);
    dynamics_perf_finish(dynamics_perf_totals.science_energy_s, started);
    return result;
}

MoistMixtureEnergyBudgetV2 measured_acoustic_energy_snapshot(const SimulationStateView& state)
{
    const auto started = dynamics_perf_start();
    const auto result = compute_moist_mixture_energy_v2_deterministic(state);
    dynamics_perf_finish(dynamics_perf_totals.acoustic_energy_snapshot_s, started);
    return result;
}

EnergyDiagnosticSnapshotV1 measured_acoustic_complete_energy_snapshot(const SimulationStateView& state)
{
    const auto started = dynamics_perf_start();
    const auto result = compute_energy_diagnostic_snapshot_v1_deterministic(state);
    dynamics_perf_finish(dynamics_perf_totals.acoustic_energy_snapshot_s, started);
    return result;
}

double measured_acoustic_velocity_energy_completion(
    const SimulationStateView& state, const EnergyDiagnosticSnapshotV1& before)
{
    const auto started = dynamics_perf_start();
    const auto result = compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(state, before);
    dynamics_perf_finish(dynamics_perf_totals.acoustic_energy_snapshot_s, started);
    return result;
}

std::array<double, 2> measured_acoustic_velocity_energy_pair(
    const SimulationStateView& state, Field3D& before_u, Field3D& before_v, Field3D& before_w)
{
    const auto started = dynamics_perf_start();
    const auto result = compute_flat_cgrid_velocity_energy_pair_deterministic(state, before_u, before_v, before_w);
    dynamics_perf_finish(dynamics_perf_totals.acoustic_energy_snapshot_s, started);
    return result;
}

struct ConservationMoistureSnapshots
{
    ConservationBudget conservation;
    MoistureBudget     moisture;
};

ConservationMoistureSnapshots measured_conservation_moisture_snapshots(const SimulationStateView& state, bool enabled)
{
    if (!enabled)
        return {};
    const auto                    started = dynamics_perf_start();
    ConservationMoistureSnapshots result;
    compute_conservation_and_moisture_budgets(state, result.conservation, result.moisture);
    // The fused traversal is charged to conservation; the moisture counter
    // now represents only standalone moisture passes (none in the ordinary
    // completed-step path). Total profiled ownership remains additive.
    dynamics_perf_finish(dynamics_perf_totals.science_conservation_s, started);
    return result;
}

#ifdef NIMBUS_SEPARATE_ENDPOINT_MOISTURE_SCANS
MoistureBudget measured_moisture_budget(const SimulationStateView& state, bool enabled)
{
    if (!enabled)
        return {};
    const auto     started = dynamics_perf_start();
    MoistureBudget result  = compute_moisture_budget(state);
    dynamics_perf_finish(dynamics_perf_totals.science_moisture_s, started);
    return result;
}
#endif

void invalidate_named_fields(const SimulationStateView& execution_state, std::initializer_list<const char*> fields)
{
    for (const char* field : fields)
        invalidate_resident_field_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"), field);
}

void invalidate_thermodynamic_residency(const SimulationStateView& execution_state)
{
    invalidate_named_fields(execution_state, {"rho", "p", "theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"});
}

void invalidate_boundary_field_residency(const SimulationStateView& execution_state, const tmv::BoundaryFieldWriteSet& writes) noexcept
{
    if (!writes.complete || (writes.count != 0 && writes.field_ids == nullptr))
    {
        invalidate_all_resident_fields_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"));
        return;
    }
    for (std::size_t i = 0; i < writes.count; ++i)
    {
        if (writes.field_ids[i] == nullptr || writes.field_ids[i][0] == '\0')
        {
            invalidate_all_resident_fields_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"));
            return;
        }
    }
    for (std::size_t i = 0; i < writes.count; ++i)
        invalidate_resident_field_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"), writes.field_ids[i]);
}

inline bool field_matches_domain(const Field3D& f, const GridStateView& grid)
{
    return f.size_r() == grid.nr && f.size_th() == grid.nth && f.size_z() == grid.nz;
}

tmv::DynamicsRuntimeState& dynamics_runtime_for(const SimulationStateView& state) noexcept
{
    return state.dynamics_runtime != nullptr ? *state.dynamics_runtime : tmv::require_level_runtime(state.dynamics_runtime, "dynamics_runtime");
}

void ensure_dynamics_tendency_buffers(tmv::DynamicsRuntimeState& runtime, const GridStateView& grid)
{
    for (Field3D* field : {&runtime.du_dt, &runtime.dv_dt, &runtime.dw_dt, &runtime.drho_dt, &runtime.dp_dt})
        if (!field_matches_domain(*field, grid))
            field->resize(grid.nr, grid.nth, grid.nz, 0.0f);
}

void ensure_slow_advection_tendency_buffers(tmv::DynamicsRuntimeState& runtime, const GridStateView& grid)
{
    for (Field3D* field : {&runtime.du_advection_dt, &runtime.dv_advection_dt, &runtime.dw_advection_dt})
        if (!field_matches_domain(*field, grid))
            field->resize(grid.nr, grid.nth, grid.nz, 0.0f);
}

struct AcousticProfileRows
{
    float* rho0   = nullptr;
    float* p0     = nullptr;
    float* dp0_dz = nullptr;
    float* dp0_dz_low = nullptr;
};

AcousticProfileRows acoustic_profile_rows(const SimulationStateView& state, int nz)
{
    if (nz <= 0)
        throw std::invalid_argument("acoustic compact profiles require a positive vertical extent");
    const std::size_t row_size = static_cast<std::size_t>(nz);
    auto&             storage  = dynamics_runtime_for(state).acoustic_profile_rows;
    storage.resize(tmv::DynamicsExecutionWorkspace::acoustic_profile_row_count * row_size);
    return {
        storage.data(),
        storage.data() + row_size,
        storage.data() + 2 * row_size,
        storage.data() + 3 * row_size,
    };
}

void fill_acoustic_base_profiles(const std::vector<double>& rho0_base,
                                 const std::vector<double>& p0_base,
                                 int                        nz,
                                 double                     dz,
                                 AcousticProfileRows        rows,
                                 bool                       fill_rho0,
                                 bool                       fill_p0,
                                 bool                       fill_dp0_dz)
{
    if ((fill_rho0 && static_cast<int>(rho0_base.size()) < nz) ||
        ((fill_p0 || fill_dp0_dz) && static_cast<int>(p0_base.size()) < nz) || !(dz > 0.0) || !std::isfinite(dz))
    {
        throw std::invalid_argument("acoustic compact profiles require complete base state and dz");
    }
    for (int k = 0; k < nz; ++k)
    {
        const std::size_t index = static_cast<std::size_t>(k);
        if (fill_rho0)
            rows.rho0[index] = static_cast<float>(rho0_base[index]);
        if (fill_p0)
            rows.p0[index] = static_cast<float>(p0_base[index]);
        if (fill_dp0_dz)
        {
            const double gradient = k + 1 < nz ? (p0_base[index + 1] - p0_base[index]) / dz : 0.0;
            rows.dp0_dz[index] = static_cast<float>(gradient);
            rows.dp0_dz_low[index] = static_cast<float>(gradient - static_cast<double>(rows.dp0_dz[index]));
        }
    }
}

bool has_compute_context(const SimulationStateView& state)
{
    return state.compute_runtime != nullptr && state.configuration != nullptr;
}

} // namespace

// Select the dynamics kernel before any model state is advanced.
void tmv::initialize_dynamics(const std::string& scheme_name)
{
    return initialize_dynamics(global_simulation_state(), scheme_name);
}

void tmv::initialize_dynamics(const SimulationStateView& execution_state, const std::string& scheme_name)
{
    try
    {
        tmv::require_level_runtime(execution_state.dynamics_runtime, "dynamics runtime").scheme =
            tmv::create_dynamics_scheme(scheme_name, execution_state);
        const std::string active_scheme_name = tmv::require_level_runtime(execution_state.dynamics_runtime, "dynamics_runtime").scheme->get_scheme_name();

        // Boundary-condition scheme matched to (coordinate, stagger).
        // Dispatch lives in src/boundary_conditions/factory.cpp.
        tmv::boundary_scheme_for(execution_state) = create_boundary_condition_scheme(execution_state);
        tmv::log_info("Initialized BC scheme: ", tmv::boundary_scheme_for(execution_state)->get_scheme_name());

        // Validate split-explicit compatibility BEFORE any time stepping.
        // The orchestrator's now-removed legacy fallback (step_dynamics_old)
        // silently ran unsplit Forward Euler when split_acoustic=true and the
        // scheme didn't implement SplitExplicitDynamics. Forward Euler is
        // unconditionally unstable for hyperbolic acoustic equations, so that
        // fallback ran broken physics with no diagnostic. That path is gone;
        // detect the mismatch here and abort with a clear message.
        if (tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime").config.split_acoustic && tmv::require_level_runtime(execution_state.dynamics_runtime, "dynamics_runtime").scheme)
        {
            auto* split = dynamic_cast<SplitExplicitDynamics*>(tmv::require_level_runtime(execution_state.dynamics_runtime, "dynamics_runtime").scheme.get());
            if (split == nullptr)
            {
                throw std::runtime_error("dynamics.scheme=" + active_scheme_name +
                                         " does not implement SplitExplicitDynamics, but "
                                         "numerics.time_stepping.split_acoustic=true. The previous "
                                         "behavior silently ran unsplit Forward Euler, which is "
                                         "unconditionally unstable for acoustic waves. "
                                         "Pick a scheme that supports split-explicit acoustics "
                                         "(e.g. klemp_wilhelmson_split_cartesian_cgrid, "
                                         "klemp_wilhelmson_split_cartesian_agrid).");
            }
        }

        // Diagnostic fields (vorticity, angular momentum, pressure decomposition)
        // are allocated on first use in compute_dynamics_diagnostics().
        const auto state = execution_state;
        ensure_dynamics_tendency_buffers(dynamics_runtime_for(state), state.grid);
    }
    catch (const std::runtime_error& e)
    {
        tmv::log_error("Error initializing dynamics scheme: ", e.what());
        throw;
    }
}

namespace
{

/// RK2 (Heun's method) on the split-explicit slow path - the DEFAULT
/// since 2026-06-11 (SCIENCE.md C2):
///
///   k1   = f_slow(u_n)
///   u'   = u_n + dt * k1                  (Euler predictor)
///   k2   = f_slow(u')
///   u^{n+1}_slow = u_n + (dt/2)(k1 + k2)  (trapezoidal corrector)
///
/// Acoustic substeps then run on u^{n+1}_slow as before. Forward Euler
/// with the 5th-order upwind stencils is linearly UNSTABLE at every
/// Courant number (von Neumann |g| up to 1.029 at the beast operating
/// point; Wicker-Skamarock 2002 pair these stencils with RK for this
/// reason) - it merely hides below dissipation at modest winds and
/// behind the wind clamps above them. Heun+UP5 is stable to C ~ 0.4
/// and is 2nd-order in time (the MMS-documented deficiency). Cost:
/// one extra slow-tendency evaluation per step (~10% at beast size).
/// extensions.solver.slow_rk2: false restores Forward Euler for A/B comparison
/// only (modularity cleanup 6a; was the NIMBUS_SLOW_RK2=0 env knob).
inline bool slow_rk2_enabled(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).slow_rk2;
}

} // namespace

// Build field-aware callbacks, then let the time-stepping scheme own the
// split-explicit algorithm. Optional Heun integration upgrades only the slow
// tendencies; acoustic substeps retain forward-backward integration.
// Optional device commit for the slow tendencies. The shared interaction
// policy owns scheme/geometry/guard admission; this runtime switch only
// expresses the request. It remains default-off because no qualified public
// profile currently promotes slow device commit.
static bool gpu_slow_apply_enabled(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).gpu_apply_slow;
}

// GPU path for the end-of-step sponge sequence. EOS is deliberately evaluated
// first by the shared FP64 CPU projection: the shader's FP32 pow path is the
// same candidate rejected by CPU-028 at full-solver scope and cannot define a
// different backend equation. SCI-161 observes the actual device endpoint
// before host copy-back and replays only per-cell FP32 pass intermediates for
// attribution; the retained physical parity registry owns trajectory sampling.
// Returns false (state untouched) when the GPU path is unavailable.
struct DeviceStateRelaxationAttributionContext
{
    const SimulationStateView* state         = nullptr;
    const float*               profiles      = nullptr;
    int                        profiles_len  = 0;
    int                        lateral_width = 0;
    bool                       do_upper      = false;
    bool                       do_w_only     = false;
    bool                       do_lateral    = false;
    float                      dt_step       = 0.0f;
    tmv::SpongeEnergyReportV1* report        = nullptr;
};

static void attribute_device_state_relaxation_completed(void*        raw_context,
                                                        const float* u_committed,
                                                        const float* v_committed,
                                                        const float* w_committed,
                                                        const float* theta_committed)
{
    auto& context   = *static_cast<DeviceStateRelaxationAttributionContext*>(raw_context);
    *context.report = attribute_state_relaxation_cartesian_device_commit(*context.state,
                                                                         context.dt_step,
                                                                         context.profiles,
                                                                         context.profiles_len,
                                                                         context.lateral_width,
                                                                         context.do_upper,
                                                                         context.do_w_only,
                                                                         context.do_lateral,
                                                                         u_committed,
                                                                         v_committed,
                                                                         w_committed,
                                                                         theta_committed);
}

static bool
try_state_relaxation_gpu(const SimulationStateView& state, double dt_dynamics, tmv::SpongeEnergyReportV1* energy_report)
{
    const auto damping_mode = state.configuration ? state.configuration->damping_mode : tmv::require_run_configuration(state).damping_mode;
    if (damping_mode == tmv::init::config::DampingMode::Implicit)
    {
        // The compact device ABI hard-codes exact exponential relaxation.
        // Backward Euler therefore remains on the attributed CPU writer until
        // the mode is represented explicitly in that ABI.
        return false;
    }
    if (!primary_state_repairs_enabled)
    {
        // Warn/strict validation must observe the unmodified candidate.  The
        // current device writer always bounds its result, so use the
        // attributed CPU writer until the shader ABI carries that policy.
        return false;
    }
    if (!active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), DynamicsKernelOp::StateRelaxationCartesian))
    {
        return false;
    }
    if (tmv::physical_grid_metrics::active_terrain_metrics(state) != nullptr)
    {
        // Mapped sponge coefficients vary by column and staggered physical
        // height; the compact level-profile shader is a flat-grid kernel.
        return false;
    }

    const auto& grid   = state.grid;
    auto&       fields = state.fields;
    const auto& base   = state.base;

    const bool do_upper   = rayleigh_upper_sponge_enabled();
    const bool do_w_only  = do_upper && rayleigh_w_only_sponge_enabled();
    const bool do_lateral = rayleigh_lateral_sponge_enabled();
    if (!do_upper && !do_w_only && !do_lateral)
        return false;
    if (energy_report == nullptr || energy_report->complete)
        throw std::invalid_argument("device state relaxation requires a fresh energy report");

    const auto& alpha_prof         = rayleigh_alpha_profile(state);
    const auto& alpha_upper_w_prof = rayleigh_upper_w_alpha_profile(state);
    const auto& alpha_w_prof       = rayleigh_w_only_alpha_profile(state);
    const auto& u_ref_prof         = rayleigh_u_ref_profile(state);
    const auto& v_ref_prof         = rayleigh_v_ref_profile(state);
    const auto& lateral_prof       = rayleigh_lateral_alpha_profile(state);

    // Mirror the CPU's size guard (apply_rayleigh_damping returns without
    // acting when the profiles do not match NZ).
    if (do_upper &&
        (static_cast<int>(alpha_prof.size()) != grid.nz || static_cast<int>(alpha_upper_w_prof.size()) != grid.nz ||
         static_cast<int>(alpha_w_prof.size()) != grid.nz))
    {
        return false;
    }
    const int lateral_width = static_cast<int>(lateral_prof.size());
    if (lateral_width > grid.nz)
    {
        // Row 7 of the packed profile block holds the lateral alphas by
        // boundary distance; widths beyond NZ cannot be packed.
        return false;
    }

    // Packed 7-row profile block, row stride NZ (see the shader header).
    auto& profiles = dynamics_runtime_for(state).state_relaxation_profiles;
    profiles.assign(static_cast<std::size_t>(7 * grid.nz), 0.0f);
    for (int k = 0; k < grid.nz; ++k)
    {
        const std::size_t ks = static_cast<std::size_t>(k);
        if (ks < alpha_prof.size())
            profiles[static_cast<std::size_t>(0 * grid.nz + k)] = static_cast<float>(alpha_prof[ks]);
        if (ks < alpha_w_prof.size())
            profiles[static_cast<std::size_t>(1 * grid.nz + k)] = static_cast<float>(alpha_w_prof[ks]);
        profiles[static_cast<std::size_t>(2 * grid.nz + k)] =
            (ks < u_ref_prof.size()) ? static_cast<float>(u_ref_prof[ks]) : 0.0f;
        profiles[static_cast<std::size_t>(3 * grid.nz + k)] =
            (ks < v_ref_prof.size()) ? static_cast<float>(v_ref_prof[ks]) : 0.0f;
        // Reference fallback mirrors apply_rayleigh_damping's per-level
        // theta0 default.
        profiles[static_cast<std::size_t>(4 * grid.nz + k)] =
            (ks < base.theta0.size()) ? static_cast<float>(base.theta0[ks]) : static_cast<float>(theta0);
        if (ks < alpha_upper_w_prof.size())
            profiles[static_cast<std::size_t>(6 * grid.nz + k)] = static_cast<float>(alpha_upper_w_prof[ks]);
    }
    for (int d = 0; d < lateral_width; ++d)
    {
        profiles[static_cast<std::size_t>(5 * grid.nz + d)] =
            static_cast<float>(lateral_prof[static_cast<std::size_t>(d)]);
    }

    DeviceStateRelaxationAttributionContext attribution_context;
    attribution_context.state         = &state;
    attribution_context.profiles      = profiles.data();
    attribution_context.profiles_len  = 7 * grid.nz;
    attribution_context.lateral_width = lateral_width;
    attribution_context.do_upper      = do_upper;
    attribution_context.do_w_only     = do_w_only;
    attribution_context.do_lateral    = do_lateral;
    attribution_context.dt_step       = static_cast<float>(dt_dynamics);
    attribution_context.report        = energy_report;

    if (!dispatch_state_relaxation_cartesian_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), fields.u.data(),
                                                     fields.v.data(),
                                                     fields.w.data(),
                                                     fields.theta.data(),
                                                     profiles.data(),
                                                     7 * grid.nz,
                                                     grid.nr,
                                                     grid.nth,
                                                     grid.nz,
                                                     lateral_width,
                                                     do_upper,
                                                     do_w_only,
                                                     do_lateral,
                                                     static_cast<float>(dt_dynamics),
                                                     &attribute_device_state_relaxation_completed,
                                                     &attribution_context))
    {
        return false;
    }

    return true;
}

static void step_unsplit_dynamics(const SimulationStateView& state,
                                  double                     dt_dynamics,
                                  double                     current_time,
                                  Field3D&                   du_dt,
                                  Field3D&                   dv_dt,
                                  Field3D&                   dw_dt,
                                  Field3D&                   drho_dt,
                                  Field3D&                   dp_dt)
{
    auto&                      bc_scheme        = tmv::boundary_scheme_for(state);
    tmv::DynamicsRuntimeState& dynamics_runtime = dynamics_runtime_for(state);
    if (!dynamics_runtime.scheme)
        throw std::runtime_error("unsplit dynamics requires an initialized dynamics scheme");
    TimeSteppingRuntimeState& time_runtime =
        state.time_stepping_runtime ? *state.time_stepping_runtime
                                    : tmv::require_level_runtime(state.time_stepping_runtime, "time_stepping_runtime");
    if (!time_runtime.scheme)
    {
        throw std::runtime_error("unsplit dynamics requires an initialized time stepper");
    }
    const auto&            grid            = state.grid;
    auto&                  fields          = state.fields;
    auto&                  tendency_fields = state.tendencies;
    AdvectionRuntimeState& advection_runtime =
        state.advection_runtime ? *state.advection_runtime
                                : tmv::require_level_runtime(state.advection_runtime, "advection_runtime");
    const auto scalar_reconstruction    = advection_runtime.config.scheme_id == "tvd"
                                              ? tmv::CartesianFaceReconstruction::MonotonicCentered
                                              : tmv::CartesianFaceReconstruction::WENO5;
    const bool scalar_transport_enabled = advection_runtime.config.scheme_id != "none";

    auto&      workspace          = time_runtime.unsplit_rk;
    const bool native_combination = state.compute_runtime &&
                                    state.compute_runtime->active_kind != ComputeBackendKind::Cpu &&
                                    !state.compute_runtime->fallback_active;
    if (native_combination &&
        (!state.compute_runtime->backend || !state.compute_runtime->backend->supports_rk_combination() ||
         !state.compute_runtime->backend->supports_rk_tendency()))
        throw std::runtime_error("native RK stage combination is unavailable");
    if (native_combination && (state.configuration->lateral_bc_west != tmv::init::config::LateralBC::Periodic ||
                               state.configuration->lateral_bc_east != tmv::init::config::LateralBC::Periodic ||
                               state.configuration->lateral_bc_south != tmv::init::config::LateralBC::Periodic ||
                               state.configuration->lateral_bc_north != tmv::init::config::LateralBC::Periodic ||
                               state.configuration->bottom_bc != tmv::init::config::BottomBC::RigidSlip ||
                               state.configuration->top_bc != tmv::init::config::TopBC::Rigid ||
                               tmv::physical_grid_metrics::terrain_is_active(state) ||
                               state.domain_frame->velocity_ms.u_ms != 0 || state.domain_frame->velocity_ms.v_ms != 0))
        throw std::runtime_error(
            "native RK requires stationary flat Cartesian periodic sides and rigid-slip bottom/rigid lid");
    const bool        ssp_rk           = time_runtime.scheme->name() == "rk3";
    const std::size_t active_rk_stages = ssp_rk ? 3 : 4;
    auto&             stage_transport  = dynamics_runtime.transport_context;
    double            integration_dt   = dt_dynamics;
    auto&             boundary_runtime =
        state.boundary_condition_runtime
            ? *state.boundary_condition_runtime
            : tmv::require_level_runtime(state.boundary_condition_runtime, "boundary_condition_runtime");
    auto&                           boundary_waves = static_cast<tmv::BoundaryConditionWaveState&>(boundary_runtime);
    tmv::BoundaryConditionWaveState boundary_base;
    const PhysicalCellDomain        scalar_domain = physical_scalar_domain(grid);
    if (!bc_scheme)
        throw std::runtime_error("unsplit RK requires initialized boundary conditions");
    const std::array<Field3D*, tmv::TimeSteppingUnsplitRkWorkspace::field_count> live{{&fields.u,
                                                                                       &fields.v,
                                                                                       &fields.w,
                                                                                       &fields.rho,
                                                                                       &fields.p,
                                                                                       &fields.theta,
                                                                                       &fields.qv,
                                                                                       &fields.tracer,
                                                                                       &fields.qc,
                                                                                       &fields.qr,
                                                                                       &fields.qi,
                                                                                       &fields.qs,
                                                                                       &fields.qg,
                                                                                       &fields.qh,
                                                                                       &fields.tke}};
    const auto                                                                   resize_like_grid = [&](Field3D& field)
    {
        if (field.size_r() != grid.nr || field.size_th() != grid.nth || field.size_z() != grid.nz)
            field.resize(grid.nr, grid.nth, grid.nz, 0.0f);
    };
    const auto copy_values = [&](Field3D& destination, const Field3D& source)
    {
        if (destination.size_r() != source.size_r() || destination.size_th() != source.size_th() ||
            destination.size_z() != source.size_z())
            throw std::runtime_error("unsplit RK workspace shape mismatch");
        std::copy_n(source.data(), source.size(), destination.data());
    };
    MicrophysicsScheme* transport_microphysics =
        state.microphysics_runtime
            ? state.microphysics_runtime->scheme.get()
            : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime").scheme.get();
    tmv::MicrophysicsPrognosticScalarList private_scalars;
    if (transport_microphysics != nullptr)
    {
        tmv::prepare_microphysics_prognostics(state, *transport_microphysics);
        private_scalars = transport_microphysics->prognostic_scalars_for_transport();
    }
    if (private_scalars.size() > tmv::TimeSteppingUnsplitRkWorkspace::private_field_capacity)
        throw std::length_error("unsplit RK private prognostic field capacity exceeded");
    for (const auto& scalar : private_scalars)
    {
        tmv::validate_microphysics_scalar_storage(scalar.storage);
        if (scalar.name == nullptr || scalar.field == nullptr || scalar.field->size_r() != grid.nr ||
            scalar.field->size_th() != grid.nth || scalar.field->size_z() != grid.nz)
            throw std::invalid_argument("unsplit RK private prognostic binding is incomplete");
        for (std::size_t index = 0; index < scalar.field->size(); ++index)
            if (!std::isfinite(scalar.field->data()[index]) ||
                (scalar.positive_definite && scalar.field->data()[index] < 0.0f))
                throw std::invalid_argument(std::string("unsplit RK private prognostic state is invalid: ") +
                                            scalar.name);
    }

    RkTendencyDispatch native_dispatch;
    native_dispatch.nx         = grid.nr;
    native_dispatch.ny         = grid.nth;
    native_dispatch.nz         = grid.nz;
    native_dispatch.dx         = grid.dr;
    native_dispatch.dy         = grid.dtheta;
    native_dispatch.dz         = grid.dz;
    native_dispatch.periodic_x = state.configuration->lateral_bc_west == tmv::init::config::LateralBC::Periodic &&
                                 state.configuration->lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    native_dispatch.periodic_y = state.configuration->lateral_bc_south == tmv::init::config::LateralBC::Periodic &&
                                 state.configuration->lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    native_dispatch.open_top   = state.configuration->top_bc == tmv::init::config::TopBC::Open;
    native_dispatch.collocated = !grid.geometry.staggered;
    const auto close_stage_boundaries = [&]
    {
        if (native_combination)
        {
            auto                                            d       = native_dispatch;
            auto&                                           backend = *state.compute_runtime->backend;
            const std::array<const std::vector<double>*, 7> base_profiles{&state.base.u0,
                                                                          &state.base.v0,
                                                                          nullptr,
                                                                          &state.base.rho0,
                                                                          &state.base.p0,
                                                                          &state.base.theta0,
                                                                          &state.base.qv0};
            for (std::size_t f = 0; f < live.size(); ++f)
            {
                d.operation = f < 2
                                  ? RkTendencyOperation::HorizontalWindHalo
                                  : (f == 2 ? RkTendencyOperation::VerticalWindHalo : RkTendencyOperation::ScalarHalo);
                d.input     = {live[f]->data()};
                if (d.collocated && f < base_profiles.size() && base_profiles[f])
                {
                    const auto& profile = *base_profiles[f];
                    if (profile.size() != std::size_t(grid.nz))
                        throw std::invalid_argument("native collocated RK halo requires a complete base profile");
                    resize_like_grid(workspace.native_base_profile);
                    for (std::size_t n = 0; n < live[f]->size(); ++n)
                        workspace.native_base_profile.data()[n] = float(profile[n % grid.nz]);
                    d.operation = RkTendencyOperation::BasePerturbationHalo;
                    d.input[2]  = workspace.native_base_profile.data();
                }
                const auto& closed = workspace.native_tendency.run(backend, d);
                tmv::NativeRkTendencyWorkspace::copy(*live[f], closed[0]);
            }
            for (const auto& scalar : private_scalars)
            {
                d.operation        = scalar.storage == tmv::MicrophysicsScalarStorage::VolumeConcentration
                                         ? RkTendencyOperation::VolumeConcentrationHalo
                                         : RkTendencyOperation::ScalarHalo;
                d.input            = {scalar.field->data(), fields.rho.data()};
                const auto& closed = workspace.native_tendency.run(backend, d);
                tmv::NativeRkTendencyWorkspace::copy(*scalar.field, closed[0]);
            }
            invalidate_all_resident_fields_backend(*state.compute_runtime);
            return;
        }
        bc_scheme->apply_rk_stage();
        tmv::rebuild_microphysics_boundary_conditions(state);
        invalidate_all_resident_fields_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"));
    };

    tmv::UnsplitRkCallbacks callbacks;
    callbacks.capture_base = [&]
    {
        if (native_combination)
            for (auto& field : workspace.native_combination)
                resize_like_grid(field);
        boundary_base = boundary_waves;
        for (std::size_t field = 0; field < live.size(); ++field)
        {
            resize_like_grid(workspace.base[field]);
            resize_like_grid(workspace.candidate[field]);
            for (std::size_t stage = 0; stage < active_rk_stages; ++stage)
                resize_like_grid(workspace.tendencies[stage][field]);
            copy_values(workspace.base[field], *live[field]);
        }
        for (auto& scalar : workspace.scalar_mass_tendency)
            resize_like_grid(scalar);
        for (std::size_t field = 0; field < private_scalars.size(); ++field)
        {
            resize_like_grid(workspace.private_base[field]);
            resize_like_grid(workspace.private_candidate[field]);
            resize_like_grid(workspace.private_mass_tendency[field]);
            for (std::size_t stage = 0; stage < active_rk_stages; ++stage)
                resize_like_grid(workspace.private_tendencies[stage][field]);
            copy_values(workspace.private_base[field], *private_scalars[field].field);
        }
    };
    const auto native_scalar_tendency =
        [&](const Field3D& source, Field3D& result, bool positive, tmv::CartesianFaceReconstruction reconstruction)
    {
        auto d              = native_dispatch;
        d.dt                = integration_dt;
        d.positive          = positive;
        d.weno              = reconstruction == tmv::CartesianFaceReconstruction::WENO5;
        d.operation         = RkTendencyOperation::ScalarFlux;
        d.input             = {source.data(),
                               stage_transport.mass_flux_x.data(),
                               stage_transport.mass_flux_y.data(),
                               stage_transport.mass_flux_z.data(),
                               fields.rho.data()};
        auto&       backend = *state.compute_runtime->backend;
        auto&       scratch = workspace.native_tendency;
        const auto& flux    = scratch.run(backend, d);
        d.input             = {source.data(), flux[0].data(), flux[1].data(), flux[2].data(), fields.rho.data()};
        if (positive)
        {
            d.operation         = RkTendencyOperation::DonorLimiter;
            const auto& limiter = scratch.run(backend, d, 1);
            d.input[5]          = limiter[0].data();
        }
        d.operation          = RkTendencyOperation::LimitedDivergence;
        const auto& tendency = scratch.run(backend, d, 2);
        tmv::NativeRkTendencyWorkspace::copy(result, tendency[0]);
    };
    const auto compute_tendencies = [&](std::size_t stage, double stage_time)
    {
        if (stage >= active_rk_stages || !std::isfinite(stage_time))
            throw std::invalid_argument("unsplit RK stage is invalid");
        if (stage == 0)
            close_stage_boundaries();
        dynamics_runtime.scheme->compute_momentum_tendencies(fields.u,
                                                             fields.v,
                                                             fields.w,
                                                             fields.rho,
                                                             fields.p,
                                                             fields.theta,
                                                             integration_dt,
                                                             du_dt,
                                                             dv_dt,
                                                             dw_dt,
                                                             drho_dt,
                                                             dp_dt);
        if (native_combination)
        {
            auto d              = native_dispatch;
            d.dt                = integration_dt;
            d.operation         = RkTendencyOperation::MassFlux;
            d.input             = {fields.u.data(), fields.v.data(), fields.w.data(), fields.rho.data()};
            auto&       backend = *state.compute_runtime->backend;
            const auto& flux    = workspace.native_tendency.run(backend, d);
            stage_transport.begin(state, grid.nr, grid.nth, grid.nz, integration_dt);
            stage_transport.capture_density_begin(fields.rho);
            tmv::NativeRkTendencyWorkspace::copy(stage_transport.mass_flux_x, flux[0]);
            tmv::NativeRkTendencyWorkspace::copy(stage_transport.mass_flux_y, flux[1]);
            tmv::NativeRkTendencyWorkspace::copy(stage_transport.mass_flux_z, flux[2]);
            stage_transport.record_external_accumulation(integration_dt, 1);
            stage_transport.complete(state);
            d.operation                  = RkTendencyOperation::Divergence;
            d.input                      = {fields.rho.data(),
                                            stage_transport.mass_flux_x.data(),
                                            stage_transport.mass_flux_y.data(),
                                            stage_transport.mass_flux_z.data()};
            const auto& density_tendency = workspace.native_tendency.run(backend, d);
            tmv::NativeRkTendencyWorkspace::copy(drho_dt, density_tendency[0]);
        }
        else
        {
            stage_transport.capture_instantaneous(
                state,
                fields.rho,
                fields.u,
                fields.v,
                fields.w,
                integration_dt,
                tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.u_ms,
                tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.v_ms);
            drho_dt.fill(0.0f);
            for (int i = scalar_domain.i_begin; i < scalar_domain.i_end; ++i)
                for (int j = scalar_domain.j_begin; j < scalar_domain.j_end; ++j)
                    for (int k = scalar_domain.k_begin; k < scalar_domain.k_end; ++k)
                    {
                        const auto&  fx     = stage_transport.mass_flux_x;
                        const auto&  fy     = stage_transport.mass_flux_y;
                        const auto&  fz     = stage_transport.mass_flux_z;
                        const double bottom = k > 0 ? fz[i][j][k - 1] : 0.0;
                        drho_dt[i][j][k] =
                            static_cast<float>(-(static_cast<double>(fx[i][j][k]) - fx[i - 1][j][k]) / grid.dr -
                                               (static_cast<double>(fy[i][j][k]) - fy[i][j - 1][k]) / grid.dtheta -
                                               (static_cast<double>(fz[i][j][k]) - bottom) / grid.dz);
                    }
        }
        const std::array<Field3D*, 10> transported_scalars{{
            &fields.theta,
            &fields.qv,
            &fields.tracer,
            &fields.qc,
            &fields.qr,
            &fields.qi,
            &fields.qs,
            &fields.qg,
            &fields.qh,
            &fields.tke,
        }};
        for (std::size_t scalar = 0; scalar < transported_scalars.size(); ++scalar)
        {
            if (scalar_transport_enabled && native_combination)
                native_scalar_tendency(*transported_scalars[scalar],
                                       workspace.scalar_mass_tendency[scalar],
                                       scalar != 0,
                                       scalar == 0 ? scalar_reconstruction
                                                   : tmv::CartesianFaceReconstruction::MonotonicCentered);
            else if (scalar_transport_enabled)
                tmv::compute_conservative_scalar_mass_tendency_cartesian(
                    state,
                    *transported_scalars[scalar],
                    workspace.scalar_mass_tendency[scalar],
                    integration_dt,
                    scalar != 0,
                    stage_transport,
                    scalar == 0 ? scalar_reconstruction : tmv::CartesianFaceReconstruction::MonotonicCentered);
            else
                workspace.scalar_mass_tendency[scalar].fill(0.0f);
        }
        for (std::size_t scalar = 0; scalar < private_scalars.size(); ++scalar)
        {
            if (scalar_transport_enabled)
            {
                FieldPool::ScopedField converted({}, FieldPool::instance());
                const Field3D*         source = private_scalars[scalar].field;
                const bool             volume =
                    private_scalars[scalar].storage == tmv::MicrophysicsScalarStorage::VolumeConcentration;
                const bool project =
                    !ssp_rk && private_scalars[scalar].positive_definite &&
                    std::any_of(source->data(), source->data() + source->size(), [](float value)
                    {
                        return value < 0;
                    });
                if (project)
                {
                    tmv::log_debug("[RK PRIVATE TRANSPORT INPUT] scheme=rk4, stage=",
                                   stage,
                                   ", field=",
                                   private_scalars[scalar].name,
                                   ", projection=nonnegative");
                    converted.field = converted.pool.acquire_for_overwrite(grid.nr, grid.nth, grid.nz);
                    if (native_combination)
                    {
                        auto d                 = native_dispatch;
                        d.operation            = volume ? RkTendencyOperation::NonnegativeMixingRatio
                                                        : RkTendencyOperation::NonnegativeInput;
                        d.input                = {source->data(), fields.rho.data()};
                        const auto& admissible = workspace.native_tendency.run(*state.compute_runtime->backend, d);
                        tmv::NativeRkTendencyWorkspace::copy(converted.field, admissible[0]);
                    }
                    else
                    {
                        for (std::size_t n = 0; n < source->size(); ++n)
                        {
                            const double value        = std::max(0.0f, source->data()[n]);
                            converted.field.data()[n] = float(volume ? value / fields.rho.data()[n] : value);
                        }
                    }
                    source = &converted.field;
                }
                else if (volume)
                {
                    converted.field = converted.pool.acquire_for_overwrite(grid.nr, grid.nth, grid.nz);
                    if (native_combination)
                    {
                        auto d            = native_dispatch;
                        d.operation       = RkTendencyOperation::MixingRatio;
                        d.input           = {source->data(), fields.rho.data()};
                        const auto& ratio = workspace.native_tendency.run(*state.compute_runtime->backend, d);
                        tmv::NativeRkTendencyWorkspace::copy(converted.field, ratio[0]);
                    }
                    else
                        tmv::microphysics_scalar_to_mixing_ratio(private_scalars[scalar], fields.rho, converted.field);
                    source = &converted.field;
                }
                if (native_combination)
                    native_scalar_tendency(*source,
                                           workspace.private_mass_tendency[scalar],
                                           private_scalars[scalar].positive_definite,
                                           tmv::CartesianFaceReconstruction::MonotonicCentered);
                else
                    tmv::compute_conservative_scalar_mass_tendency_cartesian(
                        state,
                        *source,
                        workspace.private_mass_tendency[scalar],
                        integration_dt,
                        private_scalars[scalar].positive_definite,
                        stage_transport,
                        tmv::CartesianFaceReconstruction::MonotonicCentered);
            }
            else
                workspace.private_mass_tendency[scalar].fill(0.0f);
        }
        const std::array<const Field3D*, tmv::TimeSteppingUnsplitRkWorkspace::field_count> computed{
            {&du_dt,
             &dv_dt,
             &dw_dt,
             &drho_dt,
             &dp_dt,
             &workspace.scalar_mass_tendency[0],
             &workspace.scalar_mass_tendency[1],
             &workspace.scalar_mass_tendency[2],
             &workspace.scalar_mass_tendency[3],
             &workspace.scalar_mass_tendency[4],
             &workspace.scalar_mass_tendency[5],
             &workspace.scalar_mass_tendency[6],
             &workspace.scalar_mass_tendency[7],
             &workspace.scalar_mass_tendency[8],
             &workspace.scalar_mass_tendency[9]}};
        for (const Field3D* tendency : computed)
            if (tendency->size_r() != grid.nr || tendency->size_th() != grid.nth || tendency->size_z() != grid.nz)
                throw std::runtime_error("unsplit RK tendency shape mismatch");
        if (tendency_fields.du_dt_pbl.size_r() != grid.nr || tendency_fields.du_dt_pbl.size_th() != grid.nth ||
            tendency_fields.du_dt_pbl.size_z() != grid.nz || tendency_fields.dv_dt_pbl.size_r() != grid.nr ||
            tendency_fields.dv_dt_pbl.size_th() != grid.nth || tendency_fields.dv_dt_pbl.size_z() != grid.nz)
            throw std::runtime_error("unsplit RK PBL tendency shape mismatch");

        if (native_combination)
        {
            auto d            = native_dispatch;
            d.operation       = RkTendencyOperation::Add;
            d.input           = {du_dt.data(), tendency_fields.du_dt_pbl.data()};
            const auto& u_sum = workspace.native_tendency.run(*state.compute_runtime->backend, d);
            tmv::NativeRkTendencyWorkspace::copy(du_dt, u_sum[0]);
            d.input           = {dv_dt.data(), tendency_fields.dv_dt_pbl.data()};
            const auto& v_sum = workspace.native_tendency.run(*state.compute_runtime->backend, d);
            tmv::NativeRkTendencyWorkspace::copy(dv_dt, v_sum[0]);
        }
        int invalid_tendency = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(| : invalid_tendency)
        for (int i = 0; i < grid.nr; ++i)
            for (int j = 0; j < grid.nth; ++j)
                for (int k = 0; k < grid.nz; ++k)
                {
                    std::array<double, tmv::TimeSteppingUnsplitRkWorkspace::field_count> values{};
                    values[0] = static_cast<double>(du_dt[i][j][k]) +
                                (native_combination ? 0.0 : static_cast<double>(tendency_fields.du_dt_pbl[i][j][k]));
                    values[1] = static_cast<double>(dv_dt[i][j][k]) +
                                (native_combination ? 0.0 : static_cast<double>(tendency_fields.dv_dt_pbl[i][j][k]));
                    values[2] = static_cast<double>(dw_dt[i][j][k]);
                    values[3] = static_cast<double>(drho_dt[i][j][k]);
                    values[4] = static_cast<double>(dp_dt[i][j][k]);
                    for (std::size_t scalar = 0; scalar < workspace.scalar_mass_tendency.size(); ++scalar)
                    {
                        values[5 + scalar] = static_cast<double>(workspace.scalar_mass_tendency[scalar][i][j][k]);
                    }
                    for (std::size_t field = 0; field < values.size(); ++field)
                    {
                        if (!std::isfinite(values[field]) || !std::isfinite(static_cast<float>(values[field])))
                            invalid_tendency = 1;
                        workspace.tendencies[stage][field][i][j][k] = static_cast<float>(values[field]);
                    }
                }
        if (invalid_tendency != 0)
            throw std::runtime_error("unsplit RK tendency is non-finite");
        for (std::size_t scalar = 0; scalar < private_scalars.size(); ++scalar)
        {
            const Field3D& current         = *private_scalars[scalar].field;
            Field3D&       tendency        = workspace.private_tendencies[stage][scalar];
            int            invalid_private = 0;
#pragma omp parallel for reduction(| : invalid_private)
            for (std::size_t index = 0; index < current.size(); ++index)
            {
                const double value = static_cast<double>(workspace.private_mass_tendency[scalar].data()[index]);
                if (!std::isfinite(value))
                    invalid_private = 1;
                tendency.data()[index] = static_cast<float>(value);
            }
            if (invalid_private != 0)
                throw std::runtime_error(std::string("unsplit RK private tendency is non-finite: ") +
                                         private_scalars[scalar].name);
        }
    };
    callbacks.compute_tendencies = [&](std::size_t stage, double stage_time) { compute_tendencies(stage, stage_time); };
    const auto combine_stage     = [&](double                       base,
                                       double                       current,
                                       const auto&                  rhs,
                                       std::size_t                  stages,
                                       const std::array<double, 4>& weights,
                                       double                       step,
                                       double&                      increment_scale)
    { return tmv::rk::combine(base, current, rhs, stages, weights, step, ssp_rk, increment_scale); };
    const auto combine_native = [&](const Field3D&                     base,
                                    const Field3D&                     current,
                                    const std::array<const float*, 4>& rhs,
                                    std::size_t                        stages,
                                    const std::array<double, 4>&       weights,
                                    double                             step,
                                    bool                               conserved,
                                    bool                               held)
    {
        if (held)
        {
            copy_values(workspace.native_combination[0], base);
            for (std::size_t f = 1; f < workspace.native_combination.size(); ++f)
                workspace.native_combination[f].fill(0.0f);
            return;
        }
        RkCombinationDispatch d;
        d.base                   = base.data();
        d.current                = current.data();
        d.rhs                    = rhs;
        d.cells                  = base.size();
        d.stages                 = int(stages);
        d.weights                = weights;
        d.dt                     = step;
        d.ssp                    = ssp_rk;
        d.conserved_mixing_ratio = conserved;
        d.base_density           = workspace.base[3].data();
        d.current_density        = fields.rho.data();
        d.candidate_density      = workspace.candidate[3].data();
        for (int f = 0; f < 4; ++f)
            d.output[f] = workspace.native_combination[f].data();
        const auto result = state.compute_runtime->backend->dispatch_rk_combination(d);
        if (result == RkCombinationStatus::NonFinite)
            throw tmv::UnsplitRkBoundError("native RK combination is non-finite");
        if (result != RkCombinationStatus::Completed)
            throw std::runtime_error("native RK stage combination failed");
    };
    const auto publish_combination = [&](std::size_t stage_count, const std::array<double, 4>& weights, double dt_stage)
    {
        if (stage_count == 0 || stage_count > active_rk_stages || !std::isfinite(dt_stage) || dt_stage <= 0.0)
            throw std::invalid_argument("unsplit RK combination is invalid");
        for (std::size_t stage = 0; stage < stage_count; ++stage)
            if (!std::isfinite(weights[stage]))
                throw std::invalid_argument("unsplit RK weight is non-finite");

        int        invalid_candidate         = 0;
        int        first_invalid_field       = static_cast<int>(live.size());
        long long  roundoff_projection_count = 0;
        double     minimum_bounded_value     = 0.0;
        const auto form_primary_candidates   = [&](std::size_t first_field, std::size_t end_field)
        {
            int       local_invalid = 0, local_first = int(live.size());
            long long local_roundoff = 0;
            double    local_minimum  = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(| : local_invalid)                                     \
    reduction(min : local_first, local_minimum) reduction(+ : local_roundoff)
            for (int i = 0; i < grid.nr; ++i)
                for (int j = 0; j < grid.nth; ++j)
                    for (int k = 0; k < grid.nz; ++k)
                        for (std::size_t field = first_field; field < end_field; ++field)
                        {
                            double value = 0, increment_scale = 0;
                            if (native_combination)
                            {
                                value           = double(workspace.native_combination[0][i][j][k]) +
                                                  workspace.native_combination[1][i][j][k];
                                increment_scale = double(workspace.native_combination[2][i][j][k]) +
                                                  workspace.native_combination[3][i][j][k];
                            }
                            else
                            {
                                value                       = static_cast<double>(workspace.base[field][i][j][k]);
                                const bool conserved_scalar = field >= 5 && scalar_transport_enabled;
                                if (conserved_scalar)
                                    value *= static_cast<double>(workspace.base[3][i][j][k]);
                                double current = static_cast<double>((*live[field])[i][j][k]);
                                if (conserved_scalar)
                                    current *= static_cast<double>(fields.rho[i][j][k]);
                                value = combine_stage(
                                    value,
                                    current,
                                    [&](std::size_t stage)
                                    { return static_cast<double>(workspace.tendencies[stage][field][i][j][k]); },
                                    stage_count,
                                    weights,
                                    dt_stage,
                                    increment_scale);
                                if (conserved_scalar)
                                {
                                    const double density = workspace.candidate[3][i][j][k];
                                    value /= density;
                                    increment_scale /= density;
                                }
                            }
                            const bool lower_bounded = field >= 6;
                            if (lower_bounded)
                                local_minimum = std::min(local_minimum, value);
                            const double upper_bound = field == 6   ? qv_max_kgkg
                                                       : field == 7 ? passive_tracer_max
                                                       : field >= 8 && field <= 13
                                                           ? hydrometeor_max_kgkg
                                                           : std::numeric_limits<double>::infinity();
                            const double roundoff_scale =
                                std::max({std::abs(static_cast<double>(workspace.base[field][i][j][k])),
                                          std::abs(static_cast<double>((*live[field])[i][j][k])),
                                          increment_scale});
                            const double float_epsilon = static_cast<double>(std::numeric_limits<float>::epsilon());
                            const double absolute_roundoff_bound =
                                std::isfinite(upper_bound) ? 64.0 * float_epsilon * float_epsilon * upper_bound : 0.0;
                            const double roundoff_bound =
                                std::max(8.0 * float_epsilon * roundoff_scale, absolute_roundoff_bound);
                            if (lower_bounded && value < 0.0 && value >= -roundoff_bound)
                            {
                                value = 0.0;
                                ++local_roundoff;
                            }
                            if (std::isfinite(upper_bound) && value > upper_bound &&
                                value <= upper_bound + roundoff_bound)
                            {
                                value = upper_bound;
                                ++local_roundoff;
                            }
                            bool field_invalid =
                                !std::isfinite(value) || ((field == 3 || field == 4 || field == 5) && value <= 0.0) ||
                                (field == 6 && (value < 0.0 || value > qv_max_kgkg)) ||
                                (field >= 8 && field <= 13 && (value < 0.0 || value > hydrometeor_max_kgkg)) ||
                                (field == 14 && value < 0.0);
                            if (field == 7 && (value < 0.0 || value > passive_tracer_max))
                                field_invalid = true;
                            float published = static_cast<float>(value);
                            if (field < 2)
                                published = clamp_wind_horizontal_ms(published);
                            else if (field == 2)
                                published = clamp_wind_vertical_ms(published);
                            else if (field == 3)
                                published = clamp_density_kgm3(published);
                            else if (field == 4)
                                published = clamp_pressure_pa(published);
                            else if (field == 5)
                                published = clamp_theta_k(published);
                            else if (field == 6)
                                published = clamp_qv_kgkg(published);
                            else if (field == 7)
                                published = std::clamp(published, 0.0f, passive_tracer_max);
                            else if (field >= 8 && field <= 13)
                                published = clamp_hydrometeor_kgkg(published);
                            else if (field == 14)
                                published = clamp_non_negative(published);
                            field_invalid =
                                field_invalid || !std::isfinite(static_cast<double>(published)) ||
                                ((field == 3 || field == 4 || field == 5) && published <= 0.0f) ||
                                (field == 6 && (published < 0.0f || published > qv_max_kgkg)) ||
                                (field == 7 && (published < 0.0f || published > passive_tracer_max)) ||
                                (field >= 8 && field <= 13 && (published < 0.0f || published > hydrometeor_max_kgkg)) ||
                                (field == 14 && published < 0.0f);
                            if (field_invalid)
                            {
                                local_invalid = 1;
                                local_first   = std::min(local_first, static_cast<int>(field));
                            }
                            workspace.candidate[field][i][j][k] = published;
                        }
            invalid_candidate |= local_invalid;
            first_invalid_field = std::min(first_invalid_field, local_first);
            roundoff_projection_count += local_roundoff;
            minimum_bounded_value = std::min(minimum_bounded_value, local_minimum);
        };
        if (native_combination)
        {
            for (std::size_t field = 0; field < live.size(); ++field)
            {
                std::array<const float*, 4> rhs{};
                for (std::size_t stage = 0; stage < stage_count; ++stage)
                    rhs[stage] = workspace.tendencies[stage][field].data();
                combine_native(workspace.base[field],
                               *live[field],
                               rhs,
                               stage_count,
                               weights,
                               dt_stage,
                               field >= 5 && scalar_transport_enabled,
                               field >= 5 && !scalar_transport_enabled);
                form_primary_candidates(field, field + 1);
                if (invalid_candidate)
                    break;
            }
        }
        else
            form_primary_candidates(0, live.size());
        if (roundoff_projection_count > 0)
            tmv::log_warn("[RK BOUND ROUNDOFF] projected_values=",
                          roundoff_projection_count,
                          ", tolerance=max(8_float_eps_scaled,"
                          "64_float_eps_squared_field_bound)");
        if (invalid_candidate != 0)
        {
            static constexpr std::array<const char*, tmv::TimeSteppingUnsplitRkWorkspace::field_count> names{{
                "u",
                "v",
                "w",
                "rho",
                "p",
                "theta",
                "qv",
                "tracer",
                "qc",
                "qr",
                "qi",
                "qs",
                "qg",
                "qh",
                "tke",
            }};
            std::ostringstream                                                                         message;
            message << "unsplit RK candidate violates bounds for "
                    << (first_invalid_field >= 0 && first_invalid_field < static_cast<int>(names.size())
                            ? names[static_cast<std::size_t>(first_invalid_field)]
                            : "an unknown field")
                    << ", minimum_bounded_value=" << std::scientific << std::setprecision(17) << minimum_bounded_value;
            throw tmv::UnsplitRkBoundError(message.str());
        }
        for (std::size_t field = 0; field < private_scalars.size(); ++field)
        {
            int            invalid_private        = 0;
            long long      private_roundoff_count = 0;
            const Field3D& base                   = workspace.private_base[field];
            Field3D&       candidate              = workspace.private_candidate[field];
            if (native_combination)
            {
                std::array<const float*, 4> rhs{};
                for (std::size_t stage = 0; stage < stage_count; ++stage)
                    rhs[stage] = workspace.private_tendencies[stage][field].data();
                combine_native(base,
                               *private_scalars[field].field,
                               rhs,
                               stage_count,
                               weights,
                               dt_stage,
                               scalar_transport_enabled &&
                                   private_scalars[field].storage == tmv::MicrophysicsScalarStorage::DryAirMixingRatio,
                               !scalar_transport_enabled);
            }

#pragma omp parallel for reduction(| : invalid_private) reduction(+ : private_roundoff_count)
            for (std::size_t index = 0; index < base.size(); ++index)
            {
                double value = 0, increment_scale = 0;
                if (native_combination)
                {
                    value           = double(workspace.native_combination[0].data()[index]) +
                                      workspace.native_combination[1].data()[index];
                    increment_scale = double(workspace.native_combination[2].data()[index]) +
                                      workspace.native_combination[3].data()[index];
                }
                else
                {
                    value = static_cast<double>(base.data()[index]);
                    if (scalar_transport_enabled)
                        value = tmv::microphysics_scalar_mass_density(
                            value, workspace.base[3].data()[index], private_scalars[field].storage);
                    double current = static_cast<double>(private_scalars[field].field->data()[index]);
                    if (scalar_transport_enabled)
                        current = tmv::microphysics_scalar_mass_density(
                            current, fields.rho.data()[index], private_scalars[field].storage);
                    value = combine_stage(
                        value,
                        current,
                        [&](std::size_t stage)
                        { return static_cast<double>(workspace.private_tendencies[stage][field].data()[index]); },
                        stage_count,
                        weights,
                        dt_stage,
                        increment_scale);
                    if (scalar_transport_enabled)
                    {
                        const double density = workspace.candidate[3].data()[index];
                        value =
                            tmv::microphysics_scalar_from_mass_density(value, density, private_scalars[field].storage);
                        increment_scale = tmv::microphysics_scalar_from_mass_density(
                            increment_scale, density, private_scalars[field].storage);
                    }
                }
                // An RK4 combination can omit the derivative that created
                // its current stage, but that stage still sets rounding scale.
                const double roundoff_bound =
                    8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
                        std::max({std::abs(static_cast<double>(base.data()[index])),
                                  std::abs(static_cast<double>(private_scalars[field].field->data()[index])),
                                  increment_scale}) +
                    8.0 * static_cast<double>(std::numeric_limits<float>::denorm_min()) *
                        (1.0 + tmv::microphysics_scalar_from_mass_density(
                                   dt_stage * (1.0 + 1.0 / grid.dr + 1.0 / grid.dtheta + 1.0 / grid.dz),
                                   workspace.candidate[3].data()[index],
                                   private_scalars[field].storage));
                // Raw classical RK stages are retained; only transport's temporary evaluation input is projected.
                const bool provisional = !ssp_rk && scalar_transport_enabled && stage_count < active_rk_stages;
                if (!provisional && private_scalars[field].positive_definite && value < 0.0 && value >= -roundoff_bound)
                {
                    value = 0.0;
                    ++private_roundoff_count;
                }
                if (!std::isfinite(value) || !std::isfinite(static_cast<float>(value)) ||
                    (!provisional && private_scalars[field].positive_definite && value < 0.0))
                    invalid_private = 1;
                candidate.data()[index] = static_cast<float>(value);
            }
            if (private_roundoff_count > 0)
                tmv::log_warn("[RK PRIVATE BOUND ROUNDOFF] field=",
                              private_scalars[field].name,
                              ", projected_values=",
                              private_roundoff_count,
                              ", tolerance=8_float_eps_scaled_plus_subnormal");
            if (invalid_private != 0)
            {
                const auto         minimum = std::min_element(candidate.data(), candidate.data() + candidate.size());
                const auto         index   = static_cast<std::size_t>(minimum - candidate.data());
                std::ostringstream message;
                message << "unsplit RK private candidate violates bounds for " << private_scalars[field].name
                        << ", minimum=" << std::scientific << *minimum << ", index=" << index
                        << ", base=" << base.data()[index] << ", density_begin=" << workspace.base[3].data()[index]
                        << ", density_end=" << workspace.candidate[3].data()[index] << ", dt=" << dt_stage
                        << ", stages=" << stage_count;
                for (std::size_t stage = 0; stage < stage_count; ++stage)
                    message << ", rhs" << stage << "=" << workspace.private_tendencies[stage][field].data()[index];
                throw tmv::UnsplitRkBoundError(message.str());
            }
        }
        for (std::size_t field = 0; field < live.size(); ++field)
            copy_values(*live[field], workspace.candidate[field]);
        for (std::size_t field = 0; field < private_scalars.size(); ++field)
            copy_values(*private_scalars[field].field, workspace.private_candidate[field]);
        close_stage_boundaries();
    };
    callbacks.publish_combination = [&](std::size_t stages, const std::array<double, 4>& weights, double step)
    { publish_combination(stages, weights, step); };
    callbacks.restore_base = [&]
    {
        std::swap(boundary_waves, boundary_base);
        invalidate_all_resident_fields_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"));
        for (std::size_t field = 0; field < live.size(); ++field)
            std::copy_n(workspace.base[field].data(), workspace.base[field].size(), live[field]->data());
        for (std::size_t field = 0; field < private_scalars.size(); ++field)
            copy_values(*private_scalars[field].field, workspace.private_base[field]);
    };

    try
    {
        time_runtime.scheme->step_coupled_unsplit_rk(
            time_runtime.config, dt_dynamics, current_time, callbacks, &time_runtime.diagnostics);
        return;
    }
    catch (const tmv::UnsplitRkBoundError& initial_error)
    {
        for (std::size_t field = 0; field < live.size(); ++field)
        {
            resize_like_grid(workspace.retry_base[field]);
            copy_values(workspace.retry_base[field], *live[field]);
        }
        for (std::size_t field = 0; field < private_scalars.size(); ++field)
        {
            resize_like_grid(workspace.private_retry_base[field]);
            copy_values(workspace.private_retry_base[field], *private_scalars[field].field);
        }
        const auto                         retry_boundary_base = boundary_waves;
        const tmv::TimeSteppingDiagnostics diagnostics_base    = time_runtime.diagnostics;
        constexpr int                      maximum_segments    = 64;
        for (int segments = 2; segments <= maximum_segments; segments *= 2)
        {
            for (std::size_t field = 0; field < live.size(); ++field)
                copy_values(*live[field], workspace.retry_base[field]);
            for (std::size_t field = 0; field < private_scalars.size(); ++field)
                copy_values(*private_scalars[field].field, workspace.private_retry_base[field]);
            time_runtime.diagnostics = diagnostics_base;
            boundary_waves           = retry_boundary_base;
            const double segment_dt  = dt_dynamics / segments;
            integration_dt           = segment_dt;
            bool retry               = false;
            try
            {
                for (int segment = 0; segment < segments; ++segment)
                {
                    time_runtime.scheme->step_coupled_unsplit_rk(time_runtime.config,
                                                                 segment_dt,
                                                                 current_time + segment * segment_dt,
                                                                 callbacks,
                                                                 &time_runtime.diagnostics);
                }
            }
            catch (const tmv::UnsplitRkBoundError&)
            {
                retry = true;
            }
            catch (...)
            {
                for (std::size_t field = 0; field < live.size(); ++field)
                    copy_values(*live[field], workspace.retry_base[field]);
                for (std::size_t field = 0; field < private_scalars.size(); ++field)
                    copy_values(*private_scalars[field].field, workspace.private_retry_base[field]);
                time_runtime.diagnostics = diagnostics_base;
                boundary_waves           = retry_boundary_base;
                throw;
            }
            if (!retry)
            {
                tmv::log_warn("[RK BOUND SUBCYCLE] interval_s=",
                              dt_dynamics,
                              ", segments=",
                              segments,
                              ", segment_dt_s=",
                              segment_dt);
                return;
            }
        }
        for (std::size_t field = 0; field < live.size(); ++field)
            copy_values(*live[field], workspace.retry_base[field]);
        for (std::size_t field = 0; field < private_scalars.size(); ++field)
            copy_values(*private_scalars[field].field, workspace.private_retry_base[field]);
        time_runtime.diagnostics = diagnostics_base;
        boundary_waves           = retry_boundary_base;
        throw tmv::UnsplitRkBoundError(std::string(initial_error.what()) +
                                       " after exhausting transactional RK bound subcycling");
    }
}

static void log_step_velocity_maxima_if_enabled(const SimulationStateView& state)
{
    if (!log_debug_enabled())
    {
        return;
    }

    const auto& grid   = state.grid;
    const auto& fields = state.fields;
    double      max_u = 0.0, max_v = 0.0, max_w = 0.0;
    int         u_i = 0, u_j = 0, u_k = 0;
    int         v_i = 0, v_j = 0, v_k = 0;
    int         w_i = 0, w_j = 0, w_k = 0;
    for (int i = 0; i < grid.nr; ++i)
        for (int j = 0; j < grid.nth; ++j)
            for (int k = 0; k < grid.nz; ++k)
            {
                const double au = std::abs(static_cast<double>(fields.u[i][j][k]));
                const double av = std::abs(static_cast<double>(fields.v[i][j][k]));
                const double aw = std::abs(static_cast<double>(fields.w[i][j][k]));
                if (std::isfinite(au) && au > max_u)
                {
                    max_u = au;
                    u_i   = i;
                    u_j   = j;
                    u_k   = k;
                }
                if (std::isfinite(av) && av > max_v)
                {
                    max_v = av;
                    v_i   = i;
                    v_j   = j;
                    v_k   = k;
                }
                if (std::isfinite(aw) && aw > max_w)
                {
                    max_w = aw;
                    w_i   = i;
                    w_j   = j;
                    w_k   = k;
                }
            }
    tmv::log_debug("[VEL-MAX] max_u=",
                   max_u,
                   "@(",
                   u_i,
                   ",",
                   u_j,
                   ",",
                   u_k,
                   ") max_v=",
                   max_v,
                   "@(",
                   v_i,
                   ",",
                   v_j,
                   ",",
                   v_k,
                   ") max_w=",
                   max_w,
                   "@(",
                   w_i,
                   ",",
                   w_j,
                   ",",
                   w_k,
                   ")");
}

static double slow_mixture_mass_kg(const SimulationStateView&                               state,
                                   const tmv::physical_grid_metrics::ControlVolumeMassView& mass_geometry,
                                   int                                                      i,
                                   int                                                      j,
                                   int                                                      k)
{
    const auto&  fields  = state.fields;
    const double rho_d   = static_cast<double>(fields.rho[i][j][k]);
    const double water[] = {
        static_cast<double>(fields.qv[i][j][k]),
        static_cast<double>(fields.qc[i][j][k]),
        static_cast<double>(fields.qr[i][j][k]),
        static_cast<double>(fields.qi[i][j][k]),
        static_cast<double>(fields.qs[i][j][k]),
        static_cast<double>(fields.qg[i][j][k]),
        static_cast<double>(fields.qh[i][j][k]),
    };
    if (!std::isfinite(rho_d) || rho_d <= 0.0)
        return std::numeric_limits<double>::quiet_NaN();
    double q_total = 0.0;
    for (double value : water)
    {
        if (!std::isfinite(value) || value < 0.0)
            return std::numeric_limits<double>::quiet_NaN();
        q_total += value;
    }
    const double mass = mass_geometry.scalar_mass_kg(fields.rho, i, j, k) * (1.0 + q_total);
    return std::isfinite(mass) && mass > 0.0 ? mass : std::numeric_limits<double>::quiet_NaN();
}

static std::array<double, 6> slow_momentum_requested_work(const SimulationStateView&                  state,
                                                          double                                      dt_large,
                                                          const Field3D&                              du_dt,
                                                          const Field3D&                              dv_dt,
                                                          const Field3D&                              dw_dt,
                                                          const SlowMomentumAdvectionTendencyOutputs& advection_outputs)
{
    if (!std::isfinite(dt_large) || dt_large <= 0.0 || !advection_outputs.complete())
    {
        throw std::invalid_argument(
            "slow momentum energy attribution requires finite dt > 0 and complete advection tendencies");
    }
    const auto&              grid              = state.grid;
    const auto&              fields            = state.fields;
    const auto&              tendency_fields   = state.tendencies;
    const PhysicalCellDomain domain            = physical_scalar_domain(grid);
    const bool               cgrid             = grid.geometry.staggered;
    const Field3D*           required_fields[] = {
        &fields.u,
        &fields.v,
        &fields.w,
        &fields.rho,
        &fields.qv,
        &fields.qc,
        &fields.qr,
        &fields.qi,
        &fields.qs,
        &fields.qg,
        &fields.qh,
        &du_dt,
        &dv_dt,
        &dw_dt,
        &tendency_fields.du_dt_pbl,
        &tendency_fields.dv_dt_pbl,
        advection_outputs.du_dt,
        advection_outputs.dv_dt,
        advection_outputs.dw_dt,
    };
    if (std::any_of(std::begin(required_fields),
                    std::end(required_fields),
                    [&](const Field3D* field) { return field == nullptr || !field_matches_domain(*field, grid); }))
    {
        throw std::invalid_argument("slow momentum energy attribution requires grid-shaped state and tendency fields");
    }
    const TerrainMetrics3D* metrics = tmv::physical_grid_metrics::active_terrain_metrics(state);
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(state);

    const auto sums = tmv::diagnostics::fixed_slab_reduce<7>(
        domain.i_begin,
        domain.i_end,
        [&](int i, std::array<double, 7>& partial)
        {
            const auto accumulate = [&](double mass,
                                        double velocity,
                                        double total_tendency,
                                        double advection_tendency,
                                        bool   include_committed = true)
            {
                const double delta_advection = dt_large * advection_tendency;
                const double delta_other     = dt_large * (total_tendency - advection_tendency);
                const double values[]        = {
                    mass,
                    velocity,
                    delta_advection,
                    delta_other,
                };
                if (mass <= 0.0 || std::any_of(std::begin(values),
                                               std::end(values),
                                               [](double value) { return !std::isfinite(value); }))
                {
                    partial[6] += 1.0;
                    return;
                }
                partial[0] += mass * velocity * delta_advection;
                partial[1] += mass * velocity * delta_other;
                partial[2] += 0.5 * mass * delta_advection * delta_advection;
                partial[3] += 0.5 * mass * delta_other * delta_other;
                partial[4] += mass * delta_advection * delta_other;
                if (include_committed)
                {
                    const double stored = static_cast<double>(static_cast<float>(velocity + dt_large * total_tendency));
                    if (!std::isfinite(stored))
                    {
                        partial[6] += 1.0;
                        return;
                    }
                    partial[5] += 0.5 * mass * (stored * stored - velocity * velocity);
                }
            };

            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double cell_mass = slow_mixture_mass_kg(state, mass_geometry, i, j, k);
                    if (!std::isfinite(cell_mass))
                    {
                        partial[6] += 1.0;
                        continue;
                    }
                    const double total_u =
                        static_cast<double>(du_dt[i][j][k]) + static_cast<double>(tendency_fields.du_dt_pbl[i][j][k]);
                    const double total_v =
                        static_cast<double>(dv_dt[i][j][k]) + static_cast<double>(tendency_fields.dv_dt_pbl[i][j][k]);
                    const double total_w = static_cast<double>(dw_dt[i][j][k]);
                    const double adv_u   = static_cast<double>((*advection_outputs.du_dt)[i][j][k]);
                    const double adv_v   = static_cast<double>((*advection_outputs.dv_dt)[i][j][k]);
                    const double adv_w   = static_cast<double>((*advection_outputs.dw_dt)[i][j][k]);
                    if (!cgrid)
                    {
                        accumulate(cell_mass, static_cast<double>(fields.u[i][j][k]), total_u, adv_u);
                        accumulate(cell_mass, static_cast<double>(fields.v[i][j][k]), total_v, adv_v);
                        accumulate(cell_mass, static_cast<double>(fields.w[i][j][k]), total_w, adv_w);
                        continue;
                    }

                    double u_mass = 0.5 * cell_mass;
                    if (i + 1 < domain.i_end)
                        u_mass += 0.5 * slow_mixture_mass_kg(state, mass_geometry, i + 1, j, k);
                    double v_mass = 0.5 * cell_mass;
                    if (j + 1 < domain.j_end)
                        v_mass += 0.5 * slow_mixture_mass_kg(state, mass_geometry, i, j + 1, k);
                    double w_mass = 0.5 * cell_mass;
                    if (k + 1 < domain.k_end)
                        w_mass += 0.5 * slow_mixture_mass_kg(state, mass_geometry, i, j, k + 1);
                    accumulate(u_mass, static_cast<double>(fields.u[i][j][k]), total_u, adv_u);
                    accumulate(v_mass, static_cast<double>(fields.v[i][j][k]), total_v, adv_v);
                    accumulate(w_mass, static_cast<double>(fields.w[i][j][k]), total_w, adv_w);

                    if (metrics != nullptr && k == domain.k_begin)
                    {
                        const double slope_x  = -metrics->Gx_face(i, j, 0);
                        const double slope_y  = -metrics->Gy_face(i, j, 0);
                        const double ground_w = terrain_transport::cgrid_ground_vertical_velocity(
                            *metrics,
                            fields.u,
                            fields.v,
                            i,
                            j,
                            tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms,
                            tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms);
                        const double total_ground =
                            slope_x * 0.5 *
                                (static_cast<double>(du_dt[i - 1][j][k]) + static_cast<double>(du_dt[i][j][k]) +
                                 static_cast<double>(tendency_fields.du_dt_pbl[i - 1][j][k]) +
                                 static_cast<double>(tendency_fields.du_dt_pbl[i][j][k])) +
                            slope_y * 0.5 *
                                (static_cast<double>(dv_dt[i][j - 1][k]) + static_cast<double>(dv_dt[i][j][k]) +
                                 static_cast<double>(tendency_fields.dv_dt_pbl[i][j - 1][k]) +
                                 static_cast<double>(tendency_fields.dv_dt_pbl[i][j][k]));
                        const double adv_ground = slope_x * 0.5 *
                                                      (static_cast<double>((*advection_outputs.du_dt)[i - 1][j][k]) +
                                                       static_cast<double>((*advection_outputs.du_dt)[i][j][k])) +
                                                  slope_y * 0.5 *
                                                      (static_cast<double>((*advection_outputs.dv_dt)[i][j - 1][k]) +
                                                       static_cast<double>((*advection_outputs.dv_dt)[i][j][k]));
                        accumulate(0.5 * cell_mass, ground_w, total_ground, adv_ground, false);
                        const auto stored_velocity = [&](const Field3D& velocity,
                                                         const Field3D& tendency,
                                                         const Field3D& pbl_tendency,
                                                         int            face_i,
                                                         int            face_j)
                        {
                            return static_cast<double>(
                                static_cast<float>(static_cast<double>(velocity[face_i][face_j][k]) +
                                                   dt_large * (static_cast<double>(tendency[face_i][face_j][k]) +
                                                               static_cast<double>(pbl_tendency[face_i][face_j][k]))));
                        };
                        const double ground_after =
                            slope_x * 0.5 *
                                (stored_velocity(fields.u, du_dt, tendency_fields.du_dt_pbl, i - 1, j) +
                                 stored_velocity(fields.u, du_dt, tendency_fields.du_dt_pbl, i, j)) +
                            slope_y * 0.5 *
                                (stored_velocity(fields.v, dv_dt, tendency_fields.dv_dt_pbl, i, j - 1) +
                                 stored_velocity(fields.v, dv_dt, tendency_fields.dv_dt_pbl, i, j));
                        if (!std::isfinite(ground_after))
                            partial[6] += 1.0;
                        else
                            partial[5] += 0.25 * cell_mass * (ground_after * ground_after - ground_w * ground_w);
                    }
                }
        });
    if (sums[6] != 0.0 ||
        std::any_of(sums.begin(), sums.begin() + 6, [](double value) { return !std::isfinite(value); }))
    {
        throw std::invalid_argument("slow momentum energy attribution rejected an invalid tendency or moist state");
    }
    return {{
        sums[0],
        sums[1],
        sums[2],
        sums[3],
        sums[4],
        sums[5],
    }};
}

struct DeviceSlowApplyAttributionContext
{
    const SimulationStateView*                  state                  = nullptr;
    const Field3D*                              du_dt                  = nullptr;
    const Field3D*                              dv_dt                  = nullptr;
    const Field3D*                              dw_dt                  = nullptr;
    const SlowMomentumAdvectionTendencyOutputs* advection_outputs      = nullptr;
    CartesianSlowMomentumEnergyReportV1*        momentum_energy_report = nullptr;
    float                                       dt_large               = 0.0f;
    bool                                        attribute_pbl_work     = false;
    tmv::CartesianCGridMomentumFluxRegister*    momentum_register      = nullptr;
    tmv::CartesianCGridMomentumFluxRegister     momentum_candidate;
    std::array<double, 3>                       momentum_before{};
    std::array<double, 3>                       requested_impulse{};
};

static void validate_device_slow_attribution_request(const DeviceSlowApplyAttributionContext& context)
{
    if (context.state == nullptr || !std::isfinite(context.dt_large) || context.dt_large <= 0.0f)
    {
        throw std::invalid_argument("device slow attribution requires a state and finite dt > 0");
    }
    const auto&    state      = *context.state;
    const auto&    grid       = state.grid;
    const auto&    fields     = state.fields;
    const auto&    tendencies = state.tendencies;
    const Field3D* required[] = {
        &fields.u,
        &fields.v,
        &fields.w,
        &fields.p,
        &fields.rho,
        &fields.theta,
        &fields.qv,
        &fields.qc,
        &fields.qr,
        &fields.qi,
        &fields.qs,
        &fields.qg,
        &fields.qh,
        context.du_dt,
        context.dv_dt,
        context.dw_dt,
        &tendencies.du_dt_pbl,
        &tendencies.dv_dt_pbl,
    };
    if (std::any_of(std::begin(required),
                    std::end(required),
                    [&](const Field3D* field) { return field == nullptr || !field_matches_domain(*field, grid); }))
    {
        throw std::invalid_argument("device slow attribution requires grid-shaped state and tendencies");
    }
    if (context.momentum_energy_report != nullptr)
    {
        const auto*  advection = context.advection_outputs;
        const auto&  report    = *context.momentum_energy_report;
        const double prior[]   = {
            report.advection_requested_linear_work_j,
            report.nonadvection_requested_linear_work_j,
            report.advection_finite_step_correction_j,
            report.nonadvection_finite_step_correction_j,
            report.coupled_finite_step_cross_work_j,
            report.storage_rounding_j,
            report.committed_kinetic_change_j,
            report.closure_residual_j,
            report.pressure_dry_internal_change_j,
            report.pressure_water_sensible_change_j,
            report.committed_pressure_energy_change_j,
        };
        if (report.complete || advection == nullptr || !advection->complete() ||
            !field_matches_domain(*advection->du_dt, grid) || !field_matches_domain(*advection->dv_dt, grid) ||
            !field_matches_domain(*advection->dw_dt, grid) ||
            std::any_of(
                std::begin(prior), std::end(prior), [](double value) { return !std::isfinite(value) || value != 0.0; }))
        {
            throw std::invalid_argument("device slow momentum attribution received an invalid lifecycle");
        }
    }
    tmv::device_slow_energy::validate_request(state,
                                              *context.du_dt,
                                              *context.dv_dt,
                                              *context.dw_dt,
                                              context.advection_outputs,
                                              context.dt_large,
                                              context.momentum_energy_report != nullptr,
                                              context.attribute_pbl_work);
}

static void attribute_device_slow_apply_completed_v2(void*        raw_context,
                                                     const float* u_committed,
                                                     const float* v_committed,
                                                     const float* w_committed,
                                                     const float* p_committed)
{
    auto&                      context            = *static_cast<DeviceSlowApplyAttributionContext*>(raw_context);
    const SimulationStateView& execution_state    = *context.state;
    const bool                 attribute_momentum = context.momentum_energy_report != nullptr;
    const bool wrap_x = execution_state.configuration->lateral_bc_west == tmv::init::config::LateralBC::Periodic ||
                        execution_state.configuration->lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    const bool wrap_y = execution_state.configuration->lateral_bc_south == tmv::init::config::LateralBC::Periodic ||
                        execution_state.configuration->lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    const auto result = tmv::device_slow_energy::attribute_completed_commit(*context.state,
                                                                            *context.du_dt,
                                                                            *context.dv_dt,
                                                                            *context.dw_dt,
                                                                            context.advection_outputs,
                                                                            context.dt_large,
                                                                            attribute_momentum,
                                                                            context.attribute_pbl_work,
                                                                            wrap_x,
                                                                            wrap_y,
                                                                            u_committed,
                                                                            v_committed,
                                                                            w_committed,
                                                                            p_committed);
    if (context.momentum_register)
    {
        const auto& state = *context.state;
        const auto  after = context.momentum_candidate.native_dual_inventory_buffers(
            state.fields.rho,
            {u_committed, v_committed, w_committed},
            state.fields.rho.size(),
            state.grid.dr,
            state.grid.dtheta,
            state.grid.dz,
            tmv::physical_grid_metrics::active_terrain_metrics(state));
        context.momentum_candidate.attribute_operator_commit(context.requested_impulse,
                                                             {after[0] - context.momentum_before[0],
                                                              after[1] - context.momentum_before[1],
                                                              after[2] - context.momentum_before[2]});
    }
    if (context.attribute_pbl_work)
        commit_boundary_layer_momentum_work(*context.state, result.pbl);
    if (attribute_momentum)
        *context.momentum_energy_report = result.momentum;
    if (context.momentum_register)
    {
        static_assert(std::is_nothrow_move_assignable_v<tmv::CartesianCGridMomentumFluxRegister>);
        *context.momentum_register = std::move(context.momentum_candidate);
    }
}

static void apply_slow_tendencies_cpu(const SimulationStateView&                  state,
                                      double                                      dt_large,
                                      Field3D&                                    du_dt,
                                      Field3D&                                    dv_dt,
                                      Field3D&                                    dw_dt,
                                      Field3D&                                    dp_dt,
                                      bool                                        commit_pbl_work,
                                      const SlowMomentumAdvectionTendencyOutputs* advection_outputs,
                                      CartesianSlowMomentumEnergyReportV1*        momentum_energy_report)
{
    auto&       slow_pressure_energy_columns = dynamics_runtime_for(state).slow_pressure_energy_columns;
    const auto& grid                         = state.grid;
    auto&       fields                       = state.fields;
    auto&       tendency_fields              = state.tendencies;
    const BoundaryLayerRuntimeState& pbl_runtime =
        state.boundary_layer_runtime ? *state.boundary_layer_runtime : tmv::require_level_runtime(state.boundary_layer_runtime, "boundary_layer_runtime");
    const bool attribute_pbl_work = commit_pbl_work && pbl_runtime.scheme && pbl_runtime.scheme->name() != "none";
    const bool attribute_momentum_energy = commit_pbl_work && momentum_energy_report != nullptr;
    auto& runtime = dynamics_runtime_for(state);
    const TerrainMetrics3D* operator_terrain =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    const bool capture_operator_storage =
        runtime.momentum_interface_flux_capture_enabled;
    std::array<double, 3> operator_inventory_before{};
    Field3D operator_u_before;
    Field3D operator_v_before;
    Field3D operator_w_before;
    Field3D operator_p_before;
    if (capture_operator_storage)
    {
        operator_inventory_before = operator_terrain != nullptr
            ? runtime.momentum_interface_flux_register
                  .native_dual_inventory_mapped(
                      fields.rho, fields.u, fields.v, fields.w,
                      *operator_terrain)
            : runtime.momentum_interface_flux_register.native_dual_inventory(
                  fields.rho, fields.u, fields.v, fields.w,
                  grid.dr, grid.dtheta, grid.dz);
        operator_u_before = fields.u;
        operator_v_before = fields.v;
        operator_w_before = fields.w;
        operator_p_before = fields.p;
    }
    std::array<double, 6> momentum_work_terms{};
    if (attribute_momentum_energy)
    {
        const double prior_values[] = {
            momentum_energy_report->advection_requested_linear_work_j,
            momentum_energy_report->nonadvection_requested_linear_work_j,
            momentum_energy_report->advection_finite_step_correction_j,
            momentum_energy_report->nonadvection_finite_step_correction_j,
            momentum_energy_report->coupled_finite_step_cross_work_j,
            momentum_energy_report->storage_rounding_j,
            momentum_energy_report->committed_kinetic_change_j,
            momentum_energy_report->closure_residual_j,
            momentum_energy_report->pressure_dry_internal_change_j,
            momentum_energy_report->pressure_water_sensible_change_j,
            momentum_energy_report->committed_pressure_energy_change_j,
        };
        if (momentum_energy_report->complete || advection_outputs == nullptr ||
            std::any_of(std::begin(prior_values),
                        std::end(prior_values),
                        [](double value) { return !std::isfinite(value) || value != 0.0; }))
        {
            throw std::invalid_argument("slow momentum energy attribution received an invalid lifecycle");
        }
        tmv::physical_grid_metrics::validate(state);
        momentum_work_terms = slow_momentum_requested_work(state, dt_large, du_dt, dv_dt, dw_dt, *advection_outputs);
    }
    const PhysicalCellDomain physical_domain = physical_scalar_domain(grid);
    const bool               wrap_x          = tmv::require_run_configuration(state).lateral_bc_west == tmv::init::config::LateralBC::Periodic ||
                                               tmv::require_run_configuration(state).lateral_bc_east == tmv::init::config::LateralBC::Periodic;
    const bool               wrap_y          = tmv::require_run_configuration(state).lateral_bc_south == tmv::init::config::LateralBC::Periodic ||
                                               tmv::require_run_configuration(state).lateral_bc_north == tmv::init::config::LateralBC::Periodic;
    if (attribute_pbl_work)
    {
        if (!field_matches_domain(fields.rho, grid))
            throw std::invalid_argument("PBL momentum-work attribution requires grid-shaped density");
        tmv::physical_grid_metrics::validate(state);
        for (std::size_t index = 0; index < fields.rho.size(); ++index)
        {
            const double density = static_cast<double>(fields.rho.data()[index]);
            if (!std::isfinite(density) || density <= 0.0)
                throw std::invalid_argument("PBL momentum-work attribution requires positive finite density");
        }
    }
    const tmv::physical_grid_metrics::ControlVolumeMassView mass_geometry(state);
    const auto scalar_mass_kg = [&](int i, int j, int k) { return mass_geometry.scalar_mass_kg(fields.rho, i, j, k); };
    double     pressure_dry_internal_change_j   = 0.0;
    double     pressure_water_sensible_change_j = 0.0;
    long long  invalid_pressure_energy_cells    = 0;
    if (attribute_momentum_energy)
    {
        const std::size_t columns = static_cast<std::size_t>(grid.nr) * static_cast<std::size_t>(grid.nth);
        if (slow_pressure_energy_columns.size() != columns)
            slow_pressure_energy_columns.resize(columns);
    }

    const std::array<double, 3> pbl_work_terms = tmv::diagnostics::fixed_slab_reduce<3>(
        0,
        grid.nr,
        [&](int i, std::array<double, 3>& pbl_partial)
        {
            for (int j = 0; j < grid.nth; ++j)
            {
                std::array<double, 3>* pressure_partial = nullptr;
                if (attribute_momentum_energy)
                {
                    pressure_partial =
                        &slow_pressure_energy_columns[static_cast<std::size_t>(i) * static_cast<std::size_t>(grid.nth) +
                                                      static_cast<std::size_t>(j)];
                    *pressure_partial = {};
                }
                for (int k = 0; k < grid.nz; ++k)
                {
                    const double u_old       = static_cast<double>(fields.u[i][j][k]);
                    const double v_old       = static_cast<double>(fields.v[i][j][k]);
                    const double p_old       = static_cast<double>(fields.p[i][j][k]);
                    const double du_dynamics = static_cast<double>(du_dt[i][j][k]);
                    const double dv_dynamics = static_cast<double>(dv_dt[i][j][k]);
                    const double du_pbl      = static_cast<double>(tendency_fields.du_dt_pbl[i][j][k]);
                    const double dv_pbl      = static_cast<double>(tendency_fields.dv_dt_pbl[i][j][k]);
                    double       du_slow     = du_dynamics + du_pbl;
                    double       dv_slow     = dv_dynamics + dv_pbl;
                    double       dw_slow     = static_cast<double>(dw_dt[i][j][k]);
                    double       dp_slow     = static_cast<double>(dp_dt[i][j][k]);
                    if (primary_state_repairs_enabled && !std::isfinite(du_slow))
                        du_slow = 0.0;
                    if (primary_state_repairs_enabled && !std::isfinite(dv_slow))
                        dv_slow = 0.0;
                    if (primary_state_repairs_enabled && !std::isfinite(dw_slow))
                        dw_slow = 0.0;
                    if (primary_state_repairs_enabled && !std::isfinite(dp_slow))
                        dp_slow = 0.0;

                    const float u_new = static_cast<float>(u_old + du_slow * dt_large);
                    const float v_new = static_cast<float>(v_old + dv_slow * dt_large);
                    fields.u[i][j][k] = u_new;
                    fields.v[i][j][k] = v_new;
                    fields.w[i][j][k] = static_cast<float>(static_cast<double>(fields.w[i][j][k]) + dw_slow * dt_large);
                    fields.p[i][j][k] = static_cast<float>(p_old + dp_slow * dt_large);

                    const bool physical_cell = i >= physical_domain.i_begin && i < physical_domain.i_end &&
                                               j >= physical_domain.j_begin && j < physical_domain.j_end &&
                                               k >= physical_domain.k_begin && k < physical_domain.k_end;
                    if (attribute_momentum_energy && physical_cell)
                    {
                        constexpr double cv_d =
                            physical_constants::specific_heat_cp_jkgk - physical_constants::gas_constant_dry_air_jkgk;
                        constexpr double cv_v = physical_constants::specific_heat_water_vapor_cp_jkgk -
                                                physical_constants::gas_constant_water_vapor_jkgk;
                        constexpr double c_l  = physical_constants::specific_heat_liquid_water_jkgk;
                        constexpr double c_i  = physical_constants::specific_heat_ice_jkgk;
                        constexpr double kappa =
                            physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
                        const double p_stored = static_cast<double>(fields.p[i][j][k]);
                        const double theta_k  = static_cast<double>(fields.theta[i][j][k]);
                        const double qv_k     = static_cast<double>(fields.qv[i][j][k]);
                        const double q_liquid =
                            static_cast<double>(fields.qc[i][j][k]) + static_cast<double>(fields.qr[i][j][k]);
                        const double q_ice =
                            static_cast<double>(fields.qi[i][j][k]) + static_cast<double>(fields.qs[i][j][k]) +
                            static_cast<double>(fields.qg[i][j][k]) + static_cast<double>(fields.qh[i][j][k]);
                        const double dry_mass = scalar_mass_kg(i, j, k);
                        const double old_temperature =
                            theta_k * std::pow(p_old / physical_constants::reference_pressure_pa, kappa);
                        const double new_temperature =
                            theta_k * std::pow(p_stored / physical_constants::reference_pressure_pa, kappa);
                        const double temperature_change  = new_temperature - old_temperature;
                        const double water_heat_capacity = qv_k * cv_v + q_liquid * c_l + q_ice * c_i;
                        const double values[]            = {
                            p_old,
                            p_stored,
                            theta_k,
                            qv_k,
                            q_liquid,
                            q_ice,
                            dry_mass,
                            old_temperature,
                            new_temperature,
                            temperature_change,
                            water_heat_capacity,
                        };
                        if (std::any_of(std::begin(values),
                                        std::end(values),
                                        [](double value) { return !std::isfinite(value); }) ||
                            p_old <= 0.0 || p_stored <= 0.0 || theta_k <= 0.0 || qv_k < 0.0 || q_liquid < 0.0 ||
                            q_ice < 0.0 || dry_mass <= 0.0 || old_temperature <= 0.0 || new_temperature <= 0.0)
                        {
                            (*pressure_partial)[2] += 1.0;
                        }
                        else
                        {
                            (*pressure_partial)[0] += dry_mass * cv_d * temperature_change;
                            (*pressure_partial)[1] += dry_mass * water_heat_capacity * temperature_change;
                        }
                    }

                    if (attribute_pbl_work && physical_cell)
                    {
                        const double scalar_mass      = scalar_mass_kg(i, j, k);
                        const double delta_u_dynamics = du_dynamics * dt_large;
                        const double delta_v_dynamics = dv_dynamics * dt_large;
                        const double delta_u_pbl      = du_pbl * dt_large;
                        const double delta_v_pbl      = dv_pbl * dt_large;
                        const double u_after_dynamics = u_old + delta_u_dynamics;
                        const double v_after_dynamics = v_old + delta_v_dynamics;
                        const float  u_dynamics_only  = static_cast<float>(u_after_dynamics);
                        const float  v_dynamics_only  = static_cast<float>(v_after_dynamics);
                        const bool   owns_u_face = !grid.geometry.staggered || i < physical_domain.i_end - 1 || wrap_x;
                        const bool   owns_v_face = !grid.geometry.staggered || j < physical_domain.j_end - 1 || wrap_y;
                        if (owns_u_face)
                        {
                            const int    i1 = grid.geometry.staggered
                                                  ? (i < physical_domain.i_end - 1 ? i + 1 : physical_domain.i_begin)
                                                  : i;
                            const double u_mass_kg = grid.geometry.staggered
                                                         ? mass_geometry.u_dual_mass_kg(fields.rho, i, j, k, i1)
                                                         : scalar_mass;
                            pbl_partial[0] += u_mass_kg * u_after_dynamics * delta_u_pbl;
                            pbl_partial[1] += 0.5 * u_mass_kg * delta_u_pbl * delta_u_pbl;
                            pbl_partial[2] += 0.5 * u_mass_kg *
                                              (static_cast<double>(u_new) * u_new -
                                               static_cast<double>(u_dynamics_only) * u_dynamics_only);
                        }
                        if (owns_v_face)
                        {
                            const int    j1 = grid.geometry.staggered
                                                  ? (j < physical_domain.j_end - 1 ? j + 1 : physical_domain.j_begin)
                                                  : j;
                            const double v_mass_kg = grid.geometry.staggered
                                                         ? mass_geometry.v_dual_mass_kg(fields.rho, i, j, k, j1)
                                                         : scalar_mass;
                            pbl_partial[0] += v_mass_kg * v_after_dynamics * delta_v_pbl;
                            pbl_partial[1] += 0.5 * v_mass_kg * delta_v_pbl * delta_v_pbl;
                            pbl_partial[2] += 0.5 * v_mass_kg *
                                              (static_cast<double>(v_new) * v_new -
                                               static_cast<double>(v_dynamics_only) * v_dynamics_only);
                        }
                    }
                }
            }
        });

    if (capture_operator_storage)
    {
        try
        {
            const auto requested_impulse = operator_terrain != nullptr
                ? runtime.momentum_interface_flux_register
                      .native_dual_tendency_impulse_mapped(
                          fields.rho, du_dt, dv_dt, dw_dt, dt_large,
                          *operator_terrain)
                : runtime.momentum_interface_flux_register
                      .native_dual_tendency_impulse(
                          fields.rho, du_dt, dv_dt, dw_dt, dt_large,
                          grid.dr, grid.dtheta, grid.dz);
            const auto operator_inventory_after = operator_terrain != nullptr
                ? runtime.momentum_interface_flux_register
                      .native_dual_inventory_mapped(
                          fields.rho, fields.u, fields.v, fields.w,
                          *operator_terrain)
                : runtime.momentum_interface_flux_register.native_dual_inventory(
                      fields.rho, fields.u, fields.v, fields.w,
                      grid.dr, grid.dtheta, grid.dz);
            runtime.momentum_interface_flux_register.attribute_operator_commit(
                requested_impulse,
                {operator_inventory_after[0] - operator_inventory_before[0],
                 operator_inventory_after[1] - operator_inventory_before[1],
                 operator_inventory_after[2] - operator_inventory_before[2]});
        }
        catch (...)
        {
            std::swap(fields.u, operator_u_before);
            std::swap(fields.v, operator_v_before);
            std::swap(fields.w, operator_w_before);
            std::swap(fields.p, operator_p_before);
            throw;
        }
    }

    if (attribute_momentum_energy)
    {
        for (const auto& partial : slow_pressure_energy_columns)
        {
            pressure_dry_internal_change_j += partial[0];
            pressure_water_sensible_change_j += partial[1];
            invalid_pressure_energy_cells += static_cast<long long>(partial[2]);
        }
    }

    if (attribute_pbl_work)
    {
        BoundaryLayerMomentumWorkReport report;
        report.requested_linear_work_j  = pbl_work_terms[0];
        report.finite_step_correction_j = pbl_work_terms[1];
        report.committed_work_j         = pbl_work_terms[2];
        report.storage_rounding_j       = pbl_work_terms[2] - pbl_work_terms[0] - pbl_work_terms[1];
        commit_boundary_layer_momentum_work(state, report);
    }
    if (attribute_momentum_energy)
    {
        if (invalid_pressure_energy_cells != 0 || !std::isfinite(pressure_dry_internal_change_j) ||
            !std::isfinite(pressure_water_sensible_change_j))
        {
            throw std::runtime_error("slow pressure energy attribution rejected a nonphysical "
                                     "basis-v2 state");
        }
        CartesianSlowMomentumEnergyReportV1 report;
        report.advection_requested_linear_work_j     = momentum_work_terms[0];
        report.nonadvection_requested_linear_work_j  = momentum_work_terms[1];
        report.advection_finite_step_correction_j    = momentum_work_terms[2];
        report.nonadvection_finite_step_correction_j = momentum_work_terms[3];
        report.coupled_finite_step_cross_work_j      = momentum_work_terms[4];
        report.committed_kinetic_change_j            = momentum_work_terms[5];
        const double requested_commit =
            report.advection_requested_linear_work_j + report.nonadvection_requested_linear_work_j +
            report.advection_finite_step_correction_j + report.nonadvection_finite_step_correction_j +
            report.coupled_finite_step_cross_work_j;
        report.storage_rounding_j = report.committed_kinetic_change_j - requested_commit;
        report.closure_residual_j = report.committed_kinetic_change_j - (requested_commit + report.storage_rounding_j);
        report.pressure_dry_internal_change_j   = pressure_dry_internal_change_j;
        report.pressure_water_sensible_change_j = pressure_water_sensible_change_j;
        report.committed_pressure_energy_change_j =
            report.pressure_dry_internal_change_j + report.pressure_water_sensible_change_j;
        report.complete         = true;
        *momentum_energy_report = report;
    }
}

static bool try_apply_slow_tendencies_gpu(const SimulationStateView&                  state,
                                          double                                      dt_large,
                                          Field3D&                                    du_dt,
                                          Field3D&                                    dv_dt,
                                          Field3D&                                    dw_dt,
                                          Field3D&                                    dp_dt,
                                          bool                                        commit_pbl_work,
                                          const SlowMomentumAdvectionTendencyOutputs* advection_outputs,
                                          CartesianSlowMomentumEnergyReportV1*        momentum_energy_report)
{
    const auto& grid            = state.grid;
    auto&       fields          = state.fields;
    auto&       tendency_fields = state.tendencies;
    const BoundaryLayerRuntimeState& pbl_runtime =
        state.boundary_layer_runtime ? *state.boundary_layer_runtime : tmv::require_level_runtime(state.boundary_layer_runtime, "boundary_layer_runtime");
    const bool        active_pbl = pbl_runtime.scheme && pbl_runtime.scheme->name() != "none";
    const std::size_t cell_count = fields.u.size();
    if (!gpu_slow_apply_enabled(state) || !has_compute_context(state) || cell_count == 0 ||
        tendency_fields.du_dt_pbl.size() != cell_count || tendency_fields.dv_dt_pbl.size() != cell_count)
    {
        return false;
    }

    DeviceSlowApplyAttributionContext attribution_context;
    auto&                             dynamics = dynamics_runtime_for(state);
    const bool observe_completed_commit        = dynamics.momentum_interface_flux_capture_enabled ||
                                                 (commit_pbl_work && (active_pbl || momentum_energy_report != nullptr));
    if (observe_completed_commit)
    {
        attribution_context.state                  = &state;
        attribution_context.du_dt                  = &du_dt;
        attribution_context.dv_dt                  = &dv_dt;
        attribution_context.dw_dt                  = &dw_dt;
        attribution_context.advection_outputs      = advection_outputs;
        attribution_context.momentum_energy_report = commit_pbl_work ? momentum_energy_report : nullptr;
        attribution_context.dt_large               = static_cast<float>(dt_large);
        attribution_context.attribute_pbl_work     = commit_pbl_work && active_pbl;
        validate_device_slow_attribution_request(attribution_context);
        if (dynamics.momentum_interface_flux_capture_enabled)
        {
            auto& candidate                       = attribution_context.momentum_candidate;
            candidate                             = dynamics.momentum_interface_flux_register;
            attribution_context.momentum_register = &dynamics.momentum_interface_flux_register;
            const auto* terrain                   = tmv::physical_grid_metrics::active_terrain_metrics(state);
            attribution_context.momentum_before =
                candidate.native_dual_inventory_buffers(fields.rho,
                                                        {fields.u.data(), fields.v.data(), fields.w.data()},
                                                        cell_count,
                                                        grid.dr,
                                                        grid.dtheta,
                                                        grid.dz,
                                                        terrain);
            attribution_context.requested_impulse =
                terrain
                    ? candidate.native_dual_tendency_impulse_mapped(fields.rho, du_dt, dv_dt, dw_dt, dt_large, *terrain)
                    : candidate.native_dual_tendency_impulse(
                          fields.rho, du_dt, dv_dt, dw_dt, dt_large, grid.dr, grid.dtheta, grid.dz);
            (void)candidate.pending_committed_operator_impulse();
        }
    }

    const auto try_gpu = [&]()
    {
        return dispatch_slow_apply_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
                                           fields.u.data(),
                                           fields.v.data(),
                                           fields.w.data(),
                                           fields.p.data(),
                                           du_dt.data(),
                                           dv_dt.data(),
                                           dw_dt.data(),
                                           dp_dt.data(),
                                           tendency_fields.du_dt_pbl.data(),
                                           tendency_fields.dv_dt_pbl.data(),
                                           grid.nr,
                                           grid.nth,
                                           grid.nz,
                                           static_cast<float>(dt_large),
                                           observe_completed_commit ? &attribute_device_slow_apply_completed_v2
                                                                    : nullptr,
                                           observe_completed_commit ? &attribution_context : nullptr,
                                           primary_state_repairs_enabled);
    };

    // A report observer consumes the unique old/result pair and may also
    // commit the PBL counterfactual. Do not CPU-replay and double-publish it;
    // the report itself compares requested device currency with the actual
    // committed buffers.
    if (observe_completed_commit || !nimbus::parity::sample("slow_apply"))
    {
        return try_gpu();
    }

    Field3D pre_u = fields.u, pre_v = fields.v, pre_w = fields.w, pre_p = fields.p;
    if (!try_gpu())
    {
        return false;
    }

    Field3D           gpu_u = fields.u, gpu_v = fields.v, gpu_w = fields.w, gpu_p = fields.p;
    const std::size_t bytes = cell_count * sizeof(float);
    std::memcpy(fields.u.data(), pre_u.data(), bytes);
    std::memcpy(fields.v.data(), pre_v.data(), bytes);
    std::memcpy(fields.w.data(), pre_w.data(), bytes);
    std::memcpy(fields.p.data(), pre_p.data(), bytes);

    apply_slow_tendencies_cpu(
        state, dt_large, du_dt, dv_dt, dw_dt, dp_dt, commit_pbl_work, advection_outputs, momentum_energy_report);

    using nimbus::parity::Tier;
    const char* kn = "slow_apply";
    nimbus::parity::check_field(kn, "u", fields.u.data(), gpu_u.data(), grid.nr, grid.nth, grid.nz, Tier::Pointwise);
    nimbus::parity::check_field(kn, "v", fields.v.data(), gpu_v.data(), grid.nr, grid.nth, grid.nz, Tier::Pointwise);
    nimbus::parity::check_field(kn, "w", fields.w.data(), gpu_w.data(), grid.nr, grid.nth, grid.nz, Tier::Pointwise);
    nimbus::parity::check_field(kn, "p", fields.p.data(), gpu_p.data(), grid.nr, grid.nth, grid.nz, Tier::Pointwise);

    std::memcpy(fields.u.data(), gpu_u.data(), bytes);
    std::memcpy(fields.v.data(), gpu_v.data(), bytes);
    std::memcpy(fields.w.data(), gpu_w.data(), bytes);
    std::memcpy(fields.p.data(), gpu_p.data(), bytes);
    return true;
}

static void apply_slow_tendencies(const SimulationStateView&                  state,
                                  double                                      dt_large,
                                  Field3D&                                    du_dt,
                                  Field3D&                                    dv_dt,
                                  Field3D&                                    dw_dt,
                                  Field3D&                                    dp_dt,
                                  bool                                        commit_pbl_work,
                                  const SlowMomentumAdvectionTendencyOutputs* advection_outputs,
                                  CartesianSlowMomentumEnergyReportV1*        momentum_energy_report)
{
    if (!try_apply_slow_tendencies_gpu(
            state, dt_large, du_dt, dv_dt, dw_dt, dp_dt, commit_pbl_work, advection_outputs, momentum_energy_report))
    {
        if (state.compute_runtime != nullptr &&
            state.compute_runtime->active_kind != ComputeBackendKind::Cpu &&
            !state.compute_runtime->fallback_active && !state.compute_runtime->config.allow_fallback &&
            gpu_slow_apply_enabled(state))
            throw std::runtime_error("fallback-off slow state application dispatch failed");
        apply_slow_tendencies_cpu(
            state, dt_large, du_dt, dv_dt, dw_dt, dp_dt, commit_pbl_work, advection_outputs, momentum_energy_report);
    }
}

enum class AcousticEnergyLayout
{
    FlatAGrid,
    TerrainAGrid,
    FlatCGrid,
    TerrainCGrid,
};

struct AcousticEnergyPending
{
    std::array<double, 6>                 continuity_eos_changes{};
    double                                continuity_carrier_requested_change_j = 0.0;
    CartesianCGridAcousticEnergyReportV1* report                                = nullptr;
    AcousticEnergyLayout                  layout                                = AcousticEnergyLayout::TerrainCGrid;
    bool                                  initial_state_validated               = false;
    bool                                  pressure_ready                        = false;
};

static std::array<double, 6>
acoustic_moist_v2_cell_reservoirs(const SimulationStateView& state,
                                  const TerrainMetrics3D*    metrics,
                                  AcousticEnergyLayout       layout,
                                  int                        i,
                                  int                        j,
                                  int                        k,
                                  double                     rho_d,
                                  double                     pressure,
                                  const Field3D*             u_override      = nullptr,
                                  const Field3D*             v_override      = nullptr,
                                  const Field3D*             w_override      = nullptr,
                                  double                     volume_override = std::numeric_limits<double>::quiet_NaN(),
                                  double                     height_override = std::numeric_limits<double>::quiet_NaN())
{
    const auto&      fields     = state.fields;
    const Field3D&   velocity_u = u_override != nullptr ? *u_override : fields.u;
    const Field3D&   velocity_v = v_override != nullptr ? *v_override : fields.v;
    const Field3D&   velocity_w = w_override != nullptr ? *w_override : fields.w;
    constexpr double cv_d = physical_constants::specific_heat_cp_jkgk - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l   = physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i   = physical_constants::specific_heat_ice_jkgk;
    constexpr double t_ref = physical_constants::freezing_temperature_k;
    constexpr double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
    constexpr double ice_reference = -physical_constants::latent_heat_fusion_jkg;
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;

    const double qv_k     = static_cast<double>(fields.qv[i][j][k]);
    const double qc_k     = static_cast<double>(fields.qc[i][j][k]);
    const double qr_k     = static_cast<double>(fields.qr[i][j][k]);
    const double qi_k     = static_cast<double>(fields.qi[i][j][k]);
    const double qs_k     = static_cast<double>(fields.qs[i][j][k]);
    const double qg_k     = static_cast<double>(fields.qg[i][j][k]);
    const double qh_k     = static_cast<double>(fields.qh[i][j][k]);
    const double tke_k    = static_cast<double>(fields.tke[i][j][k]);
    const double theta_k  = static_cast<double>(fields.theta[i][j][k]);
    const double q_liquid = qc_k + qr_k;
    const double q_ice    = qi_k + qs_k + qg_k + qh_k;
    const double q_total  = qv_k + q_liquid + q_ice;
    const double volume =
        std::isfinite(volume_override) ? volume_override : tmv::physical_grid_metrics::cell_volume(state, i, j, k);
    const double dry_mass     = rho_d * volume;
    const double mixture_mass = dry_mass * (1.0 + q_total);
    const double temperature  = theta_k * std::pow(pressure / physical_constants::reference_pressure_pa, kappa);
    const double delta_t      = temperature - t_ref;

    double kinetic = 0.0;
    if (layout == AcousticEnergyLayout::FlatAGrid
        || layout == AcousticEnergyLayout::TerrainAGrid)
    {
        const double u_k = static_cast<double>(velocity_u[i][j][k]);
        const double v_k = static_cast<double>(velocity_v[i][j][k]);
        const double w_k = static_cast<double>(velocity_w[i][j][k]);
        kinetic          = 0.5 * mixture_mass * (u_k * u_k + v_k * v_k + w_k * w_k);
    }
    else
    {
        if (layout == AcousticEnergyLayout::TerrainCGrid && metrics == nullptr)
            throw std::logic_error("terrain acoustic energy attribution requires mapped metrics");
        const double u_w = static_cast<double>(velocity_u[i - 1][j][k]);
        const double u_e = static_cast<double>(velocity_u[i][j][k]);
        const double v_s = static_cast<double>(velocity_v[i][j - 1][k]);
        const double v_n = static_cast<double>(velocity_v[i][j][k]);
        double       w_b = 0.0;
        if (k == 0 && layout == AcousticEnergyLayout::TerrainCGrid)
        {
            const double ground_u   = 0.5 * (u_w + u_e);
            const double ground_v   = 0.5 * (v_s + v_n);
            const double frame_u_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms;
            const double frame_v_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms;
            w_b                     = (ground_u - frame_u_ms) * (-metrics->Gx_face(i, j, 0)) +
                                      (ground_v - frame_v_ms) * (-metrics->Gy_face(i, j, 0));
        }
        else if (k == 0)
        {
            w_b = 0.0;
        }
        else
        {
            w_b = static_cast<double>(velocity_w[i][j][k - 1]);
        }
        const double w_t = static_cast<double>(velocity_w[i][j][k]);
        kinetic = 0.25 * mixture_mass * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n + w_b * w_b + w_t * w_t);
    }

    const std::array<double, 6> result{{
        dry_mass * cv_d * delta_t,
        dry_mass * delta_t * (qv_k * cv_v + q_liquid * c_l + q_ice * c_i),
        dry_mass * (qv_k * vapor_reference + q_ice * ice_reference),
        kinetic,
        mixture_mass * physical_constants::gravity_ms2 *
            (std::isfinite(height_override) ? height_override
                                            : tmv::physical_grid_metrics::scalar_height(state, i, j, k)),
        dry_mass * tke_k,
    }};
    return result;
}

static double flat_grid_cell_volume(const SimulationStateView& state, int k)
{
    const auto&  grid  = state.grid;
    const double lower = grid.geometry.z_faces.empty() ? static_cast<double>(k) * grid.dz
                                                       : grid.geometry.z_faces[static_cast<std::size_t>(k)];
    const double upper = grid.geometry.z_faces.empty() ? static_cast<double>(k + 1) * grid.dz
                                                       : grid.geometry.z_faces[static_cast<std::size_t>(k + 1)];
    return grid.dr * grid.dtheta * (upper - lower);
}

static double flat_grid_scalar_height(const SimulationStateView& state, int k)
{
    const auto& grid = state.grid;
    return grid.geometry.z.empty() ? (static_cast<double>(k) + 0.5) * grid.dz
                                   : grid.geometry.z[static_cast<std::size_t>(k)];
}

static double acoustic_mixture_mass_kg(const SimulationStateView& state, int i, int j, int k)
{
    const auto&  fields  = state.fields;
    const double q_total = static_cast<double>(fields.qv[i][j][k]) + static_cast<double>(fields.qc[i][j][k]) +
                           static_cast<double>(fields.qr[i][j][k]) + static_cast<double>(fields.qi[i][j][k]) +
                           static_cast<double>(fields.qs[i][j][k]) + static_cast<double>(fields.qg[i][j][k]) +
                           static_cast<double>(fields.qh[i][j][k]);
    const double volume  = tmv::physical_grid_metrics::cell_volume(state, i, j, k);
    return static_cast<double>(fields.rho[i][j][k]) * (1.0 + q_total) * volume;
}

static double flat_acoustic_mixture_mass_kg(const SimulationStateView& state, int i, int j, int k)
{
    const auto&  fields  = state.fields;
    const double q_total = static_cast<double>(fields.qv[i][j][k]) + static_cast<double>(fields.qc[i][j][k]) +
                           static_cast<double>(fields.qr[i][j][k]) + static_cast<double>(fields.qi[i][j][k]) +
                           static_cast<double>(fields.qs[i][j][k]) + static_cast<double>(fields.qg[i][j][k]) +
                           static_cast<double>(fields.qh[i][j][k]);
    return static_cast<double>(fields.rho[i][j][k]) * (1.0 + q_total) * flat_grid_cell_volume(state, k);
}

static std::array<double, 4> flat_cgrid_device_momentum_cell_attribution(const SimulationStateView& state,
                                                                         const PhysicalCellDomain&  domain,
                                                                         const Field3D&             u_before,
                                                                         const Field3D&             v_before,
                                                                         const Field3D&             w_before,
                                                                         int                        i,
                                                                         int                        j,
                                                                         int                        k,
                                                                         float                      dt_small,
                                                                         float                      inv_dx,
                                                                         float                      inv_dy,
                                                                         float                      inv_dz)
{
    const int  nx     = state.grid.nr;
    const int  ny     = state.grid.nth;
    const int  nz     = state.grid.nz;
    const bool active = i >= 1 && i <= nx - 2 && j >= 1 && j <= ny - 2 && k >= 0 && k <= nz - 2;
    if (!active)
        return {};

    const auto& rho              = state.fields.rho;
    const auto& p                = state.fields.p;
    const auto& u                = state.fields.u;
    const auto& v                = state.fields.v;
    const auto& w                = state.fields.w;
    const auto& p0_base          = state.base.p0;
    const auto  safe_rho_f       = [](float value) { return std::isfinite(value) && value > 1.0e-6f ? value : 1.0f; };
    const auto  finite_or_zero_f = [](float value) { return std::isfinite(value) ? value : 0.0f; };

    const float rho_x  = 0.5f * (rho[i][j][k] + rho[i + 1][j][k]);
    const float rho_y  = 0.5f * (rho[i][j][k] + rho[i][j + 1][k]);
    const float rho_z  = 0.5f * (rho[i][j][k] + rho[i][j][k + 1]);
    const float dp_dx  = (p[i + 1][j][k] - p[i][j][k]) * inv_dx;
    const float dp_dy  = (p[i][j + 1][k] - p[i][j][k]) * inv_dy;
    const float dp_dz  = (p[i][j][k + 1] - p[i][j][k]) * inv_dz;
    const float dp0_dz = static_cast<float>(
        (p0_base[static_cast<std::size_t>(k + 1)] - p0_base[static_cast<std::size_t>(k)]) / state.grid.dz);
    float du = finite_or_zero_f(-dp_dx / safe_rho_f(rho_x));
    float dv = finite_or_zero_f(-dp_dy / safe_rho_f(rho_y));
    float dw = finite_or_zero_f(-(dp_dz - dp0_dz) / safe_rho_f(rho_z));

    if (!primary_state_repairs_enabled)
    {
        const auto safe_density = [](double value)
        {
            return std::isfinite(value) && value > 1e-6 ? value : 1.0;
        };
        const double base = (p0_base[static_cast<std::size_t>(k + 1)] - p0_base[static_cast<std::size_t>(k)]) / state.grid.dz;
        const float base_high = static_cast<float>(base);
        const float base_low = static_cast<float>(base - static_cast<double>(base_high));
        du = static_cast<float>(-(static_cast<double>(p[i + 1][j][k]) - p[i][j][k]) /
             static_cast<float>(state.grid.dr) / safe_density(0.5 * (static_cast<double>(rho[i][j][k]) + rho[i + 1][j][k])));
        dv = static_cast<float>(-(static_cast<double>(p[i][j + 1][k]) - p[i][j][k]) /
             static_cast<float>(state.grid.dtheta) / safe_density(0.5 * (static_cast<double>(rho[i][j][k]) + rho[i][j + 1][k])));
        dw = static_cast<float>(-((static_cast<double>(p[i][j][k + 1]) - p[i][j][k]) /
             static_cast<float>(state.grid.dz) - (static_cast<double>(base_high) + base_low)) /
             safe_density(0.5 * (static_cast<double>(rho[i][j][k]) + rho[i][j][k + 1])));
    }

    const double old_u   = static_cast<double>(u_before[i][j][k]);
    const double old_v   = static_cast<double>(v_before[i][j][k]);
    const double old_w   = static_cast<double>(w_before[i][j][k]);
    const double delta_u = primary_state_repairs_enabled ? static_cast<double>(du * dt_small)
                                                         : static_cast<double>(du) * dt_small;
    const double delta_v = primary_state_repairs_enabled ? static_cast<double>(dv * dt_small)
                                                         : static_cast<double>(dv) * dt_small;
    const double delta_w = primary_state_repairs_enabled ? static_cast<double>(dw * dt_small)
                                                         : static_cast<double>(dw) * dt_small;

    const double center_mass = flat_acoustic_mixture_mass_kg(state, i, j, k);
    double       u_mass      = 0.5 * center_mass;
    if (i + 1 < domain.i_end)
        u_mass += 0.5 * flat_acoustic_mixture_mass_kg(state, i + 1, j, k);
    double v_mass = 0.5 * center_mass;
    if (j + 1 < domain.j_end)
        v_mass += 0.5 * flat_acoustic_mixture_mass_kg(state, i, j + 1, k);
    double w_mass = 0.0;
    if (k + 1 < domain.k_end)
    {
        w_mass = 0.5 * (center_mass + flat_acoustic_mixture_mass_kg(state, i, j, k + 1));
    }

    const double stored_u = static_cast<double>(u[i][j][k]);
    const double stored_v = static_cast<double>(v[i][j][k]);
    const double stored_w = static_cast<double>(w[i][j][k]);
    const double values[] = {
        u_mass,
        v_mass,
        w_mass,
        old_u,
        old_v,
        old_w,
        delta_u,
        delta_v,
        delta_w,
        stored_u,
        stored_v,
        stored_w,
    };
    if (std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }) ||
        u_mass <= 0.0 || v_mass <= 0.0 || w_mass <= 0.0)
    {
        return {{0.0, 0.0, 0.0, 1.0}};
    }

    return {{
        u_mass * old_u * delta_u + v_mass * old_v * delta_v + w_mass * old_w * delta_w,
        0.5 * (u_mass * delta_u * delta_u + v_mass * delta_v * delta_v + w_mass * delta_w * delta_w),
        0.5 * (u_mass * (stored_u * stored_u - old_u * old_u) + v_mass * (stored_v * stored_v - old_v * old_v) +
               w_mass * (stored_w * stored_w - old_w * old_w)),
        0.0,
    }};
}

static void finalize_flat_cgrid_device_momentum_attribution(const std::array<double, 4>& sums,
                                                            AcousticEnergyPending&       pending)
{
    const double requested        = sums[0];
    const double finite_step      = sums[1];
    const double committed        = sums[2];
    const double storage_rounding = committed - requested - finite_step;
    const double inventory =
        std::accumulate(pending.continuity_eos_changes.begin(), pending.continuity_eos_changes.end(), 0.0) + committed;
    const double values[] = {
        requested,
        finite_step,
        committed,
        storage_rounding,
        inventory,
    };
    if (sums[3] != 0.0 ||
        std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }) ||
        finite_step < 0.0)
    {
        throw std::runtime_error("device C-grid pressure-work attribution produced invalid "
                                 "basis-v2 state");
    }

    CartesianCGridAcousticEnergyReportV1 next    = *pending.report;
    const auto&                          changes = pending.continuity_eos_changes;
    next.continuity_eos_dry_internal_change_j += changes[0];
    next.continuity_eos_water_sensible_change_j += changes[1];
    next.continuity_eos_phase_reference_change_j += changes[2];
    next.continuity_carrier_kinetic_change_j += changes[3];
    next.continuity_gravitational_change_j += changes[4];
    next.continuity_subgrid_tke_change_j += changes[5];
    next.continuity_carrier_requested_change_j += pending.continuity_carrier_requested_change_j;
    next.pressure_gradient_requested_linear_work_j += requested;
    next.pressure_gradient_finite_step_correction_j += finite_step;
    next.pressure_gradient_storage_rounding_j += storage_rounding;
    next.pressure_gradient_committed_work_j += committed;
    next.committed_inventory_change_j += inventory;
    ++next.substep_count;
    next.complete                    = true;
    const double cumulative_values[] = {
        next.continuity_eos_dry_internal_change_j,
        next.continuity_eos_water_sensible_change_j,
        next.continuity_eos_phase_reference_change_j,
        next.continuity_carrier_kinetic_change_j,
        next.continuity_gravitational_change_j,
        next.continuity_subgrid_tke_change_j,
        next.continuity_carrier_requested_change_j,
        next.pressure_gradient_requested_linear_work_j,
        next.pressure_gradient_finite_step_correction_j,
        next.pressure_gradient_storage_rounding_j,
        next.pressure_gradient_committed_work_j,
        next.boundary_committed_work_j,
        next.boundary_radiative_requested_linear_work_j,
        next.boundary_radiative_finite_step_correction_j,
        next.boundary_radiative_storage_rounding_j,
        next.boundary_radiative_committed_work_j,
        next.boundary_constraint_history_committed_work_j,
        next.committed_inventory_change_j,
    };
    if (std::any_of(std::begin(cumulative_values),
                    std::end(cumulative_values),
                    [](double value) { return !std::isfinite(value); }))
    {
        throw std::overflow_error("device C-grid acoustic attribution overflowed its report");
    }
    *pending.report                               = next;
    pending.continuity_eos_changes                = {};
    pending.continuity_carrier_requested_change_j = 0.0;
    pending.pressure_ready                        = false;
}

static void attribute_flat_cgrid_device_pressure_commit(const SimulationStateView& state,
                                                        const Field3D&             rho_before,
                                                        const Field3D&             p_before,
                                                        float                      dt_small,
                                                        float                      dx,
                                                        float                      dy,
                                                        float                      dz,
                                                        AcousticEnergyPending&     pending,
                                                        const Field3D*             pressure_commit_u = nullptr,
                                                        const Field3D*             pressure_commit_v = nullptr,
                                                        const Field3D*             pressure_commit_w = nullptr)
{
    if (pending.layout != AcousticEnergyLayout::FlatCGrid || pending.report == nullptr || pending.pressure_ready ||
        !(dt_small > 0.0f) || !(dx > 0.0f) || !(dy > 0.0f) || !(dz > 0.0f))
    {
        throw std::logic_error("device C-grid pressure attribution lifecycle is invalid");
    }
    if (!pending.initial_state_validated)
    {
        (void)compute_moist_mixture_energy_v2(state);
        pending.initial_state_validated = true;
    }

    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const Field3D&           u      = pressure_commit_u != nullptr ? *pressure_commit_u : state.fields.u;
    const Field3D&           v      = pressure_commit_v != nullptr ? *pressure_commit_v : state.fields.v;
    const Field3D&           w      = pressure_commit_w != nullptr ? *pressure_commit_w : state.fields.w;
    const int                nx     = state.grid.nr;
    const int                ny     = state.grid.nth;
    const int                nz     = state.grid.nz;
    const bool               fused_momentum_attribution =
        pressure_commit_u != nullptr && pressure_commit_v != nullptr && pressure_commit_w != nullptr;
    if ((pressure_commit_u != nullptr || pressure_commit_v != nullptr || pressure_commit_w != nullptr) &&
        !fused_momentum_attribution)
    {
        throw std::invalid_argument("fused device C-grid attribution requires all velocity "
                                    "snapshots");
    }
    const float inv_dx = 1.0f / dx;
    const float inv_dy = 1.0f / dy;
    const float inv_dz = 1.0f / dz;
    const auto  sums   = tmv::diagnostics::fixed_slab_reduce<12>(
        domain.i_begin,
        domain.i_end,
        [&](int i, std::array<double, 12>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double flat_volume = flat_grid_cell_volume(state, k);
                    const double flat_height = flat_grid_scalar_height(state, k);
                    const auto   before = acoustic_moist_v2_cell_reservoirs(state,
                                                                            nullptr,
                                                                            AcousticEnergyLayout::FlatCGrid,
                                                                            i,
                                                                            j,
                                                                            k,
                                                                            static_cast<double>(rho_before[i][j][k]),
                                                                            static_cast<double>(p_before[i][j][k]),
                                                                            pressure_commit_u,
                                                                            pressure_commit_v,
                                                                            pressure_commit_w,
                                                                            flat_volume,
                                                                            flat_height);
                    // Both reservoirs use the same composition and pre-momentum velocities.
                    const bool same_thermodynamics =
                        std::memcmp(&rho_before(i, j, k), &state.fields.rho(i, j, k), sizeof(float)) == 0 &&
                        std::memcmp(&p_before(i, j, k), &state.fields.p(i, j, k), sizeof(float)) == 0;
                    const auto after = same_thermodynamics ? before : acoustic_moist_v2_cell_reservoirs(state,
                                                                         nullptr,
                                                                         AcousticEnergyLayout::FlatCGrid,
                                                                         i,
                                                                         j,
                                                                         k,
                                                                         static_cast<double>(state.fields.rho[i][j][k]),
                                                                         static_cast<double>(state.fields.p[i][j][k]),
                                                                         pressure_commit_u,
                                                                         pressure_commit_v,
                                                                         pressure_commit_w,
                                                                         flat_volume,
                                                                         flat_height);
                    bool       valid = true;
                    for (std::size_t n = 0; n < before.size(); ++n)
                    {
                        valid = valid && std::isfinite(before[n]) && std::isfinite(after[n]);
                        partial[n] += after[n] - before[n];
                    }

                    // Reproduce the selected device shader's FP32
                    // conservative continuity request from the pre-commit
                    // density and still-unmodified face velocities. The
                    // committed reservoir delta alone contains limiter and
                    // storage effects and cannot stand in for this
                    // operator-time carrier request.
                    const float frame_u = static_cast<float>(state.domain_frame->velocity_ms.u_ms);
                    const float frame_v = static_cast<float>(state.domain_frame->velocity_ms.v_ms);
                    const double requested_density_change = primary_state_repairs_enabled
                        ? static_cast<double>(tmv::device_acoustic_energy::requested_density_change(
                            rho_before, u, v, w, nx, ny, nz, i, j, k, dt_small, dx, dy, dz, frame_u, frame_v))
                        : static_cast<double>(tmv::device_acoustic_energy::density_tendency_precise(
                            rho_before, u, v, w, nz, i, j, k, dx, dy, dz, frame_u, frame_v)) * dt_small;
                    const double volume              = flat_volume;
                    const double dry_mass_before     = static_cast<double>(rho_before[i][j][k]) * volume;
                    const double carrier_coefficient = before[3] / dry_mass_before;
                    const double requested_carrier_change =
                        static_cast<double>(requested_density_change) * volume * carrier_coefficient;
                    valid = valid && std::isfinite(volume) && volume > 0.0 && std::isfinite(dry_mass_before) &&
                            dry_mass_before > 0.0 && std::isfinite(carrier_coefficient) &&
                            std::isfinite(requested_carrier_change);
                    if (valid)
                        partial[6] += requested_carrier_change;
                    else
                        partial[7] += 1.0;

                    if (fused_momentum_attribution)
                    {
                        const auto momentum = flat_cgrid_device_momentum_cell_attribution(state,
                                                                                          domain,
                                                                                          *pressure_commit_u,
                                                                                          *pressure_commit_v,
                                                                                          *pressure_commit_w,
                                                                                          i,
                                                                                          j,
                                                                                          k,
                                                                                          dt_small,
                                                                                          inv_dx,
                                                                                          inv_dy,
                                                                                          inv_dz);
                        for (std::size_t n = 0; n < momentum.size(); ++n)
                            partial[8 + n] += momentum[n];
                    }
                }
        });

    if (sums[7] != 0.0 || !std::isfinite(sums[6]) ||
        std::any_of(sums.begin(), sums.begin() + 6, [](double value) { return !std::isfinite(value); }))
    {
        throw std::runtime_error("device C-grid continuity/EOS attribution produced invalid "
                                 "basis-v2 state");
    }
    pending.continuity_eos_changes                = {{
        sums[0],
        sums[1],
        sums[2],
        sums[3],
        sums[4],
        sums[5],
    }};
    pending.continuity_carrier_requested_change_j = sums[6];
    pending.pressure_ready                        = true;
    if (fused_momentum_attribution)
    {
        finalize_flat_cgrid_device_momentum_attribution({{sums[8], sums[9], sums[10], sums[11]}}, pending);
    }
}

static void attribute_flat_cgrid_device_momentum_commit(const SimulationStateView& state,
                                                        const Field3D&             u_before,
                                                        const Field3D&             v_before,
                                                        const Field3D&             w_before,
                                                        float                      dt_small,
                                                        float                      dx,
                                                        float                      dy,
                                                        float                      dz,
                                                        AcousticEnergyPending&     pending)
{
    if (pending.layout != AcousticEnergyLayout::FlatCGrid || pending.report == nullptr || !pending.pressure_ready ||
        !(dt_small > 0.0f) || !(dx > 0.0f) || !(dy > 0.0f) || !(dz > 0.0f))
    {
        throw std::logic_error("device C-grid momentum attribution lifecycle is invalid");
    }

    const auto& p0_base = state.base.p0;
    const int   nz      = state.grid.nz;
    if (static_cast<int>(p0_base.size()) < nz)
    {
        throw std::logic_error("device C-grid momentum attribution requires the base pressure");
    }

    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const float              inv_dx = 1.0f / dx;
    const float              inv_dy = 1.0f / dy;
    const float              inv_dz = 1.0f / dz;

    const auto sums = tmv::diagnostics::fixed_slab_reduce<4>(
        domain.i_begin,
        domain.i_end,
        [&](int i, std::array<double, 4>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const auto cell = flat_cgrid_device_momentum_cell_attribution(
                        state, domain, u_before, v_before, w_before, i, j, k, dt_small, inv_dx, inv_dy, inv_dz);
                    for (std::size_t n = 0; n < cell.size(); ++n)
                        partial[n] += cell[n];
                }
        });
    finalize_flat_cgrid_device_momentum_attribution(sums, pending);
}

struct FlatAgridPressureObserverContext
{
    const SimulationStateView* state    = nullptr;
    const Field3D*             drho_dt  = nullptr;
    AcousticEnergyPending*     pending  = nullptr;
    double                     dt_small = 0.0;
};

static void observe_flat_agrid_pressure_commit(void* opaque, const float* rho_candidate, const float* p_candidate)
{
    auto&       context = *static_cast<FlatAgridPressureObserverContext*>(opaque);
    const auto& state   = *context.state;
    auto&       pending = *context.pending;
    if (pending.layout != AcousticEnergyLayout::FlatAGrid || pending.report == nullptr || pending.pressure_ready ||
        rho_candidate == nullptr || p_candidate == nullptr || context.drho_dt == nullptr ||
        !std::isfinite(context.dt_small) || context.dt_small <= 0.0)
    {
        throw std::logic_error("native A-grid pressure attribution lifecycle is invalid");
    }
    if (!pending.initial_state_validated)
    {
        (void)compute_moist_mixture_energy_v2(state);
        pending.initial_state_validated = true;
    }
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const int                ny     = state.grid.nth;
    const int                nz     = state.grid.nz;
    const auto               sums   = tmv::diagnostics::fixed_slab_reduce<8>(
        domain.i_begin,
        domain.i_end,
        [&](int i, std::array<double, 8>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t n          = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                    const double      rho_before = state.fields.rho.data()[n];
                    const auto        before     = acoustic_moist_v2_cell_reservoirs(
                        state, nullptr, AcousticEnergyLayout::FlatAGrid, i, j, k, rho_before, state.fields.p.data()[n]);
                    const auto after = acoustic_moist_v2_cell_reservoirs(
                        state, nullptr, AcousticEnergyLayout::FlatAGrid, i, j, k, rho_candidate[n], p_candidate[n]);
                    bool valid = true;
                    for (std::size_t reservoir = 0; reservoir < before.size(); ++reservoir)
                    {
                        valid = valid && std::isfinite(before[reservoir]) && std::isfinite(after[reservoir]);
                        partial[reservoir] += after[reservoir] - before[reservoir];
                    }
                    const double volume              = tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                    const double dry_mass_before     = rho_before * volume;
                    const double carrier_coefficient = before[3] / dry_mass_before;
                    const double requested = context.dt_small * static_cast<double>(context.drho_dt->data()[n]) *
                                             volume * carrier_coefficient;
                    valid = valid && std::isfinite(volume) && volume > 0.0 && std::isfinite(dry_mass_before) &&
                            dry_mass_before > 0.0 && std::isfinite(carrier_coefficient) && std::isfinite(requested);
                    if (valid)
                        partial[6] += requested;
                    else
                        partial[7] += 1.0;
                }
        });
    if (sums[7] != 0.0 || !std::isfinite(sums[6]) ||
        std::any_of(sums.begin(), sums.begin() + 6, [](double value) { return !std::isfinite(value); }))
    {
        throw std::runtime_error("native A-grid continuity/EOS attribution produced invalid basis-v2 state");
    }
    pending.continuity_eos_changes                = {{sums[0], sums[1], sums[2], sums[3], sums[4], sums[5]}};
    pending.continuity_carrier_requested_change_j = sums[6];
    pending.pressure_ready                        = true;
}

struct FlatAgridMomentumObserverContext
{
    const SimulationStateView* state   = nullptr;
    AcousticEnergyPending*     pending = nullptr;
};

static void observe_flat_agrid_momentum_commit(void*        opaque,
                                               const float* u_candidate,
                                               const float* v_candidate,
                                               const float* w_candidate)
{
    auto&       context = *static_cast<FlatAgridMomentumObserverContext*>(opaque);
    const auto& state   = *context.state;
    auto&       pending = *context.pending;
    if (pending.layout != AcousticEnergyLayout::FlatAGrid || pending.report == nullptr || !pending.pressure_ready ||
        u_candidate == nullptr || v_candidate == nullptr || w_candidate == nullptr)
    {
        throw std::logic_error("native A-grid momentum attribution lifecycle is invalid");
    }
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const int                ny     = state.grid.nth;
    const int                nz     = state.grid.nz;
    const auto               sums   = tmv::diagnostics::fixed_slab_reduce<4>(
        domain.i_begin,
        domain.i_end,
        [&](int i, std::array<double, 4>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t n        = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                    const double      mass     = acoustic_mixture_mass_kg(state, i, j, k);
                    const double      old_u    = state.fields.u.data()[n];
                    const double      old_v    = state.fields.v.data()[n];
                    const double      old_w    = state.fields.w.data()[n];
                    const double      delta_u  = static_cast<double>(u_candidate[n]) - old_u;
                    const double      delta_v  = static_cast<double>(v_candidate[n]) - old_v;
                    const double      delta_w  = static_cast<double>(w_candidate[n]) - old_w;
                    const double      values[] = {
                        mass,
                        old_u,
                        old_v,
                        old_w,
                        delta_u,
                        delta_v,
                        delta_w,
                        static_cast<double>(u_candidate[n]),
                        static_cast<double>(v_candidate[n]),
                        static_cast<double>(w_candidate[n]),
                    };
                    if (mass <= 0.0 || std::any_of(std::begin(values),
                                                   std::end(values),
                                                   [](double value) { return !std::isfinite(value); }))
                    {
                        partial[3] += 1.0;
                        continue;
                    }
                    partial[0] += mass * (old_u * delta_u + old_v * delta_v + old_w * delta_w);
                    partial[1] += 0.5 * mass * (delta_u * delta_u + delta_v * delta_v + delta_w * delta_w);
                    partial[2] += 0.5 * mass *
                                  (static_cast<double>(u_candidate[n]) * u_candidate[n] - old_u * old_u +
                                   static_cast<double>(v_candidate[n]) * v_candidate[n] - old_v * old_v +
                                   static_cast<double>(w_candidate[n]) * w_candidate[n] - old_w * old_w);
                }
        });
    finalize_flat_cgrid_device_momentum_attribution(sums, pending);
}

static void apply_fast_pressure_cpu(const SimulationStateView& state,
                                    SplitExplicitDynamics&     split_scheme,
                                    double                     dt_small,
                                    Field3D&                   drho_dt,
                                    Field3D&                   dp_dt,
                                    AcousticEnergyPending*     acoustic_energy_pending = nullptr,
                                    bool                       compute_tendencies      = true)
{
    auto&                   u               = state.fields.u;
    auto&                   v               = state.fields.v;
    auto&                   w               = state.fields.w;
    auto&                   rho             = state.fields.rho;
    auto&                   p               = state.fields.p;
    auto&                   theta           = state.fields.theta;
    const auto&             rho0_base       = state.base.rho0;
    const TerrainMetrics3D* terrain_metrics = tmv::physical_grid_metrics::active_terrain_metrics(state);
    auto& dynamics_runtime = dynamics_runtime_for(state);
    const bool capture_momentum_continuity_storage =
        dynamics_runtime.momentum_interface_flux_capture_enabled;
    std::array<double, 3> momentum_inventory_before{};
    Field3D rho_before;
    Field3D p_before;
    if (capture_momentum_continuity_storage)
    {
        momentum_inventory_before = terrain_metrics != nullptr
            ? dynamics_runtime.momentum_interface_flux_register
                  .native_dual_inventory_mapped(
                      rho, u, v, w, *terrain_metrics)
            : dynamics_runtime.momentum_interface_flux_register.native_dual_inventory(
                  rho, u, v, w,
                  state.grid.dr, state.grid.dtheta, state.grid.dz);
        rho_before = rho;
        p_before = p;
    }
    if (acoustic_energy_pending != nullptr)
    {
        const bool terrain_layout =
            acoustic_energy_pending->layout == AcousticEnergyLayout::TerrainCGrid
            || acoustic_energy_pending->layout == AcousticEnergyLayout::TerrainAGrid;
        const bool geometry_mismatch = terrain_layout
            ? terrain_metrics == nullptr : terrain_metrics != nullptr;
        if (geometry_mismatch || acoustic_energy_pending->report == nullptr || acoustic_energy_pending->pressure_ready)
        {
            throw std::logic_error("acoustic pressure attribution lifecycle is invalid");
        }
        const auto&  report          = *acoustic_energy_pending->report;
        const double report_values[] = {
            report.continuity_eos_dry_internal_change_j,
            report.continuity_eos_water_sensible_change_j,
            report.continuity_eos_phase_reference_change_j,
            report.continuity_carrier_kinetic_change_j,
            report.continuity_gravitational_change_j,
            report.continuity_subgrid_tke_change_j,
            report.continuity_carrier_requested_change_j,
            report.pressure_gradient_requested_linear_work_j,
            report.pressure_gradient_finite_step_correction_j,
            report.pressure_gradient_storage_rounding_j,
            report.pressure_gradient_committed_work_j,
            report.boundary_committed_work_j,
            report.boundary_radiative_requested_linear_work_j,
            report.boundary_radiative_finite_step_correction_j,
            report.boundary_radiative_storage_rounding_j,
            report.boundary_radiative_committed_work_j,
            report.boundary_constraint_history_committed_work_j,
            report.committed_inventory_change_j,
        };
        if (std::any_of(std::begin(report_values),
                        std::end(report_values),
                        [](double value) { return !std::isfinite(value); }) ||
            report.pressure_gradient_finite_step_correction_j < 0.0 ||
            report.boundary_radiative_finite_step_correction_j < 0.0 || report.complete != (report.substep_count > 0) ||
            !report.boundary_complete)
        {
            throw std::invalid_argument("acoustic attribution received an invalid report");
        }
        if (!acoustic_energy_pending->initial_state_validated)
        {
            // One fail-closed full-basis validation before the first
            // attributed substep. Every subsequent commit below verifies its
            // own output, so validation is inductive rather than repeated.
            (void)compute_moist_mixture_energy_v2(state);
            acoustic_energy_pending->initial_state_validated = true;
        }
    }
    if (compute_tendencies)
    {
        split_scheme.compute_fast_pressure_tendencies(u, v, w, rho, p, dt_small, drho_dt, dp_dt);
    }

    const int                nr                   = state.grid.nr;
    const int                nth                  = state.grid.nth;
    const int                nz                   = state.grid.nz;
    const PhysicalCellDomain physical_domain      = physical_scalar_domain(state.grid);
    const bool reuse_pressure_material = acoustic_energy_pending != nullptr &&
        acoustic_energy_pending->layout == AcousticEnergyLayout::FlatCGrid && terrain_metrics == nullptr &&
        state.grid.geometry.coord == CoordinateSystem::Cartesian && state.grid.geometry.staggered;
    const auto               update_pressure_slab = [&](int i, std::array<double, 8>* partial)
    {
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const bool            attribute_cell = partial != nullptr && i >= physical_domain.i_begin &&
                                                       i < physical_domain.i_end && j >= physical_domain.j_begin &&
                                                       j < physical_domain.j_end && k >= physical_domain.k_begin &&
                                                       k < physical_domain.k_end;
                std::array<double, 6> before{};
                tmv::pressure_energy_detail::CellMaterial material{};
                const double          rho_before_value = static_cast<double>(rho[i][j][k]);
                if (attribute_cell)
                {
                    if (reuse_pressure_material)
                        material = tmv::pressure_energy_detail::prepare(state, i, j, k);
                    before = reuse_pressure_material
                        ? tmv::pressure_energy_detail::reservoirs(material, rho_before_value, static_cast<double>(p[i][j][k]))
                        : acoustic_moist_v2_cell_reservoirs(state,
                                                               terrain_metrics,
                                                               acoustic_energy_pending->layout,
                                                               i,
                                                               j,
                                                               k,
                                                               rho_before_value,
                                                               static_cast<double>(p[i][j][k]));
                }
                double drho_d = static_cast<double>(drho_dt[i][j][k]);
                if (primary_state_repairs_enabled && !std::isfinite(drho_d))
                    drho_d = 0.0;
                double rho_new = static_cast<double>(rho[i][j][k]) + drho_d * dt_small;
                if (primary_state_repairs_enabled && (!std::isfinite(rho_new) || rho_new <= 0.0))
                    rho_new = std::max(0.1, rho0_base[k]);
                const float rho_stored = clamp_density_kgm3(static_cast<float>(rho_new));
                rho[i][j][k]           = rho_stored;

                double       p_new = std::numeric_limits<double>::quiet_NaN();
                const double th    = static_cast<double>(theta[i][j][k]);
                if (th > 0.0)
                {
                    p_new = dynamics_constants::p0 * std::pow(static_cast<double>(rho_stored) *
                                                                  dynamics_constants::R_d * th / dynamics_constants::p0,
                                                              dynamics_constants::gamma);
                }
                if (primary_state_repairs_enabled && (!std::isfinite(p_new) || p_new <= 0.0))
                    p_new = p0;
                p[i][j][k] = clamp_pressure_pa(static_cast<float>(p_new));
                if (attribute_cell)
                {
                    const std::array<double, 6> after =
                        reuse_pressure_material
                        ? tmv::pressure_energy_detail::reservoirs(material, static_cast<double>(rho[i][j][k]), static_cast<double>(p[i][j][k]))
                        : acoustic_moist_v2_cell_reservoirs(state,
                                                          terrain_metrics,
                                                          acoustic_energy_pending->layout,
                                                          i,
                                                          j,
                                                          k,
                                                          static_cast<double>(rho[i][j][k]),
                                                          static_cast<double>(p[i][j][k]));
                    bool valid = true;
                    for (std::size_t n = 0; n < before.size(); ++n)
                        valid = valid && std::isfinite(before[n]) && std::isfinite(after[n]);
                    const double volume                   = reuse_pressure_material ? material.volume
                        : tmv::physical_grid_metrics::cell_volume(state, i, j, k);
                    const double dry_mass_before          = rho_before_value * volume;
                    const double carrier_coefficient      = before[3] / dry_mass_before;
                    const double requested_carrier_change = dt_small * drho_d * volume * carrier_coefficient;
                    valid = valid && std::isfinite(volume) && volume > 0.0 && std::isfinite(dry_mass_before) &&
                            dry_mass_before > 0.0 && std::isfinite(carrier_coefficient) &&
                            std::isfinite(requested_carrier_change);
                    if (!valid)
                    {
                        (*partial)[7] += 1.0;
                        continue;
                    }
                    for (std::size_t n = 0; n < before.size(); ++n)
                        (*partial)[n] += after[n] - before[n];
                    (*partial)[6] += requested_carrier_change;
                }
            }
    };
    std::array<double, 8> pressure_sums{};
    if (acoustic_energy_pending != nullptr)
    {
        pressure_sums = tmv::diagnostics::fixed_slab_reduce<8>(
            0, nr, [&](int i, std::array<double, 8>& partial) { update_pressure_slab(i, &partial); });
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nr; ++i)
            update_pressure_slab(i, nullptr);
    }
    if (acoustic_energy_pending != nullptr)
    {
        const std::array<double, 6> changes{{
            pressure_sums[0],
            pressure_sums[1],
            pressure_sums[2],
            pressure_sums[3],
            pressure_sums[4],
            pressure_sums[5],
        }};
        if (pressure_sums[7] != 0.0 || !std::isfinite(pressure_sums[6]) ||
            std::any_of(changes.begin(), changes.end(), [](double value) { return !std::isfinite(value); }))
        {
            throw std::runtime_error("acoustic continuity/EOS attribution produced "
                                     "invalid basis-v2 state");
        }
        acoustic_energy_pending->continuity_eos_changes                = changes;
        acoustic_energy_pending->continuity_carrier_requested_change_j = pressure_sums[6];
        acoustic_energy_pending->pressure_ready                        = true;
    }
    if (capture_momentum_continuity_storage)
    {
        try
        {
            const auto requested_impulse = terrain_metrics != nullptr
                ? dynamics_runtime.momentum_interface_flux_register
                      .native_dual_density_tendency_impulse_mapped(
                          rho_before, drho_dt, u, v, w, dt_small,
                          *terrain_metrics)
                : dynamics_runtime.momentum_interface_flux_register
                      .native_dual_density_tendency_impulse(
                          rho_before, drho_dt, u, v, w, dt_small,
                          state.grid.dr, state.grid.dtheta, state.grid.dz);
            const auto momentum_inventory_after = terrain_metrics != nullptr
                ? dynamics_runtime.momentum_interface_flux_register
                      .native_dual_inventory_mapped(
                          rho, u, v, w, *terrain_metrics)
                : dynamics_runtime.momentum_interface_flux_register.native_dual_inventory(
                      rho, u, v, w,
                      state.grid.dr, state.grid.dtheta, state.grid.dz);
            dynamics_runtime.momentum_interface_flux_register
                .attribute_continuity_commit(
                    requested_impulse,
                    {momentum_inventory_after[0] - momentum_inventory_before[0],
                     momentum_inventory_after[1] - momentum_inventory_before[1],
                     momentum_inventory_after[2] - momentum_inventory_before[2]});
        }
        catch (...)
        {
            std::swap(rho, rho_before);
            std::swap(p, p_before);
            throw;
        }
    }
}

static void attribute_native_cgrid_continuity(const SimulationStateView& state,
                                             const Field3D& rho_before,
                                             double dt_small)
{
    auto& flux = dynamics_runtime_for(state).momentum_interface_flux_register;
    const auto& u = state.fields.u;
    const auto& v = state.fields.v;
    const auto& w = state.fields.w;
    const auto& grid = state.grid;
    const auto before = flux.native_dual_inventory(rho_before, u, v, w, grid.dr, grid.dtheta, grid.dz);
    Field3D density_rate(grid.nr, grid.nth, grid.nz, 0.0f);
    const auto& frame = tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms;
    for (int i = 1; i < grid.nr - 1; ++i)
        for (int j = 1; j < grid.nth - 1; ++j)
            for (int k = 0; k < grid.nz; ++k)
                density_rate[i][j][k] = primary_state_repairs_enabled
                    ? tmv::device_acoustic_energy::density_tendency(
                        rho_before, u, v, w, grid.nr, grid.nth, grid.nz, i, j, k,
                        static_cast<float>(grid.dr), static_cast<float>(grid.dtheta), static_cast<float>(grid.dz),
                        static_cast<float>(frame.u_ms), static_cast<float>(frame.v_ms))
                    : tmv::device_acoustic_energy::density_tendency_precise(
                        rho_before, u, v, w, grid.nz, i, j, k,
                        static_cast<float>(grid.dr), static_cast<float>(grid.dtheta), static_cast<float>(grid.dz),
                        static_cast<float>(frame.u_ms), static_cast<float>(frame.v_ms));
    const auto requested = flux.native_dual_density_tendency_impulse(
        rho_before, density_rate, u, v, w, static_cast<float>(dt_small), grid.dr, grid.dtheta, grid.dz);
    const auto after = flux.native_dual_inventory(state.fields.rho, u, v, w, grid.dr, grid.dtheta, grid.dz);
    flux.attribute_continuity_commit(requested, {after[0] - before[0], after[1] - before[1], after[2] - before[2]});
}

static bool try_apply_fast_pressure_gpu(const SimulationStateView& state,
                                        double                     dt_small,
                                        bool                       is_cgrid,
                                        bool                       flat_terrain,
                                        float                      dr_f,
                                        float                      dtheta_f,
                                        float                      dz_f)
{
    auto&       u         = state.fields.u;
    auto&       v         = state.fields.v;
    auto&       w         = state.fields.w;
    auto&       rho       = state.fields.rho;
    auto&       p         = state.fields.p;
    auto&       theta     = state.fields.theta;
    const auto& rho0_base = state.base.rho0;
    const auto& p0_base   = state.base.p0;
    const int   NR        = state.grid.nr;
    const int   NTH       = state.grid.nth;
    const int   NZ        = state.grid.nz;
    if (!active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), DynamicsKernelOp::AcousticPressureCartesianCgrid) || !is_cgrid ||
        !flat_terrain || static_cast<int>(p0_base.size()) < NZ || static_cast<int>(rho0_base.size()) < NZ)
    {
        return false;
    }

    const AcousticProfileRows profiles = acoustic_profile_rows(state, NZ);
    fill_acoustic_base_profiles(rho0_base,
                                p0_base,
                                NZ,
                                state.grid.dz,
                                profiles,
                                /*fill_rho0=*/true,
                                /*fill_p0=*/true,
                                /*fill_dp0_dz=*/false);

    return dispatch_acoustic_pressure_cartesian_cgrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), u.data(),
                                                              v.data(),
                                                              w.data(),
                                                              rho.data(),
                                                              p.data(),
                                                              profiles.rho0,
                                                              profiles.p0,
                                                              NZ,
                                                              theta.data(),
                                                              rho.data(),
                                                              p.data(),
                                                              NR,
                                                              NTH,
                                                              NZ,
                                                              dr_f,
                                                              dtheta_f,
                                                              dz_f,
                                                              dynamics_constants::gamma,
                                                              static_cast<float>(dt_small),
                                                              density_min_kgm3,
                                                              density_max_kgm3,
                                                              pressure_min_pa,
                                                              pressure_max_pa,
                                                              static_cast<float>(p0),
                                                              static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.u_ms),
                                                              static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.v_ms),
                                                              primary_state_repairs_enabled);
}

static bool try_compute_fast_pressure_agrid_gpu(const SimulationStateView& state,
                                                double                     dt_small,
                                                Field3D&                   drho_dt,
                                                Field3D&                   mass_flux_x,
                                                Field3D&                   mass_flux_y,
                                                Field3D&                   mass_flux_z)
{
    const int nr  = state.grid.nr;
    const int nth = state.grid.nth;
    const int nz  = state.grid.nz;
    if (nr < 4 || nth < 4 || nz < 4 || tmv::physical_grid_metrics::terrain_is_active(state) ||
        static_cast<int>(state.base.p0.size()) < nz ||
        !active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid))
    {
        return false;
    }
    const float dt_f = static_cast<float>(dt_small);
    if (!std::isfinite(dt_f) || dt_f <= 0.0f)
    {
        return false;
    }

    const AcousticProfileRows profiles = acoustic_profile_rows(state, nz);
    fill_acoustic_base_profiles(state.base.rho0,
                                state.base.p0,
                                nz,
                                state.grid.dz,
                                profiles,
                                /*fill_rho0=*/false,
                                /*fill_p0=*/true,
                                /*fill_dp0_dz=*/false);
    return dispatch_acoustic_continuity_mass_flux_cartesian_agrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
        state.fields.u.data(),
        state.fields.v.data(),
        state.fields.w.data(),
        state.fields.rho.data(),
        state.fields.p.data(),
        profiles.p0,
        nz,
        drho_dt.data(),
        mass_flux_x.data(),
        mass_flux_y.data(),
        mass_flux_z.data(),
        nr,
        nth,
        nz,
        static_cast<float>(state.grid.dr),
        static_cast<float>(state.grid.dtheta),
        static_cast<float>(state.grid.dz),
        dt_f,
        static_cast<int>(tmv::require_run_configuration(state).lateral_bc_west),
        static_cast<int>(tmv::require_run_configuration(state).lateral_bc_east),
        static_cast<int>(tmv::require_run_configuration(state).lateral_bc_south),
        static_cast<int>(tmv::require_run_configuration(state).lateral_bc_north),
        static_cast<int>(tmv::require_run_configuration(state).top_bc),
        /*accumulate_mass_flux=*/false,
        0.0f,
        static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.u_ms),
        static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.v_ms));
}

static bool has_zero_flat_agrid_boundary_flux(const SimulationStateView& state,
                                              const Field3D&             mass_flux_x,
                                              const Field3D&             mass_flux_y,
                                              const Field3D&             mass_flux_z)
{
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    if (domain.empty() || domain.i_begin < 1 || domain.j_begin < 1 || domain.k_begin < 1)
        return false;

    const int west   = domain.i_begin - 1;
    const int east   = domain.i_end - 1;
    const int south  = domain.j_begin - 1;
    const int north  = domain.j_end - 1;
    const int bottom = domain.k_begin - 1;
    const int top    = domain.k_end - 1;
    for (int j = domain.j_begin; j < domain.j_end; ++j)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
            if (mass_flux_x[west][j][k] != 0.0f || mass_flux_x[east][j][k] != 0.0f)
                return false;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int k = domain.k_begin; k < domain.k_end; ++k)
            if (mass_flux_y[i][south][k] != 0.0f || mass_flux_y[i][north][k] != 0.0f)
                return false;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            if (mass_flux_z[i][j][bottom] != 0.0f || mass_flux_z[i][j][top] != 0.0f)
                return false;
    return true;
}

static void apply_fast_pressure(const SimulationStateView&           state,
                                SplitExplicitDynamics&               split_scheme,
                                double                               dt_small,
                                Field3D&                             drho_dt,
                                Field3D&                             dp_dt,
                                bool                                 is_cgrid,
                                bool                                 flat_terrain,
                                float                                dr_f,
                                float                                dtheta_f,
                                float                                dz_f,
                                tmv::CartesianCGridTransportContext* transport_context,
                                bool                                 attribute_carrier_kinetic_transport,
                                AcousticEnergyPending*               acoustic_energy_pending)
{
    auto& compute = tmv::require_level_runtime(state.compute_runtime, "compute runtime");
    const bool native_flat_cgrid = is_cgrid && flat_terrain &&
        active_compute_backend_kind(compute) != ComputeBackendKind::Cpu;
    const bool capture_native_continuity = native_flat_cgrid &&
        dynamics_runtime_for(state).momentum_interface_flux_capture_enabled;
    const bool attribute_flat_cgrid_device =
        acoustic_energy_pending != nullptr && acoustic_energy_pending->layout == AcousticEnergyLayout::FlatCGrid;
    if (attribute_flat_cgrid_device)
    {
        if (acoustic_energy_pending->report == nullptr || acoustic_energy_pending->pressure_ready)
        {
            throw std::logic_error("device C-grid pressure attribution lifecycle is invalid");
        }
        const auto&  report          = *acoustic_energy_pending->report;
        const double report_values[] = {
            report.continuity_eos_dry_internal_change_j,
            report.continuity_eos_water_sensible_change_j,
            report.continuity_eos_phase_reference_change_j,
            report.continuity_carrier_kinetic_change_j,
            report.continuity_gravitational_change_j,
            report.continuity_subgrid_tke_change_j,
            report.continuity_carrier_requested_change_j,
            report.pressure_gradient_requested_linear_work_j,
            report.pressure_gradient_finite_step_correction_j,
            report.pressure_gradient_storage_rounding_j,
            report.pressure_gradient_committed_work_j,
            report.boundary_committed_work_j,
            report.boundary_radiative_requested_linear_work_j,
            report.boundary_radiative_finite_step_correction_j,
            report.boundary_radiative_storage_rounding_j,
            report.boundary_radiative_committed_work_j,
            report.boundary_constraint_history_committed_work_j,
            report.committed_inventory_change_j,
        };
        if (std::any_of(std::begin(report_values),
                        std::end(report_values),
                        [](double value) { return !std::isfinite(value); }) ||
            report.pressure_gradient_finite_step_correction_j < 0.0 ||
            report.boundary_radiative_finite_step_correction_j < 0.0 || report.complete != (report.substep_count > 0) ||
            !report.boundary_complete)
        {
            throw std::invalid_argument("device C-grid acoustic attribution received an invalid "
                                        "report");
        }
        if (!acoustic_energy_pending->initial_state_validated)
        {
            // Fail before the device can mutate state. Subsequent commits
            // validate their exact operator outputs inductively.
            (void)compute_moist_mixture_energy_v2(state);
            acoustic_energy_pending->initial_state_validated = true;
        }
        const std::size_t bytes = state.fields.rho.size() * sizeof(float);
        std::memcpy(drho_dt.data(), state.fields.rho.data(), bytes);
        std::memcpy(dp_dt.data(), state.fields.p.data(), bytes);
    }
    else if (native_flat_cgrid && (transport_context != nullptr || capture_native_continuity))
    {
        std::memcpy(drho_dt.data(), state.fields.rho.data(), state.fields.rho.size() * sizeof(float));
        if (capture_native_continuity)
            std::memcpy(dp_dt.data(), state.fields.p.data(), state.fields.p.size() * sizeof(float));
    }

    const auto accumulate_transport =
        [&](const Field3D& density_state, double transport_dt, tmv::AcousticMassFluxPrecision precision)
    {
        // Sample the explicitly supplied pre-continuity density. Ordinary
        // paths call this before mutation; the attributed device path calls
        // it after a successful dispatch with its existing rho snapshot.
        const tmv::cartesian_mass_flux::CollocatedPressureCoupling pressure_coupling{
            &state.fields.p, &state.base.p0, dt_small, state.grid.dr, state.grid.dtheta, state.grid.dz};
        const tmv::CarrierKineticBoundaryInputs carrier_inputs{{
            &state.fields.qv,
            &state.fields.qc,
            &state.fields.qr,
            &state.fields.qi,
            &state.fields.qs,
            &state.fields.qg,
            &state.fields.qh,
        }};
        transport_context->accumulate(state, density_state,
                                      state.fields.u,
                                      state.fields.v,
                                      state.fields.w,
                                      transport_dt,
                                      is_cgrid ? nullptr : &pressure_coupling,
                                      attribute_carrier_kinetic_transport ? &carrier_inputs : nullptr,
                                      precision);
    };

    const bool native_agrid_required =
        !is_cgrid && flat_terrain && active_compute_backend_kind(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) != ComputeBackendKind::Cpu &&
        !compute_backend_fallback_active(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) && !tmv::require_level_runtime(state.compute_runtime, "compute_runtime").config.allow_fallback &&
        state.configuration != nullptr && state.configuration->gpu_apply_slow;
    if (!is_cgrid && flat_terrain && transport_context != nullptr)
    {
        Field3D mass_flux_x(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
        Field3D mass_flux_y(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
        Field3D mass_flux_z(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
        if (try_compute_fast_pressure_agrid_gpu(state, dt_small, drho_dt, mass_flux_x, mass_flux_y, mass_flux_z))
        {
            if (acoustic_energy_pending != nullptr &&
                !has_zero_flat_agrid_boundary_flux(state, mass_flux_x, mass_flux_y, mass_flux_z))
            {
                throw std::runtime_error("native flat A-grid rigid boundary exported nonzero mass flux");
            }
            FlatAgridPressureObserverContext observer_context{&state, &drho_dt, acoustic_energy_pending, dt_small};
            const bool                       attributed     = acoustic_energy_pending != nullptr;
            const Field3D                    density_before = state.fields.rho;
            const bool                       eos_dispatched = dispatch_acoustic_pressure_eos_cartesian_agrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
                state.fields.rho.data(),
                state.fields.theta.data(),
                drho_dt.data(),
                state.fields.rho.data(),
                state.fields.p.data(),
                state.grid.nr,
                state.grid.nth,
                state.grid.nz,
                static_cast<float>(dt_small),
                static_cast<float>(dynamics_constants::p0),
                static_cast<float>(dynamics_constants::R_d),
                static_cast<float>(dynamics_constants::gamma),
                density_min_kgm3,
                density_max_kgm3,
                pressure_min_pa,
                pressure_max_pa,
                attributed ? observe_flat_agrid_pressure_commit : nullptr,
                attributed ? &observer_context : nullptr);
            if (eos_dispatched)
            {
                const float dt_f = static_cast<float>(dt_small);
                transport_context->capture_density_begin(density_before);
                if (attributed)
                    transport_context->record_external_carrier_kinetic_transport(0.0);
                const float weight = dt_f / static_cast<float>(transport_context->dt_large());
                for (std::size_t n = 0; n < mass_flux_x.size(); ++n)
                {
                    transport_context->mass_flux_x.data()[n] += weight * mass_flux_x.data()[n];
                    transport_context->mass_flux_y.data()[n] += weight * mass_flux_y.data()[n];
                    transport_context->mass_flux_z.data()[n] += weight * mass_flux_z.data()[n];
                }
                transport_context->record_external_accumulation(static_cast<double>(dt_f),
                                                                1,
                                                                /*fp32_time_currency=*/true);
                std::fill(dp_dt.data(), dp_dt.data() + dp_dt.size(), 0.0f);
                return;
            }
        }
        if (native_agrid_required)
            throw std::runtime_error("fallback-off native A-grid continuity/EOS dispatch failed");
    }
    if (transport_context != nullptr && !attribute_flat_cgrid_device && !native_flat_cgrid)
        accumulate_transport(state.fields.rho, dt_small, tmv::AcousticMassFluxPrecision::Host);

    const bool dispatched = try_apply_fast_pressure_gpu(state, dt_small, is_cgrid, flat_terrain, dr_f, dtheta_f, dz_f);
    if (dispatched)
    {
        if (transport_context != nullptr && (attribute_flat_cgrid_device || native_flat_cgrid))
        {
            // Retain the selected face arithmetic before any density rounding or momentum update.
            accumulate_transport(drho_dt, static_cast<double>(static_cast<float>(dt_small)),
                                 primary_state_repairs_enabled ? tmv::AcousticMassFluxPrecision::NativeFloat
                                                               : tmv::AcousticMassFluxPrecision::NativeCompensated);
        }
        if (attribute_flat_cgrid_device)
        {
            attribute_flat_cgrid_device_pressure_commit(
                state, drho_dt, dp_dt, static_cast<float>(dt_small), dr_f, dtheta_f, dz_f, *acoustic_energy_pending);
        }
        if (capture_native_continuity)
        {
            try
            {
                attribute_native_cgrid_continuity(state, drho_dt, dt_small);
            }
            catch (...)
            {
                std::memcpy(state.fields.rho.data(), drho_dt.data(), drho_dt.size() * sizeof(float));
                std::memcpy(state.fields.p.data(), dp_dt.data(), dp_dt.size() * sizeof(float));
                invalidate_resident_field_backend(compute, "rho");
                invalidate_resident_field_backend(compute, "p");
                throw;
            }
        }
        return;
    }
    else
    {
        if (native_flat_cgrid && !compute.config.allow_fallback)
        {
            throw std::runtime_error("fallback-off native C-grid pressure dispatch failed");
        }
        if (transport_context != nullptr && (attribute_flat_cgrid_device || native_flat_cgrid))
            accumulate_transport(state.fields.rho, dt_small, tmv::AcousticMassFluxPrecision::Host);
        apply_fast_pressure_cpu(state, split_scheme, dt_small, drho_dt, dp_dt, acoustic_energy_pending);
    }
}

static void apply_fast_momentum_cpu(const SimulationStateView& state,
                                    SplitExplicitDynamics&     split_scheme,
                                    double                     dt_small,
                                    Field3D&                   du_dt,
                                    Field3D&                   dv_dt,
                                    Field3D&                   dw_dt,
                                    AcousticEnergyPending*     acoustic_energy_pending = nullptr)
{
    auto& u   = state.fields.u;
    auto& v   = state.fields.v;
    auto& w   = state.fields.w;
    auto& rho = state.fields.rho;
    auto& p   = state.fields.p;
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    auto& runtime = dynamics_runtime_for(state);
    const bool capture_operator_storage =
        runtime.momentum_interface_flux_capture_enabled;
    std::array<double, 3> operator_inventory_before{};
    Field3D operator_u_before;
    Field3D operator_v_before;
    Field3D operator_w_before;
    if (capture_operator_storage)
    {
        operator_inventory_before = terrain_metrics != nullptr
            ? runtime.momentum_interface_flux_register
                  .native_dual_inventory_mapped(
                      rho, u, v, w, *terrain_metrics)
            : runtime.momentum_interface_flux_register.native_dual_inventory(
                  rho, u, v, w,
                  state.grid.dr, state.grid.dtheta, state.grid.dz);
        operator_u_before = u;
        operator_v_before = v;
        operator_w_before = w;
    }
    split_scheme.compute_fast_momentum_tendencies(
        u, v, w, rho, p, dt_small, du_dt, dv_dt, dw_dt);

    const int               nr              = state.grid.nr;
    const int               nth             = state.grid.nth;
    const int               nz              = state.grid.nz;
    const bool              terrain_layout =
        acoustic_energy_pending != nullptr && acoustic_energy_pending->layout == AcousticEnergyLayout::TerrainCGrid;
    const bool cgrid_layout =
        acoustic_energy_pending != nullptr
        && acoustic_energy_pending->layout != AcousticEnergyLayout::FlatAGrid
        && acoustic_energy_pending->layout != AcousticEnergyLayout::TerrainAGrid;
    const bool any_terrain_layout = acoustic_energy_pending != nullptr
        && (acoustic_energy_pending->layout == AcousticEnergyLayout::TerrainCGrid
            || acoustic_energy_pending->layout == AcousticEnergyLayout::TerrainAGrid);
    const bool geometry_mismatch = acoustic_energy_pending != nullptr &&
        (any_terrain_layout ? terrain_metrics == nullptr : terrain_metrics != nullptr);
    if (acoustic_energy_pending != nullptr &&
        (geometry_mismatch || acoustic_energy_pending->report == nullptr || !acoustic_energy_pending->pressure_ready))
    {
        throw std::logic_error("acoustic momentum attribution lifecycle is invalid");
    }
    const PhysicalCellDomain physical_domain = physical_scalar_domain(state.grid);
    std::array<double, 3>    ground_before_sums{};
    if (terrain_layout)
    {
        ground_before_sums = tmv::diagnostics::fixed_slab_reduce<3>(
            physical_domain.i_begin,
            physical_domain.i_end,
            [&](int i, std::array<double, 3>& partial)
            {
                for (int j = physical_domain.j_begin; j < physical_domain.j_end; ++j)
                {
                    const double mass    = acoustic_mixture_mass_kg(state, i, j, physical_domain.k_begin);
                    const double slope_x = -terrain_metrics->Gx_face(i, j, 0);
                    const double slope_y = -terrain_metrics->Gy_face(i, j, 0);
                    const double ground_u =
                        0.5 * (static_cast<double>(u[i - 1][j][0]) + static_cast<double>(u[i][j][0]));
                    const double ground_v =
                        0.5 * (static_cast<double>(v[i][j - 1][0]) + static_cast<double>(v[i][j][0]));
                    const double frame_u_ms =
                        tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms;
                    const double frame_v_ms =
                        tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms;
                    const double ground_w = (ground_u - frame_u_ms) * slope_x + (ground_v - frame_v_ms) * slope_y;
                    const double requested_delta_w =
                        0.5 * dt_small *
                        ((static_cast<double>(du_dt[i - 1][j][0]) + static_cast<double>(du_dt[i][j][0])) * slope_x +
                         (static_cast<double>(dv_dt[i][j - 1][0]) + static_cast<double>(dv_dt[i][j][0])) * slope_y);
                    partial[0] += 0.25 * mass * ground_w * ground_w;
                    partial[1] += 0.5 * mass * ground_w * requested_delta_w;
                    partial[2] += 0.25 * mass * requested_delta_w * requested_delta_w;
                }
            });
    }
    const double ground_energy_before     = ground_before_sums[0];
    const double ground_requested_work    = ground_before_sums[1];
    const double ground_finite_correction = ground_before_sums[2];

    const auto update_momentum_slab = [&](int i, std::array<double, 4>* partial)
    {
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                double du_d = static_cast<double>(du_dt[i][j][k]);
                double dv_d = static_cast<double>(dv_dt[i][j][k]);
                double dw_d = static_cast<double>(dw_dt[i][j][k]);
                if (primary_state_repairs_enabled && !std::isfinite(du_d))
                    du_d = 0.0;
                if (primary_state_repairs_enabled && !std::isfinite(dv_d))
                    dv_d = 0.0;
                if (primary_state_repairs_enabled && !std::isfinite(dw_d))
                    dw_d = 0.0;

                const double u_old   = static_cast<double>(u[i][j][k]);
                const double v_old   = static_cast<double>(v[i][j][k]);
                const double w_old   = static_cast<double>(w[i][j][k]);
                const double delta_u = du_d * dt_small;
                const double delta_v = dv_d * dt_small;
                const double delta_w = dw_d * dt_small;
                u[i][j][k]           = clamp_wind_horizontal_ms(static_cast<float>(u_old + delta_u));
                v[i][j][k]           = clamp_wind_horizontal_ms(static_cast<float>(v_old + delta_v));
                w[i][j][k]           = clamp_wind_vertical_ms(static_cast<float>(w_old + delta_w));

                if (partial == nullptr)
                    continue;

                const bool physical_j = j >= physical_domain.j_begin && j < physical_domain.j_end;
                const bool physical_i = i >= physical_domain.i_begin && i < physical_domain.i_end;
                const bool physical_k = k >= physical_domain.k_begin && k < physical_domain.k_end;
                if (physical_j && physical_i && physical_k)
                {
                    double u_mass = 0.0;
                    double v_mass = 0.0;
                    double w_mass = 0.0;
                    if (cgrid_layout)
                    {
                        u_mass = 0.5 * acoustic_mixture_mass_kg(state, i, j, k);
                        if (i + 1 < physical_domain.i_end)
                            u_mass += 0.5 * acoustic_mixture_mass_kg(state, i + 1, j, k);
                        v_mass = 0.5 * acoustic_mixture_mass_kg(state, i, j, k);
                        if (j + 1 < physical_domain.j_end)
                            v_mass += 0.5 * acoustic_mixture_mass_kg(state, i, j + 1, k);
                        w_mass = 0.5 * acoustic_mixture_mass_kg(state, i, j, k);
                        if (k + 1 < physical_domain.k_end)
                            w_mass += 0.5 * acoustic_mixture_mass_kg(state, i, j, k + 1);
                    }
                    else
                    {
                        u_mass = acoustic_mixture_mass_kg(state, i, j, k);
                        v_mass = u_mass;
                        w_mass = u_mass;
                    }
                    const double u_stored = static_cast<double>(u[i][j][k]);
                    const double v_stored = static_cast<double>(v[i][j][k]);
                    const double w_stored = static_cast<double>(w[i][j][k]);
                    const double values[] = {
                        u_mass,
                        v_mass,
                        w_mass,
                        u_old,
                        v_old,
                        w_old,
                        delta_u,
                        delta_v,
                        delta_w,
                        u_stored,
                        v_stored,
                        w_stored,
                    };
                    if (std::any_of(
                            std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }) ||
                        u_mass <= 0.0 || v_mass <= 0.0 || w_mass <= 0.0)
                    {
                        (*partial)[3] += 1.0;
                        continue;
                    }
                    (*partial)[0] += u_mass * u_old * delta_u + v_mass * v_old * delta_v + w_mass * w_old * delta_w;
                    (*partial)[1] +=
                        0.5 * (u_mass * delta_u * delta_u + v_mass * delta_v * delta_v + w_mass * delta_w * delta_w);
                    (*partial)[2] += 0.5 * (u_mass * (u_stored * u_stored - u_old * u_old) +
                                            v_mass * (v_stored * v_stored - v_old * v_old) +
                                            w_mass * (w_stored * w_stored - w_old * w_old));
                }
            }
    };
    std::array<double, 4> momentum_sums{};
    if (acoustic_energy_pending != nullptr)
    {
        momentum_sums = tmv::diagnostics::fixed_slab_reduce<4>(
            0, nr, [&](int i, std::array<double, 4>& partial) { update_momentum_slab(i, &partial); });
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nr; ++i)
            update_momentum_slab(i, nullptr);
    }

    if (capture_operator_storage)
    {
        try
        {
            const auto requested_impulse = terrain_metrics != nullptr
                ? runtime.momentum_interface_flux_register
                      .native_dual_tendency_impulse_mapped(
                          rho, du_dt, dv_dt, dw_dt, dt_small,
                          *terrain_metrics)
                : runtime.momentum_interface_flux_register
                      .native_dual_tendency_impulse(
                          rho, du_dt, dv_dt, dw_dt, dt_small,
                          state.grid.dr, state.grid.dtheta, state.grid.dz);
            const auto operator_inventory_after = terrain_metrics != nullptr
                ? runtime.momentum_interface_flux_register
                      .native_dual_inventory_mapped(
                          rho, u, v, w, *terrain_metrics)
                : runtime.momentum_interface_flux_register.native_dual_inventory(
                      rho, u, v, w,
                      state.grid.dr, state.grid.dtheta, state.grid.dz);
            runtime.momentum_interface_flux_register.attribute_operator_commit(
                requested_impulse,
                {operator_inventory_after[0] - operator_inventory_before[0],
                 operator_inventory_after[1] - operator_inventory_before[1],
                 operator_inventory_after[2] - operator_inventory_before[2]});
        }
        catch (...)
        {
            std::swap(u, operator_u_before);
            std::swap(v, operator_v_before);
            std::swap(w, operator_w_before);
            throw;
        }
    }

    if (acoustic_energy_pending != nullptr)
    {
        double ground_energy_after = 0.0;
        if (terrain_layout)
        {
            ground_energy_after = tmv::diagnostics::fixed_slab_reduce<1>(
                physical_domain.i_begin,
                physical_domain.i_end,
                [&](int i, std::array<double, 1>& partial)
                {
                    for (int j = physical_domain.j_begin; j < physical_domain.j_end; ++j)
                    {
                        const double mass = acoustic_mixture_mass_kg(state, i, j, physical_domain.k_begin);
                        const double ground_u =
                            0.5 * (static_cast<double>(u[i - 1][j][0]) + static_cast<double>(u[i][j][0]));
                        const double ground_v =
                            0.5 * (static_cast<double>(v[i][j - 1][0]) + static_cast<double>(v[i][j][0]));
                        const double frame_u_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms;
                        const double frame_v_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms;
                        const double ground_w =
                            (ground_u - frame_u_ms)
                                * (-terrain_metrics->Gx_face(i, j, 0)) +
                            (ground_v - frame_v_ms)
                                * (-terrain_metrics->Gy_face(i, j, 0));
                        partial[0] += 0.25 * mass * ground_w * ground_w;
                    }
                })[0];
        }
        double       requested_pressure_work   = momentum_sums[0] + ground_requested_work;
        double       finite_step_pressure_work = momentum_sums[1] + ground_finite_correction;
        double       committed_pressure_work   = momentum_sums[2] + ground_energy_after - ground_energy_before;
        const double storage_rounding = committed_pressure_work - requested_pressure_work - finite_step_pressure_work;
        const auto&  changes          = acoustic_energy_pending->continuity_eos_changes;
        const double committed_inventory =
            std::accumulate(changes.begin(), changes.end(), 0.0) + committed_pressure_work;
        const double values[] = {
            requested_pressure_work,
            finite_step_pressure_work,
            committed_pressure_work,
            storage_rounding,
            committed_inventory,
            ground_energy_before,
            ground_energy_after,
        };
        if (momentum_sums[3] != 0.0 ||
            std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }) ||
            finite_step_pressure_work < 0.0)
        {
            throw std::runtime_error("terrain acoustic pressure-work attribution produced "
                                     "invalid basis-v2 state");
        }

        CartesianCGridAcousticEnergyReportV1 next = *acoustic_energy_pending->report;
        next.continuity_eos_dry_internal_change_j += changes[0];
        next.continuity_eos_water_sensible_change_j += changes[1];
        next.continuity_eos_phase_reference_change_j += changes[2];
        next.continuity_carrier_kinetic_change_j += changes[3];
        next.continuity_gravitational_change_j += changes[4];
        next.continuity_subgrid_tke_change_j += changes[5];
        next.continuity_carrier_requested_change_j += acoustic_energy_pending->continuity_carrier_requested_change_j;
        next.pressure_gradient_requested_linear_work_j += requested_pressure_work;
        next.pressure_gradient_finite_step_correction_j += finite_step_pressure_work;
        next.pressure_gradient_storage_rounding_j += storage_rounding;
        next.pressure_gradient_committed_work_j += committed_pressure_work;
        next.committed_inventory_change_j += committed_inventory;
        ++next.substep_count;
        next.complete                                                  = true;
        *acoustic_energy_pending->report                               = next;
        acoustic_energy_pending->continuity_eos_changes                = {};
        acoustic_energy_pending->continuity_carrier_requested_change_j = 0.0;
        acoustic_energy_pending->pressure_ready                        = false;
    }
}

static bool
try_apply_fast_momentum_agrid_gpu(const SimulationStateView& state, double dt_small, AcousticEnergyPending* pending)
{
    const int nz = state.grid.nz;
    if (tmv::physical_grid_metrics::terrain_is_active(state) || static_cast<int>(state.base.p0.size()) < nz ||
        !active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), DynamicsKernelOp::AcousticMomentumCartesianAgrid))
    {
        return false;
    }
    const AcousticProfileRows profiles = acoustic_profile_rows(state, nz);
    fill_acoustic_base_profiles(state.base.rho0,
                                state.base.p0,
                                nz,
                                state.grid.dz,
                                profiles,
                                /*fill_rho0=*/false,
                                /*fill_p0=*/true,
                                /*fill_dp0_dz=*/false);
    FlatAgridMomentumObserverContext observer_context{&state, pending};
    const bool                       attributed = pending != nullptr;
    return dispatch_acoustic_momentum_cartesian_agrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), state.fields.rho.data(),
                                                              state.fields.p.data(),
                                                              profiles.p0,
                                                              nz,
                                                              state.fields.u.data(),
                                                              state.fields.v.data(),
                                                              state.fields.w.data(),
                                                              state.fields.u.data(),
                                                              state.fields.v.data(),
                                                              state.fields.w.data(),
                                                              state.grid.nr,
                                                              state.grid.nth,
                                                              state.grid.nz,
                                                              static_cast<float>(state.grid.dr),
                                                              static_cast<float>(state.grid.dtheta),
                                                              static_cast<float>(state.grid.dz),
                                                              static_cast<float>(dt_small),
                                                              wind_horizontal_abs_max_ms,
                                                              wind_vertical_abs_max_ms,
                                                              attributed ? observe_flat_agrid_momentum_commit : nullptr,
                                                              attributed ? &observer_context : nullptr);
}

static bool try_apply_fast_momentum_gpu(const SimulationStateView& state,
                                        double                     dt_small,
                                        bool                       flat_terrain,
                                        bool                       gpu_acoustic_ok,
                                        bool                       is_cgrid,
                                        float                      dr_f,
                                        float                      dtheta_f,
                                        float                      dz_f)
{
    auto&        u       = state.fields.u;
    auto&        v       = state.fields.v;
    auto&        w       = state.fields.w;
    auto&        rho     = state.fields.rho;
    auto&        p       = state.fields.p;
    const auto&  p0_base = state.base.p0;
    const int    NR      = state.grid.nr;
    const int    NTH     = state.grid.nth;
    const int    NZ      = state.grid.nz;
    const double dz      = state.grid.dz;
    const float  dt_s    = static_cast<float>(dt_small);

    if (active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), DynamicsKernelOp::AcousticMomentumCartesianCgrid) && is_cgrid && flat_terrain &&
        !p0_base.empty() && static_cast<int>(p0_base.size()) >= NZ)
    {
        const AcousticProfileRows profiles = acoustic_profile_rows(state, NZ);
        fill_acoustic_base_profiles(state.base.rho0,
                                    p0_base,
                                    NZ,
                                    dz,
                                    profiles,
                                    /*fill_rho0=*/false,
                                    /*fill_p0=*/false,
                                    /*fill_dp0_dz=*/true);
        if (dispatch_acoustic_momentum_cartesian_cgrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), rho.data(),
                                                               p.data(),
                                                               profiles.dp0_dz,
                                                               NZ,
                                                               u.data(),
                                                               v.data(),
                                                               w.data(),
                                                               u.data(),
                                                               v.data(),
                                                               w.data(),
                                                               NR,
                                                               NTH,
                                                               NZ,
                                                               dr_f,
                                                               dtheta_f,
                                                               dz_f,
                                                               dt_s,
                                                               wind_horizontal_abs_max_ms,
                                                               wind_vertical_abs_max_ms,
                                                               primary_state_repairs_enabled,
                                                               primary_state_repairs_enabled ? nullptr : profiles.dp0_dz_low))
        {
            return true;
        }
    }

    return primary_state_repairs_enabled && gpu_acoustic_ok && flat_terrain &&
           dispatch_acoustic_momentum_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), rho.data(),
                                              p.data(),
                                              u.data(),
                                              v.data(),
                                              w.data(),
                                              u.data(),
                                              v.data(),
                                              w.data(),
                                              NR,
                                              NTH,
                                              NZ,
                                              dr_f,
                                              dtheta_f,
                                              dz_f,
                                              dt_s,
                                              wind_horizontal_abs_max_ms,
                                              wind_vertical_abs_max_ms);
}

static void apply_fast_momentum(const SimulationStateView& state,
                                SplitExplicitDynamics&     split_scheme,
                                double                     dt_small,
                                Field3D&                   du_dt,
                                Field3D&                   dv_dt,
                                Field3D&                   dw_dt,
                                bool                       flat_terrain,
                                bool                       gpu_acoustic_ok,
                                bool                       is_cgrid,
                                float                      dr_f,
                                float                      dtheta_f,
                                float                      dz_f,
                                AcousticEnergyPending*     acoustic_energy_pending)
{
    const bool interface_flux_capture =
        split_scheme.momentum_interface_flux_capture_active();
    const bool native_agrid_required =
        !is_cgrid && flat_terrain && active_compute_backend_kind(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) != ComputeBackendKind::Cpu &&
        !compute_backend_fallback_active(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) && !tmv::require_level_runtime(state.compute_runtime, "compute_runtime").config.allow_fallback &&
        state.configuration != nullptr && state.configuration->gpu_apply_slow;
    if (!is_cgrid && flat_terrain)
    {
        if (try_apply_fast_momentum_agrid_gpu(state, dt_small, acoustic_energy_pending))
            return;
        if (native_agrid_required)
            throw std::runtime_error("fallback-off native A-grid momentum dispatch failed");
    }
    const bool attribute_flat_cgrid_device =
        acoustic_energy_pending != nullptr && acoustic_energy_pending->layout == AcousticEnergyLayout::FlatCGrid;
    if (attribute_flat_cgrid_device)
    {
        if (acoustic_energy_pending->report == nullptr || !acoustic_energy_pending->pressure_ready)
        {
            throw std::logic_error("device C-grid momentum attribution lifecycle is invalid");
        }
        const std::size_t bytes = state.fields.u.size() * sizeof(float);
        std::memcpy(du_dt.data(), state.fields.u.data(), bytes);
        std::memcpy(dv_dt.data(), state.fields.v.data(), bytes);
        std::memcpy(dw_dt.data(), state.fields.w.data(), bytes);
    }

    const bool dispatched = !interface_flux_capture &&
        try_apply_fast_momentum_gpu(state, dt_small, flat_terrain,
                                    gpu_acoustic_ok, is_cgrid,
                                    dr_f, dtheta_f, dz_f);
    if (dispatched)
    {
        if (attribute_flat_cgrid_device)
        {
            attribute_flat_cgrid_device_momentum_commit(state,
                                                        du_dt,
                                                        dv_dt,
                                                        dw_dt,
                                                        static_cast<float>(dt_small),
                                                        dr_f,
                                                        dtheta_f,
                                                        dz_f,
                                                        *acoustic_energy_pending);
        }
        return;
    }
    else
    {
        auto& compute = tmv::require_level_runtime(state.compute_runtime, "compute runtime");
        if (is_cgrid && flat_terrain && !interface_flux_capture &&
            active_compute_backend_kind(compute) != ComputeBackendKind::Cpu && !compute.config.allow_fallback)
        {
            throw std::runtime_error("fallback-off native C-grid momentum dispatch failed");
        }
        apply_fast_momentum_cpu(state, split_scheme, dt_small, du_dt, dv_dt, dw_dt, acoustic_energy_pending);
    }
}

struct FlatAgridConstraintObserverContext
{
    const SimulationStateView* state   = nullptr;
    AcousticEnergyPending*     pending = nullptr;
};

static void observe_flat_agrid_constraint_commit(void*        opaque,
                                                 const float* u_candidate,
                                                 const float* v_candidate,
                                                 const float* w_candidate)
{
    auto& context = *static_cast<FlatAgridConstraintObserverContext*>(opaque);
    if (context.pending == nullptr)
        return;
    const auto&              state  = *context.state;
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const int                ny     = state.grid.nth;
    const int                nz     = state.grid.nz;
    const double             work   = tmv::diagnostics::fixed_slab_reduce<1>(
        domain.i_begin,
        domain.i_end,
        [&](int i, std::array<double, 1>& partial)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t n     = (static_cast<std::size_t>(i) * ny + j) * nz + k;
                    const double      mass  = acoustic_mixture_mass_kg(state, i, j, k);
                    const double old_speed2 = static_cast<double>(state.fields.u.data()[n]) * state.fields.u.data()[n] +
                                              static_cast<double>(state.fields.v.data()[n]) * state.fields.v.data()[n] +
                                              static_cast<double>(state.fields.w.data()[n]) * state.fields.w.data()[n];
                    const double new_speed2 = static_cast<double>(u_candidate[n]) * u_candidate[n] +
                                              static_cast<double>(v_candidate[n]) * v_candidate[n] +
                                              static_cast<double>(w_candidate[n]) * w_candidate[n];
                    partial[0] += 0.5 * mass * (new_speed2 - old_speed2);
                }
        })[0];
    if (!std::isfinite(work))
        throw std::runtime_error("native A-grid null-mode constraint produced invalid basis-v2 work");
    auto next = *context.pending->report;
    next.boundary_committed_work_j += work;
    next.boundary_constraint_history_committed_work_j += work;
    next.committed_inventory_change_j += work;
    const double values[] = {next.boundary_committed_work_j,
                             next.boundary_constraint_history_committed_work_j,
                             next.committed_inventory_change_j};
    if (std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
        throw std::overflow_error("native A-grid constraint attribution overflowed");
    *context.pending->report = next;
}

static bool try_apply_flat_agrid_constraint_gpu(const SimulationStateView& state, AcousticEnergyPending* pending)
{
    FlatAgridConstraintObserverContext observer_context{&state, pending};
    return dispatch_acoustic_null_mode_constraint_cartesian_agrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
        state.fields.rho.data(),
        state.fields.qv.data(),
        state.fields.qc.data(),
        state.fields.qr.data(),
        state.fields.qi.data(),
        state.fields.qs.data(),
        state.fields.qg.data(),
        state.fields.qh.data(),
        state.fields.u.data(),
        state.fields.v.data(),
        state.fields.w.data(),
        state.fields.u.data(),
        state.fields.v.data(),
        state.fields.w.data(),
        state.grid.nr,
        state.grid.nth,
        state.grid.nz,
        pending != nullptr ? observe_flat_agrid_constraint_commit : nullptr,
        pending != nullptr ? &observer_context : nullptr);
}

static bool try_apply_flat_agrid_boundary_gpu(const SimulationStateView&            state,
                                              CartesianCGridAcousticEnergyReportV1* report)
{
    const int nz = state.grid.nz;
    if (static_cast<int>(state.base.p0.size()) < nz || static_cast<int>(state.base.rho0.size()) < nz ||
        static_cast<int>(state.base.u0.size()) < nz || static_cast<int>(state.base.v0.size()) < nz)
        return false;
    std::vector<float> p0_profile(static_cast<std::size_t>(nz));
    std::vector<float> rho0_profile(static_cast<std::size_t>(nz));
    std::vector<float> u0_profile(static_cast<std::size_t>(nz));
    std::vector<float> v0_profile(static_cast<std::size_t>(nz));
    for (int k = 0; k < nz; ++k)
    {
        const auto n    = static_cast<std::size_t>(k);
        p0_profile[n]   = static_cast<float>(state.base.p0[n]);
        rho0_profile[n] = static_cast<float>(state.base.rho0[n]);
        u0_profile[n]   = static_cast<float>(state.base.u0[n]);
        v0_profile[n]   = static_cast<float>(state.base.v0[n]);
    }
    if (!dispatch_acoustic_boundary_cartesian_agrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), p0_profile.data(),
                                                            rho0_profile.data(),
                                                            u0_profile.data(),
                                                            v0_profile.data(),
                                                            nz,
                                                            state.fields.u.data(),
                                                            state.fields.v.data(),
                                                            state.fields.w.data(),
                                                            state.fields.rho.data(),
                                                            state.fields.p.data(),
                                                            state.grid.nr,
                                                            state.grid.nth,
                                                            state.grid.nz))
        return false;
    if (report != nullptr)
    {
        auto next = *report;
        ++next.boundary_application_count;
        next.boundary_complete = true;
        *report                = next;
    }
    invalidate_named_fields(state, {"u", "v", "w", "rho", "p"});
    return true;
}

static bool apply_fast_fused(const SimulationStateView&            state,
                             SplitExplicitDynamics&                split_scheme,
                             double                                dt_small,
                             Field3D&                              du_dt,
                             Field3D&                              dv_dt,
                             Field3D&                              dw_dt,
                             Field3D&                              drho_dt,
                             Field3D&                              dp_dt,
                             bool                                  flat_terrain,
                             bool                                  gpu_acoustic_ok,
                             bool                                  is_cgrid,
                             float                                 dr_f,
                             float                                 dtheta_f,
                             float                                 dz_f,
                             tmv::CartesianCGridTransportContext*  transport_context,
                             tmv::CartesianCGridTransportContext&  transport_staging,
                             CartesianCGridAcousticEnergyReportV1* acoustic_energy_report)
{
    if (split_scheme.momentum_interface_flux_capture_active())
        return false;
    auto&        u         = state.fields.u;
    auto&        v         = state.fields.v;
    auto&        w         = state.fields.w;
    auto&        rho       = state.fields.rho;
    auto&        p         = state.fields.p;
    auto&        theta     = state.fields.theta;
    const auto&  rho0_base = state.base.rho0;
    const auto&  p0_base   = state.base.p0;
    const int    NR        = state.grid.nr;
    const int    NTH       = state.grid.nth;
    const int    NZ        = state.grid.nz;
    const double dr        = state.grid.dr;
    const double dtheta    = state.grid.dtheta;
    const double dz        = state.grid.dz;
    if (!primary_state_repairs_enabled)
    {
        auto& compute = tmv::require_level_runtime(state.compute_runtime, "compute runtime");
        auto* backend = mutable_compute_backend(compute);
        auto* kernels = backend != nullptr ? backend->dynamics_kernels() : nullptr;
        if (is_cgrid && flat_terrain && !compute.config.validate_parity &&
            !dynamics_runtime_for(state).momentum_interface_flux_capture_enabled &&
            active_compute_backend_kind(compute) != ComputeBackendKind::Cpu && kernels != nullptr &&
            kernels->supports_unrepaired_acoustic_fused() && static_cast<int>(rho0_base.size()) >= NZ &&
            static_cast<int>(p0_base.size()) >= NZ)
        {
            const AcousticProfileRows profiles = acoustic_profile_rows(state, NZ);
            fill_acoustic_base_profiles(rho0_base, p0_base, NZ, dz, profiles, true, true, true);
            AcousticSubstepFusedCartesianCgridDispatch dispatch;
            dispatch.u = u.data(); dispatch.v = v.data(); dispatch.w = w.data();
            dispatch.rho = rho.data(); dispatch.p = p.data(); dispatch.theta_data = theta.data();
            dispatch.rho0_base_data = profiles.rho0; dispatch.p0_base_data = profiles.p0;
            dispatch.dp0_dz_data = profiles.dp0_dz; dispatch.dp0_dz_low_data = profiles.dp0_dz_low;
            dispatch.nx = NR; dispatch.ny = NTH; dispatch.nz = NZ; dispatch.base_len = NZ;
            dispatch.dx = dr_f; dispatch.dy = dtheta_f; dispatch.dz = dz_f;
            dispatch.dt_small = static_cast<float>(dt_small);
            dispatch.gamma_val = static_cast<float>(dynamics_constants::gamma);
            dispatch.gamma_low = static_cast<float>(dynamics_constants::gamma - static_cast<double>(dispatch.gamma_val));
            dispatch.rho_min = density_min_kgm3; dispatch.rho_max = density_max_kgm3;
            dispatch.p_min = pressure_min_pa; dispatch.p_max = pressure_max_pa;
            dispatch.p_ref_fallback = static_cast<float>(p0);
            dispatch.wind_clamp_h = wind_horizontal_abs_max_ms; dispatch.wind_clamp_v = wind_vertical_abs_max_ms;
            dispatch.frame_velocity_u_ms = static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.u_ms);
            dispatch.frame_velocity_v_ms = static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.v_ms);
            dispatch.repair_primary_state = false;
            if (acoustic_energy_report == nullptr)
            {
                tmv::CartesianCGridTransportContext* next_transport = nullptr;
                if (transport_context != nullptr)
                {
                    const auto copy_started = dynamics_perf_start();
                    transport_staging = *transport_context;
                    dynamics_perf_finish(dynamics_perf_totals.acoustic_transport_copy_s, copy_started);
                    next_transport = &transport_staging;
                    const auto flux_started = dynamics_perf_start();
                    next_transport->accumulate(state, rho, u, v, w, static_cast<double>(dispatch.dt_small),
                        nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeCompensated);
                    dynamics_perf_finish(dynamics_perf_totals.acoustic_transport_flux_s, flux_started);
                }
                const auto dispatch_started = dynamics_perf_start();
                if (!kernels->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch))
                    throw std::runtime_error("native unrepaired C-grid fused acoustic dispatch failed");
                dynamics_perf_finish(dynamics_perf_totals.acoustic_dispatch_s, dispatch_started);
                if (next_transport) std::swap(*transport_context, *next_transport);
                return true;
            }

            const auto prior_report = *acoustic_energy_report;
            if (prior_report.complete != (prior_report.substep_count > 0) || !prior_report.boundary_complete)
                throw std::invalid_argument("native fused acoustic report lifecycle is invalid");
            if (prior_report.substep_count == 0)
            {
                const auto validation_started = dynamics_perf_start();
                (void)compute_moist_mixture_energy_v2(state);
                dynamics_perf_finish(dynamics_perf_totals.acoustic_initial_validation_s, validation_started);
            }
            const std::size_t bytes = u.size() * sizeof(float);
            const auto snapshot_started = dynamics_perf_start();
            tmv::copy_byte_ranges(std::array<tmv::ByteCopyRange, 5>{{
                {du_dt.data(), u.data()}, {dv_dt.data(), v.data()},
                {dw_dt.data(), w.data()}, {drho_dt.data(), rho.data()},
                {dp_dt.data(), p.data()},
            }}, bytes);
            dynamics_perf_finish(dynamics_perf_totals.acoustic_state_copy_s, snapshot_started);
            tmv::CartesianCGridTransportContext* transport_candidate = nullptr;
            if (transport_context != nullptr)
            {
                const auto copy_started = dynamics_perf_start();
                transport_staging = *transport_context;
                dynamics_perf_finish(dynamics_perf_totals.acoustic_transport_copy_s, copy_started);
                transport_candidate = &transport_staging;
                const tmv::CarrierKineticBoundaryInputs carrier_inputs{{
                    &state.fields.qv, &state.fields.qc, &state.fields.qr, &state.fields.qi,
                    &state.fields.qs, &state.fields.qg, &state.fields.qh}};
                const auto flux_started = dynamics_perf_start();
                transport_candidate->accumulate(state, rho, u, v, w, static_cast<double>(dispatch.dt_small),
                    nullptr, &carrier_inputs, tmv::AcousticMassFluxPrecision::NativeCompensated);
                dynamics_perf_finish(dynamics_perf_totals.acoustic_transport_flux_s, flux_started);
            }
            try
            {
                const auto dispatch_started = dynamics_perf_start();
                if (!kernels->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch))
                    throw std::runtime_error("native unrepaired C-grid fused acoustic dispatch failed");
                dynamics_perf_finish(dynamics_perf_totals.acoustic_dispatch_s, dispatch_started);
                const auto kinetic_endpoints = measured_acoustic_velocity_energy_pair(state, du_dt, dv_dt, dw_dt);
                const double pressure_kinetic = kinetic_endpoints[0];
                auto next_report = prior_report;
                AcousticEnergyPending pending;
                pending.report = &next_report;
                pending.layout = AcousticEnergyLayout::FlatCGrid;
                pending.initial_state_validated = true;
                const auto attribution_started = dynamics_perf_start();
                attribute_flat_cgrid_device_pressure_commit(state, drho_dt, dp_dt, dispatch.dt_small,
                    dr_f, dtheta_f, dz_f, pending, &du_dt, &dv_dt, &dw_dt);
                dynamics_perf_finish(dynamics_perf_totals.acoustic_attribution_s, attribution_started);
                const double momentum_kinetic = kinetic_endpoints[1];
                const double correction = momentum_kinetic - pressure_kinetic -
                    (next_report.pressure_gradient_committed_work_j - prior_report.pressure_gradient_committed_work_j);
                if (!std::isfinite(correction))
                    throw std::runtime_error("native fused acoustic energy commit overflowed");
                next_report.pressure_gradient_committed_work_j += correction;
                static_assert(std::is_nothrow_swappable_v<tmv::CartesianCGridTransportContext>);
                if (transport_candidate)
                    std::swap(*transport_context, *transport_candidate);
                *acoustic_energy_report = next_report;
            }
            catch (...)
            {
                std::memcpy(u.data(), du_dt.data(), bytes);
                std::memcpy(v.data(), dv_dt.data(), bytes);
                std::memcpy(w.data(), dw_dt.data(), bytes);
                std::memcpy(rho.data(), drho_dt.data(), bytes);
                std::memcpy(p.data(), dp_dt.data(), bytes);
                invalidate_named_fields(state, {"u", "v", "w", "rho", "p"});
                throw;
            }
            return true;
        }
        return false;
    }
    if (active_compute_backend_kind(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) == ComputeBackendKind::Cpu && is_cgrid && flat_terrain)
    {
        ComputeBackend*         backend = mutable_compute_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"));
        DynamicsComputeKernels* kernels = backend != nullptr ? backend->dynamics_kernels() : nullptr;
        if (kernels == nullptr || !kernels->supports(DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid))
        {
            // No transport lifecycle state has changed, so the time stepper
            // may safely select the separate pressure/momentum callbacks.
            return false;
        }
        if (transport_context != nullptr)
        {
            // Preserve the exact pre-continuity state used by scalar
            // transport before the in-place CPU kernel changes density.
            const tmv::CarrierKineticBoundaryInputs carrier_inputs{{
                &state.fields.qv,
                &state.fields.qc,
                &state.fields.qr,
                &state.fields.qi,
                &state.fields.qs,
                &state.fields.qg,
                &state.fields.qh,
            }};
            transport_context->accumulate(state,
                rho, u, v, w, dt_small, nullptr, acoustic_energy_report != nullptr ? &carrier_inputs : nullptr);
        }

        AcousticSubstepFusedCartesianCgridDispatch dispatch;
        dispatch.u              = u.data();
        dispatch.v              = v.data();
        dispatch.w              = w.data();
        dispatch.rho            = rho.data();
        dispatch.p              = p.data();
        dispatch.theta_data     = theta.data();
        dispatch.base_len       = NZ;
        dispatch.nx             = NR;
        dispatch.ny             = NTH;
        dispatch.nz             = NZ;
        dispatch.gamma_val      = static_cast<float>(dynamics_constants::gamma);
        dispatch.rho_min        = density_min_kgm3;
        dispatch.rho_max        = density_max_kgm3;
        dispatch.p_min          = pressure_min_pa;
        dispatch.p_max          = pressure_max_pa;
        dispatch.p_ref_fallback = static_cast<float>(p0);
        dispatch.wind_clamp_h   = wind_horizontal_abs_max_ms;
        dispatch.wind_clamp_v   = wind_vertical_abs_max_ms;
        dispatch.rho0_base_host = rho0_base.data();
        dispatch.p0_base_host   = p0_base.data();
        dispatch.dx_host        = dr;
        dispatch.dy_host        = dtheta;
        dispatch.dz_host        = dz;
        dispatch.dt_small_host  = dt_small;
        dispatch.gamma_host     = dynamics_constants::gamma;
        dispatch.drho_scratch   = drho_dt.data();
        if (acoustic_energy_report != nullptr)
        {
            dispatch.qv_host            = state.fields.qv.data();
            dispatch.qc_host            = state.fields.qc.data();
            dispatch.qr_host            = state.fields.qr.data();
            dispatch.qi_host            = state.fields.qi.data();
            dispatch.qs_host            = state.fields.qs.data();
            dispatch.qg_host            = state.fields.qg.data();
            dispatch.qh_host            = state.fields.qh.data();
            dispatch.tke_host           = state.fields.tke.data();
            dispatch.energy_report_host = acoustic_energy_report;
        }
        if (!kernels->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch))
        {
            // The backend advertised support after retained-flux ownership
            // began. Falling through would double-sample this acoustic
            // substep, so fail closed on the backend contract violation.
            throw std::runtime_error("CPU backend advertised fused Cartesian C-grid acoustics "
                                     "but declined the dispatch");
        }
        return true;
    }

    const float dt_s                        = static_cast<float>(dt_small);
    const bool  attribute_flat_cgrid_device = acoustic_energy_report != nullptr &&
                                              active_compute_backend_kind(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) != ComputeBackendKind::Cpu && is_cgrid &&
                                              flat_terrain;
    if (acoustic_energy_report != nullptr && !attribute_flat_cgrid_device)
    {
        return false;
    }

    if (active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid) && is_cgrid &&
        (transport_context == nullptr || attribute_flat_cgrid_device) && flat_terrain && !p0_base.empty() &&
        static_cast<int>(p0_base.size()) >= NZ)
    {
        if (static_cast<int>(rho0_base.size()) >= NZ)
        {
            if (attribute_flat_cgrid_device)
            {
                const auto&  report          = *acoustic_energy_report;
                const double report_values[] = {
                    report.continuity_eos_dry_internal_change_j,
                    report.continuity_eos_water_sensible_change_j,
                    report.continuity_eos_phase_reference_change_j,
                    report.continuity_carrier_kinetic_change_j,
                    report.continuity_gravitational_change_j,
                    report.continuity_subgrid_tke_change_j,
                    report.continuity_carrier_requested_change_j,
                    report.pressure_gradient_requested_linear_work_j,
                    report.pressure_gradient_finite_step_correction_j,
                    report.pressure_gradient_storage_rounding_j,
                    report.pressure_gradient_committed_work_j,
                    report.boundary_committed_work_j,
                    report.boundary_radiative_requested_linear_work_j,
                    report.boundary_radiative_finite_step_correction_j,
                    report.boundary_radiative_storage_rounding_j,
                    report.boundary_radiative_committed_work_j,
                    report.boundary_constraint_history_committed_work_j,
                    report.committed_inventory_change_j,
                };
                if (std::any_of(std::begin(report_values),
                                std::end(report_values),
                                [](double value) { return !std::isfinite(value); }) ||
                    report.pressure_gradient_finite_step_correction_j < 0.0 ||
                    report.boundary_radiative_finite_step_correction_j < 0.0 ||
                    report.complete != (report.substep_count > 0) || !report.boundary_complete)
                {
                    throw std::invalid_argument("fused device C-grid acoustic attribution received "
                                                "an invalid report");
                }
                // Validate the complete basis before the fused device writer
                // can mutate any field. The five existing tendency fields
                // retain its exact pre-commit FP32 state; rho/p remain the
                // pressure barrier after the fused dispatch, while u/v/w are
                // the momentum endpoint.
                if (report.substep_count == 0)
                    (void)compute_moist_mixture_energy_v2(state);
                const std::size_t bytes = u.size() * sizeof(float);
                std::memcpy(du_dt.data(), u.data(), bytes);
                std::memcpy(dv_dt.data(), v.data(), bytes);
                std::memcpy(dw_dt.data(), w.data(), bytes);
                std::memcpy(drho_dt.data(), rho.data(), bytes);
                std::memcpy(dp_dt.data(), p.data(), bytes);
            }

            const AcousticProfileRows profiles = acoustic_profile_rows(state, NZ);
            fill_acoustic_base_profiles(rho0_base,
                                        p0_base,
                                        NZ,
                                        dz,
                                        profiles,
                                        /*fill_rho0=*/true,
                                        /*fill_p0=*/true,
                                        /*fill_dp0_dz=*/true);
            const auto try_dispatch = [&]()
            {
                return dispatch_acoustic_substep_fused_cartesian_cgrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
                    u.data(),
                    v.data(),
                    w.data(),
                    rho.data(),
                    p.data(),
                    theta.data(),
                    profiles.rho0,
                    profiles.p0,
                    profiles.dp0_dz,
                    NZ,
                    NR,
                    NTH,
                    NZ,
                    dr_f,
                    dtheta_f,
                    dz_f,
                    static_cast<float>(dynamics_constants::gamma),
                    dt_s,
                    density_min_kgm3,
                    density_max_kgm3,
                    pressure_min_pa,
                    pressure_max_pa,
                    static_cast<float>(p0),
                    wind_horizontal_abs_max_ms,
                    wind_vertical_abs_max_ms,
                    static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.u_ms),
                    static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.v_ms));
            };

            const bool parity_sample = nimbus::parity::sample("acoustic_substep_fused_cartesian_cgrid");
            if (!parity_sample || attribute_flat_cgrid_device)
            {
                if (try_dispatch())
                {
                    if (attribute_flat_cgrid_device)
                    {
                        if (transport_context != nullptr)
                        {
                            const tmv::CarrierKineticBoundaryInputs carrier_inputs{{
                                &state.fields.qv,
                                &state.fields.qc,
                                &state.fields.qr,
                                &state.fields.qi,
                                &state.fields.qs,
                                &state.fields.qg,
                                &state.fields.qh,
                            }};
                            transport_context->accumulate(state, drho_dt,
                                                          du_dt,
                                                          dv_dt,
                                                          dw_dt,
                                                          static_cast<double>(dt_s),
                                                          nullptr,
                                                          &carrier_inputs,
                                                          tmv::AcousticMassFluxPrecision::NativeFloat);
                        }
                        tmv::device_acoustic_energy::attribute_flat_cgrid_fused_commit(state,
                                                                                       du_dt,
                                                                                       dv_dt,
                                                                                       dw_dt,
                                                                                       drho_dt,
                                                                                       dp_dt,
                                                                                       dt_s,
                                                                                       dr_f,
                                                                                       dtheta_f,
                                                                                       dz_f,
                                                                                       *acoustic_energy_report);
                    }
                    return true;
                }
            }
            else
            {
                auto& pool    = FieldPool::instance();
                auto  pre_u   = pool.scoped_acquire_copy(u);
                auto  pre_v   = pool.scoped_acquire_copy(v);
                auto  pre_w   = pool.scoped_acquire_copy(w);
                auto  pre_rho = pool.scoped_acquire_copy(rho);
                auto  pre_p   = pool.scoped_acquire_copy(p);
                if (try_dispatch())
                {
                    const std::size_t bytes   = u.size() * sizeof(float);
                    auto              gpu_u   = pool.scoped_acquire_copy(u);
                    auto              gpu_v   = pool.scoped_acquire_copy(v);
                    auto              gpu_w   = pool.scoped_acquire_copy(w);
                    auto              gpu_rho = pool.scoped_acquire_copy(rho);
                    auto              gpu_p   = pool.scoped_acquire_copy(p);

                    std::memcpy(u.data(), pre_u.field.data(), bytes);
                    std::memcpy(v.data(), pre_v.field.data(), bytes);
                    std::memcpy(w.data(), pre_w.field.data(), bytes);
                    std::memcpy(rho.data(), pre_rho.field.data(), bytes);
                    std::memcpy(p.data(), pre_p.field.data(), bytes);

                    apply_fast_pressure_cpu(state, split_scheme, dt_small, drho_dt, dp_dt);
                    apply_fast_momentum_cpu(state, split_scheme, dt_small, du_dt, dv_dt, dw_dt);

                    using nimbus::parity::Tier;
                    const char* kn = "acoustic_substep_fused_cartesian_cgrid";
                    nimbus::parity::check_field(kn, "u", u.data(), gpu_u.field.data(), NR, NTH, NZ, Tier::Stencil);
                    nimbus::parity::check_field(kn, "v", v.data(), gpu_v.field.data(), NR, NTH, NZ, Tier::Stencil);
                    nimbus::parity::check_field(kn, "w", w.data(), gpu_w.field.data(), NR, NTH, NZ, Tier::Stencil);
                    nimbus::parity::check_field(
                        kn, "rho", rho.data(), gpu_rho.field.data(), NR, NTH, NZ, Tier::Stencil);
                    nimbus::parity::check_field(kn, "p", p.data(), gpu_p.field.data(), NR, NTH, NZ, Tier::Stencil);

                    std::memcpy(u.data(), gpu_u.field.data(), bytes);
                    std::memcpy(v.data(), gpu_v.field.data(), bytes);
                    std::memcpy(w.data(), gpu_w.field.data(), bytes);
                    std::memcpy(rho.data(), gpu_rho.field.data(), bytes);
                    std::memcpy(p.data(), gpu_p.field.data(), bytes);
                    return true;
                }
            }
        }
    }

    // The legacy collocated fused kernel advances continuity internally but
    // cannot export its per-substep reconstructed mass flux. With coupled
    // scalar transport active, fall back to the separate pressure callback,
    // which samples the exact pre-continuity state into the retained context.
    if (transport_context != nullptr)
        return false;
    if (!gpu_acoustic_ok || !flat_terrain)
        return false;
    return dispatch_acoustic_substep_fused_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), u.data(),
                                                   v.data(),
                                                   w.data(),
                                                   rho.data(),
                                                   p.data(),
                                                   NR,
                                                   NTH,
                                                   NZ,
                                                   dr_f,
                                                   dtheta_f,
                                                   dz_f,
                                                   static_cast<float>(dynamics_constants::gamma),
                                                   dt_s,
                                                   density_min_kgm3,
                                                   pressure_min_pa,
                                                   wind_horizontal_abs_max_ms,
                                                   wind_vertical_abs_max_ms);
}

static bool apply_fast_batched(const SimulationStateView&            state,
                               SplitExplicitDynamics&                split_scheme,
                               double                                dt_small,
                               int                                   n_substeps,
                               Field3D&                              du_dt,
                               Field3D&                              dv_dt,
                               Field3D&                              dw_dt,
                               Field3D&                              drho_dt,
                               Field3D&                              dp_dt,
                               bool                                  flat_terrain,
                               bool                                  gpu_acoustic_ok,
                               bool                                  is_cgrid,
                               float                                 dr_f,
                               float                                 dtheta_f,
                               float                                 dz_f,
                               tmv::CartesianCGridTransportContext*  transport_context,
                               CartesianCGridAcousticEnergyReportV1* acoustic_energy_report)
{
    auto& bc_scheme = tmv::boundary_scheme_for(state);
    if (split_scheme.momentum_interface_flux_capture_active())
        return false;
    auto&        u         = state.fields.u;
    auto&        v         = state.fields.v;
    auto&        w         = state.fields.w;
    auto&        rho       = state.fields.rho;
    auto&        p         = state.fields.p;
    auto&        theta     = state.fields.theta;
    const auto&  rho0_base = state.base.rho0;
    const auto&  p0_base   = state.base.p0;
    const int    NR        = state.grid.nr;
    const int    NTH       = state.grid.nth;
    const int    NZ        = state.grid.nz;
    const double dz        = state.grid.dz;
    if (!primary_state_repairs_enabled)
    {
        auto& compute = tmv::require_level_runtime(state.compute_runtime, "compute runtime");
        auto* backend = mutable_compute_backend(compute);
        auto* kernels = backend ? backend->dynamics_kernels() : nullptr;
        if (!is_cgrid || !flat_terrain || acoustic_energy_report || !transport_context ||
            compute.config.validate_parity || dynamics_runtime_for(state).momentum_interface_flux_capture_enabled ||
            !kernels || !kernels->supports_unrepaired_acoustic_batched() || NR < 4 || NTH < 4 ||
            static_cast<int>(rho0_base.size()) < NZ || static_cast<int>(p0_base.size()) < NZ)
            return false;
        const AcousticProfileRows profiles = acoustic_profile_rows(state, NZ);
        fill_acoustic_base_profiles(rho0_base, p0_base, NZ, dz, profiles, true, true, true);
        const float executed_dt = static_cast<float>(dt_small);
        auto prepared_transport = transport_context->prepare_external_acoustic_loop(
            rho, static_cast<double>(executed_dt) * n_substeps, n_substeps);
        auto& wave_state = tmv::require_level_runtime(state.boundary_condition_runtime, "boundary condition runtime");
        const auto yz = std::size_t(NTH) * NZ, xz = std::size_t(NR) * NZ;
        const bool have_history = wave_state.radiative_u_west.size() == yz && wave_state.radiative_u_east.size() == yz &&
            wave_state.radiative_v_south.size() == xz && wave_state.radiative_v_north.size() == xz;
        std::vector<float> wave(2 * (yz + xz)), next_wave(wave.size());
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const auto n = std::size_t(j) * NZ + k;
                wave[n] = have_history ? wave_state.radiative_u_west[n] : u(0, j, k);
                wave[yz + n] = have_history ? wave_state.radiative_u_east[n] : u(NR - 2, j, k);
            }
        for (int i = 0; i < NR; ++i)
            for (int k = 0; k < NZ; ++k)
            {
                const auto n = std::size_t(i) * NZ + k;
                wave[2 * yz + n] = have_history ? wave_state.radiative_v_south[n] : v(i, 0, k);
                wave[2 * yz + xz + n] = have_history ? wave_state.radiative_v_north[n] : v(i, NTH - 2, k);
            }
        tmv::BoundaryConditionWaveState prepared_wave;
        prepared_wave.radiative_u_west.resize(yz);
        prepared_wave.radiative_u_east.resize(yz);
        prepared_wave.radiative_v_south.resize(xz);
        prepared_wave.radiative_v_north.resize(xz);
        AcousticSubstepsBatchedCartesianCgridDispatch dispatch;
        dispatch.u = u.data(); dispatch.v = v.data(); dispatch.w = w.data();
        dispatch.rho = rho.data(); dispatch.p = p.data(); dispatch.theta_data = theta.data();
        dispatch.rho0_base_data = profiles.rho0; dispatch.p0_base_data = profiles.p0;
        dispatch.dp0_dz_data = profiles.dp0_dz; dispatch.dp0_dz_low_data = profiles.dp0_dz_low;
        dispatch.base_len = NZ; dispatch.nx = NR; dispatch.ny = NTH; dispatch.nz = NZ;
        dispatch.dx = dr_f; dispatch.dy = dtheta_f; dispatch.dz = dz_f;
        dispatch.gamma_val = static_cast<float>(dynamics_constants::gamma);
        dispatch.gamma_low = static_cast<float>(dynamics_constants::gamma - static_cast<double>(dispatch.gamma_val));
        dispatch.dt_small = executed_dt; dispatch.n_substeps = n_substeps;
        dispatch.rho_min = density_min_kgm3; dispatch.rho_max = density_max_kgm3;
        dispatch.p_min = pressure_min_pa; dispatch.p_max = pressure_max_pa;
        dispatch.p_ref_fallback = static_cast<float>(p0);
        dispatch.wind_clamp_h = wind_horizontal_abs_max_ms; dispatch.wind_clamp_v = wind_vertical_abs_max_ms;
        const auto& configuration = tmv::require_run_configuration(state);
        dispatch.bc_west = static_cast<int>(configuration.lateral_bc_west);
        dispatch.bc_east = static_cast<int>(configuration.lateral_bc_east);
        dispatch.bc_south = static_cast<int>(configuration.lateral_bc_south);
        dispatch.bc_north = static_cast<int>(configuration.lateral_bc_north);
        dispatch.bc_bottom = static_cast<int>(configuration.bottom_bc);
        dispatch.bc_top = static_cast<int>(configuration.top_bc);
        dispatch.mass_flux_x = transport_context->mass_flux_x.data();
        dispatch.mass_flux_y = transport_context->mass_flux_y.data();
        dispatch.mass_flux_z = transport_context->mass_flux_z.data();
        dispatch.accumulate_mass_flux = true;
        dispatch.mass_flux_weight = static_cast<double>(executed_dt) / transport_context->dt_large();
        dispatch.frame_velocity_u_ms = static_cast<float>(state.domain_frame->velocity_ms.u_ms);
        dispatch.frame_velocity_v_ms = static_cast<float>(state.domain_frame->velocity_ms.v_ms);
        dispatch.repair_primary_state = false;
        dispatch.wave_history = wave.data(); dispatch.wave_history_output = next_wave.data();
        if (!kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(dispatch))
            throw std::runtime_error("native strict C-grid acoustic batch failed before publication");
        std::copy_n(next_wave.data(), yz, prepared_wave.radiative_u_west.data());
        std::copy_n(next_wave.data() + yz, yz, prepared_wave.radiative_u_east.data());
        std::copy_n(next_wave.data() + 2 * yz, xz, prepared_wave.radiative_v_south.data());
        std::copy_n(next_wave.data() + 2 * yz + xz, xz, prepared_wave.radiative_v_north.data());
        static_assert(std::is_nothrow_move_assignable_v<tmv::BoundaryConditionWaveState>);
        static_cast<tmv::BoundaryConditionWaveState&>(wave_state) = std::move(prepared_wave);
        prepared_transport.commit();
        return true;
    }
    // The one-submission batch intentionally keeps intermediate states
    // device-resident. An attributed step needs the pressure/momentum
    // operator barrier for exact work decomposition. Reject the opaque
    // all-substep batch before capture_density_begin mutates transport
    // lifecycle state; the time stepper will use one observable fused
    // pressure/momentum submission per substep.
    if (acoustic_energy_report != nullptr)
        return false;
    if (transport_context != nullptr)
    {
        // The batched accelerator does not enter accumulate(), so preserve
        // the exact scalar-paired density before it advances continuity.
        // This capture is idempotent if the dispatch falls back to the CPU.
        transport_context->capture_density_begin(rho);
    }
    const bool backend_can_export_mass_flux =
        transport_context == nullptr || active_compute_backend_supports_batched_cartesian_cgrid_mass_flux_export(tmv::require_level_runtime(state.compute_runtime, "compute runtime"));
    if (active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid) &&
        backend_can_export_mass_flux && is_cgrid && flat_terrain && static_cast<int>(p0_base.size()) >= NZ &&
        static_cast<int>(rho0_base.size()) >= NZ)
    {
        const float               dt_sb    = static_cast<float>(dt_small);
        const AcousticProfileRows profiles = acoustic_profile_rows(state, NZ);
        fill_acoustic_base_profiles(rho0_base,
                                    p0_base,
                                    NZ,
                                    dz,
                                    profiles,
                                    /*fill_rho0=*/true,
                                    /*fill_p0=*/true,
                                    /*fill_dp0_dz=*/true);
        const auto try_batched = [&]()
        {
            const bool dispatched = dispatch_acoustic_substeps_batched_cartesian_cgrid_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
                u.data(),
                v.data(),
                w.data(),
                rho.data(),
                p.data(),
                theta.data(),
                profiles.rho0,
                profiles.p0,
                profiles.dp0_dz,
                NZ,
                NR,
                NTH,
                NZ,
                dr_f,
                dtheta_f,
                dz_f,
                static_cast<float>(dynamics_constants::gamma),
                dt_sb,
                n_substeps,
                density_min_kgm3,
                density_max_kgm3,
                pressure_min_pa,
                pressure_max_pa,
                static_cast<float>(p0),
                wind_horizontal_abs_max_ms,
                wind_vertical_abs_max_ms,
                static_cast<int>(tmv::require_run_configuration(state).lateral_bc_west),
                static_cast<int>(tmv::require_run_configuration(state).lateral_bc_east),
                static_cast<int>(tmv::require_run_configuration(state).lateral_bc_south),
                static_cast<int>(tmv::require_run_configuration(state).lateral_bc_north),
                static_cast<int>(tmv::require_run_configuration(state).bottom_bc),
                static_cast<int>(tmv::require_run_configuration(state).top_bc),
                transport_context ? transport_context->mass_flux_x.data() : nullptr,
                transport_context ? transport_context->mass_flux_y.data() : nullptr,
                transport_context ? transport_context->mass_flux_z.data() : nullptr,
                transport_context != nullptr,
                static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.u_ms),
                static_cast<float>(tmv::require_level_runtime(state.domain_frame, "domain frame").velocity_ms.v_ms));
            if (dispatched && transport_context != nullptr)
            {
                const double executed_dt = static_cast<double>(dt_sb) * static_cast<double>(n_substeps);
                transport_context->record_external_accumulation(executed_dt,
                                                                n_substeps,
                                                                /*fp32_time_currency=*/true);
            }
            return dispatched;
        };

        const bool parity_sample = nimbus::parity::sample("acoustic_batched_cartesian_cgrid");
        if (!parity_sample)
        {
            if (try_batched())
                return true;
        }
        else
        {
            Field3D pre_u = u, pre_v = v, pre_w = w;
            Field3D pre_rho = rho, pre_p = p;
            if (try_batched())
            {
                const std::size_t bytes = u.size() * sizeof(float);
                Field3D           gpu_u = u, gpu_v = v, gpu_w = w;
                Field3D           gpu_rho = rho, gpu_p = p;

                std::memcpy(u.data(), pre_u.data(), bytes);
                std::memcpy(v.data(), pre_v.data(), bytes);
                std::memcpy(w.data(), pre_w.data(), bytes);
                std::memcpy(rho.data(), pre_rho.data(), bytes);
                std::memcpy(p.data(), pre_p.data(), bytes);

                for (int n = 0; n < n_substeps; ++n)
                {
                    apply_fast_pressure_cpu(state, split_scheme, dt_small, drho_dt, dp_dt);
                    apply_fast_momentum_cpu(state, split_scheme, dt_small, du_dt, dv_dt, dw_dt);
                    bc_scheme->apply_acoustic(dt_small);
                }

                using nimbus::parity::ComparisonDomain;
                using nimbus::parity::Tier;
                const char* kn = "acoustic_batched_cartesian_cgrid";
                // This batch includes acoustic_bc_face_cartesian_cgrid after
                // every substep. Record the acoustic interior and the six-face
                // outer shell separately: their disjoint union is the full
                // field, so a boundary shader cannot be certified by an
                // interior-only verdict. The east/north wall-normal face
                // writes at i=NR-2 / j=NTH-2 lie in the interior record; the
                // ghost planes lie in the shell record.
                const auto check_acoustic_domains = [&](const char* field, const float* cpu, const float* gpu)
                {
                    nimbus::parity::check_field(
                        kn, field, cpu, gpu, NR, NTH, NZ, Tier::Stencil, ComparisonDomain::StrictInterior, n_substeps);
                    nimbus::parity::check_field(
                        kn, field, cpu, gpu, NR, NTH, NZ, Tier::Stencil, ComparisonDomain::BoundaryShell, n_substeps);
                };
                check_acoustic_domains("u", u.data(), gpu_u.data());
                check_acoustic_domains("v", v.data(), gpu_v.data());
                check_acoustic_domains("w", w.data(), gpu_w.data());
                check_acoustic_domains("rho", rho.data(), gpu_rho.data());
                check_acoustic_domains("p", p.data(), gpu_p.data());

                std::memcpy(u.data(), gpu_u.data(), bytes);
                std::memcpy(v.data(), gpu_v.data(), bytes);
                std::memcpy(w.data(), gpu_w.data(), bytes);
                std::memcpy(rho.data(), gpu_rho.data(), bytes);
                std::memcpy(p.data(), gpu_p.data(), bytes);
                // The parity replay temporarily writes every acoustic host
                // field while the named device slots still describe the
                // production dispatch.  Restoring byte-identical endpoints
                // is not sufficient ownership evidence for later resident
                // consumers: the replay also advances CPU boundary/history
                // state and may touch retained flux storage.  Force the next
                // operator to upload the authoritative restored host state.
                // This cost exists only on sampled validation steps.
                invalidate_all_resident_fields_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"));
                return true;
            }
        }
    }

    // As above, the legacy collocated batch has no retained A-grid face-flux
    // export. Returning false selects the ordinary acoustic loop and prevents
    // an empty context or an end-state scalar-flux reconstruction.
    if (transport_context != nullptr)
        return false;
    if (!gpu_acoustic_ok || !flat_terrain)
        return false;
    const float dt_s = static_cast<float>(dt_small);
    return dispatch_acoustic_substeps_batched_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), u.data(),
                                                      v.data(),
                                                      w.data(),
                                                      rho.data(),
                                                      p.data(),
                                                      NR,
                                                      NTH,
                                                      NZ,
                                                      dr_f,
                                                      dtheta_f,
                                                      dz_f,
                                                      static_cast<float>(dynamics_constants::gamma),
                                                      dt_s,
                                                      n_substeps,
                                                      density_min_kgm3,
                                                      pressure_min_pa,
                                                      wind_horizontal_abs_max_ms,
                                                      wind_vertical_abs_max_ms);
}

static void prepare_slow_rk2_tendencies(const SimulationStateView&            state,
                                        tmv::TimeSteppingSlowRk2Workspace&    workspace,
                                        double                                dt_dynamics,
                                        SplitExplicitDynamics&                split_scheme,
                                        SplitExplicitCallbacks&               callbacks,
                                        Field3D&                              du_dt,
                                        Field3D&                              dv_dt,
                                        Field3D&                              dw_dt,
                                        Field3D&                              drho_dt,
                                        Field3D&                              dp_dt,
                                        bool&                                 commit_pbl_work,
                                        SlowMomentumAdvectionTendencyOutputs* advection_outputs)
{
    // This scope is the only place where the engine promises that rho,
    // moisture species, hydrometeor loading, and base-state profiles remain
    // unchanged across two consecutive slow evaluations. Expose that narrow
    // contract to accelerator implementations; never infer it from stable
    // host pointers in standalone calls. The destructor closes the window
    // if either stage or the predictor throws.
    struct SlowRk2ResidencyScope
    {
        explicit SlowRk2ResidencyScope(SplitExplicitDynamics& value) : scheme(value)
        {
            scheme.set_slow_rk2_invariant_input_residency(true);
        }
        ~SlowRk2ResidencyScope()
        {
            scheme.set_slow_rk2_invariant_input_residency(false);
        }
        SplitExplicitDynamics& scheme;
    } residency_scope(split_scheme);

    struct SlowInterfaceFluxWeightScope
    {
        explicit SlowInterfaceFluxWeightScope(SplitExplicitDynamics& value)
            : scheme(value)
        {
            if (scheme.momentum_interface_flux_capture_active())
                scheme.set_momentum_interface_flux_slow_weight(0.5);
        }
        ~SlowInterfaceFluxWeightScope()
        {
            if (!scheme.momentum_interface_flux_capture_active())
                return;
            try
            {
                scheme.set_momentum_interface_flux_slow_weight(1.0);
            }
            catch (...)
            {
            }
        }
        SplitExplicitDynamics& scheme;
    } flux_weight_scope(split_scheme);

    auto&       fields = state.fields;
    const auto& grid   = state.grid;
    // Stage 1: f_slow at u_n with current_time -> k1 in du_dt etc.
    callbacks.compute_slow_tendencies();

    auto ensure_size = [&](Field3D& f)
    {
        if (f.size_r() != grid.nr || f.size_th() != grid.nth || f.size_z() != grid.nz)
        {
            f.resize(grid.nr, grid.nth, grid.nz, 0.0f);
        }
    };
    ensure_size(workspace.u_save);
    ensure_size(workspace.v_save);
    ensure_size(workspace.w_save);
    ensure_size(workspace.p_save);
    ensure_size(workspace.du_dt_k1);
    ensure_size(workspace.dv_dt_k1);
    ensure_size(workspace.dw_dt_k1);
    ensure_size(workspace.dp_dt_k1);
    if (advection_outputs != nullptr)
    {
        ensure_size(workspace.du_advection_dt_k1);
        ensure_size(workspace.dv_advection_dt_k1);
        ensure_size(workspace.dw_advection_dt_k1);
    }

    workspace.u_save   = fields.u;
    workspace.v_save   = fields.v;
    workspace.w_save   = fields.w;
    workspace.p_save   = fields.p;
    workspace.du_dt_k1 = du_dt;
    workspace.dv_dt_k1 = dv_dt;
    workspace.dw_dt_k1 = dw_dt;
    workspace.dp_dt_k1 = dp_dt;
    if (advection_outputs != nullptr)
    {
        workspace.du_advection_dt_k1 = *advection_outputs->du_dt;
        workspace.dv_advection_dt_k1 = *advection_outputs->dv_dt;
        workspace.dw_advection_dt_k1 = *advection_outputs->dw_dt;
    }

    // Predictor step: u' = u_n + dt * k1 (slow only, no acoustic).
    commit_pbl_work = false;
    auto& predictor_runtime = dynamics_runtime_for(state);
    const bool capture_predictor =
        predictor_runtime.momentum_interface_flux_capture_enabled;
    predictor_runtime.momentum_interface_flux_capture_enabled = false;
    try
    {
        callbacks.apply_slow_tendencies(dt_dynamics);
    }
    catch (...)
    {
        predictor_runtime.momentum_interface_flux_capture_enabled =
            capture_predictor;
        throw;
    }
    predictor_runtime.momentum_interface_flux_capture_enabled =
        capture_predictor;
    commit_pbl_work = true;

    const auto slow_compute_started = dynamics_perf_start();
    split_scheme.compute_slow_tendencies(fields.u,
                                         fields.v,
                                         fields.w,
                                         fields.rho,
                                         fields.p,
                                         fields.theta,
                                         dt_dynamics,
                                         du_dt,
                                         dv_dt,
                                         dw_dt,
                                         drho_dt,
                                         dp_dt,
                                         advection_outputs);
    dynamics_perf_finish(dynamics_perf_totals.slow_compute_s, slow_compute_started);

    fields.u = workspace.u_save;
    fields.v = workspace.v_save;
    fields.w = workspace.w_save;
    fields.p = workspace.p_save;
    // Heun stage one is a predictor only. Restoring the caller-owned state is
    // a real host write, so the named slots that still contain the predictor
    // endpoint cannot seed the final averaged slow commit.
    invalidate_named_fields(state, {"u", "v", "w", "p"});

#pragma omp parallel for collapse(2)
    for (int i = 0; i < grid.nr; ++i)
        for (int j = 0; j < grid.nth; ++j)
            for (int k = 0; k < grid.nz; ++k)
            {
                du_dt[i][j][k] = 0.5f * (du_dt[i][j][k] + workspace.du_dt_k1[i][j][k]);
                dv_dt[i][j][k] = 0.5f * (dv_dt[i][j][k] + workspace.dv_dt_k1[i][j][k]);
                dw_dt[i][j][k] = 0.5f * (dw_dt[i][j][k] + workspace.dw_dt_k1[i][j][k]);
                dp_dt[i][j][k] = 0.5f * (dp_dt[i][j][k] + workspace.dp_dt_k1[i][j][k]);
                if (advection_outputs != nullptr)
                {
                    (*advection_outputs->du_dt)[i][j][k] =
                        0.5f * ((*advection_outputs->du_dt)[i][j][k] + workspace.du_advection_dt_k1[i][j][k]);
                    (*advection_outputs->dv_dt)[i][j][k] =
                        0.5f * ((*advection_outputs->dv_dt)[i][j][k] + workspace.dv_advection_dt_k1[i][j][k]);
                    (*advection_outputs->dw_dt)[i][j][k] =
                        0.5f * ((*advection_outputs->dw_dt)[i][j][k] + workspace.dw_advection_dt_k1[i][j][k]);
                }
            }

    // The CPU average supersedes the resident second-stage tendencies.
    invalidate_named_fields(state, {"slow_du_dt", "slow_dv_dt", "slow_dw_dt", "slow_dp_dt"});

    // The time stepper will still call compute_slow_tendencies before
    // applying slow tendencies; keep it from overwriting the averaged values.
    callbacks.compute_slow_tendencies = []() { /* slow already done */ };
}

void step_dynamics_split_explicit(const SimulationStateView&            state,
                                  double                                dt_dynamics,
                                  [[maybe_unused]] double               current_time,
                                  SplitExplicitDynamics&                split_scheme,
                                  Field3D&                              du_dt,
                                  Field3D&                              dv_dt,
                                  Field3D&                              dw_dt,
                                  Field3D&                              drho_dt,
                                  Field3D&                              dp_dt,
                                  tmv::CartesianCGridTransportContext*  transport_context,
                                  CartesianCGridAcousticEnergyReportV1* acoustic_energy_report,
                                  SlowMomentumAdvectionTendencyOutputs* advection_outputs,
                                  CartesianSlowMomentumEnergyReportV1*  momentum_energy_report)
{
    auto&                     bc_scheme = tmv::boundary_scheme_for(state);
    TimeSteppingRuntimeState& time_runtime =
        state.time_stepping_runtime ? *state.time_stepping_runtime
                                    : tmv::require_level_runtime(state.time_stepping_runtime, "time_stepping_runtime");
    if (!time_runtime.scheme)
    {
        throw std::runtime_error("split-explicit dynamics requires an initialized time stepper");
    }
    if (state.configuration->admission.earth_fixed_terrain_motion && state.compute_runtime &&
        state.compute_runtime->active_kind != ComputeBackendKind::Cpu)
    {
        if (!transport_context || state.compute_runtime->fallback_active)
            throw std::runtime_error("native Earth-fixed evolution lost its retained device transport owner");
        const int substeps = time_runtime.scheme->resolve_substep_count(time_runtime.config, dt_dynamics);
        tmv::terrain_native::MappedAcousticResult result;
        if (!tmv::terrain_native::advance_mapped_acoustic_core(state, dt_dynamics, substeps, result) ||
            !result.complete)
            throw std::runtime_error("native Earth-fixed acoustic evolution failed");
        transport_context->capture_density_begin(result.density_begin);
        transport_context->mass_flux_x = std::move(result.mass_flux_x);
        transport_context->mass_flux_y = std::move(result.mass_flux_y);
        transport_context->mass_flux_z = std::move(result.mass_flux_z);
        transport_context->record_external_carrier_kinetic_transport(
            result.acoustic.continuity_carrier_outward_boundary_transport_j, substeps);
        transport_context->record_external_accumulation(
            double(float(dt_dynamics / substeps)) * substeps, substeps, true);
        if (acoustic_energy_report)
            *acoustic_energy_report = result.acoustic;
        if (momentum_energy_report)
            *momentum_energy_report = result.slow;
        commit_boundary_layer_momentum_work(state, result.pbl);
        tmv::log_info("[EARTH_NATIVE] acoustic_substeps=",
                      substeps,
                      ", backend_dispatches=",
                      result.completed_backend_dispatches,
                      ", fallback=off");
        return;
    }
    SplitExplicitCallbacks callbacks;
    bool                   commit_pbl_work = true;
    const bool             flat_terrain    = !tmv::physical_grid_metrics::terrain_is_active(state);
    const bool             is_cgrid        = state.grid.geometry.staggered;
    const bool             native_complete_agrid_required =
        !is_cgrid && flat_terrain &&
        active_compute_backend_kind(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) !=
            ComputeBackendKind::Cpu &&
        !compute_backend_fallback_active(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) &&
        !tmv::require_level_runtime(state.compute_runtime, "compute_runtime").config.allow_fallback &&
        state.configuration != nullptr && state.configuration->gpu_apply_slow;

    callbacks.compute_slow_tendencies = [&]()
    {
        const auto started = dynamics_perf_start();
        split_scheme.compute_slow_tendencies(state.fields.u,
                                             state.fields.v,
                                             state.fields.w,
                                             state.fields.rho,
                                             state.fields.p,
                                             state.fields.theta,
                                             dt_dynamics,
                                             du_dt,
                                             dv_dt,
                                             dw_dt,
                                             drho_dt,
                                             dp_dt,
                                             advection_outputs);
        dynamics_perf_finish(dynamics_perf_totals.slow_compute_s, started);
    };

    // Re-close all boundary-owned faces after the large-step slow update and
    // before continuity samples its first mass flux.  In particular, the
    // C-grid slow operator includes the east/north face storage in its loops;
    // without this zero-time closure a rigid wall can leak during the first
    // acoustic substep.  dt=0 preserves a radiative normal face while closing
    // walls, periodic pairs, zero-gradient faces, ghosts, and tangentials.
    // Seed KW78 history only after that closure so both per-substep and batched
    // paths start from the identical admissible post-slow state.
    callbacks.seed_acoustic_radiation = [&]()
    {
        const auto started = dynamics_perf_start();
        if (native_complete_agrid_required)
        {
            if (!try_apply_flat_agrid_boundary_gpu(state, acoustic_energy_report))
                throw std::runtime_error("fallback-off native A-grid seed boundary dispatch failed");
        }
        else if (acoustic_energy_report != nullptr)
            bc_scheme->apply_acoustic_attributed(0.0, *acoustic_energy_report);
        else
            bc_scheme->apply_acoustic(0.0);
        bc_scheme->seed_acoustic_radiation();
        // This CPU boundary owner runs after the slow device commit and may
        // change wall/periodic/radiative faces plus scalar closure planes.
        // The following acoustic batch must therefore not reuse the
        // pre-closure named slots merely because the host pointers match.
        invalidate_named_fields(state, {"u", "v", "w", "rho", "p"});
        dynamics_perf_finish(dynamics_perf_totals.acoustic_boundary_s, started);
    };

    callbacks.apply_slow_tendencies = [&](double dt_large)
    {
        const auto started = dynamics_perf_start();
        const auto energy_before =
            compute_moist_mixture_energy_v2_deterministic(state);
        apply_slow_tendencies(
            state, dt_large, du_dt, dv_dt, dw_dt, dp_dt, commit_pbl_work, advection_outputs, momentum_energy_report);
        if (momentum_energy_report != nullptr
            && momentum_energy_report->complete)
        {
            // Slow RK stages author requested work, but only this callback
            // owns the final stored Heun endpoint.
            const auto energy_after =
                compute_moist_mixture_energy_v2_deterministic(state);
            const double committed_kinetic =
                energy_after.resolved_kinetic
                - energy_before.resolved_kinetic;
            const double pressure_dry =
                energy_after.dry_internal - energy_before.dry_internal;
            const double pressure_water =
                energy_after.water_sensible
                - energy_before.water_sensible;
            const double requested =
                momentum_energy_report->advection_requested_linear_work_j
                + momentum_energy_report->nonadvection_requested_linear_work_j
                + momentum_energy_report->advection_finite_step_correction_j
                + momentum_energy_report->nonadvection_finite_step_correction_j
                + momentum_energy_report->coupled_finite_step_cross_work_j;
            const double values[] = {
                committed_kinetic, pressure_dry, pressure_water, requested,
            };
            if (std::any_of(
                    std::begin(values), std::end(values),
                    [](double value) { return !std::isfinite(value); }))
                throw std::runtime_error(
                    "slow RK commit energy attribution overflowed");
            momentum_energy_report->committed_kinetic_change_j =
                committed_kinetic;
            momentum_energy_report->storage_rounding_j =
                committed_kinetic - requested;
            momentum_energy_report->closure_residual_j = 0.0;
            momentum_energy_report->pressure_dry_internal_change_j =
                pressure_dry;
            momentum_energy_report->pressure_water_sensible_change_j =
                pressure_water;
            momentum_energy_report->committed_pressure_energy_change_j =
                pressure_dry + pressure_water;
        }
        dynamics_perf_finish(dynamics_perf_totals.slow_apply_s, started);
    };

    // Fast-tendency buffers are allocated once inside the scheme. The fused
    // C-grid CPU path retains only drho_dt as its continuity barrier scratch;
    // separate and accelerator paths keep the complete buffer contract.
    // The legacy collocated GPU acoustic kernels (dispatch_acoustic_*_backend)
    // expect a collocated A-grid state. A C-grid scheme expects staggered
    // face placements with one-sided pressure gradients and flux-form
    // divergence; running the collocated kernels on a C-grid state
    // would silently corrupt both p/rho and u/v/w. Phase C.9 ships
    // C-grid-specific shaders; the per-callback dispatch picks the
    // right one based on global_stagger_type.
    // The legacy collocated GPU kernels do not implement the
    // momentum-weighted pressure face flux now required by A-grid continuity
    // and retained scalar transport.  Keep that path on the CPU until the
    // device kernels export the identical corrected mass-flux history.
    const bool             gpu_acoustic_ok = false;
    const float            dr_f            = static_cast<float>(state.grid.dr);
    const float            dtheta_f        = static_cast<float>(state.grid.dtheta);
    const float            dz_f            = static_cast<float>(state.grid.dz);
    tmv::CartesianCGridTransportContext acoustic_transport_staging;
    AcousticEnergyPending  acoustic_energy_pending;
    AcousticEnergyPending* attributed_separate_energy = nullptr;
    if (acoustic_energy_report != nullptr)
    {
        // Flat CPU C-grid normally attributes inside its fused writer. Warn
        // and strict deliberately bypass that repairing writer, so ownership
        // must move to the separate pressure/momentum callbacks as well.
        const bool separate_flat_cgrid =
            is_cgrid && flat_terrain &&
            (active_compute_backend_kind(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) != ComputeBackendKind::Cpu || !primary_state_repairs_enabled);
        if (!is_cgrid || (is_cgrid && !flat_terrain) || separate_flat_cgrid)
        {
            acoustic_energy_pending.report = acoustic_energy_report;
            acoustic_energy_pending.layout =
                !is_cgrid
                    ? (flat_terrain ? AcousticEnergyLayout::FlatAGrid
                                    : AcousticEnergyLayout::TerrainAGrid)
                          : (flat_terrain ? AcousticEnergyLayout::FlatCGrid : AcousticEnergyLayout::TerrainCGrid);
            attributed_separate_energy = &acoustic_energy_pending;
        }
    }

    callbacks.apply_fast_pressure = [&, is_cgrid, flat_terrain, dr_f, dtheta_f, dz_f](double dt_small)
    {
        apply_fast_pressure(state,
                            split_scheme,
                            dt_small,
                            drho_dt,
                            dp_dt,
                            is_cgrid,
                            flat_terrain,
                            dr_f,
                            dtheta_f,
                            dz_f,
                            transport_context,
                            acoustic_energy_report != nullptr,
                            attributed_separate_energy);
    };

    callbacks.close_fast_pressure_boundaries = [&, is_cgrid, flat_terrain]()
    {
        if (!is_cgrid || !flat_terrain)
        {
            // Collocated and mapped face pressure gradients use scalar ghosts.
            // Continuity has
            // just advanced physical rho while leaving ghost tendencies zero,
            // and the EOS barrier has diagnosed p from those two time levels.
            // Close rho/p now, before the backward momentum pass can turn the
            // temporary jump into horizontal acceleration. dt=0 does not
            // advance a time-dependent boundary condition; collocated
            // radiative lateral faces are rejected by configuration.
            if (native_complete_agrid_required)
            {
                if (!try_apply_flat_agrid_boundary_gpu(state, nullptr))
                    throw std::runtime_error("fallback-off native A-grid pressure boundary dispatch failed");
            }
            else
                bc_scheme->apply_acoustic(0.0);
        }
    };

    const auto acoustic_energy_snapshot = [&]()
    {
        if (acoustic_energy_report == nullptr)
            return MoistMixtureEnergyBudgetV2{};
        return measured_acoustic_energy_snapshot(state);
    };

    callbacks.apply_fast_momentum = [&, flat_terrain, is_cgrid, dr_f, dtheta_f, dz_f](double dt_small)
    {
        const bool reuse_inventory = acoustic_energy_report != nullptr && flat_terrain && is_cgrid &&
            state.grid.geometry.coord == CoordinateSystem::Cartesian &&
            state.grid.nr >= 3 && state.grid.nth >= 3 && state.grid.nz >= 2 &&
            active_compute_backend_kind(tmv::require_level_runtime(state.compute_runtime, "compute runtime")) == ComputeBackendKind::Cpu;
        // Momentum changes only velocity; retain the validated shared inventories.
        const auto complete_before = reuse_inventory
            ? measured_acoustic_complete_energy_snapshot(state) : EnergyDiagnosticSnapshotV1{};
        const auto energy_before =
            reuse_inventory ? complete_before.moist_mixture_v2 : acoustic_energy_snapshot();
        const double report_before = acoustic_energy_report != nullptr
            ? acoustic_energy_report->pressure_gradient_committed_work_j
            : 0.0;
        apply_fast_momentum(state,
                            split_scheme,
                            dt_small,
                            du_dt,
                            dv_dt,
                            dw_dt,
                            flat_terrain,
                            gpu_acoustic_ok,
                            is_cgrid,
                            dr_f,
                            dtheta_f,
                            dz_f,
                            attributed_separate_energy);
        if (acoustic_energy_report != nullptr)
        {
            // Reconcile the report to the stored momentum endpoint; requested
            // and finite-step terms remain independently authored above.
            const double kinetic_after = reuse_inventory
                ? measured_acoustic_velocity_energy_completion(state, complete_before)
                : acoustic_energy_snapshot().resolved_kinetic;
            const double committed = kinetic_after - energy_before.resolved_kinetic;
            const double reported =
                acoustic_energy_report->pressure_gradient_committed_work_j
                - report_before;
            const double correction = committed - reported;
            if (!std::isfinite(correction))
                throw std::runtime_error(
                    "fast momentum energy commit overflowed");
            acoustic_energy_report->pressure_gradient_committed_work_j +=
                correction;
        }
        if (!is_cgrid)
        {
            if (native_complete_agrid_required)
            {
                if (!try_apply_flat_agrid_constraint_gpu(state, attributed_separate_energy))
                    throw std::runtime_error("fallback-off native A-grid null-mode constraint dispatch failed");
                return;
            }
            using LateralBC   = tmv::init::config::LateralBC;
            const auto closed = [](LateralBC bc) { return bc == LateralBC::Rigid || bc == LateralBC::RigidSlip; };
            const bool constrain_x =
                (tmv::require_run_configuration(state).lateral_bc_west == LateralBC::Periodic && tmv::require_run_configuration(state).lateral_bc_east == LateralBC::Periodic) ||
                (closed(tmv::require_run_configuration(state).lateral_bc_west) && closed(tmv::require_run_configuration(state).lateral_bc_east));
            const bool constrain_y =
                (tmv::require_run_configuration(state).lateral_bc_south == LateralBC::Periodic && tmv::require_run_configuration(state).lateral_bc_north == LateralBC::Periodic) ||
                (closed(tmv::require_run_configuration(state).lateral_bc_south) && closed(tmv::require_run_configuration(state).lateral_bc_north));
            const bool constrain_z = tmv::require_run_configuration(state).top_bc != tmv::init::config::TopBC::Open;

            // Adjacent-cell face averaging has an exact alternating-velocity
            // null only when both axis faces are impermeable or periodically
            // paired. An open/zero-gradient face gives the mode a boundary
            // flux, so do not project that direction. The ordinary acoustic
            // BC call below refreshes ghosts after each selected projection.
            double  constraint_work_j = 0.0;
            double* attributed_constraint_work =
                attributed_separate_energy != nullptr &&
                        (attributed_separate_energy->layout == AcousticEnergyLayout::FlatAGrid
                         || attributed_separate_energy->layout == AcousticEnergyLayout::TerrainAGrid)
                    ? &constraint_work_j
                    : nullptr;
            tmv::collocated_velocity::constrain_directional_null_modes(state.fields.u,
                                                                       state.fields.v,
                                                                       state.fields.w,
                                                                       state.fields.rho,
                                                                       state.grid.nr,
                                                                       state.grid.nth,
                                                                       state.grid.nz,
                                                                       constrain_x,
                                                                       constrain_y,
                                                                       constrain_z,
                                                                       {{
                                                                           &state.fields.qv,
                                                                           &state.fields.qc,
                                                                           &state.fields.qr,
                                                                           &state.fields.qi,
                                                                           &state.fields.qs,
                                                                           &state.fields.qg,
                                                                           &state.fields.qh,
                                                                       }},
                                                                       state.grid.dr * state.grid.dtheta *
                                                                           state.grid.dz,
                                                                       attributed_constraint_work);
            if (attributed_constraint_work != nullptr)
            {
                if (!std::isfinite(constraint_work_j))
                {
                    throw std::runtime_error("collocated acoustic constraint produced invalid "
                                             "basis-v2 work");
                }
                CartesianCGridAcousticEnergyReportV1 next = *acoustic_energy_report;
                next.boundary_committed_work_j += constraint_work_j;
                next.boundary_constraint_history_committed_work_j += constraint_work_j;
                next.committed_inventory_change_j += constraint_work_j;
                const double values[] = {
                    next.boundary_committed_work_j,
                    next.boundary_constraint_history_committed_work_j,
                    next.committed_inventory_change_j,
                };
                if (std::any_of(
                        std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
                {
                    throw std::overflow_error("collocated acoustic constraint attribution "
                                              "overflowed");
                }
                *acoustic_energy_report = next;
            }
        }
    };

    // Fused path: pressure substep + barrier + momentum substep in one backend
    // operation. CPU C-grid keeps one OpenMP team across its three passes;
    // GPU C-grid uses the C.9.6 fused dispatch
    // (records the C.9.4 + C.9.5 pipelines back-to-back with a
    // compute-to-compute barrier); collocated uses the existing
    // single-shader fused path.
    callbacks.apply_fast_fused = [&, flat_terrain, is_cgrid, dr_f, dtheta_f, dz_f](double dt_small) -> bool
    {
        const auto energy_before =
            primary_state_repairs_enabled ? acoustic_energy_snapshot() : MoistMixtureEnergyBudgetV2{};
        const CartesianCGridAcousticEnergyReportV1 report_before =
            acoustic_energy_report != nullptr
            ? *acoustic_energy_report
            : CartesianCGridAcousticEnergyReportV1{};
        const bool applied = apply_fast_fused(state,
                                split_scheme,
                                dt_small,
                                du_dt,
                                dv_dt,
                                dw_dt,
                                drho_dt,
                                dp_dt,
                                flat_terrain,
                                gpu_acoustic_ok,
                                is_cgrid,
                                dr_f,
                                dtheta_f,
                                dz_f,
                                transport_context,
                                acoustic_transport_staging,
                                acoustic_energy_report);
        if (applied && acoustic_energy_report != nullptr && primary_state_repairs_enabled)
        {
            const auto energy_after =
                acoustic_energy_snapshot();
            const double reported =
                acoustic_energy_report->continuity_carrier_kinetic_change_j
                - report_before.continuity_carrier_kinetic_change_j
                + acoustic_energy_report->pressure_gradient_committed_work_j
                - report_before.pressure_gradient_committed_work_j;
            const double correction =
                energy_after.resolved_kinetic
                - energy_before.resolved_kinetic - reported;
            if (!std::isfinite(correction))
                throw std::runtime_error(
                    "fused fast energy commit overflowed");
            acoustic_energy_report->pressure_gradient_committed_work_j +=
                correction;
        }
        return applied;
    };

    // Batched path: all N substeps in one submission
    callbacks.batched_acoustic_boundary_complete = !primary_state_repairs_enabled;
    callbacks.apply_fast_batched = [&, flat_terrain, is_cgrid, dr_f, dtheta_f, dz_f](double dt_small,
                                                                                     int    n_substeps) -> bool
    {
        if (acoustic_energy_report != nullptr) return false;
        const bool applied = apply_fast_batched(state,
                                  split_scheme,
                                  dt_small,
                                  n_substeps,
                                  du_dt,
                                  dv_dt,
                                  dw_dt,
                                  drho_dt,
                                  dp_dt,
                                  flat_terrain,
                                  gpu_acoustic_ok,
                                  is_cgrid,
                                  dr_f,
                                  dtheta_f,
                                  dz_f,
                                  transport_context,
                                  acoustic_energy_report);
        if (applied && !primary_state_repairs_enabled)
            invalidate_named_fields(state, {"u", "v", "w", "rho", "p"});
        return applied;
    };

    callbacks.acoustic_bcs = [&](double dt_small)
    {
        const auto started = dynamics_perf_start();
        if (native_complete_agrid_required)
        {
            if (!try_apply_flat_agrid_boundary_gpu(state, acoustic_energy_report))
                throw std::runtime_error("fallback-off native A-grid acoustic boundary dispatch failed");
        }
        else if (acoustic_energy_report != nullptr)
            bc_scheme->apply_acoustic_attributed(dt_small, *acoustic_energy_report);
        else
            bc_scheme->apply_acoustic(dt_small);
        invalidate_named_fields(state, {"u", "v", "w", "rho", "p"});
        dynamics_perf_finish(dynamics_perf_totals.acoustic_boundary_s, started);
    };
    callbacks.begin_acoustic_loop_residency = [&state, flat_terrain, is_cgrid, acoustic_energy_report]()
    {
        ComputeBackend* backend = mutable_compute_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"));
        DynamicsComputeKernels* kernels = backend != nullptr ? backend->dynamics_kernels() : nullptr;
        if (kernels != nullptr)
        {
            const bool unchanged_between_operators = flat_terrain && is_cgrid && acoustic_energy_report == nullptr;
            kernels->begin_acoustic_loop_residency(unchanged_between_operators, unchanged_between_operators);
        }
    };
    callbacks.release_acoustic_loop_residency = [&state]()
    {
        ComputeBackend*         backend = mutable_compute_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"));
        DynamicsComputeKernels* kernels = backend != nullptr ? backend->dynamics_kernels() : nullptr;
        if (kernels != nullptr)
            kernels->release_acoustic_loop_residency();
    };

    // Phase H — RK2 (Heun's method) on the slow path when enabled. We do
    // the two-stage slow integration here in the orchestration layer,
    // ending with the averaged tendencies in du_dt etc. and the state
    // restored to u_n. We then override compute_slow_tendencies to a
    // no-op (so step_split_acoustic doesn't recompute and overwrite our
    // averaged tendencies) before delegating to the time stepper for the
    // apply_slow + acoustic-substep loop.
    if (slow_rk2_enabled(state))
    {
        prepare_slow_rk2_tendencies(state,
                                    time_runtime.slow_rk2,
                                    dt_dynamics,
                                    split_scheme,
                                    callbacks,
                                    du_dt,
                                    dv_dt,
                                    dw_dt,
                                    drho_dt,
                                    dp_dt,
                                    commit_pbl_work,
                                    advection_outputs);
    }

    // Delegate to the time stepping scheme.
    time_runtime.scheme->step_split_acoustic(time_runtime.config, dt_dynamics, callbacks);
}

static double canonical_acoustic_committed_inventory(const CartesianCGridAcousticEnergyReportV1& report)
{
    // This field is a published identity, not an independent reduction.
    // Recompose in the same stable owner order used by report consumers so
    // substep-by-substep accumulation cannot create a second rounding truth.
    double committed = 0.0;
    committed += report.continuity_eos_dry_internal_change_j;
    committed += report.continuity_eos_water_sensible_change_j;
    committed += report.continuity_eos_phase_reference_change_j;
    committed += report.continuity_carrier_kinetic_change_j;
    committed += report.continuity_gravitational_change_j;
    committed += report.continuity_subgrid_tke_change_j;
    committed += report.pressure_gradient_committed_work_j;
    committed += report.boundary_committed_work_j;
    return committed;
}

static void append_coupled_physics_energy(
    const SimulationStateView& state, const TimeSteppingRuntimeState& time_runtime,
    tmv::CompletedStepEnergyReportV1& report,
    const std::array<double, 6>* integrator_prefix = nullptr)
{
    const MicrophysicsRuntimeState& microphysics_runtime =
        state.microphysics_runtime ? *state.microphysics_runtime : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime");
    const TurbulenceRuntimeState& turbulence_runtime =
        state.turbulence_runtime ? *state.turbulence_runtime : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime");
    const DiffusionRuntimeState& diffusion_runtime =
        state.diffusion_runtime ? *state.diffusion_runtime : tmv::require_level_runtime(state.diffusion_runtime, "diffusion_runtime");
    const auto& thermo              = microphysics_runtime.last_thermodynamic_repair_report;
    const auto& sgs                 = turbulence_runtime.last_momentum_work;
    const auto& diffusion           = diffusion_runtime.last_momentum_work;
    const auto& eos                 = time_runtime.last_eos_pressure_energy;
    const auto& sponge              = time_runtime.last_sponge_energy;
    const auto& final_boundary      = time_runtime.last_final_boundary_energy;
    const auto  moist_owner_is_zero = [](const auto& owner)
    {
        const double values[] = {
            owner.requested_linear_work_j,
            owner.finite_step_correction_j,
            owner.storage_rounding_j,
            owner.committed_work_j,
            owner.moist_dry_internal_change_j,
            owner.moist_water_sensible_change_j,
            owner.moist_phase_reference_change_j,
            owner.moist_carrier_kinetic_change_j,
            owner.moist_gravitational_change_j,
            owner.moist_subgrid_tke_change_j,
            owner.moist_committed_energy_change_j,
            owner.moist_energy_closure_residual_j,
        };
        return std::all_of(std::begin(values), std::end(values), [](double value) { return value == 0.0; });
    };
    const bool sgs_complete =
        sgs.moist_energy_complete || (turbulence_runtime.config.scheme_id == "none" && moist_owner_is_zero(sgs));
    const bool diffusion_inactive = diffusion_runtime.config.K_h == 0.0 && diffusion_runtime.config.K_v == 0.0;
    const bool diffusion_complete =
        diffusion.moist_energy_complete || (diffusion_inactive && moist_owner_is_zero(diffusion));

    if (!thermo.moist_energy_complete || !sgs_complete || !diffusion_complete ||
        !eos.complete || !sponge.complete || !final_boundary.complete)
        throw std::runtime_error("completed-step energy composition received a partial physics owner");

    report.thermodynamic_sources_committed_change_j =
        thermo.moist_committed_energy_change_j;
    report.sgs_committed_change_j = sgs.moist_committed_energy_change_j;
    report.diffusion_committed_change_j =
        diffusion.moist_committed_energy_change_j;
    report.eos_projection_committed_change_j = eos.committed_energy_change_j;
    report.sponge_committed_change_j =
        sponge.upper.committed_energy_change_j +
        sponge.w_only.committed_energy_change_j +
        sponge.lateral.committed_energy_change_j;
    report.final_boundary_committed_change_j =
        final_boundary.committed_energy_change_j;

    // Restart identity requires the established left-to-right sum, including
    // each integrator's prefix before the common physics contributions.
    const auto ordered_sum = [&](std::size_t reservoir, std::initializer_list<double> terms) {
        auto term = terms.begin();
        double value = integrator_prefix ? (*integrator_prefix)[reservoir] : *term++;
        for (; term != terms.end(); ++term) value += *term;
        return value;
    };
    report.dry_internal_change_j += ordered_sum(0, {
        thermo.moist_dry_internal_change_j,
        sgs.moist_dry_internal_change_j,
        diffusion.moist_dry_internal_change_j,
        eos.dry_internal_change_j,
        sponge.upper.dry_internal_change_j,
        sponge.w_only.dry_internal_change_j,
        sponge.lateral.dry_internal_change_j,
        final_boundary.dry_internal_change_j});
    report.water_sensible_change_j += ordered_sum(1, {
        thermo.moist_water_sensible_change_j,
        sgs.moist_water_sensible_change_j,
        diffusion.moist_water_sensible_change_j,
        eos.water_sensible_change_j,
        sponge.upper.water_sensible_change_j,
        sponge.w_only.water_sensible_change_j,
        sponge.lateral.water_sensible_change_j,
        final_boundary.water_sensible_change_j});
    report.phase_reference_change_j += ordered_sum(2, {
        thermo.moist_phase_reference_change_j,
        sgs.moist_phase_reference_change_j,
        diffusion.moist_phase_reference_change_j,
        eos.phase_reference_change_j,
        sponge.upper.phase_reference_change_j,
        sponge.w_only.phase_reference_change_j,
        sponge.lateral.phase_reference_change_j,
        final_boundary.phase_reference_change_j});
    report.carrier_kinetic_change_j += ordered_sum(3, {
        thermo.moist_carrier_kinetic_change_j,
        sgs.committed_work_j,
        sgs.moist_carrier_kinetic_change_j,
        diffusion.committed_work_j,
        diffusion.moist_carrier_kinetic_change_j,
        eos.carrier_kinetic_change_j,
        sponge.upper.carrier_kinetic_change_j,
        sponge.w_only.carrier_kinetic_change_j,
        sponge.lateral.carrier_kinetic_change_j,
        final_boundary.carrier_kinetic_change_j});
    report.gravitational_change_j += ordered_sum(4, {
        thermo.moist_gravitational_change_j,
        sgs.moist_gravitational_change_j,
        diffusion.moist_gravitational_change_j,
        eos.gravitational_change_j,
        sponge.upper.gravitational_change_j,
        sponge.w_only.gravitational_change_j,
        sponge.lateral.gravitational_change_j,
        final_boundary.gravitational_change_j});
    report.subgrid_tke_change_j += ordered_sum(5, {
        thermo.moist_subgrid_tke_change_j,
        sgs.moist_subgrid_tke_change_j,
        diffusion.moist_subgrid_tke_change_j,
        eos.subgrid_tke_change_j,
        sponge.upper.subgrid_tke_change_j,
        sponge.w_only.subgrid_tke_change_j,
        sponge.lateral.subgrid_tke_change_j,
        final_boundary.subgrid_tke_change_j});
    report.committed_owner_sum_j = tmv::completed_step_owner_sum(report);
    report.closure_residual_j = report.committed_owner_sum_j - tmv::completed_step_reservoir_sum(report);
    report.complete = true;
    if (!tmv::structurally_valid_completed_step_energy_report(report))
        throw std::runtime_error("completed-step energy composition produced an invalid report");
}

static tmv::CompletedStepEnergyReportV1 compose_completed_step_energy(const SimulationStateView&      state,
                                                                      const TimeSteppingRuntimeState& time_runtime,
                                                                      const AdvectionRuntimeState&    advection_runtime)
{
    const auto& acoustic = time_runtime.last_acoustic_energy;
    const auto& slow = time_runtime.last_slow_momentum_energy;
    if (!acoustic.complete || !acoustic.boundary_complete || !acoustic.kinetic_transport_complete || !slow.complete ||
        slow.committed_pressure_energy_change_j !=
            slow.pressure_dry_internal_change_j + slow.pressure_water_sensible_change_j ||
        !advection_runtime.thermodynamic_moist_energy_reports_complete)
    {
        throw std::runtime_error("completed-step energy composition received a partial owner");
    }

    tmv::CompletedStepEnergyReportV1 report;
    report.acoustic_committed_change_j      = acoustic.committed_inventory_change_j;
    report.slow_dynamics_committed_change_j = slow.committed_kinetic_change_j + slow.committed_pressure_energy_change_j;
    for (const auto& scalar : advection_runtime.thermodynamic_moist_energy_reports)
    {
        if (!scalar.complete)
        {
            throw std::runtime_error("completed-step energy composition received a partial "
                                     "scalar owner");
        }
        report.scalar_transport_committed_change_j += scalar.committed_state_change_j;
        report.dry_internal_change_j += scalar.dry_internal_change_j;
        report.water_sensible_change_j += scalar.water_sensible_change_j;
        report.phase_reference_change_j += scalar.phase_reference_change_j;
        report.carrier_kinetic_change_j += scalar.carrier_kinetic_change_j;
        report.gravitational_change_j += scalar.gravitational_change_j;
        report.subgrid_tke_change_j += scalar.subgrid_tke_change_j;
    }
    const std::array<double, 6> integrator_prefix{
        acoustic.continuity_eos_dry_internal_change_j + slow.pressure_dry_internal_change_j,
        acoustic.continuity_eos_water_sensible_change_j +
                                      slow.pressure_water_sensible_change_j,
        acoustic.continuity_eos_phase_reference_change_j,
        acoustic.continuity_carrier_kinetic_change_j + acoustic.pressure_gradient_committed_work_j +
        acoustic.boundary_committed_work_j + slow.committed_kinetic_change_j,
        acoustic.continuity_gravitational_change_j,
        acoustic.continuity_subgrid_tke_change_j,
    };
    append_coupled_physics_energy(state, time_runtime, report, &integrator_prefix);
    return report;
}

static tmv::CompletedStepEnergyReportV1 compose_unsplit_rk_completed_step_energy(
    const SimulationStateView& state, const TimeSteppingRuntimeState& time_runtime,
    const tmv::CompletedStepEnergyReportV1& unsplit_core)
{
    if (!unsplit_core.complete || !tmv::structurally_valid_completed_step_energy_report(unsplit_core))
        throw std::runtime_error("unsplit-RK completed-step composition received a partial core owner");
    auto report = unsplit_core;
    append_coupled_physics_energy(state, time_runtime, report);
    return report;
}

// Advance one caller-selected dynamics interval.
void tmv::step_dynamics_new(double dt_dynamics, double current_time)
{
    return step_dynamics_new(global_simulation_state(), dt_dynamics, current_time);
}

void tmv::step_dynamics_new(const SimulationStateView& execution_state, double dt_dynamics, double current_time)
{
    const auto                 sim_state        = execution_state;
    auto& bc_scheme = tmv::boundary_scheme_for(sim_state);
    tmv::DynamicsRuntimeState& dynamics_runtime = dynamics_runtime_for(sim_state);
    if (!dynamics_runtime.scheme)
    {
        throw std::runtime_error("step_dynamics_new: no dynamics scheme initialized.");
    }
    dynamics_runtime.scheme->bind_execution_context(sim_state);
    const RuntimeConfigurationState* configuration = sim_state.configuration;
    const bool coriolis_enabled = configuration != nullptr ? configuration->coriolis_enabled : tmv::require_run_configuration(execution_state).coriolis_enabled;
    const auto coriolis_mode    = configuration != nullptr ? configuration->coriolis_mode : tmv::require_run_configuration(execution_state).coriolis_mode;
    const double coriolis_latitude_deg =
        configuration != nullptr ? configuration->coriolis_latitude_deg : tmv::require_run_configuration(execution_state).coriolis_latitude_deg;
    const auto& domain_frame = sim_state.domain_frame != nullptr ? *sim_state.domain_frame : tmv::require_level_runtime(execution_state.domain_frame, "domain frame");
    const CartesianCoriolisProfile coriolis_profile =
        make_cartesian_coriolis_profile(coriolis_enabled,
                                        coriolis_mode == tmv::init::config::CoriolisMode::BetaPlane,
                                        coriolis_latitude_deg,
                                        domain_frame,
                                        current_time);
    sim_state.coriolis_f = coriolis_profile.f_at_grid_y0;
    dynamics_runtime.scheme->set_coriolis_profile(coriolis_profile);
    const auto dynamics_total_t0 = dynamics_perf_start();

    prepare_boundary_layer_tendencies_for_application(sim_state);
    const bool mutating_repairs = tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Sanitize ||
                                  tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Off;
    set_primary_state_repair_policy(mutating_repairs, tmv::require_run_configuration(execution_state).validation_policy.mode == tmv::GuardMode::Strict);
    ensure_dynamics_tendency_buffers(dynamics_runtime, sim_state.grid);
    Field3D& du_dt   = dynamics_runtime.du_dt;
    Field3D& dv_dt   = dynamics_runtime.dv_dt;
    Field3D& dw_dt   = dynamics_runtime.dw_dt;
    Field3D& drho_dt = dynamics_runtime.drho_dt;
    Field3D& dp_dt   = dynamics_runtime.dp_dt;

    // Named-slot residency step boundary (GPU audit Phase 4.3d):
    // device-resident field content committed by last step's dispatches
    // expires here; within-step reuse windows open.
    begin_resident_step_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"));
    // GPU audit Phase 3.6: the per-stage budget reductions and the
    // Scientific budget reductions and derived output fields have independent
    // per-step gates. A derived consumer must not implicitly force the much
    // larger budget suite, and science-off must not leave a published derived
    // field stale.
    const bool step_science_diag          = step_diagnostics_enabled();
    const bool step_radar_diag            = step_radar_diagnostics_enabled();
    const bool step_vorticity_diag        = step_vorticity_diagnostics_enabled();
    const bool step_angular_momentum_diag = step_angular_momentum_diagnostics_enabled();
    const bool step_pressure_diag         = step_pressure_diagnostics_enabled();
    // Every normal CPU writer clamps the prognostic values it authors. Keep
    // the independent ten-field sweep as a strict/periodic verification mode
    // instead of paying its memory-bandwidth cost after every stage.
    const bool verify_primary_state =
        primary_state_verification_due(tmv::require_run_configuration(execution_state).validation_policy, step_science_diag, log_debug_enabled());
    const auto verify_primary_bounds = [&](const char* stage)
    { return verify_primary_state ? enforce_primary_state_bounds(sim_state, stage) : 0; };
#ifdef NIMBUS_SEPARATE_ENDPOINT_MOISTURE_SCANS
    const ConservationBudget budget_start = measured_conservation_budget(sim_state, step_science_diag);
#else
    const ConservationMoistureSnapshots inventory_start =
        measured_conservation_moisture_snapshots(sim_state, step_science_diag);
    const ConservationBudget& budget_start = inventory_start.conservation;
#endif
    // F.5b momentum/energy/moisture inventories at the start of the step; the
    // matching total_step closure lines are emitted at the end (see below).
    const MomentumBudget             momentum_start = measured_momentum_budget(sim_state, step_science_diag);
    const EnergyDiagnosticSnapshotV1 energy_start   = measured_energy_snapshot(sim_state, step_science_diag);
#ifdef NIMBUS_SEPARATE_ENDPOINT_MOISTURE_SCANS
    const MoistureBudget moisture_start = measured_moisture_budget(sim_state, step_science_diag);
#else
    const MoistureBudget& moisture_start = inventory_start.moisture;
#endif

    // initialize_dynamics() validates that split_acoustic implies the scheme
    // implements SplitExplicitDynamics, so when split_acoustic=true the cast
    // is guaranteed non-null. The cast is still needed to obtain the typed
    // pointer used by the split path below.
    auto*                     split_dynamics = dynamic_cast<SplitExplicitDynamics*>(dynamics_runtime.scheme.get());
    TimeSteppingRuntimeState& time_runtime =
        sim_state.time_stepping_runtime ? *sim_state.time_stepping_runtime : tmv::require_level_runtime(execution_state.time_stepping_runtime, "time_stepping_runtime");
    const bool use_split = time_runtime.config.split_acoustic && split_dynamics != nullptr;
    if (split_dynamics != nullptr)
    {
        split_dynamics->set_momentum_interface_flux_register(
            dynamics_runtime.momentum_interface_flux_capture_enabled
                ? &dynamics_runtime.momentum_interface_flux_register
                : nullptr);
        split_dynamics->set_momentum_interface_flux_slow_weight(1.0);
    }
    const bool use_coupled_unsplit_rk =
        !use_split && time_runtime.scheme != nullptr &&
        time_runtime.scheme->supports_coupled_unsplit_rk();
    // A row from a non-Cartesian or unsplit step must not repeat the previous
    // Cartesian boundary attribution. Cumulative state remains intact; only
    // the per-step report is cleared until a retained context commits below.
    const bool native_mapped_terrain = tmv::require_run_configuration(execution_state).admission.native_mapped_terrain;
    if (!native_mapped_terrain)
    {
        time_runtime.last_boundary_mass         = {};
        time_runtime.last_acoustic_energy       = {};
        time_runtime.last_slow_momentum_energy  = {};
        time_runtime.last_eos_pressure_energy   = {};
        time_runtime.last_completed_step_energy = {};
    }

    if (native_mapped_terrain)
    {
        const ComputeBackendKind mapped_backend = active_compute_backend_kind(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"));
        if (!use_split ||
            (mapped_backend != ComputeBackendKind::Vulkan &&
             mapped_backend != ComputeBackendKind::Metal) ||
            compute_backend_fallback_active(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")))
        {
            throw std::runtime_error(
                "exact native mapped terrain lost its fallback-off device owner");
        }
        const auto native_core_t0 = dynamics_perf_start();
        struct PreparedRuntime
        {
            tmv::CartesianCGridTransportContext transport;
            DynamicsBoundaryMassReport boundary_mass;
            ThermodynamicBoundaryTransportReports thermodynamic_boundary;
            ThermodynamicBoundaryTransportReports cumulative_thermodynamic_boundary;
            ThermodynamicMoistEnergyTransportReports thermodynamic_moist_energy;
            ThermodynamicMoistEnergyTransportReports cumulative_thermodynamic_moist_energy;
            CartesianCGridAcousticEnergyReportV1 cumulative_acoustic;
            CartesianSlowMomentumEnergyReportV1 cumulative_slow;
            tmv::dynamics_eos::PressureProjectionEnergyReportV1 cumulative_eos;
            tmv::SpongeEnergyReportV1 sponge;
            tmv::SpongeEnergyReportV1 cumulative_sponge;
            tmv::FinalBoundaryEnergyReportV1 cumulative_boundary;
            MicrophysicsThermodynamicRepairReport thermodynamic_sources;
            tmv::CompletedStepEnergyReportV1 completed_step;
            tmv::CompletedStepEnergyReportV1 cumulative_completed_step;
            std::uint64_t thermodynamic_boundary_step_count = 0;
            std::uint64_t thermodynamic_moist_energy_step_count = 0;
            std::uint64_t sponge_step_count = 0;
            std::uint64_t completed_step_count = 0;
            double cumulative_outward_boundary_mass_kg = 0.0;
            double cumulative_flux_implied_state_change_kg = 0.0;
            double cumulative_committed_state_change_kg = 0.0;
            double cumulative_boundary_mass_closure_residual_kg = 0.0;
            bool complete = false;
        } prepared;
        struct ObserverContext
        {
            const SimulationStateView* state = nullptr;
            TimeSteppingRuntimeState* time = nullptr;
            AdvectionRuntimeState* advection = nullptr;
            PreparedRuntime* prepared = nullptr;
            double dt = 0.0;
            int substeps = 0;
        } observer_context{
            &sim_state, &time_runtime,
            sim_state.advection_runtime ? sim_state.advection_runtime
                                        : &tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime"),
            &prepared,
                           dt_dynamics, time_runtime.config.n_acoustic_substeps};
        tmv::terrain_native::MappedCoreStepReport native_report;
        if (!tmv::terrain_native::advance_dry_stationary_mapped_core(
                sim_state, dt_dynamics,
                time_runtime.config.n_acoustic_substeps, native_report,
                +[](void* raw, const Field3D&, const Field3D&, const Field3D&,
                    const Field3D& rho, const Field3D&, const Field3D&,
                    const Field3D&, const Field3D&, const Field3D&,
                    const Field3D& density_begin, const Field3D& mass_flux_x,
                    const Field3D& mass_flux_y, const Field3D& mass_flux_z,
                    const tmv::terrain_native::MappedCoreStepReport& report)
                {
                    auto& context = *static_cast<ObserverContext*>(raw);
                    const auto& execution_state = *context.state;
                    PreparedRuntime next;
                    const auto* metrics = tmv::physical_grid_metrics::active_terrain_metrics(*context.state);
                    next.transport.begin(execution_state, context.state->grid.nr,
                                         context.state->grid.nth,
                                         context.state->grid.nz,
                                         context.dt,
                                         metrics,
                                         0.0,
                                         0.0);
                    next.transport.capture_density_begin(density_begin);
                    next.transport.mass_flux_x = mass_flux_x;
                    next.transport.mass_flux_y = mass_flux_y;
                    next.transport.mass_flux_z = mass_flux_z;
                    const double dt_small = static_cast<double>(
                        static_cast<float>(context.dt / context.substeps));
                    for (int substep = 0; substep < context.substeps; ++substep)
                    {
                        next.transport.record_external_carrier_kinetic_transport(0.0);
                        next.transport.record_external_accumulation(dt_small, 1, true);
                    }
                    next.transport.complete(execution_state);
                    next.transport.require_valid(execution_state, context.state->grid.nr,
                                                 context.state->grid.nth,
                                                 context.state->grid.nz,
                                                 context.dt);
                    next.boundary_mass = next.transport.boundary_mass_report(execution_state,
                        rho, context.state->grid.dr, context.state->grid.dtheta,
                        context.state->grid.dz);

                    auto& time = *context.time;
                    if (time.acoustic_energy_step_count == std::numeric_limits<std::uint64_t>::max() ||
                        time.slow_momentum_energy_step_count == std::numeric_limits<std::uint64_t>::max() ||
                        time.eos_pressure_energy_step_count == std::numeric_limits<std::uint64_t>::max() ||
                        time.sponge_energy_step_count == std::numeric_limits<std::uint64_t>::max() ||
                        time.final_boundary_energy_step_count == std::numeric_limits<std::uint64_t>::max() ||
                        time.boundary_mass_step_count == std::numeric_limits<std::uint64_t>::max() ||
                        time.completed_step_energy_count == std::numeric_limits<std::uint64_t>::max() ||
                        context.advection->thermodynamic_boundary_report_step_count ==
                            std::numeric_limits<std::uint64_t>::max() ||
                        context.advection->thermodynamic_moist_energy_report_step_count ==
                            std::numeric_limits<std::uint64_t>::max())
                        throw std::overflow_error("native mapped terrain runtime counter overflowed");

                    next.thermodynamic_boundary = report.thermodynamic_boundary;
                    next.thermodynamic_moist_energy = report.thermodynamic_moist_energy;
                    next.cumulative_thermodynamic_boundary =
                        context.advection->cumulative_thermodynamic_boundary_reports;
                    next.cumulative_thermodynamic_moist_energy =
                        context.advection->cumulative_thermodynamic_moist_energy_reports;
                    for (std::size_t scalar = 0;
                         scalar < thermodynamic_boundary_report_count; ++scalar)
                    {
                        const auto& boundary_step = next.thermodynamic_boundary[scalar];
                        const auto& energy_step = next.thermodynamic_moist_energy[scalar];
                        if (!boundary_step.complete || !energy_step.complete)
                            throw std::runtime_error(
                                "native mapped terrain scalar attribution was incomplete");
                        auto& boundary_cumulative =
                            next.cumulative_thermodynamic_boundary[scalar];
                        boundary_cumulative.outward_boundary_inventory +=
                            boundary_step.outward_boundary_inventory;
                        boundary_cumulative.flux_implied_state_change +=
                            boundary_step.flux_implied_state_change;
                        boundary_cumulative.committed_state_change +=
                            boundary_step.committed_state_change;
                        boundary_cumulative.closure_residual +=
                            boundary_step.closure_residual;
                        boundary_cumulative.complete = true;

                        auto& energy_cumulative =
                            next.cumulative_thermodynamic_moist_energy[scalar];
                        energy_cumulative.dry_internal_change_j +=
                            energy_step.dry_internal_change_j;
                        energy_cumulative.water_sensible_change_j +=
                            energy_step.water_sensible_change_j;
                        energy_cumulative.phase_reference_change_j +=
                            energy_step.phase_reference_change_j;
                        energy_cumulative.carrier_kinetic_change_j +=
                            energy_step.carrier_kinetic_change_j;
                        energy_cumulative.gravitational_change_j +=
                            energy_step.gravitational_change_j;
                        energy_cumulative.subgrid_tke_change_j +=
                            energy_step.subgrid_tke_change_j;
                        energy_cumulative.density_history_remap_j +=
                            energy_step.density_history_remap_j;
                        energy_cumulative.outward_boundary_energy_j +=
                            energy_step.outward_boundary_energy_j;
                        energy_cumulative.limited_face_state_change_j +=
                            energy_step.limited_face_state_change_j;
                        energy_cumulative.storage_rounding_j +=
                            energy_step.storage_rounding_j;
                        energy_cumulative.interior_conversion_j =
                            energy_cumulative.limited_face_state_change_j +
                            energy_cumulative.outward_boundary_energy_j;
                        energy_cumulative.requested_state_change_j =
                            energy_cumulative.density_history_remap_j +
                            energy_cumulative.limited_face_state_change_j;
                        energy_cumulative.committed_state_change_j =
                            energy_cumulative.dry_internal_change_j +
                            energy_cumulative.water_sensible_change_j +
                            energy_cumulative.phase_reference_change_j +
                            energy_cumulative.carrier_kinetic_change_j +
                            energy_cumulative.gravitational_change_j +
                            energy_cumulative.subgrid_tke_change_j;
                        energy_cumulative.closure_residual_j =
                            energy_cumulative.committed_state_change_j -
                            (energy_cumulative.requested_state_change_j +
                             energy_cumulative.storage_rounding_j);
                        energy_cumulative.complete = true;
                        const double cumulative_values[] = {
                            boundary_cumulative.outward_boundary_inventory,
                            boundary_cumulative.flux_implied_state_change,
                            boundary_cumulative.committed_state_change,
                            boundary_cumulative.closure_residual,
                            energy_cumulative.dry_internal_change_j,
                            energy_cumulative.water_sensible_change_j,
                            energy_cumulative.phase_reference_change_j,
                            energy_cumulative.carrier_kinetic_change_j,
                            energy_cumulative.gravitational_change_j,
                            energy_cumulative.subgrid_tke_change_j,
                            energy_cumulative.density_history_remap_j,
                            energy_cumulative.outward_boundary_energy_j,
                            energy_cumulative.interior_conversion_j,
                            energy_cumulative.limited_face_state_change_j,
                            energy_cumulative.requested_state_change_j,
                            energy_cumulative.storage_rounding_j,
                            energy_cumulative.committed_state_change_j,
                            energy_cumulative.closure_residual_j,
                        };
                        if (std::any_of(
                                std::begin(cumulative_values),
                                std::end(cumulative_values),
                                [](double value) { return !std::isfinite(value); }))
                            throw std::overflow_error(
                                "native mapped terrain cumulative scalar attribution overflowed");
                    }
                    next.thermodynamic_boundary_step_count =
                        context.advection->thermodynamic_boundary_report_step_count + 1;
                    next.thermodynamic_moist_energy_step_count =
                        context.advection->thermodynamic_moist_energy_report_step_count + 1;

                    next.cumulative_slow = time.cumulative_slow_momentum_energy;
                    const auto& slow = report.slow_momentum;
                    next.cumulative_slow.advection_requested_linear_work_j += slow.advection_requested_linear_work_j;
                    next.cumulative_slow.nonadvection_requested_linear_work_j += slow.nonadvection_requested_linear_work_j;
                    next.cumulative_slow.advection_finite_step_correction_j += slow.advection_finite_step_correction_j;
                    next.cumulative_slow.nonadvection_finite_step_correction_j += slow.nonadvection_finite_step_correction_j;
                    next.cumulative_slow.coupled_finite_step_cross_work_j += slow.coupled_finite_step_cross_work_j;
                    next.cumulative_slow.storage_rounding_j += slow.storage_rounding_j;
                    next.cumulative_slow.committed_kinetic_change_j += slow.committed_kinetic_change_j;
                    next.cumulative_slow.pressure_dry_internal_change_j += slow.pressure_dry_internal_change_j;
                    next.cumulative_slow.pressure_water_sensible_change_j += slow.pressure_water_sensible_change_j;
                    next.cumulative_slow.committed_pressure_energy_change_j =
                        next.cumulative_slow.pressure_dry_internal_change_j +
                        next.cumulative_slow.pressure_water_sensible_change_j;
                    const double slow_requested =
                        next.cumulative_slow.advection_requested_linear_work_j +
                        next.cumulative_slow.nonadvection_requested_linear_work_j +
                        next.cumulative_slow.advection_finite_step_correction_j +
                        next.cumulative_slow.nonadvection_finite_step_correction_j +
                        next.cumulative_slow.coupled_finite_step_cross_work_j;
                    next.cumulative_slow.closure_residual_j =
                        next.cumulative_slow.committed_kinetic_change_j -
                        (slow_requested + next.cumulative_slow.storage_rounding_j);
                    next.cumulative_slow.complete = true;

                    next.cumulative_acoustic = time.cumulative_acoustic_energy;
                    auto& acoustic = next.cumulative_acoustic;
                    const auto& step = report.acoustic;
                    acoustic.continuity_eos_dry_internal_change_j += step.continuity_eos_dry_internal_change_j;
                    acoustic.continuity_eos_water_sensible_change_j += step.continuity_eos_water_sensible_change_j;
                    acoustic.continuity_eos_phase_reference_change_j += step.continuity_eos_phase_reference_change_j;
                    acoustic.continuity_carrier_kinetic_change_j += step.continuity_carrier_kinetic_change_j;
                    acoustic.continuity_gravitational_change_j += step.continuity_gravitational_change_j;
                    acoustic.continuity_subgrid_tke_change_j += step.continuity_subgrid_tke_change_j;
                    acoustic.continuity_carrier_requested_change_j += step.continuity_carrier_requested_change_j;
                    acoustic.continuity_carrier_outward_boundary_transport_j += step.continuity_carrier_outward_boundary_transport_j;
                    acoustic.continuity_carrier_interior_conversion_j += step.continuity_carrier_interior_conversion_j;
                    acoustic.continuity_carrier_storage_rounding_j += step.continuity_carrier_storage_rounding_j;
                    acoustic.slow_momentum_advection_requested_linear_work_j += step.slow_momentum_advection_requested_linear_work_j;
                    acoustic.kinetic_transport_interior_conversion_j += step.kinetic_transport_interior_conversion_j;
                    acoustic.pressure_gradient_requested_linear_work_j += step.pressure_gradient_requested_linear_work_j;
                    acoustic.pressure_gradient_finite_step_correction_j += step.pressure_gradient_finite_step_correction_j;
                    acoustic.pressure_gradient_storage_rounding_j += step.pressure_gradient_storage_rounding_j;
                    acoustic.pressure_gradient_committed_work_j += step.pressure_gradient_committed_work_j;
                    acoustic.boundary_committed_work_j += step.boundary_committed_work_j;
                    acoustic.boundary_radiative_requested_linear_work_j += step.boundary_radiative_requested_linear_work_j;
                    acoustic.boundary_radiative_finite_step_correction_j += step.boundary_radiative_finite_step_correction_j;
                    acoustic.boundary_radiative_storage_rounding_j += step.boundary_radiative_storage_rounding_j;
                    acoustic.boundary_radiative_committed_work_j += step.boundary_radiative_committed_work_j;
                    acoustic.boundary_constraint_history_committed_work_j += step.boundary_constraint_history_committed_work_j;
                    acoustic.boundary_application_count += step.boundary_application_count;
                    acoustic.substep_count += step.substep_count;
                    acoustic.pressure_gradient_storage_rounding_j =
                        acoustic.pressure_gradient_committed_work_j -
                        acoustic.pressure_gradient_requested_linear_work_j -
                        acoustic.pressure_gradient_finite_step_correction_j;
                    acoustic.boundary_radiative_storage_rounding_j =
                        acoustic.boundary_radiative_committed_work_j -
                        acoustic.boundary_radiative_requested_linear_work_j -
                        acoustic.boundary_radiative_finite_step_correction_j;
                    acoustic.kinetic_transport_closure_residual_j = 0.0;
                    acoustic.kinetic_transport_complete = true;
                    acoustic.boundary_complete = true;
                    acoustic.committed_inventory_change_j = canonical_acoustic_committed_inventory(acoustic);
                    acoustic.complete = true;

                    next.cumulative_eos = time.cumulative_eos_pressure_energy;
                    tmv::dynamics_eos::accumulate_pressure_projection_energy(
                        next.cumulative_eos, report.final_eos);
                    next.sponge.complete = true;
                    next.sponge.upper.complete = true;
                    next.sponge.w_only.complete = true;
                    next.sponge.lateral.complete = true;
                    next.cumulative_sponge = time.cumulative_sponge_energy;
                    tmv::accumulate_sponge_energy(
                        next.cumulative_sponge, next.sponge);
                    next.sponge_step_count = time.sponge_energy_step_count + 1;
                    next.cumulative_boundary = time.cumulative_final_boundary_energy;
                    tmv::accumulate_final_boundary_energy(
                        next.cumulative_boundary, report.final_boundary);
                    if (!tmv::valid_final_boundary_energy_report(next.cumulative_boundary) ||
                        !next.cumulative_boundary.complete)
                        throw std::overflow_error("native mapped terrain cumulative boundary attribution overflowed");

                    next.cumulative_outward_boundary_mass_kg =
                        time.cumulative_outward_boundary_mass_kg + next.boundary_mass.outward_boundary_mass_kg;
                    next.cumulative_flux_implied_state_change_kg =
                        time.cumulative_flux_implied_state_change_kg + next.boundary_mass.flux_implied_state_change_kg;
                    next.cumulative_committed_state_change_kg =
                        time.cumulative_committed_state_change_kg + next.boundary_mass.committed_state_change_kg;
                    next.cumulative_boundary_mass_closure_residual_kg =
                        time.cumulative_boundary_mass_closure_residual_kg + next.boundary_mass.closure_residual_kg;
                    const double finite_values[] = {
                        next.cumulative_slow.closure_residual_j,
                        next.cumulative_acoustic.committed_inventory_change_j,
                        next.cumulative_eos.committed_energy_change_j,
                        next.cumulative_boundary.committed_energy_change_j,
                        next.cumulative_outward_boundary_mass_kg,
                        next.cumulative_flux_implied_state_change_kg,
                        next.cumulative_committed_state_change_kg,
                        next.cumulative_boundary_mass_closure_residual_kg,
                    };
                    if (std::any_of(std::begin(finite_values), std::end(finite_values),
                                    [](double value) { return !std::isfinite(value); }))
                        throw std::overflow_error("native mapped terrain cumulative attribution overflowed");

                    AdvectionRuntimeState advection_for_composition;
                    advection_for_composition.thermodynamic_boundary_reports =
                        next.thermodynamic_boundary;
                    advection_for_composition.thermodynamic_moist_energy_reports =
                        next.thermodynamic_moist_energy;
                    advection_for_composition.thermodynamic_boundary_reports_complete = true;
                    advection_for_composition.thermodynamic_moist_energy_reports_complete = true;
                    TimeSteppingRuntimeState time_for_composition;
                    time_for_composition.last_acoustic_energy = report.acoustic;
                    time_for_composition.last_slow_momentum_energy = report.slow_momentum;
                    time_for_composition.last_eos_pressure_energy = report.final_eos;
                    time_for_composition.last_final_boundary_energy = report.final_boundary;
                    time_for_composition.last_sponge_energy = next.sponge;
                    SimulationStateView state_for_composition = *context.state;
                    MicrophysicsRuntimeState inactive_microphysics;
                    next.thermodynamic_sources.moist_energy_complete = true;
                    inactive_microphysics.last_thermodynamic_repair_report =
                        next.thermodynamic_sources;
                    state_for_composition.microphysics_runtime = &inactive_microphysics;
                    next.completed_step = compose_completed_step_energy(
                        state_for_composition, time_for_composition,
                        advection_for_composition);
                    next.cumulative_completed_step =
                        time.cumulative_completed_step_energy;
                    tmv::accumulate_completed_step_energy(
                        next.cumulative_completed_step, next.completed_step);
                    if (!tmv::structurally_valid_completed_step_energy_report(
                            next.cumulative_completed_step) ||
                        !next.cumulative_completed_step.complete)
                        throw std::overflow_error(
                            "native mapped terrain cumulative completed-step attribution overflowed");
                    next.completed_step_count = time.completed_step_energy_count + 1;
                    next.complete = true;
                    *context.prepared = std::move(next);
                },
                &observer_context) ||
            !native_report.complete ||
            !native_report.final_pressure_uses_canonical_fp64 ||
            !prepared.complete)
        {
            throw std::runtime_error(
                "exact native mapped terrain transaction failed closed");
        }
        dynamics_perf_finish(dynamics_perf_totals.core_s, native_core_t0);

        dynamics_runtime.transport_context = std::move(prepared.transport);
        time_runtime.last_boundary_mass = prepared.boundary_mass;
        time_runtime.cumulative_outward_boundary_mass_kg = prepared.cumulative_outward_boundary_mass_kg;
        time_runtime.cumulative_flux_implied_state_change_kg = prepared.cumulative_flux_implied_state_change_kg;
        time_runtime.cumulative_committed_state_change_kg = prepared.cumulative_committed_state_change_kg;
        time_runtime.cumulative_boundary_mass_closure_residual_kg =
            prepared.cumulative_boundary_mass_closure_residual_kg;
        ++time_runtime.boundary_mass_step_count;

        time_runtime.last_acoustic_energy = native_report.acoustic;
        time_runtime.cumulative_acoustic_energy = prepared.cumulative_acoustic;
        ++time_runtime.acoustic_energy_step_count;
        time_runtime.last_slow_momentum_energy = native_report.slow_momentum;
        time_runtime.cumulative_slow_momentum_energy = prepared.cumulative_slow;
        ++time_runtime.slow_momentum_energy_step_count;
        time_runtime.last_eos_pressure_energy = native_report.final_eos;
        time_runtime.cumulative_eos_pressure_energy = prepared.cumulative_eos;
        ++time_runtime.eos_pressure_energy_step_count;
        time_runtime.last_sponge_energy = prepared.sponge;
        time_runtime.cumulative_sponge_energy = prepared.cumulative_sponge;
        time_runtime.sponge_energy_step_count = prepared.sponge_step_count;
        time_runtime.last_final_boundary_energy =
            native_report.final_boundary;
        time_runtime.cumulative_final_boundary_energy =
            prepared.cumulative_boundary;
        ++time_runtime.final_boundary_energy_step_count;

        time_runtime.last_completed_step_energy = prepared.completed_step;
        time_runtime.cumulative_completed_step_energy =
            prepared.cumulative_completed_step;
        time_runtime.completed_step_energy_count = prepared.completed_step_count;

        AdvectionRuntimeState& advection_runtime =
            sim_state.advection_runtime ? *sim_state.advection_runtime
                                        : tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime");
        advection_runtime.thermodynamic_boundary_reports =
            prepared.thermodynamic_boundary;
        advection_runtime.cumulative_thermodynamic_boundary_reports =
            prepared.cumulative_thermodynamic_boundary;
        advection_runtime.thermodynamic_moist_energy_reports =
            prepared.thermodynamic_moist_energy;
        advection_runtime.cumulative_thermodynamic_moist_energy_reports =
            prepared.cumulative_thermodynamic_moist_energy;
        advection_runtime.thermodynamic_boundary_reports_requested =
            step_science_diag;
        advection_runtime.thermodynamic_boundary_reports_complete = true;
        advection_runtime.thermodynamic_moist_energy_reports_requested =
            step_science_diag;
        advection_runtime.thermodynamic_moist_energy_reports_complete = true;
        advection_runtime.thermodynamic_boundary_report_step_count =
            prepared.thermodynamic_boundary_step_count;
        advection_runtime.thermodynamic_moist_energy_report_step_count =
            prepared.thermodynamic_moist_energy_step_count;
        MicrophysicsRuntimeState& microphysics_runtime =
            sim_state.microphysics_runtime ? *sim_state.microphysics_runtime
                                           : tmv::require_level_runtime(execution_state.microphysics_runtime, "microphysics_runtime");
        microphysics_runtime.last_thermodynamic_repair_report =
            prepared.thermodynamic_sources;

        verify_primary_bounds("native_mapped_terrain");
        invalidate_all_resident_fields_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"));
        write_thermodynamic_source_budget_row(
            sim_state, current_time, dt_dynamics);
        if (step_science_diag)
        {
            const ConservationBudget budget_final =
                measured_conservation_budget(sim_state, true);
            report_budget_transition(
                sim_state, "total_step", budget_start, budget_final,
                dt_dynamics);
            report_momentum_transition(
                sim_state, "total_step", momentum_start,
                measured_momentum_budget(sim_state, true), dt_dynamics);
            report_energy_transition(
                sim_state, "total_step", energy_start,
                measured_energy_snapshot(sim_state, true), dt_dynamics);
            report_moisture_transition(
                sim_state, "total_step", moisture_start,
                compute_moisture_budget(sim_state), dt_dynamics);
        }
        tmv::log_info(
            "[TERRAIN_NATIVE] backend=",
            compute_backend_kind_name(mapped_backend),
            ", fallback=off, "
            "spatial_owner=device, final_eos_owner=host_fp64, "
            "complete=1, acoustic_substeps=",
            native_report.acoustic_substeps,
            ", backend_dispatches=",
            native_report.completed_backend_dispatches,
            ", mass=", native_report.mass_continuity_dispatches,
            ", slow=", native_report.slow_tendency_dispatches,
            ", slow_average=", native_report.slow_average_dispatches,
            ", fast=", native_report.fast_momentum_dispatches,
            ", state=", native_report.state_update_dispatches,
            ", flux_accum=", native_report.flux_accumulation_dispatches,
            ", scalar_flux=", native_report.scalar_flux_dispatches,
            ", scalar_limiter=", native_report.scalar_limiter_dispatches,
            ", scalar_update=", native_report.scalar_update_dispatches,
            ", boundary=", native_report.boundary_dispatches);
        dynamics_perf_finish(
            dynamics_perf_totals.total_s, dynamics_total_t0);
        return;
    }

    auto&                                transport_context = dynamics_runtime.transport_context;
    tmv::CartesianCGridTransportContext* coupled_transport = nullptr;
    if (use_split && sim_state.grid.geometry.coord == CoordinateSystem::Cartesian)
    {
        const TerrainMetrics3D* terrain_metrics = tmv::physical_grid_metrics::active_terrain_metrics(sim_state);
        transport_context.begin(execution_state, sim_state.grid.nr,
                                sim_state.grid.nth,
                                sim_state.grid.nz,
                                dt_dynamics,
                                terrain_metrics,
                                tmv::require_level_runtime(execution_state.domain_frame, "domain frame").velocity_ms.u_ms,
                                tmv::require_level_runtime(execution_state.domain_frame, "domain frame").velocity_ms.v_ms);
        coupled_transport = &transport_context;
    }
    CartesianCGridAcousticEnergyReportV1  acoustic_energy_report;
    CartesianCGridAcousticEnergyReportV1* attributed_acoustic_energy =
        step_science_diag && use_split
            && sim_state.grid.geometry.coord == CoordinateSystem::Cartesian
            ? &acoustic_energy_report
            : nullptr;
    CartesianSlowMomentumEnergyReportV1  slow_momentum_energy_report;
    CartesianSlowMomentumEnergyReportV1* attributed_slow_momentum_energy =
        step_science_diag && use_split
            && sim_state.grid.geometry.coord == CoordinateSystem::Cartesian
            ? &slow_momentum_energy_report
            : nullptr;
    SlowMomentumAdvectionTendencyOutputs  advection_outputs;
    SlowMomentumAdvectionTendencyOutputs* attributed_advection_outputs = nullptr;
    if (attributed_slow_momentum_energy != nullptr)
    {
        ensure_slow_advection_tendency_buffers(dynamics_runtime, sim_state.grid);
        advection_outputs = {
            &dynamics_runtime.du_advection_dt,
            &dynamics_runtime.dv_advection_dt,
            &dynamics_runtime.dw_advection_dt,
        };
        attributed_advection_outputs = &advection_outputs;
    }

#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent dynamics_core_allocations(
        tmv::allocation_telemetry::Component::DynamicsCore);
#endif
    const auto dynamics_core_t0 = dynamics_perf_start();
    if (!use_split)
    {
        step_unsplit_dynamics(
            sim_state, dt_dynamics, current_time,
            du_dt, dv_dt, dw_dt, drho_dt, dp_dt);
    }
    else
    {
        // === SPLIT-EXPLICIT PATH ===
        // Delegate to the time stepping scheme in src/numerics/time_stepping/.
        step_dynamics_split_explicit(sim_state,
                                     dt_dynamics,
                                     current_time,
                                     *split_dynamics,
                                     du_dt,
                                     dv_dt,
                                     dw_dt,
                                     drho_dt,
                                     dp_dt,
                                     coupled_transport,
                                     attributed_acoustic_energy,
                                     attributed_advection_outputs,
                                     attributed_slow_momentum_energy);
    }
    dynamics_perf_finish(dynamics_perf_totals.core_s, dynamics_core_t0);
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    dynamics_core_allocations.restore();
#endif
    if (step_science_diag && use_coupled_unsplit_rk)
    {
        const auto& before = energy_start.moist_mixture_v2;
        const auto after =
            compute_moist_mixture_energy_v2_deterministic(sim_state);
        tmv::CompletedStepEnergyReportV1 core;
        core.dry_internal_change_j = after.dry_internal - before.dry_internal;
        core.water_sensible_change_j =
            after.water_sensible - before.water_sensible;
        core.phase_reference_change_j =
            after.phase_reference - before.phase_reference;
        core.carrier_kinetic_change_j =
            after.resolved_kinetic - before.resolved_kinetic;
        core.gravitational_change_j =
            after.gravitational - before.gravitational;
        core.subgrid_tke_change_j = after.subgrid_tke - before.subgrid_tke;
        core.unsplit_rk_committed_change_j =
            tmv::completed_step_reservoir_sum(core);
        core.committed_owner_sum_j = tmv::completed_step_owner_sum(core);
        core.closure_residual_j = core.committed_owner_sum_j -
            tmv::completed_step_reservoir_sum(core);
        core.complete = true;
        if (!tmv::structurally_valid_completed_step_energy_report(core))
            throw std::runtime_error(
                "unsplit-RK core commit produced an invalid energy report");
        time_runtime.last_completed_step_energy = core;
    }
    if (attributed_slow_momentum_energy != nullptr)
    {
        if (!slow_momentum_energy_report.complete)
            throw std::runtime_error("requested Cartesian slow momentum energy attribution was not completed");
        const double report_values[] = {
            slow_momentum_energy_report.advection_requested_linear_work_j,
            slow_momentum_energy_report.nonadvection_requested_linear_work_j,
            slow_momentum_energy_report.advection_finite_step_correction_j,
            slow_momentum_energy_report.nonadvection_finite_step_correction_j,
            slow_momentum_energy_report.coupled_finite_step_cross_work_j,
            slow_momentum_energy_report.storage_rounding_j,
            slow_momentum_energy_report.committed_kinetic_change_j,
            slow_momentum_energy_report.closure_residual_j,
            slow_momentum_energy_report.pressure_dry_internal_change_j,
            slow_momentum_energy_report.pressure_water_sensible_change_j,
            slow_momentum_energy_report.committed_pressure_energy_change_j,
        };
        if (std::any_of(std::begin(report_values),
                        std::end(report_values),
                        [](double value) { return !std::isfinite(value); }) ||
            slow_momentum_energy_report.advection_finite_step_correction_j < 0.0 ||
            slow_momentum_energy_report.nonadvection_finite_step_correction_j < 0.0 ||
            slow_momentum_energy_report.committed_pressure_energy_change_j !=
                slow_momentum_energy_report.pressure_dry_internal_change_j +
                    slow_momentum_energy_report.pressure_water_sensible_change_j)
        {
            throw std::runtime_error("Cartesian slow momentum energy attribution produced an invalid report");
        }

        CartesianSlowMomentumEnergyReportV1 cumulative = time_runtime.cumulative_slow_momentum_energy;
        cumulative.advection_requested_linear_work_j += slow_momentum_energy_report.advection_requested_linear_work_j;
        cumulative.nonadvection_requested_linear_work_j +=
            slow_momentum_energy_report.nonadvection_requested_linear_work_j;
        cumulative.advection_finite_step_correction_j += slow_momentum_energy_report.advection_finite_step_correction_j;
        cumulative.nonadvection_finite_step_correction_j +=
            slow_momentum_energy_report.nonadvection_finite_step_correction_j;
        cumulative.coupled_finite_step_cross_work_j += slow_momentum_energy_report.coupled_finite_step_cross_work_j;
        cumulative.storage_rounding_j += slow_momentum_energy_report.storage_rounding_j;
        cumulative.committed_kinetic_change_j += slow_momentum_energy_report.committed_kinetic_change_j;
        cumulative.pressure_dry_internal_change_j += slow_momentum_energy_report.pressure_dry_internal_change_j;
        cumulative.pressure_water_sensible_change_j += slow_momentum_energy_report.pressure_water_sensible_change_j;
        cumulative.committed_pressure_energy_change_j =
            cumulative.pressure_dry_internal_change_j + cumulative.pressure_water_sensible_change_j;
        const double cumulative_requested =
            cumulative.advection_requested_linear_work_j + cumulative.nonadvection_requested_linear_work_j +
            cumulative.advection_finite_step_correction_j + cumulative.nonadvection_finite_step_correction_j +
            cumulative.coupled_finite_step_cross_work_j;
        cumulative.closure_residual_j =
            cumulative.committed_kinetic_change_j - (cumulative_requested + cumulative.storage_rounding_j);
        cumulative.complete                          = true;
        time_runtime.last_slow_momentum_energy       = slow_momentum_energy_report;
        time_runtime.cumulative_slow_momentum_energy = cumulative;
        ++time_runtime.slow_momentum_energy_step_count;
    }
    if (coupled_transport != nullptr)
    {
        // Seal the operator-time face history before kinetic composition or
        // scalar transport can consume it.
        coupled_transport->complete(execution_state);
        coupled_transport->require_valid(execution_state, sim_state.grid.nr, sim_state.grid.nth, sim_state.grid.nz, dt_dynamics);
    }
    if (attributed_acoustic_energy != nullptr)
    {
        if (coupled_transport == nullptr || !coupled_transport->carrier_kinetic_transport_complete() ||
            attributed_slow_momentum_energy == nullptr || !slow_momentum_energy_report.complete)
        {
            throw std::runtime_error("Cartesian kinetic transport attribution is missing an "
                                     "operator-time continuity or momentum component");
        }
        acoustic_energy_report.continuity_carrier_outward_boundary_transport_j =
            coupled_transport->outward_carrier_kinetic_transport_j();
        acoustic_energy_report.continuity_carrier_storage_rounding_j =
            acoustic_energy_report.continuity_carrier_kinetic_change_j -
            acoustic_energy_report.continuity_carrier_requested_change_j;
        acoustic_energy_report.continuity_carrier_interior_conversion_j =
            acoustic_energy_report.continuity_carrier_requested_change_j +
            acoustic_energy_report.continuity_carrier_outward_boundary_transport_j;
        acoustic_energy_report.slow_momentum_advection_requested_linear_work_j =
            slow_momentum_energy_report.advection_requested_linear_work_j;
        const double combined_transport_requested =
            acoustic_energy_report.continuity_carrier_requested_change_j +
            acoustic_energy_report.slow_momentum_advection_requested_linear_work_j;
        acoustic_energy_report.kinetic_transport_interior_conversion_j =
            combined_transport_requested + acoustic_energy_report.continuity_carrier_outward_boundary_transport_j;
        acoustic_energy_report.kinetic_transport_closure_residual_j =
            combined_transport_requested - (-acoustic_energy_report.continuity_carrier_outward_boundary_transport_j +
                                            acoustic_energy_report.kinetic_transport_interior_conversion_j);
        acoustic_energy_report.kinetic_transport_complete = true;

        if (!acoustic_energy_report.complete || acoustic_energy_report.substep_count == 0 ||
            !acoustic_energy_report.boundary_complete ||
            acoustic_energy_report.boundary_application_count != acoustic_energy_report.substep_count + 1)
        {
            throw std::runtime_error("requested Cartesian acoustic energy attribution was not "
                                     "completed by the owning operator");
        }
        acoustic_energy_report.committed_inventory_change_j =
            canonical_acoustic_committed_inventory(acoustic_energy_report);
        acoustic_energy_report.pressure_gradient_storage_rounding_j =
            acoustic_energy_report.pressure_gradient_committed_work_j -
            acoustic_energy_report.pressure_gradient_requested_linear_work_j -
            acoustic_energy_report.pressure_gradient_finite_step_correction_j;
        acoustic_energy_report.boundary_radiative_storage_rounding_j =
            acoustic_energy_report.boundary_radiative_committed_work_j -
            acoustic_energy_report.boundary_radiative_requested_linear_work_j -
            acoustic_energy_report.boundary_radiative_finite_step_correction_j;
        const double report_values[] = {
            acoustic_energy_report.continuity_eos_dry_internal_change_j,
            acoustic_energy_report.continuity_eos_water_sensible_change_j,
            acoustic_energy_report.continuity_eos_phase_reference_change_j,
            acoustic_energy_report.continuity_carrier_kinetic_change_j,
            acoustic_energy_report.continuity_gravitational_change_j,
            acoustic_energy_report.continuity_subgrid_tke_change_j,
            acoustic_energy_report.continuity_carrier_requested_change_j,
            acoustic_energy_report.continuity_carrier_outward_boundary_transport_j,
            acoustic_energy_report.continuity_carrier_interior_conversion_j,
            acoustic_energy_report.continuity_carrier_storage_rounding_j,
            acoustic_energy_report.slow_momentum_advection_requested_linear_work_j,
            acoustic_energy_report.kinetic_transport_interior_conversion_j,
            acoustic_energy_report.kinetic_transport_closure_residual_j,
            acoustic_energy_report.pressure_gradient_requested_linear_work_j,
            acoustic_energy_report.pressure_gradient_finite_step_correction_j,
            acoustic_energy_report.pressure_gradient_storage_rounding_j,
            acoustic_energy_report.pressure_gradient_committed_work_j,
            acoustic_energy_report.boundary_committed_work_j,
            acoustic_energy_report.boundary_radiative_requested_linear_work_j,
            acoustic_energy_report.boundary_radiative_finite_step_correction_j,
            acoustic_energy_report.boundary_radiative_storage_rounding_j,
            acoustic_energy_report.boundary_radiative_committed_work_j,
            acoustic_energy_report.boundary_constraint_history_committed_work_j,
            acoustic_energy_report.committed_inventory_change_j,
        };
        if (std::any_of(std::begin(report_values),
                        std::end(report_values),
                        [](double value) { return !std::isfinite(value); }) ||
            acoustic_energy_report.pressure_gradient_finite_step_correction_j < 0.0 ||
            acoustic_energy_report.boundary_radiative_finite_step_correction_j < 0.0)
        {
            throw std::runtime_error("Cartesian acoustic energy attribution produced an invalid "
                                     "completed-step report");
        }

        CartesianCGridAcousticEnergyReportV1 cumulative = time_runtime.cumulative_acoustic_energy;
        cumulative.continuity_eos_dry_internal_change_j += acoustic_energy_report.continuity_eos_dry_internal_change_j;
        cumulative.continuity_eos_water_sensible_change_j +=
            acoustic_energy_report.continuity_eos_water_sensible_change_j;
        cumulative.continuity_eos_phase_reference_change_j +=
            acoustic_energy_report.continuity_eos_phase_reference_change_j;
        cumulative.continuity_carrier_kinetic_change_j += acoustic_energy_report.continuity_carrier_kinetic_change_j;
        cumulative.continuity_gravitational_change_j += acoustic_energy_report.continuity_gravitational_change_j;
        cumulative.continuity_subgrid_tke_change_j += acoustic_energy_report.continuity_subgrid_tke_change_j;
        cumulative.continuity_carrier_requested_change_j +=
            acoustic_energy_report.continuity_carrier_requested_change_j;
        cumulative.continuity_carrier_outward_boundary_transport_j +=
            acoustic_energy_report.continuity_carrier_outward_boundary_transport_j;
        cumulative.continuity_carrier_interior_conversion_j +=
            acoustic_energy_report.continuity_carrier_interior_conversion_j;
        cumulative.continuity_carrier_storage_rounding_j +=
            acoustic_energy_report.continuity_carrier_storage_rounding_j;
        cumulative.slow_momentum_advection_requested_linear_work_j +=
            acoustic_energy_report.slow_momentum_advection_requested_linear_work_j;
        cumulative.kinetic_transport_interior_conversion_j +=
            acoustic_energy_report.kinetic_transport_interior_conversion_j;
        cumulative.pressure_gradient_requested_linear_work_j +=
            acoustic_energy_report.pressure_gradient_requested_linear_work_j;
        cumulative.pressure_gradient_finite_step_correction_j +=
            acoustic_energy_report.pressure_gradient_finite_step_correction_j;
        cumulative.pressure_gradient_storage_rounding_j += acoustic_energy_report.pressure_gradient_storage_rounding_j;
        cumulative.pressure_gradient_committed_work_j += acoustic_energy_report.pressure_gradient_committed_work_j;
        cumulative.boundary_committed_work_j += acoustic_energy_report.boundary_committed_work_j;
        cumulative.boundary_radiative_requested_linear_work_j +=
            acoustic_energy_report.boundary_radiative_requested_linear_work_j;
        cumulative.boundary_radiative_finite_step_correction_j +=
            acoustic_energy_report.boundary_radiative_finite_step_correction_j;
        cumulative.boundary_radiative_storage_rounding_j +=
            acoustic_energy_report.boundary_radiative_storage_rounding_j;
        cumulative.boundary_radiative_committed_work_j += acoustic_energy_report.boundary_radiative_committed_work_j;
        cumulative.boundary_constraint_history_committed_work_j +=
            acoustic_energy_report.boundary_constraint_history_committed_work_j;
        cumulative.boundary_application_count += acoustic_energy_report.boundary_application_count;
        cumulative.pressure_gradient_storage_rounding_j =
            cumulative.pressure_gradient_committed_work_j -
            cumulative.pressure_gradient_requested_linear_work_j -
            cumulative.pressure_gradient_finite_step_correction_j;
        cumulative.boundary_radiative_storage_rounding_j =
            cumulative.boundary_radiative_committed_work_j -
            cumulative.boundary_radiative_requested_linear_work_j -
            cumulative.boundary_radiative_finite_step_correction_j;
        cumulative.boundary_complete            = true;
        cumulative.committed_inventory_change_j = canonical_acoustic_committed_inventory(cumulative);
        cumulative.substep_count += acoustic_energy_report.substep_count;
        cumulative.complete                         = true;
        const double cumulative_transport_requested = cumulative.continuity_carrier_requested_change_j +
                                                      cumulative.slow_momentum_advection_requested_linear_work_j;
        cumulative.kinetic_transport_closure_residual_j =
            cumulative_transport_requested - (-cumulative.continuity_carrier_outward_boundary_transport_j +
                                              cumulative.kinetic_transport_interior_conversion_j);
        cumulative.kinetic_transport_complete = true;
        const double cumulative_values[]      = {
            cumulative.continuity_eos_dry_internal_change_j,
            cumulative.continuity_eos_water_sensible_change_j,
            cumulative.continuity_eos_phase_reference_change_j,
            cumulative.continuity_carrier_kinetic_change_j,
            cumulative.continuity_gravitational_change_j,
            cumulative.continuity_subgrid_tke_change_j,
            cumulative.continuity_carrier_requested_change_j,
            cumulative.continuity_carrier_outward_boundary_transport_j,
            cumulative.continuity_carrier_interior_conversion_j,
            cumulative.continuity_carrier_storage_rounding_j,
            cumulative.slow_momentum_advection_requested_linear_work_j,
            cumulative.kinetic_transport_interior_conversion_j,
            cumulative.kinetic_transport_closure_residual_j,
            cumulative.pressure_gradient_requested_linear_work_j,
            cumulative.pressure_gradient_finite_step_correction_j,
            cumulative.pressure_gradient_storage_rounding_j,
            cumulative.pressure_gradient_committed_work_j,
            cumulative.boundary_committed_work_j,
            cumulative.boundary_radiative_requested_linear_work_j,
            cumulative.boundary_radiative_finite_step_correction_j,
            cumulative.boundary_radiative_storage_rounding_j,
            cumulative.boundary_radiative_committed_work_j,
            cumulative.boundary_constraint_history_committed_work_j,
            cumulative.committed_inventory_change_j,
        };
        if (std::any_of(std::begin(cumulative_values),
                        std::end(cumulative_values),
                        [](double value) { return !std::isfinite(value); }))
        {
            throw std::overflow_error("Cartesian acoustic energy cumulative attribution overflowed");
        }
        time_runtime.last_acoustic_energy       = acoustic_energy_report;
        time_runtime.cumulative_acoustic_energy = cumulative;
        ++time_runtime.acoustic_energy_step_count;
    }
    if (coupled_transport != nullptr)
    {
        // Empty/partial contexts (for example, an accelerator path that has
        // not implemented flux-history export) are a hard error. Falling
        // through to end-state flux reconstruction is scientifically wrong.
        const DynamicsBoundaryMassReport mass_report = coupled_transport->boundary_mass_report(execution_state,
            sim_state.fields.rho, sim_state.grid.dr, sim_state.grid.dtheta, sim_state.grid.dz);
        time_runtime.last_boundary_mass = mass_report;
        time_runtime.cumulative_outward_boundary_mass_kg += mass_report.outward_boundary_mass_kg;
        time_runtime.cumulative_flux_implied_state_change_kg += mass_report.flux_implied_state_change_kg;
        time_runtime.cumulative_committed_state_change_kg += mass_report.committed_state_change_kg;
        time_runtime.cumulative_boundary_mass_closure_residual_kg += mass_report.closure_residual_kg;
        ++time_runtime.boundary_mass_step_count;
    }
    const int dynamics_bound_repairs = verify_primary_bounds("dynamics");
    if (dynamics_bound_repairs > 0)
    {
        // The guard reports an aggregate count, so conservatively invalidate
        // every thermodynamic field before the transport consumer.
        invalidate_thermodynamic_residency(execution_state);
    }
    const ConservationBudget budget_after_dynamics = measured_conservation_budget(sim_state, step_science_diag);
    if (step_science_diag)
        report_budget_transition(sim_state, "dynamics", budget_start, budget_after_dynamics, dt_dynamics);

    // Thermodynamics first (GPU audit Phase 4.3c): when the GPU batch
    // runs, it carries the passive tracer as one more scalar and the
    // advect_tracer call becomes a consume-and-skip. Order is physics-
    // neutral - every scalar reads only its own field plus u/v/w/rho, and
    // none of them writes those.
    AdvectionRuntimeState& advection_runtime =
        sim_state.advection_runtime
            ? *sim_state.advection_runtime
            : tmv::require_level_runtime(execution_state.advection_runtime, "advection_runtime");
    advection_runtime.thermodynamic_boundary_reports_requested =
        step_science_diag || advection_runtime.moist_interface.enabled;
    advection_runtime.thermodynamic_moist_energy_reports_requested = step_science_diag;
    const auto dynamics_advection_t0                               = dynamics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent advection_allocations(tmv::allocation_telemetry::Component::Advection);
#endif
    if (!use_coupled_unsplit_rk)
    {
        advect_thermodynamics_3d(execution_state, dt_dynamics, 0.0, 0.0, coupled_transport, &advection_runtime);
    }
    if ((step_science_diag || advection_runtime.moist_interface.enabled) && coupled_transport != nullptr &&
        !advection_runtime.thermodynamic_boundary_reports_complete)
    {
        throw std::runtime_error("science cadence requested thermodynamic boundary ledgers, "
                                 "but the selected transport path did not publish a complete set");
    }
    if (step_science_diag && coupled_transport != nullptr &&
        !advection_runtime.thermodynamic_moist_energy_reports_complete)
    {
        throw std::runtime_error("science cadence requested thermodynamic moist-energy ledgers, "
                                 "but the selected transport path did not publish a complete set");
    }
    if (advection_runtime.thermodynamic_boundary_reports_complete)
    {
        if (advection_runtime.thermodynamic_boundary_report_step_count == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error("thermodynamic boundary ledger step counter overflow");
        }
        auto cumulative_candidate = advection_runtime.cumulative_thermodynamic_boundary_reports;
        for (std::size_t scalar = 0; scalar < thermodynamic_boundary_report_count; ++scalar)
        {
            const auto& step       = advection_runtime.thermodynamic_boundary_reports[scalar];
            auto&       cumulative = cumulative_candidate[scalar];
            cumulative.outward_boundary_inventory += step.outward_boundary_inventory;
            cumulative.flux_implied_state_change += step.flux_implied_state_change;
            cumulative.committed_state_change += step.committed_state_change;
            cumulative.closure_residual += step.closure_residual;
            if (!std::isfinite(cumulative.outward_boundary_inventory) ||
                !std::isfinite(cumulative.flux_implied_state_change) ||
                !std::isfinite(cumulative.committed_state_change) || !std::isfinite(cumulative.closure_residual))
            {
                throw std::overflow_error("cumulative thermodynamic boundary ledger became non-finite");
            }
            cumulative.complete = true;
        }
        advection_runtime.cumulative_thermodynamic_boundary_reports = cumulative_candidate;
        ++advection_runtime.thermodynamic_boundary_report_step_count;
    }
    if (advection_runtime.thermodynamic_moist_energy_reports_complete)
    {
        if (advection_runtime.thermodynamic_moist_energy_report_step_count == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error("thermodynamic moist-energy ledger step counter overflow");
        }
        auto       cumulative_candidate = advection_runtime.cumulative_thermodynamic_moist_energy_reports;
        const auto accumulate_energy_report =
            [](ScalarMoistEnergyTransportReportV1& cumulative, const ScalarMoistEnergyTransportReportV1& step)
        {
            cumulative.dry_internal_change_j += step.dry_internal_change_j;
            cumulative.water_sensible_change_j += step.water_sensible_change_j;
            cumulative.phase_reference_change_j += step.phase_reference_change_j;
            cumulative.carrier_kinetic_change_j += step.carrier_kinetic_change_j;
            cumulative.gravitational_change_j += step.gravitational_change_j;
            cumulative.subgrid_tke_change_j += step.subgrid_tke_change_j;
            cumulative.density_history_remap_j += step.density_history_remap_j;
            cumulative.outward_boundary_energy_j += step.outward_boundary_energy_j;
            cumulative.limited_face_state_change_j += step.limited_face_state_change_j;
            cumulative.storage_rounding_j += step.storage_rounding_j;
            // Accumulate only independently authored primitive terms.
            // Re-derive dependent cumulative fields so their separately
            // rounded sums cannot drift while a falsely zero residual hides
            // the arithmetic difference.
            cumulative.interior_conversion_j =
                cumulative.limited_face_state_change_j + cumulative.outward_boundary_energy_j;
            cumulative.requested_state_change_j =
                cumulative.density_history_remap_j + cumulative.limited_face_state_change_j;
            cumulative.committed_state_change_j =
                cumulative.dry_internal_change_j + cumulative.water_sensible_change_j +
                cumulative.phase_reference_change_j + cumulative.carrier_kinetic_change_j +
                cumulative.gravitational_change_j + cumulative.subgrid_tke_change_j;
            cumulative.closure_residual_j = cumulative.committed_state_change_j -
                                            (cumulative.requested_state_change_j + cumulative.storage_rounding_j);
            const double values[]         = {
                cumulative.dry_internal_change_j,
                cumulative.water_sensible_change_j,
                cumulative.phase_reference_change_j,
                cumulative.carrier_kinetic_change_j,
                cumulative.gravitational_change_j,
                cumulative.subgrid_tke_change_j,
                cumulative.density_history_remap_j,
                cumulative.outward_boundary_energy_j,
                cumulative.interior_conversion_j,
                cumulative.limited_face_state_change_j,
                cumulative.requested_state_change_j,
                cumulative.storage_rounding_j,
                cumulative.committed_state_change_j,
                cumulative.closure_residual_j,
            };
            if (std::any_of(std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
            {
                throw std::overflow_error("cumulative thermodynamic moist-energy ledger became "
                                          "non-finite");
            }
            cumulative.complete = true;
        };
        for (std::size_t scalar = 0; scalar < thermodynamic_boundary_report_count; ++scalar)
        {
            accumulate_energy_report(cumulative_candidate[scalar],
                                     advection_runtime.thermodynamic_moist_energy_reports[scalar]);
        }
        advection_runtime.cumulative_thermodynamic_moist_energy_reports = cumulative_candidate;
        ++advection_runtime.thermodynamic_moist_energy_report_step_count;
    }
    if (!use_coupled_unsplit_rk)
    {
        advect_tracer_3d(execution_state,
            dt_dynamics, 0.0, coupled_transport, sim_state.advection_runtime);
    }
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    advection_allocations.restore();
#endif
    dynamics_perf_finish(dynamics_perf_totals.advection_s, dynamics_advection_t0);
    const auto dynamics_microphysics_transport_t0 = dynamics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent microphysics_transport_allocations(
        tmv::allocation_telemetry::Component::MicrophysicsTransport);
#endif
    MicrophysicsScheme* transport_microphysics =
        sim_state.microphysics_runtime ? sim_state.microphysics_runtime->scheme.get() : tmv::require_level_runtime(execution_state.microphysics_runtime, "microphysics_runtime").scheme.get();
    if (!use_coupled_unsplit_rk && transport_microphysics != nullptr)
    {
        tmv::prepare_microphysics_prognostics(sim_state, *transport_microphysics);
        advect_microphysics_prognostics_3d(execution_state,
            *transport_microphysics, dt_dynamics, coupled_transport, sim_state.advection_runtime);
    }
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    microphysics_transport_allocations.restore();
#endif
    dynamics_perf_finish(dynamics_perf_totals.microphysics_transport_s, dynamics_microphysics_transport_t0);
    verify_primary_bounds("advection");
    // Thermodynamic transport has just authored these host fields. Slow
    // dynamics can retain qv earlier in the same large step and the acoustic
    // loop can retain theta; hydrometeor slots may likewise have been
    // published by an earlier device consumer. None of those pre-transport
    // copies is authoritative here. Invalidate only the transported scalars,
    // preserving the current wind/density slots that the conservative
    // transport batch publishes for later GPU consumers.
    invalidate_named_fields(execution_state, {"theta", "qv", "qc", "qr", "qi", "qs", "qg", "qh"});
    // Advection gets its own budget bracket (SCIENCE.md M6): scalar
    // transport plus its post-advection clamps used to be folded into the
    // "microphysics" stage, sending anyone debugging a water budget to the
    // wrong subsystem.
    const auto nested_private_physics_begin = advection_runtime.private_interface.fields.empty()
                                                  ? std::vector<double>{}
                                                  : nested_private_inventory(sim_state);
    const auto nested_physics_begin =
        advection_runtime.moist_interface.enabled ? nested_thermodynamic_inventory(sim_state) : std::array<double, 8>{};
    const auto nested_momentum_inventory = [&]()
    {
        if (!dynamics_runtime.momentum_interface_flux_capture_enabled)
            return std::array<double, 3>{};
        const auto* terrain = tmv::physical_grid_metrics::active_terrain_metrics(sim_state);
        const auto& capture = dynamics_runtime.momentum_interface_flux_register;
        return terrain ? capture.native_dual_inventory_mapped(
                             sim_state.fields.rho, sim_state.fields.u, sim_state.fields.v, sim_state.fields.w, *terrain)
                       : capture.native_dual_inventory(sim_state.fields.rho,
                                                       sim_state.fields.u,
                                                       sim_state.fields.v,
                                                       sim_state.fields.w,
                                                       sim_state.grid.dr,
                                                       sim_state.grid.dtheta,
                                                       sim_state.grid.dz);
    };
    const auto               nested_physics_momentum_begin = nested_momentum_inventory();
    const ConservationBudget budget_after_advection        = measured_conservation_budget(sim_state, step_science_diag);
    if (step_science_diag)
        report_budget_transition(sim_state, "advection", budget_after_dynamics, budget_after_advection, dt_dynamics);

    {
        const auto dynamics_microphysics_t0 = dynamics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
        tmv::allocation_telemetry::ScopedComponent allocation_component(
            tmv::allocation_telemetry::Component::Microphysics);
#endif
        step_microphysics(sim_state, dt_dynamics);
        dynamics_perf_finish(dynamics_perf_totals.microphysics_s, dynamics_microphysics_t0);
    }
    verify_primary_bounds("microphysics");
    // CPU apply, guard projection, or the GPU apply may have authored these
    // fields. The Lin batch's state reuse window ends inside step_microphysics.
    invalidate_thermodynamic_residency(execution_state);
    const ConservationBudget budget_after_microphysics = measured_conservation_budget(sim_state, step_science_diag);
    if (step_science_diag)
        report_budget_transition(
            sim_state, "microphysics", budget_after_advection, budget_after_microphysics, dt_dynamics);
    if (step_radar_diag)
        calculate_radar_observables(execution_state);

    const auto dynamics_turbulence_t0 = dynamics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent turbulence_allocations(tmv::allocation_telemetry::Component::Turbulence);
#endif
    TurbulenceRuntimeState& turbulence_runtime =
        sim_state.turbulence_runtime ? *sim_state.turbulence_runtime : tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime");
    TurbulenceTendencies& turb_tend = turbulence_runtime.held_tendencies;
    const bool active_explicit_sgs  = turbulence_runtime.scheme && turbulence_runtime.scheme->name() != "none";
    if (active_explicit_sgs)
    {
        // dt_sgs is a maximum explicit integration interval, not permission
        // to reuse a nonlinear tendency after the state has changed.  Every
        // local commit is bounded against the final realized tendency (after
        // device production and chaos), then the closure is recomputed from
        // the updated state before any remaining interval is advanced.
        double                       advanced_s                  = 0.0;
        double                       observed_limit_s            = std::numeric_limits<double>::infinity();
        std::uint64_t                substep_count               = 0;
        std::uint64_t                open_boundary_substep_count = 0;
        double                       open_boundary_work_j        = 0.0;
        double                       irreversible_dissipation_j  = 0.0;
        double                       stress_closure_residual_j   = 0.0;
        TurbulenceMomentumWorkReport aggregate_work;
        constexpr std::uint64_t      maximum_substeps = 1000000;
        while (advanced_s < dt_dynamics)
        {
            const auto turbulence_refresh_t0 = dynamics_perf_start();
            refresh_turbulence(sim_state, current_time + advanced_s, turb_tend);
            dynamics_perf_finish(dynamics_perf_totals.turbulence_refresh_s, turbulence_refresh_t0);
            const auto turbulence_chaos_t0 = dynamics_perf_start();
            apply_chaos_to_turbulence_tendencies(sim_state, turb_tend);
            dynamics_perf_finish(dynamics_perf_totals.turbulence_chaos_s, turbulence_chaos_t0);
            const auto                 turbulence_stability_t0 = dynamics_perf_start();
            const SgsExplicitStability stability               = compute_sgs_explicit_stability(sim_state, turb_tend);
            dynamics_perf_finish(dynamics_perf_totals.turbulence_stability_s, turbulence_stability_t0);
            observed_limit_s = std::min(observed_limit_s, stability.maximum_timestep_s);

            const double remaining_s = dt_dynamics - advanced_s;
            const double substep_s =
                std::min({remaining_s, turbulence_runtime.config.dt_sgs, stability.maximum_timestep_s});
            const double progress_floor_s =
                64.0 * std::numeric_limits<double>::epsilon() * std::max({1.0, dt_dynamics, std::abs(current_time)});
            if (!std::isfinite(substep_s) || substep_s <= progress_floor_s || ++substep_count > maximum_substeps)
            {
                throw std::runtime_error("SGS explicit integration cannot make stable progress");
            }
            if (!std::isfinite(stability.momentum_irreversible_dissipation_w) ||
                stability.momentum_irreversible_dissipation_w < 0.0 ||
                !std::isfinite(stability.momentum_stress_closure_residual_w))
            {
                throw std::runtime_error("SGS stability provenance is incomplete");
            }
            irreversible_dissipation_j += substep_s * stability.momentum_irreversible_dissipation_w;
            stress_closure_residual_j += substep_s * stability.momentum_stress_closure_residual_w;
            if (stability.open_boundary_work_accounted)
            {
                open_boundary_work_j += substep_s * stability.momentum_boundary_work_w;
                ++open_boundary_substep_count;
            }

            const auto                         turbulence_apply_t0 = dynamics_perf_start();
            const TurbulenceMomentumWorkReport work = apply_admitted_sgs_substep(sim_state, substep_s, turb_tend);
            dynamics_perf_finish(dynamics_perf_totals.turbulence_apply_s, turbulence_apply_t0);
            accumulate_sgs_energy_work(aggregate_work, work);
            advanced_s += substep_s;
            if (dt_dynamics - advanced_s <= progress_floor_s)
                advanced_s = dt_dynamics;
        }
        const auto turbulence_commit_t0 = dynamics_perf_start();
        commit_sgs_momentum_work(sim_state, aggregate_work);
        dynamics_perf_finish(dynamics_perf_totals.turbulence_commit_s, turbulence_commit_t0);
        turbulence_runtime.last_explicit_stability_limit_s = observed_limit_s;
        turbulence_runtime.last_explicit_substep_count     = substep_count;
        turbulence_runtime.last_open_boundary_work_j       = open_boundary_work_j;
        turbulence_runtime.last_irreversible_dissipation_j = irreversible_dissipation_j;
        turbulence_runtime.last_stress_closure_residual_j  = stress_closure_residual_j;
        if (!std::isfinite(irreversible_dissipation_j) || irreversible_dissipation_j < 0.0 ||
            !std::isfinite(stress_closure_residual_j))
        {
            throw std::runtime_error("SGS dissipative stability provenance is incomplete");
        }
        turbulence_runtime.cumulative_irreversible_dissipation_j += irreversible_dissipation_j;
        turbulence_runtime.cumulative_stress_closure_residual_j += stress_closure_residual_j;
        if (open_boundary_substep_count > 0)
        {
            if (open_boundary_substep_count != substep_count || !std::isfinite(open_boundary_work_j) ||
                irreversible_dissipation_j <= 0.0)
                throw std::runtime_error("SGS open-boundary stability provenance is incomplete");
            turbulence_runtime.cumulative_open_boundary_work_j += open_boundary_work_j;
            ++turbulence_runtime.open_boundary_stability_step_count;
        }
    }
    else
    {
        const auto turbulence_refresh_t0 = dynamics_perf_start();
        (void)step_turbulence(sim_state, current_time, turb_tend);
        dynamics_perf_finish(dynamics_perf_totals.turbulence_refresh_s, turbulence_refresh_t0);
        const auto turbulence_apply_t0 = dynamics_perf_start();
        (void)apply_sgs_tendencies(sim_state, dt_dynamics, turb_tend);
        dynamics_perf_finish(dynamics_perf_totals.turbulence_apply_s, turbulence_apply_t0);
        turbulence_runtime.last_explicit_stability_limit_s = std::numeric_limits<double>::infinity();
        turbulence_runtime.last_explicit_substep_count     = 0;
        turbulence_runtime.last_open_boundary_work_j       = 0.0;
        turbulence_runtime.last_irreversible_dissipation_j = 0.0;
        turbulence_runtime.last_stress_closure_residual_j  = 0.0;
    }
    verify_primary_bounds("turbulence");
    invalidate_named_fields(execution_state, {"u", "v", "w", "theta", "qv", "tke"});
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    turbulence_allocations.restore();
#endif
    dynamics_perf_finish(dynamics_perf_totals.turbulence_s, dynamics_turbulence_t0);
    const ConservationBudget budget_after_turbulence = measured_conservation_budget(sim_state, step_science_diag);
    if (step_science_diag)
        report_budget_transition(
            sim_state, "turbulence", budget_after_microphysics, budget_after_turbulence, dt_dynamics);
    const auto dynamics_diffusion_t0 = dynamics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent diffusion_allocations(tmv::allocation_telemetry::Component::Diffusion);
#endif
    const bool diffusion_active = runtime_diffusion_enabled(sim_state);
    apply_runtime_diffusion(sim_state, dt_dynamics);
#ifdef NIMBUS_REDUNDANT_NOOP_BUDGET_SCANS
    verify_primary_bounds("diffusion");
    invalidate_all_resident_fields_backend();
#else
    if (diffusion_active)
    {
        verify_primary_bounds("diffusion");
        // Active runtime diffusion may author any configured prognostic
        // subset. A disabled zero-coefficient stage returns without a write,
        // so invalidating every resident slot there destroys valid reuse.
        invalidate_all_resident_fields_backend(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"));
    }
#endif
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    diffusion_allocations.restore();
#endif
    dynamics_perf_finish(dynamics_perf_totals.diffusion_s, dynamics_diffusion_t0);
#ifdef NIMBUS_REDUNDANT_NOOP_BUDGET_SCANS
    const ConservationBudget budget_after_diffusion = measured_conservation_budget(sim_state, step_science_diag);
#else
    const ConservationBudget budget_after_diffusion =
        diffusion_active ? measured_conservation_budget(sim_state, step_science_diag) : budget_after_turbulence;
#endif
    if (step_science_diag)
        report_budget_transition(sim_state, "diffusion", budget_after_turbulence, budget_after_diffusion, dt_dynamics);
    // EOS pressure re-diagnosis uses one FP64 implementation on every backend.
    // Its basis-v2 attribution is fused into that pressure writer. The GPU
    // path retains the scientifically equivalent upper/w-only/lateral sponge
    // fusion, but no longer substitutes its rejected FP32 EOS candidate.
    const auto dynamics_eos_t0 = dynamics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent eos_allocations(tmv::allocation_telemetry::Component::Eos);
#endif
    const auto eos_pressure_energy = tmv::dynamics_eos::apply_pressure_projection(sim_state);
    if (!eos_pressure_energy.complete)
        throw std::runtime_error("production EOS pressure projection produced incomplete "
                                 "basis-v2 attribution");
    time_runtime.last_eos_pressure_energy = eos_pressure_energy;
    tmv::dynamics_eos::accumulate_pressure_projection_energy(time_runtime.cumulative_eos_pressure_energy,
                                                             eos_pressure_energy);
    ++time_runtime.eos_pressure_energy_step_count;
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    eos_allocations.restore();
#endif
    dynamics_perf_finish(dynamics_perf_totals.eos_s, dynamics_eos_t0);

    const ConservationBudget budget_after_eos = measured_conservation_budget(sim_state, step_science_diag);
    if (step_science_diag)
        report_budget_transition(sim_state, "eos", budget_after_diffusion, budget_after_eos, dt_dynamics);

    // Rayleigh damping sponge layers: relax fields toward the base state near
    // boundaries to absorb outgoing gravity waves and prevent reflection.
    // Flat Cartesian device execution exports all three ordered pass reports
    // at its real post-fence commit. Unsupported/terrain paths retain the
    // fused CPU writers.
    const auto dynamics_sponge_t0 = dynamics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent sponge_allocations(tmv::allocation_telemetry::Component::Sponge);
#endif
    tmv::SpongeEnergyReportV1 sponge_energy;
    if (!try_state_relaxation_gpu(sim_state, dt_dynamics, &sponge_energy))
    {
        sponge_energy                                         = apply_rayleigh_damping(sim_state, dt_dynamics);
        const tmv::SpongeEnergyReportV1 lateral_sponge_energy = apply_lateral_damping(sim_state, dt_dynamics);
        tmv::accumulate_sponge_energy(sponge_energy, lateral_sponge_energy);
    }
    if (!sponge_energy.complete)
        throw std::runtime_error("production sponge relaxation produced incomplete basis-v2 "
                                 "attribution");
    time_runtime.last_sponge_energy = sponge_energy;
    tmv::accumulate_sponge_energy(time_runtime.cumulative_sponge_energy, sponge_energy);
    ++time_runtime.sponge_energy_step_count;
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    sponge_allocations.restore();
#endif
    dynamics_perf_finish(dynamics_perf_totals.sponge_s, dynamics_sponge_t0);

#ifdef NIMBUS_SEPARATE_ENDPOINT_MOISTURE_SCANS
    const ConservationBudget budget_after_sponge = measured_conservation_budget(sim_state, step_science_diag);
#else
    const ConservationMoistureSnapshots inventory_after_sponge =
        measured_conservation_moisture_snapshots(sim_state, step_science_diag);
    const ConservationBudget& budget_after_sponge   = inventory_after_sponge.conservation;
    const MoistureBudget&     moisture_after_sponge = inventory_after_sponge.moisture;
#endif
    if (step_science_diag)
        report_budget_transition(sim_state, "sponge", budget_after_eos, budget_after_sponge, dt_dynamics);

    if (step_vorticity_diag || step_angular_momentum_diag || step_pressure_diag)
        compute_dynamics_diagnostics(execution_state, step_vorticity_diag, step_angular_momentum_diag, step_pressure_diag);

    // The orchestration path applies the scheme directly so the compatibility
    // wrapper cannot add an unconditional hidden full-volume guard sweep.
    if (dynamics_runtime.momentum_interface_flux_capture_enabled)
    {
        // Split microphysics, SGS, diffusion and damping do not write the
        // acoustic/transport flux registers; their committed impulse is a source.
        const auto after = nested_momentum_inventory();
        dynamics_runtime.momentum_interface_flux_register.accumulate_operator_source_impulse(
            {after[0] - nested_physics_momentum_begin[0],
             after[1] - nested_physics_momentum_begin[1],
             after[2] - nested_physics_momentum_begin[2]});
    }
    const auto              dynamics_final_boundary_t0       = dynamics_perf_start();
    const bool              capture_boundary_momentum_source = dynamics_runtime.momentum_interface_flux_capture_enabled;
    const TerrainMetrics3D* boundary_terrain = tmv::physical_grid_metrics::active_terrain_metrics(sim_state);
    std::array<double, 3>   boundary_momentum_before{};
    Field3D                 boundary_rho_before;
    Field3D                 boundary_u_before;
    Field3D                 boundary_v_before;
    Field3D                 boundary_w_before;
    if (capture_boundary_momentum_source)
    {
        boundary_momentum_before = boundary_terrain != nullptr
            ? dynamics_runtime.momentum_interface_flux_register
                  .native_dual_inventory_mapped(
                      sim_state.fields.rho, sim_state.fields.u,
                      sim_state.fields.v, sim_state.fields.w,
                      *boundary_terrain)
            : dynamics_runtime.momentum_interface_flux_register.native_dual_inventory(
                  sim_state.fields.rho, sim_state.fields.u,
                  sim_state.fields.v, sim_state.fields.w,
                  sim_state.grid.dr, sim_state.grid.dtheta, sim_state.grid.dz);
        boundary_rho_before = sim_state.fields.rho;
        boundary_u_before = sim_state.fields.u;
        boundary_v_before = sim_state.fields.v;
        boundary_w_before = sim_state.fields.w;
    }
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent final_boundary_allocations(
        tmv::allocation_telemetry::Component::FinalBoundary);
#endif
    tmv::FinalBoundaryEnergyReportV1 final_boundary_energy;
    if (execution_state.configuration->admission.earth_fixed_terrain_motion && execution_state.compute_runtime &&
        execution_state.compute_runtime->active_kind != ComputeBackendKind::Cpu)
    {
        auto* kernels = execution_state.compute_runtime->backend
                            ? execution_state.compute_runtime->backend->terrain_kernels()
                            : nullptr;
        if (!kernels)
            throw std::runtime_error("native Earth-fixed final boundary owner is unavailable");
        tmv::apply_native_cgrid_boundary(execution_state, *kernels, false, &final_boundary_energy);
    }
    else
        final_boundary_energy = bc_scheme->apply_full_attributed();
    invalidate_boundary_field_residency(execution_state, bc_scheme->full_attributed_write_set());
    if (!tmv::valid_final_boundary_energy_report(final_boundary_energy) || !final_boundary_energy.complete)
    {
        throw std::runtime_error("production final boundary closure produced incomplete "
                                 "basis-v2 attribution");
    }
    if (capture_boundary_momentum_source)
    {
        try
        {
            const auto boundary_momentum_after =
                boundary_terrain != nullptr
                    ? dynamics_runtime.momentum_interface_flux_register.native_dual_inventory_mapped(
                          sim_state.fields.rho,
                          sim_state.fields.u,
                          sim_state.fields.v,
                          sim_state.fields.w,
                          *boundary_terrain)
                    : dynamics_runtime.momentum_interface_flux_register.native_dual_inventory(sim_state.fields.rho,
                                                                                              sim_state.fields.u,
                                                                                              sim_state.fields.v,
                                                                                              sim_state.fields.w,
                                                                                              sim_state.grid.dr,
                                                                                              sim_state.grid.dtheta,
                                                                                              sim_state.grid.dz);
            dynamics_runtime.momentum_interface_flux_register
                .accumulate_boundary_source_impulse({
                    boundary_momentum_after[0] - boundary_momentum_before[0],
                    boundary_momentum_after[1] - boundary_momentum_before[1],
                    boundary_momentum_after[2] - boundary_momentum_before[2],
                });
        }
        catch (...)
        {
            std::swap(sim_state.fields.rho, boundary_rho_before);
            std::swap(sim_state.fields.u, boundary_u_before);
            std::swap(sim_state.fields.v, boundary_v_before);
            std::swap(sim_state.fields.w, boundary_w_before);
            throw;
        }
    }
    tmv::FinalBoundaryEnergyReportV1 cumulative_final_boundary_energy = time_runtime.cumulative_final_boundary_energy;
    tmv::accumulate_final_boundary_energy(cumulative_final_boundary_energy, final_boundary_energy);
    if (!tmv::valid_final_boundary_energy_report(cumulative_final_boundary_energy) ||
        !cumulative_final_boundary_energy.complete)
    {
        throw std::overflow_error("production cumulative final boundary energy attribution "
                                  "overflowed");
    }
    time_runtime.last_final_boundary_energy       = final_boundary_energy;
    time_runtime.cumulative_final_boundary_energy = cumulative_final_boundary_energy;
    ++time_runtime.final_boundary_energy_step_count;
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    final_boundary_allocations.restore();
#endif
    dynamics_perf_finish(dynamics_perf_totals.final_boundary_s, dynamics_final_boundary_t0);

    if (step_science_diag &&
        (use_coupled_unsplit_rk ||
         (attributed_acoustic_energy != nullptr &&
          attributed_slow_momentum_energy != nullptr)))
    {
        const tmv::CompletedStepEnergyReportV1 completed_step_energy =
            use_coupled_unsplit_rk
            ? compose_unsplit_rk_completed_step_energy(
                  sim_state, time_runtime,
                  time_runtime.last_completed_step_energy)
            : compose_completed_step_energy(
                  sim_state, time_runtime, advection_runtime);
        tmv::CompletedStepEnergyReportV1 cumulative_completed_step_energy =
            time_runtime.cumulative_completed_step_energy;
        tmv::accumulate_completed_step_energy(cumulative_completed_step_energy, completed_step_energy);
        if (!tmv::structurally_valid_completed_step_energy_report(cumulative_completed_step_energy) ||
            !cumulative_completed_step_energy.complete)
        {
            throw std::overflow_error("cumulative completed-step energy composition overflowed");
        }
        if (time_runtime.completed_step_energy_count == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error("completed-step energy composition counter overflowed");
        }
        time_runtime.last_completed_step_energy       = completed_step_energy;
        time_runtime.cumulative_completed_step_energy = cumulative_completed_step_energy;
        ++time_runtime.completed_step_energy_count;
    }
    verify_primary_bounds("boundary_conditions");
    if (!advection_runtime.private_interface.fields.empty())
    {
        const auto after  = nested_private_inventory(sim_state);
        auto&      fields = advection_runtime.private_interface.fields;
        if (after.size() != fields.size() || after.size() != nested_private_physics_begin.size())
            throw std::runtime_error("private physics source layout changed");
        std::vector<double> candidate(fields.size());
        for (std::size_t f = 0; f < fields.size(); ++f)
        {
            candidate[f] = fields[f].committed_physics + after[f] - nested_private_physics_begin[f];
            if (!std::isfinite(candidate[f]))
                throw std::overflow_error("private physics source overflowed");
        }
        for (std::size_t f = 0; f < fields.size(); ++f)
            fields[f].committed_physics = candidate[f];
    }
    if (advection_runtime.moist_interface.enabled)
    {
        const auto after     = nested_thermodynamic_inventory(sim_state);
        auto       candidate = advection_runtime.moist_interface.committed_physics;
        for (std::size_t f = 0; f < candidate.size(); ++f)
        {
            candidate[f] += after[f] - nested_physics_begin[f];
            if (!std::isfinite(candidate[f]))
                throw std::overflow_error("nested scalar physics source overflowed");
        }
        advection_runtime.moist_interface.committed_physics = candidate;
    }

#ifdef NIMBUS_REDUNDANT_NOOP_BUDGET_SCANS
    const ConservationBudget budget_after_bc = measured_conservation_budget(sim_state, step_science_diag);
#else
    // Both maintained boundary schemes leave every physical scalar-budget
    // cell unchanged: C-grid closure writes velocity faces and lateral scalar
    // halos, while A-grid closure writes only excluded halo planes. Preserve
    // the named zero transition without repeating the volume reduction.
    const ConservationBudget& budget_after_bc = budget_after_sponge;
#endif
    if (step_science_diag)
        report_budget_transition(sim_state, "bc", budget_after_sponge, budget_after_bc, dt_dynamics);

    // Publish source provenance only after every state commit represented by
    // the completed-timestep row. This keeps EOS/sponge/final-boundary owners
    // on the same restart boundary as SGS and diffusion.
    if (!sim_state.defer_step_reports)
    write_thermodynamic_source_budget_row(sim_state, current_time, dt_dynamics);

    log_step_velocity_maxima_if_enabled(sim_state);

    // No prognostic state writer runs between the post-boundary snapshot and
    // total-step reporting. Reuse that exact authoritative endpoint instead
    // of repeating a full-volume deterministic reduction. The source-ledger
    // write and optional velocity log above are read-only consumers.
    const ConservationBudget& budget_final = budget_after_bc;
    if (step_science_diag && !sim_state.defer_step_reports)
        report_budget_transition(sim_state, "total_step", budget_start, budget_final, dt_dynamics);

    // F.5b per-step closure lines (mass is covered above). These are the
    // budgets the F.6.C conservation gates assert on.
    if (step_science_diag && !sim_state.defer_step_reports)
    {
        report_momentum_transition(
            sim_state, "total_step", momentum_start, measured_momentum_budget(sim_state, true), dt_dynamics);
        report_energy_transition(
            sim_state, "total_step", energy_start, measured_energy_snapshot(sim_state, true), dt_dynamics);
        report_moisture_transition(sim_state,
                                   "total_step",
                                   moisture_start,
#ifdef NIMBUS_SEPARATE_ENDPOINT_MOISTURE_SCANS
                                   measured_moisture_budget(sim_state, true),
#else
                                   moisture_after_sponge,
#endif
                                   dt_dynamics);
    }
    if (advection_runtime.dry_mass_interface_flux_capture_enabled)
    {
        if (coupled_transport == nullptr)
            throw std::runtime_error(
                "dry-mass interface capture requires retained C-grid transport fluxes");
        coupled_transport->require_valid(execution_state,
            sim_state.grid.nr, sim_state.grid.nth, sim_state.grid.nz,
            dt_dynamics);
        const TerrainMetrics3D* interface_terrain =
            tmv::physical_grid_metrics::active_terrain_metrics(sim_state);
        advection_runtime.dry_mass_interface_flux_register.accumulate(
            coupled_transport->mass_flux_x.data(),
            coupled_transport->mass_flux_y.data(), dt_dynamics,
            interface_terrain != nullptr
                ? interface_terrain->dx : sim_state.grid.dr,
            interface_terrain != nullptr
                ? interface_terrain->dy : sim_state.grid.dtheta,
            interface_terrain != nullptr
                ? interface_terrain->dzeta : sim_state.grid.dz);
    }
    if (tmv::physical_grid_metrics::terrain_is_active(sim_state)
        && !sim_state.grid.geometry.staggered)
    {
        if (coupled_transport == nullptr
            || !coupled_transport->is_complete()
            || !advection_runtime.thermodynamic_boundary_reports_complete
            || !advection_runtime.thermodynamic_moist_energy_reports_complete
            || !time_runtime.last_acoustic_energy.complete
            || !time_runtime.last_slow_momentum_energy.complete
            || !time_runtime.last_eos_pressure_energy.complete
            || !time_runtime.last_final_boundary_energy.complete
            || !time_runtime.last_completed_step_energy.complete)
        {
            throw std::runtime_error(
                "mapped A-grid completed-step ownership is incomplete");
        }
        tmv::log_info(
            "[TERRAIN_AGRID] owner=cpu, split_explicit=1, retained_flux=1, "
            "scalar_ledgers=1, completed_step=1, acoustic_substeps=",
            coupled_transport->substep_count());
    }
    dynamics_perf_finish(dynamics_perf_totals.total_s, dynamics_total_t0);
}

void tmv::reset_dynamics_perf_stats()
{
    dynamics_perf_totals = DynamicsPerfTotals{};
    reset_microphysics_perf_stats();
}

void tmv::log_dynamics_perf_summary(double model_step_total_s)
{
    if (!global_perf_timing_enabled)
        return;

    const double profiled_s =
        dynamics_perf_totals.core_s + dynamics_perf_totals.advection_s + dynamics_perf_totals.microphysics_transport_s +
        dynamics_perf_totals.microphysics_s + dynamics_perf_totals.turbulence_s + dynamics_perf_totals.diffusion_s +
        dynamics_perf_totals.eos_s + dynamics_perf_totals.sponge_s + dynamics_perf_totals.final_boundary_s +
        dynamics_perf_totals.science_conservation_s + dynamics_perf_totals.science_momentum_s +
        dynamics_perf_totals.science_energy_s + dynamics_perf_totals.science_moisture_s;
    const double                   residual_s           = dynamics_perf_totals.total_s - profiled_s;
    const double                   dynamics_denominator = std::max(dynamics_perf_totals.total_s, 1.0e-12);
    const double                   step_denominator     = std::max(model_step_total_s, 1.0e-12);
    const MicrophysicsPerfSnapshot microphysics_perf    = microphysics_perf_snapshot();
    const double microphysics_subphase_s = microphysics_perf.sedimentation_limit_s + microphysics_perf.pbl_reprojection_s + microphysics_perf.input_validation_s +
                                           microphysics_perf.transaction_snapshot_s + microphysics_perf.scheme_s +
                                           microphysics_perf.chaos_s + microphysics_perf.column_admission_s + microphysics_perf.tendency_contract_s +
                                           microphysics_perf.validation_s + microphysics_perf.private_state_s +
                                           microphysics_perf.precipitation_s + microphysics_perf.projection_s +
                                           microphysics_perf.apply_s + microphysics_perf.finalize_s;

    tmv::log_info("[DYNAMICS PERF] dynamics_total_s=",
                  dynamics_perf_totals.total_s,
                  ", dynamics_core_s=",
                  dynamics_perf_totals.core_s,
                  ", dynamics_acoustic_energy_snapshot_s=",
                  dynamics_perf_totals.acoustic_energy_snapshot_s,
                  ", dynamics_acoustic_state_copy_s=",
                  dynamics_perf_totals.acoustic_state_copy_s,
                  ", dynamics_acoustic_transport_copy_s=",
                  dynamics_perf_totals.acoustic_transport_copy_s,
                  ", dynamics_acoustic_transport_flux_s=",
                  dynamics_perf_totals.acoustic_transport_flux_s,
                  ", dynamics_acoustic_dispatch_s=",
                  dynamics_perf_totals.acoustic_dispatch_s,
                  ", dynamics_acoustic_attribution_s=",
                  dynamics_perf_totals.acoustic_attribution_s,
                  ", dynamics_acoustic_initial_validation_s=",
                  dynamics_perf_totals.acoustic_initial_validation_s,
                  ", dynamics_slow_compute_s=",
                  dynamics_perf_totals.slow_compute_s,
                  ", dynamics_slow_apply_s=",
                  dynamics_perf_totals.slow_apply_s,
                  ", dynamics_acoustic_boundary_s=",
                  dynamics_perf_totals.acoustic_boundary_s,
                  ", dynamics_advection_s=",
                  dynamics_perf_totals.advection_s,
                  ", dynamics_microphysics_transport_s=",
                  dynamics_perf_totals.microphysics_transport_s,
                  ", dynamics_microphysics_s=",
                  dynamics_perf_totals.microphysics_s,
                  ", dynamics_microphysics_pbl_reprojection_s=",
                  microphysics_perf.pbl_reprojection_s,
                  ", dynamics_microphysics_input_validation_s=",
                  microphysics_perf.input_validation_s,
                  ", dynamics_microphysics_sedimentation_limit_s=",
                  microphysics_perf.sedimentation_limit_s,
                  ", dynamics_microphysics_transaction_snapshot_s=",
                  microphysics_perf.transaction_snapshot_s,
                  ", dynamics_microphysics_scheme_s=",
                  microphysics_perf.scheme_s,
                  ", dynamics_microphysics_chaos_s=",
                  microphysics_perf.chaos_s,
                  ", dynamics_microphysics_column_admission_s=",
                  microphysics_perf.column_admission_s,
                  ", dynamics_microphysics_tendency_contract_s=",
                  microphysics_perf.tendency_contract_s,
                  ", dynamics_microphysics_validation_s=",
                  microphysics_perf.validation_s,
                  ", dynamics_microphysics_private_state_s=",
                  microphysics_perf.private_state_s,
                  ", dynamics_microphysics_precipitation_s=",
                  microphysics_perf.precipitation_s,
                  ", dynamics_microphysics_projection_s=",
                  microphysics_perf.projection_s,
                  ", dynamics_microphysics_apply_s=",
                  microphysics_perf.apply_s,
                  ", dynamics_microphysics_finalize_s=",
                  microphysics_perf.finalize_s,
                  ", dynamics_microphysics_residual_s=",
                  dynamics_perf_totals.microphysics_s - microphysics_subphase_s,
                  ", dynamics_turbulence_s=",
                  dynamics_perf_totals.turbulence_s,
                  ", dynamics_turbulence_refresh_s=",
                  dynamics_perf_totals.turbulence_refresh_s,
                  ", dynamics_turbulence_chaos_s=",
                  dynamics_perf_totals.turbulence_chaos_s,
                  ", dynamics_turbulence_stability_s=",
                  dynamics_perf_totals.turbulence_stability_s,
                  ", dynamics_turbulence_apply_s=",
                  dynamics_perf_totals.turbulence_apply_s,
                  ", dynamics_turbulence_commit_s=",
                  dynamics_perf_totals.turbulence_commit_s,
                  ", dynamics_diffusion_s=",
                  dynamics_perf_totals.diffusion_s,
                  ", dynamics_eos_s=",
                  dynamics_perf_totals.eos_s,
                  ", dynamics_sponge_s=",
                  dynamics_perf_totals.sponge_s,
                  ", dynamics_final_boundary_s=",
                  dynamics_perf_totals.final_boundary_s,
                  ", dynamics_science_conservation_s=",
                  dynamics_perf_totals.science_conservation_s,
                  ", dynamics_science_momentum_s=",
                  dynamics_perf_totals.science_momentum_s,
                  ", dynamics_science_energy_s=",
                  dynamics_perf_totals.science_energy_s,
                  ", dynamics_science_moisture_s=",
                  dynamics_perf_totals.science_moisture_s,
                  ", dynamics_profiled_s=",
                  profiled_s,
                  ", dynamics_residual_s=",
                  residual_s,
                  ", dynamics_profiled_fraction=",
                  profiled_s / dynamics_denominator,
                  ", dynamics_step_fraction=",
                  dynamics_perf_totals.total_s / step_denominator);
}

// Refresh only the requested diagnostics to avoid unnecessary volume passes.
void tmv::compute_dynamics_diagnostics(bool compute_vorticity, bool compute_angular_momentum, bool compute_pressure)
{
    return compute_dynamics_diagnostics(
        global_simulation_state(), compute_vorticity, compute_angular_momentum, compute_pressure);
}

void tmv::compute_dynamics_diagnostics(const SimulationStateView& execution_state,
                                       bool                       compute_vorticity,
                                       bool                       compute_angular_momentum,
                                       bool                       compute_pressure)
{
    tmv::diagnostics::compute_resolved_diagnostics(
        execution_state, {compute_vorticity, compute_angular_momentum, compute_pressure, compute_pressure});
}

// Legacy process-state adapter.
void tmv::step_dynamics(double current_time)
{
    return step_dynamics(global_simulation_state(), current_time);
}

void tmv::step_dynamics(const SimulationStateView& execution_state, double current_time)
{
    // dynamics_scheme is guaranteed non-null: initialize_dynamics() throws if
    // the config does not resolve to a real kernel. The legacy no-scheme
    // fallback (step_dynamics_old) has been removed.
    if (!tmv::require_level_runtime(execution_state.dynamics_runtime, "dynamics_runtime").scheme)
    {
        throw std::runtime_error("step_dynamics: no dynamics scheme initialized; "
                                 "initialize_dynamics() must run first.");
    }
    const bool native_dry_tvd = execution_state.configuration &&
        execution_state.configuration->admission.native_dry_tvd &&
        execution_state.compute_runtime &&
        execution_state.compute_runtime->active_kind == ComputeBackendKind::Metal &&
        !execution_state.compute_runtime->fallback_active;
    const bool static_vertical_cpu = execution_state.configuration &&
        execution_state.configuration->admission.static_vertical_only_sgs_cpu &&
        execution_state.compute_runtime &&
        execution_state.compute_runtime->active_kind == ComputeBackendKind::Cpu &&
        !execution_state.compute_runtime->fallback_active;
    if (native_dry_tvd || static_vertical_cpu)
    {
        if (!execution_state.storage_owner || execution_state.budget_level != tmv::BudgetLevel::Root)
        {
            throw std::runtime_error("transactional flat C-grid step requires a root-level storage owner");
        }
        tmv::runtime::FlatCgridStepTransaction state_transaction(*execution_state.storage_owner);
        tmv::BudgetPublicationTransaction budget_transaction;
        step_dynamics_new(execution_state, execution_state.grid.dt, current_time);
        budget_transaction.commit();
        state_transaction.commit();
        return;
    }
    step_dynamics_new(execution_state, execution_state.grid.dt, current_time);
}
