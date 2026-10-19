#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "turbulence/runtime.hpp"
#include "turbulence/factory.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "util/log.hpp"
#include "util/physics_cadence.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmv
{

std::unique_ptr<TurbulenceSchemeBase>& turbulence_scheme =
    global_turbulence_runtime_state.scheme;
TurbulenceConfig& global_turbulence_config =
    global_turbulence_runtime_state.config;

namespace
{
struct TurbulencePerfTotals
{
    double smagorinsky_coefficients_s = 0.0;
    double smagorinsky_momentum_s = 0.0;
    double smagorinsky_theta_s = 0.0;
    double smagorinsky_moisture_s = 0.0;
};

TurbulencePerfTotals turbulence_perf_totals;

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string normalize_scheme_name(std::string value)
{
    return canonicalize_scheme_id(get_turbulence_scheme_descriptors(), value);
}

std::string normalize_turbulence_mode(std::string value)
{
    value = lower_copy(value);
    if (value == "full")
    {
        return "3d";
    }
    if (value == "2d" || value == "horizontal" ||
        value == "horizontal-only")
    {
        return "horizontal_only";
    }
    if (value == "vertical" || value == "vertical-only")
    {
        return "vertical_only";
    }
    return value;
}

bool valid_turbulence_mode(const std::string& mode)
{
    return mode == "3d" || mode == "horizontal_only" || mode == "vertical_only";
}

bool valid_filter_width(const std::string& value)
{
    return value == "dx" || value == "cubic_root" || value == "user";
}

bool valid_stability_correction(const std::string& value)
{
    return known_turbulence_stability_model(value);
}

bool binds_process_grid(const SimulationStateView& state)
{
    const auto process = state;
    return &state.grid.nr == &process.grid.nr &&
           &state.grid.nth == &process.grid.nth &&
           &state.grid.nz == &process.grid.nz &&
           &state.grid.dr == &process.grid.dr &&
           &state.grid.dz == &process.grid.dz &&
           &state.grid.dtheta == &process.grid.dtheta;
}

const GridMetrics& get_runtime_turbulence_grid(
    const SimulationStateView& state,
    GridMetrics& scratch)
{
    const TerrainMetrics3D* terrain_metrics =
        physical_grid_metrics::active_terrain_metrics(state);
    const bool global_grid_ready =
        binds_process_grid(state) &&
        std::isfinite(tmv::require_level_runtime(state.process_grid_metrics, "grid metrics").dx) &&
        std::isfinite(tmv::require_level_runtime(state.process_grid_metrics, "grid metrics").dy) &&
        tmv::require_level_runtime(state.process_grid_metrics, "grid metrics").dx > 0.0 &&
        tmv::require_level_runtime(state.process_grid_metrics, "grid metrics").dy > 0.0 &&
        static_cast<int>(tmv::require_level_runtime(state.process_grid_metrics, "grid metrics").dz.size()) == state.grid.nz;

    if (global_grid_ready && terrain_metrics == nullptr)
    {
        return tmv::require_level_runtime(state.process_grid_metrics, "grid metrics");
    }

    GridMetrics& grid = scratch;
    grid.dx = state.grid.dr;
    grid.dy = representative_dy(state.grid.geometry.coord, state.grid.dr, state.grid.dtheta);
    grid.staggering = state.grid.geometry.staggered
        ? StaggerType::CGrid : StaggerType::Collocated;
    grid.dz.assign(static_cast<std::size_t>(state.grid.nz), state.grid.dz);
    grid.terrain_metrics_active = false;
    grid.terrain_metrics = nullptr;
    grid.terrain_topography = nullptr;
    if (terrain_metrics != nullptr)
    {
        physical_grid_metrics::validate(state);
        grid.terrain_metrics_active = true;
        grid.terrain_metrics = terrain_metrics;
        grid.terrain_topography = &state.terrain_runtime->topography;
        for (int k = 0; k < state.grid.nz; ++k)
        {
            double minimum = std::numeric_limits<double>::infinity();
            for (int i = 0; i < state.grid.nr; ++i)
                for (int j = 0; j < state.grid.nth; ++j)
                    minimum = std::min(
                        minimum,
                        terrain_metrics->z_face(i, j, k + 1) -
                            terrain_metrics->z_face(i, j, k));
            if (!std::isfinite(minimum) || minimum <= 0.0)
                throw std::invalid_argument(
                    "terrain turbulence layer thickness must be positive and finite");
            grid.dz[static_cast<std::size_t>(k)] = minimum;
        }
    }
    grid.z_int.resize(static_cast<std::size_t>(state.grid.nz + 1));
    grid.z_int[0] = 0.0;
    for (int k = 1; k <= state.grid.nz; ++k)
    {
        grid.z_int[static_cast<std::size_t>(k)] =
            grid.z_int[static_cast<std::size_t>(k - 1)] +
            grid.dz[static_cast<std::size_t>(k - 1)];
    }
    return grid;
}

void require_finite_tendencies(const TurbulenceTendencies& tendencies,
                               const SimulationStateView& state)
{
    const std::array<const Field3D*, 6> fields = {
        &tendencies.dudt_sgs,
        &tendencies.dvdt_sgs,
        &tendencies.dwdt_sgs,
        &tendencies.dthetadt_sgs,
        &tendencies.dqvdt_sgs,
        &tendencies.dtkedt_sgs,
    };
    constexpr std::array<const char*, 6> labels = {
        "dudt_sgs",
        "dvdt_sgs",
        "dwdt_sgs",
        "dthetadt_sgs",
        "dqvdt_sgs",
        "dtkedt_sgs",
    };
    for (std::size_t field_index = 0;
         field_index < fields.size();
         ++field_index)
    {
        const Field3D& field = *fields[field_index];
        if (field.size_r() != state.grid.nr ||
            field.size_th() != state.grid.nth ||
            field.size_z() != state.grid.nz)
        {
            throw std::runtime_error(
                std::string("turbulence scheme returned malformed ")
                + labels[field_index]);
        }
    }
    const std::size_t cell_count = fields.front()->size();
    for (std::size_t index = 0; index < cell_count; ++index)
    {
        for (std::size_t field_index = 0;
             field_index < fields.size();
             ++field_index)
        {
            if (!std::isfinite(static_cast<double>(
                    fields[field_index]->data()[index])))
            {
                throw std::runtime_error(
                    std::string("turbulence scheme returned non-finite ")
                    + labels[field_index]);
            }
        }
    }
}

void require_finite_diagnostics(const TurbulenceDiagnostics& diagnostics,
                                const SimulationStateView& state)
{
    const std::array<const Field3D*, 9> fields = {
        &diagnostics.nu_t,
        &diagnostics.K_theta,
        &diagnostics.K_q,
        &diagnostics.K_tke,
        &diagnostics.shear_prod,
        &diagnostics.buoy_prod,
        &diagnostics.dissipation,
        &diagnostics.Cs_eff,
        &diagnostics.Pr_t_eff,
    };
    constexpr std::array<const char*, 9> labels = {
        "nu_t", "K_theta", "K_q", "K_tke", "shear_prod",
        "buoy_prod", "dissipation", "Cs_eff", "Pr_t_eff",
    };
    for (std::size_t field_index = 0; field_index < fields.size(); ++field_index)
    {
        const Field3D& field = *fields[field_index];
        if (field.size_r() != state.grid.nr ||
            field.size_th() != state.grid.nth ||
            field.size_z() != state.grid.nz)
        {
            throw std::runtime_error(
                std::string("turbulence scheme returned malformed diagnostic ") +
                labels[field_index]);
        }
        for (std::size_t index = 0; index < field.size(); ++index)
        {
            if (!std::isfinite(static_cast<double>(field.data()[index])))
            {
                throw std::runtime_error(
                    std::string("turbulence scheme returned non-finite diagnostic ") +
                    labels[field_index]);
            }
        }
    }
}

void initialize_turbulence_runtime(std::unique_ptr<TurbulenceSchemeBase>& scheme,
                                   TurbulenceConfig& config,
                                   double& last_update_time_s,
                                   double& next_update_time_s,
                                   TurbulenceTendencies& held_tendencies,
                                   const std::string& scheme_name,
                                   const TurbulenceConfig& cfg)
{
    TurbulenceConfig selected_config = cfg;
    std::string requested_scheme = scheme_name.empty() ? cfg.scheme_id : scheme_name;
    requested_scheme = normalize_scheme_name(requested_scheme);
    if (requested_scheme.empty())
    {
        requested_scheme = "smagorinsky";
    }
    selected_config.scheme_id = requested_scheme;

    require_physics_cadence(selected_config.dt_sgs, "Turbulence dt_sgs");
    if (!std::isfinite(selected_config.Cs) || selected_config.Cs < 0.0)
    {
        throw std::invalid_argument("Turbulence Cs must be non-negative and finite");
    }
    if (!std::isfinite(selected_config.Pr_t) || selected_config.Pr_t <= 0.0)
    {
        throw std::invalid_argument("Turbulence Pr_t must be positive and finite");
    }
    if (!std::isfinite(selected_config.Sc_t) || selected_config.Sc_t <= 0.0)
    {
        throw std::invalid_argument("Turbulence Sc_t must be positive and finite");
    }
    if (!std::isfinite(selected_config.nu_t_max) || selected_config.nu_t_max < 0.0)
    {
        throw std::invalid_argument("Turbulence nu_t_max must be non-negative and finite");
    }
    if (!std::isfinite(selected_config.K_max) || selected_config.K_max < 0.0)
    {
        throw std::invalid_argument("Turbulence K_max must be non-negative and finite");
    }

    selected_config.mode = normalize_turbulence_mode(selected_config.mode);
    if (!valid_turbulence_mode(selected_config.mode))
    {
        throw std::invalid_argument(
            "Invalid turbulence mode '" + selected_config.mode +
            "'. Supported modes: 2d/horizontal_only, 3d, vertical_only");
    }
    selected_config.filter_width = lower_copy(selected_config.filter_width);
    if (!valid_filter_width(selected_config.filter_width))
    {
        throw std::invalid_argument(
            "Invalid turbulence filter_width '" + selected_config.filter_width +
            "'. Supported values: dx, cubic_root, user");
    }
    if (!std::isfinite(selected_config.user_filter_width_m) ||
        (selected_config.filter_width == "user"
             ? selected_config.user_filter_width_m <= 0.0
             : selected_config.user_filter_width_m != 0.0))
    {
        throw std::invalid_argument(
            "Turbulence user_filter_width_m must be positive and finite only "
            "when filter_width=user, and zero otherwise");
    }
    selected_config.stability_correction =
        lower_copy(selected_config.stability_correction);
    if (!valid_stability_correction(selected_config.stability_correction))
    {
        throw std::invalid_argument(
            "Invalid turbulence stability_correction '" +
            selected_config.stability_correction +
            "'. Supported values: none, ri, ri_lilly_v1, tke_based");
    }
    if (selected_config.dynamic_Cs &&
        selected_config.scheme_id != "smagorinsky")
    {
        throw std::invalid_argument(
            "Turbulence dynamic_Cs=true requires scheme=smagorinsky");
    }
    if (selected_config.dynamic_Cs && selected_config.mode != "3d")
    {
        throw std::invalid_argument(
            "Dynamic Smagorinsky requires turbulence.mode=3d");
    }
    if (selected_config.dynamic_Cs &&
        selected_config.stability_correction != "none")
    {
        throw std::invalid_argument(
            "Dynamic Smagorinsky cannot be combined with a separate stability correction");
    }
    if (selected_config.scheme_id == "smagorinsky" &&
        selected_config.stability_correction == "tke_based")
    {
        throw std::invalid_argument(
            "TKE-based stability correction requires scheme=tke");
    }
    if (selected_config.scheme_id == "tke" &&
        (selected_config.stability_correction == "ri" || selected_config.stability_correction == kLillyRichardsonModel))
    {
        throw std::invalid_argument(
            "Richardson stability correction is not a TKE closure selector");
    }

    std::unique_ptr<TurbulenceSchemeBase> selected_scheme =
        create_turbulence_scheme(selected_config.scheme_id);
    selected_scheme->initialize(selected_config);

    scheme = std::move(selected_scheme);
    config = std::move(selected_config);
    last_update_time_s = -config.dt_sgs;
    next_update_time_s = 0.0;
    held_tendencies = {};
    tmv::log_info("Initialized turbulence scheme: ", config.scheme_id);
    tmv::log_info("  Maximum SGS substep: ", config.dt_sgs, " s");
    tmv::log_info("  Mode = ", config.mode);
    tmv::log_info("  Cs = ", config.Cs);
    if (config.filter_width == "user")
        tmv::log_info("  Filter width = user (",
                      config.user_filter_width_m, " m)");
    else
        tmv::log_info("  Filter width = ", config.filter_width);
}
}

void initialize_turbulence(const std::string& scheme_name,
                           const TurbulenceConfig& cfg)
{
    return initialize_turbulence(global_simulation_state(), scheme_name, cfg);
}

void initialize_turbulence(const SimulationStateView& execution_state, const std::string& scheme_name,
                           const TurbulenceConfig& cfg)
{
    try
    {
        initialize_turbulence_runtime(tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").scheme,
                                      tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").config,
                                      tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").last_update_time_s,
                                      tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").next_update_time_s,
                                      tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").held_tendencies,
                                      scheme_name, cfg);
        tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").last_explicit_stability_limit_s =
            std::numeric_limits<double>::infinity();
        tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").last_explicit_substep_count = 0;
        tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").held_diagnostics = {};
        tmv::require_level_runtime(execution_state.turbulence_runtime, "turbulence_runtime").candidate_diagnostics = {};
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error initializing turbulence: ", e.what());
        throw;
    }
}

void initialize_turbulence(TurbulenceRuntimeState& runtime,
                           const std::string& scheme_name,
                           const TurbulenceConfig& cfg)
{
    try
    {
        initialize_turbulence_runtime(runtime.scheme, runtime.config,
                                      runtime.last_update_time_s,
                                      runtime.next_update_time_s,
                                      runtime.held_tendencies,
                                      scheme_name, cfg);
        runtime.last_explicit_stability_limit_s =
            std::numeric_limits<double>::infinity();
        runtime.last_explicit_substep_count = 0;
        runtime.held_diagnostics = {};
        runtime.candidate_diagnostics = {};
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error initializing turbulence: ", e.what());
        throw;
    }
}

bool step_turbulence(const SimulationStateView& state,
                     double current_time,
                     TurbulenceTendencies& tendencies)
{
    TurbulenceRuntimeState* const local_runtime = state.turbulence_runtime;
    TurbulenceRuntimeState& active_runtime = local_runtime
        ? *local_runtime : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime");
    std::unique_ptr<TurbulenceSchemeBase>& active_scheme =
        local_runtime ? local_runtime->scheme : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime").scheme;
    TurbulenceConfig& active_config =
        local_runtime ? local_runtime->config : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime").config;
    double& active_last_update_time =
        local_runtime ? local_runtime->last_update_time_s
                      : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime").last_update_time_s;
    double& active_next_update_time = active_runtime.next_update_time_s;

    if (!active_scheme)
    {
        throw std::runtime_error(
            "step_turbulence: turbulence scheme is not initialized");
    }
    if (!std::isfinite(current_time) || current_time < 0.0 ||
        !std::isfinite(active_last_update_time) ||
        !std::isfinite(active_next_update_time) ||
        !std::isfinite(active_config.dt_sgs) || active_config.dt_sgs <= 0.0)
    {
        throw std::invalid_argument(
            "step_turbulence: time/cadence state must be finite and cadence positive");
    }
    if (current_time < active_last_update_time)
    {
        throw std::invalid_argument(
            "step_turbulence: simulation time moved backward across the SGS cadence");
    }

    const double cadence_tolerance_s = 1.0e-12 * std::max(
        {1.0, std::abs(current_time), std::abs(active_next_update_time),
         active_config.dt_sgs});
    const std::string active_name = active_scheme->name();
    const bool nonlinear_explicit_scheme =
        active_name == "smagorinsky" || active_name == "tke";
    if (!nonlinear_explicit_scheme &&
        current_time + cadence_tolerance_s < active_next_update_time)
    {
        return false;
    }

    const double candidate_next_update_time = next_physics_deadline(
        nonlinear_explicit_scheme ? current_time : active_next_update_time,
        current_time, active_config.dt_sgs, cadence_tolerance_s);

    if (physical_grid_metrics::terrain_is_active(state))
    {
        physical_grid_metrics::validate(state);
        if (active_name != "none" && active_name != "smagorinsky" &&
            active_name != "tke")
        {
            throw std::invalid_argument(
                "the selected turbulence scheme is unavailable with terrain; "
                "select a closure with a mapped tensor-stress equation");
        }
    }
    TurbulenceStateView turbulence_state;
    bind_flow_execution(turbulence_state, state);
    turbulence_state.u = &state.fields.u;
    turbulence_state.v = &state.fields.v;
    turbulence_state.w = &state.fields.w;
    turbulence_state.rho = &state.fields.rho;
    turbulence_state.theta = &state.fields.theta;
    turbulence_state.qv = &state.fields.qv;
    turbulence_state.tke = &state.fields.tke;
    turbulence_state.rho0 = &state.base.rho0;
    turbulence_state.c_grid = state.grid.geometry.staggered;
    turbulence_state.require_native_dispatch =
        (active_name == "smagorinsky" || active_name == "tke") && state.configuration != nullptr &&
        state.configuration->execution_mode ==
            tmv::init::config::ExecutionMode::Qualified &&
        state.compute_runtime != nullptr &&
        state.compute_runtime->active_kind != ComputeBackendKind::Cpu &&
        !state.compute_runtime->fallback_active &&
        !state.compute_runtime->config.allow_fallback;
    if (state.chaos_runtime != nullptr
        && state.chaos_runtime->scheme != nullptr
        && state.chaos_runtime->config.scheme_id != "none")
    {
        const auto& blocks =
            state.chaos_runtime->config.apply_to_tendencies;
        turbulence_state.require_host_transport_coefficients =
            std::find(blocks.begin(), blocks.end(), "diffusion")
                != blocks.end();
        turbulence_state.defer_transport_tendency_build =
            turbulence_state.require_host_transport_coefficients;
    }
    turbulence_state.NR = state.grid.nr;
    turbulence_state.NTH = state.grid.nth;
    turbulence_state.NZ = state.grid.nz;

    const GridMetrics& grid = get_runtime_turbulence_grid(
        state, active_runtime.grid_metrics_scratch);
    // The selected scheme owns complete initialization of every output. A
    // pre-clear here duplicated the scheme and diffusion-writer passes.
    TurbulenceTendencies& candidate = active_runtime.candidate_tendencies;
    active_runtime.realized_transport_tendency_current = false;
    TurbulenceDiagnostics* candidate_diagnostics =
        active_config.enable_diagnostics
            ? &active_runtime.candidate_diagnostics
            : nullptr;
    active_scheme->compute(
        active_config, grid, turbulence_state, candidate,
        candidate_diagnostics);
    // One interleaved validation traversal retains the transactional contract
    // while avoiding six complete loop/control passes over equally shaped
    // output fields.
    require_finite_tendencies(candidate, state);
    if (candidate_diagnostics != nullptr)
        require_finite_diagnostics(*candidate_diagnostics, state);

    std::swap(active_runtime.held_tendencies, candidate);
    if (candidate_diagnostics != nullptr)
        std::swap(active_runtime.held_diagnostics,
                  active_runtime.candidate_diagnostics);
    if (&tendencies != &active_runtime.held_tendencies)
    {
        tendencies = active_runtime.held_tendencies;
    }
    active_runtime.realized_transport_tendency_current = true;
    active_runtime.stochastic_transport_applied_to_current = false;
    active_last_update_time = current_time;
    active_next_update_time = candidate_next_update_time;
    return true;
}

bool step_turbulence(double current_time, TurbulenceTendencies& tendencies)
{
    return step_turbulence(::global_simulation_state(), current_time, tendencies);
}

void refresh_turbulence(const SimulationStateView& state,
                        double current_time,
                        TurbulenceTendencies& tendencies)
{
    TurbulenceRuntimeState& runtime = state.turbulence_runtime
        ? *state.turbulence_runtime : tmv::require_level_runtime(state.turbulence_runtime, "turbulence_runtime");
    const double saved_next_update_time = runtime.next_update_time_s;
    runtime.next_update_time_s = current_time;
    try
    {
        if (!step_turbulence(state, current_time, tendencies))
        {
            throw std::runtime_error(
                "refresh_turbulence failed to service an immediate SGS refresh");
        }
    }
    catch (...)
    {
        // step_turbulence publishes its state only after a valid candidate.
        // Restore the cadence deadline as well so a failed forced refresh is
        // fully transactional.
        runtime.next_update_time_s = saved_next_update_time;
        throw;
    }
}

void record_turbulence_perf(TurbulencePerfStage stage, double elapsed_s)
{
    if (!std::isfinite(elapsed_s) || elapsed_s < 0.0)
        return;
    switch (stage)
    {
        case TurbulencePerfStage::SmagorinskyCoefficients:
            turbulence_perf_totals.smagorinsky_coefficients_s += elapsed_s;
            break;
        case TurbulencePerfStage::SmagorinskyMomentum:
            turbulence_perf_totals.smagorinsky_momentum_s += elapsed_s;
            break;
        case TurbulencePerfStage::SmagorinskyTheta:
            turbulence_perf_totals.smagorinsky_theta_s += elapsed_s;
            break;
        case TurbulencePerfStage::SmagorinskyMoisture:
            turbulence_perf_totals.smagorinsky_moisture_s += elapsed_s;
            break;
    }
}

void reset_turbulence_perf_stats()
{
    turbulence_perf_totals = TurbulencePerfTotals{};
}

void log_turbulence_perf_summary()
{
    const double profiled_s =
        turbulence_perf_totals.smagorinsky_coefficients_s
        + turbulence_perf_totals.smagorinsky_momentum_s
        + turbulence_perf_totals.smagorinsky_theta_s
        + turbulence_perf_totals.smagorinsky_moisture_s;
    tmv::log_info(
        "[TURBULENCE PERF] smagorinsky_coefficients_s=",
        turbulence_perf_totals.smagorinsky_coefficients_s,
        ", smagorinsky_momentum_s=",
        turbulence_perf_totals.smagorinsky_momentum_s,
        ", smagorinsky_theta_s=",
        turbulence_perf_totals.smagorinsky_theta_s,
        ", smagorinsky_moisture_s=",
        turbulence_perf_totals.smagorinsky_moisture_s,
        ", smagorinsky_profiled_s=", profiled_s);
}

} // namespace tmv
