#include "util/physics_cadence.hpp"
// Radiation runtime initialization and column stepping.

#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#ifdef NIMBUS_ALLOCATION_TELEMETRY
#include "core/infra/allocation_telemetry.hpp"
#endif
#include "core/runtime/simulation_state.hpp"
#include "compute/backend_runtime.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "radiation/runtime.hpp"
#include "radiation/factory.hpp"
#include "radiation/column_evaluation.hpp"
#include "radiation/base/radiative_transfer.hpp"
#include "radiation/native_batch_runtime.hpp"
#include "radiation/schemes/simple_grey/simple_grey.hpp"
#include "util/log.hpp"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
#ifdef NIMBUS_ALLOCATION_TELEMETRY
void record_field_resize_if_needed(const Field3D& field, int nr, int nth, int nz)
{
    const std::size_t elements =
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth) * static_cast<std::size_t>(nz);
    if (elements <= field.capacity())
        return;
    const std::size_t     raw_bytes       = elements * sizeof(float);
    constexpr std::size_t alignment       = PageAlignedAllocator<float>::kAlignmentBytes;
    const std::size_t     allocated_bytes = ((raw_bytes + alignment - 1) / alignment) * alignment;
    tmv::allocation_telemetry::record_physics_field_allocation(allocated_bytes > 0 ? allocated_bytes : alignment);
}
#endif

void ensure_tendency_shape(const SimulationStateView& state, Field3D& field)
{
    if (field.size_r() != state.grid.nr || field.size_th() != state.grid.nth || field.size_z() != state.grid.nz)
    {
        field.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    }
}

bool matches_grid(const SimulationStateView& state, const Field3D& field)
{
    return field.size_r() == state.grid.nr && field.size_th() == state.grid.nth && field.size_z() == state.grid.nz;
}

double radiation_update_interval(const tmv::RadiationConfig& config)
{
    const std::size_t divisor = config.use_subcycling
        ? config.subcycle_count : 1;
    const double interval = config.dt_radiation
        / static_cast<double>(divisor);
    tmv::require_physics_cadence(interval, "radiation subcycle interval");
    return interval;
}

void resize_flux_diagnostics(tmv::RadiationFluxDiagnostics& diagnostics,
                             std::size_t columns_x,
                             std::size_t columns_y,
                             std::size_t levels)
{
    diagnostics.columns_x = columns_x;
    diagnostics.columns_y = columns_y;
    diagnostics.levels = levels;
    const std::size_t count = columns_x * columns_y * (levels + 1);
    diagnostics.upward_longwave_wm2.resize(count);
    diagnostics.downward_longwave_wm2.resize(count);
    diagnostics.upward_shortwave_wm2.resize(count);
    diagnostics.downward_shortwave_wm2.resize(count);
}

void clear_flux_diagnostics(tmv::RadiationFluxDiagnostics& diagnostics)
{
    diagnostics = {};
}

} // namespace

namespace tmv
{

std::unique_ptr<RadiationSchemeBase>& radiation_scheme        = global_radiation_runtime_state.scheme;
RadiationConfig&                      global_radiation_config = global_radiation_runtime_state.config;

namespace
{
void initialize_radiation_runtime(std::unique_ptr<RadiationSchemeBase>& scheme,
                                  RadiationConfig&                      config,
                                  double&                               last_update_time_s,
                                  double&                               next_update_time_s,
                                  const std::string&                    scheme_name,
                                  const RadiationConfig&                cfg)
{
    // Construct first so an unknown or recognized-but-unimplemented scheme
    // cannot partially mutate the selected runtime.
    std::unique_ptr<RadiationSchemeBase> selected_scheme = create_radiation_scheme(scheme_name);
    const std::string                    resolved_scheme = canonicalize_radiation_scheme_id(scheme_name);

    RadiationConfig selected_config = cfg;
    selected_config.scheme_id       = resolved_scheme;
    if (!std::isfinite(selected_config.dt_radiation) || selected_config.dt_radiation <= 0.0)
    {
        throw std::invalid_argument("radiation cadence must be positive and finite");
    }

    if (resolved_scheme == "simple_grey")
    {
        if (!std::isfinite(selected_config.tau_lw_ref) || selected_config.tau_lw_ref < 0.0 ||
            !std::isfinite(selected_config.tau_sw_ref) || selected_config.tau_sw_ref < 0.0 ||
            !std::isfinite(selected_config.n_lw) || selected_config.n_lw <= 0.0 ||
            !std::isfinite(selected_config.n_sw) || selected_config.n_sw <= 0.0)
        {
            throw std::invalid_argument("simple_grey requires positive finite cadence/exponents and "
                                        "finite non-negative optical depths");
        }
        if (!selected_config.do_lw && !selected_config.do_sw)
            throw std::invalid_argument("simple_grey requires at least one enabled radiative band");
        const bool clear_model = selected_config.simple_grey_optical_model ==
            "pressure_clear_sky_v1";
        const bool water_cloud_model = selected_config.simple_grey_optical_model ==
            "absorbing_water_cloud_path_v1";
        if ((!clear_model && !water_cloud_model) ||
            (clear_model && !selected_config.clear_sky_only) ||
            (water_cloud_model && selected_config.clear_sky_only))
            throw std::invalid_argument(
                "simple_grey requires a consistent versioned optical model selection");
        if ((!selected_config.use_subcycling &&
             selected_config.subcycle_count != 1) ||
            (selected_config.use_subcycling &&
             (selected_config.subcycle_count < 2 ||
              selected_config.subcycle_count > 64)))
            throw std::invalid_argument(
                "simple_grey subcycle_count must be one when disabled or [2,64] when enabled");
        // Validate the explicit forcing contract at initialization as well as
        // at each sample time. Noon avoids a special sunrise/night result.
        (void)idealized_solar_cosine_zenith(selected_config, 12.0 * 3600.0);
        if (selected_config.terrain_sw_cosine_nodes < 4 || selected_config.terrain_sw_cosine_nodes > 32 ||
            selected_config.terrain_sw_azimuth_nodes < 8 || selected_config.terrain_sw_azimuth_nodes > 128 ||
            selected_config.terrain_sw_azimuth_nodes % 4 != 0 || selected_config.terrain_sw_maximum_bounces == 0 ||
            selected_config.terrain_sw_maximum_bounces > 256 ||
            !std::isfinite(selected_config.terrain_sw_relative_residual_tolerance) ||
            selected_config.terrain_sw_relative_residual_tolerance < 0.0 ||
            selected_config.terrain_sw_relative_residual_tolerance > 1.0e-6 ||
            !std::isfinite(selected_config.terrain_sw_absolute_residual_tolerance_w) ||
            selected_config.terrain_sw_absolute_residual_tolerance_w < 0.0)
        {
            throw std::invalid_argument("simple_grey terrain shortwave quadrature, bounce count, and "
                                        "residual tolerances must be positive/finite");
        }
    }

    selected_scheme->initialize(selected_config);
    scheme             = std::move(selected_scheme);
    config             = std::move(selected_config);
    last_update_time_s = -radiation_update_interval(config);
    next_update_time_s = 0.0;
    tmv::log_info("Initialized radiation scheme: ", config.scheme_id);
}

void require_supported_radiation_geometry(const SimulationStateView& state,
                                          const RadiationSchemeBase& scheme,
                                          const RadiationConfig& config)
{
    if (scheme.name() != "none" && physical_grid_metrics::terrain_is_active(state))
    {
        // The clear-sky longwave column is expressed as vertical flux per
        // horizontal area. For a Lambertian terrain surface, actual area
        // multiplies emission by sqrt(1+|grad h|^2) while the vertical normal
        // projection divides it by the same factor, so the lower boundary is
        // exactly epsilon*sigma*T^4 per horizontal area. Physical interface
        // heights and J-volume storage are already consumed below. Shortwave
        // Mapped shortwave uses the complete SCI-274 direct, sloped-surface,
        // atmospheric, and iterative reflection operator below.
        if (scheme.name() == "rrtmg")
        {
            return;
        }
        if (scheme.name() != "simple_grey")
        {
            throw std::invalid_argument("terrain radiation currently supports only the vertical "
                                        "simple-grey operators and RRTMG longwave");
        }
        if (config.enable_diagnostics && config.do_sw)
            throw std::invalid_argument(
                "mapped shortwave does not implement retained interface diagnostics");
    }
}
} // namespace

void initialize_radiation(const std::string& scheme_name, const RadiationConfig& cfg)
{
    return initialize_radiation(global_simulation_state(), scheme_name, cfg);
}

void initialize_radiation(const SimulationStateView& execution_state, const std::string& scheme_name, const RadiationConfig& cfg)
{
    try
    {
        initialize_radiation_runtime(tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").scheme,
                                     tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").config,
                                     tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").last_update_time_s,
                                     tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").next_update_time_s,
                                     scheme_name,
                                     cfg);
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_atmospheric_power_w = 0.0;
        clear_flux_diagnostics(tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_flux_diagnostics);
        clear_flux_diagnostics(tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").scratch_flux_diagnostics);
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_cell_power_w.release_storage();
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").scratch_theta_tendency.release_storage();
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").scratch_cell_power_w.release_storage();
        std::vector<double>().swap(tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").scratch_column_power_w);
        std::vector<unsigned char>().swap(tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").scratch_column_errors);
        std::vector<RadiationWorkerContext>().swap(tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").worker_contexts);
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").terrain_shortwave_workspace.reset();
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").batch_workspace.reset();
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").last_expected_atmospheric_energy_j          = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").last_applied_radiation_heat_j               = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").accumulated_expected_atmospheric_energy_j   = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").accumulated_applied_radiation_heat_j        = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").applied_step_count                          = 0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_shortwave_incoming_power_w             = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_shortwave_atmospheric_absorbed_power_w = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_surface_absorbed_power_w               = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_top_escape_power_w                     = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_lateral_escape_power_w                 = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_unresolved_reflected_power_w           = 0.0;
        tmv::require_level_runtime(execution_state.radiation_runtime, "radiation_runtime").held_reflection_converged                   = true;
        const auto state                                                           = execution_state;
        state.tendencies.dtheta_dt_rad.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error initializing radiation: ", e.what());
        throw;
    }
}

void initialize_radiation(RadiationRuntimeState& runtime, const std::string& scheme_name, const RadiationConfig& cfg)
{
    try
    {
        initialize_radiation_runtime(
            runtime.scheme, runtime.config, runtime.last_update_time_s, runtime.next_update_time_s, scheme_name, cfg);
        runtime.held_atmospheric_power_w = 0.0;
        clear_flux_diagnostics(runtime.held_flux_diagnostics);
        clear_flux_diagnostics(runtime.scratch_flux_diagnostics);
        runtime.held_cell_power_w.release_storage();
        runtime.scratch_theta_tendency.release_storage();
        runtime.scratch_cell_power_w.release_storage();
        std::vector<double>().swap(runtime.scratch_column_power_w);
        std::vector<unsigned char>().swap(runtime.scratch_column_errors);
        std::vector<RadiationWorkerContext>().swap(runtime.worker_contexts);
        runtime.terrain_shortwave_workspace.reset();
        runtime.batch_workspace.reset();
        runtime.last_expected_atmospheric_energy_j          = 0.0;
        runtime.last_applied_radiation_heat_j               = 0.0;
        runtime.accumulated_expected_atmospheric_energy_j   = 0.0;
        runtime.accumulated_applied_radiation_heat_j        = 0.0;
        runtime.applied_step_count                          = 0;
        runtime.held_shortwave_incoming_power_w             = 0.0;
        runtime.held_shortwave_atmospheric_absorbed_power_w = 0.0;
        runtime.held_surface_absorbed_power_w               = 0.0;
        runtime.held_top_escape_power_w                     = 0.0;
        runtime.held_lateral_escape_power_w                 = 0.0;
        runtime.held_unresolved_reflected_power_w           = 0.0;
        runtime.held_reflection_converged                   = true;
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error initializing radiation: ", e.what());
        throw;
    }
}

void step_radiation(const SimulationStateView& state, double current_time)
{
    RadiationRuntimeState* const          local_runtime = state.radiation_runtime;
    std::unique_ptr<RadiationSchemeBase>& active_scheme = local_runtime ? local_runtime->scheme : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime").scheme;
    RadiationConfig& active_config = local_runtime ? local_runtime->config : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime").config;
    double&          active_last_update_time =
        local_runtime ? local_runtime->last_update_time_s : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime").last_update_time_s;
    double& active_next_update_time =
        local_runtime ? local_runtime->next_update_time_s : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime").next_update_time_s;
    const SurfaceConfig& active_surface_config =
        state.boundary_layer_runtime ? state.boundary_layer_runtime->surface_config : tmv::require_level_runtime(state.boundary_layer_runtime, "boundary_layer_runtime").surface_config;
    RadiationRuntimeState& active_runtime = local_runtime ? *local_runtime : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime");

    if (!active_scheme)
    {
        throw std::runtime_error("step_radiation: radiation scheme is not initialized");
    }
    const auto& grid            = state.grid;
    const auto& fields          = state.fields;
    auto&       tendency_fields = state.tendencies;
    if (!std::isfinite(current_time) || current_time < 0.0 || !std::isfinite(active_last_update_time) ||
        !std::isfinite(active_next_update_time) || !std::isfinite(active_config.dt_radiation) ||
        active_config.dt_radiation <= 0.0)
    {
        throw std::invalid_argument("step_radiation: time and cadence must be finite and cadence positive");
    }
    if (current_time < active_last_update_time)
    {
        throw std::invalid_argument("step_radiation: simulation time moved backward across the radiation cadence");
    }
    if (active_scheme->name() == "none")
    {
        ensure_tendency_shape(state, tendency_fields.dtheta_dt_rad);
        tendency_fields.dtheta_dt_rad.fill(0.0f);
        active_runtime.held_cell_power_w.release_storage();
        active_runtime.scratch_theta_tendency.release_storage();
        active_runtime.scratch_cell_power_w.release_storage();
        std::vector<double>().swap(active_runtime.scratch_column_power_w);
        std::vector<unsigned char>().swap(active_runtime.scratch_column_errors);
        std::vector<RadiationWorkerContext>().swap(active_runtime.worker_contexts);
        active_runtime.terrain_shortwave_workspace.reset();
        active_runtime.batch_workspace.reset();
        active_runtime.held_atmospheric_power_w                    = 0.0;
        clear_flux_diagnostics(active_runtime.held_flux_diagnostics);
        clear_flux_diagnostics(active_runtime.scratch_flux_diagnostics);
        active_runtime.held_shortwave_incoming_power_w             = 0.0;
        active_runtime.held_shortwave_atmospheric_absorbed_power_w = 0.0;
        active_runtime.held_surface_absorbed_power_w               = 0.0;
        active_runtime.held_top_escape_power_w                     = 0.0;
        active_runtime.held_lateral_escape_power_w                 = 0.0;
        active_runtime.held_unresolved_reflected_power_w           = 0.0;
        active_runtime.held_reflection_converged                   = true;
        active_last_update_time                                    = current_time;
        return;
    }
    const double candidate_next_update_time = next_physics_deadline(
        active_next_update_time, current_time, radiation_update_interval(active_config),
        1.0e-12 * std::max({1.0, std::abs(current_time), std::abs(active_next_update_time), active_config.dt_radiation}));
    require_supported_radiation_geometry(state, *active_scheme,
                                         active_config);
    if (grid.nz <= 0)
    {
        throw std::invalid_argument("step_radiation: grid.nz must be positive");
    }
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (domain.empty())
    {
        throw std::invalid_argument("step_radiation: physical-cell domain is empty");
    }
    const double cadence_tolerance_s =
        1.0e-12 *
        std::max({1.0, std::abs(current_time), std::abs(active_next_update_time), active_config.dt_radiation});
    if (current_time + cadence_tolerance_s < active_next_update_time)
    {
        return;
    }
#ifdef NIMBUS_PHYSICS_LEGACY_ALLOCATION_EXPERIMENT
    // Measurement-only reconstruction of the pre-SCI-122/123 allocation
    // lifetime. Only replaceable scratch is discarded; committed state,
    // equations, loop order, cadence, and worker count remain unchanged.
    active_runtime.scratch_theta_tendency.release_storage();
    active_runtime.scratch_cell_power_w.release_storage();
    std::vector<double>().swap(active_runtime.scratch_column_power_w);
    std::vector<unsigned char>().swap(active_runtime.scratch_column_errors);
    std::vector<RadiationWorkerContext>().swap(active_runtime.worker_contexts);
    active_runtime.batch_workspace.reset();
#endif
    if (!matches_grid(state, fields.p) || !matches_grid(state, fields.theta) || !matches_grid(state, fields.rho))
    {
        throw std::invalid_argument("step_radiation: required state field shape does not match the grid");
    }

    // Fail before entering the OpenMP column loop. A fabricated standard-
    // atmosphere temperature used to replace valid upper-atmosphere pressure
    // (<10 kPa), while malformed values were silently hidden by the same path.
    // Radiation must consume the resolved thermodynamic state exactly.
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double p_value     = fields.p[i][j][k];
                const double theta_value = fields.theta[i][j][k];
                const double rho_value   = fields.rho[i][j][k];
                if (!std::isfinite(p_value) || p_value <= 0.0 || !std::isfinite(theta_value) || theta_value <= 0.0 ||
                    !std::isfinite(rho_value) || rho_value <= 0.0)
                {
                    throw std::invalid_argument(
                        "step_radiation: pressure, theta, and density must be positive and finite");
                }
                if (k > domain.k_begin && p_value > static_cast<double>(fields.p[i][j][k - 1]))
                {
                    throw std::invalid_argument("step_radiation: pressure must be non-increasing with height");
                }
            }
        }
    }

    const IdealizedSolarGeometry solar_geometry = idealized_solar_geometry(active_config, current_time);
    const double                 mu0_idealized  = solar_geometry.cosine_zenith();
    const double                 kappa          = R_d / cp;

    const std::size_t physical_nz = static_cast<std::size_t>(domain.nz());
    // Validate every column before OpenMP. Active terrain must supply its
    // physical scalar and interface heights; a malformed terrain field must
    // never fall back to the flat grid geometry.
    physical_grid_metrics::validate(state);
    if (!std::isfinite(active_surface_config.albedo) || active_surface_config.albedo < 0.0 ||
        active_surface_config.albedo > 1.0 || !std::isfinite(active_surface_config.emissivity) ||
        active_surface_config.emissivity < 0.0 || active_surface_config.emissivity > 1.0 ||
        (active_config.do_lw && (!std::isfinite(active_surface_config.Tsfc) || active_surface_config.Tsfc < 150.0 ||
                                 active_surface_config.Tsfc > 400.0)))
    {
        throw std::invalid_argument("step_radiation: radiation requires finite surface albedo/emissivity "
                                    "in [0,1], and active longwave requires an explicit surface "
                                    "temperature in [150,400] K");
    }
    const double    surface_albedo     = active_surface_config.albedo;
    const double    surface_emissivity = active_surface_config.emissivity;
    const bool      terrain_shortwave  = physical_grid_metrics::terrain_is_active(state) && active_config.do_sw;
    const bool      rrtmg_terrain_shortwave = terrain_shortwave &&
        active_scheme->name() == "rrtmg";
    RadiationConfig column_config      = active_config;
    const std::vector<double>* terrain_shortwave_absorbed_power_w = nullptr;
    double terrain_shortwave_incoming_power_w = 0.0;
    double terrain_shortwave_atmospheric_absorbed_power_w = 0.0;
    double terrain_shortwave_surface_absorbed_power_w = 0.0;
    double terrain_shortwave_top_escape_power_w = 0.0;
    double terrain_shortwave_lateral_escape_power_w = 0.0;
    double terrain_shortwave_unresolved_power_w = 0.0;
    bool terrain_shortwave_converged = true;
    if (terrain_shortwave)
        column_config.do_sw = false;

    Field3D& candidate_tendency     = active_runtime.scratch_theta_tendency;
    Field3D& candidate_cell_power_w = active_runtime.scratch_cell_power_w;
    // resize(..., value) reuses capacity for an unchanged model shape and
    // clears every closure/physical cell before the parallel column update.
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    record_field_resize_if_needed(candidate_tendency, state.grid.nr, state.grid.nth, state.grid.nz);
    record_field_resize_if_needed(candidate_cell_power_w, state.grid.nr, state.grid.nth, state.grid.nz);
#endif
    candidate_tendency.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    candidate_cell_power_w.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    int                  failed_columns           = 0;
    const std::size_t    storage_column_count     = static_cast<std::size_t>(grid.nr) * grid.nth;
    std::vector<double>& candidate_column_power_w = active_runtime.scratch_column_power_w;
    candidate_column_power_w.resize(storage_column_count);
    std::fill(candidate_column_power_w.begin(), candidate_column_power_w.end(), 0.0);
    std::vector<unsigned char>& candidate_column_errors = active_runtime.scratch_column_errors;
    candidate_column_errors.resize(storage_column_count);
    std::fill(candidate_column_errors.begin(), candidate_column_errors.end(), 0);
    std::vector<std::string> candidate_column_messages(storage_column_count);

    const double horizontal_cell_area = grid.dr * grid.dtheta;
    if (!std::isfinite(horizontal_cell_area) || horizontal_cell_area <= 0.0)
    {
        throw std::invalid_argument("step_radiation: horizontal cell area must be positive and finite");
    }

    const std::size_t physical_column_count = static_cast<std::size_t>(domain.nx()) * domain.ny();
    const auto        field_matches_grid    = [&grid](const Field3D& field)
    { return field.size_r() == grid.nr && field.size_th() == grid.nth && field.size_z() == grid.nz; };
    const bool                        has_vapor         = field_matches_grid(fields.qv);
    const bool                        has_cloud_liquid  = field_matches_grid(fields.qc);
    const bool                        has_cloud_ice     = field_matches_grid(fields.qi);
    if (!active_config.clear_sky_only &&
        (!has_vapor || !has_cloud_liquid || !has_cloud_ice))
        throw std::invalid_argument(
            "absorbing water/cloud-path simple_grey requires grid-shaped qv, qc, and qi fields");
    RadiationFluxDiagnostics& candidate_flux_diagnostics =
        active_runtime.scratch_flux_diagnostics;
    if (active_config.enable_diagnostics)
        resize_flux_diagnostics(candidate_flux_diagnostics,
                                static_cast<std::size_t>(domain.nx()),
                                static_cast<std::size_t>(domain.ny()),
                                physical_nz);
    else
        clear_flux_diagnostics(candidate_flux_diagnostics);
    TerrainShortwaveRuntimeWorkspace* terrain_workspace = nullptr;
    if (terrain_shortwave)
    {
        if (!active_runtime.terrain_shortwave_workspace)
            active_runtime.terrain_shortwave_workspace = std::make_shared<TerrainShortwaveRuntimeWorkspace>();
        terrain_workspace = active_runtime.terrain_shortwave_workspace.get();
        terrain_workspace->interface_height_m.resize(physical_column_count * (physical_nz + 1));
        terrain_workspace->vertical_optical_depth.resize(physical_column_count * physical_nz);
        terrain_workspace->slope_x.resize(physical_column_count);
        terrain_workspace->slope_y.resize(physical_column_count);
        if (rrtmg_terrain_shortwave)
        {
            constexpr std::size_t g_points = 112;
            if (physical_column_count >
                    std::numeric_limits<std::size_t>::max() / physical_nz ||
                physical_column_count * physical_nz >
                    std::numeric_limits<std::size_t>::max() / g_points)
                throw std::overflow_error(
                    "mapped RRTMG shortwave spectral storage overflowed");
            const std::size_t spectral_cells =
                physical_column_count * physical_nz * g_points;
            terrain_workspace->spectral_absorption_optical_depth.resize(
                spectral_cells);
            terrain_workspace->spectral_scattering_optical_depth.resize(
                spectral_cells);
            terrain_workspace->spectral_scattering_asymmetry.resize(
                spectral_cells);
            terrain_workspace->spectral_solar_source_fraction.resize(
                physical_column_count * g_points);
        }

        const TerrainMetrics3D* const metrics = physical_grid_metrics::active_terrain_metrics(state);
        if (metrics == nullptr || metrics->Gx_face.size_r() != grid.nr || metrics->Gx_face.size_th() != grid.nth ||
            metrics->Gx_face.size_z() != grid.nz + 1 || metrics->Gy_face.size_r() != grid.nr ||
            metrics->Gy_face.size_th() != grid.nth || metrics->Gy_face.size_z() != grid.nz + 1)
        {
            throw std::invalid_argument("terrain shortwave requires complete bottom-face slope metrics");
        }
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const std::size_t column =
                    static_cast<std::size_t>(i - domain.i_begin) * static_cast<std::size_t>(domain.ny()) +
                    static_cast<std::size_t>(j - domain.j_begin);
                const double slope_x = -metrics->Gx_face(i, j, domain.k_begin);
                const double slope_y = -metrics->Gy_face(i, j, domain.k_begin);
                if (!std::isfinite(slope_x) || !std::isfinite(slope_y))
                    throw std::invalid_argument("terrain shortwave slopes must be finite");
                terrain_workspace->slope_x[column] = slope_x;
                terrain_workspace->slope_y[column] = slope_y;
            }
    }
    bool                                    native_batch_completed = false;
    const ComputeBackendRuntimeState* const compute                = state.compute_runtime;
    const bool active_accelerator = compute != nullptr && compute->backend != nullptr &&
                                    compute->active_kind != ComputeBackendKind::Cpu && !compute->fallback_active;
    if (rrtmg_terrain_shortwave && active_accelerator)
        throw std::runtime_error(
            "mapped RRTMG shortwave requires a native spectral terrain owner on the selected accelerator");
    if (active_accelerator && (active_scheme->name() == "rrtmg" || active_scheme->name() == "simple_grey"))
    {
        RadiationComputeKernels* const kernels = compute->backend->radiation_kernels();
        const bool                     supported =
            kernels != nullptr && active_scheme->supports_native_column_batch(column_config, *kernels);
        if (supported)
        {
            const RadiationNativeBatchRequest request{physical_nz,
                                                      mu0_idealized,
                                                      active_surface_config.Tsfc,
                                                      surface_albedo,
                                                      surface_emissivity,
                                                      horizontal_cell_area,
                                                      has_vapor,
                                                      has_cloud_liquid,
                                                      has_cloud_ice,
                                                      terrain_shortwave,
                                                      terrain_workspace};
            RadiationBatchPublication publication{candidate_tendency, candidate_cell_power_w, candidate_column_power_w, active_config.enable_diagnostics ? &candidate_flux_diagnostics : nullptr};
            native_batch_completed = evaluate_native_radiation_batches(
                state, active_runtime, *active_scheme, column_config, domain, *kernels, request, publication);
        }
        if (!native_batch_completed && !compute->config.allow_fallback)
            throw std::runtime_error(
                supported ? "native radiation batch dispatch was rejected"
                          : "active accelerator does not provide the required native radiation batch operations");
        if (!native_batch_completed)
            tmv::log_warn("[RADIATION] native radiation batch unavailable; using permitted CPU column fallback");
    }

    if (!native_batch_completed)
    {
        const bool parallel_columns = physical_column_count >= 64;
        int        worker_count     = 1;
#ifdef _OPENMP
        if (parallel_columns)
        {
            const int column_limit = static_cast<int>(
                std::min(physical_column_count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
            worker_count = std::max(1, std::min(omp_get_max_threads(), column_limit));
        }
#endif
        const std::size_t required_worker_count   = static_cast<std::size_t>(worker_count);
        bool              rebuild_worker_contexts = active_runtime.worker_contexts.size() != required_worker_count;
        if (!rebuild_worker_contexts)
        {
            for (const RadiationWorkerContext& context : active_runtime.worker_contexts)
            {
                if (!context.scheme || context.scheme->name() != active_scheme->name())
                {
                    rebuild_worker_contexts = true;
                    break;
                }
            }
        }
        if (rebuild_worker_contexts)
        {
            std::vector<RadiationWorkerContext> candidates(required_worker_count);
            for (RadiationWorkerContext& context : candidates)
            {
                context.scheme = active_scheme->clone();
                if (!context.scheme)
                {
                    throw std::runtime_error("radiation scheme does not support worker cloning: " +
                                             active_scheme->name());
                }
            }
            active_runtime.worker_contexts = std::move(candidates);
        }

#pragma omp parallel if (parallel_columns) num_threads(worker_count) reduction(+ : failed_columns)
        {
            int worker_index = 0;
#ifdef _OPENMP
            worker_index = omp_get_thread_num();
#endif
            RadiationWorkerContext&   context = active_runtime.worker_contexts[static_cast<std::size_t>(worker_index)];
            RadiationColumnWorkspace& workspace = context.workspace;
            workspace.p.resize(physical_nz);
            workspace.T.resize(physical_nz);
            workspace.rho.resize(physical_nz);
            if (has_vapor)
                workspace.qv.resize(physical_nz);
            if (has_cloud_liquid)
                workspace.qc.resize(physical_nz);
            if (has_cloud_ice)
                workspace.qi.resize(physical_nz);

#pragma omp for collapse(2) schedule(static)
            for (int i = domain.i_begin; i < domain.i_end; ++i)
            {
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                {
                    const std::size_t column_index = static_cast<std::size_t>(i) * grid.nth + j;
                    physical_grid_metrics::column_geometry(state,
                                                           i,
                                                           j,
                                                           domain.k_begin,
                                                           domain.k_end,
                                                           workspace.z_center,
                                                           workspace.z_interface,
                                                           workspace.layer_thickness);

                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const std::size_t kk = static_cast<std::size_t>(k - domain.k_begin);
                        workspace.p[kk]      = static_cast<double>(fields.p[i][j][k]);
                    }
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const std::size_t kk        = static_cast<std::size_t>(k - domain.k_begin);
                        double            theta_val = fields.theta[i][j][k];
                        double            p_val     = fields.p[i][j][k];

                        workspace.T[kk] = theta_val * std::pow(p_val / p0, kappa);
                    }
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const std::size_t kk = static_cast<std::size_t>(k - domain.k_begin);
                        workspace.rho[kk]    = static_cast<double>(fields.rho[i][j][k]);
                        if (has_vapor)
                            workspace.qv[kk] = static_cast<double>(fields.qv[i][j][k]);
                        if (has_cloud_liquid)
                            workspace.qc[kk] = static_cast<double>(fields.qc[i][j][k]);
                        if (has_cloud_ice)
                            workspace.qi[kk] = static_cast<double>(fields.qi[i][j][k]);
                    }
                    RadiationColumnTendencies& tend = context.tendencies;
                    RadiationColumnFluxes* fluxes = active_config.enable_diagnostics
                        ? &context.fluxes : nullptr;
                    try
                    {
                        RadiationPreparedColumnView prepared;
                        prepared.pressure_pa                   = &workspace.p;
                        prepared.temperature_k                 = &workspace.T;
                        prepared.density_kgm3                  = &workspace.rho;
                        prepared.vapor_mixing_ratio            = has_vapor ? &workspace.qv : nullptr;
                        prepared.cloud_liquid_mixing_ratio     = has_cloud_liquid ? &workspace.qc : nullptr;
                        prepared.cloud_ice_mixing_ratio        = has_cloud_ice ? &workspace.qi : nullptr;
                        prepared.z_center_m                    = &workspace.z_center;
                        prepared.z_interface_m                 = &workspace.z_interface;
                        prepared.layer_thickness_m             = &workspace.layer_thickness;
                        prepared.cosine_solar_zenith           = mu0_idealized;
                        prepared.surface_temperature_k         = active_surface_config.Tsfc;
                        prepared.surface_albedo                = surface_albedo;
                        prepared.surface_emissivity            = surface_emissivity;
                        prepared.horizontal_cell_area_m2       = horizontal_cell_area;
                        candidate_column_power_w[column_index] = evaluate_prepared_radiation_column(
                            *context.scheme, column_config, prepared, workspace, tend,
                            fluxes);
                        if (rrtmg_terrain_shortwave)
                        {
                            RadiationColumnStateView spectral_state;
                            spectral_state.p = &workspace.p;
                            spectral_state.p_int = &workspace.p_interface;
                            spectral_state.T = &workspace.T;
                            spectral_state.rho = &workspace.rho;
                            spectral_state.dz = &workspace.layer_thickness;
                            spectral_state.z_int = &workspace.z_interface;
                            spectral_state.qv = has_vapor ? &workspace.qv : nullptr;
                            spectral_state.qc = has_cloud_liquid ? &workspace.qc : nullptr;
                            spectral_state.qi = has_cloud_ice ? &workspace.qi : nullptr;
                            spectral_state.mu0 = mu0_idealized;
                            spectral_state.S0 = physical_constants::solar_constant_wm2;
                            spectral_state.Tsfc = active_surface_config.Tsfc;
                            spectral_state.albedo_sw = surface_albedo;
                            spectral_state.emissivity_lw = surface_emissivity;
                            spectral_state.workspace = &workspace;
                            auto& spectral = context.mapped_shortwave_optics;
                            if (!context.scheme->compute_mapped_shortwave_optics(
                                    active_config, spectral_state, spectral) ||
                                spectral.layers != physical_nz ||
                                spectral.g_points != 112)
                                throw std::runtime_error(
                                    "RRTMG did not publish complete mapped spectral optics");
                            const std::size_t physical_column =
                                static_cast<std::size_t>(i - domain.i_begin) *
                                    static_cast<std::size_t>(domain.ny()) +
                                static_cast<std::size_t>(j - domain.j_begin);
                            const std::size_t spectral_offset =
                                physical_column * physical_nz * 112;
                            std::copy(
                                spectral.absorption_optical_depth.begin(),
                                spectral.absorption_optical_depth.end(),
                                terrain_workspace->spectral_absorption_optical_depth.begin() +
                                    spectral_offset);
                            std::copy(
                                spectral.scattering_optical_depth.begin(),
                                spectral.scattering_optical_depth.end(),
                                terrain_workspace->spectral_scattering_optical_depth.begin() +
                                    spectral_offset);
                            std::copy(
                                spectral.scattering_asymmetry.begin(),
                                spectral.scattering_asymmetry.end(),
                                terrain_workspace->spectral_scattering_asymmetry.begin() +
                                    spectral_offset);
                            std::copy(
                                spectral.solar_source_fraction.begin(),
                                spectral.solar_source_fraction.end(),
                                terrain_workspace->spectral_solar_source_fraction.begin() +
                                    physical_column * 112);
                        }
                        if (fluxes != nullptr)
                        {
                            const auto valid_flux = [physical_nz](
                                const std::vector<double>& values)
                            {
                                return values.size() == physical_nz + 1 &&
                                    std::all_of(values.begin(), values.end(),
                                                [](double value)
                                                { return std::isfinite(value); });
                            };
                            if (!valid_flux(fluxes->Fup_lw) ||
                                !valid_flux(fluxes->Fdn_lw) ||
                                !valid_flux(fluxes->Fup_sw) ||
                                !valid_flux(fluxes->Fdn_sw))
                                throw std::runtime_error(
                                    "radiation scheme returned malformed retained interface fluxes");
                            if (column_config.do_lw)
                            {
                                const double interface_power =
                                    -((fluxes->Fup_lw.back()
                                       - fluxes->Fdn_lw.back())
                                      - (fluxes->Fup_lw.front()
                                         - fluxes->Fdn_lw.front()));
                                const double power_scale = std::max(
                                    1.0, std::max(
                                        std::abs(interface_power),
                                        std::abs(tend.atmospheric_power_lw_wm2)));
                                if (std::abs(interface_power
                                             - tend.atmospheric_power_lw_wm2)
                                    > 1.0e-11 * power_scale)
                                    throw std::runtime_error(
                                        "retained longwave fluxes do not close the atmospheric tendency");
                            }
                            if (column_config.do_sw)
                            {
                                const double incoming =
                                    fluxes->Fdn_sw.back();
                                const double surface_absorbed =
                                    (1.0 - surface_albedo)
                                    * fluxes->Fdn_sw.front();
                                const double interface_absorbed =
                                    incoming - surface_absorbed
                                    - fluxes->Fup_sw.back();
                                const double power_scale = std::max(
                                    1.0, std::max(
                                        std::abs(interface_absorbed),
                                        std::abs(tend.atmospheric_power_sw_wm2)));
                                if (std::abs(interface_absorbed
                                             - tend.atmospheric_power_sw_wm2)
                                    > 1.0e-11 * power_scale)
                                    throw std::runtime_error(
                                        "retained shortwave fluxes do not close the atmospheric tendency");
                            }
                            const std::size_t physical_column =
                                static_cast<std::size_t>(i - domain.i_begin)
                                    * static_cast<std::size_t>(domain.ny())
                                + static_cast<std::size_t>(j - domain.j_begin);
                            const std::size_t offset =
                                physical_column * (physical_nz + 1);
                            std::copy(fluxes->Fup_lw.begin(), fluxes->Fup_lw.end(),
                                      candidate_flux_diagnostics.upward_longwave_wm2.begin() + offset);
                            std::copy(fluxes->Fdn_lw.begin(), fluxes->Fdn_lw.end(),
                                      candidate_flux_diagnostics.downward_longwave_wm2.begin() + offset);
                            std::copy(fluxes->Fup_sw.begin(), fluxes->Fup_sw.end(),
                                      candidate_flux_diagnostics.upward_shortwave_wm2.begin() + offset);
                            std::copy(fluxes->Fdn_sw.begin(), fluxes->Fdn_sw.end(),
                                      candidate_flux_diagnostics.downward_shortwave_wm2.begin() + offset);
                        }
                        if (terrain_shortwave)
                        {
                            const std::size_t physical_column =
                                static_cast<std::size_t>(i - domain.i_begin) * static_cast<std::size_t>(domain.ny()) +
                                static_cast<std::size_t>(j - domain.j_begin);
                            std::copy(workspace.z_interface.begin(),
                                      workspace.z_interface.end(),
                                      terrain_workspace->interface_height_m.begin() +
                                          physical_column * (physical_nz + 1));
                            if (!rrtmg_terrain_shortwave)
                            {
                                compute_simple_grey_shortwave_layer_optical_depths(
                                    active_config, workspace.p_interface,
                                    workspace.layer_thickness, workspace.rho,
                                    has_vapor ? &workspace.qv : nullptr,
                                    has_cloud_liquid ? &workspace.qc : nullptr,
                                    has_cloud_ice ? &workspace.qi : nullptr,
                                    workspace.tau_profile, workspace.tau_layers);
                                std::copy(workspace.tau_layers.begin(),
                                          workspace.tau_layers.end(),
                                          terrain_workspace->vertical_optical_depth.begin() +
                                              physical_column * physical_nz);
                            }
                        }
                    }
                    catch (const RadiationColumnEvaluationError& error)
                    {
                        ++failed_columns;
                        candidate_column_errors[column_index] = static_cast<unsigned char>(error.failure());
                        candidate_column_messages[column_index] = error.what();
                        continue;
                    }
                    catch (const std::exception& error)
                    {
                        ++failed_columns;
                        candidate_column_errors[column_index] = 1;
                        candidate_column_messages[column_index] = error.what();
                        continue;
                    }
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const size_t k_idx              = static_cast<size_t>(k - domain.k_begin);
                        candidate_tendency[i][j][k]     = workspace.theta_tendency[k_idx];
                        candidate_cell_power_w[i][j][k] = workspace.cell_power_w[k_idx];
                    }
                }
            }
        }
    }

    if (failed_columns > 0)
    {
        unsigned char first_column_error = 0;
        std::string first_column_detail;
        for (int i = domain.i_begin; i < domain.i_end && first_column_error == 0; ++i)
            for (int j = domain.j_begin; j < domain.j_end && first_column_error == 0; ++j)
            {
                const std::size_t column = static_cast<std::size_t>(i) * grid.nth + j;
                first_column_error = candidate_column_errors[column];
                if (first_column_error != 0)
                    first_column_detail = candidate_column_messages[column];
            }
        const char* first_column_message = "radiation column failed";
        switch (first_column_error)
        {
        case 1:
            first_column_message = "radiation scheme rejected the column state";
            break;
        case 2:
            first_column_message = "radiation scheme returned a malformed tendency column";
            break;
        case 3:
            first_column_message = "radiation tendency does not close its interface-flux power";
            break;
        case 4:
            first_column_message = "radiation scheme produced a non-finite tendency";
            break;
        case 5:
            first_column_message = "radiation scheme produced non-finite cell power";
            break;
        default:
            break;
        }
        throw std::runtime_error("radiation update rejected " + std::to_string(failed_columns) +
                                 " column(s): " + first_column_message +
                                 (first_column_detail.empty() ? "" : " (" + first_column_detail + ")"));
    }
    if (terrain_shortwave)
    {
        if (state.configuration == nullptr)
            throw std::invalid_argument("terrain shortwave requires per-simulation lateral boundary configuration");
        const auto map_axis_boundary = [](tmv::init::config::LateralBC lower, tmv::init::config::LateralBC upper)
        {
            const bool lower_periodic = lower == tmv::init::config::LateralBC::Periodic;
            const bool upper_periodic = upper == tmv::init::config::LateralBC::Periodic;
            if (lower_periodic != upper_periodic)
                throw std::invalid_argument("terrain shortwave requires periodicity on both faces of an axis");
            return lower_periodic ? radiative_transfer::TerrainShadowBoundary::Periodic
                                  : radiative_transfer::TerrainShadowBoundary::Open;
        };
        const auto x_boundary =
            map_axis_boundary(state.configuration->lateral_bc_west, state.configuration->lateral_bc_east);
        const auto y_boundary =
            map_axis_boundary(state.configuration->lateral_bc_south, state.configuration->lateral_bc_north);
        radiative_transfer::LambertianMultipleReflectionOptions options;
        options.quadrature      = {active_config.terrain_sw_cosine_nodes, active_config.terrain_sw_azimuth_nodes};
        options.albedo          = surface_albedo;
        options.maximum_bounces = active_config.terrain_sw_maximum_bounces;
        options.relative_residual_tolerance   = active_config.terrain_sw_relative_residual_tolerance;
        options.absolute_residual_tolerance_w = active_config.terrain_sw_absolute_residual_tolerance_w;
        if (rrtmg_terrain_shortwave)
        {
            const radiative_transfer::MappedSpectralShortwaveGridView mapped_grid{
                terrain_workspace->spectral_absorption_optical_depth.data(),
                terrain_workspace->spectral_scattering_optical_depth.data(),
                terrain_workspace->spectral_scattering_asymmetry.data(),
                terrain_workspace->spectral_solar_source_fraction.data(),
                terrain_workspace->interface_height_m.data(),
                static_cast<std::size_t>(domain.nx()),
                static_cast<std::size_t>(domain.ny()), physical_nz, 112,
                grid.dr, grid.dtheta, x_boundary, y_boundary};
            radiative_transfer::evaluate_mapped_spectral_terrain_shortwave_reference(
                mapped_grid,
                {solar_geometry.east, solar_geometry.north, solar_geometry.up},
                physical_constants::solar_constant_wm2,
                terrain_workspace->slope_x.data(),
                terrain_workspace->slope_y.data(), horizontal_cell_area,
                options, terrain_workspace->spectral_operator_workspace,
                terrain_workspace->spectral_operator_result);
            const auto& shortwave = terrain_workspace->spectral_operator_result;
            terrain_shortwave_absorbed_power_w =
                &shortwave.atmospheric_absorbed_power_w;
            terrain_shortwave_incoming_power_w = shortwave.incoming_power_w;
            terrain_shortwave_atmospheric_absorbed_power_w =
                shortwave.atmospheric_absorbed_power_w_total;
            terrain_shortwave_surface_absorbed_power_w =
                shortwave.surface_absorbed_power_w_total;
            terrain_shortwave_top_escape_power_w = shortwave.top_escape_power_w;
            terrain_shortwave_lateral_escape_power_w =
                shortwave.lateral_escape_power_w;
            terrain_shortwave_unresolved_power_w =
                shortwave.unresolved_scattered_power_w;
            terrain_shortwave_converged = shortwave.converged;
        }
        else
        {
            const radiative_transfer::MappedDirectBeamGridView mapped_grid{
                terrain_workspace->vertical_optical_depth.data(),
                terrain_workspace->interface_height_m.data(),
                static_cast<std::size_t>(domain.nx()),
                static_cast<std::size_t>(domain.ny()), physical_nz,
                grid.dr, grid.dtheta, x_boundary, y_boundary};
            radiative_transfer::evaluate_mapped_terrain_shortwave_reference(
                mapped_grid,
                {solar_geometry.east, solar_geometry.north, solar_geometry.up},
                physical_constants::solar_constant_wm2,
                terrain_workspace->slope_x.data(),
                terrain_workspace->slope_y.data(), horizontal_cell_area,
                options, terrain_workspace->operator_workspace,
                terrain_workspace->operator_result);
            const auto& shortwave = terrain_workspace->operator_result;
            terrain_shortwave_absorbed_power_w =
                &shortwave.atmospheric_absorbed_power_w;
            terrain_shortwave_incoming_power_w = shortwave.incoming_power_w;
            terrain_shortwave_atmospheric_absorbed_power_w =
                shortwave.atmospheric_absorbed_power_w_total;
            terrain_shortwave_surface_absorbed_power_w =
                shortwave.surface_absorbed_power_w_total;
            terrain_shortwave_top_escape_power_w = shortwave.top_escape_power_w;
            terrain_shortwave_lateral_escape_power_w =
                shortwave.lateral_escape_power_w;
            terrain_shortwave_unresolved_power_w =
                shortwave.unresolved_reflected_power_w;
            terrain_shortwave_converged = shortwave.reflection_converged;
        }
        if (!terrain_shortwave_converged)
        {
            std::ostringstream message;
            message << "terrain shortwave reflection did not reach its configured residual tolerance"
                    << " (unresolved_w=" << terrain_shortwave_unresolved_power_w
                    << ", incoming_w=" << terrain_shortwave_incoming_power_w
                    << ")";
            throw std::runtime_error(message.str());
        }
        const double admitted_unresolved_power_w =
            1.0e-8 * std::max(1.0, terrain_shortwave_incoming_power_w);
        if (terrain_shortwave_unresolved_power_w > admitted_unresolved_power_w)
            throw std::runtime_error("terrain shortwave unresolved reflection exceeds the runtime accuracy contract");
        if (terrain_shortwave_absorbed_power_w == nullptr)
            throw std::logic_error("terrain shortwave omitted its atmospheric power field");

        constexpr double inverse_cp = 1.0 / physical_constants::specific_heat_cp_jkgk;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const std::size_t physical_column =
                    static_cast<std::size_t>(i - domain.i_begin) * static_cast<std::size_t>(domain.ny()) +
                    static_cast<std::size_t>(j - domain.j_begin);
                double stored_column_power_w = 0.0;
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const std::size_t kk         = static_cast<std::size_t>(k - domain.k_begin);
                    const std::size_t cell       = physical_column * physical_nz + kk;
                    const double      sw_power_w = (*terrain_shortwave_absorbed_power_w)[cell];
                    const double      density    = fields.rho[i][j][k];
                    const double      volume =
                        horizontal_cell_area *
                        (terrain_workspace->interface_height_m[physical_column * (physical_nz + 1) + kk + 1] -
                         terrain_workspace->interface_height_m[physical_column * (physical_nz + 1) + kk]);
                    const double dTdt              = sw_power_w * inverse_cp / (density * volume);
                    const double theta_scale       = std::pow(p0 / static_cast<double>(fields.p[i][j][k]), kappa);
                    const float  sw_theta_tendency = static_cast<float>(dTdt * theta_scale);
                    const float  combined_tendency = candidate_tendency[i][j][k] + sw_theta_tendency;
                    const float  combined_power    = candidate_cell_power_w[i][j][k] + static_cast<float>(sw_power_w);
                    if (!std::isfinite(combined_tendency) || !std::isfinite(combined_power))
                        throw std::overflow_error("terrain shortwave runtime source overflowed FP32 storage");
                    candidate_tendency[i][j][k]     = combined_tendency;
                    candidate_cell_power_w[i][j][k] = combined_power;
                    stored_column_power_w += combined_power;
                }
                candidate_column_power_w[static_cast<std::size_t>(i) * grid.nth + j] = stored_column_power_w;
            }
    }
    // The column calculation is parallel, but the conserved aggregate is
    // summed in a fixed physical-domain order. An OpenMP floating reduction
    // made this ledger depend on thread count even when every cell source was
    // bit-identical.
    double candidate_atmospheric_power_w = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            candidate_atmospheric_power_w += candidate_column_power_w[static_cast<std::size_t>(i) * grid.nth + j];
    double candidate_shortwave_incoming_power_w = 0.0;
    double candidate_shortwave_atmospheric_absorbed_power_w = 0.0;
    double candidate_surface_absorbed_power_w = 0.0;
    double candidate_top_escape_power_w = 0.0;
    double candidate_lateral_escape_power_w = 0.0;
    double candidate_unresolved_reflected_power_w = 0.0;
    bool candidate_reflection_converged = true;
    if (terrain_shortwave)
    {
        candidate_shortwave_incoming_power_w =
            terrain_shortwave_incoming_power_w;
        candidate_shortwave_atmospheric_absorbed_power_w =
            terrain_shortwave_atmospheric_absorbed_power_w;
        candidate_surface_absorbed_power_w =
            terrain_shortwave_surface_absorbed_power_w;
        candidate_top_escape_power_w = terrain_shortwave_top_escape_power_w;
        candidate_lateral_escape_power_w =
            terrain_shortwave_lateral_escape_power_w;
        candidate_unresolved_reflected_power_w =
            terrain_shortwave_unresolved_power_w;
        candidate_reflection_converged = terrain_shortwave_converged;
    }
    else if (active_config.enable_diagnostics && active_config.do_sw)
    {
        const std::size_t interface_count = physical_nz + 1;
        for (std::size_t column = 0;
             column < physical_column_count; ++column)
        {
            const std::size_t offset = column * interface_count;
            const double incoming =
                candidate_flux_diagnostics
                    .downward_shortwave_wm2[offset + physical_nz]
                * horizontal_cell_area;
            const double surface_down =
                candidate_flux_diagnostics.downward_shortwave_wm2[offset]
                * horizontal_cell_area;
            const double top_escape =
                candidate_flux_diagnostics
                    .upward_shortwave_wm2[offset + physical_nz]
                * horizontal_cell_area;
            const double surface_absorbed =
                (1.0 - surface_albedo) * surface_down;
            const double atmospheric_absorbed =
                incoming - surface_absorbed - top_escape;
            const double scale = std::max(
                1.0, std::max(std::abs(incoming),
                              std::abs(surface_absorbed + top_escape
                                       + atmospheric_absorbed)));
            if (!std::isfinite(atmospheric_absorbed) ||
                atmospheric_absorbed < -1.0e-11 * scale)
                throw std::runtime_error(
                    "simple_grey shortwave diagnostic power does not close");
            candidate_shortwave_incoming_power_w += incoming;
            candidate_shortwave_atmospheric_absorbed_power_w +=
                atmospheric_absorbed;
            candidate_surface_absorbed_power_w += surface_absorbed;
            candidate_top_escape_power_w += top_escape;
        }
    }

    std::swap(tendency_fields.dtheta_dt_rad, candidate_tendency);
    std::swap(active_runtime.held_cell_power_w, candidate_cell_power_w);
    std::swap(active_runtime.held_flux_diagnostics,
              candidate_flux_diagnostics);
    active_runtime.held_atmospheric_power_w = candidate_atmospheric_power_w;
    active_runtime.held_shortwave_incoming_power_w =
        candidate_shortwave_incoming_power_w;
    active_runtime.held_shortwave_atmospheric_absorbed_power_w =
        candidate_shortwave_atmospheric_absorbed_power_w;
    active_runtime.held_surface_absorbed_power_w =
        candidate_surface_absorbed_power_w;
    active_runtime.held_top_escape_power_w = candidate_top_escape_power_w;
    active_runtime.held_lateral_escape_power_w =
        candidate_lateral_escape_power_w;
    active_runtime.held_unresolved_reflected_power_w =
        candidate_unresolved_reflected_power_w;
    active_runtime.held_reflection_converged =
        candidate_reflection_converged;
    active_last_update_time = current_time;
    active_next_update_time = candidate_next_update_time;

}

void step_radiation(double current_time)
{
    step_radiation(::global_simulation_state(), current_time);
}

} // namespace tmv
