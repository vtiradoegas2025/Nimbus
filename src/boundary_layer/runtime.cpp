#include "core/runtime/native_held_sources.hpp"
#include "util/physics_cadence.hpp"
#include "boundary_layer/runtime.hpp"

#include "core/runtime/simulation_state.hpp"
#include "core/field/field_bounds.hpp"
#include "core/field/numerical_integrity.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#ifdef NIMBUS_ALLOCATION_TELEMETRY
#include "core/infra/allocation_telemetry.hpp"
#endif
#include "core/runtime/runtime_config.hpp"
#include "compute/backend_runtime.hpp"
#include "boundary_layer/factory.hpp"
#include "boundary_layer/column_evaluation.hpp"
#include "boundary_layer/native_batch_runtime.hpp"
#include "boundary_layer/config_validation.hpp"
#include "boundary_layer/surface_frame.hpp"
#include "core/infra/physical_constants.hpp"
#include "terrain/conservative_transport.hpp"
#include "util/log.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <limits>
#include <exception>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tmv
{

std::unique_ptr<BoundaryLayerSchemeBase>& boundary_layer_scheme =
    global_boundary_layer_runtime_state.scheme;
BoundaryLayerConfig& global_boundary_layer_config =
    global_boundary_layer_runtime_state.config;
SurfaceConfig& global_surface_config =
    global_boundary_layer_runtime_state.surface_config;

namespace
{
#ifdef NIMBUS_ALLOCATION_TELEMETRY
void record_field_resize_if_needed(
    const Field3D& field, int nr, int nth, int nz)
{
    const std::size_t elements = static_cast<std::size_t>(nr)
        * static_cast<std::size_t>(nth) * static_cast<std::size_t>(nz);
    if (elements <= field.capacity())
        return;
    const std::size_t raw_bytes = elements * sizeof(float);
    constexpr std::size_t alignment =
        PageAlignedAllocator<float>::kAlignmentBytes;
    const std::size_t allocated_bytes =
        ((raw_bytes + alignment - 1) / alignment) * alignment;
    tmv::allocation_telemetry::record_physics_field_allocation(
        allocated_bytes > 0 ? allocated_bytes : alignment);
}
#endif

// Held surface (skin) state for the boundary-layer coupling. When no skin
// temperature is prescribed (SurfaceConfig::Tsfc <= 0), the skin is the INITIAL
// surface-air state, captured on the first PBL step and held fixed thereafter,
// so the sensible/moisture flux is zero at t=0 and develops as the boundary
// layer evolves away from its initial state.
struct BoundaryLayerRuntimeView
{
    std::unique_ptr<BoundaryLayerSchemeBase>& scheme;
    BoundaryLayerConfig& config;
    SurfaceConfig& surface_config;
    double& last_update_time_s;
    double& next_update_time_s;
    bool& updated_this_step;
    std::vector<float>& skin_theta_2d;
    std::vector<float>& skin_qv_2d;
    bool& skin_captured;
    std::vector<SlabColumnState>& slab_columns;
    int& skin_nr;
    int& skin_nth;
    int& slab_nr;
    int& slab_nth;
    Field3D& held_u_force_n;
    Field3D& held_v_force_n;
    Field3D& held_theta_inventory_rate_kgk_s;
    Field3D& held_vapor_mass_rate_kg_s;
    Field3D& held_cloud_liquid_mass_rate_kg_s;
    Field3D& held_cloud_ice_mass_rate_kg_s;
    BoundaryLayerRefreshScratch& refresh_scratch;
    std::vector<BoundaryLayerWorkerContext>& worker_contexts;
    BoundaryLayerMomentumWorkReport& last_momentum_work;
    double& cumulative_requested_momentum_work_j;
    double& cumulative_finite_step_momentum_correction_j;
    double& cumulative_momentum_storage_rounding_j;
    double& cumulative_committed_momentum_work_j;
    std::uint64_t& momentum_work_step_count;
};

BoundaryLayerRuntimeView global_boundary_layer_runtime_view(const SimulationStateView& execution_state)
{
    auto& runtime = tmv::require_level_runtime(execution_state.boundary_layer_runtime, "boundary_layer_runtime");
    return {runtime.scheme, runtime.config, runtime.surface_config,
            runtime.last_update_time_s, runtime.next_update_time_s,
            runtime.updated_this_step,
            runtime.skin_theta_2d, runtime.skin_qv_2d,
            runtime.skin_captured, runtime.slab_columns,
            runtime.skin_nr, runtime.skin_nth,
            runtime.slab_nr, runtime.slab_nth,
            runtime.held_u_force_n, runtime.held_v_force_n,
            runtime.held_theta_inventory_rate_kgk_s,
            runtime.held_vapor_mass_rate_kg_s,
            runtime.held_cloud_liquid_mass_rate_kg_s,
            runtime.held_cloud_ice_mass_rate_kg_s,
            runtime.refresh_scratch,
            runtime.worker_contexts,
            runtime.last_momentum_work,
            runtime.cumulative_requested_momentum_work_j,
            runtime.cumulative_finite_step_momentum_correction_j,
            runtime.cumulative_momentum_storage_rounding_j,
            runtime.cumulative_committed_momentum_work_j,
            runtime.momentum_work_step_count};
}

BoundaryLayerRuntimeView boundary_layer_runtime_view(
    BoundaryLayerRuntimeState& runtime)
{
    return {runtime.scheme, runtime.config, runtime.surface_config,
            runtime.last_update_time_s, runtime.next_update_time_s,
            runtime.updated_this_step,
            runtime.skin_theta_2d, runtime.skin_qv_2d,
            runtime.skin_captured, runtime.slab_columns,
            runtime.skin_nr, runtime.skin_nth,
            runtime.slab_nr, runtime.slab_nth,
            runtime.held_u_force_n, runtime.held_v_force_n,
            runtime.held_theta_inventory_rate_kgk_s,
            runtime.held_vapor_mass_rate_kg_s,
            runtime.held_cloud_liquid_mass_rate_kg_s,
            runtime.held_cloud_ice_mass_rate_kg_s,
            runtime.refresh_scratch,
            runtime.worker_contexts,
            runtime.last_momentum_work,
            runtime.cumulative_requested_momentum_work_j,
            runtime.cumulative_finite_step_momentum_correction_j,
            runtime.cumulative_momentum_storage_rounding_j,
            runtime.cumulative_committed_momentum_work_j,
            runtime.momentum_work_step_count};
}

BoundaryLayerRuntimeView boundary_layer_runtime_for(
    const SimulationStateView& state)
{
    return state.boundary_layer_runtime
        ? boundary_layer_runtime_view(*state.boundary_layer_runtime)
        : global_boundary_layer_runtime_view(state);
}

struct SurfaceDiagnosticVectors
{
    std::vector<float>& ustar;
    std::vector<float>& tau_u;
    std::vector<float>& tau_v;
    std::vector<float>& sensible_heat_flux;
    std::vector<float>& latent_heat_flux;
    std::vector<float>& moisture_flux;
    std::vector<float>& pbl_height;
};

bool field_matches_domain(const SimulationStateView& state, const Field3D& field)
{
    return field.size_r() == state.grid.nr &&
           field.size_th() == state.grid.nth &&
           field.size_z() == state.grid.nz;
}

void ensure_tendency_shape(const SimulationStateView& state, Field3D& field)
{
    if (!field_matches_domain(state, field))
    {
        field.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    }
}

void ensure_pbl_tendency_shapes(const SimulationStateView& state)
{
    auto& tendencies = state.tendencies;
    ensure_tendency_shape(state, tendencies.dtheta_dt_pbl);
    ensure_tendency_shape(state, tendencies.dqv_dt_pbl);
    ensure_tendency_shape(state, tendencies.du_dt_pbl);
    ensure_tendency_shape(state, tendencies.dv_dt_pbl);
    ensure_tendency_shape(state, tendencies.dtke_dt_pbl);
    if (tendencies.dqc_dt_pbl)
        ensure_tendency_shape(state, *tendencies.dqc_dt_pbl);
    if (tendencies.dqi_dt_pbl)
        ensure_tendency_shape(state, *tendencies.dqi_dt_pbl);
}

std::vector<float>& surface_vector_or_global(std::vector<float>* local,
                                             std::vector<float>& global)
{
    return local ? *local : global;
}

SurfaceDiagnosticVectors surface_diagnostics_for(const SimulationStateView& state)
{
    auto& process = global_simulation_storage.surface;
    return {
        surface_vector_or_global(state.surface.sfc_ustar, process.ustar),
        surface_vector_or_global(state.surface.sfc_tau_u, process.tau_u),
        surface_vector_or_global(state.surface.sfc_tau_v, process.tau_v),
        surface_vector_or_global(state.surface.sfc_sensible_heat_flux,
                                 process.sensible_heat_flux),
        surface_vector_or_global(state.surface.sfc_latent_heat_flux,
                                 process.latent_heat_flux),
        surface_vector_or_global(state.surface.sfc_moisture_flux,
                                 process.moisture_flux),
        surface_vector_or_global(state.surface.pbl_height_2d,
                                 process.pbl_height)};
}

void ensure_surface_vector_shape(const SimulationStateView& state,
                                 std::vector<float>& field)
{
    const std::size_t sfc_size = static_cast<std::size_t>(state.grid.nr) *
                                 static_cast<std::size_t>(state.grid.nth);
    if (field.size() != sfc_size)
    {
        field.assign(sfc_size, 0.0f);
    }
}

void ensure_surface_diagnostic_shapes(const SimulationStateView& state,
                                      SurfaceDiagnosticVectors& surface)
{
    ensure_surface_vector_shape(state, surface.ustar);
    ensure_surface_vector_shape(state, surface.tau_u);
    ensure_surface_vector_shape(state, surface.tau_v);
    ensure_surface_vector_shape(state, surface.sensible_heat_flux);
    ensure_surface_vector_shape(state, surface.latent_heat_flux);
    ensure_surface_vector_shape(state, surface.moisture_flux);
    ensure_surface_vector_shape(state, surface.pbl_height);
}

void initialize_boundary_layer_runtime(BoundaryLayerRuntimeView runtime,
                                       const std::string& scheme_name,
                                       const BoundaryLayerConfig& cfg,
                                       const SurfaceConfig& sfc)
{
    BoundaryLayerConfig selected_config = cfg;
    std::unique_ptr<BoundaryLayerSchemeBase> selected_scheme =
        create_boundary_layer_scheme(scheme_name);
    selected_config.scheme_id = selected_scheme->name();
    validate_boundary_layer_configuration(
        selected_config, sfc, selected_scheme->name());

    selected_scheme->initialize(selected_config);

    runtime.scheme = std::move(selected_scheme);
    runtime.config = std::move(selected_config);
    runtime.surface_config = sfc;
    runtime.last_update_time_s = std::numeric_limits<double>::quiet_NaN();
    runtime.next_update_time_s = 0.0;
    runtime.updated_this_step = false;
    runtime.skin_captured = false;
    runtime.skin_nr = -1;
    runtime.skin_nth = -1;
    runtime.skin_theta_2d.clear();
    runtime.skin_qv_2d.clear();
    runtime.slab_nr = -1;
    runtime.slab_nth = -1;
    runtime.slab_columns.clear();
    runtime.held_u_force_n.release_storage();
    runtime.held_v_force_n.release_storage();
    runtime.held_theta_inventory_rate_kgk_s.release_storage();
    runtime.held_vapor_mass_rate_kg_s.release_storage();
    runtime.held_cloud_liquid_mass_rate_kg_s.release_storage();
    runtime.held_cloud_ice_mass_rate_kg_s.release_storage();
    runtime.refresh_scratch.release_storage();
    std::vector<BoundaryLayerWorkerContext>().swap(runtime.worker_contexts);
    runtime.last_momentum_work = {};
    runtime.cumulative_requested_momentum_work_j = 0.0;
    runtime.cumulative_finite_step_momentum_correction_j = 0.0;
    runtime.cumulative_momentum_storage_rounding_j = 0.0;
    runtime.cumulative_committed_momentum_work_j = 0.0;
    runtime.momentum_work_step_count = 0;

    tmv::log_info("Initialized boundary layer scheme: ", scheme_name);
    tmv::log_info("  PBL cadence: ", runtime.config.dt_pbl, " s");
    if (!runtime.config.clear_sky_only)
    {
        tmv::log_info(
            "  Cloud-top mixing: surface-coupled Lock-2000 radiative YSU; "
            "theta_l and qv/qc/qi transport; owner=cpu");
    }
    tmv::log_info("  Surface fluxes: ",
                  (runtime.config.apply_surface_fluxes ? "enabled" : "disabled"));
    if (runtime.config.apply_surface_fluxes)
    {
        if (runtime.surface_config.Tsfc > 0.0)
        {
            tmv::log_info("  Surface skin: prescribed Tsfc = ",
                          runtime.surface_config.Tsfc,
                          " K (converted to potential temperature at the surface pressure)");
        }
        else
        {
            tmv::log_info("  Surface skin: no surface.tsk_k set -> holding the INITIAL "
                          "surface-air state as the skin (zero sensible/moisture flux at t=0; "
                          "flux develops as the boundary layer evolves). Set surface.tsk_k to "
                          "prescribe a fixed skin temperature.");
        }
    }
}
}


// Initialize the process-owned PBL runtime and its output fields.
void initialize_boundary_layer(const std::string& scheme_name,const BoundaryLayerConfig& cfg, const SurfaceConfig& sfc)
{
    return initialize_boundary_layer(global_simulation_state(), scheme_name, cfg, sfc);
}

void initialize_boundary_layer(const SimulationStateView& execution_state, const std::string& scheme_name,const BoundaryLayerConfig& cfg, const SurfaceConfig& sfc)
{
    try
    {
        initialize_boundary_layer_runtime(global_boundary_layer_runtime_view(execution_state),
                                          scheme_name, cfg, sfc);

        const auto state = execution_state;
        ensure_pbl_tendency_shapes(state);

        auto surface = surface_diagnostics_for(state);
        ensure_surface_diagnostic_shapes(state, surface);
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error initializing boundary layer: ", e.what());
        throw;
    }
}

void initialize_boundary_layer(BoundaryLayerRuntimeState& runtime,
                               const std::string& scheme_name,
                               const BoundaryLayerConfig& cfg,
                               const SurfaceConfig& sfc)
{
    try
    {
        initialize_boundary_layer_runtime(boundary_layer_runtime_view(runtime),
                                          scheme_name, cfg, sfc);
    }
    catch (const std::exception& e)
    {
        tmv::log_error("Error initializing boundary layer: ", e.what());
        throw;
    }
}

// Recompute column tendencies when the module cadence is due.
void step_boundary_layer(const SimulationStateView& state, double current_time)
{
    BoundaryLayerRuntimeView runtime = boundary_layer_runtime_for(state);
    if (!runtime.scheme)
    {
        throw std::runtime_error(
            "step_boundary_layer: boundary-layer scheme is not initialized");
    }
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    auto& tendency_fields = state.tendencies;
    auto surface_fields = surface_diagnostics_for(state);

    if (grid.nr <= 0 || grid.nth <= 0 || grid.nz <= 0)
    {
        throw std::invalid_argument(
            "step_boundary_layer: grid dimensions must be positive");
    }
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (domain.empty())
    {
        throw std::invalid_argument(
            "step_boundary_layer: physical-cell domain is empty");
    }
    if (!std::isfinite(current_time) || current_time < 0.0 ||
        !std::isfinite(runtime.config.dt_pbl) || runtime.config.dt_pbl <= 0.0)
    {
        throw std::invalid_argument(
            "step_boundary_layer: time must be non-negative and finite and cadence positive");
    }

    const bool first_update = !std::isfinite(runtime.last_update_time_s);
    if (!std::isfinite(runtime.next_update_time_s))
    {
        throw std::invalid_argument(
            "step_boundary_layer: next cadence time must be finite");
    }
    if (!first_update && current_time < runtime.last_update_time_s)
    {
        throw std::invalid_argument(
            "step_boundary_layer: simulation time moved backward across the PBL cadence");
    }
    const double cadence_tolerance_s = 1.0e-12 * std::max(
        {1.0, std::abs(current_time), std::abs(runtime.next_update_time_s),
         runtime.config.dt_pbl});
    if (current_time + cadence_tolerance_s < runtime.next_update_time_s)
    {
        runtime.updated_this_step = false;
        return;
    }

    const double candidate_next_update_time_s = next_physics_deadline(
        runtime.next_update_time_s, current_time, runtime.config.dt_pbl, cadence_tolerance_s);
    const double sampled_nominal_time_s =
        candidate_next_update_time_s - runtime.config.dt_pbl;
    const double previous_nominal_time_s =
        runtime.next_update_time_s - runtime.config.dt_pbl;
    const double elapsed_time_s = first_update
        ? 0.0
        : sampled_nominal_time_s - previous_nominal_time_s;

    const auto required_field_shape_ok = [&](const Field3D& field)
    {
        return field_matches_domain(state, field);
    };
    if (!required_field_shape_ok(fields.p) ||
        !required_field_shape_ok(fields.theta) ||
        !required_field_shape_ok(fields.qv) ||
        !required_field_shape_ok(fields.u) ||
        !required_field_shape_ok(fields.v) ||
        !required_field_shape_ok(fields.rho) ||
        (!runtime.config.clear_sky_only &&
         (!required_field_shape_ok(fields.qc) ||
          !required_field_shape_ok(fields.qi) ||
          !required_field_shape_ok(tendency_fields.dtheta_dt_rad) ||
          tendency_fields.dqc_dt_pbl == nullptr ||
          tendency_fields.dqi_dt_pbl == nullptr)) ||
        (!fields.tke.empty() && !required_field_shape_ok(fields.tke)))
    {
        throw std::invalid_argument(
            "step_boundary_layer: required state field shape does not match the grid");
    }
    physical_grid_metrics::validate(state);
    const TerrainMetrics3D* const terrain_metrics =
        physical_grid_metrics::active_terrain_metrics(state);
    if (terrain_metrics != nullptr)
        terrain_transport::validate_geometry(*terrain_metrics);
    const bool stagger_x = grid.geometry.staggered && grid.nr >= 3;
    const bool stagger_y = grid.geometry.staggered && grid.nth >= 3;

    if (runtime.scheme->name() == "none")
    {
        // The disabled scheme owns no candidate or held full-volume storage.
        // Publish zero caller-owned diagnostics directly after all failure-
        // capable validation, and drop any stale active-scheme high water.
        ensure_pbl_tendency_shapes(state);
        tendency_fields.dtheta_dt_pbl.fill(0.0f);
        tendency_fields.dqv_dt_pbl.fill(0.0f);
        if (tendency_fields.dqc_dt_pbl)
            tendency_fields.dqc_dt_pbl->fill(0.0f);
        if (tendency_fields.dqi_dt_pbl)
            tendency_fields.dqi_dt_pbl->fill(0.0f);
        tendency_fields.du_dt_pbl.fill(0.0f);
        tendency_fields.dv_dt_pbl.fill(0.0f);
        tendency_fields.dtke_dt_pbl.fill(0.0f);
        ensure_surface_diagnostic_shapes(state, surface_fields);
        std::fill(surface_fields.ustar.begin(), surface_fields.ustar.end(), 0.0f);
        std::fill(surface_fields.tau_u.begin(), surface_fields.tau_u.end(), 0.0f);
        std::fill(surface_fields.tau_v.begin(), surface_fields.tau_v.end(), 0.0f);
        std::fill(surface_fields.sensible_heat_flux.begin(),
                  surface_fields.sensible_heat_flux.end(), 0.0f);
        std::fill(surface_fields.latent_heat_flux.begin(),
                  surface_fields.latent_heat_flux.end(), 0.0f);
        std::fill(surface_fields.moisture_flux.begin(),
                  surface_fields.moisture_flux.end(), 0.0f);
        std::fill(surface_fields.pbl_height.begin(),
                  surface_fields.pbl_height.end(), 0.0f);
        runtime.held_u_force_n.release_storage();
        runtime.held_v_force_n.release_storage();
        runtime.held_theta_inventory_rate_kgk_s.release_storage();
        runtime.held_vapor_mass_rate_kg_s.release_storage();
        runtime.held_cloud_liquid_mass_rate_kg_s.release_storage();
        runtime.held_cloud_ice_mass_rate_kg_s.release_storage();
        runtime.refresh_scratch.release_storage();
        std::vector<BoundaryLayerWorkerContext>().swap(
            runtime.worker_contexts);
        runtime.last_update_time_s = current_time;
        runtime.next_update_time_s = candidate_next_update_time_s;
        runtime.updated_this_step = true;
        return;
    }

#ifdef NIMBUS_PHYSICS_LEGACY_ALLOCATION_EXPERIMENT
    // Measurement-only reconstruction of the pre-SCI-122 allocation
    // lifetime. Committed tendency/source/surface/slab/skin state remains
    // intact; only replaceable candidates and worker scratch are discarded.
    runtime.refresh_scratch.release_storage();
    std::vector<BoundaryLayerWorkerContext>().swap(
        runtime.worker_contexts);
#endif

    auto& scratch = runtime.refresh_scratch;
    Field3D& candidate_dtheta = scratch.dtheta;
    Field3D& candidate_dqv = scratch.dqv;
    Field3D& candidate_dqc = scratch.dqc;
    Field3D& candidate_dqi = scratch.dqi;
    Field3D& candidate_du = scratch.du;
    Field3D& candidate_dv = scratch.dv;
    Field3D& candidate_dtke = scratch.dtke;
    Field3D& candidate_u_force = scratch.u_force;
    Field3D& candidate_v_force = scratch.v_force;
    Field3D& candidate_theta_rate = scratch.theta_inventory_rate;
    Field3D& candidate_vapor_rate = scratch.vapor_mass_rate;
    Field3D& candidate_cloud_liquid_rate = scratch.cloud_liquid_mass_rate;
    Field3D& candidate_cloud_ice_rate = scratch.cloud_ice_mass_rate;
    Field3D* const candidate_fields[] = {
        &candidate_dtheta, &candidate_dqv, &candidate_dqc, &candidate_dqi,
        &candidate_du, &candidate_dv,
        &candidate_dtke, &candidate_u_force, &candidate_v_force,
        &candidate_theta_rate, &candidate_vapor_rate,
        &candidate_cloud_liquid_rate, &candidate_cloud_ice_rate};
    for (Field3D* candidate : candidate_fields)
    {
#ifdef NIMBUS_ALLOCATION_TELEMETRY
        record_field_resize_if_needed(
            *candidate, grid.nr, grid.nth, grid.nz);
#endif
        candidate->resize(grid.nr, grid.nth, grid.nz, 0.0f);
    }

    const std::size_t surface_size = static_cast<std::size_t>(grid.nr) *
                                     static_cast<std::size_t>(grid.nth);
    std::vector<float>& candidate_ustar = scratch.ustar;
    std::vector<float>& candidate_tau_u = scratch.tau_u;
    std::vector<float>& candidate_tau_v = scratch.tau_v;
    std::vector<float>& candidate_sensible = scratch.sensible_heat_flux;
    std::vector<float>& candidate_latent = scratch.latent_heat_flux;
    std::vector<float>& candidate_moisture = scratch.moisture_flux;
    std::vector<float>& candidate_height = scratch.pbl_height;
    std::vector<float>* const candidate_surface_fields[] = {
        &candidate_ustar, &candidate_tau_u, &candidate_tau_v,
        &candidate_sensible, &candidate_latent, &candidate_moisture,
        &candidate_height};
    for (std::vector<float>* candidate : candidate_surface_fields)
        candidate->assign(surface_size, 0.0f);
    const physical_grid_metrics::ControlVolumeMassView mass_geometry(state);
    const double horizontal_cell_area = grid.dr * grid.dtheta;
    if (!std::isfinite(horizontal_cell_area) || horizontal_cell_area <= 0.0)
        throw std::invalid_argument(
            "step_boundary_layer: horizontal cell area must be positive and finite");

    std::vector<float>& candidate_skin_theta = scratch.skin_theta;
    std::vector<float>& candidate_skin_qv = scratch.skin_qv;
    if (runtime.skin_nr == grid.nr && runtime.skin_nth == grid.nth &&
        runtime.skin_theta_2d.size() == surface_size)
        candidate_skin_theta.assign(
            runtime.skin_theta_2d.begin(), runtime.skin_theta_2d.end());
    else
        candidate_skin_theta.assign(surface_size, 0.0f);
    if (runtime.skin_nr == grid.nr && runtime.skin_nth == grid.nth &&
        runtime.skin_qv_2d.size() == surface_size)
        candidate_skin_qv.assign(
            runtime.skin_qv_2d.begin(), runtime.skin_qv_2d.end());
    else
        candidate_skin_qv.assign(surface_size, 0.0f);
    const bool candidate_skin_captured =
        runtime.skin_captured && runtime.skin_nr == grid.nr &&
        runtime.skin_nth == grid.nth &&
        runtime.skin_theta_2d.size() == surface_size &&
        runtime.skin_qv_2d.size() == surface_size;
    std::vector<SlabColumnState>& candidate_slab = scratch.slab_columns;
    if (runtime.slab_nr == grid.nr && runtime.slab_nth == grid.nth &&
        runtime.slab_columns.size() == surface_size)
        candidate_slab.assign(
            runtime.slab_columns.begin(), runtime.slab_columns.end());
    else
        candidate_slab.assign(surface_size, SlabColumnState{});

    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double p_value = fields.p[i][j][k];
                const double theta_value = fields.theta[i][j][k];
                const double qv_value = fields.qv[i][j][k];
                const double rho_value = fields.rho[i][j][k];
                const double qc_value = runtime.config.clear_sky_only
                    ? 0.0 : static_cast<double>(fields.qc[i][j][k]);
                const double qi_value = runtime.config.clear_sky_only
                    ? 0.0 : static_cast<double>(fields.qi[i][j][k]);
                const double radiative_tendency = runtime.config.clear_sky_only
                    ? 0.0
                    : static_cast<double>(
                        tendency_fields.dtheta_dt_rad[i][j][k]);
                const double u_value = stagger_x
                    ? 0.5 * (static_cast<double>(fields.u[i - 1][j][k])
                           + static_cast<double>(fields.u[i][j][k]))
                    : static_cast<double>(fields.u[i][j][k]);
                const double v_value = stagger_y
                    ? 0.5 * (static_cast<double>(fields.v[i][j - 1][k])
                           + static_cast<double>(fields.v[i][j][k]))
                    : static_cast<double>(fields.v[i][j][k]);
                if (!std::isfinite(p_value) || p_value <= 0.0 ||
                    !std::isfinite(theta_value) || theta_value <= 0.0 ||
                    !std::isfinite(qv_value) || qv_value < 0.0 ||
                    !std::isfinite(rho_value) || rho_value <= 0.0 ||
                    !std::isfinite(qc_value) || qc_value < 0.0 ||
                    !std::isfinite(qi_value) || qi_value < 0.0 ||
                    !std::isfinite(radiative_tendency) ||
                    !std::isfinite(u_value) || !std::isfinite(v_value) ||
                    (!fields.tke.empty() &&
                     !std::isfinite(static_cast<double>(fields.tke[i][j][k]))))
                {
                    throw std::invalid_argument(
                        "step_boundary_layer: column state must be finite with positive pressure/theta/density and non-negative vapor/cloud condensate");
                }
                if (k > domain.k_begin &&
                    p_value > static_cast<double>(fields.p[i][j][k - 1]))
                {
                    throw std::invalid_argument(
                        "step_boundary_layer: pressure must be non-increasing with height");
                }
            }
        }
    }

    const std::size_t physical_column_count =
        static_cast<std::size_t>(domain.nx())
        * static_cast<std::size_t>(domain.ny());
    const bool capture_skin = !candidate_skin_captured;
    int failed_columns = 0;
    std::string first_column_error;
    bool native_batch_completed = false;
    const ComputeBackendRuntimeState* const compute = state.compute_runtime;
    const bool active_accelerator =
        compute != nullptr && compute->backend != nullptr &&
        compute->active_kind != ComputeBackendKind::Cpu &&
        !compute->fallback_active;
    if (active_accelerator && (runtime.scheme->name() == "mynn" || runtime.scheme->name() == "slab"))
    {
        BoundaryLayerComputeKernels* const kernels =
            compute->backend->boundary_layer_kernels();
        const bool supported = kernels != nullptr &&
            runtime.scheme->supports_native_column_batch(
                runtime.config, *kernels);
        if (supported)
        {
            BoundaryLayerBatchPublication publication{
                candidate_dtheta, candidate_dqv, candidate_du, candidate_dv,
                candidate_dtke, candidate_u_force, candidate_v_force,
                candidate_theta_rate, candidate_vapor_rate, candidate_ustar,
                candidate_tau_u, candidate_tau_v, candidate_sensible,
                candidate_latent, candidate_moisture, candidate_height,
                candidate_skin_theta, candidate_skin_qv, candidate_slab};
            native_batch_completed = evaluate_native_boundary_layer_batches(
                state, *runtime.scheme, runtime.config,
                runtime.surface_config, runtime.refresh_scratch, domain,
                stagger_x, stagger_y, capture_skin, elapsed_time_s,
                horizontal_cell_area, *kernels, publication);
        }
        if (!native_batch_completed && !compute->config.allow_fallback)
            throw std::runtime_error(
                supported
                    ? "native boundary-layer batch dispatch was rejected"
                    : "active accelerator does not provide the required native boundary-layer batch operation");
        if (!native_batch_completed)
            tmv::log_warn(
                "[BOUNDARY_LAYER] native boundary-layer batch unavailable; using permitted CPU column fallback");
    }

    if (!native_batch_completed)
    {
    const bool parallel_columns = physical_column_count >= 64;
    int worker_count = 1;
#ifdef _OPENMP
    if (parallel_columns)
    {
        const int column_limit = static_cast<int>(std::min(
            physical_column_count,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        worker_count = std::max(
            1,
            std::min(omp_get_max_threads(), column_limit));
    }
#endif
    const std::size_t required_worker_count =
        static_cast<std::size_t>(worker_count);
    bool rebuild_worker_contexts =
        runtime.worker_contexts.size() != required_worker_count;
    if (!rebuild_worker_contexts)
    {
        for (const BoundaryLayerWorkerContext& context :
             runtime.worker_contexts)
        {
            if (!context.scheme ||
                context.scheme->name() != runtime.scheme->name())
            {
                rebuild_worker_contexts = true;
                break;
            }
        }
    }
    if (rebuild_worker_contexts)
    {
        std::vector<BoundaryLayerWorkerContext> candidates(
            required_worker_count);
        for (BoundaryLayerWorkerContext& context : candidates)
        {
            context.scheme = runtime.scheme->clone();
            if (!context.scheme)
            {
                throw std::runtime_error(
                    "boundary-layer scheme does not support worker cloning: " +
                    runtime.scheme->name());
            }
            context.diagnostics.include_diffusivity_profiles = false;
        }
        runtime.worker_contexts = std::move(candidates);
    }
    for (BoundaryLayerWorkerContext& context : runtime.worker_contexts)
        context.diagnostics.include_diffusivity_profiles = false;

    #pragma omp parallel if(parallel_columns) num_threads(worker_count) reduction(+:failed_columns)
    {
        int worker_index = 0;
#ifdef _OPENMP
        worker_index = omp_get_thread_num();
#endif
        BoundaryLayerWorkerContext& context =
            runtime.worker_contexts[static_cast<std::size_t>(worker_index)];

        #pragma omp for collapse(2) schedule(static)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                // Boundary-layer heights are measured above the local ground.
                // The exact physical interface separation is retained.
                physical_grid_metrics::column_geometry(
                    state, i, j, domain.k_begin, domain.k_end,
                    context.workspace.z_center,
                    context.workspace.z_interface,
                    context.workspace.layer_thickness,
                    /*height_above_ground=*/true);
                const double z_sfc = context.workspace.z_center.front();
                if (!std::isfinite(z_sfc) || z_sfc <= 0.0 ||
                    z_sfc <= std::max(runtime.surface_config.z0m,
                                      runtime.surface_config.z0h))
                {
                    ++failed_columns;
                    #pragma omp critical(boundary_layer_column_error)
                    {
                        if (first_column_error.empty())
                            first_column_error =
                                "first scalar height above local terrain must exceed both roughness lengths";
                    }
                    continue;
                }
                const std::size_t column_offset =
                    static_cast<std::size_t>(domain.k_begin);
                const float* p_row = fields.p.row_ptr(i, j) + column_offset;
                const float* theta_row =
                    fields.theta.row_ptr(i, j) + column_offset;
                const float* qv_row = fields.qv.row_ptr(i, j) + column_offset;
                const float* u_row = fields.u.row_ptr(i, j) + column_offset;
                const float* v_row = fields.v.row_ptr(i, j) + column_offset;
                const float* rho_row = fields.rho.row_ptr(i, j) + column_offset;
                const float* tke_row = fields.tke.empty()
                    ? nullptr
                    : fields.tke.row_ptr(i, j) + column_offset;
                const float* qc_row = runtime.config.clear_sky_only
                    ? nullptr
                    : fields.qc.row_ptr(i, j) + column_offset;
                const float* qi_row = runtime.config.clear_sky_only
                    ? nullptr
                    : fields.qi.row_ptr(i, j) + column_offset;
                const float* radiative_row = runtime.config.clear_sky_only
                    ? nullptr
                    : tendency_fields.dtheta_dt_rad.row_ptr(i, j)
                        + column_offset;

                BoundaryLayerColumnStateView col;
                const std::size_t column_size =
                    static_cast<std::size_t>(domain.nz());
                col.p = BoundaryLayerProfileView(p_row, column_size);
                col.theta = BoundaryLayerProfileView(theta_row, column_size);
                col.qv = BoundaryLayerProfileView(qv_row, column_size);
                if (stagger_x)
                {
                    context.workspace.centered_u.resize(column_size);
                    const float* west_u_row =
                        fields.u.row_ptr(i - 1, j) + column_offset;
                    for (std::size_t k = 0; k < column_size; ++k)
                        context.workspace.centered_u[k] = 0.5 * (
                            static_cast<double>(west_u_row[k])
                          + static_cast<double>(u_row[k]));
                    col.u = &context.workspace.centered_u;
                }
                else
                {
                    col.u = BoundaryLayerProfileView(u_row, column_size);
                }
                if (stagger_y)
                {
                    context.workspace.centered_v.resize(column_size);
                    const float* south_v_row =
                        fields.v.row_ptr(i, j - 1) + column_offset;
                    for (std::size_t k = 0; k < column_size; ++k)
                        context.workspace.centered_v[k] = 0.5 * (
                            static_cast<double>(south_v_row[k])
                          + static_cast<double>(v_row[k]));
                    col.v = &context.workspace.centered_v;
                }
                else
                {
                    col.v = BoundaryLayerProfileView(v_row, column_size);
                }
                col.rho = BoundaryLayerProfileView(rho_row, column_size);
                col.z_int = &context.workspace.z_interface;
                col.tke = tke_row
                    ? BoundaryLayerProfileView(tke_row, column_size)
                    : BoundaryLayerProfileView{};
                col.qc = qc_row
                    ? BoundaryLayerProfileView(qc_row, column_size)
                    : BoundaryLayerProfileView{};
                col.qi = qi_row
                    ? BoundaryLayerProfileView(qi_row, column_size)
                    : BoundaryLayerProfileView{};
                col.radiative_theta_tendency = radiative_row
                    ? BoundaryLayerProfileView(radiative_row, column_size)
                    : BoundaryLayerProfileView{};
                col.workspace = &context.workspace;

                const std::size_t sfc_idx = static_cast<std::size_t>(i) *
                                            static_cast<std::size_t>(grid.nth) +
                                            static_cast<std::size_t>(j);
                if (capture_skin)
                {
                    candidate_skin_theta[sfc_idx] = theta_row[0];
                    candidate_skin_qv[sfc_idx] = qv_row[0];
                }

                const auto surface_wind =
                    boundary_layer_surface_relative_wind(
                        col.u[0], col.v[0], terrain_metrics != nullptr
                            && state.terrain_runtime->config.horizontal_placement_id != "earth_fixed",
                        state.domain_frame);
                col.u_sfc = surface_wind.first;
                col.v_sfc = surface_wind.second;
                if (runtime.surface_config.Tsfc > 0.0)
                {
                    const double p_sfc = static_cast<double>(p_row[0]);
                    const double exner_inv = std::pow(
                        physical_constants::reference_pressure_pa / p_sfc,
                        physical_constants::gas_constant_dry_air_jkgk /
                            physical_constants::specific_heat_cp_jkgk);
                    col.theta_sfc = runtime.surface_config.Tsfc * exner_inv;
                }
                else
                {
                    col.theta_sfc = static_cast<double>(
                        candidate_skin_theta[sfc_idx]);
                }
                col.qv_sfc = (runtime.surface_config.qsfc >= 0.0)
                    ? runtime.surface_config.qsfc
                    : static_cast<double>(candidate_skin_qv[sfc_idx]);
                col.z_sfc = z_sfc;
                if (terrain_metrics != nullptr)
                {
                    col.surface_slope_x =
                        -terrain_metrics->Gx_face(
                            i, j, domain.k_begin);
                    col.surface_slope_y =
                        -terrain_metrics->Gy_face(
                            i, j, domain.k_begin);
                }
                col.slab_state = &candidate_slab[sfc_idx];
                col.elapsed_time_s = elapsed_time_s;

                BoundaryLayerStoredSurfaceDiagnostics stored;
                try
                {
                    stored = evaluate_prepared_boundary_layer_column(
                        *context.scheme, runtime.config,
                        runtime.surface_config, col,
                        horizontal_cell_area, context.tendencies,
                        context.diagnostics);
                }
                catch (const std::exception& error)
                {
                    ++failed_columns;
                    #pragma omp critical(boundary_layer_column_error)
                    {
                        if (first_column_error.empty())
                            first_column_error = error.what();
                    }
                    continue;
                }
                candidate_ustar[sfc_idx] = stored.ustar;
                candidate_tau_u[sfc_idx] = stored.tau_u;
                candidate_tau_v[sfc_idx] = stored.tau_v;
                candidate_sensible[sfc_idx] = stored.sensible_heat_flux;
                candidate_moisture[sfc_idx] = stored.moisture_flux;
                candidate_latent[sfc_idx] = stored.latent_heat_flux;
                candidate_height[sfc_idx] = stored.pbl_height;
                for (int k = 0; k < domain.nz(); ++k)
                {
                    const size_t k_idx = static_cast<size_t>(k);
                    const int model_k = domain.k_begin + k;
                    candidate_dtheta[i][j][model_k] =
                        context.workspace.stored_dtheta[k_idx];
                    candidate_dqv[i][j][model_k] =
                        context.workspace.stored_dqv[k_idx];
                    candidate_dqc[i][j][model_k] =
                        context.workspace.stored_dqc[k_idx];
                    candidate_dqi[i][j][model_k] =
                        context.workspace.stored_dqi[k_idx];
                    candidate_du[i][j][model_k] =
                        context.workspace.stored_du[k_idx];
                    candidate_dv[i][j][model_k] =
                        context.workspace.stored_dv[k_idx];
                    candidate_dtke[i][j][model_k] =
                        context.workspace.stored_dtke[k_idx];
                    candidate_u_force[i][j][model_k] =
                        context.workspace.stored_u_force_n[k_idx];
                    candidate_v_force[i][j][model_k] =
                        context.workspace.stored_v_force_n[k_idx];
                    candidate_theta_rate[i][j][model_k] =
                        context.workspace
                            .stored_theta_inventory_rate_kgk_s[k_idx];
                    candidate_vapor_rate[i][j][model_k] =
                        context.workspace.stored_vapor_mass_rate_kg_s[k_idx];
                    candidate_cloud_liquid_rate[i][j][model_k] =
                        context.workspace
                            .stored_cloud_liquid_mass_rate_kg_s[k_idx];
                    candidate_cloud_ice_rate[i][j][model_k] =
                        context.workspace.stored_cloud_ice_mass_rate_kg_s[k_idx];
                }
            }
        }
    }
    }

    if (failed_columns > 0)
    {
        record_numerical_nonfinite_incidents(
            static_cast<std::size_t>(failed_columns));
        throw std::runtime_error(
            "boundary-layer update rejected " +
            std::to_string(failed_columns) + " column(s): " +
            first_column_error);
    }

    // Column closures publish force on scalar control volumes. A C-grid
    // velocity update instead owns the dual volume surrounding each face.
    // Half of each adjacent scalar force and mass belongs to that dual volume;
    // the ratio is therefore a conservative, live-density face tendency.
    if (grid.geometry.staggered)
    {
        candidate_du.fill(0.0f);
        candidate_dv.fill(0.0f);
        const bool periodic_x =
            tmv::require_run_configuration(state).lateral_bc_west == tmv::init::config::LateralBC::Periodic
            || tmv::require_run_configuration(state).lateral_bc_east == tmv::init::config::LateralBC::Periodic;
        const bool periodic_y =
            tmv::require_run_configuration(state).lateral_bc_south == tmv::init::config::LateralBC::Periodic
            || tmv::require_run_configuration(state).lateral_bc_north == tmv::init::config::LateralBC::Periodic;
        if (stagger_x)
        {
            const int face_end = periodic_x ? domain.i_end
                                            : domain.i_end - 1;
            for (int i = domain.i_begin; i < face_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const int right = i + 1 < domain.i_end
                            ? i + 1 : domain.i_begin;
                        const double dual_force = 0.5 * (
                            static_cast<double>(candidate_u_force[i][j][k])
                          + static_cast<double>(candidate_u_force[right][j][k]));
                        const double dual_mass = mass_geometry.u_dual_mass_kg(
                            fields.rho, i, j, k, right);
                        candidate_du[i][j][k] = static_cast<float>(
                            dual_force / dual_mass);
                    }
        }
        if (stagger_y)
        {
            const int face_end = periodic_y ? domain.j_end
                                            : domain.j_end - 1;
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < face_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const int north = j + 1 < domain.j_end
                            ? j + 1 : domain.j_begin;
                        const double dual_force = 0.5 * (
                            static_cast<double>(candidate_v_force[i][j][k])
                          + static_cast<double>(candidate_v_force[i][north][k]));
                        const double dual_mass = mass_geometry.v_dual_mass_kg(
                            fields.rho, i, j, k, north);
                        candidate_dv[i][j][k] = static_cast<float>(
                            dual_force / dual_mass);
                    }
        }
    }

    std::swap(tendency_fields.dtheta_dt_pbl, candidate_dtheta);
    std::swap(tendency_fields.dqv_dt_pbl, candidate_dqv);
    if (tendency_fields.dqc_dt_pbl)
        std::swap(*tendency_fields.dqc_dt_pbl, candidate_dqc);
    if (tendency_fields.dqi_dt_pbl)
        std::swap(*tendency_fields.dqi_dt_pbl, candidate_dqi);
    std::swap(tendency_fields.du_dt_pbl, candidate_du);
    std::swap(tendency_fields.dv_dt_pbl, candidate_dv);
    std::swap(tendency_fields.dtke_dt_pbl, candidate_dtke);
    std::swap(surface_fields.ustar, candidate_ustar);
    std::swap(surface_fields.tau_u, candidate_tau_u);
    std::swap(surface_fields.tau_v, candidate_tau_v);
    std::swap(surface_fields.sensible_heat_flux, candidate_sensible);
    std::swap(surface_fields.latent_heat_flux, candidate_latent);
    std::swap(surface_fields.moisture_flux, candidate_moisture);
    std::swap(surface_fields.pbl_height, candidate_height);
    std::swap(runtime.skin_theta_2d, candidate_skin_theta);
    std::swap(runtime.skin_qv_2d, candidate_skin_qv);
    runtime.skin_captured = true;
    runtime.skin_nr = grid.nr;
    runtime.skin_nth = grid.nth;
    std::swap(runtime.slab_columns, candidate_slab);
    runtime.slab_nr = grid.nr;
    runtime.slab_nth = grid.nth;
    std::swap(runtime.held_u_force_n, candidate_u_force);
    std::swap(runtime.held_v_force_n, candidate_v_force);
    std::swap(runtime.held_theta_inventory_rate_kgk_s, candidate_theta_rate);
    std::swap(runtime.held_vapor_mass_rate_kg_s, candidate_vapor_rate);
    std::swap(runtime.held_cloud_liquid_mass_rate_kg_s,
              candidate_cloud_liquid_rate);
    std::swap(runtime.held_cloud_ice_mass_rate_kg_s,
              candidate_cloud_ice_rate);
    runtime.last_update_time_s = current_time;
    runtime.next_update_time_s = candidate_next_update_time_s;
    runtime.updated_this_step = true;
}

void prepare_boundary_layer_tendencies_for_application(const SimulationStateView& state)
{
    BoundaryLayerRuntimeView runtime = boundary_layer_runtime_for(state);
    if (!runtime.scheme || runtime.scheme->name() == "none")
        return;

    const auto shape_matches = [&](const Field3D& field) { return field_matches_domain(state, field); };
    if (!shape_matches(runtime.held_u_force_n) || !shape_matches(runtime.held_v_force_n) ||
        !shape_matches(runtime.held_theta_inventory_rate_kgk_s) || !shape_matches(runtime.held_vapor_mass_rate_kg_s) ||
        (!runtime.config.clear_sky_only &&
         (!shape_matches(runtime.held_cloud_liquid_mass_rate_kg_s) ||
          !shape_matches(runtime.held_cloud_ice_mass_rate_kg_s) || state.tendencies.dqc_dt_pbl == nullptr ||
          state.tendencies.dqi_dt_pbl == nullptr)))
    {
        throw std::runtime_error("active boundary layer is missing conserved held-source fields");
    }
    if (state.configuration && state.configuration->admission.earth_fixed_terrain_motion && state.compute_runtime &&
        state.compute_runtime->active_kind != ComputeBackendKind::Cpu)
    {
        auto* kernels = state.compute_runtime->backend ? state.compute_runtime->backend->terrain_kernels() : nullptr;
        if (!kernels || state.compute_runtime->fallback_active)
            throw std::runtime_error("native Earth-fixed PBL reconstruction lost its device owner");
        HeldSourceInputs inputs;
        inputs.u_force    = &runtime.held_u_force_n;
        inputs.v_force    = &runtime.held_v_force_n;
        inputs.theta_rate = &runtime.held_theta_inventory_rate_kgk_s;
        inputs.vapor_rate = &runtime.held_vapor_mass_rate_kg_s;
        if (!runtime.config.clear_sky_only)
        {
            inputs.liquid_rate = &runtime.held_cloud_liquid_mass_rate_kg_s;
            inputs.ice_rate    = &runtime.held_cloud_ice_mass_rate_kg_s;
        }
        auto result                    = reconstruct_native_held_sources(state, inputs, *kernels);
        state.tendencies.du_dt_pbl     = std::move(result.du);
        state.tendencies.dv_dt_pbl     = std::move(result.dv);
        state.tendencies.dtheta_dt_pbl = std::move(result.dtheta_pbl);
        state.tendencies.dqv_dt_pbl    = std::move(result.dqv);
        if (state.tendencies.dqc_dt_pbl)
            *state.tendencies.dqc_dt_pbl = std::move(result.dqc);
        if (state.tendencies.dqi_dt_pbl)
            *state.tendencies.dqi_dt_pbl = std::move(result.dqi);
        return;
    }
    physical_grid_metrics::validate(state);
    const physical_grid_metrics::ControlVolumeMassView mass_geometry(state);
    Field3D&                                           candidate_du     = runtime.refresh_scratch.du;
    Field3D&                                           candidate_dv     = runtime.refresh_scratch.dv;
    Field3D&                                           candidate_dtheta = runtime.refresh_scratch.dtheta;
    Field3D&                                           candidate_dqv    = runtime.refresh_scratch.dqv;
    Field3D&                                           candidate_dqc    = runtime.refresh_scratch.dqc;
    Field3D&                                           candidate_dqi    = runtime.refresh_scratch.dqi;
    candidate_du.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    candidate_dv.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    candidate_dtheta.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    candidate_dqv.resize(
        state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    candidate_dqc.resize(
        state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    candidate_dqi.resize(
        state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    // Rebuild only prognostic scalar control volumes. In particular, the
    // collocated bottom/top and lateral closure planes are not PBL state and
    // must remain exactly zero even when held-source storage is poisoned.
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const auto cell_mass = [&](int i, int j, int k)
    {
        const double density = static_cast<double>(state.fields.rho[i][j][k]);
        if (!std::isfinite(density) || density <= 0.0)
            throw std::invalid_argument(
                "boundary-layer source conversion requires positive finite density");
        return mass_geometry.scalar_mass_kg(state.fields.rho, i, j, k);
    };
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double inverse_cell_mass = 1.0 / cell_mass(i, j, k);
                candidate_dtheta[i][j][k] = static_cast<float>(
                    runtime.held_theta_inventory_rate_kgk_s[i][j][k] *
                    inverse_cell_mass);
                candidate_dqv[i][j][k] = static_cast<float>(
                    runtime.held_vapor_mass_rate_kg_s[i][j][k] *
                    inverse_cell_mass);
                if (!runtime.config.clear_sky_only)
                {
                    candidate_dqc[i][j][k] =
                        static_cast<float>(
                            runtime.held_cloud_liquid_mass_rate_kg_s[i][j][k]
                            * inverse_cell_mass);
                    candidate_dqi[i][j][k] =
                        static_cast<float>(
                            runtime.held_cloud_ice_mass_rate_kg_s[i][j][k]
                            * inverse_cell_mass);
                }
                if (!state.grid.geometry.staggered)
                {
                    candidate_du[i][j][k] = static_cast<float>(
                        runtime.held_u_force_n[i][j][k] * inverse_cell_mass);
                    candidate_dv[i][j][k] = static_cast<float>(
                        runtime.held_v_force_n[i][j][k] * inverse_cell_mass);
                }
                if (!std::isfinite(candidate_dtheta[i][j][k]) ||
                    !std::isfinite(candidate_dqv[i][j][k]) ||
                    !std::isfinite(candidate_dqc.empty()
                        ? 0.0f : candidate_dqc[i][j][k]) ||
                    !std::isfinite(candidate_dqi.empty()
                        ? 0.0f : candidate_dqi[i][j][k]) ||
                    !std::isfinite(candidate_du[i][j][k]) ||
                    !std::isfinite(candidate_dv[i][j][k]))
                {
                    throw std::runtime_error(
                        "boundary-layer held source overflowed reconstructed tendency precision");
                }
            }

    if (state.grid.geometry.staggered)
    {
        const bool periodic_x =
            tmv::require_run_configuration(state).lateral_bc_west == tmv::init::config::LateralBC::Periodic
            || tmv::require_run_configuration(state).lateral_bc_east == tmv::init::config::LateralBC::Periodic;
        const bool periodic_y =
            tmv::require_run_configuration(state).lateral_bc_south == tmv::init::config::LateralBC::Periodic
            || tmv::require_run_configuration(state).lateral_bc_north == tmv::init::config::LateralBC::Periodic;
        if (state.grid.nr >= 3)
        {
            const int face_end = periodic_x ? domain.i_end : domain.i_end - 1;
            for (int i = domain.i_begin; i < face_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const int right = i + 1 < domain.i_end
                            ? i + 1 : domain.i_begin;
                        const double force = 0.5 * (
                            static_cast<double>(runtime.held_u_force_n[i][j][k])
                          + static_cast<double>(runtime.held_u_force_n[right][j][k]));
                        const double mass = mass_geometry.u_dual_mass_kg(
                            state.fields.rho, i, j, k, right);
                        candidate_du[i][j][k] =
                            static_cast<float>(force / mass);
                        if (!std::isfinite(candidate_du[i][j][k]))
                            throw std::runtime_error(
                                "boundary-layer held momentum overflowed reconstructed tendency precision");
                    }
        }
        if (state.grid.nth >= 3)
        {
            const int face_end = periodic_y ? domain.j_end : domain.j_end - 1;
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < face_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const int north = j + 1 < domain.j_end
                            ? j + 1 : domain.j_begin;
                        const double force = 0.5 * (
                            static_cast<double>(runtime.held_v_force_n[i][j][k])
                          + static_cast<double>(runtime.held_v_force_n[i][north][k]));
                        const double mass = mass_geometry.v_dual_mass_kg(
                            state.fields.rho, i, j, k, north);
                        candidate_dv[i][j][k] =
                            static_cast<float>(force / mass);
                        if (!std::isfinite(candidate_dv[i][j][k]))
                            throw std::runtime_error(
                                "boundary-layer held momentum overflowed reconstructed tendency precision");
                    }
        }
    }

    std::swap(state.tendencies.du_dt_pbl, candidate_du);
    std::swap(state.tendencies.dv_dt_pbl, candidate_dv);
    std::swap(state.tendencies.dtheta_dt_pbl, candidate_dtheta);
    std::swap(state.tendencies.dqv_dt_pbl, candidate_dqv);
    if (state.tendencies.dqc_dt_pbl)
        std::swap(*state.tendencies.dqc_dt_pbl, candidate_dqc);
    if (state.tendencies.dqi_dt_pbl)
        std::swap(*state.tendencies.dqi_dt_pbl, candidate_dqi);
}

void rebuild_boundary_layer_tendencies_after_domain_recycling(
    const SimulationStateView& state)
{
    BoundaryLayerRuntimeView runtime = boundary_layer_runtime_for(state);
    if (!runtime.scheme || runtime.scheme->name() == "none")
        return;
    if (runtime.updated_this_step)
        throw std::invalid_argument(
            "recycled boundary-layer state must clear its cadence event flag before tendency rebuild");
    prepare_boundary_layer_tendencies_for_application(state);
}

const char* boundary_layer_domain_recycling_finalize_algorithm_id() noexcept
{
    return "live_density_pbl_source_reconstruction_v1";
}

void commit_boundary_layer_momentum_work(const SimulationStateView&             state,
                                         const BoundaryLayerMomentumWorkReport& report)
{
    BoundaryLayerRuntimeView runtime = boundary_layer_runtime_for(state);
    if (!runtime.scheme || runtime.scheme->name() == "none")
        return;
    if (!std::isfinite(report.requested_linear_work_j) || !std::isfinite(report.finite_step_correction_j) ||
        !std::isfinite(report.storage_rounding_j) || !std::isfinite(report.committed_work_j))
    {
        throw std::runtime_error("boundary-layer momentum work attribution is non-finite");
    }
    const std::array<double, 4> cumulative{
        runtime.cumulative_requested_momentum_work_j + report.requested_linear_work_j,
        runtime.cumulative_finite_step_momentum_correction_j + report.finite_step_correction_j,
        runtime.cumulative_momentum_storage_rounding_j + report.storage_rounding_j,
        runtime.cumulative_committed_momentum_work_j + report.committed_work_j};
    if (std::any_of(cumulative.begin(), cumulative.end(), [](double value) { return !std::isfinite(value); }) ||
        runtime.momentum_work_step_count == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error("boundary-layer cumulative momentum work overflowed");
    }
    runtime.last_momentum_work                           = report;
    runtime.cumulative_requested_momentum_work_j         = cumulative[0];
    runtime.cumulative_finite_step_momentum_correction_j = cumulative[1];
    runtime.cumulative_momentum_storage_rounding_j       = cumulative[2];
    runtime.cumulative_committed_momentum_work_j         = cumulative[3];
    ++runtime.momentum_work_step_count;
}

void step_boundary_layer(double current_time)
{
    step_boundary_layer(::global_simulation_state(), current_time);
}

// True only when the most recent call crossed the PBL cadence boundary.
bool boundary_layer_updated_this_step()
{
    return global_boundary_layer_runtime_state.updated_this_step;
}

bool boundary_layer_updated_this_step(const SimulationStateView& state)
{
    return state.boundary_layer_runtime
        ? state.boundary_layer_runtime->updated_this_step
        : tmv::require_level_runtime(state.boundary_layer_runtime, "boundary_layer_runtime").updated_this_step;
}

} // namespace tmv
