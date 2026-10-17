#include "radar/runtime.hpp"

#include "core/runtime/runtime_config.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "radar/radar_base.hpp"
#include "radar/radar.hpp"
#include "radar/factory.hpp"
#include "util/log.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

std::unique_ptr<RadarSchemeBase>& radar_scheme =
    tmv::global_radar_runtime_state.selected_scheme;
RadarProductSelection global_radar_product_selection;

void tmv::RadarRuntimeState::reset() noexcept
{
    selected_scheme.reset();
    selected_config = {};
    product_selection = {};
    selected_output = {};
    polarimetric_output = {};
    reflectivity_operator.reset();
    velocity_operator.reset();
    zdr_operator.reset();
    nr = -1;
    nth = -1;
    nz = -1;
    radar_x = 0.0;
    radar_y = 0.0;
    radar_z = 0.0;
}

void initialize_radar_products(const RadarProductSelection& selection)
{
    return initialize_radar_products(global_simulation_state(), selection);
}

void initialize_radar_products(const SimulationStateView& execution_state, const RadarProductSelection& selection)
{
    tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").reset();
    tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").product_selection = selection;
    tmv::log_info("Initialized radar products: reflectivity=",
                  selection.reflectivity,
                  " radial_velocity=",
                  selection.radial_velocity,
                  " zdr=",
                  selection.zdr,
                  " kdp=",
                  selection.kdp,
                  " rhohv=",
                  selection.rhohv);
}

void initialize_radar(const std::string& scheme_name)
{
    return initialize_radar(global_simulation_state(), scheme_name);
}

void initialize_radar(const SimulationStateView& execution_state, const std::string& scheme_name)
{
    try
    {
        tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_scheme = create_radar_scheme(scheme_name);

        RadarConfig& config = tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_config;
        config = {};
        config.scheme_id = scheme_name;
        config.operator_tier = "fast_da";
        if (scheme_name == "zdr")
        {
            config.operator_tier = "polarimetric_fo";
        }

        config.has_qr = true;
        config.has_qs = true;
        config.has_qg = true;
        config.has_qh = true;
        config.has_qi = true;

        const auto state = execution_state;
        tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_scheme->initialize(
            config, state.grid.nr, state.grid.nth, state.grid.nz);
        // Product operators allocate only the fields they own into retained
        // candidate storage. Do not eagerly allocate all eleven RadarOut
        // volumes for the production reflectivity-only path.
        tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_output = {};
        tmv::log_info("Initialized radar scheme: ", scheme_name);
    }
    catch (const std::runtime_error& e)
    {
        tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_scheme.reset();
        tmv::log_error("Error initializing radar: ", e.what());
        throw;
    }
}

void calculate_radar_observables()
{
    return calculate_radar_observables(global_simulation_state());
}

void calculate_radar_observables(const SimulationStateView& execution_state)
{
    const RadarProductSelection selection =
        tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").product_selection;
    if (!selection.reflectivity && !selection.radial_velocity &&
        !selection.zdr && !selection.kdp && !selection.rhohv)
    {
        execution_state.diagnostics->radar_reflectivity.release_storage();
        tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_output = {};
        return;
    }

    const auto state = execution_state;
    const auto& fields = state.fields;
    Field3D& radar_reflectivity =
        execution_state.diagnostics->radar_reflectivity;
    const int nr = state.grid.nr;
    const int nth = state.grid.nth;
    const int nz = state.grid.nz;
    if (radar_reflectivity.size_r() != nr || radar_reflectivity.size_th() != nth || radar_reflectivity.size_z() != nz)
        radar_reflectivity.resize(nr, nth, nz, 0.0f);

    RadarStateView state_view;
    state_view.NR = nr;
    state_view.NTH = nth;
    state_view.NZ = nz;
    state_view.staggering = state.grid.geometry.staggered
        ? StaggerType::CGrid : StaggerType::Collocated;
    state_view.geometry = &state.grid.geometry;
    state_view.terrain_metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    const auto radar_grid_origin =
        tmv::require_level_runtime(execution_state.domain_frame, "domain frame").earth_origin_at(execution_state.time_s);
    state_view.grid_origin_earth_x_m = radar_grid_origin.x_m;
    state_view.grid_origin_earth_y_m = radar_grid_origin.y_m;

    state_view.u = &fields.u;
    state_view.v = &fields.v;
    state_view.w = &fields.w;
    state_view.rho = &fields.rho;

    state_view.qr = &fields.qr;
    state_view.qs = &fields.qs;
    state_view.qg = &fields.qg;
    state_view.qh = &fields.qh;
    state_view.qi = &fields.qi;

    RadarOut& radar_out = tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime").selected_output;

    try
    {
        RadarSystem::compute_observables(
            tmv::require_level_runtime(execution_state.radar_runtime, "radar_runtime"), selection, state_view, radar_out);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(
            std::string("radar observable compute failed: ") + e.what());
    }

    constexpr double radar_linear_max = 1.0e12;
    const PhysicalCellDomain domain =
        physical_scalar_domain(nr, nth, nz, state_view.staggering);
    int invalid_reflectivity = 0;

    if (!selection.reflectivity)
    {
        radar_reflectivity.release_storage();
        return;
    }

    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(nr) * nth * nz >= 32768) reduction(+:invalid_reflectivity)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double value = radar_out.Ze_linear[i][j][k];
                if (!std::isfinite(value) || value < 0.0 || value > radar_linear_max)
                    ++invalid_reflectivity;
            }
        }
    }

    if (invalid_reflectivity > 0)
    {
        throw std::runtime_error(
            "radar reflectivity operator produced " +
            std::to_string(invalid_reflectivity) +
            " non-finite, negative, or out-of-contract physical samples");
    }

    // Publish only after the complete candidate has passed validation. Closure
    // storage is deterministic zero and can never preserve an earlier frame.
    radar_reflectivity.fill(0.0f);
    #pragma omp parallel for collapse(2) if(static_cast<std::size_t>(nr) * nth * nz >= 32768)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                radar_reflectivity[i][j][k] =
                    static_cast<float>(radar_out.Ze_linear[i][j][k]);
            }
        }
    }
}

void calculate_radar_reflectivity()
{
    return calculate_radar_reflectivity(global_simulation_state());
}

void calculate_radar_reflectivity(const SimulationStateView& execution_state)
{
    calculate_radar_observables(execution_state);
}
