#include <catch2/catch.hpp>

#include "chaos/factory.hpp"
#include "core/infra/nested_execution.hpp"
#include "core/infra/physical_constants.hpp"
#include "dynamics/factory.hpp"
#include "microphysics/runtime.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "radar/factory.hpp"
#include "radiation/runtime.hpp"
#include "terrain/factory.hpp"
#include "turbulence/runtime.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace
{

void fill_primary(PrognosticFieldStorage& fields,
                  int nr, int nth, int nz)
{
    const auto allocate = [&](Field3D& field, float value) {
        field.resize(nr, nth, nz, value);
    };
    allocate(fields.rho, 0.0f);
    allocate(fields.p, 0.0f);
    allocate(fields.u, 2.0f);
    allocate(fields.v, -1.0f);
    allocate(fields.w, 0.0f);
    allocate(fields.tracer, 0.0f);
    allocate(fields.theta, 0.0f);
    allocate(fields.qv, 0.0f);
    allocate(fields.qc, 0.0f);
    allocate(fields.qr, 0.0f);
    allocate(fields.qi, 0.0f);
    allocate(fields.qs, 0.0f);
    allocate(fields.qg, 0.0f);
    allocate(fields.qh, 0.0f);
    allocate(fields.tke, 0.1f);
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double pressure = k == 0 ? 90000.0 : 85000.0;
                const double theta = k == 0 ? 300.0 : 302.0;
                const double temperature = theta * std::pow(
                    pressure /
                        physical_constants::reference_pressure_pa,
                    kappa);
                fields.p[i][j][k] = static_cast<float>(pressure);
                fields.theta[i][j][k] = static_cast<float>(theta);
                fields.rho[i][j][k] = static_cast<float>(
                    pressure /
                    (physical_constants::gas_constant_dry_air_jkgk *
                     temperature));
                fields.qv[i][j][k] = k == 0 ? 0.01f : 0.006f;
            }
}

SimulationStorageState prepared_nested_state()
{
    SimulationStorageState state(
        GridStorageState(8, 8, 2, 1000.0, 300.0, 2.0, 800.0));
    state.grid.geometry.initialize(
        8, 8, 2, 1000.0, 300.0, 800.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.grid.geometry.z_faces = {0.0, 200.0, 550.0};
    state.grid.geometry.z = {100.0, 375.0};
    state.process_grid_metrics.dx = 1000.0;
    state.process_grid_metrics.dy = 800.0;
    state.process_grid_metrics.dz = {200.0, 350.0};
    state.process_grid_metrics.z_int = state.grid.geometry.z_faces;
    state.time_s = 2.0;
    state.base.p0 = {90000.0, 85000.0};
    state.base.theta0 = {300.0, 302.0};
    state.base.rho0.resize(2);
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    for (int k = 0; k < 2; ++k)
    {
        const double temperature = state.base.theta0[k] * std::pow(
            state.base.p0[k] / physical_constants::reference_pressure_pa,
            kappa);
        state.base.rho0[k] = state.base.p0[k] /
            (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    state.base.qv0 = {0.01, 0.006};
    state.base.u0 = {2.0, 3.0};
    state.base.v0 = {-1.0, 0.0};
    fill_primary(state.fields, 8, 8, 2);

    RadiationConfig radiation;
    radiation.scheme_id = "none";
    initialize_radiation(state.runtime.radiation, "none", radiation);
    TurbulenceConfig turbulence;
    turbulence.scheme_id = "none";
    initialize_turbulence(state.runtime.turbulence, "none", turbulence);
    BoundaryLayerConfig pbl;
    pbl.scheme_id = "none";
    initialize_boundary_layer(
        state.runtime.boundary_layer, "none", pbl, SurfaceConfig{});
    state.runtime.microphysics.scheme = tmv::create_microphysics_scheme(
        "none", state.configuration.microphysics_config);

    state.runtime.chaos.config.scheme_id = "none";
    state.runtime.chaos.scheme = tmv::chaos::create_chaos_scheme("none");
    state.runtime.chaos.scheme->initialize(
        state.runtime.chaos.config, state.process_grid_metrics);
    state.runtime.terrain.config.scheme_id = "none";
    state.runtime.terrain.scheme = create_terrain_scheme("none");
    state.runtime.terrain.scheme->initialize(state.runtime.terrain.config);
    initialize_advection(state.runtime.advection, "tvd", AdvectionConfig{});
    DiffusionConfig diffusion;
    diffusion.scheme_id = "explicit";
    initialize_diffusion(
        state.runtime.diffusion, "explicit", diffusion);
    TimeSteppingConfig time;
    time.scheme_id = "split_explicit";
    time.split_acoustic = true;
    time.n_acoustic_substeps = 3;
    initialize_time_stepping(
        state.runtime.time_stepping, "split_explicit", time);
    state.runtime.dynamics.scheme =
        create_dynamics_scheme("klemp_wilhelmson_split_cartesian_cgrid");
    REQUIRE(state.runtime.dynamics.scheme);

    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = 6;
    input.parent_cells_y = 6;
    input.parent_levels_z = 2;
    input.parent_dx_m = 1000.0;
    input.parent_dy_m = 800.0;
    input.parent_dt_s = 2.0;
    input.parent_ratio_x = 2;
    input.parent_ratio_y = 2;
    input.parent_time_step_ratio = 2;
    input.child_origin_parent_cell_i = 2;
    input.child_origin_parent_cell_j = 2;
    input.child_cells_x = 4;
    input.child_cells_y = 4;
    input.child_levels_z = 2;
    input.relaxation_width_child_cells = 1;
    input.boundary_stencil_parent_cells = 1;
    prepare_nested_execution_relationship(
        state, tmv::domain::prepare_one_way_nest_relationship(input), 0.0);
    initialize_nested_shared_level_reference_state(state);
    fill_primary(state.nested_execution.child.fields, 6, 6, 2);
    state.nested_execution.physical_primary_state_ready = true;
    state.nested_execution.parent_boundary_state_ready = true;
    state.nested_execution.interface_source.available = true;
    state.nested_execution.interface_source.application_count = 1;
    require_valid_nested_execution_relationship(state);
    return state;
}

class GlobalStorageRestore
{
public:
    GlobalStorageRestore()
        : saved_(std::move(global_simulation_storage))
    {
    }
    ~GlobalStorageRestore()
    {
        global_simulation_storage = std::move(saved_);
    }

private:
    SimulationStorageState saved_;
};

} // namespace

TEST_CASE("nested child modules initialize from selection without parent history",
          "[core][nesting][runtime][modules]")
{
    GlobalStorageRestore restore;
    global_simulation_storage = prepared_nested_state();
    auto& state = global_simulation_storage;
    state.runtime.boundary_conditions.radiative_u_west = {41.0f, 42.0f};
    state.runtime.radiation.applied_step_count = 17;
    state.runtime.turbulence.momentum_work_step_count = 19;

    REQUIRE_NOTHROW(initialize_nested_child_level_runtime(state));
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(state));
    const auto& child = state.nested_execution.child;
    REQUIRE(child.runtime.boundary_conditions.radiative_u_west.empty());
    REQUIRE(child.runtime.radiation.applied_step_count == 0);
    REQUIRE(child.runtime.turbulence.momentum_work_step_count == 0);
    REQUIRE(child.runtime.radiation.last_update_time_s == child.time_s);
    REQUIRE(child.runtime.boundary_layer.last_update_time_s == child.time_s);
    REQUIRE(child.tendencies.dtheta_dt_rad.size_r() == child.grid.nr);
    REQUIRE(child.tendencies.dtheta_dt_pbl.size_r() == child.grid.nr);
    REQUIRE(child.surface.ustar.size() ==
            static_cast<std::size_t>(child.grid.nr * child.grid.nth));
    REQUIRE(state.runtime.boundary_conditions.radiative_u_west ==
            std::vector<float>{41.0f, 42.0f});
    REQUIRE(state.runtime.radiation.applied_step_count == 17);
    REQUIRE(state.runtime.turbulence.momentum_work_step_count == 19);
    REQUIRE(child.runtime.dynamics.scheme.get() !=
            state.runtime.dynamics.scheme.get());
}

TEST_CASE("nested child module initialization rejects without mutation",
          "[core][nesting][runtime][transaction]")
{
    GlobalStorageRestore restore;
    global_simulation_storage = prepared_nested_state();
    auto& state = global_simulation_storage;
    auto& child = state.nested_execution.child;
    child.runtime.boundary_conditions.radiative_u_west = {7.0f};
    child.surface.ustar = {8.0f};
    state.runtime.dynamics.scheme.reset();

    REQUIRE_THROWS_WITH(
        initialize_nested_child_level_runtime(state),
        Catch::Matchers::Contains("dynamics is not initialized"));
    REQUIRE(child.runtime.boundary_conditions.radiative_u_west ==
            std::vector<float>{7.0f});
    REQUIRE(child.surface.ustar == std::vector<float>{8.0f});
}

TEST_CASE("nested child initializes active coupled column and SGS selections",
          "[core][nesting][runtime][physics]")
{
    GlobalStorageRestore restore;
    global_simulation_storage = prepared_nested_state();
    auto& state = global_simulation_storage;

    RadiationConfig radiation;
    radiation.scheme_id = "simple_grey";
    radiation.do_sw = false;
    radiation.do_lw = true;
    initialize_radiation(
        state.runtime.radiation, "simple_grey", radiation);

    BoundaryLayerConfig pbl;
    pbl.scheme_id = "slab";
    pbl.apply_surface_fluxes = true;
    SurfaceConfig surface;
    surface.Tsfc = 290.0;
    surface.qsfc = 0.01;
    initialize_boundary_layer(
        state.runtime.boundary_layer, "slab", pbl, surface);

    TurbulenceConfig turbulence;
    turbulence.scheme_id = "smagorinsky";
    turbulence.mode = "3d";
    initialize_turbulence(
        state.runtime.turbulence, "smagorinsky", turbulence);
    state.configuration.microphysics_scheme = "kessler";
    state.runtime.microphysics.scheme = tmv::create_microphysics_scheme(
        "kessler", state.configuration.microphysics_config);

    state.runtime.radar.selected_config.scheme_id = "reflectivity";
    state.runtime.radar.selected_config.has_qr = true;
    state.runtime.radar.selected_scheme =
        tmv::create_radar_scheme("reflectivity");
    REQUIRE(state.runtime.radar.selected_scheme);
    state.runtime.radar.selected_scheme->initialize(
        state.runtime.radar.selected_config,
        state.grid.nr, state.grid.nth, state.grid.nz);
    state.runtime.radar.nr = state.grid.nr;
    state.runtime.radar.nth = state.grid.nth;
    state.runtime.radar.nz = state.grid.nz;

    state.runtime.radiation.applied_step_count = 9;
    state.runtime.radiation.held_cell_power_w.resize(
        state.grid.nr, state.grid.nth, state.grid.nz, 37.0f);
    state.runtime.boundary_layer.momentum_work_step_count = 11;
    state.runtime.turbulence.momentum_work_step_count = 13;
    state.runtime.chaos.noise_elapsed_s = 17.0;
    state.runtime.advection.thermodynamic_boundary_report_step_count = 19;
    state.runtime.rayleigh.upper_initialized = true;
    state.runtime.rayleigh.alpha = {23.0, 29.0};

    REQUIRE_NOTHROW(initialize_nested_child_level_runtime(state));
    REQUIRE_NOTHROW(require_nested_child_level_runtime_ready(state));
    const auto& child = state.nested_execution.child;
    REQUIRE(child.runtime.radiation.scheme->name() == "simple_grey");
    REQUIRE(child.runtime.boundary_layer.scheme->name() == "slab");
    REQUIRE(child.runtime.turbulence.scheme->name() == "smagorinsky");
    REQUIRE(child.runtime.microphysics.scheme->get_scheme_name() ==
            "kessler");
    REQUIRE(child.runtime.radar.selected_scheme);
    REQUIRE(child.runtime.radar.selected_scheme->name() == "reflectivity");
    REQUIRE(child.runtime.radar.nr == child.grid.nr);
    REQUIRE(child.runtime.radar.nth == child.grid.nth);
    REQUIRE(child.runtime.radar.nz == child.grid.nz);
    REQUIRE(child.runtime.radiation.held_cell_power_w.size_r() ==
            child.grid.nr);
    REQUIRE(child.runtime.boundary_layer.held_u_force_n.size_r() ==
            child.grid.nr);
    REQUIRE(child.runtime.boundary_layer.held_theta_inventory_rate_kgk_s
                .size_r() == child.grid.nr);
    REQUIRE(child.runtime.radiation.applied_step_count == 0);
    REQUIRE(child.runtime.boundary_layer.momentum_work_step_count == 0);
    REQUIRE(child.runtime.turbulence.momentum_work_step_count == 0);
    REQUIRE(child.runtime.chaos.noise_elapsed_s == 0.0);
    REQUIRE(
        child.runtime.advection.thermodynamic_boundary_report_step_count == 0);
    REQUIRE_FALSE(child.runtime.rayleigh.upper_initialized);
    REQUIRE(child.runtime.rayleigh.alpha.empty());
    REQUIRE(state.runtime.radiation.applied_step_count == 9);
    REQUIRE(state.runtime.radiation.held_cell_power_w(0, 0, 0) == 37.0f);
    REQUIRE(state.runtime.boundary_layer.momentum_work_step_count == 11);
    REQUIRE(state.runtime.turbulence.momentum_work_step_count == 13);
    REQUIRE(state.runtime.chaos.noise_elapsed_s == 17.0);
    REQUIRE(
        state.runtime.advection.thermodynamic_boundary_report_step_count ==
        19);
    REQUIRE(state.runtime.rayleigh.upper_initialized);
    REQUIRE(state.runtime.rayleigh.alpha == std::vector<double>{23.0, 29.0});
}
