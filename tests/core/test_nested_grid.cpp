#include <chrono>
#include <iostream>
#include "core/runtime/atmospheric_level.hpp"
#include "core/field/field_inventory.hpp"
#include "catch2/catch.hpp"
#include "core/infra/nested_execution.hpp"
#include "core/infra/nested_feedback.hpp"
#include "core/infra/nested_grid.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/energy_budget.hpp"
#include "dynamics/coriolis.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

void fill_linear(Field3D& field, float offset)
{
    for (int i = 0; i < static_cast<int>(field.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(field.size_th()); ++j)
        {
            for (int k = 0; k < static_cast<int>(field.size_z()); ++k)
            {
                field[i][j][k] =
                    offset + static_cast<float>(i) + 10.0f * static_cast<float>(j) + 100.0f * static_cast<float>(k);
            }
        }
    }
}

TEST_CASE("two-way child inventory owns every staggered boundary and top face",
          "[core][nested_grid][two_way][inventory]")
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = 8;
    input.parent_cells_y                = 8;
    input.parent_levels_z               = 3;
    input.parent_dx_m                   = 100.0;
    input.parent_dy_m                   = 100.0;
    input.parent_dt_s                   = 1.0;
    input.parent_ratio_x                = 2;
    input.parent_ratio_y                = 2;
    input.parent_time_step_ratio        = 2;
    input.child_origin_parent_cell_i    = 3;
    input.child_origin_parent_cell_j    = 3;
    input.child_cells_x                 = 4;
    input.child_cells_y                 = 4;
    input.child_levels_z                = 3;
    input.relaxation_width_child_cells  = 1;
    input.boundary_stencil_parent_cells = 1;
    const auto             relationship = tmv::domain::prepare_one_way_nest_relationship(input);
    PrognosticFieldStorage child;
    for (Field3D* field : {&child.rho, &child.theta, &child.p, &child.u, &child.v, &child.w})
        field->resize(6, 6, 3, 0.0f);
    child.rho.fill(1.0f);
    child.theta.fill(300.0f);
    child.p.fill(90000.0f);
    child.u[0][1][2]     = 2.0f;
    child.u[4][1][2]     = 3.0f;
    child.v[1][0][2]     = -4.0f;
    child.v[1][4][2]     = 5.0f;
    child.w[1][1][2]     = 4.0f;
    const auto inventory = nested_two_way_child_inventory_flat(relationship, {100.0, 100.0, 100.0}, child);
    REQUIRE(inventory[2] == Approx(1.25e6));
    REQUIRE(inventory[3] == Approx(0.25e6));
    REQUIRE(inventory[4] == Approx(0.5e6));

    auto invalid         = child;
    invalid.rho[2][2][1] = -1.0f;
    REQUIRE_THROWS(nested_two_way_child_inventory_flat(relationship, {100.0, 100.0, 100.0}, invalid));
    invalid            = child;
    invalid.u[0][1][2] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS(nested_two_way_child_inventory_flat(relationship, {100.0, 100.0, 100.0}, invalid));
    REQUIRE_THROWS(nested_two_way_child_inventory_flat(relationship, {100.0, 0.0, 100.0}, child));
    auto invalid_relationship       = relationship;
    invalid_relationship.child_dx_m = -50.0;
    REQUIRE_THROWS(nested_two_way_child_inventory_flat(invalid_relationship, {100.0, 100.0, 100.0}, child));
}

TEST_CASE("two-way feedback restricts and refluxes only local coarse owners", "[core][nested_grid][two_way][reflux]")
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = 8;
    input.parent_cells_y                = 8;
    input.parent_levels_z               = 3;
    input.parent_dx_m                   = 100.0;
    input.parent_dy_m                   = 100.0;
    input.parent_dt_s                   = 1.0;
    input.parent_ratio_x                = 2;
    input.parent_ratio_y                = 2;
    input.parent_time_step_ratio        = 2;
    input.child_origin_parent_cell_i    = 3;
    input.child_origin_parent_cell_j    = 3;
    input.child_cells_x                 = 4;
    input.child_cells_y                 = 4;
    input.child_levels_z                = 3;
    input.relaxation_width_child_cells  = 1;
    input.boundary_stencil_parent_cells = 1;
    const auto relationship             = tmv::domain::prepare_one_way_nest_relationship(input);

    PrognosticFieldStorage parent;
    PrognosticFieldStorage child;
    for (Field3D* field : {&parent.rho,
                           &parent.theta,
                           &parent.p,
                           &parent.u,
                           &parent.v,
                           &parent.w,
                           &parent.qv,
                           &parent.qc,
                           &parent.qr,
                           &parent.qi,
                           &parent.qs,
                           &parent.qg,
                           &parent.qh,
                           &parent.tracer,
                           &parent.tke})
        field->resize(10, 10, 3, 0.0f);
    for (Field3D* field : {&child.rho,
                           &child.theta,
                           &child.p,
                           &child.u,
                           &child.v,
                           &child.w,
                           &child.qv,
                           &child.qc,
                           &child.qr,
                           &child.qi,
                           &child.qs,
                           &child.qg,
                           &child.qh,
                           &child.tracer,
                           &child.tke})
        field->resize(6, 6, 3, 0.0f);
    parent.rho.fill(1.0f);
    parent.theta.fill(300.0f);
    parent.p.fill(90000.0f);
    parent.u.fill(1.0f);
    parent.v.fill(-0.5f);
    parent.w.fill(0.25f);
    child.rho.fill(1.0f);
    child.theta.fill(300.0f);
    child.p.fill(90000.0f);
    child.u.fill(2.0f);
    child.v.fill(-1.0f);
    child.w.fill(0.5f);
    for (int i = 1; i <= 4; ++i)
        for (int j = 1; j <= 4; ++j)
            for (int k = 0; k < 3; ++k)
            {
                child.rho[i][j][k]   = 1.0f + 0.01f * (i + j + k);
                child.theta[i][j][k] = 300.0f + 0.1f * (i - j + k);
            }

    tmv::CartesianDryMassInterfaceFluxes parent_mass;
    tmv::CartesianDryMassInterfaceFluxes child_mass;
    tmv::CartesianThetaInterfaceFluxes   parent_theta;
    tmv::CartesianThetaInterfaceFluxes   child_theta;
    const auto                           allocate_scalar = [](auto& fluxes)
    {
        fluxes.west.assign(6, 0.0);
        fluxes.east.assign(6, 0.0);
        fluxes.south.assign(6, 0.0);
        fluxes.north.assign(6, 0.0);
        fluxes.accumulated_dt_s   = 1.0;
        fluxes.contribution_count = 1;
    };
    allocate_scalar(parent_mass);
    allocate_scalar(child_mass);
    allocate_scalar(parent_theta);
    allocate_scalar(child_theta);
    child_mass.east[1]  = 8.0;
    child_theta.east[1] = 2400.0;

    tmv::CartesianMomentumInterfaceFluxes parent_advective;
    tmv::CartesianMomentumInterfaceFluxes child_advective;
    tmv::CartesianMomentumInterfaceFluxes parent_pressure;
    tmv::CartesianMomentumInterfaceFluxes child_pressure;
    const auto                            allocate_momentum = [](auto& fluxes)
    {
        for (std::size_t component = 0; component < 3; ++component)
        {
            fluxes.west.component[component].assign(4, 0.0);
            fluxes.east.component[component].assign(4, 0.0);
            fluxes.south.component[component].assign(4, 0.0);
            fluxes.north.component[component].assign(4, 0.0);
        }
        fluxes.accumulated_dt_s   = 1.0;
        fluxes.contribution_count = 1;
    };
    allocate_momentum(parent_advective);
    allocate_momentum(child_advective);
    allocate_momentum(parent_pressure);
    allocate_momentum(child_pressure);
    child_advective.east.component[0][1] = 5.0;

    const std::array<double, 5> interface_source = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::array<double, 5> dynamics_source  = {0.0, 0.0, 6.0, 7.0, 8.0};
    const std::array<double, 5> dynamics_storage = {0.0, 0.0, 1000.0, 1500.0, 2000.0};
    const std::array<double, 5> child_transport  = {-8.0, -2400.0, -5.0, 0.0, 0.0};
    const auto child_end_inventory   = nested_two_way_child_inventory_flat(relationship, {100.0, 100.0, 100.0}, child);
    auto       child_begin_inventory = child_end_inventory;
    for (std::size_t currency = 0; currency < 5; ++currency)
        child_begin_inventory[currency] -= child_transport[currency] + interface_source[currency] +
                                           dynamics_source[currency] + dynamics_storage[currency];

    const PrognosticFieldStorage before = parent;
    NestedTwoWayFeedbackReportV1 report;
    apply_nested_two_way_feedback_flat(relationship,
                                       {100.0, 100.0, 100.0},
                                       parent_mass,
                                       child_mass,
                                       parent_theta,
                                       child_theta,
                                       parent_advective,
                                       parent_pressure,
                                       child_advective,
                                       child_pressure,
                                       interface_source,
                                       dynamics_source,
                                       dynamics_storage,
                                       child_begin_inventory,
                                       parent,
                                       child,
                                       report);

    REQUIRE(report.complete);
    REQUIRE(report.restricted_scalar_cells == 12);
    REQUIRE(report.interface_mismatch[0] == Approx(8.0));
    REQUIRE(report.interface_mismatch[1] == Approx(2400.0));
    REQUIRE(report.interface_mismatch[2] == Approx(5.0));
    REQUIRE(report.applied_reflux[0] == Approx(8.0));
    REQUIRE(report.applied_reflux[1] == Approx(2400.0));
    REQUIRE(report.applied_reflux[2] == Approx(5.0));
    for (std::size_t currency = 0; currency < 5; ++currency)
    {
        REQUIRE(report.child_interface_source[currency] == interface_source[currency]);
        REQUIRE(report.child_dynamics_source[currency] == dynamics_source[currency]);
        REQUIRE(report.child_dynamics_storage[currency] == dynamics_storage[currency]);
        REQUIRE(report.child_interface_transport[currency] == Approx(child_transport[currency]));
        REQUIRE(std::abs(report.child_composite_residual[currency]) <= report.child_composite_tolerance[currency]);
    }
    REQUIRE(parent.rho[1][1][0] == before.rho[1][1][0]);
    REQUIRE(parent.theta[1][1][0] == before.theta[1][1][0]);
    REQUIRE(parent.u[1][1][0] == before.u[1][1][0]);
    REQUIRE(parent.rho[4][4][0] != before.rho[4][4][0]);
    REQUIRE(parent.u[4][4][0] != before.u[4][4][0]);

    PrognosticFieldStorage       omitted_storage_parent = before;
    NestedTwoWayFeedbackReportV1 omitted_storage_report;
    REQUIRE_THROWS(apply_nested_two_way_feedback_flat(relationship,
                                                      {100.0, 100.0, 100.0},
                                                      parent_mass,
                                                      child_mass,
                                                      parent_theta,
                                                      child_theta,
                                                      parent_advective,
                                                      parent_pressure,
                                                      child_advective,
                                                      child_pressure,
                                                      interface_source,
                                                      dynamics_source,
                                                      {},
                                                      child_begin_inventory,
                                                      omitted_storage_parent,
                                                      child,
                                                      omitted_storage_report));
    REQUIRE(std::memcmp(omitted_storage_parent.u.data(), before.u.data(), before.u.size() * sizeof(float)) == 0);

    PrognosticFieldStorage threaded_parent = before;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    NestedTwoWayFeedbackReportV1 threaded_report;
    apply_nested_two_way_feedback_flat(relationship,
                                       {100.0, 100.0, 100.0},
                                       parent_mass,
                                       child_mass,
                                       parent_theta,
                                       child_theta,
                                       parent_advective,
                                       parent_pressure,
                                       child_advective,
                                       child_pressure,
                                       interface_source,
                                       dynamics_source,
                                       dynamics_storage,
                                       child_begin_inventory,
                                       threaded_parent,
                                       child,
                                       threaded_report);
    for (const auto pair : {std::pair<const Field3D*, const Field3D*>{&parent.rho, &threaded_parent.rho},
                            {&parent.theta, &threaded_parent.theta},
                            {&parent.p, &threaded_parent.p},
                            {&parent.u, &threaded_parent.u},
                            {&parent.v, &threaded_parent.v},
                            {&parent.w, &threaded_parent.w}})
        REQUIRE(std::memcmp(pair.first->data(), pair.second->data(), pair.first->size() * sizeof(float)) == 0);

    PrognosticFieldStorage       rejected_parent = before;
    const PrognosticFieldStorage rejected_before = rejected_parent;
    child_pressure.accumulated_dt_s              = 0.5;
    REQUIRE_THROWS(apply_nested_two_way_feedback_flat(relationship,
                                                      {100.0, 100.0, 100.0},
                                                      parent_mass,
                                                      child_mass,
                                                      parent_theta,
                                                      child_theta,
                                                      parent_advective,
                                                      parent_pressure,
                                                      child_advective,
                                                      child_pressure,
                                                      interface_source,
                                                      dynamics_source,
                                                      dynamics_storage,
                                                      child_begin_inventory,
                                                      rejected_parent,
                                                      child,
                                                      threaded_report));
    REQUIRE(std::memcmp(rejected_parent.rho.data(),
                        rejected_before.rho.data(),
                        rejected_parent.rho.size() * sizeof(float)) == 0);
    REQUIRE(std::memcmp(rejected_parent.u.data(), rejected_before.u.data(), rejected_parent.u.size() * sizeof(float)) ==
            0);

    child_pressure.accumulated_dt_s = 1.0;
    auto short_parent_mass          = parent_mass;
    auto short_child_mass           = child_mass;
    short_parent_mass.west.resize(5);
    short_child_mass.west.resize(5);
    NestedTwoWayFeedbackReportV1 retained_report;
    retained_report.complete              = true;
    retained_report.interface_mismatch[0] = 77.0;
    rejected_parent                       = rejected_before;
    REQUIRE_THROWS(apply_nested_two_way_feedback_flat(relationship,
                                                      {100.0, 100.0, 100.0},
                                                      short_parent_mass,
                                                      short_child_mass,
                                                      parent_theta,
                                                      child_theta,
                                                      parent_advective,
                                                      parent_pressure,
                                                      child_advective,
                                                      child_pressure,
                                                      interface_source,
                                                      dynamics_source,
                                                      dynamics_storage,
                                                      child_begin_inventory,
                                                      rejected_parent,
                                                      child,
                                                      retained_report));
    REQUIRE(retained_report.interface_mismatch[0] == 77.0);
    REQUIRE(std::memcmp(rejected_parent.rho.data(),
                        rejected_before.rho.data(),
                        rejected_parent.rho.size() * sizeof(float)) == 0);

    auto nonclosing_begin = child_begin_inventory;
    nonclosing_begin[0] += 4.0 * report.child_composite_tolerance[0];
    rejected_parent = rejected_before;
    REQUIRE_THROWS(apply_nested_two_way_feedback_flat(relationship,
                                                      {100.0, 100.0, 100.0},
                                                      parent_mass,
                                                      child_mass,
                                                      parent_theta,
                                                      child_theta,
                                                      parent_advective,
                                                      parent_pressure,
                                                      child_advective,
                                                      child_pressure,
                                                      interface_source,
                                                      dynamics_source,
                                                      dynamics_storage,
                                                      nonclosing_begin,
                                                      rejected_parent,
                                                      child,
                                                      retained_report));
    REQUIRE(retained_report.interface_mismatch[0] == 77.0);
    REQUIRE(std::memcmp(rejected_parent.rho.data(),
                        rejected_before.rho.data(),
                        rejected_parent.rho.size() * sizeof(float)) == 0);

    auto invalid_child        = child;
    invalid_child.qv[5][5][2] = std::numeric_limits<float>::quiet_NaN();
    rejected_parent           = rejected_before;
    REQUIRE_THROWS(apply_nested_two_way_feedback_flat(relationship,
                                                      {100.0, 100.0, 100.0},
                                                      parent_mass,
                                                      child_mass,
                                                      parent_theta,
                                                      child_theta,
                                                      parent_advective,
                                                      parent_pressure,
                                                      child_advective,
                                                      child_pressure,
                                                      {},
                                                      {},
                                                      {},
                                                      {},
                                                      rejected_parent,
                                                      invalid_child,
                                                      retained_report));
    REQUIRE(retained_report.interface_mismatch[0] == 77.0);
    REQUIRE(std::memcmp(rejected_parent.rho.data(),
                        rejected_before.rho.data(),
                        rejected_parent.rho.size() * sizeof(float)) == 0);
    auto wet_parent = before;
    auto wet_child  = child;
    wet_parent.qv.fill(.01f);
    wet_child.qv.fill(.02f);
    std::array<tmv::CartesianScalarInterfaceFluxes, 7> water_parent, water_child;
    NestedMoistFeedbackInputs                          water_inputs;
    for (std::size_t f = 0; f < 7; ++f)
    {
        allocate_scalar(water_parent[f]);
        allocate_scalar(water_child[f]);
        water_inputs.parent_flux[f] = &water_parent[f];
        water_inputs.child_flux[f]  = &water_child[f];
    }
    const auto wet_inventory    = nested_two_way_child_inventory_flat(relationship, {100, 100, 100}, wet_child);
    water_inputs.child_begin[0] = wet_inventory[0] * double(.02f);
    NestedTwoWayFeedbackReportV1 water_report;
    apply_nested_two_way_feedback_flat(relationship,
                                       {100, 100, 100},
                                       parent_mass,
                                       child_mass,
                                       parent_theta,
                                       child_theta,
                                       parent_advective,
                                       parent_pressure,
                                       child_advective,
                                       child_pressure,
                                       interface_source,
                                       dynamics_source,
                                       dynamics_storage,
                                       child_begin_inventory,
                                       wet_parent,
                                       wet_child,
                                       water_report,
                                       &water_inputs);
    REQUIRE(water_report.moist);
    for (const auto& value : water_report.water)
        CHECK(valid_nested_water_feedback_report(value));
    wet_parent.qv.fill(0);
    water_child[0].west[0] = 1;
    water_inputs.child_begin[0] -= 1;
    std::array<Field3D, 7> child_candidates;
    child_candidates[0].resize(1, 1, 1, 77);
    const auto water_before       = wet_parent;
    const auto saved_water_report = water_report;
    water_child[6].east[0]        = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(apply_nested_two_way_feedback_flat(relationship,
                                                      {100, 100, 100},
                                                      parent_mass,
                                                      child_mass,
                                                      parent_theta,
                                                      child_theta,
                                                      parent_advective,
                                                      parent_pressure,
                                                      child_advective,
                                                      child_pressure,
                                                      interface_source,
                                                      dynamics_source,
                                                      dynamics_storage,
                                                      child_begin_inventory,
                                                      wet_parent,
                                                      wet_child,
                                                      water_report,
                                                      &water_inputs,
                                                      &child_candidates));
    CHECK(std::memcmp(wet_parent.rho.data(), water_before.rho.data(), water_before.rho.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(wet_parent.qv.data(), water_before.qv.data(), water_before.qv.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(wet_parent.u.data(), water_before.u.data(), water_before.u.size() * sizeof(float)) == 0);
    CHECK(water_report.water[0].parent.final_inventory == saved_water_report.water[0].parent.final_inventory);

    CHECK(child_candidates[0](0, 0, 0) == 77);
    CHECK(wet_child.qv(1, 1, 0) == .02f);
}

TEST_CASE("mapped 3:1 two-way feedback restricts and refluxes atomically and deterministically",
          "[core][nested_grid][two_way][active_terrain][ratio3][reflux][atomicity][threading]")
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = 8;
    input.parent_cells_y                = 8;
    input.parent_levels_z               = 3;
    input.parent_dx_m                   = 300.0;
    input.parent_dy_m                   = 240.0;
    input.parent_dt_s                   = 1.5;
    input.parent_ratio_x                = 3;
    input.parent_ratio_y                = 3;
    input.parent_time_step_ratio        = 3;
    input.child_origin_parent_cell_i    = 3;
    input.child_origin_parent_cell_j    = 3;
    input.child_cells_x                 = 6;
    input.child_cells_y                 = 6;
    input.child_levels_z                = 3;
    input.relaxation_width_child_cells  = 1;
    input.boundary_stencil_parent_cells = 1;
    const auto relationship             = tmv::domain::prepare_one_way_nest_relationship(input);

    tmv::TerrainConfig terrain;
    terrain.scheme_id               = "bell";
    terrain.coord_id                = "btf";
    terrain.horizontal_placement_id = "grid_attached";
    terrain.ztop                    = 6000.0;
    terrain.compute_derivatives     = true;
    terrain.compute_metrics         = true;
    terrain.bell.h0                 = GENERATE(0.0, 20.0);
    terrain.bell.a                  = 1200.0;
    terrain.bell.axisymmetric       = true;
    const auto geometry =
        tmv::domain::build_stationary_nested_terrain_geometry(terrain, relationship, {1000.0, -2000.0}, 1);

    PrognosticFieldStorage parent;
    PrognosticFieldStorage child;
    const auto             allocate_state = [](PrognosticFieldStorage& state, int nx, int ny)
    {
        for (Field3D* field : {&state.rho,
                               &state.theta,
                               &state.p,
                               &state.u,
                               &state.v,
                               &state.w,
                               &state.qv,
                               &state.qc,
                               &state.qr,
                               &state.qi,
                               &state.qs,
                               &state.qg,
                               &state.qh,
                               &state.tracer,
                               &state.tke})
            field->resize(nx + 2, ny + 2, 3, 0.0f);
        state.rho.fill(1.0f);
        state.theta.fill(300.0f);
        state.p.fill(90000.0f);
        state.u.fill(1.0f);
        state.v.fill(-0.5f);
        state.w.fill(0.25f);
    };
    allocate_state(parent, relationship.parent_cells_x, relationship.parent_cells_y);
    allocate_state(child, relationship.child_cells_x, relationship.child_cells_y);
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                child.rho[i][j][k]   = 1.0f + 0.002f * static_cast<float>(i + 2 * j + k);
                child.theta[i][j][k] = 300.0f + 0.05f * static_cast<float>(i - j + k);
                child.u[i][j][k]     = 2.0f + 0.01f * i;
                child.v[i][j][k]     = -1.0f + 0.01f * j;
                child.w[i][j][k]     = 0.5f + 0.01f * k;
            }

    tmv::CartesianDryMassInterfaceFluxes parent_mass;
    tmv::CartesianDryMassInterfaceFluxes child_mass;
    tmv::CartesianThetaInterfaceFluxes   parent_theta;
    tmv::CartesianThetaInterfaceFluxes   child_theta;
    const auto                           allocate_scalar = [](auto& fluxes)
    {
        fluxes.west.assign(6, 0.0);
        fluxes.east.assign(6, 0.0);
        fluxes.south.assign(6, 0.0);
        fluxes.north.assign(6, 0.0);
        fluxes.accumulated_dt_s   = 1.5;
        fluxes.contribution_count = 3;
    };
    allocate_scalar(parent_mass);
    allocate_scalar(child_mass);
    allocate_scalar(parent_theta);
    allocate_scalar(child_theta);
    child_mass.east[1]  = 8.0;
    child_theta.east[1] = 2400.0;

    tmv::CartesianMomentumInterfaceFluxes parent_advective;
    tmv::CartesianMomentumInterfaceFluxes child_advective;
    tmv::CartesianMomentumInterfaceFluxes parent_pressure;
    tmv::CartesianMomentumInterfaceFluxes child_pressure;
    const auto                            allocate_momentum = [](auto& fluxes)
    {
        for (std::size_t component = 0; component < 3; ++component)
        {
            fluxes.west.component[component].assign(4, 0.0);
            fluxes.east.component[component].assign(4, 0.0);
            fluxes.south.component[component].assign(4, 0.0);
            fluxes.north.component[component].assign(4, 0.0);
        }
        fluxes.accumulated_dt_s   = 1.5;
        fluxes.contribution_count = 3;
    };
    allocate_momentum(parent_advective);
    allocate_momentum(child_advective);
    allocate_momentum(parent_pressure);
    allocate_momentum(child_pressure);

    const std::array<double, 5> zeros{};
    const std::array<double, 5> transport       = {-8.0, -2400.0, 0.0, 0.0, 0.0};
    const auto                  end_inventory   = nested_two_way_child_inventory_mapped(geometry, child);
    auto                        begin_inventory = end_inventory;
    for (std::size_t currency = 0; currency < begin_inventory.size(); ++currency)
        begin_inventory[currency] -= transport[currency];

    const auto                   before = parent;
    NestedTwoWayFeedbackReportV1 one_report;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    apply_nested_two_way_feedback_mapped(geometry,
                                         parent_mass,
                                         child_mass,
                                         parent_theta,
                                         child_theta,
                                         parent_advective,
                                         parent_pressure,
                                         child_advective,
                                         child_pressure,
                                         zeros,
                                         zeros,
                                         zeros,
                                         begin_inventory,
                                         parent,
                                         child,
                                         one_report);
    REQUIRE(one_report.complete);
    REQUIRE(one_report.restricted_scalar_cells == 12);
    REQUIRE(one_report.interface_mismatch[0] == Approx(8.0));
    REQUIRE(one_report.interface_mismatch[1] == Approx(2400.0));
    REQUIRE(one_report.applied_reflux[0] == Approx(8.0));
    REQUIRE(one_report.applied_reflux[1] == Approx(2400.0));
    REQUIRE(parent.rho[4][4][0] != before.rho[4][4][0]);

    PrognosticFieldStorage       four_parent = before;
    NestedTwoWayFeedbackReportV1 four_report;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    apply_nested_two_way_feedback_mapped(geometry,
                                         parent_mass,
                                         child_mass,
                                         parent_theta,
                                         child_theta,
                                         parent_advective,
                                         parent_pressure,
                                         child_advective,
                                         child_pressure,
                                         zeros,
                                         zeros,
                                         zeros,
                                         begin_inventory,
                                         four_parent,
                                         child,
                                         four_report);
    for (const auto pair : {std::pair<const Field3D*, const Field3D*>{&parent.rho, &four_parent.rho},
                            {&parent.theta, &four_parent.theta},
                            {&parent.p, &four_parent.p},
                            {&parent.u, &four_parent.u},
                            {&parent.v, &four_parent.v},
                            {&parent.w, &four_parent.w}})
        REQUIRE(std::memcmp(pair.first->data(), pair.second->data(), pair.first->size() * sizeof(float)) == 0);
    REQUIRE(one_report.restricted_scalar_cells == four_report.restricted_scalar_cells);
    REQUIRE(one_report.restricted_x_faces == four_report.restricted_x_faces);
    REQUIRE(one_report.restricted_y_faces == four_report.restricted_y_faces);
    REQUIRE(one_report.restricted_z_faces == four_report.restricted_z_faces);
    REQUIRE(one_report.interface_mismatch == four_report.interface_mismatch);
    REQUIRE(one_report.applied_reflux == four_report.applied_reflux);
    REQUIRE(one_report.storage_residual == four_report.storage_residual);
    REQUIRE(one_report.child_composite_residual == four_report.child_composite_residual);

    if (terrain.bell.h0 == 0.0)
    {
        auto                         flat_parent = before;
        NestedTwoWayFeedbackReportV1 flat_report;
        apply_nested_two_way_feedback_flat(
            relationship,
            std::vector<double>(relationship.levels_z, terrain.ztop / relationship.levels_z),
            parent_mass,
            child_mass,
            parent_theta,
            child_theta,
            parent_advective,
            parent_pressure,
            child_advective,
            child_pressure,
            zeros,
            zeros,
            zeros,
            begin_inventory,
            flat_parent,
            child,
            flat_report);
        for (const auto pair : {std::pair<const Field3D*, const Field3D*>{&parent.u, &flat_parent.u},
                                {&parent.v, &flat_parent.v},
                                {&parent.w, &flat_parent.w},
                                {&parent.rho, &flat_parent.rho},
                                {&parent.theta, &flat_parent.theta}})
            for (std::size_t n = 0; n < pair.first->size(); ++n)
                REQUIRE(pair.first->data()[n] == Approx(pair.second->data()[n]).epsilon(2e-6));
    }

    auto invalid_geometry                                = geometry;
    invalid_geometry.relationship.parent_time_step_ratio = 2;
    PrognosticFieldStorage       rejected                = before;
    NestedTwoWayFeedbackReportV1 retained;
    retained.complete              = true;
    retained.interface_mismatch[0] = 77.0;
    REQUIRE_THROWS(apply_nested_two_way_feedback_mapped(invalid_geometry,
                                                        parent_mass,
                                                        child_mass,
                                                        parent_theta,
                                                        child_theta,
                                                        parent_advective,
                                                        parent_pressure,
                                                        child_advective,
                                                        child_pressure,
                                                        zeros,
                                                        zeros,
                                                        zeros,
                                                        begin_inventory,
                                                        rejected,
                                                        child,
                                                        retained));
    REQUIRE(retained.interface_mismatch[0] == 77.0);
    REQUIRE(std::memcmp(rejected.rho.data(), before.rho.data(), rejected.rho.size() * sizeof(float)) == 0);
    invalid_geometry                             = geometry;
    invalid_geometry.relationship.parent_ratio_y = 2;
    REQUIRE_THROWS(apply_nested_two_way_feedback_mapped(invalid_geometry,
                                                        parent_mass,
                                                        child_mass,
                                                        parent_theta,
                                                        child_theta,
                                                        parent_advective,
                                                        parent_pressure,
                                                        child_advective,
                                                        child_pressure,
                                                        zeros,
                                                        zeros,
                                                        zeros,
                                                        begin_inventory,
                                                        rejected,
                                                        child,
                                                        retained));
    invalid_geometry                                         = geometry;
    invalid_geometry.relationship.child_origin_parent_cell_i = std::numeric_limits<int>::max();
    REQUIRE_THROWS(apply_nested_two_way_feedback_mapped(invalid_geometry,
                                                        parent_mass,
                                                        child_mass,
                                                        parent_theta,
                                                        child_theta,
                                                        parent_advective,
                                                        parent_pressure,
                                                        child_advective,
                                                        child_pressure,
                                                        zeros,
                                                        zeros,
                                                        zeros,
                                                        begin_inventory,
                                                        rejected,
                                                        child,
                                                        retained));
    invalid_geometry                           = geometry;
    invalid_geometry.parent.metrics.J(0, 0, 0) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(apply_nested_two_way_feedback_mapped(invalid_geometry,
                                                        parent_mass,
                                                        child_mass,
                                                        parent_theta,
                                                        child_theta,
                                                        parent_advective,
                                                        parent_pressure,
                                                        child_advective,
                                                        child_pressure,
                                                        zeros,
                                                        zeros,
                                                        zeros,
                                                        begin_inventory,
                                                        rejected,
                                                        child,
                                                        retained));
    REQUIRE(retained.interface_mismatch[0] == 77.0);
    REQUIRE(std::memcmp(rejected.rho.data(), before.rho.data(), rejected.rho.size() * sizeof(float)) == 0);
}

tmv::domain::OneWayNestRelationshipInput valid_relationship_input()
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = 120;
    input.parent_cells_y                = 90;
    input.parent_levels_z               = 60;
    input.parent_dx_m                   = 900.0;
    input.parent_dy_m                   = 600.0;
    input.parent_dt_s                   = 3.0;
    input.parent_ratio_x                = 3;
    input.parent_ratio_y                = 2;
    input.parent_time_step_ratio        = 3;
    input.child_origin_parent_cell_i    = 20;
    input.child_origin_parent_cell_j    = 15;
    input.child_cells_x                 = 180;
    input.child_cells_y                 = 100;
    input.child_levels_z                = 60;
    input.relaxation_width_child_cells  = 12;
    input.boundary_stencil_parent_cells = 2;
    return input;
}

tmv::domain::OneWayNestRelationshipInput small_relationship_input()
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = 6;
    input.parent_cells_y                = 6;
    input.parent_levels_z               = 2;
    input.parent_dx_m                   = 1000.0;
    input.parent_dy_m                   = 800.0;
    input.parent_dt_s                   = 2.0;
    input.parent_ratio_x                = 2;
    input.parent_ratio_y                = 2;
    input.parent_time_step_ratio        = 2;
    input.child_origin_parent_cell_i    = 2;
    input.child_origin_parent_cell_j    = 2;
    input.child_cells_x                 = 4;
    input.child_cells_y                 = 4;
    input.child_levels_z                = 2;
    input.relaxation_width_child_cells  = 1;
    input.boundary_stencil_parent_cells = 1;
    return input;
}

void fill_valid_primary_state(PrognosticFieldStorage& fields, int nx, int ny, int nz, float density)
{
    fields.rho.resize(nx, ny, nz, density);
    fields.p.resize(nx, ny, nz, 90000.0f);
    fields.u.resize(nx, ny, nz, 0.0f);
    fields.w.resize(nx, ny, nz, 0.0f);
    fields.v.resize(nx, ny, nz, 0.0f);
    fields.tracer.resize(nx, ny, nz, 0.0f);
    fields.theta.resize(nx, ny, nz, 300.0f);
    fields.qv.resize(nx, ny, nz, 0.0f);
    fields.qc.resize(nx, ny, nz, 0.0f);
    fields.qr.resize(nx, ny, nz, 0.0f);
    fields.qi.resize(nx, ny, nz, 0.0f);
    fields.qs.resize(nx, ny, nz, 0.0f);
    fields.qh.resize(nx, ny, nz, 0.0f);
    fields.qg.resize(nx, ny, nz, 0.0f);
    fields.tke.resize(nx, ny, nz, 0.0f);
}

SimulationStorageState ready_process_binding_state()
{
    SimulationStorageState state({8, 8, 2, 1000.0, 300.0, 2.0, 800.0});
    state.grid.geometry.initialize(8, 8, 2, 1000.0, 300.0, 800.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.grid.geometry.z_faces      = {0.0, 200.0, 550.0};
    state.grid.geometry.z            = {100.0, 375.0};
    state.process_grid_metrics.dx    = 1000.0;
    state.process_grid_metrics.dy    = 800.0;
    state.process_grid_metrics.dz    = {200.0, 350.0};
    state.process_grid_metrics.z_int = state.grid.geometry.z_faces;
    state.time_s                     = 2.0;
    state.base.p0                    = {90000.0, 85000.0};
    state.base.theta0                = {300.0, 302.0};
    state.base.rho0.resize(2);
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    for (int k = 0; k < 2; ++k)
    {
        const double temperature =
            state.base.theta0[k] * std::pow(state.base.p0[k] / physical_constants::reference_pressure_pa, kappa);
        state.base.rho0[k] = state.base.p0[k] / (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    state.base.qv0 = {0.0, 0.0};
    state.base.u0  = {0.0, 0.0};
    state.base.v0  = {0.0, 0.0};
    fill_valid_primary_state(state.fields, 8, 8, 2, 1.0f);

    prepare_nested_execution_relationship(
        state, tmv::domain::prepare_one_way_nest_relationship(small_relationship_input()), 0.0);
    initialize_nested_shared_level_reference_state(state);
    fill_valid_primary_state(state.nested_execution.child.fields, 6, 6, 2, 1.0f);
    state.nested_execution.physical_primary_state_ready       = true;
    state.nested_execution.parent_boundary_state_ready        = true;
    state.nested_execution.interface_source.available         = true;
    state.nested_execution.interface_source.application_count = 1;
    require_valid_nested_execution_relationship(state);
    return state;
}

class CountingResidencyBackend final : public ComputeBackend
{
public:
    enum class Event
    {
        SelectRoot,
        SelectChild,
        InvalidateRoot,
        InvalidateChild,
    };

    std::string name() const override
    {
        return "counting";
    }
    bool initialize(std::string&) override
    {
        return true;
    }
    void shutdown() override
    {
    }
    void select_residency_scope(ComputeResidencyScope scope) noexcept override
    {
        active_scope = scope;
        if (selection_count < selected_scopes.size())
            selected_scopes[selection_count++] = scope;
        record(scope == ComputeResidencyScope::Child ? Event::SelectChild : Event::SelectRoot);
    }
    void invalidate_all_resident_fields() noexcept override
    {
        invalidate_residency_scope(active_scope);
    }
    void invalidate_residency_scope(ComputeResidencyScope scope) noexcept override
    {
        if (scope == ComputeResidencyScope::Child)
        {
            ++child_invalidations;
            record(Event::InvalidateChild);
        }
        else
        {
            ++root_invalidations;
            record(Event::InvalidateRoot);
        }
    }

    void record(Event event) noexcept
    {
        if (event_count < events.size())
            events[event_count++] = event;
    }

    ComputeResidencyScope                active_scope        = ComputeResidencyScope::Root;
    int                                  root_invalidations  = 0;
    int                                  child_invalidations = 0;
    std::array<ComputeResidencyScope, 4> selected_scopes{};
    std::size_t                          selection_count = 0;
    std::array<Event, 6>                 events{};
    std::size_t                          event_count = 0;
};

class GlobalStorageRestore
{
public:
    GlobalStorageRestore() : saved_(std::move(global_simulation_storage))
    {
    }
    ~GlobalStorageRestore()
    {
        global_simulation_storage = std::move(saved_);
    }

private:
    SimulationStorageState saved_;
};

struct ChildStepObservation
{
    bool   called                 = false;
    bool   child_shape            = false;
    bool   run_authority_withheld = false;
    double step_s                 = 0.0;
    double start_time_s           = 0.0;
};

void observe_child_step(const SimulationStateView& state,
                        double                     child_step_s,
                        double                     child_start_time_s,
                        void*                      raw_context)
{
    auto& observation                  = *static_cast<ChildStepObservation*>(raw_context);
    observation.called                 = true;
    observation.child_shape            = state.grid.nr == 6 && state.grid.nth == 6 && state.grid.nz == 2;
    observation.run_authority_withheld = state.runtime_owner == nullptr;
    observation.step_s                 = child_step_s;
    observation.start_time_s           = child_start_time_s;
    state.fields.tracer[2][2][0]       = 0.75f;
}

void reject_child_step(const SimulationStateView&, double, double, void*)
{
    throw std::runtime_error("injected child step failure");
}

void mutate_child_clock(const SimulationStateView& state, double, double, void*)
{
    state.time_s += 0.25;
}

} // namespace

TEST_CASE("one-way nest relationship preserves exact geometry and CFL floor", "[core][nested_grid][relationship]")
{
    const auto relationship = tmv::domain::prepare_one_way_nest_relationship(valid_relationship_input());

    REQUIRE(relationship.version == 1);
    REQUIRE(relationship.child_parent_span_x == 60);
    REQUIRE(relationship.child_parent_span_y == 50);
    REQUIRE(relationship.child_dx_m == Approx(300.0));
    REQUIRE(relationship.child_dy_m == Approx(300.0));
    REQUIRE(relationship.child_dt_s == Approx(1.0));
    REQUIRE(relationship.levels_z == 60);

    const auto scalar = tmv::domain::parent_coordinate_for_child(
        relationship, tmv::domain::NestedHorizontalLocation::ScalarCenter, 0, 0);
    REQUIRE(scalar.i == Approx(20.0 + 0.5 / 3.0 - 0.5));
    REQUIRE(scalar.j == Approx(0.5 / 2.0 + 15.0 - 0.5));

    const auto x_face =
        tmv::domain::parent_coordinate_for_child(relationship, tmv::domain::NestedHorizontalLocation::XFace, 180, 99);
    REQUIRE(x_face.i == Approx(80.0));
    REQUIRE(x_face.j == Approx((99.5 / 2.0) + 14.5));

    const auto y_face =
        tmv::domain::parent_coordinate_for_child(relationship, tmv::domain::NestedHorizontalLocation::YFace, 179, 100);
    REQUIRE(y_face.i == Approx((179.5 / 3.0) + 19.5));
    REQUIRE(y_face.j == Approx(65.0));
}

TEST_CASE("frontier operational geometry retains the parent stencil through its bound path",
          "[core][nested_grid][relationship][moving_nest][geometry]")
{
    tmv::domain::OneWayNestRelationshipInput input;
    // The YAML storage dimensions 135x108 contain 133x106 physical scalar
    // cells after the Cartesian C-grid halo is excluded.
    input.parent_cells_x                = 133;
    input.parent_cells_y                = 106;
    input.parent_levels_z               = 50;
    input.parent_dx_m                   = 1200.0;
    input.parent_dy_m                   = 1200.0;
    input.parent_dt_s                   = 1.5;
    input.parent_ratio_x                = 2;
    input.parent_ratio_y                = 2;
    input.parent_time_step_ratio        = 2;
    input.child_origin_parent_cell_i    = 2;
    input.child_origin_parent_cell_j    = 1;
    input.child_cells_x                 = 156;
    input.child_cells_y                 = 166;
    input.child_levels_z                = 50;
    input.relaxation_width_child_cells  = 8;
    input.boundary_stencil_parent_cells = 1;

    const auto initial = tmv::domain::prepare_one_way_nest_relationship(input);
    REQUIRE(initial.child_parent_span_x == 78);
    REQUIRE(initial.child_parent_span_y == 83);
    REQUIRE(initial.child_origin_parent_cell_i == 2);
    REQUIRE(initial.child_origin_parent_cell_j == 1);
    REQUIRE(initial.child_dx_m == 600.0);
    REQUIRE(initial.child_dy_m == 600.0);

    // Duration-matched reference replay reaches exactly this maximum origin.
    // It consumes the declared east/north stencil allowance without crossing
    // the parent physical ownership boundary.
    input.child_origin_parent_cell_i = 54;
    input.child_origin_parent_cell_j = 22;
    const auto terminal              = tmv::domain::prepare_one_way_nest_relationship(input);
    REQUIRE(terminal.child_origin_parent_cell_i + terminal.child_parent_span_x ==
            terminal.parent_cells_x - terminal.boundary_stencil_parent_cells);
    REQUIRE(terminal.child_origin_parent_cell_j + terminal.child_parent_span_y ==
            terminal.parent_cells_y - terminal.boundary_stencil_parent_cells);

    input.child_origin_parent_cell_i = 55;
    REQUIRE_THROWS_WITH(tmv::domain::prepare_one_way_nest_relationship(input),
                        Catch::Matchers::Contains("required parent stencil margin"));
    input.child_origin_parent_cell_i = 54;
    input.child_origin_parent_cell_j = 23;
    REQUIRE_THROWS_WITH(tmv::domain::prepare_one_way_nest_relationship(input),
                        Catch::Matchers::Contains("required parent stencil margin"));
}

TEST_CASE("one-way nest time bracket and movement are integer exact", "[core][nested_grid][relationship]")
{
    const auto relationship = tmv::domain::prepare_one_way_nest_relationship(valid_relationship_input());

    const auto start  = tmv::domain::parent_temporal_weights(relationship, 0);
    const auto middle = tmv::domain::parent_temporal_weights(relationship, 1);
    const auto end    = tmv::domain::parent_temporal_weights(relationship, 3);
    REQUIRE(start.previous == 1.0);
    REQUIRE(start.current == 0.0);
    REQUIRE(middle.previous == Approx(2.0 / 3.0));
    REQUIRE(middle.current == Approx(1.0 / 3.0));
    REQUIRE(middle.offset_from_previous_s == Approx(1.0));
    REQUIRE(end.previous == 0.0);
    REQUIRE(end.current == 1.0);
    REQUIRE(end.offset_from_previous_s == Approx(3.0));

    const auto shift = tmv::domain::child_shift_for_parent_move(relationship, -4, 5);
    REQUIRE(shift.i == -12);
    REQUIRE(shift.j == 10);
    REQUIRE_THROWS_WITH(tmv::domain::parent_temporal_weights(relationship, 4),
                        Catch::Matchers::Contains("outside the completed parent bracket"));

    auto bracket = tmv::domain::prepare_parent_time_bracket(relationship, 120.25, 123.25);
    REQUIRE(tmv::domain::nested_child_time_s(relationship, bracket) == Approx(120.25));
    tmv::domain::advance_child_substep(relationship, bracket);
    REQUIRE(bracket.completed_child_substeps == 1);
    REQUIRE(tmv::domain::nested_child_time_s(relationship, bracket) == Approx(121.25));
    tmv::domain::advance_child_substep(relationship, bracket);
    tmv::domain::advance_child_substep(relationship, bracket);
    REQUIRE(tmv::domain::nested_child_time_s(relationship, bracket) == Approx(123.25));
    REQUIRE_THROWS_WITH(tmv::domain::advance_child_substep(relationship, bracket),
                        Catch::Matchers::Contains("already complete"));
    tmv::domain::roll_parent_time_bracket(relationship, bracket, 126.25);
    REQUIRE(bracket.previous_parent_time_s == Approx(123.25));
    REQUIRE(bracket.current_parent_time_s == Approx(126.25));
    REQUIRE(bracket.completed_child_substeps == 0);

    auto shortened = tmv::domain::prepare_parent_time_bracket(relationship, 200.0, 202.25);
    tmv::domain::advance_child_substep(relationship, shortened);
    REQUIRE(tmv::domain::nested_child_time_s(relationship, shortened) == Approx(200.75));
    REQUIRE_THROWS_WITH(tmv::domain::prepare_parent_time_bracket(relationship, 200.0, 203.25),
                        Catch::Matchers::Contains("nominal ceiling"));
    REQUIRE_THROWS_WITH(tmv::domain::prepare_parent_time_bracket(
                            relationship, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()),
                        Catch::Matchers::Contains("nominal ceiling"));
}

TEST_CASE("Decimal parent timesteps retain exact child endpoints without admitting oversized steps",
          "[core][nested_grid][relationship][clock]")
{
    auto input              = valid_relationship_input();
    input.parent_dt_s       = .2;
    const auto relationship = tmv::domain::prepare_one_way_nest_relationship(input);
    double     previous     = GENERATE(0.0, 10000.0, 1e9);
    for (int step = 0; step < 20; ++step)
    {
        const double current = previous + input.parent_dt_s;
        auto         bracket = tmv::domain::prepare_parent_time_bracket(relationship, previous, current);
        for (int child = 0; child < relationship.parent_time_step_ratio; ++child)
        {
            tmv::domain::advance_child_substep(relationship, bracket);
        }
        CHECK(tmv::domain::nested_child_time_s(relationship, bracket) == current);
        CHECK_THROWS(tmv::domain::prepare_parent_time_bracket(
            relationship, previous, std::nextafter(current, std::numeric_limits<double>::infinity())));
        previous = current;
    }
}

TEST_CASE("parent and child level views own independent scientific state", "[core][nested_grid][state]")
{
    SimulationStorageState parent({122, 92, 60, 900.0, 300.0, 3.0, 600.0});
    parent.grid.geometry.initialize(122, 92, 60, 900.0, 300.0, 600.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    parent.fields.rho.resize(122, 92, 60, 1.0f);
    parent.time_s = 123.25;
    parent.base.rho0.resize(60);
    parent.base.p0.resize(60);
    parent.base.theta0.resize(60);
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    for (int k = 0; k < 60; ++k)
    {
        const auto level          = static_cast<std::size_t>(k);
        parent.base.p0[level]     = 90000.0 - 500.0 * k;
        parent.base.theta0[level] = 300.0 + 0.5 * k;
        const double temperature  = parent.base.theta0[level] *
                                    std::pow(parent.base.p0[level] / physical_constants::reference_pressure_pa, kappa);
        parent.base.rho0[level] = parent.base.p0[level] / (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    parent.base.qv0.assign(60, 0.008);
    parent.base.u0.assign(60, 6.0);
    parent.base.v0.assign(60, 0.0);
    parent.coriolis_f                                      = 8.0e-5;
    parent.configuration.domain_frame.earth_origin_at_t0_m = {1000.0, 2000.0};

    prepare_nested_execution_relationship(
        parent, tmv::domain::prepare_one_way_nest_relationship(valid_relationship_input()), 120.25);
    initialize_nested_shared_level_reference_state(parent);
    parent.nested_execution.child.fields.rho.resize(182, 102, 60, 2.0f);
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(parent));
    REQUIRE(parent.nested_execution.child.base.p0 == parent.base.p0);
    REQUIRE(parent.nested_execution.child.base.theta0 == parent.base.theta0);
    REQUIRE(parent.nested_execution.child.coriolis_f == parent.coriolis_f);
    REQUIRE(parent.nested_execution.child.grid.geometry.z == parent.grid.geometry.z);
    REQUIRE(parent.nested_execution.child.grid.geometry.z_faces == parent.grid.geometry.z_faces);

    parent.configuration.coriolis_mode         = tmv::init::config::CoriolisMode::BetaPlane;
    parent.configuration.coriolis_latitude_deg = 35.0;
    parent.coriolis_f =
        tmv::make_cartesian_coriolis_profile(
            true, true, parent.configuration.coriolis_latitude_deg, parent.configuration.domain_frame, parent.time_s)
            .f_at_grid_y0;
    parent.nested_execution.reference_state_ready = false;
    initialize_nested_shared_level_reference_state(parent);
    const auto expected_child_coriolis =
        tmv::make_cartesian_coriolis_profile(true,
                                             true,
                                             parent.configuration.coriolis_latitude_deg,
                                             parent.nested_execution.child.domain_frame,
                                             parent.nested_execution.child.time_s);
    REQUIRE(parent.nested_execution.child.coriolis_f == expected_child_coriolis.f_at_grid_y0);
    REQUIRE(parent.nested_execution.child.coriolis_f != parent.coriolis_f);
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(parent));

    const auto accepted_child_base                = parent.nested_execution.child.base;
    parent.nested_execution.reference_state_ready = false;
    parent.base.p0[1]                             = parent.base.p0[0];
    REQUIRE_THROWS_WITH(initialize_nested_shared_level_reference_state(parent),
                        Catch::Matchers::Contains("physically ordered"));
    REQUIRE(parent.nested_execution.child.base.p0 == accepted_child_base.p0);
    parent.base = accepted_child_base;
    initialize_nested_shared_level_reference_state(parent);

    auto parent_view = make_simulation_state_view(parent);
    auto child_view  = make_simulation_state_view(
        parent.nested_execution.child, parent.configuration, parent_view.compute_runtime, parent_view.parity_runtime);
    child_view.fields.rho[0][0][0] = 3.0f;
    child_view.time_s              = 121.25;
    REQUIRE(parent_view.fields.rho[0][0][0] == Approx(1.0f));
    REQUIRE(child_view.fields.rho[0][0][0] == Approx(3.0f));
    REQUIRE(parent_view.time_s == Approx(123.25));
    REQUIRE(child_view.time_s == Approx(121.25));
    REQUIRE(child_view.runtime_owner == nullptr);
    REQUIRE(parent_view.compute_runtime == child_view.compute_runtime);
    REQUIRE(parent_view.parity_runtime == child_view.parity_runtime);
    REQUIRE(parent_view.configuration == child_view.configuration);
    REQUIRE(parent_view.domain_frame != child_view.domain_frame);
    REQUIRE(child_view.domain_frame->earth_origin_at_t0_m.x_m == Approx(19600.0));

    const auto snapshot                                                  = snapshot_simulation_storage(parent);
    parent.nested_execution.child.fields.rho[0][0][0]                    = -5.0f;
    parent.nested_execution.parent_time_bracket.completed_child_substeps = 2;
    restore_simulation_storage(parent, snapshot);
    REQUIRE(parent.nested_execution.child.fields.rho[0][0][0] == Approx(3.0f));
    REQUIRE(parent.nested_execution.parent_time_bracket.completed_child_substeps == 0);
}

TEST_CASE("nested execution relationship rejects inconsistent live ownership",
          "[core][nested_grid][state][transaction]")
{
    SimulationStorageState parent({122, 92, 60, 900.0, 300.0, 3.0, 600.0});
    parent.grid.geometry.initialize(122, 92, 60, 900.0, 300.0, 600.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    parent.time_s                                          = 123.25;
    parent.configuration.domain_frame.earth_origin_at_t0_m = {1000.0, 2000.0};
    prepare_nested_execution_relationship(
        parent, tmv::domain::prepare_one_way_nest_relationship(valid_relationship_input()), 120.25);

    SECTION("child clock")
    {
        parent.nested_execution.child.time_s = 120.5;
        REQUIRE_THROWS_WITH(require_valid_nested_execution_relationship(parent),
                            Catch::Matchers::Contains("child grid or clock"));
    }
    SECTION("Earth frame")
    {
        parent.nested_execution.child.domain_frame.earth_origin_at_t0_m.x_m += 1.0;
        REQUIRE_THROWS_WITH(require_valid_nested_execution_relationship(parent),
                            Catch::Matchers::Contains("child Earth frame"));
    }
    SECTION("movement provenance")
    {
        parent.nested_execution.cumulative_parent_shift_i = 1;
        REQUIRE_THROWS_WITH(require_valid_nested_execution_relationship(parent),
                            Catch::Matchers::Contains("movement provenance"));
    }
    SECTION("parent clock")
    {
        parent.time_s += 0.25;
        REQUIRE_THROWS_WITH(require_valid_nested_execution_relationship(parent),
                            Catch::Matchers::Contains("parent clock"));
    }
    SECTION("base state")
    {
        parent.nested_execution.reference_state_ready = true;
        parent.base.theta0.assign(60, 300.0);
        REQUIRE_THROWS_WITH(require_valid_nested_execution_relationship(parent),
                            Catch::Matchers::Contains("base state or Coriolis"));
    }
}

TEST_CASE("nested execution changes only the selected level and device residency",
          "[core][nested_grid][state][binding][device]")
{
    GlobalStorageRestore restore_global;
    global_simulation_storage                           = ready_process_binding_state();
    auto& parent                                        = global_simulation_storage;
    auto  backend                                       = std::make_unique<CountingResidencyBackend>();
    auto* device                                        = backend.get();
    parent.runtime.compute.backend                      = std::move(backend);
    const auto* parent_rho                              = parent.fields.rho.data();
    const auto* child_rho                               = parent.nested_execution.child.fields.rho.data();
    const int   parent_nx                               = parent.grid.nr;
    parent.runtime.boundary_conditions.radiative_u_west = {11.0f, 12.0f};
    parent.nested_execution.child.runtime.boundary_conditions.radiative_u_west = {21.0f, 22.0f};
    {
        ScopedNestedChildResidency scope(parent);
        const auto                 child = nested_child_execution_state(parent);
        REQUIRE(parent.fields.rho.data() == parent_rho);
        REQUIRE(parent.grid.nr == parent_nx);
        REQUIRE(child.fields.rho.data() == child_rho);
        REQUIRE(child.runtime_owner == nullptr);
        REQUIRE(global_simulation_state().runtime_owner == &parent.runtime);
        REQUIRE(child.compute_runtime == &parent.runtime.compute);
        REQUIRE(child.parity_runtime == &parent.runtime.parity);
        child.fields.rho[1][1][0]                             = 3.0f;
        child.boundary_condition_runtime->radiative_u_west[0] = 31.0f;
        REQUIRE(parent.runtime.boundary_conditions.radiative_u_west[0] == 11.0f);
        REQUIRE(device->active_scope == ComputeResidencyScope::Child);
        REQUIRE_THROWS_WITH(ScopedNestedChildResidency(parent),
                            Catch::Matchers::Contains("child execution is already active"));
        auto                       detached = ready_process_binding_state();
        ScopedNestedChildResidency detached_scope(detached);
        REQUIRE(detached.nested_execution.child_execution_active);
        REQUIRE(parent.nested_execution.child_execution_active);
    }
    REQUIRE_FALSE(parent.nested_execution.child_execution_active);
    REQUIRE(parent.fields.rho.data() == parent_rho);
    REQUIRE(parent.nested_execution.child.fields.rho[1][1][0] == 3.0f);
    REQUIRE(parent.nested_execution.child.runtime.boundary_conditions.radiative_u_west[0] == 31.0f);
    REQUIRE(device->active_scope == ComputeResidencyScope::Root);
    REQUIRE(device->root_invalidations == 0);
    REQUIRE(device->child_invalidations == 1);
    REQUIRE_THROWS_AS(
        [&]
        {
            ScopedNestedChildResidency scope(parent);
            throw std::runtime_error("injected child work failure");
        }(),
        std::runtime_error);
    REQUIRE_FALSE(parent.nested_execution.child_execution_active);
    REQUIRE(device->active_scope == ComputeResidencyScope::Root);
    REQUIRE(device->child_invalidations == 2);
}

TEST_CASE("nested child subcycles commit one realized bracket phase at a time",
          "[core][nested_grid][state][subcycle][clock]")
{
    GlobalStorageRestore restore_global;
    global_simulation_storage                         = ready_process_binding_state();
    PrognosticFieldStorage previous_parent            = global_simulation_storage.fields;
    auto                   backend                    = std::make_unique<CountingResidencyBackend>();
    auto*                  backend_observer           = backend.get();
    global_simulation_storage.runtime.compute.backend = std::move(backend);

    ChildStepObservation     first_observation;
    NestedChildSubstepReport first_report;
    advance_nested_child_substep(
        global_simulation_storage, previous_parent, {4.0}, observe_child_step, &first_observation, first_report);
    REQUIRE(first_observation.called);
    REQUIRE(first_observation.child_shape);
    REQUIRE(first_observation.run_authority_withheld);
    REQUIRE(first_observation.step_s == 1.0);
    REQUIRE(first_observation.start_time_s == 0.0);
    REQUIRE(first_report.start_time_s == 0.0);
    REQUIRE(first_report.end_time_s == 1.0);
    REQUIRE(first_report.completed_child_substeps == 1);
    REQUIRE(global_simulation_storage.time_s == 2.0);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s == 1.0);
    REQUIRE(global_simulation_storage.nested_execution.parent_time_bracket.completed_child_substeps == 1);
    REQUIRE(global_simulation_storage.fields.tracer[2][2][0] == 0.0f);
    REQUIRE(global_simulation_storage.nested_execution.child.fields.tracer[2][2][0] == 0.75f);
    REQUIRE(global_simulation_storage.nested_execution.interface_source.application_count == 2);
    REQUIRE(backend_observer->active_scope == ComputeResidencyScope::Root);
    REQUIRE(backend_observer->root_invalidations == 0);
    REQUIRE(backend_observer->child_invalidations == 1);

    ChildStepObservation     second_observation;
    NestedChildSubstepReport second_report;
    advance_nested_child_substep(
        global_simulation_storage, previous_parent, {4.0}, observe_child_step, &second_observation, second_report);
    REQUIRE(second_observation.start_time_s == 1.0);
    REQUIRE(second_report.end_time_s == 2.0);
    REQUIRE(second_report.completed_child_substeps == 2);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s == global_simulation_storage.time_s);
    REQUIRE(backend_observer->active_scope == ComputeResidencyScope::Root);
    REQUIRE(backend_observer->root_invalidations == 0);
    REQUIRE(backend_observer->child_invalidations == 2);
    REQUIRE_THROWS_WITH(
        advance_nested_child_substep(
            global_simulation_storage, previous_parent, {4.0}, observe_child_step, &second_observation, second_report),
        Catch::Matchers::Contains("already synchronized"));
}

TEST_CASE("nested child subcycle failures restore binding and clock authority",
          "[core][nested_grid][state][subcycle][transaction]")
{
    GlobalStorageRestore restore_global;
    global_simulation_storage                    = ready_process_binding_state();
    const PrognosticFieldStorage previous_parent = global_simulation_storage.fields;
    NestedChildSubstepReport     report;
    report.end_time_s = 99.0;

    const auto original_child    = global_simulation_storage.nested_execution.child;
    const auto mutate_and_reject = [](const SimulationStateView& state, double, double, void*)
    {
        state.fields.theta.fill(450.0f);
        state.time_stepping_runtime->diagnostics.n_steps += 7;
        throw std::runtime_error("injected mutation before child step failure");
    };
    REQUIRE_THROWS(advance_nested_child_substep(
        global_simulation_storage, previous_parent, {4.0}, mutate_and_reject, nullptr, report));
    const auto mutate_then_return = [](const SimulationStateView& state, double, double, void*)
    { state.fields.theta.fill(450.0f); };
    REQUIRE_THROWS(advance_nested_child_substep(
        global_simulation_storage, previous_parent, {-1.0}, mutate_then_return, nullptr, report));
    const auto& restored_child = global_simulation_storage.nested_execution.child;
    REQUIRE(std::memcmp(restored_child.fields.theta.data(),
                        original_child.fields.theta.data(),
                        original_child.fields.theta.size() * sizeof(float)) == 0);
    REQUIRE(restored_child.runtime.time_stepping.diagnostics.n_steps ==
            original_child.runtime.time_stepping.diagnostics.n_steps);

    REQUIRE_THROWS_WITH(advance_nested_child_substep(
                            global_simulation_storage, previous_parent, {4.0}, reject_child_step, nullptr, report),
                        Catch::Matchers::Contains("injected child step failure"));
    REQUIRE_FALSE(global_simulation_storage.nested_execution.child_execution_active);
    REQUIRE(global_simulation_storage.time_s == 2.0);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s == 0.0);
    REQUIRE(global_simulation_storage.nested_execution.parent_time_bracket.completed_child_substeps == 0);
    REQUIRE(report.end_time_s == 99.0);

    REQUIRE_THROWS_WITH(advance_nested_child_substep(
                            global_simulation_storage, previous_parent, {4.0}, mutate_child_clock, nullptr, report),
                        Catch::Matchers::Contains("must not mutate the level clock"));
    REQUIRE_FALSE(global_simulation_storage.nested_execution.child_execution_active);
    REQUIRE(global_simulation_storage.nested_execution.child.time_s == 0.0);
    REQUIRE(global_simulation_storage.nested_execution.parent_time_bracket.completed_child_substeps == 0);
    REQUIRE(report.end_time_s == 99.0);
}

TEST_CASE("nested parent interval rolls from one synchronized endpoint",
          "[core][nested_grid][state][clock][transaction]")
{
    SimulationStorageState state                        = ready_process_binding_state();
    auto&                  nested                       = state.nested_execution;
    nested.parent_time_bracket.completed_child_substeps = nested.relationship.parent_time_step_ratio;
    nested.child.time_s                                 = state.time_s;
    require_valid_nested_execution_relationship(state);

    state.time_s  = 3.5;
    state.grid.dt = 1.5;
    REQUIRE_NOTHROW(roll_nested_parent_interval(state));
    REQUIRE(nested.parent_time_bracket.previous_parent_time_s == 2.0);
    REQUIRE(nested.parent_time_bracket.current_parent_time_s == 3.5);
    REQUIRE(nested.parent_time_bracket.completed_child_substeps == 0);
    REQUIRE(nested.child.time_s == 2.0);
    REQUIRE(nested.child.grid.dt == 0.75);
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(state));

    nested.parent_time_bracket.completed_child_substeps = nested.relationship.parent_time_step_ratio;
    nested.child.time_s                                 = state.time_s;
    const auto   accepted_bracket                       = nested.parent_time_bracket;
    const double accepted_child_dt                      = nested.child.grid.dt;
    state.time_s                                        = 6.0;
    state.grid.dt                                       = 2.0;
    REQUIRE_THROWS_WITH(roll_nested_parent_interval(state),
                        Catch::Matchers::Contains("exceeds its finite nominal ceiling"));
    REQUIRE(nested.parent_time_bracket.previous_parent_time_s == accepted_bracket.previous_parent_time_s);
    REQUIRE(nested.parent_time_bracket.current_parent_time_s == accepted_bracket.current_parent_time_s);
    REQUIRE(nested.parent_time_bracket.completed_child_substeps == accepted_bracket.completed_child_substeps);
    REQUIRE(nested.child.grid.dt == accepted_child_dt);
}

TEST_CASE("nested execution uses physical cells and realized bracket timing", "[core][nested_grid][state][clock]")
{
    SimulationStorageState parent({122, 92, 60, 900.0, 300.0, 2.25, 600.0});
    parent.grid.geometry.initialize(122, 92, 60, 900.0, 300.0, 600.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    parent.time_s                                          = 202.25;
    parent.configuration.domain_frame.earth_origin_at_t0_m = {1000.0, 2000.0};
    const auto relationship = tmv::domain::prepare_one_way_nest_relationship(valid_relationship_input());
    prepare_nested_execution_relationship(parent, relationship, 200.0);

    REQUIRE(parent.nested_execution.child.grid.nr == 182);
    REQUIRE(parent.nested_execution.child.grid.nth == 102);
    REQUIRE(parent.nested_execution.child.grid.dt == Approx(0.75));
    REQUIRE(parent.nested_execution.child.domain_frame.earth_origin_at_t0_m.x_m == Approx(19600.0));
    auto bracket = parent.nested_execution.parent_time_bracket;
    tmv::domain::advance_child_substep(relationship, bracket);
    const auto weights = tmv::domain::parent_temporal_weights(relationship, bracket);
    REQUIRE(weights.current == Approx(1.0 / 3.0));
    REQUIRE(weights.offset_from_previous_s == Approx(0.75));
}

TEST_CASE("one-way nest relationship rejects scientifically incomplete geometry", "[core][nested_grid][relationship]")
{
    SECTION("fractional parent footprint")
    {
        auto input          = valid_relationship_input();
        input.child_cells_x = 181;
        REQUIRE_THROWS_WITH(tmv::domain::prepare_one_way_nest_relationship(input),
                            Catch::Matchers::Contains("span whole parent cells"));
    }
    SECTION("time refinement below the horizontal CFL floor")
    {
        auto input                   = valid_relationship_input();
        input.parent_time_step_ratio = 2;
        REQUIRE_THROWS_WITH(tmv::domain::prepare_one_way_nest_relationship(input),
                            Catch::Matchers::Contains("largest horizontal refinement ratio"));
    }
    SECTION("vertical remapping")
    {
        auto input           = valid_relationship_input();
        input.child_levels_z = 90;
        REQUIRE_THROWS_WITH(tmv::domain::prepare_one_way_nest_relationship(input),
                            Catch::Matchers::Contains("shared parent/child vertical levels"));
    }
    SECTION("missing scientific interior")
    {
        auto input                         = valid_relationship_input();
        input.relaxation_width_child_cells = 50;
        REQUIRE_THROWS_WITH(tmv::domain::prepare_one_way_nest_relationship(input),
                            Catch::Matchers::Contains("non-empty verification interior"));
    }
    SECTION("insufficient parent stencil margin")
    {
        auto input                       = valid_relationship_input();
        input.child_origin_parent_cell_i = 1;
        REQUIRE_THROWS_WITH(tmv::domain::prepare_one_way_nest_relationship(input),
                            Catch::Matchers::Contains("required parent stencil margin"));
    }
    SECTION("corrupt checkpoint-derived geometry")
    {
        auto relationship       = tmv::domain::prepare_one_way_nest_relationship(valid_relationship_input());
        relationship.child_dx_m = 301.0;
        REQUIRE_THROWS_WITH(tmv::domain::parent_temporal_weights(relationship, 1),
                            Catch::Matchers::Contains("stored derived geometry or timestep is inconsistent"));
    }
    SECTION("unknown C-grid owner")
    {
        const auto relationship = tmv::domain::prepare_one_way_nest_relationship(valid_relationship_input());
        REQUIRE_THROWS_WITH(tmv::domain::parent_coordinate_for_child(
                                relationship, static_cast<tmv::domain::NestedHorizontalLocation>(99), 0, 0),
                            Catch::Matchers::Contains("unknown C-grid owner location"));
    }
}

TEST_CASE("nested child physical primary initialization is one conserved transaction",
          "[core][nested_grid][state][initialization][conservation]")
{
    SimulationStorageState parent({8, 8, 2, 1000.0, 300.0, 2.0, 800.0});
    parent.grid.geometry.initialize(8, 8, 2, 1000.0, 300.0, 800.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    parent.grid.geometry.z_faces = {0.0, 200.0, 550.0};
    parent.grid.geometry.z       = {100.0, 375.0};
    parent.time_s                = 2.0;
    parent.base.p0               = {90000.0, 85000.0};
    parent.base.theta0           = {300.0, 302.0};
    parent.base.rho0.resize(2);
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    for (int k = 0; k < 2; ++k)
    {
        const double temperature =
            parent.base.theta0[k] * std::pow(parent.base.p0[k] / physical_constants::reference_pressure_pa, kappa);
        parent.base.rho0[k] = parent.base.p0[k] / (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    parent.base.qv0   = {0.01, 0.006};
    parent.base.u0    = {2.0, 3.0};
    parent.base.v0    = {-1.0, 0.0};
    parent.coriolis_f = 8.0e-5;

    const float nan      = std::numeric_limits<float>::quiet_NaN();
    const auto  allocate = [&](PrognosticFieldStorage& fields, float value)
    {
        Field3D* list[] = {&fields.rho,
                           &fields.p,
                           &fields.u,
                           &fields.v,
                           &fields.w,
                           &fields.theta,
                           &fields.qv,
                           &fields.qc,
                           &fields.qr,
                           &fields.qi,
                           &fields.qs,
                           &fields.qg,
                           &fields.qh,
                           &fields.tracer,
                           &fields.tke};
        for (Field3D* field : list)
            field->resize(8, 8, 2, value);
    };
    PrognosticFieldStorage previous;
    allocate(previous, 0.0f);
    allocate(parent.fields, nan);
    previous.rho.fill(1.0f);
    previous.theta.fill(300.0f);
    previous.qv.fill(0.01f);
    previous.qc.fill(0.001f);
    previous.qr.fill(0.002f);
    previous.qi.fill(0.0005f);
    previous.qs.fill(0.0004f);
    previous.qg.fill(0.0003f);
    previous.qh.fill(0.0002f);
    previous.tracer.fill(0.25f);
    previous.tke.fill(0.25f);
    previous.u.fill(5.0f);
    previous.v.fill(-3.0f);
    previous.w.fill(1.5f);

    const auto relationship = tmv::domain::prepare_one_way_nest_relationship(small_relationship_input());
    prepare_nested_execution_relationship(parent, relationship, 0.0);
    initialize_nested_shared_level_reference_state(parent);
    NestedChildInitializationReport report;
    initialize_nested_child_physical_primary_state(parent, previous, report);

    const auto& nested = parent.nested_execution;
    REQUIRE(nested.physical_primary_state_ready);
    REQUIRE(nested.child.grid.geometry.z_faces == parent.grid.geometry.z_faces);
    REQUIRE(nested.child.fields.rho[1][1][0] == 1.0f);
    REQUIRE(nested.child.fields.theta[4][4][1] == 300.0f);
    REQUIRE(nested.child.fields.qv[2][3][0] == 0.01f);
    REQUIRE(nested.child.fields.tracer[3][2][1] == 0.25f);
    REQUIRE(nested.child.fields.u[0][1][0] == 5.0f);
    REQUIRE(nested.child.fields.v[1][0][0] == -3.0f);
    REQUIRE(nested.child.fields.w[1][1][1] == 1.5f);
    REQUIRE(std::isfinite(nested.child.fields.p[1][1][0]));
    REQUIRE(std::isnan(nested.child.fields.rho[0][0][0]));
    REQUIRE(std::isnan(nested.child.fields.p[5][5][1]));
    REQUIRE(report.theta.target_dry_mass_kg == Approx(report.qh.target_dry_mass_kg));
    REQUIRE(report.momentum.target_x_momentum_kgms ==
            Approx(report.momentum.committed_x_momentum_kgms + report.momentum.x_storage_residual_kgms));
    REQUIRE(report.pressure.diagnosed_cells == 32);

    for (int ci = 0; ci < 4; ++ci)
        for (int cj = 0; cj < 4; ++cj)
            for (int k = 0; k < 2; ++k)
                parent.nested_execution.child.fields.qv[ci + 1][cj + 1][k] = 0.005f;
    for (int face = 0; face <= 4; ++face)
        for (int cj = 0; cj < 4; ++cj)
            for (int k = 0; k < 2; ++k)
                parent.nested_execution.child.fields.u[face][cj + 1][k] = 1.0f;
    // This parent face/cell is outside the child physical footprint but owns
    // the east/south staggered closure coordinate. A nearest-child-owner fill
    // would incorrectly publish 5 m/s here.
    previous.u[5][2][0]                       = 7.0f;
    const auto                  energy_before = compute_moist_mixture_energy_v2(make_simulation_state_view(
        parent.nested_execution.child, parent.configuration, &parent.runtime.compute, &parent.runtime.parity));
    NestedChildRelaxationReport relaxation_report;
    apply_nested_child_physical_relaxation(parent, previous, {4.0}, 1.0, relaxation_report);
    REQUIRE(parent.nested_execution.child.fields.qv[1][2][0] == 0.01f);
    REQUIRE(parent.nested_execution.child.fields.qv[2][2][0] == 0.005f);
    REQUIRE(parent.nested_execution.child.fields.u[0][2][0] == 5.0f);
    REQUIRE(parent.nested_execution.child.fields.u[2][2][0] == 1.0f);
    REQUIRE(relaxation_report.theta.target_dry_mass_change_kg ==
            Approx(relaxation_report.qv.target_dry_mass_change_kg));
    REQUIRE(relaxation_report.momentum.target_x_change_kgms ==
            Approx(relaxation_report.momentum.committed_x_change_kgms +
                   relaxation_report.momentum.x_storage_residual_kgms));
    REQUIRE(std::isfinite(parent.nested_execution.child.fields.p[1][2][0]));
    REQUIRE(parent.nested_execution.parent_boundary_state_ready);
    REQUIRE(parent.nested_execution.interface_source.available);
    REQUIRE(parent.nested_execution.interface_source.application_count == 1);
    REQUIRE(std::isfinite(parent.nested_execution.child.fields.rho[0][0][0]));
    REQUIRE(std::isfinite(parent.nested_execution.child.fields.p[5][5][1]));
    REQUIRE(parent.nested_execution.child.fields.u[5][0][0] == 7.0f);
    const auto energy_after = compute_moist_mixture_energy_v2(make_simulation_state_view(
        parent.nested_execution.child, parent.configuration, &parent.runtime.compute, &parent.runtime.parity));
    REQUIRE(relaxation_report.energy.before.total == energy_before.total);
    REQUIRE(relaxation_report.energy.after.total == energy_after.total);
    REQUIRE(relaxation_report.energy.committed_change.total == energy_after.total - energy_before.total);
    REQUIRE(relaxation_report.energy.committed_change.total ==
            Approx(relaxation_report.energy.committed_change.dry_internal +
                   relaxation_report.energy.committed_change.water_sensible +
                   relaxation_report.energy.committed_change.phase_reference +
                   relaxation_report.energy.committed_change.resolved_kinetic +
                   relaxation_report.energy.committed_change.gravitational +
                   relaxation_report.energy.committed_change.subgrid_tke +
                   relaxation_report.energy.reservoir_sum_closure_residual_j));
    REQUIRE(relaxation_report.energy.committed_change.total_water_mass ==
            energy_after.total_water_mass - energy_before.total_water_mass);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto one_thread_energy = compute_moist_mixture_energy_v2_deterministic(make_simulation_state_view(
        parent.nested_execution.child, parent.configuration, &parent.runtime.compute, &parent.runtime.parity));
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto four_thread_energy = compute_moist_mixture_energy_v2_deterministic(make_simulation_state_view(
        parent.nested_execution.child, parent.configuration, &parent.runtime.compute, &parent.runtime.parity));
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    REQUIRE(one_thread_energy.dry_internal == four_thread_energy.dry_internal);
    REQUIRE(one_thread_energy.water_sensible == four_thread_energy.water_sensible);
    REQUIRE(one_thread_energy.phase_reference == four_thread_energy.phase_reference);
    REQUIRE(one_thread_energy.resolved_kinetic == four_thread_energy.resolved_kinetic);
    REQUIRE(one_thread_energy.gravitational == four_thread_energy.gravitational);
    REQUIRE(one_thread_energy.subgrid_tke == four_thread_energy.subgrid_tke);
    REQUIRE(one_thread_energy.total_water_mass == four_thread_energy.total_water_mass);
    REQUIRE(one_thread_energy.total == four_thread_energy.total);

    const double cumulative_energy_before =
        parent.nested_execution.interface_source.cumulative.energy.committed_change.total;
    previous.u.fill(6.0f);
    previous.v.fill(-4.0f);
    const auto momentum_before_second =
        nested_two_way_child_inventory_flat(relationship, nested.child.process_grid_metrics.dz, nested.child.fields);
    NestedChildRelaxationReport second_relaxation_report;
    apply_nested_child_physical_relaxation(parent, previous, {4.0}, 1.0, second_relaxation_report);
    const auto momentum_after_second =
        nested_two_way_child_inventory_flat(relationship, nested.child.process_grid_metrics.dz, nested.child.fields);
    REQUIRE(second_relaxation_report.momentum.committed_x_change_kgms ==
            Approx(momentum_after_second[2] - momentum_before_second[2]).epsilon(2e-13));
    REQUIRE(second_relaxation_report.momentum.committed_y_change_kgms ==
            Approx(momentum_after_second[3] - momentum_before_second[3]).epsilon(2e-13));
    REQUIRE(parent.nested_execution.interface_source.application_count == 2);
    REQUIRE(parent.nested_execution.interface_source.last.energy.committed_change.total ==
            second_relaxation_report.energy.committed_change.total);
    REQUIRE(parent.nested_execution.interface_source.cumulative.energy.committed_change.total ==
            Approx(cumulative_energy_before + second_relaxation_report.energy.committed_change.total));

    const float        accepted_qv      = nested.child.fields.qv[4][4][1];
    const float* const accepted_pointer = nested.child.fields.qv.data();
    report.qv.target_carrier_inventory  = 19.0;
    previous.qv[3][3][1]                = -0.5f;
    REQUIRE_THROWS_WITH(initialize_nested_child_physical_primary_state(parent, previous, report),
                        Catch::Matchers::Contains("outside its declared bounds"));
    REQUIRE(parent.nested_execution.child.fields.qv.data() == accepted_pointer);
    REQUIRE(parent.nested_execution.child.fields.qv[4][4][1] == accepted_qv);
    REQUIRE(report.qv.target_carrier_inventory == 19.0);

    previous.qv[3][3][1]                       = 0.01f;
    const float        accepted_tracer         = nested.child.fields.tracer[4][4][1];
    const float* const accepted_tracer_pointer = nested.child.fields.tracer.data();
    report.tracer.target_carrier_inventory     = 29.0;
    previous.tracer[3][3][1]                   = -0.25f;
    REQUIRE_THROWS_WITH(initialize_nested_child_physical_primary_state(parent, previous, report),
                        Catch::Matchers::Contains("outside its declared bounds"));
    REQUIRE(nested.child.fields.tracer.data() == accepted_tracer_pointer);
    REQUIRE(nested.child.fields.tracer[4][4][1] == accepted_tracer);
    REQUIRE(report.tracer.target_carrier_inventory == 29.0);

    previous.tracer[3][3][1] = 1.5001f;
    REQUIRE_THROWS_WITH(initialize_nested_child_physical_primary_state(parent, previous, report),
                        Catch::Matchers::Contains("child scalar target exceeds strict physical bounds"));
    REQUIRE(nested.child.fields.tracer.data() == accepted_tracer_pointer);
    REQUIRE(nested.child.fields.tracer[4][4][1] == accepted_tracer);
    REQUIRE(report.tracer.target_carrier_inventory == 29.0);
    previous.tracer[3][3][1] = 0.25f;

    previous.qv[2][2][0]                            = -0.5f;
    const float        closure_qv_before            = nested.child.fields.qv[1][1][0];
    const float* const closure_pointer              = nested.child.fields.qv.data();
    relaxation_report.energy.committed_change.total = 23.0;
    REQUIRE_THROWS_WITH(apply_nested_child_physical_relaxation(parent, previous, {4.0}, 1.0, relaxation_report),
                        Catch::Matchers::Contains("parent scalar closure target is invalid"));
    REQUIRE(nested.child.fields.qv.data() == closure_pointer);
    REQUIRE(nested.child.fields.qv[1][1][0] == closure_qv_before);
    REQUIRE(relaxation_report.energy.committed_change.total == 23.0);

    previous.qv[2][2][0]                           = 0.01f;
    const float        closure_tracer_before       = nested.child.fields.tracer[1][1][0];
    const float* const closure_tracer_pointer      = nested.child.fields.tracer.data();
    relaxation_report.tracer.target_carrier_change = 31.0;
    previous.tracer[2][2][0]                       = -0.25f;
    REQUIRE_THROWS_WITH(apply_nested_child_physical_relaxation(parent, previous, {4.0}, 1.0, relaxation_report),
                        Catch::Matchers::Contains("parent scalar closure target is invalid"));
    REQUIRE(nested.child.fields.tracer.data() == closure_tracer_pointer);
    REQUIRE(nested.child.fields.tracer[1][1][0] == closure_tracer_before);
    REQUIRE(relaxation_report.tracer.target_carrier_change == 31.0);

    previous.tracer[2][2][0] = 1.5001f;
    REQUIRE_THROWS_WITH(apply_nested_child_physical_relaxation(parent, previous, {4.0}, 1.0, relaxation_report),
                        Catch::Matchers::Contains("parent scalar closure target is invalid"));
    REQUIRE(nested.child.fields.tracer.data() == closure_tracer_pointer);
    REQUIRE(nested.child.fields.tracer[1][1][0] == closure_tracer_before);
    REQUIRE(relaxation_report.tracer.target_carrier_change == 31.0);
}

TEST_CASE("runtime rejects the initialization-only nested-grid scaffold", "[core][nested_grid][runtime]")
{
    REQUIRE_NOTHROW(require_supported_nested_grid_runtime(false));
    REQUIRE_THROWS_WITH(require_supported_nested_grid_runtime(true),
                        Catch::Matchers::Contains("no nested time integration"));
}

TEST_CASE("nested-grid initialization samples explicit parent state", "[core][nested_grid][state]")
{
    int          local_nr     = 4;
    int          local_nth    = 4;
    int          local_nz     = 4;
    double       local_dr     = 1.0;
    double       local_dz     = 1.0;
    double       local_dt     = 0.1;
    double       local_dtheta = 1.0;
    GridGeometry local_geometry;
    local_geometry.initialize(
        local_nr, local_nth, local_nz, local_dr, local_dz, local_dtheta, CoordinateSystem::Cartesian);

    Field3D local_rho(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qs(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qh(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qg(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tke(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_rad(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dqv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_du_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtke_dt_pbl(local_nr, local_nth, local_nz, 0.0f);

    fill_linear(local_rho, 1000.0f);
    fill_linear(local_p, 2000.0f);
    fill_linear(local_u, 3000.0f);
    fill_linear(local_w, 4000.0f);
    fill_linear(local_v, 5000.0f);
    fill_linear(local_tracer, 6000.0f);
    fill_linear(local_theta, 7000.0f);
    fill_linear(local_qv, 8000.0f);
    fill_linear(local_qc, 9000.0f);
    fill_linear(local_qr, 10000.0f);
    fill_linear(local_qi, 11000.0f);
    fill_linear(local_qs, 12000.0f);
    fill_linear(local_qh, 13000.0f);
    fill_linear(local_qg, 14000.0f);

    std::vector<double> local_rho0;
    std::vector<double> local_p0;
    std::vector<double> local_qv0;
    std::vector<double> local_theta0;
    std::vector<double> local_u0;
    std::vector<double> local_v0;
    double              local_coriolis = 0.0;
    double              local_time     = 0.0;

    SimulationStateView parent_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz, local_dt, local_dtheta, local_geometry},
        {local_rho,
         local_p,
         local_u,
         local_w,
         local_v,
         local_tracer,
         local_theta,
         local_qv,
         local_qc,
         local_qr,
         local_qi,
         local_qs,
         local_qh,
         local_qg,
         local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad,
         local_dtheta_dt_pbl,
         local_dqv_dt_pbl,
         local_du_dt_pbl,
         local_dv_dt_pbl,
         local_dtke_dt_pbl},
        local_coriolis,
        local_time};

    NestedGridStorage child(local_nr, local_nth, local_nz);
    child.config.enabled      = true;
    child.config.center_i     = 1;
    child.config.center_j     = 1;
    child.config.center_k     = 1;
    child.config.nest_size_r  = 2;
    child.config.nest_size_th = 2;
    child.config.nest_size_z  = 2;
    child.config.refinement   = 2.0;

    global_simulation_storage.nested.rho.resize(1, 1, 1, -999.0f);
    initialize_nested_grid(child, parent_state);

    REQUIRE(child.rho.size_r() == 2);
    REQUIRE(child.p.size_th() == 2);
    REQUIRE(child.theta.size_z() == 2);

    REQUIRE(child.rho[1][1][1] == Approx(1111.0f));
    REQUIRE(child.p[1][1][1] == Approx(2111.0f));
    REQUIRE(child.u[1][1][1] == Approx(3111.0f));
    REQUIRE(child.v[1][1][1] == Approx(5111.0f));
    REQUIRE(child.theta[1][1][1] == Approx(7111.0f));
    REQUIRE(child.qv[1][1][1] == Approx(8111.0f));
    REQUIRE(child.qi[1][1][1] == Approx(11111.0f));
    REQUIRE(child.qs[1][1][1] == Approx(12111.0f));
    REQUIRE(global_simulation_storage.nested.rho[0][0][0] == Approx(-999.0f));
}

TEST_CASE("Level snapshots reuse field storage and preserve every registered owner",
          "[core][nested_grid][ownership][memory]")
{
    auto  state = ready_process_binding_state();
    auto& cache = state.nested_execution.substep_rollback.get();
    tmv::state_ownership::capture_level(cache, state.nested_execution.child);
    const auto                   fields = tmv::fields::pointers(cache.fields, tmv::fields::prognostic);
    std::array<const float*, 15> addresses{};
    for (std::size_t index = 0; index < fields.size(); ++index)
        addresses[index] = fields[index]->data();
    for (int step = 0; step < 8; ++step)
    {
        state.nested_execution.child.fields.theta.fill(300.0f + step);
        tmv::state_ownership::capture_level(cache, state.nested_execution.child);
        REQUIRE(cache.fields.theta[1][1][0] == 300.0f + step);
        for (std::size_t index = 0; index < fields.size(); ++index)
            REQUIRE(fields[index]->data() == addresses[index]);
    }
    const auto before = state.nested_execution.child.fields.theta;
    state.nested_execution.child.fields.theta.fill(500.0f);
    tmv::state_ownership::exchange_levels(cache, state.nested_execution.child);
    REQUIRE(std::memcmp(
                before.data(), state.nested_execution.child.fields.theta.data(), before.size() * sizeof(float)) == 0);
}

TEST_CASE("Reusable level snapshots have bounded bulk allocation", "[core][nested_grid][snapshot_cost]")
{
    AtmosphericLevelStorage source{{64, 32, 16, 100.0, 100.0, 1.0, 100.0}};
    tmv::fields::visit(
        source.fields, tmv::fields::prognostic, [](const char*, Field3D& field) { field.resize(64, 32, 16, 1.0f); });
    constexpr int iterations = 64;
    double        checksum   = 0.0;
    const auto    begin      = std::chrono::steady_clock::now();
    for (int step = 0; step < iterations; ++step)
    {
        AtmosphericLevelStorage copied(source);
        checksum += copied.fields.rho[0][0][0];
    }
    const auto              middle = std::chrono::steady_clock::now();
    AtmosphericLevelStorage reused;
    tmv::state_ownership::capture_level(reused, source);
    const auto* pointer = reused.fields.rho.data();
    for (int step = 0; step < iterations; ++step)
    {
        tmv::state_ownership::capture_level(reused, source);
        checksum += reused.fields.rho[0][0][0];
        REQUIRE(reused.fields.rho.data() == pointer);
    }
    const auto end = std::chrono::steady_clock::now();
    REQUIRE(checksum == 2 * iterations);
    std::cout << "[level-snapshot-cost] cells=32768 iterations=" << iterations
              << " fresh_copy_s=" << std::chrono::duration<double>(middle - begin).count()
              << " reusable_copy_s=" << std::chrono::duration<double>(end - middle).count() << '\n';
}

TEST_CASE("mapped nesting preserves uniform face winds at both refinement ratios",
          "[core][nested_grid][active_terrain][uniform-wind]")
{
    for (int ratio : {2, 3})
    {
        auto state           = ready_process_binding_state();
        auto input           = small_relationship_input();
        input.parent_ratio_x = input.parent_ratio_y = input.parent_time_step_ratio = ratio;
        input.child_cells_x = input.child_cells_y = 2 * ratio;
        const auto relationship                   = tmv::domain::prepare_one_way_nest_relationship(input);
        prepare_nested_execution_relationship(state, relationship, 0.0);
        initialize_nested_shared_level_reference_state(state);
        tmv::TerrainConfig config;
        config.scheme_id               = "bell";
        config.coord_id                = "btf";
        config.horizontal_placement_id = "grid_attached";
        config.ztop                    = 550;
        config.bell.h0                 = 0;
        config.bell.a                  = 1200;
        auto& nested                   = state.nested_execution;
        nested.active_terrain.geometry =
            tmv::domain::build_stationary_nested_terrain_geometry(config, relationship, {0, 0}, 1);
        nested.child.runtime.terrain.config     = config;
        nested.child.runtime.terrain.topography = nested.active_terrain.geometry.child_storage.topography;
        nested.child.runtime.terrain.metrics    = nested.active_terrain.geometry.child_storage.metrics;
        prepare_nested_child_process_grid_metrics(nested);
        state.fields.u.fill(5);
        state.fields.v.fill(-3);
        const auto                      previous = state.fields;
        NestedChildInitializationReport initialization;
        initialize_nested_child_physical_primary_state(state, previous, initialization);
        const auto& child = nested.child.fields;
        for (int face = 0; face <= relationship.child_cells_x; ++face)
            for (int transverse = 1; transverse <= relationship.child_cells_y; ++transverse)
                for (int k = 0; k < relationship.levels_z; ++k)
                {
                    REQUIRE(child.u(face, transverse, k) == Approx(5));
                    REQUIRE(child.v(transverse, face, k) == Approx(-3));
                }
        nested.child.fields.u.fill(1);
        nested.child.fields.v.fill(-1);
        const auto physical_impulse = [&]()
        {
            std::array<double, 2> result{};
            const auto&           m = nested.active_terrain.geometry.child_storage.metrics;
            for (int i = 1; i <= relationship.child_cells_x; ++i)
                for (int j = 1; j <= relationship.child_cells_y; ++j)
                    for (int k = 0; k < relationship.levels_z; ++k)
                    {
                        const double half_mass = .5 * child.rho(i, j, k) * m.J(i, j, k) * m.dx * m.dy * m.dzeta;
                        result[0] += half_mass * (child.u(i - 1, j, k) + child.u(i, j, k));
                        result[1] += half_mass * (child.v(i, j - 1, k) + child.v(i, j, k));
                    }
            return result;
        };
        const auto                  before = physical_impulse();
        NestedChildRelaxationReport relaxation;
        apply_nested_child_physical_relaxation(state, previous, {4.0}, 1.0, relaxation);
        const auto after = physical_impulse();
        REQUIRE(child.u(0, 1, 0) == Approx(5));
        REQUIRE(child.u(relationship.child_cells_x, 1, 0) == Approx(5));
        REQUIRE(child.v(1, 0, 0) == Approx(-3));
        REQUIRE(child.v(1, relationship.child_cells_y, 0) == Approx(-3));
        REQUIRE(after[0] - before[0] == Approx(relaxation.momentum.committed_x_change_kgms).epsilon(1e-12));
        REQUIRE(after[1] - before[1] == Approx(relaxation.momentum.committed_y_change_kgms).epsilon(1e-12));
        for (int i = 0; i < state.grid.nr; ++i)
            for (int j = 0; j < state.grid.nth; ++j)
                for (int k = 0; k < state.grid.nz; ++k)
                {
                    state.fields.u(i, j, k) = float(i);
                    state.fields.v(i, j, k) = -float(j);
                }
        const auto varying                 = state.fields;
        nested.parent_boundary_state_ready = false;
        nested.interface_source            = {};
        initialize_nested_child_physical_primary_state(state, varying, initialization);
        const float west = child.u(0, 1, 0), south = child.v(1, 0, 0);
        REQUIRE(west != child.u(1, 1, 0));
        REQUIRE(south != child.v(1, 1, 0));
        nested.child.fields.u.fill(1);
        nested.child.fields.v.fill(-1);
        apply_nested_child_physical_relaxation(state, varying, {4.0}, 1.0, relaxation);
        REQUIRE(child.u(0, 1, 0) == west);
        REQUIRE(child.v(1, 0, 0) == south);
    }
}

TEST_CASE("two-way reflux audit retains small fluxes between cancelling faces",
          "[core][nested_grid][two_way][reflux][cancellation]")
{
    const auto             r = tmv::domain::prepare_one_way_nest_relationship(small_relationship_input());
    PrognosticFieldStorage parent, child;
    fill_valid_primary_state(parent, 8, 8, 2, 1.0f);
    fill_valid_primary_state(child, 6, 6, 2, 1.0f);
    constexpr double large  = 0x1p26;
    constexpr double small  = 0x1p-30;
    const auto       scalar = [&](auto& coarse, auto& fine)
    {
        for (auto* fluxes : {&coarse, &fine})
        {
            fluxes->west.assign(4, 0);
            fluxes->east.assign(4, 0);
            fluxes->south.assign(4, 0);
            fluxes->north.assign(4, 0);
            fluxes->accumulated_dt_s   = r.parent_dt_s;
            fluxes->contribution_count = 1;
        }
        coarse.west[0] = large;
        coarse.west[1] = -large;
        fine.east[0]   = small;
    };
    tmv::CartesianDryMassInterfaceFluxes parent_mass, child_mass;
    tmv::CartesianThetaInterfaceFluxes   parent_theta, child_theta;
    scalar(parent_mass, child_mass);
    scalar(parent_theta, child_theta);
    tmv::CartesianMomentumInterfaceFluxes parent_advective, child_advective, parent_pressure, child_pressure;
    for (auto* fluxes : {&parent_advective, &child_advective, &parent_pressure, &child_pressure})
    {
        for (std::size_t component = 0; component < 3; ++component)
        {
            fluxes->west.component[component].assign(2, 0);
            fluxes->east.component[component].assign(2, 0);
            fluxes->south.component[component].assign(2, 0);
            fluxes->north.component[component].assign(2, 0);
        }
        fluxes->accumulated_dt_s   = r.parent_dt_s;
        fluxes->contribution_count = 1;
    }
    for (std::size_t component = 0; component < 3; ++component)
    {
        parent_advective.west.component[component][0] = large;
        parent_advective.west.component[component][1] = -large;
        child_advective.east.component[component][0]  = small;
    }
    const std::vector<double>    layers{300, 300};
    const auto                   begin = nested_two_way_child_inventory_flat(r, layers, child);
    NestedTwoWayFeedbackReportV1 report;
    apply_nested_two_way_feedback_flat(r,
                                       layers,
                                       parent_mass,
                                       child_mass,
                                       parent_theta,
                                       child_theta,
                                       parent_advective,
                                       parent_pressure,
                                       child_advective,
                                       child_pressure,
                                       {},
                                       {},
                                       {},
                                       begin,
                                       parent,
                                       child,
                                       report);
    REQUIRE(report.complete);
    for (std::size_t currency = 0; currency < 5; ++currency)
    {
        CHECK(report.applied_reflux[currency] == small);
        CHECK(report.interface_mismatch[currency] == small);
    }
}
