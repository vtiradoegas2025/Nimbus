#include "catch2/catch.hpp"
#include "core/infra/nested_boundary_forcing.hpp"
#include "dynamics/eos_pressure.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace
{

tmv::domain::OneWayNestRelationshipV1 relationship()
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = 6;
    input.parent_cells_y = 5;
    input.parent_levels_z = 2;
    input.parent_dx_m = 900.0;
    input.parent_dy_m = 600.0;
    input.parent_dt_s = 3.0;
    input.parent_ratio_x = 3;
    input.parent_ratio_y = 2;
    input.parent_time_step_ratio = 3;
    input.child_origin_parent_cell_i = 1;
    input.child_origin_parent_cell_j = 1;
    input.child_cells_x = 6;
    input.child_cells_y = 4;
    input.child_levels_z = 2;
    input.relaxation_width_child_cells = 1;
    input.boundary_stencil_parent_cells = 1;
    return tmv::domain::prepare_one_way_nest_relationship(input);
}

tmv::domain::OneWayNestRelationshipV1 momentum_relationship()
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = 4;
    input.parent_cells_y = 4;
    input.parent_levels_z = 2;
    input.parent_dx_m = 1.0;
    input.parent_dy_m = 1.0;
    input.parent_dt_s = 3.0;
    input.parent_ratio_x = 2;
    input.parent_ratio_y = 2;
    input.parent_time_step_ratio = 2;
    input.child_origin_parent_cell_i = 1;
    input.child_origin_parent_cell_j = 1;
    input.child_cells_x = 4;
    input.child_cells_y = 4;
    input.child_levels_z = 2;
    input.relaxation_width_child_cells = 1;
    input.boundary_stencil_parent_cells = 1;
    return tmv::domain::prepare_one_way_nest_relationship(input);
}

tmv::domain::OneWayNestRelationshipV1 relaxation_relationship()
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x = 7;
    input.parent_cells_y = 7;
    input.parent_levels_z = 2;
    input.parent_dx_m = 1000.0;
    input.parent_dy_m = 800.0;
    input.parent_dt_s = 2.0;
    input.parent_ratio_x = 2;
    input.parent_ratio_y = 2;
    input.parent_time_step_ratio = 2;
    input.child_origin_parent_cell_i = 2;
    input.child_origin_parent_cell_j = 2;
    input.child_cells_x = 6;
    input.child_cells_y = 6;
    input.child_levels_z = 2;
    input.relaxation_width_child_cells = 2;
    input.boundary_stencil_parent_cells = 1;
    return tmv::domain::prepare_one_way_nest_relationship(input);
}

void fill_parent(Field3D& rho_previous,
                 Field3D& rho_current,
                 Field3D& specific_previous,
                 Field3D& specific_current)
{
    for (int i = 0; i < rho_previous.size_r(); ++i)
    {
        for (int j = 0; j < rho_previous.size_th(); ++j)
        {
            for (int k = 0; k < rho_previous.size_z(); ++k)
            {
                rho_previous[i][j][k] =
                    1.0f + 0.1f * i + 0.02f * j + 0.01f * k;
                rho_current[i][j][k] =
                    1.2f + 0.08f * i + 0.03f * j + 0.02f * k;
                specific_previous[i][j][k] =
                    0.001f * (1.0f + i + j + k);
                specific_current[i][j][k] =
                    0.002f * (1.0f + i + 2.0f * j + k);
            }
        }
    }
}

tmv::domain::ParentTimeBracketV1 boundary_bracket(
    const tmv::domain::OneWayNestRelationshipV1& relationship,
    int completed_child_substeps)
{
    auto bracket = tmv::domain::prepare_parent_time_bracket(
        relationship, 0.0, relationship.parent_dt_s);
    for (int step = 0; step < completed_child_substeps; ++step)
        tmv::domain::advance_child_substep(relationship, bracket);
    return bracket;
}

} // namespace

TEST_CASE("conserved scalar prolongation closes density and carrier inventory",
          "[core][nested_grid][boundary][conservation]")
{
    const auto rel = relationship();
    Field3D rho_previous(6, 5, 2);
    Field3D rho_current(6, 5, 2);
    Field3D q_previous(6, 5, 2);
    Field3D q_current(6, 5, 2);
    fill_parent(rho_previous, rho_current, q_previous, q_current);
    Field3D child_rho(6, 4, 2, -77.0f);
    Field3D child_q(6, 4, 2, -77.0f);
    tmv::domain::ConservedScalarProlongationReport report;
    const auto bracket = boundary_bracket(rel, 1);
    const auto weights = tmv::domain::parent_temporal_weights(rel, bracket);

    tmv::domain::prolong_conserved_scalar_flat(
        rel, bracket, {200.0, 350.0}, rho_previous, rho_current,
        q_previous, q_current, true, child_rho, child_q, report);

    for (int ci = 0; ci < 6; ++ci)
    {
        const int pi = 1 + ci / 3;
        for (int cj = 0; cj < 4; ++cj)
        {
            const int pj = 1 + cj / 2;
            for (int k = 0; k < 2; ++k)
            {
                const double expected_rho =
                    weights.previous * rho_previous[pi][pj][k] +
                    weights.current * rho_current[pi][pj][k];
                const double expected_carrier =
                    weights.previous * rho_previous[pi][pj][k] * q_previous[pi][pj][k] +
                    weights.current * rho_current[pi][pj][k] * q_current[pi][pj][k];
                REQUIRE(child_rho[ci][cj][k] == Approx(expected_rho));
                REQUIRE(child_q[ci][cj][k] == Approx(expected_carrier / expected_rho));
                REQUIRE(child_q[ci][cj][k] >=
                        std::min(q_previous[pi][pj][k], q_current[pi][pj][k]));
                REQUIRE(child_q[ci][cj][k] <=
                        std::max(q_previous[pi][pj][k], q_current[pi][pj][k]));
            }
        }
    }
    REQUIRE(report.parent_cells == 8);
    REQUIRE(report.child_cells == 48);
    REQUIRE(report.target_dry_mass_kg ==
            Approx(report.committed_dry_mass_kg + report.dry_storage_residual_kg));
    REQUIRE(report.target_carrier_inventory ==
            Approx(report.committed_carrier_inventory + report.carrier_storage_residual));
}

TEST_CASE("conserved scalar prolongation preserves exact temporal endpoints",
          "[core][nested_grid][boundary][endpoint]")
{
    const auto rel = relationship();
    Field3D rho_previous(6, 5, 2);
    Field3D rho_current(6, 5, 2);
    Field3D q_previous(6, 5, 2);
    Field3D q_current(6, 5, 2);
    fill_parent(rho_previous, rho_current, q_previous, q_current);
    Field3D child_rho(6, 4, 2);
    Field3D child_q(6, 4, 2);
    tmv::domain::ConservedScalarProlongationReport report;

    rho_current[1][1][0] = std::numeric_limits<float>::quiet_NaN();
    q_current[1][1][0] = -1.0f;

    tmv::domain::prolong_conserved_scalar_flat(
        rel, boundary_bracket(rel, 0), {200.0, 350.0},
        rho_previous, rho_current, q_previous, q_current, true,
        child_rho, child_q, report);

    for (int ci = 0; ci < 6; ++ci)
    {
        for (int cj = 0; cj < 4; ++cj)
        {
            for (int k = 0; k < 2; ++k)
            {
                const int pi = 1 + ci / 3;
                const int pj = 1 + cj / 2;
                REQUIRE(child_rho[ci][cj][k] == rho_previous[pi][pj][k]);
                REQUIRE(child_q[ci][cj][k] == q_previous[pi][pj][k]);
            }
        }
    }
}

TEST_CASE("conserved scalar prolongation rejects a corrupt temporal bracket",
          "[core][nested_grid][boundary][clock]")
{
    const auto rel = relationship();
    Field3D rho_previous(6, 5, 2);
    Field3D rho_current(6, 5, 2);
    Field3D q_previous(6, 5, 2);
    Field3D q_current(6, 5, 2);
    fill_parent(rho_previous, rho_current, q_previous, q_current);
    Field3D child_rho(6, 4, 2, 7.0f);
    Field3D child_q(6, 4, 2, 8.0f);
    tmv::domain::ConservedScalarProlongationReport report;
    auto bracket = boundary_bracket(rel, 1);
    bracket.completed_child_substeps = 4;

    REQUIRE_THROWS_WITH(
        tmv::domain::prolong_conserved_scalar_flat(
            rel, bracket, {200.0, 350.0}, rho_previous, rho_current,
            q_previous, q_current, true, child_rho, child_q, report),
        Catch::Matchers::Contains("invalid parent time bracket"));
    REQUIRE(child_rho[0][0][0] == 7.0f);
    REQUIRE(child_q[0][0][0] == 8.0f);
}

TEST_CASE("conserved scalar prolongation rejects before mutating output",
          "[core][nested_grid][boundary][transaction]")
{
    const auto rel = relationship();
    Field3D rho_previous(6, 5, 2);
    Field3D rho_current(6, 5, 2);
    Field3D q_previous(6, 5, 2);
    Field3D q_current(6, 5, 2);
    fill_parent(rho_previous, rho_current, q_previous, q_current);
    q_current[1][1][0] = -0.25f;
    Field3D child_rho(6, 4, 2, 7.0f);
    Field3D child_q(6, 4, 2, 8.0f);
    tmv::domain::ConservedScalarProlongationReport report;
    report.target_dry_mass_kg = 91.0;

    REQUIRE_THROWS_WITH(
        tmv::domain::prolong_conserved_scalar_flat(
            rel, boundary_bracket(rel, 1),
            {200.0, 350.0}, rho_previous, rho_current, q_previous,
            q_current, true, child_rho, child_q, report),
        Catch::Matchers::Contains("outside its declared bounds"));
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 2; ++k)
            {
                REQUIRE(child_rho[i][j][k] == 7.0f);
                REQUIRE(child_q[i][j][k] == 8.0f);
            }
    REQUIRE(report.target_dry_mass_kg == 91.0);

    q_current[1][1][0] = 0.001f;
    REQUIRE_THROWS_WITH(
        tmv::domain::prolong_conserved_scalar_flat(
            rel, boundary_bracket(rel, 1),
            {std::numeric_limits<double>::max(), 350.0}, rho_previous,
            rho_current, q_previous, q_current, true, child_rho, child_q,
            report),
        Catch::Matchers::Contains("volume is not finite"));
    REQUIRE(child_rho[0][0][0] == 7.0f);
    REQUIRE(child_q[0][0][0] == 8.0f);
    REQUIRE(report.target_dry_mass_kg == 91.0);
}

TEST_CASE("horizontal face prolongation preserves clipped dual inventories",
          "[core][nested_grid][boundary][c_grid][conservation]")
{
    const auto rel = relationship();

    SECTION("x-face constant state and half-width endpoints")
    {
        Field3D previous(7, 5, 2, -4.0f);
        Field3D current(7, 5, 2, 9.0f);
        Field3D child(7, 4, 2, 91.0f);
        tmv::domain::ConservedFaceProlongationReport report;
        tmv::domain::prolong_conserved_horizontal_face_flat(
            rel, boundary_bracket(rel, 0),
            tmv::domain::NestedHorizontalLocation::XFace,
            {200.0, 350.0}, previous, current, child, report);

        for (int i = 0; i < child.size_r(); ++i)
            for (int j = 0; j < child.size_th(); ++j)
                for (int k = 0; k < child.size_z(); ++k)
                    REQUIRE(child[i][j][k] == -4.0f);
        REQUIRE(report.parent_owners == 12);
        REQUIRE(report.child_owners == 56);
        REQUIRE(report.target_inventory == Approx(report.committed_inventory));
        REQUIRE(report.storage_residual == Approx(0.0).margin(1.0e-5));
    }

    SECTION("y-face overlap uses native dual owners")
    {
        Field3D previous(6, 6, 2);
        Field3D current(6, 6, 2,
                        std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < previous.size_r(); ++i)
            for (int j = 0; j < previous.size_th(); ++j)
                for (int k = 0; k < previous.size_z(); ++k)
                    previous[i][j][k] =
                        static_cast<float>(-20 + i + 2 * j + k);
        Field3D child(6, 5, 2, 91.0f);
        tmv::domain::ConservedFaceProlongationReport report;
        tmv::domain::prolong_conserved_horizontal_face_flat(
            rel, boundary_bracket(rel, 0),
            tmv::domain::NestedHorizontalLocation::YFace,
            {200.0, 350.0}, previous, current, child, report);

        const int parent_i = 1;
        const int first_parent_face = 1;
        REQUIRE(child[0][0][0] ==
                previous[parent_i][first_parent_face][0]);
        REQUIRE(child[0][1][0] == Approx(
            0.5 * (previous[parent_i][first_parent_face][0] +
                   previous[parent_i][first_parent_face + 1][0])));
        REQUIRE(child[0][4][0] ==
                previous[parent_i][first_parent_face + 2][0]);
        REQUIRE(report.parent_owners == 12);
        REQUIRE(report.child_owners == 60);
        REQUIRE(report.target_inventory ==
                Approx(report.committed_inventory + report.storage_residual));
    }
}

TEST_CASE("horizontal face prolongation rejects unsupported owners transactionally",
          "[core][nested_grid][boundary][c_grid][transaction]")
{
    const auto rel = relationship();
    Field3D previous(7, 5, 2, 1.0f);
    Field3D current(7, 5, 2, 2.0f);
    Field3D child(7, 4, 2, 91.0f);
    tmv::domain::ConservedFaceProlongationReport report;
    report.target_inventory = 17.0;
    REQUIRE_THROWS_WITH(
        tmv::domain::prolong_conserved_horizontal_face_flat(
            rel, boundary_bracket(rel, 1),
            tmv::domain::NestedHorizontalLocation::ScalarCenter,
            {200.0, 350.0}, previous, current, child, report),
        Catch::Matchers::Contains("must be an x or y face"));
    REQUIRE(child[0][0][0] == 91.0f);
    REQUIRE(report.target_inventory == 17.0);
}

TEST_CASE("native C-grid momentum uses committed runtime dual mass",
          "[core][nested_grid][boundary][c_grid][momentum]")
{
    const auto rel = momentum_relationship();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Field3D rho_previous(6, 6, 2, nan);
    Field3D rho_current(6, 6, 2, nan);
    Field3D u_previous(6, 6, 2, nan);
    Field3D u_current(6, 6, 2, nan);
    Field3D v_previous(6, 6, 2, nan);
    Field3D v_current(6, 6, 2, nan);
    Field3D w_previous(6, 6, 2, nan);
    Field3D w_current(6, 6, 2, nan);

    for (int pj = 1; pj <= 2; ++pj)
    {
        for (int k = 0; k < 2; ++k)
        {
            rho_previous[2][pj + 1][k] = 1.0f;
            rho_previous[3][pj + 1][k] = 3.0f;
            u_previous[1][pj + 1][k] = 0.0f;
            u_previous[2][pj + 1][k] = k == 0 ? 10.0f : 0.0f;
            u_previous[3][pj + 1][k] = 0.0f;
        }
        for (int pi = 1; pi <= 2; ++pi)
        {
            for (int k = 0; k < 2; ++k)
            {
                v_previous[pi + 1][1][k] = 0.0f;
                v_previous[pi + 1][2][k] = 0.0f;
                v_previous[pi + 1][3][k] = 0.0f;
                w_previous[pi + 1][pj + 1][k] = 0.0f;
            }
        }
    }

    Field3D child_rho(6, 6, 2, nan);
    for (int ci = 0; ci < 4; ++ci)
        for (int cj = 0; cj < 4; ++cj)
            for (int k = 0; k < 2; ++k)
                child_rho[ci + 1][cj + 1][k] = ci < 2 ? 1.0f : 3.0f;
    Field3D child_u(6, 6, 2, 91.0f);
    Field3D child_v(6, 6, 2, 92.0f);
    Field3D child_w(6, 6, 2, 93.0f);
    tmv::domain::ConservedMomentumProlongationReport report;

    tmv::domain::prolong_cgrid_momentum_flat_storage(
        rel, boundary_bracket(rel, 0), {1.0, 1.0},
        rho_previous, rho_current, u_previous, u_current,
        v_previous, v_current, w_previous, w_current, child_rho,
        child_u, child_v, child_w, report);

    const float expected_u[] = {0.0f, 5.0f, 10.0f, 5.0f, 0.0f};
    for (int cj = 0; cj < 4; ++cj)
        for (int face = 0; face <= 4; ++face)
        {
            REQUIRE(child_u[face][cj + 1][0] == expected_u[face]);
            REQUIRE(child_u[face][cj + 1][1] == 0.0f);
        }
    for (int ci = 0; ci < 4; ++ci)
        for (int face = 0; face <= 4; ++face)
            REQUIRE(child_v[ci + 1][face][0] == 0.0f);
    for (int ci = 0; ci < 4; ++ci)
        for (int cj = 0; cj < 4; ++cj)
            REQUIRE(child_w[ci + 1][cj + 1][0] == 0.0f);

    REQUIRE(report.target_x_momentum_kgms == Approx(40.0));
    REQUIRE(report.target_x_momentum_kgms == Approx(
        report.committed_x_momentum_kgms + report.x_storage_residual_kgms));
    REQUIRE(report.target_x_dual_dry_mass_kg == Approx(
        report.committed_x_dual_dry_mass_kg +
        report.x_dry_mass_residual_kg));
    REQUIRE(report.target_y_momentum_kgms == Approx(0.0));
    REQUIRE(report.target_z_momentum_kgms == Approx(0.0));
    REQUIRE(report.child_x_faces == 40);
    REQUIRE(report.child_y_faces == 40);
    REQUIRE(report.child_z_faces == 32);

    // Only native physical owners are published.
    REQUIRE(child_u[5][0][0] == 91.0f);
    REQUIRE(child_v[0][5][0] == 92.0f);
    REQUIRE(child_w[0][0][0] == 93.0f);
}

TEST_CASE("native C-grid momentum rejects stale child mass transactionally",
          "[core][nested_grid][boundary][c_grid][transaction]")
{
    const auto rel = momentum_relationship();
    Field3D rho_previous(6, 6, 2, 1.0f);
    Field3D rho_current(6, 6, 2, 1.0f);
    Field3D u_previous(6, 6, 2, 2.0f);
    Field3D u_current(6, 6, 2, 2.0f);
    Field3D v_previous(6, 6, 2, 3.0f);
    Field3D v_current(6, 6, 2, 3.0f);
    Field3D w_previous(6, 6, 2, 4.0f);
    Field3D w_current(6, 6, 2, 4.0f);
    Field3D child_rho(6, 6, 2, 1.0f);
    child_rho[4][4][0] = 1.5f;
    Field3D child_u(6, 6, 2, 91.0f);
    Field3D child_v(6, 6, 2, 92.0f);
    Field3D child_w(6, 6, 2, 93.0f);
    tmv::domain::ConservedMomentumProlongationReport report;
    report.target_x_momentum_kgms = 17.0;

    REQUIRE_THROWS_WITH(
        tmv::domain::prolong_cgrid_momentum_flat_storage(
            rel, boundary_bracket(rel, 1), {1.0, 1.0},
            rho_previous, rho_current, u_previous, u_current,
            v_previous, v_current, w_previous, w_current, child_rho,
            child_u, child_v, child_w, report),
        Catch::Matchers::Contains("canonical parent target"));
    REQUIRE(child_u[0][1][0] == 91.0f);
    REQUIRE(child_v[1][0][0] == 92.0f);
    REQUIRE(child_w[1][1][0] == 93.0f);
    REQUIRE(report.target_x_momentum_kgms == 17.0);
}

TEST_CASE("child pressure diagnosis is dry-EOS exact and transactional",
          "[core][nested_grid][boundary][eos]")
{
    const auto rel = momentum_relationship();
    Field3D child_rho(6, 6, 2, -1.0f);
    Field3D child_theta(6, 6, 2, -1.0f);
    for (int ci = 1; ci <= 4; ++ci)
        for (int cj = 1; cj <= 4; ++cj)
        {
            for (int k = 0; k < 2; ++k)
            {
                child_rho[ci][cj][k] = 1.0f;
                child_theta[ci][cj][k] = 300.0f;
            }
        }
    Field3D pressure(6, 6, 2, 91.0f);
    tmv::domain::PressureDiagnosisReport report;
    tmv::domain::diagnose_child_pressure_flat_storage(
        rel, {125.0, 175.0}, child_rho, child_theta, pressure, report);

    const float expected = static_cast<float>(
        tmv::dynamics_eos::pressure_fp64(1.0f, 300.0f));
    for (int ci = 1; ci <= 4; ++ci)
        for (int cj = 1; cj <= 4; ++cj)
            for (int k = 0; k < 2; ++k)
                REQUIRE(pressure[ci][cj][k] == expected);
    REQUIRE(pressure[0][0][0] == 91.0f);
    REQUIRE(pressure[5][5][0] == 91.0f);
    REQUIRE(report.diagnosed_cells == 32);
    REQUIRE(report.target_pressure_volume_integral_pa_m3 == Approx(
        report.committed_pressure_volume_integral_pa_m3 +
        report.storage_residual_pa_m3));

    pressure.fill(77.0f);
    report.target_pressure_volume_integral_pa_m3 = 19.0;
    child_theta[4][4][0] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_WITH(
        tmv::domain::diagnose_child_pressure_flat_storage(
            rel, {125.0, 175.0}, child_rho, child_theta, pressure, report),
        Catch::Matchers::Contains("does not define pressure"));
    REQUIRE(pressure[1][1][0] == 77.0f);
    REQUIRE(pressure[4][4][0] == 77.0f);
    REQUIRE(report.target_pressure_volume_integral_pa_m3 == 19.0);
}

TEST_CASE("conservative forcing accepts a shortened realized parent bracket",
          "[core][nested_grid][boundary][clock]")
{
    const auto rel = relationship();
    auto bracket = tmv::domain::prepare_parent_time_bracket(rel, 4.0, 6.25);
    tmv::domain::advance_child_substep(rel, bracket);
    const auto weights = tmv::domain::parent_temporal_weights(rel, bracket);
    REQUIRE(weights.offset_from_previous_s == Approx(0.75));

    Field3D rho_previous(6, 5, 2, 1.0f);
    Field3D rho_current(6, 5, 2, 2.0f);
    Field3D q_previous(6, 5, 2, 0.01f);
    Field3D q_current(6, 5, 2, 0.02f);
    Field3D child_rho(6, 4, 2);
    Field3D child_q(6, 4, 2);
    tmv::domain::ConservedScalarProlongationReport scalar_report;
    tmv::domain::prolong_conserved_scalar_flat(
        rel, bracket, {200.0, 350.0}, rho_previous, rho_current,
        q_previous, q_current, true, child_rho, child_q, scalar_report);
    REQUIRE(child_rho[0][0][0] == Approx(4.0 / 3.0));

    Field3D face_previous(7, 5, 2, 3.0f);
    Field3D face_current(7, 5, 2, 6.0f);
    Field3D child_face(7, 4, 2);
    tmv::domain::ConservedFaceProlongationReport face_report;
    tmv::domain::prolong_conserved_horizontal_face_flat(
        rel, bracket, tmv::domain::NestedHorizontalLocation::XFace,
        {200.0, 350.0}, face_previous, face_current,
        child_face, face_report);
    REQUIRE(child_face[0][0][0] == Approx(4.0));
}

TEST_CASE("nested relaxation profile is exact at the ring and rate based inside",
          "[core][nested_grid][boundary][relaxation][clock]")
{
    const auto rel = relaxation_relationship();
    const tmv::domain::NestedRelaxationConfigV1 config{4.0};
    REQUIRE(tmv::domain::nested_scalar_relaxation_fraction(
                rel, config, 0, 3, 1.0) == 1.0);
    const double half = tmv::domain::nested_scalar_relaxation_fraction(
        rel, config, 1, 3, 0.5);
    const double full = tmv::domain::nested_scalar_relaxation_fraction(
        rel, config, 1, 3, 1.0);
    REQUIRE(full == Approx(1.0 - (1.0 - half) * (1.0 - half)).epsilon(1e-13));
    REQUIRE(tmv::domain::nested_scalar_relaxation_fraction(
                rel, config, 2, 2, 1.0) == 0.0);
    REQUIRE(tmv::domain::nested_scalar_relaxation_fraction(
                rel, config, 0, 3, 0.0) == 1.0);
    REQUIRE(tmv::domain::nested_scalar_relaxation_fraction(
                rel, config, 1, 3, 0.0) == 0.0);
    REQUIRE_THROWS_WITH(
        tmv::domain::nested_scalar_relaxation_fraction(
            rel, {0.0}, 1, 1, 1.0),
        Catch::Matchers::Contains("configuration"));
}

TEST_CASE("momentum relaxation reports the committed full staggered-face inventory",
          "[core][nested_grid][boundary][relaxation][momentum][conservation]")
{
    const auto rel = relaxation_relationship();
    Field3D old_rho(8, 8, 2, 1.0f);
    Field3D target_rho(8, 8, 2, 1.2f);
    Field3D new_rho(8, 8, 2, 1.1f);
    Field3D old_u(8, 8, 2, 2.0f);
    Field3D old_v(8, 8, 2, -3.0f);
    Field3D old_w(8, 8, 2, 0.5f);
    Field3D target_u(8, 8, 2, 5.0f);
    Field3D target_v(8, 8, 2, -7.0f);
    Field3D target_w(8, 8, 2, 1.5f);
    Field3D candidate_u = old_u;
    Field3D candidate_v = old_v;
    Field3D candidate_w = old_w;
    for (int j = 1; j <= rel.child_cells_y; ++j)
        for (int k = 0; k < rel.levels_z; ++k)
        {
            old_rho[0][j][k] = 1.7f;
            old_rho[rel.child_cells_x + 1][j][k] = 2.1f;
            new_rho[0][j][k] = 1.9f;
            new_rho[rel.child_cells_x + 1][j][k] = 2.4f;
        }
    for (int i = 1; i <= rel.child_cells_x; ++i)
        for (int k = 0; k < rel.levels_z; ++k)
        {
            old_rho[i][0][k] = 2.5f;
            old_rho[i][rel.child_cells_y + 1][k] = 3.0f;
            new_rho[i][0][k] = 2.8f;
            new_rho[i][rel.child_cells_y + 1][k] = 3.4f;
        }
    const auto horizontal_inventory = [&](const Field3D& rho,
                                          const Field3D& u,
                                          const Field3D& v) {
        std::array<double, 2> result{};
        const double area = rel.child_dx_m * rel.child_dy_m;
        const std::array<double, 2> dz{200.0, 350.0};
        for (int face = 0; face <= rel.child_cells_x; ++face)
            for (int j = 1; j <= rel.child_cells_y; ++j)
                for (int k = 0; k < rel.levels_z; ++k)
                    result[0] += 0.5 * area * dz[k] *
                        (static_cast<double>(rho[face][j][k]) +
                         static_cast<double>(rho[face + 1][j][k])) *
                        u[face][j][k];
        for (int i = 1; i <= rel.child_cells_x; ++i)
            for (int face = 0; face <= rel.child_cells_y; ++face)
                for (int k = 0; k < rel.levels_z; ++k)
                    result[1] += 0.5 * area * dz[k] *
                        (static_cast<double>(rho[i][face][k]) +
                         static_cast<double>(rho[i][face + 1][k])) *
                        v[i][face][k];
        return result;
    };
    const auto before = horizontal_inventory(old_rho, old_u, old_v);
    tmv::domain::ConservedMomentumRelaxationReport report;
    tmv::domain::relax_cgrid_momentum_flat_storage(
        rel, {4.0}, 1.0, {200.0, 350.0},
        old_rho, old_u, old_v, old_w,
        target_rho, target_u, target_v, target_w,
        new_rho, candidate_u, candidate_v, candidate_w, report);
    const auto after = horizontal_inventory(new_rho, candidate_u, candidate_v);
    REQUIRE(report.committed_x_change_kgms ==
            Approx(after[0] - before[0]).epsilon(2e-13));
    REQUIRE(report.committed_y_change_kgms ==
            Approx(after[1] - before[1]).epsilon(2e-13));
    REQUIRE(report.target_x_change_kgms == Approx(
        report.committed_x_change_kgms + report.x_storage_residual_kgms));
    REQUIRE(report.target_y_change_kgms == Approx(
        report.committed_y_change_kgms + report.y_storage_residual_kgms));
}

TEST_CASE("scalar relaxation applies and ledgers one conserved interface source",
          "[core][nested_grid][boundary][relaxation][conservation]")
{
    const auto rel = relaxation_relationship();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Field3D parent_rho_previous(9, 9, 2, 2.0f);
    Field3D parent_rho_current(9, 9, 2, nan);
    Field3D parent_q_previous(9, 9, 2, 0.02f);
    Field3D parent_q_current(9, 9, 2, nan);
    for (int k = 0; k < 2; ++k)
    {
        parent_rho_previous[4][4][k] = nan;
        parent_q_previous[4][4][k] = nan;
    }
    Field3D old_rho(8, 8, 2, 1.0f);
    Field3D old_q(8, 8, 2, 0.01f);
    Field3D candidate_rho(8, 8, 2, 91.0f);
    Field3D candidate_q(8, 8, 2, 92.0f);
    tmv::domain::ConservedScalarRelaxationReport report;
    const tmv::domain::NestedRelaxationConfigV1 config{4.0};

    tmv::domain::relax_conserved_scalar_flat_storage(
        rel, boundary_bracket(rel, 0), config, 1.0, {200.0, 350.0},
        parent_rho_previous, parent_rho_current,
        parent_q_previous, parent_q_current, old_rho, old_q, true,
        candidate_rho, candidate_q, report);

    REQUIRE(candidate_rho[1][4][0] == 2.0f);
    REQUIRE(candidate_q[1][4][0] == 0.02f);
    REQUIRE(candidate_rho[3][3][0] == 1.0f);
    REQUIRE(candidate_q[3][3][0] == 0.01f);
    const double fraction = tmv::domain::nested_scalar_relaxation_fraction(
        rel, config, 1, 3, 1.0);
    const double expected_rho = 1.0 + fraction;
    const double expected_carrier = 0.01 + fraction * (0.04 - 0.01);
    REQUIRE(candidate_rho[2][4][0] == Approx(expected_rho));
    REQUIRE(candidate_q[2][4][0] ==
            Approx(expected_carrier / expected_rho));
    REQUIRE(candidate_rho[0][0][0] == 91.0f);
    REQUIRE(candidate_q[7][7][1] == 92.0f);
    REQUIRE(report.forced_cells == 64);
    REQUIRE(report.target_dry_mass_change_kg == Approx(
        report.committed_dry_mass_change_kg +
        report.dry_storage_residual_kg));
    REQUIRE(report.target_carrier_change == Approx(
        report.committed_carrier_change +
        report.carrier_storage_residual));

    candidate_rho.fill(71.0f);
    candidate_q.fill(72.0f);
    report.target_carrier_change = 19.0;
    old_q[6][6][1] = -0.5f;
    REQUIRE_THROWS_WITH(
        tmv::domain::relax_conserved_scalar_flat_storage(
            rel, boundary_bracket(rel, 0), config, 1.0, {200.0, 350.0},
            parent_rho_previous, parent_rho_current,
            parent_q_previous, parent_q_current, old_rho, old_q, true,
            candidate_rho, candidate_q, report),
        Catch::Matchers::Contains("old child scalar state"));
    REQUIRE(candidate_rho[1][1][0] == 71.0f);
    REQUIRE(candidate_q[6][6][1] == 72.0f);
    REQUIRE(report.target_carrier_change == 19.0);
}
