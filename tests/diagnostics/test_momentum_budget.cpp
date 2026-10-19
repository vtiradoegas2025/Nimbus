/**
 * @file test_momentum_budget.cpp
 * @brief Unit tests for the F.5b momentum budget diagnostic.
 *
 * Assertions reference analytic external truth on the Cartesian harness grid
 * (NR=NTH=NZ=8, dr=dz=100, dtheta=2pi/8 as the y-spacing under the retained
 * aliases). The cell volume is the uniform Cartesian dx*dy*dz, and the axial
 * angular momentum L_z is taken about the domain-center vertical axis with
 * (x_rel, y_rel) offsets — the same convention the diagnostic uses.
 *
 * Scenarios map to the F.5b acceptance fixtures:
 *   - hydrostatic at rest -> all momenta zero,
 *   - uniform flow        -> linear momentum = rho*U*volume,
 *   - solid-body / vortex rotation -> known axial angular momentum,
 *                            zero net linear momentum (symmetry),
 *   - bubble (buoyant)    -> buoyancy/weight forcing integrals.
 */

#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "core/runtime/simulation.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

void resize_momentum_fields(float fill = 0.0f)
{
    rho.resize(NR, NTH, NZ, fill);
    u.resize(NR, NTH, NZ, fill);
    v.resize(NR, NTH, NZ, fill);
    w.resize(NR, NTH, NZ, fill);
}

void clear_momentum_fields()
{
    rho = Field3D();
    u = Field3D();
    v = Field3D();
    w = Field3D();
}

// Uniform Cartesian cell volume dx*dy*dz (dr*dtheta*dz under the retained
// aliases) — identical for every cell.
double cell_volume() { return dr * dtheta * dz; }

double analytic_domain_volume()
{
    const auto domain = physical_scalar_domain(
        NR, NTH, NZ, global_grid_geometry.staggered
            ? StaggerType::CGrid : StaggerType::Collocated);
    return static_cast<double>(domain.nx()) * domain.ny() * domain.nz()
         * cell_volume();
}

// Cell-center offset from the domain-center vertical axis, matching the
// diagnostic's x_mid = 0.5*NR*dr, y_mid = 0.5*NTH*dtheta convention.
double x_rel_at(int i) { return (static_cast<double>(i) + 0.5) * dr - 0.5 * static_cast<double>(NR) * dr; }
double y_rel_at(int j) { return (static_cast<double>(j) + 0.5) * dtheta - 0.5 * static_cast<double>(NTH) * dtheta; }

struct StaggeredGeometryGuard
{
    explicit StaggeredGeometryGuard(GridGeometry& geometry_ref)
        : geometry(geometry_ref), saved(geometry_ref.staggered)
    {
    }

    ~StaggeredGeometryGuard() { geometry.staggered = saved; }

    GridGeometry& geometry;
    bool saved;
};

struct MomentumForcingStateGuard
{
    MomentumForcingStateGuard()
        : saved_rho0(rho0_base), saved_coriolis(coriolis_f)
    {
    }

    ~MomentumForcingStateGuard()
    {
        rho0_base = saved_rho0;
        coriolis_f = saved_coriolis;
        tmv::diagnostics::set_deterministic_reductions(false);
    }

    std::vector<double> saved_rho0;
    double saved_coriolis;
};

}  // namespace

TEST_CASE("momentum on empty fields returns zero", "[diagnostics][momentum_budget]")
{
    clear_momentum_fields();
    const auto b = compute_momentum_budget();
    REQUIRE(b.momentum_x == 0.0);
    REQUIRE(b.momentum_y == 0.0);
    REQUIRE(b.momentum_z == 0.0);
    REQUIRE(b.angular_momentum_z == 0.0);
    REQUIRE(b.total_magnitude == 0.0);
}

TEST_CASE("momentum budget fails closed on every invalid contributing field",
          "[diagnostics][momentum_budget][fail-closed][sci138]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    MomentumForcingStateGuard forcing_guard;
    global_grid_geometry.staggered = false;
    resize_momentum_fields(0.0f);
    rho.fill(1.0f);
    const std::array<std::pair<const char*, Field3D*>, 4> fields = {{
        {"rho", &rho}, {"u", &u}, {"v", &v}, {"w", &w},
    }};
    for (const bool deterministic : {false, true})
    {
        tmv::diagnostics::set_deterministic_reductions(deterministic);
        for (const auto& [name, field] : fields)
        {
            INFO("deterministic=" << deterministic << " field=" << name);
            const float saved = (*field)[1][1][1];
            (*field)[1][1][1] =
                std::numeric_limits<float>::quiet_NaN();
            REQUIRE_THROWS_WITH(
                compute_momentum_budget(),
                Catch::Matchers::Contains("rejected non-finite"));
            (*field)[1][1][1] = saved;
        }
    }
    tmv::diagnostics::set_deterministic_reductions(false);

    v.resize(NR - 1, NTH, NZ, 0.0f);
    REQUIRE_THROWS_WITH(
        compute_momentum_budget(),
        Catch::Matchers::Contains("match the grid"));
    v.resize(NR, NTH, NZ, 0.0f);

    rho0_base.assign(static_cast<std::size_t>(NZ), 1.0);
    for (const bool deterministic : {false, true})
    {
        tmv::diagnostics::set_deterministic_reductions(deterministic);
        rho0_base[1] = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_WITH(
            compute_momentum_forcing(),
            Catch::Matchers::Contains("rejected non-finite"));
        rho0_base[1] = 1.0;
    }
    tmv::diagnostics::set_deterministic_reductions(false);

    rho0_base.resize(static_cast<std::size_t>(NZ - 1));
    REQUIRE_THROWS_WITH(
        compute_momentum_forcing(),
        Catch::Matchers::Contains("must match grid.nz"));
    rho0_base.assign(static_cast<std::size_t>(NZ), 1.0);

    coriolis_f = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_WITH(
        compute_momentum_forcing(),
        Catch::Matchers::Contains("finite Coriolis"));
    MomentumBudget invalid{};
    invalid.momentum_x = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_WITH(
        evaluate_momentum_drift(MomentumBudget{}, invalid),
        Catch::Matchers::Contains("finite snapshots"));
}

TEST_CASE("hydrostatic rest has zero momentum in every component",
          "[diagnostics][momentum_budget]")
{
    resize_momentum_fields(0.0f);
    rho.fill(1.1f);  // at rest: u = v = w = 0
    const auto b = compute_momentum_budget();
    REQUIRE(b.momentum_x == 0.0);
    REQUIRE(b.momentum_y == 0.0);
    REQUIRE(b.momentum_z == 0.0);
    REQUIRE(b.angular_momentum_z == 0.0);
}

TEST_CASE("uniform Cartesian +x flow gives linear momentum = rho*U*volume",
          "[diagnostics][momentum_budget]")
{
    resize_momentum_fields(0.0f);
    const double U = 12.0;
    const float rho_val = 1.0f;
    rho.fill(rho_val);
    // A uniform (U, 0, 0) Cartesian wind: u (x-velocity) = U everywhere.
    u.fill(static_cast<float>(U));

    const auto b = compute_momentum_budget();
    const double mass = static_cast<double>(rho_val) * analytic_domain_volume();
    const double expected_px = U * mass;

    REQUIRE(b.momentum_x == Approx(expected_px).epsilon(1e-5));
    REQUIRE(b.momentum_y == 0.0);
    REQUIRE(b.momentum_z == 0.0);
    // A uniform x-wind carries no net axial angular momentum: L_z integrates
    // -U*y_rel, and sum(y_rel) over the centered grid is zero.
    REQUIRE(std::abs(b.angular_momentum_z) <= 1e-5 * expected_px * dr);
}

TEST_CASE("momentum budget uses explicit-state geometry for collocated and C-grid fields",
          "[diagnostics][momentum_budget][simulation_state]")
{
    clear_momentum_fields();

    int local_nr = 4;
    int local_nth = 5;
    int local_nz = 4;
    double local_dr = 5.0;
    double local_dz = 11.0;
    double local_dt = 0.25;
    double local_dtheta = 7.0;
    GridGeometry local_geometry;

    Field3D local_rho(local_nr, local_nth, local_nz, 2.0f);
    Field3D local_p(local_nr, local_nth, local_nz, static_cast<float>(p0));
    Field3D local_u(local_nr, local_nth, local_nz, 4.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 5.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 300.0f);
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

    std::vector<double> local_rho0(static_cast<std::size_t>(local_nz), 2.5);
    std::vector<double> local_p0;
    std::vector<double> local_qv0;
    std::vector<double> local_theta0;
    std::vector<double> local_u0;
    std::vector<double> local_v0;
    double local_coriolis_f = 2.0e-4;
    double local_time = 12.0;

    SimulationStateView state{
        {local_nr, local_nth, local_nz, local_dr, local_dz, local_dt, local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs, local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis_f,
        local_time};

    const auto local_domain = physical_scalar_domain(
        local_nr, local_nth, local_nz, StaggerType::Collocated);
    const double volume = static_cast<double>(
        local_domain.nx() * local_domain.ny() * local_domain.nz())
                        * local_dr * local_dtheta * local_dz;
    const double mass = 2.0 * volume;

    const auto b = compute_momentum_budget(state);
    REQUIRE(b.momentum_x == Approx(4.0 * mass).epsilon(1e-12));
    REQUIRE(b.momentum_y == Approx(5.0 * mass).epsilon(1e-12));
    REQUIRE(b.momentum_z == 0.0);

    const auto f = compute_momentum_forcing(state);
    REQUIRE(f.weight_z == Approx(-2.0 * g * volume).epsilon(1e-12));
    REQUIRE(f.buoyancy_z == Approx(-g * (2.0 - 2.5) * volume).epsilon(1e-12));
    REQUIRE(f.coriolis_x == Approx(local_coriolis_f * b.momentum_y).epsilon(1e-12));
    REQUIRE(f.coriolis_y == Approx(-local_coriolis_f * b.momentum_x).epsilon(1e-12));

    // Use gradients in every direction so the collocated and face-to-center
    // interpretations cannot accidentally agree.
    for (int i = 0; i < local_nr; ++i)
    {
        for (int j = 0; j < local_nth; ++j)
        {
            for (int k = 0; k < local_nz; ++k)
            {
                local_rho[i][j][k] = static_cast<float>(1.0 + 0.2 * i + 0.1 * j + 0.05 * k);
                local_u[i][j][k] = static_cast<float>(10.0 * i + 2.0 * j + 0.5 * k + 1.0);
                local_v[i][j][k] = static_cast<float>(20.0 * j + 3.0 * i + 0.25 * k + 2.0);
                local_w[i][j][k] = static_cast<float>(30.0 * k + 4.0 * i + 0.75 * j + 3.0);
            }
        }
    }

    double expected_x = 0.0;
    double expected_y = 0.0;
    double expected_z = 0.0;
    double expected_lz = 0.0;
    double expected_cgrid_mass = 0.0;
    double expected_cgrid_volume = 0.0;
    double expected_collocated_x = 0.0;
    double expected_collocated_y = 0.0;
    double expected_collocated_z = 0.0;
    double expected_collocated_lz = 0.0;
    const double local_cell_volume = local_dr * local_dtheta * local_dz;
    const double x_mid = 0.5 * static_cast<double>(local_nr) * local_dr;
    const double y_mid = 0.5 * static_cast<double>(local_nth) * local_dtheta;
    for (int i = 0; i < local_nr; ++i)
    {
        const double x_rel = (static_cast<double>(i) + 0.5) * local_dr - x_mid;
        for (int j = 0; j < local_nth; ++j)
        {
            const double y_rel = (static_cast<double>(j) + 0.5) * local_dtheta - y_mid;
            for (int k = 0; k < local_nz; ++k)
            {
                const double rho_cell = static_cast<double>(local_rho[i][j][k]);
                const double u_center = i == 0
                    ? static_cast<double>(local_u[i][j][k])
                    : 0.5 * (static_cast<double>(local_u[i - 1][j][k]) +
                             static_cast<double>(local_u[i][j][k]));
                const double v_center = j == 0
                    ? static_cast<double>(local_v[i][j][k])
                    : 0.5 * (static_cast<double>(local_v[i][j - 1][k]) +
                             static_cast<double>(local_v[i][j][k]));
                const double w_center = k == 0
                    ? 0.5 * static_cast<double>(local_w[i][j][k])
                    : 0.5 * (static_cast<double>(local_w[i][j][k - 1]) +
                             static_cast<double>(local_w[i][j][k]));
                const double mass_cell = rho_cell * local_cell_volume;
                if (i >= 1 && i < local_nr - 1 &&
                    j >= 1 && j < local_nth - 1)
                {
                    expected_x += mass_cell * u_center;
                    expected_y += mass_cell * v_center;
                    expected_z += mass_cell * w_center;
                    expected_lz += mass_cell * (x_rel * v_center - y_rel * u_center);
                    expected_cgrid_mass += mass_cell;
                    expected_cgrid_volume += local_cell_volume;
                }
                if (i >= 1 && i < local_nr - 1
                    && j >= 1 && j < local_nth - 1
                    && k >= 1 && k < local_nz - 1)
                {
                    const double u_collocated = static_cast<double>(local_u[i][j][k]);
                    const double v_collocated = static_cast<double>(local_v[i][j][k]);
                    const double w_collocated = static_cast<double>(local_w[i][j][k]);
                    expected_collocated_x += mass_cell * u_collocated;
                    expected_collocated_y += mass_cell * v_collocated;
                    expected_collocated_z += mass_cell * w_collocated;
                    expected_collocated_lz +=
                        mass_cell * (x_rel * v_collocated - y_rel * u_collocated);
                }
            }
        }
    }

    const auto collocated_gradient_budget = compute_momentum_budget(state);
    REQUIRE(collocated_gradient_budget.momentum_x ==
            Approx(expected_collocated_x).epsilon(1e-12));
    REQUIRE(collocated_gradient_budget.momentum_y ==
            Approx(expected_collocated_y).epsilon(1e-12));
    REQUIRE(collocated_gradient_budget.momentum_z ==
            Approx(expected_collocated_z).epsilon(1e-12));
    REQUIRE(collocated_gradient_budget.angular_momentum_z ==
            Approx(expected_collocated_lz).epsilon(1e-12));
    const auto collocated_gradient_forcing = compute_momentum_forcing(state);

    // The exact same storage now represents C-grid face values. The old raw
    // index implementation would return the collocated expectations above.
    local_geometry.staggered = true;
    const auto cgrid_budget = compute_momentum_budget(state);
    REQUIRE(cgrid_budget.momentum_x == Approx(expected_x).epsilon(1e-12));
    REQUIRE(cgrid_budget.momentum_y == Approx(expected_y).epsilon(1e-12));
    REQUIRE(cgrid_budget.momentum_z == Approx(expected_z).epsilon(1e-12));
    REQUIRE(cgrid_budget.angular_momentum_z == Approx(expected_lz).epsilon(1e-12));

    // Snapshot-derived Coriolis attribution must consume the same reconstructed
    // horizontal momentum, not silently fall back to raw face indices.
    const auto cgrid_forcing = compute_momentum_forcing(state);
    REQUIRE(cgrid_forcing.coriolis_x ==
            Approx(local_coriolis_f * expected_y).epsilon(1e-12));
    REQUIRE(cgrid_forcing.coriolis_y ==
            Approx(-local_coriolis_f * expected_x).epsilon(1e-12));
    REQUIRE(cgrid_forcing.weight_z ==
            Approx(-g * expected_cgrid_mass).epsilon(1e-12));
    REQUIRE(cgrid_forcing.buoyancy_z ==
            Approx(-g * (expected_cgrid_mass - 2.5 * expected_cgrid_volume))
                .epsilon(1e-12));

    // Scalar values in the one-cell C-grid lateral halo are boundary closure
    // storage, not atmospheric volume. Poison every lateral ghost density and
    // prove neither inventory nor the volume-integrated forces change.
    const MomentumBudget cgrid_before_ghost_poison = cgrid_budget;
    const MomentumForcing forcing_before_ghost_poison = cgrid_forcing;
    for (int i = 0; i < local_nr; ++i)
    {
        for (int j = 0; j < local_nth; ++j)
        {
            if (i != 0 && i != local_nr - 1 &&
                j != 0 && j != local_nth - 1)
            {
                continue;
            }
            for (int k = 0; k < local_nz; ++k)
            {
                local_rho[i][j][k] = 1.0e6f;
            }
        }
    }

    const auto cgrid_after_ghost_poison = compute_momentum_budget(state);
    const auto forcing_after_ghost_poison = compute_momentum_forcing(state);
    REQUIRE(cgrid_after_ghost_poison.momentum_x ==
            cgrid_before_ghost_poison.momentum_x);
    REQUIRE(cgrid_after_ghost_poison.momentum_y ==
            cgrid_before_ghost_poison.momentum_y);
    REQUIRE(cgrid_after_ghost_poison.momentum_z ==
            cgrid_before_ghost_poison.momentum_z);
    REQUIRE(cgrid_after_ghost_poison.angular_momentum_z ==
            cgrid_before_ghost_poison.angular_momentum_z);
    REQUIRE(cgrid_after_ghost_poison.total_mass ==
            cgrid_before_ghost_poison.total_mass);
    REQUIRE(forcing_after_ghost_poison.weight_z ==
            forcing_before_ghost_poison.weight_z);
    REQUIRE(forcing_after_ghost_poison.buoyancy_z ==
            forcing_before_ghost_poison.buoyancy_z);
    REQUIRE(forcing_after_ghost_poison.coriolis_x ==
            forcing_before_ghost_poison.coriolis_x);
    REQUIRE(forcing_after_ghost_poison.coriolis_y ==
            forcing_before_ghost_poison.coriolis_y);

    // The identical lateral edge cells are closure storage for collocated too.
    local_geometry.staggered = false;
    const auto collocated_after_edge_poison = compute_momentum_budget(state);
    const auto collocated_forcing_after_edge_poison = compute_momentum_forcing(state);
    REQUIRE(collocated_after_edge_poison.total_mass ==
            collocated_gradient_budget.total_mass);
    REQUIRE(collocated_forcing_after_edge_poison.weight_z ==
            collocated_gradient_forcing.weight_z);

    for (int i = 0; i < local_nr; ++i)
        for (int j = 0; j < local_nth; ++j)
            for (const int k : {0, local_nz - 1})
                local_rho[i][j][k] = 2.0e6f;
    const auto collocated_after_vertical_poison = compute_momentum_budget(state);
    const auto collocated_forcing_after_vertical_poison =
        compute_momentum_forcing(state);
    REQUIRE(collocated_after_vertical_poison.momentum_x ==
            collocated_gradient_budget.momentum_x);
    REQUIRE(collocated_after_vertical_poison.momentum_y ==
            collocated_gradient_budget.momentum_y);
    REQUIRE(collocated_after_vertical_poison.momentum_z ==
            collocated_gradient_budget.momentum_z);
    REQUIRE(collocated_after_vertical_poison.total_mass ==
            collocated_gradient_budget.total_mass);
    REQUIRE(collocated_forcing_after_vertical_poison.weight_z ==
            collocated_gradient_forcing.weight_z);
}

TEST_CASE("uniform updraft gives vertical momentum = rho*w*volume",
          "[diagnostics][momentum_budget]")
{
    resize_momentum_fields(0.0f);
    const float rho_val = 1.0f;
    const float W = 3.0f;
    rho.fill(rho_val);
    w.fill(W);

    const auto b = compute_momentum_budget();
    const double expected = static_cast<double>(rho_val) * static_cast<double>(W) *
                            analytic_domain_volume();
    REQUIRE(b.momentum_z == Approx(expected).epsilon(1e-6));
    REQUIRE(b.momentum_x == 0.0);
    REQUIRE(b.momentum_y == 0.0);
    // total_mass and the mass-scaled reference floor are populated.
    REQUIRE(b.total_mass == Approx(static_cast<double>(rho_val) * analytic_domain_volume()).epsilon(1e-6));
    REQUIRE(b.reference_linear == Approx(b.total_mass * MOMENTUM_REF_VELOCITY_MS).epsilon(1e-12));
}

TEST_CASE("momentum subset excludes one retained rectangle on the production basis",
          "[diagnostics][momentum_budget][subset][domain_recycling]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    resize_momentum_fields(0.0f);
    rho.fill(2.0f);
    u.fill(4.0f);
    v.fill(-3.0f);
    w.fill(0.0f);

    const HorizontalMomentumBudgetExclusion retained{2, 6, 2, 5};
    const auto ordinary =
        compute_momentum_budget_excluding_horizontal_rectangle(
            global_simulation_state(), retained);
    tmv::diagnostics::set_deterministic_reductions(true);
    const auto deterministic =
        compute_momentum_budget_excluding_horizontal_rectangle(
            global_simulation_state(), retained);
    tmv::diagnostics::set_deterministic_reductions(false);

    const auto domain = physical_scalar_domain(
        NR, NTH, NZ, StaggerType::CGrid);
    const int physical_columns = domain.nx() * domain.ny();
    const int retained_columns =
        (retained.i_end - retained.i_begin) *
        (retained.j_end - retained.j_begin);
    const double expected_mass =
        2.0 * static_cast<double>(physical_columns - retained_columns) *
        domain.nz() * cell_volume();
    REQUIRE(ordinary.total_mass == Approx(expected_mass));
    REQUIRE(ordinary.momentum_x == Approx(expected_mass * 4.0));
    REQUIRE(ordinary.momentum_y == Approx(expected_mass * -3.0));
    REQUIRE(ordinary.momentum_z == 0.0);
    REQUIRE(deterministic.total_mass == Approx(ordinary.total_mass));
    REQUIRE(deterministic.momentum_x == Approx(ordinary.momentum_x));
    REQUIRE(deterministic.momentum_y == Approx(ordinary.momentum_y));
    REQUIRE(deterministic.angular_momentum_z ==
            Approx(ordinary.angular_momentum_z));
    REQUIRE_THROWS_WITH(
        compute_momentum_budget_excluding_horizontal_rectangle(
            global_simulation_state(), {3, 3, 1, 2}),
        Catch::Matchers::Contains("exclusion is invalid"));
}

TEST_CASE("solid-body rotation gives axial angular momentum = Omega*integral(rho (x^2+y^2) dV)",
          "[diagnostics][momentum_budget]")
{
    resize_momentum_fields(0.0f);
    const double Omega = 0.01;
    const float rho_val = 1.0f;
    rho.fill(rho_val);
    // Cartesian solid-body rotation about the domain-center axis:
    // u = -Omega*y_rel, v = +Omega*x_rel.
    for (int i = 1; i < NR - 1; ++i)
    {
        const double x_rel = x_rel_at(i);
        for (int j = 1; j < NTH - 1; ++j)
        {
            const double y_rel = y_rel_at(j);
            const float ux = static_cast<float>(-Omega * y_rel);
            const float vy = static_cast<float>(Omega * x_rel);
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = ux;
                v[i][j][k] = vy;
            }
        }
    }

    // Independent analytic integral: L_z = Omega * rho * sum (x_rel^2 + y_rel^2) * dV.
    double expected = 0.0;
    for (int i = 1; i < NR - 1; ++i)
    {
        const double x_rel = x_rel_at(i);
        for (int j = 1; j < NTH - 1; ++j)
        {
            const double y_rel = y_rel_at(j);
            expected += (x_rel * x_rel + y_rel * y_rel) * cell_volume()
                      * static_cast<double>(NZ - 2);
        }
    }
    expected *= Omega * static_cast<double>(rho_val);

    const auto b = compute_momentum_budget();
    REQUIRE(b.angular_momentum_z == Approx(expected).epsilon(1e-5));
    // Pure rotation about the domain center carries no net linear momentum
    // (sum(x_rel) and sum(y_rel) over the centered grid both vanish).
    REQUIRE(std::abs(b.momentum_x) <= 1e-6 * expected / dr);
    REQUIRE(std::abs(b.momentum_y) <= 1e-6 * expected / dr);
}

TEST_CASE("axisymmetric vortex: positive angular momentum, zero net linear momentum",
          "[diagnostics][momentum_budget]")
{
    resize_momentum_fields(0.0f);
    rho.fill(1.0f);
    // A cyclonic Lamb-Oseen tangential profile v_theta(r) about the domain
    // center, projected into Cartesian (u, v) components:
    //   u = -v_theta * y_rel / r,  v = +v_theta * x_rel / r.
    const double Gamma = 5.0e4;      // circulation, m^2/s
    const double rc_core = 3.0 * dr; // core radius
    for (int i = 0; i < NR; ++i)
    {
        const double x_rel = x_rel_at(i);
        for (int j = 0; j < NTH; ++j)
        {
            const double y_rel = y_rel_at(j);
            const double r = std::sqrt(x_rel * x_rel + y_rel * y_rel);
            const double v_theta = (Gamma / (2.0 * M_PI * r)) *
                                   (1.0 - std::exp(-(r * r) / (rc_core * rc_core)));
            const float ux = static_cast<float>(-v_theta * y_rel / r);
            const float vy = static_cast<float>(v_theta * x_rel / r);
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = ux;
                v[i][j][k] = vy;
            }
        }
    }

    const auto b = compute_momentum_budget();
    // A cyclonic axisymmetric vortex has positive axial angular momentum and,
    // by symmetry about the domain center, no net linear momentum.
    REQUIRE(b.angular_momentum_z > 0.0);
    REQUIRE(std::abs(b.momentum_x) <= 1e-6 * b.angular_momentum_z / dr);
    REQUIRE(std::abs(b.momentum_y) <= 1e-6 * b.angular_momentum_z / dr);
}

TEST_CASE("momentum forcing: weight, buoyancy, and Coriolis integrals are exact",
          "[diagnostics][momentum_budget]")
{
    resize_momentum_fields(0.0f);
    const float rho_val = 1.2f;
    rho.fill(rho_val);

    SECTION("weight equals -rho*g*volume; buoyancy zero against matching base state")
    {
        rho0_base.assign(static_cast<std::size_t>(NZ), static_cast<double>(rho_val));
        const auto f = compute_momentum_forcing();
        const double V = analytic_domain_volume();
        REQUIRE(f.weight_z == Approx(-static_cast<double>(rho_val) * g * V).epsilon(1e-6));
        REQUIRE(std::abs(f.buoyancy_z) <= 1e-6 * std::abs(f.weight_z));
    }

    SECTION("buoyancy equals -g*(rho - rho0)*volume for a warm (lighter) bubble")
    {
        // Bubble-flavored: parcel density below the base state -> upward buoyancy.
        const double rho0 = 1.3;
        rho0_base.assign(static_cast<std::size_t>(NZ), rho0);
        const auto f = compute_momentum_forcing();
        const double V = analytic_domain_volume();
        const double expected = -g * (static_cast<double>(rho_val) - rho0) * V;
        REQUIRE(f.buoyancy_z == Approx(expected).epsilon(1e-6));
        REQUIRE(f.buoyancy_z > 0.0);  // lighter than base -> positive (upward)
    }

    SECTION("Coriolis force is (f*p_y, -f*p_x) from the linear momentum")
    {
        coriolis_f = 8.4e-5;
        // A uniform horizontal wind with both components nonzero so the
        // Coriolis identity is exercised on both momenta.
        u.fill(10.0f);  // x-momentum
        v.fill(4.0f);   // y-momentum
        const auto mom = compute_momentum_budget();
        const auto f = compute_momentum_forcing();
        REQUIRE(f.coriolis_x == Approx(coriolis_f * mom.momentum_y).epsilon(1e-9));
        REQUIRE(f.coriolis_y == Approx(-coriolis_f * mom.momentum_x).epsilon(1e-9));
        coriolis_f = 0.0;  // restore harness default
    }
}

TEST_CASE("momentum drift warns above threshold, stays quiet below",
          "[diagnostics][momentum_budget]")
{
    MomentumBudget before{};
    before.momentum_x = 1.0e9;
    before.momentum_y = 2.0e9;
    before.momentum_z = 5.0e8;
    before.angular_momentum_z = 3.0e12;

    SECTION("identical snapshots do not warn")
    {
        const auto d = evaluate_momentum_drift(before, before);
        REQUIRE(d.warn == false);
        REQUIRE(d.d_momentum_x == 0.0);
        REQUIRE(d.rel_angular_momentum_z == 0.0);
    }
    SECTION("drift just above the threshold warns")
    {
        MomentumBudget after = before;
        after.momentum_x *= 1.0 + BUDGET_THRESH_MOMENTUM * 10.0;
        const auto d = evaluate_momentum_drift(before, after);
        REQUIRE(d.warn == true);
        REQUIRE(d.rel_momentum_x > BUDGET_THRESH_MOMENTUM);
    }
    SECTION("drift just below the threshold does not warn")
    {
        MomentumBudget after = before;
        after.angular_momentum_z *= 1.0 + BUDGET_THRESH_MOMENTUM * 0.5;
        const auto d = evaluate_momentum_drift(before, after);
        REQUIRE(d.warn == false);
        REQUIRE(d.rel_angular_momentum_z <= BUDGET_THRESH_MOMENTUM);
    }
}

TEST_CASE("near-zero momentum on a quiescent domain is well-conditioned",
          "[diagnostics][momentum_budget]")
{
    // A calm hydrostatic domain carries a near-zero net momentum (sub-1 m/s
    // mean wind over a large mass). Without the mass-scaled reference floor,
    // a tiny absolute drift would read as a large *relative* drift and warn
    // spuriously. Numbers mirror an observed cgrid_hydrostatic step.
    MomentumBudget before{};
    before.momentum_x = 1.0e4;
    before.momentum_y = -2.0e4;
    before.momentum_z = 6.5e10;       // huge volume * ~1e-7 m/s mean w
    before.angular_momentum_z = 1.0e9;
    before.total_mass = 1.0e14;       // domain mass
    before.reference_linear = before.total_mass * MOMENTUM_REF_VELOCITY_MS;          // 1e14
    before.reference_angular = before.total_mass * MOMENTUM_REF_VELOCITY_MS * 6.4e4; // mass*Vref*R

    MomentumBudget after = before;
    after.momentum_z += 7.0e9;        // a full-percent swing of the raw inventory

    const auto d = evaluate_momentum_drift(before, after);
    // Raw relative change of momentum_z would be ~10%, but against the
    // characteristic momentum it is ~7e-5 -- correctly below the gate.
    REQUIRE(d.rel_momentum_z < BUDGET_THRESH_MOMENTUM);
    REQUIRE(d.warn == false);
}

TEST_CASE("deterministic momentum reductions are thread-count independent",
          "[diagnostics][momentum_budget][deterministic]")
{
    resize_momentum_fields(0.0f);
    rho.fill(1.05f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u[i][j][k] = static_cast<float>(0.1 * i);
                v[i][j][k] = static_cast<float>(0.2 * j);
                w[i][j][k] = static_cast<float>(0.3 * k);
            }
    rho0_base.assign(static_cast<std::size_t>(NZ), 1.0);
    tmv::diagnostics::set_deterministic_reductions(true);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto a = compute_momentum_budget();
    const auto forcing_a = compute_momentum_forcing();
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto b = compute_momentum_budget();
    const auto forcing_b = compute_momentum_forcing();
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    tmv::diagnostics::set_deterministic_reductions(false);
    REQUIRE(a.momentum_x == b.momentum_x);
    REQUIRE(a.momentum_y == b.momentum_y);
    REQUIRE(a.momentum_z == b.momentum_z);
    REQUIRE(a.angular_momentum_z == b.angular_momentum_z);
    REQUIRE(a.total_mass == b.total_mass);
    REQUIRE(forcing_a.weight_z == forcing_b.weight_z);
    REQUIRE(forcing_a.buoyancy_z == forcing_b.buoyancy_z);
}
