/**
 * @file test_moisture_budget.cpp
 * @brief Unit tests for the F.5b moisture budget diagnostic.
 *
 * Phase integrals are checked against analytic domain integrals; the
 * conservation property (microphysics redistributes phases but conserves total
 * water) is checked by moving mass between phases at fixed density and
 * asserting the total is invariant.
 */

#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "core/runtime/simulation.hpp"

#include <array>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

void resize_moisture_fields(float fill = 0.0f)
{
    rho.resize(NR, NTH, NZ, fill);
    qv.resize(NR, NTH, NZ, fill);
    qc.resize(NR, NTH, NZ, fill);
    qr.resize(NR, NTH, NZ, fill);
    qi.resize(NR, NTH, NZ, fill);
    qs.resize(NR, NTH, NZ, fill);
    qg.resize(NR, NTH, NZ, fill);
    qh.resize(NR, NTH, NZ, fill);
}

void clear_moisture_fields()
{
    rho = Field3D();
    qv = Field3D();
    qc = Field3D();
    qr = Field3D();
    qi = Field3D();
    qs = Field3D();
    qg = Field3D();
    qh = Field3D();
}

// Uniform Cartesian cell volume dx*dy*dz (dr*dtheta*dz under the retained
// aliases), summed over every cell.
double analytic_domain_volume()
{
    const auto domain = physical_scalar_domain(
        NR, NTH, NZ, global_grid_geometry.staggered
            ? StaggerType::CGrid : StaggerType::Collocated);
    return static_cast<double>(domain.nx()) * domain.ny() * domain.nz()
         * dr * dtheta * dz;
}

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

struct ReductionModeGuard
{
    bool saved = tmv::diagnostics::deterministic_reductions_enabled();
    explicit ReductionModeGuard(bool deterministic)
    {
        tmv::diagnostics::set_deterministic_reductions(deterministic);
    }
    ~ReductionModeGuard()
    {
        tmv::diagnostics::set_deterministic_reductions(saved);
    }
};

}  // namespace

TEST_CASE("moisture on empty fields returns zero", "[diagnostics][moisture_budget]")
{
    clear_moisture_fields();
    const auto m = compute_moisture_budget();
    REQUIRE(m.vapor == 0.0);
    REQUIRE(m.liquid == 0.0);
    REQUIRE(m.ice == 0.0);
    REQUIRE(m.total_water == 0.0);
}

TEST_CASE("moisture budget fails closed on every invalid contributing field",
          "[diagnostics][moisture_budget][fail-closed][sci138]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = false;
    resize_moisture_fields(0.0f);
    rho.fill(1.0f);
    const std::array<std::pair<const char*, Field3D*>, 8> fields = {{
        {"rho", &rho}, {"qv", &qv}, {"qc", &qc}, {"qr", &qr},
        {"qi", &qi}, {"qs", &qs}, {"qg", &qg}, {"qh", &qh},
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
                compute_moisture_budget(),
                Catch::Matchers::Contains("rejected non-finite"));
            (*field)[1][1][1] = saved;
        }
    }
    tmv::diagnostics::set_deterministic_reductions(false);

    qs.resize(NR, NTH - 1, NZ, 0.0f);
    REQUIRE_THROWS_WITH(
        compute_moisture_budget(),
        Catch::Matchers::Contains("match the grid"));
    qs.resize(NR, NTH, NZ, 0.0f);

    MoistureBudget invalid{};
    invalid.total_water = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_WITH(
        evaluate_moisture_drift(MoistureBudget{}, invalid),
        Catch::Matchers::Contains("finite snapshots"));
}

TEST_CASE("phase integrals = rho*q*volume for constant species",
          "[diagnostics][moisture_budget]")
{
    resize_moisture_fields(0.0f);
    const float rho_val = 1.0f;
    rho.fill(rho_val);
    // Powers of two: exact in float32, so the float fields match the double
    // "expected" to well within tolerance.
    qv.fill(0.0078125f);     // 1/128  vapor
    qc.fill(0.00390625f);    // 1/256  cloud  (liquid)
    qr.fill(0.001953125f);   // 1/512  rain   (liquid)
    qi.fill(0.0009765625f);  // 1/1024 ice
    qs.fill(0.00048828125f); // 1/2048 snow   (ice)

    const auto m = compute_moisture_budget();
    const double V = analytic_domain_volume();
    const double rho_d = static_cast<double>(rho_val);

    REQUIRE(m.vapor  == Approx(rho_d * (1.0 / 128.0) * V).epsilon(1e-6));
    REQUIRE(m.liquid == Approx(rho_d * (1.0 / 256.0 + 1.0 / 512.0) * V).epsilon(1e-6));
    REQUIRE(m.ice    == Approx(rho_d * (1.0 / 1024.0 + 1.0 / 2048.0) * V).epsilon(1e-6));
    REQUIRE(m.total_water ==
            Approx(m.vapor + m.liquid + m.ice).epsilon(1e-12));
}

TEST_CASE("moisture budget preserves signed invalid species for guard visibility",
          "[diagnostics][moisture_budget][guard]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = false;
    resize_moisture_fields(0.0f);
    rho.fill(1.0f);
    qr[1][1][1] = -0.02f;

    const auto budget = compute_moisture_budget();
    REQUIRE(budget.liquid == Approx(-0.02 * dr * dtheta * dz).epsilon(1.0e-6));
    REQUIRE(budget.total_water == Approx(budget.liquid).epsilon(1.0e-12));
}

TEST_CASE("moisture budget can compute from an explicit simulation state view",
          "[diagnostics][moisture_budget][simulation_state]")
{
    clear_moisture_fields();

    int local_nr = 4;
    int local_nth = 5;
    int local_nz = 4;
    double local_dr = 5.0;
    double local_dz = 11.0;
    double local_dt = 0.25;
    double local_dtheta = 7.0;
    GridGeometry local_geometry;

    Field3D local_rho(local_nr, local_nth, local_nz, 2.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 101325.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 300.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.125f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0625f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0625f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.03125f);
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

    std::vector<double> local_rho0;
    std::vector<double> local_p0;
    std::vector<double> local_qv0;
    std::vector<double> local_theta0;
    std::vector<double> local_u0;
    std::vector<double> local_v0;
    double local_coriolis_f = 0.0;
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

    const auto m = compute_moisture_budget(state);
    const auto local_domain = physical_scalar_domain(
        local_nr, local_nth, local_nz, StaggerType::Collocated);
    const double volume = static_cast<double>(
        local_domain.nx() * local_domain.ny() * local_domain.nz())
                        * local_dr * local_dtheta * local_dz;

    REQUIRE(m.vapor == Approx(2.0 * 0.125 * volume).epsilon(1e-12));
    REQUIRE(m.liquid == Approx(2.0 * (0.0625 + 0.0625) * volume).epsilon(1e-12));
    REQUIRE(m.ice == Approx(2.0 * 0.03125 * volume).epsilon(1e-12));
    REQUIRE(m.total_water == Approx(m.vapor + m.liquid + m.ice).epsilon(1e-12));
}

TEST_CASE("moisture ownership excludes C-grid and collocated closure storage",
          "[diagnostics][moisture_budget][cgrid][ghost]")
{
    const bool         deterministic = GENERATE(false, true);
    ReductionModeGuard reduction_guard(deterministic);
    INFO("deterministic=" << deterministic);
    // Binary-exact cell inventories isolate closure ownership from OpenMP merge order.
    double       local_dx = 8.0;
    double       local_dy = 16.0;
    double       local_dz = 4.0;
    GridGeometry geometry;
    geometry.initialize(NR, NTH, NZ, local_dx, local_dz, local_dy, CoordinateSystem::Cartesian);
    const auto          global = global_simulation_state();
    SimulationStateView state{{NR, NTH, NZ, local_dx, local_dz, dt, local_dy, geometry},
                              global.fields,
                              global.base,
                              global.tendencies,
                              global.coriolis_f,
                              global.time_s};
    resize_moisture_fields(0.0f);
    rho.fill(2.0f);
    qv.fill(0.125f);
    qc.fill(0.0625f);
    qr.fill(0.03125f);
    qi.fill(0.015625f);
    qs.fill(0.0078125f);
    qg.fill(0.00390625f);
    qh.fill(0.001953125f);
    geometry.staggered = true;

    const double physical_volume = static_cast<double>((NR - 2) * (NTH - 2) * NZ) * local_dx * local_dy * local_dz;
    const auto   clean           = compute_moisture_budget(state);
    REQUIRE(clean.vapor == Approx(2.0 * 0.125 * physical_volume).epsilon(1e-12));
    REQUIRE(clean.liquid ==
            Approx(2.0 * (0.0625 + 0.03125) * physical_volume).epsilon(1e-12));
    REQUIRE(clean.ice ==
            Approx(2.0 * (0.015625 + 0.0078125 + 0.00390625 + 0.001953125)
                   * physical_volume).epsilon(1e-12));

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            if (i != 0 && i != NR - 1 && j != 0 && j != NTH - 1)
                continue;
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] = 50.0f;
                qv[i][j][k] = 0.5f;
                qc[i][j][k] = 0.25f;
                qr[i][j][k] = 0.125f;
                qi[i][j][k] = 0.0625f;
                qs[i][j][k] = 0.03125f;
                qg[i][j][k] = 0.015625f;
                qh[i][j][k] = 0.0078125f;
            }
        }

    const auto poisoned_cgrid = compute_moisture_budget(state);
    REQUIRE(poisoned_cgrid.vapor == clean.vapor);
    REQUIRE(poisoned_cgrid.liquid == clean.liquid);
    REQUIRE(poisoned_cgrid.ice == clean.ice);
    REQUIRE(poisoned_cgrid.total_water == clean.total_water);

    geometry.staggered      = false;
    const auto   collocated = compute_moisture_budget(state);
    const double collocated_volume =
        static_cast<double>((NR - 2) * (NTH - 2) * (NZ - 2)) * local_dx * local_dy * local_dz;
    REQUIRE(collocated.vapor == Approx(2.0 * 0.125 * collocated_volume));
    REQUIRE(collocated.liquid == Approx(
        2.0 * (0.0625 + 0.03125) * collocated_volume));
    REQUIRE(collocated.ice == Approx(
        2.0 * (0.015625 + 0.0078125 + 0.00390625 + 0.001953125)
        * collocated_volume));

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (const int k : {0, NZ - 1})
            {
                rho[i][j][k] = 1.0e6f;
                qv[i][j][k] = 100.0f;
                qc[i][j][k] = 100.0f;
                qi[i][j][k] = 100.0f;
            }
    const auto collocated_vertical_poison = compute_moisture_budget(state);
    REQUIRE(collocated_vertical_poison.vapor == collocated.vapor);
    REQUIRE(collocated_vertical_poison.liquid == collocated.liquid);
    REQUIRE(collocated_vertical_poison.ice == collocated.ice);
    REQUIRE(collocated_vertical_poison.total_water == collocated.total_water);
}

TEST_CASE("microphysics phase change conserves total water",
          "[diagnostics][moisture_budget]")
{
    // Condensation moves mass vapor -> cloud at fixed density. Vapor falls,
    // liquid rises, total water is invariant -- the F.6.C conservation gate.
    // Values are exact in float32 (1/64 = 1/128 + 1/128) so the only residual
    // is double-summation order, not field storage rounding.
    resize_moisture_fields(0.0f);
    rho.fill(1.0f);
    qv.fill(0.015625f);  // 1/64
    const auto before = compute_moisture_budget();

    // Move 1/128 kg/kg from vapor to cloud everywhere.
    qv.fill(0.0078125f);  // 1/128
    qc.fill(0.0078125f);  // 1/128
    const auto after = compute_moisture_budget();

    const auto drift = evaluate_moisture_drift(before, after);
    // Conserved to the float/summation noise floor, far below the 1e-4 gate.
    REQUIRE(after.total_water == Approx(before.total_water).epsilon(1e-6));
    REQUIRE(drift.warn == false);                 // total water closed
    REQUIRE(drift.d_vapor < 0.0);                 // vapor sink
    REQUIRE(drift.d_liquid > 0.0);                // liquid source
    REQUIRE(drift.d_vapor == Approx(-drift.d_liquid).epsilon(1e-6));
    REQUIRE(drift.rel_total_water <= BUDGET_THRESH_MOISTURE);
}

TEST_CASE("moisture drift warns when total water is not conserved",
          "[diagnostics][moisture_budget]")
{
    MoistureBudget before{};
    before.vapor = 1.0e11;
    before.liquid = 2.0e10;
    before.ice = 5.0e9;
    before.total_water = before.vapor + before.liquid + before.ice;

    // A spurious total-water gain above the gate threshold.
    MoistureBudget after = before;
    after.vapor += before.total_water * (BUDGET_THRESH_MOISTURE * 10.0);
    after.total_water = after.vapor + after.liquid + after.ice;

    const auto drift = evaluate_moisture_drift(before, after);
    REQUIRE(drift.warn == true);
    REQUIRE(drift.rel_total_water > BUDGET_THRESH_MOISTURE);
}

TEST_CASE("deterministic moisture reductions are thread-count independent",
          "[diagnostics][moisture_budget][deterministic]")
{
    resize_moisture_fields(0.0f);
    rho.fill(1.05f);
    qv.fill(0.012f);
    qc.fill(0.002f);
    qr.fill(0.001f);
    tmv::diagnostics::set_deterministic_reductions(true);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto a = compute_moisture_budget();
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto b = compute_moisture_budget();
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    tmv::diagnostics::set_deterministic_reductions(false);
    REQUIRE(a.vapor == b.vapor);
    REQUIRE(a.liquid == b.liquid);
    REQUIRE(a.ice == b.ice);
    REQUIRE(a.total_water == b.total_water);
}
