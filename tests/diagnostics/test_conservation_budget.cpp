/**
 * @file test_conservation_budget.cpp
 * @brief Unit tests for the conservation budget diagnostic.
 *
 * Two layers under test:
 *   1. compute_conservation_budget(): integration math over the default
 *      global-state adapter and an explicit SimulationStateView. We seed
 *      fields with constant or structured values and assert the integrals
 *      match analytic density-weighted inventories.
 *   2. evaluate_budget_drift(): the warn-trigger logic that decides
 *      whether a stage transition trips the transport-conserved or
 *      physical-mass thresholds. Pure function; no globals beyond the
 *      two snapshots passed in, so we can construct snapshots directly.
 *
 * The test_harness sets NR=NTH=NZ=8, dr=dz=100, dtheta=2pi/8. The model is
 * Cartesian, so the cell volume is the uniform Cartesian Jacobian dx*dy*dz
 * (dr*dtheta*dz under the retained r/theta aliases) for every cell. All
 * Both layouts exclude lateral closure planes. The collocated A-grid also
 * excludes bottom/top closure planes; all C-grid vertical scalar levels are
 * physical.
 */

#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/moisture_budget.hpp"
#include "core/runtime/simulation.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

// Resize all fields the diagnostic reads to the harness grid.
void resize_all_budget_fields(float fill_value = 0.0f)
{
    rho.resize(NR, NTH, NZ, fill_value);
    p.resize(NR, NTH, NZ, fill_value);
    theta.resize(NR, NTH, NZ, fill_value);
    qv.resize(NR, NTH, NZ, fill_value);
    qc.resize(NR, NTH, NZ, fill_value);
    qr.resize(NR, NTH, NZ, fill_value);
    qi.resize(NR, NTH, NZ, fill_value);
    qs.resize(NR, NTH, NZ, fill_value);
    qg.resize(NR, NTH, NZ, fill_value);
    qh.resize(NR, NTH, NZ, fill_value);
}

void clear_all_budget_fields()
{
    rho = Field3D();
    p = Field3D();
    theta = Field3D();
    qv = Field3D();
    qc = Field3D();
    qr = Field3D();
    qi = Field3D();
    qs = Field3D();
    qg = Field3D();
    qh = Field3D();
}

// Analytic domain volume on the harness grid: the uniform Cartesian cell
// volume dx*dy*dz (dr*dtheta*dz under the retained aliases) summed over every
// (i, j, k) cell.
double analytic_domain_volume()
{
    const auto domain = physical_scalar_domain(
        NR, NTH, NZ, global_grid_geometry.staggered
            ? StaggerType::CGrid : StaggerType::Collocated);
    return static_cast<double>(domain.nx()) * domain.ny() * domain.nz()
         * dr * dtheta * dz;
}

struct BudgetCsvTestGuard
{
    explicit BudgetCsvTestGuard(std::string csv_path)
        : path(std::move(csv_path)),
          saved_simulation_time(simulation_time)
    {
        close_budget_csv();
        std::remove(path.c_str());
    }

    ~BudgetCsvTestGuard()
    {
        close_budget_csv();
        simulation_time = saved_simulation_time;
        std::remove(path.c_str());
    }

    std::string path;
    double saved_simulation_time;
};

}  // namespace

// ============================================================================
// compute_conservation_budget — integration math
// ============================================================================

TEST_CASE("budget on empty fields returns zero everything",
          "[diagnostics][conservation_budget]")
{
    clear_all_budget_fields();
    const auto budget = compute_conservation_budget();
    REQUIRE(budget.total_mass     == 0.0);
    REQUIRE(budget.theta_volume   == 0.0);
    REQUIRE(budget.q_volume_total == 0.0);
    REQUIRE(budget.dry_mass       == 0.0);
    REQUIRE(budget.total_water    == 0.0);
    REQUIRE(budget.moist_total_mass == 0.0);
    REQUIRE(budget.thermal_energy == 0.0);
}

TEST_CASE("budget total_mass = rho * domain_volume for constant rho",
          "[diagnostics][conservation_budget]")
{
    resize_all_budget_fields();
    const float rho_val = 1.2f;
    rho.fill(rho_val);
    // Need finite p > 0 and theta to keep the thermal_energy branch happy,
    // even though we only assert on total_mass here.
    p.fill(101325.0f);
    theta.fill(300.0f);

    const auto budget = compute_conservation_budget();
    const double expected = static_cast<double>(rho_val) * analytic_domain_volume();
    REQUIRE(budget.total_mass == Approx(expected).epsilon(1e-12));
}

TEST_CASE("conservation budgets use exact terrain cell volumes",
          "[diagnostics][conservation_budget][terrain][TE-005]")
{
    resize_all_budget_fields(0.0f);
    rho.fill(2.0f);
    p.fill(90000.0f);
    theta.fill(300.0f);

    TerrainRuntimeState terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    terrain.metrics.z.resize(NR, NTH, NZ, 0.0);
    terrain.metrics.z_face.resize(NR, NTH, NZ + 1, 0.0);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            double face = 20.0 * i + 10.0 * j;
            terrain.metrics.z_face(i, j, 0) = face;
            for (int k = 0; k < NZ; ++k)
            {
                const double thickness = 40.0 + 3.0 * i + 2.0 * j + k;
                terrain.metrics.z(i, j, k) = face + 0.5 * thickness;
                face += thickness;
                terrain.metrics.z_face(i, j, k + 1) = face;
            }
        }

    auto state = global_simulation_state();
    state.terrain_runtime = &terrain;
    const auto domain = physical_scalar_domain(state.grid);
    double expected_volume = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
                expected_volume += dr * dtheta
                    * (40.0 + 3.0 * i + 2.0 * j + k);

    const auto budget = compute_conservation_budget(state);
    REQUIRE(budget.total_mass == Approx(2.0 * expected_volume));
    REQUIRE(budget.theta_volume == Approx(600.0 * expected_volume));
    REQUIRE(budget.total_mass
            != Approx(2.0 * analytic_domain_volume()));
}

TEST_CASE("conservation budget exposes signed invalid inventory instead of clipping it",
          "[diagnostics][conservation_budget][guard]")
{
    const bool saved_staggered = global_grid_geometry.staggered;
    global_grid_geometry.staggered = false;
    resize_all_budget_fields(0.0f);
    rho.fill(1.0f);
    p.fill(100000.0f);
    theta.fill(300.0f);
    qv[1][1][1] = -0.01f;

    const auto budget = compute_conservation_budget();
    const double cell_volume = dr * dtheta * dz;
    REQUIRE(budget.total_water == Approx(-0.01 * cell_volume).epsilon(1.0e-6));
    global_grid_geometry.staggered = saved_staggered;
}

TEST_CASE("conservation budget fails closed on every invalid contributing field",
          "[diagnostics][conservation_budget][fail-closed][sci138]")
{
    const bool saved_staggered = global_grid_geometry.staggered;
    global_grid_geometry.staggered = false;
    resize_all_budget_fields(0.0f);
    rho.fill(1.0f);
    p.fill(100000.0f);
    theta.fill(300.0f);

    const std::array<std::pair<const char*, Field3D*>, 10> fields = {{
        {"rho", &rho}, {"p", &p}, {"theta", &theta}, {"qv", &qv},
        {"qc", &qc}, {"qr", &qr}, {"qi", &qi}, {"qs", &qs},
        {"qg", &qg}, {"qh", &qh},
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
                compute_conservation_budget(),
                Catch::Matchers::Contains("rejected non-finite"));
            (*field)[1][1][1] = saved;
        }
    }
    tmv::diagnostics::set_deterministic_reductions(false);

    qh.resize(NR - 1, NTH, NZ, 0.0f);
    REQUIRE_THROWS_WITH(
        compute_conservation_budget(),
        Catch::Matchers::Contains("match the grid"));
    qh.resize(NR, NTH, NZ, 0.0f);

    ConservationBudget invalid{};
    invalid.total_mass = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_WITH(
        evaluate_budget_drift(ConservationBudget{}, invalid),
        Catch::Matchers::Contains("finite snapshots"));
    global_grid_geometry.staggered = saved_staggered;
}

TEST_CASE("budget can compute from an explicit simulation state view",
          "[diagnostics][conservation_budget][simulation_state]")
{
    clear_all_budget_fields();

    int local_nr = 4;
    int local_nth = 5;
    int local_nz = 4;
    double local_dr = 5.0;
    double local_dz = 6.0;
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

    std::vector<double> local_rho0;
    std::vector<double> local_p0;
    std::vector<double> local_qv0;
    std::vector<double> local_theta0;
    std::vector<double> local_u0;
    std::vector<double> local_v0;
    double local_coriolis_f = 0.0;
    double local_time = 42.0;

    SimulationStateView state{
        {local_nr, local_nth, local_nz, local_dr, local_dz, local_dt, local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs, local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis_f,
        local_time};

    const auto budget = compute_conservation_budget(state);
    const auto local_domain = physical_scalar_domain(
        local_nr, local_nth, local_nz, StaggerType::Collocated);
    const double domain_volume = static_cast<double>(
        local_domain.nx() * local_domain.ny() * local_domain.nz())
                               * local_dr * local_dtheta * local_dz;

    REQUIRE(budget.total_mass == Approx(2.0 * domain_volume).epsilon(1e-12));
    REQUIRE(budget.theta_volume == Approx(2.0 * 300.0 * domain_volume).epsilon(1e-12));
    REQUIRE(budget.q_volume_total == Approx(2.0 * 0.125 * domain_volume).epsilon(1e-12));
    REQUIRE(budget.dry_mass == budget.total_mass);
    REQUIRE(budget.total_water == Approx(2.0 * 0.125 * domain_volume).epsilon(1e-12));
    REQUIRE(budget.q_volume_total == budget.total_water);
    REQUIRE(budget.moist_total_mass ==
            Approx(budget.dry_mass + budget.total_water).epsilon(1e-12));
}

TEST_CASE("budget CSV reporting uses explicit simulation state time",
          "[diagnostics][conservation_budget][simulation_state]")
{
    BudgetCsvTestGuard guard("/tmp/nimbus_budget_state_time.csv");
    simulation_time = 999.0;
    double explicit_time = 123.5;

    SimulationStateView state{
        {NR, NTH, NZ, dr, dz, dt, dtheta, global_grid_geometry},
        {rho, p, u, w, v, tracer,
         theta, qv, qc, qr, qi, qs, qh, qg, tke},
        {rho0_base, p0_base, qv0_base, theta0_base, u0_base, v0_base},
        {dtheta_dt_rad, dtheta_dt_pbl, dqv_dt_pbl,
         du_dt_pbl, dv_dt_pbl, dtke_dt_pbl},
        coriolis_f,
        explicit_time};

    ConservationBudget before{};
    before.total_mass = 1.0;
    ConservationBudget after = before;

    open_budget_csv(guard.path);
    report_budget_transition(state, "explicit_state", before, after, 1.0);
    close_budget_csv();

    std::ifstream csv(guard.path);
    REQUIRE(csv.good());

    std::string header;
    std::string row;
    REQUIRE(static_cast<bool>(std::getline(csv, header)));
    REQUIRE(static_cast<bool>(std::getline(csv, row)));
    REQUIRE(header.find("d_moist_total_mass,rel_moist_total_mass") != std::string::npos);
    REQUIRE(header.find("d_qv_mass,d_qc_mass,d_qr_mass") != std::string::npos);
    REQUIRE(row.rfind("123.5,explicit_state,", 0) == 0);
}

TEST_CASE("budget q_volume_total = rho * q_total * domain_volume for constant water species",
          "[diagnostics][conservation_budget]")
{
    resize_all_budget_fields();
    const float rho_val = 1.25f;
    rho.fill(rho_val);
    p.fill(101325.0f);
    theta.fill(300.0f);
    // Use values that are exactly representable in float32 (powers of 2)
    // so the float-stored fields and the double "expected" agree better
    // than the test's tolerance.
    qv.fill(0.0078125f);   // 1/128
    qc.fill(0.00390625f);  // 1/256
    qr.fill(0.001953125f); // 1/512
    // qi, qs, qg, qh stay at 0.

    const auto budget = compute_conservation_budget();
    const double q_total = 1.0 / 128.0 + 1.0 / 256.0 + 1.0 / 512.0;
    const double expected = static_cast<double>(rho_val) * q_total
                          * analytic_domain_volume();
    // Float32 storage + double accumulation; ~1e-7 absolute float precision
    // floor across ~500 cells gives a relative tolerance well under 1e-6.
    REQUIRE(budget.q_volume_total == Approx(expected).epsilon(1e-6));
    const double rho_volume = static_cast<double>(rho_val)
                            * analytic_domain_volume();
    REQUIRE(budget.water_species_mass[0] ==
            Approx(rho_volume / 128.0).epsilon(1e-6));
    REQUIRE(budget.water_species_mass[1] ==
            Approx(rho_volume / 256.0).epsilon(1e-6));
    REQUIRE(budget.water_species_mass[2] ==
            Approx(rho_volume / 512.0).epsilon(1e-6));
    for (std::size_t species = 3;
         species < budget.water_species_mass.size(); ++species)
    {
        REQUIRE(budget.water_species_mass[species] == 0.0);
    }
}

TEST_CASE("budget theta_volume = rho * theta * domain_volume for constant fields",
          "[diagnostics][conservation_budget]")
{
    resize_all_budget_fields();
    const float rho_val = 1.125f;
    rho.fill(rho_val);
    p.fill(101325.0f);
    theta.fill(305.0f);

    const auto budget = compute_conservation_budget();
    const double expected = static_cast<double>(rho_val) * 305.0
                          * analytic_domain_volume();
    REQUIRE(budget.theta_volume == Approx(expected).epsilon(1e-12));
}

TEST_CASE("budget density weights structured theta and water inventories",
          "[diagnostics][conservation_budget][cgrid]")
{
    // A volume-only sum cannot pass this test: rho varies independently in
    // each cell and the expected inventories are formed analytically from the
    // same rho*scalar products advanced by the C-grid transport kernel.
    resize_all_budget_fields();
    p.fill(101325.0f);

    double expected_theta = 0.0;
    double expected_water = 0.0;
    const double dv = dr * dtheta * dz;
    const auto domain = physical_scalar_domain(
        NR, NTH, NZ, StaggerType::Collocated);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float rho_cell = 0.75f + 0.125f * static_cast<float>((i + 2 * j + k) % 5);
                const float theta_cell = 280.0f + static_cast<float>(i + 3 * j + 2 * k);
                const float qv_cell = 0.001953125f * static_cast<float>(1 + (i + j + k) % 3);
                const float qc_cell = 0.0009765625f * static_cast<float>((i + 2 * k) % 2);
                rho[i][j][k] = rho_cell;
                theta[i][j][k] = theta_cell;
                qv[i][j][k] = qv_cell;
                qc[i][j][k] = qc_cell;

                if (i >= domain.i_begin && i < domain.i_end
                    && j >= domain.j_begin && j < domain.j_end
                    && k >= domain.k_begin && k < domain.k_end)
                {
                    const double rho_dv = static_cast<double>(rho_cell) * dv;
                    expected_theta += rho_dv * static_cast<double>(theta_cell);
                    expected_water += rho_dv *
                        (static_cast<double>(qv_cell) + static_cast<double>(qc_cell));
                }
            }

    const auto budget = compute_conservation_budget();
    REQUIRE(budget.theta_volume == Approx(expected_theta).epsilon(1e-12));
    REQUIRE(budget.q_volume_total == Approx(expected_water).epsilon(1e-12));
    REQUIRE(budget.total_water == Approx(expected_water).epsilon(1e-12));
}

TEST_CASE("budget ownership excludes C-grid and collocated closure storage",
          "[diagnostics][conservation_budget][cgrid][ghost]")
{
    int nx = 6;
    int ny = 7;
    int nz_local = 3;
    double dx = 20.0;
    double dy = 30.0;
    double dz_local = 40.0;
    double dt_local = 1.0;

    GridGeometry geometry;
    geometry.initialize(nx, ny, nz_local, dx, dz_local, dy,
                        CoordinateSystem::Cartesian, StaggerType::CGrid);

    Field3D local_rho(nx, ny, nz_local, 2.0f);
    Field3D local_p(nx, ny, nz_local, 101325.0f);
    Field3D local_u(nx, ny, nz_local, 0.0f);
    Field3D local_w(nx, ny, nz_local, 0.0f);
    Field3D local_v(nx, ny, nz_local, 0.0f);
    Field3D local_tracer(nx, ny, nz_local, 0.0f);
    Field3D local_theta(nx, ny, nz_local, 300.0f);
    Field3D local_qv(nx, ny, nz_local, 0.125f);
    Field3D local_qc(nx, ny, nz_local, 0.0f);
    Field3D local_qr(nx, ny, nz_local, 0.0f);
    Field3D local_qi(nx, ny, nz_local, 0.0f);
    Field3D local_qs(nx, ny, nz_local, 0.0f);
    Field3D local_qh(nx, ny, nz_local, 0.0f);
    Field3D local_qg(nx, ny, nz_local, 0.0f);
    Field3D local_tke(nx, ny, nz_local, 0.0f);
    Field3D local_dtheta_dt_rad(nx, ny, nz_local, 0.0f);
    Field3D local_dtheta_dt_pbl(nx, ny, nz_local, 0.0f);
    Field3D local_dqv_dt_pbl(nx, ny, nz_local, 0.0f);
    Field3D local_du_dt_pbl(nx, ny, nz_local, 0.0f);
    Field3D local_dv_dt_pbl(nx, ny, nz_local, 0.0f);
    Field3D local_dtke_dt_pbl(nx, ny, nz_local, 0.0f);
    std::vector<double> local_rho0;
    std::vector<double> local_p0;
    std::vector<double> local_qv0;
    std::vector<double> local_theta0;
    std::vector<double> local_u0;
    std::vector<double> local_v0;
    double local_coriolis_f = 0.0;
    double local_time = 0.0;

    SimulationStateView state{
        {nx, ny, nz_local, dx, dz_local, dt_local, dy, geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis_f,
        local_time};

    const double dv = dx * dy * dz_local;
    const double physical_volume = static_cast<double>((nx - 2) * (ny - 2) * nz_local) * dv;
    const auto clean_cgrid = compute_conservation_budget(state);
    REQUIRE(clean_cgrid.total_mass == Approx(2.0 * physical_volume).epsilon(1e-12));
    REQUIRE(clean_cgrid.theta_volume == Approx(2.0 * 300.0 * physical_volume).epsilon(1e-12));
    REQUIRE(clean_cgrid.q_volume_total == Approx(2.0 * 0.125 * physical_volume).epsilon(1e-12));
    REQUIRE(clean_cgrid.total_water == clean_cgrid.q_volume_total);

    // Poison every lateral ghost, including all four corner overlaps. None of
    // these values represents atmospheric volume on the C-grid.
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            if (i != 0 && i != nx - 1 && j != 0 && j != ny - 1)
                continue;
            for (int k = 0; k < nz_local; ++k)
            {
                local_rho[i][j][k] = 50.0f;
                local_theta[i][j][k] = 900.0f;
                local_qv[i][j][k] = 0.5f;
                local_qc[i][j][k] = 0.25f;
            }
        }

    const auto poisoned_cgrid = compute_conservation_budget(state);
    REQUIRE(poisoned_cgrid.total_mass == clean_cgrid.total_mass);
    REQUIRE(poisoned_cgrid.theta_volume == clean_cgrid.theta_volume);
    REQUIRE(poisoned_cgrid.q_volume_total == clean_cgrid.q_volume_total);
    REQUIRE(poisoned_cgrid.dry_mass == clean_cgrid.dry_mass);
    REQUIRE(poisoned_cgrid.total_water == clean_cgrid.total_water);
    REQUIRE(poisoned_cgrid.moist_total_mass == clean_cgrid.moist_total_mass);
    REQUIRE(poisoned_cgrid.thermal_energy == clean_cgrid.thermal_energy);

    // Collocated uses the same lateral closure planes and additionally excludes
    // its vertical closure planes. The poisoned lateral storage remains inert.
    geometry.staggered = false;
    const auto collocated = compute_conservation_budget(state);
    const double collocated_volume = static_cast<double>(
        (nx - 2) * (ny - 2) * (nz_local - 2)) * dv;
    REQUIRE(collocated.total_mass == Approx(2.0 * collocated_volume));
    REQUIRE(collocated.theta_volume == Approx(2.0 * 300.0 * collocated_volume));
    REQUIRE(collocated.q_volume_total == Approx(2.0 * 0.125 * collocated_volume));

    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (const int k : {0, nz_local - 1})
            {
                local_rho[i][j][k] = 1.0e6f;
                local_theta[i][j][k] = 1.0e6f;
                local_qv[i][j][k] = 100.0f;
            }
    const auto collocated_vertical_poison = compute_conservation_budget(state);
    REQUIRE(collocated_vertical_poison.total_mass == collocated.total_mass);
    REQUIRE(collocated_vertical_poison.theta_volume == collocated.theta_volume);
    REQUIRE(collocated_vertical_poison.q_volume_total == collocated.q_volume_total);
}

TEST_CASE("budget distinguishes dry, water, and moist-total mass",
          "[diagnostics][conservation_budget]")
{
    resize_all_budget_fields();
    rho.fill(1.1f);
    p.fill(101325.0f);
    theta.fill(300.0f);
    qv.fill(0.012f);
    qc.fill(0.002f);

    const auto budget = compute_conservation_budget();
    REQUIRE(budget.dry_mass == budget.total_mass);
    REQUIRE(budget.total_water == budget.q_volume_total);
    REQUIRE(budget.moist_total_mass ==
            Approx(budget.dry_mass + budget.total_water).epsilon(1e-12));
    REQUIRE(budget.moist_total_mass > budget.dry_mass);
}

TEST_CASE("budget is deterministic across repeated calls on identical fields",
          "[diagnostics][conservation_budget]")
{
    resize_all_budget_fields();
    rho.fill(1.05f);
    p.fill(101325.0f);
    theta.fill(310.0f);
    qv.fill(0.008f);

    const auto first  = compute_conservation_budget();
    const auto second = compute_conservation_budget();

    // Two consecutive calls with no field changes must produce bit-identical
    // budgets, otherwise drift diagnostics are unreliable.
    REQUIRE(first.total_mass     == second.total_mass);
    REQUIRE(first.theta_volume   == second.theta_volume);
    REQUIRE(first.q_volume_total == second.q_volume_total);
    REQUIRE(first.dry_mass       == second.dry_mass);
    REQUIRE(first.total_water    == second.total_water);
    REQUIRE(first.moist_total_mass == second.moist_total_mass);
    REQUIRE(first.thermal_energy == second.thermal_energy);
}

TEST_CASE("budget total_mass invariant under shuffling rho between cells of equal volume",
          "[diagnostics][conservation_budget]")
{
    // On the Cartesian harness grid, cells at the same i (same x center)
    // have identical cell volumes; shuffling rho values among (j, k) at
    // fixed i must not change the integral.
    resize_all_budget_fields();
    p.fill(101325.0f);
    theta.fill(300.0f);

    // Set rho on each i-shell to a unique constant.
    for (int i = 0; i < NR; ++i)
    {
        const float rho_i = 1.0f + 0.01f * static_cast<float>(i);
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                rho[i][j][k] = rho_i;
    }

    const auto before = compute_conservation_budget();

    // Swap (j=0, k=0) and (j=NTH-1, k=NZ-1) within each i-shell. Equal
    // cell volume, so the integral is invariant.
    for (int i = 0; i < NR; ++i)
    {
        const float a = rho[i][0][0];
        const float b = rho[i][NTH - 1][NZ - 1];
        rho[i][0][0]                 = b;
        rho[i][NTH - 1][NZ - 1]      = a;
    }

    const auto after = compute_conservation_budget();
    REQUIRE(after.total_mass == before.total_mass);
}

// ============================================================================
// evaluate_budget_drift — warn-trigger logic
// ============================================================================

TEST_CASE("drift on identical snapshots warns neither group",
          "[diagnostics][conservation_budget]")
{
    ConservationBudget snap{};
    snap.total_mass     = 1.0e14;
    snap.theta_volume   = 5.0e16;
    snap.q_volume_total = 1.5e11;
    snap.dry_mass       = snap.total_mass;
    snap.total_water    = snap.q_volume_total;
    snap.moist_total_mass = snap.dry_mass + snap.total_water;
    snap.thermal_energy = 3.2e20;

    const auto drift = evaluate_budget_drift(snap, snap);
    REQUIRE(drift.warn_kernel   == false);
    REQUIRE(drift.warn_physical == false);
    REQUIRE(drift.d_total_mass     == 0.0);
    REQUIRE(drift.d_theta_volume   == 0.0);
    REQUIRE(drift.d_q_volume       == 0.0);
    REQUIRE(drift.rel_total_mass   == 0.0);
    REQUIRE(drift.rel_q_volume     == 0.0);
}

TEST_CASE("drift just below kernel threshold does not warn",
          "[diagnostics][conservation_budget]")
{
    ConservationBudget before{};
    before.total_mass     = 1.0e14;
    before.theta_volume   = 5.0e16;
    before.q_volume_total = 1.5e11;
    before.dry_mass       = before.total_mass;
    before.total_water    = before.q_volume_total;
    before.moist_total_mass = before.dry_mass + before.total_water;
    before.thermal_energy = 3.2e20;

    // Drift in q_volume just under the kernel threshold (1e-10).
    ConservationBudget after = before;
    after.q_volume_total += before.q_volume_total * (BUDGET_THRESH_KERNEL * 0.5);

    const auto drift = evaluate_budget_drift(before, after);
    REQUIRE(drift.warn_kernel   == false);
    REQUIRE(drift.warn_physical == false);
    REQUIRE(drift.rel_q_volume <= BUDGET_THRESH_KERNEL);
}

TEST_CASE("drift just above kernel threshold trips warn_kernel",
          "[diagnostics][conservation_budget]")
{
    ConservationBudget before{};
    before.total_mass     = 1.0e14;
    before.theta_volume   = 5.0e16;
    before.q_volume_total = 1.5e11;
    before.dry_mass       = before.total_mass;
    before.total_water    = before.q_volume_total;
    before.moist_total_mass = before.dry_mass + before.total_water;
    before.thermal_energy = 3.2e20;

    // Drift in total_mass just above the kernel threshold.
    ConservationBudget after = before;
    after.total_mass += before.total_mass * (BUDGET_THRESH_KERNEL * 10.0);

    const auto drift = evaluate_budget_drift(before, after);
    REQUIRE(drift.warn_kernel   == true);
    REQUIRE(drift.warn_physical == false);
    REQUIRE(drift.rel_total_mass > BUDGET_THRESH_KERNEL);
}

TEST_CASE("drift trips warn_kernel for any of total_mass / theta_volume / q_volume",
          "[diagnostics][conservation_budget]")
{
    ConservationBudget before{};
    before.total_mass     = 1.0e14;
    before.theta_volume   = 5.0e16;
    before.q_volume_total = 1.5e11;
    before.dry_mass       = before.total_mass;
    before.total_water    = before.q_volume_total;
    before.moist_total_mass = before.dry_mass + before.total_water;
    before.thermal_energy = 3.2e20;

    // Perturbations chosen 10x above BUDGET_THRESH_KERNEL (1e-7, the
    // fp32-rho noise floor) — see commit a61e5b2 for the threshold
    // calibration. The semantic intent is "any real drift trips warn_kernel";
    // 1e-6 is comfortably above the noise floor and well below physical
    // bug magnitudes.
    constexpr double kKernelDrift = 1.0e-6;

    SECTION("total_mass drift")
    {
        ConservationBudget after = before;
        after.total_mass *= 1.0 + kKernelDrift;
        REQUIRE(evaluate_budget_drift(before, after).warn_kernel);
    }
    SECTION("theta_volume drift")
    {
        ConservationBudget after = before;
        after.theta_volume *= 1.0 + kKernelDrift;
        REQUIRE(evaluate_budget_drift(before, after).warn_kernel);
    }
    SECTION("q_volume drift")
    {
        ConservationBudget after = before;
        after.q_volume_total *= 1.0 + kKernelDrift;
        REQUIRE(evaluate_budget_drift(before, after).warn_kernel);
    }
}

TEST_CASE("drift in physical group only trips warn_physical (not warn_kernel)",
          "[diagnostics][conservation_budget]")
{
    ConservationBudget before{};
    before.total_mass     = 1.0e14;
    before.theta_volume   = 5.0e16;
    before.q_volume_total = 1.5e11;
    before.dry_mass       = before.total_mass;
    before.total_water    = before.q_volume_total;
    before.moist_total_mass = before.dry_mass + before.total_water;
    before.thermal_energy = 3.2e20;

    // Exercise the derived moist-total warning independently of the three
    // transported inventories. Real compute snapshots keep
    // moist_total_mass = total_mass + q_volume_total; this deliberately
    // inconsistent synthetic snapshot verifies the warning-group routing.
    ConservationBudget after = before;
    after.moist_total_mass *= 1.0 + BUDGET_THRESH_WATER * 10.0;

    const auto drift = evaluate_budget_drift(before, after);
    REQUIRE(drift.warn_physical == true);
    // total_mass, theta_volume, q_volume_total are unchanged → no kernel warn.
    REQUIRE(drift.warn_kernel == false);
}

TEST_CASE("drift relative computation guards against zero denominator",
          "[diagnostics][conservation_budget]")
{
    // before = 0 means relative_drift falls back to absolute (denominator
    // floored at 1.0). Verify a sub-1.0 absolute drift does not warn even
    // though the formal "relative" change is undefined.
    ConservationBudget before{};
    ConservationBudget after{};
    after.q_volume_total = 1.0e-12;  // tiny absolute drift, "infinite" relative

    const auto drift = evaluate_budget_drift(before, after);
    REQUIRE(drift.rel_q_volume == Approx(1.0e-12).epsilon(1e-15));
    REQUIRE(drift.warn_kernel == false);
}

extern CoordinateSystem& global_coordinate_system;

TEST_CASE("Cartesian budgets use uniform volumes: blob mass is x-position invariant",
          "[diagnostics][conservation_budget][cartesian]")
{
    // SCIENCE.md C4 regression: the diagnostics applied cylindrical
    // r-weighted volumes unconditionally, so on a Cartesian grid the same
    // mass placed at large x outweighed the identical mass at small x by
    // ~(i2/i1). Under correct uniform dx*dy*dz weights, total_mass must be
    // exactly invariant under translating a blob in x, and a constant rho
    // must integrate to rho * (nx*dx) * (ny*dy) * (nz*dz).
    const CoordinateSystem saved_coord = global_coordinate_system;
    global_coordinate_system = CoordinateSystem::Cartesian;

    resize_all_budget_fields();
    p.fill(101325.0f);
    theta.fill(300.0f);

    // Constant rho integrates to the rectangular domain volume.
    const float rho_val = 1.1f;
    rho.fill(rho_val);
    const double rect_volume = analytic_domain_volume();
    const auto uniform = compute_conservation_budget();
    REQUIRE(uniform.total_mass
            == Approx(static_cast<double>(rho_val) * rect_volume).epsilon(1e-12));

    // A rho anomaly at small x vs the same anomaly at large x: identical
    // total mass. Under the old cylindrical weighting this differed by
    // the ratio of the cell radii (~7x for i=0 vs i=NR-1 on this grid).
    rho.fill(1.0f);
    rho[1][2][3] += 0.5f;
    const auto low_x = compute_conservation_budget();

    rho.fill(1.0f);
    rho[NR - 2][2][3] += 0.5f;
    const auto high_x = compute_conservation_budget();

    REQUIRE(low_x.total_mass == Approx(high_x.total_mass).epsilon(1e-12));

    global_coordinate_system = saved_coord;
}

TEST_CASE("deterministic conservation reductions are thread-count independent",
          "[diagnostics][conservation_budget][deterministic]")
{
    resize_all_budget_fields();
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] = static_cast<float>(0.9 + 0.013 * (i + j + k));
                p[i][j][k] = static_cast<float>(90000.0 + 17.0 * (i + 2 * j + k));
                theta[i][j][k] = static_cast<float>(295.0 + 0.1 * (i + j + k));
                qv[i][j][k] = static_cast<float>(0.001 * (1 + (i + j + k) % 7));
            }

    tmv::diagnostics::set_deterministic_reductions(true);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto one = compute_conservation_budget();
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto many = compute_conservation_budget();
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    tmv::diagnostics::set_deterministic_reductions(false);

    REQUIRE(one.total_mass == many.total_mass);
    REQUIRE(one.theta_volume == many.theta_volume);
    REQUIRE(one.q_volume_total == many.q_volume_total);
    REQUIRE(one.total_water == many.total_water);
    REQUIRE(one.thermal_energy == many.thermal_energy);
}

TEST_CASE("fused conservation and moisture snapshots are standalone exact",
          "[diagnostics][conservation_budget][moisture][deterministic]")
{
    resize_all_budget_fields();
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float pattern = static_cast<float>(i + 2 * j + 3 * k);
                rho[i][j][k] = 0.85f + 0.002f * pattern;
                p[i][j][k] = 88000.0f + 11.0f * pattern;
                theta[i][j][k] = 292.0f + 0.03f * pattern;
                qv[i][j][k] = 0.001f + 1.0e-6f * pattern;
                qc[i][j][k] = 2.0e-4f + 2.0e-7f * pattern;
                qr[i][j][k] = 1.0e-4f + 3.0e-7f * pattern;
                qi[i][j][k] = 8.0e-5f + 4.0e-8f * pattern;
                qs[i][j][k] = 7.0e-5f + 5.0e-8f * pattern;
                qg[i][j][k] = 6.0e-5f + 6.0e-8f * pattern;
                qh[i][j][k] = 5.0e-5f + 7.0e-8f * pattern;
            }

    tmv::diagnostics::set_deterministic_reductions(true);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const ConservationBudget standalone_conservation =
        compute_conservation_budget();
    const MoistureBudget standalone_moisture = compute_moisture_budget();

    ConservationBudget fused_conservation;
    MoistureBudget fused_moisture;
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    compute_conservation_and_moisture_budgets(
        global_simulation_state(), fused_conservation, fused_moisture);
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    tmv::diagnostics::set_deterministic_reductions(false);

    REQUIRE(fused_conservation.total_mass
            == standalone_conservation.total_mass);
    REQUIRE(fused_conservation.theta_volume
            == standalone_conservation.theta_volume);
    REQUIRE(fused_conservation.q_volume_total
            == standalone_conservation.q_volume_total);
    REQUIRE(fused_conservation.dry_mass
            == standalone_conservation.dry_mass);
    REQUIRE(fused_conservation.total_water
            == standalone_conservation.total_water);
    REQUIRE(fused_conservation.water_species_mass
            == standalone_conservation.water_species_mass);
    REQUIRE(fused_conservation.moist_total_mass
            == standalone_conservation.moist_total_mass);
    REQUIRE(fused_conservation.thermal_energy
            == standalone_conservation.thermal_energy);
    REQUIRE(fused_moisture.vapor == standalone_moisture.vapor);
    REQUIRE(fused_moisture.liquid == standalone_moisture.liquid);
    REQUIRE(fused_moisture.ice == standalone_moisture.ice);
    REQUIRE(fused_moisture.total_water
            == standalone_moisture.total_water);

    qg[2][2][2] = std::numeric_limits<float>::quiet_NaN();
    fused_conservation.total_mass = 123.0;
    fused_moisture.vapor = 456.0;
    REQUIRE_THROWS(compute_conservation_and_moisture_budgets(
        global_simulation_state(), fused_conservation, fused_moisture));
    REQUIRE(fused_conservation.total_mass == 123.0);
    REQUIRE(fused_moisture.vapor == 456.0);
}
TEST_CASE("derived diagnostic gates select radar and dynamics independently",
          "[diagnostics][cpu016]")
{
    set_step_derived_diagnostics_enabled(false);
    REQUIRE_FALSE(step_derived_diagnostics_enabled());
    REQUIRE_FALSE(step_radar_diagnostics_enabled());
    REQUIRE_FALSE(step_dynamics_diagnostics_enabled());
    REQUIRE_FALSE(step_vorticity_diagnostics_enabled());
    REQUIRE_FALSE(step_angular_momentum_diagnostics_enabled());
    REQUIRE_FALSE(step_pressure_diagnostics_enabled());

    set_step_derived_diagnostic_requirements(true, false, false, false);
    REQUIRE(step_derived_diagnostics_enabled());
    REQUIRE(step_radar_diagnostics_enabled());
    REQUIRE_FALSE(step_dynamics_diagnostics_enabled());

    set_step_derived_diagnostic_requirements(false, true, false, false);
    REQUIRE(step_derived_diagnostics_enabled());
    REQUIRE_FALSE(step_radar_diagnostics_enabled());
    REQUIRE(step_dynamics_diagnostics_enabled());
    REQUIRE(step_vorticity_diagnostics_enabled());
    REQUIRE_FALSE(step_angular_momentum_diagnostics_enabled());
    REQUIRE_FALSE(step_pressure_diagnostics_enabled());

    set_step_derived_diagnostic_requirements(false, false, true, false);
    REQUIRE(step_dynamics_diagnostics_enabled());
    REQUIRE_FALSE(step_vorticity_diagnostics_enabled());
    REQUIRE(step_angular_momentum_diagnostics_enabled());
    REQUIRE_FALSE(step_pressure_diagnostics_enabled());

    set_step_derived_diagnostic_requirements(false, false, false, true);
    REQUIRE(step_dynamics_diagnostics_enabled());
    REQUIRE_FALSE(step_vorticity_diagnostics_enabled());
    REQUIRE_FALSE(step_angular_momentum_diagnostics_enabled());
    REQUIRE(step_pressure_diagnostics_enabled());

    // Restore the standalone/test default for subsequent cases.
    set_step_derived_diagnostics_enabled(true);
}
