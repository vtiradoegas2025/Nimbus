/**
 * @file test_energy_budget.cpp
 * @brief Unit tests for the F.5b energy budget diagnostic.
 *
 * Each reservoir is checked against its analytic domain integral on the
 * Cartesian harness grid (uniform cell volume dx*dy*dz). Physical constants
 * (g, R_d, c_p, p0, L_v) are the shared header values, so the "expected" side
 * is independent of the diagnostic's own arithmetic.
 */

#include "catch2/catch.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/orchestration/dynamics/pressure_cell_energy.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "diagnostics/energy_budget.hpp"
#include "dynamics/completed_step_energy_report.hpp"
#include "core/runtime/simulation.hpp"
#include "terrain/base/topography.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

struct DeterministicEnergyGuard
{
    bool previous = tmv::diagnostics::deterministic_reductions_enabled();
    DeterministicEnergyGuard() { tmv::diagnostics::set_deterministic_reductions(true); }
    ~DeterministicEnergyGuard() { tmv::diagnostics::set_deterministic_reductions(previous); }
};

void resize_energy_fields(float fill = 0.0f)
{
    rho.resize(NR, NTH, NZ, fill);
    p.resize(NR, NTH, NZ, fill);
    theta.resize(NR, NTH, NZ, fill);
    u.resize(NR, NTH, NZ, fill);
    v.resize(NR, NTH, NZ, fill);
    w.resize(NR, NTH, NZ, fill);
    qv.resize(NR, NTH, NZ, fill);
}

TEST_CASE("Completed-step energy distinguishes structural validity from physical closure",
          "[diagnostics][energy][completed-step][closure]")
{
    tmv::CompletedStepEnergyReportV1 report;
    report.acoustic_committed_change_j = 100.0;
    report.dry_internal_change_j = 80.0;
    report.committed_owner_sum_j = tmv::completed_step_owner_sum(report);
    report.closure_residual_j =
        report.committed_owner_sum_j -
        tmv::completed_step_reservoir_sum(report);
    report.complete = true;

    REQUIRE(tmv::structurally_valid_completed_step_energy_report(report));
    REQUIRE(tmv::valid_completed_step_energy_report(report));
    REQUIRE(tmv::completed_step_energy_normalized_closure_residual(report)
            == Approx(0.2));
    REQUIRE_FALSE(tmv::completed_step_energy_is_physically_closed(
        report, 1.0e-6));
    REQUIRE(tmv::completed_step_energy_is_physically_closed(
        report, 0.2));
    REQUIRE(tmv::completed_step_energy_is_physically_closed(
        report, 0.0, 20.0));

    report.committed_owner_sum_j += 1.0;
    REQUIRE_FALSE(tmv::structurally_valid_completed_step_energy_report(report));
    REQUIRE_FALSE(tmv::completed_step_energy_is_physically_closed(
        report, 1.0));
}

void resize_moist_energy_fields(float fill = 0.0f)
{
    resize_energy_fields(fill);
    qc.resize(NR, NTH, NZ, fill);
    qr.resize(NR, NTH, NZ, fill);
    qi.resize(NR, NTH, NZ, fill);
    qs.resize(NR, NTH, NZ, fill);
    qg.resize(NR, NTH, NZ, fill);
    qh.resize(NR, NTH, NZ, fill);
    tke.resize(NR, NTH, NZ, fill);
}

void clear_energy_fields()
{
    rho = Field3D();
    p = Field3D();
    theta = Field3D();
    u = Field3D();
    v = Field3D();
    w = Field3D();
    qv = Field3D();
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

// Independent geometric integral of geopotential weight: sum z_center * dV.
double analytic_geopotential_volume()
{
    double z_sum = 0.0;
    const auto domain = physical_scalar_domain(
        NR, NTH, NZ, global_grid_geometry.staggered
            ? StaggerType::CGrid : StaggerType::Collocated);
    for (int k = domain.k_begin; k < domain.k_end; ++k)
        z_sum += (static_cast<double>(k) + 0.5) * dz;
    return z_sum * cell_volume()
         * static_cast<double>(domain.nx()) * domain.ny();
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

}  // namespace

TEST_CASE("energy on empty fields returns zero", "[diagnostics][energy_budget]")
{
    clear_energy_fields();
    const auto e = compute_energy_budget();
    REQUIRE(e.kinetic == 0.0);
    REQUIRE(e.internal == 0.0);
    REQUIRE(e.potential == 0.0);
    REQUIRE(e.latent == 0.0);
    REQUIRE(e.total == 0.0);

    qv.resize(NR, NTH, NZ, 0.0f);
    REQUIRE_THROWS_WITH(
        compute_energy_budget(),
        Catch::Matchers::Contains("match the grid"));
    clear_energy_fields();
}

TEST_CASE("energy budget fails closed on every invalid contributing field",
          "[diagnostics][energy_budget][fail-closed][sci138]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = false;
    resize_energy_fields(0.0f);
    rho.fill(1.0f);
    p.fill(static_cast<float>(p0));
    theta.fill(300.0f);
    const std::array<std::pair<const char*, Field3D*>, 7> fields = {{
        {"rho", &rho}, {"p", &p}, {"theta", &theta}, {"u", &u},
        {"v", &v}, {"w", &w}, {"qv", &qv},
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
                compute_energy_budget(),
                Catch::Matchers::Contains("rejected non-finite"));
            (*field)[1][1][1] = saved;
        }
    }
    tmv::diagnostics::set_deterministic_reductions(false);

    w.resize(NR, NTH, NZ - 1, 0.0f);
    REQUIRE_THROWS_WITH(
        compute_energy_budget(),
        Catch::Matchers::Contains("match the grid"));
    w.resize(NR, NTH, NZ, 0.0f);

    p[1][1][1] = 0.0f;
    REQUIRE_THROWS_WITH(
        compute_energy_budget(),
        Catch::Matchers::Contains("nonphysical"));
    p[1][1][1] = static_cast<float>(p0);

    EnergyBudget invalid{};
    invalid.total = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_WITH(
        evaluate_energy_drift(EnergyBudget{}, invalid),
        Catch::Matchers::Contains("finite snapshots"));
}

TEST_CASE("kinetic energy = 0.5*rho*|U|^2*volume for a uniform wind",
          "[diagnostics][energy_budget]")
{
    resize_energy_fields(0.0f);
    const float rho_val = 1.0f;
    rho.fill(rho_val);
    p.fill(static_cast<float>(p0));   // T = theta -> internal stays finite
    theta.fill(300.0f);
    u.fill(2.0f);
    v.fill(3.0f);
    w.fill(6.0f);  // |U|^2 = 4 + 9 + 36 = 49

    const auto e = compute_energy_budget();
    const double V = analytic_domain_volume();
    const double expected = 0.5 * static_cast<double>(rho_val) * 49.0 * V;
    REQUIRE(e.kinetic == Approx(expected).epsilon(1e-5));
}

TEST_CASE("internal energy = rho*c_v*T*volume with T = theta at p = p0",
          "[diagnostics][energy_budget]")
{
    resize_energy_fields(0.0f);
    const float rho_val = 1.0f;
    const double T = 300.0;
    rho.fill(rho_val);
    p.fill(static_cast<float>(p0));   // (p/p0)^kappa = 1 -> T = theta
    theta.fill(static_cast<float>(T));

    const auto e = compute_energy_budget();
    const double cv = cp - R_d;
    const double expected = static_cast<double>(rho_val) * cv * T * analytic_domain_volume();
    REQUIRE(e.internal == Approx(expected).epsilon(1e-5));
}

TEST_CASE("energy budget can compute from an explicit simulation state view",
          "[diagnostics][energy_budget][simulation_state]")
{
    clear_energy_fields();

    int local_nr = 2;
    int local_nth = 2;
    int local_nz = 3;
    double local_dr = 5.0;
    double local_dz = 11.0;
    double local_dt = 0.25;
    double local_dtheta = 7.0;
    GridGeometry local_geometry;

    Field3D local_rho(local_nr, local_nth, local_nz, 2.0f);
    Field3D local_p(local_nr, local_nth, local_nz, static_cast<float>(p0));
    Field3D local_u(local_nr, local_nth, local_nz, 2.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 6.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 3.0f);
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

    const auto e = compute_energy_budget(state);
    const auto local_domain = physical_scalar_domain(
        local_nr, local_nth, local_nz, StaggerType::Collocated);
    const double volume = static_cast<double>(
        local_domain.nx() * local_domain.ny() * local_domain.nz())
                        * local_dr * local_dtheta * local_dz;

    REQUIRE(e.kinetic == Approx(0.5 * 2.0 * 49.0 * volume).epsilon(1e-12));
    REQUIRE(e.internal == Approx(2.0 * (cp - R_d) * 300.0 * volume).epsilon(1e-12));
    REQUIRE(e.latent == Approx(2.0 * static_cast<double>(L_v) * 0.125 * volume).epsilon(1e-12));
}

TEST_CASE("moist-mixture energy v1 resolves every phase and total-mass reservoir",
          "[diagnostics][energy_budget][moist-v1][sci140]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = false;
    resize_moist_energy_fields(0.0f);

    constexpr float rho_d = 2.0f;
    constexpr float qv_k = 1.0f / 128.0f;
    constexpr float qc_k = 1.0f / 256.0f;
    constexpr float qr_k = 1.0f / 512.0f;
    constexpr float qi_k = 1.0f / 1024.0f;
    constexpr float qs_k = 1.0f / 2048.0f;
    constexpr float qg_k = 1.0f / 4096.0f;
    constexpr float qh_k = 1.0f / 8192.0f;
    constexpr float tke_k = 2.5f;
    const float temperature = static_cast<float>(
        physical_constants::freezing_temperature_k + 8.0);
    rho.fill(rho_d);
    p.fill(static_cast<float>(p0));
    theta.fill(temperature);
    u.fill(3.0f);
    v.fill(4.0f);
    w.fill(0.0f);
    qv.fill(qv_k);
    qc.fill(qc_k);
    qr.fill(qr_k);
    qi.fill(qi_k);
    qs.fill(qs_k);
    qg.fill(qg_k);
    qh.fill(qh_k);
    tke.fill(tke_k);

    const auto snapshot = compute_energy_diagnostic_snapshot_v1();
    const auto energy = snapshot.moist_mixture;
    const auto legacy = compute_energy_budget();
    const double volume = analytic_domain_volume();
    const double delta_t = static_cast<double>(temperature)
        - physical_constants::freezing_temperature_k;
    const double q_liquid = static_cast<double>(qc_k + qr_k);
    const double q_ice =
        static_cast<double>(qi_k + qs_k + qg_k + qh_k);
    const double q_total = static_cast<double>(qv_k) + q_liquid + q_ice;
    const double dry_mass = static_cast<double>(rho_d) * volume;
    const double mixture_mass = dry_mass * (1.0 + q_total);
    const double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    const double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    const double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk
            * physical_constants::freezing_temperature_k;

    REQUIRE(energy.dry_internal
            == Approx(dry_mass * cv_d * delta_t).epsilon(1e-12));
    REQUIRE(energy.water_sensible == Approx(
        dry_mass * delta_t
        * (static_cast<double>(qv_k) * cv_v
           + q_liquid
               * physical_constants::specific_heat_liquid_water_jkgk
           + q_ice * physical_constants::specific_heat_ice_jkgk))
        .epsilon(1e-12));
    REQUIRE(energy.phase_reference == Approx(
        dry_mass
        * (static_cast<double>(qv_k) * vapor_reference
           - q_ice * physical_constants::latent_heat_fusion_jkg))
        .epsilon(1e-12));
    REQUIRE(energy.resolved_kinetic
            == Approx(0.5 * mixture_mass * 25.0).epsilon(1e-12));
    REQUIRE(energy.gravitational == Approx(
        static_cast<double>(rho_d) * (1.0 + q_total)
        * physical_constants::gravity_ms2
        * analytic_geopotential_volume()).epsilon(1e-12));
    REQUIRE(energy.subgrid_tke
            == Approx(dry_mass * static_cast<double>(tke_k)).epsilon(1e-12));
    REQUIRE(energy.total_water_mass
            == Approx(dry_mass * q_total).epsilon(1e-12));
    REQUIRE(energy.total == Approx(
        energy.dry_internal + energy.water_sensible
        + energy.phase_reference + energy.resolved_kinetic
        + energy.gravitational + energy.subgrid_tke).epsilon(1e-12));
    REQUIRE(snapshot.legacy_proxy.kinetic
            == Approx(legacy.kinetic).epsilon(1e-12));
    REQUIRE(snapshot.legacy_proxy.internal
            == Approx(legacy.internal).epsilon(1e-12));
    REQUIRE(snapshot.legacy_proxy.potential
            == Approx(legacy.potential).epsilon(1e-12));
    REQUIRE(snapshot.legacy_proxy.latent
            == Approx(legacy.latent).epsilon(1e-12));
    REQUIRE(snapshot.legacy_proxy.total
            == Approx(legacy.total).epsilon(1e-12));
    const auto& flat_v2 = snapshot.moist_mixture_v2;
    REQUIRE(flat_v2.dry_internal == energy.dry_internal);
    REQUIRE(flat_v2.water_sensible == energy.water_sensible);
    REQUIRE(flat_v2.phase_reference == energy.phase_reference);
    REQUIRE(flat_v2.resolved_kinetic == energy.resolved_kinetic);
    REQUIRE(flat_v2.gravitational == energy.gravitational);
    REQUIRE(flat_v2.subgrid_tke == energy.subgrid_tke);
    REQUIRE(flat_v2.total_water_mass == energy.total_water_mass);
    REQUIRE(flat_v2.total == energy.total);
}

TEST_CASE("moist-mixture energy v1 uses C-grid component control volumes",
          "[diagnostics][energy_budget][moist-v1][cgrid][sci140]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    resize_moist_energy_fields(0.0f);
    rho.fill(1.0f);
    p.fill(static_cast<float>(p0));
    theta.fill(static_cast<float>(
        physical_constants::freezing_temperature_k));
    u.fill(2.0f);
    v.fill(3.0f);
    w.fill(0.0f);

    const auto energy = compute_moist_mixture_energy_v1();
    REQUIRE(energy.resolved_kinetic
            == Approx(0.5 * 13.0 * analytic_domain_volume())
                   .epsilon(1e-12));
}

TEST_CASE("moist-mixture v2 subset is additive on the production C-grid basis",
          "[diagnostics][energy_budget][moist-v2][subset][domain_recycling]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    resize_moist_energy_fields(0.0f);
    rho.fill(1.25f);
    p.fill(static_cast<float>(p0));
    theta.fill(290.0f);
    u.fill(6.0f);
    v.fill(-4.0f);
    w.fill(1.0f);
    qv.fill(0.010f);
    qc.fill(0.002f);
    qr.fill(0.003f);
    qi.fill(0.004f);
    qs.fill(0.005f);
    qg.fill(0.006f);
    qh.fill(0.007f);
    tke.fill(0.2f);

    const auto full = compute_moist_mixture_energy_v2();
    const HorizontalEnergyBudgetExclusion retained{2, 6, 2, 5};
    const auto subset =
        compute_moist_mixture_energy_v2_excluding_horizontal_rectangle(
            global_simulation_state(), retained);
    tmv::diagnostics::set_deterministic_reductions(true);
    const auto deterministic =
        compute_moist_mixture_energy_v2_excluding_horizontal_rectangle(
            global_simulation_state(), retained);
    tmv::diagnostics::set_deterministic_reductions(false);

    const auto domain = physical_scalar_domain(
        NR, NTH, NZ, StaggerType::CGrid);
    const double fraction = 1.0 -
        static_cast<double>((retained.i_end - retained.i_begin) *
                            (retained.j_end - retained.j_begin)) /
        static_cast<double>(domain.nx() * domain.ny());
    REQUIRE(subset.dry_internal == Approx(full.dry_internal * fraction));
    REQUIRE(subset.water_sensible ==
            Approx(full.water_sensible * fraction));
    REQUIRE(subset.phase_reference ==
            Approx(full.phase_reference * fraction));
    REQUIRE(subset.resolved_kinetic ==
            Approx(full.resolved_kinetic * fraction));
    REQUIRE(subset.gravitational ==
            Approx(full.gravitational * fraction));
    REQUIRE(subset.subgrid_tke == Approx(full.subgrid_tke * fraction));
    REQUIRE(subset.total_water_mass ==
            Approx(full.total_water_mass * fraction));
    REQUIRE(subset.total == Approx(full.total * fraction));
    REQUIRE(deterministic.total == Approx(subset.total));
    REQUIRE_THROWS_WITH(
        compute_moist_mixture_energy_v2_excluding_horizontal_rectangle(
            global_simulation_state(), {2, 2, 1, 3}),
        Catch::Matchers::Contains("exclusion is invalid"));
}

TEST_CASE("moist-mixture energy v1 fails closed and is deterministic",
          "[diagnostics][energy_budget][moist-v1][fail-closed][sci140]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = false;
    resize_moist_energy_fields(0.0f);
    rho.fill(1.0f);
    p.fill(static_cast<float>(p0));
    theta.fill(300.0f);
    qv.fill(0.01f);
    tke.fill(0.25f);

    qi[1][1][1] = -1.0e-8f;
    REQUIRE_THROWS_WITH(
        compute_moist_mixture_energy_v1(),
        Catch::Matchers::Contains("nonphysical"));
    qi[1][1][1] = 0.0f;

    qh.resize(NR, NTH, NZ - 1, 0.0f);
    REQUIRE_THROWS_WITH(
        compute_moist_mixture_energy_v1(),
        Catch::Matchers::Contains("match the grid"));
    qh.resize(NR, NTH, NZ, 0.0f);

    tmv::diagnostics::set_deterministic_reductions(true);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto one_thread = compute_moist_mixture_energy_v1();
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto four_threads = compute_moist_mixture_energy_v1();
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    tmv::diagnostics::set_deterministic_reductions(false);
    REQUIRE(one_thread.dry_internal == four_threads.dry_internal);
    REQUIRE(one_thread.water_sensible == four_threads.water_sensible);
    REQUIRE(one_thread.phase_reference == four_threads.phase_reference);
    REQUIRE(one_thread.resolved_kinetic == four_threads.resolved_kinetic);
    REQUIRE(one_thread.gravitational == four_threads.gravitational);
    REQUIRE(one_thread.subgrid_tke == four_threads.subgrid_tke);
    REQUIRE(one_thread.total_water_mass == four_threads.total_water_mass);
    REQUIRE(one_thread.total == four_threads.total);
}

TEST_CASE("C-grid energy reconstructs face winds and excludes lateral scalar ghosts",
          "[diagnostics][energy_budget][cgrid][ghost]")
{
    DeterministicEnergyGuard reductions;
    StaggeredGeometryGuard guard(global_grid_geometry);
    resize_energy_fields(0.0f);
    p.fill(static_cast<float>(p0));
    theta.fill(300.0f);

    double expected_collocated_kinetic = 0.0;
    double expected_cgrid_kinetic = 0.0;
    const double dv = cell_volume();
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const float rho_cell = 1.0f + 0.125f * static_cast<float>(i)
                                     + 0.0625f * static_cast<float>(j)
                                     + 0.03125f * static_cast<float>(k);
                const float u_face = 10.0f * static_cast<float>(i)
                                   + 2.0f * static_cast<float>(j)
                                   + 0.5f * static_cast<float>(k) + 1.0f;
                const float v_face = 20.0f * static_cast<float>(j)
                                   + 3.0f * static_cast<float>(i)
                                   + 0.25f * static_cast<float>(k) + 2.0f;
                const float w_face = 30.0f * static_cast<float>(k)
                                   + 4.0f * static_cast<float>(i)
                                   + 0.75f * static_cast<float>(j) + 3.0f;
                rho[i][j][k] = rho_cell;
                u[i][j][k] = u_face;
                v[i][j][k] = v_face;
                w[i][j][k] = w_face;

                const double rho_dv = static_cast<double>(rho_cell) * dv;
                if (i >= 1 && i < NR - 1 && j >= 1 && j < NTH - 1
                    && k >= 1 && k < NZ - 1)
                {
                    expected_collocated_kinetic += 0.5 * rho_dv *
                        (static_cast<double>(u_face) * u_face
                       + static_cast<double>(v_face) * v_face
                       + static_cast<double>(w_face) * w_face);
                }

                if (i >= 1 && i < NR - 1 && j >= 1 && j < NTH - 1)
                {
                    const double u_center = 0.5 *
                        (static_cast<double>(u[i - 1][j][k]) + u_face);
                    const double v_center = 0.5 *
                        (static_cast<double>(v[i][j - 1][k]) + v_face);
                    const double w_center = k == 0
                        ? 0.5 * static_cast<double>(w_face)
                        : 0.5 * (static_cast<double>(w[i][j][k - 1]) + w_face);
                    expected_cgrid_kinetic += 0.5 * rho_dv *
                        (u_center * u_center + v_center * v_center + w_center * w_center);
                }
            }

    global_grid_geometry.staggered = false;
    const auto collocated_clean = compute_energy_budget();
    REQUIRE(collocated_clean.kinetic ==
            Approx(expected_collocated_kinetic).epsilon(1e-12));

    global_grid_geometry.staggered = true;
    const auto cgrid_clean = compute_energy_budget();
    REQUIRE(cgrid_clean.kinetic == Approx(expected_cgrid_kinetic).epsilon(1e-12));
    REQUIRE(cgrid_clean.kinetic != Approx(collocated_clean.kinetic).epsilon(1e-6));

    // Only scalar ghost values are poisoned. C-grid u[0]/v[0] are maintained
    // boundary faces used by the first physical cell and are intentionally not
    // ghost scalars.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            if (i != 0 && i != NR - 1 && j != 0 && j != NTH - 1)
                continue;
            for (int k = 0; k < NZ; ++k)
            {
                rho[i][j][k] = 50.0f;
                p[i][j][k] = static_cast<float>(p0);
                theta[i][j][k] = 900.0f;
                qv[i][j][k] = 0.5f;
            }
        }

    const auto cgrid_poisoned = compute_energy_budget();
    REQUIRE(cgrid_poisoned.kinetic == cgrid_clean.kinetic);
    REQUIRE(cgrid_poisoned.internal == cgrid_clean.internal);
    REQUIRE(cgrid_poisoned.potential == cgrid_clean.potential);
    REQUIRE(cgrid_poisoned.latent == cgrid_clean.latent);
    REQUIRE(cgrid_poisoned.total == cgrid_clean.total);

    global_grid_geometry.staggered = false;
    const auto collocated_poisoned = compute_energy_budget();
    REQUIRE(collocated_poisoned.kinetic == collocated_clean.kinetic);
    REQUIRE(collocated_poisoned.internal == collocated_clean.internal);
    REQUIRE(collocated_poisoned.potential == collocated_clean.potential);
    REQUIRE(collocated_poisoned.latent == collocated_clean.latent);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (const int k : {0, NZ - 1})
            {
                rho[i][j][k] = 1.0e6f;
                theta[i][j][k] = 1.0e6f;
                qv[i][j][k] = 100.0f;
            }
    const auto collocated_vertical_poison = compute_energy_budget();
    REQUIRE(collocated_vertical_poison.kinetic == collocated_clean.kinetic);
    REQUIRE(collocated_vertical_poison.internal == collocated_clean.internal);
    REQUIRE(collocated_vertical_poison.potential == collocated_clean.potential);
    REQUIRE(collocated_vertical_poison.latent == collocated_clean.latent);
}

TEST_CASE("potential energy = rho*g*integral(z dV)", "[diagnostics][energy_budget]")
{
    resize_energy_fields(0.0f);
    const float rho_val = 1.1f;
    rho.fill(rho_val);
    p.fill(static_cast<float>(p0));
    theta.fill(300.0f);

    const auto e = compute_energy_budget();
    const double expected = static_cast<double>(rho_val) * g * analytic_geopotential_volume();
    REQUIRE(e.potential == Approx(expected).epsilon(1e-5));
}

TEST_CASE("terrain energy uses physical height and exact layer volume",
          "[diagnostics][energy_budget][terrain][TE-005]")
{
    resize_energy_fields(0.0f);
    rho.fill(1.5f);
    p.fill(100000.0f);
    theta.fill(300.0f);

    TerrainRuntimeState terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    terrain.metrics.z.resize(NR, NTH, NZ, 0.0);
    terrain.metrics.z_face.resize(NR, NTH, NZ + 1, 0.0);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            double face = 100.0 + 20.0 * i + 10.0 * j;
            terrain.metrics.z_face(i, j, 0) = face;
            for (int k = 0; k < NZ; ++k)
            {
                const double thickness = 50.0 + i + 2.0 * k;
                terrain.metrics.z(i, j, k) = face + 0.5 * thickness;
                face += thickness;
                terrain.metrics.z_face(i, j, k + 1) = face;
            }
        }

    auto state = global_simulation_state();
    state.terrain_runtime = &terrain;
    const auto domain = physical_scalar_domain(state.grid);
    double expected_potential = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double thickness = 50.0 + i + 2.0 * k;
                const double volume = dr * dtheta * thickness;
                expected_potential += 1.5 * g
                    * terrain.metrics.z(i, j, k) * volume;
            }

    const auto energy = compute_energy_budget(state);
    REQUIRE(energy.potential == Approx(expected_potential));
    REQUIRE(energy.potential
            != Approx(1.5 * g * analytic_geopotential_volume()));
}

TEST_CASE("moist-mixture energy v1 uses terrain volume for water and gravity",
          "[diagnostics][energy_budget][moist-v1][terrain][sci140]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = false;
    resize_moist_energy_fields(0.0f);
    rho.fill(1.25f);
    p.fill(static_cast<float>(p0));
    theta.fill(static_cast<float>(
        physical_constants::freezing_temperature_k));
    qc.fill(0.125f);

    TerrainRuntimeState terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    terrain.metrics.z.resize(NR, NTH, NZ, 0.0);
    terrain.metrics.z_face.resize(NR, NTH, NZ + 1, 0.0);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            double face = 75.0 + 15.0 * i + 5.0 * j;
            terrain.metrics.z_face(i, j, 0) = face;
            for (int k = 0; k < NZ; ++k)
            {
                const double thickness = 40.0 + 2.0 * i + 3.0 * k;
                terrain.metrics.z(i, j, k) = face + 0.5 * thickness;
                face += thickness;
                terrain.metrics.z_face(i, j, k + 1) = face;
            }
        }

    auto state = global_simulation_state();
    state.terrain_runtime = &terrain;
    const auto domain = physical_scalar_domain(state.grid);
    double expected_water_mass = 0.0;
    double expected_gravitational = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double volume = dr * dtheta
                    * (40.0 + 2.0 * i + 3.0 * k);
                const double dry_mass = 1.25 * volume;
                expected_water_mass += dry_mass * 0.125;
                expected_gravitational += dry_mass * 1.125
                    * physical_constants::gravity_ms2
                    * terrain.metrics.z(i, j, k);
            }

    const auto energy = compute_moist_mixture_energy_v1(state);
    REQUIRE(energy.total_water_mass == Approx(expected_water_mass));
    REQUIRE(energy.gravitational == Approx(expected_gravitational));
}

TEST_CASE("SCI-144 moist-mixture energy v2 includes physical terrain ground w",
          "[diagnostics][energy_budget][moist-v2][terrain][SCI-144]")
{
    DeterministicEnergyGuard reductions;
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    resize_moist_energy_fields(0.0f);
    rho.fill(1.1f);
    p.fill(static_cast<float>(p0));
    theta.fill(static_cast<float>(
        physical_constants::freezing_temperature_k));
    u.fill(6.0f);
    v.fill(4.0f);
    w.fill(0.0f);

    TerrainConfig cfg;
    cfg.coord_id = "smoothed";
    cfg.ztop = static_cast<double>(NZ) * dz;
    cfg.smoothing_decay = 0.7;
    Topography2D topo;
    topography::initialize_topography(topo, NR, NTH);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            topo.h[i][j] = 100.0 + 10.0 * i + 5.0 * j;

    TerrainRuntimeState terrain;
    terrain.config.scheme_id = "synthetic_terrain";
    topography::initialize_metrics(terrain.metrics, NR, NTH, NZ);
    topography::build_conservative_face_metrics(
        cfg, topo, topography::build_zeta_interfaces(NZ, cfg.ztop),
        dr, dtheta, terrain.metrics);

    auto state = global_simulation_state();
    state.terrain_runtime = &terrain;
    const bool earth_fixed = GENERATE(false, true);
    terrain.config.horizontal_placement_id = earth_fixed ? "earth_fixed" : "grid_attached";
    tmv::domain::HorizontalDomainFrame frame;
    frame.velocity_ms = {3.0, -2.0};
    state.domain_frame = &frame;
    const auto v1 = compute_moist_mixture_energy_v1(state);
    const auto v2 = compute_moist_mixture_energy_v2(state);
    const auto domain = physical_scalar_domain(state.grid);
    double expected_ground_kinetic = 0.0;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const double ground_u = 0.5 * (
                static_cast<double>(u[i - 1][j][0])
                + static_cast<double>(u[i][j][0]));
            const double ground_v = 0.5 * (
                static_cast<double>(v[i][j - 1][0])
                + static_cast<double>(v[i][j][0]));
            const double ground_w =
                (ground_u - (earth_fixed ? 0.0 : 3.0)) * (-terrain.metrics.Gx_face(i, j, 0))
                + (ground_v - (earth_fixed ? 0.0 : -2.0)) * (-terrain.metrics.Gy_face(i, j, 0));
            const double volume = dr * dtheta
                * (terrain.metrics.z_face(i, j, 1)
                   - terrain.metrics.z_face(i, j, 0));
            expected_ground_kinetic +=
                0.25 * static_cast<double>(rho[i][j][0])
                * volume * ground_w * ground_w;
        }

    REQUIRE(expected_ground_kinetic > 0.0);
    REQUIRE(v2.dry_internal == v1.dry_internal);
    REQUIRE(v2.water_sensible == v1.water_sensible);
    REQUIRE(v2.phase_reference == v1.phase_reference);
    REQUIRE(v2.gravitational == v1.gravitational);
    REQUIRE(v2.subgrid_tke == v1.subgrid_tke);
    REQUIRE(v2.total_water_mass == v1.total_water_mass);
    REQUIRE(v2.resolved_kinetic - v1.resolved_kinetic
            == Approx(expected_ground_kinetic).epsilon(2.0e-12));
    REQUIRE(v2.total - v1.total
            == Approx(expected_ground_kinetic).epsilon(2.0e-12));
}

TEST_CASE("latent energy = rho*L_v*qv*volume", "[diagnostics][energy_budget]")
{
    resize_energy_fields(0.0f);
    const float rho_val = 1.0f;
    const float qv_val = 0.0078125f;  // 1/128, exact in float32
    rho.fill(rho_val);
    p.fill(static_cast<float>(p0));
    theta.fill(300.0f);
    qv.fill(qv_val);

    const auto e = compute_energy_budget();
    const double expected = static_cast<double>(rho_val) * static_cast<double>(L_v) *
                            static_cast<double>(qv_val) * analytic_domain_volume();
    REQUIRE(e.latent == Approx(expected).epsilon(1e-5));
}

TEST_CASE("SCI-024 water loading does not reinterpret prognostic dry density",
          "[diagnostics][energy_budget][SCI-024]")
{
    DeterministicEnergyGuard deterministic;
    resize_energy_fields(0.0f);
    rho.fill(1.125f);
    p.fill(static_cast<float>(p0));
    theta.fill(302.0f);
    u.fill(3.0f);
    v.fill(4.0f);
    w.fill(0.0f);

    qv.fill(0.0f);
    const auto dry = compute_energy_budget();

    qv.fill(0.015625f);  // 1/64 kg_water/kg_dry, exact in float32
    const auto moist = compute_energy_budget();

    // rho remains rho_d: adding a dry-air mixing ratio changes only the
    // explicitly water-weighted latent reservoir in this diagnostic proxy.
    REQUIRE(moist.kinetic == dry.kinetic);
    REQUIRE(moist.internal == dry.internal);
    REQUIRE(moist.potential == dry.potential);
    REQUIRE(moist.latent > dry.latent);
    REQUIRE(moist.total - dry.total == Approx(moist.latent).epsilon(1e-12));
}

TEST_CASE("total energy equals the sum of its reservoirs",
          "[diagnostics][energy_budget]")
{
    resize_energy_fields(0.0f);
    rho.fill(1.0f);
    p.fill(static_cast<float>(p0));
    theta.fill(305.0f);
    u.fill(4.0f);
    v.fill(0.0f);
    w.fill(1.0f);
    qv.fill(0.01f);

    const auto e = compute_energy_budget();
    REQUIRE(e.total == Approx(e.kinetic + e.internal + e.potential + e.latent).epsilon(1e-12));
}

TEST_CASE("hydrostatic rest carries zero kinetic energy",
          "[diagnostics][energy_budget]")
{
    resize_energy_fields(0.0f);
    rho.fill(1.0f);
    p.fill(static_cast<float>(p0));
    theta.fill(300.0f);  // u = v = w = 0
    const auto e = compute_energy_budget();
    REQUIRE(e.kinetic == 0.0);
    REQUIRE(e.internal > 0.0);   // internal/potential reservoirs are non-trivial
    REQUIRE(e.potential > 0.0);
}

TEST_CASE("energy drift warns only on the total reservoir",
          "[diagnostics][energy_budget]")
{
    EnergyBudget before{};
    before.kinetic = 1.0e12;
    before.internal = 5.0e16;
    before.potential = 2.0e15;
    before.latent = 3.0e14;
    before.total = before.kinetic + before.internal + before.potential + before.latent;

    SECTION("adiabatic exchange (KE<->internal) at fixed total does not warn")
    {
        EnergyBudget after = before;
        const double shift = before.kinetic * 0.5;  // large component swing
        after.kinetic -= shift;
        after.internal += shift;
        // total unchanged
        const auto d = evaluate_energy_drift(before, after);
        REQUIRE(d.warn == false);
        REQUIRE(d.rel_total <= BUDGET_THRESH_ENERGY_TOTAL);
    }
    SECTION("total drift above threshold warns")
    {
        EnergyBudget after = before;
        after.total *= 1.0 + BUDGET_THRESH_ENERGY_TOTAL * 10.0;
        const auto d = evaluate_energy_drift(before, after);
        REQUIRE(d.warn == true);
        REQUIRE(d.rel_total > BUDGET_THRESH_ENERGY_TOTAL);
    }
}

TEST_CASE("deterministic energy reductions are thread-count independent",
          "[diagnostics][energy_budget][deterministic]")
{
    resize_energy_fields(0.0f);
    rho.fill(1.0f);
    p.fill(95000.0f);
    theta.fill(301.0f);
    u.fill(5.0f);
    w.fill(2.0f);
    qv.fill(0.012f);
    tmv::diagnostics::set_deterministic_reductions(true);
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto a = compute_energy_budget();
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto b = compute_energy_budget();
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
    tmv::diagnostics::set_deterministic_reductions(false);
    REQUIRE(a.kinetic == b.kinetic);
    REQUIRE(a.internal == b.internal);
    REQUIRE(a.potential == b.potential);
    REQUIRE(a.latent == b.latent);
    REQUIRE(a.total == b.total);
}

TEST_CASE("Moist energy magnitudes bound cancelling warm and cold reservoirs",
          "[diagnostics][energy_budget][moist-v2][subset][roundoff]")
{
    SimulationStorageState storage({8, 8, 4, 1000, 500, .5, 800});
    storage.grid.geometry.initialize(8, 8, 4, 1000, 500, 800, CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& fields = storage.fields;
    for (auto* field : {&fields.rho,
                        &fields.p,
                        &fields.theta,
                        &fields.u,
                        &fields.v,
                        &fields.w,
                        &fields.qv,
                        &fields.qc,
                        &fields.qr,
                        &fields.qi,
                        &fields.qs,
                        &fields.qg,
                        &fields.qh,
                        &fields.tke})
        field->resize(8, 8, 4, 0);
    fields.rho.fill(1);
    fields.p.fill(100000);
    fields.u.fill(2);
    fields.v.fill(3);
    fields.qv.fill(.01f);
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 4; ++k)
                fields.theta(i, j, k) = i % 2 ? 250 : 300;
    const auto       view   = make_simulation_state_view(storage);
    constexpr double volume = 1000.0 * 800 * 500;
    const double     cv     = physical_constants::specific_heat_cp_jkgk - physical_constants::gas_constant_dry_air_jkgk;
    const auto       signed_energy = compute_moist_mixture_energy_v2_deterministic(view);
    const auto       magnitude     = moist_mixture_energy_v2_magnitudes(view);
    CHECK(magnitude[0] == Approx(72 * volume * cv * 50).epsilon(1e-12));
    CHECK(magnitude[0] > std::abs(signed_energy.dry_internal));
    CHECK(magnitude[3] == signed_energy.resolved_kinetic);
    CHECK(magnitude[6] >= std::abs(signed_energy.total));
    const HorizontalEnergyBudgetExclusion retained{2, 4, 2, 4};
    const auto                            subset = moist_mixture_energy_v2_magnitudes(view, &retained);
    CHECK(subset[0] == Approx(64 * volume * cv * 50).epsilon(1e-12));
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    const auto one = compute_moist_mixture_energy_v2_excluding_horizontal_rectangle_deterministic(view, retained);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto four = compute_moist_mixture_energy_v2_excluding_horizontal_rectangle_deterministic(view, retained);
    CHECK(one.total == four.total);
    CHECK(subset == moist_mixture_energy_v2_magnitudes(view, &retained));
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
}

TEST_CASE("Flat velocity energy pairs match both complete endpoint inventories exactly",
          "[diagnostics][energy][velocity-pair][exact][thread]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    global_grid_geometry.coord = CoordinateSystem::Cartesian;
    resize_moist_energy_fields();
    Field3D prior_u(NR, NTH, NZ), prior_v(NR, NTH, NZ), prior_w(NR, NTH, NZ);
    std::uint64_t bits = 918273;
    const auto random = [&]() {
        bits = bits * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((bits >> 40) / 16777216.0);
    };
#ifdef _OPENMP
    const int saved_workers = omp_get_max_threads();
    struct Restore { int workers; ~Restore() { omp_set_num_threads(workers); } } restore{saved_workers};
    omp_set_num_threads(GENERATE(1, 4));
#endif
    for (int iteration = 0; iteration < 128; ++iteration)
    {
        for (std::size_t n = 0; n < rho.size(); ++n)
        {
            rho.data()[n] = 0.01f + 2.0f * random();
            p.data()[n] = 15000.0f + 90000.0f * random();
            theta.data()[n] = 230.0f + 450.0f * random();
            for (auto* field : {&qv,&qc,&qr,&qi,&qs,&qg,&qh}) field->data()[n] = 0.02f * random();
            tke.data()[n] = 10.0f * random();
            for (auto* field : {&u,&v,&w,&prior_u,&prior_v,&prior_w}) field->data()[n] = 200.0f * (random() - 0.5f);
        }
        const std::array<const Field3D*,17> inputs{&rho,&p,&theta,&u,&v,&w,&qv,&qc,&qr,&qi,&qs,&qg,&qh,&tke,&prior_u,&prior_v,&prior_w};
        std::array<Field3D,17> preserved;
        for (std::size_t n=0;n<inputs.size();++n) preserved[n]=*inputs[n];
        const auto current = compute_energy_diagnostic_snapshot_v1_deterministic(global_simulation_state());
        std::swap(u, prior_u); std::swap(v, prior_v); std::swap(w, prior_w);
        const auto prior = compute_energy_diagnostic_snapshot_v1_deterministic(global_simulation_state());
        std::swap(u, prior_u); std::swap(v, prior_v); std::swap(w, prior_w);
        const auto result = compute_flat_cgrid_velocity_energy_pair_deterministic(global_simulation_state(), prior_u, prior_v, prior_w);
        CAPTURE(iteration);
        REQUIRE(result[0] == prior.moist_mixture_v2.resolved_kinetic);
        REQUIRE(result[1] == current.moist_mixture_v2.resolved_kinetic);
        REQUIRE(compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(global_simulation_state(), prior)
                == current.moist_mixture_v2.resolved_kinetic);
        for (std::size_t n=0;n<inputs.size();++n)
            REQUIRE(std::equal(inputs[n]->data(),inputs[n]->data()+inputs[n]->size(),preserved[n].data()));
    }
}

TEST_CASE("Velocity energy pair rejects invalid shared fields, endpoints and geometry atomically",
          "[diagnostics][energy][velocity-pair][invalid][atomic]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    resize_moist_energy_fields();
    rho.fill(1); p.fill(100000); theta.fill(300);
    u.fill(2); v.fill(3); w.fill(4);
    Field3D prior_u(NR,NTH,NZ,1), prior_v(NR,NTH,NZ,2), prior_w(NR,NTH,NZ,3);
    const auto evaluate = [&] { return compute_flat_cgrid_velocity_energy_pair_deterministic(global_simulation_state(),prior_u,prior_v,prior_w); };
    REQUIRE_NOTHROW(evaluate());
    for (auto* field : {&rho,&p,&theta,&qv,&qc,&qr,&qi,&qs,&qg,&qh,&tke,&u,&v,&w,&prior_u,&prior_v,&prior_w})
    {
        const float saved = (*field)(1,1,1);
        (*field)(1,1,1) = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS(evaluate());
        REQUIRE(std::isnan((*field)(1,1,1)));
        (*field)(1,1,1) = saved;
    }
    for (auto* field : {&rho,&p,&theta,&qv,&qc,&qr,&qi,&qs,&qg,&qh,&tke})
    {
        const float saved = (*field)(1,1,1);
        (*field)(1,1,1) = -1;
        REQUIRE_THROWS(evaluate());
        REQUIRE((*field)(1,1,1) == -1);
        (*field)(1,1,1) = saved;
    }
    u(0,1,1) = std::numeric_limits<float>::infinity();
    REQUIRE_THROWS(evaluate());
    u(0,1,1) = 2;
    prior_v(1,0,1) = std::numeric_limits<float>::infinity();
    REQUIRE_THROWS(evaluate());
    prior_v(1,0,1) = 2;
    prior_w.resize(NR,NTH,NZ-1);
    REQUIRE_THROWS_AS(evaluate(),std::invalid_argument);
    prior_w.resize(NR,NTH,NZ,3);
    global_grid_geometry.staggered = false;
    REQUIRE_THROWS_AS(evaluate(),std::invalid_argument);
    global_grid_geometry.staggered = true;
    const double saved_dr = dr;
    dr = -1;
    REQUIRE_THROWS_AS(evaluate(),std::invalid_argument);
    dr = saved_dr;
    REQUIRE(evaluate()[1] == compute_moist_mixture_energy_v2_deterministic(global_simulation_state()).resolved_kinetic);
}

TEST_CASE("Velocity energy pair retains complete inventory overflow rejection",
          "[diagnostics][energy][velocity-pair][overflow]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    struct RestoreSpacing
    {
        double x = dr, y = dtheta, z = dz;
        GridGeometry geometry = global_grid_geometry;
        ~RestoreSpacing() { dr=x;dtheta=y;dz=z;global_grid_geometry=std::move(geometry); }
    } restore;
    dr = 1e150; dtheta = 1e150; dz = 1;
    global_grid_geometry.initialize(NR,NTH,NZ,dr,dz,dtheta,CoordinateSystem::Cartesian,StaggerType::CGrid);
    resize_moist_energy_fields();
    rho.fill(1); p.fill(100000); theta.fill(300);
    u.fill(0); v.fill(0); w.fill(0);
    Field3D prior_u(NR,NTH,NZ,0), prior_v(NR,NTH,NZ,0), prior_w(NR,NTH,NZ,0);
    REQUIRE_NOTHROW(compute_moist_mixture_energy_v2_deterministic(global_simulation_state()));
    u.fill(550); v.fill(550); w.fill(550);
    REQUIRE_THROWS_WITH(compute_moist_mixture_energy_v2_deterministic(global_simulation_state()),
                        Catch::Matchers::Contains("legacy energy proxy"));
    REQUIRE_THROWS_AS(compute_flat_cgrid_velocity_energy_pair_deterministic(global_simulation_state(),prior_u,prior_v,prior_w),std::overflow_error);
    dr = 1e120; dtheta = 1e120;
    u.fill(std::numeric_limits<float>::max());
    REQUIRE_THROWS_AS(compute_moist_mixture_energy_v2_deterministic(global_simulation_state()),std::overflow_error);
    REQUIRE_THROWS_AS(compute_flat_cgrid_velocity_energy_pair_deterministic(global_simulation_state(),prior_u,prior_v,prior_w),std::overflow_error);
}

TEST_CASE("Velocity snapshot completion preserves invalid endpoint and overflow rejection",
          "[diagnostics][energy][velocity-pair][snapshot-completion][atomic]")
{
    StaggeredGeometryGuard guard(global_grid_geometry);
    global_grid_geometry.staggered = true;
    global_grid_geometry.coord = CoordinateSystem::Cartesian;
    resize_moist_energy_fields();
    rho.fill(1); p.fill(100000); theta.fill(300);
    u.fill(2); v.fill(3); w.fill(4);
    const auto before = compute_energy_diagnostic_snapshot_v1_deterministic(global_simulation_state());
    for (auto* field : {&u, &v, &w})
    {
        const float original = (*field)(1, 1, 1);
        (*field)(1, 1, 1) = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS_AS(compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(global_simulation_state(), before), std::runtime_error);
        REQUIRE(std::isnan((*field)(1, 1, 1)));
        (*field)(1, 1, 1) = original;
    }
    u(0, 1, 1) = std::numeric_limits<float>::infinity();
    REQUIRE_THROWS_AS(compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(global_simulation_state(), before), std::runtime_error);
    u(0, 1, 1) = 2;
    const Field3D saved = u;
    u.resize(NR, NTH, NZ - 1);
    REQUIRE_THROWS_AS(compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(global_simulation_state(), before), std::invalid_argument);
    u = saved;
    REQUIRE(compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(global_simulation_state(), before)
            == compute_moist_mixture_energy_v2_deterministic(global_simulation_state()).resolved_kinetic);
    struct RestoreSpacing
    {
        double x = dr, y = dtheta, z = dz;
        GridGeometry geometry = global_grid_geometry;
        ~RestoreSpacing() { dr = x; dtheta = y; dz = z; global_grid_geometry = std::move(geometry); }
    } restore;
    dr = 1e150; dtheta = 1e150; dz = 1;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian, StaggerType::CGrid);
    u.fill(0); v.fill(0); w.fill(0);
    const auto huge_before = compute_energy_diagnostic_snapshot_v1_deterministic(global_simulation_state());
    u.fill(550); v.fill(550); w.fill(550);
    REQUIRE_THROWS_WITH(compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(global_simulation_state(), huge_before),
                       Catch::Matchers::Contains("legacy energy proxy"));
}

TEST_CASE("Pressure material reuse agrees with complete density and pressure inventories",
          "[diagnostics][energy][pressure-material][reference][thread]")
{
#ifdef _OPENMP
    struct Restore { int n = omp_get_max_threads(); ~Restore() { omp_set_num_threads(n); } } restore;
    omp_set_num_threads(GENERATE(1, 4));
#endif
    SimulationStorageState storage({7, 9, 5, 73, 41, .1, 89});
    storage.grid.geometry.initialize(7, 9, 5, 73, 41, 89, CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (auto* field : {&f.rho,&f.p,&f.theta,&f.u,&f.v,&f.w,&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh,&f.tke})
        field->resize(7, 9, 5, 0);
    const auto state = make_simulation_state_view(storage);
    const auto domain = physical_scalar_domain(state.grid);
    std::uint64_t bits = 716389;
    const auto random = [&]() {
        bits = bits * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((bits >> 40) / 16777216.0);
    };
    const auto inventory = [](const MoistMixtureEnergyBudgetV2& b) {
        return std::array<double, 6>{b.dry_internal,b.water_sensible,b.phase_reference,
                                    b.resolved_kinetic,b.gravitational,b.subgrid_tke};
    };
    std::vector<tmv::pressure_energy_detail::CellMaterial> material(f.rho.size());
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        for (std::size_t n = 0; n < f.rho.size(); ++n)
        {
            f.rho.data()[n] = .05f + 1.5f * random();
            f.p.data()[n] = 15000 + 90000 * random();
            f.theta.data()[n] = 280 + 100 * random();
            for (auto* field : {&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh}) field->data()[n] = .01f * random();
            for (auto* field : {&f.u,&f.v,&f.w}) field->data()[n] = 100 * (random() - .5f);
            f.tke.data()[n] = 5 * random();
        }
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    material[(i * state.grid.nth + j) * state.grid.nz + k] = tmv::pressure_energy_detail::prepare(state, i, j, k);
        for (int endpoint = 0; endpoint < 2; ++endpoint)
        {
            const auto expected = inventory(compute_moist_mixture_energy_v2_deterministic(state));
            const auto actual = tmv::diagnostics::fixed_slab_reduce<6>(domain.i_begin, domain.i_end,
                [&](int i, std::array<double, 6>& partial) {
                    for (int j = domain.j_begin; j < domain.j_end; ++j)
                        for (int k = domain.k_begin; k < domain.k_end; ++k)
                        {
                            const auto n = (i * state.grid.nth + j) * state.grid.nz + k;
                            const auto cell = tmv::pressure_energy_detail::reservoirs(material[n], f.rho.data()[n], f.p.data()[n]);
                            for (unsigned r = 0; r < 6; ++r) partial[r] += cell[r];
                        }
                });
            CAPTURE(iteration, endpoint);
            for (unsigned r = 2; r < 6; ++r) REQUIRE(actual[r] == expected[r]);
            std::array<long double, 2> thermal_scale{};
            for (int i = domain.i_begin; i < domain.i_end; ++i)
                for (int j = domain.j_begin; j < domain.j_end; ++j)
                    for (int k = domain.k_begin; k < domain.k_end; ++k)
                    {
                        const long double mass = static_cast<long double>(f.rho[i][j][k]) * 73 * 89 * 41;
                        const long double temperature = f.theta[i][j][k] * std::pow(
                            static_cast<long double>(f.p[i][j][k]) / physical_constants::reference_pressure_pa,
                            static_cast<long double>(physical_constants::gas_constant_dry_air_jkgk) / physical_constants::specific_heat_cp_jkgk);
                        const long double magnitude = mass * (temperature + physical_constants::freezing_temperature_k);
                        thermal_scale[0] += magnitude * (physical_constants::specific_heat_cp_jkgk - physical_constants::gas_constant_dry_air_jkgk);
                        thermal_scale[1] += magnitude * (static_cast<long double>(f.qv[i][j][k]) *
                            (physical_constants::specific_heat_water_vapor_cp_jkgk - physical_constants::gas_constant_water_vapor_jkgk)
                            + (static_cast<long double>(f.qc[i][j][k]) + f.qr[i][j][k]) * physical_constants::specific_heat_liquid_water_jkgk
                            + (static_cast<long double>(f.qi[i][j][k]) + f.qs[i][j][k] + f.qg[i][j][k] + f.qh[i][j][k]) * physical_constants::specific_heat_ice_jkgk);
                    }
            // Pressure attribution may fuse theta*Exner-Tref; the full diagnostic observes temperature first.
            const double roundoff = (8 + 2 * (domain.ny() * domain.nz() + domain.nx())) * std::numeric_limits<double>::epsilon();
            for (unsigned r = 0; r < 2; ++r)
                REQUIRE(std::abs(static_cast<long double>(actual[r]) - expected[r]) <= roundoff * thermal_scale[r]);
            for (std::size_t n = 0; n < f.rho.size(); ++n)
            {
                f.rho.data()[n] = .05f + 1.5f * random();
                f.p.data()[n] = 15000 + 90000 * random();
            }
        }
    }
}

TEST_CASE("Pressure material reuse exposes invalid and overflowing output inventories",
          "[diagnostics][energy][pressure-material][invalid]")
{
    const tmv::pressure_energy_detail::CellMaterial material{300, 1.02, 20, 10000, 100, 2, 1e6, 100};
    const auto saved = material;
    for (const double density : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 1e308})
    {
        const auto result = tmv::pressure_energy_detail::reservoirs(material, density, 100000);
        REQUIRE(std::any_of(result.begin(), result.end(), [](double value) { return !std::isfinite(value); }));
    }
    for (const double pressure : {-100000.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        const auto result = tmv::pressure_energy_detail::reservoirs(material, 1, pressure);
        REQUIRE_FALSE(std::isfinite(result[0]));
        REQUIRE_FALSE(std::isfinite(result[1]));
    }
    REQUIRE(std::memcmp(&material, &saved, sizeof(material)) == 0);
    const auto valid = tmv::pressure_energy_detail::reservoirs(material, 1, 100000);
    REQUIRE(std::all_of(valid.begin(), valid.end(), [](double value) { return std::isfinite(value); }));
}

TEST_CASE("Velocity energy pair operator benchmark", "[.energy-pair-benchmark]")
{
#ifdef _OPENMP
    const int workers = std::getenv("NIMBUS_ENERGY_BENCHMARK_THREADS")
        ? std::stoi(std::getenv("NIMBUS_ENERGY_BENCHMARK_THREADS")) : 1;
    REQUIRE(workers >= 1);
    struct Restore { int n=omp_get_max_threads(); ~Restore(){omp_set_num_threads(n);} } restore;
    omp_set_num_threads(workers);
#endif
    SimulationStorageState storage({128,128,180,50,100,.1,50});
    storage.grid.geometry.initialize(128,128,180,50,100,50,CoordinateSystem::Cartesian,StaggerType::CGrid);
    auto& f=storage.fields;
    for (auto* field : {&f.rho,&f.p,&f.theta,&f.u,&f.v,&f.w,&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh,&f.tke})
        field->resize(128,128,180,0);
    Field3D old_u(128,128,180),old_v(128,128,180),old_w(128,128,180);
    for (std::size_t n=0;n<f.rho.size();++n)
    {
        const float x=static_cast<float>(n%997)/997;
        f.rho.data()[n]=.2f+x;f.p.data()[n]=20000+70000*x;f.theta.data()[n]=280+50*x;
        f.qv.data()[n]=.015f*x;f.qc.data()[n]=.001f*x;f.qi.data()[n]=.0005f*x;f.tke.data()[n]=x;
        f.u.data()[n]=10+20*x;f.v.data()[n]=15*x;f.w.data()[n]=5*x;
        old_u.data()[n]=9+21*x;old_v.data()[n]=14*x;old_w.data()[n]=4*x;
    }
    const auto state=make_simulation_state_view(storage);
    const auto reference=[&] {
        std::swap(f.u,old_u);std::swap(f.v,old_v);std::swap(f.w,old_w);
        const double a=compute_moist_mixture_energy_v2_deterministic(state).resolved_kinetic;
        std::swap(f.u,old_u);std::swap(f.v,old_v);std::swap(f.w,old_w);
        return std::array<double,2>{a,compute_moist_mixture_energy_v2_deterministic(state).resolved_kinetic};
    };
    const bool split_snapshot = std::getenv("NIMBUS_ENERGY_BENCHMARK_SPLIT_SNAPSHOT") != nullptr &&
        std::string(std::getenv("NIMBUS_ENERGY_BENCHMARK_SPLIT_SNAPSHOT")) == "1";
    const auto candidate=[&] {
        if (!split_snapshot) return compute_flat_cgrid_velocity_energy_pair_deterministic(state,old_u,old_v,old_w);
        std::swap(f.u,old_u);std::swap(f.v,old_v);std::swap(f.w,old_w);
        const auto before = compute_energy_diagnostic_snapshot_v1_deterministic(state);
        std::swap(f.u,old_u);std::swap(f.v,old_v);std::swap(f.w,old_w);
        return std::array<double,2>{before.moist_mixture_v2.resolved_kinetic,
            compute_flat_cgrid_velocity_energy_after_snapshot_deterministic(state,before)};
    };
    const auto expected=reference();REQUIRE(candidate()==expected);
    for(int warmup=0;warmup<3;++warmup){REQUIRE(reference()==expected);REQUIRE(candidate()==expected);}
    std::vector<double> before,after,delta;
    for(int trial=0;trial<9;++trial)
    {
        double times[2]{};
        for(int order=0;order<2;++order)
        {
            const int variant=trial%2?1-order:order;
            std::array<double,2> result;
            const auto start=std::chrono::steady_clock::now();
            for(int repeat=0;repeat<4;++repeat)result=variant?candidate():reference();
            times[variant]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/4;
            REQUIRE(result==expected);
        }
        before.push_back(times[0]);after.push_back(times[1]);delta.push_back(times[0]-times[1]);
    }
    const auto median=[](std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    const double center=median(delta);std::vector<double> deviations;
    for(double d:delta)deviations.push_back(std::abs(d-center));
    const double noise=3*1.4826*median(deviations);
    std::cout<<std::setprecision(17)<<"[ENERGY PAIR BENCHMARK] {\"split_snapshot\":"<<(split_snapshot?"true":"false")<<",\"baseline_ms\":"<<median(before)
             <<",\"candidate_ms\":"<<median(after)<<",\"paired_delta_ms\":"<<center<<",\"three_mad_ms\":"<<noise
             <<",\"paired_improvement_percent\":"<<100*center/median(before)<<",\"baseline_trials\":[";
    for(std::size_t i=0;i<before.size();++i)std::cout<<(i?",":"")<<before[i];
    std::cout<<"],\"candidate_trials\":[";
    for(std::size_t i=0;i<after.size();++i)std::cout<<(i?",":"")<<after[i];
    std::cout<<"]}\n";
}

TEST_CASE("Velocity energy pair retains the full path for extreme finite geometry",
          "[diagnostics][energy][velocity-pair][geometry]")
{
    for (const int geometry_case : {0,1,2})
    {
        const bool large_height=geometry_case==2;
        const double dx=large_height?1e30:geometry_case==1?1e51:123.45;
        const double dy=dx;
        const double dz=large_height?1e36:1;
        SimulationStorageState storage({5,7,4,dx,dz,.1,dy});
        storage.grid.geometry.initialize(5,7,4,dx,dz,dy,CoordinateSystem::Cartesian,StaggerType::CGrid);
        if (geometry_case==0)
        {
            auto& g=storage.grid.geometry;
            for(std::size_t k=0;k<g.z_faces.size();++k)g.z_faces[k]=3.3*k+0.7*k*k;
            for(std::size_t k=0;k<g.z.size();++k)g.z[k]=0.5*(g.z_faces[k]+g.z_faces[k+1]);
        }
        if (large_height)
        {
            for (auto& z:storage.grid.geometry.z)z+=2e50;
            for (auto& z:storage.grid.geometry.z_faces)z+=2e50;
        }
        auto& f=storage.fields;
        for (auto* field:{&f.rho,&f.p,&f.theta,&f.u,&f.v,&f.w,&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh,&f.tke})field->resize(5,7,4,0);
        f.rho.fill(1.2f);f.p.fill(95000);f.theta.fill(305);f.qv.fill(.01f);f.u.fill(8.321f);f.v.fill(2);f.w.fill(7);
        Field3D old_u(5,7,4,3.14159f),old_v(5,7,4,1.25f),old_w(5,7,4,3);
        const auto view=make_simulation_state_view(storage);
        const auto current=compute_moist_mixture_energy_v2_deterministic(view);
        std::swap(f.u,old_u);std::swap(f.v,old_v);std::swap(f.w,old_w);
        const auto prior=compute_moist_mixture_energy_v2_deterministic(view);
        std::swap(f.u,old_u);std::swap(f.v,old_v);std::swap(f.w,old_w);
        const auto pair=compute_flat_cgrid_velocity_energy_pair_deterministic(view,old_u,old_v,old_w);
        REQUIRE(pair[0]==prior.resolved_kinetic);
        REQUIRE(pair[1]==current.resolved_kinetic);
    }
}

TEST_CASE("Velocity pair finite envelope dominates binary32 endpoint extremes",
          "[diagnostics][energy][velocity-pair][envelope]")
{
    using namespace physical_constants;
    const long double f=std::numeric_limits<float>::max();
    const long double count=std::uint64_t{1}<<31;
    const long double volume=1e100L,height=1e50L;
    REQUIRE(reference_pressure_pa>=1);
    REQUIRE(gas_constant_dry_air_jkgk>0);
    REQUIRE(gas_constant_dry_air_jkgk/specific_heat_cp_jkgk<=1);
    const long double dry=f*volume, mass=dry*(1+7*f);
    const long double temperature=f*(f/reference_pressure_pa);
    const long double delta=temperature+freezing_temperature_k;
    const long double cv=specific_heat_cp_jkgk-gas_constant_dry_air_jkgk;
    const long double vapor_cv=specific_heat_water_vapor_cp_jkgk-gas_constant_water_vapor_jkgk;
    const long double vapor_reference=std::abs(latent_heat_vaporization_jkg-gas_constant_water_vapor_jkgk*freezing_temperature_k);
    const long double common=dry*cv*delta + dry*delta*(f*vapor_cv+2*f*specific_heat_liquid_water_jkgk+4*f*specific_heat_ice_jkgk)
        + dry*(f*vapor_reference+4*f*latent_heat_fusion_jkg) + mass*gravity_ms2*height + dry*f + dry*7*f;
    const long double kinetic=.25L*mass*(6*f*f);
    const long double legacy=.5L*dry*(3*f*f)+dry*cv*temperature+dry*gravity_ms2*height+dry*latent_heat_vaporization_jkg*f;
    // A slab has at most 2^31 additions: its IEEE binary64 accumulation
    // error is <2^-20; this margin also covers the compensated merge.
    const long double upper=16*count*(common+kinetic+legacy);
    REQUIRE(std::isfinite(upper));
    REQUIRE(upper<std::ldexp(1.0L,900));
}
