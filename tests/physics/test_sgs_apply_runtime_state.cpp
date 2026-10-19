// Explicit-state SGS tendency regressions.

#include "catch2/catch.hpp"
#include "compute/parity_check.hpp"
#include "core/orchestration/dynamics/sgs_apply.hpp"
#include "core/orchestration/dynamics/sgs_apply_internal.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "turbulence/base/eddy_viscosity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

SimulationStorageState global_simulation_storage{{2, 2, 2,
                                                  100.0, 50.0, 1.0, 100.0}};
int& NR = global_simulation_storage.grid.nr;
int& NTH = global_simulation_storage.grid.nth;
int& NZ = global_simulation_storage.grid.nz;
double& dr = global_simulation_storage.grid.dx;
double& dz = global_simulation_storage.grid.dz;
double& dt = global_simulation_storage.grid.dt;
double& dtheta = global_simulation_storage.grid.dy;
GridGeometry& global_grid_geometry = global_simulation_storage.grid.geometry;
tmv::domain::HorizontalDomainFrame& global_domain_frame =
    global_simulation_storage.configuration.domain_frame;
Field3D& u = global_simulation_storage.fields.u;
Field3D& v = global_simulation_storage.fields.v;
Field3D& w = global_simulation_storage.fields.w;
Field3D& theta = global_simulation_storage.fields.theta;
Field3D& qv = global_simulation_storage.fields.qv;
Field3D& tke = global_simulation_storage.fields.tke;
Field3D& dtke_dt_pbl = global_simulation_storage.tendencies.dtke_dt_pbl;

std::vector<float>& sfc_ustar = global_simulation_storage.surface.ustar;
std::vector<float>& sfc_tau_u = global_simulation_storage.surface.tau_u;
std::vector<float>& sfc_tau_v = global_simulation_storage.surface.tau_v;
std::vector<float>& sfc_sensible_heat_flux = global_simulation_storage.surface.sensible_heat_flux;
std::vector<float>& sfc_latent_heat_flux = global_simulation_storage.surface.latent_heat_flux;
std::vector<float>& sfc_moisture_flux = global_simulation_storage.surface.moisture_flux;
std::vector<float>& pbl_height_2d = global_simulation_storage.surface.pbl_height;

bool& global_gpu_apply_sgs =
    (global_simulation_storage.configuration.gpu_apply_sgs = false);
tmv::init::config::LateralBC& global_lateral_bc_south =
    global_simulation_storage.configuration.lateral_bc_south;
tmv::init::config::LateralBC& global_lateral_bc_north =
    global_simulation_storage.configuration.lateral_bc_north;
tmv::init::config::LateralBC& global_lateral_bc_west =
    global_simulation_storage.configuration.lateral_bc_west;
tmv::init::config::LateralBC& global_lateral_bc_east =
    global_simulation_storage.configuration.lateral_bc_east;
tmv::init::config::TopBC& global_top_bc =
    global_simulation_storage.configuration.top_bc;
TurbulenceRuntimeState& tmv::global_turbulence_runtime_state =
    global_simulation_storage.runtime.turbulence;
tmv::GridMetrics sgs_apply_test_grid_metrics;
tmv::GridMetrics& tmv::global_grid_metrics = sgs_apply_test_grid_metrics;

namespace
{

int sgs_dispatch_calls = 0;
std::vector<std::string> invalidated_resident_fields;

class LedgerScheme final : public TurbulenceSchemeBase
{
public:
    std::unique_ptr<TurbulenceSchemeBase> clone() const override
    {
        return std::make_unique<LedgerScheme>(*this);
    }
    std::string name() const override { return "ledger_test"; }
    int required_fields() const override { return 0; }
    void initialize(const TurbulenceConfig&) override {}
    void compute(const TurbulenceConfig&, const GridMetrics&,
                 const TurbulenceStateView&, TurbulenceTendencies&,
                 TurbulenceDiagnostics*) override {}
};

class SmagorinskySchemeMarker final : public TurbulenceSchemeBase
{
public:
    std::string name() const override { return "smagorinsky"; }
    int required_fields() const override { return 0; }
    void initialize(const TurbulenceConfig&) override {}
    void compute(const TurbulenceConfig&, const GridMetrics&,
                 const TurbulenceStateView&, TurbulenceTendencies&,
                 TurbulenceDiagnostics*) override {}
    const Field3D* realized_eddy_viscosity() const noexcept override
    {
        return viscosity.size() > 0 ? &viscosity : nullptr;
    }

    Field3D viscosity;
};

bool all_equal(const Field3D& field, float expected)
{
    for (int i = 0; i < static_cast<int>(field.size_r()); ++i)
    {
        for (int j = 0; j < static_cast<int>(field.size_th()); ++j)
        {
            for (int k = 0; k < static_cast<int>(field.size_z()); ++k)
            {
                if (std::abs(field[i][j][k] - expected) > 1.0e-6f)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

void resize_tendencies(TurbulenceTendencies& tendencies,
                       int nr,
                       int nth,
                       int nz)
{
    tendencies.dudt_sgs.resize(nr, nth, nz, 1.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, -2.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 3.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 4.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, -0.001f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 7.0f);
}

} // namespace

void invalidate_resident_field_backend(ComputeBackendRuntimeState&, const char* field_id)
{
    invalidated_resident_fields.emplace_back(field_id != nullptr ? field_id : "");
}

bool dispatch_sgs_apply_backend(ComputeBackendRuntimeState&,
                                float*,
                                float*,
                                float*,
                                float*,
                                float*,
                                float*,
                                const float*,
                                const float*,
                                const float*,
                                const float*,
                                const float*,
                                const float*,
                                const float*,
                                int,
                                int,
                                int,
                                bool,
                                bool,
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                                bool,
                                bool,
                                SgsApplyDispatch::CompletedObserver,
                                void*,
                                bool)
{
    ++sgs_dispatch_calls;
    return false;
}

namespace nimbus::parity
{

bool sample(const char*) { return false; }

bool check_field(const char*,
                 const char*,
                 const float*,
                 const float*,
                 int,
                 int,
                 int,
                 Tier,
                 ComparisonDomain,
                 int)
{
    return true;
}

} // namespace nimbus::parity

std::array<double, 4> sgs_benchmark_profile_seconds{};

void record_sgs_benchmark_stage(int stage, double seconds)
{
    sgs_benchmark_profile_seconds.at(static_cast<std::size_t>(stage)) += seconds;
}

TEST_CASE("measure SGS stability and complete moist application", "[.sgs-application-benchmark]")
{
    const auto setting = [](const char* name, int fallback)
    {
        const char* value = std::getenv(name);
        return value ? std::stoi(value) : fallback;
    };
    const int workers = setting("NIMBUS_SGS_BENCHMARK_THREADS", 1);
    const int nx = setting("NIMBUS_SGS_BENCHMARK_NX", 128);
    REQUIRE(workers >= 1);
    REQUIRE(workers <= 64);
    REQUIRE(nx >= 32);
    REQUIRE(nx <= 640);
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore_workers;
    omp_set_dynamic(0);
    omp_set_num_threads(workers);
#else
    REQUIRE(workers == 1);
#endif
    struct RestorePolicy
    {
        bool deterministic = tmv::diagnostics::deterministic_reductions_enabled();
        bool repairs = primary_state_repairs_enabled;
        ~RestorePolicy()
        {
            tmv::diagnostics::set_deterministic_reductions(deterministic);
            primary_state_repairs_enabled = repairs;
        }
    } restore_policy;
    tmv::diagnostics::set_deterministic_reductions(true);
    primary_state_repairs_enabled = false;
    constexpr int nz = 180;
    constexpr double step = 0.125;
    SimulationStorageState storage{{nx, nx, nz, 50.0, 100.0, step, 50.0}};
    storage.grid.geometry.initialize(nx, nx, nz, 50.0, 100.0, 50.0,
                                     CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    const std::array<Field3D*, 14> fields{
        &f.rho, &f.p, &f.u, &f.v, &f.w, &f.theta, &f.qv,
        &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh, &f.tke};
    for (auto* field : fields) field->resize(nx, nx, nz);
    storage.tendencies.dtke_dt_pbl.resize(nx, nx, nz, 0.0001f);
    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    resize_tendencies(tendencies, nx, nx, nz);
    storage.runtime.turbulence.scheme = std::make_unique<LedgerScheme>();
    for (std::size_t index = 0; index < f.rho.size(); ++index)
    {
        const float pattern = static_cast<float>((index * 17 + index / nz) % 31);
        f.rho.data()[index] = 0.75f + 0.01f * pattern;
        f.p.data()[index] = 72000.0f + 317.0f * pattern;
        f.u.data()[index] = -8.0f + 0.37f * pattern;
        f.v.data()[index] = 3.0f - 0.19f * pattern;
        f.w.data()[index] = -1.0f + 0.08f * pattern;
        f.theta.data()[index] = 298.0f + 0.1f * pattern;
        f.qv.data()[index] = 0.007f + 0.00004f * pattern;
        f.tke.data()[index] = 0.1f + 0.001f * pattern;
        for (std::size_t species = 7; species < 13; ++species)
            fields[species]->data()[index] = static_cast<float>(species - 6) * 0.00001f * pattern;
        tendencies.dudt_sgs.data()[index] = -0.001f * f.u.data()[index];
        tendencies.dvdt_sgs.data()[index] = -0.001f * f.v.data()[index];
        tendencies.dwdt_sgs.data()[index] = -0.001f * f.w.data()[index];
        tendencies.dthetadt_sgs.data()[index] = -0.001f * (f.theta.data()[index] - 299.5f);
        tendencies.dqvdt_sgs.data()[index] = -0.001f * (f.qv.data()[index] - 0.0076f);
        tendencies.dtkedt_sgs.data()[index] = -0.001f * f.tke.data()[index];
    }
    auto state = make_simulation_state_view(storage);
    std::array<double, 6> stability_ms{}, apply_ms{}, commit_ms{};
    sgs_benchmark_profile_seconds = {};
    for (int iteration = -2; iteration < 6; ++iteration)
    {
        const auto start = std::chrono::steady_clock::now();
        const auto stability = compute_sgs_explicit_stability(state, tendencies);
        const auto admitted = std::chrono::steady_clock::now();
        REQUIRE(stability.maximum_timestep_s >= step);
        const auto report = apply_admitted_sgs_substep(state, step, tendencies);
        const auto applied = std::chrono::steady_clock::now();
        commit_sgs_momentum_work(state, report);
        const auto committed = std::chrono::steady_clock::now();
        if (iteration >= 0)
        {
            stability_ms[iteration] = std::chrono::duration<double, std::milli>(admitted - start).count();
            apply_ms[iteration] = std::chrono::duration<double, std::milli>(applied - admitted).count();
            commit_ms[iteration] = std::chrono::duration<double, std::milli>(committed - applied).count();
        }
    }
    std::uint64_t fingerprint = 14695981039346656037ull;
    const auto hash_bytes = [&](const void* data, std::size_t bytes)
    {
        const auto* value = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index < bytes; ++index)
            fingerprint = (fingerprint ^ value[index]) * 1099511628211ull;
    };
    for (const auto* field : fields) hash_bytes(field->data(), field->size() * sizeof(float));
    const auto& report = storage.runtime.turbulence.cumulative_moist_energy_work;
    const double values[]{report.requested_linear_work_j, report.finite_step_correction_j,
        report.storage_rounding_j, report.committed_work_j, report.moist_dry_internal_change_j,
        report.moist_water_sensible_change_j, report.moist_phase_reference_change_j,
        report.moist_carrier_kinetic_change_j, report.moist_gravitational_change_j,
        report.moist_subgrid_tke_change_j, report.moist_committed_energy_change_j,
        report.moist_energy_closure_residual_j};
    hash_bytes(values, sizeof(values));
    REQUIRE(report.moist_energy_complete);
    std::cout << std::setprecision(17) << "NIMBUS_SGS_BENCHMARK {\"threads\":" << workers
              << ",\"shape\":[" << nx << ',' << nx << ',' << nz << "]"
              << ",\"fingerprint\":\"" << std::hex << fingerprint << std::dec << '"';
    const auto print_series = [&](const char* name, const auto& series)
    {
        std::cout << ",\"" << name << "\":[";
        for (std::size_t index = 0; index < series.size(); ++index)
            std::cout << (index ? "," : "") << series[index];
        std::cout << ']';
    };
    print_series("stability_ms", stability_ms);
    print_series("apply_ms", apply_ms);
    print_series("commit_ms", commit_ms);
    print_series("profile_total_seconds", sgs_benchmark_profile_seconds);
    std::cout << '}' << std::endl;
}

TEST_CASE("SGS carrier work agrees with an independent physical-cell kinetic inventory",
          "[turbulence][sgs][carrier][reference][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore_workers;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    const bool staggered = GENERATE(false, true);
    const bool moist = GENERATE(false, true);
    struct RestoreRepairs
    {
        bool enabled = primary_state_repairs_enabled;
        ~RestoreRepairs() { primary_state_repairs_enabled = enabled; }
    } restore_repairs;
    primary_state_repairs_enabled = false;
    constexpr int nx = 9, ny = 8, nz = 7;
    constexpr double dx = 13.0, dy = 17.0, dz_local = 23.0, step = 0.125;
    SimulationStorageState storage{{nx, ny, nz, dx, dz_local, step, dy}};
    storage.grid.geometry.initialize(nx, ny, nz, dx, dz_local, dy,
        CoordinateSystem::Cartesian, staggered ? StaggerType::CGrid : StaggerType::Collocated);
    auto& f = storage.fields;
    const std::array<Field3D*, 7> water{&f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
    const float poison = std::numeric_limits<float>::quiet_NaN();
    f.rho.resize(nx, ny, nz, poison);
    for (auto* field : water) field->resize(nx, ny, nz, poison);
    f.p.resize(nx, ny, nz, 90000.0f);
    f.theta.resize(nx, ny, nz, 300.0f);
    f.tke.resize(nx, ny, nz, 1.0f);
    f.u.resize(nx, ny, nz);
    f.v.resize(nx, ny, nz);
    f.w.resize(nx, ny, nz);
    storage.tendencies.dtke_dt_pbl.resize(nx, ny, nz, 0.0f);
    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    resize_tendencies(tendencies, nx, ny, nz);
    tendencies.dthetadt_sgs.fill(0.0f);
    tendencies.dqvdt_sgs.fill(0.0f);
    tendencies.dtkedt_sgs.fill(0.0f);
    storage.runtime.turbulence.scheme = std::make_unique<LedgerScheme>();
    for (std::size_t n = 0; n < f.u.size(); ++n)
    {
        const float pattern = static_cast<float>((n * 17 + n / nz) % 31);
        f.u.data()[n] = 2.0f + 0.03f * pattern;
        f.v.data()[n] = -3.0f - 0.07f * pattern;
        f.w.data()[n] = 0.5f + 0.01f * pattern;
        tendencies.dudt_sgs.data()[n] = -0.05f * f.u.data()[n];
        tendencies.dvdt_sgs.data()[n] = -0.04f * f.v.data()[n];
        tendencies.dwdt_sgs.data()[n] = -0.03f * f.w.data()[n];
    }
    auto state = make_simulation_state_view(storage);
    const auto domain = physical_scalar_domain(state.grid);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const float pattern = static_cast<float>((i * 13 + j * 7 + k * 3) % 23);
                f.rho(i, j, k) = 0.75f + 0.01f * pattern;
                for (std::size_t species = 0; species < water.size(); ++species)
                    (*water[species])(i, j, k) = static_cast<float>(species + 1) * 0.00001f * pattern;
            }
    if (!moist) f.qh.release_storage();
    const std::array<Field3D, 3> before{f.u, f.v, f.w};
    const auto report = apply_sgs_tendencies(state, step, tendencies);
    long double reference = 0.0L;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                double total_water = 0.0;
                if (moist)
                    for (const auto* field : water) total_water += static_cast<double>((*field)(i, j, k));
                const double mass = static_cast<double>(f.rho(i, j, k)) * (dx * dy * dz_local) * (1.0 + total_water);
                const auto add = [&](const Field3D& old, const Field3D& current, int ii, int jj, int kk, double share)
                {
                    const double a = old(ii, jj, kk), b = current(ii, jj, kk);
                    reference += static_cast<long double>(share * mass) * (b * b - a * a);
                };
                if (staggered)
                {
                    add(before[0], f.u, i - 1, j, k, 0.25);
                    add(before[0], f.u, i, j, k, 0.25);
                    add(before[1], f.v, i, j - 1, k, 0.25);
                    add(before[1], f.v, i, j, k, 0.25);
                    if (k > domain.k_begin) add(before[2], f.w, i, j, k - 1, 0.25);
                    add(before[2], f.w, i, j, k, 0.25);
                }
                else
                {
                    add(before[0], f.u, i, j, k, 0.5);
                    add(before[1], f.v, i, j, k, 0.5);
                    add(before[2], f.w, i, j, k, 0.5);
                }
            }
    CHECK(report.committed_work_j == Approx(static_cast<double>(reference)).epsilon(1e-12));
    CHECK(report.moist_energy_complete == moist);
    CHECK(storage.runtime.turbulence.momentum_work_step_count == 1);
}

TEST_CASE("SGS tendency application uses explicit SimulationStateView")
{
    NR = 2;
    NTH = 2;
    NZ = 2;
    dr = 100.0;
    dz = 50.0;
    dt = 1.0;
    dtheta = 100.0;

    u.resize(NR, NTH, NZ, 100.0f);
    v.resize(NR, NTH, NZ, 100.0f);
    w.resize(NR, NTH, NZ, 100.0f);
    theta.resize(NR, NTH, NZ, 280.0f);
    qv.resize(NR, NTH, NZ, 0.001f);
    tke.resize(NR, NTH, NZ, 0.25f);
    dtke_dt_pbl.resize(NR, NTH, NZ, 10.0f);
    global_gpu_apply_sgs = true;
    sgs_dispatch_calls = 0;

    int local_nr = 2;
    int local_nth = 2;
    int local_nz = 2;
    double local_dr = 200.0;
    double local_dz = 75.0;
    double local_dt = 1.0;
    double local_dtheta = 250.0;
    GridGeometry local_geometry;

    Field3D local_rho(local_nr, local_nth, local_nz, 1.0f);
    Field3D local_p(local_nr, local_nth, local_nz, 90000.0f);
    Field3D local_u(local_nr, local_nth, local_nz, 10.0f);
    Field3D local_w(local_nr, local_nth, local_nz, 30.0f);
    Field3D local_v(local_nr, local_nth, local_nz, 20.0f);
    Field3D local_tracer(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_theta(local_nr, local_nth, local_nz, 300.0f);
    Field3D local_qv(local_nr, local_nth, local_nz, 0.010f);
    Field3D local_qc(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qr(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qi(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qs(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qh(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_qg(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_tke(local_nr, local_nth, local_nz, 0.25f);
    Field3D local_dtheta_dt_rad(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtheta_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dqv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_du_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dv_dt_pbl(local_nr, local_nth, local_nz, 0.0f);
    Field3D local_dtke_dt_pbl(local_nr, local_nth, local_nz, 3.0f);

    std::vector<double> local_rho0(local_nz, 1.0);
    std::vector<double> local_p0(local_nz, 90000.0);
    std::vector<double> local_qv0(local_nz, 0.010);
    std::vector<double> local_theta0(local_nz, 300.0);
    std::vector<double> local_u0(local_nz, 0.0);
    std::vector<double> local_v0(local_nz, 0.0);
    double local_coriolis = 0.0;
    double local_time = 0.0;

    SimulationStateView local_state{
        {local_nr, local_nth, local_nz, local_dr, local_dz,
         local_dt, local_dtheta, local_geometry},
        {local_rho, local_p, local_u, local_w, local_v, local_tracer,
         local_theta, local_qv, local_qc, local_qr, local_qi, local_qs,
         local_qh, local_qg, local_tke},
        {local_rho0, local_p0, local_qv0, local_theta0, local_u0, local_v0},
        {local_dtheta_dt_rad, local_dtheta_dt_pbl, local_dqv_dt_pbl,
         local_du_dt_pbl, local_dv_dt_pbl, local_dtke_dt_pbl},
        local_coriolis,
        local_time};
    SimulationStorageState local_owners{{local_nr, local_nth, local_nz, local_dr, local_dz, local_dt, local_dtheta}};
    local_state.configuration = &local_owners.configuration;
    local_state.turbulence_runtime = &local_owners.runtime.turbulence;
    local_state.compute_runtime = &local_owners.runtime.compute;
    local_state.domain_frame = &local_owners.configuration.domain_frame;

    TurbulenceTendencies tendencies;
    resize_tendencies(tendencies, local_nr, local_nth, local_nz);

    apply_sgs_tendencies(local_state, 2.0, tendencies);

    REQUIRE(sgs_dispatch_calls == 0);
    REQUIRE(all_equal(local_u, 12.0f));
    REQUIRE(all_equal(local_v, 16.0f));
    REQUIRE(all_equal(local_w, 36.0f));
    REQUIRE(all_equal(local_theta, 308.0f));
    REQUIRE(all_equal(local_qv, 0.008f));
    REQUIRE(all_equal(local_tke, 20.25f));
    REQUIRE(all_equal(u, 100.0f));
    REQUIRE(all_equal(v, 100.0f));
    REQUIRE(all_equal(w, 100.0f));
    REQUIRE(all_equal(theta, 280.0f));
    REQUIRE(all_equal(qv, 0.001f));
    REQUIRE(all_equal(tke, 0.25f));
}

TEST_CASE("SGS apply preserves the exact zero-TKE equilibrium",
          "[turbulence][sgs][tke][zero-energy]")
{
    constexpr int nr = 3, nth = 3, nz = 2;
    SimulationStorageState storage{{nr, nth, nz, 100.0, 50.0, 1.0, 100.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 100.0, 50.0, 100.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 0.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    storage.fields.tke.resize(nr, nth, nz, 0.0f);
    storage.tendencies.dtke_dt_pbl.resize(nr, nth, nz, 0.0f);

    TurbulenceTendencies tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);

    global_gpu_apply_sgs = false;
    set_primary_state_repairs_enabled(true);
    SimulationStateView state = make_simulation_state_view(storage);
    apply_sgs_tendencies(state, 1.0, tendencies);

    REQUIRE(all_equal(storage.fields.tke, 0.0f));
}

TEST_CASE("SGS apply rejects malformed input before mutating state",
          "[turbulence][sgs][state][rollback]")
{
    constexpr int nr = 3, nth = 3, nz = 2;
    SimulationStorageState storage{{nr, nth, nz, 100.0, 50.0, 1.0, 100.0}};
    storage.fields.u.resize(nr, nth, nz, 1.0f);
    storage.fields.v.resize(nr, nth, nz, 2.0f);
    storage.fields.w.resize(nr, nth, nz, 3.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);

    TurbulenceTendencies tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 5.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 5.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 5.0f);
    // A wrong theta-tendency shape used to be discovered only after wind
    // fields had already been updated (or by an out-of-bounds access).
    tendencies.dthetadt_sgs.resize(nr, nth, nz - 1, 5.0f);

    const Field3D u_before = storage.fields.u;
    SimulationStateView state = make_simulation_state_view(storage);
    REQUIRE_THROWS_AS(
        apply_sgs_tendencies(state, 1.0, tendencies), std::invalid_argument);
    REQUIRE(std::equal(
        u_before.data(), u_before.data() + u_before.size(),
        storage.fields.u.data()));
}

TEST_CASE("active SGS commit attributes exact C-grid momentum work")
{
    constexpr int nr = 5, nth = 5, nz = 4;
    constexpr double dx = 10.0, dy = 20.0, dz_local = 30.0;
    constexpr double step_dt = 0.25;
    SimulationStorageState storage{
        {nr, nth, nz, dx, dz_local, step_dt, dy}};
    storage.grid.geometry.initialize(
        nr, nth, nz, dx, dz_local, dy,
        CoordinateSystem::Cartesian, StaggerType::CGrid);

    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 2.0f);
    storage.fields.v.resize(nr, nth, nz, -3.0f);
    storage.fields.w.resize(nr, nth, nz, 0.5f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                storage.fields.rho(i, j, k) =
                    0.8f + 0.1f * i + 0.02f * j + 0.01f * k;

    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dudt_sgs(1, 1, 0) = 0.125f;
    tendencies.dvdt_sgs(1, 1, 0) = -0.25f;
    tendencies.dwdt_sgs(1, 1, 0) = 0.375f;
    storage.runtime.turbulence.scheme = std::make_unique<LedgerScheme>();

    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    set_primary_state_repairs_enabled(false);
    SimulationStateView state = make_simulation_state_view(storage);

    const double volume = dx * dy * dz_local;
    const double u_old = storage.fields.u(1, 1, 0);
    const double v_old = storage.fields.v(1, 1, 0);
    const double w_old = storage.fields.w(1, 1, 0);
    const double du = static_cast<double>(tendencies.dudt_sgs(1, 1, 0)) * step_dt;
    const double dv = static_cast<double>(tendencies.dvdt_sgs(1, 1, 0)) * step_dt;
    const double dw = static_cast<double>(tendencies.dwdt_sgs(1, 1, 0)) * step_dt;
    const double mass_u = volume * 0.5 *
        (storage.fields.rho(1, 1, 0) + storage.fields.rho(2, 1, 0));
    const double mass_v = volume * 0.5 *
        (storage.fields.rho(1, 1, 0) + storage.fields.rho(1, 2, 0));
    const double mass_w = volume * 0.5 *
        (storage.fields.rho(1, 1, 0) + storage.fields.rho(1, 1, 1));
    const double expected_requested =
        mass_u * u_old * du + mass_v * v_old * dv + mass_w * w_old * dw;
    const double expected_finite = 0.5 *
        (mass_u * du * du + mass_v * dv * dv + mass_w * dw * dw);

    apply_sgs_tendencies(state, step_dt, tendencies);

    const double expected_committed = 0.5 * (
        mass_u * (std::pow(static_cast<double>(storage.fields.u(1, 1, 0)), 2)
                  - u_old * u_old) +
        mass_v * (std::pow(static_cast<double>(storage.fields.v(1, 1, 0)), 2)
                  - v_old * v_old) +
        mass_w * (std::pow(static_cast<double>(storage.fields.w(1, 1, 0)), 2)
                  - w_old * w_old));
    const auto& runtime = storage.runtime.turbulence;
    const auto& report = runtime.last_momentum_work;
    REQUIRE(report.requested_linear_work_j == Approx(expected_requested));
    REQUIRE(report.finite_step_correction_j == Approx(expected_finite));
    REQUIRE(report.committed_work_j == Approx(expected_committed));
    REQUIRE(report.storage_rounding_j == Approx(
        expected_committed - expected_requested - expected_finite));
    REQUIRE(report.committed_work_j == Approx(
        report.requested_linear_work_j + report.finite_step_correction_j +
        report.storage_rounding_j));
    REQUIRE(runtime.momentum_work_step_count == 1);
    REQUIRE(runtime.cumulative_committed_momentum_work_j ==
            Approx(report.committed_work_j));
    REQUIRE(sgs_dispatch_calls == 0);

    set_primary_state_repairs_enabled(true);
}

TEST_CASE("active SGS commit attributes exact moist basis-v2 scalar and TKE energy")
{
    constexpr int nr = 5, nth = 5, nz = 4;
    constexpr double dx = 10.0, dy = 20.0, dz_local = 30.0;
    constexpr double step_dt = 0.25;
    constexpr int ci = 2, cj = 2, ck = 1;
    SimulationStorageState storage{
        {nr, nth, nz, dx, dz_local, step_dt, dy}};
    storage.grid.geometry.initialize(
        nr, nth, nz, dx, dz_local, dy,
        CoordinateSystem::Cartesian, StaggerType::CGrid);

    auto& f = storage.fields;
    f.rho.resize(nr, nth, nz, 1.1f);
    f.p.resize(nr, nth, nz, 90000.0f);
    f.u.resize(nr, nth, nz, 2.0f);
    f.v.resize(nr, nth, nz, -3.0f);
    f.w.resize(nr, nth, nz, 0.5f);
    f.theta.resize(nr, nth, nz, 300.0f);
    f.qv.resize(nr, nth, nz, 0.010f);
    f.qc.resize(nr, nth, nz, 0.001f);
    f.qr.resize(nr, nth, nz, 0.002f);
    f.qi.resize(nr, nth, nz, 0.003f);
    f.qs.resize(nr, nth, nz, 0.004f);
    f.qg.resize(nr, nth, nz, 0.005f);
    f.qh.resize(nr, nth, nz, 0.006f);
    f.tke.resize(nr, nth, nz, 1.25f);
    storage.tendencies.dtke_dt_pbl.resize(nr, nth, nz, 0.0f);

    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dudt_sgs(ci, cj, ck) = 0.1f;
    tendencies.dthetadt_sgs(ci, cj, ck) = 0.4f;
    tendencies.dqvdt_sgs(ci, cj, ck) = 0.0002f;
    tendencies.dtkedt_sgs(ci, cj, ck) = -0.1f;
    storage.tendencies.dtke_dt_pbl(ci, cj, ck) = 0.02f;
    storage.runtime.turbulence.scheme = std::make_unique<LedgerScheme>();

    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    set_primary_state_repairs_enabled(false);
    SimulationStateView state = make_simulation_state_view(storage);

    const double theta_old = static_cast<double>(f.theta(ci, cj, ck));
    const double qv_old = static_cast<double>(f.qv(ci, cj, ck));
    const double tke_old = static_cast<double>(f.tke(ci, cj, ck));
    apply_sgs_tendencies(state, step_dt, tendencies);

    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i =
        physical_constants::specific_heat_ice_jkgk;
    constexpr double t_ref =
        physical_constants::freezing_temperature_k;
    constexpr double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    const double dry_mass =
        static_cast<double>(f.rho(ci, cj, ck)) * dx * dy * dz_local;
    const double exner = std::pow(
        static_cast<double>(f.p(ci, cj, ck))
            / physical_constants::reference_pressure_pa,
        kappa);
    const double temperature_old = theta_old * exner;
    const double temperature_new =
        static_cast<double>(f.theta(ci, cj, ck)) * exner;
    const double qv_new = static_cast<double>(f.qv(ci, cj, ck));
    const double liquid =
        static_cast<double>(f.qc(ci, cj, ck))
        + static_cast<double>(f.qr(ci, cj, ck));
    const double ice =
        static_cast<double>(f.qi(ci, cj, ck))
        + static_cast<double>(f.qs(ci, cj, ck))
        + static_cast<double>(f.qg(ci, cj, ck))
        + static_cast<double>(f.qh(ci, cj, ck));
    const double u_e_new =
        static_cast<double>(f.u(ci, cj, ck));
    const double specific_kinetic = 0.25 * (
        2.0 * 2.0 + u_e_new * u_e_new
        + 3.0 * 3.0 + 3.0 * 3.0
        + 0.5 * 0.5 + 0.5 * 0.5);
    const double z = storage.grid.geometry.z[ck];
    const double expected_dry = dry_mass * cv_d
        * (temperature_new - temperature_old);
    const double expected_water = dry_mass * (
        (temperature_new - t_ref)
            * (qv_new * cv_v + liquid * c_l + ice * c_i)
        - (temperature_old - t_ref)
            * (qv_old * cv_v + liquid * c_l + ice * c_i));
    const double expected_phase =
        dry_mass * (qv_new - qv_old) * vapor_reference;
    const double expected_carrier =
        dry_mass * (qv_new - qv_old) * specific_kinetic;
    const double expected_gravity =
        dry_mass * (qv_new - qv_old)
        * physical_constants::gravity_ms2 * z;
    const double expected_tke =
        dry_mass * (static_cast<double>(f.tke(ci, cj, ck)) - tke_old);
    const double total_water_old = qv_old + liquid + ice;
    const double face_mass = dry_mass * (1.0 + total_water_old);
    const double du = static_cast<double>(
        tendencies.dudt_sgs(ci, cj, ck)) * step_dt;
    const double expected_requested_work = face_mass * 2.0 * du;
    const double expected_committed_work =
        0.5 * face_mass * (u_e_new * u_e_new - 2.0 * 2.0);

    const auto& report = storage.runtime.turbulence.last_momentum_work;
    REQUIRE(report.moist_energy_complete);
    REQUIRE(report.requested_linear_work_j
            == Approx(expected_requested_work));
    REQUIRE(report.committed_work_j == Approx(expected_committed_work));
    REQUIRE(report.moist_dry_internal_change_j == Approx(expected_dry));
    REQUIRE(report.moist_water_sensible_change_j == Approx(expected_water));
    REQUIRE(report.moist_phase_reference_change_j == Approx(expected_phase));
    REQUIRE(report.moist_carrier_kinetic_change_j == Approx(expected_carrier));
    REQUIRE(report.moist_gravitational_change_j == Approx(expected_gravity));
    REQUIRE(report.moist_subgrid_tke_change_j == Approx(expected_tke));
    REQUIRE(report.moist_energy_closure_residual_j == 0.0);
    REQUIRE(storage.runtime.turbulence.cumulative_moist_energy_work
                .moist_energy_complete);
    set_primary_state_repairs_enabled(true);
}

TEST_CASE(
    "active SGS commit attributes both stored periodic-x seam u-face slots")
{
    constexpr int nr = 5, nth = 5, nz = 3;
    constexpr double dx = 10.0, dy = 20.0, dz_local = 30.0;
    constexpr double step_dt = 0.25;
    SimulationStorageState storage{
        {nr, nth, nz, dx, dz_local, step_dt, dy}};
    storage.grid.geometry.initialize(
        nr, nth, nz, dx, dz_local, dy,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 2.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                storage.fields.rho(i, j, k) =
                    static_cast<float>(0.8 + 0.1 * i + 0.02 * j);

    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    constexpr double east_tendency = 0.125;
    constexpr double west_tendency = -0.075;
    tendencies.dudt_sgs(nr - 2, 1, 0) =
        static_cast<float>(east_tendency);
    tendencies.dudt_sgs(0, 1, 0) =
        static_cast<float>(west_tendency);
    storage.runtime.turbulence.scheme = std::make_unique<LedgerScheme>();

    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    set_primary_state_repairs_enabled(false);
    SimulationStateView state = make_simulation_state_view(storage);

    const double volume = dx * dy * dz_local;
    // The endpoint energy basis stores west and east seam velocities as
    // distinct slots until boundary closure. Each slot therefore owns only
    // the half-cell mass on its interior side; wrapping the opposite interior
    // half-cell onto either slot double-counts the seam state change.
    const double east_mass = 0.5 * volume
        * static_cast<double>(storage.fields.rho(nr - 2, 1, 0));
    const double west_mass = 0.5 * volume
        * static_cast<double>(storage.fields.rho(1, 1, 0));
    const double east_old_u = storage.fields.u(nr - 2, 1, 0);
    const double west_old_u = storage.fields.u(0, 1, 0);
    const double east_du = east_tendency * step_dt;
    const double west_du = west_tendency * step_dt;

    apply_sgs_tendencies(state, step_dt, tendencies);

    const auto& report = storage.runtime.turbulence.last_momentum_work;
    const double expected_requested =
        east_mass * east_old_u * east_du
        + west_mass * west_old_u * west_du;
    const double expected_finite_step = 0.5 * (
        east_mass * east_du * east_du
        + west_mass * west_du * west_du);
    const double expected_committed = 0.5 * east_mass * (
        std::pow(static_cast<double>(storage.fields.u(nr - 2, 1, 0)), 2)
        - east_old_u * east_old_u)
        + 0.5 * west_mass * (
            std::pow(static_cast<double>(storage.fields.u(0, 1, 0)), 2)
            - west_old_u * west_old_u);
    REQUIRE(report.requested_linear_work_j == Approx(expected_requested));
    REQUIRE(report.finite_step_correction_j == Approx(expected_finite_step));
    REQUIRE(report.committed_work_j == Approx(expected_committed));
    REQUIRE(report.committed_work_j == Approx(
        report.requested_linear_work_j + report.finite_step_correction_j
        + report.storage_rounding_j));

    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    set_primary_state_repairs_enabled(true);
}

TEST_CASE("active SGS commit uses exact terrain-following dual masses")
{
    constexpr int nr = 5, nth = 5, nz = 4;
    constexpr double dx = 10.0, dy = 20.0, nominal_dz = 30.0;
    constexpr double step_dt = 0.25;
    SimulationStorageState storage{
        {nr, nth, nz, dx, nominal_dz, step_dt, dy}};
    storage.grid.geometry.initialize(
        nr, nth, nz, dx, nominal_dz, dy,
        CoordinateSystem::Cartesian, StaggerType::CGrid);

    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 2.0f);
    storage.fields.v.resize(nr, nth, nz, -3.0f);
    storage.fields.w.resize(nr, nth, nz, 0.5f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                storage.fields.rho(i, j, k) =
                    0.8f + 0.1f * i + 0.02f * j + 0.01f * k;

    auto& terrain = storage.runtime.terrain;
    terrain.config.scheme_id = "bell";
    terrain.metrics.z.resize(nr, nth, nz, 0.0);
    terrain.metrics.z_face.resize(nr, nth, nz + 1, 0.0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
        {
            double lower = 40.0 + 7.0 * i + 3.0 * j;
            terrain.metrics.z_face(i, j, 0) = lower;
            for (int k = 0; k < nz; ++k)
            {
                const double thickness =
                    18.0 + 5.0 * i + 3.0 * j + 2.0 * k;
                const double upper = lower + thickness;
                terrain.metrics.z(i, j, k) = 0.5 * (lower + upper);
                terrain.metrics.z_face(i, j, k + 1) = upper;
                lower = upper;
            }
        }

    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dudt_sgs(1, 1, 0) = 0.125f;
    tendencies.dvdt_sgs(1, 1, 0) = -0.25f;
    tendencies.dwdt_sgs(1, 1, 0) = 0.375f;
    storage.runtime.turbulence.scheme = std::make_unique<LedgerScheme>();

    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    set_primary_state_repairs_enabled(false);
    SimulationStateView state = make_simulation_state_view(storage);

    const auto scalar_mass = [&](int i, int j, int k)
    {
        const double thickness = terrain.metrics.z_face(i, j, k + 1)
            - terrain.metrics.z_face(i, j, k);
        return static_cast<double>(storage.fields.rho(i, j, k))
            * dx * dy * thickness;
    };
    const double u_old = storage.fields.u(1, 1, 0);
    const double v_old = storage.fields.v(1, 1, 0);
    const double w_old = storage.fields.w(1, 1, 0);
    const double du = static_cast<double>(tendencies.dudt_sgs(1, 1, 0)) * step_dt;
    const double dv = static_cast<double>(tendencies.dvdt_sgs(1, 1, 0)) * step_dt;
    const double dw = static_cast<double>(tendencies.dwdt_sgs(1, 1, 0)) * step_dt;
    const double mass_u = 0.5 *
        (scalar_mass(1, 1, 0) + scalar_mass(2, 1, 0));
    const double mass_v = 0.5 *
        (scalar_mass(1, 1, 0) + scalar_mass(1, 2, 0));
    const double mass_w = 0.5 *
        (scalar_mass(1, 1, 0) + scalar_mass(1, 1, 1));
    const double expected_requested =
        mass_u * u_old * du + mass_v * v_old * dv + mass_w * w_old * dw;
    const double expected_finite = 0.5 *
        (mass_u * du * du + mass_v * dv * dv + mass_w * dw * dw);

    apply_sgs_tendencies(state, step_dt, tendencies);

    const double expected_committed = 0.5 * (
        mass_u * (std::pow(static_cast<double>(storage.fields.u(1, 1, 0)), 2)
                  - u_old * u_old) +
        mass_v * (std::pow(static_cast<double>(storage.fields.v(1, 1, 0)), 2)
                  - v_old * v_old) +
        mass_w * (std::pow(static_cast<double>(storage.fields.w(1, 1, 0)), 2)
                  - w_old * w_old));
    const auto& runtime = storage.runtime.turbulence;
    const auto& report = runtime.last_momentum_work;
    REQUIRE(report.requested_linear_work_j == Approx(expected_requested));
    REQUIRE(report.finite_step_correction_j == Approx(expected_finite));
    REQUIRE(report.committed_work_j == Approx(expected_committed));
    REQUIRE(report.storage_rounding_j == Approx(
        expected_committed - expected_requested - expected_finite));
    REQUIRE(report.committed_work_j == Approx(
        report.requested_linear_work_j + report.finite_step_correction_j +
        report.storage_rounding_j));
    REQUIRE(runtime.momentum_work_step_count == 1);
    REQUIRE(runtime.cumulative_committed_momentum_work_j ==
            Approx(report.committed_work_j));
    REQUIRE(sgs_dispatch_calls == 0);

    set_primary_state_repairs_enabled(true);
}

TEST_CASE("realized SGS bound enforces energy, scalar, and TKE stability",
          "[turbulence][sgs][stability][sci086]")
{
    constexpr int nr = 5, nth = 5, nz = 3;
    SimulationStorageState storage{{nr, nth, nz, 10.0, 20.0, 1.0, 15.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 10.0, 20.0, 15.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 0.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 305.0f);
    storage.fields.qv.resize(nr, nth, nz, 0.01f);
    storage.fields.tke.resize(nr, nth, nz, 1.0f);
    storage.tendencies.dtke_dt_pbl.resize(nr, nth, nz, 0.0f);

    TurbulenceTendencies tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);

    // Momentum alone permits 0.9 s: the exact forward-Euler energy root is
    // -2<u,a>/<a,a> = 1 s and the shared safety factor is 0.9.
    storage.fields.u(1, 1, 0) = 2.0f;
    tendencies.dudt_sgs(1, 1, 0) = -4.0f;

    // Theta alone permits 0.45 s before either endpoint can cross the
    // current [300, 310] K range.
    storage.fields.theta(1, 1, 0) = 300.0f;
    storage.fields.theta(2, 1, 0) = 310.0f;
    tendencies.dthetadt_sgs(1, 1, 0) = 20.0f;
    tendencies.dthetadt_sgs(2, 1, 0) = -20.0f;

    // TKE is the most restrictive: 0.25/1 * 0.9 = 0.225 s.
    storage.fields.tke(1, 1, 0) = 0.25f;
    tendencies.dtkedt_sgs(1, 1, 0) = -1.0f;

    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    const SimulationStateView state = make_simulation_state_view(storage);
    const SgsExplicitStability stability =
        compute_sgs_explicit_stability(state, tendencies);

    REQUIRE(stability.momentum_energy_timestep_s == Approx(0.9));
    REQUIRE(stability.theta_monotonic_timestep_s == Approx(0.45));
    REQUIRE(std::isinf(stability.qv_monotonic_timestep_s));
    REQUIRE(stability.tke_positivity_timestep_s == Approx(0.225));
    REQUIRE(stability.maximum_timestep_s == Approx(0.225));

    storage.runtime.turbulence.scheme =
        std::make_unique<SmagorinskySchemeMarker>();
    set_primary_state_repairs_enabled(false);
    const double initial_u_energy =
        std::pow(static_cast<double>(storage.fields.u(1, 1, 0)), 2);
    const float initial_u = storage.fields.u(1, 1, 0);
    REQUIRE_THROWS_WITH(
        apply_sgs_tendencies(
            state, 1.1 * stability.maximum_timestep_s, tendencies),
        Catch::Contains("exceeds the realized explicit stability limit"));
    REQUIRE(storage.fields.u(1, 1, 0) == initial_u);
    (void)apply_sgs_tendencies(
        state, stability.maximum_timestep_s, tendencies);
    REQUIRE(std::pow(static_cast<double>(storage.fields.u(1, 1, 0)), 2) <
            initial_u_energy);
    REQUIRE(storage.fields.theta(1, 1, 0) >= 300.0f);
    REQUIRE(storage.fields.theta(2, 1, 0) <= 310.0f);
    REQUIRE(storage.fields.tke(1, 1, 0) > 0.0f);
    set_primary_state_repairs_enabled(true);
}

TEST_CASE("realized SGS bound rejects an energy-producing stress tendency",
          "[turbulence][sgs][stability][energy][fail-closed]")
{
    constexpr int nr = 5, nth = 5, nz = 3;
    SimulationStorageState storage{{nr, nth, nz, 10.0, 20.0, 1.0, 15.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 10.0, 20.0, 15.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 0.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);

    TurbulenceTendencies tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    storage.fields.u(1, 1, 0) = 2.0f;
    tendencies.dudt_sgs(1, 1, 0) = 1.0f;

    const SimulationStateView state = make_simulation_state_view(storage);
    REQUIRE_THROWS_WITH(
        compute_sgs_explicit_stability(state, tendencies),
        Catch::Contains("not dissipative"));
}

TEST_CASE("collocated SGS stability and commit exclude every closure plane",
          "[turbulence][sgs][stability][agrid][closure]")
{
    constexpr int n = 7;
    constexpr float closure = 1000.0f;
    SimulationStorageState storage{{n, n, n, 10.0, 20.0, 1.0, 15.0}};
    storage.grid.geometry.initialize(
        n, n, n, 10.0, 20.0, 15.0,
        CoordinateSystem::Cartesian, StaggerType::Collocated);
    storage.fields.rho.resize(n, n, n, closure);
    storage.fields.u.resize(n, n, n, closure);
    storage.fields.v.resize(n, n, n, closure);
    storage.fields.w.resize(n, n, n, closure);
    storage.fields.theta.resize(n, n, n, closure);
    storage.base.rho0.assign(n, 1.0);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 1; k < n - 1; ++k)
            {
                storage.fields.rho(i, j, k) = static_cast<float>(
                    0.9 + 0.02 * i + 0.01 * j + 0.005 * k);
                storage.fields.u(i, j, k) = static_cast<float>(
                    1.0 + 0.08 * i * i - 0.03 * j * k);
                storage.fields.v(i, j, k) = static_cast<float>(
                    -0.5 + 0.04 * i * k - 0.06 * j * j);
                storage.fields.w(i, j, k) = static_cast<float>(
                    0.2 - 0.03 * i * j + 0.05 * k * k);
                storage.fields.theta(i, j, k) = static_cast<float>(
                    298.0 + 0.2 * i - 0.1 * j + 0.04 * k * k);
            }

    auto scheme = std::make_unique<SmagorinskySchemeMarker>();
    scheme->viscosity.resize(n, n, n, closure);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 1; k < n - 1; ++k)
                scheme->viscosity(i, j, k) = static_cast<float>(
                    1.0 + 0.02 * i + 0.03 * j + 0.01 * k);
    storage.runtime.turbulence.scheme = std::move(scheme);
    storage.runtime.turbulence.config.mode = "3d";

    TurbulenceTendencies& tendencies =
        storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(n, n, n, 17.0f);
    tendencies.dvdt_sgs.resize(n, n, n, 17.0f);
    tendencies.dwdt_sgs.resize(n, n, n, 17.0f);
    tendencies.dthetadt_sgs.resize(n, n, n, 17.0f);

    tmv::TurbulenceStateView turbulence_state;
    turbulence_state.u = &storage.fields.u;
    turbulence_state.v = &storage.fields.v;
    turbulence_state.w = &storage.fields.w;
    turbulence_state.rho = &storage.fields.rho;
    turbulence_state.theta = &storage.fields.theta;
    turbulence_state.rho0 = &storage.base.rho0;
    turbulence_state.c_grid = false;
    turbulence_state.NR = n;
    turbulence_state.NTH = n;
    turbulence_state.NZ = n;
    tmv::GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 20.0;
    metrics.dz.assign(n, 15.0);
    metrics.staggering = StaggerType::Collocated;

    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    tmv::eddy_viscosity::compute_momentum_diffusion_tendencies(
        storage.runtime.turbulence.config, turbulence_state, metrics,
        *storage.runtime.turbulence.scheme->realized_eddy_viscosity(),
        tendencies.dudt_sgs, tendencies.dvdt_sgs, tendencies.dwdt_sgs);
    tmv::eddy_viscosity::compute_scalar_diffusion_tendencies(
        storage.runtime.turbulence.config, turbulence_state, metrics,
        *storage.runtime.turbulence.scheme->realized_eddy_viscosity(),
        storage.fields.theta, tendencies.dthetadt_sgs);

    // A hostile closure tendency would dominate the scalar bound if a
    // storage plane were mistaken for physical atmosphere.
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                if (i == 0 || i == n - 1 || j == 0 || j == n - 1
                    || k == 0 || k == n - 1)
                {
                    tendencies.dudt_sgs(i, j, k) = 1.0f;
                    tendencies.dvdt_sgs(i, j, k) = 1.0f;
                    tendencies.dwdt_sgs(i, j, k) = 1.0f;
                    tendencies.dthetadt_sgs(i, j, k) = 1.0f;
                }

    storage.runtime.turbulence.realized_transport_tendency_current = true;
    const SimulationStateView state = make_simulation_state_view(storage);
    const SgsExplicitStability result =
        compute_sgs_explicit_stability(state, tendencies);
    REQUIRE(result.open_boundary_work_accounted);
    REQUIRE(result.momentum_boundary_work_w == 0.0);
    REQUIRE(result.momentum_irreversible_dissipation_w > 0.0);
    REQUIRE(std::isfinite(result.maximum_timestep_s));
    REQUIRE(result.maximum_timestep_s > 0.0);

    double physical_u_before = 0.0;
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 1; k < n - 1; ++k)
                physical_u_before += storage.fields.u(i, j, k);
    set_primary_state_repairs_enabled(false);
    (void)apply_sgs_tendencies(
        state, 0.25 * result.maximum_timestep_s, tendencies);
    set_primary_state_repairs_enabled(true);
    double physical_u_after = 0.0;
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 1; k < n - 1; ++k)
                physical_u_after += storage.fields.u(i, j, k);
    REQUIRE(physical_u_after != physical_u_before);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                if (i == 0 || i == n - 1 || j == 0 || j == n - 1
                    || k == 0 || k == n - 1)
                {
                    REQUIRE(storage.fields.u(i, j, k) == closure);
                    REQUIRE(storage.fields.v(i, j, k) == closure);
                    REQUIRE(storage.fields.w(i, j, k) == closure);
                    REQUIRE(storage.fields.theta(i, j, k) == closure);
                }
}

TEST_CASE("open-boundary SGS separates boundary work from irreversible loss",
          "[turbulence][sgs][stability][open-boundary][energy]")
{
    constexpr int nr = 5, nth = 5, nz = 3;
    SimulationStorageState storage{{nr, nth, nz, 10.0, 20.0, 1.0, 15.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 10.0, 20.0, 15.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 0.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    storage.base.rho0.assign(nz, 1.0);
    // Equal interior face winds with faster exterior closure values describe
    // mechanical input through the two open x faces. Keep y/z uniform so the
    // analytical sign is not contaminated by another stress row.
    for (int j = 0; j < nth; ++j)
        for (int k = 0; k < nz; ++k)
        {
            storage.fields.u(0, j, k) = 10.0f;
            storage.fields.u(1, j, k) = 1.0f;
            storage.fields.u(2, j, k) = 1.0f;
            storage.fields.u(3, j, k) = 10.0f;
        }

    TurbulenceTendencies& tendencies =
        storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    auto scheme = std::make_unique<SmagorinskySchemeMarker>();
    scheme->viscosity.resize(nr, nth, nz, 1.0f);
    storage.runtime.turbulence.scheme = std::move(scheme);
    storage.runtime.turbulence.config.mode = "horizontal_only";
    global_lateral_bc_west = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_east = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_south = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_north = tmv::init::config::LateralBC::Radiative;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    const SimulationStateView state = make_simulation_state_view(storage);
    tmv::TurbulenceStateView turbulence_state;
    turbulence_state.u = &storage.fields.u;
    turbulence_state.v = &storage.fields.v;
    turbulence_state.w = &storage.fields.w;
    turbulence_state.rho = &storage.fields.rho;
    turbulence_state.theta = &storage.fields.theta;
    turbulence_state.rho0 = &storage.base.rho0;
    turbulence_state.NR = nr;
    turbulence_state.NTH = nth;
    turbulence_state.NZ = nz;
    tmv::GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 20.0;
    metrics.dz.assign(nz, 15.0);
    metrics.z_int = {0.0, 15.0, 30.0, 45.0};
    tmv::eddy_viscosity::compute_momentum_diffusion_tendencies(
        storage.runtime.turbulence.config, turbulence_state, metrics,
        *storage.runtime.turbulence.scheme->realized_eddy_viscosity(),
        tendencies.dudt_sgs, tendencies.dvdt_sgs, tendencies.dwdt_sgs);
    storage.runtime.turbulence.realized_transport_tendency_current = true;
    const SgsExplicitStability result =
        compute_sgs_explicit_stability(state, tendencies);
    REQUIRE(result.open_boundary_work_accounted);
    REQUIRE(result.momentum_irreversible_dissipation_w > 0.0);
    REQUIRE(result.momentum_boundary_work_w >
            result.momentum_irreversible_dissipation_w);
    REQUIRE(std::isfinite(result.momentum_energy_timestep_s));
    REQUIRE(result.momentum_energy_timestep_s > 0.0);

    // Merely pairing an unrelated positive tendency with the same viscosity
    // must not let it masquerade as boundary traction.
    TurbulenceTendencies forged = tendencies;
    forged.dudt_sgs.fill(0.0f);
    forged.dvdt_sgs.fill(0.0f);
    forged.dwdt_sgs.fill(0.0f);
    forged.dudt_sgs(1, 2, 0) = 1.0f;
    REQUIRE_THROWS_WITH(
        compute_sgs_explicit_stability(state, forged),
        Catch::Contains("not dissipative"));
}

TEST_CASE("Parallel SGS validation preserves serial error precedence without mutation",
          "[turbulence][sgs][stability][validation][thread]")
{
#ifdef _OPENMP
    struct RestoreWorkers
    {
        int count = omp_get_max_threads(), dynamic = omp_get_dynamic();
        ~RestoreWorkers() { omp_set_num_threads(count); omp_set_dynamic(dynamic); }
    } restore;
    omp_set_dynamic(0);
    omp_set_num_threads(GENERATE(1, 4));
#endif
    constexpr int n = 44;
    SimulationStorageState storage{{n, n, n, 100.0, 50.0, 1.0, 100.0}};
    storage.grid.geometry.initialize(n, n, n, 100.0, 50.0, 100.0,
                                     CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto& f = storage.fields;
    for (auto* field : {&f.rho, &f.u, &f.v, &f.w, &f.theta, &f.qv, &f.tke,
                        &storage.tendencies.dtke_dt_pbl})
        field->resize(n, n, n, 0);
    f.rho.resize(n, n, n, 1); f.theta.resize(n, n, n, 300);
    f.qv.resize(n, n, n, .01f); f.tke.resize(n, n, n, .02f);
    TurbulenceTendencies tendency;
    resize_tendencies(tendency, n, n, n);
    for (auto* field : {&tendency.dudt_sgs, &tendency.dvdt_sgs, &tendency.dwdt_sgs,
                        &tendency.dthetadt_sgs, &tendency.dqvdt_sgs, &tendency.dtkedt_sgs})
        std::fill(field->data(), field->data() + field->size(), 0);
    const auto state = make_simulation_state_view(storage);
    REQUIRE_NOTHROW(compute_sgs_explicit_stability(state, tendency));
    tendency.dudt_sgs(n - 2, n - 2, n - 2) = std::numeric_limits<float>::quiet_NaN();
    f.rho(1, 1, 1) = 0;
    const auto density_before = f.rho;
    const auto wind_before = f.u;
    REQUIRE_THROWS_WITH(compute_sgs_explicit_stability(state, tendency),
                       "SGS stability evaluation requires positive live density");
    CHECK(std::equal(f.rho.data(), f.rho.data() + f.rho.size(), density_before.data()));
    CHECK(std::equal(f.u.data(), f.u.data() + f.u.size(), wind_before.data()));
    f.rho(1, 1, 1) = 1; f.theta(1, 1, 1) = 0;
    REQUIRE_THROWS_WITH(compute_sgs_explicit_stability(state, tendency),
                       "SGS stability evaluation requires positive theta");
    f.theta(1, 1, 1) = 300; f.qv(1, 1, 1) = -.01f;
    REQUIRE_THROWS_WITH(compute_sgs_explicit_stability(state, tendency),
                       "SGS stability evaluation requires finite non-negative moisture and a finite tendency");
    f.qv(1, 1, 1) = .01f; f.tke(1, 1, 1) = -.02f;
    REQUIRE_THROWS_WITH(compute_sgs_explicit_stability(state, tendency),
                       "SGS stability evaluation requires finite non-negative TKE and finite tendencies");
    f.tke(1, 1, 1) = .02f;
    REQUIRE_THROWS_WITH(compute_sgs_explicit_stability(state, tendency),
                       "SGS stability evaluation requires finite physical state and tendencies");
}

TEST_CASE("deterministic SGS stability provenance is exact across thread counts",
          "[turbulence][sgs][stability][open-boundary][determinism]")
{
#ifdef _OPENMP
    constexpr int nr = 38, nth = 34, nz = 28;
    SimulationStorageState storage{{nr, nth, nz, 10.0, 20.0, 1.0, 15.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 10.0, 20.0, 15.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz);
    storage.fields.u.resize(nr, nth, nz, 1.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    storage.base.rho0.assign(nz, 1.0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                storage.fields.rho(i, j, k) = static_cast<float>(
                    0.9 + 0.003 * i + 0.002 * j + 0.001 * k);
                if (i == 0 || i == nr - 2)
                    storage.fields.u(i, j, k) = static_cast<float>(
                        9.0 + 0.002 * j + 0.001 * k);
            }

    auto scheme = std::make_unique<SmagorinskySchemeMarker>();
    scheme->viscosity.resize(nr, nth, nz);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                scheme->viscosity(i, j, k) = static_cast<float>(
                    0.8 + 0.001 * i + 0.0005 * j + 0.00025 * k);
    storage.runtime.turbulence.scheme = std::move(scheme);
    storage.runtime.turbulence.config.mode = "horizontal_only";
    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);

    global_lateral_bc_west = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_east = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_south = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_north = tmv::init::config::LateralBC::Radiative;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    tmv::TurbulenceStateView turbulence_state;
    turbulence_state.u = &storage.fields.u;
    turbulence_state.v = &storage.fields.v;
    turbulence_state.w = &storage.fields.w;
    turbulence_state.rho = &storage.fields.rho;
    turbulence_state.theta = &storage.fields.theta;
    turbulence_state.rho0 = &storage.base.rho0;
    turbulence_state.NR = nr;
    turbulence_state.NTH = nth;
    turbulence_state.NZ = nz;
    tmv::GridMetrics metrics;
    metrics.dx = 10.0;
    metrics.dy = 20.0;
    metrics.dz.assign(nz, 15.0);
    metrics.z_int.resize(static_cast<std::size_t>(nz) + 1);
    for (int k = 0; k <= nz; ++k)
        metrics.z_int[static_cast<std::size_t>(k)] = 15.0 * k;
    tmv::eddy_viscosity::compute_momentum_diffusion_tendencies(
        storage.runtime.turbulence.config, turbulence_state, metrics,
        *storage.runtime.turbulence.scheme->realized_eddy_viscosity(),
        tendencies.dudt_sgs, tendencies.dvdt_sgs, tendencies.dwdt_sgs);
    storage.runtime.turbulence.realized_transport_tendency_current = true;
    const SimulationStateView state = make_simulation_state_view(storage);

    const int previous_dynamic = omp_get_dynamic();
    const int previous_threads = omp_get_max_threads();
    const bool previous_deterministic =
        tmv::diagnostics::deterministic_reductions_enabled();
    omp_set_dynamic(0);
    tmv::diagnostics::set_deterministic_reductions(true);
    omp_set_num_threads(1);
    const SgsExplicitStability serial =
        compute_sgs_explicit_stability(state, tendencies);
    omp_set_num_threads(4);
    const SgsExplicitStability parallel =
        compute_sgs_explicit_stability(state, tendencies);
    omp_set_num_threads(previous_threads);
    omp_set_dynamic(previous_dynamic);
    tmv::diagnostics::set_deterministic_reductions(previous_deterministic);

    REQUIRE(serial.open_boundary_work_accounted);
    REQUIRE(serial.momentum_irreversible_dissipation_w > 0.0);
    REQUIRE(serial.maximum_timestep_s == parallel.maximum_timestep_s);
    REQUIRE(serial.momentum_energy_timestep_s ==
            parallel.momentum_energy_timestep_s);
    REQUIRE(serial.momentum_irreversible_dissipation_w ==
            parallel.momentum_irreversible_dissipation_w);
    REQUIRE(serial.momentum_boundary_work_w ==
            parallel.momentum_boundary_work_w);
    REQUIRE(serial.momentum_stress_closure_residual_w ==
            parallel.momentum_stress_closure_residual_w);
    REQUIRE(serial.open_boundary_work_accounted ==
            parallel.open_boundary_work_accounted);
    REQUIRE(serial.theta_monotonic_timestep_s ==
            parallel.theta_monotonic_timestep_s);
    REQUIRE(serial.qv_monotonic_timestep_s ==
            parallel.qv_monotonic_timestep_s);
    REQUIRE(serial.tke_positivity_timestep_s ==
            parallel.tke_positivity_timestep_s);
#else
    SUCCEED("OpenMP is unavailable; deterministic serial path remains covered");
#endif
}

TEST_CASE("open-boundary SGS traction identity closes every direction mode",
          "[turbulence][sgs][stability][open-boundary][traction]")
{
    constexpr int nr = 7, nth = 6, nz = 5;
    SimulationStorageState storage{{nr, nth, nz, 80.0, 20.0, 1.0, 120.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 80.0, 20.0, 120.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.grid.geometry.z_faces = {0.0, 10.0, 25.0, 45.0, 70.0, 100.0};
    storage.grid.geometry.z = {5.0, 17.5, 35.0, 57.5, 85.0};
    storage.fields.rho.resize(nr, nth, nz);
    storage.fields.u.resize(nr, nth, nz);
    storage.fields.v.resize(nr, nth, nz);
    storage.fields.w.resize(nr, nth, nz);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    storage.base.rho0.assign(nz, 1.0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                storage.fields.rho(i, j, k) = static_cast<float>(
                    1.0 + 0.02 * i + 0.01 * j + 0.03 * k);
                storage.fields.u(i, j, k) = static_cast<float>(
                    0.4 * i - 0.2 * j + 0.3 * k + 0.03 * i * j);
                storage.fields.v(i, j, k) = static_cast<float>(
                    -0.1 * i + 0.35 * j - 0.15 * k + 0.02 * j * k);
                storage.fields.w(i, j, k) = static_cast<float>(
                    0.12 * i - 0.08 * j + 0.2 * k + 0.01 * i * k);
            }

    auto scheme = std::make_unique<SmagorinskySchemeMarker>();
    scheme->viscosity.resize(nr, nth, nz);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                scheme->viscosity(i, j, k) = static_cast<float>(
                    0.5 + 0.04 * i + 0.03 * j + 0.02 * k);
    storage.runtime.turbulence.scheme = std::move(scheme);
    auto& tendencies = storage.runtime.turbulence.held_tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    global_lateral_bc_west = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_east = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_south = tmv::init::config::LateralBC::Radiative;
    global_lateral_bc_north = tmv::init::config::LateralBC::Radiative;

    tmv::TurbulenceStateView turbulence_state;
    turbulence_state.u = &storage.fields.u;
    turbulence_state.v = &storage.fields.v;
    turbulence_state.w = &storage.fields.w;
    turbulence_state.rho = &storage.fields.rho;
    turbulence_state.theta = &storage.fields.theta;
    turbulence_state.rho0 = &storage.base.rho0;
    turbulence_state.NR = nr;
    turbulence_state.NTH = nth;
    turbulence_state.NZ = nz;
    tmv::GridMetrics metrics;
    metrics.dx = 80.0;
    metrics.dy = 120.0;
    metrics.dz = {10.0, 15.0, 20.0, 25.0, 30.0};
    metrics.z_int = storage.grid.geometry.z_faces;
    const SimulationStateView state = make_simulation_state_view(storage);

    for (const std::string mode : {
             "horizontal_only", "vertical_only", "3d"})
    {
        DYNAMIC_SECTION("mode=" << mode)
        {
            storage.runtime.turbulence.config.mode = mode;
            tmv::eddy_viscosity::compute_momentum_diffusion_tendencies(
                storage.runtime.turbulence.config, turbulence_state, metrics,
                *storage.runtime.turbulence.scheme->realized_eddy_viscosity(),
                tendencies.dudt_sgs, tendencies.dvdt_sgs,
                tendencies.dwdt_sgs);
            storage.runtime.turbulence.realized_transport_tendency_current =
                true;
            const SgsExplicitStability result =
                compute_sgs_explicit_stability(state, tendencies);
            REQUIRE(result.open_boundary_work_accounted);
            REQUIRE(result.momentum_irreversible_dissipation_w > 0.0);
            const double scale = std::max(
                {1.0, std::abs(result.momentum_boundary_work_w),
                 result.momentum_irreversible_dissipation_w});
            REQUIRE(std::abs(result.momentum_stress_closure_residual_w)
                    <= 128.0 * std::numeric_limits<float>::epsilon() * scale);
        }
    }

    // An upper open face is an independent traction boundary. It must enter
    // the same summation-by-parts identity even when both lateral directions
    // are periodic and therefore contribute no exterior stress work.
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    global_top_bc = tmv::init::config::TopBC::Open;
    for (const std::string mode : {"vertical_only", "3d"})
    {
        DYNAMIC_SECTION("top-only mode=" << mode)
        {
            storage.runtime.turbulence.config.mode = mode;
            tmv::eddy_viscosity::compute_momentum_diffusion_tendencies(
                storage.runtime.turbulence.config, turbulence_state, metrics,
                *storage.runtime.turbulence.scheme->realized_eddy_viscosity(),
                tendencies.dudt_sgs, tendencies.dvdt_sgs,
                tendencies.dwdt_sgs);
            storage.runtime.turbulence.realized_transport_tendency_current =
                true;
            const SgsExplicitStability result =
                compute_sgs_explicit_stability(state, tendencies);
            REQUIRE(result.open_boundary_work_accounted);
            REQUIRE(result.momentum_irreversible_dissipation_w > 0.0);
            const double scale = std::max(
                {1.0, std::abs(result.momentum_boundary_work_w),
                 result.momentum_irreversible_dissipation_w});
            REQUIRE(std::abs(result.momentum_stress_closure_residual_w)
                    <= 128.0 * std::numeric_limits<float>::epsilon() * scale);
        }
    }
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

TEST_CASE("SGS local substeps publish one outer-step work ledger",
          "[turbulence][sgs][stability][ledger][sci086]")
{
    constexpr int nr = 5, nth = 5, nz = 3;
    SimulationStorageState storage{{nr, nth, nz, 10.0, 20.0, 1.0, 15.0}};
    storage.grid.geometry.initialize(
        nr, nth, nz, 10.0, 20.0, 15.0,
        CoordinateSystem::Cartesian, StaggerType::CGrid);
    storage.fields.rho.resize(nr, nth, nz, 1.0f);
    storage.fields.u.resize(nr, nth, nz, 2.0f);
    storage.fields.v.resize(nr, nth, nz, 0.0f);
    storage.fields.w.resize(nr, nth, nz, 0.0f);
    storage.fields.theta.resize(nr, nth, nz, 300.0f);
    storage.runtime.turbulence.scheme = std::make_unique<LedgerScheme>();

    TurbulenceTendencies tendencies;
    tendencies.dudt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dwdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dthetadt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dqvdt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dtkedt_sgs.resize(nr, nth, nz, 0.0f);
    tendencies.dudt_sgs(1, 1, 0) = -0.25f;

    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    set_primary_state_repairs_enabled(false);
    const SimulationStateView state = make_simulation_state_view(storage);
    TurbulenceMomentumWorkReport aggregate;
    invalidated_resident_fields.clear();
    for (int substep = 0; substep < 2; ++substep)
    {
        const auto report =
            apply_admitted_sgs_substep(state, 0.25, tendencies);
        aggregate.requested_linear_work_j += report.requested_linear_work_j;
        aggregate.finite_step_correction_j +=
            report.finite_step_correction_j;
        aggregate.storage_rounding_j += report.storage_rounding_j;
        aggregate.committed_work_j += report.committed_work_j;
    }
    const std::vector<std::string> expected_invalidations{
        "u", "v", "w", "theta", "qv", "tke",
        "u", "v", "w", "theta", "qv", "tke"
    };
    REQUIRE(invalidated_resident_fields == expected_invalidations);
    REQUIRE(storage.runtime.turbulence.momentum_work_step_count == 0);
    commit_sgs_momentum_work(state, aggregate);
    REQUIRE(storage.runtime.turbulence.momentum_work_step_count == 1);
    REQUIRE(storage.runtime.turbulence.last_momentum_work.committed_work_j ==
            Approx(aggregate.committed_work_j));
    REQUIRE(storage.runtime.turbulence.cumulative_committed_momentum_work_j ==
            Approx(aggregate.committed_work_j));
    set_primary_state_repairs_enabled(true);
}
