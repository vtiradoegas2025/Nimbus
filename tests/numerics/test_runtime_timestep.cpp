#include "catch2/catch.hpp"

#include "core/runtime/simulation.hpp"
#include "core/runtime/simulation_state.hpp"
#include "numerics/advection/factory.hpp"
#include "numerics/diffusion/factory.hpp"
#include "numerics/time_stepping/factory.hpp"
#include "util/log_profile.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <string>
#include <vector>
#include <cstring>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

// This is a deliberately small integration harness for numerics.cpp.  The
// regular test harness owns the same numerics globals as that production
// translation unit, so this target supplies only the runtime symbols actually
// consumed by choose_runtime_timestep().
SimulationStorageState global_simulation_storage{{4, 4, 4,
                                                  100.0, 400.0, 1.0, 200.0}};
int& NR = global_simulation_storage.grid.nr;
int& NTH = global_simulation_storage.grid.nth;
int& NZ = global_simulation_storage.grid.nz;
double& dr = global_simulation_storage.grid.dx;
double& dtheta = global_simulation_storage.grid.dy;
double& dz = global_simulation_storage.grid.dz;
double& dt = global_simulation_storage.grid.dt;
double& simulation_time = global_simulation_storage.time_s;

Field3D& u = global_simulation_storage.fields.u;
Field3D& v = global_simulation_storage.fields.v;
Field3D& w = global_simulation_storage.fields.w;
Field3D& rho = global_simulation_storage.fields.rho;
Field3D& p = global_simulation_storage.fields.p;
Field3D& theta = global_simulation_storage.fields.theta;
Field3D& qv = global_simulation_storage.fields.qv;

std::vector<double>& p0_base = global_simulation_storage.base.p0;
std::vector<double>& rho0_base = global_simulation_storage.base.rho0;

CoordinateSystem& global_coordinate_system =
    global_simulation_storage.configuration.coordinate_system;
StaggerType& global_stagger_type =
    global_simulation_storage.configuration.stagger_type;
tmv::init::config::LateralBC& global_lateral_bc_west =
    global_simulation_storage.configuration.lateral_bc_west;
tmv::init::config::LateralBC& global_lateral_bc_east =
    global_simulation_storage.configuration.lateral_bc_east;
tmv::init::config::LateralBC& global_lateral_bc_south =
    global_simulation_storage.configuration.lateral_bc_south;
tmv::init::config::LateralBC& global_lateral_bc_north =
    global_simulation_storage.configuration.lateral_bc_north;
LogProfile& global_log_profile =
    (global_simulation_storage.configuration.log_profile = LogProfile::quiet);

namespace tmv
{
TerrainRuntimeState& global_terrain_runtime_state =
    global_simulation_storage.runtime.terrain;
std::unique_ptr<TerrainSchemeBase>& terrain_scheme =
    global_terrain_runtime_state.scheme;
TerrainConfig& global_terrain_config = global_terrain_runtime_state.config;
Topography2D& global_topography = global_terrain_runtime_state.topography;
TerrainMetrics3D& global_terrain_metrics = global_terrain_runtime_state.metrics;
AdvectionRuntimeState& global_advection_runtime_state =
    global_simulation_storage.runtime.advection;
DiffusionRuntimeState& global_diffusion_runtime_state =
    global_simulation_storage.runtime.diffusion;
TimeSteppingRuntimeState& global_time_stepping_runtime_state =
    global_simulation_storage.runtime.time_stepping;
TurbulenceRuntimeState& global_turbulence_runtime_state =
    global_simulation_storage.runtime.turbulence;

// initialize_numerics() is compiled into the same translation unit but is not
// exercised here.  These stubs keep this focused target independent of every
// scheme implementation while preserving the production function's linkage.
std::unique_ptr<AdvectionSchemeBase> create_advection_scheme(const std::string&)
{
    return nullptr;
}

std::unique_ptr<DiffusionSchemeBase> create_diffusion_scheme(const std::string&)
{
    return nullptr;
}

std::unique_ptr<TimeSteppingSchemeBase> create_time_stepping_scheme(const std::string&)
{
    return nullptr;
}

void validate_runtime_time_stepping_contract(const TimeSteppingConfig&,
                                             const TimeSteppingSchemeBase&)
{
}
} // namespace tmv

namespace
{
class RecordingExplicitDiffusion final : public tmv::DiffusionSchemeBase
{
public:
    mutable bool called = false;
    mutable bool saw_c_grid = false;
    mutable bool saw_periodic = false;
    mutable bool saw_live_fields = false;

    std::string name() const override { return "explicit"; }
    void initialize() override {}
    void initialize(const DiffusionConfig&) override {}
    void compute_diffusion_tendencies(
        const DiffusionConfig&, const DiffusionStateView&,
        DiffusionTendencies&, DiffusionDiagnostics*) override {}
    double check_stability(
        const DiffusionConfig& cfg,
        const DiffusionStateView& state) override
    {
        called = true;
        saw_c_grid = state.c_grid;
        saw_periodic = state.periodic_x && state.periodic_y;
        saw_live_fields = state.rho == &rho && state.theta == &theta &&
                          state.qv == &qv && state.u == &u &&
                          state.v == &v && state.w == &w;
        REQUIRE(cfg.dt_diffusion == Approx(1.0));
        return 0.25;
    }
};

class ActiveTurbulenceStub final : public tmv::TurbulenceSchemeBase
{
public:
    std::string name() const override { return "smagorinsky"; }
    int required_fields() const override { return 0; }
    void initialize(const TurbulenceConfig&) override {}
    void compute(const TurbulenceConfig&, const GridMetrics&,
                 const TurbulenceStateView&, TurbulenceTendencies&,
                 TurbulenceDiagnostics*) override {}
};

class PrescribedFallSpeed final : public tmv::MicrophysicsScheme
{
public:
    std::string get_scheme_name() const override { return "prescribed_fall_speed"; }
    int get_num_prognostic_vars() const override { return 3; }
    double explicit_terminal_velocity_mps(std::size_t species, double, double) const override
    { return species == 0 ? 20.0 : 0.0; }
    void compute_tendencies(
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, const Field3D&, const Field3D&,
        const Field3D&, const Field3D&, double,
        Field3D&, Field3D&, Field3D&, Field3D&, Field3D&, Field3D&, Field3D&, Field3D&) override {}
};

void reset_runtime_timestep_fixture(float u_value, float v_value, float w_value,
                                    int nx = 4, int ny = 4, int levels = 4)
{
    NR = nx;
    NTH = ny;
    NZ = levels;
    dr = 100.0;
    dtheta = 200.0;
    dz = 400.0;
    dt = 10.0;
    simulation_time = 0.0;
    global_simulation_storage.runtime.microphysics.scheme.reset();

    u.resize(NR, NTH, NZ, u_value);
    v.resize(NR, NTH, NZ, v_value);
    w.resize(NR, NTH, NZ, w_value);
    p.resize(NR, NTH, NZ, 100000.0f);
    theta.resize(NR, NTH, NZ, 300.0f);
    qv.resize(NR, NTH, NZ, 0.01f);
    rho.resize(NR, NTH, NZ,
               static_cast<float>(physical_constants::dry_air_gamma * 100000.0 / (300.0 * 300.0)));

    tmv::global_grid_metrics.dx = dr;
    tmv::global_grid_metrics.dy = dtheta;
    tmv::global_grid_metrics.dz.assign(NZ, dz);
    tmv::global_grid_metrics.terrain_metrics_active = false;
    tmv::global_grid_metrics.terrain_metrics = nullptr;
    tmv::global_grid_metrics.terrain_topography = nullptr;
    tmv::global_terrain_config.scheme_id = "none";
    tmv::global_terrain_metrics = TerrainMetrics3D{};
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_stagger_type = StaggerType::CGrid;

    tmv::advection_scheme.reset();
    tmv::diffusion_scheme.reset();
    tmv::time_stepping_scheme.reset();
    tmv::global_time_stepping_config = TimeSteppingConfig{};
    tmv::global_time_stepping_config.cfl_safety = 0.8;
    tmv::global_time_stepping_config.dt_min = 1.0e-6;
    tmv::global_time_stepping_config.dt_max = 100.0;
    tmv::global_time_stepping_config.adaptive_dt = false;
    tmv::global_time_stepping_config.split_acoustic = true;
    global_simulation_storage.runtime.turbulence.scheme.reset();
    global_simulation_storage.runtime.turbulence
        .last_explicit_stability_limit_s =
        std::numeric_limits<double>::infinity();
    global_simulation_storage.runtime.turbulence
        .last_explicit_substep_count = 0;

    p0_base.clear();
    rho0_base.clear();
}

double density_for_sound_speed(double pressure_pa, double sound_speed_ms)
{
    return physical_constants::dry_air_gamma * pressure_pa / (sound_speed_ms * sound_speed_ms);
}

void configure_uniform_mapped_terrain(double jacobian,
                                      double slope_x,
                                      double slope_y)
{
    REQUIRE(jacobian > 0.0);
    auto& metrics = tmv::global_terrain_metrics;
    metrics = TerrainMetrics3D{};
    metrics.dx = dr;
    metrics.dy = dtheta;
    metrics.dzeta = dz;
    metrics.z.resize(NR, NTH, NZ, 0.0);
    metrics.J.resize(NR, NTH, NZ, jacobian);
    metrics.mx.resize(NR, NTH, NZ, -slope_x / jacobian);
    metrics.my.resize(NR, NTH, NZ, -slope_y / jacobian);
    metrics.zeta.resize(NR, NTH, NZ, 0.0);
    metrics.z_face.resize(NR, NTH, NZ + 1, 0.0);
    metrics.Jx_face.resize(NR + 1, NTH, NZ, jacobian);
    metrics.Jy_face.resize(NR, NTH + 1, NZ, jacobian);
    metrics.Gx_face.resize(NR, NTH, NZ + 1, -slope_x);
    metrics.Gy_face.resize(NR, NTH, NZ + 1, -slope_y);

    const double layer_thickness = jacobian * dz;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
        {
            const double ground = slope_x * i * dr +
                                  slope_y * j * dtheta;
            for (int k = 0; k <= NZ; ++k)
                metrics.z_face(i, j, k) = ground + k * layer_thickness;
            for (int k = 0; k < NZ; ++k)
            {
                metrics.z(i, j, k) = ground +
                    (static_cast<double>(k) + 0.5) * layer_thickness;
                metrics.zeta(i, j, k) =
                    (static_cast<double>(k) + 0.5) * dz;
            }
        }

    tmv::global_terrain_config.scheme_id = "bell";
}
} // namespace

TEST_CASE("Large CFL validation preserves the timestep and rejects bad storage without mutation",
          "[numerics][timestep][cfl][parallel-validation]")
{
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    struct RestoreThreads { int value; ~RestoreThreads() { omp_set_num_threads(value); } } restore{previous_threads};
#endif
    reset_runtime_timestep_fixture(10.0f, 10.0f, 10.0f, 65, 49, 33);
    const std::array<Field3D*, 5> fields{&u, &v, &w, &p, &rho};
    std::array<std::vector<float>, 5> originals;
    for (std::size_t field = 0; field < fields.size(); ++field)
        originals[field].assign(fields[field]->data(), fields[field]->data() + fields[field]->size());
    double reference = 0.0;
    for (int workers : {1, 2, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(workers);
#endif
        const double selected = tmv::choose_runtime_timestep();
        if (workers == 1) reference = selected;
        REQUIRE(selected == reference);
        REQUIRE(selected == Approx(32.0 / 7.0).epsilon(1.0e-12));
        REQUIRE(tmv::last_max_flow_speed() == 10.0);
        for (std::size_t field = 0; field < fields.size(); ++field)
        {
            const std::array<std::size_t, 4> positions{0, 1, fields[field]->size()/2, fields[field]->size()-1};
            for (std::size_t index : positions)
            {
                const std::array<float, 3> bad{std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(), field >= 3 ? 0.0f : -std::numeric_limits<float>::infinity()};
                for (float value : bad)
                {
                    CAPTURE(workers, field, index);
                    fields[field]->data()[index] = value;
                    const auto invalid = *fields[field];
                    REQUIRE_THROWS_WITH(tmv::choose_runtime_timestep(),
                        "CFL scan requires finite wind and positive finite pressure/density");
                    REQUIRE(dt == 10.0);
                    REQUIRE(simulation_time == 0.0);
                    REQUIRE(tmv::last_max_flow_speed() == 10.0);
                    REQUIRE(std::memcmp(fields[field]->data(), invalid.data(), invalid.size()*sizeof(float)) == 0);
                    fields[field]->data()[index] = originals[field][index];
                }
            }
        }
        for (std::size_t field = 0; field < fields.size(); ++field)
            REQUIRE(std::memcmp(fields[field]->data(), originals[field].data(), originals[field].size()*sizeof(float)) == 0);
    }
}

TEST_CASE("Full-grid public CFL selector benchmark", "[.cfl-benchmark]")
{
    const char* requested = std::getenv("NIMBUS_CFL_BENCHMARK_THREADS");
    const int workers = requested ? std::stoi(requested) : 1;
    REQUIRE(workers >= 1);
    REQUIRE(workers <= 64);
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    struct RestoreThreads { int value; ~RestoreThreads() { omp_set_num_threads(value); } } restore{previous_threads};
    omp_set_num_threads(workers);
#endif
    reset_runtime_timestep_fixture(10.0f, 10.0f, 10.0f, 640, 640, 180);
    tmv::global_grid_metrics.dx = dr = 50.0;
    tmv::global_grid_metrics.dy = dtheta = 50.0;
    tmv::global_grid_metrics.dz.assign(NZ, 100.0);
    dz = 100.0;
    const double reference = tmv::choose_runtime_timestep();
    REQUIRE(reference == Approx(1.6).epsilon(1.0e-12));
    for (int warmup = 0; warmup < 2; ++warmup) REQUIRE(tmv::choose_runtime_timestep() == reference);
    std::array<double, 8> milliseconds{};
    for (double& elapsed : milliseconds)
    {
        const auto start = std::chrono::steady_clock::now();
        const double selected = tmv::choose_runtime_timestep();
        const auto stop = std::chrono::steady_clock::now();
        REQUIRE(selected == reference);
        elapsed = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    std::cout << std::setprecision(17) << "NIMBUS_CFL_BENCHMARK {\"threads\":" << workers
              << ",\"shape\":[640,640,180],\"selected_dt\":" << reference << ",\"milliseconds\":[";
    for (std::size_t index = 0; index < milliseconds.size(); ++index)
        std::cout << (index ? "," : "") << milliseconds[index];
    std::cout << "]}\n";
}

TEST_CASE("Parallel CFL validation preserves mapped-grid selection and boundary rejection",
          "[numerics][timestep][cfl][terrain][parallel-validation]")
{
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
    struct RestoreThreads { int value; ~RestoreThreads() { omp_set_num_threads(value); } } restore{previous_threads};
#endif
    for (const auto staggering : {StaggerType::CGrid, StaggerType::Collocated})
    {
        reset_runtime_timestep_fixture(10.0f, 4.0f, 2.0f, 65, 49, 33);
        global_stagger_type = staggering;
        configure_uniform_mapped_terrain(.8, .1, .2);
        tmv::refresh_grid_metrics_from_terrain();
        double reference = 0.0;
        for (int workers : {1, 4})
        {
#ifdef _OPENMP
            omp_set_num_threads(workers);
#endif
            const double selected = tmv::choose_runtime_timestep();
            if (workers == 1) reference = selected;
            REQUIRE(selected == reference);
            REQUIRE(selected > 0.0);
            REQUIRE(std::isfinite(selected));
            const float prior = p.data()[p.size()-1];
            p.data()[p.size()-1] = -1.0f;
            REQUIRE_THROWS_WITH(tmv::choose_runtime_timestep(),
                "CFL scan requires finite wind and positive finite pressure/density");
            REQUIRE(p.data()[p.size()-1] == -1.0f);
            REQUIRE(dt == 10.0);
            REQUIRE(tmv::last_max_flow_speed() == 10.0);
            p.data()[p.size()-1] = prior;
        }
    }
}

TEST_CASE("Runtime timestep uses the live explicit diffusion stability operator",
          "[numerics][timestep][diffusion][live-density]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    global_stagger_type = StaggerType::CGrid;
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    tmv::global_diffusion_config = DiffusionConfig{};
    tmv::global_diffusion_config.K_h = 1.0;
    tmv::global_diffusion_config.K_v = 2.0;
    tmv::global_diffusion_config.apply_to = "scalars_and_momentum";
    auto recording = std::make_unique<RecordingExplicitDiffusion>();
    RecordingExplicitDiffusion* observed = recording.get();
    tmv::diffusion_scheme = std::move(recording);

    REQUIRE(tmv::choose_runtime_timestep() == Approx(3.6).epsilon(1.0e-12));
    REQUIRE(observed->called);
    REQUIRE(observed->saw_c_grid);
    REQUIRE(observed->saw_periodic);
    REQUIRE(observed->saw_live_fields);
}

TEST_CASE("Runtime timestep uses a multidimensional directional CFL sum",
          "[numerics][timestep][cfl][cartesian]")
{
    reset_runtime_timestep_fixture(10.0f, 10.0f, 10.0f);

    // Directional rate = 10/100 + 10/200 + 10/400 = 0.175 s^-1.
    // The previous max-component/min-spacing approximation returned 8 s;
    // the multidimensional transport limit is 0.8/0.175 = 32/7 s.
    const double limited = tmv::choose_runtime_timestep();

    REQUIRE(limited == Approx(32.0 / 7.0).epsilon(1.0e-12));
    REQUIRE(tmv::last_max_flow_speed() == Approx(10.0));
}

TEST_CASE("Flat metric refresh preserves exact stretched checkpoint geometry",
          "[numerics][metrics][restart][stretched]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    global_simulation_storage.grid.geometry.z_faces =
        {0.0, 50.0, 150.0, 350.0, 800.0};
    global_simulation_storage.grid.geometry.z =
        {25.0, 100.0, 250.0, 575.0};

    REQUIRE_NOTHROW(tmv::refresh_grid_metrics_from_terrain());
    REQUIRE(tmv::global_grid_metrics.z_int ==
            global_simulation_storage.grid.geometry.z_faces);
    REQUIRE(tmv::global_grid_metrics.dz ==
            std::vector<double>{50.0, 100.0, 200.0, 450.0});
    REQUIRE_FALSE(tmv::global_grid_metrics.terrain_metrics_active);
}

TEST_CASE("Runtime timestep consumes the previous realized SGS bound",
          "[numerics][timestep][turbulence][adaptive]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    auto& turbulence = global_simulation_storage.runtime.turbulence;
    turbulence.scheme = std::make_unique<ActiveTurbulenceStub>();
    turbulence.last_explicit_stability_limit_s = 0.375;

    REQUIRE(tmv::choose_runtime_timestep() == Approx(0.375));
}

TEST_CASE("Terrain runtime CFL uses contravariant flow and face-normal spacing",
          "[numerics][timestep][cfl][terrain]")
{
    reset_runtime_timestep_fixture(10.0f, 0.0f, 10.0f);
    configure_uniform_mapped_terrain(0.1, 1.0, 0.0);
    tmv::refresh_grid_metrics_from_terrain();

    for (double layer : tmv::global_grid_metrics.dz)
        REQUIRE(layer == Approx(40.0));

    // (u,w)=(10,10) follows z=x exactly, so its contravariant vertical
    // velocity is zero. The sole advective rate is u/dx=0.1 s^-1.
    REQUIRE(tmv::choose_runtime_timestep() == Approx(8.0).epsilon(1.0e-12));
    REQUIRE(tmv::last_max_flow_speed() == Approx(10.0));

    // A zeta face with dz/dx=1 has normal spacing dz/sqrt(2), not nominal dz.
    REQUIRE(tmv::global_time_stepping_config.min_acoustic_spacing_m ==
            Approx(40.0 / std::sqrt(2.0)).epsilon(1.0e-12));
}

TEST_CASE("Unsplit terrain acoustic CFL resolves compressed sloping layers",
          "[numerics][timestep][cfl][terrain][acoustic]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    configure_uniform_mapped_terrain(0.1, 1.0, 0.0);
    tmv::refresh_grid_metrics_from_terrain();
    tmv::global_time_stepping_config.split_acoustic = false;

    const double limited = tmv::choose_runtime_timestep();
    const double spacing = 40.0 / std::sqrt(2.0);
    REQUIRE(tmv::global_time_stepping_config.min_acoustic_spacing_m ==
            Approx(spacing).epsilon(1.0e-12));
    REQUIRE(limited == Approx(
        tmv::global_time_stepping_config.cfl_safety * spacing /
        tmv::global_time_stepping_config.max_acoustic_speed_ms));
    REQUIRE(limited < 0.1);
}

TEST_CASE("Unsplit runtime timestep enforces the acoustic CFL cap",
          "[numerics][timestep][cfl][acoustic]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    dt = 1.0;

    constexpr double pressure_pa = 100000.0;
    p0_base = {pressure_pa, pressure_pa};
    rho0_base = {density_for_sound_speed(pressure_pa, 250.0),
                 density_for_sound_speed(pressure_pa, 320.0)};

    SECTION("unsplit dynamics is capped by the fastest base-state sound speed")
    {
        tmv::global_time_stepping_config.split_acoustic = false;
        // c_s,max = 320 m/s and min(dx,dy,dz) = 100 m.
        REQUIRE(tmv::choose_runtime_timestep() == Approx(0.25).epsilon(1.0e-12));
        REQUIRE(tmv::global_time_stepping_config.max_acoustic_speed_ms ==
                Approx(320.0).epsilon(1.0e-6));
    }

    SECTION("split-explicit dynamics leaves acoustic control to its substeps")
    {
        tmv::global_time_stepping_config.split_acoustic = true;
        REQUIRE(tmv::choose_runtime_timestep() == Approx(1.0).epsilon(1.0e-12));
    }

    SECTION("resolved state can set a faster acoustic speed than the base state")
    {
        constexpr double resolved_speed = 410.0;
        rho[2][2][2] = static_cast<float>(
            density_for_sound_speed(pressure_pa, resolved_speed));
        tmv::global_time_stepping_config.split_acoustic = true;

        REQUIRE(tmv::choose_runtime_timestep() == Approx(1.0).epsilon(1.0e-12));
        REQUIRE(tmv::global_time_stepping_config.max_acoustic_speed_ms ==
                Approx(resolved_speed).epsilon(1.0e-5));
    }
}

TEST_CASE("Runtime timestep rejects invalid state and metrics instead of substituting",
          "[numerics][timestep][config-truth][finite]")
{
    reset_runtime_timestep_fixture(1.0f, 2.0f, 3.0f);

    SECTION("invalid runtime dt")
    {
        dt = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS_AS(tmv::choose_runtime_timestep(),
                          std::invalid_argument);
    }
    SECTION("reversed adaptive bounds")
    {
        tmv::global_time_stepping_config.dt_min = 2.0;
        tmv::global_time_stepping_config.dt_max = 1.0;
        REQUIRE_THROWS_AS(tmv::choose_runtime_timestep(),
                          std::invalid_argument);
    }
    SECTION("invalid vertical metric")
    {
        tmv::global_grid_metrics.dz[2] = 0.0;
        REQUIRE_THROWS_AS(tmv::choose_runtime_timestep(),
                          std::invalid_argument);
    }
    SECTION("non-finite wind")
    {
        u[1][1][1] = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_THROWS_AS(tmv::choose_runtime_timestep(),
                          std::invalid_argument);
    }
    SECTION("non-positive thermodynamic state")
    {
        p[1][1][1] = 0.0f;
        REQUIRE_THROWS_AS(tmv::choose_runtime_timestep(),
                          std::invalid_argument);
    }
    SECTION("mismatched base profiles")
    {
        p0_base = {100000.0};
        rho0_base.clear();
        REQUIRE_THROWS_AS(tmv::choose_runtime_timestep(),
                          std::invalid_argument);
    }
}

TEST_CASE("Grid metric construction rejects invalid spacing",
          "[numerics][metrics][config-truth]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    tmv::global_terrain_config.scheme_id = "none";

    SECTION("dx")
    {
        dr = 0.0;
        REQUIRE_THROWS_AS(tmv::refresh_grid_metrics_from_terrain(),
                          std::invalid_argument);
    }
    SECTION("dy")
    {
        dtheta = std::numeric_limits<double>::infinity();
        REQUIRE_THROWS_AS(tmv::refresh_grid_metrics_from_terrain(),
                          std::invalid_argument);
    }
    SECTION("dz")
    {
        dz = -1.0;
        REQUIRE_THROWS_AS(tmv::refresh_grid_metrics_from_terrain(),
                          std::invalid_argument);
    }
}

TEST_CASE("terrain metric refresh rejects transactionally",
          "[numerics][metrics][terrain][transactional]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    tmv::global_grid_metrics.dx = 321.0;
    tmv::global_grid_metrics.dy = 654.0;
    tmv::global_grid_metrics.dz = {11.0, 12.0, 13.0};
    tmv::global_grid_metrics.z_int = {1.0, 2.0, 3.0, 4.0};
    tmv::global_grid_metrics.terrain_metrics_active = false;
    tmv::global_grid_metrics.terrain_metrics = nullptr;
    tmv::global_grid_metrics.terrain_topography = nullptr;
    const GridMetrics published = tmv::global_grid_metrics;

    tmv::global_terrain_config.scheme_id = "bell";
    tmv::global_terrain_metrics = TerrainMetrics3D{};
    REQUIRE_THROWS_AS(tmv::refresh_grid_metrics_from_terrain(),
                      std::invalid_argument);
    REQUIRE(tmv::global_grid_metrics.dx == published.dx);
    REQUIRE(tmv::global_grid_metrics.dy == published.dy);
    REQUIRE(tmv::global_grid_metrics.dz == published.dz);
    REQUIRE(tmv::global_grid_metrics.z_int == published.z_int);
    REQUIRE(tmv::global_grid_metrics.terrain_metrics_active
            == published.terrain_metrics_active);
    REQUIRE(tmv::global_grid_metrics.terrain_metrics
            == published.terrain_metrics);
    REQUIRE(tmv::global_grid_metrics.terrain_topography
            == published.terrain_topography);
}

TEST_CASE("Runtime leaves sedimentation subcycling to the microphysics owner",
          "[numerics][timestep][sedimentation]")
{
    reset_runtime_timestep_fixture(0.0f, 0.0f, 0.0f);
    dt = 50.0;
    global_simulation_storage.runtime.microphysics.scheme = std::make_unique<PrescribedFallSpeed>();
    REQUIRE(tmv::choose_runtime_timestep() == Approx(50.0));
    tmv::global_grid_metrics.dz[2] = 200.0;
    REQUIRE(tmv::choose_runtime_timestep() == Approx(50.0));
    dr = 1.0;
    tmv::global_grid_metrics.dx = dr;
    REQUIRE(tmv::choose_runtime_timestep() == Approx(50.0));
}
