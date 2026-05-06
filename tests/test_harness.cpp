/**
 * @file test_harness.cpp
 * @brief Provides definitions for all extern globals declared in headers.
 *
 * Test binaries that link against source files using tmv::log_* or any code
 * that includes simulation.hpp / runtime_config.hpp need these symbols.
 * This file provides minimal default values so tests can link without
 * pulling in the entire simulation runtime (tornado_sim.cpp, equations.cpp,
 * runtime_config.cpp, dynamics.cpp, etc.).
 *
 * If a test needs specific global state, it should set the values in a
 * Catch2 SECTION or TEST_CASE setup block.
 */

// Include full definitions for types that simulation.hpp only forward-declares.
// Order matters: these must come before simulation.hpp so the types are complete
// when unique_ptr destructors are instantiated.
#include "dynamics/dynamics_base.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "radiation/radiation_base.hpp"
#include "terrain/terrain_base.hpp"
#include "chaos/chaos_base.hpp"
#include "microphysics/microphysics_base.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "numerics/numerics_base.hpp"
#include "diagnostics/field_validation.hpp"
#include "radar/radar_base.hpp"

#include "core/runtime/simulation.hpp"
#include "core/runtime/runtime_config.hpp"

// ---------------------------------------------------------------------------
// Grid dimensions and spacing (simulation.hpp)
// ---------------------------------------------------------------------------
int NR = 8;
int NTH = 8;
int NZ = 8;

double dr = 100.0;
double dz = 100.0;
double dt = 0.1;
double dtheta = 2.0 * 3.14159265358979323846 / 8;

GridGeometry global_grid_geometry;
StaggerType global_stagger_type = StaggerType::Collocated;

static const bool grid_geometry_initialized = []() {
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    CoordinateSystem::Cylindrical);
    return true;
}();

double simulation_time = 0.0;

// ---------------------------------------------------------------------------
// Logging and perf (simulation.hpp)
// ---------------------------------------------------------------------------
LogProfile global_log_profile = LogProfile::quiet;
bool global_perf_timing_enabled = false;
int global_perf_report_every_steps = 0;

// ---------------------------------------------------------------------------
// Base-state profile (simulation.hpp)
// ---------------------------------------------------------------------------
std::vector<double> rho0_base;
std::vector<double> p0_base;
std::vector<double> qv0_base;
std::vector<double> u0_base;
std::vector<double> v0_base;
double coriolis_f = 0.0;

// ---------------------------------------------------------------------------
// Prognostic fields (simulation.hpp / equations.cpp)
// ---------------------------------------------------------------------------
Field3D rho;
Field3D p;
Field3D u;
Field3D w;
Field3D v;
Field3D tracer;

Field3D theta;
Field3D qv;
Field3D qc;
Field3D qr;
Field3D qi;
Field3D qs;
Field3D qh;
Field3D qg;
Field3D tke;

Field3D radar_reflectivity;

// Tendency fields
Field3D dtheta_dt_pbl;
Field3D dqv_dt_pbl;
Field3D du_dt_pbl;
Field3D dv_dt_pbl;
Field3D dtke_dt_pbl;
Field3D dtheta_dt_rad;

// Dynamics diagnostics
std::unique_ptr<DynamicsScheme> dynamics_scheme = nullptr;
Field3D vorticity_r;
Field3D vorticity_theta;
Field3D vorticity_z;
Field3D stretching_term;
Field3D tilting_term;
Field3D baroclinic_term;
Field3D angular_momentum;
Field3D angular_momentum_tendency;
Field3D p_prime;
Field3D dynamic_pressure;
Field3D buoyancy_pressure;

// Nested grid
NestedGridConfig nested_config;
Field3D nest_rho;
Field3D nest_p;
Field3D nest_u;
Field3D nest_w;
Field3D nest_v;
Field3D nest_theta;
Field3D nest_qv;
Field3D nest_qc;
Field3D nest_qr;
Field3D nest_qh;
Field3D nest_qg;
Field3D nest_tracer;

// ---------------------------------------------------------------------------
// Environment / sounding (simulation.hpp + runtime_config.hpp)
// ---------------------------------------------------------------------------
WindProfile global_wind_profile = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
double global_cape_target = 2500.0;
double global_sfc_theta_k = 300.0;
double global_sfc_qv_kgkg = 0.014;
double global_tropopause_z_m = 12000.0;

double global_bubble_center_x_m = 50000.0;
double global_bubble_center_y_m = 0.0;
double global_bubble_center_z_m = 1500.0;
double global_bubble_radius_m = 10000.0;
double global_bubble_dtheta_k = 2.0;

bool global_sounding_enabled = false;
bool global_sounding_allow_placeholder_profiles = false;
SoundingConfig global_runtime_sounding_config{};

std::string global_microphysics_scheme = "kessler";
std::string global_dynamics_scheme_name = "tornado";

CoordinateSystem global_coordinate_system = CoordinateSystem::Cylindrical;

// ---------------------------------------------------------------------------
// Validation (runtime_config.hpp / field_validation.hpp)
// ---------------------------------------------------------------------------
tmv::ValidationPolicy global_validation_policy;
std::string global_validation_report_path;

// ---------------------------------------------------------------------------
// Grouped config structs (simulation.hpp)
// ---------------------------------------------------------------------------
SimulationConfig global_sim_config;
SchemeRegistry global_scheme_registry;

// SchemeRegistry destructor/move need complete types.
SchemeRegistry::~SchemeRegistry() = default;
SchemeRegistry::SchemeRegistry(SchemeRegistry&&) noexcept = default;
SchemeRegistry& SchemeRegistry::operator=(SchemeRegistry&&) noexcept = default;

// ---------------------------------------------------------------------------
// Physics configs (various _base.hpp headers)
// ---------------------------------------------------------------------------
RadiationConfig global_radiation_config;
BoundaryLayerConfig global_boundary_layer_config;
SurfaceConfig global_surface_config;
TurbulenceConfig global_turbulence_config;
TerrainConfig global_terrain_config;

// Physics scheme pointers
std::unique_ptr<RadiationSchemeBase> radiation_scheme = nullptr;
std::unique_ptr<BoundaryLayerSchemeBase> boundary_layer_scheme = nullptr;
std::unique_ptr<TurbulenceSchemeBase> turbulence_scheme = nullptr;
std::unique_ptr<TerrainSchemeBase> terrain_scheme = nullptr;
std::unique_ptr<MicrophysicsScheme> microphysics_scheme = nullptr;
std::unique_ptr<RadarSchemeBase> radar_scheme = nullptr;

// Terrain state
Topography2D global_topography;
TerrainMetrics3D global_terrain_metrics;

// ---------------------------------------------------------------------------
// Numerics configs (numerics_base.hpp)
// ---------------------------------------------------------------------------
GridMetrics global_grid_metrics;
AdvectionConfig global_advection_config;
DiffusionConfig global_diffusion_config;
TimeSteppingConfig global_time_stepping_config;

std::unique_ptr<AdvectionSchemeBase> advection_scheme;
std::unique_ptr<DiffusionSchemeBase> diffusion_scheme;
std::unique_ptr<TimeSteppingSchemeBase> time_stepping_scheme;

// ---------------------------------------------------------------------------
// Chaos (chaos_base.hpp)
// ---------------------------------------------------------------------------
chaos::ChaosConfig global_chaos_config;

// ---------------------------------------------------------------------------
// GPU dispatch stubs (compute_kernel_template.hpp)
//
// Unit tests run on CPU only. These weak stubs satisfy the linker for test
// binaries that link against cartesian.cpp without pulling in the full
// compute backend. Test binaries that DO link compute_kernel_template.cpp
// (e.g., the advection test) will use the real symbols instead.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool dispatch_cartesian_tendencies_backend(
    const float*, const float*, const float*,
    const float*, const float*, const float*,
    const float*, const float*,
    const float*,
    const float*, const float*,
    float*, float*, float*, float*, float*,
    int, int, int, float, float, float, float, float, float)
{ return false; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool dispatch_advection_x_backend(
    const float*, const float*, float*,
    int, int, int, float, float)
{ return false; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool dispatch_advection_y_backend(
    const float*, const float*, float*,
    int, int, int, float, float)
{ return false; }

// Bubble-trigger weak stubs. The real implementations live in
// src/core/orchestration/dynamics/initial_conditions_cartesian.cpp and
// src/core/orchestration/dynamics/equations.cpp. Tests that link those
// TUs override these weak symbols; tests that only need the trigger
// factory + classes (e.g. test_init_scheme_profile_validator) get the
// no-op stubs and don't need to drag in the full equations.cpp closure.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void apply_cartesian_bubble_initialization() {}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void apply_cylindrical_bubble_initialization() {}
