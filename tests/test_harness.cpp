// Minimal process-state definitions for unit tests that do not link the full
// simulation executable. Individual tests configure only the state they use.

// Include full definitions for types that simulation.hpp only forward-declares.
// Order matters: these must come before simulation.hpp so the types are complete
// when unique_ptr destructors are instantiated.
#include "dynamics/dynamics_base.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "boundary_layer/runtime.hpp"
#include "turbulence/turbulence_base.hpp"
#include "turbulence/runtime.hpp"
#include "radiation/radiation_base.hpp"
#include "terrain/terrain_base.hpp"
#include "chaos/runtime.hpp"
#include "microphysics/microphysics_base.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "numerics/numerics_base.hpp"
#include "diagnostics/field_validation.hpp"
#include "radar/radar_base.hpp"
#include "radar/runtime.hpp"
#include "compute/compute_backend.hpp"

#include "core/runtime/simulation.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_runtime.hpp"
#include "core/runtime/simulation_state.hpp"

struct SimulationStateView;

// ---------------------------------------------------------------------------
// Grid dimensions and spacing (simulation.hpp)
// ---------------------------------------------------------------------------
SimulationStorageState global_simulation_storage{{8, 8, 8, 100.0, 100.0, 0.1, 2.0 * 3.14159265358979323846 / 8}};
int&                   NR                   = global_simulation_storage.grid.nr;
int&                   NTH                  = global_simulation_storage.grid.nth;
int&                   NZ                   = global_simulation_storage.grid.nz;
double&                dr                   = global_simulation_storage.grid.dx;
double&                dz                   = global_simulation_storage.grid.dz;
double&                dt                   = global_simulation_storage.grid.dt;
double&                dtheta               = global_simulation_storage.grid.dy;
GridGeometry&          global_grid_geometry = global_simulation_storage.grid.geometry;
StaggerType&           global_stagger_type  = global_simulation_storage.configuration.stagger_type;

static const bool runtime_configuration_initialized = []()
{
    global_simulation_storage.configuration.log_profile          = LogProfile::quiet;
    global_simulation_storage.configuration.dynamics_scheme_name = "cartesian";
    return true;
}();

static const bool grid_geometry_initialized = []()
{
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian);
    return true;
}();

double& simulation_time = global_simulation_storage.time_s;

// ---------------------------------------------------------------------------
// Logging and perf (simulation.hpp)
// ---------------------------------------------------------------------------
LogProfile& global_log_profile             = global_simulation_storage.configuration.log_profile;
bool&       global_perf_timing_enabled     = global_simulation_storage.configuration.perf_timing_enabled;
int&        global_perf_report_every_steps = global_simulation_storage.configuration.perf_report_every_steps;
int&        global_perf_warmup_steps       = global_simulation_storage.configuration.perf_warmup_steps;
int&        global_scientific_diagnostics_every_steps =
    global_simulation_storage.configuration.scientific_diagnostics_every_steps;

// ---------------------------------------------------------------------------
// Base-state profile (simulation.hpp)
// ---------------------------------------------------------------------------
std::vector<double>& rho0_base   = global_simulation_storage.base.rho0;
std::vector<double>& p0_base     = global_simulation_storage.base.p0;
std::vector<double>& qv0_base    = global_simulation_storage.base.qv0;
std::vector<double>& u0_base     = global_simulation_storage.base.u0;
std::vector<double>& v0_base     = global_simulation_storage.base.v0;
std::vector<double>& theta0_base = global_simulation_storage.base.theta0;
double&              coriolis_f  = global_simulation_storage.coriolis_f;

// ---------------------------------------------------------------------------
// Prognostic fields (simulation.hpp / equations.cpp)
// ---------------------------------------------------------------------------
Field3D& rho    = global_simulation_storage.fields.rho;
Field3D& p      = global_simulation_storage.fields.p;
Field3D& u      = global_simulation_storage.fields.u;
Field3D& w      = global_simulation_storage.fields.w;
Field3D& v      = global_simulation_storage.fields.v;
Field3D& tracer = global_simulation_storage.fields.tracer;

Field3D& theta = global_simulation_storage.fields.theta;
Field3D& qv    = global_simulation_storage.fields.qv;
Field3D& qc    = global_simulation_storage.fields.qc;
Field3D& qr    = global_simulation_storage.fields.qr;
Field3D& qi    = global_simulation_storage.fields.qi;
Field3D& qs    = global_simulation_storage.fields.qs;
Field3D& qh    = global_simulation_storage.fields.qh;
Field3D& qg    = global_simulation_storage.fields.qg;
Field3D& tke   = global_simulation_storage.fields.tke;

Field3D& radar_reflectivity = global_simulation_storage.diagnostics.radar_reflectivity;

// Tendency fields
Field3D& dtheta_dt_pbl = global_simulation_storage.tendencies.dtheta_dt_pbl;
Field3D& dqv_dt_pbl    = global_simulation_storage.tendencies.dqv_dt_pbl;
Field3D& dqc_dt_pbl    = global_simulation_storage.tendencies.dqc_dt_pbl;
Field3D& dqi_dt_pbl    = global_simulation_storage.tendencies.dqi_dt_pbl;
Field3D& du_dt_pbl     = global_simulation_storage.tendencies.du_dt_pbl;
Field3D& dv_dt_pbl     = global_simulation_storage.tendencies.dv_dt_pbl;
Field3D& dtke_dt_pbl   = global_simulation_storage.tendencies.dtke_dt_pbl;
Field3D& dtheta_dt_rad = global_simulation_storage.tendencies.dtheta_dt_rad;

std::vector<float>& sfc_ustar              = global_simulation_storage.surface.ustar;
std::vector<float>& sfc_tau_u              = global_simulation_storage.surface.tau_u;
std::vector<float>& sfc_tau_v              = global_simulation_storage.surface.tau_v;
std::vector<float>& sfc_sensible_heat_flux = global_simulation_storage.surface.sensible_heat_flux;
std::vector<float>& sfc_latent_heat_flux   = global_simulation_storage.surface.latent_heat_flux;
std::vector<float>& sfc_moisture_flux      = global_simulation_storage.surface.moisture_flux;
std::vector<float>& pbl_height_2d          = global_simulation_storage.surface.pbl_height;

// Dynamics diagnostics
namespace
{
tmv::DynamicsRuntimeState harness_dynamics_runtime_state;
}
std::unique_ptr<tmv::DynamicsScheme>& tmv::dynamics_scheme =
    harness_dynamics_runtime_state.scheme;
tmv::DynamicsRuntimeState& tmv::global_dynamics_runtime_state =
    harness_dynamics_runtime_state;
Field3D&                             vorticity_r          = global_simulation_storage.diagnostics.vorticity_r;
Field3D&                             vorticity_theta      = global_simulation_storage.diagnostics.vorticity_theta;
Field3D&                             vorticity_z          = global_simulation_storage.diagnostics.vorticity_z;
Field3D&                             stretching_term      = global_simulation_storage.diagnostics.stretching_term;
Field3D&                             tilting_term         = global_simulation_storage.diagnostics.tilting_term;
Field3D&                             baroclinic_term      = global_simulation_storage.diagnostics.baroclinic_term;
Field3D&                             angular_momentum     = global_simulation_storage.diagnostics.angular_momentum;
Field3D& angular_momentum_tendency = global_simulation_storage.diagnostics.angular_momentum_tendency;
Field3D& p_prime                   = global_simulation_storage.diagnostics.p_prime;
Field3D& dynamic_pressure          = global_simulation_storage.diagnostics.dynamic_pressure;
Field3D& buoyancy_pressure         = global_simulation_storage.diagnostics.buoyancy_pressure;

// Nested grid
NestedGridConfig& nested_config = global_simulation_storage.nested.config;
Field3D&          nest_rho      = global_simulation_storage.nested.rho;
Field3D&          nest_p        = global_simulation_storage.nested.p;
Field3D&          nest_u        = global_simulation_storage.nested.u;
Field3D&          nest_w        = global_simulation_storage.nested.w;
Field3D&          nest_v        = global_simulation_storage.nested.v;
Field3D&          nest_theta    = global_simulation_storage.nested.theta;
Field3D&          nest_qv       = global_simulation_storage.nested.qv;
Field3D&          nest_qc       = global_simulation_storage.nested.qc;
Field3D&          nest_qr       = global_simulation_storage.nested.qr;
Field3D&          nest_qi       = global_simulation_storage.nested.qi;
Field3D&          nest_qs       = global_simulation_storage.nested.qs;
Field3D&          nest_qh       = global_simulation_storage.nested.qh;
Field3D&          nest_qg       = global_simulation_storage.nested.qg;
Field3D&          nest_tracer   = global_simulation_storage.nested.tracer;

// ---------------------------------------------------------------------------
// Environment / sounding (simulation.hpp + runtime_config.hpp)
// ---------------------------------------------------------------------------
static RuntimeConfigurationState& harness_configuration        = global_simulation_storage.configuration;
WindProfile&                      global_wind_profile          = harness_configuration.wind_profile;
double&                           global_configured_dt_large_s = harness_configuration.configured_dt_large_s;
tmv::init::config::DomainMotionMode& global_domain_motion_mode =
    harness_configuration.domain_motion_mode;
tmv::domain::HorizontalDomainFrame& global_domain_frame =
    harness_configuration.domain_frame;
tmv::init::config::DomainMotionSection::Tracking& global_storm_tracking_config = harness_configuration.storm_tracking;
double&                                           global_cape_target           = harness_configuration.cape_target;
double&                                           global_sfc_theta_k           = harness_configuration.sfc_theta_k;
double&                                           global_sfc_qv_kgkg           = harness_configuration.sfc_qv_kgkg;
double&                                           global_tropopause_z_m        = harness_configuration.tropopause_z_m;
double&                                           global_bubble_center_x_m = harness_configuration.bubble_center_x_m;
double&                                           global_bubble_center_y_m = harness_configuration.bubble_center_y_m;
double&                                           global_bubble_center_z_m = harness_configuration.bubble_center_z_m;
double&                                           global_bubble_radius_m   = harness_configuration.bubble_radius_m;
double&                                           global_bubble_dtheta_k   = harness_configuration.bubble_dtheta_k;
double& global_bubble_cloud_water_kgkg =
    harness_configuration.bubble_cloud_water_kgkg;

bool&           global_sounding_enabled                    = harness_configuration.sounding_enabled;
bool&           global_sounding_allow_placeholder_profiles = harness_configuration.sounding_allow_placeholder_profiles;
SoundingConfig& global_runtime_sounding_config             = harness_configuration.sounding_config;
tmv::init::SoundingSourceConfig&  global_sounding_source_config  = harness_configuration.sounding_source;
tmv::init::HodographSourceConfig& global_hodograph_source_config = harness_configuration.hodograph_source;
tmv::init::TriggerSourceConfig&   global_trigger_source_config   = harness_configuration.trigger_source;
std::string&                      global_microphysics_scheme     = harness_configuration.microphysics_scheme;
std::string&                      global_dynamics_scheme_name    = harness_configuration.dynamics_scheme_name;
CoordinateSystem&                 global_coordinate_system       = harness_configuration.coordinate_system;

tmv::init::config::LateralBC& global_lateral_bc_west  = harness_configuration.lateral_bc_west;
tmv::init::config::LateralBC& global_lateral_bc_east  = harness_configuration.lateral_bc_east;
tmv::init::config::LateralBC& global_lateral_bc_south = harness_configuration.lateral_bc_south;
tmv::init::config::LateralBC& global_lateral_bc_north = harness_configuration.lateral_bc_north;
tmv::init::config::BottomBC&  global_bottom_bc        = harness_configuration.bottom_bc;
tmv::init::config::TopBC&     global_top_bc           = harness_configuration.top_bc;

bool&                           global_damping_enabled        = harness_configuration.damping_enabled;
double&                         global_damping_z_d            = harness_configuration.damping_z_d;
double&                         global_damping_alpha          = harness_configuration.damping_alpha;
int&                            global_lateral_sponge_width   = harness_configuration.lateral_sponge_width;
double&                         global_lateral_sponge_alpha   = harness_configuration.lateral_sponge_alpha;
tmv::init::config::DampingMode& global_damping_mode           = harness_configuration.damping_mode;
bool&                           global_w_only_top_sponge      = harness_configuration.w_only_top_sponge;
int&                            global_w_only_depth_cells     = harness_configuration.w_only_depth_cells;
double&                         global_w_only_alpha_max       = harness_configuration.w_only_alpha_max;
bool&                           global_lateral_sponge_enabled = harness_configuration.lateral_sponge_enabled;
bool&                           global_slow_rk2               = harness_configuration.slow_rk2;
double&                         global_lid_bc_gamma           = harness_configuration.lid_bc_gamma;
bool&                           global_axis_u_m1_filter       = harness_configuration.axis_u_m1_filter;
bool&                           global_gpu_apply_microphysics = harness_configuration.gpu_apply_microphysics;
bool&                           global_gpu_apply_slow         = harness_configuration.gpu_apply_slow;
bool&                           global_gpu_apply_sgs          = harness_configuration.gpu_apply_sgs;

bool&                            global_coriolis_enabled      = harness_configuration.coriolis_enabled;
tmv::init::config::CoriolisMode& global_coriolis_mode         = harness_configuration.coriolis_mode;
double&                          global_coriolis_latitude_deg = harness_configuration.coriolis_latitude_deg;

// ---------------------------------------------------------------------------
// Validation (runtime_config.hpp / field_validation.hpp)
// ---------------------------------------------------------------------------
tmv::ValidationPolicy& global_validation_policy      = harness_configuration.validation_policy;
std::string&           global_validation_report_path = harness_configuration.validation_report_path;

// ---------------------------------------------------------------------------
// Physics configs (various _base.hpp headers)
// ---------------------------------------------------------------------------
ComputeBackendRuntimeState&         global_compute_backend_runtime_state = global_simulation_storage.runtime.compute;
ComputeBackendConfig&               global_compute_backend_config        = global_compute_backend_runtime_state.config;
nimbus::parity::ParityRuntimeState& nimbus::parity::global_parity_runtime_state =
    global_simulation_storage.runtime.parity;
tmv::RadiationRuntimeState&     tmv::global_radiation_runtime_state = global_simulation_storage.runtime.radiation;
tmv::BoundaryLayerRuntimeState& tmv::global_boundary_layer_runtime_state =
    global_simulation_storage.runtime.boundary_layer;
tmv::TurbulenceRuntimeState& tmv::global_turbulence_runtime_state       = global_simulation_storage.runtime.turbulence;
RayleighDampingRuntimeState& global_rayleigh_runtime_state              = global_simulation_storage.runtime.rayleigh;
#if defined(__GNUC__) || defined(__clang__)
#define NIMBUS_TEST_WEAK __attribute__((weak))
#else
#define NIMBUS_TEST_WEAK
#endif
NIMBUS_TEST_WEAK std::unique_ptr<tmv::RadiationSchemeBase>& tmv::radiation_scheme        = tmv::global_radiation_runtime_state.scheme;
NIMBUS_TEST_WEAK tmv::RadiationConfig&                      tmv::global_radiation_config = tmv::global_radiation_runtime_state.config;
NIMBUS_TEST_WEAK std::unique_ptr<tmv::BoundaryLayerSchemeBase>& tmv::boundary_layer_scheme =
    tmv::global_boundary_layer_runtime_state.scheme;
NIMBUS_TEST_WEAK tmv::BoundaryLayerConfig& tmv::global_boundary_layer_config = tmv::global_boundary_layer_runtime_state.config;
NIMBUS_TEST_WEAK tmv::SurfaceConfig&       tmv::global_surface_config        = tmv::global_boundary_layer_runtime_state.surface_config;

// Isolated physics tests publish their coupled tendency fields directly.
// The production implementation reconstructs held PBL sources with live
// density before microphysics; that behavior has its own runtime-state suite.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void tmv::prepare_boundary_layer_tendencies_for_application(const SimulationStateView&)
{
}
NIMBUS_TEST_WEAK std::unique_ptr<tmv::TurbulenceSchemeBase>& tmv::turbulence_scheme        = tmv::global_turbulence_runtime_state.scheme;
NIMBUS_TEST_WEAK tmv::TurbulenceConfig&                      tmv::global_turbulence_config = tmv::global_turbulence_runtime_state.config;
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void tmv::record_turbulence_perf(TurbulencePerfStage, double)
{
}
tmv::TerrainRuntimeState&                tmv::global_terrain_runtime_state = global_simulation_storage.runtime.terrain;
std::unique_ptr<tmv::TerrainSchemeBase>& tmv::terrain_scheme               = tmv::global_terrain_runtime_state.scheme;
tmv::TerrainConfig&                      tmv::global_terrain_config        = tmv::global_terrain_runtime_state.config;
tmv::Topography2D&                       tmv::global_topography      = tmv::global_terrain_runtime_state.topography;
tmv::TerrainMetrics3D&                   tmv::global_terrain_metrics = tmv::global_terrain_runtime_state.metrics;

// Physics scheme pointers
tmv::BoundaryConditionRuntimeState&
tmv::global_boundary_condition_runtime_state =
    global_simulation_storage.runtime.boundary_conditions;
tmv::MicrophysicsRuntimeState& tmv::global_microphysics_runtime_state = global_simulation_storage.runtime.microphysics;
std::unique_ptr<tmv::MicrophysicsScheme>& tmv::microphysics_scheme    = tmv::global_microphysics_runtime_state.scheme;
tmv::RadarRuntimeState&                   tmv::global_radar_runtime_state = global_simulation_storage.runtime.radar;
std::unique_ptr<RadarSchemeBase>&         radar_scheme = tmv::global_radar_runtime_state.selected_scheme;
RadarProductSelection                     global_radar_product_selection;

void tmv::RadarRuntimeState::reset() noexcept
{
    *this = {};
}

// Terrain state
// ---------------------------------------------------------------------------
// Numerics configs (numerics_base.hpp)
// ---------------------------------------------------------------------------
NIMBUS_TEST_WEAK tmv::GridMetrics&              tmv::global_grid_metrics =
    global_simulation_storage.process_grid_metrics;
tmv::AdvectionRuntimeState&    tmv::global_advection_runtime_state = global_simulation_storage.runtime.advection;
tmv::DiffusionRuntimeState&    tmv::global_diffusion_runtime_state = global_simulation_storage.runtime.diffusion;
tmv::TimeSteppingRuntimeState& tmv::global_time_stepping_runtime_state =
    global_simulation_storage.runtime.time_stepping;
NIMBUS_TEST_WEAK std::unique_ptr<tmv::AdvectionSchemeBase>&    tmv::advection_scheme        = tmv::global_advection_runtime_state.scheme;
NIMBUS_TEST_WEAK tmv::AdvectionConfig&                         tmv::global_advection_config = tmv::global_advection_runtime_state.config;
NIMBUS_TEST_WEAK std::unique_ptr<tmv::DiffusionSchemeBase>&    tmv::diffusion_scheme        = tmv::global_diffusion_runtime_state.scheme;
NIMBUS_TEST_WEAK tmv::DiffusionConfig&                         tmv::global_diffusion_config = tmv::global_diffusion_runtime_state.config;
NIMBUS_TEST_WEAK std::unique_ptr<tmv::TimeSteppingSchemeBase>& tmv::time_stepping_scheme =
    tmv::global_time_stepping_runtime_state.scheme;
NIMBUS_TEST_WEAK tmv::TimeSteppingConfig& tmv::global_time_stepping_config = tmv::global_time_stepping_runtime_state.config;

// ---------------------------------------------------------------------------
// Chaos (chaos_base.hpp)
// ---------------------------------------------------------------------------
tmv::chaos::ChaosRuntimeState& tmv::chaos::global_chaos_runtime_state = global_simulation_storage.runtime.chaos;
NIMBUS_TEST_WEAK std::unique_ptr<tmv::chaos::ChaosScheme>& tmv::chaos::chaos_scheme    = tmv::chaos::global_chaos_runtime_state.scheme;
NIMBUS_TEST_WEAK tmv::chaos::ChaosConfig&      tmv::chaos::global_chaos_config         = tmv::chaos::global_chaos_runtime_state.config;
NIMBUS_TEST_WEAK tmv::chaos::ChaosDiagnostics& tmv::chaos::global_chaos_diagnostics = tmv::chaos::global_chaos_runtime_state.diagnostics;

#undef NIMBUS_TEST_WEAK

// ---------------------------------------------------------------------------
// GPU dispatch stubs (compute_kernel_template.hpp)
//
// Unit tests run on CPU only. These weak stubs satisfy the linker for test
// binaries that link against cartesian.cpp without pulling in the full
// compute backend. Test binaries that do link the compute dispatch sources
// (for example, the advection test) use the real symbols instead.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool active_compute_backend_supports(AdvectionKernelOp)
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool active_compute_backend_supports(DynamicsKernelOp)
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool active_compute_backend_supports(MicrophysicsKernelOp)
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool active_compute_backend_supports(TurbulenceKernelOp)
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool active_compute_backend_supports(DiffusionKernelOp)
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool active_compute_backend_supports_supplied_cartesian_cgrid_mass_flux()
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool active_compute_backend_supports_batched_cartesian_cgrid_mass_flux_export()
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool dispatch_cartesian_tendencies_backend(const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           const float*,
                                           float*,
                                           float*,
                                           float*,
                                           float*,
                                           float*,
                                           int,
                                           int,
                                           int,
                                           float,
                                           float,
                                           float,
                                           float,
                                           float,
                                           float,
                                           int,
                                           int,
                                           int,
                                           int,
                                           int,
                                           bool,
                                           float)
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool dispatch_advection_x_backend(const float*, const float*, float*, int, int, int, float, float)
{
    return false;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
bool dispatch_advection_y_backend(const float*, const float*, float*, int, int, int, float, float)
{
    return false;
}

// Bubble-trigger weak stub. The real implementation lives in
// src/core/orchestration/dynamics/initial_conditions_cartesian.cpp.
// Tests that link that TU override this weak symbol; tests that only need
// the trigger factory + classes (e.g. test_init_scheme_profile_validator)
// get the no-op stub and don't need to drag in the full IC closure.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void apply_cartesian_bubble_initialization()
{
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak))
#endif
void apply_cartesian_bubble_initialization(const SimulationStateView&)
{
}
