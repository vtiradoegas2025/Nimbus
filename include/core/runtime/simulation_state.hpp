#include "core/field/field_inventory.hpp"
#pragma once
#include "util/transient_workspace.hpp"
#include "diagnostics/budget_level.hpp"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/nested_interface_ledger.hpp"
#include "core/infra/nested_grid.hpp"
#include "core/infra/nested_feedback.hpp"
#include "core/infra/nested_terrain_geometry.hpp"
#include "core/runtime/grid_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_runtime.hpp"

// Non-owning views over caller- or process-owned simulation storage.

struct PrognosticFieldsView
{
    // Model-wide mass basis (SCI-024): rho is dry-air density [kg_dry m-3].
    // Every q field below is a dry-air mixing ratio [kg_species kg_dry-1].
    // Thus rho*q is species mass concentration, rho is dry mass density, and
    // rho*(1 + sum(q)) is total moist mass density.
    Field3D& rho;
    Field3D& p;
    Field3D& u;
    Field3D& w;
    Field3D& v;
    Field3D& tracer;

    Field3D& theta;
    Field3D& qv;
    Field3D& qc;
    Field3D& qr;
    Field3D& qi;
    Field3D& qs;
    Field3D& qh;
    Field3D& qg;
    Field3D& tke;
};

struct BaseStateView
{
    std::vector<double>& rho0;  // dry-air density reference profile [kg_dry m-3]
    std::vector<double>& p0;
    std::vector<double>& qv0;
    std::vector<double>& theta0;
    std::vector<double>& u0;
    std::vector<double>& v0;
};

struct PhysicsTendencyFieldsView
{
    Field3D& dtheta_dt_rad;
    Field3D& dtheta_dt_pbl;
    Field3D& dqv_dt_pbl;
    Field3D& du_dt_pbl;
    Field3D& dv_dt_pbl;
    Field3D& dtke_dt_pbl;
    Field3D* dqc_dt_pbl = nullptr;
    Field3D* dqi_dt_pbl = nullptr;
};

struct SurfaceDiagnosticFieldsView
{
    std::vector<float>* sfc_ustar = nullptr;
    std::vector<float>* sfc_tau_u = nullptr;
    std::vector<float>* sfc_tau_v = nullptr;
    std::vector<float>* sfc_sensible_heat_flux = nullptr;
    std::vector<float>* sfc_latent_heat_flux = nullptr;
    std::vector<float>* sfc_moisture_flux = nullptr;
    std::vector<float>* pbl_height_2d = nullptr;
    // Exact bottom-sedimentation fallout diagnosed by microphysics. Both use
    // scalar-surface ordering `(i * nth + j)`.
    std::vector<float>* precipitation_rate_mm_h = nullptr;
    std::vector<float>* accumulated_rainfall_mm = nullptr;
    std::vector<float>* accumulated_rainfall_residual_mm = nullptr;
    double* accumulated_rainfall_last_update_s = nullptr;
};

struct PrognosticFieldStorage
{
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
};

struct BaseStateStorage
{
    std::vector<double> rho0;
    std::vector<double> p0;
    std::vector<double> qv0;
    std::vector<double> theta0;
    std::vector<double> u0;
    std::vector<double> v0;
};

struct PhysicsTendencyFieldStorage
{
    Field3D dtheta_dt_rad;
    Field3D dtheta_dt_pbl;
    Field3D dqv_dt_pbl;
    Field3D du_dt_pbl;
    Field3D dv_dt_pbl;
    Field3D dtke_dt_pbl;
    Field3D dqc_dt_pbl;
    Field3D dqi_dt_pbl;
};

struct SurfaceDiagnosticFieldStorage
{
    std::vector<float> ustar;
    std::vector<float> tau_u;
    std::vector<float> tau_v;
    std::vector<float> sensible_heat_flux;
    std::vector<float> latent_heat_flux;
    std::vector<float> moisture_flux;
    std::vector<float> pbl_height;
    // Instantaneous exact surface fallout and its run-history integral.
    // One kg m^-2 of liquid-water-equivalent fallout is one millimetre.
    std::vector<float> precipitation_rate_mm_h;
    std::vector<float> accumulated_rainfall_mm;
    // Sub-ULP remainder retained when the cumulative total is published to
    // FP32. Effective rainfall is accumulated_rainfall_mm + this residual.
    std::vector<float> accumulated_rainfall_residual_mm;
    double accumulated_rainfall_last_update_s = 0.0;
};

struct NestedGridStorage
{
    NestedGridConfig config;
    Field3D rho;
    Field3D p;
    Field3D u;
    Field3D w;
    Field3D v;
    Field3D theta;
    Field3D qv;
    Field3D qc;
    Field3D qr;
    Field3D qi;
    Field3D qs;
    Field3D qh;
    Field3D qg;
    Field3D tracer;

    explicit NestedGridStorage(int nr = 200, int nth = 128, int nz = 150)
        : config(nr, nth, nz)
    {
    }
};

struct DiagnosticFieldStorage
{
    Field3D radar_reflectivity;
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
};

static_assert(sizeof(PrognosticFieldStorage) ==
    std::tuple_size_v<decltype(tmv::fields::prognostic)> * sizeof(Field3D));
static_assert(sizeof(PhysicsTendencyFieldStorage) ==
    std::tuple_size_v<decltype(tmv::fields::physics_tendencies)> * sizeof(Field3D));
static_assert(sizeof(DiagnosticFieldStorage) ==
    std::tuple_size_v<decltype(tmv::fields::diagnostics)> * sizeof(Field3D));

struct GridStorageState
{
    int nr = 200;
    int nth = 128;
    int nz = 150;
    double dx = 100.0;
    double dz = 100.0;
    double dt = 0.1;
    double dy = 100.0;
    GridGeometry geometry;

    GridStorageState() = default;
    GridStorageState(int nr_in, int nth_in, int nz_in,
                     double dx_in, double dz_in, double dt_in, double dy_in)
        : nr(nr_in), nth(nth_in), nz(nz_in),
          dx(dx_in), dz(dz_in), dt(dt_in), dy(dy_in)
    {
    }
};

// Per-level scientific storage boundary. It mirrors the owners exposed through
// SimulationStateView without carrying run-global control or another nest.
struct AtmosphericLevelStorage
{
    GridStorageState grid;
    // Derived execution metrics travel with the level so switching between a
    // parent and child never reuses spacing or terrain pointers from the
    // other grid. They are rebuilt rather than serialized.
    GridMetrics process_grid_metrics;
    PrognosticFieldStorage fields;
    BaseStateStorage base;
    PhysicsTendencyFieldStorage tendencies;
    SurfaceDiagnosticFieldStorage surface;
    DiagnosticFieldStorage diagnostics;
    AtmosphericLevelRuntimeState runtime;
    tmv::domain::HorizontalDomainFrame domain_frame;
    double coriolis_f = 0.0;
    double time_s = 0.0;

    AtmosphericLevelStorage() = default;
    explicit AtmosphericLevelStorage(GridStorageState grid_in)
        : grid(std::move(grid_in))
    {
    }
};

// Reusable, non-durable workspace for conservative child-boundary updates.
// Copies and checkpoints intentionally start cold; moves retain allocations.
struct NestedBoundaryScratchStorage
{
    static constexpr std::size_t movement_primary_field_count = 15;

    PrognosticFieldStorage target;
    PrognosticFieldStorage candidate;
    std::array<Field3D,
               NestedPrivateScalarRelaxationReportV1::maximum_scalars>
        microphysics_private_candidate;
    Field3D theta_storage_residual_candidate;
    std::array<std::vector<float>, movement_primary_field_count>
        movement_primary_discard;
    std::array<std::vector<float>,
               NestedPrivateScalarRelaxationReportV1::maximum_scalars>
        movement_microphysics_newly_admitted;
    std::array<std::vector<float>,
               NestedPrivateScalarRelaxationReportV1::maximum_scalars>
        movement_microphysics_discard;
    std::vector<float> movement_theta_residual_newly_admitted;
    std::vector<float> movement_theta_residual_discard;
    std::array<std::vector<float>, 3>
        movement_precipitation_newly_admitted;
    std::array<std::vector<float>, 3>
        movement_precipitation_discard;
    std::array<std::vector<float>, 2> movement_radiation_newly_admitted;
    std::array<std::vector<float>, 7>
        movement_boundary_layer_cell_newly_admitted;
    std::array<std::vector<float>, 9>
        movement_boundary_layer_column_newly_admitted;
    std::vector<tmv::SlabColumnState>
        movement_boundary_layer_slab_newly_admitted;
    std::array<std::vector<float>, 5>
        movement_chaos_cache_cell_newly_admitted;
    std::array<std::vector<float>, 6>
        movement_chaos_cache_column_newly_admitted;
    std::vector<std::uint8_t> movement_storm_partition_candidate;
    std::unique_ptr<tmv::chaos::ChaosScheme>
        movement_chaos_translation_workspace;

    NestedBoundaryScratchStorage() = default;
    NestedBoundaryScratchStorage(
        const NestedBoundaryScratchStorage&) noexcept
    {
    }
    NestedBoundaryScratchStorage& operator=(
        const NestedBoundaryScratchStorage& other) noexcept
    {
        if (this != &other)
        {
            target = {};
            candidate = {};
            microphysics_private_candidate = {};
            theta_storage_residual_candidate = {};
            movement_primary_discard = {};
            movement_microphysics_newly_admitted = {};
            movement_microphysics_discard = {};
            movement_theta_residual_newly_admitted = {};
            movement_theta_residual_discard = {};
            movement_precipitation_newly_admitted = {};
            movement_precipitation_discard = {};
            movement_radiation_newly_admitted = {};
            movement_boundary_layer_cell_newly_admitted = {};
            movement_boundary_layer_column_newly_admitted = {};
            movement_boundary_layer_slab_newly_admitted = {};
            movement_chaos_cache_cell_newly_admitted = {};
            movement_chaos_cache_column_newly_admitted = {};
            movement_storm_partition_candidate = {};
            movement_chaos_translation_workspace.reset();
        }
        return *this;
    }
    NestedBoundaryScratchStorage(NestedBoundaryScratchStorage&&) noexcept =
        default;
    NestedBoundaryScratchStorage& operator=(
        NestedBoundaryScratchStorage&&) noexcept = default;
};

struct NestedExecutionStorage
{
    bool child_execution_active = false;
    tmv::TransientWorkspace<AtmosphericLevelStorage> substep_rollback;
    bool relationship_ready = false;
    bool reference_state_ready = false;
    bool physical_primary_state_ready = false;
    bool parent_boundary_state_ready = false;
    tmv::domain::OneWayNestRelationshipV1 relationship;
    tmv::domain::ParentTimeBracketV1 parent_time_bracket;
    AtmosphericLevelStorage child;
    std::uint64_t movement_sequence = 0;
    int initial_parent_origin_i = 0;
    int initial_parent_origin_j = 0;
    int cumulative_parent_shift_i = 0;
    int cumulative_parent_shift_j = 0;
    int cumulative_parent_recycling_shift_i = 0;
    int cumulative_parent_recycling_shift_j = 0;
    NestedInterfaceSourceLedgerV2 interface_source;
    NestedChildRelaxationReport two_way_interval_source_begin;
    std::array<double, 5> two_way_child_interval_begin_inventory{};
    double two_way_dry_mass_storage_begin = 0.0;
    double two_way_theta_storage_begin = 0.0;
    NestedTwoWayFeedbackReportV1 last_two_way_feedback;
    std::uint64_t two_way_feedback_application_count = 0;
    tmv::domain::NestedActiveTerrainStateV1 active_terrain;
    NestedBoundaryScratchStorage boundary_scratch;
};

// Owns the model state advanced through SimulationStateView.
struct SimulationStorageState
{
    GridStorageState grid;
    GridMetrics process_grid_metrics;
    RuntimeConfigurationState configuration;
    PrognosticFieldStorage fields;
    BaseStateStorage base;
    PhysicsTendencyFieldStorage tendencies;
    SurfaceDiagnosticFieldStorage surface;
    NestedGridStorage nested;
    NestedExecutionStorage nested_execution;
    DiagnosticFieldStorage diagnostics;
    SimulationRuntimeState runtime;
    double coriolis_f = 0.0;
    double time_s = 0.0;
    // Transient execution identity. It is never checkpointed and exists only
    // while the child level is bound into the process compatibility surface.

    SimulationStorageState()
        : nested(grid.nr, grid.nth, grid.nz)
    {
    }

    explicit SimulationStorageState(GridStorageState grid_in)
        : grid(std::move(grid_in)), nested(grid.nr, grid.nth, grid.nz)
    {
    }
};

struct SimulationStorageSnapshot
{
    SimulationStorageState state;
};

inline SimulationStorageSnapshot snapshot_simulation_storage(
    const SimulationStorageState& storage)
{
    return {storage};
}

inline void restore_simulation_storage(
    SimulationStorageState& storage,
    const SimulationStorageSnapshot& snapshot)
{
    SimulationStorageState restored(snapshot.state);
    storage = std::move(restored);
}

struct SimulationStateView
{
    GridStateView grid;
    PrognosticFieldsView fields;
    BaseStateView base;
    PhysicsTendencyFieldsView tendencies;
    double& coriolis_f;
    double& time_s;
    SurfaceDiagnosticFieldsView surface;

    // Execution binds the owners used by each operator. Legacy entry points
    // construct a complete process view before calling these interfaces.
    tmv::RadiationRuntimeState* radiation_runtime = nullptr;
    tmv::TurbulenceRuntimeState* turbulence_runtime = nullptr;
    tmv::BoundaryLayerRuntimeState* boundary_layer_runtime = nullptr;
    tmv::chaos::ChaosRuntimeState* chaos_runtime = nullptr;
    RayleighDampingRuntimeState* rayleigh_runtime = nullptr;
    tmv::MicrophysicsRuntimeState* microphysics_runtime = nullptr;
    tmv::AdvectionRuntimeState* advection_runtime = nullptr;
    tmv::DiffusionRuntimeState* diffusion_runtime = nullptr;
    tmv::TimeSteppingRuntimeState* time_stepping_runtime = nullptr;
    tmv::TerrainRuntimeState* terrain_runtime = nullptr;
    tmv::RadarRuntimeState* radar_runtime = nullptr;
    ComputeBackendRuntimeState* compute_runtime = nullptr;
    nimbus::parity::ParityRuntimeState* parity_runtime = nullptr;
    // Configuration stays caller-owned alongside the viewed fields/runtimes.
    RuntimeConfigurationState* configuration = nullptr;
    // Owning lazy derived-volume storage. Kept last so legacy aggregate views
    // retain their optional-runtime member ordering.
    DiagnosticFieldStorage* diagnostics = nullptr;
    // Per-simulation scheme and execution scratch. Appended for the same
    // aggregate-compatibility reason as diagnostics above.
    tmv::DynamicsRuntimeState* dynamics_runtime = nullptr;
    // Complete aggregate owner for cross-module transactions whose state is
    // intentionally stored directly in SimulationRuntimeState.
    SimulationRuntimeState* runtime_owner = nullptr;
    // Coordinate ownership is per level even though run configuration is
    // shared. This avoids competing copies of backend and science selection.
    tmv::domain::HorizontalDomainFrame* domain_frame = nullptr;
    // Time-integrated acoustic boundary memory belongs to the viewed level.
    // Kept last so legacy aggregate state views retain their field ordering.
    tmv::BoundaryConditionRuntimeState* boundary_condition_runtime = nullptr;
    GridMetrics* process_grid_metrics = nullptr;
    tmv::BudgetLevel budget_level = tmv::BudgetLevel::Root;
    bool defer_step_reports = false;
    SimulationStorageState* storage_owner = nullptr;
};

namespace tmv
{
template <class Runtime>
Runtime& require_level_runtime(Runtime* runtime, const char* owner)
{
    if (!runtime) throw std::invalid_argument(std::string("missing level owner: ") + owner);
    return *runtime;
}
inline RuntimeConfigurationState& require_run_configuration(const SimulationStateView& state)
{
    return require_level_runtime(state.configuration, "run configuration");
}
}

namespace tmv
{
inline void bind_flow_execution(FlowExecutionInputs& target, const SimulationStateView& source)
{
    const auto& config = tmv::require_run_configuration(source);
    target.compute_runtime = source.compute_runtime;
    using Boundary = tmv::init::config::LateralBC;
    target.periodic_x = config.lateral_bc_west == Boundary::Periodic || config.lateral_bc_east == Boundary::Periodic;
    target.periodic_y = config.lateral_bc_south == Boundary::Periodic || config.lateral_bc_north == Boundary::Periodic;
    const std::array<Boundary, 4> faces{config.lateral_bc_west, config.lateral_bc_east,
        config.lateral_bc_south, config.lateral_bc_north};
    for (std::size_t i = 0; i < faces.size(); ++i)
        target.open_lateral[i] = faces[i] == Boundary::Radiative || faces[i] == Boundary::ZeroGradient ||
            faces[i] == Boundary::LegacyOpen;
    target.open_top = config.top_bc == tmv::init::config::TopBC::Open;
    const auto& frame = tmv::require_level_runtime(source.domain_frame, "domain frame");
    target.frame_u_ms = frame.velocity_ms.u_ms;
    target.frame_v_ms = frame.velocity_ms.v_ms;
    target.earth_fixed_terrain = source.terrain_runtime != nullptr
        && source.terrain_runtime->config.horizontal_placement_id == "earth_fixed";
}
}

extern SimulationStorageState global_simulation_storage;

GridStateView make_grid_state_view(GridStorageState& grid);
SimulationStateView make_simulation_state_view(
    AtmosphericLevelStorage& storage,
    RuntimeConfigurationState& run_configuration,
    ComputeBackendRuntimeState* shared_compute = nullptr,
    nimbus::parity::ParityRuntimeState* shared_parity = nullptr);
SimulationStateView make_simulation_state_view(SimulationStorageState& storage);
SimulationStateView global_simulation_state();
