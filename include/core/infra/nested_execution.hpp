#pragma once

#include "core/infra/nested_boundary_forcing.hpp"
#include "core/infra/nested_interface_ledger.hpp"
#include "core/runtime/simulation_state.hpp"

#include <cstddef>
#include <cstdint>
#include <array>

namespace tmv::diagnostics
{
struct StormFeatureDetection;
struct StormContinuityState;
struct StormEnvelopeFaceMotion;
}

struct NestedChildInitializationReport
{
    tmv::domain::ConservedScalarProlongationReport theta;
    tmv::domain::ConservedScalarProlongationReport qv;
    tmv::domain::ConservedScalarProlongationReport qc;
    tmv::domain::ConservedScalarProlongationReport qr;
    tmv::domain::ConservedScalarProlongationReport qi;
    tmv::domain::ConservedScalarProlongationReport qs;
    tmv::domain::ConservedScalarProlongationReport qg;
    tmv::domain::ConservedScalarProlongationReport qh;
    tmv::domain::ConservedScalarProlongationReport tracer;
    tmv::domain::ConservedScalarProlongationReport tke;
    tmv::domain::ConservedMomentumProlongationReport momentum;
    tmv::domain::PressureDiagnosisReport pressure;
    std::size_t microphysics_private_scalar_count = 0;
    std::array<tmv::domain::ConservedScalarProlongationReport,
               NestedPrivateScalarRelaxationReportV1::maximum_scalars>
        microphysics_private{};
};

struct NestedParentMicrophysicsSnapshotV1
{
    static constexpr std::uint32_t schema_version = 2;
    static constexpr std::size_t maximum_scalars =
        NestedPrivateScalarRelaxationReportV1::maximum_scalars;

    bool available = false;
    double sample_time_s = 0.0;
    std::array<char,
               NestedPrivateScalarRelaxationReportV1::maximum_name_bytes>
        scheme_name{};
    std::size_t scalar_count = 0;
    std::array<std::array<char,
                          NestedPrivateScalarRelaxationReportV1::maximum_name_bytes>,
               maximum_scalars> name{};
    std::array<std::uint8_t, maximum_scalars> storage_semantic{};
    std::array<bool, maximum_scalars> positive_definite{};
    std::array<Field3D, maximum_scalars> fields{};
};

void capture_nested_parent_microphysics_snapshot(
    SimulationStorageState& storage,
    NestedParentMicrophysicsSnapshotV1& snapshot);

// Rebuilds the flat shared-z metric cache owned by the child level. This is
// called after relationship preparation and checkpoint restore; ordinary
// level switches then perform allocation-free metadata swaps.
void prepare_nested_child_process_grid_metrics(
    NestedExecutionStorage& nested);

// Temporarily binds the active child level to Nimbus' legacy process-state
// compatibility surface. All level-owned fields, clocks, metrics, diagnostics,
// and physics runtimes are exchanged in O(1); run-global compute and parity
// ownership stays in place. Root residency remains isolated while child
// residency is selected; child device authority is invalidated on exit.
SimulationStateView nested_child_execution_state(SimulationStorageState& storage);

class ScopedNestedChildResidency
{
public:
    explicit ScopedNestedChildResidency(SimulationStorageState& storage);
    ~ScopedNestedChildResidency() noexcept;
    ScopedNestedChildResidency(const ScopedNestedChildResidency&) = delete;
    ScopedNestedChildResidency& operator=(const ScopedNestedChildResidency&) = delete;
private:
    SimulationStorageState& storage_;
};

using NestedChildStepKernel = void (*)(
    const SimulationStateView& state,
    double child_step_s,
    double child_start_time_s,
    void* context);

struct NestedChildSubstepReport
{
    double start_time_s = 0.0;
    double end_time_s = 0.0;
    int completed_child_substeps = 0;
    NestedChildRelaxationReport interface_source;
};

struct NestedChildRadiationMovementReportV1
{
    static constexpr std::uint32_t schema_version = 1;

    bool active = false;
    double sample_time_s = 0.0;
    std::size_t newly_admitted_cells = 0;
    double previous_child_held_power_w = 0.0;
    double newly_admitted_target_power_w = 0.0;
    double newly_admitted_committed_power_w = 0.0;
    double newly_admitted_storage_rounding_w = 0.0;
    double committed_child_held_power_w = 0.0;
};

struct NestedChildBoundaryLayerMovementReportV1
{
    static constexpr std::uint32_t schema_version = 2;

    bool active = false;
    double sample_time_s = 0.0;
    std::size_t newly_admitted_cells = 0;
    std::size_t newly_admitted_columns = 0;
    double newly_admitted_target_x_force_n = 0.0;
    double newly_admitted_committed_x_force_n = 0.0;
    double newly_admitted_x_force_storage_rounding_n = 0.0;
    double newly_admitted_target_y_force_n = 0.0;
    double newly_admitted_committed_y_force_n = 0.0;
    double newly_admitted_y_force_storage_rounding_n = 0.0;
    double newly_admitted_target_theta_inventory_rate_kgk_s = 0.0;
    double newly_admitted_committed_theta_inventory_rate_kgk_s = 0.0;
    double newly_admitted_theta_storage_rounding_kgk_s = 0.0;
    double newly_admitted_target_vapor_mass_rate_kg_s = 0.0;
    double newly_admitted_committed_vapor_mass_rate_kg_s = 0.0;
    double newly_admitted_vapor_storage_rounding_kg_s = 0.0;
    double newly_admitted_target_cloud_liquid_mass_rate_kg_s = 0.0;
    double newly_admitted_committed_cloud_liquid_mass_rate_kg_s = 0.0;
    double newly_admitted_cloud_liquid_storage_rounding_kg_s = 0.0;
    double newly_admitted_target_cloud_ice_mass_rate_kg_s = 0.0;
    double newly_admitted_committed_cloud_ice_mass_rate_kg_s = 0.0;
    double newly_admitted_cloud_ice_storage_rounding_kg_s = 0.0;
    double committed_child_x_force_n = 0.0;
    double committed_child_y_force_n = 0.0;
    double committed_child_theta_inventory_rate_kgk_s = 0.0;
    double committed_child_vapor_mass_rate_kg_s = 0.0;
    double committed_child_cloud_liquid_mass_rate_kg_s = 0.0;
    double committed_child_cloud_ice_mass_rate_kg_s = 0.0;
};

struct NestedChildChaosMovementReportV1
{
    static constexpr std::uint32_t schema_version = 1;

    bool active = false;
    bool pbl_base_cache_active = false;
    std::uint64_t stochastic_epoch = 0;
    double noise_elapsed_s = 0.0;
    std::int64_t earth_origin_i_cells = 0;
    std::int64_t earth_origin_j_cells = 0;
    std::size_t newly_admitted_cache_cells = 0;
    std::size_t newly_admitted_cache_columns = 0;
    double newly_admitted_target_x_force_n = 0.0;
    double newly_admitted_committed_x_force_n = 0.0;
    double newly_admitted_x_force_storage_rounding_n = 0.0;
    double newly_admitted_target_y_force_n = 0.0;
    double newly_admitted_committed_y_force_n = 0.0;
    double newly_admitted_y_force_storage_rounding_n = 0.0;
    double newly_admitted_target_theta_inventory_rate_kgk_s = 0.0;
    double newly_admitted_committed_theta_inventory_rate_kgk_s = 0.0;
    double newly_admitted_theta_storage_rounding_kgk_s = 0.0;
    double newly_admitted_target_vapor_mass_rate_kg_s = 0.0;
    double newly_admitted_committed_vapor_mass_rate_kg_s = 0.0;
    double newly_admitted_vapor_storage_rounding_kg_s = 0.0;
};

struct NestedChildMicrophysicsMovementReportV1
{
    static constexpr std::uint32_t schema_version = 1;
    static constexpr std::size_t maximum_private_scalars = 16;

    bool active = false;
    bool theta_storage_residual_active = false;
    std::size_t private_scalar_count = 0;
    std::size_t newly_admitted_private_values = 0;
    std::size_t newly_admitted_theta_residual_values = 0;
    std::array<double, maximum_private_scalars>
        newly_admitted_target_inventory{};
    std::array<double, maximum_private_scalars>
        newly_admitted_committed_inventory{};
    std::array<double, maximum_private_scalars>
        newly_admitted_storage_rounding{};
};

struct NestedChildPrecipitationMovementReportV1
{
    static constexpr std::uint32_t schema_version = 1;

    bool active = false;
    double sample_time_s = 0.0;
    std::size_t newly_admitted_columns = 0;
    double previous_child_accumulated_water_m3 = 0.0;
    double discarded_child_accumulated_water_m3 = 0.0;
    double retained_child_accumulated_water_m3 = 0.0;
    double newly_admitted_parent_accumulated_water_m3 = 0.0;
    double committed_child_accumulated_water_m3 = 0.0;
    double previous_child_precipitation_volume_rate_m3s = 0.0;
    double discarded_child_precipitation_volume_rate_m3s = 0.0;
    double retained_child_precipitation_volume_rate_m3s = 0.0;
    double newly_admitted_parent_precipitation_volume_rate_m3s = 0.0;
    double committed_child_precipitation_volume_rate_m3s = 0.0;
};

struct NestedChildMovementReportV5
{
    static constexpr std::uint32_t schema_version = 5;

    std::uint64_t movement_sequence = 0;
    int parent_shift_i = 0;
    int parent_shift_j = 0;
    int child_shift_i = 0;
    int child_shift_j = 0;
    int previous_parent_origin_i = 0;
    int previous_parent_origin_j = 0;
    int committed_parent_origin_i = 0;
    int committed_parent_origin_j = 0;
    std::size_t overlap_scalar_cells = 0;
    std::size_t newly_admitted_scalar_cells = 0;
    std::size_t overlap_x_faces = 0;
    std::size_t newly_admitted_x_faces = 0;
    std::size_t overlap_y_faces = 0;
    std::size_t newly_admitted_y_faces = 0;
    std::size_t overlap_z_faces = 0;
    std::size_t newly_admitted_z_faces = 0;
    std::size_t rollback_journal_values = 0;
    std::size_t rollback_journal_bytes = 0;
    std::size_t full_primary_owner_values = 0;
    NestedTwoWayRelocationReportV1 two_way_inventory;
    NestedChildInitializationReport parent_target;
    NestedChildRelaxationReport boundary_source;
    NestedChildRadiationMovementReportV1 radiation;
    NestedChildBoundaryLayerMovementReportV1 boundary_layer;
    NestedChildChaosMovementReportV1 chaos;
    NestedChildMicrophysicsMovementReportV1 microphysics;
    NestedChildPrecipitationMovementReportV1 precipitation;
};

struct NestedTrackedEnvelopeV1
{
    double sample_time_s = 0.0;
    bool selected_feature_available = false;
    bool continuity_authoritative = false;
    bool motion_available = false;
    bool ambiguous = false;
    bool branch_candidate_overflow = false;
    bool topology_reorganization = false;
    double confidence = 0.0;
    double envelope_west_earth_m = 0.0;
    double envelope_east_earth_m = 0.0;
    double envelope_south_earth_m = 0.0;
    double envelope_north_earth_m = 0.0;
    double velocity_x_earth_ms = 0.0;
    double velocity_y_earth_ms = 0.0;
    bool   envelope_face_motion_available = false;
    double envelope_face_motion_interval_s = 0.0;
    double envelope_west_earth_velocity_ms = 0.0;
    double envelope_east_earth_velocity_ms = 0.0;
    double envelope_south_earth_velocity_ms = 0.0;
    double envelope_north_earth_velocity_ms = 0.0;
};

struct NestedChildMovementAuthorityReportV1
{
    tmv::domain::StormWindowMovementDecision decision;
    bool movement_committed = false;
    NestedChildMovementReportV5 movement;
};

// Converts the detector-selected branch and the checkpointed continuity owner
// into the one Earth-frame envelope consumed by nested movement. Keeping this
// adapter outside the runtime driver prevents qualification tools and ordinary
// execution from assigning different branches or velocity frames.
NestedTrackedEnvelopeV1 prepare_nested_tracked_envelope(
    const tmv::diagnostics::StormFeatureDetection& detection,
    const tmv::diagnostics::StormContinuityState& continuity,
    const tmv::diagnostics::StormEnvelopeFaceMotion& face_motion,
    double sample_time_s,
    double earth_origin_x_m,
    double earth_origin_y_m,
    double frame_velocity_x_ms,
    double frame_velocity_y_ms);

// Compatibility adapter for callers without a previous accepted envelope.
// Such callers publish canonical unavailable face-motion evidence.
NestedTrackedEnvelopeV1 prepare_nested_tracked_envelope(
    const tmv::diagnostics::StormFeatureDetection& detection,
    const tmv::diagnostics::StormContinuityState& continuity,
    double sample_time_s,
    double earth_origin_x_m,
    double earth_origin_y_m,
    double frame_velocity_x_ms,
    double frame_velocity_y_ms);

// Adapts a detector whose horizontal coordinates and fitted velocity are
// already Earth-relative. This is the authoritative path for a relocatable
// refined child: discrete window movement must not translate the physical
// storm track or add the child origin twice.
NestedTrackedEnvelopeV1 prepare_nested_tracked_envelope_earth(
    const tmv::diagnostics::StormFeatureDetection& detection,
    const tmv::diagnostics::StormContinuityState& continuity,
    const tmv::diagnostics::StormEnvelopeFaceMotion& face_motion,
    double sample_time_s);

NestedTrackedEnvelopeV1 prepare_nested_tracked_envelope_earth(
    const tmv::diagnostics::StormFeatureDetection& detection,
    const tmv::diagnostics::StormContinuityState& continuity,
    double sample_time_s);

// Advances one child phase against an already completed parent bracket. The
// supplied kernel executes with the child process-bound but does not own the
// clock. On return, conserved parent forcing closes the new phase endpoint and
// the bracket/child clock commit together.
void advance_nested_child_substep(
    SimulationStorageState& storage,
    const PrognosticFieldStorage& previous_parent_fields,
    const tmv::domain::NestedRelaxationConfigV1& relaxation,
    NestedChildStepKernel kernel,
    void* kernel_context,
    NestedChildSubstepReport& report,
    const NestedParentMicrophysicsSnapshotV1* previous_parent_microphysics = nullptr);

void prepare_nested_execution_relationship(
    SimulationStorageState& storage,
    const tmv::domain::OneWayNestRelationshipV1& relationship,
    double previous_parent_time_s);

void require_valid_nested_execution_relationship(
    const SimulationStorageState& storage);

// Opens the next realized parent interval after the parent level has advanced
// from a synchronized endpoint. The caller retains the previous parent fields
// used for temporal boundary forcing; this owner commits only the bracket and
// the exact realized child interval.
void roll_nested_parent_interval(
    SimulationStorageState& storage);

// Relocates a synchronized child by whole parent cells. The verification
// interior of the geometric overlap retains its native C-grid state, newly
// admitted owners receive simultaneous conservative parent targets, and the
// parent-owned outer ring is re-prescribed as one rollback-safe transaction.
// Only physics owners with an explicit persistent-state movement or
// generation-invalidation contract are admitted.
void move_nested_child_integer_parent_cells(
    SimulationStorageState& storage,
    int parent_shift_i,
    int parent_shift_j,
    const tmv::domain::NestedRelaxationConfigV1& relaxation,
    NestedChildMovementReportV5& report);

tmv::domain::StormWindowMovementEvidence
prepare_nested_child_movement_evidence(
    const SimulationStorageState& storage,
    const NestedTrackedEnvelopeV1& envelope);

// Evaluates checkpointed tracker hysteresis only at a synchronized child
// endpoint. A ready decision is expressed in parent cells and is published to
// movement authority only after the complete child relocation succeeds.
void evaluate_and_move_nested_child_from_tracker(
    SimulationStorageState& storage,
    const NestedTrackedEnvelopeV1& envelope,
    const tmv::domain::StormWindowMovementPolicy& policy,
    const tmv::domain::NestedRelaxationConfigV1& relaxation,
    NestedChildMovementAuthorityReportV1& report);

void initialize_nested_shared_level_reference_state(
    SimulationStorageState& storage);

void initialize_nested_child_physical_primary_state(
    SimulationStorageState& storage,
    const PrognosticFieldStorage& previous_parent_fields,
    NestedChildInitializationReport& report);

// Builds fresh child-grid module objects and publishes the initial held PBL
// and radiation sources at the child's actual clock. Parent histories,
// ledgers, caches, and diagnostic products are never cloned as child state.
void synchronize_nested_child_chaos_clock(SimulationStorageState& storage, double child_time_s);

void initialize_nested_child_level_runtime(
    SimulationStorageState& storage);

// Replaces factory-seeded scheme-private child scalars with the exact
// conserved previous/current parent-bracket target.
void initialize_nested_child_microphysics_private_state(
    SimulationStorageState& storage,
    const PrognosticFieldStorage& previous_parent_fields,
    const NestedParentMicrophysicsSnapshotV1& previous_parent_microphysics,
    NestedChildInitializationReport& report);

void require_nested_child_level_runtime_ready(
    const SimulationStorageState& storage);

// Recreates the child-grid dynamics operator omitted from durable checkpoints
// without resetting any restored module history, held source, or ledger.
void rehydrate_nested_child_transient_operators(
    SimulationStorageState& storage);

// Runs the production dynamics/physics step as one child bracket phase.
void advance_nested_child_dynamics_substep(
    SimulationStorageState& storage,
    const PrognosticFieldStorage& previous_parent_fields,
    const tmv::domain::NestedRelaxationConfigV1& relaxation,
    NestedChildSubstepReport& report,
    const NestedParentMicrophysicsSnapshotV1* previous_parent_microphysics = nullptr);

void apply_nested_child_physical_relaxation(
    SimulationStorageState& storage,
    const PrognosticFieldStorage& previous_parent_fields,
    const tmv::domain::NestedRelaxationConfigV1& config,
    double child_step_s,
    NestedChildRelaxationReport& report,
    const NestedParentMicrophysicsSnapshotV1* previous_parent_microphysics = nullptr);
