#pragma once

#include "core/infra/domain_recycling.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "terrain/earth_fixed_geometry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tmv::domain
{

// One source of truth for every primary atmosphere owner participating in a
// flat or Earth-fixed recycled-window transaction. The same list generates the
// enum, names, profile storage, and live Field3D bindings.
#define NIMBUS_D5_PRIMARY_OWNER_LIST(X) \
    X(rho)                              \
    X(p)                                \
    X(u)                                \
    X(w)                                \
    X(v)                                \
    X(tracer)                           \
    X(theta)                            \
    X(qv)                               \
    X(qc)                               \
    X(qr)                               \
    X(qi)                               \
    X(qs)                               \
    X(qh)                               \
    X(qg)                               \
    X(tke)

enum class HomogeneousPrimaryOwner : std::size_t
{
#define NIMBUS_D5_DECLARE_OWNER(name) name,
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_DECLARE_OWNER)
#undef NIMBUS_D5_DECLARE_OWNER
    count
};

constexpr std::size_t homogeneous_primary_owner_count() noexcept
{
    return static_cast<std::size_t>(HomogeneousPrimaryOwner::count);
}

const std::array<const char*, homogeneous_primary_owner_count()>&
homogeneous_primary_owner_names() noexcept;

struct HomogeneousPrimaryProfiles
{
    std::array<std::vector<float>, homogeneous_primary_owner_count()> values;

    std::vector<float>& operator[](HomogeneousPrimaryOwner owner) noexcept
    {
        return values[static_cast<std::size_t>(owner)];
    }

    const std::vector<float>& operator[](HomogeneousPrimaryOwner owner) const noexcept
    {
        return values[static_cast<std::size_t>(owner)];
    }
};

struct TerrainAwarePrimaryFields
{
    std::array<Field3D, homogeneous_primary_owner_count()> values;

    Field3D& operator[](HomogeneousPrimaryOwner owner) noexcept
    {
        return values[static_cast<std::size_t>(owner)];
    }

    const Field3D& operator[](HomogeneousPrimaryOwner owner) const noexcept
    {
        return values[static_cast<std::size_t>(owner)];
    }
};

/**
 * Construct a thermodynamically self-consistent flat, horizontally
 * homogeneous leading environment from the checkpointed base profiles.
 *
 * p/theta/qv/u/v come directly from the base state after strict physical
 * validation. rho is re-diagnosed from the stored-float p/theta values with
 * Nimbus's dry EOS, exactly as initial closure does. Environmental w, tracer,
 * and condensate species are zero. TKE must be supplied explicitly because it
 * is active state and Nimbus does not store a universal TKE base profile.
 */
HomogeneousPrimaryProfiles prepare_flat_homogeneous_primary_profiles(
    const SimulationStorageState& storage,
    const std::vector<double>& environmental_tke_profile);

/** Bind every primary Field3D owner to the matching generated profile. */
std::vector<HomogeneousFieldShiftBinding> bind_homogeneous_primary_fields(
    SimulationStorageState& storage,
    const HomogeneousPrimaryProfiles& profiles);

TerrainAwarePrimaryFields prepare_terrain_aware_primary_fields(
    const SimulationStorageState& storage,
    const TerrainMetrics3D& post_shift_metrics,
    const std::vector<double>& environmental_tke_profile);

std::vector<HomogeneousFieldShiftBinding>
bind_terrain_aware_primary_fields(
    SimulationStorageState& storage,
    const TerrainAwarePrimaryFields& fields);

const char* homogeneous_primary_profile_algorithm_id() noexcept;

struct PhysicalRegionInventoryLedger
{
    DomainRecyclingPrivateInventory private_inventory;
    ConservationBudget inventories;
    // Earth-relative linear momentum and domain-center-referenced angular
    // momentum over the same retiring scalar control volumes.
    MomentumBudget momentum;
    MoistMixtureEnergyBudgetV2 moist_mixture_energy;
    // Deferred effective heat carried by the checkpointed microphysics theta
    // compensation residual over the same retiring cells [J]. It is not part
    // of visible-state physical-v2 energy until released into theta.
    double theta_storage_residual_energy_j = 0.0;
    std::size_t physical_columns = 0;
    std::size_t physical_cells = 0;
};

// Historical public name retained because the retiring half is also the
// durable Earth-tile inventory.  The admitted half uses the identical
// currencies and therefore deliberately shares the representation.
using TrailingPhysicalExportLedger = PhysicalRegionInventoryLedger;

/**
 * Exact instantaneous control-volume accounting for one recycle.
 *
 * Recycling advances no model time and applies no physical tendency.  It
 * removes one set of Earth cells and admits an equally sized set initialized
 * from the declared environment.  Consequently its only physical ledger is
 * a Reynolds-control-volume exchange:
 *
 *   post inventory = pre inventory - retiring + admitted.
 *
 * Angular momentum is additionally expressed about the fixed Earth origin,
 * because the ordinary diagnostic's computational-domain-center reference
 * changes when the window origin moves.
 */
struct DomainRecyclingControlVolumeExchangeLedger
{
    PhysicalRegionInventoryLedger pre_shift;
    PhysicalRegionInventoryLedger retiring;
    PhysicalRegionInventoryLedger admitted;
    PhysicalRegionInventoryLedger expected_post_shift;
    double pre_shift_fixed_earth_angular_momentum_z = 0.0;
    double retiring_fixed_earth_angular_momentum_z = 0.0;
    double admitted_fixed_earth_angular_momentum_z = 0.0;
    double expected_post_shift_fixed_earth_angular_momentum_z = 0.0;
};

/**
 * Reduce retiring physical scalar cells with the exact production
 * ConservationBudget basis and live flat or mapped control-volume geometry.
 */
TrailingPhysicalExportLedger compute_trailing_physical_export_ledger(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan);

/**
 * Prepare the complete C-grid exchange without cloning the full model state.
 * Flat leading values use stored-float profiles; the Earth-fixed overload uses
 * destination-indexed fields and detached post-shift terrain geometry.
 */
struct PreparedMicrophysicsDomainRecyclingState;

DomainRecyclingControlVolumeExchangeLedger
compute_domain_recycling_control_volume_exchange_ledger(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& leading_profiles,
    const HorizontalDomainFrame& pre_shift_frame,
    const PreparedMicrophysicsDomainRecyclingState* private_leading = nullptr);

DomainRecyclingControlVolumeExchangeLedger
compute_terrain_aware_domain_recycling_control_volume_exchange_ledger(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& leading_fields,
    const TerrainMetrics3D& post_shift_metrics,
    const HorizontalDomainFrame& pre_shift_frame,
    const PreparedMicrophysicsDomainRecyclingState* private_leading = nullptr);

/** Fail closed unless live post-commit inventories close the prepared ledger. */
void verify_domain_recycling_control_volume_exchange_after_commit(
    const SimulationStateView& post_shift_state,
    const WholeCellShiftPlan& plan,
    const HorizontalDomainFrame& post_shift_frame,
    const DomainRecyclingControlVolumeExchangeLedger& ledger);

const char* trailing_scalar_export_ledger_basis_id() noexcept;
const char* trailing_momentum_export_ledger_basis_id() noexcept;
const char* trailing_energy_export_ledger_basis_id() noexcept;
const char* trailing_theta_residual_export_ledger_basis_id() noexcept;
const char* domain_recycling_control_volume_exchange_basis_id() noexcept;

// Root-level ownership contract for the later atomic recycling commit. This is kept
// in code because primary fields are only one part of a restart-exact model:
// held physics sources, scheme history, derived products, device residency,
// coordinate history, and durable output each require an explicit disposition.
enum class DomainRecyclingOwnerAction
{
    Preserve,
    ShiftWithLeadingState,
    ShiftAndExportEarthTile,
    RebuildFromShiftedState,
    InvalidateAndRecompute,
    ShiftPersistentInvalidateTransient,
    TranslateEarthCoordinates,
    InvalidateDeviceResidency,
    RejectActiveMode,
};

enum class DomainRecyclingOwnerReadiness
{
    ReadyForTransaction,
    Partial,
    Required,
    ExplicitlyUnavailable,
};

struct DomainRecyclingOwnerContract
{
    std::string_view id;
    DomainRecyclingOwnerAction action;
    DomainRecyclingOwnerReadiness readiness;
    std::string_view scope;
};

constexpr std::size_t domain_recycling_owner_contract_count() noexcept
{
    return 28;
}

const std::array<DomainRecyclingOwnerContract,
                 domain_recycling_owner_contract_count()>&
domain_recycling_owner_contracts() noexcept;

std::size_t remaining_domain_recycling_owner_contracts() noexcept;

struct PreparedDomainRecyclingTransactionProvenance
{
    bool active = false;
    SimulationRuntimeState* runtime = nullptr;
    DomainRecyclingTransactionProvenance committed;
};

/**
 * Prepare the restart-exact identity of one already-durable swath capture.
 * Preparation validates the exact plan, Earth displacement, ledger, manifest
 * identity, and monotone sequence without changing live state.
 */
PreparedDomainRecyclingTransactionProvenance
prepare_domain_recycling_transaction_provenance(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HorizontalDomainFrame& pre_shift_frame,
    const DomainRecyclingControlVolumeExchangeLedger& ledger,
    std::string_view transaction_id,
    std::string_view publication_generation,
    std::string_view storage_format,
    std::string_view swath_algorithm,
    double simulation_time_s,
    std::uint64_t transaction_sequence,
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64);

/** Append provenance to the allocation-free common recycling commit boundary. */
bool append_domain_recycling_transaction_provenance_binding(
    const SimulationStateView& state,
    PreparedDomainRecyclingTransactionProvenance& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings);

struct PreparedStormTrackingDomainRecyclingState
{
    bool active = false;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int physical_i_begin = 0;
    int physical_i_end = 0;
    int physical_j_begin = 0;
    int physical_j_end = 0;
    int shift_i_cells = 0;
    int shift_j_cells = 0;
    double displacement_x_m = 0.0;
    double displacement_y_m = 0.0;
    SimulationRuntimeState* runtime = nullptr;
    tmv::diagnostics::StormContinuityState committed_continuity;
    StormWindowMovementState committed_movement;
    std::vector<std::uint8_t> committed_partition;
};

/**
 * Express all retained storm association history in the post-shift grid
 * frame, shift the physical-column partition with zero leading membership,
 * and prepare the movement-authority commit matching the exact recycle plan.
 */
PreparedStormTrackingDomainRecyclingState
prepare_storm_tracking_domain_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const StormWindowMovementDecision& decision);

/** Append the prepared persistent transform to the common atomic commit. */
bool append_storm_tracking_domain_recycling_binding(
    const SimulationStateView& state,
    PreparedStormTrackingDomainRecyclingState& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings);

const char* storm_tracking_domain_recycling_algorithm_id() noexcept;

struct PreparedNestedGridDomainRecyclingState
{
    bool active = false;
    NestedExecutionStorage* nested = nullptr;
    tmv::domain::OneWayNestRelationshipV1 expected_relationship;
    tmv::domain::OneWayNestRelationshipV1 committed_relationship;
    int expected_recycling_shift_i = 0;
    int expected_recycling_shift_j = 0;
    int committed_recycling_shift_i = 0;
    int committed_recycling_shift_j = 0;
    double expected_parent_time_s = 0.0;
    double expected_child_time_s = 0.0;
    std::uint64_t preserved_state_fingerprint = 0;
    std::array<const void*, 11> scheme_identity{};
    std::unique_ptr<PreparedNestedActiveTerrainCommitV1> mapped_rebase;
    std::uint64_t expected_child_terrain_generation = 0;
};

PreparedNestedGridDomainRecyclingState
prepare_nested_grid_domain_recycling(SimulationStorageState&                           storage,
                                     const WholeCellShiftPlan&                         plan,
                                     const HorizontalDomainFrame&                      post_shift_frame,
                                     const DomainRecyclingControlVolumeExchangeLedger* exchange = nullptr);

StormWindowMovementEvidence with_nested_parent_recycling_bounds(const OneWayNestRelationshipV1& relationship,
                                                                StormWindowMovementEvidence     evidence);

bool append_nested_grid_domain_recycling_binding(
    SimulationStorageState& storage,
    PreparedNestedGridDomainRecyclingState& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings);

bool can_commit_nested_grid_domain_recycling(
    const SimulationStorageState& storage,
    const PreparedNestedGridDomainRecyclingState& prepared) noexcept;

const char* nested_grid_domain_recycling_algorithm_id() noexcept;

struct PreparedRadiationDomainRecyclingState
{
    bool active = false;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int shift_i_cells = 0;
    int shift_j_cells = 0;
    double sample_time_s = 0.0;
    bool terrain_aware = false;
    std::vector<float> leading_theta_tendency_s;
    std::vector<float> leading_cell_power_w;
    Field3D terrain_leading_theta_tendency_s;
    Field3D terrain_leading_cell_power_w;
    double committed_held_atmospheric_power_w = 0.0;
};

/**
 * Prepare the complete persistent radiation owner for a flat homogeneous recycle
 * transaction. Active radiation evaluates the leading environment through
 * the production column evaluator at the held field's original sample time,
 * then reduces the exact planned retained-plus-leading FP32 power field.
 * Nothing in the live runtime is mutated during preparation.
 */
PreparedRadiationDomainRecyclingState
prepare_flat_homogeneous_radiation_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& primary_profiles);

PreparedRadiationDomainRecyclingState
prepare_terrain_aware_radiation_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& primary_fields,
    const TerrainMetrics3D& post_shift_metrics);

/** Append held tendency/power and its aggregate to one atomic recycling commit. */
bool append_radiation_domain_recycling_bindings(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const PreparedRadiationDomainRecyclingState& prepared,
    std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    std::vector<HomogeneousScalarCommitBinding>& scalar_bindings);

/** Append exact pre-overwrite capture of both persistent radiation fields. */
bool append_radiation_trailing_export_bindings(
    const SimulationStateView& state,
    std::vector<float>& theta_tendency_values,
    std::vector<float>& cell_power_values,
    std::vector<TrailingFieldExportBinding>& bindings);

const char* radiation_domain_recycling_algorithm_id() noexcept;

struct PreparedBoundaryLayerDomainRecyclingState
{
    bool active = false;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int shift_i_cells = 0;
    int shift_j_cells = 0;
    double sample_time_s = 0.0;
    bool terrain_aware = false;

    std::vector<float> leading_u_force_n;
    std::vector<float> leading_v_force_n;
    std::vector<float> leading_theta_inventory_rate_kgk_s;
    std::vector<float> leading_vapor_mass_rate_kg_s;
    std::vector<float> leading_dtke_s;

    float leading_skin_theta = 0.0f;
    float leading_skin_qv = 0.0f;
    SlabColumnState leading_slab_column;
    float leading_ustar = 0.0f;
    float leading_tau_u = 0.0f;
    float leading_tau_v = 0.0f;
    float leading_sensible_heat_flux = 0.0f;
    float leading_latent_heat_flux = 0.0f;
    float leading_moisture_flux = 0.0f;
    float leading_pbl_height = 0.0f;

    std::array<Field3D, 5> terrain_leading_fields;
    std::array<std::vector<float>, 9> terrain_leading_surfaces;
    std::vector<SlabColumnState> terrain_leading_slab_columns;
};

PreparedBoundaryLayerDomainRecyclingState
prepare_flat_homogeneous_boundary_layer_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& primary_profiles);

PreparedBoundaryLayerDomainRecyclingState
prepare_terrain_aware_boundary_layer_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& primary_fields,
    const TerrainMetrics3D& post_shift_metrics);

bool append_boundary_layer_domain_recycling_bindings(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const PreparedBoundaryLayerDomainRecyclingState& prepared,
    std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings,
    std::vector<HomogeneousBooleanCommitBinding>& flag_bindings);

bool append_boundary_layer_trailing_export_bindings(
    const SimulationStateView& state,
    std::array<std::vector<float>, 5>& packed_values,
    std::vector<TrailingFieldExportBinding>& bindings);

const char* boundary_layer_domain_recycling_algorithm_id() noexcept;

/**
 * Append the complete state-dependent turbulence invalidation to the recycling
 * transaction. Configuration, scheme ownership, cadence history, and
 * completed/cumulative energy ledgers remain intact. Held/candidate nonlinear
 * tendencies, device-source hints, predictive stability state, and borrowed
 * geometry pointers invalidate only after the common transaction preflight.
 */
bool append_turbulence_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousScalarCommitBinding>& scalar_bindings,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* turbulence_domain_recycling_algorithm_id() noexcept;

/** Append the cache invalidation for non-evolving chaos schemes. */
bool append_non_evolving_chaos_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* chaos_domain_recycling_algorithm_id() noexcept;

/** Detached, fully rendered post-shift evolving-chaos owner. */
struct PreparedEvolvingChaosDomainRecyclingState
{
    tmv::chaos::ChaosRuntimeState* runtime = nullptr;
    std::unique_ptr<tmv::chaos::ChaosScheme> scheme;
    bool preserve_shifted_pbl_base = false;
};

/**
 * Clone and translate boundary-layer/full-stochastic Earth noise without
 * mutating the live runtime. Newly encountered controls are reconstructed
 * from the exact counter-addressed AR(1) history.
 */
PreparedEvolvingChaosDomainRecyclingState
prepare_evolving_chaos_domain_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan);

/** Shift a valid deterministic PBL source/flux cache with the same leading
 * environmental closure used by the live boundary-layer publication. */
bool append_evolving_chaos_pbl_cache_domain_recycling_bindings(
    const SimulationStateView& state,
    const PreparedBoundaryLayerDomainRecyclingState& boundary_layer,
    PreparedEvolvingChaosDomainRecyclingState& chaos,
    std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings);

/** Commit the prepared scheme swap and dependent-cache invalidation atomically. */
bool append_evolving_chaos_domain_recycling_bindings(
    const SimulationStateView& state,
    PreparedEvolvingChaosDomainRecyclingState& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings);

bool append_rayleigh_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousBooleanCommitBinding>& flag_bindings);

/**
 * Append allocation-stable invalidation of advection state that is meaningful
 * only for the pre-recycle field generation. Scheme/configuration ownership,
 * cumulative transport ledgers, ledger step counts, and performance history
 * remain intact. An outstanding fused/batched call rejects during preflight.
 */
bool append_advection_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* advection_domain_recycling_algorithm_id() noexcept;

/**
 * Append allocation-stable invalidation of diffusion outputs and transactional
 * candidates tied to the pre-recycle field generation. Scheme/configuration
 * and cumulative momentum/moist-energy ledgers remain exact history.
 */
bool append_diffusion_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* diffusion_domain_recycling_algorithm_id() noexcept;

bool append_time_stepping_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* time_stepping_domain_recycling_algorithm_id() noexcept;

/**
 * Preserve radar operator/configuration and Earth-fixed instrument geometry,
 * while invalidating state-derived public, swap, and scheme-private products.
 */
bool append_radar_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* radar_domain_recycling_algorithm_id() noexcept;

/** Invalidate checkpointed lazy diagnostic volumes without releasing them. */
bool append_derived_diagnostic_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* derived_diagnostic_domain_recycling_algorithm_id() noexcept;

/**
 * Preserve the selected dynamics operator while invalidating all reusable
 * tendency, reduction, upload, parity-replay, and residency storage tied to
 * the pre-recycle field generation. Every allocation retains its capacity.
 */
bool append_dynamics_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

const char* dynamics_domain_recycling_algorithm_id() noexcept;

struct PreparedMicrophysicsDomainRecyclingState
{
    bool active = false;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int shift_i_cells = 0;
    int shift_j_cells = 0;
    bool terrain_aware = false;
    std::vector<float> zero_leading_profile;
    Field3D zero_leading_field;
    std::vector<std::string> private_scalar_names;
    std::vector<std::vector<float>> leading_private_scalars;
    std::vector<Field3D> terrain_leading_private_scalars;
};

PreparedMicrophysicsDomainRecyclingState
prepare_flat_homogeneous_microphysics_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& primary_profiles);

PreparedMicrophysicsDomainRecyclingState
prepare_terrain_aware_microphysics_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& primary_fields,
    const TerrainMetrics3D& destination_metrics);

bool append_prepared_microphysics_domain_recycling_bindings(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const PreparedMicrophysicsDomainRecyclingState& prepared,
    std::vector<HomogeneousFieldShiftBinding>& bindings);

/**
 * Append allocation-stable invalidation of microphysics products that belong
 * only to the completed pre-recycle field generation. Prognostic private
 * moments and the theta residual are shifted separately; completed reports
 * and cumulative fallout history remain intact.
 */
bool append_microphysics_domain_recycling_invalidation_binding(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings);

bool append_prepared_microphysics_trailing_export_bindings(
    const SimulationStateView& state,
    std::vector<std::vector<float>>& packed_values,
    std::vector<TrailingFieldExportBinding>& bindings);

const char* microphysics_domain_recycling_algorithm_id() noexcept;

bool append_surface_precipitation_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings);

bool append_surface_precipitation_trailing_export_bindings(
    const SimulationStateView& state,
    std::vector<float>& precipitation_rate_values,
    std::vector<float>& accumulated_rainfall_values,
    std::vector<float>& accumulated_rainfall_residual_values,
    std::vector<TrailingSurfaceExportBinding>& bindings);

const char* surface_precipitation_domain_recycling_algorithm_id() noexcept;

/**
 * Append the checkpointed microphysics theta-compensation residual to the
 * same atomic volume transaction as the prognostic atmosphere. Retained
 * residuals move exactly; leading columns receive exact zero because they
 * have no prior FP32 rounding history. Returns false only when radiation is
 * inactive and no residual storage exists.
 */
bool append_microphysics_domain_recycling_binding(
    const SimulationStateView& state,
    const std::vector<float>& zero_leading_profile,
    std::vector<HomogeneousFieldShiftBinding>& bindings);

/** Append the same persistent residual to exact pre-overwrite raw capture. */
bool append_microphysics_trailing_export_binding(
    const SimulationStateView& state,
    std::vector<float>& packed_values,
    std::vector<TrailingFieldExportBinding>& bindings);

/**
 * Complete allocation-owning preparation for one Cartesian C-grid recycle
 * move. Flat profiles and the dry Earth-fixed terrain foundation share this
 * owner. Every potentially throwing owner validation, leading-column
 * evaluation, raw retiring capture, and binding append is completed before
 * the caller publishes the durable retiring swath. The object is heap-owned
 * so binding pointers into its prepared profiles remain stable.
 */
struct PreparedFlatHomogeneousDomainRecycling
{
    SimulationStorageState* storage = nullptr;
    ComputeBackendKind authoritative_backend = ComputeBackendKind::Cpu;
    WholeCellShiftPlan plan;
    HorizontalDomainFrame pre_shift_frame;
    HorizontalDomainFrame post_shift_frame;
    HomogeneousPrimaryProfiles primary_profiles;
    TerrainAwarePrimaryFields terrain_primary_fields;
    PreparedEarthFixedTerrainCommit terrain_commit;
    bool earth_fixed_terrain = false;
    DomainRecyclingControlVolumeExchangeLedger exchange;

    PreparedRadiationDomainRecyclingState radiation;
    PreparedBoundaryLayerDomainRecyclingState boundary_layer;
    PreparedMicrophysicsDomainRecyclingState microphysics;
    PreparedStormTrackingDomainRecyclingState storm_tracking;
    PreparedNestedGridDomainRecyclingState nested_grid;
    PreparedEvolvingChaosDomainRecyclingState evolving_chaos;
    PreparedDomainRecyclingTransactionProvenance provenance;

    std::vector<HomogeneousFieldShiftBinding> volume_bindings;
    std::vector<HomogeneousSurfaceShiftBinding> surface_bindings;
    std::vector<HomogeneousScalarCommitBinding> scalar_bindings;
    std::vector<HomogeneousSurfaceRecordShiftBinding> record_bindings;
    std::vector<HomogeneousBooleanCommitBinding> flag_bindings;
    std::vector<HomogeneousRuntimeInvalidationBinding>
        runtime_invalidation_bindings;
    std::vector<HomogeneousRuntimeCommitBinding> runtime_commit_bindings;

    // Exact pre-overwrite owner captures. These are transaction evidence and
    // the source material for durable history publication; they deliberately
    // outlive the destructive commit.
    std::vector<std::vector<float>> primary_trailing;
    std::array<std::vector<float>, 2> radiation_trailing;
    std::array<std::vector<float>, 5> boundary_layer_trailing;
    std::vector<std::vector<float>> microphysics_trailing;
    std::array<std::vector<float>, 3> precipitation_trailing;
    std::vector<TrailingFieldExportBinding> trailing_field_bindings;
    std::vector<TrailingSurfaceExportBinding> trailing_surface_bindings;
    TrailingFieldExportReport trailing_field_report;
    TrailingFieldExportReport trailing_surface_report;

    bool provenance_appended = false;
    bool committed = false;
};

// Source-compatible name for the original CPU-only API. New production code
// should use the backend-neutral type and entry points below.
using PreparedFlatHomogeneousCpuDomainRecycling =
    PreparedFlatHomogeneousDomainRecycling;

std::unique_ptr<PreparedFlatHomogeneousCpuDomainRecycling>
prepare_flat_homogeneous_cpu_domain_recycling(
    SimulationStorageState& storage,
    const StormWindowMovementDecision& decision,
    const std::vector<double>& environmental_tke_profile);

// Backend-neutral production preparation. CPU and capable accelerators share
// every scientific owner/preflight; only the volume commit executor differs.
std::unique_ptr<PreparedFlatHomogeneousDomainRecycling>
prepare_flat_homogeneous_domain_recycling(
    SimulationStorageState& storage,
    const StormWindowMovementDecision& decision,
    const std::vector<double>& environmental_tke_profile);

void append_flat_homogeneous_cpu_domain_recycling_provenance(
    PreparedFlatHomogeneousCpuDomainRecycling& prepared,
    std::string_view transaction_id,
    std::string_view publication_generation,
    std::string_view storage_format,
    std::string_view swath_algorithm,
    double simulation_time_s,
    std::uint64_t transaction_sequence,
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64);

void append_flat_homogeneous_domain_recycling_provenance(
    PreparedFlatHomogeneousDomainRecycling& prepared,
    std::string_view transaction_id,
    std::string_view publication_generation,
    std::string_view storage_format,
    std::string_view swath_algorithm,
    double simulation_time_s,
    std::uint64_t transaction_sequence,
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64);

HomogeneousWindowShiftReport commit_flat_homogeneous_cpu_domain_recycling(
    PreparedFlatHomogeneousCpuDomainRecycling& prepared);

HomogeneousWindowShiftReport commit_flat_homogeneous_domain_recycling(
    PreparedFlatHomogeneousDomainRecycling& prepared);

void verify_flat_homogeneous_cpu_domain_recycling_after_commit(
    const PreparedFlatHomogeneousCpuDomainRecycling& prepared);

void verify_flat_homogeneous_domain_recycling_after_commit(
    const PreparedFlatHomogeneousDomainRecycling& prepared);

const char* flat_homogeneous_cpu_domain_recycling_algorithm_id() noexcept;
const char* flat_homogeneous_domain_recycling_algorithm_id() noexcept;

} // namespace tmv::domain
