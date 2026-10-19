#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tmv::diagnostics
{

enum class StormDetectionStatus
{
    NotRun,
    Weak,
    Detected,
    Ambiguous,
};

const char* storm_detection_status_name(StormDetectionStatus status) noexcept;
const char* storm_feature_detector_algorithm_id() noexcept;
std::size_t storm_feature_component_capacity(std::size_t nx, std::size_t ny);

enum class StormRotationClass
{
    Nonrotating,
    Positive,
    Negative,
    MixedZeroMean,
};

const char* storm_rotation_class_name(StormRotationClass value) noexcept;

// This remains a fixed, allocation-free snapshot, but four entries proved too
// small for a mature 600 m supercell: coherent branches routinely produced
// tens of candidates while the tracked primary remained physically
// resolvable. Sixty-four retains the measured authority set with headroom;
// larger pathological/noisy sets still report explicit overflow.
inline constexpr std::size_t storm_branch_candidate_capacity = 64;

/**
 * Fixed-capacity summary of one coherent storm candidate. Candidate rank is
 * the array index in StormFeatureDetection::branch_candidates: score
 * descending, then connected-component label ascending for exact ties.
 */
struct StormBranchCandidate
{
    std::size_t        component_label          = 0;
    std::size_t        component_cells          = 0;
    std::size_t        core_cells               = 0;
    std::size_t        rotating_core_cells      = 0;
    double             score_fraction           = 0.0;
    double             maximum_updraft_ms       = 0.0;
    double             condensate_mass_kg       = 0.0;
    double             centroid_x_m             = 0.0;
    double             centroid_y_m             = 0.0;
    double             core_centroid_x_m        = 0.0;
    double             core_centroid_y_m        = 0.0;
    double             separation_from_primary_m = 0.0;
    double             envelope_west_m          = 0.0;
    double             envelope_east_m          = 0.0;
    double             envelope_south_m         = 0.0;
    double             envelope_north_m         = 0.0;
    double             distance_west_m          = 0.0;
    double             distance_east_m          = 0.0;
    double             distance_south_m         = 0.0;
    double             distance_north_m         = 0.0;
    double             maximum_absolute_updraft_helicity_m2s2 = 0.0;
    double             signed_rotating_core_mean_updraft_helicity_m2s2 = 0.0;
    StormRotationClass rotation_class           = StormRotationClass::Nonrotating;
    double             previous_partition_overlap_fraction = 0.0;
};

enum class StormContinuityStatus
{
    Uninitialized,
    Tentative,
    Tracking,
    Coasting,
    Ambiguous,
    Lost,
};

enum class StormContinuityReason
{
    None,
    WeakDetection,
    SpatialAmbiguity,
    BranchCandidateOverflow,
    SpeedGate,
    AccelerationGate,
};

const char* storm_continuity_status_name(StormContinuityStatus status) noexcept;
const char* storm_continuity_reason_name(StormContinuityReason reason) noexcept;
const char* storm_continuity_algorithm_id() noexcept;

enum class StormBranchPreference
{
    Continuity,
    BunkersRight,
    BunkersLeft,
};

enum class StormBranchMotionClass
{
    Unclassified,
    BunkersRight,
    BunkersLeft,
    Indeterminate,
};

enum class StormBranchTopology
{
    Unknown,
    Single,
    Multiple,
};

enum class StormBranchTopologyEvent
{
    None,
    SplitCandidate,
    SplitConfirmed,
    MergerCandidate,
    MergerConfirmed,
};

const char* storm_branch_preference_name(StormBranchPreference value) noexcept;
const char* storm_branch_motion_class_name(StormBranchMotionClass value) noexcept;
const char* storm_branch_topology_name(StormBranchTopology value) noexcept;
const char* storm_branch_topology_event_name(StormBranchTopologyEvent value) noexcept;

struct StormFeatureThresholds
{
    double      minimum_updraft_ms             = 5.0;
    double      minimum_condensate_path_kgm2   = 0.1;
    double      minimum_absolute_updraft_helicity_m2s2 = 25.0;
    std::size_t minimum_component_cells        = 4;
    double      minimum_core_area_m2            = 1.0e6;
    double      minimum_primary_score_fraction = 0.8;
};

struct StormColumnSignalView
{
    std::size_t nx           = 0;
    std::size_t ny           = 0;
    double      west_face_m  = 0.0;
    double      south_face_m = 0.0;
    double      dx_m         = 0.0;
    double      dy_m         = 0.0;

    // Packed as i * ny + j over physical scalar columns only.
    const std::vector<double>* maximum_updraft_ms   = nullptr;
    const std::vector<double>* condensate_path_kgm2 = nullptr;
    const std::vector<double>* updraft_helicity_m2s2 = nullptr;
};

struct StormFeatureDetection
{
    StormDetectionStatus status                    = StormDetectionStatus::NotRun;
    std::size_t          envelope_component_count  = 0;
    std::size_t          candidate_component_count = 0;
    std::size_t          selected_component_cells  = 0;
    std::size_t          selected_core_cells       = 0;
    std::size_t          selected_rotating_core_cells = 0;
    std::size_t          reported_branch_candidate_count = 0;
    bool                 branch_candidate_overflow       = false;
    bool                 previous_partition_available    = false;
    std::array<StormBranchCandidate, storm_branch_candidate_capacity> branch_candidates{};

    double confidence         = 0.0;
    double maximum_updraft_ms = 0.0;
    double maximum_absolute_updraft_helicity_m2s2 = 0.0;
    double signed_core_mean_updraft_helicity_m2s2  = 0.0;
    double condensate_mass_kg = 0.0;
    double centroid_x_m       = 0.0;
    double centroid_y_m       = 0.0;
    double core_centroid_x_m  = 0.0;
    double core_centroid_y_m  = 0.0;
    double envelope_west_m    = 0.0;
    double envelope_east_m    = 0.0;
    double envelope_south_m   = 0.0;
    double envelope_north_m   = 0.0;
    double distance_west_m    = 0.0;
    double distance_east_m    = 0.0;
    double distance_south_m   = 0.0;
    double distance_north_m   = 0.0;
};

struct StormFeatureComponentWorkspace
{
    std::size_t cells               = 0;
    std::size_t core_cells          = 0;
    std::size_t min_i               = std::numeric_limits<std::size_t>::max();
    std::size_t max_i               = 0;
    std::size_t min_j               = std::numeric_limits<std::size_t>::max();
    std::size_t max_j               = 0;
    double      score               = 0.0;
    double      condensate_path_sum = 0.0;
    double      condensate_x_sum    = 0.0;
    double      condensate_y_sum    = 0.0;
    double      core_weight_sum     = 0.0;
    double      core_x_sum          = 0.0;
    double      core_y_sum          = 0.0;
    std::size_t rotating_core_cells = 0;
    std::size_t previous_partition_overlap_cells = 0;
    double      rotating_core_weight_sum = 0.0;
    double      rotating_core_signed_uh_sum = 0.0;
    double      maximum_absolute_updraft_helicity_m2s2 = 0.0;
    double      maximum_updraft_ms  = -std::numeric_limits<double>::infinity();
};

inline constexpr std::size_t storm_continuity_history_capacity = 5;

struct StormContinuityThresholds
{
    std::size_t minimum_samples          = 3;
    std::size_t lost_after_misses        = 3;
    double      maximum_speed_ms         = 75.0;
    double      maximum_acceleration_ms2 = 0.25;
    double      position_uncertainty_m   = 1.0;
    std::size_t topology_confirmation_samples = 2;
};

/**
 * Optional environmental discriminator for physically admissible branch
 * associations. Right/left motion is supplied as a pair from the same
 * resolved wind profile (normally Bunkers et al. 2000). Updraft-helicity sign
 * is deliberately absent: it is evidence about rotation, not storm motion.
 */
struct StormBranchAssociationContext
{
    bool                  environmental_motion_available = false;
    StormBranchPreference preference                     = StormBranchPreference::Continuity;
    double                bunkers_right_velocity_x_ms     = 0.0;
    double                bunkers_right_velocity_y_ms     = 0.0;
    double                bunkers_left_velocity_x_ms      = 0.0;
    double                bunkers_left_velocity_y_ms      = 0.0;
    double                environmental_velocity_uncertainty_ms = 5.0;
};

struct StormContinuitySample
{
    double time_s     = 0.0;
    double x_m        = 0.0;
    double y_m        = 0.0;
    double confidence = 0.0;
};

/**
 * Fixed-capacity diagnostic track state. It is ordinary value state so exact
 * checkpoint continuation requires no allocator- or pointer-dependent data.
 */
struct StormContinuityState
{
    StormContinuityStatus                                                status = StormContinuityStatus::Uninitialized;
    StormContinuityReason                                                reason = StormContinuityReason::None;
    std::array<StormContinuitySample, storm_continuity_history_capacity> samples{};
    std::size_t                                                          sample_count   = 0;
    std::size_t                                                          missed_samples = 0;
    std::uint64_t                                                        branch_id      = 0;

    StormBranchTopology      topology         = StormBranchTopology::Unknown;
    StormBranchTopology      pending_topology = StormBranchTopology::Unknown;
    StormBranchTopologyEvent topology_event   = StormBranchTopologyEvent::None;
    std::size_t              pending_topology_samples = 0;
    std::uint64_t            topology_event_id        = 0;

    bool                   selected_candidate_available = false;
    std::size_t            selected_candidate_rank      = 0;
    std::size_t            selected_component_label     = 0;
    StormBranchMotionClass selected_motion_class        = StormBranchMotionClass::Unclassified;
    double                 selected_bunkers_right_mismatch_ms = 0.0;
    double                 selected_bunkers_left_mismatch_ms  = 0.0;

    // Last accepted grid-frame condensate partition. This is association
    // state, not a motion observation: it permits a mature core merger to
    // preserve branch identity without interpreting a topology-driven
    // centroid jump as storm translation.
    bool   association_envelope_available = false;
    double association_envelope_west_m     = 0.0;
    double association_envelope_east_m     = 0.0;
    double association_envelope_south_m    = 0.0;
    double association_envelope_north_m    = 0.0;
    bool   topology_reorganization         = false;
    double reorganization_overlap_fraction = 0.0;

    bool   motion_available          = false;
    double velocity_x_ms             = 0.0;
    double velocity_y_ms             = 0.0;
    double speed_ms                  = 0.0;
    double velocity_uncertainty_ms   = 0.0;
    double continuity_confidence     = 0.0;
    double innovation_distance_m     = 0.0;
    double association_gate_radius_m = 0.0;
};

StormContinuityState update_storm_continuity(StormContinuityState&            state,
                                             const StormFeatureDetection&     detection,
                                             double                           sample_time_s,
                                             const StormContinuityThresholds& thresholds,
                                             const StormBranchAssociationContext& association = {});

bool valid_storm_continuity_state(const StormContinuityState& state) noexcept;

/**
 * One-interval motion of the four faces of a continuity-selected storm
 * envelope. Velocities retain the sign of the detector coordinate: outward
 * motion is negative for west/south and positive for east/north.
 *
 * This is derived transient evidence. The prior accepted envelope and sample
 * time already belong to StormContinuityState, so no second durable history is
 * needed for exact restart continuation.
 */
struct StormEnvelopeFaceMotion
{
    bool   available                    = false;
    double interval_s                   = 0.0;
    double west_coordinate_velocity_ms  = 0.0;
    double east_coordinate_velocity_ms  = 0.0;
    double south_coordinate_velocity_ms = 0.0;
    double north_coordinate_velocity_ms = 0.0;
};

StormEnvelopeFaceMotion derive_storm_envelope_face_motion(
    const StormContinuityState& prior_continuity,
    const StormFeatureDetection& current_detection,
    const StormContinuityState& current_continuity,
    double current_sample_time_s);

/**
 * Deterministic, allocation-stable connected storm-feature detector.
 *
 * Low-threshold condensate first defines 8-connected parent envelopes. Within
 * each parent, 8-connected updraft-core islands remain separate candidates;
 * a deterministic multi-source watershed assigns every parent-envelope cell
 * to its nearest core island. Thus a transient anvil or precipitation bridge
 * cannot collapse distinct convective cores into one nonphysical aggregate
 * motion observation. A candidate's connected updraft core must contain the
 * configured minimum number of columns; assigned cloud area cannot promote a
 * one-column updraft fluctuation into a branch. The primary candidate is selected by integrated
 * `w_max * condensate_path` over its core columns. Confidence is its fraction
 * of the sum of all candidate scores; insufficient dominance is reported as
 * Ambiguous. Signed updraft helicity does not exclude nonrotating convection
 * or assume a cyclonic sign. Candidates are returned in deterministic score
 * order so temporal association never has to reconstruct discarded branches.
 */
class StormFeatureDetector
{
public:
    void reserve(std::size_t nx, std::size_t ny);

    StormFeatureDetection detect(const StormColumnSignalView& signals,
                                 const StormFeatureThresholds& thresholds,
                                 const std::vector<std::uint8_t>* previous_selected_partition = nullptr);

    // Copies the exact watershed partition from the most recent detect() call.
    // The caller owns this restart state; repeated calls reuse its capacity.
    void copy_partition(std::size_t component_label, std::vector<std::uint8_t>& output) const;

private:
    std::vector<int>                            labels_;
    std::vector<int>                            core_labels_;
    std::vector<std::size_t>                    queue_;
    std::vector<StormFeatureComponentWorkspace> components_;
};

} // namespace tmv::diagnostics
