#include "diagnostics/storm_tracker.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv::diagnostics
{
namespace
{

void validate_inputs(const StormColumnSignalView& signals, const StormFeatureThresholds& thresholds)
{
    if (signals.nx == 0 || signals.ny == 0 || signals.maximum_updraft_ms == nullptr ||
        signals.condensate_path_kgm2 == nullptr || signals.updraft_helicity_m2s2 == nullptr)
    {
        throw std::invalid_argument("storm feature detection requires non-empty column signals");
    }
    if (signals.nx > std::numeric_limits<std::size_t>::max() / signals.ny ||
        signals.nx * signals.ny > static_cast<std::size_t>(INT_MAX))
    {
        throw std::invalid_argument("storm feature grid exceeds deterministic label capacity");
    }
    const std::size_t count = signals.nx * signals.ny;
    if (signals.maximum_updraft_ms->size() != count || signals.condensate_path_kgm2->size() != count ||
        signals.updraft_helicity_m2s2->size() != count)
    {
        throw std::invalid_argument("storm feature signal lengths do not match nx*ny");
    }
    if (!std::isfinite(signals.west_face_m) || !std::isfinite(signals.south_face_m) || !std::isfinite(signals.dx_m) ||
        signals.dx_m <= 0.0 || !std::isfinite(signals.dy_m) || signals.dy_m <= 0.0)
    {
        throw std::invalid_argument("storm feature geometry must be finite with positive spacing");
    }
    if (!std::isfinite(thresholds.minimum_updraft_ms) || thresholds.minimum_updraft_ms <= 0.0 ||
        !std::isfinite(thresholds.minimum_condensate_path_kgm2) || thresholds.minimum_condensate_path_kgm2 <= 0.0 ||
        !std::isfinite(thresholds.minimum_absolute_updraft_helicity_m2s2) ||
        thresholds.minimum_absolute_updraft_helicity_m2s2 <= 0.0 ||
        thresholds.minimum_component_cells == 0 || !std::isfinite(thresholds.minimum_core_area_m2) ||
        thresholds.minimum_core_area_m2 <= 0.0 || !std::isfinite(thresholds.minimum_primary_score_fraction) ||
        thresholds.minimum_primary_score_fraction <= 0.0 || thresholds.minimum_primary_score_fraction > 1.0)
    {
        throw std::invalid_argument("storm feature thresholds are invalid");
    }
    for (std::size_t index = 0; index < count; ++index)
    {
        if (!std::isfinite((*signals.maximum_updraft_ms)[index]) ||
            !std::isfinite((*signals.condensate_path_kgm2)[index]) ||
            !std::isfinite((*signals.updraft_helicity_m2s2)[index]) ||
            (*signals.condensate_path_kgm2)[index] < 0.0)
        {
            throw std::invalid_argument("storm feature signals must be finite and condensate path non-negative");
        }
    }
}

void validate_continuity_inputs(const StormFeatureDetection&     detection,
                                double                           sample_time_s,
                                const StormContinuityThresholds& thresholds,
                                const StormBranchAssociationContext& association)
{
    const bool valid_detection_status =
        detection.status == StormDetectionStatus::NotRun || detection.status == StormDetectionStatus::Weak ||
        detection.status == StormDetectionStatus::Detected || detection.status == StormDetectionStatus::Ambiguous;
    const bool valid_preference =
        association.preference == StormBranchPreference::Continuity ||
        association.preference == StormBranchPreference::BunkersRight ||
        association.preference == StormBranchPreference::BunkersLeft;
    if (!valid_detection_status || !valid_preference || !std::isfinite(sample_time_s) || thresholds.minimum_samples < 2 ||
        thresholds.minimum_samples > storm_continuity_history_capacity || thresholds.lost_after_misses == 0 ||
        thresholds.topology_confirmation_samples == 0 ||
        !std::isfinite(thresholds.maximum_speed_ms) || thresholds.maximum_speed_ms <= 0.0 ||
        !std::isfinite(thresholds.maximum_acceleration_ms2) || thresholds.maximum_acceleration_ms2 < 0.0 ||
        !std::isfinite(thresholds.position_uncertainty_m) || thresholds.position_uncertainty_m <= 0.0 ||
        !std::isfinite(association.bunkers_right_velocity_x_ms) ||
        !std::isfinite(association.bunkers_right_velocity_y_ms) ||
        !std::isfinite(association.bunkers_left_velocity_x_ms) ||
        !std::isfinite(association.bunkers_left_velocity_y_ms) ||
        !std::isfinite(association.environmental_velocity_uncertainty_ms) ||
        association.environmental_velocity_uncertainty_ms <= 0.0 ||
        (association.preference != StormBranchPreference::Continuity &&
         !association.environmental_motion_available))
    {
        throw std::invalid_argument("storm continuity detection, thresholds, or time are invalid");
    }
    if (detection.status == StormDetectionStatus::Detected &&
        (!std::isfinite(detection.core_centroid_x_m) || !std::isfinite(detection.core_centroid_y_m) ||
         !std::isfinite(detection.confidence) || detection.confidence <= 0.0 || detection.confidence > 1.0))
    {
        throw std::invalid_argument("detected storm continuity samples require finite core position and confidence");
    }
    if ((detection.status == StormDetectionStatus::Detected ||
         detection.status == StormDetectionStatus::Ambiguous) &&
        detection.reported_branch_candidate_count > storm_branch_candidate_capacity)
    {
        throw std::invalid_argument("storm continuity branch candidate count exceeds fixed capacity");
    }
    for (std::size_t rank = 0; rank < detection.reported_branch_candidate_count; ++rank)
    {
        const auto& candidate = detection.branch_candidates[rank];
        if (!std::isfinite(candidate.core_centroid_x_m) || !std::isfinite(candidate.core_centroid_y_m) ||
            !std::isfinite(candidate.centroid_x_m) || !std::isfinite(candidate.centroid_y_m) ||
            !std::isfinite(candidate.maximum_updraft_ms) || !std::isfinite(candidate.condensate_mass_kg) ||
            candidate.condensate_mass_kg < 0.0 ||
            !std::isfinite(candidate.score_fraction) || candidate.score_fraction <= 0.0 ||
            candidate.score_fraction > 1.0 || !std::isfinite(candidate.envelope_west_m) ||
            !std::isfinite(candidate.envelope_east_m) || !std::isfinite(candidate.envelope_south_m) ||
            !std::isfinite(candidate.envelope_north_m) || !std::isfinite(candidate.distance_west_m) ||
            !std::isfinite(candidate.distance_east_m) || !std::isfinite(candidate.distance_south_m) ||
            !std::isfinite(candidate.distance_north_m) || candidate.envelope_east_m < candidate.envelope_west_m ||
            candidate.envelope_north_m < candidate.envelope_south_m || candidate.distance_west_m < 0.0 ||
            candidate.distance_east_m < 0.0 || candidate.distance_south_m < 0.0 ||
            candidate.distance_north_m < 0.0 ||
            !std::isfinite(candidate.previous_partition_overlap_fraction) ||
            candidate.previous_partition_overlap_fraction < 0.0 ||
            candidate.previous_partition_overlap_fraction > 1.0 ||
            (!detection.previous_partition_available &&
             candidate.previous_partition_overlap_fraction != 0.0))
        {
            throw std::invalid_argument("storm continuity branch candidates require finite positions and scores");
        }
    }
}

void clear_active_track(StormContinuityState& state)
{
    state.sample_count              = 0;
    state.motion_available          = false;
    state.velocity_x_ms             = 0.0;
    state.velocity_y_ms             = 0.0;
    state.speed_ms                  = 0.0;
    state.velocity_uncertainty_ms   = 0.0;
    state.continuity_confidence     = 0.0;
    state.innovation_distance_m     = 0.0;
    state.association_gate_radius_m = 0.0;
    state.topology                  = StormBranchTopology::Unknown;
    state.pending_topology          = StormBranchTopology::Unknown;
    state.topology_event            = StormBranchTopologyEvent::None;
    state.pending_topology_samples  = 0;
    state.selected_candidate_available = false;
    state.selected_candidate_rank      = 0;
    state.selected_component_label     = 0;
    state.selected_motion_class        = StormBranchMotionClass::Unclassified;
    state.selected_bunkers_right_mismatch_ms = 0.0;
    state.selected_bunkers_left_mismatch_ms  = 0.0;
    state.association_envelope_available = false;
    state.association_envelope_west_m    = 0.0;
    state.association_envelope_east_m    = 0.0;
    state.association_envelope_south_m   = 0.0;
    state.association_envelope_north_m   = 0.0;
    state.topology_reorganization       = false;
    state.reorganization_overlap_fraction = 0.0;
}

void increment_missed_samples(StormContinuityState& state) noexcept
{
    if (state.missed_samples < std::numeric_limits<std::size_t>::max())
    {
        ++state.missed_samples;
    }
}

void append_sample(StormContinuityState& state, const StormContinuitySample& sample)
{
    if (state.sample_count < storm_continuity_history_capacity)
    {
        state.samples[state.sample_count++] = sample;
        return;
    }
    std::move(state.samples.begin() + 1, state.samples.end(), state.samples.begin());
    state.samples.back() = sample;
}

void fit_motion(StormContinuityState& state, const StormContinuityThresholds& thresholds)
{
    if (state.sample_count < 2)
    {
        state.velocity_x_ms           = 0.0;
        state.velocity_y_ms           = 0.0;
        state.speed_ms                = 0.0;
        state.velocity_uncertainty_ms = 0.0;
        state.motion_available        = false;
        return;
    }

    double mean_t             = 0.0;
    double mean_x             = 0.0;
    double mean_y             = 0.0;
    double minimum_confidence = 1.0;
    for (std::size_t index = 0; index < state.sample_count; ++index)
    {
        mean_t += state.samples[index].time_s;
        mean_x += state.samples[index].x_m;
        mean_y += state.samples[index].y_m;
        minimum_confidence = std::min(minimum_confidence, state.samples[index].confidence);
    }
    const double inverse_count = 1.0 / static_cast<double>(state.sample_count);
    mean_t *= inverse_count;
    mean_x *= inverse_count;
    mean_y *= inverse_count;

    double time_variance = 0.0;
    double covariance_x  = 0.0;
    double covariance_y  = 0.0;
    for (std::size_t index = 0; index < state.sample_count; ++index)
    {
        const double centered_t = state.samples[index].time_s - mean_t;
        time_variance += centered_t * centered_t;
        covariance_x += centered_t * (state.samples[index].x_m - mean_x);
        covariance_y += centered_t * (state.samples[index].y_m - mean_y);
    }
    if (!(time_variance > 0.0) || !std::isfinite(time_variance))
    {
        throw std::invalid_argument("storm continuity samples require strictly increasing time");
    }

    state.velocity_x_ms = covariance_x / time_variance;
    state.velocity_y_ms = covariance_y / time_variance;
    state.speed_ms      = std::hypot(state.velocity_x_ms, state.velocity_y_ms);

    double residual_x2 = 0.0;
    double residual_y2 = 0.0;
    for (std::size_t index = 0; index < state.sample_count; ++index)
    {
        const double dt         = state.samples[index].time_s - mean_t;
        const double residual_x = state.samples[index].x_m - (mean_x + state.velocity_x_ms * dt);
        const double residual_y = state.samples[index].y_m - (mean_y + state.velocity_y_ms * dt);
        residual_x2 += residual_x * residual_x;
        residual_y2 += residual_y * residual_y;
    }
    const double residual_dof         = static_cast<double>(std::max<std::size_t>(1, state.sample_count - 2));
    const double observation_variance = thresholds.position_uncertainty_m * thresholds.position_uncertainty_m;
    const double variance_x           = std::max(observation_variance, residual_x2 / residual_dof);
    const double variance_y           = std::max(observation_variance, residual_y2 / residual_dof);
    state.velocity_uncertainty_ms =
        std::hypot(std::sqrt(variance_x / time_variance), std::sqrt(variance_y / time_variance));

    const double association_quality = state.association_gate_radius_m > 0.0
                                           ? std::exp(-state.innovation_distance_m / state.association_gate_radius_m)
                                           : 1.0;
    state.continuity_confidence      = minimum_confidence * association_quality;
    state.motion_available           = state.sample_count >= thresholds.minimum_samples;
}

bool valid_topology(StormBranchTopology value) noexcept
{
    return value == StormBranchTopology::Unknown || value == StormBranchTopology::Single ||
           value == StormBranchTopology::Multiple;
}

bool valid_topology_event(StormBranchTopologyEvent value) noexcept
{
    return value == StormBranchTopologyEvent::None || value == StormBranchTopologyEvent::SplitCandidate ||
           value == StormBranchTopologyEvent::SplitConfirmed || value == StormBranchTopologyEvent::MergerCandidate ||
           value == StormBranchTopologyEvent::MergerConfirmed;
}

bool valid_motion_class(StormBranchMotionClass value) noexcept
{
    return value == StormBranchMotionClass::Unclassified || value == StormBranchMotionClass::BunkersRight ||
           value == StormBranchMotionClass::BunkersLeft || value == StormBranchMotionClass::Indeterminate;
}

void reset_pending_topology(StormContinuityState& state) noexcept
{
    state.pending_topology         = StormBranchTopology::Unknown;
    state.pending_topology_samples = 0;
    state.topology_event           = StormBranchTopologyEvent::None;
}

void update_topology(StormContinuityState& state,
                     StormBranchTopology observed,
                     const StormContinuityThresholds& thresholds)
{
    state.topology_event = StormBranchTopologyEvent::None;
    if (observed == StormBranchTopology::Unknown)
    {
        reset_pending_topology(state);
        return;
    }
    if (observed == state.topology)
    {
        reset_pending_topology(state);
        return;
    }
    if (state.pending_topology != observed)
    {
        state.pending_topology         = observed;
        state.pending_topology_samples = 1;
    }
    else if (state.pending_topology_samples < std::numeric_limits<std::size_t>::max())
    {
        ++state.pending_topology_samples;
    }

    const bool split  = state.topology == StormBranchTopology::Single && observed == StormBranchTopology::Multiple;
    const bool merger = state.topology == StormBranchTopology::Multiple && observed == StormBranchTopology::Single;
    if (split)
        state.topology_event = StormBranchTopologyEvent::SplitCandidate;
    else if (merger)
        state.topology_event = StormBranchTopologyEvent::MergerCandidate;

    if (state.pending_topology_samples < thresholds.topology_confirmation_samples)
        return;

    state.topology = observed;
    reset_pending_topology(state);
    if (split)
        state.topology_event = StormBranchTopologyEvent::SplitConfirmed;
    else if (merger)
        state.topology_event = StormBranchTopologyEvent::MergerConfirmed;
    if (split || merger)
    {
        if (state.topology_event_id == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("storm topology event identifier exhausted");
        ++state.topology_event_id;
    }
}

struct AssociationCandidate
{
    StormContinuitySample sample;
    std::size_t rank            = 0;
    std::size_t component_label = 0;
    double innovation_m         = 0.0;
    double right_mismatch_ms    = 0.0;
    double left_mismatch_ms     = 0.0;
    double preferred_mismatch_ms = 0.0;
    StormBranchMotionClass motion_class = StormBranchMotionClass::Unclassified;
    StormContinuityReason rejection = StormContinuityReason::None;
    double envelope_west_m  = 0.0;
    double envelope_east_m  = 0.0;
    double envelope_south_m = 0.0;
    double envelope_north_m = 0.0;
    double previous_envelope_coverage = 0.0;
    double previous_partition_coverage = 0.0;
};

bool positive_envelope(const AssociationCandidate& candidate) noexcept
{
    return candidate.envelope_east_m > candidate.envelope_west_m &&
           candidate.envelope_north_m > candidate.envelope_south_m;
}

void remember_association_envelope(StormContinuityState& state,
                                   const AssociationCandidate& candidate) noexcept
{
    state.association_envelope_available = positive_envelope(candidate);
    if (state.association_envelope_available)
    {
        state.association_envelope_west_m  = candidate.envelope_west_m;
        state.association_envelope_east_m  = candidate.envelope_east_m;
        state.association_envelope_south_m = candidate.envelope_south_m;
        state.association_envelope_north_m = candidate.envelope_north_m;
    }
    else
    {
        state.association_envelope_west_m = state.association_envelope_east_m = 0.0;
        state.association_envelope_south_m = state.association_envelope_north_m = 0.0;
    }
}

double previous_envelope_coverage(const StormContinuityState& state,
                                  const AssociationCandidate& candidate) noexcept
{
    if (!state.association_envelope_available || !positive_envelope(candidate))
        return 0.0;
    const double previous_width = state.association_envelope_east_m - state.association_envelope_west_m;
    const double previous_height = state.association_envelope_north_m - state.association_envelope_south_m;
    const double previous_area = previous_width * previous_height;
    if (!(previous_area > 0.0))
        return 0.0;
    const double overlap_width =
        std::max(0.0, std::min(state.association_envelope_east_m, candidate.envelope_east_m) -
                          std::max(state.association_envelope_west_m, candidate.envelope_west_m));
    const double overlap_height =
        std::max(0.0, std::min(state.association_envelope_north_m, candidate.envelope_north_m) -
                          std::max(state.association_envelope_south_m, candidate.envelope_south_m));
    return overlap_width * overlap_height / previous_area;
}

StormBranchMotionClass classify_motion(double right_mismatch_ms,
                                       double left_mismatch_ms,
                                       double uncertainty_ms) noexcept
{
    if (std::abs(right_mismatch_ms - left_mismatch_ms) <= uncertainty_ms)
        return StormBranchMotionClass::Indeterminate;
    return right_mismatch_ms < left_mismatch_ms ? StormBranchMotionClass::BunkersRight
                                                : StormBranchMotionClass::BunkersLeft;
}

} // namespace

const char* storm_detection_status_name(StormDetectionStatus status) noexcept
{
    switch (status)
    {
    case StormDetectionStatus::NotRun:
        return "not_run";
    case StormDetectionStatus::Weak:
        return "weak";
    case StormDetectionStatus::Detected:
        return "detected";
    case StormDetectionStatus::Ambiguous:
        return "ambiguous";
    }
    return "not_run";
}

const char* storm_feature_detector_algorithm_id() noexcept
{
    return "updraft_core_watershed_condensate_rotation_branches_v5";
}

const char* storm_rotation_class_name(StormRotationClass value) noexcept
{
    switch (value)
    {
    case StormRotationClass::Nonrotating:
        return "nonrotating";
    case StormRotationClass::Positive:
        return "positive";
    case StormRotationClass::Negative:
        return "negative";
    case StormRotationClass::MixedZeroMean:
        return "mixed_zero_mean";
    }
    return "nonrotating";
}

std::size_t storm_feature_component_capacity(std::size_t nx, std::size_t ny)
{
    const std::size_t separated_x = nx / 2 + nx % 2;
    const std::size_t separated_y = ny / 2 + ny % 2;
    if (separated_x != 0 && separated_y > std::numeric_limits<std::size_t>::max() / separated_x)
    {
        throw std::overflow_error("storm feature component capacity exceeds size_t");
    }
    return separated_x * separated_y;
}

const char* storm_continuity_status_name(StormContinuityStatus status) noexcept
{
    switch (status)
    {
    case StormContinuityStatus::Uninitialized:
        return "uninitialized";
    case StormContinuityStatus::Tentative:
        return "tentative";
    case StormContinuityStatus::Tracking:
        return "tracking";
    case StormContinuityStatus::Coasting:
        return "coasting";
    case StormContinuityStatus::Ambiguous:
        return "ambiguous";
    case StormContinuityStatus::Lost:
        return "lost";
    }
    return "uninitialized";
}

const char* storm_continuity_reason_name(StormContinuityReason reason) noexcept
{
    switch (reason)
    {
    case StormContinuityReason::None:
        return "none";
    case StormContinuityReason::WeakDetection:
        return "weak_detection";
    case StormContinuityReason::SpatialAmbiguity:
        return "spatial_ambiguity";
    case StormContinuityReason::BranchCandidateOverflow:
        return "branch_candidate_overflow";
    case StormContinuityReason::SpeedGate:
        return "speed_gate";
    case StormContinuityReason::AccelerationGate:
        return "acceleration_gate";
    }
    return "none";
}

const char* storm_continuity_algorithm_id() noexcept
{
    return "fixed_window_temporal_branch_association_v5";
}

const char* storm_branch_preference_name(StormBranchPreference value) noexcept
{
    switch (value)
    {
    case StormBranchPreference::Continuity:
        return "continuity";
    case StormBranchPreference::BunkersRight:
        return "bunkers_right";
    case StormBranchPreference::BunkersLeft:
        return "bunkers_left";
    }
    return "continuity";
}

const char* storm_branch_motion_class_name(StormBranchMotionClass value) noexcept
{
    switch (value)
    {
    case StormBranchMotionClass::Unclassified:
        return "unclassified";
    case StormBranchMotionClass::BunkersRight:
        return "bunkers_right";
    case StormBranchMotionClass::BunkersLeft:
        return "bunkers_left";
    case StormBranchMotionClass::Indeterminate:
        return "indeterminate";
    }
    return "unclassified";
}

const char* storm_branch_topology_name(StormBranchTopology value) noexcept
{
    switch (value)
    {
    case StormBranchTopology::Unknown:
        return "unknown";
    case StormBranchTopology::Single:
        return "single";
    case StormBranchTopology::Multiple:
        return "multiple";
    }
    return "unknown";
}

const char* storm_branch_topology_event_name(StormBranchTopologyEvent value) noexcept
{
    switch (value)
    {
    case StormBranchTopologyEvent::None:
        return "none";
    case StormBranchTopologyEvent::SplitCandidate:
        return "split_candidate";
    case StormBranchTopologyEvent::SplitConfirmed:
        return "split_confirmed";
    case StormBranchTopologyEvent::MergerCandidate:
        return "merger_candidate";
    case StormBranchTopologyEvent::MergerConfirmed:
        return "merger_confirmed";
    }
    return "none";
}

void StormFeatureDetector::reserve(std::size_t nx, std::size_t ny)
{
    if (nx != 0 && ny > std::numeric_limits<std::size_t>::max() / nx)
    {
        throw std::overflow_error("storm feature workspace exceeds size_t");
    }
    const std::size_t count = nx * ny;
    labels_.resize(count);
    core_labels_.resize(count);
    queue_.resize(count);
    components_.reserve(storm_feature_component_capacity(nx, ny));
}

StormFeatureDetection StormFeatureDetector::detect(
    const StormColumnSignalView& signals,
    const StormFeatureThresholds& thresholds,
    const std::vector<std::uint8_t>* previous_selected_partition)
{
    validate_inputs(signals, thresholds);
    const std::size_t count = signals.nx * signals.ny;
    std::size_t previous_partition_cells = 0;
    if (previous_selected_partition != nullptr && !previous_selected_partition->empty())
    {
        if (previous_selected_partition->size() != count)
            throw std::invalid_argument("previous storm partition size does not match nx*ny");
        for (std::uint8_t value : *previous_selected_partition)
        {
            if (value > 1U)
                throw std::invalid_argument("previous storm partition must be a binary mask");
            previous_partition_cells += value;
        }
        if (previous_partition_cells == 0)
            throw std::invalid_argument("available previous storm partition is empty");
    }
    if (labels_.size() != count || core_labels_.size() != count || queue_.size() != count)
        reserve(signals.nx, signals.ny);
    std::fill(labels_.begin(), labels_.end(), -1);
    std::fill(core_labels_.begin(), core_labels_.end(), -1);

    const auto& updraft    = *signals.maximum_updraft_ms;
    const auto& condensate = *signals.condensate_path_kgm2;
    const auto& helicity   = *signals.updraft_helicity_m2s2;
    components_.clear();

    // First identify the low-threshold condensate parents. These labels are
    // deliberately not storm branches: a thin shared anvil may connect
    // otherwise distinct convective cores over tens of kilometres.
    std::size_t envelope_component_count = 0;
    for (std::size_t seed = 0; seed < count; ++seed)
    {
        if (labels_[seed] >= 0 || condensate[seed] < thresholds.minimum_condensate_path_kgm2)
        {
            continue;
        }

        const int component_id = static_cast<int>(envelope_component_count++);
        std::size_t                     head      = 0;
        std::size_t                     tail      = 0;
        queue_[tail++]                            = seed;
        labels_[seed]                             = component_id;

        while (head < tail)
        {
            const std::size_t index = queue_[head++];
            const std::size_t i     = index / signals.ny;
            const std::size_t j     = index % signals.ny;
            const std::size_t i0 = i == 0 ? 0 : i - 1;
            const std::size_t i1 = std::min(signals.nx - 1, i + 1);
            const std::size_t j0 = j == 0 ? 0 : j - 1;
            const std::size_t j1 = std::min(signals.ny - 1, j + 1);
            for (std::size_t ni = i0; ni <= i1; ++ni)
                for (std::size_t nj = j0; nj <= j1; ++nj)
                {
                    const std::size_t neighbor = ni * signals.ny + nj;
                    if (labels_[neighbor] < 0 && condensate[neighbor] >= thresholds.minimum_condensate_path_kgm2)
                    {
                        labels_[neighbor] = component_id;
                        queue_[tail++]    = neighbor;
                    }
                }
        }
    }

    // Label each connected updraft-core island independently, even when two
    // islands live beneath one connected condensate parent. Core statistics
    // are accumulated here; assigned-envelope statistics follow the
    // watershed below.
    for (std::size_t seed = 0; seed < count; ++seed)
    {
        if (core_labels_[seed] >= 0 || labels_[seed] < 0 ||
            updraft[seed] < thresholds.minimum_updraft_ms)
        {
            continue;
        }

        const int core_id = static_cast<int>(components_.size());
        components_.emplace_back();
        StormFeatureComponentWorkspace& component = components_.back();
        const int parent_envelope = labels_[seed];
        std::size_t head = 0;
        std::size_t tail = 0;
        queue_[tail++] = seed;
        core_labels_[seed] = core_id;

        while (head < tail)
        {
            const std::size_t index = queue_[head++];
            const std::size_t i = index / signals.ny;
            const std::size_t j = index % signals.ny;
            const double x_m = signals.west_face_m + (static_cast<double>(i) + 0.5) * signals.dx_m;
            const double y_m = signals.south_face_m + (static_cast<double>(j) + 0.5) * signals.dy_m;
            const double core_weight = updraft[index] * condensate[index];

            ++component.core_cells;
            component.score += core_weight;
            component.core_weight_sum += core_weight;
            component.core_x_sum += core_weight * x_m;
            component.core_y_sum += core_weight * y_m;
            if (std::abs(helicity[index]) >= thresholds.minimum_absolute_updraft_helicity_m2s2)
            {
                ++component.rotating_core_cells;
                component.rotating_core_weight_sum += core_weight;
                component.rotating_core_signed_uh_sum += core_weight * helicity[index];
            }

            const std::size_t i0 = i == 0 ? 0 : i - 1;
            const std::size_t i1 = std::min(signals.nx - 1, i + 1);
            const std::size_t j0 = j == 0 ? 0 : j - 1;
            const std::size_t j1 = std::min(signals.ny - 1, j + 1);
            for (std::size_t ni = i0; ni <= i1; ++ni)
                for (std::size_t nj = j0; nj <= j1; ++nj)
                {
                    const std::size_t neighbor = ni * signals.ny + nj;
                    if (core_labels_[neighbor] < 0 && labels_[neighbor] == parent_envelope &&
                        updraft[neighbor] >= thresholds.minimum_updraft_ms)
                    {
                        core_labels_[neighbor] = core_id;
                        queue_[tail++] = neighbor;
                    }
                }
        }
    }

    // Deterministic 8-neighbour multi-source watershed. The initial queue is
    // packed-index order, so equal-distance ties have one stable disposition.
    // Each condensate cell is enqueued at most once; steady-state detection
    // performs no allocation after reserve().
    std::size_t head = 0;
    std::size_t tail = 0;
    for (std::size_t index = 0; index < count; ++index)
        if (core_labels_[index] >= 0)
            queue_[tail++] = index;
    while (head < tail)
    {
        const std::size_t index = queue_[head++];
        const std::size_t i = index / signals.ny;
        const std::size_t j = index % signals.ny;
        const std::size_t i0 = i == 0 ? 0 : i - 1;
        const std::size_t i1 = std::min(signals.nx - 1, i + 1);
        const std::size_t j0 = j == 0 ? 0 : j - 1;
        const std::size_t j1 = std::min(signals.ny - 1, j + 1);
        for (std::size_t ni = i0; ni <= i1; ++ni)
            for (std::size_t nj = j0; nj <= j1; ++nj)
            {
                const std::size_t neighbor = ni * signals.ny + nj;
                if (labels_[neighbor] == labels_[index] && labels_[neighbor] >= 0 &&
                    core_labels_[neighbor] < 0)
                {
                    core_labels_[neighbor] = core_labels_[index];
                    queue_[tail++] = neighbor;
                }
            }
    }

    for (std::size_t index = 0; index < count; ++index)
    {
        if (core_labels_[index] < 0)
            continue;
        auto& component = components_[static_cast<std::size_t>(core_labels_[index])];
        const std::size_t i = index / signals.ny;
        const std::size_t j = index % signals.ny;
        const double x_m = signals.west_face_m + (static_cast<double>(i) + 0.5) * signals.dx_m;
        const double y_m = signals.south_face_m + (static_cast<double>(j) + 0.5) * signals.dy_m;
        const double path = condensate[index];
        ++component.cells;
        component.min_i = std::min(component.min_i, i);
        component.max_i = std::max(component.max_i, i);
        component.min_j = std::min(component.min_j, j);
        component.max_j = std::max(component.max_j, j);
        component.condensate_path_sum += path;
        component.condensate_x_sum += path * x_m;
        component.condensate_y_sum += path * y_m;
        component.maximum_updraft_ms = std::max(component.maximum_updraft_ms, updraft[index]);
        component.maximum_absolute_updraft_helicity_m2s2 =
            std::max(component.maximum_absolute_updraft_helicity_m2s2, std::abs(helicity[index]));
        if (previous_partition_cells > 0 && (*previous_selected_partition)[index] != 0U)
            ++component.previous_partition_overlap_cells;
    }

    StormFeatureDetection result;
    result.previous_partition_available = previous_partition_cells > 0;
    result.envelope_component_count = envelope_component_count;
    std::size_t primary             = components_.size();
    double      score_sum           = 0.0;
    for (std::size_t index = 0; index < components_.size(); ++index)
    {
        const StormFeatureComponentWorkspace& component = components_[index];
        if (component.core_cells < thresholds.minimum_component_cells ||
            static_cast<double>(component.core_cells) * signals.dx_m * signals.dy_m <
                thresholds.minimum_core_area_m2)
        {
            continue;
        }
        ++result.candidate_component_count;
        score_sum += component.score;
        if (primary == components_.size() || component.score > components_[primary].score)
        {
            primary = index;
        }
    }

    if (primary == components_.size())
    {
        result.status = StormDetectionStatus::Weak;
        return result;
    }

    const StormFeatureComponentWorkspace& selected = components_[primary];
    result.confidence                              = score_sum > 0.0 ? selected.score / score_sum : 0.0;
    result.status = result.confidence >= thresholds.minimum_primary_score_fraction ? StormDetectionStatus::Detected
                                                                                   : StormDetectionStatus::Ambiguous;
    result.selected_component_cells = selected.cells;
    result.selected_core_cells      = selected.core_cells;
    result.selected_rotating_core_cells = selected.rotating_core_cells;
    result.maximum_updraft_ms       = selected.maximum_updraft_ms;
    result.maximum_absolute_updraft_helicity_m2s2 =
        selected.maximum_absolute_updraft_helicity_m2s2;
    result.signed_core_mean_updraft_helicity_m2s2 =
        selected.rotating_core_weight_sum > 0.0
            ? selected.rotating_core_signed_uh_sum / selected.rotating_core_weight_sum
            : 0.0;
    result.condensate_mass_kg       = selected.condensate_path_sum * signals.dx_m * signals.dy_m;
    result.centroid_x_m             = selected.condensate_x_sum / selected.condensate_path_sum;
    result.centroid_y_m             = selected.condensate_y_sum / selected.condensate_path_sum;
    result.core_centroid_x_m        = selected.core_x_sum / selected.core_weight_sum;
    result.core_centroid_y_m        = selected.core_y_sum / selected.core_weight_sum;
    result.envelope_west_m          = signals.west_face_m + static_cast<double>(selected.min_i) * signals.dx_m;
    result.envelope_east_m          = signals.west_face_m + static_cast<double>(selected.max_i + 1) * signals.dx_m;
    result.envelope_south_m         = signals.south_face_m + static_cast<double>(selected.min_j) * signals.dy_m;
    result.envelope_north_m         = signals.south_face_m + static_cast<double>(selected.max_j + 1) * signals.dy_m;
    const double domain_east_m      = signals.west_face_m + static_cast<double>(signals.nx) * signals.dx_m;
    const double domain_north_m     = signals.south_face_m + static_cast<double>(signals.ny) * signals.dy_m;
    result.distance_west_m          = result.envelope_west_m - signals.west_face_m;
    result.distance_east_m          = domain_east_m - result.envelope_east_m;
    result.distance_south_m         = result.envelope_south_m - signals.south_face_m;
    result.distance_north_m         = domain_north_m - result.envelope_north_m;

    // Preserve the strongest coherent alternatives instead of collapsing an
    // ambiguous/splitting snapshot to one primary component. This is a fixed
    // insertion set rather than a per-sample allocation or full sort.
    std::array<std::size_t, storm_branch_candidate_capacity> ranked{};
    ranked.fill(components_.size());
    for (std::size_t component_index = 0; component_index < components_.size(); ++component_index)
    {
        const auto& component = components_[component_index];
        if (component.core_cells < thresholds.minimum_component_cells ||
            static_cast<double>(component.core_cells) * signals.dx_m * signals.dy_m <
                thresholds.minimum_core_area_m2)
            continue;

        std::size_t insertion = storm_branch_candidate_capacity;
        for (std::size_t rank = 0; rank < storm_branch_candidate_capacity; ++rank)
        {
            const std::size_t incumbent = ranked[rank];
            if (incumbent == components_.size() || component.score > components_[incumbent].score ||
                (component.score == components_[incumbent].score && component_index < incumbent))
            {
                insertion = rank;
                break;
            }
        }
        if (insertion == storm_branch_candidate_capacity)
            continue;
        for (std::size_t rank = storm_branch_candidate_capacity - 1; rank > insertion; --rank)
            ranked[rank] = ranked[rank - 1];
        ranked[insertion] = component_index;
    }

    result.reported_branch_candidate_count =
        std::min(result.candidate_component_count, storm_branch_candidate_capacity);
    result.branch_candidate_overflow =
        result.candidate_component_count > storm_branch_candidate_capacity;
    for (std::size_t rank = 0; rank < result.reported_branch_candidate_count; ++rank)
    {
        const std::size_t component_index = ranked[rank];
        const auto&       component       = components_[component_index];
        auto&             candidate       = result.branch_candidates[rank];
        candidate.component_label         = component_index;
        candidate.component_cells         = component.cells;
        candidate.core_cells              = component.core_cells;
        candidate.rotating_core_cells     = component.rotating_core_cells;
        candidate.score_fraction          = score_sum > 0.0 ? component.score / score_sum : 0.0;
        candidate.maximum_updraft_ms       = component.maximum_updraft_ms;
        candidate.condensate_mass_kg       = component.condensate_path_sum * signals.dx_m * signals.dy_m;
        candidate.centroid_x_m             = component.condensate_x_sum / component.condensate_path_sum;
        candidate.centroid_y_m             = component.condensate_y_sum / component.condensate_path_sum;
        candidate.core_centroid_x_m       = component.core_x_sum / component.core_weight_sum;
        candidate.core_centroid_y_m       = component.core_y_sum / component.core_weight_sum;
        candidate.separation_from_primary_m =
            std::hypot(candidate.core_centroid_x_m - result.core_centroid_x_m,
                       candidate.core_centroid_y_m - result.core_centroid_y_m);
        candidate.envelope_west_m =
            signals.west_face_m + static_cast<double>(component.min_i) * signals.dx_m;
        candidate.envelope_east_m =
            signals.west_face_m + static_cast<double>(component.max_i + 1) * signals.dx_m;
        candidate.envelope_south_m =
            signals.south_face_m + static_cast<double>(component.min_j) * signals.dy_m;
        candidate.envelope_north_m =
            signals.south_face_m + static_cast<double>(component.max_j + 1) * signals.dy_m;
        candidate.distance_west_m  = candidate.envelope_west_m - signals.west_face_m;
        candidate.distance_east_m  = domain_east_m - candidate.envelope_east_m;
        candidate.distance_south_m = candidate.envelope_south_m - signals.south_face_m;
        candidate.distance_north_m = domain_north_m - candidate.envelope_north_m;
        candidate.maximum_absolute_updraft_helicity_m2s2 =
            component.maximum_absolute_updraft_helicity_m2s2;
        if (component.rotating_core_weight_sum > 0.0)
        {
            candidate.signed_rotating_core_mean_updraft_helicity_m2s2 =
                component.rotating_core_signed_uh_sum / component.rotating_core_weight_sum;
            const double signed_mean =
                candidate.signed_rotating_core_mean_updraft_helicity_m2s2;
        candidate.rotation_class =
                signed_mean > 0.0
                    ? StormRotationClass::Positive
                    : (signed_mean < 0.0 ? StormRotationClass::Negative
                                         : StormRotationClass::MixedZeroMean);
        }
        candidate.previous_partition_overlap_fraction =
            previous_partition_cells > 0
                ? static_cast<double>(component.previous_partition_overlap_cells) /
                      static_cast<double>(previous_partition_cells)
                : 0.0;
    }
    return result;
}

void StormFeatureDetector::copy_partition(std::size_t component_label,
                                          std::vector<std::uint8_t>& output) const
{
    if (component_label >= components_.size() || core_labels_.empty())
        throw std::invalid_argument("selected storm component label is unavailable");
    output.resize(core_labels_.size());
    const int label = static_cast<int>(component_label);
    for (std::size_t index = 0; index < core_labels_.size(); ++index)
        output[index] = core_labels_[index] == label ? 1U : 0U;
}

StormContinuityState update_storm_continuity(StormContinuityState&            state,
                                             const StormFeatureDetection&     detection,
                                             double                           sample_time_s,
                                             const StormContinuityThresholds& thresholds,
                                             const StormBranchAssociationContext& association)
{
    validate_continuity_inputs(detection, sample_time_s, thresholds, association);
    if (!valid_storm_continuity_state(state))
    {
        throw std::invalid_argument("incoming storm continuity state is invalid");
    }

    state.topology_event = StormBranchTopologyEvent::None;
    const bool has_reported_candidates =
        (detection.status == StormDetectionStatus::Detected || detection.status == StormDetectionStatus::Ambiguous) &&
        detection.reported_branch_candidate_count > 0;
    const bool has_legacy_primary = detection.status == StormDetectionStatus::Detected &&
                                    detection.reported_branch_candidate_count == 0;

    state.selected_candidate_available = false;
    state.selected_candidate_rank      = 0;
    state.selected_component_label     = 0;
    state.selected_motion_class        = StormBranchMotionClass::Unclassified;
    state.selected_bunkers_right_mismatch_ms = 0.0;
    state.selected_bunkers_left_mismatch_ms  = 0.0;
    state.topology_reorganization            = false;
    state.reorganization_overlap_fraction    = 0.0;

    const auto miss = [&](StormContinuityReason reason) -> StormContinuityState&
    {
        increment_missed_samples(state);
        state.motion_available      = false;
        state.continuity_confidence = 0.0;
        state.reason                = reason;
        if (state.sample_count == 0)
        {
            state.status =
                state.status == StormContinuityStatus::Lost
                    ? StormContinuityStatus::Lost
                    : (reason == StormContinuityReason::WeakDetection ? StormContinuityStatus::Uninitialized
                                                                      : StormContinuityStatus::Ambiguous);
        }
        else if (state.missed_samples >= thresholds.lost_after_misses)
        {
            state.status = StormContinuityStatus::Lost;
            clear_active_track(state);
        }
        else
        {
            state.status =
                reason == StormContinuityReason::WeakDetection ? StormContinuityStatus::Coasting
                                                               : StormContinuityStatus::Ambiguous;
        }
        return state;
    };

    if (!has_reported_candidates && !has_legacy_primary)
    {
        reset_pending_topology(state);
        return miss(StormContinuityReason::WeakDetection);
    }
    // Overflow is unresolved only when the detector itself could not identify
    // a dominant primary.  A Detected result has already met the configured
    // primary-score dominance threshold, and rank zero is necessarily that
    // primary because the fixed snapshot is score ordered.  Rejecting that
    // sample merely because weak, lower-ranked cells exceeded the reporting
    // capacity makes a mature storm lose continuity as convection develops.
    // In the dominant-overflow case, consume only the resolved primary; do not
    // let an incomplete alternative set participate in branch association.
    const bool dominant_primary_overflow =
        detection.branch_candidate_overflow && detection.status == StormDetectionStatus::Detected;
    if (detection.branch_candidate_overflow && !dominant_primary_overflow)
    {
        reset_pending_topology(state);
        return miss(StormContinuityReason::BranchCandidateOverflow);
    }
    if (state.sample_count == 0 && detection.status == StormDetectionStatus::Ambiguous)
    {
        reset_pending_topology(state);
        return miss(StormContinuityReason::SpatialAmbiguity);
    }

    std::array<AssociationCandidate, storm_branch_candidate_capacity> candidates{};
    const std::size_t candidate_count =
        has_reported_candidates
            ? (dominant_primary_overflow ? 1 : detection.reported_branch_candidate_count)
            : 1;
    for (std::size_t rank = 0; rank < candidate_count; ++rank)
    {
        auto& output = candidates[rank];
        output.rank = rank;
        if (has_reported_candidates)
        {
            const auto& input = detection.branch_candidates[rank];
            output.component_label = input.component_label;
            output.sample = {
                sample_time_s,
                input.core_centroid_x_m,
                input.core_centroid_y_m,
                input.score_fraction,
            };
            output.envelope_west_m  = input.envelope_west_m;
            output.envelope_east_m  = input.envelope_east_m;
            output.envelope_south_m = input.envelope_south_m;
            output.envelope_north_m = input.envelope_north_m;
            output.previous_partition_coverage = input.previous_partition_overlap_fraction;
        }
        else
        {
            output.sample = {
                sample_time_s,
                detection.core_centroid_x_m,
                detection.core_centroid_y_m,
                detection.confidence,
            };
        }
    }

    if (state.sample_count == 0)
    {
        update_topology(state, StormBranchTopology::Single, thresholds);
        if (state.branch_id == std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error("storm continuity branch identifier exhausted");
        }
        ++state.branch_id;
        state.missed_samples = 0;
        state.reason         = StormContinuityReason::None;
        state.selected_candidate_available = true;
        state.selected_candidate_rank      = candidates[0].rank;
        state.selected_component_label     = candidates[0].component_label;
        append_sample(state, candidates[0].sample);
        remember_association_envelope(state, candidates[0]);
        state.status = StormContinuityStatus::Tentative;
        fit_motion(state, thresholds);
        return state;
    }

    const StormContinuitySample& previous = state.samples[state.sample_count - 1];
    const double                 elapsed_s = sample_time_s - previous.time_s;
    if (!(elapsed_s > 0.0) || !std::isfinite(elapsed_s))
    {
        throw std::invalid_argument("storm continuity sample times must increase strictly");
    }
    state.association_gate_radius_m = thresholds.maximum_speed_ms * elapsed_s;
    const double predicted_x =
        state.sample_count >= 2 ? previous.x_m + state.velocity_x_ms * elapsed_s : previous.x_m;
    const double predicted_y =
        state.sample_count >= 2 ? previous.y_m + state.velocity_y_ms * elapsed_s : previous.y_m;
    if (state.sample_count >= 2)
    {
        state.association_gate_radius_m =
            2.0 * thresholds.position_uncertainty_m + 0.5 * thresholds.maximum_acceleration_ms2 * elapsed_s * elapsed_s;
    }

    std::size_t admissible_count = 0;
    StormContinuityReason aggregate_rejection = StormContinuityReason::SpeedGate;
    for (std::size_t rank = 0; rank < candidate_count; ++rank)
    {
        auto& candidate = candidates[rank];
        const double displacement_x = candidate.sample.x_m - previous.x_m;
        const double displacement_y = candidate.sample.y_m - previous.y_m;
        const double observed_vx     = displacement_x / elapsed_s;
        const double observed_vy     = displacement_y / elapsed_s;
        const double candidate_speed = std::hypot(observed_vx, observed_vy);
        candidate.innovation_m =
            std::hypot(candidate.sample.x_m - predicted_x, candidate.sample.y_m - predicted_y);
        if (candidate_speed > thresholds.maximum_speed_ms)
        {
            candidate.rejection = StormContinuityReason::SpeedGate;
            continue;
        }
        if (state.sample_count >= 2 && candidate.innovation_m > state.association_gate_radius_m)
        {
            candidate.rejection = StormContinuityReason::AccelerationGate;
            aggregate_rejection = StormContinuityReason::AccelerationGate;
            continue;
        }
        ++admissible_count;
        if (association.environmental_motion_available)
        {
            const double right_mismatch =
                std::hypot(observed_vx - association.bunkers_right_velocity_x_ms,
                           observed_vy - association.bunkers_right_velocity_y_ms);
            const double left_mismatch =
                std::hypot(observed_vx - association.bunkers_left_velocity_x_ms,
                           observed_vy - association.bunkers_left_velocity_y_ms);
            candidate.right_mismatch_ms = right_mismatch;
            candidate.left_mismatch_ms  = left_mismatch;
            candidate.motion_class =
                classify_motion(right_mismatch, left_mismatch, association.environmental_velocity_uncertainty_ms);
            if (association.preference == StormBranchPreference::BunkersLeft)
            {
                candidate.preferred_mismatch_ms = left_mismatch;
            }
            else
            {
                candidate.preferred_mismatch_ms = right_mismatch;
            }
        }
    }
    if (admissible_count == 0)
    {
        // A core merger can move the scalar centroid several kilometres in
        // one sample even though the physical condensate partition still
        // covers the previously tracked storm. Preserve identity only when a
        // single candidate covers a majority of the last accepted envelope.
        // Resetting the motion window prevents that topology jump from being
        // diagnosed as translation. Competing split fragments remain
        // ambiguous because no unique majority successor exists.
        constexpr double minimum_previous_envelope_coverage = 0.5;
        std::size_t overlap_successor = candidate_count;
        for (std::size_t rank = 0; rank < candidate_count; ++rank)
        {
            candidates[rank].previous_envelope_coverage =
                detection.previous_partition_available
                    ? candidates[rank].previous_partition_coverage
                    : previous_envelope_coverage(state, candidates[rank]);
            if (candidates[rank].previous_envelope_coverage < minimum_previous_envelope_coverage)
                continue;
            if (overlap_successor != candidate_count)
            {
                reset_pending_topology(state);
                return miss(StormContinuityReason::SpatialAmbiguity);
            }
            overlap_successor = rank;
        }
        if (overlap_successor == candidate_count)
        {
            reset_pending_topology(state);
            return miss(aggregate_rejection);
        }

        state.missed_samples = 0;
        state.reason = StormContinuityReason::None;
        state.selected_candidate_available = true;
        state.selected_candidate_rank = candidates[overlap_successor].rank;
        state.selected_component_label = candidates[overlap_successor].component_label;
        state.innovation_distance_m = candidates[overlap_successor].innovation_m;
        state.sample_count = 0;
        append_sample(state, candidates[overlap_successor].sample);
        remember_association_envelope(state, candidates[overlap_successor]);
        fit_motion(state, thresholds);
        state.status = StormContinuityStatus::Tentative;
        state.topology_reorganization = true;
        state.reorganization_overlap_fraction = candidates[overlap_successor].previous_envelope_coverage;
        reset_pending_topology(state);
        return state;
    }
    update_topology(state,
                    admissible_count > 1 ? StormBranchTopology::Multiple : StormBranchTopology::Single,
                    thresholds);

    std::size_t best = candidate_count;
    for (std::size_t rank = 0; rank < candidate_count; ++rank)
        if (candidates[rank].rejection == StormContinuityReason::None &&
            (best == candidate_count || candidates[rank].innovation_m < candidates[best].innovation_m))
            best = rank;

    std::array<std::size_t, storm_branch_candidate_capacity> equivalent{};
    std::size_t equivalent_count = 0;
    for (std::size_t rank = 0; rank < candidate_count; ++rank)
        if (candidates[rank].rejection == StormContinuityReason::None &&
            candidates[rank].innovation_m <= candidates[best].innovation_m + thresholds.position_uncertainty_m)
            equivalent[equivalent_count++] = rank;

    if (equivalent_count > 1)
    {
        if (association.preference == StormBranchPreference::Continuity)
            return miss(StormContinuityReason::SpatialAmbiguity);
        best = equivalent[0];
        std::size_t runner_up = candidate_count;
        for (std::size_t index = 1; index < equivalent_count; ++index)
        {
            const std::size_t rank = equivalent[index];
            if (candidates[rank].preferred_mismatch_ms < candidates[best].preferred_mismatch_ms)
            {
                runner_up = best;
                best = rank;
            }
            else if (runner_up == candidate_count ||
                     candidates[rank].preferred_mismatch_ms < candidates[runner_up].preferred_mismatch_ms)
            {
                runner_up = rank;
            }
        }
        if (runner_up == candidate_count ||
            candidates[runner_up].preferred_mismatch_ms <=
                candidates[best].preferred_mismatch_ms + association.environmental_velocity_uncertainty_ms)
            return miss(StormContinuityReason::SpatialAmbiguity);
    }

    state.missed_samples = 0;
    state.reason         = StormContinuityReason::None;
    state.selected_candidate_available = true;
    state.selected_candidate_rank      = candidates[best].rank;
    state.selected_component_label     = candidates[best].component_label;
    state.selected_motion_class        = candidates[best].motion_class;
    state.selected_bunkers_right_mismatch_ms = candidates[best].right_mismatch_ms;
    state.selected_bunkers_left_mismatch_ms  = candidates[best].left_mismatch_ms;
    state.innovation_distance_m = candidates[best].innovation_m;
    append_sample(state, candidates[best].sample);
    remember_association_envelope(state, candidates[best]);
    fit_motion(state, thresholds);
    state.status = state.motion_available ? StormContinuityStatus::Tracking : StormContinuityStatus::Tentative;
    return state;
}

StormEnvelopeFaceMotion derive_storm_envelope_face_motion(
    const StormContinuityState& prior_continuity,
    const StormFeatureDetection& current_detection,
    const StormContinuityState& current_continuity,
    double current_sample_time_s)
{
    StormEnvelopeFaceMotion result;
    if (!valid_storm_continuity_state(prior_continuity) ||
        !valid_storm_continuity_state(current_continuity))
    {
        throw std::invalid_argument(
            "storm envelope face motion requires valid continuity states");
    }
    if (!std::isfinite(current_sample_time_s))
        throw std::invalid_argument(
            "storm envelope face motion requires a finite sample time");

    const auto accepted_selection = [](const StormContinuityState& state) {
        return (state.status == StormContinuityStatus::Tentative ||
                state.status == StormContinuityStatus::Tracking) &&
               state.reason == StormContinuityReason::None &&
               state.missed_samples == 0 &&
               state.selected_candidate_available &&
               state.association_envelope_available &&
               state.sample_count > 0 && state.branch_id != 0 &&
               !state.topology_reorganization &&
               state.topology_event == StormBranchTopologyEvent::None;
    };
    if (!accepted_selection(prior_continuity) ||
        !accepted_selection(current_continuity) ||
        prior_continuity.branch_id != current_continuity.branch_id ||
        current_detection.branch_candidate_overflow ||
        current_continuity.selected_candidate_rank >=
            current_detection.reported_branch_candidate_count)
    {
        return result;
    }

    const auto& current_candidate = current_detection.branch_candidates[
        current_continuity.selected_candidate_rank];
    const double current_faces[] = {
        current_candidate.envelope_west_m,
        current_candidate.envelope_east_m,
        current_candidate.envelope_south_m,
        current_candidate.envelope_north_m,
    };
    if (!std::all_of(std::begin(current_faces), std::end(current_faces),
                     [](double value) { return std::isfinite(value); }) ||
        current_candidate.envelope_east_m <=
            current_candidate.envelope_west_m ||
        current_candidate.envelope_north_m <=
            current_candidate.envelope_south_m)
    {
        throw std::invalid_argument(
            "storm envelope face motion requires valid selected geometry");
    }
    if (current_continuity.association_envelope_west_m !=
            current_candidate.envelope_west_m ||
        current_continuity.association_envelope_east_m !=
            current_candidate.envelope_east_m ||
        current_continuity.association_envelope_south_m !=
            current_candidate.envelope_south_m ||
        current_continuity.association_envelope_north_m !=
            current_candidate.envelope_north_m)
    {
        throw std::invalid_argument(
            "storm envelope face motion detection does not match continuity ownership");
    }

    const double prior_sample_time_s =
        prior_continuity.samples[prior_continuity.sample_count - 1].time_s;
    const double represented_current_time_s =
        current_continuity.samples[current_continuity.sample_count - 1].time_s;
    if (represented_current_time_s != current_sample_time_s)
        throw std::invalid_argument(
            "storm envelope face motion sample time does not match continuity ownership");
    const double interval_s = current_sample_time_s - prior_sample_time_s;
    if (!std::isfinite(interval_s) || !(interval_s > 0.0))
        throw std::invalid_argument(
            "storm envelope face motion requires a finite positive interval");

    const auto coordinate_velocity = [interval_s](double prior, double current) {
        const double scale = std::max({1.0, std::abs(prior), std::abs(current)});
        const double zero_tolerance =
            32.0 * std::numeric_limits<double>::epsilon() * scale;
        const double displacement = current - prior;
        if (std::abs(displacement) <= zero_tolerance)
            return 0.0;
        const double velocity = displacement / interval_s;
        if (!std::isfinite(velocity))
            throw std::overflow_error(
                "storm envelope face velocity overflow");
        return velocity;
    };

    result.interval_s = interval_s;
    result.west_coordinate_velocity_ms = coordinate_velocity(
        prior_continuity.association_envelope_west_m,
        current_candidate.envelope_west_m);
    result.east_coordinate_velocity_ms = coordinate_velocity(
        prior_continuity.association_envelope_east_m,
        current_candidate.envelope_east_m);
    result.south_coordinate_velocity_ms = coordinate_velocity(
        prior_continuity.association_envelope_south_m,
        current_candidate.envelope_south_m);
    result.north_coordinate_velocity_ms = coordinate_velocity(
        prior_continuity.association_envelope_north_m,
        current_candidate.envelope_north_m);
    result.available = true;
    return result;
}

bool valid_storm_continuity_state(const StormContinuityState& state) noexcept
{
    const bool valid_status =
        state.status == StormContinuityStatus::Uninitialized || state.status == StormContinuityStatus::Tentative ||
        state.status == StormContinuityStatus::Tracking || state.status == StormContinuityStatus::Coasting ||
        state.status == StormContinuityStatus::Ambiguous || state.status == StormContinuityStatus::Lost;
    const bool valid_reason =
        state.reason == StormContinuityReason::None || state.reason == StormContinuityReason::WeakDetection ||
        state.reason == StormContinuityReason::SpatialAmbiguity ||
        state.reason == StormContinuityReason::BranchCandidateOverflow ||
        state.reason == StormContinuityReason::SpeedGate ||
        state.reason == StormContinuityReason::AccelerationGate;
    const double represented_speed = std::hypot(state.velocity_x_ms, state.velocity_y_ms);
    const double speed_tolerance   = 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, represented_speed);
    const bool status_requires_samples =
        state.status == StormContinuityStatus::Tentative || state.status == StormContinuityStatus::Tracking ||
        state.status == StormContinuityStatus::Coasting;
    const bool status_requires_no_samples =
        state.status == StormContinuityStatus::Uninitialized || state.status == StormContinuityStatus::Lost;
    const bool accepted_status =
        state.status == StormContinuityStatus::Tentative || state.status == StormContinuityStatus::Tracking;
    const bool missed_status =
        state.status == StormContinuityStatus::Coasting || state.status == StormContinuityStatus::Ambiguous ||
        state.status == StormContinuityStatus::Lost;
    if (state.sample_count > storm_continuity_history_capacity || !std::isfinite(state.velocity_x_ms) ||
        !std::isfinite(state.velocity_y_ms) || !std::isfinite(state.speed_ms) ||
        !std::isfinite(state.velocity_uncertainty_ms) || !std::isfinite(state.continuity_confidence) ||
        !std::isfinite(state.innovation_distance_m) || !std::isfinite(state.association_gate_radius_m) ||
        !std::isfinite(state.selected_bunkers_right_mismatch_ms) ||
        !std::isfinite(state.selected_bunkers_left_mismatch_ms) ||
        !std::isfinite(state.association_envelope_west_m) ||
        !std::isfinite(state.association_envelope_east_m) ||
        !std::isfinite(state.association_envelope_south_m) ||
        !std::isfinite(state.association_envelope_north_m) ||
        !std::isfinite(state.reorganization_overlap_fraction) || !valid_status || !valid_reason ||
        !valid_topology(state.topology) || !valid_topology(state.pending_topology) ||
        !valid_topology_event(state.topology_event) || !valid_motion_class(state.selected_motion_class) ||
        state.speed_ms < 0.0 || state.velocity_uncertainty_ms < 0.0 ||
        state.continuity_confidence < 0.0 || state.continuity_confidence > 1.0 || state.innovation_distance_m < 0.0 ||
        state.association_gate_radius_m < 0.0 || state.selected_bunkers_right_mismatch_ms < 0.0 ||
        state.selected_bunkers_left_mismatch_ms < 0.0 ||
        state.reorganization_overlap_fraction < 0.0 || state.reorganization_overlap_fraction > 1.0 ||
        (state.association_envelope_available &&
         (!(state.association_envelope_east_m > state.association_envelope_west_m) ||
          !(state.association_envelope_north_m > state.association_envelope_south_m))) ||
        (!state.association_envelope_available &&
         (state.association_envelope_west_m != 0.0 || state.association_envelope_east_m != 0.0 ||
          state.association_envelope_south_m != 0.0 || state.association_envelope_north_m != 0.0)) ||
        (state.topology_reorganization &&
         (!state.selected_candidate_available || state.status != StormContinuityStatus::Tentative ||
          state.sample_count != 1 || state.motion_available || state.missed_samples != 0 ||
          state.reorganization_overlap_fraction < 0.5)) ||
        (!state.topology_reorganization && state.reorganization_overlap_fraction != 0.0) ||
        ((state.pending_topology_samples == 0) !=
         (state.pending_topology == StormBranchTopology::Unknown)) ||
        (state.pending_topology != StormBranchTopology::Unknown && state.pending_topology == state.topology) ||
        (state.selected_candidate_available &&
         state.selected_candidate_rank >= storm_branch_candidate_capacity) ||
        (!state.selected_candidate_available &&
         (state.selected_candidate_rank != 0 || state.selected_component_label != 0 ||
          state.selected_motion_class != StormBranchMotionClass::Unclassified ||
          state.selected_bunkers_right_mismatch_ms != 0.0 ||
          state.selected_bunkers_left_mismatch_ms != 0.0)) ||
        std::abs(state.speed_ms - represented_speed) > speed_tolerance ||
        (state.motion_available != (state.status == StormContinuityStatus::Tracking)) ||
        (state.motion_available && state.sample_count < 2) || (status_requires_samples && state.sample_count == 0) ||
        (status_requires_no_samples && state.sample_count != 0) || (state.sample_count > 0 && state.branch_id == 0) ||
        (accepted_status && (state.missed_samples != 0 || state.reason != StormContinuityReason::None)) ||
        (missed_status && state.missed_samples == 0) ||
        (state.status == StormContinuityStatus::Coasting && state.reason != StormContinuityReason::WeakDetection) ||
        (state.status == StormContinuityStatus::Ambiguous &&
         state.reason != StormContinuityReason::SpatialAmbiguity && state.reason != StormContinuityReason::SpeedGate &&
         state.reason != StormContinuityReason::AccelerationGate &&
         state.reason != StormContinuityReason::BranchCandidateOverflow) ||
        (state.status == StormContinuityStatus::Lost && state.reason == StormContinuityReason::None) ||
        (state.status == StormContinuityStatus::Uninitialized &&
         state.reason != StormContinuityReason::None && state.reason != StormContinuityReason::WeakDetection))
    {
        return false;
    }
    for (std::size_t index = 0; index < state.sample_count; ++index)
    {
        const auto& sample = state.samples[index];
        if (!std::isfinite(sample.time_s) || !std::isfinite(sample.x_m) || !std::isfinite(sample.y_m) ||
            !std::isfinite(sample.confidence) || sample.confidence <= 0.0 || sample.confidence > 1.0 ||
            (index > 0 && !(sample.time_s > state.samples[index - 1].time_s)))
        {
            return false;
        }
    }
    return true;
}

} // namespace tmv::diagnostics
