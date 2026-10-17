#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tmv::domain
{

struct HorizontalPoint
{
    double x_m = 0.0;
    double y_m = 0.0;
};

struct HorizontalVelocity
{
    double u_ms = 0.0;
    double v_ms = 0.0;
};

/**
 * Constant-velocity horizontal coordinate frame.
 *
 * This type is the single coordinate/velocity authority used by frame-aware
 * operators. Runtime admission separately limits which motion modes and
 * hardware/science combinations may execute.
 */
struct HorizontalDomainFrame
{
    HorizontalPoint    earth_origin_at_t0_m;
    HorizontalVelocity velocity_ms;

    static constexpr HorizontalDomainFrame stationary() noexcept
    {
        return {};
    }

    void validate() const;

    HorizontalPoint    earth_origin_at(double simulation_time_s) const;
    HorizontalPoint    grid_to_earth(HorizontalPoint grid_position_m, double simulation_time_s) const;
    HorizontalPoint    earth_to_grid(HorizontalPoint earth_position_m, double simulation_time_s) const;
    HorizontalVelocity grid_to_earth(HorizontalVelocity grid_velocity_ms) const;
    HorizontalVelocity earth_to_grid(HorizontalVelocity earth_velocity_ms) const;
};

enum class BoundaryFace
{
    None,
    West,
    East,
    South,
    North,
};

const char* boundary_face_name(BoundaryFace face) noexcept;

enum class StormMotionMethod
{
    Explicit,
    BunkersRight,
    BunkersLeft,
};

const char* storm_motion_method_name(StormMotionMethod method) noexcept;
const char* storm_motion_algorithm_id(StormMotionMethod method) noexcept;

struct StormMotionEstimate
{
    HorizontalVelocity velocity_ms;
    HorizontalVelocity mean_wind_0_6km_ms;
    HorizontalVelocity shear_0_6km_ms;
    StormMotionMethod  method = StormMotionMethod::Explicit;
};

/**
 * Estimate supercell motion from a resolved Cartesian wind column.
 *
 * Bunkers et al. (2000), Weather and Forecasting 15, 61--79,
 * doi:10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2.
 *
 * Nimbus uses exact height integrals of the piecewise-linear model wind
 * profile for the 0--6 km, 0--0.5 km, and 5.5--6 km layer means, followed by
 * the published 7.5 m/s shear-normal deviation. Below the first model center
 * the resolved value is held constant to the surface; the profile must reach
 * 6 km and must contain a non-degenerate 0--6 km shear vector.
 */
StormMotionEstimate estimate_bunkers_storm_motion(const std::vector<double>& z_m,
                                                  const std::vector<double>& u_ms,
                                                  const std::vector<double>& v_ms,
                                                  StormMotionMethod          method);

struct StormLifetimePlanInput
{
    std::size_t nx         = 0;
    std::size_t ny         = 0;
    double      dx_m       = 0.0;
    double      dy_m       = 0.0;
    double      run_time_s = 0.0;

    HorizontalPoint       initial_storm_center_m;
    HorizontalVelocity    earth_relative_storm_motion_ms;
    HorizontalDomainFrame frame;

    double upstream_buffer_m     = 0.0;
    double downstream_buffer_m   = 0.0;
    double cross_track_buffer_m  = 0.0;
    double motion_uncertainty_ms = 0.0;
};

struct HorizontalBounds
{
    double west_m  = 0.0;
    double east_m  = 0.0;
    double south_m = 0.0;
    double north_m = 0.0;

    double width_m() const noexcept
    {
        return east_m - west_m;
    }
    double height_m() const noexcept
    {
        return north_m - south_m;
    }
};

struct StormLifetimePlan
{
    HorizontalVelocity grid_relative_storm_motion_ms;
    HorizontalPoint    projected_final_center_m;
    HorizontalBounds   required_bounds_m;
    std::size_t        required_nx = 0;
    std::size_t        required_ny = 0;
    HorizontalPoint    recommended_initial_center_m;
    bool               contained_for_run = false;
    BoundaryFace       first_risk_face   = BoundaryFace::None;
    double             first_risk_time_s = std::numeric_limits<double>::infinity();
};

/**
 * Conservative axis-aligned lifetime corridor for a straight motion estimate.
 *
 * The storm envelope is an asymmetric along-track/cross-track rectangle swept
 * from the initial to final estimated center. Motion uncertainty expands the
 * final bounds isotropically at `motion_uncertainty_ms * run_time_s`.
 */
StormLifetimePlan plan_storm_lifetime(const StormLifetimePlanInput& input);

/**
 * Diagnostic authority for a future whole-cell storm-window move.
 *
 * This contract deliberately does not move, allocate, initialize, destroy, or
 * otherwise mutate model state.  It converts qualified tracker evidence into
 * a bounded integer-cell request which a later transactional mover may either
 * commit atomically or reject without changing this state.
 */
struct StormWindowMovementPolicy
{
    double      inner_margin_m       = 0.0;
    double      release_margin_m     = 0.0;
    double      dx_m                 = 0.0;
    double      dy_m                 = 0.0;
    std::size_t maximum_shift_cells  = 0;
    double      cooldown_s           = 0.0;
    double      minimum_confidence   = 0.0;
    std::size_t confirmation_samples = 2;

    // A zero pair disables projected-relaxation admission (for ordinary
    // non-nested window users). A positive pair identifies the child sponge
    // widths that an authoritative advancing face must not enter while a
    // second confirmation sample is pending.
    double relaxation_margin_x_m = 0.0;
    double relaxation_margin_y_m = 0.0;
};

double storm_window_release_margin_m(
    double inner_margin_m, double dx_m, double dy_m);

std::size_t minimum_storm_window_emergency_shift_cells(
    double inner_margin_m, double dx_m, double dy_m);

struct StormWindowMovementEvidence
{
    double sample_time_s = 0.0;

    bool selected_feature_available = false;
    bool continuity_authoritative   = false;
    bool motion_available           = false;
    bool ambiguous                  = false;
    bool branch_candidate_overflow  = false;
    bool topology_reorganization    = false;

    double confidence       = 0.0;
    double distance_west_m  = 0.0;
    double distance_east_m  = 0.0;
    double distance_south_m = 0.0;
    double distance_north_m = 0.0;
    double velocity_x_ms    = 0.0;
    double velocity_y_ms    = 0.0;

    // Signed coordinate velocities of the four tracked envelope faces. They
    // complement centroid motion when storm growth advances only one edge of
    // the selected physical envelope. An unavailable sample is represented
    // canonically by a zero interval and four zero velocities.
    bool   envelope_face_motion_available          = false;
    double envelope_face_motion_interval_s         = 0.0;
    double envelope_west_coordinate_velocity_ms    = 0.0;
    double envelope_east_coordinate_velocity_ms    = 0.0;
    double envelope_south_coordinate_velocity_ms   = 0.0;
    double envelope_north_coordinate_velocity_ms   = 0.0;

    // A relocatable nested child has a second, absolute movement constraint:
    // its current placement must retain the parent interpolation stencil.
    // Bounds are signed parent-cell displacements relative to this sample and
    // must include zero. Ordinary non-nested windows use the canonical
    // unavailable/zero representation and remain limited only by policy.
    bool legal_shift_bounds_available = false;
    int  minimum_i_shift_cells = 0;
    int  maximum_i_shift_cells = 0;
    int  minimum_j_shift_cells = 0;
    int  maximum_j_shift_cells = 0;
};

enum class StormWindowMovementStatus
{
    Ineligible,
    PendingConfirmation,
    Requested,
    Cooldown,
    AwaitingRearm,
};

enum class StormWindowMovementReason
{
    NoSelectedFeature,
    TrackerNotAuthoritative,
    MotionUnavailable,
    AmbiguousTrack,
    BranchCandidateOverflow,
    TopologyReorganization,
    InsufficientConfidence,
    NoApproachingBreach,
    OppositeFacesBreached,
    ShiftLimitInsufficient,
    OppositeReleaseMarginInsufficient,
    ConfirmationPending,
    RequestReady,
    CooldownActive,
    ReleaseMarginNotRecovered,
    EmergencyContainment,
    ProjectedRelaxationRisk,
    TrustedInteriorContainment,
    TrustedInteriorContained,
    ParentStencilLimitedContained,
};

const char* storm_window_movement_status_name(StormWindowMovementStatus status) noexcept;
const char* storm_window_movement_reason_name(StormWindowMovementReason reason) noexcept;
const char* storm_window_movement_algorithm_id() noexcept;

struct StormWindowMovementState
{
    bool        armed                        = true;
    int         pending_i_direction          = 0;
    int         pending_j_direction          = 0;
    std::size_t consecutive_compatible_samples = 0;

    bool   sample_time_available = false;
    double last_sample_time_s    = 0.0;
    bool   committed_move_available = false;
    double last_committed_time_s    = 0.0;

    std::int64_t cumulative_i_shift_cells = 0;
    std::int64_t cumulative_j_shift_cells = 0;
};

struct StormWindowMovementDecision
{
    StormWindowMovementStatus status = StormWindowMovementStatus::Ineligible;
    StormWindowMovementReason reason = StormWindowMovementReason::NoSelectedFeature;
    double                    sample_time_s = 0.0;
    int                       requested_i_shift_cells = 0;
    int                       requested_j_shift_cells = 0;
    std::size_t               confirmation_count = 0;
};

bool valid_storm_window_movement_state(const StormWindowMovementState& state) noexcept;

StormWindowMovementDecision evaluate_storm_window_movement(StormWindowMovementState&          state,
                                                           const StormWindowMovementEvidence& evidence,
                                                           const StormWindowMovementPolicy&   policy);

/**
 * Return whether a pending decision is the first confirmation of its current
 * request direction.  A newly introduced orthogonal component resets the
 * confirmation sequence even when the previous one-axis request was already
 * pending.
 */
bool storm_window_started_new_confirmation(
    const StormWindowMovementState& state_before,
    const StormWindowMovementDecision& decision) noexcept;

/**
 * Return whether a successfully shifted window contains the sampled envelope
 * inside the complete release box. Callers evaluate this against post-shift
 * geometry before publishing an updated movement authority.
 */
bool storm_window_release_margin_recovered(
    const StormWindowMovementEvidence& evidence,
    const StormWindowMovementPolicy& policy);

/**
 * Record a request only after a later movement transaction has succeeded.
 * Diagnostic-only callers must never call this function.
 */
void commit_storm_window_movement(StormWindowMovementState&          state,
                                  const StormWindowMovementDecision& decision);

} // namespace tmv::domain
