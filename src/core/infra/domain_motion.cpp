#include "core/infra/domain_motion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace tmv::domain
{

namespace
{

bool finite(HorizontalPoint value)
{
    return std::isfinite(value.x_m) && std::isfinite(value.y_m);
}

bool finite(HorizontalVelocity value)
{
    return std::isfinite(value.u_ms) && std::isfinite(value.v_ms);
}

void require_finite_non_negative(const char* name, double value)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
}

void consider_risk(
    BoundaryFace face, double initial_clearance_m, double clearance_rate_ms, double run_time_s, StormLifetimePlan& plan)
{
    double time_s = std::numeric_limits<double>::infinity();
    if (initial_clearance_m < 0.0)
    {
        time_s = 0.0;
    }
    else if (clearance_rate_ms < 0.0)
    {
        time_s = initial_clearance_m / -clearance_rate_ms;
    }

    if (time_s <= run_time_s && time_s < plan.first_risk_time_s)
    {
        plan.first_risk_time_s = time_s;
        plan.first_risk_face   = face;
    }
}

void validate_wind_profile(const std::vector<double>& z_m,
                           const std::vector<double>& u_ms,
                           const std::vector<double>& v_ms)
{
    if (z_m.empty() || z_m.size() != u_ms.size() || z_m.size() != v_ms.size())
        throw std::invalid_argument("storm-motion wind columns must be non-empty and have matching lengths");

    for (std::size_t k = 0; k < z_m.size(); ++k)
    {
        if (!std::isfinite(z_m[k]) || !std::isfinite(u_ms[k]) || !std::isfinite(v_ms[k]))
            throw std::invalid_argument("storm-motion wind profile must contain only finite values");
        if (k > 0 && z_m[k] <= z_m[k - 1])
            throw std::invalid_argument("storm-motion heights must be strictly increasing");
    }
    if (z_m.front() < 0.0)
        throw std::invalid_argument("storm-motion heights must not extend below the surface");
    if (z_m.back() < 6000.0)
        throw std::invalid_argument("Bunkers storm motion requires a wind profile reaching 6000 m");
}

double interpolate_profile(const std::vector<double>& z_m, const std::vector<double>& values, double height_m)
{
    if (height_m <= z_m.front())
        return values.front();

    const auto upper = std::upper_bound(z_m.begin(), z_m.end(), height_m);
    if (upper == z_m.end())
        return values.back();
    const std::size_t hi       = static_cast<std::size_t>(upper - z_m.begin());
    const std::size_t lo       = hi - 1;
    const double      fraction = (height_m - z_m[lo]) / (z_m[hi] - z_m[lo]);
    return values[lo] + fraction * (values[hi] - values[lo]);
}

double height_mean(const std::vector<double>& z_m, const std::vector<double>& values, double bottom_m, double top_m)
{
    double integral = 0.0;
    double left     = bottom_m;
    double left_v   = interpolate_profile(z_m, values, left);

    for (double level_m : z_m)
    {
        if (level_m <= bottom_m || level_m >= top_m)
            continue;
        const double right_v = interpolate_profile(z_m, values, level_m);
        integral += 0.5 * (left_v + right_v) * (level_m - left);
        left   = level_m;
        left_v = right_v;
    }

    const double top_v = interpolate_profile(z_m, values, top_m);
    integral += 0.5 * (left_v + top_v) * (top_m - left);
    return integral / (top_m - bottom_m);
}

void validate_movement_policy(const StormWindowMovementPolicy& policy)
{
    require_finite_non_negative("inner_margin_m", policy.inner_margin_m);
    if (!std::isfinite(policy.release_margin_m) || policy.release_margin_m <= policy.inner_margin_m)
        throw std::invalid_argument("release_margin_m must be finite and greater than inner_margin_m");
    if (!std::isfinite(policy.dx_m) || policy.dx_m <= 0.0 || !std::isfinite(policy.dy_m) || policy.dy_m <= 0.0)
        throw std::invalid_argument("storm-window movement requires finite positive dx and dy");
    if (policy.maximum_shift_cells == 0 ||
        policy.maximum_shift_cells > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("maximum_shift_cells must fit a positive signed cell displacement");
    require_finite_non_negative("cooldown_s", policy.cooldown_s);
    if (!std::isfinite(policy.minimum_confidence) || policy.minimum_confidence < 0.0 ||
        policy.minimum_confidence > 1.0)
        throw std::invalid_argument("minimum_confidence must be finite and in [0, 1]");
    if (policy.confirmation_samples < 2)
        throw std::invalid_argument("storm-window movement requires at least two confirmation samples");
    require_finite_non_negative(
        "relaxation_margin_x_m", policy.relaxation_margin_x_m);
    require_finite_non_negative(
        "relaxation_margin_y_m", policy.relaxation_margin_y_m);
    const bool projected_relaxation_disabled =
        policy.relaxation_margin_x_m == 0.0 &&
        policy.relaxation_margin_y_m == 0.0;
    const bool projected_relaxation_enabled =
        policy.relaxation_margin_x_m > 0.0 &&
        policy.relaxation_margin_y_m > 0.0 &&
        policy.relaxation_margin_x_m < policy.inner_margin_m &&
        policy.relaxation_margin_y_m < policy.inner_margin_m;
    if (!projected_relaxation_disabled && !projected_relaxation_enabled)
        throw std::invalid_argument(
            "projected relaxation margins must both be zero or both be positive and smaller than inner_margin_m");
}

void validate_movement_evidence(const StormWindowMovementEvidence& evidence)
{
    if (!std::isfinite(evidence.sample_time_s))
        throw std::invalid_argument("storm-window movement sample time must be finite");
    if (!std::isfinite(evidence.confidence) || evidence.confidence < 0.0 || evidence.confidence > 1.0)
        throw std::invalid_argument("storm-window movement confidence must be finite and in [0, 1]");
    if (!std::isfinite(evidence.distance_west_m) || !std::isfinite(evidence.distance_east_m) ||
        !std::isfinite(evidence.distance_south_m) || !std::isfinite(evidence.distance_north_m) ||
        !std::isfinite(evidence.velocity_x_ms) || !std::isfinite(evidence.velocity_y_ms) ||
        !std::isfinite(evidence.envelope_face_motion_interval_s) ||
        !std::isfinite(evidence.envelope_west_coordinate_velocity_ms) ||
        !std::isfinite(evidence.envelope_east_coordinate_velocity_ms) ||
        !std::isfinite(evidence.envelope_south_coordinate_velocity_ms) ||
        !std::isfinite(evidence.envelope_north_coordinate_velocity_ms))
        throw std::invalid_argument("storm-window movement geometry and velocity evidence must be finite");
    if (evidence.envelope_face_motion_available)
    {
        if (!(evidence.envelope_face_motion_interval_s > 0.0))
            throw std::invalid_argument(
                "available storm-envelope face motion requires a positive interval");
    }
    else if (evidence.envelope_face_motion_interval_s != 0.0 ||
             evidence.envelope_west_coordinate_velocity_ms != 0.0 ||
             evidence.envelope_east_coordinate_velocity_ms != 0.0 ||
             evidence.envelope_south_coordinate_velocity_ms != 0.0 ||
             evidence.envelope_north_coordinate_velocity_ms != 0.0)
    {
        throw std::invalid_argument(
            "unavailable storm-envelope face motion must be canonical zero");
    }
    if (evidence.legal_shift_bounds_available)
    {
        if (evidence.minimum_i_shift_cells > 0 ||
            evidence.maximum_i_shift_cells < 0 ||
            evidence.minimum_j_shift_cells > 0 ||
            evidence.maximum_j_shift_cells < 0)
        {
            throw std::invalid_argument(
                "available movement-placement bounds must contain zero on both axes");
        }
    }
    else if (evidence.minimum_i_shift_cells != 0 ||
             evidence.maximum_i_shift_cells != 0 ||
             evidence.minimum_j_shift_cells != 0 ||
             evidence.maximum_j_shift_cells != 0)
    {
        throw std::invalid_argument(
            "unavailable movement-placement bounds must be canonical zero");
    }
}

std::size_t legal_directional_shift_limit(
    const StormWindowMovementEvidence& evidence, bool x_axis, int direction,
    std::size_t policy_limit) noexcept
{
    if (!evidence.legal_shift_bounds_available || direction == 0)
        return policy_limit;
    const int lower = x_axis ? evidence.minimum_i_shift_cells
                             : evidence.minimum_j_shift_cells;
    const int upper = x_axis ? evidence.maximum_i_shift_cells
                             : evidence.maximum_j_shift_cells;
    const auto placement_limit = direction > 0
        ? static_cast<std::uint64_t>(upper)
        : static_cast<std::uint64_t>(-static_cast<std::int64_t>(lower));
    return std::min(
        policy_limit, static_cast<std::size_t>(placement_limit));
}

int clamp_shift_to_legal_placement(
    const StormWindowMovementEvidence& evidence, bool x_axis,
    int requested) noexcept
{
    if (!evidence.legal_shift_bounds_available)
        return requested;
    const int lower = x_axis ? evidence.minimum_i_shift_cells
                             : evidence.minimum_j_shift_cells;
    const int upper = x_axis ? evidence.maximum_i_shift_cells
                             : evidence.maximum_j_shift_cells;
    return std::clamp(requested, lower, upper);
}

void clear_pending_movement(StormWindowMovementState& state)
{
    state.pending_i_direction             = 0;
    state.pending_j_direction             = 0;
    state.consecutive_compatible_samples = 0;
}

double attainable_release_margin(double first_distance_m,
                                  double second_distance_m,
                                  double spacing_m,
                                  const StormWindowMovementPolicy& policy)
{
    // Integer translations preserve the sum of the two face distances. The
    // closest attainable centering leaves the distance difference equal to
    // its signed remainder modulo two cell widths. This makes the hysteresis
    // target reachable on the actual grid while retaining the inner margin
    // as an absolute safety floor.
    const long double difference =
        static_cast<long double>(first_distance_m)
        - static_cast<long double>(second_distance_m);
    const long double period = 2.0L * static_cast<long double>(spacing_m);
    const long double centered_error = std::abs(std::remainder(
        difference, period));
    const long double best_symmetric_margin = 0.5L * (
        static_cast<long double>(first_distance_m)
        + static_cast<long double>(second_distance_m)
        - centered_error);
    return std::max(
        policy.inner_margin_m,
        std::min(policy.release_margin_m,
                 static_cast<double>(best_symmetric_margin)));
}

int safest_shift_cells(double                           distance_m,
                       double                           opposite_distance_m,
                       double                           target_margin_m,
                       double                           spacing_m,
                       const StormWindowMovementPolicy& policy,
                       bool&                            exceeds_limit)
{
    const double required_cells = std::ceil(
        (target_margin_m - distance_m) / spacing_m);
    exceeds_limit = required_cells > static_cast<double>(policy.maximum_shift_cells);
    if (exceeds_limit)
        return 0;

    const int minimum_cells = std::max(1, static_cast<int>(required_cells));
    const double exact_centered_cells =
        (opposite_distance_m - distance_m) / (2.0 * spacing_m);
    const double lower_centered_cells = std::floor(exact_centered_cells);
    const double upper_centered_cells = lower_centered_cells + 1.0;
    const double centered_cells =
        exact_centered_cells - lower_centered_cells <=
                upper_centered_cells - exact_centered_cells
            ? lower_centered_cells
            : upper_centered_cells;
    const double bounded_centered_cells = std::max(
        static_cast<double>(minimum_cells),
        std::min(static_cast<double>(policy.maximum_shift_cells),
                 centered_cells));
    return static_cast<int>(bounded_centered_cells);
}

bool recovered_release_margin_unchecked(
    const StormWindowMovementEvidence& evidence,
    const StormWindowMovementPolicy& policy)
{
    const double x_release = attainable_release_margin(
        evidence.distance_west_m, evidence.distance_east_m,
        policy.dx_m, policy);
    const double y_release = attainable_release_margin(
        evidence.distance_south_m, evidence.distance_north_m,
        policy.dy_m, policy);
    return evidence.distance_west_m >= x_release &&
           evidence.distance_east_m >= x_release &&
           evidence.distance_south_m >= y_release &&
           evidence.distance_north_m >= y_release;
}

} // namespace

double storm_window_release_margin_m(
    double inner_margin_m, double dx_m, double dy_m)
{
    require_finite_non_negative("inner_margin_m", inner_margin_m);
    if (!std::isfinite(dx_m) || dx_m <= 0.0 ||
        !std::isfinite(dy_m) || dy_m <= 0.0)
    {
        throw std::invalid_argument(
            "storm-window release margin requires finite positive dx and dy");
    }
    const double release_margin_m = inner_margin_m + std::max(dx_m, dy_m);
    if (!std::isfinite(release_margin_m))
        throw std::overflow_error("storm-window release margin overflow");
    return release_margin_m;
}

std::size_t minimum_storm_window_emergency_shift_cells(
    double inner_margin_m, double dx_m, double dy_m)
{
    const double release_margin_m =
        storm_window_release_margin_m(inner_margin_m, dx_m, dy_m);
    const double required_cells = std::max(
        std::ceil(release_margin_m / dx_m),
        std::ceil(release_margin_m / dy_m));
    if (!std::isfinite(required_cells) || required_cells < 1.0 ||
        required_cells > static_cast<double>(
            std::numeric_limits<std::size_t>::max()))
    {
        throw std::overflow_error(
            "storm-window emergency shift capacity overflow");
    }
    return static_cast<std::size_t>(required_cells);
}

void HorizontalDomainFrame::validate() const
{
    if (!finite(earth_origin_at_t0_m) || !finite(velocity_ms))
        throw std::invalid_argument("domain frame origin and velocity must be finite");
}

HorizontalPoint HorizontalDomainFrame::earth_origin_at(double simulation_time_s) const
{
    validate();
    if (!std::isfinite(simulation_time_s))
        throw std::invalid_argument("simulation time must be finite");
    return {
        earth_origin_at_t0_m.x_m + velocity_ms.u_ms * simulation_time_s,
        earth_origin_at_t0_m.y_m + velocity_ms.v_ms * simulation_time_s,
    };
}

HorizontalPoint HorizontalDomainFrame::grid_to_earth(HorizontalPoint grid_position_m, double simulation_time_s) const
{
    if (!finite(grid_position_m))
        throw std::invalid_argument("grid position must be finite");
    const HorizontalPoint origin = earth_origin_at(simulation_time_s);
    return {grid_position_m.x_m + origin.x_m, grid_position_m.y_m + origin.y_m};
}

HorizontalPoint HorizontalDomainFrame::earth_to_grid(HorizontalPoint earth_position_m, double simulation_time_s) const
{
    if (!finite(earth_position_m))
        throw std::invalid_argument("earth position must be finite");
    const HorizontalPoint origin = earth_origin_at(simulation_time_s);
    return {earth_position_m.x_m - origin.x_m, earth_position_m.y_m - origin.y_m};
}

HorizontalVelocity HorizontalDomainFrame::grid_to_earth(HorizontalVelocity grid_velocity_ms) const
{
    validate();
    if (!finite(grid_velocity_ms))
        throw std::invalid_argument("grid-relative velocity must be finite");
    return {
        grid_velocity_ms.u_ms + velocity_ms.u_ms,
        grid_velocity_ms.v_ms + velocity_ms.v_ms,
    };
}

HorizontalVelocity HorizontalDomainFrame::earth_to_grid(HorizontalVelocity earth_velocity_ms) const
{
    validate();
    if (!finite(earth_velocity_ms))
        throw std::invalid_argument("earth-relative velocity must be finite");
    return {
        earth_velocity_ms.u_ms - velocity_ms.u_ms,
        earth_velocity_ms.v_ms - velocity_ms.v_ms,
    };
}

const char* boundary_face_name(BoundaryFace face) noexcept
{
    switch (face)
    {
    case BoundaryFace::None:
        return "none";
    case BoundaryFace::West:
        return "west";
    case BoundaryFace::East:
        return "east";
    case BoundaryFace::South:
        return "south";
    case BoundaryFace::North:
        return "north";
    }
    return "none";
}

const char* storm_motion_method_name(StormMotionMethod method) noexcept
{
    switch (method)
    {
    case StormMotionMethod::Explicit:
        return "explicit";
    case StormMotionMethod::BunkersRight:
        return "bunkers_right";
    case StormMotionMethod::BunkersLeft:
        return "bunkers_left";
    }
    return "explicit";
}

const char* storm_motion_algorithm_id(StormMotionMethod method) noexcept
{
    switch (method)
    {
    case StormMotionMethod::Explicit:
        return "explicit_earth_relative_v1";
    case StormMotionMethod::BunkersRight:
    case StormMotionMethod::BunkersLeft:
        return "bunkers_2000_height_integrated_v1";
    }
    return "unknown";
}

StormMotionEstimate estimate_bunkers_storm_motion(const std::vector<double>& z_m,
                                                  const std::vector<double>& u_ms,
                                                  const std::vector<double>& v_ms,
                                                  StormMotionMethod          method)
{
    if (method != StormMotionMethod::BunkersRight && method != StormMotionMethod::BunkersLeft)
        throw std::invalid_argument("Bunkers estimator requires bunkers_right or bunkers_left");
    validate_wind_profile(z_m, u_ms, v_ms);

    const HorizontalVelocity mean{
        height_mean(z_m, u_ms, 0.0, 6000.0),
        height_mean(z_m, v_ms, 0.0, 6000.0),
    };
    const HorizontalVelocity low{
        height_mean(z_m, u_ms, 0.0, 500.0),
        height_mean(z_m, v_ms, 0.0, 500.0),
    };
    const HorizontalVelocity high{
        height_mean(z_m, u_ms, 5500.0, 6000.0),
        height_mean(z_m, v_ms, 5500.0, 6000.0),
    };
    const HorizontalVelocity shear{
        high.u_ms - low.u_ms,
        high.v_ms - low.v_ms,
    };
    const double shear_speed_ms = std::hypot(shear.u_ms, shear.v_ms);
    if (!std::isfinite(shear_speed_ms) || shear_speed_ms <= 1.0e-9)
    {
        throw std::invalid_argument("Bunkers storm motion is undefined for negligible 0--6 km shear; "
                                    "configure an explicit storm-motion vector");
    }

    // Right of the low-to-high shear vector in Cartesian (east, north)
    // coordinates is (+dv, -du). The left mover uses the opposite sign.
    const double             side               = method == StormMotionMethod::BunkersRight ? 1.0 : -1.0;
    constexpr double         deviation_speed_ms = 7.5;
    const HorizontalVelocity deviation{
        side * deviation_speed_ms * shear.v_ms / shear_speed_ms,
        -side * deviation_speed_ms * shear.u_ms / shear_speed_ms,
    };

    return {
        {mean.u_ms + deviation.u_ms, mean.v_ms + deviation.v_ms},
        mean,
        shear,
        method,
    };
}

StormLifetimePlan plan_storm_lifetime(const StormLifetimePlanInput& input)
{
    if (input.nx == 0 || input.ny == 0)
        throw std::invalid_argument("storm lifetime planning requires non-zero nx and ny");
    if (!std::isfinite(input.dx_m) || input.dx_m <= 0.0 || !std::isfinite(input.dy_m) || input.dy_m <= 0.0)
    {
        throw std::invalid_argument("storm lifetime planning requires finite positive dx and dy");
    }
    require_finite_non_negative("run_time_s", input.run_time_s);
    require_finite_non_negative("upstream_buffer_m", input.upstream_buffer_m);
    require_finite_non_negative("downstream_buffer_m", input.downstream_buffer_m);
    require_finite_non_negative("cross_track_buffer_m", input.cross_track_buffer_m);
    require_finite_non_negative("motion_uncertainty_ms", input.motion_uncertainty_ms);
    if (!finite(input.initial_storm_center_m) || !finite(input.earth_relative_storm_motion_ms))
    {
        throw std::invalid_argument("storm center and motion must be finite");
    }
    input.frame.validate();

    StormLifetimePlan plan;
    plan.grid_relative_storm_motion_ms = input.frame.earth_to_grid(input.earth_relative_storm_motion_ms);

    const double vx               = plan.grid_relative_storm_motion_ms.u_ms;
    const double vy               = plan.grid_relative_storm_motion_ms.v_ms;
    plan.projected_final_center_m = {
        input.initial_storm_center_m.x_m + vx * input.run_time_s,
        input.initial_storm_center_m.y_m + vy * input.run_time_s,
    };

    // Use relative motion for the window corridor. When translation makes the
    // storm stationary in grid coordinates, preserve the Earth-motion
    // direction so asymmetric along-track buffers remain meaningful.
    double direction_x    = vx;
    double direction_y    = vy;
    double direction_norm = std::hypot(direction_x, direction_y);
    if (direction_norm == 0.0)
    {
        direction_x    = input.earth_relative_storm_motion_ms.u_ms;
        direction_y    = input.earth_relative_storm_motion_ms.v_ms;
        direction_norm = std::hypot(direction_x, direction_y);
    }
    if (direction_norm == 0.0)
    {
        direction_x    = 1.0;
        direction_y    = 0.0;
        direction_norm = 1.0;
    }
    direction_x /= direction_norm;
    direction_y /= direction_norm;
    const double normal_x = -direction_y;
    const double normal_y = direction_x;

    std::array<HorizontalPoint, 4> offsets{};
    std::size_t                    index = 0;
    for (double along : {-input.upstream_buffer_m, input.downstream_buffer_m})
    {
        for (double cross : {-input.cross_track_buffer_m, input.cross_track_buffer_m})
        {
            offsets[index++] = {
                along * direction_x + cross * normal_x,
                along * direction_y + cross * normal_y,
            };
        }
    }

    double min_offset_x = offsets.front().x_m;
    double max_offset_x = offsets.front().x_m;
    double min_offset_y = offsets.front().y_m;
    double max_offset_y = offsets.front().y_m;
    for (const HorizontalPoint offset : offsets)
    {
        min_offset_x = std::min(min_offset_x, offset.x_m);
        max_offset_x = std::max(max_offset_x, offset.x_m);
        min_offset_y = std::min(min_offset_y, offset.y_m);
        max_offset_y = std::max(max_offset_y, offset.y_m);
    }

    const double uncertainty_m = input.motion_uncertainty_ms * input.run_time_s;
    const double start_x       = input.initial_storm_center_m.x_m;
    const double start_y       = input.initial_storm_center_m.y_m;
    const double end_x         = plan.projected_final_center_m.x_m;
    const double end_y         = plan.projected_final_center_m.y_m;

    plan.required_bounds_m = {
        std::min(start_x + min_offset_x, end_x + min_offset_x - uncertainty_m),
        std::max(start_x + max_offset_x, end_x + max_offset_x + uncertainty_m),
        std::min(start_y + min_offset_y, end_y + min_offset_y - uncertainty_m),
        std::max(start_y + max_offset_y, end_y + max_offset_y + uncertainty_m),
    };

    plan.required_nx =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(plan.required_bounds_m.width_m() / input.dx_m)));
    plan.required_ny =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(plan.required_bounds_m.height_m() / input.dy_m)));

    const double domain_width_m  = static_cast<double>(input.nx) * input.dx_m;
    const double domain_height_m = static_cast<double>(input.ny) * input.dy_m;
    plan.contained_for_run = plan.required_bounds_m.west_m >= 0.0 && plan.required_bounds_m.east_m <= domain_width_m &&
                             plan.required_bounds_m.south_m >= 0.0 && plan.required_bounds_m.north_m <= domain_height_m;

    const double envelope_center_x = 0.5 * (plan.required_bounds_m.west_m + plan.required_bounds_m.east_m);
    const double envelope_center_y = 0.5 * (plan.required_bounds_m.south_m + plan.required_bounds_m.north_m);
    // A placement recommendation inside an undersized domain is impossible.
    // Center the envelope inside the minimum whole-cell domain in that case
    // instead of returning a misleading outside/negative trigger location.
    const double recommendation_width_m = std::max(domain_width_m, static_cast<double>(plan.required_nx) * input.dx_m);
    const double recommendation_height_m =
        std::max(domain_height_m, static_cast<double>(plan.required_ny) * input.dy_m);
    plan.recommended_initial_center_m = {
        start_x + 0.5 * recommendation_width_m - envelope_center_x,
        start_y + 0.5 * recommendation_height_m - envelope_center_y,
    };

    const double initial_west            = start_x + min_offset_x;
    const double initial_east_clearance  = domain_width_m - (start_x + max_offset_x);
    const double initial_south           = start_y + min_offset_y;
    const double initial_north_clearance = domain_height_m - (start_y + max_offset_y);
    const double uncertainty_rate        = input.motion_uncertainty_ms;

    consider_risk(BoundaryFace::West, initial_west, vx - uncertainty_rate, input.run_time_s, plan);
    consider_risk(BoundaryFace::East, initial_east_clearance, -(vx + uncertainty_rate), input.run_time_s, plan);
    consider_risk(BoundaryFace::South, initial_south, vy - uncertainty_rate, input.run_time_s, plan);
    consider_risk(BoundaryFace::North, initial_north_clearance, -(vy + uncertainty_rate), input.run_time_s, plan);

    return plan;
}

const char* storm_window_movement_status_name(StormWindowMovementStatus status) noexcept
{
    switch (status)
    {
    case StormWindowMovementStatus::Ineligible:
        return "ineligible";
    case StormWindowMovementStatus::PendingConfirmation:
        return "pending_confirmation";
    case StormWindowMovementStatus::Requested:
        return "requested";
    case StormWindowMovementStatus::Cooldown:
        return "cooldown";
    case StormWindowMovementStatus::AwaitingRearm:
        return "awaiting_rearm";
    }
    return "ineligible";
}

const char* storm_window_movement_reason_name(StormWindowMovementReason reason) noexcept
{
    switch (reason)
    {
    case StormWindowMovementReason::NoSelectedFeature:
        return "no_selected_feature";
    case StormWindowMovementReason::TrackerNotAuthoritative:
        return "tracker_not_authoritative";
    case StormWindowMovementReason::MotionUnavailable:
        return "motion_unavailable";
    case StormWindowMovementReason::AmbiguousTrack:
        return "ambiguous_track";
    case StormWindowMovementReason::BranchCandidateOverflow:
        return "branch_candidate_overflow";
    case StormWindowMovementReason::TopologyReorganization:
        return "topology_reorganization";
    case StormWindowMovementReason::InsufficientConfidence:
        return "insufficient_confidence";
    case StormWindowMovementReason::NoApproachingBreach:
        return "no_approaching_breach";
    case StormWindowMovementReason::OppositeFacesBreached:
        return "opposite_faces_breached";
    case StormWindowMovementReason::ShiftLimitInsufficient:
        return "shift_limit_insufficient";
    case StormWindowMovementReason::OppositeReleaseMarginInsufficient:
        return "opposite_release_margin_insufficient";
    case StormWindowMovementReason::ConfirmationPending:
        return "confirmation_pending";
    case StormWindowMovementReason::RequestReady:
        return "request_ready";
    case StormWindowMovementReason::ProjectedRelaxationRisk:
        return "projected_relaxation_risk";
    case StormWindowMovementReason::TrustedInteriorContainment:
        return "trusted_interior_containment";
    case StormWindowMovementReason::TrustedInteriorContained:
        return "trusted_interior_contained";
    case StormWindowMovementReason::ParentStencilLimitedContained:
        return "parent_stencil_limited_contained";
    case StormWindowMovementReason::EmergencyContainment:
        return "emergency_containment";
    case StormWindowMovementReason::CooldownActive:
        return "cooldown_active";
    case StormWindowMovementReason::ReleaseMarginNotRecovered:
        return "release_margin_not_recovered";
    }
    return "no_selected_feature";
}

const char* storm_window_movement_algorithm_id() noexcept
{
    return "qualified_envelope_hysteresis_v14";
}

bool valid_storm_window_movement_state(const StormWindowMovementState& state) noexcept
{
    const bool valid_i_direction = state.pending_i_direction >= -1 && state.pending_i_direction <= 1;
    const bool valid_j_direction = state.pending_j_direction >= -1 && state.pending_j_direction <= 1;
    const bool has_pending_direction = state.pending_i_direction != 0 || state.pending_j_direction != 0;
    const bool valid_pending =
        (state.consecutive_compatible_samples == 0) == !has_pending_direction;
    const bool valid_sample_time =
        !state.sample_time_available || std::isfinite(state.last_sample_time_s);
    const bool valid_commit_time =
        !state.committed_move_available || std::isfinite(state.last_committed_time_s);
    const bool causal_commit =
        !state.committed_move_available ||
        (state.sample_time_available && state.last_committed_time_s <= state.last_sample_time_s);
    const bool valid_disarm = state.armed || state.committed_move_available;
    return valid_i_direction && valid_j_direction && valid_pending && valid_sample_time &&
           valid_commit_time && causal_commit && valid_disarm;
}

StormWindowMovementDecision evaluate_storm_window_movement(StormWindowMovementState&          state,
                                                           const StormWindowMovementEvidence& evidence,
                                                           const StormWindowMovementPolicy&   policy)
{
    validate_movement_policy(policy);
    validate_movement_evidence(evidence);
    if (!valid_storm_window_movement_state(state))
        throw std::invalid_argument("storm-window movement state is internally inconsistent");
    if (state.sample_time_available && evidence.sample_time_s <= state.last_sample_time_s)
        throw std::invalid_argument("storm-window movement samples must have strictly increasing times");

    state.sample_time_available = true;
    state.last_sample_time_s    = evidence.sample_time_s;

    StormWindowMovementDecision decision;
    decision.sample_time_s = evidence.sample_time_s;

    const auto ineligible = [&](StormWindowMovementReason reason) {
        clear_pending_movement(state);
        decision.status = StormWindowMovementStatus::Ineligible;
        decision.reason = reason;
        return decision;
    };

    if (evidence.ambiguous)
        return ineligible(StormWindowMovementReason::AmbiguousTrack);
    if (evidence.branch_candidate_overflow)
        return ineligible(StormWindowMovementReason::BranchCandidateOverflow);
    if (evidence.topology_reorganization)
        return ineligible(StormWindowMovementReason::TopologyReorganization);
    if (!evidence.selected_feature_available)
        return ineligible(StormWindowMovementReason::NoSelectedFeature);
    if (!evidence.continuity_authoritative)
        return ineligible(StormWindowMovementReason::TrackerNotAuthoritative);
    if (!evidence.motion_available)
        return ineligible(StormWindowMovementReason::MotionUnavailable);
    if (evidence.confidence < policy.minimum_confidence)
        return ineligible(StormWindowMovementReason::InsufficientConfidence);

    if (!state.armed)
    {
        clear_pending_movement(state);
        if (!recovered_release_margin_unchecked(evidence, policy))
        {
            decision.status = StormWindowMovementStatus::AwaitingRearm;
            decision.reason = StormWindowMovementReason::ReleaseMarginNotRecovered;
            return decision;
        }
        state.armed = true;
    }

    const bool west_breached  = evidence.distance_west_m < policy.inner_margin_m;
    const bool east_breached  = evidence.distance_east_m < policy.inner_margin_m;
    const bool south_breached = evidence.distance_south_m < policy.inner_margin_m;
    const bool north_breached = evidence.distance_north_m < policy.inner_margin_m;
    const bool projected_relaxation_enabled =
        policy.relaxation_margin_x_m > 0.0 &&
        policy.relaxation_margin_y_m > 0.0;
    const bool trusted_opposite_x =
        projected_relaxation_enabled && west_breached && east_breached &&
        evidence.distance_west_m >= policy.relaxation_margin_x_m &&
        evidence.distance_east_m >= policy.relaxation_margin_x_m;
    const bool trusted_opposite_y =
        projected_relaxation_enabled && south_breached && north_breached &&
        evidence.distance_south_m >= policy.relaxation_margin_y_m &&
        evidence.distance_north_m >= policy.relaxation_margin_y_m;
    const bool west_approaching =
        evidence.velocity_x_ms < 0.0 ||
        (evidence.envelope_face_motion_available &&
         evidence.envelope_west_coordinate_velocity_ms < 0.0);
    const bool east_approaching =
        evidence.velocity_x_ms > 0.0 ||
        (evidence.envelope_face_motion_available &&
         evidence.envelope_east_coordinate_velocity_ms > 0.0);
    const bool south_approaching =
        evidence.velocity_y_ms < 0.0 ||
        (evidence.envelope_face_motion_available &&
         evidence.envelope_south_coordinate_velocity_ms < 0.0);
    const bool north_approaching =
        evidence.velocity_y_ms > 0.0 ||
        (evidence.envelope_face_motion_available &&
         evidence.envelope_north_coordinate_velocity_ms > 0.0);
    if (((west_breached && east_breached) && !trusted_opposite_x) ||
        ((south_breached && north_breached) && !trusted_opposite_y))
        return ineligible(StormWindowMovementReason::OppositeFacesBreached);

    const bool any_inner_margin_breached =
        west_breached || east_breached || south_breached || north_breached;
    const bool every_breached_face_is_outside_relaxation =
        projected_relaxation_enabled &&
        (!west_breached ||
         evidence.distance_west_m >= policy.relaxation_margin_x_m) &&
        (!east_breached ||
         evidence.distance_east_m >= policy.relaxation_margin_x_m) &&
        (!south_breached ||
         evidence.distance_south_m >= policy.relaxation_margin_y_m) &&
        (!north_breached ||
         evidence.distance_north_m >= policy.relaxation_margin_y_m);
    const auto no_bounded_shift_strictly_improves_minimum = [](
        int direction, double negative_clearance_m,
        double positive_clearance_m, double spacing_m,
        double relaxation_margin_m, std::size_t maximum_shift_cells) {
        const long double current_minimum = std::min(
            static_cast<long double>(negative_clearance_m),
            static_cast<long double>(positive_clearance_m));
        for (std::size_t magnitude = 1; magnitude <= maximum_shift_cells;
             ++magnitude)
        {
            const long double candidate = static_cast<long double>(
                direction * static_cast<int>(magnitude));
            const long double shifted_negative =
                static_cast<long double>(negative_clearance_m) -
                candidate * static_cast<long double>(spacing_m);
            const long double shifted_positive =
                static_cast<long double>(positive_clearance_m) +
                candidate * static_cast<long double>(spacing_m);
            if (shifted_negative >= relaxation_margin_m &&
                shifted_positive >= relaxation_margin_m &&
                std::min(shifted_negative, shifted_positive) >
                    current_minimum)
            {
                return false;
            }
        }
        return true;
    };
    const bool trusted_receding_x =
        projected_relaxation_enabled &&
        (west_breached || east_breached) &&
        (!west_breached ||
         evidence.distance_west_m >= policy.relaxation_margin_x_m) &&
        (!east_breached ||
         evidence.distance_east_m >= policy.relaxation_margin_x_m) &&
        !(west_breached && west_approaching) &&
        !(east_breached && east_approaching);
    const bool trusted_receding_y =
        projected_relaxation_enabled &&
        (south_breached || north_breached) &&
        (!south_breached ||
         evidence.distance_south_m >= policy.relaxation_margin_y_m) &&
        (!north_breached ||
         evidence.distance_north_m >= policy.relaxation_margin_y_m) &&
        !(south_breached && south_approaching) &&
        !(north_breached && north_approaching);
    const bool trusted_quantized_x =
        projected_relaxation_enabled && (west_breached != east_breached) &&
        evidence.distance_west_m >= policy.relaxation_margin_x_m &&
        evidence.distance_east_m >= policy.relaxation_margin_x_m &&
        ((west_breached && west_approaching &&
          no_bounded_shift_strictly_improves_minimum(
              -1, evidence.distance_west_m, evidence.distance_east_m,
              policy.dx_m, policy.relaxation_margin_x_m,
              policy.maximum_shift_cells)) ||
         (east_breached && east_approaching &&
          no_bounded_shift_strictly_improves_minimum(
              1, evidence.distance_west_m, evidence.distance_east_m,
              policy.dx_m, policy.relaxation_margin_x_m,
              policy.maximum_shift_cells)));
    const bool trusted_quantized_y =
        projected_relaxation_enabled && (south_breached != north_breached) &&
        evidence.distance_south_m >= policy.relaxation_margin_y_m &&
        evidence.distance_north_m >= policy.relaxation_margin_y_m &&
        ((south_breached && south_approaching &&
          no_bounded_shift_strictly_improves_minimum(
              -1, evidence.distance_south_m, evidence.distance_north_m,
              policy.dy_m, policy.relaxation_margin_y_m,
              policy.maximum_shift_cells)) ||
         (north_breached && north_approaching &&
          no_bounded_shift_strictly_improves_minimum(
              1, evidence.distance_south_m, evidence.distance_north_m,
              policy.dy_m, policy.relaxation_margin_y_m,
              policy.maximum_shift_cells)));
    const bool trusted_parent_limited_x =
        evidence.legal_shift_bounds_available &&
        projected_relaxation_enabled && (west_breached != east_breached) &&
        evidence.distance_west_m >= policy.relaxation_margin_x_m &&
        evidence.distance_east_m >= policy.relaxation_margin_x_m &&
        ((west_breached && west_approaching &&
          legal_directional_shift_limit(evidence, true, -1,
                                        policy.maximum_shift_cells) <
              policy.maximum_shift_cells &&
          no_bounded_shift_strictly_improves_minimum(
              -1, evidence.distance_west_m, evidence.distance_east_m,
              policy.dx_m, policy.relaxation_margin_x_m,
              legal_directional_shift_limit(
                  evidence, true, -1, policy.maximum_shift_cells)) &&
          !no_bounded_shift_strictly_improves_minimum(
              -1, evidence.distance_west_m, evidence.distance_east_m,
              policy.dx_m, policy.relaxation_margin_x_m,
              policy.maximum_shift_cells)) ||
         (east_breached && east_approaching &&
          legal_directional_shift_limit(evidence, true, 1,
                                        policy.maximum_shift_cells) <
              policy.maximum_shift_cells &&
          no_bounded_shift_strictly_improves_minimum(
              1, evidence.distance_west_m, evidence.distance_east_m,
              policy.dx_m, policy.relaxation_margin_x_m,
              legal_directional_shift_limit(
                  evidence, true, 1, policy.maximum_shift_cells)) &&
          !no_bounded_shift_strictly_improves_minimum(
              1, evidence.distance_west_m, evidence.distance_east_m,
              policy.dx_m, policy.relaxation_margin_x_m,
              policy.maximum_shift_cells)));
    const bool trusted_parent_limited_y =
        evidence.legal_shift_bounds_available &&
        projected_relaxation_enabled && (south_breached != north_breached) &&
        evidence.distance_south_m >= policy.relaxation_margin_y_m &&
        evidence.distance_north_m >= policy.relaxation_margin_y_m &&
        ((south_breached && south_approaching &&
          legal_directional_shift_limit(evidence, false, -1,
                                        policy.maximum_shift_cells) <
              policy.maximum_shift_cells &&
          no_bounded_shift_strictly_improves_minimum(
              -1, evidence.distance_south_m, evidence.distance_north_m,
              policy.dy_m, policy.relaxation_margin_y_m,
              legal_directional_shift_limit(
                  evidence, false, -1, policy.maximum_shift_cells)) &&
          !no_bounded_shift_strictly_improves_minimum(
              -1, evidence.distance_south_m, evidence.distance_north_m,
              policy.dy_m, policy.relaxation_margin_y_m,
              policy.maximum_shift_cells)) ||
         (north_breached && north_approaching &&
          legal_directional_shift_limit(evidence, false, 1,
                                        policy.maximum_shift_cells) <
              policy.maximum_shift_cells &&
          no_bounded_shift_strictly_improves_minimum(
              1, evidence.distance_south_m, evidence.distance_north_m,
              policy.dy_m, policy.relaxation_margin_y_m,
              legal_directional_shift_limit(
                  evidence, false, 1, policy.maximum_shift_cells)) &&
          !no_bounded_shift_strictly_improves_minimum(
              1, evidence.distance_south_m, evidence.distance_north_m,
              policy.dy_m, policy.relaxation_margin_y_m,
              policy.maximum_shift_cells)));
    const bool every_breached_axis_is_trusted =
        ((!west_breached && !east_breached) || trusted_opposite_x ||
         trusted_receding_x || trusted_quantized_x ||
         trusted_parent_limited_x) &&
        ((!south_breached && !north_breached) || trusted_opposite_y ||
         trusted_receding_y || trusted_quantized_y ||
         trusted_parent_limited_y);
    // An inner-margin face that is receding is not an unresolved threat while
    // it remains in the trusted interior. An approaching face is likewise
    // already at the safest representable position when no bounded whole-cell
    // translation in its approach direction can strictly improve the weaker
    // opposite-face clearance. Keep both cases as no-mutation states; a face
    // already inside relaxation cannot satisfy any trusted predicate.
    if (any_inner_margin_breached &&
        every_breached_face_is_outside_relaxation &&
        every_breached_axis_is_trusted)
    {
        return ineligible(
            trusted_parent_limited_x || trusted_parent_limited_y
                ? StormWindowMovementReason::ParentStencilLimitedContained
                : StormWindowMovementReason::TrustedInteriorContained);
    }
    if (!any_inner_margin_breached && state.committed_move_available &&
        evidence.sample_time_s - state.last_committed_time_s < policy.cooldown_s)
    {
        clear_pending_movement(state);
        decision.status = StormWindowMovementStatus::Cooldown;
        decision.reason = StormWindowMovementReason::CooldownActive;
        return decision;
    }

    int requested_i = 0;
    int requested_j = 0;
    bool shift_limit_exceeded = false;
    bool axis_limit_exceeded  = false;
    const double x_release = attainable_release_margin(
        evidence.distance_west_m, evidence.distance_east_m,
        policy.dx_m, policy);
    const double y_release = attainable_release_margin(
        evidence.distance_south_m, evidence.distance_north_m,
        policy.dy_m, policy);
    if (west_breached && west_approaching)
    {
        requested_i = -safest_shift_cells(
            evidence.distance_west_m, evidence.distance_east_m, x_release,
            policy.dx_m, policy, axis_limit_exceeded);
        shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
    }
    else if (east_breached && east_approaching)
    {
        requested_i = safest_shift_cells(
            evidence.distance_east_m, evidence.distance_west_m, x_release,
            policy.dx_m, policy, axis_limit_exceeded);
        shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
    }
    if (south_breached && south_approaching)
    {
        requested_j = -safest_shift_cells(
            evidence.distance_south_m, evidence.distance_north_m, y_release,
            policy.dy_m, policy, axis_limit_exceeded);
        shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
    }
    else if (north_breached && north_approaching)
    {
        requested_j = safest_shift_cells(
            evidence.distance_north_m, evidence.distance_south_m, y_release,
            policy.dy_m, policy, axis_limit_exceeded);
        shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
    }

    if (shift_limit_exceeded)
        return ineligible(StormWindowMovementReason::ShiftLimitInsufficient);

    if (requested_i == 0 && requested_j == 0)
        return ineligible(StormWindowMovementReason::NoApproachingBreach);

    // A confirmed move must restore the complete two-dimensional release
    // box. An orthogonal face may still be outside that box even though it
    // has not crossed the smaller trigger margin. Include the smallest
    // integer correction for such an axis in the same transaction; otherwise
    // a scientifically valid move can leave the authority permanently
    // disarmed while the tracked envelope continues to drift.
    if (requested_i == 0)
    {
        if (evidence.distance_west_m < x_release)
        {
            requested_i = -safest_shift_cells(
                evidence.distance_west_m, evidence.distance_east_m, x_release,
                policy.dx_m, policy, axis_limit_exceeded);
            shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
        }
        else if (evidence.distance_east_m < x_release)
        {
            requested_i = safest_shift_cells(
                evidence.distance_east_m, evidence.distance_west_m, x_release,
                policy.dx_m, policy, axis_limit_exceeded);
            shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
        }
    }
    if (requested_j == 0)
    {
        if (evidence.distance_south_m < y_release)
        {
            requested_j = -safest_shift_cells(
                evidence.distance_south_m, evidence.distance_north_m, y_release,
                policy.dy_m, policy, axis_limit_exceeded);
            shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
        }
        else if (evidence.distance_north_m < y_release)
        {
            requested_j = safest_shift_cells(
                evidence.distance_north_m, evidence.distance_south_m, y_release,
                policy.dy_m, policy, axis_limit_exceeded);
            shift_limit_exceeded = shift_limit_exceeded || axis_limit_exceeded;
        }
    }

    if (shift_limit_exceeded)
        return ineligible(StormWindowMovementReason::ShiftLimitInsufficient);

    // The policy cap is relative to a single decision. A nested child also
    // owns an absolute legal-placement interval which can be narrower near a
    // parent edge. Intersect the complete two-axis request only after the
    // orthogonal release correction has been assembled, and before any
    // projection or transaction can observe it.
    requested_i = clamp_shift_to_legal_placement(
        evidence, true, requested_i);
    requested_j = clamp_shift_to_legal_placement(
        evidence, false, requested_j);
    if (requested_i == 0 && requested_j == 0)
        return ineligible(StormWindowMovementReason::ShiftLimitInsufficient);

    const bool emergency_containment =
        (west_breached && evidence.distance_west_m < policy.dx_m) ||
        (east_breached && evidence.distance_east_m < policy.dx_m) ||
        (south_breached && evidence.distance_south_m < policy.dy_m) ||
        (north_breached && evidence.distance_north_m < policy.dy_m);
    const bool outside_relaxation =
        !projected_relaxation_enabled ||
        (evidence.distance_west_m >= policy.relaxation_margin_x_m &&
         evidence.distance_east_m >= policy.relaxation_margin_x_m &&
         evidence.distance_south_m >= policy.relaxation_margin_y_m &&
         evidence.distance_north_m >= policy.relaxation_margin_y_m);
    const long double face_interval_s =
        static_cast<long double>(evidence.envelope_face_motion_interval_s);
    const bool west_projected_relaxation_risk =
        projected_relaxation_enabled && outside_relaxation &&
        evidence.envelope_face_motion_available && west_breached &&
          evidence.envelope_west_coordinate_velocity_ms < 0.0 &&
          static_cast<long double>(evidence.distance_west_m) +
                  static_cast<long double>(
                      evidence.envelope_west_coordinate_velocity_ms) *
                      face_interval_s <
              static_cast<long double>(policy.relaxation_margin_x_m);
    const bool east_projected_relaxation_risk =
        projected_relaxation_enabled && outside_relaxation &&
        evidence.envelope_face_motion_available && east_breached &&
          evidence.envelope_east_coordinate_velocity_ms > 0.0 &&
          static_cast<long double>(evidence.distance_east_m) -
                  static_cast<long double>(
                      evidence.envelope_east_coordinate_velocity_ms) *
                      face_interval_s <
              static_cast<long double>(policy.relaxation_margin_x_m);
    const bool south_projected_relaxation_risk =
        projected_relaxation_enabled && outside_relaxation &&
        evidence.envelope_face_motion_available && south_breached &&
          evidence.envelope_south_coordinate_velocity_ms < 0.0 &&
          static_cast<long double>(evidence.distance_south_m) +
                  static_cast<long double>(
                      evidence.envelope_south_coordinate_velocity_ms) *
                      face_interval_s <
              static_cast<long double>(policy.relaxation_margin_y_m);
    const bool north_projected_relaxation_risk =
        projected_relaxation_enabled && outside_relaxation &&
        evidence.envelope_face_motion_available && north_breached &&
          evidence.envelope_north_coordinate_velocity_ms > 0.0 &&
          static_cast<long double>(evidence.distance_north_m) -
                  static_cast<long double>(
                      evidence.envelope_north_coordinate_velocity_ms) *
                      face_interval_s <
              static_cast<long double>(policy.relaxation_margin_y_m);
    const bool projected_relaxation_risk =
        west_projected_relaxation_risk || east_projected_relaxation_risk ||
        south_projected_relaxation_risk || north_projected_relaxation_risk;
    const bool bounded_projected_relaxation_risk =
        projected_relaxation_risk && !emergency_containment;

    // A projected-risk request has already proved that waiting one cadence can
    // lose the trusted interior. The minimum current-box shift leaves no
    // protection against accelerating envelope growth. On each projected-risk
    // axis, consume the complete admitted shift budget that still preserves
    // the opposite release face. Ordinary requests retain their minimum shift.
    const auto maximize_projected_shift = [](
        int requested, double negative_clearance_m,
        double positive_clearance_m, double spacing_m,
        double release_margin_m, std::size_t maximum_shift_cells) {
        if (requested == 0)
            return requested;
        const int direction = requested > 0 ? 1 : -1;
        for (std::size_t magnitude = maximum_shift_cells;
             magnitude > static_cast<std::size_t>(std::abs(requested));
             --magnitude)
        {
            const int candidate = direction * static_cast<int>(magnitude);
            const double shifted_negative =
                negative_clearance_m - candidate * spacing_m;
            const double shifted_positive =
                positive_clearance_m + candidate * spacing_m;
            if (shifted_negative >= release_margin_m &&
                shifted_positive >= release_margin_m)
            {
                return candidate;
            }
        }
        return requested;
    };
    if (bounded_projected_relaxation_risk &&
        (west_projected_relaxation_risk || east_projected_relaxation_risk))
    {
        requested_i = maximize_projected_shift(
            requested_i, evidence.distance_west_m, evidence.distance_east_m,
            policy.dx_m, policy.release_margin_m,
            legal_directional_shift_limit(
                evidence, true, requested_i > 0 ? 1 : -1,
                policy.maximum_shift_cells));
    }
    if (bounded_projected_relaxation_risk &&
        (south_projected_relaxation_risk || north_projected_relaxation_risk))
    {
        requested_j = maximize_projected_shift(
            requested_j, evidence.distance_south_m, evidence.distance_north_m,
            policy.dy_m, policy.release_margin_m,
            legal_directional_shift_limit(
                evidence, false, requested_j > 0 ? 1 : -1,
                policy.maximum_shift_cells));
    }

    auto shifted_west_m =
        evidence.distance_west_m - requested_i * policy.dx_m;
    auto shifted_east_m =
        evidence.distance_east_m + requested_i * policy.dx_m;
    auto shifted_south_m =
        evidence.distance_south_m - requested_j * policy.dy_m;
    auto shifted_north_m =
        evidence.distance_north_m + requested_j * policy.dy_m;
    auto complete_release_box_after_shift = [&] {
        return shifted_west_m >= policy.release_margin_m &&
               shifted_east_m >= policy.release_margin_m &&
               shifted_south_m >= policy.release_margin_m &&
               shifted_north_m >= policy.release_margin_m;
    };

    // A wide but still trusted envelope can temporarily make the larger
    // hysteresis release box unattainable. Refusing every translation in that
    // state conflates an operational rearm target with physical validity. For
    // a bounded projected risk only, choose the whole-cell translation in the
    // approaching direction that maximizes the current minimum face clearance
    // while keeping both faces outside relaxation. This admits no envelope
    // already in relaxation and leaves ordinary confirmed moves unchanged.
    const auto best_trusted_containment_shift = [](
        int direction, double negative_clearance_m,
        double positive_clearance_m, double spacing_m,
        double relaxation_margin_m, std::size_t maximum_shift_cells) {
        int best_shift = 0;
        double best_minimum =
            std::min(negative_clearance_m, positive_clearance_m);
        for (std::size_t magnitude = 1; magnitude <= maximum_shift_cells;
             ++magnitude)
        {
            const int candidate =
                direction * static_cast<int>(magnitude);
            const double shifted_negative =
                negative_clearance_m - candidate * spacing_m;
            const double shifted_positive =
                positive_clearance_m + candidate * spacing_m;
            const double candidate_minimum =
                std::min(shifted_negative, shifted_positive);
            if (shifted_negative >= relaxation_margin_m &&
                shifted_positive >= relaxation_margin_m &&
                candidate_minimum > best_minimum)
            {
                best_shift = candidate;
                best_minimum = candidate_minimum;
            }
        }
        return best_shift;
    };
    const auto maximize_orthogonal_release_reserve = [](
        int direction, double negative_clearance_m,
        double positive_clearance_m, double spacing_m,
        double release_margin_m, std::size_t maximum_shift_cells) {
        int best_shift = 0;
        for (std::size_t magnitude = 1; magnitude <= maximum_shift_cells;
             ++magnitude)
        {
            const int candidate =
                direction * static_cast<int>(magnitude);
            const double shifted_negative =
                negative_clearance_m - candidate * spacing_m;
            const double shifted_positive =
                positive_clearance_m + candidate * spacing_m;
            if (shifted_negative >= release_margin_m &&
                shifted_positive >= release_margin_m)
            {
                best_shift = candidate;
            }
        }
        return best_shift;
    };
    bool trusted_interior_containment = false;
    if (bounded_projected_relaxation_risk &&
        !complete_release_box_after_shift())
    {
        if (west_projected_relaxation_risk ||
            east_projected_relaxation_risk)
        {
            requested_i = best_trusted_containment_shift(
                west_projected_relaxation_risk ? -1 : 1,
                evidence.distance_west_m, evidence.distance_east_m,
                policy.dx_m, policy.relaxation_margin_x_m,
                legal_directional_shift_limit(
                    evidence, true,
                    west_projected_relaxation_risk ? -1 : 1,
                    policy.maximum_shift_cells));
        }
        if (south_projected_relaxation_risk ||
            north_projected_relaxation_risk)
        {
            requested_j = best_trusted_containment_shift(
                south_projected_relaxation_risk ? -1 : 1,
                evidence.distance_south_m, evidence.distance_north_m,
                policy.dy_m, policy.relaxation_margin_y_m,
                legal_directional_shift_limit(
                    evidence, false,
                    south_projected_relaxation_risk ? -1 : 1,
                    policy.maximum_shift_cells));
        }
        // A wide envelope can expand abruptly on the axis orthogonal to the
        // currently projected face. The containment transaction is already
        // immediate, so consume only the orthogonal whole-cell slack above
        // the immutable release margin in the authoritative approach
        // direction. This builds reserve without admitting a current
        // relaxation entry or weakening either opposite release face.
        if ((west_projected_relaxation_risk ||
             east_projected_relaxation_risk) &&
            requested_j == 0)
        {
            const int reserve_direction =
                north_approaching != south_approaching
                    ? (north_approaching ? 1 : -1)
                    : 0;
            if (reserve_direction != 0)
            {
                requested_j = maximize_orthogonal_release_reserve(
                    reserve_direction,
                    evidence.distance_south_m,
                    evidence.distance_north_m,
                    policy.dy_m, policy.release_margin_m,
                    legal_directional_shift_limit(
                        evidence, false, reserve_direction,
                        policy.maximum_shift_cells));
            }
        }
        if ((south_projected_relaxation_risk ||
             north_projected_relaxation_risk) &&
            requested_i == 0)
        {
            const int reserve_direction =
                east_approaching != west_approaching
                    ? (east_approaching ? 1 : -1)
                    : 0;
            if (reserve_direction != 0)
            {
                requested_i = maximize_orthogonal_release_reserve(
                    reserve_direction,
                    evidence.distance_west_m,
                    evidence.distance_east_m,
                    policy.dx_m, policy.release_margin_m,
                    legal_directional_shift_limit(
                        evidence, true, reserve_direction,
                        policy.maximum_shift_cells));
            }
        }
        shifted_west_m =
            evidence.distance_west_m - requested_i * policy.dx_m;
        shifted_east_m =
            evidence.distance_east_m + requested_i * policy.dx_m;
        shifted_south_m =
            evidence.distance_south_m - requested_j * policy.dy_m;
        shifted_north_m =
            evidence.distance_north_m + requested_j * policy.dy_m;
        trusted_interior_containment =
            (requested_i != 0 || requested_j != 0) &&
            shifted_west_m >= policy.relaxation_margin_x_m &&
            shifted_east_m >= policy.relaxation_margin_x_m &&
            shifted_south_m >= policy.relaxation_margin_y_m &&
            shifted_north_m >= policy.relaxation_margin_y_m;
    }
    // An ordinary approaching breach can also be wider than the attainable
    // inner release box. Unlike the quantized no-mutation state above, admit
    // a relocation here only when the existing trusted optimizer finds a
    // strict improvement in the weaker opposite-face clearance. Ordinary
    // confirmation remains in force because this is not a projected
    // relaxation emergency.
    if (projected_relaxation_enabled &&
        !bounded_projected_relaxation_risk &&
        !complete_release_box_after_shift())
    {
        const bool x_release_unattainable =
            requested_i != 0 &&
            (shifted_west_m < x_release || shifted_east_m < x_release);
        const bool y_release_unattainable =
            requested_j != 0 &&
            (shifted_south_m < y_release || shifted_north_m < y_release);
        if (x_release_unattainable)
        {
            requested_i = best_trusted_containment_shift(
                requested_i < 0 ? -1 : 1,
                evidence.distance_west_m, evidence.distance_east_m,
                policy.dx_m, policy.relaxation_margin_x_m,
                legal_directional_shift_limit(
                    evidence, true, requested_i < 0 ? -1 : 1,
                    policy.maximum_shift_cells));
        }
        if (y_release_unattainable)
        {
            requested_j = best_trusted_containment_shift(
                requested_j < 0 ? -1 : 1,
                evidence.distance_south_m, evidence.distance_north_m,
                policy.dy_m, policy.relaxation_margin_y_m,
                legal_directional_shift_limit(
                    evidence, false, requested_j < 0 ? -1 : 1,
                    policy.maximum_shift_cells));
        }
        shifted_west_m =
            evidence.distance_west_m - requested_i * policy.dx_m;
        shifted_east_m =
            evidence.distance_east_m + requested_i * policy.dx_m;
        shifted_south_m =
            evidence.distance_south_m - requested_j * policy.dy_m;
        shifted_north_m =
            evidence.distance_north_m + requested_j * policy.dy_m;
        const bool every_unattainable_axis_strictly_improved =
            (!x_release_unattainable || requested_i != 0) &&
            (!y_release_unattainable || requested_j != 0);
        trusted_interior_containment =
            (x_release_unattainable || y_release_unattainable) &&
            every_unattainable_axis_strictly_improved &&
            (requested_i != 0 || requested_j != 0) &&
            shifted_west_m >= policy.relaxation_margin_x_m &&
            shifted_east_m >= policy.relaxation_margin_x_m &&
            shifted_south_m >= policy.relaxation_margin_y_m &&
            shifted_north_m >= policy.relaxation_margin_y_m;
    }
    if (!trusted_interior_containment &&
        ((requested_i != 0 &&
          (shifted_west_m < x_release || shifted_east_m < x_release)) ||
         (requested_j != 0 &&
          (shifted_south_m < y_release || shifted_north_m < y_release))))
    {
        return ineligible(
            StormWindowMovementReason::OppositeReleaseMarginInsufficient);
    }
    if (bounded_projected_relaxation_risk &&
        !complete_release_box_after_shift() &&
        !trusted_interior_containment)
    {
        return ineligible(
            StormWindowMovementReason::OppositeReleaseMarginInsufficient);
    }
    const bool immediate_projected_request =
        bounded_projected_relaxation_risk;

    const int i_direction = (requested_i > 0) - (requested_i < 0);
    const int j_direction = (requested_j > 0) - (requested_j < 0);
    if (i_direction != state.pending_i_direction || j_direction != state.pending_j_direction)
    {
        state.pending_i_direction             = i_direction;
        state.pending_j_direction             = j_direction;
        state.consecutive_compatible_samples = 1;
    }
    else if (state.consecutive_compatible_samples < policy.confirmation_samples)
    {
        ++state.consecutive_compatible_samples;
    }

    decision.requested_i_shift_cells = requested_i;
    decision.requested_j_shift_cells = requested_j;
    decision.confirmation_count       = state.consecutive_compatible_samples;
    if (state.consecutive_compatible_samples < policy.confirmation_samples &&
        !emergency_containment && !immediate_projected_request)
    {
        decision.status = StormWindowMovementStatus::PendingConfirmation;
        decision.reason = StormWindowMovementReason::ConfirmationPending;
    }
    else
    {
        decision.status = StormWindowMovementStatus::Requested;
        decision.reason = emergency_containment
            ? StormWindowMovementReason::EmergencyContainment
            : trusted_interior_containment
                ? StormWindowMovementReason::TrustedInteriorContainment
            : immediate_projected_request
                ? StormWindowMovementReason::ProjectedRelaxationRisk
                : StormWindowMovementReason::RequestReady;
    }
    return decision;
}

bool storm_window_started_new_confirmation(
    const StormWindowMovementState& state_before,
    const StormWindowMovementDecision& decision) noexcept
{
    if (decision.status != StormWindowMovementStatus::PendingConfirmation ||
        decision.reason != StormWindowMovementReason::ConfirmationPending ||
        decision.confirmation_count != 1)
    {
        return false;
    }
    const int requested_i_direction =
        (decision.requested_i_shift_cells > 0) -
        (decision.requested_i_shift_cells < 0);
    const int requested_j_direction =
        (decision.requested_j_shift_cells > 0) -
        (decision.requested_j_shift_cells < 0);
    return state_before.consecutive_compatible_samples == 0 ||
           requested_i_direction != state_before.pending_i_direction ||
           requested_j_direction != state_before.pending_j_direction;
}

bool storm_window_release_margin_recovered(
    const StormWindowMovementEvidence& evidence,
    const StormWindowMovementPolicy& policy)
{
    validate_movement_policy(policy);
    validate_movement_evidence(evidence);
    return recovered_release_margin_unchecked(evidence, policy);
}

void commit_storm_window_movement(StormWindowMovementState&          state,
                                  const StormWindowMovementDecision& decision)
{
    if (decision.status != StormWindowMovementStatus::Requested ||
        (decision.requested_i_shift_cells == 0 && decision.requested_j_shift_cells == 0))
        throw std::invalid_argument("only a non-zero requested storm-window movement may be committed");
    if (!state.armed || !state.sample_time_available || decision.sample_time_s != state.last_sample_time_s)
        throw std::invalid_argument("storm-window movement commit does not match the current armed decision");

    const auto checked_add = [](std::int64_t current, int displacement) {
        if ((displacement > 0 && current > std::numeric_limits<std::int64_t>::max() - displacement) ||
            (displacement < 0 && current < std::numeric_limits<std::int64_t>::min() - displacement))
            throw std::overflow_error("cumulative storm-window displacement overflow");
        return current + displacement;
    };
    state.cumulative_i_shift_cells = checked_add(state.cumulative_i_shift_cells,
                                                  decision.requested_i_shift_cells);
    state.cumulative_j_shift_cells = checked_add(state.cumulative_j_shift_cells,
                                                  decision.requested_j_shift_cells);
    state.committed_move_available = true;
    state.last_committed_time_s    = decision.sample_time_s;
    state.armed                    = false;
    clear_pending_movement(state);
    if (!valid_storm_window_movement_state(state))
        throw std::logic_error("committed storm-window movement produced invalid authority state");
}

} // namespace tmv::domain
