#include "catch2/catch.hpp"
#include "core/infra/domain_motion.hpp"
#include "numerics/advection/cartesian_mass_flux.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <utility>

using namespace tmv::domain;

TEST_CASE("constant horizontal frame conversions are exact inverses", "[core][domain_motion][frame]")
{
    HorizontalDomainFrame frame;
    frame.earth_origin_at_t0_m = {1000.0, -500.0};
    frame.velocity_ms          = {15.0, -4.0};

    const HorizontalPoint origin = frame.earth_origin_at(20.0);
    REQUIRE(origin.x_m == Approx(1300.0));
    REQUIRE(origin.y_m == Approx(-580.0));

    const HorizontalPoint grid{250.0, 900.0};
    const HorizontalPoint earth = frame.grid_to_earth(grid, 20.0);
    REQUIRE(earth.x_m == Approx(1550.0));
    REQUIRE(earth.y_m == Approx(320.0));

    const HorizontalPoint round_trip = frame.earth_to_grid(earth, 20.0);
    REQUIRE(round_trip.x_m == Approx(grid.x_m));
    REQUIRE(round_trip.y_m == Approx(grid.y_m));

    const HorizontalVelocity earth_wind{30.0, 6.0};
    const HorizontalVelocity grid_wind = frame.earth_to_grid(earth_wind);
    REQUIRE(grid_wind.u_ms == Approx(15.0));
    REQUIRE(grid_wind.v_ms == Approx(10.0));
    const HorizontalVelocity wind_round_trip = frame.grid_to_earth(grid_wind);
    REQUIRE(wind_round_trip.u_ms == Approx(earth_wind.u_ms));
    REQUIRE(wind_round_trip.v_ms == Approx(earth_wind.v_ms));
}

TEST_CASE("domain frame rejects non-finite coordinate evidence", "[core][domain_motion][frame]")
{
    HorizontalDomainFrame frame;
    frame.velocity_ms.u_ms = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_WITH(frame.validate(), Catch::Matchers::Contains("must be finite"));
}

TEST_CASE("Cartesian face flux derives grid-relative transport without changing stored wind",
          "[core][domain_motion][frame][mass_flux]")
{
    using tmv::init::config::LateralBC;
    Field3D rho(5, 5, 3, 2.0f);
    Field3D u(5, 5, 3, 10.0f);
    Field3D v(5, 5, 3, -4.0f);

    for (StaggerType staggering : {StaggerType::Collocated, StaggerType::CGrid})
    {
        const double stationary_x = tmv::cartesian_mass_flux::x_face(
            rho, u, 2, 2, 1, 5, staggering,
            LateralBC::Periodic, LateralBC::Periodic);
        const double stationary_y = tmv::cartesian_mass_flux::y_face(
            rho, v, 2, 2, 1, 5, staggering,
            LateralBC::Periodic, LateralBC::Periodic);
        REQUIRE(stationary_x == 20.0);
        REQUIRE(stationary_y == -8.0);

        const double translated_x = tmv::cartesian_mass_flux::x_face(
            rho, u, 2, 2, 1, 5, staggering,
            LateralBC::Periodic, LateralBC::Periodic, nullptr, 3.0);
        const double translated_y = tmv::cartesian_mass_flux::y_face(
            rho, v, 2, 2, 1, 5, staggering,
            LateralBC::Periodic, LateralBC::Periodic, nullptr, -1.0);
        REQUIRE(translated_x == 14.0);
        REQUIRE(translated_y == -6.0);

        if (staggering == StaggerType::Collocated)
        {
            REQUIRE(tmv::cartesian_mass_flux::x_face(
                        rho, u, 0, 2, 1, 5, staggering,
                        LateralBC::Rigid, LateralBC::Rigid, nullptr, 3.0) == 0.0);
            REQUIRE(tmv::cartesian_mass_flux::y_face(
                        rho, v, 2, 0, 1, 5, staggering,
                        LateralBC::Rigid, LateralBC::Rigid, nullptr, -1.0) == 0.0);
        }
    }

    REQUIRE(u[2][2][1] == 10.0f);
    REQUIRE(v[2][2][1] == -4.0f);
}

TEST_CASE("Bunkers estimator height-integrates the resolved model wind", "[core][domain_motion][storm_motion]")
{
    const std::vector<double> z{0.0, 500.0, 5500.0, 6000.0};
    const std::vector<double> u{0.0, 0.0, 20.0, 20.0};
    const std::vector<double> v(z.size(), 0.0);

    const StormMotionEstimate right = estimate_bunkers_storm_motion(z, u, v, StormMotionMethod::BunkersRight);
    const StormMotionEstimate left  = estimate_bunkers_storm_motion(z, u, v, StormMotionMethod::BunkersLeft);

    REQUIRE(right.mean_wind_0_6km_ms.u_ms == Approx(10.0));
    REQUIRE(right.shear_0_6km_ms.u_ms == Approx(20.0));
    REQUIRE(right.velocity_ms.u_ms == Approx(10.0));
    REQUIRE(right.velocity_ms.v_ms == Approx(-7.5));
    REQUIRE(left.velocity_ms.u_ms == Approx(10.0));
    REQUIRE(left.velocity_ms.v_ms == Approx(7.5));
    REQUIRE(std::string(storm_motion_algorithm_id(right.method)) == "bunkers_2000_height_integrated_v1");
}

TEST_CASE("Bunkers estimator preserves a uniform Galilean translation", "[core][domain_motion][storm_motion]")
{
    const std::vector<double> z{250.0, 1000.0, 3000.0, 6000.0};
    const std::vector<double> u{5.0, 8.0, 14.0, 25.0};
    const std::vector<double> v{-3.0, 0.0, 5.0, 12.0};
    std::vector<double>       translated_u = u;
    std::vector<double>       translated_v = v;
    for (std::size_t k = 0; k < z.size(); ++k)
    {
        translated_u[k] += 31.0;
        translated_v[k] -= 9.0;
    }

    const auto base = estimate_bunkers_storm_motion(z, u, v, StormMotionMethod::BunkersRight);
    const auto translated =
        estimate_bunkers_storm_motion(z, translated_u, translated_v, StormMotionMethod::BunkersRight);
    REQUIRE(translated.velocity_ms.u_ms == Approx(base.velocity_ms.u_ms + 31.0));
    REQUIRE(translated.velocity_ms.v_ms == Approx(base.velocity_ms.v_ms - 9.0));
}

TEST_CASE("Bunkers estimator fails closed on insufficient depth and calm shear", "[core][domain_motion][storm_motion]")
{
    REQUIRE_THROWS_WITH(
        estimate_bunkers_storm_motion({0.0, 5000.0}, {0.0, 10.0}, {0.0, 0.0}, StormMotionMethod::BunkersRight),
        Catch::Matchers::Contains("reaching 6000 m"));
    REQUIRE_THROWS_WITH(
        estimate_bunkers_storm_motion({0.0, 6000.0}, {10.0, 10.0}, {5.0, 5.0}, StormMotionMethod::BunkersRight),
        Catch::Matchers::Contains("negligible"));
}

TEST_CASE("lifetime planner catches the maintained 38.4 km storm exit", "[core][domain_motion][planner]")
{
    StormLifetimePlanInput input;
    input.nx                             = 64;
    input.ny                             = 64;
    input.dx_m                           = 600.0;
    input.dy_m                           = 600.0;
    input.run_time_s                     = 1800.0;
    input.initial_storm_center_m         = {19200.0, 19200.0};
    input.earth_relative_storm_motion_ms = {15.0, 0.0};

    const StormLifetimePlan plan = plan_storm_lifetime(input);
    REQUIRE_FALSE(plan.contained_for_run);
    REQUIRE(plan.projected_final_center_m.x_m == Approx(46200.0));
    REQUIRE(plan.first_risk_face == BoundaryFace::East);
    REQUIRE(plan.first_risk_time_s == Approx(1280.0));
    REQUIRE(plan.recommended_initial_center_m.x_m == Approx(5700.0));
}

TEST_CASE("matching constant translation contains the buffered storm corridor",
          "[core][domain_motion][planner][translation]")
{
    StormLifetimePlanInput input;
    input.nx                             = 64;
    input.ny                             = 64;
    input.dx_m                           = 600.0;
    input.dy_m                           = 600.0;
    input.run_time_s                     = 7200.0;
    input.initial_storm_center_m         = {19200.0, 19200.0};
    input.earth_relative_storm_motion_ms = {15.0, 0.0};
    input.frame.velocity_ms              = {15.0, 0.0};
    input.upstream_buffer_m              = 5000.0;
    input.downstream_buffer_m            = 8000.0;
    input.cross_track_buffer_m           = 4000.0;

    const StormLifetimePlan plan = plan_storm_lifetime(input);
    REQUIRE(plan.grid_relative_storm_motion_ms.u_ms == Approx(0.0));
    REQUIRE(plan.projected_final_center_m.x_m == Approx(19200.0));
    REQUIRE(plan.required_bounds_m.west_m == Approx(14200.0));
    REQUIRE(plan.required_bounds_m.east_m == Approx(27200.0));
    REQUIRE(plan.required_bounds_m.south_m == Approx(15200.0));
    REQUIRE(plan.required_bounds_m.north_m == Approx(23200.0));
    REQUIRE(plan.contained_for_run);
    REQUIRE(plan.first_risk_face == BoundaryFace::None);
    REQUIRE(std::isinf(plan.first_risk_time_s));
}

TEST_CASE("motion uncertainty expands both sides of the lifetime corridor",
          "[core][domain_motion][planner][uncertainty]")
{
    StormLifetimePlanInput input;
    input.nx                     = 10;
    input.ny                     = 10;
    input.dx_m                   = 1000.0;
    input.dy_m                   = 1000.0;
    input.run_time_s             = 3000.0;
    input.initial_storm_center_m = {5000.0, 5000.0};
    input.motion_uncertainty_ms  = 2.0;

    const StormLifetimePlan plan = plan_storm_lifetime(input);
    REQUIRE(plan.required_bounds_m.west_m == Approx(-1000.0));
    REQUIRE(plan.required_bounds_m.east_m == Approx(11000.0));
    REQUIRE(plan.required_bounds_m.south_m == Approx(-1000.0));
    REQUIRE(plan.required_bounds_m.north_m == Approx(11000.0));
    REQUIRE_FALSE(plan.contained_for_run);
    REQUIRE(plan.first_risk_time_s == Approx(2500.0));
}

TEST_CASE("lifetime planner rejects invalid geometry before arithmetic", "[core][domain_motion][planner]")
{
    StormLifetimePlanInput input;
    input.nx   = 64;
    input.ny   = 64;
    input.dx_m = 0.0;
    input.dy_m = 600.0;

    REQUIRE_THROWS_WITH(plan_storm_lifetime(input), Catch::Matchers::Contains("positive dx and dy"));
}

namespace
{

StormWindowMovementPolicy movement_policy()
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m       = 500.0;
    policy.release_margin_m     = 800.0;
    policy.dx_m                 = 100.0;
    policy.dy_m                 = 200.0;
    policy.maximum_shift_cells  = 4;
    policy.cooldown_s           = 10.0;
    policy.minimum_confidence   = 0.8;
    policy.confirmation_samples = 2;
    return policy;
}

StormWindowMovementEvidence qualified_movement_evidence(double sample_time_s)
{
    StormWindowMovementEvidence evidence;
    evidence.sample_time_s              = sample_time_s;
    evidence.selected_feature_available = true;
    evidence.continuity_authoritative   = true;
    evidence.motion_available           = true;
    evidence.confidence                 = 0.95;
    evidence.distance_west_m            = 2000.0;
    evidence.distance_east_m            = 2000.0;
    evidence.distance_south_m           = 2000.0;
    evidence.distance_north_m           = 2000.0;
    return evidence;
}

} // namespace

TEST_CASE("storm-window release geometry owns emergency shift capacity",
          "[core][domain_motion][movement_authority][containment]")
{
    REQUIRE(storm_window_release_margin_m(15600.0, 1200.0, 1200.0) ==
            Approx(16800.0));
    REQUIRE(minimum_storm_window_emergency_shift_cells(
                15600.0, 1200.0, 1200.0) == 14);

    REQUIRE(storm_window_release_margin_m(500.0, 100.0, 200.0) ==
            Approx(700.0));
    REQUIRE(minimum_storm_window_emergency_shift_cells(
                500.0, 100.0, 200.0) == 7);

    REQUIRE_THROWS_WITH(
        minimum_storm_window_emergency_shift_cells(500.0, 0.0, 200.0),
        Catch::Matchers::Contains("positive dx and dy"));
    REQUIRE_THROWS_WITH(
        minimum_storm_window_emergency_shift_cells(
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(), 1.0),
        Catch::Matchers::Contains("overflow"));
}

TEST_CASE("storm-window emergency capacity reaches the release box",
          "[core][domain_motion][movement_authority][containment]")
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.release_margin_m = storm_window_release_margin_m(
        policy.inner_margin_m, policy.dx_m, policy.dy_m);
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;

    auto evidence = qualified_movement_evidence(30.0);
    evidence.distance_west_m = 33001.0;
    evidence.distance_east_m = 599.0;
    evidence.distance_south_m = 33001.0;
    evidence.distance_north_m = 599.0;
    evidence.velocity_x_ms = 10.0;
    evidence.velocity_y_ms = 10.0;

    StormWindowMovementState state;
    const auto decision = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(decision.status == StormWindowMovementStatus::Requested);
    REQUIRE(decision.reason == StormWindowMovementReason::EmergencyContainment);
    REQUIRE(decision.requested_i_shift_cells == 14);
    REQUIRE(decision.requested_j_shift_cells == 14);

    evidence.distance_west_m -= decision.requested_i_shift_cells * policy.dx_m;
    evidence.distance_east_m += decision.requested_i_shift_cells * policy.dx_m;
    evidence.distance_south_m -= decision.requested_j_shift_cells * policy.dy_m;
    evidence.distance_north_m += decision.requested_j_shift_cells * policy.dy_m;
    REQUIRE(storm_window_release_margin_recovered(evidence, policy));

    policy.maximum_shift_cells = 13;
    state = {};
    evidence.distance_west_m = 33001.0;
    evidence.distance_east_m = 599.0;
    evidence.distance_south_m = 33001.0;
    evidence.distance_north_m = 599.0;
    const auto insufficient = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(insufficient.status == StormWindowMovementStatus::Ineligible);
    REQUIRE(insufficient.reason ==
            StormWindowMovementReason::ShiftLimitInsufficient);
}

TEST_CASE("storm-window movement requires persistent motion toward a breached envelope face",
          "[core][domain_motion][movement_authority]")
{
    StormWindowMovementState state;
    const auto               policy = movement_policy();

    auto first               = qualified_movement_evidence(60.0);
    first.distance_east_m    = 450.0;
    first.velocity_x_ms      = 12.0;
    const auto pending       = evaluate_storm_window_movement(state, first, policy);
    REQUIRE(pending.status == StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(pending.reason == StormWindowMovementReason::ConfirmationPending);
    REQUIRE(pending.requested_i_shift_cells == 4);
    REQUIRE(pending.requested_j_shift_cells == 0);
    REQUIRE(pending.confirmation_count == 1);

    auto second          = first;
    second.sample_time_s = 120.0;
    // Sub-cell envelope motion does not reset compatible direction evidence.
    second.distance_east_m = 410.0;
    const auto requested = evaluate_storm_window_movement(state, second, policy);
    REQUIRE(requested.status == StormWindowMovementStatus::Requested);
    REQUIRE(requested.reason == StormWindowMovementReason::RequestReady);
    REQUIRE(requested.requested_i_shift_cells == 4);
    REQUIRE(requested.confirmation_count == 2);
    REQUIRE(std::string(storm_window_movement_algorithm_id()) == "qualified_envelope_hysteresis_v14");
}

TEST_CASE("storm-window recognizes confirmation reset when an orthogonal request appears",
          "[core][domain_motion][movement_authority][confirmation]")
{
    auto policy = movement_policy();
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.confirmation_samples = 2;
    policy.relaxation_margin_x_m = 4800.0;
    policy.relaxation_margin_y_m = 4800.0;

    StormWindowMovementState state;
    auto first = qualified_movement_evidence(4260.0);
    first.distance_west_m = 16200.0;
    first.distance_east_m = 12600.0;
    first.distance_south_m = 19800.0;
    first.distance_north_m = 16800.0;
    first.velocity_x_ms = 19.281478239331772;
    first.velocity_y_ms = 13.330028432648541;
    first.envelope_face_motion_available = true;
    first.envelope_face_motion_interval_s = 30.0;
    first.envelope_west_coordinate_velocity_ms = 20.0;
    first.envelope_east_coordinate_velocity_ms = 60.0;
    first.envelope_south_coordinate_velocity_ms = 0.0;
    first.envelope_north_coordinate_velocity_ms = 20.0;
    first.legal_shift_bounds_available = true;
    first.minimum_i_shift_cells = -51;
    first.maximum_i_shift_cells = 2;
    first.minimum_j_shift_cells = -19;
    first.maximum_j_shift_cells = 2;

    const auto first_pending = evaluate_storm_window_movement(
        state, first, policy);
    REQUIRE(first_pending.status ==
            StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(first_pending.requested_i_shift_cells == 1);
    REQUIRE(first_pending.requested_j_shift_cells == 0);
    REQUIRE(storm_window_started_new_confirmation({}, first_pending));

    const StormWindowMovementState state_before_second = state;
    auto second = first;
    second.sample_time_s = 4290.0;
    second.distance_west_m = 16800.0;
    second.distance_east_m = 12000.0;
    second.distance_south_m = 20400.0;
    second.distance_north_m = 16200.0;
    second.velocity_x_ms = 19.65121194357426;
    second.velocity_y_ms = 13.34103941909896;
    second.envelope_west_coordinate_velocity_ms = 20.0;
    second.envelope_east_coordinate_velocity_ms = 20.0;
    second.envelope_south_coordinate_velocity_ms = 20.0;
    second.envelope_north_coordinate_velocity_ms = 20.0;

    const auto reset_pending = evaluate_storm_window_movement(
        state, second, policy);
    REQUIRE(reset_pending.status ==
            StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(reset_pending.requested_i_shift_cells == 2);
    REQUIRE(reset_pending.requested_j_shift_cells == 2);
    REQUIRE(reset_pending.confirmation_count == 1);
    REQUIRE(state_before_second.consecutive_compatible_samples == 1);
    REQUIRE(storm_window_started_new_confirmation(
        state_before_second, reset_pending));

    auto malformed_same_direction = reset_pending;
    malformed_same_direction.requested_j_shift_cells = 0;
    REQUIRE_FALSE(storm_window_started_new_confirmation(
        state_before_second, malformed_same_direction));
}

TEST_CASE("storm-window keeps a receding inner-margin face trusted without moving",
          "[core][domain_motion][movement_authority][containment]")
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;
    policy.relaxation_margin_x_m = 4800.0;
    policy.relaxation_margin_y_m = 4800.0;

    auto evidence = qualified_movement_evidence(3900.0);
    evidence.distance_west_m = 13800.0;
    evidence.distance_east_m = 19800.0;
    evidence.distance_south_m = 17400.0;
    evidence.distance_north_m = 23400.0;
    evidence.velocity_x_ms = 18.555018395603497;
    evidence.velocity_y_ms = 14.47405701452546;
    evidence.envelope_face_motion_available = true;
    evidence.envelope_face_motion_interval_s = 30.0;
    evidence.envelope_west_coordinate_velocity_ms = 20.0;
    evidence.envelope_east_coordinate_velocity_ms = -240.0;
    evidence.envelope_south_coordinate_velocity_ms = 0.0;
    evidence.envelope_north_coordinate_velocity_ms = 320.0;

    SECTION("the captured v8 face is trusted and does not mutate")
    {
        StormWindowMovementState state;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::TrustedInteriorContained);
        REQUIRE(decision.requested_i_shift_cells == 0);
        REQUIRE(decision.requested_j_shift_cells == 0);
    }

    SECTION("an approaching breached face still enters relocation")
    {
        StormWindowMovementState state;
        evidence.envelope_west_coordinate_velocity_ms = -20.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.reason !=
                StormWindowMovementReason::TrustedInteriorContained);
        REQUIRE(decision.requested_i_shift_cells < 0);
    }

    SECTION("a current relaxation entry is never admitted as trusted")
    {
        StormWindowMovementState state;
        evidence.distance_west_m = 4799.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::NoApproachingBreach);
    }
}

TEST_CASE("storm-window keeps a quantized balanced approaching face trusted without moving",
          "[core][domain_motion][movement_authority][containment][quantization]")
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;
    policy.relaxation_margin_x_m = 4800.0;
    policy.relaxation_margin_y_m = 4800.0;

    auto evidence = qualified_movement_evidence(4110.0);
    evidence.distance_west_m = 16200.0;
    evidence.distance_east_m = 15000.0;
    evidence.distance_south_m = 19200.0;
    evidence.distance_north_m = 19800.0;
    evidence.velocity_x_ms = 18.26752321682104;
    evidence.velocity_y_ms = 13.724780787969454;
    evidence.envelope_face_motion_available = true;
    evidence.envelope_face_motion_interval_s = 30.0;
    evidence.envelope_west_coordinate_velocity_ms = 20.0;
    evidence.envelope_east_coordinate_velocity_ms = 20.0;
    evidence.envelope_south_coordinate_velocity_ms = 20.0;
    evidence.envelope_north_coordinate_velocity_ms = 20.0;

    SECTION("the captured v9 tie is trusted and does not mutate")
    {
        StormWindowMovementState state;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::TrustedInteriorContained);
        REQUIRE(decision.requested_i_shift_cells == 0);
        REQUIRE(decision.requested_j_shift_cells == 0);
        REQUIRE(state.consecutive_compatible_samples == 0);
    }

    SECTION("one legal cell of strict improvement retains relocation")
    {
        StormWindowMovementState state;
        evidence.distance_west_m = 17400.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::ConfirmationPending);
        REQUIRE(decision.requested_i_shift_cells == 1);
        REQUIRE(decision.requested_j_shift_cells == 0);
    }

    SECTION("a current relaxation entry is never a quantized trusted state")
    {
        StormWindowMovementState state;
        evidence.distance_east_m = 4799.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.reason !=
                StormWindowMovementReason::TrustedInteriorContained);
    }

    SECTION("the mirror-image west-face tie is equally representable")
    {
        StormWindowMovementState state;
        std::swap(evidence.distance_west_m, evidence.distance_east_m);
        evidence.velocity_x_ms = -18.26752321682104;
        evidence.envelope_west_coordinate_velocity_ms = -20.0;
        evidence.envelope_east_coordinate_velocity_ms = -20.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::TrustedInteriorContained);
    }
}

TEST_CASE("storm-window confirms an ordinary trusted improvement when release is unattainable",
          "[core][domain_motion][movement_authority][containment][quantization]")
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;
    policy.relaxation_margin_x_m = 4800.0;
    policy.relaxation_margin_y_m = 4800.0;

    auto evidence = qualified_movement_evidence(4140.0);
    evidence.distance_west_m = 16200.0;
    evidence.distance_east_m = 13800.0;
    evidence.distance_south_m = 19200.0;
    evidence.distance_north_m = 19200.0;
    evidence.velocity_x_ms = 18.263856395226902;
    evidence.velocity_y_ms = 13.743802762445267;
    evidence.envelope_face_motion_available = true;
    evidence.envelope_face_motion_interval_s = 30.0;
    evidence.envelope_west_coordinate_velocity_ms = 0.0;
    evidence.envelope_east_coordinate_velocity_ms = 40.0;
    evidence.envelope_south_coordinate_velocity_ms = 0.0;
    evidence.envelope_north_coordinate_velocity_ms = 20.0;

    StormWindowMovementState state;
    const auto pending = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(pending.status ==
            StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(pending.reason == StormWindowMovementReason::ConfirmationPending);
    REQUIRE(pending.requested_i_shift_cells == 1);
    REQUIRE(pending.requested_j_shift_cells == 0);
    REQUIRE(pending.confirmation_count == 1);

    evidence.sample_time_s = 4170.0;
    evidence.distance_west_m = 16800.0;
    evidence.distance_east_m = 13200.0;
    const auto requested = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(requested.status == StormWindowMovementStatus::Requested);
    REQUIRE(requested.reason ==
            StormWindowMovementReason::TrustedInteriorContainment);
    REQUIRE(requested.requested_i_shift_cells == 1);
    REQUIRE(requested.requested_j_shift_cells == 0);
    REQUIRE(requested.confirmation_count == 2);

    const double shifted_west_m = evidence.distance_west_m -
        requested.requested_i_shift_cells * policy.dx_m;
    const double shifted_east_m = evidence.distance_east_m +
        requested.requested_i_shift_cells * policy.dx_m;
    REQUIRE(shifted_west_m == 15600.0);
    REQUIRE(shifted_east_m == 14400.0);
    REQUIRE(std::min(shifted_west_m, shifted_east_m) >
            std::min(evidence.distance_west_m, evidence.distance_east_m));
    REQUIRE_FALSE(storm_window_release_margin_recovered(evidence, policy));

    SECTION("current relaxation entry remains inadmissible")
    {
        StormWindowMovementState relaxation_state;
        evidence.sample_time_s = 4140.0;
        evidence.distance_west_m = 16200.0;
        evidence.distance_east_m = 4799.0;
        const auto rejected = evaluate_storm_window_movement(
            relaxation_state, evidence, policy);
        REQUIRE(rejected.reason !=
                StormWindowMovementReason::TrustedInteriorContainment);
    }
}

TEST_CASE("storm-window honors exact parent placement bounds at the 4380-second frontier",
          "[core][domain_motion][movement_authority][containment][parent_stencil]")
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;
    policy.relaxation_margin_x_m = 4800.0;
    policy.relaxation_margin_y_m = 4800.0;

    auto evidence = qualified_movement_evidence(4350.0);
    evidence.distance_west_m = 15600.0;
    evidence.distance_east_m = 12600.0;
    evidence.distance_south_m = 18600.0;
    evidence.distance_north_m = 17400.0;
    evidence.velocity_x_ms = 18.0;
    evidence.velocity_y_ms = 14.0;
    evidence.envelope_face_motion_available = true;
    evidence.envelope_face_motion_interval_s = 30.0;
    evidence.envelope_west_coordinate_velocity_ms = 20.0;
    evidence.envelope_east_coordinate_velocity_ms = 40.0;
    evidence.envelope_south_coordinate_velocity_ms = 20.0;
    evidence.envelope_north_coordinate_velocity_ms = 20.0;
    evidence.legal_shift_bounds_available = true;
    evidence.minimum_i_shift_cells = -53;
    evidence.maximum_i_shift_cells = 0;
    evidence.minimum_j_shift_cells = -21;
    evidence.maximum_j_shift_cells = 0;

    StormWindowMovementState state;
    const auto at_limit = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(at_limit.status == StormWindowMovementStatus::Ineligible);
    REQUIRE(at_limit.reason ==
            StormWindowMovementReason::ParentStencilLimitedContained);
    REQUIRE(at_limit.requested_i_shift_cells == 0);
    REQUIRE(at_limit.requested_j_shift_cells == 0);
    REQUIRE(state.consecutive_compatible_samples == 0);

    SECTION("the same face inside relaxation is never trusted")
    {
        StormWindowMovementState relaxation_state;
        evidence.distance_east_m = 4799.0;
        const auto rejected = evaluate_storm_window_movement(
            relaxation_state, evidence, policy);
        REQUIRE(rejected.reason !=
                StormWindowMovementReason::ParentStencilLimitedContained);
        REQUIRE(rejected.reason !=
                StormWindowMovementReason::TrustedInteriorContained);
    }

    SECTION("a remaining legal cell is used and never exceeded")
    {
        StormWindowMovementState limited_state;
        evidence.maximum_i_shift_cells = 1;
        evidence.distance_west_m = 18000.0;
        evidence.distance_east_m = 12600.0;
        const auto pending = evaluate_storm_window_movement(
            limited_state, evidence, policy);
        REQUIRE(pending.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(pending.requested_i_shift_cells == 1);
        evidence.sample_time_s = 4380.0;
        const auto requested = evaluate_storm_window_movement(
            limited_state, evidence, policy);
        REQUIRE(requested.status == StormWindowMovementStatus::Requested);
        REQUIRE(requested.reason ==
                StormWindowMovementReason::TrustedInteriorContainment);
        REQUIRE(requested.requested_i_shift_cells == 1);
    }

    SECTION("malformed placement bounds fail closed")
    {
        StormWindowMovementState malformed_state;
        evidence.minimum_i_shift_cells = 1;
        REQUIRE_THROWS_WITH(
            evaluate_storm_window_movement(
                malformed_state, evidence, policy),
            Catch::Matchers::Contains("must contain zero"));
    }
}

TEST_CASE("storm-window movement commits an ordinary projected-relaxation request before a trusted face is lost",
          "[core][domain_motion][movement_authority][projected_relaxation]")
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;
    policy.relaxation_margin_x_m = 4800.0;
    policy.relaxation_margin_y_m = 4800.0;

    auto evidence = qualified_movement_evidence(3510.0);
    evidence.distance_west_m = 34200.0;
    evidence.distance_east_m = 11400.0;
    evidence.distance_south_m = 37800.0;
    evidence.distance_north_m = 9600.0;
    evidence.velocity_x_ms = 18.861131315995493;
    evidence.velocity_y_ms = 13.820786516951411;
    evidence.envelope_face_motion_available = true;
    evidence.envelope_face_motion_interval_s = 30.0;
    evidence.envelope_west_coordinate_velocity_ms = 20.0;
    evidence.envelope_east_coordinate_velocity_ms = 440.0;
    evidence.envelope_south_coordinate_velocity_ms = 0.0;
    evidence.envelope_north_coordinate_velocity_ms = 320.0;

    StormWindowMovementState state;
    state.sample_time_available = true;
    state.last_sample_time_s = 3480.0;
    state.committed_move_available = true;
    state.last_committed_time_s = 3180.0;
    const auto decision = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(decision.status == StormWindowMovementStatus::Requested);
    REQUIRE(decision.reason ==
            StormWindowMovementReason::ProjectedRelaxationRisk);
    REQUIRE(std::string(storm_window_movement_reason_name(decision.reason)) ==
            "projected_relaxation_risk");
    REQUIRE(decision.confirmation_count == 1);
    REQUIRE(decision.requested_i_shift_cells == 14);
    REQUIRE(decision.requested_j_shift_cells == 14);

    evidence.distance_west_m -= decision.requested_i_shift_cells * policy.dx_m;
    evidence.distance_east_m += decision.requested_i_shift_cells * policy.dx_m;
    evidence.distance_south_m -= decision.requested_j_shift_cells * policy.dy_m;
    evidence.distance_north_m += decision.requested_j_shift_cells * policy.dy_m;
    REQUIRE(evidence.distance_west_m >= policy.release_margin_m);
    REQUIRE(evidence.distance_east_m >= policy.release_margin_m);
    REQUIRE(evidence.distance_south_m >= policy.release_margin_m);
    REQUIRE(evidence.distance_north_m >= policy.release_margin_m);

    // The invalid v4 campaign accelerated beyond the 3510-second face-rate
    // projection. The maximum safe v7 shift keeps its observed 3540-second
    // envelope outside the child relaxation zone.
    evidence.distance_west_m += 600.0;
    evidence.distance_east_m -= 19200.0;
    evidence.distance_south_m += 600.0;
    evidence.distance_north_m -= 3000.0;
    REQUIRE(evidence.distance_west_m >= policy.relaxation_margin_x_m);
    REQUIRE(evidence.distance_east_m >= policy.relaxation_margin_x_m);
    REQUIRE(evidence.distance_south_m >= policy.relaxation_margin_y_m);
    REQUIRE(evidence.distance_north_m >= policy.relaxation_margin_y_m);

    SECTION("the observed v5 envelope receives a trusted-interior balancing move")
    {
        commit_storm_window_movement(state, decision);
        state.armed = true;
        evidence.sample_time_s = 3540.0;
        evidence.envelope_west_coordinate_velocity_ms = 20.0;
        evidence.envelope_east_coordinate_velocity_ms = 640.0;
        evidence.envelope_south_coordinate_velocity_ms = 20.0;
        evidence.envelope_north_coordinate_velocity_ms = 100.0;
        const auto containment = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(containment.status == StormWindowMovementStatus::Requested);
        REQUIRE(containment.reason ==
                StormWindowMovementReason::TrustedInteriorContainment);
        REQUIRE(std::string(storm_window_movement_reason_name(
                    containment.reason)) == "trusted_interior_containment");
        REQUIRE(containment.confirmation_count == 1);
        REQUIRE(containment.requested_i_shift_cells == 4);
        REQUIRE(containment.requested_j_shift_cells == 4);

        evidence.distance_west_m -=
            containment.requested_i_shift_cells * policy.dx_m;
        evidence.distance_east_m +=
            containment.requested_i_shift_cells * policy.dx_m;
        evidence.distance_south_m -=
            containment.requested_j_shift_cells * policy.dy_m;
        evidence.distance_north_m +=
            containment.requested_j_shift_cells * policy.dy_m;
        REQUIRE(evidence.distance_west_m == Approx(13200.0));
        REQUIRE(evidence.distance_east_m == Approx(13800.0));
        REQUIRE(evidence.distance_south_m == Approx(16800.0));
        REQUIRE(evidence.distance_north_m == Approx(28200.0));
        REQUIRE_FALSE(storm_window_release_margin_recovered(
            evidence, policy));
        REQUIRE(evidence.distance_west_m >= policy.relaxation_margin_x_m);
        REQUIRE(evidence.distance_east_m >= policy.relaxation_margin_x_m);

        SECTION("the next balanced wide envelope remains trusted without mutation")
        {
            commit_storm_window_movement(state, containment);
            state.armed = true;
            evidence.sample_time_s = 3570.0;
            evidence.distance_east_m = 13200.0;
            evidence.distance_south_m = 16800.0;
            evidence.distance_north_m = 30000.0;
            evidence.envelope_west_coordinate_velocity_ms = 0.0;
            evidence.envelope_east_coordinate_velocity_ms = 20.0;
            evidence.envelope_south_coordinate_velocity_ms = 0.0;
            evidence.envelope_north_coordinate_velocity_ms = -60.0;
            const auto contained = evaluate_storm_window_movement(
                state, evidence, policy);
            REQUIRE(contained.status ==
                    StormWindowMovementStatus::Ineligible);
            REQUIRE(contained.reason ==
                    StormWindowMovementReason::TrustedInteriorContained);
            REQUIRE(std::string(storm_window_movement_reason_name(
                        contained.reason)) == "trusted_interior_contained");
            REQUIRE(contained.requested_i_shift_cells == 0);
            REQUIRE(contained.requested_j_shift_cells == 0);
            REQUIRE(state.consecutive_compatible_samples == 0);

            evidence.sample_time_s = 3690.0;
            evidence.distance_west_m = 15000.0;
            evidence.distance_east_m = 10200.0;
            evidence.distance_south_m = 17400.0;
            evidence.distance_north_m = 30000.0;
            evidence.envelope_west_coordinate_velocity_ms = 20.0;
            evidence.envelope_east_coordinate_velocity_ms = 60.0;
            evidence.envelope_south_coordinate_velocity_ms = 0.0;
            evidence.envelope_north_coordinate_velocity_ms = -160.0;
            const auto before_expansion = evaluate_storm_window_movement(
                state, evidence, policy);
            REQUIRE(before_expansion.status ==
                    StormWindowMovementStatus::Ineligible);
            REQUIRE(before_expansion.reason ==
                    StormWindowMovementReason::TrustedInteriorContained);

            // The failed v7 campaign grew the selected north envelope by
            // 24 km in one cadence. The orthogonal reserve leaves the current
            // envelope trusted, so a new containment transaction is still
            // scientifically admissible rather than attempting to rescue a
            // sample that has already entered relaxation.
            evidence.sample_time_s = 3720.0;
            evidence.distance_north_m = 6000.0;
            evidence.envelope_west_coordinate_velocity_ms = 0.0;
            evidence.envelope_east_coordinate_velocity_ms = 0.0;
            evidence.envelope_north_coordinate_velocity_ms = 800.0;
            REQUIRE(evidence.distance_north_m >=
                    policy.relaxation_margin_y_m);
            const auto expansion = evaluate_storm_window_movement(
                state, evidence, policy);
            REQUIRE(expansion.status ==
                    StormWindowMovementStatus::Requested);
            REQUIRE(expansion.reason ==
                    StormWindowMovementReason::TrustedInteriorContainment);
            REQUIRE(expansion.requested_i_shift_cells == 5);
            REQUIRE(expansion.requested_j_shift_cells == 5);
        }
    }

    SECTION("unavailable face motion preserves ordinary confirmation")
    {
        StormWindowMovementState unavailable_state;
        auto unavailable = qualified_movement_evidence(3510.0);
        unavailable.distance_west_m = 34200.0;
        unavailable.distance_east_m = 11400.0;
        unavailable.distance_south_m = 37800.0;
        unavailable.distance_north_m = 9600.0;
        unavailable.velocity_x_ms = 18.0;
        unavailable.velocity_y_ms = 13.0;
        const auto pending = evaluate_storm_window_movement(
            unavailable_state, unavailable, policy);
        REQUIRE(pending.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(pending.reason ==
                StormWindowMovementReason::ConfirmationPending);
    }

    SECTION("projection must cross strictly below the relaxation margin")
    {
        StormWindowMovementState threshold_state;
        auto threshold = qualified_movement_evidence(3510.0);
        threshold.distance_west_m = 34200.0;
        threshold.distance_east_m = 11400.0;
        threshold.distance_south_m = 37800.0;
        threshold.distance_north_m = 9600.0;
        threshold.velocity_x_ms = 18.0;
        threshold.envelope_face_motion_available = true;
        threshold.envelope_face_motion_interval_s = 30.0;
        threshold.envelope_east_coordinate_velocity_ms = 220.0;
        const auto pending = evaluate_storm_window_movement(
            threshold_state, threshold, policy);
        REQUIRE(pending.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(pending.reason ==
                StormWindowMovementReason::ConfirmationPending);
    }

    SECTION("a face already in relaxation is never rescued by projection")
    {
        StormWindowMovementState entered_state;
        auto entered = qualified_movement_evidence(3540.0);
        entered.distance_west_m = 39000.0;
        entered.distance_east_m = 2400.0;
        entered.distance_south_m = 37800.0;
        entered.distance_north_m = 9600.0;
        entered.velocity_x_ms = 18.0;
        entered.envelope_face_motion_available = true;
        entered.envelope_face_motion_interval_s = 30.0;
        entered.envelope_east_coordinate_velocity_ms = 100.0;
        const auto pending = evaluate_storm_window_movement(
            entered_state, entered, policy);
        REQUIRE(pending.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(pending.reason ==
                StormWindowMovementReason::ConfirmationPending);
    }

    SECTION("projected admission uses trusted containment when release is unattainable")
    {
        auto narrow_policy = policy;
        narrow_policy.inner_margin_m = 500.0;
        narrow_policy.release_margin_m = 800.0;
        narrow_policy.relaxation_margin_x_m = 100.0;
        narrow_policy.relaxation_margin_y_m = 100.0;
        narrow_policy.dx_m = 100.0;
        narrow_policy.dy_m = 200.0;
        narrow_policy.maximum_shift_cells = 4;

        StormWindowMovementState narrow_state;
        auto narrow = qualified_movement_evidence(60.0);
        narrow.distance_west_m = 900.0;
        narrow.distance_east_m = 450.0;
        narrow.distance_south_m = 2000.0;
        narrow.distance_north_m = 2000.0;
        narrow.velocity_x_ms = 10.0;
        narrow.envelope_face_motion_available = true;
        narrow.envelope_face_motion_interval_s = 60.0;
        narrow.envelope_east_coordinate_velocity_ms = 10.0;
        const auto pending = evaluate_storm_window_movement(
            narrow_state, narrow, narrow_policy);
        REQUIRE(pending.status == StormWindowMovementStatus::Requested);
        REQUIRE(pending.reason ==
                StormWindowMovementReason::TrustedInteriorContainment);
        REQUIRE(pending.requested_i_shift_cells == 2);
        REQUIRE(narrow_state.consecutive_compatible_samples == 1);
    }

    SECTION("projected margins are a complete paired policy")
    {
        auto invalid_policy = policy;
        invalid_policy.relaxation_margin_y_m = 0.0;
        StormWindowMovementState invalid_state;
        REQUIRE_THROWS_AS(evaluate_storm_window_movement(
            invalid_state, qualified_movement_evidence(1.0),
            invalid_policy), std::invalid_argument);

        invalid_policy = policy;
        invalid_policy.relaxation_margin_x_m = policy.inner_margin_m;
        invalid_state = {};
        REQUIRE_THROWS_AS(evaluate_storm_window_movement(
            invalid_state, qualified_movement_evidence(1.0),
            invalid_policy), std::invalid_argument);
    }
}

TEST_CASE("storm-window movement admits an advancing envelope face independently of centroid motion",
          "[core][domain_motion][movement_authority][envelope_motion]")
{
    const auto policy = movement_policy();
    const auto require_face_request = [&](double west_velocity_ms,
                                          double east_velocity_ms,
                                          double south_velocity_ms,
                                          double north_velocity_ms,
                                          double west_distance_m,
                                          double east_distance_m,
                                          double south_distance_m,
                                          double north_distance_m,
                                          int expected_i,
                                          int expected_j) {
        StormWindowMovementState state;
        auto evidence = qualified_movement_evidence(60.0);
        evidence.distance_west_m = west_distance_m;
        evidence.distance_east_m = east_distance_m;
        evidence.distance_south_m = south_distance_m;
        evidence.distance_north_m = north_distance_m;
        // The centroid is stationary; only the selected envelope face advances.
        evidence.velocity_x_ms = 0.0;
        evidence.velocity_y_ms = 0.0;
        evidence.envelope_face_motion_available = true;
        evidence.envelope_face_motion_interval_s = 30.0;
        evidence.envelope_west_coordinate_velocity_ms = west_velocity_ms;
        evidence.envelope_east_coordinate_velocity_ms = east_velocity_ms;
        evidence.envelope_south_coordinate_velocity_ms = south_velocity_ms;
        evidence.envelope_north_coordinate_velocity_ms = north_velocity_ms;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::ConfirmationPending);
        REQUIRE(decision.requested_i_shift_cells == expected_i);
        REQUIRE(decision.requested_j_shift_cells == expected_j);
        REQUIRE(decision.confirmation_count == 1);
    };

    require_face_request(-3.0, 0.0, 0.0, 0.0,
                         450.0, 2000.0, 2000.0, 2000.0, -4, 0);
    require_face_request(0.0, 3.0, 0.0, 0.0,
                         2000.0, 450.0, 2000.0, 2000.0, 4, 0);
    require_face_request(0.0, 0.0, -3.0, 0.0,
                         2000.0, 2000.0, 450.0, 2000.0, 0, -4);
    require_face_request(0.0, 0.0, 0.0, 3.0,
                         2000.0, 2000.0, 2000.0, 450.0, 0, 4);
}

TEST_CASE("storm-window face-motion evidence is canonical and does not weaken existing motion",
          "[core][domain_motion][movement_authority][envelope_motion]")
{
    const auto policy = movement_policy();
    auto evidence = qualified_movement_evidence(1.0);
    evidence.distance_east_m = 450.0;
    evidence.velocity_x_ms = 12.0;
    evidence.envelope_face_motion_available = true;
    evidence.envelope_face_motion_interval_s = 30.0;
    // A contracting east face cannot veto the approaching centroid.
    evidence.envelope_east_coordinate_velocity_ms = -5.0;
    StormWindowMovementState state;
    REQUIRE(evaluate_storm_window_movement(state, evidence, policy).status ==
            StormWindowMovementStatus::PendingConfirmation);

    evidence.sample_time_s = 2.0;
    evidence.envelope_face_motion_available = false;
    REQUIRE_THROWS_AS(
        evaluate_storm_window_movement(state, evidence, policy),
        std::invalid_argument);

    evidence.envelope_face_motion_interval_s = 0.0;
    evidence.envelope_east_coordinate_velocity_ms = 0.0;
    evidence.envelope_face_motion_available = true;
    REQUIRE_THROWS_AS(
        evaluate_storm_window_movement(state, evidence, policy),
        std::invalid_argument);

    auto emergency_policy = policy;
    emergency_policy.maximum_shift_cells = 8;
    auto emergency = qualified_movement_evidence(1.0);
    emergency.distance_east_m = emergency_policy.dx_m - 1.0;
    emergency.envelope_face_motion_available = true;
    emergency.envelope_face_motion_interval_s = 30.0;
    emergency.envelope_east_coordinate_velocity_ms = 1.0;
    StormWindowMovementState emergency_state;
    const auto emergency_decision = evaluate_storm_window_movement(
        emergency_state, emergency, emergency_policy);
    REQUIRE(emergency_decision.status ==
            StormWindowMovementStatus::Requested);
    REQUIRE(emergency_decision.reason ==
            StormWindowMovementReason::EmergencyContainment);
    REQUIRE(emergency_decision.confirmation_count == 1);
}

TEST_CASE("storm-window movement restores an orthogonal release margin in the same transaction",
          "[core][domain_motion][movement_authority][hysteresis]")
{
    StormWindowMovementState state;
    auto policy = movement_policy();
    policy.dy_m = 100.0;

    auto first = qualified_movement_evidence(1.0);
    first.distance_west_m = 1200.0;
    first.distance_east_m = 300.0;
    first.distance_south_m = 800.0;
    first.distance_north_m = 800.0;
    first.velocity_x_ms = 15.0;
    first.velocity_y_ms = 2.0;
    const auto x_only = evaluate_storm_window_movement(state, first, policy);
    REQUIRE(x_only.status == StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(x_only.requested_i_shift_cells == 4);
    REQUIRE(x_only.requested_j_shift_cells == 0);

    auto second = first;
    second.sample_time_s = 2.0;
    second.distance_south_m = 900.0;
    second.distance_north_m = 700.0;
    const auto full_release_pending = evaluate_storm_window_movement(state, second, policy);
    REQUIRE(full_release_pending.status == StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(full_release_pending.requested_i_shift_cells == 4);
    REQUIRE(full_release_pending.requested_j_shift_cells == 1);
    REQUIRE(full_release_pending.confirmation_count == 1);

    second.sample_time_s = 3.0;
    const auto requested = evaluate_storm_window_movement(state, second, policy);
    REQUIRE(requested.status == StormWindowMovementStatus::Requested);
    REQUIRE(requested.requested_i_shift_cells == 4);
    REQUIRE(requested.requested_j_shift_cells == 1);

    second.distance_west_m -= requested.requested_i_shift_cells * policy.dx_m;
    second.distance_east_m += requested.requested_i_shift_cells * policy.dx_m;
    second.distance_south_m -= requested.requested_j_shift_cells * policy.dy_m;
    second.distance_north_m += requested.requested_j_shift_cells * policy.dy_m;
    REQUIRE(storm_window_release_margin_recovered(second, policy));
}

TEST_CASE("storm-window movement recenters a threatened axis within its bounded transaction",
          "[core][domain_motion][movement_authority][containment]")
{
    StormWindowMovementState state;
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 10000.0;
    policy.release_margin_m = 11200.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 9;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.8;
    policy.confirmation_samples = 2;

    auto evidence = qualified_movement_evidence(1.0);
    evidence.distance_west_m = 22800.0;
    evidence.distance_east_m = 7200.0;
    evidence.distance_south_m = 18600.0;
    evidence.distance_north_m = 13200.0;
    evidence.velocity_x_ms = 20.0;
    const auto pending = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(pending.status == StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(pending.requested_i_shift_cells == 6);

    evidence.sample_time_s = 2.0;
    const auto requested = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(requested.status == StormWindowMovementStatus::Requested);
    REQUIRE(requested.requested_i_shift_cells == 6);
    evidence.distance_west_m -= requested.requested_i_shift_cells * policy.dx_m;
    evidence.distance_east_m += requested.requested_i_shift_cells * policy.dx_m;
    REQUIRE(evidence.distance_west_m == 15600.0);
    REQUIRE(evidence.distance_east_m == 14400.0);
    REQUIRE(storm_window_release_margin_recovered(evidence, policy));
}

TEST_CASE("storm-window cooldown never suppresses an inner-margin breach",
          "[core][domain_motion][movement_authority][containment]")
{
    StormWindowMovementState state;
    const auto policy = movement_policy();
    auto evidence = qualified_movement_evidence(1.0);
    evidence.distance_east_m = 400.0;
    evidence.velocity_x_ms = 10.0;
    evaluate_storm_window_movement(state, evidence, policy);
    evidence.sample_time_s = 2.0;
    const auto requested = evaluate_storm_window_movement(state, evidence, policy);
    commit_storm_window_movement(state, requested);

    auto recovered = qualified_movement_evidence(3.0);
    recovered.velocity_x_ms = 10.0;
    REQUIRE(evaluate_storm_window_movement(state, recovered, policy).status ==
            StormWindowMovementStatus::Cooldown);
    REQUIRE(state.armed);

    recovered.sample_time_s = 4.0;
    recovered.distance_east_m = 400.0;
    const auto safety_request = evaluate_storm_window_movement(
        state, recovered, policy);
    REQUIRE(safety_request.status ==
            StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(safety_request.reason ==
            StormWindowMovementReason::ConfirmationPending);
}

TEST_CASE("storm-window authority immediately contains a sub-cell authoritative envelope",
          "[core][domain_motion][movement_authority][containment]")
{
    StormWindowMovementState state;
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 10000.0;
    policy.release_margin_m = 11200.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 9;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;

    auto evidence = qualified_movement_evidence(2640.0);
    evidence.distance_west_m = 21000.0;
    evidence.distance_east_m = 600.0;
    evidence.distance_south_m = 16200.0;
    evidence.distance_north_m = 8400.0;
    evidence.velocity_x_ms = 20.0;
    evidence.velocity_y_ms = 8.0;
    const auto emergency = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(emergency.status == StormWindowMovementStatus::Requested);
    REQUIRE(emergency.reason ==
            StormWindowMovementReason::EmergencyContainment);
    REQUIRE(emergency.confirmation_count == 1);
    REQUIRE(emergency.requested_i_shift_cells == 8);
    REQUIRE(emergency.requested_j_shift_cells == 3);

    commit_storm_window_movement(state, emergency);
    REQUIRE_FALSE(state.armed);
    evidence.distance_west_m -= emergency.requested_i_shift_cells * policy.dx_m;
    evidence.distance_east_m += emergency.requested_i_shift_cells * policy.dx_m;
    evidence.distance_south_m -= emergency.requested_j_shift_cells * policy.dy_m;
    evidence.distance_north_m += emergency.requested_j_shift_cells * policy.dy_m;
    REQUIRE(storm_window_release_margin_recovered(evidence, policy));
}

TEST_CASE("storm-window emergency threshold is exact per axis and remains fail-closed",
          "[core][domain_motion][movement_authority][containment]")
{
    auto policy = movement_policy();
    policy.release_margin_m = 600.0;
    policy.maximum_shift_cells = 10;

    SECTION("strictly below x spacing is immediate")
    {
        StormWindowMovementState state;
        auto evidence = qualified_movement_evidence(1.0);
        evidence.distance_east_m = policy.dx_m - 1.0;
        evidence.velocity_x_ms = 10.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Requested);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::EmergencyContainment);
        REQUIRE(decision.confirmation_count == 1);
    }

    SECTION("exactly x spacing retains ordinary confirmation")
    {
        StormWindowMovementState state;
        auto evidence = qualified_movement_evidence(1.0);
        evidence.distance_east_m = policy.dx_m;
        evidence.velocity_x_ms = 10.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::ConfirmationPending);
        REQUIRE(decision.confirmation_count == 1);
    }

    SECTION("y uses its own anisotropic spacing")
    {
        StormWindowMovementState state;
        auto evidence = qualified_movement_evidence(1.0);
        evidence.distance_south_m = policy.dy_m - 1.0;
        evidence.velocity_y_ms = -10.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Requested);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::EmergencyContainment);
    }

    SECTION("critical clearance cannot override missing authority")
    {
        StormWindowMovementState state;
        auto evidence = qualified_movement_evidence(1.0);
        evidence.distance_east_m = policy.dx_m - 1.0;
        evidence.velocity_x_ms = 10.0;
        evidence.continuity_authoritative = false;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::TrackerNotAuthoritative);
    }

    SECTION("critical clearance cannot override incomplete tracker evidence")
    {
        for (int failure = 0; failure < 4; ++failure)
        {
            StormWindowMovementState state;
            auto evidence = qualified_movement_evidence(1.0);
            evidence.distance_east_m = policy.dx_m - 1.0;
            evidence.velocity_x_ms = 10.0;
            StormWindowMovementReason expected =
                StormWindowMovementReason::NoSelectedFeature;
            if (failure == 0)
                evidence.selected_feature_available = false;
            else if (failure == 1)
            {
                evidence.motion_available = false;
                expected = StormWindowMovementReason::MotionUnavailable;
            }
            else if (failure == 2)
            {
                evidence.confidence = policy.minimum_confidence - 0.01;
                expected = StormWindowMovementReason::InsufficientConfidence;
            }
            else
            {
                evidence.ambiguous = true;
                expected = StormWindowMovementReason::AmbiguousTrack;
            }
            const auto decision = evaluate_storm_window_movement(
                state, evidence, policy);
            REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
            REQUIRE(decision.reason == expected);
        }
    }

    SECTION("critical clearance moving away is not actionable")
    {
        StormWindowMovementState state;
        auto evidence = qualified_movement_evidence(1.0);
        evidence.distance_east_m = policy.dx_m - 1.0;
        evidence.velocity_x_ms = -10.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::NoApproachingBreach);
    }

    SECTION("critical clearance cannot override an insufficient shift cap")
    {
        StormWindowMovementState state;
        auto constrained = policy;
        constrained.maximum_shift_cells = 1;
        auto evidence = qualified_movement_evidence(1.0);
        evidence.distance_east_m = -1000.0;
        evidence.velocity_x_ms = 10.0;
        const auto decision = evaluate_storm_window_movement(
            state, evidence, constrained);
        REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
        REQUIRE(decision.reason ==
                StormWindowMovementReason::ShiftLimitInsufficient);
    }
}

TEST_CASE("storm-window movement permits comfort-box corner requests",
          "[core][domain_motion][movement_authority]")
{
    StormWindowMovementState state;
    const auto               policy = movement_policy();

    auto evidence             = qualified_movement_evidence(1.0);
    evidence.distance_west_m  = 450.0;
    evidence.distance_north_m = 300.0;
    evidence.velocity_x_ms    = -20.0;
    evidence.velocity_y_ms    = 5.0;
    const auto pending = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(pending.requested_i_shift_cells == -4);
    REQUIRE(pending.requested_j_shift_cells == 4);

    evidence.sample_time_s = 2.0;
    const auto requested = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(requested.status == StormWindowMovementStatus::Requested);
    REQUIRE(requested.requested_i_shift_cells == -4);
    REQUIRE(requested.requested_j_shift_cells == 4);
}

TEST_CASE("storm-window movement rejects a request that its configured limit cannot recover",
          "[core][domain_motion][movement_authority]")
{
    StormWindowMovementState state;
    const auto               policy = movement_policy();
    auto                     evidence = qualified_movement_evidence(1.0);
    evidence.distance_west_m = -5000.0;
    evidence.velocity_x_ms   = -20.0;
    const auto decision = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
    REQUIRE(decision.reason == StormWindowMovementReason::ShiftLimitInsufficient);

    StormWindowMovementState narrow_state;
    evidence                   = qualified_movement_evidence(1.0);
    evidence.distance_west_m   = 550.0;
    evidence.distance_east_m   = 400.0;
    evidence.velocity_x_ms     = 20.0;
    const auto narrow_decision = evaluate_storm_window_movement(narrow_state, evidence, policy);
    REQUIRE(narrow_decision.status == StormWindowMovementStatus::Ineligible);
    REQUIRE(narrow_decision.reason == StormWindowMovementReason::OppositeReleaseMarginInsufficient);
}

TEST_CASE("storm-window movement uses the safest grid-attainable release box",
          "[core][domain_motion][movement_authority][quantization]")
{
    StormWindowMovementState state;
    auto policy = movement_policy();
    policy.dx_m = 200.0;

    auto evidence = qualified_movement_evidence(1.0);
    evidence.distance_west_m = 1200.0;
    evidence.distance_east_m = 300.0;
    evidence.velocity_x_ms = 15.0;
    const auto pending = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(pending.status == StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(pending.requested_i_shift_cells == 2);

    evidence.sample_time_s = 2.0;
    const auto requested = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(requested.status == StormWindowMovementStatus::Requested);
    REQUIRE(requested.requested_i_shift_cells == 2);

    evidence.distance_west_m -= 400.0;
    evidence.distance_east_m += 400.0;
    REQUIRE(storm_window_release_margin_recovered(evidence, policy));
}

TEST_CASE("storm-window movement chooses the nearest attainable integer center",
          "[core][domain_motion][movement_authority][quantization]")
{
    StormWindowMovementState state;
    auto policy = movement_policy();
    policy.release_margin_m = 600.0;
    policy.maximum_shift_cells = 5;

    auto evidence = qualified_movement_evidence(1.0);
    evidence.distance_west_m = 1140.0;
    evidence.distance_east_m = 400.0;
    evidence.velocity_x_ms = 15.0;
    const auto pending = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(pending.status == StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(pending.requested_i_shift_cells == 4);

    evidence.sample_time_s = 2.0;
    const auto requested = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(requested.status == StormWindowMovementStatus::Requested);
    REQUIRE(requested.requested_i_shift_cells == 4);

    const double shifted_west_m = evidence.distance_west_m -
        requested.requested_i_shift_cells * policy.dx_m;
    const double shifted_east_m = evidence.distance_east_m +
        requested.requested_i_shift_cells * policy.dx_m;
    REQUIRE(shifted_west_m == Approx(740.0));
    REQUIRE(shifted_east_m == Approx(800.0));
    REQUIRE(std::min(shifted_west_m, shifted_east_m) == Approx(740.0));
}

TEST_CASE("the historical 3090-second envelope is impossible on the old child lattice",
          "[core][domain_motion][movement_authority][quantization][regression]")
{
    constexpr double old_child_extent_x_m = 69'600.0;
    constexpr double old_child_extent_y_m = 67'200.0;
    constexpr double observed_envelope_span_x_m = 33'000.0;
    constexpr double observed_envelope_span_y_m = 36'000.0;

    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;

    auto evidence = qualified_movement_evidence(3090.0);
    evidence.confidence = 0.95614387860665828;
    evidence.distance_west_m = 22200.0;
    evidence.distance_east_m = 14400.0;
    evidence.distance_south_m = 16200.0;
    evidence.distance_north_m = 15000.0;
    evidence.velocity_x_ms = 19.542402521976328;
    evidence.velocity_y_ms = 14.031202496631158;
    evidence.envelope_face_motion_available = true;
    evidence.envelope_face_motion_interval_s = 30.0;
    evidence.envelope_west_coordinate_velocity_ms = 0.0;
    evidence.envelope_east_coordinate_velocity_ms = 200.0;
    evidence.envelope_south_coordinate_velocity_ms = 20.0;
    evidence.envelope_north_coordinate_velocity_ms = 20.0;
    REQUIRE(evidence.distance_west_m + evidence.distance_east_m ==
            old_child_extent_x_m - observed_envelope_span_x_m);
    REQUIRE(evidence.distance_south_m + evidence.distance_north_m ==
            old_child_extent_y_m - observed_envelope_span_y_m);

    StormWindowMovementState state;
    const auto decision = evaluate_storm_window_movement(
        state, evidence, policy);
    REQUIRE(decision.status == StormWindowMovementStatus::Ineligible);
    REQUIRE(decision.reason ==
            StormWindowMovementReason::OppositeReleaseMarginInsufficient);
    REQUIRE(decision.requested_i_shift_cells == 0);
    REQUIRE(decision.requested_j_shift_cells == 0);
    REQUIRE(state.consecutive_compatible_samples == 0);
    REQUIRE_FALSE(storm_window_release_margin_recovered(evidence, policy));

    evidence.distance_west_m -= 3.0 * policy.dx_m;
    evidence.distance_east_m += 3.0 * policy.dx_m;
    evidence.distance_south_m -= policy.dy_m;
    evidence.distance_north_m += policy.dy_m;
    REQUIRE(evidence.distance_west_m == 18600.0);
    REQUIRE(evidence.distance_east_m == 18000.0);
    REQUIRE(evidence.distance_south_m == 15000.0);
    REQUIRE(evidence.distance_north_m == 16200.0);
    REQUIRE_FALSE(storm_window_release_margin_recovered(evidence, policy));
}

TEST_CASE("the enlarged child restores the release box in both parent-lattice phases",
          "[core][domain_motion][movement_authority][quantization][geometry]")
{
    StormWindowMovementPolicy policy;
    policy.inner_margin_m = 15600.0;
    policy.release_margin_m = 16800.0;
    policy.dx_m = 1200.0;
    policy.dy_m = 1200.0;
    policy.maximum_shift_cells = 14;
    policy.cooldown_s = 300.0;
    policy.minimum_confidence = 0.65;
    policy.confirmation_samples = 2;

    const auto prove_phase = [&](double west_m, double east_m,
                                 int expected_shift) {
        auto evidence = qualified_movement_evidence(30.0);
        evidence.distance_west_m = west_m;
        evidence.distance_east_m = east_m;
        evidence.distance_south_m = 17400.0;
        evidence.distance_north_m = 18000.0;
        evidence.velocity_x_ms = 20.0;
        StormWindowMovementState state;
        const auto pending = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(pending.status ==
                StormWindowMovementStatus::PendingConfirmation);
        REQUIRE(pending.reason ==
                StormWindowMovementReason::ConfirmationPending);
        REQUIRE(pending.requested_i_shift_cells == expected_shift);
        REQUIRE(pending.requested_j_shift_cells == 0);

        evidence.sample_time_s = 60.0;
        const auto request = evaluate_storm_window_movement(
            state, evidence, policy);
        REQUIRE(request.status == StormWindowMovementStatus::Requested);
        REQUIRE(request.reason == StormWindowMovementReason::RequestReady);
        evidence.distance_west_m -= request.requested_i_shift_cells *
            policy.dx_m;
        evidence.distance_east_m += request.requested_i_shift_cells *
            policy.dx_m;
        REQUIRE(storm_window_release_margin_recovered(evidence, policy));
        REQUIRE(std::min(evidence.distance_west_m,
                         evidence.distance_east_m) >= 16800.0);
    };

    // A 58.2 km envelope in the 93.6 km child leaves 35.4 km total
    // clearance. Translating the child-grid envelope by one 600 m phase
    // changes the nearest parent-cell center without consuming the release
    // box because the footprint owns one full 1.2 km lattice allowance.
    prove_phase(20400.0, 15000.0, 2);
    prove_phase(21000.0, 14400.0, 3);

    auto y_phase = qualified_movement_evidence(30.0);
    y_phase.distance_west_m = 17400.0;
    y_phase.distance_east_m = 18000.0;
    y_phase.distance_south_m = 20400.0;
    y_phase.distance_north_m = 15000.0;
    y_phase.velocity_y_ms = 20.0;
    StormWindowMovementState y_state;
    const auto y_pending = evaluate_storm_window_movement(
        y_state, y_phase, policy);
    REQUIRE(y_pending.status ==
            StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(y_pending.requested_i_shift_cells == 0);
    REQUIRE(y_pending.requested_j_shift_cells == 2);
    y_phase.sample_time_s = 60.0;
    const auto y_request = evaluate_storm_window_movement(
        y_state, y_phase, policy);
    REQUIRE(y_request.status == StormWindowMovementStatus::Requested);
    REQUIRE(y_request.requested_i_shift_cells == 0);
    REQUIRE(y_request.requested_j_shift_cells == 2);
    y_phase.distance_south_m -= y_request.requested_j_shift_cells *
        policy.dy_m;
    y_phase.distance_north_m += y_request.requested_j_shift_cells *
        policy.dy_m;
    REQUIRE(storm_window_release_margin_recovered(y_phase, policy));
}

TEST_CASE("storm-window movement fails closed on ambiguity and impossible envelope geometry",
          "[core][domain_motion][movement_authority]")
{
    StormWindowMovementState state;
    const auto               policy = movement_policy();

    auto evidence            = qualified_movement_evidence(1.0);
    evidence.distance_east_m = 400.0;
    evidence.velocity_x_ms   = 10.0;
    REQUIRE(evaluate_storm_window_movement(state, evidence, policy).status ==
            StormWindowMovementStatus::PendingConfirmation);

    evidence.sample_time_s = 2.0;
    evidence.ambiguous     = true;
    const auto ambiguous = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(ambiguous.status == StormWindowMovementStatus::Ineligible);
    REQUIRE(ambiguous.reason == StormWindowMovementReason::AmbiguousTrack);
    REQUIRE(state.consecutive_compatible_samples == 0);

    evidence.sample_time_s   = 3.0;
    evidence.ambiguous       = false;
    evidence.distance_west_m = 400.0;
    const auto conflict = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(conflict.status == StormWindowMovementStatus::Ineligible);
    REQUIRE(conflict.reason == StormWindowMovementReason::OppositeFacesBreached);
}

TEST_CASE("storm-window movement commit disarms until release and preserves cooldown",
          "[core][domain_motion][movement_authority]")
{
    StormWindowMovementState state;
    const auto               policy = movement_policy();
    auto                     evidence = qualified_movement_evidence(1.0);
    evidence.distance_east_m = 400.0;
    evidence.velocity_x_ms   = 10.0;
    evaluate_storm_window_movement(state, evidence, policy);
    evidence.sample_time_s = 2.0;
    const auto requested = evaluate_storm_window_movement(state, evidence, policy);
    commit_storm_window_movement(state, requested);
    REQUIRE_FALSE(state.armed);
    REQUIRE(state.cumulative_i_shift_cells == 4);

    evidence.sample_time_s = 3.0;
    const auto unrecovered = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(unrecovered.status == StormWindowMovementStatus::AwaitingRearm);
    REQUIRE(unrecovered.reason == StormWindowMovementReason::ReleaseMarginNotRecovered);

    evidence               = qualified_movement_evidence(4.0);
    evidence.velocity_x_ms = 10.0;
    const auto cooldown = evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE(state.armed);
    REQUIRE(cooldown.status == StormWindowMovementStatus::Cooldown);

    evidence                 = qualified_movement_evidence(12.0);
    evidence.distance_east_m = 400.0;
    evidence.velocity_x_ms   = 10.0;
    REQUIRE(evaluate_storm_window_movement(state, evidence, policy).status ==
            StormWindowMovementStatus::PendingConfirmation);
    evidence.sample_time_s = 13.0;
    REQUIRE(evaluate_storm_window_movement(state, evidence, policy).status ==
            StormWindowMovementStatus::Requested);
}

TEST_CASE("storm-window movement rejects non-causal samples and unsafe policy",
          "[core][domain_motion][movement_authority]")
{
    StormWindowMovementState state;
    auto                     policy   = movement_policy();
    auto                     evidence = qualified_movement_evidence(10.0);
    evaluate_storm_window_movement(state, evidence, policy);
    REQUIRE_THROWS_WITH(evaluate_storm_window_movement(state, evidence, policy),
                        Catch::Matchers::Contains("strictly increasing"));

    StormWindowMovementState fresh_state;
    policy.release_margin_m = policy.inner_margin_m;
    REQUIRE_THROWS_WITH(evaluate_storm_window_movement(fresh_state, evidence, policy),
                        Catch::Matchers::Contains("greater than inner_margin_m"));
}
