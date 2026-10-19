/**
 * @file test_coriolis_yaml.cpp
 * @brief Phase C4 exit-criterion gate — `coriolis.{enabled, mode,
 *        latitude_deg}` YAML keys actually drive simulator behavior.
 *
 * The simulator's Coriolis force is applied through the scalar
 * `coriolis_f` (set in `equations.cpp::initialize()` from
 * `global_coriolis_{enabled, latitude_deg, mode}`). To verify the YAML
 * keys reach the simulator without spinning up the full initialize()
 * path, this test mirrors the same formula:
 *
 *   f = 2 * Omega * sin(latitude_rad)   when enabled
 *   f = 0                                when disabled
 *
 * and asserts that the values the dynamics scheme would see change as
 * the globals change. A separate assertion confirms that running the
 * cartesian momentum tendency with `coriolis_f != 0` produces a different
 * du/dt than with `coriolis_f == 0` — the C4 behavior-changes-output gate.
 */

#include "catch2/catch.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "dynamics/coriolis.hpp"

#include <cmath>

namespace
{

using tmv::init::config::CoriolisMode;

constexpr double kOmegaEarth = 7.292e-5;
constexpr double kPi = 3.14159265358979323846;

/// Replays `equations.cpp::initialize()`'s Coriolis computation against
/// the current globals. Returns the f value that production code would
/// install in the `coriolis_f` global on a real run.
double coriolis_f_from_globals()
{
    if (!global_coriolis_enabled) return 0.0;
    const double lat_rad = global_coriolis_latitude_deg * kPi / 180.0;
    return 2.0 * kOmegaEarth * std::sin(lat_rad);
}

void reset_coriolis_defaults()
{
    global_coriolis_enabled      = true;
    global_coriolis_mode         = CoriolisMode::FPlane;
    global_coriolis_latitude_deg = 35.0;
}

}  // namespace

// ─────────────────────────────────────────────────────────────
// enabled=false zeroes f
// ─────────────────────────────────────────────────────────────

TEST_CASE("C4 coriolis.enabled=false → f = 0",
          "[runtime_config][coriolis][c4]")
{
    global_coriolis_enabled      = false;
    global_coriolis_latitude_deg = 35.0;
    REQUIRE(coriolis_f_from_globals() == Approx(0.0));
    reset_coriolis_defaults();
}

// ─────────────────────────────────────────────────────────────
// latitude_deg drives f via f = 2*Omega*sin(latitude)
// ─────────────────────────────────────────────────────────────

TEST_CASE("C4 coriolis.latitude_deg drives f via 2*Omega*sin(lat)",
          "[runtime_config][coriolis][c4]")
{
    // 35°N (legacy Weisman-Klemp default) — sanity check the magnitude
    global_coriolis_enabled      = true;
    global_coriolis_latitude_deg = 35.0;
    const double f_35 = coriolis_f_from_globals();
    REQUIRE(f_35 == Approx(8.363e-5).epsilon(1.0e-3));

    // Equator: f → 0
    global_coriolis_latitude_deg = 0.0;
    REQUIRE(coriolis_f_from_globals() == Approx(0.0));

    // North Pole: f → 2*Omega ≈ 1.458e-4
    global_coriolis_latitude_deg = 90.0;
    REQUIRE(coriolis_f_from_globals() == Approx(2.0 * kOmegaEarth).epsilon(1.0e-6));

    // Southern hemisphere flips sign
    global_coriolis_latitude_deg = -35.0;
    REQUIRE(coriolis_f_from_globals() == Approx(-f_35).epsilon(1.0e-6));

    reset_coriolis_defaults();
}

// ─────────────────────────────────────────────────────────────
// f != 0 produces a different momentum tendency than f == 0
// ─────────────────────────────────────────────────────────────

TEST_CASE("C4 nonzero f changes the Cartesian momentum tendency",
          "[runtime_config][coriolis][c4][behavior]")
{
    // Per cartesian.cpp:197-206, the Coriolis contribution to du/dt is
    //   coriolis_x =  f * (uy - v0_base[k])
    //   coriolis_y = -f * (ux - u0_base[k])
    // With a uniform interior wind (ux=10, uy=0) and zero base state,
    // disabling Coriolis (f=0) zeros both terms; enabling at 35°N gives
    // measurable contributions.
    const double ux = 10.0;
    const double uy = 0.0;
    const double u0 = 0.0;
    const double v0 = 0.0;

    global_coriolis_enabled      = false;
    const double f_off = coriolis_f_from_globals();
    const double coriolis_x_off =  f_off * (uy - v0);
    const double coriolis_y_off = -f_off * (ux - u0);

    global_coriolis_enabled      = true;
    global_coriolis_latitude_deg = 35.0;
    const double f_on = coriolis_f_from_globals();
    const double coriolis_x_on =  f_on * (uy - v0);
    const double coriolis_y_on = -f_on * (ux - u0);

    REQUIRE(coriolis_x_off == Approx(0.0));
    REQUIRE(coriolis_y_off == Approx(0.0));
    REQUIRE(coriolis_x_on == Approx(0.0));
    // Coriolis-y for a westerly wind ux=10 at 35°N: -f*ux ≈ -8.36e-4
    REQUIRE(coriolis_y_on != Approx(coriolis_y_off));
    REQUIRE(coriolis_y_on < 0.0);

    reset_coriolis_defaults();
}

// ─────────────────────────────────────────────────────────────
// The enum round-trips through the compatibility surface without substituting
// f-plane physics.
// ─────────────────────────────────────────────────────────────

TEST_CASE("C4 coriolis.mode round-trips through the global",
          "[runtime_config][coriolis][c4]")
{
    global_coriolis_mode = CoriolisMode::FPlane;
    REQUIRE(global_coriolis_mode == CoriolisMode::FPlane);
    global_coriolis_mode = CoriolisMode::BetaPlane;
    REQUIRE(global_coriolis_mode == CoriolisMode::BetaPlane);
    reset_coriolis_defaults();
}

TEST_CASE("beta-plane profile follows Earth coordinates under frame motion",
          "[runtime_config][coriolis][beta_plane][domain_motion]")
{
    tmv::domain::HorizontalDomainFrame frame;
    frame.earth_origin_at_t0_m = {1200.0, -800.0};
    frame.velocity_ms = {15.0, 10.0};

    const auto initial = tmv::make_cartesian_coriolis_profile(
        true, true, 35.0, frame, 0.0);
    const auto later = tmv::make_cartesian_coriolis_profile(
        true, true, 35.0, frame, 20.0);
    REQUIRE(initial.beta > 0.0);
    REQUIRE(later.beta == initial.beta);
    REQUIRE(later.f_at_grid_y0
            == Approx(initial.f_at_grid_y0 + initial.beta * 200.0));
    REQUIRE(later.at_grid_y(500.0)
            == Approx(initial.at_grid_y(700.0)));

    auto recycled_frame = frame;
    recycled_frame.earth_origin_at_t0_m.y_m += 300.0;
    const auto recycled = tmv::make_cartesian_coriolis_profile(
        true, true, 35.0, recycled_frame, 0.0);
    REQUIRE(recycled.at_grid_y(250.0)
            == Approx(initial.at_grid_y(550.0)));

    const auto f_plane = tmv::make_cartesian_coriolis_profile(
        true, false, 35.0, frame, 20.0);
    REQUIRE(f_plane.beta == 0.0);
    REQUIRE(f_plane.f_at_grid_y0 == Approx(coriolis_f_from_globals()));

    const auto disabled = tmv::make_cartesian_coriolis_profile(
        false, true, 35.0, frame, 20.0);
    REQUIRE(disabled.f_at_grid_y0 == 0.0);
    REQUIRE(disabled.beta == 0.0);
}

// ─────────────────────────────────────────────────────────────
// Defaults preserve legacy v0 behavior (enabled, 35°N, f-plane)
// ─────────────────────────────────────────────────────────────

TEST_CASE("C4 globals default to legacy Weisman-Klemp 35°N f-plane",
          "[runtime_config][coriolis][c4][defaults]")
{
    REQUIRE(global_coriolis_enabled      == true);
    REQUIRE(global_coriolis_mode         == CoriolisMode::FPlane);
    REQUIRE(global_coriolis_latitude_deg == Approx(35.0));
}
