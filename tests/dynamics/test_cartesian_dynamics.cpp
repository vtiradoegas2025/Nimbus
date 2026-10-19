#include "diagnostics/resolved_diagnostics.hpp"
#include "core/runtime/simulation_state.hpp"
/**
 * @file test_cartesian_dynamics.cpp
 * @brief Unit tests for the Cartesian dynamics scheme (Phase A.2 gates).
 *
 * Verification gates for the retained Cartesian A-grid comparison path:
 *
 *   1. Hydrostatic equilibrium + zero wind   -> all tendencies |.| <= 1e-3
 *   2. Hydrostatic + uniform Cartesian wind  -> all tendencies |.| <= 1e-3
 *      *** This is the test that the cylindrical scheme cannot pass.     ***
 *      *** It is the proof Cartesian solves Bug 7.                        ***
 *   3. Warm bubble (deltaTheta = 2 K, r = 1 km)
 *      -> vertical tendency at bubble center within 10 % of g*deltaTheta/theta0.
 *
 * Test strategy:
 *
 *   - Build a tiny hydrostatic atmosphere with linear pressure
 *     p(z) = p0 - rho0*g*z and constant density rho0. Centered differences
 *     on a linear function are exact, so dp/dz is reproduced exactly, and
 *     the vertical momentum equation -dp/dz/rho - g evaluates to 0 in the
 *     hydrostatic case.
 *
 *   - Drive the scheme directly by instantiating a `CartesianAGridScheme` and
 *     calling `compute_momentum_tendencies` on local Field3D buffers. The
 *     grid globals (NR, NTH, NZ, dr, dz, dtheta, rho0_base) come from the
 *     test_harness; the test sets them to small workable values in each
 *     SECTION, matching the pattern used by tests/numerics/test_advection_tvd.cpp.
 */

#include "catch2/catch.hpp"
#include "core/field/field3d.hpp"
#include "core/runtime/simulation.hpp"
#include "dynamics/collocated_velocity_constraint.hpp"
#include "dynamics/schemes/klemp_wilhelmson_split/cartesian_agrid.hpp"
#include "dynamics/dynamics_base.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

// Sea-level air density and reference pressure for the synthetic atmosphere.
// These are chosen so the hydrostatic profile p(z) = p0 - rho0*g*z stays
// positive over the vertical extent of the test grid (NZ*dz = 8 km here).
constexpr double kRho0 = 1.225;       // kg / m^3
constexpr double kP0   = 101325.0;    // Pa

/**
 * @brief Initializes the grid globals used by CartesianAGridScheme's constructor.
 *        Called by every test case because tests may have mutated them.
 */
void setup_cartesian_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;   // 1 km x spacing
    dz = 500.0;    // 0.5 km vertical
    dt = 0.1;
    dtheta = dr;   // Cartesian y spacing; square by default in these tests
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_stagger_type = StaggerType::Collocated;
    global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();

    // The pressure-diagnostic path reads rho0_base by vertical index. It is
    // not exercised by compute_momentum_tendencies, but populate it anyway
    // so any incidental access (or a future test that calls it) is safe.
    rho0_base.assign(NZ, kRho0);

    // The reference-state subtraction in the vertical momentum equation reads
    // p0_base[k-1] and p0_base[k+1]. Populate with the same linear profile
    // used by initialize_hydrostatic_atmosphere.
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
    {
        p0_base[k] = kP0 - kRho0 * dynamics_constants::g * static_cast<double>(k) * dz;
    }

    // Base-state wind profiles for perturbation Coriolis. Zero wind base
    // state for the synthetic test atmosphere.
    u0_base.assign(NZ, 0.0);
    v0_base.assign(NZ, 0.0);
    qv0_base.assign(NZ, 0.012);
}

/**
 * @brief Populates (rho, p) with a hydrostatic base state:
 *          rho(z) = rho0,      p(z) = p0 - rho0 * g * z.
 *        Centered differences of the linear p(z) reproduce -rho0*g exactly,
 *        so -dp/dz/rho - g = 0 to machine precision on the interior stencil.
 */
void initialize_hydrostatic_atmosphere(Field3D& rho_field, Field3D& p_field)
{
    rho_field.resize(NR, NTH, NZ, static_cast<float>(kRho0));
    p_field.resize(NR, NTH, NZ);
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double z = static_cast<double>(k) * dz;
                p_field[i][j][k] = static_cast<float>(kP0 - kRho0 * dynamics_constants::g * z);
            }
        }
    }
}

void initialize_isothermal_theta(Field3D& theta_field)
{
    theta_field.resize(NR, NTH, NZ,
                       static_cast<float>(dynamics_constants::theta0));
}

void zero_velocity_field(Field3D& u_field, Field3D& v_field, Field3D& w_field)
{
    u_field.resize(NR, NTH, NZ, 0.0f);
    v_field.resize(NR, NTH, NZ, 0.0f);
    w_field.resize(NR, NTH, NZ, 0.0f);
}

void uniform_horizontal_wind(Field3D& u_field, Field3D& v_field, Field3D& w_field,
                             double ux, double uy)
{
    u_field.resize(NR, NTH, NZ, static_cast<float>(ux));
    v_field.resize(NR, NTH, NZ, static_cast<float>(uy));
    w_field.resize(NR, NTH, NZ, 0.0f);
}

/**
 * @brief Returns the largest magnitude of a field over its interior cells
 *        (the range [1..N-1) in each dimension — i.e. exactly the cells the
 *        scheme actually computes). Excluding the boundary zeros makes the
 *        assertion test the stencil output rather than the zero pad.
 */
double max_abs_interior(const Field3D& f)
{
    double max_val = 0.0;
    for (int i = 1; i < NR - 1; ++i)
    {
        for (int j = 1; j < NTH - 1; ++j)
        {
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double v = std::abs(static_cast<double>(f[i][j][k]));
                if (v > max_val) max_val = v;
            }
        }
    }
    return max_val;
}

struct TendencyBuffers
{
    Field3D du_x_dt;
    Field3D du_y_dt;
    Field3D dw_dt;
    Field3D drho_dt_out;
    Field3D dp_dt_out;
};

TendencyBuffers make_tendency_buffers()
{
    TendencyBuffers tb;
    tb.du_x_dt.resize(NR, NTH, NZ, 0.0f);
    tb.du_y_dt.resize(NR, NTH, NZ, 0.0f);
    tb.dw_dt.resize(NR, NTH, NZ, 0.0f);
    tb.drho_dt_out.resize(NR, NTH, NZ, 0.0f);
    tb.dp_dt_out.resize(NR, NTH, NZ, 0.0f);
    return tb;
}

}  // namespace

// ===========================================================================
// Gate 1: hydrostatic equilibrium + zero wind
// ===========================================================================
TEST_CASE("CartesianAGridScheme: hydrostatic equilibrium has near-zero tendencies",
          "[dynamics][cartesian]")
{
    setup_cartesian_grid();

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity_field(u_f, v_f, w_f);

    TendencyBuffers tb = make_tendency_buffers();

    CartesianAGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f,
        rho_f, p_f, theta_f,
        dt,
        tb.du_x_dt, tb.du_y_dt, tb.dw_dt,
        tb.drho_dt_out, tb.dp_dt_out);

    // All tendencies must be at machine-noise level for a flow that is
    // already in equilibrium: no winds, no pressure imbalance, no mass
    // imbalance. 1e-3 is the gate specified in the plan.
    const double kTol = 1.0e-3;
    REQUIRE(max_abs_interior(tb.du_x_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.du_y_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.dw_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.drho_dt_out) <= kTol);
    REQUIRE(max_abs_interior(tb.dp_dt_out) <= kTol);
}

// ===========================================================================
// Gate 2: hydrostatic + uniform Cartesian wind — THE BUG 7 PROOF
// ===========================================================================
TEST_CASE("CartesianAGridScheme: uniform Cartesian wind preserves equilibrium (Bug 7 proof)",
          "[dynamics][cartesian][bug7]")
{
    setup_cartesian_grid();

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);

    // Asymmetric Cartesian wind — both components nonzero, different
    // magnitudes, so that any fake azimuthal projection (which the
    // cylindrical grid would produce via u = u_x cos theta + u_y sin theta)
    // would be visible as a spurious body force. On the Cartesian grid
    // there is no such projection and the tendencies must stay at noise.
    uniform_horizontal_wind(u_f, v_f, w_f, /*ux=*/10.0, /*uy=*/5.0);

    TendencyBuffers tb = make_tendency_buffers();

    CartesianAGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f,
        rho_f, p_f, theta_f,
        dt,
        tb.du_x_dt, tb.du_y_dt, tb.dw_dt,
        tb.drho_dt_out, tb.dp_dt_out);

    // KEY VERIFICATION GATE for Phase A.2: a uniform Cartesian wind on a
    // Cartesian grid produces zero spurious body force. This is the test
    // the cylindrical grid cannot pass — the antisymmetric BC at i=0 creates
    // a false radial gradient, which drives a false divergence, which drives
    // a false dp/dt and breaks hydrostatic balance. This regression is the
    // executable contract for the retained Cartesian A-grid comparison path.
    const double kTol = 1.0e-3;
    REQUIRE(max_abs_interior(tb.du_x_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.du_y_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.dw_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.drho_dt_out) <= kTol);
    REQUIRE(max_abs_interior(tb.dp_dt_out) <= kTol);
}

TEST_CASE("Cartesian A-grid tendencies are invariant under a constant translating frame",
          "[dynamics][cartesian][agrid][domain-motion][SCI-195]")
{
    setup_cartesian_grid();
    coriolis_f = 0.0;

    Field3D rho_f(NR, NTH, NZ, 0.0f);
    Field3D p_f(NR, NTH, NZ, 0.0f);
    Field3D theta_f(NR, NTH, NZ,
                    static_cast<float>(dynamics_constants::theta0));
    Field3D u_stationary(NR, NTH, NZ, 0.0f);
    Field3D v_stationary(NR, NTH, NZ, 0.0f);
    Field3D w_f(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double wave = std::sin(0.31 * i)
                                  * std::cos(0.27 * j);
                rho_f[i][j][k] = static_cast<float>(
                    kRho0 * (1.0 + 0.002 * wave + 0.0001 * k));
                p_f[i][j][k] = static_cast<float>(
                    p0_base[k] + 18.0 * wave + 0.2 * i * j);
                u_stationary[i][j][k] = static_cast<float>(
                    3.0 + 0.08 * i - 0.04 * j + 0.015 * k);
                v_stationary[i][j][k] = static_cast<float>(
                    -1.5 + 0.03 * i + 0.07 * j - 0.01 * k);
                w_f[i][j][k] = static_cast<float>(
                    0.05 * std::sin(0.23 * k) * wave);
            }

    CartesianAGridScheme stationary_scheme;
    TendencyBuffers stationary = make_tendency_buffers();
    stationary_scheme.compute_momentum_tendencies(
        u_stationary, v_stationary, w_f, rho_f, p_f, theta_f, dt,
        stationary.du_x_dt, stationary.du_y_dt, stationary.dw_dt,
        stationary.drho_dt_out, stationary.dp_dt_out);

    constexpr double frame_u = 7.25;
    constexpr double frame_v = -4.5;
    Field3D u_translated = u_stationary;
    Field3D v_translated = v_stationary;
    for (std::size_t index = 0; index < u_translated.size(); ++index)
    {
        u_translated.data()[index] = static_cast<float>(
            static_cast<double>(u_translated.data()[index]) + frame_u);
        v_translated.data()[index] = static_cast<float>(
            static_cast<double>(v_translated.data()[index]) + frame_v);
    }
    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::ConstantVelocity;
    global_domain_frame.velocity_ms = {frame_u, frame_v};

    CartesianAGridScheme translated_scheme;
    TendencyBuffers translated = make_tendency_buffers();
    translated_scheme.compute_momentum_tendencies(
        u_translated, v_translated, w_f, rho_f, p_f, theta_f, dt,
        translated.du_x_dt, translated.du_y_dt, translated.dw_dt,
        translated.drho_dt_out, translated.dp_dt_out);

    const auto require_same = [](const Field3D& reference,
                                 const Field3D& candidate,
                                 double margin)
    {
        for (int i = 2; i < NR - 2; ++i)
            for (int j = 2; j < NTH - 2; ++j)
                for (int k = 2; k < NZ - 2; ++k)
                    REQUIRE(candidate[i][j][k]
                            == Approx(reference[i][j][k]).margin(margin));
    };
    require_same(stationary.du_x_dt, translated.du_x_dt, 2.0e-7);
    require_same(stationary.du_y_dt, translated.du_y_dt, 2.0e-7);
    require_same(stationary.dw_dt, translated.dw_dt, 2.0e-7);
    require_same(stationary.drho_dt_out, translated.drho_dt_out, 2.0e-9);
    require_same(stationary.dp_dt_out, translated.dp_dt_out, 2.0e-3);

    global_domain_motion_mode =
        tmv::init::config::DomainMotionMode::Stationary;
    global_domain_frame = tmv::domain::HorizontalDomainFrame::stationary();
}

TEST_CASE("CartesianAGridScheme: continuity is conservative on closed collocated faces",
          "[dynamics][cartesian][agrid][continuity][AG-003]")
{
    setup_cartesian_grid();
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;

    Field3D rho_f(NR, NTH, NZ, 0.0f);
    Field3D p_f(NR, NTH, NZ, 90000.0f);
    Field3D u_f(NR, NTH, NZ, 0.0f);
    Field3D v_f(NR, NTH, NZ, 0.0f);
    Field3D w_f(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                rho_f[i][j][k] = static_cast<float>(
                    0.9 + 0.01 * i + 0.004 * j + 0.002 * k);
                p_f[i][j][k] = static_cast<float>(p0_base[k]);
                u_f[i][j][k] = static_cast<float>(0.3 * i - 0.1 * j);
                v_f[i][j][k] = static_cast<float>(-0.2 * i + 0.25 * j);
                w_f[i][j][k] = static_cast<float>(0.05 * k - 0.02 * i);
            }

    Field3D drho_dt(NR, NTH, NZ, 0.0f);
    Field3D dp_dt(NR, NTH, NZ, 0.0f);
    CartesianAGridScheme scheme;
    scheme.compute_fast_pressure_tendencies(
        u_f, v_f, w_f, rho_f, p_f, /*dt_small=*/0.1,
        drho_dt, dp_dt);

    long double domain_tendency = 0.0L;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
                domain_tendency += static_cast<long double>(drho_dt[i][j][k]);

    // All interior face fluxes cancel pairwise and every exterior normal flux
    // is exactly zero, so only float tendency rounding remains.
    REQUIRE(static_cast<double>(std::abs(domain_tendency)) < 2.0e-7);

    const int i = 5, j = 6, k = 7;
    const auto face = [](float rho_a, float rho_b, float vel_a, float vel_b) {
        return 0.25 * (static_cast<double>(rho_a) + rho_b)
                    * (static_cast<double>(vel_a) + vel_b);
    };
    const double expected = -(
        (face(rho_f[i][j][k], rho_f[i + 1][j][k],
              u_f[i][j][k], u_f[i + 1][j][k])
         - face(rho_f[i - 1][j][k], rho_f[i][j][k],
                u_f[i - 1][j][k], u_f[i][j][k])) / dr
      + (face(rho_f[i][j][k], rho_f[i][j + 1][k],
              v_f[i][j][k], v_f[i][j + 1][k])
         - face(rho_f[i][j - 1][k], rho_f[i][j][k],
                v_f[i][j - 1][k], v_f[i][j][k])) / dtheta
      + (face(rho_f[i][j][k], rho_f[i][j][k + 1],
              w_f[i][j][k], w_f[i][j][k + 1])
         - face(rho_f[i][j][k - 1], rho_f[i][j][k],
                w_f[i][j][k - 1], w_f[i][j][k])) / dz);
    REQUIRE(drho_dt[i][j][k] == Approx(expected).margin(1.0e-10));
}

TEST_CASE("CartesianAGridScheme couples the collocated pressure checkerboard to continuity",
          "[dynamics][cartesian][agrid][pressure-coupling][AG-004]")
{
    setup_cartesian_grid();
    global_lateral_bc_west = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_east = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
    global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;
    global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
    global_top_bc = tmv::init::config::TopBC::Rigid;

    Field3D rho_f(NR, NTH, NZ, 1.0f);
    Field3D p_f(NR, NTH, NZ, 0.0f);
    Field3D u_f(NR, NTH, NZ, 0.0f);
    Field3D v_f(NR, NTH, NZ, 0.0f);
    Field3D w_f(NR, NTH, NZ, 0.0f);
    constexpr double pressure_amplitude = 100.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double sign = (i % 2 == 0) ? 1.0 : -1.0;
                p_f[i][j][k] = static_cast<float>(
                    static_cast<float>(p0_base[k])
                    + sign * pressure_amplitude);
            }

    Field3D drho_dt(NR, NTH, NZ, 0.0f);
    Field3D dp_dt(NR, NTH, NZ, 0.0f);
    Field3D du_dt(NR, NTH, NZ, 0.0f);
    Field3D dv_dt(NR, NTH, NZ, 0.0f);
    Field3D dw_dt(NR, NTH, NZ, 0.0f);
    CartesianAGridScheme scheme;
    constexpr double dt_small = 0.1;
    scheme.compute_fast_pressure_tendencies(
        u_f, v_f, w_f, rho_f, p_f, dt_small, drho_dt, dp_dt);
    scheme.compute_fast_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, dt_small, du_dt, dv_dt, dw_dt);

    long double mass_tendency = 0.0L;
    long double checker_projection = 0.0L;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double sign = (i % 2 == 0) ? 1.0 : -1.0;
                mass_tendency += drho_dt[i][j][k];
                checker_projection += sign * drho_dt[i][j][k];
            }

    // The legacy centered cell gradient is blind to this exact 2dx mode.
    const int ic = NR / 2;
    const int jc = NTH / 2;
    const int kc = NZ / 2;
    REQUIRE(du_dt[ic][jc][kc] == 0.0f);

    // The momentum-weighted face flux sees it. Positive-pressure parity loses
    // mass and negative-pressure parity gains it, reducing the EOS pressure
    // projection without creating net mass in the closed domain.
    REQUIRE(static_cast<double>(checker_projection) < 0.0);
    REQUIRE(std::abs(static_cast<double>(mass_tendency)) < 2.0e-7);
    const double center_sign = (ic % 2 == 0) ? 1.0 : -1.0;
    REQUIRE(center_sign * static_cast<double>(drho_dt[ic][jc][kc]) < 0.0);
}

TEST_CASE("A-grid velocity constraint removes selected directional alternating residuals",
          "[dynamics][cartesian][agrid][velocity-coupling][AG-004]")
{
    setup_cartesian_grid();
    Field3D rho_f(NR, NTH, NZ, 1.0f);
    Field3D u_f(NR, NTH, NZ, -17.0f);
    Field3D v_f(NR, NTH, NZ, -19.0f);
    Field3D w_f(NR, NTH, NZ, -23.0f);

    // Vary density so preservation is tested in the kinetic-energy metric,
    // not only for the constant-density special case. Each component is a
    // linear resolved profile plus an exact alternating face-average null.
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                rho_f[i][j][k] = static_cast<float>(
                    0.8 + 0.01 * i + 0.02 * j + 0.03 * k);
                u_f[i][j][k] = static_cast<float>(
                    2.0 + 0.1 * i + ((i - 1) & 1 ? -0.75 : 0.75));
                v_f[i][j][k] = static_cast<float>(
                    -1.0 + 0.05 * j + ((j - 1) & 1 ? 0.5 : -0.5));
                w_f[i][j][k] = static_cast<float>(
                    0.02 * k + ((k - 1) & 1 ? -1.25 : 1.25));
            }

    const auto line_moments = [](const Field3D& velocity,
                                 const Field3D& density,
                                 int axis,
                                 int a,
                                 int b)
    {
        const int count = axis == 0 ? NR - 2 : axis == 1 ? NTH - 2 : NZ - 2;
        const double midpoint = 0.5 * static_cast<double>(count - 1);
        double momentum = 0.0;
        double first_moment = 0.0;
        double kinetic_twice = 0.0;
        for (int n = 0; n < count; ++n)
        {
            const int i = axis == 0 ? n + 1 : a;
            const int j = axis == 1 ? n + 1 : axis == 0 ? a : b;
            const int k = axis == 2 ? n + 1 : b;
            const double mass = density[i][j][k];
            const double value = velocity[i][j][k];
            const double x = static_cast<double>(n) - midpoint;
            momentum += mass * value;
            first_moment += mass * x * value;
            kinetic_twice += mass * value * value;
        }
        return std::vector<double>{momentum, first_moment, kinetic_twice};
    };

    const auto u_before = line_moments(u_f, rho_f, 0, 5, 6);
    const auto v_before = line_moments(v_f, rho_f, 1, 5, 6);
    const auto w_before = line_moments(w_f, rho_f, 2, 5, 6);

    tmv::collocated_velocity::constrain_directional_null_modes(
        u_f, v_f, w_f, rho_f, NR, NTH, NZ,
        /*constrain_x=*/true, /*constrain_y=*/true,
        /*constrain_z=*/true);

    const auto u_after = line_moments(u_f, rho_f, 0, 5, 6);
    const auto v_after = line_moments(v_f, rho_f, 1, 5, 6);
    const auto w_after = line_moments(w_f, rho_f, 2, 5, 6);
    for (int component = 0; component < 3; ++component)
    {
        const auto& before = component == 0 ? u_before
                           : component == 1 ? v_before : w_before;
        const auto& after = component == 0 ? u_after
                          : component == 1 ? v_after : w_after;
        REQUIRE(after[0] == Approx(before[0]).margin(2.0e-6));
        REQUIRE(after[1] == Approx(before[1]).margin(2.0e-5));
        REQUIRE(after[2] < before[2]);
    }

    // The correction never touches storage ghosts.
    REQUIRE(u_f[0][5][6] == -17.0f);
    REQUIRE(v_f[5][0][6] == -19.0f);
    REQUIRE(w_f[5][6][0] == -23.0f);

    // The alternating residual after the best rho-weighted linear fit is zero
    // to FP32 commit tolerance on representative lines.
    const auto alternating_residual = [](const Field3D& velocity,
                                         const Field3D& density,
                                         int axis,
                                         int a,
                                         int b)
    {
        const int count = axis == 0 ? NR - 2 : axis == 1 ? NTH - 2 : NZ - 2;
        const double midpoint = 0.5 * static_cast<double>(count - 1);
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, t0 = 0.0, t1 = 0.0;
        for (int n = 0; n < count; ++n)
        {
            const int i = axis == 0 ? n + 1 : a;
            const int j = axis == 1 ? n + 1 : axis == 0 ? a : b;
            const int k = axis == 2 ? n + 1 : b;
            const double mass = density[i][j][k];
            const double x = static_cast<double>(n) - midpoint;
            const double parity = (n & 1) == 0 ? 1.0 : -1.0;
            s0 += mass; s1 += mass * x; s2 += mass * x * x;
            t0 += mass * parity; t1 += mass * x * parity;
        }
        const double determinant = s0 * s2 - s1 * s1;
        const double intercept = (t0 * s2 - t1 * s1) / determinant;
        const double slope = (t1 * s0 - t0 * s1) / determinant;
        double projection = 0.0;
        for (int n = 0; n < count; ++n)
        {
            const int i = axis == 0 ? n + 1 : a;
            const int j = axis == 1 ? n + 1 : axis == 0 ? a : b;
            const int k = axis == 2 ? n + 1 : b;
            const double mass = density[i][j][k];
            const double x = static_cast<double>(n) - midpoint;
            const double parity = (n & 1) == 0 ? 1.0 : -1.0;
            projection += mass * (parity - intercept - slope * x)
                        * velocity[i][j][k];
        }
        return projection;
    };
    REQUIRE(std::abs(alternating_residual(u_f, rho_f, 0, 5, 6)) < 2.0e-6);
    REQUIRE(std::abs(alternating_residual(v_f, rho_f, 1, 5, 6)) < 2.0e-6);
    REQUIRE(std::abs(alternating_residual(w_f, rho_f, 2, 5, 6)) < 2.0e-6);

    Field3D linear_u(NR, NTH, NZ, 0.0f);
    Field3D linear_v(NR, NTH, NZ, 0.0f);
    Field3D linear_w(NR, NTH, NZ, 0.0f);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                linear_u[i][j][k] = static_cast<float>(-0.5 + 0.01 * i);
                linear_v[i][j][k] = static_cast<float>(0.75 - 0.03 * j);
                linear_w[i][j][k] = static_cast<float>(0.25 + 0.02 * k);
            }
    const Field3D linear_u_before = linear_u;
    const Field3D linear_v_before = linear_v;
    const Field3D linear_before = linear_w;
    tmv::collocated_velocity::constrain_directional_null_modes(
        linear_u, linear_v, linear_w, rho_f, NR, NTH, NZ,
        true, true, true);
    REQUIRE(std::equal(
        linear_u.data(), linear_u.data() + linear_u.size(),
        linear_u_before.data()));
    REQUIRE(std::equal(
        linear_v.data(), linear_v.data() + linear_v.size(),
        linear_v_before.data()));
    REQUIRE(std::equal(
        linear_w.data(), linear_w.data() + linear_w.size(),
        linear_before.data()));

    Field3D invalid_rho = rho_f;
    Field3D invalid_w = w_f;
    const Field3D invalid_before = invalid_w;
    invalid_rho[5][6][7] = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        tmv::collocated_velocity::constrain_directional_null_modes(
            u_f, v_f, invalid_w, invalid_rho, NR, NTH, NZ,
            true, true, true),
        std::invalid_argument);
    REQUIRE(std::equal(
        invalid_w.data(), invalid_w.data() + invalid_w.size(),
        invalid_before.data()));
}

TEST_CASE("Collocated null projection uses total mixture kinetic mass",
          "[dynamics][cartesian][agrid][constraint][energy][SCI-146]")
{
    setup_cartesian_grid();
    Field3D rho_f(NR, NTH, NZ, 1.0f);
    Field3D u_f(NR, NTH, NZ, 0.0f);
    Field3D v_f(NR, NTH, NZ, 0.0f);
    Field3D w_f(NR, NTH, NZ, 0.0f);
    Field3D qv_f(NR, NTH, NZ, 0.0f);
    Field3D qc_f(NR, NTH, NZ, 0.0f);
    Field3D qr_f(NR, NTH, NZ, 0.0f);
    Field3D qi_f(NR, NTH, NZ, 0.0f);
    Field3D qs_f(NR, NTH, NZ, 0.0f);
    Field3D qg_f(NR, NTH, NZ, 0.0f);
    Field3D qh_f(NR, NTH, NZ, 0.0f);
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 1; j < NTH - 1; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                rho_f[i][j][k] =
                    static_cast<float>(0.85 + 0.015 * i);
                qv_f[i][j][k] = static_cast<float>(
                    0.002 + 0.0008 * i
                    + (((i - 1) & 1) == 0 ? 0.018 : 0.0));
                u_f[i][j][k] = static_cast<float>(
                    1.0 + 0.04 * i
                    + (((i - 1) & 1) == 0 ? 0.9 : -0.9));
            }
    const std::array<const Field3D*, 7> water{{
        &qv_f, &qc_f, &qr_f, &qi_f, &qs_f, &qg_f, &qh_f,
    }};
    const auto mixture_mass = [&](int i, int j, int k)
    {
        double factor = 1.0;
        for (const Field3D* field : water)
            factor += static_cast<double>((*field)[i][j][k]);
        return static_cast<double>(rho_f[i][j][k]) * factor;
    };
    const int representative_j = 5;
    const int representative_k = 6;
    const int count = NR - 2;
    const double midpoint = 0.5 * static_cast<double>(count - 1);
    const auto moments = [&]()
    {
        std::array<double, 3> result{};
        for (int n = 0; n < count; ++n)
        {
            const int i = n + 1;
            const double mass =
                mixture_mass(i, representative_j, representative_k);
            const double velocity =
                static_cast<double>(
                    u_f[i][representative_j][representative_k]);
            const double x = static_cast<double>(n) - midpoint;
            result[0] += mass * velocity;
            result[1] += mass * x * velocity;
            result[2] += mass * velocity * velocity;
        }
        return result;
    };
    const auto mixture_kinetic_density = [&]()
    {
        double result = 0.0;
        for (int i = 1; i < NR - 1; ++i)
            for (int j = 1; j < NTH - 1; ++j)
                for (int k = 1; k < NZ - 1; ++k)
                {
                    const double u_k = u_f[i][j][k];
                    const double v_k = v_f[i][j][k];
                    const double w_k = w_f[i][j][k];
                    result += 0.5 * mixture_mass(i, j, k)
                        * (u_k * u_k + v_k * v_k + w_k * w_k);
                }
        return result;
    };
    const auto before = moments();
    const double kinetic_before = mixture_kinetic_density();
    double committed_work = std::numeric_limits<double>::quiet_NaN();

    tmv::collocated_velocity::constrain_directional_null_modes(
        u_f, v_f, w_f, rho_f, NR, NTH, NZ,
        true, false, false, water,
        dr * dtheta * dz, &committed_work);
    const auto after = moments();
    const double kinetic_after = mixture_kinetic_density();

    REQUIRE(after[0] == Approx(before[0]).margin(3.0e-6));
    REQUIRE(after[1] == Approx(before[1]).margin(3.0e-5));
    REQUIRE(after[2] < before[2]);
    REQUIRE(committed_work
            == Approx(
                (kinetic_after - kinetic_before) * dr * dtheta * dz)
                   .epsilon(2.0e-12).margin(1.0e-5));
    REQUIRE(committed_work <= 0.0);

    const Field3D u_before_reject = u_f;
    std::array<const Field3D*, 7> partial_water = water;
    partial_water[6] = nullptr;
    REQUIRE_THROWS_AS(
        tmv::collocated_velocity::constrain_directional_null_modes(
            u_f, v_f, w_f, rho_f, NR, NTH, NZ,
            true, false, false, partial_water),
        std::invalid_argument);
    REQUIRE(std::equal(
        u_f.data(), u_f.data() + u_f.size(),
        u_before_reject.data()));
}

TEST_CASE("CartesianAGridScheme pressure coupling leaves a linear pressure field unchanged",
          "[dynamics][cartesian][agrid][pressure-coupling][AG-004]")
{
    setup_cartesian_grid();
    Field3D rho_f(NR, NTH, NZ, 1.0f);
    Field3D p_f(NR, NTH, NZ, 0.0f);
    Field3D u_f(NR, NTH, NZ, 0.0f);
    Field3D v_f(NR, NTH, NZ, 0.0f);
    Field3D w_f(NR, NTH, NZ, 0.0f);
    constexpr double gradient = 0.01;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p_f[i][j][k] = static_cast<float>(
                    static_cast<float>(p0_base[k]) + gradient * i * dr);

    Field3D drho_dt(NR, NTH, NZ, 0.0f);
    Field3D dp_dt(NR, NTH, NZ, 0.0f);
    CartesianAGridScheme scheme;
    scheme.compute_fast_pressure_tendencies(
        u_f, v_f, w_f, rho_f, p_f, /*dt_small=*/0.1,
        drho_dt, dp_dt);

    // Away from boundary closures, direct face and interpolated cell pressure
    // accelerations are identical for a linear field, so the correction is
    // zero to FP32 storage precision.
    REQUIRE(std::abs(static_cast<double>(drho_dt[NR / 2][NTH / 2][NZ / 2]))
            < 1.0e-10);
}

TEST_CASE("CartesianAGridScheme pressure coupling rejects invalid input transactionally",
          "[dynamics][cartesian][agrid][pressure-coupling][transactional][AG-004]")
{
    setup_cartesian_grid();
    Field3D rho_f(NR, NTH, NZ, 1.0f);
    Field3D p_f(NR, NTH, NZ, 0.0f);
    Field3D u_f(NR, NTH, NZ, 0.0f);
    Field3D v_f(NR, NTH, NZ, 0.0f);
    Field3D w_f(NR, NTH, NZ, 0.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p_f[i][j][k] = static_cast<float>(p0_base[k]);

    constexpr float sentinel_rho = 17.0f;
    constexpr float sentinel_p = -23.0f;
    Field3D drho_dt(NR, NTH, NZ, sentinel_rho);
    Field3D dp_dt(NR, NTH, NZ, sentinel_p);
    CartesianAGridScheme scheme;

    REQUIRE_THROWS_AS(
        scheme.compute_fast_pressure_tendencies(
            u_f, v_f, w_f, rho_f, p_f, /*dt_small=*/0.0,
            drho_dt, dp_dt),
        std::invalid_argument);
    REQUIRE(drho_dt[NR / 2][NTH / 2][NZ / 2] == sentinel_rho);
    REQUIRE(dp_dt[NR / 2][NTH / 2][NZ / 2] == sentinel_p);

    rho_f[NR / 2][NTH / 2][NZ / 2] = 0.0f;
    REQUIRE_THROWS_AS(
        scheme.compute_fast_pressure_tendencies(
            u_f, v_f, w_f, rho_f, p_f, /*dt_small=*/0.1,
            drho_dt, dp_dt),
        std::invalid_argument);
    REQUIRE(drho_dt[NR / 2][NTH / 2][NZ / 2] == sentinel_rho);
    REQUIRE(dp_dt[NR / 2][NTH / 2][NZ / 2] == sentinel_p);
}

TEST_CASE("CartesianAGridScheme propagates a periodic linear acoustic wave with refinement",
          "[dynamics][cartesian][agrid][acoustic][dispersion][energy][AG-004][AG-005]")
{
    struct WaveMetrics
    {
        double relative_l2 = 0.0;
        double relative_energy_drift = 0.0;
        double relative_mass_drift = 0.0;
    };
    const auto run_wave = [](int physical_nx)
    {
        constexpr double domain_length = 32000.0;
        constexpr double rho_ref = 1.0;
        constexpr double p_ref = 100000.0;
        constexpr double pressure_amplitude = 50.0;
        constexpr double acoustic_cfl = 0.2;
        const double sound_speed = std::sqrt(
            dynamics_constants::gamma * p_ref / rho_ref);

        NR = physical_nx + 2;
        NTH = 4;
        NZ = 4;
        dr = domain_length / physical_nx;
        dtheta = dr;
        dz = dr;
        dt = acoustic_cfl * dr / sound_speed;
        global_coordinate_system = CoordinateSystem::Cartesian;
        global_stagger_type = StaggerType::Collocated;
        global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
        global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
        global_lateral_bc_south = tmv::init::config::LateralBC::RigidSlip;
        global_lateral_bc_north = tmv::init::config::LateralBC::RigidSlip;
        global_bottom_bc = tmv::init::config::BottomBC::RigidSlip;
        global_top_bc = tmv::init::config::TopBC::Rigid;
        p0_base.assign(NZ, p_ref);
        rho0_base.assign(NZ, rho_ref);
        u0_base.assign(NZ, 0.0);
        v0_base.assign(NZ, 0.0);
        qv0_base.assign(NZ, 0.0);

        Field3D rho_f(NR, NTH, NZ, static_cast<float>(rho_ref));
        Field3D p_f(NR, NTH, NZ, static_cast<float>(p_ref));
        Field3D u_f(NR, NTH, NZ, 0.0f);
        Field3D v_f(NR, NTH, NZ, 0.0f);
        Field3D w_f(NR, NTH, NZ, 0.0f);
        Field3D rho_initial = rho_f;
        const double wave_number = 2.0 * 3.14159265358979323846
                                 / domain_length;
        for (int i = 1; i < NR - 1; ++i)
        {
            const double x = (static_cast<double>(i) - 1.0) * dr;
            const double p_prime = pressure_amplitude
                                 * std::cos(wave_number * x);
            const double rho_prime = p_prime / (sound_speed * sound_speed);
            const double u_prime = p_prime / (rho_ref * sound_speed);
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    rho_f[i][j][k] = static_cast<float>(rho_ref + rho_prime);
                    p_f[i][j][k] = static_cast<float>(p_ref + p_prime);
                    u_f[i][j][k] = static_cast<float>(u_prime);
                    rho_initial[i][j][k] = rho_f[i][j][k];
                }
        }
        const auto close_periodic_x = [](Field3D& field)
        {
            for (int j = 0; j < NTH; ++j)
                for (int k = 0; k < NZ; ++k)
                {
                    field[0][j][k] = field[NR - 2][j][k];
                    field[NR - 1][j][k] = field[1][j][k];
                }
        };
        close_periodic_x(rho_f);
        close_periodic_x(p_f);
        close_periodic_x(u_f);
        close_periodic_x(rho_initial);

        const auto mass = [](const Field3D& density)
        {
            long double value = 0.0L;
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 1; k < NZ - 1; ++k)
                        value += density[i][j][k];
            return value;
        };
        const auto energy = [=](const Field3D& density,
                                const Field3D& pressure,
                                const Field3D& velocity)
        {
            long double value = 0.0L;
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 1; k < NZ - 1; ++k)
                    {
                        const double u_prime = velocity[i][j][k];
                        const double p_prime = pressure[i][j][k] - p_ref;
                        value += 0.5 * rho_ref * u_prime * u_prime
                               + 0.5 * p_prime * p_prime
                                 / (rho_ref * sound_speed * sound_speed);
                    }
            return value;
        };

        const long double mass_initial = mass(rho_f);
        const long double energy_initial = energy(rho_f, p_f, u_f);
        Field3D drho_dt(NR, NTH, NZ, 0.0f);
        Field3D dp_dt(NR, NTH, NZ, 0.0f);
        Field3D du_dt(NR, NTH, NZ, 0.0f);
        Field3D dv_dt(NR, NTH, NZ, 0.0f);
        Field3D dw_dt(NR, NTH, NZ, 0.0f);
        CartesianAGridScheme scheme;
        const int step_count = 5 * physical_nx;
        for (int step = 0; step < step_count; ++step)
        {
            scheme.compute_fast_pressure_tendencies(
                u_f, v_f, w_f, rho_f, p_f, dt, drho_dt, dp_dt);
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 1; k < NZ - 1; ++k)
                    {
                        rho_f[i][j][k] += static_cast<float>(
                            dt * static_cast<double>(drho_dt[i][j][k]));
                        p_f[i][j][k] = static_cast<float>(
                            p_ref + sound_speed * sound_speed
                            * (static_cast<double>(rho_f[i][j][k]) - rho_ref));
                    }
            close_periodic_x(rho_f);
            close_periodic_x(p_f);

            scheme.compute_fast_momentum_tendencies(
                u_f, v_f, w_f, rho_f, p_f, dt, du_dt, dv_dt, dw_dt);
            for (int i = 1; i < NR - 1; ++i)
                for (int j = 1; j < NTH - 1; ++j)
                    for (int k = 1; k < NZ - 1; ++k)
                        u_f[i][j][k] += static_cast<float>(
                            dt * static_cast<double>(du_dt[i][j][k]));
            close_periodic_x(u_f);
        }

        long double error_squared = 0.0L;
        long double reference_squared = 0.0L;
        for (int i = 1; i < NR - 1; ++i)
        {
            const double actual = static_cast<double>(rho_f[i][1][1]) - rho_ref;
            const double expected = static_cast<double>(rho_initial[i][1][1])
                                  - rho_ref;
            error_squared += (actual - expected) * (actual - expected);
            reference_squared += expected * expected;
        }
        const long double mass_final = mass(rho_f);
        const long double energy_final = energy(rho_f, p_f, u_f);
        WaveMetrics metrics;
        metrics.relative_l2 = std::sqrt(static_cast<double>(
            error_squared / reference_squared));
        metrics.relative_energy_drift = std::abs(static_cast<double>(
            (energy_final - energy_initial) / energy_initial));
        metrics.relative_mass_drift = std::abs(static_cast<double>(
            (mass_final - mass_initial) / mass_initial));
        return metrics;
    };

    const WaveMetrics coarse = run_wave(16);
    const WaveMetrics medium = run_wave(32);
    const WaveMetrics fine = run_wave(64);
    INFO("A-grid acoustic one-period relative L2: "
         << coarse.relative_l2 << ", " << medium.relative_l2 << ", "
         << fine.relative_l2);
    INFO("A-grid acoustic energy drift: "
         << coarse.relative_energy_drift << ", "
         << medium.relative_energy_drift << ", "
         << fine.relative_energy_drift);
    INFO("A-grid acoustic mass drift: "
         << coarse.relative_mass_drift << ", "
         << medium.relative_mass_drift << ", "
         << fine.relative_mass_drift);
    REQUIRE(medium.relative_l2 < coarse.relative_l2);
    REQUIRE(fine.relative_l2 < medium.relative_l2);
    REQUIRE(coarse.relative_l2 / medium.relative_l2 > 3.0);
    REQUIRE(medium.relative_l2 / fine.relative_l2 > 3.0);
    REQUIRE(fine.relative_l2 < 0.02);
    REQUIRE(fine.relative_energy_drift < 0.01);
    REQUIRE(coarse.relative_mass_drift < 2.0e-7);
    REQUIRE(medium.relative_mass_drift < 2.0e-7);
    // The flux divergence telescopes exactly; this bound includes 320 FP32
    // state commits at the finest resolution.
    REQUIRE(fine.relative_mass_drift < 5.0e-7);
}

// ===========================================================================
// Gate 3: warm bubble produces the expected buoyancy acceleration
// ===========================================================================
TEST_CASE("CartesianAGridScheme: warm bubble produces correct buoyancy acceleration",
          "[dynamics][cartesian][buoyancy]")
{
    setup_cartesian_grid();

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity_field(u_f, v_f, w_f);

    // Apply a localized warm bubble at the geometric center of the domain.
    //
    // The bubble is "single-cell" — the bubble radius (1 km) is exactly
    // one horizontal grid step (dr = 1 km), so only the center cell has its
    // theta and rho overwritten. This is the strongest form of the test:
    // at the bubble center, the centered dp/dz stencil uses cells k-1 and
    // k+1 (both UNMODIFIED hydrostatic values), so dp/dz = -rho0*g exactly.
    // The center cell's rho is reduced to rho0 * (theta0 / (theta0 + dtheta))
    // to keep pressure consistent.
    //
    // Analytic result at the bubble center:
    //   dw/dt = -dp/dz / rho_warm - g
    //         =  rho0 g / (rho0 * theta0/theta_warm) - g
    //         =  g * (theta_warm / theta0 - 1)
    //         =  g * deltaTheta / theta0
    //
    // which is EXACTLY the 'Boussinesq buoyancy' formula the verification
    // gate asks for. The test tolerance (10 %) is slack for float roundoff.
    const int i_c = NR / 2;
    const int j_c = NTH / 2;
    const int k_c = NZ / 2;
    const double dtheta_bubble = 2.0;  // K
    const double new_theta = dynamics_constants::theta0 + dtheta_bubble;
    const double rho_warm  = kRho0 * (dynamics_constants::theta0 / new_theta);

    theta_f[i_c][j_c][k_c] = static_cast<float>(new_theta);
    rho_f[i_c][j_c][k_c]   = static_cast<float>(rho_warm);

    TendencyBuffers tb = make_tendency_buffers();

    CartesianAGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f,
        rho_f, p_f, theta_f,
        dt,
        tb.du_x_dt, tb.du_y_dt, tb.dw_dt,
        tb.drho_dt_out, tb.dp_dt_out);

    const double expected = dynamics_constants::g * dtheta_bubble /
                            dynamics_constants::theta0;
    const double observed = static_cast<double>(tb.dw_dt[i_c][j_c][k_c]);

    INFO("expected dw/dt at bubble center = " << expected << " m/s^2");
    INFO("observed dw/dt at bubble center = " << observed << " m/s^2");

    // Verification gate: within 10 % of g * deltaTheta / theta0.
    REQUIRE(std::abs(observed - expected) <= 0.10 * std::abs(expected));
    // And the sign must be upward — a warm parcel rises.
    REQUIRE(observed > 0.0);
}

// ===========================================================================
// Scheme metadata: the factory & base-interface queries
// ===========================================================================
TEST_CASE("CartesianAGridScheme reports its identity", "[dynamics][cartesian]")
{
    setup_cartesian_grid();
    CartesianAGridScheme scheme;
    REQUIRE(scheme.get_scheme_name() == "klemp_wilhelmson_split_cartesian_agrid");
    REQUIRE(scheme.get_coordinate_system() == "cartesian");
    REQUIRE(scheme.get_num_prognostic_vars() == 5);
}

TEST_CASE("Collocated dynamics diagnostics publish all Cartesian curl components for Ertel PV",
          "[dynamics][cartesian][agrid][vorticity][potential-vorticity]")
{
    setup_cartesian_grid();
    dtheta = 750.0;
    constexpr double du_dz = 0.004;
    constexpr double dv_dx = 0.006;
    constexpr double dw_dy = -0.003;

    Field3D u_f(NR, NTH, NZ, 0.0f);
    Field3D v_f(NR, NTH, NZ, 0.0f);
    Field3D w_f(NR, NTH, NZ, 0.0f);
    Field3D rho_f(NR, NTH, NZ, 1.0f);
    Field3D p_f(NR, NTH, NZ, 90000.0f);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u_f(i, j, k) = static_cast<float>(du_dz * k * dz);
                v_f(i, j, k) = static_cast<float>(dv_dx * i * dr);
                w_f(i, j, k) = static_cast<float>(dw_dy * j * dtheta);
            }

    SimulationStorageState storage(GridStorageState(NR, NTH, NZ, dr, dz, 1.0, dtheta));
    storage.grid.geometry.initialize(NR, NTH, NZ, dr, dz, dtheta, CoordinateSystem::Cartesian, StaggerType::Collocated);
    storage.fields.u   = u_f;
    storage.fields.v   = v_f;
    storage.fields.w   = w_f;
    storage.fields.rho = rho_f;
    storage.fields.p   = p_f;
    tmv::diagnostics::compute_resolved_diagnostics(make_simulation_state_view(storage), {true, false, false, false});
    const auto& vort_x     = storage.diagnostics.vorticity_r;
    const auto& vort_y     = storage.diagnostics.vorticity_theta;
    const auto& vort_z_f   = storage.diagnostics.vorticity_z;
    const auto& baroclinic = storage.diagnostics.baroclinic_term;

    const int i = NR / 2;
    const int j = NTH / 2;
    const int k = NZ / 2;
    CHECK(vort_x(i, j, k) == Approx(dw_dy).margin(2.0e-7));
    CHECK(vort_y(i, j, k) == Approx(du_dz).margin(2.0e-7));
    CHECK(vort_z_f(i, j, k) == Approx(dv_dx).margin(2.0e-7));
    CHECK(baroclinic(i, j, k) == Approx(0.0));
}

TEST_CASE("CartesianAGridScheme uses configured y spacing for pressure gradients",
          "[dynamics][cartesian][rectangular-grid]")
{
    setup_cartesian_grid();
    dtheta = 250.0;

    Field3D u_f, v_f, w_f, rho_f, p_f;
    u_f.resize(NR, NTH, NZ, 0.0f);
    v_f.resize(NR, NTH, NZ, 0.0f);
    w_f.resize(NR, NTH, NZ, 0.0f);
    rho_f.resize(NR, NTH, NZ, static_cast<float>(kRho0));
    p_f.resize(NR, NTH, NZ, 0.0f);

    constexpr double pressure_gradient_y = 2.0;  // Pa/m
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p_f[i][j][k] = static_cast<float>(
                    p0_base[k] + pressure_gradient_y * j * dtheta);

    Field3D du_dt, dv_dt, dw_dt;
    du_dt.resize(NR, NTH, NZ, 0.0f);
    dv_dt.resize(NR, NTH, NZ, 0.0f);
    dw_dt.resize(NR, NTH, NZ, 0.0f);
    CartesianAGridScheme scheme;
    scheme.compute_fast_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, /*dt_small=*/0.1,
        du_dt, dv_dt, dw_dt);

    const double expected_dv_dt = -pressure_gradient_y / kRho0;
    INFO("expected dv/dt = " << expected_dv_dt
         << ", actual = " << dv_dt[NR / 2][NTH / 2][NZ / 2]);
    REQUIRE(dv_dt[NR / 2][NTH / 2][NZ / 2]
            == Approx(expected_dv_dt).epsilon(1.0e-4));
}

TEST_CASE("Cartesian A-grid beta-plane uses Earth-referenced cell-center f",
          "[dynamics][cartesian][agrid][coriolis][beta_plane]")
{
    setup_cartesian_grid();
    Field3D u_f, v_f, w_f, rho_f, p_f, theta_f;
    uniform_horizontal_wind(u_f, v_f, w_f, 10.0, 0.0);
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);

    constexpr double f_y0 = 8.0e-5;
    constexpr double beta = 2.0e-11;
    CartesianAGridScheme scheme;
    scheme.set_coriolis_profile({f_y0, beta});
    TendencyBuffers tb = make_tendency_buffers();
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du_x_dt, tb.du_y_dt, tb.dw_dt,
        tb.drho_dt_out, tb.dp_dt_out);

    const int i = NR / 2;
    const int k = NZ / 2;
    for (const int j : {2, NTH - 3})
    {
        const double local_f = f_y0 + beta * (j + 0.5) * dtheta;
        CHECK(tb.du_x_dt(i, j, k) == Approx(0.0).margin(1.0e-12));
        CHECK(tb.du_y_dt(i, j, k)
              == Approx(-10.0 * local_f).epsilon(2.0e-6));
    }
    CHECK(tb.du_y_dt(i, 2, k) != tb.du_y_dt(i, NTH - 3, k));
}

TEST_CASE("Cartesian A-grid Coriolis is pointwise work-free and closure-safe",
          "[dynamics][cartesian][agrid][coriolis][energy][closure]")
{
    setup_cartesian_grid();
    Field3D u_f, v_f, w_f, rho_f, p_f, theta_f;
    uniform_horizontal_wind(u_f, v_f, w_f, 10.0, 5.0);
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);

    constexpr double f_y0 = 8.0e-5;
    constexpr double beta = 2.0e-11;
    CartesianAGridScheme scheme;
    scheme.set_coriolis_profile({f_y0, beta});
    TendencyBuffers tb = make_tendency_buffers();
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du_x_dt, tb.du_y_dt, tb.dw_dt,
        tb.drho_dt_out, tb.dp_dt_out);

    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const bool physical = i > 0 && i < NR - 1
                    && j > 0 && j < NTH - 1
                    && k > 0 && k < NZ - 1;
                if (!physical)
                {
                    CHECK(tb.du_x_dt(i, j, k) == 0.0f);
                    CHECK(tb.du_y_dt(i, j, k) == 0.0f);
                    continue;
                }
                const double local_f = f_y0 + beta * (j + 0.5) * dtheta;
                CHECK(tb.du_x_dt(i, j, k)
                      == Approx(5.0 * local_f).epsilon(2.0e-6));
                CHECK(tb.du_y_dt(i, j, k)
                      == Approx(-10.0 * local_f).epsilon(2.0e-6));
                const double work =
                    10.0 * static_cast<double>(tb.du_x_dt(i, j, k))
                    + 5.0 * static_cast<double>(tb.du_y_dt(i, j, k));
                CHECK(std::abs(work) <= 5.0e-10);
            }
}
