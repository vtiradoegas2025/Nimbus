/**
 * @file test_cartesian_dynamics.cpp
 * @brief Unit tests for the Cartesian dynamics scheme (Phase A.2 gates).
 *
 * Verification gates from docs/CoordinateBackend_Plan.md §A.2:
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
 *   - Drive the scheme directly by instantiating a `CartesianScheme` and
 *     calling `compute_momentum_tendencies` on local Field3D buffers. The
 *     grid globals (NR, NTH, NZ, dr, dz, dtheta, rho0_base) come from the
 *     test_harness; the test sets them to small workable values in each
 *     SECTION, matching the pattern used by tests/numerics/test_advection_tvd.cpp.
 */

#include "catch2/catch.hpp"
#include "core/field3d.hpp"
#include "core/simulation.hpp"
#include "dynamics/schemes/cartesian/cartesian.hpp"
#include "dynamics/dynamics_base.hpp"

#include <cmath>
#include <vector>

namespace
{

// Sea-level air density and reference pressure for the synthetic atmosphere.
// These are chosen so the hydrostatic profile p(z) = p0 - rho0*g*z stays
// positive over the vertical extent of the test grid (NZ*dz = 8 km here).
constexpr double kRho0 = 1.225;       // kg / m^3
constexpr double kP0   = 101325.0;    // Pa

/**
 * @brief Initializes the grid globals used by CartesianScheme's constructor.
 *        Called by every test case because tests may have mutated them.
 */
void setup_cartesian_grid()
{
    NR = 16;
    NTH = 16;
    NZ = 16;
    dr = 1000.0;   // 1 km horizontal (dx = dy = dr in Phase A, square cells)
    dz = 500.0;    // 0.5 km vertical
    dt = 0.1;
    dtheta = 0.0;  // unused by the Cartesian scheme

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
TEST_CASE("CartesianScheme: hydrostatic equilibrium has near-zero tendencies",
          "[dynamics][cartesian]")
{
    setup_cartesian_grid();

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity_field(u_f, v_f, w_f);

    TendencyBuffers tb = make_tendency_buffers();

    CartesianScheme scheme;
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
TEST_CASE("CartesianScheme: uniform Cartesian wind preserves equilibrium (Bug 7 proof)",
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

    CartesianScheme scheme;
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
    // a false dp/dt, and breaks hydrostatic balance. See docs/Journey.md
    // Bug 7 and docs/CoordinateBackend_Plan.md for the full writeup.
    const double kTol = 1.0e-3;
    REQUIRE(max_abs_interior(tb.du_x_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.du_y_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.dw_dt) <= kTol);
    REQUIRE(max_abs_interior(tb.drho_dt_out) <= kTol);
    REQUIRE(max_abs_interior(tb.dp_dt_out) <= kTol);
}

// ===========================================================================
// Gate 3: warm bubble produces the expected buoyancy acceleration
// ===========================================================================
TEST_CASE("CartesianScheme: warm bubble produces correct buoyancy acceleration",
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

    CartesianScheme scheme;
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
TEST_CASE("CartesianScheme reports its identity", "[dynamics][cartesian]")
{
    setup_cartesian_grid();
    CartesianScheme scheme;
    REQUIRE(scheme.get_scheme_name() == "cartesian");
    REQUIRE(scheme.get_coordinate_system() == "cartesian");
    REQUIRE(scheme.get_num_prognostic_vars() == 5);
}
