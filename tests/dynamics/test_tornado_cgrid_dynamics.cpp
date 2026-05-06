/**
 * @file test_tornado_cgrid_dynamics.cpp
 * @brief Verification gates for Phase C.4 -- C-grid axisymmetric tornado
 *        dynamics scheme (TornadoCGridScheme).
 *
 * Verification gates from docs/CoordinateBackend_Plan.md C.4:
 *
 *   1. Hydrostatic equilibrium, 300s, tendencies < 1e-10.
 *      -> Tendencies at t=0 are at machine-noise level (float-storage ULP)
 *      -> 300 Euler steps preserve the state bit-exactly.
 *
 *   2. Lamb-Oseen vortex, cyclostrophic drift < 0.1% over 60s.
 *      -> Discrete cyclostrophic init makes du/dt vanish to float roundoff
 *         everywhere (the centrifugal-vs-pressure-gradient cancellation is
 *         exact in this discretization)
 *      -> dv/dt, drho/dt, dp/dt are bit-exactly zero at balance
 *      -> 120 Euler steps (dt=0.5s) hold the v profile within 0.1% of v_max
 *
 *   3. C-grid axis advantage. The C-grid scheme computes du/dt at i=0 (the
 *      first interior r-face at r = 0.5*dr) using the antisymmetric ghost
 *      u[-1] = -u[0]. With u = 0 at cyclostrophic balance, this gives
 *      du/dt[0] = float-roundoff (no spurious-divergence term and no
 *      antisymmetric-BC artifact -- there is no antisymmetric BC writing
 *      u[0] = -u[1] like the collocated grid does).
 *
 * Test strategy:
 *
 *   The dynamics scheme is exercised directly by instantiating a
 *   TornadoCGridScheme and calling compute_momentum_tendencies on local
 *   Field3D buffers, mirroring tests/dynamics/test_cartesian_dynamics.cpp.
 *
 *   For the Lamb-Oseen drift test, the Euler loop applies a partial BC --
 *   only the outer-wall u = 0 and lid w = 0 -- because the cylindrical
 *   C-grid axis BC zero-gradient on pressure (p[0] = p[1]) is the standard
 *   non-rotating-flow choice and is incompatible with cyclostrophic balance
 *   at the first interior r-face. The dynamics scheme handles the axis
 *   ghost inline (u[-1] = -u[0]); no axis BC application is required for
 *   the dynamics step itself.
 */

#include "catch2/catch.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/field/field3d.hpp"
#include "core/infra/grid_geometry.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "dynamics/dynamics_base.hpp"
#include "dynamics/schemes/tornado/tornado_cgrid.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{

constexpr double kPi   = 3.14159265358979323846;
// kRho0 chosen so that float and double representations agree bit-exactly
// (1.25 = 1 + 1/4, exact in binary).  This is what lets the hydrostatic
// equilibrium gate close to machine precision: with rho_f[k] == rho0_base[k]
// in both storage formats, the perturbation-density buoyancy term
// -g*(rho_face - rho0_face)/rho_safe cancels to bit-exact zero.
constexpr double kRho0 = 1.25;        // kg/m^3 (exact in float and double)
constexpr double kP0   = 101325.0;    // Pa  (exact in float and double)


/// @brief Initializes the grid globals + reference profiles (rho0_base,
///        p0_base, qv0_base) for the C-grid cylindrical configuration.
void setup_cgrid_grid(int nr, int nth, int nz, double dr_v, double dz_v)
{
    NR     = nr;
    NTH    = nth;
    NZ     = nz;
    dr     = dr_v;
    dz     = dz_v;
    dt     = 0.1;
    dtheta = 2.0 * kPi / static_cast<double>(NTH);

    global_coordinate_system = CoordinateSystem::Cylindrical;
    global_stagger_type      = StaggerType::CGrid;
    global_grid_geometry.initialize(NR, NTH, NZ, dr, dz, dtheta,
                                    global_coordinate_system,
                                    global_stagger_type);

    rho0_base.assign(NZ, kRho0);
    p0_base.resize(NZ);
    for (int k = 0; k < NZ; ++k)
        p0_base[k] = kP0 - kRho0 * dynamics_constants::g
                          * static_cast<double>(k) * dz;
    u0_base.assign(NZ, 0.0);
    v0_base.assign(NZ, 0.0);
    qv0_base.clear();  // dry test; tornado_cgrid skips moisture branches
}


void initialize_hydrostatic_atmosphere(Field3D& rho_f, Field3D& p_f)
{
    rho_f.resize(NR, NTH, NZ, static_cast<float>(kRho0));
    p_f.resize(NR, NTH, NZ);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                p_f[i][j][k] = static_cast<float>(p0_base[k]);

    // Make rho0_base / p0_base bit-exactly match the float storage of
    // rho_f / p_f. Without this round-trip, computing p0_base[k] as
    // kP0 - kRho0 * g * k * dz cell-by-cell in double leaves a residual
    // ~k * ULP(double) in the diff (p0_base[k+1] - p0_base[k]) that does
    // not match the float diff (p_f[k+1] - p_f[k]). The mismatch produces
    // a small, k-dependent dp_prime_dz residual in the dynamics, which in
    // turn drives a tiny dw/dt, which integrates into a w_face accumulation,
    // which closes the loop into a slowly-growing dp/dt via -gamma*p*div(w).
    // Round-tripping at init makes the discrete reference state self-
    // consistent with the float-stored full state, so the perturbation
    // terms in the vertical momentum and pressure equations vanish to
    // bit-exact zero.
    for (int k = 0; k < NZ; ++k)
    {
        rho0_base[k] = static_cast<double>(rho_f[0][0][k]);
        p0_base[k]   = static_cast<double>(p_f[0][0][k]);
    }
}


void zero_velocity(Field3D& u_f, Field3D& v_f, Field3D& w_f)
{
    u_f.resize(NR, NTH, NZ, 0.0f);
    v_f.resize(NR, NTH, NZ, 0.0f);
    w_f.resize(NR, NTH, NZ, 0.0f);
}


void initialize_isothermal_theta(Field3D& theta_f)
{
    theta_f.resize(NR, NTH, NZ,
                   static_cast<float>(dynamics_constants::theta0));
}


struct TendencyBuffers
{
    Field3D du, dv, dw, drho, dp;
};

TendencyBuffers make_tendency_buffers()
{
    TendencyBuffers tb;
    tb.du.resize(NR, NTH, NZ, 0.0f);
    tb.dv.resize(NR, NTH, NZ, 0.0f);
    tb.dw.resize(NR, NTH, NZ, 0.0f);
    tb.drho.resize(NR, NTH, NZ, 0.0f);
    tb.dp.resize(NR, NTH, NZ, 0.0f);
    return tb;
}


/// @brief Returns the largest absolute value over all (i, j, k) cells.
double max_abs_all(const Field3D& f)
{
    double m = 0.0;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                const double v = std::abs(static_cast<double>(f[i][j][k]));
                if (v > m) m = v;
            }
    return m;
}


/// @brief Returns the largest absolute value over the cells the C-grid
///        scheme actually computes for cell-center quantities (drho/dt,
///        dp/dt: i = 1..NR-2, k = 1..NZ-2).
double max_abs_center_interior(const Field3D& f)
{
    double m = 0.0;
    for (int i = 1; i < NR - 1; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 1; k < NZ - 1; ++k)
            {
                const double v = std::abs(static_cast<double>(f[i][j][k]));
                if (v > m) m = v;
            }
    return m;
}


/// @brief Lamb-Oseen vortex tangential velocity (cylindrical).
///        v(r) = (Gamma / (2 pi r)) * (1 - exp(-r^2 / r_c^2))
///        With v(0) = 0 (limiting value of the smooth profile).
double lamb_oseen_v(double r, double Gamma, double r_c)
{
    if (r <= 0.0) return 0.0;
    return (Gamma / (2.0 * kPi * r)) * (1.0 - std::exp(-(r * r) / (r_c * r_c)));
}


/// @brief Initializes a Lamb-Oseen vortex (axisymmetric in v) and the
///        cyclostrophic-balanced p(r,z) such that the discrete
///        StaggeredCylindricalDerivatives::grad_r(p) at every r-face equals
///        rho_face * v_face^2 / r_face[i] EXACTLY (modulo float-storage
///        roundoff). This makes the centrifugal-vs-pressure-gradient
///        cancellation at every u-face cell precise.
void initialize_lamb_oseen(double Gamma, double r_c,
                           Field3D& rho_f, Field3D& p_f, Field3D& v_f)
{
    rho_f.resize(NR, NTH, NZ, static_cast<float>(kRho0));
    v_f.resize(NR, NTH, NZ, 0.0f);
    p_f.resize(NR, NTH, NZ);

    for (int i = 0; i < NR; ++i)
    {
        const double v_i = lamb_oseen_v(global_grid_geometry.r[i], Gamma, r_c);
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
                v_f[i][j][k] = static_cast<float>(v_i);
    }

    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
        {
            // Anchor at axis (i=0): hydrostatic pressure with no cyclostrophic
            // contribution (v(r=0) = 0).
            p_f[0][j][k] = static_cast<float>(p0_base[k]);

            // Walk outward, accumulating the cyclostrophic pressure offset.
            for (int i = 0; i < NR - 1; ++i)
            {
                const double v_i      = lamb_oseen_v(global_grid_geometry.r[i],     Gamma, r_c);
                const double v_ip1    = lamb_oseen_v(global_grid_geometry.r[i + 1], Gamma, r_c);
                const double v_face   = 0.5 * (v_i + v_ip1);
                const double r_face   = global_grid_geometry.r_face[i];
                const double rho_face = kRho0;
                const double dp_cyclo = rho_face * v_face * v_face / r_face * dr;
                p_f[i + 1][j][k] = static_cast<float>(
                    static_cast<double>(p_f[i][j][k]) + dp_cyclo);
            }
        }
}


/// @brief Applies the partial boundary condition appropriate for the
///        cyclostrophic Lamb-Oseen drift test: outer-wall u = 0, lid w = 0.
///        See file-level rationale.
void apply_partial_bc(Field3D& u_f, Field3D& w_f)
{
    for (int j = 0; j < NTH; ++j)
        for (int k = 0; k < NZ; ++k)
            u_f[NR - 1][j][k] = 0.0f;
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            w_f[i][j][NZ - 1] = 0.0f;
}


/// @brief Forward-Euler integrator for one step of the prognostic state.
void euler_step(double step_dt,
                Field3D& u_f, Field3D& v_f, Field3D& w_f,
                Field3D& rho_f, Field3D& p_f,
                const TendencyBuffers& T)
{
    const float dt_f = static_cast<float>(step_dt);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u_f[i][j][k]   += dt_f * T.du[i][j][k];
                v_f[i][j][k]   += dt_f * T.dv[i][j][k];
                w_f[i][j][k]   += dt_f * T.dw[i][j][k];
                rho_f[i][j][k] += dt_f * T.drho[i][j][k];
                p_f[i][j][k]   += dt_f * T.dp[i][j][k];
            }
}


/// @brief Forward-Euler integrator that updates only momentum (u, v, w),
///        leaving rho and p frozen.  Used by the Lamb-Oseen drift test:
///        Forward Euler is unconditionally unstable for the acoustic wave
///        equation (dp/dt = -gamma p div(u), du/dt = -grad p/rho), so the
///        drift gate isolates the slow dynamics from acoustics. The
///        production split_explicit time stepping handles the acoustic
///        substep with a stable Klemp-Wilhelmson scheme; that integration
///        is exercised in the supercell_cgrid C.6 work.
void euler_step_momentum_only(double step_dt,
                              Field3D& u_f, Field3D& v_f, Field3D& w_f,
                              const TendencyBuffers& T)
{
    const float dt_f = static_cast<float>(step_dt);
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                u_f[i][j][k] += dt_f * T.du[i][j][k];
                v_f[i][j][k] += dt_f * T.dv[i][j][k];
                w_f[i][j][k] += dt_f * T.dw[i][j][k];
            }
}

}  // namespace


// ============================================================================
// Gate 1 -- Hydrostatic equilibrium: tendencies at t=0 are at machine noise.
//
// The verification gate "tendencies < 1e-10" is achievable here because the
// kRho0 * g * dz step (1.225 * 9.81 * 250 = 3004.3125) is exactly
// representable in float32 (21 mantissa bits), so static_cast<float> of the
// hydrostatic profile p_z(z) preserves the slope -kRho0*g exactly when
// taking adjacent-cell differences.  All other tendency contributions are
// proportional to u, v, w (zero) or rho - rho0_base[k] (zero).
// ============================================================================
TEST_CASE("TornadoCGridScheme: hydrostatic equilibrium has machine-zero tendencies",
          "[dynamics][cylindrical][cgrid][c4]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    TendencyBuffers tb = make_tendency_buffers();
    TornadoCGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

    const double kTol = 1.0e-10;
    INFO("max |du/dt| = "   << max_abs_all(tb.du));
    INFO("max |dv/dt| = "   << max_abs_all(tb.dv));
    INFO("max |dw/dt| = "   << max_abs_all(tb.dw));
    INFO("max |drho/dt| = " << max_abs_all(tb.drho));
    INFO("max |dp/dt| = "   << max_abs_all(tb.dp));
    REQUIRE(max_abs_all(tb.du)   <= kTol);
    REQUIRE(max_abs_all(tb.dv)   <= kTol);
    REQUIRE(max_abs_all(tb.dw)   <= kTol);
    REQUIRE(max_abs_all(tb.drho) <= kTol);
    REQUIRE(max_abs_all(tb.dp)   <= kTol);
}


// ============================================================================
// Gate 1b -- Hydrostatic equilibrium preserved over 300 Euler steps.
// ============================================================================
TEST_CASE("TornadoCGridScheme: hydrostatic state preserved over 300 Euler steps",
          "[dynamics][cylindrical][cgrid][c4]")
{
    setup_cgrid_grid(/*nr=*/16, /*nth=*/8, /*nz=*/16,
                     /*dr=*/250.0, /*dz=*/250.0);

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_hydrostatic_atmosphere(rho_f, p_f);
    initialize_isothermal_theta(theta_f);
    zero_velocity(u_f, v_f, w_f);

    Field3D rho_snap = rho_f;
    Field3D p_snap   = p_f;

    TendencyBuffers tb = make_tendency_buffers();
    TornadoCGridScheme scheme;

    const double step_dt = 1.0;
    const int    n_steps = 300;
    const double kTol    = 1.0e-10;

    for (int step = 0; step < n_steps; ++step)
    {
        scheme.compute_momentum_tendencies(
            u_f, v_f, w_f, rho_f, p_f, theta_f, step_dt,
            tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

        // Tendencies must remain at machine noise at every step.  See the
        // initialize_hydrostatic_atmosphere round-trip note for why this
        // closes to bit-exact zero with the 1.25 kg/m^3 / 250 m setup.
        INFO("step=" << step
             << " max|du|="   << max_abs_all(tb.du)
             << " max|dv|="   << max_abs_all(tb.dv)
             << " max|dw|="   << max_abs_all(tb.dw)
             << " max|drho|=" << max_abs_all(tb.drho)
             << " max|dp|="   << max_abs_all(tb.dp));
        REQUIRE(max_abs_all(tb.du)   <= kTol);
        REQUIRE(max_abs_all(tb.dv)   <= kTol);
        REQUIRE(max_abs_all(tb.dw)   <= kTol);
        REQUIRE(max_abs_all(tb.drho) <= kTol);
        REQUIRE(max_abs_all(tb.dp)   <= kTol);

        euler_step(step_dt, u_f, v_f, w_f, rho_f, p_f, tb);
    }

    // Bit-exact preservation of rho and p; u, v, w stay identically zero.
    for (int i = 0; i < NR; ++i)
        for (int j = 0; j < NTH; ++j)
            for (int k = 0; k < NZ; ++k)
            {
                REQUIRE(rho_f[i][j][k] == rho_snap[i][j][k]);
                REQUIRE(p_f[i][j][k]   == p_snap[i][j][k]);
                REQUIRE(u_f[i][j][k] == 0.0f);
                REQUIRE(v_f[i][j][k] == 0.0f);
                REQUIRE(w_f[i][j][k] == 0.0f);
            }
}


// ============================================================================
// Gate 2a -- Lamb-Oseen vortex at discrete cyclostrophic balance: at t=0,
// du/dt is at float-roundoff and dv/dt, drho/dt, dp/dt are bit-exactly zero.
//
// At cyclostrophic balance with u = w = 0:
//   - centrifugal (v_face^2/r_face) cancels pressure_grad_r (-dp_dr/rho_face)
//     to within float storage roundoff (the construction makes the discrete
//     dp_cyclo equal exactly to rho*v_face^2/r_face*dr modulo cast).
//   - dv/dt = -u*dv/dr - w*dv/dz - u*v/r = 0  exactly (u=w=0).
//   - drho/dt = -rho*div_flux(u=0, v_axisym, w=0) = 0 exactly.
//   - dp/dt = -gamma*p*0 - 0 - 0 = 0 exactly.
// ============================================================================
TEST_CASE("TornadoCGridScheme: Lamb-Oseen cyclostrophic balance is discretely exact",
          "[dynamics][cylindrical][cgrid][c4][lamb-oseen]")
{
    setup_cgrid_grid(/*nr=*/32, /*nth=*/8, /*nz=*/12,
                     /*dr=*/200.0, /*dz=*/250.0);

    // Vortex parameters: r_c = 10*dr (well-resolved core away from axis),
    // peak v_max = 50 m/s near r ~ r_c.
    const double r_c   = 10.0 * dr;
    const double v_max = 50.0;
    // v_max at the LO peak is 0.715 * Gamma / (2 pi r_c).
    const double Gamma = v_max * 2.0 * kPi * r_c / 0.715;

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_lamb_oseen(Gamma, r_c, rho_f, p_f, v_f);
    initialize_isothermal_theta(theta_f);
    u_f.resize(NR, NTH, NZ, 0.0f);
    w_f.resize(NR, NTH, NZ, 0.0f);

    TendencyBuffers tb = make_tendency_buffers();
    TornadoCGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

    // Exactly zero (no float-storage cancellation at all).
    REQUIRE(max_abs_all(tb.dv)   == 0.0);
    REQUIRE(max_abs_center_interior(tb.drho) == 0.0);
    REQUIRE(max_abs_center_interior(tb.dp)   == 0.0);

    // Float roundoff of cyclostrophic cancellation at u-faces.
    // For v_max=50, r=1000m, dr=200m, p~1e5: float32 ULP at p ~ 0.008 Pa,
    // dp_dr roundoff ~ 0.008/200 = 4e-5 Pa/m, du/dt residual ~ 3e-5 m/s^2.
    // Tolerance 1e-3 leaves 30x margin.
    INFO("max |du/dt| at cyclostrophic balance = " << max_abs_all(tb.du));
    REQUIRE(max_abs_all(tb.du) < 1.0e-3);

    // Vertical: dp_prime_dz residual from float storage of the (purely-radial)
    // pressure perturbation should be similarly small. Buoyancy is exactly
    // zero (rho = rho0). Tolerance 1e-3 leaves ample margin over expected
    // ~1e-5 m/s^2.
    INFO("max |dw/dt| at cyclostrophic balance = " << max_abs_all(tb.dw));
    REQUIRE(max_abs_all(tb.dw) < 1.0e-3);
}


// ============================================================================
// Gate 2b -- Lamb-Oseen vortex profile preserved over 60 simulated seconds.
//
// 120 Forward-Euler steps with dt = 0.5s. Partial BC (outer wall u=0, lid
// w=0) preserves the cyclostrophic balance on the axis (the C-grid axis BC
// zero-gradient on p that boundary_conditions_cylindrical_cgrid.cpp applies
// is intentionally skipped here -- see file-level docstring). The dynamics
// scheme handles the axis ghost u[-1] = -u[0] inline, so no axis-side BC
// application is required for the dynamics step itself.
// ============================================================================
TEST_CASE("TornadoCGridScheme: Lamb-Oseen vortex drifts < 0.1% over 60 sim seconds",
          "[dynamics][cylindrical][cgrid][c4][lamb-oseen]")
{
    setup_cgrid_grid(/*nr=*/32, /*nth=*/8, /*nz=*/12,
                     /*dr=*/200.0, /*dz=*/250.0);

    const double r_c   = 10.0 * dr;
    const double v_max = 50.0;
    const double Gamma = v_max * 2.0 * kPi * r_c / 0.715;

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_lamb_oseen(Gamma, r_c, rho_f, p_f, v_f);
    initialize_isothermal_theta(theta_f);
    u_f.resize(NR, NTH, NZ, 0.0f);
    w_f.resize(NR, NTH, NZ, 0.0f);

    // Snapshot the initial v profile at the probe column.
    const int j_probe = 0;
    const int k_probe = NZ / 2;
    std::vector<float> v0_profile(NR);
    for (int i = 0; i < NR; ++i)
        v0_profile[i] = v_f[i][j_probe][k_probe];

    TendencyBuffers tb = make_tendency_buffers();
    TornadoCGridScheme scheme;

    const double step_dt = 0.5;
    const int    n_steps = 120;  // 60 simulated seconds

    for (int step = 0; step < n_steps; ++step)
    {
        scheme.compute_momentum_tendencies(
            u_f, v_f, w_f, rho_f, p_f, theta_f, step_dt,
            tb.du, tb.dv, tb.dw, tb.drho, tb.dp);
        euler_step_momentum_only(step_dt, u_f, v_f, w_f, tb);
        apply_partial_bc(u_f, w_f);
    }

    double max_drift = 0.0;
    int    i_argmax  = 0;
    for (int i = 0; i < NR; ++i)
    {
        const double drift = std::abs(static_cast<double>(v_f[i][j_probe][k_probe])
                                    - static_cast<double>(v0_profile[i]));
        if (drift > max_drift)
        {
            max_drift = drift;
            i_argmax  = i;
        }
    }

    INFO("max v drift = " << max_drift << " m/s at i = " << i_argmax
         << " (v_max = " << v_max << " m/s)");
    INFO("relative drift = " << max_drift / v_max);
    REQUIRE(max_drift / v_max < 1.0e-3);  // 0.1% gate
}


// ============================================================================
// Gate 3 -- C-grid axis advantage. With u = 0 at cyclostrophic balance,
// du/dt at the first interior r-face (i=0, r = 0.5*dr) is at float-roundoff:
// the discrete centrifugal force v_face[0]^2 / r_face[0] is exactly cancelled
// by the one-sided pressure gradient (p[1] - p[0]) / dr, just like at every
// other r-face. The collocated tornado scheme cannot make this claim because
// its antisymmetric BC u[0] = -u[1] is a hack on a stored field that drives
// a spurious gradient at i=1.
// ============================================================================
TEST_CASE("TornadoCGridScheme: axis r-face has no spurious tendency at balance",
          "[dynamics][cylindrical][cgrid][c4][lamb-oseen][axis]")
{
    setup_cgrid_grid(/*nr=*/32, /*nth=*/8, /*nz=*/12,
                     /*dr=*/200.0, /*dz=*/250.0);

    const double r_c   = 10.0 * dr;
    const double v_max = 50.0;
    const double Gamma = v_max * 2.0 * kPi * r_c / 0.715;

    Field3D rho_f, p_f, theta_f, u_f, v_f, w_f;
    initialize_lamb_oseen(Gamma, r_c, rho_f, p_f, v_f);
    initialize_isothermal_theta(theta_f);
    u_f.resize(NR, NTH, NZ, 0.0f);
    w_f.resize(NR, NTH, NZ, 0.0f);

    TendencyBuffers tb = make_tendency_buffers();
    TornadoCGridScheme scheme;
    scheme.compute_momentum_tendencies(
        u_f, v_f, w_f, rho_f, p_f, theta_f, dt,
        tb.du, tb.dv, tb.dw, tb.drho, tb.dp);

    // Confirm the axis r-face residual is similar in magnitude to a
    // non-axis r-face residual: the axis stencil has no special-cased
    // "antisymmetric ghost cell artifact" amplification.
    double axis_res = 0.0;
    for (int j = 0; j < NTH; ++j)
        for (int k = 1; k < NZ - 1; ++k)
            axis_res = std::max(axis_res, std::abs(static_cast<double>(tb.du[0][j][k])));

    double mid_res = 0.0;
    const int i_mid = NR / 2;
    for (int j = 0; j < NTH; ++j)
        for (int k = 1; k < NZ - 1; ++k)
            mid_res = std::max(mid_res, std::abs(static_cast<double>(tb.du[i_mid][j][k])));

    INFO("max |du/dt[0]|       = " << axis_res);
    INFO("max |du/dt[NR/2]|    = " << mid_res);

    // Both should be at the float-storage cancellation floor.
    REQUIRE(axis_res < 1.0e-3);
    REQUIRE(mid_res  < 1.0e-3);

    // The axis residual must not dwarf the interior residual by an order of
    // magnitude or more (which would betray a special axis artifact). Allow
    // a 100x margin since float roundoff is non-uniform across the grid.
    if (mid_res > 0.0)
    {
        const double ratio = axis_res / mid_res;
        INFO("axis_res / mid_res = " << ratio);
        REQUIRE(ratio < 100.0);
    }
}


// ============================================================================
// Scheme metadata sanity check.
// ============================================================================
TEST_CASE("TornadoCGridScheme: scheme metadata reports cylindrical_cgrid",
          "[dynamics][cylindrical][cgrid][c4]")
{
    setup_cgrid_grid(8, 4, 8, 250.0, 250.0);
    TornadoCGridScheme scheme;
    REQUIRE(scheme.get_scheme_name()        == "tornado_cgrid");
    REQUIRE(scheme.get_coordinate_system()  == "cylindrical_cgrid");
    REQUIRE(scheme.get_num_prognostic_vars() == 5);
}
