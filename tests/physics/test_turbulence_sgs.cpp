/**
 * @file test_turbulence_sgs.cpp
 * @brief Smagorinsky SGS strain + tendency tests (SCIENCE.md M14, M15).
 *
 * M14: |S|^2 is built at the cell center from STAGGERED-NATURAL face
 * differences - the diagonals from the cell's own faces, the off-diagonals
 * from the four surrounding edges with their squares averaged - so a
 * single-cell (2dx) shear the old collocated centered read was blind to is
 * now resolved.
 *
 * M15: the SGS momentum/scalar mixing is a CONSERVATIVE stress / flux
 * divergence with live-density weighting and zero-flux ground/lid closures.
 * Two properties follow and are pinned here:
 *   - free-stream: a uniform field produces no tendency, for any nu/K
 *     (every strain/gradient is exactly zero);
 *   - conservation: the domain integral of the density-weighted tendency
 *     vanishes (the flux form telescopes; with the active flow held away
 *     from the boundaries every surviving boundary stress is zero). A
 *     non-conservative local-nu Laplacian fails this wherever grad(nu)
 *     is nonzero - exactly the cloud/outflow edges the audit flagged.
 */

#include "catch2/catch.hpp"
#include "turbulence/turbulence_base.hpp"
#include "core/runtime/simulation_state.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "turbulence/schemes/smagorinsky/dynamic_smagorinsky.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"   // rho0_base (M15 density weighting)
#include "core/infra/coordinate_system.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/field/field3d.hpp"
#include "terrain/conservative_transport.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

constexpr int kN = 12;
Field3D live_rho;

// Cartesian, open lateral y: periodic_y_topology() == false, so the SGS
// stencils skip the j edge rows exactly like the i edges. The conservation
// tests rely on this: the flux form telescopes to the domain-boundary
// faces, and the active flow is held >= 4 cells from every boundary so
// those faces carry zero stress.
void set_cartesian_open()
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_west  = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_east  = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
    global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

void set_cartesian_periodic()
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    global_top_bc = tmv::init::config::TopBC::Rigid;
}

// Stratified base state plus a horizontally perturbed live density. The latter
// distinguishes conservation of the prognostic currency from the obsolete
// rho0-weighted approximation.
void set_stratified_rho0()
{
    rho0_base.assign(static_cast<std::size_t>(kN), 1.0);
    live_rho.resize(kN, kN, kN, 1.0f);
    for (int k = 0; k < kN; ++k)
    {
        rho0_base[static_cast<std::size_t>(k)] =
            1.2 * std::exp(-static_cast<double>(k) / 10.0);
        for (int i = 0; i < kN; ++i)
            for (int j = 0; j < kN; ++j)
                live_rho(i, j, k) = static_cast<float>(
                    rho0_base[static_cast<std::size_t>(k)]
                    * (0.8 + 0.025 * i + 0.015 * j));
    }
}

double rho_live_c(int i, int j, int k)
{
    return static_cast<double>(live_rho(i, j, k));
}

double rho_live_xface(int i, int j, int k)
{
    return 0.5 * (rho_live_c(i, j, k) + rho_live_c(i + 1, j, k));
}

double rho_live_yface(int i, int j, int k)
{
    return 0.5 * (rho_live_c(i, j, k) + rho_live_c(i, j + 1, k));
}

GridMetrics make_grid()
{
    GridMetrics g;
    g.dx = 100.0;
    g.dy = 100.0;
    g.dz.assign(static_cast<std::size_t>(kN), 100.0);
    return g;
}

void apply_periodic_face_closure(Field3D& u, Field3D& v, int n)
{
    for (int j = 1; j < n - 1; ++j)
        for (int k = 0; k < n; ++k)
            u(0, j, k) = u(n - 2, j, k);
    for (int i = 1; i < n - 1; ++i)
        for (int k = 0; k < n; ++k)
            v(i, 0, k) = v(i, n - 2, k);
}

TerrainMetrics3D make_sloping_metrics(double slope_x, double slope_y)
{
    TerrainMetrics3D metrics;
    metrics.dx = 100.0;
    metrics.dy = 80.0;
    metrics.dzeta = 50.0;
    constexpr double jacobian = 1.2;
    metrics.z.resize(kN, kN, kN, 0.0);
    metrics.J.resize(kN, kN, kN, jacobian);
    metrics.mx.resize(kN, kN, kN, -slope_x / jacobian);
    metrics.my.resize(kN, kN, kN, -slope_y / jacobian);
    metrics.zeta.resize(kN, kN, kN, 0.0);
    metrics.z_face.resize(kN, kN, kN + 1, 0.0);
    metrics.Jx_face.resize(kN + 1, kN, kN, jacobian);
    metrics.Jy_face.resize(kN, kN + 1, kN, jacobian);
    metrics.Gx_face.resize(kN, kN, kN + 1, -slope_x);
    metrics.Gy_face.resize(kN, kN, kN + 1, -slope_y);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
        {
            const double ground = slope_x * i * metrics.dx
                                + slope_y * j * metrics.dy;
            for (int k = 0; k <= kN; ++k)
                metrics.z_face(i, j, k) =
                    ground + jacobian * metrics.dzeta * k;
            for (int k = 0; k < kN; ++k)
            {
                metrics.z(i, j, k) = ground
                    + jacobian * metrics.dzeta * (k + 0.5);
                metrics.zeta(i, j, k) = metrics.dzeta * (k + 0.5);
            }
        }
    return metrics;
}

TurbulenceConfig make_cfg()
{
    TurbulenceConfig c;
    c.mode = "3d";  // use_h && use_v
    return c;
}

// Smooth perturbation confined to the [4,7]^3 interior block (>= 4 cells
// from every boundary), zero everywhere else - so every domain-boundary
// face sees a flat (zero) field. The block-edge gradients give the SGS
// kernels real strain to act on without ever touching the boundaries.
float interior_bump(int i, int j, int k, float amp)
{
    auto in = [](int t) { return t >= 4 && t <= 7; };
    if (!in(i) || !in(j) || !in(k))
    {
        return 0.0f;
    }
    auto f = [](int t)  // sin half-wave over 4..7, zero at the block edges
    {
        return std::sin(3.14159265358979 * (static_cast<double>(t) - 3.0) / 5.0);
    };
    return amp * static_cast<float>(f(i) * f(j) * f(k));
}

// nu_t / K with a strong spatial gradient - the discriminating ingredient:
// a conservative flux form is invariant to it under free-stream and still
// conserves; a local-nu Laplacian is not.
void fill_variable_coeff(Field3D& field, float base, float slope)
{
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                field(i, j, k) = base + slope * static_cast<float>(i + j + k);
}

}  // namespace

// science audit re-audit 2026-06-18: the Cartesian SGS y-spacing was computed
// as dtheta*dr (a cylindrical arc-length formula). On a 400 m square grid that
// set dy = 160000 m, inflating the Smagorinsky filter width ~7x and shrinking
// the y-strain ~400x. representative_dy() is the single source of truth for the
// y-spacing; pin the Cartesian behaviour here.
TEST_CASE("representative_dy: Cartesian dy is the metre y-spacing, not dtheta*dr",
          "[turbulence][sgs][metric]")
{
    // Cartesian: dtheta IS the physical y-spacing in metres.
    CHECK(representative_dy(CoordinateSystem::Cartesian, 400.0, 400.0) == Approx(400.0));  // square: was 160000
    CHECK(representative_dy(CoordinateSystem::Cartesian, 400.0, 250.0) == Approx(250.0));  // non-square dy<dx
    CHECK(representative_dy(CoordinateSystem::Cartesian, 250.0, 400.0) == Approx(400.0));  // non-square dy>dx
    // Invalid geometry is configuration corruption, not a request for square cells.
    CHECK_THROWS_AS(representative_dy(CoordinateSystem::Cartesian, 400.0, 0.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        representative_dy(CoordinateSystem::Cartesian, 400.0,
                          std::numeric_limits<double>::quiet_NaN()),
        std::invalid_argument);

    // The 3rd argument is the y-spacing in metres taken literally - there is no
    // radians-to-arc-length conversion any more: a sub-metre dy passes straight
    // through; a non-positive dy fails closed.
    CHECK(representative_dy(CoordinateSystem::Cartesian, 100.0, 0.1) == Approx(0.1));
    CHECK_THROWS_AS(representative_dy(CoordinateSystem::Cartesian, 100.0, 0.0),
                    std::invalid_argument);
}

TEST_CASE("user SGS filter width is a constant physical length with quadratic Smagorinsky scaling",
          "[turbulence][sgs][filter-width][sci357]")
{
    GridMetrics grid = make_grid();
    TurbulenceConfig cfg = make_cfg();
    cfg.filter_width = "user";
    cfg.user_filter_width_m = 75.0;

    const double delta = eddy_viscosity::compute_filter_width(
        cfg, grid, 4, 5, 6);
    REQUIRE(delta == 75.0);

    const double nu_75 = eddy_viscosity::compute_smagorinsky_viscosity(
        0.18, delta, 0.02, 1.0);
    cfg.user_filter_width_m = 150.0;
    const double nu_150 = eddy_viscosity::compute_smagorinsky_viscosity(
        0.18,
        eddy_viscosity::compute_filter_width(cfg, grid, 1, 2, 3),
        0.02, 1.0);
    REQUIRE(nu_150 == Approx(4.0 * nu_75).epsilon(1.0e-13));

    cfg.user_filter_width_m = 0.0;
    REQUIRE_THROWS_AS(eddy_viscosity::compute_filter_width(
                          cfg, grid, 0, 0, 0),
                      std::invalid_argument);
}

TEST_CASE("dynamic Smagorinsky computes deterministic clipped Favre coefficients",
          "[turbulence][sgs][dynamic][germano][lilly]")
{
    set_cartesian_periodic();
    set_stratified_rho0();
    GridMetrics grid = make_grid();
    TurbulenceConfig config = make_cfg();
    config.dynamic_Cs = true;
    config.Cs = 0.23;

    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f);
    constexpr double pi = 3.14159265358979323846;
    constexpr double physical_n = kN - 2;
    for (int i = 1; i < kN - 1; ++i)
        for (int j = 1; j < kN - 1; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double x = 2.0 * pi * (i - 1) / physical_n;
                const double y = 2.0 * pi * (j - 1) / physical_n;
                const double z = pi * (k + 0.5) / kN;
                u(i, j, k) = static_cast<float>(
                    -8.0 * std::sin(x) * std::cos(y)
                    - 1.5 * std::sin(2.0 * y + z));
                v(i, j, k) = static_cast<float>(
                    6.0 * std::cos(x) * std::sin(y)
                    - 1.0 * std::sin(x - 2.0 * z));
                w(i, j, k) = static_cast<float>(
                    -2.0 * std::sin(x + y) * std::sin(z));
                theta(i, j, k) = static_cast<float>(
                    300.0 + 0.8 * std::cos(x - y) * std::sin(z));
            }
    apply_periodic_face_closure(u, v, kN);
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.rho = &live_rho;
    state.rho0 = &rho0_base;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;

    Field3D first, second;
    tmv::dynamic_smagorinsky::compute_coefficient(
        config, grid, state, first);
    tmv::dynamic_smagorinsky::compute_coefficient(
        config, grid, state, second);
    REQUIRE(first(1, 1, 0) == Approx(0.01048683468).epsilon(2.0e-6));
    REQUIRE(first(1, 1, 3) == Approx(0.01926579885).epsilon(2.0e-6));
    REQUIRE(first(1, 1, 7) == Approx(0.01256603375).epsilon(2.0e-6));
    REQUIRE(first(1, 1, 11) == Approx(0.00696504442).epsilon(2.0e-6));
    double maximum = 0.0;
    for (int k = 0; k < kN; ++k)
    {
        const float plane_value = first(1, 1, k);
        REQUIRE(plane_value >= 0.0f);
        REQUIRE(plane_value <= static_cast<float>(config.Cs));
        maximum = std::max(maximum, static_cast<double>(plane_value));
        for (int i = 1; i < kN - 1; ++i)
            for (int j = 1; j < kN - 1; ++j)
            {
                REQUIRE(first(i, j, k) == plane_value);
                REQUIRE(first(i, j, k) == second(i, j, k));
            }
    }
    REQUIRE(maximum > 0.0);
    REQUIRE(maximum <= config.Cs);

    Field3D viscosity_field(kN, kN, kN, 0.0f);
    Field3D diffusivity(kN, kN, kN, 0.0f);
    for (int i = 1; i < kN - 1; ++i)
        for (int j = 1; j < kN - 1; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double strain = std::sqrt(
                    eddy_viscosity::compute_strain_sq_centered(
                        state, grid, i, j, k));
                const double delta = eddy_viscosity::compute_filter_width(
                    config, grid, i, j, k);
                const double viscosity =
                    eddy_viscosity::compute_smagorinsky_viscosity(
                        first(i, j, k), delta, strain);
                viscosity_field(i, j, k) = static_cast<float>(viscosity);
                diffusivity(i, j, k) = static_cast<float>(
                    viscosity / config.Pr_t);
            }
    Field3D du(kN, kN, kN, 0.0f), dv(kN, kN, kN, 0.0f),
            dw(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        config, state, grid, viscosity_field, du, dv, dw);
    double net_momentum[3] = {0.0, 0.0, 0.0};
    double momentum_activity[3] = {0.0, 0.0, 0.0};
    double resolved_work = 0.0;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double dz = grid.dz[static_cast<std::size_t>(k)];
                const double mass_u =
                    (i < kN - 1 ? rho_live_xface(i, j, k)
                                : rho_live_c(i, j, k)) * dz;
                const double mass_v =
                    (j < kN - 1 ? rho_live_yface(i, j, k)
                                : rho_live_c(i, j, k)) * dz;
                const double mass_w = k < kN - 1
                    ? 0.5 * (rho_live_c(i, j, k) * dz
                           + rho_live_c(i, j, k + 1)
                             * grid.dz[static_cast<std::size_t>(k + 1)])
                    : rho_live_c(i, j, k) * dz;
                const double weighted[] = {
                    mass_u * du(i, j, k), mass_v * dv(i, j, k),
                    mass_w * dw(i, j, k)};
                const double velocity[] = {
                    u(i, j, k), v(i, j, k), w(i, j, k)};
                for (int component = 0; component < 3; ++component)
                {
                    net_momentum[component] += weighted[component];
                    momentum_activity[component] +=
                        std::abs(weighted[component]);
                    resolved_work += velocity[component]
                        * weighted[component];
                }
            }
    for (int component = 0; component < 3; ++component)
    {
        REQUIRE(momentum_activity[component] > 0.0);
        REQUIRE(std::abs(net_momentum[component])
                < 2.0e-3 * momentum_activity[component]);
    }
    REQUIRE(resolved_work < 0.0);

    Field3D scalar_tendency(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        config, state, grid, diffusivity, theta, scalar_tendency);
    double net_scalar = 0.0;
    double scalar_activity = 0.0;
    for (int i = 1; i < kN - 1; ++i)
        for (int j = 1; j < kN - 1; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double currency = static_cast<double>(live_rho(i, j, k))
                    * grid.dz[static_cast<std::size_t>(k)]
                    * static_cast<double>(scalar_tendency(i, j, k));
                net_scalar += currency;
                scalar_activity += std::abs(currency);
            }
    REQUIRE(scalar_activity > 0.0);
    REQUIRE(std::abs(net_scalar) < 2.0e-6 * scalar_activity);

    Field3D previous(kN, kN, kN, 7.0f);
    const float saved_u = u(4, 4, 4);
    u(4, 4, 4) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(tmv::dynamic_smagorinsky::compute_coefficient(
                          config, grid, state, previous),
                      std::invalid_argument);
    for (std::size_t index = 0; index < previous.size(); ++index)
        REQUIRE(previous.data()[index] == 7.0f);

    u(4, 4, 4) = saved_u;
    GridMetrics overflow_grid = grid;
    overflow_grid.dx = 1.0e308;
    overflow_grid.dy = 1.0e308;
    previous.fill(9.0f);
    REQUIRE_THROWS_AS(tmv::dynamic_smagorinsky::compute_coefficient(
                          config, overflow_grid, state, previous),
                      std::runtime_error);
    for (std::size_t index = 0; index < previous.size(); ++index)
        REQUIRE(previous.data()[index] == 9.0f);

    global_lateral_bc_north =
        tmv::init::config::LateralBC::ZeroGradient;
    previous.fill(11.0f);
    REQUIRE_THROWS_AS(tmv::dynamic_smagorinsky::compute_coefficient(
                          config, grid, state, previous),
                      std::invalid_argument);
    for (std::size_t index = 0; index < previous.size(); ++index)
        REQUIRE(previous.data()[index] == 11.0f);
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
}

TEST_CASE("dynamic Smagorinsky vanishes for a uniform resolved flow",
          "[turbulence][sgs][dynamic][laminar]")
{
    set_cartesian_periodic();
    set_stratified_rho0();
    GridMetrics grid = make_grid();
    TurbulenceConfig config = make_cfg();
    config.dynamic_Cs = true;
    config.Cs = 0.23;
    Field3D u(kN, kN, kN, 5.0f), v(kN, kN, kN, -2.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f);
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.rho = &live_rho;
    state.rho0 = &rho0_base;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    Field3D coefficient;
    tmv::dynamic_smagorinsky::compute_coefficient(
        config, grid, state, coefficient);
    for (std::size_t index = 0; index < coefficient.size(); ++index)
        REQUIRE(coefficient.data()[index] == 0.0f);
}

TEST_CASE("dynamic Smagorinsky is bitwise stable across worker counts",
          "[turbulence][sgs][dynamic][openmp][determinism]")
{
#ifndef _OPENMP
    SUCCEED("OpenMP is not enabled in this build");
#else
    set_cartesian_periodic();
    constexpr int n = 34;
    Field3D u(n, n, n, 0.0f), v(n, n, n, 0.0f), w(n, n, n, 0.0f),
            theta(n, n, n, 300.0f), density(n, n, n, 1.0f);
    constexpr double pi = 3.14159265358979323846;
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
            {
                const double x = 2.0 * pi * (i - 1) / (n - 2);
                const double y = 2.0 * pi * (j - 1) / (n - 2);
                const double z = pi * (k + 0.5) / n;
                u(i, j, k) = static_cast<float>(
                    -5.0 * std::sin(x) * std::cos(y + z));
                v(i, j, k) = static_cast<float>(
                    4.0 * std::cos(x - z) * std::sin(y));
                w(i, j, k) = static_cast<float>(
                    -1.5 * std::sin(x + y) * std::sin(z));
                density(i, j, k) = static_cast<float>(
                    0.9 + 0.002 * k + 0.01 * std::cos(x - y));
            }
    apply_periodic_face_closure(u, v, n);
    std::vector<double> base_density(static_cast<std::size_t>(n), 1.0);
    GridMetrics grid;
    grid.dx = 80.0;
    grid.dy = 90.0;
    grid.dz.assign(static_cast<std::size_t>(n), 60.0);
    TurbulenceConfig config = make_cfg();
    config.dynamic_Cs = true;
    config.Cs = 0.23;
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.rho = &density;
    state.rho0 = &base_density;
    state.NR = n;
    state.NTH = n;
    state.NZ = n;

    const int saved_threads = omp_get_max_threads();
    const int saved_dynamic = omp_get_dynamic();
    struct Restore
    {
        int threads;
        int dynamic;
        ~Restore()
        {
            omp_set_dynamic(dynamic);
            omp_set_num_threads(threads);
        }
    } restore{saved_threads, saved_dynamic};
    omp_set_dynamic(0);
    Field3D serial, parallel;
    omp_set_num_threads(1);
    tmv::dynamic_smagorinsky::compute_coefficient(
        config, grid, state, serial);
    omp_set_num_threads(4);
    tmv::dynamic_smagorinsky::compute_coefficient(
        config, grid, state, parallel);
    REQUIRE(serial.size() == parallel.size());
    REQUIRE(std::memcmp(serial.data(), parallel.data(),
                        serial.size() * sizeof(float)) == 0);
#endif
}

TEST_CASE("M14: centered strain resolves a 2dx shear the collocated read misses",
          "[turbulence][sgs][m14]")
{
    set_cartesian_open();
    GridMetrics grid = make_grid();

    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f);

    // A 2dx zig-zag of u in z: u alternates sign every level. The centered
    // difference (u[k+1]-u[k-1])/2dz reads ZERO (same-parity neighbors are
    // equal); the face-to-center natural difference (u[k+1]-u[k])/dz does
    // not - this is precisely the mode M14 was blind to before.
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                u(i, j, k) = (k % 2 == 0) ? 1.0f : -1.0f;

    TurbulenceStateView st;
    st.u = &u; st.v = &v; st.w = &w; st.theta = &theta; st.rho = &live_rho;
    st.rho0 = &rho0_base;
    st.NR = kN; st.NTH = kN; st.NZ = kN;

    const int i = 6, j = 6, k = 6;  // deep interior

    const double s_collocated =
        eddy_viscosity::compute_strain_rate_3d(st, grid, i, j, k).magnitude;
    const double ssq_centered =
        eddy_viscosity::compute_strain_sq_centered(st, grid, i, j, k);

    // The collocated read is blind to the 2dx mode.
    REQUIRE(std::abs(s_collocated) < 1.0e-9);
    // The M14 staggered-natural |S|^2 captures it (du/dz ~ 2/dz at the
    // surrounding edges): 4*(0.5*2/100)^2-scale, comfortably > 0.
    REQUIRE(ssq_centered > 1.0e-5);
}

TEST_CASE("directional SGS strain excludes gradients owned by the other closure",
          "[turbulence][sgs][directional][cpu-gpu-contract]")
{
    set_cartesian_open();
    set_stratified_rho0();
    const GridMetrics grid = make_grid();
    constexpr double shear = 0.0125;
    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                u(i, j, k) = static_cast<float>(
                    shear * (static_cast<double>(k) + 0.5) * grid.dz[k]);

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
    state.rho = &live_rho; state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN; state.NTH = kN; state.NZ = kN;

    const double horizontal =
        eddy_viscosity::compute_directional_strain_sq_centered(
            state, grid, 6, 6, 6, "horizontal_only");
    const double vertical =
        eddy_viscosity::compute_directional_strain_sq_centered(
            state, grid, 6, 6, 6, "vertical_only");
    const double full =
        eddy_viscosity::compute_directional_strain_sq_centered(
            state, grid, 6, 6, 6, "3d");

    REQUIRE(horizontal == Approx(0.0).margin(1.0e-14));
    REQUIRE(vertical == Approx(shear * shear).epsilon(2.0e-5));
    REQUIRE(full == Approx(shear * shear).epsilon(2.0e-5));
}

TEST_CASE("directional SGS momentum stress has no unpaired cross-gradient",
          "[turbulence][sgs][directional][dissipation][cpu-gpu-contract]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    live_rho.resize(kN, kN, kN, 1.0f);
    const GridMetrics grid = make_grid();
    constexpr double gradient = 2.0e-7;
    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f),
            nu(kN, kN, kN, 5.0f), du(kN, kN, kN, 0.0f),
            dv(kN, kN, kN, 0.0f), dw(kN, kN, kN, 0.0f);
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
    state.rho = &live_rho; state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN; state.NTH = kN; state.NZ = kN;

    // du/dz varies in x. A horizontal closure may mix u horizontally, but it
    // must not feed du/dz into the w equation without the reciprocal vertical
    // u-flux. The former partial symmetric tensor did exactly that.
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                u(i, j, k) = static_cast<float>(
                    gradient * (i * grid.dx)
                    * ((static_cast<double>(k) + 0.5) * grid.dz[k]));
    TurbulenceConfig cfg = make_cfg();
    cfg.mode = "horizontal_only";
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu, du, dv, dw);
    REQUIRE(dw(6, 6, 5) == Approx(0.0).margin(1.0e-12));

    // dw/dx varies in z. A vertical closure must likewise not feed that
    // horizontal gradient into the vertical divergence of u stress.
    u.fill(0.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                w(i, j, k) = static_cast<float>(
                    gradient * (i * grid.dx)
                    * ((static_cast<double>(k) + 1.0) * grid.dz[k]));
    du.fill(0.0f); dv.fill(0.0f); dw.fill(0.0f);
    tmv::bind_flow_execution(state, global_simulation_state());
    cfg.mode = "vertical_only";
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu, du, dv, dw);
    REQUIRE(du(6, 6, 5) == Approx(0.0).margin(1.0e-12));
}

TEST_CASE("Richardson correction uses the staggered-natural shear norm",
          "[turbulence][sgs][m14][ri][sci073]")
{
    set_cartesian_open();
    GridMetrics grid = make_grid();
    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                u(i, j, k) = (k % 2 == 0) ? 1.0f : -1.0f;
                theta(i, j, k) = static_cast<float>(300.0 + k);
            }
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    const double strain_sq = eddy_viscosity::compute_strain_sq_centered(
        state, grid, 6, 6, 6);
    const double frequency =
        eddy_viscosity::compute_brunt_vaisala_frequency(
            state, grid, 6, 6, 6);
    const double expected = eddy_viscosity::stability_correction_ri(
        frequency * frequency / strain_sq);
    const double actual = eddy_viscosity::cgrid_stability_factor_ri(
        state, grid, 6, 6, 6);
    REQUIRE(strain_sq > 0.0);
    REQUIRE(actual < 1.0);
    REQUIRE(actual == Approx(expected).epsilon(1.0e-13));
}

TEST_CASE("Richardson correction optionally uses moist virtual stability",
          "[turbulence][sgs][ri][moist]")
{
    set_cartesian_open();
    GridMetrics grid = make_grid();
    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f),
            vapor(kN, kN, kN, 0.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                u(i, j, k) = 0.02f * static_cast<float>(k * grid.dz[k]);
                vapor(i, j, k) = 0.001f * static_cast<float>(k);
            }
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.qv = &vapor;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;

    const double dry = eddy_viscosity::cgrid_stability_factor_ri(
        state, grid, 6, 6, 6, false);
    const double moist = eddy_viscosity::cgrid_stability_factor_ri(
        state, grid, 6, 6, 6, true);
    REQUIRE(dry == Approx(1.0));
    REQUIRE(moist < dry);
    REQUIRE(moist >= 0.1);
}

TEST_CASE("M15: a uniform velocity field produces no SGS momentum tendency",
          "[turbulence][sgs][m15][freestream]")
{
    set_cartesian_open();
    set_stratified_rho0();
    GridMetrics grid = make_grid();
    TurbulenceConfig cfg = make_cfg();

    // Uniform horizontal flow with w = 0 - the only BC-consistent uniform
    // state (the rigid ground pins w(ground) = 0, so a uniform w != 0 is a
    // genuine ground shear, not a free-stream).
    Field3D u(kN, kN, kN, 7.0f), v(kN, kN, kN, -3.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f);
    Field3D nu(kN, kN, kN, 0.0f);
    fill_variable_coeff(nu, 5.0f, 3.0f);  // strongly varying eddy viscosity

    TurbulenceStateView st;
    st.u = &u; st.v = &v; st.w = &w; st.theta = &theta; st.rho = &live_rho;
    st.rho0 = &rho0_base;
    st.NR = kN; st.NTH = kN; st.NZ = kN;

    Field3D du, dv, dw;
    du.resize(kN, kN, kN, 0.0f);
    dv.resize(kN, kN, kN, 0.0f);
    dw.resize(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, st, grid, nu, du, dv, dw);

    float max_abs = 0.0f;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                max_abs = std::max({max_abs, std::abs(du(i, j, k)),
                                    std::abs(dv(i, j, k)), std::abs(dw(i, j, k))});

    // Every strain is exactly zero -> every stress -> zero divergence,
    // regardless of the nu gradient.
    REQUIRE(max_abs < 1.0e-6f);
}

TEST_CASE("M15: a uniform scalar field produces no SGS scalar tendency",
          "[turbulence][sgs][m15][freestream]")
{
    set_cartesian_open();
    set_stratified_rho0();
    GridMetrics grid = make_grid();
    TurbulenceConfig cfg = make_cfg();

    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f);
    Field3D K(kN, kN, kN, 0.0f), phi(kN, kN, kN, 285.0f);
    fill_variable_coeff(K, 4.0f, 2.0f);  // strongly varying eddy diffusivity

    TurbulenceStateView st;
    st.u = &u; st.v = &v; st.w = &w; st.theta = &theta; st.rho = &live_rho;
    st.rho0 = &rho0_base;
    st.NR = kN; st.NTH = kN; st.NZ = kN;

    Field3D dphi;
    dphi.resize(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, st, grid, K, phi, dphi);

    float max_abs = 0.0f;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                max_abs = std::max(max_abs, std::abs(dphi(i, j, k)));

    REQUIRE(max_abs < 1.0e-6f);
}

TEST_CASE("terrain SGS scalar flux uses variable mapped diffusivity conservatively",
          "[turbulence][sgs][terrain][scalar][sci080]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    TerrainMetrics3D terrain = make_sloping_metrics(0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, terrain.dzeta * 1.2);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;

    Field3D rho(kN, kN, kN, 1.0f);
    Field3D K(kN, kN, kN, 0.0f);
    Field3D phi(kN, kN, kN, 287.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                rho(i, j, k) = 0.8f + 0.01f * (i + 2 * j + k);
                K(i, j, k) = 2.0f + 0.2f * (i + j + k);
            }

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.rho = &rho;
    state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    TurbulenceConfig cfg = make_cfg();
    Field3D tendency(kN, kN, kN, 9.0f);

    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, K, phi, tendency);
    for (std::size_t index = 0; index < tendency.size(); ++index)
        REQUIRE(tendency.data()[index] == 0.0f);

    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                phi(i, j, k) = interior_bump(i, j, k, 3.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, K, phi, tendency);

    double inventory_rate = 0.0;
    double absolute_rate = 0.0;
    double scalar_energy_rate = 0.0;
    for (int i = 1; i < kN - 1; ++i)
        for (int j = 1; j < kN - 1; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double weighted = static_cast<double>(rho(i, j, k))
                    * static_cast<double>(tendency(i, j, k))
                    * terrain.J(i, j, k) * terrain.dx * terrain.dy
                    * terrain.dzeta;
                inventory_rate += weighted;
                absolute_rate += std::abs(weighted);
                scalar_energy_rate += static_cast<double>(phi(i, j, k))
                    * weighted;
            }
    REQUIRE(absolute_rate > 0.0);
    REQUIRE(std::abs(inventory_rate)
            <= 2.0e-6 * absolute_rate);
    REQUIRE(scalar_energy_rate < 0.0);
}

TEST_CASE("mapped collocated SGS conserves physical momentum and scalar currency",
          "[turbulence][sgs][terrain][collocated][conservation][work][sci417]")
{
    set_cartesian_open();
    TerrainMetrics3D terrain = make_sloping_metrics(0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, terrain.dzeta * 1.2);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    grid.staggering = StaggerType::Collocated;

    Field3D rho(kN, kN, kN, 1.0f);
    Field3D u(kN, kN, kN, 0.0f);
    Field3D v(kN, kN, kN, 0.0f);
    Field3D w(kN, kN, kN, 0.0f);
    Field3D scalar(kN, kN, kN, 0.0f);
    Field3D coefficient(kN, kN, kN, 0.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                rho(i, j, k) = static_cast<float>(
                    0.9 + 0.01 * i + 0.005 * j + 0.002 * k);
                coefficient(i, j, k) = static_cast<float>(
                    12.0 + 0.3 * i + 0.2 * j + 0.1 * k);
                u(i, j, k) = interior_bump(i, j, k, 4.0f);
                v(i, j, k) = interior_bump(i, j, k, -2.0f);
                w(i, j, k) = interior_bump(i, j, k, 1.0f);
                scalar(i, j, k) = interior_bump(i, j, k, 3.0f);
            }

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.rho = &rho;
    state.theta = &scalar;
    state.c_grid = false;
    state.NR = state.NTH = state.NZ = kN;
    TurbulenceConfig cfg = make_cfg();
    Field3D du(kN, kN, kN, 17.0f);
    Field3D dv(kN, kN, kN, 17.0f);
    Field3D dw(kN, kN, kN, 17.0f);
    double boundary_power = std::numeric_limits<double>::quiet_NaN();
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, coefficient, du, dv, dw, &boundary_power);
    Field3D dscalar(kN, kN, kN, 17.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, coefficient, scalar, dscalar);

    const PhysicalCellDomain domain = physical_scalar_domain(
        kN, kN, kN, StaggerType::Collocated);
    long double x_momentum = 0.0L;
    long double y_momentum = 0.0L;
    long double z_momentum = 0.0L;
    long double scalar_currency = 0.0L;
    long double momentum_work = 0.0L;
    long double scalar_work = 0.0L;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const bool physical = i >= domain.i_begin && i < domain.i_end
                    && j >= domain.j_begin && j < domain.j_end
                    && k >= domain.k_begin && k < domain.k_end;
                if (!physical)
                {
                    REQUIRE(du(i, j, k) == 0.0f);
                    REQUIRE(dv(i, j, k) == 0.0f);
                    REQUIRE(dw(i, j, k) == 0.0f);
                    REQUIRE(dscalar(i, j, k) == 0.0f);
                    continue;
                }
                const long double mass = static_cast<long double>(rho(i, j, k))
                    * terrain.J(i, j, k) * terrain.dx * terrain.dy
                    * terrain.dzeta;
                x_momentum += mass * du(i, j, k);
                y_momentum += mass * dv(i, j, k);
                z_momentum += mass * dw(i, j, k);
                scalar_currency += mass * dscalar(i, j, k);
                momentum_work += mass * (
                    static_cast<long double>(u(i, j, k)) * du(i, j, k)
                  + static_cast<long double>(v(i, j, k)) * dv(i, j, k)
                  + static_cast<long double>(w(i, j, k)) * dw(i, j, k));
                scalar_work += mass
                    * static_cast<long double>(scalar(i, j, k))
                    * dscalar(i, j, k);
            }
    const long double currency_scale = terrain.dx * terrain.dy
        * terrain.dzeta;
    REQUIRE(std::abs(x_momentum) < 2.0e-5L * currency_scale);
    REQUIRE(std::abs(y_momentum) < 2.0e-5L * currency_scale);
    REQUIRE(std::abs(z_momentum) < 2.0e-5L * currency_scale);
    REQUIRE(std::abs(scalar_currency) < 2.0e-5L * currency_scale);
    REQUIRE(momentum_work < 0.0L);
    REQUIRE(scalar_work < 0.0L);
    REQUIRE(boundary_power == 0.0);

    const Field3D committed_du = du;
    coefficient(5, 5, 5) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        eddy_viscosity::compute_momentum_diffusion_tendencies(
            cfg, state, grid, coefficient, du, dv, dw, &boundary_power),
        std::invalid_argument);
    REQUIRE(std::memcmp(
        du.data(), committed_du.data(), du.size() * sizeof(float)) == 0);
}

TEST_CASE("mapped collocated strain recovers a physical linear tensor",
          "[turbulence][sgs][terrain][collocated][strain][reference][sci417]")
{
    set_cartesian_open();
    TerrainMetrics3D terrain = make_sloping_metrics(0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, terrain.dzeta * 1.2);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    grid.staggering = StaggerType::Collocated;
    Field3D rho(kN, kN, kN, 1.0f);
    Field3D u(kN, kN, kN, 0.0f);
    Field3D v(kN, kN, kN, 0.0f);
    Field3D w(kN, kN, kN, 0.0f);
    constexpr double a = 0.0010, b = -0.0004, c = 0.0007;
    constexpr double d = 0.0003, e = -0.0008, f = 0.0002;
    constexpr double g = -0.0005, h = 0.0006, q = 0.0009;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double x = i * terrain.dx;
                const double y = j * terrain.dy;
                const double z = terrain.z(i, j, k);
                u(i, j, k) = static_cast<float>(a * x + b * y + c * z);
                v(i, j, k) = static_cast<float>(d * x + e * y + f * z);
                w(i, j, k) = static_cast<float>(g * x + h * y + q * z);
            }
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.rho = &rho;
    state.c_grid = false;
    state.NR = state.NTH = state.NZ = kN;
    const double full = eddy_viscosity::compute_directional_strain_sq_centered(
        state, grid, 5, 5, 5, "3d");
    const double horizontal =
        eddy_viscosity::compute_directional_strain_sq_centered(
            state, grid, 5, 5, 5, "horizontal_only");
    const double vertical =
        eddy_viscosity::compute_directional_strain_sq_centered(
            state, grid, 5, 5, 5, "vertical_only");
    const double expected_full = 2.0 * (a * a + e * e + q * q)
        + (b + d) * (b + d) + (c + g) * (c + g)
        + (f + h) * (f + h);
    const double expected_horizontal = 2.0 * (a * a + e * e)
        + (b + d) * (b + d) + g * g + h * h;
    const double expected_vertical = c * c + f * f + 2.0 * q * q;
    REQUIRE(full == Approx(expected_full).epsilon(2.0e-5));
    REQUIRE(horizontal == Approx(expected_horizontal).epsilon(2.0e-5));
    REQUIRE(vertical == Approx(expected_vertical).epsilon(2.0e-5));
}

TEST_CASE("terrain SGS scalar adjoint reduces to the flat C-grid operator",
          "[turbulence][sgs][terrain][scalar][flat-reduction]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    TerrainMetrics3D terrain = make_sloping_metrics(0.0, 0.0);
    GridMetrics mapped;
    mapped.dx = terrain.dx;
    mapped.dy = terrain.dy;
    mapped.dz.assign(kN, 1.2 * terrain.dzeta);
    mapped.terrain_metrics_active = true;
    mapped.terrain_metrics = &terrain;
    GridMetrics flat = mapped;
    flat.terrain_metrics_active = false;
    flat.terrain_metrics = nullptr;

    Field3D rho(kN, kN, kN, 1.0f), diffusivity(kN, kN, kN, 1.0f),
            scalar(kN, kN, kN, 280.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                rho(i, j, k) = 0.8f + 0.01f * (i + 2 * j + k);
                diffusivity(i, j, k) = 1.0f + 0.04f * (2 * i + j + k);
                scalar(i, j, k) = 280.0f
                    + interior_bump(i, j, k, 4.0f);
            }
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.rho = &rho;
    state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    Field3D mapped_tendency(kN, kN, kN, 0.0f);
    Field3D flat_tendency(kN, kN, kN, 0.0f);
    for (const std::string mode : {"3d", "horizontal_only", "vertical_only"})
    {
        TurbulenceConfig cfg = make_cfg();
        cfg.mode = mode;
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, mapped, diffusivity, scalar, mapped_tendency);
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, flat, diffusivity, scalar, flat_tendency);
        INFO("directional mode=" << mode);
        for (std::size_t n = 0; n < mapped_tendency.size(); ++n)
        {
            CAPTURE(n);
            REQUIRE(mapped_tendency.data()[n]
                    == Approx(flat_tendency.data()[n]).margin(2.0e-8));
        }
    }
}

TEST_CASE("terrain SGS uniform tangent flow has zero strain and mapped stress",
          "[turbulence][sgs][terrain][tensor][uniform][sci114]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    TerrainMetrics3D terrain = make_sloping_metrics(0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, terrain.dzeta * 1.2);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;

    constexpr float uniform_u = 2.0f;
    constexpr float uniform_v = -1.0f;
    const float terrain_tangent_w = static_cast<float>(
        uniform_u * 0.12 + uniform_v * -0.07);
    Field3D rho(kN, kN, kN, 1.0f), u(kN, kN, kN, uniform_u),
            v(kN, kN, kN, uniform_v),
            w(kN, kN, kN, terrain_tangent_w),
            theta(kN, kN, kN, 300.0f), nu(kN, kN, kN, 4.0f);
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.rho = &rho;
    state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;

    REQUIRE_THROWS_AS(eddy_viscosity::compute_strain_rate_3d(
                          state, grid, 5, 5, 5),
                      std::invalid_argument);
    REQUIRE(eddy_viscosity::compute_strain_sq_centered(
                state, grid, 5, 5, 5) == 0.0);
    REQUIRE(eddy_viscosity::compute_filter_width(
                make_cfg(), grid, 5, 5, 5)
            == Approx(std::cbrt(
                terrain.dx * terrain.dy * 1.2 * terrain.dzeta)));
    REQUIRE(eddy_viscosity::cgrid_stability_factor_ri(
                state, grid, 5, 5, 5) == 1.0);

    Field3D du(kN, kN, kN, 17.0f), dv(kN, kN, kN, 17.0f),
            dw(kN, kN, kN, 17.0f);
    REQUIRE_NOTHROW(eddy_viscosity::compute_momentum_diffusion_tendencies(
        make_cfg(), state, grid, nu, du, dv, dw));
    for (const Field3D* output : {&du, &dv, &dw})
        for (std::size_t n = 0; n < output->size(); ++n)
            REQUIRE(output->data()[n] == Approx(0.0f).margin(1.0e-10));

    du.fill(17.0f); dv.fill(17.0f); dw.fill(17.0f);
    terrain.Gx_face(5, 5, 5) =
        std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(eddy_viscosity::compute_momentum_diffusion_tendencies(
                          make_cfg(), state, grid, nu, du, dv, dw),
                      std::invalid_argument);
    for (const Field3D* output : {&du, &dv, &dw})
        for (std::size_t n = 0; n < output->size(); ++n)
            REQUIRE(output->data()[n] == 17.0f);
}

TEST_CASE("terrain C-grid strain recovers a physical linear velocity tensor",
          "[turbulence][sgs][terrain][strain][mapped][sci113]")
{
    set_cartesian_open();
    TerrainMetrics3D terrain = make_sloping_metrics(0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, terrain.dzeta * 1.2);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;

    constexpr double ax = 0.010;
    constexpr double ay = -0.006;
    constexpr double az = 0.003;
    constexpr double bx = 0.004;
    constexpr double by = -0.008;
    constexpr double bz = 0.002;
    constexpr double cx = -0.005;
    constexpr double cy = 0.007;
    constexpr double cz = 0.009;
    Field3D u(kN, kN, kN, 0.0f);
    Field3D v(kN, kN, kN, 0.0f);
    Field3D w(kN, kN, kN, 0.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const int i1 = std::min(i + 1, kN - 1);
                const int j1 = std::min(j + 1, kN - 1);
                const double xu = (static_cast<double>(i) + 0.5)
                    * terrain.dx;
                const double yu = static_cast<double>(j) * terrain.dy;
                const double zu = 0.5 * (
                    terrain.z(i, j, k) + terrain.z(i1, j, k));
                const double xv = static_cast<double>(i) * terrain.dx;
                const double yv = (static_cast<double>(j) + 0.5)
                    * terrain.dy;
                const double zv = 0.5 * (
                    terrain.z(i, j, k) + terrain.z(i, j1, k));
                const double xw = static_cast<double>(i) * terrain.dx;
                const double yw = static_cast<double>(j) * terrain.dy;
                const double zw = terrain.z_face(i, j, k + 1);
                u(i, j, k) = static_cast<float>(
                    ax * xu + ay * yu + az * zu);
                v(i, j, k) = static_cast<float>(
                    bx * xv + by * yv + bz * zv);
                w(i, j, k) = static_cast<float>(
                    cx * xw + cy * yw + cz * zw);
            }

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    const double s12 = 0.5 * (ay + bx);
    const double s13 = 0.5 * (az + cx);
    const double s23 = 0.5 * (bz + cy);
    const double expected = 2.0 * (
        ax * ax + by * by + cz * cz
        + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23));
    REQUIRE(eddy_viscosity::compute_strain_sq_centered(
                state, grid, 6, 6, 5)
            == Approx(expected).epsilon(2.0e-5));
    const double expected_horizontal =
        2.0 * (ax * ax + by * by) + 4.0 * s12 * s12
        + cx * cx + cy * cy;
    const double expected_vertical = az * az + bz * bz + 2.0 * cz * cz;
    REQUIRE(eddy_viscosity::compute_directional_strain_sq_centered(
                state, grid, 6, 6, 5, "horizontal_only")
            == Approx(expected_horizontal).epsilon(2.0e-5));
    REQUIRE(eddy_viscosity::compute_directional_strain_sq_centered(
                state, grid, 6, 6, 5, "vertical_only")
            == Approx(expected_vertical).epsilon(2.0e-5));
}

TEST_CASE("terrain ground strain uses the physical surface reference frame",
          "[turbulence][sgs][terrain][strain][earth_fixed]")
{
    set_cartesian_open();
    constexpr double slope_x = 0.125, slope_y = -0.0625;
    TerrainMetrics3D terrain = make_sloping_metrics(slope_x, slope_y);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, terrain.dzeta * 1.2);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;
    Field3D u(kN, kN, kN, 8.0f), v(kN, kN, kN, 4.0f);
    Field3D w(kN, kN, kN, static_cast<float>(8.0 * slope_x + 4.0 * slope_y));
    const bool earth_fixed = GENERATE(false, true);
    TerrainRuntimeState runtime;
    runtime.config.scheme_id = "bell";
    runtime.config.horizontal_placement_id = earth_fixed ? "earth_fixed" : "grid_attached";
    tmv::domain::HorizontalDomainFrame frame;
    frame.velocity_ms = {4.0, -2.0};
    auto execution = global_simulation_state();
    execution.terrain_runtime = &runtime;
    execution.domain_frame = &frame;
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, execution);
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.NR = state.NTH = state.NZ = kN;
    const double thickness = terrain.z_face(6, 6, 1) - terrain.z_face(6, 6, 0);
    const double s33 = earth_fixed ? 0.0
        : (slope_x * frame.velocity_ms.u_ms + slope_y * frame.velocity_ms.v_ms) / thickness;
    CHECK(eddy_viscosity::compute_strain_sq_centered(state, grid, 6, 6, 0)
          == Approx(2.0 * s33 * s33).margin(1.0e-12));
    CHECK(eddy_viscosity::compute_directional_strain_sq_centered(
              state, grid, 6, 6, 0, "vertical_only")
          == Approx(2.0 * s33 * s33).margin(1.0e-12));
}

TEST_CASE("terrain SGS mapped momentum operator reduces to the flat C-grid",
          "[turbulence][sgs][terrain][momentum][flat-reduction][sci114]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    TerrainMetrics3D terrain = make_sloping_metrics(0.0, 0.0);
    GridMetrics mapped;
    mapped.dx = terrain.dx;
    mapped.dy = terrain.dy;
    mapped.dz.assign(kN, 1.2 * terrain.dzeta);
    mapped.terrain_metrics_active = true;
    mapped.terrain_metrics = &terrain;
    GridMetrics flat = mapped;
    flat.terrain_metrics_active = false;
    flat.terrain_metrics = nullptr;

    Field3D rho(kN, kN, kN, 1.0f), u(kN, kN, kN, 0.0f),
            v(kN, kN, kN, 0.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f), nu(kN, kN, kN, 1.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                rho(i, j, k) = 0.8f + 0.011f * (i + 2 * j + k);
                nu(i, j, k) = 1.0f + 0.07f * (i + j + 2 * k);
                u(i, j, k) = interior_bump(i, j, k, 5.0f);
                v(i, j, k) = interior_bump(j, k, i, -3.0f);
                w(i, j, k) = interior_bump(k, i, j, 2.0f);
            }
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
    state.rho = &rho; state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN; state.NTH = kN; state.NZ = kN;
    Field3D flat_u(kN, kN, kN, 0.0f), flat_v(kN, kN, kN, 0.0f),
            flat_w(kN, kN, kN, 0.0f);
    Field3D map_u(kN, kN, kN, 0.0f), map_v(kN, kN, kN, 0.0f),
            map_w(kN, kN, kN, 0.0f);
    for (const std::string mode : {"3d", "horizontal_only", "vertical_only"})
    {
        TurbulenceConfig cfg = make_cfg();
        cfg.mode = mode;
        eddy_viscosity::compute_momentum_diffusion_tendencies(
            cfg, state, flat, nu, flat_u, flat_v, flat_w);
        eddy_viscosity::compute_momentum_diffusion_tendencies(
            cfg, state, mapped, nu, map_u, map_v, map_w);

        INFO("directional mode=" << mode);
        for (std::size_t n = 0; n < flat_u.size(); ++n)
        {
            REQUIRE(map_u.data()[n]
                    == Approx(flat_u.data()[n]).margin(2.0e-8));
            REQUIRE(map_v.data()[n]
                    == Approx(flat_v.data()[n]).margin(2.0e-8));
            REQUIRE(map_w.data()[n]
                    == Approx(flat_w.data()[n]).margin(2.0e-8));
        }
    }
}

TEST_CASE("terrain SGS mapped stress conserves momentum and removes resolved energy",
          "[turbulence][sgs][terrain][momentum][conservation][work][sci114]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    TerrainMetrics3D terrain = make_sloping_metrics(0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, 1.2 * terrain.dzeta);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;

    Field3D rho(kN, kN, kN, 1.0f), u(kN, kN, kN, 0.0f),
            v(kN, kN, kN, 0.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f), nu(kN, kN, kN, 1.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                rho(i, j, k) = 0.75f + 0.013f * (2 * i + j + k);
                nu(i, j, k) = 2.0f + 0.11f * (i + 2 * j + k);
                u(i, j, k) = interior_bump(i, j, k, 5.0f);
                v(i, j, k) = interior_bump(j, k, i, -3.0f);
                w(i, j, k) = interior_bump(k, i, j, 2.0f);
            }
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
    state.rho = &rho; state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN; state.NTH = kN; state.NZ = kN;
    Field3D du(kN, kN, kN, 0.0f), dv(kN, kN, kN, 0.0f),
            dw(kN, kN, kN, 0.0f);
    for (const std::string mode : {"3d", "horizontal_only", "vertical_only"})
    {
        TurbulenceConfig cfg = make_cfg();
        cfg.mode = mode;
        eddy_viscosity::compute_momentum_diffusion_tendencies(
            cfg, state, grid, nu, du, dv, dw);

        double net_u = 0.0, net_v = 0.0, net_w = 0.0;
        double act_u = 0.0, act_v = 0.0, act_w = 0.0;
        double resolved_work = 0.0;
        for (int i = 0; i < kN; ++i)
            for (int j = 0; j < kN; ++j)
                for (int k = 0; k < kN; ++k)
                {
                    const int ip = std::min(i + 1, kN - 1);
                    const int jp = std::min(j + 1, kN - 1);
                    const int kp = std::min(k + 1, kN - 1);
                    const double mass_u = 0.5 * (
                        static_cast<double>(rho(i, j, k)) * terrain.J(i, j, k)
                      + static_cast<double>(rho(ip, j, k)) * terrain.J(ip, j, k));
                    const double mass_v = 0.5 * (
                        static_cast<double>(rho(i, j, k)) * terrain.J(i, j, k)
                      + static_cast<double>(rho(i, jp, k)) * terrain.J(i, jp, k));
                    const double mass_w = 0.5 * (
                        static_cast<double>(rho(i, j, k)) * terrain.J(i, j, k)
                      + static_cast<double>(rho(i, j, kp)) * terrain.J(i, j, kp));
                    const double force_u = mass_u * du(i, j, k);
                    const double force_v = mass_v * dv(i, j, k);
                    const double force_w = mass_w * dw(i, j, k);
                    net_u += force_u; act_u += std::abs(force_u);
                    net_v += force_v; act_v += std::abs(force_v);
                    net_w += force_w; act_w += std::abs(force_w);
                    resolved_work += static_cast<double>(u(i, j, k)) * force_u
                                   + static_cast<double>(v(i, j, k)) * force_v
                                   + static_cast<double>(w(i, j, k)) * force_w;
                }
        INFO("directional mode=" << mode);
        REQUIRE(act_u > 1.0e-4);
        REQUIRE(act_v > 1.0e-4);
        REQUIRE(act_w > 1.0e-4);
        REQUIRE(std::abs(net_u) <= 3.0e-6 * act_u);
        REQUIRE(std::abs(net_v) <= 3.0e-6 * act_v);
        REQUIRE(std::abs(net_w) <= 3.0e-6 * act_w);
        REQUIRE(resolved_work < 0.0);
    }
}

TEST_CASE("terrain SGS publishes exact open-boundary traction work",
          "[turbulence][sgs][terrain][momentum][boundary-work]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    TerrainMetrics3D terrain = make_sloping_metrics(0.12, -0.07);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, 1.2 * terrain.dzeta);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;

    Field3D rho(kN, kN, kN, 1.0f), u(kN, kN, kN, 1.0f),
            v(kN, kN, kN, 0.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f), nu(kN, kN, kN, 1.0f);
    for (int j = 0; j < kN; ++j)
        for (int k = 0; k < kN; ++k)
        {
            u(0, j, k) = 10.0f;
            u(kN - 2, j, k) = 10.0f;
        }
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
    state.rho = &rho; state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN; state.NTH = kN; state.NZ = kN;
    TurbulenceConfig cfg = make_cfg();
    cfg.mode = "horizontal_only";
    Field3D du(kN, kN, kN, 0.0f), dv(kN, kN, kN, 0.0f),
            dw(kN, kN, kN, 0.0f);
    double boundary_power = std::numeric_limits<double>::quiet_NaN();
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu, du, dv, dw, &boundary_power);

    const double face_area = terrain.dy * terrain.dzeta;
    const double transformed_stress = 1.2 * 2.0 * (1.0 - 10.0)
        / terrain.dx;
    const int face_count = (kN - 2) * (kN - 1);
    const double expected = 2.0 * face_count * face_area
        * (-10.0 * transformed_stress);
    REQUIRE(boundary_power == Approx(expected));

    double linear_power = 0.0;
    const double cell_volume = terrain.dx * terrain.dy * terrain.dzeta;
    for (int i = 1; i <= kN - 3; ++i)
        for (int j = 1; j < kN - 1; ++j)
            for (int k = 0; k < kN - 1; ++k)
            {
                const double mass = 0.5 * (
                    static_cast<double>(rho(i, j, k)) * terrain.J(i, j, k)
                  + static_cast<double>(rho(i + 1, j, k))
                        * terrain.J(i + 1, j, k)) * cell_volume;
                linear_power += mass * static_cast<double>(u(i, j, k))
                    * static_cast<double>(du(i, j, k));
            }
    REQUIRE(linear_power > 0.0);
    REQUIRE(boundary_power - linear_power > 0.0);

    global_lateral_bc_west = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_east = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_south = tmv::init::config::LateralBC::Rigid;
    global_lateral_bc_north = tmv::init::config::LateralBC::Rigid;
    global_top_bc = tmv::init::config::TopBC::Open;
    u.fill(1.0f);
    v.fill(0.0f);
    w.fill(0.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            u(i, j, kN - 1) = 10.0f;
    tmv::bind_flow_execution(state, global_simulation_state());
    cfg.mode = "vertical_only";
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu, du, dv, dw, &boundary_power);
    const double top_stress = (10.0 - 1.0)
        / (1.2 * terrain.dzeta);
    const int top_u_face_count = (kN - 3) * (kN - 2);
    REQUIRE(boundary_power == Approx(
        top_u_face_count * terrain.dx * terrain.dy * 10.0 * top_stress));
    set_cartesian_open();
}

TEST_CASE("terrain SGS scalar flux closes a periodic physical seam",
          "[turbulence][sgs][terrain][scalar][periodic][sci087]")
{
    set_cartesian_open();
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    rho0_base.assign(kN, 1.0);
    // A y-only slope is single-valued across the periodic x seam while still
    // exercising the terrain cross-flux in the mapped divergence.
    TerrainMetrics3D terrain = make_sloping_metrics(0.0, -0.06);
    GridMetrics grid;
    grid.dx = terrain.dx;
    grid.dy = terrain.dy;
    grid.dz.assign(kN, terrain.dzeta * 1.2);
    grid.terrain_metrics_active = true;
    grid.terrain_metrics = &terrain;

    Field3D rho(kN, kN, kN, 1.0f);
    Field3D K(kN, kN, kN, 1.0f);
    Field3D phi(kN, kN, kN, 280.0f);
    for (int i = 1; i < kN - 1; ++i)
        for (int j = 1; j < kN - 1; ++j)
            for (int k = 0; k < kN; ++k)
            {
                rho(i, j, k) = 0.9f + 0.012f * (i + j + k);
                K(i, j, k) = 1.5f + 0.08f * (2 * i + j + k);
                phi(i, j, k) = 280.0f + 0.3f * i * i
                    - 0.2f * j + 0.1f * i * j + 0.15f * k * k;
            }

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.rho = &rho;
    state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    Field3D tendency(kN, kN, kN, 0.0f);
    REQUIRE_NOTHROW(eddy_viscosity::compute_scalar_diffusion_tendencies(
        make_cfg(), state, grid, K, phi, tendency));

    double inventory_rate = 0.0;
    double absolute_rate = 0.0;
    for (int i = 1; i < kN - 1; ++i)
        for (int j = 1; j < kN - 1; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double weighted = static_cast<double>(rho(i, j, k))
                    * static_cast<double>(tendency(i, j, k))
                    * terrain.J(i, j, k) * terrain.dx * terrain.dy
                    * terrain.dzeta;
                inventory_rate += weighted;
                absolute_rate += std::abs(weighted);
            }
    REQUIRE(absolute_rate > 0.0);
    REQUIRE(std::abs(inventory_rate) <= 2.0e-6 * absolute_rate);
}

TEST_CASE("stretched-grid SGS uses cell and center distances at their natural locations",
          "[turbulence][sgs][nonuniform-dz][sci074]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    live_rho.resize(kN, kN, kN, 1.0f);
    GridMetrics grid = make_grid();
    for (int k = 0; k < kN; ++k)
        grid.dz[static_cast<std::size_t>(k)] = 40.0 + 7.0 * k;
    std::vector<double> z_face(static_cast<std::size_t>(kN + 1), 0.0);
    std::vector<double> z_center(static_cast<std::size_t>(kN), 0.0);
    for (int k = 0; k < kN; ++k)
    {
        z_face[static_cast<std::size_t>(k + 1)] =
            z_face[static_cast<std::size_t>(k)]
            + grid.dz[static_cast<std::size_t>(k)];
        z_center[static_cast<std::size_t>(k)] = 0.5 * (
            z_face[static_cast<std::size_t>(k)]
          + z_face[static_cast<std::size_t>(k + 1)]);
    }

    constexpr double shear = 0.015;
    constexpr double scalar_gradient = 0.02;
    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 0.0f),
            coefficient(kN, kN, kN, 6.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                u(i, j, k) = static_cast<float>(
                    shear * z_center[static_cast<std::size_t>(k)]);
                w(i, j, k) = static_cast<float>(
                    shear * z_face[static_cast<std::size_t>(k + 1)]);
                theta(i, j, k) = static_cast<float>(
                    300.0 + scalar_gradient
                    * z_center[static_cast<std::size_t>(k)]);
            }

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.rho = &live_rho;
    state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;

    const double strain_sq = eddy_viscosity::compute_strain_sq_centered(
        state, grid, 6, 6, 6);
    REQUIRE(strain_sq == Approx(3.0 * shear * shear).epsilon(2.0e-5));
    // u(z) contributes |S|^2=shear^2 and w(z) contributes 2*shear^2.

    const double frequency =
        eddy_viscosity::compute_brunt_vaisala_frequency(
            state, grid, 6, 6, 6);
    const double expected_n2 = turbulence_constants::g
        / static_cast<double>(theta(6, 6, 6)) * scalar_gradient;
    REQUIRE(frequency * frequency
            == Approx(expected_n2).epsilon(2.0e-5));

    TurbulenceConfig cfg = make_cfg();
    cfg.mode = "vertical_only";
    Field3D du(kN, kN, kN, 0.0f), dv(kN, kN, kN, 0.0f),
            dw(kN, kN, kN, 0.0f), dtheta(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, coefficient, du, dv, dw);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, state, grid, coefficient, theta, dtheta);
    for (int k = 1; k < kN - 2; ++k)
    {
        REQUIRE(du(6, 6, k) == Approx(0.0).margin(2.0e-7));
        REQUIRE(dw(6, 6, k) == Approx(0.0).margin(2.0e-7));
        REQUIRE(dtheta(6, 6, k) == Approx(0.0).margin(2.0e-7));
    }
}

TEST_CASE("stretched-grid SGS w stress uses exact dual mass",
          "[turbulence][sgs][nonuniform-dz][mass][sci103]")
{
    set_cartesian_open();
    rho0_base.assign(kN, 1.0);
    live_rho.resize(kN, kN, kN, 4.0f);
    GridMetrics grid = make_grid();
    grid.dz.assign(kN, 5.0);
    grid.dz[0] = 1.0;
    grid.dz[1] = 3.0;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
        {
            live_rho(i, j, 0) = 1.0f;
            live_rho(i, j, 1) = 2.0f;
        }

    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f),
            nu(kN, kN, kN, 1.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 1; k < kN; ++k)
                w(i, j, k) = 2.0f;

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.rho = &live_rho;
    state.rho0 = &rho0_base;
    state.c_grid = true;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    TurbulenceConfig cfg = make_cfg();
    cfg.mode = "vertical_only";
    Field3D du(kN, kN, kN, 0.0f), dv(kN, kN, kN, 0.0f),
            dw(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu, du, dv, dw);

    const double stress = 2.0 * 2.0 * (2.0 - 0.0) / 3.0;
    const double lower_mass = 0.5 * (1.0 * 1.0 + 2.0 * 3.0);
    const double upper_mass = 0.5 * (2.0 * 3.0 + 4.0 * 5.0);
    REQUIRE(dw(6, 6, 0) == Approx(stress / lower_mass));
    REQUIRE(dw(6, 6, 1) == Approx(-stress / upper_mass));
    REQUIRE(du(6, 6, 1) == 0.0f);
    REQUIRE(dv(6, 6, 1) == 0.0f);
}

TEST_CASE("vertical SGS damps a closed-column cosine mode without horizontal mixing",
          "[turbulence][sgs][vertical_only][analytic][conservation]")
{
    set_cartesian_periodic();
    constexpr int nx = 43, ny = 37, nz = 48;
    constexpr double spacing = 25.0, diffusivity = 7.0;
    const double pi = std::acos(-1.0);
    GridMetrics grid;
    grid.dx = 75.0;
    grid.dy = 125.0;
    grid.dz.assign(nz, spacing);
    std::vector<double> base_density(nz, 1.0);
    Field3D density(nx, ny, nz, 1.0f), coefficient(nx, ny, nz, diffusivity);
    Field3D scalar(nx, ny, nz, 0.0f), tendency(nx, ny, nz, 0.0f);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                scalar(i, j, k) = static_cast<float>(
                    20.0 + 0.1 * i * j + (1.0 + 0.03 * i + 0.02 * j)
                        * std::cos(pi * (k + 0.5) / nz));
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.rho = &density;
    state.rho0 = &base_density;
    state.c_grid = true;
    state.NR = nx;
    state.NTH = ny;
    state.NZ = nz;
    TurbulenceConfig cfg = make_cfg();
    cfg.mode = "vertical_only";
    Field3D serial;
#ifdef _OPENMP
    const int previous_threads = omp_get_max_threads();
#endif
    const double eigenvalue = -4.0 * diffusivity
        * std::pow(std::sin(pi / (2.0 * nz)), 2) / (spacing * spacing);
    for (const int threads : {1, 4})
    {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        eddy_viscosity::compute_scalar_diffusion_tendencies(
            cfg, state, grid, coefficient, scalar, tendency);
        double inventory_rate = 0.0, absolute_rate = 0.0, variance_rate = 0.0;
        for (int i = 1; i < nx - 1; ++i)
            for (int j = 1; j < ny - 1; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const double mode = (1.0 + 0.03 * i + 0.02 * j)
                        * std::cos(pi * (k + 0.5) / nz);
                    const double expected = eigenvalue * mode;
                    // Bound input binary32 quantization in the second difference.
                    const double quantization = 8.0 * std::numeric_limits<float>::epsilon()
                        * (21.0 + 0.1 * i * j + 0.03 * i + 0.02 * j)
                        * diffusivity / (spacing * spacing);
                    REQUIRE(std::abs(tendency(i, j, k) - expected) <= quantization);
                    inventory_rate += tendency(i, j, k);
                    absolute_rate += std::abs(tendency(i, j, k));
                    variance_rate += mode * tendency(i, j, k);
                }
        REQUIRE(absolute_rate > 0.0);
        REQUIRE(variance_rate < 0.0);
        REQUIRE(std::abs(inventory_rate) <= 32.0 * std::numeric_limits<float>::epsilon() * absolute_rate);
        if (threads == 1)
            serial = tendency;
        else
            REQUIRE(std::memcmp(serial.data(), tendency.data(), tendency.size() * sizeof(float)) == 0);
    }
#ifdef _OPENMP
    omp_set_num_threads(previous_threads);
#endif
    set_cartesian_open();
}

TEST_CASE("M15: SGS momentum mixing conserves density-weighted domain momentum",
          "[turbulence][sgs][m15][conservation]")
{
    set_cartesian_open();
    set_stratified_rho0();
    GridMetrics grid = make_grid();
    for (int k = 0; k < kN; ++k)
        grid.dz[static_cast<std::size_t>(k)] = 45.0 + 6.0 * k;
    TurbulenceConfig cfg = make_cfg();

    // Three distinct interior velocity perturbations (permuted indices),
    // each zero within 4 cells of every boundary.
    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                u(i, j, k) = interior_bump(i, j, k, 5.0f);
                v(i, j, k) = interior_bump(j, k, i, -3.0f);
                w(i, j, k) = interior_bump(k, i, j, 2.0f);
            }
    Field3D nu(kN, kN, kN, 0.0f);
    fill_variable_coeff(nu, 5.0f, 3.0f);

    TurbulenceStateView st;
    st.u = &u; st.v = &v; st.w = &w; st.theta = &theta;
    st.rho = &live_rho;
    st.rho0 = &rho0_base;
    st.NR = kN; st.NTH = kN; st.NZ = kN;

    Field3D du, dv, dw;
    du.resize(kN, kN, kN, 0.0f);
    dv.resize(kN, kN, kN, 0.0f);
    dw.resize(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, st, grid, nu, du, dv, dw);

    // d/dt integral(rho_live * vel) is the boundary stress and therefore
    // zero here. Face densities match each staggered velocity component.
    double net_u = 0.0, act_u = 0.0;
    double net_v = 0.0, act_v = 0.0;
    double net_w = 0.0, act_w = 0.0;
    double resolved_ke_work = 0.0;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double dz_k = grid.dz[static_cast<std::size_t>(k)];
                const double wu = (i < kN - 1 ? rho_live_xface(i, j, k)
                                              : rho_live_c(i, j, k))
                                * dz_k * du(i, j, k);
                const double wv = (j < kN - 1 ? rho_live_yface(i, j, k)
                                              : rho_live_c(i, j, k))
                                * dz_k * dv(i, j, k);
                const double w_mass = k < kN - 1
                    ? 0.5 * (rho_live_c(i, j, k) * dz_k
                           + rho_live_c(i, j, k + 1) *
                             grid.dz[static_cast<std::size_t>(k + 1)])
                    : rho_live_c(i, j, k) * dz_k;
                const double ww = w_mass * dw(i, j, k);
                net_u += wu; act_u += std::abs(wu);
                net_v += wv; act_v += std::abs(wv);
                net_w += ww; act_w += std::abs(ww);
                resolved_ke_work +=
                    static_cast<double>(u(i, j, k)) * wu +
                    static_cast<double>(v(i, j, k)) * wv +
                    static_cast<double>(w(i, j, k)) * ww;
            }

    // Non-vacuous: the mixing actually did something.
    REQUIRE(act_u > 1.0e-2);
    REQUIRE(act_v > 1.0e-2);
    REQUIRE(act_w > 1.0e-2);
    // Conservative to fp32 reconstruction noise (a non-conservative form
    // would leave net/act ~ O(0.1-1) under this nu gradient).
    REQUIRE(std::abs(net_u) < 1.0e-3 * act_u);
    REQUIRE(std::abs(net_v) < 1.0e-3 * act_v);
    REQUIRE(std::abs(net_w) < 1.0e-3 * act_w);
    // With no boundary stress work, a positive eddy viscosity may only
    // remove resolved kinetic energy.  Conservation alone would not catch a
    // sign reversal that anti-diffused while retaining zero net momentum.
    REQUIRE(resolved_ke_work < 0.0);
}

TEST_CASE("M15: SGS scalar mixing conserves density-weighted domain scalar",
          "[turbulence][sgs][m15][conservation]")
{
    set_cartesian_open();
    set_stratified_rho0();
    GridMetrics grid = make_grid();
    TurbulenceConfig cfg = make_cfg();

    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f), w(kN, kN, kN, 0.0f),
            theta(kN, kN, kN, 300.0f);
    Field3D K(kN, kN, kN, 0.0f), phi(kN, kN, kN, 285.0f);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
                phi(i, j, k) = 285.0f + interior_bump(i, j, k, 8.0f);
    fill_variable_coeff(K, 4.0f, 2.0f);

    TurbulenceStateView st;
    st.u = &u; st.v = &v; st.w = &w; st.theta = &theta; st.rho = &live_rho;
    st.rho0 = &rho0_base;
    st.NR = kN; st.NTH = kN; st.NZ = kN;

    Field3D dphi;
    dphi.resize(kN, kN, kN, 0.0f);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        cfg, st, grid, K, phi, dphi);

    double net = 0.0, act = 0.0;
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const double wq = rho_live_c(i, j, k) * dphi(i, j, k);
                net += wq;
                act += std::abs(wq);
            }

    REQUIRE(act > 1.0e-2);
    REQUIRE(std::abs(net) < 1.0e-3 * act);
}

TEST_CASE("Smagorinsky momentum tendencies are bit-exact across thread counts",
          "[turbulence][sgs][CPU-019]")
{
#ifdef _OPENMP
    set_cartesian_open();
    set_stratified_rho0();
    const GridMetrics grid = make_grid();
    const TurbulenceConfig cfg = make_cfg();

    Field3D u(kN, kN, kN), v(kN, kN, kN), w(kN, kN, kN),
            theta(kN, kN, kN, 300.0f), nu(kN, kN, kN);
    for (int i = 0; i < kN; ++i)
        for (int j = 0; j < kN; ++j)
            for (int k = 0; k < kN; ++k)
            {
                const float pattern = static_cast<float>((11 * i + 7 * j + 3 * k) % 19);
                u(i, j, k) = 0.3f * pattern - 2.0f;
                v(i, j, k) = 1.5f - 0.2f * pattern;
                w(i, j, k) = (k == kN - 1) ? 0.0f : 0.1f * pattern;
                nu(i, j, k) = 2.0f + 0.4f * pattern;
            }

    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
    state.rho = &live_rho;
    state.rho0 = &rho0_base;
    state.NR = kN; state.NTH = kN; state.NZ = kN;

    Field3D du1, dv1, dw1, du8, dv8, dw8;
    for (Field3D* field : {&du1, &dv1, &dw1, &du8, &dv8, &dw8})
        field->resize(kN, kN, kN, 0.0f);
    const int saved_dynamic = omp_get_dynamic();
    const int saved_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu, du1, dv1, dw1);
    omp_set_num_threads(8);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        cfg, state, grid, nu, du8, dv8, dw8);
    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);

    const Field3D* single[] = {&du1, &dv1, &dw1};
    const Field3D* parallel[] = {&du8, &dv8, &dw8};
    for (std::size_t field = 0; field < std::size(single); ++field)
        for (std::size_t n = 0; n < single[field]->size(); ++n)
            REQUIRE(single[field]->data()[n] == parallel[field]->data()[n]);
#else
    SUCCEED("OpenMP is unavailable in this build");
#endif
}

TEST_CASE("collocated SGS crosses its OpenMP threshold bit-exactly",
          "[turbulence][sgs][agrid][threading][determinism]")
{
#ifdef _OPENMP
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;

    // 34 storage points leave exactly 32^3 physical cells, which is the
    // production threshold for entering the collocated OpenMP path.
    constexpr int n = 34;
    Field3D rho(n, n, n, 1.0f), u(n, n, n, 0.0f),
            v(n, n, n, 0.0f), w(n, n, n, 0.0f),
            theta(n, n, n, 300.0f), viscosity(n, n, n, 1.0f),
            diffusivity(n, n, n, 1.0f);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 1; k < n - 1; ++k)
            {
                const float pattern = static_cast<float>(
                    (13 * i + 7 * j + 5 * k) % 29);
                rho(i, j, k) = 0.8f + 0.01f * pattern;
                u(i, j, k) = -2.0f + 0.15f * pattern;
                v(i, j, k) = 1.0f - 0.11f * pattern;
                w(i, j, k) = -0.5f + 0.07f * pattern;
                theta(i, j, k) = 295.0f + 0.2f * pattern;
                viscosity(i, j, k) = 1.0f + 0.03f * pattern;
                diffusivity(i, j, k) = 1.5f + 0.02f * pattern;
            }

    GridMetrics grid;
    grid.dx = 90.0;
    grid.dy = 120.0;
    grid.dz.assign(n, 60.0);
    grid.staggering = StaggerType::Collocated;
    TurbulenceConfig config;
    config.mode = "3d";
    std::vector<double> rho0(n, 1.0);
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.rho = &rho;
    state.rho0 = &rho0;
    state.c_grid = false;
    state.NR = n;
    state.NTH = n;
    state.NZ = n;

    Field3D du1(n, n, n), dv1(n, n, n), dw1(n, n, n),
            dt1(n, n, n), du4(n, n, n), dv4(n, n, n),
            dw4(n, n, n), dt4(n, n, n);
    const int saved_dynamic = omp_get_dynamic();
    const int saved_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        config, state, grid, viscosity, du1, dv1, dw1);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        config, state, grid, diffusivity, theta, dt1);
    omp_set_num_threads(4);
    eddy_viscosity::compute_momentum_diffusion_tendencies(
        config, state, grid, viscosity, du4, dv4, dw4);
    eddy_viscosity::compute_scalar_diffusion_tendencies(
        config, state, grid, diffusivity, theta, dt4);
    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);

    const Field3D* single[] = {&du1, &dv1, &dw1, &dt1};
    const Field3D* parallel[] = {&du4, &dv4, &dw4, &dt4};
    for (std::size_t field = 0; field < std::size(single); ++field)
        for (std::size_t index = 0; index < single[field]->size(); ++index)
            REQUIRE(single[field]->data()[index]
                    == parallel[field]->data()[index]);
#else
    SUCCEED("OpenMP is unavailable in this build");
#endif
}

TEST_CASE("SGS diffusion rejects invalid coefficients before publishing output",
          "[turbulence][sgs][validation][rollback]")
{
    set_cartesian_open();
    set_stratified_rho0();
    GridMetrics grid = make_grid();
    const TurbulenceConfig cfg = make_cfg();
    Field3D u(kN, kN, kN, 0.0f), v(kN, kN, kN, 0.0f),
            w(kN, kN, kN, 0.0f), theta(kN, kN, kN, 300.0f);
    Field3D K(kN, kN, kN, 2.0f), phi(kN, kN, kN, 300.0f);
    Field3D output(kN, kN, kN, 17.0f);
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u; state.v = &v; state.w = &w; state.theta = &theta;
    state.rho = &live_rho;
    state.rho0 = &rho0_base;
    state.NR = kN; state.NTH = kN; state.NZ = kN;

    SECTION("negative diffusivity")
    {
        K(4, 4, 4) = -1.0f;
        REQUIRE_THROWS_AS(
            eddy_viscosity::compute_scalar_diffusion_tendencies(
                cfg, state, grid, K, phi, output),
            std::invalid_argument);
    }

    SECTION("invalid horizontal metric")
    {
        grid.dx = 0.0;
        REQUIRE_THROWS_AS(
            eddy_viscosity::compute_scalar_diffusion_tendencies(
                cfg, state, grid, K, phi, output),
            std::invalid_argument);
    }

    for (std::size_t index = 0; index < output.size(); ++index)
    {
        REQUIRE(output.data()[index] == 17.0f);
    }
}

TEST_CASE("flat A-grid SGS owns only physical cells and closes momentum and scalar budgets",
          "[turbulence][sgs][agrid][conservation][closure]")
{
    global_coordinate_system = CoordinateSystem::Cartesian;
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;

    constexpr int n = 8;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    Field3D rho(n, n, n, poison), u(n, n, n, poison),
            v(n, n, n, poison), w(n, n, n, poison),
            theta(n, n, n, poison), qv(n, n, n, poison),
            viscosity(n, n, n, poison), diffusivity(n, n, n, poison);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 1; k < n - 1; ++k)
            {
                rho(i, j, k) = static_cast<float>(
                    0.9 + 0.03 * i + 0.02 * j + 0.01 * k);
                u(i, j, k) = static_cast<float>(
                    2.0 + 0.11 * i * i - 0.07 * j + 0.02 * j * k);
                v(i, j, k) = static_cast<float>(
                    -1.0 + 0.05 * i - 0.09 * j * j + 0.03 * i * k);
                w(i, j, k) = static_cast<float>(
                    0.4 - 0.04 * i * j + 0.06 * k * k);
                theta(i, j, k) = static_cast<float>(
                    299.0 + 0.2 * i - 0.15 * j + 0.08 * k * k);
                qv(i, j, k) = 0.01f;
                viscosity(i, j, k) = static_cast<float>(
                    1.0 + 0.04 * i + 0.03 * j + 0.02 * k);
                diffusivity(i, j, k) = static_cast<float>(
                    1.5 + 0.02 * i + 0.01 * j + 0.03 * k);
            }

    GridMetrics grid;
    grid.dx = 3.0;
    grid.dy = 4.0;
    grid.dz.assign(n, 5.0);
    grid.staggering = StaggerType::Collocated;
    TurbulenceConfig config;
    config.mode = "3d";
    std::vector<double> rho0(n, 1.0);
    TurbulenceStateView state;
    tmv::bind_flow_execution(state, global_simulation_state());
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.qv = &qv;
    state.rho = &rho;
    state.rho0 = &rho0;
    state.c_grid = false;
    state.NR = n;
    state.NTH = n;
    state.NZ = n;

    for (const char* mode : {"3d", "horizontal_only", "vertical_only"})
        for (const bool periodic : {false, true})
        {
            INFO("mode=" << mode << ", periodic=" << periodic);
            const auto boundary = periodic
                ? tmv::init::config::LateralBC::Periodic
                : tmv::init::config::LateralBC::ZeroGradient;
            global_lateral_bc_west = boundary;
            global_lateral_bc_east = boundary;
            global_lateral_bc_south = boundary;
            global_lateral_bc_north = boundary;
            config.mode = mode;

            Field3D du(n, n, n, 17.0f), dv(n, n, n, 17.0f),
                    dw(n, n, n, 17.0f), dtheta(n, n, n, 17.0f);
            REQUIRE_NOTHROW(
                eddy_viscosity::compute_momentum_diffusion_tendencies(
                    config, state, grid, viscosity, du, dv, dw));
            REQUIRE_NOTHROW(
                eddy_viscosity::compute_scalar_diffusion_tendencies(
                    config, state, grid, diffusivity, theta, dtheta));

            double momentum[3] = {0.0, 0.0, 0.0};
            double kinetic_work = 0.0;
            double scalar_inventory = 0.0;
            double scalar_work = 0.0;
            double activity = 0.0;
            for (int i = 1; i < n - 1; ++i)
                for (int j = 1; j < n - 1; ++j)
                    for (int k = 1; k < n - 1; ++k)
                    {
                        const double mass =
                            static_cast<double>(rho(i, j, k))
                            * grid.dx * grid.dy
                            * grid.dz[static_cast<std::size_t>(k)];
                        const double values[] = {
                            du(i, j, k), dv(i, j, k), dw(i, j, k)};
                        const double winds[] = {
                            u(i, j, k), v(i, j, k), w(i, j, k)};
                        for (int component = 0; component < 3; ++component)
                        {
                            momentum[component] += mass * values[component];
                            kinetic_work +=
                                mass * winds[component] * values[component];
                            activity += std::abs(values[component]);
                        }
                        scalar_inventory += mass * dtheta(i, j, k);
                        scalar_work += mass
                            * static_cast<double>(theta(i, j, k))
                            * dtheta(i, j, k);
                    }
            REQUIRE(activity > 0.0);
            REQUIRE(momentum[0] == Approx(0.0).margin(2.0e-3));
            REQUIRE(momentum[1] == Approx(0.0).margin(2.0e-3));
            REQUIRE(momentum[2] == Approx(0.0).margin(2.0e-3));
            REQUIRE(kinetic_work < 0.0);
            REQUIRE(scalar_inventory == Approx(0.0).margin(2.0e-3));
            REQUIRE(scalar_work < 0.0);

            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int k = 0; k < n; ++k)
                        if (i == 0 || i == n - 1
                            || j == 0 || j == n - 1
                            || k == 0 || k == n - 1)
                {
                    REQUIRE(du(i, j, k) == 0.0f);
                    REQUIRE(dv(i, j, k) == 0.0f);
                    REQUIRE(dw(i, j, k) == 0.0f);
                    REQUIRE(dtheta(i, j, k) == 0.0f);
                }
        }
}

TEST_CASE("C-grid SGS scalar walls ignore stale ghost values", "[turbulence][sgs][closed_scalar]")
{
    constexpr int n = 6;
    Field3D rho(n, n, n, 1.0f), scalar(n, n, n, 11.0f);
    Field3D diffusivity(n, n, n, 2.0f), tendency(n, n, n, -99.0f);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
                scalar(i, j, k) = 3.0f;
    GridMetrics grid;
    grid.dx = 4.0;
    grid.dy = 4.0;
    grid.dz.assign(n, 4.0);
    TurbulenceConfig config;
    config.mode = "horizontal_only";
    std::vector<double> rho0(n, 1.0);
    TurbulenceStateView state;
    state.rho = &rho;
    state.rho0 = &rho0;
    state.c_grid = true;
    state.NR = n;
    state.NTH = n;
    state.NZ = n;
    state.open_lateral = {false, false, false, false};
    eddy_viscosity::compute_scalar_diffusion_tendencies(config, state, grid, diffusivity, scalar, tendency);
    for (std::size_t index = 0; index < tendency.size(); ++index)
        CHECK(tendency.data()[index] == 0.0f);
    state.open_lateral[0] = true;
    eddy_viscosity::compute_scalar_diffusion_tendencies(config, state, grid, diffusivity, scalar, tendency);
    for (int i = 1; i < n - 1; ++i)
        for (int j = 1; j < n - 1; ++j)
            for (int k = 0; k < n; ++k)
                CHECK(tendency(i, j, k) == (i == 1 ? 1.0f : 0.0f));
    state.periodic_x = true;
    eddy_viscosity::compute_scalar_diffusion_tendencies(config, state, grid, diffusivity, scalar, tendency);
    for (std::size_t index = 0; index < tendency.size(); ++index)
        CHECK(tendency.data()[index] == 0.0f);
}
