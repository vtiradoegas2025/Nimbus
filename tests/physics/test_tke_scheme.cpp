#include "catch2/catch.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation.hpp"
#include "terrain/conservative_transport.hpp"
#include "turbulence/schemes/tke/tke.hpp"
#include "turbulence/base/eddy_viscosity.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

constexpr int kNx = 8;
constexpr int kNy = 8;
constexpr int kNz = 10;

struct TKEFixture
{
    Field3D u{kNx, kNy, kNz, 0.0f};
    Field3D v{kNx, kNy, kNz, 0.0f};
    Field3D w{kNx, kNy, kNz, 0.0f};
    Field3D theta{kNx, kNy, kNz, 300.0f};
    Field3D rho{kNx, kNy, kNz, 1.0f};
    Field3D e{kNx, kNy, kNz, 0.5f};
    Field3D qv{kNx, kNy, kNz, 0.0f};

    GridMetrics grid;
    TurbulenceConfig cfg;
    TurbulenceStateView state;

    TKEFixture()
    {
        global_coordinate_system = CoordinateSystem::Cartesian;
        global_lateral_bc_west = tmv::init::config::LateralBC::ZeroGradient;
        global_lateral_bc_east = tmv::init::config::LateralBC::ZeroGradient;
        global_lateral_bc_south = tmv::init::config::LateralBC::ZeroGradient;
        global_lateral_bc_north = tmv::init::config::LateralBC::ZeroGradient;

        rho0_base.resize(kNz);
        for (int k = 0; k < kNz; ++k)
        {
            rho0_base[static_cast<std::size_t>(k)] =
                1.2 * std::exp(-static_cast<double>(k) / 8.0);
            for (int i = 0; i < kNx; ++i)
                for (int j = 0; j < kNy; ++j)
                    rho(i, j, k) = static_cast<float>(
                        rho0_base[static_cast<std::size_t>(k)]
                        * (0.85 + 0.02 * i + 0.01 * j));
        }

        grid.dx = 100.0;
        grid.dy = 100.0;
        grid.dz.assign(kNz, 100.0);

        cfg.mode = "3d";
        cfg.filter_width = "cubic_root";
        cfg.Pr_t = 1.0 / 3.0;
        cfg.Sc_t = 1.0 / 3.0;
        cfg.nu_t_max = 1000.0;
        cfg.K_max = 1000.0;

        state.u = &u;
        state.v = &v;
        state.w = &w;
        state.theta = &theta;
        state.qv = &qv;
        state.rho = &rho;
        state.tke = &e;
        state.rho0 = &rho0_base;
        state.NR = kNx;
        state.NTH = kNy;
        state.NZ = kNz;
    }
};

void run_tke(TKEFixture& f,
             TurbulenceTendencies& tendencies,
             TurbulenceDiagnostics& diagnostics)
{
    TKEScheme scheme;
    scheme.initialize(f.cfg);
    scheme.compute(f.cfg, f.grid, f.state, tendencies, &diagnostics);
}

double reconstructed_diffusion(const TurbulenceTendencies& tendencies,
                               const TurbulenceDiagnostics& diagnostics,
                               int i, int j, int k)
{
    return static_cast<double>(tendencies.dtkedt_sgs(i, j, k))
         - static_cast<double>(diagnostics.shear_prod(i, j, k))
         - static_cast<double>(diagnostics.buoy_prod(i, j, k))
         + static_cast<double>(diagnostics.dissipation(i, j, k));
}

TerrainMetrics3D make_sloping_metrics()
{
    TerrainMetrics3D metrics;
    metrics.dx = 100.0;
    metrics.dy = 100.0;
    metrics.dzeta = 80.0;
    constexpr double jacobian = 1.25;
    constexpr double slope_x = 0.08;
    constexpr double slope_y = -0.04;
    metrics.z.resize(kNx, kNy, kNz, 0.0);
    metrics.J.resize(kNx, kNy, kNz, jacobian);
    metrics.mx.resize(kNx, kNy, kNz, -slope_x / jacobian);
    metrics.my.resize(kNx, kNy, kNz, -slope_y / jacobian);
    metrics.zeta.resize(kNx, kNy, kNz, 0.0);
    metrics.z_face.resize(kNx, kNy, kNz + 1, 0.0);
    metrics.Jx_face.resize(kNx + 1, kNy, kNz, jacobian);
    metrics.Jy_face.resize(kNx, kNy + 1, kNz, jacobian);
    metrics.Gx_face.resize(kNx, kNy, kNz + 1, -slope_x);
    metrics.Gy_face.resize(kNx, kNy, kNz + 1, -slope_y);
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
        {
            const double ground = slope_x * i * metrics.dx +
                                  slope_y * j * metrics.dy;
            for (int k = 0; k <= kNz; ++k)
                metrics.z_face(i, j, k) =
                    ground + jacobian * metrics.dzeta * k;
            for (int k = 0; k < kNz; ++k)
            {
                metrics.z(i, j, k) = ground +
                    jacobian * metrics.dzeta * (k + 0.5);
                metrics.zeta(i, j, k) = metrics.dzeta * (k + 0.5);
            }
        }
    return metrics;
}

bool bitwise_equal(const Field3D& left, const Field3D& right)
{
    return left.size_r() == right.size_r()
        && left.size_th() == right.size_th()
        && left.size_z() == right.size_z()
        && left.size() == right.size()
        && std::memcmp(left.data(), right.data(),
                       left.size() * sizeof(float)) == 0;
}

bool bitwise_equal(const TurbulenceTendencies& left,
                   const TurbulenceTendencies& right)
{
    return bitwise_equal(left.dudt_sgs, right.dudt_sgs)
        && bitwise_equal(left.dvdt_sgs, right.dvdt_sgs)
        && bitwise_equal(left.dwdt_sgs, right.dwdt_sgs)
        && bitwise_equal(left.dthetadt_sgs, right.dthetadt_sgs)
        && bitwise_equal(left.dqvdt_sgs, right.dqvdt_sgs)
        && bitwise_equal(left.dtkedt_sgs, right.dtkedt_sgs)
        && std::memcmp(&left.mapped_boundary_power_w,
                       &right.mapped_boundary_power_w,
                       sizeof(double)) == 0
        && left.resident_source_unchanged
            == right.resident_source_unchanged;
}

bool bitwise_equal(const TurbulenceDiagnostics& left,
                   const TurbulenceDiagnostics& right)
{
    return bitwise_equal(left.nu_t, right.nu_t)
        && bitwise_equal(left.K_theta, right.K_theta)
        && bitwise_equal(left.K_q, right.K_q)
        && bitwise_equal(left.K_tke, right.K_tke)
        && bitwise_equal(left.shear_prod, right.shear_prod)
        && bitwise_equal(left.buoy_prod, right.buoy_prod)
        && bitwise_equal(left.dissipation, right.dissipation)
        && bitwise_equal(left.Cs_eff, right.Cs_eff)
        && bitwise_equal(left.Pr_t_eff, right.Pr_t_eff);
}

void fill(TurbulenceTendencies& tendencies, float value)
{
    for (Field3D* field : {
             &tendencies.dudt_sgs, &tendencies.dvdt_sgs,
             &tendencies.dwdt_sgs, &tendencies.dthetadt_sgs,
             &tendencies.dqvdt_sgs, &tendencies.dtkedt_sgs})
    {
        field->resize(2, 3, 4, value);
    }
    tendencies.mapped_boundary_power_w = 17.25;
    tendencies.resident_source_unchanged = true;
}

void fill(TurbulenceDiagnostics& diagnostics, float value)
{
    for (Field3D* field : {
             &diagnostics.nu_t, &diagnostics.K_theta,
             &diagnostics.K_q, &diagnostics.K_tke,
             &diagnostics.shear_prod, &diagnostics.buoy_prod,
             &diagnostics.dissipation, &diagnostics.Cs_eff,
             &diagnostics.Pr_t_eff})
    {
        field->resize(2, 3, 4, value);
    }
}

}  // namespace

TEST_CASE("SCI-010: uniform TKE is a free stream of the diffusion operator",
          "[turbulence][tke][sci010][freestream]")
{
    TKEFixture f;
    f.e.fill(0.8f);

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);

    double max_abs_diffusion = 0.0;
    double max_dissipation = 0.0;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                max_abs_diffusion = std::max(
                    max_abs_diffusion,
                    std::abs(reconstructed_diffusion(
                        tendencies, diagnostics, i, j, k)));
                max_dissipation = std::max(
                    max_dissipation,
                    static_cast<double>(diagnostics.dissipation(i, j, k)));
            }

    REQUIRE(max_dissipation > 0.0);
    REQUIRE(max_abs_diffusion < 1.0e-7);
}

TEST_CASE("SCI-010: zero TKE is an exact no-mixing equilibrium",
          "[turbulence][tke][sci010][zero-energy]")
{
    TKEFixture f;
    f.e.fill(0.0f);

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);

    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                REQUIRE(diagnostics.nu_t(i, j, k) == 0.0f);
                REQUIRE(diagnostics.K_theta(i, j, k) == 0.0f);
                REQUIRE(diagnostics.K_q(i, j, k) == 0.0f);
                REQUIRE(diagnostics.K_tke(i, j, k) == 0.0f);
                REQUIRE(diagnostics.dissipation(i, j, k) == 0.0f);
                REQUIRE(tendencies.dtkedt_sgs(i, j, k) == 0.0f);
            }
}

TEST_CASE("SCI-010: variable-K TKE diffusion conserves density-weighted TKE",
          "[turbulence][tke][sci010][conservation]")
{
    TKEFixture f;
    f.cfg.mode = "vertical_only";
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                const double wave = std::sin(
                    3.14159265358979323846 * static_cast<double>(k)
                    / static_cast<double>(kNz - 1));
                f.e(i, j, k) = static_cast<float>(0.15 + 0.85 * wave * wave);
            }

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);

    double k_min = 1.0e30;
    double k_max = -1.0e30;
    double net = 0.0;
    double activity = 0.0;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                const double K = diagnostics.K_tke(i, j, k);
                k_min = std::min(k_min, K);
                k_max = std::max(k_max, K);

                const double weighted =
                    rho0_base[static_cast<std::size_t>(k)]
                    * reconstructed_diffusion(tendencies, diagnostics, i, j, k);
                net += weighted;
                activity += std::abs(weighted);
            }

    REQUIRE(k_max - k_min > 0.01);
    REQUIRE(activity > 1.0e-5);
    REQUIRE(std::abs(net) < 1.0e-3 * activity);
}

TEST_CASE("stochastic TKE transport defers and rebuilds conservative writers",
          "[turbulence][tke][chaos][coefficients]")
{
    TKEFixture f;
    f.state.defer_transport_tendency_build = true;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                f.u(i, j, k) = static_cast<float>(0.3 * i + 0.1 * k);
                f.v(i, j, k) = static_cast<float>(-0.2 * j + 0.05 * k);
                f.e(i, j, k) = static_cast<float>(
                    0.2 + 0.01 * i + 0.02 * j + 0.03 * k);
            }

    TKEScheme scheme;
    scheme.initialize(f.cfg);
    TurbulenceTendencies tendencies;
    scheme.compute(f.cfg, f.grid, f.state, tendencies, nullptr);
    for (std::size_t index = 0; index < tendencies.dudt_sgs.size(); ++index)
    {
        REQUIRE(tendencies.dudt_sgs.data()[index] == 0.0f);
        REQUIRE(tendencies.dthetadt_sgs.data()[index] == 0.0f);
        REQUIRE(tendencies.dtkedt_sgs.data()[index] == 0.0f);
    }

    tmv::TurbulenceTransportFields fields;
    REQUIRE(scheme.mutable_realized_transport_fields(fields));
    REQUIRE(fields.eddy_viscosity != nullptr);
    REQUIRE(fields.theta_diffusivity != nullptr);
    REQUIRE(fields.moisture_diffusivity != nullptr);
    REQUIRE(fields.tke_diffusivity != nullptr);
    for (std::size_t index = 0; index < fields.eddy_viscosity->size(); ++index)
    {
        const float multiplier = static_cast<float>(
            0.75 + 0.25 * static_cast<double>(index % 7) / 6.0);
        fields.eddy_viscosity->data()[index] *= multiplier;
        fields.theta_diffusivity->data()[index] *= multiplier;
        fields.moisture_diffusivity->data()[index] *= multiplier;
        fields.tke_diffusivity->data()[index] *= multiplier;
    }
    REQUIRE(scheme.rebuild_from_realized_transport(
        f.cfg, f.grid, f.state, tendencies));

    double momentum_activity = 0.0;
    double scalar_activity = 0.0;
    for (std::size_t index = 0; index < tendencies.dudt_sgs.size(); ++index)
    {
        momentum_activity += std::abs(
            static_cast<double>(tendencies.dudt_sgs.data()[index]));
        momentum_activity += std::abs(
            static_cast<double>(tendencies.dvdt_sgs.data()[index]));
        scalar_activity += std::abs(
            static_cast<double>(tendencies.dtkedt_sgs.data()[index]));
    }
    REQUIRE(momentum_activity > 0.0);
    REQUIRE(scalar_activity > 0.0);
    REQUIRE_FALSE(tendencies.resident_source_unchanged);
}

TEST_CASE("SCI-010: diagnosed dissipation is the prognostic decay term",
          "[turbulence][tke][sci010][dissipation]")
{
    TKEFixture f;
    f.cfg.mode = "vertical_only";
    f.cfg.filter_width = "dx";

    // Each column is vertically uniform, so vertical diffusion is zero. The
    // horizontal variation also verifies that vertical_only is respected.
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
                f.e(i, j, k) = 0.2f + 0.1f * static_cast<float>(i);

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);

    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                const bool physical = i >= 1 && i < kNx - 1
                    && j >= 1 && j < kNy - 1;
                if (!physical)
                {
                    CHECK(diagnostics.dissipation(i, j, k) == 0.0f);
                    CHECK(tendencies.dtkedt_sgs(i, j, k) == 0.0f);
                    continue;
                }
                const double e = static_cast<double>(f.e(i, j, k));
                const double expected = 0.70 * std::pow(e, 1.5) / 100.0;
                CHECK(diagnostics.dissipation(i, j, k)
                      == Approx(expected).margin(2.0e-7));
                CHECK(tendencies.dtkedt_sgs(i, j, k)
                      == Approx(-diagnostics.dissipation(i, j, k))
                             .margin(1.0e-7));
            }
}

TEST_CASE("Deardorff TKE coefficients match stable and unstable limits",
          "[turbulence][tke][deardorff][reference]")
{
    TKEFixture unstable;
    unstable.cfg.stability_correction = "tke_based";
    unstable.cfg.filter_width = "dx";
    unstable.e.fill(0.5f);
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
                unstable.theta(i, j, k) =
                    static_cast<float>(302.0 - 0.2 * k);

    TurbulenceTendencies unstable_tendencies;
    TurbulenceDiagnostics unstable_diagnostics;
    run_tke(unstable, unstable_tendencies, unstable_diagnostics);

    constexpr int i = 3;
    constexpr int j = 3;
    constexpr int k = 4;
    const double energy = unstable.e(i, j, k);
    const double unstable_length = unstable.grid.dx;
    const double unstable_viscosity =
        0.1 * unstable_length * std::sqrt(energy);
    CHECK(unstable_diagnostics.nu_t(i, j, k)
          == Approx(unstable_viscosity).margin(2.0e-6));
    CHECK(unstable_diagnostics.K_theta(i, j, k)
          == Approx(3.0 * unstable_viscosity).margin(5.0e-6));
    CHECK(unstable_diagnostics.K_tke(i, j, k)
          == Approx(2.0 * unstable_viscosity).margin(5.0e-6));
    CHECK(unstable_diagnostics.Pr_t_eff(i, j, k)
          == Approx(1.0 / 3.0).margin(2.0e-7));
    CHECK(unstable_diagnostics.dissipation(i, j, k)
          == Approx(0.70 * std::pow(energy, 1.5) / unstable_length)
                 .margin(2.0e-7));
    CHECK(unstable_diagnostics.buoy_prod(i, j, k) > 0.0f);

    TKEFixture stable;
    stable.cfg.stability_correction = "tke_based";
    stable.cfg.filter_width = "dx";
    stable.e.fill(0.5f);
    constexpr double gradient = 0.02;
    for (int ii = 0; ii < kNx; ++ii)
        for (int jj = 0; jj < kNy; ++jj)
            for (int kk = 0; kk < kNz; ++kk)
                stable.theta(ii, jj, kk) = static_cast<float>(
                    300.0 + gradient * stable.grid.dz.front() * kk);

    TurbulenceTendencies stable_tendencies;
    TurbulenceDiagnostics stable_diagnostics;
    run_tke(stable, stable_tendencies, stable_diagnostics);

    const double theta = stable.theta(i, j, k);
    const double frequency = std::sqrt(
        turbulence_constants::g / theta * gradient);
    const double stable_length = 0.76 * std::sqrt(energy) / frequency;
    REQUIRE(stable_length < stable.grid.dx);
    const double stable_viscosity =
        0.1 * stable_length * std::sqrt(energy);
    const double length_ratio = stable_length / stable.grid.dx;
    const double heat_ratio = 1.0 + 2.0 * length_ratio;
    const double dissipation_coefficient = 0.19 + 0.51 * length_ratio;
    CHECK(stable_diagnostics.nu_t(i, j, k)
          == Approx(stable_viscosity).margin(2.0e-6));
    CHECK(stable_diagnostics.K_theta(i, j, k)
          == Approx(heat_ratio * stable_viscosity).margin(5.0e-6));
    CHECK(stable_diagnostics.K_tke(i, j, k)
          == Approx(2.0 * stable_viscosity).margin(5.0e-6));
    CHECK(stable_diagnostics.Pr_t_eff(i, j, k)
          == Approx(1.0 / heat_ratio).margin(2.0e-7));
    CHECK(stable_diagnostics.dissipation(i, j, k)
          == Approx(dissipation_coefficient * std::pow(energy, 1.5)
                    / stable_length).margin(2.0e-7));
    CHECK(stable_diagnostics.buoy_prod(i, j, k) < 0.0f);
}

TEST_CASE("TKE stability selector controls the Deardorff stable length",
          "[turbulence][tke][stability][selector]")
{
    TKEFixture neutral;
    neutral.cfg.filter_width = "dx";
    neutral.cfg.stability_correction = "none";
    neutral.e.fill(0.5f);
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
                neutral.theta(i, j, k) = static_cast<float>(300.0 + 2.0 * k);
    TKEFixture corrected = neutral;
    corrected.state.u = &corrected.u;
    corrected.state.v = &corrected.v;
    corrected.state.w = &corrected.w;
    corrected.state.theta = &corrected.theta;
    corrected.state.qv = &corrected.qv;
    corrected.state.rho = &corrected.rho;
    corrected.state.tke = &corrected.e;
    corrected.cfg.stability_correction = "tke_based";

    TurbulenceTendencies neutral_tendencies, corrected_tendencies;
    TurbulenceDiagnostics neutral_diagnostics, corrected_diagnostics;
    run_tke(neutral, neutral_tendencies, neutral_diagnostics);
    run_tke(corrected, corrected_tendencies, corrected_diagnostics);

    constexpr int i = 3;
    constexpr int j = 3;
    constexpr int k = 4;
    REQUIRE(neutral_diagnostics.nu_t(i, j, k) == Approx(
        0.1 * neutral.grid.dx * std::sqrt(0.5)).margin(2.0e-6));
    REQUIRE(corrected_diagnostics.nu_t(i, j, k)
            < neutral_diagnostics.nu_t(i, j, k));
    REQUIRE(corrected_diagnostics.Pr_t_eff(i, j, k)
            > neutral_diagnostics.Pr_t_eff(i, j, k));
}

TEST_CASE("TKE failures preserve caller and realized publications",
          "[turbulence][tke][transaction]")
{
    TKEFixture fixture;
    TKEScheme scheme;
    scheme.initialize(fixture.cfg);
    TurbulenceTendencies valid_tendencies;
    TurbulenceDiagnostics valid_diagnostics;
    scheme.compute(fixture.cfg, fixture.grid, fixture.state,
                   valid_tendencies, &valid_diagnostics);

    const float realized_before = *scheme.realized_eddy_viscosity()->data();
    TurbulenceTendencies sentinel_tendencies;
    TurbulenceDiagnostics sentinel_diagnostics;
    fill(sentinel_tendencies, -17.0f);
    fill(sentinel_diagnostics, 23.0f);
    const TurbulenceTendencies expected_tendencies = sentinel_tendencies;
    const TurbulenceDiagnostics expected_diagnostics = sentinel_diagnostics;

    fixture.theta(kNx - 2, kNy - 2, kNz - 2) =
        std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS_AS(
        scheme.compute(fixture.cfg, fixture.grid, fixture.state,
                       sentinel_tendencies, &sentinel_diagnostics),
        std::invalid_argument);
    CHECK(bitwise_equal(sentinel_tendencies, expected_tendencies));
    CHECK(bitwise_equal(sentinel_diagnostics, expected_diagnostics));
    CHECK(*scheme.realized_eddy_viscosity()->data() == realized_before);

    fixture.theta(kNx - 2, kNy - 2, kNz - 2) = 300.0f;
    tmv::TurbulenceTransportFields fields;
    REQUIRE(scheme.mutable_realized_transport_fields(fields));
    fields.eddy_viscosity->data()[0] =
        static_cast<float>(fixture.cfg.nu_t_max + 10.0);
    fields.tke_diffusivity->data()[fields.tke_diffusivity->size() - 1] =
        std::numeric_limits<float>::quiet_NaN();
    const float unclamped = fields.eddy_viscosity->data()[0];
    REQUIRE_THROWS_AS(
        scheme.rebuild_from_realized_transport(
            fixture.cfg, fixture.grid, fixture.state, sentinel_tendencies),
        std::runtime_error);
    CHECK(fields.eddy_viscosity->data()[0] == unclamped);
    CHECK(bitwise_equal(sentinel_tendencies, expected_tendencies));
}

TEST_CASE("SCI-010: linear shear production uses one strain contraction",
          "[turbulence][tke][sci010][shear]")
{
    TKEFixture f;
    constexpr double shear_rate = 0.02;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
                f.u(i, j, k) = static_cast<float>(
                    shear_rate * f.grid.dy * static_cast<double>(j));

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);

    constexpr int i = 3;
    constexpr int j = 3;
    constexpr int k = 3;
    const double nu_t = diagnostics.nu_t(i, j, k);
    REQUIRE(nu_t > 0.0);
    REQUIRE(diagnostics.shear_prod(i, j, k)
            == Approx(nu_t * shear_rate * shear_rate).margin(1.0e-8));
}

TEST_CASE("SCI-010: TKE shear production closes resolved stress work",
          "[turbulence][tke][sci010][energy-pair]")
{
    TKEFixture f;
    f.state.periodic_x = true;
    f.state.periodic_y = true;
    f.rho.fill(1.0f);
    std::fill(rho0_base.begin(), rho0_base.end(), 1.0);
    f.e.fill(0.4f);
    global_lateral_bc_west = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_east = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_south = tmv::init::config::LateralBC::Periodic;
    global_lateral_bc_north = tmv::init::config::LateralBC::Periodic;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                const int physical_i = i == 0 ? kNx - 2
                    : (i == kNx - 1 ? 1 : i);
                f.u(i, j, k) = 0.0f;
                f.v(i, j, k) = static_cast<float>(
                    1.5 * std::sin(
                        2.0 * 3.14159265358979323846
                        * (physical_i - 1) / (kNx - 2)));
                f.w(i, j, k) = 0.0f;
            }

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);

    const double cell_volume = f.grid.dx * f.grid.dy * f.grid.dz.front();
    double resolved_work_w = 0.0;
    double tke_production_w = 0.0;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                const double w_dual_volume = k + 1 < kNz
                    ? 0.5 * cell_volume + 0.5 * cell_volume
                    : cell_volume;
                resolved_work_w += cell_volume * (
                    static_cast<double>(f.u(i, j, k))
                        * tendencies.dudt_sgs(i, j, k)
                    + static_cast<double>(f.v(i, j, k))
                        * tendencies.dvdt_sgs(i, j, k));
                resolved_work_w += w_dual_volume
                    * static_cast<double>(f.w(i, j, k))
                    * tendencies.dwdt_sgs(i, j, k);
                if (i >= 1 && i < kNx - 1 && j >= 1 && j < kNy - 1
                    && k < kNz - 1)
                    tke_production_w += cell_volume
                        * static_cast<double>(diagnostics.shear_prod(i, j, k));
            }

    REQUIRE(resolved_work_w < 0.0);
    REQUIRE(tke_production_w > 0.0);
    const double transfer_scale = std::max(
        std::abs(resolved_work_w), std::abs(tke_production_w));
    INFO("resolved_work_w=" << resolved_work_w
         << ", tke_production_w=" << tke_production_w);
    REQUIRE(std::abs(resolved_work_w + tke_production_w)
            <= 2.0e-5 * transfer_scale);
}

TEST_CASE("TKE production sees staggered two-level vertical shear",
          "[turbulence][tke][staggered][sci076]")
{
    TKEFixture f;
    constexpr double amplitude = 8.0;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
                f.u(i, j, k) = static_cast<float>(
                    (k % 2 == 0) ? amplitude : -amplitude);

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);

    constexpr int i = 3;
    constexpr int j = 3;
    constexpr int k = 4;
    const double strain_sq = eddy_viscosity::compute_strain_sq_centered(
        f.state, f.grid, i, j, k);
    REQUIRE(strain_sq > 0.0);
    CHECK(diagnostics.shear_prod(i, j, k) == Approx(
        static_cast<double>(diagnostics.nu_t(i, j, k)) * strain_sq)
        .epsilon(2.0e-6));
}

TEST_CASE("TKE closure owns the flat collocated prognostic equation",
          "[turbulence][tke][agrid][collocated]")
{
    TKEFixture f;
    f.state.c_grid = false;
    f.grid.staggering = StaggerType::Collocated;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                f.u(i, j, k) = static_cast<float>(0.2 * i + 0.05 * k);
                f.v(i, j, k) = static_cast<float>(-0.15 * j + 0.03 * k);
                f.e(i, j, k) = static_cast<float>(0.2 + 0.01 * (i + j + k));
            }

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    REQUIRE_NOTHROW(run_tke(f, tendencies, diagnostics));

    const PhysicalCellDomain domain = physical_scalar_domain(
        kNx, kNy, kNz, StaggerType::Collocated);
    double activity = 0.0;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                const bool physical = i >= domain.i_begin && i < domain.i_end &&
                                      j >= domain.j_begin && j < domain.j_end &&
                                      k >= domain.k_begin && k < domain.k_end;
                for (const Field3D* field : {&tendencies.dudt_sgs,
                                             &tendencies.dvdt_sgs,
                                             &tendencies.dwdt_sgs,
                                             &tendencies.dthetadt_sgs,
                                             &tendencies.dtkedt_sgs,
                                             &diagnostics.nu_t})
                    REQUIRE(std::isfinite(static_cast<double>((*field)(i, j, k))));
                if (physical)
                    activity += std::abs(static_cast<double>(tendencies.dtkedt_sgs(i, j, k)));
                else
                {
                    REQUIRE(tendencies.dtkedt_sgs(i, j, k) == 0.0f);
                    REQUIRE(diagnostics.nu_t(i, j, k) == 0.0f);
                }
            }
    REQUIRE(activity > 0.0);
}

TEST_CASE("TKE closure owns mapped C-grid transport and production",
          "[turbulence][tke][terrain][mapped]")
{
    TKEFixture f;
    TerrainMetrics3D metrics = make_sloping_metrics();
    f.grid.dx = metrics.dx;
    f.grid.dy = metrics.dy;
    f.grid.dz.assign(kNz, metrics.dzeta * 1.25);
    f.grid.terrain_metrics_active = true;
    f.grid.terrain_metrics = &metrics;
    f.grid.staggering = StaggerType::CGrid;
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                f.u(i, j, k) = static_cast<float>(2.0 + 0.05 * k);
                f.v(i, j, k) = static_cast<float>(-1.0 + 0.03 * k);
                f.w(i, j, k) = 0.0f;
                f.e(i, j, k) = static_cast<float>(0.3 + 0.01 * (i + 2 * j + k));
            }

    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    REQUIRE_NOTHROW(run_tke(f, tendencies, diagnostics));

    double activity = 0.0;
    for (int i = 1; i < kNx - 1; ++i)
        for (int j = 1; j < kNy - 1; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                REQUIRE(std::isfinite(static_cast<double>(tendencies.dtkedt_sgs(i, j, k))));
                REQUIRE(std::isfinite(static_cast<double>(diagnostics.nu_t(i, j, k))));
                REQUIRE(diagnostics.nu_t(i, j, k) > 0.0f);
                activity += std::abs(static_cast<double>(tendencies.dtkedt_sgs(i, j, k)));
            }
    REQUIRE(activity > 0.0);
}

TEST_CASE("TKE moist buoyancy uses the virtual-potential-temperature gradient",
          "[turbulence][tke][moist][buoyancy]")
{
    TKEFixture dry;
    dry.theta.fill(300.0f);
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
                dry.qv(i, j, k) = 0.001f * static_cast<float>(k);
    TKEFixture moist = dry;
    moist.state.u = &moist.u;
    moist.state.v = &moist.v;
    moist.state.w = &moist.w;
    moist.state.theta = &moist.theta;
    moist.state.rho = &moist.rho;
    moist.state.tke = &moist.e;
    moist.state.qv = &moist.qv;
    dry.cfg.enable_moist_buoyancy = false;
    dry.cfg.stability_correction = "tke_based";
    moist.cfg.enable_moist_buoyancy = true;
    moist.cfg.stability_correction = "tke_based";

    TurbulenceTendencies dry_tendencies, moist_tendencies;
    TurbulenceDiagnostics dry_diagnostics, moist_diagnostics;
    run_tke(dry, dry_tendencies, dry_diagnostics);
    run_tke(moist, moist_tendencies, moist_diagnostics);

    REQUIRE(dry_diagnostics.buoy_prod(3, 3, 4) == 0.0f);
    REQUIRE(moist_diagnostics.buoy_prod(3, 3, 4) < 0.0f);
    REQUIRE(moist_diagnostics.nu_t(3, 3, 4) <
            dry_diagnostics.nu_t(3, 3, 4));
}

TEST_CASE("TKE closure is bitwise deterministic across OpenMP worker counts",
          "[turbulence][tke][openmp][determinism]")
{
#ifndef _OPENMP
    SUCCEED("OpenMP is not enabled in this build");
#else
    constexpr int n = 34;
    Field3D u(n, n, n, 0.0f), v(n, n, n, 0.0f), w(n, n, n, 0.0f);
    Field3D theta(n, n, n, 300.0f), qv(n, n, n, 0.0f);
    Field3D density(n, n, n, 1.0f), tke(n, n, n, 0.2f);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
            {
                u(i, j, k) = static_cast<float>(0.03 * i - 0.01 * j + 0.02 * k);
                v(i, j, k) = static_cast<float>(-0.02 * i + 0.04 * j - 0.01 * k);
                w(i, j, k) = static_cast<float>(0.01 * i + 0.015 * j + 0.005 * k);
                theta(i, j, k) = static_cast<float>(300.0 + 0.002 * k);
                qv(i, j, k) = static_cast<float>(0.006 + 1.0e-5 * k);
                density(i, j, k) = static_cast<float>(1.1 - 0.002 * k);
                tke(i, j, k) = static_cast<float>(0.15 + 0.001 * (i + j + k));
            }
    std::vector<double> base_density(static_cast<std::size_t>(n), 1.0);
    GridMetrics grid;
    grid.dx = 75.0;
    grid.dy = 90.0;
    grid.dz.assign(static_cast<std::size_t>(n), 60.0);
    TurbulenceConfig config;
    config.scheme_id = "tke";
    config.enable_moist_buoyancy = true;
    config.enable_diagnostics = true;
    TurbulenceStateView state;
    state.u = &u;
    state.v = &v;
    state.w = &w;
    state.theta = &theta;
    state.qv = &qv;
    state.rho = &density;
    state.tke = &tke;
    state.rho0 = &base_density;
    state.NR = n;
    state.NTH = n;
    state.NZ = n;

    const int saved_threads = omp_get_max_threads();
    const int saved_dynamic = omp_get_dynamic();
    struct OpenMpRestore
    {
        int threads;
        int dynamic;
        ~OpenMpRestore()
        {
            omp_set_dynamic(dynamic);
            omp_set_num_threads(threads);
        }
    } restore{saved_threads, saved_dynamic};
    omp_set_dynamic(0);

    for (bool c_grid : {true, false})
    {
        INFO("c_grid=" << c_grid);
        state.c_grid = c_grid;
        grid.staggering = c_grid ? StaggerType::CGrid
                                 : StaggerType::Collocated;
        TurbulenceTendencies serial_tendencies, parallel_tendencies;
        TurbulenceDiagnostics serial_diagnostics, parallel_diagnostics;
        TKEScheme serial_scheme, parallel_scheme;
        serial_scheme.initialize(config);
        parallel_scheme.initialize(config);
        omp_set_num_threads(1);
        serial_scheme.compute(
            config, grid, state, serial_tendencies, &serial_diagnostics);
        omp_set_num_threads(4);
        parallel_scheme.compute(
            config, grid, state, parallel_tendencies, &parallel_diagnostics);

        const Field3D* serial_fields[] = {
            &serial_tendencies.dudt_sgs, &serial_tendencies.dvdt_sgs,
            &serial_tendencies.dwdt_sgs, &serial_tendencies.dthetadt_sgs,
            &serial_tendencies.dqvdt_sgs, &serial_tendencies.dtkedt_sgs,
            &serial_diagnostics.nu_t, &serial_diagnostics.K_theta,
            &serial_diagnostics.K_q, &serial_diagnostics.K_tke,
            &serial_diagnostics.shear_prod, &serial_diagnostics.buoy_prod,
            &serial_diagnostics.dissipation, &serial_diagnostics.Cs_eff,
            &serial_diagnostics.Pr_t_eff};
        const Field3D* parallel_fields[] = {
            &parallel_tendencies.dudt_sgs, &parallel_tendencies.dvdt_sgs,
            &parallel_tendencies.dwdt_sgs, &parallel_tendencies.dthetadt_sgs,
            &parallel_tendencies.dqvdt_sgs, &parallel_tendencies.dtkedt_sgs,
            &parallel_diagnostics.nu_t, &parallel_diagnostics.K_theta,
            &parallel_diagnostics.K_q, &parallel_diagnostics.K_tke,
            &parallel_diagnostics.shear_prod, &parallel_diagnostics.buoy_prod,
            &parallel_diagnostics.dissipation, &parallel_diagnostics.Cs_eff,
            &parallel_diagnostics.Pr_t_eff};
        for (std::size_t field = 0;
             field < sizeof(serial_fields) / sizeof(serial_fields[0]);
             ++field)
            REQUIRE(bitwise_equal(*serial_fields[field], *parallel_fields[field]));
    }
#endif
}

TEST_CASE("TKE retains physical top stratification and respects directional buoyancy",
          "[turbulence][tke][buoyancy][top][direction]")
{
    TKEFixture f;
    f.state.c_grid = true;
    f.cfg.stability_correction = "tke_based";
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
                f.theta(i, j, k) = 300.0f + k;
    TurbulenceTendencies tendencies;
    TurbulenceDiagnostics diagnostics;
    run_tke(f, tendencies, diagnostics);
    for (int k : {kNz - 2, kNz - 1})
    {
        CAPTURE(k);
        const double n2 = turbulence_constants::g / (300.0 + k) / 100.0;
        REQUIRE(tmv::eddy_viscosity::compute_buoyancy_frequency_squared(
            f.state, f.grid, 3, 3, k, false) == Approx(n2));
        REQUIRE(diagnostics.buoy_prod(3, 3, k) < 0.0f);
    }
    REQUIRE(diagnostics.nu_t(3, 3, kNz - 1)
        == Approx(diagnostics.nu_t(3, 3, kNz - 2)).epsilon(0.01));
    f.cfg.mode = "horizontal_only";
    run_tke(f, tendencies, diagnostics);
    for (int k = 0; k < kNz; ++k)
    {
        REQUIRE(diagnostics.buoy_prod(3, 3, k) == 0.0f);
        REQUIRE(tendencies.dthetadt_sgs(3, 3, k) == 0.0f);
    }
    for (int i = 0; i < kNx; ++i)
        for (int j = 0; j < kNy; ++j)
            for (int k = 0; k < kNz; ++k)
            {
                f.theta(i, j, k) = 300.0f - k;
                f.u(i, j, k) = static_cast<float>(k);
            }
    const double n2 = -turbulence_constants::g / 297.0 / 100.0;
    REQUIRE(tmv::eddy_viscosity::cgrid_stability_factor_ri(f.state, f.grid, 3, 3, 3)
        == Approx(1.0 - 0.5 * n2 / 0.0001));
}

TEST_CASE("flat A-grid TKE production matches physical face stress work",
          "[turbulence][tke][agrid][energy-pair]")
{
    for (const auto* mode : {"3d", "horizontal_only", "vertical_only"})
    {
        TKEFixture f;
        f.state.c_grid = false;
        f.grid.staggering = StaggerType::Collocated;
        f.state.periodic_x = f.state.periodic_y = true;
        f.cfg.mode = mode;
        constexpr double pi = 3.14159265358979323846;
        for (int i = 0; i < kNx; ++i)
            for (int j = 0; j < kNy; ++j)
                for (int k = 0; k < kNz; ++k)
                {
                    f.u(i,j,k) = static_cast<float>(std::sin(2*pi*(j-1)/(kNy-2)) + 0.1*k);
                    f.v(i,j,k) = static_cast<float>(1.5*std::sin(2*pi*(i-1)/(kNx-2)) + 0.2*k);
                    f.w(i,j,k) = static_cast<float>(0.2*std::cos(2*pi*(i-1)/(kNx-2)) + 0.1*k);
                    f.e(i,j,k) = 0.1f + 0.02f*i + 0.01f*k;
                }
        TurbulenceTendencies tendency;
        TurbulenceDiagnostics diagnostics;
        run_tke(f, tendency, diagnostics);
        const auto domain = physical_scalar_domain(kNx, kNy, kNz, StaggerType::Collocated);
        double work = 0.0, production = 0.0;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double mass = f.rho(i,j,k) * f.grid.dx * f.grid.dy * f.grid.dz[k];
                    work += mass * (static_cast<double>(f.u(i,j,k))*tendency.dudt_sgs(i,j,k)
                        + static_cast<double>(f.v(i,j,k))*tendency.dvdt_sgs(i,j,k)
                        + static_cast<double>(f.w(i,j,k))*tendency.dwdt_sgs(i,j,k));
                    production += mass * diagnostics.shear_prod(i,j,k);
                }
        CAPTURE(mode, work, production);
        REQUIRE(work < 0.0);
        REQUIRE(production > 0.0);
        REQUIRE(work + production == Approx(0.0).margin(1.0e-6 * std::abs(work)));
    }
}
