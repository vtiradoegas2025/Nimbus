/**
 * @file test_parametric_cape_sounding.cpp
 * @brief Verification of ParametricCAPESoundingSource.
 *
 * Step 1 of the IC-system rebuild moves the previously-inline parametric
 * column build out of `equations.cpp::initialize()` into a free-standing
 * SoundingSource. These tests pin the contract:
 *
 *   - Default construction with the legacy defaults produces a column
 *     whose surface and tropopause values match the previously-inline
 *     formulas (regression net for the extraction).
 *   - Pressure column is monotonic and self-consistent with T (hydrostatic).
 *   - Density, theta, and qv all satisfy the SoundingSource invariants.
 *   - Invalid grids and hydrostatic boundary pressure fail closed.
 *   - CAPE scaling steepens the unstable-layer lapse rate as documented.
 */

#include "catch2/catch.hpp"
#include "core/infra/physical_constants.hpp"
#include "init/sounding/diagnostics.hpp"
#include "init/sounding/parametric_cape.hpp"

#include <cmath>
#include <vector>

namespace
{

std::vector<double> make_uniform_z(int nz, double dz)
{
    std::vector<double> z(static_cast<std::size_t>(nz));
    for (int k = 0; k < nz; ++k)
    {
        z[static_cast<std::size_t>(k)] = static_cast<double>(k) * dz;
    }
    return z;
}

}  // namespace

TEST_CASE("ParametricCAPESoundingSource default build is self-consistent",
          "[init][sounding][parametric_cape]")
{
    tmv::init::ParametricCAPEParams params;  // legacy defaults
    tmv::init::ParametricCAPESoundingSource source(params);

    constexpr int NZ = 64;
    constexpr double DZ = 250.0;
    const auto z = make_uniform_z(NZ, DZ);

    const auto sounding = source.build(z, DZ);
    REQUIRE(sounding.size() == static_cast<std::size_t>(NZ));
    REQUIRE_NOTHROW(sounding.verify_self_consistent());

    SECTION("surface T equals the surface potential temperature")
    {
        // Well-mixed boundary layer: T(0) == surface_theta_k (no offset).
        REQUIRE(sounding.T_k[0] == Approx(300.0).margin(1.0e-9));
        REQUIRE(sounding.p_pa[0] == Approx(p0).margin(1.0e-9));
    }

    SECTION("pressure column decreases monotonically with height")
    {
        for (std::size_t k = 1; k < sounding.size(); ++k)
        {
            REQUIRE(sounding.p_pa[k] < sounding.p_pa[k - 1]);
        }
    }

    SECTION("density follows EOS at every level")
    {
        for (std::size_t k = 0; k < sounding.size(); ++k)
        {
            const double rho_eos = sounding.p_pa[k] / (R_d * sounding.T_k[k]);
            REQUIRE(sounding.rho_kgm3[k] == Approx(rho_eos).epsilon(1.0e-12));
        }
    }

    SECTION("theta = T * (p0/p)^(R_d/cp) at every level")
    {
        const double kappa = R_d / cp;
        for (std::size_t k = 0; k < sounding.size(); ++k)
        {
            const double theta_expected =
                sounding.T_k[k] * std::pow(p0 / sounding.p_pa[k], kappa);
            REQUIRE(sounding.theta_k[k] == Approx(theta_expected).epsilon(1.0e-12));
        }
    }

    SECTION("qv is bounded by 0.95 * qvsat at every level")
    {
        for (std::size_t k = 0; k < sounding.size(); ++k)
        {
            const double T = sounding.T_k[k];
            const double T_c = T - physical_constants::freezing_temperature_k;
            const double e_sat =
                (T >= physical_constants::freezing_temperature_k)
                    ? 611.21 * std::exp((18.678 - T_c / 234.5) * T_c / (257.14 + T_c))
                    : 611.15 * std::exp((23.036 - T_c / 333.7) * T_c / (279.82 + T_c));
            const double qvsat = 0.622 * e_sat / std::max(sounding.p_pa[k] - e_sat, 1.0);
            REQUIRE(sounding.qv_kgkg[k] <= qvsat * params.rh_cap + 1.0e-12);
            REQUIRE(sounding.qv_kgkg[k] >= 0.0);
        }
    }
}

TEST_CASE("Parametric CAPE sounding integrates from the surface face to first center",
          "[init][sounding][parametric_cape][coordinates]")
{
    constexpr double DZ = 250.0;
    std::vector<double> z(8);
    for (std::size_t k = 0; k < z.size(); ++k)
    {
        z[k] = (static_cast<double>(k) + 0.5) * DZ;
    }

    tmv::init::ParametricCAPEParams params;
    params.cape_target_jkg = 0.0;
    params.surface_theta_k = 300.0;
    const auto s = tmv::init::ParametricCAPESoundingSource(params).build(z, DZ);

    const double dry_lapse = g / cp;
    const double first_temperature = 300.0 - dry_lapse * z[0];
    const double layer_temperature = 0.5 * (300.0 + first_temperature);
    const double expected_pressure =
        p0 * std::exp(-g * z[0] / (R_d * layer_temperature));
    REQUIRE(s.z_m[0] == Approx(0.5 * DZ));
    REQUIRE(s.T_k[0] == Approx(first_temperature).epsilon(1.0e-12));
    REQUIRE(s.p_pa[0] == Approx(expected_pressure).epsilon(1.0e-12));
    REQUIRE(s.p_pa[0] < p0);
}

TEST_CASE("Parametric CAPE sounding honors a non-reference surface pressure",
          "[init][sounding][parametric_cape][coordinates]")
{
    tmv::init::ParametricCAPEParams params;
    params.cape_target_jkg = 0.0;
    params.surface_theta_k = 300.0;
    params.surface_pressure_pa = 85000.0;

    const auto s = tmv::init::ParametricCAPESoundingSource(params).build({0.0, 250.0}, 250.0);
    const double expected_temperature = params.surface_theta_k
        * std::pow(params.surface_pressure_pa / p0, R_d / cp);

    REQUIRE(s.p_pa[0] == Approx(params.surface_pressure_pa).epsilon(1.0e-12));
    REQUIRE(s.T_k[0] == Approx(expected_temperature).epsilon(1.0e-12));
    REQUIRE(s.theta_k[0] == Approx(params.surface_theta_k).epsilon(1.0e-12));
}

TEST_CASE("ParametricCAPESoundingSource rejects inputs that would require repair",
          "[init][sounding][parametric_cape]")
{
    const auto z = make_uniform_z(64, 250.0);

    tmv::init::ParametricCAPEParams bad_theta;
    bad_theta.surface_theta_k = 200.0;
    REQUIRE_THROWS_AS(
        tmv::init::ParametricCAPESoundingSource(bad_theta).build(z, 250.0),
        std::invalid_argument);

    tmv::init::ParametricCAPEParams bad_qv;
    bad_qv.surface_qv_kgkg = -0.1;
    REQUIRE_THROWS_AS(
        tmv::init::ParametricCAPESoundingSource(bad_qv).build(z, 250.0),
        std::invalid_argument);

    tmv::init::ParametricCAPEParams bad_tropopause;
    bad_tropopause.tropopause_z_m = 5000.0;
    REQUIRE_THROWS_AS(
        tmv::init::ParametricCAPESoundingSource(bad_tropopause).build(z, 250.0),
        std::invalid_argument);
}

TEST_CASE("ParametricCAPESoundingSource CAPE scaling steepens unstable lapse",
          "[init][sounding][parametric_cape]")
{
    const auto z = make_uniform_z(64, 250.0);

    tmv::init::ParametricCAPEParams low;
    low.cape_target_jkg = 1000.0;

    tmv::init::ParametricCAPEParams high;
    high.cape_target_jkg = 5000.0;

    const auto s_low = tmv::init::ParametricCAPESoundingSource(low).build(z, 250.0);
    const auto s_high = tmv::init::ParametricCAPESoundingSource(high).build(z, 250.0);

    // Mid-unstable-layer is around z=2500m with the default tropopause, where
    // unstable_top = clamp(0.5 * 12000, 2500, 7000) = 6000. Pick z=2500m.
    const std::size_t k_mid = 10;  // 2500 m at dz=250
    REQUIRE(z[k_mid] == 2500.0);
    // Higher CAPE -> steeper lapse rate -> colder T at the same height.
    REQUIRE(s_high.T_k[k_mid] < s_low.T_k[k_mid]);
}

TEST_CASE("ParametricCAPESoundingSource describe() returns canonical name",
          "[init][sounding][parametric_cape]")
{
    tmv::init::ParametricCAPESoundingSource source({});
    REQUIRE(source.describe() == "parametric_cape");
}

TEST_CASE("ParametricCAPESoundingSource throws on invalid grid",
          "[init][sounding][parametric_cape]")
{
    tmv::init::ParametricCAPESoundingSource source({});
    REQUIRE_THROWS_AS(source.build({}, 100.0), std::invalid_argument);
    REQUIRE_THROWS_AS(source.build({0.0, 100.0}, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(source.build({0.0, 100.0}, -10.0), std::invalid_argument);

    tmv::init::ParametricCAPEParams invalid_pressure;
    invalid_pressure.surface_pressure_pa = 0.0;
    REQUIRE_THROWS_AS(
        tmv::init::ParametricCAPESoundingSource(invalid_pressure).build({0.0, 100.0}, 100.0),
        std::invalid_argument);
}

TEST_CASE("ParametricCAPESoundingSource hits the CAPE target with a conditionally-unstable column",
          "[init][sounding][parametric_cape][regression]")
{
    // The generator is closed-loop: it tunes the free-tropospheric lapse rate
    // until the realized CAPE (parcel ascent) matches the request. Pin THAT
    // contract -- a well-mixed boundary layer, a real LFC, and CAPE on target --
    // rather than brittle per-level temperatures (the previous regression pinned
    // the old open-loop column, which had a spurious isothermal mixed layer and
    // realized zero CAPE for any normal target).
    tmv::init::ParametricCAPEParams params;
    params.cape_target_jkg = 2500.0;
    params.surface_theta_k = 300.0;
    params.surface_qv_kgkg = 0.014;
    params.tropopause_z_m = 12000.0;

    tmv::init::ParametricCAPESoundingSource source(params);
    const auto z = make_uniform_z(64, 250.0);
    const auto s = source.build(z, 250.0);

    SECTION("surface T equals the surface potential temperature")
    {
        REQUIRE(s.z_m[0] == 0.0);
        REQUIRE(s.T_k[0] == Approx(300.0).margin(1.0e-9));
        REQUIRE(s.p_pa[0] == Approx(100000.0).margin(1.0e-9));
        REQUIRE(s.theta_k[0] == Approx(300.0).margin(1.0e-6));
    }

    SECTION("boundary layer is well-mixed (near-constant theta below 1000 m)")
    {
        for (std::size_t k = 0; k < s.size() && s.z_m[k] < 1000.0; ++k)
        {
            REQUIRE(s.theta_k[k] == Approx(s.theta_k[0]).margin(0.5));
        }
    }

    SECTION("realized CAPE matches the requested target with a real LFC")
    {
        const auto d = tmv::init::compute_sounding_diagnostics(s);
        REQUIRE(d.cape_jkg == Approx(2500.0).margin(100.0));
        REQUIRE(std::isfinite(d.lfc_m));   // a real level of free convection exists
        REQUIRE(d.el_m > d.lfc_m);
    }
}
