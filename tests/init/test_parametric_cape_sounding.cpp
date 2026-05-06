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
 *   - Surface input clamps fire when the user supplies absurd values.
 *   - CAPE scaling steepens the unstable-layer lapse rate as documented.
 */

#include "catch2/catch.hpp"
#include "core/infra/physical_constants.hpp"
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

    SECTION("surface level reflects mixed_layer_dtheta offset")
    {
        // Default: surface_theta_k=300, mixed_layer_dtheta_k=1 -> T(0)=301K.
        REQUIRE(sounding.T_k[0] == Approx(301.0).margin(1.0e-12));
        REQUIRE(sounding.p_pa[0] == Approx(p0).margin(1.0e-9));
    }

    SECTION("pressure column decreases monotonically with height")
    {
        for (std::size_t k = 1; k < sounding.size(); ++k)
        {
            REQUIRE(sounding.p_pa[k] < sounding.p_pa[k - 1]);
        }
    }

    SECTION("density follows EOS at every level (excluding rho-floor cells)")
    {
        for (std::size_t k = 0; k < sounding.size(); ++k)
        {
            const double rho_eos = sounding.p_pa[k] / (R_d * sounding.T_k[k]);
            const double rho_floor = params.rho_floor_kgm3;
            if (rho_eos >= rho_floor)
            {
                REQUIRE(sounding.rho_kgm3[k] == Approx(rho_eos).epsilon(1.0e-12));
            }
            else
            {
                REQUIRE(sounding.rho_kgm3[k] == Approx(rho_floor).margin(1.0e-12));
            }
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

TEST_CASE("ParametricCAPESoundingSource surface clamps preserve previous behavior",
          "[init][sounding][parametric_cape]")
{
    tmv::init::ParametricCAPEParams params;
    params.surface_theta_k = 200.0;     // physically absurd, must clamp to 250
    params.surface_qv_kgkg = -0.1;      // must clamp to 1e-5
    params.tropopause_z_m = 5000.0;     // must clamp to 8000

    tmv::init::ParametricCAPESoundingSource source(params);
    const auto z = make_uniform_z(64, 250.0);
    const auto s = source.build(z, 250.0);

    // Surface T = clamped_theta + mixed_layer_dtheta = 250 + 1 = 251 K.
    // (negative qv clamps to 1.0e-5; that part is exercised by the build
    //  not throwing on a self-consistent column.)
    REQUIRE(s.T_k[0] == Approx(251.0).margin(1.0e-12));
    REQUIRE(s.qv_kgkg[0] >= 0.0);
    // 251 K is well below freezing, so the saturation cap dominates and
    // produces qv well under the would-be base_moisture floor of 0.004.
    REQUIRE(s.qv_kgkg[0] < 0.004);
    // Tropopause clamp: with tropo at 8000 (clamped), unstable_top is
    // clamp(0.5 * 8000, 2500, 7000) = 4000, so the unstable layer ends
    // at 4000 m and standard lapse takes over there.
    REQUIRE_NOTHROW(s.verify_self_consistent());
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
}

TEST_CASE("ParametricCAPESoundingSource regression: default-default column values",
          "[init][sounding][parametric_cape][regression]")
{
    // The legacy defaults the previously-inline initialize() body produced
    // for cape=2500, theta=300, qv=0.014, tropopause=12000, dz=250 at NZ=64.
    // These values were captured from the inline code BEFORE the extraction
    // and pin the result so future refactors don't drift.
    tmv::init::ParametricCAPEParams params;
    params.cape_target_jkg = 2500.0;
    params.surface_theta_k = 300.0;
    params.surface_qv_kgkg = 0.014;
    params.tropopause_z_m = 12000.0;

    tmv::init::ParametricCAPESoundingSource source(params);
    const auto z = make_uniform_z(64, 250.0);
    const auto s = source.build(z, 250.0);

    SECTION("k=0 surface")
    {
        REQUIRE(s.z_m[0] == 0.0);
        REQUIRE(s.T_k[0] == Approx(301.0).margin(1.0e-12));
        REQUIRE(s.p_pa[0] == Approx(100000.0).margin(1.0e-9));
        // theta = T * (p0/p)^kappa = 301 * 1 = 301
        REQUIRE(s.theta_k[0] == Approx(301.0).margin(1.0e-9));
        // rho = p/(R_d T) = 100000 / (287 * 301) = 1.15725...
        REQUIRE(s.rho_kgm3[0] == Approx(100000.0 / (287.0 * 301.0)).epsilon(1.0e-12));
    }

    SECTION("k=4 (z=1000m) start of unstable layer (still in mixed layer at z<1000)")
    {
        // z[4] = 1000m, which is exactly mixed_layer_top. The condition
        // `z < mixed_layer_top` is false, so this falls into the unstable
        // layer with z - mixed_layer_top = 0 -> T = 301 - 0 = 301.
        REQUIRE(s.z_m[4] == 1000.0);
        REQUIRE(s.T_k[4] == Approx(301.0).margin(1.0e-9));
    }

    SECTION("k=24 (z=6000m) at unstable_top boundary")
    {
        // unstable_top = clamp(0.5*12000, 2500, 7000) = 6000.
        // T_at_unstable_top = 301 - 0.006 * (6000 - 1000) = 301 - 30 = 271 K
        // (with cape_scaling = 2500/2500 = 1.0,
        //  unstable_lapse = 0.004 + 0.002*1.0 = 0.006)
        REQUIRE(s.z_m[24] == 6000.0);
        REQUIRE(s.T_k[24] == Approx(271.0).margin(1.0e-9));
    }

    SECTION("k=48 (z=12000m) at tropopause boundary")
    {
        // T_at_tropopause = 271 - 0.005 * (12000 - 6000) = 271 - 30 = 241 K
        REQUIRE(s.z_m[48] == 12000.0);
        REQUIRE(s.T_k[48] == Approx(241.0).margin(1.0e-9));
    }

    SECTION("k=52 (z=13000m) inside tropopause cap (isothermal)")
    {
        // tropopause_depth = 1000, so [12000, 13000) is isothermal at 241K.
        REQUIRE(s.z_m[52] == 13000.0);
        REQUIRE(s.T_k[52] == Approx(241.0).margin(1.0e-9));
    }

    SECTION("k=60 (z=15000m) inside stratosphere (warming)")
    {
        // strat_base = 12000 + 1000 = 13000.
        // T = 241 + 0.002 * (15000 - 13000) = 241 + 4 = 245 K.
        REQUIRE(s.z_m[60] == 15000.0);
        REQUIRE(s.T_k[60] == Approx(245.0).margin(1.0e-9));
    }
}
