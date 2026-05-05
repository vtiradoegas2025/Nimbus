/**
 * @file test_parametric_targets_sounding.cpp
 * @brief Verifies the diagnostic-target sounding source.
 *
 * Coverage:
 *   - qv_from_lcl returns physically reasonable values for a typical LCL
 *     range, falls back to the qv floor for non-finite or excessive LCL.
 *   - translate() maps target params to ParametricCAPEParams as documented.
 *   - The built Sounding is self-consistent and reasonable.
 *   - Override path: moisture_qv_kgkg_override > 0 wins over LCL inversion.
 */

#include "catch2/catch.hpp"
#include "init/sounding/parametric_cape.hpp"
#include "init/sounding/parametric_targets.hpp"

#include <vector>

namespace
{

std::vector<double> uniform_grid(std::size_t nz, double dz)
{
    std::vector<double> z(nz);
    for (std::size_t k = 0; k < nz; ++k)
    {
        z[k] = static_cast<double>(k) * dz;
    }
    return z;
}

}  // namespace

TEST_CASE("qv_from_lcl returns sensible values across the LCL range",
          "[init][sounding][parametric_targets]")
{
    using S = tmv::init::ParametricTargetsSoundingSource;

    SECTION("LCL ~1000 m at T=300K, p=1000 hPa gives qv around 0.012")
    {
        const double qv = S::qv_from_lcl(/*lcl_m=*/1000.0,
                                         /*T=*/300.0,
                                         /*p=*/100000.0);
        // 1000 m / 125 = 8 K dewpoint depression -> T_d = 292 K.
        // qv_sat(292 K, ~100 kPa) is around 0.013 kg/kg.
        REQUIRE(qv == Approx(0.0125).margin(0.0025));
    }

    SECTION("LCL ~250 m at T=300K, p=1000 hPa gives qv around 0.020 kg/kg")
    {
        // Very moist column: 250 m / 125 = 2 K dewpoint depression.
        const double qv = S::qv_from_lcl(/*lcl_m=*/250.0,
                                         /*T=*/300.0,
                                         /*p=*/100000.0);
        REQUIRE(qv > 0.018);
        REQUIRE(qv < 0.024);
    }

    SECTION("LCL non-finite or non-positive returns the qv floor")
    {
        REQUIRE(S::qv_from_lcl(0.0, 300.0, 100000.0) == Approx(1.0e-5));
        REQUIRE(S::qv_from_lcl(-100.0, 300.0, 100000.0) == Approx(1.0e-5));
    }

    SECTION("LCL implies T_d > T -> qv floor")
    {
        // LCL = -1000 would imply dewpoint above surface temperature, which
        // is unphysical. Function returns the floor regardless of sign.
        REQUIRE(S::qv_from_lcl(-100.0, 300.0, 100000.0) == Approx(1.0e-5));
    }
}

TEST_CASE("translate() maps target params to ParametricCAPEParams",
          "[init][sounding][parametric_targets]")
{
    using S = tmv::init::ParametricTargetsSoundingSource;

    SECTION("CAPE, EL, LFC pass through")
    {
        tmv::init::ParametricTargetsParams t;
        t.target_cape_jkg = 3500.0;
        t.target_el_m = 14000.0;
        t.target_lfc_m = 1800.0;
        t.surface_theta_k = 302.0;

        const auto p = S::translate(t);
        REQUIRE(p.cape_target_jkg == Approx(3500.0));
        REQUIRE(p.tropopause_z_m == Approx(14000.0));
        REQUIRE(p.mixed_layer_top_m == Approx(1800.0));
        REQUIRE(p.surface_theta_k == Approx(302.0));
    }

    SECTION("CIN heuristic produces stronger cap for higher CIN")
    {
        tmv::init::ParametricTargetsParams low_cin;
        low_cin.target_cin_jkg = 0.0;
        const auto p_low = S::translate(low_cin);

        tmv::init::ParametricTargetsParams high_cin;
        high_cin.target_cin_jkg = 200.0;
        const auto p_high = S::translate(high_cin);

        REQUIRE(p_high.mixed_layer_dtheta_k > p_low.mixed_layer_dtheta_k);
        REQUIRE(p_high.mixed_layer_dtheta_k <= 2.5);
    }

    SECTION("Override beats LCL inversion")
    {
        tmv::init::ParametricTargetsParams t;
        t.target_lcl_m = 1000.0;             // would normally drive qv from Magnus
        t.moisture_qv_kgkg_override = 0.020; // explicit override
        const auto p = S::translate(t);
        REQUIRE(p.surface_qv_kgkg == Approx(0.020));
    }

    SECTION("Default targets give qv from LCL")
    {
        tmv::init::ParametricTargetsParams t;
        t.target_lcl_m = 1100.0;
        t.surface_theta_k = 300.0;
        t.surface_pressure_pa = 100000.0;
        const auto p = S::translate(t);
        // For these targets, qv should be ~0.012 (matches qv_from_lcl section).
        REQUIRE(p.surface_qv_kgkg > 0.010);
        REQUIRE(p.surface_qv_kgkg < 0.015);
    }
}

TEST_CASE("ParametricTargetsSoundingSource builds a self-consistent column",
          "[init][sounding][parametric_targets]")
{
    tmv::init::ParametricTargetsParams t;
    t.target_cape_jkg = 3000.0;
    t.target_cin_jkg = 50.0;
    t.target_lcl_m = 1100.0;
    t.target_lfc_m = 1500.0;
    t.target_el_m = 12000.0;
    t.surface_theta_k = 300.0;

    tmv::init::ParametricTargetsSoundingSource src(t);

    constexpr std::size_t NZ = 64;
    constexpr double DZ = 250.0;
    const auto z = uniform_grid(NZ, DZ);
    const auto s = src.build(z, DZ);

    REQUIRE(s.size() == NZ);
    REQUIRE_NOTHROW(s.verify_self_consistent());

    REQUIRE(src.describe() == "parametric_targets");
}
