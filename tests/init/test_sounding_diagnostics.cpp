/**
 * @file test_sounding_diagnostics.cpp
 * @brief Verifies CAPE/CIN/LCL/LFC/EL/PWAT/shear computation.
 *
 * Coverage:
 *   - Convectively unstable column (warm moist surface, cooler aloft):
 *     positive CAPE, finite LFC and EL, modest CIN, sensible LCL.
 *   - Convectively suppressed column (dry, cold cap): CAPE=0, no LFC/EL.
 *   - Hodograph wiring: bulk_shear_0_6km_ms = |V(6 km) - V(sfc)|.
 */

#include "catch2/catch.hpp"
#include "init/sounding/diagnostics.hpp"

#include <cmath>
#include <vector>

namespace
{

tmv::init::Sounding make_supercell_sounding()
{
    // Hand-built supercell-like profile: warm moist surface, steep midlevel
    // lapse, dry tropopause cap. Hydrostatically self-consistent enough to
    // pass verify_self_consistent thresholds.
    constexpr int NZ = 33;
    constexpr double DZ = 500.0;

    tmv::init::Sounding s;
    s.z_m.resize(NZ);
    s.T_k.resize(NZ);
    s.theta_k.resize(NZ);
    s.qv_kgkg.resize(NZ);
    s.p_pa.resize(NZ);
    s.rho_kgm3.resize(NZ);

    // Build T(z) for an aggressive supercell environment: warm moist BL,
    // steep mid-trop lapse, dry tropopause cap. Surface T = 303 K, qv =
    // 0.016 produces ~2-3 kJ/kg CAPE — well within "supercell" range.
    auto T_at = [](double z) {
        if (z < 1000.0) return 303.0;
        if (z < 6000.0) return 303.0 - 0.0085 * (z - 1000.0);   // ~261 K at 6 km
        const double T_top_unstable = 303.0 - 0.0085 * 5000.0;
        if (z < 12000.0) return T_top_unstable - 0.005 * (z - 6000.0);
        return T_top_unstable - 0.005 * 6000.0;
    };

    auto qv_at = [](double z) {
        if (z < 2000.0) return 0.016;
        return 0.016 * std::exp(-(z - 2000.0) / 2000.0);
    };

    // Hydrostatic pressure: integrate from p0 = 1e5 with the local T.
    constexpr double R_d = 287.0;
    constexpr double cp = 1004.0;
    constexpr double g = 9.81;
    constexpr double p0 = 100000.0;

    s.z_m[0] = 0.0;
    s.T_k[0] = T_at(0.0);
    s.qv_kgkg[0] = qv_at(0.0);
    s.p_pa[0] = p0;
    s.rho_kgm3[0] = s.p_pa[0] / (R_d * s.T_k[0]);
    s.theta_k[0] = s.T_k[0];

    for (int k = 1; k < NZ; ++k)
    {
        s.z_m[k] = static_cast<double>(k) * DZ;
        s.T_k[k] = T_at(s.z_m[k]);
        s.qv_kgkg[k] = qv_at(s.z_m[k]);
        const double T_avg = 0.5 * (s.T_k[k] + s.T_k[k - 1]);
        s.p_pa[k] = s.p_pa[k - 1] * std::exp(-g * DZ / (R_d * T_avg));
        s.rho_kgm3[k] = std::max(s.p_pa[k] / (R_d * s.T_k[k]), 0.1);
        s.theta_k[k] = s.T_k[k] * std::pow(p0 / s.p_pa[k], R_d / cp);
    }
    return s;
}

}  // namespace

TEST_CASE("compute_sounding_diagnostics on supercell-like profile",
          "[init][diagnostics]")
{
    const auto s = make_supercell_sounding();
    const auto d = tmv::init::compute_sounding_diagnostics(s);

    SECTION("CAPE is in the supercell range")
    {
        // Hand-built profile delivers ~2-4 kJ/kg CAPE; tolerate a wide band.
        REQUIRE(d.cape_jkg > 1500.0);
        REQUIRE(d.cape_jkg < 6000.0);
    }

    SECTION("CIN is non-negative and reasonable")
    {
        REQUIRE(d.cin_jkg >= 0.0);
        // Synthetic profile has a deep cap before the steep mid-trop
        // lapse kicks in; CIN can hit the upper hundreds. Real profiles
        // usually fall under 500 J/kg.
        REQUIRE(d.cin_jkg < 1000.0);
    }

    SECTION("LCL is in the boundary layer")
    {
        REQUIRE(d.lcl_m > 200.0);
        REQUIRE(d.lcl_m < 2500.0);
    }

    SECTION("LFC and EL are finite and ordered")
    {
        REQUIRE(std::isfinite(d.lfc_m));
        REQUIRE(std::isfinite(d.el_m));
        REQUIRE(d.lfc_m > d.lcl_m);
        REQUIRE(d.el_m > d.lfc_m);
        REQUIRE(d.el_m > 5000.0);  // at least into the mid-troposphere
    }

    SECTION("PWAT is reasonable for moist column")
    {
        REQUIRE(d.pwat_mm > 10.0);
        REQUIRE(d.pwat_mm < 80.0);
    }
}

TEST_CASE("compute_sounding_diagnostics: convectively suppressed profile",
          "[init][diagnostics]")
{
    // Cold dry surface, isothermal column above. Surface parcel never lifts.
    constexpr int NZ = 16;
    constexpr double DZ = 500.0;

    tmv::init::Sounding s;
    s.z_m.resize(NZ);
    s.T_k.resize(NZ);
    s.theta_k.resize(NZ);
    s.qv_kgkg.resize(NZ);
    s.p_pa.resize(NZ);
    s.rho_kgm3.resize(NZ);

    constexpr double R_d = 287.0;
    constexpr double cp = 1004.0;
    constexpr double g = 9.81;
    constexpr double p0 = 100000.0;

    for (int k = 0; k < NZ; ++k)
    {
        s.z_m[k] = static_cast<double>(k) * DZ;
        s.T_k[k] = 280.0;            // cold isothermal
        s.qv_kgkg[k] = 0.001;        // dry
    }
    s.p_pa[0] = p0;
    s.rho_kgm3[0] = s.p_pa[0] / (R_d * s.T_k[0]);
    s.theta_k[0] = s.T_k[0];
    for (int k = 1; k < NZ; ++k)
    {
        s.p_pa[k] = s.p_pa[k - 1] * std::exp(-g * DZ / (R_d * s.T_k[k]));
        s.rho_kgm3[k] = std::max(s.p_pa[k] / (R_d * s.T_k[k]), 0.1);
        s.theta_k[k] = s.T_k[k] * std::pow(p0 / s.p_pa[k], R_d / cp);
    }

    const auto d = tmv::init::compute_sounding_diagnostics(s);

    REQUIRE(d.cape_jkg == Approx(0.0).margin(1.0e-9));
    REQUIRE_FALSE(std::isfinite(d.lfc_m));
    REQUIRE_FALSE(std::isfinite(d.el_m));
    // CIN must be 0 when there's no LFC (per the implementation choice that
    // the concept loses meaning otherwise).
    REQUIRE(d.cin_jkg == Approx(0.0).margin(1.0e-9));
}

TEST_CASE("compute_sounding_diagnostics: bulk shear from wind column",
          "[init][diagnostics]")
{
    auto s = make_supercell_sounding();
    tmv::init::WindColumn winds;
    winds.u_ms.resize(s.size(), 0.0);
    winds.v_ms.resize(s.size(), 0.0);
    // Linear wind profile: 0 at surface, 30 m/s eastward at 6 km.
    for (std::size_t k = 0; k < s.size(); ++k)
    {
        const double frac = std::min(s.z_m[k] / 6000.0, 1.0);
        winds.u_ms[k] = 30.0 * frac;
    }

    const auto d = tmv::init::compute_sounding_diagnostics(s, winds);
    REQUIRE(d.has_kinematic);
    // u(6km) - u(sfc) = 30 - 0 = 30, v unchanged. Shear magnitude = 30.
    REQUIRE(d.bulk_shear_0_6km_ms == Approx(30.0).margin(0.5));
}

TEST_CASE("compute_sounding_diagnostics: empty wind column = no kinematic",
          "[init][diagnostics]")
{
    const auto s = make_supercell_sounding();
    tmv::init::WindColumn winds;  // empty
    const auto d = tmv::init::compute_sounding_diagnostics(s, winds);
    REQUIRE_FALSE(d.has_kinematic);
    REQUIRE(d.bulk_shear_0_6km_ms == 0.0);
}
