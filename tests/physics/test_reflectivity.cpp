/**
 * @file test_reflectivity.cpp
 * @brief Single-moment radar reflectivity Z(q) tests (science audit).
 *
 * Pins the per-species Marshall-Palmer reflectivity law that replaced the old
 * q^2 * (rho_x/1000)^2 heuristic (the rain Rayleigh coefficient mis-applied to
 * every species with the wrong exponent, which buried frozen species ~15-20
 * dBZ low - a trap for the NWS-dBZ rendering goal).
 *
 * Z[mm^6/m^3] = A_x * (rho_d*q)^1.75, with A_x derived from the
 * gamma-integrated exponential PSD (LFO83 N0/rho_x) and the Smith-1984 ice
 * dielectric factor on frozen species. Reference dBZ at q = 1 g/kg,
 * rho_d = 1.2: rain 44.5, snow 58.0, graupel 54.0, hail 55.2.
 */

#include "catch2/catch.hpp"
#include "radar/schemes/reflectivity/reflectivity.hpp"
#include "radar/radar_base.hpp"
#include "core/field/field3d.hpp"

#include <cmath>
#include <string>

namespace
{

constexpr int kN = 4;

// Run the single-moment (fast_da, no number concentration) reflectivity for a
// uniform field of one species at mixing ratio q, and return the interior-cell
// dBZ (out.Z_dBZ, which for a single species equals that species' value).
double reflectivity_dbz(const std::string& species, double q)
{
    Field3D qf(kN, kN, kN, static_cast<float>(q));
    Field3D rhof(kN, kN, kN, 1.2f);

    RadarConfig cfg;
    cfg.operator_tier = "fast_da";
    cfg.has_qr = cfg.has_qs = cfg.has_qg = cfg.has_qh = cfg.has_qi = false;

    RadarStateView st;
    st.NR = kN; st.NTH = kN; st.NZ = kN;
    st.rho = &rhof;
    if (species == "rain")         { cfg.has_qr = true; st.qr = &qf; }
    else if (species == "snow")    { cfg.has_qs = true; st.qs = &qf; }
    else if (species == "graupel") { cfg.has_qg = true; st.qg = &qf; }
    else if (species == "hail")    { cfg.has_qh = true; st.qh = &qf; }

    ReflectivityScheme scheme;
    scheme.initialize(cfg, kN, kN, kN);
    RadarOut out;
    scheme.compute(cfg, st, out);
    return static_cast<double>(out.Z_dBZ(kN / 2, kN / 2, kN / 2));
}

// As reflectivity_dbz, but supplies a caller-selected dry-air density.
double reflectivity_dbz_rho(const std::string& species, double q, double rho_dry)
{
    Field3D qf(kN, kN, kN, static_cast<float>(q));
    Field3D rhof(kN, kN, kN, static_cast<float>(rho_dry));

    RadarConfig cfg;
    cfg.operator_tier = "fast_da";
    cfg.has_qr = cfg.has_qs = cfg.has_qg = cfg.has_qh = cfg.has_qi = false;

    RadarStateView st;
    st.NR = kN; st.NTH = kN; st.NZ = kN;
    st.rho = &rhof;
    if (species == "rain")         { cfg.has_qr = true; st.qr = &qf; }
    else if (species == "snow")    { cfg.has_qs = true; st.qs = &qf; }
    else if (species == "graupel") { cfg.has_qg = true; st.qg = &qf; }
    else if (species == "hail")    { cfg.has_qh = true; st.qh = &qf; }

    ReflectivityScheme scheme;
    scheme.initialize(cfg, kN, kN, kN);
    RadarOut out;
    scheme.compute(cfg, st, out);
    return static_cast<double>(out.Z_dBZ(kN / 2, kN / 2, kN / 2));
}

}  // namespace

TEST_CASE("Reflectivity matches the per-species Marshall-Palmer dBZ at 1 g/kg",
          "[radar][reflectivity]")
{
    // Reference dBZ from the gamma-integrated exponential PSD with the
    // Smith-1984 ice dielectric factor (rho_d=1.2). A regression to the old
    // q^2 heuristic (rain coefficient on all species) would read rain ~35.6
    // and snow ~15.6 dBZ - far below these.
    REQUIRE(reflectivity_dbz("rain", 1.0e-3)    == Approx(44.49).margin(0.3));
    REQUIRE(reflectivity_dbz("snow", 1.0e-3)    == Approx(57.95).margin(0.3));
    REQUIRE(reflectivity_dbz("graupel", 1.0e-3) == Approx(53.98).margin(0.3));
    REQUIRE(reflectivity_dbz("hail", 1.0e-3)    == Approx(55.21).margin(0.3));

    // Light rain is well above the -30 dBZ no-echo floor (not clamped low).
    REQUIRE(reflectivity_dbz("rain", 1.0e-4) == Approx(26.99).margin(0.3));
}

TEST_CASE("Reflectivity scales as q^1.75, not q^2",
          "[radar][reflectivity]")
{
    // Doubling the mixing ratio raises dBZ by 10*log10(2^1.75) = 5.27 dB for
    // the gamma-integrated power law - NOT the 10*log10(2^2) = 6.02 dB the old
    // q^2 heuristic produced. This pins the exponent.
    const double d1 = reflectivity_dbz("rain", 1.0e-3);
    const double d2 = reflectivity_dbz("rain", 2.0e-3);
    REQUIRE(d2 - d1 == Approx(10.0 * 1.75 * std::log10(2.0)).margin(0.05));  // 5.27 dB
}

TEST_CASE("Reflectivity floors finite sub-sensitivity echoes",
          "[radar][reflectivity][range]")
{
    REQUIRE(reflectivity_dbz("rain", 1.0e-12) == -30.0);
}

TEST_CASE("Reflectivity uses local dry-air density (mass = rho_d*q), not a fixed 1.2",
          "[radar][reflectivity][density][SCI-024]")
{
    // Supplying the representative 1.2 kg/m^3 reproduces the fixed-density
    // reference, so populating rho is backward-compatible with the prior runs.
    REQUIRE(reflectivity_dbz_rho("rain", 1.0e-3, 1.2) == Approx(44.49).margin(0.3));

    // Aloft the air is thinner: the SAME mixing ratio at rho = 0.4 carries 1/3
    // the mass concentration, so Z drops by 10*1.75*log10(1.2/0.4) = 8.35 dB.
    // The old fixed 1.2 kg/m^3 ignored this and biased high-altitude dBZ high.
    const double dbz_dense = reflectivity_dbz_rho("rain", 1.0e-3, 1.2);
    const double dbz_thin  = reflectivity_dbz_rho("rain", 1.0e-3, 0.4);
    REQUIRE(dbz_dense - dbz_thin ==
            Approx(10.0 * 1.75 * std::log10(1.2 / 0.4)).margin(0.05));  // 8.35 dB

    // The reference helper explicitly supplies 1.2 kg/m3.
    REQUIRE(reflectivity_dbz("rain", 1.0e-3) ==
            Approx(reflectivity_dbz_rho("rain", 1.0e-3, 1.2)).margin(0.05));
}

TEST_CASE("Reflectivity rejects missing density without modifying output",
          "[radar][reflectivity][strict]")
{
    Field3D qr(kN, kN, kN, 1.0e-3f);
    RadarStateView state;
    state.NR = kN;
    state.NTH = kN;
    state.NZ = kN;
    state.qr = &qr;
    RadarConfig config;
    config.operator_tier = "fast_da";
    ReflectivityScheme scheme;
    scheme.initialize(config, kN, kN, kN);
    RadarOut output;
    output.Z_dBZ.resize(kN, kN, kN, 47.0f);
    REQUIRE_THROWS_AS(
        scheme.compute(config, state, output), std::invalid_argument);
    REQUIRE(output.Z_dBZ(1, 1, 1) == 47.0f);
}
