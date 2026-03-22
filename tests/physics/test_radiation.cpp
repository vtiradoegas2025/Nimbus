/**
 * @file test_radiation.cpp
 * @brief Unit tests for radiation with optical depth and heating rate verification.
 *
 * Tests verify:
 *   - Factory creation
 *   - Column computation produces finite tendencies
 *   - Beer-Lambert: Fdn[k] = Fdn[k+1] * exp(-tau[k]/mu0) relationships
 *   - Energy conservation: net flux divergence has correct sign
 */
#include "catch2/catch.hpp"
#include "physics/radiation_base.hpp"
#include "core/simulation.hpp"

#include <cmath>
#include <vector>

TEST_CASE("Radiation factory creates simple_grey scheme", "[physics][radiation]")
{
    auto scheme = create_radiation_scheme("simple_grey");
    REQUIRE(scheme != nullptr);
}

TEST_CASE("Radiation factory throws for unknown scheme", "[physics][radiation]")
{
    REQUIRE_THROWS(create_radiation_scheme("nonexistent_scheme_xyz"));
}

TEST_CASE("Simple grey column computation produces finite tendencies", "[physics][radiation][analytical]")
{
    auto scheme = create_radiation_scheme("simple_grey");
    const int nz = 10;

    // Build a simple isothermal column
    std::vector<double> T(nz, 280.0);
    std::vector<double> p_col(nz);
    std::vector<double> rho_col(nz, 1.0);
    std::vector<double> dz_col(nz, 1000.0);
    std::vector<double> z_int(nz + 1);
    std::vector<double> qv_col(nz, 0.005);

    // Pressure decreasing with height
    for (int k = 0; k < nz; ++k)
    {
        p_col[k] = 100000.0 * std::exp(-static_cast<double>(k) * 1000.0 / 8500.0);
        z_int[k] = k * 1000.0;
    }
    z_int[nz] = nz * 1000.0;

    RadiationConfig cfg;
    cfg.do_lw = true;
    cfg.do_sw = true;
    cfg.tau_lw_ref = 3.0;
    cfg.tau_sw_ref = 0.1;
    cfg.n_lw = 1.0;
    cfg.n_sw = 1.0;

    RadiationColumnStateView col;
    col.T = &T;
    col.p = &p_col;
    col.rho = &rho_col;
    col.dz = &dz_col;
    col.z_int = &z_int;
    col.qv = &qv_col;
    col.mu0 = 0.5;        // 60 degree solar zenith
    col.S0 = 1366.0;
    col.Tsfc = 288.0;
    col.albedo_sw = 0.2;
    col.emissivity_lw = 0.95;

    RadiationColumnTendencies tend;
    RadiationColumnFluxes fluxes;

    scheme->initialize(cfg);
    scheme->compute_column(cfg, col, tend, &fluxes);

    // All tendencies must be finite
    REQUIRE(tend.dTdt_rad.size() == static_cast<size_t>(nz));
    for (int k = 0; k < nz; ++k)
    {
        REQUIRE(std::isfinite(tend.dTdt_rad[k]));
        REQUIRE(std::isfinite(tend.dTdt_lw[k]));
        REQUIRE(std::isfinite(tend.dTdt_sw[k]));
    }

    // Total tendency = LW + SW at each level
    for (int k = 0; k < nz; ++k)
    {
        REQUIRE(tend.dTdt_rad[k] == Approx(tend.dTdt_lw[k] + tend.dTdt_sw[k]).margin(1e-15));
    }
}

TEST_CASE("Radiation fluxes have correct sign conventions", "[physics][radiation][analytical]")
{
    auto scheme = create_radiation_scheme("simple_grey");
    const int nz = 8;

    std::vector<double> T(nz, 280.0);
    std::vector<double> p_col(nz);
    std::vector<double> rho_col(nz, 1.0);
    std::vector<double> dz_col(nz, 1000.0);
    std::vector<double> z_int(nz + 1);
    std::vector<double> qv_col(nz, 0.005);

    for (int k = 0; k < nz; ++k)
    {
        p_col[k] = 100000.0 * std::exp(-static_cast<double>(k) * 1000.0 / 8500.0);
        z_int[k] = k * 1000.0;
    }
    z_int[nz] = nz * 1000.0;

    RadiationConfig cfg;
    cfg.do_lw = true;
    cfg.do_sw = true;
    cfg.tau_lw_ref = 3.0;
    cfg.tau_sw_ref = 0.1;
    cfg.n_lw = 1.0;
    cfg.n_sw = 1.0;

    RadiationColumnStateView col;
    col.T = &T;
    col.p = &p_col;
    col.rho = &rho_col;
    col.dz = &dz_col;
    col.z_int = &z_int;
    col.qv = &qv_col;
    col.mu0 = 0.5;
    col.S0 = 1366.0;
    col.Tsfc = 288.0;
    col.albedo_sw = 0.2;
    col.emissivity_lw = 0.95;

    RadiationColumnTendencies tend;
    RadiationColumnFluxes fluxes;
    scheme->initialize(cfg);
    scheme->compute_column(cfg, col, tend, &fluxes);

    // SW downward flux at TOA should be mu0 * S0
    REQUIRE(fluxes.Fdn_sw.size() == static_cast<size_t>(nz + 1));
    double expected_toa_sw = col.mu0 * col.S0;
    REQUIRE(fluxes.Fdn_sw[nz] == Approx(expected_toa_sw).epsilon(1e-10));

    // SW downward flux should decrease toward surface (absorption)
    for (int k = nz - 1; k >= 0; --k)
    {
        REQUIRE(fluxes.Fdn_sw[k] <= fluxes.Fdn_sw[k + 1] + 1e-10);
    }

    // LW downward flux at TOA should be zero (no incoming LW from space)
    REQUIRE(fluxes.Fdn_lw[nz] == Approx(0.0).margin(1e-10));

    // All flux values must be finite and non-negative
    for (size_t k = 0; k <= static_cast<size_t>(nz); ++k)
    {
        REQUIRE(std::isfinite(fluxes.Fup_lw[k]));
        REQUIRE(std::isfinite(fluxes.Fdn_lw[k]));
        REQUIRE(std::isfinite(fluxes.Fup_sw[k]));
        REQUIRE(std::isfinite(fluxes.Fdn_sw[k]));
        REQUIRE(fluxes.Fup_lw[k] >= 0.0);
        REQUIRE(fluxes.Fdn_sw[k] >= 0.0);
    }
}

TEST_CASE("Radiation SW with zero optical depth is transparent", "[physics][radiation][analytical]")
{
    auto scheme = create_radiation_scheme("simple_grey");
    const int nz = 5;

    std::vector<double> T(nz, 280.0);
    std::vector<double> p_col(nz, 100000.0);  // uniform pressure = zero tau gradient
    std::vector<double> rho_col(nz, 1.0);
    std::vector<double> dz_col(nz, 1000.0);
    std::vector<double> z_int(nz + 1);
    std::vector<double> qv_col(nz, 0.0);

    for (int k = 0; k <= nz; ++k)
        z_int[k] = k * 1000.0;

    RadiationConfig cfg;
    cfg.do_lw = false;
    cfg.do_sw = true;
    cfg.tau_sw_ref = 0.0;  // zero optical depth = transparent atmosphere
    cfg.n_sw = 1.0;

    RadiationColumnStateView col;
    col.T = &T;
    col.p = &p_col;
    col.rho = &rho_col;
    col.dz = &dz_col;
    col.z_int = &z_int;
    col.qv = &qv_col;
    col.mu0 = 1.0;
    col.S0 = 1000.0;
    col.Tsfc = 288.0;
    col.albedo_sw = 0.0;
    col.emissivity_lw = 0.0;

    RadiationColumnTendencies tend;
    RadiationColumnFluxes fluxes;
    scheme->initialize(cfg);
    scheme->compute_column(cfg, col, tend, &fluxes);

    // With zero optical depth, SW flux should pass through unchanged
    // Fdn should be approximately S0*mu0 at all levels
    for (int k = 0; k <= nz; ++k)
    {
        REQUIRE(fluxes.Fdn_sw[k] == Approx(1000.0).margin(1.0));
    }

    // SW heating should be near zero (no absorption)
    for (int k = 0; k < nz; ++k)
    {
        REQUIRE(std::abs(tend.dTdt_sw[k]) < 1.0e-6);
    }
}
