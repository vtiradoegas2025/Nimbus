#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/activation_table.hpp"
#include "microphysics/schemes/thompson/nucleation.hpp"

#include <cmath>
#include <limits>

namespace
{

tmv::thompson::ActivationTable activation_table()
{
    tmv::thompson::ActivationTable table;
    std::string error;
    REQUIRE(tmv::thompson::import_wrf_activation_table(
        "tests/fixtures/reference/thompson/CCN_ACTIVATE.BIN", table, error));
    return table;
}

} // namespace

TEST_CASE("Thompson activation transfers water-friendly aerosol number exactly",
          "[microphysics][thompson][activation][conservation]")
{
    const auto table = activation_table();
    const auto exchange = tmv::thompson::activate_cloud_droplets(
        table, 273.15, 1.0, 1.0, 10.0e6, 300.0e6, 1e-3);
    REQUIRE(exchange.hydrometeor_kg_dry
            == Approx(219921189.4560381).epsilon(2.0e-13));
    REQUIRE(exchange.aerosol_kg_dry == -exchange.hydrometeor_kg_dry);

    const auto donor_limited = tmv::thompson::activate_cloud_droplets(
        table, 273.15, 1.0, 1.0, 0.0, 1.0, 1e-3);
    REQUIRE(donor_limited.hydrometeor_kg_dry == 1.0);
    REQUIRE(donor_limited.aerosol_kg_dry == -1.0);

    const auto returned =
        tmv::thompson::return_evaporated_cloud_droplets(1234.0);
    REQUIRE(returned.hydrometeor_kg_dry == -1234.0);
    REQUIRE(returned.aerosol_kg_dry == 1234.0);
}

TEST_CASE("Thompson DeMott ice nucleation matches WRF v4.7.1 literals",
          "[microphysics][thompson][nucleation][literal]")
{
    REQUIRE(tmv::thompson::demott_ice_number_m3(-20.0, 1.0, 1.5e6)
            == Approx(1404.6890248759178).epsilon(2.0e-13));
    REQUIRE(tmv::thompson::demott_ice_number_m3(-10.0, 0.8, 0.5e6)
            == Approx(74.25458227180071).epsilon(2.0e-13));
    REQUIRE(tmv::thompson::demott_ice_number_m3(-30.0, 1.2, 5.0e6)
            == Approx(17454.38356644733).epsilon(2.0e-13));
}

TEST_CASE("Thompson activation respects liquid support before spending aerosol",
          "[microphysics][thompson][activation][conservation]")
{
    const auto       table   = activation_table();
    constexpr double content = 1e-9;
    const double supported   = content * std::pow(19.0 / 1e-6, 3) / (16.0 * 17.0 * 18.0 * 3.1415926536 * 1000.0 / 6.0);
    for (double density : {.5, 1.0, 1.5})
    {
        const auto exchange = tmv::thompson::activate_cloud_droplets(table, 273.15, 1.0, density, 0, 300e6, content);
        CHECK(exchange.hydrometeor_kg_dry == Approx(supported / density).epsilon(1e-12));
        CHECK(exchange.aerosol_kg_dry == -exchange.hydrometeor_kg_dry);
        for (double mass : {0.0, content})
        {
            const auto exhausted =
                tmv::thompson::activate_cloud_droplets(table, 273.15, 1.0, density, 1e8, 300e6, mass);
            CHECK(exhausted.hydrometeor_kg_dry == 0);
            CHECK(exhausted.aerosol_kg_dry == 0);
        }
    }
    for (double invalid : {-content, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        CHECK_THROWS_AS(tmv::thompson::activate_cloud_droplets(table, 273.15, 1, 1, 0, 300e6, invalid),
                        std::invalid_argument);
    }
}

TEST_CASE("Thompson Koop homogeneous freezing matches WRF v4.7.1 literals",
          "[microphysics][thompson][nucleation][literal]")
{
    constexpr double water_activity = 0.6948868438693884;
    REQUIRE(tmv::thompson::koop_frozen_aerosol_number_m3(235.0, water_activity + 0.3, 1.0, 300.0e6, 10.0) ==
            Approx(78.16814995287302).epsilon(2.0e-12));
    REQUIRE(tmv::thompson::koop_frozen_aerosol_number_m3(235.0, 1.5, 1.0, 300.0e6, 10.0) == Approx(1000.0e3));
}

TEST_CASE("Thompson nucleation inputs fail closed",
          "[microphysics][thompson][nucleation][invalid]")
{
    REQUIRE_THROWS_AS(tmv::thompson::demott_ice_number_m3(
                          1.0, 1.0, 1.0e6),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(tmv::thompson::koop_frozen_aerosol_number_m3(
                          235.0,
                          std::numeric_limits<double>::quiet_NaN(),
                          1.0,
                          1.0e6,
                          1.0),
                      std::invalid_argument);
}

TEST_CASE("Thompson aerosol ice caller applies WRF gates and donor transfers",
          "[microphysics][thompson][nucleation][literal][conservation]")
{
    const auto demott = tmv::thompson::nucleate_aerosol_ice(
        250.0, 1.1e-3, 1.0e-3, 0.8e-3, 1.0,
        0.0, 0.0, 300.0e6, 1.5e6, 10.0);
    REQUIRE(demott.ice_number_kg_dry
            == Approx(2415.513041486495).epsilon(2.0e-13));
    REQUIRE(demott.ice_mass_kg_dry
            == Approx(2.415513041486495e-9).epsilon(2.0e-13));
    REQUIRE(demott.vapor_mass_kg_dry == -demott.ice_mass_kg_dry);
    REQUIRE(demott.ice_aerosol_kg_dry
            == -demott.ice_number_kg_dry);
    REQUIRE(demott.water_aerosol_kg_dry == 0.0);

    const auto koop = tmv::thompson::nucleate_aerosol_ice(
        235.0, 1.2e-3, 1.0e-3, 0.7e-3, 1.0,
        100.0e3, 0.0, 300.0e6, 1.5e6, 10.0);
    REQUIRE(koop.ice_number_kg_dry == Approx(1.0e6));
    REQUIRE(koop.ice_mass_kg_dry == Approx(1.0e-7));
    REQUIRE(koop.water_aerosol_kg_dry == Approx(-1.0e6));
    REQUIRE(koop.ice_aerosol_kg_dry == 0.0);

    const auto closed = tmv::thompson::nucleate_aerosol_ice(
        260.0, 0.9e-3, 1.0e-3, 0.85e-3, 1.0,
        0.0, 0.0, 300.0e6, 1.5e6, 10.0);
    REQUIRE(closed.ice_mass_kg_dry == 0.0);
    REQUIRE(closed.ice_number_kg_dry == 0.0);

    const auto limited = tmv::thompson::nucleate_aerosol_ice(
        235.0, 1.0e-3, 1.0e-3, 0.7e-3, 1.0,
        0.0, 0.0, 300.0e6, 1.0e12, 10.0);
    REQUIRE(limited.ice_mass_kg_dry
            == Approx((1.0e-3 - 0.7e-3) * 0.999));
    REQUIRE(limited.vapor_mass_kg_dry == -limited.ice_mass_kg_dry);
}

TEST_CASE("Thompson DeMott admission cannot nucleate absent aerosol", "[physics][thompson][nucleation][donor]")
{
    for (double density : {.3, 1.2})
        for (double aerosol : {0.0, 1e-12, 1.0, 500.0})
        {
            const auto exchange =
                tmv::thompson::nucleate_aerosol_ice(245, .002, .0015, .001, density, 0, 0, 0, aerosol, .2);
            CHECK(exchange.ice_number_kg_dry == Approx(aerosol));
            CHECK(exchange.ice_aerosol_kg_dry == -exchange.ice_number_kg_dry);
            CHECK(exchange.water_aerosol_kg_dry == 0);
            CHECK(exchange.ice_mass_kg_dry == -exchange.vapor_mass_kg_dry);
            CHECK(aerosol + exchange.ice_aerosol_kg_dry >= 0);
        }
}
