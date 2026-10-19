#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/cloud_evaporation.hpp"

#include <limits>

namespace
{

void require_close(double actual, double expected, double epsilon = 2.0e-12)
{
    REQUIRE(actual == Approx(expected).epsilon(epsilon));
}

} // namespace

TEST_CASE("Thompson cloud evaporation table reproduces WRF v4.7.1 literals",
          "[microphysics][thompson][evaporation][literal]")
{
    const auto table = tmv::thompson::generate_cloud_evaporation_table();
    std::string error;
    REQUIRE(tmv::thompson::validate_cloud_evaporation_table(table, error));
    REQUIRE(table.evaporated_number_m3.size() == 370'000);

    const auto low = tmv::thompson::cloud_evaporation_index(0, 0, 0);
    require_close(table.evaporated_mass_m3[low], 1.9530947108658868e-20);
    require_close(table.evaporated_number_m3[low], 3.730136130719185e-5);

    const auto middle = tmv::thompson::cloud_evaporation_index(49, 18, 49);
    require_close(table.evaporated_mass_m3[middle], 9.999999964233777e-5);
    require_close(table.evaporated_number_m3[middle], 52622929.451683246);

    const auto high = tmv::thompson::cloud_evaporation_index(99, 36, 99);
    require_close(table.evaporated_mass_m3[high], 0.009999653411780324);
    require_close(table.evaporated_number_m3[high], 2882253407.502489);
}

TEST_CASE("Thompson cloud evaporation uses the official discrete locate rules",
          "[microphysics][thompson][evaporation][literal]")
{
    REQUIRE(tmv::thompson::cloud_evaporation_diameter_index(0.2e-6) == 0);
    REQUIRE(tmv::thompson::cloud_evaporation_diameter_index(42.9e-6) == 41);
    REQUIRE(tmv::thompson::cloud_evaporation_diameter_index(2.0e-3) == 99);
    REQUIRE(tmv::thompson::cloud_evaporation_content_index(1.0e-6) == 0);
    REQUIRE(tmv::thompson::cloud_evaporation_content_index(9.9e-6) == 8);
    REQUIRE(tmv::thompson::cloud_evaporation_content_index(1.0e-4) == 18);
    REQUIRE(tmv::thompson::cloud_evaporation_content_index(1.0) == 36);
    REQUIRE(tmv::thompson::cloud_evaporation_number_index(1.0) == 0);
    REQUIRE(tmv::thompson::cloud_evaporation_number_index(1.0e12) == 99);
    REQUIRE(tmv::thompson::cloud_evaporation_diameter_index(std::numeric_limits<double>::max()) == 99);
    REQUIRE(tmv::thompson::cloud_evaporation_diameter_index(std::numeric_limits<double>::denorm_min()) == 0);
    REQUIRE(tmv::thompson::cloud_evaporation_number_index(std::numeric_limits<double>::max()) == 99);
    REQUIRE(tmv::thompson::cloud_evaporation_number_index(std::numeric_limits<double>::denorm_min()) == 0);
}

TEST_CASE("Thompson cloud evaporation returns droplet number to aerosol",
          "[microphysics][thompson][evaporation][conservation]")
{
    const auto table = tmv::thompson::generate_cloud_evaporation_table();
    const auto diameter = tmv::thompson::maximum_fully_evaporated_cloud_diameter(
        280.0, 90000.0, 1.0, 0.006, 0.008, 10.0);
    require_close(diameter, 4.219375478270128e-5);
    const auto exchange = tmv::thompson::return_evaporated_cloud_number(
        table, diameter, 1.0e-4, 52'622'929.45665924, 1.0);
    REQUIRE(exchange.cloud_number_kg_dry < 0.0);
    REQUIRE(exchange.water_aerosol_kg_dry == -exchange.cloud_number_kg_dry);
    REQUIRE(-exchange.cloud_number_kg_dry
            <= 0.99 * 52'622'929.45665924);
}

TEST_CASE("Thompson cloud evaporation rejects invalid state",
          "[microphysics][thompson][evaporation][invalid]")
{
    auto table = tmv::thompson::generate_cloud_evaporation_table();
    table.evaporated_number_m3.back() =
        std::numeric_limits<double>::quiet_NaN();
    std::string error;
    REQUIRE_FALSE(tmv::thompson::validate_cloud_evaporation_table(table, error));
    REQUIRE_THROWS_AS(tmv::thompson::cloud_evaporation_diameter_index(0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        tmv::thompson::maximum_fully_evaporated_cloud_diameter(
            280.0, 90000.0, 1.0, 0.006, 0.0, 10.0),
        std::invalid_argument);
}
