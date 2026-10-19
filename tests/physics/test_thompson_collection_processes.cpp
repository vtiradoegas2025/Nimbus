#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/collection_processes.hpp"

#include <cmath>
#include <limits>

namespace
{

std::shared_ptr<const tmv::thompson::CollectionTables> packaged_tables()
{
    tmv::thompson::CollectionTableStore store;
    std::string error;
    REQUIRE(store.publish_packaged_file(
        tmv::thompson::CollectionTableMetadata::packaged_path,
        error,
        tmv::thompson::checked_payload_bytes()));
    return store.snapshot();
}

constexpr double central_rain_number = 13357.110900762824;

}

TEST_CASE("Thompson collection indices reproduce WRF v4.7.1 lookup rules",
          "[microphysics][thompson][collection][literal]")
{
    using namespace tmv::thompson;
    const auto snow = rain_snow_lookup_index(
        1.0e-4, central_rain_number, 1.0e-4, 253.15);
    REQUIRE(snow.rain_content == 18);
    REQUIRE(snow.rain_intercept == 18);
    REQUIRE(snow.snow_content == 18);
    REQUIRE(snow.temperature == 4);

    const auto graupel = rain_graupel_lookup_index(
        1.0e-4, central_rain_number, 1.0e-4);
    REQUIRE(graupel.rain_content == 18);
    REQUIRE(graupel.rain_intercept == 18);
    REQUIRE(graupel.graupel_content == 18);
    REQUIRE(graupel.graupel_intercept == 18);
    REQUIRE(graupel.graupel_density == 0);

    const auto threshold = rain_snow_lookup_index(
        1.0e-6, 0.0, 1.0e-6, 280.0);
    REQUIRE(threshold.rain_content == 0);
    REQUIRE(threshold.rain_intercept == 36);
    REQUIRE(threshold.snow_content == 0);
    REQUIRE(threshold.temperature == 0);
}

TEST_CASE("Thompson cold collection uses the official table ordering conservatively",
          "[microphysics][thompson][collection]")
{
    using namespace tmv::thompson;
    const auto tables = packaged_tables();
    const auto snow_index = rain_snow_lookup_index(
        1.0e-4, central_rain_number, 1.0e-4, 253.15);
    const auto graupel_index = rain_graupel_lookup_index(
        1.0e-4, central_rain_number, 1.0e-4);
    const auto& snow = tables->rain_snow[flatten(snow_index)].values;
    const auto& graupel = tables->rain_graupel[flatten(graupel_index)];
    const auto rates = collection_process_rates(
        *tables, 1.0e-4, central_rain_number, 1.0e-4, 1.0e-4,
        253.15, 1.0e-3);

    REQUIRE(rates.rain_kg_m3_s == Approx(
        -(snow[3] + snow[6] + snow[1] + snow[4])
        - graupel.rain_mass_by_rain
        - graupel.rain_mass_by_graupel));
    REQUIRE(rates.snow_kg_m3_s == Approx(
        snow[3] + snow[6] - snow[0] - snow[5]));
    REQUIRE(rates.graupel_kg_m3_s == Approx(
        snow[1] + snow[4] + snow[0] + snow[5]
        + graupel.rain_mass_by_rain
        + graupel.rain_mass_by_graupel));
    REQUIRE(rates.rain_number_m3_s == Approx(
        -(snow[8] + snow[9] + snow[10] + snow[11]
          + graupel.graupel_number_by_rain
          + graupel.rain_number_by_graupel)));
    REQUIRE(rates.rain_kg_m3_s + rates.snow_kg_m3_s
                + rates.graupel_kg_m3_s
            == Approx(0.0).margin(1.0e-15));
}

TEST_CASE("Thompson warm collection melts frozen collectors and creates rain number",
          "[microphysics][thompson][collection]")
{
    using namespace tmv::thompson;
    const auto tables = packaged_tables();
    const auto graupel_rates = collection_process_rates(
        *tables, 1.0e-4, central_rain_number, 0.0, 1.0e-4,
        278.15, 1.0e-3);
    REQUIRE(graupel_rates.rain_kg_m3_s > 0.0);
    REQUIRE(graupel_rates.snow_kg_m3_s == 0.0);
    REQUIRE(graupel_rates.graupel_kg_m3_s < 0.0);
    REQUIRE(graupel_rates.rain_number_m3_s > 0.0);
    REQUIRE(graupel_rates.rain_kg_m3_s
                + graupel_rates.graupel_kg_m3_s
            == Approx(0.0).margin(1.0e-15));

    const auto snow_rates = collection_process_rates(
        *tables, 1.0e-4, central_rain_number, 1.0e-4, 0.0,
        278.15, 1.0e-3);
    REQUIRE(snow_rates.rain_kg_m3_s == Approx(-snow_rates.snow_kg_m3_s));
    REQUIRE(snow_rates.graupel_kg_m3_s == 0.0);
    REQUIRE(snow_rates.rain_number_m3_s == 0.0);
}

TEST_CASE("Thompson collection rejects invalid contracts",
          "[microphysics][thompson][collection][invalid]")
{
    using namespace tmv::thompson;
    CollectionTables empty;
    REQUIRE_THROWS_AS(collection_process_rates(
        empty, 1.0e-4, central_rain_number, 1.0e-4, 1.0e-4,
        253.15, 1.0), std::invalid_argument);
    REQUIRE_THROWS_AS(rain_snow_lookup_index(
        -1.0, central_rain_number, 1.0e-4, 253.15),
        std::invalid_argument);
    REQUIRE_THROWS_AS(rain_graupel_lookup_index(
        1.0e-4, std::numeric_limits<double>::infinity(), 1.0e-4),
        std::invalid_argument);
}
