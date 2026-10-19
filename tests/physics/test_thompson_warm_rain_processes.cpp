#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/warm_rain_processes.hpp"

#include <limits>

TEST_CASE("Thompson autoconversion creates rain without a rain seed",
          "[microphysics][thompson][warm-rain][autoconversion][literal]")
{
    const auto rates = tmv::thompson::warm_rain_conversion_rates(290.0, 1.2, 1.0e-3, 0.0, 100.0e6, 0.0, 1.0);
    CHECK(rates.rain_water_kg_m3_s == Approx(9.499199570041437e-8).epsilon(1e-9));
    CHECK(rates.cloud_water_kg_m3_s == -rates.rain_water_kg_m3_s);
    CHECK(rates.rain_number_m3_s == Approx(12.09475653590801).epsilon(1e-9));
    CHECK(rates.cloud_number_m3_s == Approx(-6737.152686035734).epsilon(1e-9));
}

TEST_CASE("Thompson warm-rain conversion preserves water and both moments",
          "[microphysics][thompson][warm-rain][literal]")
{
    using namespace tmv::thompson;
    const auto rates = warm_rain_conversion_rates(
        290.0, 1.2, 5.0e-4, 2.0e-4, 100.0e6, 100.0e3, 1.0);
    REQUIRE(rates.cloud_water_kg_m3_s == Approx(-6.10831735887e-7).epsilon(2.0e-9));
    REQUIRE(rates.cloud_number_m3_s == Approx(-121958.47750834223).epsilon(2.0e-12));
    REQUIRE(rates.rain_number_m3_s == Approx(-38.612097256982).epsilon(2.0e-10));
    REQUIRE(rates.cloud_water_kg_m3_s < 0.0);
    REQUIRE(rates.rain_water_kg_m3_s > 0.0);
    REQUIRE(rates.cloud_number_m3_s < 0.0);
    REQUIRE(rates.rain_number_m3_s < 0.0);
    REQUIRE(rates.cloud_water_kg_m3_s + rates.rain_water_kg_m3_s
            == Approx(0.0).margin(1.0e-15));
    REQUIRE(5.0e-4 + rates.cloud_water_kg_m3_s >= 0.0);
    REQUIRE(2.0e-4 + rates.rain_water_kg_m3_s >= 0.0);
    REQUIRE(100.0e6 + rates.cloud_number_m3_s >= 0.0);
    REQUIRE(100.0e3 + rates.rain_number_m3_s >= 0.0);
}

TEST_CASE("Thompson rain breakup reverses the self-collection number sink",
          "[microphysics][thompson][warm-rain][literal]")
{
    const auto rates = tmv::thompson::warm_rain_conversion_rates(
        290.0, 1.2, 0.0, 1.0e-3, 0.0, 100.0, 1.0);
    REQUIRE(rates.cloud_water_kg_m3_s == 0.0);
    REQUIRE(rates.rain_water_kg_m3_s == 0.0);
    REQUIRE(rates.rain_number_m3_s
            == Approx(560.4827026425).epsilon(2.0e-12));
}

TEST_CASE("Thompson rain evaporation returns number to water-friendly aerosol",
          "[microphysics][thompson][warm-rain][literal]")
{
    using namespace tmv::thompson;
    const auto rates = rain_evaporation_rates(
        100000.0, 290.0, 1.2, 0.005, 1.0e-4, 100.0e3, 1.0);
    REQUIRE(rates.vapor_kg_m3_s
            == Approx(4.7096e-6).epsilon(2.0e-5));
    REQUIRE(rates.rain_number_m3_s
            == Approx(-4709.6292634546).epsilon(2.0e-12));
    REQUIRE(rates.temperature_k_s
            == Approx(-0.0095909887).margin(1.0e-10));
    REQUIRE(rates.vapor_kg_m3_s > 0.0);
    REQUIRE(rates.rain_water_kg_m3_s == -rates.vapor_kg_m3_s);
    REQUIRE(rates.rain_number_m3_s < 0.0);
    REQUIRE(rates.water_aerosol_m3_s == -rates.rain_number_m3_s);
    REQUIRE(rates.temperature_k_s < 0.0);
    REQUIRE(rates.vapor_kg_m3_s <= 1.0e-4);
    REQUIRE(rates.rain_number_m3_s >= -99.0e3);
}

TEST_CASE("Thompson warm-rain processes reject invalid state",
          "[microphysics][thompson][warm-rain][invalid]")
{
    using namespace tmv::thompson;
    REQUIRE_THROWS_AS(warm_rain_conversion_rates(
        290.0, 1.2, -1.0, 0.0, 0.0, 0.0, 1.0),
        std::invalid_argument);
    REQUIRE_THROWS_AS(rain_evaporation_rates(
        100000.0, 290.0, 1.2,
        std::numeric_limits<double>::quiet_NaN(),
        1.0e-4, 1.0e5, 1.0),
        std::invalid_argument);
}
