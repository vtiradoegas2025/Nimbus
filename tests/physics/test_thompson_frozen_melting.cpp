#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/frozen_melting.hpp"

#include <limits>

TEST_CASE("Thompson snow and graupel melting is conservative and bounded",
          "[microphysics][thompson][melting][literal]")
{
    const auto rates = tmv::thompson::frozen_melting_rates(
        90000.0, 278.15, 1.0, 0.004, 2.0e-4, 3.0e-4, 1.0);
    REQUIRE(rates.rain_kg_m3_s
            == Approx(6.4973e-6).epsilon(2.0e-5));
    REQUIRE(rates.rain_number_m3_s
            == Approx(49.2484334371).epsilon(2.0e-12));
    REQUIRE(rates.temperature_k_s
            == Approx(-0.0021538115).margin(1.0e-10));
    REQUIRE(rates.rain_kg_m3_s > 0.0);
    REQUIRE(rates.snow_kg_m3_s < 0.0);
    REQUIRE(rates.graupel_kg_m3_s < 0.0);
    REQUIRE(rates.rain_number_m3_s > 0.0);
    REQUIRE(rates.temperature_k_s < 0.0);
    REQUIRE(rates.rain_kg_m3_s + rates.snow_kg_m3_s
                + rates.graupel_kg_m3_s
            == Approx(0.0).margin(1.0e-15));
    REQUIRE(rates.snow_kg_m3_s >= -2.0e-4);
    REQUIRE(rates.graupel_kg_m3_s >= -3.0e-4);
}

TEST_CASE("Thompson frozen melting is neutral below freezing and rejects invalid state",
          "[microphysics][thompson][melting][invalid]")
{
    using tmv::thompson::frozen_melting_rates;
    const auto frozen = frozen_melting_rates(
        90000.0, 270.0, 1.0, 0.004, 2.0e-4, 3.0e-4, 1.0);
    REQUIRE(frozen.rain_kg_m3_s == 0.0);
    REQUIRE(frozen.rain_number_m3_s == 0.0);
    REQUIRE_THROWS_AS(frozen_melting_rates(
        90000.0, 278.15, 1.0,
        std::numeric_limits<double>::quiet_NaN(),
        2.0e-4, 3.0e-4, 1.0), std::invalid_argument);
}
