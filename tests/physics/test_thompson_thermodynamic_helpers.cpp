#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/thermodynamic_helpers.hpp"

#include <limits>

TEST_CASE("Thompson wet-bulb helpers match WRF v4.7.1 literals",
          "[microphysics][thompson][thermodynamics][literal]")
{
    using namespace tmv::thompson;
    REQUIRE(liquid_saturation_mixing_ratio(100000.0, 300.0)
            == Approx(0.022783353054169782).epsilon(2.0e-13));
    REQUIRE(dewpoint_temperature(100000.0, 0.005)
            == Approx(276.8826589487415).epsilon(2.0e-13));
    REQUIRE(lifting_condensation_temperature(300.0, 276.8826589487415)
            == Approx(272.0981679537251).epsilon(2.0e-13));
    REQUIRE(equivalent_potential_temperature(
                100000.0, 300.0, 0.005, 272.0981679537251)
            == Approx(315.23513495789916).epsilon(2.0e-13));
    REQUIRE(wet_bulb_temperature(100000.0, 300.0, 0.005)
            == Approx(287.0861333743566).epsilon(2.0e-13));
    REQUIRE(wet_bulb_temperature(80000.0, 270.0, 0.001)
            == Approx(265.5660211847812).epsilon(2.0e-13));
    REQUIRE(wet_bulb_temperature(50000.0, 250.0, 0.0002)
            == Approx(247.9023569452871).epsilon(2.0e-13));
}

TEST_CASE("Thompson wet-bulb helpers reject invalid domains",
          "[microphysics][thompson][thermodynamics][invalid]")
{
    using namespace tmv::thompson;
    REQUIRE_THROWS_AS(wet_bulb_temperature(0.0, 300.0, 0.005),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(wet_bulb_temperature(100000.0, -1.0, 0.005),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(wet_bulb_temperature(
                          100000.0, 300.0,
                          std::numeric_limits<double>::quiet_NaN()),
                      std::invalid_argument);
}
