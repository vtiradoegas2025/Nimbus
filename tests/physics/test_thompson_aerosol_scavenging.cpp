#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/aerosol_scavenging.hpp"

#include <limits>

TEST_CASE("Thompson aerosol collision efficiencies match WRF v4.7.1 literals",
          "[microphysics][thompson][aerosol][literal]")
{
    using tmv::thompson::AerosolCollector;
    REQUIRE(tmv::thompson::aerosol_collection_efficiency(
                1.0e-3, 0.04e-6, 1.8e-5, 1.0, 280.0,
                AerosolCollector::Rain)
            == Approx(0.0008896804230562125).epsilon(2.0e-12));
    REQUIRE(tmv::thompson::aerosol_collection_efficiency(
                1.0e-3, 0.8e-6, 1.8e-5, 1.0, 260.0,
                AerosolCollector::Snow)
            == Approx(0.0002976987686342207).epsilon(2.0e-12));
    REQUIRE(tmv::thompson::aerosol_collection_efficiency(
                2.0e-3, 0.04e-6, 1.7e-5, 0.8, 250.0,
                AerosolCollector::Graupel)
            == Approx(0.0009056842446755681).epsilon(2.0e-12));
}

TEST_CASE("Thompson wet scavenging is donor limited",
          "[microphysics][thompson][aerosol][conservation]")
{
    const auto tendencies = tmv::thompson::aerosol_scavenging_tendencies(
        260.0, 0.9, 1.0e-3, 1.0e5, 8.0e-4, 6.0e-4,
        2.0e8, 1.0e6, 10.0);
    REQUIRE(tendencies.water_aerosol_kg_dry_s < 0.0);
    REQUIRE(tendencies.ice_aerosol_kg_dry_s < 0.0);
    REQUIRE(tendencies.water_aerosol_kg_dry_s >= -2.0e7);
    REQUIRE(tendencies.ice_aerosol_kg_dry_s >= -1.0e5);

    const auto clear = tmv::thompson::aerosol_scavenging_tendencies(
        260.0, 0.9, 0.0, 0.0, 0.0, 0.0, 2.0e8, 1.0e6, 10.0);
    REQUIRE(clear.water_aerosol_kg_dry_s == 0.0);
    REQUIRE(clear.ice_aerosol_kg_dry_s == 0.0);
}

TEST_CASE("Thompson wet scavenging rejects invalid inputs",
          "[microphysics][thompson][aerosol][invalid]")
{
    REQUIRE_THROWS_AS(tmv::thompson::aerosol_scavenging_tendencies(
                          260.0, 0.9, 1.0e-3, 1.0e5, 0.0, 0.0,
                          -1.0, 1.0e6, 10.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(tmv::thompson::aerosol_collection_efficiency(
                          0.0, 0.04e-6, 1.8e-5, 1.0, 280.0,
                          tmv::thompson::AerosolCollector::Rain),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(tmv::thompson::aerosol_collection_efficiency(
                          1.0e-3, 0.04e-6, 1.8e-5, 1.0, 280.0,
                          static_cast<tmv::thompson::AerosolCollector>(99)),
                      std::invalid_argument);
    REQUIRE_THROWS(tmv::thompson::aerosol_scavenging_tendencies(
        260.0, 0.9, 1.0e300, 1.0e300, 1.0e300,
        1.0e300, 1.0e300, 1.0e300, 10.0));
}
