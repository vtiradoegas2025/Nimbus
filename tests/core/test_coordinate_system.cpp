/**
 * @file test_coordinate_system.cpp
 * @brief Unit tests for the CoordinateSystem enum, name helper, and parser.
 *
 * Covers Phase A.1 of the Coordinate Backend Plan: the parser must accept
 * the canonical labels and a small set of aliases, must reject empty and
 * unknown values without modifying its output, and must report names that
 * round-trip cleanly through `parse_coordinate_system`.
 */

#include "catch2/catch.hpp"
#include "core/infra/coordinate_system.hpp"

#include <string>

TEST_CASE("CoordinateSystem default is Cylindrical", "[core][coordinate_system]")
{
    // The default-constructed enum value is Cylindrical (0). This protects
    // existing tornado configs from accidentally flipping to Cartesian if
    // a global is created without explicit initialization.
    CoordinateSystem system{};
    REQUIRE(system == CoordinateSystem::Cylindrical);
    REQUIRE(static_cast<int>(system) == 0);
}

TEST_CASE("coordinate_system_name returns canonical labels", "[core][coordinate_system]")
{
    REQUIRE(std::string(coordinate_system_name(CoordinateSystem::Cylindrical)) == "cylindrical");
    REQUIRE(std::string(coordinate_system_name(CoordinateSystem::Cartesian)) == "cartesian");
}

TEST_CASE("parse_coordinate_system accepts canonical cylindrical labels",
          "[core][coordinate_system]")
{
    CoordinateSystem result = CoordinateSystem::Cartesian;  // sentinel != default
    REQUIRE(parse_coordinate_system("cylindrical", result));
    REQUIRE(result == CoordinateSystem::Cylindrical);

    result = CoordinateSystem::Cartesian;
    REQUIRE(parse_coordinate_system("Cylindrical", result));
    REQUIRE(result == CoordinateSystem::Cylindrical);

    result = CoordinateSystem::Cartesian;
    REQUIRE(parse_coordinate_system("CYLINDRICAL", result));
    REQUIRE(result == CoordinateSystem::Cylindrical);

    result = CoordinateSystem::Cartesian;
    REQUIRE(parse_coordinate_system("cyl", result));
    REQUIRE(result == CoordinateSystem::Cylindrical);
}

TEST_CASE("parse_coordinate_system accepts canonical cartesian labels",
          "[core][coordinate_system]")
{
    CoordinateSystem result = CoordinateSystem::Cylindrical;  // sentinel != default
    REQUIRE(parse_coordinate_system("cartesian", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    result = CoordinateSystem::Cylindrical;
    REQUIRE(parse_coordinate_system("Cartesian", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    result = CoordinateSystem::Cylindrical;
    REQUIRE(parse_coordinate_system("CARTESIAN", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    result = CoordinateSystem::Cylindrical;
    REQUIRE(parse_coordinate_system("cart", result));
    REQUIRE(result == CoordinateSystem::Cartesian);
}

TEST_CASE("parse_coordinate_system rejects empty input", "[core][coordinate_system]")
{
    CoordinateSystem result = CoordinateSystem::Cartesian;
    REQUIRE_FALSE(parse_coordinate_system("", result));
    // On failure, the output must be left untouched.
    REQUIRE(result == CoordinateSystem::Cartesian);
}

TEST_CASE("parse_coordinate_system rejects unknown values", "[core][coordinate_system]")
{
    CoordinateSystem result = CoordinateSystem::Cartesian;
    REQUIRE_FALSE(parse_coordinate_system("polar", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    result = CoordinateSystem::Cylindrical;
    REQUIRE_FALSE(parse_coordinate_system("xy", result));
    REQUIRE(result == CoordinateSystem::Cylindrical);

    result = CoordinateSystem::Cylindrical;
    REQUIRE_FALSE(parse_coordinate_system("cartisian", result));  // common typo
    REQUIRE(result == CoordinateSystem::Cylindrical);

    result = CoordinateSystem::Cylindrical;
    REQUIRE_FALSE(parse_coordinate_system("   ", result));  // whitespace only
    REQUIRE(result == CoordinateSystem::Cylindrical);
}

TEST_CASE("coordinate_system_name labels round-trip through parse_coordinate_system",
          "[core][coordinate_system]")
{
    for (const auto system : {CoordinateSystem::Cylindrical, CoordinateSystem::Cartesian})
    {
        const char* label = coordinate_system_name(system);
        CoordinateSystem parsed{};
        REQUIRE(parse_coordinate_system(label, parsed));
        REQUIRE(parsed == system);
    }
}
