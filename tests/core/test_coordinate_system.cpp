/**
 * @file test_coordinate_system.cpp
 * @brief Unit tests for the CoordinateSystem enum, name helper, and parser.
 *
 * The model is Cartesian-only (the cylindrical backend was removed). The
 * enum is single-valued: the parser must accept the
 * canonical Cartesian labels and aliases, must reject empty, unknown, and
 * the now-retired cylindrical labels without modifying its output, and must
 * report a name that round-trips cleanly through `parse_coordinate_system`.
 */

#include "catch2/catch.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/grid_geometry.hpp"

#include <string>

TEST_CASE("CoordinateSystem default is Cartesian", "[core][coordinate_system]")
{
    // The single enum value is Cartesian (0); a default-constructed enum
    // value lands on it.
    CoordinateSystem system{};
    REQUIRE(system == CoordinateSystem::Cartesian);
    REQUIRE(static_cast<int>(system) == 0);
}

TEST_CASE("coordinate_system_name returns the canonical label", "[core][coordinate_system]")
{
    REQUIRE(std::string(coordinate_system_name(CoordinateSystem::Cartesian)) == "cartesian");
}

TEST_CASE("parse_coordinate_system accepts canonical cartesian labels",
          "[core][coordinate_system]")
{
    CoordinateSystem result{};
    REQUIRE(parse_coordinate_system("cartesian", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    REQUIRE(parse_coordinate_system("Cartesian", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    REQUIRE(parse_coordinate_system("CARTESIAN", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

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

TEST_CASE("parse_coordinate_system rejects the retired cylindrical labels",
          "[core][coordinate_system]")
{
    // Cylindrical was removed; its labels must no longer parse.
    CoordinateSystem result = CoordinateSystem::Cartesian;
    REQUIRE_FALSE(parse_coordinate_system("cylindrical", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    REQUIRE_FALSE(parse_coordinate_system("Cylindrical", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    REQUIRE_FALSE(parse_coordinate_system("cyl", result));
    REQUIRE(result == CoordinateSystem::Cartesian);
}

TEST_CASE("parse_coordinate_system rejects unknown values", "[core][coordinate_system]")
{
    CoordinateSystem result = CoordinateSystem::Cartesian;
    REQUIRE_FALSE(parse_coordinate_system("polar", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    REQUIRE_FALSE(parse_coordinate_system("xy", result));
    REQUIRE(result == CoordinateSystem::Cartesian);

    REQUIRE_FALSE(parse_coordinate_system("cartisian", result));  // common typo
    REQUIRE(result == CoordinateSystem::Cartesian);

    REQUIRE_FALSE(parse_coordinate_system("   ", result));  // whitespace only
    REQUIRE(result == CoordinateSystem::Cartesian);
}

TEST_CASE("coordinate_system_name label round-trips through parse_coordinate_system",
          "[core][coordinate_system]")
{
    const char* label = coordinate_system_name(CoordinateSystem::Cartesian);
    CoordinateSystem parsed{};
    REQUIRE(parse_coordinate_system(label, parsed));
    REQUIRE(parsed == CoordinateSystem::Cartesian);
}

TEST_CASE("SCI-019 Cartesian scalar centers and faces share one coordinate contract",
          "[core][coordinates][SCI-019]")
{
    using namespace tmv::grid_coordinates;

    REQUIRE(face(0, 200.0) == Approx(0.0));
    REQUIRE(lower_face(3, 200.0) == Approx(600.0));
    REQUIRE(scalar_center(3, 200.0) == Approx(700.0));
    REQUIRE(upper_face(3, 200.0) == Approx(800.0));
    REQUIRE(physical_extent(4, 200.0) == Approx(800.0));
    REQUIRE(scalar_fractional_index(scalar_center(3, 200.0), 200.0) == Approx(3.0));
    REQUIRE(upper_face_fractional_index(upper_face(3, 200.0), 200.0) == Approx(3.0));

    GridGeometry geometry;
    geometry.initialize(4, 3, 2, 200.0, 50.0, 100.0,
                        CoordinateSystem::Cartesian, StaggerType::CGrid);
    REQUIRE(geometry.staggered);
    REQUIRE(geometry.r == std::vector<double>{100.0, 300.0, 500.0, 700.0});
    REQUIRE(geometry.theta == std::vector<double>{50.0, 150.0, 250.0});
    REQUIRE(geometry.z == std::vector<double>{25.0, 75.0});
    REQUIRE(geometry.r_faces == std::vector<double>{0.0, 200.0, 400.0, 600.0, 800.0});
    REQUIRE(geometry.theta_faces == std::vector<double>{0.0, 100.0, 200.0, 300.0});
    REQUIRE(geometry.z_faces == std::vector<double>{0.0, 50.0, 100.0});
}
