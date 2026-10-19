#include "catch2/catch.hpp"

#include "data/grid_geometry.hpp"

#include <array>
#include <limits>
#include <string>

TEST_CASE("modern Cartesian manifests expose metre cell spacing", "[vulkan][geometry]")
{
    const std::string json = R"({
        "grid": {
            "nr": 128, "nth": 96, "nz": 64,
            "coordinate_system": "cartesian",
            "dr_m": 400, "dy_m": 500, "dtheta_rad": 500, "dz_m": 250
        }
    })";
    oglcpp::PhysicalGridSpacing spacing{};
    std::string detail;
    REQUIRE(oglcpp::parse_grid_manifest(json, 128, 96, 64, spacing, detail) ==
            oglcpp::GridManifestStatus::available);
    CHECK(spacing.x_m == Approx(400.0));
    CHECK(spacing.y_m == Approx(500.0));
    CHECK(spacing.z_m == Approx(250.0));
}

TEST_CASE("manifest v4 exposes the physical scalar range for renderer cropping",
          "[vulkan][geometry]")
{
    const std::string json = R"({
        "schema": "nimbus.step",
        "manifest_version": 4,
        "grid": {
            "coordinate_system": "cartesian",
            "dimensions": {"x": 8, "y": 7, "z": 6},
            "spacing_m": {"x": 400, "y": 500, "z": 250},
            "native_model_staggering": "c_grid",
            "valid_index_range": {"lower": [1, 1, 0], "upper": [6, 5, 5]},
            "halo_width": {"lower": [1, 1, 0], "upper": [1, 1, 0]}
        }
    })";
    oglcpp::PhysicalGridSpacing spacing{};
    oglcpp::PhysicalIndexRange range{};
    std::string detail;
    REQUIRE(oglcpp::parse_grid_manifest(json, 8, 7, 6, spacing, detail, &range) ==
            oglcpp::GridManifestStatus::available);
    CHECK(spacing.x_m == Approx(400.0));
    CHECK(spacing.y_m == Approx(500.0));
    CHECK(spacing.z_m == Approx(250.0));
    CHECK(range.lower == std::array<int, 3>{1, 1, 0});
    CHECK(range.upper == std::array<int, 3>{6, 5, 5});
    CHECK(range.nx() == 6);
    CHECK(range.ny() == 5);
    CHECK(range.nz() == 6);
}

TEST_CASE("manifest v4 rejects missing or contradictory physical ownership",
          "[vulkan][geometry]")
{
    const auto parse = [](const std::string& ownership) {
        const std::string json =
            "{\"manifest_version\":4,\"grid\":{"
            "\"coordinate_system\":\"cartesian\"," 
            "\"dimensions\":{\"x\":8,\"y\":7,\"z\":6},"
            "\"spacing_m\":{\"x\":1,\"y\":1,\"z\":1},"
            "\"native_model_staggering\":\"c_grid\"," + ownership + "}}";
        oglcpp::PhysicalGridSpacing spacing{};
        oglcpp::PhysicalIndexRange range{};
        std::string detail;
        return oglcpp::parse_grid_manifest(json, 8, 7, 6, spacing, detail, &range);
    };

    CHECK(parse("\"halo_width\":{\"lower\":[1,1,0],\"upper\":[1,1,0]}") ==
          oglcpp::GridManifestStatus::invalid);
    CHECK(parse(
        "\"valid_index_range\":{\"lower\":[0,0,0],\"upper\":[7,6,5]},"
        "\"halo_width\":{\"lower\":[1,1,0],\"upper\":[1,1,0]}") ==
          oglcpp::GridManifestStatus::invalid);
    CHECK(parse(
        "\"valid_index_range\":{\"lower\":[1,1,0],\"upper\":[6,5,5]},"
        "\"halo_width\":{\"lower\":[1,1,0],\"upper\":[0,1,0]}") ==
          oglcpp::GridManifestStatus::invalid);
}

TEST_CASE("manifest v5 retains flat geometry and rejects terrain flattening",
          "[vulkan][geometry][terrain][TE-005]")
{
    const std::string flat = R"({
        "manifest_version": 5,
        "grid": {
            "coordinate_system": "cartesian",
            "dimensions": {"x": 8, "y": 7, "z": 6},
            "spacing_m": {"x": 400, "y": 500, "z": 250},
            "native_model_staggering": "c_grid",
            "valid_index_range": {"lower": [1, 1, 0], "upper": [6, 5, 5]},
            "halo_width": {"lower": [1, 1, 0], "upper": [1, 1, 0]}
        }
    })";
    oglcpp::PhysicalGridSpacing spacing{};
    std::string detail;
    REQUIRE(oglcpp::parse_grid_manifest(flat, 8, 7, 6, spacing, detail) ==
            oglcpp::GridManifestStatus::available);

    const std::string terrain = R"({
        "manifest_version": 5,
        "grid": {
            "coordinate_system": "cartesian",
            "dimensions": {"x": 8, "y": 7, "z": 6},
            "spacing_m": {"x": 400, "y": 500, "z": 250},
            "native_model_staggering": "c_grid",
            "vertical_geometry": {"kind": "terrain_following"},
            "valid_index_range": {"lower": [1, 1, 0], "upper": [6, 5, 5]},
            "halo_width": {"lower": [1, 1, 0], "upper": [1, 1, 0]}
        }
    })";
    CHECK(oglcpp::parse_grid_manifest(terrain, 8, 7, 6, spacing, detail) ==
          oglcpp::GridManifestStatus::invalid);
    CHECK(detail.find("not supported") != std::string::npos);
}

TEST_CASE("legacy Cartesian manifests retain their y-metre compatibility alias",
          "[vulkan][geometry]")
{
    const std::string json = R"({
        "grid": {"nr": 8, "nth": 10, "nz": 4,
                 "dr_m": 300, "dtheta_rad": 450, "dz_m": 125}
    })";
    oglcpp::PhysicalGridSpacing spacing{};
    std::string detail;
    REQUIRE(oglcpp::parse_grid_manifest(json, 8, 10, 4, spacing, detail) ==
            oglcpp::GridManifestStatus::available);
    CHECK(spacing.y_m == Approx(450.0));
}

TEST_CASE("grid metadata fails closed on contradictory dimensions and spacing",
          "[vulkan][geometry]")
{
    oglcpp::PhysicalGridSpacing spacing{};
    std::string detail;
    CHECK(oglcpp::parse_grid_manifest(
              R"({"nr":8,"nth":10,"nz":5,"dr_m":1,"dy_m":1,"dz_m":1})",
              8, 10, 4, spacing, detail) == oglcpp::GridManifestStatus::invalid);
    CHECK(oglcpp::parse_grid_manifest(
              R"({"nr":8,"nth":10,"nz":4,"dr_m":-1,"dy_m":1,"dz_m":1})",
              8, 10, 4, spacing, detail) == oglcpp::GridManifestStatus::invalid);
}

TEST_CASE("non-Cartesian manifests require an explicit display geometry",
          "[vulkan][geometry]")
{
    oglcpp::PhysicalGridSpacing spacing{};
    std::string detail;
    CHECK(oglcpp::parse_grid_manifest(
              R"({"nr":8,"nth":10,"nz":4,"coordinate_system":"cylindrical",
                   "dr_m":100,"dtheta_rad":0.01,"dz_m":50})",
              8, 10, 4, spacing, detail) == oglcpp::GridManifestStatus::unavailable);
    CHECK_FALSE(spacing.available());
}

TEST_CASE("normalized world scale follows physical extents and explicit exaggeration",
          "[vulkan][geometry]")
{
    const oglcpp::PhysicalGridSpacing spacing{400.0, 400.0, 250.0};
    std::array<float, 3> scale{};
    std::string error;

    REQUIRE(oglcpp::normalized_volume_world_scale(128, 128, 64, spacing, 1.0f, scale, error));
    CHECK(scale[0] == Approx(1.0f));
    CHECK(scale[1] == Approx(0.3125f));
    CHECK(scale[2] == Approx(1.0f));

    REQUIRE(oglcpp::normalized_volume_world_scale(128, 128, 64, spacing, 2.0f, scale, error));
    CHECK(scale[1] == Approx(0.625f));

    CHECK_FALSE(oglcpp::normalized_volume_world_scale(
        128, 128, 64, spacing, std::numeric_limits<float>::infinity(), scale, error));
}
