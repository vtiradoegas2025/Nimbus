#include "catch2/catch.hpp"
#include "init/config/config_v1_serialization_common.hpp"
#include "init/config/v1_loader.hpp"

#include <string>

using namespace tmv::init::config;

TEST_CASE("reviewed radar source profile keeps NPY exact and stages both ZFP I/O policies",
          "[init][radar_zfp_profile]")
{
    for (const char* name : {"cgrid_radar_ready.yaml",
                             "cgrid_radar_polarimetric_ready.yaml"})
    {
        const auto path = std::string("tests/fixtures/configs/qualification/") + name;
        const auto loaded = load_v1_config(path);
        REQUIRE(loaded.ok());
        const auto reference = *loaded.config;
        REQUIRE(is_public_qualified_radar_profile(reference));
        auto invalid_npy = reference;
        invalid_npy.output.async_io = true;
        REQUIRE_FALSE(is_public_qualified_radar_profile(invalid_npy));
        for (const auto tier : {OutputFormat::ZfpLow, OutputFormat::ZfpMedium,
                                OutputFormat::ZfpHigh})
        {
            for (const bool async_io : {false, true})
            {
                auto candidate = reference;
                candidate.output.format = tier;
                candidate.output.async_io = async_io;
                REQUIRE(is_public_qualified_radar_profile(candidate));
                candidate.grid.nx = 13;
                REQUIRE_FALSE(is_public_qualified_radar_profile(candidate));
            }
        }
    }
    const auto loaded = load_v1_config(
        "tests/fixtures/configs/qualification/cgrid_radar_polarimetric_ready.yaml");
    REQUIRE(loaded.ok());
    auto all_five_minimal = *loaded.config;
    all_five_minimal.output.preset = OutputPreset::Minimal;
    all_five_minimal.radar.products = {
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    REQUIRE(is_public_qualified_radar_profile(all_five_minimal));
    all_five_minimal.radar.products = {
        "rhohv", "kdp", "zdr", "radial_velocity", "reflectivity_dbz"};
    REQUIRE(is_public_qualified_radar_profile(all_five_minimal));
}
