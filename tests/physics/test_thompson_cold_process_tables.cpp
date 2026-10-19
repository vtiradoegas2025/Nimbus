#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/cold_process_tables.hpp"

#include <filesystem>
#include <fstream>

namespace
{

void require_close(double actual, double expected)
{
    REQUIRE(actual == Approx(expected).epsilon(3.0e-13));
}

}

TEST_CASE("Thompson cold tables pin WRF v4.7.1 mp=28 dimensions",
          "[microphysics][thompson][cold][tables]")
{
    using tmv::thompson::ColdProcessTableMetadata;
    REQUIRE(std::string(ColdProcessTableMetadata::upstream_release)
            == "WRF v4.7.1");
    REQUIRE(std::string(ColdProcessTableMetadata::upstream_commit) == "f52c197");
    REQUIRE(std::string(ColdProcessTableMetadata::upstream_source_sha256)
            == "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802");
    REQUIRE(tmv::thompson::checked_rain_freezing_size() == 3'388'275);
    REQUIRE(tmv::thompson::checked_cloud_freezing_size() == 9'157'500);
    REQUIRE(tmv::thompson::checked_ice_snow_size() == 3'520);
    REQUIRE(tmv::thompson::checked_cold_payload_bytes() == 255'029'280);
}

TEST_CASE("Thompson cold table cells match the pinned WRF integrations",
          "[microphysics][thompson][cold][tables][literal]")
{
    const auto rain = tmv::thompson::generate_rain_freezing_cell(
        {18, 18, 14, 27});
    const auto cloud = tmv::thompson::generate_cloud_freezing_cell(
        {18, 49, 14, 27});
    const auto ice = tmv::thompson::generate_ice_snow_cell({36, 27});
    require_close(rain.ice_mass, 8.6387778901638637e-12);
    require_close(rain.graupel_mass, 7.6210310780626602e-11);
    require_close(rain.ice_number, 1.8509222757646616e-3);
    require_close(rain.rain_number, 1.5314512445938346e-3);
    require_close(cloud.ice_mass, 6.3181239101899094e-14);
    require_close(cloud.ice_number, 2.0398661219860986e-2);
    require_close(ice.snow_mass, 3.5795382754500939e-7);
    require_close(ice.ice_number, 1.425969090705798e1);
    require_close(ice.ice_deposition_fraction, 9.2366745697939467e-1);
}

TEST_CASE("Thompson cold table loader rejects corruption atomically",
          "[microphysics][thompson][cold][tables][atomic]")
{
    namespace fs = std::filesystem;
    using namespace tmv::thompson;
    const auto valid = fs::temp_directory_path() / "nimbus-cold-table-valid.bin";
    const auto corrupt = fs::temp_directory_path() / "nimbus-cold-table-corrupt.bin";
    ColdProcessTables tiny;
    tiny.rain_freezing.resize(checked_rain_freezing_size());
    tiny.cloud_freezing.resize(checked_cloud_freezing_size());
    tiny.ice_snow.resize(checked_ice_snow_size());
    std::string error;
    REQUIRE(write_cold_process_tables(valid, tiny, error));
    ColdProcessTableStore store;
    REQUIRE(store.publish_file(valid, error));
    const auto published = store.snapshot();
    fs::copy_file(valid, corrupt, fs::copy_options::overwrite_existing);
    {
        std::fstream file(corrupt, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(file);
        file.seekp(-1, std::ios::end);
        char value = 1;
        file.write(&value, 1);
    }
    REQUIRE_FALSE(store.publish_file(corrupt, error));
    REQUIRE(error.find("hash mismatch") != std::string::npos);
    REQUIRE(store.snapshot() == published);
    REQUIRE_FALSE(store.publish_file(
        valid, error, checked_cold_payload_bytes() - 1));
    REQUIRE(error.find("resident-memory budget") != std::string::npos);
    REQUIRE(store.snapshot() == published);
    fs::remove(valid);
    fs::remove(corrupt);
}
