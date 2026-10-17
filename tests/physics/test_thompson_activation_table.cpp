#include "catch2/catch.hpp"
#include "microphysics/schemes/thompson/activation_table.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>

namespace
{

const std::filesystem::path source_path =
    "tests/fixtures/reference/thompson/CCN_ACTIVATE.BIN";

std::filesystem::path temporary_path(const char* name)
{
    return std::filesystem::temp_directory_path() / name;
}

void flip_byte(const std::filesystem::path& path, std::streamoff offset)
{
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    REQUIRE(file);
    if (offset < 0) file.seekg(offset, std::ios::end);
    else file.seekg(offset, std::ios::beg);
    char byte = 0;
    file.read(&byte, 1);
    REQUIRE(file);
    byte ^= 1;
    if (offset < 0) file.seekp(offset, std::ios::end);
    else file.seekp(offset, std::ios::beg);
    file.write(&byte, 1);
    REQUIRE(file);
}

} // namespace

TEST_CASE("Thompson activation metadata pins the official mp=28 artifact",
          "[microphysics][thompson][activation]")
{
    using Metadata = tmv::thompson::ActivationTableMetadata;
    REQUIRE(std::string(Metadata::upstream_release) == "WRF v4.7.1");
    REQUIRE(std::string(Metadata::upstream_commit) == "f52c197");
    REQUIRE(std::string(Metadata::upstream_module_sha256)
            == "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802");
    REQUIRE(std::string(Metadata::upstream_license_sha256)
            == "b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6");
    REQUIRE(std::string(Metadata::upstream_sha256)
            == "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd");
    REQUIRE(std::string(Metadata::payload_sha256)
            == "547e2592d01bae74e096f3e280e4eacd8ce22562a5a412039b28bed2c2c5d2be");
    REQUIRE(std::string(Metadata::packaged_sha256)
            == "6f72d69fcedb89b3d9e978e3f5b2a0eda204c5652114a4f4e7e0d033a73f66bf");
    REQUIRE(std::string(Metadata::packaged_path)
            == "data/thompson/ccn_activation.ntact");
    REQUIRE(Metadata::element_count == 8820);
    REQUIRE(Metadata::selected_mean_radius_index == 2);
    REQUIRE(Metadata::selected_hygroscopicity_index == 1);
    REQUIRE(Metadata::selected_mean_radius_um == Approx(0.04));
    REQUIRE(Metadata::selected_hygroscopicity == Approx(0.4));
}

TEST_CASE("Thompson imports the WRF big-endian record with literal values",
          "[microphysics][thompson][activation][literal]")
{
    tmv::thompson::ActivationTable table;
    std::string error;
    REQUIRE(tmv::thompson::import_wrf_activation_table(source_path, table, error));
    REQUIRE(table.payload_sha256
            == tmv::thompson::ActivationTableMetadata::payload_sha256);
    REQUIRE(table.fractions[tmv::thompson::activation_flatten(2, 4, 3, 2, 1)]
            == Approx(0.8682309389114380));
    REQUIRE(table.fractions[tmv::thompson::activation_flatten(3, 4, 3, 2, 1)]
            == Approx(0.7615879774093628));
    REQUIRE(table.fractions[tmv::thompson::activation_flatten(3, 5, 3, 2, 1)]
            == Approx(0.9175649881362915));

    REQUIRE(tmv::thompson::activated_cloud_number_m3(
                table, 273.15, 1.0, 300.0e6)
            == Approx(229921189.4560381).epsilon(2.0e-13));
    REQUIRE(tmv::thompson::activated_cloud_number_m3(
                table, 303.15, 100.0, 10000.0e6)
            == Approx(9676568881.661604).epsilon(2.0e-13));
    REQUIRE(tmv::thompson::activated_cloud_number_m3(
                table, 240.0, -0.5, 1.0e6)
            == Approx(492503.4828811119).epsilon(2.0e-13));
}

TEST_CASE("Thompson activation portable conversion is deterministic and bounded",
          "[microphysics][thompson][activation][portable]")
{
    const auto path = temporary_path("nimbus_thompson_activation_portable.bin");
    std::filesystem::remove(path);
    tmv::thompson::ActivationTable imported;
    std::string error;
    REQUIRE(tmv::thompson::import_wrf_activation_table(source_path, imported, error));
    REQUIRE(tmv::thompson::write_activation_table(path, imported, error));

    std::ifstream generated(path, std::ios::binary);
    std::ifstream packaged(
        tmv::thompson::ActivationTableMetadata::packaged_path,
        std::ios::binary);
    REQUIRE(generated);
    REQUIRE(packaged);
    REQUIRE(std::vector<char>(std::istreambuf_iterator<char>(generated),
                              std::istreambuf_iterator<char>())
            == std::vector<char>(std::istreambuf_iterator<char>(packaged),
                                 std::istreambuf_iterator<char>()));

    tmv::thompson::ActivationTable loaded;
    REQUIRE(tmv::thompson::load_activation_table(path, loaded, error));
    REQUIRE(loaded.payload_sha256 == imported.payload_sha256);
    REQUIRE(loaded.fractions == imported.fractions);
    REQUIRE_FALSE(tmv::thompson::load_activation_table(
        path,
        loaded,
        error,
        tmv::thompson::ActivationTableMetadata::element_count * sizeof(float) - 1));
    REQUIRE(error.find("resident-memory budget") != std::string::npos);
    std::filesystem::remove(path);
}

TEST_CASE("Thompson activation publication rejects source and cache corruption atomically",
          "[microphysics][thompson][activation][integrity][atomic]")
{
    const auto missing = temporary_path("nimbus_thompson_activation_missing.bin");
    const auto portable = temporary_path("nimbus_thompson_activation_valid.bin");
    const auto corrupt_portable = temporary_path("nimbus_thompson_activation_corrupt.bin");
    const auto corrupt_marker = temporary_path("nimbus_thompson_activation_marker.bin");
    const auto corrupt_source = temporary_path("nimbus_thompson_activation_source.bin");
    std::filesystem::remove(missing);
    std::filesystem::remove(portable);
    std::filesystem::remove(corrupt_portable);
    std::filesystem::remove(corrupt_marker);
    std::filesystem::remove(corrupt_source);

    tmv::thompson::ActivationTableStore store;
    std::string error;
    REQUIRE_FALSE(store.publish_file(missing, error));
    REQUIRE(error == "unable to open Thompson activation table");
    REQUIRE_FALSE(store.snapshot());
    REQUIRE(store.publish_wrf_file(source_path, error));
    const auto source_snapshot = store.snapshot();
    REQUIRE(source_snapshot);
    REQUIRE(tmv::thompson::write_activation_table(portable, *source_snapshot, error));
    REQUIRE(store.publish_file(portable, error));
    const auto portable_snapshot = store.snapshot();
    REQUIRE(portable_snapshot != source_snapshot);

    std::filesystem::copy_file(portable, corrupt_portable);
    flip_byte(corrupt_portable, -1);
    REQUIRE_FALSE(store.publish_file(corrupt_portable, error));
    REQUIRE(error.find("hash mismatch") != std::string::npos);
    REQUIRE(store.snapshot() == portable_snapshot);

    std::filesystem::copy_file(source_path, corrupt_marker);
    flip_byte(corrupt_marker, 0);
    REQUIRE_FALSE(store.publish_wrf_file(corrupt_marker, error));
    REQUIRE(error.find("record markers") != std::string::npos);
    REQUIRE(store.snapshot() == portable_snapshot);

    std::filesystem::copy_file(source_path, corrupt_source);
    flip_byte(corrupt_source, 8);
    REQUIRE_FALSE(store.publish_wrf_file(corrupt_source, error));
    REQUIRE(error.find("source hash mismatch") != std::string::npos);
    REQUIRE(store.snapshot() == portable_snapshot);

    std::filesystem::remove(portable);
    std::filesystem::remove(corrupt_portable);
    std::filesystem::remove(corrupt_marker);
    std::filesystem::remove(corrupt_source);
}

TEST_CASE("Thompson activation rejects invalid indices and inputs",
          "[microphysics][thompson][activation][invalid]")
{
    REQUIRE_THROWS_AS(tmv::thompson::activation_flatten(7, 0, 0, 0, 0),
                      std::out_of_range);
    tmv::thompson::ActivationTable table;
    std::string error;
    REQUIRE(tmv::thompson::import_wrf_activation_table(source_path, table, error));
    REQUIRE_THROWS_AS(tmv::thompson::activated_cloud_number_m3(
                          table,
                          std::numeric_limits<double>::quiet_NaN(),
                          1.0,
                          100.0e6),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(tmv::thompson::activated_cloud_number_m3(
                          table, 273.15, 1.0, -1.0),
                      std::invalid_argument);
    REQUIRE(tmv::thompson::activated_cloud_number_m3(table, std::numeric_limits<double>::max(), 1.0, 100e6) ==
            tmv::thompson::activated_cloud_number_m3(table, 303.15, 1.0, 100e6));
    REQUIRE(tmv::thompson::activated_cloud_number_m3(table, std::numeric_limits<double>::denorm_min(), 1.0, 100e6) ==
            tmv::thompson::activated_cloud_number_m3(table, 243.15, 1.0, 100e6));
}
