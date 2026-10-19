#include "catch2/catch.hpp"
#include "core/infra/physical_constants.hpp"
#include "soundings/soundings_base.hpp"
#include "soundings/soundings.hpp"
#include "soundings/base/sounding_algorithms.hpp"
#include "soundings/schemes/sharpy/sharpy_native_reader.hpp"
#include "soundings/schemes/sharpy/sharpy_native_common.hpp"
#include "soundings/schemes/sharpy/sharpy_profile_parse.hpp"

#include <cmath>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

TEST_CASE("SoundingData default construction has empty vectors", "[data][soundings]")
{
    SoundingData data;
    REQUIRE(data.height_m.empty());
    REQUIRE(data.pressure_hpa.empty());
    REQUIRE(data.temperature_k.empty());
}

TEST_CASE("SoundingData stores multi-level profile correctly", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 5000.0, 10000.0};
    data.pressure_hpa = {1013.25, 900.0, 800.0, 550.0, 265.0};
    data.temperature_k = {288.0, 281.5, 275.0, 255.0, 223.0};

    REQUIRE(data.height_m.size() == 5);

    // Physical consistency: pressure decreases with height
    for (size_t k = 1; k < data.pressure_hpa.size(); ++k)
        REQUIRE(data.pressure_hpa[k] < data.pressure_hpa[k - 1]);

    // Temperature decreases in troposphere
    REQUIRE(data.temperature_k.back() < data.temperature_k.front());
}

TEST_CASE("Sounding factory creates sharpy scheme", "[data][soundings]")
{
    auto scheme = create_sounding_scheme("sharpy");
    REQUIRE(scheme != nullptr);
    REQUIRE(create_sounding_scheme("cm1") != nullptr);
    REQUIRE(create_sounding_scheme("auto") != nullptr);
}

TEST_CASE("SHARPY text profile rejects unchecked input growth before parsing",
          "[data][soundings][sharpy][resource][guard]")
{
    const std::filesystem::path fixture = "tests/init/fixtures/test_supercell.tmv";
    std::ifstream input(fixture, std::ios::binary);
    REQUIRE(input.good());
    const std::string valid((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    REQUIRE(tmv::soundings::sharpy::read_sharpy_text_file(fixture.string()).num_levels() == 33);

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("nimbus-sharpy-input-bounds-" + std::to_string(nonce) + ".tmv");
    const auto write = [&](const std::string& text)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        REQUIRE(output.good());
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        REQUIRE(output.good());
    };
    const auto replace_once = [&](const std::string& old_value, const std::string& new_value)
    {
        std::string text = valid;
        const std::size_t position = text.find(old_value);
        REQUIRE(position != std::string::npos);
        text.replace(position, old_value.size(), new_value);
        return text;
    };

    write(replace_once("levels\t33", "levels\t100001"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("100,000-level safety limit"));

    write(replace_once("levels\t33", "levels\t184467440737095516160000"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("invalid levels header"));
    write(replace_once("levels\t33", "levels\t33junk"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("invalid levels header"));

    write(valid + "levels\t33\n");
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("duplicate levels header"));

    write(valid + "#" + std::string(4097, 'x') + "\n");
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("4096-byte line safety limit"));
    write(std::string(4097, 'x') + "\n" + valid);
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("4096-byte line safety limit"));

    write(valid);
    std::filesystem::resize_file(path, 8u * 1024u * 1024u + 1u);
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("8 MiB safety limit"));

    REQUIRE(valid.back() == '\n');
    write(valid.substr(0, valid.size() - 1));
    REQUIRE(tmv::soundings::sharpy::read_sharpy_text_file(path.string()).num_levels() == 33);

    write(replace_once("levels\t33", "levels\t34"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::read_sharpy_text_file(path.string()),
                        Catch::Matchers::Contains("does not match declared levels"));
    std::filesystem::remove(path);
}

TEST_CASE("SHARPY classic NetCDF bounds dimension count before reserving",
          "[data][soundings][sharpy][netcdf][classic][resource]")
{
    for (const char* version : {"cdf1", "cdf2"})
    {
        const std::string path = std::string("tests/init/fixtures/sharpy_netcdf_classic_safe_") +
                                 version + ".nc";
        const auto profile = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path);
        REQUIRE(profile.rows.size() == 33);
        REQUIRE(profile.rows.front().size() == 6);
        REQUIRE(profile.rows.front()[0] == 0.0);
        REQUIRE(profile.rows.front()[1] == 1000.0);
        REQUIRE(profile.rows.front()[2] == 301.0);
        REQUIRE(profile.rows.back()[0] == 16000.0);
    }

    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("nimbus-sharpy-classic-dimensions-" + std::to_string(nonce) + ".cdf");
    const std::array<unsigned char, 16> malformed = {
        'C', 'D', 'F', 1, 0, 0, 0, 0, 0, 0, 0, 10, 0, 1, 0x86, 0xa1};
    {
        std::ofstream output(path, std::ios::binary);
        REQUIRE(output.good());
        output.write(reinterpret_cast<const char*>(malformed.data()),
                     static_cast<std::streamsize>(malformed.size()));
        REQUIRE(output.good());
    }
    REQUIRE(std::filesystem::file_size(path) == malformed.size());
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path.string()),
                        Catch::Matchers::Contains("dimension count exceeds available header bytes"));
    std::filesystem::remove(path);
}

TEST_CASE("SHARPY classic NetCDF bounds variable count before reserving",
          "[data][soundings][sharpy][netcdf][classic][resource]")
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int version : {1, 2})
    {
        const std::string suffix = version == 1 ? "cdf1" : "cdf2";
        const std::string valid = "tests/init/fixtures/sharpy_netcdf_classic_safe_" + suffix + ".nc";
        const auto profile = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(valid);
        REQUIRE(profile.rows.size() == 33);
        REQUIRE(profile.rows.front()[0] == 0.0);
        REQUIRE(profile.rows.back()[0] == 16000.0);

        std::array<unsigned char, 32> malformed{};
        malformed[0] = 'C';
        malformed[1] = 'D';
        malformed[2] = 'F';
        malformed[3] = static_cast<unsigned char>(version);
        malformed[27] = 11;
        malformed[29] = 1;
        malformed[30] = 0x86;
        malformed[31] = 0xa1;
        const std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("nimbus-sharpy-classic-variables-" + std::to_string(nonce) + "-" + suffix + ".nc");
        {
            std::ofstream output(path, std::ios::binary);
            REQUIRE(output.good());
            output.write(reinterpret_cast<const char*>(malformed.data()),
                         static_cast<std::streamsize>(malformed.size()));
            REQUIRE(output.good());
        }
        REQUIRE(std::filesystem::file_size(path) == malformed.size());
        REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path.string()),
                            Catch::Matchers::Contains("variable count exceeds available header bytes"));
        std::filesystem::remove(path);
    }
}

TEST_CASE("SHARPY classic NetCDF checks attribute payload before allocating",
          "[data][soundings][sharpy][netcdf][classic][resource]")
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int version : {1, 2})
    {
        const std::string suffix = version == 1 ? "cdf1" : "cdf2";
        const std::string valid = "tests/init/fixtures/sharpy_netcdf_classic_attribute_safe_" +
                                  suffix + ".nc";
        const auto profile = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(valid);
        REQUIRE(profile.rows.size() == 2);
        REQUIRE(profile.rows.front()[0] == 0.0);
        REQUIRE(profile.rows.back()[0] == 1000.0);
        REQUIRE(profile.station_id == std::string(32, 'M'));

        std::array<unsigned char, 40> malformed{};
        malformed[0] = 'C';
        malformed[1] = 'D';
        malformed[2] = 'F';
        malformed[3] = static_cast<unsigned char>(version);
        malformed[19] = 12;
        malformed[23] = 1;
        malformed[27] = 1;
        malformed[28] = 'x';
        malformed[35] = 6;
        malformed[37] = 0x0f;
        malformed[38] = 0x42;
        malformed[39] = 0x41;
        const std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("nimbus-sharpy-classic-attribute-" + std::to_string(nonce) + "-" + suffix + ".nc");
        {
            std::ofstream output(path, std::ios::binary);
            REQUIRE(output.good());
            output.write(reinterpret_cast<const char*>(malformed.data()),
                         static_cast<std::streamsize>(malformed.size()));
            REQUIRE(output.good());
        }
        REQUIRE(std::filesystem::file_size(path) == malformed.size());
        REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path.string()),
                            Catch::Matchers::Contains("attribute payload exceeds available file bytes"));
        std::filesystem::remove(path);
    }
}

TEST_CASE("SHARPY classic NetCDF checks name bytes before allocating",
          "[data][soundings][sharpy][netcdf][classic][resource]")
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int version : {1, 2})
    {
        const std::string suffix = version == 1 ? "cdf1" : "cdf2";
        const auto valid = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(
            "tests/init/fixtures/sharpy_netcdf_classic_safe_" + suffix + ".nc");
        REQUIRE(valid.rows.size() == 33);

        std::array<unsigned char, 24> malformed{};
        malformed[0] = 'C';
        malformed[1] = 'D';
        malformed[2] = 'F';
        malformed[3] = static_cast<unsigned char>(version);
        malformed[11] = 10;
        malformed[15] = 1;
        malformed[16] = 1;
        const std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("nimbus-sharpy-classic-name-" + std::to_string(nonce) + "-" + suffix + ".nc");
        {
            std::ofstream output(path, std::ios::binary);
            REQUIRE(output.good());
            output.write(reinterpret_cast<const char*>(malformed.data()),
                         static_cast<std::streamsize>(malformed.size()));
            REQUIRE(output.good());
        }
        REQUIRE(std::filesystem::file_size(path) == malformed.size());
        REQUIRE_THROWS_WITH(
            tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path.string()),
            Catch::Matchers::Contains("NetCDF name length exceeds available file bytes"));
        std::filesystem::remove(path);
    }
}

TEST_CASE("SHARPY classic NetCDF bounds variable dimension IDs before reserving",
          "[data][soundings][sharpy][netcdf][classic][resource]")
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int version : {1, 2})
    {
        const std::string suffix = version == 1 ? "cdf1" : "cdf2";
        const auto valid = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(
            "tests/init/fixtures/sharpy_netcdf_classic_safe_" + suffix + ".nc");
        REQUIRE(valid.rows.size() == 33);

        std::array<unsigned char, 64> malformed{};
        malformed[0] = 'C';
        malformed[1] = 'D';
        malformed[2] = 'F';
        malformed[3] = static_cast<unsigned char>(version);
        malformed[27] = 11;
        malformed[31] = 1;
        malformed[36] = 0x7f;
        malformed[37] = 0xff;
        malformed[38] = 0xff;
        malformed[39] = 0xff;
        const std::filesystem::path path = std::filesystem::temp_directory_path() /
            ("nimbus-sharpy-classic-dimension-ids-" + std::to_string(nonce) + "-" + suffix + ".nc");
        {
            std::ofstream output(path, std::ios::binary);
            REQUIRE(output.good());
            output.write(reinterpret_cast<const char*>(malformed.data()),
                         static_cast<std::streamsize>(malformed.size()));
            REQUIRE(output.good());
        }
        REQUIRE_THROWS_WITH(
            tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path.string()),
            Catch::Matchers::Contains("dimension ID count exceeds available file bytes"));
        std::filesystem::remove(path);
    }
}

TEST_CASE("SHARPY classic CDF selected profiles obey the shared level limit",
          "[data][soundings][sharpy][netcdf][classic][limits]")
{
    auto reader = create_sounding_scheme("sharpy");
    REQUIRE(reader != nullptr);
    SoundingConfig config;
    config.scheme_id = "sharpy";
    config.use_fallback_profiles = false;
    reader->initialize(config);
    for (int version : {1, 2})
    {
        const auto path = [version](const char* key)
        {
            return "tests/init/fixtures/sharpy_classic_limits_" + std::string(key) +
                   "_cdf" + std::to_string(version) + ".nc";
        };
        const auto boundary = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(
            path("fixed_100000"));
        REQUIRE(boundary.rows.size() == 100000);
        CHECK(boundary.rows.front()[0] == 0.0);
        CHECK(boundary.rows.back()[0] == 16000.0);
        const auto record = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(
            path("record_33"));
        REQUIRE(record.rows.size() == 33);
        CHECK(record.rows.front()[1] == 1000.0);
        CHECK(record.rows.back()[0] == 16000.0);
        for (const char* key : {"fixed_100001", "record_100001", "optional_100001"})
        {
            CHECK_THROWS_AS(tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path(key)),
                            tmv::soundings::sharpy::NetcdfProfileResourceError);
            CHECK_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path(key)),
                              Catch::Matchers::Contains("100,000-level safety limit"));
            CHECK_THROWS_AS(reader->load_sounding(path(key)),
                            tmv::soundings::sharpy::NetcdfProfileResourceError);
        }
    }
}

TEST_CASE("SHARPY classic CDF metadata obeys the shared scalar and byte contract",
          "[data][soundings][sharpy][netcdf][classic][limits]")
{
    auto reader = create_sounding_scheme("sharpy");
    REQUIRE(reader != nullptr);
    SoundingConfig config;
    config.scheme_id = "sharpy";
    config.use_fallback_profiles = false;
    reader->initialize(config);
    for (int version : {1, 2})
    {
        const auto path = [version](const char* key)
        {
            return "tests/init/fixtures/sharpy_classic_limits_" + std::string(key) +
                   "_cdf" + std::to_string(version) + ".nc";
        };
        const auto station = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(
            path("station_1023"));
        CHECK(station.station_id == std::string(1023, 'M'));
        const auto timestamp = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(
            path("timestamp_1023"));
        CHECK(timestamp.timestamp_utc == std::string(1023, 'T'));
        const auto latitude = tmv::soundings::sharpy::parse_netcdf_classic_profile_native(
            path("latitude_scalar"));
        CHECK(latitude.latitude_deg == 18.5);
        for (const char* key : {"station_1024", "timestamp_1024", "latitude_array"})
        {
            CHECK_THROWS_AS(tmv::soundings::sharpy::parse_netcdf_classic_profile_native(path(key)),
                            tmv::soundings::sharpy::NetcdfMetadataContractError);
            CHECK_THROWS_AS(reader->load_sounding(path(key)),
                            tmv::soundings::sharpy::NetcdfMetadataContractError);
        }
    }
}

TEST_CASE("SHARPY HDF5 metadata rejects oversized, variable and array values before reading",
          "[data][soundings][sharpy][hdf5][resource][guard]")
{
    const auto path = [](const char* name)
    {
        return std::string("tests/init/fixtures/sharpy_hdf5_") + name + ".h5";
    };
    try
    {
        const auto safe = tmv::soundings::sharpy::parse_hdf5_profile_native(path("safe_attribute"));
        REQUIRE(safe.rows.size() == 2);
        REQUIRE(safe.station_id.size() == 63);
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("Unable to load HDF5 shared library") != std::string::npos)
        {
            SUCCEED("HDF5 runtime unavailable; installed-package qualification owns availability");
            return;
        }
        throw;
    }
    const auto dataset = tmv::soundings::sharpy::parse_hdf5_profile_native(path("safe_dataset"));
    REQUIRE(dataset.rows.size() == 2);
    REQUIRE(dataset.station_id.size() == 15);
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_hdf5_profile_native(path("oversized_attribute")),
                        Catch::Matchers::Contains("1023-byte fixed metadata contract"));
    REQUIRE_THROWS_AS(tmv::soundings::sharpy::parse_hdf5_profile_native(path("oversized_attribute")),
                      tmv::soundings::sharpy::Hdf5MetadataContractError);
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_hdf5_profile_native(path("oversized_dataset")),
                        Catch::Matchers::Contains("1023-byte fixed metadata contract"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_hdf5_profile_native(path("variable_attribute")),
                        Catch::Matchers::Contains("variable-length string attribute"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_hdf5_profile_native(path("variable_dataset")),
                        Catch::Matchers::Contains("variable-length string dataset"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_hdf5_profile_native(path("array_dataset")),
                        Catch::Matchers::Contains("not scalar"));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_hdf5_profile_native(path("numeric_array")),
                        Catch::Matchers::Contains("not scalar"));
}

TEST_CASE("SHARPY HDF5 numeric profiles cap decoded levels without narrowing singleton dimensions",
          "[data][soundings][sharpy][hdf5][resource][volume]")
{
    const auto path = [](const char* name)
    {
        return std::string("tests/init/fixtures/sharpy_hdf5_numeric_") + name + ".h5";
    };
    REQUIRE(std::filesystem::file_size(path("oversized_100001")) < 65536);
    try
    {
        const auto boundary = tmv::soundings::sharpy::parse_hdf5_profile_native(path("safe_100000"));
        REQUIRE(boundary.rows.size() == 100000);
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("Unable to load HDF5 shared library") != std::string::npos)
        {
            SUCCEED("HDF5 runtime unavailable; installed-package qualification owns availability");
            return;
        }
        throw;
    }
    const auto singleton = tmv::soundings::sharpy::parse_hdf5_profile_native(path("singleton_2d_33"));
    REQUIRE(singleton.rows.size() == 33);
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_hdf5_profile_native(path("oversized_100001")),
                        Catch::Matchers::Contains("100,000-level safety limit"));
    REQUIRE_THROWS_AS(tmv::soundings::sharpy::parse_hdf5_profile_native(path("oversized_100001")),
                      tmv::soundings::sharpy::Hdf5ProfileResourceError);
}

TEST_CASE("SHARPY NetCDF dynamic profiles cap decoded levels and retain scalar and singleton inputs",
          "[data][soundings][sharpy][netcdf][resource][volume]")
{
    const auto path = [](const char* name)
    {
        return std::string("tests/init/fixtures/sharpy_netcdf_numeric_") + name + ".nc";
    };
    REQUIRE(std::filesystem::file_size(path("oversized_100001")) < 65536);
    try
    {
        const auto boundary = tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(path("safe_100000"));
        REQUIRE(boundary.rows.size() == 100000);
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("Unable to load NetCDF shared library") != std::string::npos)
        {
            SUCCEED("NetCDF runtime unavailable; installed-package qualification owns availability");
            return;
        }
        throw;
    }
    const auto singleton = tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(path("singleton_2d_33"));
    REQUIRE(singleton.rows.size() == 33);
    REQUIRE(singleton.rows.front()[3] == Approx(280.0));
    REQUIRE(singleton.rows.back()[3] == Approx(280.0));
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(path("oversized_100001")),
                        Catch::Matchers::Contains("100,000-level safety limit"));
    REQUIRE_THROWS_AS(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(path("oversized_100001")),
                      tmv::soundings::sharpy::NetcdfProfileResourceError);
}

TEST_CASE("SHARPY NetCDF numeric coordinate metadata is scalar before conversion",
          "[data][soundings][sharpy][netcdf][metadata][guard]")
{
    const std::string scalar = "tests/init/fixtures/sharpy_netcdf_metadata_scalar_latitude.nc";
    const std::string array = "tests/init/fixtures/sharpy_netcdf_metadata_array_latitude.nc";
    REQUIRE(std::filesystem::file_size(scalar) < 65536);
    REQUIRE(std::filesystem::file_size(array) < 65536);
    try
    {
        const auto valid = tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(scalar);
        REQUIRE(valid.rows.size() == 2);
        REQUIRE(valid.latitude_deg == Approx(35.0));
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("Unable to load NetCDF shared library") != std::string::npos)
        {
            SUCCEED("NetCDF runtime unavailable; installed-package qualification owns availability");
            return;
        }
        throw;
    }
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(array),
                        Catch::Matchers::Contains("numeric metadata must be scalar"));
    REQUIRE_THROWS_AS(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(array),
                      tmv::soundings::sharpy::NetcdfMetadataContractError);
}

TEST_CASE("SHARPY NetCDF NC_STRING station and timestamp metadata require one value",
          "[data][soundings][sharpy][netcdf][metadata][string]")
{
    const auto path = [](const char* name)
    {
        return std::string("tests/init/fixtures/sharpy_netcdf_") + name + ".nc";
    };
    try
    {
        const auto station = tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
            path("string_station_scalar"));
        REQUIRE(station.rows.size() == 2);
        REQUIRE(station.station_id == "STATION_A");
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("Unable to load NetCDF shared library") != std::string::npos)
        {
            SUCCEED("NetCDF runtime unavailable; installed-package qualification owns availability");
            return;
        }
        throw;
    }
    const auto timestamp = tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
        path("string_timestamp_scalar"));
    REQUIRE(timestamp.timestamp_utc == "2026-09-29T00:00:00Z");
    const auto char_station = tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
        path("char_station_scalar"));
    REQUIRE(char_station.station_id == "STATION_A");
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                            path("string_station_array")),
                        Catch::Matchers::Contains("string metadata must be scalar"));
    REQUIRE_THROWS_AS(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                          path("string_station_array")),
                      tmv::soundings::sharpy::NetcdfMetadataContractError);
    REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                            path("string_timestamp_array")),
                        Catch::Matchers::Contains("string metadata must be scalar"));
}

TEST_CASE("SHARPY NetCDF station and timestamp metadata enforce reviewed byte edges",
          "[data][soundings][sharpy][netcdf][metadata][bytes]")
{
    const auto path = [](const char* name)
    {
        return std::string("tests/init/fixtures/sharpy_netcdf_") + name + ".nc";
    };
    try
    {
        const auto empty_char = tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
            path("char_station_0"));
        REQUIRE(empty_char.station_id.empty());
    }
    catch (const std::runtime_error& error)
    {
        if (std::string(error.what()).find("Unable to load NetCDF shared library") != std::string::npos)
        {
            SUCCEED("NetCDF runtime unavailable; installed-package qualification owns availability");
            return;
        }
        throw;
    }
    REQUIRE(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                path("char_station_32")).station_id.size() == 32);
    REQUIRE(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                path("char_station_1023")).station_id.size() == 1023);
    REQUIRE(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                path("string_station_0")).station_id.empty());
    REQUIRE(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                path("string_station_1023")).station_id.size() == 1023);
    REQUIRE(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                path("char_timestamp_1023")).timestamp_utc.size() == 1023);
    REQUIRE(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                path("string_timestamp_1023")).timestamp_utc.size() == 1023);
    for (const char* name : {"char_station_1024", "char_timestamp_1024"})
    {
        REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(path(name)),
                            Catch::Matchers::Contains("character metadata exceeds the 1023-byte limit"));
    }
    for (const char* name : {"string_station_1024", "string_timestamp_1024"})
    {
        REQUIRE_THROWS_WITH(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(path(name)),
                            Catch::Matchers::Contains("string metadata exceeds the 1023-byte limit"));
    }
    REQUIRE_THROWS_AS(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                          path("char_station_1024")),
                      tmv::soundings::sharpy::NetcdfMetadataContractError);
    REQUIRE_THROWS_AS(tmv::soundings::sharpy::parse_netcdf_dynamic_profile_native(
                          path("string_station_1024")),
                      tmv::soundings::sharpy::NetcdfMetadataContractError);
}

TEST_CASE("Sounding descriptors contain only executable file readers",
          "[data][soundings][descriptor]")
{
    const auto& descriptors = get_sounding_scheme_descriptors();
    REQUIRE(descriptors.size() == 3);

    const auto* cm1 = tmv::find_scheme_descriptor(descriptors, "input_sounding");
    REQUIRE(cm1 != nullptr);
    REQUIRE(cm1->id == "cm1");
    REQUIRE(cm1->maturity == tmv::SchemeMaturity::Qualified);

    const auto* sharpy = tmv::find_scheme_descriptor(descriptors, "sharpy");
    REQUIRE(sharpy != nullptr);
    REQUIRE(sharpy->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(sharpy->capabilities.declared);
    REQUIRE_FALSE(sharpy->capabilities.produced_state_fields.empty());

    const auto* none = tmv::find_scheme_descriptor(descriptors, "disabled");
    REQUIRE(none == nullptr);
    REQUIRE_THROWS_WITH(
        create_sounding_scheme("OFF"),
        Catch::Matchers::Contains("Unknown sounding scheme"));
    REQUIRE(create_sounding_scheme("none") == nullptr);
}

TEST_CASE("Sounding linear interpolation at exact levels returns source values", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
    data.pressure_hpa = {1013.25, 900.0, 800.0, 700.0, 600.0, 500.0};
    data.temperature_k = {288.0, 281.5, 275.0, 269.0, 262.0, 255.0};

    SoundingConfig cfg;
    std::vector<double> targets = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};

    auto result = interpolate_sounding_linear(data, targets, cfg);

    REQUIRE(result.height_m.size() == 6);
    for (size_t k = 0; k < 6; ++k)
    {
        REQUIRE(result.temperature_k[k] == Approx(data.temperature_k[k]).margin(1e-6));
        REQUIRE(result.pressure_hpa[k] == Approx(data.pressure_hpa[k]).margin(1e-6));
    }
}

TEST_CASE("Sounding linear interpolation at midpoints yields arithmetic mean", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 2000.0, 4000.0, 6000.0, 8000.0, 10000.0};
    data.pressure_hpa = {1000.0, 800.0, 600.0, 400.0, 300.0, 200.0};
    data.temperature_k = {300.0, 280.0, 260.0, 240.0, 220.0, 200.0};

    SoundingConfig cfg;
    std::vector<double> targets = {1000.0};  // midpoint of first interval

    auto result = interpolate_sounding_linear(data, targets, cfg);

    // Linear midpoint: (300 + 280) / 2 = 290
    REQUIRE(result.temperature_k.size() == 1);
    REQUIRE(result.temperature_k[0] == Approx(290.0).margin(0.5));

    // Pressure midpoint: (1000 + 800) / 2 = 900
    REQUIRE(result.pressure_hpa[0] == Approx(900.0).margin(0.5));
}

TEST_CASE("Sounding QC filters NaN temperature levels", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
    data.pressure_hpa = {1013.0, 900.0, 800.0, 700.0, 600.0, 500.0};
    data.temperature_k = {288.0, std::numeric_limits<double>::quiet_NaN(),
                          275.0, 269.0, 262.0, 255.0};

    SoundingConfig cfg;
    bool valid = quality_control_sounding(data, cfg);

    if (valid)
    {
        // All remaining values must be finite
        for (double t : data.temperature_k)
            REQUIRE(std::isfinite(t));
        for (double p : data.pressure_hpa)
            REQUIRE(std::isfinite(p));

        // The NaN level should be removed
        REQUIRE(data.height_m.size() == 5);
    }
}

TEST_CASE("Sounding QC preserves clean data", "[data][soundings][analytical]")
{
    SoundingData data;
    data.height_m = {0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0};
    data.pressure_hpa = {1013.0, 900.0, 800.0, 700.0, 600.0, 500.0};
    data.temperature_k = {288.0, 281.5, 275.0, 269.0, 262.0, 255.0};

    SoundingConfig cfg;
    bool valid = quality_control_sounding(data, cfg);

    REQUIRE(valid);
    REQUIRE(data.height_m.size() == 6);
    REQUIRE(data.temperature_k[0] == Approx(288.0).margin(1e-6));
    REQUIRE(data.temperature_k[5] == Approx(255.0).margin(1e-6));
}

TEST_CASE("Native CM1 reader accepts all supplied real input_soundings",
          "[data][soundings][cm1][reference]")
{
    const std::array<std::string, 4> paths = {
        "tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE",
        "tests/fixtures/soundings/cm1/HIGHCAPENEW_SEMICIRCLE_02",
        "tests/fixtures/soundings/cm1/MEDCAPENEW_QUARTERCIRCLE",
        "tests/fixtures/soundings/cm1/MEDCAPENEW_SEMICIRCLE_06",
    };

    for (const auto& path : paths)
    {
        SoundingConfig cfg;
        cfg.scheme_id = "cm1";
        cfg.file_path = path;
        cfg.use_fallback_profiles = false;

        auto reader = create_sounding_scheme(cfg.scheme_id);
        REQUIRE(reader != nullptr);
        REQUIRE_NOTHROW(reader->initialize(cfg));
        const SoundingData data = reader->load_sounding(path);

        REQUIRE(data.is_valid());
        REQUIRE(data.num_levels() >= 90);
        REQUIRE(data.potential_temperature_is_primary);
        REQUIRE(data.potential_temperature_k.size() == data.num_levels());
        REQUIRE(data.mixing_ratio_kgkg.size() == data.num_levels());
        REQUIRE(data.wind_u_ms.size() == data.num_levels());
        REQUIRE(data.wind_v_ms.size() == data.num_levels());
        REQUIRE(data.height_m.front() == Approx(0.0));
        REQUIRE(data.height_m.back() >= 19000.0);

        for (std::size_t k = 0; k < data.num_levels(); ++k)
        {
            REQUIRE(std::isfinite(data.temperature_k[k]));
            REQUIRE(data.temperature_k[k] > 0.0);
            REQUIRE(data.mixing_ratio_kgkg[k] >= 0.0);
            if (k > 0)
            {
                REQUIRE(data.height_m[k] > data.height_m[k - 1]);
                REQUIRE(data.pressure_hpa[k] < data.pressure_hpa[k - 1]);
            }
            const double theta_round_trip =
                data.temperature_k[k]
                * std::pow(p0 / (data.pressure_hpa[k] * 100.0), R_d / cp);
            REQUIRE(theta_round_trip
                    == Approx(data.potential_temperature_k[k]).epsilon(1.0e-12));
        }
    }
}

TEST_CASE("Automatic sounding reader detects extensionless native CM1",
          "[data][soundings][cm1][auto]")
{
    SoundingConfig cfg;
    cfg.scheme_id = "auto";
    cfg.file_path = "tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE";
    cfg.use_fallback_profiles = false;

    auto reader = create_sounding_scheme(cfg.scheme_id);
    REQUIRE(reader != nullptr);
    REQUIRE_NOTHROW(reader->initialize(cfg));
    const auto data = reader->load_sounding(cfg.file_path);
    REQUIRE(data.station_id == "CM1_INPUT_SOUNDING");
    REQUIRE(data.potential_temperature_is_primary);
    REQUIRE(data.mixing_ratio_kgkg.front() == Approx(0.0147394));
    REQUIRE(data.wind_u_ms.front() == Approx(-15.798));
    REQUIRE(data.wind_v_ms.front() == Approx(0.0));
}

TEST_CASE("Sounding interpolation preserves direct moisture and native theta",
          "[data][soundings][cm1][interpolation]")
{
    SoundingConfig cfg;
    cfg.scheme_id = "cm1";
    cfg.file_path = "tests/fixtures/soundings/cm1/HIGHCAPENEW_QUARTERCIRCLE";
    auto reader = create_sounding_scheme(cfg.scheme_id);
    reader->initialize(cfg);
    const auto source = reader->load_sounding(cfg.file_path);
    const auto result = reader->interpolate_to_heights(source, {25.0});

    REQUIRE(result.potential_temperature_is_primary);
    REQUIRE(result.potential_temperature_k[0]
            == Approx(0.5 * (306.465 + 306.494)).margin(1.0e-12));
    REQUIRE(result.mixing_ratio_kgkg[0]
            == Approx(0.5 * (14.7394 + 14.710) / 1000.0).margin(1.0e-12));
    REQUIRE(result.wind_u_ms[0]
            == Approx(0.5 * (-15.798 - 15.388)).margin(1.0e-12));
    REQUIRE(result.wind_v_ms[0]
            == Approx(0.5 * (0.0 + 0.848)).margin(1.0e-12));
}

TEST_CASE("Wind interpolation crosses north in Cartesian component space",
          "[data][soundings][wind][regression]")
{
    SoundingData source;
    source.height_m = {0.0, 1000.0};
    source.pressure_hpa = {1000.0, 900.0};
    source.temperature_k = {290.0, 284.0};
    source.wind_speed_ms = {10.0, 10.0};
    source.wind_direction_deg = {359.0, 1.0};

    SoundingConfig cfg;
    const auto result = interpolate_sounding_linear(source, {500.0}, cfg);
    REQUIRE(result.wind_speed_ms[0] > 9.99);
    REQUIRE((result.wind_direction_deg[0] < 1.0
             || result.wind_direction_deg[0] > 359.0));
    REQUIRE(result.wind_v_ms[0] < -9.99);
    REQUIRE(std::abs(result.wind_u_ms[0]) < 1.0e-12);
}

TEST_CASE("SHARPY NetCDF metadata inquiry absence is distinct from declared read failures",
          "[data][soundings][sharpy][netcdf-metadata-contract]")
{
    using namespace tmv::soundings::sharpy;
    REQUIRE_FALSE(netcdf_metadata_attribute_available(-43, "latitude_deg", "attribute absent"));
    REQUIRE(netcdf_metadata_attribute_available(0, "latitude", "success"));
    for (const int status : {-33, -101, 1})
    {
        bool available = true;
        REQUIRE_THROWS_AS(available = netcdf_metadata_attribute_available(status, "latitude_deg", "injected"),
                          NetcdfMetadataContractError);
        REQUIRE(available);
        REQUIRE_THROWS_WITH(netcdf_metadata_attribute_available(status, "latitude_deg", "injected"),
                            Catch::Matchers::Contains("Cannot inspect NetCDF metadata 'latitude_deg'"));
    }
    REQUIRE_NOTHROW(require_netcdf_metadata_read(0, "station_id", "success"));
    for (const int status : {-43, -33, -101, 1})
    {
        REQUIRE_THROWS_AS(require_netcdf_metadata_read(status, "station_id", "injected"),
                          NetcdfMetadataContractError);
        REQUIRE_THROWS_WITH(require_netcdf_metadata_read(status, "station_id", "injected"),
                            Catch::Matchers::Contains("Cannot read NetCDF metadata 'station_id'"));
    }
    for (const int type : {1, 3, 4, 5, 6, 7, 8, 9, 10, 11})
    {
        REQUIRE(netcdf_primitive_numeric_type(type));
    }
    for (const int type : {0, 2, 12, 13, 32, std::numeric_limits<int>::max()})
    {
        REQUIRE_FALSE(netcdf_primitive_numeric_type(type));
    }
}

TEST_CASE("SHARPY NetCDF declared metadata types cannot silently default or alias around malformed values",
          "[data][soundings][sharpy][netcdf-metadata-contract]")
{
    using namespace tmv::soundings::sharpy;
    const auto path = [](const char* name)
    {
        return std::string("tests/init/fixtures/sharpy_netcdf_metadata_types_") + name + ".nc";
    };
    const auto reference = parse_netcdf_dynamic_profile_native(path("absent"));
    REQUIRE(reference.rows.size() == 33);
    REQUIRE(reference.station_id.empty());
    REQUIRE(reference.timestamp_utc.empty());
    const auto character = parse_netcdf_dynamic_profile_native(path("station_char"));
    REQUIRE(character.rows == reference.rows);
    REQUIRE(character.station_id == std::string(1023, 'S'));
    const auto string = parse_netcdf_dynamic_profile_native(path("station_string"));
    REQUIRE(string.rows == reference.rows);
    REQUIRE(string.station_id == "METADATA_TYPES_VALID");
    const auto aliases = parse_netcdf_dynamic_profile_native(path("geo_alias"));
    REQUIRE(aliases.rows == reference.rows);
    REQUIRE(aliases.latitude_deg == 35.25);
    REQUIRE(aliases.longitude_deg == -97);
    REQUIRE(aliases.elevation_m == 324);
    for (const char* name : {"station_int", "timestamp_double", "latitude_char", "longitude_string",
                            "elevation_empty", "invalid_primary_alias"})
    {
        INFO(name);
        REQUIRE(std::filesystem::file_size(path(name)) < 65536);
        REQUIRE_THROWS_AS(parse_netcdf_dynamic_profile_native(path(name)), NetcdfMetadataContractError);
        REQUIRE_THROWS_AS(parse_netcdf_dynamic_profile_native(path(name)), NetcdfInputContractError);
    }
}

TEST_CASE("SHARPY streaming NetCDF selection retains six winners and never inspects unused shapes",
          "[data][soundings][sharpy][netcdf-selected-enumeration]")
{
    using namespace tmv::soundings::sharpy;
    const auto reference = parse_netcdf_dynamic_profile_native("tests/init/fixtures/sharpy_netcdf_public_safe.nc");
    const std::array<const char*, 4> names{"dense", "priority", "depth_first", "normalized"};
    const std::array<std::size_t, 4> counts{262, 12, 18, 6};
    const std::array<std::string, 6> fields{"height_m", "pressure_hpa", "temperature_k", "dewpoint_k",
        "wind_speed_ms", "wind_direction_deg"};
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        const std::string path = std::string("tests/init/fixtures/sharpy_netcdf_selected_") + names[index] + ".nc";
        REQUIRE(std::filesystem::file_size(path) < 2 * 1024 * 1024);
        NetcdfEnumerationStatistics statistics;
        const auto selected = parse_netcdf_dynamic_profile_with_statistics_for_test(path, statistics);
        INFO(names[index]);
        REQUIRE(selected.rows == reference.rows);
        REQUIRE(selected.station_id == "STREAMING_NATIVE");
        REQUIRE(statistics.variable_metadata_queries == counts[index]);
        REQUIRE(statistics.visited_groups == (index == 2 ? 4 : 1));
        REQUIRE(statistics.peak_selected_descriptors == 6);
        REQUIRE(statistics.dimension_id_queries == 6);
        REQUIRE(statistics.dimension_length_queries == 6);
        REQUIRE(statistics.numeric_payload_reads == 6);
        for (std::size_t field = 0; field < fields.size(); ++field)
        {
            if (index == 2)
            {
                REQUIRE(statistics.selected_variable_names[field] == "first/deep/" + fields[field]);
            }
            else if (index < 2)
            {
                REQUIRE(statistics.selected_variable_names[field] == fields[field]);
            }
            else
            {
                REQUIRE(statistics.selected_variable_names[field].rfind("Sensor_", 0) == 0);
            }
        }
        REQUIRE(parse_netcdf_dynamic_profile_native(path).rows == selected.rows);
    }
}

TEST_CASE("SHARPY deferred NetCDF selection keeps selected profile limits and diagnostic publication atomic",
          "[data][soundings][sharpy][netcdf-selected-enumeration]")
{
    using namespace tmv::soundings::sharpy;
    const std::string path = "tests/init/fixtures/sharpy_netcdf_selected_selected_oversized.nc";
    REQUIRE(std::filesystem::file_size(path) < 65536);
    NetcdfEnumerationStatistics statistics;
    statistics.variable_metadata_queries = 777;
    statistics.selected_variable_names[0] = "preserved";
    REQUIRE_THROWS_AS(parse_netcdf_dynamic_profile_with_statistics_for_test(path, statistics),
                      NetcdfProfileResourceError);
    REQUIRE(statistics.variable_metadata_queries == 777);
    REQUIRE(statistics.selected_variable_names[0] == "preserved");
    REQUIRE(statistics.numeric_payload_reads == 0);
    REQUIRE_THROWS_WITH(parse_netcdf_dynamic_profile_native(path),
                        Catch::Matchers::Contains("100,000-level safety limit"));
    NetcdfEnumerationStatistics recovered;
    REQUIRE(parse_netcdf_dynamic_profile_with_statistics_for_test(
        "tests/init/fixtures/sharpy_netcdf_selected_dense.nc", recovered).rows.size() == 33);
    REQUIRE(recovered.dimension_length_queries == 6);
}

TEST_CASE("SHARPY iterative NetCDF DFS preserves deep and wide alias order with one group path",
          "[data][soundings][sharpy][netcdf-iterative-groups]")
{
    using namespace tmv::soundings::sharpy;
    const auto reference = parse_netcdf_dynamic_profile_native("tests/init/fixtures/sharpy_netcdf_public_safe.nc");
    const std::array<const char*, 4> names{"deep", "priority", "wide", "classic"};
    const std::array<std::size_t, 4> groups{98, 98, 513, 1};
    const std::array<std::size_t, 4> metadata{18, 12, 522, 6};
    const std::array<std::size_t, 4> depth{97, 97, 2, 1};
    const std::array<std::size_t, 4> paths{479, 479, 4, 0};
    const std::array<std::size_t, 4> children{97, 97, 512, 0};
    std::string deep_path;
    for (int index = 0; index < 96; ++index)
    {
        if (!deep_path.empty())
        {
            deep_path += '/';
        }
        deep_path += "n" + std::string(index < 10 ? "00" : "0") + std::to_string(index);
    }
    const std::array<std::string, 6> fields{"height_m", "pressure_hpa", "temperature_k", "dewpoint_k",
        "wind_speed_ms", "wind_direction_deg"};
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        const std::string path = std::string("tests/init/fixtures/sharpy_netcdf_groups_") + names[index] + ".nc";
        REQUIRE(std::filesystem::file_size(path) < 2 * 1024 * 1024);
        NetcdfEnumerationStatistics statistics;
        const auto actual = parse_netcdf_dynamic_profile_with_statistics_for_test(path, statistics);
        INFO(names[index]);
        REQUIRE(actual.rows == reference.rows);
        REQUIRE(actual.station_id == "GROUPS");
        REQUIRE(statistics.visited_groups == groups[index]);
        REQUIRE(statistics.variable_metadata_queries == metadata[index]);
        REQUIRE(statistics.peak_selected_descriptors == 6);
        REQUIRE(statistics.peak_traversal_depth == depth[index]);
        REQUIRE(statistics.peak_group_path_bytes == paths[index]);
        REQUIRE(statistics.peak_child_id_slots == children[index]);
        REQUIRE(statistics.group_count_queries == groups[index]);
        REQUIRE(statistics.group_id_queries == (index < 2 ? 96 : index == 2 ? 1 : 0));
        REQUIRE(statistics.group_name_queries == groups[index] - 1);
        REQUIRE(statistics.dimension_id_queries == 6);
        REQUIRE(statistics.dimension_length_queries == 6);
        REQUIRE(statistics.numeric_payload_reads == 6);
        for (std::size_t field = 0; field < fields.size(); ++field)
        {
            const std::string prefix = index == 0 ? deep_path + "/"
                : index == 1 ? "later/" : index == 2 ? "g000/" : "";
            REQUIRE(statistics.selected_variable_names[field] == prefix + fields[field]);
        }
        REQUIRE(parse_netcdf_dynamic_profile_native(path).rows == actual.rows);
    }
}

TEST_CASE("SHARPY NetCDF early and late group query errors reject before publishing extraction or statistics",
          "[data][soundings][sharpy][netcdf-iterative-groups]")
{
    using namespace tmv::soundings::sharpy;
    const std::string path = "tests/init/fixtures/sharpy_netcdf_groups_deep.nc";
    const auto reference = parse_netcdf_dynamic_profile_native(path);
    const std::array<NetcdfGroupQueryForTest, 3> queries{
        NetcdfGroupQueryForTest::Count, NetcdfGroupQueryForTest::Ids, NetcdfGroupQueryForTest::Name};
    for (std::size_t query = 0; query < queries.size(); ++query)
    {
        for (std::size_t ordinal : {std::size_t{1}, std::array<std::size_t, 3>{98, 96, 97}[query]})
        {
            for (int status : {-33, -61, -101})
            {
                NetcdfEnumerationStatistics statistics;
                statistics.visited_groups = 777;
                statistics.variable_metadata_queries = 888;
                statistics.group_count_queries = 999;
                statistics.peak_group_path_bytes = 123;
                statistics.peak_child_id_slots = 42;
                statistics.selected_variable_names[0] = "preserved";
                auto extraction = reference;
                NetcdfGroupQueryOverrideForTest override;
                override.query = queries[query];
                override.ordinal = ordinal;
                override.status = status;
                INFO("query " << query << ", ordinal " << ordinal << ", status " << status);
                REQUIRE_THROWS_AS(extraction = parse_netcdf_dynamic_profile_with_group_override_for_test(
                    path, statistics, override), NetcdfInputContractError);
                REQUIRE(extraction.rows == reference.rows);
                REQUIRE(extraction.station_id == reference.station_id);
                REQUIRE(statistics.visited_groups == 777);
                REQUIRE(statistics.variable_metadata_queries == 888);
                REQUIRE(statistics.group_count_queries == 999);
                REQUIRE(statistics.peak_group_path_bytes == 123);
                REQUIRE(statistics.peak_child_id_slots == 42);
                REQUIRE(statistics.selected_variable_names[0] == "preserved");
                REQUIRE(statistics.numeric_payload_reads == 0);
                REQUIRE(parse_netcdf_dynamic_profile_with_statistics_for_test(path, statistics).rows == reference.rows);
                REQUIRE(statistics.visited_groups == 98);
                REQUIRE(statistics.numeric_payload_reads == 6);
            }
        }
    }
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        NetcdfGroupQueryOverrideForTest override;
        override.query = mutation == 0 ? NetcdfGroupQueryForTest::Count : NetcdfGroupQueryForTest::Ids;
        override.returned_count = mutation == 0 ? -1 : mutation == 1 ? 0 : 3;
        NetcdfEnumerationStatistics statistics;
        statistics.visited_groups = 777;
        REQUIRE_THROWS_AS(parse_netcdf_dynamic_profile_with_group_override_for_test(path, statistics, override),
            NetcdfInputContractError);
        REQUIRE(statistics.visited_groups == 777);
        REQUIRE(statistics.numeric_payload_reads == 0);
    }
}

TEST_CASE("SHARPY iterative NetCDF keeps classic and absent optional group symbol compatibility",
          "[data][soundings][sharpy][netcdf-iterative-groups]")
{
    using namespace tmv::soundings::sharpy;
    REQUIRE(netcdf_group_enumeration_available(0, "success"));
    REQUIRE_FALSE(netcdf_group_enumeration_available(-111, "classic format"));
    for (int status : {-33, -61, -101})
    {
        REQUIRE_THROWS_AS(netcdf_group_enumeration_available(status, "injected"), NetcdfInputContractError);
    }
    REQUIRE_NOTHROW(require_netcdf_group_read(0, "IDs", "success"));
    for (int status : {-33, -61, -101, -111})
    {
        REQUIRE_THROWS_AS(require_netcdf_group_read(status, "group names", "injected"), NetcdfInputContractError);
    }
    const std::string classic = "tests/init/fixtures/sharpy_netcdf_groups_classic.nc";
    const auto reference = parse_netcdf_dynamic_profile_native(classic);
    for (int mutation = 0; mutation < 3; ++mutation)
    {
        NetcdfGroupQueryOverrideForTest override;
        override.omit_group_inquiry_symbol = mutation == 0;
        override.omit_group_name_symbol = mutation == 1;
        override.status = mutation == 2 ? -111 : 0;
        NetcdfEnumerationStatistics statistics;
        const auto actual = parse_netcdf_dynamic_profile_with_group_override_for_test(classic, statistics, override);
        REQUIRE(actual.rows == reference.rows);
        REQUIRE(statistics.visited_groups == 1);
        REQUIRE(statistics.group_count_queries == (mutation == 2 ? 1 : 0));
        REQUIRE(statistics.group_id_queries == 0);
        REQUIRE(statistics.group_name_queries == 0);
        REQUIRE(statistics.numeric_payload_reads == 6);
    }
    for (bool omit_inquiry : {false, true})
    {
        NetcdfGroupQueryOverrideForTest override;
        override.omit_group_inquiry_symbol = omit_inquiry;
        override.omit_group_name_symbol = !omit_inquiry;
        NetcdfEnumerationStatistics statistics;
        const auto actual = parse_netcdf_dynamic_profile_with_group_override_for_test(
            "tests/init/fixtures/sharpy_netcdf_groups_deep.nc", statistics, override);
        REQUIRE(actual.rows.size() == 33);
        REQUIRE(statistics.visited_groups == 1);
        REQUIRE(statistics.selected_variable_names[0] == "z");
        REQUIRE(statistics.group_count_queries == 0);
        REQUIRE(statistics.group_id_queries == 0);
        REQUIRE(statistics.group_name_queries == 0);
    }
}

TEST_CASE("SHARPY scalar NetCDF strings inspect at most the existing limit plus terminator",
          "[data][soundings][sharpy][netcdf-bounded-string-scan]")
{
    using namespace tmv::soundings::sharpy;
    for (std::size_t length : {std::size_t{0}, std::size_t{1}, std::size_t{1022}, std::size_t{1023}})
    {
        std::string value(length, 'S');
        std::size_t scanned = 0;
        std::size_t released = 0;
        const auto free = [&](char* pointer)
        {
            REQUIRE(pointer == value.data());
            ++released;
        };
        REQUIRE(consume_netcdf_scalar_metadata_string(value.data(), 0, "station_id",
            []
            {
                return std::string("success");
            }, free, &scanned) == value);
        REQUIRE(scanned == length + 1);
        REQUIRE(released == 1);
    }
    for (std::size_t length : {std::size_t{1024}, std::size_t{65536}})
    {
        std::string value(length, 'S');
        std::size_t scanned = 0;
        std::size_t released = 0;
        const auto free = [&](char* pointer)
        {
            REQUIRE(pointer == value.data());
            ++released;
        };
        REQUIRE_THROWS_AS(consume_netcdf_scalar_metadata_string(value.data(), 0, "station_id",
            []
            {
                return std::string("success");
            }, free, &scanned),
            NetcdfMetadataContractError);
        REQUIRE(scanned == 1024);
        REQUIRE(released == 1);
    }
    std::array<char, 1024> bounded;
    bounded.fill('S');
    std::size_t scanned = 0;
    REQUIRE_THROWS_AS(netcdf_bounded_metadata_string_length(bounded.data(), "station_id", &scanned),
        NetcdfMetadataContractError);
    REQUIRE(scanned == bounded.size());
}

TEST_CASE("SHARPY owned scalar strings release buffers on read failure and preserve null compatibility",
          "[data][soundings][sharpy][netcdf-bounded-string-scan]")
{
    using namespace tmv::soundings::sharpy;
    for (int status : {0, -33, -61})
    {
        std::size_t scanned = 0;
        std::size_t released = 0;
        const auto free = [&](char*)
        {
            ++released;
        };
        if (status == 0)
        {
            REQUIRE(consume_netcdf_scalar_metadata_string(nullptr, status, "station_id",
                []
                {
                    return std::string("injected");
                }, free, &scanned).empty());
        }
        else
        {
            REQUIRE_THROWS_AS(consume_netcdf_scalar_metadata_string(nullptr, status, "station_id",
                []
                {
                    return std::string("injected");
                }, free, &scanned),
                NetcdfMetadataContractError);
        }
        REQUIRE(scanned == 0);
        REQUIRE(released == 0);
    }
    for (int failed_field : {0, 1})
    {
        for (int status : {-33, -61})
        {
            std::array<std::string, 2> values{"STATION", "TIMESTAMP"};
            std::size_t released = 0;
            for (int field = 0; field <= failed_field; ++field)
            {
                std::size_t scanned = 0;
                const auto free = [&](char* pointer)
                {
                    REQUIRE(pointer == values[field].data());
                    ++released;
                };
                if (field == failed_field)
                {
                    REQUIRE_THROWS_AS(consume_netcdf_scalar_metadata_string(values[field].data(), status,
                        field == 0 ? "station_id" : "timestamp_utc",
                        []
                        {
                            return std::string("injected");
                        }, free, &scanned),
                        NetcdfMetadataContractError);
                    REQUIRE(scanned == 0);
                }
                else
                {
                    REQUIRE(consume_netcdf_scalar_metadata_string(values[field].data(), 0,
                        "station_id",
                        []
                        {
                            return std::string("success");
                        }, free, &scanned) == values[field]);
                    REQUIRE(scanned == values[field].size() + 1);
                }
            }
            REQUIRE(released == static_cast<std::size_t>(failed_field + 1));
        }
    }
}

TEST_CASE("SHARPY real NetCDF empty and boundary scalar metadata retain exact bytes and bounded scans",
          "[data][soundings][sharpy][netcdf-bounded-string-scan]")
{
    using namespace tmv::soundings::sharpy;
    const auto reference = parse_netcdf_dynamic_profile_native("tests/init/fixtures/sharpy_netcdf_public_safe.nc");
    for (const char* name : {"empty", "limit"})
    {
        const std::string path = std::string("tests/init/fixtures/sharpy_netcdf_bounded_string_") + name + ".nc";
        REQUIRE(std::filesystem::file_size(path) <= 256 * 1024);
        NetcdfEnumerationStatistics statistics;
        const auto actual = parse_netcdf_dynamic_profile_with_statistics_for_test(path, statistics);
        REQUIRE(actual.rows == reference.rows);
        const std::size_t length = std::string(name) == "limit" ? 1023 : 0;
        REQUIRE(actual.station_id == std::string(length, 'S'));
        REQUIRE(actual.timestamp_utc == std::string(length, 'T'));
        REQUIRE(statistics.numeric_payload_reads == 6);
        if (length > 0)
        {
            REQUIRE(statistics.metadata_string_bytes_scanned == 2048);
        }
        else
        {
            REQUIRE(statistics.metadata_string_bytes_scanned <= 2);
        }
        REQUIRE(parse_netcdf_dynamic_profile_native(path).rows == actual.rows);
    }
}

TEST_CASE("SHARPY early and late overlength scalar strings leave extraction and statistics unpublished",
          "[data][soundings][sharpy][netcdf-bounded-string-scan]")
{
    using namespace tmv::soundings::sharpy;
    const auto reference = parse_netcdf_dynamic_profile_native("tests/init/fixtures/sharpy_netcdf_public_safe.nc");
    for (const char* name : {"station_1024", "station_65536", "timestamp_65536"})
    {
        const std::string path = std::string("tests/init/fixtures/sharpy_netcdf_bounded_string_") + name + ".nc";
        REQUIRE(std::filesystem::file_size(path) <= 256 * 1024);
        NetcdfEnumerationStatistics statistics;
        statistics.visited_groups = 777;
        statistics.variable_metadata_queries = 888;
        statistics.metadata_string_bytes_scanned = 42;
        statistics.selected_variable_names[0] = "preserved";
        auto extraction = reference;
        REQUIRE_THROWS_AS(extraction = parse_netcdf_dynamic_profile_with_statistics_for_test(path, statistics),
            NetcdfMetadataContractError);
        REQUIRE(extraction.rows == reference.rows);
        REQUIRE(extraction.station_id == reference.station_id);
        REQUIRE(extraction.timestamp_utc == reference.timestamp_utc);
        REQUIRE(statistics.visited_groups == 777);
        REQUIRE(statistics.variable_metadata_queries == 888);
        REQUIRE(statistics.metadata_string_bytes_scanned == 42);
        REQUIRE(statistics.selected_variable_names[0] == "preserved");
        REQUIRE(statistics.numeric_payload_reads == 0);
        REQUIRE_THROWS_WITH(parse_netcdf_dynamic_profile_native(path),
            Catch::Matchers::Contains("string metadata exceeds the 1023-byte limit"));
        const auto recovered = parse_netcdf_dynamic_profile_with_statistics_for_test(
            "tests/init/fixtures/sharpy_netcdf_bounded_string_limit.nc", statistics);
        REQUIRE(recovered.rows == reference.rows);
        REQUIRE(recovered.station_id.size() == 1023);
        REQUIRE(statistics.metadata_string_bytes_scanned == 2048);
        REQUIRE(statistics.numeric_payload_reads == 6);
    }
}

TEST_CASE("SHARPY owns scalar buffers before throwing error detail and skips detail on success",
          "[data][soundings][sharpy][netcdf-bounded-string-scan][netcdf-owned-string-detail]")
{
    using namespace tmv::soundings::sharpy;
    for (int failure : {0, 1})
    {
        std::string value = "OWNED";
        std::string published = "preserved";
        const std::string name = "station_id";
        NetcdfEnumerationStatistics statistics;
        statistics.metadata_string_bytes_scanned = 42;
        statistics.selected_variable_names[0] = "preserved";
        std::size_t released = 0;
        std::size_t details = 0;
        const auto free = [&](char* pointer)
        {
            CHECK(pointer == value.data());
            ++released;
        };
        const auto detail = [&]() -> std::string
        {
            ++details;
            if (failure == 0)
            {
                throw std::bad_alloc();
            }
            throw std::runtime_error("injected error detail failure");
        };
        if (failure == 0)
        {
            REQUIRE_THROWS_AS(published = consume_netcdf_scalar_metadata_string(value.data(), -33, name,
                detail, free, &statistics.metadata_string_bytes_scanned), std::bad_alloc);
        }
        else
        {
            REQUIRE_THROWS_WITH(published = consume_netcdf_scalar_metadata_string(value.data(), -33, name,
                detail, free, &statistics.metadata_string_bytes_scanned),
                Catch::Matchers::Contains("injected error detail failure"));
        }
        REQUIRE(released == 1);
        REQUIRE(details == 1);
        REQUIRE(published == "preserved");
        REQUIRE(statistics.metadata_string_bytes_scanned == 42);
        REQUIRE(statistics.selected_variable_names[0] == "preserved");
        REQUIRE(statistics.numeric_payload_reads == 0);
        REQUIRE_NOTHROW(published = consume_netcdf_scalar_metadata_string(value.data(), 0, name,
            detail, free, &statistics.metadata_string_bytes_scanned));
        REQUIRE(published == value);
        REQUIRE(released == 2);
        REQUIRE(details == 1);
        REQUIRE(statistics.metadata_string_bytes_scanned == 42 + value.size() + 1);
    }
}
