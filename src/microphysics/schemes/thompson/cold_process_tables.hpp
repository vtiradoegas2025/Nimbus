#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace tmv::thompson
{

struct ColdProcessTableMetadata
{
    static constexpr std::uint32_t format_version = 1;
    static constexpr std::size_t water_contents = 37;
    static constexpr std::size_t rain_intercepts = 37;
    static constexpr std::size_t cloud_numbers = 100;
    static constexpr std::size_t temperatures = 45;
    static constexpr std::size_t ice_nuclei = 55;
    static constexpr std::size_t ice_contents = 64;
    static constexpr std::size_t ice_numbers = 55;
    static constexpr std::size_t particle_bins = 100;

    static constexpr const char* upstream_release = "WRF v4.7.1";
    static constexpr const char* upstream_commit = "f52c197";
    static constexpr const char* upstream_source_sha256 =
        "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802";
    static constexpr const char* constants_sha256 =
        "b8290733ccf631c8d2a1c7c1c2136445f20842fac476efae8b288a0aaba796c7";
    static constexpr const char* packaged_payload_sha256 =
        "dd8fa3d5dd9a67bfa7ad7c61ed415929c4f4550205110802d79266f588ab20e8";
    static constexpr const char* packaged_sha256 =
        "c4bdf82a41bc22cd4bd2f7c6c2c5f7f6d75d527941437ad57e00bbbb53a9df53";
    static constexpr const char* packaged_path =
        "data/thompson/cold_process_v1.ntcold";
};

struct RainFreezingCell
{
    double ice_mass = 0.0;
    double graupel_mass = 0.0;
    double ice_number = 0.0;
    double rain_number = 0.0;
};

struct CloudFreezingCell
{
    double ice_mass = 0.0;
    double ice_number = 0.0;
};

struct IceSnowCell
{
    double snow_mass = 0.0;
    double ice_number = 0.0;
    double ice_deposition_fraction = 0.0;
};

struct ColdProcessTables
{
    std::vector<RainFreezingCell> rain_freezing;
    std::vector<CloudFreezingCell> cloud_freezing;
    std::vector<IceSnowCell> ice_snow;
    std::string payload_sha256;
};

struct RainFreezingIndex
{
    std::size_t rain_content = 0;
    std::size_t rain_intercept = 0;
    std::size_t temperature = 0;
    std::size_t ice_nuclei = 0;
};

struct CloudFreezingIndex
{
    std::size_t cloud_content = 0;
    std::size_t cloud_number = 0;
    std::size_t temperature = 0;
    std::size_t ice_nuclei = 0;
};

struct IceSnowIndex
{
    std::size_t ice_content = 0;
    std::size_t ice_number = 0;
};

std::size_t checked_rain_freezing_size();
std::size_t checked_cloud_freezing_size();
std::size_t checked_ice_snow_size();
std::size_t checked_cold_payload_bytes();
std::size_t cold_resident_bytes(const ColdProcessTables& tables) noexcept;
std::size_t flatten(RainFreezingIndex index);
std::size_t flatten(CloudFreezingIndex index);
std::size_t flatten(IceSnowIndex index);
double freezing_water_content(std::size_t index);
double freezing_cloud_number(std::size_t index);
double ice_table_content(std::size_t index);
double ice_table_number(std::size_t index);

RainFreezingCell generate_rain_freezing_cell(RainFreezingIndex index);
CloudFreezingCell generate_cloud_freezing_cell(CloudFreezingIndex index);
CloudFreezingCell generate_cloud_freezing_cell(CloudFreezingIndex index, double cloud_content);
IceSnowCell generate_ice_snow_cell(IceSnowIndex index);
ColdProcessTables generate_cold_process_tables();

bool write_cold_process_tables(const std::filesystem::path& path, const ColdProcessTables& tables, std::string& error);
bool load_cold_process_tables(const std::filesystem::path& path,
                              ColdProcessTables&           tables,
                              std::string&                 error,
                              std::size_t maximum_resident_bytes = std::numeric_limits<std::size_t>::max());
bool validate_cold_process_tables(const ColdProcessTables& tables, std::string& error);

class ColdProcessTableStore
{
public:
    bool publish_file(const std::filesystem::path& path,
                      std::string&                 error,
                      std::size_t                  maximum_resident_bytes = std::numeric_limits<std::size_t>::max());
    bool publish_packaged_file(
        const std::filesystem::path& path, std::string& error,
        std::size_t maximum_resident_bytes =
            std::numeric_limits<std::size_t>::max());
    std::shared_ptr<const ColdProcessTables> snapshot() const noexcept;

private:
    std::shared_ptr<const ColdProcessTables> tables_;
};

} // namespace tmv::thompson
