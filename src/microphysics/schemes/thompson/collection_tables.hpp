#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace tmv::thompson
{

struct CollectionTableMetadata
{
    static constexpr std::uint32_t format_version = 1;
    static constexpr std::size_t bins = 100;
    static constexpr std::size_t rain_contents = 37;
    static constexpr std::size_t rain_intercepts = 37;
    static constexpr std::size_t snow_contents = 37;
    static constexpr std::size_t temperatures = 9;
    static constexpr std::size_t graupel_contents = 37;
    static constexpr std::size_t graupel_intercepts = 37;
    static constexpr std::size_t graupel_densities = 1;
    static constexpr std::uint64_t pair_evaluations_per_cell = bins * bins;

    static constexpr const char* upstream_release = "WRF v4.7.1";
    static constexpr const char* upstream_commit = "f52c197";
    static constexpr const char* upstream_module = "phys/module_mp_thompson.F";
    static constexpr const char* upstream_source_sha256 =
        "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802";
    static constexpr const char* upstream_license_sha256 =
        "b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6";
    static constexpr const char* constants_sha256 =
        "d7accde31ef645d72673bb963339e6be62689ddf4bb1292c0d164a53449ae77c";
    static constexpr const char* packaged_payload_sha256 =
        "0d5b7a28b56d1d90bedcf19d81b3d686451612fd4bc41d8eac19e23a74a4e43d";
    static constexpr const char* packaged_sha256 =
        "b01efcfe79208b820ea43d8c0d59b4557fb94706586c600dbd43b70e36e83547";
    static constexpr const char* packaged_path =
        "data/thompson/collection_v1.ntctbl";
};

struct RainGraupelCollection
{
    double graupel_mass_by_rain = 0.0;
    double rain_mass_by_rain = 0.0;
    double rain_mass_by_graupel = 0.0;
    double graupel_number_by_rain = 0.0;
    double rain_number_by_graupel = 0.0;
};

struct RainSnowCollection
{
    std::array<double, 12> values{};
};

struct CollectionTables
{
    std::vector<RainGraupelCollection> rain_graupel;
    std::vector<RainSnowCollection> rain_snow;
    std::string payload_sha256;
};

struct GeneratorMeasurement
{
    std::size_t sampled_cells = 0;
    std::uint64_t sampled_pair_evaluations = 0;
    double elapsed_seconds = 0.0;
    double projected_full_generation_seconds = 0.0;
    double checksum = 0.0;
};

struct RainGraupelIndex
{
    std::size_t graupel_intercept = 0;
    std::size_t graupel_content = 0;
    std::size_t graupel_density = 0;
    std::size_t rain_intercept = 0;
    std::size_t rain_content = 0;
};

struct RainSnowIndex
{
    std::size_t snow_content = 0;
    std::size_t temperature = 0;
    std::size_t rain_intercept = 0;
    std::size_t rain_content = 0;
};

std::size_t checked_rain_graupel_size();
std::size_t checked_rain_snow_size();
std::size_t checked_payload_bytes();
std::uint64_t checked_full_pair_evaluations();
std::size_t resident_bytes(const CollectionTables& tables) noexcept;
std::size_t flatten(RainGraupelIndex index);
std::size_t flatten(RainSnowIndex index);

RainGraupelCollection generate_rain_graupel_cell(RainGraupelIndex index);
RainSnowCollection generate_rain_snow_cell(RainSnowIndex index);

CollectionTables generate_collection_tables();
GeneratorMeasurement measure_generator_cost(std::size_t sampled_cells);
bool write_collection_tables(const std::filesystem::path& path,
                             const CollectionTables& tables,
                             std::string& error);
bool load_collection_tables(const std::filesystem::path& path,
                            CollectionTables& tables,
                            std::string& error,
                            std::size_t maximum_resident_bytes =
                                std::numeric_limits<std::size_t>::max());
bool validate_collection_tables(const CollectionTables& tables,
                                std::string& error);

class CollectionTableStore
{
public:
    bool publish_file(const std::filesystem::path& path, std::string& error,
                      std::size_t maximum_resident_bytes =
                          std::numeric_limits<std::size_t>::max());
    bool publish_packaged_file(
        const std::filesystem::path& path, std::string& error,
        std::size_t maximum_resident_bytes =
            std::numeric_limits<std::size_t>::max());
    std::shared_ptr<const CollectionTables> snapshot() const noexcept;

private:
    std::shared_ptr<const CollectionTables> tables_;
};

} // namespace tmv::thompson
