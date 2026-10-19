#pragma once

#include "sharpy_native_reader.hpp"
#include "soundings/sounding_input_limits.hpp"

#include <cstddef>
#include <array>
#include <optional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tmv::soundings::sharpy
{

struct NetcdfEnumerationStatistics
{
    std::size_t variable_metadata_queries = 0;
    std::size_t visited_groups = 0;
    std::size_t peak_selected_descriptors = 0;
    std::size_t dimension_id_queries = 0;
    std::size_t dimension_length_queries = 0;
    std::size_t numeric_payload_reads = 0;
    std::size_t group_count_queries = 0;
    std::size_t group_id_queries = 0;
    std::size_t group_name_queries = 0;
    std::size_t peak_traversal_depth = 0;
    std::size_t peak_group_path_bytes = 0;
    std::size_t peak_child_id_slots = 0;
    std::size_t metadata_string_bytes_scanned = 0;
    std::array<std::string, 6> selected_variable_names;
};

NativeProfileExtraction parse_netcdf_dynamic_profile_with_statistics_for_test(
    const std::string& file_path, NetcdfEnumerationStatistics& statistics);

enum class NetcdfGroupQueryForTest
{
    Count,
    Ids,
    Name
};

struct NetcdfGroupQueryOverrideForTest
{
    NetcdfGroupQueryForTest query = NetcdfGroupQueryForTest::Count;
    std::size_t ordinal = 1;
    int status = 0;
    std::optional<int> returned_count;
    bool omit_group_inquiry_symbol = false;
    bool omit_group_name_symbol = false;
};

NativeProfileExtraction parse_netcdf_dynamic_profile_with_group_override_for_test(
    const std::string& file_path, NetcdfEnumerationStatistics& statistics,
    const NetcdfGroupQueryOverrideForTest& override);

inline bool netcdf_group_enumeration_available(int status, const std::string& detail)
{
    constexpr int kNcNotNetcdf4 = -111;
    if (status == kNcNotNetcdf4)
    {
        return false;
    }
    if (status != 0)
    {
        throw NetcdfInputContractError("Cannot inspect NetCDF child groups: " + detail);
    }
    return true;
}

inline void require_netcdf_group_read(int status, const std::string& operation,
                                    const std::string& detail)
{
    if (status != 0)
    {
        throw NetcdfInputContractError("Cannot read NetCDF " + operation + ": " + detail);
    }
}

inline bool netcdf_metadata_attribute_available(int status, const std::string& name,
                                               const std::string& detail)
{
    constexpr int kNcAttributeNotFound = -43;
    if (status == kNcAttributeNotFound)
    {
        return false;
    }
    if (status != 0)
    {
        throw NetcdfMetadataContractError("Cannot inspect NetCDF metadata '" + name + "': " + detail);
    }
    return true;
}

inline void require_netcdf_metadata_read(int status, const std::string& name,
                                        const std::string& detail)
{
    if (status != 0)
    {
        throw NetcdfMetadataContractError("Cannot read NetCDF metadata '" + name + "': " + detail);
    }
}

inline std::size_t netcdf_bounded_metadata_string_length(const char* value,
                                                        const std::string& name,
                                                        std::size_t* scanned_bytes = nullptr)
{
    if (value == nullptr)
    {
        return 0;
    }
    for (std::size_t index = 0; index <= kMaxSoundingMetadataBytes; ++index)
    {
        if (scanned_bytes != nullptr)
        {
            ++*scanned_bytes;
        }
        if (value[index] == '\0')
        {
            return index;
        }
    }
    throw NetcdfMetadataContractError("NetCDF string metadata exceeds the 1023-byte limit for '" + name + "'");
}

template <class FreeString, class ErrorDetail>
std::string consume_netcdf_scalar_metadata_string(char* pointer, int status,
                                                const std::string& name, ErrorDetail detail,
                                                FreeString release,
                                                std::size_t* scanned_bytes = nullptr)
{
    std::unique_ptr<char, FreeString> owned(pointer, std::move(release));
    if (status != 0)
    {
        require_netcdf_metadata_read(status, name, detail());
    }
    const std::size_t bytes = netcdf_bounded_metadata_string_length(owned.get(), name, scanned_bytes);
    return owned ? std::string(owned.get(), bytes) : std::string{};
}

inline bool netcdf_primitive_numeric_type(int type) noexcept
{
    switch (type)
    {
        case 1:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
            return true;
        default:
            return false;
    }
}

bool        checked_multiply_size(std::size_t a, std::size_t b, std::size_t& out);
bool        ends_with(const std::string& value, const std::string& suffix);
std::string normalize_profile_name(const std::string& value);

const std::vector<std::string>& profile_height_candidates();
const std::vector<std::string>& profile_pressure_candidates();
const std::vector<std::string>& profile_temperature_candidates();
const std::vector<std::string>& profile_dewpoint_candidates();
const std::vector<std::string>& profile_wind_speed_candidates();
const std::vector<std::string>& profile_wind_direction_candidates();

std::string              normalize_profile_lookup_key(const std::string& value);
std::vector<std::string> expand_profile_candidates(const std::vector<std::string>& candidates);

NativeProfileExtraction build_native_profile_rows(const std::vector<double>& height,
                                                  const std::vector<double>& pressure,
                                                  const std::vector<double>& temperature,
                                                  const std::vector<double>& dewpoint,
                                                  const std::vector<double>& wind_speed,
                                                  const std::vector<double>& wind_direction);

} // namespace tmv::soundings::sharpy
