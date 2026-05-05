/**
 * @file sharpy_sounding.cpp
 * @brief Implementation for the soundings module.
 *
 * Provides executable logic for the soundings runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/soundings subsystem.
 */

#include "sharpy_sounding.hpp"
#include "soundings/base/soundings_base.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace
{

/**
 * @brief Converts an ASCII string to lowercase.
 */
std::string to_lower_ascii(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

/**
 * @brief Normalizes profile variable names for matching.
 */
std::string normalize_profile_name(const std::string& value)
{
    std::string normalized = to_lower_ascii(value);
    for (char& ch : normalized)
    {
        if (ch == '/')
        {
            ch = '_';
        }
    }
    return normalized;
}

/**
 * @brief Returns true when `value` ends with `suffix`.
 */
bool ends_with(const std::string& value, const std::string& suffix)
{
    if (suffix.size() > value.size())
    {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

/**
 * @brief Multiplies two `size_t` values with overflow checking.
 */
bool checked_multiply_size(std::size_t a, std::size_t b, std::size_t& out)
{
    if (a == 0 || b == 0)
    {
        out = 0;
        return true;
    }
    if (a > (std::numeric_limits<std::size_t>::max() / b))
    {
        return false;
    }
    out = a * b;
    return true;
}

/**
 * @brief Raw NetCDF attribute payload metadata and bytes.
 */
struct NetcdfAttribute
{
    int32_t type = 0;
    std::size_t element_count = 0;
    std::vector<std::uint8_t> raw;
};

/**
 * @brief NetCDF dimension descriptor.
 */
struct NetcdfDimension
{
    std::string name;
    std::uint64_t length = 0;
    bool unlimited = false;
};

/**
 * @brief NetCDF variable descriptor from classic header metadata.
 */
struct NetcdfVariable
{
    std::string name;
    std::vector<int32_t> dim_ids;
    std::map<std::string, NetcdfAttribute> attributes;
    int32_t type = 0;
    std::uint32_t vsize = 0;
    std::uint64_t begin = 0;
    bool record_var = false;
};

/**
 * @brief Parsed metadata for a NetCDF classic/64-bit-offset file.
 */
struct NetcdfClassicHeader
{
    int version = 0;
    std::uint64_t num_records = 0;
    std::vector<NetcdfDimension> dimensions;
    std::map<std::string, NetcdfAttribute> global_attributes;
    std::vector<NetcdfVariable> variables;
};

/**
 * @brief Native profile extraction payload and optional station metadata.
 */
struct NativeProfileExtraction
{
    std::vector<std::vector<double>> rows;
    std::string station_id;
    std::string timestamp_utc;
    double latitude_deg = std::numeric_limits<double>::quiet_NaN();
    double longitude_deg = std::numeric_limits<double>::quiet_NaN();
    double elevation_m = std::numeric_limits<double>::quiet_NaN();
};

constexpr int32_t kNcDimensionTag = 10;
constexpr int32_t kNcVariableTag = 11;
constexpr int32_t kNcAttributeTag = 12;

constexpr int32_t kNcByte = 1;
constexpr int32_t kNcChar = 2;
constexpr int32_t kNcShort = 3;
constexpr int32_t kNcInt = 4;
constexpr int32_t kNcFloat = 5;
constexpr int32_t kNcDouble = 6;
constexpr int32_t kNcUByte = 7;
constexpr int32_t kNcUShort = 8;
constexpr int32_t kNcUInt = 9;
constexpr int32_t kNcInt64 = 10;
constexpr int32_t kNcUInt64 = 11;

/**
 * @brief Reads one big-endian unsigned 32-bit integer.
 */
std::uint32_t read_be_u32(std::istream& in, const std::string& context)
{
    std::array<unsigned char, 4> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in)
    {
        throw std::runtime_error("Unexpected EOF while reading " + context);
    }
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

/**
 * @brief Reads one big-endian signed 32-bit integer.
 */
std::int32_t read_be_i32(std::istream& in, const std::string& context)
{
    return static_cast<std::int32_t>(read_be_u32(in, context));
}

/**
 * @brief Reads one big-endian unsigned 64-bit integer.
 */
std::uint64_t read_be_u64(std::istream& in, const std::string& context)
{
    std::array<unsigned char, 8> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in)
    {
        throw std::runtime_error("Unexpected EOF while reading " + context);
    }
    return (static_cast<std::uint64_t>(bytes[0]) << 56) |
           (static_cast<std::uint64_t>(bytes[1]) << 48) |
           (static_cast<std::uint64_t>(bytes[2]) << 40) |
           (static_cast<std::uint64_t>(bytes[3]) << 32) |
           (static_cast<std::uint64_t>(bytes[4]) << 24) |
           (static_cast<std::uint64_t>(bytes[5]) << 16) |
           (static_cast<std::uint64_t>(bytes[6]) << 8) |
           static_cast<std::uint64_t>(bytes[7]);
}

/**
 * @brief Skips byte padding and validates stream state.
 */
void skip_padding(std::istream& in, std::size_t byte_count, const std::string& context)
{
    if (byte_count == 0)
    {
        return;
    }
    in.seekg(static_cast<std::streamoff>(byte_count), std::ios::cur);
    if (!in)
    {
        throw std::runtime_error("Unexpected EOF while skipping padding in " + context);
    }
}

/**
 * @brief Returns padding bytes needed to align to 4-byte boundary.
 */
std::size_t padding_to_4(std::size_t byte_count)
{
    const std::size_t rem = byte_count % 4;
    return rem == 0 ? 0 : (4 - rem);
}

/**
 * @brief Returns byte width for a NetCDF classic numeric type id.
 */
std::size_t nc_type_size_bytes(int32_t type)
{
    switch (type)
    {
        case kNcByte:
        case kNcChar:
        case kNcUByte:
            return 1;
        case kNcShort:
        case kNcUShort:
            return 2;
        case kNcInt:
        case kNcFloat:
        case kNcUInt:
            return 4;
        case kNcDouble:
        case kNcInt64:
        case kNcUInt64:
            return 8;
        default:
            throw std::runtime_error("Unsupported NetCDF type id: " + std::to_string(type));
    }
}

/**
 * @brief Reads a length-prefixed NetCDF identifier and trailing padding.
 */
std::string read_netcdf_name(std::istream& in, const std::string& context)
{
    const std::uint32_t length = read_be_u32(in, context + " name length");
    if (length > (32u * 1024u * 1024u))
    {
        throw std::runtime_error("NetCDF name too long in " + context);
    }
    std::string value(length, '\0');
    if (length > 0)
    {
        in.read(value.data(), static_cast<std::streamsize>(length));
        if (!in)
        {
            throw std::runtime_error("Unexpected EOF while reading NetCDF name in " + context);
        }
    }
    skip_padding(in, padding_to_4(length), context + " name padding");
    return value;
}

/**
 * @brief Reads one NetCDF attribute list block.
 */
std::map<std::string, NetcdfAttribute> read_netcdf_attribute_list(std::istream& in, const std::string& context)
{
    const std::int32_t tag = read_be_i32(in, context + " tag");
    const std::int32_t count_raw = read_be_i32(in, context + " count");
    if (tag == 0 && count_raw == 0)
    {
        return {};
    }
    if (tag != kNcAttributeTag)
    {
        throw std::runtime_error("Malformed NetCDF attribute list in " + context);
    }
    if (count_raw < 0)
    {
        throw std::runtime_error("Negative NetCDF attribute count in " + context);
    }

    const std::size_t count = static_cast<std::size_t>(count_raw);
    std::map<std::string, NetcdfAttribute> attrs;
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::string name = read_netcdf_name(in, context + " attribute");
        const std::int32_t type = read_be_i32(in, context + " attribute type");
        const std::int32_t elements_raw = read_be_i32(in, context + " attribute size");
        if (elements_raw < 0)
        {
            throw std::runtime_error("Negative NetCDF attribute size for '" + name + "'");
        }

        const std::size_t element_count = static_cast<std::size_t>(elements_raw);
        const std::size_t type_size = nc_type_size_bytes(type);
        std::size_t byte_count = 0;
        if (!checked_multiply_size(element_count, type_size, byte_count))
        {
            throw std::runtime_error("NetCDF attribute is too large: '" + name + "'");
        }

        NetcdfAttribute attr;
        attr.type = type;
        attr.element_count = element_count;
        attr.raw.resize(byte_count);
        if (byte_count > 0)
        {
            in.read(reinterpret_cast<char*>(attr.raw.data()), static_cast<std::streamsize>(byte_count));
            if (!in)
            {
                throw std::runtime_error("Unexpected EOF while reading NetCDF attribute '" + name + "'");
            }
        }
        skip_padding(in, padding_to_4(byte_count), context + " attribute padding");
        attrs[name] = std::move(attr);
    }
    return attrs;
}

/**
 * @brief Reads the NetCDF dimension list from the file header.
 */
std::vector<NetcdfDimension> read_netcdf_dimension_list(std::istream& in)
{
    const std::int32_t tag = read_be_i32(in, "dimension list tag");
    const std::int32_t count_raw = read_be_i32(in, "dimension list count");
    if (tag == 0 && count_raw == 0)
    {
        return {};
    }
    if (tag != kNcDimensionTag)
    {
        throw std::runtime_error("Malformed NetCDF dimension list");
    }
    if (count_raw < 0)
    {
        throw std::runtime_error("Negative NetCDF dimension count");
    }

    const std::size_t count = static_cast<std::size_t>(count_raw);
    std::vector<NetcdfDimension> dims;
    dims.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        NetcdfDimension dim;
        dim.name = read_netcdf_name(in, "dimension");
        const std::int32_t length_raw = read_be_i32(in, "dimension length");
        if (length_raw < 0)
        {
            throw std::runtime_error("Negative NetCDF dimension length for '" + dim.name + "'");
        }
        dim.unlimited = (length_raw == 0);
        dim.length = dim.unlimited ? 0u : static_cast<std::uint64_t>(length_raw);
        dims.push_back(std::move(dim));
    }
    return dims;
}

/**
 * @brief Reads the NetCDF variable list from the file header.
 */
std::vector<NetcdfVariable> read_netcdf_variable_list(std::istream& in, int version)
{
    const std::int32_t tag = read_be_i32(in, "variable list tag");
    const std::int32_t count_raw = read_be_i32(in, "variable list count");
    if (tag == 0 && count_raw == 0)
    {
        return {};
    }
    if (tag != kNcVariableTag)
    {
        throw std::runtime_error("Malformed NetCDF variable list");
    }
    if (count_raw < 0)
    {
        throw std::runtime_error("Negative NetCDF variable count");
    }

    const std::size_t count = static_cast<std::size_t>(count_raw);
    std::vector<NetcdfVariable> vars;
    vars.reserve(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        NetcdfVariable var;
        var.name = read_netcdf_name(in, "variable");

        const std::int32_t dim_count_raw = read_be_i32(in, "variable dimension count");
        if (dim_count_raw < 0)
        {
            throw std::runtime_error("Negative NetCDF dimension count for variable '" + var.name + "'");
        }
        const std::size_t dim_count = static_cast<std::size_t>(dim_count_raw);
        var.dim_ids.reserve(dim_count);
        for (std::size_t d = 0; d < dim_count; ++d)
        {
            var.dim_ids.push_back(read_be_i32(in, "variable dimension id"));
        }

        var.attributes = read_netcdf_attribute_list(in, "variable '" + var.name + "' attributes");
        var.type = read_be_i32(in, "variable '" + var.name + "' type");
        var.vsize = read_be_u32(in, "variable '" + var.name + "' vsize");
        var.begin = (version == 2)
            ? read_be_u64(in, "variable '" + var.name + "' begin")
            : static_cast<std::uint64_t>(read_be_u32(in, "variable '" + var.name + "' begin"));

        vars.push_back(std::move(var));
    }

    return vars;
}

/**
 * @brief Parses a NetCDF classic/64-bit-offset header and metadata tables.
 */
NetcdfClassicHeader parse_netcdf_classic_header(const std::string& file_path)
{
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open NetCDF file: " + file_path);
    }

    std::array<char, 4> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in)
    {
        throw std::runtime_error("Failed to read NetCDF signature from: " + file_path);
    }
    if (magic[0] != 'C' || magic[1] != 'D' || magic[2] != 'F')
    {
        throw std::runtime_error("Not a NetCDF classic file (invalid CDF signature)");
    }

    const int version = static_cast<unsigned char>(magic[3]);
    if (version != 1 && version != 2)
    {
        throw std::runtime_error(
            "Native NetCDF parser supports only classic/64-bit-offset formats (CDF1/CDF2)");
    }

    NetcdfClassicHeader header;
    header.version = version;
    header.num_records = static_cast<std::uint64_t>(read_be_u32(in, "numrecs"));
    header.dimensions = read_netcdf_dimension_list(in);
    header.global_attributes = read_netcdf_attribute_list(in, "global attributes");
    header.variables = read_netcdf_variable_list(in, version);

    for (NetcdfVariable& var : header.variables)
    {
        for (const int32_t dim_id : var.dim_ids)
        {
            if (dim_id < 0 || static_cast<std::size_t>(dim_id) >= header.dimensions.size())
            {
                throw std::runtime_error(
                    "NetCDF variable '" + var.name + "' references invalid dimension id");
            }
        }
        var.record_var = !var.dim_ids.empty() &&
                         header.dimensions[static_cast<std::size_t>(var.dim_ids.front())].unlimited;
    }

    return header;
}

/**
 * @brief Finds the first matching variable from prioritized candidate names.
 */
const NetcdfVariable* find_netcdf_variable(
    const NetcdfClassicHeader& header,
    const std::vector<std::string>& candidates)
{
    std::map<std::string, const NetcdfVariable*> lower_exact;
    std::map<std::string, const NetcdfVariable*> normalized_exact;
    for (const NetcdfVariable& var : header.variables)
    {
        const std::string lower = to_lower_ascii(var.name);
        lower_exact.emplace(lower, &var);
        normalized_exact.emplace(normalize_profile_name(var.name), &var);
    }

    for (const std::string& candidate : candidates)
    {
        const std::string key_lower = to_lower_ascii(candidate);
        if (auto it = lower_exact.find(key_lower); it != lower_exact.end())
        {
            return it->second;
        }

        const std::string key_norm = normalize_profile_name(candidate);
        if (auto it = normalized_exact.find(key_norm); it != normalized_exact.end())
        {
            return it->second;
        }

        for (const NetcdfVariable& var : header.variables)
        {
            const std::string lower_name = to_lower_ascii(var.name);
            if (lower_name == key_lower ||
                ends_with(lower_name, "/" + key_lower) ||
                ends_with(lower_name, "_" + key_lower))
            {
                return &var;
            }
        }
    }

    return nullptr;
}

/**
 * @brief Computes total element count for a NetCDF variable.
 */
std::size_t netcdf_variable_element_count(const NetcdfClassicHeader& header, const NetcdfVariable& var)
{
    if (var.dim_ids.empty())
    {
        return 1;
    }

    std::size_t count = 1;
    for (const int32_t dim_id_raw : var.dim_ids)
    {
        const std::size_t dim_id = static_cast<std::size_t>(dim_id_raw);
        const NetcdfDimension& dim = header.dimensions[dim_id];
        const std::uint64_t dim_len_u64 = dim.unlimited ? header.num_records : dim.length;
        if (dim_len_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::runtime_error(
                "NetCDF dimension length is too large for variable '" + var.name + "'");
        }
        std::size_t updated = 0;
        if (!checked_multiply_size(count, static_cast<std::size_t>(dim_len_u64), updated))
        {
            throw std::runtime_error("NetCDF variable element count overflow for '" + var.name + "'");
        }
        count = updated;
    }
    return count;
}

/**
 * @brief Computes per-record element count for a record variable.
 */
std::size_t netcdf_variable_elements_per_record(const NetcdfClassicHeader& header, const NetcdfVariable& var)
{
    if (!var.record_var)
    {
        return netcdf_variable_element_count(header, var);
    }
    if (var.dim_ids.empty())
    {
        throw std::runtime_error("Record variable has no dimensions: '" + var.name + "'");
    }

    std::size_t count = 1;
    for (std::size_t i = 1; i < var.dim_ids.size(); ++i)
    {
        const int32_t dim_id_raw = var.dim_ids[i];
        const std::size_t dim_id = static_cast<std::size_t>(dim_id_raw);
        const NetcdfDimension& dim = header.dimensions[dim_id];
        if (dim.unlimited)
        {
            throw std::runtime_error(
                "Only leading unlimited dimension is supported for '" + var.name + "'");
        }
        const std::uint64_t dim_len_u64 = dim.length;
        if (dim_len_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::runtime_error("Dimension length too large for '" + var.name + "'");
        }
        std::size_t updated = 0;
        if (!checked_multiply_size(count, static_cast<std::size_t>(dim_len_u64), updated))
        {
            throw std::runtime_error("Record element count overflow for '" + var.name + "'");
        }
        count = updated;
    }
    return count;
}

/**
 * @brief Computes aggregate byte stride across all record variables.
 */
std::uint64_t netcdf_record_stride_bytes(const NetcdfClassicHeader& header)
{
    std::uint64_t stride = 0;
    for (const NetcdfVariable& candidate : header.variables)
    {
        if (!candidate.record_var)
        {
            continue;
        }
        stride += static_cast<std::uint64_t>(candidate.vsize);
    }
    return stride;
}

/**
 * @brief Decodes one NetCDF numeric scalar from big-endian bytes.
 */
double decode_netcdf_numeric_value(const std::uint8_t* bytes, int32_t type)
{
    switch (type)
    {
        case kNcByte:
            return static_cast<double>(static_cast<std::int8_t>(bytes[0]));
        case kNcUByte:
        case kNcChar:
            return static_cast<double>(bytes[0]);
        case kNcShort:
        {
            const std::uint16_t u = (static_cast<std::uint16_t>(bytes[0]) << 8) |
                                    static_cast<std::uint16_t>(bytes[1]);
            return static_cast<double>(static_cast<std::int16_t>(u));
        }
        case kNcUShort:
        {
            const std::uint16_t u = (static_cast<std::uint16_t>(bytes[0]) << 8) |
                                    static_cast<std::uint16_t>(bytes[1]);
            return static_cast<double>(u);
        }
        case kNcInt:
        {
            const std::uint32_t u = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                                    (static_cast<std::uint32_t>(bytes[1]) << 16) |
                                    (static_cast<std::uint32_t>(bytes[2]) << 8) |
                                    static_cast<std::uint32_t>(bytes[3]);
            return static_cast<double>(static_cast<std::int32_t>(u));
        }
        case kNcUInt:
        {
            const std::uint32_t u = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                                    (static_cast<std::uint32_t>(bytes[1]) << 16) |
                                    (static_cast<std::uint32_t>(bytes[2]) << 8) |
                                    static_cast<std::uint32_t>(bytes[3]);
            return static_cast<double>(u);
        }
        case kNcFloat:
        {
            const std::uint32_t u = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                                    (static_cast<std::uint32_t>(bytes[1]) << 16) |
                                    (static_cast<std::uint32_t>(bytes[2]) << 8) |
                                    static_cast<std::uint32_t>(bytes[3]);
            float f = 0.0f;
            std::memcpy(&f, &u, sizeof(f));
            return static_cast<double>(f);
        }
        case kNcDouble:
        {
            const std::uint64_t u = (static_cast<std::uint64_t>(bytes[0]) << 56) |
                                    (static_cast<std::uint64_t>(bytes[1]) << 48) |
                                    (static_cast<std::uint64_t>(bytes[2]) << 40) |
                                    (static_cast<std::uint64_t>(bytes[3]) << 32) |
                                    (static_cast<std::uint64_t>(bytes[4]) << 24) |
                                    (static_cast<std::uint64_t>(bytes[5]) << 16) |
                                    (static_cast<std::uint64_t>(bytes[6]) << 8) |
                                    static_cast<std::uint64_t>(bytes[7]);
            double d = 0.0;
            std::memcpy(&d, &u, sizeof(d));
            return d;
        }
        case kNcInt64:
        {
            const std::uint64_t u = (static_cast<std::uint64_t>(bytes[0]) << 56) |
                                    (static_cast<std::uint64_t>(bytes[1]) << 48) |
                                    (static_cast<std::uint64_t>(bytes[2]) << 40) |
                                    (static_cast<std::uint64_t>(bytes[3]) << 32) |
                                    (static_cast<std::uint64_t>(bytes[4]) << 24) |
                                    (static_cast<std::uint64_t>(bytes[5]) << 16) |
                                    (static_cast<std::uint64_t>(bytes[6]) << 8) |
                                    static_cast<std::uint64_t>(bytes[7]);
            return static_cast<double>(static_cast<std::int64_t>(u));
        }
        case kNcUInt64:
        {
            const std::uint64_t u = (static_cast<std::uint64_t>(bytes[0]) << 56) |
                                    (static_cast<std::uint64_t>(bytes[1]) << 48) |
                                    (static_cast<std::uint64_t>(bytes[2]) << 40) |
                                    (static_cast<std::uint64_t>(bytes[3]) << 32) |
                                    (static_cast<std::uint64_t>(bytes[4]) << 24) |
                                    (static_cast<std::uint64_t>(bytes[5]) << 16) |
                                    (static_cast<std::uint64_t>(bytes[6]) << 8) |
                                    static_cast<std::uint64_t>(bytes[7]);
            return static_cast<double>(u);
        }
        default:
            throw std::runtime_error("Unsupported NetCDF numeric type: " + std::to_string(type));
    }
}

/**
 * @brief Reads one NetCDF variable payload and returns values as doubles.
 */
std::vector<double> read_netcdf_variable_as_double(
    std::ifstream& in,
    const NetcdfClassicHeader& header,
    const NetcdfVariable& var)
{
    const std::size_t type_size = nc_type_size_bytes(var.type);

    if (!var.record_var)
    {
        const std::size_t element_count = netcdf_variable_element_count(header, var);
        std::size_t byte_count = 0;
        if (!checked_multiply_size(element_count, type_size, byte_count))
        {
            throw std::runtime_error("NetCDF variable byte count overflow: " + var.name);
        }

        std::vector<std::uint8_t> raw(byte_count);
        in.clear();
        in.seekg(static_cast<std::streamoff>(var.begin), std::ios::beg);
        if (!in)
        {
            throw std::runtime_error("Failed to seek to NetCDF variable '" + var.name + "'");
        }
        if (byte_count > 0)
        {
            in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_count));
            if (!in)
            {
                throw std::runtime_error("Failed to read NetCDF variable payload: '" + var.name + "'");
            }
        }

        std::vector<double> values(element_count);
        for (std::size_t i = 0; i < element_count; ++i)
        {
            values[i] = decode_netcdf_numeric_value(raw.data() + (i * type_size), var.type);
        }
        return values;
    }

    const std::size_t per_record_elements = netcdf_variable_elements_per_record(header, var);
    const std::uint64_t record_count_u64 = header.num_records;
    if (record_count_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::runtime_error("Record count too large in NetCDF variable '" + var.name + "'");
    }
    const std::size_t record_count = static_cast<std::size_t>(record_count_u64);
    std::size_t total_elements = 0;
    if (!checked_multiply_size(per_record_elements, record_count, total_elements))
    {
        throw std::runtime_error("Record variable size overflow: '" + var.name + "'");
    }

    std::size_t payload_bytes = 0;
    if (!checked_multiply_size(per_record_elements, type_size, payload_bytes))
    {
        throw std::runtime_error("Record payload overflow: '" + var.name + "'");
    }
    if (payload_bytes > static_cast<std::size_t>(var.vsize))
    {
        throw std::runtime_error(
            "Record payload exceeds vsize for '" + var.name + "'");
    }

    const std::uint64_t record_stride = netcdf_record_stride_bytes(header);
    if (record_stride == 0)
    {
        throw std::runtime_error("Invalid NetCDF record stride for '" + var.name + "'");
    }

    std::vector<double> values(total_elements);
    std::vector<std::uint8_t> payload(payload_bytes);
    for (std::size_t rec = 0; rec < record_count; ++rec)
    {
        const std::uint64_t pos_u64 = var.begin + (static_cast<std::uint64_t>(rec) * record_stride);
        if (pos_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
        {
            throw std::runtime_error("Seek offset overflow for record variable '" + var.name + "'");
        }
        in.clear();
        in.seekg(static_cast<std::streamoff>(pos_u64), std::ios::beg);
        if (!in)
        {
            throw std::runtime_error("Failed to seek NetCDF record data for '" + var.name + "'");
        }

        if (payload_bytes > 0)
        {
            in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload_bytes));
            if (!in)
            {
                throw std::runtime_error("Failed to read NetCDF record payload for '" + var.name + "'");
            }
        }

        const std::size_t base = rec * per_record_elements;
        for (std::size_t i = 0; i < per_record_elements; ++i)
        {
            values[base + i] = decode_netcdf_numeric_value(payload.data() + (i * type_size), var.type);
        }
    }

    return values;
}

/**
 * @brief Finds a matching attribute in an attribute map.
 */
const NetcdfAttribute* find_netcdf_attribute(
    const std::map<std::string, NetcdfAttribute>& attrs,
    const std::vector<std::string>& candidates)
{
    for (const std::string& candidate : candidates)
    {
        const std::string key = to_lower_ascii(candidate);
        for (const auto& entry : attrs)
        {
            if (to_lower_ascii(entry.first) == key)
            {
                return &entry.second;
            }
        }
    }
    return nullptr;
}

/**
 * @brief Decodes a NetCDF char attribute into a C++ string.
 */
std::string decode_netcdf_attribute_string(const NetcdfAttribute& attr)
{
    if (attr.type != kNcChar || attr.raw.empty())
    {
        return std::string{};
    }

    const std::size_t length = std::min(attr.element_count, attr.raw.size());
    std::string value(reinterpret_cast<const char*>(attr.raw.data()), length);
    while (!value.empty() && value.back() == '\0')
    {
        value.pop_back();
    }
    return value;
}

/**
 * @brief Decodes a scalar numeric attribute when available.
 */
std::optional<double> decode_netcdf_attribute_numeric(const NetcdfAttribute& attr)
{
    if (attr.element_count == 0 || attr.raw.empty() || attr.type == kNcChar)
    {
        return std::nullopt;
    }
    const std::size_t type_size = nc_type_size_bytes(attr.type);
    if (attr.raw.size() < type_size)
    {
        return std::nullopt;
    }
    return decode_netcdf_numeric_value(attr.raw.data(), attr.type);
}

/**
 * @brief Reads a profile variable by candidate names with optional fallback.
 */
std::vector<double> try_read_profile_variable(
    std::ifstream& in,
    const NetcdfClassicHeader& header,
    const std::vector<std::string>& candidates,
    bool required,
    const std::string& label)
{
    const NetcdfVariable* var = find_netcdf_variable(header, candidates);
    if (var == nullptr)
    {
        if (required)
        {
            throw std::runtime_error("Missing required NetCDF variable: " + label);
        }
        return {};
    }

    try
    {
        return read_netcdf_variable_as_double(in, header, *var);
    }
    catch (const std::exception& e)
    {
        if (required)
        {
            throw;
        }
        std::cerr << "Warning: Failed optional NetCDF variable '" << var->name
                  << "': " << e.what() << std::endl;
        return {};
    }
}

/**
 * @brief Candidate aliases for profile height fields.
 */
const std::vector<std::string>& profile_height_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/height_m", "height_m", "height", "z", "altitude"
    };
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile pressure fields.
 */
const std::vector<std::string>& profile_pressure_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/pressure_hpa", "pressure_hpa", "pressure", "pres", "p"
    };
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile temperature fields.
 */
const std::vector<std::string>& profile_temperature_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/temperature_k", "temperature_k", "temperature", "temp", "t"
    };
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile dewpoint fields.
 */
const std::vector<std::string>& profile_dewpoint_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/dewpoint_k", "dewpoint_k", "dewpoint", "td"
    };
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile wind-speed fields.
 */
const std::vector<std::string>& profile_wind_speed_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/wind_speed_ms", "wind_speed_ms", "wind_speed", "wspd"
    };
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile wind-direction fields.
 */
const std::vector<std::string>& profile_wind_direction_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/wind_direction_deg", "wind_direction_deg", "wind_direction", "wdir"
    };
    return k_candidates;
}

/**
 * @brief Fits profile vectors to the target profile length.
 */
std::vector<double> resize_profile_series(const std::vector<double>& values, std::size_t target_count)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out(target_count, nan);
    if (target_count == 0)
    {
        return out;
    }
    if (values.empty())
    {
        return out;
    }
    if (values.size() == 1)
    {
        std::fill(out.begin(), out.end(), values.front());
        return out;
    }

    const std::size_t ncopy = std::min(values.size(), target_count);
    for (std::size_t i = 0; i < ncopy; ++i)
    {
        out[i] = values[i];
    }
    return out;
}

/**
 * @brief Converts profile vectors into canonical extraction rows.
 */
NativeProfileExtraction build_native_profile_rows(
    const std::vector<double>& height,
    const std::vector<double>& pressure,
    const std::vector<double>& temperature,
    const std::vector<double>& dewpoint,
    const std::vector<double>& wind_speed,
    const std::vector<double>& wind_direction)
{
    const std::size_t n = std::min({height.size(), pressure.size(), temperature.size()});
    if (n < 2)
    {
        throw std::runtime_error("Native reader found insufficient profile levels");
    }

    const std::vector<double> dewpoint_resized = resize_profile_series(dewpoint, n);
    const std::vector<double> wind_speed_resized = resize_profile_series(wind_speed, n);
    const std::vector<double> wind_direction_resized = resize_profile_series(wind_direction, n);

    NativeProfileExtraction extracted;
    extracted.rows.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        extracted.rows.push_back({
            height[i],
            pressure[i],
            temperature[i],
            dewpoint_resized[i],
            wind_speed_resized[i],
            wind_direction_resized[i]
        });
    }
    return extracted;
}

/**
 * @brief Extracts a vertical profile from a NetCDF classic SHARPY file.
 */
NativeProfileExtraction parse_netcdf_classic_profile_native(const std::string& file_path)
{
    const NetcdfClassicHeader header = parse_netcdf_classic_header(file_path);
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to reopen NetCDF file: " + file_path);
    }

    const std::vector<double> height = try_read_profile_variable(
        in, header, profile_height_candidates(), true, "height");
    const std::vector<double> pressure = try_read_profile_variable(
        in, header, profile_pressure_candidates(), true, "pressure");
    const std::vector<double> temperature = try_read_profile_variable(
        in, header, profile_temperature_candidates(), true, "temperature");
    const std::vector<double> dewpoint = try_read_profile_variable(
        in, header, profile_dewpoint_candidates(), false, "dewpoint");
    const std::vector<double> wind_speed = try_read_profile_variable(
        in, header, profile_wind_speed_candidates(), false, "wind_speed");
    const std::vector<double> wind_dir = try_read_profile_variable(
        in, header, profile_wind_direction_candidates(), false, "wind_direction");

    NativeProfileExtraction extracted = build_native_profile_rows(
        height, pressure, temperature, dewpoint, wind_speed, wind_dir);

    if (const NetcdfAttribute* station = find_netcdf_attribute(header.global_attributes, {"station_id"}))
    {
        extracted.station_id = decode_netcdf_attribute_string(*station);
    }
    if (const NetcdfAttribute* timestamp = find_netcdf_attribute(header.global_attributes, {"timestamp_utc"}))
    {
        extracted.timestamp_utc = decode_netcdf_attribute_string(*timestamp);
    }
    if (const NetcdfAttribute* latitude = find_netcdf_attribute(header.global_attributes, {"latitude_deg", "latitude"}))
    {
        if (auto value = decode_netcdf_attribute_numeric(*latitude); value.has_value())
        {
            extracted.latitude_deg = *value;
        }
    }
    if (const NetcdfAttribute* longitude = find_netcdf_attribute(header.global_attributes, {"longitude_deg", "longitude"}))
    {
        if (auto value = decode_netcdf_attribute_numeric(*longitude); value.has_value())
        {
            extracted.longitude_deg = *value;
        }
    }
    if (const NetcdfAttribute* elevation = find_netcdf_attribute(header.global_attributes, {"elevation_m", "elevation"}))
    {
        if (auto value = decode_netcdf_attribute_numeric(*elevation); value.has_value())
        {
            extracted.elevation_m = *value;
        }
    }

    return extracted;
}

/**
 * @brief Normalizes a path-like identifier for case-insensitive comparisons.
 */
std::string normalize_profile_lookup_key(const std::string& value)
{
    std::string out = to_lower_ascii(value);
    while (!out.empty() && out.front() == '/')
    {
        out.erase(out.begin());
    }
    return out;
}

/**
 * @brief Returns candidate variants for profile lookup across backends.
 */
std::vector<std::string> expand_profile_candidates(const std::vector<std::string>& candidates)
{
    std::set<std::string> dedup;
    for (const std::string& candidate : candidates)
    {
        std::string base = candidate;
        while (!base.empty() && base.front() == '/')
        {
            base.erase(base.begin());
        }
        if (base.empty())
        {
            continue;
        }

        dedup.insert(base);
        dedup.insert("/" + base);

        const std::size_t slash = base.find_last_of('/');
        const std::string leaf = (slash == std::string::npos)
            ? base
            : base.substr(slash + 1);
        if (!leaf.empty())
        {
            dedup.insert(leaf);
            dedup.insert("/" + leaf);
            dedup.insert("profiles/" + leaf);
            dedup.insert("/profiles/" + leaf);
            dedup.insert("metadata/" + leaf);
            dedup.insert("/metadata/" + leaf);
        }
    }
    return std::vector<std::string>(dedup.begin(), dedup.end());
}

/**
 * @brief Stores one backend attempt diagnostic for native profile loading.
 */
struct NativeReadAttempt
{
    std::string backend;
    std::string detail;
};

/**
 * @brief Appends a backend failure diagnostic.
 */
void append_native_attempt_failure(std::vector<NativeReadAttempt>& attempts,
                                   std::string backend,
                                   const std::exception& error)
{
    attempts.push_back(NativeReadAttempt{std::move(backend), error.what()});
}

/**
 * @brief Throws an aggregated error for failed native ingestion attempts.
 */
[[noreturn]] void throw_native_ingest_failure(const std::string& file_path,
                                              const std::vector<NativeReadAttempt>& attempts)
{
    std::ostringstream oss;
    oss << "Native sounding ingest failed for '" << file_path << "'";
    if (attempts.empty())
    {
        oss << " with no backend attempts recorded.";
        throw std::runtime_error(oss.str());
    }

    oss << " (attempts:";
    for (const NativeReadAttempt& attempt : attempts)
    {
        oss << "\n  - " << attempt.backend << ": " << attempt.detail;
    }
    oss << "\n)";
    throw std::runtime_error(oss.str());
}

/**
 * @brief Applies native metadata fields onto parsed sounding output.
 */
void apply_native_metadata(SoundingData& data, const NativeProfileExtraction& native)
{
    data.station_id = native.station_id;
    data.timestamp_utc = native.timestamp_utc;
    data.latitude_deg = native.latitude_deg;
    data.longitude_deg = native.longitude_deg;
    data.elevation_m = native.elevation_m;
}

/**
 * @brief RAII wrapper for a `dlopen`-acquired shared library handle.
 */
class SharedLibraryHandle
{
public:
    SharedLibraryHandle() = default;
    explicit SharedLibraryHandle(void* handle)
        : handle_(handle)
    {
    }

    SharedLibraryHandle(const SharedLibraryHandle&) = delete;
    SharedLibraryHandle& operator=(const SharedLibraryHandle&) = delete;

    SharedLibraryHandle(SharedLibraryHandle&& other) noexcept
        : handle_(other.handle_)
    {
        other.handle_ = nullptr;
    }
    SharedLibraryHandle& operator=(SharedLibraryHandle&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    ~SharedLibraryHandle()
    {
        reset();
    }

    void* get() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }

    void* release()
    {
        void* out = handle_;
        handle_ = nullptr;
        return out;
    }

private:
    void reset()
    {
        if (handle_ != nullptr)
        {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    void* handle_ = nullptr;
};

/**
 * @brief Opens the first available shared library path from a candidate list.
 */
SharedLibraryHandle open_shared_library_first(const std::vector<std::string>& candidates,
                                              std::string& opened_path,
                                              std::string& error_out)
{
    std::ostringstream errors;
    for (const std::string& candidate : candidates)
    {
        dlerror();
        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr)
        {
            opened_path = candidate;
            return SharedLibraryHandle(handle);
        }

        const char* dl_err = dlerror();
        errors << candidate << ": " << (dl_err ? dl_err : "unknown dlopen error") << "; ";
    }

    error_out = errors.str();
    return SharedLibraryHandle{};
}

/**
 * @brief Loads one symbol function pointer from a dynamic library.
 */
template <typename FnPtr>
bool load_symbol(void* handle, const char* symbol, FnPtr& out)
{
    dlerror();
    void* raw = dlsym(handle, symbol);
    const char* err = dlerror();
    if (raw == nullptr || err != nullptr)
    {
        out = nullptr;
        return false;
    }
    out = reinterpret_cast<FnPtr>(raw);
    return true;
}

/**
 * @brief Dynamic adapter for the NetCDF C API.
 */
struct DynamicNetcdfApi
{
    using NcOpenFn = int (*)(const char*, int, int*);
    using NcCloseFn = int (*)(int);
    using NcInqFn = int (*)(int, int*, int*, int*, int*);
    using NcInqVarFn = int (*)(int, int, char*, int*, int*, int*, int*);
    using NcInqDimFn = int (*)(int, int, char*, std::size_t*);
    using NcGetVarDoubleFn = int (*)(int, int, double*);
    using NcInqAttFn = int (*)(int, int, const char*, int*, std::size_t*);
    using NcGetAttTextFn = int (*)(int, int, const char*, char*);
    using NcGetAttDoubleFn = int (*)(int, int, const char*, double*);
    using NcGetAttStringFn = int (*)(int, int, const char*, char**);
    using NcFreeStringFn = int (*)(std::size_t, char**);
    using NcStrErrorFn = const char* (*)(int);
    using NcInqGrpsFn = int (*)(int, int*, int*);
    using NcInqGrpNameFn = int (*)(int, char*);

    SharedLibraryHandle library;
    std::string library_path;

    NcOpenFn nc_open = nullptr;
    NcCloseFn nc_close = nullptr;
    NcInqFn nc_inq = nullptr;
    NcInqVarFn nc_inq_var = nullptr;
    NcInqDimFn nc_inq_dim = nullptr;
    NcGetVarDoubleFn nc_get_var_double = nullptr;
    NcInqAttFn nc_inq_att = nullptr;
    NcGetAttTextFn nc_get_att_text = nullptr;
    NcGetAttDoubleFn nc_get_att_double = nullptr;
    NcGetAttStringFn nc_get_att_string = nullptr;
    NcFreeStringFn nc_free_string = nullptr;
    NcStrErrorFn nc_strerror = nullptr;
    NcInqGrpsFn nc_inq_grps = nullptr;
    NcInqGrpNameFn nc_inq_grpname = nullptr;
};

/**
 * @brief Loads the NetCDF C API dynamically (without link-time dependency).
 */
DynamicNetcdfApi load_dynamic_netcdf_api()
{
    const std::vector<std::string> library_candidates = {
        "libnetcdf.dylib",
        "libnetcdf.so",
        "libnetcdf.so.19",
        "/opt/homebrew/lib/libnetcdf.dylib",
        "/usr/local/lib/libnetcdf.dylib",
        "/opt/homebrew/opt/netcdf/lib/libnetcdf.dylib",
        "/usr/local/opt/netcdf/lib/libnetcdf.dylib"
    };

    std::string opened_path;
    std::string open_error;
    SharedLibraryHandle handle = open_shared_library_first(library_candidates, opened_path, open_error);
    if (!handle)
    {
        throw std::runtime_error("Unable to load NetCDF shared library: " + open_error);
    }

    DynamicNetcdfApi api;
    api.library = std::move(handle);
    api.library_path = opened_path;

    if (!load_symbol(api.library.get(), "nc_open", api.nc_open) ||
        !load_symbol(api.library.get(), "nc_close", api.nc_close) ||
        !load_symbol(api.library.get(), "nc_inq", api.nc_inq) ||
        !load_symbol(api.library.get(), "nc_inq_var", api.nc_inq_var) ||
        !load_symbol(api.library.get(), "nc_inq_dim", api.nc_inq_dim) ||
        !load_symbol(api.library.get(), "nc_get_var_double", api.nc_get_var_double) ||
        !load_symbol(api.library.get(), "nc_inq_att", api.nc_inq_att) ||
        !load_symbol(api.library.get(), "nc_get_att_text", api.nc_get_att_text) ||
        !load_symbol(api.library.get(), "nc_get_att_double", api.nc_get_att_double) ||
        !load_symbol(api.library.get(), "nc_strerror", api.nc_strerror))
    {
        throw std::runtime_error(
            "NetCDF shared library '" + api.library_path + "' is missing required symbols");
    }

    // Optional symbols (available in NetCDF4-enabled builds).
    load_symbol(api.library.get(), "nc_get_att_string", api.nc_get_att_string);
    load_symbol(api.library.get(), "nc_free_string", api.nc_free_string);
    load_symbol(api.library.get(), "nc_inq_grps", api.nc_inq_grps);
    load_symbol(api.library.get(), "nc_inq_grpname", api.nc_inq_grpname);

    return api;
}

/**
 * @brief Builds a consistent NetCDF status message.
 */
std::string netcdf_status_message(const DynamicNetcdfApi& api, int status)
{
    if (api.nc_strerror != nullptr)
    {
        const char* text = api.nc_strerror(status);
        if (text != nullptr)
        {
            return std::string(text);
        }
    }
    return "status=" + std::to_string(status);
}

/**
 * @brief RAII close wrapper for dynamically opened NetCDF files.
 */
class NetcdfFileHandle
{
public:
    NetcdfFileHandle(const DynamicNetcdfApi* api, int id)
        : api_(api), id_(id)
    {
    }

    NetcdfFileHandle(const NetcdfFileHandle&) = delete;
    NetcdfFileHandle& operator=(const NetcdfFileHandle&) = delete;

    NetcdfFileHandle(NetcdfFileHandle&& other) noexcept
        : api_(other.api_), id_(other.id_)
    {
        other.api_ = nullptr;
        other.id_ = -1;
    }

    ~NetcdfFileHandle()
    {
        if (api_ != nullptr && id_ >= 0 && api_->nc_close != nullptr)
        {
            api_->nc_close(id_);
        }
    }

    int id() const { return id_; }

private:
    const DynamicNetcdfApi* api_ = nullptr;
    int id_ = -1;
};

/**
 * @brief Lightweight descriptor for NetCDF variables discovered via C API.
 */
struct DynamicNetcdfVariable
{
    int group_id = -1;
    int var_id = -1;
    int type = 0;
    std::string short_name;
    std::string full_name;
    std::vector<std::size_t> dim_lengths;
};

/**
 * @brief Recursively enumerates variables across NetCDF groups.
 */
void enumerate_netcdf_variables_recursive(const DynamicNetcdfApi& api,
                                          int group_id,
                                          const std::string& group_path,
                                          std::vector<DynamicNetcdfVariable>& out)
{
    int ndims = 0;
    int nvars = 0;
    int ngatts = 0;
    int unlim = -1;
    const int rc_inq = api.nc_inq(group_id, &ndims, &nvars, &ngatts, &unlim);
    if (rc_inq != 0)
    {
        throw std::runtime_error("nc_inq failed while enumerating variables: " +
                                 netcdf_status_message(api, rc_inq));
    }

    constexpr int k_name_buffer_size = 1024;
    for (int var_id = 0; var_id < nvars; ++var_id)
    {
        char var_name[k_name_buffer_size];
        std::memset(var_name, 0, sizeof(var_name));

        int var_type = 0;
        int var_ndims = 0;
        int var_natts = 0;
        const int rc_meta = api.nc_inq_var(
            group_id, var_id, var_name, &var_type, &var_ndims, nullptr, &var_natts);
        if (rc_meta != 0)
        {
            throw std::runtime_error("nc_inq_var failed for variable metadata: " +
                                     netcdf_status_message(api, rc_meta));
        }

        std::vector<int> dim_ids(static_cast<std::size_t>(std::max(0, var_ndims)), 0);
        const int rc_dims = api.nc_inq_var(
            group_id, var_id, var_name, &var_type, &var_ndims,
            dim_ids.empty() ? nullptr : dim_ids.data(), &var_natts);
        if (rc_dims != 0)
        {
            throw std::runtime_error("nc_inq_var failed while reading dimensions: " +
                                     netcdf_status_message(api, rc_dims));
        }

        DynamicNetcdfVariable desc;
        desc.group_id = group_id;
        desc.var_id = var_id;
        desc.type = var_type;
        desc.short_name = std::string(var_name);
        desc.full_name = group_path.empty() ? desc.short_name : (group_path + "/" + desc.short_name);
        desc.dim_lengths.reserve(dim_ids.size());

        for (const int dim_id : dim_ids)
        {
            std::size_t dim_len = 0;
            const int rc_dim = api.nc_inq_dim(group_id, dim_id, nullptr, &dim_len);
            if (rc_dim != 0)
            {
                throw std::runtime_error("nc_inq_dim failed for variable '" + desc.full_name + "': " +
                                         netcdf_status_message(api, rc_dim));
            }
            desc.dim_lengths.push_back(dim_len);
        }

        out.push_back(std::move(desc));
    }

    if (api.nc_inq_grps == nullptr || api.nc_inq_grpname == nullptr)
    {
        return;
    }

    int child_count = 0;
    const int rc_child_count = api.nc_inq_grps(group_id, &child_count, nullptr);
    if (rc_child_count != 0 || child_count <= 0)
    {
        return;
    }

    std::vector<int> child_ids(static_cast<std::size_t>(child_count), -1);
    int child_count_req = child_count;
    const int rc_children = api.nc_inq_grps(group_id, &child_count_req, child_ids.data());
    if (rc_children != 0)
    {
        return;
    }

    for (int child_id : child_ids)
    {
        char group_name[k_name_buffer_size];
        std::memset(group_name, 0, sizeof(group_name));
        if (api.nc_inq_grpname(child_id, group_name) != 0)
        {
            continue;
        }

        const std::string child_path = group_path.empty()
            ? std::string(group_name)
            : group_path + "/" + std::string(group_name);
        enumerate_netcdf_variables_recursive(api, child_id, child_path, out);
    }
}

/**
 * @brief Finds a variable in NetCDF C API metadata using candidate aliases.
 */
const DynamicNetcdfVariable* find_dynamic_netcdf_variable(
    const std::vector<DynamicNetcdfVariable>& vars,
    const std::vector<std::string>& candidates)
{
    for (const std::string& candidate_raw : candidates)
    {
        const std::string candidate = normalize_profile_lookup_key(candidate_raw);
        if (candidate.empty())
        {
            continue;
        }

        const std::string candidate_norm = normalize_profile_name(candidate);
        const std::size_t slash = candidate.find_last_of('/');
        const std::string candidate_leaf = (slash == std::string::npos)
            ? candidate
            : candidate.substr(slash + 1);

        for (const DynamicNetcdfVariable& var : vars)
        {
            const std::string full = normalize_profile_lookup_key(var.full_name);
            const std::string short_name = normalize_profile_lookup_key(var.short_name);
            const std::string full_norm = normalize_profile_name(full);
            const std::string short_norm = normalize_profile_name(short_name);

            if (full == candidate ||
                short_name == candidate ||
                full_norm == candidate_norm ||
                short_norm == candidate_norm ||
                ends_with(full, "/" + candidate_leaf) ||
                ends_with(full, "_" + candidate_leaf) ||
                ends_with(short_name, "_" + candidate_leaf))
            {
                return &var;
            }
        }
    }
    return nullptr;
}

/**
 * @brief Reads one discovered NetCDF variable and flattens values to doubles.
 */
std::vector<double> read_dynamic_netcdf_variable_as_double(const DynamicNetcdfApi& api,
                                                           const DynamicNetcdfVariable& var)
{
    std::size_t total = var.dim_lengths.empty() ? 1 : 1;
    if (!var.dim_lengths.empty())
    {
        for (const std::size_t dim_len : var.dim_lengths)
        {
            std::size_t updated = 0;
            if (!checked_multiply_size(total, dim_len, updated))
            {
                throw std::runtime_error("Variable size overflow for '" + var.full_name + "'");
            }
            total = updated;
        }
    }

    std::vector<double> values(total, 0.0);
    if (total == 0)
    {
        return values;
    }

    const int rc = api.nc_get_var_double(var.group_id, var.var_id, values.data());
    if (rc != 0)
    {
        throw std::runtime_error("nc_get_var_double failed for '" + var.full_name + "': " +
                                 netcdf_status_message(api, rc));
    }
    return values;
}

/**
 * @brief Reads a global NetCDF string attribute by candidate names.
 */
std::optional<std::string> read_dynamic_netcdf_global_attr_string(
    const DynamicNetcdfApi& api,
    int ncid,
    const std::vector<std::string>& candidates)
{
    constexpr int kNcGlobal = -1;
    constexpr int kNcChar = 2;
    constexpr int kNcString = 12;

    for (const std::string& name : candidates)
    {
        int attr_type = 0;
        std::size_t attr_len = 0;
        const int rc_inq = api.nc_inq_att(ncid, kNcGlobal, name.c_str(), &attr_type, &attr_len);
        if (rc_inq != 0)
        {
            continue;
        }

        if (attr_type == kNcChar)
        {
            std::string value(attr_len, '\0');
            if (!value.empty())
            {
                const int rc_read = api.nc_get_att_text(ncid, kNcGlobal, name.c_str(), value.data());
                if (rc_read != 0)
                {
                    continue;
                }
            }
            while (!value.empty() && value.back() == '\0')
            {
                value.pop_back();
            }
            return value;
        }

        if (attr_type == kNcString &&
            api.nc_get_att_string != nullptr &&
            api.nc_free_string != nullptr &&
            attr_len > 0)
        {
            std::vector<char*> values(attr_len, nullptr);
            const int rc_read = api.nc_get_att_string(ncid, kNcGlobal, name.c_str(), values.data());
            if (rc_read != 0)
            {
                continue;
            }

            std::string value;
            if (!values.empty() && values[0] != nullptr)
            {
                value = values[0];
            }
            api.nc_free_string(attr_len, values.data());
            return value;
        }
    }

    return std::nullopt;
}

/**
 * @brief Reads a global NetCDF numeric attribute by candidate names.
 */
std::optional<double> read_dynamic_netcdf_global_attr_numeric(
    const DynamicNetcdfApi& api,
    int ncid,
    const std::vector<std::string>& candidates)
{
    constexpr int kNcGlobal = -1;
    constexpr int kNcChar = 2;
    constexpr int kNcString = 12;

    for (const std::string& name : candidates)
    {
        int attr_type = 0;
        std::size_t attr_len = 0;
        const int rc_inq = api.nc_inq_att(ncid, kNcGlobal, name.c_str(), &attr_type, &attr_len);
        if (rc_inq != 0 || attr_len == 0)
        {
            continue;
        }
        if (attr_type == kNcChar || attr_type == kNcString)
        {
            continue;
        }

        std::vector<double> values(attr_len, std::numeric_limits<double>::quiet_NaN());
        const int rc_read = api.nc_get_att_double(ncid, kNcGlobal, name.c_str(), values.data());
        if (rc_read == 0 && !values.empty())
        {
            return values[0];
        }
    }

    return std::nullopt;
}

/**
 * @brief Extracts profiles using a dynamic NetCDF C API backend.
 */
NativeProfileExtraction parse_netcdf_dynamic_profile_native(const std::string& file_path)
{
    DynamicNetcdfApi api = load_dynamic_netcdf_api();

    constexpr int kNcNowrite = 0;
    int ncid = -1;
    const int rc_open = api.nc_open(file_path.c_str(), kNcNowrite, &ncid);
    if (rc_open != 0 || ncid < 0)
    {
        throw std::runtime_error("nc_open failed: " + netcdf_status_message(api, rc_open));
    }

    NetcdfFileHandle file_guard(&api, ncid);
    std::vector<DynamicNetcdfVariable> variables;
    enumerate_netcdf_variables_recursive(api, ncid, "", variables);
    if (variables.empty())
    {
        throw std::runtime_error("No variables found in NetCDF file");
    }

    const DynamicNetcdfVariable* height_var = find_dynamic_netcdf_variable(variables, profile_height_candidates());
    const DynamicNetcdfVariable* pressure_var = find_dynamic_netcdf_variable(variables, profile_pressure_candidates());
    const DynamicNetcdfVariable* temp_var = find_dynamic_netcdf_variable(variables, profile_temperature_candidates());
    if (height_var == nullptr || pressure_var == nullptr || temp_var == nullptr)
    {
        throw std::runtime_error("Missing required profile variables in NetCDF file");
    }

    const DynamicNetcdfVariable* dewpoint_var = find_dynamic_netcdf_variable(variables, profile_dewpoint_candidates());
    const DynamicNetcdfVariable* wind_speed_var = find_dynamic_netcdf_variable(variables, profile_wind_speed_candidates());
    const DynamicNetcdfVariable* wind_dir_var = find_dynamic_netcdf_variable(variables, profile_wind_direction_candidates());

    const std::vector<double> height = read_dynamic_netcdf_variable_as_double(api, *height_var);
    const std::vector<double> pressure = read_dynamic_netcdf_variable_as_double(api, *pressure_var);
    const std::vector<double> temperature = read_dynamic_netcdf_variable_as_double(api, *temp_var);
    const std::vector<double> dewpoint = (dewpoint_var != nullptr)
        ? read_dynamic_netcdf_variable_as_double(api, *dewpoint_var)
        : std::vector<double>{};
    const std::vector<double> wind_speed = (wind_speed_var != nullptr)
        ? read_dynamic_netcdf_variable_as_double(api, *wind_speed_var)
        : std::vector<double>{};
    const std::vector<double> wind_dir = (wind_dir_var != nullptr)
        ? read_dynamic_netcdf_variable_as_double(api, *wind_dir_var)
        : std::vector<double>{};

    NativeProfileExtraction extracted = build_native_profile_rows(
        height, pressure, temperature, dewpoint, wind_speed, wind_dir);

    if (auto station = read_dynamic_netcdf_global_attr_string(api, ncid, {"station_id"}); station.has_value())
    {
        extracted.station_id = *station;
    }
    if (auto timestamp = read_dynamic_netcdf_global_attr_string(api, ncid, {"timestamp_utc"}); timestamp.has_value())
    {
        extracted.timestamp_utc = *timestamp;
    }
    if (auto latitude = read_dynamic_netcdf_global_attr_numeric(api, ncid, {"latitude_deg", "latitude"}); latitude.has_value())
    {
        extracted.latitude_deg = *latitude;
    }
    if (auto longitude = read_dynamic_netcdf_global_attr_numeric(api, ncid, {"longitude_deg", "longitude"}); longitude.has_value())
    {
        extracted.longitude_deg = *longitude;
    }
    if (auto elevation = read_dynamic_netcdf_global_attr_numeric(api, ncid, {"elevation_m", "elevation"}); elevation.has_value())
    {
        extracted.elevation_m = *elevation;
    }

    return extracted;
}

/**
 * @brief Dynamic adapter for HDF5 core + high-level APIs.
 */
struct DynamicHdf5Api
{
    using HidT = long long;
    using HerrT = int;
    using HSizeT = unsigned long long;

    using H5OpenFn = HerrT (*)();
    using H5FOpenFn = HidT (*)(const char*, unsigned, HidT);
    using H5FCloseFn = HerrT (*)(HidT);
    using H5LTGetDatasetNdimsFn = HerrT (*)(HidT, const char*, int*);
    using H5LTGetDatasetInfoFn = HerrT (*)(HidT, const char*, HSizeT*, int*, std::size_t*);
    using H5LTReadDatasetDoubleFn = HerrT (*)(HidT, const char*, double*);
    using H5LTReadDatasetStringFn = HerrT (*)(HidT, const char*, char*);
    using H5LTGetAttributeStringFn = HerrT (*)(HidT, const char*, const char*, char*);
    using H5LTGetAttributeDoubleFn = HerrT (*)(HidT, const char*, const char*, double*);

    SharedLibraryHandle core_library;
    SharedLibraryHandle hl_library;
    std::string core_path;
    std::string hl_path;

    H5OpenFn H5open = nullptr;
    H5FOpenFn H5Fopen = nullptr;
    H5FCloseFn H5Fclose = nullptr;
    H5LTGetDatasetNdimsFn H5LTget_dataset_ndims = nullptr;
    H5LTGetDatasetInfoFn H5LTget_dataset_info = nullptr;
    H5LTReadDatasetDoubleFn H5LTread_dataset_double = nullptr;
    H5LTReadDatasetStringFn H5LTread_dataset_string = nullptr;
    H5LTGetAttributeStringFn H5LTget_attribute_string = nullptr;
    H5LTGetAttributeDoubleFn H5LTget_attribute_double = nullptr;
};

/**
 * @brief Loads HDF5 + HDF5-HL shared libraries dynamically.
 */
DynamicHdf5Api load_dynamic_hdf5_api()
{
    const std::vector<std::string> core_candidates = {
        "libhdf5.dylib",
        "libhdf5.so",
        "libhdf5.so.310",
        "/opt/homebrew/lib/libhdf5.dylib",
        "/usr/local/lib/libhdf5.dylib",
        "/opt/homebrew/opt/hdf5/lib/libhdf5.dylib",
        "/usr/local/opt/hdf5/lib/libhdf5.dylib"
    };
    const std::vector<std::string> hl_candidates = {
        "libhdf5_hl.dylib",
        "libhdf5_hl.so",
        "libhdf5_hl.so.310",
        "/opt/homebrew/lib/libhdf5_hl.dylib",
        "/usr/local/lib/libhdf5_hl.dylib",
        "/opt/homebrew/opt/hdf5/lib/libhdf5_hl.dylib",
        "/usr/local/opt/hdf5/lib/libhdf5_hl.dylib"
    };

    std::string core_path;
    std::string core_error;
    SharedLibraryHandle core = open_shared_library_first(core_candidates, core_path, core_error);
    if (!core)
    {
        throw std::runtime_error("Unable to load HDF5 shared library: " + core_error);
    }

    std::string hl_path;
    std::string hl_error;
    SharedLibraryHandle hl = open_shared_library_first(hl_candidates, hl_path, hl_error);
    if (!hl)
    {
        throw std::runtime_error("Unable to load HDF5 HL shared library: " + hl_error);
    }

    DynamicHdf5Api api;
    api.core_library = std::move(core);
    api.hl_library = std::move(hl);
    api.core_path = core_path;
    api.hl_path = hl_path;

    if (!load_symbol(api.core_library.get(), "H5open", api.H5open) ||
        !load_symbol(api.core_library.get(), "H5Fopen", api.H5Fopen) ||
        !load_symbol(api.core_library.get(), "H5Fclose", api.H5Fclose) ||
        !load_symbol(api.hl_library.get(), "H5LTget_dataset_ndims", api.H5LTget_dataset_ndims) ||
        !load_symbol(api.hl_library.get(), "H5LTget_dataset_info", api.H5LTget_dataset_info) ||
        !load_symbol(api.hl_library.get(), "H5LTread_dataset_double", api.H5LTread_dataset_double))
    {
        throw std::runtime_error("HDF5 shared libraries are missing required symbols");
    }

    // Optional metadata helpers.
    load_symbol(api.hl_library.get(), "H5LTread_dataset_string", api.H5LTread_dataset_string);
    load_symbol(api.hl_library.get(), "H5LTget_attribute_string", api.H5LTget_attribute_string);
    load_symbol(api.hl_library.get(), "H5LTget_attribute_double", api.H5LTget_attribute_double);

    return api;
}

/**
 * @brief Reads an HDF5 dataset (matched via alias candidates) as doubles.
 */
std::vector<double> read_dynamic_hdf5_dataset_as_double(const DynamicHdf5Api& api,
                                                        DynamicHdf5Api::HidT file_id,
                                                        const std::vector<std::string>& candidates,
                                                        bool required,
                                                        const char* label)
{
    const std::vector<std::string> expanded = expand_profile_candidates(candidates);
    for (const std::string& candidate_raw : expanded)
    {
        if (candidate_raw.empty())
        {
            continue;
        }
        const std::string candidate = candidate_raw;

        int rank = 0;
        if (api.H5LTget_dataset_ndims(file_id, candidate.c_str(), &rank) < 0)
        {
            continue;
        }

        std::vector<DynamicHdf5Api::HSizeT> dims(static_cast<std::size_t>(std::max(1, rank)), 1u);
        int type_class = 0;
        std::size_t type_size = 0;
        if (api.H5LTget_dataset_info(
                file_id,
                candidate.c_str(),
                rank > 0 ? dims.data() : nullptr,
                &type_class,
                &type_size) < 0)
        {
            continue;
        }

        std::size_t count = 1;
        if (rank > 0)
        {
            for (int i = 0; i < rank; ++i)
            {
                std::size_t updated = 0;
                if (!checked_multiply_size(count, static_cast<std::size_t>(dims[static_cast<std::size_t>(i)]), updated))
                {
                    throw std::runtime_error("HDF5 dataset size overflow for '" + candidate + "'");
                }
                count = updated;
            }
        }

        std::vector<double> values(count, 0.0);
        if (count > 0 &&
            api.H5LTread_dataset_double(file_id, candidate.c_str(), values.data()) < 0)
        {
            throw std::runtime_error("Failed to read HDF5 dataset '" + candidate + "' as doubles");
        }
        return values;
    }

    if (required)
    {
        throw std::runtime_error(std::string("Missing required HDF5 dataset: ") + label);
    }
    return {};
}

/**
 * @brief Reads a string metadata field from HDF5 attributes or scalar datasets.
 */
std::optional<std::string> read_dynamic_hdf5_string_metadata(
    const DynamicHdf5Api& api,
    DynamicHdf5Api::HidT file_id,
    const std::vector<std::string>& candidates)
{
    const std::array<const char*, 2> metadata_objects = {"/metadata", "/"};

    if (api.H5LTget_attribute_string != nullptr)
    {
        for (const char* object : metadata_objects)
        {
            for (const std::string& candidate : candidates)
            {
                std::array<char, 1024> buf{};
                if (api.H5LTget_attribute_string(file_id, object, candidate.c_str(), buf.data()) >= 0)
                {
                    std::string value(buf.data());
                    while (!value.empty() && value.back() == '\0')
                    {
                        value.pop_back();
                    }
                    return value;
                }
            }
        }
    }

    if (api.H5LTread_dataset_string != nullptr)
    {
        const std::vector<std::string> expanded = expand_profile_candidates(candidates);
        for (const std::string& candidate_raw : expanded)
        {
            if (candidate_raw.empty())
            {
                continue;
            }
            const std::string candidate = candidate_raw;

            int rank = 0;
            if (api.H5LTget_dataset_ndims(file_id, candidate.c_str(), &rank) < 0)
            {
                continue;
            }

            std::array<char, 1024> buf{};
            if (api.H5LTread_dataset_string(file_id, candidate.c_str(), buf.data()) >= 0)
            {
                std::string value(buf.data());
                while (!value.empty() && value.back() == '\0')
                {
                    value.pop_back();
                }
                return value;
            }
        }
    }

    return std::nullopt;
}

/**
 * @brief Reads a numeric metadata field from HDF5 attributes.
 */
std::optional<double> read_dynamic_hdf5_numeric_metadata(
    const DynamicHdf5Api& api,
    DynamicHdf5Api::HidT file_id,
    const std::vector<std::string>& candidates)
{
    if (api.H5LTget_attribute_double == nullptr)
    {
        return std::nullopt;
    }

    const std::array<const char*, 2> metadata_objects = {"/metadata", "/"};
    for (const char* object : metadata_objects)
    {
        for (const std::string& candidate : candidates)
        {
            double value = std::numeric_limits<double>::quiet_NaN();
            if (api.H5LTget_attribute_double(file_id, object, candidate.c_str(), &value) >= 0)
            {
                return value;
            }
        }
    }
    return std::nullopt;
}

/**
 * @brief Extracts profiles using a dynamic HDF5 backend.
 */
NativeProfileExtraction parse_hdf5_profile_native(const std::string& file_path)
{
    DynamicHdf5Api api = load_dynamic_hdf5_api();
    if (api.H5open != nullptr)
    {
        (void)api.H5open();
    }

    constexpr unsigned kH5fAccRdonly = 0u;
    constexpr DynamicHdf5Api::HidT kH5pDefault = 0;
    const DynamicHdf5Api::HidT file_id = api.H5Fopen(file_path.c_str(), kH5fAccRdonly, kH5pDefault);
    if (file_id < 0)
    {
        throw std::runtime_error("H5Fopen failed for '" + file_path + "'");
    }

    try
    {
        const std::vector<double> height = read_dynamic_hdf5_dataset_as_double(
            api, file_id, profile_height_candidates(), true, "height");
        const std::vector<double> pressure = read_dynamic_hdf5_dataset_as_double(
            api, file_id, profile_pressure_candidates(), true, "pressure");
        const std::vector<double> temperature = read_dynamic_hdf5_dataset_as_double(
            api, file_id, profile_temperature_candidates(), true, "temperature");
        const std::vector<double> dewpoint = read_dynamic_hdf5_dataset_as_double(
            api, file_id, profile_dewpoint_candidates(), false, "dewpoint");
        const std::vector<double> wind_speed = read_dynamic_hdf5_dataset_as_double(
            api, file_id, profile_wind_speed_candidates(), false, "wind_speed");
        const std::vector<double> wind_direction = read_dynamic_hdf5_dataset_as_double(
            api, file_id, profile_wind_direction_candidates(), false, "wind_direction");

        NativeProfileExtraction extracted = build_native_profile_rows(
            height, pressure, temperature, dewpoint, wind_speed, wind_direction);

        if (auto station = read_dynamic_hdf5_string_metadata(api, file_id, {"station_id"}); station.has_value())
        {
            extracted.station_id = *station;
        }
        if (auto timestamp = read_dynamic_hdf5_string_metadata(api, file_id, {"timestamp_utc"}); timestamp.has_value())
        {
            extracted.timestamp_utc = *timestamp;
        }
        if (auto latitude = read_dynamic_hdf5_numeric_metadata(api, file_id, {"latitude_deg", "latitude"}); latitude.has_value())
        {
            extracted.latitude_deg = *latitude;
        }
        if (auto longitude = read_dynamic_hdf5_numeric_metadata(api, file_id, {"longitude_deg", "longitude"}); longitude.has_value())
        {
            extracted.longitude_deg = *longitude;
        }
        if (auto elevation = read_dynamic_hdf5_numeric_metadata(api, file_id, {"elevation_m", "elevation"}); elevation.has_value())
        {
            extracted.elevation_m = *elevation;
        }

        (void)api.H5Fclose(file_id);
        return extracted;
    }
    catch (...)
    {
        (void)api.H5Fclose(file_id);
        throw;
    }
}

/**
 * @brief Performs linear interpolation between two sample points.
 */
double linear_interpolate_value(double x1, double y1, double x2, double y2, double x_target)
{
    if (x1 == x2)
    {
        return y1;
    }
    return y1 + (y2 - y1) * (x_target - x1) / (x2 - x1);
}

/**
 * @brief Performs log-linear interpolation for positive-valued series.
 */
double log_linear_interpolate_positive(double x1, double y1, double x2, double y2, double x_target)
{
    if (x1 == x2)
    {
        return y1;
    }
    if (!std::isfinite(y1) || !std::isfinite(y2) || y1 <= 0.0 || y2 <= 0.0)
    {
        return linear_interpolate_value(x1, y1, x2, y2, x_target);
    }

    const double ln_y1 = std::log(y1);
    const double ln_y2 = std::log(y2);
    const double ln_y = linear_interpolate_value(x1, ln_y1, x2, ln_y2, x_target);
    return std::exp(ln_y);
}

/**
 * @brief Interpolates pressure at one target height using log-linear profile space.
 */
double interpolate_pressure_log_linear(const SoundingData& source,
                                       double target_height_m,
                                       const SoundingConfig& config)
{
    if (source.height_m.empty() || source.pressure_hpa.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (source.height_m.size() == 1 || source.pressure_hpa.size() == 1)
    {
        return source.pressure_hpa.front();
    }

    auto it = std::lower_bound(source.height_m.begin(), source.height_m.end(), target_height_m);

    if (it == source.height_m.begin())
    {
        if (config.extrapolate_below_ground)
        {
            return log_linear_interpolate_positive(
                source.height_m[0], source.pressure_hpa[0],
                source.height_m[1], source.pressure_hpa[1],
                target_height_m);
        }
        return source.pressure_hpa.front();
    }

    if (it == source.height_m.end())
    {
        if (config.extrapolate_above_top)
        {
            const std::size_t idx2 = source.height_m.size() - 1;
            const std::size_t idx1 = idx2 - 1;
            return log_linear_interpolate_positive(
                source.height_m[idx1], source.pressure_hpa[idx1],
                source.height_m[idx2], source.pressure_hpa[idx2],
                target_height_m);
        }
        return source.pressure_hpa.back();
    }

    const std::size_t idx2 = static_cast<std::size_t>(it - source.height_m.begin());
    const std::size_t idx1 = idx2 - 1;
    return log_linear_interpolate_positive(
        source.height_m[idx1], source.pressure_hpa[idx1],
        source.height_m[idx2], source.pressure_hpa[idx2],
        target_height_m);
}

/**
 * @brief Builds pressure profile on target levels using log-linear interpolation.
 */
void apply_log_linear_pressure_profile(const SoundingData& source,
                                       const std::vector<double>& target_heights_m,
                                       const SoundingConfig& config,
                                       SoundingData& interpolated)
{
    interpolated.pressure_hpa.resize(target_heights_m.size());
    for (std::size_t i = 0; i < target_heights_m.size(); ++i)
    {
        interpolated.pressure_hpa[i] = interpolate_pressure_log_linear(source, target_heights_m[i], config);
    }
}

/**
 * @brief Checks whether a monotone PCHIP spline can be safely applied.
 */
bool can_use_monotone_spline(const std::vector<double>& x, const std::vector<double>& y)
{
    if (x.size() < 2 || y.size() != x.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
        {
            return false;
        }
        if (i > 0 && !(x[i] > x[i - 1]))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Computes endpoint tangent for monotone PCHIP interpolation.
 */
double pchip_endpoint_slope(double h0, double h1, double delta0, double delta1)
{
    const double denom = h0 + h1;
    if (!std::isfinite(denom) || denom == 0.0)
    {
        return delta0;
    }

    double m = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / denom;
    if (m * delta0 <= 0.0)
    {
        m = 0.0;
    }
    else if (std::abs(m) > 3.0 * std::abs(delta0))
    {
        m = 3.0 * delta0;
    }
    return m;
}

/**
 * @brief Computes monotone PCHIP tangents for a sampled series.
 */
std::vector<double> compute_pchip_tangents(const std::vector<double>& x, const std::vector<double>& y)
{
    const std::size_t n = x.size();
    std::vector<double> tangents(n, 0.0);
    if (n == 0)
    {
        return tangents;
    }
    if (n == 1)
    {
        tangents[0] = 0.0;
        return tangents;
    }

    std::vector<double> h(n - 1, 0.0);
    std::vector<double> delta(n - 1, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i)
    {
        h[i] = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }

    if (n == 2)
    {
        tangents[0] = delta[0];
        tangents[1] = delta[0];
        return tangents;
    }

    tangents[0] = pchip_endpoint_slope(h[0], h[1], delta[0], delta[1]);
    tangents[n - 1] = pchip_endpoint_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);

    for (std::size_t i = 1; i + 1 < n; ++i)
    {
        if (delta[i - 1] == 0.0 || delta[i] == 0.0 || delta[i - 1] * delta[i] < 0.0)
        {
            tangents[i] = 0.0;
            continue;
        }
        const double w1 = 2.0 * h[i] + h[i - 1];
        const double w2 = h[i] + 2.0 * h[i - 1];
        tangents[i] = (w1 + w2) / ((w1 / delta[i - 1]) + (w2 / delta[i]));
    }

    return tangents;
}

/**
 * @brief Evaluates one cubic Hermite segment at target abscissa.
 */
double evaluate_cubic_hermite(
    double x0,
    double x1,
    double y0,
    double y1,
    double m0,
    double m1,
    double x_target)
{
    const double h = x1 - x0;
    if (h == 0.0)
    {
        return y0;
    }
    const double t = (x_target - x0) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = (2.0 * t3) - (3.0 * t2) + 1.0;
    const double h10 = t3 - (2.0 * t2) + t;
    const double h01 = (-2.0 * t3) + (3.0 * t2);
    const double h11 = t3 - t2;
    return (h00 * y0) + (h10 * h * m0) + (h01 * y1) + (h11 * h * m1);
}

/**
 * @brief Interpolates one sounding series using monotone spline when possible.
 */
std::vector<double> interpolate_series_monotone_spline(
    const std::vector<double>& source_heights_m,
    const std::vector<double>& source_values,
    const std::vector<double>& target_heights_m,
    const SoundingConfig& config)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(target_heights_m.size(), nan);
    if (source_values.empty() || source_heights_m.empty() || source_values.size() != source_heights_m.size())
    {
        return result;
    }
    if (source_values.size() == 1)
    {
        std::fill(result.begin(), result.end(), source_values.front());
        return result;
    }

    bool use_spline = can_use_monotone_spline(source_heights_m, source_values);
    std::vector<double> tangents;
    if (use_spline)
    {
        tangents = compute_pchip_tangents(source_heights_m, source_values);
        use_spline = tangents.size() == source_values.size();
    }

    for (std::size_t i = 0; i < target_heights_m.size(); ++i)
    {
        const double target = target_heights_m[i];
        auto it = std::lower_bound(source_heights_m.begin(), source_heights_m.end(), target);

        if (it == source_heights_m.begin())
        {
            result[i] = config.extrapolate_below_ground
                ? linear_interpolate_value(
                    source_heights_m[0], source_values[0],
                    source_heights_m[1], source_values[1], target)
                : source_values.front();
            continue;
        }

        if (it == source_heights_m.end())
        {
            const std::size_t idx2 = source_heights_m.size() - 1;
            const std::size_t idx1 = idx2 - 1;
            result[i] = config.extrapolate_above_top
                ? linear_interpolate_value(
                    source_heights_m[idx1], source_values[idx1],
                    source_heights_m[idx2], source_values[idx2], target)
                : source_values.back();
            continue;
        }

        const std::size_t idx2 = static_cast<std::size_t>(it - source_heights_m.begin());
        const std::size_t idx1 = idx2 - 1;
        if (source_heights_m[idx2] == target)
        {
            result[i] = source_values[idx2];
            continue;
        }

        if (use_spline)
        {
            result[i] = evaluate_cubic_hermite(
                source_heights_m[idx1], source_heights_m[idx2],
                source_values[idx1], source_values[idx2],
                tangents[idx1], tangents[idx2],
                target);
        }
        else
        {
            result[i] = linear_interpolate_value(
                source_heights_m[idx1], source_values[idx1],
                source_heights_m[idx2], source_values[idx2],
                target);
        }
    }

    return result;
}

/**
 * @brief Interpolates an entire sounding profile with monotone spline support.
 */
SoundingData interpolate_sounding_monotone_spline(
    const SoundingData& source,
    const std::vector<double>& target_heights_m,
    const SoundingConfig& config)
{
    SoundingData result;
    result.height_m = target_heights_m;
    result.pressure_hpa = interpolate_series_monotone_spline(
        source.height_m, source.pressure_hpa, target_heights_m, config);
    result.temperature_k = interpolate_series_monotone_spline(
        source.height_m, source.temperature_k, target_heights_m, config);

    if (!source.dewpoint_k.empty() && source.dewpoint_k.size() == source.height_m.size())
    {
        result.dewpoint_k = interpolate_series_monotone_spline(
            source.height_m, source.dewpoint_k, target_heights_m, config);
    }
    if (!source.wind_speed_ms.empty() && source.wind_speed_ms.size() == source.height_m.size())
    {
        result.wind_speed_ms = interpolate_series_monotone_spline(
            source.height_m, source.wind_speed_ms, target_heights_m, config);
    }
    if (!source.wind_direction_deg.empty() && source.wind_direction_deg.size() == source.height_m.size())
    {
        result.wind_direction_deg = interpolate_series_monotone_spline(
            source.height_m, source.wind_direction_deg, target_heights_m, config);
    }

    calculate_derived_quantities(result);
    return result;
}

}

/**
 * @brief Constructs the SHARPY sounding scheme in uninitialized state.
 */
SharpySoundingScheme::SharpySoundingScheme()
    : initialized_(false) {
}

/**
 * @brief Initializes the SHARPY sounding scheme.
 */
void SharpySoundingScheme::initialize(const SoundingConfig& config) 
{
    config_ = config;
    initialized_ = true;

    std::cout << "Initialized SHARPY sounding scheme:" << std::endl;
    std::cout << "  File path: " << config_.file_path << std::endl;
    std::cout << "  Interpolation method: " <<
        (config_.interpolation_method == 0 ? "linear" :
         config_.interpolation_method == 1 ? "spline" : "log-linear") << std::endl;
    std::cout << "  Extrapolate below ground: " << (config_.extrapolate_below_ground ? "yes" : "no") << std::endl;
    std::cout << "  Extrapolate above top: " << (config_.extrapolate_above_top ? "yes" : "no") << std::endl;

    if (!config_.file_path.empty() && !std::filesystem::exists(config_.file_path)) 
    {
        std::cerr << "Warning: SHARPY file does not exist: " << config_.file_path << std::endl;

        if (!config_.use_fallback_profiles) 
        {
            throw std::runtime_error("SHARPY file not found and fallback profiles disabled");
        }
    }
}

/**
 * @brief Loads the SHARPY sounding.
 */
SoundingData SharpySoundingScheme::load_sounding(const std::string& file_path) 
{
    if (!initialized_) 
    {
        throw std::runtime_error("SHARPY sounding scheme not initialized");
    }

    std::cout << "Loading SHARPY sounding from: " << file_path << std::endl;

    if (!std::filesystem::exists(file_path)) 
    {
        throw std::runtime_error("SHARPY file not found: " + file_path);
    }

    std::string format = detect_file_format(file_path);
    std::cout << "Detected file format: " << format << std::endl;

    SoundingData data;

    try 
    {
        if (format == "hdf5")
        {
            data = read_sharpy_hdf5(file_path);
        }
        else if (format == "netcdf")
        {
            data = read_sharpy_netcdf(file_path);
        }
        else if (format == "tmv_text")
        {
            data = read_sharpy_text(file_path);
        }
        else
        {
            throw std::runtime_error("Unsupported file format: " + format);
        }

        if (!quality_control_sounding(data, config_)) 
        {
            throw std::runtime_error("Sounding failed quality control");
        }

        calculate_derived_quantities(data);

        std::cout << "Successfully loaded sounding with " << data.num_levels() << " levels" << std::endl;
        std::cout << "  Height range: " << data.height_m.front() << " - " << data.height_m.back() << " m" << std::endl;
        std::cout << "  Pressure range: " << data.pressure_hpa.front() << " - " << data.pressure_hpa.back() << " hPa" << std::endl;

    }
     catch (const std::exception& e) 
     {
        std::cerr << "Error loading SHARPY sounding: " << e.what() << std::endl;

        if (!config_.use_fallback_profiles) 
        {
            throw;
        }
        std::cerr << "Using fallback profiles as configured" << std::endl;

        data.clear();
    }

    return data;
}

/**
 * @brief Interpolates the SHARPY sounding to the target heights.
 */
SoundingData SharpySoundingScheme::interpolate_to_heights(
    const SoundingData& sounding,
    const std::vector<double>& target_heights_m) 
    {

    if (!initialized_) 
    {
        throw std::runtime_error("SHARPY sounding scheme not initialized");
    }

    if (!sounding.is_valid()) 
    {
        throw std::runtime_error("Invalid sounding data provided for interpolation");
    }

    std::cout << "Interpolating sounding to " << target_heights_m.size() << " target heights" << std::endl;

    SoundingData interpolated;
    switch (static_cast<int>(config_.interpolation_method)) 
    {
        case 0:
        default:
            interpolated = interpolate_sounding_linear(sounding, target_heights_m, config_);
            break;
        case 1:
            interpolated = interpolate_sounding_monotone_spline(sounding, target_heights_m, config_);
            break;
        case 2:
            interpolated = interpolate_sounding_linear(sounding, target_heights_m, config_);
            apply_log_linear_pressure_profile(sounding, target_heights_m, config_, interpolated);
            calculate_derived_quantities(interpolated);
            break;
    }

    return interpolated;
}

/**
 * @brief Detects the file format of the SHARPY sounding.
 */
std::string SharpySoundingScheme::detect_file_format(const std::string& file_path)
{
    std::string extension = std::filesystem::path(file_path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".h5" || extension == ".hdf5")
    {
        return "hdf5";
    } else if (extension == ".nc" || extension == ".nc4" || extension == ".netcdf") {
        return "netcdf";
    } else if (extension == ".tmv" || extension == ".txt") {
        // Text-format hint; confirm via magic string below.
    }

    std::ifstream file(file_path, std::ios::binary);

    if (!file.is_open())
    {
        return "unknown";
    }

    // Magic strings for text-based TMV sounding fixtures occupy the start of
    // the file. Read the first 22 chars (length of "TMV_SHARPY_PROFILE_V1")
    // and only fall back to the binary-format checks if the magic is absent.
    char text_magic[22];
    file.read(text_magic, 21);
    text_magic[21] = '\0';
    if (file.gcount() >= 21
        && std::string(text_magic) == "TMV_SHARPY_PROFILE_V1")
    {
        return "tmv_text";
    }
    file.clear();
    file.seekg(0);

    char buffer[8];
    file.read(buffer, 8);

    if (file.gcount() >= 8)
    {
        if (buffer[0] == '\211' && buffer[1] == 'H' && buffer[2] == 'D' &&
            buffer[3] == 'F' && buffer[4] == '\r' && buffer[5] == '\n' &&
            buffer[6] == '\032' && buffer[7] == '\n') {
            return "hdf5";
        }
        if (buffer[0] == 'C' && buffer[1] == 'D' && buffer[2] == 'F' &&
            (buffer[3] == '\001' || buffer[3] == '\002' || buffer[3] == '\005')) {
            return "netcdf";
        }
    }

    return "unknown";
}

/**
 * @brief Validates the SHARPY file.
 */
bool SharpySoundingScheme::validate_sharpy_file(const std::string& file_path) 
{
    if (!std::filesystem::exists(file_path)) 
    {
        return false;
    }

    auto file_size = std::filesystem::file_size(file_path);
    if (file_size < 1024) 
    {
        std::cerr << "Warning: SHARPY file seems too small: " << file_size << " bytes" << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Reads the SHARPY HDF5 file.
 */
SoundingData SharpySoundingScheme::read_sharpy_hdf5(const std::string& file_path) 
{
    auto finalize = [&](const NativeProfileExtraction& native, const char* backend) -> SoundingData
    {
        SoundingData data = parse_sharpy_profile(native.rows);
        apply_native_metadata(data, native);
        std::cout << "Loaded sounding via native backend: " << backend << std::endl;
        return data;
    };

    std::vector<NativeReadAttempt> attempts;
    try
    {
        return finalize(parse_hdf5_profile_native(file_path), "hdf5_api");
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "hdf5_api", e);
    }

    try
    {
        return finalize(parse_netcdf_dynamic_profile_native(file_path), "netcdf_c_api");
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "netcdf_c_api", e);
    }

    try
    {
        return finalize(parse_netcdf_classic_profile_native(file_path), "netcdf_classic");
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "netcdf_classic", e);
    }

    throw_native_ingest_failure(file_path, attempts);
}

/**
 * @brief Reads the SHARPY NetCDF file.
 */
SoundingData SharpySoundingScheme::read_sharpy_netcdf(const std::string& file_path) 
{
    auto finalize = [&](const NativeProfileExtraction& native, const char* backend) -> SoundingData
    {
        SoundingData data = parse_sharpy_profile(native.rows);
        apply_native_metadata(data, native);
        std::cout << "Loaded sounding via native backend: " << backend << std::endl;
        return data;
    };

    std::vector<NativeReadAttempt> attempts;
    try
    {
        return finalize(parse_netcdf_classic_profile_native(file_path), "netcdf_classic");
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "netcdf_classic", e);
    }

    try
    {
        return finalize(parse_netcdf_dynamic_profile_native(file_path), "netcdf_c_api");
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "netcdf_c_api", e);
    }

    try
    {
        return finalize(parse_hdf5_profile_native(file_path), "hdf5_api");
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "hdf5_api", e);
    }

    throw_native_ingest_failure(file_path, attempts);
}

/**
 * @brief Parses a TMV_SHARPY_PROFILE_V1 text file.
 *
 * The format is what `sharpy_extract.py` produces: a magic header line,
 * tab-separated metadata key/value lines (station_id, timestamp_utc,
 * latitude_deg, longitude_deg, elevation_m, levels) followed by `levels`
 * data rows of (z, p_hpa, T_k, Td_k, wspd_ms, wdir_deg). Comment lines
 * starting with '#' and blank lines are tolerated.
 *
 * Used as the canonical fixture format for end-to-end integration tests.
 */
SoundingData SharpySoundingScheme::read_sharpy_text(const std::string& file_path)
{
    std::ifstream in(file_path);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to open TMV sounding text file: " + file_path);
    }

    auto trim = [](std::string s) {
        const auto first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return std::string{};
        const auto last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    };

    std::string line;
    if (!std::getline(in, line) || trim(line) != "TMV_SHARPY_PROFILE_V1")
    {
        throw std::runtime_error("File is not TMV_SHARPY_PROFILE_V1: " + file_path);
    }

    SoundingData data;
    std::size_t expected_levels = 0;
    bool levels_seen = false;
    std::vector<std::vector<double>> rows;

    auto split_tab = [](const std::string& s) {
        std::vector<std::string> parts;
        std::string token;
        for (char c : s)
        {
            if (c == '\t')
            {
                parts.push_back(token);
                token.clear();
            }
            else
            {
                token.push_back(c);
            }
        }
        parts.push_back(token);
        return parts;
    };

    while (std::getline(in, line))
    {
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed.front() == '#')
        {
            continue;
        }
        const auto parts = split_tab(line);
        if (parts.size() < 2)
        {
            throw std::runtime_error(
                "Malformed TMV sounding line: '" + line + "' in " + file_path);
        }
        const std::string& key = parts[0];
        const std::string& value = parts[1];

        if (key == "station_id")        { data.station_id = value; }
        else if (key == "timestamp_utc"){ data.timestamp_utc = value; }
        else if (key == "latitude_deg") { data.latitude_deg = std::stod(value); }
        else if (key == "longitude_deg"){ data.longitude_deg = std::stod(value); }
        else if (key == "elevation_m")  { data.elevation_m = std::stod(value); }
        else if (key == "levels")
        {
            expected_levels = static_cast<std::size_t>(std::stoul(value));
            levels_seen = true;
            rows.reserve(expected_levels);
            // Skip the column-header row "h\tp\tt\ttd\twspd\twdir".
            if (std::getline(in, line))
            {
                const std::string header = trim(line);
                const bool is_header = (!header.empty()
                    && (header.find("\th") != std::string::npos
                        || header.front() == 'h' || header.front() == '#'));
                if (!is_header)
                {
                    // Treat the line as a data row instead.
                    const auto vals = split_tab(line);
                    if (vals.size() >= 3)
                    {
                        std::vector<double> row;
                        row.reserve(vals.size());
                        for (const auto& v : vals)
                        {
                            row.push_back(v.empty() ? std::numeric_limits<double>::quiet_NaN()
                                                    : std::stod(v));
                        }
                        rows.push_back(std::move(row));
                    }
                }
            }
        }
        else
        {
            // Unknown keys before the data table are ignored. Once levels
            // is seen, any further line should be a data row, but we
            // already advanced past the header above; the rest is handled
            // below.
            if (!levels_seen)
            {
                continue;
            }
            std::vector<double> row;
            row.reserve(parts.size());
            for (const auto& v : parts)
            {
                row.push_back(v.empty() ? std::numeric_limits<double>::quiet_NaN()
                                        : std::stod(v));
            }
            rows.push_back(std::move(row));
        }
    }

    if (!levels_seen)
    {
        throw std::runtime_error(
            "TMV sounding text file lacks 'levels' header: " + file_path);
    }
    if (rows.size() != expected_levels)
    {
        throw std::runtime_error(
            "TMV sounding row count " + std::to_string(rows.size())
            + " does not match declared levels " + std::to_string(expected_levels)
            + " in " + file_path);
    }

    SoundingData parsed = parse_sharpy_profile(rows);
    parsed.station_id = data.station_id;
    parsed.timestamp_utc = data.timestamp_utc;
    parsed.latitude_deg = data.latitude_deg;
    parsed.longitude_deg = data.longitude_deg;
    parsed.elevation_m = data.elevation_m;
    return parsed;
}


/**
 * @brief Parses the SHARPY profile data.
 */
SoundingData SharpySoundingScheme::parse_sharpy_profile(const std::vector<std::vector<double>>& profile_data) 
{
    if (profile_data.size() < 2)
    {
        throw std::runtime_error("SHARPY profile parse requires at least 2 levels");
    }

    SoundingData data;
    data.height_m.reserve(profile_data.size());
    data.pressure_hpa.reserve(profile_data.size());
    data.temperature_k.reserve(profile_data.size());
    std::vector<double> dewpoint_values;
    std::vector<double> wind_speed_values;
    std::vector<double> wind_dir_values;
    dewpoint_values.reserve(profile_data.size());
    wind_speed_values.reserve(profile_data.size());
    wind_dir_values.reserve(profile_data.size());

    bool dewpoint_all_finite = true;
    bool wind_speed_all_finite = true;
    bool wind_dir_all_finite = true;

    for (const auto& row : profile_data)
    {
        if (row.size() < 3)
        {
            throw std::runtime_error("SHARPY profile row must contain at least height/pressure/temperature");
        }
        const double h = row[0];
        const double p = row[1];
        const double t = row[2];
        if (!std::isfinite(h) || !std::isfinite(p) || !std::isfinite(t))
        {
            throw std::runtime_error("SHARPY profile has non-finite required values");
        }

        data.height_m.push_back(h);
        data.pressure_hpa.push_back(p);
        data.temperature_k.push_back(t);

        const double td = (row.size() > 3) ? row[3] : std::numeric_limits<double>::quiet_NaN();
        const double ws = (row.size() > 4) ? row[4] : std::numeric_limits<double>::quiet_NaN();
        const double wd = (row.size() > 5) ? row[5] : std::numeric_limits<double>::quiet_NaN();
        dewpoint_values.push_back(td);
        wind_speed_values.push_back(ws);
        wind_dir_values.push_back(wd);
        if (!std::isfinite(td))
        {
            dewpoint_all_finite = false;
        }
        if (!std::isfinite(ws))
        {
            wind_speed_all_finite = false;
        }
        if (!std::isfinite(wd))
        {
            wind_dir_all_finite = false;
        }
    }

    std::vector<std::size_t> order(data.height_m.size());
    for (std::size_t i = 0; i < order.size(); ++i)
    {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return data.height_m[a] < data.height_m[b];
    });

    auto reorder = [&](std::vector<double>& values)
    {
        std::vector<double> sorted(values.size());
        for (std::size_t i = 0; i < order.size(); ++i)
        {
            sorted[i] = values[order[i]];
        }
        values.swap(sorted);
    };

    reorder(data.height_m);
    reorder(data.pressure_hpa);
    reorder(data.temperature_k);
    reorder(dewpoint_values);
    reorder(wind_speed_values);
    reorder(wind_dir_values);

    if (dewpoint_all_finite)
    {
        data.dewpoint_k = std::move(dewpoint_values);
    }
    if (wind_speed_all_finite && wind_dir_all_finite)
    {
        data.wind_speed_ms = std::move(wind_speed_values);
        data.wind_direction_deg = std::move(wind_dir_values);
    }

    return data;
}
