#include "sharpy_native_common.hpp"
#include "soundings/sounding_input_limits.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tmv::soundings::sharpy
{

/**
 * @brief Converts an ASCII string to lowercase.
 */
std::string to_lower_ascii(std::string value)
{
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
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
    int32_t                   type          = 0;
    std::size_t               element_count = 0;
    std::vector<std::uint8_t> raw;
};

/**
 * @brief NetCDF dimension descriptor.
 */
struct NetcdfDimension
{
    std::string   name;
    std::uint64_t length    = 0;
    bool          unlimited = false;
};

/**
 * @brief NetCDF variable descriptor from classic header metadata.
 */
struct NetcdfVariable
{
    std::string                            name;
    std::vector<int32_t>                   dim_ids;
    std::map<std::string, NetcdfAttribute> attributes;
    int32_t                                type       = 0;
    std::uint32_t                          vsize      = 0;
    std::uint64_t                          begin      = 0;
    bool                                   record_var = false;
};

/**
 * @brief Parsed metadata for a NetCDF classic/64-bit-offset file.
 */
struct NetcdfClassicHeader
{
    int                                    version     = 0;
    std::uint64_t                          num_records = 0;
    std::vector<NetcdfDimension>           dimensions;
    std::map<std::string, NetcdfAttribute> global_attributes;
    std::vector<NetcdfVariable>            variables;
};

constexpr int32_t kNcDimensionTag = 10;
constexpr int32_t kNcVariableTag  = 11;
constexpr int32_t kNcAttributeTag = 12;

constexpr int32_t kNcByte   = 1;
constexpr int32_t kNcChar   = 2;
constexpr int32_t kNcShort  = 3;
constexpr int32_t kNcInt    = 4;
constexpr int32_t kNcFloat  = 5;
constexpr int32_t kNcDouble = 6;
constexpr int32_t kNcUByte  = 7;
constexpr int32_t kNcUShort = 8;
constexpr int32_t kNcUInt   = 9;
constexpr int32_t kNcInt64  = 10;
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
    return (static_cast<std::uint32_t>(bytes[0]) << 24) | (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
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
    return (static_cast<std::uint64_t>(bytes[0]) << 56) | (static_cast<std::uint64_t>(bytes[1]) << 48) |
           (static_cast<std::uint64_t>(bytes[2]) << 40) | (static_cast<std::uint64_t>(bytes[3]) << 32) |
           (static_cast<std::uint64_t>(bytes[4]) << 24) | (static_cast<std::uint64_t>(bytes[5]) << 16) |
           (static_cast<std::uint64_t>(bytes[6]) << 8) | static_cast<std::uint64_t>(bytes[7]);
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
    const std::streampos name_begin = in.tellg();
    if (name_begin == std::streampos(-1))
    {
        throw std::runtime_error("Cannot determine NetCDF name position in " + context);
    }
    in.seekg(0, std::ios::end);
    if (!in)
    {
        throw std::runtime_error("Cannot determine NetCDF name bytes in " + context);
    }
    const std::streampos file_end = in.tellg();
    if (file_end == std::streampos(-1) || file_end < name_begin)
    {
        throw std::runtime_error("Cannot determine NetCDF name bytes in " + context);
    }
    in.seekg(name_begin);
    if (!in)
    {
        throw std::runtime_error("Cannot restore NetCDF name position in " + context);
    }
    const std::size_t padded_length = static_cast<std::size_t>(length) + padding_to_4(length);
    if (static_cast<std::uintmax_t>(padded_length) >
        static_cast<std::uintmax_t>(file_end - name_begin))
    {
        throw std::runtime_error("NetCDF name length exceeds available file bytes in " + context);
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
    const std::int32_t tag       = read_be_i32(in, context + " tag");
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

    const std::size_t                      count = static_cast<std::size_t>(count_raw);
    std::map<std::string, NetcdfAttribute> attrs;
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::string  name         = read_netcdf_name(in, context + " attribute");
        const std::int32_t type         = read_be_i32(in, context + " attribute type");
        const std::int32_t elements_raw = read_be_i32(in, context + " attribute size");
        if (elements_raw < 0)
        {
            throw std::runtime_error("Negative NetCDF attribute size for '" + name + "'");
        }

        const std::size_t element_count = static_cast<std::size_t>(elements_raw);
        if (context == "global attributes")
        {
            const std::string key = to_lower_ascii(name);
            if (key == "station_id" || key == "timestamp_utc")
            {
                if (type != kNcChar)
                {
                    throw NetcdfMetadataContractError("NetCDF metadata '" + name + "' must be text");
                }
                if (element_count > kMaxSoundingMetadataBytes)
                {
                    throw NetcdfMetadataContractError(
                        "NetCDF metadata '" + name + "' exceeds the 1,023-byte safety limit");
                }
            }
            if (key == "latitude_deg" || key == "latitude" ||
                key == "longitude_deg" || key == "longitude" ||
                key == "elevation_m" || key == "elevation")
            {
                if (element_count != 1 || type == kNcChar)
                {
                    throw NetcdfMetadataContractError("NetCDF metadata '" + name + "' must be scalar numeric");
                }
            }
        }
        const std::size_t type_size     = nc_type_size_bytes(type);
        std::size_t       byte_count    = 0;
        if (!checked_multiply_size(element_count, type_size, byte_count))
        {
            throw std::runtime_error("NetCDF attribute is too large: '" + name + "'");
        }
        if (static_cast<std::uintmax_t>(byte_count) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        {
            throw std::runtime_error("NetCDF attribute payload exceeds streamsize for '" + name + "'");
        }
        const std::streampos payload_begin = in.tellg();
        if (payload_begin == std::streampos(-1))
        {
            throw std::runtime_error("Cannot determine NetCDF attribute payload position for '" + name + "'");
        }
        in.seekg(0, std::ios::end);
        if (!in)
        {
            throw std::runtime_error("Cannot determine NetCDF attribute payload size for '" + name + "'");
        }
        const std::streampos file_end = in.tellg();
        if (file_end == std::streampos(-1) || file_end < payload_begin)
        {
            throw std::runtime_error("Cannot determine NetCDF attribute payload size for '" + name + "'");
        }
        in.seekg(payload_begin);
        if (!in)
        {
            throw std::runtime_error("Cannot restore NetCDF attribute payload position for '" + name + "'");
        }
        const std::streamoff remaining_bytes = file_end - payload_begin;
        if (static_cast<std::uintmax_t>(byte_count) > static_cast<std::uintmax_t>(remaining_bytes))
        {
            throw std::runtime_error("NetCDF attribute payload exceeds available file bytes for '" + name + "'");
        }

        NetcdfAttribute attr;
        attr.type          = type;
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
    const std::int32_t tag       = read_be_i32(in, "dimension list tag");
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
    const std::streampos dimensions_begin = in.tellg();
    if (dimensions_begin == std::streampos(-1))
    {
        throw std::runtime_error("Cannot determine NetCDF dimension header position");
    }
    in.seekg(0, std::ios::end);
    if (!in)
    {
        throw std::runtime_error("Cannot determine NetCDF dimension header size");
    }
    const std::streampos file_end = in.tellg();
    if (file_end == std::streampos(-1) || file_end < dimensions_begin)
    {
        throw std::runtime_error("Cannot determine NetCDF dimension header size");
    }
    in.seekg(dimensions_begin);
    if (!in)
    {
        throw std::runtime_error("Cannot restore NetCDF dimension header position");
    }
    const std::streamoff remaining_bytes = file_end - dimensions_begin;
    if (static_cast<std::uintmax_t>(count) > static_cast<std::uintmax_t>(remaining_bytes) / 8u)
    {
        throw std::runtime_error("NetCDF dimension count exceeds available header bytes");
    }
    std::vector<NetcdfDimension> dims;
    dims.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        NetcdfDimension dim;
        dim.name                      = read_netcdf_name(in, "dimension");
        const std::int32_t length_raw = read_be_i32(in, "dimension length");
        if (length_raw < 0)
        {
            throw std::runtime_error("Negative NetCDF dimension length for '" + dim.name + "'");
        }
        dim.unlimited = (length_raw == 0);
        dim.length    = dim.unlimited ? 0u : static_cast<std::uint64_t>(length_raw);
        dims.push_back(std::move(dim));
    }
    return dims;
}

/**
 * @brief Reads the NetCDF variable list from the file header.
 */
std::vector<NetcdfVariable> read_netcdf_variable_list(std::istream& in, int version)
{
    const std::int32_t tag       = read_be_i32(in, "variable list tag");
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
    const std::streampos variables_begin = in.tellg();
    if (variables_begin == std::streampos(-1))
    {
        throw std::runtime_error("Cannot determine NetCDF variable header position");
    }
    in.seekg(0, std::ios::end);
    if (!in)
    {
        throw std::runtime_error("Cannot determine NetCDF variable header size");
    }
    const std::streampos file_end = in.tellg();
    if (file_end == std::streampos(-1) || file_end < variables_begin)
    {
        throw std::runtime_error("Cannot determine NetCDF variable header size");
    }
    in.seekg(variables_begin);
    if (!in)
    {
        throw std::runtime_error("Cannot restore NetCDF variable header position");
    }
    const std::streamoff remaining_bytes = file_end - variables_begin;
    const std::uintmax_t minimum_entry_bytes = (version == 2) ? 32u : 28u;
    if (static_cast<std::uintmax_t>(count) >
        static_cast<std::uintmax_t>(remaining_bytes) / minimum_entry_bytes)
    {
        throw std::runtime_error("NetCDF variable count exceeds available header bytes");
    }
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
        const std::streampos dim_ids_begin = in.tellg();
        if (dim_ids_begin == std::streampos(-1))
        {
            throw std::runtime_error("Cannot determine NetCDF dimension ID position for variable '" + var.name + "'");
        }
        in.seekg(0, std::ios::end);
        if (!in)
        {
            throw std::runtime_error("Cannot determine NetCDF dimension ID bytes for variable '" + var.name + "'");
        }
        const std::streampos file_end = in.tellg();
        if (file_end == std::streampos(-1) || file_end < dim_ids_begin)
        {
            throw std::runtime_error("Cannot determine NetCDF dimension ID bytes for variable '" + var.name + "'");
        }
        in.seekg(dim_ids_begin);
        if (!in)
        {
            throw std::runtime_error("Cannot restore NetCDF dimension ID position for variable '" + var.name + "'");
        }
        if (static_cast<std::uintmax_t>(dim_count) >
            static_cast<std::uintmax_t>(file_end - dim_ids_begin) / sizeof(std::int32_t))
        {
            throw std::runtime_error("NetCDF variable '" + var.name +
                                     "' dimension ID count exceeds available file bytes");
        }
        var.dim_ids.reserve(dim_count);
        for (std::size_t d = 0; d < dim_count; ++d)
        {
            var.dim_ids.push_back(read_be_i32(in, "variable dimension id"));
        }

        var.attributes = read_netcdf_attribute_list(in, "variable '" + var.name + "' attributes");
        var.type       = read_be_i32(in, "variable '" + var.name + "' type");
        var.vsize      = read_be_u32(in, "variable '" + var.name + "' vsize");
        var.begin = (version == 2) ? read_be_u64(in, "variable '" + var.name + "' begin")
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
        throw std::runtime_error("Native NetCDF parser supports only classic/64-bit-offset formats (CDF1/CDF2)");
    }

    NetcdfClassicHeader header;
    header.version           = version;
    header.num_records       = static_cast<std::uint64_t>(read_be_u32(in, "numrecs"));
    header.dimensions        = read_netcdf_dimension_list(in);
    header.global_attributes = read_netcdf_attribute_list(in, "global attributes");
    header.variables         = read_netcdf_variable_list(in, version);

    for (NetcdfVariable& var : header.variables)
    {
        for (const int32_t dim_id : var.dim_ids)
        {
            if (dim_id < 0 || static_cast<std::size_t>(dim_id) >= header.dimensions.size())
            {
                throw std::runtime_error("NetCDF variable '" + var.name + "' references invalid dimension id");
            }
        }
        var.record_var =
            !var.dim_ids.empty() && header.dimensions[static_cast<std::size_t>(var.dim_ids.front())].unlimited;
    }

    return header;
}

/**
 * @brief Finds the first matching variable from prioritized candidate names.
 */
const NetcdfVariable* find_netcdf_variable(const NetcdfClassicHeader&      header,
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
            if (lower_name == key_lower || ends_with(lower_name, "/" + key_lower) ||
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
        const std::size_t      dim_id      = static_cast<std::size_t>(dim_id_raw);
        const NetcdfDimension& dim         = header.dimensions[dim_id];
        const std::uint64_t    dim_len_u64 = dim.unlimited ? header.num_records : dim.length;
        if (dim_len_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        {
            throw std::runtime_error("NetCDF dimension length is too large for variable '" + var.name + "'");
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
        const int32_t          dim_id_raw = var.dim_ids[i];
        const std::size_t      dim_id     = static_cast<std::size_t>(dim_id_raw);
        const NetcdfDimension& dim        = header.dimensions[dim_id];
        if (dim.unlimited)
        {
            throw std::runtime_error("Only leading unlimited dimension is supported for '" + var.name + "'");
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
        const std::uint16_t u = (static_cast<std::uint16_t>(bytes[0]) << 8) | static_cast<std::uint16_t>(bytes[1]);
        return static_cast<double>(static_cast<std::int16_t>(u));
    }
    case kNcUShort:
    {
        const std::uint16_t u = (static_cast<std::uint16_t>(bytes[0]) << 8) | static_cast<std::uint16_t>(bytes[1]);
        return static_cast<double>(u);
    }
    case kNcInt:
    {
        const std::uint32_t u = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                                (static_cast<std::uint32_t>(bytes[1]) << 16) |
                                (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
        return static_cast<double>(static_cast<std::int32_t>(u));
    }
    case kNcUInt:
    {
        const std::uint32_t u = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                                (static_cast<std::uint32_t>(bytes[1]) << 16) |
                                (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
        return static_cast<double>(u);
    }
    case kNcFloat:
    {
        const std::uint32_t u = (static_cast<std::uint32_t>(bytes[0]) << 24) |
                                (static_cast<std::uint32_t>(bytes[1]) << 16) |
                                (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
        float               f = 0.0f;
        std::memcpy(&f, &u, sizeof(f));
        return static_cast<double>(f);
    }
    case kNcDouble:
    {
        const std::uint64_t u =
            (static_cast<std::uint64_t>(bytes[0]) << 56) | (static_cast<std::uint64_t>(bytes[1]) << 48) |
            (static_cast<std::uint64_t>(bytes[2]) << 40) | (static_cast<std::uint64_t>(bytes[3]) << 32) |
            (static_cast<std::uint64_t>(bytes[4]) << 24) | (static_cast<std::uint64_t>(bytes[5]) << 16) |
            (static_cast<std::uint64_t>(bytes[6]) << 8) | static_cast<std::uint64_t>(bytes[7]);
        double d = 0.0;
        std::memcpy(&d, &u, sizeof(d));
        return d;
    }
    case kNcInt64:
    {
        const std::uint64_t u =
            (static_cast<std::uint64_t>(bytes[0]) << 56) | (static_cast<std::uint64_t>(bytes[1]) << 48) |
            (static_cast<std::uint64_t>(bytes[2]) << 40) | (static_cast<std::uint64_t>(bytes[3]) << 32) |
            (static_cast<std::uint64_t>(bytes[4]) << 24) | (static_cast<std::uint64_t>(bytes[5]) << 16) |
            (static_cast<std::uint64_t>(bytes[6]) << 8) | static_cast<std::uint64_t>(bytes[7]);
        return static_cast<double>(static_cast<std::int64_t>(u));
    }
    case kNcUInt64:
    {
        const std::uint64_t u =
            (static_cast<std::uint64_t>(bytes[0]) << 56) | (static_cast<std::uint64_t>(bytes[1]) << 48) |
            (static_cast<std::uint64_t>(bytes[2]) << 40) | (static_cast<std::uint64_t>(bytes[3]) << 32) |
            (static_cast<std::uint64_t>(bytes[4]) << 24) | (static_cast<std::uint64_t>(bytes[5]) << 16) |
            (static_cast<std::uint64_t>(bytes[6]) << 8) | static_cast<std::uint64_t>(bytes[7]);
        return static_cast<double>(u);
    }
    default:
        throw std::runtime_error("Unsupported NetCDF numeric type: " + std::to_string(type));
    }
}

/**
 * @brief Reads one NetCDF variable payload and returns values as doubles.
 */
std::vector<double>
read_netcdf_variable_as_double(std::ifstream& in, const NetcdfClassicHeader& header, const NetcdfVariable& var)
{
    const std::size_t type_size = nc_type_size_bytes(var.type);

    if (!var.record_var)
    {
        const std::size_t element_count = netcdf_variable_element_count(header, var);
        if (element_count > kMaxTextSoundingLevels)
        {
            throw NetcdfProfileResourceError(
                "NetCDF numeric profile exceeds the 100,000-level safety limit for '" + var.name + "'");
        }
        std::size_t       byte_count    = 0;
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

    const std::size_t   per_record_elements = netcdf_variable_elements_per_record(header, var);
    const std::uint64_t record_count_u64    = header.num_records;
    if (record_count_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::runtime_error("Record count too large in NetCDF variable '" + var.name + "'");
    }
    const std::size_t record_count   = static_cast<std::size_t>(record_count_u64);
    std::size_t       total_elements = 0;
    if (!checked_multiply_size(per_record_elements, record_count, total_elements))
    {
        throw NetcdfProfileResourceError("Record variable size overflow: '" + var.name + "'");
    }
    if (total_elements > kMaxTextSoundingLevels)
    {
        throw NetcdfProfileResourceError(
            "NetCDF numeric profile exceeds the 100,000-level safety limit for '" + var.name + "'");
    }

    std::size_t payload_bytes = 0;
    if (!checked_multiply_size(per_record_elements, type_size, payload_bytes))
    {
        throw std::runtime_error("Record payload overflow: '" + var.name + "'");
    }
    if (payload_bytes > static_cast<std::size_t>(var.vsize))
    {
        throw std::runtime_error("Record payload exceeds vsize for '" + var.name + "'");
    }

    const std::uint64_t record_stride = netcdf_record_stride_bytes(header);
    if (record_stride == 0)
    {
        throw std::runtime_error("Invalid NetCDF record stride for '" + var.name + "'");
    }

    std::vector<double>       values(total_elements);
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
const NetcdfAttribute* find_netcdf_attribute(const std::map<std::string, NetcdfAttribute>& attrs,
                                             const std::vector<std::string>&               candidates)
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
    std::string       value(reinterpret_cast<const char*>(attr.raw.data()), length);
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
std::vector<double> try_read_profile_variable(std::ifstream&                  in,
                                              const NetcdfClassicHeader&      header,
                                              const std::vector<std::string>& candidates,
                                              bool                            required,
                                              const std::string&              label)
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
    catch (const NetcdfInputContractError&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        if (required)
        {
            throw;
        }
        tmv::log_warn("Failed optional NetCDF variable '", var->name, "': ", e.what());
        return {};
    }
}

/**
 * @brief Candidate aliases for profile height fields.
 */
const std::vector<std::string>& profile_height_candidates()
{
    static const std::vector<std::string> k_candidates = {"profiles/height_m", "height_m", "height", "z", "altitude"};
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile pressure fields.
 */
const std::vector<std::string>& profile_pressure_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/pressure_hpa", "pressure_hpa", "pressure", "pres", "p"};
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile temperature fields.
 */
const std::vector<std::string>& profile_temperature_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/temperature_k", "temperature_k", "temperature", "temp", "t"};
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile dewpoint fields.
 */
const std::vector<std::string>& profile_dewpoint_candidates()
{
    static const std::vector<std::string> k_candidates = {"profiles/dewpoint_k", "dewpoint_k", "dewpoint", "td"};
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile wind-speed fields.
 */
const std::vector<std::string>& profile_wind_speed_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/wind_speed_ms", "wind_speed_ms", "wind_speed", "wspd"};
    return k_candidates;
}

/**
 * @brief Candidate aliases for profile wind-direction fields.
 */
const std::vector<std::string>& profile_wind_direction_candidates()
{
    static const std::vector<std::string> k_candidates = {
        "profiles/wind_direction_deg", "wind_direction_deg", "wind_direction", "wdir"};
    return k_candidates;
}

/**
 * @brief Fits profile vectors to the target profile length.
 */
std::vector<double> resize_profile_series(const std::vector<double>& values, std::size_t target_count)
{
    const double        nan = std::numeric_limits<double>::quiet_NaN();
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
NativeProfileExtraction build_native_profile_rows(const std::vector<double>& height,
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

    const std::vector<double> dewpoint_resized       = resize_profile_series(dewpoint, n);
    const std::vector<double> wind_speed_resized     = resize_profile_series(wind_speed, n);
    const std::vector<double> wind_direction_resized = resize_profile_series(wind_direction, n);

    NativeProfileExtraction extracted;
    extracted.rows.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        extracted.rows.push_back({height[i],
                                  pressure[i],
                                  temperature[i],
                                  dewpoint_resized[i],
                                  wind_speed_resized[i],
                                  wind_direction_resized[i]});
    }
    return extracted;
}

/**
 * @brief Extracts a vertical profile from a NetCDF classic SHARPY file.
 */
NativeProfileExtraction parse_netcdf_classic_profile_native(const std::string& file_path)
{
    const NetcdfClassicHeader header = parse_netcdf_classic_header(file_path);
    std::ifstream             in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Failed to reopen NetCDF file: " + file_path);
    }

    const std::vector<double> height =
        try_read_profile_variable(in, header, profile_height_candidates(), true, "height");
    const std::vector<double> pressure =
        try_read_profile_variable(in, header, profile_pressure_candidates(), true, "pressure");
    const std::vector<double> temperature =
        try_read_profile_variable(in, header, profile_temperature_candidates(), true, "temperature");
    const std::vector<double> dewpoint =
        try_read_profile_variable(in, header, profile_dewpoint_candidates(), false, "dewpoint");
    const std::vector<double> wind_speed =
        try_read_profile_variable(in, header, profile_wind_speed_candidates(), false, "wind_speed");
    const std::vector<double> wind_dir =
        try_read_profile_variable(in, header, profile_wind_direction_candidates(), false, "wind_direction");

    NativeProfileExtraction extracted =
        build_native_profile_rows(height, pressure, temperature, dewpoint, wind_speed, wind_dir);

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
    if (const NetcdfAttribute* longitude =
            find_netcdf_attribute(header.global_attributes, {"longitude_deg", "longitude"}))
    {
        if (auto value = decode_netcdf_attribute_numeric(*longitude); value.has_value())
        {
            extracted.longitude_deg = *value;
        }
    }
    if (const NetcdfAttribute* elevation =
            find_netcdf_attribute(header.global_attributes, {"elevation_m", "elevation"}))
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
        const std::string leaf  = (slash == std::string::npos) ? base : base.substr(slash + 1);
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
 * @brief Appends a backend failure diagnostic.
 */
void append_native_attempt_failure(std::vector<NativeReadAttempt>& attempts,
                                   std::string                     backend,
                                   const std::exception&           error)
{
    attempts.push_back(NativeReadAttempt{std::move(backend), error.what()});
}

/**
 * @brief Throws an aggregated error for failed native ingestion attempts.
 */
[[noreturn]] void throw_native_ingest_failure(const std::string&                    file_path,
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
    data.station_id    = native.station_id;
    data.timestamp_utc = native.timestamp_utc;
    data.latitude_deg  = native.latitude_deg;
    data.longitude_deg = native.longitude_deg;
    data.elevation_m   = native.elevation_m;
}

} // namespace tmv::soundings::sharpy
