#include "sharpy_dynamic_library.hpp"
#include "sharpy_native_common.hpp"
#include "soundings/sounding_input_limits.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tmv::soundings::sharpy
{

/**
 * @brief Dynamic adapter for the NetCDF C API.
 */
struct DynamicNetcdfApi
{
    using NcOpenFn         = int (*)(const char*, int, int*);
    using NcCloseFn        = int (*)(int);
    using NcInqFn          = int (*)(int, int*, int*, int*, int*);
    using NcInqVarFn       = int (*)(int, int, char*, int*, int*, int*, int*);
    using NcInqDimFn       = int (*)(int, int, char*, std::size_t*);
    using NcGetVarDoubleFn = int (*)(int, int, double*);
    using NcInqAttFn       = int (*)(int, int, const char*, int*, std::size_t*);
    using NcGetAttTextFn   = int (*)(int, int, const char*, char*);
    using NcGetAttDoubleFn = int (*)(int, int, const char*, double*);
    using NcGetAttStringFn = int (*)(int, int, const char*, char**);
    using NcFreeStringFn   = int (*)(std::size_t, char**);
    using NcStrErrorFn     = const char* (*)(int);
    using NcInqGrpsFn      = int (*)(int, int*, int*);
    using NcInqGrpNameFn   = int (*)(int, char*);

    SharedLibraryHandle library;
    std::string         library_path;

    NcOpenFn         nc_open           = nullptr;
    NcCloseFn        nc_close          = nullptr;
    NcInqFn          nc_inq            = nullptr;
    NcInqVarFn       nc_inq_var        = nullptr;
    NcInqDimFn       nc_inq_dim        = nullptr;
    NcGetVarDoubleFn nc_get_var_double = nullptr;
    NcInqAttFn       nc_inq_att        = nullptr;
    NcGetAttTextFn   nc_get_att_text   = nullptr;
    NcGetAttDoubleFn nc_get_att_double = nullptr;
    NcGetAttStringFn nc_get_att_string = nullptr;
    NcFreeStringFn   nc_free_string    = nullptr;
    NcStrErrorFn     nc_strerror       = nullptr;
    NcInqGrpsFn      nc_inq_grps       = nullptr;
    NcInqGrpNameFn   nc_inq_grpname    = nullptr;
};

/**
 * @brief Loads the NetCDF C API dynamically (without link-time dependency).
 */
DynamicNetcdfApi load_dynamic_netcdf_api()
{
    const std::vector<std::string> library_candidates = {"netcdf.dll",
                                                         "libnetcdf.dll",
                                                         "libnetcdf.dylib",
                                                         "libnetcdf.so",
                                                         "libnetcdf.so.19",
                                                         "/opt/homebrew/lib/libnetcdf.dylib",
                                                         "/usr/local/lib/libnetcdf.dylib",
                                                         "/opt/homebrew/opt/netcdf/lib/libnetcdf.dylib",
                                                         "/usr/local/opt/netcdf/lib/libnetcdf.dylib"};

    std::string         opened_path;
    std::string         open_error;
    SharedLibraryHandle handle = open_shared_library_first(library_candidates, opened_path, open_error);
    if (!handle)
    {
        throw std::runtime_error("Unable to load NetCDF shared library: " + open_error);
    }

    DynamicNetcdfApi api;
    api.library      = std::move(handle);
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
        throw std::runtime_error("NetCDF shared library '" + api.library_path + "' is missing required symbols");
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
    NetcdfFileHandle(const DynamicNetcdfApi* api, int id) : api_(api), id_(id)
    {
    }

    NetcdfFileHandle(const NetcdfFileHandle&)            = delete;
    NetcdfFileHandle& operator=(const NetcdfFileHandle&) = delete;

    NetcdfFileHandle(NetcdfFileHandle&& other) noexcept : api_(other.api_), id_(other.id_)
    {
        other.api_ = nullptr;
        other.id_  = -1;
    }

    ~NetcdfFileHandle()
    {
        if (api_ != nullptr && id_ >= 0 && api_->nc_close != nullptr)
        {
            api_->nc_close(id_);
        }
    }

    int id() const
    {
        return id_;
    }

private:
    const DynamicNetcdfApi* api_ = nullptr;
    int                     id_  = -1;
};

/**
 * @brief Lightweight descriptor for NetCDF variables discovered via C API.
 */
struct DynamicNetcdfVariable
{
    int                      group_id = -1;
    int                      var_id   = -1;
    int                      type     = 0;
    int                      dimension_count = 0;
    std::string              short_name;
    std::string              full_name;
    std::vector<std::size_t> dim_lengths;
};

struct NetcdfProfileSelection
{
    std::array<std::optional<DynamicNetcdfVariable>, 6> variables;
    std::array<std::size_t, 6> alias_priority;

    NetcdfProfileSelection()
    {
        alias_priority.fill(std::numeric_limits<std::size_t>::max());
    }
};

bool matches_netcdf_profile_alias(const DynamicNetcdfVariable& variable, const std::string& candidate_raw)
{
    const std::string candidate = normalize_profile_lookup_key(candidate_raw);
    if (candidate.empty())
    {
        return false;
    }
    const std::string candidate_norm = normalize_profile_name(candidate);
    const std::size_t slash = candidate.find_last_of('/');
    const std::string candidate_leaf = slash == std::string::npos ? candidate : candidate.substr(slash + 1);
    const std::string full = normalize_profile_lookup_key(variable.full_name);
    const std::string short_name = normalize_profile_lookup_key(variable.short_name);
    const std::string full_norm = normalize_profile_name(full);
    const std::string short_norm = normalize_profile_name(short_name);
    return full == candidate || short_name == candidate || full_norm == candidate_norm ||
        short_norm == candidate_norm || ends_with(full, "/" + candidate_leaf) ||
        ends_with(full, "_" + candidate_leaf) || ends_with(short_name, "_" + candidate_leaf);
}

void select_netcdf_variable(const DynamicNetcdfVariable& variable,
                            NetcdfProfileSelection& selected,
                            NetcdfEnumerationStatistics& statistics)
{
    const std::array<const std::vector<std::string>*, 6> candidates{
        &profile_height_candidates(), &profile_pressure_candidates(), &profile_temperature_candidates(),
        &profile_dewpoint_candidates(), &profile_wind_speed_candidates(), &profile_wind_direction_candidates()};
    for (std::size_t field = 0; field < candidates.size(); ++field)
    {
        const auto& aliases = *candidates[field];
        const std::size_t end = std::min(aliases.size(), selected.alias_priority[field]);
        for (std::size_t alias = 0; alias < end; ++alias)
        {
            if (matches_netcdf_profile_alias(variable, aliases[alias]))
            {
                selected.variables[field] = variable;
                selected.alias_priority[field] = alias;
                break;
            }
        }
    }
    std::size_t count = 0;
    for (const auto& field : selected.variables)
    {
        count += field.has_value();
    }
    statistics.peak_selected_descriptors = std::max(statistics.peak_selected_descriptors, count);
}

void enumerate_netcdf_group_variables(const DynamicNetcdfApi& api, int group_id,
                                     const std::string& group_path,
                                     NetcdfProfileSelection& selected,
                                     NetcdfEnumerationStatistics& statistics)
{
    ++statistics.visited_groups;
    int       ndims  = 0;
    int       nvars  = 0;
    int       ngatts = 0;
    int       unlim  = -1;
    const int rc_inq = api.nc_inq(group_id, &ndims, &nvars, &ngatts, &unlim);
    if (rc_inq != 0)
    {
        throw std::runtime_error("nc_inq failed while enumerating variables: " + netcdf_status_message(api, rc_inq));
    }

    constexpr int k_name_buffer_size = 1024;
    for (int var_id = 0; var_id < nvars; ++var_id)
    {
        char var_name[k_name_buffer_size];
        std::memset(var_name, 0, sizeof(var_name));

        int       var_type  = 0;
        int       var_ndims = 0;
        int       var_natts = 0;
        const int rc_meta   = api.nc_inq_var(group_id, var_id, var_name, &var_type, &var_ndims, nullptr, &var_natts);
        ++statistics.variable_metadata_queries;
        if (rc_meta != 0)
        {
            throw std::runtime_error("nc_inq_var failed for variable metadata: " + netcdf_status_message(api, rc_meta));
        }

        DynamicNetcdfVariable desc;
        desc.group_id   = group_id;
        desc.var_id     = var_id;
        desc.type       = var_type;
        desc.dimension_count = var_ndims;
        desc.short_name = std::string(var_name);
        desc.full_name  = group_path.empty() ? desc.short_name : (group_path + "/" + desc.short_name);
        select_netcdf_variable(desc, selected, statistics);
    }
}

int override_netcdf_group_query_for_test(NetcdfGroupQueryForTest query, std::size_t ordinal,
                                       int status, int* count,
                                       const NetcdfGroupQueryOverrideForTest* override)
{
    if (override != nullptr && override->query == query && override->ordinal == ordinal)
    {
        if (count != nullptr && override->returned_count)
        {
            *count = *override->returned_count;
        }
        return override->status;
    }
    return status;
}

std::vector<int> read_netcdf_child_ids(const DynamicNetcdfApi& api, int group_id,
                                     NetcdfEnumerationStatistics& statistics,
                                     const NetcdfGroupQueryOverrideForTest* override)
{
    if (api.nc_inq_grps == nullptr || api.nc_inq_grpname == nullptr)
    {
        return {};
    }

    int child_count = 0;
    int status = api.nc_inq_grps(group_id, &child_count, nullptr);
    ++statistics.group_count_queries;
    status = override_netcdf_group_query_for_test(NetcdfGroupQueryForTest::Count,
        statistics.group_count_queries, status, &child_count, override);
    if (!netcdf_group_enumeration_available(status, netcdf_status_message(api, status)))
    {
        return {};
    }
    if (child_count < 0)
    {
        throw NetcdfInputContractError("NetCDF child group count is negative");
    }
    if (child_count == 0)
    {
        return {};
    }

    std::vector<int> child_ids(static_cast<std::size_t>(child_count), -1);
    int returned_count = child_count;
    status = api.nc_inq_grps(group_id, &returned_count, child_ids.data());
    ++statistics.group_id_queries;
    status = override_netcdf_group_query_for_test(NetcdfGroupQueryForTest::Ids,
        statistics.group_id_queries, status, &returned_count, override);
    require_netcdf_group_read(status, "child group IDs", netcdf_status_message(api, status));
    if (returned_count != child_count)
    {
        throw NetcdfInputContractError("NetCDF child group count changed while reading IDs");
    }
    return child_ids;
}

void enumerate_netcdf_variables_iterative(const DynamicNetcdfApi& api, int root_group_id,
                                         NetcdfProfileSelection& selected,
                                         NetcdfEnumerationStatistics& statistics,
                                         const NetcdfGroupQueryOverrideForTest* override)
{
    struct Frame
    {
        std::vector<int> child_ids;
        std::size_t next_child = 0;
        std::size_t path_size = 0;
    };

    std::string group_path;
    std::vector<Frame> frames;
    enumerate_netcdf_group_variables(api, root_group_id, group_path, selected, statistics);
    frames.push_back({read_netcdf_child_ids(api, root_group_id, statistics, override), 0, 0});
    std::size_t retained_child_ids = frames.back().child_ids.size();
    statistics.peak_child_id_slots = retained_child_ids;
    statistics.peak_traversal_depth = 1;
    while (!frames.empty())
    {
        auto& parent = frames.back();
        if (parent.next_child == parent.child_ids.size())
        {
            retained_child_ids -= parent.child_ids.size();
            frames.pop_back();
            group_path.resize(frames.empty() ? 0 : frames.back().path_size);
            continue;
        }
        const int child_id = parent.child_ids[parent.next_child++];
        char group_name[1024];
        std::memset(group_name, 0, sizeof(group_name));
        int status = api.nc_inq_grpname(child_id, group_name);
        ++statistics.group_name_queries;
        status = override_netcdf_group_query_for_test(NetcdfGroupQueryForTest::Name,
            statistics.group_name_queries, status, nullptr, override);
        require_netcdf_group_read(status, "child group name", netcdf_status_message(api, status));
        group_path.resize(parent.path_size);
        if (!group_path.empty())
        {
            group_path += '/';
        }
        group_path += group_name;
        statistics.peak_group_path_bytes = std::max(statistics.peak_group_path_bytes, group_path.size());
        enumerate_netcdf_group_variables(api, child_id, group_path, selected, statistics);
        auto children = read_netcdf_child_ids(api, child_id, statistics, override);
        if (retained_child_ids > std::numeric_limits<std::size_t>::max() - children.size())
        {
            throw NetcdfProfileResourceError("NetCDF retained child group count overflows");
        }
        retained_child_ids += children.size();
        statistics.peak_child_id_slots = std::max(statistics.peak_child_id_slots, retained_child_ids);
        frames.push_back({std::move(children), 0, group_path.size()});
        statistics.peak_traversal_depth = std::max(statistics.peak_traversal_depth, frames.size());
    }
}

void load_selected_netcdf_dimensions(const DynamicNetcdfApi& api, DynamicNetcdfVariable& variable,
                                     NetcdfEnumerationStatistics& statistics)
{
    if (variable.dimension_count < 0)
    {
        throw NetcdfProfileResourceError("NetCDF selected profile has a negative dimension count for '" + variable.full_name + "'");
    }
    std::vector<int> dimension_ids(static_cast<std::size_t>(variable.dimension_count), 0);
    int returned_count = variable.dimension_count;
    const int status = api.nc_inq_var(variable.group_id, variable.var_id, nullptr, nullptr, &returned_count,
        dimension_ids.empty() ? nullptr : dimension_ids.data(), nullptr);
    ++statistics.dimension_id_queries;
    if (status != 0 || returned_count != variable.dimension_count)
    {
        throw NetcdfInputContractError("Cannot inspect selected NetCDF profile dimensions for '" + variable.full_name + "'");
    }
    variable.dim_lengths.reserve(dimension_ids.size());
    for (const int dimension : dimension_ids)
    {
        std::size_t length = 0;
        const int dimension_status = api.nc_inq_dim(variable.group_id, dimension, nullptr, &length);
        ++statistics.dimension_length_queries;
        if (dimension_status != 0)
        {
            throw NetcdfInputContractError("Cannot inspect selected NetCDF profile dimension for '" + variable.full_name +
                "': " + netcdf_status_message(api, dimension_status));
        }
        variable.dim_lengths.push_back(length);
    }
}

/**
 * @brief Reads one discovered NetCDF variable and flattens values to doubles.
 */
std::vector<double> read_dynamic_netcdf_variable_as_double(const DynamicNetcdfApi&      api,
                                                           const DynamicNetcdfVariable& var,
                                                           bool                         required,
                                                           NetcdfEnumerationStatistics& statistics)
{
    std::size_t total = 1;
    if (!var.dim_lengths.empty())
    {
        for (const std::size_t dim_len : var.dim_lengths)
        {
            std::size_t updated = 0;
            if (!checked_multiply_size(total, dim_len, updated))
            {
                throw NetcdfProfileResourceError("NetCDF numeric profile size overflow for '" + var.full_name + "'");
            }
            total = updated;
            if (total > kMaxTextSoundingLevels)
            {
                throw NetcdfProfileResourceError(
                    "NetCDF numeric profile exceeds the 100,000-level safety limit for '" + var.full_name + "'");
            }
        }
    }
    if (required && total < 2)
    {
        throw NetcdfProfileResourceError("required NetCDF numeric profile has fewer than two levels for '" +
                                         var.full_name + "'");
    }
    std::size_t decoded_bytes = 0;
    if (!checked_multiply_size(total, sizeof(double), decoded_bytes) ||
        decoded_bytes > kMaxTextSoundingLevels * sizeof(double))
    {
        throw NetcdfProfileResourceError("NetCDF numeric profile decoded byte count is unbounded for '" +
                                         var.full_name + "'");
    }

    std::vector<double> values(total, 0.0);
    if (total == 0)
    {
        return values;
    }

    const int rc = api.nc_get_var_double(var.group_id, var.var_id, values.data());
    ++statistics.numeric_payload_reads;
    if (rc != 0)
    {
        throw std::runtime_error("nc_get_var_double failed for '" + var.full_name +
                                 "': " + netcdf_status_message(api, rc));
    }
    return values;
}

/**
 * @brief Reads a global NetCDF string attribute by candidate names.
 */
std::optional<std::string> read_dynamic_netcdf_global_attr_string(const DynamicNetcdfApi&         api,
                                                                  int                             ncid,
                                                                  const std::vector<std::string>& candidates,
                                                                  NetcdfEnumerationStatistics& statistics)
{
    constexpr int kNcGlobal = -1;
    constexpr int kNcChar   = 2;
    constexpr int kNcString = 12;

    for (const std::string& name : candidates)
    {
        int         attr_type = 0;
        std::size_t attr_len  = 0;
        const int   rc_inq    = api.nc_inq_att(ncid, kNcGlobal, name.c_str(), &attr_type, &attr_len);
        if (!netcdf_metadata_attribute_available(rc_inq, name, netcdf_status_message(api, rc_inq)))
        {
            continue;
        }

        if (attr_type == kNcChar)
        {
            if (attr_len > kMaxSoundingMetadataBytes)
            {
                throw NetcdfMetadataContractError("NetCDF character metadata exceeds the 1023-byte limit for '" +
                                                  name + "'");
            }
            std::string value(attr_len, '\0');
            if (!value.empty())
            {
                const int rc_read = api.nc_get_att_text(ncid, kNcGlobal, name.c_str(), value.data());
                require_netcdf_metadata_read(rc_read, name, netcdf_status_message(api, rc_read));
            }
            while (!value.empty() && value.back() == '\0')
            {
                value.pop_back();
            }
            return value;
        }

        if (attr_type == kNcString)
        {
            if (attr_len != 1)
            {
                throw NetcdfMetadataContractError("NetCDF string metadata must be scalar for '" + name + "'");
            }
            if (api.nc_get_att_string == nullptr || api.nc_free_string == nullptr)
            {
                throw NetcdfMetadataContractError("NetCDF string metadata reader is unavailable for '" + name + "'");
            }
            char* value_pointer = nullptr;
            const int rc_read = api.nc_get_att_string(ncid, kNcGlobal, name.c_str(), &value_pointer);
            const auto free_string = [&api](char* pointer)
            {
                if (pointer != nullptr)
                {
                    char* owned = pointer;
                    (void)api.nc_free_string(1, &owned);
                }
            };
            return consume_netcdf_scalar_metadata_string(value_pointer, rc_read, name,
                [&api, rc_read]
                {
                    return netcdf_status_message(api, rc_read);
                }, free_string, &statistics.metadata_string_bytes_scanned);
        }
        throw NetcdfMetadataContractError("NetCDF metadata '" + name + "' must be text");
    }

    return std::nullopt;
}

/**
 * @brief Reads a global NetCDF numeric attribute by candidate names.
 */
std::optional<double> read_dynamic_netcdf_global_attr_numeric(const DynamicNetcdfApi&         api,
                                                              int                             ncid,
                                                              const std::vector<std::string>& candidates)
{
    constexpr int kNcGlobal = -1;
    for (const std::string& name : candidates)
    {
        int         attr_type = 0;
        std::size_t attr_len  = 0;
        const int   rc_inq    = api.nc_inq_att(ncid, kNcGlobal, name.c_str(), &attr_type, &attr_len);
        if (!netcdf_metadata_attribute_available(rc_inq, name, netcdf_status_message(api, rc_inq)))
        {
            continue;
        }
        if (!netcdf_primitive_numeric_type(attr_type))
        {
            throw NetcdfMetadataContractError("NetCDF metadata '" + name + "' must be scalar numeric");
        }
        if (attr_len != 1)
        {
            throw NetcdfMetadataContractError("NetCDF numeric metadata must be scalar for '" + name + "'");
        }

        double value = std::numeric_limits<double>::quiet_NaN();
        const int rc_read = api.nc_get_att_double(ncid, kNcGlobal, name.c_str(), &value);
        require_netcdf_metadata_read(rc_read, name, netcdf_status_message(api, rc_read));
        return value;
    }

    return std::nullopt;
}

/**
 * @brief Extracts profiles using a dynamic NetCDF C API backend.
 */
NativeProfileExtraction parse_netcdf_dynamic_profile_selected(const std::string& file_path,
                                                             NetcdfEnumerationStatistics& statistics,
                                                             const NetcdfGroupQueryOverrideForTest* override = nullptr)
{
    DynamicNetcdfApi api = load_dynamic_netcdf_api();
    if (override != nullptr)
    {
        if (override->omit_group_inquiry_symbol)
        {
            api.nc_inq_grps = nullptr;
        }
        if (override->omit_group_name_symbol)
        {
            api.nc_inq_grpname = nullptr;
        }
    }

    constexpr int kNcNowrite = 0;
    int           ncid       = -1;
    const int     rc_open    = api.nc_open(file_path.c_str(), kNcNowrite, &ncid);
    if (rc_open != 0 || ncid < 0)
    {
        throw std::runtime_error("nc_open failed: " + netcdf_status_message(api, rc_open));
    }

    NetcdfFileHandle                   file_guard(&api, ncid);
    NetcdfProfileSelection selected;
    enumerate_netcdf_variables_iterative(api, ncid, selected, statistics, override);
    if (statistics.variable_metadata_queries == 0)
    {
        throw std::runtime_error("No variables found in NetCDF file");
    }

    if (!selected.variables[0] || !selected.variables[1] || !selected.variables[2])
    {
        throw std::runtime_error("Missing required profile variables in NetCDF file");
    }

    std::array<std::vector<double>, 6> values;
    for (std::size_t field = 0; field < selected.variables.size(); ++field)
    {
        if (selected.variables[field])
        {
            auto& variable = *selected.variables[field];
            statistics.selected_variable_names[field] = variable.full_name;
            load_selected_netcdf_dimensions(api, variable, statistics);
            values[field] = read_dynamic_netcdf_variable_as_double(api, variable, field < 3, statistics);
        }
    }

    NativeProfileExtraction extracted =
        build_native_profile_rows(values[0], values[1], values[2], values[3], values[4], values[5]);

    if (auto station = read_dynamic_netcdf_global_attr_string(api, ncid, {"station_id"}, statistics); station.has_value())
    {
        extracted.station_id = *station;
    }
    if (auto timestamp = read_dynamic_netcdf_global_attr_string(api, ncid, {"timestamp_utc"}, statistics); timestamp.has_value())
    {
        extracted.timestamp_utc = *timestamp;
    }
    if (auto latitude = read_dynamic_netcdf_global_attr_numeric(api, ncid, {"latitude_deg", "latitude"});
        latitude.has_value())
    {
        extracted.latitude_deg = *latitude;
    }
    if (auto longitude = read_dynamic_netcdf_global_attr_numeric(api, ncid, {"longitude_deg", "longitude"});
        longitude.has_value())
    {
        extracted.longitude_deg = *longitude;
    }
    if (auto elevation = read_dynamic_netcdf_global_attr_numeric(api, ncid, {"elevation_m", "elevation"});
        elevation.has_value())
    {
        extracted.elevation_m = *elevation;
    }

    return extracted;
}

NativeProfileExtraction parse_netcdf_dynamic_profile_native(const std::string& file_path)
{
    NetcdfEnumerationStatistics statistics;
    return parse_netcdf_dynamic_profile_selected(file_path, statistics);
}

NativeProfileExtraction parse_netcdf_dynamic_profile_with_statistics_for_test(
    const std::string& file_path, NetcdfEnumerationStatistics& statistics)
{
    NetcdfEnumerationStatistics staged;
    auto result = parse_netcdf_dynamic_profile_selected(file_path, staged);
    statistics = std::move(staged);
    return result;
}

NativeProfileExtraction parse_netcdf_dynamic_profile_with_group_override_for_test(
    const std::string& file_path, NetcdfEnumerationStatistics& statistics,
    const NetcdfGroupQueryOverrideForTest& override)
{
    NetcdfEnumerationStatistics staged;
    auto result = parse_netcdf_dynamic_profile_selected(file_path, staged, &override);
    statistics = std::move(staged);
    return result;
}

} // namespace tmv::soundings::sharpy
