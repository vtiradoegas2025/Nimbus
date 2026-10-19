#include "sharpy_dynamic_library.hpp"
#include "sharpy_native_common.hpp"
#include "soundings/sounding_input_limits.hpp"

#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmv::soundings::sharpy
{

/**
 * @brief Dynamic adapter for HDF5 core + high-level APIs.
 */
struct DynamicHdf5Api
{
    using HidT   = long long;
    using HerrT  = int;
    using HSizeT = unsigned long long;

    using H5OpenFn                 = HerrT (*)();
    using H5FOpenFn                = HidT (*)(const char*, unsigned, HidT);
    using H5FCloseFn               = HerrT (*)(HidT);
    using H5LTGetDatasetNdimsFn    = HerrT (*)(HidT, const char*, int*);
    using H5LTGetDatasetInfoFn     = HerrT (*)(HidT, const char*, HSizeT*, int*, std::size_t*);
    using H5LTGetAttributeNdimsFn  = HerrT (*)(HidT, const char*, const char*, int*);
    using H5LTGetAttributeInfoFn   = HerrT (*)(HidT, const char*, const char*, HSizeT*, int*, std::size_t*);
    using H5LTReadDatasetDoubleFn  = HerrT (*)(HidT, const char*, double*);
    using H5LTReadDatasetStringFn  = HerrT (*)(HidT, const char*, char*);
    using H5LTGetAttributeStringFn = HerrT (*)(HidT, const char*, const char*, char*);
    using H5LTGetAttributeDoubleFn = HerrT (*)(HidT, const char*, const char*, double*);
    using H5AOpenByNameFn          = HidT (*)(HidT, const char*, const char*, HidT, HidT);
    using H5AGetTypeFn             = HidT (*)(HidT);
    using H5ACloseFn               = HerrT (*)(HidT);
    using H5DOpenFn                = HidT (*)(HidT, const char*, HidT);
    using H5DGetTypeFn             = HidT (*)(HidT);
    using H5DCloseFn               = HerrT (*)(HidT);
    using H5TIsVariableStrFn       = int (*)(HidT);
    using H5TCloseFn               = HerrT (*)(HidT);

    SharedLibraryHandle core_library;
    SharedLibraryHandle hl_library;
    std::string         core_path;
    std::string         hl_path;

    H5OpenFn                 H5open                   = nullptr;
    H5FOpenFn                H5Fopen                  = nullptr;
    H5FCloseFn               H5Fclose                 = nullptr;
    H5LTGetDatasetNdimsFn    H5LTget_dataset_ndims    = nullptr;
    H5LTGetDatasetInfoFn     H5LTget_dataset_info     = nullptr;
    H5LTGetAttributeNdimsFn  H5LTget_attribute_ndims  = nullptr;
    H5LTGetAttributeInfoFn   H5LTget_attribute_info   = nullptr;
    H5LTReadDatasetDoubleFn  H5LTread_dataset_double  = nullptr;
    H5LTReadDatasetStringFn  H5LTread_dataset_string  = nullptr;
    H5LTGetAttributeStringFn H5LTget_attribute_string = nullptr;
    H5LTGetAttributeDoubleFn H5LTget_attribute_double = nullptr;
    H5AOpenByNameFn          H5Aopen_by_name          = nullptr;
    H5AGetTypeFn             H5Aget_type              = nullptr;
    H5ACloseFn               H5Aclose                 = nullptr;
    H5DOpenFn                H5Dopen2                 = nullptr;
    H5DGetTypeFn             H5Dget_type              = nullptr;
    H5DCloseFn               H5Dclose                 = nullptr;
    H5TIsVariableStrFn       H5Tis_variable_str       = nullptr;
    H5TCloseFn               H5Tclose                 = nullptr;
};

/**
 * @brief Loads HDF5 + HDF5-HL shared libraries dynamically.
 */
DynamicHdf5Api load_dynamic_hdf5_api()
{
    std::vector<std::string> core_candidates = {"hdf5.dll",
                                                "libhdf5.dll",
                                                "libhdf5-320.dll",
                                                "libhdf5-310.dll",
                                                "libhdf5.dylib",
                                                "libhdf5.so",
                                                "libhdf5.so.310",
                                                "libhdf5_serial.so",
                                                "libhdf5_serial.so.310",
                                                "libhdf5_serial.so.103",
                                                "/opt/homebrew/lib/libhdf5.dylib",
                                                "/usr/local/lib/libhdf5.dylib",
                                                "/opt/homebrew/opt/hdf5/lib/libhdf5.dylib",
                                                "/usr/local/opt/hdf5/lib/libhdf5.dylib"};
    std::vector<std::string> hl_candidates   = {"hdf5_hl.dll",
                                                "libhdf5_hl.dll",
                                                "libhdf5_hl-320.dll",
                                                "libhdf5_hl-310.dll",
                                                "libhdf5_hl.dylib",
                                                "libhdf5_hl.so",
                                                "libhdf5_hl.so.310",
                                                "libhdf5_serial_hl.so",
                                                "libhdf5_serial_hl.so.310",
                                                "libhdf5_serial_hl.so.100",
                                                "/opt/homebrew/lib/libhdf5_hl.dylib",
                                                "/usr/local/lib/libhdf5_hl.dylib",
                                                "/opt/homebrew/opt/hdf5/lib/libhdf5_hl.dylib",
                                                "/usr/local/opt/hdf5/lib/libhdf5_hl.dylib"};
    append_packaged_windows_library_candidates(core_candidates, "libhdf5-");
    append_packaged_windows_library_candidates(hl_candidates, "libhdf5_hl-");

    std::string         core_path;
    std::string         core_error;
    SharedLibraryHandle core = open_shared_library_first(core_candidates, core_path, core_error);
    if (!core)
    {
        throw std::runtime_error("Unable to load HDF5 shared library: " + core_error);
    }

    std::string         hl_path;
    std::string         hl_error;
    SharedLibraryHandle hl = open_shared_library_first(hl_candidates, hl_path, hl_error);
    if (!hl)
    {
        throw std::runtime_error("Unable to load HDF5 HL shared library: " + hl_error);
    }

    DynamicHdf5Api api;
    api.core_library = std::move(core);
    api.hl_library   = std::move(hl);
    api.core_path    = core_path;
    api.hl_path      = hl_path;

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
    if (!load_symbol(api.hl_library.get(), "H5LTget_attribute_ndims", api.H5LTget_attribute_ndims) ||
        !load_symbol(api.hl_library.get(), "H5LTget_attribute_info", api.H5LTget_attribute_info) ||
        !load_symbol(api.core_library.get(), "H5Aopen_by_name", api.H5Aopen_by_name) ||
        !load_symbol(api.core_library.get(), "H5Aget_type", api.H5Aget_type) ||
        !load_symbol(api.core_library.get(), "H5Aclose", api.H5Aclose) ||
        !load_symbol(api.core_library.get(), "H5Dopen2", api.H5Dopen2) ||
        !load_symbol(api.core_library.get(), "H5Dget_type", api.H5Dget_type) ||
        !load_symbol(api.core_library.get(), "H5Dclose", api.H5Dclose) ||
        !load_symbol(api.core_library.get(), "H5Tis_variable_str", api.H5Tis_variable_str) ||
        !load_symbol(api.core_library.get(), "H5Tclose", api.H5Tclose))
    {
        throw Hdf5MetadataContractError("HDF5 shared library lacks checked metadata type APIs");
    }

    return api;
}

namespace
{

constexpr int kH5tInteger = 0;
constexpr int kH5tFloat = 1;
constexpr int kH5tString = 3;
constexpr std::size_t kMetadataBufferBytes = 1024;
constexpr int kH5MaxRank = 32;

bool checked_hdf5_attribute_metadata(const DynamicHdf5Api& api,
                                     DynamicHdf5Api::HidT file_id,
                                     const char* object,
                                     const std::string& name,
                                     bool string_expected)
{
    int rank = 0;
    if (api.H5LTget_attribute_ndims(file_id, object, name.c_str(), &rank) < 0)
    {
        return false;
    }
    if (rank < 0 || rank > 1)
    {
        throw Hdf5MetadataContractError("HDF5 metadata attribute has unsupported rank: " + name);
    }
    DynamicHdf5Api::HSizeT dims[1] = {1};
    int type_class = -1;
    std::size_t type_size = 0;
    if (api.H5LTget_attribute_info(file_id, object, name.c_str(),
                                   rank == 0 ? nullptr : dims, &type_class, &type_size) < 0 ||
        (rank == 1 && dims[0] != 1))
    {
        throw Hdf5MetadataContractError("HDF5 metadata attribute is not scalar: " + name);
    }
    if (string_expected)
    {
        if (type_class != kH5tString || type_size == 0 || type_size >= kMetadataBufferBytes)
        {
            throw Hdf5MetadataContractError("HDF5 string attribute exceeds the 1023-byte fixed metadata contract: " + name);
        }
        const auto attribute = api.H5Aopen_by_name(file_id, object, name.c_str(), 0, 0);
        if (attribute < 0)
        {
            throw Hdf5MetadataContractError("HDF5 string attribute type cannot be inspected: " + name);
        }
        const auto type = api.H5Aget_type(attribute);
        const int variable = type >= 0 ? api.H5Tis_variable_str(type) : -1;
        if (type >= 0)
        {
            api.H5Tclose(type);
        }
        api.H5Aclose(attribute);
        if (variable != 0)
        {
            throw Hdf5MetadataContractError("HDF5 variable-length string attribute is unsupported: " + name);
        }
    }
    else if ((type_class != kH5tInteger && type_class != kH5tFloat) || type_size == 0)
    {
        throw Hdf5MetadataContractError("HDF5 numeric attribute has an unsupported type: " + name);
    }
    return true;
}

bool checked_hdf5_string_dataset(const DynamicHdf5Api& api,
                                 DynamicHdf5Api::HidT file_id,
                                 const std::string& name)
{
    int rank = 0;
    if (api.H5LTget_dataset_ndims(file_id, name.c_str(), &rank) < 0)
    {
        return false;
    }
    if (rank < 0 || rank > 1)
    {
        throw Hdf5MetadataContractError("HDF5 string metadata dataset has unsupported rank: " + name);
    }
    DynamicHdf5Api::HSizeT dims[1] = {1};
    int type_class = -1;
    std::size_t type_size = 0;
    if (api.H5LTget_dataset_info(file_id, name.c_str(),
                                 rank == 0 ? nullptr : dims, &type_class, &type_size) < 0 ||
        (rank == 1 && dims[0] != 1))
    {
        throw Hdf5MetadataContractError("HDF5 string metadata dataset is not scalar: " + name);
    }
    if (type_class != kH5tString || type_size == 0 || type_size >= kMetadataBufferBytes)
    {
        throw Hdf5MetadataContractError("HDF5 string dataset exceeds the 1023-byte fixed metadata contract: " + name);
    }
    const auto dataset = api.H5Dopen2(file_id, name.c_str(), 0);
    if (dataset < 0)
    {
        throw Hdf5MetadataContractError("HDF5 string dataset type cannot be inspected: " + name);
    }
    const auto type = api.H5Dget_type(dataset);
    const int variable = type >= 0 ? api.H5Tis_variable_str(type) : -1;
    if (type >= 0)
    {
        api.H5Tclose(type);
    }
    api.H5Dclose(dataset);
    if (variable != 0)
    {
        throw Hdf5MetadataContractError("HDF5 variable-length string dataset is unsupported: " + name);
    }
    return true;
}

} // namespace

/**
 * @brief Reads an HDF5 dataset (matched via alias candidates) as doubles.
 */
std::vector<double> read_dynamic_hdf5_dataset_as_double(const DynamicHdf5Api&           api,
                                                        DynamicHdf5Api::HidT            file_id,
                                                        const std::vector<std::string>& candidates,
                                                        bool                            required,
                                                        const char*                     label)
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
        if (rank < 0 || rank > kH5MaxRank)
        {
            throw Hdf5ProfileResourceError("HDF5 numeric profile has unsupported rank for '" + candidate + "'");
        }

        std::vector<DynamicHdf5Api::HSizeT> dims(static_cast<std::size_t>(std::max(1, rank)), 1u);
        int                                 type_class = 0;
        std::size_t                         type_size  = 0;
        if (api.H5LTget_dataset_info(
                file_id, candidate.c_str(), rank > 0 ? dims.data() : nullptr, &type_class, &type_size) < 0)
        {
            continue;
        }
        if (type_class != kH5tInteger && type_class != kH5tFloat)
        {
            throw Hdf5ProfileResourceError("HDF5 numeric profile has unsupported dataset type for '" + candidate + "'");
        }

        std::size_t count = 1;
        if (rank > 0)
        {
            for (int i = 0; i < rank; ++i)
            {
                const auto length = dims[static_cast<std::size_t>(i)];
                if (length > static_cast<DynamicHdf5Api::HSizeT>(std::numeric_limits<std::size_t>::max()))
                {
                    throw Hdf5ProfileResourceError("HDF5 numeric profile size overflow for '" + candidate + "'");
                }
                std::size_t updated = 0;
                if (!checked_multiply_size(count, static_cast<std::size_t>(length), updated))
                {
                    throw Hdf5ProfileResourceError("HDF5 numeric profile size overflow for '" + candidate + "'");
                }
                count = updated;
                if (count > kMaxTextSoundingLevels)
                {
                    throw Hdf5ProfileResourceError("HDF5 numeric profile exceeds the 100,000-level safety limit for '" + candidate + "'");
                }
            }
        }
        if (required && count < 2)
        {
            throw Hdf5ProfileResourceError("required HDF5 numeric profile has fewer than two levels for '" + candidate + "'");
        }
        std::size_t decoded_bytes = 0;
        if (!checked_multiply_size(count, sizeof(double), decoded_bytes) ||
            decoded_bytes > kMaxTextSoundingLevels * sizeof(double))
        {
            throw Hdf5ProfileResourceError("HDF5 numeric profile decoded byte count is unbounded for '" + candidate + "'");
        }

        std::vector<double> values(count, 0.0);
        if (count > 0 && api.H5LTread_dataset_double(file_id, candidate.c_str(), values.data()) < 0)
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
std::optional<std::string> read_dynamic_hdf5_string_metadata(const DynamicHdf5Api&           api,
                                                             DynamicHdf5Api::HidT            file_id,
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
                if (!checked_hdf5_attribute_metadata(api, file_id, object, candidate, true))
                {
                    continue;
                }
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

            if (!checked_hdf5_string_dataset(api, file_id, candidate))
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
std::optional<double> read_dynamic_hdf5_numeric_metadata(const DynamicHdf5Api&           api,
                                                         DynamicHdf5Api::HidT            file_id,
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
            if (!checked_hdf5_attribute_metadata(api, file_id, object, candidate, false))
            {
                continue;
            }
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

    constexpr unsigned             kH5fAccRdonly = 0u;
    constexpr DynamicHdf5Api::HidT kH5pDefault   = 0;
    const DynamicHdf5Api::HidT     file_id       = api.H5Fopen(file_path.c_str(), kH5fAccRdonly, kH5pDefault);
    if (file_id < 0)
    {
        throw std::runtime_error("H5Fopen failed for '" + file_path + "'");
    }

    try
    {
        const std::vector<double> height =
            read_dynamic_hdf5_dataset_as_double(api, file_id, profile_height_candidates(), true, "height");
        const std::vector<double> pressure =
            read_dynamic_hdf5_dataset_as_double(api, file_id, profile_pressure_candidates(), true, "pressure");
        const std::vector<double> temperature =
            read_dynamic_hdf5_dataset_as_double(api, file_id, profile_temperature_candidates(), true, "temperature");
        const std::vector<double> dewpoint =
            read_dynamic_hdf5_dataset_as_double(api, file_id, profile_dewpoint_candidates(), false, "dewpoint");
        const std::vector<double> wind_speed =
            read_dynamic_hdf5_dataset_as_double(api, file_id, profile_wind_speed_candidates(), false, "wind_speed");
        const std::vector<double> wind_direction = read_dynamic_hdf5_dataset_as_double(
            api, file_id, profile_wind_direction_candidates(), false, "wind_direction");

        NativeProfileExtraction extracted =
            build_native_profile_rows(height, pressure, temperature, dewpoint, wind_speed, wind_direction);

        if (auto station = read_dynamic_hdf5_string_metadata(api, file_id, {"station_id"}); station.has_value())
        {
            extracted.station_id = *station;
        }
        if (auto timestamp = read_dynamic_hdf5_string_metadata(api, file_id, {"timestamp_utc"}); timestamp.has_value())
        {
            extracted.timestamp_utc = *timestamp;
        }
        if (auto latitude = read_dynamic_hdf5_numeric_metadata(api, file_id, {"latitude_deg", "latitude"});
            latitude.has_value())
        {
            extracted.latitude_deg = *latitude;
        }
        if (auto longitude = read_dynamic_hdf5_numeric_metadata(api, file_id, {"longitude_deg", "longitude"});
            longitude.has_value())
        {
            extracted.longitude_deg = *longitude;
        }
        if (auto elevation = read_dynamic_hdf5_numeric_metadata(api, file_id, {"elevation_m", "elevation"});
            elevation.has_value())
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

} // namespace tmv::soundings::sharpy
