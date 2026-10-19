/**
 * @file sharpy_sounding.cpp
 * @brief SHARPY sounding scheme orchestration.
 */

#include "sharpy_sounding.hpp"

#include "sharpy_interpolation.hpp"
#include "sharpy_native_reader.hpp"
#include "sharpy_profile_parse.hpp"
#include "soundings/base/sounding_algorithms.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace tmv::soundings::sharpy;

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

    tmv::log_info("Initialized SHARPY sounding scheme:");
    tmv::log_info("  File path: ", config_.file_path);
    tmv::log_info("  Interpolation method: ",
                  config_.interpolation_method == 0 ? "linear" :
                  config_.interpolation_method == 1 ? "spline" : "log-linear");
    tmv::log_info("  Extrapolate below ground: ",
                  config_.extrapolate_below_ground ? "yes" : "no");
    tmv::log_info("  Extrapolate above top: ",
                  config_.extrapolate_above_top ? "yes" : "no");

    if (!config_.file_path.empty() && !std::filesystem::exists(config_.file_path)) 
    {
        tmv::log_warn("SHARPY file does not exist: ", config_.file_path);

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

    tmv::log_info("Loading SHARPY sounding from: ", file_path);

    if (!std::filesystem::exists(file_path)) 
    {
        throw std::runtime_error("SHARPY file not found: " + file_path);
    }

    std::string format = detect_file_format(file_path);
    tmv::log_info("Detected file format: ", format);

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

        const std::size_t loaded_levels = data.num_levels();
        if (!quality_control_sounding(data, config_))
        {
            throw std::runtime_error("Sounding failed quality control");
        }
        if (data.num_levels() != loaded_levels)
        {
            throw std::runtime_error(
                "Sounding quality control rejected one or more levels; "
                "refusing a silently shortened profile");
        }

        calculate_derived_quantities(data);

        tmv::log_info("Successfully loaded sounding with ", data.num_levels(), " levels");
        tmv::log_info("  Height range: ", data.height_m.front(), " - ",
                      data.height_m.back(), " m");
        tmv::log_info("  Pressure range: ", data.pressure_hpa.front(), " - ",
                      data.pressure_hpa.back(), " hPa");

    }
     catch (const std::exception& e) 
     {
        tmv::log_error("Error loading SHARPY sounding: ", e.what());

        if (!config_.use_fallback_profiles) 
        {
            throw;
        }
        tmv::log_warn("Using fallback profiles as configured");

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

    tmv::log_info("Interpolating sounding to ", target_heights_m.size(), " target heights");

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
        tmv::log_warn("SHARPY file seems too small: ", file_size, " bytes");
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
        SoundingData data = parse_sharpy_profile_rows(native.rows);
        apply_native_metadata(data, native);
        tmv::log_info("Loaded sounding via native backend: ", backend);
        return data;
    };

    std::vector<NativeReadAttempt> attempts;
    try
    {
        return finalize(parse_hdf5_profile_native(file_path), "hdf5_api");
    }
    catch (const Hdf5MetadataContractError&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "hdf5_api", e);
    }

    try
    {
        return finalize(parse_netcdf_dynamic_profile_native(file_path), "netcdf_c_api");
    }
    catch (const NetcdfInputContractError&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "netcdf_c_api", e);
    }

    try
    {
        return finalize(parse_netcdf_classic_profile_native(file_path), "netcdf_classic");
    }
    catch (const NetcdfInputContractError&)
    {
        throw;
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
        SoundingData data = parse_sharpy_profile_rows(native.rows);
        apply_native_metadata(data, native);
        tmv::log_info("Loaded sounding via native backend: ", backend);
        return data;
    };

    std::vector<NativeReadAttempt> attempts;
    try
    {
        return finalize(parse_netcdf_classic_profile_native(file_path), "netcdf_classic");
    }
    catch (const NetcdfInputContractError&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "netcdf_classic", e);
    }

    try
    {
        return finalize(parse_netcdf_dynamic_profile_native(file_path), "netcdf_c_api");
    }
    catch (const NetcdfInputContractError&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "netcdf_c_api", e);
    }

    try
    {
        return finalize(parse_hdf5_profile_native(file_path), "hdf5_api");
    }
    catch (const Hdf5MetadataContractError&)
    {
        throw;
    }
    catch (const std::exception& e)
    {
        append_native_attempt_failure(attempts, "hdf5_api", e);
    }

    throw_native_ingest_failure(file_path, attempts);
}

/**
 * @brief Parses a TMV_SHARPY_PROFILE_V1 text file.
 */
SoundingData SharpySoundingScheme::read_sharpy_text(const std::string& file_path)
{
    return read_sharpy_text_file(file_path);
}
