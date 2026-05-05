/**
 * @file sharpy_sounding.hpp
 * @brief Declarations for the soundings module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the soundings runtime and scheme implementations.
 * This file is part of the src/soundings subsystem.
 */

#pragma once
#include "soundings/base/soundings_base.hpp"

/**
 * @brief SHARPY sounding scheme implementation
 * Reads atmospheric sounding data from SHARPY HDF5/NetCDF files
 */
class SharpySoundingScheme : public SoundingScheme 
{
public:
    SharpySoundingScheme();
    ~SharpySoundingScheme() override = default;

    /**
     * @brief Initialize the SHARPY sounding scheme
     * @param config Sounding configuration
     */
    void initialize(const SoundingConfig& config) override;

    /**
     * @brief Load sounding data from SHARPY file
     * @param file_path Path to SHARPY HDF5/NetCDF file
     * @return SoundingData structure with loaded data
     */
    SoundingData load_sounding(const std::string& file_path) override;

    /**
     * @brief Interpolate sounding data to target heights
     * @param sounding Original sounding data
     * @param target_heights_m Target heights for interpolation
     * @return Interpolated sounding data
     */
    SoundingData interpolate_to_heights(
        const SoundingData& sounding,
        const std::vector<double>& target_heights_m) override;

    /**
     * @brief Get current configuration
     * @return Reference to current configuration
     */
    const SoundingConfig& get_config() const override { return config_; }

    /**
     * @brief Check if scheme is properly initialized
     * @return true if ready to use
     */
    bool is_initialized() const override { return initialized_; }

private:
    SoundingConfig config_;
    bool initialized_;

    /**
     * @brief Read SHARPY data from HDF5 file
     * @param file_path Path to HDF5 file
     * @return SoundingData structure
     */
    SoundingData read_sharpy_hdf5(const std::string& file_path);

    /**
     * @brief Read SHARPY data from NetCDF file
     * @param file_path Path to NetCDF file
     * @return SoundingData structure
     */
    SoundingData read_sharpy_netcdf(const std::string& file_path);

    /**
     * @brief Read SHARPY data from a TMV_SHARPY_PROFILE_V1 text file.
     *
     * The text format is what `sharpy_extract.py` produces and is the
     * fixture format used by integration tests. Header line is the magic
     * "TMV_SHARPY_PROFILE_V1", followed by tab-separated metadata lines
     * (station_id, timestamp_utc, latitude_deg, longitude_deg,
     * elevation_m, levels), then `levels` rows of
     *   z\tp_hpa\tT_k\tTd_k\twspd_ms\twdir_deg
     */
    SoundingData read_sharpy_text(const std::string& file_path);

    /**
     * @brief Parse SHARPY profile data structure
     * @param profile_data Raw profile data from file
     * @return Parsed SoundingData
     */
    SoundingData parse_sharpy_profile(const std::vector<std::vector<double>>& profile_data);

    /**
     * @brief Detect file format from extension or content
     * @param file_path Path to file
     * @return File format string ("hdf5", "netcdf", or "unknown")
     */
    std::string detect_file_format(const std::string& file_path);

    /**
     * @brief Validate SHARPY file structure
     * @param file_path Path to file
     * @return true if file appears to be valid SHARPY format
     */
    bool validate_sharpy_file(const std::string& file_path);

};
