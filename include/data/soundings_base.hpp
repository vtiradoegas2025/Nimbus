#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

/**
 * @file soundings_base.hpp
 * @brief Base interfaces and data structures for sounding ingestion.
 *
 * Defines raw sounding profiles, interpolation configuration, and
 * abstract loading/interpolation APIs used during model initialization.
 * Factory construction allows selecting different sounding providers.
 */

struct SoundingData
{
    std::vector<double> height_m;
    std::vector<double> pressure_hpa;
    std::vector<double> temperature_k;
    std::vector<double> dewpoint_k;
    std::vector<double> wind_speed_ms;
    std::vector<double> wind_direction_deg;
    std::vector<double> relative_humidity;
    std::vector<double> mixing_ratio_kgkg;

    std::vector<double> potential_temperature_k;
    std::vector<double> equivalent_potential_temperature_k;

    std::string station_id;
    std::string timestamp_utc;
    double latitude_deg = 0.0;
    double longitude_deg = 0.0;
    double elevation_m = 0.0;

    std::vector<bool> valid_temperature;
    std::vector<bool> valid_moisture;
    std::vector<bool> valid_wind;

    /**
     * @brief Checks whether core sounding arrays are internally consistent.
     * @return True when required vectors are non-empty and size-aligned.
     */
    bool is_valid() const
    {
        return !height_m.empty() &&
               height_m.size() == temperature_k.size() &&
               height_m.size() == pressure_hpa.size();
    }

    /**
     * @brief Returns the number of sounding levels.
     * @return Number of vertical levels.
     */
    std::size_t num_levels() const { return height_m.size(); }

    /**
     * @brief Clears all sounding arrays and metadata fields.
     */
    void clear()
    {
        height_m.clear();
        pressure_hpa.clear();
        temperature_k.clear();
        dewpoint_k.clear();
        wind_speed_ms.clear();
        wind_direction_deg.clear();
        relative_humidity.clear();
        mixing_ratio_kgkg.clear();
        potential_temperature_k.clear();
        equivalent_potential_temperature_k.clear();
        valid_temperature.clear();
        valid_moisture.clear();
        valid_wind.clear();
    }
};

struct SoundingConfig
{
    std::string scheme_id = "none";
    std::string file_path = "";
    bool use_fallback_profiles = true;
    double interpolation_method = 0;
    bool extrapolate_below_ground = false;
    bool extrapolate_above_top = false;
    double min_pressure_hpa = 100.0;
    double max_pressure_hpa = 1100.0;
    double min_temperature_k = 200.0;
    double max_temperature_k = 350.0;
};

class SoundingScheme
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~SoundingScheme() = default;

    /**
     * @brief Initializes the sounding scheme.
     * @param config Sounding configuration.
     */
    virtual void initialize(const SoundingConfig& config) = 0;

    /**
     * @brief Loads sounding data from a source path.
     * @param file_path Input source path.
     * @return Loaded sounding profile.
     */
    virtual SoundingData load_sounding(const std::string& file_path) = 0;

    /**
     * @brief Interpolates sounding data to target heights.
     * @param sounding Input sounding profile.
     * @param target_heights_m Target heights in meters.
     * @return Interpolated sounding profile.
     */
    virtual SoundingData interpolate_to_heights(const SoundingData& sounding,
                                                const std::vector<double>& target_heights_m) = 0;

    /**
     * @brief Returns current scheme configuration.
     * @return Immutable configuration reference.
     */
    virtual const SoundingConfig& get_config() const = 0;

    /**
     * @brief Reports whether the scheme is initialized.
     * @return True when scheme state is ready for use.
     */
    virtual bool is_initialized() const = 0;
};

/**
 * @brief Creates a sounding scheme by identifier.
 * @param scheme_id Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<SoundingScheme> create_sounding_scheme(const std::string& scheme_id);
