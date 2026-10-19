#pragma once

#include "soundings/soundings_base.hpp"
#include "init/sounding/sounding_source.hpp"

#include <string>

namespace tmv::init
{

/**
 * @brief Construction parameters for FileSoundingSource.
 *
 * The path + reader scheme map directly to the existing SoundingScheme
 * factory in src/soundings/factory.cpp. file_path is the on-disk file the
 * reader is asked to open; scheme_id selects the format-specific reader
 * ("auto", native "cm1", or "sharpy" HDF5/netCDF/TMV text).
 */
struct FileSoundingParams
{
    std::string file_path;
    std::string scheme_id = "auto";
    double interpolation_method = 0.0;
    bool extrapolate_below_ground = false;
    bool extrapolate_above_top = false;
    double min_pressure_hpa = 1.0;
    double max_pressure_hpa = 1100.0;
    double min_temperature_k = 200.0;
    double max_temperature_k = 350.0;

    /// When false, build() may produce a Sounding with empty wind columns
    /// if the file lacks wind data; the runtime then falls back to the
    /// configured hodograph source. When true, missing winds throw.
    bool require_winds = true;
};

/**
 * @brief Build a self-consistent Sounding from already-interpolated data.
 *
 * Independent of file IO so it can be unit-tested without a fixture file.
 * FileSoundingSource::build() composes this helper with load + interpolate.
 *
 * Inputs:
 *   - data : a SoundingData whose vectors have been interpolated to z_m.
 *            Must contain temperature_k, mixing_ratio_kgkg (or dewpoint_k),
 *            and (if require_winds) wind_speed_ms + wind_direction_deg.
 *   - z_m  : the model heights the runtime will broadcast onto.
 *   - dz   : vertical spacing for the hydrostatic re-integration.
 *   - require_winds : if true, missing/invalid winds throw; if false,
 *                     missing winds leave Sounding::u_ms / v_ms empty.
 *
 * Behavior:
 *   - Pressure is re-integrated hydrostatically from the file's T column,
 *     starting from data.pressure_hpa[0] (converted to Pa) when valid,
 *     otherwise p0. This avoids the 1-2% pressure jump at t=0 that would
 *     otherwise seed a spurious vertical-velocity transient when the
 *     file's T and p are not in exact discrete hydrostatic balance.
 *   - theta = T * (p0 / p)^(R_d/cp) at every level.
 *   - rho  = p / (R_d * T) exactly; no independent density floor is applied.
 *   - qv   = mixing_ratio_kgkg if present and finite, otherwise derived
 *            from dewpoint via standard formulas. Valid supplied
 *            supersaturation is preserved for microphysics to process.
 *   - u_ms / v_ms in Cartesian (u_x, u_y) basis (meteorological convention:
 *     wind_direction_deg is the direction the wind blows from, so
 *     u_x = -speed * sin(dir_rad), u_y = -speed * cos(dir_rad)).
 *
 * Throws std::invalid_argument on missing/inconsistent inputs.
 */
Sounding sounding_from_data(const SoundingData& data,
                            const std::vector<double>& z_m,
                            double dz,
                            bool require_winds);

/**
 * @brief Reads a sounding file via the existing SoundingScheme infrastructure
 *        and produces a Sounding column on the model grid.
 *
 * build() pipeline:
 *   1. Create a local SoundingRuntimeState for scheme_id + file_path.
 *   2. Load through the selected native CM1/SHARPy/HDF5/netCDF reader.
 *   3. Interpolate onto z_m within that same reader instance.
 *   4. sounding_from_data() — re-integrates p hydrostatically and packs the
 *      Sounding column.
 *
 * Unless explicit above-top extrapolation is enabled, the native profile must
 * cover the model's upper interface, not merely its highest scalar center.
 * Throws std::invalid_argument for an invalid target grid and
 * std::runtime_error if the load fails or the loaded data is invalid.
 */
class FileSoundingSource final : public SoundingSource
{
public:
    explicit FileSoundingSource(FileSoundingParams params);

    Sounding build(const std::vector<double>& z_m, double dz) const override;
    std::string describe() const override;

private:
    FileSoundingParams params_;
};

}  // namespace tmv::init
