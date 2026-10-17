/**
 * @file file_sounding.cpp
 * @brief FileSoundingSource implementation. The build() pipeline reads
 *        the file via the SoundingScheme factory (native CM1, SHARPy,
 *        HDF5, or netCDF through explicit/content-detected readers),
 *        interpolates to model heights, and then re-integrates pressure
 *        hydrostatically so the column the dynamics sees is self-consistent
 *        with the file's temperature.
 */

#include "init/sounding/file_sounding.hpp"

#include "core/infra/physical_constants.hpp"
#include "soundings/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tmv::init
{

namespace
{

constexpr double kPi = physical_constants::pi;

bool is_valid_sized(const std::vector<double>& v, std::size_t n)
{
    return v.size() == n;
}

double saturation_vapor_pressure_pa(double T_k)
{
    const double T_c = T_k - physical_constants::freezing_temperature_k;
    if (T_k >= physical_constants::freezing_temperature_k)
    {
        return 611.21 * std::exp((18.678 - T_c / 234.5) * T_c / (257.14 + T_c));
    }
    return 611.15 * std::exp((23.036 - T_c / 333.7) * T_c / (279.82 + T_c));
}

/**
 * @brief Mixing ratio from dewpoint (K) and total pressure (Pa).
 *
 * w = 0.622 * e(T_d) / (p - e(T_d)), where e(T_d) is the saturation vapor
 * pressure evaluated at the dewpoint temperature. This is the standard
 * SHARPpy / Iribarne & Godson definition and matches what the existing
 * nimbus_sim.cpp overlay would have produced for cells where the file
 * provided dewpoint but not mixing ratio directly.
 */
double mixing_ratio_from_dewpoint(double T_d_k, double p_pa)
{
    const double e_d = saturation_vapor_pressure_pa(T_d_k);
    return 0.622 * e_d / std::max(p_pa - e_d, 1.0);
}

}  // namespace

Sounding sounding_from_data(const SoundingData& data,
                            const std::vector<double>& z_m,
                            double dz,
                            bool require_winds)
{
    const std::size_t nz = z_m.size();
    if (nz == 0)
    {
        throw std::invalid_argument("sounding_from_data: z_m is empty");
    }
    if (!(dz > 0.0))
    {
        throw std::invalid_argument("sounding_from_data: dz must be positive");
    }
    if (!std::isfinite(z_m.front()) || z_m.front() < 0.0)
    {
        throw std::invalid_argument("sounding_from_data: model heights must be non-negative");
    }
    for (std::size_t k = 1; k < nz; ++k)
    {
        if (!std::isfinite(z_m[k]) || !(z_m[k] > z_m[k - 1]))
        {
            throw std::invalid_argument(
                "sounding_from_data: model heights must increase strictly");
        }
    }

    if (!is_valid_sized(data.temperature_k, nz)
        || !std::all_of(data.temperature_k.begin(), data.temperature_k.end(),
                       [](double value) { return std::isfinite(value) && value > 0.0; }))
    {
        throw std::invalid_argument(
            "sounding_from_data: SoundingData temperature_k must be complete, finite, positive, and match the model heights");
    }

    const bool has_mixing = is_valid_sized(data.mixing_ratio_kgkg, nz);
    const bool has_dewpoint = is_valid_sized(data.dewpoint_k, nz);
    if (!has_mixing && !has_dewpoint)
    {
        throw std::invalid_argument(
            "sounding_from_data: SoundingData has neither mixing_ratio_kgkg "
            "nor dewpoint_k; cannot derive moisture column");
    }

    const bool has_component_winds =
        is_valid_sized(data.wind_u_ms, nz)
        && is_valid_sized(data.wind_v_ms, nz);
    const bool has_polar_winds =
        is_valid_sized(data.wind_speed_ms, nz)
        && is_valid_sized(data.wind_direction_deg, nz);
    const bool has_winds = has_component_winds || has_polar_winds;
    if (require_winds && !has_winds)
    {
        throw std::invalid_argument(
            "sounding_from_data: SoundingData lacks complete Cartesian or "
            "speed/direction wind columns and require_winds is true");
    }

    Sounding s;
    s.z_m = z_m;
    s.T_k.resize(nz);
    s.theta_k.resize(nz);
    s.qv_kgkg.resize(nz);
    s.p_pa.resize(nz);
    s.rho_kgm3.resize(nz);

    // Hydrostatic re-integration.
    //
    // The input has already been interpolated to z_m, so pressure_hpa[0] is the
    // pressure at the first scalar center, not necessarily at the surface. Use
    // it directly when supplied. With no file pressure, p0 is the surface-face
    // boundary and must be integrated through z_m[0] to the first center.
    if (!data.pressure_hpa.empty() && !is_valid_sized(data.pressure_hpa, nz))
    {
        throw std::invalid_argument(
            "sounding_from_data: supplied pressure_hpa must match the model-height count");
    }
    const bool has_pressure_column = is_valid_sized(data.pressure_hpa, nz);
    if (has_pressure_column
        && (!std::isfinite(data.pressure_hpa[0]) || data.pressure_hpa[0] <= 0.0))
    {
        throw std::invalid_argument(
            "sounding_from_data: supplied first-center pressure must be positive and finite");
    }
    const bool has_first_center_pressure = has_pressure_column;
    const double kappa = R_d / cp;
    const bool complete_physical_theta =
        is_valid_sized(data.potential_temperature_k, nz)
        && std::all_of(
            data.potential_temperature_k.begin(),
            data.potential_temperature_k.end(),
            [](double value)
            {
                return std::isfinite(value) && value > 0.0;
            });
    if (data.potential_temperature_is_primary && !complete_physical_theta)
    {
        throw std::invalid_argument(
            "sounding_from_data: source marks potential temperature as "
            "primary, but the theta column is incomplete or non-physical");
    }
    const bool primary_theta =
        data.potential_temperature_is_primary && complete_physical_theta;

    if (primary_theta)
    {
        // Native theta-coordinate profiles (CM1 input_sounding) must not be
        // converted theta->T,p->theta through independently interpolated
        // columns. Keep theta exact and integrate the dry Exner hydrostatic
        // relation used by Nimbus:
        //
        //   dPi/dz = -g/(cp theta), T = theta Pi,
        //   p = p0 Pi^(1/kappa).
        //
        // Moisture remains an explicit buoyancy variable in the dynamics.
        double exner = has_first_center_pressure
            ? std::pow(data.pressure_hpa[0] * 100.0 / p0, kappa)
            : 1.0;
        if (!has_first_center_pressure && z_m[0] > 0.0)
        {
            exner -= (g / cp) * z_m[0]
                   / data.potential_temperature_k[0];
        }
        if (!std::isfinite(exner) || exner <= 0.0)
        {
            throw std::invalid_argument(
                "sounding_from_data: native-theta surface Exner pressure is invalid");
        }

        for (std::size_t k = 0; k < nz; ++k)
        {
            if (k > 0)
            {
                const double dz_layer = z_m[k] - z_m[k - 1];
                exner -= (g / cp) * dz_layer * 0.5
                       * ((1.0 / data.potential_temperature_k[k - 1])
                          + (1.0 / data.potential_temperature_k[k]));
                if (!std::isfinite(exner) || exner <= 0.0)
                {
                    throw std::invalid_argument(
                        "sounding_from_data: native-theta hydrostatic Exner "
                        "pressure became non-positive");
                }
            }
            s.theta_k[k] = data.potential_temperature_k[k];
            s.T_k[k] = s.theta_k[k] * exner;
            s.p_pa[k] = p0 * std::pow(exner, 1.0 / kappa);
            s.rho_kgm3[k] = s.p_pa[k] / (R_d * s.T_k[k]);
        }
    }
    else
    {
        s.T_k[0] = data.temperature_k[0];
        s.p_pa[0] = has_first_center_pressure
            ? (data.pressure_hpa[0] * 100.0)
            : p0 * std::exp(-g * z_m[0] / (R_d * s.T_k[0]));
        s.rho_kgm3[0] = s.p_pa[0] / (R_d * s.T_k[0]);

        for (std::size_t k = 1; k < nz; ++k)
        {
            s.T_k[k] = data.temperature_k[k];
            const double T_avg = 0.5 * (s.T_k[k] + s.T_k[k - 1]);
            const double layer_dz = z_m[k] - z_m[k - 1];
            s.p_pa[k] =
                s.p_pa[k - 1] * std::exp(-g * layer_dz / (R_d * T_avg));
            s.rho_kgm3[k] = s.p_pa[k] / (R_d * s.T_k[k]);
        }

        for (std::size_t k = 0; k < nz; ++k)
        {
            s.theta_k[k] =
                s.T_k[k] * std::pow(p0 / s.p_pa[k], kappa);
        }
    }

    // Moisture column. Preserve a valid supplied mixing ratio exactly; any
    // initial supersaturation belongs to the selected microphysics, not to a
    // hidden file-reader repair.
    for (std::size_t k = 0; k < nz; ++k)
    {
        double qv_val = std::numeric_limits<double>::quiet_NaN();
        if (has_mixing && std::isfinite(data.mixing_ratio_kgkg[k]))
        {
            qv_val = data.mixing_ratio_kgkg[k];
        }
        else if (has_dewpoint && std::isfinite(data.dewpoint_k[k]))
        {
            qv_val = mixing_ratio_from_dewpoint(data.dewpoint_k[k], s.p_pa[k]);
        }
        if (!std::isfinite(qv_val) || qv_val < 0.0)
        {
            throw std::invalid_argument(
                "sounding_from_data: every level requires a finite non-negative mixing ratio or dewpoint-derived value");
        }
        s.qv_kgkg[k] = qv_val;
    }

    // Preserve native Cartesian components when present. Otherwise convert
    // meteorological speed/direction to components. Interpolation upstream is
    // component-wise in both cases, avoiding the 0/360-degree wrap hazard.
    if (has_winds)
    {
        s.u_ms.resize(nz, 0.0);
        s.v_ms.resize(nz, 0.0);
        bool any_wind = false;
        for (std::size_t k = 0; k < nz; ++k)
        {
            if (has_component_winds)
            {
                s.u_ms[k] = data.wind_u_ms[k];
                s.v_ms[k] = data.wind_v_ms[k];
                if (!std::isfinite(s.u_ms[k]) || !std::isfinite(s.v_ms[k]))
                {
                    if (require_winds)
                    {
                        throw std::invalid_argument(
                            "sounding_from_data: required Cartesian winds "
                            "must be finite at every level");
                    }
                    s.u_ms.clear();
                    s.v_ms.clear();
                    return s;
                }
            }
            else
            {
                const double speed = data.wind_speed_ms[k];
                const double dir_deg = data.wind_direction_deg[k];
                if (!std::isfinite(speed) || speed < 0.0
                    || !std::isfinite(dir_deg))
                {
                    if (require_winds)
                    {
                        throw std::invalid_argument(
                            "sounding_from_data: required winds must be finite "
                            "with non-negative speed at every level");
                    }
                    s.u_ms.clear();
                    s.v_ms.clear();
                    return s;
                }
                const double dir_rad = dir_deg * kPi / 180.0;
                s.u_ms[k] = -speed * std::sin(dir_rad);
                s.v_ms[k] = -speed * std::cos(dir_rad);
            }
            any_wind = true;
        }
        if (!any_wind)
        {
            // Every level was non-finite. Treat as no winds rather than
            // returning a column of zeros that would silently override the
            // parametric hodograph.
            s.u_ms.clear();
            s.v_ms.clear();
            if (require_winds)
            {
                throw std::invalid_argument(
                    "sounding_from_data: every wind_speed_ms / "
                    "wind_direction_deg entry is non-finite");
            }
        }
    }

    return s;
}

FileSoundingSource::FileSoundingSource(FileSoundingParams params)
    : params_(std::move(params))
{
    if (params_.file_path.empty())
    {
        throw std::invalid_argument("FileSoundingSource: file_path is empty");
    }
    if (params_.scheme_id.empty() || params_.scheme_id == "none")
    {
        throw std::invalid_argument(
            "FileSoundingSource: scheme_id must name a registered SoundingScheme "
            "(e.g. \"auto\", \"cm1\", or \"sharpy\")");
    }
}

Sounding FileSoundingSource::build(const std::vector<double>& z_m, double dz) const
{
    if (z_m.empty())
    {
        throw std::invalid_argument(
            "FileSoundingSource: target model-height column is empty");
    }
    if (!std::isfinite(dz) || dz <= 0.0)
    {
        throw std::invalid_argument(
            "FileSoundingSource: target vertical spacing must be positive and finite");
    }
    if (!std::isfinite(z_m.front()) || z_m.front() < 0.0)
    {
        throw std::invalid_argument(
            "FileSoundingSource: target model heights must be finite and non-negative");
    }
    for (std::size_t k = 1; k < z_m.size(); ++k)
    {
        if (!std::isfinite(z_m[k]) || z_m[k] <= z_m[k - 1])
        {
            throw std::invalid_argument(
                "FileSoundingSource: target model heights must increase strictly");
        }
    }

    SoundingConfig cfg;
    cfg.scheme_id = params_.scheme_id;
    cfg.file_path = params_.file_path;
    cfg.use_fallback_profiles = false;  // we throw on failure here
    cfg.interpolation_method = params_.interpolation_method;
    cfg.extrapolate_below_ground = params_.extrapolate_below_ground;
    cfg.extrapolate_above_top = params_.extrapolate_above_top;
    cfg.min_pressure_hpa = params_.min_pressure_hpa;
    cfg.max_pressure_hpa = params_.max_pressure_hpa;
    cfg.min_temperature_k = params_.min_temperature_k;
    cfg.max_temperature_k = params_.max_temperature_k;

    SoundingRuntimeState reader_runtime;
    initialize_soundings(reader_runtime, cfg);

    SoundingData raw;
    try
    {
        raw = load_sounding_data(reader_runtime);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(
            std::string("FileSoundingSource: failed to load \"") + params_.file_path
            + "\" via scheme \"" + params_.scheme_id + "\": " + e.what());
    }
    if (!raw.is_valid())
    {
        throw std::runtime_error(
            "FileSoundingSource: SoundingScheme returned an invalid SoundingData "
            "for \"" + params_.file_path + "\"");
    }
    const double upper_half_cell_m = z_m.size() > 1
        ? 0.5 * (z_m.back() - z_m[z_m.size() - 2])
        : 0.5 * dz;
    const double model_top_interface_m = z_m.back() + upper_half_cell_m;
    const double coverage_tolerance_m = 1.0e-10 * std::max(
        {1.0, std::abs(model_top_interface_m), std::abs(raw.height_m.back())});
    if (model_top_interface_m > raw.height_m.back() + coverage_tolerance_m
        && !cfg.extrapolate_above_top)
    {
        throw std::runtime_error(
            "FileSoundingSource: model top interface ("
            + std::to_string(model_top_interface_m)
            + " m; highest scalar center "
            + std::to_string(z_m.back()) + " m) exceeds sounding top ("
            + std::to_string(raw.height_m.back())
            + " m); refusing to extrapolate an uncovered upper half-cell");
    }

    SoundingData interp;
    try
    {
        interp = interpolate_sounding_to_grid(reader_runtime, raw, z_m);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error(
            std::string("FileSoundingSource: interpolation to model grid failed: ")
            + e.what());
    }
    if (!interp.is_valid())
    {
        throw std::runtime_error(
            "FileSoundingSource: interpolated SoundingData is invalid");
    }

    return sounding_from_data(interp, z_m, dz, params_.require_winds);
}

std::string FileSoundingSource::describe() const
{
    return "file/" + params_.scheme_id;
}

}  // namespace tmv::init
