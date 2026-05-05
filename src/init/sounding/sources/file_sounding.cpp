/**
 * @file file_sounding.cpp
 * @brief FileSoundingSource implementation. The build() pipeline reads
 *        the file via the existing SoundingScheme factory (today: SHARPY),
 *        interpolates to model heights, and then re-integrates pressure
 *        hydrostatically so the column the dynamics sees is self-consistent
 *        with the file's temperature.
 */

#include "init/sounding/file_sounding.hpp"

#include "core/physical_constants.hpp"
#include "data/soundings.hpp"

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

bool any_finite(const std::vector<double>& v)
{
    for (double x : v)
    {
        if (std::isfinite(x))
        {
            return true;
        }
    }
    return false;
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

double saturation_mixing_ratio(double T_k, double p_pa)
{
    const double e_sat = saturation_vapor_pressure_pa(T_k);
    return 0.622 * e_sat / std::max(p_pa - e_sat, 1.0);
}

/**
 * @brief Mixing ratio from dewpoint (K) and total pressure (Pa).
 *
 * w = 0.622 * e(T_d) / (p - e(T_d)), where e(T_d) is the saturation vapor
 * pressure evaluated at the dewpoint temperature. This is the standard
 * SHARPpy / Iribarne & Godson definition and matches what the existing
 * tornado_sim.cpp overlay would have produced for cells where the file
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

    if (!is_valid_sized(data.temperature_k, nz) || !any_finite(data.temperature_k))
    {
        throw std::invalid_argument(
            "sounding_from_data: SoundingData lacks a complete temperature_k column "
            "matching the model heights");
    }

    const bool has_mixing = is_valid_sized(data.mixing_ratio_kgkg, nz);
    const bool has_dewpoint = is_valid_sized(data.dewpoint_k, nz);
    if (!has_mixing && !has_dewpoint)
    {
        throw std::invalid_argument(
            "sounding_from_data: SoundingData has neither mixing_ratio_kgkg "
            "nor dewpoint_k; cannot derive moisture column");
    }

    const bool has_winds = is_valid_sized(data.wind_speed_ms, nz)
                        && is_valid_sized(data.wind_direction_deg, nz);
    if (require_winds && !has_winds)
    {
        throw std::invalid_argument(
            "sounding_from_data: SoundingData lacks complete wind_speed_ms / "
            "wind_direction_deg columns and require_winds is true");
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
    // Surface boundary: prefer the file's surface pressure (converted from
    // hPa) when present and finite; fall back to p0 = 1e5 Pa otherwise.
    // We re-integrate from there even when the file ostensibly provides a
    // pressure at every level — file pressures are not generally in
    // discrete hydrostatic balance with the file's temperature column once
    // both are interpolated, and the residual seeds a startup transient.
    const bool has_surface_pressure = is_valid_sized(data.pressure_hpa, nz)
                                   && std::isfinite(data.pressure_hpa[0])
                                   && data.pressure_hpa[0] > 0.0;
    s.T_k[0] = data.temperature_k[0];
    s.p_pa[0] = has_surface_pressure ? (data.pressure_hpa[0] * 100.0) : p0;
    s.rho_kgm3[0] = std::max(s.p_pa[0] / (R_d * s.T_k[0]), 0.1);

    for (std::size_t k = 1; k < nz; ++k)
    {
        s.T_k[k] = data.temperature_k[k];
        if (!std::isfinite(s.T_k[k]) || s.T_k[k] <= 0.0)
        {
            throw std::invalid_argument(
                "sounding_from_data: temperature_k contains non-positive or "
                "non-finite entries after interpolation");
        }
        const double T_avg = 0.5 * (s.T_k[k] + s.T_k[k - 1]);
        s.p_pa[k] = s.p_pa[k - 1] * std::exp(-g * dz / (R_d * T_avg));
        s.rho_kgm3[k] = std::max(s.p_pa[k] / (R_d * s.T_k[k]), 0.1);
    }

    const double kappa = R_d / cp;
    for (std::size_t k = 0; k < nz; ++k)
    {
        s.theta_k[k] = s.T_k[k] * std::pow(p0 / s.p_pa[k], kappa);
    }

    // Moisture column.
    constexpr double rh_cap = 0.95;
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
            qv_val = 0.0;
        }
        const double qvsat = saturation_mixing_ratio(s.T_k[k], s.p_pa[k]);
        s.qv_kgkg[k] = std::min(qv_val, qvsat * rh_cap);
    }

    // Wind columns: meteorological convention. wind_direction_deg is the
    // direction the wind blows FROM, so the (u, v) Cartesian components
    // are -speed*sin(dir) and -speed*cos(dir). Same convention used by
    // the existing tornado_sim.cpp overlay code.
    if (has_winds)
    {
        s.u_ms.resize(nz, 0.0);
        s.v_ms.resize(nz, 0.0);
        bool any_wind = false;
        for (std::size_t k = 0; k < nz; ++k)
        {
            const double speed = data.wind_speed_ms[k];
            const double dir_deg = data.wind_direction_deg[k];
            if (!std::isfinite(speed) || !std::isfinite(dir_deg))
            {
                continue;
            }
            const double dir_rad = dir_deg * kPi / 180.0;
            s.u_ms[k] = -speed * std::sin(dir_rad);
            s.v_ms[k] = -speed * std::cos(dir_rad);
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
            "(e.g. \"sharpy\")");
    }
}

Sounding FileSoundingSource::build(const std::vector<double>& z_m, double dz) const
{
    SoundingConfig cfg;
    cfg.scheme_id = params_.scheme_id;
    cfg.file_path = params_.file_path;
    cfg.use_fallback_profiles = false;  // we throw on failure here

    initialize_soundings(cfg);

    SoundingData raw;
    try
    {
        raw = load_sounding_data();
    }
    catch (const std::exception& e)
    {
        reset_soundings();
        throw std::runtime_error(
            std::string("FileSoundingSource: failed to load \"") + params_.file_path
            + "\" via scheme \"" + params_.scheme_id + "\": " + e.what());
    }
    if (!raw.is_valid())
    {
        reset_soundings();
        throw std::runtime_error(
            "FileSoundingSource: SoundingScheme returned an invalid SoundingData "
            "for \"" + params_.file_path + "\"");
    }

    SoundingData interp;
    try
    {
        interp = interpolate_sounding_to_grid(raw, z_m);
    }
    catch (const std::exception& e)
    {
        reset_soundings();
        throw std::runtime_error(
            std::string("FileSoundingSource: interpolation to model grid failed: ")
            + e.what());
    }
    reset_soundings();

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
