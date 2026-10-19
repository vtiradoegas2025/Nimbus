#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_constants.hpp"
#include "diagnostics/field_contract.hpp"

/**
 * @file derived_field_math.hpp
 * @brief Reusable scalar math for derived diagnostic/export fields.
 */

namespace tmv
{
namespace derived
{

inline constexpr const char* cold_pool_boundary_algorithm_id =
    "surface_density_potential_temperature_deficit_v1";

/**
 * Exact density potential temperature for Nimbus's dry-air mixing-ratio
 * storage basis. Every condensate contributes mass without gas pressure.
 * Invalid primary state is returned as NaN so publication cannot silently
 * turn a broken cell into a plausible cold-pool signal.
 */
inline float density_potential_temperature(float theta_k,
                                           float qv,
                                           float qc,
                                           float qr,
                                           float qi,
                                           float qs,
                                           float qg,
                                           float qh)
{
    const float species[] = {qv, qc, qr, qi, qs, qg, qh};
    if (!std::isfinite(theta_k) || theta_k <= 0.0f)
        return std::numeric_limits<float>::quiet_NaN();
    double total_water = 0.0;
    for (float value : species)
    {
        if (!std::isfinite(value) || value < 0.0f)
            return std::numeric_limits<float>::quiet_NaN();
        total_water += static_cast<double>(value);
    }
    constexpr double epsilon =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::gas_constant_water_vapor_jkgk;
    const double result = static_cast<double>(theta_k) *
        (1.0 + static_cast<double>(qv) / epsilon) /
        (1.0 + total_water);
    return std::isfinite(result)
        ? static_cast<float>(result)
        : std::numeric_limits<float>::quiet_NaN();
}

inline float density_potential_temperature_deficit(
    float theta_k,
    float qv,
    float qc,
    float qr,
    float qi,
    float qs,
    float qg,
    float qh,
    float base_theta_k,
    float base_qv)
{
    const float local = density_potential_temperature(
        theta_k, qv, qc, qr, qi, qs, qg, qh);
    const float reference = density_potential_temperature(
        base_theta_k, base_qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    if (!std::isfinite(local) || !std::isfinite(reference))
        return std::numeric_limits<float>::quiet_NaN();
    return std::max(0.0f, reference - local);
}

/**
 * Compute one horizontal slice of the qualified cold-pool diagnostic from
 * explicit level-owned fields. The compact NR result is assembled into a
 * Surface2D map by the tranche writer; no column-parcel diagnostics are
 * evaluated along this path.
 */
template <typename ReferenceAtRadius>
inline bool fill_density_potential_temperature_deficit_surface_slice_impl(
    const Field3D& theta,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    int nr,
    int nth,
    int nz,
    int theta_index,
    int surface_k,
    ReferenceAtRadius reference_at_radius,
    std::vector<float>& out,
    std::string& error_message)
{
    if (nr <= 0 || nth <= 0 || nz <= 0 || theta_index < 0 ||
        theta_index >= nth || surface_k < 0 || surface_k >= nz)
    {
        error_message = "cold_pool_boundary received invalid diagnostic indices";
        return false;
    }
    const Field3D* required_fields[] = {
        &theta, &qv, &qc, &qr, &qi, &qs, &qg, &qh};
    for (const Field3D* field : required_fields)
    {
        if (field->size_r() != nr || field->size_th() != nth ||
            field->size_z() != nz)
        {
            error_message =
                "cold_pool_boundary primary-state shape does not match the diagnostic grid";
            return false;
        }
    }
    out.resize(static_cast<std::size_t>(nr));
    for (int i = 0; i < nr; ++i)
    {
        const auto reference = reference_at_radius(i);
        const float base_theta_k = reference.first;
        const float base_qv = reference.second;
        if (!std::isfinite(density_potential_temperature(
                base_theta_k, base_qv, 0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f)))
        {
            error_message =
                "cold_pool_boundary received an invalid initialized base state at i=" +
                std::to_string(i);
            return false;
        }
        const float deficit = density_potential_temperature_deficit(
            theta[i][theta_index][surface_k],
            qv[i][theta_index][surface_k],
            qc[i][theta_index][surface_k],
            qr[i][theta_index][surface_k],
            qi[i][theta_index][surface_k],
            qs[i][theta_index][surface_k],
            qg[i][theta_index][surface_k],
            qh[i][theta_index][surface_k],
            base_theta_k,
            base_qv);
        if (!std::isfinite(deficit))
        {
            error_message =
                "cold_pool_boundary encountered invalid primary state at i=" +
                std::to_string(i) + ", j=" + std::to_string(theta_index) +
                ", k=" + std::to_string(surface_k);
            return false;
        }
        out[static_cast<std::size_t>(i)] = deficit;
    }
    return true;
}

inline bool fill_density_potential_temperature_deficit_surface_slice(
    const Field3D& theta,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    int nr,
    int nth,
    int nz,
    int theta_index,
    int surface_k,
    float base_theta_k,
    float base_qv,
    std::vector<float>& out,
    std::string& error_message)
{
    return fill_density_potential_temperature_deficit_surface_slice_impl(
        theta, qv, qc, qr, qi, qs, qg, qh,
        nr, nth, nz, theta_index, surface_k,
        [base_theta_k, base_qv](int)
        {
            return std::pair<float, float>{base_theta_k, base_qv};
        },
        out, error_message);
}

inline bool fill_density_potential_temperature_deficit_surface_slice(
    const Field3D& theta,
    const Field3D& qv,
    const Field3D& qc,
    const Field3D& qr,
    const Field3D& qi,
    const Field3D& qs,
    const Field3D& qg,
    const Field3D& qh,
    int nr,
    int nth,
    int nz,
    int theta_index,
    int surface_k,
    const std::vector<float>& base_theta_by_radius,
    const std::vector<float>& base_qv_by_radius,
    std::vector<float>& out,
    std::string& error_message)
{
    if (base_theta_by_radius.size() != static_cast<std::size_t>(nr)
        || base_qv_by_radius.size() != static_cast<std::size_t>(nr))
    {
        error_message =
            "cold_pool_boundary terrain reference shape does not match the diagnostic slice";
        return false;
    }
    return fill_density_potential_temperature_deficit_surface_slice_impl(
        theta, qv, qc, qr, qi, qs, qg, qh,
        nr, nth, nz, theta_index, surface_k,
        [&](int i)
        {
            const std::size_t index = static_cast<std::size_t>(i);
            return std::pair<float, float>{base_theta_by_radius[index],
                                           base_qv_by_radius[index]};
        },
        out, error_message);
}

/** Horizontal Okubo-Weiss parameter W = s_n^2 + s_s^2 - zeta^2.
 * Negative values are rotation dominated; positive values are strain dominated.
 * Inputs are physical Cartesian derivatives (or their metric-aware cylindrical
 * equivalents prepared by the caller). */
inline float okubo_weiss_parameter(float du_dx,
                                   float dv_dy,
                                   float dv_dx,
                                   float du_dy,
                                   float vertical_vorticity)
{
    if (!std::isfinite(du_dx) || !std::isfinite(dv_dy) ||
        !std::isfinite(dv_dx) || !std::isfinite(du_dy) ||
        !std::isfinite(vertical_vorticity))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float normal_strain = du_dx - dv_dy;
    const float shear_strain  = dv_dx + du_dy;
    return (normal_strain * normal_strain) +
           (shear_strain * shear_strain) -
           (vertical_vorticity * vertical_vorticity);
}

// Convert linear radar reflectivity factor Z (mm^6/m^3) to dBZ = 10*log10 Z,
// flooring non-finite / non-positive input to dbz_min and clamping the result.
inline float linear_z_to_dbz(float z_linear, float dbz_min, float dbz_max)
{
    if (!std::isfinite(z_linear) || z_linear <= 0.0f)
    {
        return dbz_min;
    }
    const float dbz = 10.0f * std::log10(z_linear);
    if (!std::isfinite(dbz))
    {
        return dbz_min;
    }
    return std::max(dbz_min, std::min(dbz_max, dbz));
}

// Reflectivity display bounds from the field contract (defaults -30 / 120 dBZ).
inline void reflectivity_dbz_bounds(float& dbz_min, float& dbz_max)
{
    dbz_min = -30.0f;
    dbz_max = 120.0f;
    if (const tmv::FieldContract* contract = tmv::find_field_contract("reflectivity_dbz"))
    {
        if (contract->default_bounds.has_min)
        {
            dbz_min = static_cast<float>(contract->default_bounds.min_value);
        }
        if (contract->default_bounds.has_max)
        {
            dbz_max = static_cast<float>(contract->default_bounds.max_value);
        }
    }
}

inline void fill_reflectivity_dbz_field(const Field3D& linear_reflectivity,
                                        Field3D& reflectivity_dbz,
                                        int nr,
                                        int nth,
                                        int nz,
                                        float dbz_min,
                                        float dbz_max)
{
    reflectivity_dbz.resize(nr, nth, nz);
    for (int i = 0; i < nr; ++i)
    {
        for (int th = 0; th < nth; ++th)
        {
            for (int k = 0; k < nz; ++k)
            {
                reflectivity_dbz(i, th, k) = linear_z_to_dbz(
                    static_cast<float>(linear_reflectivity(i, th, k)),
                    dbz_min,
                    dbz_max);
            }
        }
    }
}

inline float temperature_from_theta_and_pressure(float theta_k, float pressure_pa)
{
    if (!std::isfinite(theta_k) || !std::isfinite(pressure_pa) || pressure_pa <= 0.0f)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return theta_k * std::pow(pressure_pa / static_cast<float>(p0),
                              static_cast<float>(R_d / cp));
}

inline float sample_field_at_height(const Field3D& field,
                                    int i,
                                    int th,
                                    float z_target_m,
                                    int nz,
                                    double dz_m,
                                    int k_begin = 0,
                                    int k_end = -1)
{
    if (nz <= 0)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    k_begin = std::clamp(k_begin, 0, nz);
    k_end = (k_end < 0) ? nz : std::clamp(k_end, k_begin, nz);
    if (k_begin >= k_end)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (k_end - k_begin == 1 || dz_m <= 0.0 || !std::isfinite(z_target_m))
    {
        return static_cast<float>(field[i][th][k_begin]);
    }

    const float z_index = static_cast<float>(
        tmv::grid_coordinates::scalar_fractional_index(z_target_m, dz_m));
    if (z_index <= static_cast<float>(k_begin))
    {
        return static_cast<float>(field[i][th][k_begin]);
    }
    if (z_index >= static_cast<float>(k_end - 1))
    {
        return static_cast<float>(field[i][th][k_end - 1]);
    }

    const int k0 = std::clamp(static_cast<int>(std::floor(z_index)), k_begin, k_end - 1);
    const int k1 = std::min(k0 + 1, k_end - 1);
    const float frac = std::clamp(z_index - static_cast<float>(k0), 0.0f, 1.0f);
    const float v0 = static_cast<float>(field[i][th][k0]);
    const float v1 = static_cast<float>(field[i][th][k1]);
    if (!std::isfinite(v0) && !std::isfinite(v1))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (!std::isfinite(v0))
    {
        return v1;
    }
    if (!std::isfinite(v1))
    {
        return v0;
    }
    return v0 + frac * (v1 - v0);
}

inline float lcl_height_from_temperature_dewpoint(float temperature_k,
                                                  float dewpoint_k,
                                                  int nz,
                                                  double dz_m,
                                                  int k_begin = 0,
                                                  int k_end = -1)
{
    constexpr float t_freezing_k = 273.15f;
    if (!std::isfinite(temperature_k) || !std::isfinite(dewpoint_k))
    {
        return 0.0f;
    }
    const float t_c = temperature_k - t_freezing_k;
    const float td_c = dewpoint_k - t_freezing_k;
    const float lcl = 125.0f * std::max(0.0f, t_c - td_c);
    k_begin = std::clamp(k_begin, 0, std::max(0, nz));
    k_end = (k_end < 0) ? nz : std::clamp(k_end, k_begin, std::max(0, nz));
    if (k_begin >= k_end)
    {
        return 0.0f;
    }
    const double safe_dz = std::max(1.0, dz_m);
    const float z_top = static_cast<float>((k_end) * safe_dz);
    return std::clamp(lcl, 0.0f, z_top);
}

template <typename ValueAtRadius>
inline void fill_slice_from_radius(std::vector<float>& out,
                                   int nr,
                                   int nz,
                                   const ValueAtRadius& value_at_radius)
{
    out.resize(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz));
    for (int i = 0; i < nr; ++i)
    {
        const float value = value_at_radius(i);
        for (int k = 0; k < nz; ++k)
        {
            out[static_cast<std::size_t>(k) * static_cast<std::size_t>(nr) +
                static_cast<std::size_t>(i)] = value;
        }
    }
}

inline float mean_level_component(const std::vector<float>& level_means,
                                  float top_m,
                                  int nz,
                                  double dz_m,
                                  int k_begin = 0,
                                  int k_end = -1)
{
    if (level_means.empty() || nz <= 0 || dz_m <= 0.0)
    {
        return 0.0f;
    }
    k_begin = std::clamp(k_begin, 0, nz);
    k_end = (k_end < 0) ? nz : std::clamp(k_end, k_begin, nz);
    if (k_begin >= k_end || level_means.size() < static_cast<std::size_t>(k_end))
    {
        return 0.0f;
    }
    const int top_k = std::max(k_begin, std::min(
        k_end - 1,
        static_cast<int>(std::floor(
            tmv::grid_coordinates::scalar_fractional_index(top_m, dz_m)))));
    double sum = 0.0;
    int count = 0;
    for (int k = k_begin; k <= top_k; ++k)
    {
        const float sample = level_means[static_cast<std::size_t>(k)];
        if (std::isfinite(sample))
        {
            sum += static_cast<double>(sample);
            ++count;
        }
    }
    if (count <= 0)
    {
        return 0.0f;
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

inline float dewpoint_from_temp_qv_pressure(float temperature_k, float qv_kgkg, float pressure_pa)
{
    constexpr float t_freezing_k = 273.15f;
    if (!std::isfinite(temperature_k) || !std::isfinite(qv_kgkg) ||
        !std::isfinite(pressure_pa) || pressure_pa <= 0.0f)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const float p_hpa = pressure_pa * 0.01f;
    float es_hpa = 6.112f * std::exp(
        17.67f * (temperature_k - t_freezing_k) / (temperature_k - 29.65f));
    if (!std::isfinite(es_hpa) || p_hpa <= 1.0e-3f)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    es_hpa = std::max(0.0f, std::min(es_hpa, 0.99f * p_hpa));
    const float denom = std::max(1.0e-3f, p_hpa - es_hpa);
    const float qsat = std::max(1.0e-8f, std::min(0.10f, 0.622f * es_hpa / denom));
    if (!std::isfinite(qsat) || qsat <= 0.0f)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const float rh = std::max(0.0f, std::min(200.0f, 100.0f * (qv_kgkg / qsat)));
    const float rh_frac = std::max(1.0e-6f, rh * 0.01f);
    const float temperature_c = temperature_k - t_freezing_k;
    const float gamma_val = std::log(rh_frac) +
        (17.625f * temperature_c) / (243.04f + temperature_c);
    const float gamma_denom = 17.625f - gamma_val;
    if (std::abs(gamma_denom) <= 1.0e-6f)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const float td_c = 243.04f * gamma_val / gamma_denom;
    return td_c + t_freezing_k;
}

inline float saturation_mixing_ratio_from_temp_pressure(float temperature_k, float pressure_pa)
{
    constexpr float t_freezing_k = 273.15f;
    if (!std::isfinite(temperature_k) || !std::isfinite(pressure_pa) || pressure_pa <= 0.0f)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const float p_hpa = pressure_pa * 0.01f;
    float es_hpa = 6.112f * std::exp(
        17.67f * (temperature_k - t_freezing_k) / std::max(1.0e-3f, (temperature_k - 29.65f)));
    if (!std::isfinite(es_hpa))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    es_hpa = std::max(0.0f, std::min(es_hpa, 0.99f * p_hpa));
    const float denom = std::max(1.0e-3f, p_hpa - es_hpa);
    const float qsat = 0.622f * es_hpa / denom;
    if (!std::isfinite(qsat))
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::max(0.0f, qsat);
}

struct ThermodynamicDerivedScalars
{
    float temperature_k = std::numeric_limits<float>::quiet_NaN();
    float saturation_mixing_ratio_kgkg = std::numeric_limits<float>::quiet_NaN();
    float relative_humidity_pct = std::numeric_limits<float>::quiet_NaN();
    float dewpoint_k = std::numeric_limits<float>::quiet_NaN();
};

inline ThermodynamicDerivedScalars compute_thermodynamic_derived_scalars(
    float theta_k,
    float pressure_pa,
    float qv_kgkg)
{
    constexpr float t_freezing_k = 273.15f;
    ThermodynamicDerivedScalars out;
    out.temperature_k = temperature_from_theta_and_pressure(theta_k, pressure_pa);
    if (!std::isfinite(out.temperature_k) || !std::isfinite(pressure_pa) || pressure_pa <= 0.0f)
    {
        return out;
    }

    const float p_hpa = pressure_pa * 0.01f;
    float es_hpa = 6.112f * std::exp(
        17.67f * (out.temperature_k - t_freezing_k) / (out.temperature_k - 29.65f));
    if (!std::isfinite(es_hpa) || p_hpa <= 1.0e-3f)
    {
        return out;
    }

    es_hpa = std::max(0.0f, std::min(es_hpa, 0.99f * p_hpa));
    const float denom = std::max(1.0e-3f, p_hpa - es_hpa);
    float qsat = 0.622f * es_hpa / denom;
    qsat = std::max(1.0e-8f, std::min(0.10f, qsat));
    out.saturation_mixing_ratio_kgkg = qsat;

    if (std::isfinite(qv_kgkg) && std::isfinite(qsat) && qsat > 0.0f)
    {
        const float rh = std::max(0.0f, std::min(200.0f, 100.0f * (qv_kgkg / qsat)));
        out.relative_humidity_pct = rh;
        const float rh_frac = std::max(1.0e-6f, rh * 0.01f);
        const float temperature_c = out.temperature_k - t_freezing_k;
        const float gamma_val = std::log(rh_frac) +
            (17.625f * temperature_c) / (243.04f + temperature_c);
        const float gamma_denom = 17.625f - gamma_val;
        if (std::abs(gamma_denom) > 1.0e-6f)
        {
            const float td_c = 243.04f * gamma_val / gamma_denom;
            out.dewpoint_k = td_c + t_freezing_k;
        }
    }

    return out;
}

} // namespace derived
} // namespace tmv
