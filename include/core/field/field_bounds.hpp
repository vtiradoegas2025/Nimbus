#pragma once

#include <algorithm>
#include <cmath>

#include "core/field/numerical_integrity.hpp"
#include "core/infra/physical_constants.hpp"

inline constexpr double theta0 = physical_constants::theta_reference_k;
inline constexpr float theta_min_k = 200.0f;
inline constexpr float theta_max_k = 700.0f;  // Stratospheric theta easily exceeds 500K
inline constexpr float pressure_min_pa = 1000.0f;
inline constexpr float pressure_max_pa = 200000.0f;
inline constexpr float density_min_kgm3 = 0.05f;
inline constexpr float density_max_kgm3 = 2.0f;
inline constexpr float wind_horizontal_abs_max_ms = 180.0f;
inline constexpr float wind_vertical_abs_max_ms = 120.0f;
inline constexpr float qv_max_kgkg = 0.05f;
inline constexpr float hydrometeor_max_kgkg = 0.05f;
inline constexpr float passive_tracer_max = 1.5f;

// Writer-owned repair policy. The orchestrator disables repairs for Warn and
// Strict so post-stage validation can observe the actual numerical result.
// Sanitize and Off preserve the production safety-net behavior.
inline bool primary_state_repairs_enabled = true;
inline bool primary_state_repairs_fail_closed = false;

inline void set_primary_state_repairs_enabled(bool enabled) noexcept
{
    primary_state_repairs_enabled = enabled;
}

inline void set_primary_state_repair_policy(bool enabled, bool fail_closed) noexcept
{
    primary_state_repairs_enabled = enabled;
    primary_state_repairs_fail_closed = fail_closed;
}

/**
 * @brief Sanitizes a scalar value: replaces non-finite with a fallback, then clamps.
 * @param value   Candidate scalar.
 * @param fallback Value to substitute when input is NaN or Inf.
 * @param lo      Lower bound for clamping.
 * @param hi      Upper bound for clamping.
 * @return Sanitized and clamped value.
 *
 * Every field clamp function below delegates to this so that NaN/Inf never
 * propagates silently through any prognostic variable.
 */
inline float sanitize_value(float value, float fallback, float lo, float hi)
{
    if (!primary_state_repairs_enabled)
    {
        return value;
    }
    if (!std::isfinite(value))
    {
        tmv::record_numerical_nonfinite_incidents();
        return fallback;
    }
    return std::clamp(value, lo, hi);
}

/**
 * @brief Clamps potential temperature into physically allowed bounds.
 * Non-finite input is replaced with the reference potential temperature.
 */
inline float clamp_theta_k(float value)
{
    return sanitize_value(value, static_cast<float>(theta0), theta_min_k, theta_max_k);
}

/**
 * @brief Clamps pressure into physically allowed bounds.
 * Non-finite input is replaced with standard sea-level pressure (101325 Pa).
 */
inline float clamp_pressure_pa(float value)
{
    return sanitize_value(value, 101325.0f, pressure_min_pa, pressure_max_pa);
}

/**
 * @brief Sanitizes non-finite values and enforces non-negativity.
 * @param value Candidate scalar.
 * @return Zero for non-finite input, otherwise max(value, 0).
 */
inline float clamp_non_negative(float value)
{
    if (!primary_state_repairs_enabled)
    {
        return value;
    }
    if (!std::isfinite(value))
    {
        tmv::record_numerical_nonfinite_incidents();
        return 0.0f;
    }
    return std::max(0.0f, value);
}

/**
 * @brief Clamps density into physically allowed bounds.
 * Non-finite input is replaced with a standard tropospheric density (1.0 kg/m^3).
 */
inline float clamp_density_kgm3(float value)
{
    return sanitize_value(value, 1.0f, density_min_kgm3, density_max_kgm3);
}

/**
 * @brief Clamps horizontal wind speed magnitude.
 * Non-finite input is replaced with zero (calm).
 */
inline float clamp_wind_horizontal_ms(float value)
{
    return sanitize_value(value, 0.0f, -wind_horizontal_abs_max_ms, wind_horizontal_abs_max_ms);
}

/**
 * @brief Clamps vertical wind speed magnitude.
 * Non-finite input is replaced with zero (calm).
 */
inline float clamp_wind_vertical_ms(float value)
{
    return sanitize_value(value, 0.0f, -wind_vertical_abs_max_ms, wind_vertical_abs_max_ms);
}

/**
 * @brief Clamps water-vapor mixing ratio and sanitizes non-finite input.
 * Non-finite input is replaced with zero.
 */
inline float clamp_qv_kgkg(float value)
{
    return sanitize_value(value, 0.0f, 0.0f, qv_max_kgkg);
}

/**
 * @brief Clamps hydrometeor mixing ratio and sanitizes non-finite input.
 * Non-finite input is replaced with zero.
 */
inline float clamp_hydrometeor_kgkg(float value)
{
    return sanitize_value(value, 0.0f, 0.0f, hydrometeor_max_kgkg);
}
