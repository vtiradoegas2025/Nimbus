#pragma once

/**
 * @file physical_constants.hpp
 * @brief Shared physical constants used across model components.
 *
 * Centralizes thermodynamic and radiative constants to keep parameter
 * values consistent between dynamics, microphysics, radiation,
 * turbulence, and boundary-layer schemes.
 */

namespace physical_constants
{
inline constexpr double gravity_ms2 = 9.81;
inline constexpr double gas_constant_dry_air_jkgk = 287.0;
inline constexpr double specific_heat_cp_jkgk = 1004.0;
inline constexpr double reference_pressure_pa = 100000.0;

inline constexpr double freezing_temperature_k = 273.15;
inline constexpr double latent_heat_vaporization_jkg = 2.5e6;
inline constexpr double latent_heat_fusion_jkg = 3.34e5;
inline constexpr double latent_heat_sublimation_jkg =
    latent_heat_vaporization_jkg + latent_heat_fusion_jkg;
inline constexpr double von_karman_constant = 0.4;
inline constexpr double pi = 3.14159265358979323846;
inline constexpr double stefan_boltzmann_wm2k4 = 5.67e-8;
inline constexpr double solar_constant_wm2 = 1366.0;
inline constexpr double theta_reference_k = 300.0;
inline constexpr double dry_air_gamma =
    specific_heat_cp_jkgk / (specific_heat_cp_jkgk - gas_constant_dry_air_jkgk);
} // namespace physical_constants

inline constexpr double g = physical_constants::gravity_ms2;
inline constexpr double R_d = physical_constants::gas_constant_dry_air_jkgk;
inline constexpr double cp = physical_constants::specific_heat_cp_jkgk;
inline constexpr double p0 = physical_constants::reference_pressure_pa;
