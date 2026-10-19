#pragma once

#include <algorithm>
#include <cmath>

#include "core/infra/physical_constants.hpp"
#include "microphysics/microphysics_base.hpp"

/**
 * @file fall_speeds.hpp
 * @brief Mass-weighted hydrometeor fall speeds over Marshall-Palmer
 *        size distributions (Lin-Farley-Orville 1983).
 *
 * SCIENCE.md M12: the old per-species power laws a q^b were far
 * from the gamma-integrated literature forms - rain reached 27 m/s at
 * 1 g/kg (physical ~5-6) and the max(q, 1e-6) floor imposed a MINIMUM
 * 11.6 m/s on trace rain; graupel and hail shared one coefficient set.
 *
 * For N(D) = N0 exp(-lambda D), m(D) = (pi/6) rho_x D^3 and
 * V(D) = a D^b, the mass-weighted mean speed is
 *
 *   V_q = a Gamma(4 + b) / (6 lambda^b),
 *   lambda = (pi rho_x N0 / (rho_air q))^(1/4).
 *
 * Rain and snow use LFO83 power-law constants; their speeds carry the
 * separate Foote-du Toit (rho0/rho)^1/2 altitude correction at the CALL
 * SITE (it is the surface-calibrated-coefficient correction, exactly as
 * in LFO83 eq 9-10). Graupel and hail use the Wisner et al. (1972)
 * drag-balance form V(D) = sqrt(4 g rho_x D / (3 C_d rho_air)) whose
 * air-density dependence is explicit - callers must NOT apply the
 * altitude correction on top.
 *
 * Reference values at q = 1 g/kg, rho_air = 1.2 kg/m^3:
 * rain 5.42, snow 1.21, graupel 5.83, hail 10.58 m/s.
 *
 * The sedimentation shaders (lin_sedimentation.comp,
 * kessler_sedimentation.comp) mirror these formulas with the same
 * constants as GLSL literals - keep them in sync.
 */
namespace tmv
{
namespace fall_speeds
{

// Gamma(4 + b) for the exponents used here.
inline constexpr double kGamma4p8 = 17.837862;   // rain b = 0.8
inline constexpr double kGamma4p25 = 8.2850847;  // snow b = 0.25
inline constexpr double kGamma4p5 = 11.6317284;  // Wisner b = 1/2
inline constexpr double kGamma3p25 = 2.5492514;  // PSACW, 3 + d_snow

// LFO83 power-law constants.
inline constexpr double kRainA = 841.99667;      // m^(1-b)/s
inline constexpr double kRainB = 0.8;
inline constexpr double kSnowC = 4.83607122;
inline constexpr double kSnowD = 0.25;
inline constexpr double kDragCd = 0.6;           // Wisner drag coefficient

// Classic Kessler rain speed (Soong-Ogura 1973; CM1's kessler):
// V = 36.34 (1e-3 rho_air q)^0.1346 with rho_air q in kg/m^3.
inline constexpr double kKesslerRainA = 36.34;
inline constexpr double kKesslerRainB = 0.1346;

// Air density stand-in when no base-state profile is available.
inline constexpr double kFallbackAirDensity = 1.2;

// Below this mixing ratio the flux V q is dynamically negligible and
// the slope parameter would overflow toward infinity; callers skip.
inline constexpr double kMinQ = 1.0e-9;

inline double marshall_palmer_lambda(double q, double rho_air,
                                     double rho_x, double N0)
{
    const double pi = 3.14159265358979323846;
    return std::pow(pi * rho_x * N0 / (rho_air * q), 0.25);
}

/// Rain, mass-weighted (no altitude correction inside).
inline double rain_mass_weighted(double q, double rho_air)
{
    if (q < kMinQ) return 0.0;
    const double lam = marshall_palmer_lambda(
        q, rho_air, microphysics_constants::rho_w,
        microphysics_constants::N0_rain);
    return kRainA * kGamma4p8 / (6.0 * std::pow(lam, kRainB));
}

/// Snow, mass-weighted (no altitude correction inside).
inline double snow_mass_weighted(double q, double rho_air)
{
    if (q < kMinQ) return 0.0;
    const double lam = marshall_palmer_lambda(
        q, rho_air, microphysics_constants::rho_snow,
        microphysics_constants::N0_snow);
    return kSnowC * kGamma4p25 / (6.0 * std::pow(lam, kSnowD));
}

/// Wisner drag-balance speed; air density dependence is explicit, so
/// callers must not add the (rho0/rho)^1/2 correction.
inline double wisner_mass_weighted(double q, double rho_air,
                                   double rho_x, double N0)
{
    if (q < kMinQ) return 0.0;
    const double lam = marshall_palmer_lambda(q, rho_air, rho_x, N0);
    const double drag = std::sqrt(
        4.0 * physical_constants::gravity_ms2 * rho_x
        / (3.0 * kDragCd * rho_air));
    return (kGamma4p5 / 6.0) * drag / std::sqrt(lam);
}

inline double graupel_mass_weighted(double q, double rho_air)
{
    return wisner_mass_weighted(q, rho_air,
                                microphysics_constants::rho_graupel,
                                microphysics_constants::N0_graupel);
}

inline double hail_mass_weighted(double q, double rho_air)
{
    return wisner_mass_weighted(q, rho_air,
                                microphysics_constants::rho_hail,
                                microphysics_constants::N0_hail);
}

/// Classic Kessler rain speed (no altitude correction inside).
inline double kessler_rain(double q, double rho_air)
{
    if (q < kMinQ) return 0.0;
    return kKesslerRainA
        * std::pow(1.0e-3 * rho_air * q, kKesslerRainB);
}

// -- Marshall-Palmer collection / freezing integrals (SCIENCE.md
// M9): same spectrum machinery as the fall speeds above, mirrored in
// lin_pointwise.comp. --

/// Bigg stochastic freezing of supercooled rain (LFO83 Pgfr): the
/// mixing-ratio rate (kg/kg/s) of rain freezing into hail embryos at
/// supercooling = T0 - T > 0:
///   dq/dt = 20 pi^2 B' N0r (rho_w / rho_air)
///           (exp(A' dT) - 1) / lambda_r^7.
/// The exponential explodes toward cold temperatures (all supercooled
/// rain freezes within seconds below ~ -25 C); callers mass-limit.
inline double rain_bigg_freezing_rate(double qr, double rho_air,
                                      double supercooling)
{
    if (qr < kMinQ || supercooling <= 0.0) return 0.0;
    const double pi = 3.14159265358979323846;
    const double lam = marshall_palmer_lambda(
        qr, rho_air, microphysics_constants::rho_w,
        microphysics_constants::N0_rain);
    const double growth =
        std::exp(std::min(microphysics_constants::bigg_A * supercooling,
                          60.0)) - 1.0;
    return 20.0 * pi * pi * microphysics_constants::bigg_B
        * microphysics_constants::N0_rain
        * (microphysics_constants::rho_w / rho_air)
        * growth / std::pow(lam, 7.0);
}

/// Snow collects supercooled cloud water (LFO83 PSACW): continuous-
/// collection integral over the snow spectrum with V(D) = c D^d,
/// returned PER UNIT CLOUD mixing ratio (multiply by qc for kg/kg/s):
///   rate = (pi/4) E_sc N0s c Gamma(3 + d) / lambda_s^(3 + d).
inline double snow_cloud_collection_rate(double qs, double rho_air)
{
    if (qs < kMinQ) return 0.0;
    const double pi = 3.14159265358979323846;
    const double lam = marshall_palmer_lambda(
        qs, rho_air, microphysics_constants::rho_snow,
        microphysics_constants::N0_snow);
    return (pi / 4.0) * microphysics_constants::E_snow_cloud
        * microphysics_constants::N0_snow * kSnowC * kGamma3p25
        / std::pow(lam, 3.0 + kSnowD);
}

} // namespace fall_speeds
} // namespace tmv

namespace fall_speeds = tmv::fall_speeds;
