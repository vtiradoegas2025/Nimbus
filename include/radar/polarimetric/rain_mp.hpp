#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tmv::radar::polarimetric
{

struct RainMpObservables
{
    double dm_mm = 0.0;
    double zh_linear = 0.0;
    double zdr_db = 0.0;
    double kdp_deg_km = 0.0;
    double rhohv = 1.0;
};

// Zhang, Gao, and Du (2021), doi:10.1007/s00376-021-0289-6,
// Eqs. 13-16 use W in g m^-3 and Dm in mm. The fixed N0 conversion is the
// Marshall-Palmer rain distribution used by Nimbus single-moment schemes.
inline RainMpObservables zhang2021_rain_mp(double rain_mixing_ratio,
                                           double dry_air_density)
{
    if (!std::isfinite(rain_mixing_ratio) || rain_mixing_ratio < 0.0 ||
        !std::isfinite(dry_air_density) || dry_air_density <= 0.0)
    {
        throw std::invalid_argument(
            "rain polarimetric operator requires finite nonnegative qr and positive density");
    }
    if (rain_mixing_ratio == 0.0)
    {
        return {};
    }

    constexpr double water_density_kg_m3 = 1000.0;
    constexpr double rain_intercept_m4 = 8.0e6;
    constexpr double pi = 3.14159265358979323846;
    const double water_kg_m3 = dry_air_density * rain_mixing_ratio;
    const double lambda_m1 = std::pow(
        pi * water_density_kg_m3 * rain_intercept_m4 / water_kg_m3, 0.25);
    const double dm = 4000.0 / lambda_m1;
    // The polynomial does not have the physical dry-limit asymptote. Values
    // below Nimbus's 0.1 mm resolved-rain floor publish the exact dry limit.
    if (dm < 0.1)
    {
        return {};
    }
    const double dm2 = dm * dm;
    const double dm3 = dm2 * dm;
    const double dm4 = dm2 * dm2;
    const double water_g_m3 = 1000.0 * water_kg_m3;

    const double zh_amplitude =
        -0.3078 + 20.87 * dm + 46.04 * dm2 - 6.403 * dm3 + 0.2248 * dm4;
    const double zdr_ratio =
        1.019 - 0.1430 * dm + 0.3165 * dm2 - 0.06498 * dm3 + 0.004163 * dm4;
    const double kdp = water_g_m3 *
        (0.00926 - 0.0870 * dm + 0.1994 * dm2 - 0.02824 * dm3 + 0.001772 * dm4);
    const double rhohv_fit =
        0.9987 + 0.008289 * dm - 0.01160 * dm2 + 0.003513 * dm3 - 0.0003187 * dm4;

    if (!(zh_amplitude > 0.0) || !(zdr_ratio > 0.0) ||
        !std::isfinite(kdp) || !std::isfinite(rhohv_fit))
    {
        throw std::domain_error(
            "rain polarimetric state is outside the Zhang et al. (2021) fit domain");
    }

    RainMpObservables result;
    result.dm_mm = dm;
    result.zh_linear = water_g_m3 * zh_amplitude * zh_amplitude;
    result.zdr_db = 10.0 * std::log10(zdr_ratio);
    result.kdp_deg_km = std::max(kdp, 0.0);
    result.rhohv = std::min(rhohv_fit, 1.0);
    if (!std::isfinite(result.zh_linear) || result.zh_linear < 0.0 ||
        result.zdr_db < 0.0 || result.zdr_db > 6.0 ||
        result.kdp_deg_km < 0.0 || result.kdp_deg_km > 4.0 ||
        result.rhohv < 0.8 || result.rhohv > 1.0)
    {
        throw std::domain_error(
            "rain polarimetric state is outside the maintained 2021 operator range: "
            "Dm_mm=" + std::to_string(result.dm_mm) +
            " ZDR_dB=" + std::to_string(result.zdr_db) +
            " KDP_deg_km=" + std::to_string(result.kdp_deg_km) +
            " rhohv=" + std::to_string(result.rhohv));
    }
    return result;
}

} // namespace tmv::radar::polarimetric
