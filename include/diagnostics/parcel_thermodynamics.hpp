#pragma once

#include "core/infra/physical_constants.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::diagnostics
{
// Liquid-water pseudoadiabat, with the pressure-coordinate lapse equation
// documented by Unidata's moist_lapse. Nimbus stores dry-air mixing ratios.
inline double wet_bulb_potential_temperature(double temperature_k,
                                             double pressure_pa,
                                             double vapor_mixing_ratio)
{
    constexpr double rd = physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cp = physical_constants::specific_heat_cp_jkgk;
    constexpr double lv = physical_constants::latent_heat_vaporization_jkg;
    constexpr double epsilon = rd / physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double kappa = rd / cp;
    constexpr double p0 = physical_constants::reference_pressure_pa;
    const double invalid = std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(temperature_k) || temperature_k <= 80.0 ||
        !std::isfinite(pressure_pa) || pressure_pa <= 0.0 ||
        !std::isfinite(vapor_mixing_ratio) || vapor_mixing_ratio < 0.0)
        return invalid;
    const auto saturation = [&](double t, double p)
    {
        const double es = 611.2 * std::exp(17.67 * (t - 273.15) / (t - 29.65));
        return es < p ? epsilon * es / (p - es) : invalid;
    };
    const double rs = saturation(temperature_k, pressure_pa);
    if (!std::isfinite(rs)) return invalid;
    double x = std::log(pressure_pa);
    double t = temperature_k;
    if (vapor_mixing_ratio < rs)
    {
        // At 80 K the remaining liquid saturation mixing ratio is negligible;
        // this also defines the continuous zero-vapor limit without log(0).
        double low = std::log(80.0 / temperature_k) / kappa;
        double high = 0.0;
        for (int iteration = 0; iteration < 48; ++iteration)
        {
            const double mid = 0.5 * (low + high);
            const double trial = saturation(temperature_k * std::exp(kappa * mid),
                                             pressure_pa * std::exp(mid));
            if (trial > vapor_mixing_ratio) high = mid;
            else low = mid;
        }
        const double lift = 0.5 * (low + high);
        x += lift;
        t *= std::exp(kappa * lift);
    }
    const double span = std::log(p0) - x;
    if (!std::isfinite(span) || std::abs(span) > 40.0) return invalid;
    const int steps = std::max(1, static_cast<int>(std::ceil(std::abs(span) / 0.04)));
    const double dx = span / steps;
    const auto lapse = [&](double log_p, double value)
    {
        const double r = saturation(value, std::exp(log_p));
        return (rd * value + lv * r)
            / (cp + lv * lv * r * epsilon / (rd * value * value));
    };
    for (int step = 0; step < steps; ++step)
    {
        const double a = lapse(x, t);
        const double b = lapse(x + 0.5 * dx, t + 0.5 * dx * a);
        const double c = lapse(x + 0.5 * dx, t + 0.5 * dx * b);
        const double d = lapse(x + dx, t + dx * c);
        t += dx * (a + 2.0 * b + 2.0 * c + d) / 6.0;
        x += dx;
    }
    return std::isfinite(t) && t > 0.0 ? t : invalid;
}
}
