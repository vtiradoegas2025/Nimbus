#pragma once

#include "microphysics/microphysics_base.hpp"

#include <cmath>

namespace tmv::milbrandt_detail
{

inline bool internal_triple_raw_temperature(double theta, double pressure,
                                           double& temperature) noexcept
{
    if (!std::isfinite(theta) || theta < 150.0 || theta > 700.0 ||
        !std::isfinite(pressure) || pressure < 100.0 || pressure > 200000.0)
    {
        return false;
    }
    const double candidate = theta * std::pow(pressure / microphysics_constants::p0,
        microphysics_constants::R_d / microphysics_constants::cp);
    if (!std::isfinite(candidate))
    {
        return false;
    }
    temperature = candidate;
    return true;
}

inline bool internal_triple_raw_temperature_pair(double theta_unrounded, float theta_stored,
                                                double pressure, double& unrounded_temperature,
                                                double& stored_temperature) noexcept
{
    return internal_triple_raw_temperature(theta_unrounded, pressure, unrounded_temperature)
        && internal_triple_raw_temperature(theta_stored, pressure, stored_temperature);
}

} // namespace tmv::milbrandt_detail
