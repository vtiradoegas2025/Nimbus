#include "microphysics/schemes/thompson/cloud_phase.hpp"
#include "microphysics/schemes/thompson/moment_initialization.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::thompson
{
namespace
{
float stored(float initial, double candidate, double dt)
{
    float rate = static_cast<float>((candidate - initial) / dt);
    if (!std::isfinite(rate))
    {
        throw std::overflow_error("Thompson cloud-phase tendency is not representable");
    }
    if (initial + dt * rate < 0)
    {
        rate = std::nextafter(rate, std::numeric_limits<float>::infinity());
    }
    return rate;
}
} // namespace

std::array<float, 5> cloud_phase_tendencies(const CloudPhaseState& s, double dt)
{
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(s.temperature) || s.temperature <= 0 ||
        !std::isfinite(s.density) || s.density <= 0 || !std::isfinite(s.vapor) || s.vapor < 0)
    {
        throw std::invalid_argument("invalid Thompson cloud-phase thermodynamics");
    }
    const std::array<float, 4> initial{s.cloud, s.ice, s.cloud_number, s.ice_number};
    auto                       result = s.tendencies;
    for (std::size_t f = 0; f < result.size(); ++f)
    {
        if (!std::isfinite(result[f]))
        {
            throw std::invalid_argument("invalid Thompson cloud-phase tendency");
        }
        if (f < initial.size())
        {
            if (!std::isfinite(initial[f]) || initial[f] < 0)
            {
                throw std::invalid_argument("invalid Thompson cloud-phase donor");
            }
            const double candidate = initial[f] + dt * result[f];
            const double envelope  = 4e-7 * (initial[f] + std::abs(dt * result[f])) + (f < 2 ? 2e-12 : 0);
            if (!std::isfinite(candidate) || candidate < -envelope)
            {
                throw std::invalid_argument("invalid Thompson cloud-phase candidate");
            }
            if (candidate < 0)
            {
                result[f] = stored(initial[f], 0, dt);
            }
        }
    }
    const bool melt            = s.temperature > cloud_melting_temperature && s.ice + dt * result[1] > 0;
    const bool freeze = s.temperature < homogeneous_freezing_temperature &&
                        (s.cloud + dt * result[0] > 0 ||
                         (s.cloud_number + dt * result[2] > 0 && s.density * (s.ice + dt * result[1]) > 1e-12));
    const auto normalize_cloud = [&]()
    {
        const double mass   = s.density * (s.cloud + dt * result[0]);
        const double number = s.density * std::max(0.0, s.cloud_number + dt * result[2]);
        if (mass > 1e-12)
        {
            const double target   = normalize_cloud_number_m3(mass, number);
            const double rounding = 8 * std::numeric_limits<float>::epsilon() * std::max({1.0, target, number});
            if (std::abs(target - number) > rounding || number > 1999e6)
                result[2] = stored(s.cloud_number, target / s.density, dt);
        }
    };
    const auto normalize_ice = [&]()
    {
        const double mass   = s.density * (s.ice + dt * result[1]);
        const double number = s.density * std::max(0.0, s.ice_number + dt * result[3]);
        if (mass > 1e-12)
        {
            const double target   = std::min(999e3, normalize_ice_number_m3(mass, number));
            const double rounding = 8 * std::numeric_limits<float>::epsilon() * std::max({1.0, target, number});
            if (std::abs(target - number) > rounding || number > 999e3)
            {
                result[3] = stored(s.ice_number, target / s.density, dt);
                if (s.ice_number + dt * result[3] > 999e3 / s.density)
                    result[3] = std::nextafter(result[3], -std::numeric_limits<float>::infinity());
            }
        }
    };
    if (!melt && !freeze)
    {
        normalize_cloud();
        normalize_ice();
        return result;
    }
    const int    donor = melt ? 1 : 0, receiver = 1 - donor;
    const float  donor_rate      = stored(initial[donor], 0, dt);
    const double mass_rate       = static_cast<double>(result[donor]) - donor_rate;
    const float  number_rate     = stored(initial[donor + 2], 0, dt);
    const double number_transfer = static_cast<double>(result[donor + 2]) - number_rate;
    result[donor]                = donor_rate;
    result[receiver]             = static_cast<float>(result[receiver] + mass_rate);
    result[donor + 2]            = number_rate;
    double       number          = initial[receiver + 2] + dt * (result[receiver + 2] + number_transfer);
    const double mass            = initial[receiver] + dt * result[receiver];
    if (s.density * mass > 1e-12)
    {
        number = melt ? normalize_cloud_number_m3(s.density * mass, s.density * number) / s.density
                      : std::min(999e3, normalize_ice_number_m3(s.density * mass, s.density * number)) / s.density;
    }
    const double maximum_number = (melt ? 1999e6 : 999e3) / s.density;
    result[receiver + 2]        = stored(initial[receiver + 2], std::min(number, maximum_number), dt);
    if (initial[receiver + 2] + dt * result[receiver + 2] > maximum_number)
    {
        result[receiver + 2] = std::nextafter(result[receiver + 2], -std::numeric_limits<float>::infinity());
    }
    const double latent = melt ? 334000.0 : 334000.0 + 2112.0 * (s.temperature - cloud_melting_temperature);
    if (latent <= 0)
    {
        throw std::invalid_argument("invalid Thompson cloud-phase latent heat");
    }
    result[4] =
        static_cast<float>(result[4] + (melt ? -1.0 : 1.0) * latent / (1004.0 * (1.0 + 0.887 * s.vapor)) * mass_rate);
    normalize_cloud();
    normalize_ice();
    for (float value : result)
    {
        if (!std::isfinite(value))
        {
            throw std::overflow_error("Thompson cloud-phase result is not representable");
        }
    }
    return result;
}
} // namespace tmv::thompson
