#pragma once
#include "compute/dispatch/reference_atmosphere.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::reference_atmosphere
{
inline std::size_t input_count(const ReferenceAtmosphereDispatch& d)
{
    return std::size_t(d.constants.levels) * ReferenceAtmosphereDispatch::profile_stride +
           2 * std::size_t(d.constants.count);
}
inline bool validate(const ReferenceAtmosphereDispatch& d)
{
    const auto& p     = d.constants;
    const auto  limit = std::size_t(std::numeric_limits<int>::max());
    if (p.count < 1 || p.levels < 1 || p.levels > (1 << 24) || !d.input || !d.output || p.padding != 0 ||
        std::size_t(p.count) > limit / ReferenceAtmosphereDispatch::output_stride ||
        std::size_t(p.levels) > (limit - 2 * std::size_t(p.count)) / ReferenceAtmosphereDispatch::profile_stride)
        return false;
    const std::array<float, 8> constants{
        p.dz_high, p.dz_low, p.rd_high, p.rd_low, p.p0_high, p.p0_low, p.kappa_high, p.kappa_low};
    for (std::size_t n = 0; n < constants.size(); n += 2)
        if (!std::isfinite(constants[n]) || !std::isfinite(constants[n + 1]) ||
            double(constants[n]) + constants[n + 1] <= 0 ||
            std::abs(double(constants[n + 1])) > std::abs(double(constants[n])) * std::numeric_limits<float>::epsilon())
            return false;
    for (int exponent : {p.theta_exponent, p.pressure_exponent, p.vapor_exponent, p.u_exponent, p.v_exponent})
        if (exponent < -1073 || exponent > 128)
            return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < input_count(d) * sizeof(float)) ||
        (b < a && a - b < std::size_t(p.count) * ReferenceAtmosphereDispatch::output_stride * sizeof(float)))
        return false;
    for (std::size_t n = 0; n < input_count(d); n += 2)
        if (!std::isfinite(d.input[n]) || !std::isfinite(d.input[n + 1]) ||
            std::abs(double(d.input[n + 1])) > std::abs(double(d.input[n])) * std::numeric_limits<float>::epsilon())
            return false;
    for (int k = 0; k < p.levels; ++k)
    {
        const auto* v = d.input + std::size_t(k) * ReferenceAtmosphereDispatch::profile_stride;
        if (double(v[0]) + v[1] <= 0 || double(v[2]) + v[3] <= 0 || double(v[4]) + v[5] < 0)
            return false;
    }
    return true;
}
inline bool finite_candidate(const float* values, const ReferenceAtmosphereDispatch& d)
{
    for (int n = 0; n < d.constants.count; ++n)
    {
        const auto* v = values + std::size_t(n) * ReferenceAtmosphereDispatch::output_stride;
        for (std::size_t f = 0; f < ReferenceAtmosphereDispatch::output_stride; ++f)
            if (!std::isfinite(v[f]))
                return false;
        if (v[0] <= 0 || v[1] <= 0 || v[2] < 0 || v[5] <= 0)
            return false;
    }
    return true;
}
} // namespace nimbus::compute::reference_atmosphere
