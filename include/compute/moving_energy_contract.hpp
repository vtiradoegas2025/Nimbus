#pragma once
#include "compute/dispatch/moving_energy.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
namespace nimbus::compute::moving_energy
{
inline std::size_t cells(const MovingEnergyDispatch& d)
{
    return std::size_t(d.constants.nx) * d.constants.ny * d.constants.nz;
}
inline bool validate(const MovingEnergyDispatch& d)
{
    const auto& p = d.constants;
    if (p.nx < 3 || p.ny < 3 || p.nz < 1 || p.operation < 0 || p.operation > 1 || !d.input || !d.output)
        return false;
    if (std::size_t(p.nx) > std::size_t(std::numeric_limits<int>::max()) / MovingEnergyDispatch::input_stride /
                                std::size_t(p.ny) / std::size_t(p.nz))
        return false;
    const std::array<float, 28> parameters{
        p.rd_high,        p.rd_low,        p.cv_high,        p.cv_low,       p.cvv_high,     p.cvv_low,
        p.cl_high,        p.cl_low,        p.ci_high,        p.ci_low,       p.tref_high,    p.tref_low,
        p.ev_high,        p.ev_low,        p.ei_high,        p.ei_low,       p.gravity_high, p.gravity_low,
        p.p0_high,        p.p0_low,        p.kappa_high,     p.kappa_low,    p.gamma_high,   p.gamma_low,
        p.theta_min_high, p.theta_min_low, p.theta_max_high, p.theta_max_low};
    for (float v : parameters)
        if (!std::isfinite(v))
            return false;
    for (std::size_t f = 0; f < parameters.size(); f += 2)
        if (std::abs(double(parameters[f + 1])) >
            std::abs(double(parameters[f])) * std::numeric_limits<float>::epsilon())
            return false;
    for (int f : {0, 2, 4, 6, 8, 10, 16, 18, 20, 22, 24, 26})
        if (double(parameters[f]) + parameters[f + 1] <= 0)
            return false;
    if (double(p.theta_max_high) + p.theta_max_low < double(p.theta_min_high) + p.theta_min_low)
        return false;
    const auto a = reinterpret_cast<std::uintptr_t>(d.input), b = reinterpret_cast<std::uintptr_t>(d.output);
    if ((a <= b && b - a < cells(d) * MovingEnergyDispatch::input_stride * sizeof(float)) ||
        (b < a && a - b < cells(d) * MovingEnergyDispatch::output_stride * sizeof(float)))
        return false;
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        const auto* v = d.input + n * MovingEnergyDispatch::input_stride;
        for (std::size_t f = 0; f < MovingEnergyDispatch::input_stride; ++f)
            if (!std::isfinite(v[f]))
                return false;
        if (v[0] <= 0 || v[1] <= 0 || v[2] <= 0 || double(v[22]) + v[23] <= 0)
            return false;
        for (int f = 6; f <= 13; ++f)
            if (v[f] < 0)
                return false;
        for (int f = 14; f < 24; f += 2)
            if (std::abs(double(v[f + 1])) > std::abs(double(v[f])) * std::numeric_limits<float>::epsilon())
                return false;
    }
    return true;
}
inline bool finite_candidate(const float* values, const MovingEnergyDispatch& d)
{
    for (std::size_t n = 0; n < cells(d); ++n)
    {
        const auto* v = values + n * MovingEnergyDispatch::output_stride;
        for (std::size_t f = 0; f < MovingEnergyDispatch::output_stride; ++f)
            if (!std::isfinite(v[f]))
                return false;
        const int  i = int(n) / (d.constants.ny * d.constants.nz), j = (int(n) / d.constants.nz) % d.constants.ny;
        const bool owned = i > 0 && i < d.constants.nx - 1 && j > 0 && j < d.constants.ny - 1;
        if (v[0] <= 0 || v[1] <= 0 || ((d.constants.operation == 0 || owned) && double(v[2]) + v[3] <= 0))
            return false;
        if (d.constants.operation == 1 && owned &&
            (v[0] < double(d.constants.theta_min_high) + d.constants.theta_min_low ||
             v[0] > double(d.constants.theta_max_high) + d.constants.theta_max_low))
            return false;
    }
    return true;
}
} // namespace nimbus::compute::moving_energy
