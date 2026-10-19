#pragma once
#include "compute/dispatch/earth_geometry.hpp"
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::earth_geometry
{
inline std::size_t cells(const EarthGeometryDispatch& d)
{
    const auto& p = d.constants;
    return (std::size_t(p.nx) + 4) * (std::size_t(p.ny) + 4) * (p.operation >= 2 ? std::size_t(p.nz) + 1 : 1);
}
inline bool validate(const EarthGeometryDispatch& d)
{
    const auto& p = d.constants;
    if (p.nx < 1 || p.ny < 1 || p.nz < 1 || p.operation < 0 || p.operation > 3 || p.scheme < 0 || p.scheme > 2 ||
        p.smoothed < 0 || p.smoothed > 1)
        return false;
    if (std::size_t(p.nx) + 4 >
        std::size_t(std::numeric_limits<int>::max()) / (std::size_t(p.ny) + 4) / (std::size_t(p.nz) + 1))
        return false;
    const std::array<float, 22> parameters{
        p.dx_high,      p.dx_low,      p.dy_high,      p.dy_low,     p.dz_high,         p.dz_low,
        p.height_high,  p.height_low,  p.width_high,   p.width_low,  p.wavelength_high, p.wavelength_low,
        p.decay_high,   p.decay_low,   p.x_high,       p.x_low,      p.y_high,          p.y_low,
        p.shift_x_high, p.shift_x_low, p.shift_y_high, p.shift_y_low};
    for (float v : parameters)
        if (!std::isfinite(v))
            return false;
    for (std::size_t f = 0; f < parameters.size(); f += 2)
        if (std::abs(double(parameters[f + 1])) >
            std::abs(double(parameters[f])) * std::numeric_limits<float>::epsilon())
            return false;
    if (double(p.dx_high) + p.dx_low <= 1e-12 || double(p.dy_high) + p.dy_low <= 1e-12 ||
        double(p.dz_high) + p.dz_low <= 1e-12 || double(p.width_high) + p.width_low <= 1e-12 ||
        double(p.wavelength_high) + p.wavelength_low <= 1e-12 || double(p.height_high) + p.height_low < 0 ||
        double(p.decay_high) + p.decay_low < 0)
        return false;
    if (p.operation == 1 && p.scheme > 1)
        return false;
    if (p.operation == 1 &&
        (double(p.x_high) + p.x_low < 0 || double(p.x_high) + p.x_low >= 3 || double(p.y_high) + p.y_low < 0 ||
         double(p.y_high) + p.y_low >= 3 || double(p.x_high) + p.x_low + p.shift_x_high + p.shift_x_low < 0 ||
         double(p.x_high) + p.x_low + p.shift_x_high + p.shift_x_low >= 3 ||
         double(p.y_high) + p.y_low + p.shift_y_high + p.shift_y_low < 0 ||
         double(p.y_high) + p.y_low + p.shift_y_high + p.shift_y_low >= 3 ||
         std::abs(double(p.shift_x_high) + p.shift_x_low) > 1 + 64 * std::numeric_limits<double>::epsilon() ||
         std::abs(double(p.shift_y_high) + p.shift_y_low) > 1 + 64 * std::numeric_limits<double>::epsilon()))
        return false;
    const auto overlap = [&](const float* a, const float* b)
    {
        const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        return (x <= y ? y - x : x - y) < cells(d) * sizeof(float);
    };
    for (auto p : d.input)
        if (!p)
            return false;
    for (std::size_t f = 0; f < d.output.size(); ++f)
    {
        if (!d.output[f])
            return false;
        for (auto p : d.input)
            if (overlap(p, d.output[f]))
                return false;
        for (std::size_t g = 0; g < f; ++g)
            if (overlap(d.output[g], d.output[f]))
                return false;
    }
    for (auto p : d.input)
        for (std::size_t n = 0; n < cells(d); ++n)
            if (!std::isfinite(p[n]))
                return false;
    if (p.operation == 1 || p.operation == 2)
    {
        const int    fields = p.operation == 1 ? 2 : 6;
        const double top    = (double(p.dz_high) + p.dz_low) * p.nz;
        for (int f = 0; f < fields; f += 2)
            for (std::size_t n = 0; n < cells(d); ++n)
            {
                const double value = double(d.input[f][n]) + d.input[f + 1][n];
                if (value < 0 || value >= top ||
                    std::abs(double(d.input[f + 1][n])) >
                        std::abs(double(d.input[f][n])) * std::numeric_limits<float>::epsilon())
                    return false;
            }
    }
    if (p.operation == 3)
        for (int k = 0; k < p.nz; ++k)
            for (int i = 0; i <= p.nx; ++i)
                for (int j = 0; j <= p.ny; ++j)
                {
                    const auto n = (std::size_t(k) * (p.nx + 4) + i) * (p.ny + 4) + j;
                    if (j < p.ny && double(d.input[0][n]) + d.input[1][n] <= 1e-12)
                        return false;
                    if (i < p.nx && double(d.input[2][n]) + d.input[3][n] <= 1e-12)
                        return false;
                }
    return true;
}
inline auto constants(const EarthGeometryDispatch& d)
{
    return d.constants;
}
inline bool finite_candidate(const std::array<const float*, 20>& out, const EarthGeometryDispatch& d)
{
    for (auto p : out)
        for (std::size_t n = 0; n < cells(d); ++n)
            if (!std::isfinite(p[n]))
                return false;
    const auto& p = d.constants;
    if (p.operation < 2)
    {
        const double top    = (double(p.dz_high) + p.dz_low) * p.nz;
        const int    fields = p.operation == 0 ? 2 : 6;
        for (int f = 0; f < fields; f += 2)
            for (std::size_t n = 0; n < cells(d); ++n)
                if (double(out[f][n]) + out[f + 1][n] < 0 || double(out[f][n]) + out[f + 1][n] >= top)
                    return false;
    }
    if (p.operation == 2)
        for (int k = 0; k < p.nz; ++k)
            for (int i = 0; i <= p.nx; ++i)
                for (int j = 0; j <= p.ny; ++j)
                {
                    const auto n = (std::size_t(k) * (p.nx + 4) + i) * (p.ny + 4) + j;
                    if (i < p.nx && j < p.ny && double(out[2][n]) + out[3][n] <= 1e-12)
                        return false;
                    if (j < p.ny && double(out[12][n]) + out[13][n] <= 1e-12)
                        return false;
                    if (i < p.nx && double(out[14][n]) + out[15][n] <= 1e-12)
                        return false;
                }
    return true;
}
} // namespace nimbus::compute::earth_geometry
