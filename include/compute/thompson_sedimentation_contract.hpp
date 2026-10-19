#pragma once

#include "compute/dispatch/microphysics.hpp"
#include "compute/kernel_abi.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace nimbus::compute::thompson_sedimentation
{
inline std::array<const float*, 7> inputs(const ThompsonMomentSedimentationDispatch& d)
{
    return {d.mass, d.number, d.density, d.temperature, d.thickness, d.vertical_velocity, d.snow_speed_multiplier};
}

inline bool overlap(const float* a, std::size_t na, const float* b, std::size_t nb)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a);
    const auto y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na * sizeof(float) : x - y < nb * sizeof(float);
}

inline bool validate(const ThompsonMomentSedimentationDispatch& d, std::size_t& cells, std::size_t& columns)
{
    if (d.nx <= 0 || d.ny <= 0 || d.nz <= 0)
    {
        return false;
    }
    const auto xy = static_cast<std::uint64_t>(d.nx) * d.ny;
    if (xy > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / d.nz)
    {
        return false;
    }
    columns           = static_cast<std::size_t>(xy);
    cells             = columns * d.nz;
    const int species = static_cast<int>(d.species);
    if (species < 0 || species > 4 || d.maximum_substeps < 1 || d.maximum_substeps > 4096 || !std::isfinite(d.dt) ||
        d.dt <= 0 || d.i_begin < 0 || d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 || d.j_begin >= d.j_end ||
        d.j_end > d.ny || d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
    {
        return false;
    }
    const auto                       in = inputs(d);
    const std::array<float*, 3>      out{d.mass_tendency, d.number_tendency, d.surface_mass};
    const std::array<std::size_t, 3> sizes{cells, cells, columns};
    for (const auto* field : in)
    {
        if (field == nullptr)
        {
            return false;
        }
    }
    for (std::size_t a = 0; a < out.size(); ++a)
    {
        if (out[a] == nullptr)
        {
            return false;
        }
        for (const auto* field : in)
        {
            if (overlap(out[a], sizes[a], field, cells))
            {
                return false;
            }
        }
        for (std::size_t b = 0; b < a; ++b)
        {
            if (overlap(out[a], sizes[a], out[b], sizes[b]))
            {
                return false;
            }
        }
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
    {
        for (int j = d.j_begin; j < d.j_end; ++j)
        {
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                for (const auto* field : in)
                {
                    if (!std::isfinite(field[n]))
                    {
                        return false;
                    }
                }
                if (d.mass[n] < 0 || d.number[n] < 0 || d.density[n] <= 0 || d.temperature[n] <= 0 ||
                    d.thickness[n] <= 0 || d.snow_speed_multiplier[n] < 0 || !std::isfinite(d.mass[n] * d.density[n]) ||
                    !std::isfinite(d.number[n] * d.density[n]))
                {
                    return false;
                }
                const double volume_number = static_cast<double>(d.number[n]) * d.density[n];
                if ((species == 0 && volume_number > 1999.0e6) || (species < 3 && volume_number > 1.0e12))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

inline abi::ThompsonMomentSedimentationPushConstants constants(const ThompsonMomentSedimentationDispatch& d)
{
    return {d.nx,
            d.ny,
            d.nz,
            d.i_begin,
            d.i_end,
            d.j_begin,
            d.j_end,
            d.k_begin,
            d.k_end,
            static_cast<int>(d.species),
            d.maximum_substeps,
            d.dt};
}

inline bool
valid_candidate(const float* mass, const float* number, const float* surface, std::size_t cells, std::size_t columns)
{
    for (std::size_t n = 0; n < cells; ++n)
    {
        if (!std::isfinite(mass[n]) || !std::isfinite(number[n]))
        {
            return false;
        }
    }
    for (std::size_t n = 0; n < columns; ++n)
    {
        if (!std::isfinite(surface[n]) || surface[n] < 0)
        {
            return false;
        }
    }
    return true;
}
} // namespace nimbus::compute::thompson_sedimentation
