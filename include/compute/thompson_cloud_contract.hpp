#pragma once

#include "compute/dispatch/microphysics.hpp"
#include "compute/kernel_abi.hpp"
#include "microphysics/base/thermodynamics.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::thompson_cloud
{
inline float timestep_low(double dt) noexcept
{
    const float high = static_cast<float>(dt);
    float       low  = static_cast<float>(dt - high);
    // A downward split can accept a depletion rate whose host FP64 endpoint is negative.
    if (static_cast<double>(high) + low < dt)
    {
        low = std::nextafter(low, std::numeric_limits<float>::infinity());
    }
    return low;
}

inline std::array<const float*, 11> inputs(const ThompsonCloudAdjustmentDispatch& d)
{
    return {d.temperature,
            d.pressure,
            d.density,
            d.vapor,
            d.cloud,
            d.cloud_number,
            d.water_aerosol,
            d.velocity,
            d.tables->activation().data(),
            d.tables->evaporation().data(),
            d.potential_temperature ? d.potential_temperature : d.temperature};
}

inline std::array<float*, 7> outputs(const ThompsonCloudAdjustmentDispatch& d)
{
    return {d.vapor_change,
            d.temperature_change,
            d.cloud_number_tendency,
            d.aerosol_tendency,
            d.lookup_index,
            d.vapor_change_low,
            d.temperature_change_low};
}

inline bool overlap(const float* a, std::size_t na, const float* b, std::size_t nb)
{
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < na * sizeof(float) : x - y < nb * sizeof(float);
}

inline bool validate(const ThompsonCloudAdjustmentDispatch& d, std::size_t& cells)
{
    if (d.nx <= 0 || d.ny <= 0 || d.nz <= 0 || !d.tables || !std::isfinite(d.dt) || d.dt <= 0 ||
        !std::isfinite(static_cast<float>(d.dt)) || static_cast<float>(d.dt) <= 0 || d.i_begin < 0 ||
        d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 || d.j_begin >= d.j_end || d.j_end > d.ny ||
        d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
    {
        return false;
    }
    const auto columns = static_cast<std::uint64_t>(d.nx) * d.ny;
    if (columns > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / d.nz)
    {
        return false;
    }
    cells          = columns * d.nz;
    const auto in  = inputs(d);
    const auto out = outputs(d);
    for (const auto* field : in)
    {
        if (field == nullptr)
        {
            return false;
        }
    }
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        if (out[i] == nullptr)
        {
            return false;
        }
        for (std::size_t j = 0; j < in.size(); ++j)
        {
            const auto count = (j < 8 || j == 10) ? cells
                               : j == 8           ? d.tables->activation().size()
                                                  : d.tables->evaporation().size();
            if (overlap(out[i], cells, in[j], count))
            {
                return false;
            }
        }
        for (std::size_t j = 0; j < i; ++j)
        {
            if (overlap(out[i], cells, out[j], cells))
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
                for (std::size_t f = 0; f < 8; ++f)
                {
                    if (!std::isfinite(in[f][n]))
                    {
                        return false;
                    }
                }
                if (d.potential_temperature &&
                    (!std::isfinite(d.potential_temperature[n]) || d.potential_temperature[n] <= 0 ||
                     d.temperature[n] != static_cast<float>(tmv::thermodynamics::theta_to_temperature(
                                             d.potential_temperature[n], d.pressure[n]))))
                {
                    return false;
                }
                if (d.temperature[n] <= 0 || d.pressure[n] <= 0 || d.density[n] <= 0 || d.vapor[n] < 0 ||
                    d.cloud[n] < 0 || d.cloud_number[n] < 0 || d.water_aerosol[n] < 0 ||
                    static_cast<double>(d.density[n]) * d.cloud_number[n] > 1999.0e6 ||
                    static_cast<double>(d.density[n]) * d.water_aerosol[n] > 9999.0e6 ||
                    !std::isfinite(d.density[n] * d.cloud[n]))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

inline abi::ThompsonCloudSourcePushConstants constants(const ThompsonCloudAdjustmentDispatch& d)
{
    const float timestep_high = static_cast<float>(d.dt);
    const int input_flags = (d.potential_temperature ? 1 : 0) |
                            (static_cast<double>(timestep_high) == d.dt ? 2 : 0);
    return {d.nx,
            d.ny,
            d.nz,
            d.i_begin,
            d.i_end,
            d.j_begin,
            d.j_end,
            d.k_begin,
            d.k_end,
            timestep_high,
            thompson_cloud::timestep_low(d.dt),
            input_flags};
}

inline bool valid_candidate(const std::array<const float*, 7>& fields, std::size_t cells)
{
    for (const auto* field : fields)
    {
        for (std::size_t n = 0; n < cells; ++n)
        {
            if (!std::isfinite(field[n]))
            {
                return false;
            }
        }
    }
    for (std::size_t n = 0; n < cells; ++n)
    {
        if (fields[4][n] < -8 || fields[4][n] >= ThompsonCloudTables::evaporation_count ||
            fields[4][n] != std::floor(fields[4][n]) || fields[2][n] != -fields[3][n])
        {
            return false;
        }
    }
    return true;
}
} // namespace nimbus::compute::thompson_cloud
