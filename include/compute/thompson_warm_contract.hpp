#pragma once

#include "compute/thompson_cloud_contract.hpp"

namespace nimbus::compute::thompson_warm
{
inline std::array<const float*, 14> inputs(const ThompsonWarmRainDispatch& d)
{
    return {d.temperature,
            d.pressure,
            d.density,
            d.vapor,
            d.cloud,
            d.rain,
            d.cloud_number,
            d.rain_number,
            d.water_aerosol,
            d.cloud_vapor_change,
            d.cloud_number_source,
            d.cloud_aerosol_source,
            d.cloud_vapor_change_low,
            d.coefficients->data()};
}

inline std::array<float*, 10> outputs(const ThompsonWarmRainDispatch& d)
{
    return {d.cloud_to_rain_high,
            d.cloud_to_rain_low,
            d.rain_to_vapor_high,
            d.rain_to_vapor_low,
            d.temperature_high,
            d.temperature_low,
            d.cloud_number_tendency,
            d.rain_number_tendency,
            d.aerosol_tendency,
            d.status};
}

inline bool validate(const ThompsonWarmRainDispatch& d, std::size_t& cells)
{
    if (!d.coefficients || d.coefficients->size() != 20200 || d.nx <= 0 || d.ny <= 0 || d.nz <= 0 ||
        !std::isfinite(d.dt) || d.dt <= 0 || !std::isfinite(static_cast<float>(d.dt)) ||
        static_cast<float>(d.dt) <= 0 || d.i_begin < 0 || d.i_begin >= d.i_end || d.i_end > d.nx || d.j_begin < 0 ||
        d.j_begin >= d.j_end || d.j_end > d.ny || d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz)
    {
        return false;
    }
    const auto columns = static_cast<std::uint64_t>(d.nx) * d.ny;
    if (columns > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / d.nz)
    {
        return false;
    }
    cells = columns * d.nz;
    for (float value : *d.coefficients)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    double previous_edge = 0.0;
    for (std::size_t offset = 0; offset < d.coefficients->size(); offset += 2)
    {
        const double value = static_cast<double>((*d.coefficients)[offset]) + (*d.coefficients)[offset + 1];
        if (offset < 20000)
        {
            if (value < 0.0 || value > 0.95 + 1e-14)
            {
                return false;
            }
        }
        else
        {
            if (value <= previous_edge || value > 0.005)
            {
                return false;
            }
            previous_edge = value;
        }
    }
    const auto in  = inputs(d);
    const auto out = outputs(d);
    for (const auto* field : in)
    {
        if (!field)
        {
            return false;
        }
    }
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        if (!out[i])
        {
            return false;
        }
        for (std::size_t f = 0; f < in.size(); ++f)
        {
            if (thompson_cloud::overlap(out[i], cells, in[f], f < 13 ? cells : d.coefficients->size()))
            {
                return false;
            }
        }
        for (std::size_t j = 0; j < i; ++j)
        {
            if (thompson_cloud::overlap(out[i], cells, out[j], cells))
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
                for (std::size_t f = 0; f < 13; ++f)
                {
                    if (!std::isfinite(in[f][n]))
                    {
                        return false;
                    }
                }
                if (d.temperature[n] <= 0 || d.pressure[n] <= 0 || d.density[n] <= 0 || d.vapor[n] < 0 ||
                    d.cloud[n] < 0 || d.rain[n] < 0 || d.cloud_number[n] < 0 || d.rain_number[n] < 0 ||
                    d.water_aerosol[n] < 0 || d.cloud_number_source[n] != -d.cloud_aerosol_source[n] ||
                    static_cast<double>(d.vapor[n]) + (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) <
                        0 ||
                    static_cast<double>(d.cloud[n]) - (double(d.cloud_vapor_change[n]) + d.cloud_vapor_change_low[n]) <
                        0 ||
                    d.cloud_number[n] + d.dt * d.cloud_number_source[n] < 0 ||
                    d.water_aerosol[n] + d.dt * d.cloud_aerosol_source[n] < 0 ||
                    static_cast<double>(d.density[n]) * d.cloud_number[n] > 1999e6 ||
                    static_cast<double>(d.density[n]) * d.rain_number[n] > 1e12 ||
                    static_cast<double>(d.density[n]) * d.water_aerosol[n] > 9999e6)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

inline abi::ThompsonCloudAdjustmentPushConstants constants(const ThompsonWarmRainDispatch& d)
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
            static_cast<float>(d.dt),
            thompson_cloud::timestep_low(d.dt)};
}

inline bool
valid_candidate(const ThompsonWarmRainDispatch& d, const std::array<const float*, 10>& fields, std::size_t cells)
{
    for (const auto* field : fields)
    {
        for (std::size_t i = 0; i < cells; ++i)
        {
            if (!std::isfinite(field[i]))
            {
                return false;
            }
        }
    }
    for (std::size_t i = 0; i < cells; ++i)
    {
        if (fields[9][i] != 0 || static_cast<double>(fields[0][i]) + fields[1][i] < 0 ||
            static_cast<double>(fields[2][i]) + fields[3][i] < 0)
        {
            return false;
        }
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
    {
        for (int j = d.j_begin; j < d.j_end; ++j)
        {
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (static_cast<std::size_t>(i) * d.ny + j) * d.nz + k;
                if (d.cloud_number[n] + d.dt * fields[6][n] < 0 || d.rain_number[n] + d.dt * fields[7][n] < 0 ||
                    d.water_aerosol[n] + d.dt * fields[8][n] < 0)
                {
                    return false;
                }
            }
        }
    }
    return true;
}
} // namespace nimbus::compute::thompson_warm
