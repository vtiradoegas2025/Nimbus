#pragma once
#include "compute/dispatch/microphysics.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute
{
inline bool valid_kessler_sedimentation(const KesslerSedimentationDispatch& d)
{
    if (d.nr <= 0 || d.nth <= 0 || d.nz <= 0 ||
        std::uint64_t(d.nr) * d.nth > std::numeric_limits<std::uint32_t>::max() / std::uint64_t(d.nz) ||
        d.i_begin < 0 || d.i_begin >= d.i_end || d.i_end > d.nr || d.j_begin < 0 || d.j_begin >= d.j_end ||
        d.j_end > d.nth || d.k_begin < 0 || d.k_begin >= d.k_end || d.k_end > d.nz || !std::isfinite(d.dz_val) ||
        d.dz_val <= 0 || !std::isfinite(d.dt_step) || d.dt_step <= 0 || !std::isfinite(d.Vt_max_rain) ||
        d.Vt_max_rain < 0 || !std::isfinite(d.vt_cap) || d.vt_cap < 0)
        return false;
    const std::array<const float*, 4> inputs{d.qr_data, d.qg_data, d.qh_data, d.rho_d_data};
    const std::array<float*, 3>       outputs{d.dqr_dt_data, d.dqg_dt_data, d.dqh_dt_data};
    for (auto* input : inputs)
        if (!input)
            return false;
    for (std::size_t f = 0; f < outputs.size(); ++f)
    {
        if (!outputs[f] || outputs[f] == d.layer_thickness_m)
            return false;
        for (auto* input : inputs)
            if (outputs[f] == input)
                return false;
        for (std::size_t prior = 0; prior < f; ++prior)
            if (outputs[f] == outputs[prior])
                return false;
    }
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (std::size_t(i) * d.nth + j) * d.nz + k;
                for (auto* input : inputs)
                    if (!std::isfinite(input[n]))
                        return false;
                for (auto* output : outputs)
                    if (!std::isfinite(output[n]))
                        return false;
                if (d.rho_d_data[n] <= 0 || d.qr_data[n] < 0 ||
                    (d.layer_thickness_m && (!std::isfinite(d.layer_thickness_m[n]) || d.layer_thickness_m[n] <= 0)))
                    return false;
            }
    return true;
}
inline bool finite_kessler_sedimentation(const KesslerSedimentationDispatch& d,
                                         const std::array<const float*, 3>&  result)
{
    for (int i = d.i_begin; i < d.i_end; ++i)
        for (int j = d.j_begin; j < d.j_end; ++j)
            for (int k = d.k_begin; k < d.k_end; ++k)
            {
                const auto n = (std::size_t(i) * d.nth + j) * d.nz + k;
                for (auto* field : result)
                    if (!field || !std::isfinite(field[n]))
                        return false;
            }
    return true;
}
} // namespace nimbus::compute
