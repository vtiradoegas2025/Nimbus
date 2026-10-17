#pragma once
#include "compute/dispatch/microphysics.hpp"
#include "compute/kernel_abi.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace nimbus::compute::lin
{
using LinPointwiseConstants     = abi::LinPointwisePushConstants;
using LinSedimentationConstants = abi::LinSedimentationPushConstants;
inline bool checked_bytes(int nx, int ny, int nz, std::size_t& cells, std::size_t& bytes) noexcept
{
    if (nx <= 0 || ny <= 0 || nz <= 0 || std::uint64_t(nx) * ny > std::uint64_t(std::numeric_limits<int>::max()) / nz)
        return false;
    cells = std::size_t(nx) * ny * nz;
    if (cells > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return false;
    bytes = cells * sizeof(float);
    return true;
}
template <class Dispatch>
std::array<int, 6> domain(const Dispatch& d) noexcept
{
    return {d.i_begin,
            d.i_end == -1 ? d.nx : d.i_end,
            d.j_begin,
            d.j_end == -1 ? d.ny : d.j_end,
            d.k_begin,
            d.k_end == -1 ? d.nz : d.k_end};
}
template <class Dispatch>
bool valid_domain(const Dispatch& d) noexcept
{
    const auto r = domain(d);
    return r[0] >= 0 && r[0] < r[1] && r[1] <= d.nx && r[2] >= 0 && r[2] < r[3] && r[3] <= d.ny && r[4] >= 0 &&
           r[4] < r[5] && r[5] <= d.nz;
}
template <class Constants, class Dispatch>
void set_domain(Constants& c, const Dispatch& d) noexcept
{
    const auto r = domain(d);
    c.i_begin    = r[0];
    c.i_end      = r[1];
    c.j_begin    = r[2];
    c.j_end      = r[3];
    c.k_begin    = r[4];
    c.k_end      = r[5];
}
inline bool overlap(const float* a, const float* b, std::size_t bytes) noexcept
{
    if (!a || !b)
        return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < bytes : x - y < bytes;
}
template <class Dispatch, std::size_t N>
bool finite_result(const Dispatch& d, const std::array<const float*, N>& fields) noexcept
{
    const auto r = domain(d);
    for (auto* f : fields)
        if (!f)
            return false;
    int valid = 1;
    const auto cells = std::size_t(r[1] - r[0]) * (r[3] - r[2]) * (r[5] - r[4]);
    #pragma omp parallel for collapse(2) schedule(static) reduction(&:valid) if(cells >= 65536)
    for (int i = r[0]; i < r[1]; ++i)
        for (int j = r[2]; j < r[3]; ++j)
            for (int k = r[4]; k < r[5]; ++k)
            {
                const auto n = (std::size_t(i) * d.ny + j) * d.nz + k;
                for (auto* f : fields)
                    valid &= std::isfinite(f[n]);
            }
    return valid != 0;
}
inline bool valid_lin_pointwise(const LinPointwiseDispatch& dispatch, std::size_t& cells, std::size_t& bytes) noexcept
{
    const float* inputs[9]     = {dispatch.theta_data,
                                  dispatch.p_data,
                                  dispatch.qv,
                                  dispatch.qc,
                                  dispatch.qr,
                                  dispatch.qi,
                                  dispatch.qs,
                                  dispatch.qg,
                                  dispatch.qh};
    float*       outputs[8]    = {dispatch.dtheta_dt,
                                  dispatch.dqv_dt,
                                  dispatch.dqc_dt,
                                  dispatch.dqr_dt,
                                  dispatch.dqi_dt,
                                  dispatch.dqs_dt,
                                  dispatch.dqg_dt,
                                  dispatch.dqh_dt};
    const float  constants[16] = {dispatch.dt_step,
                                  dispatch.inv_dt,
                                  dispatch.qc0,
                                  dispatch.c_auto,
                                  dispatch.c_accr,
                                  dispatch.c_evap,
                                  dispatch.c_ihom,
                                  dispatch.c_rime,
                                  dispatch.c_agg,
                                  dispatch.c_melt_s,
                                  dispatch.c_melt_g,
                                  dispatch.c_melt_h,
                                  dispatch.c_subl,
                                  dispatch.Lv_cp,
                                  dispatch.Lf_cp,
                                  dispatch.Ls_cp};
    if (!valid_domain(dispatch) || !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, bytes))
        return false;
    for (const float* input : inputs)
        if (input == nullptr)
            return false;
    for (float* output : outputs)
        if (output == nullptr)
            return false;
    for (float value : constants)
        if (!std::isfinite(value))
            return false;
    if (dispatch.dt_step <= 0.0f || dispatch.inv_dt <= 0.0f || dispatch.qc0 < 0.0f || dispatch.c_auto < 0.0f ||
        dispatch.c_accr < 0.0f || dispatch.c_evap < 0.0f || dispatch.c_ihom < 0.0f || dispatch.c_rime < 0.0f ||
        dispatch.c_agg < 0.0f || dispatch.c_melt_s < 0.0f || dispatch.c_melt_g < 0.0f || dispatch.c_melt_h < 0.0f ||
        dispatch.c_subl < 0.0f || dispatch.Lv_cp <= 0.0f || dispatch.Lf_cp <= 0.0f || dispatch.Ls_cp <= 0.0f)
        return false;
    for (int output = 0; output < 8; ++output)
    {
        for (int prior = 0; prior < output; ++prior)
            if (overlap(outputs[output], outputs[prior], bytes))
                return false;
        for (const float* input : inputs)
            if (overlap(outputs[output], input, bytes))
                return false;
    }
    const auto bounds = domain(dispatch);
    int valid = 1;
    #pragma omp parallel for collapse(2) schedule(static) reduction(&:valid) if(cells >= 65536)
    for (int i = bounds[0]; i < bounds[1]; ++i)
        for (int j = bounds[2]; j < bounds[3]; ++j)
            for (int k = bounds[4]; k < bounds[5]; ++k)
            {
                const auto index = (std::size_t(i) * dispatch.ny + j) * dispatch.nz + k;
                valid &= std::isfinite(inputs[0][index]) && inputs[0][index] > 0.0f &&
                         std::isfinite(inputs[1][index]) && inputs[1][index] > 0.0f;
                for (int field = 2; field < 9; ++field)
                    valid &= std::isfinite(inputs[field][index]) && inputs[field][index] >= 0.0f;
            }
    return valid != 0;
}

inline bool valid_lin_sedimentation(const LinSedimentationDispatch& dispatch,
                                    std::size_t&                    cells,
                                    std::size_t&                    bytes,
                                    bool                            require_finite_tendencies = true) noexcept
{
    const float* inputs[5]  = {dispatch.qr, dispatch.qs, dispatch.qg, dispatch.qh, dispatch.rho_d};
    float*       outputs[4] = {dispatch.dqr_dt, dispatch.dqs_dt, dispatch.dqg_dt, dispatch.dqh_dt};
    if (!valid_domain(dispatch) || !checked_bytes(dispatch.nx, dispatch.ny, dispatch.nz, cells, bytes) ||
        !std::isfinite(dispatch.dz) || dispatch.dz <= 0.0f || !std::isfinite(dispatch.vt_cap) || dispatch.vt_cap < 0.0f)
        return false;
    for (const float* input : inputs)
        if (input == nullptr)
            return false;
    for (float* output : outputs)
        if (output == nullptr)
            return false;
    for (int output = 0; output < 4; ++output)
    {
        for (int prior = 0; prior < output; ++prior)
            if (overlap(outputs[output], outputs[prior], bytes))
                return false;
        for (const float* input : inputs)
            if (overlap(outputs[output], input, bytes))
                return false;
    }
    if (dispatch.layer_thickness_m)
        for (auto* output : outputs)
            if (overlap(output, dispatch.layer_thickness_m, bytes))
                return false;
    const auto bounds = domain(dispatch);
    int valid = 1;
    #pragma omp parallel for collapse(2) schedule(static) reduction(&:valid) if(cells >= 65536)
    for (int i = bounds[0]; i < bounds[1]; ++i)
        for (int j = bounds[2]; j < bounds[3]; ++j)
            for (int k = bounds[4]; k < bounds[5]; ++k)
            {
                const auto index = (std::size_t(i) * dispatch.ny + j) * dispatch.nz + k;
                for (int field = 0; field < 4; ++field)
                    valid &= std::isfinite(inputs[field][index]) && inputs[field][index] >= 0.0f;
                valid &= std::isfinite(inputs[4][index]) && inputs[4][index] > 0.0f;
                if (dispatch.layer_thickness_m &&
                    (!std::isfinite(dispatch.layer_thickness_m[index]) || dispatch.layer_thickness_m[index] <= 0))
                    valid = 0;
                if (require_finite_tendencies)
                    for (float* output : outputs)
                        valid &= std::isfinite(output[index]);
            }
    return valid != 0;
}

inline bool valid_lin_fused(const LinFusedDispatch& d, std::size_t& cells, std::size_t& bytes) noexcept
{
    const auto& point     = d.pointwise;
    const auto& sed       = d.sedimentation;
    std::size_t sed_cells = 0, sed_bytes = 0;
    if (!valid_lin_pointwise(point, cells, bytes) || !valid_lin_sedimentation(sed, sed_cells, sed_bytes, false) ||
        point.nx != sed.nx || point.ny != sed.ny || point.nz != sed.nz || domain(point) != domain(sed) ||
        point.qr != sed.qr || point.qs != sed.qs || point.qg != sed.qg || point.qh != sed.qh ||
        point.dqr_dt != sed.dqr_dt || point.dqs_dt != sed.dqs_dt || point.dqg_dt != sed.dqg_dt ||
        point.dqh_dt != sed.dqh_dt)
        return false;
    for (auto* output : {point.dtheta_dt,
                         point.dqv_dt,
                         point.dqc_dt,
                         point.dqr_dt,
                         point.dqi_dt,
                         point.dqs_dt,
                         point.dqg_dt,
                         point.dqh_dt})
        if (overlap(output, sed.rho_d, bytes) || overlap(output, sed.layer_thickness_m, bytes))
            return false;
    return true;
}

inline LinPointwiseConstants lin_pointwise_constants(const LinPointwiseDispatch& d) noexcept
{
    LinPointwiseConstants c{};
    c.nx       = d.nx;
    c.ny       = d.ny;
    c.nz       = d.nz;
    c.dt_step  = d.dt_step;
    c.inv_dt   = d.inv_dt;
    c.qc0      = d.qc0;
    c.c_auto   = d.c_auto;
    c.c_accr   = d.c_accr;
    c.c_evap   = d.c_evap;
    c.c_ihom   = d.c_ihom;
    c.c_rime   = d.c_rime;
    c.c_agg    = d.c_agg;
    c.c_melt_s = d.c_melt_s;
    c.c_melt_g = d.c_melt_g;
    c.c_melt_h = d.c_melt_h;
    c.c_subl   = d.c_subl;
    c.Lv_cp    = d.Lv_cp;
    c.Lf_cp    = d.Lf_cp;
    c.Ls_cp    = d.Ls_cp;
    set_domain(c, d);
    return c;
}

inline LinSedimentationConstants lin_sedimentation_constants(const LinSedimentationDispatch& d) noexcept
{
    LinSedimentationConstants c{};
    c.nx                   = d.nx;
    c.ny                   = d.ny;
    c.nz                   = d.nz;
    c.dz_val               = d.dz;
    c.vt_cap               = d.vt_cap;
    c.have_rho_d           = 1;
    c.have_layer_thickness = d.layer_thickness_m != nullptr;
    set_domain(c, d);
    return c;
}

} // namespace nimbus::compute::lin
