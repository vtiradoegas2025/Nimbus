#pragma once

#include "compute/dispatch/dynamics.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

inline bool valid_unrepaired_acoustic_state(const std::array<const float*, 5>& fields, std::size_t count)
{
    for (const float* field : fields) if (!field) return false;
    int valid = 1;
    #pragma omp parallel reduction(&:valid) if(count >= 65536)
    {
        std::size_t worker = 0, workers = 1;
#ifdef _OPENMP
        worker = static_cast<std::size_t>(omp_get_thread_num());
        workers = static_cast<std::size_t>(omp_get_num_threads());
#endif
        const auto begin = count * worker / workers, end = count * (worker + 1) / workers;
        for (int field = 0; field < 5; ++field)
        {
            int field_valid = 1;
            #pragma omp simd reduction(&:field_valid)
            for (std::size_t n = begin; n < end; ++n)
                field_valid &= std::isfinite(fields[field][n]) && (field < 3 || fields[field][n] > 0.0f);
            valid &= field_valid;
        }
    }
    return valid != 0;
}

inline bool valid_acoustic_cgrid_dispatch(const AcousticPressureCartesianCgridDispatch& d)
{
    if (!d.u || !d.v || !d.w || !d.rho_in || !d.p_in || !d.rho0_base_data || !d.p0_base_data || !d.theta_data ||
        !d.rho_out || !d.p_out || d.rho_out == d.p_out || d.base_len < d.nz || d.nx < 3 || d.ny < 3 || d.nz < 2 ||
        !std::isfinite(d.dx) || d.dx <= 0 || !std::isfinite(d.dy) || d.dy <= 0 || !std::isfinite(d.dz) || d.dz <= 0 ||
        !std::isfinite(d.gamma_val) || d.gamma_val <= 0 || d.gamma_val > std::numeric_limits<float>::max() ||
        static_cast<float>(d.gamma_val) <= 0 ||
        !std::isfinite(d.dt_small) || d.dt_small < 0 ||
        !std::isfinite(d.rho_min) || !std::isfinite(d.rho_max) || d.rho_min > d.rho_max || !std::isfinite(d.p_min) ||
        !std::isfinite(d.p_max) || d.p_min > d.p_max || !std::isfinite(d.p_ref_fallback) ||
        !std::isfinite(d.frame_velocity_u_ms) || !std::isfinite(d.frame_velocity_v_ms))
        return false;
    const void* inputs[] = {d.u, d.v, d.w, d.rho_in, d.p_in, d.rho0_base_data, d.p0_base_data, d.theta_data};
    for (int i = 0; i < 8; ++i)
    {
        if ((d.rho_out == inputs[i] && i != 3) || (d.p_out == inputs[i] && i != 4))
            return false;
    }
    return true;
}

inline bool valid_acoustic_cgrid_dispatch(const AcousticMomentumCartesianCgridDispatch& d)
{
    if (!d.rho_data || !d.p_data || !d.dp0_dz_data || !d.u_in || !d.v_in || !d.w_in || !d.u_out || !d.v_out ||
        !d.w_out || d.u_out == d.v_out || d.u_out == d.w_out || d.v_out == d.w_out || d.base_len < d.nz || d.nx < 3 ||
        d.ny < 3 || d.nz < 2 || !std::isfinite(d.dx) || d.dx <= 0 || !std::isfinite(d.dy) || d.dy <= 0 ||
        !std::isfinite(d.dz) || d.dz <= 0 || !std::isfinite(d.dt_small) || d.dt_small < 0 ||
        !std::isfinite(d.wind_clamp_h) || d.wind_clamp_h < 0 || !std::isfinite(d.wind_clamp_v) || d.wind_clamp_v < 0)
        return false;
    const void* inputs[]  = {d.rho_data, d.p_data, d.dp0_dz_data, d.u_in, d.v_in, d.w_in, d.dp0_dz_low_data};
    const void* outputs[] = {d.u_out, d.v_out, d.w_out};
    for (int output = 0; output < 3; ++output)
        for (int input = 0; input < 7; ++input)
            if (outputs[output] == inputs[input] && input != output + 3)
                return false;
    return true;
}

inline bool valid_acoustic_cgrid_dispatch(const AcousticSubstepFusedCartesianCgridDispatch& d)
{
    AcousticPressureCartesianCgridDispatch p;
    p.u = d.u; p.v = d.v; p.w = d.w; p.rho_in = p.rho_out = d.rho; p.p_in = p.p_out = d.p;
    p.rho0_base_data = d.rho0_base_data; p.p0_base_data = d.p0_base_data; p.theta_data = d.theta_data;
    p.base_len = d.base_len; p.nx = d.nx; p.ny = d.ny; p.nz = d.nz;
    p.dx = d.dx; p.dy = d.dy; p.dz = d.dz; p.dt_small = d.dt_small;
    p.gamma_val = static_cast<double>(d.gamma_val) + static_cast<double>(d.gamma_low);
    p.rho_min = d.rho_min; p.rho_max = d.rho_max; p.p_min = d.p_min; p.p_max = d.p_max;
    p.p_ref_fallback = d.p_ref_fallback;
    p.frame_velocity_u_ms = d.frame_velocity_u_ms; p.frame_velocity_v_ms = d.frame_velocity_v_ms;
    AcousticMomentumCartesianCgridDispatch m;
    m.rho_data = d.rho; m.p_data = d.p; m.dp0_dz_data = d.dp0_dz_data;
    m.u_in = m.u_out = d.u; m.v_in = m.v_out = d.v; m.w_in = m.w_out = d.w;
    m.base_len = d.base_len; m.nx = d.nx; m.ny = d.ny; m.nz = d.nz;
    m.dx = d.dx; m.dy = d.dy; m.dz = d.dz; m.dt_small = d.dt_small;
    m.wind_clamp_h = d.wind_clamp_h; m.wind_clamp_v = d.wind_clamp_v;
    m.dp0_dz_low_data = d.dp0_dz_low_data;
    if (!std::isfinite(d.gamma_low) || !valid_acoustic_cgrid_dispatch(p) || !valid_acoustic_cgrid_dispatch(m))
        return false;
    for (const float* immutable : {d.theta_data, d.rho0_base_data, d.p0_base_data, d.dp0_dz_data, d.dp0_dz_low_data})
        for (const float* output : {d.u, d.v, d.w, d.rho, d.p})
            if (immutable != nullptr && immutable == output) return false;
    return true;
}

inline const float* pack_acoustic_base_gradient(const AcousticMomentumCartesianCgridDispatch& d,
                                               std::vector<float>& storage)
{
    if (!d.dp0_dz_low_data)
        return d.dp0_dz_data;
    const auto size = static_cast<std::size_t>(d.nz);
    storage.resize(2 * size);
    std::copy_n(d.dp0_dz_data, size, storage.data());
    std::copy_n(d.dp0_dz_low_data, size, storage.data() + size);
    return storage.data();
}

inline bool valid_strict_acoustic_batch(const AcousticSubstepsBatchedCartesianCgridDispatch& d)
{
    if (d.nx < 4 || d.ny < 4 || d.nz < 2 || d.n_substeps < 1 || d.n_substeps > 4096 ||
        !d.wave_history || !d.wave_history_output || !d.accumulate_mass_flux ||
        !std::isfinite(d.mass_flux_weight) || d.mass_flux_weight <= 0 ||
        std::abs(d.mass_flux_weight * d.n_substeps - 1.0) > 4 * std::numeric_limits<float>::epsilon() ||
        std::size_t(d.nx) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(d.ny) / std::size_t(d.nz))
        return false;
    for (int bc : {d.bc_west, d.bc_east, d.bc_south, d.bc_north})
        if (bc < 0 || bc > 4) return false;
    if (d.bc_bottom < 0 || d.bc_bottom > 1 || d.bc_top < 0 || d.bc_top > 2) return false;
    AcousticSubstepFusedCartesianCgridDispatch fused;
    fused.u = d.u; fused.v = d.v; fused.w = d.w; fused.rho = d.rho; fused.p = d.p;
    fused.theta_data = d.theta_data; fused.rho0_base_data = d.rho0_base_data; fused.p0_base_data = d.p0_base_data;
    fused.dp0_dz_data = d.dp0_dz_data; fused.dp0_dz_low_data = d.dp0_dz_low_data;
    fused.base_len = d.base_len; fused.nx = d.nx; fused.ny = d.ny; fused.nz = d.nz;
    fused.dx = d.dx; fused.dy = d.dy; fused.dz = d.dz; fused.dt_small = d.dt_small;
    fused.gamma_val = d.gamma_val; fused.gamma_low = d.gamma_low;
    fused.rho_min = d.rho_min; fused.rho_max = d.rho_max; fused.p_min = d.p_min; fused.p_max = d.p_max;
    fused.p_ref_fallback = d.p_ref_fallback; fused.wind_clamp_h = d.wind_clamp_h; fused.wind_clamp_v = d.wind_clamp_v;
    fused.frame_velocity_u_ms = d.frame_velocity_u_ms; fused.frame_velocity_v_ms = d.frame_velocity_v_ms;
    if (!(d.dt_small > 0) || !valid_acoustic_cgrid_dispatch(fused)) return false;
    const std::size_t count = std::size_t(d.nx) * d.ny * d.nz;
    const std::size_t waves = 2 * (std::size_t(d.nx) + d.ny) * d.nz;
    const float* outputs[] = {d.u, d.v, d.w, d.rho, d.p, d.mass_flux_x, d.mass_flux_y, d.mass_flux_z, d.wave_history_output};
    const auto overlaps = [](const float* a, std::size_t na, const float* b, std::size_t nb) {
        const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
        return x <= y ? y - x < na * sizeof(float) : x - y < nb * sizeof(float);
    };
    for (int i = 0; i < 9; ++i)
    {
        if (!outputs[i]) return false;
        const auto ni = i == 8 ? waves : count;
        for (int j = 0; j < i; ++j)
            if (overlaps(outputs[i], ni, outputs[j], j == 8 ? waves : count)) return false;
        if (i != 8 && overlaps(outputs[i], ni, d.wave_history, waves)) return false;
        for (const auto& input : {std::pair{d.theta_data, count}, std::pair{d.rho0_base_data, std::size_t(d.nz)},
                                 std::pair{d.p0_base_data, std::size_t(d.nz)}, std::pair{d.dp0_dz_data, std::size_t(d.nz)},
                                 std::pair{d.dp0_dz_low_data, std::size_t(d.nz)}})
            if (input.first && overlaps(outputs[i], ni, input.first, input.second)) return false;
    }
    return true;
}
