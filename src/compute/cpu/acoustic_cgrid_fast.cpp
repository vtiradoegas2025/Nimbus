#include "cpu_backend.hpp"

#include "core/infra/physical_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace nimbus::compute::cpu_internal
{

bool CpuComputeBackend::dispatch_acoustic_fast_path(
    const AcousticSubstepFusedCartesianCgridDispatch& d,
    const AcousticCgridContext& context)
{
    const int nx = context.nx;
    const int ny = context.ny;
    const int nz = context.nz;
    const double inv_dx = context.inv_dx;
    const double inv_dy = context.inv_dy;
    const double inv_dz = context.inv_dz;
    const double dt_small = context.dt_small;
    const auto idx = [ny, nz](int i, int j, int k) -> std::size_t
    {
        return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny)
                + static_cast<std::size_t>(j))
                   * static_cast<std::size_t>(nz)
               + static_cast<std::size_t>(k);
    };
    const auto finite_or_zero = [](double value)
    {
        return std::isfinite(value) ? value : 0.0;
    };
    const auto safe_rho = [](double value)
    {
        return std::isfinite(value) && value > 1.0e-6 ? value : 1.0;
    };
    const auto clamp_value = [](float value, float fallback,
                                float lo, float hi)
    {
        if (!std::isfinite(value)) value = fallback;
        return std::clamp(value, lo, hi);
    };

    // Preserve the hot CPU-009 kernel without dormant diagnostic
    // reductions or per-cell report branches on ordinary steps.
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    double drho = 0.0;
                    if (i >= 1 && i < nx - 1
                        && j >= 1 && j < ny - 1)
                    {
                        const std::size_t c = idx(i, j, k);
                        const std::size_t im = idx(i - 1, j, k);
                        const std::size_t jm = idx(i, j - 1, k);
                        double fx_pos = 0.0;
                        if (i + 1 < nx)
                        {
                            const std::size_t ip = idx(i + 1, j, k);
                            fx_pos = 0.5 * (
                                static_cast<double>(d.rho[c])
                                + static_cast<double>(d.rho[ip]))
                                * static_cast<double>(d.u[c]);
                        }
                        const double fx_neg = 0.5 * (
                            static_cast<double>(d.rho[im])
                            + static_cast<double>(d.rho[c]))
                            * static_cast<double>(d.u[im]);
                        double fy_pos = 0.0;
                        if (j + 1 < ny)
                        {
                            const std::size_t jp = idx(i, j + 1, k);
                            fy_pos = 0.5 * (
                                static_cast<double>(d.rho[c])
                                + static_cast<double>(d.rho[jp]))
                                * static_cast<double>(d.v[c]);
                        }
                        const double fy_neg = 0.5 * (
                            static_cast<double>(d.rho[jm])
                            + static_cast<double>(d.rho[c]))
                            * static_cast<double>(d.v[jm]);
                        double fz_pos = 0.0;
                        if (k + 1 < nz)
                        {
                            const std::size_t kp = idx(i, j, k + 1);
                            fz_pos = 0.5 * (
                                static_cast<double>(d.rho[c])
                                + static_cast<double>(d.rho[kp]))
                                * static_cast<double>(d.w[c]);
                        }
                        else
                        {
                            fz_pos = static_cast<double>(d.rho[c])
                                * static_cast<double>(d.w[c]);
                        }
                        double fz_neg = 0.0;
                        if (k > 0)
                        {
                            const std::size_t km = idx(i, j, k - 1);
                            fz_neg = 0.5 * (
                                static_cast<double>(d.rho[km])
                                + static_cast<double>(d.rho[c]))
                                * static_cast<double>(d.w[km]);
                        }
                        drho = -((fx_pos - fx_neg) * inv_dx
                                + (fy_pos - fy_neg) * inv_dy
                                + (fz_pos - fz_neg) * inv_dz);
                    }
                    d.drho_scratch[idx(i, j, k)] =
                        static_cast<float>(finite_or_zero(drho));
                }

        #pragma omp for collapse(2)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t c = idx(i, j, k);
                    double rho_new = static_cast<double>(d.rho[c])
                        + static_cast<double>(d.drho_scratch[c])
                            * dt_small;
                    if (!std::isfinite(rho_new) || rho_new <= 0.0)
                        rho_new = std::max(
                            0.1, d.rho0_base_host[k]);
                    const float rho_stored = clamp_value(
                        static_cast<float>(rho_new), 1.0f,
                        d.rho_min, d.rho_max);
                    d.rho[c] = rho_stored;
                    double p_new =
                        static_cast<double>(d.p_ref_fallback);
                    const double th =
                        static_cast<double>(d.theta_data[c]);
                    if (th > 0.0)
                    {
                        p_new =
                            physical_constants::reference_pressure_pa
                            * std::pow(
                                static_cast<double>(rho_stored)
                                    * physical_constants::
                                        gas_constant_dry_air_jkgk
                                    * th
                                    / physical_constants::
                                        reference_pressure_pa,
                                d.gamma_host);
                    }
                    if (!std::isfinite(p_new) || p_new <= 0.0)
                        p_new =
                            static_cast<double>(d.p_ref_fallback);
                    d.p[c] = clamp_value(
                        static_cast<float>(p_new), 101325.0f,
                        d.p_min, d.p_max);
                }

        #pragma omp for collapse(2)
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t c = idx(i, j, k);
                    float du = 0.0f;
                    float dv = 0.0f;
                    float dw = 0.0f;
                    if (i >= 1 && i < nx - 1
                        && j >= 1 && j < ny - 1 && k < nz - 1)
                    {
                        const std::size_t ip = idx(i + 1, j, k);
                        const std::size_t jp = idx(i, j + 1, k);
                        const std::size_t kp = idx(i, j, k + 1);
                        const double rho_x = 0.5 * (
                            static_cast<double>(d.rho[c])
                            + static_cast<double>(d.rho[ip]));
                        const double rho_y = 0.5 * (
                            static_cast<double>(d.rho[c])
                            + static_cast<double>(d.rho[jp]));
                        const double rho_z = 0.5 * (
                            static_cast<double>(d.rho[c])
                            + static_cast<double>(d.rho[kp]));
                        const double dp0_dz =
                            (d.p0_base_host[k + 1]
                             - d.p0_base_host[k]) * inv_dz;
                        du = static_cast<float>(finite_or_zero(
                            -(static_cast<double>(d.p[ip])
                              - static_cast<double>(d.p[c]))
                            * inv_dx / safe_rho(rho_x)));
                        dv = static_cast<float>(finite_or_zero(
                            -(static_cast<double>(d.p[jp])
                              - static_cast<double>(d.p[c]))
                            * inv_dy / safe_rho(rho_y)));
                        dw = static_cast<float>(finite_or_zero(
                            -(((static_cast<double>(d.p[kp])
                                - static_cast<double>(d.p[c]))
                               * inv_dz) - dp0_dz)
                            / safe_rho(rho_z)));
                    }
                    d.u[c] = clamp_value(
                        static_cast<float>(
                            static_cast<double>(d.u[c])
                            + static_cast<double>(du) * dt_small),
                        0.0f, -d.wind_clamp_h, d.wind_clamp_h);
                    d.v[c] = clamp_value(
                        static_cast<float>(
                            static_cast<double>(d.v[c])
                            + static_cast<double>(dv) * dt_small),
                        0.0f, -d.wind_clamp_h, d.wind_clamp_h);
                    d.w[c] = clamp_value(
                        static_cast<float>(
                            static_cast<double>(d.w[c])
                            + static_cast<double>(dw) * dt_small),
                        0.0f, -d.wind_clamp_v, d.wind_clamp_v);
                }
    }
    ++acoustic_fused_dispatches_;
    return true;
}

} // namespace nimbus::compute::cpu_internal
