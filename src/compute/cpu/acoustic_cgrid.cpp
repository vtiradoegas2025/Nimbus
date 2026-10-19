#include "cpu_backend.hpp"
#include "dynamics/acoustic_energy_report.hpp"

#include <cmath>

namespace nimbus::compute::cpu_internal
{
namespace
{

bool valid_acoustic_cgrid_dispatch(
    const AcousticSubstepFusedCartesianCgridDispatch& d,
    bool attribute_energy) noexcept
{
    const auto valid_clamp_bounds = [](float lower, float upper) noexcept
    {
        return std::isfinite(lower) && std::isfinite(upper)
            && lower > 0.0f && lower <= upper;
    };
    const auto finite_positive = [](auto value) noexcept
    {
        return std::isfinite(value) && value > 0;
    };

    return d.u && d.v && d.w && d.rho && d.p && d.theta_data
        && d.rho0_base_host && d.p0_base_host && d.drho_scratch
        && d.nx >= 3 && d.ny >= 3 && d.nz >= 2
        && d.base_len >= d.nz
        && finite_positive(d.dx_host)
        && finite_positive(d.dy_host)
        && finite_positive(d.dz_host)
        && finite_positive(d.dt_small_host)
        && finite_positive(d.gamma_host)
        && valid_clamp_bounds(d.rho_min, d.rho_max)
        && valid_clamp_bounds(d.p_min, d.p_max)
        && finite_positive(d.p_ref_fallback)
        && std::isfinite(d.wind_clamp_h) && d.wind_clamp_h >= 0.0f
        && std::isfinite(d.wind_clamp_v) && d.wind_clamp_v >= 0.0f
        && (!attribute_energy
            || (d.qv_host && d.qc_host && d.qr_host && d.qi_host
                && d.qs_host && d.qg_host && d.qh_host && d.tke_host
                && (d.energy_report_host->substep_count == 0
                    || d.energy_report_host->complete)
                && (d.energy_report_host->boundary_application_count == 0
                    || d.energy_report_host->boundary_complete)));
}

} // namespace

bool CpuComputeBackend::dispatch_acoustic_substep_fused_cartesian_cgrid(
    const AcousticSubstepFusedCartesianCgridDispatch& d)
{
    if (!d.repair_primary_state) return false;
    const bool attribute_energy = d.energy_report_host != nullptr;
    if (!valid_acoustic_cgrid_dispatch(d, attribute_energy))
    {
        return false;
    }

    const int nx = d.nx;
    const int ny = d.ny;
    const int nz = d.nz;
    const double inv_dx = 1.0 / d.dx_host;
    const double inv_dy = 1.0 / d.dy_host;
    const double inv_dz = 1.0 / d.dz_host;
    const double dt_small = d.dt_small_host;
    const double cell_volume =
        d.dx_host * d.dy_host * d.dz_host;
    const AcousticCgridContext context{
        nx, ny, nz, inv_dx, inv_dy, inv_dz, dt_small, cell_volume,
    };

    return attribute_energy
        ? dispatch_acoustic_energy_path(d, context)
        : dispatch_acoustic_fast_path(d, context);
}

} // namespace nimbus::compute::cpu_internal
