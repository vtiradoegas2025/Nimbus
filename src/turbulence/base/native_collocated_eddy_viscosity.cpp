#include "native_collocated_eddy_viscosity.hpp"
#include "compute/backend_runtime.hpp"
#include "util/grid_metric_utils.hpp"
#include <algorithm>
#include <cmath>

namespace tmv::collocated_eddy_viscosity
{
bool compute_native(const TurbulenceConfig&    cfg,
                    const GridMetrics&         grid,
                    const TurbulenceStateView& state,
                    bool                       tke,
                    NativeResult&              result)
{
    auto* backend = state.compute_runtime ? state.compute_runtime->backend.get() : nullptr;
    auto* kernels = backend ? backend->turbulence_kernels() : nullptr;
    if (state.c_grid || grid_metric::has_terrain_metrics(grid) || cfg.filter_width == "user" || cfg.dynamic_Cs ||
        state.defer_transport_tendency_build || !kernels || !kernels->supports(TurbulenceKernelOp::CollocatedSgs) ||
        (cfg.stability_correction != "none" && (!tke || cfg.stability_correction != "tke_based")))
        return false;
    NativeResult next;
    Field3D      zero(state.NR, state.NTH, state.NZ, 0.0f), dz(state.NR, state.NTH, state.NZ, 0.0f);
    for (int i = 0; i < state.NR; ++i)
        for (int j = 0; j < state.NTH; ++j)
            for (int k = 0; k < state.NZ; ++k)
                dz(i, j, k) = float(grid_metric::vertical_cell_thickness(grid, i, j, k, state.NZ));
    CollocatedSgsDispatch d;
    d.nx          = state.NR;
    d.ny          = state.NTH;
    d.nz          = state.NZ;
    d.tke         = tke;
    d.horizontal  = cfg.mode != "vertical_only";
    d.vertical    = cfg.mode != "horizontal_only";
    d.periodic_x  = state.periodic_x;
    d.periodic_y  = state.periodic_y;
    d.stability   = cfg.stability_correction == "tke_based";
    d.moist       = cfg.enable_moist_buoyancy;
    d.has_qv      = state.qv != nullptr;
    d.filter_mode = cfg.filter_width == "dx" ? 1 : 0;
    d.dx          = float(grid.dx);
    d.dy          = float(grid.dy);
    d.Cs          = float(cfg.Cs);
    d.Pr          = float(cfg.Pr_t);
    d.Sc          = float(cfg.Sc_t);
    d.nu_max      = float(cfg.nu_t_max);
    d.K_max       = float(cfg.K_max);
    d.input.fill(zero.data());
    d.input[0] = state.u->data();
    d.input[1] = state.v->data();
    d.input[2] = state.w->data();
    d.input[3] = state.theta->data();
    d.input[4] = state.qv ? state.qv->data() : zero.data();
    d.input[5] = tke ? state.tke->data() : zero.data();
    d.input[6] = state.rho->data();
    d.input[7] = dz.data();
    for (std::size_t f = 0; f < 8; ++f)
    {
        next.closure[f].resize(d.nx, d.ny, d.nz, 0);
        next.tendency[f].resize(d.nx, d.ny, d.nz, 0);
        d.output[f] = next.closure[f].data();
    }
    if (!kernels->dispatch_collocated_sgs(d))
        return false;
    d.phase = 1;
    for (int f = 0; f < 4; ++f)
        d.input[8 + f] = next.closure[f].data();
    d.input[12] = next.closure[4].data();
    d.input[13] = next.closure[5].data();
    d.input[14] = next.closure[7].data();
    for (std::size_t f = 0; f < 8; ++f)
        d.output[f] = next.tendency[f].data();
    if (!kernels->dispatch_collocated_sgs(d))
        return false;
    result = std::move(next);
    return true;
}
void publish_tendencies(NativeResult& result, TurbulenceTendencies& tendencies)
{
    TurbulenceTendencies next;
    next.dudt_sgs                  = std::move(result.tendency[0]);
    next.dvdt_sgs                  = std::move(result.tendency[1]);
    next.dwdt_sgs                  = std::move(result.tendency[2]);
    next.dthetadt_sgs              = std::move(result.tendency[3]);
    next.dqvdt_sgs                 = std::move(result.tendency[4]);
    next.dtkedt_sgs                = std::move(result.tendency[5]);
    next.resident_source_unchanged = false;
    tendencies                     = std::move(next);
}
} // namespace tmv::collocated_eddy_viscosity
