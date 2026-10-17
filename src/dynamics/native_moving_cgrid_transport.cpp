#include "dynamics/native_moving_cgrid_transport.hpp"
#include "dynamics/native_moving_buffers.hpp"
#include "dynamics/moving_cgrid_transport_internal.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/moving_momentum_contract.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
using moving_transport_detail::MomentumBuffers;
MovingCGridMomentumTransport prepare_moving_cgrid_momentum_transport(const SimulationStateView&        state,
                                                                     const TerrainMetrics3D&           after,
                                                                     const Field3D&                    density_after,
                                                                     const TerrainContravariantFluxes& flux,
                                                                     TerrainComputeKernels&            kernels,
                                                                     bool include_boundary_faces)
{
    const auto&     before = moving_transport_detail::validate(state, after, density_after, flux);
    const int       nx = state.grid.nr, ny = state.grid.nth, nz = state.grid.nz;
    MomentumBuffers buffers(kernels, nx, ny, nz);
    auto&           p  = buffers.dispatch.constants;
    p.include_boundary = include_boundary_faces;
    MomentumBuffers::pack(before.dx, p.dx_high, p.dx_low);
    MomentumBuffers::pack(before.dy, p.dy_high, p.dy_low);
    MomentumBuffers::pack(before.dzeta, p.dz_high, p.dz_low);
    const auto surface = physical_grid_metrics::surface_velocity_ms(state);
    MomentumBuffers::pack(surface.u_ms, p.surface_u_high, p.surface_u_low);
    MomentumBuffers::pack(surface.v_ms, p.surface_v_high, p.surface_v_low);
    buffers.field(state.fields.rho, 0);
    buffers.field(density_after, 1);
    buffers.field(before.J, 2, nx, ny, nz);
    buffers.field(after.J, 4, nx, ny, nz);
    buffers.field(state.fields.u, 6);
    buffers.field(state.fields.v, 7);
    buffers.field(state.fields.w, 8);
    buffers.field(flux.x, 9, nx + 1, ny, nz);
    buffers.field(flux.y, 11, nx, ny + 1, nz);
    buffers.field(flux.z, 13, nx, ny, nz + 1);
    buffers.field(before.Gx_face, 15, nx, ny, nz + 1);
    buffers.field(before.Gy_face, 17, nx, ny, nz + 1);
    MovingCGridMomentumTransport  result{state.fields.u, state.fields.v, state.fields.w, {}};
    const std::array<Field3D*, 3> output{&result.u, &result.v, &result.w};
    for (int component = 0; component < 3; ++component)
    {
        p.component = component;

        buffers.execute(kernels);
        std::array<diagnostics::CompensatedSum, 11> sums{};
        for (std::size_t n = 0; n < nimbus::compute::moving_momentum::cells(buffers.dispatch); ++n)
        {
            const auto* record = buffers.output.data() + n * MovingMomentumDispatch::output_stride;
            for (std::size_t f = 0; f < sums.size(); ++f)
            {
                const int exponent = -int(record[23]) * (f < 4 ? 1 : 2);
                sums[f].add(std::ldexp(double(record[1 + 2 * f]), exponent));
                sums[f].add(std::ldexp(double(record[2 + 2 * f]), exponent));
            }
        }
        for (const auto& sum : sums)
            if (!std::isfinite(sum.value()))
                throw std::overflow_error("native moving-momentum ledger overflowed");
        result.ledger[component] = {sums[0].value(),
                                    sums[1].value(),
                                    sums[2].value(),
                                    sums[3].value(),
                                    sums[4].value(),
                                    sums[5].value(),
                                    sums[6].value(),
                                    sums[7].value(),
                                    sums[8].value(),
                                    sums[9].value(),
                                    sums[10].value()};
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                    (*output[component])(i, j, k) =
                        buffers.output[buffers.index(i, j, k) * MovingMomentumDispatch::output_stride];
    }
    return result;
}
namespace terrain_native
{
TerrainContravariantFluxes
moving_mesh_mass_flux(TerrainComputeKernels& kernels, const Field3D& density, const TerrainMeshSweptVolumes& sweep)
{
    if (!std::isfinite(sweep.interval_s) || sweep.interval_s <= 0)
        throw std::invalid_argument("native moving mass flux requires an advancing interval");
    const int       nx = density.size_r(), ny = density.size_th(), nz = density.size_z();
    MomentumBuffers buffers(kernels, nx, ny, nz);
    buffers.dispatch.constants.operation = 1;
    buffers.field(density, 0);
    buffers.field(sweep.x, 9, nx + 1, ny, nz);
    buffers.field(sweep.y, 11, nx, ny + 1, nz);
    buffers.field(sweep.z, 13, nx, ny, nz + 1);
    buffers.execute(kernels);
    TerrainContravariantFluxes result;
    buffers.flux(result.x, 0, nx + 1, ny, nz);
    buffers.flux(result.y, 2, nx, ny + 1, nz);
    buffers.flux(result.z, 4, nx, ny, nz + 1);
    return result;
}
} // namespace terrain_native
} // namespace tmv
