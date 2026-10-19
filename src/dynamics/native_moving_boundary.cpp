#include "dynamics/moving_boundary_internal.hpp"
#include "dynamics/native_moving_buffers.hpp"
#include <cmath>
#include <stdexcept>

namespace tmv
{
namespace
{
using moving_transport_detail::MomentumBuffers;
void pack_state(MomentumBuffers&        buffers,
                const TerrainMetrics3D& g,
                const Field3D&          rho,
                const Field3D&          u,
                const Field3D&          v,
                const Field3D*          w,
                int                     first,
                bool                    ground)
{
    auto& p = buffers.dispatch.constants;
    MomentumBuffers::pack(g.dx, p.dx_high, p.dx_low);
    MomentumBuffers::pack(g.dy, p.dy_high, p.dy_low);
    MomentumBuffers::pack(g.dzeta, p.dz_high, p.dz_low);
    for (int k = 0; k < 2; ++k)
    {
        buffers.layer(rho, 0, first + k, k);
        buffers.layer(rho, 1, first + k, k);
        buffers.layer(g.J, 2, first + k, k);
        buffers.layer(g.J, 4, first + k, k);
        buffers.layer(u, 6, first + k, k);
        buffers.layer(v, 7, first + k, k);
        if (w)
            buffers.layer(*w, 8, first + k, k);
    }
    if (ground)
    {
        buffers.layer(g.Gx_face, 15, 0, 0);
        buffers.layer(g.Gy_face, 17, 0, 0);
    }
}
MovingBoundaryLedger reduce_ledger(const MomentumBuffers& buffers)
{
    std::array<diagnostics::CompensatedSum, 7> sums{};
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
    return moving_boundary_detail::ledger(sums);
}
} // namespace
std::array<MovingBoundaryLedger, 2> project_moving_top_momentum(
    const TerrainMetrics3D& g, const Field3D& rho, Field3D& u, Field3D& v, TerrainComputeKernels& kernels)
{
    moving_boundary_detail::validate(g, rho, u, v, nullptr, false);
    moving_boundary_detail::validate_top_outputs(rho, u, v);
    const int       nx = g.J.nr, ny = g.J.nth;
    MomentumBuffers buffers(kernels, nx, ny, 2);
    pack_state(buffers, g, rho, u, v, nullptr, g.J.nz - 2, false);
    buffers.dispatch.constants.operation = 2;
    std::array<Field3D, 2>              means{Field3D(nx, ny, 1), Field3D(nx, ny, 1)};
    std::array<MovingBoundaryLedger, 2> result;
    for (int c = 0; c < 2; ++c)
    {
        buffers.dispatch.constants.component = c;
        buffers.execute(kernels);
        result[c] = reduce_ledger(buffers);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                means[c](i, j, 0) = buffers.output[buffers.index(i, j, 0) * MovingMomentumDispatch::output_stride];
    }
    moving_boundary_detail::publish_top(means, u, v);
    return result;
}
MovingGroundTargets prepare_moving_ground_targets(const TerrainMetrics3D&           g,
                                                  const Field3D&                    rho,
                                                  const Field3D&                    u,
                                                  const Field3D&                    v,
                                                  const TerrainContravariantFluxes& flux,
                                                  TerrainComputeKernels&            kernels)
{
    moving_boundary_detail::validate(g, rho, u, v, nullptr, true);
    moving_boundary_detail::validate_flux(g, flux);
    const int       nx = g.J.nr, ny = g.J.nth;
    MomentumBuffers buffers(kernels, nx, ny, 2);
    pack_state(buffers, g, rho, u, v, nullptr, 0, true);
    buffers.layer(flux.x, 9, 0, 0);
    buffers.layer(flux.y, 11, 0, 0);
    buffers.dispatch.constants.operation = 3;
    buffers.execute(kernels);
    MovingGroundTargets result;
    result.impulse.resize(nx, ny, 1, 0);
    diagnostics::CompensatedSum initial, outward;
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
        {
            const auto* record = buffers.output.data() + buffers.index(i, j, 0) * MovingMomentumDispatch::output_stride;
            const int   exponent    = -int(record[23]);
            result.impulse(i, j, 0) = std::ldexp(double(record[0]) + record[1], exponent);
            initial.add(std::ldexp(double(record[2]), exponent));
            initial.add(std::ldexp(double(record[3]), exponent));
            outward.add(std::ldexp(double(record[4]), exponent));
            outward.add(std::ldexp(double(record[5]), exponent));
        }
    result.initial_impulse = initial.value();
    result.outward_impulse = outward.value();
    if (!std::isfinite(result.initial_impulse) || !std::isfinite(result.outward_impulse))
        throw std::overflow_error("native moving ground target ledger overflowed");
    return result;
}
MovingBoundaryLedger correct_moving_ground_momentum(const TerrainMetrics3D&    g,
                                                    const Field3D&             rho,
                                                    const Field3D&             u,
                                                    const Field3D&             v,
                                                    Field3D&                   w,
                                                    const MovingGroundTargets& targets,
                                                    TerrainComputeKernels&     kernels)
{
    moving_boundary_detail::validate(g, rho, u, v, &w, true);
    moving_boundary_detail::validate_targets(g, targets);
    const int       nx = g.J.nr, ny = g.J.nth;
    MomentumBuffers buffers(kernels, nx, ny, 2);
    pack_state(buffers, g, rho, u, v, &w, 0, true);
    buffers.dispatch.constants.operation = 4;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
        {
            auto*        record = buffers.input.data() + buffers.index(i, j, 0) * MovingMomentumDispatch::input_stride;
            int          exponent = 0;
            const double fraction = std::frexp(targets.impulse(i, j, 0), &exponent);
            MomentumBuffers::pack(fraction, record[9], record[10]);
            record[11] = float(exponent);
        }
    buffers.execute(kernels);
    auto result = reduce_ledger(buffers);
    for (int i = 1; i < nx - 1; ++i)
        for (int j = 1; j < ny - 1; ++j)
            w(i, j, 0) = buffers.output[buffers.index(i, j, 0) * MovingMomentumDispatch::output_stride];
    return result;
}
} // namespace tmv
