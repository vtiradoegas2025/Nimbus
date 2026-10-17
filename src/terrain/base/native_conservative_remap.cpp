#include "terrain/native_conservative_remap.hpp"
#include "compute/conservative_remap_contract.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::terrain_native
{
namespace
{
void shape(const TerrainField3D& field, int nx, int ny, int nz)
{
    if (field.nr != nx || field.nth != ny || field.nz != nz ||
        field.size() != std::size_t(nx) * std::size_t(ny) * std::size_t(nz))
        throw std::invalid_argument("native remapping received malformed geometry storage");
}
void shape(const Field3D& field, int nx, int ny, int nz)
{
    if (field.size_r() != nx || field.size_th() != ny || field.size_z() != nz ||
        field.size() != std::size_t(nx) * std::size_t(ny) * std::size_t(nz))
        throw std::invalid_argument("native remapping received malformed scalar storage");
}
struct Candidate
{
    std::array<std::vector<float>, 10> fields;
    MovingMeshInventoryUpdate          report;
};
template <class Value>
Candidate dispatch(TerrainComputeKernels&         kernels,
                   Value                          value,
                   const Field3D*                 rho0,
                   const Field3D*                 rho1,
                   const TerrainMetrics3D&        g0,
                   const TerrainMetrics3D&        g1,
                   const TerrainMeshSweptVolumes& sweep,
                   bool                           store_float,
                   RemapLowerBound                lower_bound)
{
    const int nx = g0.J.nr, ny = g0.J.nth, nz = g0.J.nz;
    if (nx < 3 || ny < 3 || nz < 1 ||
        std::size_t(nx) > std::size_t(std::numeric_limits<int>::max()) / std::size_t(ny) / std::size_t(nz))
        throw std::invalid_argument("native remapping received invalid grid dimensions");
    shape(g0.J, nx, ny, nz);
    shape(g1.J, nx, ny, nz);
    shape(sweep.x, nx + 1, ny, nz);
    shape(sweep.y, nx, ny + 1, nz);
    shape(sweep.z, nx, ny, nz + 1);
    if (!std::isfinite(sweep.interval_s) || sweep.interval_s <= 0)
        throw std::invalid_argument("native remapping requires an advancing mesh interval");
    for (double z : sweep.z.values)
        if (z != 0)
            throw std::invalid_argument("native horizontal remapping cannot discard vertical mesh motion");
    for (const auto* geometry : {&g0, &g1})
        for (double spacing : {geometry->dx, geometry->dy, geometry->dzeta})
            if (!std::isfinite(spacing) || spacing <= 0)
                throw std::invalid_argument("native remapping received invalid logical cell spacing");
    if (rho0)
        shape(*rho0, nx, ny, nz);
    if (rho1)
        shape(*rho1, nx, ny, nz);
    const auto                         count = g0.J.size();
    std::array<std::vector<float>, 16> inputs;
    Candidate                          result;
    ConservativeRemapDispatch          d;
    d.nx           = nx;
    d.ny           = ny;
    d.nz           = nz;
    d.mixing_ratio = rho0 != nullptr;
    d.store_float  = store_float;
    d.lower_bound  = static_cast<int>(lower_bound);
    for (std::size_t f = 0; f < inputs.size(); ++f)
    {
        inputs[f].resize(count);
        d.input[f] = inputs[f].data();
    }
    for (std::size_t f = 0; f < result.fields.size(); ++f)
    {
        result.fields[f].resize(count);
        d.output[f] = result.fields[f].data();
    }
    const auto pack = [&](int field, std::size_t n, double v)
    {
        const float high = static_cast<float>(v);
        const float low  = static_cast<float>(v - high);
        if (!std::isfinite(v) || !std::isfinite(high) || !std::isfinite(low))
            throw std::overflow_error("native remapping input exceeds compensated storage");
        inputs[field][n]     = high;
        inputs[field + 1][n] = low;
    };
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const std::size_t n = (std::size_t(i) * ny + j) * nz + k;
                pack(0, n, value(n));
                inputs[2][n] = rho0 ? rho0->data()[n] : 1;
                inputs[3][n] = rho1 ? rho1->data()[n] : 1;
                pack(4, n, g0.J(i, j, k) * g0.dx * g0.dy * g0.dzeta);
                pack(6, n, g1.J(i, j, k) * g1.dx * g1.dy * g1.dzeta);
                pack(8, n, sweep.x(i, j, k));
                pack(10, n, sweep.x(i + 1, j, k));
                pack(12, n, sweep.y(i, j, k));
                pack(14, n, sweep.y(i, j + 1, k));
            }
    if (!nimbus::compute::conservative_remap::validate(d))
        throw std::invalid_argument("native remapping input violates the inventory contract");
    if (!kernels.supports_conservative_remap() || !kernels.dispatch_conservative_remap(d))
        throw std::runtime_error("native conservative inventory remapping failed");
    std::array<const float*, 10> views;
    for (std::size_t f = 0; f < views.size(); ++f)
        views[f] = result.fields[f].data();
    if (!nimbus::compute::conservative_remap::finite_candidate(views, d))
        throw std::overflow_error("native remapping returned an invalid candidate");
    std::array<diagnostics::CompensatedSum, 4> sums{};
    for (std::size_t n = 0; n < count; ++n)
        for (std::size_t f = 0; f < sums.size(); ++f)
        {
            sums[f].add(result.fields[2 + 2 * f][n]);
            sums[f].add(result.fields[3 + 2 * f][n]);
        }
    for (const auto& sum : sums)
        if (!std::isfinite(sum.value()))
            throw std::overflow_error("native remapping overflowed an inventory ledger");
    result.report = {sums[0].value(), sums[1].value(), sums[2].value(), sums[3].value()};
    return result;
}
} // namespace
MovingMeshInventoryUpdate remap_inventory(TerrainComputeKernels&         kernels,
                                          const Field3D&                 before,
                                          Field3D&                       after,
                                          const Field3D&                 rho0,
                                          const Field3D&                 rho1,
                                          const TerrainMetrics3D&        g0,
                                          const TerrainMetrics3D&        g1,
                                          const TerrainMeshSweptVolumes& sweep,
                                          bool                           mixing_ratio,
                                          RemapLowerBound                lower_bound)
{
    shape(before, g0.J.nr, g0.J.nth, g0.J.nz);
    auto candidate = dispatch(
        kernels,
        [&](std::size_t n)
        {
            return double(before.data()[n]);
        },
        mixing_ratio ? &rho0 : nullptr,
        mixing_ratio ? &rho1 : nullptr,
        g0,
        g1,
        sweep,
        true,
        lower_bound);
    Field3D result = before;
    std::copy(candidate.fields[0].begin(), candidate.fields[0].end(), result.data());
    after = std::move(result);
    return candidate.report;
}
MovingMeshInventoryUpdate remap_inventory(TerrainComputeKernels&         kernels,
                                          const std::vector<double>&     before,
                                          std::vector<double>&           after,
                                          const TerrainMetrics3D&        g0,
                                          const TerrainMetrics3D&        g1,
                                          const TerrainMeshSweptVolumes& sweep,
                                          RemapLowerBound                lower_bound)
{
    if (before.size() != g0.J.size())
        throw std::invalid_argument("native remapping received malformed double storage");
    auto candidate = dispatch(
        kernels,
        [&](std::size_t n)
        {
            return before[n];
        },
        nullptr,
        nullptr,
        g0,
        g1,
        sweep,
        false,
        lower_bound);
    auto result = before;
    for (int i = 1; i < g0.J.nr - 1; ++i)
        for (int j = 1; j < g0.J.nth - 1; ++j)
            for (int k = 0; k < g0.J.nz; ++k)
            {
                const std::size_t n = (std::size_t(i) * g0.J.nth + j) * g0.J.nz + k;
                result[n]           = double(candidate.fields[0][n]) + candidate.fields[1][n];
            }
    after = std::move(result);
    return candidate.report;
}
} // namespace tmv::terrain_native
