#include "terrain/native_earth_geometry.hpp"
#include "compute/dispatch/terrain.hpp"
#include "compute/earth_geometry_contract.hpp"
#include "terrain/base/earth_fixed_geometry_internal.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/conservative_transport.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv::terrain_native
{
namespace
{
using Constants = nimbus::compute::abi::EarthGeometryPushConstants;
using Values    = std::array<std::vector<float>, 20>;
void pack(double value, float& high, float& low)
{
    high = static_cast<float>(value);
    low  = static_cast<float>(value - high);
    if (!std::isfinite(value) || !std::isfinite(high) || !std::isfinite(low))
        throw std::invalid_argument("native Earth geometry exceeds compensated storage");
}
struct GeometryBuilder
{
    TerrainComputeKernels& kernels;
    Constants              constants{};
    int                    nx, ny, nz;
    std::size_t            plane;
    double                 dx_m, dy_m, dz_m;
    GeometryBuilder(TerrainComputeKernels&               backend,
                    const TerrainConfig&                 config,
                    const EarthFixedTerrainBuildRequest& request)
        : kernels(backend), nx(request.nr), ny(request.nth), nz(request.nz)
    {
        earth_fixed_detail::require_build_contract(config, request);
        if (!kernels.supports_earth_geometry())
            throw std::runtime_error("native Earth geometry kernels are unavailable");
        if (request.sampling != EarthFixedTerrainSampling::CellAverageBilinear)
            throw std::invalid_argument("native Earth geometry requires cell averages");
        if (std::size_t(nx) + 4 >
            std::size_t(std::numeric_limits<int>::max()) / (std::size_t(ny) + 4) / (std::size_t(nz) + 1))
            throw std::invalid_argument("native Earth geometry exceeds dispatch dimensions");
        dx_m               = request.dx_m;
        dy_m               = request.dy_m;
        dz_m               = config.ztop / nz;
        plane              = (std::size_t(nx) + 4) * (std::size_t(ny) + 4);
        constants.nx       = nx;
        constants.ny       = ny;
        constants.nz       = nz;
        constants.scheme   = config.scheme_id == "schar" ? 2 : (config.bell.axisymmetric ? 1 : 0);
        constants.smoothed = config.coord_id == "smoothed";
        pack(request.dx_m, constants.dx_high, constants.dx_low);
        pack(request.dy_m, constants.dy_high, constants.dy_low);
        pack(config.ztop / nz, constants.dz_high, constants.dz_low);
        pack(
            config.scheme_id == "schar" ? config.schar.H : config.bell.h0, constants.height_high, constants.height_low);
        pack(config.scheme_id == "schar" ? config.schar.a : config.bell.a, constants.width_high, constants.width_low);
        pack(config.scheme_id == "schar" ? config.schar.ell : 1, constants.wavelength_high, constants.wavelength_low);
        pack(config.smoothing_decay, constants.decay_high, constants.decay_low);
    }
    Values run(Constants c, const std::array<const std::vector<float>*, 8>& input = {})
    {
        EarthGeometryDispatch d;
        d.constants                             = c;
        const auto                        count = nimbus::compute::earth_geometry::cells(d);
        std::array<std::vector<float>, 8> staged;
        Values                            output;
        for (std::size_t f = 0; f < staged.size(); ++f)
        {
            staged[f].assign(count, 0);
            if (input[f])
            {
                if (input[f]->size() > count)
                    throw std::invalid_argument("native Earth geometry binding is too large");
                std::copy(input[f]->begin(), input[f]->end(), staged[f].begin());
            }
            d.input[f] = staged[f].data();
        }
        for (std::size_t f = 0; f < output.size(); ++f)
        {
            output[f].resize(count);
            d.output[f] = output[f].data();
        }
        if (!nimbus::compute::earth_geometry::validate(d))
            throw std::invalid_argument("native Earth geometry dispatch violates its input contract");
        if (!kernels.supports_earth_geometry() || !kernels.dispatch_earth_geometry(d))
            throw std::runtime_error("native Earth geometry dispatch failed");
        std::array<const float*, 20> views;
        for (std::size_t f = 0; f < views.size(); ++f)
            views[f] = output[f].data();
        if (!nimbus::compute::earth_geometry::finite_candidate(views, d))
            throw std::overflow_error("native Earth geometry returned invalid fields");
        return output;
    }
    Values nodes(double first_x, double first_y)
    {
        auto c      = constants;
        c.operation = 0;
        pack(first_x, c.x_high, c.x_low);
        pack(first_y, c.y_high, c.y_low);
        return run(c);
    }
    Values means(const Values& nodes, double x, double y, double dx = 0, double dy = 0, bool average = false)
    {
        auto c      = constants;
        c.operation = 1;
        c.scheme    = average ? 1 : 0;
        pack(x, c.x_high, c.x_low);
        pack(y, c.y_high, c.y_low);
        pack(dx, c.shift_x_high, c.shift_x_low);
        pack(dy, c.shift_y_high, c.shift_y_low);
        return run(c, {&nodes[0], &nodes[1]});
    }
    double value(const Values& output, int field, int i, int j, int k = 0) const
    {
        const auto n = std::size_t(k) * plane + std::size_t(i) * (ny + 4) + j;
        return double(output[2 * field][n]) + output[2 * field + 1][n];
    }
    void unpack(TerrainField3D& field, const Values& output, int component, int sx, int sy, int sz) const
    {
        field.resize(sx, sy, sz, 0);
        for (int i = 0; i < sx; ++i)
            for (int j = 0; j < sy; ++j)
                for (int k = 0; k < sz; ++k)
                    field(i, j, k) = value(output, component, i, j, k);
    }
    EarthFixedTerrainGeometry geometry(const Values& means)
    {
        auto c                           = constants;
        c.operation                      = 2;
        const auto                output = run(c, {&means[0], &means[1], &means[2], &means[3], &means[4], &means[5]});
        EarthFixedTerrainGeometry result;
        auto&                     g = result.metrics;
        unpack(g.z, output, 0, nx, ny, nz);
        unpack(g.J, output, 1, nx, ny, nz);
        unpack(g.mx, output, 2, nx, ny, nz);
        unpack(g.my, output, 3, nx, ny, nz);
        unpack(g.zeta, output, 4, nx, ny, nz);
        unpack(g.z_face, output, 5, nx, ny, nz + 1);
        unpack(g.Jx_face, output, 6, nx + 1, ny, nz);
        unpack(g.Jy_face, output, 7, nx, ny + 1, nz);
        unpack(g.Gx_face, output, 8, nx, ny, nz + 1);
        unpack(g.Gy_face, output, 9, nx, ny, nz + 1);
        g.dx    = dx_m;
        g.dy    = dy_m;
        g.dzeta = dz_m;
        topography::initialize_topography(result.topography, nx, ny);
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                result.topography.h[i][j]     = value(means, 0, i, j);
                result.topography.hx[i][j]    = -g.Gx_face(i, j, 0);
                result.topography.hy[i][j]    = -g.Gy_face(i, j, 0);
                result.diagnostics.max_height = std::max(result.diagnostics.max_height, result.topography.h[i][j]);
                result.diagnostics.max_slope_x =
                    std::max(result.diagnostics.max_slope_x, std::abs(result.topography.hx[i][j]));
                result.diagnostics.max_slope_y =
                    std::max(result.diagnostics.max_slope_y, std::abs(result.topography.hy[i][j]));
            }
        for (const auto* field : {&g.Jx_face, &g.Jy_face})
            for (double value : field->values)
                if (!std::isfinite(value) || value <= terrain_constants::epsilon)
                    throw std::domain_error("native Earth geometry has folded side faces");
        if (!topography::check_coordinate_validity(g, result.diagnostics))
            throw std::domain_error("native Earth geometry has folded cells");
        return result;
    }
    TerrainMeshSweptVolumes sweeps(const TerrainMetrics3D& stage, double dx, double dy, double interval)
    {
        auto c      = constants;
        c.operation = 3;
        pack(dx, c.shift_x_high, c.shift_x_low);
        pack(dy, c.shift_y_high, c.shift_y_low);
        std::array<std::vector<float>, 4> input;
        for (auto& field : input)
            field.assign(plane * (nz + 1), 0);
        for (int component = 0; component < 2; ++component)
        {
            const auto& field = component == 0 ? stage.Jx_face : stage.Jy_face;
            for (int i = 0; i < field.nr; ++i)
                for (int j = 0; j < field.nth; ++j)
                    for (int k = 0; k < field.nz; ++k)
                    {
                        const auto n = std::size_t(k) * plane + std::size_t(i) * (ny + 4) + j;
                        pack(field(i, j, k), input[2 * component][n], input[2 * component + 1][n]);
                    }
        }
        const auto              output = run(c, {&input[0], &input[1], &input[2], &input[3]});
        TerrainMeshSweptVolumes result;
        result.interval_s = interval;
        unpack(result.x, output, 0, nx + 1, ny, nz);
        unpack(result.y, output, 1, nx, ny + 1, nz);
        result.z.resize(nx, ny, nz + 1, 0);
        return result;
    }
};
std::pair<double, double> origin(const EarthFixedTerrainBuildRequest& request)
{
    const double x = (request.grid_origin_earth_m.x_m - request.feature_center_earth_m.x_m) / request.dx_m;
    const double y = (request.grid_origin_earth_m.y_m - request.feature_center_earth_m.y_m) / request.dy_m;
    for (double value : {x, y})
        if (!std::isfinite(value) || std::abs(value) + std::max(request.nr, request.nth) + 4 >= 0x1p48)
            throw std::invalid_argument("native Earth geometry cannot resolve its fixed reference grid");
    return {x, y};
}
} // namespace
EarthFixedTerrainGeometry build_earth_geometry(TerrainComputeKernels&               kernels,
                                               const TerrainConfig&                 config,
                                               const EarthFixedTerrainBuildRequest& request)
{
    GeometryBuilder builder(kernels, config, request);
    const auto [x, y]    = origin(request);
    const double first_x = std::floor(x), first_y = std::floor(y);
    const auto   nodes = builder.nodes(first_x, first_y);
    return builder.geometry(builder.means(nodes, x - first_x, y - first_y));
}
EarthFixedVolumeGeometryInterval build_earth_geometry_interval(TerrainComputeKernels&               kernels,
                                                               const TerrainConfig&                 config,
                                                               const EarthFixedTerrainBuildRequest& begin,
                                                               domain::HorizontalPoint              end_origin,
                                                               double                               interval,
                                                               const TerrainMetrics3D*              expected)
{
    GeometryBuilder builder(kernels, config, begin);
    if (!std::isfinite(interval) || interval <= 0)
        throw std::invalid_argument("native Earth motion requires a positive finite interval");
    auto end                = begin;
    end.grid_origin_earth_m = end_origin;
    earth_fixed_detail::require_build_contract(config, end);
    const auto [x0, y0] = origin(begin);
    const auto [x1, y1] = origin(end);
    const double dx = x1 - x0, dy = y1 - y0;
    if (std::abs(dx) > 1 + 64 * std::numeric_limits<double>::epsilon() ||
        std::abs(dy) > 1 + 64 * std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("native Earth motion spans more than one reference cell per axis");
    const double first_x = std::floor(std::min(x0, x1)), first_y = std::floor(std::min(y0, y1));
    const auto   nodes  = builder.nodes(first_x, first_y);
    const auto   before = builder.geometry(builder.means(nodes, x0 - first_x, y0 - first_y));
    if (expected)
    {
        terrain_transport::validate_geometry(*expected);
        if (expected->J.nr != begin.nr || expected->J.nth != begin.nth || expected->J.nz != begin.nz ||
            expected->dx != begin.dx_m || expected->dy != begin.dy_m || expected->dzeta != config.ztop / begin.nz)
            throw std::invalid_argument("native Earth motion received incompatible beginning metrics");
        const double tolerance = 64 * std::numeric_limits<double>::epsilon() * std::max(1.0, config.ztop);
        for (std::size_t n = 0; n < before.metrics.z_face.size(); ++n)
            if (std::abs(expected->z_face.values[n] - before.metrics.z_face.values[n]) > tolerance)
                throw std::invalid_argument("native Earth motion does not match its beginning surface");
        for (std::size_t n = 0; n < before.metrics.J.size(); ++n)
            if (std::abs(expected->J.values[n] - before.metrics.J.values[n]) * expected->dzeta > tolerance)
                throw std::invalid_argument("native Earth motion does not match its beginning volumes");
    }
    EarthFixedVolumeGeometryInterval result;
    result.end           = builder.geometry(builder.means(nodes, x1 - first_x, y1 - first_y));
    result.stage         = builder.geometry(builder.means(nodes, x0 - first_x, y0 - first_y, dx, dy, true));
    result.swept_volumes = builder.sweeps(result.stage.metrics, dx, dy, interval);
    const double logical = begin.dx_m * begin.dy_m * result.stage.metrics.dzeta;
    for (int i = 0; i < begin.nr; ++i)
        for (int j = 0; j < begin.nth; ++j)
            for (int k = 0; k < begin.nz; ++k)
            {
                const double old_j  = before.metrics.J(i, j, k);
                const double change = logical * (result.end.metrics.J(i, j, k) - old_j);
                const double swept  = result.swept_volumes.x(i + 1, j, k) - result.swept_volumes.x(i, j, k) +
                                      result.swept_volumes.y(i, j + 1, k) - result.swept_volumes.y(i, j, k);
                if (!std::isfinite(swept) || std::abs(change - swept) > 512 * std::numeric_limits<double>::epsilon() *
                                                                            logical * std::max(1.0, std::abs(old_j)))
                    throw std::runtime_error("native Earth motion violates geometric conservation");
            }
    return result;
}
} // namespace tmv::terrain_native

namespace tmv
{
PreparedEarthFixedTerrainCommit prepare_earth_fixed_terrain_commit(const TerrainRuntimeState&           live,
                                                                   const EarthFixedTerrainBuildRequest& request,
                                                                   TerrainComputeKernels&               kernels)
{
    auto prepared     = earth_fixed_detail::prepare_earth_fixed_terrain_identity(live, request);
    prepared.geometry = terrain_native::build_earth_geometry(kernels, live.config, request);
    return prepared;
}

PreparedEarthFixedTerrainMotion prepare_earth_fixed_terrain_motion(const TerrainRuntimeState&           live,
                                                                   const domain::HorizontalDomainFrame& frame,
                                                                   double                               begin_time_s,
                                                                   double                               end_time_s,
                                                                   TerrainComputeKernels&               kernels)
{
    auto identity = earth_fixed_detail::prepare_earth_fixed_motion_identity(live, frame, begin_time_s, end_time_s);
    auto interval = terrain_native::build_earth_geometry_interval(
        kernels, live.config, identity.begin, identity.end, end_time_s - begin_time_s, &live.metrics);
    auto& result            = identity.prepared;
    result.terrain.geometry = std::move(interval.end);
    result.stage_geometry   = std::move(interval.stage);
    result.swept_volumes    = std::move(interval.swept_volumes);
    return std::move(result);
}
} // namespace tmv
