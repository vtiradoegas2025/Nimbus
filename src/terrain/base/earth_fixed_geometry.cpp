#include "terrain/earth_fixed_geometry.hpp"
#include "terrain/base/earth_fixed_geometry_internal.hpp"

#include "core/infra/grid_coordinates.hpp"
#include "terrain/base/topography.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tmv
{
namespace earth_fixed_detail
{

void require_build_contract(const TerrainConfig& config,
                            const EarthFixedTerrainBuildRequest& request)
{
    if (request.nr <= 0 || request.nth <= 0 || request.nz <= 0
        || !std::isfinite(request.dx_m) || request.dx_m <= 0.0
        || !std::isfinite(request.dy_m) || request.dy_m <= 0.0
        || !std::isfinite(request.grid_origin_earth_m.x_m)
        || !std::isfinite(request.grid_origin_earth_m.y_m)
        || !std::isfinite(request.feature_center_earth_m.x_m)
        || !std::isfinite(request.feature_center_earth_m.y_m))
    {
        throw std::invalid_argument(
            "Earth-fixed terrain requires a finite non-empty horizontal frame");
    }
    const auto checked_multiply = [](std::size_t left, std::size_t right) {
        if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
            throw std::overflow_error("Earth-fixed terrain extent overflow");
        return left * right;
    };
    if (request.nr == std::numeric_limits<int>::max()
        || request.nth == std::numeric_limits<int>::max()
        || request.nz == std::numeric_limits<int>::max())
        throw std::overflow_error("Earth-fixed terrain face extent overflow");
    const auto horizontal = checked_multiply(
        static_cast<std::size_t>(request.nr),
        static_cast<std::size_t>(request.nth));
    const auto cells = checked_multiply(
        horizontal, static_cast<std::size_t>(request.nz));
    if (cells > std::vector<double>{}.max_size())
        throw std::length_error("Earth-fixed terrain exceeds portable storage");
    if (request.sampling != EarthFixedTerrainSampling::Point
        && request.sampling != EarthFixedTerrainSampling::CellAverageBilinear)
        throw std::invalid_argument("invalid Earth-fixed terrain sampling contract");
    if ((config.scheme_id != "bell" && config.scheme_id != "schar")
        || config.horizontal_placement_id != "earth_fixed"
        || (config.coord_id != "btf" && config.coord_id != "smoothed")
        || !config.compute_derivatives || !config.compute_metrics
        || !std::isfinite(config.ztop) || config.ztop <= 0.0
        || !std::isfinite(config.smoothing_decay)
        || config.smoothing_decay < 0.0)
    {
        throw std::invalid_argument(
            "Earth-fixed terrain requires an active analytic scheme and complete metrics");
    }
    if (!std::isfinite(config.feature_center_earth_x_m)
        || !std::isfinite(config.feature_center_earth_y_m)
        || request.feature_center_earth_m.x_m
            != config.feature_center_earth_x_m
        || request.feature_center_earth_m.y_m
            != config.feature_center_earth_y_m)
        throw std::invalid_argument(
            "Earth-fixed terrain request does not match its configured feature center");
    const bool valid_bell = config.scheme_id != "bell"
        || (std::isfinite(config.bell.h0) && config.bell.h0 >= 0.0
            && config.bell.h0 < config.ztop
            && std::isfinite(config.bell.a) && config.bell.a > 0.0);
    const bool valid_schar = config.scheme_id != "schar"
        || (std::isfinite(config.schar.H) && config.schar.H >= 0.0
            && config.schar.H < config.ztop
            && std::isfinite(config.schar.a) && config.schar.a > 0.0
            && std::isfinite(config.schar.ell) && config.schar.ell > 0.0);
    if (!valid_bell || !valid_schar)
        throw std::invalid_argument(
            "Earth-fixed terrain analytic parameters are invalid");
}

} // namespace earth_fixed_detail

namespace
{

bool same_config(const TerrainConfig& left, const TerrainConfig& right) noexcept
{
    return left.scheme_id == right.scheme_id && left.coord_id == right.coord_id &&
           left.horizontal_placement_id == right.horizontal_placement_id && left.ztop == right.ztop &&
           left.compute_derivatives == right.compute_derivatives && left.compute_metrics == right.compute_metrics &&
           left.feature_center_earth_x_m == right.feature_center_earth_x_m &&
           left.feature_center_earth_y_m == right.feature_center_earth_y_m && left.bell.h0 == right.bell.h0 &&
           left.bell.a == right.bell.a && left.bell.axisymmetric == right.bell.axisymmetric &&
           left.schar.H == right.schar.H && left.schar.a == right.schar.a && left.schar.ell == right.schar.ell &&
           left.smoothing_decay == right.smoothing_decay;
}

bool same_placement(const EarthFixedTerrainPlacement& left, const EarthFixedTerrainPlacement& right) noexcept
{
    return left.active == right.active && left.nr == right.nr && left.nth == right.nth && left.nz == right.nz &&
           left.dx_m == right.dx_m && left.dy_m == right.dy_m &&
           left.grid_origin_earth_m.x_m == right.grid_origin_earth_m.x_m &&
           left.grid_origin_earth_m.y_m == right.grid_origin_earth_m.y_m &&
           left.feature_center_earth_m.x_m == right.feature_center_earth_m.x_m &&
           left.feature_center_earth_m.y_m == right.feature_center_earth_m.y_m && left.generation == right.generation &&
           left.sampling == right.sampling;
}

void swap_field(TerrainField3D& left, TerrainField3D& right) noexcept
{
    std::swap(left.nr, right.nr);
    std::swap(left.nth, right.nth);
    std::swap(left.nz, right.nz);
    left.values.swap(right.values);
}

void swap_topography(Topography2D& left, Topography2D& right) noexcept
{
    left.h.swap(right.h);
    left.hx.swap(right.hx);
    left.hy.swap(right.hy);
}

void swap_metrics(TerrainMetrics3D& left, TerrainMetrics3D& right) noexcept
{
    swap_field(left.z, right.z);
    swap_field(left.J, right.J);
    swap_field(left.mx, right.mx);
    swap_field(left.my, right.my);
    swap_field(left.zeta, right.zeta);
    swap_field(left.z_face, right.z_face);
    swap_field(left.Jx_face, right.Jx_face);
    swap_field(left.Jy_face, right.Jy_face);
    swap_field(left.Gx_face, right.Gx_face);
    swap_field(left.Gy_face, right.Gy_face);
    std::swap(left.dx, right.dx);
    std::swap(left.dy, right.dy);
    std::swap(left.dzeta, right.dzeta);
}

} // namespace

static EarthFixedTerrainGeometry build_terrain_geometry(const TerrainConfig&                 config,
                                                        const EarthFixedTerrainBuildRequest& request,
                                                        const TerrainSamplingLattice*        lattice)
{
    earth_fixed_detail::require_build_contract(config, request);
    if (lattice)
    {
        constexpr std::int64_t limit = INT64_C(1) << 51;
        if (request.sampling != EarthFixedTerrainSampling::Point || !std::isfinite(lattice->anchor_earth_m.x_m) ||
            !std::isfinite(lattice->anchor_earth_m.y_m) || lattice->i_origin < -limit || lattice->i_origin > limit ||
            lattice->j_origin < -limit || lattice->j_origin > limit)
            throw std::invalid_argument("terrain sampling lattice is not representable");
    }
    if (request.sampling == EarthFixedTerrainSampling::CellAverageBilinear)
        return build_earth_fixed_volume_geometry(config, request);

    EarthFixedTerrainGeometry candidate;
    topography::initialize_topography(candidate.topography, request.nr, request.nth);
    topography::initialize_metrics(candidate.metrics, request.nr, request.nth, request.nz);

    for (int i = 0; i < request.nr; ++i)
        for (int j = 0; j < request.nth; ++j)
        {
            const double earth_x =
                lattice ? lattice->anchor_earth_m.x_m + (double(lattice->i_origin + i) + .5) * request.dx_m
                        : request.grid_origin_earth_m.x_m + grid_coordinates::scalar_center(i, request.dx_m);
            const double earth_y =
                lattice ? lattice->anchor_earth_m.y_m + (double(lattice->j_origin + j) + .5) * request.dy_m
                        : request.grid_origin_earth_m.y_m + grid_coordinates::scalar_center(j, request.dy_m);
            if (!std::isfinite(earth_x) || !std::isfinite(earth_y))
                throw std::overflow_error("Earth-fixed terrain coordinate overflow");
            const double x = earth_x - request.feature_center_earth_m.x_m;
            const double y = earth_y - request.feature_center_earth_m.y_m;
            if (!std::isfinite(x) || !std::isfinite(y))
                throw std::overflow_error("Earth-fixed terrain feature-relative coordinate overflow");

            if (config.scheme_id == "bell")
            {
                const auto value = topography::eval_bell(x, y, config.bell);
                candidate.topography.h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]  = value.h;
                candidate.topography.hx[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = value.hx;
                candidate.topography.hy[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = value.hy;
            }
            else
            {
                const auto value = topography::eval_schar(x, config.schar);
                candidate.topography.h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]  = value.h;
                candidate.topography.hx[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = value.hx;
                candidate.topography.hy[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.0;
            }
        }

    const auto zeta_interfaces = topography::build_zeta_interfaces(request.nz, config.ztop);
    topography::build_conservative_face_metrics(
        config, candidate.topography, zeta_interfaces, request.dx_m, request.dy_m, candidate.metrics);
    if (!topography::check_coordinate_validity(candidate.metrics, candidate.diagnostics))
    {
        throw std::domain_error("Earth-fixed terrain geometry is folded");
    }

    for (int i = 0; i < request.nr; ++i)
        for (int j = 0; j < request.nth; ++j)
        {
            const auto ui = static_cast<std::size_t>(i);
            const auto uj = static_cast<std::size_t>(j);
            candidate.diagnostics.max_height =
                std::max(candidate.diagnostics.max_height, candidate.topography.h[ui][uj]);
            candidate.diagnostics.max_slope_x =
                std::max(candidate.diagnostics.max_slope_x, std::abs(candidate.topography.hx[ui][uj]));
            candidate.diagnostics.max_slope_y =
                std::max(candidate.diagnostics.max_slope_y, std::abs(candidate.topography.hy[ui][uj]));
        }
    return candidate;
}

namespace earth_fixed_detail
{
PreparedEarthFixedTerrainCommit prepare_earth_fixed_terrain_identity(const TerrainRuntimeState&           live,
                                                                     const EarthFixedTerrainBuildRequest& request)
{
    earth_fixed_detail::require_build_contract(live.config, request);
    if (live.earth_fixed_placement.active)
    {
        const auto& placement = live.earth_fixed_placement;
        if (request.nr != placement.nr || request.nth != placement.nth || request.nz != placement.nz ||
            request.sampling != placement.sampling || request.dx_m != placement.dx_m ||
            request.dy_m != placement.dy_m ||
            request.feature_center_earth_m.x_m != placement.feature_center_earth_m.x_m ||
            request.feature_center_earth_m.y_m != placement.feature_center_earth_m.y_m)
        {
            throw std::invalid_argument("Earth-fixed terrain translation cannot change its grid or feature identity");
        }
        if (placement.generation == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error(
                "Earth-fixed terrain placement generation overflow");
    }

    PreparedEarthFixedTerrainCommit prepared;
    prepared.expected_config = live.config;
    prepared.expected_placement = live.earth_fixed_placement;
    prepared.next_placement.active = true;
    prepared.next_placement.sampling = request.sampling;
    prepared.next_placement.nr = request.nr;
    prepared.next_placement.nth = request.nth;
    prepared.next_placement.nz = request.nz;
    prepared.next_placement.dx_m = request.dx_m;
    prepared.next_placement.dy_m = request.dy_m;
    prepared.next_placement.grid_origin_earth_m = request.grid_origin_earth_m;
    prepared.next_placement.feature_center_earth_m =
        request.feature_center_earth_m;
    prepared.next_placement.generation = live.earth_fixed_placement.active
        ? live.earth_fixed_placement.generation + 1 : 1;
    return prepared;
}
} // namespace earth_fixed_detail

EarthFixedTerrainGeometry build_earth_fixed_terrain_geometry(const TerrainConfig&                 config,
                                                             const EarthFixedTerrainBuildRequest& request)
{
    return build_terrain_geometry(config, request, nullptr);
}

EarthFixedTerrainGeometry build_lattice_anchored_terrain_geometry(const TerrainConfig&                 config,
                                                                  const EarthFixedTerrainBuildRequest& request,
                                                                  const TerrainSamplingLattice&        lattice)
{
    return build_terrain_geometry(config, request, &lattice);
}

PreparedEarthFixedTerrainCommit prepare_earth_fixed_terrain_commit(const TerrainRuntimeState&           live,
                                                                   const EarthFixedTerrainBuildRequest& request)
{
    auto prepared = earth_fixed_detail::prepare_earth_fixed_terrain_identity(live, request);
    prepared.geometry = build_earth_fixed_terrain_geometry(live.config, request);
    return prepared;
}

bool commit_earth_fixed_terrain(
    PreparedEarthFixedTerrainCommit& prepared,
    TerrainRuntimeState& live) noexcept
{
    if (!can_commit_earth_fixed_terrain(prepared, live))
        return false;

    swap_topography(live.topography, prepared.geometry.topography);
    swap_metrics(live.metrics, prepared.geometry.metrics);
    live.earth_fixed_placement = prepared.next_placement;
    prepared.committed = true;
    return true;
}

bool can_commit_earth_fixed_terrain(
    const PreparedEarthFixedTerrainCommit& prepared,
    const TerrainRuntimeState& live) noexcept
{
    return !prepared.committed
        && same_config(live.config, prepared.expected_config)
        && same_placement(
            live.earth_fixed_placement, prepared.expected_placement);
}

TerrainMeshSweptVolumes build_terrain_mesh_swept_volumes(
    const TerrainMetrics3D& before, const TerrainMetrics3D& after,
    domain::HorizontalPoint displacement_m, double interval_s)
{
    const int nr = before.J.nr, nth = before.J.nth, nz = before.J.nz;
    if (nr <= 0 || nth <= 0 || nz <= 0
        || nr == std::numeric_limits<int>::max()
        || nth == std::numeric_limits<int>::max()
        || nz == std::numeric_limits<int>::max()
        || !std::isfinite(interval_s) || interval_s <= 0.0
        || !std::isfinite(displacement_m.x_m) || !std::isfinite(displacement_m.y_m)
        || !std::isfinite(before.dx) || before.dx <= 0.0
        || !std::isfinite(before.dy) || before.dy <= 0.0
        || !std::isfinite(before.dzeta) || before.dzeta <= 0.0
        || after.dx != before.dx || after.dy != before.dy || after.dzeta != before.dzeta)
        throw std::invalid_argument("moving terrain requires compatible finite grid intervals");
    const auto require_shape = [](const TerrainField3D& field, int nx, int ny, int levels)
    {
        const std::size_t size = field.values.size();
        if (field.nr != nx || field.nth != ny || field.nz != levels
            || static_cast<std::size_t>(nx) > size
            || static_cast<std::size_t>(ny) > size / nx
            || static_cast<std::size_t>(levels) > size / nx / ny
            || static_cast<std::size_t>(nx) * ny * levels != size
            || !std::all_of(field.values.begin(), field.values.end(),
                            [](double value) { return std::isfinite(value); }))
            throw std::invalid_argument("moving terrain requires complete finite metric fields");
    };
    for (const auto* metrics : {&before, &after})
    {
        require_shape(metrics->z, nr, nth, nz);
        require_shape(metrics->J, nr, nth, nz);
        require_shape(metrics->mx, nr, nth, nz);
        require_shape(metrics->my, nr, nth, nz);
        require_shape(metrics->zeta, nr, nth, nz);
        require_shape(metrics->z_face, nr, nth, nz + 1);
        require_shape(metrics->Jx_face, nr + 1, nth, nz);
        require_shape(metrics->Jy_face, nr, nth + 1, nz);
        require_shape(metrics->Gx_face, nr, nth, nz + 1);
        require_shape(metrics->Gy_face, nr, nth, nz + 1);
        if (!std::all_of(metrics->Jx_face.values.begin(), metrics->Jx_face.values.end(),
                        [](double value) { return value > 0.0; })
            || !std::all_of(metrics->Jy_face.values.begin(), metrics->Jy_face.values.end(),
                           [](double value) { return value > 0.0; }))
            throw std::invalid_argument("moving terrain has folded side faces");
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const double lower = metrics->z_face(i,j,k);
                    const double upper = metrics->z_face(i,j,k+1);
                    const double thickness = metrics->J(i,j,k) * metrics->dzeta;
                    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
                        * std::max({1.0, std::abs(lower), std::abs(upper), std::abs(thickness)});
                    if (!(lower < metrics->z(i,j,k) && metrics->z(i,j,k) < upper)
                        || metrics->J(i,j,k) <= 0.0
                        || std::abs(thickness - (upper - lower)) > tolerance)
                        throw std::invalid_argument("moving terrain has folded or inconsistent cell volumes");
                    const double d_jx = (metrics->Jx_face(i+1,j,k) - metrics->Jx_face(i,j,k)) / metrics->dx;
                    const double d_jy = (metrics->Jy_face(i,j+1,k) - metrics->Jy_face(i,j,k)) / metrics->dy;
                    const double d_gx = (metrics->Gx_face(i,j,k+1) - metrics->Gx_face(i,j,k)) / metrics->dzeta;
                    const double d_gy = (metrics->Gy_face(i,j,k+1) - metrics->Gy_face(i,j,k)) / metrics->dzeta;
                    const double metric_scale = std::max({1.0 / metrics->dx, 1.0 / metrics->dy,
                        std::abs(d_jx), std::abs(d_jy), std::abs(d_gx), std::abs(d_gy)});
                    if (std::abs(d_jx + d_gx) > 256.0 * std::numeric_limits<double>::epsilon() * metric_scale
                        || std::abs(d_jy + d_gy) > 256.0 * std::numeric_limits<double>::epsilon() * metric_scale)
                        throw std::invalid_argument("moving terrain requires compatible spatial metric identities");
                }
    }
    if (before.zeta.values != after.zeta.values)
        throw std::invalid_argument("moving terrain cannot change the computational vertical partition");
    const double dx = before.dx, dy = before.dy, dzeta = before.dzeta;
    TerrainMeshSweptVolumes result;
    result.interval_s = interval_s;
    result.x.resize(nr + 1, nth, nz, 0.0);
    result.y.resize(nr, nth + 1, nz, 0.0);
    result.z.resize(nr, nth, nz + 1, 0.0);
    for (std::size_t n = 0; n < result.x.values.size(); ++n)
        result.x.values[n] = displacement_m.x_m * dy * dzeta
            * (0.5 * before.Jx_face.values[n] + 0.5 * after.Jx_face.values[n]);
    for (std::size_t n = 0; n < result.y.values.size(); ++n)
        result.y.values[n] = displacement_m.y_m * dx * dzeta
            * (0.5 * before.Jy_face.values[n] + 0.5 * after.Jy_face.values[n]);
    for (std::size_t n = 0; n < result.z.values.size(); ++n)
        result.z.values[n] = dx * dy * (
            after.z_face.values[n] - before.z_face.values[n]
            + displacement_m.x_m * (0.5 * before.Gx_face.values[n] + 0.5 * after.Gx_face.values[n])
            + displacement_m.y_m * (0.5 * before.Gy_face.values[n] + 0.5 * after.Gy_face.values[n]));
    require_shape(result.x, nr + 1, nth, nz);
    require_shape(result.y, nr, nth + 1, nz);
    require_shape(result.z, nr, nth, nz + 1);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double swept = result.x(i+1,j,k) - result.x(i,j,k)
                    + result.y(i,j+1,k) - result.y(i,j,k)
                    + result.z(i,j,k+1) - result.z(i,j,k);
                const double old_volume = dx * dy * dzeta * before.J(i,j,k);
                const double new_volume = dx * dy * dzeta * after.J(i,j,k);
                const double scale = std::max({1.0, old_volume, new_volume,
                    std::abs(result.x(i+1,j,k)), std::abs(result.x(i,j,k)),
                    std::abs(result.y(i,j+1,k)), std::abs(result.y(i,j,k)),
                    std::abs(result.z(i,j,k+1)), std::abs(result.z(i,j,k))});
                if (!std::isfinite(old_volume) || !std::isfinite(new_volume)
                    || !std::isfinite(swept)
                    || std::abs(swept - (new_volume - old_volume))
                        > 512.0 * std::numeric_limits<double>::epsilon() * scale)
                    throw std::invalid_argument("moving terrain violates the geometric conservation law");
            }
    return result;
}

MovingMeshInventoryUpdate update_moving_mesh_inventory(
    const TerrainMetrics3D& before, const TerrainMetrics3D& after,
    const TerrainContravariantFluxes& flux,
    const PhysicalCellDomain& domain, const Field3D& initial, Field3D& result,
    double minimum_value)
{
    const int nr = initial.size_r(), nth = initial.size_th(), nz = initial.size_z();
    if (nr <= 0 || nth <= 0 || nz <= 0 || domain.empty()
        || domain.i_begin < 0 || domain.i_end > nr
        || domain.j_begin < 0 || domain.j_end > nth
        || domain.k_begin < 0 || domain.k_end > nz
        || std::isnan(minimum_value) || minimum_value == std::numeric_limits<double>::infinity()
        || before.dx != after.dx || before.dy != after.dy || before.dzeta != after.dzeta)
        throw std::invalid_argument("moving-mesh inventory has an invalid physical domain");
    const auto has_shape = [](const TerrainField3D& field, int nx, int ny, int levels)
    {
        const auto size = field.values.size();
        return field.nr == nx && field.nth == ny && field.nz == levels
            && static_cast<std::size_t>(nx) <= size
            && static_cast<std::size_t>(ny) <= size / nx
            && static_cast<std::size_t>(levels) <= size / nx / ny
            && static_cast<std::size_t>(nx) * ny * levels == size;
    };
    if (!has_shape(before.J, nr, nth, nz) || !has_shape(after.J, nr, nth, nz)
        || nr == std::numeric_limits<int>::max() || nth == std::numeric_limits<int>::max()
        || nz == std::numeric_limits<int>::max()
        || !has_shape(flux.x, nr + 1, nth, nz)
        || !has_shape(flux.y, nr, nth + 1, nz)
        || !has_shape(flux.z, nr, nth, nz + 1)
        || !std::isfinite(before.dx) || before.dx <= 0.0
        || !std::isfinite(before.dy) || before.dy <= 0.0
        || !std::isfinite(before.dzeta) || before.dzeta <= 0.0)
        throw std::invalid_argument("moving-mesh inventory has malformed geometry or fluxes");
    const double logical_volume = before.dx * before.dy * before.dzeta;
    Field3D candidate = initial;
    struct Sum
    {
        double value = 0.0;
        double correction = 0.0;
        void add(double increment)
        {
            const double adjusted = increment - correction;
            const double next = value + adjusted;
            correction = (next - value) - adjusted;
            value = next;
        }
    };
    Sum initial_sum, final_sum, boundary_sum, rounding_sum;
    MovingMeshInventoryUpdate report;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double old_volume = logical_volume * before.J(i,j,k);
                const double new_volume = logical_volume * after.J(i,j,k);
                const double value = initial(i,j,k);
                const double west = flux.x(i,j,k), east = flux.x(i+1,j,k);
                const double south = flux.y(i,j,k), north = flux.y(i,j+1,k);
                const double bottom = flux.z(i,j,k), top = flux.z(i,j,k+1);
                if (!std::isfinite(value) || value < minimum_value
                    || !std::isfinite(old_volume) || old_volume <= 0.0
                    || !std::isfinite(new_volume) || new_volume <= 0.0
                    || !std::isfinite(west) || !std::isfinite(east)
                    || !std::isfinite(south) || !std::isfinite(north)
                    || !std::isfinite(bottom) || !std::isfinite(top))
                    throw std::invalid_argument("moving-mesh inventory requires finite physical values and fluxes");
                const double old_inventory = old_volume * value;
                const double new_inventory = old_inventory - (east - west + north - south + top - bottom);
                const double next = new_inventory / new_volume;
                const float stored = static_cast<float>(next);
                if (!std::isfinite(next) || !std::isfinite(stored) || next < minimum_value
                    || static_cast<double>(stored) < minimum_value)
                    throw std::domain_error("moving-mesh transport violates its admissible inventory bound");
                candidate(i,j,k) = stored;
                initial_sum.add(old_inventory);
                final_sum.add(static_cast<double>(stored) * new_volume);
                rounding_sum.add((static_cast<double>(stored) - next) * new_volume);
                if (i == domain.i_begin) boundary_sum.add(-west);
                if (i + 1 == domain.i_end) boundary_sum.add(east);
                if (j == domain.j_begin) boundary_sum.add(-south);
                if (j + 1 == domain.j_end) boundary_sum.add(north);
                if (k == domain.k_begin) boundary_sum.add(-bottom);
                if (k + 1 == domain.k_end) boundary_sum.add(top);
            }
    report.initial_inventory = initial_sum.value;
    report.final_inventory = final_sum.value;
    report.boundary_outflow = boundary_sum.value;
    report.storage_rounding = rounding_sum.value;
    if (!std::isfinite(report.initial_inventory) || !std::isfinite(report.final_inventory)
        || !std::isfinite(report.boundary_outflow) || !std::isfinite(report.storage_rounding))
        throw std::overflow_error("moving-mesh inventory totals are not representable");
    result = std::move(candidate);
    return report;
}

earth_fixed_detail::EarthFixedMotionIdentity earth_fixed_detail::prepare_earth_fixed_motion_identity(
    const TerrainRuntimeState& live, const domain::HorizontalDomainFrame& frame, double begin_time_s, double end_time_s)
{
    if (!std::isfinite(begin_time_s) || begin_time_s < 0.0 || !std::isfinite(end_time_s) || end_time_s <= begin_time_s)
        throw std::invalid_argument("Earth-fixed motion requires an advancing finite time interval");
    frame.validate();
    const auto begin = frame.earth_origin_at(begin_time_s);
    const auto end   = frame.earth_origin_at(end_time_s);
    if ((frame.velocity_ms.u_ms != 0.0 && end.x_m == begin.x_m) ||
        (frame.velocity_ms.v_ms != 0.0 && end.y_m == begin.y_m))
        throw std::overflow_error("Earth-fixed displacement is not representable at the declared origin");
    const auto& placement = live.earth_fixed_placement;
    if (!placement.active || placement.grid_origin_earth_m.x_m != begin.x_m ||
        placement.grid_origin_earth_m.y_m != begin.y_m)
        throw std::invalid_argument("Earth-fixed motion does not begin at the committed geometry origin");
    EarthFixedTerrainBuildRequest request;
    request.nr                     = placement.nr;
    request.nth                    = placement.nth;
    request.nz                     = placement.nz;
    request.dx_m                   = placement.dx_m;
    request.dy_m                   = placement.dy_m;
    request.grid_origin_earth_m    = end;
    request.feature_center_earth_m = placement.feature_center_earth_m;
    request.sampling               = placement.sampling;
    PreparedEarthFixedTerrainMotion result;
    result.begin_time_s         = begin_time_s;
    result.end_time_s           = end_time_s;
    result.terrain              = prepare_earth_fixed_terrain_identity(live, request);
    request.grid_origin_earth_m = begin;
    return {std::move(result), request, end};
}

PreparedEarthFixedTerrainMotion prepare_earth_fixed_terrain_motion(const TerrainRuntimeState&           live,
                                                                   const domain::HorizontalDomainFrame& frame,
                                                                   double                               begin_time_s,
                                                                   double                               end_time_s)
{
    auto  identity = earth_fixed_detail::prepare_earth_fixed_motion_identity(live, frame, begin_time_s, end_time_s);
    auto& result   = identity.prepared;
    auto  request  = identity.begin;
    const auto end = identity.end;
    if (request.sampling == EarthFixedTerrainSampling::CellAverageBilinear)
    {
        auto interval = build_earth_fixed_volume_geometry_interval(
            live.config, request, end, end_time_s - begin_time_s, &live.metrics);
        result.terrain.geometry = std::move(interval.end);
        result.stage_geometry   = std::move(interval.stage);
        result.swept_volumes    = std::move(interval.swept_volumes);
    }
    else
    {
        const auto begin            = request.grid_origin_earth_m;
        request.grid_origin_earth_m = end;
        result.terrain.geometry     = build_earth_fixed_terrain_geometry(live.config, request);
        result.swept_volumes        = build_terrain_mesh_swept_volumes(live.metrics,
                                                                       result.terrain.geometry.metrics,
                                                                       {end.x_m - begin.x_m, end.y_m - begin.y_m},
                                                                       end_time_s - begin_time_s);
    }
    return std::move(result);
}

} // namespace tmv
