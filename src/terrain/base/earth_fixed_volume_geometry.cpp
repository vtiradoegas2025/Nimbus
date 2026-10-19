#include "terrain/base/earth_fixed_geometry_internal.hpp"
#include "terrain/base/topography.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tmv
{
namespace
{
struct HeightMeans
{
    TerrainField3D cell;
    TerrainField3D x_face;
    TerrainField3D y_face;

    HeightMeans(int nr, int nth)
    {
        cell.resize(nr, nth, 1, 0.0);
        x_face.resize(nr + 1, nth, 1, 0.0);
        y_face.resize(nr, nth + 1, 1, 0.0);
    }

    void add(const HeightMeans& other, double weight)
    {
        for (std::size_t n = 0; n < cell.values.size(); ++n) cell.values[n] += weight * other.cell.values[n];
        for (std::size_t n = 0; n < x_face.values.size(); ++n) x_face.values[n] += weight * other.x_face.values[n];
        for (std::size_t n = 0; n < y_face.values.size(); ++n) y_face.values[n] += weight * other.y_face.values[n];
    }
};

class FixedEarthSurface
{
public:
    FixedEarthSurface(const TerrainConfig& config, const EarthFixedTerrainBuildRequest& request,
                      double x0, double y0, double x1, double y1)
        : config_(config), dx_(request.dx_m), dy_(request.dy_m)
    {
        for (double value : {x0,y0,x1,y1})
            if (!std::isfinite(value) || std::abs(value) + std::max(request.nr,request.nth) + 4.0 >= 0x1p48)
                throw std::invalid_argument("Earth-volume coordinates cannot resolve their fixed reference grid");
        first_x_ = std::floor(std::min(x0,x1));
        first_y_ = std::floor(std::min(y0,y1));
        const double nx = std::floor(std::max(x0,x1)) - first_x_ + request.nr + 2.0;
        const double ny = std::floor(std::max(y0,y1)) - first_y_ + request.nth + 2.0;
        if (nx >= std::numeric_limits<int>::max() || ny >= std::numeric_limits<int>::max())
            throw std::length_error("Earth-volume reference grid is too large");
        nodes_.resize(static_cast<int>(nx),static_cast<int>(ny),1,0.0);
        const bool varies_y = config_.scheme_id == "bell" && config_.bell.axisymmetric;
        for (int i=0; i<nodes_.nr; ++i)
        {
            if (!varies_y)
                std::fill_n(nodes_.values.data()+static_cast<std::size_t>(i)*nodes_.nth,
                    nodes_.nth,evaluate_node(first_x_+i,0.0));
            else
                for (int j=0; j<nodes_.nth; ++j)
                    nodes_(i,j,0) = evaluate_node(first_x_+i,first_y_+j);
        }
    }

    double height(double x, double y) const
    {
        const double ix = std::floor(x), iy = std::floor(y);
        const double tx = x - ix, ty = y - iy;
        const double lower = node(ix,iy) + tx * (node(ix+1.0,iy)-node(ix,iy));
        const double upper = node(ix,iy+1.0) + tx * (node(ix+1.0,iy+1.0)-node(ix,iy+1.0));
        return lower + ty * (upper-lower);
    }

    template<class F>
    static double integrate_unit_interval(double begin, const F& integral)
    {
        const double end = begin + 1.0;
        const double cut = std::min(end, std::floor(begin) + 1.0);
        double result = integral(begin, cut);
        if (cut < end) result += integral(cut, end);
        return result;
    }

    HeightMeans means(int nr, int nth, double x, double y) const
    {
        if (!std::isfinite(x) || !std::isfinite(y)
            || std::abs(x) + nr + 2.0 >= 0x1p48 || std::abs(y) + nth + 2.0 >= 0x1p48)
            throw std::invalid_argument("Earth-volume coordinates cannot resolve their fixed reference grid");
        HeightMeans result(nr, nth);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                result.cell(i,j,0) = integrate_unit_interval(x+i, [&](double a, double b) {
                    return integrate_unit_interval(y+j, [&](double c, double d) {
                        return 0.25 * (b-a) * (d-c)
                            * (height(a,c) + height(b,c) + height(a,d) + height(b,d));
                    });
                });
        for (int i = 0; i <= nr; ++i)
            for (int j = 0; j < nth; ++j)
                result.x_face(i,j,0) = integrate_unit_interval(y+j, [&](double a, double b) {
                    return 0.5 * (b-a) * (height(x+i,a) + height(x+i,b));
                });
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j <= nth; ++j)
                result.y_face(i,j,0) = integrate_unit_interval(x+i, [&](double a, double b) {
                    return 0.5 * (b-a) * (height(a,y+j) + height(b,y+j));
                });
        return result;
    }

private:
    double node(double x, double y) const
    {
        const int i = static_cast<int>(x-first_x_), j = static_cast<int>(y-first_y_);
        if (i < 0 || j < 0 || i >= nodes_.nr || j >= nodes_.nth)
            throw std::out_of_range("Earth-volume quadrature exceeded its reference patch");
        return nodes_(i,j,0);
    }
    double evaluate_node(double x, double y) const
    {
        const double value = config_.scheme_id == "bell"
            ? topography::eval_bell(x*dx_, y*dy_, config_.bell).h
            : topography::eval_schar(x*dx_, config_.schar).h;
        if (!std::isfinite(value) || value < 0.0 || value >= config_.ztop)
            throw std::domain_error("fixed Earth surface has an invalid height");
        return value;
    }
    const TerrainConfig& config_;
    double dx_, dy_;
    double first_x_ = 0.0, first_y_ = 0.0;
    TerrainField3D nodes_;
};

EarthFixedTerrainGeometry geometry_from_means(
    const TerrainConfig& config, const EarthFixedTerrainBuildRequest& request, const HeightMeans& heights)
{
    EarthFixedTerrainGeometry result;
    topography::initialize_topography(result.topography, request.nr, request.nth);
    topography::initialize_metrics(result.metrics, request.nr, request.nth, request.nz);
    for (int i = 0; i < request.nr; ++i)
        for (int j = 0; j < request.nth; ++j)
        {
            result.topography.h[i][j] = heights.cell(i,j,0);
            result.topography.hx[i][j] = (heights.x_face(i+1,j,0) - heights.x_face(i,j,0)) / request.dx_m;
            result.topography.hy[i][j] = (heights.y_face(i,j+1,0) - heights.y_face(i,j,0)) / request.dy_m;
        }
    const auto interfaces = topography::build_zeta_interfaces(request.nz, config.ztop);
    topography::build_conservative_face_metrics(config, result.topography, interfaces,
        request.dx_m, request.dy_m, result.metrics);
    const auto terrain_weight = [&](int k) {
        const double fraction = static_cast<double>(k) / request.nz;
        return (1.0 - fraction) * (config.coord_id == "smoothed"
            ? std::exp(-config.smoothing_decay * fraction) : 1.0);
    };
    for (int i=0; i<request.nr; ++i)
        for (int j=0; j<request.nth; ++j)
        {
            for (int k=0; k<=request.nz; ++k)
                result.metrics.z_face(i,j,k) = interfaces[k] + terrain_weight(k) * heights.cell(i,j,0);
            for (int k=0; k<request.nz; ++k)
                result.metrics.J(i,j,k) = (result.metrics.z_face(i,j,k+1) - result.metrics.z_face(i,j,k)) / result.metrics.dzeta;
        }
    for (int i = 0; i <= request.nr; ++i)
        for (int j = 0; j < request.nth; ++j)
            for (int k = 0; k < request.nz; ++k)
                result.metrics.Jx_face(i,j,k) = (interfaces[k+1]-interfaces[k])/result.metrics.dzeta + heights.x_face(i,j,0)
                    * (terrain_weight(k+1) - terrain_weight(k)) / result.metrics.dzeta;
    for (int i = 0; i < request.nr; ++i)
        for (int j = 0; j <= request.nth; ++j)
            for (int k = 0; k < request.nz; ++k)
                result.metrics.Jy_face(i,j,k) = (interfaces[k+1]-interfaces[k])/result.metrics.dzeta + heights.y_face(i,j,0)
                    * (terrain_weight(k+1) - terrain_weight(k)) / result.metrics.dzeta;
    for (int i = 0; i < request.nr; ++i)
        for (int j = 0; j < request.nth; ++j)
        {
            for (int k = 0; k <= request.nz; ++k)
            {
                result.metrics.Gx_face(i,j,k) = -terrain_weight(k) * result.topography.hx[i][j];
                result.metrics.Gy_face(i,j,k) = -terrain_weight(k) * result.topography.hy[i][j];
            }
            result.diagnostics.max_height = std::max(result.diagnostics.max_height, result.topography.h[i][j]);
            result.diagnostics.max_slope_x = std::max(result.diagnostics.max_slope_x, std::abs(result.topography.hx[i][j]));
            result.diagnostics.max_slope_y = std::max(result.diagnostics.max_slope_y, std::abs(result.topography.hy[i][j]));
        }
    for (const auto* field : {&result.metrics.Jx_face, &result.metrics.Jy_face})
        if (!std::all_of(field->values.begin(), field->values.end(), [](double value) {
            return std::isfinite(value) && value > terrain_constants::epsilon;
        }))
            throw std::domain_error("Earth-volume geometry has folded side faces");
    if (!topography::check_coordinate_validity(result.metrics, result.diagnostics))
        throw std::domain_error("Earth-volume geometry is folded");
    return result;
}

std::pair<double,double> surface_origin(const EarthFixedTerrainBuildRequest& request)
{
    return {(request.grid_origin_earth_m.x_m - request.feature_center_earth_m.x_m) / request.dx_m,
            (request.grid_origin_earth_m.y_m - request.feature_center_earth_m.y_m) / request.dy_m};
}
}

EarthFixedTerrainGeometry build_earth_fixed_volume_geometry(
    const TerrainConfig& config, const EarthFixedTerrainBuildRequest& request)
{
    earth_fixed_detail::require_build_contract(config, request);
    if (request.sampling != EarthFixedTerrainSampling::CellAverageBilinear)
        throw std::invalid_argument("Earth-volume geometry requires cell-average sampling");
    const auto [x,y] = surface_origin(request);
    const FixedEarthSurface surface(config, request, x, y, x, y);
    return geometry_from_means(config, request, surface.means(request.nr, request.nth, x, y));
}

EarthFixedVolumeGeometryInterval build_earth_fixed_volume_geometry_interval(
    const TerrainConfig& config, const EarthFixedTerrainBuildRequest& begin,
    domain::HorizontalPoint end_origin_earth_m, double interval_s,
    const TerrainMetrics3D* expected_begin)
{
    earth_fixed_detail::require_build_contract(config, begin);
    if (begin.sampling != EarthFixedTerrainSampling::CellAverageBilinear
        || !std::isfinite(interval_s) || interval_s <= 0.0)
        throw std::invalid_argument("Earth-volume motion requires cell averages and a positive finite interval");
    auto end = begin;
    end.grid_origin_earth_m = end_origin_earth_m;
    earth_fixed_detail::require_build_contract(config, end);
    const auto [x0,y0] = surface_origin(begin);
    const auto [x1,y1] = surface_origin(end);
    const double delta_x = x1-x0, delta_y = y1-y0;
    if (std::abs(delta_x) > 1.0 + 64.0*std::numeric_limits<double>::epsilon()
        || std::abs(delta_y) > 1.0 + 64.0*std::numeric_limits<double>::epsilon())
        throw std::invalid_argument("Earth-volume motion must span at most one reference cell per axis");
    const FixedEarthSurface surface(config, begin, x0,y0,x1,y1);
    const auto before_heights = surface.means(begin.nr, begin.nth, x0, y0);
    const auto after_heights = surface.means(begin.nr, begin.nth, x1, y1);
    if (expected_begin != nullptr)
    {
        const auto& metrics = *expected_begin;
        if (metrics.J.nr != begin.nr || metrics.J.nth != begin.nth || metrics.J.nz != begin.nz
            || metrics.J.values.size() != static_cast<std::size_t>(begin.nr)*begin.nth*begin.nz
            || metrics.z_face.nr != begin.nr || metrics.z_face.nth != begin.nth || metrics.z_face.nz != begin.nz+1
            || metrics.z_face.values.size() != static_cast<std::size_t>(begin.nr)*begin.nth*(begin.nz+1)
            || metrics.dx != begin.dx_m || metrics.dy != begin.dy_m
            || metrics.dzeta != config.ztop / begin.nz)
            throw std::invalid_argument("Earth-volume motion has incompatible beginning metrics");
        for (int i=0; i<begin.nr; ++i)
            for (int j=0; j<begin.nth; ++j)
                for (int k=0; k<=begin.nz; ++k)
                {
                    const double fraction = static_cast<double>(k)/begin.nz;
                    const double weight = (1.0-fraction) * (config.coord_id == "smoothed"
                        ? std::exp(-config.smoothing_decay*fraction) : 1.0);
                    const double height = k * (config.ztop/begin.nz) + weight*before_heights.cell(i,j,0);
                    const double tolerance = 64.0*std::numeric_limits<double>::epsilon()*std::max(1.0,config.ztop);
                    if (!std::isfinite(metrics.z_face(i,j,k)) || std::abs(metrics.z_face(i,j,k)-height) > tolerance)
                        throw std::invalid_argument("Earth-volume motion does not match its reference surface");
                    if (k < begin.nz)
                    {
                        const double thickness = metrics.z_face(i,j,k+1)-metrics.z_face(i,j,k);
                        if (!std::isfinite(metrics.J(i,j,k)) || metrics.J(i,j,k) <= 0.0
                            || std::abs(metrics.J(i,j,k)*metrics.dzeta-thickness) > tolerance)
                            throw std::invalid_argument("Earth-volume motion has inconsistent beginning volumes");
                    }
                }
    }
    std::vector<double> cuts{0.0, 1.0};
    const auto add_crossings = [&](double start, double finish) {
        if (start == finish) return;
        for (double edge = std::floor(std::min(start,finish)) + 1.0; edge < std::max(start,finish); edge += 1.0)
        {
            const double t = (edge-start)/(finish-start);
            if (t > 0.0 && t < 1.0) cuts.push_back(t);
        }
    };
    add_crossings(x0,x1); add_crossings(y0,y1);
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    HeightMeans stage_heights(begin.nr, begin.nth);
    const double root = std::sqrt(3.0/5.0);
    const std::array<double,3> nodes{-root,0.0,root};
    const std::array<double,3> weights{5.0/9.0,8.0/9.0,5.0/9.0};
    // Fixed-grid bilinear heights give piecewise degree-four moving cell
    // integrals. Split at knot crossings; three Gauss nodes integrate exactly.
    for (std::size_t part = 1; part < cuts.size(); ++part)
    {
        const double midpoint = 0.5 * (cuts[part-1] + cuts[part]);
        const double half = 0.5 * (cuts[part] - cuts[part-1]);
        for (std::size_t q = 0; q < nodes.size(); ++q)
        {
            const double t = midpoint + half * nodes[q];
            stage_heights.add(surface.means(begin.nr, begin.nth, x0+t*delta_x, y0+t*delta_y), half*weights[q]);
        }
    }
    EarthFixedVolumeGeometryInterval result;
    result.stage = geometry_from_means(config, begin, stage_heights);
    result.end = geometry_from_means(config, end, after_heights);
    auto& swept = result.swept_volumes;
    swept.interval_s = interval_s;
    swept.x = result.stage.metrics.Jx_face;
    swept.y = result.stage.metrics.Jy_face;
    swept.z.resize(begin.nr, begin.nth, begin.nz+1, 0.0);
    const double logical_volume = begin.dx_m * begin.dy_m * result.stage.metrics.dzeta;
    for (double& value : swept.x.values) value *= delta_x * logical_volume;
    for (double& value : swept.y.values) value *= delta_y * logical_volume;
    const auto interfaces = topography::build_zeta_interfaces(begin.nz, config.ztop);
    const auto weight = [&](int k) {
        const double fraction = static_cast<double>(k) / begin.nz;
        return (1.0-fraction) * (config.coord_id == "smoothed" ? std::exp(-config.smoothing_decay*fraction) : 1.0);
    };
    for (int i = 0; i < begin.nr; ++i)
        for (int j = 0; j < begin.nth; ++j)
            for (int k = 0; k < begin.nz; ++k)
            {
                const double old_j = (interfaces[k+1]-interfaces[k])/result.stage.metrics.dzeta + before_heights.cell(i,j,0)
                    * (weight(k+1)-weight(k)) / result.stage.metrics.dzeta;
                const double change = logical_volume * (result.end.metrics.J(i,j,k) - old_j);
                const double boundary = swept.x(i+1,j,k) - swept.x(i,j,k)
                    + swept.y(i,j+1,k) - swept.y(i,j,k);
                const double tolerance = 512.0 * std::numeric_limits<double>::epsilon()
                    * logical_volume * std::max(1.0,std::abs(old_j));
                if (!std::isfinite(boundary) || std::abs(change-boundary) > tolerance)
                    throw std::runtime_error("Earth-volume motion does not close its impermeable-surface geometry");
            }
    return result;
}
}
