#include "core/infra/nested_terrain_geometry.hpp"

#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tmv::domain
{
namespace
{

[[noreturn]] void reject(const char* reason)
{
    throw std::invalid_argument(std::string("invalid active-terrain nest: ") + reason);
}

void require_finite_point(HorizontalPoint point)
{
    if (!std::isfinite(point.x_m) || !std::isfinite(point.y_m))
        reject("Earth-frame origin must be finite");
}

bool same_relationship(const OneWayNestRelationshipV1& left, const OneWayNestRelationshipV1& right) noexcept
{
    return left.version == right.version && left.parent_cells_x == right.parent_cells_x &&
           left.parent_cells_y == right.parent_cells_y && left.levels_z == right.levels_z &&
           left.parent_ratio_x == right.parent_ratio_x && left.parent_ratio_y == right.parent_ratio_y &&
           left.parent_time_step_ratio == right.parent_time_step_ratio &&
           left.child_origin_parent_cell_i == right.child_origin_parent_cell_i &&
           left.child_origin_parent_cell_j == right.child_origin_parent_cell_j &&
           left.child_cells_x == right.child_cells_x && left.child_cells_y == right.child_cells_y &&
           left.child_parent_span_x == right.child_parent_span_x &&
           left.child_parent_span_y == right.child_parent_span_y &&
           left.relaxation_width_child_cells == right.relaxation_width_child_cells &&
           left.boundary_stencil_parent_cells == right.boundary_stencil_parent_cells &&
           left.parent_dx_m == right.parent_dx_m && left.parent_dy_m == right.parent_dy_m &&
           left.parent_dt_s == right.parent_dt_s && left.child_dx_m == right.child_dx_m &&
           left.child_dy_m == right.child_dy_m && left.child_dt_s == right.child_dt_s;
}

bool same_field(const TerrainField3D& left, const TerrainField3D& right) noexcept
{
    if (left.nr != right.nr || left.nth != right.nth || left.nz != right.nz ||
        left.values.size() != right.values.size())
        return false;
    for (std::size_t index = 0; index < left.values.size(); ++index)
        if (left.values[index] != right.values[index])
            return false;
    return true;
}

bool same_topography(const Topography2D& left, const Topography2D& right) noexcept
{
    const auto same_rows = [](const auto& first, const auto& second) noexcept
    {
        if (first.size() != second.size())
            return false;
        for (std::size_t i = 0; i < first.size(); ++i)
        {
            if (first[i].size() != second[i].size())
                return false;
            for (std::size_t j = 0; j < first[i].size(); ++j)
                if (first[i][j] != second[i][j])
                    return false;
        }
        return true;
    };
    return same_rows(left.h, right.h) && same_rows(left.hx, right.hx) && same_rows(left.hy, right.hy);
}

bool same_metrics(const TerrainMetrics3D& left, const TerrainMetrics3D& right) noexcept
{
    return same_field(left.z, right.z) && same_field(left.J, right.J) && same_field(left.mx, right.mx) &&
           same_field(left.my, right.my) && same_field(left.zeta, right.zeta) &&
           same_field(left.z_face, right.z_face) && same_field(left.Jx_face, right.Jx_face) &&
           same_field(left.Jy_face, right.Jy_face) && same_field(left.Gx_face, right.Gx_face) &&
           same_field(left.Gy_face, right.Gy_face) && left.dx == right.dx && left.dy == right.dy &&
           left.dzeta == right.dzeta;
}

bool same_diagnostics(const TerrainDiagnostics& left, const TerrainDiagnostics& right) noexcept
{
    if (left.max_height != right.max_height || left.max_slope_x != right.max_slope_x ||
        left.max_slope_y != right.max_slope_y || left.min_jacobian != right.min_jacobian ||
        left.max_jacobian != right.max_jacobian || left.coordinate_folding != right.coordinate_folding ||
        left.warnings.size() != right.warnings.size())
        return false;
    for (std::size_t index = 0; index < left.warnings.size(); ++index)
        if (left.warnings[index] != right.warnings[index])
            return false;
    return true;
}

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

bool same_geometry(const NestedMappedTerrainGeometryV1& left, const NestedMappedTerrainGeometryV1& right) noexcept
{
    const auto same_point = [](HorizontalPoint first, HorizontalPoint second) noexcept
    { return first.x_m == second.x_m && first.y_m == second.y_m; };
    const auto same_built = [](const EarthFixedTerrainGeometry& first, const EarthFixedTerrainGeometry& second) noexcept
    {
        return same_topography(first.topography, second.topography) && same_metrics(first.metrics, second.metrics) &&
               same_diagnostics(first.diagnostics, second.diagnostics);
    };
    return same_config(left.config, right.config) && same_relationship(left.relationship, right.relationship) &&
           same_point(left.parent_origin_earth_m, right.parent_origin_earth_m) &&
           same_point(left.child_origin_earth_m, right.child_origin_earth_m) &&
           same_point(left.feature_center_earth_m, right.feature_center_earth_m) &&
           left.sampling_version == right.sampling_version &&
           same_built(left.parent, right.parent) && same_built(left.child_storage, right.child_storage) &&
           left.generation == right.generation;
}

bool same_conservative_state(const NestedMappedConservativeStateV1& left,
                             const NestedMappedConservativeStateV1& right) noexcept
{
    return same_field(left.scalar, right.scalar) && same_field(left.x_flux, right.x_flux) &&
           same_field(left.y_flux, right.y_flux) && same_field(left.zeta_flux, right.zeta_flux);
}

bool same_boundary(const NestedTerrainTangentBoundaryV1& left, const NestedTerrainTangentBoundaryV1& right) noexcept
{
    return same_field(left.relaxation_weight, right.relaxation_weight) &&
           same_field(left.ground_u_ms, right.ground_u_ms) && same_field(left.ground_v_ms, right.ground_v_ms) &&
           same_field(left.ground_w_ms, right.ground_w_ms);
}

bool same_ledger(const NestedTerrainTransferLedgerV1& left, const NestedTerrainTransferLedgerV1& right) noexcept
{
    return left.parent_time_s == right.parent_time_s && left.child_time_s == right.child_time_s &&
           left.parent_scalar_inventory == right.parent_scalar_inventory &&
           left.child_scalar_inventory == right.child_scalar_inventory &&
           left.scalar_inventory_residual == right.scalar_inventory_residual &&
           left.normalized_scalar_inventory_residual == right.normalized_scalar_inventory_residual &&
           left.maximum_x_flux_residual == right.maximum_x_flux_residual &&
           left.maximum_normalized_x_flux_residual == right.maximum_normalized_x_flux_residual &&
           left.maximum_y_flux_residual == right.maximum_y_flux_residual &&
           left.maximum_normalized_y_flux_residual == right.maximum_normalized_y_flux_residual &&
           left.maximum_zeta_flux_residual == right.maximum_zeta_flux_residual &&
           left.maximum_normalized_zeta_flux_residual == right.maximum_normalized_zeta_flux_residual &&
           left.maximum_ground_tangency_residual_ms == right.maximum_ground_tangency_residual_ms &&
           left.relaxation_owned_values == right.relaxation_owned_values;
}

bool same_state(const NestedActiveTerrainStateV1& left, const NestedActiveTerrainStateV1& right) noexcept
{
    return left.ready == right.ready && same_geometry(left.geometry, right.geometry) &&
           same_conservative_state(left.child_state, right.child_state) &&
           same_boundary(left.boundary, right.boundary) && same_ledger(left.ledger, right.ledger) &&
           left.parent_time_s == right.parent_time_s && left.child_time_s == right.child_time_s &&
           left.generation == right.generation && left.publication_count == right.publication_count;
}

void require_relationship(const OneWayNestRelationshipV1& relationship)
{
    OneWayNestRelationshipInput input;
    input.parent_cells_x                = relationship.parent_cells_x;
    input.parent_cells_y                = relationship.parent_cells_y;
    input.parent_levels_z               = relationship.levels_z;
    input.parent_dx_m                   = relationship.parent_dx_m;
    input.parent_dy_m                   = relationship.parent_dy_m;
    input.parent_dt_s                   = relationship.parent_dt_s;
    input.parent_ratio_x                = relationship.parent_ratio_x;
    input.parent_ratio_y                = relationship.parent_ratio_y;
    input.parent_time_step_ratio        = relationship.parent_time_step_ratio;
    input.child_origin_parent_cell_i    = relationship.child_origin_parent_cell_i;
    input.child_origin_parent_cell_j    = relationship.child_origin_parent_cell_j;
    input.child_cells_x                 = relationship.child_cells_x;
    input.child_cells_y                 = relationship.child_cells_y;
    input.child_levels_z                = relationship.levels_z;
    input.relaxation_width_child_cells  = relationship.relaxation_width_child_cells;
    input.boundary_stencil_parent_cells = relationship.boundary_stencil_parent_cells;
    const auto rebuilt                  = prepare_one_way_nest_relationship(input);
    if (!same_relationship(relationship, rebuilt))
        reject("relationship identity is inconsistent");
    const int ratio = relationship.parent_ratio_x;
    if (ratio != relationship.parent_ratio_y ||
        (ratio != 2 && ratio != 3))
        reject("mapped terrain requires isotropic 2:1 or 3:1 horizontal refinement");
    if (relationship.parent_time_step_ratio < ratio)
        reject("mapped terrain time-step ratio is below its horizontal refinement ratio");
}

void require_shape(const TerrainField3D& field, int nr, int nth, int nz, const char* name)
{
    if (field.size_r() != nr || field.size_th() != nth || field.size_z() != nz)
        throw std::invalid_argument(std::string(name) + " has the wrong shape");
    for (double value : field.values)
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string(name) + " is non-finite");
}

void require_parent_state(const NestedMappedTerrainGeometryV1& geometry, const NestedMappedConservativeStateV1& parent)
{
    const auto& relationship = geometry.relationship;
    require_shape(parent.scalar,
                  relationship.parent_cells_x,
                  relationship.parent_cells_y,
                  relationship.levels_z,
                  "parent mapped scalar");
    require_shape(parent.x_flux,
                  relationship.parent_cells_x + 1,
                  relationship.parent_cells_y,
                  relationship.levels_z,
                  "parent mapped x flux");
    require_shape(parent.y_flux,
                  relationship.parent_cells_x,
                  relationship.parent_cells_y + 1,
                  relationship.levels_z,
                  "parent mapped y flux");
    require_shape(parent.zeta_flux,
                  relationship.parent_cells_x,
                  relationship.parent_cells_y,
                  relationship.levels_z + 1,
                  "parent mapped zeta flux");
}

void require_child_state(const NestedMappedTerrainGeometryV1& geometry, const NestedMappedConservativeStateV1& child)
{
    const auto& relationship = geometry.relationship;
    require_shape(child.scalar,
                  relationship.child_cells_x,
                  relationship.child_cells_y,
                  relationship.levels_z,
                  "child mapped scalar");
    require_shape(child.x_flux,
                  relationship.child_cells_x + 1,
                  relationship.child_cells_y,
                  relationship.levels_z,
                  "child mapped x flux");
    require_shape(child.y_flux,
                  relationship.child_cells_x,
                  relationship.child_cells_y + 1,
                  relationship.levels_z,
                  "child mapped y flux");
    require_shape(child.zeta_flux,
                  relationship.child_cells_x,
                  relationship.child_cells_y,
                  relationship.levels_z + 1,
                  "child mapped zeta flux");
}

long double cell_measure(const TerrainMetrics3D& metrics, int i, int j, int k)
{
    return static_cast<long double>(metrics.J(i, j, k)) * static_cast<long double>(metrics.dx) *
           static_cast<long double>(metrics.dy) * static_cast<long double>(metrics.dzeta);
}

long double x_face_measure(const TerrainMetrics3D& metrics, int i, int j, int k)
{
    return static_cast<long double>(metrics.Jx_face(i, j, k)) * static_cast<long double>(metrics.dy) *
           static_cast<long double>(metrics.dzeta);
}

long double y_face_measure(const TerrainMetrics3D& metrics, int i, int j, int k)
{
    return static_cast<long double>(metrics.Jy_face(i, j, k)) * static_cast<long double>(metrics.dx) *
           static_cast<long double>(metrics.dzeta);
}

double finite_double(long double value, const char* reason)
{
    const double converted = static_cast<double>(value);
    if (!std::isfinite(converted))
        throw std::overflow_error(reason);
    return converted;
}

long double scalar_inventory(const TerrainField3D&   field,
                             const TerrainMetrics3D& metrics,
                             int                     field_begin_i,
                             int                     field_begin_j,
                             int                     metric_begin_i,
                             int                     metric_begin_j,
                             int                     cells_i,
                             int                     cells_j)
{
    long double result = 0.0L;
    for (int i = 0; i < cells_i; ++i)
        for (int j = 0; j < cells_j; ++j)
            for (int k = 0; k < metrics.J.size_z(); ++k)
                result += static_cast<long double>(field(field_begin_i + i, field_begin_j + j, k)) *
                          cell_measure(metrics, metric_begin_i + i, metric_begin_j + j, k);
    return result;
}

struct MaximumResidual
{
    long double absolute   = 0.0L;
    long double normalized = 0.0L;
};

MaximumResidual maximum_x_residual(const NestedMappedTerrainGeometryV1&   geometry,
                                   const NestedMappedConservativeStateV1& parent,
                                   const NestedMappedConservativeStateV1& child)
{
    const auto&     r = geometry.relationship;
    MaximumResidual maximum;
    for (int ci = 0; ci <= r.child_cells_x; ci += r.parent_ratio_x)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int         pi       = r.child_origin_parent_cell_i + ci / r.parent_ratio_x;
                const int         parent_j = r.child_origin_parent_cell_j + pj;
                const long double expected = static_cast<long double>(parent.x_flux(pi, parent_j, k)) *
                                             x_face_measure(geometry.parent.metrics, pi, parent_j, k);
                long double       actual   = 0.0L;
                for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                {
                    const int cj = pj * r.parent_ratio_y + fj;
                    actual += static_cast<long double>(child.x_flux(ci, cj, k)) *
                              x_face_measure(geometry.child_storage.metrics, ci + 1, cj + 1, k);
                }
                const long double residual = std::abs(actual - expected);
                const long double scale    = std::max(1.0L, std::max(std::abs(actual), std::abs(expected)));
                maximum.absolute           = std::max(maximum.absolute, residual);
                maximum.normalized         = std::max(maximum.normalized, residual / scale);
            }
    return maximum;
}

MaximumResidual maximum_y_residual(const NestedMappedTerrainGeometryV1&   geometry,
                                   const NestedMappedConservativeStateV1& parent,
                                   const NestedMappedConservativeStateV1& child)
{
    const auto&     r = geometry.relationship;
    MaximumResidual maximum;
    for (int cj = 0; cj <= r.child_cells_y; cj += r.parent_ratio_y)
        for (int pi = 0; pi < r.child_parent_span_x; ++pi)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int         parent_i = r.child_origin_parent_cell_i + pi;
                const int         pj       = r.child_origin_parent_cell_j + cj / r.parent_ratio_y;
                const long double expected = static_cast<long double>(parent.y_flux(parent_i, pj, k)) *
                                             y_face_measure(geometry.parent.metrics, parent_i, pj, k);
                long double       actual   = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                {
                    const int ci = pi * r.parent_ratio_x + fi;
                    actual += static_cast<long double>(child.y_flux(ci, cj, k)) *
                              y_face_measure(geometry.child_storage.metrics, ci + 1, cj + 1, k);
                }
                const long double residual = std::abs(actual - expected);
                const long double scale    = std::max(1.0L, std::max(std::abs(actual), std::abs(expected)));
                maximum.absolute           = std::max(maximum.absolute, residual);
                maximum.normalized         = std::max(maximum.normalized, residual / scale);
            }
    return maximum;
}

MaximumResidual maximum_zeta_residual(const NestedMappedTerrainGeometryV1&   geometry,
                                      const NestedMappedConservativeStateV1& parent,
                                      const NestedMappedConservativeStateV1& child)
{
    const auto&     r = geometry.relationship;
    MaximumResidual maximum;
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k <= r.levels_z; ++k)
            {
                const int         parent_i = r.child_origin_parent_cell_i + pi;
                const int         parent_j = r.child_origin_parent_cell_j + pj;
                const long double expected = static_cast<long double>(parent.zeta_flux(parent_i, parent_j, k)) *
                                             static_cast<long double>(r.parent_dx_m) *
                                             static_cast<long double>(r.parent_dy_m);
                long double       actual   = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                    {
                        const int ci = pi * r.parent_ratio_x + fi;
                        const int cj = pj * r.parent_ratio_y + fj;
                        actual += static_cast<long double>(child.zeta_flux(ci, cj, k)) *
                                  static_cast<long double>(r.child_dx_m) * static_cast<long double>(r.child_dy_m);
                    }
                const long double residual = std::abs(actual - expected);
                const long double scale    = std::max(1.0L, std::max(std::abs(actual), std::abs(expected)));
                maximum.absolute           = std::max(maximum.absolute, residual);
                maximum.normalized         = std::max(maximum.normalized, residual / scale);
            }
    return maximum;
}

} // namespace

NestedMappedTerrainGeometryV1 build_stationary_nested_terrain_geometry(const TerrainConfig&            config,
                                                                       const OneWayNestRelationshipV1& relationship,
                                                                       HorizontalPoint parent_origin_earth_m,
                                                                       std::uint64_t   generation,
                                                                       std::uint32_t   sampling_version)
{
    require_relationship(relationship);
    require_finite_point(parent_origin_earth_m);
    if (generation == 0 || (sampling_version != 1 && sampling_version != 2))
        reject("geometry generation and sampling version must be valid");
    const bool attached = config.horizontal_placement_id == "grid_attached";
    if ((config.scheme_id != "bell" && config.scheme_id != "schar") ||
        (!attached && config.horizontal_placement_id != "earth_fixed"))
        reject("nested geometry requires grid-attached or Earth-fixed Bell or Schar terrain");
    if (attached && (config.feature_center_earth_x_m != 0.0 || config.feature_center_earth_y_m != 0.0))
        reject("grid-attached nested terrain cannot carry an Earth-fixed feature center");

    const double child_x = parent_origin_earth_m.x_m +
                           static_cast<double>(relationship.child_origin_parent_cell_i) * relationship.parent_dx_m;
    const double child_y = parent_origin_earth_m.y_m +
                           static_cast<double>(relationship.child_origin_parent_cell_j) * relationship.parent_dy_m;
    if (!std::isfinite(child_x) || !std::isfinite(child_y))
        throw std::overflow_error("nested terrain child origin overflow");

    HorizontalPoint feature_center;
    feature_center.x_m = attached ? parent_origin_earth_m.x_m + 0.5 * static_cast<double>(relationship.parent_cells_x) *
                                                                    relationship.parent_dx_m
                                  : config.feature_center_earth_x_m;
    feature_center.y_m = attached ? parent_origin_earth_m.y_m + 0.5 * static_cast<double>(relationship.parent_cells_y) *
                                                                    relationship.parent_dy_m
                                  : config.feature_center_earth_y_m;
    require_finite_point(feature_center);

    TerrainConfig build_config            = config;
    build_config.horizontal_placement_id  = "earth_fixed";
    build_config.feature_center_earth_x_m = feature_center.x_m;
    build_config.feature_center_earth_y_m = feature_center.y_m;

    EarthFixedTerrainBuildRequest parent_request;
    parent_request.nr                     = relationship.parent_cells_x;
    parent_request.nth                    = relationship.parent_cells_y;
    parent_request.nz                     = relationship.levels_z;
    parent_request.dx_m                   = relationship.parent_dx_m;
    parent_request.dy_m                   = relationship.parent_dy_m;
    parent_request.grid_origin_earth_m    = parent_origin_earth_m;
    parent_request.feature_center_earth_m = feature_center;

    EarthFixedTerrainBuildRequest child_request;
    if (relationship.child_cells_x > std::numeric_limits<int>::max() - 2 ||
        relationship.child_cells_y > std::numeric_limits<int>::max() - 2)
        throw std::overflow_error("nested terrain child halo extent overflow");
    child_request.nr                     = relationship.child_cells_x + 2;
    child_request.nth                    = relationship.child_cells_y + 2;
    child_request.nz                     = relationship.levels_z;
    child_request.dx_m                   = relationship.child_dx_m;
    child_request.dy_m                   = relationship.child_dy_m;
    child_request.grid_origin_earth_m    = {child_x - relationship.child_dx_m, child_y - relationship.child_dy_m};
    child_request.feature_center_earth_m = feature_center;

    NestedMappedTerrainGeometryV1 result;
    result.config                 = config;
    result.relationship           = relationship;
    result.parent_origin_earth_m  = parent_origin_earth_m;
    result.child_origin_earth_m   = {child_x, child_y};
    result.feature_center_earth_m = feature_center;
    result.parent                 = build_earth_fixed_terrain_geometry(build_config, parent_request);
    result.sampling_version       = sampling_version;
    result.child_storage =
        sampling_version == 1
            ? build_earth_fixed_terrain_geometry(build_config, child_request)
            : build_lattice_anchored_terrain_geometry(
                  build_config,
                  child_request,
                  {parent_origin_earth_m,
                   std::int64_t(relationship.child_origin_parent_cell_i) * relationship.parent_ratio_x - 1,
                   std::int64_t(relationship.child_origin_parent_cell_j) * relationship.parent_ratio_y - 1});
    terrain_transport::validate_geometry(result.parent.metrics);
    terrain_transport::validate_geometry(result.child_storage.metrics);
    result.generation = generation;
    return result;
}

NestedMappedConservativeStateV1 prolong_nested_mapped_conservative_state(const NestedMappedTerrainGeometryV1& geometry,
                                                                         const NestedMappedConservativeStateV1& parent,
                                                                         NestedTerrainTransferLedgerV1*         ledger)
{
    require_relationship(geometry.relationship);
    terrain_transport::validate_geometry(geometry.parent.metrics);
    terrain_transport::validate_geometry(geometry.child_storage.metrics);
    require_parent_state(geometry, parent);
    const auto& r = geometry.relationship;

    NestedMappedConservativeStateV1 child;
    child.scalar.resize(r.child_cells_x, r.child_cells_y, r.levels_z, 0.0);
    child.x_flux.resize(r.child_cells_x + 1, r.child_cells_y, r.levels_z, 0.0);
    child.y_flux.resize(r.child_cells_x, r.child_cells_y + 1, r.levels_z, 0.0);
    child.zeta_flux.resize(r.child_cells_x, r.child_cells_y, r.levels_z + 1, 0.0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int         parent_i     = r.child_origin_parent_cell_i + pi;
                const int         parent_j     = r.child_origin_parent_cell_j + pj;
                const long double inventory    = static_cast<long double>(parent.scalar(parent_i, parent_j, k)) *
                                                 cell_measure(geometry.parent.metrics, parent_i, parent_j, k);
                long double       fine_measure = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                        fine_measure += cell_measure(geometry.child_storage.metrics,
                                                     pi * r.parent_ratio_x + fi + 1,
                                                     pj * r.parent_ratio_y + fj + 1,
                                                     k);
                const double value = static_cast<double>(inventory / fine_measure);
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                        child.scalar(pi * r.parent_ratio_x + fi, pj * r.parent_ratio_y + fj, k) = value;
            }

#pragma omp parallel for collapse(3) schedule(static)
    for (int ci = 0; ci <= r.child_cells_x; ++ci)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int         coarse_i    = ci / r.parent_ratio_x;
                const int         remainder_i = ci % r.parent_ratio_x;
                const int         parent_left = r.child_origin_parent_cell_i + coarse_i;
                const int         parent_j    = r.child_origin_parent_cell_j + pj;
                const long double alpha =
                    static_cast<long double>(remainder_i) / static_cast<long double>(r.parent_ratio_x);
                const int parent_right =
                    std::min(parent_left + 1, r.child_origin_parent_cell_i + r.child_parent_span_x);
                const long double left_integral  = static_cast<long double>(parent.x_flux(parent_left, parent_j, k)) *
                                                   x_face_measure(geometry.parent.metrics, parent_left, parent_j, k);
                const long double right_integral = static_cast<long double>(parent.x_flux(parent_right, parent_j, k)) *
                                                   x_face_measure(geometry.parent.metrics, parent_right, parent_j, k);
                const long double target         = (1.0L - alpha) * left_integral + alpha * right_integral;
                long double       fine_measure   = 0.0L;
                for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                    fine_measure +=
                        x_face_measure(geometry.child_storage.metrics, ci + 1, pj * r.parent_ratio_y + fj + 1, k);
                const double value = static_cast<double>(target / fine_measure);
                for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                    child.x_flux(ci, pj * r.parent_ratio_y + fj, k) = value;
            }

#pragma omp parallel for collapse(3) schedule(static)
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int cj = 0; cj <= r.child_cells_y; ++cj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int         coarse_j     = cj / r.parent_ratio_y;
                const int         remainder_j  = cj % r.parent_ratio_y;
                const int         parent_i     = r.child_origin_parent_cell_i + pi;
                const int         parent_south = r.child_origin_parent_cell_j + coarse_j;
                const long double alpha =
                    static_cast<long double>(remainder_j) / static_cast<long double>(r.parent_ratio_y);
                const int parent_north =
                    std::min(parent_south + 1, r.child_origin_parent_cell_j + r.child_parent_span_y);
                const long double south_integral = static_cast<long double>(parent.y_flux(parent_i, parent_south, k)) *
                                                   y_face_measure(geometry.parent.metrics, parent_i, parent_south, k);
                const long double north_integral = static_cast<long double>(parent.y_flux(parent_i, parent_north, k)) *
                                                   y_face_measure(geometry.parent.metrics, parent_i, parent_north, k);
                const long double target         = (1.0L - alpha) * south_integral + alpha * north_integral;
                long double       fine_measure   = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    fine_measure +=
                        y_face_measure(geometry.child_storage.metrics, pi * r.parent_ratio_x + fi + 1, cj + 1, k);
                const double value = static_cast<double>(target / fine_measure);
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    child.y_flux(pi * r.parent_ratio_x + fi, cj, k) = value;
            }

#pragma omp parallel for collapse(3) schedule(static)
    for (int ci = 0; ci < r.child_cells_x; ++ci)
        for (int cj = 0; cj < r.child_cells_y; ++cj)
            for (int k = 0; k <= r.levels_z; ++k)
            {
                const int parent_i         = r.child_origin_parent_cell_i + ci / r.parent_ratio_x;
                const int parent_j         = r.child_origin_parent_cell_j + cj / r.parent_ratio_y;
                child.zeta_flux(ci, cj, k) = parent.zeta_flux(parent_i, parent_j, k);
            }

    require_shape(child.scalar, r.child_cells_x, r.child_cells_y, r.levels_z, "child mapped scalar candidate");
    require_shape(child.x_flux, r.child_cells_x + 1, r.child_cells_y, r.levels_z, "child mapped x-flux candidate");
    require_shape(child.y_flux, r.child_cells_x, r.child_cells_y + 1, r.levels_z, "child mapped y-flux candidate");
    require_shape(
        child.zeta_flux, r.child_cells_x, r.child_cells_y, r.levels_z + 1, "child mapped zeta-flux candidate");

    if (ledger)
    {
        const long double parent_inventory = scalar_inventory(parent.scalar,
                                                              geometry.parent.metrics,
                                                              r.child_origin_parent_cell_i,
                                                              r.child_origin_parent_cell_j,
                                                              r.child_origin_parent_cell_i,
                                                              r.child_origin_parent_cell_j,
                                                              r.child_parent_span_x,
                                                              r.child_parent_span_y);
        const long double child_inventory  = scalar_inventory(
            child.scalar, geometry.child_storage.metrics, 0, 0, 1, 1, r.child_cells_x, r.child_cells_y);
        ledger->parent_scalar_inventory = finite_double(parent_inventory, "parent scalar inventory overflow");
        ledger->child_scalar_inventory  = finite_double(child_inventory, "child scalar inventory overflow");
        ledger->scalar_inventory_residual =
            finite_double(child_inventory - parent_inventory, "scalar inventory residual overflow");
        ledger->normalized_scalar_inventory_residual =
            finite_double(std::abs(child_inventory - parent_inventory) /
                              std::max(1.0L, std::max(std::abs(child_inventory), std::abs(parent_inventory))),
                          "normalized scalar inventory residual overflow");
        const auto x_residual           = maximum_x_residual(geometry, parent, child);
        const auto y_residual           = maximum_y_residual(geometry, parent, child);
        const auto zeta_residual        = maximum_zeta_residual(geometry, parent, child);
        ledger->maximum_x_flux_residual = finite_double(x_residual.absolute, "mapped x-flux residual overflow");
        ledger->maximum_normalized_x_flux_residual =
            finite_double(x_residual.normalized, "normalized mapped x-flux residual overflow");
        ledger->maximum_y_flux_residual = finite_double(y_residual.absolute, "mapped y-flux residual overflow");
        ledger->maximum_normalized_y_flux_residual =
            finite_double(y_residual.normalized, "normalized mapped y-flux residual overflow");
        ledger->maximum_zeta_flux_residual =
            finite_double(zeta_residual.absolute, "mapped zeta-flux residual overflow");
        ledger->maximum_normalized_zeta_flux_residual =
            finite_double(zeta_residual.normalized, "normalized mapped zeta-flux residual overflow");
    }
    return child;
}

NestedMappedConservativeStateV1 restrict_nested_mapped_conservative_state(const NestedMappedTerrainGeometryV1& geometry,
                                                                          const NestedMappedConservativeStateV1& child)
{
    require_relationship(geometry.relationship);
    terrain_transport::validate_geometry(geometry.parent.metrics);
    terrain_transport::validate_geometry(geometry.child_storage.metrics);
    require_child_state(geometry, child);
    const auto& r = geometry.relationship;

    NestedMappedConservativeStateV1 parent;
    parent.scalar.resize(r.child_parent_span_x, r.child_parent_span_y, r.levels_z, 0.0);
    parent.x_flux.resize(r.child_parent_span_x + 1, r.child_parent_span_y, r.levels_z, 0.0);
    parent.y_flux.resize(r.child_parent_span_x, r.child_parent_span_y + 1, r.levels_z, 0.0);
    parent.zeta_flux.resize(r.child_parent_span_x, r.child_parent_span_y, r.levels_z + 1, 0.0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                long double inventory = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                    {
                        const int ci = pi * r.parent_ratio_x + fi;
                        const int cj = pj * r.parent_ratio_y + fj;
                        inventory += static_cast<long double>(child.scalar(ci, cj, k)) *
                                     cell_measure(geometry.child_storage.metrics, ci + 1, cj + 1, k);
                    }
                const int parent_i = r.child_origin_parent_cell_i + pi;
                const int parent_j = r.child_origin_parent_cell_j + pj;
                parent.scalar(pi, pj, k) =
                    static_cast<double>(inventory / cell_measure(geometry.parent.metrics, parent_i, parent_j, k));
            }

#pragma omp parallel for collapse(3) schedule(static)
    for (int pi = 0; pi <= r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int   ci       = pi * r.parent_ratio_x;
                long double integral = 0.0L;
                for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                {
                    const int cj = pj * r.parent_ratio_y + fj;
                    integral += static_cast<long double>(child.x_flux(ci, cj, k)) *
                                x_face_measure(geometry.child_storage.metrics, ci + 1, cj + 1, k);
                }
                const int parent_i = r.child_origin_parent_cell_i + pi;
                const int parent_j = r.child_origin_parent_cell_j + pj;
                parent.x_flux(pi, pj, k) =
                    static_cast<double>(integral / x_face_measure(geometry.parent.metrics, parent_i, parent_j, k));
            }

#pragma omp parallel for collapse(3) schedule(static)
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj <= r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int   cj       = pj * r.parent_ratio_y;
                long double integral = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                {
                    const int ci = pi * r.parent_ratio_x + fi;
                    integral += static_cast<long double>(child.y_flux(ci, cj, k)) *
                                y_face_measure(geometry.child_storage.metrics, ci + 1, cj + 1, k);
                }
                const int parent_i = r.child_origin_parent_cell_i + pi;
                const int parent_j = r.child_origin_parent_cell_j + pj;
                parent.y_flux(pi, pj, k) =
                    static_cast<double>(integral / y_face_measure(geometry.parent.metrics, parent_i, parent_j, k));
            }

#pragma omp parallel for collapse(3) schedule(static)
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k <= r.levels_z; ++k)
            {
                long double integral = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                        integral += static_cast<long double>(
                                        child.zeta_flux(pi * r.parent_ratio_x + fi, pj * r.parent_ratio_y + fj, k)) *
                                    static_cast<long double>(r.child_dx_m) * static_cast<long double>(r.child_dy_m);
                parent.zeta_flux(pi, pj, k) = static_cast<double>(
                    integral / (static_cast<long double>(r.parent_dx_m) * static_cast<long double>(r.parent_dy_m)));
            }

    require_shape(parent.scalar,
                  r.child_parent_span_x,
                  r.child_parent_span_y,
                  r.levels_z,
                  "restricted parent mapped scalar candidate");
    require_shape(parent.x_flux,
                  r.child_parent_span_x + 1,
                  r.child_parent_span_y,
                  r.levels_z,
                  "restricted parent mapped x-flux candidate");
    require_shape(parent.y_flux,
                  r.child_parent_span_x,
                  r.child_parent_span_y + 1,
                  r.levels_z,
                  "restricted parent mapped y-flux candidate");
    require_shape(parent.zeta_flux,
                  r.child_parent_span_x,
                  r.child_parent_span_y,
                  r.levels_z + 1,
                  "restricted parent mapped zeta-flux candidate");
    return parent;
}

TerrainField3D restrict_nested_mapped_scalar(const NestedMappedTerrainGeometryV1& geometry,
                                             const TerrainField3D&                child_scalar)
{
    require_relationship(geometry.relationship);
    const auto& r = geometry.relationship;
    require_shape(child_scalar, r.child_cells_x, r.child_cells_y, r.levels_z, "child mapped scalar");
    TerrainField3D parent_span;
    parent_span.resize(r.child_parent_span_x, r.child_parent_span_y, r.levels_z, 0.0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
            {
                long double inventory = 0.0L;
                for (int fi = 0; fi < r.parent_ratio_x; ++fi)
                    for (int fj = 0; fj < r.parent_ratio_y; ++fj)
                    {
                        const int ci = pi * r.parent_ratio_x + fi;
                        const int cj = pj * r.parent_ratio_y + fj;
                        inventory += static_cast<long double>(child_scalar(ci, cj, k)) *
                                     cell_measure(geometry.child_storage.metrics, ci + 1, cj + 1, k);
                    }
                const int parent_i = r.child_origin_parent_cell_i + pi;
                const int parent_j = r.child_origin_parent_cell_j + pj;
                parent_span(pi, pj, k) =
                    static_cast<double>(inventory / cell_measure(geometry.parent.metrics, parent_i, parent_j, k));
            }
    require_shape(parent_span,
                  r.child_parent_span_x,
                  r.child_parent_span_y,
                  r.levels_z,
                  "restricted parent mapped scalar candidate");
    return parent_span;
}

NestedTerrainTangentBoundaryV1 build_nested_terrain_tangent_boundary(const NestedMappedTerrainGeometryV1& geometry,
                                                                     double                               ground_u_ms,
                                                                     double                               ground_v_ms,
                                                                     NestedTerrainTransferLedgerV1*       ledger)
{
    require_relationship(geometry.relationship);
    if (!std::isfinite(ground_u_ms) || !std::isfinite(ground_v_ms))
        reject("terrain boundary wind must be finite");
    const auto&                    r       = geometry.relationship;
    const auto&                    metrics = geometry.child_storage.metrics;
    NestedTerrainTangentBoundaryV1 result;
    result.relaxation_weight.resize(r.child_cells_x, r.child_cells_y, r.levels_z, 0.0);
    result.ground_u_ms.resize(r.child_cells_x, r.child_cells_y, 1, ground_u_ms);
    result.ground_v_ms.resize(r.child_cells_x, r.child_cells_y, 1, ground_v_ms);
    result.ground_w_ms.resize(r.child_cells_x, r.child_cells_y, 1, 0.0);

    const int width = r.relaxation_width_child_cells;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
        {
            const int distance = std::min(std::min(i, r.child_cells_x - 1 - i), std::min(j, r.child_cells_y - 1 - j));
            const double weight =
                distance < width ? static_cast<double>(width - distance) / static_cast<double>(width) : 0.0;
            for (int k = 0; k < r.levels_z; ++k)
                result.relaxation_weight(i, j, k) = weight;
            result.ground_w_ms(i, j, 0) =
                -metrics.Gx_face(i + 1, j + 1, 0) * ground_u_ms - metrics.Gy_face(i + 1, j + 1, 0) * ground_v_ms;
        }

    if (ledger)
    {
        long double   maximum = 0.0L;
        std::uint64_t owned   = 0;
        for (int i = 0; i < r.child_cells_x; ++i)
            for (int j = 0; j < r.child_cells_y; ++j)
            {
                const long double residual =
                    static_cast<long double>(result.ground_w_ms(i, j, 0)) +
                    static_cast<long double>(metrics.Gx_face(i + 1, j + 1, 0)) * static_cast<long double>(ground_u_ms) +
                    static_cast<long double>(metrics.Gy_face(i + 1, j + 1, 0)) * static_cast<long double>(ground_v_ms);
                maximum = std::max(maximum, std::abs(residual));
                for (int k = 0; k < r.levels_z; ++k)
                    if (result.relaxation_weight(i, j, k) > 0.0)
                    {
                        if (owned == std::numeric_limits<std::uint64_t>::max())
                            throw std::overflow_error("terrain relaxation ownership overflow");
                        ++owned;
                    }
            }
        ledger->maximum_ground_tangency_residual_ms =
            finite_double(maximum, "terrain ground tangency residual overflow");
        ledger->relaxation_owned_values = owned;
    }
    return result;
}

PreparedNestedActiveTerrainCommitV1
prepare_nested_active_terrain_commit(const NestedActiveTerrainStateV1&      live,
                                     const TerrainConfig&                   config,
                                     const OneWayNestRelationshipV1&        relationship,
                                     HorizontalPoint                        parent_origin_earth_m,
                                     const NestedMappedConservativeStateV1& parent_state,
                                     double                                 parent_time_s,
                                     double                                 ground_u_ms,
                                     double                                 ground_v_ms,
                                     NestedActiveTerrainCandidateObserver   observer,
                                     void*                                  observer_context)
{
    if (!std::isfinite(parent_time_s) || parent_time_s < 0.0)
        reject("parent publication time must be finite and non-negative");
    if (live.generation == std::numeric_limits<std::uint64_t>::max() ||
        live.publication_count == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("active-terrain publication counter overflow");

    PreparedNestedActiveTerrainCommitV1 prepared;
    prepared.expected_live              = live;
    const std::uint64_t next_generation = live.generation + 1;
    prepared.candidate.geometry         = build_stationary_nested_terrain_geometry(
        config, relationship, parent_origin_earth_m, next_generation, live.geometry.sampling_version);
    prepared.candidate.ledger.parent_time_s = parent_time_s;
    prepared.candidate.ledger.child_time_s  = parent_time_s;
    prepared.candidate.child_state =
        prolong_nested_mapped_conservative_state(prepared.candidate.geometry, parent_state, &prepared.candidate.ledger);
    prepared.candidate.boundary = build_nested_terrain_tangent_boundary(
        prepared.candidate.geometry, ground_u_ms, ground_v_ms, &prepared.candidate.ledger);
    prepared.candidate.parent_time_s     = parent_time_s;
    prepared.candidate.child_time_s      = parent_time_s;
    prepared.candidate.generation        = next_generation;
    prepared.candidate.publication_count = live.publication_count + 1;
    prepared.candidate.ready             = true;
    if (observer && !observer(prepared.candidate, observer_context))
        throw std::runtime_error("active-terrain candidate observer rejected publication");
    return prepared;
}

bool can_commit_nested_active_terrain(const PreparedNestedActiveTerrainCommitV1& prepared,
                                      const NestedActiveTerrainStateV1&          live) noexcept
{
    return !prepared.committed && prepared.candidate.ready && same_state(live, prepared.expected_live);
}

PreparedNestedActiveTerrainCommitV1 prepare_nested_terrain_parent_rebase(const NestedActiveTerrainStateV1& live,
                                                                         const OneWayNestRelationshipV1&   relationship,
                                                                         HorizontalPoint parent_origin_earth_m)
{
    if (!live.ready || live.geometry.config.horizontal_placement_id != "earth_fixed" || live.generation == 0 ||
        live.generation != live.geometry.generation || live.generation != live.publication_count ||
        live.parent_time_s != live.child_time_s)
    {
        reject("parent rebase requires synchronized Earth-fixed nested terrain");
    }
    auto prior_relationship                       = relationship;
    prior_relationship.child_origin_parent_cell_i = live.geometry.relationship.child_origin_parent_cell_i;
    prior_relationship.child_origin_parent_cell_j = live.geometry.relationship.child_origin_parent_cell_j;
    if (!same_relationship(prior_relationship, live.geometry.relationship))
    {
        reject("parent rebase cannot change the nested mesh or integration contract");
    }
    if (live.generation == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error("nested terrain rebase generation overflow");
    }
    auto geometry = build_stationary_nested_terrain_geometry(
        live.geometry.config, relationship, parent_origin_earth_m, live.generation + 1, live.geometry.sampling_version);
    const auto& old_child = live.geometry.child_storage;
    const auto& new_child = geometry.child_storage;
    if (geometry.child_origin_earth_m.x_m != live.geometry.child_origin_earth_m.x_m ||
        geometry.child_origin_earth_m.y_m != live.geometry.child_origin_earth_m.y_m ||
        !same_topography(old_child.topography, new_child.topography) ||
        !same_metrics(old_child.metrics, new_child.metrics) ||
        !same_diagnostics(old_child.diagnostics, new_child.diagnostics))
    {
        reject("parent rebase must preserve the child Earth-frame geometry exactly");
    }
    PreparedNestedActiveTerrainCommitV1 prepared;
    prepared.expected_live      = live;
    prepared.candidate          = live;
    prepared.candidate.geometry = std::move(geometry);
    ++prepared.candidate.generation;
    ++prepared.candidate.publication_count;
    return prepared;
}

bool commit_nested_active_terrain(PreparedNestedActiveTerrainCommitV1& prepared,
                                  NestedActiveTerrainStateV1&          live) noexcept
{
    if (!can_commit_nested_active_terrain(prepared, live))
        return false;
    using std::swap;
    swap(live, prepared.candidate);
    prepared.committed = true;
    return true;
}

} // namespace tmv::domain
