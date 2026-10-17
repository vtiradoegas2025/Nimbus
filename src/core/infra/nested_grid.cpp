// Legacy parent-snapshot interpolation retained only for compatibility tests.

#include "core/infra/nested_grid.hpp"
#include "core/runtime/simulation_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

double interpolate_parent_field(const Field3D& field,
                                int i0, int j0, int k0,
                                double fi, double fj, double fk)
{
    const double v000 = static_cast<double>(field[i0][j0][k0]);
    const double v001 = static_cast<double>(field[i0][j0][k0 + 1]);
    const double v010 = static_cast<double>(field[i0][j0 + 1][k0]);
    const double v011 = static_cast<double>(field[i0][j0 + 1][k0 + 1]);
    const double v100 = static_cast<double>(field[i0 + 1][j0][k0]);
    const double v101 = static_cast<double>(field[i0 + 1][j0][k0 + 1]);
    const double v110 = static_cast<double>(field[i0 + 1][j0 + 1][k0]);
    const double v111 = static_cast<double>(field[i0 + 1][j0 + 1][k0 + 1]);

    return v000 * (1 - fi) * (1 - fj) * (1 - fk) +
           v001 * (1 - fi) * (1 - fj) * fk +
           v010 * (1 - fi) * fj * (1 - fk) +
           v011 * (1 - fi) * fj * fk +
           v100 * fi * (1 - fj) * (1 - fk) +
           v101 * fi * (1 - fj) * fk +
           v110 * fi * fj * (1 - fk) +
           v111 * fi * fj * fk;
}

} // namespace

namespace tmv::domain
{

namespace
{

[[noreturn]] void reject_relationship(const char* reason)
{
    throw std::invalid_argument(std::string("invalid one-way nest relationship: ") + reason);
}

void require_relationship_v1(const OneWayNestRelationshipV1& relationship)
{
    if (relationship.version != 1)
        reject_relationship("unsupported relationship version");

    OneWayNestRelationshipInput input;
    input.parent_cells_x = relationship.parent_cells_x;
    input.parent_cells_y = relationship.parent_cells_y;
    input.parent_levels_z = relationship.levels_z;
    input.parent_dx_m = relationship.parent_dx_m;
    input.parent_dy_m = relationship.parent_dy_m;
    input.parent_dt_s = relationship.parent_dt_s;
    input.parent_ratio_x = relationship.parent_ratio_x;
    input.parent_ratio_y = relationship.parent_ratio_y;
    input.parent_time_step_ratio = relationship.parent_time_step_ratio;
    input.child_origin_parent_cell_i = relationship.child_origin_parent_cell_i;
    input.child_origin_parent_cell_j = relationship.child_origin_parent_cell_j;
    input.child_cells_x = relationship.child_cells_x;
    input.child_cells_y = relationship.child_cells_y;
    input.child_levels_z = relationship.levels_z;
    input.relaxation_width_child_cells = relationship.relaxation_width_child_cells;
    input.boundary_stencil_parent_cells = relationship.boundary_stencil_parent_cells;
    const auto expected = prepare_one_way_nest_relationship(input);
    if (relationship.child_parent_span_x != expected.child_parent_span_x ||
        relationship.child_parent_span_y != expected.child_parent_span_y ||
        relationship.child_dx_m != expected.child_dx_m ||
        relationship.child_dy_m != expected.child_dy_m ||
        relationship.child_dt_s != expected.child_dt_s)
    {
        reject_relationship("stored derived geometry or timestep is inconsistent");
    }
}

int checked_scaled_shift(int parent_shift, int ratio)
{
    const auto result = static_cast<long long>(parent_shift) * static_cast<long long>(ratio);
    if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
        reject_relationship("parent-to-child movement overflows integer cells");
    return static_cast<int>(result);
}

void require_valid_bracket(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket)
{
    require_relationship_v1(relationship);
    if (bracket.version != 1 || !bracket.available ||
        !std::isfinite(bracket.previous_parent_time_s) ||
        !std::isfinite(bracket.current_parent_time_s) ||
        bracket.current_parent_time_s <= bracket.previous_parent_time_s ||
        bracket.completed_child_substeps < 0 ||
        bracket.completed_child_substeps > relationship.parent_time_step_ratio)
    {
        reject_relationship("invalid parent time bracket");
    }
    const double span = bracket.current_parent_time_s -
                        bracket.previous_parent_time_s;
    const double nominal_endpoint = bracket.previous_parent_time_s + relationship.parent_dt_s;
    if (!std::isfinite(span) || span <= 0.0 ||
        !std::isfinite(nominal_endpoint) || bracket.current_parent_time_s > nominal_endpoint)
    {
        reject_relationship(
            "realized parent time bracket exceeds its finite nominal ceiling");
    }
}

} // namespace

void validate_one_way_nest_relationship(const OneWayNestRelationshipV1& relationship)
{
    const int maximum = std::numeric_limits<int>::max() - 2;
    if (relationship.parent_cells_x > maximum || relationship.parent_cells_y > maximum ||
        relationship.child_cells_x > maximum || relationship.child_cells_y > maximum)
        reject_relationship("horizontal storage dimensions overflow their closure cells");
    require_relationship_v1(relationship);
}

OneWayNestRelationshipV1 prepare_one_way_nest_relationship(
    const OneWayNestRelationshipInput& input)
{
    if (input.parent_cells_x <= 0 || input.parent_cells_y <= 0 || input.parent_levels_z <= 1)
        reject_relationship("parent dimensions must be positive with at least two vertical levels");
    if (!std::isfinite(input.parent_dx_m) || !std::isfinite(input.parent_dy_m) ||
        !std::isfinite(input.parent_dt_s) || input.parent_dx_m <= 0.0 ||
        input.parent_dy_m <= 0.0 || input.parent_dt_s <= 0.0)
    {
        reject_relationship("parent spacing and timestep must be finite and positive");
    }
    if (input.parent_ratio_x < 2 || input.parent_ratio_y < 2)
        reject_relationship("horizontal refinement ratios must be at least two");
    if (input.parent_time_step_ratio < std::max(input.parent_ratio_x, input.parent_ratio_y))
    {
        reject_relationship(
            "time-step ratio must be at least the largest horizontal refinement ratio");
    }
    if (input.child_cells_x <= 0 || input.child_cells_y <= 0 ||
        input.child_cells_x % input.parent_ratio_x != 0 ||
        input.child_cells_y % input.parent_ratio_y != 0)
    {
        reject_relationship("child dimensions must span whole parent cells");
    }
    if (input.child_levels_z != input.parent_levels_z)
        reject_relationship("this nesting contract requires shared parent/child vertical levels");
    if (input.relaxation_width_child_cells <= 0 ||
        2 * static_cast<long long>(input.relaxation_width_child_cells) >=
            std::min(input.child_cells_x, input.child_cells_y))
    {
        reject_relationship("relaxation zones must leave a non-empty verification interior");
    }
    if (input.boundary_stencil_parent_cells <= 0)
        reject_relationship("boundary stencil margin must be positive");

    const int span_x = input.child_cells_x / input.parent_ratio_x;
    const int span_y = input.child_cells_y / input.parent_ratio_y;
    const auto end_i = static_cast<long long>(input.child_origin_parent_cell_i) + span_x;
    const auto end_j = static_cast<long long>(input.child_origin_parent_cell_j) + span_y;
    if (input.child_origin_parent_cell_i < input.boundary_stencil_parent_cells ||
        input.child_origin_parent_cell_j < input.boundary_stencil_parent_cells ||
        end_i > static_cast<long long>(input.parent_cells_x - input.boundary_stencil_parent_cells) ||
        end_j > static_cast<long long>(input.parent_cells_y - input.boundary_stencil_parent_cells))
    {
        reject_relationship("child placement does not retain the required parent stencil margin");
    }

    OneWayNestRelationshipV1 result;
    result.parent_cells_x = input.parent_cells_x;
    result.parent_cells_y = input.parent_cells_y;
    result.levels_z = input.parent_levels_z;
    result.parent_ratio_x = input.parent_ratio_x;
    result.parent_ratio_y = input.parent_ratio_y;
    result.parent_time_step_ratio = input.parent_time_step_ratio;
    result.child_origin_parent_cell_i = input.child_origin_parent_cell_i;
    result.child_origin_parent_cell_j = input.child_origin_parent_cell_j;
    result.child_cells_x = input.child_cells_x;
    result.child_cells_y = input.child_cells_y;
    result.child_parent_span_x = span_x;
    result.child_parent_span_y = span_y;
    result.relaxation_width_child_cells = input.relaxation_width_child_cells;
    result.boundary_stencil_parent_cells = input.boundary_stencil_parent_cells;
    result.parent_dx_m = input.parent_dx_m;
    result.parent_dy_m = input.parent_dy_m;
    result.parent_dt_s = input.parent_dt_s;
    result.child_dx_m = input.parent_dx_m / input.parent_ratio_x;
    result.child_dy_m = input.parent_dy_m / input.parent_ratio_y;
    result.child_dt_s = input.parent_dt_s / input.parent_time_step_ratio;
    if (!std::isfinite(result.child_dx_m) || !std::isfinite(result.child_dy_m) ||
        !std::isfinite(result.child_dt_s) || result.child_dx_m <= 0.0 ||
        result.child_dy_m <= 0.0 || result.child_dt_s <= 0.0)
    {
        reject_relationship("derived child spacing or timestep is not representable");
    }
    return result;
}

ParentHorizontalCoordinate parent_coordinate_for_child(
    const OneWayNestRelationshipV1& relationship,
    NestedHorizontalLocation location,
    int child_i,
    int child_j)
{
    require_relationship_v1(relationship);
    if (location != NestedHorizontalLocation::ScalarCenter &&
        location != NestedHorizontalLocation::XFace &&
        location != NestedHorizontalLocation::YFace)
    {
        reject_relationship("unknown C-grid owner location");
    }
    const bool x_face = location == NestedHorizontalLocation::XFace;
    const bool y_face = location == NestedHorizontalLocation::YFace;
    const int max_i = relationship.child_cells_x + (x_face ? 1 : 0);
    const int max_j = relationship.child_cells_y + (y_face ? 1 : 0);
    if (child_i < 0 || child_i >= max_i || child_j < 0 || child_j >= max_j)
        reject_relationship("child coordinate is outside its physical C-grid owner");

    // Values are continuous indices in the corresponding parent owner.  A
    // scalar index denotes a cell centre; an x/y face index denotes that face.
    const double offset_i = x_face ? 0.0 : 0.5;
    const double offset_j = y_face ? 0.0 : 0.5;
    ParentHorizontalCoordinate result;
    result.i = relationship.child_origin_parent_cell_i +
               (static_cast<double>(child_i) + offset_i) / relationship.parent_ratio_x - offset_i;
    result.j = relationship.child_origin_parent_cell_j +
               (static_cast<double>(child_j) + offset_j) / relationship.parent_ratio_y - offset_j;
    return result;
}

ParentTemporalWeights parent_temporal_weights(
    const OneWayNestRelationshipV1& relationship,
    int completed_child_substeps)
{
    require_relationship_v1(relationship);
    if (completed_child_substeps < 0 ||
        completed_child_substeps > relationship.parent_time_step_ratio)
    {
        reject_relationship("child boundary time lies outside the completed parent bracket");
    }
    ParentTemporalWeights result;
    result.completed_child_substeps = completed_child_substeps;
    result.current = static_cast<double>(completed_child_substeps) /
                     static_cast<double>(relationship.parent_time_step_ratio);
    result.previous = 1.0 - result.current;
    result.offset_from_previous_s = completed_child_substeps * relationship.child_dt_s;
    return result;
}

ParentTemporalWeights parent_temporal_weights(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket)
{
    require_valid_bracket(relationship, bracket);
    ParentTemporalWeights result;
    result.completed_child_substeps = bracket.completed_child_substeps;
    result.current =
        static_cast<double>(bracket.completed_child_substeps) /
        static_cast<double>(relationship.parent_time_step_ratio);
    result.previous = 1.0 - result.current;
    result.offset_from_previous_s = result.current *
        (bracket.current_parent_time_s - bracket.previous_parent_time_s);
    return result;
}

ChildCellShift child_shift_for_parent_move(
    const OneWayNestRelationshipV1& relationship,
    int parent_shift_i,
    int parent_shift_j)
{
    require_relationship_v1(relationship);
    return {checked_scaled_shift(parent_shift_i, relationship.parent_ratio_x),
            checked_scaled_shift(parent_shift_j, relationship.parent_ratio_y)};
}

ParentTimeBracketV1 prepare_parent_time_bracket(
    const OneWayNestRelationshipV1& relationship,
    double previous_parent_time_s,
    double current_parent_time_s)
{
    ParentTimeBracketV1 result;
    result.available = true;
    result.previous_parent_time_s = previous_parent_time_s;
    result.current_parent_time_s = current_parent_time_s;
    require_valid_bracket(relationship, result);
    return result;
}

void advance_child_substep(
    const OneWayNestRelationshipV1& relationship,
    ParentTimeBracketV1& bracket)
{
    require_valid_bracket(relationship, bracket);
    if (bracket.completed_child_substeps == relationship.parent_time_step_ratio)
        reject_relationship("parent time bracket is already complete");
    ++bracket.completed_child_substeps;
}

void roll_parent_time_bracket(
    const OneWayNestRelationshipV1& relationship,
    ParentTimeBracketV1& bracket,
    double next_parent_time_s)
{
    require_valid_bracket(relationship, bracket);
    if (bracket.completed_child_substeps != relationship.parent_time_step_ratio)
        reject_relationship("cannot roll an incomplete parent time bracket");
    ParentTimeBracketV1 candidate = prepare_parent_time_bracket(
        relationship, bracket.current_parent_time_s, next_parent_time_s);
    bracket = candidate;
}

double nested_child_time_s(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket)
{
    require_valid_bracket(relationship, bracket);
    if (bracket.completed_child_substeps == relationship.parent_time_step_ratio)
        return bracket.current_parent_time_s;
    const double realized_child_dt_s =
        (bracket.current_parent_time_s - bracket.previous_parent_time_s) /
        relationship.parent_time_step_ratio;
    return bracket.previous_parent_time_s +
           bracket.completed_child_substeps * realized_child_dt_s;
}

} // namespace tmv::domain

void require_supported_nested_grid_runtime(bool enabled)
{
    if (enabled)
    {
        throw std::runtime_error(
            "nested.enabled=true is not supported: Nimbus currently only "
            "interpolates an initialization snapshot and has no nested "
            "time integration, boundary forcing, subcycling, or parent "
            "feedback. Disable nesting until the evolving-nest contract "
            "is implemented and qualified.");
    }
}

void initialize_nested_grid(NestedGridStorage& target,
                            const SimulationStateView& parent_state)
{
    auto& nested_config = target.config;
    if (!nested_config.enabled) return;
    const auto& grid = parent_state.grid;
    const auto& fields = parent_state.fields;
    if (grid.nr < 2 || grid.nth < 2 || grid.nz < 2)
    {
        return;
    }

    int nr_nest = nested_config.nest_size_r;
    int nth_nest = nested_config.nest_size_th;
    int nz_nest = nested_config.nest_size_z;

    auto& nest_rho = target.rho;
    auto& nest_p = target.p;
    auto& nest_u = target.u;
    auto& nest_w = target.w;
    auto& nest_v = target.v;
    auto& nest_theta = target.theta;
    auto& nest_qv = target.qv;
    auto& nest_qc = target.qc;
    auto& nest_qr = target.qr;
    auto& nest_qi = target.qi;
    auto& nest_qs = target.qs;
    auto& nest_qh = target.qh;
    auto& nest_qg = target.qg;
    auto& nest_tracer = target.tracer;

    nest_rho.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_p.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_u.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_w.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_v.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_theta.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qv.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qc.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qr.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qi.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qs.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qh.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qg.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_tracer.resize(nr_nest, nth_nest, nz_nest, 0.0f);

    double ref = nested_config.refinement;
    int ci = nested_config.center_i;
    int cj = nested_config.center_j;
    int ck = nested_config.center_k;

    #pragma omp parallel for collapse(2)
    for (int i_nest = 0; i_nest < nr_nest; ++i_nest)
    {
        for (int j_nest = 0; j_nest < nth_nest; ++j_nest)
        {
            for (int k_nest = 0; k_nest < nz_nest; ++k_nest)
            {
                double i_parent = ci + (i_nest - nr_nest/2.0) / ref;
                double j_parent = cj + (j_nest - nth_nest/2.0) / ref;
                double k_parent = ck + (k_nest - nz_nest/2.0) / ref;

                int i0 = std::max(0, std::min(grid.nr - 2, (int)std::floor(i_parent)));
                int j0 = std::max(0, std::min(grid.nth - 2, (int)std::floor(j_parent)));
                int k0 = std::max(0, std::min(grid.nz - 2, (int)std::floor(k_parent)));

                double fi = i_parent - i0;
                double fj = j_parent - j0;
                double fk = k_parent - k0;

                nest_rho[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.rho, i0, j0, k0, fi, fj, fk);
                nest_p[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.p, i0, j0, k0, fi, fj, fk);
                nest_u[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.u, i0, j0, k0, fi, fj, fk);
                nest_w[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.w, i0, j0, k0, fi, fj, fk);
                nest_v[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.v, i0, j0, k0, fi, fj, fk);
                nest_theta[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.theta, i0, j0, k0, fi, fj, fk);
                nest_qv[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.qv, i0, j0, k0, fi, fj, fk);
                nest_qc[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.qc, i0, j0, k0, fi, fj, fk);
                nest_qr[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.qr, i0, j0, k0, fi, fj, fk);
                nest_qi[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.qi, i0, j0, k0, fi, fj, fk);
                nest_qs[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.qs, i0, j0, k0, fi, fj, fk);
                nest_qh[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.qh, i0, j0, k0, fi, fj, fk);
                nest_qg[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.qg, i0, j0, k0, fi, fj, fk);
                nest_tracer[i_nest][j_nest][k_nest] =
                    interpolate_parent_field(fields.tracer, i0, j0, k0, fi, fj, fk);
            }
        }
    }
}

void initialize_nested_grid(const SimulationStateView& parent_state)
{
    initialize_nested_grid(global_simulation_storage.nested, parent_state);
}

void initialize_nested_grid()
{
    initialize_nested_grid(global_simulation_storage.nested,
                           global_simulation_state());
}
