#pragma once

#include <cstdint>

struct SimulationStateView;
struct NestedGridStorage;

struct NestedGridConfig
{
    bool enabled = false;
    int center_i;
    int center_j;
    int center_k;
    int nest_size_r = 32;
    int nest_size_th = 32;
    int nest_size_z = 32;
    double refinement = 3.0;

    explicit NestedGridConfig(int nr = 200, int nth = 128, int nz = 150)
        : center_i(nr / 2), center_j(nth / 2), center_k(nz / 2)
    {
    }
};

namespace tmv::domain
{

// This geometry/time relationship stays free of field storage so the same
// contract can be used by runtime orchestration, checkpoints, and accelerator
// implementations without duplicating nest semantics.
enum class NestedHorizontalLocation
{
    ScalarCenter,
    XFace,
    YFace,
};

struct OneWayNestRelationshipInput
{
    int    parent_cells_x                  = 0;
    int    parent_cells_y                  = 0;
    int    parent_levels_z                 = 0;
    double parent_dx_m                     = 0.0;
    double parent_dy_m                     = 0.0;
    double parent_dt_s                     = 0.0;
    int    parent_ratio_x                  = 0;
    int    parent_ratio_y                  = 0;
    int    parent_time_step_ratio          = 0;
    int    child_origin_parent_cell_i      = 0;
    int    child_origin_parent_cell_j      = 0;
    int    child_cells_x                   = 0;
    int    child_cells_y                   = 0;
    int    child_levels_z                  = 0;
    int    relaxation_width_child_cells    = 0;
    int    boundary_stencil_parent_cells   = 0;
};

// Portable, checkpoint-ready v1 relationship.  All placement is expressed in
// integer parent cells; no floating coordinate is authoritative.
struct OneWayNestRelationshipV1
{
    std::uint32_t version = 1;
    int parent_cells_x = 0;
    int parent_cells_y = 0;
    int levels_z = 0;
    int parent_ratio_x = 0;
    int parent_ratio_y = 0;
    int parent_time_step_ratio = 0;
    int child_origin_parent_cell_i = 0;
    int child_origin_parent_cell_j = 0;
    int child_cells_x = 0;
    int child_cells_y = 0;
    int child_parent_span_x = 0;
    int child_parent_span_y = 0;
    int relaxation_width_child_cells = 0;
    int boundary_stencil_parent_cells = 0;
    double parent_dx_m = 0.0;
    double parent_dy_m = 0.0;
    double parent_dt_s = 0.0;
    double child_dx_m = 0.0;
    double child_dy_m = 0.0;
    double child_dt_s = 0.0;
};

struct ParentHorizontalCoordinate
{
    double i = 0.0;
    double j = 0.0;
};

struct ParentTemporalWeights
{
    int    completed_child_substeps = 0;
    double previous                 = 1.0;
    double current                  = 0.0;
    double offset_from_previous_s   = 0.0;
};

struct ChildCellShift
{
    int i = 0;
    int j = 0;
};

struct ParentTimeBracketV1
{
    std::uint32_t version = 1;
    bool   available = false;
    double previous_parent_time_s = 0.0;
    double current_parent_time_s = 0.0;
    int    completed_child_substeps = 0;
};

void validate_one_way_nest_relationship(const OneWayNestRelationshipV1& relationship);

OneWayNestRelationshipV1 prepare_one_way_nest_relationship(
    const OneWayNestRelationshipInput& input);

ParentHorizontalCoordinate parent_coordinate_for_child(
    const OneWayNestRelationshipV1& relationship,
    NestedHorizontalLocation location,
    int child_i,
    int child_j);

ParentTemporalWeights parent_temporal_weights(
    const OneWayNestRelationshipV1& relationship,
    int completed_child_substeps);

ParentTemporalWeights parent_temporal_weights(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket);

ChildCellShift child_shift_for_parent_move(
    const OneWayNestRelationshipV1& relationship,
    int parent_shift_i,
    int parent_shift_j);

ParentTimeBracketV1 prepare_parent_time_bracket(
    const OneWayNestRelationshipV1& relationship,
    double previous_parent_time_s,
    double current_parent_time_s);

void advance_child_substep(
    const OneWayNestRelationshipV1& relationship,
    ParentTimeBracketV1& bracket);

void roll_parent_time_bracket(
    const OneWayNestRelationshipV1& relationship,
    ParentTimeBracketV1& bracket,
    double next_parent_time_s);

double nested_child_time_s(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket);

} // namespace tmv::domain

// The legacy primitive-interpolation initializer remains unsupported. Runtime
// nesting uses the conservative one-way execution path instead.
void require_supported_nested_grid_runtime(bool enabled);

void initialize_nested_grid(NestedGridStorage& target,
                            const SimulationStateView& parent_state);
// Compatibility adapters for the process-owned nested storage.
void initialize_nested_grid(const SimulationStateView& parent_state);
void initialize_nested_grid();
