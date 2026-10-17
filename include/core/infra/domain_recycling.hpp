#pragma once

#include "core/field/field3d.hpp"
#include "core/infra/domain_motion.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

namespace tmv::domain
{

/**
 * Exact storage-space mapping for an integer Earth-window displacement.
 *
 * Positive i/j means that the computational window moves toward increasing
 * Earth x/y. A retained destination cell therefore receives
 * `old(i + shift_i, j + shift_j, k)`.
 */
struct WholeCellShiftPlan
{
    int nr = 0;
    int nth = 0;
    int nz = 0;
    int shift_i_cells = 0;
    int shift_j_cells = 0;

    int physical_i_begin = 0;
    int physical_i_end = 0;
    int physical_j_begin = 0;
    int physical_j_end = 0;

    // Retained destination rectangle inside the physical horizontal domain.
    int destination_i_begin = 0;
    int destination_i_end = 0;
    int destination_j_begin = 0;
    int destination_j_end = 0;

    // Matching retained rectangle in the old physical horizontal domain.
    int source_i_begin = 0;
    int source_i_end = 0;
    int source_j_begin = 0;
    int source_j_end = 0;

    // Retained values span the retained horizontal rectangle and every stored
    // vertical level. Leading values include every storage value outside that
    // rectangle, including closure storage which must be rebuilt. Trailing
    // values include only retiring physical horizontal columns, again at every
    // stored vertical level so raw owner capture remains restart-complete.
    std::size_t retained_cells = 0;
    std::size_t leading_cells = 0;
    std::size_t trailing_cells = 0;
};

WholeCellShiftPlan plan_whole_cell_shift(int nr, int nth, int nz, int shift_i_cells, int shift_j_cells);

/** Plan a shift when lateral storage includes nonphysical closure planes. */
WholeCellShiftPlan plan_whole_cell_shift_for_physical_domain(
    int nr, int nth, int nz, int shift_i_cells, int shift_j_cells,
    int physical_i_begin, int physical_i_end,
    int physical_j_begin, int physical_j_end);

enum class HorizontalFieldLocation
{
    ScalarCenter,
    XFace,
    YFace,
};

/** Derive exact stagger-aware ownership from a canonical scalar-domain plan. */
WholeCellShiftPlan plan_whole_cell_shift_for_horizontal_location(
    const WholeCellShiftPlan& scalar_plan,
    HorizontalFieldLocation location);

struct DomainFrameShiftReport
{
    int shift_i_cells = 0;
    int shift_j_cells = 0;
    HorizontalPoint displacement_m;
    HorizontalPoint previous_earth_origin_at_t0_m;
    HorizontalPoint committed_earth_origin_at_t0_m;
};

/**
 * Commit the exact Earth-coordinate consequence of a validated whole-cell
 * shift. Positive i/j moves the computational window toward increasing Earth
 * x/y, matching WholeCellShiftPlan. Every input and result is validated before
 * the frame is mutated; velocity remains unchanged because stored model winds
 * are Earth-relative.
 */
DomainFrameShiftReport commit_whole_cell_shift_to_domain_frame(
    HorizontalDomainFrame& frame,
    const WholeCellShiftPlan& plan,
    double dx_m,
    double dy_m);

const char* domain_frame_shift_algorithm_id() noexcept;

struct HomogeneousFieldShiftBinding
{
    std::string_view          field_name;
    Field3D*                  field = nullptr;
    const std::vector<float>* leading_profile = nullptr;
    // Collocated/scalar is the default; C-grid u/v owners must declare their
    // x/y-face locations so retained physics is not mistaken for closure.
    HorizontalFieldLocation  horizontal_location =
        HorizontalFieldLocation::ScalarCenter;
    // A terrain-aware owner supplies complete destination-indexed leading
    // values instead of one horizontally homogeneous vertical profile.
    const Field3D* leading_field = nullptr;
};

struct HomogeneousFieldShiftReport
{
    std::size_t fields_shifted = 0;
    std::size_t retained_values = 0;
    std::size_t initialized_values = 0;
    std::size_t retained_bytes = 0;
    std::size_t initialized_bytes = 0;
};

struct HomogeneousSurfaceShiftBinding
{
    std::string_view     field_name;
    std::vector<float>*  field = nullptr;
    float                leading_value = 0.0f;
    const std::vector<float>* leading_field = nullptr;
};

struct HomogeneousScalarCommitBinding
{
    std::string_view scalar_name;
    double* target = nullptr;
    double committed_value = 0.0;
};

// Type-erased trivially-copyable horizontal records. This owns persistent
// per-column scheme state (for example SlabColumnState) without coupling the
// domain transaction core to any physics module.
struct HomogeneousSurfaceRecordShiftBinding
{
    std::string_view record_name;
    void* records = nullptr;
    std::size_t record_count = 0;
    std::size_t record_size = 0;
    const void* leading_record = nullptr;
    const void* leading_records = nullptr;
};

struct HomogeneousBooleanCommitBinding
{
    std::string_view flag_name;
    bool* target = nullptr;
    bool committed_value = false;
};

// Type-erased, allocation-free runtime-cache invalidation performed only
// after every potentially throwing recycle preflight has succeeded. The callback
// contract is deliberately noexcept: it may clear or mark already-owned
// scratch, but it may not allocate, rebuild physics, or publish output.
using HomogeneousRuntimeInvalidationFunction = void (*)(void*) noexcept;

struct HomogeneousRuntimeInvalidationBinding
{
    std::string_view invalidation_name;
    void* runtime = nullptr;
    HomogeneousRuntimeInvalidationFunction invalidate = nullptr;
};

// Prepared persistent runtime transforms use a separate semantic type from
// replaceable-cache invalidation. The callback is still allocation-free and
// noexcept, but may commit already-prepared value state after common preflight.
using HomogeneousRuntimeCommitFunction = void (*)(void*) noexcept;

struct HomogeneousRuntimeCommitBinding
{
    std::string_view commit_name;
    void* prepared_state = nullptr;
    HomogeneousRuntimeCommitFunction commit = nullptr;
};

struct HomogeneousWindowShiftReport
{
    HomogeneousFieldShiftReport volumes;
    std::size_t surfaces_shifted = 0;
    std::size_t retained_surface_values = 0;
    std::size_t initialized_surface_values = 0;
    std::size_t retained_surface_bytes = 0;
    std::size_t initialized_surface_bytes = 0;
    std::size_t scalars_committed = 0;
    std::size_t record_surfaces_shifted = 0;
    std::size_t retained_record_values = 0;
    std::size_t initialized_record_values = 0;
    std::size_t retained_record_bytes = 0;
    std::size_t initialized_record_bytes = 0;
    std::size_t flags_committed = 0;
    std::size_t runtime_invalidations_committed = 0;
    std::size_t runtime_commits_committed = 0;
};

/** Validate every owner and compute the exact report without changing state. */
HomogeneousWindowShiftReport preflight_homogeneous_window_shift(
    const WholeCellShiftPlan& plan,
    const std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    const std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings = {},
    const std::vector<HomogeneousScalarCommitBinding>& scalar_bindings = {},
    const std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings = {},
    const std::vector<HomogeneousBooleanCommitBinding>& flag_bindings = {},
    const std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings = {},
    const std::vector<HomogeneousRuntimeCommitBinding>&
        runtime_commit_bindings = {});

/**
 * Commit only the already-preflighted non-volume owners after a device fence.
 * The caller must pass the unchanged bindings from a successful complete
 * preflight. This function is allocation-free and noexcept.
 */
void commit_preflighted_homogeneous_window_nonvolume(
    const WholeCellShiftPlan& plan,
    const std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    const std::vector<HomogeneousScalarCommitBinding>& scalar_bindings,
    const std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings,
    const std::vector<HomogeneousBooleanCommitBinding>& flag_bindings,
    const std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings,
    const std::vector<HomogeneousRuntimeCommitBinding>&
        runtime_commit_bindings) noexcept;

struct TrailingFieldExportBinding
{
    std::string_view    field_name;
    const Field3D*      source = nullptr;
    std::vector<float>* packed_values = nullptr;
    // Raw retirement follows the same owner-specific physical domain as shift.
    HorizontalFieldLocation horizontal_location =
        HorizontalFieldLocation::ScalarCenter;
};

struct TrailingFieldExportReport
{
    std::size_t fields_exported = 0;
    std::size_t values_exported = 0;
    std::size_t bytes_exported = 0;
};

struct TrailingSurfaceExportBinding
{
    std::string_view field_name;
    const std::vector<float>* source = nullptr;
    std::vector<float>* packed_values = nullptr;
};

/**
 * Atomically shift a set of equally shaped fields with homogeneous leading
 * profiles.
 *
 * Every pointer, shape, unique owner, profile length, and profile value is
 * validated before the first write. The commit phase allocates nothing and
 * contains only non-throwing in-place moves/fills, so a preflight failure
 * leaves every bound field byte-identical.
 */
HomogeneousFieldShiftReport execute_homogeneous_field_shift(
    const WholeCellShiftPlan&                         plan,
    const std::vector<HomogeneousFieldShiftBinding>& bindings);

/**
 * Atomically shift volume and horizontal surface owners, commit scalar/record
 * metadata, and run allocation-free runtime-cache invalidations in one
 * preflight/commit boundary. Surface fields use flattened `(i * nth + j)`
 * storage. At least one binding is required.
 */
HomogeneousWindowShiftReport execute_homogeneous_window_shift(
    const WholeCellShiftPlan&                           plan,
    const std::vector<HomogeneousFieldShiftBinding>&   volume_bindings,
    const std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    const std::vector<HomogeneousScalarCommitBinding>& scalar_bindings = {},
    const std::vector<HomogeneousSurfaceRecordShiftBinding>&
        record_bindings = {},
    const std::vector<HomogeneousBooleanCommitBinding>& flag_bindings = {},
    const std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings = {},
    const std::vector<HomogeneousRuntimeCommitBinding>&
        runtime_commit_bindings = {});

/**
 * Copy every retiring old-grid value into caller-preallocated buffers.
 *
 * Packing order is canonical `(i,j,k)` order over physical horizontal columns
 * outside the retained source rectangle and every stored vertical level.
 * Closure columns are never exported. Each destination must already have
 * exactly `plan.trailing_cells` values, so capture performs no allocation and
 * can run before the destructive shift transaction.
 */
TrailingFieldExportReport export_trailing_fields(
    const WholeCellShiftPlan&                       plan,
    const std::vector<TrailingFieldExportBinding>& bindings);

/** Export retiring physical horizontal cells from `(i * nth + j)` surfaces. */
TrailingFieldExportReport export_trailing_surfaces(
    const WholeCellShiftPlan& plan,
    const std::vector<TrailingSurfaceExportBinding>& bindings);

const char* homogeneous_field_shift_algorithm_id() noexcept;
const char* terrain_aware_field_shift_algorithm_id() noexcept;

} // namespace tmv::domain
