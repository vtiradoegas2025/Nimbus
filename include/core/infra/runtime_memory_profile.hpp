#pragma once

#include "boundary_layer/scheme_descriptor.hpp"
#include "microphysics/scheme_descriptor.hpp"
#include "numerics/advection/scheme_descriptor.hpp"
#include "radiation/scheme_descriptor.hpp"
#include "radiation/memory_policy.hpp"
#include "turbulence/scheme_descriptor.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

struct RuntimeMemoryPhaseInputs
{
    int solver_resident_fields = 0;
    int lazy_diagnostic_fields = 0;

    bool output_enabled = false;
    bool volume_output  = false;
    bool c_grid         = false;
    bool async_io       = false;
    bool borrow_core_output_fields = false;
    // Terrain history owns float64 scalar-center and interface-height copies.
    bool terrain_geometry = false;

    // Number of u/v/w components that the selected output products actually
    // require at scalar centers, resolved from the output dependency registry.
    int centered_velocity_fields = 3;

    int core_volume_output_fields    = 0;
    int derived_volume_output_fields = 0;
    int map_output_fields            = 0;
    int vertical_levels              = 1;
    int horizontal_rows              = 1;

    // Product-selected NR*NZ scratch arrays. The caller replaces this default
    // with the exact active derived-field dependency closure.
    int slice_scratch_arrays = 39;

    // ZFP delta mode retains one prior full 3-D field per output (two for
    // predictive mode). Its per-field codec/update high water also includes
    // the reusable residual buffer plus simultaneous compressed/decode or
    // decoded/reconstructed buffers (three full-field equivalents total).
    int zfp_history_depth = 0;
};

struct RuntimeMemoryPhaseProfile
{
    int solver_resident_fields = 0;
    int lazy_diagnostic_fields = 0;
    int resident_fields        = 0;

    int centered_velocity_fields              = 0;
    int slice_scratch_field_equivalents       = 0;
    int current_output_field_equivalents      = 0;
    int concurrent_snapshot_field_equivalents = 0;
    int compression_history_field_equivalents = 0;

    int output_working_field_equivalents = 0;
    int high_water_field_equivalents     = 0;
};

struct RuntimeSolverMemoryInputs
{
    std::string_view microphysics   = "kessler";
    std::string_view turbulence     = "smagorinsky";
    std::string_view advection      = "weno5";
    std::string_view boundary_layer = "none";
    std::string_view radiation      = "none";
};

inline std::uint64_t terrain_geometry_memory_bytes(std::uint64_t nx, std::uint64_t ny, std::uint64_t nz)
{
    using namespace tmv::radiative_transfer;
    const auto columns = memory_product(nx, ny), cells = memory_product(columns, nz);
    const auto zfaces = memory_product(columns, memory_sum(nz, 1));
    const auto xfaces = memory_product(memory_product(memory_sum(nx, 1), ny), nz);
    const auto yfaces = memory_product(memory_product(nx, memory_sum(ny, 1)), nz);
    return memory_product(sizeof(double), memory_sum(memory_product(5, cells),
        memory_sum(memory_product(3, zfaces), memory_sum(xfaces, memory_sum(yfaces, memory_product(3, columns))))));
}

inline std::uint64_t earth_fixed_remap_scratch_bytes(std::uint64_t nx, std::uint64_t ny, std::uint64_t nz)
{
    using namespace tmv::radiative_transfer;
    const auto columns = memory_product(nx, ny), cells = memory_product(columns, nz);
    const auto zfaces = memory_product(columns, memory_sum(nz, 1));
    const auto xfaces = memory_product(memory_product(memory_sum(nx, 1), ny), nz);
    const auto yfaces = memory_product(memory_product(nx, memory_sum(ny, 1)), nz);
    const auto face_fluxes = memory_product(2, memory_sum(zfaces, memory_sum(xfaces, yfaces)));
    const auto energy_and_columns = memory_sum(memory_product(2, cells),
        memory_sum(memory_product(5, zfaces), memory_product(8, columns)));
    return memory_sum(memory_product(4, terrain_geometry_memory_bytes(nx, ny, nz)),
        memory_sum(memory_product(sizeof(double), memory_sum(face_fluxes, energy_and_columns)),
                   memory_product(8 * sizeof(float), cells)));
}

inline constexpr int kCheckpointCoreDurableFieldEquivalents = 33;

// Checkpoint publication encodes each section directly into checksum and file
// sinks without retaining the complete section or envelope. A 5/2 reserve over
// the durable full-volume payload remains deliberately conservative for
// scheme-local portable-state serializers, nested terrain payloads, and
// allocator high water. Arithmetic saturates so preflight fails closed for
// unrepresentable requests.
inline std::uint64_t estimate_checkpoint_encoding_transient_bytes(
    std::uint64_t root_cells,
    std::uint64_t child_cells,
    int durable_field_equivalents_per_level) noexcept
{
    if (durable_field_equivalents_per_level <= 0) return 0;
    constexpr std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    if (child_cells > maximum - root_cells) return maximum;
    const std::uint64_t total_cells = root_cells + child_cells;
    const std::uint64_t fields = static_cast<std::uint64_t>(
        durable_field_equivalents_per_level);
    if (total_cells > maximum / sizeof(float) ||
        total_cells * sizeof(float) > maximum / fields)
        return maximum;
    const std::uint64_t payload = total_cells * sizeof(float) * fields;
    if (payload > maximum / 2U) return maximum;
    const std::uint64_t doubled = payload * 2U;
    const std::uint64_t half = payload / 2U + payload % 2U;
    return doubled > maximum - half ? maximum : doubled + half;
}

// The 75-field Kessler/Smagorinsky/WENO basis includes a ten-field process and
// allocator reserve. On audited 128x128x64 arm64 runs, the original slab PBL
// raised Kessler high water by 23.25 field equivalents. Conserved sample/hold
// coupling adds four persistent source volumes plus four simultaneous source
// candidates during transactional refresh, so the descriptor adds eight to
// the former rounded delta: 31 field equivalents.
inline int estimate_runtime_solver_fields(const RuntimeSolverMemoryInputs& inputs)
{
    int fields = 75;
    const auto add_calibrated_delta = [&fields](const auto& descriptors,
                                                std::string_view scheme) {
        const auto* descriptor = tmv::find_scheme_descriptor(
            descriptors, std::string(scheme));
        if (descriptor && descriptor->capabilities.memory_delta_calibrated)
            fields += descriptor->capabilities.solver_high_water_delta_fields;
    };

    add_calibrated_delta(tmv::get_microphysics_scheme_descriptors(), inputs.microphysics);
    add_calibrated_delta(tmv::get_turbulence_scheme_descriptors(), inputs.turbulence);
    add_calibrated_delta(tmv::get_advection_scheme_descriptors(), inputs.advection);
    add_calibrated_delta(tmv::get_boundary_layer_scheme_descriptors(), inputs.boundary_layer);
    add_calibrated_delta(tmv::get_radiation_scheme_descriptors(), inputs.radiation);
    return std::max(0, fields);
}

inline int ceil_div_nonnegative(int numerator, int denominator) noexcept
{
    if (numerator <= 0)
    {
        return 0;
    }
    return (numerator + std::max(1, denominator) - 1) / std::max(1, denominator);
}

// Counts NR*NZ slices relative to horizontal rows and NR*NTH maps relative to
// vertical levels. Rounding upward keeps the high-water estimate conservative.
inline RuntimeMemoryPhaseProfile estimate_runtime_memory_phases(const RuntimeMemoryPhaseInputs& inputs) noexcept
{
    RuntimeMemoryPhaseProfile profile;
    profile.solver_resident_fields = std::max(0, inputs.solver_resident_fields);
    profile.lazy_diagnostic_fields = std::max(0, inputs.lazy_diagnostic_fields);
    profile.resident_fields        = profile.solver_resident_fields + profile.lazy_diagnostic_fields;

    if (!inputs.output_enabled)
    {
        profile.high_water_field_equivalents = profile.resident_fields;
        return profile;
    }

    profile.centered_velocity_fields = inputs.c_grid ? std::clamp(inputs.centered_velocity_fields, 0, 3) : 0;
    profile.slice_scratch_field_equivalents =
        ceil_div_nonnegative(std::max(0, inputs.slice_scratch_arrays), inputs.horizontal_rows);

    const int core_fields     = std::max(0, inputs.core_volume_output_fields);
    const int derived_fields  = std::max(0, inputs.derived_volume_output_fields);
    const int map_fields      = std::max(0, inputs.map_output_fields);
    const int map_equivalents = ceil_div_nonnegative(map_fields, inputs.vertical_levels);

    if (inputs.volume_output && inputs.async_io)
    {
        const int terrain_geometry_fields = inputs.terrain_geometry
            ? 2 + ceil_div_nonnegative(2 * (std::max(1, inputs.vertical_levels) + 1),
                                       std::max(1, inputs.vertical_levels))
            : 0;
        const int snapshot_fields =
            (inputs.borrow_core_output_fields ? 0 : core_fields) +
            derived_fields + map_equivalents + terrain_geometry_fields;
        profile.current_output_field_equivalents = snapshot_fields;
        // The producer waits for the writer before assembling another frame,
        // so the queued/writing snapshot and next assembly cannot coexist.
        profile.concurrent_snapshot_field_equivalents = 0;
    }
    else if (inputs.volume_output)
    {
        // Core fields write directly. Derived volumes remain fully assembled;
        // 2-D products are produced one at a time.
        profile.current_output_field_equivalents = derived_fields + (map_fields > 0 ? 1 : 0);
    }

    if (inputs.volume_output && inputs.async_io && inputs.zfp_history_depth > 0)
    {
        const int history_fields = core_fields + derived_fields;
        constexpr int zfp_codec_working_fields = 3;
        profile.compression_history_field_equivalents =
            std::max(0, inputs.zfp_history_depth) * history_fields +
            (history_fields > 0 ? zfp_codec_working_fields : 0);
    }

    profile.output_working_field_equivalents =
        profile.centered_velocity_fields + profile.slice_scratch_field_equivalents +
        profile.current_output_field_equivalents + profile.concurrent_snapshot_field_equivalents +
        profile.compression_history_field_equivalents;
    profile.high_water_field_equivalents = profile.resident_fields + profile.output_working_field_equivalents;
    return profile;
}
