#pragma once

#include "core/infra/domain_motion.hpp"
#include "core/infra/domain_recycling.hpp"
#include "core/output/output_config.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace nimbus::output
{

/** One non-overlapping scalar-cell rectangle in the retiring old window. */
struct RetiringSwathTile
{
    std::size_t ordinal = 0;
    int i_begin = 0;
    int i_end = 0;
    int j_begin = 0;
    int j_end = 0;

    [[nodiscard]] int nx() const noexcept { return i_end - i_begin; }
    [[nodiscard]] int ny() const noexcept { return j_end - j_begin; }
    [[nodiscard]] std::size_t columns() const noexcept
    {
        return static_cast<std::size_t>(nx())
             * static_cast<std::size_t>(ny());
    }
};

/**
 * Canonical west/east/south/north decomposition of `physical - retained`.
 * Empty bands are omitted. Their union is exact and duplicate-free.
 */
std::vector<RetiringSwathTile> plan_retiring_scalar_swath_tiles(
    const tmv::domain::WholeCellShiftPlan& plan);

/** A pre-captured scalar-centered history field in canonical recycling packing. */
struct RetiringSwathField
{
    std::string field_id;
    const std::vector<float>* packed_values = nullptr;
    bool is_3d = true;
};

struct RetiringSwathPublication
{
    std::filesystem::path swath_root;
    std::string transaction_id;
    std::string publication_generation;
    std::uint64_t transaction_sequence = 0;
    std::uint64_t previous_committed_chain_fnv1a64 = 0;
    double simulation_time_s = 0.0;
    double dx_m = 0.0;
    double dy_m = 0.0;
    double dz_m = 0.0;
    std::string domain_motion_mode = "stationary";
    std::string native_staggering = "c_grid";
    tmv::domain::HorizontalDomainFrame pre_shift_frame;
    tmv::domain::WholeCellShiftPlan plan;
    OutputConfig output;
    std::vector<RetiringSwathField> fields;
};

struct RetiringSwathPublicationResult
{
    std::filesystem::path event_directory;
    std::size_t tiles_published = 0;
    std::size_t fields_published = 0;
    std::size_t values_published = 0;
    std::size_t bytes_published = 0;
    std::uint64_t event_manifest_bytes = 0;
    std::uint64_t event_manifest_fnv1a64 = 0;
};

/**
 * Durably capture one recycling retirement event before atmospheric overwrite.
 *
 * The final event directory appears through one same-filesystem rename only
 * after every tile payload and manifest is on stable storage. It is marked
 * `captured_precommit`; checkpoint transaction provenance later decides
 * whether the captured event belongs to the committed global swath.
 */
bool publish_retiring_swath_precommit(
    const RetiringSwathPublication& publication,
    RetiringSwathPublicationResult& result,
    std::string& error);

const char* retiring_swath_algorithm_id() noexcept;

} // namespace nimbus::output
