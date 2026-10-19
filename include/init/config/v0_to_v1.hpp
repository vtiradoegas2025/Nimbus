#pragma once

#include "init/config/config_v1.hpp"
#include "init/config/v1_loader.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @file v0_to_v1.hpp
 * @brief Auto-migrator from legacy v0 YAML configs to the v1 schema.
 *
 * Phase A3 deliverable (docs/CM1Mapping.md §21). Reads any current
 * `tests/fixtures/configs/test/<name>.yaml` (v0 format — no `schema:` stamp, flat-ish
 * section names), maps each key to its v1 destination, and produces a
 * `ConfigV1` struct that loads cleanly through `load_v1_config`.
 *
 * Migration policy (locked decision §23.4): **best-guess + warn** for
 * ambiguous mappings. Each translation emits a `MigrationWarning` so the
 * `--migrate-config` CLI can present a diff to the user before they commit
 * to the new YAML.
 *
 * The migrator does NOT silently drop content: every v0 key is either
 * mapped, dropped with an explicit warning, or recorded as an error if it
 * cannot be interpreted at all. Per §19.6 ("no silent runs"), there is no
 * default-to-empty path.
 *
 * Unit conversions: v0 trigger center coordinates use km (`center_x_km`);
 * v1 uses meters (`center_x_m`). The migrator multiplies by 1000 and warns
 * so the user can sanity-check.
 *
 * Scheme-name de-conflation: a `dynamics.scheme` name can imply a coordinate
 * system and staggering. The migrator infers `grid.staggering` from known
 * Cartesian aliases / compound ids where the file omits it; bare
 * `klemp_wilhelmson_split` keeps deriving from the explicit staggering axis
 * (or the C-grid default). The scheme string itself is preserved.
 */

namespace tmv::init::config
{

/**
 * @brief A single migration decision worth surfacing to the user.
 *
 * `line == 0` indicates a synthesized/derived warning not tied to a single
 * source line (e.g. "inferred grid.staggering from dynamics.scheme").
 */
struct MigrationWarning
{
    std::string key;      ///< v0 key path or empty for synthesized decisions
    int         line;     ///< 1-based source line; 0 if not applicable
    std::string action;   ///< e.g. "dropped", "renamed to grid.ztop", "unit km→m"
    std::string reason;   ///< human explanation of the decision
};

/**
 * @brief Outcome of `migrate_v0_to_v1`.
 *
 * Mirrors `LoadResult` from v1_loader: on success `config` is populated and
 * `errors` is empty; on hard failure `config` is std::nullopt and `errors`
 * lists everything that prevented producing a valid v1 struct. Warnings
 * are advisory — they accompany a successful migration too.
 */
struct MigrationResult
{
    std::optional<ConfigV1>       config;
    std::vector<MigrationWarning> warnings;
    std::vector<LoadError>        errors;

    bool ok() const { return config.has_value() && errors.empty(); }
};

/**
 * @brief Read a v0 YAML config from disk and translate it to a ConfigV1.
 *
 * @param v0_path Path to a v0 YAML file (without `schema:` stamp).
 * @return MigrationResult. Check `result.ok()` for hard success; inspect
 *         `result.warnings` for advisory decisions even when successful.
 */
MigrationResult migrate_v0_to_v1(const std::string& v0_path);

/**
 * @brief Serialize a ConfigV1 back to YAML text matching docs/CM1Mapping.md §20 layout.
 *
 * Output starts with `schema: nimbus.v1`, includes section header comments
 * referencing CM1 namelist groups, and writes the complete canonical field
 * set for the selected source/scheme. Mutually exclusive source fields are
 * deliberately omitted so serialization cannot manufacture inert requests.
 *
 * Used by the `--migrate-config` CLI to print the translated config to
 * stdout for user review.
 */
std::string serialize_v1_to_yaml(const ConfigV1& config);

/**
 * @brief Format a warning list as a human-readable multi-line string.
 *
 * Format: `<file>:<line>: <key>: <action> (<reason>)` per warning. Line
 * and key are omitted when 0 / empty.
 */
std::string format_warnings(const std::string& file,
                            const std::vector<MigrationWarning>& warnings);

/**
 * @brief Project a ConfigV1 onto the legacy v0 flat-key map shape.
 *
 * Used by `runtime_config.cpp::load_config` to transparently accept v1
 * configs without rewriting the v0 ingestion path: detect the `schema:`
 * stamp, parse via `load_v1_config`, project back through this function,
 * and feed the resulting map to the existing v0 setup logic.
 *
 * The projection is **lossy** in one direction: v1 schema is a strict
 * subset of v0 (per §19 design principles), so v0-only knobs like
 * `numerics.diffusion.dt_diffusion` or `output.zfp_tolerance` are NOT
 * emitted — they revert to their built-in defaults on the v0 side. v1
 * fields that have no flat-map equivalent (`boundary_conditions.*`,
 * `damping.*`, `coriolis.*`) are applied directly by the typed ingestion
 * boundary. Diffusion operator order is projected and validated explicitly.
 *
 * Unit conversions match the inverse of the migrator:
 *   - trigger center coordinates: m → km (÷1000)
 *
 * The compound canonical scheme name (e.g.
 * `klemp_wilhelmson_split_cartesian_cgrid`) is emitted when
 * `dynamics.scheme=klemp_wilhelmson_split` is the unqualified canonical;
 * legacy aliases like `cart` are passed through unchanged.
 *
 * @param config v1 configuration.
 * @return Flat dotted-key map matching `parse_yaml_simple` output shape.
 */
std::unordered_map<std::string, std::string>
serialize_v1_to_v0_map(const ConfigV1& config);

}  // namespace tmv::init::config
