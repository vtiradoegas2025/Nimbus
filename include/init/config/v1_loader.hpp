#pragma once

#include "init/config/config_v1.hpp"
#include "core/runtime/yaml_helpers.hpp"

#include <optional>
#include <string>
#include <vector>

/**
 * @file v1_loader.hpp
 * @brief YAML loader and validator for Nimbus v1 configuration files.
 *
 * Phase A2 deliverable of the v1 config refactor (docs/CM1Mapping.md §21).
 *
 * Loads a `schema: nimbus.v1` YAML file, validates each key against the
 * schema in §20 (required keys present, enums in allowed set, numeric
 * ranges sensible), and returns either a populated `ConfigV1` struct or a
 * list of line-numbered errors.
 *
 * Strictness: unrecognized v1 keys are hard errors. This implements the
 * §19.6 design principle ("no silent runs") — a typo like `microhpysics:`
 * surfaces as a loader error rather than silently dropping the section.
 *
 * Error collection is exhaustive: the loader walks the full schema before
 * returning, so a config with multiple errors yields a single report
 * covering all of them (researchers fix one round-trip, not N).
 *
 * Pre-v1 (v0) configs are rejected here. Translating v0 to v1 is the job
 * of `v0_to_v1` (Phase A3).
 */

namespace tmv::init::config
{

/**
 * @brief A single line-numbered validation error.
 *
 * `line == 0` indicates a file-level error not tied to a specific YAML
 * line (e.g. "required key missing"); otherwise `line` is 1-based and
 * points into the source file.
 */
struct LoadError
{
    std::string file;
    int         line;
    std::string key;
    std::string message;
};

/**
 * @brief Outcome of `load_v1_config`.
 *
 * On success: `config` is populated and `errors` is empty.
 * On failure: `config` is std::nullopt and `errors` lists everything wrong
 * with the input (exhaustive — not stopped at the first error).
 */
struct LoadResult
{
    std::optional<ConfigV1> config;
    std::vector<LoadError>  errors;

    bool ok() const { return config.has_value() && errors.empty(); }
};

/**
 * @brief Parse and validate a v1 YAML config from disk.
 *
 * Returns LoadResult with `config` populated on success, or with `errors`
 * populated and `config` empty on failure. Failure cases include:
 *   - file cannot be opened
 *   - missing or wrong `schema:` stamp (must be exactly `nimbus.v1`)
 *   - required key absent
 *   - value fails to parse as the expected type
 *   - enum value not in the allowed set
 *   - numeric value out of physical range (e.g. negative grid size)
 *   - unrecognized top-level key (typo or stale field)
 *
 * @param path Path to the YAML config file.
 * @return LoadResult; check `result.ok()` to discriminate success.
 */
LoadResult load_v1_config(const std::string& path);
LoadResult load_v1_config_fields(const std::string& path,
    std::unordered_map<std::string, YamlField> fields);

/**
 * @brief Format an error list as a human-readable multi-line string.
 *
 * Format: `<file>:<line>: <key>: <message>` per line (line omitted if 0,
 * key omitted if empty). Suitable for piping to stderr or to a Studio
 * upload-validation banner.
 */
std::string format_errors(const std::vector<LoadError>& errors);

}  // namespace tmv::init::config
