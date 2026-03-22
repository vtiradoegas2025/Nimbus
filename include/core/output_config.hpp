#pragma once

/**
 * @file output_config.hpp
 * @brief Output pipeline configuration for field export.
 *
 * Defines the output format, field selection presets, compression settings,
 * and async I/O options. Parsed from the `output.*` YAML config keys.
 */

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Output file format for exported fields.
 */
enum class OutputFormat : int
{
    npy_3d = 0,       ///< One 3D NPY file per field per step (default)
    npy_2d_slices,     ///< Legacy per-theta 2D NPY slices
    zfp,               ///< ZFP-compressed 3D volumes
    csv                ///< CSV files (i,j,k,value) for student/spreadsheet use
};

/**
 * @brief Named presets for field selection.
 */
enum class FieldPreset : int
{
    all = 0,       ///< All 102 fields (current behavior)
    minimal,       ///< 8 essential fields for basic visualization
    presentation,  ///< 16 fields curated for talks and papers
    core_only,     ///< 26 core Field3D fields (no derived/tranche)
    custom         ///< User-specified list from config
};

/**
 * @brief Complete output pipeline configuration.
 */
struct OutputConfig
{
    /// Output file format. Defaults to legacy 2D slices for backward compatibility.
    /// Set to npy_3d in config for 256x fewer files.
    OutputFormat format = OutputFormat::npy_2d_slices;

    /// Field selection preset
    FieldPreset preset = FieldPreset::all;

    /// User-specified field names (used when preset == custom)
    std::vector<std::string> custom_fields;

    /// Resolved set of field names to export (populated by resolve_output_fields)
    std::unordered_set<std::string> resolved_fields;

    /// Enable async I/O with background writer thread.
    /// Currently defaults to false; set to true in config to enable.
    bool async_io = false;

    /// ZFP absolute error tolerance (accuracy mode)
    double zfp_tolerance = 1.0e-4;

    /// ZFP compression mode: "accuracy", "precision", or "rate"
    std::string zfp_mode = "accuracy";

    /// ZFP bits per scalar for rate mode
    int zfp_rate_bps = 8;

    /// Interval between keyframes for ZFP delta encoding.
    /// Every keyframe_interval-th step writes a full (keyframe) ZFP volume;
    /// intermediate steps write delta-compressed volumes (current - previous).
    /// Set to 0 to disable delta encoding (all frames are keyframes).
    int zfp_keyframe_interval = 0;

    /// Log estimated total output size at startup
    bool estimate_disk_budget = true;

    /// Estimated bytes per export step (computed, not parsed)
    std::size_t estimated_bytes_per_export = 0;

    /// Estimated total bytes for the full simulation (computed, not parsed)
    std::size_t estimated_total_bytes = 0;
};

/**
 * @brief Returns the canonical field name list for a given preset.
 */
const std::vector<std::string>& get_preset_fields(FieldPreset preset);

/**
 * @brief Returns the full list of all known exportable field names.
 */
const std::vector<std::string>& get_all_field_names();

/**
 * @brief Resolves the preset and custom fields into the resolved_fields set.
 *
 * After calling this, config.resolved_fields contains the exact set of
 * field names that should be exported. Unknown field names in custom_fields
 * are logged as warnings and skipped.
 */
void resolve_output_fields(OutputConfig& config);

/**
 * @brief Estimates total output size and logs the disk budget.
 *
 * Populates config.estimated_bytes_per_export and config.estimated_total_bytes.
 * Logs a warning if estimated output exceeds 100 GB.
 */
void estimate_disk_budget(OutputConfig& config,
                          int nr, int nth, int nz,
                          int duration_s, int write_every_s);

/**
 * @brief Parses output config from a YAML key-value map.
 *
 * Reads output.format, output.fields, output.async_io, output.zfp_*,
 * and output.estimate_disk_budget. Does NOT read output.write_every_s
 * or output.outdir (those are handled separately for backward compatibility).
 */
OutputConfig parse_output_config(
    const std::unordered_map<std::string, std::string>& config);

/**
 * @brief Returns a human-readable label for an OutputFormat.
 */
const char* output_format_name(OutputFormat fmt);

/**
 * @brief Returns a human-readable label for a FieldPreset.
 */
const char* field_preset_name(FieldPreset preset);
