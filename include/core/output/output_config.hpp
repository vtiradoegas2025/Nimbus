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
 * @brief Cadence tier for tiered write scheduling (Tier 2b).
 *
 * Fields are assigned to a cadence tier that determines how frequently
 * they are written. Tight compression fields → fast cadence, moderate →
 * medium cadence, loose → slow cadence.
 */
enum class WriteCadenceTier : int
{
    fast = 0,    ///< Core dynamics: written at write_every_s
    medium,      ///< Thermo/moisture: written at write_cadence_medium_s
    slow         ///< Diagnostics/viz: written at write_cadence_slow_s
};

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

    /// ZFP absolute error tolerance (accuracy mode).
    /// Used as the global default when per-field tolerances are disabled.
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

    /// Use predictive (linear extrapolation) delta instead of simple delta.
    /// Computes residual = current - 2*prev + prev_prev, which is smaller
    /// than simple delta for smoothly evolving fields (1.5-2x improvement).
    /// Requires two previous frames in memory instead of one.
    bool zfp_predictive_delta = false;

    /// Enable sparse field zero-thresholding (Tier 3a).
    /// For sparse hydrometeor fields (qi, qs, qh, qg, qc, qr, radar),
    /// values below this threshold are zeroed before compression.
    /// ZFP compresses exact-zero blocks extremely efficiently (~3-10x
    /// improvement on sparse fields). Set to 0 to disable.
    float zfp_sparse_threshold = 0.0f;

    /// Enable float16 pre-quantization for bounded fields (Tier 3c).
    /// Reduces mantissa precision to 10 bits before ZFP compression,
    /// lowering entropy for ~2x improvement on applicable fields.
    /// Only affects loose-tier bounded fields (diagnostics, surface, column).
    bool zfp_float16_prefilter = false;

    /// Enable per-field ZFP tolerance tiers (Tier 2 compression).
    /// When true, fields are assigned tolerances based on their compression
    /// tier (tight, moderate, loose) rather than using a single global value.
    /// See apply_default_compression_tiers() for tier assignments.
    bool zfp_per_field_tolerances = false;

    /// Per-tier tolerance values (used when zfp_per_field_tolerances is true).
    /// Tight: core dynamics (u, v, w, rho, p, theta, vorticity).
    double zfp_tight_tolerance = 1.0e-4;
    /// Moderate: thermodynamics, moisture, derived fields.
    double zfp_moderate_tolerance = 1.0e-3;
    /// Loose: diagnostics, visualization, radar, surface/column fields.
    double zfp_loose_tolerance = 1.0e-2;

    /// Per-field tolerance overrides (highest priority).
    /// Populated by apply_default_compression_tiers() from tier assignments,
    /// then overwritten by any explicit YAML per-field entries.
    /// In the write path, looked up per field with fallback to zfp_tolerance.
    std::unordered_map<std::string, double> zfp_field_tolerances;

    /// Enable tiered write cadence (Tier 2b).
    /// When true, fields are written at different intervals based on their
    /// cadence tier: fast (write_every_s), medium, slow. Reduces total
    /// output by ~3.5x independently of compression ratio.
    bool tiered_write_cadence = false;

    /// Write interval for medium-cadence fields (seconds).
    /// Applies to thermodynamics, moisture, and derived fields.
    int write_cadence_medium_s = 30;

    /// Write interval for slow-cadence fields (seconds).
    /// Applies to diagnostics, visualization, radar, surface/column fields.
    int write_cadence_slow_s = 60;

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
 * @brief Returns the write cadence tier for a given field name.
 *
 * Tight compression fields → fast, moderate → medium, loose → slow.
 * Uses the same field categorization as ZFP compression tiers.
 */
WriteCadenceTier get_field_write_cadence_tier(const std::string& field_name);

/**
 * @brief Returns true if a field is eligible for sparse zero-thresholding.
 *
 * Sparse fields are hydrometeors and radar that are zero over >90% of the
 * domain. Thresholding removes floating-point noise in "zero" voxels.
 */
bool is_sparse_eligible(const std::string& field_name);

/**
 * @brief Returns true if a field is eligible for float16 pre-quantization.
 *
 * Eligible fields are bounded diagnostics and surface/column fields
 * where float16 precision (10-bit mantissa) is sufficient for export.
 */
bool is_float16_eligible(const std::string& field_name);

/**
 * @brief Populates zfp_field_tolerances from tier assignments.
 *
 * Assigns each field in resolved_fields to a compression tier (tight,
 * moderate, or loose) and writes the corresponding tolerance into
 * config.zfp_field_tolerances. Only active when zfp_per_field_tolerances
 * is true. Call after resolve_output_fields().
 */
void apply_default_compression_tiers(OutputConfig& config);

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
