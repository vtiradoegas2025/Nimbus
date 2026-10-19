#pragma once

/**
 * @file output_config.hpp
 * @brief Output pipeline configuration for field export.
 *
 * Defines the output format, field selection presets, compression settings,
 * and async I/O options. Parsed from the `output.*` YAML config keys.
 */

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/output/qualified_radar_zfp_scope.hpp"

/**
 * @brief Cadence tier for tiered write scheduling (Tier 2b).
 *
 * Fields are assigned to a cadence tier that determines how frequently
 * they are written. Tight compression fields → fast cadence, moderate →
 * medium cadence, loose → slow cadence.
 */
enum class WriteCadenceTier : int
{
    fast = 0, ///< Core dynamics: written at write_every_s
    medium,   ///< Thermo/moisture: written at write_cadence_medium_s
    slow      ///< Diagnostics/viz: written at write_cadence_slow_s
};

/**
 * @brief Output file format for exported fields.
 */
enum class OutputFormat : int
{
    npy_3d = 0,    ///< One 3D NPY file per field per step (default)
    npy_2d_slices, ///< Legacy per-theta 2D NPY slices
    netcdf,        ///< One self-describing netCDF4 history file per step
    zfp,           ///< ZFP-compressed 3D volumes
    csv            ///< CSV files (i,j,k,value) for student/spreadsheet use
};

/**
 * Whether a volume frame must be assembled into an ExportSnapshot before it
 * is written. netCDF and ZFP require their dedicated writers even when
 * background I/O is disabled; only NPY/CSV have a direct synchronous path.
 */
inline bool requires_snapshot_assembly(OutputFormat format, bool async_io)
{
    return async_io || format == OutputFormat::netcdf || format == OutputFormat::zfp;
}

/**
 * Return true only when a fixed model timestep lands exactly on every sparse
 * qualification-output deadline. The runtime publishes after root macrosteps,
 * so callers with nesting must validate both the root and child/effective dt.
 */
inline bool qualification_diagnostics_cadence_reachable(
    double interval_s, double timestep_s)
{
    const double cadence_steps = timestep_s > 0.0
        ? interval_s / timestep_s : 0.0;
    return std::isfinite(cadence_steps) && cadence_steps > 0.0 &&
        std::abs(cadence_steps - std::round(cadence_steps)) <=
            1.0e-12 * std::max(1.0, std::abs(cadence_steps));
}

/** Sparse reference ownership is root-only without configured nesting and
 * child-only once configured nesting has established its live relationship. */
inline bool qualification_diagnostics_child_ownership_consistent(
    bool nesting_configured, bool child_relationship_ready)
{
    return nesting_configured == child_relationship_ready;
}

/**
 * @brief Named presets for field selection.
 */
enum class FieldPreset : int
{
    all = 0,      ///< Every registered output field (dynamics_full)
    minimal,      ///< 8 essential fields for basic visualization
    radar_ready,  ///< Minimal state plus bounded reflectivity and radial velocity
    radar_zdr_exploratory,
    radar_polarimetric_ready,
    presentation, ///< Compact fields curated for talks and papers
    studio_full,  ///< Studio's complete Nimbus-native visualization catalog
    core_only,    ///< 26 core Field3D fields (no derived/tranche)
    custom,            ///< User-specified list from config
    science_validation ///< Minimal state plus cold-pool and precipitation maps
};

enum class QualificationDiagnosticsPreset : int
{
    none = 0,
    moving_reference_v1,
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

    /// Live shared-memory publication cadence in simulation seconds.
    double live_shm_interval_s = 1.0;

    /// Explicit opt-in to publish after every completed model step.
    bool live_shm_full_rate = false;

    /// Independent lossless planar stream used by the moving/reference
    /// frontier. Its fixed field inventory cannot inherit a broad history
    /// preset or compression mode.
    QualificationDiagnosticsPreset qualification_diagnostics_preset =
        QualificationDiagnosticsPreset::none;
    int qualification_diagnostics_interval_s = 30;

    /// Enable online Lagrangian parcel trajectories. Keeping this in the
    /// output configuration makes the YAML the complete run definition; the
    /// legacy CLI switches remain explicit overrides.
    bool trajectories = false;

    /// Simulation time at which the trajectory seed lattice is released.
    double trajectory_seed_time_s = 0.0;

    /// Horizontal trajectory seed count in each direction (nx == ny).
    int trajectory_count = 5;

    /// Zero records every completed model step; positive values define the
    /// trajectory diagnostic sampling cadence independently of integration.
    double trajectory_sample_interval_s = 0.0;

    /// Per-parcel retained history bound, including the seed sample.
    int trajectory_max_samples_per_parcel = 16384;

    /// netCDF4/HDF5 deflate level [0, 9]. Zero disables deflate.
    int netcdf_deflate_level = 2;

    /// Enable the HDF5 shuffle filter before netCDF deflate.
    bool netcdf_shuffle = true;

    /// Approximate upper bound for each netCDF field chunk.
    int netcdf_target_chunk_kib = 1024;

    /// ZFP absolute error tolerance (accuracy mode).
    /// Used as the global default when per-field tolerances are disabled.
    double zfp_tolerance = 1.0e-4;

    /// ZFP compression mode: "accuracy", "precision", or "rate"
    std::string zfp_mode = "accuracy";

    /// Exact public tier token retained after format normalization.
    std::string zfp_requested_format_tier;

    /// Significant binary bits retained in ZFP precision mode.
    int zfp_precision_bits = 16;

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

    /// Field-level execution policy. Libzfp remains serial inside every field.
    std::string zfp_field_execution = "serial";

    /// Zero selects a backend-aware bounded worker count.
    int zfp_field_workers = 0;

    /// Hard cap for transactional parallel codec/history scratch.
    int zfp_field_scratch_mib = 64;

    /// Runtime-resolved count; not part of serialized configuration.
    int zfp_resolved_field_workers = 1;

    /// Internal qualified radar candidate; public admission remains separate.
    bool zfp_reversible_radar_v4 = false;

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

    /// Number of newest completed frames to retain in each output series.
    /// Manifest-declared compression ancestors are retained automatically.
    /// Zero preserves the historical unlimited-retention behavior.
    int keep_last_frames = 0;

    /// Free-space reserve checked before each frame write. The estimated frame
    /// size is required in addition to this reserve. Zero disables the guard.
    int minimum_free_space_mib = 0;

    /// Per-series ceiling for the conservative retained history file estimate.
    /// Zero disables the ceiling for exploratory/legacy configurations.
    int maximum_retained_history_files = 0;

    /// Log estimated total output size at startup
    bool estimate_disk_budget = true;

    /// Estimated bytes per export step (computed, not parsed)
    std::size_t estimated_bytes_per_export = 0;

    /// Estimated total bytes for the full simulation (computed, not parsed)
    std::size_t estimated_total_bytes = 0;

    /// Conservative final regular-file count for one history series.
    std::size_t estimated_retained_history_files = 0;

    std::size_t qualification_diagnostics_estimated_bytes_per_export = 0;
    std::size_t qualification_diagnostics_estimated_total_bytes = 0;
};

/// Validate the negotiated live-volume field inventory before SHM allocation.
inline bool validate_live_shm_fields(const std::vector<std::string>& fields,
                                     std::string& error)
{
    static const std::unordered_set<std::string> supported = {
        "u", "v", "w", "rho", "p", "theta", "qv", "qc", "qr",
        "qh", "qg", "radar", "reflectivity_dbz", "tracer", "tke",
        "qi", "qs", "vorticity_r", "vorticity_theta", "vorticity_z",
        "stretching_term", "tilting_term", "baroclinic_term", "p_prime",
        "dynamic_pressure", "buoyancy_pressure", "angular_momentum",
        "angular_momentum_tendency"};
    if (fields.empty())
    {
        error = "live shared-memory fields must not be empty";
        return false;
    }
    if (fields.size() > 16)
    {
        error = "live shared-memory field count exceeds the transport limit of 16";
        return false;
    }
    std::unordered_set<std::string> unique;
    for (const std::string& field : fields)
    {
        if (!supported.count(field))
        {
            error = "unsupported live shared-memory field '" + field + "'";
            return false;
        }
        if (!unique.insert(field).second)
        {
            error = "duplicate live shared-memory field '" + field + "'";
            return false;
        }
    }
    error.clear();
    return true;
}

inline int resolve_zfp_field_workers(int requested_workers,
                                     int scratch_budget_mib,
                                     int field_count,
                                     bool accelerator_backend,
                                     int solver_threads,
                                     int hardware_threads) noexcept
{
    constexpr int maximum_workers = 16;
    constexpr int worker_scratch_reserve_mib = 2;
    if (requested_workers < 0 || requested_workers > maximum_workers ||
        scratch_budget_mib < worker_scratch_reserve_mib || field_count <= 0 ||
        solver_threads <= 0 || hardware_threads <= 0)
        return 0;
    const int backend_auto = accelerator_backend
        ? std::max(1, std::min(8, hardware_threads - 2))
        : std::max(1, std::min(4, hardware_threads - solver_threads - 1));
    const int hard_cap = std::min({maximum_workers, field_count,
                                   scratch_budget_mib /
                                       worker_scratch_reserve_mib});
    if (hard_cap < 1 || (requested_workers > 0 && requested_workers > hard_cap))
        return 0;
    return requested_workers > 0 ? requested_workers
                                 : std::min(backend_auto, hard_cap);
}

/** One field admitted by the qualified ZFP core-history contract. */
struct QualifiedZfpFieldBudget
{
    double      max_configured_absolute_error = 0.0;
    std::string units;
};

/**
 * Unit-aware maximum codec tolerances for fields admitted to qualified ZFP
 * history. Fields absent from this registry remain visualization-only.
 */
const std::unordered_map<std::string, QualifiedZfpFieldBudget>& qualified_zfp_field_budgets();

/** Exact field membership for every public preset admitted to qualified ZFP. */
const std::unordered_map<std::string, std::vector<std::string>>& qualified_zfp_preset_fields();

/** Exact non-field settings admitted by qualified ZFP history. These accessors
 * are consumed by both native validation and the exact-binary capability
 * catalog so Studio cannot maintain a second compression policy. */
const char*                     qualified_zfp_mode();
bool                            qualified_zfp_requires_per_field_tolerances();

/**
 * Effective absolute-accuracy tolerance for one field. Explicit/per-tier
 * values take precedence over the global value, then science-critical
 * field-scale ceilings prevent a broad visualization tolerance from erasing a
 * small-in-SI diagnostic.
 */
double effective_zfp_accuracy_tolerance(const OutputConfig& config,
                                        const std::string& field_name);
double                          qualified_zfp_max_sparse_threshold();
bool                            qualified_zfp_allows_float16_prefilter();
const std::vector<std::string>& qualified_zfp_preset_ids();
const std::vector<std::string>& qualified_zfp_field_execution_policies();
int qualified_zfp_max_field_workers();
int qualified_zfp_min_field_scratch_mib();
int qualified_zfp_max_field_scratch_mib();

/**
 * Validate the complete qualified ZFP history contract. An empty result means
 * the selected field set/mode/prefilters and every resolved field tolerance
 * are admitted. This does not apply to restart/checkpoint files.
 */
std::vector<std::string> qualified_radar_zfp_v4_candidate_errors(
    const OutputConfig& config,
    const nimbus::output::QualifiedRadarZfpV4Context& context);
std::vector<std::string> validate_qualified_zfp_history(
    const OutputConfig& config,
    const nimbus::output::QualifiedRadarZfpV4Context* radar_context = nullptr);

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
 * field names that should be exported. Unknown or empty custom selections are
 * rejected so the published experiment cannot differ from the request.
 */
void resolve_output_fields(OutputConfig& config);

/** Merge the ordinary radar product selection into the resolved publication set. */
void merge_selected_radar_output_fields(
    OutputConfig& config, const std::vector<std::string>& selected_products);

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
 * Volume formats account for registered surface/column products as 2-D maps.
 * Logs a warning if estimated output exceeds 100 GB.
 */
void estimate_disk_budget(OutputConfig& config, int nr, int nth, int nz, int duration_s, int write_every_s);
void estimate_qualification_diagnostics_disk_budget(
    OutputConfig& config, int nr, int nth, int duration_s);

const std::vector<std::string>& qualification_diagnostics_fields(
    QualificationDiagnosticsPreset preset);
const char* qualification_diagnostics_preset_name(
    QualificationDiagnosticsPreset preset);

/**
 * @brief Parses output config from a YAML key-value map.
 *
 * Reads output.format, output.fields, output.async_io, output.zfp_*,
 * and output.estimate_disk_budget. Does NOT read output.write_every_s
 * or output.outdir (those are handled separately for backward compatibility).
 */
OutputConfig parse_output_config(const std::unordered_map<std::string, std::string>& config);

/**
 * @brief Returns a human-readable label for an OutputFormat.
 */
const char* output_format_name(OutputFormat fmt);

/**
 * @brief Returns a human-readable label for a FieldPreset.
 */
const char* field_preset_name(FieldPreset preset);
