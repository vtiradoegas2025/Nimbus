/**
 * @file output_config.cpp
 * @brief Output pipeline configuration parsing and field resolution.
 */

#include "core/output/output_config.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

std::string local_to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool local_parse_bool(const std::string& value)
{
    const std::string v = local_to_lower(value);
    return v == "true" || v == "1" || v == "yes" || v == "on";
}

bool local_try_parse_double(const std::string& value, double& out)
{
    try
    {
        std::size_t pos = 0;
        out = std::stod(value, &pos);
        return pos > 0 && std::isfinite(out);
    }
    catch (...)
    {
        return false;
    }
}

bool local_try_parse_int(const std::string& value, int& out)
{
    try
    {
        std::size_t pos = 0;
        out = std::stoi(value, &pos);
        return pos > 0;
    }
    catch (...)
    {
        return false;
    }
}

std::vector<std::string> local_parse_string_list(const std::string& value)
{
    std::string cleaned = value;
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '"'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\''), cleaned.end());

    std::vector<std::string> out;
    std::stringstream ss(cleaned);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        item.erase(item.begin(), std::find_if(item.begin(), item.end(),
                   [](unsigned char ch) { return !std::isspace(ch); }));
        item.erase(std::find_if(item.rbegin(), item.rend(),
                   [](unsigned char ch) { return !std::isspace(ch); }).base(), item.end());
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

} // anonymous namespace

namespace {

// -- Preset field lists -------------------------------------------------------

const std::vector<std::string> k_minimal_fields = {
    "u", "v", "w", "theta", "p", "rho", "qv", "qr"
};

const std::vector<std::string> k_presentation_fields = {
    // minimal
    "u", "v", "w", "theta", "p", "rho", "qv", "qr",
    // additional diagnostics for talks/papers
    "reflectivity_dbz", "vorticity_z", "vorticity_magnitude",
    "total_condensate", "temperature", "buoyancy",
    "composite_reflectivity", "stp"
};

const std::vector<std::string> k_core_only_fields = {
    "u", "v", "w", "rho", "p", "theta",
    "qv", "qc", "qr", "qi", "qs", "qh", "qg",
    "radar", "tracer",
    "vorticity_r", "vorticity_theta", "vorticity_z",
    "stretching_term", "tilting_term", "baroclinic_term",
    "p_prime", "dynamic_pressure", "buoyancy_pressure",
    "angular_momentum", "angular_momentum_tendency"
};

const std::vector<std::string> k_all_fields = {
    // Core fields (26)
    "u", "v", "w", "rho", "p", "theta",
    "qv", "qc", "qr", "qi", "qs", "qh", "qg",
    "radar", "tracer",
    "vorticity_r", "vorticity_theta", "vorticity_z",
    "stretching_term", "tilting_term", "baroclinic_term",
    "p_prime", "dynamic_pressure", "buoyancy_pressure",
    "angular_momentum", "angular_momentum_tendency",

    // Derived fields (35)
    "temperature", "theta_prime", "theta_v", "theta_e",
    "dewpoint", "relative_humidity", "saturation_mixing_ratio",
    "total_condensate", "reflectivity_dbz",
    "vorticity_magnitude", "divergence", "buoyancy",
    "horizontal_vorticity_streamwise", "horizontal_vorticity_crosswise",
    "pressure_gradient_force_x", "pressure_gradient_force_y", "pressure_gradient_force_z",
    "storm_relative_winds", "helicity_density", "okubo_weiss",
    "theta_w", "zdr", "kdp", "rhohv",
    "streamlines", "trajectory_paths", "q_vectors",
    "turbulent_diffusion_term", "cross_section", "rhi_slice",
    "hodograph_aligned_cross_section",
    "forward_trajectories", "backward_trajectories",
    "parcel_buoyancy_trajectory", "vorticity_trajectory",
    "circulation_material_surface",

    // Tranche: surface (12)
    "u10", "v10", "t2", "td2",
    "surface_pressure_perturbation",
    "surface_sensible_heat_flux", "surface_latent_heat_flux",
    "surface_moisture_flux", "skin_temperature",
    "cold_pool_boundary", "precip_rate", "accumulated_rainfall",

    // Tranche: column (19)
    "composite_reflectivity", "column_max_w", "column_max_vorticity",
    "vil", "cloud_top_height", "cloud_base_height",
    "lcl", "lfc", "el", "cape", "cin",
    "lifted_index", "k_index", "showalter_index", "total_totals",
    "srh_0_1km", "srh_0_3km", "ehi", "scp", "stp",

    // Tranche: radar synthetic (5)
    "ppi_sweep", "rhi_sweep", "bwer",
    "mesocyclone_diagnostic", "vrot"
};

const std::unordered_set<std::string> k_all_fields_set(
    k_all_fields.begin(), k_all_fields.end());

// -- Per-field ZFP compression tiers (Tier 2) --------------------------------
// Tight: core prognostic state + vorticity budget — highest fidelity needed
// for numerical stability and scientific reproducibility.
const std::unordered_set<std::string> k_zfp_tight_fields = {
    "u", "v", "w", "rho", "p", "theta",
    "p_prime", "dynamic_pressure", "buoyancy_pressure",
    "vorticity_r", "vorticity_theta", "vorticity_z",
    "stretching_term", "tilting_term", "baroclinic_term"
};

// Moderate: thermodynamics, moisture species, and derived thermo fields —
// important for physical fidelity but tolerant of slightly relaxed precision.
const std::unordered_set<std::string> k_zfp_moderate_fields = {
    "qv", "qc", "qr", "qi", "qs", "qh", "qg",
    "temperature", "theta_prime", "theta_v", "theta_e", "theta_w",
    "dewpoint", "relative_humidity", "saturation_mixing_ratio",
    "total_condensate", "buoyancy",
    "angular_momentum", "angular_momentum_tendency", "tracer"
};

// Loose: everything else — diagnostics, visualization, radar, surface/column.
// These are derived or post-processed fields where 1e-2 is visually and
// scientifically indistinguishable from full precision.
// (No explicit list needed — any field not in tight or moderate is loose.)

// -- Sparse-eligible fields (Tier 3a) -----------------------------------------
// Hydrometeor and radar fields that are zero over >90% of the domain.
// Thresholding removes floating-point noise so ZFP can compress
// zero blocks efficiently.
const std::unordered_set<std::string> k_sparse_eligible_fields = {
    "qi", "qs", "qh", "qg", "qc", "qr",
    "radar", "total_condensate",
    "reflectivity_dbz", "zdr", "kdp"
};

// -- Float16-eligible fields (Tier 3c) ----------------------------------------
// Bounded diagnostic and surface/column fields where 10-bit mantissa precision
// is sufficient for export. These fields have well-defined physical ranges
// and don't participate in numerical integration.
const std::unordered_set<std::string> k_float16_eligible_fields = {
    // Bounded diagnostics (0-100% or similar small ranges)
    "relative_humidity", "rhohv", "tracer",
    // Radar-derived (bounded, visualization-oriented)
    "reflectivity_dbz", "zdr", "kdp",
    // Surface fields (near-surface values with limited range)
    "u10", "v10", "t2", "td2", "skin_temperature",
    "surface_sensible_heat_flux", "surface_latent_heat_flux",
    "surface_moisture_flux", "surface_pressure_perturbation",
    "cold_pool_boundary", "precip_rate", "accumulated_rainfall",
    // Column diagnostics (integrated/derived indices)
    "composite_reflectivity", "column_max_w", "column_max_vorticity",
    "vil", "cloud_top_height", "cloud_base_height",
    "lcl", "lfc", "el", "cape", "cin",
    "lifted_index", "k_index", "showalter_index", "total_totals",
    "srh_0_1km", "srh_0_3km", "ehi", "scp", "stp",
    // Radar synthetic
    "ppi_sweep", "rhi_sweep", "bwer", "mesocyclone_diagnostic", "vrot"
};

} // anonymous namespace

const std::vector<std::string>& get_preset_fields(FieldPreset preset)
{
    switch (preset)
    {
    case FieldPreset::minimal:      return k_minimal_fields;
    case FieldPreset::presentation: return k_presentation_fields;
    case FieldPreset::core_only:    return k_core_only_fields;
    case FieldPreset::all:          return k_all_fields;
    case FieldPreset::custom:       return k_all_fields; // fallback
    }
    return k_all_fields;
}

const std::vector<std::string>& get_all_field_names()
{
    return k_all_fields;
}

void resolve_output_fields(OutputConfig& config)
{
    config.resolved_fields.clear();

    if (config.preset == FieldPreset::custom)
    {
        for (const auto& name : config.custom_fields)
        {
            if (k_all_fields_set.count(name))
            {
                config.resolved_fields.insert(name);
            }
            else
            {
                tmv::log_warn("[OUTPUT] unknown field '", name, "' in output.fields — skipping");
            }
        }
        if (config.resolved_fields.empty())
        {
            tmv::log_warn("[OUTPUT] no valid fields in custom list; falling back to 'minimal' preset");
            config.preset = FieldPreset::minimal;
            const auto& fallback = k_minimal_fields;
            config.resolved_fields.insert(fallback.begin(), fallback.end());
        }
    }
    else
    {
        const auto& preset_fields = get_preset_fields(config.preset);
        config.resolved_fields.insert(preset_fields.begin(), preset_fields.end());
    }
}

bool is_sparse_eligible(const std::string& field_name)
{
    return k_sparse_eligible_fields.count(field_name) > 0;
}

bool is_float16_eligible(const std::string& field_name)
{
    return k_float16_eligible_fields.count(field_name) > 0;
}

WriteCadenceTier get_field_write_cadence_tier(const std::string& field_name)
{
    if (k_zfp_tight_fields.count(field_name))
    {
        return WriteCadenceTier::fast;
    }
    if (k_zfp_moderate_fields.count(field_name))
    {
        return WriteCadenceTier::medium;
    }
    return WriteCadenceTier::slow;
}

void apply_default_compression_tiers(OutputConfig& config)
{
    if (!config.zfp_per_field_tolerances)
    {
        return;
    }

    // Assign each resolved field to its tier tolerance.
    // Existing entries (from YAML overrides) are preserved.
    std::size_t tight_count = 0;
    std::size_t moderate_count = 0;
    std::size_t loose_count = 0;

    for (const auto& name : config.resolved_fields)
    {
        if (config.zfp_field_tolerances.count(name))
        {
            continue; // Explicit YAML override takes precedence
        }

        if (k_zfp_tight_fields.count(name))
        {
            config.zfp_field_tolerances[name] = config.zfp_tight_tolerance;
            ++tight_count;
        }
        else if (k_zfp_moderate_fields.count(name))
        {
            config.zfp_field_tolerances[name] = config.zfp_moderate_tolerance;
            ++moderate_count;
        }
        else
        {
            config.zfp_field_tolerances[name] = config.zfp_loose_tolerance;
            ++loose_count;
        }
    }

    tmv::log_info("[OUTPUT] ZFP per-field tiers: ",
                  tight_count, " tight (", config.zfp_tight_tolerance, "), ",
                  moderate_count, " moderate (", config.zfp_moderate_tolerance, "), ",
                  loose_count, " loose (", config.zfp_loose_tolerance, ")");
}

void estimate_disk_budget(OutputConfig& config,
                          int nr, int nth, int nz,
                          int duration_s, int write_every_s)
{
    if (write_every_s <= 0 || duration_s <= 0)
    {
        config.estimated_bytes_per_export = 0;
        config.estimated_total_bytes = 0;
        return;
    }

    const auto num_exports = static_cast<std::size_t>(duration_s / write_every_s);
    const auto voxels = static_cast<std::size_t>(nr) *
                        static_cast<std::size_t>(nth) *
                        static_cast<std::size_t>(nz);
    const std::size_t num_fields = config.resolved_fields.size();
    const std::size_t bytes_per_field = voxels * sizeof(float);

    double compression_ratio = 1.0;
    if (config.format == OutputFormat::zfp)
    {
        // ZFP alone typically achieves 4-10x on smooth float data.
        // Delta encoding improves this to 30-150x since deltas between
        // consecutive timesteps are small and highly compressible.
        compression_ratio = (config.zfp_keyframe_interval > 0) ? 30.0 : 5.0;
    }

    config.estimated_bytes_per_export = static_cast<std::size_t>(
        static_cast<double>(num_fields * bytes_per_field) / compression_ratio);

    // Add overhead for manifest/validation JSON (~7KB per export)
    config.estimated_total_bytes =
        config.estimated_bytes_per_export * num_exports + num_exports * 7000;

    if (!config.estimate_disk_budget)
    {
        return;
    }

    const double total_gb = static_cast<double>(config.estimated_total_bytes)
                            / (1024.0 * 1024.0 * 1024.0);
    const double per_export_mb = static_cast<double>(config.estimated_bytes_per_export)
                                 / (1024.0 * 1024.0);

    tmv::log_info("[OUTPUT] Disk budget: ", num_fields, " fields x ", num_exports, " exports");
    tmv::log_info("[OUTPUT]   Per export: ", std::round(per_export_mb * 10.0) / 10.0, " MB");
    if (config.format == OutputFormat::zfp)
    {
        const char* delta_note = (config.zfp_keyframe_interval > 0)
            ? " (ZFP+delta ~30x estimate)" : " (ZFP ~5x estimate)";
        tmv::log_info("[OUTPUT]   Total estimated: ",
                      std::round(total_gb * 10.0) / 10.0, " GB", delta_note);
    }
    else
    {
        tmv::log_info("[OUTPUT]   Total estimated: ",
                      std::round(total_gb * 10.0) / 10.0, " GB");
    }

    if (total_gb > 100.0)
    {
        tmv::log_warn("[OUTPUT] estimated output exceeds 100 GB. "
                      "Consider output.fields: minimal or output.format: zfp");
    }
}

OutputConfig parse_output_config(
    const std::unordered_map<std::string, std::string>& config)
{
    OutputConfig out;

    // output.format
    if (config.count("output.format"))
    {
        const std::string val = local_to_lower(config.at("output.format"));
        if (val == "npy_3d" || val == "npy3d")
        {
            out.format = OutputFormat::npy_3d;
        }
        else if (val == "npy_2d_slices" || val == "npy2d" || val == "legacy")
        {
            out.format = OutputFormat::npy_2d_slices;
        }
        else if (val == "zfp")
        {
#ifdef HAVE_ZFP
            out.format = OutputFormat::zfp;
#else
            throw std::runtime_error(
                "output.format 'zfp' requested but binary was compiled without ZFP support. "
                "Rebuild with ZFP=1 or change output.format.");
#endif
        }
        else if (val == "csv")
        {
            out.format = OutputFormat::csv;
        }
        else
        {
            tmv::log_warn("[OUTPUT] unknown output.format '", val, "'; using npy_3d");
        }
    }

    // output.fields
    if (config.count("output.fields"))
    {
        const std::string val = local_to_lower(config.at("output.fields"));
        if (val == "all")
        {
            out.preset = FieldPreset::all;
        }
        else if (val == "minimal")
        {
            out.preset = FieldPreset::minimal;
        }
        else if (val == "presentation")
        {
            out.preset = FieldPreset::presentation;
        }
        else if (val == "core_only" || val == "core")
        {
            out.preset = FieldPreset::core_only;
        }
        else
        {
            // Assume it's a bracketed list: [u, v, w, ...]
            out.preset = FieldPreset::custom;
            out.custom_fields = local_parse_string_list(config.at("output.fields"));
        }
    }

    // output.async_io
    if (config.count("output.async_io"))
    {
        out.async_io = local_parse_bool(config.at("output.async_io"));
    }

    // output.zfp_tolerance
    if (config.count("output.zfp_tolerance"))
    {
        double parsed = 0.0;
        if (local_try_parse_double(config.at("output.zfp_tolerance"), parsed)
            && parsed > 0.0)
        {
            out.zfp_tolerance = parsed;
        }
    }

    // output.zfp_mode
    if (config.count("output.zfp_mode"))
    {
        const std::string val = local_to_lower(config.at("output.zfp_mode"));
        if (val == "accuracy" || val == "precision" || val == "rate")
        {
            out.zfp_mode = val;
        }
        else
        {
            tmv::log_warn("[OUTPUT] unknown output.zfp_mode '", val, "'; using accuracy");
        }
    }

    // output.zfp_rate_bps
    if (config.count("output.zfp_rate_bps"))
    {
        int parsed = 0;
        if (local_try_parse_int(config.at("output.zfp_rate_bps"), parsed)
            && parsed > 0)
        {
            out.zfp_rate_bps = parsed;
        }
    }

    // output.zfp_keyframe_interval
    if (config.count("output.zfp_keyframe_interval"))
    {
        int parsed = 0;
        if (local_try_parse_int(config.at("output.zfp_keyframe_interval"), parsed)
            && parsed >= 0)
        {
            out.zfp_keyframe_interval = parsed;
        }
    }

    // output.zfp_sparse_threshold
    if (config.count("output.zfp_sparse_threshold"))
    {
        double parsed = 0.0;
        if (local_try_parse_double(config.at("output.zfp_sparse_threshold"), parsed)
            && parsed >= 0.0)
        {
            out.zfp_sparse_threshold = static_cast<float>(parsed);
        }
    }

    // output.zfp_float16_prefilter
    if (config.count("output.zfp_float16_prefilter"))
    {
        out.zfp_float16_prefilter =
            local_parse_bool(config.at("output.zfp_float16_prefilter"));
    }

    // output.zfp_predictive_delta
    if (config.count("output.zfp_predictive_delta"))
    {
        out.zfp_predictive_delta =
            local_parse_bool(config.at("output.zfp_predictive_delta"));
    }

    // output.zfp_per_field_tolerances
    if (config.count("output.zfp_per_field_tolerances"))
    {
        out.zfp_per_field_tolerances =
            local_parse_bool(config.at("output.zfp_per_field_tolerances"));
    }

    // output.zfp_tight_tolerance
    if (config.count("output.zfp_tight_tolerance"))
    {
        double parsed = 0.0;
        if (local_try_parse_double(config.at("output.zfp_tight_tolerance"), parsed)
            && parsed > 0.0)
        {
            out.zfp_tight_tolerance = parsed;
        }
    }

    // output.zfp_moderate_tolerance
    if (config.count("output.zfp_moderate_tolerance"))
    {
        double parsed = 0.0;
        if (local_try_parse_double(config.at("output.zfp_moderate_tolerance"), parsed)
            && parsed > 0.0)
        {
            out.zfp_moderate_tolerance = parsed;
        }
    }

    // output.zfp_loose_tolerance
    if (config.count("output.zfp_loose_tolerance"))
    {
        double parsed = 0.0;
        if (local_try_parse_double(config.at("output.zfp_loose_tolerance"), parsed)
            && parsed > 0.0)
        {
            out.zfp_loose_tolerance = parsed;
        }
    }

    // output.zfp_field_tolerance.<field_name> — per-field overrides
    {
        const std::string prefix = "output.zfp_field_tolerance.";
        for (const auto& [key, val] : config)
        {
            if (key.size() > prefix.size() &&
                key.compare(0, prefix.size(), prefix) == 0)
            {
                const std::string field_name = key.substr(prefix.size());
                double parsed = 0.0;
                if (local_try_parse_double(val, parsed) && parsed > 0.0)
                {
                    out.zfp_field_tolerances[field_name] = parsed;
                }
                else
                {
                    tmv::log_warn("[OUTPUT] invalid zfp_field_tolerance for '",
                                  field_name, "': ", val);
                }
            }
        }
    }

    // output.tiered_write_cadence
    if (config.count("output.tiered_write_cadence"))
    {
        out.tiered_write_cadence =
            local_parse_bool(config.at("output.tiered_write_cadence"));
    }

    // output.write_cadence_medium_s
    if (config.count("output.write_cadence_medium_s"))
    {
        int parsed = 0;
        if (local_try_parse_int(config.at("output.write_cadence_medium_s"), parsed)
            && parsed > 0)
        {
            out.write_cadence_medium_s = parsed;
        }
    }

    // output.write_cadence_slow_s
    if (config.count("output.write_cadence_slow_s"))
    {
        int parsed = 0;
        if (local_try_parse_int(config.at("output.write_cadence_slow_s"), parsed)
            && parsed > 0)
        {
            out.write_cadence_slow_s = parsed;
        }
    }

    // output.estimate_disk_budget
    if (config.count("output.estimate_disk_budget"))
    {
        out.estimate_disk_budget =
            local_parse_bool(config.at("output.estimate_disk_budget"));
    }

    return out;
}

const char* output_format_name(OutputFormat fmt)
{
    switch (fmt)
    {
    case OutputFormat::npy_3d:       return "npy_3d";
    case OutputFormat::npy_2d_slices: return "npy_2d_slices";
    case OutputFormat::zfp:          return "zfp";
    case OutputFormat::csv:          return "csv";
    }
    return "unknown";
}

const char* field_preset_name(FieldPreset preset)
{
    switch (preset)
    {
    case FieldPreset::all:          return "all";
    case FieldPreset::minimal:      return "minimal";
    case FieldPreset::presentation: return "presentation";
    case FieldPreset::core_only:    return "core_only";
    case FieldPreset::custom:       return "custom";
    }
    return "unknown";
}
