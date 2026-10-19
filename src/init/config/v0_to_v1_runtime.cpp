#include "v0_to_v1_internal.hpp"

#include "core/output/output_config.hpp"
#include "init/sounding/parametric_target_math.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tmv::init::config::detail
{

void V0ToV1Migrator::migrate_numerics()
{
    // time_stepping
    int subs;
    if (take_int("numerics.time_stepping.n_acoustic_substeps", subs))
    {
        config_.time.acoustic_substeps = subs;
        warn_rename("numerics.time_stepping.n_acoustic_substeps", "time.acoustic_substeps");
    }
    bool b;
    if (take_bool("numerics.time_stepping.adaptive_dt", b))
    {
        config_.time.adaptive_dt = b;
        warn_rename("numerics.time_stepping.adaptive_dt", "time.adaptive_dt");
    }
    double cfl;
    if (take_double("numerics.time_stepping.cfl_safety", cfl))
    {
        config_.time.cfl_safety = cfl;
        warn_rename("numerics.time_stepping.cfl_safety", "time.cfl_safety");
    }
    const char* ts_dropped[] = {
        "numerics.time_stepping.scheme",
        "numerics.time_stepping.split_acoustic",
        "numerics.time_stepping.physics_splitting",
        "numerics.time_stepping.dt_min",
        "numerics.time_stepping.dt_max",
    };
    for (const char* k : ts_dropped)
    {
        if (has(k))
            warn_drop(k, "v1 dynamics core has one solver; knob not exposed in §20");
    }

    // diffusion
    std::string ds;
    if (take_string("numerics.diffusion.scheme", ds))
    {
        DiffusionScheme d = DiffusionScheme::Explicit;
        if (parse_diffusion_scheme(ds, d))
        {
            if (d == DiffusionScheme::None)
            {
                error(line_of("numerics.diffusion.scheme"),
                      "numerics.diffusion.scheme",
                      "no registered runtime diffusion scheme exists for "
                      "'none'; "
                      "use 'explicit' with K_h=0 and K_v=0 to disable "
                      "numerical diffusion");
            }
            else
            {
                config_.diffusion.scheme = d;
                warn_rename("numerics.diffusion.scheme", "diffusion.scheme");
            }
        }
        else
        {
            error(line_of("numerics.diffusion.scheme"),
                  "numerics.diffusion.scheme",
                  "unknown diffusion scheme '" + ds + "'");
        }
    }
    int operator_order = 2;
    if (take_int("numerics.diffusion.operator_order", operator_order))
    {
        if (operator_order != 2)
        {
            error(line_of("numerics.diffusion.operator_order"),
                  "numerics.diffusion.operator_order",
                  "only the implemented runtime order 2 is supported; got " + std::to_string(operator_order));
        }
        else
        {
            config_.diffusion.operator_order = operator_order;
            warn_rename("numerics.diffusion.operator_order", "diffusion.operator_order");
        }
    }
    double dv;
    if (take_double("numerics.diffusion.K_h", dv))
    {
        if (!std::isfinite(dv) || dv < 0.0)
        {
            error(line_of("numerics.diffusion.K_h"), "numerics.diffusion.K_h", "must be a non-negative finite number");
        }
        else
        {
            config_.diffusion.K_h = dv;
            warn_rename("numerics.diffusion.K_h", "diffusion.K_h");
        }
    }
    if (take_double("numerics.diffusion.K_v", dv))
    {
        config_.diffusion.K_v = dv;
        warn_rename("numerics.diffusion.K_v", "diffusion.K_v");
    }
    std::string apply;
    if (take_string("numerics.diffusion.apply_to", apply))
    {
        DiffusionTarget t = DiffusionTarget::Scalars;
        if (parse_diffusion_target(apply, t))
        {
            config_.diffusion.apply_to = t;
            warn_rename("numerics.diffusion.apply_to", "diffusion.apply_to");
        }
        else
        {
            error(line_of("numerics.diffusion.apply_to"),
                  "numerics.diffusion.apply_to",
                  "unknown diffusion target '" + apply + "'");
        }
    }
    if (has("numerics.diffusion.use_variable_K"))
    {
        consume_silent("numerics.diffusion.use_variable_K");
        bool requested = false;
        if (try_parse_bool_value(yaml_["numerics.diffusion.use_variable_K"].value, requested))
        {
            if (requested)
                error(
                    line_of("numerics.diffusion.use_variable_K"),
                    "numerics.diffusion.use_variable_K",
                    "variable diffusion coefficients are unavailable because runtime does not bind coefficient fields");
            else
                warn_drop("numerics.diffusion.use_variable_K",
                          "disabled legacy knob; v1 has no variable-coefficient diffusion path");
        }
        else
        {
            error(line_of("numerics.diffusion.use_variable_K"),
                  "numerics.diffusion.use_variable_K",
                  "expected a boolean");
        }
    }
    const char* dif_dropped[] = {
        "numerics.diffusion.operator_type",
        "numerics.diffusion.implicit_dim",
        "numerics.diffusion.dt_diffusion",
    };
    for (const char* k : dif_dropped)
    {
        if (has(k))
            warn_drop(k, "Nimbus-specific knob; v1 §20 exposes K_h, K_v, operator_order, apply_to only");
    }

    // compute backend
    std::string cb;
    if (take_string("numerics.compute.backend", cb))
    {
        ComputeBackend b1 = ComputeBackend::CPU;
        if (parse_compute_backend(cb, b1))
        {
            config_.extensions.compute.backend = b1;
            warn_rename("numerics.compute.backend", "extensions.compute.backend");
        }
        else
        {
            error(line_of("numerics.compute.backend"),
                  "numerics.compute.backend",
                  "unknown compute backend '" + cb + "'");
        }
    }
    std::string memory_admission;
    if (take_string("hardware.memory_admission", memory_admission))
    {
        MemoryAdmissionPolicy policy = MemoryAdmissionPolicy::Warn;
        if (parse_memory_admission_policy(memory_admission, policy))
        {
            config_.extensions.compute.memory_admission = policy;
            warn_rename("hardware.memory_admission", "extensions.compute.memory_admission");
        }
        else
        {
            error(line_of("hardware.memory_admission"), "hardware.memory_admission", "must be warn or reject");
        }
    }
    // validate_parity graduated from "dropped tuning knob" to a v1 key when
    // the parity harness landed (GPU.md Phase 1).
    {
        bool vp = false;
        if (take_bool("numerics.compute.validate_parity", vp))
        {
            config_.extensions.compute.validate_parity = vp;
            warn_rename("numerics.compute.validate_parity", "extensions.compute.validate_parity");
        }
    }
    {
        bool allow_fallback = true;
        if (take_bool("numerics.compute.allow_fallback", allow_fallback))
        {
            config_.extensions.compute.allow_fallback = allow_fallback;
            warn_rename("numerics.compute.allow_fallback", "extensions.compute.allow_fallback");
        }
    }
    {
        bool enabled = false;
        if (take_bool("numerics.compute.cuda_assist", enabled))
        {
            config_.extensions.compute.cuda_assist = enabled;
            warn_rename("numerics.compute.cuda_assist", "extensions.compute.cuda_assist");
        }
    }
    const char* compute_dropped[] = {
        "numerics.compute.device_index",
        "numerics.compute.templates.vertical_flux",
    };
    for (const char* k : compute_dropped)
    {
        if (has(k))
            warn_drop(k, "GPU-dispatch tuning knob; not part of v1 §20");
    }

    // advection
    std::string advection;
    if (take_string("numerics.advection.scheme", advection))
    {
        if (parse_advection_scheme(advection, config_.advection.scheme)
            && config_.advection.scheme != AdvectionScheme::Auto)
        {
            warn_rename("numerics.advection.scheme", "advection.scheme");
        }
        else
        {
            error(line_of("numerics.advection.scheme"),
                  "numerics.advection.scheme",
                  "unknown advection scheme '" + advection + "'");
        }
    }
}

// ─────────────────────────────────────────────────────────────
// terrain.* — preserve the now-explicit strict-v1 terrain contract
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_terrain()
{
    double ztop;
    if (take_double("terrain.ztop", ztop))
    {
        config_.grid.ztop = ztop;
        warn_rename("terrain.ztop", "grid.ztop");
    }
    std::string scheme;
    if (take_string("terrain.scheme", scheme))
    {
        if (!parse_terrain_scheme(scheme, config_.terrain.scheme))
            error(line_of("terrain.scheme"),
                  "terrain.scheme",
                  "unknown terrain scheme '" + scheme + "'; expected none, bell, or schar");
    }
    std::string coordinate;
    if (take_string("terrain.coord_id", coordinate))
    {
        if (!parse_terrain_coordinate(coordinate, config_.terrain.coordinate))
            error(line_of("terrain.coord_id"),
                  "terrain.coord_id",
                  "unknown terrain coordinate '" + coordinate + "'; expected btf, smoothed, or cartesian");
        else
            warn_rename("terrain.coord_id", "terrain.coordinate");
    }
    take_bool("terrain.compute_derivatives", config_.terrain.compute_derivatives);
    take_bool("terrain.compute_metrics", config_.terrain.compute_metrics);
    take_double("terrain.smoothing_decay", config_.terrain.smoothing_decay);
    take_double("terrain.bell.h0", config_.terrain.bell.h0);
    take_double("terrain.bell.a", config_.terrain.bell.a);
    take_bool("terrain.bell.axisymmetric", config_.terrain.bell.axisymmetric);
    take_double("terrain.schar.H", config_.terrain.schar.H);
    take_double("terrain.schar.a", config_.terrain.schar.a);
    take_double("terrain.schar.ell", config_.terrain.schar.ell);

    const auto require_non_negative = [this](const char* key, double value)
    {
        if (!std::isfinite(value) || value < 0.0)
            error(line_of(key), key, "must be a non-negative finite number");
    };
    const auto require_positive = [this](const char* key, double value)
    {
        if (!std::isfinite(value) || value <= 0.0)
            error(line_of(key), key, "must be a positive finite number");
    };
    require_non_negative("terrain.smoothing_decay", config_.terrain.smoothing_decay);
    require_non_negative("terrain.bell.h0", config_.terrain.bell.h0);
    require_positive("terrain.bell.a", config_.terrain.bell.a);
    require_non_negative("terrain.schar.H", config_.terrain.schar.H);
    require_positive("terrain.schar.a", config_.terrain.schar.a);
    require_positive("terrain.schar.ell", config_.terrain.schar.ell);

    if (config_.terrain.scheme != TerrainScheme::None)
    {
        if (!config_.terrain.compute_derivatives)
            error(line_of("terrain.compute_derivatives"),
                  "terrain.compute_derivatives",
                  "active terrain requires derivative fields for metric-consistent dynamics");
        if (!config_.terrain.compute_metrics)
            error(line_of("terrain.compute_metrics"),
                  "terrain.compute_metrics",
                  "active terrain requires terrain-following metrics");
        if (config_.terrain.coordinate == TerrainCoordinate::Cartesian)
            error(
                line_of("terrain.coord_id"), "terrain.coord_id", "active terrain requires btf or smoothed coordinates");

        const double computational_top = static_cast<double>(config_.grid.nz) * config_.grid.dz;
        const double scale             = std::max({1.0, std::abs(computational_top), std::abs(config_.grid.ztop)});
        const double tolerance         = 64.0 * std::numeric_limits<double>::epsilon() * scale;
        if (std::abs(config_.grid.ztop - computational_top) > tolerance)
            error(line_of("terrain.ztop"), "terrain.ztop", "active terrain requires terrain.ztop == grid.nz * grid.dz");

        const bool   bell           = config_.terrain.scheme == TerrainScheme::Bell;
        const double maximum_height = bell ? config_.terrain.bell.h0 : config_.terrain.schar.H;
        if (maximum_height >= config_.grid.ztop)
            error(line_of(bell ? "terrain.bell.h0" : "terrain.schar.H"),
                  bell ? "terrain.bell.h0" : "terrain.schar.H",
                  "terrain height must remain below terrain.ztop");
    }
}

// ─────────────────────────────────────────────────────────────
// output.* — preset/format/IO + ZFP tuning + tiered cadence kept;
// estimate_disk_budget / downsample dropped
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_output()
{
    double oi;
    if (take_double("output.write_every_s", oi))
    {
        if (!valid_runtime_seconds(oi, true))
        {
            error(line_of("output.write_every_s"),
                  "output.write_every_s",
                  "must be a non-negative whole number of seconds within the runtime integer range");
        }
        else
        {
            config_.time.output_interval = oi;
            warn_rename("output.write_every_s", "time.output_interval");
        }
    }
    std::string fmt;
    if (take_string("output.format", fmt))
    {
        OutputFormat f = OutputFormat::Npy3D;
        if (parse_output_format(fmt, f))
        {
            if (f == OutputFormat::Netcdf)
            {
#ifndef HAVE_NETCDF
                error(line_of("output.format"),
                      "output.format",
                      "netcdf output is unavailable in this binary; rebuild with libnetcdf or use npy_3d");
#else
                config_.output.format = f;
#endif
            }
            else
            {
                config_.output.format = f;
            }
        }
        else
        {
            error(line_of("output.format"), "output.format", "unknown output format '" + fmt + "'");
        }
    }
    std::string fields;
    if (take_string("output.fields", fields))
    {
        OutputPreset p = OutputPreset::Minimal;
        if (parse_output_preset(fields, p))
        {
            config_.output.preset = p;
            warn_rename("output.fields", "output.preset");
        }
        else
        {
            // Some configs use "all", "custom", etc. — best-guess to minimal.
            warn_synth("output.fields",
                       "best-guess: output.preset=minimal",
                       "v0 value '" + fields +
                           "' has no v1 preset equivalent; "
                           "defaulted to minimal — set explicitly if a richer preset is needed");
            config_.output.preset = OutputPreset::Minimal;
        }
    }
    bool b;
    if (take_bool("output.async_io", b))
        config_.output.async_io = b;
    if (take_bool("output.live_shm", b))
        config_.output.live_shm = b;
    std::string live_shm_fields;
    if (take_string("output.live_shm_fields", live_shm_fields))
    {
        config_.output.live_shm_fields = parse_string_list(live_shm_fields);
        std::string field_error;
        if (!validate_live_shm_fields(config_.output.live_shm_fields,
                                      field_error))
            error(line_of("output.live_shm_fields"),
                  "output.live_shm_fields", field_error);
    }
    double live_shm_interval_s = 0.0;
    if (take_double("output.live_shm_interval_s", live_shm_interval_s))
    {
        if (std::isfinite(live_shm_interval_s) && live_shm_interval_s > 0.0)
            config_.output.live_shm_interval_s = live_shm_interval_s;
        else
            error(line_of("output.live_shm_interval_s"),
                  "output.live_shm_interval_s",
                  "must be a positive finite number");
    }
    if (take_bool("output.live_shm_full_rate", b))
        config_.output.live_shm_full_rate = b;
    if (take_bool("output.trajectories", b))
        config_.output.trajectories = b;
    double trajectory_seed_time_s = 0.0;
    if (take_double("output.trajectory_seed_time_s", trajectory_seed_time_s))
    {
        if (std::isfinite(trajectory_seed_time_s) && trajectory_seed_time_s >= 0.0)
            config_.output.trajectory_seed_time_s = trajectory_seed_time_s;
        else
            error(line_of("output.trajectory_seed_time_s"),
                  "output.trajectory_seed_time_s",
                  "must be a non-negative finite number");
    }
    int trajectory_count = 0;
    if (take_int("output.trajectory_count", trajectory_count))
    {
        if (trajectory_count > 0)
            config_.output.trajectory_count = trajectory_count;
        else
            error(line_of("output.trajectory_count"), "output.trajectory_count", "must be a positive integer");
    }
    double trajectory_sample_interval_s = 0.0;
    if (take_double("output.trajectory_sample_interval_s",
                    trajectory_sample_interval_s))
    {
        if (std::isfinite(trajectory_sample_interval_s) &&
            trajectory_sample_interval_s >= 0.0)
            config_.output.trajectory_sample_interval_s =
                trajectory_sample_interval_s;
        else
            error(line_of("output.trajectory_sample_interval_s"),
                  "output.trajectory_sample_interval_s",
                  "must be a non-negative finite number");
    }
    int trajectory_max_samples_per_parcel = 0;
    if (take_int("output.trajectory_max_samples_per_parcel",
                 trajectory_max_samples_per_parcel))
    {
        if (trajectory_max_samples_per_parcel > 0)
            config_.output.trajectory_max_samples_per_parcel =
                trajectory_max_samples_per_parcel;
        else
            error(line_of("output.trajectory_max_samples_per_parcel"),
                  "output.trajectory_max_samples_per_parcel",
                  "must be a positive integer");
    }

    int n;
    if (take_int("output.netcdf_deflate_level", n))
    {
        if (n < 0 || n > 9)
            error(line_of("output.netcdf_deflate_level"), "output.netcdf_deflate_level", "must be in [0, 9]");
        else
            config_.output.netcdf_deflate_level = n;
    }
    if (take_bool("output.netcdf_shuffle", b))
        config_.output.netcdf_shuffle = b;
    if (take_int("output.netcdf_target_chunk_kib", n))
    {
        if (n <= 0)
            error(line_of("output.netcdf_target_chunk_kib"), "output.netcdf_target_chunk_kib", "must be positive");
        else
            config_.output.netcdf_target_chunk_kib = n;
    }

    // ZFP tuning + tiered cadence: preserved into v1 (previously dropped).
    // These mirror the keys parse_output_config reads on the writer side.
    double      d;
    std::string s;
    if (take_double("output.zfp_tolerance", d))
        config_.output.zfp_tolerance = d;
    if (take_string("output.zfp_mode", s))
        config_.output.zfp_mode = s;
    if (take_int("output.zfp_precision_bits", n))
        config_.output.zfp_precision_bits = n;
    if (take_int("output.zfp_rate_bps", n))
        config_.output.zfp_rate_bps = n;
    if (take_int("output.zfp_keyframe_interval", n))
        config_.output.zfp_keyframe_interval = n;
    if (take_double("output.zfp_sparse_threshold", d))
        config_.output.zfp_sparse_threshold = d;
    if (take_bool("output.zfp_float16_prefilter", b))
        config_.output.zfp_float16_prefilter = b;
    if (take_bool("output.zfp_predictive_delta", b))
        config_.output.zfp_predictive_delta = b;
    if (take_string("output.zfp_field_execution", s))
    {
        if (s == "serial" || s == "field_parallel")
            config_.output.zfp_field_execution = s;
        else
            error(line_of("output.zfp_field_execution"),
                  "output.zfp_field_execution",
                  "must be serial or field_parallel");
    }
    if (take_int("output.zfp_field_workers", n))
    {
        if (n >= 0 && n <= 16)
            config_.output.zfp_field_workers = n;
        else
            error(line_of("output.zfp_field_workers"),
                  "output.zfp_field_workers", "must be in [0, 16]");
    }
    if (take_int("output.zfp_field_scratch_mib", n))
    {
        if (n > 0 && n <= 4096)
            config_.output.zfp_field_scratch_mib = n;
        else
            error(line_of("output.zfp_field_scratch_mib"),
                  "output.zfp_field_scratch_mib", "must be in [1, 4096]");
    }
    if (take_bool("output.zfp_per_field_tolerances", b))
        config_.output.zfp_per_field_tolerances = b;
    if (take_double("output.zfp_tight_tolerance", d))
        config_.output.zfp_tight_tolerance = d;
    if (take_double("output.zfp_moderate_tolerance", d))
        config_.output.zfp_moderate_tolerance = d;
    if (take_double("output.zfp_loose_tolerance", d))
        config_.output.zfp_loose_tolerance = d;
    const std::string        field_tolerance_prefix = "output.zfp_field_tolerance.";
    std::vector<std::string> field_tolerance_keys;
    for (const auto& [key, field] : yaml_)
    {
        (void)field;
        if (key.rfind(field_tolerance_prefix, 0) == 0)
            field_tolerance_keys.push_back(key);
    }
    for (const std::string& key : field_tolerance_keys)
    {
        const std::string field = key.substr(field_tolerance_prefix.size());
        if (field.empty())
        {
            consumed_.insert(key);
            error(line_of(key), key, "field name must not be empty");
            continue;
        }
        double tolerance = 0.0;
        if (take_double(key, tolerance))
        {
            if (!std::isfinite(tolerance) || tolerance <= 0.0)
                error(line_of(key), key, "must be a positive finite number");
            else
                config_.output.zfp_field_tolerances[field] = tolerance;
        }
    }
    if (take_bool("output.tiered_write_cadence", b))
        config_.output.tiered_write_cadence = b;
    if (take_int("output.write_cadence_medium_s", n))
        config_.output.write_cadence_medium_s = n;
    if (take_int("output.write_cadence_slow_s", n))
        config_.output.write_cadence_slow_s = n;

    const char* dropped[] = {
        "output.estimate_disk_budget",
        "output.downsample",
    };
    for (const char* k : dropped)
    {
        if (has(k))
            warn_drop(k, "Nimbus-specific output knob; not part of v1 §20");
    }
}

// ─────────────────────────────────────────────────────────────
// chaos.* — scheme/seed/member_id move to extensions.chaos
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_chaos()
{
    std::string scheme;
    if (take_string("chaos.scheme", scheme))
    {
        ChaosScheme c = ChaosScheme::None;
        if (parse_chaos_scheme(scheme, c))
        {
            config_.extensions.chaos.scheme = c;
            warn_rename("chaos.scheme", "extensions.chaos.scheme");
        }
        else
        {
            error(line_of("chaos.scheme"), "chaos.scheme", "unknown chaos scheme '" + scheme + "'");
        }
    }
    int i;
    if (take_int("chaos.seed", i))
    {
        config_.extensions.chaos.seed = i;
        warn_rename("chaos.seed", "extensions.chaos.seed");
    }
    if (take_int("chaos.member_id", i))
    {
        config_.extensions.chaos.member_id = i;
        warn_rename("chaos.member_id", "extensions.chaos.member_id");
    }
    const char* dropped[] = {
        "chaos.dt_chaos",
        "chaos.tau_t",
        "chaos.Lx",
        "chaos.Ly",
        "chaos.filter_id",
        "chaos.xi_max",
        "chaos.taper_id",
        "chaos.taper_z1",
        "chaos.taper_z2",
        "chaos.apply_to_ic",
        "chaos.apply_to_tendencies",
    };
    for (const char* k : dropped)
    {
        if (has(k))
            warn_drop(k, "chaos-scheme-specific knob; v1 §20 exposes scheme/seed/member_id only");
    }
}

// ─────────────────────────────────────────────────────────────
// validation.* — guard_mode kept under extensions
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_validation()
{
    std::string gm;
    if (take_string("validation.guard_mode", gm))
    {
        ValidationGuardMode v = ValidationGuardMode::Warn;
        if (parse_validation_guard_mode(gm, v))
        {
            config_.extensions.validation.guard_mode = v;
            warn_rename("validation.guard_mode", "extensions.validation.guard_mode");
        }
        else
        {
            error(line_of("validation.guard_mode"), "validation.guard_mode", "unknown guard mode '" + gm + "'");
        }
    }
    const char* dropped[] = {
        "validation.guard_fail_on",
        "validation.guard_scope",
        "validation.report_path",
    };
    for (const char* k : dropped)
    {
        if (has(k))
            warn_drop(k, "Nimbus-specific validation knob; v1 §20 exposes guard_mode only");
    }
}

// ─────────────────────────────────────────────────────────────
// logging.* + runtime diagnostics/performance metadata
// ─────────────────────────────────────────────────────────────

void V0ToV1Migrator::migrate_metadata_sections()
{
    if (has("logging.profile"))
    {
        warn_drop("logging.profile", "operational logging metadata; not part of the v1 schema");
    }

    bool timing = false;
    if (take_bool("performance.timing", timing))
    {
        config_.extensions.performance.timing = timing;
        warn_rename("performance.timing", "extensions.performance.timing");
    }

    int report_every_steps = 0;
    if (take_int("performance.report_every_steps", report_every_steps))
    {
        if (report_every_steps < 0)
        {
            error(line_of("performance.report_every_steps"), "performance.report_every_steps", "must be >= 0");
        }
        else
        {
            config_.extensions.performance.report_every_steps = report_every_steps;
            warn_rename("performance.report_every_steps", "extensions.performance.report_every_steps");
        }
    }

    int warmup_steps = 0;
    if (take_int("performance.warmup_steps", warmup_steps))
    {
        if (warmup_steps < 0)
        {
            error(line_of("performance.warmup_steps"), "performance.warmup_steps", "must be >= 0");
        }
        else
        {
            config_.extensions.performance.warmup_steps = warmup_steps;
            warn_rename("performance.warmup_steps", "extensions.performance.warmup_steps");
        }
    }

    int scientific_every_steps = -1;
    if (take_int("diagnostics.scientific_every_steps", scientific_every_steps))
    {
        if (scientific_every_steps < 0)
        {
            error(line_of("diagnostics.scientific_every_steps"), "diagnostics.scientific_every_steps", "must be >= 0");
        }
        else
        {
            config_.extensions.diagnostics.scientific_every_steps = scientific_every_steps;
            warn_rename("diagnostics.scientific_every_steps", "extensions.diagnostics.scientific_every_steps");
        }
    }
    bool deterministic_reductions = false;
    if (take_bool("diagnostics.deterministic_reductions", deterministic_reductions))
    {
        config_.extensions.diagnostics.deterministic_reductions = deterministic_reductions;
        warn_rename("diagnostics.deterministic_reductions", "extensions.diagnostics.deterministic_reductions");
    }
}

// ─────────────────────────────────────────────────────────────

} // namespace tmv::init::config::detail
