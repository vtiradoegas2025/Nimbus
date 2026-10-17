// Process composition root: configuration, initialization, and run startup.

#include <vector>
#include <string>
#include <string_view>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <regex>
#include <cstdlib>
#include <cctype>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "core/runtime/process_state.hpp"
#include "core/runtime/qualified_agrid_core_profile.hpp"
#include "core/runtime/qualified_variable_diffusion_profile.hpp"
#include "core/runtime/termination_signal.hpp"
#include "init/config/v0_to_v1.hpp"
#include "init/config/config_v1_serialization_common.hpp"
#include "init/config/qualified_terrain_profile.hpp"
#include "init/config/v1_loader.hpp"
#include "init/hodograph/factory.hpp"
#include "init/sounding/factory.hpp"
#include "init/trigger/factory.hpp"
#include "boundary_conditions/runtime.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "chaos/runtime.hpp"
#include "compute/compute_backend.hpp"
#include "init/scheme_profile.hpp"
#include "util/log.hpp"
#include "radiation/runtime.hpp"
#include "terrain/runtime.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "turbulence/runtime.hpp"
#include "core/runtime/headless_runtime.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/generated_metal_output_scope_v2.hpp"
#include "core/runtime/qualified_generated_metal_cli_scope.hpp"
#include "core/runtime/validate_only_context.hpp"
#include "core/output/netcdf_history_writer.hpp"
#include "core/runtime/scheme_catalog.hpp"
#include "core/runtime/science_profile.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/runtime/yaml_helpers.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/domain_motion.hpp"
#include "core/infra/hardware_info.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/field/numerical_integrity.hpp"
#include "core/field/field_pool.hpp"
#include "core/field/field_bounds.hpp"
#include "core/orchestration/dynamics/thermodynamic_closure.hpp"
#include "numerics/advection/advection.hpp"
#include "diagnostics/field_contract.hpp"
#include "diagnostics/field_validation.hpp"
#include "dynamics/dynamics_base.hpp"
#include "boundary_layer/scheme_descriptor.hpp"
#include "chaos/scheme_descriptor.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "microphysics/base/thermodynamics.hpp"
#include "microphysics/scheme_descriptor.hpp"
#include "numerics/advection/scheme_descriptor.hpp"
#include "numerics/diffusion/scheme_descriptor.hpp"
#include "numerics/time_stepping/scheme_descriptor.hpp"
#include "radiation/scheme_descriptor.hpp"
#include "radar/scheme_descriptor.hpp"
#include "terrain/scheme_descriptor.hpp"
#include "turbulence/scheme_descriptor.hpp"
#include "soundings/runtime.hpp"
#include "util/string_utils.hpp"

#include "radar/runtime.hpp"

/**
 * @brief Computes the wind profile.
 */

void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v)
{
    const double z_sfc = 0.0;
    const double z_1km = 1000.0;
    const double z_6km = 6000.0;

    if (z <= z_1km)
    {
        double frac = (z - z_sfc) / (z_1km - z_sfc);
        u           = profile.u_sfc + frac * (profile.u_1km - profile.u_sfc);
        v           = profile.v_sfc + frac * (profile.v_1km - profile.v_sfc);
    }

    else if (z <= z_6km)
    {
        double frac = (z - z_1km) / (z_6km - z_1km);
        u           = profile.u_1km + frac * (profile.u_6km - profile.u_1km);
        v           = profile.v_1km + frac * (profile.v_6km - profile.v_1km);
    }
    else
    {
        u = profile.u_6km;
        v = profile.v_6km;
    }
}

namespace
{

struct RuntimeCliOptions
{
    bool                     headless                             = false;
    bool                     validate_only                        = false;
    bool                     preflight_context                    = false;
    int                      export_ms                            = 0;
    int                      duration_s                           = -1;
    int                      write_every_s                        = 0;
    bool                     duration_from_cli                    = false;
    bool                     write_every_from_cli                 = false;
    bool                     threads_from_cli                     = false;
    int                      cli_threads                          = 0;
    bool                     acoustic_substeps_from_cli           = false;
    int                      cli_acoustic_substeps                = 0;
    bool                     dt_large_from_cli                    = false;
    double                   cli_dt_large_s                       = 0.0;
    bool                     qualification_hydrometeor_pulse      = false;
    bool                     qualification_enable_milbrandt       = false;
    bool                     qualification_milbrandt_mixed_phase  = false;
    bool                     qualification_acoustic_wave          = false;
    bool                     qualification_acoustic_standing_wave = false;
    bool                     qualification_acoustic_pulse         = false;
    bool                     qualification_moving_nest            = false;
    int                      cli_duration_s                       = -1;
    int                      cli_write_every_s                    = 0;
    bool                     traj_enabled                         = false;
    bool                     traj_enabled_from_cli                = false;
    double                   traj_seed_time_s                     = 0.0;
    bool                     traj_seed_time_from_cli              = false;
    int                      traj_count                           = 5;
    bool                     traj_count_from_cli                  = false;
    bool                     log_profile_from_cli                 = false;
    LogProfile               cli_log_profile                      = LogProfile::normal;
    bool                     timing_from_cli                      = false;
    bool                     cli_perf_timing_enabled              = false;
    int                      cli_perf_report_every_steps          = -1;
    bool                     guard_mode_from_cli                  = false;
    tmv::GuardMode           cli_guard_mode                       = global_validation_policy.mode;
    bool                     guard_fail_on_from_cli               = false;
    tmv::GuardFailOn         cli_guard_fail_on                    = global_validation_policy.fail_on;
    bool                     guard_scope_from_cli                 = false;
    tmv::StrictGuardScope    cli_guard_scope                      = global_validation_policy.strict_scope;
    bool                     guard_report_from_cli                = false;
    std::string              cli_guard_report_path;
    bool                     compute_backend_from_cli = false;
    std::string              cli_compute_backend;
    bool                     live_shm_from_cli        = false;
    bool                     cli_live_shm             = false;
    bool                     live_shm_fields_from_cli = false;
    std::vector<std::string> cli_live_shm_fields;
    bool                     live_shm_name_from_cli = false;
    std::string              cli_live_shm_name = tmv_shm::kDefaultShmName;
    std::string              outdir                    = "data/exports";
    int                      checkpoint_every_s        = 0;
    bool                     checkpoint_every_from_cli = false;
    std::filesystem::path    checkpoint_dir;
    bool                     checkpoint_dir_from_cli = false;
    std::filesystem::path    resume_checkpoint;
    bool                     resume_checkpoint_from_cli = false;
    std::string              config_path;
    std::string              cli_sounding_path;
    bool                     sounding_path_from_cli = false;
};

struct RuntimeRunOptions
{
    RuntimeCliOptions        cli;
    bool                     live_shm        = false;
    std::vector<std::string> live_shm_fields = {"w", "reflectivity_dbz", "vorticity_z"};
    std::string              live_shm_name = tmv_shm::kDefaultShmName;
    OutputConfig             output_config;
};

std::string qualified_cli_scope_error(const RuntimeCliOptions& cli,
                                      bool                     qualified,
                                      bool                     generated_metal,
                                      bool                     generated_metal_east_npy,
                                      bool                     generated_metal_mixed_npy,
                                      bool                     radar_zfp_v4_requested,
                                      std::string_view         output_format)
{
    if (!qualified)
    {
        return {};
    }
    if (cli.guard_mode_from_cli && cli.cli_guard_mode != tmv::GuardMode::Strict)
    {
        return "Qualified execution requires --guard-mode=strict; a CLI override cannot weaken the configured science guard.";
    }
    if (cli.guard_fail_on_from_cli && cli.cli_guard_fail_on != tmv::GuardFailOn::Both)
    {
        return "Qualified execution requires --guard-fail-on=both; a CLI override cannot weaken the configured science guard.";
    }
    if (radar_zfp_v4_requested)
    {
        const nimbus::output::QualifiedRadarZfpV4CliOverrides overrides{
            cli.dt_large_from_cli, cli.cli_dt_large_s,
            cli.acoustic_substeps_from_cli, cli.cli_acoustic_substeps,
            cli.write_every_from_cli, cli.cli_write_every_s,
            cli.live_shm_from_cli || cli.live_shm_fields_from_cli ||
                cli.live_shm_name_from_cli,
            cli.traj_enabled_from_cli || cli.traj_seed_time_from_cli ||
                cli.traj_count_from_cli,
            cli.guard_scope_from_cli &&
                cli.cli_guard_scope != tmv::StrictGuardScope::RequiredOnly,
            cli.threads_from_cli, cli.cli_threads};
        const auto error = nimbus::output::qualified_radar_zfp_v4_cli_scope_error(overrides);
        if (!error.empty())
        {
            return std::string(error);
        }
    }
    if (!generated_metal)
    {
        return {};
    }
    const tmv::runtime::QualifiedGeneratedMetalCliOverrides overrides{
        cli.dt_large_from_cli, cli.cli_dt_large_s,
        cli.acoustic_substeps_from_cli, cli.cli_acoustic_substeps,
        cli.write_every_from_cli, cli.cli_write_every_s,
        cli.live_shm_from_cli || cli.live_shm_fields_from_cli || cli.live_shm_name_from_cli,
        cli.traj_enabled_from_cli || cli.traj_seed_time_from_cli || cli.traj_count_from_cli,
        cli.guard_mode_from_cli, cli.cli_guard_mode,
        cli.guard_fail_on_from_cli, cli.cli_guard_fail_on,
        cli.guard_scope_from_cli, cli.cli_guard_scope};
    return std::string(tmv::runtime::qualified_generated_metal_cli_scope_error(
        overrides, generated_metal_east_npy, generated_metal_mixed_npy, output_format));
}

struct MigrationCommandResult
{
    bool handled   = false;
    int  exit_code = 0;
};

constexpr const char* kRuntimeUsage = R"(Usage:
  nimbus_sim --config PATH [options]
  nimbus_sim --sounding PATH [options]
  nimbus_sim --migrate-config PATH
  nimbus_sim --scheme-catalog=json
  nimbus_sim --compute-ownership-matrix=json
  nimbus_sim --compute-process-ownership-matrix=json

Required input:
  --config PATH                 Simulation YAML. Required unless --sounding is used.
  --sounding PATH               Override the sounding; uses the public 1000 m Kessler preset as base.

Execution:
  --validate-only                Strictly load/project the config and build its
                                 1-D initial-condition sounding; allocate no
                                 simulation fields and run no timestep.
  --preflight-context=json        Emit a bounded native context receipt for
                                 --validate-only without science overrides.
  --headless                    Run without the Studio UI.
  --duration SECONDS            Override configured run duration.
  --write-every=SECONDS         Override output cadence (0 disables snapshots).
  --outdir=PATH                 Snapshot directory (default: data/exports).
  --checkpoint-every=SECONDS    Portable checkpoint cadence (0 disables).
  --checkpoint-dir=PATH         Checkpoint directory (default: OUTDIR/checkpoints).
  --resume=PATH                 Continue from a verified Nimbus checkpoint into a new OUTDIR.
  --compute-backend KIND        cpu, vulkan, gpu, or metal.
  --threads COUNT               OpenMP worker count; overrides OMP_NUM_THREADS.
  --acoustic-substeps COUNT     Override split-explicit acoustic substep count.
  --dt-large SECONDS            Override the positive large timestep.
  --qualification-hydrometeor-pulse
                                 Internal A-grid qualification binary only.
  --qualification-enable-milbrandt
                                 Internal qualification binary only; admits
                                 the maintained Milbrandt CPU C-grid campaign.
  --qualification-milbrandt-mixed-phase
                                 Same admission plus a mixed-phase seed.
  --qualification-acoustic-wave Internal A-grid qualification binary only.
  --qualification-acoustic-standing-wave
                                 Internal A-grid qualification binary only.
  --qualification-acoustic-pulse
                                 Internal A-grid qualification binary only.
  --qualification-moving-nest  Internal qualification binary only; admits the
                                 exact moving-supercell campaign profile.
  --export-ms=MILLISECONDS      Delay between live exports.

Diagnostics and output:
  --log-profile PROFILE         quiet, normal, or debug.
  --timing | --no-timing        Enable or disable performance timing.
  --timing-every=STEPS          Timing report cadence.
  --guard-mode MODE             off, sanitize, or strict.
  --guard-fail-on CONDITION     nonfinite, bounds, or both.
  --guard-scope SCOPE           required or exported.
  --guard-report PATH           Validation report path.
  --live-shm                    Publish fields to Studio shared memory.
  --live-shm-fields=LIST        Comma-separated shared-memory fields.
  --live-shm-name=NAME          Logical endpoint (default: /tmv_live); use an
                                 explicit name to isolate concurrent runs.

Trajectories:
  --trajectories
  --trajectory-seed-time=SECONDS
  --trajectory-count=COUNT

Other:
  --migrate-config PATH         Write a migrated nimbus.v1 config to stdout.
  --scheme-catalog=json         Write the exact compiled scheme catalog as JSON.
  --compute-ownership-matrix=json
                                 Write the native state-commit ownership matrix as JSON.
  --compute-process-ownership-matrix=json
                                 Write the native solver-process ownership matrix as JSON.
  -h, --help                    Show this help and exit.
  --version                     Show build/schema identity and exit.
)";

MigrationCommandResult maybe_handle_cli_metadata(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            std::fputs(kRuntimeUsage, stdout);
            return {true, 0};
        }
        if (arg == "--version")
        {
#if defined(NIMBUS_PORTABLE_BUILD)
            constexpr const char* build_profile = "portable";
#elif defined(NIMBUS_CPU_ONLY)
            constexpr const char* build_profile = "cpu-native";
#else
            constexpr const char* build_profile = "native";
#endif
            std::fprintf(
                stdout, "Nimbus simulator (development build; profile %s; config schema nimbus.v1)\n", build_profile);
            const NetcdfHistoryCapability netcdf = netcdf_history_capability();
            if (netcdf.available)
            {
                std::fprintf(stdout, "netCDF4 history output: available (%s)\n", netcdf.implementation.c_str());
            }
            else
            {
                std::fprintf(stdout, "netCDF4 history output: unavailable: %s\n", netcdf.reason.c_str());
            }
#ifdef _OPENMP
            std::fputs("OpenMP runtime: available\n", stdout);
#else
            std::fputs("OpenMP runtime: unavailable\n", stdout);
#endif
#ifdef HAVE_ZFP
            std::fputs("ZFP compression: available\n", stdout);
#else
            std::fputs("ZFP compression: unavailable\n", stdout);
#endif
#ifdef NIMBUS_CPU_ONLY
            std::fputs("Vulkan compute: unavailable\n", stdout);
#else
            std::fputs("Vulkan compute: available\n", stdout);
#endif
#if defined(_WIN32) || defined(__APPLE__) || defined(__linux__)
            std::fputs("live shared memory: available\n", stdout);
#else
            std::fputs("live shared memory: unavailable\n", stdout);
#endif
            return {true, 0};
        }
        if (arg == "--scheme-catalog=json")
        {
            const std::string catalog = tmv::runtime::serialize_scheme_catalog_json();
            std::fputs(catalog.c_str(), stdout);
            return {true, 0};
        }
        if (arg == "--compute-ownership-matrix=json")
        {
            const std::string matrix = tmv::runtime::serialize_compute_ownership_matrix_json();
            std::fputs(matrix.c_str(), stdout);
            return {true, 0};
        }
        if (arg == "--compute-process-ownership-matrix=json")
        {
            const std::string matrix = tmv::runtime::serialize_compute_process_ownership_matrix_json();
            std::fputs(matrix.c_str(), stdout);
            return {true, 0};
        }
    }
    return {};
}

void report_cli_error(const std::string& message)
{
    std::fputs(message.c_str(), stderr);
    std::fputs("\nTry 'nimbus_sim --help' for usage.\n", stderr);
}

MigrationCommandResult maybe_run_config_migration(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        std::string       v0_path;
        if (a.rfind("--migrate-config=", 0) == 0)
        {
            v0_path = a.substr(17);
        }
        else if (a == "--migrate-config" && i + 1 < argc)
        {
            v0_path = argv[i + 1];
        }
        if (v0_path.empty())
            continue;

        auto result = tmv::init::config::migrate_v0_to_v1(v0_path);
        if (!result.warnings.empty())
        {
            std::cerr << "# Migration warnings (" << result.warnings.size() << "):\n";
            std::cerr << tmv::init::config::format_warnings(v0_path, result.warnings) << "\n";
        }
        if (!result.ok())
        {
            std::cerr << "# Migration FAILED — " << result.errors.size() << " error(s):\n";
            for (const auto& e : result.errors)
            {
                std::cerr << "  " << e.file;
                if (e.line > 0)
                    std::cerr << ":" << e.line;
                std::cerr << ": ";
                if (!e.key.empty())
                    std::cerr << e.key << ": ";
                std::cerr << e.message << "\n";
            }
            return {true, 2};
        }
        std::cout << tmv::init::config::serialize_v1_to_yaml(*result.config);
        return {true, 0};
    }

    return {};
}

void apply_environment_overrides()
{
    if (const char* env_log_profile = std::getenv("NIMBUS_LOG_PROFILE"))
    {
        bool             valid  = false;
        const LogProfile parsed = parse_log_profile(env_log_profile, &valid);
        if (valid)
        {
            global_log_profile = parsed;
        }
        else
        {
            tmv::log_warn("Invalid NIMBUS_LOG_PROFILE '", env_log_profile, "'. Valid values: quiet, normal, debug.");
        }
    }
    if (const char* env_perf = std::getenv("NIMBUS_PERF_TIMING"))
    {
        global_perf_timing_enabled = parse_bool_value(env_perf);
    }
    if (const char* env_perf_every = std::getenv("NIMBUS_PERF_EVERY_STEPS"))
    {
        int parsed = 0;
        if (try_parse_non_negative_int_value(env_perf_every, parsed))
        {
            global_perf_report_every_steps = parsed;
        }
        else
        {
            tmv::log_warn("Invalid NIMBUS_PERF_EVERY_STEPS '", env_perf_every, "'. Expected a non-negative integer.");
        }
    }
}

bool parse_runtime_cli(int argc, char** argv, RuntimeCliOptions& cli)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--headless")
            cli.headless = true;
        else if (arg == "--validate-only")
            cli.validate_only = true;
        else if (arg == "--preflight-context=json")
            cli.preflight_context = true;
        else if (arg.rfind("--export-ms=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(12);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                report_cli_error("Invalid --export-ms value '" + value + "'. Expected a non-negative integer.");
                return false;
            }
            cli.export_ms = parsed;
        }
        else if (arg.rfind("--duration=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(11);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                report_cli_error("Invalid --duration value '" + value + "'. Expected a non-negative integer.");
                return false;
            }
            cli.duration_s        = parsed;
            cli.duration_from_cli = true;
            cli.cli_duration_s    = cli.duration_s;
        }
        else if (arg == "--duration" && i + 1 < argc)
        {
            int               parsed = 0;
            const std::string value  = argv[++i];
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                report_cli_error("Invalid --duration value '" + value + "'. Expected a non-negative integer.");
                return false;
            }
            cli.duration_s        = parsed;
            cli.duration_from_cli = true;
            cli.cli_duration_s    = cli.duration_s;
        }
        else if (arg.rfind("--threads=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(10);
            if (!try_parse_int_value(value, parsed) || parsed <= 0)
            {
                report_cli_error("Invalid --threads value '" + value + "'. Expected a positive integer.");
                return false;
            }
            cli.threads_from_cli = true;
            cli.cli_threads      = parsed;
        }
        else if (arg == "--threads" && i + 1 < argc)
        {
            int               parsed = 0;
            const std::string value  = argv[++i];
            if (!try_parse_int_value(value, parsed) || parsed <= 0)
            {
                report_cli_error("Invalid --threads value '" + value + "'. Expected a positive integer.");
                return false;
            }
            cli.threads_from_cli = true;
            cli.cli_threads      = parsed;
        }
        else if (arg.rfind("--acoustic-substeps=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(20);
            if (!try_parse_int_value(value, parsed) || parsed <= 0)
            {
                report_cli_error("Invalid --acoustic-substeps value '" + value + "'. Expected a positive integer.");
                return false;
            }
            cli.acoustic_substeps_from_cli = true;
            cli.cli_acoustic_substeps      = parsed;
        }
        else if (arg == "--acoustic-substeps" && i + 1 < argc)
        {
            int               parsed = 0;
            const std::string value  = argv[++i];
            if (!try_parse_int_value(value, parsed) || parsed <= 0)
            {
                report_cli_error("Invalid --acoustic-substeps value '" + value + "'. Expected a positive integer.");
                return false;
            }
            cli.acoustic_substeps_from_cli = true;
            cli.cli_acoustic_substeps      = parsed;
        }
        else if (arg == "--qualification-hydrometeor-pulse")
        {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
            cli.qualification_hydrometeor_pulse = true;
#else
            report_cli_error("--qualification-hydrometeor-pulse is available only in "
                             "bin/nimbus_sim_agrid_qualification.");
            return false;
#endif
        }
        else if (arg == "--qualification-enable-milbrandt")
        {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
            cli.qualification_enable_milbrandt = true;
#else
            report_cli_error("--qualification-enable-milbrandt is available only in "
                             "the internal qualification binary.");
            return false;
#endif
        }
        else if (arg == "--qualification-milbrandt-mixed-phase")
        {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
            cli.qualification_enable_milbrandt      = true;
            cli.qualification_milbrandt_mixed_phase = true;
#else
            report_cli_error("--qualification-milbrandt-mixed-phase is available only in "
                             "the internal qualification binary.");
            return false;
#endif
        }
        else if (arg == "--qualification-acoustic-wave")
        {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
            cli.qualification_acoustic_wave = true;
#else
            report_cli_error("--qualification-acoustic-wave is available only in "
                             "bin/nimbus_sim_agrid_qualification.");
            return false;
#endif
        }
        else if (arg.rfind("--dt-large=", 0) == 0)
        {
            const std::string value = arg.substr(11);
            if (!try_parse_double_value(value, cli.cli_dt_large_s) || cli.cli_dt_large_s <= 0.0)
            {
                report_cli_error("Invalid --dt-large value '" + value + "'. Expected a positive finite number.");
                return false;
            }
            cli.dt_large_from_cli = true;
        }
        else if (arg == "--qualification-acoustic-standing-wave")
        {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
            cli.qualification_acoustic_standing_wave = true;
#else
            report_cli_error("--qualification-acoustic-standing-wave is available only in "
                             "bin/nimbus_sim_agrid_qualification.");
            return false;
#endif
        }
        else if (arg == "--qualification-acoustic-pulse")
        {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
            cli.qualification_acoustic_pulse = true;
#else
            report_cli_error("--qualification-acoustic-pulse is available only in "
                             "bin/nimbus_sim_agrid_qualification.");
            return false;
#endif
        }
        else if (arg == "--qualification-moving-nest")
        {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
            cli.qualification_moving_nest = true;
#else
            report_cli_error("--qualification-moving-nest is available only in "
                             "the internal qualification binary.");
            return false;
#endif
        }
        else if (arg == "--dt-large" && i + 1 < argc)
        {
            const std::string value = argv[++i];
            if (!try_parse_double_value(value, cli.cli_dt_large_s) || cli.cli_dt_large_s <= 0.0)
            {
                report_cli_error("Invalid --dt-large value '" + value + "'. Expected a positive finite number.");
                return false;
            }
            cli.dt_large_from_cli = true;
        }
        else if (arg == "--trajectories")
        {
            cli.traj_enabled          = true;
            cli.traj_enabled_from_cli = true;
        }
        else if (arg.rfind("--trajectory-seed-time=", 0) == 0)
        {
            const std::string value = arg.substr(23);
            if (!try_parse_double_value(value, cli.traj_seed_time_s) || cli.traj_seed_time_s < 0.0)
            {
                report_cli_error("Invalid --trajectory-seed-time value '" + value +
                                 "'. Expected a non-negative finite number.");
                return false;
            }
            cli.traj_enabled            = true;
            cli.traj_enabled_from_cli   = true;
            cli.traj_seed_time_from_cli = true;
        }
        else if (arg.rfind("--trajectory-count=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(19);
            if (!try_parse_int_value(value, parsed) || parsed <= 0)
            {
                report_cli_error("Invalid --trajectory-count value '" + value + "'. Expected a positive integer.");
                return false;
            }
            cli.traj_count            = parsed;
            cli.traj_enabled          = true;
            cli.traj_enabled_from_cli = true;
            cli.traj_count_from_cli   = true;
        }
        else if (arg.rfind("--write-every=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(14);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                report_cli_error("Invalid --write-every value '" + value + "'. Expected a non-negative integer.");
                return false;
            }
            cli.write_every_s        = parsed;
            cli.write_every_from_cli = true;
            cli.cli_write_every_s    = cli.write_every_s;
        }
        else if (arg.rfind("--outdir=", 0) == 0)
        {
            cli.outdir = arg.substr(9);
            if (cli.outdir.empty())
            {
                report_cli_error("Invalid --outdir value: path must not be empty.");
                return false;
            }
        }
        else if (arg.rfind("--checkpoint-every=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(19);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                report_cli_error("Invalid --checkpoint-every value '" + value + "'. Expected a non-negative integer.");
                return false;
            }
            cli.checkpoint_every_s        = parsed;
            cli.checkpoint_every_from_cli = true;
        }
        else if (arg.rfind("--checkpoint-dir=", 0) == 0)
        {
            cli.checkpoint_dir          = arg.substr(17);
            cli.checkpoint_dir_from_cli = true;
            if (cli.checkpoint_dir.empty())
            {
                report_cli_error("Invalid --checkpoint-dir value: path must not be empty.");
                return false;
            }
        }
        else if (arg.rfind("--resume=", 0) == 0)
        {
            cli.resume_checkpoint          = arg.substr(9);
            cli.resume_checkpoint_from_cli = true;
            if (cli.resume_checkpoint.empty())
            {
                report_cli_error("Invalid --resume value: path must not be empty.");
                return false;
            }
        }
        else if (arg.rfind("--config=", 0) == 0)
        {
            cli.config_path = arg.substr(9);
            if (cli.config_path.empty())
            {
                report_cli_error("Invalid --config value: path must not be empty.");
                return false;
            }
        }
        else if (arg == "--config" && i + 1 < argc)
        {
            cli.config_path = argv[++i];
            if (cli.config_path.empty())
            {
                report_cli_error("Invalid --config value: path must not be empty.");
                return false;
            }
        }
        else if (arg.rfind("--sounding=", 0) == 0)
        {
            cli.cli_sounding_path      = arg.substr(11);
            cli.sounding_path_from_cli = true;
            if (cli.cli_sounding_path.empty())
            {
                report_cli_error("Invalid --sounding value: path must not be empty.");
                return false;
            }
        }
        else if (arg == "--sounding" && i + 1 < argc)
        {
            cli.cli_sounding_path      = argv[++i];
            cli.sounding_path_from_cli = true;
            if (cli.cli_sounding_path.empty())
            {
                report_cli_error("Invalid --sounding value: path must not be empty.");
                return false;
            }
        }
        else if (arg.rfind("--log-profile=", 0) == 0)
        {
            bool valid          = false;
            cli.cli_log_profile = parse_log_profile(arg.substr(14), &valid);
            if (!valid)
            {
                std::cerr << "Invalid --log-profile value. Use quiet, normal, or debug." << std::endl;
                return false;
            }
            cli.log_profile_from_cli = true;
        }
        else if (arg == "--log-profile" && i + 1 < argc)
        {
            bool valid          = false;
            cli.cli_log_profile = parse_log_profile(argv[++i], &valid);
            if (!valid)
            {
                std::cerr << "Invalid --log-profile value. Use quiet, normal, or debug." << std::endl;
                return false;
            }
            cli.log_profile_from_cli = true;
        }
        else if (arg == "--timing")
        {
            cli.timing_from_cli         = true;
            cli.cli_perf_timing_enabled = true;
        }
        else if (arg == "--no-timing")
        {
            cli.timing_from_cli         = true;
            cli.cli_perf_timing_enabled = false;
        }
        else if (arg.rfind("--timing-every=", 0) == 0)
        {
            int               parsed = 0;
            const std::string value  = arg.substr(15);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                report_cli_error("Invalid --timing-every value '" + value + "'. Expected a non-negative integer.");
                return false;
            }
            cli.cli_perf_report_every_steps = parsed;
            if (cli.cli_perf_report_every_steps > 0)
            {
                cli.timing_from_cli         = true;
                cli.cli_perf_timing_enabled = true;
            }
        }
        else if (arg.rfind("--guard-mode=", 0) == 0)
        {
            tmv::GuardMode parsed = cli.cli_guard_mode;
            if (!tmv::parse_guard_mode(arg.substr(13), parsed))
            {
                std::cerr << "Invalid --guard-mode value. Use off, sanitize, or strict." << std::endl;
                return false;
            }
            cli.cli_guard_mode      = parsed;
            cli.guard_mode_from_cli = true;
        }
        else if (arg == "--guard-mode" && i + 1 < argc)
        {
            tmv::GuardMode parsed = cli.cli_guard_mode;
            if (!tmv::parse_guard_mode(argv[++i], parsed))
            {
                std::cerr << "Invalid --guard-mode value. Use off, sanitize, or strict." << std::endl;
                return false;
            }
            cli.cli_guard_mode      = parsed;
            cli.guard_mode_from_cli = true;
        }
        else if (arg.rfind("--guard-fail-on=", 0) == 0)
        {
            tmv::GuardFailOn parsed = cli.cli_guard_fail_on;
            if (!tmv::parse_guard_fail_on(arg.substr(16), parsed))
            {
                std::cerr << "Invalid --guard-fail-on value. Use nonfinite, bounds, or both." << std::endl;
                return false;
            }
            cli.cli_guard_fail_on      = parsed;
            cli.guard_fail_on_from_cli = true;
        }
        else if (arg == "--guard-fail-on" && i + 1 < argc)
        {
            tmv::GuardFailOn parsed = cli.cli_guard_fail_on;
            if (!tmv::parse_guard_fail_on(argv[++i], parsed))
            {
                std::cerr << "Invalid --guard-fail-on value. Use nonfinite, bounds, or both." << std::endl;
                return false;
            }
            cli.cli_guard_fail_on      = parsed;
            cli.guard_fail_on_from_cli = true;
        }
        else if (arg.rfind("--guard-scope=", 0) == 0)
        {
            tmv::StrictGuardScope parsed = cli.cli_guard_scope;
            if (!tmv::parse_strict_guard_scope(arg.substr(14), parsed))
            {
                std::cerr << "Invalid --guard-scope value. Use required or exported." << std::endl;
                return false;
            }
            cli.cli_guard_scope      = parsed;
            cli.guard_scope_from_cli = true;
        }
        else if (arg == "--guard-scope" && i + 1 < argc)
        {
            tmv::StrictGuardScope parsed = cli.cli_guard_scope;
            if (!tmv::parse_strict_guard_scope(argv[++i], parsed))
            {
                std::cerr << "Invalid --guard-scope value. Use required or exported." << std::endl;
                return false;
            }
            cli.cli_guard_scope      = parsed;
            cli.guard_scope_from_cli = true;
        }
        else if (arg.rfind("--guard-report=", 0) == 0)
        {
            cli.cli_guard_report_path = arg.substr(15);
            cli.guard_report_from_cli = true;
            if (cli.cli_guard_report_path.empty())
            {
                report_cli_error("Invalid --guard-report value: path must not be empty.");
                return false;
            }
        }
        else if (arg == "--guard-report" && i + 1 < argc)
        {
            cli.cli_guard_report_path = argv[++i];
            cli.guard_report_from_cli = true;
            if (cli.cli_guard_report_path.empty())
            {
                report_cli_error("Invalid --guard-report value: path must not be empty.");
                return false;
            }
        }
        else if (arg.rfind("--compute-backend=", 0) == 0)
        {
            const std::string  value  = arg.substr(18);
            ComputeBackendKind parsed = ComputeBackendKind::Cpu;
            if (!parse_compute_backend_kind(value, parsed))
            {
                report_cli_error("Invalid --compute-backend value '" + value + "'. Expected: cpu, vulkan, gpu, metal.");
                return false;
            }
            cli.cli_compute_backend      = compute_backend_kind_name(parsed);
            cli.compute_backend_from_cli = true;
        }
        else if (arg == "--compute-backend" && i + 1 < argc)
        {
            const std::string  value  = argv[++i];
            ComputeBackendKind parsed = ComputeBackendKind::Cpu;
            if (!parse_compute_backend_kind(value, parsed))
            {
                report_cli_error("Invalid --compute-backend value '" + value + "'. Expected: cpu, vulkan, gpu, metal.");
                return false;
            }
            cli.cli_compute_backend      = compute_backend_kind_name(parsed);
            cli.compute_backend_from_cli = true;
        }
        else if (arg == "--live-shm")
        {
            cli.live_shm_from_cli = true;
            cli.cli_live_shm      = true;
        }
        else if (arg.rfind("--live-shm-fields=", 0) == 0)
        {
            const std::string value = arg.substr(18);
            cli.cli_live_shm_fields.clear();
            std::istringstream stream(value);
            std::string        token;
            while (std::getline(stream, token, ','))
            {
                auto start = token.find_first_not_of(" \t");
                auto end   = token.find_last_not_of(" \t");
                if (start != std::string::npos)
                {
                    cli.cli_live_shm_fields.push_back(token.substr(start, end - start + 1));
                }
            }
            if (!cli.cli_live_shm_fields.empty())
            {
                cli.live_shm_fields_from_cli = true;
            }
            else
            {
                report_cli_error("Invalid --live-shm-fields value: provide at least one field.");
                return false;
            }
        }
        else if (arg.rfind("--live-shm-name=", 0) == 0)
        {
            cli.cli_live_shm_name = arg.substr(16);
            if (!tmv_shm::valid_logical_shm_name(cli.cli_live_shm_name))
            {
                report_cli_error("Invalid --live-shm-name value: use / followed by at most 29 letters, digits, '_' or '-'.");
                return false;
            }
            cli.live_shm_name_from_cli = true;
        }
        else if (arg == "--duration" || arg == "--config" || arg == "--sounding" || arg == "--log-profile" ||
                 arg == "--guard-mode" || arg == "--guard-fail-on" || arg == "--guard-scope" ||
                 arg == "--guard-report" || arg == "--compute-backend" || arg == "--migrate-config" ||
                 arg == "--export-ms" || arg == "--write-every" || arg == "--outdir" || arg == "--timing-every" ||
                 arg == "--live-shm-fields" || arg == "--live-shm-name" ||
                 arg == "--trajectory-seed-time" || arg == "--trajectory-count" ||
                 arg == "--threads" || arg == "--checkpoint-every" || arg == "--checkpoint-dir" || arg == "--resume" ||
                 arg == "--acoustic-substeps" || arg == "--dt-large")
        {
            report_cli_error("Missing value for " + arg + ".");
            return false;
        }
        else
        {
            report_cli_error("Unknown argument '" + arg + "'.");
            return false;
        }
    }

    if (cli.preflight_context && !cli.validate_only)
    {
        report_cli_error("--preflight-context=json requires --validate-only");
        return false;
    }
    if (cli.preflight_context && (cli.compute_backend_from_cli || cli.sounding_path_from_cli ||
        cli.dt_large_from_cli || cli.acoustic_substeps_from_cli || cli.write_every_from_cli ||
        cli.guard_mode_from_cli || cli.guard_fail_on_from_cli || cli.guard_scope_from_cli ||
        cli.live_shm_from_cli || cli.live_shm_fields_from_cli || cli.live_shm_name_from_cli ||
        cli.traj_enabled_from_cli || cli.traj_seed_time_from_cli || cli.traj_count_from_cli))
    {
        report_cli_error("--preflight-context=json validates original source controls without science or publication overrides");
        return false;
    }
    return true;
}

bool apply_thread_cli_override(const RuntimeCliOptions& cli)
{
#ifdef _OPENMP
    if (cli.threads_from_cli)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(cli.cli_threads);
        return true;
    }
    // An explicit standard worker-count request retains authority. Other
    // OpenMP controls (dynamic teams, binding, places, and runtime-specific
    // settings) remain owned by the runtime in either case.
    if (std::getenv("OMP_NUM_THREADS") != nullptr)
    {
        return true;
    }
    const HardwareInfo hardware         = detect_hardware();
    const int          topology_default = topology_aware_openmp_default(hardware);
    if (topology_default > 0)
    {
        omp_set_num_threads(topology_default);
        tmv::log_info("[HARDWARE] Default OpenMP workers: ",
                      topology_default,
                      " based on CPU topology; use --threads or "
                      "OMP_NUM_THREADS to override");
    }
    return true;
#else
    if (!cli.threads_from_cli)
    {
        return true;
    }
    if (cli.cli_threads == 1)
    {
        return true;
    }
    report_cli_error("--threads requires an OpenMP-enabled build (except --threads=1).");
    return false;
#endif
}

void apply_pre_config_cli_overrides(const RuntimeCliOptions& cli)
{
    if (cli.log_profile_from_cli)
    {
        global_log_profile = cli.cli_log_profile;
    }
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
    set_internal_milbrandt_qualification(cli.qualification_enable_milbrandt);
    set_internal_moving_nest_qualification(cli.qualification_moving_nest);
#endif
}

bool load_runtime_config(RuntimeRunOptions& run)
{
    RuntimeCliOptions& cli = run.cli;
    try
    {
        load_config(cli.config_path,
                    cli.duration_s,
                    cli.write_every_s,
                    cli.outdir,
                    &run.output_config,
                    &run.live_shm,
                    &run.live_shm_fields,
                    cli.write_every_from_cli ? cli.cli_write_every_s : -1,
                    cli.live_shm_from_cli ? static_cast<int>(cli.cli_live_shm) : -1,
                    cli.live_shm_fields_from_cli ? &cli.cli_live_shm_fields : nullptr,
                    cli.traj_enabled,
                    cli.compute_backend_from_cli ? &cli.cli_compute_backend : nullptr,
                    cli.sounding_path_from_cli ? &cli.cli_sounding_path : nullptr,
                    cli.duration_from_cli ? cli.cli_duration_s : -1);
    }
    catch (const std::exception& e)
    {
        // Configuration may have initialized the selected backend to perform
        // device admission immediately before field allocation.
        shutdown_compute_backend_runtime();
        std::cerr << "[CONFIG] " << e.what() << std::endl;
        return false;
    }

    if (cli.sounding_path_from_cli)
        tmv::log_info("[CLI] sounding override preflighted as type=file, path=", cli.cli_sounding_path);

    return true;
}

tmv::init::SoundingSourceConfig validation_sounding_config(const tmv::init::config::ConfigV1& config)
{
    using tmv::init::config::SoundingType;
    const auto&                     requested = config.initial_conditions.sounding;
    tmv::init::SoundingSourceConfig source;
    switch (requested.type)
    {
    case SoundingType::ParametricCAPE:
        source.type = tmv::init::SoundingSourceConfig::Type::ParametricCAPE;
        break;
    case SoundingType::ParametricTargets:
        source.type = tmv::init::SoundingSourceConfig::Type::ParametricTargets;
        break;
    case SoundingType::File:
        source.type = tmv::init::SoundingSourceConfig::Type::File;
        break;
    }
    source.parametric.cape_target_jkg        = requested.cape_jkg;
    source.parametric.surface_theta_k        = requested.surface_theta_k;
    source.parametric.surface_qv_kgkg        = requested.moisture_qv_kgkg.value_or(0.014);
    source.parametric.tropopause_z_m         = requested.tropopause_m;
    source.targets.target_cape_jkg           = requested.cape_jkg;
    source.targets.target_cin_jkg            = requested.cin_jkg;
    source.targets.target_lcl_m              = requested.lcl_m;
    source.targets.target_lfc_m              = requested.lfc_m;
    source.targets.target_el_m               = requested.el_m;
    source.targets.surface_theta_k           = requested.surface_theta_k;
    source.targets.moisture_qv_kgkg_override = requested.moisture_qv_kgkg.value_or(0.0);
    source.file.path                         = requested.file_path;
    source.file.scheme_id                    = "auto";
    source.file.require_winds                = true;
    source.file.use_fallback_profiles        = false;
    return source;
}

tmv::init::HodographSourceConfig validation_hodograph_config(const tmv::init::config::ConfigV1& config)
{
    using tmv::init::config::HodographType;
    const auto&                      requested = config.initial_conditions.hodograph;
    tmv::init::HodographSourceConfig source;
    source.type = requested.type == HodographType::Custom ? tmv::init::HodographSourceConfig::Type::WKParam
                                                          : tmv::init::HodographSourceConfig::Type::Auto;
    source.wk_anchors.u_sfc_ms = requested.u_sfc_ms;
    source.wk_anchors.v_sfc_ms = requested.v_sfc_ms;
    source.wk_anchors.u_1km_ms = requested.u_1km_ms;
    source.wk_anchors.v_1km_ms = requested.v_1km_ms;
    source.wk_anchors.u_6km_ms = requested.u_6km_ms;
    source.wk_anchors.v_6km_ms = requested.v_6km_ms;
    return source;
}

tmv::init::TriggerSourceConfig validation_trigger_config(const tmv::init::config::ConfigV1& config)
{
    using tmv::init::config::TriggerType;
    const auto&                    requested = config.initial_conditions.trigger;
    tmv::init::TriggerSourceConfig source;
    switch (requested.type)
    {
    case TriggerType::None:
        source.type = tmv::init::TriggerSourceConfig::Type::None;
        break;
    case TriggerType::Bubble:
        source.type = tmv::init::TriggerSourceConfig::Type::WarmBubble;
        break;
    case TriggerType::VortexSeed:
        source.type = tmv::init::TriggerSourceConfig::Type::VortexSeed;
        break;
    }
    source.vortex_seed.r_max_m    = requested.vortex_seed.r_max_m;
    source.vortex_seed.v_max_ms   = requested.vortex_seed.v_max_ms;
    source.vortex_seed.z_top_m    = requested.vortex_seed.z_top_m;
    source.vortex_seed.center_x_m = requested.vortex_seed.center_x_m;
    source.vortex_seed.center_y_m = requested.vortex_seed.center_y_m;
    source.vortex_seed.center_z_m = requested.vortex_seed.center_z_m;
    return source;
}

void validate_domain_lifetime_preflight(const tmv::init::config::ConfigV1&      config,
                                        const std::vector<double>&              z_m,
                                        const tmv::init::Sounding&              sounding,
                                        const tmv::init::HodographSourceConfig& hodograph)
{
    using namespace tmv::init::config;
    const auto& planning = config.domain_motion.lifetime_planning;
    const auto& tracking = config.domain_motion.tracking;
    const bool  branch_motion_required =
        tracking.enabled && tracking.branch_preference != StormBranchPreference::Continuity;
    if (!planning.enabled && !branch_motion_required)
        return;

    const tmv::init::WindColumn wind = tmv::init::resolve_wind_column(z_m, sounding, hodograph);
    if (branch_motion_required)
    {
        (void)tmv::domain::estimate_bunkers_storm_motion(
            z_m, wind.u_ms, wind.v_ms, tmv::domain::StormMotionMethod::BunkersRight);
        (void)tmv::domain::estimate_bunkers_storm_motion(
            z_m, wind.u_ms, wind.v_ms, tmv::domain::StormMotionMethod::BunkersLeft);
    }
    if (!planning.enabled)
        return;

    tmv::domain::StormMotionEstimate estimate;
    switch (planning.motion_source)
    {
    case StormMotionSource::Explicit:
        estimate.method      = tmv::domain::StormMotionMethod::Explicit;
        estimate.velocity_ms = {
            planning.explicit_u_ms,
            planning.explicit_v_ms,
        };
        break;
    case StormMotionSource::BunkersRight:
        estimate = tmv::domain::estimate_bunkers_storm_motion(
            z_m, wind.u_ms, wind.v_ms, tmv::domain::StormMotionMethod::BunkersRight);
        break;
    case StormMotionSource::BunkersLeft:
        estimate = tmv::domain::estimate_bunkers_storm_motion(
            z_m, wind.u_ms, wind.v_ms, tmv::domain::StormMotionMethod::BunkersLeft);
        break;
    }

    tmv::domain::HorizontalPoint center;
    switch (config.initial_conditions.trigger.type)
    {
    case TriggerType::Bubble:
        center = {
            config.initial_conditions.trigger.bubble.center_x_m,
            config.initial_conditions.trigger.bubble.center_y_m,
        };
        break;
    case TriggerType::VortexSeed:
        center = {
            config.initial_conditions.trigger.vortex_seed.center_x_m,
            config.initial_conditions.trigger.vortex_seed.center_y_m,
        };
        break;
    case TriggerType::None:
        throw std::runtime_error("storm-lifetime planning requires a bubble or vortex_seed trigger "
                                 "so the initial storm center is defined");
    }

    const PhysicalCellDomain physical =
        physical_scalar_domain(config.grid.nx, config.grid.ny, config.grid.nz, config.grid.staggering);
    const double west_face_m  = tmv::grid_coordinates::face(physical.i_begin, config.grid.dx);
    const double south_face_m = tmv::grid_coordinates::face(physical.j_begin, config.grid.dy);

    tmv::domain::StormLifetimePlanInput input;
    input.nx                     = static_cast<std::size_t>(physical.nx());
    input.ny                     = static_cast<std::size_t>(physical.ny());
    input.dx_m                   = config.grid.dx;
    input.dy_m                   = config.grid.dy;
    input.run_time_s             = config.time.run_time;
    input.initial_storm_center_m = {
        center.x_m - west_face_m,
        center.y_m - south_face_m,
    };
    input.earth_relative_storm_motion_ms = estimate.velocity_ms;
    input.frame.velocity_ms              = {
        config.domain_motion.frame_velocity.u_ms,
        config.domain_motion.frame_velocity.v_ms,
    };
    input.upstream_buffer_m     = planning.upstream_buffer_m;
    input.downstream_buffer_m   = planning.downstream_buffer_m;
    input.cross_track_buffer_m  = planning.cross_track_buffer_m;
    input.motion_uncertainty_ms = planning.motion_uncertainty_ms;

    const tmv::domain::StormLifetimePlan plan = tmv::domain::plan_storm_lifetime(input);
    std::ostringstream                   report;
    report << std::fixed << std::setprecision(1) << "storm-lifetime preflight ["
           << tmv::domain::storm_motion_method_name(estimate.method) << "/"
           << tmv::domain::storm_motion_algorithm_id(estimate.method) << "]: motion=(" << estimate.velocity_ms.u_ms
           << ", " << estimate.velocity_ms.v_ms << ") m/s, physical domain=" << physical.nx() << "x" << physical.ny()
           << " cells, required=" << plan.required_nx << "x" << plan.required_ny
           << " physical cells (configure grid.nx>="
           << plan.required_nx + static_cast<std::size_t>(config.grid.nx - physical.nx())
           << ", grid.ny>=" << plan.required_ny + static_cast<std::size_t>(config.grid.ny - physical.ny())
           << " for current halo topology), recommended trigger center=("
           << (plan.recommended_initial_center_m.x_m + west_face_m) / 1000.0 << ", "
           << (plan.recommended_initial_center_m.y_m + south_face_m) / 1000.0 << ") km";
    if (estimate.method != tmv::domain::StormMotionMethod::Explicit)
    {
        report << ", 0--6 km mean=(" << estimate.mean_wind_0_6km_ms.u_ms << ", " << estimate.mean_wind_0_6km_ms.v_ms
               << ") m/s, shear=(" << estimate.shear_0_6km_ms.u_ms << ", " << estimate.shear_0_6km_ms.v_ms << ") m/s";
    }
    if (plan.contained_for_run)
    {
        std::cout << "[VALIDATE] " << report.str() << "; PASS\n";
        return;
    }

    report << "; FAIL: first envelope risk at " << tmv::domain::boundary_face_name(plan.first_risk_face)
           << " boundary, t=" << plan.first_risk_time_s << " s";
    if (planning.action == LifetimePlanningAction::Warn)
    {
        std::cerr << "[VALIDATE WARNING] " << report.str() << "\n";
        return;
    }
    throw std::runtime_error(report.str());
}

std::vector<std::string> validate_v1_selected_science_profile(const tmv::init::config::ConfigV1& config)
{
    using namespace tmv::init::config;
    std::string dynamics_id = config.dynamics.scheme;
    if (tmv::strutil::trim_and_lower(dynamics_id) == "klemp_wilhelmson_split")
    {
        dynamics_id = config.grid.staggering == StaggerType::CGrid ? tmv::kCartesianCGridDynamicsId
                                                                   : tmv::kCartesianAGridDynamicsId;
    }
    const bool qualified_thompson_cpu =
        tmv::init::config::is_public_qualified_thompson_cpu_profile(config);
    const auto* microphysics_descriptors =
        &tmv::get_microphysics_scheme_descriptors();
    const auto* radiation_descriptors =
        &tmv::get_radiation_scheme_descriptors();
    const auto* time_stepping_descriptors =
        &tmv::get_time_stepping_scheme_descriptors();
    std::vector<tmv::SchemeDescriptor> qualified_time_stepping_descriptors;
    if (is_public_qualified_coupled_rk_profile(config))
    {
        qualified_time_stepping_descriptors = *time_stepping_descriptors;
        for (auto& descriptor : qualified_time_stepping_descriptors)
        {
            if (descriptor.id == "rk3" || descriptor.id == "rk4")
            {
                descriptor.maturity = tmv::SchemeMaturity::Qualified;
                descriptor.runtime_selectable = true;
            }
        }
        time_stepping_descriptors = &qualified_time_stepping_descriptors;
    }
    std::vector<tmv::runtime::SelectedScheme> selections = {
        {"dynamics.scheme", dynamics_id, &tmv::get_dynamics_scheme_descriptors()},
        {"microphysics.scheme",
         microphysics_scheme_name(config.microphysics.scheme),
         microphysics_descriptors},
        {"radiation.scheme", radiation_scheme_name(config.radiation.scheme), radiation_descriptors},
        {"boundary_layer.scheme",
         boundary_layer_scheme_name(config.boundary_layer.scheme),
         &tmv::get_boundary_layer_scheme_descriptors()},
        {"turbulence.scheme",
         turbulence_scheme_name(config.turbulence.scheme),
         &tmv::get_turbulence_scheme_descriptors()},
        {"numerics.advection.scheme",
         resolved_advection_scheme_name(config.advection.scheme,
                                         config.grid.staggering),
         &tmv::get_advection_scheme_descriptors()},
        {"diffusion.scheme", diffusion_scheme_name(config.diffusion.scheme), &tmv::get_diffusion_scheme_descriptors()},
        {"numerics.time_stepping.scheme",
         time_integrator_name(config.time.integrator),
         time_stepping_descriptors},
        {"chaos.scheme",
         chaos_scheme_name(config.extensions.chaos.scheme),
         &tmv::chaos::get_chaos_scheme_descriptors()},
        {"terrain.scheme", terrain_scheme_name(config.terrain.scheme), &tmv::get_terrain_scheme_descriptors()},
    };
    if (config.radar.enabled)
    {
        for (const std::string& product : config.radar.products)
        {
            const char* scheme = product == "reflectivity_dbz" ? "reflectivity"
                : product == "radial_velocity" ? "velocity"
                : product == "zdr" ? "zdr" : product.c_str();
            selections.push_back({
                "radar.products", scheme, &tmv::get_radar_scheme_descriptors()});
        }
    }
    if (config.initial_conditions.sounding.type == SoundingType::File)
    {
        selections.push_back({
            "environment.sounding.file.scheme_id",
            "auto",
            &tmv::get_sounding_scheme_descriptors(),
        });
    }
    const bool qualified_collocated_mapped =
        tmv::init::config::
            is_public_qualified_collocated_mapped_terrain_profile(config);
    auto errors = tmv::runtime::validate_science_profile(
        selections,
        config.extensions.execution.mode == ExecutionMode::Qualified,
        coordinate_system_name(config.grid.coordinate_system),
        stagger_type_name(config.grid.staggering),
        config.extensions.compute.backend == tmv::init::config::ComputeBackend::CPU ? "cpu" : "gpu",
        qualified_collocated_mapped);
    const auto admission = evaluate_runtime_science_admission(config);
    const bool qualified_domain_recycling = admission.domain_recycling;
    const bool qualified_moving_nest = admission.moving_nest;
    const bool qualified_two_way_nested = admission.two_way_nested;
    const bool qualified_mapped_terrain = admission.mapped_terrain;
    const bool qualified_lin_rrtmg = admission.lin_rrtmg;
    const bool qualified_cloud_top_radiative_ysu = admission.cloud_top_radiative_ysu;
    const bool qualified_radar = admission.radar;
    const bool qualified_user_filter_width = admission.user_filter_width;
    const bool qualified_dynamic_smagorinsky = admission.dynamic_smagorinsky;
    const bool qualified_tke_stability = admission.tke_stability;
    const bool qualified_variable_diffusion = admission.variable_diffusion;
    const bool qualified_simple_grey_absorbing = admission.simple_grey_absorbing;
    const bool qualified_agrid = admission.agrid_core || qualified_collocated_mapped;
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.radiation.scheme == RadiationScheme::SimpleGrey &&
        config.radiation.simple_grey_optical_model ==
            "absorbing_water_cloud_path_v1" &&
        !qualified_simple_grey_absorbing)
    {
        errors.push_back(
            "absorbing water/cloud-path simple_grey is qualified only for the exact flat periodic C-grid CPU profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::LinearHeight &&
        !qualified_variable_diffusion)
    {
        errors.push_back(
            "qualified linear-height variable diffusion is limited to the exact flat native reference profiles");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.dynamic_coefficient &&
        !qualified_dynamic_smagorinsky)
    {
        errors.push_back(
            "dynamic Smagorinsky is qualified only for the exact flat periodic C-grid CPU reference profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.scheme == TurbulenceScheme::TKE &&
        !qualified_tke_stability &&
        !is_public_qualified_coupled_rk_profile(config) &&
        !qualified_agrid && !qualified_mapped_terrain)
    {
        errors.push_back(
            "turbulence.scheme=tke is outside the qualified flat, RK, A-grid, and mapped CPU families");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.stability_correction ==
            TurbulenceStabilityCorrection::Richardson && !admission.richardson_smagorinsky)
    {
        errors.push_back(
            "Richardson-corrected Smagorinsky requires the maintained Lilly CPU reference profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.filter_width == TurbulenceFilterWidth::User &&
        !qualified_user_filter_width)
    {
        errors.push_back(
            "turbulence.filter_width=user is qualified only for the exact flat periodic C-grid CPU Smagorinsky profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        !config.boundary_layer.clear_sky_only &&
        !qualified_cloud_top_radiative_ysu)
    {
        errors.push_back(
            "cloud-top radiative YSU mixing is qualified only for the exact surface-coupled CPU reference profile");
    }
    if (config.domain_motion.nesting.feedback == NestingFeedback::TwoWay &&
        !qualified_two_way_nested)
    {
        errors.push_back(
            "domain_motion.nesting.feedback=two_way requires the exact maintained stationary, parent-recycling, or moving-child "
            "CPU profiles with complete interface-register, native-dual "
            "restriction, and local reflux ownership");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.extensions.compute.backend == tmv::init::config::ComputeBackend::Metal &&
        !admits_qualified_metal(admission))
    {
        errors.push_back(kQualifiedMetalScopeError);
    }
    tmv::runtime::append_qualified_kessler_scope_error(
        errors,
        config.extensions.execution.mode == ExecutionMode::Qualified,
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::Kessler,
        config.domain_motion.nesting.enabled,
        qualified_domain_recycling || qualified_radar ||
            qualified_simple_grey_absorbing ||
            is_public_qualified_coupled_rk_profile(config) ||
            qualified_agrid || qualified_mapped_terrain);
    tmv::runtime::append_qualified_lin_scope_error(
        errors,
        config.extensions.execution.mode == ExecutionMode::Qualified,
        config.microphysics.scheme == tmv::init::config::MicrophysicsScheme::Lin,
        is_public_qualified_two_way_nested_profile(config) || qualified_lin_rrtmg || admission.lin_storm ||
            is_public_qualified_coupled_rk_profile(config) || qualified_agrid || qualified_mapped_terrain);
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.microphysics.scheme == tmv::init::config::MicrophysicsScheme::Thompson && !qualified_thompson_cpu &&
        !qualified_two_way_nested && !tmv::init::config::is_public_ready_thompson_native_profile(config) &&
        !is_public_qualified_coupled_rk_profile(config) && !qualified_agrid && !qualified_mapped_terrain)
        errors.push_back(
            "microphysics.scheme=thompson requires a maintained CPU profile or the native C-grid Thompson reference");
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.microphysics.scheme == tmv::init::config::MicrophysicsScheme::Milbrandt &&
        config.extensions.compute.backend != tmv::init::config::ComputeBackend::CPU && !admission.milbrandt_reference &&
        !qualified_two_way_nested && !is_public_qualified_coupled_rk_profile(config))
        errors.push_back("microphysics.scheme=milbrandt requires the maintained native C-grid reference profile");
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        (config.radar.enabled || config.output.preset == OutputPreset::RadarReady ||
         config.output.preset == OutputPreset::RadarPolarimetricReady) &&
        !qualified_radar)
    {
        errors.push_back(
            "qualified radar execution is limited to exact maintained CPU radar profiles");
    }
    tmv::runtime::append_qualified_tke_scope_error(
        errors,
        config.extensions.execution.mode == ExecutionMode::Qualified &&
            qualified_tke_stability,
        {
            turbulence_scheme_name(config.turbulence.scheme),
            config.extensions.compute.backend ==
                    tmv::init::config::ComputeBackend::CPU
                ? "cpu" : "gpu",
            coordinate_system_name(config.grid.coordinate_system),
            stagger_type_name(config.grid.staggering),
            turbulence_mode_name(config.turbulence.mode),
            microphysics_scheme_name(config.microphysics.scheme),
            boundary_layer_scheme_name(config.boundary_layer.scheme),
            surface_scheme_name(config.surface.scheme),
            terrain_scheme_name(config.terrain.scheme),
            chaos_scheme_name(config.extensions.chaos.scheme),
            diffusion_scheme_name(config.diffusion.scheme),
            diffusion_target_name(config.diffusion.apply_to),
            lateral_bc_name(config.boundary_conditions.lateral.west),
            lateral_bc_name(config.boundary_conditions.lateral.east),
            lateral_bc_name(config.boundary_conditions.lateral.south),
            lateral_bc_name(config.boundary_conditions.lateral.north),
            bottom_bc_name(config.boundary_conditions.bottom),
            top_bc_name(config.boundary_conditions.top),
            config.surface.apply_fluxes,
            config.diffusion.K_h > 0.0 || config.diffusion.K_v > 0.0 ||
                config.diffusion.coefficient_source ==
                    DiffusionCoefficientSource::LinearHeight,
            config.damping.enabled,
            config.damping.lateral_sponge_enabled,
            config.damping.w_only_top_sponge,
            config.turbulence.Pr_t,
            config.turbulence.Sc_t,
            turbulence_stability_correction_name(
                config.turbulence.stability_correction),
        });
    tmv::runtime::append_qualified_weno5_scope_error(
        errors,
        config.extensions.execution.mode == ExecutionMode::Qualified && !admission.native_weno5 && !admission.two_way_accelerator_reference &&
            !qualified_tke_stability && !qualified_thompson_cpu && !is_public_qualified_coupled_rk_profile(config) &&
            !qualified_agrid && !qualified_mapped_terrain,
        {
            resolved_advection_scheme_name(config.advection.scheme, config.grid.staggering),
            config.extensions.compute.backend == tmv::init::config::ComputeBackend::CPU ? "cpu" : "gpu",
            coordinate_system_name(config.grid.coordinate_system),
            stagger_type_name(config.grid.staggering),
            dynamics_id,
            microphysics_scheme_name(config.microphysics.scheme),
            turbulence_scheme_name(config.turbulence.scheme),
            boundary_layer_scheme_name(config.boundary_layer.scheme),
            surface_scheme_name(config.surface.scheme),
            radiation_scheme_name(config.radiation.scheme),
            terrain_scheme_name(config.terrain.scheme),
            chaos_scheme_name(config.extensions.chaos.scheme),
            diffusion_scheme_name(config.diffusion.scheme),
            diffusion_target_name(config.diffusion.apply_to),
            lateral_bc_name(config.boundary_conditions.lateral.west),
            lateral_bc_name(config.boundary_conditions.lateral.east),
            lateral_bc_name(config.boundary_conditions.lateral.south),
            lateral_bc_name(config.boundary_conditions.lateral.north),
            bottom_bc_name(config.boundary_conditions.bottom),
            top_bc_name(config.boundary_conditions.top),
            domain_motion_mode_name(config.domain_motion.mode),
            config.surface.apply_fluxes,
            config.diffusion.K_h > 0.0 || config.diffusion.K_v > 0.0 ||
                config.diffusion.coefficient_source == DiffusionCoefficientSource::LinearHeight,
            config.damping.enabled,
            config.damping.lateral_sponge_enabled,
            config.damping.w_only_top_sponge,
            config.domain_motion.nesting.enabled,
        });
    tmv::runtime::append_qualified_implicit_diffusion_scope_error(
        errors,
        config.extensions.execution.mode == ExecutionMode::Qualified,
        {
            diffusion_scheme_name(config.diffusion.scheme),
            config.extensions.compute.backend ==
                    tmv::init::config::ComputeBackend::CPU
                ? "cpu" : "gpu",
            coordinate_system_name(config.grid.coordinate_system),
            stagger_type_name(config.grid.staggering),
            dynamics_id,
            resolved_advection_scheme_name(
                config.advection.scheme, config.grid.staggering),
            microphysics_scheme_name(config.microphysics.scheme),
            turbulence_scheme_name(config.turbulence.scheme),
            boundary_layer_scheme_name(config.boundary_layer.scheme),
            surface_scheme_name(config.surface.scheme),
            radiation_scheme_name(config.radiation.scheme),
            terrain_scheme_name(config.terrain.scheme),
            chaos_scheme_name(config.extensions.chaos.scheme),
            diffusion_target_name(config.diffusion.apply_to),
            lateral_bc_name(config.boundary_conditions.lateral.west),
            lateral_bc_name(config.boundary_conditions.lateral.east),
            lateral_bc_name(config.boundary_conditions.lateral.south),
            lateral_bc_name(config.boundary_conditions.lateral.north),
            bottom_bc_name(config.boundary_conditions.bottom),
            top_bc_name(config.boundary_conditions.top),
            domain_motion_mode_name(config.domain_motion.mode),
            config.surface.apply_fluxes,
            false,
            config.damping.enabled,
            config.damping.lateral_sponge_enabled,
            config.damping.w_only_top_sponge,
            config.domain_motion.nesting.enabled,
            config.diffusion.K_h,
            config.diffusion.K_v,
        },
        qualified_mapped_terrain,
        admission.native_implicit_diffusion || is_public_qualified_coupled_rk_profile(config));
    tmv::runtime::append_qualified_terrain_scope_error(
        errors,
        config.extensions.execution.mode == ExecutionMode::Qualified,
        terrain_scheme_name(config.terrain.scheme),
        qualified_mapped_terrain);
    const tmv::runtime::ScienceInteractionSelection interactions{
        boundary_layer_scheme_name(config.boundary_layer.scheme),
        turbulence_scheme_name(config.turbulence.scheme),
        turbulence_mode_name(config.turbulence.mode),
        surface_scheme_name(config.surface.scheme),
        config.surface.apply_fluxes,
        terrain_scheme_name(config.terrain.scheme),
        radiation_scheme_name(config.radiation.scheme),
        config.radiation.do_sw,
        chaos_scheme_name(config.extensions.chaos.scheme),
        diffusion_scheme_name(config.diffusion.scheme),
        diffusion_target_name(config.diffusion.apply_to),
        config.diffusion.K_h > 0.0 || config.diffusion.K_v > 0.0,
        tmv::init::config::output_format_name(config.output.format),
        config.extensions.execution.mode == ExecutionMode::Qualified &&
            !qualified_mapped_terrain,
        stagger_type_name(config.grid.staggering),
        lateral_bc_name(config.boundary_conditions.lateral.west),
        lateral_bc_name(config.boundary_conditions.lateral.east),
        lateral_bc_name(config.boundary_conditions.lateral.south),
        lateral_bc_name(config.boundary_conditions.lateral.north),
        bottom_bc_name(config.boundary_conditions.bottom),
        top_bc_name(config.boundary_conditions.top),
        config.extensions.diagnostics.scientific_every_steps,
        config.extensions.compute.backend == tmv::init::config::ComputeBackend::CPU
            ? "cpu" : "gpu",
        dynamics_id,
        microphysics_scheme_name(config.microphysics.scheme),
        config.extensions.compute.gpu_apply_microphysics,
        config.extensions.compute.gpu_apply_slow,
        config.extensions.compute.gpu_apply_sgs,
    };
    const auto interaction_errors = tmv::runtime::validate_science_interactions(interactions);
    errors.insert(errors.end(), interaction_errors.begin(), interaction_errors.end());
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        (config.output.format == tmv::init::config::OutputFormat::ZfpLow ||
         config.output.format == tmv::init::config::OutputFormat::ZfpMedium ||
         config.output.format == tmv::init::config::OutputFormat::ZfpHigh))
    {
        const auto   projected = tmv::init::config::serialize_v1_to_v0_map(config);
        OutputConfig output    = parse_output_config(projected);
        resolve_output_fields(output);
        merge_selected_radar_output_fields(output, config.radar.products);
        const auto radar_context = make_qualified_radar_zfp_v4_context(
            config, output, config.time.output_interval,
            config.output.live_shm, config.output.trajectories);
        const auto zfp_errors = validate_qualified_zfp_history(output, &radar_context);
        for (const std::string& error : zfp_errors)
            errors.push_back("output.format=zfp: " + error);
    }
    return errors;
}

int validate_config_without_runtime_allocation(const std::string& config_path,
                                               const std::string* compute_backend_override,
                                               const std::string* sounding_file_override,
                                               int                duration_override,
                                               const RuntimeCliOptions& cli)
{
    tmv::init::config::LoadResult loaded;
    std::string captured_source;
    if (cli.preflight_context)
    {
        try
        {
            std::ifstream input(config_path, std::ios::binary);
            if (!input)
            {
                throw std::runtime_error("could not open preflight context source");
            }
            captured_source = tmv::runtime::read_validate_only_source_bytes(input);
            loaded = tmv::init::config::load_v1_config_fields(
                config_path, parse_yaml_text_with_lines(captured_source));
        }
        catch (const std::exception& error)
        {
            std::cerr << "[VALIDATE] preflight context source failed: " << error.what() << std::endl;
            return 2;
        }
    }
    else
    {
        loaded = tmv::init::config::load_v1_config(config_path);
    }
    if (!loaded.ok())
    {
        std::cerr << "[VALIDATE] strict nimbus.v1 load failed:\n"
                  << tmv::init::config::format_errors(loaded.errors) << std::endl;
        return 2;
    }

    try
    {
        auto effective_config = *loaded.config;
        if (compute_backend_override != nullptr)
        {
            effective_config.extensions.compute.backend = *compute_backend_override == "cpu"
                ? tmv::init::config::ComputeBackend::CPU
                : (*compute_backend_override == "metal"
                       ? tmv::init::config::ComputeBackend::Metal
                       : tmv::init::config::ComputeBackend::GPU);
        }
        if (sounding_file_override != nullptr)
        {
            effective_config.initial_conditions.sounding.type      = tmv::init::config::SoundingType::File;
            effective_config.initial_conditions.sounding.file_path = *sounding_file_override;
        }
        const auto profile_errors = validate_v1_selected_science_profile(effective_config);
        if (!profile_errors.empty())
        {
            for (const auto& error : profile_errors)
            {
                std::cerr << "[VALIDATE] " << error << std::endl;
            }
            return 2;
        }
        const auto admission = evaluate_runtime_science_admission(effective_config);
        const bool radar_zfp_requested =
            effective_config.extensions.execution.mode == tmv::init::config::ExecutionMode::Qualified &&
            (effective_config.output.format == tmv::init::config::OutputFormat::ZfpLow ||
             effective_config.output.format == tmv::init::config::OutputFormat::ZfpMedium ||
             effective_config.output.format == tmv::init::config::OutputFormat::ZfpHigh) &&
            !effective_config.radar.products.empty();
        const std::string cli_error = qualified_cli_scope_error(
            cli, effective_config.extensions.execution.mode == tmv::init::config::ExecutionMode::Qualified,
            admission.generated_metal,
            admission.generated_metal_east_npy,
            admission.generated_metal_mixed_npy,
            radar_zfp_requested,
            tmv::init::config::output_format_name(effective_config.output.format));
        if (!cli_error.empty())
        {
            std::cerr << "[VALIDATE] " << cli_error << std::endl;
            return 2;
        }
        if (duration_override >= 0)
        {
            effective_config.time.run_time = static_cast<double>(duration_override);
        }
        const auto projected = tmv::init::config::serialize_v1_to_v0_map(effective_config);
        const auto sounding  = validation_sounding_config(effective_config);
        const auto hodograph = validation_hodograph_config(effective_config);
        const auto trigger   = validation_trigger_config(effective_config);

        tmv::init::ValidationInputs inputs;
        inputs.scheme_id            = projected.at("dynamics.scheme");
        inputs.turbulence_scheme_id = projected.at("turbulence.scheme");
        inputs.coordinate           = effective_config.grid.coordinate_system;
        inputs.stagger              = effective_config.grid.staggering;
        inputs.sounding             = sounding;
        inputs.hodograph            = hodograph;
        inputs.trigger              = trigger;
        const auto compatibility    = tmv::init::validate_initial_condition_config(inputs);
        for (const auto& warning : compatibility.warnings)
        {
            std::cerr << "[VALIDATE WARNING] " << warning << std::endl;
        }
        if (!compatibility.ok)
        {
            for (const auto& error : compatibility.errors)
            {
                std::cerr << "[VALIDATE] " << error << std::endl;
            }
            return 2;
        }

        std::vector<double> z_m(static_cast<std::size_t>(effective_config.grid.nz));
        for (int k = 0; k < effective_config.grid.nz; ++k)
        {
            z_m[static_cast<std::size_t>(k)] = tmv::grid_coordinates::scalar_center(k, effective_config.grid.dz);
        }
        auto source = tmv::init::make_sounding_source(sounding);
        auto column = source->build(z_m, effective_config.grid.dz);
        column.verify_self_consistent();
        validate_domain_lifetime_preflight(effective_config, z_m, column, hodograph);

        std::string context_receipt;
        if (cli.preflight_context)
        {
            const std::string_view scope_kind = admission.generated_metal_mixed_npy ? "mixed_npy" :
                admission.generated_metal_east_npy ? "east_npy" : admission.generated_metal ? "canonical" : "none";
            context_receipt = tmv::runtime::serialize_validate_only_context_v1(
                effective_config, nimbus::crypto::sha256_hex(captured_source),
                scope_kind, tmv::runtime::runtime_state_fields());
        }
        std::cout << "[VALIDATE] PASS: strict nimbus.v1 load/projection; " << source->describe()
                  << " initial-condition column built on " << column.size()
                  << " model levels; dynamics/coordinate/staggering/source "
                  << "compatibility passed; no 3-D simulation fields allocated and no timestep run." << std::endl;
        if (cli.preflight_context)
        {
            std::cout << "[VALIDATE_CONTEXT] " << context_receipt << std::endl;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "[VALIDATE] initial-condition validation failed: " << error.what() << std::endl;
        return 2;
    }
}

bool apply_post_config_cli_overrides(RuntimeRunOptions& run)
{
    RuntimeCliOptions& cli       = run.cli;
    const bool         qualified = global_execution_mode == tmv::init::config::ExecutionMode::Qualified;
    const std::string_view format = run.output_config.format == OutputFormat::zfp
        ? "zfp_high" : output_format_name(run.output_config.format);
    const bool radar_zfp_requested = qualified &&
        run.output_config.format == OutputFormat::zfp &&
        (global_radar_product_selection.reflectivity ||
         global_radar_product_selection.radial_velocity ||
         global_radar_product_selection.zdr || global_radar_product_selection.kdp ||
         global_radar_product_selection.rhohv);
    const std::string cli_error = qualified_cli_scope_error(
        cli, qualified, global_simulation_storage.configuration.admission.generated_metal,
        global_simulation_storage.configuration.admission.generated_metal_east_npy,
        global_simulation_storage.configuration.admission.generated_metal_mixed_npy,
        radar_zfp_requested, format);
    if (!cli_error.empty())
    {
        report_cli_error(cli_error);
        return false;
    }
    if (cli.duration_from_cli)
    {
        cli.duration_s = cli.cli_duration_s;
    }
    if (cli.write_every_from_cli)
    {
        cli.write_every_s = cli.cli_write_every_s;
    }
    if (cli.acoustic_substeps_from_cli)
    {
        global_time_stepping_config.n_acoustic_substeps = cli.cli_acoustic_substeps;
    }
    if (cli.dt_large_from_cli)
    {
        dt = cli.cli_dt_large_s;
    }
    if (cli.log_profile_from_cli)
    {
        global_log_profile = cli.cli_log_profile;
    }
    if (cli.timing_from_cli)
    {
        global_perf_timing_enabled = cli.cli_perf_timing_enabled;
    }
    if (cli.cli_perf_report_every_steps >= 0)
    {
        global_perf_report_every_steps = cli.cli_perf_report_every_steps;
    }
    if (cli.guard_mode_from_cli)
    {
        global_validation_policy.mode = cli.cli_guard_mode;
    }
    if (cli.guard_fail_on_from_cli)
    {
        global_validation_policy.fail_on = cli.cli_guard_fail_on;
    }
    if (cli.guard_scope_from_cli)
    {
        global_validation_policy.strict_scope = cli.cli_guard_scope;
    }
    if (cli.guard_report_from_cli)
    {
        global_validation_report_path = cli.cli_guard_report_path;
    }
    if (cli.live_shm_from_cli)
    {
        run.live_shm = cli.cli_live_shm;
    }
    if (cli.live_shm_fields_from_cli)
    {
        run.live_shm_fields = cli.cli_live_shm_fields;
    }
    if (cli.live_shm_name_from_cli)
    {
        run.live_shm_name = cli.cli_live_shm_name;
    }

    // CLI guard mode is applied after YAML loading. Synchronize the overwrite
    // validation contract here as well so --guard-mode=strict enables poison
    // before initialization first acquires any scratch fields.
    FieldPool::instance().set_overwrite_poisoning(global_validation_policy.mode == tmv::GuardMode::Strict);
    const bool mutating_repairs = global_validation_policy.mode == tmv::GuardMode::Sanitize ||
                                  global_validation_policy.mode == tmv::GuardMode::Off;
    set_primary_state_repair_policy(mutating_repairs, global_validation_policy.mode == tmv::GuardMode::Strict);
    return true;
}

void apply_quiet_log_profile()
{
    if (global_log_profile == LogProfile::quiet)
    {
        std::cout.setstate(std::ios_base::failbit);
        std::clog.setstate(std::ios_base::failbit);
    }
}

void log_runtime_settings(const RuntimeRunOptions& run)
{
    const RuntimeCliOptions& cli = run.cli;
    if (cli.duration_from_cli || cli.write_every_from_cli)
    {
        if (log_normal_enabled())
        {
            std::cout << "[CLI OVERRIDE] duration=" << cli.duration_s << "s, write_every=" << cli.write_every_s << "s"
                      << std::endl;
        }
    }
    if (log_normal_enabled())
    {
        std::cout << "[RUN SETTINGS] log_profile=" << log_profile_name(global_log_profile)
                  << ", perf_timing=" << (global_perf_timing_enabled ? "on" : "off");
        if (cli.threads_from_cli)
        {
            std::cout << ", threads=" << cli.cli_threads << " (CLI override)";
        }
        if (cli.acoustic_substeps_from_cli)
        {
            std::cout << ", acoustic_substeps=" << cli.cli_acoustic_substeps << " (CLI override)";
        }
        if (cli.dt_large_from_cli)
        {
            std::cout << ", dt_large=" << cli.cli_dt_large_s << "s (CLI override)";
        }
        if (global_perf_report_every_steps > 0)
        {
            std::cout << ", timing_every_steps=" << global_perf_report_every_steps;
        }
        std::cout << ", guard_mode=" << tmv::to_string(global_validation_policy.mode)
                  << ", guard_fail_on=" << tmv::to_string(global_validation_policy.fail_on)
                  << ", guard_scope=" << tmv::to_string(global_validation_policy.strict_scope);
        if (!global_validation_report_path.empty())
        {
            std::cout << ", guard_report=" << global_validation_report_path;
        }
        std::cout << std::endl;
    }
}

std::string selected_dynamics_scheme_name()
{
    if (!global_dynamics_scheme_name.empty())
    {
        return global_dynamics_scheme_name;
    }
    return "klemp_wilhelmson_split_cartesian_cgrid";
}

bool initialize_initial_conditions()
{
    try
    {
        initialize();
        if (exact_two_way_nested_profile_selected())
        {
            std::fill(global_simulation_storage.fields.tracer.data(),
                      global_simulation_storage.fields.tracer.data() +
                          global_simulation_storage.fields.tracer.size(),
                      0.0f);
            std::fill(global_simulation_storage.fields.tke.data(),
                      global_simulation_storage.fields.tke.data() +
                          global_simulation_storage.fields.tke.size(),
                      0.0f);
            tmv::log_info(
                "[NESTING TWO-WAY] ", global_microphysics_scheme=="none"?"exact dry":"moist",
                " initialization suppresses the passive tracer marker and inactive TKE carrier");
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[CONFIG ERROR] Initial-condition build failed: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool initialize_runtime_modules(const std::string& dynamics_scheme_name)
{
    try
    {
        initialize_dynamics(dynamics_scheme_name);

        if (global_radiation_config.scheme_id.empty())
        {
            // Absence of an explicit radiation selection means no radiative
            // forcing. Never turn a missing value into an active physics
            // scheme with an assumed forcing contract.
            global_radiation_config.scheme_id = "none";
        }
        initialize_radiation(global_radiation_config.scheme_id, global_radiation_config);

        if (global_boundary_layer_config.scheme_id.empty())
        {
            global_boundary_layer_config.scheme_id = "slab";
        }
        initialize_boundary_layer(
            global_boundary_layer_config.scheme_id, global_boundary_layer_config, global_surface_config);

        if (global_turbulence_config.scheme_id.empty())
        {
            global_turbulence_config.scheme_id = "smagorinsky";
        }
        initialize_turbulence(global_turbulence_config.scheme_id, global_turbulence_config);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[CONFIG ERROR] Module initialization failed: " << e.what() << std::endl;
        return false;
    }

    return true;
}

bool initialize_simulation_runtime()
{
    // Normal startup resolves and admits the backend in load_config() before
    // allocating the 3-D state. Direct harnesses that bypass configuration
    // retain the historical lazy initialization here.
    if (active_compute_backend() == nullptr)
    {
        std::string compute_backend_error;
        if (!initialize_compute_backend_runtime(compute_backend_error))
        {
            std::cerr << "[COMPUTE] Failed to initialize runtime compute backend: " << compute_backend_error
                      << std::endl;
            return false;
        }
    }

    try
    {
        initialize_microphysics(global_microphysics_scheme, global_microphysics_config);
        initialize_radar_products(global_radar_product_selection);
        // Terrain geometry is an input to physical-height thermodynamic,
        // staggered-wind, and trigger initialization. Building it after
        // initialize() made an active strict-v1 request dereference an empty
        // metric set, while flat terrain happened to conceal the ordering
        // defect. Build the caller-owned geometry exactly once before any
        // initial-condition sampling.
        if (global_terrain_config.scheme_id.empty())
        {
            global_terrain_config.scheme_id = "none";
        }
        if (global_terrain_config.horizontal_placement_id == "earth_fixed")
        {
            tmv::EarthFixedTerrainBuildRequest request;
            request.nr = NR;
            request.nth = NTH;
            request.nz = NZ;
            request.dx_m = dr;
            request.dy_m = dtheta;
            request.grid_origin_earth_m = global_domain_frame.earth_origin_at(0.0);
            request.feature_center_earth_m = {
                global_terrain_config.feature_center_earth_x_m,
                global_terrain_config.feature_center_earth_y_m};
            if (global_simulation_storage.configuration.admission.earth_fixed_terrain_motion)
                request.sampling = tmv::EarthFixedTerrainSampling::CellAverageBilinear;
            initialize_earth_fixed_terrain(
                global_terrain_config.scheme_id,
                global_terrain_config,
                request);
        }
        else
        {
            initialize_terrain(
                global_terrain_config.scheme_id, global_terrain_config);
        }
    }
    catch (const std::exception& e)
    {
        tmv::log_error("[CONFIG ERROR] Scientific module selection failed: ", e.what());
        return false;
    }

    if (!initialize_initial_conditions())
    {
        return false;
    }

    try
    {
        initialize_numerics();
    }
    catch (const std::exception& e)
    {
        tmv::log_error("[CONFIG ERROR] Numerics initialization failed: ", e.what());
        return false;
    }

    if (global_chaos_config.scheme_id.empty())
    {
        global_chaos_config.scheme_id = "none";
    }
    initialize_chaos(global_simulation_state(), global_chaos_config);
    apply_chaos_initial_conditions(global_simulation_state());
    apply_initial_thermodynamic_closure(global_simulation_state());

    return initialize_runtime_modules(selected_dynamics_scheme_name());
}

void apply_internal_qualification_initial_state(const RuntimeCliOptions& cli)
{
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
    const int hook_count =
        static_cast<int>(cli.qualification_hydrometeor_pulse) +
        static_cast<int>(cli.qualification_milbrandt_mixed_phase) + static_cast<int>(cli.qualification_acoustic_wave) +
        static_cast<int>(cli.qualification_acoustic_standing_wave) + static_cast<int>(cli.qualification_acoustic_pulse);
    if (hook_count > 1)
    {
        throw std::runtime_error("qualification initial-state hooks are mutually exclusive");
    }
    if (hook_count == 0)
    {
        return;
    }
    SimulationStateView      state  = global_simulation_state();
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    if (domain.empty())
    {
        throw std::runtime_error("qualification hydrometeor pulse has an empty physical domain");
    }
    if (cli.qualification_milbrandt_mixed_phase)
    {
        const double     center_i = 0.5 * static_cast<double>(domain.i_begin + domain.i_end - 1);
        const double     center_j = 0.5 * static_cast<double>(domain.j_begin + domain.j_end - 1);
        const double     center_k = static_cast<double>(domain.k_begin) + 0.65 * static_cast<double>(domain.nz() - 1);
        constexpr double horizontal_sigma_cells = 1.8;
        constexpr double vertical_sigma_cells   = 2.2;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double di       = (static_cast<double>(i) - center_i) / horizontal_sigma_cells;
                    const double dj       = (static_cast<double>(j) - center_j) / horizontal_sigma_cells;
                    const double dk       = (static_cast<double>(k) - center_k) / vertical_sigma_cells;
                    const double envelope = std::exp(-0.5 * (di * di + dj * dj + dk * dk));
                    if (envelope < 1.0e-5)
                        continue;
                    const double pressure = static_cast<double>(state.fields.p[i][j][k]);
                    const double temperature =
                        tmv::thermodynamics::theta_to_temperature(state.fields.theta[i][j][k], pressure);
                    const double saturation = tmv::thermodynamics::saturation_mixing_ratio_water(temperature, pressure);
                    state.fields.qv[i][j][k] =
                        static_cast<float>(std::max(static_cast<double>(state.fields.qv[i][j][k]), saturation));
                    state.fields.qc[i][j][k] = static_cast<float>(6.0e-4 * envelope);
                    state.fields.qr[i][j][k] = static_cast<float>(5.0e-4 * envelope);
                    state.fields.qi[i][j][k] = static_cast<float>(1.5e-4 * envelope);
                    state.fields.qs[i][j][k] = static_cast<float>(3.0e-4 * envelope);
                    state.fields.qg[i][j][k] = static_cast<float>(2.5e-4 * envelope);
                    state.fields.qh[i][j][k] = static_cast<float>(1.0e-4 * envelope);
                }
            }
        }
        if (state.advection_runtime != nullptr)
        {
            state.advection_runtime->publish_hydrometeor_activity(
                {true, true, true, true, true, true}, state.grid.nr, state.grid.nth, state.grid.nz);
        }
        tmv::log_warn("[INTERNAL QUALIFICATION] seeded a smooth six-category "
                      "Milbrandt mixed-phase volume; this artificial state is not a "
                      "user-facing configuration feature");
        return;
    }
    if (cli.qualification_acoustic_wave || cli.qualification_acoustic_standing_wave || cli.qualification_acoustic_pulse)
    {
        const double     domain_length_m               = static_cast<double>(domain.nx()) * state.grid.dr;
        constexpr double fractional_pressure_amplitude = 1.0e-4;
        constexpr double two_pi                        = 6.28318530717958647692;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            const double x_m                       = (static_cast<double>(i - domain.i_begin) + 0.5) * state.grid.dr;
            const double normalized_pulse_distance = (x_m - 0.25 * domain_length_m) / (2.0 * state.grid.dr);
            const double fraction = cli.qualification_acoustic_pulse
                                        ? fractional_pressure_amplitude *
                                              std::exp(-0.5 * normalized_pulse_distance * normalized_pulse_distance)
                                        : fractional_pressure_amplitude * std::cos(two_pi * x_m / domain_length_m);
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double p_base      = state.base.p0[static_cast<std::size_t>(k)];
                    const double rho_base    = state.base.rho0[static_cast<std::size_t>(k)];
                    const double sound_speed = std::sqrt(dynamics_constants::gamma * p_base / rho_base);
                    state.fields.p[i][j][k]  = static_cast<float>(p_base * (1.0 + fraction));
                    state.fields.rho[i][j][k] =
                        static_cast<float>(rho_base * std::pow(1.0 + fraction, 1.0 / dynamics_constants::gamma));
                    state.fields.u[i][j][k] = cli.qualification_acoustic_standing_wave
                                                  ? 0.0f
                                                  : static_cast<float>(p_base * fraction / (rho_base * sound_speed));
                }
            }
        }
        tmv::log_warn(cli.qualification_acoustic_pulse
                          ? "[INTERNAL QUALIFICATION] seeded a nonlinear-EOS-consistent right-going acoustic pulse; "
                      : cli.qualification_acoustic_standing_wave
                          ? "[INTERNAL QUALIFICATION] seeded a nonlinear-EOS-consistent standing acoustic wave; "
                          : "[INTERNAL QUALIFICATION] seeded a nonlinear-EOS-consistent right-going acoustic wave; ",
                      "this artificial initial state is not a user-facing configuration feature");
        return;
    }
    const double     center_i     = static_cast<double>(domain.i_begin) + 0.35 * static_cast<double>(domain.nx() - 1);
    const double     center_j     = 0.5 * static_cast<double>(domain.j_begin + domain.j_end - 1);
    const double     center_k     = 0.5 * static_cast<double>(domain.k_begin + domain.k_end - 1);
    constexpr double sigma_cells  = 1.25;
    constexpr float  peak_qr_kgkg = 1.0e-3f;
    constexpr float  uniform_u_ms = 10.0f;
    std::fill(state.fields.u.data(), state.fields.u.data() + state.fields.u.size(), uniform_u_ms);
    std::fill(state.fields.v.data(), state.fields.v.data() + state.fields.v.size(), 0.0f);
    std::fill(state.base.u0.begin(), state.base.u0.end(), static_cast<double>(uniform_u_ms));
    std::fill(state.base.v0.begin(), state.base.v0.end(), 0.0);
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double di          = (static_cast<double>(i) - center_i) / sigma_cells;
                const double dj          = (static_cast<double>(j) - center_j) / sigma_cells;
                const double dk          = (static_cast<double>(k) - center_k) / sigma_cells;
                state.fields.qr[i][j][k] = static_cast<float>(static_cast<double>(peak_qr_kgkg) *
                                                              std::exp(-0.5 * (di * di + dj * dj + dk * dk)));
            }
        }
    }
    if (state.advection_runtime != nullptr)
    {
        state.advection_runtime->publish_hydrometeor_activity(
            {false, true, false, false, false, false}, state.grid.nr, state.grid.nth, state.grid.nz);
    }
    tmv::log_warn("[INTERNAL QUALIFICATION] seeded a nonzero rain-water pulse in a 10 m/s eastward base flow; "
                  "this artificial initial state is not a user-facing configuration feature");
#else
    (void)cli;
#endif
}

HeadlessRunOptions make_headless_options(const RuntimeRunOptions& run)
{
    const RuntimeCliOptions& cli = run.cli;
    HeadlessRunOptions       headless_options;
    headless_options.export_ms                          = cli.export_ms;
    headless_options.duration_s                         = cli.duration_s;
    headless_options.write_every_s                      = cli.write_every_s;
    headless_options.scientific_diagnostics_every_steps = global_scientific_diagnostics_every_steps;
    headless_options.outdir                             = cli.outdir;
    headless_options.output_config                      = run.output_config;
    headless_options.checkpoint_every_s =
        cli.checkpoint_every_from_cli ? cli.checkpoint_every_s : global_checkpoint_every_s;
    headless_options.checkpoint_keep_last = global_checkpoint_keep_last;
    headless_options.checkpoint_minimum_free_space_mib = global_checkpoint_minimum_free_space_mib;
    headless_options.checkpoint_dir =
        cli.checkpoint_dir_from_cli ? cli.checkpoint_dir : std::filesystem::path(global_checkpoint_directory);
    headless_options.resume_checkpoint =
        cli.resume_checkpoint_from_cli ? cli.resume_checkpoint : std::filesystem::path(global_checkpoint_resume_from);
    headless_options.live_shm           = run.live_shm;
    headless_options.live_shm_fields    = run.live_shm_fields;
    headless_options.live_shm_name      = run.live_shm_name;
    headless_options.live_shm_interval_s =
        run.output_config.live_shm_interval_s;
    headless_options.live_shm_full_rate =
        run.output_config.live_shm_full_rate;
    headless_options.trajectory.enabled = cli.traj_enabled_from_cli ? cli.traj_enabled : run.output_config.trajectories;
    headless_options.trajectory.seed_time_s =
        cli.traj_seed_time_from_cli ? cli.traj_seed_time_s : run.output_config.trajectory_seed_time_s;
    const int trajectory_count     = cli.traj_count_from_cli ? cli.traj_count : run.output_config.trajectory_count;
    headless_options.trajectory.nx = trajectory_count;
    headless_options.trajectory.ny = trajectory_count;
    headless_options.trajectory.sample_interval_s =
        run.output_config.trajectory_sample_interval_s;
    headless_options.trajectory.max_samples_per_parcel =
        run.output_config.trajectory_max_samples_per_parcel;
    return headless_options;
}

} // namespace

/**
 * @brief Program entry point.
 * @param argc CLI argument count.
 * @param argv CLI argument vector.
 * @return Zero on success, non-zero on configuration/runtime failure.
 */
int main(int argc, char** argv)
{
    if (const MigrationCommandResult metadata = maybe_handle_cli_metadata(argc, argv); metadata.handled)
    {
        return metadata.exit_code;
    }

    if (const MigrationCommandResult migration = maybe_run_config_migration(argc, argv); migration.handled)
    {
        return migration.exit_code;
    }

    RuntimeRunOptions run;
    apply_environment_overrides();
    if (!parse_runtime_cli(argc, argv, run.cli))
    {
        return 1;
    }
    if (!apply_thread_cli_override(run.cli))
    {
        return 1;
    }
    if (run.cli.config_path.empty() && !run.cli.sounding_path_from_cli)
    {
        report_cli_error("Missing required --config PATH (or use --sounding PATH).");
        return 1;
    }
    if (run.cli.config_path.empty() && run.cli.sounding_path_from_cli)
    {
        run.cli.config_path = "configs/1000meterResolution/kessler/baseline.yaml";
        tmv::log_info("[CLI] --sounding given without --config; using ", run.cli.config_path, " as the base preset.");
    }
    if (run.cli.validate_only)
    {
#if defined(NIMBUS_INTERNAL_AGRID_QUALIFICATION)
        set_internal_moving_nest_qualification(
            run.cli.qualification_moving_nest);
#endif
        return validate_config_without_runtime_allocation(
            run.cli.config_path,
            run.cli.compute_backend_from_cli ? &run.cli.cli_compute_backend : nullptr,
            run.cli.sounding_path_from_cli ? &run.cli.cli_sounding_path : nullptr,
            run.cli.duration_from_cli ? run.cli.cli_duration_s : -1,
            run.cli);
    }
    apply_pre_config_cli_overrides(run.cli);

    if (!load_runtime_config(run))
    {
        return 1;
    }
    if (!apply_post_config_cli_overrides(run))
    {
        shutdown_compute_backend_runtime();
        return 1;
    }
    apply_quiet_log_profile();
    log_runtime_settings(run);

    tmv::reset_numerical_nonfinite_incidents();
    if (!initialize_simulation_runtime())
    {
        shutdown_compute_backend_runtime();
        return 1;
    }
    if (tmv::numerical_nonfinite_incidents() > 0)
    {
        tmv::log_error("[NUMERICAL INTEGRITY] initialization detected ",
                       tmv::numerical_nonfinite_incidents(),
                       " non-finite incidents");
        shutdown_compute_backend_runtime();
        return 2;
    }
    try
    {
        apply_internal_qualification_initial_state(run.cli);
        // Initialization and internal qualification hooks author the physical
        // state, but boundary-owned faces/halos must already satisfy the
        // selected closure before t=0 is published or used as the first
        // conservation baseline. Waiting until the end of the first dynamics
        // step creates a spurious startup impulse and makes the initial frame
        // inconsistent with every subsequently committed state.
        apply_boundary_conditions(global_simulation_state());
    }
    catch (const std::exception& error)
    {
        tmv::log_error("[INTERNAL QUALIFICATION] state injection failed: ", error.what());
        shutdown_compute_backend_runtime();
        return 2;
    }

    int run_status = 0;
    if (run.cli.headless)
    {
        tmv::runtime::ScopedTerminationSignalHandlers termination_handlers;
        if (!termination_handlers.installed())
        {
            tmv::log_error("[RUNTIME] ", termination_handlers.error_message());
            shutdown_compute_backend_runtime();
            return 1;
        }
        run_status = run_headless_simulation(make_headless_options(run));
    }

    shutdown_compute_backend_runtime();
    return run_status;
}
