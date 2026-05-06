/**
 * @file tornado_sim.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include <vector>
#include <string>
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
#include "core/runtime/simulation.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "chaos/chaos_base.hpp"
#include "compute/compute_backend.hpp"
#include "init/scheme_profile.hpp"
#include "util/log.hpp"
#include "radiation/radiation_base.hpp"
#include "terrain/terrain_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "core/runtime/headless_runtime.hpp"
#include "core/runtime/runtime_config.hpp"
#include "numerics/advection/advection.hpp"
#include "diagnostics/field_contract.hpp"
#include "diagnostics/field_validation.hpp"
#include "data/soundings.hpp"
#include "util/string_utils.hpp"



extern void initialize_radar(const std::string& scheme_name);


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
        u = profile.u_sfc + frac * (profile.u_1km - profile.u_sfc);
        v = profile.v_sfc + frac * (profile.v_1km - profile.v_sfc);
    } 

    else if (z <= z_6km) 
    {
        double frac = (z - z_1km) / (z_6km - z_1km);
        u = profile.u_1km + frac * (profile.u_6km - profile.u_1km);
        v = profile.v_1km + frac * (profile.v_6km - profile.v_1km);
    } 
    else 
    {
        u = profile.u_6km;
        v = profile.v_6km;
    }
}

/**
 * @brief Program entry point.
 * @param argc CLI argument count.
 * @param argv CLI argument vector.
 * @return Zero on success, non-zero on configuration/runtime failure.
 */
int main(int argc, char** argv) 
{
    bool headless = false;
    int export_ms = 0;
    int duration_s = -1;
    int write_every_s = 0;
    bool duration_from_cli = false;
    bool write_every_from_cli = false;
    int cli_duration_s = -1;
    int cli_write_every_s = 0;
    bool log_profile_from_cli = false;
    LogProfile cli_log_profile = LogProfile::normal;
    bool timing_from_cli = false;
    bool cli_perf_timing_enabled = false;
    int cli_perf_report_every_steps = -1;
    bool guard_mode_from_cli = false;
    tmv::GuardMode cli_guard_mode = global_validation_policy.mode;
    bool guard_fail_on_from_cli = false;
    tmv::GuardFailOn cli_guard_fail_on = global_validation_policy.fail_on;
    bool guard_scope_from_cli = false;
    tmv::StrictGuardScope cli_guard_scope = global_validation_policy.strict_scope;
    bool guard_report_from_cli = false;
    std::string cli_guard_report_path;
    bool compute_backend_from_cli = false;
    std::string cli_compute_backend;
    bool live_shm_from_cli = false;
    bool cli_live_shm = false;
    bool live_shm_fields_from_cli = false;
    std::vector<std::string> cli_live_shm_fields;
    std::string outdir = "data/exports";
    std::string config_path = "";
    std::string cli_sounding_path;
    bool sounding_path_from_cli = false;

    if (const char* env_log_profile = std::getenv("TORNADO_LOG_PROFILE"))
    {
        bool valid = false;
        const LogProfile parsed = parse_log_profile(env_log_profile, &valid);
        if (valid)
        {
            global_log_profile = parsed;
        }
        else
        {
            std::cerr << "Warning: Invalid TORNADO_LOG_PROFILE '" << env_log_profile
                      << "'. Valid values: quiet, normal, debug." << std::endl;
        }
    }
    if (const char* env_perf = std::getenv("TORNADO_PERF_TIMING"))
    {
        global_perf_timing_enabled = parse_bool_value(env_perf);
    }
    if (const char* env_perf_every = std::getenv("TORNADO_PERF_EVERY_STEPS"))
    {
        int parsed = 0;
        if (try_parse_non_negative_int_value(env_perf_every, parsed))
        {
            global_perf_report_every_steps = parsed;
        }
        else
        {
            std::cerr << "Warning: Invalid TORNADO_PERF_EVERY_STEPS '" << env_perf_every
                      << "'. Expected a non-negative integer." << std::endl;
        }
    }

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--headless") headless = true;
        else if (arg.rfind("--export-ms=", 0) == 0)
        {
            int parsed = 0;
            const std::string value = arg.substr(12);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                std::cerr << "Invalid --export-ms value '" << value
                          << "'. Expected a non-negative integer." << std::endl;
                return 1;
            }
            export_ms = parsed;
        }
        else if (arg.rfind("--duration=", 0) == 0)
        {
            int parsed = 0;
            const std::string value = arg.substr(11);
            if (!try_parse_int_value(value, parsed))
            {
                std::cerr << "Invalid --duration value '" << value
                          << "'. Expected an integer." << std::endl;
                return 1;
            }
            duration_s = parsed;
            duration_from_cli = true;
            cli_duration_s = duration_s;
        }
        else if (arg == "--duration" && i + 1 < argc)
        {
            int parsed = 0;
            const std::string value = argv[++i];
            if (!try_parse_int_value(value, parsed))
            {
                std::cerr << "Invalid --duration value '" << value
                          << "'. Expected an integer." << std::endl;
                return 1;
            }
            duration_s = parsed;
            duration_from_cli = true;
            cli_duration_s = duration_s;
        }
        else if (arg.rfind("--write-every=", 0) == 0)
        {
            int parsed = 0;
            const std::string value = arg.substr(14);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                std::cerr << "Invalid --write-every value '" << value
                          << "'. Expected a non-negative integer." << std::endl;
                return 1;
            }
            write_every_s = parsed;
            write_every_from_cli = true;
            cli_write_every_s = write_every_s;
        }
        else if (arg.rfind("--outdir=", 0) == 0)
        {
            outdir = arg.substr(9);
        }
        else if (arg.rfind("--config=", 0) == 0)
        {
            config_path = arg.substr(9);
        }
        else if (arg == "--config" && i + 1 < argc)
        {
            config_path = argv[++i];
        }
        else if (arg.rfind("--sounding=", 0) == 0)
        {
            cli_sounding_path = arg.substr(11);
            sounding_path_from_cli = true;
        }
        else if (arg == "--sounding" && i + 1 < argc)
        {
            cli_sounding_path = argv[++i];
            sounding_path_from_cli = true;
        }
        else if (arg.rfind("--log-profile=", 0) == 0)
        {
            bool valid = false;
            cli_log_profile = parse_log_profile(arg.substr(14), &valid);
            if (!valid)
            {
                std::cerr << "Invalid --log-profile value. Use quiet, normal, or debug." << std::endl;
                return 1;
            }
            log_profile_from_cli = true;
        }
        else if (arg == "--log-profile" && i + 1 < argc)
        {
            bool valid = false;
            cli_log_profile = parse_log_profile(argv[++i], &valid);
            if (!valid)
            {
                std::cerr << "Invalid --log-profile value. Use quiet, normal, or debug." << std::endl;
                return 1;
            }
            log_profile_from_cli = true;
        }
        else if (arg == "--timing")
        {
            timing_from_cli = true;
            cli_perf_timing_enabled = true;
        }
        else if (arg == "--no-timing")
        {
            timing_from_cli = true;
            cli_perf_timing_enabled = false;
        }
        else if (arg.rfind("--timing-every=", 0) == 0)
        {
            int parsed = 0;
            const std::string value = arg.substr(15);
            if (!try_parse_non_negative_int_value(value, parsed))
            {
                std::cerr << "Invalid --timing-every value '" << value
                          << "'. Expected a non-negative integer." << std::endl;
                return 1;
            }
            cli_perf_report_every_steps = parsed;
            if (cli_perf_report_every_steps > 0)
            {
                timing_from_cli = true;
                cli_perf_timing_enabled = true;
            }
        }
        else if (arg.rfind("--guard-mode=", 0) == 0)
        {
            tmv::GuardMode parsed = cli_guard_mode;
            if (!tmv::parse_guard_mode(arg.substr(13), parsed))
            {
                std::cerr << "Invalid --guard-mode value. Use off, sanitize, or strict." << std::endl;
                return 1;
            }
            cli_guard_mode = parsed;
            guard_mode_from_cli = true;
        }
        else if (arg == "--guard-mode" && i + 1 < argc)
        {
            tmv::GuardMode parsed = cli_guard_mode;
            if (!tmv::parse_guard_mode(argv[++i], parsed))
            {
                std::cerr << "Invalid --guard-mode value. Use off, sanitize, or strict." << std::endl;
                return 1;
            }
            cli_guard_mode = parsed;
            guard_mode_from_cli = true;
        }
        else if (arg.rfind("--guard-fail-on=", 0) == 0)
        {
            tmv::GuardFailOn parsed = cli_guard_fail_on;
            if (!tmv::parse_guard_fail_on(arg.substr(16), parsed))
            {
                std::cerr << "Invalid --guard-fail-on value. Use nonfinite, bounds, or both." << std::endl;
                return 1;
            }
            cli_guard_fail_on = parsed;
            guard_fail_on_from_cli = true;
        }
        else if (arg == "--guard-fail-on" && i + 1 < argc)
        {
            tmv::GuardFailOn parsed = cli_guard_fail_on;
            if (!tmv::parse_guard_fail_on(argv[++i], parsed))
            {
                std::cerr << "Invalid --guard-fail-on value. Use nonfinite, bounds, or both." << std::endl;
                return 1;
            }
            cli_guard_fail_on = parsed;
            guard_fail_on_from_cli = true;
        }
        else if (arg.rfind("--guard-scope=", 0) == 0)
        {
            tmv::StrictGuardScope parsed = cli_guard_scope;
            if (!tmv::parse_strict_guard_scope(arg.substr(14), parsed))
            {
                std::cerr << "Invalid --guard-scope value. Use required or exported." << std::endl;
                return 1;
            }
            cli_guard_scope = parsed;
            guard_scope_from_cli = true;
        }
        else if (arg == "--guard-scope" && i + 1 < argc)
        {
            tmv::StrictGuardScope parsed = cli_guard_scope;
            if (!tmv::parse_strict_guard_scope(argv[++i], parsed))
            {
                std::cerr << "Invalid --guard-scope value. Use required or exported." << std::endl;
                return 1;
            }
            cli_guard_scope = parsed;
            guard_scope_from_cli = true;
        }
        else if (arg.rfind("--guard-report=", 0) == 0)
        {
            cli_guard_report_path = arg.substr(15);
            guard_report_from_cli = true;
        }
        else if (arg == "--guard-report" && i + 1 < argc)
        {
            cli_guard_report_path = argv[++i];
            guard_report_from_cli = true;
        }
        else if (arg.rfind("--compute-backend=", 0) == 0)
        {
            const std::string value = arg.substr(18);
            ComputeBackendKind parsed = ComputeBackendKind::Cpu;
            if (!parse_compute_backend_kind(value, parsed))
            {
                std::cerr << "Invalid --compute-backend value '" << value
                          << "'. Expected: cpu, vulkan." << std::endl;
                return 1;
            }
            cli_compute_backend = compute_backend_kind_name(parsed);
            compute_backend_from_cli = true;
        }
        else if (arg == "--compute-backend" && i + 1 < argc)
        {
            const std::string value = argv[++i];
            ComputeBackendKind parsed = ComputeBackendKind::Cpu;
            if (!parse_compute_backend_kind(value, parsed))
            {
                std::cerr << "Invalid --compute-backend value '" << value
                          << "'. Expected: cpu, vulkan." << std::endl;
                return 1;
            }
            cli_compute_backend = compute_backend_kind_name(parsed);
            compute_backend_from_cli = true;
        }
        else if (arg == "--live-shm")
        {
            live_shm_from_cli = true;
            cli_live_shm = true;
        }
        else if (arg.rfind("--live-shm-fields=", 0) == 0)
        {
            const std::string value = arg.substr(18);
            cli_live_shm_fields.clear();
            std::istringstream stream(value);
            std::string token;
            while (std::getline(stream, token, ','))
            {
                auto start = token.find_first_not_of(" \t");
                auto end = token.find_last_not_of(" \t");
                if (start != std::string::npos)
                {
                    cli_live_shm_fields.push_back(token.substr(start, end - start + 1));
                }
            }
            if (!cli_live_shm_fields.empty())
            {
                live_shm_fields_from_cli = true;
            }
        }
    }

    if (log_profile_from_cli)
    {
        global_log_profile = cli_log_profile;
    }

    bool live_shm = false;
    std::vector<std::string> live_shm_fields = {"w", "reflectivity_dbz", "vorticity_z"};
    OutputConfig output_config;

    // --sounding without --config: load the drag-and-drop preset (sensible
    // supercell defaults) so the user gets a runnable scenario from one
    // CLI flag. Explicit --config wins as usual.
    if (sounding_path_from_cli && config_path.empty())
    {
        config_path = "configs/dragdrop.yaml";
        std::cout << "[CLI] --sounding given without --config; using "
                  << config_path << " as the base preset." << std::endl;
    }

    try
    {
        load_config(config_path, duration_s, write_every_s, outdir, &output_config,
                    &live_shm, &live_shm_fields);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[CONFIG] " << e.what() << std::endl;
        return 1;
    }

    // --sounding override: applied AFTER load_config so it wins over any
    // environment.sounding.* key in the YAML. Switch the sounding source
    // to File with the supplied path; require_winds=true so a malformed
    // file fails loudly rather than silently falling back to parametric.
    if (sounding_path_from_cli)
    {
        global_sounding_source_config.type =
            tmv::init::SoundingSourceConfig::Type::File;
        global_sounding_source_config.file.path = cli_sounding_path;
        if (global_sounding_source_config.file.scheme_id.empty()
            || global_sounding_source_config.file.scheme_id == "none")
        {
            global_sounding_source_config.file.scheme_id = "sharpy";
        }
        global_sounding_source_config.file.require_winds = true;
        global_sounding_source_config.file.use_fallback_profiles = false;
        global_sounding_enabled = false;  // disable any legacy overlay path
        std::cout << "[CLI] sounding override: type=file, path="
                  << cli_sounding_path << std::endl;
    }
    if (duration_from_cli)
    {
        duration_s = cli_duration_s;
    }
    if (write_every_from_cli)
    {
        write_every_s = cli_write_every_s;
    }
    if (log_profile_from_cli)
    {
        global_log_profile = cli_log_profile;
    }
    if (timing_from_cli)
    {
        global_perf_timing_enabled = cli_perf_timing_enabled;
    }
    if (cli_perf_report_every_steps >= 0)
    {
        global_perf_report_every_steps = cli_perf_report_every_steps;
    }
    if (guard_mode_from_cli)
    {
        global_validation_policy.mode = cli_guard_mode;
    }
    if (guard_fail_on_from_cli)
    {
        global_validation_policy.fail_on = cli_guard_fail_on;
    }
    if (guard_scope_from_cli)
    {
        global_validation_policy.strict_scope = cli_guard_scope;
    }
    if (guard_report_from_cli)
    {
        global_validation_report_path = cli_guard_report_path;
    }
    if (compute_backend_from_cli)
    {
        global_compute_backend_config.backend = cli_compute_backend;
    }
    if (live_shm_from_cli)
    {
        live_shm = cli_live_shm;
    }
    if (live_shm_fields_from_cli)
    {
        live_shm_fields = cli_live_shm_fields;
    }

    if (global_log_profile == LogProfile::quiet)
    {
        std::cout.setstate(std::ios_base::failbit);
        std::clog.setstate(std::ios_base::failbit);
    }

    if (duration_from_cli || write_every_from_cli)
    {
        if (log_normal_enabled())
        {
            std::cout << "[CLI OVERRIDE] duration=" << duration_s
                      << "s, write_every=" << write_every_s << "s" << std::endl;
        }
    }
    if (log_normal_enabled())
    {
        std::cout << "[RUN SETTINGS] log_profile=" << log_profile_name(global_log_profile)
                  << ", perf_timing=" << (global_perf_timing_enabled ? "on" : "off");
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

    std::string compute_backend_error;
    if (!initialize_compute_backend_runtime(compute_backend_error))
    {
        std::cerr << "[COMPUTE] Failed to initialize runtime compute backend: "
                  << compute_backend_error << std::endl;
        return 1;
    }

    initialize_microphysics(global_microphysics_scheme);

    initialize_radar("reflectivity");

    // initialize() now performs the full IC build via the SoundingSource +
    // HodographSource + TriggerSource factories: the file-based path that
    // used to live as a post-init overlay (apply_soundings_to_initial_state)
    // runs inside initialize() through FileSoundingSource. Legacy YAML that
    // sets only environment.sounding.scheme_id is auto-promoted to
    // type=file in load_config(), so existing configs keep working.
    //
    // Wrap in a catch so a malformed sounding file (or any other config
    // problem the IC pipeline detects on first build) produces a clean
    // [CONFIG ERROR] message + exit 1 rather than a libc++abi terminate.
    try
    {
        initialize();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[CONFIG ERROR] Initial-condition build failed: "
                  << e.what() << std::endl;
        return 1;
    }

    initialize_numerics();

    if (global_chaos_config.scheme_id.empty()) 
    {
        global_chaos_config.scheme_id = "none";
    }
    initialize_chaos(global_chaos_config);

    apply_chaos_initial_conditions();

    std::string dynamics_scheme_name = "tornado";
    if (!global_dynamics_scheme_name.empty())
    {
        dynamics_scheme_name = global_dynamics_scheme_name;
    }

    // Validate the parsed config against the active scheme's IC profile.
    // This replaces the previously-hardcoded cartesian-vs-cylindrical
    // check; the registry in src/init/scheme_profile.cpp now owns all
    // per-scheme requirements (coordinate, stagger, allowed sounding /
    // hodograph / trigger types, recommended trigger, shear sanity).
    {
        tmv::init::ValidationInputs vi;
        vi.scheme_id = dynamics_scheme_name;
        vi.coordinate = global_coordinate_system;
        vi.stagger = global_stagger_type;
        vi.sounding = global_sounding_source_config;
        vi.hodograph = global_hodograph_source_config;
        vi.trigger = global_trigger_source_config;

        const auto report = tmv::init::validate_initial_condition_config(vi);
        for (const auto& w : report.warnings)
        {
            tmv::log_warn(w);
        }
        if (!report.ok)
        {
            for (const auto& e : report.errors)
            {
                std::cerr << e << std::endl;
            }
            return 1;
        }
    }

    initialize_dynamics(dynamics_scheme_name);

    if (global_radiation_config.scheme_id.empty()) 
    {
        global_radiation_config.scheme_id = "simple_grey";
    }
    initialize_radiation(global_radiation_config.scheme_id, global_radiation_config);

    if (global_boundary_layer_config.scheme_id.empty()) 
    {
        global_boundary_layer_config.scheme_id = "slab";
    }
    initialize_boundary_layer(global_boundary_layer_config.scheme_id, global_boundary_layer_config, global_surface_config);

    if (global_turbulence_config.scheme_id.empty()) 
    {
        global_turbulence_config.scheme_id = "smagorinsky";
    }
    initialize_turbulence(global_turbulence_config.scheme_id, global_turbulence_config);

    if (global_terrain_config.scheme_id.empty()) 
    {
        global_terrain_config.scheme_id = "none";
    }
    initialize_terrain(global_terrain_config.scheme_id, global_terrain_config);

    int run_status = 0;
    if (headless)
    {
        HeadlessRunOptions headless_options;
        headless_options.export_ms = export_ms;
        headless_options.duration_s = duration_s;
        headless_options.write_every_s = write_every_s;
        headless_options.outdir = outdir;
        headless_options.output_config = output_config;
        headless_options.live_shm = live_shm;
        headless_options.live_shm_fields = live_shm_fields;
        run_status = run_headless_simulation(headless_options);
    }

    shutdown_compute_backend_runtime();
    return run_status;
}
