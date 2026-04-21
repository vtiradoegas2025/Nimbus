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
#include "core/simulation.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "chaos/chaos_base.hpp"
#include "compute/compute_backend.hpp"
#include "radiation/radiation_base.hpp"
#include "terrain/terrain_base.hpp"
#include "turbulence/turbulence_base.hpp"
#include "core/headless_runtime.hpp"
#include "core/runtime_config.hpp"
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

namespace
{
constexpr double kPi = 3.14159265358979323846;
}

/**
 * @brief Applies sounding-derived thermodynamic and wind profiles to state fields.
 * @return True when sounding data was successfully applied.
 */
bool apply_soundings_to_initial_state()
{
    if (!global_sounding_enabled)
    {
        return false;
    }

    SoundingConfig runtime_cfg = global_runtime_sounding_config;
    runtime_cfg.scheme_id = to_lower_copy(runtime_cfg.scheme_id);

    if (runtime_cfg.scheme_id.empty())
    {
        runtime_cfg.scheme_id = "sharpy";
    }
    if (runtime_cfg.scheme_id == "none")
    {
        if (log_normal_enabled())
        {
            std::cout << "[SOUNDINGS] environment.sounding.enabled=true but scheme is 'none'; skipping sounding initialization."
                      << std::endl;
        }
        return false;
    }

    try
    {
        initialize_soundings(runtime_cfg);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[SOUNDINGS] Failed to initialize sounding scheme: " << e.what()
                  << ". Keeping procedural initialization." << std::endl;
        return false;
    }

    try
    {
        SoundingData source = load_sounding_data();
        if (!source.is_valid())
        {
            std::cerr << "[SOUNDINGS] No valid sounding profile loaded; keeping procedural initialization." << std::endl;
            reset_soundings();
            return false;
        }

        const bool placeholder_profile = (source.station_id == "KSAMPLE");
        if (placeholder_profile && !global_sounding_allow_placeholder_profiles)
        {
            std::cerr << "[SOUNDINGS] Placeholder sample sounding detected (station_id=KSAMPLE). "
                      << "Set environment.sounding.allow_placeholder_profiles=true to apply it. "
                      << "Keeping procedural initialization." << std::endl;
            reset_soundings();
            return false;
        }

        std::vector<double> model_heights(static_cast<std::size_t>(NZ), 0.0);
        for (int k = 0; k < NZ; ++k)
        {
            model_heights[static_cast<std::size_t>(k)] = static_cast<double>(k) * dz;
        }

        SoundingData interp = interpolate_sounding_to_grid(source, model_heights);
        if (!interp.is_valid())
        {
            std::cerr << "[SOUNDINGS] Interpolated sounding profile is invalid; keeping procedural initialization."
                      << std::endl;
            reset_soundings();
            return false;
        }

        std::vector<float> theta_baseline(static_cast<std::size_t>(NZ), std::numeric_limits<float>::max());
        std::vector<float> qv_baseline(static_cast<std::size_t>(NZ), std::numeric_limits<float>::max());

        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                for (int k = 0; k < NZ; ++k)
                {
                    const std::size_t kz = static_cast<std::size_t>(k);
                    theta_baseline[kz] = std::min(theta_baseline[kz], static_cast<float>(theta[i][j][k]));
                    qv_baseline[kz] = std::min(qv_baseline[kz], static_cast<float>(qv[i][j][k]));
                }
            }
        }

        for (int k = 0; k < NZ; ++k)
        {
            const std::size_t kz = static_cast<std::size_t>(k);
            if (!std::isfinite(theta_baseline[kz]) || theta_baseline[kz] == std::numeric_limits<float>::max())
            {
                theta_baseline[kz] = static_cast<float>(theta[0][0][k]);
            }
            if (!std::isfinite(qv_baseline[kz]) || qv_baseline[kz] == std::numeric_limits<float>::max())
            {
                qv_baseline[kz] = static_cast<float>(qv[0][0][k]);
            }
        }

        std::vector<float> theta_profile(static_cast<std::size_t>(NZ), 0.0f);
        std::vector<float> qv_profile(static_cast<std::size_t>(NZ), 0.0f);
        std::vector<double> u_cart_profile(static_cast<std::size_t>(NZ), 0.0);
        std::vector<double> v_cart_profile(static_cast<std::size_t>(NZ), 0.0);

        const bool has_theta_profile = (interp.potential_temperature_k.size() == model_heights.size());
        const bool has_qv_profile = (interp.mixing_ratio_kgkg.size() == model_heights.size());
        const bool has_wind_profile =
            (interp.wind_speed_ms.size() == model_heights.size() &&
             interp.wind_direction_deg.size() == model_heights.size());

        int theta_levels_from_sounding = 0;
        int qv_levels_from_sounding = 0;
        int wind_levels_from_sounding = 0;

        for (int k = 0; k < NZ; ++k)
        {
            const std::size_t kz = static_cast<std::size_t>(k);

            double theta_target = std::numeric_limits<double>::quiet_NaN();
            if (has_theta_profile && std::isfinite(interp.potential_temperature_k[kz]))
            {
                theta_target = interp.potential_temperature_k[kz];
            }
            else if (interp.temperature_k.size() == model_heights.size() &&
                     interp.pressure_hpa.size() == model_heights.size())
            {
                theta_target = potential_temperature_from_temperature_pressure(
                    interp.temperature_k[kz], interp.pressure_hpa[kz]);
            }

            if (std::isfinite(theta_target))
            {
                theta_profile[kz] = clamp_theta_k(static_cast<float>(theta_target));
                ++theta_levels_from_sounding;
            }
            else
            {
                theta_profile[kz] = theta_baseline[kz];
            }

            if (has_qv_profile && std::isfinite(interp.mixing_ratio_kgkg[kz]))
            {
                qv_profile[kz] = clamp_qv_kgkg(static_cast<float>(interp.mixing_ratio_kgkg[kz]));
                ++qv_levels_from_sounding;
            }
            else
            {
                qv_profile[kz] = qv_baseline[kz];
            }

            double fallback_u_cart = 0.0;
            double fallback_v_cart = 0.0;
            compute_wind_profile(global_wind_profile, model_heights[kz], fallback_u_cart, fallback_v_cart);
            u_cart_profile[kz] = fallback_u_cart;
            v_cart_profile[kz] = fallback_v_cart;

            if (has_wind_profile &&
                std::isfinite(interp.wind_speed_ms[kz]) &&
                std::isfinite(interp.wind_direction_deg[kz]))
            {
                const double speed = interp.wind_speed_ms[kz];
                const double direction_rad = interp.wind_direction_deg[kz] * kPi / 180.0;

                u_cart_profile[kz] = -speed * std::sin(direction_rad);
                v_cart_profile[kz] = -speed * std::cos(direction_rad);
                ++wind_levels_from_sounding;
            }
        }

        // Coordinate-system branch (Phase A.4 of the Coordinate Backend Plan).
        // The cylindrical path projects the Cartesian sounding hodograph onto
        // (u_r, u_θ); the Cartesian path stores u_x / u_y directly. Same
        // logic as `equations.cpp::initialize` — see that function for the
        // full rationale and the Bug 7 reference. Both branches still apply
        // the horizontal-wind clamp for sounding-injected NaN protection.
        const bool use_cartesian_wind =
            (global_coordinate_system == CoordinateSystem::Cartesian);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < NR; ++i)
        {
            for (int j = 0; j < NTH; ++j)
            {
                const double th = static_cast<double>(j) * dtheta;
                const double cos_th = std::cos(th);
                const double sin_th = std::sin(th);

                for (int k = 0; k < NZ; ++k)
                {
                    const std::size_t kz = static_cast<std::size_t>(k);

                    const float theta_anomaly = static_cast<float>(theta[i][j][k]) - theta_baseline[kz];
                    theta[i][j][k] = clamp_theta_k(theta_profile[kz] + theta_anomaly);

                    const float qv_anomaly = static_cast<float>(qv[i][j][k]) - qv_baseline[kz];
                    qv[i][j][k] = clamp_qv_kgkg(qv_profile[kz] + qv_anomaly);

                    if (use_cartesian_wind)
                    {
                        u[i][j][k]       = clamp_wind_horizontal_ms(static_cast<float>(u_cart_profile[kz]));
                        v[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(v_cart_profile[kz]));
                    }
                    else
                    {
                        const double u_rad =  u_cart_profile[kz] * cos_th + v_cart_profile[kz] * sin_th;
                        const double v_azi = -u_cart_profile[kz] * sin_th + v_cart_profile[kz] * cos_th;
                        u[i][j][k]       = clamp_wind_horizontal_ms(static_cast<float>(u_rad));
                        v[i][j][k] = clamp_wind_horizontal_ms(static_cast<float>(v_azi));
                    }
                }
            }
        }

        initialize_nested_grid();

        if (log_normal_enabled())
        {
            std::cout << "[SOUNDINGS] Applied sounding initialization from scheme='" << runtime_cfg.scheme_id
                      << "', levels=" << interp.num_levels()
                      << ", theta_levels=" << theta_levels_from_sounding
                      << ", qv_levels=" << qv_levels_from_sounding
                      << ", wind_levels=" << wind_levels_from_sounding;
            if (placeholder_profile)
            {
                std::cout << " (placeholder profile)";
            }
            std::cout << std::endl;
        }

        reset_soundings();
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[SOUNDINGS] Failed to apply sounding profile: " << e.what()
                  << ". Keeping procedural initialization." << std::endl;
        reset_soundings();
        return false;
    }
}

/**
 * @brief Program entry point for GUI and headless execution modes.
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

    initialize();

    apply_soundings_to_initial_state();

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

    // Validate that coordinate_system and dynamics scheme are consistent.
    // Cartesian coordinates require the cartesian dynamics scheme (no 1/r terms),
    // and the cartesian scheme assumes Cartesian indexing that is wrong on a
    // cylindrical grid. A mismatch is always a config error.
    {
        const bool coord_is_cartesian =
            (global_coordinate_system == CoordinateSystem::Cartesian);
        const bool scheme_is_cartesian =
            (dynamics_scheme_name == "cartesian" ||
             dynamics_scheme_name == "cart" ||
             dynamics_scheme_name == "cartesian_cpu");

        if (coord_is_cartesian && !scheme_is_cartesian)
        {
            std::cerr << "[CONFIG ERROR] coordinate_system=cartesian requires "
                      << "dynamics.scheme=cartesian, but got '"
                      << dynamics_scheme_name << "'. The cylindrical dynamics "
                      << "scheme uses 1/r terms that are invalid on a Cartesian "
                      << "grid. Aborting." << std::endl;
            return 1;
        }
        if (!coord_is_cartesian && scheme_is_cartesian)
        {
            std::cerr << "[CONFIG ERROR] dynamics.scheme=cartesian requires "
                      << "coordinate_system=cartesian, but got '"
                      << coordinate_system_name(global_coordinate_system)
                      << "'. The Cartesian dynamics scheme assumes straight "
                      << "indexing that is wrong on a cylindrical grid. "
                      << "Aborting." << std::endl;
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
    else
    {
        #ifdef ENABLE_GUI
        run_gui(export_ms);
        #else
        (void)export_ms;
        #endif
    }

    shutdown_compute_backend_runtime();
    return run_status;
}
