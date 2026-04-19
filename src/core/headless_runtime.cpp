/**
 * @file headless_runtime.cpp
 * @brief Core runtime implementation for the tornado model.
 *
 * Provides simulation orchestration and subsystem integration
 * for dynamics, numerics, physics, and runtime execution paths.
 * This file belongs to the primary src/core execution layer.
 */

#include "core/headless_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "numerics/advection.hpp"
#include "diagnostics/field_contract.hpp"
#include "diagnostics/field_validation.hpp"
#include "core/field_snapshot.hpp"
#include "core/npy_writer.hpp"
#include "core/output_writer.hpp"
#include "core/hardware_info.hpp"
#include "core/runtime_config.hpp"
#include "core/shm_writer.hpp"
#include "core/simulation.hpp"

/**
 * @brief Executes the simulation loop in headless mode with periodic exports.
 * @param options Headless runtime options from CLI/config integration.
 * @return Zero on success, non-zero on runtime or export failure.
 */
int run_headless_simulation(const HeadlessRunOptions& options)
{
    const int export_ms = options.export_ms;
    const int duration_s = options.duration_s;
    const int write_every_s = options.write_every_s;
    const std::string& outdir = options.outdir;
    OutputConfig output_config = options.output_config;
    const int thetaIndex = 0;
    const bool verbose_export_debug = log_debug_enabled() || (std::getenv("TORNADO_DEBUG_EXPORTS") != nullptr);
    std::vector<tmv::ValidationReport> export_validation_reports;
    std::string validation_error_message;
    std::vector<float> accumulated_rainfall_surface_mm;
    double accumulated_rainfall_last_update_s = 0.0;

    // Resolve field selection, compression tiers, and estimate disk budget
    resolve_output_fields(output_config);
    apply_default_compression_tiers(output_config);
    if (log_normal_enabled())
    {
        std::cout << "[CONFIG] coordinate_system="
                  << coordinate_system_name(global_coordinate_system) << std::endl;
        std::cout << "[OUTPUT] Format: " << output_format_name(output_config.format)
                  << ", fields: " << field_preset_name(output_config.preset)
                  << " (" << output_config.resolved_fields.size() << " fields)"
                  << ", async_io: " << (output_config.async_io ? "on" : "off")
                  << std::endl;
    }
    if (output_config.tiered_write_cadence && log_normal_enabled())
    {
        std::cout << "[OUTPUT] Tiered write cadence: fast=" << write_every_s
                  << "s, medium=" << output_config.write_cadence_medium_s
                  << "s, slow=" << output_config.write_cadence_slow_s
                  << "s" << std::endl;
    }

    // Runtime ZFP fallback: if ZFP format requested but not compiled in, use npy_3d
    if (output_config.format == OutputFormat::zfp)
    {
#ifndef HAVE_ZFP
        std::cerr << "[OUTPUT] Warning: ZFP compression requested but binary was built "
                  << "without ZFP support (rebuild with ZFP=1). Falling back to npy_3d."
                  << std::endl;
        output_config.format = OutputFormat::npy_3d;
#endif
    }

    estimate_disk_budget(output_config, NR, NTH, NZ, duration_s, write_every_s);

    // Check that the requested grid fits in available RAM; auto-scale if needed
    {
        HardwareInfo hw = detect_hardware();
        if (apply_hardware_defaults(hw, NR, NTH, NZ))
        {
            // Grid was resized — recompute dependent values
            extern void resize_fields();
            resize_fields();
            estimate_disk_budget(output_config, NR, NTH, NZ, duration_s, write_every_s);
        }
    }

    // Async output writer — handles both sync and async paths based on config.
    // In async mode, a background thread serializes snapshots while the sim continues.
    // In sync mode, submit() writes directly on the calling thread (no copy overhead).
    AsyncOutputWriter async_writer(output_config);

    // Live shared-memory transport for zero-I/O visualization
    ShmWriter shm_writer;
    if (options.live_shm && !options.live_shm_fields.empty())
    {
        // Viewer expects [Z][TH][X] layout; ShmWriter handles transposition.
        // Dimensions passed as viewer-space (X=NR, Y=NTH, Z=NZ).
        if (!shm_writer.open(NR, NTH, NZ, options.live_shm_fields))
        {
            std::cerr << "[SHM] Warning: failed to initialize live transport; "
                      << "continuing without SHM\n";
        }
    }

    auto validate_core_fields = [&](const std::string& context,
                                    int step_index,
                                    bool include_percentiles,
                                    const std::filesystem::path* report_path,
                                    bool persist_report) -> bool
    {
        tmv::ValidationReport report;
        report.context = context;
        report.step_index = step_index;
        report.guard_mode = tmv::to_string(global_validation_policy.mode);
        report.guard_fail_on = tmv::to_string(global_validation_policy.fail_on);
        report.guard_scope = tmv::to_string(global_validation_policy.strict_scope);

        const std::vector<std::pair<std::string, Field3D*>> fields = 
        {
            {"u", &u},
            {"v", &v_theta},
            {"w", &w},
            {"rho", &rho},
            {"p", &p},
            {"theta", &theta},
            {"qv", &qv},
            {"qc", &qc},
            {"qr", &qr},
            {"qh", &qh},
            {"qg", &qg},
            {"radar", &radar_reflectivity},
            {"tracer", &tracer},
            {"qi", &qi},
            {"qs", &qs},
            {"vorticity_r", &vorticity_r},
            {"vorticity_theta", &vorticity_theta},
            {"vorticity_z", &vorticity_z},
            {"stretching_term", &stretching_term},
            {"tilting_term", &tilting_term},
            {"baroclinic_term", &baroclinic_term},
            {"p_prime", &p_prime},
            {"dynamic_pressure", &dynamic_pressure},
            {"buoyancy_pressure", &buoyancy_pressure},
            {"angular_momentum", &angular_momentum},
            {"angular_momentum_tendency", &angular_momentum_tendency},
        };

        for (const auto& binding : fields)
        {
            const tmv::FieldContract* contract = tmv::find_field_contract(binding.first);
            if (contract == nullptr)
            {
                report.missing_required.push_back(binding.first);
                continue;
            }
            if (contract->status == tmv::FieldImplementationStatus::NotImplemented)
            {
                if (contract->requirement == tmv::FieldRequirementTier::RequiredNow)
                {
                    report.missing_required.push_back(binding.first);
                }
                continue;
            }

            tmv::FieldValidationReport field_report;
            field_report.field_id = contract->id;
            field_report.status = tmv::to_string(contract->status);
            field_report.requirement = tmv::to_string(contract->requirement);
            field_report.result = tmv::validate_field3d_inplace(
                *binding.second,
                *contract,
                global_validation_policy,
                include_percentiles);

            if (field_report.result.failed)
            {
                report.failed = true;
            }
            report.fields.push_back(std::move(field_report));
        }

        if (persist_report)
        {
            const auto not_implemented = tmv::contracts_with_status(tmv::FieldImplementationStatus::NotImplemented);
            report.known_not_implemented.reserve(not_implemented.size());
            for (const tmv::FieldContract* contract : not_implemented)
            {
                report.known_not_implemented.push_back(contract->id);
                if (contract->requirement == tmv::FieldRequirementTier::RequiredNow)
                {
                    report.missing_not_implemented.push_back(contract->id);
                }
            }
        }

        if (!report.missing_required.empty() && global_validation_policy.mode == tmv::GuardMode::Strict)
        {
            report.failed = true;
        }

        if (report_path != nullptr)
        {
            std::string write_error;
            if (!tmv::write_validation_report_json(report, *report_path, write_error))
            {
                validation_error_message = write_error;
                return false;
            }
        }

        if (persist_report)
        {
            export_validation_reports.push_back(report);
        }

        if (report.failed && global_validation_policy.mode == tmv::GuardMode::Strict)
        {
            validation_error_message = "strict guard failure in context='" + context +
                "' step=" + std::to_string(step_index);
            std::cerr << "[VALIDATION] strict guard failure in context='" << context
                      << "' step=" << step_index << std::endl;
            return false;
        }

        return true;
    };

    auto write_validation_summary = [&](const std::filesystem::path& summary_path) -> bool
    {
        tmv::ValidationReport summary;
        summary.context = "run_summary";
        summary.step_index = static_cast<int>(export_validation_reports.size());
        summary.guard_mode = tmv::to_string(global_validation_policy.mode);
        summary.guard_fail_on = tmv::to_string(global_validation_policy.fail_on);
        summary.guard_scope = tmv::to_string(global_validation_policy.strict_scope);
        summary.failed = false;

        std::size_t failed_reports = 0;
        std::size_t total_violations = 0;
        std::size_t total_nonfinite = 0;
        std::size_t total_bounds = 0;
        std::size_t total_sanitized_nonfinite = 0;
        std::size_t total_sanitized_bounds = 0;

        for (const auto& report : export_validation_reports)
        {
            if (report.failed)
            {
                ++failed_reports;
                summary.failed = true;
            }

            for (const auto& field : report.fields)
            {
                total_nonfinite += field.result.stats.nan_count + field.result.stats.inf_count;
                total_bounds += field.result.stats.below_min_count + field.result.stats.above_max_count;
                total_sanitized_nonfinite += field.result.stats.sanitized_nonfinite_count;
                total_sanitized_bounds += field.result.stats.sanitized_bounds_count;
                total_violations += field.result.violations.size();
            }
        }

        std::error_code ec;
        std::filesystem::create_directories(summary_path.parent_path(), ec);
        if (ec)
        {
            validation_error_message = "failed to create summary directory '" +
                summary_path.parent_path().string() + "': " + ec.message();
            return false;
        }

        std::ofstream out(summary_path);
        if (!out)
        {
            validation_error_message = "failed to open summary report: " + summary_path.string();
            return false;
        }

        out << "{\n";
        out << "  \"context\": \"run_summary\",\n";
        out << "  \"guard_mode\": \"" << json_escape_local(summary.guard_mode) << "\",\n";
        out << "  \"guard_fail_on\": \"" << json_escape_local(summary.guard_fail_on) << "\",\n";
        out << "  \"guard_scope\": \"" << json_escape_local(summary.guard_scope) << "\",\n";
        out << "  \"report_count\": " << export_validation_reports.size() << ",\n";
        out << "  \"failed_report_count\": " << failed_reports << ",\n";
        out << "  \"total_violations\": " << total_violations << ",\n";
        out << "  \"total_nonfinite\": " << total_nonfinite << ",\n";
        out << "  \"total_bounds\": " << total_bounds << ",\n";
        out << "  \"total_sanitized_nonfinite\": " << total_sanitized_nonfinite << ",\n";
        out << "  \"total_sanitized_bounds\": " << total_sanitized_bounds << ",\n";
        out << "  \"failed\": " << (summary.failed ? "true" : "false") << "\n";
        out << "}\n";

        if (!out.good())
        {
            validation_error_message = "failed to write summary report: " + summary_path.string();
            return false;
        }

        return true;
    };
#ifdef EXPORT_NPY
    auto write_npy_2d = [&](const std::vector<float>& buf, const std::string& filename)
    {
        npy::write_2d(buf, NZ, NR, filename);
    };

    auto save_field_slice_npy = [&](const Field3D& field, int theta, const std::string& filename)
    {
        npy::write_field_slice(field, theta, filename);
    };

    // Active fields for this export tick. When tiered_write_cadence is enabled,
    // this is a subset of resolved_fields based on which cadence tiers are due.
    // When disabled, this mirrors resolved_fields (all fields every tick).
    std::unordered_set<std::string> active_export_fields;

    auto write_all_fields = [&](int export_index) -> bool
    {
        if (verbose_export_debug && export_index == 0) 
        {
            std::cout << "\n[EXPORT DEBUG] Writing timestep " << export_index << " (t=" << simulation_time << "s)" << std::endl;

            float theta_min = 1e10, theta_max = -1e10;
            float u_min = 1e10, u_max = -1e10;
            int nan_count = 0;

            for (int i = 0; i < NR && i < 10; ++i) 
            {
                for (int j = 0; j < NTH && j < 3; ++j) 
                {
                    for (int k = 0; k < NZ && k < 3; ++k) 
                    {
                        if (std::isnan(theta[i][j][k])) nan_count++;
                        if (theta[i][j][k] < theta_min) theta_min = theta[i][j][k];
                        if (theta[i][j][k] > theta_max) theta_max = theta[i][j][k];
                        if (u[i][j][k] < u_min) u_min = u[i][j][k];
                        if (u[i][j][k] > u_max) u_max = u[i][j][k];
                    }
                }
            }
            std::cout << "  Theta sample: [" << theta_min << ", " << theta_max << "] K" << std::endl;
            std::cout << "  Wind (u) sample: [" << u_min << ", " << u_max << "] m/s" << std::endl;
            std::cout << "  NaN count (sample): " << nan_count << std::endl;

            if (theta_min < 0 || theta_max > 500) 
            {
                std::cerr << "  ⚠️  ERROR: Theta values are wrong before export!" << std::endl;
            }
            std::cout << std::endl;
        }
        
        std::ostringstream stepdir;
        stepdir << outdir << "/step_" << std::setfill('0') << std::setw(6) << export_index;
        const std::filesystem::path step_path(stepdir.str());
        std::error_code remove_ec;
        std::filesystem::remove_all(step_path, remove_ec);
        std::filesystem::create_directories(step_path);

        std::filesystem::path report_path = step_path / "validation_report.json";
        if (!global_validation_report_path.empty())
        {
            std::filesystem::path report_root(global_validation_report_path);
            std::ostringstream report_name;
            report_name << "step_" << std::setfill('0') << std::setw(6) << export_index
                        << "_validation_report.json";
            report_path = report_root / report_name.str();
        }

        if (!validate_core_fields("pre_export", export_index, true, &report_path, true))
        {
            return false;
        }

        const std::size_t slice_size = static_cast<std::size_t>(NR) * static_cast<std::size_t>(NZ);
        float reflectivity_dbz_min = -30.0f;
        float reflectivity_dbz_max = 120.0f;

        if (const tmv::FieldContract* reflectivity_contract = tmv::find_field_contract("reflectivity_dbz"))
        {
            if (reflectivity_contract->default_bounds.has_min)
            {
                reflectivity_dbz_min =
                    static_cast<float>(reflectivity_contract->default_bounds.min_value);
            }
            if (reflectivity_contract->default_bounds.has_max)
            {
                reflectivity_dbz_max =
                    static_cast<float>(reflectivity_contract->default_bounds.max_value);
            }
        }
        std::vector<float> theta_azimuth_mean(slice_size, 0.0f);
        for (int i = 0; i < NR; ++i)
        {
            for (int k = 0; k < NZ; ++k)
            {
                double sum = 0.0;
                for (int j = 0; j < NTH; ++j)
                {
                    sum += static_cast<double>(theta[i][j][k]);
                }
                theta_azimuth_mean[static_cast<std::size_t>(i) * static_cast<std::size_t>(NZ) + static_cast<std::size_t>(k)] =
                    static_cast<float>(sum / static_cast<double>(NTH));
            }
        }

        std::vector<float> temperature_slice(slice_size, 0.0f);
        std::vector<float> theta_prime_slice(slice_size, 0.0f);
        std::vector<float> theta_v_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> theta_e_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> dewpoint_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> relative_humidity_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> saturation_mixing_ratio_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> total_condensate_slice(slice_size, 0.0f);
        std::vector<float> reflectivity_dbz_slice(slice_size, -30.0f);
        std::vector<float> vorticity_magnitude_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> divergence_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> buoyancy_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> horizontal_vorticity_streamwise_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> horizontal_vorticity_crosswise_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> pressure_gradient_force_x_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> pressure_gradient_force_y_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> pressure_gradient_force_z_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> storm_relative_winds_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> helicity_density_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> okubo_weiss_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> theta_w_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> zdr_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> kdp_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> rhohv_slice(slice_size, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> streamlines_slice(slice_size, 0.0f);
        std::vector<float> trajectory_paths_slice(slice_size, 0.0f);
        std::vector<float> q_vectors_slice(slice_size, 0.0f);
        std::vector<float> turbulent_diffusion_term_slice(slice_size, 0.0f);
        std::vector<float> cross_section_slice(slice_size, 0.0f);
        std::vector<float> rhi_slice_diag(slice_size, 0.0f);
        std::vector<float> hodograph_aligned_cross_section_slice(slice_size, 0.0f);
        std::vector<float> forward_trajectories_slice(slice_size, 0.0f);
        std::vector<float> backward_trajectories_slice(slice_size, 0.0f);
        std::vector<float> parcel_buoyancy_trajectory_slice(slice_size, 0.0f);
        std::vector<float> vorticity_trajectory_slice(slice_size, 0.0f);
        std::vector<float> circulation_material_surface_slice(slice_size, 0.0f);

        const float kappa = static_cast<float>(R_d / cp);
        const float p0f = static_cast<float>(p0);
        const float t_freezing_k = 273.15f;
        const float latent_heat_v = 2.5e6f;
        const float cp_f = static_cast<float>(cp);
        const float g_f = static_cast<float>(g);

        std::vector<float> storm_u_level_mean(static_cast<std::size_t>(NZ), 0.0f);
        std::vector<float> storm_v_level_mean(static_cast<std::size_t>(NZ), 0.0f);
        for (int k = 0; k < NZ; ++k)
        {
            double u_sum = 0.0;
            double v_sum = 0.0;
            std::size_t count = 0;

            for (int th = 0; th < NTH; ++th)
            {
                for (int i = 0; i < NR; ++i)
                {
                    const float u_sample = static_cast<float>(u[i][th][k]);
                    const float v_sample = static_cast<float>(v_theta[i][th][k]);
                    if (std::isfinite(u_sample) && std::isfinite(v_sample))
                    {
                        u_sum += static_cast<double>(u_sample);
                        v_sum += static_cast<double>(v_sample);
                        ++count;
                    }
                }
            }
            if (count > 0)
            {
                const double denom = static_cast<double>(count);
                storm_u_level_mean[static_cast<std::size_t>(k)] =
                    static_cast<float>(u_sum / denom);
                storm_v_level_mean[static_cast<std::size_t>(k)] =
                    static_cast<float>(v_sum / denom);
            }
            else
            {
                storm_u_level_mean[static_cast<std::size_t>(k)] = std::numeric_limits<float>::quiet_NaN();
                storm_v_level_mean[static_cast<std::size_t>(k)] = std::numeric_limits<float>::quiet_NaN();
            }
        }

        auto validate_derived_export_slice = [&](const char* field_id, std::vector<float>& values, int theta_index) -> bool
        {
            const tmv::FieldContract* contract = tmv::find_field_contract(field_id);
            if (contract == nullptr || contract->status != tmv::FieldImplementationStatus::ExportedNow)
            {
                return true;
            }

            tmv::FieldValidationResult result = tmv::validate_buffer_inplace(
                values.data(),
                values.size(),
                *contract,
                global_validation_policy,
                false);

            if (result.failed && global_validation_policy.mode == tmv::GuardMode::Strict)
            {
                validation_error_message =
                    "strict guard failure for derived export field='" +
                    std::string(field_id) +
                    "' step=" + std::to_string(export_index) +
                    " theta=" + std::to_string(theta_index);
                std::cerr << "[VALIDATION] strict guard failure for derived export field='"
                          << field_id << "' step=" << export_index
                          << " theta=" << theta_index << std::endl;
                return false;
            }

            return true;
        };

        const std::size_t polar_size = static_cast<std::size_t>(NR) * static_cast<std::size_t>(NTH);
        if (accumulated_rainfall_surface_mm.size() != polar_size)
        {
            accumulated_rainfall_surface_mm.assign(polar_size, 0.0f);
            accumulated_rainfall_last_update_s = simulation_time;
        }

        const auto polar_index = [&](int i, int th) -> std::size_t
        {
            return static_cast<std::size_t>(th) * static_cast<std::size_t>(NR) + static_cast<std::size_t>(i);
        };
        const auto slice_index = [&](int i, int k) -> std::size_t
        {
            return static_cast<std::size_t>(k) * static_cast<std::size_t>(NR) + static_cast<std::size_t>(i);
        };

        auto safe_reflectivity_dbz = [&](float z_linear) -> float
        {
            if (!std::isfinite(z_linear) || z_linear <= 0.0f)
            {
                return reflectivity_dbz_min;
            }
            const float dbz = 10.0f * std::log10(z_linear);
            if (!std::isfinite(dbz))
            {
                return reflectivity_dbz_min;
            }
            return std::max(reflectivity_dbz_min, std::min(reflectivity_dbz_max, dbz));
        };

        auto sample_field_at_height = [&](const Field3D& field, int i, int th, float z_target_m) -> float
        {
            if (NZ <= 0)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (NZ == 1 || dz <= 0.0 || !std::isfinite(z_target_m))
            {
                return static_cast<float>(field[i][th][0]);
            }

            const float z_index = z_target_m / static_cast<float>(dz);
            if (z_index <= 0.0f)
            {
                return static_cast<float>(field[i][th][0]);
            }
            if (z_index >= static_cast<float>(NZ - 1))
            {
                return static_cast<float>(field[i][th][NZ - 1]);
            }

            const int k0 = static_cast<int>(std::floor(z_index));
            const int k1 = std::min(k0 + 1, NZ - 1);
            const float frac = std::clamp(z_index - static_cast<float>(k0), 0.0f, 1.0f);
            const float v0 = static_cast<float>(field[i][th][k0]);
            const float v1 = static_cast<float>(field[i][th][k1]);
            if (!std::isfinite(v0) && !std::isfinite(v1))
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (!std::isfinite(v0))
            {
                return v1;
            }
            if (!std::isfinite(v1))
            {
                return v0;
            }
            return v0 + frac * (v1 - v0);
        };

        auto temperature_from_theta_and_pressure = [&](float theta_k, float pressure_pa) -> float
        {
            if (!std::isfinite(theta_k) || !std::isfinite(pressure_pa) || pressure_pa <= 0.0f)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return theta_k * std::pow(pressure_pa / p0f, kappa);
        };

        auto dewpoint_from_temp_qv_pressure = [&](float temperature_k, float qv_kgkg, float pressure_pa) -> float
        {
            if (!std::isfinite(temperature_k) || !std::isfinite(qv_kgkg) ||
                !std::isfinite(pressure_pa) || pressure_pa <= 0.0f)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }

            const float p_hpa = pressure_pa * 0.01f;
            float es_hpa = 6.112f * std::exp(
                17.67f * (temperature_k - t_freezing_k) / (temperature_k - 29.65f));
            if (!std::isfinite(es_hpa) || p_hpa <= 1.0e-3f)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }

            es_hpa = std::max(0.0f, std::min(es_hpa, 0.99f * p_hpa));
            const float denom = std::max(1.0e-3f, p_hpa - es_hpa);
            const float qsat = std::max(1.0e-8f, std::min(0.10f, 0.622f * es_hpa / denom));
            if (!std::isfinite(qsat) || qsat <= 0.0f)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }

            const float rh = std::max(0.0f, std::min(200.0f, 100.0f * (qv_kgkg / qsat)));
            const float rh_frac = std::max(1.0e-6f, rh * 0.01f);
            const float temperature_c = temperature_k - t_freezing_k;
            const float gamma_val = std::log(rh_frac) +
                (17.625f * temperature_c) / (243.04f + temperature_c);
            const float gamma_denom = 17.625f - gamma_val;
            if (std::abs(gamma_denom) <= 1.0e-6f)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }

            const float td_c = 243.04f * gamma_val / gamma_denom;
            return td_c + t_freezing_k;
        };

        auto fill_slice_from_radius = [&](std::vector<float>& out, const std::function<float(int)>& value_at_radius)
        {
            out.resize(slice_size);
            for (int i = 0; i < NR; ++i)
            {
                const float value = value_at_radius(i);
                for (int k = 0; k < NZ; ++k)
                {
                    out[slice_index(i, k)] = value;
                }
            }
        };

        auto mean_level_component = [&](const std::vector<float>& level_means, float top_m) -> float
        {
            if (level_means.empty() || NZ <= 0 || dz <= 0.0)
            {
                return 0.0f;
            }
            const int top_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(top_m / static_cast<float>(dz)))));
            double sum = 0.0;
            int count = 0;
            for (int k = 0; k <= top_k; ++k)
            {
                const float sample = level_means[static_cast<std::size_t>(k)];
                if (std::isfinite(sample))
                {
                    sum += static_cast<double>(sample);
                    ++count;
                }
            }
            if (count <= 0)
            {
                return 0.0f;
            }
            return static_cast<float>(sum / static_cast<double>(count));
        };

        const float storm_u_motion = mean_level_component(storm_u_level_mean, 6000.0f);
        const float storm_v_motion = mean_level_component(storm_v_level_mean, 6000.0f);

        auto integrate_srh = [&](int i, int th, float depth_m) -> float
        {
            if (NZ <= 1 || dz <= 0.0 || depth_m <= 0.0f)
            {
                return 0.0f;
            }

            const int top_k = std::max(1, std::min(NZ - 1, static_cast<int>(std::floor(depth_m / static_cast<float>(dz)))));
            double srh = 0.0;
            bool has_samples = false;

            for (int k = 0; k < top_k; ++k)
            {
                const float u0 = static_cast<float>(u[i][th][k]) - storm_u_motion;
                const float v0 = static_cast<float>(v_theta[i][th][k]) - storm_v_motion;
                const float u1 = static_cast<float>(u[i][th][k + 1]) - storm_u_motion;
                const float v1 = static_cast<float>(v_theta[i][th][k + 1]) - storm_v_motion;
                if (!std::isfinite(u0) || !std::isfinite(v0) || !std::isfinite(u1) || !std::isfinite(v1))
                {
                    continue;
                }

                const float u_bar = 0.5f * (u0 + u1);
                const float v_bar = 0.5f * (v0 + v1);
                const float du = u1 - u0;
                const float dv = v1 - v0;
                srh += static_cast<double>(u_bar * dv - v_bar * du);
                has_samples = true;
            }

            if (!has_samples)
            {
                return 0.0f;
            }
            return static_cast<float>(srh);
        };

        std::vector<float> surface_pressure_mean_by_radius(static_cast<std::size_t>(NR), std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < NR; ++i)
        {
            double sum = 0.0;
            int count = 0;
            for (int th = 0; th < NTH; ++th)
            {
                const float p_surface = static_cast<float>(p[i][th][0]);
                if (std::isfinite(p_surface))
                {
                    sum += static_cast<double>(p_surface);
                    ++count;
                }
            }
            if (count > 0)
            {
                surface_pressure_mean_by_radius[static_cast<std::size_t>(i)] =
                    static_cast<float>(sum / static_cast<double>(count));
            }
        }

        std::vector<float> surface_t2_mean_by_radius(static_cast<std::size_t>(NR), std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < NR; ++i)
        {
            double sum = 0.0;
            int count = 0;
            for (int th = 0; th < NTH; ++th)
            {
                const float theta2 = sample_field_at_height(theta, i, th, 2.0f);
                const float p2 = sample_field_at_height(p, i, th, 2.0f);
                const float t2 = temperature_from_theta_and_pressure(theta2, p2);
                if (std::isfinite(t2))
                {
                    sum += static_cast<double>(t2);
                    ++count;
                }
            }
            if (count > 0)
            {
                surface_t2_mean_by_radius[static_cast<std::size_t>(i)] =
                    static_cast<float>(sum / static_cast<double>(count));
            }
        }

        const float accumulation_dt_h = static_cast<float>(
            std::max(0.0, simulation_time - accumulated_rainfall_last_update_s) / 3600.0);

        auto compute_precip_rate_mmh = [&](int i, int th) -> float
        {
            const float rho10 = sample_field_at_height(rho, i, th, 10.0f);
            const float qr10 = std::max(0.0f, sample_field_at_height(qr, i, th, 10.0f));
            const float qs10 = std::max(0.0f, sample_field_at_height(qs, i, th, 10.0f));
            const float qg10 = std::max(0.0f, sample_field_at_height(qg, i, th, 10.0f));
            const float qh10 = std::max(0.0f, sample_field_at_height(qh, i, th, 10.0f));
            if (!std::isfinite(rho10) || !std::isfinite(qr10) || !std::isfinite(qs10) ||
                !std::isfinite(qg10) || !std::isfinite(qh10))
            {
                return std::numeric_limits<float>::quiet_NaN();
            }

            const float water_flux_kgm2s = rho10 * (
                (7.0f * qr10) +
                (1.5f * qs10) +
                (3.0f * qg10) +
                (8.0f * qh10));
            return std::max(0.0f, water_flux_kgm2s * 3600.0f);
        };

        const std::pair<const char*, const char*> exported_core_fields[] = {
            {"u", "u"},
            {"v", "v"},
            {"w", "w"},
            {"rho", "rho"},
            {"p", "p"},
            {"theta", "theta"},
            {"qv", "qv"},
            {"qc", "qc"},
            {"qr", "qr"},
            {"qi", "qi"},
            {"qs", "qs"},
            {"qh", "qh"},
            {"qg", "qg"},
            {"radar", "radar"},
            {"tracer", "tracer"},
            {"vorticity_r", "vorticity_r"},
            {"vorticity_theta", "vorticity_theta"},
            {"vorticity_z", "vorticity_z"},
            {"stretching_term", "stretching_term"},
            {"tilting_term", "tilting_term"},
            {"baroclinic_term", "baroclinic_term"},
            {"p_prime", "p_prime"},
            {"dynamic_pressure", "dynamic_pressure"},
            {"buoyancy_pressure", "buoyancy_pressure"},
            {"angular_momentum", "angular_momentum"},
            {"angular_momentum_tendency", "angular_momentum_tendency"},
        };

        const std::pair<const char*, const char*> exported_derived_fields[] = 
        {
            {"temperature", "temperature"},
            {"theta_prime", "theta_prime"},
            {"theta_v", "theta_v"},
            {"theta_e", "theta_e"},
            {"dewpoint", "dewpoint"},
            {"relative_humidity", "relative_humidity"},
            {"saturation_mixing_ratio", "saturation_mixing_ratio"},
            {"total_condensate", "total_condensate"},
            {"reflectivity_dbz", "reflectivity_dbz"},
            {"vorticity_magnitude", "vorticity_magnitude"},
            {"divergence", "divergence"},
            {"buoyancy", "buoyancy"},
            {"horizontal_vorticity_streamwise", "horizontal_vorticity_streamwise"},
            {"horizontal_vorticity_crosswise", "horizontal_vorticity_crosswise"},
            {"pressure_gradient_force_x", "pressure_gradient_force_x"},
            {"pressure_gradient_force_y", "pressure_gradient_force_y"},
            {"pressure_gradient_force_z", "pressure_gradient_force_z"},
            {"storm_relative_winds", "storm_relative_winds"},
            {"helicity_density", "helicity_density"},
            {"okubo_weiss", "okubo_weiss"},
            {"theta_w", "theta_w"},
            {"zdr", "zdr"},
            {"kdp", "kdp"},
            {"rhohv", "rhohv"},
            {"streamlines", "streamlines"},
            {"trajectory_paths", "trajectory_paths"},
            {"q_vectors", "q_vectors"},
            {"turbulent_diffusion_term", "turbulent_diffusion_term"},
            {"cross_section", "cross_section"},
            {"rhi_slice", "rhi_slice"},
            {"hodograph_aligned_cross_section", "hodograph_aligned_cross_section"},
            {"forward_trajectories", "forward_trajectories"},
            {"backward_trajectories", "backward_trajectories"},
            {"parcel_buoyancy_trajectory", "parcel_buoyancy_trajectory"},
            {"vorticity_trajectory", "vorticity_trajectory"},
            {"circulation_material_surface", "circulation_material_surface"},
        };

        struct TrancheFieldDescriptor
        {
            const char* field_id;
            const char* suffix;
            const char* category;
        };

        const TrancheFieldDescriptor exported_tranche_fields[] = {
            {"u10", "u10", "surface"},
            {"v10", "v10", "surface"},
            {"t2", "t2", "surface"},
            {"td2", "td2", "surface"},
            {"surface_pressure_perturbation", "surface_pressure_perturbation", "surface"},
            {"surface_sensible_heat_flux", "surface_sensible_heat_flux", "surface"},
            {"surface_latent_heat_flux", "surface_latent_heat_flux", "surface"},
            {"surface_moisture_flux", "surface_moisture_flux", "surface"},
            {"skin_temperature", "skin_temperature", "surface"},
            {"cold_pool_boundary", "cold_pool_boundary", "surface"},
            {"precip_rate", "precip_rate", "surface"},
            {"accumulated_rainfall", "accumulated_rainfall", "surface"},
            {"composite_reflectivity", "composite_reflectivity", "column"},
            {"column_max_w", "column_max_w", "column"},
            {"column_max_vorticity", "column_max_vorticity", "column"},
            {"vil", "vil", "column"},
            {"cloud_top_height", "cloud_top_height", "column"},
            {"cloud_base_height", "cloud_base_height", "column"},
            {"lcl", "lcl", "column"},
            {"lfc", "lfc", "column"},
            {"el", "el", "column"},
            {"cape", "cape", "column"},
            {"cin", "cin", "column"},
            {"lifted_index", "lifted_index", "column"},
            {"k_index", "k_index", "column"},
            {"showalter_index", "showalter_index", "column"},
            {"total_totals", "total_totals", "column"},
            {"srh_0_1km", "srh_0_1km", "column"},
            {"srh_0_3km", "srh_0_3km", "column"},
            {"ehi", "ehi", "column"},
            {"scp", "scp", "column"},
            {"stp", "stp", "column"},
            {"ppi_sweep", "ppi_sweep", "radar_synthetic"},
            {"rhi_sweep", "rhi_sweep", "radar_synthetic"},
            {"bwer", "bwer", "radar_synthetic"},
            {"mesocyclone_diagnostic", "mesocyclone_diagnostic", "radar_synthetic"},
            {"vrot", "vrot", "radar_synthetic"},
        };

        struct RegisteredDiagnosticProduct
        {
            const char* field_id;
            const char* suffix;
            const char* category;
            std::function<bool(int, std::vector<float>&)> compute_slice;
        };

        std::vector<RegisteredDiagnosticProduct> tranche_registry;
        tranche_registry.reserve(40);

        std::vector<float> surface_sensible_heat_flux_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> surface_latent_heat_flux_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> surface_moisture_flux_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> skin_temperature_by_radius(static_cast<std::size_t>(NR), 273.15f);
        std::vector<float> cold_pool_boundary_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> cloud_top_height_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> cloud_base_height_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> lcl_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> lfc_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> el_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> cape_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> cin_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> lifted_index_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> k_index_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> showalter_index_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> total_totals_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> ehi_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> scp_by_radius(static_cast<std::size_t>(NR), 0.0f);
        std::vector<float> stp_by_radius(static_cast<std::size_t>(NR), 0.0f);
        int cached_surface_column_theta = -1;

        auto finite_or = [](float value, float fallback) -> float
        {
            return std::isfinite(value) ? value : fallback;
        };

        auto saturation_mixing_ratio_from_temp_pressure = [&](float temperature_k, float pressure_pa) -> float
        {
            if (!std::isfinite(temperature_k) || !std::isfinite(pressure_pa) || pressure_pa <= 0.0f)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }

            const float p_hpa = pressure_pa * 0.01f;
            float es_hpa = 6.112f * std::exp(
                17.67f * (temperature_k - t_freezing_k) / std::max(1.0e-3f, (temperature_k - 29.65f)));
            if (!std::isfinite(es_hpa))
            {
                return std::numeric_limits<float>::quiet_NaN();
            }

            es_hpa = std::max(0.0f, std::min(es_hpa, 0.99f * p_hpa));
            const float denom = std::max(1.0e-3f, p_hpa - es_hpa);
            const float qsat = 0.622f * es_hpa / denom;
            if (!std::isfinite(qsat))
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return std::max(0.0f, qsat);
        };

        auto lcl_height_from_temperature_dewpoint = [&](float temperature_k, float dewpoint_k) -> float
        {
            if (!std::isfinite(temperature_k) || !std::isfinite(dewpoint_k))
            {
                return 0.0f;
            }
            const float t_c = temperature_k - t_freezing_k;
            const float td_c = dewpoint_k - t_freezing_k;
            const float lcl = 125.0f * std::max(0.0f, t_c - td_c);
            const float z_top = static_cast<float>(std::max(0, NZ - 1)) * static_cast<float>(std::max(1.0, dz));
            return std::clamp(lcl, 0.0f, z_top);
        };

        auto ensure_surface_column_diagnostics = [&](int th)
        {
            if (cached_surface_column_theta == th)
            {
                return;
            }
            cached_surface_column_theta = th;

            constexpr float kExchangeCoeff = 1.5e-3f;
            constexpr float kDryLapseKPerM = 9.8e-3f;
            constexpr float kMoistLapseKPerM = 6.0e-3f;
            constexpr float kCloudMixingThreshold = 1.0e-5f;
            const float dz_m = static_cast<float>(std::max(1.0, dz));
            const float z_top = static_cast<float>(std::max(0, NZ - 1)) * dz_m;

            for (int i = 0; i < NR; ++i)
            {
                const std::size_t idx = static_cast<std::size_t>(i);
                const float p_surface = finite_or(static_cast<float>(p[i][th][0]), p0f);
                const float theta_surface = finite_or(static_cast<float>(theta[i][th][0]), 300.0f);
                const float t_skin = finite_or(
                    temperature_from_theta_and_pressure(theta_surface, p_surface),
                    300.0f);
                const float qv_surface = std::max(0.0f, finite_or(static_cast<float>(qv[i][th][0]), 0.0f));

                const float theta2 = sample_field_at_height(theta, i, th, 2.0f);
                const float p2 = sample_field_at_height(p, i, th, 2.0f);
                const float t2 = finite_or(temperature_from_theta_and_pressure(theta2, p2), t_skin);
                const float qv2 = std::max(0.0f, finite_or(sample_field_at_height(qv, i, th, 2.0f), qv_surface));
                const float td2 = finite_or(dewpoint_from_temp_qv_pressure(t2, qv2, finite_or(p2, p_surface)), t2);

                const float rho10 = std::max(
                    0.1f,
                    finite_or(sample_field_at_height(rho, i, th, 10.0f), finite_or(static_cast<float>(rho[i][th][0]), 1.0f)));
                const float u10_local = finite_or(sample_field_at_height(u, i, th, 10.0f), 0.0f);
                const float v10_local = finite_or(sample_field_at_height(v_theta, i, th, 10.0f), 0.0f);
                const float wind10 = std::hypot(u10_local, v10_local);
                const float qsat_surface = finite_or(
                    saturation_mixing_ratio_from_temp_pressure(t_skin, p_surface),
                    qv_surface);

                const float sensible_flux =
                    rho10 * cp_f * kExchangeCoeff * wind10 * (t_skin - t2);
                const float moisture_flux =
                    rho10 * kExchangeCoeff * wind10 * (qsat_surface - qv2);
                const float latent_flux = latent_heat_v * moisture_flux;

                skin_temperature_by_radius[idx] = t_skin;
                surface_sensible_heat_flux_by_radius[idx] = finite_or(sensible_flux, 0.0f);
                surface_moisture_flux_by_radius[idx] = finite_or(moisture_flux, 0.0f);
                surface_latent_heat_flux_by_radius[idx] = finite_or(latent_flux, 0.0f);

                const float t2_mean = finite_or(surface_t2_mean_by_radius[idx], t2);
                cold_pool_boundary_by_radius[idx] = std::max(0.0f, t2_mean - t2);

                float cloud_base = 0.0f;
                float cloud_top = 0.0f;
                bool cloud_found = false;
                float cape = 0.0f;
                float cin = 0.0f;

                const float td_surface = finite_or(
                    dewpoint_from_temp_qv_pressure(t_skin, qv_surface, p_surface),
                    td2);
                const float lcl = lcl_height_from_temperature_dewpoint(t_skin, td_surface);
                float lfc = 0.0f;
                float el = 0.0f;
                bool lfc_found = false;
                bool el_found = false;

                auto nearest_level_for_pressure = [&](float target_pa) -> int
                {
                    int best_k = 0;
                    float best_diff = std::numeric_limits<float>::infinity();
                    for (int k = 0; k < NZ; ++k)
                    {
                        const float pk = static_cast<float>(p[i][th][k]);
                        if (!std::isfinite(pk))
                        {
                            continue;
                        }
                        const float diff = std::abs(pk - target_pa);
                        if (diff < best_diff)
                        {
                            best_diff = diff;
                            best_k = k;
                        }
                    }
                    return best_k;
                };

                auto env_temperature_at_k = [&](int k) -> float
                {
                    const float pk = finite_or(static_cast<float>(p[i][th][k]), p_surface);
                    const float thetak = finite_or(static_cast<float>(theta[i][th][k]), theta_surface);
                    return finite_or(temperature_from_theta_and_pressure(thetak, pk), t_skin);
                };

                auto env_dewpoint_at_k = [&](int k, float temp_k) -> float
                {
                    const float pk = finite_or(static_cast<float>(p[i][th][k]), p_surface);
                    const float qvk = std::max(0.0f, finite_or(static_cast<float>(qv[i][th][k]), qv_surface));
                    return finite_or(dewpoint_from_temp_qv_pressure(temp_k, qvk, pk), temp_k);
                };

                auto parcel_temp_from_surface = [&](float z_m) -> float
                {
                    const float dry_depth = std::min(z_m, lcl);
                    const float moist_depth = std::max(0.0f, z_m - lcl);
                    const float parcel_t =
                        t_skin - (kDryLapseKPerM * dry_depth) - (kMoistLapseKPerM * moist_depth);
                    return std::max(150.0f, parcel_t);
                };

                for (int k = 0; k < NZ; ++k)
                {
                    const float z_m = static_cast<float>(k) * dz_m;
                    const float env_t = env_temperature_at_k(k);
                    const float parcel_t = parcel_temp_from_surface(z_m);
                    const float buoyancy = g_f * (parcel_t - env_t) / std::max(150.0f, env_t);

                    if (std::isfinite(buoyancy))
                    {
                        if (buoyancy > 0.0f && z_m >= lcl)
                        {
                            cape += buoyancy * dz_m;
                            if (!lfc_found)
                            {
                                lfc = z_m;
                                lfc_found = true;
                            }
                        }
                        else if (buoyancy < 0.0f)
                        {
                            cin += buoyancy * dz_m;
                            if (lfc_found && !el_found)
                            {
                                el = z_m;
                                el_found = true;
                            }
                        }
                    }

                    const float qc_k = std::max(0.0f, finite_or(static_cast<float>(qc[i][th][k]), 0.0f));
                    const float qr_k = std::max(0.0f, finite_or(static_cast<float>(qr[i][th][k]), 0.0f));
                    const float qi_k = std::max(0.0f, finite_or(static_cast<float>(qi[i][th][k]), 0.0f));
                    const float qs_k = std::max(0.0f, finite_or(static_cast<float>(qs[i][th][k]), 0.0f));
                    const float qg_k = std::max(0.0f, finite_or(static_cast<float>(qg[i][th][k]), 0.0f));
                    const float qh_k = std::max(0.0f, finite_or(static_cast<float>(qh[i][th][k]), 0.0f));
                    const float condensate = qc_k + qr_k + qi_k + qs_k + qg_k + qh_k;
                    if (condensate > kCloudMixingThreshold)
                    {
                        if (!cloud_found)
                        {
                            cloud_base = z_m;
                            cloud_found = true;
                        }
                        cloud_top = z_m;
                    }
                }

                if (!lfc_found)
                {
                    lfc = 0.0f;
                }
                if (!el_found)
                {
                    el = lfc_found ? z_top : 0.0f;
                }
                if (!cloud_found)
                {
                    cloud_base = 0.0f;
                    cloud_top = 0.0f;
                }

                const int k850 = nearest_level_for_pressure(85000.0f);
                const int k700 = nearest_level_for_pressure(70000.0f);
                const int k500 = nearest_level_for_pressure(50000.0f);
                const float z850 = static_cast<float>(k850) * dz_m;
                const float z500 = static_cast<float>(k500) * dz_m;

                const float t850 = env_temperature_at_k(k850);
                const float t700 = env_temperature_at_k(k700);
                const float t500 = env_temperature_at_k(k500);
                const float td850 = env_dewpoint_at_k(k850, t850);
                const float td700 = env_dewpoint_at_k(k700, t700);
                const float parcel_t500_surface = parcel_temp_from_surface(z500);

                const float lcl_850 = lcl_height_from_temperature_dewpoint(t850, td850);
                const float z_rel_500 = std::max(0.0f, z500 - z850);
                const float dry_depth_850 = std::min(z_rel_500, lcl_850);
                const float moist_depth_850 = std::max(0.0f, z_rel_500 - lcl_850);
                const float parcel_t500_850 = std::max(
                    150.0f,
                    t850 - (kDryLapseKPerM * dry_depth_850) - (kMoistLapseKPerM * moist_depth_850));

                const float srh01 = integrate_srh(i, th, 1000.0f);
                const float srh03 = integrate_srh(i, th, 3000.0f);
                const float u6 = finite_or(sample_field_at_height(u, i, th, 6000.0f), u10_local);
                const float v6 = finite_or(sample_field_at_height(v_theta, i, th, 6000.0f), v10_local);
                const float bulk_shear_0_6 = std::hypot(u6 - u10_local, v6 - v10_local);
                const float cin_mag = std::abs(std::min(0.0f, cin));

                const float lifted_index = t500 - parcel_t500_surface;
                const float k_index = (t850 - t500) + td850 - (t700 - td700);
                const float showalter_index = t500 - parcel_t500_850;
                const float total_totals = (t850 - t500) + (td850 - t500);
                const float ehi = std::max(0.0f, cape) * std::max(0.0f, srh01) / 160000.0f;
                const float scp = std::max(0.0f, cape / 1000.0f) *
                    std::max(0.0f, srh03 / 100.0f) *
                    std::max(0.0f, bulk_shear_0_6 / 20.0f);
                const float stp = std::max(0.0f, cape / 1500.0f) *
                    std::clamp((2000.0f - lcl) / 1000.0f, 0.0f, 2.0f) *
                    std::max(0.0f, srh01 / 150.0f) *
                    std::clamp((200.0f - cin_mag) / 200.0f, 0.0f, 1.0f);

                cloud_top_height_by_radius[idx] = finite_or(cloud_top, 0.0f);
                cloud_base_height_by_radius[idx] = finite_or(cloud_base, 0.0f);
                lcl_by_radius[idx] = finite_or(lcl, 0.0f);
                lfc_by_radius[idx] = finite_or(lfc, 0.0f);
                el_by_radius[idx] = finite_or(el, 0.0f);
                cape_by_radius[idx] = finite_or(std::max(0.0f, cape), 0.0f);
                cin_by_radius[idx] = finite_or(std::min(0.0f, cin), 0.0f);
                lifted_index_by_radius[idx] = finite_or(lifted_index, 0.0f);
                k_index_by_radius[idx] = finite_or(k_index, 0.0f);
                showalter_index_by_radius[idx] = finite_or(showalter_index, 0.0f);
                total_totals_by_radius[idx] = finite_or(total_totals, 0.0f);
                ehi_by_radius[idx] = finite_or(std::max(0.0f, ehi), 0.0f);
                scp_by_radius[idx] = finite_or(std::max(0.0f, scp), 0.0f);
                stp_by_radius[idx] = finite_or(std::max(0.0f, stp), 0.0f);
            }
        };

        auto register_cached_radius_field = [&](const char* field_id,
                                                const char* suffix,
                                                const char* category,
                                                const std::vector<float>* values_by_radius)
        {
            const std::vector<float>* values = values_by_radius;
            tranche_registry.push_back({field_id, suffix, category, [&, values](int th, std::vector<float>& out) {
                ensure_surface_column_diagnostics(th);
                fill_slice_from_radius(out, [&](int i) {
                    return (*values)[static_cast<std::size_t>(i)];
                });
                return true;
            }});
        };

        tranche_registry.push_back({"u10", "u10", "surface", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) { return sample_field_at_height(u, i, th, 10.0f); });
            return true;
        }});
        tranche_registry.push_back({"v10", "v10", "surface", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) { return sample_field_at_height(v_theta, i, th, 10.0f); });
            return true;
        }});
        tranche_registry.push_back({"t2", "t2", "surface", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) {
                const float theta2 = sample_field_at_height(theta, i, th, 2.0f);
                const float p2 = sample_field_at_height(p, i, th, 2.0f);
                return temperature_from_theta_and_pressure(theta2, p2);
            });
            return true;
        }});
        tranche_registry.push_back({"td2", "td2", "surface", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) {
                const float theta2 = sample_field_at_height(theta, i, th, 2.0f);
                const float p2 = sample_field_at_height(p, i, th, 2.0f);
                const float t2 = temperature_from_theta_and_pressure(theta2, p2);
                const float qv2 = sample_field_at_height(qv, i, th, 2.0f);
                return dewpoint_from_temp_qv_pressure(t2, qv2, p2);
            });
            return true;
        }});
        tranche_registry.push_back({"surface_pressure_perturbation", "surface_pressure_perturbation", "surface",
            [&](int th, std::vector<float>& out) {
                fill_slice_from_radius(out, [&](int i) {
                    const float p_surface = static_cast<float>(p[i][th][0]);
                    const float p_mean = surface_pressure_mean_by_radius[static_cast<std::size_t>(i)];
                    if (!std::isfinite(p_surface) || !std::isfinite(p_mean))
                    {
                        return std::numeric_limits<float>::quiet_NaN();
                    }
                    return p_surface - p_mean;
                });
                return true;
            }});
        register_cached_radius_field(
            "surface_sensible_heat_flux",
            "surface_sensible_heat_flux",
            "surface",
            &surface_sensible_heat_flux_by_radius);
        register_cached_radius_field(
            "surface_latent_heat_flux",
            "surface_latent_heat_flux",
            "surface",
            &surface_latent_heat_flux_by_radius);
        register_cached_radius_field(
            "surface_moisture_flux",
            "surface_moisture_flux",
            "surface",
            &surface_moisture_flux_by_radius);
        register_cached_radius_field(
            "skin_temperature",
            "skin_temperature",
            "surface",
            &skin_temperature_by_radius);
        register_cached_radius_field(
            "cold_pool_boundary",
            "cold_pool_boundary",
            "surface",
            &cold_pool_boundary_by_radius);
        tranche_registry.push_back({"precip_rate", "precip_rate", "surface", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) { return compute_precip_rate_mmh(i, th); });
            return true;
        }});
        tranche_registry.push_back({"accumulated_rainfall", "accumulated_rainfall", "surface",
            [&](int th, std::vector<float>& out) {
                fill_slice_from_radius(out, [&](int i) {
                    const float rate = compute_precip_rate_mmh(i, th);
                    const std::size_t idx = polar_index(i, th);
                    float total = accumulated_rainfall_surface_mm[idx];
                    if (!std::isfinite(total) || total < 0.0f)
                    {
                        total = 0.0f;
                    }
                    if (std::isfinite(rate) && accumulation_dt_h > 0.0f)
                    {
                        total += rate * accumulation_dt_h;
                    }
                    accumulated_rainfall_surface_mm[idx] = total;
                    return total;
                });
                return true;
            }});
        tranche_registry.push_back({"composite_reflectivity", "composite_reflectivity", "column",
            [&](int th, std::vector<float>& out) {
                fill_slice_from_radius(out, [&](int i) {
                    float max_dbz = reflectivity_dbz_min;
                    for (int k = 0; k < NZ; ++k)
                    {
                        max_dbz = std::max(max_dbz,
                            safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[i][th][k])));
                    }
                    return max_dbz;
                });
                return true;
            }});
        tranche_registry.push_back({"column_max_w", "column_max_w", "column", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) {
                float max_w = -std::numeric_limits<float>::infinity();
                bool has_samples = false;
                for (int k = 0; k < NZ; ++k)
                {
                    const float sample = static_cast<float>(w[i][th][k]);
                    if (std::isfinite(sample))
                    {
                        max_w = std::max(max_w, sample);
                        has_samples = true;
                    }
                }
                if (!has_samples)
                {
                    return std::numeric_limits<float>::quiet_NaN();
                }
                return max_w;
            });
            return true;
        }});
        tranche_registry.push_back({"column_max_vorticity", "column_max_vorticity", "column",
            [&](int th, std::vector<float>& out) {
                fill_slice_from_radius(out, [&](int i) {
                    float max_vort = 0.0f;
                    bool has_samples = false;
                    for (int k = 0; k < NZ; ++k)
                    {
                        const float vort_r = static_cast<float>(vorticity_r[i][th][k]);
                        const float vort_th = static_cast<float>(vorticity_theta[i][th][k]);
                        const float vort_z = static_cast<float>(vorticity_z[i][th][k]);
                        if (std::isfinite(vort_r) && std::isfinite(vort_th) && std::isfinite(vort_z))
                        {
                            const float mag = std::sqrt((vort_r * vort_r) + (vort_th * vort_th) + (vort_z * vort_z));
                            max_vort = std::max(max_vort, mag);
                            has_samples = true;
                        }
                    }
                    if (!has_samples)
                    {
                        return std::numeric_limits<float>::quiet_NaN();
                    }
                    return max_vort;
                });
                return true;
            }});
        tranche_registry.push_back({"vil", "vil", "column", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) {
                double vil = 0.0;
                bool has_samples = false;
                for (int k = 0; k < NZ; ++k)
                {
                    const float rho_value = std::max(0.0f, static_cast<float>(rho[i][th][k]));
                    const float qc_value = std::max(0.0f, static_cast<float>(qc[i][th][k]));
                    const float qr_value = std::max(0.0f, static_cast<float>(qr[i][th][k]));
                    const float qi_value = std::max(0.0f, static_cast<float>(qi[i][th][k]));
                    const float qs_value = std::max(0.0f, static_cast<float>(qs[i][th][k]));
                    const float qg_value = std::max(0.0f, static_cast<float>(qg[i][th][k]));
                    const float qh_value = std::max(0.0f, static_cast<float>(qh[i][th][k]));
                    const float hydro = qc_value + qr_value + qi_value + qs_value + qg_value + qh_value;
                    if (std::isfinite(rho_value) && std::isfinite(hydro))
                    {
                        vil += static_cast<double>(rho_value * hydro * static_cast<float>(dz));
                        has_samples = true;
                    }
                }
                if (!has_samples)
                {
                    return std::numeric_limits<float>::quiet_NaN();
                }
                return static_cast<float>(std::max(0.0, vil));
            });
            return true;
        }});
        register_cached_radius_field(
            "cloud_top_height",
            "cloud_top_height",
            "column",
            &cloud_top_height_by_radius);
        register_cached_radius_field(
            "cloud_base_height",
            "cloud_base_height",
            "column",
            &cloud_base_height_by_radius);
        register_cached_radius_field(
            "lcl",
            "lcl",
            "column",
            &lcl_by_radius);
        register_cached_radius_field(
            "lfc",
            "lfc",
            "column",
            &lfc_by_radius);
        register_cached_radius_field(
            "el",
            "el",
            "column",
            &el_by_radius);
        register_cached_radius_field(
            "cape",
            "cape",
            "column",
            &cape_by_radius);
        register_cached_radius_field(
            "cin",
            "cin",
            "column",
            &cin_by_radius);
        register_cached_radius_field(
            "lifted_index",
            "lifted_index",
            "column",
            &lifted_index_by_radius);
        register_cached_radius_field(
            "k_index",
            "k_index",
            "column",
            &k_index_by_radius);
        register_cached_radius_field(
            "showalter_index",
            "showalter_index",
            "column",
            &showalter_index_by_radius);
        register_cached_radius_field(
            "total_totals",
            "total_totals",
            "column",
            &total_totals_by_radius);
        tranche_registry.push_back({"srh_0_1km", "srh_0_1km", "column", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) { return integrate_srh(i, th, 1000.0f); });
            return true;
        }});
        tranche_registry.push_back({"srh_0_3km", "srh_0_3km", "column", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) { return integrate_srh(i, th, 3000.0f); });
            return true;
        }});
        register_cached_radius_field(
            "ehi",
            "ehi",
            "column",
            &ehi_by_radius);
        register_cached_radius_field(
            "scp",
            "scp",
            "column",
            &scp_by_radius);
        register_cached_radius_field(
            "stp",
            "stp",
            "column",
            &stp_by_radius);

        const float ppi_elevation_rad = 0.5f * static_cast<float>(3.14159265358979323846 / 180.0);
        const int low_top_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(2000.0 / std::max(1.0, dz)))));
        const int mid_bottom_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(3000.0 / std::max(1.0, dz)))));
        const int mid_top_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(6000.0 / std::max(1.0, dz)))));
        const int upper_bottom_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(6000.0 / std::max(1.0, dz)))));
        const int upper_top_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(10000.0 / std::max(1.0, dz)))));
        const int vrot_top_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(1000.0 / std::max(1.0, dz)))));
        const int meso_top_k = std::max(0, std::min(NZ - 1, static_cast<int>(std::floor(3000.0 / std::max(1.0, dz)))));

        tranche_registry.push_back({"ppi_sweep", "ppi_sweep", "radar_synthetic", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) {
                const float r_m = static_cast<float>(i) * static_cast<float>(dr);
                const float z_target = r_m * std::tan(ppi_elevation_rad);
                const int k_ppi = std::max(0, std::min(NZ - 1, static_cast<int>(std::round(z_target / static_cast<float>(dz)))));
                return safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[i][th][k_ppi]));
            });
            return true;
        }});
        tranche_registry.push_back({"rhi_sweep", "rhi_sweep", "radar_synthetic", [&](int th, std::vector<float>& out) {
            out.resize(slice_size);
            for (int k = 0; k < NZ; ++k)
            {
                for (int i = 0; i < NR; ++i)
                {
                    float sum = 0.0f;
                    int count = 0;
                    for (int dk = -1; dk <= 1; ++dk)
                    {
                        const int kk = k + dk;
                        if (kk < 0 || kk >= NZ)
                        {
                            continue;
                        }
                        for (int di = -1; di <= 1; ++di)
                        {
                            const int ii = i + di;
                            if (ii < 0 || ii >= NR)
                            {
                                continue;
                            }
                            const float dbz = safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[ii][th][kk]));
                            if (std::isfinite(dbz))
                            {
                                sum += dbz;
                                ++count;
                            }
                        }
                    }
                    out[slice_index(i, k)] = (count > 0)
                        ? (sum / static_cast<float>(count))
                        : reflectivity_dbz_min;
                }
            }
            return true;
        }});
        tranche_registry.push_back({"bwer", "bwer", "radar_synthetic", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) {
                float low_max = reflectivity_dbz_min;
                float mid_min = 120.0f;
                float upper_max = reflectivity_dbz_min;
                for (int k = 0; k <= low_top_k; ++k)
                {
                    low_max = std::max(low_max,
                        safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[i][th][k])));
                }
                for (int k = mid_bottom_k; k <= mid_top_k; ++k)
                {
                    mid_min = std::min(mid_min,
                        safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[i][th][k])));
                }
                for (int k = upper_bottom_k; k <= upper_top_k; ++k)
                {
                    upper_max = std::max(upper_max,
                        safe_reflectivity_dbz(static_cast<float>(radar_reflectivity[i][th][k])));
                }
                if (low_max >= 45.0f && mid_min <= 20.0f && upper_max >= 45.0f)
                {
                    return 1.0f;
                }
                return 0.0f;
            });
            return true;
        }});
        tranche_registry.push_back({"mesocyclone_diagnostic", "mesocyclone_diagnostic", "radar_synthetic",
            [&](int th, std::vector<float>& out) {
                fill_slice_from_radius(out, [&](int i) 
                {
                    float low_vort_max = 0.0f;
                    float column_max_updraft = -std::numeric_limits<float>::infinity();
                    float low_level_vrot = 0.0f;
                    bool has_updraft = false;
                    for (int k = 0; k < NZ; ++k)
                    {
                        const float w_sample = static_cast<float>(w[i][th][k]);
                        if (std::isfinite(w_sample))
                        {
                            column_max_updraft = std::max(column_max_updraft, w_sample);
                            has_updraft = true;
                        }
                        if (k <= meso_top_k)
                        {
                            const float vort_z = std::abs(static_cast<float>(vorticity_z[i][th][k]));
                            if (std::isfinite(vort_z))
                            {
                                low_vort_max = std::max(low_vort_max, vort_z);
                            }
                        }
                        if (k <= vrot_top_k)
                        {
                            const float vrot = std::abs(static_cast<float>(v_theta[i][th][k]));
                            if (std::isfinite(vrot))
                            {
                                low_level_vrot = std::max(low_level_vrot, vrot);
                            }
                        }
                    }
                    if (has_updraft && column_max_updraft >= 10.0f &&
                        low_vort_max >= 0.01f && low_level_vrot >= 20.0f)
                    {
                        return 1.0f;
                    }
                    return 0.0f;
                });
                return true;
            }});
        tranche_registry.push_back({"vrot", "vrot", "radar_synthetic", [&](int th, std::vector<float>& out) {
            fill_slice_from_radius(out, [&](int i) {
                float max_vrot = 0.0f;
                for (int k = 0; k <= vrot_top_k; ++k)
                {
                    const float sample = std::abs(static_cast<float>(v_theta[i][th][k]));
                    if (std::isfinite(sample))
                    {
                        max_vrot = std::max(max_vrot, sample);
                    }
                }
                return max_vrot;
            });
            return true;
        }});

        // Serialize step manifest to a JSON string (reused by both sync and async paths)
        auto build_manifest_json = [&]() -> std::string
        {
            std::ostringstream out;

            out << "{\n";
            out << "  \"step_index\": " << export_index << ",\n";
            out << "  \"simulation_time_s\": " << simulation_time << ",\n";
            out << "  \"step_dir\": \"" << json_escape_local(step_path.filename().string()) << "\",\n";
            out << "  \"grid\": {\n";
            out << "    \"nr\": " << NR << ",\n";
            out << "    \"nth\": " << NTH << ",\n";
            out << "    \"nz\": " << NZ << ",\n";
            out << "    \"dr_m\": " << dr << ",\n";
            out << "    \"dtheta_rad\": " << dtheta << ",\n";
            out << "    \"dz_m\": " << dz << "\n";
            out << "  },\n";
            out << "  \"theta_index\": {\n";
            out << "    \"min\": 0,\n";
            out << "    \"max\": " << (NTH > 0 ? NTH - 1 : 0) << ",\n";
            out << "    \"count\": " << NTH << ",\n";
            out << "    \"file_prefix\": \"th{theta}_\"\n";
            out << "  },\n";
            out << "  \"soundings\": {\n";
            out << "    \"enabled\": " << (global_sounding_enabled ? "true" : "false") << ",\n";
            out << "    \"scheme\": \"" << json_escape_local(global_runtime_sounding_config.scheme_id) << "\",\n";
            out << "    \"file_path\": \"" << json_escape_local(global_runtime_sounding_config.file_path) << "\",\n";
            out << "    \"interpolation_method\": \""
                << json_escape_local(sounding_interpolation_method_name(global_runtime_sounding_config.interpolation_method))
                << "\"\n";
            out << "  },\n";
            out << "  \"fields\": [\n";

            bool first_field = true;

            auto write_field_entry = [&](const char* field_id, const char* suffix, const char* category, bool derived)
            {
                if (!first_field)
                {
                    out << ",\n";
                }
                first_field = false;

                out << "    {\n";
                out << "      \"field_id\": \"" << json_escape_local(field_id) << "\",\n";
                out << "      \"category\": \"" << json_escape_local(category) << "\",\n";
                out << "      \"derived\": " << (derived ? "true" : "false") << ",\n";
                out << "      \"file_suffix\": \"" << json_escape_local(suffix) << "\",\n";
                out << "      \"file_pattern\": \"th{theta}_" << json_escape_local(suffix) << ".npy\"";

                if (const tmv::FieldContract* contract = tmv::find_field_contract(field_id))
                {
                    out << ",\n";
                    out << "      \"units\": \"" << json_escape_local(contract->units) << "\",\n";
                    out << "      \"description\": \"" << json_escape_local(contract->description) << "\"";

                    if (contract->default_bounds.has_min || contract->default_bounds.has_max)
                    {
                        out << ",\n";
                        out << "      \"bounds\": {";
                        bool first_bound = true;
                        if (contract->default_bounds.has_min)
                        {
                            out << "\"min\": " << contract->default_bounds.min_value;
                            first_bound = false;
                        }
                        if (contract->default_bounds.has_max)
                        {
                            if (!first_bound)
                            {
                                out << ", ";
                            }
                            out << "\"max\": " << contract->default_bounds.max_value;
                        }
                        out << "}";
                    }
                }

                out << "\n";
                out << "    }";
            };

            for (const auto& field : exported_core_fields)
            {
                write_field_entry(field.first, field.second, "core", false);
            }
            for (const auto& field : exported_derived_fields)
            {
                write_field_entry(field.first, field.second, "derived", true);
            }
            for (const auto& field : exported_tranche_fields)
            {
                write_field_entry(field.field_id, field.suffix, field.category, true);
            }

            out << "\n";
            out << "  ]\n";
            out << "}\n";
            return out.str();
        };

        auto write_step_manifest = [&](const std::filesystem::path& manifest_path) -> bool
        {
            std::ofstream out(manifest_path);
            if (!out)
            {
                return false;
            }
            out << build_manifest_json();
            return out.good();
        };

        const auto& rf = active_export_fields;
        const bool use_3d = (output_config.format == OutputFormat::npy_3d ||
                             output_config.format == OutputFormat::csv ||
                             output_config.format == OutputFormat::zfp);
        const std::string step_str = stepdir.str();

        // Async handoff is only used for 3D format with async_io enabled.
        // 2D slice mode interleaves computation and I/O per-theta, so it stays synchronous.
        const bool use_async = output_config.async_io && use_3d;
        ExportSnapshot snapshot;
        if (use_async)
        {
            snapshot.export_index = export_index;
            snapshot.simulation_time_s = simulation_time;
            snapshot.step_dir = step_path;
        }

        // --- 3D mode: write core fields as full volumes directly from Field3D ---
        if (use_3d)
        {
            struct CoreBinding { const char* name; const Field3D* field; };
            const CoreBinding core_bindings[] = {
                {"u", &u}, {"v", &v_theta}, {"w", &w},
                {"rho", &rho}, {"p", &p}, {"theta", &theta},
                {"qv", &qv}, {"qc", &qc}, {"qr", &qr},
                {"qi", &qi}, {"qs", &qs}, {"qh", &qh}, {"qg", &qg},
                {"radar", &radar_reflectivity}, {"tracer", &tracer},
                {"vorticity_r", &vorticity_r}, {"vorticity_theta", &vorticity_theta},
                {"vorticity_z", &vorticity_z},
                {"stretching_term", &stretching_term}, {"tilting_term", &tilting_term},
                {"baroclinic_term", &baroclinic_term},
                {"p_prime", &p_prime}, {"dynamic_pressure", &dynamic_pressure},
                {"buoyancy_pressure", &buoyancy_pressure},
                {"angular_momentum", &angular_momentum},
                {"angular_momentum_tendency", &angular_momentum_tendency},
            };
            for (const auto& cb : core_bindings)
            {
                if (rf.count(cb.name))
                {
                    if (use_async)
                    {
                        // Deep-copy field data for background writer
                        FieldSnapshotEntry entry;
                        entry.name = cb.name;
                        entry.dim0 = NR;
                        entry.dim1 = NTH;
                        entry.dim2 = NZ;
                        entry.is_3d = true;
                        entry.data.assign(cb.field->data(),
                                          cb.field->data() + cb.field->size());
                        snapshot.fields.push_back(std::move(entry));
                    }
                    else if (output_config.format == OutputFormat::csv)
                    {
                        csv::write_field3d(*cb.field, step_str + "/" + cb.name + ".csv");
                    }
                    else
                    {
                        npy::write_field3d(*cb.field, step_str + "/" + cb.name + ".npy");
                    }
                }
            }
        }

        // --- Pre-allocate 3D accumulation buffers for derived fields in 3D mode ---
        // Maps derived field name → contiguous [NR][NTH][NZ] buffer
        std::unordered_map<std::string, std::vector<float>> derived_3d_bufs;
        if (use_3d)
        {
            const auto vol_size = static_cast<std::size_t>(NR) *
                                  static_cast<std::size_t>(NTH) *
                                  static_cast<std::size_t>(NZ);
            for (const auto& df : exported_derived_fields)
            {
                if (rf.count(df.first))
                {
                    derived_3d_bufs[df.first].resize(vol_size, 0.0f);
                }
            }
        }

        // --- Helper: scatter a (NZ, NR) slice into a [NR][NTH][NZ] 3D buffer ---
        auto scatter_slice_to_3d = [&](const std::vector<float>& slice,
                                       std::vector<float>& buf_3d, int th)
        {
            // slice is [k * NR + i] (NZ-major), buf_3d is [i * NTH * NZ + j * NZ + k]
            for (int k = 0; k < NZ; ++k)
            {
                for (int i = 0; i < NR; ++i)
                {
                    buf_3d[static_cast<std::size_t>(i) * static_cast<std::size_t>(NTH) *
                           static_cast<std::size_t>(NZ) +
                           static_cast<std::size_t>(th) * static_cast<std::size_t>(NZ) +
                           static_cast<std::size_t>(k)] =
                        slice[static_cast<std::size_t>(k) * static_cast<std::size_t>(NR) +
                              static_cast<std::size_t>(i)];
                }
            }
        };

        std::vector<float> tranche_slice_buffer;
        for (int th = 0; th < NTH; ++th)
        {
            // --- 2D slice mode: write core fields per-theta ---
            if (!use_3d)
            {
                std::string base_path = step_str + "/th" + std::to_string(th);
                if (rf.count("u")) save_field_slice_npy(u, th, base_path + "_u.npy");
                if (rf.count("v")) save_field_slice_npy(v_theta, th, base_path + "_v.npy");
                if (rf.count("w")) save_field_slice_npy(w, th, base_path + "_w.npy");
                if (rf.count("rho")) save_field_slice_npy(rho, th, base_path + "_rho.npy");
                if (rf.count("p")) save_field_slice_npy(p, th, base_path + "_p.npy");
                if (rf.count("theta")) save_field_slice_npy(theta, th, base_path + "_theta.npy");
                if (rf.count("qv")) save_field_slice_npy(qv, th, base_path + "_qv.npy");
                if (rf.count("qc")) save_field_slice_npy(qc, th, base_path + "_qc.npy");
                if (rf.count("qr")) save_field_slice_npy(qr, th, base_path + "_qr.npy");
                if (rf.count("qi")) save_field_slice_npy(qi, th, base_path + "_qi.npy");
                if (rf.count("qs")) save_field_slice_npy(qs, th, base_path + "_qs.npy");
                if (rf.count("qh")) save_field_slice_npy(qh, th, base_path + "_qh.npy");
                if (rf.count("qg")) save_field_slice_npy(qg, th, base_path + "_qg.npy");
                if (rf.count("radar")) save_field_slice_npy(radar_reflectivity, th, base_path + "_radar.npy");
                if (rf.count("tracer")) save_field_slice_npy(tracer, th, base_path + "_tracer.npy");
                if (rf.count("vorticity_r")) save_field_slice_npy(vorticity_r, th, base_path + "_vorticity_r.npy");
                if (rf.count("vorticity_theta")) save_field_slice_npy(vorticity_theta, th, base_path + "_vorticity_theta.npy");
                if (rf.count("vorticity_z")) save_field_slice_npy(vorticity_z, th, base_path + "_vorticity_z.npy");
                if (rf.count("stretching_term")) save_field_slice_npy(stretching_term, th, base_path + "_stretching_term.npy");
                if (rf.count("tilting_term")) save_field_slice_npy(tilting_term, th, base_path + "_tilting_term.npy");
                if (rf.count("baroclinic_term")) save_field_slice_npy(baroclinic_term, th, base_path + "_baroclinic_term.npy");
                if (rf.count("p_prime")) save_field_slice_npy(p_prime, th, base_path + "_p_prime.npy");
                if (rf.count("dynamic_pressure")) save_field_slice_npy(dynamic_pressure, th, base_path + "_dynamic_pressure.npy");
                if (rf.count("buoyancy_pressure")) save_field_slice_npy(buoyancy_pressure, th, base_path + "_buoyancy_pressure.npy");
                if (rf.count("angular_momentum")) save_field_slice_npy(angular_momentum, th, base_path + "_angular_momentum.npy");
                if (rf.count("angular_momentum_tendency")) save_field_slice_npy(angular_momentum_tendency, th, base_path + "_angular_momentum_tendency.npy");
            }

            std::size_t idx = 0;
            for (int k = 0; k < NZ; ++k)
            {
                for (int i = 0; i < NR; ++i)
                {
                    const float p_pa = static_cast<float>(p[i][th][k]);
                    const float theta_value = static_cast<float>(theta[i][th][k]);
                    const float qv_value = static_cast<float>(qv[i][th][k]);
                    if (std::isfinite(p_pa) && std::isfinite(theta_value) && p_pa > 0.0f)
                    {
                        temperature_slice[idx] = theta_value * std::pow(p_pa / p0f, kappa);
                    }
                    else
                    {
                        temperature_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    const float temperature_k = temperature_slice[idx];
                    if (std::isfinite(temperature_k) && std::isfinite(p_pa) && p_pa > 0.0f)
                    {
                        const float p_hpa = p_pa * 0.01f;
                        float es_hpa = 6.112f * std::exp(
                            17.67f * (temperature_k - t_freezing_k) / (temperature_k - 29.65f));
                        if (std::isfinite(es_hpa) && p_hpa > 1.0e-3f)
                        {
                            es_hpa = std::max(0.0f, std::min(es_hpa, 0.99f * p_hpa));
                            const float denom = std::max(1.0e-3f, p_hpa - es_hpa);
                            float qsat = 0.622f * es_hpa / denom;

                            qsat = std::max(1.0e-8f, std::min(0.10f, qsat));
                            saturation_mixing_ratio_slice[idx] = qsat;

                            if (std::isfinite(qv_value) && std::isfinite(qsat) && qsat > 0.0f)
                            {
                                const float rh = std::max(0.0f, std::min(200.0f, 100.0f * (qv_value / qsat)));
                                relative_humidity_slice[idx] = rh;
                                const float rh_frac = std::max(1.0e-6f, rh * 0.01f);
                                const float temperature_c = temperature_k - t_freezing_k;
                                const float gamma = std::log(rh_frac) +
                                    (17.625f * temperature_c) / (243.04f + temperature_c);
                                const float gamma_denom = 17.625f - gamma;
                                if (std::abs(gamma_denom) > 1.0e-6f)
                                {
                                    const float td_c = 243.04f * gamma / gamma_denom;
                                    dewpoint_slice[idx] = td_c + t_freezing_k;
                                }
                            }
                        }
                    }

                    const std::size_t mean_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(NZ) + static_cast<std::size_t>(k);
                    theta_prime_slice[idx] = theta_value - theta_azimuth_mean[mean_idx];

                    const float qc_value = std::max(0.0f, static_cast<float>(qc[i][th][k]));
                    const float qr_value = std::max(0.0f, static_cast<float>(qr[i][th][k]));
                    const float qi_value = std::max(0.0f, static_cast<float>(qi[i][th][k]));
                    const float qs_value = std::max(0.0f, static_cast<float>(qs[i][th][k]));
                    const float qg_value = std::max(0.0f, static_cast<float>(qg[i][th][k]));
                    const float qh_value = std::max(0.0f, static_cast<float>(qh[i][th][k]));

                    total_condensate_slice[idx] = qc_value + qr_value + qi_value + qs_value + qg_value + qh_value;

                    const float qv_nonnegative = std::isfinite(qv_value) ? std::max(0.0f, qv_value) : std::numeric_limits<float>::quiet_NaN();

                    if (std::isfinite(theta_value) &&std::isfinite(qv_nonnegative) && std::isfinite(total_condensate_slice[idx]))
                    {
                        theta_v_slice[idx] = theta_value * (1.0f + (0.61f * qv_nonnegative) - total_condensate_slice[idx]);
                    }
                    else
                    {
                        theta_v_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    if (std::isfinite(theta_value) &&
                        std::isfinite(qv_nonnegative) &&
                        std::isfinite(temperature_k) &&
                        temperature_k > 0.0f)
                    {
                        const float exponent = (latent_heat_v * qv_nonnegative) / (cp_f * temperature_k);
                        theta_e_slice[idx] = theta_value * std::exp(exponent);
                    }
                    else
                    {
                        theta_e_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    const float z_linear = static_cast<float>(radar_reflectivity[i][th][k]);
                    if (!std::isfinite(z_linear))
                    {
                        reflectivity_dbz_slice[idx] = reflectivity_dbz_min;
                    }
                    else if (z_linear <= 0.0f)
                    {
                        reflectivity_dbz_slice[idx] = reflectivity_dbz_min;
                    }
                    else
                    {
                        float dbz_value = 10.0f * std::log10(z_linear);
                        if (!std::isfinite(dbz_value))
                        {
                            dbz_value = reflectivity_dbz_min;
                        }
                        reflectivity_dbz_slice[idx] =
                            std::max(reflectivity_dbz_min, std::min(reflectivity_dbz_max, dbz_value));
                    }

                    const float vort_r = static_cast<float>(vorticity_r[i][th][k]);
                    const float vort_theta = static_cast<float>(vorticity_theta[i][th][k]);
                    const float vort_z = static_cast<float>(vorticity_z[i][th][k]);
                    if (std::isfinite(vort_r) && std::isfinite(vort_theta) && std::isfinite(vort_z))
                    {
                        vorticity_magnitude_slice[idx] = std::sqrt(
                            (vort_r * vort_r) + (vort_theta * vort_theta) + (vort_z * vort_z));
                    }
                    else
                    {
                        vorticity_magnitude_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    const int th_plus = (th + 1) % NTH;
                    const int th_minus = (th + NTH - 1) % NTH;
                    const float u_center = static_cast<float>(u[i][th][k]);
                    const float v_center = static_cast<float>(v_theta[i][th][k]);
                    const float w_center = static_cast<float>(w[i][th][k]);
                    const float u_plus = static_cast<float>(u[(i + 1 < NR) ? i + 1 : i][th][k]);
                    const float u_minus = static_cast<float>(u[(i > 0) ? i - 1 : i][th][k]);
                    const float u_th_plus = static_cast<float>(u[i][th_plus][k]);
                    const float u_th_minus = static_cast<float>(u[i][th_minus][k]);
                    const float v_plus = static_cast<float>(v_theta[i][th_plus][k]);
                    const float v_minus = static_cast<float>(v_theta[i][th_minus][k]);
                    const float v_r_plus = static_cast<float>(v_theta[(i + 1 < NR) ? i + 1 : i][th][k]);
                    const float v_r_minus = static_cast<float>(v_theta[(i > 0) ? i - 1 : i][th][k]);
                    const float w_plus = static_cast<float>(w[i][th][(k + 1 < NZ) ? k + 1 : k]);
                    const float w_minus = static_cast<float>(w[i][th][(k > 0) ? k - 1 : k]);

                    float radial_term = std::numeric_limits<float>::quiet_NaN();
                    if (std::isfinite(u_center) && std::isfinite(u_plus) && std::isfinite(u_minus) && dr > 0.0)
                    {
                        const float dr_f = static_cast<float>(dr);
                        if (i == 0 && NR > 1)
                        {
                            radial_term = 2.0f * (u_plus - u_center) / dr_f;
                        }
                        else
                        {
                            const float r_center = static_cast<float>(i) * dr_f;
                            if (r_center > 0.0f)
                            {
                                const float r_plus = static_cast<float>(std::min(i + 1, NR - 1)) * dr_f;
                                const float r_minus = static_cast<float>((i > 0) ? i - 1 : 0) * dr_f;
                                const float denominator = (i == NR - 1) ? dr_f : (2.0f * dr_f);
                                const float d_ru_dr = ((r_plus * u_plus) - (r_minus * u_minus)) / denominator;
                                radial_term = d_ru_dr / r_center;
                            }
                        }
                    }

                    float azimuthal_term = 0.0f;
                    if (i > 0 && dtheta > 0.0 && dr > 0.0 &&
                        std::isfinite(v_plus) && std::isfinite(v_minus))
                    {
                        const float r_center = static_cast<float>(i) * static_cast<float>(dr);
                        if (r_center > 0.0f)
                        {
                            azimuthal_term =
                                (v_plus - v_minus) / (2.0f * static_cast<float>(dtheta) * r_center);
                        }
                    }

                    float vertical_term = std::numeric_limits<float>::quiet_NaN();
                    if (std::isfinite(w_plus) && std::isfinite(w_minus) && dz > 0.0)
                    {
                        const float dz_f = static_cast<float>(dz);
                        const float denominator = (k == 0 || k == NZ - 1) ? dz_f : (2.0f * dz_f);
                        vertical_term = (w_plus - w_minus) / denominator;
                    }

                    if (std::isfinite(radial_term) && std::isfinite(azimuthal_term) && std::isfinite(vertical_term))
                    {
                        divergence_slice[idx] = radial_term + azimuthal_term + vertical_term;
                    }
                    else
                    {
                        divergence_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    const float theta_mean = theta_azimuth_mean[mean_idx];
                    if (std::isfinite(theta_prime_slice[idx]) &&
                        std::isfinite(theta_mean) &&
                        std::abs(theta_mean) > 1.0e-6f)
                    {
                        buoyancy_slice[idx] = g_f * (theta_prime_slice[idx] / theta_mean);
                    }
                    else
                    {
                        buoyancy_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    if (std::isfinite(vort_r) &&
                        std::isfinite(vort_theta) &&
                        std::isfinite(u_center) &&
                        std::isfinite(v_center))
                    {
                        const float speed_h = std::sqrt((u_center * u_center) + (v_center * v_center));
                        if (speed_h > 1.0e-6f)
                        {
                            horizontal_vorticity_streamwise_slice[idx] =
                                ((vort_r * u_center) + (vort_theta * v_center)) / speed_h;
                            horizontal_vorticity_crosswise_slice[idx] =
                                ((-vort_r * v_center) + (vort_theta * u_center)) / speed_h;
                        }
                        else
                        {
                            horizontal_vorticity_streamwise_slice[idx] = 0.0f;
                            horizontal_vorticity_crosswise_slice[idx] = 0.0f;
                        }
                    }
                    else
                    {
                        horizontal_vorticity_streamwise_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                        horizontal_vorticity_crosswise_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    const float storm_u_mean = storm_u_level_mean[static_cast<std::size_t>(k)];
                    const float storm_v_mean = storm_v_level_mean[static_cast<std::size_t>(k)];
                    if (std::isfinite(u_center) &&
                        std::isfinite(v_center) &&
                        std::isfinite(storm_u_mean) &&
                        std::isfinite(storm_v_mean))
                    {
                        const float u_sr = u_center - storm_u_mean;
                        const float v_sr = v_center - storm_v_mean;
                        storm_relative_winds_slice[idx] = std::sqrt((u_sr * u_sr) + (v_sr * v_sr));
                    }
                    else
                    {
                        storm_relative_winds_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    if (std::isfinite(u_center) &&
                        std::isfinite(v_center) &&
                        std::isfinite(w_center) &&
                        std::isfinite(vort_r) &&
                        std::isfinite(vort_theta) &&
                        std::isfinite(vort_z))
                    {
                        helicity_density_slice[idx] =
                            (u_center * vort_r) +
                            (v_center * vort_theta) +
                            (w_center * vort_z);
                    }
                    else
                    {
                        helicity_density_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    if (dr > 0.0 && dtheta > 0.0)
                    {
                        const float dr_f = static_cast<float>(dr);
                        const float radial_denom = (i == 0 || i == NR - 1) ? dr_f : (2.0f * dr_f);
                        float du_dr = std::numeric_limits<float>::quiet_NaN();
                        float dv_dr = std::numeric_limits<float>::quiet_NaN();
                        if (std::isfinite(u_plus) && std::isfinite(u_minus))
                        {
                            du_dr = (u_plus - u_minus) / radial_denom;
                        }
                        if (std::isfinite(v_r_plus) && std::isfinite(v_r_minus))
                        {
                            dv_dr = (v_r_plus - v_r_minus) / radial_denom;
                        }

                        float du_dy = 0.0f;
                        float dv_dy = 0.0f;
                        bool azimuthal_derivatives_valid = false;
                        const float r_center = static_cast<float>(i) * dr_f;
                        if (r_center > 0.0f &&
                            std::isfinite(u_th_plus) &&
                            std::isfinite(u_th_minus) &&
                            std::isfinite(v_plus) &&
                            std::isfinite(v_minus))
                        {
                            const float inv_arc = 1.0f / (2.0f * static_cast<float>(dtheta) * r_center);
                            du_dy = (u_th_plus - u_th_minus) * inv_arc;
                            dv_dy = (v_plus - v_minus) * inv_arc;
                            azimuthal_derivatives_valid = true;
                        }
                        else if (r_center <= 0.0f)
                        {
                            du_dy = 0.0f;
                            dv_dy = 0.0f;
                            azimuthal_derivatives_valid = true;
                        }

                        if (std::isfinite(du_dr) &&
                            std::isfinite(dv_dr) &&
                            azimuthal_derivatives_valid &&
                            std::isfinite(vort_z))
                        {
                            const float normal_strain = du_dr - dv_dy;
                            const float shear_strain = dv_dr + du_dy;
                            okubo_weiss_slice[idx] =
                                (normal_strain * normal_strain) +
                                (shear_strain * shear_strain) -
                                (vort_z * vort_z);
                        }
                        else
                        {
                            okubo_weiss_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                        }
                    }
                    else
                    {
                        okubo_weiss_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    const float p_center = static_cast<float>(p[i][th][k]);
                    const float p_r_plus = static_cast<float>(p[(i + 1 < NR) ? i + 1 : i][th][k]);
                    const float p_r_minus = static_cast<float>(p[(i > 0) ? i - 1 : i][th][k]);
                    const float p_th_plus = static_cast<float>(p[i][th_plus][k]);
                    const float p_th_minus = static_cast<float>(p[i][th_minus][k]);
                    const float p_z_plus = static_cast<float>(p[i][th][(k + 1 < NZ) ? k + 1 : k]);
                    const float p_z_minus = static_cast<float>(p[i][th][(k > 0) ? k - 1 : k]);
                    const float rho_center = static_cast<float>(rho[i][th][k]);

                    if (std::isfinite(p_center) &&
                        std::isfinite(p_r_plus) &&
                        std::isfinite(p_r_minus) &&
                        std::isfinite(p_th_plus) &&
                        std::isfinite(p_th_minus) &&
                        std::isfinite(p_z_plus) &&
                        std::isfinite(p_z_minus) &&
                        std::isfinite(rho_center) &&
                        rho_center > 1.0e-6f &&
                        dr > 0.0 &&
                        dz > 0.0)
                    {
                        const float dr_f = static_cast<float>(dr);
                        const float dz_f = static_cast<float>(dz);
                        const float radial_denom = (i == 0 || i == NR - 1) ? dr_f : (2.0f * dr_f);
                        const float vertical_denom = (k == 0 || k == NZ - 1) ? dz_f : (2.0f * dz_f);
                        const float dp_dr = (p_r_plus - p_r_minus) / radial_denom;
                        const float dp_dz = (p_z_plus - p_z_minus) / vertical_denom;

                        float dp_dy = 0.0f;
                        const float r_center = static_cast<float>(i) * dr_f;
                        if (r_center > 0.0f && dtheta > 0.0)
                        {
                            dp_dy = (p_th_plus - p_th_minus) /
                                (2.0f * static_cast<float>(dtheta) * r_center);
                        }

                        pressure_gradient_force_x_slice[idx] = -dp_dr / rho_center;
                        pressure_gradient_force_y_slice[idx] = -dp_dy / rho_center;
                        pressure_gradient_force_z_slice[idx] = -dp_dz / rho_center;
                    }
                    else
                    {
                        pressure_gradient_force_x_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                        pressure_gradient_force_y_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                        pressure_gradient_force_z_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    if (std::isfinite(temperature_k) &&
                        std::isfinite(relative_humidity_slice[idx]) &&
                        std::isfinite(p_pa) &&
                        p_pa > 0.0f)
                    {
                        const float rh_clamped = std::max(1.0e-3f, std::min(100.0f, relative_humidity_slice[idx]));
                        const float t_c = temperature_k - t_freezing_k;
                        const float tw_c =
                            (t_c * std::atan(0.151977f * std::sqrt(rh_clamped + 8.313659f))) +
                            std::atan(t_c + rh_clamped) -
                            std::atan(rh_clamped - 1.676331f) +
                            (0.00391838f * std::pow(rh_clamped, 1.5f) * std::atan(0.023101f * rh_clamped)) -
                            4.686035f;
                        const float tw_k = tw_c + t_freezing_k;
                        if (std::isfinite(tw_k) && tw_k > 0.0f)
                        {
                            theta_w_slice[idx] = tw_k * std::pow(p0f / p_pa, kappa);
                        }
                        else
                        {
                            theta_w_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                        }
                    }
                    else
                    {
                        theta_w_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    if (std::isfinite(qr_value))
                    {
                        const float qrain_nonnegative = std::max(0.0f, qr_value);
                        if (qrain_nonnegative <= 1.0e-10f)
                        {
                            zdr_slice[idx] = 0.0f;
                        }
                        else
                        {
                            const float rain_rate_proxy = qrain_nonnegative * 3600.0f;
                            float axis_ratio = 0.95f;
                            if (rain_rate_proxy >= 10.0f)
                            {
                                axis_ratio = 0.75f;
                            }
                            else if (rain_rate_proxy >= 1.0f)
                            {
                                axis_ratio = 0.85f;
                            }
                            zdr_slice[idx] = -40.0f * std::log10(axis_ratio);
                        }

                        const float mixed_ice = qi_value + qs_value + qg_value + qh_value;
                        kdp_slice[idx] = std::max(0.0f, (1500.0f * qrain_nonnegative) + (200.0f * mixed_ice));
                        const float hydro_sum = qrain_nonnegative + mixed_ice + 1.0e-8f;
                        const float mixed_fraction = mixed_ice / hydro_sum;
                        float rhohv = 1.0f - (0.35f * mixed_fraction) -
                            (0.05f * std::min(1.0f, qrain_nonnegative / 0.005f));
                        rhohv = std::max(0.5f, std::min(1.0f, rhohv));
                        rhohv_slice[idx] = rhohv;
                    }
                    else
                    {
                        zdr_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                        kdp_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                        rhohv_slice[idx] = std::numeric_limits<float>::quiet_NaN();
                    }

                    const float stream_speed =
                        (std::isfinite(u_center) && std::isfinite(v_center) && std::isfinite(w_center))
                            ? std::sqrt((u_center * u_center) + (v_center * v_center) + (w_center * w_center))
                            : 0.0f;
                    const float dt_f = std::isfinite(static_cast<float>(dt))
                        ? std::max(0.0f, static_cast<float>(dt))
                        : 0.0f;

                    streamlines_slice[idx] = stream_speed;
                    trajectory_paths_slice[idx] = stream_speed * dt_f;
                    forward_trajectories_slice[idx] = trajectory_paths_slice[idx];
                    backward_trajectories_slice[idx] = -trajectory_paths_slice[idx];
                    parcel_buoyancy_trajectory_slice[idx] =
                        std::isfinite(buoyancy_slice[idx]) ? (buoyancy_slice[idx] * dt_f) : 0.0f;
                    vorticity_trajectory_slice[idx] =
                        std::isfinite(vorticity_magnitude_slice[idx]) ? (vorticity_magnitude_slice[idx] * dt_f) : 0.0f;

                    const float ang_momentum = static_cast<float>(angular_momentum[i][th][k]);
                    circulation_material_surface_slice[idx] = std::isfinite(ang_momentum) ? ang_momentum : 0.0f;
                    cross_section_slice[idx] = std::isfinite(theta_value) ? theta_value : 0.0f;
                    rhi_slice_diag[idx] = std::isfinite(reflectivity_dbz_slice[idx])
                        ? reflectivity_dbz_slice[idx]
                        : reflectivity_dbz_min;
                    hodograph_aligned_cross_section_slice[idx] = std::isfinite(storm_relative_winds_slice[idx])
                        ? storm_relative_winds_slice[idx]
                        : 0.0f;

                    const float dr_f = static_cast<float>(dr);
                    const float dz_f = static_cast<float>(dz);
                    const float theta_center = theta_value;
                    const float theta_r_plus = static_cast<float>(theta[(i + 1 < NR) ? i + 1 : i][th][k]);
                    const float theta_r_minus = static_cast<float>(theta[(i > 0) ? i - 1 : i][th][k]);
                    const float theta_th_plus = static_cast<float>(theta[i][th_plus][k]);
                    const float theta_th_minus = static_cast<float>(theta[i][th_minus][k]);
                    const float theta_z_plus = static_cast<float>(theta[i][th][(k + 1 < NZ) ? k + 1 : k]);
                    const float theta_z_minus = static_cast<float>(theta[i][th][(k > 0) ? k - 1 : k]);
                    const float u_z_plus = static_cast<float>(u[i][th][(k + 1 < NZ) ? k + 1 : k]);
                    const float u_z_minus = static_cast<float>(u[i][th][(k > 0) ? k - 1 : k]);

                    float dtheta_dr = 0.0f;
                    float dtheta_dy = 0.0f;
                    float dtheta_dz = 0.0f;
                    float du_dr = 0.0f;
                    float dv_dr = 0.0f;
                    float du_dz = 0.0f;
                    bool gradients_valid = false;
                    if (std::isfinite(theta_center) &&
                        std::isfinite(theta_r_plus) &&
                        std::isfinite(theta_r_minus) &&
                        std::isfinite(theta_th_plus) &&
                        std::isfinite(theta_th_minus) &&
                        std::isfinite(theta_z_plus) &&
                        std::isfinite(theta_z_minus) &&
                        std::isfinite(u_plus) &&
                        std::isfinite(u_minus) &&
                        std::isfinite(v_r_plus) &&
                        std::isfinite(v_r_minus) &&
                        std::isfinite(u_z_plus) &&
                        std::isfinite(u_z_minus) &&
                        dr_f > 0.0f &&
                        dz_f > 0.0f)
                    {
                        const float radial_denom = (i == 0 || i == NR - 1) ? dr_f : (2.0f * dr_f);
                        const float vertical_denom = (k == 0 || k == NZ - 1) ? dz_f : (2.0f * dz_f);
                        dtheta_dr = (theta_r_plus - theta_r_minus) / radial_denom;
                        dtheta_dz = (theta_z_plus - theta_z_minus) / vertical_denom;
                        du_dr = (u_plus - u_minus) / radial_denom;
                        dv_dr = (v_r_plus - v_r_minus) / radial_denom;
                        du_dz = (u_z_plus - u_z_minus) / vertical_denom;

                        const float r_center = static_cast<float>(i) * dr_f;
                        if (r_center > 0.0f && dtheta > 0.0)
                        {
                            dtheta_dy = (theta_th_plus - theta_th_minus) /
                                (2.0f * static_cast<float>(dtheta) * r_center);
                        }
                        else
                        {
                            dtheta_dy = 0.0f;
                        }
                        gradients_valid = true;
                    }

                    if (gradients_valid)
                    {
                        const float qx = -((du_dr * dtheta_dr) + (dv_dr * dtheta_dy));
                        const float qy = -((dv_dr * dtheta_dr) - (du_dr * dtheta_dy));
                        const float qz = -(du_dz * dtheta_dz);
                        const float q_mag = std::sqrt((qx * qx) + (qy * qy) + (qz * qz));
                        q_vectors_slice[idx] = std::isfinite(q_mag) ? q_mag : 0.0f;

                        float d2theta_dr2 = 0.0f;
                        float d2theta_dy2 = 0.0f;
                        float d2theta_dz2 = (theta_z_plus - (2.0f * theta_center) + theta_z_minus) /
                            std::max(1.0e-6f, dz_f * dz_f);
                        d2theta_dr2 = (theta_r_plus - (2.0f * theta_center) + theta_r_minus) /
                            std::max(1.0e-6f, dr_f * dr_f);
                        const float r_center = static_cast<float>(i) * dr_f;
                        if (r_center > 0.0f && dtheta > 0.0)
                        {
                            const float arc = r_center * static_cast<float>(dtheta);
                            d2theta_dy2 = (theta_th_plus - (2.0f * theta_center) + theta_th_minus) /
                                std::max(1.0e-6f, arc * arc);
                        }
                        const float diffusion_proxy = d2theta_dr2 + d2theta_dy2 + d2theta_dz2;
                        turbulent_diffusion_term_slice[idx] = std::isfinite(diffusion_proxy) ? diffusion_proxy : 0.0f;
                    }
                    else
                    {
                        q_vectors_slice[idx] = 0.0f;
                        turbulent_diffusion_term_slice[idx] = 0.0f;
                    }

                    ++idx;
                }
            }

            const std::pair<const char*, std::vector<float>*> derived_bindings[] = {
                {"temperature", &temperature_slice},
                {"theta_prime", &theta_prime_slice},
                {"theta_v", &theta_v_slice},
                {"theta_e", &theta_e_slice},
                {"dewpoint", &dewpoint_slice},
                {"relative_humidity", &relative_humidity_slice},
                {"saturation_mixing_ratio", &saturation_mixing_ratio_slice},
                {"total_condensate", &total_condensate_slice},
                {"reflectivity_dbz", &reflectivity_dbz_slice},
                {"vorticity_magnitude", &vorticity_magnitude_slice},
                {"divergence", &divergence_slice},
                {"buoyancy", &buoyancy_slice},
                {"horizontal_vorticity_streamwise", &horizontal_vorticity_streamwise_slice},
                {"horizontal_vorticity_crosswise", &horizontal_vorticity_crosswise_slice},
                {"pressure_gradient_force_x", &pressure_gradient_force_x_slice},
                {"pressure_gradient_force_y", &pressure_gradient_force_y_slice},
                {"pressure_gradient_force_z", &pressure_gradient_force_z_slice},
                {"storm_relative_winds", &storm_relative_winds_slice},
                {"helicity_density", &helicity_density_slice},
                {"okubo_weiss", &okubo_weiss_slice},
                {"theta_w", &theta_w_slice},
                {"zdr", &zdr_slice},
                {"kdp", &kdp_slice},
                {"rhohv", &rhohv_slice},
                {"streamlines", &streamlines_slice},
                {"trajectory_paths", &trajectory_paths_slice},
                {"q_vectors", &q_vectors_slice},
                {"turbulent_diffusion_term", &turbulent_diffusion_term_slice},
                {"cross_section", &cross_section_slice},
                {"rhi_slice", &rhi_slice_diag},
                {"hodograph_aligned_cross_section", &hodograph_aligned_cross_section_slice},
                {"forward_trajectories", &forward_trajectories_slice},
                {"backward_trajectories", &backward_trajectories_slice},
                {"parcel_buoyancy_trajectory", &parcel_buoyancy_trajectory_slice},
                {"vorticity_trajectory", &vorticity_trajectory_slice},
                {"circulation_material_surface", &circulation_material_surface_slice},
            };
            for (const auto& binding : derived_bindings)
            {
                if (!validate_derived_export_slice(binding.first, *binding.second, th))
                {
                    return false;
                }
            }

            // --- Derived field output: 3D scatter or 2D write ---
            if (use_3d)
            {
                // Scatter each computed slice into its 3D accumulation buffer
                for (const auto& binding : derived_bindings)
                {
                    auto it = derived_3d_bufs.find(binding.first);
                    if (it != derived_3d_bufs.end())
                    {
                        scatter_slice_to_3d(*binding.second, it->second, th);
                    }
                }
            }
            else
            {
                std::string base_path = step_str + "/th" + std::to_string(th);
                if (rf.count("temperature")) write_npy_2d(temperature_slice, base_path + "_temperature.npy");
                if (rf.count("theta_prime")) write_npy_2d(theta_prime_slice, base_path + "_theta_prime.npy");
                if (rf.count("theta_v")) write_npy_2d(theta_v_slice, base_path + "_theta_v.npy");
                if (rf.count("theta_e")) write_npy_2d(theta_e_slice, base_path + "_theta_e.npy");
                if (rf.count("dewpoint")) write_npy_2d(dewpoint_slice, base_path + "_dewpoint.npy");
                if (rf.count("relative_humidity")) write_npy_2d(relative_humidity_slice, base_path + "_relative_humidity.npy");
                if (rf.count("saturation_mixing_ratio")) write_npy_2d(saturation_mixing_ratio_slice, base_path + "_saturation_mixing_ratio.npy");
                if (rf.count("total_condensate")) write_npy_2d(total_condensate_slice, base_path + "_total_condensate.npy");
                if (rf.count("reflectivity_dbz")) write_npy_2d(reflectivity_dbz_slice, base_path + "_reflectivity_dbz.npy");
                if (rf.count("vorticity_magnitude")) write_npy_2d(vorticity_magnitude_slice, base_path + "_vorticity_magnitude.npy");
                if (rf.count("divergence")) write_npy_2d(divergence_slice, base_path + "_divergence.npy");
                if (rf.count("buoyancy")) write_npy_2d(buoyancy_slice, base_path + "_buoyancy.npy");
                if (rf.count("horizontal_vorticity_streamwise")) write_npy_2d(horizontal_vorticity_streamwise_slice, base_path + "_horizontal_vorticity_streamwise.npy");
                if (rf.count("horizontal_vorticity_crosswise")) write_npy_2d(horizontal_vorticity_crosswise_slice, base_path + "_horizontal_vorticity_crosswise.npy");
                if (rf.count("pressure_gradient_force_x")) write_npy_2d(pressure_gradient_force_x_slice, base_path + "_pressure_gradient_force_x.npy");
                if (rf.count("pressure_gradient_force_y")) write_npy_2d(pressure_gradient_force_y_slice, base_path + "_pressure_gradient_force_y.npy");
                if (rf.count("pressure_gradient_force_z")) write_npy_2d(pressure_gradient_force_z_slice, base_path + "_pressure_gradient_force_z.npy");
                if (rf.count("storm_relative_winds")) write_npy_2d(storm_relative_winds_slice, base_path + "_storm_relative_winds.npy");
                if (rf.count("helicity_density")) write_npy_2d(helicity_density_slice, base_path + "_helicity_density.npy");
                if (rf.count("okubo_weiss")) write_npy_2d(okubo_weiss_slice, base_path + "_okubo_weiss.npy");
                if (rf.count("theta_w")) write_npy_2d(theta_w_slice, base_path + "_theta_w.npy");
                if (rf.count("zdr")) write_npy_2d(zdr_slice, base_path + "_zdr.npy");
                if (rf.count("kdp")) write_npy_2d(kdp_slice, base_path + "_kdp.npy");
                if (rf.count("rhohv")) write_npy_2d(rhohv_slice, base_path + "_rhohv.npy");
                if (rf.count("streamlines")) write_npy_2d(streamlines_slice, base_path + "_streamlines.npy");
                if (rf.count("trajectory_paths")) write_npy_2d(trajectory_paths_slice, base_path + "_trajectory_paths.npy");
                if (rf.count("q_vectors")) write_npy_2d(q_vectors_slice, base_path + "_q_vectors.npy");
                if (rf.count("turbulent_diffusion_term")) write_npy_2d(turbulent_diffusion_term_slice, base_path + "_turbulent_diffusion_term.npy");
                if (rf.count("cross_section")) write_npy_2d(cross_section_slice, base_path + "_cross_section.npy");
                if (rf.count("rhi_slice")) write_npy_2d(rhi_slice_diag, base_path + "_rhi_slice.npy");
                if (rf.count("hodograph_aligned_cross_section")) write_npy_2d(hodograph_aligned_cross_section_slice, base_path + "_hodograph_aligned_cross_section.npy");
                if (rf.count("forward_trajectories")) write_npy_2d(forward_trajectories_slice, base_path + "_forward_trajectories.npy");
                if (rf.count("backward_trajectories")) write_npy_2d(backward_trajectories_slice, base_path + "_backward_trajectories.npy");
                if (rf.count("parcel_buoyancy_trajectory")) write_npy_2d(parcel_buoyancy_trajectory_slice, base_path + "_parcel_buoyancy_trajectory.npy");
                if (rf.count("vorticity_trajectory")) write_npy_2d(vorticity_trajectory_slice, base_path + "_vorticity_trajectory.npy");
                if (rf.count("circulation_material_surface")) write_npy_2d(circulation_material_surface_slice, base_path + "_circulation_material_surface.npy");
            }

            // --- Tranche field output ---
            for (const auto& product : tranche_registry)
            {
                if (!rf.count(product.field_id))
                {
                    continue;
                }
                if (!product.compute_slice(th, tranche_slice_buffer))
                {
                    std::cerr << "[EXPORT] failed to compute tranche diagnostic field='"
                              << product.field_id << "' step=" << export_index
                              << " theta=" << th << std::endl;
                    return false;
                }
                if (!validate_derived_export_slice(product.field_id, tranche_slice_buffer, th))
                {
                    return false;
                }
                if (!use_3d)
                {
                    std::string base_path = step_str + "/th" + std::to_string(th);
                    write_npy_2d(tranche_slice_buffer, base_path + "_" + product.suffix + ".npy");
                }
                // Tranche fields in 3D mode: write per-theta (they are summary fields)
                // A future enhancement could accumulate these into 3D volumes too
            }
        }

        // --- 3D mode: write accumulated derived field volumes ---
        if (use_3d)
        {
            for (auto& [name, buf] : derived_3d_bufs)
            {
                if (use_async)
                {
                    // Move derived buffer into snapshot (avoids copy — buffer is local)
                    FieldSnapshotEntry entry;
                    entry.name = name;
                    entry.dim0 = NR;
                    entry.dim1 = NTH;
                    entry.dim2 = NZ;
                    entry.is_3d = true;
                    entry.data = std::move(buf);
                    snapshot.fields.push_back(std::move(entry));
                }
                else if (output_config.format == OutputFormat::csv)
                {
                    csv::write_3d(buf.data(), NR, NTH, NZ,
                                  step_str + "/" + name + ".csv");
                }
                else
                {
                    npy::write_3d(buf.data(), NR, NTH, NZ,
                                  step_str + "/" + name + ".npy");
                }
            }
        }

        accumulated_rainfall_last_update_s = simulation_time;

        if (use_async)
        {
            // Serialize manifest and submit snapshot for background writing
            snapshot.manifest_json = build_manifest_json();
            if (!async_writer.submit(std::move(snapshot)))
            {
                std::cerr << "[EXPORT] Async writer error: "
                          << async_writer.error_message() << std::endl;
                return false;
            }
        }
        else
        {
            if (!write_step_manifest(step_path / "manifest.json"))
            {
                std::cerr << "[EXPORT] failed to write step manifest: "
                          << (step_path / "manifest.json") << std::endl;
                return false;
            }
        }
        return true;
    };

#endif

    if (write_every_s > 0)
    {
        std::filesystem::create_directories(outdir);

        static const std::regex step_pattern(R"(^step_[0-9]{6}$)");
        for (const auto& entry : std::filesystem::directory_iterator(outdir))
        {
            if (!entry.is_directory())
            {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (std::regex_match(name, step_pattern))
            {
                std::error_code ec;
                std::filesystem::remove_all(entry.path(), ec);
            }
        }
    }

    auto lastGuiExport = std::chrono::steady_clock::now();
    int steps = 0;
    int export_index = 0;
    ::simulation_time = 0.0;
    double next_field_export_time_s = (write_every_s > 0) ? static_cast<double>(write_every_s) : -1.0;
    double next_medium_export_time_s = (output_config.tiered_write_cadence && write_every_s > 0)
        ? static_cast<double>(output_config.write_cadence_medium_s) : -1.0;
    double next_slow_export_time_s = (output_config.tiered_write_cadence && write_every_s > 0)
        ? static_cast<double>(output_config.write_cadence_slow_s) : -1.0;

    using PerfClock = std::chrono::steady_clock;
    /**
     * @brief Aggregated runtime timing totals for headless simulation profiling.
     */
    struct PerfTotals
    {
        uint64_t steps = 0;
        double simulated_s = 0.0;
        double total_step_s = 0.0;
        double initial_export_s = 0.0;
        double radiation_s = 0.0;
        double boundary_layer_s = 0.0;
        double chaos_noise_s = 0.0;
        double chaos_tendency_s = 0.0;
        double dynamics_s = 0.0;
        double export_s = 0.0;
    } perf_totals;

    auto timed_call = [&](double& accumulator, auto&& fn)
    {
        if (!global_perf_timing_enabled)
        {
            fn();
            return;
        }
        const auto t0 = PerfClock::now();
        fn();
        const auto t1 = PerfClock::now();
        accumulator += std::chrono::duration<double>(t1 - t0).count();
    };

    if (global_perf_timing_enabled)
    {
        reset_advection_perf_stats();
    }
    
    if (verbose_export_debug)
    {
        float theta_min = 1e10, theta_max = -1e10;
        float u_min = 1e10, u_max = -1e10;
        int nan_count = 0;
        for (int i = 0; i < NR && i < 10; ++i) {
            for (int j = 0; j < NTH && j < 5; ++j) {
                for (int k = 0; k < NZ && k < 5; ++k) {
                    if (std::isnan(theta[i][j][k])) nan_count++;
                    if (theta[i][j][k] < theta_min) theta_min = theta[i][j][k];
                    if (theta[i][j][k] > theta_max) theta_max = theta[i][j][k];
                    if (u[i][j][k] < u_min) u_min = u[i][j][k];
                    if (u[i][j][k] > u_max) u_max = u[i][j][k];
                }
            }
        }
        std::cout << "\n[TIME STEP DEBUG] Before time stepping (t=0):" << std::endl;
        std::cout << "  Theta sample: min=" << theta_min << "K, max=" << theta_max << "K" << std::endl;
        std::cout << "  Wind (u) sample: min=" << u_min << "m/s, max=" << u_max << "m/s" << std::endl;
        std::cout << "  NaN count (sample): " << nan_count << std::endl;
        if (theta_min < 0 || theta_max > 500) {
            std::cerr << "  ⚠️  ERROR: Theta already corrupted before time stepping!" << std::endl;
        }
        std::cout << "  Headless duration/export cadence use simulation seconds" << std::endl;
        std::cout << std::endl;
    }

    // SHM live update: maps field names to global Field3D references
    auto shm_update = [&]()
    {
        if (!shm_writer.is_open())
        {
            return;
        }
        const auto& shm_fields = options.live_shm_fields;
        // Map of field name → Field3D pointer for core fields
        const std::pair<const char*, const Field3D*> core_map[] = {
            {"u", &u}, {"v", &v_theta}, {"w", &w},
            {"rho", &rho}, {"p", &p}, {"theta", &theta},
            {"qv", &qv}, {"qc", &qc}, {"qr", &qr},
            {"radar", &radar_reflectivity}, {"reflectivity_dbz", &radar_reflectivity},
            {"tracer", &tracer},
            {"vorticity_r", &vorticity_r}, {"vorticity_theta", &vorticity_theta},
            {"vorticity_z", &vorticity_z},
            {"stretching_term", &stretching_term}, {"tilting_term", &tilting_term},
            {"baroclinic_term", &baroclinic_term},
            {"p_prime", &p_prime}, {"dynamic_pressure", &dynamic_pressure},
            {"buoyancy_pressure", &buoyancy_pressure},
        };
        for (int fi = 0; fi < static_cast<int>(shm_fields.size()); ++fi)
        {
            for (const auto& [name, field_ptr] : core_map)
            {
                if (shm_fields[fi] == name)
                {
                    shm_writer.write_field(fi, *field_ptr);
                    break;
                }
            }
        }
        shm_writer.commit(simulation_time);
    };

#ifdef EXPORT_NPY
    if (write_every_s > 0)
    {
        // Initial export writes all fields regardless of cadence tier
        active_export_fields = output_config.resolved_fields;

        bool initial_export_ok = true;
        timed_call(perf_totals.initial_export_s, [&] {
            initial_export_ok = write_all_fields(export_index++);
        });
        if (!initial_export_ok)
        {
            std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
            return 2;
        }
        shm_update();
    }
#endif
    
    while (true)
    {
        const double runtime_dt = choose_runtime_timestep();
        if (std::isfinite(runtime_dt) && runtime_dt > 0.0)
        {
            dt = runtime_dt;
        }

        const auto step_t0 = PerfClock::now();

        timed_call(perf_totals.radiation_s, [&] { step_radiation(simulation_time); });
        if (!validate_core_fields("after_radiation", steps, false, nullptr, false))
        {
            std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
            return 2;
        }
        timed_call(perf_totals.boundary_layer_s, [&] { step_boundary_layer(simulation_time); });
        if (!validate_core_fields("after_boundary_layer", steps, false, nullptr, false))
        {
            std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
            return 2;
        }
        timed_call(perf_totals.chaos_noise_s, [&] { step_chaos_noise(dt); });
        if (!validate_core_fields("after_chaos_noise", steps, false, nullptr, false))
        {
            std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
            return 2;
        }
        timed_call(perf_totals.chaos_tendency_s, [&] { apply_chaos_tendencies(); });
        if (!validate_core_fields("after_chaos_tendency", steps, false, nullptr, false))
        {
            std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
            return 2;
        }
        timed_call(perf_totals.dynamics_s, [&] { step_dynamics(simulation_time); });
        if (!validate_core_fields("after_dynamics", steps, false, nullptr, false))
        {
            std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
            return 2;
        }
        simulation_time += dt;
        perf_totals.simulated_s += dt;

        // Stream fields to SHM every timestep for smooth live visualization.
        // The viewer polls the sequence number to detect new frames.
        shm_update();

        if (verbose_export_debug && steps % 10 == 0) 
        {
            float theta_min = 1e10, theta_max = -1e10;
            float u_min = 1e10, u_max = -1e10;
            int nan_count = 0, inf_count = 0;
            for (int i = 0; i < NR && i < 10; ++i) 
            {
                for (int j = 0; j < NTH && j < 5; ++j) 
                {
                    for (int k = 0; k < NZ && k < 5; ++k) 
                    {
                        if (std::isnan(theta[i][j][k])) nan_count++;
                        if (std::isinf(theta[i][j][k])) inf_count++;
                        if (theta[i][j][k] < theta_min) theta_min = theta[i][j][k];
                        if (theta[i][j][k] > theta_max) theta_max = theta[i][j][k];
                        if (u[i][j][k] < u_min) u_min = u[i][j][k];
                        if (u[i][j][k] > u_max) u_max = u[i][j][k];
                    }
                }
            }
            std::cout << "[TIME STEP DEBUG] Step " << steps << " (t=" << simulation_time << "s):" << std::endl;
            std::cout << "  Theta sample: min=" << theta_min << "K, max=" << theta_max << "K" << std::endl;
            std::cout << "  Wind (u) sample: min=" << u_min << "m/s, max=" << u_max << "m/s" << std::endl;
            std::cout << "  NaN/Inf count (sample): " << nan_count << "/" << inf_count << std::endl;

            // Full-field w_max scan (cheap: one pass, no allocation)
            float w_global_min = 1e10f, w_global_max = -1e10f;
            for (int i = 0; i < NR; ++i)
                for (int j = 0; j < NTH; ++j)
                    for (int k = 0; k < NZ; ++k)
                    {
                        const float wv = static_cast<float>(w[i][j][k]);
                        if (wv < w_global_min) w_global_min = wv;
                        if (wv > w_global_max) w_global_max = wv;
                    }
            std::cout << "  Vertical velocity (w): min=" << w_global_min
                      << "m/s, max=" << w_global_max << "m/s" << std::endl;

            if (theta_min < 0 || theta_max > 500 || std::abs(u_min) > 150 || std::abs(u_max) > 150)
            {
                std::cerr << "  ⚠️  WARNING: Values going wrong at step " << steps << "!" << std::endl;
            }
            std::cout << std::endl;
        }
#ifdef EXPORT_NPY
        if (export_ms > 0)
        {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - lastGuiExport).count() >= export_ms)
            {
                lastGuiExport = now;
                std::string tmp = std::string("data/.tracer_slice_th0.npy.tmp");
                std::string fin = std::string("data/tracer_slice_th0.npy");
                save_field_slice_npy(tracer, thetaIndex, tmp);
                std::rename(tmp.c_str(), fin.c_str());
            }
        }
        if (write_every_s > 0)
        {
            while (simulation_time + 1.0e-9 >= next_field_export_time_s)
            {
                // Populate active_export_fields for this tick.
                // When tiered cadence is disabled, all resolved fields are active.
                active_export_fields.clear();
                const bool medium_due = !output_config.tiered_write_cadence ||
                    (simulation_time + 1.0e-9 >= next_medium_export_time_s);
                const bool slow_due = !output_config.tiered_write_cadence ||
                    (simulation_time + 1.0e-9 >= next_slow_export_time_s);

                for (const auto& name : output_config.resolved_fields)
                {
                    if (!output_config.tiered_write_cadence)
                    {
                        active_export_fields.insert(name);
                    }
                    else
                    {
                        const WriteCadenceTier tier = get_field_write_cadence_tier(name);
                        if (tier == WriteCadenceTier::fast ||
                            (tier == WriteCadenceTier::medium && medium_due) ||
                            (tier == WriteCadenceTier::slow && slow_due))
                        {
                            active_export_fields.insert(name);
                        }
                    }
                }

                bool export_ok = true;
                timed_call(perf_totals.export_s, [&] {
                    export_ok = write_all_fields(export_index++);
                });
                if (!export_ok)
                {
                    std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
                    return 2;
                }
                shm_update();
                next_field_export_time_s += static_cast<double>(write_every_s);
                if (medium_due && output_config.tiered_write_cadence)
                {
                    next_medium_export_time_s +=
                        static_cast<double>(output_config.write_cadence_medium_s);
                }
                if (slow_due && output_config.tiered_write_cadence)
                {
                    next_slow_export_time_s +=
                        static_cast<double>(output_config.write_cadence_slow_s);
                }
            }
        }
#endif
        ++steps;
        if (global_perf_timing_enabled)
        {
            const auto step_t1 = PerfClock::now();
            const double step_wall_s = std::chrono::duration<double>(step_t1 - step_t0).count();
            perf_totals.total_step_s += step_wall_s;
            ++perf_totals.steps;

            {
                const AdvectionKernelStepTiming kt = current_advection_kernel_step_timing();
                std::cout << "[KERNEL STEP PERF] step=" << perf_totals.steps
                          << ", wall_ms=" << (step_wall_s * 1000.0)
                          << ", kernel_total_s=" << kt.kernel_total_s
                          << ", backend=" << kt.backend
                          << ", fallback=" << (kt.fallback_active ? "active" : "off")
                          << ", cpu_calls=" << kt.cpu_calls
                          << ", gpu_calls=" << kt.gpu_calls
                          << std::endl;
            }

            if (global_perf_report_every_steps > 0 &&
                (perf_totals.steps % static_cast<uint64_t>(global_perf_report_every_steps) == 0))
            {
                const double step_ms = (perf_totals.total_step_s / std::max<uint64_t>(1, perf_totals.steps)) * 1000.0;
                const double sim_seconds_per_wall_second =
                    perf_totals.simulated_s / std::max(1e-9, perf_totals.total_step_s);
                std::cout << "[PERF] steps=" << perf_totals.steps
                          << ", avg_step_ms=" << step_ms
                          << ", sim_s_per_wall_s=" << sim_seconds_per_wall_second
                          << std::endl;
            }
        }
        if (steps % 1000 == 0) { }
        if (duration_s >= 0 && simulation_time + 1.0e-9 >= static_cast<double>(duration_s))
        {
            break;
        }
    }

    if (write_every_s > 0 && !export_validation_reports.empty())
    {
        std::filesystem::path summary_path = std::filesystem::path(outdir) / "validation_summary.json";
        if (!global_validation_report_path.empty())
        {
            summary_path = std::filesystem::path(global_validation_report_path) / "validation_summary.json";
        }

        if (!write_validation_summary(summary_path))
        {
            std::cerr << "[VALIDATION] " << validation_error_message << std::endl;
            return 2;
        }
    }

    if (global_perf_timing_enabled && perf_totals.steps > 0)
    {
        const double step_total = std::max(perf_totals.total_step_s, 1e-9);
        const double total_profiled = step_total + perf_totals.initial_export_s;
        const auto pct_step = [&](double component) { return 100.0 * component / step_total; };
        std::cout << "\n[PERF SUMMARY] steps=" << perf_totals.steps
                  << ", step_wall_s=" << step_total
                  << ", avg_step_ms=" << (1000.0 * step_total / perf_totals.steps)
                  << ", sim_s_per_wall_s="
                  << (perf_totals.simulated_s / step_total)
                  << std::endl;
        std::cout << "  radiation_s=" << perf_totals.radiation_s << " (" << pct_step(perf_totals.radiation_s) << "%)" << std::endl;
        std::cout << "  boundary_layer_s=" << perf_totals.boundary_layer_s << " (" << pct_step(perf_totals.boundary_layer_s) << "%)" << std::endl;
        std::cout << "  chaos_noise_s=" << perf_totals.chaos_noise_s << " (" << pct_step(perf_totals.chaos_noise_s) << "%)" << std::endl;
        std::cout << "  chaos_tendency_s=" << perf_totals.chaos_tendency_s << " (" << pct_step(perf_totals.chaos_tendency_s) << "%)" << std::endl;
        std::cout << "  dynamics_s=" << perf_totals.dynamics_s << " (" << pct_step(perf_totals.dynamics_s) << "%)" << std::endl;
        std::cout << "  export_s=" << perf_totals.export_s << " (" << pct_step(perf_totals.export_s) << "%)" << std::endl;
        
        if (perf_totals.initial_export_s > 0.0)
        {
            std::cout << "  initial_export_s=" << perf_totals.initial_export_s << std::endl;
            std::cout << "  total_profiled_s=" << total_profiled << std::endl;
        }
        log_advection_perf_summary();
        std::cout << std::endl;
    }

    // Drain the async writer and report I/O statistics
    if (!async_writer.flush())
    {
        std::cerr << "[OUTPUT] Async writer error during flush: "
                  << async_writer.error_message() << std::endl;
        return 2;
    }
    if (async_writer.snapshots_written() > 0 && log_normal_enabled())
    {
        const double write_time = async_writer.total_write_time_s();
        const std::size_t total_bytes = async_writer.total_bytes_written();
        const double mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        std::cout << "[OUTPUT] " << async_writer.snapshots_written() << " snapshots written"
                  << ", " << mb << " MB total"
                  << ", " << write_time << "s write time"
                  << ", " << (write_time > 0.0 ? mb / write_time : 0.0) << " MB/s"
                  << std::endl;
    }

    return 0;
}
