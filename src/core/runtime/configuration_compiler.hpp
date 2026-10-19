#pragma once

#include "core/runtime/simulation_state.hpp"
#include "core/output/output_config.hpp"
#include <limits>
#include "init/sounding/parametric_target_math.hpp"
#include "compute/vulkan_startup_profile.hpp"

namespace tmv::configuration
{
struct CompiledPhysicsConfiguration
{
    RadiationConfig      radiation;
    BoundaryLayerConfig  boundary_layer;
    SurfaceConfig        surface;
    TurbulenceConfig     turbulence;
    DiffusionConfig      diffusion;
    AdvectionConfig      advection;
    TimeSteppingConfig   time_stepping;
    TerrainConfig        terrain;
    chaos::ChaosConfig   chaos;
    ComputeBackendConfig compute;
    MicrophysicsConfig   microphysics;
    std::string          microphysics_scheme;
    std::string          dynamics_scheme;

    void apply_to(SimulationStorageState& destination) &&
    {
        auto& runtime                                  = destination.runtime;
        runtime.radiation.config                       = std::move(radiation);
        runtime.boundary_layer.config                  = std::move(boundary_layer);
        runtime.boundary_layer.surface_config          = std::move(surface);
        runtime.turbulence.config                      = std::move(turbulence);
        runtime.diffusion.config                       = std::move(diffusion);
        runtime.advection.config                       = std::move(advection);
        runtime.time_stepping.config                   = std::move(time_stepping);
        runtime.terrain.config                         = std::move(terrain);
        runtime.chaos.config                           = std::move(chaos);
        runtime.compute.config                         = std::move(compute);
        runtime.compute.vertical_flux_template_id      = "tvd_vertical_flux_v2";
        destination.configuration.microphysics_config  = std::move(microphysics);
        destination.configuration.microphysics_scheme  = std::move(microphysics_scheme);
        destination.configuration.dynamics_scheme_name = std::move(dynamics_scheme);
    }
};

inline CompiledPhysicsConfiguration compile_physics(const init::config::ConfigV1& c)
{
    using namespace init::config;
    CompiledPhysicsConfiguration result;
    result.radiation.scheme_id                                = radiation_scheme_name(c.radiation.scheme);
    result.radiation.solar_forcing                            = radiation_solar_forcing_name(c.radiation.solar_forcing);
    result.radiation.latitude_deg                             = c.radiation.latitude_deg;
    result.radiation.declination_deg                          = c.radiation.declination_deg;
    result.radiation.local_solar_time_at_start_h              = c.radiation.local_solar_time_at_start_h;
    result.radiation.do_lw                                    = c.radiation.do_lw;
    result.radiation.do_sw                                    = c.radiation.do_sw;
    result.radiation.dt_radiation                             = c.radiation.cadence_s;
    result.radiation.clear_sky_only                           = c.radiation.clear_sky_only;
    result.radiation.simple_grey_optical_model                = c.radiation.simple_grey_optical_model;
    result.radiation.use_subcycling                           = c.radiation.use_subcycling;
    result.radiation.subcycle_count                           = c.radiation.subcycle_count;
    result.radiation.enable_diagnostics                       = c.radiation.enable_diagnostics;
    result.radiation.tau_lw_ref                               = c.radiation.tau_lw_ref;
    result.radiation.tau_sw_ref                               = c.radiation.tau_sw_ref;
    result.radiation.n_lw                                     = c.radiation.n_lw;
    result.radiation.n_sw                                     = c.radiation.n_sw;
    result.radiation.simple_grey_lw_vapor_absorption_m2kg     = c.radiation.simple_grey_lw_vapor_absorption_m2kg;
    result.radiation.simple_grey_lw_liquid_absorption_m2kg    = c.radiation.simple_grey_lw_liquid_absorption_m2kg;
    result.radiation.simple_grey_lw_ice_absorption_m2kg       = c.radiation.simple_grey_lw_ice_absorption_m2kg;
    result.radiation.simple_grey_sw_vapor_absorption_m2kg     = c.radiation.simple_grey_sw_vapor_absorption_m2kg;
    result.radiation.simple_grey_sw_liquid_absorption_m2kg    = c.radiation.simple_grey_sw_liquid_absorption_m2kg;
    result.radiation.simple_grey_sw_ice_absorption_m2kg       = c.radiation.simple_grey_sw_ice_absorption_m2kg;
    result.radiation.rrtmg_ozone_vmr                          = c.radiation.rrtmg_ozone_vmr;
    result.radiation.rrtmg_co2_vmr                            = c.radiation.rrtmg_co2_vmr;
    result.radiation.rrtmg_ch4_vmr                            = c.radiation.rrtmg_ch4_vmr;
    result.radiation.rrtmg_n2o_vmr                            = c.radiation.rrtmg_n2o_vmr;
    result.radiation.rrtmg_co_vmr                             = c.radiation.rrtmg_co_vmr;
    result.radiation.rrtmg_o2_vmr                             = c.radiation.rrtmg_o2_vmr;
    result.radiation.rrtmg_n2_vmr                             = c.radiation.rrtmg_n2_vmr;
    result.radiation.rrtmg_cfc11_vmr                          = c.radiation.rrtmg_cfc11_vmr;
    result.radiation.rrtmg_cfc12_vmr                          = c.radiation.rrtmg_cfc12_vmr;
    result.radiation.rrtmg_cfc22_vmr                          = c.radiation.rrtmg_cfc22_vmr;
    result.radiation.rrtmg_ccl4_vmr                           = c.radiation.rrtmg_ccl4_vmr;
    result.radiation.rrtmg_liquid_effective_radius_um         = c.radiation.rrtmg_liquid_effective_radius_um;
    result.radiation.rrtmg_ice_effective_radius_um            = c.radiation.rrtmg_ice_effective_radius_um;
    result.radiation.terrain_sw_cosine_nodes                  = c.radiation.terrain_sw_cosine_nodes;
    result.radiation.terrain_sw_azimuth_nodes                 = c.radiation.terrain_sw_azimuth_nodes;
    result.radiation.terrain_sw_maximum_bounces               = c.radiation.terrain_sw_maximum_bounces;
    result.radiation.terrain_sw_relative_residual_tolerance   = c.radiation.terrain_sw_relative_residual_tolerance;
    result.radiation.terrain_sw_absolute_residual_tolerance_w = c.radiation.terrain_sw_absolute_residual_tolerance_w;
    result.microphysics                                       = c.microphysics.parameters;
    result.microphysics_scheme                                = microphysics_scheme_name(c.microphysics.scheme);
    result.dynamics_scheme = c.dynamics.scheme == "klemp_wilhelmson_split"
                                 ? (c.grid.staggering == StaggerType::CGrid ? "klemp_wilhelmson_split_cartesian_cgrid"
                                                                            : "klemp_wilhelmson_split_cartesian_agrid")
                                 : c.dynamics.scheme;
    result.boundary_layer.scheme_id            = boundary_layer_scheme_name(c.boundary_layer.scheme);
    result.boundary_layer.pbl_top_method       = boundary_layer_top_method_name(c.boundary_layer.pbl_top_method);
    result.boundary_layer.dt_pbl               = c.boundary_layer.cadence_s;
    result.boundary_layer.clear_sky_only       = c.boundary_layer.clear_sky_only;
    result.boundary_layer.enable_tke           = c.boundary_layer.scheme == BoundaryLayerScheme::MYNN;
    result.boundary_layer.apply_surface_fluxes = c.surface.apply_fluxes;
    result.surface.z0m                         = c.surface.z0m_m;
    result.surface.z0h                         = c.surface.z0h_m;
    result.surface.albedo                      = c.surface.albedo;
    result.surface.emissivity                  = c.surface.emissivity;
    result.surface.Tsfc                        = c.surface.tsk_k;
    result.surface.qsfc                        = c.surface.qsfc_kgkg;
    result.turbulence.scheme_id                = turbulence_scheme_name(c.turbulence.scheme);
    result.turbulence.mode                     = c.turbulence.mode == TurbulenceMode::TwoD           ? "horizontal_only"
                                                 : c.turbulence.mode == TurbulenceMode::VerticalOnly ? "vertical_only"
                                                                                                     : "3d";
    result.turbulence.filter_width             = turbulence_filter_width_name(c.turbulence.filter_width);
    result.turbulence.user_filter_width_m      = c.turbulence.user_filter_width_m;
    result.turbulence.dynamic_Cs               = c.turbulence.dynamic_coefficient;
    result.turbulence.stability_correction = c.turbulence.stability_correction == TurbulenceStabilityCorrection::Richardson
        ? kLillyRichardsonModel : turbulence_stability_correction_name(c.turbulence.stability_correction);
    result.turbulence.Cs                   = c.turbulence.Cs;
    result.turbulence.Pr_t                 = c.turbulence.Pr_t;
    result.turbulence.Sc_t                 = c.turbulence.Sc_t;
    result.diffusion.scheme_id             = diffusion_scheme_name(c.diffusion.scheme);
    result.diffusion.operator_order        = c.diffusion.operator_order;
    result.diffusion.K_h                   = c.diffusion.K_h;
    result.diffusion.K_v                   = c.diffusion.K_v;
    result.diffusion.apply_to              = diffusion_target_name(c.diffusion.apply_to);
    result.diffusion.coefficient_source    = diffusion_coefficient_source_name(c.diffusion.coefficient_source);
    result.diffusion.use_variable_K        = c.diffusion.coefficient_source == DiffusionCoefficientSource::LinearHeight;
    if (result.diffusion.use_variable_K)
    {
        result.diffusion.K_v_surface = c.diffusion.K_v_surface;
        result.diffusion.K_v_top     = c.diffusion.K_v_top;
    }
    result.advection.scheme_id               = resolved_advection_scheme_name(c.advection.scheme, c.grid.staggering);
    result.time_stepping.scheme_id           = time_integrator_name(c.time.integrator);
    result.time_stepping.split_acoustic      = c.time.integrator == TimeIntegrator::SplitExplicit;
    result.time_stepping.n_acoustic_substeps = c.time.acoustic_substeps;
    result.time_stepping.adaptive_dt         = c.time.adaptive_dt;
    result.time_stepping.cfl_safety          = c.time.cfl_safety;
    result.terrain.scheme_id                 = terrain_scheme_name(c.terrain.scheme);
    result.terrain.coord_id                  = terrain_coordinate_name(c.terrain.coordinate);
    result.terrain.horizontal_placement_id   = terrain_horizontal_placement_name(c.terrain.horizontal_placement);
    result.terrain.ztop                      = c.grid.ztop;
    result.terrain.compute_derivatives       = c.terrain.compute_derivatives;
    result.terrain.compute_metrics           = c.terrain.compute_metrics;
    result.terrain.smoothing_decay           = c.terrain.smoothing_decay;
    result.terrain.bell.h0                   = c.terrain.bell.h0;
    result.terrain.bell.a                    = c.terrain.bell.a;
    result.terrain.bell.axisymmetric         = c.terrain.bell.axisymmetric;
    result.terrain.schar.H                   = c.terrain.schar.H;
    result.terrain.schar.a                   = c.terrain.schar.a;
    result.terrain.schar.ell                 = c.terrain.schar.ell;
    if (c.terrain.horizontal_placement == TerrainHorizontalPlacement::EarthFixed)
    {
        result.terrain.feature_center_earth_x_m = c.terrain.feature_center_earth_x_m;
        result.terrain.feature_center_earth_y_m = c.terrain.feature_center_earth_y_m;
    }
    result.chaos.scheme_id                  = chaos_scheme_name(c.extensions.chaos.scheme);
    result.chaos.seed                       = c.extensions.chaos.seed;
    result.chaos.member_id                  = c.extensions.chaos.member_id;
    result.chaos.native_generation_required = c.extensions.chaos.native_generation_required;
    result.chaos.apply_to_ic                = c.extensions.chaos.apply_to_ic;
    result.chaos.apply_to_tendencies        = c.extensions.chaos.apply_to_tendencies;
    result.chaos.sigma_ic                   = c.extensions.chaos.sigma_ic;
    result.chaos.alpha_tend                 = c.extensions.chaos.alpha_tend;
    result.compute.backend = c.extensions.compute.backend == ComputeBackend::GPU
        ? "vulkan" : compute_backend_name(c.extensions.compute.backend);
    result.compute.allow_fallback           = c.extensions.compute.allow_fallback;
    result.compute.validate_parity          = c.extensions.compute.validate_parity;
    result.compute.cuda_assist              = c.extensions.compute.cuda_assist;
    if (c.extensions.execution.mode == ExecutionMode::Qualified)
        result.compute.allow_fallback = false;
    result.compute.startup_plan = nimbus::compute::vulkan_startup_plan_for_profile(c);
    return result;
}
inline void compile_initialization(const init::config::ConfigV1& c, RuntimeConfigurationState& result)
{
    using namespace init::config;
    const auto& sounding    = c.initial_conditions.sounding;
    result.sounding_source  = {};
    result.sounding_enabled = false;
    if (sounding.type == SoundingType::ParametricCAPE)
    {
        result.sounding_source.type = init::SoundingSourceConfig::Type::ParametricCAPE;
        result.cape_target          = sounding.cape_jkg;
        result.sfc_theta_k          = sounding.surface_theta_k;
        result.sfc_qv_kgkg          = sounding.moisture_qv_kgkg.value_or(0.014);
        result.tropopause_z_m       = sounding.tropopause_m;
    }
    else if (sounding.type == SoundingType::File)
    {
        result.sounding_source.type           = init::SoundingSourceConfig::Type::File;
        result.sounding_config.scheme_id      = "auto";
        result.sounding_config.file_path      = sounding.file_path;
        result.sounding_source.file.path      = sounding.file_path;
        result.sounding_source.file.scheme_id = "auto";
    }
    else
    {
        result.sounding_source.type = init::SoundingSourceConfig::Type::ParametricTargets;
    }
    result.sounding_source.parametric = {
        result.cape_target, result.sfc_theta_k, result.sfc_qv_kgkg, result.tropopause_z_m};
    auto& targets           = result.sounding_source.targets;
    targets.surface_theta_k = result.sfc_theta_k;
    targets.target_el_m     = result.tropopause_z_m;
    targets.target_cape_jkg = result.cape_target;
    if (sounding.type == SoundingType::ParametricTargets)
    {
        targets.surface_theta_k = sounding.surface_theta_k;
        targets.target_cape_jkg = sounding.cape_jkg;
        targets.target_cin_jkg  = sounding.cin_jkg;
        targets.target_lfc_m    = sounding.lfc_m;
        targets.target_el_m     = sounding.el_m;
        if (sounding.moisture_qv_kgkg)
            targets.moisture_qv_kgkg_override = *sounding.moisture_qv_kgkg;
        else
            targets.target_lcl_m = sounding.lcl_m;
    }
    if (c.extensions.execution.mode == ExecutionMode::Qualified)
        result.sounding_config.use_fallback_profiles = false;

    const auto& wind    = c.initial_conditions.hodograph;
    result.wind_profile = {wind.u_sfc_ms, wind.v_sfc_ms, wind.u_1km_ms, wind.v_1km_ms, wind.u_6km_ms, wind.v_6km_ms};
    result.hodograph_source            = {};
    result.hodograph_source.type       = wind.type == HodographType::Auto ? init::HodographSourceConfig::Type::Auto
                                                                          : init::HodographSourceConfig::Type::WKParam;
    result.hodograph_source.wk_anchors = {
        wind.u_sfc_ms, wind.v_sfc_ms, wind.u_1km_ms, wind.v_1km_ms, wind.u_6km_ms, wind.v_6km_ms};

    const auto& trigger        = c.initial_conditions.trigger;
    result.trigger_source      = {};
    result.trigger_source.type = trigger.type == TriggerType::None     ? init::TriggerSourceConfig::Type::None
                                 : trigger.type == TriggerType::Bubble ? init::TriggerSourceConfig::Type::WarmBubble
                                                                       : init::TriggerSourceConfig::Type::VortexSeed;
    result.bubble_center_x_m   = trigger.bubble.center_x_m;
    result.bubble_center_y_m   = trigger.bubble.center_y_m;
    result.bubble_center_z_m   = trigger.bubble.center_z_m;
    result.bubble_radius_m     = trigger.bubble.radius_km * 1000.0;
    result.bubble_dtheta_k     = trigger.bubble.dtheta_k;
    result.bubble_cloud_water_kgkg               = trigger.bubble.cloud_water_kgkg;
    result.trigger_source.vortex_seed.r_max_m    = trigger.vortex_seed.r_max_m;
    result.trigger_source.vortex_seed.v_max_ms   = trigger.vortex_seed.v_max_ms;
    result.trigger_source.vortex_seed.z_top_m    = trigger.vortex_seed.z_top_m;
    result.trigger_source.vortex_seed.center_x_m = trigger.vortex_seed.center_x_m;
    result.trigger_source.vortex_seed.center_y_m = trigger.vortex_seed.center_y_m;
    result.trigger_source.vortex_seed.center_z_m = trigger.vortex_seed.center_z_m;
    result.domain_motion_mode                    = c.domain_motion.mode;
    result.domain_frame                          = domain::HorizontalDomainFrame::stationary();
    result.domain_frame.velocity_ms = {c.domain_motion.frame_velocity.u_ms, c.domain_motion.frame_velocity.v_ms};
    result.domain_frame.validate();
    result.storm_tracking = c.domain_motion.tracking;
    result.nesting        = c.domain_motion.nesting;
}

struct CompiledRunControls
{
    int                      duration_s;
    int                      write_every_s;
    OutputConfig             output;
    bool                     live_shm;
    std::vector<std::string> live_shm_fields;
};

inline CompiledRunControls compile_controls(const init::config::ConfigV1& c, RuntimeConfigurationState& runtime)
{
    using namespace init::config;
    const auto seconds = [](double value, const char* name)
    {
        if (!std::isfinite(value) || value < 0.0 || value > std::numeric_limits<int>::max() ||
            std::floor(value) != value)
            throw std::invalid_argument(std::string(name) + " must be a non-negative whole number of seconds");
        return static_cast<int>(value);
    };
    CompiledRunControls result{seconds(c.time.run_time, "duration_s"),
                               seconds(c.time.output_interval, "output.write_every_s"),
                               {},
                               c.output.live_shm,
                               c.output.live_shm_fields};
    runtime.execution_mode                    = c.extensions.execution.mode;
    runtime.checkpoint_every_s                = c.extensions.checkpoint.every_s;
    runtime.checkpoint_keep_last              = c.extensions.checkpoint.keep_last;
    runtime.checkpoint_minimum_free_space_mib = c.extensions.checkpoint.minimum_free_space_mib;
    runtime.checkpoint_directory              = c.extensions.checkpoint.directory;
    runtime.checkpoint_resume_from            = c.extensions.checkpoint.resume_from;
    runtime.coordinate_system                 = c.grid.coordinate_system;
    runtime.stagger_type                      = c.grid.staggering;
    if (c.extensions.performance.timing)
        runtime.perf_timing_enabled = *c.extensions.performance.timing;
    if (c.extensions.performance.report_every_steps)
        runtime.perf_report_every_steps = *c.extensions.performance.report_every_steps;
    if (c.extensions.performance.warmup_steps)
        runtime.perf_warmup_steps = *c.extensions.performance.warmup_steps;
    runtime.scientific_diagnostics_every_steps = c.extensions.diagnostics.scientific_every_steps;
    runtime.validation_policy.mode = c.extensions.validation.guard_mode == ValidationGuardMode::Off ? GuardMode::Off
                                     : c.extensions.validation.guard_mode == ValidationGuardMode::Warn
                                         ? GuardMode::Warn
                                         : GuardMode::Strict;
    if (runtime.execution_mode == ExecutionMode::Qualified)
    {
        runtime.validation_policy.mode    = GuardMode::Strict;
        runtime.validation_policy.fail_on = GuardFailOn::Both;
    }
    auto& out = result.output;
    switch (c.output.preset)
    {
    case OutputPreset::Minimal:
        out.preset = FieldPreset::minimal;
        break;
    case OutputPreset::RadarReady:
        out.preset = FieldPreset::radar_ready;
        break;
    case OutputPreset::RadarZdrExploratory:
        out.preset = FieldPreset::radar_zdr_exploratory;
        break;
    case OutputPreset::RadarPolarimetricReady:
        out.preset = FieldPreset::radar_polarimetric_ready;
        break;
    case OutputPreset::Presentation:
        out.preset = FieldPreset::presentation;
        break;
    case OutputPreset::StudioFull:
        out.preset = FieldPreset::studio_full;
        break;
    case OutputPreset::DynamicsFull:
        out.preset = FieldPreset::all;
        break;
    case OutputPreset::ScienceValidation:
        out.preset = FieldPreset::science_validation;
        break;
    }
    switch (c.output.format)
    {
    case init::config::OutputFormat::Npy3D:
        out.format = ::OutputFormat::npy_3d;
        break;
    case init::config::OutputFormat::Netcdf:
        out.format = ::OutputFormat::netcdf;
        break;
    case init::config::OutputFormat::ZfpLow:
        out.format        = ::OutputFormat::zfp;
        out.zfp_requested_format_tier = "zfp_low";
        out.zfp_tolerance = 1e-2;
        break;
    case init::config::OutputFormat::ZfpMedium:
        out.format        = ::OutputFormat::zfp;
        out.zfp_requested_format_tier = "zfp_medium";
        out.zfp_tolerance = 1e-3;
        break;
    case init::config::OutputFormat::ZfpHigh:
        out.format        = ::OutputFormat::zfp;
        out.zfp_requested_format_tier = "zfp_high";
        out.zfp_tolerance = 1e-4;
        break;
    }
    out.async_io                          = c.output.async_io;
    out.live_shm_interval_s               = c.output.live_shm_interval_s;
    out.live_shm_full_rate                = c.output.live_shm_full_rate;
    out.trajectories                      = c.output.trajectories;
    out.trajectory_seed_time_s            = c.output.trajectory_seed_time_s;
    out.trajectory_count                  = c.output.trajectory_count;
    out.trajectory_sample_interval_s      = c.output.trajectory_sample_interval_s;
    out.trajectory_max_samples_per_parcel = c.output.trajectory_max_samples_per_parcel;
    out.keep_last_frames                  = c.output.keep_last_frames;
    out.maximum_retained_history_files    = c.output.maximum_retained_history_files;
    out.minimum_free_space_mib            = c.output.minimum_free_space_mib;
    if (c.output.qualification_diagnostics_preset != init::config::QualificationDiagnosticsPreset::None)
    {
        out.qualification_diagnostics_preset     = ::QualificationDiagnosticsPreset::moving_reference_v1;
        out.qualification_diagnostics_interval_s = c.output.qualification_diagnostics_interval_s;
    }
    if (out.format == ::OutputFormat::netcdf)
    {
        out.netcdf_deflate_level    = c.output.netcdf_deflate_level;
        out.netcdf_shuffle          = c.output.netcdf_shuffle;
        out.netcdf_target_chunk_kib = c.output.netcdf_target_chunk_kib;
    }
    if (out.format == ::OutputFormat::zfp)
    {
        if (c.output.zfp_tolerance > 0.0)
            out.zfp_tolerance = c.output.zfp_tolerance;
        out.zfp_mode                 = c.output.zfp_mode;
        out.zfp_precision_bits       = c.output.zfp_precision_bits;
        out.zfp_rate_bps             = c.output.zfp_rate_bps;
        out.zfp_keyframe_interval    = c.output.zfp_keyframe_interval;
        out.zfp_sparse_threshold     = c.output.zfp_sparse_threshold;
        out.zfp_float16_prefilter    = c.output.zfp_float16_prefilter;
        out.zfp_predictive_delta     = c.output.zfp_predictive_delta;
        out.zfp_field_execution      = c.output.zfp_field_execution;
        out.zfp_field_workers        = c.output.zfp_field_workers;
        out.zfp_field_scratch_mib    = c.output.zfp_field_scratch_mib;
        out.zfp_per_field_tolerances = c.output.zfp_per_field_tolerances;
        out.zfp_tight_tolerance      = c.output.zfp_tight_tolerance;
        out.zfp_moderate_tolerance   = c.output.zfp_moderate_tolerance;
        out.zfp_loose_tolerance      = c.output.zfp_loose_tolerance;
        out.zfp_field_tolerances     = c.output.zfp_field_tolerances;
    }
    if (c.output.tiered_write_cadence)
    {
        out.tiered_write_cadence   = true;
        out.write_cadence_medium_s = c.output.write_cadence_medium_s;
        out.write_cadence_slow_s   = c.output.write_cadence_slow_s;
    }
    return result;
}

} // namespace tmv::configuration
