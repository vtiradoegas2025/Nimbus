#pragma once

#include "core/infra/sha256.hpp"
#include "init/config/config_v1.hpp"
#include "init/config/v0_to_v1.hpp"

#include <exception>
#include <string_view>

namespace tmv::runtime
{

inline constexpr std::string_view kFlatMixedMetalRetainedQualifiedYamlSha256 =
    "985c698858be9be4be29dc2adf819f31c20788c13a414747c422e21d7ebe51da";

inline bool matches_flat_mixed_metal_physics(
    const tmv::init::config::ConfigV1& config)
{
    namespace cfg = tmv::init::config;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& motion = config.domain_motion;
    const auto& compute = config.extensions.compute;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    const auto& chaos = config.extensions.chaos;
    return config.schema_version == "nimbus.v1" &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 18 && config.grid.ny == 18 && config.grid.nz == 18 &&
        config.grid.dx == 500.0 && config.grid.dy == 500.0 &&
        config.grid.dz == 250.0 && config.grid.ztop == 4500.0 &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.time.integrator == cfg::TimeIntegrator::SplitExplicit &&
        config.time.dt_large == 0.1 && config.time.acoustic_substeps == 6 &&
        !config.time.adaptive_dt && config.time.cfl_safety == 0.7 &&
        config.advection.scheme == cfg::AdvectionScheme::TVD &&
        lateral.west == cfg::LateralBC::RigidSlip && lateral.east == cfg::LateralBC::RigidSlip &&
        lateral.south == cfg::LateralBC::RigidSlip && lateral.north == cfg::LateralBC::RigidSlip &&
        config.boundary_conditions.bottom == cfg::BottomBC::RigidNoSlip &&
        config.boundary_conditions.top == cfg::TopBC::Open &&
        motion.mode == cfg::DomainMotionMode::Stationary &&
        motion.frame_velocity.u_ms == 0.0 && motion.frame_velocity.v_ms == 0.0 &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && !motion.nesting.enabled &&
        motion.nesting.feedback == cfg::NestingFeedback::OneWay &&
        config.terrain.scheme == cfg::TerrainScheme::None &&
        !config.damping.enabled && config.damping.lateral_sponge_enabled &&
        config.damping.lateral_width == 20 && config.damping.lateral_alpha == 0.05 &&
        config.damping.w_only_top_sponge && config.damping.w_only_depth_cells == 8 &&
        config.damping.w_only_alpha_max == 0.20 && !config.coriolis.enabled &&
        config.microphysics.scheme == cfg::MicrophysicsScheme::None &&
        config.turbulence.scheme == cfg::TurbulenceScheme::None &&
        config.diffusion.scheme == cfg::DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        config.diffusion.K_h == 0.0 && config.diffusion.K_v == 0.0 &&
        config.diffusion.apply_to == cfg::DiffusionTarget::Scalars &&
        config.diffusion.coefficient_source == cfg::DiffusionCoefficientSource::Constant &&
        config.surface.scheme == cfg::SurfaceScheme::None && !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == cfg::BoundaryLayerScheme::None &&
        config.radiation.scheme == cfg::RadiationScheme::None &&
        !config.radar.enabled && config.radar.products.empty() &&
        !compute.gpu_apply_microphysics && !compute.gpu_apply_slow && !compute.gpu_apply_sgs &&
        config.extensions.solver.slow_rk2 &&
        config.extensions.solver.lid_bc_gamma == -1.0 &&
        !config.extensions.solver.axis_u_m1_filter &&
        sounding.type == cfg::SoundingType::ParametricCAPE &&
        sounding.cape_jkg == 0.0 && sounding.tropopause_m == 8000.0 &&
        sounding.surface_theta_k == 300.0 &&
        sounding.moisture_qv_kgkg.has_value() && *sounding.moisture_qv_kgkg == 0.004 &&
        hodograph.type == cfg::HodographType::WKParam &&
        hodograph.u_sfc_ms == 0.0 && hodograph.v_sfc_ms == 0.0 &&
        hodograph.u_1km_ms == 0.0 && hodograph.v_1km_ms == 0.0 &&
        hodograph.u_6km_ms == 0.0 && hodograph.v_6km_ms == 0.0 &&
        config.initial_conditions.trigger.type == cfg::TriggerType::Bubble &&
        bubble.dtheta_k == 2.0 && bubble.cloud_water_kgkg == 0.0 &&
        bubble.radius_km == 1.5 && bubble.center_x_m == 4500.0 &&
        bubble.center_y_m == 4500.0 && bubble.center_z_m == 1500.0 &&
        chaos.scheme == cfg::ChaosScheme::InitialConditions &&
        chaos.seed == 0 && chaos.member_id == 0 && !chaos.native_generation_required &&
        chaos.apply_to_ic.empty() && chaos.apply_to_tendencies.empty();
}

inline bool matches_flat_mixed_metal_cadence1_output(
    const tmv::init::config::ConfigV1& config)
{
    namespace cfg = tmv::init::config;
    const auto& output = config.output;
    return config.time.output_interval == 1.0 &&
        output.preset == cfg::OutputPreset::Minimal && output.format == cfg::OutputFormat::Npy3D &&
        !output.async_io && !output.live_shm && !output.trajectories &&
        output.qualification_diagnostics_preset == cfg::QualificationDiagnosticsPreset::None &&
        !output.tiered_write_cadence && output.keep_last_frames == 0 &&
        output.minimum_free_space_mib == 0 && output.maximum_retained_history_files == 0;
}

inline bool matches_flat_mixed_metal_unserialized_defaults(
    const tmv::init::config::ConfigV1& config)
{
    namespace cfg = tmv::init::config;
    const cfg::OutputSection output;
    const cfg::RadiationSection radiation;
    const auto& selected = config.output;
    const auto& selected_radiation = config.radiation;
    return config.diffusion.K_v_surface == 0.0 && config.diffusion.K_v_top == 0.0 &&
        config.surface.tsk_k == -1.0 && config.surface.qsfc_kgkg == -1.0 &&
        config.terrain.feature_center_earth_x_m == 0.0 &&
        config.terrain.feature_center_earth_y_m == 0.0 &&
        config.initial_conditions.sounding.file_path.empty() &&
        !config.extensions.execution.acknowledge_maturity_limits &&
        config.extensions.checkpoint.keep_last == 0 &&
        config.extensions.checkpoint.minimum_free_space_mib == 0 &&
        selected.qualification_diagnostics_interval_s == output.qualification_diagnostics_interval_s &&
        selected.write_cadence_medium_s == output.write_cadence_medium_s &&
        selected.write_cadence_slow_s == output.write_cadence_slow_s &&
        selected.netcdf_deflate_level == output.netcdf_deflate_level &&
        selected.netcdf_shuffle == output.netcdf_shuffle &&
        selected.netcdf_target_chunk_kib == output.netcdf_target_chunk_kib &&
        selected.zfp_tolerance == output.zfp_tolerance && selected.zfp_mode == output.zfp_mode &&
        selected.zfp_precision_bits == output.zfp_precision_bits &&
        selected.zfp_rate_bps == output.zfp_rate_bps &&
        selected.zfp_keyframe_interval == output.zfp_keyframe_interval &&
        selected.zfp_sparse_threshold == output.zfp_sparse_threshold &&
        selected.zfp_float16_prefilter == output.zfp_float16_prefilter &&
        selected.zfp_predictive_delta == output.zfp_predictive_delta &&
        selected.zfp_field_execution == output.zfp_field_execution &&
        selected.zfp_field_workers == output.zfp_field_workers &&
        selected.zfp_field_scratch_mib == output.zfp_field_scratch_mib &&
        selected.zfp_per_field_tolerances == output.zfp_per_field_tolerances &&
        selected.zfp_tight_tolerance == output.zfp_tight_tolerance &&
        selected.zfp_moderate_tolerance == output.zfp_moderate_tolerance &&
        selected.zfp_loose_tolerance == output.zfp_loose_tolerance &&
        selected.zfp_field_tolerances.empty() &&
        selected_radiation.simple_grey_optical_model == radiation.simple_grey_optical_model &&
        selected_radiation.use_subcycling == radiation.use_subcycling &&
        selected_radiation.subcycle_count == radiation.subcycle_count &&
        selected_radiation.enable_diagnostics == radiation.enable_diagnostics &&
        selected_radiation.simple_grey_lw_vapor_absorption_m2kg == radiation.simple_grey_lw_vapor_absorption_m2kg &&
        selected_radiation.simple_grey_lw_liquid_absorption_m2kg == radiation.simple_grey_lw_liquid_absorption_m2kg &&
        selected_radiation.simple_grey_lw_ice_absorption_m2kg == radiation.simple_grey_lw_ice_absorption_m2kg &&
        selected_radiation.simple_grey_sw_vapor_absorption_m2kg == radiation.simple_grey_sw_vapor_absorption_m2kg &&
        selected_radiation.simple_grey_sw_liquid_absorption_m2kg == radiation.simple_grey_sw_liquid_absorption_m2kg &&
        selected_radiation.simple_grey_sw_ice_absorption_m2kg == radiation.simple_grey_sw_ice_absorption_m2kg &&
        selected_radiation.rrtmg_ozone_vmr == radiation.rrtmg_ozone_vmr &&
        selected_radiation.rrtmg_co2_vmr == radiation.rrtmg_co2_vmr &&
        selected_radiation.rrtmg_ch4_vmr == radiation.rrtmg_ch4_vmr &&
        selected_radiation.rrtmg_n2o_vmr == radiation.rrtmg_n2o_vmr &&
        selected_radiation.rrtmg_co_vmr == radiation.rrtmg_co_vmr &&
        selected_radiation.rrtmg_o2_vmr == radiation.rrtmg_o2_vmr &&
        selected_radiation.rrtmg_n2_vmr == radiation.rrtmg_n2_vmr &&
        selected_radiation.rrtmg_cfc11_vmr == radiation.rrtmg_cfc11_vmr &&
        selected_radiation.rrtmg_cfc12_vmr == radiation.rrtmg_cfc12_vmr &&
        selected_radiation.rrtmg_cfc22_vmr == radiation.rrtmg_cfc22_vmr &&
        selected_radiation.rrtmg_ccl4_vmr == radiation.rrtmg_ccl4_vmr &&
        selected_radiation.rrtmg_liquid_effective_radius_um == radiation.rrtmg_liquid_effective_radius_um &&
        selected_radiation.rrtmg_ice_effective_radius_um == radiation.rrtmg_ice_effective_radius_um;
}

// The normalized resource policy is backed by the sealed warn/reject science
// identity. Every other stored inactive or operational value stays hash-pinned.
inline bool is_proposed_exact_flat_mixed_metal_cadence1_profile(
    const tmv::init::config::ConfigV1& config)
{
    namespace cfg = tmv::init::config;
    const auto& compute = config.extensions.compute;
    if (config.extensions.execution.mode != cfg::ExecutionMode::Qualified ||
        compute.backend != cfg::ComputeBackend::Metal || compute.allow_fallback ||
        compute.validate_parity ||
        (compute.memory_admission != cfg::MemoryAdmissionPolicy::Warn &&
         compute.memory_admission != cfg::MemoryAdmissionPolicy::Reject) ||
        config.extensions.validation.guard_mode != cfg::ValidationGuardMode::Error ||
        config.extensions.diagnostics.scientific_every_steps != 1 ||
        !config.extensions.diagnostics.deterministic_reductions ||
        !matches_flat_mixed_metal_physics(config) ||
        !matches_flat_mixed_metal_cadence1_output(config) ||
        !matches_flat_mixed_metal_unserialized_defaults(config))
    {
        return false;
    }
    try
    {
        auto canonical = config;
        canonical.extensions.compute.memory_admission = cfg::MemoryAdmissionPolicy::Warn;
        return nimbus::crypto::sha256_hex(cfg::serialize_v1_to_yaml(canonical)) ==
            kFlatMixedMetalRetainedQualifiedYamlSha256;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

inline bool is_public_qualified_flat_mixed_metal_cadence1_profile(
    const tmv::init::config::ConfigV1& config)
{
    return is_proposed_exact_flat_mixed_metal_cadence1_profile(config);
}

}
