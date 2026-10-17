#include "qualified_terrain_profile.hpp"

#include "chaos/config_contract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::init::config
{
namespace
{

bool same_profile_number(double value, double expected)
{
    return std::isfinite(value) &&
        std::abs(value - expected) <=
            8.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(expected));
}

bool exact_stationary_bell(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    const auto& hodograph = config.initial_conditions.hodograph;
    return motion.mode == DomainMotionMode::Stationary &&
        same_profile_number(motion.frame_velocity.u_ms, 0.0) &&
        same_profile_number(motion.frame_velocity.v_ms, 0.0) &&
        config.terrain.scheme == TerrainScheme::Bell &&
        config.terrain.horizontal_placement ==
            TerrainHorizontalPlacement::GridAttached &&
        same_profile_number(config.terrain.bell.h0, 300.0) &&
        same_profile_number(config.terrain.bell.a, 1800.0) &&
        config.terrain.bell.axisymmetric &&
        same_profile_number(hodograph.u_sfc_ms, 0.0) &&
        same_profile_number(hodograph.v_sfc_ms, 0.0) &&
        same_profile_number(hodograph.u_1km_ms, 2.0) &&
        same_profile_number(hodograph.v_1km_ms, 2.0) &&
        same_profile_number(hodograph.u_6km_ms, 7.0) &&
        same_profile_number(hodograph.v_6km_ms, 6.0);
}

bool exact_translating_schar(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    const auto& hodograph = config.initial_conditions.hodograph;
    return motion.mode == DomainMotionMode::ConstantVelocity &&
        same_profile_number(motion.frame_velocity.u_ms, 5.0) &&
        same_profile_number(motion.frame_velocity.v_ms, -2.0) &&
        config.terrain.scheme == TerrainScheme::Schar &&
        config.terrain.horizontal_placement ==
            TerrainHorizontalPlacement::GridAttached &&
        same_profile_number(config.terrain.schar.H, 300.0) &&
        same_profile_number(config.terrain.schar.a, 1800.0) &&
        same_profile_number(config.terrain.schar.ell, 900.0) &&
        same_profile_number(hodograph.u_sfc_ms, 5.0) &&
        same_profile_number(hodograph.v_sfc_ms, -2.0) &&
        same_profile_number(hodograph.u_1km_ms, 7.0) &&
        same_profile_number(hodograph.v_1km_ms, 0.0) &&
        same_profile_number(hodograph.u_6km_ms, 12.0) &&
        same_profile_number(hodograph.v_6km_ms, 4.0);
}

bool exact_collocated_bell(const ConfigV1& config)
{
    const auto& compute = config.extensions.compute;
    const auto& motion = config.domain_motion;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    const bool stationary_motion =
        motion.mode == DomainMotionMode::Stationary &&
        same_profile_number(motion.frame_velocity.u_ms, 0.0) &&
        same_profile_number(motion.frame_velocity.v_ms, 0.0);
    const bool translating_motion =
        motion.mode == DomainMotionMode::ConstantVelocity &&
        same_profile_number(motion.frame_velocity.u_ms, 7.0) &&
        same_profile_number(motion.frame_velocity.v_ms, -3.0);
    const bool supported_hodograph = hodograph.type == HodographType::WKParam &&
        same_profile_number(
            hodograph.u_sfc_ms, translating_motion ? 7.0 : 0.0) &&
        same_profile_number(
            hodograph.v_sfc_ms, translating_motion ? -3.0 : 0.0) &&
        same_profile_number(
            hodograph.u_1km_ms, translating_motion ? 7.0 : 0.0) &&
        same_profile_number(
            hodograph.v_1km_ms, translating_motion ? -3.0 : 0.0) &&
        same_profile_number(
            hodograph.u_6km_ms, translating_motion ? 7.0 : 0.0) &&
        same_profile_number(
            hodograph.v_6km_ms, translating_motion ? -3.0 : 0.0);
    return config.extensions.execution.mode == ExecutionMode::Qualified
        && compute.backend == ComputeBackend::CPU
        && !compute.allow_fallback && !compute.validate_parity
        && !compute.gpu_apply_microphysics && !compute.gpu_apply_slow
        && !compute.gpu_apply_sgs
        && config.grid.coordinate_system == CoordinateSystem::Cartesian
        && config.grid.staggering == StaggerType::Collocated
        && config.grid.nx == 18 && config.grid.ny == 18
        && config.grid.nz == 18
        && same_profile_number(config.grid.dx, 500.0)
        && same_profile_number(config.grid.dy, 500.0)
        && same_profile_number(config.grid.dz, 250.0)
        && same_profile_number(config.grid.ztop, 4500.0)
        && config.time.integrator == TimeIntegrator::SplitExplicit
        && same_profile_number(config.time.run_time, 2.0)
        && same_profile_number(config.time.dt_large, 0.1)
        && config.time.acoustic_substeps == 6
        && !config.time.adaptive_dt
        && same_profile_number(config.time.cfl_safety, 0.7)
        && same_profile_number(config.time.output_interval, 1.0)
        && (stationary_motion || translating_motion)
        && !motion.lifetime_planning.enabled && !motion.tracking.enabled
        && !motion.tracking.recycle_domain && !motion.nesting.enabled
        && config.dynamics.scheme
            == "klemp_wilhelmson_split_cartesian_agrid"
        && !config.dynamics.axisymmetric
        && config.advection.scheme == AdvectionScheme::TVD
        && lateral.west == LateralBC::Periodic
        && lateral.east == LateralBC::Periodic
        && lateral.south == LateralBC::Periodic
        && lateral.north == LateralBC::Periodic
        && config.boundary_conditions.bottom == BottomBC::RigidSlip
        && config.boundary_conditions.top == TopBC::Rigid
        && !config.damping.enabled
        && same_profile_number(config.damping.z_d, 4500.0)
        && same_profile_number(config.damping.alpha, 0.0)
        && config.damping.mode == DampingMode::Explicit
        && config.damping.lateral_width == 0
        && same_profile_number(config.damping.lateral_alpha, 0.0)
        && !config.damping.lateral_sponge_enabled
        && !config.damping.w_only_top_sponge
        && config.damping.w_only_depth_cells == 0
        && same_profile_number(config.damping.w_only_alpha_max, 0.0)
        && !config.coriolis.enabled
        && config.microphysics.scheme == MicrophysicsScheme::None
        && config.turbulence.scheme == TurbulenceScheme::None
        && config.turbulence.mode == TurbulenceMode::ThreeD
        && same_profile_number(config.turbulence.Cs, 0.18)
        && same_profile_number(config.turbulence.Pr_t, 0.7)
        && same_profile_number(config.turbulence.Sc_t, 0.7)
        && config.diffusion.scheme == DiffusionScheme::Explicit
        && config.diffusion.operator_order == 2
        && same_profile_number(config.diffusion.K_h, 0.0)
        && same_profile_number(config.diffusion.K_v, 0.0)
        && config.diffusion.apply_to == DiffusionTarget::Scalars
        && config.surface.scheme == SurfaceScheme::None
        && !config.surface.apply_fluxes
        && config.boundary_layer.scheme == BoundaryLayerScheme::None
        && config.boundary_layer.pbl_top_method
            == BoundaryLayerTopMethod::Richardson
        && config.radiation.scheme == RadiationScheme::None
        && config.terrain.scheme == TerrainScheme::Bell
        && config.terrain.horizontal_placement
            == TerrainHorizontalPlacement::GridAttached
        && config.terrain.coordinate == TerrainCoordinate::Smoothed
        && config.terrain.compute_derivatives
        && config.terrain.compute_metrics
        && same_profile_number(config.terrain.smoothing_decay, 0.1)
        && same_profile_number(config.terrain.bell.h0, 220.0)
        && same_profile_number(config.terrain.bell.a, 1600.0)
        && config.terrain.bell.axisymmetric
        && sounding.type == SoundingType::ParametricCAPE
        && same_profile_number(sounding.cape_jkg, 1411.79)
        && same_profile_number(sounding.tropopause_m, 8000.0)
        && same_profile_number(sounding.surface_theta_k, 300.0)
        && sounding.moisture_qv_kgkg.has_value()
        && same_profile_number(*sounding.moisture_qv_kgkg, 0.0)
        && supported_hodograph
        && config.initial_conditions.trigger.type == TriggerType::Bubble
        && same_profile_number(bubble.dtheta_k, 2.0)
        && same_profile_number(bubble.radius_km, 1.5)
        && same_profile_number(bubble.center_x_m, 4500.0)
        && same_profile_number(bubble.center_y_m, 4500.0)
        && same_profile_number(bubble.center_z_m, 1500.0)
        && config.output.preset == OutputPreset::Minimal
        && config.output.format == OutputFormat::Npy3D
        && !config.output.async_io && !config.output.live_shm
        && config.output.qualification_diagnostics_preset
            == QualificationDiagnosticsPreset::None
        && !config.output.trajectories
        && config.extensions.solver.slow_rk2
        && same_profile_number(config.extensions.solver.lid_bc_gamma, -1.0)
        && !config.extensions.solver.axis_u_m1_filter
        && config.extensions.chaos.scheme == ChaosScheme::None
        && !config.radar.enabled && config.radar.products.empty()
        && config.extensions.checkpoint.every_s == 1
        && config.extensions.checkpoint.keep_last == 0
        && config.extensions.checkpoint.minimum_free_space_mib == 0
        && config.extensions.checkpoint.directory.empty()
        && config.extensions.checkpoint.resume_from.empty()
        && config.extensions.validation.guard_mode
            == ValidationGuardMode::Error
        && config.extensions.diagnostics.scientific_every_steps == 1
        && config.extensions.diagnostics.deterministic_reductions;
}

bool exact_active_collocated_terrain(const ConfigV1& config)
{
    const auto& compute = config.extensions.compute;
    const auto& motion = config.domain_motion;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& micro = config.microphysics.parameters;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    const auto& chaos = config.extensions.chaos;
    const bool supported_microphysics =
        config.microphysics.scheme == MicrophysicsScheme::None ||
        config.microphysics.scheme == MicrophysicsScheme::Kessler ||
        config.microphysics.scheme == MicrophysicsScheme::Lin ||
        config.microphysics.scheme == MicrophysicsScheme::Thompson ||
        config.microphysics.scheme == MicrophysicsScheme::Milbrandt;
    const bool canonical_microphysics_parameters =
        same_profile_number(micro.shared.cloud_water_threshold_kgkg, 0.001) &&
        same_profile_number(micro.shared.autoconversion_rate_s_inv, 0.001) &&
        same_profile_number(micro.kessler.accretion_coefficient, 2.2) &&
        same_profile_number(micro.kessler.evaporation_efficiency, 1.0) &&
        same_profile_number(micro.lin.accretion_coefficient, 2.2) &&
        same_profile_number(micro.lin.evaporation_rate_s_inv, 0.001) &&
        same_profile_number(micro.thompson.evaporation_rate_s_inv, 0.001) &&
        same_profile_number(
            micro.thompson.water_friendly_aerosol_surface_m3, 350.0e6) &&
        same_profile_number(
            micro.thompson.ice_friendly_aerosol_surface_m3, 2.0e6) &&
        same_profile_number(micro.milbrandt.evaporation_rate_s_inv, 0.001) &&
        !micro.milbrandt.triple_moment && micro.milbrandt.hail_processes;
    const std::string_view chaos_scheme_id = chaos_scheme_name(chaos.scheme);
    const bool supported_chaos =
        (chaos.scheme == ChaosScheme::None ||
         chaos.scheme == ChaosScheme::InitialConditions ||
         chaos.scheme == ChaosScheme::BoundaryLayer ||
         chaos.scheme == ChaosScheme::FullStochastic) &&
        tmv::chaos::chaos_targets_are_canonical_or_implicit(
            chaos_scheme_id, chaos.apply_to_ic,
            chaos.apply_to_tendencies);
    const bool exact_bell = config.terrain.scheme == TerrainScheme::Bell
        && same_profile_number(config.terrain.bell.h0, 220.0)
        && same_profile_number(config.terrain.bell.a, 1600.0)
        && config.terrain.bell.axisymmetric;
    const bool exact_schar = config.terrain.scheme == TerrainScheme::Schar
        && same_profile_number(config.terrain.schar.H, 220.0)
        && same_profile_number(config.terrain.schar.a, 1600.0)
        && same_profile_number(config.terrain.schar.ell, 800.0);
    return config.extensions.execution.mode == ExecutionMode::Qualified
        && compute.backend == ComputeBackend::CPU
        && !compute.allow_fallback && !compute.validate_parity
        && !compute.gpu_apply_microphysics && !compute.gpu_apply_slow
        && !compute.gpu_apply_sgs
        && config.grid.coordinate_system == CoordinateSystem::Cartesian
        && config.grid.staggering == StaggerType::Collocated
        && config.grid.nx == 18 && config.grid.ny == 18
        && config.grid.nz == 18
        && same_profile_number(config.grid.dx, 500.0)
        && same_profile_number(config.grid.dy, 500.0)
        && same_profile_number(config.grid.dz, 250.0)
        && same_profile_number(config.grid.ztop, 4500.0)
        && config.time.integrator == TimeIntegrator::SplitExplicit
        && same_profile_number(config.time.run_time, 2.0)
        && same_profile_number(config.time.dt_large, 0.1)
        && config.time.acoustic_substeps == 6
        && !config.time.adaptive_dt
        && same_profile_number(config.time.cfl_safety, 0.7)
        && same_profile_number(config.time.output_interval, 1.0)
        && motion.mode == DomainMotionMode::Stationary
        && same_profile_number(motion.frame_velocity.u_ms, 0.0)
        && same_profile_number(motion.frame_velocity.v_ms, 0.0)
        && !motion.lifetime_planning.enabled && !motion.tracking.enabled
        && !motion.tracking.recycle_domain && !motion.nesting.enabled
        && config.dynamics.scheme
            == "klemp_wilhelmson_split_cartesian_agrid"
        && !config.dynamics.axisymmetric
        && (config.advection.scheme == AdvectionScheme::None ||
            config.advection.scheme == AdvectionScheme::TVD ||
            config.advection.scheme == AdvectionScheme::WENO5)
        && lateral.west == LateralBC::Rigid
        && lateral.east == LateralBC::Rigid
        && lateral.south == LateralBC::Rigid
        && lateral.north == LateralBC::Rigid
        && config.boundary_conditions.bottom == BottomBC::RigidSlip
        && config.boundary_conditions.top == TopBC::Rigid
        && !config.damping.enabled
        && same_profile_number(config.damping.z_d, 4500.0)
        && same_profile_number(config.damping.alpha, 0.0)
        && config.damping.mode == DampingMode::Explicit
        && config.damping.lateral_width == 0
        && same_profile_number(config.damping.lateral_alpha, 0.0)
        && !config.damping.lateral_sponge_enabled
        && !config.damping.w_only_top_sponge
        && config.damping.w_only_depth_cells == 0
        && same_profile_number(config.damping.w_only_alpha_max, 0.0)
        && config.coriolis.enabled
        && config.coriolis.mode == CoriolisMode::BetaPlane
        && same_profile_number(config.coriolis.latitude_deg, 35.0)
        && supported_microphysics && canonical_microphysics_parameters
        && config.turbulence.scheme == TurbulenceScheme::Smagorinsky
        && config.turbulence.mode == TurbulenceMode::TwoD
        && config.turbulence.filter_width ==
            TurbulenceFilterWidth::CubicRoot
        && config.turbulence.stability_correction ==
            TurbulenceStabilityCorrection::None
        && !config.turbulence.dynamic_coefficient
        && same_profile_number(config.turbulence.Cs, 0.18)
        && same_profile_number(config.turbulence.Pr_t, 1.0 / 3.0)
        && same_profile_number(config.turbulence.Sc_t, 1.0 / 3.0)
        && config.diffusion.scheme == DiffusionScheme::Implicit
        && config.diffusion.operator_order == 2
        && same_profile_number(config.diffusion.K_h, 4000.0)
        && same_profile_number(config.diffusion.K_v, 4000.0)
        && config.diffusion.apply_to ==
            DiffusionTarget::ScalarsAndMomentum
        && config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::Constant
        && config.surface.scheme == SurfaceScheme::Slab
        && config.surface.apply_fluxes
        && same_profile_number(config.surface.z0m_m, 0.1)
        && same_profile_number(config.surface.z0h_m, 0.01)
        && same_profile_number(config.surface.albedo, 0.2)
        && same_profile_number(config.surface.emissivity, 0.96)
        && same_profile_number(config.surface.tsk_k, 305.0)
        && config.boundary_layer.scheme == BoundaryLayerScheme::MYNN
        && config.boundary_layer.pbl_top_method == BoundaryLayerTopMethod::TKE
        && same_profile_number(config.boundary_layer.cadence_s, 60.0)
        && config.boundary_layer.clear_sky_only
        && config.radiation.scheme == RadiationScheme::RRTMG
        && config.radiation.solar_forcing ==
            RadiationSolarForcing::IdealizedDiurnal
        && same_profile_number(config.radiation.latitude_deg, 35.0)
        && same_profile_number(config.radiation.declination_deg, 15.0)
        && same_profile_number(
            config.radiation.local_solar_time_at_start_h, 12.0)
        && config.radiation.do_lw && config.radiation.do_sw
        && same_profile_number(config.radiation.cadence_s, 1.0)
        && config.radiation.clear_sky_only
        && same_profile_number(config.radiation.rrtmg_ozone_vmr, 4.0e-6)
        && (exact_bell || exact_schar)
        && config.terrain.horizontal_placement ==
            TerrainHorizontalPlacement::GridAttached
        && config.terrain.coordinate == TerrainCoordinate::Smoothed
        && config.terrain.compute_derivatives
        && config.terrain.compute_metrics
        && same_profile_number(config.terrain.smoothing_decay, 0.1)
        && sounding.type == SoundingType::ParametricCAPE
        && same_profile_number(sounding.cape_jkg, 0.0)
        && same_profile_number(sounding.tropopause_m, 8000.0)
        && same_profile_number(sounding.surface_theta_k, 300.0)
        && sounding.moisture_qv_kgkg.has_value()
        && same_profile_number(*sounding.moisture_qv_kgkg, 0.004)
        && hodograph.type == HodographType::WKParam
        && same_profile_number(hodograph.u_sfc_ms, 0.0)
        && same_profile_number(hodograph.v_sfc_ms, 0.0)
        && same_profile_number(hodograph.u_1km_ms, 0.0)
        && same_profile_number(hodograph.v_1km_ms, 0.0)
        && same_profile_number(hodograph.u_6km_ms, 0.0)
        && same_profile_number(hodograph.v_6km_ms, 0.0)
        && config.initial_conditions.trigger.type == TriggerType::Bubble
        && same_profile_number(bubble.dtheta_k, 2.0)
        && same_profile_number(bubble.cloud_water_kgkg, 0.001)
        && same_profile_number(bubble.radius_km, 1.5)
        && same_profile_number(bubble.center_x_m, 4500.0)
        && same_profile_number(bubble.center_y_m, 4500.0)
        && same_profile_number(bubble.center_z_m, 1500.0)
        && config.output.preset == OutputPreset::DynamicsFull
        && config.output.format == OutputFormat::Npy3D
        && !config.output.async_io && !config.output.live_shm
        && config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::None
        && !config.output.trajectories
        && config.extensions.solver.slow_rk2
        && same_profile_number(config.extensions.solver.lid_bc_gamma, -1.0)
        && !config.extensions.solver.axis_u_m1_filter
        && supported_chaos
        && (chaos.scheme == ChaosScheme::None ||
            (chaos.seed == 918273 && chaos.member_id == 4))
        && !chaos.native_generation_required
        && !config.radar.enabled && config.radar.products.empty()
        && config.extensions.checkpoint.every_s == 1
        && config.extensions.checkpoint.keep_last == 0
        && config.extensions.checkpoint.minimum_free_space_mib == 0
        && config.extensions.checkpoint.directory.empty()
        && config.extensions.checkpoint.resume_from.empty()
        && config.extensions.validation.guard_mode == ValidationGuardMode::Error
        && config.extensions.diagnostics.scientific_every_steps == 1
        && config.extensions.diagnostics.deterministic_reductions;
}

bool exact_common_profile(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    const auto& compute = config.extensions.compute;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        compute.backend == ComputeBackend::CPU && !compute.allow_fallback &&
        !compute.validate_parity && !compute.gpu_apply_microphysics &&
        !compute.gpu_apply_slow && !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 12 && config.grid.ny == 12 && config.grid.nz == 18 &&
        same_profile_number(config.grid.dx, 500.0) &&
        same_profile_number(config.grid.dy, 500.0) &&
        same_profile_number(config.grid.dz, 500.0) &&
        same_profile_number(config.grid.ztop, 9000.0) &&
        same_profile_number(config.time.run_time, 2.0) &&
        same_profile_number(config.time.dt_large, 0.2) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        same_profile_number(config.time.cfl_safety, 0.7) &&
        same_profile_number(config.time.output_interval, 1.0) &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && !motion.nesting.enabled &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.advection.scheme == AdvectionScheme::TVD &&
        lateral.west == LateralBC::Radiative &&
        lateral.east == LateralBC::Radiative &&
        lateral.south == LateralBC::Radiative &&
        lateral.north == LateralBC::Radiative &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled &&
        same_profile_number(config.damping.z_d, 9000.0) &&
        same_profile_number(config.damping.alpha, 0.0) &&
        config.damping.mode == DampingMode::Explicit &&
        !config.damping.lateral_sponge_enabled &&
        config.damping.lateral_width == 0 &&
        same_profile_number(config.damping.lateral_alpha, 0.0) &&
        !config.damping.w_only_top_sponge &&
        config.damping.w_only_depth_cells == 0 &&
        same_profile_number(config.damping.w_only_alpha_max, 0.0) &&
        !config.coriolis.enabled &&
        config.microphysics.scheme == MicrophysicsScheme::None &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.turbulence.mode == TurbulenceMode::ThreeD &&
        same_profile_number(config.turbulence.Cs, 0.18) &&
        same_profile_number(config.turbulence.Pr_t, 0.7) &&
        same_profile_number(config.turbulence.Sc_t, 0.7) &&
        config.diffusion.scheme == DiffusionScheme::Implicit &&
        config.diffusion.operator_order == 2 &&
        same_profile_number(config.diffusion.K_h, 0.0) &&
        same_profile_number(config.diffusion.K_v, 4.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::Slab &&
        config.surface.apply_fluxes &&
        same_profile_number(config.surface.z0m_m, 0.1) &&
        same_profile_number(config.surface.z0h_m, 0.01) &&
        same_profile_number(config.surface.albedo, 0.2) &&
        same_profile_number(config.surface.emissivity, 0.96) &&
        same_profile_number(config.surface.tsk_k, 302.0) &&
        same_profile_number(config.surface.qsfc_kgkg, 0.014) &&
        config.boundary_layer.scheme == BoundaryLayerScheme::Slab &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        config.radiation.scheme == RadiationScheme::None &&
        config.terrain.coordinate == TerrainCoordinate::Smoothed &&
        config.terrain.compute_derivatives && config.terrain.compute_metrics &&
        same_profile_number(config.terrain.smoothing_decay, 0.1) &&
        sounding.type == SoundingType::ParametricCAPE &&
        same_profile_number(sounding.cape_jkg, 600.0) &&
        same_profile_number(sounding.tropopause_m, 8000.0) &&
        same_profile_number(sounding.surface_theta_k, 300.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        same_profile_number(*sounding.moisture_qv_kgkg, 0.012) &&
        config.initial_conditions.hodograph.type == HodographType::WKParam &&
        config.initial_conditions.trigger.type == TriggerType::Bubble &&
        same_profile_number(bubble.dtheta_k, 0.5) &&
        same_profile_number(bubble.radius_km, 1.0) &&
        same_profile_number(bubble.center_x_m, 3000.0) &&
        same_profile_number(bubble.center_y_m, 3000.0) &&
        same_profile_number(bubble.center_z_m, 1200.0) &&
        config.output.preset == OutputPreset::Minimal &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::None &&
        !config.output.trajectories &&
        config.extensions.solver.slow_rk2 &&
        same_profile_number(config.extensions.solver.lid_bc_gamma, -1.0) &&
        !config.extensions.solver.axis_u_m1_filter &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        config.extensions.checkpoint.every_s == 1 &&
        config.extensions.checkpoint.keep_last == 0 &&
        config.extensions.checkpoint.minimum_free_space_mib == 0 &&
        config.extensions.checkpoint.directory.empty() &&
        config.extensions.checkpoint.resume_from.empty() &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions;
}

bool structural_coupled_mapped_cpu_profile(const ConfigV1& config)
{
    const auto& compute = config.extensions.compute;
    const auto& motion = config.domain_motion;
    const auto& lateral = config.boundary_conditions.lateral;
    const bool supported_motion =
        motion.mode == DomainMotionMode::Stationary ||
        motion.mode == DomainMotionMode::ConstantVelocity;
    const bool supported_terrain =
        config.terrain.scheme == TerrainScheme::Bell ||
        config.terrain.scheme == TerrainScheme::Schar;
    const bool supported_microphysics =
        config.microphysics.scheme == MicrophysicsScheme::None ||
        config.microphysics.scheme == MicrophysicsScheme::Kessler ||
        config.microphysics.scheme == MicrophysicsScheme::Lin ||
        config.microphysics.scheme == MicrophysicsScheme::Thompson ||
        config.microphysics.scheme == MicrophysicsScheme::Milbrandt;
    const bool no_surface_pbl =
        config.surface.scheme == SurfaceScheme::None &&
         !config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::None;
    const bool slab_surface_pbl =
        config.surface.scheme == SurfaceScheme::Slab &&
         config.surface.apply_fluxes &&
         (config.boundary_layer.scheme == BoundaryLayerScheme::Slab ||
          config.boundary_layer.scheme == BoundaryLayerScheme::YSU ||
          config.boundary_layer.scheme == BoundaryLayerScheme::MYNN);
    const bool supported_surface_pbl =
        no_surface_pbl || slab_surface_pbl;
    const bool supported_turbulence =
        (no_surface_pbl &&
         ((config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
           config.turbulence.mode == TurbulenceMode::ThreeD &&
           config.turbulence.stability_correction ==
               TurbulenceStabilityCorrection::None) ||
          (config.turbulence.scheme == TurbulenceScheme::TKE &&
           config.turbulence.mode == TurbulenceMode::ThreeD &&
           config.turbulence.stability_correction ==
               TurbulenceStabilityCorrection::TkeBased))) ||
        (slab_surface_pbl &&
         ((config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
           config.turbulence.mode == TurbulenceMode::TwoD &&
           config.turbulence.stability_correction ==
               TurbulenceStabilityCorrection::None) ||
          (config.turbulence.scheme == TurbulenceScheme::TKE &&
           config.turbulence.mode == TurbulenceMode::TwoD &&
           config.turbulence.stability_correction ==
               TurbulenceStabilityCorrection::TkeBased)));
    const bool supported_radiation =
        (config.radiation.scheme == RadiationScheme::SimpleGrey &&
         config.radiation.do_lw &&
         ((config.radiation.clear_sky_only &&
           config.radiation.simple_grey_optical_model ==
               "pressure_clear_sky_v1") ||
          (!config.radiation.clear_sky_only &&
           config.radiation.simple_grey_optical_model ==
               "absorbing_water_cloud_path_v1"))) ||
        (config.radiation.scheme == RadiationScheme::RRTMG &&
         config.radiation.do_lw);
    const bool supported_chaos =
        config.extensions.chaos.scheme == ChaosScheme::None ||
        config.extensions.chaos.scheme == ChaosScheme::InitialConditions ||
        config.extensions.chaos.scheme == ChaosScheme::BoundaryLayer ||
        config.extensions.chaos.scheme == ChaosScheme::FullStochastic;
    const bool constant_diffusion =
        config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::Constant &&
        std::isfinite(config.diffusion.K_h) && config.diffusion.K_h >= 0.0 &&
        std::isfinite(config.diffusion.K_v) && config.diffusion.K_v >= 0.0;
    const bool supported_diffusion = constant_diffusion &&
        ((config.diffusion.scheme == DiffusionScheme::Explicit &&
          (config.diffusion.apply_to == DiffusionTarget::Scalars ||
           config.diffusion.apply_to ==
               DiffusionTarget::ScalarsAndMomentum)) ||
         (config.diffusion.scheme == DiffusionScheme::Implicit &&
          (config.diffusion.apply_to == DiffusionTarget::Scalars ||
           config.diffusion.apply_to ==
               DiffusionTarget::ScalarsAndMomentum)));
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        compute.backend == ComputeBackend::CPU && !compute.allow_fallback &&
        !compute.validate_parity && !compute.gpu_apply_microphysics &&
        !compute.gpu_apply_slow && !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.time.integrator == TimeIntegrator::SplitExplicit &&
        !config.time.adaptive_dt &&
        supported_motion &&
        std::isfinite(motion.frame_velocity.u_ms) &&
        std::isfinite(motion.frame_velocity.v_ms) &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && !motion.nesting.enabled &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        (config.advection.scheme == AdvectionScheme::Auto ||
         config.advection.scheme == AdvectionScheme::TVD ||
         config.advection.scheme == AdvectionScheme::WENO5) &&
        lateral.west == LateralBC::Radiative &&
        lateral.east == LateralBC::Radiative &&
        lateral.south == LateralBC::Radiative &&
        lateral.north == LateralBC::Radiative &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled && !config.coriolis.enabled &&
        supported_microphysics &&
        supported_turbulence &&
        config.turbulence.filter_width == TurbulenceFilterWidth::CubicRoot &&
        !config.turbulence.dynamic_coefficient &&
        config.diffusion.operator_order == 2 &&
        supported_diffusion &&
        supported_surface_pbl &&
        supported_radiation && supported_terrain &&
        config.terrain.horizontal_placement ==
            TerrainHorizontalPlacement::GridAttached &&
        config.terrain.coordinate == TerrainCoordinate::Smoothed &&
        config.terrain.compute_derivatives &&
        config.terrain.compute_metrics &&
        supported_chaos &&
        !config.radar.enabled && config.radar.products.empty();
}

bool structural_mapped_pbl_implicit_cpu_profile(const ConfigV1& config)
{
    const auto& compute = config.extensions.compute;
    const auto& motion = config.domain_motion;
    const auto& lateral = config.boundary_conditions.lateral;
    const bool supported_motion =
        motion.mode == DomainMotionMode::Stationary ||
        motion.mode == DomainMotionMode::ConstantVelocity;
    const bool supported_terrain =
        config.terrain.scheme == TerrainScheme::Bell ||
        config.terrain.scheme == TerrainScheme::Schar;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        compute.backend == ComputeBackend::CPU && !compute.allow_fallback &&
        !compute.validate_parity && !compute.gpu_apply_microphysics &&
        !compute.gpu_apply_slow && !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.time.integrator == TimeIntegrator::SplitExplicit &&
        !config.time.adaptive_dt && supported_motion &&
        std::isfinite(motion.frame_velocity.u_ms) &&
        std::isfinite(motion.frame_velocity.v_ms) &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && !motion.nesting.enabled &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.advection.scheme == AdvectionScheme::TVD &&
        lateral.west == LateralBC::Radiative &&
        lateral.east == LateralBC::Radiative &&
        lateral.south == LateralBC::Radiative &&
        lateral.north == LateralBC::Radiative &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled && !config.coriolis.enabled &&
        config.microphysics.scheme == MicrophysicsScheme::None &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.diffusion.scheme == DiffusionScheme::Implicit &&
        config.diffusion.operator_order == 2 &&
        std::isfinite(config.diffusion.K_h) && config.diffusion.K_h >= 0.0 &&
        std::isfinite(config.diffusion.K_v) && config.diffusion.K_v > 0.0 &&
        (config.diffusion.apply_to == DiffusionTarget::Scalars ||
         config.diffusion.apply_to ==
             DiffusionTarget::ScalarsAndMomentum) &&
        config.surface.scheme == SurfaceScheme::Slab &&
        config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::Slab &&
        config.radiation.scheme == RadiationScheme::None &&
        supported_terrain &&
        config.terrain.horizontal_placement ==
            TerrainHorizontalPlacement::GridAttached &&
        config.terrain.coordinate == TerrainCoordinate::Smoothed &&
        config.terrain.compute_derivatives &&
        config.terrain.compute_metrics &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        !config.radar.enabled && config.radar.products.empty();
}

bool exact_dry_stationary_mapped_reference(const ConfigV1& config)
{
    const auto& compute = config.extensions.compute;
    const auto& motion = config.domain_motion;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& terrain = config.terrain;
    const bool supported_shape =
        (terrain.scheme == TerrainScheme::Bell
         && same_profile_number(terrain.bell.h0,220.0)
         && same_profile_number(terrain.bell.a,1600.0) && terrain.bell.axisymmetric)
        || (terrain.scheme == TerrainScheme::Schar
            && same_profile_number(terrain.schar.H,220.0)
            && same_profile_number(terrain.schar.a,1600.0)
            && same_profile_number(terrain.schar.ell,2000.0));
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        (compute.backend == ComputeBackend::CPU || compute.backend == ComputeBackend::GPU ||
         compute.backend == ComputeBackend::Metal) &&
        !compute.allow_fallback &&
        !compute.validate_parity && !compute.gpu_apply_microphysics &&
        (compute.gpu_apply_slow == (compute.backend != ComputeBackend::CPU)) && !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 12 && config.grid.ny == 12 && config.grid.nz == 18 &&
        same_profile_number(config.grid.dx, 500.0) &&
        same_profile_number(config.grid.dy, 500.0) &&
        same_profile_number(config.grid.dz, 500.0) &&
        same_profile_number(config.grid.ztop, 9000.0) &&
        config.time.integrator == TimeIntegrator::SplitExplicit &&
        same_profile_number(config.time.run_time, 2.0) &&
        same_profile_number(config.time.dt_large, 0.1) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        same_profile_number(config.time.cfl_safety, 0.7) &&
        same_profile_number(config.time.output_interval, 1.0) &&
        motion.mode == DomainMotionMode::Stationary &&
        same_profile_number(motion.frame_velocity.u_ms, 0.0) &&
        same_profile_number(motion.frame_velocity.v_ms, 0.0) &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && !motion.nesting.enabled &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.advection.scheme == AdvectionScheme::WENO5 &&
        lateral.west == LateralBC::Periodic &&
        lateral.east == LateralBC::Periodic &&
        lateral.south == LateralBC::Periodic &&
        lateral.north == LateralBC::Periodic &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled &&
        same_profile_number(config.damping.z_d, 9000.0) &&
        same_profile_number(config.damping.alpha, 0.0) &&
        config.damping.mode == DampingMode::Explicit &&
        config.damping.lateral_width == 0 &&
        same_profile_number(config.damping.lateral_alpha, 0.0) &&
        !config.damping.lateral_sponge_enabled &&
        !config.damping.w_only_top_sponge &&
        config.damping.w_only_depth_cells == 0 &&
        same_profile_number(config.damping.w_only_alpha_max, 0.0) &&
        !config.coriolis.enabled &&
        config.microphysics.scheme == MicrophysicsScheme::None &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.turbulence.mode == TurbulenceMode::ThreeD &&
        config.turbulence.filter_width == TurbulenceFilterWidth::CubicRoot &&
        config.turbulence.stability_correction ==
            TurbulenceStabilityCorrection::None &&
        same_profile_number(config.turbulence.user_filter_width_m, 0.0) &&
        !config.turbulence.dynamic_coefficient &&
        same_profile_number(config.turbulence.Cs, 0.18) &&
        same_profile_number(config.turbulence.Pr_t, 1.0 / 3.0) &&
        same_profile_number(config.turbulence.Sc_t, 1.0 / 3.0) &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        same_profile_number(config.diffusion.K_h, 0.0) &&
        same_profile_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::Constant &&
        config.surface.scheme == SurfaceScheme::None &&
        !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::None &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        config.radiation.scheme == RadiationScheme::None &&
        supported_shape &&
        config.terrain.horizontal_placement ==
            TerrainHorizontalPlacement::GridAttached &&
        config.terrain.coordinate == TerrainCoordinate::Smoothed &&
        config.terrain.compute_derivatives && config.terrain.compute_metrics &&
        same_profile_number(config.terrain.smoothing_decay, 0.1) &&
        sounding.type == SoundingType::ParametricCAPE &&
        same_profile_number(sounding.cape_jkg, 3275.016) &&
        same_profile_number(sounding.tropopause_m, 8000.0) &&
        same_profile_number(sounding.surface_theta_k, 300.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        same_profile_number(*sounding.moisture_qv_kgkg, 0.0) &&
        hodograph.type == HodographType::WKParam &&
        same_profile_number(hodograph.u_sfc_ms, 2.0) &&
        same_profile_number(hodograph.v_sfc_ms, -1.0) &&
        same_profile_number(hodograph.u_1km_ms, 2.0) &&
        same_profile_number(hodograph.v_1km_ms, -1.0) &&
        same_profile_number(hodograph.u_6km_ms, 2.0) &&
        same_profile_number(hodograph.v_6km_ms, -1.0) &&
        config.initial_conditions.trigger.type == TriggerType::None &&
        config.output.preset == OutputPreset::Minimal &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::None &&
        !config.output.trajectories &&
        config.extensions.solver.slow_rk2 &&
        same_profile_number(config.extensions.solver.lid_bc_gamma, -1.0) &&
        !config.extensions.solver.axis_u_m1_filter &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        !config.extensions.chaos.native_generation_required &&
        !config.radar.enabled && config.radar.products.empty() &&
        config.extensions.checkpoint.every_s == 1 &&
        config.extensions.checkpoint.keep_last == 0 &&
        config.extensions.checkpoint.minimum_free_space_mib == 0 &&
        config.extensions.checkpoint.directory.empty() &&
        config.extensions.checkpoint.resume_from.empty() &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions;
}

} // namespace

bool is_public_qualified_mapped_terrain_profile(
    const ConfigV1& config) noexcept
{
    return (exact_common_profile(config) &&
            (exact_stationary_bell(config) || exact_translating_schar(config))) ||
        exact_dry_stationary_mapped_reference(config) ||
        structural_coupled_mapped_cpu_profile(config) ||
        structural_mapped_pbl_implicit_cpu_profile(config) ||
        exact_collocated_bell(config) ||
        exact_active_collocated_terrain(config);
}

bool is_public_qualified_collocated_mapped_terrain_profile(
    const ConfigV1& config) noexcept
{
    return exact_collocated_bell(config)
        || exact_active_collocated_terrain(config);
}

bool is_public_qualified_native_mapped_terrain_profile(
    const ConfigV1& config) noexcept
{
    return config.extensions.compute.backend != ComputeBackend::CPU &&
        exact_dry_stationary_mapped_reference(config);
}

bool is_public_qualified_vulkan_mapped_terrain_profile(
    const ConfigV1& config) noexcept
{
    return config.extensions.compute.backend == ComputeBackend::GPU &&
        exact_dry_stationary_mapped_reference(config);
}

bool is_public_qualified_metal_mapped_terrain_profile(
    const ConfigV1& config) noexcept
{
    return config.extensions.compute.backend == ComputeBackend::Metal &&
        exact_dry_stationary_mapped_reference(config);
}

} // namespace tmv::init::config
