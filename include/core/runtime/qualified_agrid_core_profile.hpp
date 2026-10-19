#pragma once

#include "init/config/config_v1.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::runtime
{

inline bool qualified_agrid_core_number(double value, double expected)
{
    return std::isfinite(value) &&
        std::abs(value - expected) <=
            8.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(expected));
}

inline bool is_exact_qualified_agrid_core_profile(
    const tmv::init::config::ConfigV1& config)
{
    using namespace tmv::init::config;
    const auto& motion = config.domain_motion;
    const auto& compute = config.extensions.compute;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    const bool cpu_owner = compute.backend == ComputeBackend::CPU &&
        !compute.gpu_apply_slow;
    const bool native_owner =
        (compute.backend == ComputeBackend::GPU ||
         compute.backend == ComputeBackend::Metal) &&
        compute.gpu_apply_slow;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        (cpu_owner || native_owner) && !compute.allow_fallback &&
        !compute.gpu_apply_microphysics &&
        !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::Collocated &&
        config.grid.nx == 18 && config.grid.ny == 18 && config.grid.nz == 18 &&
        qualified_agrid_core_number(config.grid.dx, 500.0) &&
        qualified_agrid_core_number(config.grid.dy, 500.0) &&
        qualified_agrid_core_number(config.grid.dz, 250.0) &&
        qualified_agrid_core_number(config.grid.ztop, 4500.0) &&
        qualified_agrid_core_number(config.time.run_time, 2.0) &&
        qualified_agrid_core_number(config.time.dt_large, 0.1) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        qualified_agrid_core_number(config.time.cfl_safety, 0.7) &&
        qualified_agrid_core_number(config.time.output_interval, 1.0) &&
        motion.mode == DomainMotionMode::Stationary &&
        qualified_agrid_core_number(motion.frame_velocity.u_ms, 0.0) &&
        qualified_agrid_core_number(motion.frame_velocity.v_ms, 0.0) &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && !motion.nesting.enabled &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_agrid" &&
        !config.dynamics.axisymmetric &&
        (config.advection.scheme == AdvectionScheme::Auto ||
         config.advection.scheme == AdvectionScheme::None ||
         config.advection.scheme == AdvectionScheme::TVD) &&
        lateral.west == LateralBC::Rigid &&
        lateral.east == LateralBC::Rigid &&
        lateral.south == LateralBC::Rigid &&
        lateral.north == LateralBC::Rigid &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled &&
        qualified_agrid_core_number(config.damping.z_d, 4500.0) &&
        qualified_agrid_core_number(config.damping.alpha, 0.0) &&
        config.damping.mode == DampingMode::Explicit &&
        config.damping.lateral_width == 0 &&
        qualified_agrid_core_number(config.damping.lateral_alpha, 0.0) &&
        !config.damping.lateral_sponge_enabled &&
        !config.damping.w_only_top_sponge &&
        config.damping.w_only_depth_cells == 0 &&
        qualified_agrid_core_number(config.damping.w_only_alpha_max, 0.0) &&
        !config.coriolis.enabled &&
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::None &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        qualified_agrid_core_number(config.diffusion.K_h, 0.0) &&
        qualified_agrid_core_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::None &&
        !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::None &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        config.radiation.scheme == RadiationScheme::None &&
        config.terrain.scheme == TerrainScheme::None &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        !config.radar.enabled && config.radar.products.empty() &&
        sounding.type == SoundingType::ParametricCAPE &&
        qualified_agrid_core_number(sounding.cape_jkg, 0.0) &&
        qualified_agrid_core_number(sounding.tropopause_m, 8000.0) &&
        qualified_agrid_core_number(sounding.surface_theta_k, 300.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        qualified_agrid_core_number(*sounding.moisture_qv_kgkg, 0.004) &&
        hodograph.type == HodographType::WKParam &&
        qualified_agrid_core_number(hodograph.u_sfc_ms, 0.0) &&
        qualified_agrid_core_number(hodograph.v_sfc_ms, 0.0) &&
        qualified_agrid_core_number(hodograph.u_1km_ms, 0.0) &&
        qualified_agrid_core_number(hodograph.v_1km_ms, 0.0) &&
        qualified_agrid_core_number(hodograph.u_6km_ms, 0.0) &&
        qualified_agrid_core_number(hodograph.v_6km_ms, 0.0) &&
        config.initial_conditions.trigger.type == TriggerType::Bubble &&
        qualified_agrid_core_number(bubble.dtheta_k, 2.0) &&
        qualified_agrid_core_number(bubble.radius_km, 1.5) &&
        qualified_agrid_core_number(bubble.center_x_m, 4500.0) &&
        qualified_agrid_core_number(bubble.center_y_m, 4500.0) &&
        qualified_agrid_core_number(bubble.center_z_m, 1500.0) &&
        config.output.preset == OutputPreset::Minimal &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::None &&
        !config.output.trajectories &&
        config.extensions.solver.slow_rk2 &&
        qualified_agrid_core_number(
            config.extensions.solver.lid_bc_gamma, -1.0) &&
        !config.extensions.solver.axis_u_m1_filter &&
        config.extensions.checkpoint.every_s == 1 &&
        config.extensions.checkpoint.directory.empty() &&
        config.extensions.checkpoint.resume_from.empty() &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions;
}

inline bool is_public_ready_flat_agrid_cpu_family(
    const tmv::init::config::ConfigV1& config)
{
    using namespace tmv::init::config;
    const auto& compute = config.extensions.compute;
    const auto& motion = config.domain_motion;
    const auto& lateral = config.boundary_conditions.lateral;
    const bool supported_lateral =
        lateral.west != LateralBC::Radiative &&
        lateral.east != LateralBC::Radiative &&
        lateral.south != LateralBC::Radiative &&
        lateral.north != LateralBC::Radiative;
    const bool supported_microphysics =
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::None ||
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::Kessler ||
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::Lin ||
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::Thompson ||
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::Milbrandt;
    const bool supported_column_physics =
        (config.surface.scheme == SurfaceScheme::None &&
         !config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::None &&
         (config.radiation.scheme == RadiationScheme::None ||
          config.radiation.scheme == RadiationScheme::SimpleGrey ||
          config.radiation.scheme == RadiationScheme::RRTMG)) ||
        (config.surface.scheme == SurfaceScheme::Slab &&
         config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::Slab &&
         config.radiation.scheme == RadiationScheme::SimpleGrey) ||
        (config.surface.scheme == SurfaceScheme::Slab &&
         config.surface.apply_fluxes &&
         (config.boundary_layer.scheme == BoundaryLayerScheme::YSU ||
          config.boundary_layer.scheme == BoundaryLayerScheme::MYNN) &&
         config.radiation.scheme == RadiationScheme::RRTMG);
    const bool supported_turbulence =
        config.turbulence.scheme == TurbulenceScheme::None ||
        (config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
         config.turbulence.filter_width ==
             TurbulenceFilterWidth::CubicRoot &&
         !config.turbulence.dynamic_coefficient &&
         config.turbulence.stability_correction ==
             TurbulenceStabilityCorrection::None) ||
        (config.turbulence.scheme == TurbulenceScheme::TKE &&
         config.turbulence.filter_width ==
             TurbulenceFilterWidth::CubicRoot &&
         !config.turbulence.dynamic_coefficient &&
         config.turbulence.stability_correction ==
             TurbulenceStabilityCorrection::TkeBased &&
         qualified_agrid_core_number(
             config.turbulence.Pr_t, 1.0 / 3.0) &&
         qualified_agrid_core_number(
             config.turbulence.Sc_t, 1.0 / 3.0));
    const bool supported_diffusion =
        (config.diffusion.scheme == DiffusionScheme::Explicit ||
         config.diffusion.scheme == DiffusionScheme::Implicit) &&
        config.diffusion.operator_order == 2 &&
        config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::Constant &&
        (config.diffusion.apply_to == DiffusionTarget::Scalars ||
         config.diffusion.apply_to == DiffusionTarget::ScalarsAndMomentum);
    const bool supported_chaos =
        !config.extensions.chaos.native_generation_required &&
        (config.extensions.chaos.scheme == ChaosScheme::None ||
         config.extensions.chaos.scheme == ChaosScheme::InitialConditions ||
         config.extensions.chaos.scheme == ChaosScheme::BoundaryLayer ||
         config.extensions.chaos.scheme == ChaosScheme::FullStochastic);
    const bool stationary_family =
        motion.mode == DomainMotionMode::Stationary;
    const bool moving_rk =
        config.time.integrator == TimeIntegrator::SSPRK3 ||
        config.time.integrator == TimeIntegrator::RK4;
    const bool supported_moving_integrator =
        config.time.integrator == TimeIntegrator::SplitExplicit ||
        (moving_rk &&
         config.microphysics.scheme !=
             tmv::init::config::MicrophysicsScheme::Thompson &&
         config.microphysics.scheme !=
             tmv::init::config::MicrophysicsScheme::Milbrandt &&
         config.turbulence.scheme != TurbulenceScheme::TKE &&
         config.surface.scheme == SurfaceScheme::None &&
         !config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::None &&
         config.extensions.chaos.scheme == ChaosScheme::None);
    const bool qualified_constant_velocity =
        motion.mode == DomainMotionMode::ConstantVelocity &&
        std::isfinite(motion.frame_velocity.u_ms) &&
        std::isfinite(motion.frame_velocity.v_ms) &&
        supported_moving_integrator;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        compute.backend == ComputeBackend::CPU && !compute.allow_fallback &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::Collocated &&
        (config.time.integrator == TimeIntegrator::SplitExplicit ||
         config.time.integrator == TimeIntegrator::SSPRK3 ||
         config.time.integrator == TimeIntegrator::RK4) &&
        !config.time.adaptive_dt &&
        (stationary_family || qualified_constant_velocity) &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && !motion.nesting.enabled &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_agrid" &&
        !config.dynamics.axisymmetric &&
        (config.advection.scheme == AdvectionScheme::Auto ||
         config.advection.scheme == AdvectionScheme::None ||
         config.advection.scheme == AdvectionScheme::TVD ||
         config.advection.scheme == AdvectionScheme::WENO5) &&
        supported_lateral &&
        supported_microphysics && supported_turbulence &&
        supported_diffusion &&
        supported_column_physics &&
        config.terrain.scheme == TerrainScheme::None &&
        supported_chaos &&
        config.extensions.solver.slow_rk2 &&
        !config.extensions.solver.axis_u_m1_filter;
}

inline bool is_exact_qualified_agrid_vulkan_profile(
    const tmv::init::config::ConfigV1& config)
{
    return is_exact_qualified_agrid_core_profile(config) &&
        config.extensions.compute.backend ==
            tmv::init::config::ComputeBackend::GPU;
}

inline bool is_exact_qualified_agrid_metal_profile(
    const tmv::init::config::ConfigV1& config)
{
    return is_exact_qualified_agrid_core_profile(config) &&
        config.extensions.compute.backend ==
            tmv::init::config::ComputeBackend::Metal;
}

} // namespace tmv::runtime
