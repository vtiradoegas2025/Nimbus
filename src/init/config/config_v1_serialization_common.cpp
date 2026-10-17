#include "util/physics_cadence.hpp"
#include "config_v1_serialization_common.hpp"

#include "core/runtime/qualified_agrid_core_profile.hpp"
#include "core/runtime/qualified_variable_diffusion_profile.hpp"
#include "core/runtime/science_profile.hpp"
#include "core/runtime/flat_lin_storm_profile.hpp"
#include "init/sounding/parametric_target_math.hpp"
#include "microphysics/milbrandt_triple_scope.hpp"
#include "qualified_pbl_top_profile.hpp"
#include "qualified_terrain_profile.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tmv::init::config
{

namespace
{
bool same_campaign_number(double value, double expected)
{
    return std::isfinite(value) &&
        std::abs(value - expected) <=
            8.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(expected));
}

bool exact_moving_nest_science_profile(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    const auto& tracking = motion.tracking;
    const auto& nesting = motion.nesting;
    const auto& micro = config.microphysics.parameters;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    return config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 135 && config.grid.ny == 108 &&
        config.grid.nz == 50 && same_campaign_number(config.grid.dx, 1200.0) &&
        same_campaign_number(config.grid.dy, 1200.0) &&
        same_campaign_number(config.grid.dz, 320.0) &&
        same_campaign_number(config.grid.ztop, 16000.0) &&
        same_campaign_number(config.time.run_time, 4500.0) &&
        same_campaign_number(config.time.dt_large, 1.5) &&
        config.time.acoustic_substeps == 8 && !config.time.adaptive_dt &&
        same_campaign_number(config.time.cfl_safety, 0.7) &&
        same_campaign_number(config.time.output_interval, 450.0) &&
        motion.mode == DomainMotionMode::MovingNest &&
        same_campaign_number(motion.frame_velocity.u_ms, 0.0) &&
        same_campaign_number(motion.frame_velocity.v_ms, 0.0) &&
        !motion.lifetime_planning.enabled && tracking.enabled &&
        !tracking.recycle_domain && same_campaign_number(tracking.interval_s, 30.0) &&
        same_campaign_number(tracking.inner_margin_m, 15600.0) &&
        tracking.maximum_shift_cells == 14 &&
        same_campaign_number(tracking.cooldown_s, 300.0) &&
        same_campaign_number(tracking.minimum_confidence, 0.65) &&
        same_campaign_number(tracking.minimum_updraft_ms, 5.0) &&
        same_campaign_number(
            tracking.minimum_condensate_path_kgm2, 0.1) &&
        same_campaign_number(
            tracking.updraft_helicity_bottom_m_agl, 2000.0) &&
        same_campaign_number(
            tracking.updraft_helicity_top_m_agl, 5000.0) &&
        same_campaign_number(
            tracking.minimum_absolute_updraft_helicity_m2s2, 25.0) &&
        tracking.minimum_component_cells == 4 &&
        same_campaign_number(tracking.minimum_core_area_m2, 1.0e6) &&
        same_campaign_number(tracking.maximum_motion_speed_ms, 75.0) &&
        same_campaign_number(
            tracking.maximum_motion_acceleration_ms2, 0.25) &&
        tracking.minimum_continuity_samples == 3 &&
        tracking.lost_after_samples == 3 &&
        tracking.topology_confirmation_samples == 2 &&
        tracking.branch_preference == StormBranchPreference::BunkersRight &&
        same_campaign_number(
            tracking.branch_velocity_uncertainty_ms, 3.0) &&
        nesting.enabled && nesting.parent_ratio_x == 2 &&
        nesting.parent_ratio_y == 2 && nesting.parent_time_step_ratio == 2 &&
        nesting.child_origin_parent_cell_i == 2 &&
        nesting.child_origin_parent_cell_j == 1 &&
        nesting.child_cells_x == 156 && nesting.child_cells_y == 166 &&
        nesting.relaxation_width_child_cells == 8 &&
        nesting.boundary_stencil_parent_cells == 1 &&
        same_campaign_number(nesting.relaxation_e_folding_time_s, 30.0) &&
        nesting.feedback == NestingFeedback::OneWay &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.boundary_conditions.lateral.west == LateralBC::Radiative &&
        config.boundary_conditions.lateral.east == LateralBC::Radiative &&
        config.boundary_conditions.lateral.south == LateralBC::Radiative &&
        config.boundary_conditions.lateral.north == LateralBC::Radiative &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        config.damping.enabled &&
        same_campaign_number(config.damping.z_d, 14000.0) &&
        same_campaign_number(config.damping.alpha, 0.00333) &&
        config.damping.mode == DampingMode::Explicit &&
        config.damping.lateral_sponge_enabled &&
        config.damping.lateral_width == 10 &&
        same_campaign_number(config.damping.lateral_alpha, 0.05) &&
        config.damping.w_only_top_sponge &&
        config.damping.w_only_depth_cells == 8 &&
        same_campaign_number(config.damping.w_only_alpha_max, 0.2) &&
        config.coriolis.enabled &&
        config.coriolis.mode == CoriolisMode::FPlane &&
        same_campaign_number(config.coriolis.latitude_deg, 35.0) &&
        config.microphysics.scheme == MicrophysicsScheme::Kessler &&
        same_campaign_number(micro.shared.cloud_water_threshold_kgkg, 1.0e-3) &&
        same_campaign_number(micro.shared.autoconversion_rate_s_inv, 1.0e-3) &&
        same_campaign_number(micro.kessler.accretion_coefficient, 2.2) &&
        same_campaign_number(micro.kessler.evaporation_efficiency, 1.0) &&
        config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
        config.turbulence.mode == TurbulenceMode::ThreeD &&
        same_campaign_number(config.turbulence.Cs, 0.18) &&
        same_campaign_number(config.turbulence.Pr_t, 1.0 / 3.0) &&
        same_campaign_number(config.turbulence.Sc_t, 1.0 / 3.0) &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 && config.diffusion.K_h == 0.0 &&
        config.diffusion.K_v == 0.0 &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::None && !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::None &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        config.radiation.scheme == RadiationScheme::None &&
        config.terrain.scheme == TerrainScheme::None &&
        sounding.type == SoundingType::ParametricCAPE &&
        same_campaign_number(sounding.cape_jkg, 3000.0) &&
        same_campaign_number(sounding.tropopause_m, 12000.0) &&
        same_campaign_number(sounding.surface_theta_k, 301.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        same_campaign_number(*sounding.moisture_qv_kgkg, 0.012) &&
        hodograph.type == HodographType::WKParam &&
        same_campaign_number(hodograph.u_sfc_ms, 4.0) &&
        same_campaign_number(hodograph.v_sfc_ms, 1.0) &&
        same_campaign_number(hodograph.u_1km_ms, 14.0) &&
        same_campaign_number(hodograph.v_1km_ms, 6.0) &&
        same_campaign_number(hodograph.u_6km_ms, 28.0) &&
        same_campaign_number(hodograph.v_6km_ms, 22.0) &&
        config.initial_conditions.trigger.type == TriggerType::Bubble &&
        same_campaign_number(bubble.dtheta_k, 4.0) &&
        same_campaign_number(bubble.radius_km, 4.0) &&
        same_campaign_number(bubble.center_x_m, 26000.0) &&
        same_campaign_number(bubble.center_y_m, 20000.0) &&
        same_campaign_number(bubble.center_z_m, 1500.0) &&
        config.output.preset == OutputPreset::ScienceValidation &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::MovingReferenceV1 &&
        config.output.qualification_diagnostics_interval_s == 30 &&
        !config.output.trajectories &&
        config.extensions.solver.slow_rk2 &&
        !config.extensions.solver.axis_u_m1_filter &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        config.extensions.checkpoint.every_s == 450 &&
        config.extensions.checkpoint.directory.empty() &&
        config.extensions.checkpoint.resume_from.empty() &&
        config.extensions.validation.guard_mode ==
            ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps == 10 &&
        config.extensions.diagnostics.deterministic_reductions;
}

bool exact_moving_nest_device_ownership(const ConfigV1& config)
{
    const auto& compute = config.extensions.compute;
    return !compute.allow_fallback &&
        !compute.gpu_apply_microphysics &&
        !compute.gpu_apply_slow && compute.gpu_apply_sgs;
}

bool exact_moving_nest_vulkan_ownership(const ConfigV1& config)
{
    return config.extensions.compute.backend == ComputeBackend::GPU &&
        exact_moving_nest_device_ownership(config);
}

bool exact_moving_nest_metal_ownership(const ConfigV1& config)
{
    return config.extensions.compute.backend == ComputeBackend::Metal &&
        exact_moving_nest_device_ownership(config);
}

bool exact_moving_nest_cpu_ownership(const ConfigV1& config)
{
    return config.extensions.compute.backend == ComputeBackend::CPU;
}

bool valid_internal_moving_nest_campaign(const ConfigV1& config)
{
    return exact_moving_nest_science_profile(config) &&
        config.extensions.execution.mode == ExecutionMode::Exploratory &&
        (exact_moving_nest_cpu_ownership(config) ||
         exact_moving_nest_vulkan_ownership(config) ||
         exact_moving_nest_metal_ownership(config));
}

bool exact_qualified_two_way_moving_profile(const ConfigV1& config);

bool valid_public_qualified_moving_nest_profile(const ConfigV1& config)
{
    return exact_qualified_two_way_moving_profile(config) ||
        (config.domain_motion.mode == DomainMotionMode::MovingNest && is_public_qualified_accelerated_two_way_profile(config)) ||
        (exact_moving_nest_science_profile(config) &&
         (exact_moving_nest_cpu_ownership(config) || exact_moving_nest_vulkan_ownership(config) ||
          exact_moving_nest_metal_ownership(config)) &&
         config.extensions.execution.mode == ExecutionMode::Qualified);
}

bool exact_qualified_domain_recycling_profile(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    const auto& tracking = motion.tracking;
    const auto& compute = config.extensions.compute;
    const auto& micro = config.microphysics.parameters;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    const bool supported_backend =
        compute.backend == ComputeBackend::CPU ||
        compute.backend == ComputeBackend::GPU ||
        compute.backend == ComputeBackend::Metal;
    const bool supported_motion =
        (motion.mode == DomainMotionMode::ConstantVelocity &&
         same_campaign_number(motion.frame_velocity.u_ms, 5.0) &&
         same_campaign_number(motion.frame_velocity.v_ms, 0.0)) ||
        (motion.mode == DomainMotionMode::TrackedWindow &&
         same_campaign_number(motion.frame_velocity.u_ms, 0.0) &&
         same_campaign_number(motion.frame_velocity.v_ms, 0.0));
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        supported_backend && !compute.allow_fallback &&
        !compute.gpu_apply_microphysics && !compute.gpu_apply_slow &&
        !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 18 && config.grid.ny == 18 && config.grid.nz == 24 &&
        same_campaign_number(config.grid.dx, 1000.0) &&
        same_campaign_number(config.grid.dy, 1000.0) &&
        same_campaign_number(config.grid.dz, 500.0) &&
        same_campaign_number(config.grid.ztop, 12000.0) &&
        same_campaign_number(config.time.run_time, 120.0) &&
        same_campaign_number(config.time.dt_large, 0.5) &&
        config.time.acoustic_substeps == 8 && !config.time.adaptive_dt &&
        same_campaign_number(config.time.cfl_safety, 0.7) &&
        same_campaign_number(config.time.output_interval, 0.0) &&
        supported_motion &&
        !motion.nesting.enabled && tracking.enabled && tracking.recycle_domain &&
        same_campaign_number(tracking.interval_s, 1.0) &&
        same_campaign_number(tracking.inner_margin_m, 6000.0) &&
        tracking.maximum_shift_cells == 6 &&
        same_campaign_number(tracking.cooldown_s, 2.0) &&
        same_campaign_number(tracking.minimum_confidence, 0.01) &&
        same_campaign_number(tracking.minimum_updraft_ms, 0.01) &&
        same_campaign_number(tracking.minimum_condensate_path_kgm2, 1.0e-6) &&
        same_campaign_number(tracking.updraft_helicity_bottom_m_agl, 500.0) &&
        same_campaign_number(tracking.updraft_helicity_top_m_agl, 3000.0) &&
        same_campaign_number(tracking.minimum_absolute_updraft_helicity_m2s2, 1.0e-6) &&
        tracking.minimum_component_cells == 1 &&
        same_campaign_number(tracking.minimum_core_area_m2, 1.0) &&
        same_campaign_number(tracking.maximum_motion_speed_ms, 1000.0) &&
        same_campaign_number(tracking.maximum_motion_acceleration_ms2, 1000.0) &&
        tracking.minimum_continuity_samples == 2 &&
        tracking.lost_after_samples == 3 &&
        tracking.topology_confirmation_samples == 1 &&
        tracking.branch_preference == StormBranchPreference::Continuity &&
        same_campaign_number(tracking.branch_velocity_uncertainty_ms, 5.0) &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        lateral.west == LateralBC::Radiative &&
        lateral.east == LateralBC::Radiative &&
        lateral.south == LateralBC::Radiative &&
        lateral.north == LateralBC::Radiative &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        config.damping.enabled && config.damping.mode == DampingMode::Explicit &&
        same_campaign_number(config.damping.z_d, 9000.0) &&
        same_campaign_number(config.damping.alpha, 0.01) &&
        config.damping.lateral_sponge_enabled &&
        config.damping.lateral_width == 20 &&
        same_campaign_number(config.damping.lateral_alpha, 0.05) &&
        config.damping.w_only_top_sponge &&
        config.damping.w_only_depth_cells == 8 &&
        same_campaign_number(config.damping.w_only_alpha_max, 0.2) &&
        !config.coriolis.enabled &&
        config.microphysics.scheme == MicrophysicsScheme::Kessler &&
        same_campaign_number(micro.shared.cloud_water_threshold_kgkg, 1.0e-3) &&
        same_campaign_number(micro.shared.autoconversion_rate_s_inv, 1.0e-3) &&
        same_campaign_number(micro.kessler.accretion_coefficient, 2.2) &&
        same_campaign_number(micro.kessler.evaporation_efficiency, 1.0) &&
        config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
        config.turbulence.mode == TurbulenceMode::TwoD &&
        same_campaign_number(config.turbulence.Cs, 0.18) &&
        same_campaign_number(config.turbulence.Pr_t, 1.0 / 3.0) &&
        same_campaign_number(config.turbulence.Sc_t, 1.0 / 3.0) &&
        config.advection.scheme == AdvectionScheme::TVD &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        config.diffusion.K_h == 0.0 && config.diffusion.K_v == 0.0 &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::Slab && config.surface.apply_fluxes &&
        same_campaign_number(config.surface.z0m_m, 0.1) &&
        same_campaign_number(config.surface.z0h_m, 0.01) &&
        same_campaign_number(config.surface.albedo, 0.2) &&
        same_campaign_number(config.surface.emissivity, 0.96) &&
        same_campaign_number(config.surface.tsk_k, 301.0) &&
        same_campaign_number(config.surface.qsfc_kgkg, 0.014) &&
        config.boundary_layer.scheme == BoundaryLayerScheme::Slab &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        config.radiation.scheme == RadiationScheme::None &&
        config.terrain.scheme == TerrainScheme::None &&
        sounding.type == SoundingType::ParametricCAPE &&
        same_campaign_number(sounding.cape_jkg, 3000.0) &&
        same_campaign_number(sounding.tropopause_m, 10000.0) &&
        same_campaign_number(sounding.surface_theta_k, 301.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        same_campaign_number(*sounding.moisture_qv_kgkg, 0.014) &&
        hodograph.type == HodographType::WKParam &&
        same_campaign_number(hodograph.u_sfc_ms, 9.0) &&
        same_campaign_number(hodograph.v_sfc_ms, 1.0) &&
        same_campaign_number(hodograph.u_1km_ms, 14.0) &&
        same_campaign_number(hodograph.v_1km_ms, 6.0) &&
        same_campaign_number(hodograph.u_6km_ms, 28.0) &&
        same_campaign_number(hodograph.v_6km_ms, 22.0) &&
        config.initial_conditions.trigger.type == TriggerType::Bubble &&
        same_campaign_number(bubble.dtheta_k, 4.0) &&
        same_campaign_number(bubble.radius_km, 3.0) &&
        same_campaign_number(bubble.center_x_m, 13000.0) &&
        same_campaign_number(bubble.center_y_m, 9000.0) &&
        same_campaign_number(bubble.center_z_m, 1500.0) &&
        config.extensions.chaos.scheme == ChaosScheme::FullStochastic &&
        config.extensions.chaos.seed == 233 &&
        config.extensions.chaos.member_id == 0 &&
        config.output.preset == OutputPreset::Minimal &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::None &&
        !config.output.trajectories &&
        config.extensions.solver.slow_rk2 &&
        same_campaign_number(config.extensions.solver.lid_bc_gamma, -1.0) &&
        !config.extensions.solver.axis_u_m1_filter &&
        !config.extensions.checkpoint.every_s &&
        config.extensions.checkpoint.directory.empty() &&
        config.extensions.checkpoint.resume_from.empty() &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions;
}

bool exact_qualified_mapped_nested_recycling_profile(const ConfigV1& config);

bool exact_qualified_nested_grid_recycling_profile(const ConfigV1& config)
{
    if (exact_qualified_mapped_nested_recycling_profile(config))
    {
        return true;
    }
    const auto& nesting                          = config.domain_motion.nesting;
    ConfigV1    parent_profile                   = config;
    parent_profile.domain_motion.nesting.enabled = false;
    if (nesting.feedback == NestingFeedback::TwoWay)
    {
        if (!same_campaign_number(config.domain_motion.tracking.inner_margin_m, 4000.0) ||
            config.output.preset != OutputPreset::ScienceValidation)
            return false;
        parent_profile.domain_motion.tracking.inner_margin_m = 6000.0;
        parent_profile.output.preset                         = OutputPreset::Minimal;
    }
    return exact_qualified_domain_recycling_profile(parent_profile) && nesting.enabled &&
           (nesting.feedback == NestingFeedback::OneWay ||
            (nesting.feedback == NestingFeedback::TwoWay && config.extensions.compute.backend == ComputeBackend::CPU &&
             config.domain_motion.mode == DomainMotionMode::TrackedWindow)) &&
           nesting.parent_ratio_x == 2 && nesting.parent_ratio_y == 2 && nesting.parent_time_step_ratio == 2 &&
           nesting.child_origin_parent_cell_i == 11 && nesting.child_origin_parent_cell_j == 5 &&
           nesting.child_cells_x == 8 && nesting.child_cells_y == 8 && nesting.relaxation_width_child_cells == 1 &&
           nesting.boundary_stencil_parent_cells == 1 && same_campaign_number(nesting.relaxation_e_folding_time_s, 2.0);
}

bool exact_qualified_two_way_moving_profile(const ConfigV1& config)
{
    if (config.terrain.scheme != TerrainScheme::None)
    {
        const auto& terrain = config.terrain;
        const auto& nesting = config.domain_motion.nesting;
        const int   ratio   = nesting.parent_ratio_x;
        const bool  shape   = (ratio == 2 || ratio == 3) && nesting.parent_ratio_y == ratio &&
                              nesting.parent_time_step_ratio == ratio && nesting.child_cells_x == 10 * ratio &&
                              nesting.child_cells_y == 10 * ratio;
        const bool  surface =
            terrain.coordinate == TerrainCoordinate::Smoothed &&
            terrain.horizontal_placement == TerrainHorizontalPlacement::GridAttached && terrain.compute_derivatives &&
            terrain.compute_metrics && same_campaign_number(terrain.smoothing_decay, .1) &&
            same_campaign_number(terrain.feature_center_earth_x_m, 0) &&
            same_campaign_number(terrain.feature_center_earth_y_m, 0) &&
            ((terrain.scheme == TerrainScheme::Bell && same_campaign_number(terrain.bell.h0, 20) &&
              same_campaign_number(terrain.bell.a, 1600) && terrain.bell.axisymmetric) ||
             (terrain.scheme == TerrainScheme::Schar && same_campaign_number(terrain.schar.H, 20) &&
              same_campaign_number(terrain.schar.a, 1600) && same_campaign_number(terrain.schar.ell, 2400)));
        if (!shape || !surface || config.turbulence.scheme != TurbulenceScheme::None ||
            config.boundary_layer.scheme != BoundaryLayerScheme::None || config.surface.scheme != SurfaceScheme::None ||
            config.surface.apply_fluxes || config.extensions.chaos.scheme != ChaosScheme::None ||
            (config.microphysics.scheme != MicrophysicsScheme::Kessler &&
             config.microphysics.scheme != MicrophysicsScheme::Lin))
            return false;
        if (!same_campaign_number(config.initial_conditions.trigger.bubble.center_x_m, 12000))
            return false;
        if (config.microphysics.scheme == MicrophysicsScheme::Lin &&
            (config.output.preset != OutputPreset::DynamicsFull ||
             !same_campaign_number(config.microphysics.parameters.lin.accretion_coefficient, 2.2) ||
             !same_campaign_number(config.microphysics.parameters.lin.evaporation_rate_s_inv, .001)))
            return false;
        auto reference                                         = config;
        reference.terrain.scheme                               = TerrainScheme::None;
        reference.microphysics.scheme                          = MicrophysicsScheme::Kessler;
        reference.output.preset                                = OutputPreset::ScienceValidation;
        reference.turbulence.scheme                            = TurbulenceScheme::Smagorinsky;
        reference.boundary_layer.scheme                        = BoundaryLayerScheme::Slab;
        reference.surface.scheme                               = SurfaceScheme::Slab;
        reference.surface.apply_fluxes                         = true;
        reference.extensions.chaos.scheme                      = ChaosScheme::FullStochastic;
        reference.domain_motion.nesting.parent_ratio_x         = 2;
        reference.domain_motion.nesting.parent_ratio_y         = 2;
        reference.domain_motion.nesting.parent_time_step_ratio = 2;
        reference.domain_motion.nesting.child_cells_x          = 20;
        reference.domain_motion.nesting.child_cells_y          = 20;
        reference.initial_conditions.trigger.bubble.center_x_m = 13000;
        return exact_qualified_two_way_moving_profile(reference);
    }

    const auto& motion  = config.domain_motion;
    const auto& nesting = motion.nesting;
    const auto& bubble  = config.initial_conditions.trigger.bubble;
    if (motion.mode != DomainMotionMode::MovingNest || motion.tracking.recycle_domain ||
        nesting.feedback != NestingFeedback::TwoWay || config.extensions.compute.backend != ComputeBackend::CPU ||
        config.grid.nx != 32 || config.grid.ny != 26 || nesting.child_cells_x != 20 || nesting.child_cells_y != 20 ||
        nesting.child_origin_parent_cell_i != 5 || nesting.child_origin_parent_cell_j != 6 ||
        !same_campaign_number(motion.tracking.inner_margin_m, 2000) ||
        !same_campaign_number(motion.tracking.minimum_updraft_ms, 1.0) ||
        !same_campaign_number(motion.tracking.minimum_condensate_path_kgm2, .1) ||
        config.initial_conditions.sounding.type != SoundingType::File ||
        config.initial_conditions.sounding.file_path !=
            "tests/fixtures/configs/qualification/soundings/moving_two_way_thermal.txt" ||
        !same_campaign_number(bubble.dtheta_k, 8) || !same_campaign_number(bubble.cloud_water_kgkg, .001) ||
        !same_campaign_number(bubble.radius_km, 3) || !same_campaign_number(bubble.center_x_m, 13000) ||
        !same_campaign_number(bubble.center_y_m, 12000))
        return false;
    auto reference    = config;
    reference.grid.nx = reference.grid.ny                         = 18;
    reference.domain_motion.mode                                  = DomainMotionMode::TrackedWindow;
    reference.domain_motion.tracking.recycle_domain               = true;
    reference.domain_motion.tracking.inner_margin_m               = 4000;
    reference.domain_motion.tracking.minimum_updraft_ms           = .01;
    reference.domain_motion.tracking.minimum_condensate_path_kgm2 = 1e-6;
    reference.domain_motion.nesting.child_cells_x = reference.domain_motion.nesting.child_cells_y = 8;
    reference.domain_motion.nesting.child_origin_parent_cell_i                                    = 11;
    reference.domain_motion.nesting.child_origin_parent_cell_j                                    = 5;
    reference.initial_conditions.sounding.type                   = SoundingType::ParametricCAPE;
    reference.initial_conditions.sounding.cape_jkg               = 3000;
    reference.initial_conditions.sounding.tropopause_m           = 10000;
    reference.initial_conditions.sounding.surface_theta_k        = 301;
    reference.initial_conditions.sounding.moisture_qv_kgkg       = .014;
    reference.initial_conditions.trigger.bubble.dtheta_k         = 4;
    reference.initial_conditions.trigger.bubble.cloud_water_kgkg = 0;
    reference.initial_conditions.trigger.bubble.radius_km        = 3;
    reference.initial_conditions.trigger.bubble.center_y_m       = 9000;
    return exact_qualified_nested_grid_recycling_profile(reference);
}

bool exact_qualified_earth_fixed_terrain_recycling_profile(const ConfigV1& config)
{
    const auto& motion    = config.domain_motion;
    const auto& tracking  = motion.tracking;
    const auto& compute   = config.extensions.compute;
    const auto& micro = config.microphysics.parameters;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& radiation = config.radiation;
    const auto& terrain = config.terrain;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& bubble = config.initial_conditions.trigger.bubble;
    return config.extensions.execution.mode == ExecutionMode::Qualified && compute.backend == ComputeBackend::CPU &&
           !compute.allow_fallback && !compute.validate_parity && !compute.gpu_apply_microphysics &&
           !compute.gpu_apply_slow && !compute.gpu_apply_sgs &&
           config.grid.coordinate_system == CoordinateSystem::Cartesian &&
           config.grid.staggering == StaggerType::CGrid && config.grid.nx == 18 && config.grid.ny == 18 &&
           config.grid.nz == 18 && same_campaign_number(config.grid.dx, 500.0) &&
           same_campaign_number(config.grid.dy, 500.0) && same_campaign_number(config.grid.dz, 500.0) &&
           same_campaign_number(config.grid.ztop, 9000.0) && config.time.integrator == TimeIntegrator::SplitExplicit &&
           same_campaign_number(config.time.run_time, 65.0) && same_campaign_number(config.time.dt_large, 0.2) &&
           config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
           same_campaign_number(config.time.cfl_safety, 0.7) &&
           same_campaign_number(config.time.output_interval, 65.0) && motion.mode == DomainMotionMode::TrackedWindow &&
           same_campaign_number(motion.frame_velocity.u_ms, 0.0) &&
           same_campaign_number(motion.frame_velocity.v_ms, 0.0) && !motion.nesting.enabled && tracking.enabled &&
           tracking.recycle_domain && same_campaign_number(tracking.interval_s, 1.0) &&
           same_campaign_number(tracking.inner_margin_m, 250.0) && tracking.maximum_shift_cells == 8 &&
           same_campaign_number(tracking.cooldown_s, 65.0) && same_campaign_number(tracking.minimum_confidence, 0.01) &&
           same_campaign_number(tracking.minimum_updraft_ms, 0.01) &&
           same_campaign_number(tracking.minimum_condensate_path_kgm2, 0.06) &&
           same_campaign_number(tracking.updraft_helicity_bottom_m_agl, 500.0) &&
           same_campaign_number(tracking.updraft_helicity_top_m_agl, 3000.0) &&
           same_campaign_number(tracking.minimum_absolute_updraft_helicity_m2s2, 1.0e-9) &&
           tracking.minimum_component_cells == 1 && same_campaign_number(tracking.minimum_core_area_m2, 1.0) &&
           same_campaign_number(tracking.maximum_motion_speed_ms, 1000.0) &&
           same_campaign_number(tracking.maximum_motion_acceleration_ms2, 1000.0) &&
           tracking.minimum_continuity_samples == 2 && tracking.lost_after_samples == 3 &&
           tracking.topology_confirmation_samples == 1 &&
           tracking.branch_preference == StormBranchPreference::Continuity &&
           same_campaign_number(tracking.branch_velocity_uncertainty_ms, 5.0) &&
           config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" && !config.dynamics.axisymmetric &&
           lateral.west == LateralBC::Radiative && lateral.east == LateralBC::Radiative &&
           lateral.south == LateralBC::Radiative && lateral.north == LateralBC::Radiative &&
           config.boundary_conditions.bottom == BottomBC::RigidSlip && config.boundary_conditions.top == TopBC::Rigid &&
           !config.damping.enabled && config.damping.mode == DampingMode::Explicit &&
           same_campaign_number(config.damping.z_d, 9000.0) && same_campaign_number(config.damping.alpha, 0.0) &&
           !config.damping.lateral_sponge_enabled && config.damping.lateral_width == 0 &&
           same_campaign_number(config.damping.lateral_alpha, 0.0) && !config.damping.w_only_top_sponge &&
           config.damping.w_only_depth_cells == 0 && same_campaign_number(config.damping.w_only_alpha_max, 0.0) &&
           !config.coriolis.enabled && config.microphysics.scheme == MicrophysicsScheme::Kessler &&
           same_campaign_number(micro.shared.cloud_water_threshold_kgkg, 1.0e-3) &&
           same_campaign_number(micro.shared.autoconversion_rate_s_inv, 1.0e-3) &&
           same_campaign_number(micro.kessler.accretion_coefficient, 2.2) &&
           same_campaign_number(micro.kessler.evaporation_efficiency, 1.0) &&
           config.turbulence.scheme == TurbulenceScheme::None && config.turbulence.mode == TurbulenceMode::ThreeD &&
           same_campaign_number(config.turbulence.Cs, 0.18) && same_campaign_number(config.turbulence.Pr_t, 0.7) &&
           same_campaign_number(config.turbulence.Sc_t, 0.7) && config.advection.scheme == AdvectionScheme::TVD &&
           config.diffusion.scheme == DiffusionScheme::Explicit && config.diffusion.operator_order == 2 &&
           same_campaign_number(config.diffusion.K_h, 0.0) && same_campaign_number(config.diffusion.K_v, 0.0) &&
           config.diffusion.apply_to == DiffusionTarget::Scalars && config.surface.scheme == SurfaceScheme::Slab &&
           config.surface.apply_fluxes && same_campaign_number(config.surface.z0m_m, 0.1) &&
           same_campaign_number(config.surface.z0h_m, 0.01) && same_campaign_number(config.surface.albedo, 0.2) &&
           same_campaign_number(config.surface.emissivity, 0.96) && same_campaign_number(config.surface.tsk_k, 300.0) &&
           same_campaign_number(config.surface.qsfc_kgkg, 0.012) &&
           config.boundary_layer.scheme == BoundaryLayerScheme::Slab &&
           same_campaign_number(config.boundary_layer.cadence_s, 60.0) && config.boundary_layer.clear_sky_only &&
           config.boundary_layer.pbl_top_method == BoundaryLayerTopMethod::Richardson &&
           radiation.scheme == RadiationScheme::SimpleGrey &&
           radiation.solar_forcing == RadiationSolarForcing::IdealizedDiurnal &&
           same_campaign_number(radiation.latitude_deg, 0.0) && same_campaign_number(radiation.declination_deg, 0.0) &&
           same_campaign_number(radiation.local_solar_time_at_start_h, 12.0) && radiation.do_lw && !radiation.do_sw &&
           same_campaign_number(radiation.cadence_s, 2.0) && radiation.clear_sky_only &&
           radiation.simple_grey_optical_model == "pressure_clear_sky_v1" && !radiation.use_subcycling &&
           radiation.subcycle_count == 1 && !radiation.enable_diagnostics &&
           same_campaign_number(radiation.tau_lw_ref, 2.5) && same_campaign_number(radiation.tau_sw_ref, 0.15) &&
           same_campaign_number(radiation.n_lw, 1.0) && same_campaign_number(radiation.n_sw, 1.0) &&
           terrain.scheme == TerrainScheme::Bell && terrain.coordinate == TerrainCoordinate::Smoothed &&
           terrain.horizontal_placement == TerrainHorizontalPlacement::EarthFixed && terrain.compute_derivatives &&
           terrain.compute_metrics && same_campaign_number(terrain.smoothing_decay, 0.1) &&
           same_campaign_number(terrain.feature_center_earth_x_m, 4500.0) &&
           same_campaign_number(terrain.feature_center_earth_y_m, 4500.0) &&
           same_campaign_number(terrain.bell.h0, 300.0) && same_campaign_number(terrain.bell.a, 1800.0) &&
           terrain.bell.axisymmetric && sounding.type == SoundingType::ParametricCAPE &&
           same_campaign_number(sounding.cape_jkg, 2500.0) && same_campaign_number(sounding.tropopause_m, 8000.0) &&
           same_campaign_number(sounding.surface_theta_k, 300.0) && sounding.moisture_qv_kgkg.has_value() &&
           same_campaign_number(*sounding.moisture_qv_kgkg, 0.012) && hodograph.type == HodographType::WKParam &&
           same_campaign_number(hodograph.u_sfc_ms, 5.0) && same_campaign_number(hodograph.v_sfc_ms, 0.0) &&
           same_campaign_number(hodograph.u_1km_ms, 10.0) && same_campaign_number(hodograph.v_1km_ms, 5.0) &&
           same_campaign_number(hodograph.u_6km_ms, 25.0) && same_campaign_number(hodograph.v_6km_ms, 15.0) &&
           config.initial_conditions.trigger.type == TriggerType::Bubble &&
           same_campaign_number(bubble.dtheta_k, 4.0) && same_campaign_number(bubble.radius_km, 2.0) &&
           same_campaign_number(bubble.center_x_m, 7000.0) && same_campaign_number(bubble.center_y_m, 4500.0) &&
           same_campaign_number(bubble.center_z_m, 1500.0) && config.output.preset == OutputPreset::Minimal &&
           config.output.format == OutputFormat::Npy3D && !config.output.async_io && !config.output.live_shm &&
           config.output.qualification_diagnostics_preset == QualificationDiagnosticsPreset::None &&
           !config.output.trajectories && config.extensions.solver.slow_rk2 &&
           same_campaign_number(config.extensions.solver.lid_bc_gamma, -1.0) &&
           !config.extensions.solver.axis_u_m1_filter &&
           config.extensions.chaos.scheme == ChaosScheme::FullStochastic && config.extensions.chaos.seed == 233 &&
           config.extensions.chaos.member_id == 0 && !config.extensions.chaos.native_generation_required &&
           config.extensions.chaos.apply_to_ic.empty() && config.extensions.chaos.apply_to_tendencies.empty() &&
           config.extensions.chaos.sigma_ic ==
               std::unordered_map<std::string, double>{
                   {{"u", 0.5}, {"v", 0.5}, {"w", 0.1}, {"theta", 0.5}, {"qv", 0.0005}, {"qc", 0.0}, {"qr", 0.0}}} &&
           config.extensions.chaos.alpha_tend ==
               std::unordered_map<std::string, double>{{{"microphysics", 0.3}, {"pbl", 0.2}, {"diffusion", 0.1}}} &&
           !config.radar.enabled && config.radar.products.empty() && config.extensions.checkpoint.every_s == 0 &&
           config.extensions.checkpoint.keep_last == 0 && config.extensions.checkpoint.minimum_free_space_mib == 0 &&
           config.extensions.checkpoint.directory.empty() && config.extensions.checkpoint.resume_from.empty() &&
           config.extensions.validation.guard_mode == ValidationGuardMode::Error &&
           config.extensions.diagnostics.scientific_every_steps == 1 &&
           config.extensions.diagnostics.deterministic_reductions;
}

bool exact_qualified_mapped_nested_recycling_profile(const ConfigV1& config)
{
    const auto& nesting = config.domain_motion.nesting;
    const auto& compute = config.extensions.compute;
    const bool  device  = compute.backend == ComputeBackend::Metal || compute.backend == ComputeBackend::GPU;
    if ((!device && compute.backend != ComputeBackend::CPU) || !nesting.enabled ||
        nesting.feedback != NestingFeedback::TwoWay || (nesting.parent_ratio_x != 2 && nesting.parent_ratio_x != 3) ||
        nesting.parent_ratio_y != nesting.parent_ratio_x || nesting.parent_time_step_ratio != nesting.parent_ratio_x ||
        nesting.child_origin_parent_cell_i != 11 || nesting.child_origin_parent_cell_j != 5 ||
        nesting.child_cells_x != (nesting.parent_ratio_x == 2 ? 8 : 12) ||
        nesting.child_cells_y != nesting.child_cells_x || nesting.relaxation_width_child_cells != 1 ||
        nesting.boundary_stencil_parent_cells != 1 || !same_campaign_number(nesting.relaxation_e_folding_time_s, 2.0) ||
        compute.gpu_apply_microphysics || (!device && compute.gpu_apply_slow) || compute.gpu_apply_sgs ||
        config.extensions.chaos.native_generation_required != device)
    {
        return false;
    }
    if (config.terrain.scheme != TerrainScheme::Bell && config.terrain.scheme != TerrainScheme::Schar)
        return false;
    if (config.terrain.scheme == TerrainScheme::Schar &&
        (!same_campaign_number(config.terrain.schar.H, 300) || !same_campaign_number(config.terrain.schar.a, 1800) ||
         !same_campaign_number(config.terrain.schar.ell, 2400)))
        return false;
    auto reference                                        = config;
    reference.domain_motion.nesting.enabled               = false;
    reference.extensions.compute.backend                  = ComputeBackend::CPU;
    reference.extensions.chaos.native_generation_required = false;
    reference.extensions.compute.gpu_apply_slow           = false;
    if (config.terrain.scheme == TerrainScheme::Schar)
    {
        reference.terrain.scheme            = TerrainScheme::Bell;
        reference.terrain.bell.h0           = 300;
        reference.terrain.bell.a            = 1800;
        reference.terrain.bell.axisymmetric = true;
    }
    return exact_qualified_earth_fixed_terrain_recycling_profile(reference);
}

bool exact_qualified_earth_fixed_terrain_motion_profile(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    if (motion.mode != DomainMotionMode::ConstantVelocity || motion.tracking.enabled ||
        motion.tracking.recycle_domain || !std::isfinite(motion.frame_velocity.u_ms) ||
        !std::isfinite(motion.frame_velocity.v_ms) || std::abs(motion.frame_velocity.u_ms) > 180.0 ||
        std::abs(motion.frame_velocity.v_ms) > 180.0 ||
        config.terrain.horizontal_placement != TerrainHorizontalPlacement::EarthFixed)
        return false;
    ConfigV1 reference                     = config;
    reference.radiation.enable_diagnostics = false;
    if (config.microphysics.scheme == MicrophysicsScheme::Lin)
    {
        const auto& lin = config.microphysics.parameters.lin;
        if (!same_campaign_number(lin.accretion_coefficient, 2.2) ||
            !same_campaign_number(lin.evaporation_rate_s_inv, 1.0e-3))
            return false;
        reference.microphysics.scheme = MicrophysicsScheme::Kessler;
    }
    const auto& compute = config.extensions.compute;
    if (compute.backend == ComputeBackend::GPU || compute.backend == ComputeBackend::Metal)
    {
        if (compute.allow_fallback || compute.validate_parity || !compute.gpu_apply_slow ||
            compute.gpu_apply_microphysics || compute.gpu_apply_sgs ||
            !config.extensions.chaos.native_generation_required)
            return false;
        reference.extensions.compute.backend                  = ComputeBackend::CPU;
        reference.extensions.compute.gpu_apply_slow           = false;
        reference.extensions.chaos.native_generation_required = false;
    }
    reference.domain_motion.mode                    = DomainMotionMode::TrackedWindow;
    reference.domain_motion.frame_velocity          = {0.0, 0.0};
    reference.domain_motion.tracking.enabled        = true;
    reference.domain_motion.tracking.recycle_domain = true;
    return exact_qualified_earth_fixed_terrain_recycling_profile(reference);
}

bool exact_qualified_polarimetric_radar_profile(const ConfigV1& config);

bool exact_qualified_stationary_nested_profile(const ConfigV1& config)
{
    const auto& motion    = config.domain_motion;
    const auto& nesting   = motion.nesting;
    const auto& lateral   = config.boundary_conditions.lateral;
    const auto& sounding  = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& vortex    = config.initial_conditions.trigger.vortex_seed;
    const auto& compute   = config.extensions.compute;
    const auto& terrain   = config.terrain;
    const bool  exact_bell =
        terrain.scheme == TerrainScheme::Bell && terrain.coordinate == TerrainCoordinate::Smoothed &&
        terrain.horizontal_placement == TerrainHorizontalPlacement::GridAttached && terrain.compute_derivatives &&
        terrain.compute_metrics && same_campaign_number(terrain.smoothing_decay, 0.1) &&
        same_campaign_number(terrain.feature_center_earth_x_m, 0.0) &&
        same_campaign_number(terrain.feature_center_earth_y_m, 0.0) && same_campaign_number(terrain.bell.h0, 20.0) &&
        same_campaign_number(terrain.bell.a, 1600.0) && terrain.bell.axisymmetric;
    const bool exact_schar =
        terrain.scheme == TerrainScheme::Schar
            && terrain.coordinate == TerrainCoordinate::Smoothed
            && terrain.horizontal_placement
                == TerrainHorizontalPlacement::GridAttached
            && terrain.compute_derivatives && terrain.compute_metrics
            && same_campaign_number(terrain.smoothing_decay, 0.1)
            && same_campaign_number(terrain.feature_center_earth_x_m, 0.0)
            && same_campaign_number(terrain.feature_center_earth_y_m, 0.0)
            && same_campaign_number(terrain.schar.H, 20.0)
            && same_campaign_number(terrain.schar.a, 1600.0)
            && same_campaign_number(terrain.schar.ell, 800.0);
    const bool exact_active_terrain = exact_bell || exact_schar;
    const bool exact_terrain = terrain.scheme == TerrainScheme::None
        || exact_active_terrain;
    const bool exact_two_to_one =
        nesting.parent_ratio_x == 2 && nesting.parent_ratio_y == 2 &&
        nesting.parent_time_step_ratio == 2 &&
        nesting.child_cells_x == 8 && nesting.child_cells_y == 8;
    const bool exact_three_to_one = exact_active_terrain &&
        nesting.parent_ratio_x == 3 && nesting.parent_ratio_y == 3 &&
        nesting.parent_time_step_ratio == 3 &&
        nesting.child_cells_x == 9 && nesting.child_cells_y == 9;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        compute.backend == ComputeBackend::CPU && !compute.allow_fallback &&
        !compute.gpu_apply_microphysics && !compute.gpu_apply_slow &&
        !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 16 && config.grid.ny == 16 && config.grid.nz == 24 &&
        same_campaign_number(config.grid.dx, 1000.0) &&
        same_campaign_number(config.grid.dy, 1000.0) &&
        same_campaign_number(config.grid.dz, 500.0) &&
        same_campaign_number(config.grid.ztop, 12000.0) &&
        same_campaign_number(
            config.time.run_time, exact_active_terrain ? 2.0 : 4.0) &&
        same_campaign_number(config.time.dt_large, 0.5) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        same_campaign_number(config.time.cfl_safety, 0.7) &&
        same_campaign_number(
            config.time.output_interval,
            1.0) &&
        motion.mode == DomainMotionMode::Stationary &&
        same_campaign_number(motion.frame_velocity.u_ms, 0.0) &&
        same_campaign_number(motion.frame_velocity.v_ms, 0.0) &&
        !motion.lifetime_planning.enabled && !motion.tracking.enabled &&
        !motion.tracking.recycle_domain && nesting.enabled &&
        (nesting.feedback == NestingFeedback::TwoWay ||
         (exact_active_terrain &&
          nesting.feedback == NestingFeedback::OneWay)) &&
        (exact_two_to_one || exact_three_to_one) &&
        nesting.child_origin_parent_cell_i == 5 &&
        nesting.child_origin_parent_cell_j == 5 &&
        nesting.relaxation_width_child_cells == 1 &&
        nesting.boundary_stencil_parent_cells == 1 &&
        same_campaign_number(nesting.relaxation_e_folding_time_s, 2.0) &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        lateral.west == LateralBC::ZeroGradient &&
        lateral.east == LateralBC::ZeroGradient &&
        lateral.south == LateralBC::ZeroGradient &&
        lateral.north == LateralBC::ZeroGradient &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled && !config.damping.lateral_sponge_enabled &&
        !config.damping.w_only_top_sponge && !config.coriolis.enabled &&
        config.microphysics.scheme == MicrophysicsScheme::None &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.advection.scheme == AdvectionScheme::TVD &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        same_campaign_number(config.diffusion.K_h, 0.0) &&
        same_campaign_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::None &&
        !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::None &&
        config.radiation.scheme == RadiationScheme::None &&
        exact_terrain &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        sounding.type == SoundingType::ParametricCAPE &&
        same_campaign_number(sounding.cape_jkg, 3275.016) &&
        same_campaign_number(sounding.tropopause_m, 8000.0) &&
        same_campaign_number(sounding.surface_theta_k, 300.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        same_campaign_number(*sounding.moisture_qv_kgkg, 0.0) &&
        hodograph.type == HodographType::WKParam &&
        same_campaign_number(hodograph.u_sfc_ms, 4.0) &&
        same_campaign_number(hodograph.v_sfc_ms, 1.0) &&
        same_campaign_number(hodograph.u_1km_ms, 4.0) &&
        same_campaign_number(hodograph.v_1km_ms, 1.0) &&
        same_campaign_number(hodograph.u_6km_ms, 4.0) &&
        same_campaign_number(hodograph.v_6km_ms, 1.0) &&
        config.initial_conditions.trigger.type == TriggerType::VortexSeed &&
        same_campaign_number(vortex.r_max_m, 2000.0) &&
        same_campaign_number(vortex.v_max_ms, 1.0) &&
        same_campaign_number(vortex.z_top_m, 4000.0) &&
        same_campaign_number(vortex.center_x_m, 8000.0) &&
        same_campaign_number(vortex.center_y_m, 8000.0) &&
        same_campaign_number(vortex.center_z_m, 2000.0) &&
        config.output.preset == OutputPreset::ScienceValidation &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        !config.output.trajectories && config.extensions.solver.slow_rk2 &&
        !config.extensions.solver.axis_u_m1_filter &&
        config.extensions.checkpoint.every_s == 1 &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error;
}

bool valid_public_qualified_stationary_kessler_profile(
    const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    const auto& nesting = motion.nesting;
    const auto& compute = config.extensions.compute;
    const auto& micro = config.microphysics.parameters;
    const auto& lateral = config.boundary_conditions.lateral;
    const bool supported_backend =
        compute.backend == ComputeBackend::CPU
        || compute.backend == ComputeBackend::GPU;
    const bool canonical_kessler =
        same_campaign_number(
            micro.shared.cloud_water_threshold_kgkg, 1.0e-3)
        && same_campaign_number(
            micro.shared.autoconversion_rate_s_inv, 1.0e-3)
        && same_campaign_number(
            micro.kessler.accretion_coefficient, 2.2)
        && same_campaign_number(
            micro.kessler.evaporation_efficiency, 1.0);
    const bool canonical_smagorinsky =
        config.turbulence.scheme == TurbulenceScheme::Smagorinsky
        && config.turbulence.mode == TurbulenceMode::ThreeD
        && same_campaign_number(config.turbulence.Cs, 0.18)
        && same_campaign_number(config.turbulence.Pr_t, 1.0 / 3.0)
        && same_campaign_number(config.turbulence.Sc_t, 1.0 / 3.0);
    return config.extensions.execution.mode == ExecutionMode::Qualified
        && supported_backend && !compute.allow_fallback
        && compute.gpu_apply_microphysics && compute.gpu_apply_sgs
        && motion.mode == DomainMotionMode::Stationary
        && nesting.enabled
        && nesting.feedback == NestingFeedback::OneWay
        && nesting.parent_ratio_x == 2
        && nesting.parent_ratio_y == 2
        && nesting.parent_time_step_ratio == 2
        && !motion.tracking.enabled && !motion.tracking.recycle_domain
        && config.grid.coordinate_system == CoordinateSystem::Cartesian
        && config.grid.staggering == StaggerType::CGrid
        && config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid"
        && !config.dynamics.axisymmetric
        && lateral.west == LateralBC::ZeroGradient
        && lateral.east == LateralBC::ZeroGradient
        && lateral.south == LateralBC::ZeroGradient
        && lateral.north == LateralBC::ZeroGradient
        && config.boundary_conditions.bottom == BottomBC::RigidSlip
        && config.boundary_conditions.top == TopBC::Rigid
        && !config.damping.enabled
        && !config.damping.lateral_sponge_enabled
        && !config.damping.w_only_top_sponge
        && !config.coriolis.enabled
        && config.microphysics.scheme == MicrophysicsScheme::Kessler
        && canonical_kessler && canonical_smagorinsky
        && config.diffusion.scheme == DiffusionScheme::Explicit
        && config.diffusion.operator_order == 2
        && config.diffusion.K_h == 0.0
        && config.diffusion.K_v == 0.0
        && config.surface.scheme == SurfaceScheme::None
        && !config.surface.apply_fluxes
        && config.boundary_layer.scheme == BoundaryLayerScheme::None
        && config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson
        && config.radiation.scheme == RadiationScheme::None
        && config.terrain.scheme == TerrainScheme::None
        && config.extensions.chaos.scheme == ChaosScheme::None;
}

bool valid_public_qualified_kessler_profile(const ConfigV1& config)
{
    return valid_public_qualified_stationary_kessler_profile(config)
        || valid_public_qualified_moving_nest_profile(config)
        || exact_qualified_domain_recycling_profile(config)
        || exact_qualified_nested_grid_recycling_profile(config)
        || exact_qualified_earth_fixed_terrain_recycling_profile(config)
        || exact_qualified_earth_fixed_terrain_motion_profile(config)
        || exact_qualified_polarimetric_radar_profile(config)
        || is_public_qualified_mapped_terrain_profile(config)
        || tmv::runtime::is_public_ready_flat_agrid_cpu_family(config);
}

bool exact_qualified_rrtmg_environment_profile(
    const ConfigV1& config,
    bool clear_sky_only,
    const char* sounding_path)
{
    const auto& compute = config.extensions.compute;
    const auto& micro = config.microphysics.parameters;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& radiation = config.radiation;
    const auto& lateral = config.boundary_conditions.lateral;
    const bool cpu_ownership =
        compute.backend == ComputeBackend::CPU &&
        !compute.gpu_apply_microphysics;
    const bool accelerator_ownership =
        (compute.backend == ComputeBackend::GPU ||
         compute.backend == ComputeBackend::Metal) &&
        compute.gpu_apply_microphysics;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        (cpu_ownership || accelerator_ownership) &&
        !compute.allow_fallback && !compute.gpu_apply_slow &&
        !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 12 && config.grid.ny == 12 && config.grid.nz == 48 &&
        same_campaign_number(config.grid.dx, 1000.0) &&
        same_campaign_number(config.grid.dy, 1000.0) &&
        same_campaign_number(config.grid.dz, 250.0) &&
        same_campaign_number(config.grid.ztop, 12000.0) &&
        same_campaign_number(config.time.run_time, 4.0) &&
        same_campaign_number(config.time.dt_large, 0.2) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        same_campaign_number(config.time.cfl_safety, 0.7) &&
        same_campaign_number(config.time.output_interval, 4.0) &&
        config.domain_motion.mode == DomainMotionMode::Stationary &&
        !config.domain_motion.lifetime_planning.enabled &&
        !config.domain_motion.tracking.enabled &&
        !config.domain_motion.tracking.recycle_domain &&
        !config.domain_motion.nesting.enabled &&
        same_campaign_number(config.domain_motion.frame_velocity.u_ms, 0.0) &&
        same_campaign_number(config.domain_motion.frame_velocity.v_ms, 0.0) &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.advection.scheme == AdvectionScheme::TVD &&
        lateral.west == LateralBC::Periodic &&
        lateral.east == LateralBC::Periodic &&
        lateral.south == LateralBC::Periodic &&
        lateral.north == LateralBC::Periodic &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled && config.damping.mode == DampingMode::Explicit &&
        same_campaign_number(config.damping.z_d, 12000.0) &&
        same_campaign_number(config.damping.alpha, 0.0) &&
        !config.damping.lateral_sponge_enabled &&
        config.damping.lateral_width == 0 &&
        same_campaign_number(config.damping.lateral_alpha, 0.0) &&
        !config.damping.w_only_top_sponge &&
        config.damping.w_only_depth_cells == 0 &&
        same_campaign_number(config.damping.w_only_alpha_max, 0.0) &&
        !config.coriolis.enabled &&
        same_campaign_number(micro.shared.cloud_water_threshold_kgkg, 1.0e-3) &&
        same_campaign_number(micro.shared.autoconversion_rate_s_inv, 1.0e-3) &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.turbulence.mode == TurbulenceMode::ThreeD &&
        same_campaign_number(config.turbulence.Cs, 0.18) &&
        same_campaign_number(config.turbulence.Pr_t, 0.7) &&
        same_campaign_number(config.turbulence.Sc_t, 0.7) &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        same_campaign_number(config.diffusion.K_h, 0.0) &&
        same_campaign_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::Slab &&
        config.surface.apply_fluxes &&
        same_campaign_number(config.surface.z0m_m, 0.1) &&
        same_campaign_number(config.surface.z0h_m, 0.01) &&
        same_campaign_number(config.surface.albedo, 0.2) &&
        same_campaign_number(config.surface.emissivity, 0.96) &&
        same_campaign_number(config.surface.tsk_k, 300.0) &&
        same_campaign_number(config.surface.qsfc_kgkg, -1.0) &&
        config.boundary_layer.scheme == BoundaryLayerScheme::YSU &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        radiation.scheme == RadiationScheme::RRTMG &&
        radiation.solar_forcing == RadiationSolarForcing::IdealizedDiurnal &&
        same_campaign_number(radiation.latitude_deg, 35.0) &&
        same_campaign_number(radiation.declination_deg, 15.0) &&
        same_campaign_number(radiation.local_solar_time_at_start_h, 12.0) &&
        radiation.do_lw && radiation.do_sw &&
        same_campaign_number(radiation.cadence_s, 1.0) &&
        radiation.clear_sky_only == clear_sky_only &&
        same_campaign_number(radiation.rrtmg_ozone_vmr, 4.0e-6) &&
        same_campaign_number(radiation.rrtmg_co2_vmr, 420.0e-6) &&
        same_campaign_number(radiation.rrtmg_ch4_vmr, 1.9e-6) &&
        same_campaign_number(radiation.rrtmg_n2o_vmr, 0.335e-6) &&
        same_campaign_number(radiation.rrtmg_liquid_effective_radius_um, 10.0) &&
        same_campaign_number(radiation.rrtmg_ice_effective_radius_um, 30.0) &&
        config.terrain.scheme == TerrainScheme::None &&
        sounding.type == SoundingType::File &&
        sounding.file_path == sounding_path &&
        hodograph.type == HodographType::WKParam &&
        same_campaign_number(hodograph.u_sfc_ms, 5.0) &&
        same_campaign_number(hodograph.v_sfc_ms, 2.0) &&
        same_campaign_number(hodograph.u_1km_ms, 8.0) &&
        same_campaign_number(hodograph.v_1km_ms, 5.0) &&
        same_campaign_number(hodograph.u_6km_ms, 20.0) &&
        same_campaign_number(hodograph.v_6km_ms, 12.0) &&
        config.initial_conditions.trigger.type == TriggerType::None &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::None &&
        !config.output.trajectories &&
        config.extensions.solver.slow_rk2 &&
        same_campaign_number(config.extensions.solver.lid_bc_gamma, -1.0) &&
        !config.extensions.solver.axis_u_m1_filter &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        config.extensions.checkpoint.every_s == 2 &&
        config.extensions.checkpoint.directory.empty() &&
        config.extensions.checkpoint.resume_from.empty() &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions;
}

bool exact_qualified_lin_rrtmg_profile(const ConfigV1& config)
{
    const auto& lin = config.microphysics.parameters.lin;
    return exact_qualified_rrtmg_environment_profile(
               config,
               false,
               "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt") &&
        config.microphysics.scheme == MicrophysicsScheme::Lin &&
        config.time.integrator == TimeIntegrator::SplitExplicit &&
        same_campaign_number(lin.accretion_coefficient, 2.2) &&
        same_campaign_number(lin.evaporation_rate_s_inv, 1.0e-3) &&
        same_campaign_number(config.boundary_layer.cadence_s, 60.0) &&
        config.boundary_layer.clear_sky_only &&
        config.output.preset == OutputPreset::Minimal;
}

bool reference_thompson_parameters(const ConfigV1& config)
{
    const auto& parameters = config.microphysics.parameters;
    return same_campaign_number(parameters.shared.cloud_water_threshold_kgkg, 1.0e-3) &&
           same_campaign_number(parameters.shared.autoconversion_rate_s_inv, 1.0e-3) &&
           same_campaign_number(parameters.thompson.evaporation_rate_s_inv, 1.0e-3) &&
           same_campaign_number(parameters.thompson.water_friendly_aerosol_surface_m3, 350.0e6) &&
           same_campaign_number(parameters.thompson.ice_friendly_aerosol_surface_m3, 2.0e6);
}

bool reference_implicit_diffusion(const ConfigV1& config)
{
    const auto& diffusion = config.diffusion;
    return diffusion.scheme == DiffusionScheme::Implicit && config.time.integrator == TimeIntegrator::SplitExplicit &&
           diffusion.operator_order == 2 && diffusion.coefficient_source == DiffusionCoefficientSource::Constant &&
           diffusion.apply_to == DiffusionTarget::ScalarsAndMomentum && same_campaign_number(diffusion.K_h, 20.0) &&
           same_campaign_number(diffusion.K_v, 20.0);
}

bool exact_qualified_thompson_cpu_profile(const ConfigV1& config)
{
    if (config.microphysics.scheme != MicrophysicsScheme::Thompson)
        return false;
    ConfigV1 base = config;
    if (config.advection.scheme == AdvectionScheme::WENO5)
    {
        if (config.time.integrator != TimeIntegrator::SplitExplicit || !config.extensions.solver.slow_rk2)
            return false;
        base.advection.scheme = AdvectionScheme::TVD;
    }
    if (config.diffusion.scheme == DiffusionScheme::Implicit)
    {
        if (!reference_implicit_diffusion(config))
        {
            return false;
        }
        base.diffusion.scheme   = DiffusionScheme::Explicit;
        base.diffusion.apply_to = DiffusionTarget::Scalars;
        base.diffusion.K_h = base.diffusion.K_v = 0.0;
    }
    base.microphysics.scheme                                = MicrophysicsScheme::Lin;
    base.microphysics.parameters.lin.accretion_coefficient = 2.2;
    base.microphysics.parameters.lin.evaporation_rate_s_inv = 1.0e-3;
    base.radiation.clear_sky_only = false;
    base.output.preset = OutputPreset::Minimal;
    return exact_qualified_lin_rrtmg_profile(base) && config.extensions.compute.backend == ComputeBackend::CPU &&
           config.radiation.clear_sky_only && config.output.preset == OutputPreset::DynamicsFull &&
           reference_thompson_parameters(config);
}

bool exact_qualified_cloud_top_radiative_ysu_profile(
    const ConfigV1& config)
{
    ConfigV1 base = config;
    base.boundary_layer.cadence_s = 60.0;
    base.boundary_layer.clear_sky_only = true;
    base.radiation.local_solar_time_at_start_h = 12.0;
    base.surface.tsk_k = 300.0;
    base.initial_conditions.sounding.file_path =
        "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt";
    base.output.preset = OutputPreset::Minimal;
    return exact_qualified_lin_rrtmg_profile(base) &&
        config.extensions.compute.backend == ComputeBackend::CPU &&
        same_campaign_number(config.boundary_layer.cadence_s, 0.2) &&
        !config.boundary_layer.clear_sky_only &&
        same_campaign_number(
            config.radiation.local_solar_time_at_start_h, 0.0) &&
        same_campaign_number(config.surface.tsk_k, 285.0) &&
        config.initial_conditions.sounding.file_path ==
            "tests/fixtures/configs/qualification/soundings/ysu_surface_coupled_cloud_profile.txt" &&
        !config.radar.enabled && config.radar.products.empty() &&
        config.output.preset == OutputPreset::ScienceValidation;
}

bool exact_qualified_radar_profile(const ConfigV1& config)
{
    const bool supported_history =
        (config.output.format == OutputFormat::Npy3D && !config.output.async_io) ||
        config.output.format == OutputFormat::ZfpLow ||
        config.output.format == OutputFormat::ZfpMedium ||
        config.output.format == OutputFormat::ZfpHigh;
    ConfigV1 base = config;
    base.radar = {};
    base.output.preset = OutputPreset::Minimal;
    base.output.format = OutputFormat::Npy3D;
    base.output.async_io = false;
    return supported_history && exact_qualified_lin_rrtmg_profile(base) &&
        config.extensions.compute.backend == ComputeBackend::CPU &&
        config.radar.enabled &&
        config.radar.products ==
            std::vector<std::string>{"reflectivity_dbz", "radial_velocity"} &&
        config.radar.reflectivity_tier == "fast_da" &&
        config.radar.velocity_sampling == "point" &&
        config.radar.zdr_tier == "polarimetric_fo" &&
        same_campaign_number(config.radar.radar_x_m, -1500.0) &&
        same_campaign_number(config.radar.radar_y_m, 750.0) &&
        same_campaign_number(config.radar.radar_z_m, 250.0) &&
        config.radar.apply_scatterer_correction &&
        config.output.preset == OutputPreset::RadarReady;
}

bool exact_qualified_polarimetric_radar_profile(const ConfigV1& config)
{
    const bool supported_history =
        (config.output.format == OutputFormat::Npy3D && !config.output.async_io) ||
        config.output.format == OutputFormat::ZfpLow ||
        config.output.format == OutputFormat::ZfpMedium ||
        config.output.format == OutputFormat::ZfpHigh;
    ConfigV1 base = config;
    base.output.format = OutputFormat::Npy3D;
    base.output.async_io = false;
    const auto selected = [&](const char* product)
    {
        return std::find(config.radar.products.begin(),
                         config.radar.products.end(), product) !=
               config.radar.products.end();
    };
    const bool polarimetric = selected("zdr") || selected("kdp") ||
                              selected("rhohv");
    const bool admitted_preset =
        config.output.preset == OutputPreset::Minimal ||
        config.output.preset == OutputPreset::Presentation ||
        config.output.preset == OutputPreset::StudioFull ||
        config.output.preset == OutputPreset::DynamicsFull ||
        config.output.preset == OutputPreset::ScienceValidation ||
        config.output.preset == OutputPreset::RadarReady ||
        config.output.preset == OutputPreset::RadarPolarimetricReady;
    return supported_history && exact_qualified_rrtmg_environment_profile(
               base,
               true,
               "tests/fixtures/configs/qualification/soundings/radar_warm_cloud_profile.txt") &&
        config.microphysics.scheme == MicrophysicsScheme::Kessler &&
        same_campaign_number(
            config.microphysics.parameters.kessler.accretion_coefficient,
            2.2) &&
        same_campaign_number(
            config.microphysics.parameters.kessler.evaporation_efficiency,
            1.0) &&
        config.extensions.compute.backend == ComputeBackend::CPU &&
        config.radar.enabled &&
        !config.radar.products.empty() && admitted_preset &&
        (!polarimetric || config.radar.zdr_tier == "zhang2021_rain_mp");
}

bool exact_qualified_any_radar_profile(const ConfigV1& config)
{
    return exact_qualified_radar_profile(config) ||
        exact_qualified_polarimetric_radar_profile(config);
}

bool exact_qualified_turbulence_reference_environment(
    const ConfigV1& config, bool allow_native_tke = false)
{
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& radiation = config.radiation;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& compute = config.extensions.compute;
    const bool cpu = compute.backend == ComputeBackend::CPU &&
        !compute.gpu_apply_slow && !compute.gpu_apply_sgs;
    const bool native = allow_native_tke &&
        (compute.backend == ComputeBackend::GPU || compute.backend == ComputeBackend::Metal) &&
        compute.gpu_apply_slow && compute.gpu_apply_sgs &&
        config.extensions.solver.slow_rk2 && config.time.integrator == TimeIntegrator::SplitExplicit;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        (cpu || native) &&
        !config.extensions.compute.allow_fallback &&
        !config.extensions.compute.gpu_apply_microphysics &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 12 && config.grid.ny == 12 && config.grid.nz == 16 &&
        same_campaign_number(config.grid.dx, 1000.0) &&
        same_campaign_number(config.grid.dy, 1000.0) &&
        same_campaign_number(config.grid.dz, 250.0) &&
        same_campaign_number(config.grid.ztop, 4000.0) &&
        same_campaign_number(config.time.run_time, 2.0) &&
        same_campaign_number(config.time.dt_large, 0.2) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        same_campaign_number(config.time.cfl_safety, 0.7) &&
        same_campaign_number(config.time.output_interval, 1.0) &&
        config.domain_motion.mode == DomainMotionMode::Stationary &&
        !config.domain_motion.lifetime_planning.enabled &&
        !config.domain_motion.tracking.enabled &&
        !config.domain_motion.tracking.recycle_domain &&
        !config.domain_motion.nesting.enabled &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.advection.scheme == AdvectionScheme::TVD &&
        lateral.west == LateralBC::Periodic &&
        lateral.east == LateralBC::Periodic &&
        lateral.south == LateralBC::Periodic &&
        lateral.north == LateralBC::Periodic &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled &&
        !config.damping.lateral_sponge_enabled &&
        !config.damping.w_only_top_sponge &&
        !config.coriolis.enabled &&
        config.microphysics.scheme == MicrophysicsScheme::None &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        same_campaign_number(config.diffusion.K_h, 0.0) &&
        same_campaign_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::None &&
        !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::None &&
        radiation.scheme == RadiationScheme::RRTMG &&
        radiation.solar_forcing == RadiationSolarForcing::IdealizedDiurnal &&
        same_campaign_number(radiation.latitude_deg, 35.0) &&
        same_campaign_number(radiation.declination_deg, 15.0) &&
        same_campaign_number(radiation.local_solar_time_at_start_h, 12.0) &&
        radiation.do_lw && radiation.do_sw &&
        same_campaign_number(radiation.cadence_s, 1.0) &&
        !radiation.clear_sky_only &&
        same_campaign_number(radiation.rrtmg_ozone_vmr, 4.0e-6) &&
        config.terrain.scheme == TerrainScheme::None &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        !config.radar.enabled && config.radar.products.empty() &&
        sounding.type == SoundingType::ParametricCAPE &&
        same_campaign_number(sounding.cape_jkg, 0.0) &&
        same_campaign_number(sounding.tropopause_m, 8000.0) &&
        same_campaign_number(sounding.surface_theta_k, 300.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        same_campaign_number(*sounding.moisture_qv_kgkg, 0.008) &&
        hodograph.type == HodographType::WKParam &&
        same_campaign_number(hodograph.u_sfc_ms, 5.0) &&
        same_campaign_number(hodograph.v_sfc_ms, 2.0) &&
        same_campaign_number(hodograph.u_1km_ms, 8.0) &&
        same_campaign_number(hodograph.v_1km_ms, 4.0) &&
        same_campaign_number(hodograph.u_6km_ms, 18.0) &&
        same_campaign_number(hodograph.v_6km_ms, 10.0) &&
        config.initial_conditions.trigger.type == TriggerType::None &&
        config.output.preset == OutputPreset::DynamicsFull &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.extensions.checkpoint.every_s == 1 &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions;
}

bool exact_qualified_user_filter_width_profile(const ConfigV1& config)
{
    return exact_qualified_turbulence_reference_environment(config) &&
        config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
        config.turbulence.mode == TurbulenceMode::ThreeD &&
        config.turbulence.filter_width == TurbulenceFilterWidth::User &&
        same_campaign_number(config.turbulence.user_filter_width_m, 500.0) &&
        !config.turbulence.dynamic_coefficient &&
        config.turbulence.stability_correction ==
            TurbulenceStabilityCorrection::None &&
        same_campaign_number(config.turbulence.Cs, 0.18) &&
        same_campaign_number(config.turbulence.Pr_t, 1.0 / 3.0) &&
        same_campaign_number(config.turbulence.Sc_t, 1.0 / 3.0);
}

bool exact_qualified_smagorinsky_reference_profile(const ConfigV1& config)
{
    return exact_qualified_turbulence_reference_environment(config) &&
        config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
        config.turbulence.mode == TurbulenceMode::ThreeD &&
        config.turbulence.filter_width ==
            TurbulenceFilterWidth::CubicRoot &&
        same_campaign_number(config.turbulence.user_filter_width_m, 0.0) &&
        same_campaign_number(config.turbulence.Cs, 0.23) &&
        same_campaign_number(config.turbulence.Pr_t, 1.0 / 3.0) &&
        same_campaign_number(config.turbulence.Sc_t, 1.0 / 3.0);
}

bool exact_qualified_dynamic_smagorinsky_profile(const ConfigV1& config)
{
    return exact_qualified_smagorinsky_reference_profile(config)
        && config.turbulence.dynamic_coefficient
        && config.turbulence.stability_correction == TurbulenceStabilityCorrection::None;
}

bool exact_qualified_richardson_smagorinsky_profile(const ConfigV1& config)
{
    return exact_qualified_smagorinsky_reference_profile(config)
        && !config.turbulence.dynamic_coefficient
        && config.turbulence.stability_correction == TurbulenceStabilityCorrection::Richardson;
}

bool reference_transport_bubble(const ConfigV1& config, double cloud_water_kgkg)
{
    const auto& trigger = config.initial_conditions.trigger;
    const auto& bubble  = trigger.bubble;
    return trigger.type == TriggerType::Bubble && same_campaign_number(bubble.dtheta_k, 2.0) &&
           same_campaign_number(bubble.cloud_water_kgkg, cloud_water_kgkg) &&
           same_campaign_number(bubble.radius_km, 2.0) && same_campaign_number(bubble.center_x_m, 6000.0) &&
           same_campaign_number(bubble.center_y_m, 6000.0) && same_campaign_number(bubble.center_z_m, 1500.0);
}

bool exact_qualified_tke_stability_profile(const ConfigV1& config)
{
    if (config.advection.scheme == AdvectionScheme::WENO5 &&
        (config.time.integrator != TimeIntegrator::SplitExplicit || !config.extensions.solver.slow_rk2))
        return false;
    ConfigV1 reference = config;
    if (config.microphysics.scheme == MicrophysicsScheme::Thompson)
    {
        const bool native = config.extensions.compute.backend != ComputeBackend::CPU;
        if (!reference_thompson_parameters(config) || !config.radiation.clear_sky_only ||
            config.extensions.compute.gpu_apply_microphysics != native)
        {
            return false;
        }
        if (!reference_transport_bubble(config, 0.001))
            return false;
        if (config.advection.scheme == AdvectionScheme::WENO5)
            reference.advection.scheme = AdvectionScheme::TVD;
        reference.initial_conditions.trigger.type           = TriggerType::None;
        reference.microphysics.scheme                       = MicrophysicsScheme::None;
        reference.extensions.compute.gpu_apply_microphysics = false;
        reference.radiation.clear_sky_only                  = false;
    }

    if (config.microphysics.scheme == MicrophysicsScheme::None && config.advection.scheme == AdvectionScheme::WENO5)
    {
        if (!reference_transport_bubble(config, 0.0))
            return false;
        reference.advection.scheme                = AdvectionScheme::TVD;
        reference.initial_conditions.trigger.type = TriggerType::None;
    }

    if (config.diffusion.scheme == DiffusionScheme::Implicit)
    {
        if (!reference_implicit_diffusion(config))
        {
            return false;
        }
        reference.diffusion.scheme   = DiffusionScheme::Explicit;
        reference.diffusion.apply_to = DiffusionTarget::Scalars;
        reference.diffusion.K_h = reference.diffusion.K_v = 0.0;
    }
    return exact_qualified_turbulence_reference_environment(reference, true) &&
           config.turbulence.scheme == TurbulenceScheme::TKE && config.turbulence.mode == TurbulenceMode::ThreeD &&
           config.turbulence.filter_width == TurbulenceFilterWidth::CubicRoot &&
           same_campaign_number(config.turbulence.user_filter_width_m, 0.0) && !config.turbulence.dynamic_coefficient &&
           config.turbulence.stability_correction == TurbulenceStabilityCorrection::TkeBased &&
           same_campaign_number(config.turbulence.Cs, 0.18) &&
           same_campaign_number(config.turbulence.Pr_t, 1.0 / 3.0) &&
           same_campaign_number(config.turbulence.Sc_t, 1.0 / 3.0);
}

bool exact_qualified_coupled_rk_profile(const ConfigV1& config)
{
    const auto& lateral = config.boundary_conditions.lateral;
    const bool  supported_integrator =
        config.time.integrator == TimeIntegrator::SSPRK3 || config.time.integrator == TimeIntegrator::RK4;
    const bool supported_microphysics = config.microphysics.scheme == MicrophysicsScheme::None ||
                                        config.microphysics.scheme == MicrophysicsScheme::Kessler ||
                                        config.microphysics.scheme == MicrophysicsScheme::Lin ||
                                        config.microphysics.scheme == MicrophysicsScheme::Thompson ||
                                        config.microphysics.scheme == MicrophysicsScheme::Milbrandt;
    const bool supported_turbulence =
        config.turbulence.scheme == TurbulenceScheme::None ||
        (config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
         config.turbulence.filter_width == TurbulenceFilterWidth::CubicRoot && !config.turbulence.dynamic_coefficient &&
         config.turbulence.stability_correction == TurbulenceStabilityCorrection::None) ||
        (config.turbulence.scheme == TurbulenceScheme::TKE &&
         config.turbulence.filter_width == TurbulenceFilterWidth::CubicRoot && !config.turbulence.dynamic_coefficient &&
         config.turbulence.stability_correction == TurbulenceStabilityCorrection::TkeBased);
    const bool supported_column_physics =
        (config.surface.scheme == SurfaceScheme::None && !config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::None &&
         (config.radiation.scheme == RadiationScheme::None || config.radiation.scheme == RadiationScheme::RRTMG)) ||
        (config.surface.scheme == SurfaceScheme::Slab && config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::Slab &&
         config.radiation.scheme == RadiationScheme::SimpleGrey &&
         config.radiation.simple_grey_optical_model == "pressure_clear_sky_v1") ||
        (config.surface.scheme == SurfaceScheme::Slab && config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::YSU &&
         config.radiation.scheme == RadiationScheme::RRTMG) ||
        (config.surface.scheme == SurfaceScheme::Slab && config.surface.apply_fluxes &&
         config.boundary_layer.scheme == BoundaryLayerScheme::MYNN &&
         config.radiation.scheme == RadiationScheme::RRTMG);
    const bool supported_reconstruction = config.advection.scheme == AdvectionScheme::None ||
                                          config.advection.scheme == AdvectionScheme::TVD ||
                                          config.advection.scheme == AdvectionScheme::WENO5;
    return config.extensions.execution.mode == ExecutionMode::Qualified && !config.extensions.compute.allow_fallback &&
           ((config.extensions.compute.backend == ComputeBackend::CPU && !config.extensions.compute.gpu_apply_slow &&
             !config.extensions.compute.gpu_apply_microphysics && !config.extensions.compute.gpu_apply_sgs) ||
            ((config.extensions.compute.backend == ComputeBackend::GPU ||
              config.extensions.compute.backend == ComputeBackend::Metal) &&
             config.extensions.compute.gpu_apply_slow && config.extensions.compute.gpu_apply_microphysics &&
             config.extensions.compute.gpu_apply_sgs)) &&
           supported_integrator && config.grid.coordinate_system == CoordinateSystem::Cartesian &&
           (config.grid.staggering == StaggerType::CGrid || config.grid.staggering == StaggerType::Collocated) &&
           !config.time.adaptive_dt && config.domain_motion.mode == DomainMotionMode::Stationary &&
           !config.domain_motion.lifetime_planning.enabled && !config.domain_motion.tracking.enabled &&
           !config.domain_motion.tracking.recycle_domain && !config.domain_motion.nesting.enabled &&
           config.dynamics.scheme == (config.grid.staggering == StaggerType::CGrid
                                          ? "klemp_wilhelmson_split_cartesian_cgrid"
                                          : "klemp_wilhelmson_split_cartesian_agrid") &&
           !config.dynamics.axisymmetric && supported_reconstruction && lateral.west == LateralBC::Periodic &&
           lateral.east == LateralBC::Periodic && lateral.south == LateralBC::Periodic &&
           lateral.north == LateralBC::Periodic && config.boundary_conditions.bottom == BottomBC::RigidSlip &&
           config.boundary_conditions.top == TopBC::Rigid && !config.damping.enabled &&
           !config.damping.lateral_sponge_enabled && !config.damping.w_only_top_sponge && !config.coriolis.enabled &&
           supported_microphysics && supported_turbulence &&
           (config.diffusion.scheme == DiffusionScheme::Explicit ||
            config.diffusion.scheme == DiffusionScheme::Implicit) &&
           config.diffusion.operator_order == 2 &&
           config.diffusion.coefficient_source == DiffusionCoefficientSource::Constant && supported_column_physics &&
           config.terrain.scheme == TerrainScheme::None && !config.radar.enabled && config.radar.products.empty() &&
           config.extensions.chaos.scheme == ChaosScheme::None;
}

bool exact_simple_grey_absorbing_water_cloud_profile(const ConfigV1& config)
{
    const auto& radiation = config.radiation;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.extensions.compute.backend == ComputeBackend::CPU &&
        !config.extensions.compute.allow_fallback &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 12 && config.grid.ny == 12 &&
        config.grid.nz == 16 && same_campaign_number(config.grid.dx, 1000.0) &&
        same_campaign_number(config.grid.dy, 1000.0) &&
        same_campaign_number(config.grid.dz, 250.0) &&
        same_campaign_number(config.grid.ztop, 4000.0) &&
        same_campaign_number(config.time.run_time, 2.0) &&
        same_campaign_number(config.time.dt_large, 0.2) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        same_campaign_number(config.time.cfl_safety, 0.7) &&
        same_campaign_number(config.time.output_interval, 1.0) &&
        config.domain_motion.mode == DomainMotionMode::Stationary &&
        !config.domain_motion.nesting.enabled &&
        !config.domain_motion.tracking.enabled &&
        !config.domain_motion.tracking.recycle_domain &&
        config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.advection.scheme == AdvectionScheme::TVD &&
        lateral.west == LateralBC::Periodic &&
        lateral.east == LateralBC::Periodic &&
        lateral.south == LateralBC::Periodic &&
        lateral.north == LateralBC::Periodic &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled &&
        !config.damping.lateral_sponge_enabled &&
        !config.damping.w_only_top_sponge &&
        !config.coriolis.enabled &&
        config.microphysics.scheme == MicrophysicsScheme::Kessler &&
        same_campaign_number(
            config.microphysics.parameters.shared.cloud_water_threshold_kgkg,
            1.0e-3) &&
        same_campaign_number(
            config.microphysics.parameters.shared.autoconversion_rate_s_inv,
            1.0e-3) &&
        same_campaign_number(
            config.microphysics.parameters.kessler.accretion_coefficient,
            2.2) &&
        same_campaign_number(
            config.microphysics.parameters.kessler.evaporation_efficiency,
            1.0) &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        same_campaign_number(config.diffusion.K_h, 0.0) &&
        same_campaign_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.surface.scheme == SurfaceScheme::None &&
        !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::None &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        tmv::runtime::is_exact_qualified_simple_grey_absorbing_water_cloud({
            radiation_scheme_name(radiation.scheme),
            radiation.simple_grey_optical_model,
            radiation_solar_forcing_name(radiation.solar_forcing),
            radiation.do_lw,
            radiation.do_sw,
            radiation.clear_sky_only,
            radiation.use_subcycling,
            static_cast<std::size_t>(radiation.subcycle_count),
            radiation.enable_diagnostics,
            radiation.cadence_s,
            radiation.tau_lw_ref,
            radiation.tau_sw_ref,
            radiation.n_lw,
            radiation.n_sw,
            radiation.simple_grey_lw_vapor_absorption_m2kg,
            radiation.simple_grey_lw_liquid_absorption_m2kg,
            radiation.simple_grey_lw_ice_absorption_m2kg,
            radiation.simple_grey_sw_vapor_absorption_m2kg,
            radiation.simple_grey_sw_liquid_absorption_m2kg,
            radiation.simple_grey_sw_ice_absorption_m2kg,
            radiation.latitude_deg,
            radiation.declination_deg,
            radiation.local_solar_time_at_start_h,
            config.surface.albedo,
            config.surface.emissivity,
            config.surface.tsk_k,
        }) &&
        config.terrain.scheme == TerrainScheme::None &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        sounding.type == SoundingType::File &&
        sounding.file_path ==
            "tests/fixtures/configs/qualification/soundings/simple_grey_condensing_profile.txt" &&
        hodograph.type == HodographType::WKParam &&
        same_campaign_number(hodograph.u_sfc_ms, 0.0) &&
        same_campaign_number(hodograph.v_sfc_ms, 0.0) &&
        same_campaign_number(hodograph.u_1km_ms, 0.0) &&
        same_campaign_number(hodograph.v_1km_ms, 0.0) &&
        same_campaign_number(hodograph.u_6km_ms, 0.0) &&
        same_campaign_number(hodograph.v_6km_ms, 0.0) &&
        config.initial_conditions.trigger.type == TriggerType::None &&
        config.output.preset == OutputPreset::ScienceValidation &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.extensions.checkpoint.every_s == 1 &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error;
}
} // namespace

bool is_public_qualified_domain_recycling_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_domain_recycling_profile(config);
}

bool is_public_qualified_nested_grid_recycling_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_nested_grid_recycling_profile(config) ||
        (config.domain_motion.mode == DomainMotionMode::TrackedWindow && is_public_qualified_accelerated_two_way_profile(config));
}

bool is_public_qualified_earth_fixed_terrain_motion_profile(const ConfigV1& config) noexcept
{
    try { return exact_qualified_earth_fixed_terrain_motion_profile(config); }
    catch (...) { return false; }
}

bool is_public_qualified_earth_fixed_terrain_recycling_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_earth_fixed_terrain_recycling_profile(config) ||
           exact_qualified_mapped_nested_recycling_profile(config);
}

bool is_public_qualified_moving_nest_profile(
    const ConfigV1& config) noexcept
{
    return valid_public_qualified_moving_nest_profile(config);
}

bool is_public_ready_private_two_way_profile(const ConfigV1& config) noexcept
{
    const bool thompson  = config.microphysics.scheme == MicrophysicsScheme::Thompson;
    const bool milbrandt = config.microphysics.scheme == MicrophysicsScheme::Milbrandt;
    const bool native    = config.extensions.compute.backend == ComputeBackend::Metal ||
                           config.extensions.compute.backend == ComputeBackend::GPU;
    const bool moving = (thompson || milbrandt) && config.domain_motion.mode == DomainMotionMode::MovingNest;
    const bool recycling = config.domain_motion.mode == DomainMotionMode::TrackedWindow &&
                           (milbrandt || thompson);
    if ((moving || recycling) && config.domain_motion.nesting.feedback == NestingFeedback::TwoWay)
    {
        if ((thompson && !reference_thompson_parameters(config)) ||
            (milbrandt && (config.microphysics.parameters.milbrandt.triple_moment ||
                !same_campaign_number(config.microphysics.parameters.milbrandt.evaporation_rate_s_inv, .001))) ||
            !same_campaign_number(config.microphysics.parameters.shared.cloud_water_threshold_kgkg, .001) ||
            !same_campaign_number(config.microphysics.parameters.shared.autoconversion_rate_s_inv, .001) ||
            config.output.preset != OutputPreset::DynamicsFull)
            return false;
        try
        {
            auto reference = config;
            reference.output.preset = OutputPreset::ScienceValidation;
            reference.microphysics.scheme = MicrophysicsScheme::Kessler;
            reference.microphysics.parameters.kessler.accretion_coefficient = 2.2;
            reference.microphysics.parameters.kessler.evaporation_efficiency = 1;
            return is_public_qualified_two_way_nested_profile(reference);
        }
        catch (...)
        {
            return false;
        }
    }
    if ((!milbrandt && !thompson) || config.domain_motion.mode != DomainMotionMode::Stationary ||
        config.domain_motion.nesting.feedback != NestingFeedback::TwoWay ||
        (config.extensions.compute.backend != ComputeBackend::CPU && !native) ||
        config.extensions.compute.gpu_apply_microphysics != native ||
        (config.advection.scheme != AdvectionScheme::TVD && config.advection.scheme != AdvectionScheme::WENO5))
        return false;
    if (thompson && (!reference_thompson_parameters(config)))
        return false;
    const auto& trigger = config.initial_conditions.trigger;
    const auto& bubble  = trigger.bubble;
    const bool seeded = trigger.type == TriggerType::Bubble && same_campaign_number(bubble.dtheta_k, 2) &&
                        same_campaign_number(bubble.cloud_water_kgkg, .001) && same_campaign_number(bubble.radius_km, 30) &&
                        same_campaign_number(bubble.center_x_m, 8000) && same_campaign_number(bubble.center_y_m, 8000) &&
                        same_campaign_number(bubble.center_z_m, 6000);
    const auto& vortex = trigger.vortex_seed;
    const bool unseeded = trigger.type == TriggerType::VortexSeed && same_campaign_number(vortex.r_max_m, 2000) &&
                          same_campaign_number(vortex.v_max_ms, 1) && same_campaign_number(vortex.z_top_m, 4000) &&
                          same_campaign_number(vortex.center_x_m, 8000) && same_campaign_number(vortex.center_y_m, 8000) &&
                          same_campaign_number(vortex.center_z_m, 2000);
    if (!seeded && !unseeded)
    {
        return false;
    }
    const auto expected_sounding = thompson
                                       ? "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt"
                                       : "tests/fixtures/configs/qualification/soundings/milbrandt_mixed_profile.txt";
    if (config.initial_conditions.sounding.type != SoundingType::File ||
        config.initial_conditions.sounding.file_path != expected_sounding)
        return false;
    const auto& parameters = config.microphysics.parameters;
    if (milbrandt && (parameters.milbrandt.triple_moment ||
                      !same_campaign_number(parameters.milbrandt.evaporation_rate_s_inv, .001) ||
                      !same_campaign_number(parameters.shared.cloud_water_threshold_kgkg, .001) ||
                      !same_campaign_number(parameters.shared.autoconversion_rate_s_inv, .001)))
        return false;
    try
    {
        auto reference = config;
        reference.initial_conditions.sounding.file_path =
            "tests/fixtures/configs/qualification/soundings/simple_grey_condensing_profile.txt";
        reference.initial_conditions.trigger.type                    = TriggerType::VortexSeed;
        auto& vortex                                                 = reference.initial_conditions.trigger.vortex_seed;
        vortex.r_max_m                                               = 2000;
        vortex.v_max_ms                                              = 1;
        vortex.z_top_m                                               = 4000;
        vortex.center_x_m                                            = 8000;
        vortex.center_y_m                                            = 8000;
        vortex.center_z_m                                            = 2000;
        reference.microphysics.scheme                                = MicrophysicsScheme::Lin;
        reference.microphysics.parameters.lin.accretion_coefficient  = 2.2;
        reference.microphysics.parameters.lin.evaporation_rate_s_inv = .001;
        reference.extensions.compute.backend                         = ComputeBackend::CPU;
        reference.extensions.compute.gpu_apply_microphysics          = false;
        reference.advection.scheme                                   = AdvectionScheme::TVD;
        return is_public_qualified_two_way_nested_profile(reference);
    }
    catch (...)
    {
        return false;
    }
}

bool is_public_qualified_accelerated_two_way_profile(const ConfigV1& config) noexcept
{
    if (config.extensions.compute.backend != ComputeBackend::CPU && exact_qualified_mapped_nested_recycling_profile(config))
    {
        return true;
    }
    if (is_public_ready_private_two_way_profile(config))
        return true;
    if (config.microphysics.scheme == MicrophysicsScheme::Thompson ||
        config.microphysics.scheme == MicrophysicsScheme::Milbrandt)
        return false;
    const bool device = config.extensions.compute.backend == ComputeBackend::GPU ||
                        config.extensions.compute.backend == ComputeBackend::Metal;
    if ((config.advection.scheme != AdvectionScheme::WENO5 &&
         !(device && config.advection.scheme == AdvectionScheme::TVD)) ||
        (config.terrain.scheme != TerrainScheme::None && config.domain_motion.mode != DomainMotionMode::Stationary &&
         config.domain_motion.mode != DomainMotionMode::MovingNest) ||
        (config.domain_motion.mode != DomainMotionMode::Stationary &&
         config.domain_motion.mode != DomainMotionMode::MovingNest &&
         config.domain_motion.mode != DomainMotionMode::TrackedWindow) ||
        config.domain_motion.nesting.feedback != NestingFeedback::TwoWay ||
        (config.extensions.compute.backend != ComputeBackend::CPU &&
         config.extensions.compute.backend != ComputeBackend::GPU &&
         config.extensions.compute.backend != ComputeBackend::Metal))
        return false;
    try
    {
        auto reference = config;
        if (config.domain_motion.mode != DomainMotionMode::Stationary)
        {
            const bool native = config.extensions.compute.backend != ComputeBackend::CPU &&
                                config.terrain.scheme == TerrainScheme::None;
            if (config.extensions.compute.gpu_apply_sgs != native ||
                config.extensions.chaos.native_generation_required != native)
                return false;
            reference.extensions.compute.gpu_apply_sgs            = false;
            reference.extensions.chaos.native_generation_required = false;
        }
        reference.advection.scheme           = AdvectionScheme::TVD;
        reference.extensions.compute.backend = ComputeBackend::CPU;
        return is_public_qualified_two_way_nested_profile(reference);
    }
    catch (...)
    {
        return false;
    }
}

bool is_public_qualified_two_way_nested_profile(const ConfigV1& config) noexcept
{
    if (config.domain_motion.nesting.feedback != NestingFeedback::TwoWay)
        return false;
    if (is_public_qualified_accelerated_two_way_profile(config))
        return true;
    if (exact_qualified_stationary_nested_profile(config) || exact_qualified_nested_grid_recycling_profile(config) ||
        exact_qualified_two_way_moving_profile(config))
        return true;
    const auto& sounding = config.initial_conditions.sounding;
    const bool  humid    = sounding.type == SoundingType::ParametricCAPE && sounding.moisture_qv_kgkg &&
                           same_campaign_number(*sounding.moisture_qv_kgkg, .022);
    const bool  condensing =
        sounding.type == SoundingType::File &&
        sounding.file_path == "tests/fixtures/configs/qualification/soundings/simple_grey_condensing_profile.txt";
    const bool lin = config.microphysics.scheme == MicrophysicsScheme::Lin;
    if ((!lin && config.microphysics.scheme != MicrophysicsScheme::Kessler) || (!humid && !condensing) ||
        (lin && (!condensing || config.output.preset != OutputPreset::DynamicsFull)))
        return false;
    const auto& micro = config.microphysics.parameters;
    if (!same_campaign_number(micro.shared.cloud_water_threshold_kgkg, .001) ||
        !same_campaign_number(micro.shared.autoconversion_rate_s_inv, .001) ||
        (lin ? (!same_campaign_number(micro.lin.accretion_coefficient, 2.2) ||
                !same_campaign_number(micro.lin.evaporation_rate_s_inv, .001))
             : (!same_campaign_number(micro.kessler.accretion_coefficient, 2.2) ||
                !same_campaign_number(micro.kessler.evaporation_efficiency, 1))))
        return false;
    try
    {
        auto reference                = config;
        reference.microphysics.scheme = MicrophysicsScheme::None;
        if (lin)
            reference.output.preset = OutputPreset::ScienceValidation;
        if (condensing)
        {
            reference.initial_conditions.sounding.type            = SoundingType::ParametricCAPE;
            reference.initial_conditions.sounding.cape_jkg        = 3275.016;
            reference.initial_conditions.sounding.tropopause_m    = 8000;
            reference.initial_conditions.sounding.surface_theta_k = 300;
        }
        reference.initial_conditions.sounding.moisture_qv_kgkg = 0;
        return exact_qualified_stationary_nested_profile(reference);
    }
    catch (...)
    {
        return false;
    }
}

bool is_public_qualified_active_terrain_nested_profile(const ConfigV1& config) noexcept
{
    return config.terrain.scheme != TerrainScheme::None &&
           (exact_qualified_stationary_nested_profile(config) || is_public_qualified_two_way_nested_profile(config));
}

bool is_public_qualified_lin_rrtmg_profile(const ConfigV1& config) noexcept
{
    return exact_qualified_lin_rrtmg_profile(config) ||
        exact_qualified_cloud_top_radiative_ysu_profile(config) ||
        exact_qualified_radar_profile(config) ||
        exact_qualified_polarimetric_radar_profile(config);
}

bool is_public_ready_milbrandt_reference_profile(const ConfigV1& config)
{
    const bool native = config.extensions.compute.backend == ComputeBackend::GPU ||
                        config.extensions.compute.backend == ComputeBackend::Metal;
    if (config.microphysics.scheme != MicrophysicsScheme::Milbrandt ||
        (!native && config.extensions.compute.backend != ComputeBackend::CPU) ||
        config.extensions.compute.gpu_apply_microphysics != native || config.extensions.compute.allow_fallback ||
        config.microphysics.parameters.milbrandt.triple_moment || config.advection.scheme != AdvectionScheme::TVD ||
        config.diffusion.scheme != DiffusionScheme::Explicit || config.turbulence.scheme != TurbulenceScheme::None)
        return false;
    if (config.initial_conditions.sounding.file_path !=
        "tests/fixtures/configs/qualification/soundings/milbrandt_mixed_profile.txt")
        return false;
    ConfigV1 reference = config;
    reference.initial_conditions.sounding.file_path =
        "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt";
    reference.extensions.compute.backend                                         = ComputeBackend::CPU;
    reference.extensions.compute.gpu_apply_microphysics                          = false;
    reference.microphysics.scheme                                                = MicrophysicsScheme::Thompson;
    reference.microphysics.parameters.thompson.water_friendly_aerosol_surface_m3 = 350e6;
    reference.microphysics.parameters.thompson.ice_friendly_aerosol_surface_m3   = 2e6;
    return exact_qualified_thompson_cpu_profile(reference);
}

bool is_public_ready_thompson_native_profile(const ConfigV1& config)
{
    if (config.microphysics.scheme != MicrophysicsScheme::Thompson ||
        (config.extensions.compute.backend != ComputeBackend::GPU &&
         config.extensions.compute.backend != ComputeBackend::Metal) ||
        !config.extensions.compute.gpu_apply_microphysics)
        return false;
    if (exact_qualified_tke_stability_profile(config))
    {
        return true;
    }
    ConfigV1 cpu                                  = config;
    cpu.extensions.compute.backend                = ComputeBackend::CPU;
    cpu.extensions.compute.gpu_apply_microphysics = false;
    return exact_qualified_thompson_cpu_profile(cpu);
}

bool is_public_qualified_thompson_cpu_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_thompson_cpu_profile(config) ||
           (config.extensions.compute.backend == ComputeBackend::CPU &&
            config.microphysics.scheme == MicrophysicsScheme::Thompson &&
            exact_qualified_tke_stability_profile(config));
}

bool is_public_qualified_cloud_top_radiative_ysu_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_cloud_top_radiative_ysu_profile(config);
}

bool is_public_qualified_radar_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_any_radar_profile(config);
}

bool is_public_qualified_user_filter_width_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_user_filter_width_profile(config);
}

bool is_public_qualified_richardson_smagorinsky_profile(const ConfigV1& config) noexcept
{
    return exact_qualified_richardson_smagorinsky_profile(config);
}

bool is_public_qualified_dynamic_smagorinsky_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_dynamic_smagorinsky_profile(config);
}

bool is_transactional_static_vertical_only_cpu_candidate(
    const ConfigV1& config)
{
    ConfigV1 profile = config;
    profile.extensions.execution.mode = ExecutionMode::Qualified;
    return exact_qualified_turbulence_reference_environment(profile) &&
        profile.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
        profile.turbulence.mode == TurbulenceMode::VerticalOnly &&
        profile.turbulence.filter_width == TurbulenceFilterWidth::CubicRoot &&
        same_campaign_number(profile.turbulence.user_filter_width_m, 0.0) &&
        !profile.turbulence.dynamic_coefficient &&
        profile.turbulence.stability_correction == TurbulenceStabilityCorrection::None &&
        same_campaign_number(profile.turbulence.Cs, 0.23) &&
        same_campaign_number(profile.turbulence.Pr_t, 1.0 / 3.0) &&
        same_campaign_number(profile.turbulence.Sc_t, 1.0 / 3.0);
}

bool is_public_ready_native_weno5_profile(const ConfigV1& config)
{
    if (config.advection.scheme == AdvectionScheme::WENO5 &&
        (config.extensions.compute.backend == ComputeBackend::GPU ||
         config.extensions.compute.backend == ComputeBackend::Metal) &&
        (exact_qualified_tke_stability_profile(config) || is_public_ready_thompson_native_profile(config)))
    {
        return true;
    }
    const auto& compute = config.extensions.compute;
    if ((compute.backend != ComputeBackend::GPU && compute.backend != ComputeBackend::Metal) ||
        compute.allow_fallback || compute.gpu_apply_slow || compute.gpu_apply_sgs || compute.gpu_apply_microphysics ||
        config.advection.scheme != AdvectionScheme::WENO5 || config.time.integrator != TimeIntegrator::SplitExplicit ||
        !config.extensions.solver.slow_rk2 || !reference_transport_bubble(config, 0.0) ||
        config.turbulence.scheme != TurbulenceScheme::None)
    {
        return false;
    }
    ConfigV1 reference                        = config;
    reference.extensions.compute.backend      = ComputeBackend::CPU;
    reference.advection.scheme                = AdvectionScheme::TVD;
    reference.initial_conditions.trigger.type = TriggerType::None;
    return exact_qualified_turbulence_reference_environment(reference);
}

bool is_public_qualified_native_implicit_diffusion_profile(const ConfigV1& config)
{
    if (config.diffusion.scheme == DiffusionScheme::Implicit &&
        (config.extensions.compute.backend == ComputeBackend::GPU ||
         config.extensions.compute.backend == ComputeBackend::Metal) &&
        (exact_qualified_tke_stability_profile(config) || is_public_ready_thompson_native_profile(config)))
    {
        return true;
    }
    const auto& compute   = config.extensions.compute;
    const auto& trigger   = config.initial_conditions.trigger;
    const auto& bubble    = trigger.bubble;
    if (config.extensions.execution.mode != ExecutionMode::Qualified ||
        (compute.backend != ComputeBackend::GPU && compute.backend != ComputeBackend::Metal) ||
        compute.allow_fallback || !compute.gpu_apply_slow || compute.gpu_apply_sgs || compute.gpu_apply_microphysics ||
        !config.extensions.solver.slow_rk2 || !reference_implicit_diffusion(config) ||
        config.turbulence.scheme != TurbulenceScheme::None || trigger.type != TriggerType::Bubble ||
        !same_campaign_number(bubble.dtheta_k, 2.0) || !same_campaign_number(bubble.cloud_water_kgkg, 0.0) ||
        !same_campaign_number(bubble.radius_km, 2.0) || !same_campaign_number(bubble.center_x_m, 6000.0) ||
        !same_campaign_number(bubble.center_y_m, 6000.0) || !same_campaign_number(bubble.center_z_m, 1500.0))
        return false;
    ConfigV1 reference                          = config;
    reference.extensions.compute.backend        = ComputeBackend::CPU;
    reference.extensions.compute.gpu_apply_slow = false;
    reference.diffusion.scheme                  = DiffusionScheme::Explicit;
    reference.diffusion.apply_to                = DiffusionTarget::Scalars;
    reference.diffusion.K_h = reference.diffusion.K_v = 0.0;
    reference.initial_conditions.trigger.type         = TriggerType::None;
    return exact_qualified_turbulence_reference_environment(reference);
}

bool is_public_qualified_tke_stability_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_tke_stability_profile(config);
}

bool is_public_qualified_coupled_rk_profile(
    const ConfigV1& config) noexcept
{
    return exact_qualified_coupled_rk_profile(config);
}

bool is_public_qualified_stationary_simple_grey_profile(
    const ConfigV1& config) noexcept
{
    const auto& motion = config.domain_motion;
    const auto& nesting = motion.nesting;
    const auto& compute = config.extensions.compute;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& radiation = config.radiation;
    const bool supported_backend =
        compute.backend == ComputeBackend::CPU
        || compute.backend == ComputeBackend::GPU;

    return config.extensions.execution.mode == ExecutionMode::Qualified
        && supported_backend && !compute.allow_fallback
        && motion.mode == DomainMotionMode::Stationary
        && nesting.enabled
        && nesting.feedback == NestingFeedback::OneWay
        && nesting.parent_ratio_x == 2
        && nesting.parent_ratio_y == 2
        && nesting.parent_time_step_ratio == 2
        && !motion.tracking.enabled && !motion.tracking.recycle_domain
        && config.grid.coordinate_system == CoordinateSystem::Cartesian
        && config.grid.staggering == StaggerType::CGrid
        && config.dynamics.scheme ==
            "klemp_wilhelmson_split_cartesian_cgrid"
        && !config.dynamics.axisymmetric
        && lateral.west == LateralBC::ZeroGradient
        && lateral.east == LateralBC::ZeroGradient
        && lateral.south == LateralBC::ZeroGradient
        && lateral.north == LateralBC::ZeroGradient
        && config.boundary_conditions.bottom == BottomBC::RigidSlip
        && config.boundary_conditions.top == TopBC::Rigid
        && !config.damping.enabled
        && !config.damping.lateral_sponge_enabled
        && !config.damping.w_only_top_sponge
        && !config.coriolis.enabled
        && config.microphysics.scheme == MicrophysicsScheme::None
        && config.turbulence.scheme == TurbulenceScheme::None
        && config.diffusion.scheme == DiffusionScheme::Explicit
        && config.diffusion.operator_order == 2
        && config.diffusion.K_h == 0.0
        && config.diffusion.K_v == 0.0
        && config.surface.scheme == SurfaceScheme::None
        && !config.surface.apply_fluxes
        && config.boundary_layer.scheme == BoundaryLayerScheme::None
        && config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson
        && tmv::runtime::is_exact_qualified_simple_grey_longwave({
            radiation_scheme_name(radiation.scheme),
            radiation_solar_forcing_name(radiation.solar_forcing),
            radiation.do_lw,
            radiation.do_sw,
            radiation.clear_sky_only,
            false,
            false,
            radiation.cadence_s,
            radiation.tau_lw_ref,
            radiation.n_lw,
            config.surface.emissivity,
            config.surface.tsk_k,
        })
        && config.terrain.scheme == TerrainScheme::None
        && config.extensions.chaos.scheme == ChaosScheme::None;
}

bool is_public_qualified_simple_grey_absorbing_water_cloud_profile(
    const ConfigV1& config) noexcept
{
    return exact_simple_grey_absorbing_water_cloud_profile(config);
}

void require_runtime_supported_diffusion(const ConfigV1& config)
{
    if (config.diffusion.scheme == DiffusionScheme::None)
    {
        throw std::invalid_argument("diffusion.scheme=none has no registered runtime scheme; use "
                                    "explicit with K_h=0 and K_v=0");
    }
    if (config.diffusion.operator_order != 2)
    {
        throw std::invalid_argument("diffusion.operator_order=" + std::to_string(config.diffusion.operator_order) +
                                    " is not runtime-supported; only order 2 is implemented");
    }
}

void require_runtime_supported_enum_selections(const ConfigV1& config)
{
    if (config.microphysics.parameters.milbrandt.triple_moment &&
        !tmv::microphysics::kMilbrandtTripleMomentPublicAdmissionOpen)
    {
        throw std::invalid_argument(
            "microphysics.milbrandt.triple_moment is retained but engineering incomplete and not admitted");
    }
    if (config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
        config.turbulence.dynamic_coefficient &&
        config.turbulence.mode != TurbulenceMode::ThreeD)
    {
        throw std::invalid_argument(
            "dynamic Smagorinsky requires turbulence.mode=3d in every execution mode");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.scheme == TurbulenceScheme::Smagorinsky &&
        config.turbulence.mode == TurbulenceMode::VerticalOnly &&
        !is_transactional_static_vertical_only_cpu_candidate(config))
    {
        throw std::invalid_argument(
            "qualified static vertical_only Smagorinsky is guarded pending whole-step failure atomicity");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.dynamic_coefficient &&
        !exact_qualified_dynamic_smagorinsky_profile(config))
    {
        throw std::invalid_argument(
            "dynamic Smagorinsky is qualified only for the exact flat periodic C-grid CPU reference profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.scheme == TurbulenceScheme::TKE &&
        !exact_qualified_tke_stability_profile(config) &&
        !exact_qualified_coupled_rk_profile(config) &&
        !tmv::runtime::is_public_ready_flat_agrid_cpu_family(config) &&
        !is_public_qualified_mapped_terrain_profile(config))
    {
        throw std::invalid_argument(
            "turbulence.scheme=tke is outside the qualified flat, RK, A-grid, and mapped CPU families");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.turbulence.stability_correction ==
            TurbulenceStabilityCorrection::Richardson &&
        !exact_qualified_richardson_smagorinsky_profile(config))
    {
        throw std::invalid_argument(
            "Richardson-corrected Smagorinsky requires the maintained Lilly CPU reference profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        !config.boundary_layer.clear_sky_only &&
        !exact_qualified_cloud_top_radiative_ysu_profile(config))
    {
        throw std::invalid_argument(
            "cloud-top radiative YSU mixing is qualified only for the exact surface-coupled CPU reference profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.time.integrator != TimeIntegrator::SplitExplicit &&
        !exact_qualified_coupled_rk_profile(config) &&
        !tmv::runtime::is_public_ready_flat_agrid_cpu_family(config))
    {
        throw std::invalid_argument(
            "qualified coupled RK integration requires a ready flat CPU profile or native Cartesian profile with device ownership");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.boundary_layer.pbl_top_method !=
            BoundaryLayerTopMethod::Richardson &&
        !is_public_qualified_pbl_top_profile(config) &&
        !exact_qualified_coupled_rk_profile(config) &&
        !tmv::runtime::is_public_ready_flat_agrid_cpu_family(config) &&
        !is_public_qualified_mapped_terrain_profile(config))
    {
        throw std::invalid_argument(
            "qualified non-Richardson PBL-top diagnosis is limited to the exact YSU/theta and MYNN/TKE CPU reference profiles");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.radiation.scheme == RadiationScheme::SimpleGrey &&
        config.radiation.simple_grey_optical_model ==
            "absorbing_water_cloud_path_v1" &&
        !exact_simple_grey_absorbing_water_cloud_profile(config) &&
        !is_public_qualified_mapped_terrain_profile(config))
    {
        throw std::invalid_argument(
            "absorbing water/cloud-path simple_grey is qualified only for the exact flat periodic C-grid CPU profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.domain_motion.nesting.enabled &&
        config.radiation.scheme == RadiationScheme::SimpleGrey &&
        !is_public_qualified_stationary_simple_grey_profile(config) &&
        !exact_qualified_mapped_nested_recycling_profile(config))
    {
        throw std::invalid_argument(
            "domain_motion.nesting is limited to the exact stationary simple_grey longwave reference profile");
    }
    if (config.extensions.chaos.native_generation_required)
    {
        const auto backend = config.extensions.compute.backend;
        if ((backend != ComputeBackend::GPU && backend != ComputeBackend::Metal) ||
            (!exact_qualified_domain_recycling_profile(config) &&
             !exact_qualified_earth_fixed_terrain_motion_profile(config) &&
             !is_public_qualified_accelerated_two_way_profile(config)))
            throw std::invalid_argument("extensions.chaos.native_generation_required is admitted only for the exact "
                                        "qualified Vulkan or Metal domain-recycling profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.grid.staggering == StaggerType::Collocated &&
        !tmv::runtime::is_exact_qualified_agrid_core_profile(config) &&
        !tmv::runtime::is_public_ready_flat_agrid_cpu_family(config) && !exact_qualified_coupled_rk_profile(config) &&
        !is_public_qualified_collocated_mapped_terrain_profile(config))
    {
        throw std::invalid_argument(
            "qualified Cartesian A-grid execution is limited to the exact flat/native complete or mapped CPU reference profiles");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::LinearHeight &&
        !tmv::runtime::is_exact_qualified_variable_diffusion_profile(config))
    {
        throw std::invalid_argument(
            "qualified linear-height variable diffusion is limited to the exact flat native reference profiles");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified
        && config.microphysics.scheme == MicrophysicsScheme::Kessler
        && !valid_public_qualified_kessler_profile(config)
        && !exact_simple_grey_absorbing_water_cloud_profile(config)
        && !exact_qualified_coupled_rk_profile(config)
        && !is_public_qualified_two_way_nested_profile(config))
    {
        throw std::invalid_argument(
            "microphysics.scheme=kessler is qualified only in an exact public reference profile");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        (config.radar.enabled || config.output.preset == OutputPreset::RadarReady ||
         config.output.preset == OutputPreset::RadarPolarimetricReady) &&
        !exact_qualified_any_radar_profile(config))
    {
        throw std::invalid_argument(
            "qualified radar execution is limited to exact maintained CPU radar profiles");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified
        && config.microphysics.scheme == MicrophysicsScheme::Lin
        && !tmv::runtime::is_flat_lin_storm_profile(config)
        && !is_public_qualified_lin_rrtmg_profile(config)
        && !is_public_qualified_two_way_nested_profile(config)
        && !exact_qualified_coupled_rk_profile(config)
        && !tmv::runtime::is_public_ready_flat_agrid_cpu_family(config)
        && !is_public_qualified_mapped_terrain_profile(config)
        && !exact_qualified_earth_fixed_terrain_motion_profile(config))
    {
        throw std::invalid_argument(
            "microphysics.scheme=lin is qualified only for maintained flat, RK, mapped, Earth-fixed, or stationary two-way nesting profiles with declared backend ownership");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.microphysics.scheme == MicrophysicsScheme::Thompson &&
        !is_public_qualified_thompson_cpu_profile(config) && !is_public_ready_thompson_native_profile(config) &&
        !is_public_ready_private_two_way_profile(config) && !exact_qualified_coupled_rk_profile(config) &&
        !tmv::runtime::is_public_ready_flat_agrid_cpu_family(config) &&
        !is_public_qualified_mapped_terrain_profile(config))
    {
        throw std::invalid_argument(
            "microphysics.scheme=thompson requires a maintained CPU profile or the native C-grid Thompson reference");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        config.microphysics.scheme == MicrophysicsScheme::Milbrandt &&
        config.extensions.compute.backend != ComputeBackend::CPU &&
        !is_public_ready_milbrandt_reference_profile(config) && !is_public_ready_private_two_way_profile(config) &&
        !exact_qualified_coupled_rk_profile(config))
        throw std::invalid_argument(
            "microphysics.scheme=milbrandt requires the maintained native C-grid reference profile");
    if (config.output.format == OutputFormat::Netcdf)
    {
#ifndef HAVE_NETCDF
        throw std::invalid_argument(
            "output.format=netcdf is unavailable in this binary; rebuild with libnetcdf or use npy_3d");
#else
        if (config.output.netcdf_deflate_level < 0 || config.output.netcdf_deflate_level > 9)
        {
            throw std::invalid_argument("output.netcdf_deflate_level must be in [0, 9]");
        }
        if (config.output.netcdf_target_chunk_kib <= 0)
        {
            throw std::invalid_argument("output.netcdf_target_chunk_kib must be positive");
        }
#endif
    }
}

void require_runtime_supported_domain_motion(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    if (config.terrain.horizontal_placement == TerrainHorizontalPlacement::EarthFixed
        && (motion.frame_velocity.u_ms != 0.0 || motion.frame_velocity.v_ms != 0.0)
        && !exact_qualified_earth_fixed_terrain_motion_profile(config))
        throw std::invalid_argument(
            "continuous Earth-fixed motion requires the maintained Bell remapping profile");
    const auto& nesting = motion.nesting;
    if (nesting.parent_ratio_x < 2 || nesting.parent_ratio_y < 2)
        throw std::invalid_argument("domain_motion.nesting horizontal parent ratios must be at least 2");
    if (nesting.parent_time_step_ratio < std::max(nesting.parent_ratio_x, nesting.parent_ratio_y))
    {
        throw std::invalid_argument(
            "domain_motion.nesting.parent_time_step_ratio must be at least the largest horizontal parent ratio");
    }
    if (nesting.enabled)
    {
        if (nesting.child_cells_x <= 0 || nesting.child_cells_y <= 0 ||
            nesting.relaxation_width_child_cells <= 0 ||
            nesting.boundary_stencil_parent_cells <= 0 ||
            !std::isfinite(nesting.relaxation_e_folding_time_s) ||
            nesting.relaxation_e_folding_time_s <= 0.0)
        {
            throw std::invalid_argument(
                "enabled domain_motion.nesting requires positive child dimensions, relaxation width, "
                "stencil margin, and relaxation e-folding time");
        }
        if (nesting.child_cells_x % nesting.parent_ratio_x != 0 ||
            nesting.child_cells_y % nesting.parent_ratio_y != 0)
        {
            throw std::invalid_argument(
                "enabled domain_motion.nesting child dimensions must span whole parent cells");
        }
        if (2LL * nesting.relaxation_width_child_cells >=
            std::min(nesting.child_cells_x, nesting.child_cells_y))
        {
            throw std::invalid_argument(
                "enabled domain_motion.nesting relaxation zones must leave a non-empty child interior");
        }
        if (nesting.feedback == NestingFeedback::TwoWay &&
            !is_public_qualified_two_way_nested_profile(config))
        {
            throw std::invalid_argument(
                "domain_motion.nesting.feedback=two_way is admitted only for the maintained stationary "
                "dry/Kessler/Lin/Thompson/Milbrandt "
                "flat 2:1 and Bell/Schär 2:1/3:1 TVD/MC or WENO CPU/Metal/Vulkan references, or the maintained flat "
                "moving-child and tracked-window references, with complete interface-register, native-dual "
                "restriction, and local reflux ownership");
        }
    }
    if (motion.mode == DomainMotionMode::TrackedWindow)
    {
        if (!motion.tracking.enabled || !motion.tracking.recycle_domain
            || motion.tracking.maximum_shift_cells <= 0)
            throw std::invalid_argument(
                "domain_motion.mode=tracked_window requires enabled tracking, live recycling, and a positive shift limit");
        if (motion.frame_velocity.u_ms != 0.0
            || motion.frame_velocity.v_ms != 0.0)
            throw std::invalid_argument(
                "domain_motion.mode=tracked_window requires zero continuous frame velocity");
        if (config.extensions.execution.mode != ExecutionMode::Exploratory &&
            !exact_qualified_domain_recycling_profile(config) &&
            !is_public_qualified_nested_grid_recycling_profile(config) &&
            !exact_qualified_earth_fixed_terrain_recycling_profile(config) &&
            !is_public_qualified_accelerated_two_way_profile(config))
            throw std::invalid_argument(
                "domain_motion.mode=tracked_window is not reference-qualified; use exploratory execution");
    }
    if (motion.mode == DomainMotionMode::MovingNest)
    {
        const bool admitted = internal_moving_nest_qualification_enabled()
            ? valid_internal_moving_nest_campaign(config)
            : valid_public_qualified_moving_nest_profile(config);
        if (!admitted)
        {
            throw std::invalid_argument(
                internal_moving_nest_qualification_enabled()
                    ? "the internal moving-nest campaign requires its exact versioned CPU or fallback-off Vulkan/Metal exploratory supercell profile"
                    : "public moving_nest requires an exact maintained CPU or fallback-off Vulkan/Metal profile");
        }
    }
    if (!std::isfinite(motion.frame_velocity.u_ms) || !std::isfinite(motion.frame_velocity.v_ms))
    {
        throw std::invalid_argument("domain_motion.frame_velocity components must be finite");
    }
    if (motion.mode == DomainMotionMode::Stationary &&
        (motion.frame_velocity.u_ms != 0.0 || motion.frame_velocity.v_ms != 0.0))
    {
        throw std::invalid_argument("domain_motion.mode=stationary requires zero frame velocity");
    }
    const auto& planning = motion.lifetime_planning;
    const bool  planning_finite =
        std::isfinite(planning.explicit_u_ms) && std::isfinite(planning.explicit_v_ms) &&
        std::isfinite(planning.upstream_buffer_m) && std::isfinite(planning.downstream_buffer_m) &&
        std::isfinite(planning.cross_track_buffer_m) && std::isfinite(planning.motion_uncertainty_ms);
    if (!planning_finite)
        throw std::invalid_argument("domain_motion.lifetime_planning values must be finite");
    if (planning.motion_uncertainty_ms < 0.0)
        throw std::invalid_argument("domain_motion.lifetime_planning.motion_uncertainty_ms must be non-negative");
    if (planning.enabled && (planning.upstream_buffer_m <= 0.0 || planning.downstream_buffer_m <= 0.0 ||
                             planning.cross_track_buffer_m <= 0.0))
    {
        throw std::invalid_argument("enabled domain_motion.lifetime_planning requires positive "
                                    "upstream, downstream, and cross-track buffers");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified && planning.enabled &&
        planning.action != LifetimePlanningAction::Error)
    {
        throw std::invalid_argument("qualified execution requires domain_motion.lifetime_planning.action=error");
    }
    const auto& tracking = motion.tracking;
    const bool  tracking_finite =
        std::isfinite(tracking.interval_s) && std::isfinite(tracking.inner_margin_m) &&
        std::isfinite(tracking.cooldown_s) && std::isfinite(tracking.minimum_confidence) &&
        std::isfinite(tracking.minimum_updraft_ms) && std::isfinite(tracking.minimum_condensate_path_kgm2) &&
        std::isfinite(tracking.updraft_helicity_bottom_m_agl) &&
        std::isfinite(tracking.updraft_helicity_top_m_agl) &&
        std::isfinite(tracking.minimum_absolute_updraft_helicity_m2s2) &&
        std::isfinite(tracking.minimum_core_area_m2) &&
        std::isfinite(tracking.maximum_motion_speed_ms) && std::isfinite(tracking.maximum_motion_acceleration_ms2);
    if (!tracking_finite)
    {
        throw std::invalid_argument("domain_motion.tracking values must be finite");
    }
    if (tracking.minimum_continuity_samples < 2 || tracking.minimum_continuity_samples > 5)
    {
        throw std::invalid_argument("domain_motion.tracking.minimum_continuity_samples must be in [2, 5]");
    }
    if (tracking.enabled && (tracking.interval_s <= 0.0 || tracking.inner_margin_m <= 0.0 ||
                             tracking.minimum_confidence <= 0.0 || tracking.minimum_confidence > 1.0 ||
                             tracking.minimum_updraft_ms <= 0.0 || tracking.minimum_condensate_path_kgm2 <= 0.0 ||
                             tracking.updraft_helicity_bottom_m_agl < 0.0 ||
                             tracking.updraft_helicity_top_m_agl <= tracking.updraft_helicity_bottom_m_agl ||
                             tracking.minimum_absolute_updraft_helicity_m2s2 <= 0.0 ||
                             tracking.minimum_component_cells <= 0 || tracking.minimum_core_area_m2 <= 0.0 ||
                             tracking.maximum_motion_speed_ms <= 0.0 ||
                             tracking.maximum_motion_acceleration_ms2 < 0.0 || tracking.lost_after_samples <= 0))
    {
        throw std::invalid_argument("enabled domain_motion.tracking requires positive interval, inner "
                                    "margin, confidence, signal/motion thresholds, component size, "
                                    "2-5 continuity samples, and a positive lost-track count");
    }
    if (tracking.recycle_domain &&
        (!tracking.enabled || tracking.maximum_shift_cells <= 0))
    {
        throw std::invalid_argument(
            "domain_motion.tracking.recycle_domain requires enabled tracking "
            "and maximum_shift_cells > 0");
    }
    if (config.extensions.execution.mode == ExecutionMode::Qualified &&
        tracking.enabled &&
        !valid_public_qualified_moving_nest_profile(config) &&
        !exact_qualified_domain_recycling_profile(config) &&
        !is_public_qualified_nested_grid_recycling_profile(config) &&
        !exact_qualified_earth_fixed_terrain_recycling_profile(config))
    {
        throw std::invalid_argument("domain_motion.tracking diagnostic evidence is not reference-qualified");
    }
}

constexpr double default_parametric_qv_kgkg = 0.014;

double sounding_qv_or_default(const SoundingSubsection& sounding)
{
    return sounding.moisture_qv_kgkg.value_or(default_parametric_qv_kgkg);
}

void require_runtime_supported_sounding(const ConfigV1& config)
{
    const auto& s             = config.initial_conditions.sounding;
    const auto  require_range = [](const char* key, double value, double minimum, double maximum)
    {
        if (value >= minimum && value <= maximum)
            return;
        std::ostringstream oss;
        oss << key << '=' << value << " is outside the runtime-supported [" << minimum << ", " << maximum
            << "] range and would be silently clamped";
        throw std::invalid_argument(oss.str());
    };

    switch (s.type)
    {
    case SoundingType::ParametricCAPE:
        require_range("initial_conditions.sounding.surface_theta_k",
                      s.surface_theta_k,
                      250.0,
                      std::numeric_limits<double>::max());
        require_range(
            "initial_conditions.sounding.tropopause_m", s.tropopause_m, 8000.0, std::numeric_limits<double>::max());
        if (sounding_qv_or_default(s) != 0.0)
            require_range("initial_conditions.sounding.moisture_qv_kgkg", sounding_qv_or_default(s), 0.004, 0.024);
        return;

    case SoundingType::ParametricTargets:
        require_range("initial_conditions.sounding.surface_theta_k",
                      s.surface_theta_k,
                      250.0,
                      std::numeric_limits<double>::max());
        require_range("initial_conditions.sounding.cin_jkg", s.cin_jkg, 0.0, 300.0);
        require_range("initial_conditions.sounding.el_m", s.el_m, 8000.0, std::numeric_limits<double>::max());
        require_range("initial_conditions.sounding.lfc_m", s.lfc_m, 50.0, s.el_m - 1000.0);
        if (s.moisture_qv_kgkg.has_value())
        {
            require_range("initial_conditions.sounding.moisture_qv_kgkg", *s.moisture_qv_kgkg, 0.004, 0.024);
        }
        else
        {
            require_range("initial_conditions.sounding.lcl_m", s.lcl_m, 1.0, std::numeric_limits<double>::max());
            const double derived_qv = tmv::init::parametric_target_qv_from_lcl(s.lcl_m, s.surface_theta_k, 100000.0);
            require_range("initial_conditions.sounding.lcl_m translated qv", derived_qv, 0.004, 0.024);
        }
        return;

    case SoundingType::File:
        if (s.file_path.empty())
        {
            throw std::invalid_argument("initial_conditions.sounding.file_path must not be empty "
                                        "for sounding.type=file");
        }
        return;
    }
}

void require_runtime_supported_configuration(const ConfigV1& c)
{
    if (c.extensions.compute.cuda_assist && c.extensions.compute.backend != ComputeBackend::GPU)
        throw std::invalid_argument("extensions.compute.cuda_assist requires the Vulkan backend on a compatible NVIDIA GPU");
    require_runtime_supported_diffusion(c);
    require_runtime_supported_domain_motion(c);
    require_runtime_supported_enum_selections(c);
    require_runtime_supported_sounding(c);
    if (c.surface.scheme == SurfaceScheme::None && c.surface.apply_fluxes)
    {
        throw std::invalid_argument("surface.apply_fluxes cannot be true when surface.scheme=none");
    }
    if (c.surface.scheme == SurfaceScheme::Slab && c.boundary_layer.scheme == BoundaryLayerScheme::None)
    {
        throw std::invalid_argument("boundary_layer.scheme cannot be none when surface.scheme=slab; "
                                    "select boundary_layer.scheme=slab explicitly");
    }
    tmv::require_physics_cadence(
        c.radiation.cadence_s / static_cast<double>(c.radiation.use_subcycling ? c.radiation.subcycle_count : 1),
        "radiation update interval");
    if (!tmv::valid_physics_cadence(c.boundary_layer.cadence_s))
        throw std::invalid_argument(
            "boundary_layer.cadence_s must exceed the 1e-12 s scheduling tolerance");
    if (!c.boundary_layer.clear_sky_only &&
        c.boundary_layer.scheme != BoundaryLayerScheme::YSU)
        throw std::invalid_argument(
            "boundary_layer.clear_sky_only=false requires boundary_layer.scheme=ysu");

}

} // namespace tmv::init::config
