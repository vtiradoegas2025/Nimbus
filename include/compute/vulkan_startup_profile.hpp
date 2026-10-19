#pragma once

#include "compute/kernel_startup_plan.hpp"
#include "init/config/config_v1.hpp"

#include <stdexcept>

namespace nimbus::compute
{

namespace startup_config = tmv::init::config;

inline bool is_flat_dry_cgrid18_vulkan_startup_profile(const tmv::init::config::ConfigV1& config)
{
    return config.extensions.compute.backend == startup_config::ComputeBackend::GPU &&
           !config.extensions.compute.allow_fallback && !config.extensions.compute.validate_parity &&
           !config.extensions.compute.gpu_apply_microphysics && !config.extensions.compute.gpu_apply_slow &&
           !config.extensions.compute.gpu_apply_sgs && config.grid.coordinate_system == CoordinateSystem::Cartesian &&
           config.grid.staggering == StaggerType::CGrid && config.grid.nx == 18 && config.grid.ny == 18 &&
           config.grid.nz == 18 && config.grid.dx == 500.0 && config.grid.dy == 500.0 && config.grid.dz == 250.0 &&
           config.grid.ztop == 4500.0 && config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
           !config.dynamics.axisymmetric && config.time.integrator == startup_config::TimeIntegrator::SplitExplicit &&
           config.time.dt_large == 0.1 && config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
           config.advection.scheme == startup_config::AdvectionScheme::TVD &&
           config.diffusion.scheme == startup_config::DiffusionScheme::Explicit &&
           config.diffusion.operator_order == 2 && config.diffusion.K_h == 0.0 && config.diffusion.K_v == 0.0 &&
           config.diffusion.coefficient_source == startup_config::DiffusionCoefficientSource::Constant &&
           config.diffusion.apply_to == startup_config::DiffusionTarget::Scalars &&
           config.microphysics.scheme == startup_config::MicrophysicsScheme::None &&
           config.turbulence.scheme == startup_config::TurbulenceScheme::None &&
           config.boundary_layer.scheme == startup_config::BoundaryLayerScheme::None &&
           config.radiation.scheme == startup_config::RadiationScheme::None &&
           config.surface.scheme == startup_config::SurfaceScheme::None && !config.surface.apply_fluxes &&
           config.terrain.scheme == startup_config::TerrainScheme::None &&
           config.domain_motion.mode == startup_config::DomainMotionMode::Stationary &&
           config.domain_motion.frame_velocity.u_ms == 0.0 && config.domain_motion.frame_velocity.v_ms == 0.0 &&
           !config.domain_motion.lifetime_planning.enabled && !config.domain_motion.tracking.enabled &&
           !config.domain_motion.tracking.recycle_domain && !config.domain_motion.nesting.enabled &&
           config.domain_motion.nesting.feedback == startup_config::NestingFeedback::OneWay &&
           config.boundary_conditions.lateral.west == startup_config::LateralBC::RigidSlip &&
           config.boundary_conditions.lateral.east == startup_config::LateralBC::RigidSlip &&
           config.boundary_conditions.lateral.south == startup_config::LateralBC::RigidSlip &&
           config.boundary_conditions.lateral.north == startup_config::LateralBC::RigidSlip &&
           config.boundary_conditions.bottom == startup_config::BottomBC::RigidNoSlip &&
           config.boundary_conditions.top == startup_config::TopBC::Open && !config.damping.enabled &&
           config.damping.lateral_sponge_enabled && config.damping.lateral_width == 20 &&
           config.damping.lateral_alpha == 0.05 && config.damping.w_only_top_sponge &&
           config.damping.w_only_depth_cells == 8 && config.damping.w_only_alpha_max == 0.20 &&
           !config.coriolis.enabled &&
           config.extensions.chaos.scheme == startup_config::ChaosScheme::InitialConditions &&
           !config.extensions.chaos.native_generation_required && config.extensions.solver.slow_rk2 &&
           config.extensions.solver.lid_bc_gamma == -1.0 && !config.extensions.solver.axis_u_m1_filter &&
           !config.radar.enabled && !config.output.trajectories && !config.output.live_shm &&
           config.output.preset == startup_config::OutputPreset::Minimal &&
           config.output.format == startup_config::OutputFormat::Npy3D && !config.output.async_io &&
           config.extensions.execution.mode == startup_config::ExecutionMode::Qualified;
}

inline std::shared_ptr<const KernelStartupPlan>
vulkan_startup_plan_for_profile(const tmv::init::config::ConfigV1& config)
{
    if (!is_flat_dry_cgrid18_vulkan_startup_profile(config))
    {
        return {};
    }
    const std::vector<KernelStartupOwner> owners = {KernelStartupOwner::FlatCartesianCgridDynamics,
                                                    KernelStartupOwner::CartesianScalarTransport,
                                                    KernelStartupOwner::CartesianExplicitDiffusion};
    std::vector<KernelId>                 kernels;
    for (const auto owner : owners)
    {
        const auto dependencies = startup_owner_kernels(owner);
        kernels.insert(kernels.end(), dependencies.begin(), dependencies.end());
    }
    std::string error;
    auto        plan = KernelStartupPlan::make("flat-dry-cgrid18-v1", owners, kernels, error);
    if (!plan)
    {
        throw std::runtime_error("Vulkan startup plan failed: " + error);
    }
    return plan;
}

} // namespace nimbus::compute
