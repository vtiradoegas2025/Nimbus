#pragma once

#include "init/config/config_v1.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tmv::runtime
{

inline bool qualified_variable_diffusion_number(double value, double expected)
{
    return std::isfinite(value) &&
        std::abs(value - expected) <=
            8.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(expected));
}

inline bool is_exact_qualified_variable_diffusion_profile(
    const tmv::init::config::ConfigV1& config)
{
    using namespace tmv::init::config;
    const auto backend = config.extensions.compute.backend;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        (backend == ComputeBackend::GPU || backend == ComputeBackend::Metal) &&
        !config.extensions.compute.allow_fallback &&
        !config.extensions.compute.gpu_apply_microphysics &&
        !config.extensions.compute.gpu_apply_slow &&
        !config.extensions.compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 12 && config.grid.ny == 12 && config.grid.nz == 12 &&
        qualified_variable_diffusion_number(config.grid.dx, 1000.0) &&
        qualified_variable_diffusion_number(config.grid.dy, 1000.0) &&
        qualified_variable_diffusion_number(config.grid.dz, 250.0) &&
        qualified_variable_diffusion_number(config.grid.ztop, 3000.0) &&
        qualified_variable_diffusion_number(config.time.run_time, 2.0) &&
        qualified_variable_diffusion_number(config.time.dt_large, 0.1) &&
        config.time.acoustic_substeps == 6 && !config.time.adaptive_dt &&
        qualified_variable_diffusion_number(config.time.cfl_safety, 0.7) &&
        config.time.output_interval == 1 &&
        config.domain_motion.mode == DomainMotionMode::Stationary &&
        !config.domain_motion.lifetime_planning.enabled &&
        !config.domain_motion.tracking.enabled &&
        !config.domain_motion.tracking.recycle_domain &&
        !config.domain_motion.nesting.enabled &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" &&
        !config.dynamics.axisymmetric &&
        config.advection.scheme == AdvectionScheme::TVD &&
        config.boundary_conditions.lateral.west == LateralBC::Periodic &&
        config.boundary_conditions.lateral.east == LateralBC::Periodic &&
        config.boundary_conditions.lateral.south == LateralBC::Periodic &&
        config.boundary_conditions.lateral.north == LateralBC::Periodic &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled && !config.damping.lateral_sponge_enabled &&
        !config.damping.w_only_top_sponge && !config.coriolis.enabled &&
        config.microphysics.scheme ==
            tmv::init::config::MicrophysicsScheme::None &&
        config.turbulence.scheme == TurbulenceScheme::None &&
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        qualified_variable_diffusion_number(config.diffusion.K_h, 7.0) &&
        qualified_variable_diffusion_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::ScalarsAndMomentum &&
        config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::LinearHeight &&
        qualified_variable_diffusion_number(config.diffusion.K_v_surface, 4.0) &&
        qualified_variable_diffusion_number(config.diffusion.K_v_top, 20.0) &&
        config.surface.scheme == SurfaceScheme::None &&
        !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == BoundaryLayerScheme::None &&
        config.boundary_layer.pbl_top_method ==
            BoundaryLayerTopMethod::Richardson &&
        config.radiation.scheme == RadiationScheme::None &&
        config.terrain.scheme == TerrainScheme::None &&
        config.extensions.chaos.scheme == ChaosScheme::None &&
        !config.radar.enabled && config.radar.products.empty() &&
        config.output.preset == OutputPreset::Minimal &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.extensions.checkpoint.every_s == 1 &&
        config.extensions.diagnostics.scientific_every_steps == 1 &&
        config.extensions.diagnostics.deterministic_reductions &&
        config.extensions.validation.guard_mode == ValidationGuardMode::Error;
}

} // namespace tmv::runtime
