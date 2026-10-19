#pragma once

#include "init/config/config_v1.hpp"

#include <cmath>

namespace tmv::runtime
{
inline bool is_flat_lin_storm_profile(const init::config::ConfigV1& config)
{
    namespace cfg = init::config;
    const auto& compute = config.extensions.compute;
    const bool native = compute.backend == cfg::ComputeBackend::GPU || compute.backend == cfg::ComputeBackend::Metal;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& lin = config.microphysics.parameters.lin;
    const auto& shared = config.microphysics.parameters.shared;
    return (native || compute.backend == cfg::ComputeBackend::CPU) && !compute.allow_fallback &&
        (!native || (compute.gpu_apply_microphysics && !compute.gpu_apply_slow && !compute.gpu_apply_sgs)) &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian && config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx >= 6 && config.grid.ny >= 6 && config.grid.nz >= 3 &&
        std::isfinite(config.grid.dx) && config.grid.dx > 0 &&
        std::isfinite(config.grid.dy) && config.grid.dy > 0 &&
        std::isfinite(config.grid.dz) && config.grid.dz > 0 &&
        config.time.integrator == cfg::TimeIntegrator::SplitExplicit && !config.time.adaptive_dt &&
        config.extensions.solver.slow_rk2 &&
        config.dynamics.scheme == "klemp_wilhelmson_split_cartesian_cgrid" && !config.dynamics.axisymmetric &&
        config.advection.scheme == cfg::AdvectionScheme::TVD &&
        config.microphysics.scheme == cfg::MicrophysicsScheme::Lin &&
        shared.cloud_water_threshold_kgkg == 0.001 && shared.autoconversion_rate_s_inv == 0.001 &&
        lin.accretion_coefficient == 2.2 && lin.evaporation_rate_s_inv == 0.001 &&
        config.turbulence.scheme == cfg::TurbulenceScheme::Smagorinsky &&
        config.turbulence.mode == cfg::TurbulenceMode::ThreeD && config.turbulence.filter_width == cfg::TurbulenceFilterWidth::CubicRoot &&
        !config.turbulence.dynamic_coefficient && config.turbulence.stability_correction == cfg::TurbulenceStabilityCorrection::None &&
        config.turbulence.Cs == 0.18 && config.turbulence.Pr_t == 1.0 / 3.0 && config.turbulence.Sc_t == 1.0 / 3.0 &&
        config.diffusion.scheme == cfg::DiffusionScheme::Explicit && config.diffusion.operator_order == 2 &&
        config.diffusion.coefficient_source == cfg::DiffusionCoefficientSource::Constant &&
        config.diffusion.K_h == 0 && config.diffusion.K_v == 0 && config.diffusion.apply_to == cfg::DiffusionTarget::Scalars &&
        config.surface.scheme == cfg::SurfaceScheme::None && !config.surface.apply_fluxes &&
        config.boundary_layer.scheme == cfg::BoundaryLayerScheme::None && config.radiation.scheme == cfg::RadiationScheme::None &&
        config.terrain.scheme == cfg::TerrainScheme::None && config.extensions.chaos.scheme == cfg::ChaosScheme::None &&
        config.domain_motion.mode == cfg::DomainMotionMode::Stationary && !config.domain_motion.nesting.enabled &&
        !config.domain_motion.tracking.enabled && !config.domain_motion.tracking.recycle_domain &&
        !config.domain_motion.lifetime_planning.enabled && !config.coriolis.enabled &&
        lateral.west == cfg::LateralBC::Radiative && lateral.east == cfg::LateralBC::Radiative &&
        lateral.south == cfg::LateralBC::Radiative && lateral.north == cfg::LateralBC::Radiative &&
        config.boundary_conditions.bottom == cfg::BottomBC::RigidSlip && config.boundary_conditions.top == cfg::TopBC::RigidWithSponge &&
        !config.radar.enabled && config.radar.products.empty() &&
        config.initial_conditions.sounding.type == cfg::SoundingType::File &&
        !config.initial_conditions.sounding.file_path.empty() &&
        config.extensions.validation.guard_mode == cfg::ValidationGuardMode::Error &&
        config.extensions.diagnostics.scientific_every_steps > 0 && config.extensions.diagnostics.deterministic_reductions;
}
}
