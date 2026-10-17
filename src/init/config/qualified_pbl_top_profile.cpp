#include "qualified_pbl_top_profile.hpp"

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

bool exact_method(const ConfigV1& config)
{
    return
        (config.boundary_layer.scheme == BoundaryLayerScheme::YSU &&
         config.boundary_layer.pbl_top_method ==
             BoundaryLayerTopMethod::PotentialTemperature) ||
        (config.boundary_layer.scheme == BoundaryLayerScheme::MYNN &&
         config.boundary_layer.pbl_top_method == BoundaryLayerTopMethod::TKE);
}

bool exact_profile(const ConfigV1& config)
{
    const auto& motion = config.domain_motion;
    const auto& compute = config.extensions.compute;
    const auto& lateral = config.boundary_conditions.lateral;
    const auto& sounding = config.initial_conditions.sounding;
    const auto& hodograph = config.initial_conditions.hodograph;
    const auto& radiation = config.radiation;
    return config.extensions.execution.mode == ExecutionMode::Qualified &&
        compute.backend == ComputeBackend::CPU && !compute.allow_fallback &&
        !compute.validate_parity && !compute.gpu_apply_microphysics &&
        !compute.gpu_apply_slow && !compute.gpu_apply_sgs &&
        config.grid.coordinate_system == CoordinateSystem::Cartesian &&
        config.grid.staggering == StaggerType::CGrid &&
        config.grid.nx == 12 && config.grid.ny == 12 && config.grid.nz == 16 &&
        same_profile_number(config.grid.dx, 1000.0) &&
        same_profile_number(config.grid.dy, 1000.0) &&
        same_profile_number(config.grid.dz, 250.0) &&
        same_profile_number(config.grid.ztop, 4000.0) &&
        same_profile_number(config.time.run_time, 2.0) &&
        same_profile_number(config.time.dt_large, 0.2) &&
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
        config.advection.scheme == AdvectionScheme::TVD &&
        lateral.west == LateralBC::Periodic &&
        lateral.east == LateralBC::Periodic &&
        lateral.south == LateralBC::Periodic &&
        lateral.north == LateralBC::Periodic &&
        config.boundary_conditions.bottom == BottomBC::RigidSlip &&
        config.boundary_conditions.top == TopBC::Rigid &&
        !config.damping.enabled &&
        same_profile_number(config.damping.z_d, 4000.0) &&
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
        config.diffusion.scheme == DiffusionScheme::Explicit &&
        config.diffusion.operator_order == 2 &&
        same_profile_number(config.diffusion.K_h, 0.0) &&
        same_profile_number(config.diffusion.K_v, 0.0) &&
        config.diffusion.apply_to == DiffusionTarget::Scalars &&
        config.diffusion.coefficient_source ==
            DiffusionCoefficientSource::Constant &&
        config.surface.scheme == SurfaceScheme::Slab &&
        config.surface.apply_fluxes &&
        same_profile_number(config.surface.z0m_m, 0.1) &&
        same_profile_number(config.surface.z0h_m, 0.01) &&
        same_profile_number(config.surface.albedo, 0.2) &&
        same_profile_number(config.surface.emissivity, 0.96) &&
        same_profile_number(config.surface.tsk_k, 300.0) &&
        same_profile_number(config.surface.qsfc_kgkg, -1.0) &&
        exact_method(config) &&
        radiation.scheme == RadiationScheme::RRTMG &&
        radiation.solar_forcing == RadiationSolarForcing::IdealizedDiurnal &&
        same_profile_number(radiation.latitude_deg, 35.0) &&
        same_profile_number(radiation.declination_deg, 15.0) &&
        same_profile_number(radiation.local_solar_time_at_start_h, 12.0) &&
        radiation.do_lw && radiation.do_sw &&
        same_profile_number(radiation.cadence_s, 1.0) &&
        !radiation.clear_sky_only &&
        same_profile_number(radiation.rrtmg_ozone_vmr, 4.0e-6) &&
        same_profile_number(radiation.rrtmg_co2_vmr, 420.0e-6) &&
        same_profile_number(radiation.rrtmg_ch4_vmr, 1.9e-6) &&
        same_profile_number(radiation.rrtmg_n2o_vmr, 0.335e-6) &&
        same_profile_number(radiation.rrtmg_liquid_effective_radius_um, 10.0) &&
        same_profile_number(radiation.rrtmg_ice_effective_radius_um, 30.0) &&
        config.terrain.scheme == TerrainScheme::None &&
        sounding.type == SoundingType::ParametricCAPE &&
        same_profile_number(sounding.cape_jkg, 0.0) &&
        same_profile_number(sounding.tropopause_m, 8000.0) &&
        same_profile_number(sounding.surface_theta_k, 300.0) &&
        sounding.moisture_qv_kgkg.has_value() &&
        same_profile_number(*sounding.moisture_qv_kgkg, 0.008) &&
        hodograph.type == HodographType::WKParam &&
        same_profile_number(hodograph.u_sfc_ms, 5.0) &&
        same_profile_number(hodograph.v_sfc_ms, 2.0) &&
        same_profile_number(hodograph.u_1km_ms, 8.0) &&
        same_profile_number(hodograph.v_1km_ms, 4.0) &&
        same_profile_number(hodograph.u_6km_ms, 18.0) &&
        same_profile_number(hodograph.v_6km_ms, 10.0) &&
        config.initial_conditions.trigger.type == TriggerType::None &&
        config.output.preset == OutputPreset::Presentation &&
        config.output.format == OutputFormat::Npy3D &&
        !config.output.async_io && !config.output.live_shm &&
        config.output.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::None &&
        !config.output.trajectories && !config.radar.enabled &&
        config.radar.products.empty() &&
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

} // namespace

bool is_public_qualified_pbl_top_profile(const ConfigV1& config) noexcept
{
    return exact_profile(config);
}

} // namespace tmv::init::config
