#include "util/numeric_text.hpp"
#include "init/config/v0_to_v1.hpp"

#include "config_v1_serialization_common.hpp"

#include "init/sounding/parametric_target_math.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace tmv::init::config
{

namespace
{

/// Render a double with enough precision for round-trip, no trailing zeros.
using tmv::strutil::format_double_roundtrip;

std::string fmt_bool(bool v)
{
    return v ? "true" : "false";
}

} // namespace

std::string serialize_v1_to_yaml(const ConfigV1& c)
{
    require_runtime_supported_configuration(c);
    std::ostringstream o;

    o << "# Nimbus v1 configuration (machine-generated)\n";
    o << "# Mirrors docs/CM1Mapping.md §20. CM1 namelist groups noted per section.\n";
    o << "schema: " << c.schema_version << "\n\n";

    o << "# CM1 &param0 / &param11 — Grid setup\n";
    o << "grid:\n";
    o << "  coordinate_system: " << coordinate_system_name(c.grid.coordinate_system) << "\n";
    o << "  staggering: " << stagger_type_name(c.grid.staggering) << "\n";
    o << "  nx: " << c.grid.nx << "\n";
    o << "  ny: " << c.grid.ny << "\n";
    o << "  nz: " << c.grid.nz << "\n";
    o << "  dx: " << format_double_roundtrip(c.grid.dx) << "\n";
    o << "  dy: " << format_double_roundtrip(c.grid.dy) << "\n";
    o << "  dz: " << format_double_roundtrip(c.grid.dz) << "\n";
    o << "  ztop: " << format_double_roundtrip(c.grid.ztop) << "\n\n";

    o << "# CM1 &param1 / &param2 — Time control + output cadence\n";
    o << "time:\n";
    o << "  integrator: " << time_integrator_name(c.time.integrator) << "\n";
    o << "  run_time: " << format_double_roundtrip(c.time.run_time) << "\n";
    o << "  dt_large: " << format_double_roundtrip(c.time.dt_large) << "\n";
    o << "  acoustic_substeps: " << c.time.acoustic_substeps << "\n";
    o << "  adaptive_dt: " << fmt_bool(c.time.adaptive_dt) << "\n";
    o << "  cfl_safety: " << format_double_roundtrip(c.time.cfl_safety) << "\n";
    o << "  output_interval: " << format_double_roundtrip(c.time.output_interval) << "\n\n";

    o << "# Nimbus horizontal domain-frame and lifetime-planning contract\n";
    o << "domain_motion:\n";
    o << "  mode: " << domain_motion_mode_name(c.domain_motion.mode) << "\n";
    o << "  frame_velocity:\n";
    o << "    u_ms: " << format_double_roundtrip(c.domain_motion.frame_velocity.u_ms) << "\n";
    o << "    v_ms: " << format_double_roundtrip(c.domain_motion.frame_velocity.v_ms) << "\n";
    o << "  lifetime_planning:\n";
    o << "    enabled: " << fmt_bool(c.domain_motion.lifetime_planning.enabled) << "\n";
    o << "    action: " << lifetime_planning_action_name(c.domain_motion.lifetime_planning.action) << "\n";
    o << "    motion_source: " << storm_motion_source_name(c.domain_motion.lifetime_planning.motion_source) << "\n";
    o << "    explicit_u_ms: " << format_double_roundtrip(c.domain_motion.lifetime_planning.explicit_u_ms) << "\n";
    o << "    explicit_v_ms: " << format_double_roundtrip(c.domain_motion.lifetime_planning.explicit_v_ms) << "\n";
    o << "    upstream_buffer_m: " << format_double_roundtrip(c.domain_motion.lifetime_planning.upstream_buffer_m) << "\n";
    o << "    downstream_buffer_m: " << format_double_roundtrip(c.domain_motion.lifetime_planning.downstream_buffer_m) << "\n";
    o << "    cross_track_buffer_m: " << format_double_roundtrip(c.domain_motion.lifetime_planning.cross_track_buffer_m) << "\n";
    o << "    motion_uncertainty_ms: " << format_double_roundtrip(c.domain_motion.lifetime_planning.motion_uncertainty_ms) << "\n";
    o << "  tracking:\n";
    o << "    enabled: " << fmt_bool(c.domain_motion.tracking.enabled) << "\n";
    o << "    recycle_domain: " << fmt_bool(c.domain_motion.tracking.recycle_domain) << "\n";
    o << "    interval_s: " << format_double_roundtrip(c.domain_motion.tracking.interval_s) << "\n";
    o << "    inner_margin_m: " << format_double_roundtrip(c.domain_motion.tracking.inner_margin_m) << "\n";
    o << "    maximum_shift_cells: " << c.domain_motion.tracking.maximum_shift_cells << "\n";
    o << "    cooldown_s: " << format_double_roundtrip(c.domain_motion.tracking.cooldown_s) << "\n";
    o << "    minimum_confidence: " << format_double_roundtrip(c.domain_motion.tracking.minimum_confidence) << "\n";
    o << "    minimum_updraft_ms: " << format_double_roundtrip(c.domain_motion.tracking.minimum_updraft_ms) << "\n";
    o << "    minimum_condensate_path_kgm2: " << format_double_roundtrip(c.domain_motion.tracking.minimum_condensate_path_kgm2)
      << "\n";
    o << "    updraft_helicity_bottom_m_agl: " << format_double_roundtrip(c.domain_motion.tracking.updraft_helicity_bottom_m_agl)
      << "\n";
    o << "    updraft_helicity_top_m_agl: " << format_double_roundtrip(c.domain_motion.tracking.updraft_helicity_top_m_agl) << "\n";
    o << "    minimum_absolute_updraft_helicity_m2s2: "
      << format_double_roundtrip(c.domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2) << "\n";
    o << "    minimum_component_cells: " << c.domain_motion.tracking.minimum_component_cells << "\n";
    o << "    minimum_core_area_m2: " << format_double_roundtrip(c.domain_motion.tracking.minimum_core_area_m2) << "\n";
    o << "    maximum_motion_speed_ms: " << format_double_roundtrip(c.domain_motion.tracking.maximum_motion_speed_ms) << "\n";
    o << "    maximum_motion_acceleration_ms2: " << format_double_roundtrip(c.domain_motion.tracking.maximum_motion_acceleration_ms2)
      << "\n";
    o << "    minimum_continuity_samples: " << c.domain_motion.tracking.minimum_continuity_samples << "\n";
    o << "    lost_after_samples: " << c.domain_motion.tracking.lost_after_samples << "\n";
    o << "    topology_confirmation_samples: " << c.domain_motion.tracking.topology_confirmation_samples << "\n";
    o << "    branch_preference: " << storm_branch_preference_name(c.domain_motion.tracking.branch_preference) << "\n";
    o << "    branch_velocity_uncertainty_ms: " << format_double_roundtrip(c.domain_motion.tracking.branch_velocity_uncertainty_ms)
      << "\n";
    o << "  nesting:\n";
    o << "    enabled: " << fmt_bool(c.domain_motion.nesting.enabled) << "\n";
    o << "    parent_ratio_x: " << c.domain_motion.nesting.parent_ratio_x << "\n";
    o << "    parent_ratio_y: " << c.domain_motion.nesting.parent_ratio_y << "\n";
    o << "    parent_time_step_ratio: " << c.domain_motion.nesting.parent_time_step_ratio << "\n";
    o << "    child_origin_parent_cell_i: " << c.domain_motion.nesting.child_origin_parent_cell_i << "\n";
    o << "    child_origin_parent_cell_j: " << c.domain_motion.nesting.child_origin_parent_cell_j << "\n";
    o << "    child_cells_x: " << c.domain_motion.nesting.child_cells_x << "\n";
    o << "    child_cells_y: " << c.domain_motion.nesting.child_cells_y << "\n";
    o << "    relaxation_width_child_cells: "
      << c.domain_motion.nesting.relaxation_width_child_cells << "\n";
    o << "    boundary_stencil_parent_cells: "
      << c.domain_motion.nesting.boundary_stencil_parent_cells << "\n";
    o << "    relaxation_e_folding_time_s: "
      << format_double_roundtrip(c.domain_motion.nesting.relaxation_e_folding_time_s) << "\n";
    o << "    feedback: " << nesting_feedback_name(c.domain_motion.nesting.feedback) << "\n\n";

    o << "# Dynamics core (one solver; CM1 calls this Klemp-Wilhelmson split-explicit)\n";
    o << "dynamics:\n";
    o << "  scheme: " << c.dynamics.scheme << "\n";
    o << "  axisymmetric: " << fmt_bool(c.dynamics.axisymmetric) << "\n\n";

    o << "# CM1 &param3 — Boundary conditions\n";
    o << "boundary_conditions:\n";
    o << "  lateral:\n";
    o << "    west: "
      << lateral_bc_name(resolve_legacy_lateral_bc(c.boundary_conditions.lateral.west, c.grid.staggering)) << "\n";
    o << "    east: "
      << lateral_bc_name(resolve_legacy_lateral_bc(c.boundary_conditions.lateral.east, c.grid.staggering)) << "\n";
    o << "    south: "
      << lateral_bc_name(resolve_legacy_lateral_bc(c.boundary_conditions.lateral.south, c.grid.staggering)) << "\n";
    o << "    north: "
      << lateral_bc_name(resolve_legacy_lateral_bc(c.boundary_conditions.lateral.north, c.grid.staggering)) << "\n";
    o << "  bottom: " << bottom_bc_name(c.boundary_conditions.bottom) << "\n";
    o << "  top: " << top_bc_name(c.boundary_conditions.top) << "\n\n";

    o << "# CM1 &param7 — Rayleigh damping (top sponge)\n";
    o << "damping:\n";
    o << "  enabled: " << fmt_bool(c.damping.enabled) << "\n";
    o << "  z_d: " << format_double_roundtrip(c.damping.z_d) << "\n";
    o << "  alpha: " << format_double_roundtrip(c.damping.alpha) << "\n";
    o << "  mode: " << damping_mode_name(c.damping.mode) << "\n\n";

    o << "# CM1 &param12 — Coriolis\n";
    o << "coriolis:\n";
    o << "  enabled: " << fmt_bool(c.coriolis.enabled) << "\n";
    o << "  mode: " << coriolis_mode_name(c.coriolis.mode) << "\n";
    o << "  latitude_deg: " << format_double_roundtrip(c.coriolis.latitude_deg) << "\n\n";

    o << "# CM1 &param4 — Microphysics\n";
    o << "microphysics:\n";
    o << "  scheme: " << microphysics_scheme_name(c.microphysics.scheme) << "\n";
    if (c.microphysics.scheme != MicrophysicsScheme::None && c.microphysics.scheme != MicrophysicsScheme::Milbrandt)
    {
        o << "  shared:\n";
        o << "    cloud_water_threshold_kgkg: "
          << format_double_roundtrip(c.microphysics.parameters.shared.cloud_water_threshold_kgkg) << "\n";
        o << "    autoconversion_rate_s_inv: " << format_double_roundtrip(c.microphysics.parameters.shared.autoconversion_rate_s_inv)
          << "\n";
    }
    if (c.microphysics.scheme == MicrophysicsScheme::Kessler)
    {
        o << "  kessler:\n";
        o << "    accretion_coefficient: " << format_double_roundtrip(c.microphysics.parameters.kessler.accretion_coefficient)
          << "\n";
        o << "    evaporation_efficiency: " << format_double_roundtrip(c.microphysics.parameters.kessler.evaporation_efficiency)
          << "\n";
    }
    else if (c.microphysics.scheme == MicrophysicsScheme::Lin)
    {
        o << "  lin:\n";
        o << "    accretion_coefficient: " << format_double_roundtrip(c.microphysics.parameters.lin.accretion_coefficient) << "\n";
        o << "    evaporation_rate_s_inv: " << format_double_roundtrip(c.microphysics.parameters.lin.evaporation_rate_s_inv) << "\n";
    }
    else if (c.microphysics.scheme == MicrophysicsScheme::Thompson)
    {
        o << "  thompson:\n";
        o << "    evaporation_rate_s_inv: " << format_double_roundtrip(c.microphysics.parameters.thompson.evaporation_rate_s_inv)
          << "\n";
        o << "    water_friendly_aerosol_surface_m3: "
          << format_double_roundtrip(c.microphysics.parameters.thompson.water_friendly_aerosol_surface_m3)
          << "\n";
        o << "    ice_friendly_aerosol_surface_m3: "
          << format_double_roundtrip(c.microphysics.parameters.thompson.ice_friendly_aerosol_surface_m3)
          << "\n";
    }
    else if (c.microphysics.scheme == MicrophysicsScheme::Milbrandt)
    {
        o << "  milbrandt:\n";
        o << "    hail_processes: " << fmt_bool(c.microphysics.parameters.milbrandt.hail_processes) << "\n";
    }
    o << "\n";

    o << "# CM1 &param5 — SGS turbulence\n";
    o << "turbulence:\n";
    o << "  scheme: " << turbulence_scheme_name(c.turbulence.scheme) << "\n";
    o << "  mode: " << turbulence_mode_name(c.turbulence.mode) << "\n";
    o << "  filter_width: "
      << turbulence_filter_width_name(c.turbulence.filter_width) << "\n";
    o << "  user_filter_width_m: "
      << format_double_roundtrip(c.turbulence.user_filter_width_m) << "\n";
    o << "  dynamic_coefficient: "
      << fmt_bool(c.turbulence.dynamic_coefficient) << "\n";
    o << "  stability_correction: "
      << turbulence_stability_correction_name(
             c.turbulence.stability_correction) << "\n";
    o << "  Cs: " << format_double_roundtrip(c.turbulence.Cs) << "\n";
    o << "  Pr_t: " << format_double_roundtrip(c.turbulence.Pr_t) << "\n";
    o << "  Sc_t: " << format_double_roundtrip(c.turbulence.Sc_t) << "\n\n";

    o << "# Nimbus scalar face reconstruction\n";
    o << "advection:\n";
    o << "  scheme: " << advection_scheme_name(c.advection.scheme) << "\n\n";

    o << "# CM1 &param6 — Numerical diffusion\n";
    o << "diffusion:\n";
    o << "  scheme: " << diffusion_scheme_name(c.diffusion.scheme) << "\n";
    o << "  operator_order: " << c.diffusion.operator_order << "\n";
    o << "  K_h: " << format_double_roundtrip(c.diffusion.K_h) << "\n";
    o << "  K_v: " << format_double_roundtrip(c.diffusion.K_v) << "\n";
    o << "  apply_to: " << diffusion_target_name(c.diffusion.apply_to) << "\n";
    o << "  coefficient_source: "
      << diffusion_coefficient_source_name(c.diffusion.coefficient_source)
      << "\n";
    if (c.diffusion.coefficient_source == DiffusionCoefficientSource::LinearHeight)
    {
        o << "  K_v_surface: " << format_double_roundtrip(c.diffusion.K_v_surface) << "\n";
        o << "  K_v_top: " << format_double_roundtrip(c.diffusion.K_v_top) << "\n";
    }
    o << "\n";

    o << "# CM1 &param9 — Surface\n";
    o << "surface:\n";
    o << "  scheme: " << surface_scheme_name(c.surface.scheme) << "\n";
    o << "  apply_fluxes: " << fmt_bool(c.surface.apply_fluxes) << "\n";
    o << "  z0m_m: " << format_double_roundtrip(c.surface.z0m_m) << "\n";
    o << "  z0h_m: " << format_double_roundtrip(c.surface.z0h_m) << "\n";
    o << "  albedo: " << format_double_roundtrip(c.surface.albedo) << "\n";
    o << "  emissivity: " << format_double_roundtrip(c.surface.emissivity) << "\n";
    // Prescribed skin state (CM1 tsk0); emitted only when set, else the runtime
    // holds the initial surface-air state as the skin.
    if (c.surface.tsk_k > 0.0)
    {
        o << "  tsk_k: " << format_double_roundtrip(c.surface.tsk_k) << "\n";
    }
    if (c.surface.qsfc_kgkg >= 0.0)
    {
        o << "  qsfc_kgkg: " << format_double_roundtrip(c.surface.qsfc_kgkg) << "\n";
    }
    o << "\n";

    o << "# Boundary-layer mixing (Nimbus addition; not in CM1)\n";
    o << "boundary_layer:\n";
    o << "  scheme: " << boundary_layer_scheme_name(c.boundary_layer.scheme) << "\n";
    o << "  pbl_top_method: "
      << boundary_layer_top_method_name(c.boundary_layer.pbl_top_method)
      << "\n";
    o << "  cadence_s: " << format_double_roundtrip(c.boundary_layer.cadence_s) << "\n";
    o << "  clear_sky_only: "
      << fmt_bool(c.boundary_layer.clear_sky_only) << "\n\n";

    o << "# CM1 &param8 — Radiation with explicit idealized solar forcing\n";
    o << "radiation:\n";
    o << "  scheme: " << radiation_scheme_name(c.radiation.scheme) << "\n";
    o << "  solar_forcing: " << radiation_solar_forcing_name(c.radiation.solar_forcing) << "\n";
    o << "  latitude_deg: " << format_double_roundtrip(c.radiation.latitude_deg) << "\n";
    o << "  declination_deg: " << format_double_roundtrip(c.radiation.declination_deg) << "\n";
    o << "  local_solar_time_at_start_h: " << format_double_roundtrip(c.radiation.local_solar_time_at_start_h) << "\n";
    o << "  do_lw: " << fmt_bool(c.radiation.do_lw) << "\n";
    o << "  do_sw: " << fmt_bool(c.radiation.do_sw) << "\n";
    o << "  cadence_s: " << format_double_roundtrip(c.radiation.cadence_s) << "\n";
    o << "  clear_sky_only: " << fmt_bool(c.radiation.clear_sky_only) << "\n";
    if (c.radiation.scheme == RadiationScheme::SimpleGrey)
    {
        o << "  simple_grey_optical_model: "
          << c.radiation.simple_grey_optical_model << "\n";
        o << "  use_subcycling: "
          << fmt_bool(c.radiation.use_subcycling) << "\n";
        o << "  subcycle_count: " << c.radiation.subcycle_count << "\n";
        o << "  enable_diagnostics: "
          << fmt_bool(c.radiation.enable_diagnostics) << "\n";
    }
    o << "  tau_lw_ref: " << format_double_roundtrip(c.radiation.tau_lw_ref) << "\n";
    o << "  tau_sw_ref: " << format_double_roundtrip(c.radiation.tau_sw_ref) << "\n";
    o << "  n_lw: " << format_double_roundtrip(c.radiation.n_lw) << "\n";
    o << "  n_sw: " << format_double_roundtrip(c.radiation.n_sw) << "\n";
    if (c.radiation.scheme == RadiationScheme::SimpleGrey)
    {
        o << "  simple_grey_lw_vapor_absorption_m2kg: "
          << format_double_roundtrip(c.radiation.simple_grey_lw_vapor_absorption_m2kg) << "\n";
        o << "  simple_grey_lw_liquid_absorption_m2kg: "
          << format_double_roundtrip(c.radiation.simple_grey_lw_liquid_absorption_m2kg) << "\n";
        o << "  simple_grey_lw_ice_absorption_m2kg: "
          << format_double_roundtrip(c.radiation.simple_grey_lw_ice_absorption_m2kg) << "\n";
        o << "  simple_grey_sw_vapor_absorption_m2kg: "
          << format_double_roundtrip(c.radiation.simple_grey_sw_vapor_absorption_m2kg) << "\n";
        o << "  simple_grey_sw_liquid_absorption_m2kg: "
          << format_double_roundtrip(c.radiation.simple_grey_sw_liquid_absorption_m2kg) << "\n";
        o << "  simple_grey_sw_ice_absorption_m2kg: "
          << format_double_roundtrip(c.radiation.simple_grey_sw_ice_absorption_m2kg) << "\n";
    }
    if (c.radiation.scheme == RadiationScheme::RRTMG)
    {
        o << "  rrtmg_ozone_vmr: " << format_double_roundtrip(c.radiation.rrtmg_ozone_vmr) << "\n";
        o << "  rrtmg_co2_vmr: " << format_double_roundtrip(c.radiation.rrtmg_co2_vmr) << "\n";
        o << "  rrtmg_ch4_vmr: " << format_double_roundtrip(c.radiation.rrtmg_ch4_vmr) << "\n";
        o << "  rrtmg_n2o_vmr: " << format_double_roundtrip(c.radiation.rrtmg_n2o_vmr) << "\n";
        o << "  rrtmg_co_vmr: " << format_double_roundtrip(c.radiation.rrtmg_co_vmr) << "\n";
        o << "  rrtmg_o2_vmr: " << format_double_roundtrip(c.radiation.rrtmg_o2_vmr) << "\n";
        o << "  rrtmg_n2_vmr: " << format_double_roundtrip(c.radiation.rrtmg_n2_vmr) << "\n";
        o << "  rrtmg_cfc11_vmr: " << format_double_roundtrip(c.radiation.rrtmg_cfc11_vmr) << "\n";
        o << "  rrtmg_cfc12_vmr: " << format_double_roundtrip(c.radiation.rrtmg_cfc12_vmr) << "\n";
        o << "  rrtmg_cfc22_vmr: " << format_double_roundtrip(c.radiation.rrtmg_cfc22_vmr) << "\n";
        o << "  rrtmg_ccl4_vmr: " << format_double_roundtrip(c.radiation.rrtmg_ccl4_vmr) << "\n";
        o << "  rrtmg_liquid_effective_radius_um: "
          << format_double_roundtrip(c.radiation.rrtmg_liquid_effective_radius_um) << "\n";
        o << "  rrtmg_ice_effective_radius_um: "
          << format_double_roundtrip(c.radiation.rrtmg_ice_effective_radius_um) << "\n";
    }
    o << "  terrain_sw_cosine_nodes: "
      << c.radiation.terrain_sw_cosine_nodes << "\n";
    o << "  terrain_sw_azimuth_nodes: "
      << c.radiation.terrain_sw_azimuth_nodes << "\n";
    o << "  terrain_sw_maximum_bounces: "
      << c.radiation.terrain_sw_maximum_bounces << "\n";
    o << "  terrain_sw_relative_residual_tolerance: "
      << format_double_roundtrip(c.radiation.terrain_sw_relative_residual_tolerance)
      << "\n";
    o << "  terrain_sw_absolute_residual_tolerance_w: "
      << format_double_roundtrip(c.radiation.terrain_sw_absolute_residual_tolerance_w)
      << "\n\n";

    o << "# Terrain lower boundary and coordinate mapping\n";
    o << "terrain:\n";
    o << "  scheme: " << terrain_scheme_name(c.terrain.scheme) << "\n";
    o << "  coordinate: " << terrain_coordinate_name(c.terrain.coordinate) << "\n";
    o << "  horizontal_placement: "
      << terrain_horizontal_placement_name(c.terrain.horizontal_placement)
      << "\n";
    if (c.terrain.horizontal_placement == TerrainHorizontalPlacement::EarthFixed)
    {
        o << "  feature_center_earth_x_m: "
          << format_double_roundtrip(c.terrain.feature_center_earth_x_m) << "\n";
        o << "  feature_center_earth_y_m: "
          << format_double_roundtrip(c.terrain.feature_center_earth_y_m) << "\n";
    }
    o << "  compute_derivatives: " << fmt_bool(c.terrain.compute_derivatives) << "\n";
    o << "  compute_metrics: " << fmt_bool(c.terrain.compute_metrics) << "\n";
    o << "  smoothing_decay: " << format_double_roundtrip(c.terrain.smoothing_decay) << "\n";
    o << "  bell:\n";
    o << "    h0: " << format_double_roundtrip(c.terrain.bell.h0) << "\n";
    o << "    a: " << format_double_roundtrip(c.terrain.bell.a) << "\n";
    o << "    axisymmetric: " << fmt_bool(c.terrain.bell.axisymmetric) << "\n";
    o << "  schar:\n";
    o << "    H: " << format_double_roundtrip(c.terrain.schar.H) << "\n";
    o << "    a: " << format_double_roundtrip(c.terrain.schar.a) << "\n";
    o << "    ell: " << format_double_roundtrip(c.terrain.schar.ell) << "\n\n";

    o << "# CM1 iinit family — Initial conditions\n";
    o << "initial_conditions:\n";
    o << "  sounding:\n";
    const auto& sounding = c.initial_conditions.sounding;
    o << "    type: " << sounding_type_name(sounding.type) << "\n";
    switch (sounding.type)
    {
    case SoundingType::ParametricCAPE:
        o << "    cape_jkg: " << format_double_roundtrip(sounding.cape_jkg) << "\n";
        o << "    tropopause_m: " << format_double_roundtrip(sounding.tropopause_m) << "\n";
        o << "    surface_theta_k: " << format_double_roundtrip(sounding.surface_theta_k) << "\n";
        if (sounding.moisture_qv_kgkg.has_value())
        {
            o << "    moisture_qv_kgkg: " << format_double_roundtrip(*sounding.moisture_qv_kgkg) << "\n";
        }
        break;
    case SoundingType::ParametricTargets:
        o << "    cape_jkg: " << format_double_roundtrip(sounding.cape_jkg) << "\n";
        o << "    cin_jkg: " << format_double_roundtrip(sounding.cin_jkg) << "\n";
        if (sounding.moisture_qv_kgkg.has_value())
        {
            o << "    moisture_qv_kgkg: " << format_double_roundtrip(*sounding.moisture_qv_kgkg) << "\n";
        }
        else
        {
            o << "    lcl_m: " << format_double_roundtrip(sounding.lcl_m) << "\n";
        }
        o << "    lfc_m: " << format_double_roundtrip(sounding.lfc_m) << "\n";
        o << "    el_m: " << format_double_roundtrip(sounding.el_m) << "\n";
        o << "    surface_theta_k: " << format_double_roundtrip(sounding.surface_theta_k) << "\n";
        break;
    case SoundingType::File:
        o << "    file_path: " << sounding.file_path << "\n";
        break;
    }
    o << "  hodograph:\n";
    o << "    type: " << hodograph_type_name(c.initial_conditions.hodograph.type) << "\n";
    o << "    u_sfc_ms: " << format_double_roundtrip(c.initial_conditions.hodograph.u_sfc_ms) << "\n";
    o << "    v_sfc_ms: " << format_double_roundtrip(c.initial_conditions.hodograph.v_sfc_ms) << "\n";
    o << "    u_1km_ms: " << format_double_roundtrip(c.initial_conditions.hodograph.u_1km_ms) << "\n";
    o << "    v_1km_ms: " << format_double_roundtrip(c.initial_conditions.hodograph.v_1km_ms) << "\n";
    o << "    u_6km_ms: " << format_double_roundtrip(c.initial_conditions.hodograph.u_6km_ms) << "\n";
    o << "    v_6km_ms: " << format_double_roundtrip(c.initial_conditions.hodograph.v_6km_ms) << "\n";
    o << "  trigger:\n";
    o << "    type: " << trigger_type_name(c.initial_conditions.trigger.type) << "\n";
    o << "    bubble:\n";
    o << "      dtheta_k: " << format_double_roundtrip(c.initial_conditions.trigger.bubble.dtheta_k) << "\n";
    o << "      cloud_water_kgkg: "
      << format_double_roundtrip(c.initial_conditions.trigger.bubble.cloud_water_kgkg)
      << "\n";
    o << "      radius_km: " << format_double_roundtrip(c.initial_conditions.trigger.bubble.radius_km) << "\n";
    o << "      center_x_m: " << format_double_roundtrip(c.initial_conditions.trigger.bubble.center_x_m) << "\n";
    o << "      center_y_m: " << format_double_roundtrip(c.initial_conditions.trigger.bubble.center_y_m) << "\n";
    o << "      center_z_m: " << format_double_roundtrip(c.initial_conditions.trigger.bubble.center_z_m) << "\n";
    o << "    vortex_seed:\n";
    o << "      r_max_m: " << format_double_roundtrip(c.initial_conditions.trigger.vortex_seed.r_max_m) << "\n";
    o << "      v_max_ms: " << format_double_roundtrip(c.initial_conditions.trigger.vortex_seed.v_max_ms) << "\n";
    o << "      z_top_m: " << format_double_roundtrip(c.initial_conditions.trigger.vortex_seed.z_top_m) << "\n";
    o << "      center_x_m: " << format_double_roundtrip(c.initial_conditions.trigger.vortex_seed.center_x_m) << "\n";
    o << "      center_y_m: " << format_double_roundtrip(c.initial_conditions.trigger.vortex_seed.center_y_m) << "\n";
    o << "      center_z_m: " << format_double_roundtrip(c.initial_conditions.trigger.vortex_seed.center_z_m) << "\n\n";

    o << "# Synthetic radar\n";
    o << "radar:\n";
    o << "  enabled: " << fmt_bool(c.radar.enabled) << "\n";
    o << "  products: [";
    for (std::size_t index = 0; index < c.radar.products.size(); ++index)
    {
        if (index != 0) o << ", ";
        o << c.radar.products[index];
    }
    o << "]\n";
    o << "  reflectivity_tier: " << c.radar.reflectivity_tier << "\n";
    o << "  velocity_sampling: " << c.radar.velocity_sampling << "\n";
    o << "  zdr_tier: " << c.radar.zdr_tier << "\n";
    o << "  location:\n";
    o << "    x_m: " << format_double_roundtrip(c.radar.radar_x_m) << "\n";
    o << "    y_m: " << format_double_roundtrip(c.radar.radar_y_m) << "\n";
    o << "    z_m: " << format_double_roundtrip(c.radar.radar_z_m) << "\n";
    o << "  apply_scatterer_correction: "
      << fmt_bool(c.radar.apply_scatterer_correction) << "\n\n";

    o << "# Output\n";
    o << "output:\n";
    o << "  preset: " << output_preset_name(c.output.preset) << "\n";
    o << "  format: " << output_format_name(c.output.format) << "\n";
    o << "  async_io: " << fmt_bool(c.output.async_io) << "\n";
    o << "  live_shm: " << fmt_bool(c.output.live_shm) << "\n";
    o << "  live_shm_fields: [";
    for (std::size_t index = 0;
         index < c.output.live_shm_fields.size(); ++index)
    {
        if (index != 0) o << ", ";
        o << c.output.live_shm_fields[index];
    }
    o << "]\n";
    o << "  live_shm_interval_s: "
      << format_double_roundtrip(c.output.live_shm_interval_s) << "\n";
    o << "  live_shm_full_rate: "
      << fmt_bool(c.output.live_shm_full_rate) << "\n";
    if (c.output.qualification_diagnostics_preset !=
        QualificationDiagnosticsPreset::None)
    {
        o << "  qualification_diagnostics:\n";
        o << "    preset: "
          << qualification_diagnostics_preset_name(
                 c.output.qualification_diagnostics_preset)
          << "\n";
        o << "    interval_s: "
          << c.output.qualification_diagnostics_interval_s << "\n";
    }
    o << "  trajectories: " << fmt_bool(c.output.trajectories) << "\n";
    o << "  trajectory_seed_time_s: " << format_double_roundtrip(c.output.trajectory_seed_time_s) << "\n";
    o << "  trajectory_count: " << c.output.trajectory_count << "\n";
    o << "  trajectory_sample_interval_s: "
      << format_double_roundtrip(c.output.trajectory_sample_interval_s) << "\n";
    o << "  trajectory_max_samples_per_parcel: "
      << c.output.trajectory_max_samples_per_parcel << "\n";
    if (c.output.format == OutputFormat::Netcdf)
    {
        o << "  netcdf_deflate_level: " << c.output.netcdf_deflate_level << "\n";
        o << "  netcdf_shuffle: " << fmt_bool(c.output.netcdf_shuffle) << "\n";
        o << "  netcdf_target_chunk_kib: " << c.output.netcdf_target_chunk_kib << "\n";
    }
    const bool out_is_zfp = c.output.format == OutputFormat::ZfpLow || c.output.format == OutputFormat::ZfpMedium ||
                            c.output.format == OutputFormat::ZfpHigh;
    if (out_is_zfp)
    {
        o << "  zfp_tolerance: " << format_double_roundtrip(c.output.zfp_tolerance) << "\n";
        o << "  zfp_mode: " << c.output.zfp_mode << "\n";
        o << "  zfp_precision_bits: " << c.output.zfp_precision_bits << "\n";
        o << "  zfp_rate_bps: " << c.output.zfp_rate_bps << "\n";
        o << "  zfp_keyframe_interval: " << c.output.zfp_keyframe_interval << "\n";
        o << "  zfp_sparse_threshold: " << format_double_roundtrip(c.output.zfp_sparse_threshold) << "\n";
        o << "  zfp_float16_prefilter: " << fmt_bool(c.output.zfp_float16_prefilter) << "\n";
        o << "  zfp_predictive_delta: " << fmt_bool(c.output.zfp_predictive_delta) << "\n";
        o << "  zfp_field_execution: " << c.output.zfp_field_execution << "\n";
        o << "  zfp_field_workers: " << c.output.zfp_field_workers << "\n";
        o << "  zfp_field_scratch_mib: " << c.output.zfp_field_scratch_mib << "\n";
        o << "  zfp_per_field_tolerances: " << fmt_bool(c.output.zfp_per_field_tolerances) << "\n";
        o << "  zfp_tight_tolerance: " << format_double_roundtrip(c.output.zfp_tight_tolerance) << "\n";
        o << "  zfp_moderate_tolerance: " << format_double_roundtrip(c.output.zfp_moderate_tolerance) << "\n";
        o << "  zfp_loose_tolerance: " << format_double_roundtrip(c.output.zfp_loose_tolerance) << "\n";
        if (!c.output.zfp_field_tolerances.empty())
        {
            o << "  zfp_field_tolerance:\n";
            std::vector<std::pair<std::string, double>> tolerances(c.output.zfp_field_tolerances.begin(),
                                                                   c.output.zfp_field_tolerances.end());
            std::sort(tolerances.begin(), tolerances.end());
            for (const auto& [field, tolerance] : tolerances)
                o << "    " << field << ": " << format_double_roundtrip(tolerance) << "\n";
        }
    }
    if (c.output.tiered_write_cadence)
    {
        o << "  tiered_write_cadence: " << fmt_bool(c.output.tiered_write_cadence) << "\n";
        o << "  write_cadence_medium_s: " << c.output.write_cadence_medium_s << "\n";
        o << "  write_cadence_slow_s: " << c.output.write_cadence_slow_s << "\n";
    }
    if (c.output.keep_last_frames > 0)
        o << "  keep_last_frames: " << c.output.keep_last_frames << "\n";
    if (c.output.maximum_retained_history_files > 0)
        o << "  maximum_retained_history_files: "
          << c.output.maximum_retained_history_files << "\n";
    if (c.output.minimum_free_space_mib > 0)
        o << "  minimum_free_space_mib: " << c.output.minimum_free_space_mib << "\n";
    o << "\n";

    o << "# Nimbus extensions (additive — not part of CM1)\n";
    o << "extensions:\n";
    o << "  compute:\n";
    o << "    backend: " << compute_backend_name(c.extensions.compute.backend) << "\n";
    o << "    memory_admission: " << memory_admission_policy_name(c.extensions.compute.memory_admission) << "\n";
    o << "    allow_fallback: " << fmt_bool(c.extensions.compute.allow_fallback) << "\n";
    if (c.extensions.compute.cuda_assist) o << "    cuda_assist: true\n";
    o << "    validate_parity: " << fmt_bool(c.extensions.compute.validate_parity) << "\n";
    o << "    gpu_apply_microphysics: " << fmt_bool(c.extensions.compute.gpu_apply_microphysics) << "\n";
    o << "    gpu_apply_slow: " << fmt_bool(c.extensions.compute.gpu_apply_slow) << "\n";
    o << "    gpu_apply_sgs: " << fmt_bool(c.extensions.compute.gpu_apply_sgs) << "\n";
    o << "  chaos:\n";
    o << "    scheme: " << chaos_scheme_name(c.extensions.chaos.scheme) << "\n";
    o << "    seed: " << c.extensions.chaos.seed << "\n";
    o << "    member_id: " << c.extensions.chaos.member_id << "\n";
    o << "    native_generation_required: "
      << fmt_bool(c.extensions.chaos.native_generation_required) << "\n";
    const auto write_string_list = [&o](const char* key,
                                        const std::vector<std::string>& values)
    {
        if (values.empty()) return;
        o << "    " << key << ": [";
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index != 0) o << ", ";
            o << values[index];
        }
        o << "]\n";
    };
    write_string_list("apply_to_ic", c.extensions.chaos.apply_to_ic);
    write_string_list(
        "apply_to_tendencies", c.extensions.chaos.apply_to_tendencies);
    const auto write_amplitudes = [&o](
        const char* key,
        const std::unordered_map<std::string, double>& amplitudes)
    {
        if (amplitudes.empty()) return;
        o << "    " << key << ":\n";
        std::vector<std::pair<std::string, double>> ordered(
            amplitudes.begin(), amplitudes.end());
        std::sort(ordered.begin(), ordered.end());
        for (const auto& [target, amplitude] : ordered)
            o << "      " << target << ": " << format_double_roundtrip(amplitude) << "\n";
    };
    write_amplitudes("sigma_ic", c.extensions.chaos.sigma_ic);
    write_amplitudes("alpha_tend", c.extensions.chaos.alpha_tend);
    o << "  execution:\n";
    o << "    mode: " << execution_mode_name(c.extensions.execution.mode) << "\n";
    if (c.extensions.checkpoint.every_s > 0 || c.extensions.checkpoint.keep_last > 0 ||
        c.extensions.checkpoint.minimum_free_space_mib > 0 || !c.extensions.checkpoint.directory.empty() ||
        !c.extensions.checkpoint.resume_from.empty())
    {
        o << "  checkpoint:\n";
        o << "    every_s: " << c.extensions.checkpoint.every_s << "\n";
        if (c.extensions.checkpoint.keep_last > 0)
            o << "    keep_last: " << c.extensions.checkpoint.keep_last << "\n";
        if (c.extensions.checkpoint.minimum_free_space_mib > 0)
            o << "    minimum_free_space_mib: " << c.extensions.checkpoint.minimum_free_space_mib << "\n";
        if (!c.extensions.checkpoint.directory.empty())
        {
            o << "    directory: " << c.extensions.checkpoint.directory << "\n";
        }
        if (!c.extensions.checkpoint.resume_from.empty())
        {
            o << "    resume_from: " << c.extensions.checkpoint.resume_from << "\n";
        }
    }
    o << "  validation:\n";
    o << "    guard_mode: " << validation_guard_mode_name(c.extensions.validation.guard_mode) << "\n";
    if (c.extensions.diagnostics.scientific_every_steps >= 0 || c.extensions.diagnostics.deterministic_reductions)
    {
        o << "  diagnostics:\n";
        if (c.extensions.diagnostics.scientific_every_steps >= 0)
        {
            o << "    scientific_every_steps: " << c.extensions.diagnostics.scientific_every_steps << "\n";
        }
        if (c.extensions.diagnostics.deterministic_reductions)
        {
            o << "    deterministic_reductions: true\n";
        }
    }
    const auto& performance = c.extensions.performance;
    if (performance.timing.has_value() || performance.report_every_steps.has_value() ||
        performance.warmup_steps.has_value())
    {
        o << "  performance:\n";
        if (performance.timing.has_value())
        {
            o << "    timing: " << fmt_bool(*performance.timing) << "\n";
        }
        if (performance.report_every_steps.has_value())
        {
            o << "    report_every_steps: " << *performance.report_every_steps << "\n";
        }
        if (performance.warmup_steps.has_value())
        {
            o << "    warmup_steps: " << *performance.warmup_steps << "\n";
        }
    }

    return o.str();
}
} // namespace tmv::init::config
