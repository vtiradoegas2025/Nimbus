#include "util/numeric_text.hpp"
#include "init/config/v0_to_v1.hpp"

#include "config_v1_serialization_common.hpp"

#include "init/sounding/parametric_target_math.hpp"
#include "turbulence/turbulence_base.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tmv::init::config
{

namespace
{

/// v1 dynamics.scheme + staggering -> compound v0 scheme name. When the v1
/// scheme is the unqualified canonical, derive the Cartesian compound;
/// otherwise pass it through (covers B1a aliases like cart).
std::string v0_scheme_from_v1(const DynamicsSection& dyn, const GridSection& grid)
{
    if (dyn.scheme != "klemp_wilhelmson_split")
    {
        return dyn.scheme; // pre-qualified or legacy alias — let B1a handle it
    }
    const bool cgrid = (grid.staggering == StaggerType::CGrid);
    return cgrid ? "klemp_wilhelmson_split_cartesian_cgrid" : "klemp_wilhelmson_split_cartesian_agrid";
}

/// v1 microphysics enum → v0 string. The canonical name ("milbrandt") already
/// matches the v0 factory registration, so this is a straight name lookup.
std::string v0_microphysics_string(MicrophysicsScheme s)
{
    return microphysics_scheme_name(s);
}

/// Strict-v1 calls horizontal SGS "2d"; the runtime names the same physical
/// selection "horizontal_only". Project the semantic runtime token directly
/// rather than relying on a warning/fallback path.
const char* runtime_turbulence_mode_name(TurbulenceMode mode)
{
    switch (mode)
    {
    case TurbulenceMode::TwoD:
        return "horizontal_only";
    case TurbulenceMode::VerticalOnly:
        return "vertical_only";
    case TurbulenceMode::ThreeD:
        return "3d";
    }
    return "3d";
}

std::string fmt_int_to_string(int v)
{
    return std::to_string(v);
}

using tmv::strutil::format_double_roundtrip;

std::string fmt_bool_to_string(bool v)
{
    return v ? "true" : "false";
}

} // namespace

std::unordered_map<std::string, std::string> serialize_v1_to_v0_map(const ConfigV1& c)
{
    require_runtime_supported_configuration(c);

    std::unordered_map<std::string, std::string> m;

    // ── Top-level + grid ──────────────────────────────────────
    m["coordinate_system"]           = coordinate_system_name(c.grid.coordinate_system);
    m["grid.staggering"]             = stagger_type_name(c.grid.staggering);
    m["grid.nx"]                     = fmt_int_to_string(c.grid.nx);
    m["grid.ny"]                     = fmt_int_to_string(c.grid.ny);
    m["grid.nz"]                     = fmt_int_to_string(c.grid.nz);
    m["grid.dx"]                     = format_double_roundtrip(c.grid.dx);
    m["grid.dy"]                     = format_double_roundtrip(c.grid.dy);
    m["grid.dz"]                     = format_double_roundtrip(c.grid.dz);
    m["grid.dt"]                     = format_double_roundtrip(c.time.dt_large);
    // The process runtime binds this v1-only selection before consuming the
    // compatibility map. Retain it here as provenance for map-only validators
    // and programmatic consumers instead of silently dropping the equation.
    m["damping.mode"]                = damping_mode_name(c.damping.mode);
    m["coriolis.enabled"]            = fmt_bool_to_string(c.coriolis.enabled);
    m["coriolis.mode"]               = coriolis_mode_name(c.coriolis.mode);
    m["coriolis.latitude_deg"]       = format_double_roundtrip(c.coriolis.latitude_deg);
    m["terrain.ztop"]                = format_double_roundtrip(c.grid.ztop);
    m["terrain.scheme"]              = terrain_scheme_name(c.terrain.scheme);
    m["terrain.coord_id"]            = terrain_coordinate_name(c.terrain.coordinate);
    m["terrain.horizontal_placement"] =
        terrain_horizontal_placement_name(c.terrain.horizontal_placement);
    if (c.terrain.horizontal_placement == TerrainHorizontalPlacement::EarthFixed)
    {
        m["terrain.feature_center_earth_x_m"] =
            format_double_roundtrip(c.terrain.feature_center_earth_x_m);
        m["terrain.feature_center_earth_y_m"] =
            format_double_roundtrip(c.terrain.feature_center_earth_y_m);
    }
    m["terrain.compute_derivatives"] = fmt_bool_to_string(c.terrain.compute_derivatives);
    m["terrain.compute_metrics"]     = fmt_bool_to_string(c.terrain.compute_metrics);
    m["terrain.smoothing_decay"]     = format_double_roundtrip(c.terrain.smoothing_decay);
    m["terrain.bell.h0"]             = format_double_roundtrip(c.terrain.bell.h0);
    m["terrain.bell.a"]              = format_double_roundtrip(c.terrain.bell.a);
    m["terrain.bell.axisymmetric"]   = fmt_bool_to_string(c.terrain.bell.axisymmetric);
    m["terrain.schar.H"]             = format_double_roundtrip(c.terrain.schar.H);
    m["terrain.schar.a"]             = format_double_roundtrip(c.terrain.schar.a);
    m["terrain.schar.ell"]           = format_double_roundtrip(c.terrain.schar.ell);

    // ── Time + run cadence ────────────────────────────────────
    m["duration_s"]                                 = format_double_roundtrip(c.time.run_time);
    m["output.write_every_s"]                       = format_double_roundtrip(c.time.output_interval);
    m["numerics.time_stepping.n_acoustic_substeps"] = fmt_int_to_string(c.time.acoustic_substeps);
    m["numerics.time_stepping.adaptive_dt"]         = fmt_bool_to_string(c.time.adaptive_dt);
    m["numerics.time_stepping.cfl_safety"]          = format_double_roundtrip(c.time.cfl_safety);
    m["domain_motion.mode"]                         = domain_motion_mode_name(c.domain_motion.mode);
    m["domain_motion.frame_velocity.u_ms"]          = format_double_roundtrip(c.domain_motion.frame_velocity.u_ms);
    m["domain_motion.frame_velocity.v_ms"]          = format_double_roundtrip(c.domain_motion.frame_velocity.v_ms);
    m["domain_motion.lifetime_planning.enabled"]    = fmt_bool_to_string(c.domain_motion.lifetime_planning.enabled);
    m["domain_motion.lifetime_planning.action"] =
        lifetime_planning_action_name(c.domain_motion.lifetime_planning.action);
    m["domain_motion.lifetime_planning.motion_source"] =
        storm_motion_source_name(c.domain_motion.lifetime_planning.motion_source);
    m["domain_motion.lifetime_planning.explicit_u_ms"] =
        format_double_roundtrip(c.domain_motion.lifetime_planning.explicit_u_ms);
    m["domain_motion.lifetime_planning.explicit_v_ms"] =
        format_double_roundtrip(c.domain_motion.lifetime_planning.explicit_v_ms);
    m["domain_motion.lifetime_planning.upstream_buffer_m"] =
        format_double_roundtrip(c.domain_motion.lifetime_planning.upstream_buffer_m);
    m["domain_motion.lifetime_planning.downstream_buffer_m"] =
        format_double_roundtrip(c.domain_motion.lifetime_planning.downstream_buffer_m);
    m["domain_motion.lifetime_planning.cross_track_buffer_m"] =
        format_double_roundtrip(c.domain_motion.lifetime_planning.cross_track_buffer_m);
    m["domain_motion.lifetime_planning.motion_uncertainty_ms"] =
        format_double_roundtrip(c.domain_motion.lifetime_planning.motion_uncertainty_ms);
    m["domain_motion.tracking.enabled"]             = fmt_bool_to_string(c.domain_motion.tracking.enabled);
    m["domain_motion.tracking.recycle_domain"]      = fmt_bool_to_string(c.domain_motion.tracking.recycle_domain);
    m["domain_motion.tracking.interval_s"]          = format_double_roundtrip(c.domain_motion.tracking.interval_s);
    m["domain_motion.tracking.inner_margin_m"]      = format_double_roundtrip(c.domain_motion.tracking.inner_margin_m);
    m["domain_motion.tracking.maximum_shift_cells"] = std::to_string(c.domain_motion.tracking.maximum_shift_cells);
    m["domain_motion.tracking.cooldown_s"]          = format_double_roundtrip(c.domain_motion.tracking.cooldown_s);
    m["domain_motion.tracking.minimum_confidence"]  = format_double_roundtrip(c.domain_motion.tracking.minimum_confidence);
    m["domain_motion.tracking.minimum_updraft_ms"]  = format_double_roundtrip(c.domain_motion.tracking.minimum_updraft_ms);
    m["domain_motion.tracking.minimum_condensate_path_kgm2"] =
        format_double_roundtrip(c.domain_motion.tracking.minimum_condensate_path_kgm2);
    m["domain_motion.tracking.updraft_helicity_bottom_m_agl"] =
        format_double_roundtrip(c.domain_motion.tracking.updraft_helicity_bottom_m_agl);
    m["domain_motion.tracking.updraft_helicity_top_m_agl"] =
        format_double_roundtrip(c.domain_motion.tracking.updraft_helicity_top_m_agl);
    m["domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2"] =
        format_double_roundtrip(c.domain_motion.tracking.minimum_absolute_updraft_helicity_m2s2);
    m["domain_motion.tracking.minimum_component_cells"] =
        std::to_string(c.domain_motion.tracking.minimum_component_cells);
    m["domain_motion.tracking.minimum_core_area_m2"] =
        format_double_roundtrip(c.domain_motion.tracking.minimum_core_area_m2);
    m["domain_motion.tracking.maximum_motion_speed_ms"] =
        format_double_roundtrip(c.domain_motion.tracking.maximum_motion_speed_ms);
    m["domain_motion.tracking.maximum_motion_acceleration_ms2"] =
        format_double_roundtrip(c.domain_motion.tracking.maximum_motion_acceleration_ms2);
    m["domain_motion.tracking.minimum_continuity_samples"] =
        std::to_string(c.domain_motion.tracking.minimum_continuity_samples);
    m["domain_motion.tracking.lost_after_samples"] = std::to_string(c.domain_motion.tracking.lost_after_samples);
    m["domain_motion.tracking.topology_confirmation_samples"] =
        std::to_string(c.domain_motion.tracking.topology_confirmation_samples);
    m["domain_motion.tracking.branch_preference"] =
        storm_branch_preference_name(c.domain_motion.tracking.branch_preference);
    m["domain_motion.tracking.branch_velocity_uncertainty_ms"] =
        format_double_roundtrip(c.domain_motion.tracking.branch_velocity_uncertainty_ms);
    m["domain_motion.nesting.enabled"] = fmt_bool_to_string(c.domain_motion.nesting.enabled);
    m["domain_motion.nesting.parent_ratio_x"] =
        std::to_string(c.domain_motion.nesting.parent_ratio_x);
    m["domain_motion.nesting.parent_ratio_y"] =
        std::to_string(c.domain_motion.nesting.parent_ratio_y);
    m["domain_motion.nesting.parent_time_step_ratio"] =
        std::to_string(c.domain_motion.nesting.parent_time_step_ratio);
    m["domain_motion.nesting.child_origin_parent_cell_i"] =
        std::to_string(c.domain_motion.nesting.child_origin_parent_cell_i);
    m["domain_motion.nesting.child_origin_parent_cell_j"] =
        std::to_string(c.domain_motion.nesting.child_origin_parent_cell_j);
    m["domain_motion.nesting.child_cells_x"] =
        std::to_string(c.domain_motion.nesting.child_cells_x);
    m["domain_motion.nesting.child_cells_y"] =
        std::to_string(c.domain_motion.nesting.child_cells_y);
    m["domain_motion.nesting.relaxation_width_child_cells"] =
        std::to_string(c.domain_motion.nesting.relaxation_width_child_cells);
    m["domain_motion.nesting.boundary_stencil_parent_cells"] =
        std::to_string(c.domain_motion.nesting.boundary_stencil_parent_cells);
    m["domain_motion.nesting.relaxation_e_folding_time_s"] =
        format_double_roundtrip(c.domain_motion.nesting.relaxation_e_folding_time_s);
    m["domain_motion.nesting.feedback"] =
        nesting_feedback_name(c.domain_motion.nesting.feedback);
    m["numerics.time_stepping.scheme"] = time_integrator_name(c.time.integrator);
    m["numerics.time_stepping.split_acoustic"] = fmt_bool_to_string(
        c.time.integrator == TimeIntegrator::SplitExplicit);
    m["numerics.advection.scheme"] =
        resolved_advection_scheme_name(c.advection.scheme, c.grid.staggering);

    // ── Dynamics ──────────────────────────────────────────────
    m["dynamics.scheme"] = v0_scheme_from_v1(c.dynamics, c.grid);

    // ── Microphysics ──────────────────────────────────────────
    m["microphysics.scheme"] = v0_microphysics_string(c.microphysics.scheme);
    if (c.microphysics.scheme != MicrophysicsScheme::None && c.microphysics.scheme != MicrophysicsScheme::Milbrandt)
    {
        m["microphysics.shared.cloud_water_threshold_kgkg"] =
            format_double_roundtrip(c.microphysics.parameters.shared.cloud_water_threshold_kgkg);
        m["microphysics.shared.autoconversion_rate_s_inv"] =
            format_double_roundtrip(c.microphysics.parameters.shared.autoconversion_rate_s_inv);
    }
    if (c.microphysics.scheme == MicrophysicsScheme::Kessler)
    {
        m["microphysics.kessler.accretion_coefficient"] =
            format_double_roundtrip(c.microphysics.parameters.kessler.accretion_coefficient);
        m["microphysics.kessler.evaporation_efficiency"] =
            format_double_roundtrip(c.microphysics.parameters.kessler.evaporation_efficiency);
    }
    else if (c.microphysics.scheme == MicrophysicsScheme::Lin)
    {
        m["microphysics.lin.accretion_coefficient"] =
            format_double_roundtrip(c.microphysics.parameters.lin.accretion_coefficient);
        m["microphysics.lin.evaporation_rate_s_inv"] =
            format_double_roundtrip(c.microphysics.parameters.lin.evaporation_rate_s_inv);
    }
    else if (c.microphysics.scheme == MicrophysicsScheme::Thompson)
    {
        m["microphysics.thompson.evaporation_rate_s_inv"] =
            format_double_roundtrip(c.microphysics.parameters.thompson.evaporation_rate_s_inv);
        m["microphysics.thompson.water_friendly_aerosol_surface_m3"] =
            format_double_roundtrip(
                c.microphysics.parameters.thompson.water_friendly_aerosol_surface_m3);
        m["microphysics.thompson.ice_friendly_aerosol_surface_m3"] =
            format_double_roundtrip(
                c.microphysics.parameters.thompson.ice_friendly_aerosol_surface_m3);
    }
    else if (c.microphysics.scheme == MicrophysicsScheme::Milbrandt)
    {
        m["microphysics.milbrandt.hail_processes"] =
            fmt_bool_to_string(c.microphysics.parameters.milbrandt.hail_processes);
    }

    // ── Turbulence ────────────────────────────────────────────
    m["turbulence.scheme"] = turbulence_scheme_name(c.turbulence.scheme);
    m["turbulence.mode"]   = runtime_turbulence_mode_name(c.turbulence.mode);
    m["turbulence.filter_width"] =
        turbulence_filter_width_name(c.turbulence.filter_width);
    m["turbulence.user_filter_width_m"] =
        format_double_roundtrip(c.turbulence.user_filter_width_m);
    m["turbulence.dynamic_Cs"] =
        fmt_bool_to_string(c.turbulence.dynamic_coefficient);
    m["turbulence.stability_correction"] =
        c.turbulence.stability_correction == TurbulenceStabilityCorrection::Richardson
            ? tmv::kLillyRichardsonModel
            : turbulence_stability_correction_name(c.turbulence.stability_correction);
    m["turbulence.Cs"]     = format_double_roundtrip(c.turbulence.Cs);
    m["turbulence.Pr_t"]   = format_double_roundtrip(c.turbulence.Pr_t);
    m["turbulence.Sc_t"]   = format_double_roundtrip(c.turbulence.Sc_t);

    // ── Diffusion ─────────────────────────────────────────────
    m["numerics.diffusion.scheme"]         = diffusion_scheme_name(c.diffusion.scheme);
    m["numerics.diffusion.operator_order"] = fmt_int_to_string(c.diffusion.operator_order);
    m["numerics.diffusion.K_h"]            = format_double_roundtrip(c.diffusion.K_h);
    m["numerics.diffusion.K_v"]            = format_double_roundtrip(c.diffusion.K_v);
    m["numerics.diffusion.apply_to"]       = diffusion_target_name(c.diffusion.apply_to);
    m["numerics.diffusion.coefficient_source"] =
        diffusion_coefficient_source_name(c.diffusion.coefficient_source);
    if (c.diffusion.coefficient_source == DiffusionCoefficientSource::LinearHeight)
    {
        m["numerics.diffusion.K_v_surface"] =
            format_double_roundtrip(c.diffusion.K_v_surface);
        m["numerics.diffusion.K_v_top"] =
            format_double_roundtrip(c.diffusion.K_v_top);
    }

    // ── Radiation ─────────────────────────────────────────────
    m["radiation.scheme"]                      = radiation_scheme_name(c.radiation.scheme);
    m["radiation.solar_forcing"]               = radiation_solar_forcing_name(c.radiation.solar_forcing);
    m["radiation.latitude_deg"]                = format_double_roundtrip(c.radiation.latitude_deg);
    m["radiation.declination_deg"]             = format_double_roundtrip(c.radiation.declination_deg);
    m["radiation.local_solar_time_at_start_h"] = format_double_roundtrip(c.radiation.local_solar_time_at_start_h);
    m["radiation.do_lw"]                       = fmt_bool_to_string(c.radiation.do_lw);
    m["radiation.do_sw"]                       = fmt_bool_to_string(c.radiation.do_sw);
    m["radiation.dt"]                          = format_double_roundtrip(c.radiation.cadence_s);
    m["radiation.clear_sky_only"]              = fmt_bool_to_string(c.radiation.clear_sky_only);
    m["radiation.simple_grey_optical_model"]   = c.radiation.simple_grey_optical_model;
    m["radiation.use_subcycling"]              = fmt_bool_to_string(c.radiation.use_subcycling);
    m["radiation.subcycle_count"]              = fmt_int_to_string(c.radiation.subcycle_count);
    m["radiation.enable_diagnostics"]          = fmt_bool_to_string(c.radiation.enable_diagnostics);
    m["radiation.tau_lw_ref"]                  = format_double_roundtrip(c.radiation.tau_lw_ref);
    m["radiation.tau_sw_ref"]                  = format_double_roundtrip(c.radiation.tau_sw_ref);
    m["radiation.n_lw"]                        = format_double_roundtrip(c.radiation.n_lw);
    m["radiation.n_sw"]                        = format_double_roundtrip(c.radiation.n_sw);
    m["radiation.simple_grey_lw_vapor_absorption_m2kg"] =
        format_double_roundtrip(c.radiation.simple_grey_lw_vapor_absorption_m2kg);
    m["radiation.simple_grey_lw_liquid_absorption_m2kg"] =
        format_double_roundtrip(c.radiation.simple_grey_lw_liquid_absorption_m2kg);
    m["radiation.simple_grey_lw_ice_absorption_m2kg"] =
        format_double_roundtrip(c.radiation.simple_grey_lw_ice_absorption_m2kg);
    m["radiation.simple_grey_sw_vapor_absorption_m2kg"] =
        format_double_roundtrip(c.radiation.simple_grey_sw_vapor_absorption_m2kg);
    m["radiation.simple_grey_sw_liquid_absorption_m2kg"] =
        format_double_roundtrip(c.radiation.simple_grey_sw_liquid_absorption_m2kg);
    m["radiation.simple_grey_sw_ice_absorption_m2kg"] =
        format_double_roundtrip(c.radiation.simple_grey_sw_ice_absorption_m2kg);
    if (c.radiation.scheme == RadiationScheme::RRTMG)
    {
        m["radiation.rrtmg_ozone_vmr"] = format_double_roundtrip(c.radiation.rrtmg_ozone_vmr);
        m["radiation.rrtmg_co2_vmr"]   = format_double_roundtrip(c.radiation.rrtmg_co2_vmr);
        m["radiation.rrtmg_ch4_vmr"]   = format_double_roundtrip(c.radiation.rrtmg_ch4_vmr);
        m["radiation.rrtmg_n2o_vmr"]   = format_double_roundtrip(c.radiation.rrtmg_n2o_vmr);
        m["radiation.rrtmg_co_vmr"]    = format_double_roundtrip(c.radiation.rrtmg_co_vmr);
        m["radiation.rrtmg_o2_vmr"]    = format_double_roundtrip(c.radiation.rrtmg_o2_vmr);
        m["radiation.rrtmg_n2_vmr"]    = format_double_roundtrip(c.radiation.rrtmg_n2_vmr);
        m["radiation.rrtmg_cfc11_vmr"] = format_double_roundtrip(c.radiation.rrtmg_cfc11_vmr);
        m["radiation.rrtmg_cfc12_vmr"] = format_double_roundtrip(c.radiation.rrtmg_cfc12_vmr);
        m["radiation.rrtmg_cfc22_vmr"] = format_double_roundtrip(c.radiation.rrtmg_cfc22_vmr);
        m["radiation.rrtmg_ccl4_vmr"]  = format_double_roundtrip(c.radiation.rrtmg_ccl4_vmr);
        m["radiation.rrtmg_liquid_effective_radius_um"] =
            format_double_roundtrip(c.radiation.rrtmg_liquid_effective_radius_um);
        m["radiation.rrtmg_ice_effective_radius_um"] =
            format_double_roundtrip(c.radiation.rrtmg_ice_effective_radius_um);
    }
    m["radiation.terrain_sw_cosine_nodes"] =
        fmt_int_to_string(c.radiation.terrain_sw_cosine_nodes);
    m["radiation.terrain_sw_azimuth_nodes"] =
        fmt_int_to_string(c.radiation.terrain_sw_azimuth_nodes);
    m["radiation.terrain_sw_maximum_bounces"] =
        fmt_int_to_string(c.radiation.terrain_sw_maximum_bounces);
    m["radiation.terrain_sw_relative_residual_tolerance"] =
        format_double_roundtrip(
            c.radiation.terrain_sw_relative_residual_tolerance);
    m["radiation.terrain_sw_absolute_residual_tolerance_w"] =
        format_double_roundtrip(
            c.radiation.terrain_sw_absolute_residual_tolerance_w);

    // ── Surface + PBL → single v0 boundary_layer.scheme ───────
    // The v0 runtime has ONE boundary-layer slot, and each scheme
    // (slab/ysu/mynn) bundles its own surface-flux computation, gated by
    // apply_surface_fluxes. So an explicit PBL scheme (YSU/MYNN) takes the slot
    // and self-computes fluxes. New v1 configs select the slab mixed-layer
    // explicitly through boundary_layer.scheme=slab.  Do not infer a PBL from
    // surface.scheme: the strict loader rejects that ambiguous combination.
    m["boundary_layer.scheme"] = boundary_layer_scheme_name(c.boundary_layer.scheme);
    m["boundary_layer.pbl_top_method"] =
        boundary_layer_top_method_name(c.boundary_layer.pbl_top_method);
    m["boundary_layer.dt_pbl"] =
        format_double_roundtrip(c.boundary_layer.cadence_s);
    m["boundary_layer.clear_sky_only"] =
        fmt_bool_to_string(c.boundary_layer.clear_sky_only);
    // Prognostic TKE is intrinsic to MYNN, not an independent strict-v1
    // selection. Emit both values so an earlier legacy load cannot leak this
    // hidden runtime switch into the next typed configuration.
    m["boundary_layer.enable_tke"] = fmt_bool_to_string(
        c.boundary_layer.scheme == BoundaryLayerScheme::MYNN);
    // With surface:none a PBL still mixes but sees zero surface forcing (a
    // free-slip PBL test). The validation above makes the projection exact.
    m["boundary_layer.apply_surface_fluxes"] = fmt_bool_to_string(c.surface.apply_fluxes);
    m["surface.scheme"]                      = surface_scheme_name(c.surface.scheme);
    m["surface.z0m"]                         = format_double_roundtrip(c.surface.z0m_m);
    m["surface.z0h"]                         = format_double_roundtrip(c.surface.z0h_m);
    m["surface.albedo"]                      = format_double_roundtrip(c.surface.albedo);
    m["surface.emissivity"]                  = format_double_roundtrip(c.surface.emissivity);

    // Prescribed surface (skin) state (CM1 tsk0) -> v0 surface.tsfc / surface.qsfc.
    // Emitted only when explicitly set; when omitted the runtime holds the
    // initial surface-air state as the skin (zero flux at t=0). See
    // src/boundary_layer/runtime.cpp.
    if (c.surface.tsk_k > 0.0)
    {
        m["surface.tsfc"] = format_double_roundtrip(c.surface.tsk_k);
    }
    if (c.surface.qsfc_kgkg >= 0.0)
    {
        m["surface.qsfc"] = format_double_roundtrip(c.surface.qsfc_kgkg);
    }

    // ── Initial conditions → environment + trigger ────────────
    const auto& s                  = c.initial_conditions.sounding;
    m["environment.sounding.type"] = sounding_type_name(s.type);
    switch (s.type)
    {
    case SoundingType::ParametricCAPE:
        m["environment.cape_target_jkg"] = format_double_roundtrip(s.cape_jkg);
        m["environment.sfc_theta_k"]     = format_double_roundtrip(s.surface_theta_k);
        m["environment.sfc_qv_kgkg"]     = format_double_roundtrip(sounding_qv_or_default(s));
        m["environment.tropopause_z_m"]  = format_double_roundtrip(s.tropopause_m);
        break;

    case SoundingType::ParametricTargets:
        m["environment.sounding.targets.cape_jkg"] = format_double_roundtrip(s.cape_jkg);
        m["environment.sounding.targets.cin_jkg"]  = format_double_roundtrip(s.cin_jkg);
        if (s.moisture_qv_kgkg.has_value())
        {
            m["environment.sounding.targets.moisture_qv_kgkg"] = format_double_roundtrip(*s.moisture_qv_kgkg);
        }
        else
        {
            m["environment.sounding.targets.lcl_m"] = format_double_roundtrip(s.lcl_m);
        }
        m["environment.sounding.targets.lfc_m"]           = format_double_roundtrip(s.lfc_m);
        m["environment.sounding.targets.el_m"]            = format_double_roundtrip(s.el_m);
        m["environment.sounding.targets.surface_theta_k"] = format_double_roundtrip(s.surface_theta_k);
        break;

    case SoundingType::File:
        m["environment.sounding.file_path"] = s.file_path;
        m["environment.sounding.scheme_id"] = "auto";
        break;
    }

    const auto& h = c.initial_conditions.hodograph;
    // Runtime "wk_param" is the concrete three-anchor source used by the
    // strict-v1 "custom" branch. Emitting "custom" here previously fell
    // through the runtime parser to Auto, allowing file-sounding winds to
    // override explicitly configured anchors.
    m["environment.hodograph.type"]     = (h.type == HodographType::Custom) ? "wk_param" : hodograph_type_name(h.type);
    m["environment.hodograph.u_sfc_ms"] = format_double_roundtrip(h.u_sfc_ms);
    m["environment.hodograph.v_sfc_ms"] = format_double_roundtrip(h.v_sfc_ms);
    m["environment.hodograph.u_1km_ms"] = format_double_roundtrip(h.u_1km_ms);
    m["environment.hodograph.v_1km_ms"] = format_double_roundtrip(h.v_1km_ms);
    m["environment.hodograph.u_6km_ms"] = format_double_roundtrip(h.u_6km_ms);
    m["environment.hodograph.v_6km_ms"] = format_double_roundtrip(h.v_6km_ms);

    const auto& tr = c.initial_conditions.trigger;
    // v0 trigger.type uses "warm_bubble" for the bubble case.
    std::string trig_type;
    switch (tr.type)
    {
    case TriggerType::None:
        trig_type = "none";
        break;
    case TriggerType::Bubble:
        trig_type = "warm_bubble";
        break;
    case TriggerType::VortexSeed:
        trig_type = "vortex_seed";
        break;
    }
    m["trigger.type"] = trig_type;

    m["trigger.bubble.dtheta_k"]    = format_double_roundtrip(tr.bubble.dtheta_k);
    m["trigger.bubble.cloud_water_kgkg"] =
        format_double_roundtrip(tr.bubble.cloud_water_kgkg);
    m["trigger.bubble.radius_km"]   = format_double_roundtrip(tr.bubble.radius_km);
    m["trigger.bubble.center_x_km"] = format_double_roundtrip(tr.bubble.center_x_m / 1000.0);
    m["trigger.bubble.center_y_km"] = format_double_roundtrip(tr.bubble.center_y_m / 1000.0);
    m["trigger.bubble.center_z_km"] = format_double_roundtrip(tr.bubble.center_z_m / 1000.0);

    m["trigger.vortex_seed.r_max_m"]     = format_double_roundtrip(tr.vortex_seed.r_max_m);
    m["trigger.vortex_seed.v_max_ms"]    = format_double_roundtrip(tr.vortex_seed.v_max_ms);
    m["trigger.vortex_seed.z_top_m"]     = format_double_roundtrip(tr.vortex_seed.z_top_m);
    m["trigger.vortex_seed.center_x_km"] = format_double_roundtrip(tr.vortex_seed.center_x_m / 1000.0);
    m["trigger.vortex_seed.center_y_km"] = format_double_roundtrip(tr.vortex_seed.center_y_m / 1000.0);
    m["trigger.vortex_seed.center_z_km"] = format_double_roundtrip(tr.vortex_seed.center_z_m / 1000.0);

    std::string radar_products = "[";
    for (std::size_t index = 0; index < c.radar.products.size(); ++index)
    {
        if (index != 0) radar_products += ",";
        radar_products += c.radar.products[index];
    }
    radar_products += "]";
    m["radar.enabled"] = fmt_bool_to_string(c.radar.enabled);
    m["radar.products"] = radar_products;
    m["radar.reflectivity_tier"] = c.radar.reflectivity_tier;
    m["radar.velocity_sampling"] = c.radar.velocity_sampling;
    m["radar.zdr_tier"] = c.radar.zdr_tier;
    m["radar.location.x_m"] = format_double_roundtrip(c.radar.radar_x_m);
    m["radar.location.y_m"] = format_double_roundtrip(c.radar.radar_y_m);
    m["radar.location.z_m"] = format_double_roundtrip(c.radar.radar_z_m);
    m["radar.apply_scatterer_correction"] =
        fmt_bool_to_string(c.radar.apply_scatterer_correction);

    // ── Output ────────────────────────────────────────────────
    m["output.preset"]                 = output_preset_name(c.output.preset);
    m["output.fields"]                 = output_preset_name(c.output.preset); // v0 read this name
    m["output.format"]                 = output_format_name(c.output.format);
    m["output.async_io"]               = fmt_bool_to_string(c.output.async_io);
    m["output.live_shm"]               = fmt_bool_to_string(c.output.live_shm);
    std::string live_shm_fields;
    for (std::size_t index = 0;
         index < c.output.live_shm_fields.size(); ++index)
    {
        if (index != 0) live_shm_fields += ',';
        live_shm_fields += c.output.live_shm_fields[index];
    }
    m["output.live_shm_fields"] = live_shm_fields;
    m["output.live_shm_interval_s"] =
        format_double_roundtrip(c.output.live_shm_interval_s);
    m["output.live_shm_full_rate"] =
        fmt_bool_to_string(c.output.live_shm_full_rate);
    if (c.output.qualification_diagnostics_preset !=
        QualificationDiagnosticsPreset::None)
    {
        m["output.qualification_diagnostics.preset"] =
            qualification_diagnostics_preset_name(
                c.output.qualification_diagnostics_preset);
        m["output.qualification_diagnostics.interval_s"] =
            fmt_int_to_string(c.output.qualification_diagnostics_interval_s);
    }
    m["output.trajectories"]           = fmt_bool_to_string(c.output.trajectories);
    m["output.trajectory_seed_time_s"] = format_double_roundtrip(c.output.trajectory_seed_time_s);
    m["output.trajectory_count"]       = fmt_int_to_string(c.output.trajectory_count);
    m["output.trajectory_sample_interval_s"] =
        format_double_roundtrip(c.output.trajectory_sample_interval_s);
    m["output.trajectory_max_samples_per_parcel"] =
        fmt_int_to_string(c.output.trajectory_max_samples_per_parcel);
    if (c.output.format == OutputFormat::Netcdf)
    {
        m["output.netcdf_deflate_level"]    = fmt_int_to_string(c.output.netcdf_deflate_level);
        m["output.netcdf_shuffle"]          = fmt_bool_to_string(c.output.netcdf_shuffle);
        m["output.netcdf_target_chunk_kib"] = fmt_int_to_string(c.output.netcdf_target_chunk_kib);
    }

    // ZFP tuning round-trips only for Zfp* tiers (parse_output_config reads
    // these into OutputConfig; the writer consults them only when format=zfp).
    // The sentinel zfp_tolerance 0.0 is omitted so parse_output_config retains
    // the selected format tier's positive default.
    // Emitting only on demand keeps non-ZFP / non-tiered configs' v0 maps
    // byte-identical to before this change.
    const bool is_zfp = c.output.format == OutputFormat::ZfpLow || c.output.format == OutputFormat::ZfpMedium ||
                        c.output.format == OutputFormat::ZfpHigh;
    if (is_zfp)
    {
        if (c.output.zfp_tolerance > 0.0)
            m["output.zfp_tolerance"] = format_double_roundtrip(c.output.zfp_tolerance);
        m["output.zfp_mode"]                 = c.output.zfp_mode;
        m["output.zfp_precision_bits"]       = fmt_int_to_string(c.output.zfp_precision_bits);
        m["output.zfp_rate_bps"]             = fmt_int_to_string(c.output.zfp_rate_bps);
        m["output.zfp_keyframe_interval"]    = fmt_int_to_string(c.output.zfp_keyframe_interval);
        m["output.zfp_sparse_threshold"]     = format_double_roundtrip(c.output.zfp_sparse_threshold);
        m["output.zfp_float16_prefilter"]    = fmt_bool_to_string(c.output.zfp_float16_prefilter);
        m["output.zfp_predictive_delta"]     = fmt_bool_to_string(c.output.zfp_predictive_delta);
        m["output.zfp_field_execution"]      = c.output.zfp_field_execution;
        m["output.zfp_field_workers"]        = fmt_int_to_string(c.output.zfp_field_workers);
        m["output.zfp_field_scratch_mib"]    = fmt_int_to_string(c.output.zfp_field_scratch_mib);
        m["output.zfp_per_field_tolerances"] = fmt_bool_to_string(c.output.zfp_per_field_tolerances);
        m["output.zfp_tight_tolerance"]      = format_double_roundtrip(c.output.zfp_tight_tolerance);
        m["output.zfp_moderate_tolerance"]   = format_double_roundtrip(c.output.zfp_moderate_tolerance);
        m["output.zfp_loose_tolerance"]      = format_double_roundtrip(c.output.zfp_loose_tolerance);
        for (const auto& [field, tolerance] : c.output.zfp_field_tolerances)
        {
            m["output.zfp_field_tolerance." + field] = format_double_roundtrip(tolerance);
        }
    }
    if (c.output.tiered_write_cadence)
    {
        m["output.tiered_write_cadence"]   = "true";
        m["output.write_cadence_medium_s"] = fmt_int_to_string(c.output.write_cadence_medium_s);
        m["output.write_cadence_slow_s"]   = fmt_int_to_string(c.output.write_cadence_slow_s);
    }
    m["output.keep_last_frames"]       = fmt_int_to_string(c.output.keep_last_frames);
    m["output.maximum_retained_history_files"] =
        fmt_int_to_string(c.output.maximum_retained_history_files);
    m["output.minimum_free_space_mib"] = fmt_int_to_string(c.output.minimum_free_space_mib);

    // ── Extensions → compute / chaos / validation ─────────────
    m["numerics.compute.backend"]              = compute_backend_name(c.extensions.compute.backend);
    m["hardware.memory_admission"]             = memory_admission_policy_name(c.extensions.compute.memory_admission);
    m["numerics.compute.allow_fallback"]        = c.extensions.compute.allow_fallback ? "true" : "false";
    if (c.extensions.compute.cuda_assist) m["numerics.compute.cuda_assist"] = "true";
    m["numerics.compute.validate_parity"]      = c.extensions.compute.validate_parity ? "true" : "false";
    m["numerics.compute.gpu_apply_microphysics"] =
        c.extensions.compute.gpu_apply_microphysics ? "true" : "false";
    m["numerics.compute.gpu_apply_slow"] =
        c.extensions.compute.gpu_apply_slow ? "true" : "false";
    m["numerics.compute.gpu_apply_sgs"] =
        c.extensions.compute.gpu_apply_sgs ? "true" : "false";
    m["chaos.scheme"]                          = chaos_scheme_name(c.extensions.chaos.scheme);
    m["chaos.seed"]                            = fmt_int_to_string(c.extensions.chaos.seed);
    m["chaos.member_id"]                       = fmt_int_to_string(c.extensions.chaos.member_id);
    m["chaos.native_generation_required"] =
        c.extensions.chaos.native_generation_required ? "true" : "false";
    if (!c.extensions.chaos.apply_to_ic.empty())
    {
        std::string targets;
        for (const std::string& target : c.extensions.chaos.apply_to_ic)
        {
            if (!targets.empty()) targets += ",";
            targets += target;
        }
        m["chaos.apply_to_ic"] = targets;
    }
    if (!c.extensions.chaos.apply_to_tendencies.empty())
    {
        std::string targets;
        for (const std::string& target : c.extensions.chaos.apply_to_tendencies)
        {
            if (!targets.empty()) targets += ",";
            targets += target;
        }
        m["chaos.apply_to_tendencies"] = targets;
    }
    for (const auto& [target, amplitude] : c.extensions.chaos.sigma_ic)
        m["chaos.sigma_ic." + target] = format_double_roundtrip(amplitude);
    for (const auto& [target, amplitude] : c.extensions.chaos.alpha_tend)
        m["chaos.alpha_tend." + target] = format_double_roundtrip(amplitude);
    m["execution.mode"]                        = execution_mode_name(c.extensions.execution.mode);
    m["checkpoint.every_s"]                    = fmt_int_to_string(c.extensions.checkpoint.every_s);
    m["checkpoint.keep_last"]                  = fmt_int_to_string(c.extensions.checkpoint.keep_last);
    m["checkpoint.minimum_free_space_mib"]     = fmt_int_to_string(c.extensions.checkpoint.minimum_free_space_mib);
    if (!c.extensions.checkpoint.directory.empty())
    {
        m["checkpoint.directory"] = c.extensions.checkpoint.directory;
    }
    if (!c.extensions.checkpoint.resume_from.empty())
    {
        m["checkpoint.resume_from"] = c.extensions.checkpoint.resume_from;
    }
    m["validation.guard_mode"] = validation_guard_mode_name(c.extensions.validation.guard_mode);
    if (c.extensions.execution.mode == ExecutionMode::Qualified)
    {
        // Qualified execution is a fail-closed policy, not a presentation
        // label. It overrides recovery-capable knobs in the projected runtime
        // map so raw imports and future callers cannot retain silent fallback.
        m["validation.guard_mode"]                      = "error";
        m["numerics.compute.allow_fallback"]            = "false";
        m["environment.sounding.use_fallback_profiles"] = "false";
    }
    if (c.extensions.diagnostics.scientific_every_steps >= 0)
    {
        m["diagnostics.scientific_every_steps"] = fmt_int_to_string(c.extensions.diagnostics.scientific_every_steps);
    }
    if (c.extensions.diagnostics.deterministic_reductions)
    {
        m["diagnostics.deterministic_reductions"] = "true";
    }
    if (c.extensions.performance.timing.has_value())
    {
        m["performance.timing"] = fmt_bool_to_string(*c.extensions.performance.timing);
    }
    if (c.extensions.performance.report_every_steps.has_value())
    {
        m["performance.report_every_steps"] = fmt_int_to_string(*c.extensions.performance.report_every_steps);
    }
    if (c.extensions.performance.warmup_steps.has_value())
    {
        m["performance.warmup_steps"] = fmt_int_to_string(*c.extensions.performance.warmup_steps);
    }

    return m;
}

std::string format_warnings(const std::string& file, const std::vector<MigrationWarning>& warnings)
{
    std::ostringstream oss;
    for (size_t i = 0; i < warnings.size(); ++i)
    {
        const auto& w = warnings[i];
        oss << file;
        if (w.line > 0)
            oss << ':' << w.line;
        oss << ": ";
        if (!w.key.empty())
            oss << w.key << ": ";
        oss << w.action;
        if (!w.reason.empty())
            oss << " (" << w.reason << ")";
        if (i + 1 < warnings.size())
            oss << '\n';
    }
    return oss.str();
}
} // namespace tmv::init::config
