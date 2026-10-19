#include <catch2/catch.hpp>

#include "chaos/factory.hpp"
#include "core/infra/domain_recycling.hpp"
#include "core/infra/nested_execution.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/checkpoint_format.hpp"
#include "core/runtime/simulation_checkpoint.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/base/topography.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "terrain/factory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>

namespace
{

SimulationStorageState sample_state()
{
    SimulationStorageState state(GridStorageState(3, 2, 2, 125.0, 80.0, 0.25, 140.0));
    state.grid.geometry.initialize(3, 2, 2, 125.0, 80.0, 140.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.time_s                            = 17.5;
    state.coriolis_f                        = 8.1e-5;
    state.configuration.microphysics_scheme = "milbrandt";
    state.configuration.execution_mode      = tmv::init::config::ExecutionMode::Qualified;
    state.configuration.microphysics_config.shared.autoconversion_rate_s_inv           = 4.5e-4;
    state.configuration.microphysics_config.kessler.evaporation_efficiency             = 0.65;
    state.configuration.microphysics_config.thompson.evaporation_rate_s_inv            = 7.0e-4;
    state.configuration.microphysics_config.thompson.water_friendly_aerosol_surface_m3 = 325.0e6;
    state.configuration.microphysics_config.thompson.ice_friendly_aerosol_surface_m3   = 1.75e6;
    state.configuration.microphysics_config.milbrandt.evaporation_rate_s_inv           = 6.0e-4;
    state.configuration.microphysics_config.milbrandt.triple_moment                    = false;
    state.configuration.bubble_cloud_water_kgkg                                        = 0.0015;
    state.configuration.dynamics_scheme_name                                           = "klemp_wilhelmson_split";
    state.configuration.validation_report_path                                         = "reports/validation.json";
    state.configuration.configured_dt_large_s                                          = 0.25;
    state.configuration.storm_tracking.enabled                                         = true;
    state.configuration.storm_tracking.recycle_domain                                  = true;
    state.configuration.storm_tracking.interval_s                                      = 45.0;
    state.configuration.storm_tracking.inner_margin_m                                  = 18000.0;
    state.configuration.storm_tracking.maximum_shift_cells                             = 7;
    state.configuration.storm_tracking.cooldown_s                                      = 240.0;
    state.configuration.storm_tracking.minimum_confidence                              = 0.85;
    state.configuration.storm_tracking.minimum_updraft_ms                              = 6.5;
    state.configuration.storm_tracking.minimum_condensate_path_kgm2                    = 0.2;
    state.configuration.storm_tracking.updraft_helicity_bottom_m_agl                   = 1500.0;
    state.configuration.storm_tracking.updraft_helicity_top_m_agl                      = 4500.0;
    state.configuration.storm_tracking.minimum_absolute_updraft_helicity_m2s2          = 40.0;
    state.configuration.storm_tracking.minimum_component_cells                         = 12;
    state.configuration.storm_tracking.minimum_core_area_m2                            = 3.5e6;
    state.configuration.storm_tracking.maximum_motion_speed_ms                         = 68.0;
    state.configuration.storm_tracking.maximum_motion_acceleration_ms2                 = 0.18;
    state.configuration.storm_tracking.minimum_continuity_samples                      = 4;
    state.configuration.storm_tracking.lost_after_samples                              = 5;
    state.configuration.storm_tracking.topology_confirmation_samples                   = 3;
    state.configuration.storm_tracking.branch_preference = tmv::init::config::StormBranchPreference::BunkersLeft;
    state.configuration.storm_tracking.branch_velocity_uncertainty_ms = 4.5;
    state.configuration.domain_motion_mode                = tmv::init::config::DomainMotionMode::ConstantVelocity;
    state.configuration.domain_frame.earth_origin_at_t0_m = {125000.0, -48000.0};
    state.configuration.domain_frame.velocity_ms          = {18.5, -7.25};
    state.configuration.validation_policy.field_overrides["theta"] = {true, true, 180.0, 380.0};

    auto initialize_field = [](Field3D& field, float base)
    {
        field.resize(3, 2, 2, 0.0f);
        for (std::size_t index = 0; index < field.size(); ++index)
            field.data()[index] = base + static_cast<float>(index) * 0.125f;
    };
    initialize_field(state.fields.rho, 1.0f);
    initialize_field(state.fields.p, 90000.0f);
    initialize_field(state.fields.theta, 302.0f);
    initialize_field(state.fields.qv, 0.012f);
    initialize_field(state.fields.qr, 0.001f);
    initialize_field(state.tendencies.dtheta_dt_rad, -0.01f);
    initialize_field(state.tendencies.dqc_dt_pbl, -2.0e-6f);
    initialize_field(state.tendencies.dqi_dt_pbl, 3.0e-6f);
    initialize_field(state.diagnostics.radar_reflectivity, 25.0f);
    initialize_field(state.nested.qi, 0.002f);
    initialize_field(state.nested.qs, 0.003f);
    state.base.rho0                                  = {1.1, 1.0};
    state.base.p0                                    = {95000.0, 90000.0};
    state.base.theta0                                = {300.0, 301.0};
    state.surface.ustar                              = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    state.surface.precipitation_rate_mm_h            = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    state.surface.accumulated_rainfall_mm            = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    state.surface.accumulated_rainfall_residual_mm   = {1.0e-8f, -2.0e-8f, 3.0e-8f, -4.0e-8f, 5.0e-8f, -6.0e-8f};
    state.surface.accumulated_rainfall_last_update_s = 17.0;
    state.nested.config.enabled                      = false;

    state.runtime.compute.config.backend      = "vulkan";
    state.runtime.compute.config.device_index = 2;
    state.runtime.compute.requested_kind      = ComputeBackendKind::Vulkan;
    nimbus::parity::ParityFieldStats parity_stats;
    parity_stats.tier           = nimbus::parity::Tier::Stencil;
    parity_stats.domain         = nimbus::parity::ComparisonDomain::FullField;
    parity_stats.samples        = 4;
    parity_stats.cells_compared = 24;
    parity_stats.worst_abs      = 2.5e-6;
    state.runtime.parity.stats.emplace(
        nimbus::parity::ParityRuntimeState::Key{"slow_tendency", "theta", nimbus::parity::ComparisonDomain::FullField},
        parity_stats);
    state.runtime.parity.call_counters["slow_tendency"]                  = 9;
    state.runtime.parity.total_samples                                   = 4;
    state.runtime.boundary_conditions.radiative_u_west                   = {1.0f, 2.0f};
    state.runtime.boundary_conditions.radiative_u_east                   = {3.0f, 4.0f};
    state.runtime.boundary_conditions.radiative_v_south                  = {5.0f, 6.0f};
    state.runtime.boundary_conditions.radiative_v_north                  = {7.0f, 8.0f};
    state.runtime.radiation.config.scheme_id                             = "simple_grey";
    state.runtime.radiation.config.clear_sky_only                        = false;
    state.runtime.radiation.config.use_subcycling                        = true;
    state.runtime.radiation.config.subcycle_count                        = 3;
    state.runtime.radiation.config.enable_diagnostics                    = true;
    state.runtime.radiation.config.simple_grey_optical_model             = "absorbing_water_cloud_path_v1";
    state.runtime.radiation.config.simple_grey_lw_vapor_absorption_m2kg  = 0.03;
    state.runtime.radiation.config.simple_grey_lw_liquid_absorption_m2kg = 82.0;
    state.runtime.radiation.config.simple_grey_lw_ice_absorption_m2kg    = 43.0;
    state.runtime.radiation.config.simple_grey_sw_vapor_absorption_m2kg  = 0.002;
    state.runtime.radiation.config.simple_grey_sw_liquid_absorption_m2kg = 18.0;
    state.runtime.radiation.config.simple_grey_sw_ice_absorption_m2kg    = 11.0;
    state.runtime.radiation.config.latitude_deg                          = -12.5;
    state.runtime.radiation.config.declination_deg                       = -20.0;
    state.runtime.radiation.config.local_solar_time_at_start_h           = 7.25;
    state.runtime.radiation.config.rrtmg_ozone_vmr                       = 4.0e-6;
    state.runtime.radiation.config.rrtmg_co2_vmr                         = 421.0e-6;
    state.runtime.radiation.config.rrtmg_ch4_vmr                         = 1.91e-6;
    state.runtime.radiation.config.rrtmg_n2o_vmr                         = 0.336e-6;
    state.runtime.radiation.config.rrtmg_co_vmr                          = 0.1e-6;
    state.runtime.radiation.config.rrtmg_o2_vmr                          = 0.2094;
    state.runtime.radiation.config.rrtmg_n2_vmr                          = 0.7807;
    state.runtime.radiation.config.rrtmg_cfc11_vmr                       = 2.0e-10;
    state.runtime.radiation.config.rrtmg_cfc12_vmr                       = 5.0e-10;
    state.runtime.radiation.config.rrtmg_cfc22_vmr                       = 3.0e-10;
    state.runtime.radiation.config.rrtmg_ccl4_vmr                        = 1.0e-10;
    state.runtime.radiation.config.rrtmg_liquid_effective_radius_um      = 11.0;
    state.runtime.radiation.config.rrtmg_ice_effective_radius_um         = 35.0;
    state.runtime.radiation.scheme                                       = create_radiation_scheme("simple_grey");
    state.runtime.radiation.scheme->initialize(state.runtime.radiation.config);
    state.runtime.radiation.last_update_time_s       = 15.0;
    state.runtime.radiation.next_update_time_s       = 300.0;
    state.runtime.radiation.held_atmospheric_power_w = -1250.0;
    state.runtime.radiation.held_cell_power_w.resize(3, 2, 2, -125.0f);
    auto& radiation_fluxes     = state.runtime.radiation.held_flux_diagnostics;
    radiation_fluxes.columns_x = 3;
    radiation_fluxes.columns_y = 2;
    radiation_fluxes.levels    = 2;
    radiation_fluxes.upward_longwave_wm2.assign(18, 210.0);
    radiation_fluxes.downward_longwave_wm2.assign(18, 110.0);
    radiation_fluxes.upward_shortwave_wm2.assign(18, 45.0);
    radiation_fluxes.downward_shortwave_wm2.assign(18, 520.0);
    state.runtime.radiation.last_expected_atmospheric_energy_j              = -625.0;
    state.runtime.radiation.last_applied_radiation_heat_j                   = -624.9999;
    state.runtime.radiation.accumulated_expected_atmospheric_energy_j       = -8125.0;
    state.runtime.radiation.accumulated_applied_radiation_heat_j            = -8124.999;
    state.runtime.radiation.applied_step_count                              = 13;
    state.runtime.radiation.config.terrain_sw_cosine_nodes                  = 12;
    state.runtime.radiation.config.terrain_sw_azimuth_nodes                 = 24;
    state.runtime.radiation.config.terrain_sw_maximum_bounces               = 48;
    state.runtime.radiation.config.terrain_sw_relative_residual_tolerance   = 2.0e-11;
    state.runtime.radiation.config.terrain_sw_absolute_residual_tolerance_w = 3.0e-8;
    state.runtime.radiation.held_shortwave_incoming_power_w                 = 1200.0;
    state.runtime.radiation.held_shortwave_atmospheric_absorbed_power_w     = 200.0;
    state.runtime.radiation.held_surface_absorbed_power_w                   = 700.0;
    state.runtime.radiation.held_top_escape_power_w                         = 250.0;
    state.runtime.radiation.held_lateral_escape_power_w                     = 50.0;
    state.runtime.radiation.held_unresolved_reflected_power_w               = 0.0;
    state.runtime.radiation.held_reflection_converged                       = true;
    state.runtime.turbulence.config.scheme_id                               = "none";
    state.runtime.turbulence.scheme                                         = create_turbulence_scheme("none");
    state.runtime.turbulence.scheme->initialize(state.runtime.turbulence.config);
    state.runtime.turbulence.last_update_time_s = 18.0;
    state.runtime.turbulence.next_update_time_s = 20.0;
    state.runtime.turbulence.held_tendencies.dudt_sgs.resize(3, 2, 2, 1.25f);
    state.runtime.turbulence.held_tendencies.dvdt_sgs.resize(3, 2, 2, 2.25f);
    state.runtime.turbulence.held_tendencies.dwdt_sgs.resize(3, 2, 2, 3.25f);
    state.runtime.turbulence.held_tendencies.dthetadt_sgs.resize(3, 2, 2, 4.25f);
    state.runtime.turbulence.held_tendencies.dqvdt_sgs.resize(3, 2, 2, 5.25f);
    state.runtime.turbulence.held_tendencies.dtkedt_sgs.resize(3, 2, 2, 6.25f);
    state.runtime.turbulence.held_tendencies.resident_source_unchanged = true;
    state.runtime.turbulence.last_momentum_work                        = {-19.0, 0.75, -0.002, -18.252};
    auto& sgs_energy                                                   = state.runtime.turbulence.last_momentum_work;
    sgs_energy.moist_dry_internal_change_j                             = 1.0;
    sgs_energy.moist_water_sensible_change_j                           = 2.0;
    sgs_energy.moist_phase_reference_change_j                          = 3.0;
    sgs_energy.moist_carrier_kinetic_change_j                          = 4.0;
    sgs_energy.moist_gravitational_change_j                            = 5.0;
    sgs_energy.moist_subgrid_tke_change_j                              = 6.0;
    sgs_energy.moist_committed_energy_change_j = sgs_energy.committed_work_j + 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0;
    sgs_energy.moist_energy_closure_residual_j =
        sgs_energy.moist_committed_energy_change_j -
        (sgs_energy.committed_work_j + sgs_energy.moist_dry_internal_change_j +
         sgs_energy.moist_water_sensible_change_j + sgs_energy.moist_phase_reference_change_j +
         sgs_energy.moist_carrier_kinetic_change_j + sgs_energy.moist_gravitational_change_j +
         sgs_energy.moist_subgrid_tke_change_j);
    sgs_energy.moist_energy_complete                                      = true;
    state.runtime.turbulence.cumulative_requested_momentum_work_j         = -190.0;
    state.runtime.turbulence.cumulative_finite_step_momentum_correction_j = 7.5;
    state.runtime.turbulence.cumulative_momentum_storage_rounding_j       = -0.02;
    state.runtime.turbulence.cumulative_committed_momentum_work_j         = -182.52;
    auto& cumulative_sgs                          = state.runtime.turbulence.cumulative_moist_energy_work;
    cumulative_sgs                                = {-190.0, 7.5, -0.02, -182.52};
    cumulative_sgs.moist_dry_internal_change_j    = 10.0;
    cumulative_sgs.moist_water_sensible_change_j  = 20.0;
    cumulative_sgs.moist_phase_reference_change_j = 30.0;
    cumulative_sgs.moist_carrier_kinetic_change_j = 40.0;
    cumulative_sgs.moist_gravitational_change_j   = 50.0;
    cumulative_sgs.moist_subgrid_tke_change_j     = 60.0;
    cumulative_sgs.moist_committed_energy_change_j =
        cumulative_sgs.committed_work_j + 10.0 + 20.0 + 30.0 + 40.0 + 50.0 + 60.0;
    cumulative_sgs.moist_energy_closure_residual_j =
        cumulative_sgs.moist_committed_energy_change_j -
        (cumulative_sgs.committed_work_j + cumulative_sgs.moist_dry_internal_change_j +
         cumulative_sgs.moist_water_sensible_change_j + cumulative_sgs.moist_phase_reference_change_j +
         cumulative_sgs.moist_carrier_kinetic_change_j + cumulative_sgs.moist_gravitational_change_j +
         cumulative_sgs.moist_subgrid_tke_change_j);
    cumulative_sgs.moist_energy_complete                           = true;
    state.runtime.turbulence.momentum_work_step_count              = 10;
    state.runtime.turbulence.last_explicit_stability_limit_s       = 0.375;
    state.runtime.turbulence.last_explicit_substep_count           = 4;
    state.runtime.turbulence.last_open_boundary_work_j             = 12.0;
    state.runtime.turbulence.last_irreversible_dissipation_j       = 15.0;
    state.runtime.turbulence.last_stress_closure_residual_j        = 1.0e-12;
    state.runtime.turbulence.cumulative_open_boundary_work_j       = 120.0;
    state.runtime.turbulence.cumulative_irreversible_dissipation_j = 150.0;
    state.runtime.turbulence.cumulative_stress_closure_residual_j  = 1.0e-11;
    state.runtime.turbulence.open_boundary_stability_step_count    = 10;
    state.runtime.boundary_layer.config.scheme_id                  = "slab";
    state.runtime.boundary_layer.scheme                            = create_boundary_layer_scheme("slab");
    state.runtime.boundary_layer.scheme->initialize(state.runtime.boundary_layer.config);
    state.runtime.boundary_layer.next_update_time_s = 30.0;
    state.runtime.boundary_layer.skin_nr            = 3;
    state.runtime.boundary_layer.skin_nth           = 2;
    state.runtime.boundary_layer.skin_theta_2d.assign(6, 301.0f);
    state.runtime.boundary_layer.skin_qv_2d.assign(6, 0.012f);
    state.runtime.boundary_layer.slab_nr  = 3;
    state.runtime.boundary_layer.slab_nth = 2;
    state.runtime.boundary_layer.slab_columns.resize(6);
    state.runtime.boundary_layer.slab_columns[2] = {750.0, 301.5, 0.011, true};
    state.runtime.boundary_layer.held_u_force_n.resize(3, 2, 2, 1.0f);
    state.runtime.boundary_layer.held_v_force_n.resize(3, 2, 2, 2.0f);
    state.runtime.boundary_layer.held_theta_inventory_rate_kgk_s.resize(3, 2, 2, 3.0f);
    state.runtime.boundary_layer.held_vapor_mass_rate_kg_s.resize(3, 2, 2, 4.0f);
    state.runtime.boundary_layer.held_cloud_liquid_mass_rate_kg_s.resize(3, 2, 2, 5.0f);
    state.runtime.boundary_layer.held_cloud_ice_mass_rate_kg_s.resize(3, 2, 2, 6.0f);

    state.runtime.chaos.config.scheme_id = "none";
    state.runtime.chaos.scheme           = chaos::create_chaos_scheme("none");
    GridMetrics metrics;
    metrics.dx = state.grid.dx;
    metrics.dy = state.grid.dy;
    metrics.dz.assign(2, state.grid.dz);
    metrics.z_int = state.grid.geometry.z_faces;
    state.runtime.chaos.scheme->initialize(state.runtime.chaos.config, metrics);
    state.runtime.chaos.noise_elapsed_s               = 7.0;
    state.runtime.chaos.diagnostics.warnings          = {"checkpoint warning"};
    state.runtime.chaos.diagnostics.time_noise_update = 0.03125;
    state.runtime.chaos.diagnostics.time_apply_ic     = 0.0625;
    state.runtime.chaos.diagnostics.time_apply_tend   = 0.125;
    state.runtime.chaos.pbl_base_tendencies_valid     = true;
    state.runtime.chaos.base_du_dt_pbl.resize(3, 2, 2, 1.0f);
    state.runtime.chaos.base_dv_dt_pbl.resize(3, 2, 2, 2.0f);
    state.runtime.chaos.base_dtheta_dt_pbl.resize(3, 2, 2, 3.0f);
    state.runtime.chaos.base_dqv_dt_pbl.resize(3, 2, 2, 4.0f);
    state.runtime.chaos.base_dtke_dt_pbl.resize(3, 2, 2, 5.0f);
    state.runtime.chaos.base_sfc_ustar.assign(6, 6.0f);
    state.runtime.chaos.base_sfc_tau_u.assign(6, 7.0f);
    state.runtime.chaos.base_sfc_tau_v.assign(6, 8.0f);
    state.runtime.chaos.base_sfc_sensible_heat_flux.assign(6, 9.0f);
    state.runtime.chaos.base_sfc_latent_heat_flux.assign(6, 10.0f);
    state.runtime.chaos.base_sfc_moisture_flux.assign(6, 11.0f);

    state.runtime.microphysics.scheme = create_microphysics_scheme("milbrandt");
    state.runtime.microphysics.last_water_repair_report.species[0].lower_bound_count                    = 3;
    state.runtime.microphysics.last_water_repair_report.species[0].lower_bound_roundoff_count           = 2;
    state.runtime.microphysics.last_water_repair_report.species[0].lower_bound_roundoff_correction_kg   = 1.25e-8;
    state.runtime.microphysics.last_water_repair_report.species[0].storage_rounding_kg                  = -2.5e-9;
    state.runtime.microphysics.last_thermodynamic_repair_report.microphysics_heat_j                     = 42.5;
    state.runtime.microphysics.last_thermodynamic_repair_report.theta_storage_residual_j                = 0.125;
    state.runtime.microphysics.last_thermodynamic_repair_report.theta_storage_residual_release_j        = -0.0625;
    state.runtime.microphysics.last_thermodynamic_repair_report.boundary_layer_cloud_liquid_tendency_kg = -0.375;
    state.runtime.microphysics.last_thermodynamic_repair_report.boundary_layer_cloud_ice_tendency_kg    = 0.625;
    auto& moist_commit                           = state.runtime.microphysics.last_thermodynamic_repair_report;
    moist_commit.moist_dry_internal_change_j     = 1.0;
    moist_commit.moist_water_sensible_change_j   = 2.0;
    moist_commit.moist_phase_reference_change_j  = 3.0;
    moist_commit.moist_carrier_kinetic_change_j  = 4.0;
    moist_commit.moist_gravitational_change_j    = 5.0;
    moist_commit.moist_subgrid_tke_change_j      = 0.0;
    moist_commit.moist_committed_energy_change_j = 15.25;
    moist_commit.moist_energy_closure_residual_j = 0.25;
    moist_commit.moist_energy_complete           = true;
    state.runtime.microphysics.last_thermodynamic_repair_report.microphysics_water_tendency_kg_by_species = {
        -7.0, 1.0, 2.0, 0.5, 1.5, 1.25, 0.75};
    state.runtime.microphysics.last_conservation_report = {true, 2.0e-12, 4.0, 0.25, 5.0e-13};
    state.runtime.microphysics.theta_storage_residual.resize(3, 2, 2, 1.25e-6f);
    state.runtime.terrain.config.scheme_id = "none";
    state.runtime.terrain.scheme           = create_terrain_scheme("none");
    state.runtime.terrain.scheme->initialize(state.runtime.terrain.config);
    state.runtime.advection.config.scheme_id = "tvd";
    state.runtime.advection.scheme           = create_advection_scheme("tvd");
    state.runtime.advection.scheme->initialize(state.runtime.advection.config);
    for (std::size_t scalar = 0; scalar < tmv::thermodynamic_boundary_report_count; ++scalar)
    {
        const double base                                              = static_cast<double>(scalar + 1);
        state.runtime.advection.thermodynamic_boundary_reports[scalar] = {base, -base, -base + 1.0e-5, 1.0e-5, true};
        state.runtime.advection.cumulative_thermodynamic_boundary_reports[scalar] = {
            10.0 * base, -10.0 * base, -10.0 * base + 1.0e-4, 1.0e-4, true};
        auto& energy                       = state.runtime.advection.thermodynamic_moist_energy_reports[scalar];
        energy.dry_internal_change_j       = base;
        energy.water_sensible_change_j     = 2.0 * base;
        energy.phase_reference_change_j    = 3.0 * base;
        energy.carrier_kinetic_change_j    = 4.0 * base;
        energy.gravitational_change_j      = 5.0 * base;
        energy.density_history_remap_j     = 6.0 * base;
        energy.outward_boundary_energy_j   = 7.0 * base;
        energy.interior_conversion_j       = 8.0 * base;
        energy.limited_face_state_change_j = base;
        energy.requested_state_change_j    = 7.0 * base;
        energy.storage_rounding_j          = 8.0 * base;
        energy.committed_state_change_j    = 15.0 * base;
        energy.complete                    = true;
        auto cumulative_energy             = energy;
        cumulative_energy.committed_state_change_j *= 10.0;
        cumulative_energy.requested_state_change_j *= 10.0;
        cumulative_energy.storage_rounding_j *= 10.0;
        state.runtime.advection.cumulative_thermodynamic_moist_energy_reports[scalar] = cumulative_energy;
    }
    state.runtime.advection.thermodynamic_boundary_reports_complete      = true;
    state.runtime.advection.thermodynamic_boundary_report_step_count     = 10;
    state.runtime.advection.thermodynamic_moist_energy_reports_complete  = true;
    state.runtime.advection.thermodynamic_moist_energy_report_step_count = 10;
    state.runtime.advection.tracer_roundoff_report                       = {
        2.5e-35, 0.0, -11.0f * std::numeric_limits<float>::denorm_min(), 3, 1, 1, true};
    state.runtime.advection.cumulative_tracer_roundoff_report = {
        7.5e-35, 0.0, -17.0f * std::numeric_limits<float>::denorm_min(), 9, 2, 3, true};
    state.runtime.advection.tracer_roundoff_report_step_count        = 10;
    state.runtime.advection.tracer_roundoff_cumulative_history_known = true;
    state.runtime.diffusion.config.scheme_id                         = "explicit";
    state.runtime.diffusion.scheme                                   = create_diffusion_scheme("explicit");
    state.runtime.diffusion.scheme->initialize(state.runtime.diffusion.config);
    state.runtime.diffusion.last_momentum_work       = {-23.0, 0.5, -0.004, -22.504};
    auto& diffusion_energy                           = state.runtime.diffusion.last_momentum_work;
    diffusion_energy.moist_dry_internal_change_j     = 1.0;
    diffusion_energy.moist_water_sensible_change_j   = 2.0;
    diffusion_energy.moist_phase_reference_change_j  = 3.0;
    diffusion_energy.moist_carrier_kinetic_change_j  = 4.0;
    diffusion_energy.moist_gravitational_change_j    = 5.0;
    diffusion_energy.moist_subgrid_tke_change_j      = 0.0;
    diffusion_energy.moist_committed_energy_change_j = diffusion_energy.committed_work_j + 1.0 + 2.0 + 3.0 + 4.0 + 5.0;
    diffusion_energy.moist_energy_closure_residual_j =
        diffusion_energy.moist_committed_energy_change_j -
        (diffusion_energy.committed_work_j + 1.0 + 2.0 + 3.0 + 4.0 + 5.0);
    diffusion_energy.moist_energy_complete                               = true;
    state.runtime.diffusion.cumulative_requested_momentum_work_j         = -230.0;
    state.runtime.diffusion.cumulative_finite_step_momentum_correction_j = 5.0;
    state.runtime.diffusion.cumulative_momentum_storage_rounding_j       = -0.04;
    state.runtime.diffusion.cumulative_committed_momentum_work_j         = -225.04;
    auto& cumulative_diffusion                          = state.runtime.diffusion.cumulative_moist_energy_work;
    cumulative_diffusion                                = {-230.0, 5.0, -0.04, -225.04};
    cumulative_diffusion.moist_dry_internal_change_j    = 10.0;
    cumulative_diffusion.moist_water_sensible_change_j  = 20.0;
    cumulative_diffusion.moist_phase_reference_change_j = 30.0;
    cumulative_diffusion.moist_carrier_kinetic_change_j = 40.0;
    cumulative_diffusion.moist_gravitational_change_j   = 50.0;
    cumulative_diffusion.moist_subgrid_tke_change_j     = 0.0;
    cumulative_diffusion.moist_committed_energy_change_j =
        cumulative_diffusion.committed_work_j + 10.0 + 20.0 + 30.0 + 40.0 + 50.0;
    cumulative_diffusion.moist_energy_closure_residual_j =
        cumulative_diffusion.moist_committed_energy_change_j -
        (cumulative_diffusion.committed_work_j + 10.0 + 20.0 + 30.0 + 40.0 + 50.0);
    cumulative_diffusion.moist_energy_complete       = true;
    state.runtime.diffusion.momentum_work_step_count = 10;
    state.runtime.diffusion.moist_energy_step_count  = 10;
    state.runtime.time_stepping.config.scheme_id     = "rk3";
    state.runtime.time_stepping.scheme               = create_time_stepping_scheme("rk3");
    state.runtime.time_stepping.scheme->initialize(state.runtime.time_stepping.config, nullptr);
    state.runtime.time_stepping.diagnostics.n_steps                          = 70;
    state.runtime.time_stepping.diagnostics.dt_history                       = {0.25, 0.25};
    state.runtime.time_stepping.last_boundary_mass                           = {12.5, -12.5, -12.499, 0.001};
    state.runtime.time_stepping.cumulative_outward_boundary_mass_kg          = 125.0;
    state.runtime.time_stepping.cumulative_flux_implied_state_change_kg      = -125.0;
    state.runtime.time_stepping.cumulative_committed_state_change_kg         = -124.99;
    state.runtime.time_stepping.cumulative_boundary_mass_closure_residual_kg = 0.01;
    state.runtime.time_stepping.boundary_mass_step_count                     = 10;
    auto& acoustic                                            = state.runtime.time_stepping.last_acoustic_energy;
    acoustic.continuity_eos_dry_internal_change_j             = -31.0;
    acoustic.continuity_eos_water_sensible_change_j           = -2.0;
    acoustic.continuity_eos_phase_reference_change_j          = 0.5;
    acoustic.continuity_carrier_kinetic_change_j              = -0.25;
    acoustic.continuity_gravitational_change_j                = 1.5;
    acoustic.continuity_subgrid_tke_change_j                  = -0.125;
    acoustic.continuity_carrier_requested_change_j            = -0.3;
    acoustic.continuity_carrier_outward_boundary_transport_j  = 0.1;
    acoustic.continuity_carrier_interior_conversion_j         = -0.2;
    acoustic.continuity_carrier_storage_rounding_j            = 0.05;
    acoustic.slow_momentum_advection_requested_linear_work_j  = -21.0;
    acoustic.kinetic_transport_interior_conversion_j          = -21.2;
    acoustic.kinetic_transport_closure_residual_j             = 0.0;
    acoustic.kinetic_transport_complete                       = true;
    acoustic.pressure_gradient_requested_linear_work_j        = 27.0;
    acoustic.pressure_gradient_finite_step_correction_j       = 0.75;
    acoustic.pressure_gradient_storage_rounding_j             = -0.002;
    acoustic.pressure_gradient_committed_work_j               = 27.748;
    acoustic.boundary_committed_work_j                        = -0.375;
    acoustic.boundary_radiative_requested_linear_work_j       = -0.5;
    acoustic.boundary_radiative_finite_step_correction_j      = 0.125;
    acoustic.boundary_radiative_storage_rounding_j            = -0.0005;
    acoustic.boundary_radiative_committed_work_j              = -0.3755;
    acoustic.boundary_constraint_history_committed_work_j     = 0.0005;
    acoustic.boundary_application_count                       = 8;
    acoustic.boundary_complete                                = true;
    acoustic.committed_inventory_change_j                     = -3.627;
    acoustic.substep_count                                    = 7;
    acoustic.complete                                         = true;
    auto& cumulative_acoustic                                 = state.runtime.time_stepping.cumulative_acoustic_energy;
    cumulative_acoustic                                       = acoustic;
    cumulative_acoustic.continuity_eos_dry_internal_change_j  = -310.0;
    cumulative_acoustic.continuity_carrier_requested_change_j = -3.0;
    cumulative_acoustic.continuity_carrier_outward_boundary_transport_j = 1.0;
    cumulative_acoustic.continuity_carrier_interior_conversion_j        = -2.0;
    cumulative_acoustic.continuity_carrier_storage_rounding_j           = 0.5;
    cumulative_acoustic.slow_momentum_advection_requested_linear_work_j = -210.0;
    cumulative_acoustic.kinetic_transport_interior_conversion_j         = -212.0;
    cumulative_acoustic.pressure_gradient_committed_work_j              = 277.48;
    cumulative_acoustic.boundary_committed_work_j                       = -3.75;
    cumulative_acoustic.boundary_radiative_requested_linear_work_j      = -5.0;
    cumulative_acoustic.boundary_radiative_finite_step_correction_j     = 1.25;
    cumulative_acoustic.boundary_radiative_storage_rounding_j           = -0.005;
    cumulative_acoustic.boundary_radiative_committed_work_j             = -3.755;
    cumulative_acoustic.boundary_constraint_history_committed_work_j    = 0.005;
    cumulative_acoustic.boundary_application_count                      = 80;
    cumulative_acoustic.committed_inventory_change_j                    = -36.27;
    cumulative_acoustic.substep_count                                   = 70;
    state.runtime.time_stepping.acoustic_energy_step_count              = 10;
    auto& slow_momentum                                 = state.runtime.time_stepping.last_slow_momentum_energy;
    slow_momentum.advection_requested_linear_work_j     = -21.0;
    slow_momentum.nonadvection_requested_linear_work_j  = 4.0;
    slow_momentum.advection_finite_step_correction_j    = 0.75;
    slow_momentum.nonadvection_finite_step_correction_j = 0.25;
    slow_momentum.coupled_finite_step_cross_work_j      = -0.125;
    slow_momentum.storage_rounding_j                    = 0.001;
    slow_momentum.committed_kinetic_change_j            = -16.124;
    slow_momentum.closure_residual_j                    = 0.0;
    slow_momentum.pressure_dry_internal_change_j        = 2.5;
    slow_momentum.pressure_water_sensible_change_j      = 0.125;
    slow_momentum.committed_pressure_energy_change_j    = 2.625;
    slow_momentum.complete                              = true;
    auto& cumulative_slow_momentum                      = state.runtime.time_stepping.cumulative_slow_momentum_energy;
    cumulative_slow_momentum                            = slow_momentum;
    cumulative_slow_momentum.advection_requested_linear_work_j  = -210.0;
    cumulative_slow_momentum.committed_kinetic_change_j         = -161.24;
    cumulative_slow_momentum.pressure_dry_internal_change_j     = 25.0;
    cumulative_slow_momentum.pressure_water_sensible_change_j   = 1.25;
    cumulative_slow_momentum.committed_pressure_energy_change_j = 26.25;
    state.runtime.time_stepping.slow_momentum_energy_step_count = 10;
    auto& eos                                                   = state.runtime.time_stepping.last_eos_pressure_energy;
    eos.dry_internal_change_j                                   = 12.0;
    eos.water_sensible_change_j                                 = 0.5;
    eos.committed_energy_change_j                               = 12.5;
    eos.closure_residual_j                                      = 0.0;
    eos.complete                                                = true;
    auto& cumulative_eos                     = state.runtime.time_stepping.cumulative_eos_pressure_energy;
    cumulative_eos                           = eos;
    cumulative_eos.dry_internal_change_j     = 120.0;
    cumulative_eos.water_sensible_change_j   = 5.0;
    cumulative_eos.committed_energy_change_j = 125.0;
    state.runtime.time_stepping.eos_pressure_energy_step_count = 10;
    const auto seed_sponge_pass                                = [](tmv::SpongePassEnergyReportV1& pass, double scale)
    {
        pass.momentum_requested_linear_work_j  = -10.0 * scale;
        pass.momentum_finite_step_correction_j = 1.0 * scale;
        pass.carrier_kinetic_change_j          = -9.1 * scale;
        pass.momentum_storage_rounding_j       = pass.carrier_kinetic_change_j - pass.momentum_requested_linear_work_j -
                                                 pass.momentum_finite_step_correction_j;
        pass.dry_internal_change_j             = 2.0 * scale;
        pass.water_sensible_change_j           = 0.1 * scale;
        pass.committed_energy_change_j =
            pass.dry_internal_change_j + pass.water_sensible_change_j + pass.carrier_kinetic_change_j;
        pass.closure_residual_j =
            pass.committed_energy_change_j -
            (pass.dry_internal_change_j + pass.water_sensible_change_j + pass.carrier_kinetic_change_j);
        pass.applied  = true;
        pass.complete = true;
    };
    auto& sponge = state.runtime.time_stepping.last_sponge_energy;
    seed_sponge_pass(sponge.upper, 1.0);
    seed_sponge_pass(sponge.w_only, 0.5);
    seed_sponge_pass(sponge.lateral, 2.0);
    sponge.complete         = true;
    auto& cumulative_sponge = state.runtime.time_stepping.cumulative_sponge_energy;
    seed_sponge_pass(cumulative_sponge.upper, 10.0);
    seed_sponge_pass(cumulative_sponge.w_only, 5.0);
    seed_sponge_pass(cumulative_sponge.lateral, 20.0);
    cumulative_sponge.complete                           = true;
    state.runtime.time_stepping.sponge_energy_step_count = 10;
    auto& final_boundary                                 = state.runtime.time_stepping.last_final_boundary_energy;
    final_boundary.constraint_committed_work_j           = -3.0;
    final_boundary.carrier_kinetic_change_j              = -3.0;
    final_boundary.committed_energy_change_j             = -3.0;
    final_boundary.applied                               = true;
    final_boundary.complete                              = true;
    auto& cumulative_final_boundary                      = state.runtime.time_stepping.cumulative_final_boundary_energy;
    cumulative_final_boundary.constraint_committed_work_j        = -30.0;
    cumulative_final_boundary.carrier_kinetic_change_j           = -30.0;
    cumulative_final_boundary.committed_energy_change_j          = -30.0;
    cumulative_final_boundary.applied                            = true;
    cumulative_final_boundary.complete                           = true;
    state.runtime.time_stepping.final_boundary_energy_step_count = 10;
    auto& completed                                    = state.runtime.time_stepping.last_completed_step_energy;
    completed.acoustic_committed_change_j              = 1.0;
    completed.slow_dynamics_committed_change_j         = 2.0;
    completed.scalar_transport_committed_change_j      = 3.0;
    completed.thermodynamic_sources_committed_change_j = 4.0;
    completed.sgs_committed_change_j                   = 5.0;
    completed.diffusion_committed_change_j             = 6.0;
    completed.eos_projection_committed_change_j        = 7.0;
    completed.sponge_committed_change_j                = 8.0;
    completed.final_boundary_committed_change_j        = 9.0;
    completed.committed_owner_sum_j                    = tmv::completed_step_owner_sum(completed);
    completed.dry_internal_change_j                    = 10.0;
    completed.water_sensible_change_j                  = 5.0;
    completed.phase_reference_change_j                 = 4.0;
    completed.carrier_kinetic_change_j                 = 20.0;
    completed.gravitational_change_j                   = 3.0;
    completed.subgrid_tke_change_j                     = 2.0;
    completed.closure_residual_j = completed.committed_owner_sum_j - tmv::completed_step_reservoir_sum(completed);
    completed.complete           = true;
    auto& cumulative_completed   = state.runtime.time_stepping.cumulative_completed_step_energy;
    cumulative_completed         = completed;
    cumulative_completed.acoustic_committed_change_j              = 10.0;
    cumulative_completed.slow_dynamics_committed_change_j         = 20.0;
    cumulative_completed.scalar_transport_committed_change_j      = 30.0;
    cumulative_completed.thermodynamic_sources_committed_change_j = 40.0;
    cumulative_completed.sgs_committed_change_j                   = 50.0;
    cumulative_completed.diffusion_committed_change_j             = 60.0;
    cumulative_completed.eos_projection_committed_change_j        = 70.0;
    cumulative_completed.sponge_committed_change_j                = 80.0;
    cumulative_completed.final_boundary_committed_change_j        = 90.0;
    cumulative_completed.committed_owner_sum_j                    = tmv::completed_step_owner_sum(cumulative_completed);
    cumulative_completed.dry_internal_change_j                    = 100.0;
    cumulative_completed.water_sensible_change_j                  = 50.0;
    cumulative_completed.phase_reference_change_j                 = 40.0;
    cumulative_completed.carrier_kinetic_change_j                 = 200.0;
    cumulative_completed.gravitational_change_j                   = 30.0;
    cumulative_completed.subgrid_tke_change_j                     = 20.0;
    cumulative_completed.closure_residual_j =
        cumulative_completed.committed_owner_sum_j - tmv::completed_step_reservoir_sum(cumulative_completed);
    state.runtime.time_stepping.completed_step_energy_count = 10;
    auto& storm                                             = state.runtime.storm_tracking;
    storm.status                                            = tmv::diagnostics::StormContinuityStatus::Tracking;
    storm.reason                                            = tmv::diagnostics::StormContinuityReason::None;
    storm.sample_count                                      = 4;
    storm.samples[0]                                        = {10.0, 1000.0, 2000.0, 0.91};
    storm.samples[1]                                        = {20.0, 1100.0, 1950.0, 0.92};
    storm.samples[2]                                        = {30.0, 1200.0, 1900.0, 0.93};
    storm.samples[3]                                        = {40.0, 1300.0, 1850.0, 0.94};
    storm.missed_samples                                    = 0;
    storm.branch_id                                         = 4;
    storm.motion_available                                  = true;
    storm.velocity_x_ms                                     = 10.0;
    storm.velocity_y_ms                                     = -5.0;
    storm.speed_ms                                          = std::hypot(10.0, -5.0);
    storm.velocity_uncertainty_ms                           = 0.2;
    storm.continuity_confidence                             = 0.89;
    storm.innovation_distance_m                             = 2.0;
    storm.association_gate_radius_m                         = 14.5;
    storm.topology                                          = tmv::diagnostics::StormBranchTopology::Multiple;
    storm.pending_topology                                  = tmv::diagnostics::StormBranchTopology::Single;
    storm.topology_event                             = tmv::diagnostics::StormBranchTopologyEvent::MergerCandidate;
    storm.pending_topology_samples                   = 1;
    storm.topology_event_id                          = 8;
    storm.selected_candidate_available               = true;
    storm.selected_candidate_rank                    = 1;
    storm.selected_component_label                   = 42;
    storm.selected_motion_class                      = tmv::diagnostics::StormBranchMotionClass::BunkersRight;
    storm.selected_bunkers_right_mismatch_ms         = 1.25;
    storm.selected_bunkers_left_mismatch_ms          = 17.5;
    storm.association_envelope_available             = true;
    storm.association_envelope_west_m                = 900.0;
    storm.association_envelope_east_m                = 1800.0;
    storm.association_envelope_south_m               = 1500.0;
    storm.association_envelope_north_m               = 2400.0;
    state.runtime.storm_selected_partition_available = true;
    state.runtime.storm_selected_partition_nx        = 2;
    state.runtime.storm_selected_partition_ny        = 2;
    state.runtime.storm_selected_partition           = {1U, 1U, 0U, 0U};
    auto& movement                                   = state.runtime.storm_window_movement;
    movement.pending_i_direction                     = 1;
    movement.pending_j_direction                     = -1;
    movement.consecutive_compatible_samples          = 1;
    movement.sample_time_available                   = true;
    movement.last_sample_time_s                      = 40.0;
    movement.committed_move_available                = true;
    movement.last_committed_time_s                   = 10.0;
    movement.cumulative_i_shift_cells                = 12;
    movement.cumulative_j_shift_cells                = -5;
    return state;
}

SimulationStorageState
synchronized_nested_state(
    tmv::init::config::NestingFeedback feedback =
        tmv::init::config::NestingFeedback::OneWay,
    int ratio = 2)
{
    SimulationStorageState state(GridStorageState(8, 8, 2, 1000.0, 300.0, 2.0, 800.0));
    state.grid.geometry.initialize(8, 8, 2, 1000.0, 300.0, 800.0, CoordinateSystem::Cartesian, StaggerType::CGrid);
    state.grid.geometry.z_faces                               = {0.0, 200.0, 550.0};
    state.grid.geometry.z                                     = {100.0, 375.0};
    state.time_s                                              = 2.0;
    state.coriolis_f                                          = 8.0e-5;
    state.configuration.nesting.parent_ratio_x                = ratio;
    state.configuration.nesting.parent_ratio_y                = ratio;
    state.configuration.nesting.parent_time_step_ratio        = ratio;
    state.configuration.nesting.enabled                       = true;
    state.configuration.nesting.feedback                      = feedback;
    state.configuration.nesting.child_origin_parent_cell_i    = 2;
    state.configuration.nesting.child_origin_parent_cell_j    = 2;
    state.configuration.nesting.child_cells_x                 = 2 * ratio;
    state.configuration.nesting.child_cells_y                 = 2 * ratio;
    state.configuration.nesting.relaxation_width_child_cells  = 1;
    state.configuration.nesting.boundary_stencil_parent_cells = 1;
    state.configuration.nesting.relaxation_e_folding_time_s   = 4.0;
    state.base.p0                                             = {90000.0, 85000.0};
    state.base.theta0                                         = {300.0, 302.0};
    state.base.rho0.resize(2);
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    for (int k = 0; k < 2; ++k)
    {
        const double temperature =
            state.base.theta0[k] * std::pow(state.base.p0[k] / physical_constants::reference_pressure_pa, kappa);
        state.base.rho0[k] = state.base.p0[k] / (physical_constants::gas_constant_dry_air_jkgk * temperature);
    }
    state.base.qv0 = {0.01, 0.006};
    state.base.u0  = {2.0, 3.0};
    state.base.v0  = {-1.0, 0.0};

    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = 6;
    input.parent_cells_y                = 6;
    input.parent_levels_z               = 2;
    input.parent_dx_m                   = 1000.0;
    input.parent_dy_m                   = 800.0;
    input.parent_dt_s                   = 2.0;
    input.parent_ratio_x                = ratio;
    input.parent_ratio_y                = ratio;
    input.parent_time_step_ratio        = ratio;
    input.child_origin_parent_cell_i    = 2;
    input.child_origin_parent_cell_j    = 2;
    input.child_cells_x                 = 2 * ratio;
    input.child_cells_y                 = 2 * ratio;
    input.child_levels_z                = 2;
    input.relaxation_width_child_cells  = 1;
    input.boundary_stencil_parent_cells = 1;
    const auto relationship             = tmv::domain::prepare_one_way_nest_relationship(input);
    prepare_nested_execution_relationship(state, relationship, 0.0);
    initialize_nested_shared_level_reference_state(state);

    auto&      child    = state.nested_execution.child;
    const auto allocate = [&](Field3D& field, float value)
    { field.resize(child.grid.nr, child.grid.nth, child.grid.nz, value); };
    allocate(child.fields.rho, 1.0f);
    allocate(child.fields.p, 90000.0f);
    allocate(child.fields.u, 2.0f);
    allocate(child.fields.v, -1.0f);
    allocate(child.fields.w, 0.5f);
    allocate(child.fields.tracer, 0.2f);
    allocate(child.fields.theta, 300.0f);
    allocate(child.fields.qv, 0.01f);
    allocate(child.fields.qc, 0.001f);
    allocate(child.fields.qr, 0.002f);
    allocate(child.fields.qi, 0.0005f);
    allocate(child.fields.qs, 0.0004f);
    allocate(child.fields.qg, 0.0003f);
    allocate(child.fields.qh, 0.0002f);
    allocate(child.fields.tke, 0.25f);
    allocate(child.tendencies.dtheta_dt_rad, -0.01f);
    allocate(child.diagnostics.radar_reflectivity, 25.0f);
    child.surface.ustar.assign(static_cast<std::size_t>(child.grid.nr * child.grid.nth), 0.2f);
    child.surface.precipitation_rate_mm_h.assign(static_cast<std::size_t>(child.grid.nr * child.grid.nth), 3.5f);
    child.surface.accumulated_rainfall_mm.assign(static_cast<std::size_t>(child.grid.nr * child.grid.nth), 1.25f);
    child.surface.accumulated_rainfall_residual_mm.assign(static_cast<std::size_t>(child.grid.nr * child.grid.nth),
                                                          -2.5e-8f);
    child.surface.accumulated_rainfall_last_update_s  = 2.0;
    child.runtime.radiation.last_update_time_s        = 1.5;
    child.runtime.turbulence.momentum_work_step_count = 3;
    child.runtime.advection.tracer_roundoff_report    = {
        1.5e-36, 0.0, -7.0f * std::numeric_limits<float>::denorm_min(), 2, 1, 1, true};
    child.runtime.advection.cumulative_tracer_roundoff_report = {
        4.5e-36, 0.0, -13.0f * std::numeric_limits<float>::denorm_min(), 6, 2, 3, false};
    child.runtime.advection.tracer_roundoff_report_step_count        = 4;
    child.runtime.advection.tracer_roundoff_cumulative_history_known = false;
    state.runtime.microphysics.scheme                                = create_microphysics_scheme("none");
    child.runtime.microphysics.scheme                                = create_microphysics_scheme("none");

    state.nested_execution.physical_primary_state_ready                 = true;
    state.nested_execution.parent_boundary_state_ready                  = true;
    state.nested_execution.interface_source.available                   = true;
    state.nested_execution.interface_source.application_count           = 1;
    auto& interface                                                     = state.nested_execution.interface_source;
    interface.last.theta.target_dry_mass_change_kg                      = 2.0;
    interface.last.theta.committed_dry_mass_change_kg                   = 1.5;
    interface.last.theta.dry_storage_residual_kg                        = 0.5;
    interface.last.energy.before.dry_internal                           = 10.0;
    interface.last.energy.before.total                                  = 10.0;
    interface.last.energy.after.dry_internal                            = 14.0;
    interface.last.energy.after.total                                   = 14.0;
    interface.last.energy.committed_change.dry_internal                 = 4.0;
    interface.last.energy.committed_change.total                        = 4.0;
    interface.cumulative                                                = interface.last;
    state.nested_execution.parent_time_bracket.completed_child_substeps = ratio;
    child.time_s                                                        = state.time_s;
    require_valid_nested_execution_relationship(state);
    return state;
}

void publish_active_terrain(SimulationStorageState& state,
                            const std::string&      scheme_id        = "bell",
                            std::uint32_t           sampling_version = 1)
{
    TerrainConfig config;
    config.scheme_id               = scheme_id;
    config.coord_id                = "btf";
    config.horizontal_placement_id = "grid_attached";
    config.ztop                    = 12000.0;
    config.bell.h0                 = 350.0;
    config.bell.a                  = 2400.0;
    config.schar.H                 = 300.0;
    config.schar.a                 = 2200.0;
    config.schar.ell               = 1400.0;
    const auto& relationship       = state.nested_execution.relationship;

    tmv::domain::NestedMappedConservativeStateV1 parent;
    parent.scalar.resize(relationship.parent_cells_x, relationship.parent_cells_y, relationship.levels_z, 0.0);
    parent.x_flux.resize(relationship.parent_cells_x + 1, relationship.parent_cells_y, relationship.levels_z, 0.0);
    parent.y_flux.resize(relationship.parent_cells_x, relationship.parent_cells_y + 1, relationship.levels_z, 0.0);
    parent.zeta_flux.resize(relationship.parent_cells_x, relationship.parent_cells_y, relationship.levels_z + 1, 0.0);
    const auto fill = [](TerrainField3D& field, double base)
    {
        for (std::size_t index = 0; index < field.values.size(); ++index)
            field.values[index] = base + 0.001 * static_cast<double>(index);
    };
    fill(parent.scalar, 1.25);
    fill(parent.x_flux, 3.0);
    fill(parent.y_flux, -2.0);
    fill(parent.zeta_flux, 0.5);

    auto       origin = state.configuration.domain_frame.earth_origin_at(state.time_s);
    const auto parent_physical =
        physical_scalar_domain(state.grid.nr, state.grid.nth, state.grid.nz, StaggerType::CGrid);
    origin.x_m += parent_physical.i_begin * state.grid.dx;
    origin.y_m += parent_physical.j_begin * state.grid.dy;
    state.nested_execution.active_terrain.geometry.sampling_version = sampling_version;
    auto prepared = tmv::domain::prepare_nested_active_terrain_commit(
        state.nested_execution.active_terrain, config, relationship, origin, parent, state.time_s, 7.0, -1.5);
    REQUIRE(tmv::domain::commit_nested_active_terrain(prepared, state.nested_execution.active_terrain));

    const auto& geometry    = state.nested_execution.active_terrain.geometry;
    auto        set_terrain = [&](TerrainRuntimeState& runtime, const tmv::EarthFixedTerrainGeometry& built)
    {
        runtime.config = config;
        runtime.scheme = create_terrain_scheme(scheme_id);
        runtime.scheme->initialize(config);
        runtime.topography = built.topography;
        runtime.metrics    = built.metrics;
    };
    set_terrain(state.runtime.terrain, geometry.parent);
    set_terrain(state.nested_execution.child.runtime.terrain, geometry.child_storage);
    prepare_nested_child_process_grid_metrics(state.nested_execution);
    require_valid_nested_execution_relationship(state);
}

void check_exact_terrain_field(const TerrainField3D& actual, const TerrainField3D& expected)
{
    REQUIRE(actual.nr == expected.nr);
    REQUIRE(actual.nth == expected.nth);
    REQUIRE(actual.nz == expected.nz);
    REQUIRE(actual.values == expected.values);
}

void strip_active_terrain_nesting_v8_suffix(tmv::checkpoint::wire::Section& nesting)
{
    REQUIRE(nesting.version == 8);
    REQUIRE(nesting.payload.size() >= 17);
    const auto read_u64 = [&](std::size_t offset)
    {
        std::uint64_t value = 0;
        for (std::size_t byte = 0; byte < sizeof(value); ++byte)
            value |= static_cast<std::uint64_t>(nesting.payload[offset + byte]) << (8U * byte);
        return value;
    };
    const std::uint64_t size = read_u64(nesting.payload.size() - sizeof(std::uint64_t));
    REQUIRE(size <= nesting.payload.size() - 16);
    const std::size_t start = nesting.payload.size() - static_cast<std::size_t>(size) - 16;
    REQUIRE(read_u64(start) == size);
    nesting.payload.resize(start);
    nesting.version = 7;
}

tmv::checkpoint::wire::Encoder movement_authority_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    movement = source.runtime.storm_window_movement;
    suffix.boolean(movement.armed);
    suffix.integer(movement.pending_i_direction);
    suffix.integer(movement.pending_j_direction);
    suffix.unsigned_value<std::uint64_t>(movement.consecutive_compatible_samples);
    suffix.boolean(movement.sample_time_available);
    suffix.floating(movement.last_sample_time_s);
    suffix.boolean(movement.committed_move_available);
    suffix.floating(movement.last_committed_time_s);
    suffix.unsigned_value<std::uint64_t>(static_cast<std::uint64_t>(movement.cumulative_i_shift_cells));
    suffix.unsigned_value<std::uint64_t>(static_cast<std::uint64_t>(movement.cumulative_j_shift_cells));
    return suffix;
}

tmv::checkpoint::wire::Encoder chaos_pbl_cache_v37_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto                     encode_field = [&](const Field3D& field)
    {
        suffix.integer(field.size_r());
        suffix.integer(field.size_th());
        suffix.integer(field.size_z());
        suffix.unsigned_value<std::uint64_t>(field.size());
        for (std::size_t index = 0; index < field.size(); ++index)
            suffix.floating(field.data()[index]);
    };
    const auto encode_vector = [&](const std::vector<float>& values)
    {
        suffix.unsigned_value<std::uint64_t>(values.size());
        for (float value : values)
            suffix.floating(value);
    };
    const auto& chaos = source.runtime.chaos;
    encode_field(chaos.base_dtke_dt_pbl);
    encode_vector(chaos.base_sfc_ustar);
    encode_vector(chaos.base_sfc_tau_u);
    encode_vector(chaos.base_sfc_tau_v);
    encode_vector(chaos.base_sfc_sensible_heat_flux);
    encode_vector(chaos.base_sfc_latent_heat_flux);
    encode_vector(chaos.base_sfc_moisture_flux);
    return suffix;
}

tmv::checkpoint::wire::Encoder open_stress_v38_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    turbulence = source.runtime.turbulence;
    suffix.floating(turbulence.last_open_boundary_work_j);
    suffix.floating(turbulence.last_irreversible_dissipation_j);
    suffix.floating(turbulence.last_stress_closure_residual_j);
    suffix.floating(turbulence.cumulative_open_boundary_work_j);
    suffix.floating(turbulence.cumulative_irreversible_dissipation_j);
    suffix.floating(turbulence.cumulative_stress_closure_residual_j);
    suffix.unsigned_value(turbulence.open_boundary_stability_step_count);
    return suffix;
}

tmv::checkpoint::wire::Encoder terrain_shortwave_v39_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    radiation = source.runtime.radiation;
    suffix.unsigned_value<std::uint64_t>(radiation.config.terrain_sw_cosine_nodes);
    suffix.unsigned_value<std::uint64_t>(radiation.config.terrain_sw_azimuth_nodes);
    suffix.unsigned_value<std::uint64_t>(radiation.config.terrain_sw_maximum_bounces);
    suffix.floating(radiation.config.terrain_sw_relative_residual_tolerance);
    suffix.floating(radiation.config.terrain_sw_absolute_residual_tolerance_w);
    suffix.floating(radiation.held_shortwave_incoming_power_w);
    suffix.floating(radiation.held_shortwave_atmospheric_absorbed_power_w);
    suffix.floating(radiation.held_surface_absorbed_power_w);
    suffix.floating(radiation.held_top_escape_power_w);
    suffix.floating(radiation.held_lateral_escape_power_w);
    suffix.floating(radiation.held_unresolved_reflected_power_w);
    suffix.boolean(radiation.held_reflection_converged);
    return suffix;
}

tmv::checkpoint::wire::Encoder tracer_roundoff_v40_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto                     encode_report = [&suffix](const tmv::PositiveDefiniteScalarRoundoffReport& report)
    {
        suffix.unsigned_value<std::uint32_t>(tmv::PositiveDefiniteScalarRoundoffReport::report_schema_version);
        suffix.unsigned_value<std::uint32_t>(tmv::PositiveDefiniteScalarRoundoffReport::classification_policy_version);
        suffix.floating(report.physical_inventory_correction_kg);
        suffix.floating(report.thermodynamic_energy_correction_j);
        suffix.floating(report.most_negative_repaired_value);
        suffix.unsigned_value<std::uint64_t>(report.negative_subnormal_count);
        suffix.unsigned_value<std::uint64_t>(report.negative_zero_count);
        suffix.unsigned_value<std::uint64_t>(report.closure_canonicalization_count);
        suffix.boolean(report.complete);
    };
    encode_report(source.runtime.advection.tracer_roundoff_report);
    encode_report(source.runtime.advection.cumulative_tracer_roundoff_report);
    suffix.unsigned_value(source.runtime.advection.tracer_roundoff_report_step_count);
    suffix.boolean(source.runtime.advection.tracer_roundoff_cumulative_history_known);
    return suffix;
}

tmv::checkpoint::wire::Encoder rrtmg_v41_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    c = source.runtime.radiation.config;
    suffix.floating(c.rrtmg_ozone_vmr);
    suffix.floating(c.rrtmg_co2_vmr);
    suffix.floating(c.rrtmg_ch4_vmr);
    suffix.floating(c.rrtmg_n2o_vmr);
    suffix.floating(c.rrtmg_co_vmr);
    suffix.floating(c.rrtmg_o2_vmr);
    suffix.floating(c.rrtmg_n2_vmr);
    suffix.floating(c.rrtmg_cfc11_vmr);
    suffix.floating(c.rrtmg_cfc12_vmr);
    suffix.floating(c.rrtmg_cfc22_vmr);
    suffix.floating(c.rrtmg_ccl4_vmr);
    suffix.floating(c.rrtmg_liquid_effective_radius_um);
    suffix.floating(c.rrtmg_ice_effective_radius_um);
    return suffix;
}

tmv::checkpoint::wire::Encoder variable_diffusion_v42_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    c = source.runtime.diffusion.config;
    suffix.string(c.coefficient_source);
    suffix.floating(c.K_v_surface);
    suffix.floating(c.K_v_top);
    return suffix;
}

tmv::checkpoint::wire::Encoder radar_products_v43_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    selection = source.runtime.radar.product_selection;
    suffix.boolean(selection.reflectivity);
    suffix.boolean(selection.radial_velocity);
    suffix.boolean(selection.zdr);
    suffix.string(selection.reflectivity_tier);
    suffix.string(selection.velocity_sampling);
    suffix.string(selection.zdr_tier);
    suffix.floating(selection.radar_x);
    suffix.floating(selection.radar_y);
    suffix.floating(selection.radar_z);
    suffix.boolean(selection.apply_scatterer_correction);
    return suffix;
}

tmv::checkpoint::wire::Encoder radar_products_v44_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    selection = source.runtime.radar.product_selection;
    suffix.boolean(selection.reflectivity);
    suffix.boolean(selection.radial_velocity);
    suffix.boolean(selection.zdr);
    suffix.boolean(selection.kdp);
    suffix.boolean(selection.rhohv);
    suffix.string(selection.reflectivity_tier);
    suffix.string(selection.velocity_sampling);
    suffix.string(selection.zdr_tier);
    suffix.floating(selection.radar_x);
    suffix.floating(selection.radar_y);
    suffix.floating(selection.radar_z);
    suffix.boolean(selection.apply_scatterer_correction);
    return suffix;
}

tmv::checkpoint::wire::Encoder simple_grey_v45_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    radiation = source.runtime.radiation;
    const auto&                    config    = radiation.config;
    suffix.unsigned_value<std::uint64_t>(config.subcycle_count);
    suffix.string(config.simple_grey_optical_model);
    suffix.floating(config.simple_grey_lw_vapor_absorption_m2kg);
    suffix.floating(config.simple_grey_lw_liquid_absorption_m2kg);
    suffix.floating(config.simple_grey_lw_ice_absorption_m2kg);
    suffix.floating(config.simple_grey_sw_vapor_absorption_m2kg);
    suffix.floating(config.simple_grey_sw_liquid_absorption_m2kg);
    suffix.floating(config.simple_grey_sw_ice_absorption_m2kg);
    const auto& fluxes = radiation.held_flux_diagnostics;
    suffix.unsigned_value<std::uint64_t>(fluxes.columns_x);
    suffix.unsigned_value<std::uint64_t>(fluxes.columns_y);
    suffix.unsigned_value<std::uint64_t>(fluxes.levels);
    const auto encode_vector = [&suffix](const std::vector<double>& values)
    {
        suffix.unsigned_value<std::uint64_t>(values.size());
        for (double value : values)
            suffix.floating(value);
    };
    encode_vector(fluxes.upward_longwave_wm2);
    encode_vector(fluxes.downward_longwave_wm2);
    encode_vector(fluxes.upward_shortwave_wm2);
    encode_vector(fluxes.downward_shortwave_wm2);
    return suffix;
}

tmv::checkpoint::wire::Encoder terrain_placement_v49_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    const auto&                    config    = source.runtime.terrain.config;
    const auto&                    placement = source.runtime.terrain.earth_fixed_placement;
    suffix.string(config.horizontal_placement_id);
    suffix.floating(config.feature_center_earth_x_m);
    suffix.floating(config.feature_center_earth_y_m);
    suffix.boolean(placement.active);
    suffix.integer(placement.nr);
    suffix.integer(placement.nth);
    suffix.integer(placement.nz);
    suffix.floating(placement.dx_m);
    suffix.floating(placement.dy_m);
    suffix.floating(placement.grid_origin_earth_m.x_m);
    suffix.floating(placement.grid_origin_earth_m.y_m);
    suffix.floating(placement.feature_center_earth_m.x_m);
    suffix.floating(placement.feature_center_earth_m.y_m);
    suffix.unsigned_value(placement.generation);
    return suffix;
}

tmv::checkpoint::wire::Encoder unsplit_rk_v50_suffix(const SimulationStorageState& source)
{
    tmv::checkpoint::wire::Encoder suffix;
    suffix.floating(source.runtime.time_stepping.last_completed_step_energy.unsplit_rk_committed_change_j);
    suffix.floating(source.runtime.time_stepping.cumulative_completed_step_energy.unsplit_rk_committed_change_j);
    return suffix;
}

void strip_private_inventory_v55_suffix(tmv::checkpoint::wire::Section& runtime)
{
    REQUIRE(runtime.version == 55);
    REQUIRE_FALSE(runtime.payload.empty());
    REQUIRE(runtime.payload.back() == 0);
    runtime.payload.pop_back();
    runtime.version = 54;
}

void strip_coordinate_remap_v54_suffix(tmv::checkpoint::wire::Section& runtime)
{
    if (runtime.version == 55)
        strip_private_inventory_v55_suffix(runtime);
    REQUIRE(runtime.version == 54);
    REQUIRE(runtime.payload.size() >= 16);
    runtime.payload.resize(runtime.payload.size()-16);
    runtime.version=53;
}

void strip_native_chaos_v52_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 54) strip_coordinate_remap_v54_suffix(runtime);
    if (runtime.version == 53)
    {
        REQUIRE_FALSE(runtime.payload.empty());
        REQUIRE(runtime.payload.back() == static_cast<std::uint8_t>(source.runtime.terrain.earth_fixed_placement.sampling));
        runtime.payload.pop_back();
        runtime.version = 52;
    }

    REQUIRE(runtime.version == 52);
    REQUIRE_FALSE(runtime.payload.empty());
    REQUIRE(runtime.payload.back() == (source.runtime.chaos.config.native_generation_required ? 1 : 0));
    runtime.payload.pop_back();
    runtime.version = 51;
}

void strip_execution_v51_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 52) strip_native_chaos_v52_suffix(runtime, source);
    REQUIRE(runtime.version == 51);
    tmv::checkpoint::wire::Encoder suffix;
    suffix.string(source.runtime.compute.vertical_flux_template_id);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 50;
}

void strip_unsplit_rk_v50_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 51) strip_execution_v51_suffix(runtime, source);
    REQUIRE(runtime.version == 50);
    const auto suffix = unsplit_rk_v50_suffix(source);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 49;
}

void strip_terrain_placement_v49_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 50)
        strip_unsplit_rk_v50_suffix(runtime, source);
    REQUIRE(runtime.version == 49);
    const auto suffix = terrain_placement_v49_suffix(source);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 48;
}

void strip_simple_grey_v45_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 49)
        strip_terrain_placement_v49_suffix(runtime, source);
    if (runtime.version == 48)
        runtime.version = 47;
    if (runtime.version == 47)
    {
        tmv::checkpoint::wire::Encoder suffix;
        const auto&                    thermo = source.runtime.microphysics.last_thermodynamic_repair_report;
        suffix.floating(thermo.boundary_layer_cloud_liquid_tendency_kg);
        suffix.floating(thermo.boundary_layer_cloud_ice_tendency_kg);
        REQUIRE(runtime.payload.size() >= suffix.bytes.size());
        const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
        REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
        runtime.payload.erase(start, runtime.payload.end());
        runtime.version = 46;
    }
    if (runtime.version == 46)
    {
        tmv::checkpoint::wire::Encoder suffix;
        suffix.floating(source.runtime.turbulence.config.user_filter_width_m);
        REQUIRE(runtime.payload.size() >= suffix.bytes.size());
        const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
        REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
        runtime.payload.erase(start, runtime.payload.end());
        runtime.version = 45;
    }
    REQUIRE(runtime.version == 45);
    const auto suffix = simple_grey_v45_suffix(source);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 44;
}

void downgrade_radar_products_v44(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 45)
        strip_simple_grey_v45_suffix(runtime, source);
    REQUIRE(runtime.version == 44);
    const auto current = radar_products_v44_suffix(source);
    const auto legacy  = radar_products_v43_suffix(source);
    REQUIRE(runtime.payload.size() >= current.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(current.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), current.bytes.begin(), current.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.payload.insert(runtime.payload.end(), legacy.bytes.begin(), legacy.bytes.end());
    runtime.version = 43;
}

void strip_radar_products_v43_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 45)
        strip_simple_grey_v45_suffix(runtime, source);
    if (runtime.version == 44)
        downgrade_radar_products_v44(runtime, source);
    REQUIRE(runtime.version == 43);
    const auto suffix = radar_products_v43_suffix(source);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 42;
}

void strip_variable_diffusion_v42_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 45)
        strip_simple_grey_v45_suffix(runtime, source);
    if (runtime.version == 44 || runtime.version == 43)
        strip_radar_products_v43_suffix(runtime, source);
    REQUIRE(runtime.version == 42);
    const auto suffix = variable_diffusion_v42_suffix(source);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 41;
}

void strip_rrtmg_v41_suffix(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 45)
        strip_simple_grey_v45_suffix(runtime, source);
    if (runtime.version == 44 || runtime.version == 43 || runtime.version == 42)
        strip_variable_diffusion_v42_suffix(runtime, source);
    REQUIRE(runtime.version == 41);
    const auto suffix = rrtmg_v41_suffix(source);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 40;
}

void replace_rrtmg_v41_suffix(tmv::checkpoint::wire::Section& runtime,
                              const SimulationStorageState&   encoded_source,
                              const SimulationStorageState&   replacement)
{
    if (runtime.version >= 45)
        strip_simple_grey_v45_suffix(runtime, encoded_source);
    if (runtime.version == 44 || runtime.version == 43 || runtime.version == 42)
        strip_variable_diffusion_v42_suffix(runtime, encoded_source);
    REQUIRE(runtime.version == 41);
    const auto encoded_suffix     = rrtmg_v41_suffix(encoded_source);
    const auto replacement_suffix = rrtmg_v41_suffix(replacement);
    REQUIRE(encoded_suffix.bytes.size() == replacement_suffix.bytes.size());
    REQUIRE(runtime.payload.size() >= encoded_suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(encoded_suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), encoded_suffix.bytes.begin(), encoded_suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.payload.insert(runtime.payload.end(), replacement_suffix.bytes.begin(), replacement_suffix.bytes.end());
}

void replace_tracer_roundoff_v40_suffix(tmv::checkpoint::wire::Section& runtime,
                                        const SimulationStorageState&   encoded_source,
                                        const SimulationStorageState&   replacement)
{
    if (runtime.version >= 45)
        strip_simple_grey_v45_suffix(runtime, encoded_source);
    if (runtime.version == 44 || runtime.version == 43 || runtime.version == 42 || runtime.version == 41)
        strip_rrtmg_v41_suffix(runtime, encoded_source);
    REQUIRE(runtime.version == 40);
    const auto encoded_suffix     = tracer_roundoff_v40_suffix(encoded_source);
    const auto replacement_suffix = tracer_roundoff_v40_suffix(replacement);
    REQUIRE(encoded_suffix.bytes.size() == replacement_suffix.bytes.size());
    REQUIRE(runtime.payload.size() >= encoded_suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(encoded_suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), encoded_suffix.bytes.begin(), encoded_suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.payload.insert(runtime.payload.end(), replacement_suffix.bytes.begin(), replacement_suffix.bytes.end());
}

void strip_chaos_pbl_cache_v37(tmv::checkpoint::wire::Section& runtime, const SimulationStorageState& source)
{
    if (runtime.version >= 45)
        strip_simple_grey_v45_suffix(runtime, source);
    if (runtime.version == 44 || runtime.version == 43 || runtime.version == 42 || runtime.version == 41)
        strip_rrtmg_v41_suffix(runtime, source);
    if (runtime.version == 40)
    {
        const auto suffix = tracer_roundoff_v40_suffix(source);
        REQUIRE(runtime.payload.size() >= suffix.bytes.size());
        const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
        REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
        runtime.payload.erase(start, runtime.payload.end());
        runtime.version = 39;
    }
    if (runtime.version == 39)
    {
        const auto suffix = terrain_shortwave_v39_suffix(source);
        REQUIRE(runtime.payload.size() >= suffix.bytes.size());
        const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
        REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
        runtime.payload.erase(start, runtime.payload.end());
        runtime.version = 38;
    }
    if (runtime.version == 38)
    {
        const auto suffix = open_stress_v38_suffix(source);
        REQUIRE(runtime.payload.size() >= suffix.bytes.size());
        const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
        REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
        runtime.payload.erase(start, runtime.payload.end());
        runtime.version = 37;
    }
    REQUIRE(runtime.version == 37);
    const auto suffix = chaos_pbl_cache_v37_suffix(source);
    REQUIRE(runtime.payload.size() >= suffix.bytes.size());
    const auto start = runtime.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime.payload.erase(start, runtime.payload.end());
    runtime.version = 36;
}

template <std::size_t Capacity>
void set_provenance_text(tmv::domain::DomainRecyclingProvenanceText<Capacity>& destination, const std::string& text)
{
    REQUIRE(text.size() <= Capacity);
    destination = {};
    std::copy(text.begin(), text.end(), destination.bytes.begin());
    destination.size = static_cast<std::uint16_t>(text.size());
}

void seed_committed_domain_recycling_provenance(SimulationStorageState& state)
{
    const auto plan = tmv::domain::plan_whole_cell_shift(3, 2, 2, 1, 0);
    auto&      p    = state.runtime.domain_recycling_provenance;
    p.available     = true;
    p.sequence      = 7;
    set_provenance_text(p.transaction_id, "recycle_000007");
    set_provenance_text(p.publication_generation, "run_generation_4");
    set_provenance_text(p.storage_format, "npy_3d");
    p.event_manifest_bytes                    = 4096;
    p.event_manifest_fnv1a64                  = UINT64_C(0x123456789abcdef0);
    p.previous_chain_fnv1a64                  = UINT64_C(0x0f0e0d0c0b0a0908);
    p.simulation_time_s                       = 17.0;
    p.pre_shift_origin_t0_m                   = {124875.0, -48000.0};
    p.post_shift_origin_t0_m                  = {125000.0, -48000.0};
    p.frame_velocity_ms                       = state.configuration.domain_frame.velocity_ms;
    p.dx_m                                    = 125.0;
    p.dy_m                                    = 140.0;
    p.nr                                      = plan.nr;
    p.nth                                     = plan.nth;
    p.nz                                      = plan.nz;
    p.shift_i_cells                           = plan.shift_i_cells;
    p.shift_j_cells                           = plan.shift_j_cells;
    p.physical_i_begin                        = plan.physical_i_begin;
    p.physical_i_end                          = plan.physical_i_end;
    p.physical_j_begin                        = plan.physical_j_begin;
    p.physical_j_end                          = plan.physical_j_end;
    p.source_i_begin                          = plan.source_i_begin;
    p.source_i_end                            = plan.source_i_end;
    p.source_j_begin                          = plan.source_j_begin;
    p.source_j_end                            = plan.source_j_end;
    p.physical_columns                        = plan.trailing_cells / 2;
    p.physical_cells                          = plan.trailing_cells;
    p.control_volume_exchange_complete        = true;
    p.admitted_inventories.total_mass         = 10.0;
    p.admitted_inventories.dry_mass           = 10.0;
    p.admitted_inventories.theta_volume       = 3000.0;
    p.admitted_inventories.moist_total_mass   = 10.0;
    p.admitted_momentum.total_mass            = 10.0;
    p.admitted_momentum.reference_linear      = 10.0;
    p.admitted_momentum.reference_angular     = 1250.0;
    p.retiring_fixed_earth_angular_momentum_z = 0.0;
    p.admitted_fixed_earth_angular_momentum_z = 0.0;
    set_provenance_text(p.shift_algorithm, "stagger_aware_in_place_homogeneous_whole_cell_shift_v2");
    set_provenance_text(p.frame_algorithm, "integer_whole_cell_earth_origin_commit_v1");
    set_provenance_text(p.swath_algorithm, "canonical_rectangular_earth_swath_capture_v1");
    set_provenance_text(p.scalar_ledger_basis, "production_conservation_budget_retiring_cells_v1");
    set_provenance_text(p.momentum_ledger_basis, "production_scalar_centered_momentum_retiring_cells_v1");
    set_provenance_text(p.energy_ledger_basis, "physical_moist_mixture_v2_retiring_cells_v1");
    set_provenance_text(p.theta_residual_ledger_basis, "deferred_theta_storage_residual_energy_retiring_cells_v1");
    set_provenance_text(p.control_volume_exchange_basis, "instantaneous_fixed_earth_control_volume_exchange_v1");
    p.committed_chain_fnv1a64 =
        tmv::domain::compute_domain_recycling_transaction_chain_fnv1a64(p.previous_chain_fnv1a64,
                                                                        p.sequence,
                                                                        p.event_manifest_bytes,
                                                                        p.event_manifest_fnv1a64,
                                                                        p.transaction_id.view(),
                                                                        p.publication_generation.view());
    REQUIRE(tmv::domain::valid_domain_recycling_transaction_provenance(p));
}

void downgrade_configuration_v16_to_v15(tmv::checkpoint::wire::Section& configuration,
                                        const SimulationStorageState&   source)
{
    REQUIRE(configuration.version == 16);
    tmv::checkpoint::wire::Encoder suffix;
    suffix.floating(source.configuration.bubble_cloud_water_kgkg);
    REQUIRE(configuration.payload.size() >= suffix.bytes.size());
    const auto begin = configuration.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(begin, configuration.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    configuration.payload.erase(begin, configuration.payload.end());
    configuration.version = 15;
}

void downgrade_configuration_v16_to_v14(tmv::checkpoint::wire::Section& configuration,
                                        const SimulationStorageState&   source)
{
    downgrade_configuration_v16_to_v15(configuration, source);
    tmv::checkpoint::wire::Encoder suffix;
    suffix.floating(source.configuration.microphysics_config.thompson.water_friendly_aerosol_surface_m3);
    suffix.floating(source.configuration.microphysics_config.thompson.ice_friendly_aerosol_surface_m3);
    REQUIRE(configuration.payload.size() >= suffix.bytes.size());
    const auto begin = configuration.payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(begin, configuration.payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    configuration.payload.erase(begin, configuration.payload.end());
    configuration.version = 14;
}

} // namespace

TEST_CASE("direct wrapped sections preserve the established checkpoint bytes", "[core][checkpoint][wire][memory]")
{
    // The direct path is an allocation change only; the durable bytes remain
    // exactly compatible with the established two-stage encoder.
    constexpr std::array<std::uint8_t, 8>       magic    = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    std::vector<tmv::checkpoint::wire::Section> sections = {
        {1, 3, tmv::checkpoint::wire::section_required, {1, 2, 3, 4}},
        {7, 2, tmv::checkpoint::wire::section_required, {9, 8, 7}},
    };
    std::vector<std::uint8_t> payload;
    std::vector<std::uint8_t> established;
    std::vector<std::uint8_t> direct;
    std::string               error;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, established);
    REQUIRE(tmv::checkpoint::wire::wrap_sections(
        magic, tmv::checkpoint::simulation_checkpoint_version, sections, direct, error));
    REQUIRE(direct == established);
    std::uint64_t planned_file_size = 0;
    REQUIRE(tmv::checkpoint::wire::wrapped_sections_file_size<magic.size()>(sections, planned_file_size, error));
    REQUIRE(planned_file_size == established.size());

    const auto streamed_path = std::filesystem::temp_directory_path() / "nimbus_checkpoint_streamed_sections.bin";
    std::filesystem::remove(streamed_path);
    tmv::checkpoint::wire::PublishedFileIntegrity integrity;
    REQUIRE(tmv::checkpoint::wire::write_sections_atomic(
        streamed_path, magic, tmv::checkpoint::simulation_checkpoint_version, sections, error, "test", &integrity));
    std::vector<std::uint8_t>                     streamed;
    tmv::checkpoint::wire::PublishedFileIntegrity read_integrity;
    REQUIRE(tmv::checkpoint::wire::read_file(streamed_path, streamed, error, "test", &read_integrity));
    REQUIRE(streamed == established);
    std::uint64_t expected_fnv = 14695981039346656037ULL;
    for (const std::uint8_t byte : streamed)
    {
        expected_fnv ^= byte;
        expected_fnv *= 1099511628211ULL;
    }
    nimbus::crypto::Sha256 expected_sha;
    expected_sha.update(streamed.data(), streamed.size());
    CHECK(integrity.bytes == streamed.size());
    CHECK(integrity.fnv1a64 == expected_fnv);
    CHECK(integrity.sha256 == expected_sha.hex_digest());
    CHECK(read_integrity.bytes == integrity.bytes);
    CHECK(read_integrity.fnv1a64 == integrity.fnv1a64);
    CHECK(read_integrity.sha256 == integrity.sha256);
    std::filesystem::remove(streamed_path);

    const auto rejected_path =
        std::filesystem::temp_directory_path() / "nimbus_checkpoint_rejected_for_disk_headroom.bin";
    std::filesystem::remove(rejected_path);
    REQUIRE_FALSE(tmv::checkpoint::wire::write_sections_atomic(rejected_path,
                                                               magic,
                                                               tmv::checkpoint::simulation_checkpoint_version,
                                                               sections,
                                                               error,
                                                               "test",
                                                               nullptr,
                                                               std::numeric_limits<std::uint64_t>::max()));
    REQUIRE(error == "test checkpoint disk requirement exceeds the representable limit");
    REQUIRE_FALSE(std::filesystem::exists(rejected_path));
    REQUIRE_FALSE(std::filesystem::exists(rejected_path.string() + ".tmp"));

    sections.push_back(sections.front());
    const auto prior = direct;
    REQUIRE_FALSE(tmv::checkpoint::wire::wrap_sections(
        magic, tmv::checkpoint::simulation_checkpoint_version, sections, direct, error));
    REQUIRE(error == "invalid or duplicate checkpoint section id");
    REQUIRE(direct == prior);
}

TEST_CASE("simulation checkpoint is deterministic and restores an independent run", "[core][checkpoint][mod03]")
{
    SimulationStorageState    source = sample_state();
    std::vector<std::uint8_t> first, second;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, first, error));
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, second, error));
    REQUIRE(first == second);

    const auto streamed_path = std::filesystem::temp_directory_path() / "nimbus_simulation_checkpoint_streamed.bin";
    std::filesystem::remove(streamed_path);
    tmv::checkpoint::SimulationCheckpointIntegrity streamed_integrity;
    REQUIRE(tmv::checkpoint::write_simulation_checkpoint(streamed_path, source, nullptr, streamed_integrity, error));
    std::ifstream                   streamed_file(streamed_path, std::ios::binary);
    const std::vector<std::uint8_t> streamed_bytes{std::istreambuf_iterator<char>(streamed_file),
                                                   std::istreambuf_iterator<char>()};
    REQUIRE(streamed_bytes == first);
    REQUIRE(streamed_integrity.bytes == first.size());
    std::filesystem::remove(streamed_path);

    const auto rejected_path =
        std::filesystem::temp_directory_path() / "nimbus_simulation_checkpoint_disk_admission_rejected.bin";
    std::filesystem::remove(rejected_path);
    tmv::checkpoint::SimulationCheckpointIntegrity rejected_integrity;
    REQUIRE_FALSE(tmv::checkpoint::write_simulation_checkpoint(
        rejected_path, source, nullptr, rejected_integrity, error, std::numeric_limits<std::uint64_t>::max()));
    REQUIRE(error == "simulation checkpoint disk requirement exceeds the representable limit");
    REQUIRE_FALSE(std::filesystem::exists(rejected_path));
    REQUIRE_FALSE(std::filesystem::exists(rejected_path.string() + ".tmp"));

    SimulationStorageState restored;
    const bool             restored_ok = tmv::checkpoint::deserialize_simulation_storage(first, restored, error);
    INFO(error);
    REQUIRE(restored_ok);
    CHECK(restored.runtime.boundary_conditions.radiative_u_west.empty());
    CHECK(restored.runtime.boundary_conditions.radiative_u_east.empty());
    CHECK(restored.runtime.boundary_conditions.radiative_v_south.empty());
    CHECK(restored.runtime.boundary_conditions.radiative_v_north.empty());
    CHECK(restored.time_s == Approx(17.5));
    CHECK(restored.grid.nr == 3);
    CHECK(restored.grid.geometry.staggered);
    CHECK(restored.fields.theta[1][1][1] == Approx(source.fields.theta[1][1][1]));
    CHECK(restored.base.p0 == source.base.p0);
    CHECK(restored.surface.ustar == source.surface.ustar);
    CHECK(restored.surface.precipitation_rate_mm_h == source.surface.precipitation_rate_mm_h);
    CHECK(restored.surface.accumulated_rainfall_mm == source.surface.accumulated_rainfall_mm);
    CHECK(restored.surface.accumulated_rainfall_residual_mm == source.surface.accumulated_rainfall_residual_mm);
    CHECK(restored.surface.accumulated_rainfall_last_update_s == Approx(17.0));
    CHECK(restored.nested.qi[1][1][1] == Approx(source.nested.qi[1][1][1]));
    CHECK(restored.nested.qs[1][1][1] == Approx(source.nested.qs[1][1][1]));
    CHECK(restored.configuration.validation_report_path == "reports/validation.json");
    CHECK(restored.configuration.configured_dt_large_s == Approx(0.25));
    CHECK(restored.configuration.bubble_cloud_water_kgkg == Approx(0.0015));
    CHECK(restored.configuration.storm_tracking.enabled);
    CHECK(restored.configuration.storm_tracking.recycle_domain);
    CHECK(restored.configuration.storm_tracking.interval_s == Approx(45.0));
    CHECK(restored.configuration.storm_tracking.inner_margin_m == Approx(18000.0));
    CHECK(restored.configuration.storm_tracking.maximum_shift_cells == 7);
    CHECK(restored.configuration.storm_tracking.minimum_confidence == Approx(0.85));
    CHECK(restored.configuration.storm_tracking.minimum_updraft_ms == Approx(6.5));
    CHECK(restored.configuration.storm_tracking.minimum_condensate_path_kgm2 == Approx(0.2));
    CHECK(restored.configuration.storm_tracking.updraft_helicity_bottom_m_agl == Approx(1500.0));
    CHECK(restored.configuration.storm_tracking.updraft_helicity_top_m_agl == Approx(4500.0));
    CHECK(restored.configuration.storm_tracking.minimum_absolute_updraft_helicity_m2s2 == Approx(40.0));
    CHECK(restored.configuration.storm_tracking.minimum_component_cells == 12);
    CHECK(restored.configuration.storm_tracking.minimum_core_area_m2 == Approx(3.5e6));
    CHECK(restored.configuration.storm_tracking.maximum_motion_speed_ms == Approx(68.0));
    CHECK(restored.configuration.storm_tracking.maximum_motion_acceleration_ms2 == Approx(0.18));
    CHECK(restored.configuration.storm_tracking.minimum_continuity_samples == 4);
    CHECK(restored.configuration.storm_tracking.lost_after_samples == 5);
    CHECK(restored.configuration.storm_tracking.topology_confirmation_samples == 3);
    CHECK(restored.configuration.storm_tracking.branch_preference ==
          tmv::init::config::StormBranchPreference::BunkersLeft);
    CHECK(restored.configuration.storm_tracking.branch_velocity_uncertainty_ms == Approx(4.5));
    CHECK(restored.configuration.domain_motion_mode == tmv::init::config::DomainMotionMode::ConstantVelocity);
    CHECK(restored.configuration.domain_frame.earth_origin_at_t0_m.x_m == Approx(125000.0));
    CHECK(restored.configuration.domain_frame.earth_origin_at_t0_m.y_m == Approx(-48000.0));
    CHECK(restored.configuration.domain_frame.velocity_ms.u_ms == Approx(18.5));
    CHECK(restored.configuration.domain_frame.velocity_ms.v_ms == Approx(-7.25));
    CHECK(restored.runtime.storm_tracking.status == tmv::diagnostics::StormContinuityStatus::Tracking);
    CHECK(restored.runtime.storm_tracking.reason == tmv::diagnostics::StormContinuityReason::None);
    CHECK(restored.runtime.storm_tracking.sample_count == 4);
    CHECK(restored.runtime.storm_tracking.samples[3].time_s == Approx(40.0));
    CHECK(restored.runtime.storm_tracking.samples[3].x_m == Approx(1300.0));
    CHECK(restored.runtime.storm_tracking.samples[3].y_m == Approx(1850.0));
    CHECK(restored.runtime.storm_tracking.branch_id == 4);
    CHECK(restored.runtime.storm_tracking.motion_available);
    CHECK(restored.runtime.storm_tracking.velocity_x_ms == Approx(10.0));
    CHECK(restored.runtime.storm_tracking.velocity_y_ms == Approx(-5.0));
    CHECK(restored.runtime.storm_tracking.velocity_uncertainty_ms == Approx(0.2));
    CHECK(restored.runtime.storm_tracking.topology == tmv::diagnostics::StormBranchTopology::Multiple);
    CHECK(restored.runtime.storm_tracking.pending_topology == tmv::diagnostics::StormBranchTopology::Single);
    CHECK(restored.runtime.storm_tracking.topology_event ==
          tmv::diagnostics::StormBranchTopologyEvent::MergerCandidate);
    CHECK(restored.runtime.storm_tracking.pending_topology_samples == 1);
    CHECK(restored.runtime.storm_tracking.topology_event_id == 8);
    CHECK(restored.runtime.storm_tracking.selected_candidate_available);
    CHECK(restored.runtime.storm_tracking.selected_candidate_rank == 1);
    CHECK(restored.runtime.storm_tracking.selected_component_label == 42);
    CHECK(restored.runtime.storm_tracking.selected_motion_class ==
          tmv::diagnostics::StormBranchMotionClass::BunkersRight);
    CHECK(restored.runtime.storm_tracking.selected_bunkers_right_mismatch_ms == Approx(1.25));
    CHECK(restored.runtime.storm_tracking.selected_bunkers_left_mismatch_ms == Approx(17.5));
    CHECK(restored.runtime.storm_tracking.association_envelope_available);
    CHECK(restored.runtime.storm_tracking.association_envelope_west_m == Approx(900.0));
    CHECK(restored.runtime.storm_tracking.association_envelope_east_m == Approx(1800.0));
    CHECK(restored.runtime.storm_tracking.association_envelope_south_m == Approx(1500.0));
    CHECK(restored.runtime.storm_tracking.association_envelope_north_m == Approx(2400.0));
    CHECK_FALSE(restored.runtime.storm_tracking.topology_reorganization);
    CHECK(restored.runtime.storm_tracking.reorganization_overlap_fraction == Approx(0.0));
    CHECK(restored.runtime.storm_window_movement.armed);
    CHECK(restored.runtime.storm_window_movement.pending_i_direction == 1);
    CHECK(restored.runtime.storm_window_movement.pending_j_direction == -1);
    CHECK(restored.runtime.storm_window_movement.consecutive_compatible_samples == 1);
    CHECK(restored.runtime.storm_window_movement.sample_time_available);
    CHECK(restored.runtime.storm_window_movement.last_sample_time_s == Approx(40.0));
    CHECK(restored.runtime.storm_window_movement.committed_move_available);
    CHECK(restored.runtime.storm_window_movement.last_committed_time_s == Approx(10.0));
    CHECK(restored.runtime.storm_window_movement.cumulative_i_shift_cells == 12);
    CHECK(restored.runtime.storm_window_movement.cumulative_j_shift_cells == -5);
    CHECK(restored.runtime.storm_selected_partition_available);
    CHECK(restored.runtime.storm_selected_partition_nx == 2);
    CHECK(restored.runtime.storm_selected_partition_ny == 2);
    CHECK(restored.runtime.storm_selected_partition == std::vector<std::uint8_t>{1U, 1U, 0U, 0U});
    CHECK(restored.configuration.microphysics_config.shared.autoconversion_rate_s_inv == Approx(4.5e-4));
    CHECK(restored.configuration.execution_mode == tmv::init::config::ExecutionMode::Qualified);
    CHECK(restored.configuration.microphysics_config.kessler.evaporation_efficiency == Approx(0.65));
    CHECK(restored.configuration.microphysics_config.thompson.evaporation_rate_s_inv == Approx(7.0e-4));
    CHECK(restored.configuration.microphysics_config.thompson.water_friendly_aerosol_surface_m3 == Approx(325.0e6));
    CHECK(restored.configuration.microphysics_config.thompson.ice_friendly_aerosol_surface_m3 == Approx(1.75e6));
    CHECK(restored.configuration.microphysics_config.milbrandt.evaporation_rate_s_inv == Approx(6.0e-4));
    CHECK_FALSE(restored.configuration.microphysics_config.milbrandt.triple_moment);
    CHECK(restored.configuration.validation_policy.field_overrides.at("theta").max_value == Approx(380.0));
    REQUIRE(restored.runtime.radiation.scheme);
    CHECK(restored.runtime.radiation.scheme->name() == "simple_grey");
    CHECK(restored.runtime.radiation.config.latitude_deg == Approx(-12.5));
    CHECK(restored.runtime.radiation.config.declination_deg == Approx(-20.0));
    CHECK(restored.runtime.radiation.config.local_solar_time_at_start_h == Approx(7.25));
    CHECK_FALSE(restored.runtime.radiation.config.clear_sky_only);
    CHECK(restored.runtime.radiation.config.use_subcycling);
    CHECK(restored.runtime.radiation.config.subcycle_count == 3);
    CHECK(restored.runtime.radiation.config.enable_diagnostics);
    CHECK(restored.runtime.radiation.config.simple_grey_optical_model == "absorbing_water_cloud_path_v1");
    CHECK(restored.runtime.radiation.config.simple_grey_lw_vapor_absorption_m2kg == Approx(0.03));
    CHECK(restored.runtime.radiation.config.simple_grey_lw_liquid_absorption_m2kg == Approx(82.0));
    CHECK(restored.runtime.radiation.config.simple_grey_lw_ice_absorption_m2kg == Approx(43.0));
    CHECK(restored.runtime.radiation.config.simple_grey_sw_vapor_absorption_m2kg == Approx(0.002));
    CHECK(restored.runtime.radiation.config.simple_grey_sw_liquid_absorption_m2kg == Approx(18.0));
    CHECK(restored.runtime.radiation.config.simple_grey_sw_ice_absorption_m2kg == Approx(11.0));
    CHECK(restored.runtime.radiation.config.rrtmg_ozone_vmr == Approx(4.0e-6));
    CHECK(restored.runtime.radiation.config.rrtmg_co2_vmr == Approx(421.0e-6));
    CHECK(restored.runtime.radiation.config.rrtmg_ch4_vmr == Approx(1.91e-6));
    CHECK(restored.runtime.radiation.config.rrtmg_n2o_vmr == Approx(0.336e-6));
    CHECK(restored.runtime.radiation.config.rrtmg_co_vmr == Approx(0.1e-6));
    CHECK(restored.runtime.radiation.config.rrtmg_o2_vmr == Approx(0.2094));
    CHECK(restored.runtime.radiation.config.rrtmg_n2_vmr == Approx(0.7807));
    CHECK(restored.runtime.radiation.config.rrtmg_cfc11_vmr == Approx(2.0e-10));
    CHECK(restored.runtime.radiation.config.rrtmg_cfc12_vmr == Approx(5.0e-10));
    CHECK(restored.runtime.radiation.config.rrtmg_cfc22_vmr == Approx(3.0e-10));
    CHECK(restored.runtime.radiation.config.rrtmg_ccl4_vmr == Approx(1.0e-10));
    CHECK(restored.runtime.radiation.config.rrtmg_liquid_effective_radius_um == Approx(11.0));
    CHECK(restored.runtime.radiation.config.rrtmg_ice_effective_radius_um == Approx(35.0));
    CHECK(restored.runtime.radiation.held_atmospheric_power_w == Approx(-1250.0));
    CHECK(restored.runtime.radiation.held_cell_power_w[1][1][1] == Approx(-125.0f));
    CHECK(restored.runtime.radiation.next_update_time_s == Approx(300.0));
    CHECK(restored.runtime.radiation.last_expected_atmospheric_energy_j == Approx(-625.0));
    CHECK(restored.runtime.radiation.last_applied_radiation_heat_j == Approx(-624.9999));
    CHECK(restored.runtime.radiation.accumulated_expected_atmospheric_energy_j == Approx(-8125.0));
    CHECK(restored.runtime.radiation.accumulated_applied_radiation_heat_j == Approx(-8124.999));
    CHECK(restored.runtime.radiation.applied_step_count == 13);
    CHECK(restored.runtime.radiation.held_flux_diagnostics.columns_x == 3);
    CHECK(restored.runtime.radiation.held_flux_diagnostics.columns_y == 2);
    CHECK(restored.runtime.radiation.held_flux_diagnostics.levels == 2);
    CHECK(restored.runtime.radiation.held_flux_diagnostics.upward_longwave_wm2 == std::vector<double>(18, 210.0));
    CHECK(restored.runtime.radiation.held_flux_diagnostics.downward_longwave_wm2 == std::vector<double>(18, 110.0));
    CHECK(restored.runtime.radiation.held_flux_diagnostics.upward_shortwave_wm2 == std::vector<double>(18, 45.0));
    CHECK(restored.runtime.radiation.held_flux_diagnostics.downward_shortwave_wm2 == std::vector<double>(18, 520.0));
    CHECK(restored.runtime.radiation.config.terrain_sw_cosine_nodes == 12);
    CHECK(restored.runtime.radiation.config.terrain_sw_azimuth_nodes == 24);
    CHECK(restored.runtime.radiation.config.terrain_sw_maximum_bounces == 48);
    CHECK(restored.runtime.radiation.config.terrain_sw_relative_residual_tolerance == Approx(2.0e-11));
    CHECK(restored.runtime.radiation.config.terrain_sw_absolute_residual_tolerance_w == Approx(3.0e-8));
    CHECK(restored.runtime.radiation.held_shortwave_incoming_power_w == Approx(1200.0));
    CHECK(restored.runtime.radiation.held_shortwave_atmospheric_absorbed_power_w == Approx(200.0));
    CHECK(restored.runtime.radiation.held_surface_absorbed_power_w == Approx(700.0));
    CHECK(restored.runtime.radiation.held_top_escape_power_w == Approx(250.0));
    CHECK(restored.runtime.radiation.held_lateral_escape_power_w == Approx(50.0));
    CHECK(restored.runtime.radiation.held_unresolved_reflected_power_w == 0.0);
    CHECK(restored.runtime.radiation.held_reflection_converged);
    CHECK(restored.runtime.turbulence.last_update_time_s == Approx(18.0));
    CHECK(restored.runtime.turbulence.next_update_time_s == Approx(20.0));
    CHECK(restored.runtime.turbulence.held_tendencies.dudt_sgs[1][1][1] == Approx(1.25f));
    CHECK(restored.runtime.turbulence.held_tendencies.dtkedt_sgs[1][1][1] == Approx(6.25f));
    CHECK_FALSE(restored.runtime.turbulence.held_tendencies.resident_source_unchanged);
    CHECK(restored.runtime.turbulence.last_momentum_work.requested_linear_work_j == Approx(-19.0));
    CHECK(restored.runtime.turbulence.last_momentum_work.finite_step_correction_j == Approx(0.75));
    CHECK(restored.runtime.turbulence.last_momentum_work.storage_rounding_j == Approx(-0.002));
    CHECK(restored.runtime.turbulence.last_momentum_work.committed_work_j == Approx(-18.252));
    CHECK(restored.runtime.turbulence.last_momentum_work.moist_dry_internal_change_j == Approx(1.0));
    CHECK(restored.runtime.turbulence.last_momentum_work.moist_committed_energy_change_j == Approx(2.748));
    CHECK(restored.runtime.turbulence.last_momentum_work.moist_energy_complete);
    CHECK(restored.runtime.turbulence.cumulative_requested_momentum_work_j == Approx(-190.0));
    CHECK(restored.runtime.turbulence.cumulative_finite_step_momentum_correction_j == Approx(7.5));
    CHECK(restored.runtime.turbulence.cumulative_momentum_storage_rounding_j == Approx(-0.02));
    CHECK(restored.runtime.turbulence.cumulative_committed_momentum_work_j == Approx(-182.52));
    CHECK(restored.runtime.turbulence.cumulative_moist_energy_work.moist_subgrid_tke_change_j == Approx(60.0));
    CHECK(restored.runtime.turbulence.cumulative_moist_energy_work.moist_committed_energy_change_j == Approx(27.48));
    CHECK(restored.runtime.turbulence.cumulative_moist_energy_work.moist_energy_complete);
    CHECK(restored.runtime.turbulence.momentum_work_step_count == 10);
    CHECK(restored.runtime.turbulence.last_explicit_stability_limit_s == Approx(0.375));
    CHECK(restored.runtime.turbulence.last_explicit_substep_count == 4);
    CHECK(restored.runtime.turbulence.last_open_boundary_work_j == Approx(12.0));
    CHECK(restored.runtime.turbulence.last_irreversible_dissipation_j == Approx(15.0));
    CHECK(restored.runtime.turbulence.last_stress_closure_residual_j == Approx(1.0e-12));
    CHECK(restored.runtime.turbulence.cumulative_open_boundary_work_j == Approx(120.0));
    CHECK(restored.runtime.turbulence.cumulative_irreversible_dissipation_j == Approx(150.0));
    CHECK(restored.runtime.turbulence.cumulative_stress_closure_residual_j == Approx(1.0e-11));
    CHECK(restored.runtime.turbulence.open_boundary_stability_step_count == 10);
    CHECK(restored.runtime.diffusion.last_momentum_work.requested_linear_work_j == Approx(-23.0));
    CHECK(restored.runtime.diffusion.last_momentum_work.finite_step_correction_j == Approx(0.5));
    CHECK(restored.runtime.diffusion.last_momentum_work.storage_rounding_j == Approx(-0.004));
    CHECK(restored.runtime.diffusion.last_momentum_work.committed_work_j == Approx(-22.504));
    CHECK(restored.runtime.diffusion.last_momentum_work.moist_dry_internal_change_j == Approx(1.0));
    CHECK(restored.runtime.diffusion.last_momentum_work.moist_energy_complete);
    CHECK(restored.runtime.diffusion.cumulative_requested_momentum_work_j == Approx(-230.0));
    CHECK(restored.runtime.diffusion.cumulative_finite_step_momentum_correction_j == Approx(5.0));
    CHECK(restored.runtime.diffusion.cumulative_momentum_storage_rounding_j == Approx(-0.04));
    CHECK(restored.runtime.diffusion.cumulative_committed_momentum_work_j == Approx(-225.04));
    CHECK(restored.runtime.diffusion.cumulative_moist_energy_work.moist_gravitational_change_j == Approx(50.0));
    CHECK(restored.runtime.diffusion.cumulative_moist_energy_work.moist_energy_complete);
    CHECK(restored.runtime.diffusion.momentum_work_step_count == 10);
    CHECK(restored.runtime.diffusion.moist_energy_step_count == 10);
    CHECK(restored.runtime.boundary_layer.next_update_time_s == Approx(30.0));
    CHECK(restored.runtime.boundary_layer.held_cloud_liquid_mass_rate_kg_s(1, 1, 1) == Approx(5.0f));
    CHECK(restored.runtime.boundary_layer.held_cloud_ice_mass_rate_kg_s(1, 1, 1) == Approx(6.0f));
    CHECK(restored.tendencies.dqc_dt_pbl(1, 1, 1) == Approx(source.tendencies.dqc_dt_pbl(1, 1, 1)));
    CHECK(restored.tendencies.dqi_dt_pbl(1, 1, 1) == Approx(source.tendencies.dqi_dt_pbl(1, 1, 1)));
    REQUIRE(restored.runtime.microphysics.scheme);
    CHECK(restored.runtime.microphysics.scheme->get_scheme_name() == "milbrandt");
    CHECK(restored.runtime.time_stepping.diagnostics.n_steps == 70);
    CHECK(restored.runtime.time_stepping.last_boundary_mass.outward_boundary_mass_kg == Approx(12.5));
    CHECK(restored.runtime.time_stepping.last_boundary_mass.closure_residual_kg == Approx(0.001));
    CHECK(restored.runtime.time_stepping.cumulative_outward_boundary_mass_kg == Approx(125.0));
    CHECK(restored.runtime.time_stepping.cumulative_boundary_mass_closure_residual_kg == Approx(0.01));
    CHECK(restored.runtime.time_stepping.boundary_mass_step_count == 10);
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.continuity_eos_dry_internal_change_j == Approx(-31.0));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.pressure_gradient_storage_rounding_j == Approx(-0.002));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.committed_inventory_change_j == Approx(-3.627));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_committed_work_j == Approx(-0.375));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_radiative_requested_linear_work_j ==
          Approx(-0.5));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_radiative_finite_step_correction_j ==
          Approx(0.125));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_radiative_storage_rounding_j == Approx(-0.0005));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_radiative_committed_work_j == Approx(-0.3755));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_constraint_history_committed_work_j ==
          Approx(0.0005));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_application_count == 8);
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.boundary_complete);
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.continuity_carrier_requested_change_j == Approx(-0.3));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.continuity_carrier_outward_boundary_transport_j ==
          Approx(0.1));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.kinetic_transport_interior_conversion_j == Approx(-21.2));
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.kinetic_transport_complete);
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.substep_count == 7);
    CHECK(restored.runtime.time_stepping.last_acoustic_energy.complete);
    CHECK(restored.runtime.time_stepping.cumulative_acoustic_energy.pressure_gradient_committed_work_j ==
          Approx(277.48));
    CHECK(restored.runtime.time_stepping.cumulative_acoustic_energy.substep_count == 70);
    CHECK(restored.runtime.time_stepping.acoustic_energy_step_count == 10);
    CHECK(restored.runtime.time_stepping.last_slow_momentum_energy.advection_requested_linear_work_j == Approx(-21.0));
    CHECK(restored.runtime.time_stepping.last_slow_momentum_energy.coupled_finite_step_cross_work_j == Approx(-0.125));
    CHECK(restored.runtime.time_stepping.last_slow_momentum_energy.committed_kinetic_change_j == Approx(-16.124));
    CHECK(restored.runtime.time_stepping.last_slow_momentum_energy.pressure_dry_internal_change_j == Approx(2.5));
    CHECK(restored.runtime.time_stepping.last_slow_momentum_energy.committed_pressure_energy_change_j == Approx(2.625));
    CHECK(restored.runtime.time_stepping.last_slow_momentum_energy.complete);
    CHECK(restored.runtime.time_stepping.cumulative_slow_momentum_energy.advection_requested_linear_work_j ==
          Approx(-210.0));
    CHECK(restored.runtime.time_stepping.cumulative_slow_momentum_energy.committed_pressure_energy_change_j ==
          Approx(26.25));
    CHECK(restored.runtime.time_stepping.slow_momentum_energy_step_count == 10);
    CHECK(restored.runtime.time_stepping.last_eos_pressure_energy.dry_internal_change_j == Approx(12.0));
    CHECK(restored.runtime.time_stepping.last_eos_pressure_energy.water_sensible_change_j == Approx(0.5));
    CHECK(restored.runtime.time_stepping.last_eos_pressure_energy.complete);
    CHECK(restored.runtime.time_stepping.cumulative_eos_pressure_energy.committed_energy_change_j == Approx(125.0));
    CHECK(restored.runtime.time_stepping.eos_pressure_energy_step_count == 10);
    CHECK(restored.runtime.time_stepping.last_sponge_energy.upper.momentum_requested_linear_work_j == Approx(-10.0));
    CHECK(restored.runtime.time_stepping.last_sponge_energy.w_only.carrier_kinetic_change_j == Approx(-4.55));
    CHECK(restored.runtime.time_stepping.last_sponge_energy.lateral.committed_energy_change_j == Approx(-14.0));
    CHECK(restored.runtime.time_stepping.last_sponge_energy.complete);
    CHECK(restored.runtime.time_stepping.cumulative_sponge_energy.upper.carrier_kinetic_change_j == Approx(-91.0));
    CHECK(restored.runtime.time_stepping.sponge_energy_step_count == 10);
    CHECK(restored.runtime.time_stepping.last_final_boundary_energy.constraint_committed_work_j == Approx(-3.0));
    CHECK(restored.runtime.time_stepping.last_final_boundary_energy.applied);
    CHECK(restored.runtime.time_stepping.last_final_boundary_energy.complete);
    CHECK(restored.runtime.time_stepping.cumulative_final_boundary_energy.carrier_kinetic_change_j == Approx(-30.0));
    CHECK(restored.runtime.time_stepping.final_boundary_energy_step_count == 10);
    CHECK(restored.runtime.time_stepping.last_completed_step_energy.committed_owner_sum_j == Approx(45.0));
    CHECK(restored.runtime.time_stepping.last_completed_step_energy.closure_residual_j == Approx(1.0));
    CHECK(restored.runtime.time_stepping.last_completed_step_energy.complete);
    CHECK(restored.runtime.time_stepping.cumulative_completed_step_energy.committed_owner_sum_j == Approx(450.0));
    CHECK(restored.runtime.time_stepping.completed_step_energy_count == 10);
    CHECK(restored.runtime.advection.thermodynamic_boundary_reports_complete);
    CHECK(restored.runtime.advection.thermodynamic_boundary_reports[tmv::qc_boundary_report_index]
              .outward_boundary_inventory == Approx(3.0));
    CHECK(restored.runtime.advection.cumulative_thermodynamic_boundary_reports[tmv::qh_boundary_report_index]
              .closure_residual == Approx(1.0e-4));
    CHECK(restored.runtime.advection.thermodynamic_boundary_report_step_count == 10);
    CHECK(restored.runtime.advection.thermodynamic_moist_energy_reports_complete);
    CHECK(restored.runtime.advection.thermodynamic_moist_energy_reports[tmv::qv_boundary_report_index]
              .outward_boundary_energy_j == Approx(14.0));
    CHECK(restored.runtime.advection.cumulative_thermodynamic_moist_energy_reports[tmv::qh_boundary_report_index]
              .committed_state_change_j == Approx(1200.0));
    CHECK(restored.runtime.advection.thermodynamic_moist_energy_report_step_count == 10);
    CHECK(restored.runtime.advection.tracer_roundoff_report.complete);
    CHECK(restored.runtime.advection.tracer_roundoff_report.physical_inventory_correction_kg == Approx(2.5e-35));
    CHECK(restored.runtime.advection.tracer_roundoff_report.negative_subnormal_count == 3);
    CHECK(restored.runtime.advection.tracer_roundoff_report.closure_canonicalization_count == 1);
    CHECK(restored.runtime.advection.cumulative_tracer_roundoff_report.most_negative_repaired_value ==
          -17.0f * std::numeric_limits<float>::denorm_min());
    CHECK(restored.runtime.advection.tracer_roundoff_report_step_count == 10);
    CHECK(restored.runtime.advection.tracer_roundoff_cumulative_history_known);
    CHECK(restored.runtime.compute.backend == nullptr);
    CHECK(restored.runtime.compute.requested_kind == ComputeBackendKind::Vulkan);
    CHECK(restored.runtime.parity.total_samples == 4);
    CHECK(restored.runtime.parity.call_counters.at("slow_tendency") == 9);
    REQUIRE(restored.runtime.parity.stats.size() == 1);
    CHECK(restored.runtime.parity.stats.begin()->second.worst_abs == Approx(2.5e-6));
    CHECK(restored.runtime.chaos.diagnostics.warnings == std::vector<std::string>{"checkpoint warning"});
    // Wall-clock observer timings are deliberately canonicalized so they
    // cannot make scientifically identical restarts bytewise different.
    CHECK(restored.runtime.chaos.diagnostics.time_noise_update == 0.0);
    CHECK(restored.runtime.chaos.diagnostics.time_apply_ic == 0.0);
    CHECK(restored.runtime.chaos.diagnostics.time_apply_tend == 0.0);
    CHECK(restored.runtime.chaos.pbl_base_tendencies_valid);
    CHECK(restored.runtime.chaos.base_dtke_dt_pbl(1, 1, 1) == Approx(5.0f));
    CHECK(restored.runtime.chaos.base_sfc_ustar == std::vector<float>(6, 6.0f));
    CHECK(restored.runtime.chaos.base_sfc_tau_u == std::vector<float>(6, 7.0f));
    CHECK(restored.runtime.chaos.base_sfc_tau_v == std::vector<float>(6, 8.0f));
    CHECK(restored.runtime.chaos.base_sfc_sensible_heat_flux == std::vector<float>(6, 9.0f));
    CHECK(restored.runtime.chaos.base_sfc_latent_heat_flux == std::vector<float>(6, 10.0f));
    CHECK(restored.runtime.chaos.base_sfc_moisture_flux == std::vector<float>(6, 11.0f));
    CHECK(restored.runtime.microphysics.last_water_repair_report.species[0].lower_bound_count == 3);
    CHECK(restored.runtime.microphysics.last_water_repair_report.species[0].lower_bound_roundoff_count == 2);
    CHECK(restored.runtime.microphysics.last_water_repair_report.species[0].lower_bound_roundoff_correction_kg ==
          Approx(1.25e-8));
    CHECK(restored.runtime.microphysics.last_water_repair_report.species[0].storage_rounding_kg == Approx(-2.5e-9));
    CHECK(restored.runtime.microphysics.last_thermodynamic_repair_report.microphysics_heat_j == Approx(42.5));
    CHECK(restored.runtime.microphysics.last_thermodynamic_repair_report.theta_storage_residual_j == Approx(0.125));
    CHECK(restored.runtime.microphysics.last_thermodynamic_repair_report.theta_storage_residual_release_j ==
          Approx(-0.0625));
    CHECK(restored.runtime.microphysics.last_thermodynamic_repair_report.boundary_layer_cloud_liquid_tendency_kg ==
          Approx(-0.375));
    CHECK(restored.runtime.microphysics.last_thermodynamic_repair_report.boundary_layer_cloud_ice_tendency_kg ==
          Approx(0.625));
    const auto& restored_moist_commit = restored.runtime.microphysics.last_thermodynamic_repair_report;
    CHECK(restored_moist_commit.moist_dry_internal_change_j == Approx(1.0));
    CHECK(restored_moist_commit.moist_water_sensible_change_j == Approx(2.0));
    CHECK(restored_moist_commit.moist_phase_reference_change_j == Approx(3.0));
    CHECK(restored_moist_commit.moist_carrier_kinetic_change_j == Approx(4.0));
    CHECK(restored_moist_commit.moist_gravitational_change_j == Approx(5.0));
    CHECK(restored_moist_commit.moist_subgrid_tke_change_j == Approx(0.0));
    CHECK(restored_moist_commit.moist_committed_energy_change_j == Approx(15.25));
    CHECK(restored_moist_commit.moist_energy_closure_residual_j == Approx(0.25));
    CHECK(restored_moist_commit.moist_energy_complete);
    const std::array<double, 7> expected_species_tendencies = {-7.0, 1.0, 2.0, 0.5, 1.5, 1.25, 0.75};
    for (std::size_t species = 0; species < expected_species_tendencies.size(); ++species)
    {
        CHECK(restored.runtime.microphysics.last_thermodynamic_repair_report
                  .microphysics_water_tendency_kg_by_species[species] == Approx(expected_species_tendencies[species]));
    }
    CHECK(restored.runtime.microphysics.last_conservation_report.active);
    CHECK(restored.runtime.microphysics.last_conservation_report.surface_precipitation_kg == Approx(0.25));
    CHECK(restored.runtime.microphysics.last_conservation_report.relative_imbalance == Approx(5.0e-13));
    CHECK(restored.runtime.microphysics.theta_storage_residual[1][1][1] == Approx(1.25e-6f));

    restored.fields.theta[0][0][0] = -99.0f;
    CHECK(source.fields.theta[0][0][0] != Approx(-99.0f));
}

TEST_CASE("mapped shortwave checkpoint validation matches the runtime residual contract",
          "[core][checkpoint][radiation][terrain][shortwave]")
{
    SimulationStorageState    source = sample_state();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    source.runtime.radiation.held_unresolved_reflected_power_w = 6.0e-6;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));

    source.runtime.radiation.held_unresolved_reflected_power_w = 2.0e-5;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error.find("invalid mapped-terrain shortwave contract") != std::string::npos);
}

TEST_CASE("simple-grey checkpoint rejects malformed retained flux state", "[core][checkpoint][radiation][simple-grey]")
{
    SimulationStorageState source = sample_state();
    source.runtime.radiation.held_flux_diagnostics.downward_shortwave_wm2.pop_back();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error == "invalid simple-grey checkpoint state");
    CHECK(bytes.empty());
}

TEST_CASE("user SGS filter width is restart portable and malformed state is atomic",
          "[core][checkpoint][turbulence][filter-width][sci357]")
{
    SimulationStorageState source                        = sample_state();
    source.runtime.turbulence.config.filter_width        = "user";
    source.runtime.turbulence.config.user_filter_width_m = 500.0;
    source.runtime.turbulence.scheme                     = create_turbulence_scheme("smagorinsky");
    source.runtime.turbulence.scheme->initialize(source.runtime.turbulence.config);

    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.turbulence.config.filter_width == "user");
    REQUIRE(restored.runtime.turbulence.config.user_filter_width_m == 500.0);

    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(source, restored, error));
    restored.runtime.turbulence.config.user_filter_width_m = 1000.0;
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(source, restored, error));
    REQUIRE(error.find("turbulence") != std::string::npos);
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(restored, source, error));

    source.runtime.turbulence.config.user_filter_width_m = 0.0;
    bytes.assign(3, 0x5a);
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error == "invalid user filter-width checkpoint configuration");
    REQUIRE(bytes == std::vector<std::uint8_t>(3, 0x5a));
}

TEST_CASE("legacy TKE checkpoints retain their historical stability correction",
          "[core][checkpoint][turbulence][tke][compatibility][sci359]")
{
    constexpr std::array<std::uint8_t, 8> magic           = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source          = sample_state();
    source.runtime.turbulence.config.scheme_id            = "tke";
    source.runtime.turbulence.config.stability_correction = "none";
    source.runtime.turbulence.scheme                      = create_turbulence_scheme("tke");
    source.runtime.turbulence.scheme->initialize(source.runtime.turbulence.config);

    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));

    SimulationStorageState current;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, current, error));
    REQUIRE(current.runtime.turbulence.config.stability_correction == "none");

    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto runtime = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    REQUIRE(runtime->version == 55);
    strip_terrain_placement_v49_suffix(*runtime, source);
    runtime->version = 47;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState legacy;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, legacy, error));
    REQUIRE(legacy.runtime.turbulence.config.stability_correction == "tke_based");
}

TEST_CASE("variable diffusion configuration reconstructs from checkpoint", "[core][checkpoint][diffusion][variable-k]")
{
    SimulationStorageState source = sample_state();
    auto&                  config = source.runtime.diffusion.config;
    config.scheme_id              = "explicit";
    config.K_v                    = 0.0;
    config.use_variable_K         = true;
    config.coefficient_source     = "linear_height";
    config.K_v_surface            = 4.0;
    config.K_v_top                = 20.0;
    source.runtime.diffusion.prescribed_vertical_K.resize(3, 3, 3, 11.0f);
    source.runtime.diffusion.prescribed_vertical_dz = {1.0, 2.0, 3.0};

    std::vector<std::uint8_t> bytes;
    std::string               error;
    INFO(error);
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    const auto& restored_config = restored.runtime.diffusion.config;
    REQUIRE(restored_config.use_variable_K);
    REQUIRE(restored_config.coefficient_source == "linear_height");
    REQUIRE(restored_config.K_v == 0.0);
    REQUIRE(restored_config.K_v_surface == 4.0);
    REQUIRE(restored_config.K_v_top == 20.0);
    REQUIRE(restored.runtime.diffusion.prescribed_vertical_K.empty());
    REQUIRE(restored.runtime.diffusion.prescribed_vertical_dz.empty());
}

TEST_CASE("trajectory checkpoint section is optional, required when present, and exact",
          "[core][checkpoint][trajectories]")
{
    SimulationStorageState source = sample_state();
    source.time_s                 = 1.0;
    constexpr int n               = 8;
    Field3D u(n, n, n, 2.0f), v(n, n, n, -1.0f), w(n, n, n, 0.25f), rho(n, n, n, 1.0f), pressure(n, n, n, 90000.0f),
        vorticity(n, n, n, 0.01f);
    tmv::TrajectoryConfig config;
    config.enabled = true;
    config.nx = config.ny = config.nz = 1;
    config.seed_time_s                = 0.0;
    config.sample_interval_s          = 0.75;
    config.max_samples_per_parcel     = 4;
    config.half_x_m = config.half_y_m = config.half_z_m = 0.0;
    config.center_x_m = config.center_y_m = 3000.0;
    config.center_z_m                     = 1500.0;

    tmv::TrajectorySet live;
    live.configure(config, CoordinateSystem::Cartesian, false, n, n, n, 1000.0, 1000.0, 500.0);
    live.seed(0.0, u, v, w, rho, pressure, vorticity);
    live.advance(1.0, 1.0, u, v, w, rho, pressure, vorticity);

    std::vector<std::uint8_t> ordinary;
    std::vector<std::uint8_t> enriched;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, ordinary, error));
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, &live, enriched, error));
    REQUIRE(enriched.size() > ordinary.size());

    tmv::checkpoint::SimulationCheckpointAttachments ordinary_attachments;
    SimulationStorageState                           ordinary_state;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(ordinary, ordinary_state, ordinary_attachments, error));
    CHECK_FALSE(ordinary_attachments.trajectories.has_value());

    tmv::checkpoint::SimulationCheckpointAttachments attachments;
    SimulationStorageState                           restored_state;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(enriched, restored_state, attachments, error));
    REQUIRE(attachments.trajectories.has_value());
    REQUIRE(attachments.trajectories->parcels.size() == 1);
    CHECK(attachments.trajectories->configuration.sample_interval_s == Approx(0.75));
    CHECK(attachments.trajectories->configuration.max_samples_per_parcel == 4);
    CHECK(attachments.trajectories->next_sample_time_s == Approx(1.5));
    REQUIRE(attachments.trajectories->parcels[0].path.size() == 2);
    CHECK(attachments.trajectories->parcels[0].path[1].t == 1.0f);
    CHECK(attachments.trajectories->parcels[0].path[1].x == live.parcels()[0].path[1].x);

    std::vector<std::uint8_t>         payload;
    const std::array<std::uint8_t, 8> magic = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    REQUIRE(tmv::checkpoint::wire::unwrap(
        enriched, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    const auto trajectory =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 8; });
    REQUIRE(trajectory != sections.end());
    CHECK((trajectory->flags & tmv::checkpoint::wire::section_required) != 0);
    CHECK(trajectory->version == 2);

    const auto* prior_parcels = attachments.trajectories->parcels.data();
    std::vector<std::uint8_t> prior_state;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored_state, prior_state, error));
    for (bool valid_envelope : {false, true})
    {
        auto corrupt = enriched;
        if (valid_envelope)
        {
            trajectory->payload.pop_back();
            REQUIRE(tmv::checkpoint::wire::wrap_sections(
                magic, tmv::checkpoint::simulation_checkpoint_version, sections, corrupt, error));
        }
        else
            corrupt.back() ^= 1;
        REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(corrupt, restored_state, attachments, error));
        REQUIRE(attachments.trajectories.has_value());
        CHECK(attachments.trajectories->parcels.data() == prior_parcels);
        CHECK(attachments.trajectories->parcels[0].path.size() == 2);
        CHECK(attachments.trajectories->parcels[0].path[1].x == live.parcels()[0].path[1].x);
        std::vector<std::uint8_t> after;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored_state, after, error));
        CHECK(after == prior_state);
    }
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(ordinary, restored_state, attachments, error));
    CHECK_FALSE(attachments.trajectories.has_value());
}

TEST_CASE("checkpointed movement confirmation preserves ordinary and emergency decisions",
          "[core][checkpoint][movement_authority]")
{
    tmv::domain::StormWindowMovementPolicy policy;
    policy.inner_margin_m       = 500.0;
    policy.release_margin_m     = 600.0;
    policy.dx_m                 = 100.0;
    policy.dy_m                 = 200.0;
    policy.maximum_shift_cells  = 10;
    policy.minimum_confidence   = 0.8;
    policy.confirmation_samples = 2;
    const auto evidence_at      = [](double time_s, double east_clearance_m)
    {
        tmv::domain::StormWindowMovementEvidence evidence;
        evidence.sample_time_s              = time_s;
        evidence.selected_feature_available = true;
        evidence.continuity_authoritative   = true;
        evidence.motion_available           = true;
        evidence.confidence                 = 0.95;
        evidence.distance_west_m            = 2000.0;
        evidence.distance_east_m            = east_clearance_m;
        evidence.distance_south_m           = 2000.0;
        evidence.distance_north_m           = 2000.0;
        evidence.velocity_x_ms              = 10.0;
        return evidence;
    };

    SimulationStorageState continuous        = sample_state();
    continuous.runtime.storm_window_movement = {};
    const auto pending                       = tmv::domain::evaluate_storm_window_movement(
        continuous.runtime.storm_window_movement, evidence_at(1.0, policy.dx_m), policy);
    REQUIRE(pending.status == tmv::domain::StormWindowMovementStatus::PendingConfirmation);
    REQUIRE(pending.confirmation_count == 1);

    std::vector<std::uint8_t> split_bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(continuous, split_bytes, error));
    SimulationStorageState resumed;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(split_bytes, resumed, error));
    const auto continuous_request = tmv::domain::evaluate_storm_window_movement(
        continuous.runtime.storm_window_movement, evidence_at(2.0, policy.dx_m), policy);
    const auto resumed_request = tmv::domain::evaluate_storm_window_movement(
        resumed.runtime.storm_window_movement, evidence_at(2.0, policy.dx_m), policy);
    REQUIRE(continuous_request.status == tmv::domain::StormWindowMovementStatus::Requested);
    REQUIRE(continuous_request.reason == tmv::domain::StormWindowMovementReason::RequestReady);
    REQUIRE(resumed_request.status == continuous_request.status);
    REQUIRE(resumed_request.reason == continuous_request.reason);
    REQUIRE(resumed_request.requested_i_shift_cells == continuous_request.requested_i_shift_cells);
    REQUIRE(resumed_request.confirmation_count == 2);
    std::vector<std::uint8_t> continuous_bytes;
    std::vector<std::uint8_t> resumed_bytes;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(continuous, continuous_bytes, error));
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(resumed, resumed_bytes, error));
    REQUIRE(resumed_bytes == continuous_bytes);

    SimulationStorageState clean        = sample_state();
    clean.runtime.storm_window_movement = {};
    std::vector<std::uint8_t> clean_bytes;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(clean, clean_bytes, error));
    SimulationStorageState clean_resumed;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(clean_bytes, clean_resumed, error));
    const auto emergency = tmv::domain::evaluate_storm_window_movement(
        clean_resumed.runtime.storm_window_movement, evidence_at(1.0, policy.dx_m - 1.0), policy);
    REQUIRE(emergency.status == tmv::domain::StormWindowMovementStatus::Requested);
    REQUIRE(emergency.reason == tmv::domain::StormWindowMovementReason::EmergencyContainment);
    REQUIRE(emergency.confirmation_count == 1);
}

TEST_CASE("checkpointed movement state preserves projected-relaxation v11 decisions",
          "[core][checkpoint][movement_authority][projected_relaxation]")
{
    tmv::domain::StormWindowMovementPolicy policy;
    policy.inner_margin_m        = 15600.0;
    policy.release_margin_m      = 16800.0;
    policy.dx_m                  = 1200.0;
    policy.dy_m                  = 1200.0;
    policy.maximum_shift_cells   = 14;
    policy.cooldown_s            = 300.0;
    policy.minimum_confidence    = 0.65;
    policy.confirmation_samples  = 2;
    policy.relaxation_margin_x_m = 4800.0;
    policy.relaxation_margin_y_m = 4800.0;

    const auto evidence_at = [](double time_s)
    {
        tmv::domain::StormWindowMovementEvidence evidence;
        evidence.sample_time_s              = time_s;
        evidence.selected_feature_available = true;
        evidence.continuity_authoritative   = true;
        evidence.motion_available           = true;
        evidence.confidence                 = 0.95;
        return evidence;
    };

    SimulationStorageState continuous         = sample_state();
    continuous.runtime.storm_window_movement  = {};
    auto safe                                 = evidence_at(3480.0);
    safe.distance_west_m                      = 33600.0;
    safe.distance_east_m                      = 24600.0;
    safe.distance_south_m                     = 37800.0;
    safe.distance_north_m                     = 19200.0;
    safe.velocity_x_ms                        = 18.55;
    safe.velocity_y_ms                        = 13.51;
    safe.envelope_face_motion_available       = true;
    safe.envelope_face_motion_interval_s      = 30.0;
    safe.envelope_east_coordinate_velocity_ms = 20.0;
    const auto safe_decision =
        tmv::domain::evaluate_storm_window_movement(continuous.runtime.storm_window_movement, safe, policy);
    REQUIRE(safe_decision.status == tmv::domain::StormWindowMovementStatus::Ineligible);
    REQUIRE(safe_decision.reason == tmv::domain::StormWindowMovementReason::NoApproachingBreach);

    std::vector<std::uint8_t> split_bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(continuous, split_bytes, error));
    SimulationStorageState resumed;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(split_bytes, resumed, error));

    auto projected                                  = evidence_at(3510.0);
    projected.distance_west_m                       = 34200.0;
    projected.distance_east_m                       = 11400.0;
    projected.distance_south_m                      = 37800.0;
    projected.distance_north_m                      = 9600.0;
    projected.velocity_x_ms                         = 18.861131315995493;
    projected.velocity_y_ms                         = 13.820786516951411;
    projected.envelope_face_motion_available        = true;
    projected.envelope_face_motion_interval_s       = 30.0;
    projected.envelope_west_coordinate_velocity_ms  = 20.0;
    projected.envelope_east_coordinate_velocity_ms  = 440.0;
    projected.envelope_south_coordinate_velocity_ms = 0.0;
    projected.envelope_north_coordinate_velocity_ms = 320.0;

    const auto continuous_request =
        tmv::domain::evaluate_storm_window_movement(continuous.runtime.storm_window_movement, projected, policy);
    const auto resumed_request =
        tmv::domain::evaluate_storm_window_movement(resumed.runtime.storm_window_movement, projected, policy);
    REQUIRE(continuous_request.status == tmv::domain::StormWindowMovementStatus::Requested);
    REQUIRE(continuous_request.reason == tmv::domain::StormWindowMovementReason::ProjectedRelaxationRisk);
    REQUIRE(continuous_request.confirmation_count == 1);
    REQUIRE(continuous_request.requested_i_shift_cells == 14);
    REQUIRE(continuous_request.requested_j_shift_cells == 14);
    REQUIRE(resumed_request.status == continuous_request.status);
    REQUIRE(resumed_request.reason == continuous_request.reason);
    REQUIRE(resumed_request.confirmation_count == continuous_request.confirmation_count);
    REQUIRE(resumed_request.requested_i_shift_cells == continuous_request.requested_i_shift_cells);
    REQUIRE(resumed_request.requested_j_shift_cells == continuous_request.requested_j_shift_cells);

    std::vector<std::uint8_t> continuous_bytes;
    std::vector<std::uint8_t> resumed_bytes;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(continuous, continuous_bytes, error));
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(resumed, resumed_bytes, error));
    REQUIRE(resumed_bytes == continuous_bytes);
}

TEST_CASE("runtime v39 restores passive-tracer cumulative history as unknown",
          "[core][checkpoint][advection][tracer][compatibility]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                      source = sample_state();
    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto runtime = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    REQUIRE(runtime->version == 55);
    strip_simple_grey_v45_suffix(*runtime, source);
    REQUIRE(runtime->version == 44);
    strip_rrtmg_v41_suffix(*runtime, source);
    const auto suffix = tracer_roundoff_v40_suffix(source);
    REQUIRE(runtime->payload.size() >= suffix.bytes.size());
    const auto start = runtime->payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime->payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime->payload.erase(start, runtime->payload.end());
    runtime->version = 39;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.advection.tracer_roundoff_report_step_count == 0);
    REQUIRE_FALSE(restored.runtime.advection.tracer_roundoff_report.complete);
    REQUIRE_FALSE(restored.runtime.advection.cumulative_tracer_roundoff_report.complete);
    REQUIRE_FALSE(restored.runtime.advection.tracer_roundoff_cumulative_history_known);

    std::vector<std::uint8_t> upgraded;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored, upgraded, error));
    SimulationStorageState upgraded_restore;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(upgraded, upgraded_restore, error));
    REQUIRE_FALSE(upgraded_restore.runtime.advection.tracer_roundoff_cumulative_history_known);
}

TEST_CASE("runtime v40 rejects corrupt passive-tracer ledgers transactionally",
          "[core][checkpoint][advection][tracer][corruption]")
{
    constexpr std::array<std::uint8_t, 8> magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    const SimulationStorageState          source = sample_state();

    const auto reject = [&](const SimulationStorageState& replacement)
    {
        std::vector<std::uint8_t>                   bytes, payload;
        std::vector<tmv::checkpoint::wire::Section> sections;
        std::string                                 error;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
        REQUIRE(tmv::checkpoint::wire::unwrap(
            bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
        REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
        auto runtime =
            std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
        REQUIRE(runtime != sections.end());
        replace_tracer_roundoff_v40_suffix(*runtime, source, replacement);
        REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
        tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
        SimulationStorageState destination;
        destination.time_s = 444.0;
        REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
        REQUIRE(error.find("passive-tracer roundoff checkpoint ledger") != std::string::npos);
        REQUIRE(destination.time_s == 444.0);
    };

    SimulationStorageState zero_step_nonzero                                       = source;
    zero_step_nonzero.runtime.advection.tracer_roundoff_report.complete            = false;
    zero_step_nonzero.runtime.advection.cumulative_tracer_roundoff_report.complete = false;
    zero_step_nonzero.runtime.advection.tracer_roundoff_report_step_count          = 0;
    reject(zero_step_nonzero);

    SimulationStorageState correction_without_events = source;
    auto&                  last = correction_without_events.runtime.advection.tracer_roundoff_report;
    last                        = {1.0, 0.0, 0.0f, 0, 0, 0, true};
    correction_without_events.runtime.advection.cumulative_tracer_roundoff_report = last;
    correction_without_events.runtime.advection.tracer_roundoff_report_step_count = 1;
    reject(correction_without_events);
}

TEST_CASE("runtime v41 rejects corrupt RRTMG composition transactionally",
          "[core][checkpoint][radiation][rrtmg][corruption]")
{
    constexpr std::array<std::uint8_t, 8> magic          = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    const SimulationStorageState          source         = sample_state();
    SimulationStorageState                replacement    = source;
    replacement.runtime.radiation.config.rrtmg_ozone_vmr = 2.0e-2;

    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto runtime = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    replace_rrtmg_v41_suffix(*runtime, source, replacement);
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState destination;
    destination.time_s = 444.0;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
    REQUIRE(error.find("invalid RRTMG checkpoint configuration") != std::string::npos);
    REQUIRE(destination.time_s == 444.0);
}

TEST_CASE("simulation checkpoint corruption is fail-closed and transactional", "[core][checkpoint][mod03]")
{
    SimulationStorageState    source = sample_state();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(bytes.size() > 80);
    bytes[bytes.size() / 2] ^= 0x40;

    SimulationStorageState destination;
    destination.time_s                            = 1234.0;
    destination.configuration.microphysics_scheme = "sentinel";
    CHECK_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
    CHECK_FALSE(error.empty());
    CHECK(destination.time_s == Approx(1234.0));
    CHECK(destination.configuration.microphysics_scheme == "sentinel");
}

TEST_CASE("simulation checkpoint preserves complete conservative terrain geometry",
          "[core][checkpoint][terrain][sci075]")
{
    SimulationStorageState source    = sample_state();
    TerrainRuntimeState&   terrain   = source.runtime.terrain;
    terrain.config.scheme_id         = "bell";
    terrain.config.coord_id          = "smoothed";
    terrain.config.ztop              = static_cast<double>(source.grid.nz) * source.grid.dz;
    terrain.config.smoothing_decay   = 0.4;
    terrain.config.bell.h0           = 20.0;
    terrain.config.bell.a            = 250.0;
    terrain.config.bell.axisymmetric = true;
    terrain.scheme                   = create_terrain_scheme(terrain.config.scheme_id);
    terrain.scheme->initialize(terrain.config);
    topography::initialize_topography(terrain.topography, source.grid.nr, source.grid.nth);
    topography::initialize_metrics(terrain.metrics, source.grid.nr, source.grid.nth, source.grid.nz);
    terrain.scheme->build_topography(terrain.config, terrain.topography);
    terrain.scheme->build_metrics(terrain.config, terrain.topography, terrain.metrics);
    terrain_transport::validate_geometry(terrain.metrics);

    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.terrain.scheme);
    CHECK(restored.runtime.terrain.scheme->name() == "bell");
    terrain_transport::validate_geometry(restored.runtime.terrain.metrics);

    const TerrainMetrics3D& expected = terrain.metrics;
    const TerrainMetrics3D& actual   = restored.runtime.terrain.metrics;
    CHECK(actual.z.values == expected.z.values);
    CHECK(actual.J.values == expected.J.values);
    CHECK(actual.mx.values == expected.mx.values);
    CHECK(actual.my.values == expected.my.values);
    CHECK(actual.zeta.values == expected.zeta.values);
    CHECK(actual.z_face.values == expected.z_face.values);
    CHECK(actual.Jx_face.values == expected.Jx_face.values);
    CHECK(actual.Jy_face.values == expected.Jy_face.values);
    CHECK(actual.Gx_face.values == expected.Gx_face.values);
    CHECK(actual.Gy_face.values == expected.Gy_face.values);
    CHECK(actual.dx == expected.dx);
    CHECK(actual.dy == expected.dy);
    CHECK(actual.dzeta == expected.dzeta);

    source.runtime.terrain.metrics.Gx_face(1, 1, 1) = std::numeric_limits<double>::quiet_NaN();
    bytes.clear();
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("invalid active terrain geometry") != std::string::npos);
}

TEST_CASE("runtime v49 preserves Earth-fixed terrain identity transactionally",
          "[core][checkpoint][terrain][earth_fixed][compatibility]")
{
    constexpr std::array<std::uint8_t, 8> magic   = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source  = sample_state();
    auto&                                 terrain = source.runtime.terrain;
    terrain.config.scheme_id                      = "bell";
    terrain.config.coord_id                       = "smoothed";
    terrain.config.horizontal_placement_id        = "earth_fixed";
    terrain.config.ztop                           = static_cast<double>(source.grid.nz) * source.grid.dz;
    terrain.config.smoothing_decay                = 0.2;
    terrain.config.bell.h0                        = 20.0;
    terrain.config.bell.a                         = 250.0;
    terrain.config.bell.axisymmetric              = true;
    terrain.config.feature_center_earth_x_m       = 127500.0;
    terrain.config.feature_center_earth_y_m       = -46250.0;
    terrain.scheme                                = create_terrain_scheme("bell");
    terrain.scheme->initialize(terrain.config);

    tmv::EarthFixedTerrainBuildRequest request;
    request.nr                     = source.grid.nr;
    request.nth                    = source.grid.nth;
    request.nz                     = source.grid.nz;
    request.dx_m                   = source.grid.dx;
    request.dy_m                   = source.grid.dy;
    request.grid_origin_earth_m    = source.configuration.domain_frame.earth_origin_at(source.time_s);
    request.feature_center_earth_m = {terrain.config.feature_center_earth_x_m, terrain.config.feature_center_earth_y_m};
    request.sampling = GENERATE(tmv::EarthFixedTerrainSampling::Point, tmv::EarthFixedTerrainSampling::CellAverageBilinear);
    auto prepared                  = tmv::prepare_earth_fixed_terrain_commit(terrain, request);
    REQUIRE(tmv::commit_earth_fixed_terrain(prepared, terrain));

    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.terrain.config.horizontal_placement_id == "earth_fixed");
    REQUIRE(restored.runtime.terrain.config.feature_center_earth_x_m == terrain.config.feature_center_earth_x_m);
    REQUIRE(restored.runtime.terrain.earth_fixed_placement.sampling == request.sampling);
    auto different_sampling = restored;
    different_sampling.runtime.terrain.earth_fixed_placement.sampling =
        request.sampling == tmv::EarthFixedTerrainSampling::Point
            ? tmv::EarthFixedTerrainSampling::CellAverageBilinear : tmv::EarthFixedTerrainSampling::Point;
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(source, different_sampling, error));
    REQUIRE(restored.runtime.terrain.earth_fixed_placement.active);
    REQUIRE(restored.runtime.terrain.earth_fixed_placement.generation == 1);
    REQUIRE(restored.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.x_m == request.grid_origin_earth_m.x_m);
    REQUIRE(restored.runtime.terrain.topography.h == terrain.topography.h);
    REQUIRE(restored.runtime.terrain.metrics.J.values == terrain.metrics.J.values);

    SimulationStorageState mismatched = restored;
    mismatched.runtime.terrain.config.feature_center_earth_x_m += 1.0;
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(source, mismatched, error));
    REQUIRE(error.find("terrain configuration") != std::string::npos);
    REQUIRE(source.runtime.terrain.config.feature_center_earth_x_m == 127500.0);

    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto runtime = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    REQUIRE(runtime->version == 55);
    strip_unsplit_rk_v50_suffix(*runtime, source);
    const auto             valid_suffix     = terrain_placement_v49_suffix(source);
    SimulationStorageState corrupt_identity = source;
    corrupt_identity.runtime.terrain.earth_fixed_placement.feature_center_earth_m.x_m += 1.0;
    const auto invalid_suffix = terrain_placement_v49_suffix(corrupt_identity);
    REQUIRE(valid_suffix.bytes.size() == invalid_suffix.bytes.size());
    REQUIRE(runtime->payload.size() >= valid_suffix.bytes.size());
    const auto suffix_begin = runtime->payload.end() - static_cast<std::ptrdiff_t>(valid_suffix.bytes.size());
    REQUIRE(std::equal(suffix_begin, runtime->payload.end(), valid_suffix.bytes.begin(), valid_suffix.bytes.end()));
    runtime->payload.erase(suffix_begin, runtime->payload.end());
    runtime->payload.insert(runtime->payload.end(), invalid_suffix.bytes.begin(), invalid_suffix.bytes.end());
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState destination;
    destination.time_s                           = 444.0;
    destination.runtime.terrain.config.scheme_id = "sentinel";
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
    REQUIRE(error.find("invalid Earth-fixed terrain checkpoint identity") != std::string::npos);
    REQUIRE(destination.time_s == 444.0);
    REQUIRE(destination.runtime.terrain.config.scheme_id == "sentinel");
}

TEST_CASE("runtime v50 preserves unsplit-RK completed-step ownership", "[checkpoint][runtime][unsplit-rk][v50]")
{
    constexpr std::array<std::uint8_t, 8> magic        = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source       = sample_state();
    auto                                  seed_unsplit = [](tmv::CompletedStepEnergyReportV1& report, double value)
    {
        report.unsplit_rk_committed_change_j = value;
        report.carrier_kinetic_change_j += value;
        report.committed_owner_sum_j = tmv::completed_step_owner_sum(report);
        report.closure_residual_j    = report.committed_owner_sum_j - tmv::completed_step_reservoir_sum(report);
    };
    seed_unsplit(source.runtime.time_stepping.last_completed_step_energy, 12.5);
    seed_unsplit(source.runtime.time_stepping.cumulative_completed_step_energy, 125.0);

    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.time_stepping.last_completed_step_energy.unsplit_rk_committed_change_j == Approx(12.5));
    REQUIRE(restored.runtime.time_stepping.cumulative_completed_step_energy.unsplit_rk_committed_change_j ==
            Approx(125.0));

    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto runtime = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    REQUIRE(runtime->version == 55);
    strip_execution_v51_suffix(*runtime, source);
    SimulationStorageState invalid = source;
    invalid.runtime.time_stepping.cumulative_completed_step_energy.unsplit_rk_committed_change_j =
        std::numeric_limits<double>::quiet_NaN();
    const auto valid_suffix   = unsplit_rk_v50_suffix(source);
    const auto invalid_suffix = unsplit_rk_v50_suffix(invalid);
    REQUIRE(valid_suffix.bytes.size() == invalid_suffix.bytes.size());
    const auto start = runtime->payload.end() - static_cast<std::ptrdiff_t>(valid_suffix.bytes.size());
    REQUIRE(std::equal(start, runtime->payload.end(), valid_suffix.bytes.begin(), valid_suffix.bytes.end()));
    std::copy(invalid_suffix.bytes.begin(), invalid_suffix.bytes.end(), start);
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    SimulationStorageState rejected;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, rejected, error));
    REQUIRE(error.find("unsplit-RK completed-step checkpoint owner") != std::string::npos);
}

TEST_CASE("runtime v48 terrain identity defaults backward to grid-attached",
          "[core][checkpoint][terrain][compatibility]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    const SimulationStorageState                source = sample_state();
    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto runtime = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    strip_terrain_placement_v49_suffix(*runtime, source);
    REQUIRE(runtime->version == 48);
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.terrain.config.horizontal_placement_id == "grid_attached");
    REQUIRE(restored.runtime.terrain.config.feature_center_earth_x_m == 0.0);
    REQUIRE(restored.runtime.terrain.config.feature_center_earth_y_m == 0.0);
    REQUIRE_FALSE(restored.runtime.terrain.earth_fixed_placement.active);
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(source, restored, error));

    SimulationStorageState invalid                          = source;
    invalid.runtime.terrain.config.feature_center_earth_x_m = 1.0;
    bytes.assign(3, 0x5a);
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(invalid, bytes, error));
    REQUIRE(error == "cannot checkpoint invalid Earth-fixed terrain identity");
    REQUIRE(bytes == std::vector<std::uint8_t>(3, 0x5a));
}

TEST_CASE("simulation checkpoint configuration v2 restores new evaporation defaults",
          "[core][checkpoint][compatibility][microphysics]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                      source = sample_state();
    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));

    const auto configuration =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 2; });
    REQUIRE(configuration != sections.end());
    downgrade_configuration_v16_to_v14(*configuration, source);
    REQUIRE(configuration->version == 14);

    tmv::checkpoint::wire::Encoder added_v14_execution_mode;
    added_v14_execution_mode.integer(static_cast<std::int32_t>(source.configuration.execution_mode));
    REQUIRE(configuration->payload.size() >= added_v14_execution_mode.bytes.size());
    const auto added_v14_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v14_execution_mode.bytes.size());
    REQUIRE(std::equal(added_v14_begin,
                       configuration->payload.end(),
                       added_v14_execution_mode.bytes.begin(),
                       added_v14_execution_mode.bytes.end()));
    configuration->payload.erase(added_v14_begin, configuration->payload.end());
    configuration->version = 13;

    tmv::checkpoint::wire::Encoder added_v13_nesting_selection;
    added_v13_nesting_selection.boolean(source.configuration.nesting.enabled);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_origin_parent_cell_i);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_origin_parent_cell_j);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_cells_x);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_cells_y);
    added_v13_nesting_selection.integer(source.configuration.nesting.relaxation_width_child_cells);
    added_v13_nesting_selection.integer(source.configuration.nesting.boundary_stencil_parent_cells);
    added_v13_nesting_selection.floating(source.configuration.nesting.relaxation_e_folding_time_s);
    REQUIRE(configuration->payload.size() >= added_v13_nesting_selection.bytes.size());
    const auto added_v13_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v13_nesting_selection.bytes.size());
    REQUIRE(std::equal(added_v13_begin,
                       configuration->payload.end(),
                       added_v13_nesting_selection.bytes.begin(),
                       added_v13_nesting_selection.bytes.end()));
    configuration->payload.erase(added_v13_begin, configuration->payload.end());
    configuration->version = 12;

    tmv::checkpoint::wire::Encoder added_v12_nesting;
    added_v12_nesting.integer(source.configuration.nesting.parent_ratio_x);
    added_v12_nesting.integer(source.configuration.nesting.parent_ratio_y);
    added_v12_nesting.integer(source.configuration.nesting.parent_time_step_ratio);
    added_v12_nesting.integer(static_cast<std::int32_t>(source.configuration.nesting.feedback));
    REQUIRE(configuration->payload.size() >= added_v12_nesting.bytes.size());
    const auto added_v12_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v12_nesting.bytes.size());
    REQUIRE(std::equal(
        added_v12_begin, configuration->payload.end(), added_v12_nesting.bytes.begin(), added_v12_nesting.bytes.end()));
    configuration->payload.erase(added_v12_begin, configuration->payload.end());
    configuration->version = 11;

    tmv::checkpoint::wire::Encoder added_v11_recycling;
    added_v11_recycling.boolean(source.configuration.storm_tracking.recycle_domain);
    REQUIRE(configuration->payload.size() >= added_v11_recycling.bytes.size());
    const auto added_v11_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v11_recycling.bytes.size());
    REQUIRE(std::equal(added_v11_begin,
                       configuration->payload.end(),
                       added_v11_recycling.bytes.begin(),
                       added_v11_recycling.bytes.end()));
    configuration->payload.erase(added_v11_begin, configuration->payload.end());
    configuration->version = 10;

    tmv::checkpoint::wire::Encoder added_v10_core_area;
    added_v10_core_area.floating(source.configuration.storm_tracking.minimum_core_area_m2);
    REQUIRE(configuration->payload.size() >= added_v10_core_area.bytes.size());
    const auto added_v10_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v10_core_area.bytes.size());
    REQUIRE(std::equal(added_v10_begin,
                       configuration->payload.end(),
                       added_v10_core_area.bytes.begin(),
                       added_v10_core_area.bytes.end()));
    configuration->payload.erase(added_v10_begin, configuration->payload.end());
    configuration->version = 9;

    tmv::checkpoint::wire::Encoder added_v9_frame;
    added_v9_frame.integer(static_cast<std::int32_t>(source.configuration.domain_motion_mode));
    added_v9_frame.floating(source.configuration.domain_frame.earth_origin_at_t0_m.x_m);
    added_v9_frame.floating(source.configuration.domain_frame.earth_origin_at_t0_m.y_m);
    added_v9_frame.floating(source.configuration.domain_frame.velocity_ms.u_ms);
    added_v9_frame.floating(source.configuration.domain_frame.velocity_ms.v_ms);
    REQUIRE(configuration->payload.size() >= added_v9_frame.bytes.size());
    const auto added_v9_begin = configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v9_frame.bytes.size());
    REQUIRE(std::equal(
        added_v9_begin, configuration->payload.end(), added_v9_frame.bytes.begin(), added_v9_frame.bytes.end()));
    configuration->payload.erase(added_v9_begin, configuration->payload.end());

    const auto&                    tracking = source.configuration.storm_tracking;
    tmv::checkpoint::wire::Encoder added_v8_branches;
    added_v8_branches.integer(tracking.topology_confirmation_samples);
    added_v8_branches.integer(static_cast<std::int32_t>(tracking.branch_preference));
    added_v8_branches.floating(tracking.branch_velocity_uncertainty_ms);
    REQUIRE(configuration->payload.size() >= added_v8_branches.bytes.size());
    const auto added_v8_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v8_branches.bytes.size());
    REQUIRE(std::equal(
        added_v8_begin, configuration->payload.end(), added_v8_branches.bytes.begin(), added_v8_branches.bytes.end()));
    configuration->payload.erase(added_v8_begin, configuration->payload.end());

    tmv::checkpoint::wire::Encoder added_v7_rotation;
    added_v7_rotation.floating(tracking.updraft_helicity_bottom_m_agl);
    added_v7_rotation.floating(tracking.updraft_helicity_top_m_agl);
    added_v7_rotation.floating(tracking.minimum_absolute_updraft_helicity_m2s2);
    REQUIRE(configuration->payload.size() >= added_v7_rotation.bytes.size());
    const auto added_v7_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v7_rotation.bytes.size());
    REQUIRE(std::equal(
        added_v7_begin, configuration->payload.end(), added_v7_rotation.bytes.begin(), added_v7_rotation.bytes.end()));
    configuration->payload.erase(added_v7_begin, configuration->payload.end());

    tmv::checkpoint::wire::Encoder added_v6_continuity;
    added_v6_continuity.floating(tracking.maximum_motion_speed_ms);
    added_v6_continuity.floating(tracking.maximum_motion_acceleration_ms2);
    added_v6_continuity.integer(tracking.minimum_continuity_samples);
    added_v6_continuity.integer(tracking.lost_after_samples);
    REQUIRE(configuration->payload.size() >= added_v6_continuity.bytes.size());
    const auto added_v6_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v6_continuity.bytes.size());
    REQUIRE(std::equal(added_v6_begin,
                       configuration->payload.end(),
                       added_v6_continuity.bytes.begin(),
                       added_v6_continuity.bytes.end()));
    configuration->payload.erase(added_v6_begin, configuration->payload.end());

    tmv::checkpoint::wire::Encoder added_v5_timestep;
    added_v5_timestep.floating(source.configuration.configured_dt_large_s);
    REQUIRE(configuration->payload.size() >= added_v5_timestep.bytes.size());
    const auto added_v5_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v5_timestep.bytes.size());
    REQUIRE(std::equal(
        added_v5_begin, configuration->payload.end(), added_v5_timestep.bytes.begin(), added_v5_timestep.bytes.end()));
    configuration->payload.erase(added_v5_begin, configuration->payload.end());

    tmv::checkpoint::wire::Encoder added_v4_tracking;
    added_v4_tracking.boolean(tracking.enabled);
    added_v4_tracking.floating(tracking.interval_s);
    added_v4_tracking.floating(tracking.inner_margin_m);
    added_v4_tracking.integer(tracking.maximum_shift_cells);
    added_v4_tracking.floating(tracking.cooldown_s);
    added_v4_tracking.floating(tracking.minimum_confidence);
    added_v4_tracking.floating(tracking.minimum_updraft_ms);
    added_v4_tracking.floating(tracking.minimum_condensate_path_kgm2);
    added_v4_tracking.integer(tracking.minimum_component_cells);
    REQUIRE(configuration->payload.size() >= added_v4_tracking.bytes.size());
    const auto added_v4_begin =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v4_tracking.bytes.size());
    REQUIRE(std::equal(
        added_v4_begin, configuration->payload.end(), added_v4_tracking.bytes.begin(), added_v4_tracking.bytes.end()));
    configuration->payload.erase(added_v4_begin, configuration->payload.end());

    tmv::checkpoint::wire::Encoder added_v3_scalars;
    added_v3_scalars.floating(7.0e-4);
    added_v3_scalars.floating(6.0e-4);
    const auto added = std::search(configuration->payload.begin(),
                                   configuration->payload.end(),
                                   added_v3_scalars.bytes.begin(),
                                   added_v3_scalars.bytes.end());
    REQUIRE(added != configuration->payload.end());
    configuration->payload.erase(added, added + static_cast<std::ptrdiff_t>(added_v3_scalars.bytes.size()));
    configuration->version = 2;

    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    CHECK(restored.configuration.microphysics_config.thompson.evaporation_rate_s_inv == Approx(1.0e-3));
    CHECK(restored.configuration.microphysics_config.thompson.water_friendly_aerosol_surface_m3 == Approx(350.0e6));
    CHECK(restored.configuration.microphysics_config.thompson.ice_friendly_aerosol_surface_m3 == Approx(2.0e6));
    CHECK(restored.configuration.microphysics_config.milbrandt.evaporation_rate_s_inv == Approx(1.0e-3));
    CHECK_FALSE(restored.configuration.microphysics_config.milbrandt.triple_moment);
    CHECK(restored.configuration.storm_tracking.minimum_core_area_m2 == Approx(1.0e6));
}

TEST_CASE("simulation checkpoint runtime v29 starts storm continuity history cleanly",
          "[core][checkpoint][compatibility][storm_tracker]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                      source = sample_state();
    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));

    const auto runtime =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    strip_chaos_pbl_cache_v37(*runtime, source);
    REQUIRE(runtime->version == 36);
    tmv::checkpoint::wire::Encoder added_v35_provenance;
    added_v35_provenance.boolean(false);
    REQUIRE(runtime->payload.size() >= added_v35_provenance.bytes.size());
    const auto v35_suffix = runtime->payload.end() - static_cast<std::ptrdiff_t>(added_v35_provenance.bytes.size());
    REQUIRE(std::equal(
        v35_suffix, runtime->payload.end(), added_v35_provenance.bytes.begin(), added_v35_provenance.bytes.end()));
    runtime->payload.erase(v35_suffix, runtime->payload.end());
    runtime->version = 34;

    const auto added_v34_movement = movement_authority_suffix(source);
    REQUIRE(runtime->payload.size() >= added_v34_movement.bytes.size());
    const auto v34_suffix = runtime->payload.end() - static_cast<std::ptrdiff_t>(added_v34_movement.bytes.size());
    REQUIRE(std::equal(
        v34_suffix, runtime->payload.end(), added_v34_movement.bytes.begin(), added_v34_movement.bytes.end()));
    runtime->payload.erase(v34_suffix, runtime->payload.end());
    runtime->version = 33;

    const auto&                    storm = source.runtime.storm_tracking;
    tmv::checkpoint::wire::Encoder added_v33_partition;
    added_v33_partition.boolean(source.runtime.storm_selected_partition_available);
    added_v33_partition.unsigned_value<std::uint64_t>(source.runtime.storm_selected_partition_nx);
    added_v33_partition.unsigned_value<std::uint64_t>(source.runtime.storm_selected_partition_ny);
    added_v33_partition.unsigned_value<std::uint64_t>(source.runtime.storm_selected_partition.size());
    added_v33_partition.raw(source.runtime.storm_selected_partition);
    REQUIRE(runtime->payload.size() >= added_v33_partition.bytes.size());
    const auto v33_suffix = runtime->payload.end() - static_cast<std::ptrdiff_t>(added_v33_partition.bytes.size());
    REQUIRE(std::equal(
        v33_suffix, runtime->payload.end(), added_v33_partition.bytes.begin(), added_v33_partition.bytes.end()));
    runtime->payload.erase(v33_suffix, runtime->payload.end());
    runtime->version = 32;

    tmv::checkpoint::wire::Encoder added_v32_overlap;
    added_v32_overlap.boolean(storm.association_envelope_available);
    added_v32_overlap.floating(storm.association_envelope_west_m);
    added_v32_overlap.floating(storm.association_envelope_east_m);
    added_v32_overlap.floating(storm.association_envelope_south_m);
    added_v32_overlap.floating(storm.association_envelope_north_m);
    added_v32_overlap.boolean(storm.topology_reorganization);
    added_v32_overlap.floating(storm.reorganization_overlap_fraction);
    REQUIRE(runtime->payload.size() >= added_v32_overlap.bytes.size());
    const auto v32_suffix = runtime->payload.end() - static_cast<std::ptrdiff_t>(added_v32_overlap.bytes.size());
    REQUIRE(
        std::equal(v32_suffix, runtime->payload.end(), added_v32_overlap.bytes.begin(), added_v32_overlap.bytes.end()));
    runtime->payload.erase(v32_suffix, runtime->payload.end());
    runtime->version = 31;

    tmv::checkpoint::wire::Encoder added_v31_branches;
    added_v31_branches.integer(static_cast<std::int32_t>(storm.topology));
    added_v31_branches.integer(static_cast<std::int32_t>(storm.pending_topology));
    added_v31_branches.integer(static_cast<std::int32_t>(storm.topology_event));
    added_v31_branches.unsigned_value<std::uint64_t>(storm.pending_topology_samples);
    added_v31_branches.unsigned_value(storm.topology_event_id);
    added_v31_branches.boolean(storm.selected_candidate_available);
    added_v31_branches.unsigned_value<std::uint64_t>(storm.selected_candidate_rank);
    added_v31_branches.unsigned_value<std::uint64_t>(storm.selected_component_label);
    added_v31_branches.integer(static_cast<std::int32_t>(storm.selected_motion_class));
    added_v31_branches.floating(storm.selected_bunkers_right_mismatch_ms);
    added_v31_branches.floating(storm.selected_bunkers_left_mismatch_ms);
    REQUIRE(runtime->payload.size() >= added_v31_branches.bytes.size());
    const auto v31_suffix = runtime->payload.end() - static_cast<std::ptrdiff_t>(added_v31_branches.bytes.size());
    REQUIRE(std::equal(
        v31_suffix, runtime->payload.end(), added_v31_branches.bytes.begin(), added_v31_branches.bytes.end()));
    runtime->payload.erase(v31_suffix, runtime->payload.end());
    runtime->version = 30;

    tmv::checkpoint::wire::Encoder added_v30_continuity;
    added_v30_continuity.integer(static_cast<std::int32_t>(storm.status));
    added_v30_continuity.integer(static_cast<std::int32_t>(storm.reason));
    added_v30_continuity.unsigned_value<std::uint64_t>(storm.sample_count);
    for (std::size_t index = 0; index < storm.sample_count; ++index)
    {
        added_v30_continuity.floating(storm.samples[index].time_s);
        added_v30_continuity.floating(storm.samples[index].x_m);
        added_v30_continuity.floating(storm.samples[index].y_m);
        added_v30_continuity.floating(storm.samples[index].confidence);
    }
    added_v30_continuity.unsigned_value<std::uint64_t>(storm.missed_samples);
    added_v30_continuity.unsigned_value(storm.branch_id);
    added_v30_continuity.boolean(storm.motion_available);
    added_v30_continuity.floating(storm.velocity_x_ms);
    added_v30_continuity.floating(storm.velocity_y_ms);
    added_v30_continuity.floating(storm.speed_ms);
    added_v30_continuity.floating(storm.velocity_uncertainty_ms);
    added_v30_continuity.floating(storm.continuity_confidence);
    added_v30_continuity.floating(storm.innovation_distance_m);
    added_v30_continuity.floating(storm.association_gate_radius_m);

    REQUIRE(runtime->payload.size() >= added_v30_continuity.bytes.size());
    const auto suffix = runtime->payload.end() - static_cast<std::ptrdiff_t>(added_v30_continuity.bytes.size());
    REQUIRE(std::equal(
        suffix, runtime->payload.end(), added_v30_continuity.bytes.begin(), added_v30_continuity.bytes.end()));
    runtime->payload.erase(suffix, runtime->payload.end());
    runtime->version = 29;

    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    CHECK(restored.runtime.storm_tracking.status == tmv::diagnostics::StormContinuityStatus::Uninitialized);
    CHECK(restored.runtime.storm_tracking.sample_count == 0);
    CHECK(restored.runtime.storm_tracking.branch_id == 0);
    CHECK_FALSE(restored.runtime.storm_tracking.motion_available);
}

TEST_CASE("simulation checkpoint runtime v31 restores continuity without overlap association state",
          "[core][checkpoint][compatibility][storm_tracker][overlap]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                      source = sample_state();
    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));

    const auto runtime =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    strip_chaos_pbl_cache_v37(*runtime, source);
    REQUIRE(runtime->version == 36);
    tmv::checkpoint::wire::Encoder provenance_suffix;
    provenance_suffix.boolean(false);
    REQUIRE(runtime->payload.size() >= provenance_suffix.bytes.size());
    const auto provenance_start = runtime->payload.end() - static_cast<std::ptrdiff_t>(provenance_suffix.bytes.size());
    REQUIRE(std::equal(
        provenance_start, runtime->payload.end(), provenance_suffix.bytes.begin(), provenance_suffix.bytes.end()));
    runtime->payload.erase(provenance_start, runtime->payload.end());
    runtime->version           = 34;
    const auto movement_suffix = movement_authority_suffix(source);
    REQUIRE(runtime->payload.size() >= movement_suffix.bytes.size());
    const auto movement_start = runtime->payload.end() - static_cast<std::ptrdiff_t>(movement_suffix.bytes.size());
    REQUIRE(
        std::equal(movement_start, runtime->payload.end(), movement_suffix.bytes.begin(), movement_suffix.bytes.end()));
    runtime->payload.erase(movement_start, runtime->payload.end());
    runtime->version = 33;
    tmv::checkpoint::wire::Encoder partition_suffix;
    partition_suffix.boolean(source.runtime.storm_selected_partition_available);
    partition_suffix.unsigned_value<std::uint64_t>(source.runtime.storm_selected_partition_nx);
    partition_suffix.unsigned_value<std::uint64_t>(source.runtime.storm_selected_partition_ny);
    partition_suffix.unsigned_value<std::uint64_t>(source.runtime.storm_selected_partition.size());
    partition_suffix.raw(source.runtime.storm_selected_partition);
    REQUIRE(runtime->payload.size() >= partition_suffix.bytes.size());
    const auto partition_start = runtime->payload.end() - static_cast<std::ptrdiff_t>(partition_suffix.bytes.size());
    REQUIRE(std::equal(
        partition_start, runtime->payload.end(), partition_suffix.bytes.begin(), partition_suffix.bytes.end()));
    runtime->payload.erase(partition_start, runtime->payload.end());
    runtime->version                     = 32;
    const auto&                    storm = source.runtime.storm_tracking;
    tmv::checkpoint::wire::Encoder suffix;
    suffix.boolean(storm.association_envelope_available);
    suffix.floating(storm.association_envelope_west_m);
    suffix.floating(storm.association_envelope_east_m);
    suffix.floating(storm.association_envelope_south_m);
    suffix.floating(storm.association_envelope_north_m);
    suffix.boolean(storm.topology_reorganization);
    suffix.floating(storm.reorganization_overlap_fraction);
    REQUIRE(runtime->payload.size() >= suffix.bytes.size());
    const auto start = runtime->payload.end() - static_cast<std::ptrdiff_t>(suffix.bytes.size());
    REQUIRE(std::equal(start, runtime->payload.end(), suffix.bytes.begin(), suffix.bytes.end()));
    runtime->payload.erase(start, runtime->payload.end());
    runtime->version = 31;

    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    CHECK(restored.runtime.storm_tracking.status == source.runtime.storm_tracking.status);
    CHECK(restored.runtime.storm_tracking.branch_id == source.runtime.storm_tracking.branch_id);
    CHECK_FALSE(restored.runtime.storm_tracking.association_envelope_available);
    CHECK(restored.runtime.storm_tracking.association_envelope_west_m == 0.0);
    CHECK(restored.runtime.storm_tracking.association_envelope_east_m == 0.0);
    CHECK(restored.runtime.storm_tracking.association_envelope_south_m == 0.0);
    CHECK(restored.runtime.storm_tracking.association_envelope_north_m == 0.0);
    CHECK_FALSE(restored.runtime.storm_tracking.topology_reorganization);
    CHECK(restored.runtime.storm_tracking.reorganization_overlap_fraction == 0.0);
}

TEST_CASE("simulation checkpoint configuration v6 restores declared UH defaults",
          "[core][checkpoint][compatibility][storm_tracker][rotation]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                      source = sample_state();
    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));

    const auto configuration =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 2; });
    REQUIRE(configuration != sections.end());
    downgrade_configuration_v16_to_v14(*configuration, source);
    REQUIRE(configuration->version == 14);
    tmv::checkpoint::wire::Encoder added_v14_execution_mode;
    added_v14_execution_mode.integer(static_cast<std::int32_t>(source.configuration.execution_mode));
    REQUIRE(configuration->payload.size() >= added_v14_execution_mode.bytes.size());
    const auto v14_suffix =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v14_execution_mode.bytes.size());
    REQUIRE(std::equal(v14_suffix,
                       configuration->payload.end(),
                       added_v14_execution_mode.bytes.begin(),
                       added_v14_execution_mode.bytes.end()));
    configuration->payload.erase(v14_suffix, configuration->payload.end());
    configuration->version = 13;
    tmv::checkpoint::wire::Encoder added_v13_nesting_selection;
    added_v13_nesting_selection.boolean(source.configuration.nesting.enabled);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_origin_parent_cell_i);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_origin_parent_cell_j);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_cells_x);
    added_v13_nesting_selection.integer(source.configuration.nesting.child_cells_y);
    added_v13_nesting_selection.integer(source.configuration.nesting.relaxation_width_child_cells);
    added_v13_nesting_selection.integer(source.configuration.nesting.boundary_stencil_parent_cells);
    added_v13_nesting_selection.floating(source.configuration.nesting.relaxation_e_folding_time_s);
    REQUIRE(configuration->payload.size() >= added_v13_nesting_selection.bytes.size());
    const auto v13_suffix =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v13_nesting_selection.bytes.size());
    REQUIRE(std::equal(v13_suffix,
                       configuration->payload.end(),
                       added_v13_nesting_selection.bytes.begin(),
                       added_v13_nesting_selection.bytes.end()));
    configuration->payload.erase(v13_suffix, configuration->payload.end());
    configuration->version = 12;
    tmv::checkpoint::wire::Encoder added_v12_nesting;
    added_v12_nesting.integer(source.configuration.nesting.parent_ratio_x);
    added_v12_nesting.integer(source.configuration.nesting.parent_ratio_y);
    added_v12_nesting.integer(source.configuration.nesting.parent_time_step_ratio);
    added_v12_nesting.integer(static_cast<std::int32_t>(source.configuration.nesting.feedback));
    REQUIRE(configuration->payload.size() >= added_v12_nesting.bytes.size());
    const auto v12_suffix = configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v12_nesting.bytes.size());
    REQUIRE(std::equal(
        v12_suffix, configuration->payload.end(), added_v12_nesting.bytes.begin(), added_v12_nesting.bytes.end()));
    configuration->payload.erase(v12_suffix, configuration->payload.end());
    configuration->version = 11;
    tmv::checkpoint::wire::Encoder added_v11_recycling;
    added_v11_recycling.boolean(source.configuration.storm_tracking.recycle_domain);
    REQUIRE(configuration->payload.size() >= added_v11_recycling.bytes.size());
    const auto v11_suffix =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v11_recycling.bytes.size());
    REQUIRE(std::equal(
        v11_suffix, configuration->payload.end(), added_v11_recycling.bytes.begin(), added_v11_recycling.bytes.end()));
    configuration->payload.erase(v11_suffix, configuration->payload.end());
    configuration->version = 10;
    tmv::checkpoint::wire::Encoder added_v10_core_area;
    added_v10_core_area.floating(source.configuration.storm_tracking.minimum_core_area_m2);
    REQUIRE(configuration->payload.size() >= added_v10_core_area.bytes.size());
    const auto v10_suffix =
        configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v10_core_area.bytes.size());
    REQUIRE(std::equal(
        v10_suffix, configuration->payload.end(), added_v10_core_area.bytes.begin(), added_v10_core_area.bytes.end()));
    configuration->payload.erase(v10_suffix, configuration->payload.end());
    configuration->version = 9;
    tmv::checkpoint::wire::Encoder added_v9_frame;
    added_v9_frame.integer(static_cast<std::int32_t>(source.configuration.domain_motion_mode));
    added_v9_frame.floating(source.configuration.domain_frame.earth_origin_at_t0_m.x_m);
    added_v9_frame.floating(source.configuration.domain_frame.earth_origin_at_t0_m.y_m);
    added_v9_frame.floating(source.configuration.domain_frame.velocity_ms.u_ms);
    added_v9_frame.floating(source.configuration.domain_frame.velocity_ms.v_ms);
    REQUIRE(configuration->payload.size() >= added_v9_frame.bytes.size());
    const auto v9_suffix = configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v9_frame.bytes.size());
    REQUIRE(
        std::equal(v9_suffix, configuration->payload.end(), added_v9_frame.bytes.begin(), added_v9_frame.bytes.end()));
    configuration->payload.erase(v9_suffix, configuration->payload.end());
    const auto&                    tracking = source.configuration.storm_tracking;
    tmv::checkpoint::wire::Encoder added_v8_branches;
    added_v8_branches.integer(tracking.topology_confirmation_samples);
    added_v8_branches.integer(static_cast<std::int32_t>(tracking.branch_preference));
    added_v8_branches.floating(tracking.branch_velocity_uncertainty_ms);
    REQUIRE(configuration->payload.size() >= added_v8_branches.bytes.size());
    const auto v8_suffix = configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v8_branches.bytes.size());
    REQUIRE(std::equal(
        v8_suffix, configuration->payload.end(), added_v8_branches.bytes.begin(), added_v8_branches.bytes.end()));
    configuration->payload.erase(v8_suffix, configuration->payload.end());
    configuration->version = 7;

    tmv::checkpoint::wire::Encoder added_v7_rotation;
    added_v7_rotation.floating(tracking.updraft_helicity_bottom_m_agl);
    added_v7_rotation.floating(tracking.updraft_helicity_top_m_agl);
    added_v7_rotation.floating(tracking.minimum_absolute_updraft_helicity_m2s2);
    REQUIRE(configuration->payload.size() >= added_v7_rotation.bytes.size());
    const auto suffix = configuration->payload.end() - static_cast<std::ptrdiff_t>(added_v7_rotation.bytes.size());
    REQUIRE(std::equal(
        suffix, configuration->payload.end(), added_v7_rotation.bytes.begin(), added_v7_rotation.bytes.end()));
    configuration->payload.erase(suffix, configuration->payload.end());
    configuration->version = 6;

    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    CHECK(restored.configuration.storm_tracking.updraft_helicity_bottom_m_agl == Approx(2000.0));
    CHECK(restored.configuration.storm_tracking.updraft_helicity_top_m_agl == Approx(5000.0));
    CHECK(restored.configuration.storm_tracking.minimum_absolute_updraft_helicity_m2s2 == Approx(25.0));
    CHECK(restored.configuration.storm_tracking.topology_confirmation_samples == 2);
    CHECK(restored.configuration.storm_tracking.branch_preference ==
          tmv::init::config::StormBranchPreference::Continuity);
    CHECK(restored.configuration.storm_tracking.branch_velocity_uncertainty_ms == Approx(5.0));
}

TEST_CASE("simulation checkpoint rejects obsolete active Milbrandt controls",
          "[core][checkpoint][microphysics][milbrandt]")
{
    SimulationStorageState source                                    = sample_state();
    source.configuration.microphysics_config.milbrandt.triple_moment = true;
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(bytes.empty());
    REQUIRE(error.find("obsolete triple_moment") != std::string::npos);
}

TEST_CASE("simulation checkpoint rejects invalid Thompson aerosol profiles",
          "[core][checkpoint][microphysics][thompson]")
{
    SimulationStorageState source                                                       = sample_state();
    source.configuration.microphysics_config.thompson.water_friendly_aerosol_surface_m3 = 50.0e6;
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(bytes.empty());
    REQUIRE(error.find("invalid Thompson aerosol profile") != std::string::npos);

    source = sample_state();
    source.configuration.microphysics_config.thompson.ice_friendly_aerosol_surface_m3 =
        std::numeric_limits<double>::infinity();
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(bytes.empty());
}

TEST_CASE("simulation checkpoint rejects contradictory storm continuity claims",
          "[core][checkpoint][storm_tracker][validation]")
{
    SimulationStorageState source                                  = sample_state();
    source.configuration.storm_tracking.minimum_continuity_samples = 5;
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(bytes.empty());
    CHECK(error.find("conflicts with its configured thresholds") != std::string::npos);

    source                               = sample_state();
    source.runtime.storm_tracking.status = static_cast<tmv::diagnostics::StormContinuityStatus>(99);
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("invalid storm continuity state") != std::string::npos);

    source                                 = sample_state();
    source.runtime.storm_tracking.topology = static_cast<tmv::diagnostics::StormBranchTopology>(99);
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("invalid storm continuity state") != std::string::npos);
}

TEST_CASE("simulation checkpoint rejects non-finite acoustic energy attribution",
          "[core][checkpoint][acoustic-energy][SCI-141]")
{
    SimulationStorageState source = sample_state();
    source.runtime.time_stepping.last_acoustic_energy.pressure_gradient_committed_work_j =
        std::numeric_limits<double>::infinity();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source = sample_state();
    source.runtime.time_stepping.last_acoustic_energy.kinetic_transport_closure_residual_j =
        std::numeric_limits<double>::infinity();
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects non-finite slow momentum attribution",
          "[core][checkpoint][slow-momentum-energy][SCI-149]")
{
    SimulationStorageState source = sample_state();
    source.runtime.time_stepping.last_slow_momentum_energy.advection_requested_linear_work_j =
        std::numeric_limits<double>::infinity();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects inconsistent slow pressure energy",
          "[core][checkpoint][slow-pressure-energy][SCI-157]")
{
    SimulationStorageState source = sample_state();
    source.runtime.time_stepping.last_slow_momentum_energy.committed_pressure_energy_change_j += 1.0;
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects invalid microphysics moist energy",
          "[core][checkpoint][microphysics-energy][SCI-151]")
{
    SimulationStorageState source = sample_state();
    source.runtime.microphysics.last_thermodynamic_repair_report.moist_committed_energy_change_j =
        std::numeric_limits<double>::infinity();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source                                                                             = sample_state();
    source.runtime.microphysics.last_thermodynamic_repair_report.moist_energy_complete = false;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects invalid SGS moist energy", "[core][checkpoint][sgs-energy][SCI-152]")
{
    SimulationStorageState source = sample_state();
    source.runtime.turbulence.last_momentum_work.moist_carrier_kinetic_change_j =
        std::numeric_limits<double>::infinity();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source                                                                       = sample_state();
    source.runtime.turbulence.cumulative_moist_energy_work.moist_energy_complete = false;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects invalid diffusion moist energy",
          "[core][checkpoint][diffusion-energy][SCI-153]")
{
    SimulationStorageState source = sample_state();
    source.runtime.diffusion.last_momentum_work.moist_water_sensible_change_j =
        std::numeric_limits<double>::quiet_NaN();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source                                                                      = sample_state();
    source.runtime.diffusion.cumulative_moist_energy_work.moist_energy_complete = false;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects invalid EOS pressure energy", "[core][checkpoint][eos-energy][SCI-154]")
{
    SimulationStorageState source = sample_state();
    source.runtime.time_stepping.last_eos_pressure_energy.dry_internal_change_j =
        std::numeric_limits<double>::infinity();
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source                                                               = sample_state();
    source.runtime.time_stepping.cumulative_eos_pressure_energy.complete = false;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects invalid sponge energy", "[core][checkpoint][sponge-energy][SCI-155]")
{
    SimulationStorageState source                                                     = sample_state();
    source.runtime.time_stepping.last_sponge_energy.upper.momentum_storage_rounding_j = 42.0;
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source                                                         = sample_state();
    source.runtime.time_stepping.cumulative_sponge_energy.complete = false;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects invalid final boundary energy", "[core][checkpoint][boundary-energy][SCI-156]")
{
    SimulationStorageState source                                                       = sample_state();
    source.runtime.time_stepping.last_final_boundary_energy.constraint_committed_work_j = 42.0;
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source                                                                 = sample_state();
    source.runtime.time_stepping.cumulative_final_boundary_energy.complete = false;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint rejects invalid completed-step energy",
          "[core][checkpoint][completed-energy][SCI-157]")
{
    SimulationStorageState source                                                 = sample_state();
    source.runtime.time_stepping.last_completed_step_energy.committed_owner_sum_j = 42.0;
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));

    source                                                                 = sample_state();
    source.runtime.time_stepping.cumulative_completed_step_energy.complete = false;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("simulation checkpoint compatibility is exact but backend portable", "[core][checkpoint][compatibility]")
{
    SimulationStorageState requested  = sample_state();
    SimulationStorageState checkpoint = sample_state();
    std::string            error;

    checkpoint.runtime.compute.config.backend      = "cpu";
    checkpoint.runtime.compute.config.device_index = -1;
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));

    checkpoint                                      = sample_state();
    checkpoint.configuration.gpu_apply_microphysics = !requested.configuration.gpu_apply_microphysics;
    checkpoint.configuration.gpu_apply_slow         = !requested.configuration.gpu_apply_slow;
    checkpoint.configuration.gpu_apply_sgs          = !requested.configuration.gpu_apply_sgs;
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));

    checkpoint.configuration.microphysics_scheme = "lin";
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("scientific configuration") != std::string::npos);

    checkpoint = sample_state();
    checkpoint.configuration.sounding_config.use_fallback_profiles =
        !requested.configuration.sounding_config.use_fallback_profiles;
    checkpoint.configuration.sounding_source.file.path = "inactive.snd";
    checkpoint.configuration.sounding_source.targets.target_cape_jkg += 500.0;
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(
        requested, checkpoint, error));
    requested.configuration.admission.generated_metal = true;
    requested.configuration.admission.generated_metal_mixed_npy = true;
    requested.configuration.admission.generated_metal_east_npy = false;
    checkpoint.configuration.admission.generated_metal_mixed_npy = false;
    checkpoint.configuration.admission.generated_metal_east_npy = true;
    tmv::checkpoint::adopt_checkpoint_branch_configuration(
        requested.configuration, checkpoint.configuration);
    REQUIRE(checkpoint.configuration.admission.generated_metal);
    REQUIRE(checkpoint.configuration.admission.generated_metal_mixed_npy);
    REQUIRE_FALSE(checkpoint.configuration.admission.generated_metal_east_npy);
    REQUIRE(checkpoint.configuration.execution_mode ==
            requested.configuration.execution_mode);
    REQUIRE(checkpoint.configuration.sounding_config.use_fallback_profiles ==
            requested.configuration.sounding_config.use_fallback_profiles);
    REQUIRE(checkpoint.configuration.sounding_source.file.path ==
            requested.configuration.sounding_source.file.path);

    checkpoint = sample_state();
    requested.configuration.sounding_source.type =
        tmv::init::SoundingSourceConfig::Type::File;
    checkpoint.configuration.sounding_source.type =
        tmv::init::SoundingSourceConfig::Type::File;
    checkpoint.configuration.sounding_source.file.path = "different.snd";
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(
        requested, checkpoint, error));
    requested = sample_state();

    checkpoint = sample_state();
    checkpoint.configuration.microphysics_config.shared.autoconversion_rate_s_inv *= 2.0;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));

    checkpoint = sample_state();
    checkpoint.runtime.radiation.config.rrtmg_ozone_vmr += 1.0e-6;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("radiation configuration") != std::string::npos);

    checkpoint                                                 = sample_state();
    checkpoint.runtime.radar.product_selection.radial_velocity = true;
    checkpoint.runtime.radar.product_selection.radar_x         = -1500.0;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("radar configuration") != std::string::npos);

    checkpoint                                             = sample_state();
    checkpoint.runtime.radar.selected_config.scheme_id     = "reflectivity";
    checkpoint.runtime.radar.selected_config.operator_tier = "fast_da";
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));

    checkpoint = sample_state();
    checkpoint.runtime.radar.product_selection.apply_scatterer_correction =
        !requested.runtime.radar.product_selection.apply_scatterer_correction;
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));

    checkpoint                                              = sample_state();
    requested.runtime.radar.product_selection.reflectivity  = true;
    checkpoint.runtime.radar.product_selection.reflectivity = true;
    checkpoint.runtime.radar.product_selection.apply_scatterer_correction =
        !requested.runtime.radar.product_selection.apply_scatterer_correction;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("apply_scatterer_correction") != std::string::npos);

    requested          = sample_state();
    checkpoint         = sample_state();
    checkpoint.grid.dt = 0.5;
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));

    checkpoint                                     = sample_state();
    checkpoint.configuration.configured_dt_large_s = 0.5;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("scientific configuration") != std::string::npos);

    checkpoint = sample_state();
    checkpoint.configuration.domain_frame.velocity_ms.u_ms += 0.5;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("scientific configuration") != std::string::npos);

    checkpoint = sample_state();
    checkpoint.configuration.domain_frame.earth_origin_at_t0_m.x_m += 125.0;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("scientific configuration") != std::string::npos);

    checkpoint         = sample_state();
    checkpoint.grid.dx = 250.0;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, checkpoint, error));
    CHECK(error.find("grid/timestep") != std::string::npos);
}

TEST_CASE("checkpoint execution policy is exact, backward compatible, and validated",
          "[core][checkpoint][compatibility][execution]")
{
    using tmv::init::config::ExecutionMode;
    constexpr std::array<std::uint8_t, 8> magic       = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                requested   = sample_state();
    SimulationStorageState                exploratory = sample_state();
    exploratory.configuration.execution_mode          = ExecutionMode::Exploratory;
    std::string error;
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(requested, exploratory, error));

    requested.configuration.execution_mode = ExecutionMode::Exploratory;
    exploratory.configuration.execution_mode = ExecutionMode::Qualified;
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(
        requested, exploratory, error));
    REQUIRE(error.find("scientific configuration") != std::string::npos);
    requested.configuration.execution_mode = ExecutionMode::Qualified;

    std::vector<std::uint8_t> bytes;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(requested, bytes, error));
    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    const auto configuration =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 2; });
    REQUIRE(configuration != sections.end());
    downgrade_configuration_v16_to_v14(*configuration, requested);
    REQUIRE(configuration->version == 14);

    tmv::checkpoint::wire::Encoder qualified_suffix;
    qualified_suffix.integer(static_cast<std::int32_t>(ExecutionMode::Qualified));
    REQUIRE(configuration->payload.size() >= qualified_suffix.bytes.size());
    const auto suffix = configuration->payload.end() - static_cast<std::ptrdiff_t>(qualified_suffix.bytes.size());
    REQUIRE(
        std::equal(suffix, configuration->payload.end(), qualified_suffix.bytes.begin(), qualified_suffix.bytes.end()));
    configuration->payload.erase(suffix, configuration->payload.end());
    configuration->version = 13;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.configuration.execution_mode == ExecutionMode::Exploratory);
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, restored, error));

    REQUIRE(tmv::checkpoint::serialize_simulation_storage(requested, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    const auto corrupt_configuration =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 2; });
    REQUIRE(corrupt_configuration != sections.end());
    tmv::checkpoint::wire::Encoder corrupt_suffix;
    corrupt_suffix.floating(std::numeric_limits<double>::quiet_NaN());
    REQUIRE(corrupt_configuration->payload.size() >= corrupt_suffix.bytes.size());
    std::copy(corrupt_suffix.bytes.begin(),
              corrupt_suffix.bytes.end(),
              corrupt_configuration->payload.end() - static_cast<std::ptrdiff_t>(corrupt_suffix.bytes.size()));
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
}

TEST_CASE("D5 transaction provenance is restart-exact and alone authorizes a mutable Earth origin",
          "[core][checkpoint][domain_recycling][SCI-227]")
{
    SimulationStorageState source = sample_state();
    seed_committed_domain_recycling_provenance(source);
    std::vector<std::uint8_t> bytes;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    const auto& actual = restored.runtime.domain_recycling_provenance;
    REQUIRE(actual.available);
    CHECK(actual.sequence == 7);
    CHECK(actual.transaction_id.view() == "recycle_000007");
    CHECK(actual.publication_generation.view() == "run_generation_4");
    CHECK(actual.storage_format.view() == "npy_3d");
    CHECK(actual.event_manifest_bytes == 4096);
    CHECK(actual.event_manifest_fnv1a64 == UINT64_C(0x123456789abcdef0));
    CHECK(actual.previous_chain_fnv1a64 == UINT64_C(0x0f0e0d0c0b0a0908));
    CHECK(actual.committed_chain_fnv1a64 == source.runtime.domain_recycling_provenance.committed_chain_fnv1a64);
    CHECK(actual.pre_shift_origin_t0_m.x_m == 124875.0);
    CHECK(actual.post_shift_origin_t0_m.x_m == 125000.0);
    CHECK(actual.shift_i_cells == 1);
    CHECK(actual.shift_j_cells == 0);
    CHECK(actual.physical_columns == 2);
    CHECK(actual.physical_cells == 4);
    CHECK(actual.swath_algorithm.view() == "canonical_rectangular_earth_swath_capture_v1");
    CHECK(actual.control_volume_exchange_complete);
    CHECK(actual.admitted_inventories.total_mass == 10.0);
    CHECK(actual.admitted_momentum.reference_angular == 1250.0);
    CHECK(actual.retiring_fixed_earth_angular_momentum_z == 0.0);
    CHECK(actual.admitted_fixed_earth_angular_momentum_z == 0.0);
    CHECK(actual.control_volume_exchange_basis.view() == "instantaneous_fixed_earth_control_volume_exchange_v1");

    SimulationStorageState requested                          = sample_state();
    requested.configuration.domain_frame.earth_origin_at_t0_m = actual.pre_shift_origin_t0_m;
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(requested, restored, error));

    restored.runtime.domain_recycling_provenance.post_shift_origin_t0_m.x_m += 1.0;
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(requested, restored, error));
    CHECK(error.find("invalid D5 frame provenance") != std::string::npos);
}

TEST_CASE("Private recycling inventories survive restart and reject incomplete ownership",
          "[core][checkpoint][domain_recycling][private-inventory][v55]")
{
    auto source = sample_state();
    seed_committed_domain_recycling_provenance(source);
    auto& provenance                      = source.runtime.domain_recycling_provenance;
    provenance.private_inventory_complete = true;
    const auto fields                     = source.runtime.microphysics.scheme->prognostic_scalars_for_transport();
    provenance.retiring_private_inventory.count = static_cast<std::uint16_t>(fields.size());
    provenance.admitted_private_inventory.count = static_cast<std::uint16_t>(fields.size());
    for (std::size_t f = 0; f < fields.size(); ++f)
    {
        auto& retiring = provenance.retiring_private_inventory.entries[f];
        auto& admitted = provenance.admitted_private_inventory.entries[f];
        set_provenance_text(retiring.name, fields[f].name);
        admitted.name      = retiring.name;
        retiring.inventory = 1.25e16 * (f + 1);
        admitted.inventory = 2.5e15 * (f + 1);
    }
    std::vector<std::uint8_t> bytes, continued;
    std::string               error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    const auto& actual = restored.runtime.domain_recycling_provenance;
    CHECK(actual.private_inventory_complete);
    REQUIRE(actual.retiring_private_inventory.count == fields.size());
    for (std::size_t f = 0; f < fields.size(); ++f)
    {
        CHECK(actual.retiring_private_inventory.entries[f].name.view() == fields[f].name);
        CHECK(actual.retiring_private_inventory.entries[f].inventory ==
              provenance.retiring_private_inventory.entries[f].inventory);
        CHECK(actual.admitted_private_inventory.entries[f].inventory ==
              provenance.admitted_private_inventory.entries[f].inventory);
    }
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored, continued, error));
    CHECK(bytes == continued);
    const auto reject = [&](auto corrupt)
    {
        auto invalid = source;
        corrupt(invalid.runtime.domain_recycling_provenance);
        std::vector<std::uint8_t> output;
        CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(invalid, output, error));
    };
    reject([](auto& p) { p.retiring_private_inventory.count = 17; });
    reject([](auto& p) { p.admitted_private_inventory.count = 0; });
    reject([](auto& p) { p.private_inventory_complete = false; });
    reject([](auto& p) { p.retiring_private_inventory.entries[0].inventory = -1; });
    reject([](auto& p)
           { p.admitted_private_inventory.entries[0].inventory = std::numeric_limits<double>::infinity(); });
    reject([](auto& p)
           { p.admitted_private_inventory.entries[0].name = p.admitted_private_inventory.entries[1].name; });
    reject(
        [](auto& p)
        {
            p.retiring_private_inventory.entries[0].name.bytes[0] = 'x';
            p.admitted_private_inventory.entries[0].name          = p.retiring_private_inventory.entries[0].name;
        });
}

TEST_CASE("checkpoint rejects D5 provenance inconsistent with live state",
          "[core][checkpoint][domain_recycling][SCI-227]")
{
    SimulationStorageState source = sample_state();
    seed_committed_domain_recycling_provenance(source);
    std::vector<std::uint8_t> bytes;
    std::string               error;

    source.configuration.domain_frame.earth_origin_at_t0_m.x_m += 125.0;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("inconsistent with live grid/frame/time") != std::string::npos);

    source = sample_state();
    seed_committed_domain_recycling_provenance(source);
    source.runtime.domain_recycling_provenance.event_manifest_bytes = 0;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("invalid D5 transaction provenance") != std::string::npos);

    source = sample_state();
    seed_committed_domain_recycling_provenance(source);
    source.runtime.domain_recycling_provenance.committed_chain_fnv1a64 ^= 1U;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("invalid D5 transaction provenance") != std::string::npos);

    source = sample_state();
    seed_committed_domain_recycling_provenance(source);
    source.runtime.domain_recycling_provenance.admitted_inventories.moist_total_mass += 1.0;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("invalid D5 transaction provenance") != std::string::npos);
}

TEST_CASE("checkpoint rejects incomplete active child state", "[core][checkpoint][nesting][transaction]")
{
    SimulationStorageState    source   = sample_state();
    std::vector<std::uint8_t> bytes    = {1, 2, 3, 4};
    const auto                original = bytes;
    std::string               error;

    source.nested_execution.child_execution_active = true;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error.find("nested child is process-bound") != std::string::npos);
    REQUIRE(bytes == original);

    source.nested_execution.child_execution_active = false;
    source.nested_execution.relationship_ready = true;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error.find("complete child and boundary state") != std::string::npos);
    REQUIRE(bytes == original);
}

TEST_CASE("synchronized active nesting checkpoint round-trips exactly",
          "[core][checkpoint][nesting][synchronized][roundtrip]")
{
    constexpr std::array<std::uint8_t, 8> magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source = synchronized_nested_state();
    source.configuration.nesting.feedback        = tmv::init::config::NestingFeedback::TwoWay;
    for (Field3D* field : {&source.fields.qv,
                           &source.fields.qc,
                           &source.fields.qr,
                           &source.fields.qi,
                           &source.fields.qs,
                           &source.fields.qg,
                           &source.fields.qh,
                           &source.fields.tracer,
                           &source.fields.tke,
                           &source.nested_execution.child.fields.qv,
                           &source.nested_execution.child.fields.qc,
                           &source.nested_execution.child.fields.qr,
                           &source.nested_execution.child.fields.qi,
                           &source.nested_execution.child.fields.qs,
                           &source.nested_execution.child.fields.qg,
                           &source.nested_execution.child.fields.qh,
                           &source.nested_execution.child.fields.tracer,
                           &source.nested_execution.child.fields.tke})
        field->fill(0.0f);
    source.nested_execution.two_way_interval_source_begin  = source.nested_execution.interface_source.cumulative;
    source.nested_execution.last_two_way_feedback.complete = true;
    source.nested_execution.last_two_way_feedback.interface_mismatch     = {1.0, 2.0, 3.0, 4.0, 5.0};
    source.nested_execution.last_two_way_feedback.applied_reflux         = {1.0, 2.0, 3.0, 4.0, 5.0};
    source.nested_execution.last_two_way_feedback.child_interface_source = {6.0, 7.0, 8.0, 9.0, 10.0};
    source.nested_execution.last_two_way_feedback.child_dynamics_storage = {0.0, 0.0, 0.25, -0.5, 0.75};
    source.nested_execution.two_way_dry_mass_storage_begin               = 12.5;
    source.nested_execution.two_way_theta_storage_begin                  = -7.25;
    source.nested_execution.two_way_feedback_application_count           = 3;
    source.configuration.domain_frame.earth_origin_at_t0_m.x_m += 1000.0;
    source.configuration.domain_frame.earth_origin_at_t0_m.y_m -= 800.0;
    source.nested_execution.relationship.child_origin_parent_cell_i -= 1;
    source.nested_execution.relationship.child_origin_parent_cell_j += 1;
    source.nested_execution.cumulative_parent_recycling_shift_i                = 1;
    source.nested_execution.cumulative_parent_recycling_shift_j                = -1;
    source.runtime.storm_window_movement.cumulative_i_shift_cells              = 1;
    source.runtime.storm_window_movement.cumulative_j_shift_cells              = -1;
    source.nested_execution.child.runtime.boundary_conditions.radiative_u_west = {9.0f, 10.0f};
    std::vector<std::uint8_t>                   first;
    std::vector<std::uint8_t>                   second;
    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;

    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, first, error));
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, second, error));
    REQUIRE(first == second);
    REQUIRE(tmv::checkpoint::wire::unwrap(
        first, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    const auto nesting =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(nesting != sections.end());
    REQUIRE(nesting->version == 8);
    REQUIRE((nesting->flags & tmv::checkpoint::wire::section_required) != 0);

    SimulationStorageState restored;
    INFO(error);
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(first, restored, error));
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(restored));
    const auto& expected = source.nested_execution;
    const auto& actual   = restored.nested_execution;
    REQUIRE(actual.relationship_ready);
    REQUIRE(actual.reference_state_ready);
    REQUIRE(actual.physical_primary_state_ready);
    REQUIRE(actual.parent_boundary_state_ready);
    REQUIRE(actual.two_way_feedback_application_count == 3);
    REQUIRE(actual.last_two_way_feedback.complete);
    REQUIRE(actual.last_two_way_feedback.child_interface_source ==
            expected.last_two_way_feedback.child_interface_source);
    REQUIRE(actual.last_two_way_feedback.child_dynamics_storage ==
            expected.last_two_way_feedback.child_dynamics_storage);
    REQUIRE(actual.two_way_dry_mass_storage_begin == 12.5);
    REQUIRE(actual.two_way_theta_storage_begin == -7.25);
    REQUIRE(actual.cumulative_parent_recycling_shift_i == 1);
    REQUIRE(actual.cumulative_parent_recycling_shift_j == -1);
    REQUIRE(actual.relationship.child_origin_parent_cell_i == 1);
    REQUIRE(actual.relationship.child_origin_parent_cell_j == 3);
    REQUIRE(actual.two_way_interval_source_begin.theta.committed_dry_mass_change_kg ==
            expected.two_way_interval_source_begin.theta.committed_dry_mass_change_kg);
    REQUIRE(actual.parent_time_bracket.completed_child_substeps ==
            expected.parent_time_bracket.completed_child_substeps);
    REQUIRE(actual.child.time_s == expected.child.time_s);
    REQUIRE(actual.child.domain_frame.earth_origin_at_t0_m.x_m == expected.child.domain_frame.earth_origin_at_t0_m.x_m);
    REQUIRE(actual.child.grid.geometry.z_faces == expected.child.grid.geometry.z_faces);
    REQUIRE(actual.child.fields.qv.size() == expected.child.fields.qv.size());
    REQUIRE(std::equal(actual.child.fields.qv.data(),
                       actual.child.fields.qv.data() + actual.child.fields.qv.size(),
                       expected.child.fields.qv.data()));
    REQUIRE(actual.child.surface.ustar == expected.child.surface.ustar);
    REQUIRE(actual.child.surface.precipitation_rate_mm_h == expected.child.surface.precipitation_rate_mm_h);
    REQUIRE(actual.child.surface.accumulated_rainfall_mm == expected.child.surface.accumulated_rainfall_mm);

    auto invalid_storage = source;
    invalid_storage.nested_execution.last_two_way_feedback.child_dynamics_storage[2] =
        std::numeric_limits<double>::quiet_NaN();
    std::vector<std::uint8_t> rejected = first;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(invalid_storage, rejected, error));
    REQUIRE(error.find("feedback provenance") != std::string::npos);
    REQUIRE(rejected == first);
    REQUIRE(actual.child.surface.accumulated_rainfall_residual_mm ==
            expected.child.surface.accumulated_rainfall_residual_mm);
    REQUIRE(actual.child.surface.accumulated_rainfall_last_update_s ==
            expected.child.surface.accumulated_rainfall_last_update_s);
    REQUIRE(actual.child.runtime.radiation.last_update_time_s == expected.child.runtime.radiation.last_update_time_s);
    REQUIRE(actual.child.runtime.boundary_conditions.radiative_u_west.empty());
    REQUIRE(actual.child.runtime.turbulence.momentum_work_step_count == 3);
    REQUIRE(actual.child.runtime.advection.tracer_roundoff_report.complete);
    REQUIRE(actual.child.runtime.advection.tracer_roundoff_report.negative_subnormal_count == 2);
    REQUIRE(actual.child.runtime.advection.cumulative_tracer_roundoff_report.negative_subnormal_count == 6);
    REQUIRE_FALSE(actual.child.runtime.advection.cumulative_tracer_roundoff_report.complete);
    REQUIRE(actual.child.runtime.advection.tracer_roundoff_report_step_count == 4);
    REQUIRE_FALSE(actual.child.runtime.advection.tracer_roundoff_cumulative_history_known);
    REQUIRE(actual.interface_source.available);
    REQUIRE(actual.interface_source.application_count == 1);
    REQUIRE(actual.interface_source.last.theta.target_dry_mass_change_kg == 2.0);
    REQUIRE(actual.interface_source.cumulative.energy.committed_change.total == 4.0);

    std::vector<std::uint8_t> restored_bytes;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored, restored_bytes, error));
    REQUIRE(restored_bytes == first);
}

TEST_CASE("active nesting checkpoint rejects open brackets and corrupt sections transactionally",
          "[core][checkpoint][nesting][synchronized][corruption]")
{
    constexpr std::array<std::uint8_t, 8> magic    = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source   = synchronized_nested_state();
    std::vector<std::uint8_t>             bytes    = {4, 3, 2, 1};
    const auto                            original = bytes;
    std::string                           error;

    source.nested_execution.parent_time_bracket.completed_child_substeps = 1;
    source.nested_execution.child.time_s                                 = 1.0;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error.find("synchronized parent/child endpoint") != std::string::npos);
    REQUIRE(bytes == original);

    source                                                                      = synchronized_nested_state();
    source.nested_execution.interface_source.last.theta.dry_storage_residual_kg = 0.75;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error.find("interface source provenance") != std::string::npos);
    REQUIRE(bytes == original);

    source                                               = synchronized_nested_state();
    source.nested_execution.child.fields.tracer(2, 2, 1) = -std::numeric_limits<float>::denorm_min();
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error.find("child state exceeds strict physical bounds") != std::string::npos);
    REQUIRE(bytes == original);

    source                                                      = synchronized_nested_state();
    source.nested_execution.cumulative_parent_recycling_shift_i = 1;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(error.find("movement provenance") != std::string::npos);
    REQUIRE(bytes == original);

    source = synchronized_nested_state();
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto nesting = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(nesting != sections.end());

    auto semantically_corrupt = sections;
    auto corrupt_nesting      = std::find_if(
        semantically_corrupt.begin(), semantically_corrupt.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(corrupt_nesting != semantically_corrupt.end());
    strip_active_terrain_nesting_v8_suffix(*corrupt_nesting);
    REQUIRE(corrupt_nesting->payload.size() >= 8);
    corrupt_nesting->payload.back() ^= 1U;
    REQUIRE(tmv::checkpoint::wire::encode_sections(semantically_corrupt, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    SimulationStorageState destination;
    destination.time_s = 444.0;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
    REQUIRE(error.find("movement provenance") != std::string::npos);
    REQUIRE(destination.time_s == 444.0);

    nesting->flags = 0;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
    REQUIRE(error.find("must be required") != std::string::npos);
    REQUIRE(destination.time_s == 444.0);

    nesting->flags = tmv::checkpoint::wire::section_required;
    REQUIRE(nesting->payload.size() > 8);
    nesting->payload[4] = 2U;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
    REQUIRE(destination.time_s == 444.0);
}

TEST_CASE("active-terrain nesting checkpoint rebuilds geometry and restores durable state exactly",
          "[core][checkpoint][nesting][active_terrain][roundtrip]")
{
    constexpr std::array<std::uint8_t, 8> magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source = synchronized_nested_state();
    publish_active_terrain(source);
    std::vector<std::uint8_t>                   first;
    std::vector<std::uint8_t>                   second;
    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;

    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, first, error));
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, second, error));
    REQUIRE(first == second);
    REQUIRE(tmv::checkpoint::wire::unwrap(
        first, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    const auto nesting =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(nesting != sections.end());
    REQUIRE(nesting->version == 8);

    SimulationStorageState restored;
    INFO(error);
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(first, restored, error));
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(restored));
    const auto& expected = source.nested_execution.active_terrain;
    const auto& actual   = restored.nested_execution.active_terrain;
    REQUIRE(actual.ready);
    REQUIRE(actual.generation == expected.generation);
    REQUIRE(actual.publication_count == expected.publication_count);
    REQUIRE(actual.parent_time_s == expected.parent_time_s);
    REQUIRE(actual.child_time_s == expected.child_time_s);
    REQUIRE(actual.geometry.generation == expected.geometry.generation);
    REQUIRE(actual.geometry.parent_origin_earth_m.x_m == expected.geometry.parent_origin_earth_m.x_m);
    REQUIRE(actual.geometry.child_origin_earth_m.y_m == expected.geometry.child_origin_earth_m.y_m);
    REQUIRE(actual.geometry.feature_center_earth_m.x_m == expected.geometry.feature_center_earth_m.x_m);
    REQUIRE(actual.geometry.parent.metrics.J.values == expected.geometry.parent.metrics.J.values);
    REQUIRE(actual.geometry.child_storage.metrics.Gx_face.values ==
            expected.geometry.child_storage.metrics.Gx_face.values);
    check_exact_terrain_field(actual.child_state.scalar, expected.child_state.scalar);
    check_exact_terrain_field(actual.child_state.x_flux, expected.child_state.x_flux);
    check_exact_terrain_field(actual.child_state.y_flux, expected.child_state.y_flux);
    check_exact_terrain_field(actual.child_state.zeta_flux, expected.child_state.zeta_flux);
    check_exact_terrain_field(actual.boundary.relaxation_weight, expected.boundary.relaxation_weight);
    check_exact_terrain_field(actual.boundary.ground_w_ms, expected.boundary.ground_w_ms);
    REQUIRE(actual.ledger.parent_scalar_inventory == expected.ledger.parent_scalar_inventory);
    REQUIRE(actual.ledger.normalized_scalar_inventory_residual == expected.ledger.normalized_scalar_inventory_residual);
    REQUIRE(actual.ledger.maximum_normalized_zeta_flux_residual ==
            expected.ledger.maximum_normalized_zeta_flux_residual);
    REQUIRE(actual.ledger.relaxation_owned_values == expected.ledger.relaxation_owned_values);
    REQUIRE(restored.nested_execution.child.runtime.terrain.metrics.J.values ==
            actual.geometry.child_storage.metrics.J.values);

    std::vector<std::uint8_t> restored_bytes;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored, restored_bytes, error));
    REQUIRE(restored_bytes == first);
}

TEST_CASE("stationary nested Schar checkpoint rebuild is byte exact",
          "[core][checkpoint][nesting][active_terrain][schar][roundtrip]")
{
    for (const auto feedback : {tmv::init::config::NestingFeedback::OneWay, tmv::init::config::NestingFeedback::TwoWay})
    {
        CAPTURE(static_cast<int>(feedback));
        SimulationStorageState source = synchronized_nested_state(feedback, 3);
        if (feedback == tmv::init::config::NestingFeedback::TwoWay)
            for (Field3D* field : {&source.fields.qv,
                                   &source.fields.qc,
                                   &source.fields.qr,
                                   &source.fields.qi,
                                   &source.fields.qs,
                                   &source.fields.qg,
                                   &source.fields.qh,
                                   &source.fields.tracer,
                                   &source.fields.tke,
                                   &source.nested_execution.child.fields.qv,
                                   &source.nested_execution.child.fields.qc,
                                   &source.nested_execution.child.fields.qr,
                                   &source.nested_execution.child.fields.qi,
                                   &source.nested_execution.child.fields.qs,
                                   &source.nested_execution.child.fields.qg,
                                   &source.nested_execution.child.fields.qh,
                                   &source.nested_execution.child.fields.tracer,
                                   &source.nested_execution.child.fields.tke})
                field->fill(0.0f);
        publish_active_terrain(source, "schar");
        if (feedback == tmv::init::config::NestingFeedback::TwoWay)
        {
            source.nested_execution.two_way_interval_source_begin = source.nested_execution.interface_source.cumulative;
            source.nested_execution.last_two_way_feedback.complete               = true;
            source.nested_execution.last_two_way_feedback.interface_mismatch     = {1.0, 2.0, 3.0, 4.0, 5.0};
            source.nested_execution.last_two_way_feedback.applied_reflux         = {1.0, 2.0, 3.0, 4.0, 5.0};
            source.nested_execution.last_two_way_feedback.child_interface_source = {6.0, 7.0, 8.0, 9.0, 10.0};
            source.nested_execution.last_two_way_feedback.child_dynamics_storage = {0.0, 0.0, 0.25, -0.5, 0.75};
            source.nested_execution.two_way_dry_mass_storage_begin               = 12.5;
            source.nested_execution.two_way_theta_storage_begin                  = -7.25;
            source.nested_execution.two_way_feedback_application_count           = 1;
        }
        std::vector<std::uint8_t> bytes;
        std::string               error;
        INFO(error);
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));

        SimulationStorageState restored;
        REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
        REQUIRE_NOTHROW(require_valid_nested_execution_relationship(restored));
        const auto& expected = source.nested_execution.active_terrain;
        const auto& actual   = restored.nested_execution.active_terrain;
        REQUIRE(restored.nested_execution.relationship.parent_ratio_x == 3);
        REQUIRE(restored.nested_execution.relationship.parent_ratio_y == 3);
        REQUIRE(restored.nested_execution.relationship.parent_time_step_ratio == 3);
        REQUIRE(actual.ready);
        REQUIRE(actual.geometry.config.scheme_id == "schar");
        REQUIRE(actual.geometry.parent.metrics.J.values == expected.geometry.parent.metrics.J.values);
        REQUIRE(actual.geometry.child_storage.metrics.z_face.values ==
                expected.geometry.child_storage.metrics.z_face.values);
        REQUIRE(actual.child_state.scalar.values == expected.child_state.scalar.values);
        REQUIRE(actual.boundary.ground_w_ms.values == expected.boundary.ground_w_ms.values);

        std::vector<std::uint8_t> roundtrip;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored, roundtrip, error));
        REQUIRE(roundtrip == bytes);
    }
}

TEST_CASE("nesting v7 checkpoints default active terrain cleanly",
          "[core][checkpoint][nesting][active_terrain][compatibility]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    const SimulationStorageState                source = synchronized_nested_state();
    std::vector<std::uint8_t>                   bytes;
    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto nesting = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(nesting != sections.end());
    strip_active_terrain_nesting_v8_suffix(*nesting);
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE_FALSE(restored.nested_execution.active_terrain.ready);
    REQUIRE(restored.nested_execution.active_terrain.generation == 0);
    REQUIRE(restored.nested_execution.active_terrain.publication_count == 0);
    REQUIRE(restored.nested_execution.active_terrain.child_state.scalar.values.empty());
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(restored));
}

TEST_CASE("active-terrain checkpoint corruption and identity mismatch roll back transactionally",
          "[core][checkpoint][nesting][active_terrain][corruption]")
{
    constexpr std::array<std::uint8_t, 8> magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source = synchronized_nested_state();
    publish_active_terrain(source);
    std::vector<std::uint8_t>                   bytes;
    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    const auto original = bytes;

    auto mismatch = source;
    mismatch.nested_execution.active_terrain.geometry.relationship.child_dx_m += 1.0;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(mismatch, bytes, error));
    REQUIRE(error.find("active-terrain checkpoint identity") != std::string::npos);
    REQUIRE(bytes == original);

    REQUIRE(tmv::checkpoint::wire::unwrap(
        original, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    const auto locate_active_payload = [](auto& section)
    {
        std::uint64_t size = 0;
        for (std::size_t byte = 0; byte < sizeof(size); ++byte)
            size |= static_cast<std::uint64_t>(section.payload[section.payload.size() - 8 + byte]) << (8U * byte);
        const std::size_t start = section.payload.size() - static_cast<std::size_t>(size) - 8;
        return std::pair<std::size_t, std::size_t>{start, static_cast<std::size_t>(size)};
    };

    auto assert_rejected = [&](std::vector<tmv::checkpoint::wire::Section> corrupt, const std::string& expected_error)
    {
        REQUIRE(tmv::checkpoint::wire::encode_sections(corrupt, payload, error));
        tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
        SimulationStorageState destination;
        destination.time_s = 444.0;
        REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, destination, error));
        REQUIRE(error.find(expected_error) != std::string::npos);
        REQUIRE(destination.time_s == 444.0);
    };

    auto bad_sampling = source;
    bad_sampling.nested_execution.active_terrain.geometry.sampling_version = 3;
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(bad_sampling, bytes, error));
    REQUIRE(bytes == original);

    auto bad_schema = sections;
    auto nesting =
        std::find_if(bad_schema.begin(), bad_schema.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(nesting != bad_schema.end());
    const auto [active_start, active_size] = locate_active_payload(*nesting);
    REQUIRE(active_size > 32);
    REQUIRE(nesting->payload[active_start] == 1U);
    nesting->payload[active_start + 1] = tmv::domain::NestedActiveTerrainStateV1::schema_version + 1;
    assert_rejected(std::move(bad_schema), "unsupported active-terrain checkpoint schema version");

    auto bad_clock = sections;
    nesting = std::find_if(bad_clock.begin(), bad_clock.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(nesting != bad_clock.end());
    const auto [clock_start, clock_size] = locate_active_payload(*nesting);
    REQUIRE(clock_size >= 32);
    nesting->payload[clock_start + clock_size - 24] ^= 1U;
    assert_rejected(std::move(bad_clock), "active-terrain checkpoint clocks are not synchronized");
}

TEST_CASE("nesting v6 checkpoints default parent recycling provenance to zero",
          "[core][checkpoint][nesting][compatibility]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    const SimulationStorageState                source = synchronized_nested_state();
    std::vector<std::uint8_t>                   bytes;
    std::vector<std::uint8_t>                   payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;

    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    auto nesting = std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 7; });
    REQUIRE(nesting != sections.end());
    strip_active_terrain_nesting_v8_suffix(*nesting);
    REQUIRE(nesting->payload.size() >= 8);
    nesting->version = 6;
    nesting->payload.resize(nesting->payload.size() - 8);
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.nested_execution.cumulative_parent_recycling_shift_i == 0);
    REQUIRE(restored.nested_execution.cumulative_parent_recycling_shift_j == 0);
    REQUIRE_NOTHROW(require_valid_nested_execution_relationship(restored));
}

TEST_CASE("runtime v35 D5 provenance restores as an explicitly incomplete legacy exchange",
          "[core][checkpoint][domain_recycling][compatibility][SCI-228]")
{
    constexpr std::array<std::uint8_t, 8> magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                source = sample_state();
    seed_committed_domain_recycling_provenance(source);
    auto& p                                    = source.runtime.domain_recycling_provenance;
    p.control_volume_exchange_complete         = false;
    p.admitted_inventories                     = {};
    p.admitted_momentum                        = {};
    p.admitted_moist_mixture_energy            = {};
    p.admitted_theta_storage_residual_energy_j = 0.0;
    p.retiring_fixed_earth_angular_momentum_z  = 0.0;
    p.admitted_fixed_earth_angular_momentum_z  = 0.0;
    p.control_volume_exchange_basis            = {};
    REQUIRE(tmv::domain::valid_domain_recycling_transaction_provenance(p));

    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    const auto runtime =
        std::find_if(sections.begin(), sections.end(), [](const auto& section) { return section.id == 6; });
    REQUIRE(runtime != sections.end());
    strip_chaos_pbl_cache_v37(*runtime, source);
    REQUIRE(runtime->version == 36);
    REQUIRE_FALSE(runtime->payload.empty());
    // The only v36 suffix for an incomplete record is its explicit false
    // marker. Runtime v35 ended immediately after the retiring-side basis.
    REQUIRE(runtime->payload.back() == 0U);
    runtime->payload.pop_back();
    runtime->version = 35;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    const auto& actual = restored.runtime.domain_recycling_provenance;
    REQUIRE(actual.available);
    CHECK_FALSE(actual.control_volume_exchange_complete);
    CHECK(actual.transaction_id.view() == "recycle_000007");
    CHECK(tmv::domain::valid_domain_recycling_transaction_provenance(actual));
}

TEST_CASE("simulation checkpoint sections are order-independent and evolvable", "[core][checkpoint][mod03]")
{
    constexpr std::array<std::uint8_t, 8>       magic  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
    SimulationStorageState                      source = sample_state();
    std::vector<std::uint8_t>                   bytes, payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string                                 error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    REQUIRE(tmv::checkpoint::wire::unwrap(
        bytes, magic, tmv::checkpoint::simulation_checkpoint_version, payload, error, "simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload, sections, error));
    REQUIRE(std::none_of(sections.begin(), sections.end(), [](const auto& section) { return section.id == 7; }));
    std::reverse(sections.begin(), sections.end());
    sections.push_back({9001, 77, 0, {1, 2, 3, 4}});
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);

    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    CHECK(restored.time_s == Approx(source.time_s));

    sections.back().flags = tmv::checkpoint::wire::section_required;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections, payload, error));
    tmv::checkpoint::wire::wrap(magic, tmv::checkpoint::simulation_checkpoint_version, payload, bytes);
    restored.time_s = 444.0;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    CHECK(restored.time_s == Approx(444.0));
}

TEST_CASE("RunContext publishes checkpoints atomically", "[core][checkpoint][mod03]")
{
    tmv::RunContext source;
    source.storage()               = sample_state();
    const auto                path = std::filesystem::temp_directory_path() / "nimbus_mod03_simulation_checkpoint.bin";
    std::string               error;
    std::vector<std::uint8_t> expected;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source.storage(), expected, error));
    REQUIRE(source.save(path, error));
    std::vector<std::uint8_t> published;
    REQUIRE(tmv::checkpoint::wire::read_file(path, published, error, "simulation"));
    REQUIRE(published == expected);

    tmv::RunContext restored;
    const bool      loaded = restored.load(path, error);
    INFO(error);
    REQUIRE(loaded);
    CHECK(restored.storage().time_s == Approx(source.storage().time_s));
    CHECK(restored.state().compute_runtime == &restored.storage().runtime.compute);
    CHECK(restored.state().configuration == &restored.storage().configuration);
    std::filesystem::remove(path);
}

TEST_CASE("checkpoint carries the level-owned vertical flux execution selection",
          "[core][checkpoint][execution_owner]")
{
    auto source = sample_state();
    source.runtime.compute.vertical_flux_template_id = "tvd_vertical_flux_v1";
    std::vector<std::uint8_t> bytes;
    std::string error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.compute.vertical_flux_template_id == "tvd_vertical_flux_v1");
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(source, restored, error));
    source.runtime.compute.vertical_flux_template_id = "tvd_vertical_flux_v2";
    REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(source, restored, error));
    REQUIRE(error.find("vertical-flux execution selection") != std::string::npos);
    source.runtime.compute.vertical_flux_template_id = "missing_template";
    REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
}

TEST_CASE("checkpoint retains the native chaos policy with a detached backend",
          "[core][checkpoint][chaos][execution_owner]")
{
    auto source = sample_state();
    source.runtime.chaos.config.native_generation_required = true;
    std::vector<std::uint8_t> bytes;
    std::string error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.runtime.chaos.config.native_generation_required);
    REQUIRE_FALSE(restored.runtime.compute.backend);
}

TEST_CASE("checkpoint compatibility covers radiation and diffusion suffix parameters",
          "[core][checkpoint][science][compatibility]")
{
    auto original = sample_state();
    std::string error;
    const auto reject = [&](const auto& mutate)
    {
        auto changed = original;
        mutate(changed);
        REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(original, changed, error));
        REQUIRE_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(changed, original, error));
    };
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(original, original, error));
    reject([](auto& s) { s.runtime.radiation.config.subcycle_count += 1; });
    reject([](auto& s) { s.runtime.radiation.config.simple_grey_optical_model = "pressure_clear_sky_v1"; });
    for (auto member : {&RadiationConfig::simple_grey_lw_vapor_absorption_m2kg,
                        &RadiationConfig::simple_grey_lw_liquid_absorption_m2kg,
                        &RadiationConfig::simple_grey_lw_ice_absorption_m2kg,
                        &RadiationConfig::simple_grey_sw_vapor_absorption_m2kg,
                        &RadiationConfig::simple_grey_sw_liquid_absorption_m2kg,
                        &RadiationConfig::simple_grey_sw_ice_absorption_m2kg})
        reject([&](auto& s) { s.runtime.radiation.config.*member += 1.0; });
    reject([](auto& s) { s.runtime.diffusion.config.coefficient_source = s.runtime.diffusion.config.coefficient_source == "linear_height" ? "constant" : "linear_height"; });
    reject([](auto& s) { s.runtime.diffusion.config.K_v_surface += 1.0; });
    reject([](auto& s) { s.runtime.diffusion.config.K_v_top += 1.0; });
}

TEST_CASE("runtime v53 sampling identity is validated and legacy v52 remains readable",
          "[core][checkpoint][terrain][sampling][v53]")
{
    constexpr std::array<std::uint8_t,8> magic={'N','I','M','B','S','I','M','1'};
    const auto source=sample_state();
    std::vector<std::uint8_t> bytes,payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source,bytes,error));
    REQUIRE(tmv::checkpoint::wire::unwrap(bytes,magic,tmv::checkpoint::simulation_checkpoint_version,payload,error,"simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload,sections,error));
    auto runtime=std::find_if(sections.begin(),sections.end(),[](const auto& value){return value.id==6;});
    REQUIRE(runtime!=sections.end());
    REQUIRE(runtime->version==55);
    strip_coordinate_remap_v54_suffix(*runtime);
    REQUIRE(runtime->payload.back()==0);
    runtime->payload.back()=255;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections,payload,error));
    tmv::checkpoint::wire::wrap(magic,tmv::checkpoint::simulation_checkpoint_version,payload,bytes);
    SimulationStorageState restored;
    restored.time_s=444.0;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes,restored,error));
    REQUIRE(restored.time_s==444.0);
    REQUIRE(error.find("sampling")!=std::string::npos);
    runtime->payload.pop_back();
    runtime->version=52;
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections,payload,error));
    tmv::checkpoint::wire::wrap(magic,tmv::checkpoint::simulation_checkpoint_version,payload,bytes);
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes,restored,error));
    REQUIRE(restored.runtime.terrain.earth_fixed_placement.sampling==tmv::EarthFixedTerrainSampling::Point);
    REQUIRE(restored.time_s==source.time_s);
}

TEST_CASE("runtime v54 preserves coordinate-remapping energy and rejects corrupt owners",
          "[core][checkpoint][earth_fixed][v54]")
{
    constexpr std::array<std::uint8_t,8> magic={'N','I','M','B','S','I','M','1'};
    auto source=sample_state();
    tmv::CompletedStepEnergyReportV1 report;
    report.coordinate_remap_committed_change_j=3.25;
    report.dry_internal_change_j=3.25;
    report.committed_owner_sum_j=3.25;
    report.complete=true;
    source.runtime.time_stepping.last_completed_step_energy=report;
    source.runtime.time_stepping.cumulative_completed_step_energy=report;
    source.runtime.time_stepping.completed_step_energy_count=1;
    std::vector<std::uint8_t> bytes,payload;
    std::string error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source,bytes,error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes,restored,error));
    CHECK(restored.runtime.time_stepping.last_completed_step_energy.coordinate_remap_committed_change_j==3.25);
    CHECK(restored.runtime.time_stepping.cumulative_completed_step_energy.coordinate_remap_committed_change_j==3.25);
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::unwrap(bytes,magic,tmv::checkpoint::simulation_checkpoint_version,payload,error,"simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload,sections,error));
    auto runtime=std::find_if(sections.begin(),sections.end(),[](const auto& section){return section.id==6;});
    REQUIRE(runtime!=sections.end());
    REQUIRE(runtime->version==55);
    strip_private_inventory_v55_suffix(*runtime);
    tmv::checkpoint::wire::Encoder invalid;
    invalid.unsigned_value<std::uint64_t>(UINT64_C(0x7ff8000000000000));
    std::copy(invalid.bytes.begin(),invalid.bytes.end(),runtime->payload.end()-8);
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections,payload,error));
    tmv::checkpoint::wire::wrap(magic,tmv::checkpoint::simulation_checkpoint_version,payload,bytes);
    restored.time_s=123.0;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes,restored,error));
    CHECK(restored.time_s==123.0);
    CHECK(error.find("coordinate-remapping")!=std::string::npos);
}

TEST_CASE("Checkpoint preserves Lilly equation identity and rejects legacy Ri continuation",
          "[core][checkpoint][richardson][science]")
{
    auto source=sample_state();
    source.runtime.turbulence.config.scheme_id="smagorinsky";
    source.runtime.turbulence.config.stability_correction=tmv::kLillyRichardsonModel;
    source.runtime.turbulence.config.mode="3d";
    source.runtime.turbulence.scheme=create_turbulence_scheme("smagorinsky");
    std::vector<std::uint8_t> bytes;
    std::string error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source,bytes,error));
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes,restored,error));
    CHECK(restored.runtime.turbulence.config.stability_correction==tmv::kLillyRichardsonModel);
    REQUIRE(tmv::checkpoint::simulation_checkpoint_compatible(source,restored,error));
    restored.runtime.turbulence.config.stability_correction="ri";
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(source,restored,error));
    CHECK_FALSE(tmv::checkpoint::simulation_checkpoint_compatible(restored,source,error));
}

TEST_CASE("Unknown Richardson checkpoint model rejects without replacing live state",
          "[core][checkpoint][richardson][atomicity]")
{
    constexpr std::array<std::uint8_t,8> magic={'N','I','M','B','S','I','M','1'};
    auto source=sample_state();
    source.runtime.turbulence.config.stability_correction=tmv::kLillyRichardsonModel;
    std::vector<std::uint8_t> bytes,payload;
    std::vector<tmv::checkpoint::wire::Section> sections;
    std::string error;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source,bytes,error));
    REQUIRE(tmv::checkpoint::wire::unwrap(bytes,magic,tmv::checkpoint::simulation_checkpoint_version,payload,error,"simulation"));
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload,sections,error));
    auto runtime=std::find_if(sections.begin(),sections.end(),[](const auto& value){return value.id==6;});
    REQUIRE(runtime!=sections.end());
    const std::string tag=tmv::kLillyRichardsonModel;
    auto pos=std::search(runtime->payload.begin(),runtime->payload.end(),tag.begin(),tag.end());
    REQUIRE(pos!=runtime->payload.end());
    *(pos+tag.size()-1)='9';
    REQUIRE(tmv::checkpoint::wire::encode_sections(sections,payload,error));
    tmv::checkpoint::wire::wrap(magic,tmv::checkpoint::simulation_checkpoint_version,payload,bytes);
    auto destination=sample_state();destination.time_s=444.0;
    REQUIRE_FALSE(tmv::checkpoint::deserialize_simulation_storage(bytes,destination,error));
    CHECK(destination.time_s==444.0);
    CHECK(error.find("unknown turbulence stability model")!=std::string::npos);
}

TEST_CASE("Moist two-way feedback ledgers survive checkpoints and reject inconsistent sources",
          "[core][checkpoint][nesting][moist]")
{
    auto source = synchronized_nested_state();
    tmv::fields::visit(source.fields,
                       tmv::fields::prognostic,
                       [&](const char*, Field3D& field)
                       { field.resize(source.grid.nr, source.grid.nth, source.grid.nz, 0); });
    source.fields.rho.fill(1);
    source.fields.theta.fill(300);
    source.fields.p.fill(90000);
    source.configuration.nesting.feedback = tmv::init::config::NestingFeedback::TwoWay;
    auto& nested                          = source.nested_execution;
    for (auto* fields : {&source.fields, &nested.child.fields})
    {
        fields->qv.fill(.005f);
        for (auto* field : {&fields->qc,
                            &fields->qr,
                            &fields->qi,
                            &fields->qs,
                            &fields->qg,
                            &fields->qh,
                            &fields->tracer,
                            &fields->tke})
            field->fill(0);
    }
    auto& report                              = nested.last_two_way_feedback;
    report.complete                           = true;
    report.moist                              = true;
    nested.two_way_feedback_application_count = 1;
    nested.two_way_interval_source_begin      = nested.interface_source.cumulative;
    const auto&                         r     = nested.relationship;
    tmv::CartesianScalarInterfaceFluxes flux;
    flux.west.assign(r.child_parent_span_y * r.levels_z, 0);
    flux.east = flux.west;
    flux.south.assign(r.child_parent_span_x * r.levels_z, 0);
    flux.north              = flux.south;
    flux.accumulated_dt_s   = r.parent_dt_s;
    flux.contribution_count = 1;
    const std::array<Field3D*, 7> parent{&source.fields.qv,
                                         &source.fields.qc,
                                         &source.fields.qr,
                                         &source.fields.qi,
                                         &source.fields.qs,
                                         &source.fields.qg,
                                         &source.fields.qh};
    const std::array<Field3D*, 7> child{&nested.child.fields.qv,
                                        &nested.child.fields.qc,
                                        &nested.child.fields.qr,
                                        &nested.child.fields.qi,
                                        &nested.child.fields.qs,
                                        &nested.child.fields.qg,
                                        &nested.child.fields.qh};
    for (std::size_t f = 0; f < 7; ++f)
    {
        auto  candidate   = prepare_nested_scalar_feedback_flat(r,
                                                                nested.child.process_grid_metrics.dz,
                                                                source.fields.rho,
                                                                source.fields.rho,
                                                                nested.child.fields.rho,
                                                                *parent[f],
                                                                *child[f],
                                                                flux,
                                                                flux);
        auto& value       = report.water[f];
        value.parent      = candidate.report;
        value.child_begin = candidate.report.child_inventory;
        value.tolerance   = 512 * std::numeric_limits<float>::epsilon() * value.child_begin;
        *parent[f]        = std::move(candidate.scalar);
        REQUIRE(valid_nested_water_feedback_report(value));
    }
    constexpr double phase_exchange = 1e-30;
    report.phase_projection_cells = 1;
    report.phase_thermal_scale_j = 1;
    report.water[0].parent.phase_exchange_inventory = -phase_exchange;
    report.water[0].parent.storage_rounding += phase_exchange;
    report.water[1].parent.applied_reflux = -phase_exchange;
    report.water[1].parent.interface_mismatch = -phase_exchange;
    report.water[1].parent.phase_exchange_inventory = phase_exchange;
    REQUIRE(valid_nested_phase_feedback_report(report));
    std::vector<std::uint8_t> bytes;
    std::string               error;
    const bool                saved = tmv::checkpoint::serialize_simulation_storage(source, bytes, error);
    INFO(error);
    REQUIRE(saved);
    SimulationStorageState restored;
    REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
    REQUIRE(restored.nested_execution.last_two_way_feedback.moist);
    CHECK(restored.nested_execution.last_two_way_feedback.water[0].parent.child_inventory ==
          report.water[0].parent.child_inventory);
    CHECK(restored.nested_execution.last_two_way_feedback.phase_projection_cells == 1);
    CHECK(restored.nested_execution.last_two_way_feedback.phase_thermal_scale_j == 1);
    CHECK(restored.nested_execution.last_two_way_feedback.water[1].parent.phase_exchange_inventory == phase_exchange);
    const auto original = bytes;
    report.water[1].parent.phase_exchange_inventory += 1;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(bytes == original);
    report.water[1].parent.phase_exchange_inventory = phase_exchange;
    report.phase_thermal_roundoff_j = 1;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(bytes == original);
    report.phase_thermal_roundoff_j = 0;
    report.water[0].physics_source += 1e6;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(bytes == original);
    report.water[0].physics_source -= 1e6;
    nested.child.fields.qv(1, 1, 0) += .001f;
    CHECK_FALSE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
    CHECK(error.find("synchronized water inventory") != std::string::npos);
    CHECK(bytes == original);
}

TEST_CASE("moving mapped checkpoint preserves lattice sampling and rejects legacy placement",
          "[core][checkpoint][nesting][active_terrain][movement]")
{
    for (const int ratio : {2, 3})
    {
        auto source = synchronized_nested_state(tmv::init::config::NestingFeedback::OneWay, ratio);
        source.configuration.domain_motion_mode = tmv::init::config::DomainMotionMode::MovingNest;
        publish_active_terrain(source, "schar", 2);
        std::vector<std::uint8_t> bytes;
        std::string               error;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(source, bytes, error));
        SimulationStorageState restored;
        REQUIRE(tmv::checkpoint::deserialize_simulation_storage(bytes, restored, error));
        CHECK(restored.nested_execution.active_terrain.geometry.sampling_version == 2);
        CHECK(restored.nested_execution.active_terrain.geometry.child_storage.metrics.J.values ==
              source.nested_execution.active_terrain.geometry.child_storage.metrics.J.values);
        std::vector<std::uint8_t> roundtrip;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(restored, roundtrip, error));
        CHECK(roundtrip == bytes);
        auto legacy = synchronized_nested_state(tmv::init::config::NestingFeedback::OneWay, ratio);
        legacy.configuration.domain_motion_mode = tmv::init::config::DomainMotionMode::MovingNest;
        publish_active_terrain(legacy, "schar", 1);
        REQUIRE_FALSE(tmv::checkpoint::serialize_simulation_storage(legacy, roundtrip, error));
        CHECK(error.find("parent-lattice terrain sampling") != std::string::npos);
        CHECK(roundtrip == bytes);
    }
}

TEST_CASE("public checkpoint reads preserve state and integrity after semantic rejection",
          "[core][checkpoint][atomicity][integrity]")
{
    SimulationStorageState source=sample_state();
    SimulationStorageState live=sample_state();
    live.time_s=3.0;
    std::string error;
    std::vector<std::uint8_t> source_bytes,prior_bytes;
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(source,source_bytes,error));
    REQUIRE(tmv::checkpoint::serialize_simulation_storage(live,prior_bytes,error));
    constexpr std::array<std::uint8_t,8> magic{{'N','I','M','B','S','I','M','1'}};
    std::vector<std::uint8_t> payload;
    REQUIRE(tmv::checkpoint::wire::unwrap(source_bytes,magic,tmv::checkpoint::simulation_checkpoint_version,payload,error,"test"));
    std::vector<tmv::checkpoint::wire::Section> sections;
    REQUIRE(tmv::checkpoint::wire::decode_sections(payload,sections,error));
    const auto path=std::filesystem::temp_directory_path()/"nimbus_semantic_checkpoint_rejection.nimbus";
    for (bool unknown_section:{false,true})
    {
        auto malformed=sections;
        if (unknown_section)
            malformed.push_back({99,1,tmv::checkpoint::wire::section_required,{1}});
        else
        {
            auto section=std::find_if(malformed.begin(),malformed.end(),[](const auto& item){return item.id==6;});
            REQUIRE(section!=malformed.end());
            section->version=std::numeric_limits<std::uint32_t>::max();
        }
        REQUIRE(tmv::checkpoint::wire::write_sections_atomic(path,magic,tmv::checkpoint::simulation_checkpoint_version,
                                                             malformed,error,"test"));
        tmv::checkpoint::SimulationCheckpointAttachments attachments;
        attachments.trajectories.emplace();
        attachments.trajectories->next_sample_time_s=7.0;
        tmv::checkpoint::SimulationCheckpointIntegrity integrity{17,"prior-fnv","prior-sha"};
        REQUIRE_FALSE(tmv::checkpoint::read_simulation_checkpoint(path,live,attachments,integrity,error));
        CHECK_FALSE(error.empty());
        CHECK(integrity.bytes==17);
        CHECK(integrity.fnv1a64=="prior-fnv");
        CHECK(integrity.sha256=="prior-sha");
        REQUIRE(attachments.trajectories.has_value());
        CHECK(attachments.trajectories->next_sample_time_s==7.0);
        std::vector<std::uint8_t> after;
        REQUIRE(tmv::checkpoint::serialize_simulation_storage(live,after,error));
        CHECK(after==prior_bytes);
    }
    std::filesystem::remove(path);
}
