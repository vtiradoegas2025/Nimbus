#include "diagnostics/level_budget_csv.hpp"
#include "microphysics/runtime.hpp"

#include "core/runtime/simulation_state.hpp"
#include "boundary_layer/runtime.hpp"
#include "microphysics/factory.hpp"
#include "turbulence/runtime.hpp"
#include "util/log.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <mutex>

namespace
{
tmv::LevelBudgetCsv g_source_budget_files;
std::mutex g_source_budget_csv_mutex;
constexpr std::array<const char*, 7> k_water_species{{
    "qv", "qc", "qr", "qi", "qs", "qg", "qh"}};
constexpr std::array<const char*, tmv::thermodynamic_boundary_report_count>
    k_transport_scalars{{"theta", "qv", "qc", "qr",
                         "qi", "qs", "qg", "qh"}};
}

namespace tmv
{

std::unique_ptr<MicrophysicsScheme>& microphysics_scheme =
    global_microphysics_runtime_state.scheme;

void initialize_microphysics(MicrophysicsRuntimeState& runtime,
                             const std::string& scheme_name)
{
    initialize_microphysics(runtime, scheme_name, MicrophysicsConfig{});
}

void initialize_microphysics(MicrophysicsRuntimeState& runtime,
                             const std::string& scheme_name,
                             const MicrophysicsConfig& config)
{
    std::unique_ptr<MicrophysicsScheme> selected =
        create_microphysics_scheme(scheme_name, config);
    const std::string canonical_name = selected->get_scheme_name();
    const std::string caveat = selected->capability_caveat();

    runtime = {};
    runtime.scheme = std::move(selected);
    tmv::log_info("Initialized microphysics scheme: ", canonical_name);
    if (canonical_name != "none")
    {
        tmv::log_info(
            "Microphysics parameters: shared.cloud_water_threshold_kgkg=",
            config.shared.cloud_water_threshold_kgkg,
            ", shared.autoconversion_rate_s_inv=",
            config.shared.autoconversion_rate_s_inv);
        if (canonical_name == "kessler")
        {
            tmv::log_info(
                "Microphysics parameters: kessler.accretion_coefficient=",
                config.kessler.accretion_coefficient,
                ", kessler.evaporation_efficiency=",
                config.kessler.evaporation_efficiency);
        }
        else if (canonical_name == "lin")
        {
            tmv::log_info(
                "Microphysics parameters: lin.accretion_coefficient=",
                config.lin.accretion_coefficient,
                ", lin.evaporation_rate_s_inv=",
                config.lin.evaporation_rate_s_inv);
        }
        else if (canonical_name == "thompson")
        {
            tmv::log_info(
                "Microphysics parameters: thompson.evaporation_rate_s_inv=",
                config.thompson.evaporation_rate_s_inv);
        }
        else if (canonical_name == "milbrandt")
        {
            tmv::log_info(
                "Microphysics parameters: published double-moment closures; "
                "milbrandt.hail_processes=",
                config.milbrandt.hail_processes ? "true" : "false");
        }
    }
    if (!caveat.empty())
    {
        tmv::log_warn("[MICROPHYSICS] '", canonical_name,
                      "' caveat: ", caveat);
    }
}

void initialize_microphysics(const std::string& scheme_name)
{
    initialize_microphysics(global_microphysics_runtime_state, scheme_name);
}

void initialize_microphysics(const std::string& scheme_name,
                             const MicrophysicsConfig& config)
{
    initialize_microphysics(global_microphysics_runtime_state, scheme_name,
                            config);
}

void open_thermodynamic_source_budget_csv(const std::string& path)
{
    std::lock_guard<std::mutex> lock(g_source_budget_csv_mutex);
    std::ostringstream header;
    header
        << "time_s,dt_s,microphysics_heat_j,radiation_heat_j,pbl_heat_j,"
           "requested_heat_j,theta_repair_j,theta_storage_rounding_j,"
           "theta_storage_residual_j,committed_heat_j,"
           "state_heat_change_j,"
           "microphysics_moist_energy_report_schema_version,"
           "microphysics_moist_energy_basis_version,"
           "microphysics_moist_dry_internal_change_j,"
           "microphysics_moist_water_sensible_change_j,"
           "microphysics_moist_phase_reference_change_j,"
           "microphysics_moist_carrier_kinetic_change_j,"
           "microphysics_moist_gravitational_change_j,"
           "microphysics_moist_subgrid_tke_change_j,"
           "microphysics_moist_committed_energy_change_j,"
           "microphysics_moist_energy_closure_residual_j,"
           "microphysics_moist_energy_complete,"
           "microphysics_water_tendency_kg,pbl_vapor_tendency_kg,"
           "pbl_cloud_liquid_tendency_kg,pbl_cloud_ice_tendency_kg,"
           "surface_precipitation_kg,microphysics_conservation_residual_kg,"
           "microphysics_conservation_active,microphysics_baseline_water_kg,"
           "water_storage_rounding_kg,water_repair_kg,"
           "committed_water_change_kg";
    for (const char* species : k_water_species)
    {
        header
            << ",microphysics_" << species << "_requested_kg"
            << ',' << species << "_storage_rounding_kg"
            << ',' << species << "_repair_kg"
            << ',' << species << "_committed_change_kg";
    }
    header
        << ",pbl_requested_momentum_work_j,"
           "pbl_finite_step_momentum_correction_j,"
           "pbl_momentum_storage_rounding_j,"
           "pbl_committed_momentum_work_j,"
           "pbl_cumulative_requested_momentum_work_j,"
           "pbl_cumulative_finite_step_momentum_correction_j,"
           "pbl_cumulative_momentum_storage_rounding_j,"
           "pbl_cumulative_committed_momentum_work_j,"
           "pbl_momentum_work_step_count,"
           "sgs_requested_momentum_work_j,"
           "sgs_finite_step_momentum_correction_j,"
           "sgs_momentum_storage_rounding_j,"
           "sgs_committed_momentum_work_j,"
           "sgs_cumulative_requested_momentum_work_j,"
           "sgs_cumulative_finite_step_momentum_correction_j,"
           "sgs_cumulative_momentum_storage_rounding_j,"
           "sgs_cumulative_committed_momentum_work_j,"
           "sgs_momentum_work_step_count,"
           "sgs_open_boundary_work_j,"
           "sgs_irreversible_dissipation_j,"
           "sgs_stress_closure_residual_j,"
           "sgs_cumulative_open_boundary_work_j,"
           "sgs_cumulative_irreversible_dissipation_j,"
           "sgs_cumulative_stress_closure_residual_j,"
           "sgs_open_boundary_stability_step_count,"
           "sgs_moist_energy_report_schema_version,"
           "sgs_moist_energy_basis_version,"
           "sgs_moist_energy_step_dry_internal_change_j,"
           "sgs_moist_energy_step_water_sensible_change_j,"
           "sgs_moist_energy_step_phase_reference_change_j,"
           "sgs_moist_energy_step_carrier_kinetic_change_j,"
           "sgs_moist_energy_step_gravitational_change_j,"
           "sgs_moist_energy_step_subgrid_tke_change_j,"
           "sgs_moist_energy_step_committed_energy_change_j,"
           "sgs_moist_energy_step_closure_residual_j,"
           "sgs_moist_energy_step_complete,"
           "sgs_moist_energy_cumulative_dry_internal_change_j,"
           "sgs_moist_energy_cumulative_water_sensible_change_j,"
           "sgs_moist_energy_cumulative_phase_reference_change_j,"
           "sgs_moist_energy_cumulative_carrier_kinetic_change_j,"
           "sgs_moist_energy_cumulative_gravitational_change_j,"
           "sgs_moist_energy_cumulative_subgrid_tke_change_j,"
           "sgs_moist_energy_cumulative_committed_energy_change_j,"
           "sgs_moist_energy_cumulative_closure_residual_j,"
           "sgs_moist_energy_cumulative_complete,"
           "diffusion_requested_momentum_work_j,"
           "diffusion_finite_step_momentum_correction_j,"
           "diffusion_momentum_storage_rounding_j,"
           "diffusion_committed_momentum_work_j,"
           "diffusion_cumulative_requested_momentum_work_j,"
           "diffusion_cumulative_finite_step_momentum_correction_j,"
           "diffusion_cumulative_momentum_storage_rounding_j,"
           "diffusion_cumulative_committed_momentum_work_j,"
           "diffusion_momentum_work_step_count,"
           "diffusion_moist_energy_report_schema_version,"
           "diffusion_moist_energy_basis_version,"
           "diffusion_moist_energy_step_dry_internal_change_j,"
           "diffusion_moist_energy_step_water_sensible_change_j,"
           "diffusion_moist_energy_step_phase_reference_change_j,"
           "diffusion_moist_energy_step_carrier_kinetic_change_j,"
           "diffusion_moist_energy_step_gravitational_change_j,"
           "diffusion_moist_energy_step_subgrid_tke_change_j,"
           "diffusion_moist_energy_step_committed_energy_change_j,"
           "diffusion_moist_energy_step_closure_residual_j,"
           "diffusion_moist_energy_step_complete,"
           "diffusion_moist_energy_cumulative_dry_internal_change_j,"
           "diffusion_moist_energy_cumulative_water_sensible_change_j,"
           "diffusion_moist_energy_cumulative_phase_reference_change_j,"
           "diffusion_moist_energy_cumulative_carrier_kinetic_change_j,"
           "diffusion_moist_energy_cumulative_gravitational_change_j,"
           "diffusion_moist_energy_cumulative_subgrid_tke_change_j,"
           "diffusion_moist_energy_cumulative_committed_energy_change_j,"
           "diffusion_moist_energy_cumulative_closure_residual_j,"
           "diffusion_moist_energy_cumulative_complete,"
           "diffusion_moist_energy_step_count,"
           "eos_pressure_energy_report_schema_version,"
           "eos_pressure_energy_basis_version,"
           "eos_pressure_energy_step_dry_internal_change_j,"
           "eos_pressure_energy_step_water_sensible_change_j,"
           "eos_pressure_energy_step_phase_reference_change_j,"
           "eos_pressure_energy_step_carrier_kinetic_change_j,"
           "eos_pressure_energy_step_gravitational_change_j,"
           "eos_pressure_energy_step_subgrid_tke_change_j,"
           "eos_pressure_energy_step_committed_energy_change_j,"
           "eos_pressure_energy_step_closure_residual_j,"
           "eos_pressure_energy_step_complete,"
           "eos_pressure_energy_cumulative_dry_internal_change_j,"
           "eos_pressure_energy_cumulative_water_sensible_change_j,"
           "eos_pressure_energy_cumulative_phase_reference_change_j,"
           "eos_pressure_energy_cumulative_carrier_kinetic_change_j,"
           "eos_pressure_energy_cumulative_gravitational_change_j,"
           "eos_pressure_energy_cumulative_subgrid_tke_change_j,"
           "eos_pressure_energy_cumulative_committed_energy_change_j,"
           "eos_pressure_energy_cumulative_closure_residual_j,"
           "eos_pressure_energy_cumulative_complete,"
           "eos_pressure_energy_step_count";
    header
        << ",sponge_energy_report_schema_version,"
           "sponge_energy_basis_version";
    for (const char* scope : {"step", "cumulative"})
        for (const char* pass : {"upper", "w_only", "lateral"})
        {
            header
                << ",sponge_energy_" << scope << '_' << pass
                << "_momentum_requested_linear_work_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_momentum_finite_step_correction_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_momentum_storage_rounding_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_dry_internal_change_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_water_sensible_change_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_phase_reference_change_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_carrier_kinetic_change_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_gravitational_change_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_subgrid_tke_change_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_committed_energy_change_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_closure_residual_j"
                << ",sponge_energy_" << scope << '_' << pass
                << "_applied"
                << ",sponge_energy_" << scope << '_' << pass
                << "_complete";
        }
    header
        << ",sponge_energy_step_complete,"
           "sponge_energy_cumulative_complete,"
           "sponge_energy_step_count,"
           "final_boundary_energy_report_schema_version,"
           "final_boundary_energy_basis_version";
    for (const char* scope : {"step", "cumulative"})
    {
        header
            << ",final_boundary_energy_" << scope
            << "_constraint_committed_work_j"
            << ",final_boundary_energy_" << scope
            << "_dry_internal_change_j"
            << ",final_boundary_energy_" << scope
            << "_water_sensible_change_j"
            << ",final_boundary_energy_" << scope
            << "_phase_reference_change_j"
            << ",final_boundary_energy_" << scope
            << "_carrier_kinetic_change_j"
            << ",final_boundary_energy_" << scope
            << "_gravitational_change_j"
            << ",final_boundary_energy_" << scope
            << "_subgrid_tke_change_j"
            << ",final_boundary_energy_" << scope
            << "_committed_energy_change_j"
            << ",final_boundary_energy_" << scope
            << "_closure_residual_j"
            << ",final_boundary_energy_" << scope
            << "_applied"
            << ",final_boundary_energy_" << scope
            << "_complete";
    }
    header
        << ",final_boundary_energy_step_count,"
           "completed_step_energy_report_schema_version,"
           "completed_step_energy_basis_version";
    for (const char* scope : {"step", "cumulative"})
    {
        const char* fields[] = {
            "acoustic_committed_change_j",
            "slow_dynamics_committed_change_j",
            "scalar_transport_committed_change_j",
            "thermodynamic_sources_committed_change_j",
            "sgs_committed_change_j",
            "diffusion_committed_change_j",
            "eos_projection_committed_change_j",
            "sponge_committed_change_j",
            "final_boundary_committed_change_j",
            "unsplit_rk_committed_change_j",
            "coordinate_remap_committed_change_j",
            "committed_owner_sum_j",
            "dry_internal_change_j",
            "water_sensible_change_j",
            "phase_reference_change_j",
            "carrier_kinetic_change_j",
            "gravitational_change_j",
            "subgrid_tke_change_j",
            "closure_residual_j",
            "complete",
        };
        for (const char* field : fields)
            header
                << ",completed_step_energy_" << scope << '_' << field;
    }
    header
        << ",completed_step_energy_count,"
           "dynamics_outward_boundary_dry_mass_kg,"
           "dynamics_flux_implied_dry_mass_change_kg,"
           "dynamics_committed_dry_mass_change_kg,"
           "dynamics_dry_mass_closure_residual_kg,"
           "dynamics_cumulative_outward_boundary_dry_mass_kg,"
           "dynamics_cumulative_flux_implied_dry_mass_change_kg,"
           "dynamics_cumulative_committed_dry_mass_change_kg,"
           "dynamics_cumulative_dry_mass_closure_residual_kg,"
           "dynamics_boundary_mass_step_count,"
           "acoustic_report_schema_version,"
           "acoustic_energy_basis_version,"
           "acoustic_v1_dry_internal_change_j,"
           "acoustic_v1_water_sensible_change_j,"
           "acoustic_v1_phase_reference_change_j,"
           "acoustic_v1_carrier_kinetic_change_j,"
           "acoustic_v1_gravitational_change_j,"
           "acoustic_v1_subgrid_tke_change_j,"
           "acoustic_v1_pressure_requested_linear_work_j,"
           "acoustic_v1_pressure_finite_step_correction_j,"
           "acoustic_v1_pressure_storage_rounding_j,"
           "acoustic_v1_pressure_committed_work_j,"
           "acoustic_v1_boundary_committed_work_j,"
           "acoustic_v1_boundary_radiative_requested_linear_work_j,"
           "acoustic_v1_boundary_radiative_finite_step_correction_j,"
           "acoustic_v1_boundary_radiative_storage_rounding_j,"
           "acoustic_v1_boundary_radiative_committed_work_j,"
           "acoustic_v1_boundary_constraint_history_committed_work_j,"
           "acoustic_v1_boundary_application_count,"
           "acoustic_v1_boundary_complete,"
           "acoustic_v1_committed_inventory_change_j,"
           "acoustic_v1_substep_count,"
           "acoustic_v1_complete,"
           "acoustic_v1_continuity_carrier_requested_change_j,"
           "acoustic_v1_continuity_carrier_outward_boundary_transport_j,"
           "acoustic_v1_continuity_carrier_interior_conversion_j,"
           "acoustic_v1_continuity_carrier_storage_rounding_j,"
           "acoustic_v1_slow_momentum_advection_requested_linear_work_j,"
           "acoustic_v1_kinetic_transport_interior_conversion_j,"
           "acoustic_v1_kinetic_transport_closure_residual_j,"
           "acoustic_v1_kinetic_transport_complete,"
           "acoustic_v1_cumulative_dry_internal_change_j,"
           "acoustic_v1_cumulative_water_sensible_change_j,"
           "acoustic_v1_cumulative_phase_reference_change_j,"
           "acoustic_v1_cumulative_carrier_kinetic_change_j,"
           "acoustic_v1_cumulative_gravitational_change_j,"
           "acoustic_v1_cumulative_subgrid_tke_change_j,"
           "acoustic_v1_cumulative_pressure_requested_linear_work_j,"
           "acoustic_v1_cumulative_pressure_finite_step_correction_j,"
           "acoustic_v1_cumulative_pressure_storage_rounding_j,"
           "acoustic_v1_cumulative_pressure_committed_work_j,"
           "acoustic_v1_cumulative_boundary_committed_work_j,"
           "acoustic_v1_cumulative_boundary_radiative_requested_linear_work_j,"
           "acoustic_v1_cumulative_boundary_radiative_finite_step_correction_j,"
           "acoustic_v1_cumulative_boundary_radiative_storage_rounding_j,"
           "acoustic_v1_cumulative_boundary_radiative_committed_work_j,"
           "acoustic_v1_cumulative_boundary_constraint_history_committed_work_j,"
           "acoustic_v1_cumulative_boundary_application_count,"
           "acoustic_v1_cumulative_boundary_complete,"
           "acoustic_v1_cumulative_committed_inventory_change_j,"
           "acoustic_v1_cumulative_substep_count,"
           "acoustic_v1_cumulative_complete,"
           "acoustic_v1_cumulative_continuity_carrier_requested_change_j,"
           "acoustic_v1_cumulative_continuity_carrier_outward_boundary_transport_j,"
           "acoustic_v1_cumulative_continuity_carrier_interior_conversion_j,"
           "acoustic_v1_cumulative_continuity_carrier_storage_rounding_j,"
           "acoustic_v1_cumulative_slow_momentum_advection_requested_linear_work_j,"
           "acoustic_v1_cumulative_kinetic_transport_interior_conversion_j,"
           "acoustic_v1_cumulative_kinetic_transport_closure_residual_j,"
           "acoustic_v1_cumulative_kinetic_transport_complete,"
           "acoustic_v1_step_count,"
           "slow_momentum_report_schema_version,"
           "slow_momentum_energy_basis_version";
    for (const char* cadence : {"step", "cumulative"})
    {
        header
            << ",slow_momentum_energy_" << cadence
            << "_advection_requested_linear_work_j"
            << ",slow_momentum_energy_" << cadence
            << "_nonadvection_requested_linear_work_j"
            << ",slow_momentum_energy_" << cadence
            << "_advection_finite_step_correction_j"
            << ",slow_momentum_energy_" << cadence
            << "_nonadvection_finite_step_correction_j"
            << ",slow_momentum_energy_" << cadence
            << "_coupled_finite_step_cross_work_j"
            << ",slow_momentum_energy_" << cadence
            << "_storage_rounding_j"
            << ",slow_momentum_energy_" << cadence
            << "_committed_kinetic_change_j"
            << ",slow_momentum_energy_" << cadence
            << "_closure_residual_j"
            << ",slow_momentum_energy_" << cadence
            << "_pressure_dry_internal_change_j"
            << ",slow_momentum_energy_" << cadence
            << "_pressure_water_sensible_change_j"
            << ",slow_momentum_energy_" << cadence
            << "_committed_pressure_energy_change_j"
            << ",slow_momentum_energy_" << cadence
            << "_complete";
    }
    header
        << ",slow_momentum_energy_step_count";
    for (std::size_t scalar = 0; scalar < k_transport_scalars.size(); ++scalar)
    {
        const char* name = k_transport_scalars[scalar];
        const char* unit = scalar == tmv::theta_boundary_report_index
            ? "kg_k" : "kg";
        header
            << ",advection_" << name << "_outward_boundary_" << unit
            << ",advection_" << name << "_flux_implied_change_" << unit
            << ",advection_" << name << "_committed_change_" << unit
            << ",advection_" << name << "_closure_residual_" << unit
            << ",advection_" << name << "_cumulative_outward_boundary_"
            << unit
            << ",advection_" << name << "_cumulative_flux_implied_change_"
            << unit
            << ",advection_" << name << "_cumulative_committed_change_"
            << unit
            << ",advection_" << name << "_cumulative_closure_residual_"
            << unit;
    }
    header
        << ",advection_thermodynamic_boundary_reports_complete,"
           "advection_thermodynamic_boundary_report_step_count,"
           "advection_scalar_energy_report_schema_version,"
           "advection_scalar_energy_basis_version";
    for (const char* name : k_transport_scalars)
    {
        for (const char* cadence : {"step", "cumulative"})
        {
            header
                << ",advection_" << name << "_energy_" << cadence
                << "_dry_internal_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_water_sensible_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_phase_reference_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_carrier_kinetic_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_gravitational_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_subgrid_tke_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_density_history_remap_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_outward_boundary_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_interior_conversion_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_limited_face_state_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_requested_state_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_storage_rounding_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_committed_state_change_j"
                << ",advection_" << name << "_energy_" << cadence
                << "_closure_residual_j";
        }
    }
    header
        << ",advection_thermodynamic_moist_energy_reports_complete,"
           "advection_thermodynamic_moist_energy_report_step_count\n";
    if (!g_source_budget_files.open(path, header.str(), 17))
        tmv::log_warn("[SOURCE BUDGET CSV] failed to open '", path,
                      "'; source provenance disabled");
}

void close_thermodynamic_source_budget_csv()
{
    std::lock_guard<std::mutex> lock(g_source_budget_csv_mutex);
    g_source_budget_files.close();
}

void write_thermodynamic_source_budget_row(
    const SimulationStateView& state, double current_time_s, double dt_s)
{
    std::lock_guard<std::mutex> lock(g_source_budget_csv_mutex);
    auto* selected = g_source_budget_files.stream(state.budget_level, g_source_budget_csv_mutex);
    if (!selected) return;
    auto& g_source_budget_csv = *selected;

    const MicrophysicsRuntimeState& runtime = state.microphysics_runtime
        ? *state.microphysics_runtime : global_microphysics_runtime_state;
    const auto& thermo = runtime.last_thermodynamic_repair_report;
    const auto& water = runtime.last_water_repair_report;
    const auto& conservation = runtime.last_conservation_report;
    const BoundaryLayerRuntimeState* pbl = state.boundary_layer_runtime;
    const BoundaryLayerMomentumWorkReport pbl_work = pbl
        ? pbl->last_momentum_work : BoundaryLayerMomentumWorkReport{};
    const TurbulenceRuntimeState* turbulence = state.turbulence_runtime;
    const TurbulenceMomentumWorkReport sgs_work = turbulence
        ? turbulence->last_momentum_work : TurbulenceMomentumWorkReport{};
    const TurbulenceMomentumWorkReport cumulative_sgs_energy = turbulence
        ? turbulence->cumulative_moist_energy_work
        : TurbulenceMomentumWorkReport{};
    const DiffusionRuntimeState* diffusion = state.diffusion_runtime;
    const DiffusionMomentumWorkReport diffusion_work = diffusion
        ? diffusion->last_momentum_work : DiffusionMomentumWorkReport{};
    const DiffusionMomentumWorkReport cumulative_diffusion_energy = diffusion
        ? diffusion->cumulative_moist_energy_work
        : DiffusionMomentumWorkReport{};
    const TimeSteppingRuntimeState* time_stepping =
        state.time_stepping_runtime;
    const AdvectionRuntimeState* advection = state.advection_runtime;
    const DynamicsBoundaryMassReport boundary_mass = time_stepping
        ? time_stepping->last_boundary_mass : DynamicsBoundaryMassReport{};
    const double requested_heat = thermo.requested_heat_j();
    const double theta_repair = thermo.net_theta_repair_j();
    const double committed_water =
        thermo.microphysics_water_tendency_kg
        + thermo.boundary_layer_vapor_tendency_kg
        + thermo.boundary_layer_cloud_liquid_tendency_kg
        + thermo.boundary_layer_cloud_ice_tendency_kg
        + water.storage_rounding_kg() + water.net_correction_kg();

    g_source_budget_csv
        << current_time_s + dt_s << ',' << dt_s << ','
        << thermo.microphysics_heat_j << ',' << thermo.radiation_heat_j << ','
        << thermo.boundary_layer_heat_j << ',' << requested_heat << ','
        << theta_repair << ',' << thermo.theta_storage_rounding_j << ','
        << thermo.theta_storage_residual_j << ',' << thermo.committed_heat_j()
        << ',' << thermo.state_heat_change_j() << ','
        << MicrophysicsThermodynamicRepairReport::report_schema_version << ','
        << MicrophysicsThermodynamicRepairReport::energy_basis_version << ','
        << thermo.moist_dry_internal_change_j << ','
        << thermo.moist_water_sensible_change_j << ','
        << thermo.moist_phase_reference_change_j << ','
        << thermo.moist_carrier_kinetic_change_j << ','
        << thermo.moist_gravitational_change_j << ','
        << thermo.moist_subgrid_tke_change_j << ','
        << thermo.moist_committed_energy_change_j << ','
        << thermo.moist_energy_closure_residual_j << ','
        << (thermo.moist_energy_complete ? 1 : 0) << ','
        << thermo.microphysics_water_tendency_kg << ','
        << thermo.boundary_layer_vapor_tendency_kg << ','
        << thermo.boundary_layer_cloud_liquid_tendency_kg << ','
        << thermo.boundary_layer_cloud_ice_tendency_kg << ','
        << conservation.surface_precipitation_kg << ','
        << conservation.tendency_imbalance_kg << ','
        << (conservation.active ? 1 : 0) << ','
        << conservation.baseline_q_kg << ','
        << water.storage_rounding_kg() << ',' << water.net_correction_kg()
        << ',' << committed_water;
    for (std::size_t species = 0; species < k_water_species.size(); ++species)
    {
        const auto& repair = water.species[species];
        const double repair_kg = repair.lower_bound_correction_kg
            + repair.upper_bound_correction_kg
            + repair.nonfinite_correction_kg;
        const double requested_kg =
            thermo.microphysics_water_tendency_kg_by_species[species]
            + (species == 0 ? thermo.boundary_layer_vapor_tendency_kg : 0.0)
            + (species == 1
                ? thermo.boundary_layer_cloud_liquid_tendency_kg : 0.0)
            + (species == 3
                ? thermo.boundary_layer_cloud_ice_tendency_kg : 0.0);
        const double committed_species_kg = requested_kg
            + repair.storage_rounding_kg + repair_kg;
        g_source_budget_csv
            << ',' << thermo.microphysics_water_tendency_kg_by_species[species]
            << ',' << repair.storage_rounding_kg
            << ',' << repair_kg
            << ',' << committed_species_kg;
    }
    g_source_budget_csv
        << ','
        << pbl_work.requested_linear_work_j << ','
        << pbl_work.finite_step_correction_j << ','
        << pbl_work.storage_rounding_j << ','
        << pbl_work.committed_work_j << ','
        << (pbl ? pbl->cumulative_requested_momentum_work_j : 0.0) << ','
        << (pbl ? pbl->cumulative_finite_step_momentum_correction_j : 0.0)
        << ','
        << (pbl ? pbl->cumulative_momentum_storage_rounding_j : 0.0) << ','
        << (pbl ? pbl->cumulative_committed_momentum_work_j : 0.0) << ','
        << (pbl ? pbl->momentum_work_step_count : 0) << ','
        << sgs_work.requested_linear_work_j << ','
        << sgs_work.finite_step_correction_j << ','
        << sgs_work.storage_rounding_j << ','
        << sgs_work.committed_work_j << ','
        << (turbulence
                ? turbulence->cumulative_requested_momentum_work_j : 0.0)
        << ','
        << (turbulence
                ? turbulence->cumulative_finite_step_momentum_correction_j
                : 0.0) << ','
        << (turbulence
                ? turbulence->cumulative_momentum_storage_rounding_j : 0.0)
        << ','
        << (turbulence
                ? turbulence->cumulative_committed_momentum_work_j : 0.0)
        << ','
        << (turbulence ? turbulence->momentum_work_step_count : 0) << ','
        << (turbulence ? turbulence->last_open_boundary_work_j : 0.0)
        << ','
        << (turbulence ? turbulence->last_irreversible_dissipation_j : 0.0)
        << ','
        << (turbulence ? turbulence->last_stress_closure_residual_j : 0.0)
        << ','
        << (turbulence ? turbulence->cumulative_open_boundary_work_j : 0.0)
        << ','
        << (turbulence
                ? turbulence->cumulative_irreversible_dissipation_j : 0.0)
        << ','
        << (turbulence
                ? turbulence->cumulative_stress_closure_residual_j : 0.0)
        << ','
        << (turbulence
                ? turbulence->open_boundary_stability_step_count : 0)
        << ','
        << TurbulenceMomentumWorkReport::report_schema_version << ','
        << TurbulenceMomentumWorkReport::energy_basis_version << ','
        << sgs_work.moist_dry_internal_change_j << ','
        << sgs_work.moist_water_sensible_change_j << ','
        << sgs_work.moist_phase_reference_change_j << ','
        << sgs_work.moist_carrier_kinetic_change_j << ','
        << sgs_work.moist_gravitational_change_j << ','
        << sgs_work.moist_subgrid_tke_change_j << ','
        << sgs_work.moist_committed_energy_change_j << ','
        << sgs_work.moist_energy_closure_residual_j << ','
        << (sgs_work.moist_energy_complete ? 1 : 0) << ','
        << cumulative_sgs_energy.moist_dry_internal_change_j << ','
        << cumulative_sgs_energy.moist_water_sensible_change_j << ','
        << cumulative_sgs_energy.moist_phase_reference_change_j << ','
        << cumulative_sgs_energy.moist_carrier_kinetic_change_j << ','
        << cumulative_sgs_energy.moist_gravitational_change_j << ','
        << cumulative_sgs_energy.moist_subgrid_tke_change_j << ','
        << cumulative_sgs_energy.moist_committed_energy_change_j << ','
        << cumulative_sgs_energy.moist_energy_closure_residual_j << ','
        << (cumulative_sgs_energy.moist_energy_complete ? 1 : 0) << ','
        << diffusion_work.requested_linear_work_j << ','
        << diffusion_work.finite_step_correction_j << ','
        << diffusion_work.storage_rounding_j << ','
        << diffusion_work.committed_work_j << ','
        << (diffusion
                ? diffusion->cumulative_requested_momentum_work_j : 0.0)
        << ','
        << (diffusion
                ? diffusion->cumulative_finite_step_momentum_correction_j
                : 0.0) << ','
        << (diffusion
                ? diffusion->cumulative_momentum_storage_rounding_j : 0.0)
        << ','
        << (diffusion
                ? diffusion->cumulative_committed_momentum_work_j : 0.0)
        << ','
        << (diffusion ? diffusion->momentum_work_step_count : 0) << ','
        << DiffusionMomentumWorkReport::report_schema_version << ','
        << DiffusionMomentumWorkReport::energy_basis_version << ','
        << diffusion_work.moist_dry_internal_change_j << ','
        << diffusion_work.moist_water_sensible_change_j << ','
        << diffusion_work.moist_phase_reference_change_j << ','
        << diffusion_work.moist_carrier_kinetic_change_j << ','
        << diffusion_work.moist_gravitational_change_j << ','
        << diffusion_work.moist_subgrid_tke_change_j << ','
        << diffusion_work.moist_committed_energy_change_j << ','
        << diffusion_work.moist_energy_closure_residual_j << ','
        << (diffusion_work.moist_energy_complete ? 1 : 0) << ','
        << cumulative_diffusion_energy.moist_dry_internal_change_j << ','
        << cumulative_diffusion_energy.moist_water_sensible_change_j << ','
        << cumulative_diffusion_energy.moist_phase_reference_change_j << ','
        << cumulative_diffusion_energy.moist_carrier_kinetic_change_j << ','
        << cumulative_diffusion_energy.moist_gravitational_change_j << ','
        << cumulative_diffusion_energy.moist_subgrid_tke_change_j << ','
        << cumulative_diffusion_energy.moist_committed_energy_change_j << ','
        << cumulative_diffusion_energy.moist_energy_closure_residual_j << ','
        << (cumulative_diffusion_energy.moist_energy_complete ? 1 : 0)
        << ','
        << (diffusion ? diffusion->moist_energy_step_count : 0);
    const auto write_eos_pressure_energy =
        [](std::ostream& out,
           const dynamics_eos::PressureProjectionEnergyReportV1& report)
    {
        out
            << ',' << report.dry_internal_change_j
            << ',' << report.water_sensible_change_j
            << ',' << report.phase_reference_change_j
            << ',' << report.carrier_kinetic_change_j
            << ',' << report.gravitational_change_j
            << ',' << report.subgrid_tke_change_j
            << ',' << report.committed_energy_change_j
            << ',' << report.closure_residual_j
            << ',' << (report.complete ? 1 : 0);
    };
    g_source_budget_csv
        << ','
        << dynamics_eos::PressureProjectionEnergyReportV1::
               report_schema_version
        << ','
        << dynamics_eos::PressureProjectionEnergyReportV1::
               energy_basis_version;
    write_eos_pressure_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->last_eos_pressure_energy
            : dynamics_eos::PressureProjectionEnergyReportV1{});
    write_eos_pressure_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->cumulative_eos_pressure_energy
            : dynamics_eos::PressureProjectionEnergyReportV1{});
    g_source_budget_csv
        << ',' << (time_stepping
                       ? time_stepping->eos_pressure_energy_step_count : 0);
    const auto write_sponge_pass =
        [](std::ostream& out, const SpongePassEnergyReportV1& pass)
    {
        out
            << ',' << pass.momentum_requested_linear_work_j
            << ',' << pass.momentum_finite_step_correction_j
            << ',' << pass.momentum_storage_rounding_j
            << ',' << pass.dry_internal_change_j
            << ',' << pass.water_sensible_change_j
            << ',' << pass.phase_reference_change_j
            << ',' << pass.carrier_kinetic_change_j
            << ',' << pass.gravitational_change_j
            << ',' << pass.subgrid_tke_change_j
            << ',' << pass.committed_energy_change_j
            << ',' << pass.closure_residual_j
            << ',' << (pass.applied ? 1 : 0)
            << ',' << (pass.complete ? 1 : 0);
    };
    const auto write_sponge_energy =
        [&write_sponge_pass](
            std::ostream& out, const SpongeEnergyReportV1& report)
    {
        write_sponge_pass(out, report.upper);
        write_sponge_pass(out, report.w_only);
        write_sponge_pass(out, report.lateral);
    };
    g_source_budget_csv
        << ',' << SpongeEnergyReportV1::report_schema_version
        << ',' << SpongeEnergyReportV1::energy_basis_version;
    write_sponge_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->last_sponge_energy
            : SpongeEnergyReportV1{});
    write_sponge_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->cumulative_sponge_energy
            : SpongeEnergyReportV1{});
    g_source_budget_csv
        << ',' << (time_stepping
                       && time_stepping->last_sponge_energy.complete ? 1 : 0)
        << ',' << (time_stepping
                       && time_stepping
                              ->cumulative_sponge_energy.complete ? 1 : 0)
        << ',' << (time_stepping
                       ? time_stepping->sponge_energy_step_count : 0)
        << ',' << FinalBoundaryEnergyReportV1::report_schema_version
        << ',' << FinalBoundaryEnergyReportV1::energy_basis_version;
    const auto write_final_boundary_energy =
        [](std::ostream& out,
           const FinalBoundaryEnergyReportV1& report)
    {
        out
            << ',' << report.constraint_committed_work_j
            << ',' << report.dry_internal_change_j
            << ',' << report.water_sensible_change_j
            << ',' << report.phase_reference_change_j
            << ',' << report.carrier_kinetic_change_j
            << ',' << report.gravitational_change_j
            << ',' << report.subgrid_tke_change_j
            << ',' << report.committed_energy_change_j
            << ',' << report.closure_residual_j
            << ',' << (report.applied ? 1 : 0)
            << ',' << (report.complete ? 1 : 0);
    };
    write_final_boundary_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->last_final_boundary_energy
            : FinalBoundaryEnergyReportV1{});
    write_final_boundary_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->cumulative_final_boundary_energy
            : FinalBoundaryEnergyReportV1{});
    g_source_budget_csv
        << ',' << (time_stepping
                       ? time_stepping->final_boundary_energy_step_count
                       : 0)
        << ',' << CompletedStepEnergyReportV1::report_schema_version
        << ',' << CompletedStepEnergyReportV1::energy_basis_version;
    const auto write_completed_step_energy =
        [](std::ostream& out,
           const CompletedStepEnergyReportV1& report)
    {
        out
            << ',' << report.acoustic_committed_change_j
            << ',' << report.slow_dynamics_committed_change_j
            << ',' << report.scalar_transport_committed_change_j
            << ',' << report.thermodynamic_sources_committed_change_j
            << ',' << report.sgs_committed_change_j
            << ',' << report.diffusion_committed_change_j
            << ',' << report.eos_projection_committed_change_j
            << ',' << report.sponge_committed_change_j
            << ',' << report.final_boundary_committed_change_j
            << ',' << report.unsplit_rk_committed_change_j
            << ',' << report.coordinate_remap_committed_change_j
            << ',' << report.committed_owner_sum_j
            << ',' << report.dry_internal_change_j
            << ',' << report.water_sensible_change_j
            << ',' << report.phase_reference_change_j
            << ',' << report.carrier_kinetic_change_j
            << ',' << report.gravitational_change_j
            << ',' << report.subgrid_tke_change_j
            << ',' << report.closure_residual_j
            << ',' << (report.complete ? 1 : 0);
    };
    write_completed_step_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->last_completed_step_energy
            : CompletedStepEnergyReportV1{});
    write_completed_step_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->cumulative_completed_step_energy
            : CompletedStepEnergyReportV1{});
    g_source_budget_csv
        << ',' << (time_stepping
                       ? time_stepping->completed_step_energy_count : 0)
        << ','
        << boundary_mass.outward_boundary_mass_kg << ','
        << boundary_mass.flux_implied_state_change_kg << ','
        << boundary_mass.committed_state_change_kg << ','
        << boundary_mass.closure_residual_kg << ','
        << (time_stepping
                ? time_stepping->cumulative_outward_boundary_mass_kg : 0.0)
        << ','
        << (time_stepping
                ? time_stepping->cumulative_flux_implied_state_change_kg
                : 0.0) << ','
        << (time_stepping
                ? time_stepping->cumulative_committed_state_change_kg : 0.0)
        << ','
        << (time_stepping
                ? time_stepping->cumulative_boundary_mass_closure_residual_kg
                : 0.0) << ','
        << (time_stepping ? time_stepping->boundary_mass_step_count : 0)
        << ',' << CartesianCGridAcousticEnergyReportV1::report_schema_version
        << ',' << CartesianCGridAcousticEnergyReportV1::basis_version;
    const auto write_acoustic_energy =
        [](std::ostream& out,
           const CartesianCGridAcousticEnergyReportV1& report)
    {
        out
            << ',' << report.continuity_eos_dry_internal_change_j
            << ',' << report.continuity_eos_water_sensible_change_j
            << ',' << report.continuity_eos_phase_reference_change_j
            << ',' << report.continuity_carrier_kinetic_change_j
            << ',' << report.continuity_gravitational_change_j
            << ',' << report.continuity_subgrid_tke_change_j
            << ',' << report.pressure_gradient_requested_linear_work_j
            << ',' << report.pressure_gradient_finite_step_correction_j
            << ',' << report.pressure_gradient_storage_rounding_j
            << ',' << report.pressure_gradient_committed_work_j
            << ',' << report.boundary_committed_work_j
            << ',' << report.boundary_radiative_requested_linear_work_j
            << ',' << report.boundary_radiative_finite_step_correction_j
            << ',' << report.boundary_radiative_storage_rounding_j
            << ',' << report.boundary_radiative_committed_work_j
            << ',' << report.boundary_constraint_history_committed_work_j
            << ',' << report.boundary_application_count
            << ',' << (report.boundary_complete ? 1 : 0)
            << ',' << report.committed_inventory_change_j
            << ',' << report.substep_count
            << ',' << (report.complete ? 1 : 0)
            << ',' << report.continuity_carrier_requested_change_j
            << ',' << report.continuity_carrier_outward_boundary_transport_j
            << ',' << report.continuity_carrier_interior_conversion_j
            << ',' << report.continuity_carrier_storage_rounding_j
            << ',' << report.slow_momentum_advection_requested_linear_work_j
            << ',' << report.kinetic_transport_interior_conversion_j
            << ',' << report.kinetic_transport_closure_residual_j
            << ',' << (report.kinetic_transport_complete ? 1 : 0);
    };
    write_acoustic_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->last_acoustic_energy
            : CartesianCGridAcousticEnergyReportV1{});
    write_acoustic_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->cumulative_acoustic_energy
            : CartesianCGridAcousticEnergyReportV1{});
    g_source_budget_csv
        << ',' << (time_stepping
                       ? time_stepping->acoustic_energy_step_count : 0)
        << ',' << CartesianSlowMomentumEnergyReportV1::report_schema_version
        << ',' << CartesianSlowMomentumEnergyReportV1::basis_version;
    const auto write_slow_momentum_energy =
        [](std::ostream& out,
           const CartesianSlowMomentumEnergyReportV1& report)
    {
        out
            << ',' << report.advection_requested_linear_work_j
            << ',' << report.nonadvection_requested_linear_work_j
            << ',' << report.advection_finite_step_correction_j
            << ',' << report.nonadvection_finite_step_correction_j
            << ',' << report.coupled_finite_step_cross_work_j
            << ',' << report.storage_rounding_j
            << ',' << report.committed_kinetic_change_j
            << ',' << report.closure_residual_j
            << ',' << report.pressure_dry_internal_change_j
            << ',' << report.pressure_water_sensible_change_j
            << ',' << report.committed_pressure_energy_change_j
            << ',' << (report.complete ? 1 : 0);
    };
    write_slow_momentum_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->last_slow_momentum_energy
            : CartesianSlowMomentumEnergyReportV1{});
    write_slow_momentum_energy(
        g_source_budget_csv,
        time_stepping
            ? time_stepping->cumulative_slow_momentum_energy
            : CartesianSlowMomentumEnergyReportV1{});
    g_source_budget_csv
        << ',' << (time_stepping
                       ? time_stepping->slow_momentum_energy_step_count : 0);
    for (std::size_t scalar = 0; scalar < k_transport_scalars.size(); ++scalar)
    {
        const ScalarBoundaryTransportReport step = advection
            ? advection->thermodynamic_boundary_reports[scalar]
            : ScalarBoundaryTransportReport{};
        const ScalarBoundaryTransportReport cumulative = advection
            ? advection->cumulative_thermodynamic_boundary_reports[scalar]
            : ScalarBoundaryTransportReport{};
        g_source_budget_csv
            << ',' << step.outward_boundary_inventory
            << ',' << step.flux_implied_state_change
            << ',' << step.committed_state_change
            << ',' << step.closure_residual
            << ',' << cumulative.outward_boundary_inventory
            << ',' << cumulative.flux_implied_state_change
            << ',' << cumulative.committed_state_change
            << ',' << cumulative.closure_residual;
    }
    g_source_budget_csv
        << ',' << (advection
                       && advection->thermodynamic_boundary_reports_complete
                       ? 1 : 0)
        << ',' << (advection
                       ? advection
                             ->thermodynamic_boundary_report_step_count
                       : 0)
        << ',' << ScalarMoistEnergyTransportReportV1::report_schema_version
        << ',' << ScalarMoistEnergyTransportReportV1::energy_basis_version;
    const auto write_scalar_energy =
        [](std::ostream& out,
           const ScalarMoistEnergyTransportReportV1& report)
    {
        out
            << ',' << report.dry_internal_change_j
            << ',' << report.water_sensible_change_j
            << ',' << report.phase_reference_change_j
            << ',' << report.carrier_kinetic_change_j
            << ',' << report.gravitational_change_j
            << ',' << report.subgrid_tke_change_j
            << ',' << report.density_history_remap_j
            << ',' << report.outward_boundary_energy_j
            << ',' << report.interior_conversion_j
            << ',' << report.limited_face_state_change_j
            << ',' << report.requested_state_change_j
            << ',' << report.storage_rounding_j
            << ',' << report.committed_state_change_j
            << ',' << report.closure_residual_j;
    };
    for (std::size_t scalar = 0; scalar < k_transport_scalars.size(); ++scalar)
    {
        write_scalar_energy(
            g_source_budget_csv,
            advection
                ? advection->thermodynamic_moist_energy_reports[scalar]
                : ScalarMoistEnergyTransportReportV1{});
        write_scalar_energy(
            g_source_budget_csv,
            advection
                ? advection
                      ->cumulative_thermodynamic_moist_energy_reports[scalar]
                : ScalarMoistEnergyTransportReportV1{});
    }
    g_source_budget_csv
        << ',' << (advection
                       && advection
                              ->thermodynamic_moist_energy_reports_complete
                       ? 1 : 0)
        << ',' << (advection
                       ? advection
                             ->thermodynamic_moist_energy_report_step_count
                       : 0)
        << '\n';
}

} // namespace tmv
