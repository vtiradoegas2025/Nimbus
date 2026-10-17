#include "microphysics/grid_geometry.hpp"
#include "core/field/field_inventory.hpp"
// Microphysics tendency preparation, guarded application, and repair ledgers.

#include "core/runtime/runtime_config.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/field/field_bounds.hpp"
#include "core/field/numerical_integrity.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#ifdef NIMBUS_ALLOCATION_TELEMETRY
#include "core/infra/allocation_telemetry.hpp"
#endif
#include "diagnostics/diagnostic_cell_volume.hpp"
#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "util/log.hpp"
#include "util/parallel_copy.hpp"
#include "microphysics/factory.hpp"
#include "numerics/advection/advection.hpp"
#include "radiation/runtime.hpp"
#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "terrain/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace tmv
{

double MicrophysicsWaterRepairReport::lower_bound_correction_kg() const
{
    double total = 0.0;
    for (const auto& field : species) { total += field.lower_bound_correction_kg; }
    return total;
}

double MicrophysicsWaterRepairReport::lower_bound_roundoff_correction_kg() const
{
    double total = 0.0;
    for (const auto& field : species)
        total += field.lower_bound_roundoff_correction_kg;
    return total;
}

double MicrophysicsWaterRepairReport::upper_bound_correction_kg() const
{
    double total = 0.0;
    for (const auto& field : species) { total += field.upper_bound_correction_kg; }
    return total;
}

double MicrophysicsWaterRepairReport::nonfinite_correction_kg() const
{
    double total = 0.0;
    for (const auto& field : species) { total += field.nonfinite_correction_kg; }
    return total;
}

double MicrophysicsWaterRepairReport::storage_rounding_kg() const
{
    double total = 0.0;
    for (const auto& field : species) { total += field.storage_rounding_kg; }
    return total;
}

double MicrophysicsWaterRepairReport::net_correction_kg() const
{
    return lower_bound_correction_kg() + upper_bound_correction_kg()
         + nonfinite_correction_kg();
}

std::size_t MicrophysicsWaterRepairReport::lower_bound_count() const
{
    std::size_t total = 0;
    for (const auto& field : species) { total += field.lower_bound_count; }
    return total;
}

std::size_t MicrophysicsWaterRepairReport::lower_bound_roundoff_count() const
{
    std::size_t total = 0;
    for (const auto& field : species)
        total += field.lower_bound_roundoff_count;
    return total;
}

std::size_t MicrophysicsWaterRepairReport::upper_bound_count() const
{
    std::size_t total = 0;
    for (const auto& field : species) { total += field.upper_bound_count; }
    return total;
}

std::size_t MicrophysicsWaterRepairReport::nonfinite_count() const
{
    std::size_t total = 0;
    for (const auto& field : species) { total += field.nonfinite_count; }
    return total;
}

struct MicrophysicsSubcycleWorkspace
{
    std::array<Field3D, 8> primary;
    Field3D protected_primary_wind;
    std::array<Field3D, 8> coupling;
    MicrophysicsStepState microphysics;
    std::unique_ptr<chaos::ChaosRuntimeState> chaos;
    std::tuple<int, int, int, std::array<bool, AdvectionRuntimeState::hydrometeor_count>, bool> advection;
    std::array<std::vector<float>, 3> precipitation;
    std::vector<double> rate_integral;
    double precipitation_time = 0.0;
    bool reaction_complete = false;
    const MicrophysicsScheme* reaction_owner = nullptr;
    double reaction_end_s = 0.0;
    double reaction_interval_s = 0.0;
    tmv::domain::HorizontalPoint reaction_origin;
};

namespace
{

using MicrophysicsPerfClock = std::chrono::steady_clock;
MicrophysicsPerfSnapshot microphysics_perf_totals;

MicrophysicsPerfClock::time_point microphysics_perf_start()
{
    return global_perf_timing_enabled
        ? MicrophysicsPerfClock::now()
        : MicrophysicsPerfClock::time_point{};
}

void microphysics_perf_finish(
    double& accumulator,
    MicrophysicsPerfClock::time_point started)
{
    if (global_perf_timing_enabled)
    {
        accumulator += std::chrono::duration<double>(
            MicrophysicsPerfClock::now() - started).count();
    }
}

constexpr std::array<const char*, MicrophysicsWaterRepairReport::species_count>
    water_species_names{{"qv", "qc", "qr", "qi", "qs", "qg", "qh"}};

void merge_water_repair_report(MicrophysicsWaterRepairReport& destination,
                               const MicrophysicsWaterRepairReport& source)
{
    for (std::size_t n = 0; n < destination.species.size(); ++n)
    {
        auto& dst = destination.species[n];
        const auto& src = source.species[n];
        dst.storage_rounding_kg += src.storage_rounding_kg;
        dst.lower_bound_correction_kg += src.lower_bound_correction_kg;
        dst.lower_bound_roundoff_correction_kg +=
            src.lower_bound_roundoff_correction_kg;
        dst.upper_bound_correction_kg += src.upper_bound_correction_kg;
        dst.nonfinite_correction_kg += src.nonfinite_correction_kg;
        dst.lower_bound_count += src.lower_bound_count;
        dst.lower_bound_roundoff_count += src.lower_bound_roundoff_count;
        dst.upper_bound_count += src.upper_bound_count;
        dst.nonfinite_count += src.nonfinite_count;
    }
    destination.nonfinite_tendency_count += source.nonfinite_tendency_count;
}

void merge_thermodynamic_repair_report(
    MicrophysicsThermodynamicRepairReport& destination,
    const MicrophysicsThermodynamicRepairReport& source)
{
    destination.microphysics_heat_j += source.microphysics_heat_j;
    destination.radiation_heat_j += source.radiation_heat_j;
    destination.boundary_layer_heat_j += source.boundary_layer_heat_j;
    destination.theta_tendency_limit_correction_j +=
        source.theta_tendency_limit_correction_j;
    destination.theta_nonfinite_correction_j +=
        source.theta_nonfinite_correction_j;
    destination.theta_storage_rounding_j += source.theta_storage_rounding_j;
    destination.theta_storage_residual_release_j +=
        source.theta_storage_residual_release_j;
    destination.theta_storage_residual_j += source.theta_storage_residual_j;
    destination.theta_bound_correction_j += source.theta_bound_correction_j;
    destination.microphysics_water_tendency_kg +=
        source.microphysics_water_tendency_kg;
    for (std::size_t species = 0;
         species < destination.microphysics_water_tendency_kg_by_species.size();
         ++species)
    {
        destination.microphysics_water_tendency_kg_by_species[species] +=
            source.microphysics_water_tendency_kg_by_species[species];
    }
    destination.boundary_layer_vapor_tendency_kg +=
        source.boundary_layer_vapor_tendency_kg;
    destination.boundary_layer_cloud_liquid_tendency_kg +=
        source.boundary_layer_cloud_liquid_tendency_kg;
    destination.boundary_layer_cloud_ice_tendency_kg +=
        source.boundary_layer_cloud_ice_tendency_kg;
    destination.moist_dry_internal_change_j +=
        source.moist_dry_internal_change_j;
    destination.moist_water_sensible_change_j +=
        source.moist_water_sensible_change_j;
    destination.moist_phase_reference_change_j +=
        source.moist_phase_reference_change_j;
    destination.moist_carrier_kinetic_change_j +=
        source.moist_carrier_kinetic_change_j;
    destination.moist_gravitational_change_j +=
        source.moist_gravitational_change_j;
    destination.moist_subgrid_tke_change_j +=
        source.moist_subgrid_tke_change_j;
    destination.moist_committed_energy_change_j +=
        source.moist_committed_energy_change_j;
    destination.theta_tendency_limit_count +=
        source.theta_tendency_limit_count;
    destination.theta_nonfinite_count += source.theta_nonfinite_count;
    destination.theta_lower_bound_count += source.theta_lower_bound_count;
    destination.theta_upper_bound_count += source.theta_upper_bound_count;
}

double moist_basis_v2_specific_kinetic_energy(
    const SimulationStateView& state,
    const TerrainMetrics3D* terrain_metrics,
    int physical_k_begin,
    int i, int j, int k)
{
    const auto& fields = state.fields;
    // Direct one-/two-cell physics tests do not contain the lateral closure
    // storage required by a production C grid. In that degenerate domain the
    // three supplied velocity samples are necessarily collocated.
    if (!state.grid.geometry.staggered || i == 0 || j == 0)
    {
        const double u = static_cast<double>(fields.u[i][j][k]);
        const double v = static_cast<double>(fields.v[i][j][k]);
        const double w = static_cast<double>(fields.w[i][j][k]);
        return 0.5 * (u * u + v * v + w * w);
    }

    const double u_w = static_cast<double>(fields.u[i - 1][j][k]);
    const double u_e = static_cast<double>(fields.u[i][j][k]);
    const double v_s = static_cast<double>(fields.v[i][j - 1][k]);
    const double v_n = static_cast<double>(fields.v[i][j][k]);
    double w_b = k == physical_k_begin
        ? 0.0 : static_cast<double>(fields.w[i][j][k - 1]);
    if (k == physical_k_begin && terrain_metrics != nullptr)
    {
        const double ground_u = 0.5 * (u_w + u_e);
        const double ground_v = 0.5 * (v_s + v_n);
        const double frame_u_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).u_ms;
        const double frame_v_ms = tmv::physical_grid_metrics::surface_velocity_ms(state).v_ms;
        w_b = (ground_u - frame_u_ms)
                * (-terrain_metrics->Gx_face(i, j, physical_k_begin))
            + (ground_v - frame_v_ms)
                * (-terrain_metrics->Gy_face(i, j, physical_k_begin));
    }
    const double w_t = static_cast<double>(fields.w[i][j][k]);
    return 0.25 * (
        u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n
        + w_b * w_b + w_t * w_t);
}

bool moist_basis_v2_commit_inputs_available(
    const SimulationStateView& state)
{
    const auto matches = [&](const Field3D& field)
    {
        return field.size_r() == state.grid.nr
            && field.size_th() == state.grid.nth
            && field.size_z() == state.grid.nz;
    };
    return matches(state.fields.u)
        && matches(state.fields.v)
        && matches(state.fields.w)
        && matches(state.fields.tke);
}

bool observe_moist_basis_v2_commit(
    MicrophysicsThermodynamicRepairReport& report,
    double dry_mass,
    double exner,
    double specific_kinetic,
    double z,
    double theta_old,
    const std::array<double, 7>& water_old,
    double theta_new,
    const std::array<double, 7>& water_new)
{
    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i =
        physical_constants::specific_heat_ice_jkgk;
    constexpr double t_ref =
        physical_constants::freezing_temperature_k;
    constexpr double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
    constexpr double ice_reference =
        -physical_constants::latent_heat_fusion_jkg;
    const double temperature_old = theta_old * exner;
    const double temperature_new = theta_new * exner;
    const double old_liquid = water_old[1] + water_old[2];
    const double new_liquid = water_new[1] + water_new[2];
    const double old_ice =
        water_old[3] + water_old[4] + water_old[5] + water_old[6];
    const double new_ice =
        water_new[3] + water_new[4] + water_new[5] + water_new[6];
    const double old_total = water_old[0] + old_liquid + old_ice;
    const double new_total = water_new[0] + new_liquid + new_ice;
    const double dry_internal = dry_mass * cv_d
        * (temperature_new - temperature_old);
    const double water_sensible = dry_mass * (
        (temperature_new - t_ref)
            * (water_new[0] * cv_v + new_liquid * c_l + new_ice * c_i)
        - (temperature_old - t_ref)
            * (water_old[0] * cv_v + old_liquid * c_l + old_ice * c_i));
    const double phase_reference = dry_mass * (
        (water_new[0] - water_old[0]) * vapor_reference
        + (new_ice - old_ice) * ice_reference);
    const double carrier_kinetic =
        dry_mass * (new_total - old_total) * specific_kinetic;
    const double gravitational =
        dry_mass * (new_total - old_total)
        * physical_constants::gravity_ms2 * z;
    constexpr double subgrid_tke = 0.0;
    const double committed = dry_internal + water_sensible
        + phase_reference + carrier_kinetic + gravitational + subgrid_tke;
    const double values[] = {
        dry_mass, exner,
        temperature_old, temperature_new, old_total, new_total,
        specific_kinetic, z, dry_internal, water_sensible,
        phase_reference, carrier_kinetic, gravitational, committed,
    };
    if (std::any_of(
            std::begin(values), std::end(values),
            [](double value) { return !std::isfinite(value); })
        || dry_mass <= 0.0 || exner <= 0.0
        || temperature_old <= 0.0 || temperature_new <= 0.0
        || old_total < 0.0 || new_total < 0.0)
    {
        return false;
    }

    report.moist_dry_internal_change_j += dry_internal;
    report.moist_water_sensible_change_j += water_sensible;
    report.moist_phase_reference_change_j += phase_reference;
    report.moist_carrier_kinetic_change_j += carrier_kinetic;
    report.moist_gravitational_change_j += gravitational;
    report.moist_subgrid_tke_change_j += subgrid_tke;
    report.moist_committed_energy_change_j += committed;
    return true;
}

void finalize_moist_basis_v2_commit(
    MicrophysicsThermodynamicRepairReport& report,
    const char* owner)
{
    if (!std::isfinite(report.moist_committed_energy_change_j))
        throw std::runtime_error(
            std::string(owner)
            + " produced invalid basis-v2 moist-energy attribution");
    const double reservoir_sum =
        report.moist_dry_internal_change_j
        + report.moist_water_sensible_change_j
        + report.moist_phase_reference_change_j
        + report.moist_carrier_kinetic_change_j
        + report.moist_gravitational_change_j
        + report.moist_subgrid_tke_change_j;
    report.moist_committed_energy_change_j = reservoir_sum;
    report.moist_energy_closure_residual_j = 0.0;
    const double values[] = {
        report.moist_dry_internal_change_j,
        report.moist_water_sensible_change_j,
        report.moist_phase_reference_change_j,
        report.moist_carrier_kinetic_change_j,
        report.moist_gravitational_change_j,
        report.moist_subgrid_tke_change_j,
        report.moist_committed_energy_change_j,
        report.moist_energy_closure_residual_j,
    };
    if (std::any_of(
            std::begin(values), std::end(values),
            [](double value) { return !std::isfinite(value); }))
    {
        throw std::runtime_error(
            std::string(owner)
            + " produced invalid basis-v2 moist-energy attribution");
    }
    report.moist_energy_complete = true;
}

void close_microphysics_species_total(
    MicrophysicsThermodynamicRepairReport& report)
{
    report.microphysics_water_tendency_kg = 0.0;
    for (double species_tendency :
         report.microphysics_water_tendency_kg_by_species)
    {
        report.microphysics_water_tendency_kg += species_tendency;
    }
}

void observe_water_repair(MicrophysicsWaterRepairReport& report,
                          std::size_t species,
                          float old_value,
                          double exact_candidate,
                          float raw_value,
                          float bounded_value,
                          double kg_per_kgkg)
{
    auto& field = report.species[species];
    if (std::isfinite(exact_candidate) && std::isfinite(raw_value))
    {
        field.storage_rounding_kg +=
            (static_cast<double>(raw_value) - exact_candidate) * kg_per_kgkg;
    }
    if (bounded_value == raw_value)
    {
        return;
    }
    if (!std::isfinite(raw_value))
    {
        // A non-finite candidate has no meaningful raw mass. Measure the
        // actual state change against the last finite pre-apply value; when
        // that value is itself invalid, retain a zero reference and rely on
        // the explicit count to prevent silent qualification.
        const double reference = std::isfinite(old_value)
            ? static_cast<double>(old_value) : 0.0;
        field.nonfinite_correction_kg +=
            (static_cast<double>(bounded_value) - reference) * kg_per_kgkg;
        ++field.nonfinite_count;
    }
    else if (bounded_value > raw_value)
    {
        const double correction_kg =
            (static_cast<double>(bounded_value) - static_cast<double>(raw_value))
            * kg_per_kgkg;
        field.lower_bound_correction_kg += correction_kg;
        ++field.lower_bound_count;
        // The candidate is old + fp32_tendency * fp64_dt. Eight fp32
        // epsilons cover stored-operand and short process-sum cancellation
        // without admitting a physically material negative reservoir.
        const double scale = std::max(
            std::abs(static_cast<double>(old_value)),
            std::abs(static_cast<double>(raw_value)
                     - static_cast<double>(old_value)));
        const double roundoff_bound = 8.0
            * static_cast<double>(std::numeric_limits<float>::epsilon())
            * scale;
        if (static_cast<double>(raw_value) >= -roundoff_bound)
        {
            field.lower_bound_roundoff_correction_kg += correction_kg;
            ++field.lower_bound_roundoff_count;
        }
    }
    else if (bounded_value < raw_value)
    {
        field.upper_bound_correction_kg +=
            (static_cast<double>(bounded_value) - static_cast<double>(raw_value))
            * kg_per_kgkg;
        ++field.upper_bound_count;
    }
}

float bounded_water_raw_value(MicrophysicsWaterRepairReport& report,
                              std::size_t species,
                              float old_value,
                              double exact_candidate,
                              float raw_value,
                              double kg_per_kgkg,
                              float upper_bound)
{
    // Strict runs disable ordinary writer repairs, but this writer has already
    // projected and rejected every material correction transactionally. It
    // must still commit an admitted sub-ULP correction instead of publishing
    // the negative sample that the projection just ledgered.
    const bool writer_may_repair = primary_state_repairs_enabled
        || primary_state_repairs_fail_closed;
    const float bounded_value = !writer_may_repair
        ? raw_value
        : (!std::isfinite(raw_value)
               ? 0.0f : std::clamp(raw_value, 0.0f, upper_bound));
    observe_water_repair(report, species, old_value, exact_candidate, raw_value,
                         bounded_value, kg_per_kgkg);
    return bounded_value;
}

float bounded_water_value(MicrophysicsWaterRepairReport& report,
                          std::size_t species,
                          float old_value,
                          double increment,
                          double kg_per_kgkg,
                          float upper_bound)
{
    const double exact_candidate = static_cast<double>(old_value) + increment;
    return bounded_water_raw_value(
        report, species, old_value,
        exact_candidate, static_cast<float>(exact_candidate),
        kg_per_kgkg, upper_bound);
}

float projected_bounded_water_value(
    MicrophysicsWaterRepairReport& report,
    std::size_t species,
    float old_value,
    double increment,
    double kg_per_kgkg,
    float upper_bound)
{
    const double exact_candidate = static_cast<double>(old_value) + increment;
    const float raw_value = static_cast<float>(exact_candidate);
    const float bounded_value = !std::isfinite(raw_value)
        ? 0.0f : std::clamp(raw_value, 0.0f, upper_bound);
    observe_water_repair(report, species, old_value, exact_candidate, raw_value,
                         bounded_value, kg_per_kgkg);
    return bounded_value;
}

double live_water_mass_density(const SimulationStateView& state,
                               int i, int j, int k)
{
    return static_cast<double>(state.fields.rho[i][j][k]);
}

double boundary_layer_cloud_tendency(
    const SimulationStateView& state,
    const Field3D* tendency, int i, int j, int k) noexcept
{
    return state.boundary_layer_runtime != nullptr
            && !state.boundary_layer_runtime->config.clear_sky_only
            && tendency != nullptr
        ? static_cast<double>((*tendency)[i][j][k]) : 0.0;
}

std::size_t repair_report_accumulator_count(const PhysicalCellDomain& domain)
{
    // Repair ledgers are persistent scientific runtime state and are written
    // to checkpoints. Their grouping must therefore follow the physical
    // domain, independent of the OpenMP team size and of the optional
    // diagnostics-reduction policy. One retained report per i slab also keeps
    // parallel writes disjoint without allocating per cell.
    return static_cast<std::size_t>(domain.i_end - domain.i_begin);
}

std::size_t repair_report_accumulator_index(
    const PhysicalCellDomain& domain, int i, int j)
{
    (void)j;
    return static_cast<std::size_t>(i - domain.i_begin);
}

double theta_heat_factor(const SimulationStateView& state,
                         int i, int j, int k,
                         double cell_dV)
{
    const double pressure = static_cast<double>(state.fields.p[i][j][k]);
    const double exner = std::pow(
        pressure / microphysics_constants::p0,
        microphysics_constants::R_d / microphysics_constants::cp);
    return live_water_mass_density(state, i, j, k)
         * microphysics_constants::cp * exner * cell_dV;
}

double radiation_theta_delta(
    const SimulationStateView& state,
    const RadiationRuntimeState& runtime,
    int i, int j, int k,
    double dt_s,
    double heat_factor)
{
    if (runtime.config.scheme_id == "none")
    {
        return static_cast<double>(state.tendencies.dtheta_dt_rad[i][j][k])
             * dt_s;
    }
    if (runtime.held_cell_power_w.size_r() != state.grid.nr ||
        runtime.held_cell_power_w.size_th() != state.grid.nth ||
        runtime.held_cell_power_w.size_z() != state.grid.nz ||
        !std::isfinite(heat_factor) || heat_factor <= 0.0)
    {
        throw std::runtime_error(
            "active radiation is missing a valid held cell-power field");
    }
    const double power_w = static_cast<double>(
        runtime.held_cell_power_w[i][j][k]);
    if (!std::isfinite(power_w))
    {
        throw std::runtime_error(
            "active radiation held a non-finite cell power");
    }
    return power_w * dt_s / heat_factor;
}

float project_theta_update(
    MicrophysicsThermodynamicRepairReport& report,
    double theta_old,
    double theta_storage_residual_old,
    double* theta_storage_residual_new,
    double delta_microphysics,
    double delta_radiation,
    double delta_boundary_layer,
    double heat_factor,
    double max_step,
    bool force_bounds = false)
{
    if (theta_storage_residual_new) *theta_storage_residual_new = 0.0;
    const double finite_microphysics = std::isfinite(delta_microphysics)
        ? delta_microphysics : 0.0;
    const double finite_radiation = std::isfinite(delta_radiation)
        ? delta_radiation : 0.0;
    const double finite_boundary_layer = std::isfinite(delta_boundary_layer)
        ? delta_boundary_layer : 0.0;
    report.microphysics_heat_j += finite_microphysics * heat_factor;
    report.radiation_heat_j += finite_radiation * heat_factor;
    report.boundary_layer_heat_j += finite_boundary_layer * heat_factor;

    const double raw_delta =
        delta_microphysics + delta_radiation + delta_boundary_layer;
    double limited_delta = 0.0;
    if (std::isfinite(raw_delta))
    {
        limited_delta = std::clamp(raw_delta, -max_step, max_step);
        if (limited_delta != raw_delta)
        {
            report.theta_tendency_limit_correction_j +=
                (limited_delta - raw_delta) * heat_factor;
            ++report.theta_tendency_limit_count;
        }
    }
    else
    {
        // The magnitude of a NaN/Inf source is undefined. Close the finite
        // part of the request to the zero increment actually applied and keep
        // an explicit count so strict runs fail transactionally.
        report.theta_nonfinite_correction_j -=
            (finite_microphysics + finite_radiation + finite_boundary_layer)
            * heat_factor;
        ++report.theta_nonfinite_count;
    }

    double theta_candidate = theta_old + theta_storage_residual_old + limited_delta;
    if (theta_storage_residual_new && std::isfinite(theta_storage_residual_old))
    {
        report.theta_storage_residual_release_j +=
            theta_storage_residual_old * heat_factor;
    }
    if (!std::isfinite(theta_candidate))
    {
        const double replacement = std::isfinite(theta_old)
            ? theta_old : theta0;
        if (std::isfinite(theta_old))
        {
            // Express the rejected transaction from its finite operands.  A
            // direct replacement - theta_candidate subtraction would retain
            // Inf when overflow is the reason the candidate is non-finite.
            report.theta_nonfinite_correction_j +=
                (replacement - theta_old - limited_delta) * heat_factor;
        }
        theta_candidate = replacement;
        ++report.theta_nonfinite_count;
    }

    const float raw_theta = static_cast<float>(theta_candidate);
    if (std::isfinite(theta_candidate) && std::isfinite(raw_theta))
    {
        report.theta_storage_rounding_j +=
            (static_cast<double>(raw_theta) - theta_candidate) * heat_factor;
    }
    // A strict projection must remain independent of the writer repair
    // switch: strict orchestration deliberately disables writer mutation
    // before calling this routine.  Without force_bounds the preflight would
    // report no repair and allow the invalid value to be published.
    const float bounded = force_bounds
        ? (!std::isfinite(raw_theta)
               ? static_cast<float>(theta0)
               : std::clamp(raw_theta, theta_min_k, theta_max_k))
        : clamp_theta_k(raw_theta);
    if (bounded != raw_theta)
    {
        report.theta_bound_correction_j +=
            (static_cast<double>(bounded) - static_cast<double>(raw_theta))
            * heat_factor;
        if (bounded > raw_theta)
        {
            ++report.theta_lower_bound_count;
        }
        else
        {
            ++report.theta_upper_bound_count;
        }
    }
    else if (theta_storage_residual_new && std::isfinite(theta_candidate))
    {
        *theta_storage_residual_new =
            theta_candidate - static_cast<double>(bounded);
        report.theta_storage_residual_j +=
            *theta_storage_residual_new * heat_factor;
    }
    return bounded;
}

float bits_to_float(std::uint32_t bits)
{
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Public runtime selection for the optional device state commit. Scheme,
// guard-mode, geometry, and backend capability checks remain authoritative;
// this switch alone never implies that a dispatch is admissible.
bool gpu_mp_apply_enabled(const SimulationStateView& execution_state)
{
    return tmv::require_run_configuration(execution_state).gpu_apply_microphysics;
}

bool gpu_mp_apply_commit_admitted(const SimulationStateView& state)
{
    return primary_state_repairs_enabled &&
        (state.boundary_layer_runtime == nullptr || state.boundary_layer_runtime->config.clear_sky_only);
}

/**
 * GPU microphysics tendency application (GPU audit Phase 4.2).
 *
 * Dispatches the pointwise apply kernel for the eight thermodynamic state
 * fields and unpacks the guard-telemetry stats block into the caller's
 * counter variables so the [MICROPHYSICS GUARD] logging stays identical.
 * Under the parity harness the CPU apply loop replays from a snapshot and
 * the eight results are compared at the Pointwise tier; the counters then
 * come from the CPU loop (identical semantics, fp64 arithmetic).
 *
 * Returns false (state untouched) when the backend lacks the kernel or a
 * precondition fails - the caller runs the CPU loop.
 */
template <typename CpuApplyFn>
bool try_microphysics_apply_gpu(
    const SimulationStateView& state, double dt_micro, float max_theta_step,
    const Field3D& dtheta_dt, const Field3D& dqv_dt, const Field3D& dqc_dt,
    const Field3D& dqr_dt, const Field3D& dqi_dt, const Field3D& dqs_dt,
    const Field3D& dqg_dt, const Field3D& dqh_dt,
    const CpuApplyFn& cpu_apply,
    int& non_finite_tendency_sanitized_total,
    int& non_finite_theta_tendency_count,
    int& theta_tendency_limited_count,
    int& theta_bounds_clamp_count,
    std::vector<int>& clamp_count_by_k,
    std::vector<int>& clamp_count_by_i,
    int& first_clamp_i, int& first_clamp_j, int& first_clamp_k,
    double& first_clamp_theta_new, double& first_clamp_theta_bounded,
    double& first_clamp_dtheta_total, double& first_clamp_dtheta_limited)
{
    if (!gpu_mp_apply_commit_admitted(state))
    {
        // This shader publishes repair counts but always commits its bounded
        // candidate. Warn/strict require a non-mutating observation or the
        // FP64 transactional projection, so retain device tendency
        // acceleration and perform the admitted state commit on the CPU.
        return false;
    }
    auto& fields = state.fields;
    const auto& coupling = state.tendencies;
    const auto& grid = state.grid;
    auto& theta = fields.theta;
    auto& qv = fields.qv;
    auto& qc = fields.qc;
    auto& qr = fields.qr;
    auto& qi = fields.qi;
    auto& qs = fields.qs;
    auto& qg = fields.qg;
    auto& qh = fields.qh;
    const std::size_t count = theta.size();
    if (count == 0
        || coupling.dtheta_dt_rad.size() != count
        || coupling.dtheta_dt_pbl.size() != count
        || coupling.dqv_dt_pbl.size() != count)
    {
        return false;
    }

    std::vector<unsigned int> stats(
        static_cast<std::size_t>(10 + grid.nz + grid.nr), 0u);
    const auto try_gpu = [&]() {
        return dispatch_microphysics_apply_backend(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
            theta.data(), qv.data(), qc.data(), qr.data(),
            qi.data(), qs.data(), qg.data(), qh.data(),
            dtheta_dt.data(), dqv_dt.data(), dqc_dt.data(), dqr_dt.data(),
            dqi_dt.data(), dqs_dt.data(), dqg_dt.data(), dqh_dt.data(),
            coupling.dtheta_dt_rad.data(), coupling.dtheta_dt_pbl.data(),
            coupling.dqv_dt_pbl.data(), grid.nr, grid.nth, grid.nz,
            physical_scalar_domain(grid).i_begin,
            physical_scalar_domain(grid).i_end,
            physical_scalar_domain(grid).j_begin,
            physical_scalar_domain(grid).j_end,
            static_cast<float>(dt_micro), static_cast<float>(theta0),
            theta_min_k, theta_max_k, max_theta_step,
            qv_max_kgkg, hydrometeor_max_kgkg,
            stats.data(), static_cast<int>(stats.size()));
    };

    if (!nimbus::parity::sample("microphysics_apply"))
    {
        if (!try_gpu())
        {
            return false;
        }
        // Unpack the guard telemetry. stats[0] holds the in-shader
        // non-finite tendency reads; the driver's sanitize sweeps already
        // zeroed the arrays, so this normally adds zero on top.
        non_finite_tendency_sanitized_total += static_cast<int>(stats[0]);
        non_finite_theta_tendency_count = static_cast<int>(stats[1]);
        theta_tendency_limited_count = static_cast<int>(stats[2]);
        theta_bounds_clamp_count = static_cast<int>(stats[3]);
        for (int k = 0; k < grid.nz; ++k)
        {
            clamp_count_by_k[static_cast<std::size_t>(k)] =
                static_cast<int>(stats[static_cast<std::size_t>(10 + k)]);
        }
        for (int i = 0; i < grid.nr; ++i)
        {
            clamp_count_by_i[static_cast<std::size_t>(i)] =
                static_cast<int>(
                    stats[static_cast<std::size_t>(10 + grid.nz + i)]);
        }
        if (stats[4] != 0u)
        {
            const std::uint32_t flat = stats[5];
            first_clamp_k = static_cast<int>(
                flat % static_cast<std::uint32_t>(grid.nz));
            const std::uint32_t ij =
                flat / static_cast<std::uint32_t>(grid.nz);
            first_clamp_j = static_cast<int>(
                ij % static_cast<std::uint32_t>(grid.nth));
            first_clamp_i = static_cast<int>(
                ij / static_cast<std::uint32_t>(grid.nth));
            first_clamp_theta_new = static_cast<double>(bits_to_float(stats[6]));
            first_clamp_theta_bounded = static_cast<double>(bits_to_float(stats[7]));
            first_clamp_dtheta_total = static_cast<double>(bits_to_float(stats[8]));
            first_clamp_dtheta_limited = static_cast<double>(bits_to_float(stats[9]));
        }
        return true;
    }

    // Parity replay (GPU audit Phase 1): snapshot, GPU, restore,
    // CPU reference, compare, keep the GPU result.
    Field3D pre_theta = theta, pre_qv = qv, pre_qc = qc, pre_qr = qr;
    Field3D pre_qi = qi, pre_qs = qs, pre_qg = qg, pre_qh = qh;
    if (!try_gpu())
    {
        return false;
    }
    Field3D gpu_theta = theta, gpu_qv = qv, gpu_qc = qc, gpu_qr = qr;
    Field3D gpu_qi = qi, gpu_qs = qs, gpu_qg = qg, gpu_qh = qh;

    const std::size_t bytes = count * sizeof(float);
    std::memcpy(theta.data(), pre_theta.data(), bytes);
    std::memcpy(qv.data(), pre_qv.data(), bytes);
    std::memcpy(qc.data(), pre_qc.data(), bytes);
    std::memcpy(qr.data(), pre_qr.data(), bytes);
    std::memcpy(qi.data(), pre_qi.data(), bytes);
    std::memcpy(qs.data(), pre_qs.data(), bytes);
    std::memcpy(qg.data(), pre_qg.data(), bytes);
    std::memcpy(qh.data(), pre_qh.data(), bytes);

    cpu_apply();

    using nimbus::parity::Tier;
    const char* kn = "microphysics_apply";
    nimbus::parity::check_field(kn, "theta",
        theta.data(), gpu_theta.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qv",
        qv.data(), gpu_qv.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qc",
        qc.data(), gpu_qc.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qr",
        qr.data(), gpu_qr.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qi",
        qi.data(), gpu_qi.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qs",
        qs.data(), gpu_qs.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qg",
        qg.data(), gpu_qg.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);
    nimbus::parity::check_field(kn, "qh",
        qh.data(), gpu_qh.data(), grid.nr, grid.nth, grid.nz,
        Tier::Pointwise);

    std::memcpy(theta.data(), gpu_theta.data(), bytes);
    std::memcpy(qv.data(), gpu_qv.data(), bytes);
    std::memcpy(qc.data(), gpu_qc.data(), bytes);
    std::memcpy(qr.data(), gpu_qr.data(), bytes);
    std::memcpy(qi.data(), gpu_qi.data(), bytes);
    std::memcpy(qs.data(), gpu_qs.data(), bytes);
    std::memcpy(qg.data(), gpu_qg.data(), bytes);
    std::memcpy(qh.data(), gpu_qh.data(), bytes);
    return true;
}

MicrophysicsWaterRepairReport evaluate_projected_water_repairs(
    const SimulationStateView& state,
    double dt_micro,
    const MicrophysicsTendencyFields& tendencies,
    std::vector<MicrophysicsWaterRepairReport>& slab_reports)
{
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    const auto& coupling = state.tendencies;
    slab_reports.assign(
        repair_report_accumulator_count(domain),
        MicrophysicsWaterRepairReport{});

    // A deterministic report owns one complete i slab. Parallelizing the
    // outer index keeps every j/k contribution private while limiting scratch
    // to O(nx), rather than allocating one large report per horizontal cell.
    #pragma omp parallel for schedule(static)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            auto& report = slab_reports[
                repair_report_accumulator_index(domain, i, j)];
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_dV = diagnostic_cell_volume(state, i, j, k);
                const double kg_per_kgkg =
                    live_water_mass_density(state, i, j, k) * cell_dV;
                // Mirror bounded_water_value() in the consuming CPU apply
                // byte-for-byte: tendencies are promoted individually, the
                // increment is accumulated in fp64, and only the completed
                // candidate is rounded to fp32.  The former fp32 projection
                // could miss a tiny negative candidate that the strict,
                // repair-disabled apply then published.
                (void)projected_bounded_water_value(
                    report, 0, fields.qv[i][j][k],
                    (static_cast<double>(tendencies.dqv_dt[i][j][k])
                     + static_cast<double>(coupling.dqv_dt_pbl[i][j][k]))
                        * dt_micro,
                    kg_per_kgkg, qv_max_kgkg);
                (void)projected_bounded_water_value(
                    report, 1, fields.qc[i][j][k],
                    (static_cast<double>(tendencies.dqc_dt[i][j][k])
                     + boundary_layer_cloud_tendency(
                         state, coupling.dqc_dt_pbl, i, j, k)) * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                (void)projected_bounded_water_value(
                    report, 2, fields.qr[i][j][k],
                    static_cast<double>(tendencies.dqr_dt[i][j][k])
                        * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                (void)projected_bounded_water_value(
                    report, 3, fields.qi[i][j][k],
                    (static_cast<double>(tendencies.dqi_dt[i][j][k])
                     + boundary_layer_cloud_tendency(
                         state, coupling.dqi_dt_pbl, i, j, k)) * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                (void)projected_bounded_water_value(
                    report, 4, fields.qs[i][j][k],
                    static_cast<double>(tendencies.dqs_dt[i][j][k])
                        * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                (void)projected_bounded_water_value(
                    report, 5, fields.qg[i][j][k],
                    static_cast<double>(tendencies.dqg_dt[i][j][k])
                        * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                (void)projected_bounded_water_value(
                    report, 6, fields.qh[i][j][k],
                    static_cast<double>(tendencies.dqh_dt[i][j][k])
                        * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
            }
        }
    }

    MicrophysicsWaterRepairReport result;
    for (const auto& report : slab_reports)
    {
        merge_water_repair_report(result, report);
    }
    return result;
}

MicrophysicsThermodynamicRepairReport
evaluate_projected_thermodynamic_repairs(
    const SimulationStateView& state,
    double dt_micro,
    const MicrophysicsTendencyFields& tendencies,
    const MicrophysicsScheme& scheme,
    const std::vector<float>& column_multipliers,
    const Field3D* theta_storage_residual,
    double max_theta_step_change_k,
    std::vector<MicrophysicsThermodynamicRepairReport>& slab_reports,
    std::vector<std::array<double, 3>>* conservation_slab_partials)
{
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    const auto& coupling = state.tendencies;
    const RadiationRuntimeState& radiation_runtime = state.radiation_runtime
        ? *state.radiation_runtime : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime");
    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    const bool attribute_moist_energy =
        moist_basis_v2_commit_inputs_available(state);
    slab_reports.assign(
        repair_report_accumulator_count(domain),
        MicrophysicsThermodynamicRepairReport{});
    if (conservation_slab_partials != nullptr)
    {
        conservation_slab_partials->assign(
            repair_report_accumulator_count(domain),
            std::array<double, 3>{});
    }

    #pragma omp parallel for schedule(static)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            auto& report = slab_reports[
                repair_report_accumulator_index(domain, i, j)];
            std::array<double, 3>* conservation_partial =
                conservation_slab_partials != nullptr
                ? &(*conservation_slab_partials)[
                    repair_report_accumulator_index(domain, i, j)]
                : nullptr;
            if (conservation_partial != nullptr)
            {
                const double rho_d_sfc =
                    live_water_mass_density(state, i, j, domain.k_begin);
                (*conservation_partial)[2] +=
                    scheme.bottom_sedimentation_sink_rate(
                        fields.rho, fields.qr, fields.qs,
                        fields.qg, fields.qh, i, j, dt_micro)
                    * static_cast<double>(
                        column_multipliers[
                            static_cast<std::size_t>(i)
                                * static_cast<std::size_t>(grid.nth)
                            + static_cast<std::size_t>(j)])
                    * rho_d_sfc
                    * diagnostic_cell_volume(
                        state, i, j, domain.k_begin);
            }
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_dV = diagnostic_cell_volume(state, i, j, k);
                const double heat_factor =
                    theta_heat_factor(state, i, j, k, cell_dV);
                const double dt_step = dt_micro;
                const double radiation_delta = radiation_theta_delta(
                    state, radiation_runtime, i, j, k, dt_step, heat_factor);
                const double kg_per_kgkg =
                    live_water_mass_density(state, i, j, k) * cell_dV;
                double projected_residual_new = 0.0;
                const double theta_old =
                    static_cast<double>(fields.theta[i][j][k]);
                const float projected_theta = project_theta_update(
                    report,
                    theta_old,
                    theta_storage_residual
                        ? static_cast<double>((*theta_storage_residual)[i][j][k])
                        : 0.0,
                    theta_storage_residual ? &projected_residual_new : nullptr,
                    static_cast<double>(tendencies.dtheta_dt[i][j][k])
                        * dt_step,
                    radiation_delta,
                    static_cast<double>(coupling.dtheta_dt_pbl[i][j][k])
                        * dt_step,
                    heat_factor,
                    max_theta_step_change_k,
                    true);
                const std::array<double, 7> water_old{{
                    static_cast<double>(fields.qv[i][j][k]),
                    static_cast<double>(fields.qc[i][j][k]),
                    static_cast<double>(fields.qr[i][j][k]),
                    static_cast<double>(fields.qi[i][j][k]),
                    static_cast<double>(fields.qs[i][j][k]),
                    static_cast<double>(fields.qg[i][j][k]),
                    static_cast<double>(fields.qh[i][j][k])}};
                if (conservation_partial != nullptr)
                {
                    const double dq_sum =
                        static_cast<double>(tendencies.dqv_dt[i][j][k])
                        + static_cast<double>(tendencies.dqc_dt[i][j][k])
                        + static_cast<double>(tendencies.dqr_dt[i][j][k])
                        + static_cast<double>(tendencies.dqi_dt[i][j][k])
                        + static_cast<double>(tendencies.dqs_dt[i][j][k])
                        + static_cast<double>(tendencies.dqg_dt[i][j][k])
                        + static_cast<double>(tendencies.dqh_dt[i][j][k]);
                    const double q_sum =
                        water_old[0] + water_old[1] + water_old[2]
                        + water_old[3] + water_old[4] + water_old[5]
                        + water_old[6];
                    const double rho_d_k =
                        live_water_mass_density(state, i, j, k);
                    // Preserve the standalone gate's exact parenthesization.
                    (*conservation_partial)[0] +=
                        rho_d_k * dq_sum * dt_micro * cell_dV;
                    (*conservation_partial)[1] +=
                        rho_d_k * q_sum * cell_dV;
                }
                const auto projected_water =
                    [](float old_value, double increment, float upper_bound)
                {
                    const float raw = static_cast<float>(
                        static_cast<double>(old_value) + increment);
                    return static_cast<double>(
                        !std::isfinite(raw)
                            ? 0.0f
                            : std::clamp(raw, 0.0f, upper_bound));
                };
                const std::array<double, 7> water_new{{
                    projected_water(
                        fields.qv[i][j][k],
                        (static_cast<double>(tendencies.dqv_dt[i][j][k])
                         + static_cast<double>(
                             coupling.dqv_dt_pbl[i][j][k])) * dt_step,
                        qv_max_kgkg),
                    projected_water(
                        fields.qc[i][j][k],
                        (static_cast<double>(tendencies.dqc_dt[i][j][k])
                         + boundary_layer_cloud_tendency(
                             state, coupling.dqc_dt_pbl, i, j, k)) * dt_step,
                        hydrometeor_max_kgkg),
                    projected_water(
                        fields.qr[i][j][k],
                        static_cast<double>(tendencies.dqr_dt[i][j][k])
                            * dt_step,
                        hydrometeor_max_kgkg),
                    projected_water(
                        fields.qi[i][j][k],
                        (static_cast<double>(tendencies.dqi_dt[i][j][k])
                         + boundary_layer_cloud_tendency(
                             state, coupling.dqi_dt_pbl, i, j, k)) * dt_step,
                        hydrometeor_max_kgkg),
                    projected_water(
                        fields.qs[i][j][k],
                        static_cast<double>(tendencies.dqs_dt[i][j][k])
                            * dt_step,
                        hydrometeor_max_kgkg),
                    projected_water(
                        fields.qg[i][j][k],
                        static_cast<double>(tendencies.dqg_dt[i][j][k])
                            * dt_step,
                        hydrometeor_max_kgkg),
                    projected_water(
                        fields.qh[i][j][k],
                        static_cast<double>(tendencies.dqh_dt[i][j][k])
                            * dt_step,
                        hydrometeor_max_kgkg)}};
                if (attribute_moist_energy
                    && !observe_moist_basis_v2_commit(
                        report,
                        kg_per_kgkg,
                        heat_factor
                            / (kg_per_kgkg
                               * physical_constants::
                                   specific_heat_cp_jkgk),
                        moist_basis_v2_specific_kinetic_energy(
                            state, terrain_metrics, domain.k_begin,
                            i, j, k),
                        tmv::physical_grid_metrics::scalar_height(
                            state, i, j, k),
                        theta_old, water_old,
                        static_cast<double>(projected_theta), water_new))
                {
                    report.moist_committed_energy_change_j =
                        std::numeric_limits<double>::quiet_NaN();
                }

                const double microphysics_dq =
                    static_cast<double>(tendencies.dqv_dt[i][j][k])
                  + static_cast<double>(tendencies.dqc_dt[i][j][k])
                  + static_cast<double>(tendencies.dqr_dt[i][j][k])
                  + static_cast<double>(tendencies.dqi_dt[i][j][k])
                  + static_cast<double>(tendencies.dqs_dt[i][j][k])
                  + static_cast<double>(tendencies.dqg_dt[i][j][k])
                  + static_cast<double>(tendencies.dqh_dt[i][j][k]);
                if (std::isfinite(microphysics_dq))
                {
                    report.microphysics_water_tendency_kg +=
                        microphysics_dq * dt_step * kg_per_kgkg;
                }
                const std::array<double, 7> microphysics_species{{
                    static_cast<double>(tendencies.dqv_dt[i][j][k]),
                    static_cast<double>(tendencies.dqc_dt[i][j][k]),
                    static_cast<double>(tendencies.dqr_dt[i][j][k]),
                    static_cast<double>(tendencies.dqi_dt[i][j][k]),
                    static_cast<double>(tendencies.dqs_dt[i][j][k]),
                    static_cast<double>(tendencies.dqg_dt[i][j][k]),
                    static_cast<double>(tendencies.dqh_dt[i][j][k])}};
                for (std::size_t species = 0;
                     species < microphysics_species.size(); ++species)
                {
                    if (std::isfinite(microphysics_species[species]))
                    {
                        report.microphysics_water_tendency_kg_by_species[species]
                            += microphysics_species[species] * dt_step
                             * kg_per_kgkg;
                    }
                }
                const double pbl_dqv = static_cast<double>(
                    coupling.dqv_dt_pbl[i][j][k]);
                if (std::isfinite(pbl_dqv))
                {
                    report.boundary_layer_vapor_tendency_kg +=
                        pbl_dqv * dt_step * kg_per_kgkg;
                }
                const double pbl_dqc = boundary_layer_cloud_tendency(
                    state, coupling.dqc_dt_pbl, i, j, k);
                const double pbl_dqi = boundary_layer_cloud_tendency(
                    state, coupling.dqi_dt_pbl, i, j, k);
                if (std::isfinite(pbl_dqc))
                    report.boundary_layer_cloud_liquid_tendency_kg +=
                        pbl_dqc * dt_step * kg_per_kgkg;
                if (std::isfinite(pbl_dqi))
                    report.boundary_layer_cloud_ice_tendency_kg +=
                        pbl_dqi * dt_step * kg_per_kgkg;
            }
        }
    }

    MicrophysicsThermodynamicRepairReport result;
    for (const auto& report : slab_reports)
    {
        merge_thermodynamic_repair_report(result, report);
    }
    close_microphysics_species_total(result);
    if (attribute_moist_energy)
    {
        finalize_moist_basis_v2_commit(
            result, "projected shared thermodynamic commit");
    }
    return result;
}

void ensure_tendency_shape(const SimulationStateView& state, Field3D& f)
{
    if (f.size_r() != state.grid.nr ||
        f.size_th() != state.grid.nth ||
        f.size_z() != state.grid.nz)
    {
        f.resize(state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    }
}

void ensure_tendency_shapes(const SimulationStateView& state,
                            MicrophysicsTendencyFields& tendencies)
{
    ensure_tendency_shape(state, tendencies.dtheta_dt);
    ensure_tendency_shape(state, tendencies.dqv_dt);
    ensure_tendency_shape(state, tendencies.dqc_dt);
    ensure_tendency_shape(state, tendencies.dqr_dt);
    ensure_tendency_shape(state, tendencies.dqi_dt);
    ensure_tendency_shape(state, tendencies.dqs_dt);
    ensure_tendency_shape(state, tendencies.dqg_dt);
    ensure_tendency_shape(state, tendencies.dqh_dt);
}

bool field_matches_grid(const SimulationStateView& state, const Field3D& field)
{
    return field.size_r() == state.grid.nr
        && field.size_th() == state.grid.nth
        && field.size_z() == state.grid.nz;
}

void validate_microphysics_step_inputs(const SimulationStateView& state)
{
    const auto& grid = state.grid;
    if (grid.nr <= 0 || grid.nth <= 0 || grid.nz <= 0
        || !std::isfinite(grid.dr) || grid.dr <= 0.0
        || !std::isfinite(grid.dtheta) || grid.dtheta <= 0.0
        || !std::isfinite(grid.dz) || grid.dz <= 0.0)
    {
        throw std::invalid_argument(
            "microphysics requires positive dimensions and finite Cartesian spacing");
    }

    const std::array<const Field3D*, 13> required_fields{{
        &state.fields.p, &state.fields.rho, &state.fields.theta,
        &state.fields.qv, &state.fields.qc, &state.fields.qr,
        &state.fields.qi, &state.fields.qs, &state.fields.qg,
        &state.fields.qh, &state.tendencies.dtheta_dt_rad,
        &state.tendencies.dtheta_dt_pbl, &state.tendencies.dqv_dt_pbl
    }};
    for (const Field3D* field : required_fields)
    {
        if (!field_matches_grid(state, *field))
        {
            throw std::invalid_argument(
                "microphysics state/coupling field shape does not match the grid");
        }
    }
    const bool cloud_aware_boundary_layer =
        state.boundary_layer_runtime != nullptr
        && !state.boundary_layer_runtime->config.clear_sky_only;
    if (cloud_aware_boundary_layer
        && (state.tendencies.dqc_dt_pbl == nullptr
            || state.tendencies.dqi_dt_pbl == nullptr
            || !field_matches_grid(state, *state.tendencies.dqc_dt_pbl)
            || !field_matches_grid(state, *state.tendencies.dqi_dt_pbl)))
    {
        throw std::invalid_argument(
            "cloud-aware boundary-layer coupling requires aligned qc and qi tendencies");
    }

    const PhysicalCellDomain domain = physical_scalar_domain(grid);
    if (domain.empty())
    {
        throw std::invalid_argument("microphysics physical cell domain is empty");
    }
    tmv::physical_grid_metrics::validate(state);
#ifdef NIMBUS_BRANCHING_MICROPHYSICS_INPUT_VALIDATION
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double pressure = state.fields.p[i][j][k];
                const double density = state.fields.rho[i][j][k];
                const double theta = state.fields.theta[i][j][k];
                if (!std::isfinite(pressure) || pressure <= 0.0
                    || !std::isfinite(density) || density <= 0.0
                    || !std::isfinite(theta)
                    || theta < theta_min_k || theta > theta_max_k)
                {
                    throw std::invalid_argument(
                        "microphysics received invalid pressure, density, or potential temperature");
                }
                const std::array<float, 7> water{{
                    state.fields.qv[i][j][k], state.fields.qc[i][j][k],
                    state.fields.qr[i][j][k], state.fields.qi[i][j][k],
                    state.fields.qs[i][j][k], state.fields.qg[i][j][k],
                    state.fields.qh[i][j][k]
                }};
                for (std::size_t species = 0; species < water.size(); ++species)
                {
                    const float upper = species == 0
                        ? qv_max_kgkg : hydrometeor_max_kgkg;
                    if (!std::isfinite(water[species])
                        || water[species] < 0.0f || water[species] > upper)
                    {
                        throw std::invalid_argument(
                            "microphysics received water outside its physical state bounds");
                    }
                }
                if (!std::isfinite(state.tendencies.dtheta_dt_rad[i][j][k])
                    || !std::isfinite(state.tendencies.dtheta_dt_pbl[i][j][k])
                    || !std::isfinite(state.tendencies.dqv_dt_pbl[i][j][k])
                    || (cloud_aware_boundary_layer
                        && (!std::isfinite(
                            (*state.tendencies.dqc_dt_pbl)[i][j][k])
                            || !std::isfinite(
                            (*state.tendencies.dqi_dt_pbl)[i][j][k]))))
                {
                    throw std::invalid_argument(
                        "microphysics received a non-finite coupled tendency");
                }
            }
        }
    }
#else
    // Encode the serial fail-first order in the reduction key so SIMD may
    // evaluate several vertical cells together without changing which
    // diagnostic category is reported. Category order within one cell remains
    // thermodynamic, water, then coupled tendency, exactly as above.
    constexpr std::uint64_t no_invalid_input =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t first_invalid_code = no_invalid_input;
    const std::uint64_t j_extent = static_cast<std::uint64_t>(
        domain.j_end - domain.j_begin);
    const std::uint64_t k_extent = static_cast<std::uint64_t>(
        domain.k_end - domain.k_begin);
    const auto input_cells = static_cast<std::uint64_t>(domain.i_end - domain.i_begin) * j_extent * k_extent;
    #pragma omp parallel for collapse(2) schedule(static) reduction(min:first_invalid_code) if(input_cells >= 65536)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            #pragma omp simd reduction(min:first_invalid_code)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double pressure = state.fields.p[i][j][k];
                const double density = state.fields.rho[i][j][k];
                const double theta_value = state.fields.theta[i][j][k];
                const bool thermodynamic_bad =
                    !std::isfinite(pressure) || pressure <= 0.0
                    || !std::isfinite(density) || density <= 0.0
                    || !std::isfinite(theta_value)
                    || theta_value < theta_min_k
                    || theta_value > theta_max_k;
                const float qv_value = state.fields.qv[i][j][k];
                const bool water_bad =
                    !std::isfinite(qv_value)
                    || qv_value < 0.0f || qv_value > qv_max_kgkg
                    || !std::isfinite(state.fields.qc[i][j][k])
                    || state.fields.qc[i][j][k] < 0.0f
                    || state.fields.qc[i][j][k] > hydrometeor_max_kgkg
                    || !std::isfinite(state.fields.qr[i][j][k])
                    || state.fields.qr[i][j][k] < 0.0f
                    || state.fields.qr[i][j][k] > hydrometeor_max_kgkg
                    || !std::isfinite(state.fields.qi[i][j][k])
                    || state.fields.qi[i][j][k] < 0.0f
                    || state.fields.qi[i][j][k] > hydrometeor_max_kgkg
                    || !std::isfinite(state.fields.qs[i][j][k])
                    || state.fields.qs[i][j][k] < 0.0f
                    || state.fields.qs[i][j][k] > hydrometeor_max_kgkg
                    || !std::isfinite(state.fields.qg[i][j][k])
                    || state.fields.qg[i][j][k] < 0.0f
                    || state.fields.qg[i][j][k] > hydrometeor_max_kgkg
                    || !std::isfinite(state.fields.qh[i][j][k])
                    || state.fields.qh[i][j][k] < 0.0f
                    || state.fields.qh[i][j][k] > hydrometeor_max_kgkg;
                const bool coupled_bad =
                    !std::isfinite(
                        state.tendencies.dtheta_dt_rad[i][j][k])
                    || !std::isfinite(
                        state.tendencies.dtheta_dt_pbl[i][j][k])
                    || !std::isfinite(
                        state.tendencies.dqv_dt_pbl[i][j][k])
                    || (cloud_aware_boundary_layer
                        && (!std::isfinite(
                            (*state.tendencies.dqc_dt_pbl)[i][j][k])
                            || !std::isfinite(
                            (*state.tendencies.dqi_dt_pbl)[i][j][k])));
                const std::uint64_t category = thermodynamic_bad ? 0u
                    : water_bad ? 1u : coupled_bad ? 2u : 3u;
                const std::uint64_t ordinal =
                    (static_cast<std::uint64_t>(i - domain.i_begin)
                         * j_extent
                     + static_cast<std::uint64_t>(j - domain.j_begin))
                        * k_extent
                    + static_cast<std::uint64_t>(k - domain.k_begin);
                const std::uint64_t invalid_code = category < 3u
                    ? ordinal * 4u + category : no_invalid_input;
                if (invalid_code < first_invalid_code)
                {
                    first_invalid_code = invalid_code;
                }
            }
        }
    }
    if (first_invalid_code != no_invalid_input)
    {
        switch (first_invalid_code % 4u)
        {
        case 0:
            throw std::invalid_argument(
                "microphysics received invalid pressure, density, or potential temperature");
        case 1:
            throw std::invalid_argument(
                "microphysics received water outside its physical state bounds");
        default:
            throw std::invalid_argument(
                "microphysics received a non-finite coupled tendency");
        }
    }
#endif
}

#ifdef NIMBUS_SEPARATE_MICROPHYSICS_TENDENCY_VALIDATION_SCANS
int count_nonfinite_tendency(const Field3D& field,
                             const PhysicalCellDomain& domain)
{
    int count = 0;
    #pragma omp parallel for collapse(2) reduction(+:count)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                count += !std::isfinite(field[i][j][k]);
            }
        }
    }
    return count;
}
#endif

int count_nonfinite_tendencies(const SimulationStateView& state,
                               const MicrophysicsTendencyFields& tendencies,
                               int& nonfinite_theta_count)
{
    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
#ifdef NIMBUS_SEPARATE_MICROPHYSICS_TENDENCY_VALIDATION_SCANS
    nonfinite_theta_count =
        count_nonfinite_tendency(tendencies.dtheta_dt, domain);
    return nonfinite_theta_count +
           count_nonfinite_tendency(tendencies.dqv_dt, domain) +
           count_nonfinite_tendency(tendencies.dqc_dt, domain) +
           count_nonfinite_tendency(tendencies.dqr_dt, domain) +
           count_nonfinite_tendency(tendencies.dqi_dt, domain) +
           count_nonfinite_tendency(tendencies.dqs_dt, domain) +
           count_nonfinite_tendency(tendencies.dqg_dt, domain) +
           count_nonfinite_tendency(tendencies.dqh_dt, domain);
#else
    int theta_count = 0;
    int total_count = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:theta_count,total_count)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const int theta_bad =
                    !std::isfinite(tendencies.dtheta_dt[i][j][k]);
                theta_count += theta_bad;
                total_count += theta_bad
                    + !std::isfinite(tendencies.dqv_dt[i][j][k])
                    + !std::isfinite(tendencies.dqc_dt[i][j][k])
                    + !std::isfinite(tendencies.dqr_dt[i][j][k])
                    + !std::isfinite(tendencies.dqi_dt[i][j][k])
                    + !std::isfinite(tendencies.dqs_dt[i][j][k])
                    + !std::isfinite(tendencies.dqg_dt[i][j][k])
                    + !std::isfinite(tendencies.dqh_dt[i][j][k]);
            }
        }
    }
    nonfinite_theta_count = theta_count;
    return total_count;
#endif
}

int prepare_microphysics_tendencies(const SimulationStateView& state,
                                    double dt_micro,
                                    MicrophysicsTendencyFields& tendencies,
                                    MicrophysicsScheme& scheme,
                                    std::vector<float>& column_multipliers,
                                    bool& stochastic_column_control_active,
                                    bool independently_validate,
                                    int& nonfinite_theta_count)
{
    const auto& fields = state.fields;
    ensure_tendency_shapes(state, tendencies);
    scheme.set_physical_cell_domain(
        state.grid.nr, state.grid.nth, state.grid.nz,
        physical_scalar_domain(state.grid));
    bind_microphysics_velocity(state, scheme);
    const RuntimeConfigurationState* configuration = state.configuration;
    const ComputeBackendRuntimeState* compute = state.compute_runtime;
    const bool                        require_native_dispatch =
        configuration != nullptr && compute != nullptr &&
        (configuration->microphysics_scheme == "kessler" || configuration->microphysics_scheme == "lin" ||
         configuration->microphysics_scheme == "thompson" || configuration->microphysics_scheme == "milbrandt") &&
        compute->active_kind != ComputeBackendKind::Cpu && !compute->fallback_active && !compute->config.allow_fallback;
    scheme.bind_execution_owners(tmv::require_level_runtime(state.compute_runtime, "compute runtime"), state.parity_runtime);
    scheme.set_require_native_dispatch(require_native_dispatch);

    bind_microphysics_grid_geometry(state, scheme);
    const auto scheme_t0 = microphysics_perf_start();
    scheme.compute_tendencies(
        fields.p, fields.theta, fields.rho, fields.qv, fields.qc, fields.qr,
        fields.qi, fields.qs, fields.qg, fields.qh,
        dt_micro,
        tendencies.dtheta_dt, tendencies.dqv_dt,
        tendencies.dqc_dt, tendencies.dqr_dt,
        tendencies.dqi_dt, tendencies.dqs_dt,
        tendencies.dqg_dt, tendencies.dqh_dt);
    microphysics_perf_finish(microphysics_perf_totals.scheme_s, scheme_t0);

    const auto chaos_t0 = microphysics_perf_start();
    stochastic_column_control_active =
        apply_chaos_to_microphysics_tendencies(
        state,
        tendencies.dtheta_dt, tendencies.dqv_dt,
        tendencies.dqc_dt, tendencies.dqr_dt,
        tendencies.dqi_dt, tendencies.dqs_dt,
        tendencies.dqg_dt, tendencies.dqh_dt,
        column_multipliers);
    microphysics_perf_finish(microphysics_perf_totals.chaos_s, chaos_t0);

    const auto tendency_contract_t0 = microphysics_perf_start();
    const std::array<const Field3D*, 8> produced{{
        &tendencies.dtheta_dt, &tendencies.dqv_dt,
        &tendencies.dqc_dt, &tendencies.dqr_dt,
        &tendencies.dqi_dt, &tendencies.dqs_dt,
        &tendencies.dqg_dt, &tendencies.dqh_dt
    }};
    for (const Field3D* field : produced)
    {
        if (!field_matches_grid(state, *field))
        {
            throw std::runtime_error(
                "microphysics scheme returned a tendency with the wrong shape");
        }
    }
    microphysics_perf_finish(
        microphysics_perf_totals.tendency_contract_s, tendency_contract_t0);

    if (independently_validate)
    {
        const auto validation_t0 = microphysics_perf_start();
        const int count = count_nonfinite_tendencies(
            state, tendencies, nonfinite_theta_count);
        microphysics_perf_finish(
            microphysics_perf_totals.validation_s, validation_t0);
        return count;
    }
    nonfinite_theta_count = 0;
    return 0;
}

void admit_and_apply_microphysics_column_multipliers(
    const SimulationStateView& state,
    double dt_micro,
    MicrophysicsTendencyFields& tendencies,
    MicrophysicsScheme& scheme,
    std::vector<float>& multipliers)
{
    const std::size_t expected =
        static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    if (multipliers.size() != expected)
        throw std::runtime_error(
            "microphysics column multiplier shape mismatch");

    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const std::array<const Field3D*, 7> state_water{{
        &state.fields.qv, &state.fields.qc, &state.fields.qr,
        &state.fields.qi, &state.fields.qs, &state.fields.qg,
        &state.fields.qh}};
    const std::array<const Field3D*, 7> tendency_water{{
        &tendencies.dqv_dt, &tendencies.dqc_dt, &tendencies.dqr_dt,
        &tendencies.dqi_dt, &tendencies.dqs_dt, &tendencies.dqg_dt,
        &tendencies.dqh_dt}};

    std::exception_ptr failure;
    std::size_t first_failed_column = std::numeric_limits<std::size_t>::max();
    const std::size_t columns = static_cast<std::size_t>(domain.i_end - domain.i_begin) *
        static_cast<std::size_t>(domain.j_end - domain.j_begin);
#pragma omp parallel for collapse(2) schedule(static) if(columns >= 64)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            try
            {
                const std::size_t index =
                    static_cast<std::size_t>(i)
                        * static_cast<std::size_t>(state.grid.nth)
                    + static_cast<std::size_t>(j);
                double admitted = static_cast<double>(multipliers[index]);
                if (!std::isfinite(admitted) || admitted < 0.0)
                    throw std::runtime_error(
                        "invalid microphysics column multiplier");
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    for (std::size_t species = 0;
                         species < state_water.size(); ++species)
                    {
                        const double old_value =
                            (*state_water[species])[i][j][k];
                        const double rate =
                            (*tendency_water[species])[i][j][k];
                        const double upper = species == 0
                            ? static_cast<double>(qv_max_kgkg)
                            : static_cast<double>(hydrometeor_max_kgkg);
                        if (!std::isfinite(old_value) || old_value < 0.0
                            || old_value > upper || !std::isfinite(rate))
                            throw std::runtime_error(
                                "invalid state or tendency during microphysics donor admission");
                        if (rate < 0.0)
                            admitted = std::min(
                                admitted, old_value / (-dt_micro * rate));
                        else if (rate > 0.0)
                            admitted = std::min(
                                admitted,
                                (upper - old_value) / (dt_micro * rate));
                    }
                float stored = static_cast<float>(std::max(0.0, admitted));
                // A factor computed exactly at a reservoir boundary may round up
                // in fp32. Step inward once only when admission reduced the
                // requested value; this is process-rate admission, not a state
                // repair, and every coupled source/sink uses the same stored bit.
                if (static_cast<double>(stored)
                        < static_cast<double>(multipliers[index]))
                    stored = std::nextafter(stored, 0.0f);
                multipliers[index] = stored;
            }
            catch (...)
            {
                const std::size_t column = static_cast<std::size_t>(i) * state.grid.nth + j;
#pragma omp critical(nimbus_microphysics_column_admission_failure)
                {
                    if (column < first_failed_column)
                    {
                        first_failed_column = column;
                        failure = std::current_exception();
                    }
                }
            }
        }
    if (failure) std::rethrow_exception(failure);

    std::string private_error;
    if (!scheme.apply_stochastic_column_multipliers(
            dt_micro, state.fields.rho, multipliers, private_error))
        throw std::runtime_error(
            "microphysics private-prognostic column admission rejected: "
            + private_error);

    Field3D* fields[] = {
        &tendencies.dtheta_dt, &tendencies.dqv_dt,
        &tendencies.dqc_dt, &tendencies.dqr_dt,
        &tendencies.dqi_dt, &tendencies.dqs_dt,
        &tendencies.dqg_dt, &tendencies.dqh_dt};
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const float multiplier = multipliers[static_cast<std::size_t>(i) * state.grid.nth + j];
            if (!std::isfinite(multiplier) || multiplier < 0.0f)
                throw std::runtime_error("microphysics scheme produced an invalid admitted column multiplier");
        }
#pragma omp parallel for collapse(2) schedule(static) if(columns >= 64)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const float multiplier =
                multipliers[static_cast<std::size_t>(i)
                                * static_cast<std::size_t>(state.grid.nth)
                            + static_cast<std::size_t>(j)];
            for (Field3D* field : fields)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    (*field)[i][j][k] *= multiplier;
        }
}

bool microphysics_conservation_gate_requested(const SimulationStateView& execution_state) noexcept
{
    return step_diagnostics_enabled()
        || tmv::require_run_configuration(execution_state).validation_policy.mode == GuardMode::Strict;
}

MicrophysicsConservationReport finalize_microphysics_conservation_gate(
    const std::vector<std::array<double, 3>>& slab_partials,
    double dt_micro)
{
    MicrophysicsConservationReport gate;
    gate.active = true;
    std::array<tmv::diagnostics::CompensatedSum, 3> merged{};
    for (const auto& partial : slab_partials)
    {
        for (std::size_t n = 0; n < partial.size(); ++n)
            merged[n].add(partial[n]);
    }
    const double tendency_imbalance_kg = merged[0].value();
    gate.baseline_q_kg = merged[1].value();
    gate.surface_precipitation_kg = merged[2].value() * dt_micro;
    gate.tendency_imbalance_kg =
        tendency_imbalance_kg + gate.surface_precipitation_kg;
    gate.relative_imbalance =
        std::abs(gate.tendency_imbalance_kg)
        / std::max(1.0, gate.baseline_q_kg);
    return gate;
}

#ifdef NIMBUS_SEPARATE_MICROPHYSICS_CONSERVATION_SCAN
MicrophysicsConservationReport evaluate_microphysics_conservation_gate(
    const SimulationStateView& state,
    double dt_micro,
    const MicrophysicsTendencyFields& tendencies,
    const MicrophysicsScheme& scheme,
    const std::vector<float>& column_multipliers)
{
    MicrophysicsConservationReport gate;
    gate.active = microphysics_conservation_gate_requested(state);
    if (!gate.active)
    {
        return gate;
    }

    double tendency_imbalance_kg = 0.0;
    double baseline_q_kg = 0.0;
    double surface_sink_kg_s = 0.0;
    const auto& grid = state.grid;
    const auto& fields = state.fields;
    const PhysicalCellDomain domain = physical_scalar_domain(grid);

    const auto accumulate_slab = [&](int i,
                                     double& tendency_imbalance,
                                     double& baseline_q,
                                     double& surface_sink)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const double rho_d_sfc =
                live_water_mass_density(state, i, j, domain.k_begin);
            surface_sink +=
                scheme.bottom_sedimentation_sink_rate(
                    fields.rho, fields.qr, fields.qs, fields.qg, fields.qh,
                    i, j, dt_micro)
                * static_cast<double>(
                    column_multipliers[static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(grid.nth)
                                       + static_cast<std::size_t>(j)])
                * rho_d_sfc * diagnostic_cell_volume(
                    state, i, j, domain.k_begin);

            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double cell_dV = diagnostic_cell_volume(state, i, j, k);
                const double rho_d_k = live_water_mass_density(state, i, j, k);
                const double dq_sum =
                    static_cast<double>(tendencies.dqv_dt[i][j][k]) +
                    static_cast<double>(tendencies.dqc_dt[i][j][k]) +
                    static_cast<double>(tendencies.dqr_dt[i][j][k]) +
                    static_cast<double>(tendencies.dqi_dt[i][j][k]) +
                    static_cast<double>(tendencies.dqs_dt[i][j][k]) +
                    static_cast<double>(tendencies.dqg_dt[i][j][k]) +
                    static_cast<double>(tendencies.dqh_dt[i][j][k]);
                tendency_imbalance +=
                    rho_d_k * dq_sum * dt_micro * cell_dV;

                const double q_sum =
                    static_cast<double>(fields.qv[i][j][k]) +
                    static_cast<double>(fields.qc[i][j][k]) +
                    static_cast<double>(fields.qr[i][j][k]) +
                    static_cast<double>(fields.qi[i][j][k]) +
                    static_cast<double>(fields.qs[i][j][k]) +
                    static_cast<double>(fields.qg[i][j][k]) +
                    static_cast<double>(fields.qh[i][j][k]);
                baseline_q += rho_d_k * q_sum * cell_dV;
            }
        }
    };

    if (tmv::diagnostics::deterministic_reductions_enabled())
    {
        const auto sums = tmv::diagnostics::fixed_slab_reduce<3>(
            domain.i_begin,
            domain.i_end,
            [&](int i, std::array<double, 3>& partial)
            {
                accumulate_slab(i, partial[0], partial[1], partial[2]);
            });
        tendency_imbalance_kg = sums[0];
        baseline_q_kg = sums[1];
        surface_sink_kg_s = sums[2];
    }
    else
    {
        #pragma omp parallel for reduction(+:tendency_imbalance_kg, baseline_q_kg, surface_sink_kg_s)
        for (int i = domain.i_begin; i < domain.i_end; ++i)
        {
            accumulate_slab(
                i, tendency_imbalance_kg, baseline_q_kg, surface_sink_kg_s);
        }
    }

    gate.surface_precipitation_kg = surface_sink_kg_s * dt_micro;
    gate.tendency_imbalance_kg =
        tendency_imbalance_kg + gate.surface_precipitation_kg;
    gate.baseline_q_kg = baseline_q_kg;
    gate.relative_imbalance =
        std::abs(gate.tendency_imbalance_kg) / std::max(1.0, gate.baseline_q_kg);
    return gate;
}
#endif

void report_microphysics_conservation_gate(const SimulationStateView& execution_state,
    const MicrophysicsConservationReport& gate)
{
    if (!gate.active)
    {
        return;
    }

    if (gate.relative_imbalance > BUDGET_THRESH_KERNEL * 100.0)
    {
        tmv::log_error("[MICROPHYSICS GATE FAIL] scheme produced non-conservative "
                       "q tendencies: rel_imbalance=", gate.relative_imbalance,
                       " (threshold=", BUDGET_THRESH_KERNEL * 100.0, "), ",
                       "absolute=", gate.tendency_imbalance_kg, " kg, ",
                       "surface_precip=", gate.surface_precipitation_kg, " kg, ",
                       "baseline_q_total=", gate.baseline_q_kg, " kg");
    }
    if (tmv::require_run_configuration(execution_state).validation_policy.mode == GuardMode::Strict &&
        gate.relative_imbalance > BUDGET_THRESH_KERNEL)
    {
        throw std::runtime_error(
            "strict validation rejected non-conservative microphysics tendencies: relative imbalance=" +
            std::to_string(gate.relative_imbalance));
    }
    else if (gate.relative_imbalance > BUDGET_THRESH_KERNEL)
    {
        tmv::log_warn("[MICROPHYSICS GATE] scheme q-tendency imbalance "
                      "rel=", gate.relative_imbalance,
                      " (threshold=", BUDGET_THRESH_KERNEL, "), ",
                      "absolute=", gate.tendency_imbalance_kg, " kg, ",
                      "surface_precip=", gate.surface_precipitation_kg, " kg");
    }
}

bool prepare_surface_precipitation_candidates(
    const SimulationStateView& state,
    double dt_micro,
    const MicrophysicsScheme& scheme,
    const std::vector<float>& column_multipliers,
    MicrophysicsExecutionScratch& scratch)
{
    auto* const rate = state.surface.precipitation_rate_mm_h;
    auto* const accumulated = state.surface.accumulated_rainfall_mm;
    auto* const residual = state.surface.accumulated_rainfall_residual_mm;
    double* const last_update =
        state.surface.accumulated_rainfall_last_update_s;
    const bool any_bound = rate != nullptr || accumulated != nullptr
        || residual != nullptr
        || last_update != nullptr;
    if (!any_bound) return false;
    if (rate == nullptr || accumulated == nullptr || residual == nullptr
        || last_update == nullptr)
        throw std::invalid_argument(
            "microphysics surface precipitation state is only partially bound");

    const std::size_t surface_size =
        static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    if (rate->empty()) rate->assign(surface_size, 0.0f);
    if (accumulated->empty()) accumulated->assign(surface_size, 0.0f);
    if (residual->empty()) residual->assign(surface_size, 0.0f);
    if (rate->size() != surface_size || accumulated->size() != surface_size
        || residual->size() != surface_size)
        throw std::invalid_argument(
            "microphysics surface precipitation state shape does not match the grid");
    const double committed_time_s = state.time_s + dt_micro;
    if (!std::isfinite(state.time_s) || state.time_s < 0.0
        || !std::isfinite(committed_time_s)
        || !std::isfinite(*last_update) || *last_update < 0.0
        || *last_update > committed_time_s
            + 1.0e-9 * std::max(1.0, std::abs(committed_time_s)))
        throw std::invalid_argument(
            "microphysics surface precipitation clock is invalid");

    scratch.precipitation_rate_candidate_mm_h.assign(surface_size, 0.0f);
    scratch.accumulated_rainfall_candidate_mm = *accumulated;
    scratch.accumulated_rainfall_residual_candidate_mm = *residual;
    for (float value : *rate)
        if (!std::isfinite(value) || value < 0.0f)
            throw std::invalid_argument(
                "surface precipitation rate contains an invalid value");
    for (float value : *accumulated)
        if (!std::isfinite(value) || value < 0.0f)
            throw std::invalid_argument(
                "accumulated rainfall contains an invalid value");
    for (float value : *residual)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "accumulated rainfall compensation contains a non-finite value");

    const PhysicalCellDomain domain = physical_scalar_domain(state.grid);
    const int k_surface = domain.k_begin;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            const double sink_kgkg_s =
                scheme.bottom_sedimentation_sink_rate(
                    state.fields.rho, state.fields.qr, state.fields.qs,
                    state.fields.qg, state.fields.qh, i, j, dt_micro)
                * static_cast<double>(
                    column_multipliers[static_cast<std::size_t>(i)
                                           * static_cast<std::size_t>(state.grid.nth)
                                       + static_cast<std::size_t>(j)]);
            const double density_kgm3 =
                live_water_mass_density(state, i, j, k_surface);
            const double thickness_m =
                tmv::physical_grid_metrics::layer_thickness(
                    state, i, j, k_surface);
            const double flux_kgm2s =
                sink_kgkg_s * density_kgm3 * thickness_m;
            const std::size_t index =
                static_cast<std::size_t>(i)
                    * static_cast<std::size_t>(state.grid.nth)
                + static_cast<std::size_t>(j);
            const double rate_mm_h = flux_kgm2s * 3600.0;
            const double total_mm =
                static_cast<double>((*accumulated)[index])
                + static_cast<double>((*residual)[index])
                + flux_kgm2s * dt_micro;
            if (!std::isfinite(sink_kgkg_s) || sink_kgkg_s < 0.0
                || !std::isfinite(flux_kgm2s) || flux_kgm2s < 0.0
                || !std::isfinite(rate_mm_h) || rate_mm_h < 0.0
                || !std::isfinite(total_mm) || total_mm < 0.0
                || rate_mm_h > std::numeric_limits<float>::max()
                || total_mm > std::numeric_limits<float>::max())
                throw std::runtime_error(
                    "microphysics surface precipitation publication overflowed");
            scratch.precipitation_rate_candidate_mm_h[index] =
                static_cast<float>(rate_mm_h);
            const float stored_total = static_cast<float>(total_mm);
            const float stored_residual = static_cast<float>(
                total_mm - static_cast<double>(stored_total));
            if (!std::isfinite(stored_total)
                || !std::isfinite(stored_residual))
                throw std::runtime_error(
                    "microphysics accumulated rainfall storage overflowed");
            scratch.accumulated_rainfall_candidate_mm[index] = stored_total;
            scratch.accumulated_rainfall_residual_candidate_mm[index] =
                stored_residual;
        }
    return true;
}

void commit_surface_precipitation_candidates(
    const SimulationStateView& state,
    double dt_micro,
    const MicrophysicsExecutionScratch& scratch) noexcept
{
    auto& rate = *state.surface.precipitation_rate_mm_h;
    auto& accumulated = *state.surface.accumulated_rainfall_mm;
    auto& residual = *state.surface.accumulated_rainfall_residual_mm;
    std::memcpy(rate.data(),
                scratch.precipitation_rate_candidate_mm_h.data(),
                rate.size() * sizeof(float));
    std::memcpy(accumulated.data(),
                scratch.accumulated_rainfall_candidate_mm.data(),
                accumulated.size() * sizeof(float));
    std::memcpy(residual.data(),
                scratch.accumulated_rainfall_residual_candidate_mm.data(),
                residual.size() * sizeof(float));
    *state.surface.accumulated_rainfall_last_update_s =
        state.time_s + dt_micro;
}

void report_microphysics_water_repairs(
    const MicrophysicsWaterRepairReport& report)
{
    if (report.lower_bound_count() == 0 &&
        report.upper_bound_count() == 0 &&
        report.nonfinite_count() == 0 &&
        report.nonfinite_tendency_count == 0)
    {
        return;
    }

    tmv::log_warn(
        "[MICROPHYSICS WATER REPAIR] lower_bound_correction=",
        report.lower_bound_correction_kg(), " kg, upper_bound_correction=",
        report.upper_bound_correction_kg(), " kg, nonfinite_correction=",
        report.nonfinite_correction_kg(), " kg, net_correction=",
        report.net_correction_kg(), " kg, roundoff_lower_correction=",
        report.lower_bound_roundoff_correction_kg(), " kg, storage_rounding=",
        report.storage_rounding_kg(), " kg, lower_count=",
        report.lower_bound_count(), ", roundoff_lower_count=",
        report.lower_bound_roundoff_count(), ", upper_count=",
        report.upper_bound_count(),
        ", nonfinite_state_count=", report.nonfinite_count(),
        ", nonfinite_tendency_count=", report.nonfinite_tendency_count);

    for (std::size_t n = 0; n < report.species.size(); ++n)
    {
        const auto& field = report.species[n];
        if (field.lower_bound_count == 0 && field.upper_bound_count == 0 &&
            field.nonfinite_count == 0)
        {
            continue;
        }
        tmv::log_warn(
            "[MICROPHYSICS WATER REPAIR FIELD] field=", water_species_names[n],
            ", lower_bound_correction=", field.lower_bound_correction_kg,
            " kg, roundoff_lower_correction=",
            field.lower_bound_roundoff_correction_kg,
            " kg, storage_rounding=", field.storage_rounding_kg,
            " kg, upper_bound_correction=", field.upper_bound_correction_kg,
            " kg, nonfinite_correction=", field.nonfinite_correction_kg,
            " kg, lower_count=", field.lower_bound_count,
            ", roundoff_lower_count=", field.lower_bound_roundoff_count,
            ", upper_count=", field.upper_bound_count,
            ", nonfinite_count=", field.nonfinite_count);
    }
}

void report_microphysics_thermodynamic_repairs(
    const MicrophysicsThermodynamicRepairReport& report)
{
    if (report.theta_tendency_limit_count == 0
        && report.theta_nonfinite_count == 0
        && report.theta_lower_bound_count == 0
        && report.theta_upper_bound_count == 0)
    {
        return;
    }

    tmv::log_warn(
        "[MICROPHYSICS THERMODYNAMIC REPAIR] requested_heat=",
        report.requested_heat_j(), " J, committed_heat=",
        report.committed_heat_j(), " J, limit_correction=",
        report.theta_tendency_limit_correction_j,
        " J, nonfinite_correction=", report.theta_nonfinite_correction_j,
        " J, storage_rounding=", report.theta_storage_rounding_j,
        " J, storage_residual=", report.theta_storage_residual_j,
        " J, bound_correction=", report.theta_bound_correction_j,
        " J, microphysics_water_tendency=",
        report.microphysics_water_tendency_kg,
        " kg, pbl_vapor_tendency=",
        report.boundary_layer_vapor_tendency_kg,
        " kg, pbl_cloud_liquid_tendency=",
        report.boundary_layer_cloud_liquid_tendency_kg,
        " kg, pbl_cloud_ice_tendency=",
        report.boundary_layer_cloud_ice_tendency_kg,
        " kg, limit_count=", report.theta_tendency_limit_count,
        ", nonfinite_count=", report.theta_nonfinite_count,
        ", lower_count=", report.theta_lower_bound_count,
        ", upper_count=", report.theta_upper_bound_count);
}

void enforce_strict_water_repair_policy(const SimulationStateView& execution_state,
    const MicrophysicsWaterRepairReport& report)
{
    if (tmv::require_run_configuration(execution_state).validation_policy.mode != GuardMode::Strict)
    {
        return;
    }

    const bool fail_bounds =
        tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::Bounds ||
        tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::Both;
    const bool fail_nonfinite =
        tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::NonFinite ||
        tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::Both;
    const bool bounds_repaired =
        report.lower_bound_count() > report.lower_bound_roundoff_count()
        || report.upper_bound_count() > 0;
    const bool nonfinite_repaired =
        report.nonfinite_count() > 0 || report.nonfinite_tendency_count > 0;

    if ((fail_bounds && bounds_repaired) ||
        (fail_nonfinite && nonfinite_repaired))
    {
        throw std::runtime_error(
            "strict validation rejected microphysics water repair: net=" +
            std::to_string(report.net_correction_kg()) +
            " kg, lower_count=" + std::to_string(report.lower_bound_count()) +
            ", upper_count=" + std::to_string(report.upper_bound_count()) +
            ", nonfinite_state_count=" +
            std::to_string(report.nonfinite_count()) +
            ", nonfinite_tendency_count=" +
            std::to_string(report.nonfinite_tendency_count));
    }
}

void enforce_strict_thermodynamic_repair_policy(const SimulationStateView& execution_state,
    const MicrophysicsThermodynamicRepairReport& report)
{
    if (tmv::require_run_configuration(execution_state).validation_policy.mode != GuardMode::Strict)
    {
        return;
    }

    const bool fail_bounds =
        tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::Bounds
        || tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::Both;
    const bool fail_nonfinite =
        tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::NonFinite
        || tmv::require_run_configuration(execution_state).validation_policy.fail_on == GuardFailOn::Both;
    const bool repaired_bounds =
        report.theta_tendency_limit_count > 0
        || report.theta_lower_bound_count > 0
        || report.theta_upper_bound_count > 0;

    if ((fail_bounds && repaired_bounds)
        || (fail_nonfinite && report.theta_nonfinite_count > 0))
    {
        throw std::runtime_error(
            "strict validation rejected thermodynamic repair: net="
            + std::to_string(report.net_theta_repair_j())
            + " J, limit_count="
            + std::to_string(report.theta_tendency_limit_count)
            + ", lower_count="
            + std::to_string(report.theta_lower_bound_count)
            + ", upper_count="
            + std::to_string(report.theta_upper_bound_count)
            + ", nonfinite_count="
            + std::to_string(report.theta_nonfinite_count));
    }
}

void report_microphysics_guard_summary(
    int non_finite_theta_tendency_count,
    int non_finite_tendency_sanitized_total,
    int theta_tendency_limited_count,
    int theta_bounds_clamp_count)
{
    if (non_finite_theta_tendency_count <= 0 &&
        non_finite_tendency_sanitized_total <= 0 &&
        theta_tendency_limited_count <= 0 &&
        theta_bounds_clamp_count <= 0)
    {
        return;
    }

    tmv::log_warn("[MICROPHYSICS GUARD] non_finite_dtheta=",
                  non_finite_theta_tendency_count,
                  ", non_finite_tendency_sanitized=", non_finite_tendency_sanitized_total,
                  ", limited_dtheta=", theta_tendency_limited_count,
                  ", theta_bounds_clamped=", theta_bounds_clamp_count);
}

std::string build_clamp_histogram(
    const std::vector<int>& counts,
    char axis_label)
{
    std::string histogram;
    for (std::size_t index = 0; index < counts.size(); ++index)
    {
        if (counts[index] <= 0)
        {
            continue;
        }
        if (!histogram.empty())
        {
            histogram += " ";
        }
        histogram += axis_label;
        histogram += std::to_string(index);
        histogram += ":";
        histogram += std::to_string(counts[index]);
    }
    return histogram;
}

void report_microphysics_clamp_detail(
    int theta_bounds_clamp_count,
    const std::vector<int>& clamp_count_by_k,
    const std::vector<int>& clamp_count_by_i,
    int first_clamp_i,
    int first_clamp_j,
    int first_clamp_k,
    double first_clamp_theta_new,
    double first_clamp_theta_bounded,
    double first_clamp_dtheta_total,
    double first_clamp_dtheta_limited)
{
    if (theta_bounds_clamp_count <= 0)
    {
        return;
    }

    const std::string k_hist = build_clamp_histogram(clamp_count_by_k, 'k');
    const std::string i_hist = build_clamp_histogram(clamp_count_by_i, 'i');
    tmv::log_warn("[MICROPHYSICS THETA CLAMP DETAIL] total=", theta_bounds_clamp_count,
                  " first=(i=", first_clamp_i,
                  ",j=", first_clamp_j,
                  ",k=", first_clamp_k,
                  ",theta_new=", first_clamp_theta_new,
                  ",bounded=", first_clamp_theta_bounded,
                  ",dtheta_total=", first_clamp_dtheta_total,
                  ",dtheta_limited=", first_clamp_dtheta_limited,
                  ") k_hist=[", k_hist,
                  "] i_hist=[", i_hist, "]");
}

bool zero_endpoint_source(const Field3D* field)
{
    if (!field)
    {
        return true;
    }
    for (std::size_t index = 0; index < field->size(); ++index)
    {
        if (field->data()[index] != 0.0f)
        {
            return false;
        }
    }
    return true;
}

void validate_primary_endpoint_profile(const SimulationStateView& state)
{
    const auto& config = tmv::require_run_configuration(state);
    if (!state.compute_runtime || !state.boundary_layer_runtime ||
        !state.radiation_runtime || !state.chaos_runtime || !state.terrain_runtime ||
        state.compute_runtime->active_kind != ComputeBackendKind::Cpu ||
        state.compute_runtime->requested_kind != ComputeBackendKind::Cpu ||
        state.compute_runtime->fallback_active || gpu_mp_apply_enabled(state) ||
        config.validation_policy.mode != GuardMode::Strict ||
        config.validation_policy.fail_on != GuardFailOn::Both ||
        state.grid.geometry.coord != CoordinateSystem::Cartesian ||
        state.grid.geometry.staggered ||
        state.grid.nr <= 0 || state.grid.nth <= 0 || state.grid.nz <= 0 ||
        state.grid.nr > 4096 || state.grid.nth > 4096 || state.grid.nz > 4096)
    {
        throw std::runtime_error("primary microphysics endpoints require strict flat CPU ownership");
    }
    const std::size_t nr = static_cast<std::size_t>(state.grid.nr);
    const std::size_t nth = static_cast<std::size_t>(state.grid.nth);
    const std::size_t nz = static_cast<std::size_t>(state.grid.nz);
    if (nr > 4096 / nth || nr * nth > 4096 / nz ||
        state.boundary_layer_runtime->config.scheme_id != "none" ||
        state.boundary_layer_runtime->scheme ||
        state.radiation_runtime->config.scheme_id != "none" ||
        state.radiation_runtime->scheme ||
        state.chaos_runtime->config.scheme_id != "none" || state.chaos_runtime->scheme ||
        state.terrain_runtime->config.scheme_id != "none" || state.terrain_runtime->scheme ||
        state.terrain_runtime->earth_fixed_placement.active ||
        tmv::physical_grid_metrics::active_terrain_metrics(state) ||
        state.chaos_runtime->noise_elapsed_s != 0.0 ||
        state.chaos_runtime->pbl_base_tendencies_valid ||
        (state.domain_frame && (state.domain_frame->velocity_ms.u_ms != 0.0 ||
                                state.domain_frame->velocity_ms.v_ms != 0.0)))
    {
        throw std::runtime_error("primary microphysics endpoints have an unowned profile or source");
    }
    const auto& pbl = *state.boundary_layer_runtime;
    const auto& rad = *state.radiation_runtime;
    const auto& chaos = *state.chaos_runtime;
    const std::array<const Field3D*, 20> held{
        &state.tendencies.dtheta_dt_rad, &state.tendencies.dtheta_dt_pbl,
        &state.tendencies.dqv_dt_pbl, state.tendencies.dqc_dt_pbl,
        state.tendencies.dqi_dt_pbl, &state.tendencies.du_dt_pbl,
        &state.tendencies.dv_dt_pbl, &state.tendencies.dtke_dt_pbl,
        &pbl.held_u_force_n, &pbl.held_v_force_n,
        &pbl.held_theta_inventory_rate_kgk_s, &pbl.held_vapor_mass_rate_kg_s,
        &pbl.held_cloud_liquid_mass_rate_kg_s, &pbl.held_cloud_ice_mass_rate_kg_s,
        &rad.held_cell_power_w, &chaos.base_du_dt_pbl, &chaos.base_dv_dt_pbl,
        &chaos.base_dtheta_dt_pbl, &chaos.base_dqv_dt_pbl,
        &chaos.base_dtke_dt_pbl};
    for (const Field3D* field : held)
    {
        if (!zero_endpoint_source(field))
        {
            throw std::runtime_error("primary microphysics endpoints have a held source field");
        }
    }
    for (const double power : {rad.held_atmospheric_power_w,
                              rad.held_shortwave_incoming_power_w,
                              rad.held_shortwave_atmospheric_absorbed_power_w,
                              rad.held_surface_absorbed_power_w,
                              rad.held_top_escape_power_w,
                              rad.held_lateral_escape_power_w,
                              rad.held_unresolved_reflected_power_w})
    {
        if (power != 0.0)
        {
            throw std::runtime_error("primary microphysics endpoints have held radiation power");
        }
    }
}

bool same_endpoint_water_attribution(const MicrophysicsWaterRepairReport& a,
                                     const MicrophysicsWaterRepairReport& b)
{
    if (a.nonfinite_tendency_count != b.nonfinite_tendency_count)
    {
        return false;
    }
    for (std::size_t index = 0; index < a.species.size(); ++index)
    {
        const auto& x = a.species[index];
        const auto& y = b.species[index];
        if (std::tie(x.storage_rounding_kg, x.lower_bound_correction_kg,
                     x.lower_bound_roundoff_correction_kg, x.upper_bound_correction_kg,
                     x.nonfinite_correction_kg, x.lower_bound_count,
                     x.lower_bound_roundoff_count, x.upper_bound_count, x.nonfinite_count) !=
            std::tie(y.storage_rounding_kg, y.lower_bound_correction_kg,
                     y.lower_bound_roundoff_correction_kg, y.upper_bound_correction_kg,
                     y.nonfinite_correction_kg, y.lower_bound_count,
                     y.lower_bound_roundoff_count, y.upper_bound_count, y.nonfinite_count))
        {
            return false;
        }
    }
    return true;
}

bool same_endpoint_heat_attribution(const MicrophysicsThermodynamicRepairReport& a,
                                    const MicrophysicsThermodynamicRepairReport& b)
{
    const auto fields = [](const MicrophysicsThermodynamicRepairReport& r)
    {
        return std::tie(r.microphysics_heat_j, r.radiation_heat_j, r.boundary_layer_heat_j,
            r.theta_tendency_limit_correction_j, r.theta_nonfinite_correction_j,
            r.theta_storage_rounding_j, r.theta_storage_residual_release_j,
            r.theta_storage_residual_j, r.theta_bound_correction_j,
            r.microphysics_water_tendency_kg, r.microphysics_water_tendency_kg_by_species,
            r.boundary_layer_vapor_tendency_kg, r.boundary_layer_cloud_liquid_tendency_kg,
            r.boundary_layer_cloud_ice_tendency_kg,
            r.moist_dry_internal_change_j, r.moist_water_sensible_change_j,
            r.moist_phase_reference_change_j, r.moist_carrier_kinetic_change_j,
            r.moist_gravitational_change_j, r.moist_subgrid_tke_change_j,
            r.moist_committed_energy_change_j, r.moist_energy_closure_residual_j,
            r.moist_energy_complete, r.theta_tendency_limit_count,
            r.theta_nonfinite_count, r.theta_lower_bound_count, r.theta_upper_bound_count);
    };
    return fields(a) == fields(b);
}

void commit_owned_primary_endpoints(const SimulationStateView& state,
                                    double duration,
                                    MicrophysicsScheme& scheme,
                                    const std::vector<float>& column_multipliers,
                                    int sanitized_tendencies)
{
    validate_primary_endpoint_profile(state);
    if (sanitized_tendencies != 0 || !scheme.owns_post_primary_private_commit() ||
        scheme.requires_runtime_column_donor_admission())
    {
        throw std::runtime_error("primary microphysics endpoint owner is incomplete");
    }
    auto& runtime = tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime");
    auto& f = state.fields;
    const std::array<const Field3D*, 10> sources{
        &f.rho, &f.p, &f.theta, &f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
    std::array<Field3D, 10> before;
    for (std::size_t index = 0; index < sources.size(); ++index)
    {
        if (sources[index]->size_r() != state.grid.nr ||
            sources[index]->size_th() != state.grid.nth ||
            sources[index]->size_z() != state.grid.nz)
        {
            throw std::runtime_error("primary microphysics endpoint source shape differs");
        }
        for (std::size_t other = 0; other < index; ++other)
        {
            if (sources[index] == sources[other] ||
                sources[index]->data() == sources[other]->data())
            {
                throw std::runtime_error("primary microphysics endpoint source is aliased");
            }
        }
        before[index] = *sources[index];
    }
    struct SourceRollback
    {
        std::array<Field3D*, 10> fields;
        std::array<Field3D, 10>& before;
        bool committed = false;

        ~SourceRollback()
        {
            if (!committed)
            {
                for (std::size_t index = 0; index < fields.size(); ++index)
                {
                    std::swap(*fields[index], before[index]);
                }
            }
        }
    } source_rollback{{&f.rho, &f.p, &f.theta, &f.qv, &f.qc,
                       &f.qr, &f.qi, &f.qs, &f.qg, &f.qh}, before};
    MicrophysicsPrimaryEndpointCandidate candidate;
    std::string error;
    if (!scheme.prepare_primary_endpoint_commit(state, duration, candidate, error))
    {
        throw std::runtime_error("primary microphysics endpoint candidate rejected: " + error);
    }
    validate_primary_endpoint_profile(state);
    if (candidate.owner != &scheme || candidate.generation == 0 ||
        candidate.duration_s != duration || candidate.nr != state.grid.nr ||
        candidate.nth != state.grid.nth || candidate.nz != state.grid.nz ||
        candidate.source_fields != sources ||
        candidate.theta_increment.size() != f.rho.size())
    {
        throw std::runtime_error("primary microphysics endpoint provenance differs");
    }
    for (std::size_t index = 0; index < sources.size(); ++index)
    {
        if (sources[index]->size_r() != before[index].size_r() ||
            sources[index]->size_th() != before[index].size_th() ||
            sources[index]->size_z() != before[index].size_z() ||
            sources[index]->size() != before[index].size() ||
            std::memcmp(sources[index]->data(), before[index].data(),
                        before[index].size() * sizeof(float)) != 0)
        {
            throw std::runtime_error("primary microphysics endpoint getter changed its source");
        }
    }
    for (const auto& water : candidate.water_after)
    {
        if (water.size() != f.rho.size())
        {
            throw std::runtime_error("primary microphysics endpoint payload is partial");
        }
    }
    if (!scheme.validate_primary_endpoint_commit(state, duration, candidate, error))
    {
        throw std::runtime_error("primary microphysics endpoint validation rejected: " + error);
    }
    const auto domain = physical_scalar_domain(state.grid);
    if (domain.empty())
    {
        throw std::runtime_error("primary microphysics endpoint physical domain is empty");
    }
    const std::array<Field3D*, 7> water_fields{&f.qv, &f.qc, &f.qr, &f.qi, &f.qs, &f.qg, &f.qh};
    std::array<Field3D, 7> staged_water;
    for (std::size_t species = 0; species < staged_water.size(); ++species)
    {
        staged_water[species] = *water_fields[species];
    }
    Field3D staged_theta(f.theta);
    std::vector<MicrophysicsWaterRepairReport> water_reports(repair_report_accumulator_count(domain));
    std::vector<MicrophysicsThermodynamicRepairReport> heat_reports(repair_report_accumulator_count(domain));
    std::vector<std::array<double, 3>> conservation_partials(repair_report_accumulator_count(domain));
    const bool attribute_energy = moist_basis_v2_commit_inputs_available(state);
    for (int i = 0; i < state.grid.nr; ++i)
    {
        for (int j = 0; j < state.grid.nth; ++j)
        {
            for (int k = 0; k < state.grid.nz; ++k)
            {
                const std::size_t cell = (static_cast<std::size_t>(i) * state.grid.nth + j) * state.grid.nz + k;
                const bool physical = i >= domain.i_begin && i < domain.i_end &&
                    j >= domain.j_begin && j < domain.j_end && k >= domain.k_begin && k < domain.k_end;
                if (!std::isfinite(candidate.theta_increment[cell]))
                {
                    throw std::runtime_error("primary microphysics theta increment is invalid");
                }
                for (std::size_t species = 0; species < water_fields.size(); ++species)
                {
                    const double endpoint = candidate.water_after[species][cell];
                    const float upper = species == 0 ? qv_max_kgkg : hydrometeor_max_kgkg;
                    if (!std::isfinite(endpoint) || endpoint < 0.0 || endpoint > upper ||
                        (!physical && endpoint != before[3 + species].data()[cell]))
                    {
                        throw std::runtime_error("primary microphysics water endpoint is invalid");
                    }
                }
                if (!physical && candidate.theta_increment[cell] != 0.0)
                {
                    throw std::runtime_error("primary microphysics endpoint changed a boundary owner");
                }
            }
        }
    }
    const bool publish_precipitation = prepare_surface_precipitation_candidates(
        state, duration, scheme, column_multipliers, runtime.execution_scratch);
    #pragma omp parallel for schedule(static)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        auto& water_report = water_reports[repair_report_accumulator_index(domain, i, 0)];
        auto& heat_report = heat_reports[repair_report_accumulator_index(domain, i, 0)];
        auto& conservation = conservation_partials[repair_report_accumulator_index(domain, i, 0)];
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::size_t cell = (static_cast<std::size_t>(i) * state.grid.nth + j) * state.grid.nz + k;
                const double volume = diagnostic_cell_volume(state, i, j, k);
                const double dry_mass = live_water_mass_density(state, i, j, k) * volume;
                const double heat_factor = theta_heat_factor(state, i, j, k, volume);
                const float theta_after = project_theta_update(
                    heat_report, f.theta(i, j, k), 0.0, nullptr,
                    candidate.theta_increment[cell], 0.0, 0.0, heat_factor, 50.0, true);
                staged_theta(i, j, k) = theta_after;
                std::array<double, 7> old_water{};
                std::array<double, 7> new_water{};
                double change = 0.0;
                double baseline = 0.0;
                for (std::size_t species = 0; species < water_fields.size(); ++species)
                {
                    const float old_value = water_fields[species]->data()[cell];
                    const double endpoint = candidate.water_after[species][cell];
                    const float stored = static_cast<float>(endpoint);
                    observe_water_repair(water_report, species, old_value, endpoint, stored, stored, dry_mass);
                    staged_water[species].data()[cell] = stored;
                    old_water[species] = old_value;
                    new_water[species] = stored;
                    const double delta = endpoint - old_value;
                    heat_report.microphysics_water_tendency_kg_by_species[species] += delta * dry_mass;
                    change += delta;
                    baseline += old_value;
                }
                conservation[0] += change * dry_mass;
                conservation[1] += baseline * dry_mass;
                if (attribute_energy && !observe_moist_basis_v2_commit(
                        heat_report, dry_mass,
                        heat_factor / (dry_mass * physical_constants::specific_heat_cp_jkgk),
                        moist_basis_v2_specific_kinetic_energy(state, nullptr, domain.k_begin, i, j, k),
                        tmv::physical_grid_metrics::scalar_height(state, i, j, k),
                        f.theta(i, j, k), old_water, theta_after, new_water))
                {
                    heat_report.moist_committed_energy_change_j = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }
    MicrophysicsWaterRepairReport water_report;
    MicrophysicsThermodynamicRepairReport heat_report;
    for (std::size_t slab = 0; slab < water_reports.size(); ++slab)
    {
        merge_water_repair_report(water_report, water_reports[slab]);
        merge_thermodynamic_repair_report(heat_report, heat_reports[slab]);
    }
    close_microphysics_species_total(heat_report);
    if (attribute_energy)
    {
        finalize_moist_basis_v2_commit(heat_report, "owned primary microphysics endpoints");
    }
    enforce_strict_water_repair_policy(state, water_report);
    enforce_strict_thermodynamic_repair_policy(state, heat_report);
    MicrophysicsConservationReport conservation;
    if (microphysics_conservation_gate_requested(state))
    {
        conservation = finalize_microphysics_conservation_gate(conservation_partials, duration);
        report_microphysics_conservation_gate(state, conservation);
    }
    for (std::size_t species = 0; species < water_fields.size(); ++species)
    {
        std::copy(staged_water[species].data(), staged_water[species].data() + staged_water[species].size(),
                  water_fields[species]->data());
    }
    std::copy(staged_theta.data(), staged_theta.data() + staged_theta.size(), f.theta.data());
    runtime.last_water_repair_report = water_report;
    runtime.last_thermodynamic_repair_report = heat_report;
    runtime.last_conservation_report = conservation;
    const MicrophysicsAcceptedPrimaryFields accepted{
        f.rho, f.qv, f.qc, f.qr, f.qi, f.qs, f.qg, f.qh,
        &runtime.last_water_repair_report, &runtime.last_thermodynamic_repair_report, &candidate};
    if (!scheme.prepare_post_primary_private_commit(duration, accepted, error))
    {
        throw std::runtime_error("post-primary microphysics private candidate rejected: " + error);
    }
    if (!same_endpoint_water_attribution(water_report, runtime.last_water_repair_report) ||
        !same_endpoint_heat_attribution(heat_report, runtime.last_thermodynamic_repair_report))
    {
        throw std::runtime_error("primary microphysics accepted attribution changed");
    }
    if (!scheme.commit_prepared_private_candidate())
    {
        throw std::runtime_error("post-primary microphysics private candidate was not prepared");
    }
    auto& advection = tmv::require_level_runtime(state.advection_runtime, "advection_runtime");
    std::array<bool, AdvectionRuntimeState::hydrometeor_count> activity{};
    for (int i = domain.i_begin; i < domain.i_end; ++i)
    {
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const std::array<float, 6> values{
                    f.qc(i, j, k), f.qr(i, j, k), f.qi(i, j, k),
                    f.qs(i, j, k), f.qh(i, j, k), f.qg(i, j, k)};
                for (std::size_t species = 0; species < activity.size(); ++species)
                {
                    activity[species] = activity[species] || values[species] > 0.0f;
                }
            }
        }
    }
    advection.publish_hydrometeor_activity(activity, state.grid.nr, state.grid.nth, state.grid.nz);
    if (publish_precipitation)
    {
        commit_surface_precipitation_candidates(state, duration, runtime.execution_scratch);
    }
    source_rollback.committed = true;
}

}  // namespace

static void step_microphysics_once(const SimulationStateView& state, double dt_micro)
{
    if (!std::isfinite(dt_micro) || dt_micro <= 0.0)
    {
        throw std::invalid_argument(
            "microphysics timestep must be positive and finite");
    }

    MicrophysicsScheme* active_scheme = state.microphysics_runtime
        ? state.microphysics_runtime->scheme.get()
        : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime").scheme.get();
    if (!active_scheme)
    {
        throw std::runtime_error(
            "microphysics step requested before a scheme was initialized");
    }

#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent preflight_allocations(
        tmv::allocation_telemetry::Component::MicrophysicsPreflight);
#endif
    // Dynamics can change density after the held PBL source was prepared for
    // momentum. Reproject the same conserved rates for scalar application.
    const auto pbl_reprojection_t0 = microphysics_perf_start();
    prepare_boundary_layer_tendencies_for_application(state);
    microphysics_perf_finish(
        microphysics_perf_totals.pbl_reprojection_s, pbl_reprojection_t0);
    const auto input_validation_t0 = microphysics_perf_start();
    validate_microphysics_step_inputs(state);
    microphysics_perf_finish(
        microphysics_perf_totals.input_validation_s, input_validation_t0);
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    preflight_allocations.restore();
#endif

    MicrophysicsRuntimeState& active_microphysics_runtime =
        state.microphysics_runtime
            ? *state.microphysics_runtime : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime");
    RadiationRuntimeState& active_radiation_runtime =
        state.radiation_runtime
            ? *state.radiation_runtime : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime");
    const bool compensate_theta_storage =
        active_radiation_runtime.config.scheme_id != "none";
    Field3D& theta_storage_residual =
        active_microphysics_runtime.theta_storage_residual;
    MicrophysicsExecutionScratch& execution_scratch =
        active_microphysics_runtime.execution_scratch;
    std::vector<float>& stochastic_column_multipliers =
        execution_scratch.stochastic_column_multipliers;
    if (compensate_theta_storage &&
        (theta_storage_residual.size_r() != state.grid.nr ||
         theta_storage_residual.size_th() != state.grid.nth ||
         theta_storage_residual.size_z() != state.grid.nz))
    {
        theta_storage_residual.resize(
            state.grid.nr, state.grid.nth, state.grid.nz, 0.0f);
    }
    MicrophysicsTendencyFields& tendencies = state.microphysics_runtime
        ? state.microphysics_runtime->tendencies
        : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime").tendencies;
    MicrophysicsWaterRepairReport& last_water_repair_report =
        state.microphysics_runtime
            ? state.microphysics_runtime->last_water_repair_report
            : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime").last_water_repair_report;
    MicrophysicsThermodynamicRepairReport& last_thermodynamic_repair_report =
        state.microphysics_runtime
            ? state.microphysics_runtime->last_thermodynamic_repair_report
            : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime").last_thermodynamic_repair_report;
    Field3D& dtheta_dt = tendencies.dtheta_dt;
    Field3D& dqv_dt = tendencies.dqv_dt;
    Field3D& dqc_dt = tendencies.dqc_dt;
    Field3D& dqr_dt = tendencies.dqr_dt;
    Field3D& dqi_dt = tendencies.dqi_dt;
    Field3D& dqs_dt = tendencies.dqs_dt;
    Field3D& dqg_dt = tendencies.dqg_dt;
    Field3D& dqh_dt = tendencies.dqh_dt;

    const PhysicalCellDomain physical_domain = physical_scalar_domain(state.grid);
    const TerrainMetrics3D* terrain_metrics =
        tmv::physical_grid_metrics::active_terrain_metrics(state);
    const bool attribute_moist_energy =
        moist_basis_v2_commit_inputs_available(state);
    const bool gpu_apply_candidate =
        !compensate_theta_storage
        && gpu_mp_apply_enabled(state)
        && physical_domain.k_begin == 0
        && physical_domain.k_end == state.grid.nz
        && active_compute_backend_supports(tmv::require_level_runtime(state.compute_runtime, "compute runtime"),
               MicrophysicsKernelOp::MicrophysicsApply);
    // Strict/diagnostic execution deliberately retains an independent full-
    // field validation sweep. The ordinary CPU path guards each tendency as
    // the application loop consumes it, avoiding eight extra volume scans.
    const bool independently_validate_tendencies =
        gpu_apply_candidate
        || tmv::require_run_configuration(state).validation_policy.mode == GuardMode::Strict
        || step_diagnostics_enabled();

    int sanitized_microphysics_theta_tendency_count = 0;
    bool stochastic_column_control_active = false;
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent tendency_allocations(
        tmv::allocation_telemetry::Component::MicrophysicsTendencies);
#endif
    const int non_finite_tendency_sanitized =
        prepare_microphysics_tendencies(
            state, dt_micro, tendencies, *active_scheme,
            stochastic_column_multipliers,
            stochastic_column_control_active,
            independently_validate_tendencies,
            sanitized_microphysics_theta_tendency_count);
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tendency_allocations.restore();
    tmv::allocation_telemetry::ScopedComponent apply_allocations(
        tmv::allocation_telemetry::Component::MicrophysicsApply);
#endif

    if (active_scheme->owns_primary_endpoint_commit() &&
        (stochastic_column_control_active || active_scheme->requires_runtime_column_donor_admission()))
    {
        throw std::runtime_error("primary microphysics endpoints cannot share column donor admission");
    }
    if (stochastic_column_control_active
        || active_scheme->requires_runtime_column_donor_admission())
    {
        const auto admission_t0 = microphysics_perf_start();
        admit_and_apply_microphysics_column_multipliers(
            state, dt_micro, tendencies, *active_scheme,
            stochastic_column_multipliers);
        microphysics_perf_finish(microphysics_perf_totals.column_admission_s, admission_t0);
    }

    const auto private_state_t0 = microphysics_perf_start();
    std::string prognostic_update_error;
    if (!active_scheme->validate_prognostic_update(
            dt_micro, prognostic_update_error))
    {
        throw std::runtime_error(
            "microphysics private-prognostic update rejected: "
            + prognostic_update_error);
    }
    microphysics_perf_finish(
        microphysics_perf_totals.private_state_s, private_state_t0);
    if (active_scheme->owns_primary_endpoint_commit())
    {
        const auto owned_apply_t0 = microphysics_perf_start();
        commit_owned_primary_endpoints(state, dt_micro, *active_scheme,
                                      stochastic_column_multipliers, non_finite_tendency_sanitized);
        microphysics_perf_finish(microphysics_perf_totals.apply_s, owned_apply_t0);
        return;
    }
    const auto precipitation_t0 = microphysics_perf_start();
    const bool publish_surface_precipitation =
        prepare_surface_precipitation_candidates(
            state, dt_micro, *active_scheme,
            stochastic_column_multipliers, execution_scratch);
    microphysics_perf_finish(
        microphysics_perf_totals.precipitation_s, precipitation_t0);

    // Moisture conservation gate.
    //
    // Integrate the MASS imbalance of the microphysics scheme's per-species
    // q tendencies over the full domain: sum rho_d(i,j,k) dq dV. For an
    // internally conservative scheme this is zero up to the credited
    // surface sink: phase changes preserve total water within a cell under
    // any weighting, and sedimentation conserves the FALLING MASS rho q Vt
    // (SCIENCE.md M11, SCI-024) - its receiver weighting
    // rho_d(k)/rho_d(k-1) cancels exactly in this live-density-weighted
    // integral and ONLY in it (the
    // unweighted sum q dV shrinks whenever rain falls into denser air,
    // which false-failed this gate on mature storms). A non-zero residual
    // therefore directly measures non-conservation in the scheme's own
    // process rates - independent of PBL surface evaporation (which writes
    // to dqv_dt_pbl, not the mp tendency buffer) and independent of
    // post-apply repairs. Those repairs can add mass at the lower bound,
    // remove mass at the upper bound, or replace a non-finite value; their
    // signed contribution is recorded in a separate ledger below.
    //
    // Threshold tracks BUDGET_THRESH_KERNEL (fp32 storage noise floor for
    // q_volume_total on a typical 32x32x24 grid). 100x that
    // raises a louder GATE FAIL message that flags scheme regressions.
    const bool conservation_gate_active =
        microphysics_conservation_gate_requested(state);
    MicrophysicsConservationReport mp_gate{};
#ifdef NIMBUS_SEPARATE_MICROPHYSICS_CONSERVATION_SCAN
    mp_gate =
        evaluate_microphysics_conservation_gate(
            state, dt_micro, tendencies, *active_scheme,
            stochastic_column_multipliers);
    active_microphysics_runtime.last_conservation_report = mp_gate;
    report_microphysics_conservation_gate(state, mp_gate);
#else
    active_microphysics_runtime.last_conservation_report = {};
#endif

    constexpr float max_theta_step_change_k = 50.0f;
    int non_finite_tendency_sanitized_total = non_finite_tendency_sanitized;
    int non_finite_theta_tendency_count = 0;
    int theta_tendency_limited_count = 0;
    int theta_bounds_clamp_count = 0;

    const auto& grid = state.grid;
    auto& fields = state.fields;
    auto& coupling_tendencies = state.tendencies;
    const RadiationRuntimeState& radiation_runtime = state.radiation_runtime
        ? *state.radiation_runtime : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime");

    // GPU execution needs a host-side ledger before dispatch, and strict mode
    // must reject transactionally before mutation. The normal CPU path folds
    // this exact projection into its consuming application pass below.
    const bool project_thermodynamics_before_apply =
        gpu_apply_candidate
        || tmv::require_run_configuration(state).validation_policy.mode == GuardMode::Strict;
    auto* projected_conservation_partials =
#ifdef NIMBUS_SEPARATE_MICROPHYSICS_CONSERVATION_SCAN
        static_cast<std::vector<std::array<double, 3>>*>(nullptr);
#else
        project_thermodynamics_before_apply && conservation_gate_active
            ? &execution_scratch.conservation_report_partials
            : nullptr;
#endif
    const auto projection_t0 = microphysics_perf_start();
    last_thermodynamic_repair_report = project_thermodynamics_before_apply
        ? evaluate_projected_thermodynamic_repairs(
              state, dt_micro, tendencies, *active_scheme,
              stochastic_column_multipliers,
              compensate_theta_storage ? &theta_storage_residual : nullptr,
              max_theta_step_change_k,
              execution_scratch.thermodynamic_report_partials,
              projected_conservation_partials)
        : MicrophysicsThermodynamicRepairReport{};
    microphysics_perf_finish(
        microphysics_perf_totals.projection_s, projection_t0);
#ifndef NIMBUS_SEPARATE_MICROPHYSICS_CONSERVATION_SCAN
    if (projected_conservation_partials != nullptr)
    {
        mp_gate = finalize_microphysics_conservation_gate(
            *projected_conservation_partials, dt_micro);
        active_microphysics_runtime.last_conservation_report = mp_gate;
        report_microphysics_conservation_gate(state, mp_gate);
    }
#endif
    if (tmv::require_run_configuration(state).validation_policy.mode == GuardMode::Strict)
    {
        // Strict water-repair failures are transactional: project the exact
        // bounded update and reject it before mutating prognostic state. The
        // normal CPU path still records repairs in its apply pass, and the GPU
        // path uses the same projection because its telemetry ABI carries only
        // integer clamp counters.
        const auto strict_projection_t0 = microphysics_perf_start();
        last_water_repair_report =
            evaluate_projected_water_repairs(
                state, dt_micro, tendencies,
                execution_scratch.water_report_partials);
        last_water_repair_report.nonfinite_tendency_count =
            static_cast<std::size_t>(
                std::max(0, non_finite_tendency_sanitized_total
                              - sanitized_microphysics_theta_tendency_count));
        report_microphysics_water_repairs(last_water_repair_report);
        enforce_strict_water_repair_policy(state, last_water_repair_report);
        report_microphysics_thermodynamic_repairs(
            last_thermodynamic_repair_report);
        enforce_strict_thermodynamic_repair_policy(state,
            last_thermodynamic_repair_report);
        microphysics_perf_finish(
            microphysics_perf_totals.projection_s, strict_projection_t0);
    }

    auto& clamp_count_by_k = execution_scratch.clamp_count_by_k;
    auto& clamp_count_by_i = execution_scratch.clamp_count_by_i;
    clamp_count_by_k.assign(static_cast<std::size_t>(grid.nz), 0);
    clamp_count_by_i.assign(static_cast<std::size_t>(grid.nr), 0);
    int first_clamp_i = -1;
    int first_clamp_j = -1;
    int first_clamp_k = -1;
    double first_clamp_theta_new = 0.0;
    double first_clamp_theta_bounded = 0.0;
    double first_clamp_dtheta_total = 0.0;
    double first_clamp_dtheta_limited = 0.0;
    auto& cpu_water_slab_reports =
        execution_scratch.water_report_partials;
    auto& cpu_thermodynamic_slab_reports =
        execution_scratch.thermodynamic_report_partials;
    cpu_thermodynamic_slab_reports.assign(
        repair_report_accumulator_count(physical_domain),
        MicrophysicsThermodynamicRepairReport{});
    auto& cpu_conservation_slab_partials =
        execution_scratch.conservation_report_partials;
#ifndef NIMBUS_SEPARATE_MICROPHYSICS_CONSERVATION_SCAN
    const bool accumulate_conservation_in_cpu_apply =
        !project_thermodynamics_before_apply && conservation_gate_active;
    if (accumulate_conservation_in_cpu_apply)
    {
        cpu_conservation_slab_partials.assign(
            repair_report_accumulator_count(physical_domain),
            std::array<double, 3>{});
    }
#else
    constexpr bool accumulate_conservation_in_cpu_apply = false;
#endif
    int qc_activity = 0;
    int qr_activity = 0;
    int qi_activity = 0;
    int qs_activity = 0;
    int qh_activity = 0;
    int qg_activity = 0;

    // CPU apply loop, shared by the no-GPU fallback and the parity replay
    // (GPU audit Phase 4.2).
    const auto cpu_apply = [&]()
    {
    // Device projection shares this scratch; a declined dispatch must not
    // count its predicted rounding again when the CPU commits the update.
    cpu_water_slab_reports.assign(
        repair_report_accumulator_count(physical_domain),
        MicrophysicsWaterRepairReport{});
    #pragma omp parallel for schedule(static) reduction(+:non_finite_tendency_sanitized_total, sanitized_microphysics_theta_tendency_count, non_finite_theta_tendency_count, theta_tendency_limited_count, theta_bounds_clamp_count) reduction(|:qc_activity, qr_activity, qi_activity, qs_activity, qh_activity, qg_activity)
    for (int i = physical_domain.i_begin; i < physical_domain.i_end; ++i)
    {
        std::array<double, 3>* conservation_partial =
            accumulate_conservation_in_cpu_apply
            ? &cpu_conservation_slab_partials[
                repair_report_accumulator_index(physical_domain, i, 0)]
            : nullptr;
        for (int j = physical_domain.j_begin; j < physical_domain.j_end; ++j)
        {
            auto& water_report = cpu_water_slab_reports[
                repair_report_accumulator_index(physical_domain, i, j)];
            auto& thermodynamic_report =
                cpu_thermodynamic_slab_reports[
                    repair_report_accumulator_index(
                        physical_domain, i, j)];
            if (conservation_partial != nullptr)
            {
                const double rho_d_sfc =
                    live_water_mass_density(
                        state, i, j, physical_domain.k_begin);
                (*conservation_partial)[2] +=
                    active_scheme->bottom_sedimentation_sink_rate(
                        fields.rho, fields.qr, fields.qs,
                        fields.qg, fields.qh, i, j, dt_micro)
                    * static_cast<double>(
                        stochastic_column_multipliers[
                            static_cast<std::size_t>(i)
                                * static_cast<std::size_t>(grid.nth)
                            + static_cast<std::size_t>(j)])
                    * rho_d_sfc
                    * diagnostic_cell_volume(
                        state, i, j, physical_domain.k_begin);
            }
            for (int k = physical_domain.k_begin; k < physical_domain.k_end; ++k)
            {
                const double cell_dV = diagnostic_cell_volume(state, i, j, k);
                const double kg_per_kgkg =
                    live_water_mass_density(state, i, j, k) * cell_dV;
                float micro_dtheta = dtheta_dt[i][j][k];
                float micro_dqv = dqv_dt[i][j][k];
                float micro_dqc = dqc_dt[i][j][k];
                float micro_dqr = dqr_dt[i][j][k];
                float micro_dqi = dqi_dt[i][j][k];
                float micro_dqs = dqs_dt[i][j][k];
                float micro_dqg = dqg_dt[i][j][k];
                float micro_dqh = dqh_dt[i][j][k];
                if (!std::isfinite(micro_dtheta))
                {
                    micro_dtheta = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                    ++sanitized_microphysics_theta_tendency_count;
                }
                if (!std::isfinite(micro_dqv))
                {
                    micro_dqv = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                }
                if (!std::isfinite(micro_dqc))
                {
                    micro_dqc = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                }
                if (!std::isfinite(micro_dqr))
                {
                    micro_dqr = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                }
                if (!std::isfinite(micro_dqi))
                {
                    micro_dqi = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                }
                if (!std::isfinite(micro_dqs))
                {
                    micro_dqs = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                }
                if (!std::isfinite(micro_dqg))
                {
                    micro_dqg = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                }
                if (!std::isfinite(micro_dqh))
                {
                    micro_dqh = 0.0f;
                    ++non_finite_tendency_sanitized_total;
                }

                const double theta_old = static_cast<double>(fields.theta[i][j][k]);
                const std::array<double, 7> water_old{{
                    static_cast<double>(fields.qv[i][j][k]),
                    static_cast<double>(fields.qc[i][j][k]),
                    static_cast<double>(fields.qr[i][j][k]),
                    static_cast<double>(fields.qi[i][j][k]),
                    static_cast<double>(fields.qs[i][j][k]),
                    static_cast<double>(fields.qg[i][j][k]),
                    static_cast<double>(fields.qh[i][j][k])}};
                if (conservation_partial != nullptr)
                {
                    const double dq_sum =
                        static_cast<double>(dqv_dt[i][j][k])
                        + static_cast<double>(dqc_dt[i][j][k])
                        + static_cast<double>(dqr_dt[i][j][k])
                        + static_cast<double>(dqi_dt[i][j][k])
                        + static_cast<double>(dqs_dt[i][j][k])
                        + static_cast<double>(dqg_dt[i][j][k])
                        + static_cast<double>(dqh_dt[i][j][k]);
                    const double q_sum =
                        water_old[0] + water_old[1] + water_old[2]
                        + water_old[3] + water_old[4] + water_old[5]
                        + water_old[6];
                    const double rho_d_k =
                        live_water_mass_density(state, i, j, k);
                    (*conservation_partial)[0] +=
                        rho_d_k * dq_sum * dt_micro * cell_dV;
                    (*conservation_partial)[1] +=
                        rho_d_k * q_sum * cell_dV;
                }
                const double theta_residual_old = compensate_theta_storage
                    ? static_cast<double>(theta_storage_residual[i][j][k]) : 0.0;
                double theta_residual_new = 0.0;
                const double heat_factor =
                    theta_heat_factor(state, i, j, k, cell_dV);
                if (!project_thermodynamics_before_apply)
                {
                    const double radiation_delta = radiation_theta_delta(
                        state, radiation_runtime, i, j, k, dt_micro,
                        heat_factor);
                    (void)project_theta_update(
                        thermodynamic_report,
                        theta_old,
                        theta_residual_old,
                        compensate_theta_storage ? &theta_residual_new : nullptr,
                        static_cast<double>(dtheta_dt[i][j][k]) * dt_micro,
                        radiation_delta,
                        static_cast<double>(
                            coupling_tendencies.dtheta_dt_pbl[i][j][k])
                            * dt_micro,
                        heat_factor,
                        max_theta_step_change_k);

                    const double microphysics_dq =
                        static_cast<double>(dqv_dt[i][j][k])
                      + static_cast<double>(dqc_dt[i][j][k])
                      + static_cast<double>(dqr_dt[i][j][k])
                      + static_cast<double>(dqi_dt[i][j][k])
                      + static_cast<double>(dqs_dt[i][j][k])
                      + static_cast<double>(dqg_dt[i][j][k])
                      + static_cast<double>(dqh_dt[i][j][k]);
                    if (std::isfinite(microphysics_dq))
                    {
                        thermodynamic_report.microphysics_water_tendency_kg +=
                            microphysics_dq * dt_micro * kg_per_kgkg;
                    }
                    const std::array<double, 7> microphysics_species{{
                        static_cast<double>(dqv_dt[i][j][k]),
                        static_cast<double>(dqc_dt[i][j][k]),
                        static_cast<double>(dqr_dt[i][j][k]),
                        static_cast<double>(dqi_dt[i][j][k]),
                        static_cast<double>(dqs_dt[i][j][k]),
                        static_cast<double>(dqg_dt[i][j][k]),
                        static_cast<double>(dqh_dt[i][j][k])}};
                    for (std::size_t species = 0;
                         species < microphysics_species.size(); ++species)
                    {
                        if (std::isfinite(microphysics_species[species]))
                        {
                            thermodynamic_report
                                .microphysics_water_tendency_kg_by_species[species]
                                += microphysics_species[species] * dt_micro
                                 * kg_per_kgkg;
                        }
                    }
                    const double pbl_dqv = static_cast<double>(
                        coupling_tendencies.dqv_dt_pbl[i][j][k]);
                    if (std::isfinite(pbl_dqv))
                    {
                        thermodynamic_report.boundary_layer_vapor_tendency_kg +=
                            pbl_dqv * dt_micro * kg_per_kgkg;
                    }
                    const double pbl_dqc = boundary_layer_cloud_tendency(
                        state, coupling_tendencies.dqc_dt_pbl, i, j, k);
                    const double pbl_dqi = boundary_layer_cloud_tendency(
                        state, coupling_tendencies.dqi_dt_pbl, i, j, k);
                    if (std::isfinite(pbl_dqc))
                        thermodynamic_report
                            .boundary_layer_cloud_liquid_tendency_kg +=
                            pbl_dqc * dt_micro * kg_per_kgkg;
                    if (std::isfinite(pbl_dqi))
                        thermodynamic_report
                            .boundary_layer_cloud_ice_tendency_kg +=
                            pbl_dqi * dt_micro * kg_per_kgkg;
                }

                const double radiation_delta = radiation_theta_delta(
                    state, radiation_runtime, i, j, k, dt_micro,
                    heat_factor);
                double dtheta_total = static_cast<double>(micro_dtheta)
                                        * dt_micro
                                    + radiation_delta
                                    + static_cast<double>(
                                          coupling_tendencies.dtheta_dt_pbl[i][j][k])
                                        * dt_micro;
                if (!std::isfinite(dtheta_total))
                {
                    dtheta_total = 0.0;
                    ++non_finite_theta_tendency_count;
                }

                const double max_step = static_cast<double>(max_theta_step_change_k);
                double dtheta_limited = std::clamp(dtheta_total, -max_step, max_step);
                if (dtheta_limited != dtheta_total)
                {
                    ++theta_tendency_limited_count;
                }

                double theta_new = theta_old + theta_residual_old + dtheta_limited;
                // Keep the pre-step theta (its own altitude) on a non-finite
                // result, not the surface theta0 cold bomb (science audit
                // NaN-theta); theta0 only if theta_old is itself corrupt. The
                // microphysics_apply shader mirrors this.
                const bool theta_nonfinite = !std::isfinite(theta_new);
                if (theta_nonfinite)
                {
                    theta_new = std::isfinite(theta_old) ? theta_old : theta0;
                }
                float theta_bounded = clamp_theta_k(static_cast<float>(theta_new));
                if (theta_nonfinite || theta_bounded != static_cast<float>(theta_new))
                {
                    ++theta_bounds_clamp_count;
                    #pragma omp atomic
                    ++clamp_count_by_k[k];
                    #pragma omp atomic
                    ++clamp_count_by_i[i];
                    #pragma omp critical(microphysics_first_clamp)
                    {
                        if (first_clamp_k < 0)
                        {
                            first_clamp_i = i;
                            first_clamp_j = j;
                            first_clamp_k = k;
                            first_clamp_theta_new = theta_new;
                            first_clamp_theta_bounded = static_cast<double>(theta_bounded);
                            first_clamp_dtheta_total = dtheta_total;
                            first_clamp_dtheta_limited = dtheta_limited;
                        }
                    }
                }
                fields.theta[i][j][k] = theta_bounded;
                if (compensate_theta_storage)
                {
                    theta_storage_residual[i][j][k] =
                        (theta_nonfinite || theta_bounded != static_cast<float>(theta_new))
                            ? 0.0f
                            : static_cast<float>(
                                theta_new - static_cast<double>(theta_bounded));
                }

                fields.qv[i][j][k] = bounded_water_value(
                    water_report, 0, fields.qv[i][j][k],
                    (static_cast<double>(micro_dqv) +
                     static_cast<double>(
                         coupling_tendencies.dqv_dt_pbl[i][j][k])) * dt_micro,
                    kg_per_kgkg, qv_max_kgkg);
                fields.qc[i][j][k] = bounded_water_value(
                    water_report, 1, fields.qc[i][j][k],
                    (static_cast<double>(micro_dqc)
                     + boundary_layer_cloud_tendency(
                         state, coupling_tendencies.dqc_dt_pbl, i, j, k))
                        * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                fields.qr[i][j][k] = bounded_water_value(
                    water_report, 2, fields.qr[i][j][k],
                    static_cast<double>(micro_dqr) * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                fields.qi[i][j][k] = bounded_water_value(
                    water_report, 3, fields.qi[i][j][k],
                    (static_cast<double>(micro_dqi)
                     + boundary_layer_cloud_tendency(
                         state, coupling_tendencies.dqi_dt_pbl, i, j, k))
                        * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                fields.qs[i][j][k] = bounded_water_value(
                    water_report, 4, fields.qs[i][j][k],
                    static_cast<double>(micro_dqs) * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                fields.qg[i][j][k] = bounded_water_value(
                    water_report, 5, fields.qg[i][j][k],
                    static_cast<double>(micro_dqg) * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                fields.qh[i][j][k] = bounded_water_value(
                    water_report, 6, fields.qh[i][j][k],
                    static_cast<double>(micro_dqh) * dt_micro,
                    kg_per_kgkg, hydrometeor_max_kgkg);
                const std::array<double, 7> water_new{{
                    static_cast<double>(fields.qv[i][j][k]),
                    static_cast<double>(fields.qc[i][j][k]),
                    static_cast<double>(fields.qr[i][j][k]),
                    static_cast<double>(fields.qi[i][j][k]),
                    static_cast<double>(fields.qs[i][j][k]),
                    static_cast<double>(fields.qg[i][j][k]),
                    static_cast<double>(fields.qh[i][j][k])}};
                if (attribute_moist_energy && !project_thermodynamics_before_apply
                    && !observe_moist_basis_v2_commit(
                        thermodynamic_report,
                        kg_per_kgkg,
                        heat_factor
                            / (kg_per_kgkg
                               * physical_constants::
                                   specific_heat_cp_jkgk),
                        moist_basis_v2_specific_kinetic_energy(
                            state, terrain_metrics,
                            physical_domain.k_begin, i, j, k),
                        tmv::physical_grid_metrics::scalar_height(
                            state, i, j, k),
                        theta_old, water_old,
                        static_cast<double>(fields.theta[i][j][k]),
                        water_new))
                {
                    thermodynamic_report.moist_committed_energy_change_j =
                        std::numeric_limits<double>::quiet_NaN();
                }

                // Transport may skip only an exactly absent category. A
                // semantic epsilon leaves trace but finite water stationary
                // while density evolves, changing its mass without a face
                // flux and making the boundary ledger incomplete.
                qc_activity |= fields.qc[i][j][k] != 0.0f;
                qr_activity |= fields.qr[i][j][k] != 0.0f;
                qi_activity |= fields.qi[i][j][k] != 0.0f;
                qs_activity |= fields.qs[i][j][k] != 0.0f;
                qh_activity |= fields.qh[i][j][k] != 0.0f;
                qg_activity |= fields.qg[i][j][k] != 0.0f;
            }
        }
    }
    };

    // GPU apply (GPU audit Phase 4.2): one pointwise dispatch updates
    // all eight thermodynamic fields in place with the loop's exact guard
    // semantics; the stats buffer carries the guard counters and clamp
    // histograms so the warn blocks below stay identical. Logging profile is
    // deliberately absent from dispatch selection: observability must not
    // change state ownership.
    MicrophysicsWaterRepairReport projected_gpu_water_repairs;
    const bool try_gpu_apply = gpu_apply_candidate && gpu_mp_apply_commit_admitted(state);
    if (try_gpu_apply)
    {
        // The current GPU telemetry block contains integer clamp counters but
        // not density-weighted signed mass. Project the shader's exact update
        // before dispatch so GPU and CPU applications publish the same ledger.
        // This pass is GPU-only; the CPU path accumulates in its apply loop.
        const auto gpu_projection_t0 = microphysics_perf_start();
        projected_gpu_water_repairs =
            evaluate_projected_water_repairs(
                state, dt_micro, tendencies,
                execution_scratch.water_report_partials);
        microphysics_perf_finish(
            microphysics_perf_totals.projection_s, gpu_projection_t0);
    }

    const auto apply_t0 = microphysics_perf_start();
    const bool applied_on_gpu =
        try_gpu_apply
        && try_microphysics_apply_gpu(
               state, dt_micro, max_theta_step_change_k,
               dtheta_dt, dqv_dt, dqc_dt,
               dqr_dt, dqi_dt, dqs_dt, dqg_dt, dqh_dt, cpu_apply,
               non_finite_tendency_sanitized_total,
               non_finite_theta_tendency_count, theta_tendency_limited_count,
               theta_bounds_clamp_count, clamp_count_by_k, clamp_count_by_i,
               first_clamp_i, first_clamp_j, first_clamp_k,
               first_clamp_theta_new, first_clamp_theta_bounded,
               first_clamp_dtheta_total, first_clamp_dtheta_limited);
    if (!applied_on_gpu)
    {
        cpu_apply();
    }
    microphysics_perf_finish(microphysics_perf_totals.apply_s, apply_t0);

    const auto finalize_t0 = microphysics_perf_start();

#ifndef NIMBUS_SEPARATE_MICROPHYSICS_CONSERVATION_SCAN
    if (accumulate_conservation_in_cpu_apply)
    {
        mp_gate = finalize_microphysics_conservation_gate(
            cpu_conservation_slab_partials, dt_micro);
        active_microphysics_runtime.last_conservation_report = mp_gate;
        report_microphysics_conservation_gate(state, mp_gate);
    }
#endif

    if (!project_thermodynamics_before_apply)
    {
        for (const auto& slab_report : cpu_thermodynamic_slab_reports)
        {
            merge_thermodynamic_repair_report(
                last_thermodynamic_repair_report, slab_report);
        }
        if (attribute_moist_energy)
        {
            finalize_moist_basis_v2_commit(
                last_thermodynamic_repair_report,
                "shared thermodynamic commit");
        }
    }
    close_microphysics_species_total(last_thermodynamic_repair_report);

    AdvectionRuntimeState& advection_runtime = state.advection_runtime != nullptr
        ? *state.advection_runtime : tmv::require_level_runtime(state.advection_runtime, "advection_runtime");
    if (applied_on_gpu)
    {
        // The current GPU telemetry ABI does not export per-species activity.
        // Fail safe: the next transport performs one consolidated scan.
        advection_runtime.invalidate_hydrometeor_activity();
    }
    else
    {
        advection_runtime.publish_hydrometeor_activity(
            {qc_activity != 0, qr_activity != 0, qi_activity != 0,
             qs_activity != 0, qh_activity != 0, qg_activity != 0},
            grid.nr, grid.nth, grid.nz);
    }

    MicrophysicsWaterRepairReport water_repair_report;
    if (applied_on_gpu)
    {
        water_repair_report = projected_gpu_water_repairs;
    }
    else
    {
        for (const auto& slab_report : cpu_water_slab_reports)
        {
            merge_water_repair_report(water_repair_report, slab_report);
        }
    }
    water_repair_report.nonfinite_tendency_count =
        static_cast<std::size_t>(
            std::max(0, non_finite_tendency_sanitized_total
                          - sanitized_microphysics_theta_tendency_count));
    last_water_repair_report = water_repair_report;

    record_numerical_nonfinite_incidents(
        static_cast<std::size_t>(
            std::max(0, non_finite_tendency_sanitized_total))
        + last_water_repair_report.nonfinite_count()
        + last_thermodynamic_repair_report.theta_nonfinite_count);

    report_microphysics_guard_summary(
        non_finite_theta_tendency_count,
        non_finite_tendency_sanitized_total,
        theta_tendency_limited_count,
        theta_bounds_clamp_count);
    report_microphysics_clamp_detail(
        theta_bounds_clamp_count,
        clamp_count_by_k,
        clamp_count_by_i,
        first_clamp_i, first_clamp_j, first_clamp_k,
        first_clamp_theta_new, first_clamp_theta_bounded,
        first_clamp_dtheta_total, first_clamp_dtheta_limited);
    report_microphysics_water_repairs(last_water_repair_report);
    enforce_strict_water_repair_policy(state, last_water_repair_report);
    report_microphysics_thermodynamic_repairs(
        last_thermodynamic_repair_report);
    enforce_strict_thermodynamic_repair_policy(state,
        last_thermodynamic_repair_report);

    const std::size_t nonfinite_incidents =
        static_cast<std::size_t>(
            std::max(0, non_finite_tendency_sanitized_total))
        + last_water_repair_report.nonfinite_count()
        + last_thermodynamic_repair_report.theta_nonfinite_count;
    if (nonfinite_incidents > 0)
    {
        throw std::runtime_error(
            "microphysics update rejected non-finite tendency or state repair");
    }
    const MicrophysicsAcceptedPrimaryFields accepted{
        state.fields.rho, state.fields.qv, state.fields.qc,
        state.fields.qr, state.fields.qi, state.fields.qs,
        state.fields.qg, state.fields.qh,
        &last_water_repair_report,
        &last_thermodynamic_repair_report};
    std::string post_primary_error;
    if (!active_scheme->prepare_post_primary_private_commit(
            dt_micro, accepted, post_primary_error))
    {
        throw std::runtime_error(
            "post-primary microphysics private candidate rejected: " +
            post_primary_error);
    }
    if (active_scheme->owns_post_primary_private_commit())
    {
        if (!active_scheme->commit_prepared_private_candidate())
        {
            throw std::runtime_error(
                "post-primary microphysics private candidate was not prepared");
        }
    }
    else
    {
        active_scheme->commit_prognostic_update(dt_micro);
    }
    if (publish_surface_precipitation)
        commit_surface_precipitation_candidates(
            state, dt_micro, execution_scratch);
    microphysics_perf_finish(
        microphysics_perf_totals.finalize_s, finalize_t0);
}

namespace
{
double sedimentation_step_limit(const SimulationStateView& state, const MicrophysicsScheme& scheme)
{
    if (scheme.maximum_explicit_terminal_velocity_mps(hydrometeor_max_kgkg, 0.01) == 0.0)
        return std::numeric_limits<double>::infinity();
    const std::array<const Field3D*, 7> water{
        &state.fields.qv, &state.fields.qc, &state.fields.qr, &state.fields.qi,
        &state.fields.qs, &state.fields.qg, &state.fields.qh};
    const auto matches = [&](const Field3D& field) {
        return field.size_r() == state.grid.nr && field.size_th() == state.grid.nth &&
            field.size_z() == state.grid.nz;
    };
    if (!matches(state.fields.rho))
        throw std::invalid_argument("sedimentation density shape mismatch");
    for (const auto* field : water)
        if (!matches(*field)) throw std::invalid_argument("sedimentation species shape mismatch");
    physical_grid_metrics::validate(state);
    const auto domain = physical_scalar_domain(state.grid);
    double limit = std::numeric_limits<double>::infinity();
    const auto& chaos_config = state.chaos_runtime
        ? state.chaos_runtime->config : tmv::require_level_runtime(state.chaos_runtime, "chaos_runtime").config;
    const bool stochastic = chaos_config.scheme_id != "none" &&
        std::find(chaos_config.apply_to_tendencies.begin(), chaos_config.apply_to_tendencies.end(),
            "microphysics") != chaos_config.apply_to_tendencies.end();
    const double courant = stochastic ? 0.4 : 0.8;
    std::exception_ptr failure;
    std::size_t first_failed_column = std::numeric_limits<std::size_t>::max();
    const std::size_t columns = static_cast<std::size_t>(domain.i_end - domain.i_begin) *
        static_cast<std::size_t>(domain.j_end - domain.j_begin);
#pragma omp parallel for collapse(2) schedule(static) reduction(min:limit) if(columns >= 64)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
        {
            try
            {
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double density = state.fields.rho[i][j][k];
                    if (!std::isfinite(density) || density <= 0.0)
                        throw std::invalid_argument("sedimentation density must be positive and finite");
                    const double thickness = physical_grid_metrics::layer_thickness(state, i, j, k);
                    double available_water = 0.0;
                    for (const auto* field : water)
                    {
                        const double q = (*field)[i][j][k];
                        if (!std::isfinite(q) || q < 0.0)
                            throw std::invalid_argument("sedimentation input must be nonnegative and finite");
                        available_water += q;
                    }
                    // Conversion can create rain inside the interval, even when
                    // its initial precipitation fields are empty.
                    const double speed = scheme.maximum_explicit_terminal_velocity_mps(available_water, density);
                    if (!std::isfinite(speed) || speed < 0.0)
                        throw std::invalid_argument("sedimentation speed must be nonnegative and finite");
                    if (speed > 0.0) limit = std::min(limit, courant * thickness / speed);
                }
            }
            catch (...)
            {
                const std::size_t column = static_cast<std::size_t>(i) * state.grid.nth + j;
#pragma omp critical(nimbus_sedimentation_limit_failure)
                {
                    if (column < first_failed_column)
                    {
                        first_failed_column = column;
                        failure = std::current_exception();
                    }
                }
            }
        }
    if (failure) std::rethrow_exception(failure);
    return limit;
}
}

void step_microphysics(const SimulationStateView& state, double interval)
{
    if (!std::isfinite(interval) || interval <= 0.0)
        throw std::invalid_argument("microphysics timestep must be positive and finite");
    auto& runtime = state.microphysics_runtime ? *state.microphysics_runtime : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime");
    if (!runtime.scheme) throw std::runtime_error("microphysics scheme is not initialized");
    Field3D* protected_wind = nullptr;
    const Field3D* declared_wind = runtime.scheme->protected_primary_endpoint_wind_source();
    if (declared_wind)
    {
        if (!state.storage_owner || !runtime.scheme->owns_primary_endpoint_commit())
        {
            throw std::runtime_error("protected microphysics wind requires an owning primary endpoint scheme");
        }
        const auto canonical = make_simulation_state_view(*state.storage_owner);
        if (declared_wind != &canonical.fields.w || declared_wind != &state.fields.w)
        {
            throw std::runtime_error("protected microphysics wind is not its canonical source");
        }
        const std::array<const Field3D*, 10> primary_sources{
            &state.fields.rho, &state.fields.p, &state.fields.theta, &state.fields.qv, &state.fields.qc,
            &state.fields.qr, &state.fields.qi, &state.fields.qs, &state.fields.qg, &state.fields.qh};
        for (const Field3D* source : primary_sources)
        {
            if (declared_wind == source)
            {
                throw std::runtime_error("protected microphysics wind aliases a primary source");
            }
        }
        if (declared_wind->size_r() != state.grid.nr || declared_wind->size_th() != state.grid.nth ||
            declared_wind->size_z() != state.grid.nz)
        {
            throw std::runtime_error("protected microphysics wind source shape differs");
        }
        for (const Field3D* source : primary_sources)
        {
            if (!declared_wind->empty() && declared_wind->data() == source->data())
            {
                throw std::runtime_error("protected microphysics wind storage aliases a primary source");
            }
        }
        protected_wind = &state.fields.w;
    }
    const auto initial_limit_t0 = microphysics_perf_start();
    double limit = sedimentation_step_limit(state, *runtime.scheme);
    microphysics_perf_finish(microphysics_perf_totals.sedimentation_limit_s, initial_limit_t0);
    const auto transaction_snapshot_t0 = microphysics_perf_start();
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    tmv::allocation_telemetry::ScopedComponent transaction_allocations(
        tmv::allocation_telemetry::Component::MicrophysicsTransaction);
#endif
    if (!runtime.subcycle_workspace)
        runtime.subcycle_workspace = std::make_shared<MicrophysicsSubcycleWorkspace>();
    const auto workspace = runtime.subcycle_workspace;
    if (protected_wind)
    {
        workspace->protected_primary_wind = *protected_wind;
    }
    workspace->reaction_complete = false;
    workspace->reaction_owner = runtime.scheme.get();
    auto& chaos_runtime = state.chaos_runtime ? *state.chaos_runtime : tmv::require_level_runtime(state.chaos_runtime, "chaos_runtime");
    const bool capture_chaos = chaos_runtime.scheme && chaos_runtime.config.scheme_id != "none" &&
        std::find(chaos_runtime.config.apply_to_tendencies.begin(), chaos_runtime.config.apply_to_tendencies.end(),
            "microphysics") != chaos_runtime.config.apply_to_tendencies.end();
    if (capture_chaos)
    {
        if (!workspace->chaos) workspace->chaos = std::make_unique<chaos::ChaosRuntimeState>();
        *workspace->chaos = chaos_runtime;
    }
    const auto primary = fields::pointers(state.fields, fields::microphysics_state);
    const auto coupling = fields::pointers(state.tendencies, fields::physics_tendencies);
    std::array<ByteCopyRange, 16> snapshot_ranges{};
    const std::size_t snapshot_bytes = state.fields.theta.size() * sizeof(float);
    const auto capture_field = [&](Field3D& destination, const Field3D* source, std::size_t index)
    {
        if (!source) return;
        if (source->size() * sizeof(float) == snapshot_bytes &&
            destination.size_r() == source->size_r() &&
            destination.size_th() == source->size_th() &&
            destination.size_z() == source->size_z())
        {
            snapshot_ranges[index] = {destination.data(), source->data()};
        }
        else
        {
            // Shape changes may allocate; finish them before entering the copy team.
            destination = *source;
        }
    };
    for (std::size_t index = 0; index < primary.size(); ++index)
        capture_field(workspace->primary[index], primary[index], index);
    for (std::size_t index = 0; index < coupling.size(); ++index)
        capture_field(workspace->coupling[index], coupling[index], primary.size() + index);
    copy_byte_ranges(snapshot_ranges, snapshot_bytes);
    workspace->microphysics.capture_from(runtime);
    auto& advection = state.advection_runtime ? *state.advection_runtime : tmv::require_level_runtime(state.advection_runtime, "advection_runtime");
    auto activity = std::tie(advection.nr, advection.nth, advection.nz,
        advection.hydrometeor_active, advection.activity_current);
    workspace->advection = activity;
    auto& radiation = state.radiation_runtime ? *state.radiation_runtime : tmv::require_level_runtime(state.radiation_runtime, "radiation_runtime");
    auto radiation_ledger = std::tie(radiation.last_expected_atmospheric_energy_j,
        radiation.last_applied_radiation_heat_j, radiation.accumulated_expected_atmospheric_energy_j,
        radiation.accumulated_applied_radiation_heat_j, radiation.applied_step_count);
    const auto radiation_before = std::make_tuple(radiation.last_expected_atmospheric_energy_j,
        radiation.last_applied_radiation_heat_j, radiation.accumulated_expected_atmospheric_energy_j,
        radiation.accumulated_applied_radiation_heat_j, radiation.applied_step_count);
    const std::array<std::vector<float>*, 3> precipitation{
        state.surface.precipitation_rate_mm_h, state.surface.accumulated_rainfall_mm,
        state.surface.accumulated_rainfall_residual_mm};
    for (std::size_t index = 0; index < precipitation.size(); ++index)
        if (precipitation[index]) workspace->precipitation[index] = *precipitation[index];
    if (state.surface.accumulated_rainfall_last_update_s)
        workspace->precipitation_time = *state.surface.accumulated_rainfall_last_update_s;
    if (interval > limit)
        workspace->rate_integral.assign(static_cast<std::size_t>(state.grid.nr) * state.grid.nth, 0.0);
    const double start_time = state.time_s;
    if (!std::isfinite(start_time) || start_time < 0.0 ||
        !std::isfinite(start_time + interval) || start_time + interval <= start_time)
        throw std::invalid_argument("microphysics interval clock cannot advance");
    workspace->reaction_end_s = start_time + interval;
    workspace->reaction_interval_s = interval;
    workspace->reaction_origin = state.domain_frame ? state.domain_frame->earth_origin_at_t0_m
                                                     : tmv::domain::HorizontalPoint{};
    microphysics_perf_finish(microphysics_perf_totals.transaction_snapshot_s, transaction_snapshot_t0);
#ifdef NIMBUS_ALLOCATION_TELEMETRY
    transaction_allocations.restore();
#endif
    MicrophysicsWaterRepairReport water;
    MicrophysicsThermodynamicRepairReport heat;
    MicrophysicsConservationReport conservation;
    bool complete_energy = true;
    std::size_t substeps = 0;
    try
    {
        if (interval <= limit)
        {
            step_microphysics_once(state, interval);
            commit_radiation_applied_heat(state, interval,
                runtime.last_thermodynamic_repair_report.radiation_heat_j);
            ++microphysics_perf_totals.sedimentation_substeps;
            workspace->reaction_complete = true;
            return;
        }
        double elapsed = 0.0;
        while (elapsed < interval)
        {
            if (++substeps > 100000)
                throw std::runtime_error("microphysics interval requires too many sedimentation substeps");
            const double step = std::min(interval - elapsed, limit);
            if (!std::isfinite(step) || step <= 0.0 || elapsed + step <= elapsed)
                throw std::runtime_error("microphysics substep cannot advance time");
            state.time_s = start_time + elapsed;
            if (state.time_s + step <= state.time_s)
                throw std::runtime_error("microphysics substep is below clock resolution");
            step_microphysics_once(state, step);
            merge_water_repair_report(water, runtime.last_water_repair_report);
            merge_thermodynamic_repair_report(heat, runtime.last_thermodynamic_repair_report);
            complete_energy = complete_energy && runtime.last_thermodynamic_repair_report.moist_energy_complete;
            if (substeps == 1) conservation.baseline_q_kg = runtime.last_conservation_report.baseline_q_kg;
            conservation.active = conservation.active || runtime.last_conservation_report.active;
            conservation.tendency_imbalance_kg += runtime.last_conservation_report.tendency_imbalance_kg;
            conservation.surface_precipitation_kg += runtime.last_conservation_report.surface_precipitation_kg;
            if (precipitation[0])
                for (std::size_t index = 0; index < workspace->rate_integral.size(); ++index)
                    workspace->rate_integral[index] += step * (*precipitation[0])[index];
            elapsed += step;
            if (elapsed < interval)
            {
                const auto limit_t0 = microphysics_perf_start();
                limit = sedimentation_step_limit(state, *runtime.scheme);
                microphysics_perf_finish(microphysics_perf_totals.sedimentation_limit_s, limit_t0);
            }
        }
        state.time_s = start_time;
        conservation.relative_imbalance = std::abs(conservation.tendency_imbalance_kg) /
            std::max(1.0, conservation.baseline_q_kg);
        close_microphysics_species_total(heat);
        if (complete_energy) finalize_moist_basis_v2_commit(heat, "microphysics subcycled interval");
        runtime.last_water_repair_report = water;
        runtime.last_thermodynamic_repair_report = heat;
        runtime.last_conservation_report = conservation;
        if (precipitation[0])
            for (std::size_t index = 0; index < workspace->rate_integral.size(); ++index)
                (*precipitation[0])[index] = static_cast<float>(workspace->rate_integral[index] / interval);
        commit_radiation_applied_heat(state, interval, heat.radiation_heat_j);
        microphysics_perf_totals.sedimentation_substeps += substeps;
        workspace->reaction_complete = true;
    }
    catch (...)
    {
        state.time_s = start_time;
        if (protected_wind)
        {
            std::swap(*protected_wind, workspace->protected_primary_wind);
        }
        for (std::size_t index = 0; index < primary.size(); ++index)
            std::swap(*primary[index], workspace->primary[index]);
        for (std::size_t index = 0; index < coupling.size(); ++index)
            if (coupling[index]) std::swap(*coupling[index], workspace->coupling[index]);
        workspace->microphysics.restore_to(runtime);
        activity = workspace->advection;
        radiation_ledger = radiation_before;
        if (capture_chaos) std::swap(chaos_runtime, *workspace->chaos);
        for (std::size_t index = 0; index < precipitation.size(); ++index)
            if (precipitation[index]) std::swap(*precipitation[index], workspace->precipitation[index]);
        if (state.surface.accumulated_rainfall_last_update_s)
            *state.surface.accumulated_rainfall_last_update_s = workspace->precipitation_time;
        if (state.compute_runtime && state.compute_runtime->backend)
            state.compute_runtime->backend->invalidate_all_resident_fields();
        throw;
    }
}

Field3D completed_cloud_evaporation_return(const SimulationStateView& state, double interval)
{
    const auto* runtime = state.microphysics_runtime;
    if (!runtime || !runtime->scheme || !runtime->subcycle_workspace)
    {
        return {};
    }
    const auto& snapshot = *runtime->subcycle_workspace;
    if (!snapshot.reaction_complete || snapshot.reaction_owner != runtime->scheme.get() ||
        snapshot.reaction_end_s != state.time_s ||
        snapshot.reaction_interval_s != interval || !snapshot.microphysics.scheme)
    {
        return {};
    }
    if (state.domain_frame && (state.domain_frame->velocity_ms.u_ms != 0 || state.domain_frame->velocity_ms.v_ms != 0 ||
                               state.domain_frame->earth_origin_at_t0_m.x_m != snapshot.reaction_origin.x_m ||
                               state.domain_frame->earth_origin_at_t0_m.y_m != snapshot.reaction_origin.y_m))
    {
        return {};
    }
    const auto before = snapshot.microphysics.scheme->prognostic_scalars_for_transport();
    const auto after  = runtime->scheme->prognostic_scalars_for_transport();
    if (before.size() != 5 || after.size() != 5 || std::string(after[0].name) != "thompson.Nc_kg-1" ||
        std::string(after[3].name) != "thompson.Nwfa_kg-1")
    {
        return {};
    }
    const auto same_shape = [&](const Field3D& field)
    {
        return field.size_r() == state.fields.qc.size_r() && field.size_th() == state.fields.qc.size_th() &&
               field.size_z() == state.fields.qc.size_z();
    };
    for (std::size_t f = 0; f < before.size(); ++f)
    {
        if (std::string(before[f].name) != after[f].name || !same_shape(*before[f].field) ||
            !same_shape(*after[f].field))
        {
            return {};
        }
    }
    for (const auto& field : snapshot.primary)
    {
        if (!same_shape(field))
        {
            return {};
        }
    }
    Field3D                             returned(state.grid.nr, state.grid.nth, state.grid.nz, 0);
    const std::array<const Field3D*, 5> other_mass{
        &state.fields.qr, &state.fields.qi, &state.fields.qs, &state.fields.qg, &state.fields.qh};
    const auto ulp = [](float value)
    { return double(std::nextafter(value, std::numeric_limits<float>::infinity())) - value; };
    for (int i = 1; i < state.grid.nr - 1; ++i)
        for (int j = 1; j < state.grid.nth - 1; ++j)
            for (int k = 0; k < state.grid.nz; ++k)
            {
                const float number         = (*before[0].field)(i, j, k);
                const float aerosol_before = (*before[3].field)(i, j, k);
                const float aerosol_after  = (*after[3].field)(i, j, k);
                if (!std::isfinite(number) || number <= 0 || snapshot.primary[2](i, j, k) <= 0 ||
                    state.fields.qc(i, j, k) != 0 || (*after[0].field)(i, j, k) != 0 ||
                    !std::isfinite(aerosol_before) || !std::isfinite(aerosol_after) || aerosol_before < 0 ||
                    aerosol_after < number)
                {
                    continue;
                }
                bool isolated = true;
                for (std::size_t f : {1u, 2u, 4u})
                {
                    isolated = isolated && (*before[f].field)(i, j, k) == (*after[f].field)(i, j, k);
                }
                for (std::size_t f = 0; f < other_mass.size(); ++f)
                {
                    isolated = isolated && snapshot.primary[f + 3](i, j, k) == (*other_mass[f])(i, j, k);
                }
                const double residual = double(aerosol_after) - aerosol_before - number;
                const double rounding = ulp(aerosol_before) + ulp(aerosol_after);
                if (isolated && std::isfinite(rounding) && std::abs(residual) <= rounding)
                {
                    returned(i, j, k) = number;
                }
            }
    return returned;
}

void invalidate_microphysics_reaction_snapshot(MicrophysicsRuntimeState& runtime) noexcept
{
    if (runtime.subcycle_workspace)
    {
        runtime.subcycle_workspace->reaction_complete = false;
    }
}

void reset_microphysics_perf_stats() noexcept
{
    microphysics_perf_totals = MicrophysicsPerfSnapshot{};
}

MicrophysicsPerfSnapshot microphysics_perf_snapshot() noexcept
{
    return microphysics_perf_totals;
}

const char* surface_precipitation_accumulation_algorithm_id() noexcept
{
    return "exact_bottom_sedimentation_fallout_accumulation_v1";
}

const MicrophysicsWaterRepairReport& last_microphysics_water_repair_report()
{
    return global_microphysics_runtime_state.last_water_repair_report;
}

const MicrophysicsWaterRepairReport& last_microphysics_water_repair_report(const SimulationStateView& state)
{
    return state.microphysics_runtime ? state.microphysics_runtime->last_water_repair_report
                                      : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime")
                                            .last_water_repair_report;
}

const MicrophysicsThermodynamicRepairReport&
last_microphysics_thermodynamic_repair_report()
{
    return global_microphysics_runtime_state.last_thermodynamic_repair_report;
}

const MicrophysicsThermodynamicRepairReport&
last_microphysics_thermodynamic_repair_report(
    const SimulationStateView& state)
{
    return state.microphysics_runtime
        ? state.microphysics_runtime->last_thermodynamic_repair_report
        : tmv::require_level_runtime(state.microphysics_runtime, "microphysics_runtime").last_thermodynamic_repair_report;
}

void step_microphysics(double dt_micro)
{
    step_microphysics(::global_simulation_state(), dt_micro);
}

} // namespace tmv
