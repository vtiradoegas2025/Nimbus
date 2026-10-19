#include "microphysics/grid_geometry.hpp"
#include "core/infra/nested_feedback_interval.hpp"
#include "core/field/field_inventory.hpp"
#include "core/infra/domain_recycling_state.hpp"

#include "core/infra/physical_constants.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/grid_coordinates.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "diagnostics/diagnostic_cell_volume.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "boundary_layer/column_evaluation.hpp"
#include "boundary_layer/surface_frame.hpp"
#include "dynamics/scheme_descriptor.hpp"
#include "radiation/column_evaluation.hpp"
#include "terrain/conservative_transport.hpp"
#include "diagnostics/deterministic_reduction.hpp"

#include <cmath>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tmv::domain
{
namespace
{

constexpr std::size_t owner_index(HomogeneousPrimaryOwner owner) noexcept
{
    return static_cast<std::size_t>(owner);
}

void fingerprint_bytes(
    std::uint64_t& hash, const void* data, std::size_t bytes) noexcept
{
    const auto* values = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < bytes; ++index)
    {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
}

template <typename Value>
void fingerprint_value(std::uint64_t& hash, const Value& value) noexcept
{
    static_assert(std::is_trivially_copyable<Value>::value,
                  "D5 fingerprints require trivially-copyable values");
    fingerprint_bytes(hash, &value, sizeof(value));
}

template <typename Value>
void fingerprint_vector(
    std::uint64_t& hash, const std::vector<Value>& values) noexcept
{
    fingerprint_value(hash, values.size());
    if (!values.empty())
        fingerprint_bytes(
            hash, values.data(), values.size() * sizeof(Value));
}

void fingerprint_field(std::uint64_t& hash, const Field3D& field) noexcept
{
    fingerprint_value(hash, field.size_r());
    fingerprint_value(hash, field.size_th());
    fingerprint_value(hash, field.size_z());
    if (!field.empty())
        fingerprint_bytes(
            hash, field.data(), field.size() * sizeof(float));
}

template <typename Fluxes>
void fingerprint_scalar_interface_fluxes(
    std::uint64_t& hash, const Fluxes& fluxes) noexcept
{
    fingerprint_vector(hash, fluxes.west);
    fingerprint_vector(hash, fluxes.east);
    fingerprint_vector(hash, fluxes.south);
    fingerprint_vector(hash, fluxes.north);
    fingerprint_value(hash, fluxes.accumulated_dt_s);
    fingerprint_value(hash, fluxes.contribution_count);
}

std::uint64_t nested_preserved_state_fingerprint(
    const NestedExecutionStorage& nested) noexcept
{
    std::uint64_t hash = UINT64_C(1469598103934665603);
    const auto& child = nested.child;
    fingerprint_value(hash, child.time_s);
    fingerprint_value(hash, child.coriolis_f);
    fingerprint_value(hash, child.domain_frame);
    fingerprint_value(hash, child.grid.nr);
    fingerprint_value(hash, child.grid.nth);
    fingerprint_value(hash, child.grid.nz);
    fingerprint_value(hash, child.grid.dx);
    fingerprint_value(hash, child.grid.dy);
    fingerprint_value(hash, child.grid.dz);
    fingerprint_value(hash, child.grid.dt);
    const auto& terrain = child.runtime.terrain;
    const auto& config  = terrain.config;
    for (const auto* text : {&config.scheme_id, &config.coord_id, &config.horizontal_placement_id})
    {
        fingerprint_value(hash, text->size());
        fingerprint_bytes(hash, text->data(), text->size());
    }
    for (double value : {config.ztop,
                         config.feature_center_earth_x_m,
                         config.feature_center_earth_y_m,
                         config.bell.h0,
                         config.bell.a,
                         config.schar.H,
                         config.schar.a,
                         config.schar.ell,
                         config.smoothing_decay})
    {
        fingerprint_value(hash, value);
    }
    fingerprint_value(hash, config.compute_derivatives);
    fingerprint_value(hash, config.compute_metrics);
    fingerprint_value(hash, config.bell.axisymmetric);
    const auto& placement = terrain.earth_fixed_placement;
    fingerprint_value(hash, placement.active);
    fingerprint_value(hash, placement.nr);
    fingerprint_value(hash, placement.nth);
    fingerprint_value(hash, placement.nz);
    fingerprint_value(hash, placement.dx_m);
    fingerprint_value(hash, placement.dy_m);
    fingerprint_value(hash, placement.grid_origin_earth_m);
    fingerprint_value(hash, placement.feature_center_earth_m);
    fingerprint_value(hash, placement.sampling);
    for (const auto* rows : {&terrain.topography.h, &terrain.topography.hx, &terrain.topography.hy})
    {
        fingerprint_value(hash, rows->size());
        for (const auto& row : *rows)
        {
            fingerprint_vector(hash, row);
        }
    }
    const auto& metrics = terrain.metrics;
    for (const auto* field : {&metrics.z,
                              &metrics.J,
                              &metrics.mx,
                              &metrics.my,
                              &metrics.zeta,
                              &metrics.z_face,
                              &metrics.Jx_face,
                              &metrics.Jy_face,
                              &metrics.Gx_face,
                              &metrics.Gy_face})
    {
        fingerprint_value(hash, field->nr);
        fingerprint_value(hash, field->nth);
        fingerprint_value(hash, field->nz);
        fingerprint_vector(hash, field->values);
    }
    fingerprint_value(hash, metrics.dx);
    fingerprint_value(hash, metrics.dy);
    fingerprint_value(hash, metrics.dzeta);
#define NIMBUS_D5_FINGERPRINT_CHILD_PRIMARY(name) fingerprint_field(hash, child.fields.name);
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_FINGERPRINT_CHILD_PRIMARY)
#undef NIMBUS_D5_FINGERPRINT_CHILD_PRIMARY
    for (const auto* profile :
         {&child.base.rho0, &child.base.p0, &child.base.qv0, &child.base.theta0, &child.base.u0, &child.base.v0})
        fingerprint_vector(hash, *profile);
    for (const auto* field : {&child.tendencies.dtheta_dt_rad,
                              &child.tendencies.dtheta_dt_pbl,
                              &child.tendencies.dqv_dt_pbl,
                              &child.tendencies.du_dt_pbl,
                              &child.tendencies.dv_dt_pbl,
                              &child.tendencies.dtke_dt_pbl,
                              &child.tendencies.dqc_dt_pbl,
                              &child.tendencies.dqi_dt_pbl})
        fingerprint_field(hash, *field);
    for (const auto* surface : {
             &child.surface.ustar, &child.surface.tau_u,
             &child.surface.tau_v, &child.surface.sensible_heat_flux,
             &child.surface.latent_heat_flux,
             &child.surface.moisture_flux, &child.surface.pbl_height,
             &child.surface.precipitation_rate_mm_h,
             &child.surface.accumulated_rainfall_mm,
             &child.surface.accumulated_rainfall_residual_mm})
        fingerprint_vector(hash, *surface);
    fingerprint_value(
        hash, child.surface.accumulated_rainfall_last_update_s);
    fingerprint_bytes(hash, &nested.interface_source,
                      sizeof(nested.interface_source));
    fingerprint_bytes(hash, &nested.two_way_interval_source_begin,
                      sizeof(nested.two_way_interval_source_begin));
    fingerprint_value(
        hash, nested.two_way_child_interval_begin_inventory);
    fingerprint_value(hash, nested.two_way_dry_mass_storage_begin);
    fingerprint_value(hash, nested.two_way_theta_storage_begin);
    const auto& feedback = nested.last_two_way_feedback;
    fingerprint_value(hash, feedback.complete);
    fingerprint_value(hash, feedback.moist);
    fingerprint_value(hash, feedback.phase_projection_cells);
    fingerprint_value(hash, feedback.phase_theta_inventory_change);
    fingerprint_value(hash, feedback.phase_thermal_roundoff_j);
    fingerprint_value(hash, feedback.phase_thermal_scale_j);
    for (auto value : {feedback.restricted_scalar_cells,
                       feedback.restricted_x_faces,
                       feedback.restricted_y_faces,
                       feedback.restricted_z_faces,
                       feedback.refluxed_scalar_cells})
        fingerprint_value(hash, value);
    for (const auto* values : {&feedback.interface_mismatch,
                               &feedback.applied_reflux,
                               &feedback.storage_residual,
                               &feedback.child_interface_source,
                               &feedback.child_dynamics_source,
                               &feedback.child_dynamics_storage,
                               &feedback.child_interface_transport,
                               &feedback.child_interval_change,
                               &feedback.child_composite_residual,
                               &feedback.child_composite_tolerance})
        fingerprint_value(hash, *values);
    const auto fingerprint_carrier = [&](const NestedWaterFeedbackReport& item)
    {
        const auto& p = item.parent;
        for (double value : {p.initial_inventory,
                             p.final_inventory,
                             p.child_inventory,
                             p.restriction_change,
                             p.interface_mismatch,
                             p.applied_reflux,
                             p.storage_rounding,
                             p.closure_residual,
                             p.redistributed_inventory,
                             p.global_redistributed_inventory,
                             p.child_synchronized_inventory,
                             p.fine_requested_transfer,
                             p.fine_storage_rounding,
                             p.phase_exchange_inventory,
                             item.child_begin,
                             item.child_transport,
                             item.interface_source,
                             item.physics_source,
                             item.transport_storage,
                             item.child_change,
                             item.closure_residual,
                             item.tolerance})
            fingerprint_value(hash, value);
        fingerprint_value(hash, p.constrained_cells);
        fingerprint_value(hash, p.redistributed_cells);
        fingerprint_value(hash, p.global_redistributed_cells);
        fingerprint_value(hash, p.restricted_cells);
        fingerprint_value(hash, p.refluxed_cells);
        fingerprint_value(hash, p.complete);
    };
    for (const auto& item : feedback.water)
        fingerprint_carrier(item);
    fingerprint_value(hash, feedback.private_scalars.size());
    for (const auto& item : feedback.private_scalars)
    {
        fingerprint_value(hash, item.name.size());
        fingerprint_bytes(hash, item.name.data(), item.name.size());
        fingerprint_carrier(item.carrier);
    }
    const auto& relocation = feedback.relocation;
    fingerprint_value(hash, relocation.complete);
    fingerprint_value(hash, relocation.movement_sequence);
    fingerprint_value(hash, relocation.feedback_application_count);
    for (const auto* values : {&relocation.before,
                               &relocation.retired,
                               &relocation.admitted,
                               &relocation.boundary_source,
                               &relocation.after,
                               &relocation.arithmetic_scale})
        fingerprint_value(hash, *values);
    fingerprint_value(hash, relocation.private_scalars.size());
    for (const auto& scalar : relocation.private_scalars)
    {
        fingerprint_value(hash, scalar.name.size());
        fingerprint_bytes(hash, scalar.name.data(), scalar.name.size());
        for (double value : {scalar.before,
                             scalar.retired,
                             scalar.admitted,
                             scalar.boundary_source,
                             scalar.after,
                             scalar.arithmetic_scale})
            fingerprint_value(hash, value);
    }
    fingerprint_value(hash, nested.two_way_feedback_application_count);
    const auto& advection = child.runtime.advection;
    fingerprint_value(hash, advection.moist_interface.enabled);
    fingerprint_value(hash, advection.moist_interface.committed_physics);
    fingerprint_value(hash, advection.private_interface.fields.size());
    for (const auto& field : advection.private_interface.fields)
    {
        fingerprint_value(hash, field.name.size());
        fingerprint_bytes(hash, field.name.data(), field.name.size());
        fingerprint_value(hash, field.committed_physics);
        fingerprint_value(hash, field.boundary.outward_boundary_inventory);
        fingerprint_value(hash, field.boundary.flux_implied_state_change);
        fingerprint_value(hash, field.boundary.committed_state_change);
        fingerprint_value(hash, field.boundary.closure_residual);
        fingerprint_value(hash, field.boundary.complete);
        fingerprint_scalar_interface_fluxes(hash, field.flux.scalar());
    }
    for (const auto& field : advection.moist_interface.water)
        fingerprint_scalar_interface_fluxes(hash, field.scalar());
    fingerprint_scalar_interface_fluxes(hash, advection.theta_interface_flux_register.theta());
    fingerprint_scalar_interface_fluxes(
        hash, advection.dry_mass_interface_flux_register.fluxes());
    fingerprint_value(hash, advection.theta_interface_flux_capture_enabled);
    fingerprint_value(hash, advection.dry_mass_interface_flux_capture_enabled);
    fingerprint_value(
        hash, child.runtime.dynamics.momentum_interface_flux_capture_enabled);
    fingerprint_field(
        hash, child.runtime.microphysics.theta_storage_residual);
    if (child.runtime.microphysics.scheme)
        for (const auto& scalar : child.runtime.microphysics.scheme
                 ->prognostic_scalars_for_transport())
            if (scalar.field)
                fingerprint_field(hash, *scalar.field);
    return hash;
}

std::array<const void*, 11> nested_scheme_identity(
    const NestedExecutionStorage& nested) noexcept
{
    const auto& runtime = nested.child.runtime;
    return {{
        runtime.radiation.scheme.get(), runtime.turbulence.scheme.get(),
        runtime.boundary_layer.scheme.get(), runtime.chaos.scheme.get(),
        runtime.microphysics.scheme.get(), runtime.terrain.scheme.get(),
        runtime.radar.selected_scheme.get(), runtime.advection.scheme.get(),
        runtime.diffusion.scheme.get(), runtime.time_stepping.scheme.get(),
        runtime.dynamics.scheme.get(),
    }};
}

HorizontalFieldLocation primary_horizontal_location(
    HomogeneousPrimaryOwner owner, bool staggered) noexcept
{
    if (!staggered)
        return HorizontalFieldLocation::ScalarCenter;
    if (owner == HomogeneousPrimaryOwner::u)
        return HorizontalFieldLocation::XFace;
    if (owner == HomogeneousPrimaryOwner::v)
        return HorizontalFieldLocation::YFace;
    return HorizontalFieldLocation::ScalarCenter;
}

void require_profile(const std::vector<double>& profile,
                     int nz,
                     const char* name,
                     bool require_positive,
                     bool require_nonnegative)
{
    if (profile.size() != static_cast<std::size_t>(nz))
        throw std::invalid_argument(std::string("homogeneous leading state requires complete base.") + name);
    for (double value : profile)
    {
        if (!std::isfinite(value) || (require_positive && value <= 0.0) ||
            (require_nonnegative && value < 0.0))
            throw std::invalid_argument(std::string("homogeneous leading base.") + name +
                                        " contains an unphysical value");
        const float stored = static_cast<float>(value);
        if (!std::isfinite(stored) || (require_positive && stored <= 0.0f) ||
            (require_nonnegative && stored < 0.0f))
            throw std::invalid_argument(std::string("homogeneous leading base.") + name +
                                        " is not representable in model storage");
    }
}

void copy_profile(HomogeneousPrimaryProfiles& output,
                  HomogeneousPrimaryOwner owner,
                  const std::vector<double>& source)
{
    auto& destination = output[owner];
    destination.resize(source.size());
    for (std::size_t level = 0; level < source.size(); ++level)
        destination[level] = static_cast<float>(source[level]);
}

std::size_t checked_ledger_product(std::size_t a, std::size_t b)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("trailing physical ledger cell count overflow");
    return a * b;
}

void require_grid_field(
    const Field3D& field,
    const SimulationStateView& state,
    const char* name)
{
    if (field.size_r() != state.grid.nr
        || field.size_th() != state.grid.nth
        || field.size_z() != state.grid.nz)
        throw std::invalid_argument(
            std::string(name) + " shape does not match the recycling grid");
}

bool plan_matches_grid(
    const WholeCellShiftPlan& plan,
    const SimulationStateView& state)
{
    return plan.nr == state.grid.nr && plan.nth == state.grid.nth
        && plan.nz == state.grid.nz;
}

std::array<Field3D*, 6> turbulence_tendency_fields(
    TurbulenceTendencies& tendencies) noexcept
{
    return {{
        &tendencies.dudt_sgs,
        &tendencies.dvdt_sgs,
        &tendencies.dwdt_sgs,
        &tendencies.dthetadt_sgs,
        &tendencies.dqvdt_sgs,
        &tendencies.dtkedt_sgs,
    }};
}

void require_turbulence_tendency_group(
    TurbulenceTendencies& tendencies,
    const SimulationStateView& state,
    const char* group_name,
    bool require_finite)
{
    const auto fields = turbulence_tendency_fields(tendencies);
    const bool any = std::any_of(
        fields.begin(), fields.end(),
        [](const Field3D* field) { return !field->empty(); });
    const bool all = std::all_of(
        fields.begin(), fields.end(),
        [](const Field3D* field) { return !field->empty(); });
    if (any != all)
        throw std::invalid_argument(
            std::string("D5 turbulence ") + group_name
            + " tendency storage is only partially allocated");
    if (!all) return;
    for (const Field3D* field : fields)
    {
        require_grid_field(*field, state, group_name);
        if (require_finite)
            for (std::size_t index = 0; index < field->size(); ++index)
                if (!std::isfinite(static_cast<double>(field->data()[index])))
                    throw std::invalid_argument(
                        std::string("D5 turbulence ") + group_name
                        + " tendency contains a non-finite value");
    }
}

void invalidate_turbulence_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime = *static_cast<TurbulenceRuntimeState*>(opaque_runtime);
    for (Field3D* field : turbulence_tendency_fields(runtime.held_tendencies))
        field->fill(0.0f);
    for (Field3D* field : turbulence_tendency_fields(runtime.candidate_tendencies))
        field->fill(0.0f);
    runtime.held_tendencies.resident_source_unchanged = false;
    runtime.candidate_tendencies.resident_source_unchanged = false;
    runtime.realized_transport_tendency_current = false;
    runtime.stochastic_transport_applied_to_current = false;

    // Preserve scratch capacity while making accidental pre-rebuild use fail
    // closed. get_runtime_turbulence_grid overwrites every value and pointer
    // before the next scheme evaluation.
    runtime.grid_metrics_scratch.dx =
        std::numeric_limits<double>::quiet_NaN();
    runtime.grid_metrics_scratch.dy =
        std::numeric_limits<double>::quiet_NaN();
    std::fill(runtime.grid_metrics_scratch.dz.begin(),
              runtime.grid_metrics_scratch.dz.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(runtime.grid_metrics_scratch.z_int.begin(),
              runtime.grid_metrics_scratch.z_int.end(),
              std::numeric_limits<double>::quiet_NaN());
    runtime.grid_metrics_scratch.terrain_metrics_active = false;
    runtime.grid_metrics_scratch.terrain_metrics = nullptr;
    runtime.grid_metrics_scratch.terrain_topography = nullptr;

    runtime.last_explicit_stability_limit_s =
        std::numeric_limits<double>::infinity();
    runtime.last_explicit_substep_count = 0;
}

void invalidate_chaos_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime =
        *static_cast<tmv::chaos::ChaosRuntimeState*>(opaque_runtime);
    for (Field3D* field : {
             &runtime.base_du_dt_pbl,
             &runtime.base_dv_dt_pbl,
             &runtime.base_dtheta_dt_pbl,
             &runtime.base_dqv_dt_pbl,
             &runtime.base_dtke_dt_pbl})
        field->fill(0.0f);
    runtime.base_sfc_ustar.clear();
    runtime.base_sfc_tau_u.clear();
    runtime.base_sfc_tau_v.clear();
    runtime.base_sfc_sensible_heat_flux.clear();
    runtime.base_sfc_latent_heat_flux.clear();
    runtime.base_sfc_moisture_flux.clear();
    runtime.pbl_base_tendencies_valid = false;

    runtime.grid_metrics_scratch.dx =
        std::numeric_limits<double>::quiet_NaN();
    runtime.grid_metrics_scratch.dy =
        std::numeric_limits<double>::quiet_NaN();
    std::fill(runtime.grid_metrics_scratch.dz.begin(),
              runtime.grid_metrics_scratch.dz.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(runtime.grid_metrics_scratch.z_int.begin(),
              runtime.grid_metrics_scratch.z_int.end(),
              std::numeric_limits<double>::quiet_NaN());
    runtime.grid_metrics_scratch.terrain_metrics_active = false;
    runtime.grid_metrics_scratch.terrain_metrics = nullptr;
    runtime.grid_metrics_scratch.terrain_topography = nullptr;
}

void commit_prepared_evolving_chaos_domain_recycling(
    void* opaque_prepared) noexcept
{
    auto& prepared = *static_cast<
        PreparedEvolvingChaosDomainRecyclingState*>(opaque_prepared);
    prepared.runtime->scheme.swap(prepared.scheme);
    if (!prepared.preserve_shifted_pbl_base)
    {
        invalidate_chaos_runtime_after_domain_recycling(prepared.runtime);
        return;
    }
    auto& runtime = *prepared.runtime;
    runtime.grid_metrics_scratch.dx =
        std::numeric_limits<double>::quiet_NaN();
    runtime.grid_metrics_scratch.dy =
        std::numeric_limits<double>::quiet_NaN();
    std::fill(runtime.grid_metrics_scratch.dz.begin(),
              runtime.grid_metrics_scratch.dz.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(runtime.grid_metrics_scratch.z_int.begin(),
              runtime.grid_metrics_scratch.z_int.end(),
              std::numeric_limits<double>::quiet_NaN());
    runtime.grid_metrics_scratch.terrain_metrics_active = false;
    runtime.grid_metrics_scratch.terrain_metrics = nullptr;
    runtime.grid_metrics_scratch.terrain_topography = nullptr;
}

void invalidate_advection_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime = *static_cast<AdvectionRuntimeState*>(opaque_runtime);

    // These fields are reusable output/scratch, not a held physical source.
    // Keep their allocations hot while making accidental pre-rebuild reads
    // deterministic and harmless.
    runtime.scalar_scratch_a.fill(0.0f);
    runtime.scalar_scratch_b.fill(0.0f);
    runtime.vertical_tendencies.dqdt_adv.fill(0.0f);
    runtime.vertical_tendencies.fused_dst = nullptr;
    runtime.vertical_tendencies.fused_src = nullptr;
    runtime.vertical_tendencies.fused_dt = 0.0f;
    runtime.vertical_tendencies.fused_completed = false;

    runtime.tracer_advected_in_batch = false;
    runtime.hydrometeor_active.fill(false);
    runtime.activity_current = false;

    runtime.thermodynamic_boundary_reports = {};
    runtime.thermodynamic_boundary_reports_complete = false;
    runtime.thermodynamic_moist_energy_reports = {};
    runtime.thermodynamic_moist_energy_reports_complete = false;
    runtime.thermodynamic_boundary_reports_requested = false;
    runtime.thermodynamic_moist_energy_reports_requested = false;
    runtime.theta_interface_flux_register = {};
    runtime.theta_interface_flux_capture_enabled = false;
    runtime.dry_mass_interface_flux_register = {};
    runtime.dry_mass_interface_flux_capture_enabled = false;
    runtime.moist_interface = {};
    runtime.private_interface                            = {};
}

template <typename Reports>
bool all_reports_complete(const Reports& reports) noexcept
{
    return std::all_of(
        reports.begin(), reports.end(),
        [](const auto& report) { return report.complete; });
}

template <typename Reports>
bool no_reports_complete(const Reports& reports) noexcept
{
    return std::none_of(
        reports.begin(), reports.end(),
        [](const auto& report) { return report.complete; });
}

bool finite_boundary_report(
    const ScalarBoundaryTransportReport& report) noexcept
{
    return std::isfinite(report.outward_boundary_inventory)
        && std::isfinite(report.flux_implied_state_change)
        && std::isfinite(report.committed_state_change)
        && std::isfinite(report.closure_residual);
}

bool finite_moist_energy_report(
    const ScalarMoistEnergyTransportReportV1& report) noexcept
{
    const double values[] = {
        report.dry_internal_change_j,
        report.water_sensible_change_j,
        report.phase_reference_change_j,
        report.carrier_kinetic_change_j,
        report.gravitational_change_j,
        report.subgrid_tke_change_j,
        report.density_history_remap_j,
        report.outward_boundary_energy_j,
        report.interior_conversion_j,
        report.limited_face_state_change_j,
        report.requested_state_change_j,
        report.storage_rounding_j,
        report.committed_state_change_j,
        report.closure_residual_j,
    };
    return std::all_of(
        std::begin(values), std::end(values),
        [](double value) { return std::isfinite(value); });
}

std::uint32_t tracer_roundoff_float_bits(float value) noexcept
{
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool valid_tracer_roundoff_report(
    const PositiveDefiniteScalarRoundoffReport& report) noexcept
{
    if (!std::isfinite(report.physical_inventory_correction_kg)
        || report.physical_inventory_correction_kg < 0.0
        || report.thermodynamic_energy_correction_j != 0.0
        || report.negative_zero_count
            > std::numeric_limits<std::size_t>::max()
                - report.negative_subnormal_count)
    {
        return false;
    }
    const std::size_t canonicalization_count =
        report.negative_subnormal_count + report.negative_zero_count;
    if (report.closure_canonicalization_count > canonicalization_count
        || (report.negative_subnormal_count == 0
            && report.physical_inventory_correction_kg != 0.0))
        return false;

    const std::uint32_t minimum_bits = tracer_roundoff_float_bits(
        report.most_negative_repaired_value);
    const bool minimum_is_negative_subnormal =
        (minimum_bits & 0x80000000U) != 0U
        && (minimum_bits & 0x7f800000U) == 0U
        && (minimum_bits & 0x007fffffU) != 0U;
    return report.negative_subnormal_count == 0
        ? minimum_bits == 0U
        : minimum_is_negative_subnormal;
}

bool zero_tracer_roundoff_report(
    const PositiveDefiniteScalarRoundoffReport& report) noexcept
{
    return report.physical_inventory_correction_kg == 0.0
        && report.thermodynamic_energy_correction_j == 0.0
        && tracer_roundoff_float_bits(
               report.most_negative_repaired_value) == 0U
        && report.negative_subnormal_count == 0
        && report.negative_zero_count == 0
        && report.closure_canonicalization_count == 0
        && !report.complete;
}

bool valid_tracer_roundoff_runtime(
    const AdvectionRuntimeState& runtime) noexcept
{
    const auto& last = runtime.tracer_roundoff_report;
    const auto& cumulative = runtime.cumulative_tracer_roundoff_report;
    if (!valid_tracer_roundoff_report(last)
        || !valid_tracer_roundoff_report(cumulative))
    {
        return false;
    }
    if (runtime.tracer_roundoff_report_step_count == 0)
    {
        return zero_tracer_roundoff_report(last)
            && zero_tracer_roundoff_report(cumulative);
    }
    if (!last.complete
        || cumulative.complete
            != runtime.tracer_roundoff_cumulative_history_known
        || cumulative.negative_subnormal_count
            < last.negative_subnormal_count
        || cumulative.negative_zero_count < last.negative_zero_count
        || cumulative.closure_canonicalization_count
            < last.closure_canonicalization_count
        || cumulative.physical_inventory_correction_kg
            < last.physical_inventory_correction_kg)
    {
        return false;
    }
    const std::uint32_t last_minimum = tracer_roundoff_float_bits(
        last.most_negative_repaired_value) & 0x007fffffU;
    const std::uint32_t cumulative_minimum = tracer_roundoff_float_bits(
        cumulative.most_negative_repaired_value) & 0x007fffffU;
    return cumulative_minimum >= last_minimum;
}

std::array<Field3D*, 5> diffusion_tendency_fields(
    DiffusionTendencies& tendencies) noexcept
{
    return {{
        &tendencies.dudt_diff,
        &tendencies.dvdt_diff,
        &tendencies.dwdt_diff,
        &tendencies.dthetadt_diff,
        &tendencies.dqvdt_diff,
    }};
}

std::array<Field3D*, 5> diffusion_candidate_fields(
    DiffusionRuntimeState& runtime) noexcept
{
    return {{
        &runtime.scratch_u,
        &runtime.scratch_v,
        &runtime.scratch_w,
        &runtime.scratch_theta,
        &runtime.scratch_qv,
    }};
}

void invalidate_diffusion_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime = *static_cast<DiffusionRuntimeState*>(opaque_runtime);
    for (Field3D* field : diffusion_tendency_fields(runtime.tendencies))
        field->fill(0.0f);
    for (Field3D* field : diffusion_candidate_fields(runtime))
        field->fill(0.0f);
    runtime.diagnostics.max_diffusion_number = 0.0;
    runtime.diagnostics.K_effective.fill(0.0f);
    runtime.last_momentum_work = {};
}

bool finite_diffusion_work_report(
    const DiffusionMomentumWorkReport& report) noexcept
{
    const double values[] = {
        report.requested_linear_work_j,
        report.finite_step_correction_j,
        report.storage_rounding_j,
        report.committed_work_j,
        report.moist_dry_internal_change_j,
        report.moist_water_sensible_change_j,
        report.moist_phase_reference_change_j,
        report.moist_carrier_kinetic_change_j,
        report.moist_gravitational_change_j,
        report.moist_subgrid_tke_change_j,
        report.moist_committed_energy_change_j,
        report.moist_energy_closure_residual_j,
    };
    if (!std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return std::isfinite(value); })
        || report.finite_step_correction_j < 0.0)
        return false;

    const double moist_values[] = {
        report.moist_dry_internal_change_j,
        report.moist_water_sensible_change_j,
        report.moist_phase_reference_change_j,
        report.moist_carrier_kinetic_change_j,
        report.moist_gravitational_change_j,
        report.moist_subgrid_tke_change_j,
        report.moist_committed_energy_change_j,
        report.moist_energy_closure_residual_j,
    };
    if (!report.moist_energy_complete)
        return std::all_of(
            std::begin(moist_values), std::end(moist_values),
            [](double value) { return value == 0.0; });

    const double reservoir_sum =
        report.committed_work_j
        + report.moist_dry_internal_change_j
        + report.moist_water_sensible_change_j
        + report.moist_phase_reference_change_j
        + report.moist_carrier_kinetic_change_j
        + report.moist_gravitational_change_j
        + report.moist_subgrid_tke_change_j;
    return report.moist_energy_closure_residual_j
        == report.moist_committed_energy_change_j - reservoir_sum;
}

void require_diffusion_field_group(
    const std::array<Field3D*, 5>& fields,
    const SimulationStateView& state,
    const char* group_name,
    bool require_all_or_none)
{
    const bool any = std::any_of(
        fields.begin(), fields.end(),
        [](const Field3D* field) { return !field->empty(); });
    const bool all = std::all_of(
        fields.begin(), fields.end(),
        [](const Field3D* field) { return !field->empty(); });
    if (require_all_or_none && any != all)
        throw std::invalid_argument(
            std::string("D5 diffusion ") + group_name
            + " storage is only partially allocated");
    for (const Field3D* field : fields)
    {
        if (field->empty()) continue;
        require_grid_field(*field, state, group_name);
        for (std::size_t index = 0; index < field->size(); ++index)
            if (!std::isfinite(static_cast<double>(field->data()[index])))
                throw std::invalid_argument(
                    std::string("D5 diffusion ") + group_name
                    + " contains a non-finite value");
    }
}

std::array<Field3D*, 11> time_stepping_workspace_fields(
    TimeSteppingSlowRk2Workspace& workspace) noexcept
{
    return {{
        &workspace.u_save,
        &workspace.v_save,
        &workspace.w_save,
        &workspace.p_save,
        &workspace.du_dt_k1,
        &workspace.dv_dt_k1,
        &workspace.dw_dt_k1,
        &workspace.dp_dt_k1,
        &workspace.du_advection_dt_k1,
        &workspace.dv_advection_dt_k1,
        &workspace.dw_advection_dt_k1,
    }};
}

void invalidate_time_stepping_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime = *static_cast<TimeSteppingRuntimeState*>(opaque_runtime);
    for (Field3D* field : time_stepping_workspace_fields(runtime.slow_rk2))
        field->fill(0.0f);
    runtime.config.max_acoustic_speed_ms = 0.0;
    runtime.config.min_acoustic_spacing_m = 0.0;
    runtime.last_boundary_mass = {};
    runtime.last_acoustic_energy = {};
    runtime.last_slow_momentum_energy = {};
    runtime.last_eos_pressure_energy = {};
    runtime.last_sponge_energy = {};
    runtime.last_final_boundary_energy = {};
    runtime.last_completed_step_energy = {};
    if (runtime.scheme != nullptr)
        runtime.scheme
            ->invalidate_generation_state_after_domain_recycling();
}

void invalidate_microphysics_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime =
        *static_cast<MicrophysicsRuntimeState*>(opaque_runtime);
    Field3D* const tendencies[] = {
        &runtime.tendencies.dtheta_dt,
        &runtime.tendencies.dqv_dt,
        &runtime.tendencies.dqc_dt,
        &runtime.tendencies.dqr_dt,
        &runtime.tendencies.dqi_dt,
        &runtime.tendencies.dqs_dt,
        &runtime.tendencies.dqg_dt,
        &runtime.tendencies.dqh_dt,
    };
    for (Field3D* field : tendencies)
        if (!field->empty()) field->fill(0.0f);
    if (runtime.scheme != nullptr)
        runtime.scheme->invalidate_transient_state_after_domain_recycling();
}

bool valid_boundary_mass_report(
    const DynamicsBoundaryMassReport& report) noexcept
{
    return std::isfinite(report.outward_boundary_mass_kg)
        && std::isfinite(report.flux_implied_state_change_kg)
        && std::isfinite(report.committed_state_change_kg)
        && std::isfinite(report.closure_residual_kg)
        && report.flux_implied_state_change_kg
            == -report.outward_boundary_mass_kg
        && report.closure_residual_kg
            == report.committed_state_change_kg
                - report.flux_implied_state_change_kg;
}

bool valid_acoustic_energy_report(
    const CartesianCGridAcousticEnergyReportV1& report) noexcept
{
    const double values[] = {
        report.continuity_eos_dry_internal_change_j,
        report.continuity_eos_water_sensible_change_j,
        report.continuity_eos_phase_reference_change_j,
        report.continuity_carrier_kinetic_change_j,
        report.continuity_gravitational_change_j,
        report.continuity_subgrid_tke_change_j,
        report.continuity_carrier_requested_change_j,
        report.continuity_carrier_outward_boundary_transport_j,
        report.continuity_carrier_interior_conversion_j,
        report.continuity_carrier_storage_rounding_j,
        report.slow_momentum_advection_requested_linear_work_j,
        report.kinetic_transport_interior_conversion_j,
        report.kinetic_transport_closure_residual_j,
        report.pressure_gradient_requested_linear_work_j,
        report.pressure_gradient_finite_step_correction_j,
        report.pressure_gradient_storage_rounding_j,
        report.pressure_gradient_committed_work_j,
        report.boundary_committed_work_j,
        report.boundary_radiative_requested_linear_work_j,
        report.boundary_radiative_finite_step_correction_j,
        report.boundary_radiative_storage_rounding_j,
        report.boundary_radiative_committed_work_j,
        report.boundary_constraint_history_committed_work_j,
        report.committed_inventory_change_j,
    };
    const bool finite = std::all_of(
        std::begin(values), std::end(values),
        [](double value) { return std::isfinite(value); });
    if (!finite
        || report.pressure_gradient_finite_step_correction_j < 0.0
        || report.boundary_radiative_finite_step_correction_j < 0.0
        || report.complete != (report.substep_count > 0)
        || report.boundary_complete
            != (report.boundary_application_count > 0)
        || report.kinetic_transport_complete != report.complete)
        return false;
    if (report.complete) return true;
    return report.substep_count == 0
        && report.boundary_application_count == 0
        && std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return value == 0.0; });
}

bool valid_slow_momentum_energy_report(
    const CartesianSlowMomentumEnergyReportV1& report) noexcept
{
    const double values[] = {
        report.advection_requested_linear_work_j,
        report.nonadvection_requested_linear_work_j,
        report.advection_finite_step_correction_j,
        report.nonadvection_finite_step_correction_j,
        report.coupled_finite_step_cross_work_j,
        report.storage_rounding_j,
        report.committed_kinetic_change_j,
        report.closure_residual_j,
        report.pressure_dry_internal_change_j,
        report.pressure_water_sensible_change_j,
        report.committed_pressure_energy_change_j,
    };
    if (!std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return std::isfinite(value); })
        || report.advection_finite_step_correction_j < 0.0
        || report.nonadvection_finite_step_correction_j < 0.0
        || report.committed_pressure_energy_change_j
            != report.pressure_dry_internal_change_j
                + report.pressure_water_sensible_change_j)
        return false;
    if (!report.complete)
        return std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return value == 0.0; });
    const double requested =
        report.advection_requested_linear_work_j
        + report.nonadvection_requested_linear_work_j
        + report.advection_finite_step_correction_j
        + report.nonadvection_finite_step_correction_j
        + report.coupled_finite_step_cross_work_j;
    return report.closure_residual_j
        == report.committed_kinetic_change_j
            - (requested + report.storage_rounding_j);
}

bool valid_pressure_energy_report(
    const dynamics_eos::PressureProjectionEnergyReportV1& report) noexcept
{
    const double values[] = {
        report.dry_internal_change_j,
        report.water_sensible_change_j,
        report.phase_reference_change_j,
        report.carrier_kinetic_change_j,
        report.gravitational_change_j,
        report.subgrid_tke_change_j,
        report.committed_energy_change_j,
        report.closure_residual_j,
    };
    if (!std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return std::isfinite(value); }))
        return false;
    if (!report.complete)
        return std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return value == 0.0; });
    const double reservoir_sum =
        report.dry_internal_change_j
        + report.water_sensible_change_j
        + report.phase_reference_change_j
        + report.carrier_kinetic_change_j
        + report.gravitational_change_j
        + report.subgrid_tke_change_j;
    return report.closure_residual_j
        == report.committed_energy_change_j - reservoir_sum;
}

bool valid_sponge_pass_energy_report(
    const SpongePassEnergyReportV1& pass) noexcept
{
    const double values[] = {
        pass.momentum_requested_linear_work_j,
        pass.momentum_finite_step_correction_j,
        pass.momentum_storage_rounding_j,
        pass.dry_internal_change_j,
        pass.water_sensible_change_j,
        pass.phase_reference_change_j,
        pass.carrier_kinetic_change_j,
        pass.gravitational_change_j,
        pass.subgrid_tke_change_j,
        pass.committed_energy_change_j,
        pass.closure_residual_j,
    };
    if (!std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return std::isfinite(value); })
        || pass.momentum_finite_step_correction_j < 0.0)
        return false;
    if (!pass.complete)
        return !pass.applied && std::all_of(
            std::begin(values), std::end(values),
            [](double value) { return value == 0.0; });
    const double reservoir_sum =
        pass.dry_internal_change_j
        + pass.water_sensible_change_j
        + pass.phase_reference_change_j
        + pass.carrier_kinetic_change_j
        + pass.gravitational_change_j
        + pass.subgrid_tke_change_j;
    return pass.momentum_storage_rounding_j
            == pass.carrier_kinetic_change_j
                - pass.momentum_requested_linear_work_j
                - pass.momentum_finite_step_correction_j
        && pass.closure_residual_j
            == pass.committed_energy_change_j - reservoir_sum;
}

bool valid_sponge_energy_report(
    const SpongeEnergyReportV1& report) noexcept
{
    return valid_sponge_pass_energy_report(report.upper)
        && valid_sponge_pass_energy_report(report.w_only)
        && valid_sponge_pass_energy_report(report.lateral)
        && report.complete
            == (report.upper.complete && report.w_only.complete
                && report.lateral.complete);
}

template <typename Report, typename Validator>
bool valid_cumulative_report(
    const Report& report,
    std::uint64_t count,
    Validator validator) noexcept
{
    return validator(report) && ((count > 0) == report.complete);
}

} // namespace

const std::array<const char*, homogeneous_primary_owner_count()>&
homogeneous_primary_owner_names() noexcept
{
    static const std::array<const char*, homogeneous_primary_owner_count()> names{{
#define NIMBUS_D5_OWNER_NAME(name) #name,
        NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_OWNER_NAME)
#undef NIMBUS_D5_OWNER_NAME
    }};
    return names;
}

HomogeneousPrimaryProfiles prepare_flat_homogeneous_primary_profiles(
    const SimulationStorageState& storage,
    const std::vector<double>& environmental_tke_profile)
{
    const int nz = storage.grid.nz;
    if (storage.grid.nr <= 0 || storage.grid.nth <= 0 || nz <= 0)
        throw std::invalid_argument("homogeneous leading state requires a positive grid");
    if (!std::isfinite(storage.grid.dx) || storage.grid.dx <= 0.0 ||
        !std::isfinite(storage.grid.dy) || storage.grid.dy <= 0.0 ||
        !std::isfinite(storage.grid.dz) || storage.grid.dz <= 0.0)
        throw std::invalid_argument("homogeneous leading state requires finite positive spacing");

    require_profile(storage.base.rho0, nz, "rho0", true, false);
    require_profile(storage.base.p0, nz, "p0", true, false);
    require_profile(storage.base.theta0, nz, "theta0", true, false);
    require_profile(storage.base.qv0, nz, "qv0", false, true);
    require_profile(storage.base.u0, nz, "u0", false, false);
    require_profile(storage.base.v0, nz, "v0", false, false);
    require_profile(environmental_tke_profile, nz, "environmental_tke", false, true);
    for (int k = 1; k < nz; ++k)
        if (storage.base.p0[static_cast<std::size_t>(k)] >=
            storage.base.p0[static_cast<std::size_t>(k - 1)])
            throw std::invalid_argument(
                "homogeneous leading base pressure must decrease with height");

    HomogeneousPrimaryProfiles output;
    for (auto& profile : output.values)
        profile.assign(static_cast<std::size_t>(nz), 0.0f);

    copy_profile(output, HomogeneousPrimaryOwner::p, storage.base.p0);
    copy_profile(output, HomogeneousPrimaryOwner::theta, storage.base.theta0);
    copy_profile(output, HomogeneousPrimaryOwner::qv, storage.base.qv0);
    copy_profile(output, HomogeneousPrimaryOwner::u, storage.base.u0);
    copy_profile(output, HomogeneousPrimaryOwner::v, storage.base.v0);
    copy_profile(output, HomogeneousPrimaryOwner::tke, environmental_tke_profile);

    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk /
                             physical_constants::specific_heat_cp_jkgk;
    auto& rho = output[HomogeneousPrimaryOwner::rho];
    const auto& pressure = output[HomogeneousPrimaryOwner::p];
    const auto& theta = output[HomogeneousPrimaryOwner::theta];
    for (int k = 0; k < nz; ++k)
    {
        const std::size_t level = static_cast<std::size_t>(k);
        const double stored_pressure = static_cast<double>(pressure[level]);
        const double stored_theta = static_cast<double>(theta[level]);
        const double temperature = stored_theta * std::pow(
            stored_pressure / physical_constants::reference_pressure_pa, kappa);
        const double diagnosed_density = stored_pressure /
            (physical_constants::gas_constant_dry_air_jkgk * temperature);
        const double reference_density = storage.base.rho0[level];
        const double relative_error = std::abs(reference_density - diagnosed_density) /
                                      std::max(diagnosed_density, 0.01);
        if (!std::isfinite(diagnosed_density) || diagnosed_density < 0.01 ||
            diagnosed_density > 5.0 || relative_error >= 5.0e-3)
            throw std::invalid_argument(
                "homogeneous leading thermodynamic base violates the dry EOS");
        rho[level] = static_cast<float>(diagnosed_density);
    }
    return output;
}

std::vector<HomogeneousFieldShiftBinding> bind_homogeneous_primary_fields(
    SimulationStorageState& storage,
    const HomogeneousPrimaryProfiles& profiles)
{
    std::vector<HomogeneousFieldShiftBinding> bindings;
    bindings.reserve(homogeneous_primary_owner_count());
#define NIMBUS_D5_BIND_OWNER(name)                                                   \
    bindings.push_back({#name, &storage.fields.name,                                \
                        &profiles.values[owner_index(HomogeneousPrimaryOwner::name)], \
                        primary_horizontal_location(                                \
                            HomogeneousPrimaryOwner::name,                          \
                            storage.grid.geometry.staggered)});
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_BIND_OWNER)
#undef NIMBUS_D5_BIND_OWNER
    return bindings;
}

TerrainAwarePrimaryFields prepare_terrain_aware_primary_fields(
    const SimulationStorageState& storage,
    const TerrainMetrics3D& post_shift_metrics,
    const std::vector<double>& environmental_tke_profile)
{
    const int nr = storage.grid.nr;
    const int nth = storage.grid.nth;
    const int nz = storage.grid.nz;
    (void)prepare_flat_homogeneous_primary_profiles(
        storage, environmental_tke_profile);
    if (post_shift_metrics.z.size_r() != nr
        || post_shift_metrics.z.size_th() != nth
        || post_shift_metrics.z.size_z() != nz)
        throw std::invalid_argument(
            "terrain-aware leading state requires grid-matching physical heights");

    TerrainAwarePrimaryFields output;
    for (auto& field : output.values)
        field.resize(nr, nth, nz, 0.0f);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
            {
                const double scalar_height = post_shift_metrics.z(i, j, k);
                const double u_height = i + 1 < nr
                    ? 0.5 * (scalar_height
                             + post_shift_metrics.z(i + 1, j, k))
                    : scalar_height;
                const double v_height = j + 1 < nth
                    ? 0.5 * (scalar_height
                             + post_shift_metrics.z(i, j + 1, k))
                    : scalar_height;
                output[HomogeneousPrimaryOwner::p](i, j, k) =
                    static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            storage.base.p0, scalar_height, storage.grid.dz));
                output[HomogeneousPrimaryOwner::theta](i, j, k) =
                    static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            storage.base.theta0, scalar_height,
                            storage.grid.dz));
                output[HomogeneousPrimaryOwner::qv](i, j, k) =
                    static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            storage.base.qv0, scalar_height,
                            storage.grid.dz));
                output[HomogeneousPrimaryOwner::rho](i, j, k) =
                    static_cast<float>(
                        terrain_transport::reference_density_fp32(
                            storage.base.p0, storage.base.theta0,
                            scalar_height, storage.grid.dz));
                output[HomogeneousPrimaryOwner::u](i, j, k) =
                    static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            storage.base.u0, u_height, storage.grid.dz));
                output[HomogeneousPrimaryOwner::v](i, j, k) =
                    static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            storage.base.v0, v_height, storage.grid.dz));
                output[HomogeneousPrimaryOwner::tke](i, j, k) =
                    static_cast<float>(
                        terrain_transport::reference_profile_value_fp32(
                            environmental_tke_profile, scalar_height,
                            storage.grid.dz));
            }
    return output;
}

std::vector<HomogeneousFieldShiftBinding>
bind_terrain_aware_primary_fields(
    SimulationStorageState& storage,
    const TerrainAwarePrimaryFields& fields)
{
    std::vector<HomogeneousFieldShiftBinding> bindings;
    bindings.reserve(homogeneous_primary_owner_count());
#define NIMBUS_D5_BIND_TERRAIN_OWNER(name)                                         \
    bindings.push_back({#name, &storage.fields.name, nullptr,                     \
                        primary_horizontal_location(                              \
                            HomogeneousPrimaryOwner::name,                        \
                            storage.grid.geometry.staggered),                     \
                        &fields.values[owner_index(                               \
                            HomogeneousPrimaryOwner::name)]});
    NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_BIND_TERRAIN_OWNER)
#undef NIMBUS_D5_BIND_TERRAIN_OWNER
    return bindings;
}

const char* homogeneous_primary_profile_algorithm_id() noexcept
{
    return "checkpointed_flat_base_profiles_v1";
}

namespace
{
template <typename Included, typename Population, typename Density, typename Volume>
DomainRecyclingPrivateInventory reduce_private_inventory(
    const SimulationStateView& state, Included included, Population population, Density density, Volume volume)
{
    DomainRecyclingPrivateInventory result;
    if (!state.microphysics_runtime || !state.microphysics_runtime->scheme)
        return result;
    const auto fields = state.microphysics_runtime->scheme->prognostic_scalars_for_transport();
    if (fields.size() > result.entries.size())
        throw std::invalid_argument("D5 private inventory has too many carriers");
    result.count      = static_cast<std::uint16_t>(fields.size());
    const auto domain = physical_scalar_domain(state.grid);
    for (std::size_t f = 0; f < fields.size(); ++f)
    {
        const auto& field = fields[f];
        if (!field.name || !field.field || !field.positive_definite ||
            (field.storage != MicrophysicsScalarStorage::DryAirMixingRatio &&
             field.storage != MicrophysicsScalarStorage::VolumeConcentration))
            throw std::invalid_argument("D5 private inventory binding is invalid");
        require_grid_field(*field.field, state, "private inventory");
        const std::string_view name(field.name);
        auto&                  entry = result.entries[f];
        if (name.empty() || name.size() >= entry.name.bytes.size())
            throw std::invalid_argument("D5 private inventory name is invalid");
        std::copy(name.begin(), name.end(), entry.name.bytes.begin());
        entry.name.size = static_cast<std::uint16_t>(name.size());
        tmv::diagnostics::CompensatedSum sum;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                if (!included(i, j))
                    continue;
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                {
                    const double n   = population(f, field, i, j, k);
                    const double rho = density(i, j, k), cell_volume = volume(i, j, k);
                    const double amount =
                        n * cell_volume * (field.storage == MicrophysicsScalarStorage::DryAirMixingRatio ? rho : 1.0);
                    if (!std::isfinite(n) || n < 0 || !std::isfinite(rho) || rho <= 0 || !std::isfinite(cell_volume) ||
                        cell_volume <= 0 || !std::isfinite(amount))
                        throw std::invalid_argument("D5 private inventory has an invalid physical carrier");
                    sum.add(amount);
                }
            }
        entry.inventory = sum.value();
    }
    if (!valid_domain_recycling_private_inventory(result))
        throw std::invalid_argument("D5 private inventory names or totals are invalid");
    return result;
}

DomainRecyclingPrivateInventory live_private_inventory(const SimulationStateView& state,
                                                       const WholeCellShiftPlan*  retiring = nullptr)
{
    return reduce_private_inventory(
        state,
        [&](int i, int j)
        {
            return !retiring || i < retiring->source_i_begin || i >= retiring->source_i_end ||
                   j < retiring->source_j_begin || j >= retiring->source_j_end;
        },
        [](std::size_t, const auto& field, int i, int j, int k) { return (*field.field)(i, j, k); },
        [&](int i, int j, int k) { return state.fields.rho(i, j, k); },
        [&](int i, int j, int k) { return diagnostic_cell_volume(state, i, j, k); });
}
} // namespace

TrailingPhysicalExportLedger compute_trailing_physical_export_ledger(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan)
{
    const WholeCellShiftPlan canonical = plan_whole_cell_shift_for_physical_domain(
        plan.nr, plan.nth, plan.nz, plan.shift_i_cells, plan.shift_j_cells,
        plan.physical_i_begin, plan.physical_i_end,
        plan.physical_j_begin, plan.physical_j_end);
    if (plan.nr != state.grid.nr || plan.nth != state.grid.nth ||
        plan.nz != state.grid.nz ||
        canonical.destination_i_begin != plan.destination_i_begin ||
        canonical.destination_i_end != plan.destination_i_end ||
        canonical.destination_j_begin != plan.destination_j_begin ||
        canonical.destination_j_end != plan.destination_j_end ||
        canonical.source_i_begin != plan.source_i_begin ||
        canonical.source_i_end != plan.source_i_end ||
        canonical.source_j_begin != plan.source_j_begin ||
        canonical.source_j_end != plan.source_j_end ||
        canonical.physical_i_begin != plan.physical_i_begin ||
        canonical.physical_i_end != plan.physical_i_end ||
        canonical.physical_j_begin != plan.physical_j_begin ||
        canonical.physical_j_end != plan.physical_j_end ||
        canonical.retained_cells != plan.retained_cells ||
        canonical.leading_cells != plan.leading_cells ||
        canonical.trailing_cells != plan.trailing_cells)
        throw std::invalid_argument(
            "trailing physical ledger requires a canonical grid-matching shift plan");
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    if (plan.physical_i_begin != physical.i_begin ||
        plan.physical_i_end != physical.i_end ||
        plan.physical_j_begin != physical.j_begin ||
        plan.physical_j_end != physical.j_end)
        throw std::invalid_argument(
            "trailing physical ledger plan must match the model physical scalar domain");
    if (state.fields.rho.empty())
        throw std::invalid_argument(
            "trailing physical ledger requires allocated atmospheric state");

    const Field3D* theta_residual = nullptr;
    if (state.microphysics_runtime == nullptr ||
        state.radiation_runtime == nullptr)
        throw std::invalid_argument(
            "trailing physical ledger requires bound physics runtimes");
    if (!state.microphysics_runtime->theta_storage_residual.empty())
    {
        const Field3D& residual =
            state.microphysics_runtime->theta_storage_residual;
        if (residual.size_r() != state.grid.nr ||
            residual.size_th() != state.grid.nth ||
            residual.size_z() != state.grid.nz)
            throw std::invalid_argument(
                "trailing theta storage residual shape does not match the grid");
        theta_residual = &residual;
    }
    else if (state.radiation_runtime->config.scheme_id != "none")
    {
        throw std::invalid_argument(
            "active radiation trailing ledger requires theta storage residual");
    }

    const HorizontalBudgetExclusion retained_source{
        plan.source_i_begin, plan.source_i_end,
        plan.source_j_begin, plan.source_j_end};
    TrailingPhysicalExportLedger ledger;
    ledger.inventories =
        compute_conservation_budget_excluding_horizontal_rectangle(
            state, retained_source);
    const HorizontalMomentumBudgetExclusion retained_momentum_source{
        plan.source_i_begin, plan.source_i_end,
        plan.source_j_begin, plan.source_j_end};
    ledger.momentum =
        compute_momentum_budget_excluding_horizontal_rectangle(
            state, retained_momentum_source);
    const HorizontalEnergyBudgetExclusion retained_energy_source{
        plan.source_i_begin, plan.source_i_end,
        plan.source_j_begin, plan.source_j_end};
    ledger.moist_mixture_energy =
        compute_moist_mixture_energy_v2_excluding_horizontal_rectangle(
            state, retained_energy_source);

    double residual_sum = 0.0;
    double residual_correction = 0.0;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    const auto add_residual_energy = [&](double value)
    {
        const double next = residual_sum + value;
        if (std::abs(residual_sum) >= std::abs(value))
            residual_correction += (residual_sum - next) + value;
        else
            residual_correction += (value - next) + residual_sum;
        residual_sum = next;
    };

    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
        {
            if (i >= plan.source_i_begin && i < plan.source_i_end &&
                j >= plan.source_j_begin && j < plan.source_j_end)
                continue;
            ++ledger.physical_columns;
            if (theta_residual != nullptr)
                for (int k = physical.k_begin; k < physical.k_end; ++k)
                {
                    const double residual =
                        static_cast<double>((*theta_residual)[i][j][k]);
                    const double rho_d =
                        static_cast<double>(state.fields.rho[i][j][k]);
                    const double pressure =
                        static_cast<double>(state.fields.p[i][j][k]);
                    const double volume =
                        diagnostic_cell_volume(state, i, j, k);
                    if (!std::isfinite(residual) ||
                        !std::isfinite(rho_d) || rho_d <= 0.0 ||
                        !std::isfinite(pressure) || pressure <= 0.0 ||
                        !std::isfinite(volume) || volume <= 0.0)
                        throw std::invalid_argument(
                            "trailing theta residual energy requires finite physical state");
                    const double exner = std::pow(
                        pressure /
                            physical_constants::reference_pressure_pa,
                        kappa);
                    const double energy = residual * rho_d *
                        physical_constants::specific_heat_cp_jkgk *
                        exner * volume;
                    if (!std::isfinite(energy))
                        throw std::overflow_error(
                            "trailing theta residual energy overflowed");
                    add_residual_energy(energy);
                }
        }
    ledger.theta_storage_residual_energy_j =
        residual_sum + residual_correction;
    if (!std::isfinite(ledger.theta_storage_residual_energy_j))
        throw std::overflow_error(
            "trailing theta residual energy sum overflowed");
    ledger.physical_cells = checked_ledger_product(
        ledger.physical_columns, static_cast<std::size_t>(physical.nz()));
    ledger.private_inventory = live_private_inventory(state, &plan);
    return ledger;
}

const char* trailing_scalar_export_ledger_basis_id() noexcept
{
    return "production_conservation_budget_retiring_cells_v1";
}

const char* trailing_momentum_export_ledger_basis_id() noexcept
{
    return "production_scalar_centered_momentum_retiring_cells_v1";
}

const char* trailing_energy_export_ledger_basis_id() noexcept
{
    return "physical_moist_mixture_v2_retiring_cells_v1";
}

const char* trailing_theta_residual_export_ledger_basis_id() noexcept
{
    return "deferred_theta_storage_residual_energy_retiring_cells_v1";
}

namespace
{

class CompensatedLedgerSum
{
public:
    void add(double value) noexcept
    {
        const double next = sum_ + value;
        if (std::abs(sum_) >= std::abs(value))
            correction_ += (sum_ - next) + value;
        else
            correction_ += (value - next) + sum_;
        sum_ = next;
    }

    [[nodiscard]] double value() const noexcept
    {
        return sum_ + correction_;
    }

private:
    double sum_ = 0.0;
    double correction_ = 0.0;
};

const Field3D& primary_owner_field(
    const SimulationStateView& state, HomogeneousPrimaryOwner owner)
{
    switch (owner)
    {
#define NIMBUS_D5_PRIMARY_FIELD_CASE(name)                                           \
        case HomogeneousPrimaryOwner::name: return state.fields.name;
        NIMBUS_D5_PRIMARY_OWNER_LIST(NIMBUS_D5_PRIMARY_FIELD_CASE)
#undef NIMBUS_D5_PRIMARY_FIELD_CASE
        case HomogeneousPrimaryOwner::count: break;
    }
    throw std::invalid_argument("invalid homogeneous primary owner");
}

float virtual_post_shift_primary_value(
    const SimulationStateView& state,
    const std::array<WholeCellShiftPlan,
                     homogeneous_primary_owner_count()>& owner_plans,
    const HomogeneousPrimaryProfiles& profiles,
    const TerrainAwarePrimaryFields* terrain_fields,
    HomogeneousPrimaryOwner owner,
    int i, int j, int k)
{
    const WholeCellShiftPlan& owner_plan = owner_plans[owner_index(owner)];
    if (i < owner_plan.physical_i_begin || i >= owner_plan.physical_i_end
        || j < owner_plan.physical_j_begin || j >= owner_plan.physical_j_end
        || k < 0 || k >= owner_plan.nz)
        throw std::out_of_range(
            "virtual D5 primary access is outside owner physical storage");
    if (i >= owner_plan.destination_i_begin
        && i < owner_plan.destination_i_end
        && j >= owner_plan.destination_j_begin
        && j < owner_plan.destination_j_end)
    {
        return primary_owner_field(state, owner)
            [i + owner_plan.shift_i_cells]
            [j + owner_plan.shift_j_cells][k];
    }
    return terrain_fields != nullptr
        ? (*terrain_fields)[owner](i, j, k)
        : profiles[owner][static_cast<std::size_t>(k)];
}

void validate_terrain_leading_fields(
    const SimulationStateView& state,
    const TerrainAwarePrimaryFields& fields)
{
    for (const auto& field : fields.values)
    {
        require_grid_field(field, state, "terrain-aware leading primary");
        for (std::size_t value = 0; value < field.size(); ++value)
            if (!std::isfinite(static_cast<double>(field.data()[value])))
                throw std::invalid_argument(
                    "terrain-aware leading primary contains a non-finite value");
    }
}

void validate_leading_profiles(
    const SimulationStateView& state,
    const HomogeneousPrimaryProfiles& profiles)
{
    for (std::size_t owner = 0; owner < homogeneous_primary_owner_count(); ++owner)
    {
        const auto& profile = profiles.values[owner];
        if (profile.size() != static_cast<std::size_t>(state.grid.nz))
            throw std::invalid_argument(
                "D5 exchange ledger requires every leading primary profile");
        for (float value : profile)
            if (!std::isfinite(static_cast<double>(value)))
                throw std::invalid_argument(
                    "D5 exchange ledger leading profile is non-finite");
    }
}

double complete_theta_storage_residual_energy(
    const SimulationStateView& state)
{
    if (state.microphysics_runtime == nullptr
        || state.radiation_runtime == nullptr)
        throw std::invalid_argument(
            "D5 exchange ledger requires bound physics runtimes");
    const Field3D& residual =
        state.microphysics_runtime->theta_storage_residual;
    if (residual.empty())
    {
        if (state.radiation_runtime->config.scheme_id != "none")
            throw std::invalid_argument(
                "active radiation exchange ledger requires theta storage residual");
        return 0.0;
    }
    require_grid_field(residual, state, "theta storage residual");
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    CompensatedLedgerSum sum;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
            for (int k = physical.k_begin; k < physical.k_end; ++k)
            {
                const double residual_k =
                    static_cast<double>(residual[i][j][k]);
                const double rho_d =
                    static_cast<double>(state.fields.rho[i][j][k]);
                const double pressure =
                    static_cast<double>(state.fields.p[i][j][k]);
                const double volume = diagnostic_cell_volume(state, i, j, k);
                const double exner = std::pow(
                    pressure / physical_constants::reference_pressure_pa,
                    kappa);
                const double energy = residual_k * rho_d
                    * physical_constants::specific_heat_cp_jkgk * exner
                    * volume;
                if (!std::isfinite(residual_k) || !std::isfinite(rho_d)
                    || rho_d <= 0.0 || !std::isfinite(pressure)
                    || pressure <= 0.0 || !std::isfinite(volume)
                    || volume <= 0.0 || !std::isfinite(energy))
                    throw std::invalid_argument(
                        "D5 exchange theta residual inventory is invalid");
                sum.add(energy);
            }
    if (!std::isfinite(sum.value()))
        throw std::overflow_error(
            "D5 exchange theta residual inventory overflowed");
    return sum.value();
}

PhysicalRegionInventoryLedger complete_physical_inventory_ledger(
    const SimulationStateView& state)
{
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    PhysicalRegionInventoryLedger ledger;
    ledger.inventories = compute_conservation_budget(state);
    ledger.momentum = compute_momentum_budget(state);
    ledger.moist_mixture_energy = compute_moist_mixture_energy_v2(state);
    ledger.theta_storage_residual_energy_j =
        complete_theta_storage_residual_energy(state);
    ledger.physical_columns = checked_ledger_product(
        static_cast<std::size_t>(physical.nx()),
        static_cast<std::size_t>(physical.ny()));
    ledger.physical_cells = checked_ledger_product(
        ledger.physical_columns, static_cast<std::size_t>(physical.nz()));
    ledger.private_inventory = live_private_inventory(state);
    return ledger;
}

PhysicalRegionInventoryLedger compute_admitted_physical_inventory_ledger(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& profiles,
    const TerrainAwarePrimaryFields* terrain_fields,
    const TerrainMetrics3D* post_shift_metrics,
    bool admitted_only,
    const PreparedMicrophysicsDomainRecyclingState* private_leading)
{
    constexpr double cv_d =
        physical_constants::specific_heat_cp_jkgk
        - physical_constants::gas_constant_dry_air_jkgk;
    constexpr double cv_v =
        physical_constants::specific_heat_water_vapor_cp_jkgk
        - physical_constants::gas_constant_water_vapor_jkgk;
    constexpr double c_l =
        physical_constants::specific_heat_liquid_water_jkgk;
    constexpr double c_i = physical_constants::specific_heat_ice_jkgk;
    constexpr double t_ref = physical_constants::freezing_temperature_k;
    constexpr double vapor_reference =
        physical_constants::latent_heat_vaporization_jkg
        - physical_constants::gas_constant_water_vapor_jkgk * t_ref;
    constexpr double ice_reference =
        -physical_constants::latent_heat_fusion_jkg;
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;

    enum Sum : std::size_t
    {
        dry_mass,
        theta_mass,
        water_mass,
        thermal,
        qv_mass,
        qc_mass,
        qr_mass,
        qi_mass,
        qs_mass,
        qg_mass,
        qh_mass,
        momentum_x,
        momentum_y,
        momentum_z,
        angular_z,
        dry_internal,
        water_sensible,
        phase_reference,
        resolved_kinetic,
        gravitational,
        subgrid_tke,
        energy_water_mass,
        count
    };
    std::array<CompensatedLedgerSum, Sum::count> sums;
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    std::array<WholeCellShiftPlan, homogeneous_primary_owner_count()>
        owner_plans;
    for (std::size_t owner = 0; owner < owner_plans.size(); ++owner)
    {
        const auto primary = static_cast<HomogeneousPrimaryOwner>(owner);
        owner_plans[owner] = plan_whole_cell_shift_for_horizontal_location(
            plan, primary_horizontal_location(
                primary, state.grid.geometry.staggered));
    }
    const double x_mid = 0.5
        * static_cast<double>(physical.i_begin + physical.i_end)
        * state.grid.dr;
    const double y_mid = 0.5
        * static_cast<double>(physical.j_begin + physical.j_end)
        * state.grid.dtheta;
    std::size_t columns = 0;

    const auto value = [&](HomogeneousPrimaryOwner owner, int i, int j, int k)
    {
        return static_cast<double>(virtual_post_shift_primary_value(
            state, owner_plans, profiles, terrain_fields, owner, i, j, k));
    };

    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
        {
            if (admitted_only
                && i >= plan.destination_i_begin
                && i < plan.destination_i_end
                && j >= plan.destination_j_begin
                && j < plan.destination_j_end)
                continue;
            ++columns;
            const double x_coord =
                static_cast<std::size_t>(i) < state.grid.geometry.r.size()
                ? state.grid.geometry.r[static_cast<std::size_t>(i)]
                : tmv::grid_coordinates::scalar_center(i, state.grid.dr);
            const double y_coord =
                static_cast<std::size_t>(j) < state.grid.geometry.theta.size()
                ? state.grid.geometry.theta[static_cast<std::size_t>(j)]
                : tmv::grid_coordinates::scalar_center(j, state.grid.dtheta);
            const double x_rel = x_coord - x_mid;
            const double y_rel = y_coord - y_mid;
            for (int k = physical.k_begin; k < physical.k_end; ++k)
            {
                const double rho_d = value(HomogeneousPrimaryOwner::rho, i, j, k);
                const double pressure = value(HomogeneousPrimaryOwner::p, i, j, k);
                const double theta_k = value(HomogeneousPrimaryOwner::theta, i, j, k);
                const double qv = value(HomogeneousPrimaryOwner::qv, i, j, k);
                const double qc = value(HomogeneousPrimaryOwner::qc, i, j, k);
                const double qr = value(HomogeneousPrimaryOwner::qr, i, j, k);
                const double qi = value(HomogeneousPrimaryOwner::qi, i, j, k);
                const double qs = value(HomogeneousPrimaryOwner::qs, i, j, k);
                const double qg = value(HomogeneousPrimaryOwner::qg, i, j, k);
                const double qh = value(HomogeneousPrimaryOwner::qh, i, j, k);
                const double tke = value(HomogeneousPrimaryOwner::tke, i, j, k);
                const double u_w = value(HomogeneousPrimaryOwner::u, i - 1, j, k);
                const double u_e = value(HomogeneousPrimaryOwner::u, i, j, k);
                const double v_s = value(HomogeneousPrimaryOwner::v, i, j - 1, k);
                const double v_n = value(HomogeneousPrimaryOwner::v, i, j, k);
                const double w_b = k == physical.k_begin ? 0.0
                    : value(HomogeneousPrimaryOwner::w, i, j, k - 1);
                const double w_t = value(HomogeneousPrimaryOwner::w, i, j, k);
                const double u_center = 0.5 * (u_w + u_e);
                const double v_center = 0.5 * (v_s + v_n);
                double kinetic_w_b = w_b;
                if (post_shift_metrics != nullptr
                    && k == physical.k_begin)
                {
                    const double frame_u = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.u_ms : 0.0;
                    const double frame_v = state.domain_frame != nullptr
                        ? state.domain_frame->velocity_ms.v_ms : 0.0;
                    kinetic_w_b = (u_center - frame_u)
                            * (-post_shift_metrics->Gx_face(i, j, 0))
                        + (v_center - frame_v)
                            * (-post_shift_metrics->Gy_face(i, j, 0));
                }
                const double w_center = 0.5 * (w_b + w_t);
                const double volume = post_shift_metrics != nullptr
                    ? state.grid.dr * state.grid.dtheta
                        * (post_shift_metrics->z_face(i, j, k + 1)
                           - post_shift_metrics->z_face(i, j, k))
                    : tmv::physical_grid_metrics::cell_volume(
                        state, i, j, k);
                const double z = post_shift_metrics != nullptr
                    ? post_shift_metrics->z(i, j, k)
                    : tmv::physical_grid_metrics::scalar_height(
                        state, i, j, k);
                const double temperature = theta_k * std::pow(
                    pressure / physical_constants::reference_pressure_pa,
                    kappa);
                const double q_liquid = qc + qr;
                const double q_ice = qi + qs + qg + qh;
                const double q_total = qv + q_liquid + q_ice;
                const double values[] = {
                    rho_d, pressure, theta_k, qv, qc, qr, qi, qs, qg, qh,
                    tke, u_w, u_e, v_s, v_n, w_b, w_t, kinetic_w_b,
                    volume, z,
                    temperature, q_total, x_rel, y_rel,
                };
                if (std::any_of(
                        std::begin(values), std::end(values),
                        [](double sample) { return !std::isfinite(sample); })
                    || rho_d <= 0.0 || pressure <= 0.0 || theta_k <= 0.0
                    || temperature <= 0.0 || volume <= 0.0 || qv < 0.0
                    || qc < 0.0 || qr < 0.0 || qi < 0.0 || qs < 0.0
                    || qg < 0.0 || qh < 0.0 || tke < 0.0)
                    throw std::invalid_argument(
                        "D5 admitted physical inventory is nonphysical");

                const double dry_kg = rho_d * volume;
                const double mixture_kg = dry_kg * (1.0 + q_total);
                const double delta_t = temperature - t_ref;
                const double kinetic = 0.25 * mixture_kg
                    * (u_w * u_w + u_e * u_e + v_s * v_s + v_n * v_n
                       + kinetic_w_b * kinetic_w_b + w_t * w_t);
                sums[Sum::dry_mass].add(dry_kg);
                sums[Sum::theta_mass].add(dry_kg * theta_k);
                sums[Sum::water_mass].add(dry_kg * q_total);
                sums[Sum::thermal].add(
                    dry_kg * physical_constants::specific_heat_cp_jkgk
                    * temperature);
                sums[Sum::qv_mass].add(dry_kg * qv);
                sums[Sum::qc_mass].add(dry_kg * qc);
                sums[Sum::qr_mass].add(dry_kg * qr);
                sums[Sum::qi_mass].add(dry_kg * qi);
                sums[Sum::qs_mass].add(dry_kg * qs);
                sums[Sum::qg_mass].add(dry_kg * qg);
                sums[Sum::qh_mass].add(dry_kg * qh);
                sums[Sum::momentum_x].add(dry_kg * u_center);
                sums[Sum::momentum_y].add(dry_kg * v_center);
                sums[Sum::momentum_z].add(dry_kg * w_center);
                sums[Sum::angular_z].add(
                    dry_kg * (x_rel * v_center - y_rel * u_center));
                sums[Sum::dry_internal].add(dry_kg * cv_d * delta_t);
                sums[Sum::water_sensible].add(
                    dry_kg * delta_t
                    * (qv * cv_v + q_liquid * c_l + q_ice * c_i));
                sums[Sum::phase_reference].add(
                    dry_kg * (qv * vapor_reference + q_ice * ice_reference));
                sums[Sum::resolved_kinetic].add(kinetic);
                sums[Sum::gravitational].add(
                    mixture_kg * physical_constants::gravity_ms2 * z);
                sums[Sum::subgrid_tke].add(dry_kg * tke);
                sums[Sum::energy_water_mass].add(dry_kg * q_total);
            }
        }

    PhysicalRegionInventoryLedger ledger;
    ledger.physical_columns = columns;
    ledger.physical_cells = checked_ledger_product(
        columns, static_cast<std::size_t>(physical.nz()));
    auto& c = ledger.inventories;
    c.total_mass = c.dry_mass = sums[Sum::dry_mass].value();
    c.theta_volume = sums[Sum::theta_mass].value();
    c.q_volume_total = c.total_water = sums[Sum::water_mass].value();
    c.water_species_mass = {{
        sums[Sum::qv_mass].value(), sums[Sum::qc_mass].value(),
        sums[Sum::qr_mass].value(), sums[Sum::qi_mass].value(),
        sums[Sum::qs_mass].value(), sums[Sum::qg_mass].value(),
        sums[Sum::qh_mass].value(),
    }};
    c.moist_total_mass = c.dry_mass + c.total_water;
    c.thermal_energy = sums[Sum::thermal].value();
    auto& m = ledger.momentum;
    m.momentum_x = sums[Sum::momentum_x].value();
    m.momentum_y = sums[Sum::momentum_y].value();
    m.momentum_z = sums[Sum::momentum_z].value();
    m.angular_momentum_z = sums[Sum::angular_z].value();
    m.horizontal_magnitude = std::hypot(m.momentum_x, m.momentum_y);
    m.total_magnitude = std::hypot(m.horizontal_magnitude, m.momentum_z);
    m.total_mass = c.dry_mass;
    m.reference_linear = m.total_mass * MOMENTUM_REF_VELOCITY_MS;
    m.reference_angular = m.reference_linear
        * static_cast<double>(physical.nx()) * state.grid.dr;
    auto& e = ledger.moist_mixture_energy;
    e.dry_internal = sums[Sum::dry_internal].value();
    e.water_sensible = sums[Sum::water_sensible].value();
    e.phase_reference = sums[Sum::phase_reference].value();
    e.resolved_kinetic = sums[Sum::resolved_kinetic].value();
    e.gravitational = sums[Sum::gravitational].value();
    e.subgrid_tke = sums[Sum::subgrid_tke].value();
    e.total_water_mass = sums[Sum::energy_water_mass].value();
    e.total = e.dry_internal + e.water_sensible + e.phase_reference
        + e.resolved_kinetic + e.gravitational + e.subgrid_tke;
    ledger.theta_storage_residual_energy_j = 0.0;
    ledger.private_inventory               = reduce_private_inventory(
        state,
        [&](int i, int j)
        {
            return !admitted_only || i < plan.destination_i_begin || i >= plan.destination_i_end ||
                   j < plan.destination_j_begin || j >= plan.destination_j_end;
        },
        [&](std::size_t f, const auto& field, int i, int j, int k)
        {
            if (!private_leading || private_leading->nr != plan.nr || private_leading->nth != plan.nth ||
                private_leading->nz != plan.nz || private_leading->shift_i_cells != plan.shift_i_cells ||
                private_leading->shift_j_cells != plan.shift_j_cells ||
                private_leading->terrain_aware != (terrain_fields != nullptr) ||
                f >= private_leading->private_scalar_names.size() ||
                private_leading->private_scalar_names[f] != field.name)
                throw std::invalid_argument("D5 private leading state does not match the exchange");
            if (i >= plan.destination_i_begin && i < plan.destination_i_end && j >= plan.destination_j_begin &&
                j < plan.destination_j_end)
                return (*field.field)(i + plan.shift_i_cells, j + plan.shift_j_cells, k);
            if (terrain_fields)
            {
                if (f >= private_leading->terrain_leading_private_scalars.size())
                    throw std::invalid_argument("D5 mapped private leading field is missing");
                const auto& leading = private_leading->terrain_leading_private_scalars[f];
                require_grid_field(leading, state, "private leading inventory");
                return leading(i, j, k);
            }
            if (f >= private_leading->leading_private_scalars.size() ||
                private_leading->leading_private_scalars[f].size() != static_cast<std::size_t>(plan.nz))
                throw std::invalid_argument("D5 private leading profile is missing");
            return private_leading->leading_private_scalars[f][k];
        },
        [&](int i, int j, int k) { return value(HomogeneousPrimaryOwner::rho, i, j, k); },
        [&](int i, int j, int k)
        {
            return post_shift_metrics
                       ? state.grid.dr * state.grid.dtheta *
                             (post_shift_metrics->z_face(i, j, k + 1) - post_shift_metrics->z_face(i, j, k))
                       : tmv::physical_grid_metrics::cell_volume(state, i, j, k);
        });
    return ledger;
}

double fixed_earth_angular_momentum(
    const MomentumBudget& momentum,
    const SimulationStateView& state,
    const HorizontalDomainFrame& frame)
{
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    const HorizontalPoint local_center{
        0.5 * static_cast<double>(physical.i_begin + physical.i_end)
            * state.grid.dr,
        0.5 * static_cast<double>(physical.j_begin + physical.j_end)
            * state.grid.dtheta,
    };
    const HorizontalPoint earth_center =
        frame.grid_to_earth(local_center, state.time_s);
    const double fixed = momentum.angular_momentum_z
        + earth_center.x_m * momentum.momentum_y
        - earth_center.y_m * momentum.momentum_x;
    if (!std::isfinite(fixed))
        throw std::overflow_error(
            "D5 fixed-Earth angular momentum overflowed");
    return fixed;
}

PhysicalRegionInventoryLedger subtract_add_ledgers(
    const PhysicalRegionInventoryLedger& before,
    const PhysicalRegionInventoryLedger& retiring,
    const PhysicalRegionInventoryLedger& admitted,
    double expected_local_angular)
{
    PhysicalRegionInventoryLedger out;
    const auto combine = [](double a, double b, double c)
    {
        return a - b + c;
    };
    auto& oc = out.inventories;
    const auto& bc = before.inventories;
    const auto& rc = retiring.inventories;
    const auto& ac = admitted.inventories;
    oc.total_mass = combine(bc.total_mass, rc.total_mass, ac.total_mass);
    oc.theta_volume = combine(
        bc.theta_volume, rc.theta_volume, ac.theta_volume);
    oc.q_volume_total = combine(
        bc.q_volume_total, rc.q_volume_total, ac.q_volume_total);
    oc.dry_mass = combine(bc.dry_mass, rc.dry_mass, ac.dry_mass);
    oc.total_water = combine(
        bc.total_water, rc.total_water, ac.total_water);
    for (std::size_t species = 0; species < oc.water_species_mass.size(); ++species)
        oc.water_species_mass[species] = combine(
            bc.water_species_mass[species],
            rc.water_species_mass[species],
            ac.water_species_mass[species]);
    oc.moist_total_mass = combine(
        bc.moist_total_mass, rc.moist_total_mass, ac.moist_total_mass);
    oc.thermal_energy = combine(
        bc.thermal_energy, rc.thermal_energy, ac.thermal_energy);
    auto& om = out.momentum;
    const auto& bm = before.momentum;
    const auto& rm = retiring.momentum;
    const auto& am = admitted.momentum;
    om.momentum_x = combine(bm.momentum_x, rm.momentum_x, am.momentum_x);
    om.momentum_y = combine(bm.momentum_y, rm.momentum_y, am.momentum_y);
    om.momentum_z = combine(bm.momentum_z, rm.momentum_z, am.momentum_z);
    om.angular_momentum_z = expected_local_angular;
    om.horizontal_magnitude = std::hypot(om.momentum_x, om.momentum_y);
    om.total_magnitude = std::hypot(om.horizontal_magnitude, om.momentum_z);
    om.total_mass = oc.dry_mass;
    om.reference_linear = om.total_mass * MOMENTUM_REF_VELOCITY_MS;
    om.reference_angular = before.momentum.reference_angular
        / before.momentum.total_mass * om.total_mass;
    auto& oe = out.moist_mixture_energy;
    const auto& be = before.moist_mixture_energy;
    const auto& re = retiring.moist_mixture_energy;
    const auto& ae = admitted.moist_mixture_energy;
    oe.dry_internal = combine(
        be.dry_internal, re.dry_internal, ae.dry_internal);
    oe.water_sensible = combine(
        be.water_sensible, re.water_sensible, ae.water_sensible);
    oe.phase_reference = combine(
        be.phase_reference, re.phase_reference, ae.phase_reference);
    oe.resolved_kinetic = combine(
        be.resolved_kinetic, re.resolved_kinetic, ae.resolved_kinetic);
    oe.gravitational = combine(
        be.gravitational, re.gravitational, ae.gravitational);
    oe.subgrid_tke = combine(
        be.subgrid_tke, re.subgrid_tke, ae.subgrid_tke);
    oe.total_water_mass = combine(
        be.total_water_mass, re.total_water_mass, ae.total_water_mass);
    oe.total = combine(be.total, re.total, ae.total);
    out.theta_storage_residual_energy_j = combine(
        before.theta_storage_residual_energy_j,
        retiring.theta_storage_residual_energy_j,
        admitted.theta_storage_residual_energy_j);
    if (!valid_domain_recycling_private_inventory(before.private_inventory) ||
        !valid_domain_recycling_private_inventory(retiring.private_inventory) ||
        !valid_domain_recycling_private_inventory(admitted.private_inventory) ||
        before.private_inventory.count != retiring.private_inventory.count ||
        before.private_inventory.count != admitted.private_inventory.count)
        throw std::invalid_argument("D5 private inventory layout differs across the exchange");
    out.private_inventory = before.private_inventory;
    for (std::size_t f = 0; f < out.private_inventory.count; ++f)
    {
        auto&       entry   = out.private_inventory.entries[f];
        const auto& removed = retiring.private_inventory.entries[f];
        const auto& added   = admitted.private_inventory.entries[f];
        if (entry.name.view() != removed.name.view() || entry.name.view() != added.name.view())
            throw std::invalid_argument("D5 private inventory names differ across the exchange");
        entry.inventory = combine(entry.inventory, removed.inventory, added.inventory);
    }
    if (!valid_domain_recycling_private_inventory(out.private_inventory))
        throw std::invalid_argument("D5 private exchange produces an invalid inventory");
    out.physical_columns = before.physical_columns;
    out.physical_cells = before.physical_cells;
    return out;
}

void require_close(double actual, double expected, const char* name,
                   double physical_scale_floor = 1.0)
{
    const double scale = std::max(
        {1.0, std::abs(actual), std::abs(expected),
         std::abs(physical_scale_floor)});
    if (!std::isfinite(actual) || !std::isfinite(expected)
        || std::abs(actual - expected) > 2.0e-11 * scale)
    {
        std::ostringstream message;
        message << std::setprecision(17)
                << "D5 control-volume exchange failed " << name
                << " closure: actual=" << actual
                << ", expected=" << expected
                << ", residual=" << (actual - expected)
                << ", relative=" << std::abs(actual - expected) / scale
                << ", tolerance=2e-11";
        throw std::runtime_error(message.str());
    }
}

void require_ledger_close(
    const PhysicalRegionInventoryLedger& actual,
    const PhysicalRegionInventoryLedger& expected)
{
    if (!valid_domain_recycling_private_inventory(actual.private_inventory) ||
        !valid_domain_recycling_private_inventory(expected.private_inventory) ||
        actual.private_inventory.count != expected.private_inventory.count)
        throw std::invalid_argument("D5 private closure has inconsistent carrier ownership");
    for (std::size_t f = 0; f < actual.private_inventory.count; ++f)
    {
        const auto& a = actual.private_inventory.entries[f];
        const auto& e = expected.private_inventory.entries[f];
        if (a.name.view() != e.name.view())
            throw std::invalid_argument("D5 private closure has inconsistent carrier names");
        require_close(a.inventory, e.inventory, a.name.bytes.data());
    }
    const auto& a = actual.inventories;
    const auto& e = expected.inventories;
    require_close(a.total_mass, e.total_mass, "dry mass");
    require_close(a.theta_volume, e.theta_volume, "rho-theta");
    require_close(a.q_volume_total, e.q_volume_total, "total water");
    require_close(a.moist_total_mass, e.moist_total_mass, "moist mass");
    require_close(a.thermal_energy, e.thermal_energy, "thermal proxy");
    for (std::size_t species = 0; species < a.water_species_mass.size(); ++species)
        require_close(a.water_species_mass[species],
                      e.water_species_mass[species], "water species");
    const auto& am = actual.momentum;
    const auto& em = expected.momentum;
    // Net momentum and especially domain-centred angular momentum may be a
    // small cancellation of large, physically transported cell inventories.
    // Use the same mass/velocity/lever-arm conditioning as the production
    // momentum diagnostic so a different deterministic summation order is not
    // mistaken for torque.  A real inventory defect is still tested at the
    // unchanged 2e-11 fraction of that physical characteristic scale.
    const double linear_scale = std::max(
        am.reference_linear, em.reference_linear);
    const double angular_scale = std::max(
        am.reference_angular, em.reference_angular);
    require_close(am.momentum_x, em.momentum_x, "x momentum", linear_scale);
    require_close(am.momentum_y, em.momentum_y, "y momentum", linear_scale);
    require_close(am.momentum_z, em.momentum_z, "z momentum", linear_scale);
    require_close(am.angular_momentum_z, em.angular_momentum_z,
                  "local angular momentum", angular_scale);
    const auto& ae = actual.moist_mixture_energy;
    const auto& ee = expected.moist_mixture_energy;
    require_close(ae.dry_internal, ee.dry_internal, "dry internal energy");
    require_close(ae.water_sensible, ee.water_sensible,
                  "water sensible energy");
    require_close(ae.phase_reference, ee.phase_reference,
                  "phase-reference energy");
    require_close(ae.resolved_kinetic, ee.resolved_kinetic,
                  "resolved kinetic energy");
    require_close(ae.gravitational, ee.gravitational,
                  "gravitational energy");
    require_close(ae.subgrid_tke, ee.subgrid_tke, "subgrid TKE energy");
    require_close(ae.total_water_mass, ee.total_water_mass,
                  "energy water mass");
    require_close(ae.total, ee.total, "physical-v2 total energy");
    require_close(actual.theta_storage_residual_energy_j,
                  expected.theta_storage_residual_energy_j,
                  "theta residual energy");
    if (actual.physical_columns != expected.physical_columns
        || actual.physical_cells != expected.physical_cells)
        throw std::runtime_error(
            "D5 control-volume exchange failed physical-count closure");
}

} // namespace

static DomainRecyclingControlVolumeExchangeLedger
compute_domain_recycling_control_volume_exchange_ledger_impl(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& leading_profiles,
    const TerrainAwarePrimaryFields* terrain_fields,
    const TerrainMetrics3D* post_shift_metrics,
    const HorizontalDomainFrame& pre_shift_frame,
    const PreparedMicrophysicsDomainRecyclingState* private_leading)
{
    if (!state.grid.geometry.staggered)
        throw std::invalid_argument(
            "D5 control-volume exchange currently requires the C-grid");
    if (state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "D5 control-volume exchange requires terrain runtime ownership");
    const bool terrain_aware = terrain_fields != nullptr;
    if ((!terrain_aware
         && state.terrain_runtime->config.scheme_id != "none")
        || (terrain_aware
            && (state.terrain_runtime->config.scheme_id == "none"
                || state.terrain_runtime->config.horizontal_placement_id
                    != "earth_fixed"
                || post_shift_metrics == nullptr)))
        throw std::invalid_argument(
            "D5 control-volume exchange terrain ownership is inconsistent");
    pre_shift_frame.validate();
    if (state.configuration == nullptr
        || pre_shift_frame.earth_origin_at_t0_m.x_m
               != state.configuration->domain_frame.earth_origin_at_t0_m.x_m
        || pre_shift_frame.earth_origin_at_t0_m.y_m
               != state.configuration->domain_frame.earth_origin_at_t0_m.y_m
        || pre_shift_frame.velocity_ms.u_ms
               != state.configuration->domain_frame.velocity_ms.u_ms
        || pre_shift_frame.velocity_ms.v_ms
               != state.configuration->domain_frame.velocity_ms.v_ms)
        throw std::invalid_argument(
            "D5 exchange pre-shift frame does not match live configuration");
    if (terrain_aware)
        validate_terrain_leading_fields(state, *terrain_fields);
    else
        validate_leading_profiles(state, leading_profiles);
    DomainRecyclingControlVolumeExchangeLedger ledger;
    ledger.pre_shift = complete_physical_inventory_ledger(state);
    ledger.retiring = compute_trailing_physical_export_ledger(state, plan);
    ledger.admitted = compute_admitted_physical_inventory_ledger(
        state, plan, leading_profiles, terrain_fields, post_shift_metrics,
        true, private_leading);
    if (ledger.retiring.physical_columns != ledger.admitted.physical_columns
        || ledger.retiring.physical_cells != ledger.admitted.physical_cells)
        throw std::runtime_error(
            "D5 retiring/admitted control volumes do not have equal measure");
    ledger.pre_shift_fixed_earth_angular_momentum_z =
        fixed_earth_angular_momentum(
            ledger.pre_shift.momentum, state, pre_shift_frame);
    ledger.retiring_fixed_earth_angular_momentum_z =
        fixed_earth_angular_momentum(
            ledger.retiring.momentum, state, pre_shift_frame);
    HorizontalDomainFrame post_shift_frame = pre_shift_frame;
    commit_whole_cell_shift_to_domain_frame(
        post_shift_frame, plan, state.grid.dr, state.grid.dtheta);
    ledger.admitted_fixed_earth_angular_momentum_z =
        fixed_earth_angular_momentum(
            ledger.admitted.momentum, state, post_shift_frame);
    ledger.expected_post_shift_fixed_earth_angular_momentum_z =
        ledger.pre_shift_fixed_earth_angular_momentum_z
        - ledger.retiring_fixed_earth_angular_momentum_z
        + ledger.admitted_fixed_earth_angular_momentum_z;
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    const HorizontalPoint local_center{
        0.5 * static_cast<double>(physical.i_begin + physical.i_end)
            * state.grid.dr,
        0.5 * static_cast<double>(physical.j_begin + physical.j_end)
            * state.grid.dtheta,
    };
    const HorizontalPoint post_earth_center =
        post_shift_frame.grid_to_earth(local_center, state.time_s);
    const double expected_px = ledger.pre_shift.momentum.momentum_x
        - ledger.retiring.momentum.momentum_x
        + ledger.admitted.momentum.momentum_x;
    const double expected_py = ledger.pre_shift.momentum.momentum_y
        - ledger.retiring.momentum.momentum_y
        + ledger.admitted.momentum.momentum_y;
    const double expected_local_angular =
        ledger.expected_post_shift_fixed_earth_angular_momentum_z
        - post_earth_center.x_m * expected_py
        + post_earth_center.y_m * expected_px;
    ledger.expected_post_shift = subtract_add_ledgers(
        ledger.pre_shift, ledger.retiring, ledger.admitted,
        expected_local_angular);
    if (terrain_aware)
    {
        const double expected_theta_residual_energy_j =
            ledger.expected_post_shift.theta_storage_residual_energy_j;
        ledger.expected_post_shift =
            compute_admitted_physical_inventory_ledger(
                state, plan, leading_profiles, terrain_fields,
                post_shift_metrics, false, private_leading);
        ledger.expected_post_shift.theta_storage_residual_energy_j =
            expected_theta_residual_energy_j;
        ledger.expected_post_shift_fixed_earth_angular_momentum_z =
            fixed_earth_angular_momentum(
                ledger.expected_post_shift.momentum, state,
                post_shift_frame);
    }
    return ledger;
}

DomainRecyclingControlVolumeExchangeLedger
compute_domain_recycling_control_volume_exchange_ledger(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& leading_profiles,
    const HorizontalDomainFrame& pre_shift_frame,
    const PreparedMicrophysicsDomainRecyclingState* private_leading)
{
    return compute_domain_recycling_control_volume_exchange_ledger_impl(
        state, plan, leading_profiles, nullptr, nullptr,
        pre_shift_frame, private_leading);
}

DomainRecyclingControlVolumeExchangeLedger
compute_terrain_aware_domain_recycling_control_volume_exchange_ledger(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& leading_fields,
    const TerrainMetrics3D& post_shift_metrics,
    const HorizontalDomainFrame& pre_shift_frame,
    const PreparedMicrophysicsDomainRecyclingState* private_leading)
{
    return compute_domain_recycling_control_volume_exchange_ledger_impl(
        state, plan, {}, &leading_fields, &post_shift_metrics,
        pre_shift_frame, private_leading);
}

void verify_domain_recycling_control_volume_exchange_after_commit(
    const SimulationStateView& post_shift_state,
    const WholeCellShiftPlan& plan,
    const HorizontalDomainFrame& post_shift_frame,
    const DomainRecyclingControlVolumeExchangeLedger& ledger)
{
    const std::size_t expected_exchange_cells = plan.trailing_cells;
    if (ledger.retiring.physical_cells != expected_exchange_cells
        || ledger.admitted.physical_cells != expected_exchange_cells)
        throw std::invalid_argument(
            "D5 exchange verification ledger does not match the shift plan");
    const PhysicalRegionInventoryLedger actual =
        complete_physical_inventory_ledger(post_shift_state);
    require_ledger_close(actual, ledger.expected_post_shift);
    require_close(
        fixed_earth_angular_momentum(
            actual.momentum, post_shift_state, post_shift_frame),
        ledger.expected_post_shift_fixed_earth_angular_momentum_z,
        "fixed-Earth angular momentum");
}

const char* domain_recycling_control_volume_exchange_basis_id() noexcept
{
    return "instantaneous_fixed_earth_control_volume_exchange_v1";
}

const std::array<DomainRecyclingOwnerContract,
                 domain_recycling_owner_contract_count()>&
domain_recycling_owner_contracts() noexcept
{
    using Action = DomainRecyclingOwnerAction;
    using Ready = DomainRecyclingOwnerReadiness;
    static constexpr std::array<DomainRecyclingOwnerContract,
                                domain_recycling_owner_contract_count()> owners{{
        {"primary_atmosphere", Action::ShiftWithLeadingState,
         Ready::ReadyForTransaction,
         "15 prognostic Field3D owners with scalar/x-face/y-face plans"},
        {"base_state_profiles", Action::Preserve,
         Ready::ReadyForTransaction,
         "rho0/p0/qv0/theta0/u0/v0 remain the environmental reference sampled directly on flat grids and at each mapped physical height"},
        {"simulation_clock_and_constants", Action::Preserve,
         Ready::ReadyForTransaction,
         "time, timestep, spacing, Coriolis, and immutable configuration"},
        {"retiring_physical_ledgers", Action::ShiftAndExportEarthTile,
         Ready::ReadyForTransaction,
         "retiring/admitted mass-water-momentum-physical-v2 energy close as an instantaneous fixed-Earth control-volume exchange"},
        {"surface_instantaneous_diagnostics", Action::ShiftWithLeadingState,
         Ready::ReadyForTransaction,
         "sampled ustar, stresses, heat/moisture fluxes, and PBL height shift with evaluated leading columns"},
        {"surface_accumulated_rainfall", Action::ShiftAndExportEarthTile,
         Ready::ReadyForTransaction,
         "exact fallout rate/history shift and raw retiring-column export; update clock is preserved"},
        {"nested_grid", Action::TranslateEarthCoordinates,
         Ready::ReadyForTransaction,
         "synchronized separately evolving child fields, clocks, module objects, feedback currencies, and registers remain byte-stable while parent indices rebase to preserve the child Earth frame"},
        {"derived_diagnostic_fields", Action::InvalidateAndRecompute,
         Ready::ReadyForTransaction,
         "radar, vorticity/source, angular-momentum, and pressure products invalidate allocation-stably as four lazy publication groups"},
        {"dynamics_operator", Action::InvalidateAndRecompute,
         Ready::ReadyForTransaction,
         "per-simulation scheme persists; tendencies, compact reductions/uploads, packing, parity replay, and residency hints invalidate allocation-stably"},
        {"boundary_closure_and_kw78_history", Action::RebuildFromShiftedState,
         Ready::ReadyForTransaction,
         "C-grid velocity/rho/p/scalar closure and four radiative face histories"},
        {"radiation_runtime", Action::ShiftPersistentInvalidateTransient,
         Ready::ReadyForTransaction,
         "held tendency/power shift and raw-export; flat profiles or mapped physical-height columns use held sample time; aggregate commits atomically; scratch is overwrite-only"},
        {"turbulence_runtime", Action::InvalidateAndRecompute,
         Ready::ReadyForTransaction,
         "held/candidate nonlinear tendencies, residency hints, predictive stability, and geometry scratch invalidate atomically; cadence history and completed/cumulative ledgers persist"},
        {"boundary_layer_runtime", Action::ShiftPersistentInvalidateTransient,
         Ready::ReadyForTransaction,
         "skin/slab records, held forces/inventory rates, and held TKE shift with flat or mapped AGL leading columns; live-density tendencies rebuild"},
        {"chaos_runtime", Action::ShiftPersistentInvalidateTransient,
         Ready::ReadyForTransaction,
         "non-evolving caches invalidate atomically; evolving schemes swap a detached Earth-control-lattice AR(1) translation while deterministic PBL caches use the matching flat or mapped leading owner"},
        {"rayleigh_runtime", Action::Preserve,
         Ready::ReadyForTransaction,
         "validated vertical references and distance-from-computational-boundary coefficients are invariant under a flat horizontal translation"},
        {"microphysics_runtime", Action::ShiftPersistentInvalidateTransient,
         Ready::ReadyForTransaction,
         "theta residual and private prognostic moments shift/export with flat profiles or mapped full-field leading state; step products invalidate allocation-stably"},
        {"advection_runtime", Action::InvalidateAndRecompute,
         Ready::ReadyForTransaction,
         "scheme/config and cumulative ledgers persist; scratch, fused aliases, activity, request flags, and current reports invalidate atomically"},
        {"diffusion_runtime", Action::InvalidateAndRecompute,
         Ready::ReadyForTransaction,
         "scheme/config and cumulative work persist; tendencies, effective-K diagnostics, transactional candidates, and last-step report invalidate atomically"},
        {"time_stepping_runtime", Action::InvalidateAndRecompute,
         Ready::ReadyForTransaction,
         "diagnostic/cumulative history persists; slow-RK and scheme-private stages, acoustic CFL cache, and last-step reports invalidate atomically"},
        {"terrain_runtime", Action::ShiftWithLeadingState,
         Ready::ReadyForTransaction,
         "CPU Earth-fixed Bell/Schar geometry, metrics, placement generation, physical-height primary columns, slab PBL/surface, held radiation, private microphysics, theta residual, and evolving-chaos caches commit atomically"},
        {"radar_runtime", Action::InvalidateAndRecompute,
         Ready::ReadyForTransaction,
         "operator/configuration and Earth-fixed instrument geometry persist; selected, polarimetric, and scheme-private generation products invalidate allocation-stably"},
        {"compute_residency", Action::InvalidateDeviceResidency,
         Ready::ReadyForTransaction,
         "CPU rebuild invalidates every resident copy before closure; native accelerator shift is a separate device path"},
        {"parity_history", Action::Preserve,
         Ready::ReadyForTransaction,
         "cumulative comparison statistics and cadence counters are coordinate-independent history"},
        {"storm_tracking_and_partition", Action::TranslateEarthCoordinates,
         Ready::ReadyForTransaction,
         "continuity samples/envelope translate into the new grid frame; the selected physical-column partition shifts with zero leading membership; matching movement authority commits atomically"},
        {"lagrangian_trajectories", Action::TranslateEarthCoordinates,
         Ready::ReadyForTransaction,
         "Earth-relative path samples persist; live parcel working coordinates translate into the new grid frame and parcels in the retired swath deactivate"},
        {"domain_frame", Action::TranslateEarthCoordinates,
         Ready::ReadyForTransaction,
         "validated atomic Earth-origin commit advances by integer cells without changing stored Earth-relative wind"},
        {"durable_global_swath_output", Action::ShiftAndExportEarthTile,
         Ready::ReadyForTransaction,
         "canonical nonoverlapping physical rectangles publish synchronously in native/netCDF/qualified-ZFP with Earth coordinates and immutable captured-precommit transaction identity"},
        {"checkpoint_transaction_provenance", Action::TranslateEarthCoordinates,
         Ready::ReadyForTransaction,
         "allocation-free committed sequence/hash-chain head binds transaction id, pre/post origin, exact shift, retiring/admitted ledgers, algorithms, and durable swath manifest into checkpoint runtime v36"},
    }};
    return owners;
}

std::size_t remaining_domain_recycling_owner_contracts() noexcept
{
    std::size_t remaining = 0;
    for (const DomainRecyclingOwnerContract& owner :
         domain_recycling_owner_contracts())
    {
        if (owner.readiness !=
            DomainRecyclingOwnerReadiness::ReadyForTransaction)
            ++remaining;
    }
    return remaining;
}

namespace
{

template <std::size_t Capacity>
bool valid_provenance_text(
    const DomainRecyclingProvenanceText<Capacity>& text,
    bool identifier_only) noexcept
{
    if (text.size == 0 || text.size > Capacity || text.bytes[text.size] != '\0')
        return false;
    for (std::size_t index = 0; index < text.size; ++index)
    {
        const unsigned char value =
            static_cast<unsigned char>(text.bytes[index]);
        if (identifier_only)
        {
            if (!((value >= 'a' && value <= 'z')
                  || (value >= 'A' && value <= 'Z')
                  || (value >= '0' && value <= '9') || value == '-'
                  || value == '_'))
                return false;
        }
        else if (value < 0x21 || value > 0x7e)
            return false;
    }
    return true;
}

template <std::size_t Capacity>
void assign_provenance_text(
    DomainRecyclingProvenanceText<Capacity>& destination,
    std::string_view value,
    const char* label,
    bool identifier_only)
{
    DomainRecyclingProvenanceText<Capacity> candidate;
    if (value.empty() || value.size() > Capacity)
        throw std::invalid_argument(std::string("D5 provenance ") + label
                                    + " has invalid length");
    std::copy(value.begin(), value.end(), candidate.bytes.begin());
    candidate.size = static_cast<std::uint16_t>(value.size());
    if (!valid_provenance_text(candidate, identifier_only))
        throw std::invalid_argument(std::string("D5 provenance ") + label
                                    + " contains invalid characters");
    destination = candidate;
}

void commit_prepared_domain_recycling_provenance(
    void* opaque_prepared) noexcept
{
    auto& prepared = *static_cast<
        PreparedDomainRecyclingTransactionProvenance*>(opaque_prepared);
    prepared.runtime->domain_recycling_provenance = prepared.committed;
}

void commit_prepared_storm_tracking_domain_recycling(
    void* opaque_prepared) noexcept
{
    auto& prepared = *static_cast<
        PreparedStormTrackingDomainRecyclingState*>(opaque_prepared);
    SimulationRuntimeState& runtime = *prepared.runtime;
    runtime.storm_tracking = prepared.committed_continuity;
    runtime.storm_window_movement = prepared.committed_movement;
    std::copy(prepared.committed_partition.begin(),
              prepared.committed_partition.end(),
              runtime.storm_selected_partition.begin());
    runtime.storm_selected_partition_available = true;
    runtime.storm_selected_partition_nx =
        static_cast<std::size_t>(
            prepared.physical_i_end - prepared.physical_i_begin);
    runtime.storm_selected_partition_ny =
        static_cast<std::size_t>(
            prepared.physical_j_end - prepared.physical_j_begin);
}

} // namespace


PreparedDomainRecyclingTransactionProvenance
prepare_domain_recycling_transaction_provenance(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HorizontalDomainFrame& pre_shift_frame,
    const DomainRecyclingControlVolumeExchangeLedger& exchange,
    std::string_view transaction_id,
    std::string_view publication_generation,
    std::string_view storage_format,
    std::string_view swath_algorithm,
    double simulation_time_s,
    std::uint64_t transaction_sequence,
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64)
{
    if (state.runtime_owner == nullptr || state.configuration == nullptr)
        throw std::invalid_argument(
            "D5 provenance requires bound aggregate runtime and configuration");
    if (simulation_time_s != state.time_s || !std::isfinite(simulation_time_s)
        || simulation_time_s < 0.0)
        throw std::invalid_argument(
            "D5 provenance time must equal the live finite simulation time");
    pre_shift_frame.validate();
    const auto& live_frame = state.configuration->domain_frame;
    live_frame.validate();
    if (pre_shift_frame.earth_origin_at_t0_m.x_m
            != live_frame.earth_origin_at_t0_m.x_m
        || pre_shift_frame.earth_origin_at_t0_m.y_m
            != live_frame.earth_origin_at_t0_m.y_m
        || pre_shift_frame.velocity_ms.u_ms != live_frame.velocity_ms.u_ms
        || pre_shift_frame.velocity_ms.v_ms != live_frame.velocity_ms.v_ms)
        throw std::invalid_argument(
            "D5 provenance pre-shift frame does not match live state");
    if (plan.nr != state.grid.nr || plan.nth != state.grid.nth
        || plan.nz != state.grid.nz || !std::isfinite(state.grid.dr)
        || state.grid.dr <= 0.0 || !std::isfinite(state.grid.dtheta)
        || state.grid.dtheta <= 0.0)
        throw std::invalid_argument(
            "D5 provenance requires a grid-matching plan and spacing");
    const PhysicalRegionInventoryLedger& ledger = exchange.retiring;
    const auto private_fields = state.microphysics_runtime && state.microphysics_runtime->scheme
                                    ? state.microphysics_runtime->scheme->prognostic_scalars_for_transport()
                                    : MicrophysicsPrognosticScalarList{};
    for (const auto* inventory : {&ledger.private_inventory, &exchange.admitted.private_inventory})
    {
        if (!valid_domain_recycling_private_inventory(*inventory) || inventory->count != private_fields.size())
            throw std::invalid_argument("D5 provenance private inventory ownership is incomplete");
        for (std::size_t f = 0; f < private_fields.size(); ++f)
            if (!private_fields[f].name || inventory->entries[f].name.view() != private_fields[f].name)
                throw std::invalid_argument("D5 provenance private inventory names differ from the live owner");
    }
    if (ledger.physical_cells != plan.trailing_cells
        || ledger.physical_columns
               != plan.trailing_cells / static_cast<std::size_t>(plan.nz))
        throw std::invalid_argument(
            "D5 provenance ledger does not match the retiring plan");
    if (exchange.admitted.physical_cells != plan.trailing_cells
        || exchange.admitted.physical_columns != ledger.physical_columns)
        throw std::invalid_argument(
            "D5 provenance admitted ledger does not match the retiring measure");
    HorizontalDomainFrame expected_post_frame = pre_shift_frame;
    commit_whole_cell_shift_to_domain_frame(
        expected_post_frame, plan, state.grid.dr, state.grid.dtheta);
    require_close(
        exchange.retiring_fixed_earth_angular_momentum_z,
        fixed_earth_angular_momentum(ledger.momentum, state, pre_shift_frame),
        "retiring fixed-Earth angular momentum");
    require_close(
        exchange.admitted_fixed_earth_angular_momentum_z,
        fixed_earth_angular_momentum(
            exchange.admitted.momentum, state, expected_post_frame),
        "admitted fixed-Earth angular momentum");
    const auto& previous =
        state.runtime_owner->domain_recycling_provenance;
    if (!valid_domain_recycling_transaction_provenance(previous))
        throw std::invalid_argument(
            "D5 provenance cannot extend invalid transaction history");
    if (previous.available && previous.transaction_id.view() == transaction_id)
        throw std::invalid_argument(
            "D5 provenance transaction id must not be reused");
    if (previous.sequence == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("D5 provenance sequence overflow");
    const std::uint64_t expected_sequence = previous.sequence + 1;
    const std::uint64_t expected_previous_chain = previous.available
        ? previous.committed_chain_fnv1a64 : 0;
    if (transaction_sequence != expected_sequence
        || previous_chain_fnv1a64 != expected_previous_chain)
        throw std::invalid_argument(
            "D5 provenance swath sequence/chain does not extend live history");

    PreparedDomainRecyclingTransactionProvenance prepared;
    prepared.active = true;
    prepared.runtime = state.runtime_owner;
    auto& committed = prepared.committed;
    committed.available = true;
    committed.sequence = transaction_sequence;
    assign_provenance_text(
        committed.transaction_id, transaction_id, "transaction id", true);
    assign_provenance_text(committed.publication_generation,
                           publication_generation,
                           "publication generation", true);
    assign_provenance_text(
        committed.storage_format, storage_format, "storage format", true);
    committed.event_manifest_bytes = event_manifest_bytes;
    committed.event_manifest_fnv1a64 = event_manifest_fnv1a64;
    committed.previous_chain_fnv1a64 = previous_chain_fnv1a64;
    committed.simulation_time_s = simulation_time_s;
    committed.pre_shift_origin_t0_m = pre_shift_frame.earth_origin_at_t0_m;
    committed.frame_velocity_ms = pre_shift_frame.velocity_ms;
    committed.dx_m = state.grid.dr;
    committed.dy_m = state.grid.dtheta;
    committed.nr = plan.nr;
    committed.nth = plan.nth;
    committed.nz = plan.nz;
    committed.shift_i_cells = plan.shift_i_cells;
    committed.shift_j_cells = plan.shift_j_cells;
    committed.physical_i_begin = plan.physical_i_begin;
    committed.physical_i_end = plan.physical_i_end;
    committed.physical_j_begin = plan.physical_j_begin;
    committed.physical_j_end = plan.physical_j_end;
    committed.source_i_begin = plan.source_i_begin;
    committed.source_i_end = plan.source_i_end;
    committed.source_j_begin = plan.source_j_begin;
    committed.source_j_end = plan.source_j_end;
    HorizontalDomainFrame post_shift_frame = pre_shift_frame;
    const auto frame_report = commit_whole_cell_shift_to_domain_frame(
        post_shift_frame, plan, committed.dx_m, committed.dy_m);
    committed.post_shift_origin_t0_m =
        frame_report.committed_earth_origin_at_t0_m;
    committed.inventories = ledger.inventories;
    committed.momentum = ledger.momentum;
    committed.moist_mixture_energy = ledger.moist_mixture_energy;
    committed.theta_storage_residual_energy_j =
        ledger.theta_storage_residual_energy_j;
    committed.control_volume_exchange_complete = true;
    committed.private_inventory_complete = true;
    committed.retiring_private_inventory = ledger.private_inventory;
    committed.admitted_private_inventory = exchange.admitted.private_inventory;
    committed.admitted_inventories = exchange.admitted.inventories;
    committed.admitted_momentum = exchange.admitted.momentum;
    committed.admitted_moist_mixture_energy =
        exchange.admitted.moist_mixture_energy;
    committed.admitted_theta_storage_residual_energy_j =
        exchange.admitted.theta_storage_residual_energy_j;
    committed.retiring_fixed_earth_angular_momentum_z =
        exchange.retiring_fixed_earth_angular_momentum_z;
    committed.admitted_fixed_earth_angular_momentum_z =
        exchange.admitted_fixed_earth_angular_momentum_z;
    committed.physical_columns =
        static_cast<std::uint64_t>(ledger.physical_columns);
    committed.physical_cells =
        static_cast<std::uint64_t>(ledger.physical_cells);
    const bool terrain_aware = state.terrain_runtime != nullptr
        && state.terrain_runtime->config.scheme_id != "none";
    assign_provenance_text(committed.shift_algorithm,
                           terrain_aware
                               ? terrain_aware_field_shift_algorithm_id()
                               : homogeneous_field_shift_algorithm_id(),
                           "shift algorithm", false);
    assign_provenance_text(committed.frame_algorithm,
                           domain_frame_shift_algorithm_id(),
                           "frame algorithm", false);
    assign_provenance_text(committed.swath_algorithm, swath_algorithm,
                           "swath algorithm", false);
    assign_provenance_text(committed.scalar_ledger_basis,
                           trailing_scalar_export_ledger_basis_id(),
                           "scalar ledger basis", false);
    assign_provenance_text(committed.momentum_ledger_basis,
                           trailing_momentum_export_ledger_basis_id(),
                           "momentum ledger basis", false);
    assign_provenance_text(committed.energy_ledger_basis,
                           trailing_energy_export_ledger_basis_id(),
                           "energy ledger basis", false);
    assign_provenance_text(committed.theta_residual_ledger_basis,
                           trailing_theta_residual_export_ledger_basis_id(),
                           "theta residual ledger basis", false);
    assign_provenance_text(committed.control_volume_exchange_basis,
                           domain_recycling_control_volume_exchange_basis_id(),
                           "control-volume exchange basis", false);
    committed.committed_chain_fnv1a64 =
        compute_domain_recycling_transaction_chain_fnv1a64(
            committed.previous_chain_fnv1a64, committed.sequence,
            committed.event_manifest_bytes,
            committed.event_manifest_fnv1a64,
            committed.transaction_id.view(),
            committed.publication_generation.view());
    if (!valid_domain_recycling_transaction_provenance(committed))
        throw std::invalid_argument(
            "D5 provenance preparation produced an invalid record");
    return prepared;
}

bool append_domain_recycling_transaction_provenance_binding(
    const SimulationStateView& state,
    PreparedDomainRecyclingTransactionProvenance& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings)
{
    if (!prepared.active || prepared.runtime == nullptr
        || state.runtime_owner != prepared.runtime
        || !valid_domain_recycling_transaction_provenance(prepared.committed))
        throw std::invalid_argument(
            "D5 provenance binding requires matching prepared state");
    runtime_commit_bindings.push_back({
        "checkpoint.domain_recycling_transaction_provenance", &prepared,
        &commit_prepared_domain_recycling_provenance});
    return true;
}

PreparedStormTrackingDomainRecyclingState
prepare_storm_tracking_domain_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const StormWindowMovementDecision& decision)
{
    if (state.runtime_owner == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "storm-tracking recycling requires bound aggregate and terrain runtimes");
    if (!plan_matches_grid(plan, state))
        throw std::invalid_argument(
            "storm-tracking recycling requires a grid-matching shift plan");
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    if (plan.physical_i_begin != physical.i_begin
        || plan.physical_i_end != physical.i_end
        || plan.physical_j_begin != physical.j_begin
        || plan.physical_j_end != physical.j_end)
        throw std::invalid_argument(
            "storm-tracking recycling plan must match the physical scalar domain");
    const WholeCellShiftPlan canonical =
        plan_whole_cell_shift_for_physical_domain(
            plan.nr, plan.nth, plan.nz,
            plan.shift_i_cells, plan.shift_j_cells,
            physical.i_begin, physical.i_end,
            physical.j_begin, physical.j_end);
    const bool canonical_match =
        canonical.destination_i_begin == plan.destination_i_begin
        && canonical.destination_i_end == plan.destination_i_end
        && canonical.destination_j_begin == plan.destination_j_begin
        && canonical.destination_j_end == plan.destination_j_end
        && canonical.source_i_begin == plan.source_i_begin
        && canonical.source_i_end == plan.source_i_end
        && canonical.source_j_begin == plan.source_j_begin
        && canonical.source_j_end == plan.source_j_end
        && canonical.retained_cells == plan.retained_cells
        && canonical.leading_cells == plan.leading_cells
        && canonical.trailing_cells == plan.trailing_cells;
    if (!canonical_match)
        throw std::invalid_argument(
            "storm-tracking recycling requires a canonical shift plan");

    SimulationRuntimeState& runtime = *state.runtime_owner;
    const auto& continuity = runtime.storm_tracking;
    const auto& movement = runtime.storm_window_movement;
    if (!tmv::diagnostics::valid_storm_continuity_state(continuity)
        || continuity.status
               != tmv::diagnostics::StormContinuityStatus::Tracking
        || !continuity.motion_available
        || !continuity.selected_candidate_available)
        throw std::invalid_argument(
            "D5 storm tracking requires an authoritative selected track");
    if (!valid_storm_window_movement_state(movement))
        throw std::invalid_argument(
            "D5 storm-window movement authority is invalid");

    const int decision_i_direction =
        (decision.requested_i_shift_cells > 0)
        - (decision.requested_i_shift_cells < 0);
    const int decision_j_direction =
        (decision.requested_j_shift_cells > 0)
        - (decision.requested_j_shift_cells < 0);
    const bool ready_reason =
        decision.reason == StormWindowMovementReason::RequestReady ||
        decision.reason == StormWindowMovementReason::ProjectedRelaxationRisk ||
        decision.reason == StormWindowMovementReason::TrustedInteriorContainment ||
        decision.reason == StormWindowMovementReason::EmergencyContainment;
    if (decision.status != StormWindowMovementStatus::Requested
        || !ready_reason
        || decision.requested_i_shift_cells != plan.shift_i_cells
        || decision.requested_j_shift_cells != plan.shift_j_cells
        || decision.confirmation_count
               != movement.consecutive_compatible_samples
        || decision_i_direction != movement.pending_i_direction
        || decision_j_direction != movement.pending_j_direction
        || decision.sample_time_s != movement.last_sample_time_s)
        throw std::invalid_argument(
            "D5 shift plan does not match the current storm-window decision");

    const std::size_t nx = static_cast<std::size_t>(physical.nx());
    const std::size_t ny = static_cast<std::size_t>(physical.ny());
    if (!runtime.storm_selected_partition_available
        || runtime.storm_selected_partition_nx != nx
        || runtime.storm_selected_partition_ny != ny
        || nx == 0 || ny == 0
        || nx > std::numeric_limits<std::size_t>::max() / ny
        || runtime.storm_selected_partition.size() != nx * ny)
        throw std::invalid_argument(
            "D5 storm tracking requires a live physical-domain selected partition");
    std::size_t source_selected_cells = 0;
    for (std::uint8_t value : runtime.storm_selected_partition)
    {
        if (value > 1U)
            throw std::invalid_argument(
                "D5 storm selected partition contains a non-binary value");
        source_selected_cells += value;
    }
    if (source_selected_cells == 0)
        throw std::invalid_argument(
            "D5 storm selected partition is empty");

    PreparedStormTrackingDomainRecyclingState prepared;
    prepared.active = true;
    prepared.nr = plan.nr;
    prepared.nth = plan.nth;
    prepared.nz = plan.nz;
    prepared.physical_i_begin = physical.i_begin;
    prepared.physical_i_end = physical.i_end;
    prepared.physical_j_begin = physical.j_begin;
    prepared.physical_j_end = physical.j_end;
    prepared.shift_i_cells = plan.shift_i_cells;
    prepared.shift_j_cells = plan.shift_j_cells;
    prepared.displacement_x_m =
        static_cast<double>(plan.shift_i_cells) * state.grid.dr;
    prepared.displacement_y_m =
        static_cast<double>(plan.shift_j_cells) * state.grid.dtheta;
    if (!std::isfinite(prepared.displacement_x_m)
        || !std::isfinite(prepared.displacement_y_m))
        throw std::overflow_error(
            "D5 storm coordinate displacement is not finite");
    prepared.runtime = &runtime;
    prepared.committed_continuity = continuity;
    for (std::size_t index = 0;
         index < prepared.committed_continuity.sample_count; ++index)
    {
        prepared.committed_continuity.samples[index].x_m -=
            prepared.displacement_x_m;
        prepared.committed_continuity.samples[index].y_m -=
            prepared.displacement_y_m;
    }
    if (prepared.committed_continuity.association_envelope_available)
    {
        prepared.committed_continuity.association_envelope_west_m -=
            prepared.displacement_x_m;
        prepared.committed_continuity.association_envelope_east_m -=
            prepared.displacement_x_m;
        prepared.committed_continuity.association_envelope_south_m -=
            prepared.displacement_y_m;
        prepared.committed_continuity.association_envelope_north_m -=
            prepared.displacement_y_m;
    }
    if (!tmv::diagnostics::valid_storm_continuity_state(
            prepared.committed_continuity))
        throw std::overflow_error(
            "D5 translated storm continuity state is invalid");

    prepared.committed_movement = movement;
    commit_storm_window_movement(
        prepared.committed_movement, decision);

    prepared.committed_partition.assign(nx * ny, 0U);
    std::size_t retained_selected_cells = 0;
    for (std::size_t destination_i = 0; destination_i < nx;
         ++destination_i)
        for (std::size_t destination_j = 0; destination_j < ny;
             ++destination_j)
        {
            const std::int64_t source_i =
                static_cast<std::int64_t>(destination_i)
                + static_cast<std::int64_t>(plan.shift_i_cells);
            const std::int64_t source_j =
                static_cast<std::int64_t>(destination_j)
                + static_cast<std::int64_t>(plan.shift_j_cells);
            if (source_i < 0 || source_j < 0
                || source_i >= static_cast<std::int64_t>(nx)
                || source_j >= static_cast<std::int64_t>(ny))
                continue;
            const std::uint8_t value = runtime.storm_selected_partition[
                static_cast<std::size_t>(source_i) * ny
                + static_cast<std::size_t>(source_j)];
            prepared.committed_partition[destination_i * ny + destination_j] =
                value;
            retained_selected_cells += value;
        }
    if (retained_selected_cells == 0)
        throw std::invalid_argument(
            "D5 shift would retire the complete selected storm partition");
    return prepared;
}

bool append_storm_tracking_domain_recycling_binding(
    const SimulationStateView& state,
    PreparedStormTrackingDomainRecyclingState& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings)
{
    if (!prepared.active || state.runtime_owner == nullptr
        || prepared.runtime != state.runtime_owner)
        throw std::invalid_argument(
            "storm-tracking recycling preparation does not match the live runtime");
    const std::size_t nx = static_cast<std::size_t>(
        prepared.physical_i_end - prepared.physical_i_begin);
    const std::size_t ny = static_cast<std::size_t>(
        prepared.physical_j_end - prepared.physical_j_begin);
    if (prepared.nr != state.grid.nr || prepared.nth != state.grid.nth
        || prepared.nz != state.grid.nz || nx == 0 || ny == 0
        || prepared.committed_partition.size() != nx * ny
        || state.runtime_owner->storm_selected_partition.size() != nx * ny
        || !tmv::diagnostics::valid_storm_continuity_state(
            prepared.committed_continuity)
        || !valid_storm_window_movement_state(
            prepared.committed_movement))
        throw std::invalid_argument(
            "storm-tracking recycling preparation is stale or malformed");
    runtime_commit_bindings.push_back({
        "storm_tracking.coordinate_partition_authority", &prepared,
        commit_prepared_storm_tracking_domain_recycling});
    return true;
}

const char* storm_tracking_domain_recycling_algorithm_id() noexcept
{
    return "translate_grid_track_shift_partition_commit_authority_v1";
}

namespace
{


bool same_nested_relationship(const OneWayNestRelationshipV1& left, const OneWayNestRelationshipV1& right) noexcept
{
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

void commit_prepared_nested_grid_domain_recycling(void* opaque) noexcept
{
    auto& prepared = *static_cast<PreparedNestedGridDomainRecyclingState*>(opaque);
    if (prepared.mapped_rebase)
    {
        if (!commit_nested_active_terrain(*prepared.mapped_rebase, prepared.nested->active_terrain))
        {
            std::terminate();
        }
        prepared.nested->child.runtime.terrain.earth_fixed_placement.generation =
            prepared.nested->active_terrain.generation;
    }
    prepared.nested->relationship                        = prepared.committed_relationship;
    prepared.nested->cumulative_parent_recycling_shift_i = prepared.committed_recycling_shift_i;
    prepared.nested->cumulative_parent_recycling_shift_j = prepared.committed_recycling_shift_j;
}

} // namespace

StormWindowMovementEvidence with_nested_parent_recycling_bounds(const OneWayNestRelationshipV1& relationship,
                                                                StormWindowMovementEvidence     evidence)
{
    validate_one_way_nest_relationship(relationship);
    evidence.legal_shift_bounds_available = true;
    evidence.minimum_i_shift_cells        = relationship.child_origin_parent_cell_i + relationship.child_parent_span_x -
                                            (relationship.parent_cells_x - relationship.boundary_stencil_parent_cells);
    evidence.maximum_i_shift_cells =
        relationship.child_origin_parent_cell_i - relationship.boundary_stencil_parent_cells;
    evidence.minimum_j_shift_cells = relationship.child_origin_parent_cell_j + relationship.child_parent_span_y -
                                     (relationship.parent_cells_y - relationship.boundary_stencil_parent_cells);
    evidence.maximum_j_shift_cells =
        relationship.child_origin_parent_cell_j - relationship.boundary_stencil_parent_cells;
    return evidence;
}

PreparedNestedGridDomainRecyclingState
prepare_nested_grid_domain_recycling(SimulationStorageState&                           storage,
                                     const WholeCellShiftPlan&                         plan,
                                     const HorizontalDomainFrame&                      post_shift_frame,
                                     const DomainRecyclingControlVolumeExchangeLedger* exchange)
{
    if (storage.nested_execution.child_execution_active)
        throw std::invalid_argument(
            "nested D5 preparation cannot run while the child is process-bound");
    auto& nested = storage.nested_execution;
    const auto private_fields = storage.runtime.microphysics.scheme
        ? storage.runtime.microphysics.scheme->prognostic_scalars_for_transport() : MicrophysicsPrognosticScalarList{};
    const auto& feedback_private = nested.last_two_way_feedback.private_scalars;
    if (!private_fields.empty() || !feedback_private.empty())
    {
        if (!exchange)
            throw std::invalid_argument(
                "nested D5 private carriers require complete parent-recycling inventory ownership");
        const auto live = live_private_inventory(make_simulation_state_view(storage));
        if (!feedback_private.empty() && live.count != feedback_private.size())
            throw std::invalid_argument("nested D5 private feedback carrier layout differs");
        for (const auto* inventory : {&exchange->pre_shift.private_inventory,
                                      &exchange->retiring.private_inventory,
                                      &exchange->admitted.private_inventory,
                                      &exchange->expected_post_shift.private_inventory})
        {
            if (!valid_domain_recycling_private_inventory(*inventory) || inventory->count != live.count)
                throw std::invalid_argument("nested D5 private exchange carrier layout differs");
            for (std::size_t f = 0; f < live.count; ++f)
                if (inventory->entries[f].name.view() != live.entries[f].name.view() ||
                    (!feedback_private.empty() && inventory->entries[f].name.view() != feedback_private[f].name))
                    throw std::invalid_argument("nested D5 private exchange carrier names differ");
        }
        for (std::size_t f = 0; f < live.count; ++f)
            require_close(live.entries[f].inventory,
                          exchange->pre_shift.private_inventory.entries[f].inventory,
                          live.entries[f].name.bytes.data());
    }

    if (!nested_two_way_feedback_interval_closed(storage))
        throw std::invalid_argument("nested D5 requires fully consumed two-way feedback registers");
    if (!nested.relationship_ready || !nested.reference_state_ready
        || !nested.physical_primary_state_ready
        || !nested.parent_boundary_state_ready
        || nested.parent_time_bracket.completed_child_substeps
            != nested.relationship.parent_time_step_ratio
        || nested.child.time_s != storage.time_s)
        throw std::invalid_argument(
            "nested D5 requires a complete synchronized child endpoint");
    if (nested.relationship.version != 1
        || nested.parent_time_bracket.version != 1
        || !nested.parent_time_bracket.available)
        throw std::invalid_argument(
            "nested D5 relationship or time bracket is invalid");
    const PhysicalCellDomain parent_physical =
        physical_scalar_domain(
            storage.grid.nr, storage.grid.nth, storage.grid.nz,
            storage.grid.geometry.staggered
                ? StaggerType::CGrid : StaggerType::Collocated);
    if (nested.relationship.parent_cells_x != parent_physical.nx()
        || nested.relationship.parent_cells_y != parent_physical.ny()
        || plan.physical_i_begin != parent_physical.i_begin
        || plan.physical_i_end != parent_physical.i_end
        || plan.physical_j_begin != parent_physical.j_begin
        || plan.physical_j_end != parent_physical.j_end
        || nested.relationship.parent_dx_m != storage.grid.dx
        || nested.relationship.parent_dy_m != storage.grid.dy)
        throw std::invalid_argument(
            "nested D5 relationship does not match the parent shift grid");

    const auto next_origin_i = static_cast<std::int64_t>(
        nested.relationship.child_origin_parent_cell_i)
        - static_cast<std::int64_t>(plan.shift_i_cells);
    const auto next_origin_j = static_cast<std::int64_t>(
        nested.relationship.child_origin_parent_cell_j)
        - static_cast<std::int64_t>(plan.shift_j_cells);
    const auto next_recycling_i = static_cast<std::int64_t>(
        nested.cumulative_parent_recycling_shift_i)
        + static_cast<std::int64_t>(plan.shift_i_cells);
    const auto next_recycling_j = static_cast<std::int64_t>(
        nested.cumulative_parent_recycling_shift_j)
        + static_cast<std::int64_t>(plan.shift_j_cells);
    const auto in_int = [](std::int64_t value) {
        return value >= std::numeric_limits<int>::min()
            && value <= std::numeric_limits<int>::max();
    };
    if (!in_int(next_origin_i) || !in_int(next_origin_j)
        || !in_int(next_recycling_i) || !in_int(next_recycling_j))
        throw std::overflow_error(
            "nested D5 placement provenance overflows integer parent cells");

    const auto& relationship = nested.relationship;
    const auto end_i = next_origin_i + relationship.child_parent_span_x;
    const auto end_j = next_origin_j + relationship.child_parent_span_y;
    if (next_origin_i < relationship.boundary_stencil_parent_cells
        || next_origin_j < relationship.boundary_stencil_parent_cells
        || end_i > relationship.parent_cells_x
                - relationship.boundary_stencil_parent_cells
        || end_j > relationship.parent_cells_y
                - relationship.boundary_stencil_parent_cells)
        throw std::invalid_argument(
            "nested D5 rebase would lose the parent stencil margin");
    auto committed_relationship = relationship;
    committed_relationship.child_origin_parent_cell_i =
        static_cast<int>(next_origin_i);
    committed_relationship.child_origin_parent_cell_j =
        static_cast<int>(next_origin_j);

    post_shift_frame.validate();
    const PhysicalCellDomain child_physical =
        physical_scalar_domain(
            nested.child.grid.nr, nested.child.grid.nth,
            nested.child.grid.nz,
            nested.child.grid.geometry.staggered
                ? StaggerType::CGrid : StaggerType::Collocated);
    const double expected_child_x =
        post_shift_frame.earth_origin_at_t0_m.x_m
        + (parent_physical.i_begin
           + committed_relationship.child_origin_parent_cell_i)
            * committed_relationship.parent_dx_m
        - child_physical.i_begin * committed_relationship.child_dx_m;
    const double expected_child_y =
        post_shift_frame.earth_origin_at_t0_m.y_m
        + (parent_physical.j_begin
           + committed_relationship.child_origin_parent_cell_j)
            * committed_relationship.parent_dy_m
        - child_physical.j_begin * committed_relationship.child_dy_m;
    if (nested.child.domain_frame.earth_origin_at_t0_m.x_m
            != expected_child_x
        || nested.child.domain_frame.earth_origin_at_t0_m.y_m
            != expected_child_y
        || nested.child.domain_frame.velocity_ms.u_ms
            != post_shift_frame.velocity_ms.u_ms
        || nested.child.domain_frame.velocity_ms.v_ms
            != post_shift_frame.velocity_ms.v_ms)
        throw std::invalid_argument(
            "nested D5 rebase would change the child Earth frame");

    PreparedNestedGridDomainRecyclingState prepared;
    prepared.active = true;
    prepared.nested = &nested;
    prepared.expected_relationship = nested.relationship;
    prepared.committed_relationship = committed_relationship;
    prepared.expected_recycling_shift_i =
        nested.cumulative_parent_recycling_shift_i;
    prepared.expected_recycling_shift_j =
        nested.cumulative_parent_recycling_shift_j;
    prepared.committed_recycling_shift_i = static_cast<int>(next_recycling_i);
    prepared.committed_recycling_shift_j = static_cast<int>(next_recycling_j);
    prepared.expected_parent_time_s      = storage.time_s;
    prepared.expected_child_time_s       = nested.child.time_s;
    if (nested.active_terrain.ready)
    {
        const auto origin      = post_shift_frame.earth_origin_at(storage.time_s);
        prepared.mapped_rebase = std::make_unique<PreparedNestedActiveTerrainCommitV1>(
            prepare_nested_terrain_parent_rebase(nested.active_terrain,
                                                 committed_relationship,
                                                 {origin.x_m + parent_physical.i_begin * storage.grid.dx,
                                                  origin.y_m + parent_physical.j_begin * storage.grid.dy}));
        prepared.expected_child_terrain_generation = nested.child.runtime.terrain.earth_fixed_placement.generation;
    }
    prepared.preserved_state_fingerprint = nested_preserved_state_fingerprint(nested);
    prepared.scheme_identity             = nested_scheme_identity(nested);
    return prepared;
}

bool append_nested_grid_domain_recycling_binding(
    SimulationStorageState& storage,
    PreparedNestedGridDomainRecyclingState& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings)
{
    if (!can_commit_nested_grid_domain_recycling(storage, prepared))
        throw std::invalid_argument(
            "nested D5 prepared owner is stale or incomplete");
    runtime_commit_bindings.push_back({
        "nested_grid.parent_frame_rebase", &prepared,
        commit_prepared_nested_grid_domain_recycling});
    return true;
}

bool can_commit_nested_grid_domain_recycling(
    const SimulationStorageState& storage,
    const PreparedNestedGridDomainRecyclingState& prepared) noexcept
{
    return prepared.active && prepared.nested != nullptr && nested_two_way_feedback_interval_closed(storage) &&
           prepared.nested == &storage.nested_execution &&
           same_nested_relationship(prepared.nested->relationship, prepared.expected_relationship) &&
           prepared.nested->cumulative_parent_recycling_shift_i == prepared.expected_recycling_shift_i &&
           prepared.nested->cumulative_parent_recycling_shift_j == prepared.expected_recycling_shift_j &&
           storage.time_s == prepared.expected_parent_time_s &&
           prepared.nested->child.time_s == prepared.expected_child_time_s &&
           nested_preserved_state_fingerprint(*prepared.nested) == prepared.preserved_state_fingerprint &&
           nested_scheme_identity(*prepared.nested) == prepared.scheme_identity &&
           (!prepared.mapped_rebase ||
            (can_commit_nested_active_terrain(*prepared.mapped_rebase, prepared.nested->active_terrain) &&
             prepared.nested->child.runtime.terrain.earth_fixed_placement.generation ==
                 prepared.expected_child_terrain_generation));
}

const char* nested_grid_domain_recycling_algorithm_id() noexcept
{
    return "preserve_child_earth_frame_rebase_parent_indices_v1";
}

PreparedRadiationDomainRecyclingState
prepare_flat_homogeneous_radiation_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& primary_profiles)
{
    if (!plan_matches_grid(plan, state))
        throw std::invalid_argument(
            "radiation recycling requires a grid-matching shift plan");
    const WholeCellShiftPlan scalar_plan =
        plan_whole_cell_shift_for_horizontal_location(
            plan, HorizontalFieldLocation::ScalarCenter);
    if (scalar_plan.destination_i_begin != plan.destination_i_begin
        || scalar_plan.destination_i_end != plan.destination_i_end
        || scalar_plan.destination_j_begin != plan.destination_j_begin
        || scalar_plan.destination_j_end != plan.destination_j_end)
        throw std::invalid_argument(
            "radiation recycling requires the canonical scalar shift plan");
    if (state.radiation_runtime == nullptr
        || state.boundary_layer_runtime == nullptr
        || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "radiation recycling requires bound radiation, surface, and terrain runtimes");

    PreparedRadiationDomainRecyclingState prepared;
    prepared.nr = plan.nr;
    prepared.nth = plan.nth;
    prepared.nz = plan.nz;
    prepared.shift_i_cells = plan.shift_i_cells;
    prepared.shift_j_cells = plan.shift_j_cells;
    RadiationRuntimeState& runtime = *state.radiation_runtime;
    if (runtime.config.scheme_id == "none")
        return prepared;
    if (state.terrain_runtime->config.scheme_id != "none")
        throw std::invalid_argument(
            "D5 radiation recycling currently requires flat terrain");
    if (!runtime.scheme || runtime.scheme->name() == "none"
        || runtime.scheme->name() != runtime.config.scheme_id)
        throw std::invalid_argument(
            "active radiation recycling requires a matching initialized scheme");
    if (!std::isfinite(runtime.last_update_time_s)
        || runtime.last_update_time_s < 0.0
        || !std::isfinite(runtime.next_update_time_s)
        || runtime.next_update_time_s <= runtime.last_update_time_s)
        throw std::invalid_argument(
            "active radiation recycling requires a sampled held cadence");
    require_grid_field(
        state.tendencies.dtheta_dt_rad, state,
        "radiation held theta tendency");
    require_grid_field(
        runtime.held_cell_power_w, state,
        "radiation held cell power");

    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    if (plan.physical_i_begin != physical.i_begin
        || plan.physical_i_end != physical.i_end
        || plan.physical_j_begin != physical.j_begin
        || plan.physical_j_end != physical.j_end)
        throw std::invalid_argument(
            "radiation recycling plan must match the physical scalar domain");
    double existing_power_w = 0.0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
            for (int k = physical.k_begin; k < physical.k_end; ++k)
            {
                const float power = runtime.held_cell_power_w[i][j][k];
                const float tendency = state.tendencies.dtheta_dt_rad[i][j][k];
                if (!std::isfinite(power) || !std::isfinite(tendency))
                    throw std::invalid_argument(
                        "radiation held fields contain a non-finite value");
                existing_power_w += static_cast<double>(power);
            }
    const double existing_scale = std::max(
        1.0, std::max(std::abs(existing_power_w),
                      std::abs(runtime.held_atmospheric_power_w)));
    if (!std::isfinite(runtime.held_atmospheric_power_w)
        || std::abs(existing_power_w - runtime.held_atmospheric_power_w)
               > 1.0e-12 * existing_scale)
        throw std::invalid_argument(
            "radiation held cell power does not close its aggregate ledger");

    const auto& pressure =
        primary_profiles[HomogeneousPrimaryOwner::p];
    const auto& theta =
        primary_profiles[HomogeneousPrimaryOwner::theta];
    const auto& density =
        primary_profiles[HomogeneousPrimaryOwner::rho];
    const auto& vapor =
        primary_profiles[HomogeneousPrimaryOwner::qv];
    const auto& cloud_liquid =
        primary_profiles[HomogeneousPrimaryOwner::qc];
    const auto& cloud_ice =
        primary_profiles[HomogeneousPrimaryOwner::qi];
    const std::size_t nz = static_cast<std::size_t>(state.grid.nz);
    if (pressure.size() != nz || theta.size() != nz || density.size() != nz
        || vapor.size() != nz || cloud_liquid.size() != nz
        || cloud_ice.size() != nz)
        throw std::invalid_argument(
            "radiation leading state requires complete thermodynamic/moisture profiles");

    if ((!state.grid.geometry.z.empty()
         && state.grid.geometry.z.size() != nz)
        || (!state.grid.geometry.z_faces.empty()
            && state.grid.geometry.z_faces.size() != nz + 1))
        throw std::invalid_argument(
            "radiation recycling flat geometry shape does not match nz");
    RadiationColumnWorkspace workspace;
    workspace.p.resize(nz);
    workspace.T.resize(nz);
    workspace.rho.resize(nz);
    workspace.qv.resize(nz);
    workspace.qc.resize(nz);
    workspace.qi.resize(nz);
    workspace.z_center.resize(nz);
    workspace.z_interface.resize(nz + 1);
    workspace.layer_thickness.resize(nz);
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::specific_heat_cp_jkgk;
    for (std::size_t k = 0; k < nz; ++k)
    {
        workspace.p[k] = static_cast<double>(pressure[k]);
        workspace.rho[k] = static_cast<double>(density[k]);
        workspace.qv[k] = static_cast<double>(vapor[k]);
        workspace.qc[k] = static_cast<double>(cloud_liquid[k]);
        workspace.qi[k] = static_cast<double>(cloud_ice[k]);
        workspace.T[k] = static_cast<double>(theta[k]) * std::pow(
            workspace.p[k] /
                physical_constants::reference_pressure_pa,
            kappa);
        workspace.z_center[k] = state.grid.geometry.z.empty()
            ? (static_cast<double>(k) + 0.5) * state.grid.dz
            : state.grid.geometry.z[k];
        workspace.z_interface[k] = state.grid.geometry.z_faces.empty()
            ? static_cast<double>(k) * state.grid.dz
            : state.grid.geometry.z_faces[k];
    }
    workspace.z_interface[nz] = state.grid.geometry.z_faces.empty()
        ? static_cast<double>(nz) * state.grid.dz
        : state.grid.geometry.z_faces[nz];
    for (std::size_t k = 0; k < nz; ++k)
        workspace.layer_thickness[k] =
            workspace.z_interface[k + 1] - workspace.z_interface[k];

    std::unique_ptr<RadiationSchemeBase> scheme = runtime.scheme->clone();
    if (!scheme)
        throw std::runtime_error(
            "radiation scheme does not support recycling preparation snapshots");
    RadiationPreparedColumnView column;
    column.pressure_pa = &workspace.p;
    column.temperature_k = &workspace.T;
    column.density_kgm3 = &workspace.rho;
    column.vapor_mixing_ratio = &workspace.qv;
    column.cloud_liquid_mixing_ratio = &workspace.qc;
    column.cloud_ice_mixing_ratio = &workspace.qi;
    column.z_center_m = &workspace.z_center;
    column.z_interface_m = &workspace.z_interface;
    column.layer_thickness_m = &workspace.layer_thickness;
    column.cosine_solar_zenith = idealized_solar_cosine_zenith(
        runtime.config, runtime.last_update_time_s);
    column.surface_temperature_k =
        state.boundary_layer_runtime->surface_config.Tsfc;
    column.surface_albedo =
        state.boundary_layer_runtime->surface_config.albedo;
    column.surface_emissivity =
        state.boundary_layer_runtime->surface_config.emissivity;
    column.horizontal_cell_area_m2 = state.grid.dr * state.grid.dtheta;
    RadiationColumnTendencies tendencies;
    (void)evaluate_prepared_radiation_column(
        *scheme, runtime.config, column, workspace, tendencies);

    prepared.active = true;
    prepared.sample_time_s = runtime.last_update_time_s;
    prepared.leading_theta_tendency_s = workspace.theta_tendency;
    prepared.leading_cell_power_w = workspace.cell_power_w;

    double committed_power_w = 0.0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
        {
            const bool retained = i >= plan.destination_i_begin
                && i < plan.destination_i_end
                && j >= plan.destination_j_begin
                && j < plan.destination_j_end;
            for (int k = physical.k_begin; k < physical.k_end; ++k)
                committed_power_w += retained
                    ? static_cast<double>(runtime.held_cell_power_w[
                          i + plan.shift_i_cells]
                          [j + plan.shift_j_cells][k])
                    : static_cast<double>(
                          prepared.leading_cell_power_w[
                              static_cast<std::size_t>(k)]);
        }
    if (!std::isfinite(committed_power_w))
        throw std::overflow_error(
            "radiation recycled aggregate power overflowed");
    prepared.committed_held_atmospheric_power_w = committed_power_w;
    return prepared;
}

PreparedRadiationDomainRecyclingState
prepare_terrain_aware_radiation_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& primary_fields,
    const TerrainMetrics3D& post_shift_metrics)
{
    if (!plan_matches_grid(plan, state) || state.radiation_runtime == nullptr
        || state.boundary_layer_runtime == nullptr
        || state.terrain_runtime == nullptr
        || state.terrain_runtime->config.horizontal_placement_id
            != "earth_fixed")
        throw std::invalid_argument(
            "terrain radiation recycling requires an Earth-fixed runtime owner");
    auto& runtime = *state.radiation_runtime;
    if (runtime.config.scheme_id == "none")
        return {};
    if (!runtime.scheme || runtime.scheme->name() == "none"
        || runtime.scheme->name() != runtime.config.scheme_id)
        throw std::invalid_argument(
            "terrain radiation recycling requires matching scheme ownership");
    if (!std::isfinite(runtime.last_update_time_s)
        || runtime.last_update_time_s < 0.0
        || !std::isfinite(runtime.next_update_time_s)
        || runtime.next_update_time_s <= runtime.last_update_time_s)
        throw std::invalid_argument(
            "terrain radiation recycling requires sampled held state");
    require_grid_field(
        state.tendencies.dtheta_dt_rad, state,
        "terrain radiation held theta tendency");
    require_grid_field(
        runtime.held_cell_power_w, state,
        "terrain radiation held cell power");
    for (const auto& field : primary_fields.values)
        require_grid_field(field, state, "terrain radiation leading primary");
    if (post_shift_metrics.z.size_r() != state.grid.nr
        || post_shift_metrics.z.size_th() != state.grid.nth
        || post_shift_metrics.z.size_z() != state.grid.nz
        || post_shift_metrics.z_face.size_z() != state.grid.nz + 1)
        throw std::invalid_argument(
            "terrain radiation post-shift geometry shape mismatch");

    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    double existing_power_w = 0.0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
            for (int k = physical.k_begin; k < physical.k_end; ++k)
            {
                const float power = runtime.held_cell_power_w(i, j, k);
                const float tendency = state.tendencies.dtheta_dt_rad(i, j, k);
                if (!std::isfinite(power) || !std::isfinite(tendency))
                    throw std::invalid_argument(
                        "terrain radiation held fields are non-finite");
                existing_power_w += power;
            }
    const double scale = std::max(
        {1.0, std::abs(existing_power_w),
         std::abs(runtime.held_atmospheric_power_w)});
    if (!std::isfinite(runtime.held_atmospheric_power_w)
        || std::abs(existing_power_w - runtime.held_atmospheric_power_w)
            > 1.0e-12 * scale)
        throw std::invalid_argument(
            "terrain radiation held power does not close its aggregate");

    PreparedRadiationDomainRecyclingState prepared;
    prepared.active = true;
    prepared.terrain_aware = true;
    prepared.nr = plan.nr;
    prepared.nth = plan.nth;
    prepared.nz = plan.nz;
    prepared.shift_i_cells = plan.shift_i_cells;
    prepared.shift_j_cells = plan.shift_j_cells;
    prepared.sample_time_s = runtime.last_update_time_s;
    prepared.terrain_leading_theta_tendency_s.resize(
        plan.nr, plan.nth, plan.nz, 0.0f);
    prepared.terrain_leading_cell_power_w.resize(
        plan.nr, plan.nth, plan.nz, 0.0f);

    std::unique_ptr<RadiationSchemeBase> scheme = runtime.scheme->clone();
    if (!scheme)
        throw std::runtime_error(
            "terrain radiation scheme lacks detached preparation");
    const std::size_t nz = static_cast<std::size_t>(plan.nz);
    RadiationColumnWorkspace workspace;
    workspace.p.resize(nz);
    workspace.T.resize(nz);
    workspace.rho.resize(nz);
    workspace.qv.resize(nz);
    workspace.qc.resize(nz);
    workspace.qi.resize(nz);
    workspace.z_center.resize(nz);
    workspace.z_interface.resize(nz + 1);
    workspace.layer_thickness.resize(nz);
    constexpr double kappa =
        physical_constants::gas_constant_dry_air_jkgk
        / physical_constants::specific_heat_cp_jkgk;
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            for (std::size_t kk = 0; kk < nz; ++kk)
            {
                const int k = static_cast<int>(kk);
                workspace.p[kk] = primary_fields[HomogeneousPrimaryOwner::p](i, j, k);
                workspace.rho[kk] = primary_fields[HomogeneousPrimaryOwner::rho](i, j, k);
                workspace.qv[kk] = primary_fields[HomogeneousPrimaryOwner::qv](i, j, k);
                workspace.qc[kk] = primary_fields[HomogeneousPrimaryOwner::qc](i, j, k);
                workspace.qi[kk] = primary_fields[HomogeneousPrimaryOwner::qi](i, j, k);
                workspace.T[kk] =
                    primary_fields[HomogeneousPrimaryOwner::theta](i, j, k)
                    * std::pow(
                        workspace.p[kk]
                            / physical_constants::reference_pressure_pa,
                        kappa);
                workspace.z_center[kk] = post_shift_metrics.z(i, j, k);
                workspace.z_interface[kk] =
                    post_shift_metrics.z_face(i, j, k);
                workspace.layer_thickness[kk] =
                    post_shift_metrics.z_face(i, j, k + 1)
                    - post_shift_metrics.z_face(i, j, k);
            }
            workspace.z_interface[nz] =
                post_shift_metrics.z_face(i, j, plan.nz);
            RadiationPreparedColumnView column;
            column.pressure_pa = &workspace.p;
            column.temperature_k = &workspace.T;
            column.density_kgm3 = &workspace.rho;
            column.vapor_mixing_ratio = &workspace.qv;
            column.cloud_liquid_mixing_ratio = &workspace.qc;
            column.cloud_ice_mixing_ratio = &workspace.qi;
            column.z_center_m = &workspace.z_center;
            column.z_interface_m = &workspace.z_interface;
            column.layer_thickness_m = &workspace.layer_thickness;
            column.cosine_solar_zenith = idealized_solar_cosine_zenith(
                runtime.config, runtime.last_update_time_s);
            column.surface_temperature_k =
                state.boundary_layer_runtime->surface_config.Tsfc;
            column.surface_albedo =
                state.boundary_layer_runtime->surface_config.albedo;
            column.surface_emissivity =
                state.boundary_layer_runtime->surface_config.emissivity;
            column.horizontal_cell_area_m2 =
                state.grid.dr * state.grid.dtheta;
            RadiationColumnTendencies tendencies;
            (void)evaluate_prepared_radiation_column(
                *scheme, runtime.config, column, workspace, tendencies);
            std::copy(
                workspace.theta_tendency.begin(),
                workspace.theta_tendency.end(),
                prepared.terrain_leading_theta_tendency_s.row_ptr(i, j));
            std::copy(
                workspace.cell_power_w.begin(), workspace.cell_power_w.end(),
                prepared.terrain_leading_cell_power_w.row_ptr(i, j));
        }

    double committed_power_w = 0.0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
        {
            const bool retained = i >= plan.destination_i_begin
                && i < plan.destination_i_end
                && j >= plan.destination_j_begin
                && j < plan.destination_j_end;
            for (int k = physical.k_begin; k < physical.k_end; ++k)
                committed_power_w += retained
                    ? runtime.held_cell_power_w(
                        i + plan.shift_i_cells,
                        j + plan.shift_j_cells, k)
                    : prepared.terrain_leading_cell_power_w(i, j, k);
        }
    if (!std::isfinite(committed_power_w))
        throw std::overflow_error(
            "terrain radiation recycled aggregate overflowed");
    prepared.committed_held_atmospheric_power_w = committed_power_w;
    return prepared;
}

bool append_radiation_domain_recycling_bindings(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const PreparedRadiationDomainRecyclingState& prepared,
    std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    std::vector<HomogeneousScalarCommitBinding>& scalar_bindings)
{
    if (!prepared.active)
        return false;
    if (state.radiation_runtime == nullptr || !plan_matches_grid(plan, state)
        || prepared.nr != state.grid.nr || prepared.nth != state.grid.nth
        || prepared.nz != state.grid.nz
        || prepared.shift_i_cells != plan.shift_i_cells
        || prepared.shift_j_cells != plan.shift_j_cells
        || !std::isfinite(prepared.committed_held_atmospheric_power_w))
        throw std::invalid_argument(
            "prepared radiation recycling state does not match the live grid");
    if (prepared.terrain_aware)
    {
        require_grid_field(
            prepared.terrain_leading_theta_tendency_s, state,
            "terrain radiation leading tendency");
        require_grid_field(
            prepared.terrain_leading_cell_power_w, state,
            "terrain radiation leading power");
    }
    else if (prepared.leading_theta_tendency_s.size()
                 != static_cast<std::size_t>(state.grid.nz)
             || prepared.leading_cell_power_w.size()
                 != static_cast<std::size_t>(state.grid.nz))
        throw std::invalid_argument(
            "prepared radiation leading profile does not match nz");
    RadiationRuntimeState& runtime = *state.radiation_runtime;
    require_grid_field(
        state.tendencies.dtheta_dt_rad, state,
        "radiation held theta tendency");
    require_grid_field(
        runtime.held_cell_power_w, state,
        "radiation held cell power");
    volume_bindings.push_back({
        "radiation.dtheta_dt_rad", &state.tendencies.dtheta_dt_rad,
        prepared.terrain_aware ? nullptr : &prepared.leading_theta_tendency_s,
        HorizontalFieldLocation::ScalarCenter,
        prepared.terrain_aware
            ? &prepared.terrain_leading_theta_tendency_s : nullptr,
    });
    volume_bindings.push_back({
        "radiation.held_cell_power_w", &runtime.held_cell_power_w,
        prepared.terrain_aware ? nullptr : &prepared.leading_cell_power_w,
        HorizontalFieldLocation::ScalarCenter,
        prepared.terrain_aware
            ? &prepared.terrain_leading_cell_power_w : nullptr,
    });
    scalar_bindings.push_back({
        "radiation.held_atmospheric_power_w",
        &runtime.held_atmospheric_power_w,
        prepared.committed_held_atmospheric_power_w,
    });
    return true;
}

bool append_radiation_trailing_export_bindings(
    const SimulationStateView& state,
    std::vector<float>& theta_tendency_values,
    std::vector<float>& cell_power_values,
    std::vector<TrailingFieldExportBinding>& bindings)
{
    if (state.radiation_runtime == nullptr)
        throw std::invalid_argument(
            "radiation trailing binding requires a bound runtime");
    RadiationRuntimeState& runtime = *state.radiation_runtime;
    if (runtime.config.scheme_id == "none")
        return false;
    require_grid_field(
        state.tendencies.dtheta_dt_rad, state,
        "radiation held theta tendency");
    require_grid_field(
        runtime.held_cell_power_w, state,
        "radiation held cell power");
    bindings.push_back({
        "radiation.dtheta_dt_rad", &state.tendencies.dtheta_dt_rad,
        &theta_tendency_values, HorizontalFieldLocation::ScalarCenter,
    });
    bindings.push_back({
        "radiation.held_cell_power_w", &runtime.held_cell_power_w,
        &cell_power_values, HorizontalFieldLocation::ScalarCenter,
    });
    return true;
}

const char* radiation_domain_recycling_algorithm_id() noexcept
{
    return "held_sample_time_flat_or_mapped_radiation_shift_v2";
}

PreparedBoundaryLayerDomainRecyclingState
prepare_flat_homogeneous_boundary_layer_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& primary_profiles)
{
    if (!plan_matches_grid(plan, state))
        throw std::invalid_argument(
            "boundary-layer recycling requires a grid-matching shift plan");
    (void)plan_whole_cell_shift_for_horizontal_location(
        plan, HorizontalFieldLocation::ScalarCenter);
    if (!state.grid.geometry.staggered)
        throw std::invalid_argument(
            "D5 boundary-layer recycling currently requires the C-grid");
    if (state.boundary_layer_runtime == nullptr
        || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "boundary-layer recycling requires bound module runtimes");

    PreparedBoundaryLayerDomainRecyclingState prepared;
    prepared.nr = plan.nr;
    prepared.nth = plan.nth;
    prepared.nz = plan.nz;
    prepared.shift_i_cells = plan.shift_i_cells;
    prepared.shift_j_cells = plan.shift_j_cells;
    BoundaryLayerRuntimeState& runtime = *state.boundary_layer_runtime;
    if (runtime.config.scheme_id == "none")
        return prepared;
    if (state.terrain_runtime->config.scheme_id != "none")
        throw std::invalid_argument(
            "D5 boundary-layer recycling currently requires flat terrain");
    if (!runtime.scheme || runtime.scheme->name() == "none"
        || runtime.scheme->name() != runtime.config.scheme_id)
        throw std::invalid_argument(
            "active boundary-layer recycling requires a matching initialized scheme");
    if (runtime.scheme->name() != "slab"
        && runtime.scheme->name() != "ysu"
        && runtime.scheme->name() != "mynn")
        throw std::invalid_argument(
            "boundary-layer scheme has no D5 leading-column time contract");
    if (!std::isfinite(runtime.last_update_time_s)
        || runtime.last_update_time_s < 0.0
        || !std::isfinite(runtime.next_update_time_s)
        || runtime.next_update_time_s <= runtime.last_update_time_s)
        throw std::invalid_argument(
            "active boundary-layer recycling requires sampled held state");
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    if (plan.physical_i_begin != physical.i_begin
        || plan.physical_i_end != physical.i_end
        || plan.physical_j_begin != physical.j_begin
        || plan.physical_j_end != physical.j_end)
        throw std::invalid_argument(
            "boundary-layer recycling plan must match the physical scalar domain");

    const std::size_t surface_size = static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    if (!runtime.skin_captured || runtime.skin_nr != state.grid.nr
        || runtime.skin_nth != state.grid.nth
        || runtime.skin_theta_2d.size() != surface_size
        || runtime.skin_qv_2d.size() != surface_size
        || runtime.slab_nr != state.grid.nr
        || runtime.slab_nth != state.grid.nth
        || runtime.slab_columns.size() != surface_size)
        throw std::invalid_argument(
            "active boundary-layer recycling requires complete skin and column memory");
    const Field3D* const persistent_fields[] = {
        &runtime.held_u_force_n,
        &runtime.held_v_force_n,
        &runtime.held_theta_inventory_rate_kgk_s,
        &runtime.held_vapor_mass_rate_kg_s,
        &state.tendencies.dtke_dt_pbl,
    };
    for (const Field3D* field : persistent_fields)
    {
        require_grid_field(*field, state, "boundary-layer persistent field");
        for (std::size_t index = 0; index < field->size(); ++index)
            if (!std::isfinite(field->data()[index]))
                throw std::invalid_argument(
                    "boundary-layer persistent field contains a non-finite value");
    }
    const std::vector<float>* const surface_fields[] = {
        &runtime.skin_theta_2d, &runtime.skin_qv_2d,
        state.surface.sfc_ustar, state.surface.sfc_tau_u,
        state.surface.sfc_tau_v, state.surface.sfc_sensible_heat_flux,
        state.surface.sfc_latent_heat_flux,
        state.surface.sfc_moisture_flux, state.surface.pbl_height_2d,
    };
    for (const std::vector<float>* field : surface_fields)
    {
        if (field == nullptr || field->size() != surface_size)
            throw std::invalid_argument(
                "boundary-layer surface state shape does not match the recycling grid");
        for (float value : *field)
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "boundary-layer surface state contains a non-finite value");
    }
    for (const SlabColumnState& column : runtime.slab_columns)
        if (column.initialized
            && (!std::isfinite(column.h_m) || column.h_m <= 0.0
                || !std::isfinite(column.theta_m_k)
                || column.theta_m_k <= 0.0
                || !std::isfinite(column.qv_m_kgkg)
                || column.qv_m_kgkg < 0.0))
            throw std::invalid_argument(
                "boundary-layer column memory contains an unphysical value");

    const auto& pressure = primary_profiles[HomogeneousPrimaryOwner::p];
    const auto& theta = primary_profiles[HomogeneousPrimaryOwner::theta];
    const auto& qv = primary_profiles[HomogeneousPrimaryOwner::qv];
    const auto& u = primary_profiles[HomogeneousPrimaryOwner::u];
    const auto& v = primary_profiles[HomogeneousPrimaryOwner::v];
    const auto& rho = primary_profiles[HomogeneousPrimaryOwner::rho];
    const auto& tke = primary_profiles[HomogeneousPrimaryOwner::tke];
    const std::size_t nz = static_cast<std::size_t>(state.grid.nz);
    if (pressure.size() != nz || theta.size() != nz || qv.size() != nz
        || u.size() != nz || v.size() != nz || rho.size() != nz
        || tke.size() != nz)
        throw std::invalid_argument(
            "boundary-layer leading state requires complete environmental profiles");

    BoundaryLayerColumnWorkspace workspace;
    workspace.z_interface.resize(nz + 1);
    if (!state.grid.geometry.z_faces.empty()
        && state.grid.geometry.z_faces.size() != nz + 1)
        throw std::invalid_argument(
            "boundary-layer leading interface geometry does not match nz");
    for (std::size_t k = 0; k <= nz; ++k)
        workspace.z_interface[k] = state.grid.geometry.z_faces.empty()
            ? static_cast<double>(k) * state.grid.dz
            : state.grid.geometry.z_faces[k];
    std::vector<double> p64(nz), theta64(nz), qv64(nz), u64(nz),
        v64(nz), rho64(nz), tke64(nz);
    for (std::size_t k = 0; k < nz; ++k)
    {
        p64[k] = pressure[k];
        theta64[k] = theta[k];
        qv64[k] = qv[k];
        u64[k] = u[k];
        v64[k] = v[k];
        rho64[k] = rho[k];
        tke64[k] = tke[k];
    }
    prepared.leading_skin_theta = theta.front();
    prepared.leading_skin_qv = qv.front();
    BoundaryLayerColumnStateView column;
    column.p = &p64;
    column.theta = &theta64;
    column.qv = &qv64;
    column.u = &u64;
    column.v = &v64;
    column.rho = &rho64;
    column.tke = &tke64;
    column.z_int = &workspace.z_interface;
    column.u_sfc = u64.front();
    column.v_sfc = v64.front();
    if (runtime.surface_config.Tsfc > 0.0)
    {
        constexpr double kappa =
            physical_constants::gas_constant_dry_air_jkgk /
            physical_constants::specific_heat_cp_jkgk;
        column.theta_sfc = runtime.surface_config.Tsfc * std::pow(
            physical_constants::reference_pressure_pa / p64.front(), kappa);
    }
    else
        column.theta_sfc = prepared.leading_skin_theta;
    column.qv_sfc = runtime.surface_config.qsfc >= 0.0
        ? runtime.surface_config.qsfc
        : prepared.leading_skin_qv;
    column.z_sfc = 0.5 * (workspace.z_interface[0]
                          + workspace.z_interface[1]);
    column.slab_state = &prepared.leading_slab_column;
    column.workspace = &workspace;
    column.elapsed_time_s = 0.0;

    std::unique_ptr<BoundaryLayerSchemeBase> scheme = runtime.scheme->clone();
    if (!scheme)
        throw std::runtime_error(
            "boundary-layer scheme does not support recycling preparation snapshots");
    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
    const BoundaryLayerStoredSurfaceDiagnostics stored =
        evaluate_prepared_boundary_layer_column(
            *scheme, runtime.config, runtime.surface_config, column,
            state.grid.dr * state.grid.dtheta, tendencies, diagnostics);
    prepared.active = true;
    prepared.sample_time_s = runtime.last_update_time_s;
    prepared.leading_u_force_n = workspace.stored_u_force_n;
    prepared.leading_v_force_n = workspace.stored_v_force_n;
    prepared.leading_theta_inventory_rate_kgk_s =
        workspace.stored_theta_inventory_rate_kgk_s;
    prepared.leading_vapor_mass_rate_kg_s =
        workspace.stored_vapor_mass_rate_kg_s;
    prepared.leading_dtke_s = workspace.stored_dtke;
    prepared.leading_ustar = stored.ustar;
    prepared.leading_tau_u = stored.tau_u;
    prepared.leading_tau_v = stored.tau_v;
    prepared.leading_sensible_heat_flux = stored.sensible_heat_flux;
    prepared.leading_latent_heat_flux = stored.latent_heat_flux;
    prepared.leading_moisture_flux = stored.moisture_flux;
    prepared.leading_pbl_height = stored.pbl_height;
    return prepared;
}

PreparedBoundaryLayerDomainRecyclingState
prepare_terrain_aware_boundary_layer_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& primary_fields,
    const TerrainMetrics3D& post_shift_metrics)
{
    if (!plan_matches_grid(plan, state) || !state.grid.geometry.staggered
        || state.boundary_layer_runtime == nullptr
        || state.terrain_runtime == nullptr
        || state.terrain_runtime->config.horizontal_placement_id
            != "earth_fixed")
        throw std::invalid_argument(
            "terrain PBL recycling requires an Earth-fixed C-grid owner");
    auto& runtime = *state.boundary_layer_runtime;
    if (runtime.config.scheme_id == "none")
        return {};
    if (!runtime.scheme || runtime.scheme->name() != "slab"
        || runtime.config.scheme_id != "slab")
        throw std::invalid_argument(
            "terrain PBL recycling currently owns the active slab scheme");
    if (!std::isfinite(runtime.last_update_time_s)
        || runtime.last_update_time_s < 0.0
        || !std::isfinite(runtime.next_update_time_s)
        || runtime.next_update_time_s <= runtime.last_update_time_s)
        throw std::invalid_argument(
            "active terrain PBL recycling requires sampled held state");

    const std::size_t surface_size =
        static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    if (!runtime.skin_captured || runtime.skin_nr != state.grid.nr
        || runtime.skin_nth != state.grid.nth
        || runtime.skin_theta_2d.size() != surface_size
        || runtime.skin_qv_2d.size() != surface_size
        || runtime.slab_nr != state.grid.nr
        || runtime.slab_nth != state.grid.nth
        || runtime.slab_columns.size() != surface_size)
        throw std::invalid_argument(
            "active terrain slab recycling requires complete surface memory");
    const Field3D* const live_fields[] = {
        &runtime.held_u_force_n, &runtime.held_v_force_n,
        &runtime.held_theta_inventory_rate_kgk_s,
        &runtime.held_vapor_mass_rate_kg_s,
        &state.tendencies.dtke_dt_pbl};
    for (const Field3D* field : live_fields)
        require_grid_field(*field, state, "terrain PBL persistent field");
    for (const auto& field : primary_fields.values)
        require_grid_field(field, state, "terrain PBL leading primary");
    if (post_shift_metrics.z.size_r() != state.grid.nr
        || post_shift_metrics.z.size_th() != state.grid.nth
        || post_shift_metrics.z.size_z() != state.grid.nz
        || post_shift_metrics.z_face.size_z() != state.grid.nz + 1)
        throw std::invalid_argument(
            "terrain PBL post-shift geometry does not match the grid");

    PreparedBoundaryLayerDomainRecyclingState prepared;
    prepared.active = true;
    prepared.terrain_aware = true;
    prepared.nr = plan.nr;
    prepared.nth = plan.nth;
    prepared.nz = plan.nz;
    prepared.shift_i_cells = plan.shift_i_cells;
    prepared.shift_j_cells = plan.shift_j_cells;
    prepared.sample_time_s = runtime.last_update_time_s;
    for (auto& field : prepared.terrain_leading_fields)
        field.resize(plan.nr, plan.nth, plan.nz, 0.0f);
    for (auto& surface : prepared.terrain_leading_surfaces)
        surface.assign(surface_size, 0.0f);
    prepared.terrain_leading_slab_columns.assign(
        surface_size, SlabColumnState{});

    std::unique_ptr<BoundaryLayerSchemeBase> scheme = runtime.scheme->clone();
    if (!scheme)
        throw std::runtime_error(
            "terrain slab scheme does not support detached recycling preparation");
    BoundaryLayerColumnWorkspace workspace;
    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
    const std::size_t nz = static_cast<std::size_t>(plan.nz);
    std::vector<double> p(nz), theta(nz), qv(nz), u(nz), v(nz), rho(nz), tke(nz);
    workspace.z_center.resize(nz);
    workspace.z_interface.resize(nz + 1);
    workspace.layer_thickness.resize(nz);
    for (int i = 0; i < plan.nr; ++i)
        for (int j = 0; j < plan.nth; ++j)
        {
            const std::size_t surface_index =
                static_cast<std::size_t>(i) * static_cast<std::size_t>(plan.nth)
                + static_cast<std::size_t>(j);
            const double ground = post_shift_metrics.z_face(i, j, 0);
            for (std::size_t kk = 0; kk < nz; ++kk)
            {
                const int k = static_cast<int>(kk);
                p[kk] = primary_fields[HomogeneousPrimaryOwner::p](i, j, k);
                theta[kk] = primary_fields[HomogeneousPrimaryOwner::theta](i, j, k);
                qv[kk] = primary_fields[HomogeneousPrimaryOwner::qv](i, j, k);
                rho[kk] = primary_fields[HomogeneousPrimaryOwner::rho](i, j, k);
                tke[kk] = primary_fields[HomogeneousPrimaryOwner::tke](i, j, k);
                u[kk] = i > 0 ? 0.5 * (
                    static_cast<double>(primary_fields[HomogeneousPrimaryOwner::u](i - 1, j, k))
                    + static_cast<double>(primary_fields[HomogeneousPrimaryOwner::u](i, j, k)))
                    : primary_fields[HomogeneousPrimaryOwner::u](i, j, k);
                v[kk] = j > 0 ? 0.5 * (
                    static_cast<double>(primary_fields[HomogeneousPrimaryOwner::v](i, j - 1, k))
                    + static_cast<double>(primary_fields[HomogeneousPrimaryOwner::v](i, j, k)))
                    : primary_fields[HomogeneousPrimaryOwner::v](i, j, k);
                workspace.z_center[kk] = post_shift_metrics.z(i, j, k) - ground;
                workspace.z_interface[kk] =
                    post_shift_metrics.z_face(i, j, k) - ground;
                workspace.layer_thickness[kk] =
                    post_shift_metrics.z_face(i, j, k + 1)
                    - post_shift_metrics.z_face(i, j, k);
            }
            workspace.z_interface[nz] =
                post_shift_metrics.z_face(i, j, plan.nz) - ground;

            auto& skin_theta = prepared.terrain_leading_surfaces[0][surface_index];
            auto& skin_qv = prepared.terrain_leading_surfaces[1][surface_index];
            skin_theta = static_cast<float>(theta.front());
            skin_qv = static_cast<float>(qv.front());
            BoundaryLayerColumnStateView column;
            column.p = &p;
            column.theta = &theta;
            column.qv = &qv;
            column.u = &u;
            column.v = &v;
            column.rho = &rho;
            column.tke = &tke;
            column.z_int = &workspace.z_interface;
            const auto relative_wind = boundary_layer_surface_relative_wind(
                u.front(), v.front(), true, state.domain_frame);
            column.u_sfc = relative_wind.first;
            column.v_sfc = relative_wind.second;
            constexpr double kappa =
                physical_constants::gas_constant_dry_air_jkgk
                / physical_constants::specific_heat_cp_jkgk;
            column.theta_sfc = runtime.surface_config.Tsfc > 0.0
                ? runtime.surface_config.Tsfc * std::pow(
                    physical_constants::reference_pressure_pa / p.front(),
                    kappa)
                : skin_theta;
            column.qv_sfc = runtime.surface_config.qsfc >= 0.0
                ? runtime.surface_config.qsfc : skin_qv;
            column.z_sfc = workspace.z_center.front();
            column.surface_slope_x = -post_shift_metrics.Gx_face(i, j, 0);
            column.surface_slope_y = -post_shift_metrics.Gy_face(i, j, 0);
            column.slab_state =
                &prepared.terrain_leading_slab_columns[surface_index];
            column.workspace = &workspace;
            column.elapsed_time_s = 0.0;
            const auto stored = evaluate_prepared_boundary_layer_column(
                *scheme, runtime.config, runtime.surface_config, column,
                state.grid.dr * state.grid.dtheta, tendencies, diagnostics);
            const std::vector<float>* source_fields[] = {
                &workspace.stored_u_force_n,
                &workspace.stored_v_force_n,
                &workspace.stored_theta_inventory_rate_kgk_s,
                &workspace.stored_vapor_mass_rate_kg_s,
                &workspace.stored_dtke};
            for (std::size_t owner = 0; owner < 5; ++owner)
                std::copy(
                    source_fields[owner]->begin(), source_fields[owner]->end(),
                    prepared.terrain_leading_fields[owner].row_ptr(i, j));
            const float surface_values[] = {
                stored.ustar, stored.tau_u, stored.tau_v,
                stored.sensible_heat_flux, stored.latent_heat_flux,
                stored.moisture_flux, stored.pbl_height};
            for (std::size_t owner = 0; owner < 7; ++owner)
                prepared.terrain_leading_surfaces[owner + 2][surface_index]
                    = surface_values[owner];
        }
    return prepared;
}

bool append_boundary_layer_domain_recycling_bindings(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const PreparedBoundaryLayerDomainRecyclingState& prepared,
    std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings,
    std::vector<HomogeneousSurfaceRecordShiftBinding>& record_bindings,
    std::vector<HomogeneousBooleanCommitBinding>& flag_bindings)
{
    if (!prepared.active)
        return false;
    if (state.boundary_layer_runtime == nullptr
        || !plan_matches_grid(plan, state)
        || prepared.nr != state.grid.nr || prepared.nth != state.grid.nth
        || prepared.nz != state.grid.nz
        || prepared.shift_i_cells != plan.shift_i_cells
        || prepared.shift_j_cells != plan.shift_j_cells)
        throw std::invalid_argument(
            "prepared boundary-layer recycling state does not match the live plan");
    const std::size_t nz = static_cast<std::size_t>(state.grid.nz);
    if (prepared.terrain_aware)
    {
        for (const auto& field : prepared.terrain_leading_fields)
            require_grid_field(
                field, state, "prepared terrain PBL leading field");
        const std::size_t expected_surface =
            static_cast<std::size_t>(state.grid.nr)
            * static_cast<std::size_t>(state.grid.nth);
        for (const auto& surface : prepared.terrain_leading_surfaces)
            if (surface.size() != expected_surface)
                throw std::invalid_argument(
                    "prepared terrain PBL leading surface shape mismatch");
        if (prepared.terrain_leading_slab_columns.size()
            != expected_surface)
            throw std::invalid_argument(
                "prepared terrain PBL slab-column shape mismatch");
    }
    else
    {
        const std::vector<float>* const profiles[] = {
            &prepared.leading_u_force_n, &prepared.leading_v_force_n,
            &prepared.leading_theta_inventory_rate_kgk_s,
            &prepared.leading_vapor_mass_rate_kg_s,
            &prepared.leading_dtke_s,
        };
        for (const auto* profile : profiles)
            if (profile->size() != nz)
                throw std::invalid_argument(
                    "prepared boundary-layer leading profile does not match nz");
    }
    BoundaryLayerRuntimeState& runtime = *state.boundary_layer_runtime;
    const std::size_t surface_size = static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    if (runtime.slab_columns.size() != surface_size)
        throw std::invalid_argument(
            "boundary-layer slab storage does not match the live grid");
    static_assert(std::is_trivially_copyable<SlabColumnState>::value,
                  "D5 record shifting requires trivial slab storage");
    Field3D* const live_volume[] = {
        &runtime.held_u_force_n, &runtime.held_v_force_n,
        &runtime.held_theta_inventory_rate_kgk_s,
        &runtime.held_vapor_mass_rate_kg_s,
        &state.tendencies.dtke_dt_pbl};
    const char* const volume_names[] = {
        "boundary_layer.held_u_force_n",
        "boundary_layer.held_v_force_n",
        "boundary_layer.held_theta_inventory_rate_kgk_s",
        "boundary_layer.held_vapor_mass_rate_kg_s",
        "boundary_layer.dtke_dt_pbl"};
    const std::vector<float>* const flat_profiles[] = {
        &prepared.leading_u_force_n, &prepared.leading_v_force_n,
        &prepared.leading_theta_inventory_rate_kgk_s,
        &prepared.leading_vapor_mass_rate_kg_s,
        &prepared.leading_dtke_s};
    for (std::size_t owner = 0; owner < 5; ++owner)
        volume_bindings.push_back({
            volume_names[owner], live_volume[owner],
            prepared.terrain_aware ? nullptr : flat_profiles[owner],
            HorizontalFieldLocation::ScalarCenter,
            prepared.terrain_aware
                ? &prepared.terrain_leading_fields[owner] : nullptr});

    std::vector<float>* const live_surfaces[] = {
        &runtime.skin_theta_2d, &runtime.skin_qv_2d,
        state.surface.sfc_ustar, state.surface.sfc_tau_u,
        state.surface.sfc_tau_v, state.surface.sfc_sensible_heat_flux,
        state.surface.sfc_latent_heat_flux,
        state.surface.sfc_moisture_flux, state.surface.pbl_height_2d};
    const char* const surface_names[] = {
        "boundary_layer.skin_theta", "boundary_layer.skin_qv",
        "surface.ustar", "surface.tau_u", "surface.tau_v",
        "surface.sensible_heat_flux", "surface.latent_heat_flux",
        "surface.moisture_flux", "surface.pbl_height"};
    const float flat_surface_values[] = {
        prepared.leading_skin_theta, prepared.leading_skin_qv,
        prepared.leading_ustar, prepared.leading_tau_u,
        prepared.leading_tau_v, prepared.leading_sensible_heat_flux,
        prepared.leading_latent_heat_flux,
        prepared.leading_moisture_flux, prepared.leading_pbl_height};
    for (std::size_t owner = 0; owner < 9; ++owner)
        surface_bindings.push_back({
            surface_names[owner], live_surfaces[owner],
            flat_surface_values[owner],
            prepared.terrain_aware
                ? &prepared.terrain_leading_surfaces[owner] : nullptr});
    record_bindings.push_back({
        "boundary_layer.slab_columns", runtime.slab_columns.data(),
        runtime.slab_columns.size(), sizeof(SlabColumnState),
        prepared.terrain_aware ? nullptr : &prepared.leading_slab_column,
        prepared.terrain_aware
            ? prepared.terrain_leading_slab_columns.data() : nullptr});
    flag_bindings.push_back({
        "boundary_layer.updated_this_step", &runtime.updated_this_step, false});
    return true;
}

bool append_boundary_layer_trailing_export_bindings(
    const SimulationStateView& state,
    std::array<std::vector<float>, 5>& packed_values,
    std::vector<TrailingFieldExportBinding>& bindings)
{
    if (state.boundary_layer_runtime == nullptr)
        throw std::invalid_argument(
            "boundary-layer trailing binding requires a bound runtime");
    BoundaryLayerRuntimeState& runtime = *state.boundary_layer_runtime;
    if (runtime.config.scheme_id == "none")
        return false;
    const Field3D* const fields[] = {
        &runtime.held_u_force_n, &runtime.held_v_force_n,
        &runtime.held_theta_inventory_rate_kgk_s,
        &runtime.held_vapor_mass_rate_kg_s,
        &state.tendencies.dtke_dt_pbl,
    };
    const char* const names[] = {
        "boundary_layer.held_u_force_n",
        "boundary_layer.held_v_force_n",
        "boundary_layer.held_theta_inventory_rate_kgk_s",
        "boundary_layer.held_vapor_mass_rate_kg_s",
        "boundary_layer.dtke_dt_pbl",
    };
    for (std::size_t index = 0; index < 5; ++index)
    {
        require_grid_field(
            *fields[index], state, "boundary-layer persistent field");
        bindings.push_back({
            names[index], fields[index], &packed_values[index],
            HorizontalFieldLocation::ScalarCenter});
    }
    return true;
}

const char* boundary_layer_domain_recycling_algorithm_id() noexcept
{
    return "held_sample_flat_or_mapped_pbl_shift_v2";
}

bool append_turbulence_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousScalarCommitBinding>& scalar_bindings,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.turbulence_runtime == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "turbulence recycling requires bound turbulence and terrain runtimes");
    if (!std::isfinite(state.time_s) || state.time_s < 0.0)
        throw std::invalid_argument(
            "D5 turbulence recycling requires a finite non-negative model time");

    TurbulenceRuntimeState& runtime = *state.turbulence_runtime;
    if (runtime.scheme == nullptr)
        throw std::invalid_argument(
            "D5 turbulence recycling requires an initialized scheme");
    if (runtime.scheme->name() != runtime.config.scheme_id)
        throw std::invalid_argument(
            "D5 turbulence scheme and configuration ownership disagree");
    if (!std::isfinite(runtime.config.dt_sgs)
        || runtime.config.dt_sgs <= 0.0
        || !std::isfinite(runtime.last_update_time_s)
        || !std::isfinite(runtime.next_update_time_s))
        throw std::invalid_argument(
            "D5 turbulence cadence state is invalid");
    const double time_tolerance_s = 1.0e-12 * std::max(
        {1.0, std::abs(state.time_s),
         std::abs(runtime.last_update_time_s)});
    if (runtime.last_update_time_s > state.time_s + time_tolerance_s)
        throw std::invalid_argument(
            "D5 turbulence sample time lies in the model future");
    const double stability_limit =
        runtime.last_explicit_stability_limit_s;
    if (!((std::isfinite(stability_limit) && stability_limit > 0.0)
          || (std::isinf(stability_limit) && stability_limit > 0.0)))
        throw std::invalid_argument(
            "D5 turbulence predictive stability state is invalid");

    require_turbulence_tendency_group(
        runtime.held_tendencies, state, "held", true);
    // A rejected scheme evaluation may intentionally leave non-finite values
    // in unpublished candidate scratch. Its allocation and shape must still
    // be complete, but recycling may erase that unpublished scratch.
    require_turbulence_tendency_group(
        runtime.candidate_tendencies, state, "candidate", false);

    const TurbulenceMomentumWorkReport* const reports[] = {
        &runtime.last_momentum_work,
        &runtime.cumulative_moist_energy_work,
    };
    for (const auto* report : reports)
    {
        const double values[] = {
            report->requested_linear_work_j,
            report->finite_step_correction_j,
            report->storage_rounding_j,
            report->committed_work_j,
            report->moist_dry_internal_change_j,
            report->moist_water_sensible_change_j,
            report->moist_phase_reference_change_j,
            report->moist_carrier_kinetic_change_j,
            report->moist_gravitational_change_j,
            report->moist_subgrid_tke_change_j,
            report->moist_committed_energy_change_j,
            report->moist_energy_closure_residual_j,
        };
        if (!std::all_of(
                std::begin(values), std::end(values),
                [](double value) { return std::isfinite(value); })
            || report->finite_step_correction_j < 0.0)
            throw std::invalid_argument(
                "D5 turbulence energy history contains an invalid value");
    }
    const double cumulative_values[] = {
        runtime.cumulative_requested_momentum_work_j,
        runtime.cumulative_finite_step_momentum_correction_j,
        runtime.cumulative_momentum_storage_rounding_j,
        runtime.cumulative_committed_momentum_work_j,
    };
    if (!std::all_of(
            std::begin(cumulative_values), std::end(cumulative_values),
            [](double value) { return std::isfinite(value); })
        || runtime.cumulative_finite_step_momentum_correction_j < 0.0)
        throw std::invalid_argument(
            "D5 turbulence cumulative work history is invalid");

    scalar_bindings.push_back({
        "turbulence.next_update_time_s", &runtime.next_update_time_s,
        state.time_s});
    runtime_invalidation_bindings.push_back({
        "turbulence.state_dependent_runtime", &runtime,
        invalidate_turbulence_runtime_after_domain_recycling});
    return true;
}

const char* turbulence_domain_recycling_algorithm_id() noexcept
{
    return "preserve_history_invalidate_nonlinear_sgs_state_v1";
}

bool append_non_evolving_chaos_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.chaos_runtime == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "chaos recycling requires bound chaos and terrain runtimes");
    auto& runtime = *state.chaos_runtime;
    if (runtime.scheme == nullptr)
        throw std::invalid_argument(
            "D5 chaos recycling requires an initialized scheme");
    const std::string scheme_name = runtime.scheme->name();
    if (scheme_name != runtime.config.scheme_id)
        throw std::invalid_argument(
            "D5 chaos scheme and configuration ownership disagree");
    if (scheme_name != "none" && scheme_name != "initial_conditions")
        throw std::invalid_argument(
            "D5 evolving chaos requires an Earth-coordinate correlated-noise extension; local RNG or zero-filled leading history is unphysical");
    if (!empty_portable_scheme_state(runtime.scheme->portable_state()))
        throw std::invalid_argument(
            "D5 non-evolving chaos scheme unexpectedly owns portable spatial history");
    if (!std::isfinite(runtime.config.dt_chaos)
        || runtime.config.dt_chaos <= 0.0
        || !std::isfinite(runtime.noise_elapsed_s)
        || runtime.noise_elapsed_s < 0.0
        || runtime.noise_elapsed_s >= runtime.config.dt_chaos + 1.0e-12)
        throw std::invalid_argument(
            "D5 chaos cadence state is invalid");

    const double diagnostic_values[] = {
        runtime.diagnostics.mean_perturbation,
        runtime.diagnostics.variance_perturbation,
        runtime.diagnostics.correlation_length_x,
        runtime.diagnostics.correlation_length_y,
        runtime.diagnostics.min_qv_after_perturbation,
        runtime.diagnostics.time_noise_update,
        runtime.diagnostics.time_apply_ic,
        runtime.diagnostics.time_apply_tend,
    };
    if (!std::all_of(
            std::begin(diagnostic_values), std::end(diagnostic_values),
            [](double value) { return std::isfinite(value); })
        || runtime.diagnostics.correlation_length_x < 0.0
        || runtime.diagnostics.correlation_length_y < 0.0
        || runtime.diagnostics.negative_moisture_count < 0
        || runtime.diagnostics.time_noise_update < 0.0
        || runtime.diagnostics.time_apply_ic < 0.0
        || runtime.diagnostics.time_apply_tend < 0.0)
        throw std::invalid_argument(
            "D5 chaos diagnostic history is invalid");

    const Field3D* const base_fields[] = {
        &runtime.base_du_dt_pbl,
        &runtime.base_dv_dt_pbl,
        &runtime.base_dtheta_dt_pbl,
        &runtime.base_dqv_dt_pbl,
        &runtime.base_dtke_dt_pbl,
    };
    bool complete = true;
    for (const Field3D* field : base_fields)
    {
        if (field->empty())
        {
            complete = false;
            continue;
        }
        require_grid_field(*field, state, "chaos cached PBL baseline");
        for (std::size_t index = 0; index < field->size(); ++index)
            if (!std::isfinite(static_cast<double>(field->data()[index])))
                throw std::invalid_argument(
                    "D5 chaos cached PBL baseline contains a non-finite value");
    }
    if (runtime.pbl_base_tendencies_valid && !complete)
        throw std::invalid_argument(
            "D5 chaos valid PBL baseline is only partially allocated");
    if (runtime.pbl_base_tendencies_valid)
    {
        const std::size_t surface_size =
            static_cast<std::size_t>(state.grid.nr) *
            static_cast<std::size_t>(state.grid.nth);
        const std::vector<float>* const surface_fields[] = {
            &runtime.base_sfc_ustar,
            &runtime.base_sfc_tau_u,
            &runtime.base_sfc_tau_v,
            &runtime.base_sfc_sensible_heat_flux,
            &runtime.base_sfc_latent_heat_flux,
            &runtime.base_sfc_moisture_flux};
        for (const std::vector<float>* field : surface_fields)
        {
            if (field->size() != surface_size ||
                !std::all_of(field->begin(), field->end(), [](float value) {
                    return std::isfinite(static_cast<double>(value));
                }))
                throw std::invalid_argument(
                    "D5 chaos cached PBL surface baseline is malformed");
        }
    }

    runtime_invalidation_bindings.push_back({
        "chaos.non_evolving_runtime", &runtime,
        invalidate_chaos_runtime_after_domain_recycling});
    return true;
}

const char* chaos_domain_recycling_algorithm_id() noexcept
{
    return "earth_control_lattice_translate_or_shift_pbl_cache_v3";
}

PreparedEvolvingChaosDomainRecyclingState
prepare_evolving_chaos_domain_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan)
{
    if (state.chaos_runtime == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "evolving chaos recycling requires bound chaos and terrain runtimes");
    if (!plan_matches_grid(plan, state))
        throw std::invalid_argument(
            "evolving chaos recycling requires a grid-matching shift plan");
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    if (plan.physical_i_begin != physical.i_begin
        || plan.physical_i_end != physical.i_end
        || plan.physical_j_begin != physical.j_begin
        || plan.physical_j_end != physical.j_end)
        throw std::invalid_argument(
            "evolving chaos recycling plan must match the physical scalar domain");

    auto& runtime = *state.chaos_runtime;
    if (!runtime.scheme || runtime.scheme->name() != runtime.config.scheme_id)
        throw std::invalid_argument(
            "D5 evolving chaos requires matching scheme ownership");
    const std::string scheme_name = runtime.scheme->name();
    if (scheme_name != "boundary_layer" && scheme_name != "full_stochastic")
        throw std::invalid_argument(
            "D5 evolving chaos preparation requires an evolving scheme");
    if (!std::isfinite(runtime.config.dt_chaos)
        || runtime.config.dt_chaos <= 0.0
        || !std::isfinite(runtime.config.tau_t)
        || runtime.config.tau_t < 0.0
        || !std::isfinite(runtime.noise_elapsed_s)
        || runtime.noise_elapsed_s < 0.0
        || runtime.noise_elapsed_s >= runtime.config.dt_chaos + 1.0e-12)
        throw std::invalid_argument(
            "D5 evolving chaos cadence state is invalid");

    PreparedEvolvingChaosDomainRecyclingState prepared;
    prepared.runtime = &runtime;
    prepared.scheme = runtime.scheme->clone();
    if (!prepared.scheme)
        throw std::invalid_argument(
            "D5 evolving chaos scheme does not support detached preparation");
    GridMetrics metrics;
    metrics.dx = state.grid.dr;
    metrics.dy = state.grid.dtheta;
    metrics.dz.assign(
        static_cast<std::size_t>(state.grid.nz), state.grid.dz);
    metrics.z_int = state.grid.geometry.z_faces;
    if (metrics.z_int.size()
        != static_cast<std::size_t>(state.grid.nz + 1))
    {
        metrics.z_int.resize(static_cast<std::size_t>(state.grid.nz + 1));
        for (int k = 0; k <= state.grid.nz; ++k)
            metrics.z_int[static_cast<std::size_t>(k)] =
                static_cast<double>(k) * state.grid.dz;
    }
    std::string error;
    if (!prepared.scheme->translate_horizontal_stochastic_state(
            plan.shift_i_cells, plan.shift_j_cells,
            runtime.config, metrics, error))
        throw std::invalid_argument(
            "D5 evolving chaos translation failed: " + error);
    return prepared;
}

bool append_evolving_chaos_pbl_cache_domain_recycling_bindings(
    const SimulationStateView& state,
    const PreparedBoundaryLayerDomainRecyclingState& boundary_layer,
    PreparedEvolvingChaosDomainRecyclingState& prepared,
    std::vector<HomogeneousFieldShiftBinding>& volume_bindings,
    std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings)
{
    if (state.chaos_runtime == nullptr ||
        prepared.runtime != state.chaos_runtime || !prepared.scheme)
        throw std::invalid_argument(
            "D5 evolving chaos PBL cache binding requires matching ownership");
    auto& runtime = *state.chaos_runtime;
    if (!runtime.pbl_base_tendencies_valid) return false;
    if (!boundary_layer.active)
        throw std::invalid_argument(
            "D5 stochastic PBL cache requires an active prepared boundary layer");

    const Field3D* const fields[] = {
        &runtime.base_du_dt_pbl,
        &runtime.base_dv_dt_pbl,
        &runtime.base_dtheta_dt_pbl,
        &runtime.base_dqv_dt_pbl,
        &runtime.base_dtke_dt_pbl};
    for (const Field3D* field : fields)
        require_grid_field(*field, state, "stochastic PBL deterministic cache");
    const std::size_t surface_size =
        static_cast<std::size_t>(state.grid.nr) *
        static_cast<std::size_t>(state.grid.nth);
    std::vector<float>* const surfaces[] = {
        &runtime.base_sfc_ustar,
        &runtime.base_sfc_tau_u,
        &runtime.base_sfc_tau_v,
        &runtime.base_sfc_sensible_heat_flux,
        &runtime.base_sfc_latent_heat_flux,
        &runtime.base_sfc_moisture_flux};
    for (const std::vector<float>* surface : surfaces)
        if (surface->size() != surface_size)
            throw std::invalid_argument(
                "D5 stochastic PBL surface cache shape mismatch");

    Field3D* const live_fields[] = {
        &runtime.base_du_dt_pbl, &runtime.base_dv_dt_pbl,
        &runtime.base_dtheta_dt_pbl, &runtime.base_dqv_dt_pbl,
        &runtime.base_dtke_dt_pbl};
    const char* const field_names[] = {
        "chaos.pbl_base_u_force", "chaos.pbl_base_v_force",
        "chaos.pbl_base_theta_inventory", "chaos.pbl_base_vapor_mass",
        "chaos.pbl_base_tke"};
    const std::vector<float>* const flat_profiles[] = {
        &boundary_layer.leading_u_force_n,
        &boundary_layer.leading_v_force_n,
        &boundary_layer.leading_theta_inventory_rate_kgk_s,
        &boundary_layer.leading_vapor_mass_rate_kg_s,
        &boundary_layer.leading_dtke_s};
    for (std::size_t owner = 0; owner < 5; ++owner)
        volume_bindings.push_back({
            field_names[owner], live_fields[owner],
            boundary_layer.terrain_aware ? nullptr : flat_profiles[owner],
            HorizontalFieldLocation::ScalarCenter,
            boundary_layer.terrain_aware
                ? &boundary_layer.terrain_leading_fields[owner] : nullptr});
    std::vector<float>* const live_surfaces[] = {
        &runtime.base_sfc_ustar, &runtime.base_sfc_tau_u,
        &runtime.base_sfc_tau_v,
        &runtime.base_sfc_sensible_heat_flux,
        &runtime.base_sfc_latent_heat_flux,
        &runtime.base_sfc_moisture_flux};
    const char* const surface_names[] = {
        "chaos.pbl_base_ustar", "chaos.pbl_base_tau_u",
        "chaos.pbl_base_tau_v", "chaos.pbl_base_sensible",
        "chaos.pbl_base_latent", "chaos.pbl_base_moisture"};
    const float flat_surface_values[] = {
        boundary_layer.leading_ustar, boundary_layer.leading_tau_u,
        boundary_layer.leading_tau_v,
        boundary_layer.leading_sensible_heat_flux,
        boundary_layer.leading_latent_heat_flux,
        boundary_layer.leading_moisture_flux};
    for (std::size_t owner = 0; owner < 6; ++owner)
        surface_bindings.push_back({
            surface_names[owner], live_surfaces[owner],
            flat_surface_values[owner],
            boundary_layer.terrain_aware
                ? &boundary_layer.terrain_leading_surfaces[owner + 2]
                : nullptr});
    prepared.preserve_shifted_pbl_base = true;
    return true;
}

bool append_evolving_chaos_domain_recycling_bindings(
    const SimulationStateView& state,
    PreparedEvolvingChaosDomainRecyclingState& prepared,
    std::vector<HomogeneousRuntimeCommitBinding>& runtime_commit_bindings)
{
    if (state.chaos_runtime == nullptr
        || prepared.runtime != state.chaos_runtime
        || !prepared.scheme
        || !state.chaos_runtime->scheme
        || prepared.scheme->name() != state.chaos_runtime->scheme->name())
        throw std::invalid_argument(
            "D5 evolving chaos binding requires matching prepared state");
    runtime_commit_bindings.push_back({
        "chaos.evolving_earth_stochastic_state", &prepared,
        &commit_prepared_evolving_chaos_domain_recycling});
    return true;
}

bool append_rayleigh_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousBooleanCommitBinding>& flag_bindings)
{
    validate_rayleigh_runtime_for_flat_horizontal_recycling(state);
    RayleighDampingRuntimeState& runtime = *state.rayleigh_runtime;
    flag_bindings.push_back({
        "rayleigh.upper_initialized", &runtime.upper_initialized,
        runtime.upper_initialized});
    flag_bindings.push_back({
        "rayleigh.lateral_initialized", &runtime.lateral_initialized,
        runtime.lateral_initialized});
    return true;
}

bool append_advection_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.advection_runtime == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "advection recycling requires bound advection and terrain runtimes");

    AdvectionRuntimeState& runtime = *state.advection_runtime;
    if (runtime.scheme == nullptr)
        throw std::invalid_argument(
            "D5 advection recycling requires an initialized scheme");
    if (runtime.scheme->name() != runtime.config.scheme_id)
        throw std::invalid_argument(
            "D5 advection scheme and configuration ownership disagree");
    if (!std::isfinite(runtime.config.positivity_dt)
        || runtime.config.positivity_dt <= 0.0
        || !std::isfinite(runtime.config.cfl_target)
        || runtime.config.cfl_target <= 0.0)
        throw std::invalid_argument(
            "D5 advection configuration contains an invalid timestep or CFL target");

    // This flag is an outstanding obligation: thermodynamic batching has
    // transported tracer, but the orchestrator has not consumed the handoff.
    // Clearing it at this point would make the current timestep ambiguous.
    if (runtime.tracer_advected_in_batch)
        throw std::invalid_argument(
            "D5 advection recycling requires a timestep boundary with no outstanding batched tracer handoff");
    if (runtime.vertical_tendencies.fused_dst != nullptr
        || runtime.vertical_tendencies.fused_src != nullptr
        || runtime.vertical_tendencies.fused_completed)
        throw std::invalid_argument(
            "D5 advection recycling cannot interrupt an outstanding fused advection call");

    if (!valid_tracer_roundoff_runtime(runtime))
        throw std::invalid_argument(
            "D5 advection passive-tracer roundoff ledger is invalid");

    const bool boundary_all =
        all_reports_complete(runtime.thermodynamic_boundary_reports);
    const bool boundary_none =
        no_reports_complete(runtime.thermodynamic_boundary_reports);
    if ((!boundary_all && !boundary_none)
        || runtime.thermodynamic_boundary_reports_complete != boundary_all)
        throw std::invalid_argument(
            "D5 advection current boundary report set is only partially published");
    if (boundary_all
        && !std::all_of(
            runtime.thermodynamic_boundary_reports.begin(),
            runtime.thermodynamic_boundary_reports.end(),
            finite_boundary_report))
        throw std::invalid_argument(
            "D5 advection current boundary report contains a non-finite value");

    const bool energy_all =
        all_reports_complete(runtime.thermodynamic_moist_energy_reports);
    const bool energy_none =
        no_reports_complete(runtime.thermodynamic_moist_energy_reports);
    if ((!energy_all && !energy_none)
        || runtime.thermodynamic_moist_energy_reports_complete != energy_all)
        throw std::invalid_argument(
            "D5 advection current moist-energy report set is only partially published");
    if (energy_all
        && !std::all_of(
            runtime.thermodynamic_moist_energy_reports.begin(),
            runtime.thermodynamic_moist_energy_reports.end(),
            finite_moist_energy_report))
        throw std::invalid_argument(
            "D5 advection current moist-energy report contains a non-finite value");

    const bool cumulative_boundary_all = all_reports_complete(
        runtime.cumulative_thermodynamic_boundary_reports);
    const bool cumulative_boundary_none = no_reports_complete(
        runtime.cumulative_thermodynamic_boundary_reports);
    if ((!cumulative_boundary_all && !cumulative_boundary_none)
        || (runtime.thermodynamic_boundary_report_step_count == 0
            ? !cumulative_boundary_none : !cumulative_boundary_all)
        || !std::all_of(
            runtime.cumulative_thermodynamic_boundary_reports.begin(),
            runtime.cumulative_thermodynamic_boundary_reports.end(),
            finite_boundary_report))
        throw std::invalid_argument(
            "D5 advection cumulative boundary ledger is invalid");

    const bool cumulative_energy_all = all_reports_complete(
        runtime.cumulative_thermodynamic_moist_energy_reports);
    const bool cumulative_energy_none = no_reports_complete(
        runtime.cumulative_thermodynamic_moist_energy_reports);
    if ((!cumulative_energy_all && !cumulative_energy_none)
        || (runtime.thermodynamic_moist_energy_report_step_count == 0
            ? !cumulative_energy_none : !cumulative_energy_all)
        || !std::all_of(
            runtime.cumulative_thermodynamic_moist_energy_reports.begin(),
            runtime.cumulative_thermodynamic_moist_energy_reports.end(),
            finite_moist_energy_report))
        throw std::invalid_argument(
            "D5 advection cumulative moist-energy ledger is invalid");

    if (runtime.activity_current
        && (runtime.nr != state.grid.nr
            || runtime.nth != state.grid.nth
            || runtime.nz != state.grid.nz))
        throw std::invalid_argument(
            "D5 advection hydrometeor activity metadata has a stale shape");

    runtime_invalidation_bindings.push_back({
        "advection.generation_local_runtime", &runtime,
        invalidate_advection_runtime_after_domain_recycling});
    return true;
}

const char* advection_domain_recycling_algorithm_id() noexcept
{
    return "preserve_transport_history_invalidate_generation_state_v1";
}

bool append_diffusion_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.diffusion_runtime == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "diffusion recycling requires bound diffusion and terrain runtimes");

    DiffusionRuntimeState& runtime = *state.diffusion_runtime;
    const DiffusionConfig& config = runtime.config;
    if (!std::isfinite(config.K_h) || config.K_h < 0.0
        || !std::isfinite(config.K_v) || config.K_v < 0.0
        || !std::isfinite(config.dt_diffusion)
        || config.dt_diffusion <= 0.0
        || config.operator_order != 2
        || config.operator_type != "laplacian")
        throw std::invalid_argument(
            "D5 diffusion configuration is invalid");
    if (config.apply_to != "scalars"
        && config.apply_to != "scalars_and_momentum"
        && config.apply_to != "momentum"
        && config.apply_to != "all")
        throw std::invalid_argument(
            "D5 diffusion configuration has an unknown target");

    const bool active = config.K_h > 0.0 || config.K_v > 0.0
        || config.use_variable_K;
    if (active && runtime.scheme == nullptr)
        throw std::invalid_argument(
            "D5 active diffusion requires an initialized scheme");
    if (runtime.scheme != nullptr
        && runtime.scheme->name() != config.scheme_id)
        throw std::invalid_argument(
            "D5 diffusion scheme and configuration ownership disagree");

    require_diffusion_field_group(
        diffusion_tendency_fields(runtime.tendencies), state,
        "published tendency", true);
    // Candidates can be allocated independently according to apply_to. After
    // a successful production swap each allocated field is an old-grid
    // prognostic buffer, not an additional scientific state owner.
    require_diffusion_field_group(
        diffusion_candidate_fields(runtime), state,
        "transaction candidate", false);

    if (!std::isfinite(runtime.diagnostics.max_diffusion_number)
        || runtime.diagnostics.max_diffusion_number < 0.0)
        throw std::invalid_argument(
            "D5 diffusion stability diagnostic is invalid");
    if (!runtime.diagnostics.K_effective.empty())
    {
        require_grid_field(
            runtime.diagnostics.K_effective, state,
            "diffusion effective coefficient diagnostic");
        for (std::size_t index = 0;
             index < runtime.diagnostics.K_effective.size(); ++index)
        {
            const float value =
                runtime.diagnostics.K_effective.data()[index];
            if (!std::isfinite(static_cast<double>(value)) || value < 0.0f)
                throw std::invalid_argument(
                    "D5 diffusion effective coefficient diagnostic contains an invalid value");
        }
    }

    if (!finite_diffusion_work_report(runtime.last_momentum_work)
        || !finite_diffusion_work_report(
            runtime.cumulative_moist_energy_work))
        throw std::invalid_argument(
            "D5 diffusion work history contains an invalid value");
    const double cumulative_values[] = {
        runtime.cumulative_requested_momentum_work_j,
        runtime.cumulative_finite_step_momentum_correction_j,
        runtime.cumulative_momentum_storage_rounding_j,
        runtime.cumulative_committed_momentum_work_j,
    };
    if (!std::all_of(
            std::begin(cumulative_values), std::end(cumulative_values),
            [](double value) { return std::isfinite(value); })
        || runtime.cumulative_finite_step_momentum_correction_j < 0.0
        || (runtime.momentum_work_step_count == 0
            && std::any_of(
                std::begin(cumulative_values), std::end(cumulative_values),
                [](double value) { return value != 0.0; }))
        || (runtime.moist_energy_step_count == 0
            && runtime.cumulative_moist_energy_work.moist_energy_complete)
        || (runtime.moist_energy_step_count > 0
            && !runtime.cumulative_moist_energy_work.moist_energy_complete))
        throw std::invalid_argument(
            "D5 diffusion cumulative work ledger is inconsistent");

    runtime_invalidation_bindings.push_back({
        "diffusion.generation_local_runtime", &runtime,
        invalidate_diffusion_runtime_after_domain_recycling});
    return true;
}

const char* diffusion_domain_recycling_algorithm_id() noexcept
{
    return "preserve_cumulative_work_invalidate_diffusion_generation_v1";
}

bool append_time_stepping_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.time_stepping_runtime == nullptr
        || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "time-stepping recycling requires bound time-stepping and terrain runtimes");

    TimeSteppingRuntimeState& runtime = *state.time_stepping_runtime;
    const TimeSteppingConfig& config = runtime.config;
    if (runtime.scheme == nullptr)
        throw std::invalid_argument(
            "D5 time-stepping recycling requires an initialized scheme");
    if (runtime.scheme->name() != config.scheme_id)
        throw std::invalid_argument(
            "D5 time-stepping scheme and configuration ownership disagree");
    if (!runtime.scheme->supports_domain_recycling_invalidation())
        throw std::invalid_argument(
            "D5 time-stepping scheme does not implement generation-state invalidation");
    if (config.split_acoustic && !runtime.scheme->supports_split_acoustic())
        throw std::invalid_argument(
            "D5 split-acoustic configuration is not supported by its time-stepping scheme");
    if (config.n_acoustic_substeps < 0
        || !std::isfinite(config.max_acoustic_speed_ms)
        || config.max_acoustic_speed_ms < 0.0
        || !std::isfinite(config.min_acoustic_spacing_m)
        || config.min_acoustic_spacing_m < 0.0
        || !std::isfinite(config.cfl_safety)
        || config.cfl_safety <= 0.0
        || !std::isfinite(config.dt_min) || config.dt_min <= 0.0
        || !std::isfinite(config.dt_max) || config.dt_max <= 0.0
        || config.dt_min > config.dt_max)
        throw std::invalid_argument(
            "D5 time-stepping configuration is invalid");

    if (runtime.diagnostics.n_steps < 0
        || !std::isfinite(runtime.diagnostics.max_cfl)
        || runtime.diagnostics.max_cfl < 0.0
        || !std::isfinite(runtime.diagnostics.cpu_time_per_step)
        || runtime.diagnostics.cpu_time_per_step < 0.0
        || !std::all_of(
            runtime.diagnostics.dt_history.begin(),
            runtime.diagnostics.dt_history.end(),
            [](double value) {
                return std::isfinite(value) && value > 0.0;
            }))
        throw std::invalid_argument(
            "D5 time-stepping diagnostic history is invalid");

    const auto workspace = time_stepping_workspace_fields(runtime.slow_rk2);
    const auto validate_workspace_group =
        [&](std::size_t begin, std::size_t end, const char* group)
    {
        const bool any = std::any_of(
            workspace.begin() + static_cast<std::ptrdiff_t>(begin),
            workspace.begin() + static_cast<std::ptrdiff_t>(end),
            [](const Field3D* field) { return !field->empty(); });
        const bool all = std::all_of(
            workspace.begin() + static_cast<std::ptrdiff_t>(begin),
            workspace.begin() + static_cast<std::ptrdiff_t>(end),
            [](const Field3D* field) { return !field->empty(); });
        if (any != all)
            throw std::invalid_argument(
                std::string("D5 time-stepping ") + group
                + " workspace is only partially allocated");
        if (!all) return false;
        for (std::size_t field_index = begin;
             field_index < end; ++field_index)
        {
            const Field3D& field = *workspace[field_index];
            require_grid_field(field, state, group);
            for (std::size_t index = 0; index < field.size(); ++index)
                if (!std::isfinite(
                        static_cast<double>(field.data()[index])))
                    throw std::invalid_argument(
                        std::string("D5 time-stepping ") + group
                        + " workspace contains a non-finite value");
        }
        return true;
    };
    const bool primary_workspace =
        validate_workspace_group(0, 8, "slow RK2 primary");
    const bool advection_workspace =
        validate_workspace_group(8, 11, "slow RK2 advection");
    if (advection_workspace && !primary_workspace)
        throw std::invalid_argument(
            "D5 time-stepping advection workspace exists without the primary slow RK2 workspace");

    if (!valid_boundary_mass_report(runtime.last_boundary_mass))
        throw std::invalid_argument(
            "D5 time-stepping last boundary-mass report is invalid");
    const double cumulative_boundary[] = {
        runtime.cumulative_outward_boundary_mass_kg,
        runtime.cumulative_flux_implied_state_change_kg,
        runtime.cumulative_committed_state_change_kg,
        runtime.cumulative_boundary_mass_closure_residual_kg,
    };
    if (!std::all_of(
            std::begin(cumulative_boundary), std::end(cumulative_boundary),
            [](double value) { return std::isfinite(value); })
        || runtime.cumulative_flux_implied_state_change_kg
            != -runtime.cumulative_outward_boundary_mass_kg
        || (runtime.boundary_mass_step_count == 0
            && std::any_of(
                std::begin(cumulative_boundary),
                std::end(cumulative_boundary),
                [](double value) { return value != 0.0; })))
        throw std::invalid_argument(
            "D5 time-stepping cumulative boundary-mass ledger is invalid");

    if (!valid_acoustic_energy_report(runtime.last_acoustic_energy)
        || !valid_cumulative_report(
            runtime.cumulative_acoustic_energy,
            runtime.acoustic_energy_step_count,
            valid_acoustic_energy_report)
        || (runtime.acoustic_energy_step_count > 0
            && !runtime.cumulative_acoustic_energy.boundary_complete)
        || !valid_slow_momentum_energy_report(
            runtime.last_slow_momentum_energy)
        || !valid_cumulative_report(
            runtime.cumulative_slow_momentum_energy,
            runtime.slow_momentum_energy_step_count,
            valid_slow_momentum_energy_report)
        || !valid_pressure_energy_report(runtime.last_eos_pressure_energy)
        || !valid_cumulative_report(
            runtime.cumulative_eos_pressure_energy,
            runtime.eos_pressure_energy_step_count,
            valid_pressure_energy_report)
        || !valid_sponge_energy_report(runtime.last_sponge_energy)
        || !valid_cumulative_report(
            runtime.cumulative_sponge_energy,
            runtime.sponge_energy_step_count,
            valid_sponge_energy_report)
        || !valid_final_boundary_energy_report(
            runtime.last_final_boundary_energy)
        || !valid_cumulative_report(
            runtime.cumulative_final_boundary_energy,
            runtime.final_boundary_energy_step_count,
            valid_final_boundary_energy_report)
        || !structurally_valid_completed_step_energy_report(
            runtime.last_completed_step_energy)
        || !valid_cumulative_report(
            runtime.cumulative_completed_step_energy,
            runtime.completed_step_energy_count,
            structurally_valid_completed_step_energy_report))
        throw std::invalid_argument(
            "D5 time-stepping energy history is invalid or count-inconsistent");

    runtime_invalidation_bindings.push_back({
        "time_stepping.generation_local_runtime", &runtime,
        invalidate_time_stepping_runtime_after_domain_recycling});
    return true;
}

const char* time_stepping_domain_recycling_algorithm_id() noexcept
{
    return "preserve_completed_ledgers_invalidate_timestep_generation_v1";
}

namespace
{

std::array<const Field3D*, 11> radar_output_fields(
    const RadarOut& output) noexcept
{
    return {{
        &output.Z_dBZ, &output.Vr, &output.ZDR_dB, &output.Ze_linear,
        &output.ZH_dBZ, &output.ZV_dBZ, &output.Ze_rain, &output.Ze_snow,
        &output.Ze_graupel, &output.Ze_hail, &output.Ze_ice,
    }};
}

bool radar_output_has_storage(const RadarOut& output) noexcept
{
    const auto fields = radar_output_fields(output);
    return std::any_of(
        fields.begin(), fields.end(),
        [](const Field3D* field) { return !field->empty(); });
}

void validate_radar_output_generation(
    const RadarOut& output,
    const SimulationStateView& state,
    const char* owner)
{
    const auto fields = radar_output_fields(output);
    constexpr double minima[] = {
        -40.0, -RadarConstants::velocity_abs_max_ms, -12.0, 0.0,
        -40.0, -40.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    constexpr double maxima[] = {
        100.0, RadarConstants::velocity_abs_max_ms, 12.0, 1.0e12,
        100.0, 100.0, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12,
    };
    const bool has_storage = radar_output_has_storage(output);
    if (has_storage
        && (output.NR != state.grid.nr || output.NTH != state.grid.nth
            || output.NZ != state.grid.nz))
        throw std::invalid_argument(
            std::string("D5 radar ") + owner
            + " metadata does not match the live grid");
    if (!has_storage
        && !((output.NR == 0 && output.NTH == 0 && output.NZ == 0)
             || (output.NR == state.grid.nr
                 && output.NTH == state.grid.nth
                 && output.NZ == state.grid.nz)))
        throw std::invalid_argument(
            std::string("D5 radar ") + owner
            + " empty metadata is inconsistent");
    for (std::size_t field_index = 0;
         field_index < fields.size(); ++field_index)
    {
        const Field3D* field = fields[field_index];
        if (field->empty()) continue;
        require_grid_field(*field, state, owner);
        for (std::size_t index = 0; index < field->size(); ++index)
        {
            const double value = static_cast<double>(field->data()[index]);
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    std::string("D5 radar ") + owner
                    + " contains a non-finite product");
            if (value < minima[field_index] || value > maxima[field_index])
                throw std::invalid_argument(
                    std::string("D5 radar ") + owner
                    + " contains an out-of-contract product");
        }
    }
}

void invalidate_radar_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime = *static_cast<RadarRuntimeState*>(opaque_runtime);
    runtime.selected_output.invalidate_generation_values();
    runtime.polarimetric_output.invalidate_generation_values();
    for (RadarSchemeBase* scheme : {
             runtime.selected_scheme.get(),
             runtime.reflectivity_operator.get(),
             runtime.velocity_operator.get(),
             runtime.zdr_operator.get()})
        if (scheme != nullptr)
            scheme->invalidate_generation_state_after_domain_recycling();
}

} // namespace

bool append_radar_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.radar_runtime == nullptr || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "radar recycling requires bound radar and terrain runtimes");

    RadarRuntimeState& runtime = *state.radar_runtime;
    const RadarConfig& config = runtime.selected_config;
    const double coordinates[] = {
        config.radar_x, config.radar_y, config.radar_z,
        runtime.radar_x, runtime.radar_y, runtime.radar_z,
    };
    if (!std::all_of(
            std::begin(coordinates), std::end(coordinates),
            [](double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            "D5 radar instrument geometry contains a non-finite coordinate");

    const bool cache_uninitialized =
        runtime.nr == -1 && runtime.nth == -1 && runtime.nz == -1;
    const bool cache_matches =
        runtime.nr == state.grid.nr && runtime.nth == state.grid.nth
        && runtime.nz == state.grid.nz;
    if (!cache_uninitialized && !cache_matches)
        throw std::invalid_argument(
            "D5 radar dimension cache does not match the live grid");
    if ((runtime.reflectivity_operator != nullptr
         || runtime.velocity_operator != nullptr
         || runtime.zdr_operator != nullptr)
        && !cache_matches)
        throw std::invalid_argument(
            "D5 radar product operators require a current dimension cache");

    const auto validate_operator = [](const RadarSchemeBase* scheme,
                                      const char* expected,
                                      const char* owner)
    {
        if (scheme == nullptr) return;
        if (expected != nullptr && scheme->name() != expected)
            throw std::invalid_argument(
                std::string("D5 radar ") + owner
                + " owns the wrong operator");
        if (!scheme->supports_domain_recycling_invalidation())
            throw std::invalid_argument(
                std::string("D5 radar ") + owner
                + " does not implement generation-state invalidation");
    };
    validate_operator(
        runtime.selected_scheme.get(),
        runtime.selected_scheme != nullptr ? config.scheme_id.c_str() : nullptr,
        "selected scheme");
    validate_operator(
        runtime.reflectivity_operator.get(), "reflectivity",
        "reflectivity cache");
    validate_operator(
        runtime.velocity_operator.get(), "velocity", "velocity cache");
    validate_operator(runtime.zdr_operator.get(), "zdr", "ZDR cache");
    if (runtime.selected_scheme == nullptr
        && radar_output_has_storage(runtime.selected_output))
        throw std::invalid_argument(
            "D5 radar selected output exists without its operator");

    validate_radar_output_generation(
        runtime.selected_output, state, "selected output");
    validate_radar_output_generation(
        runtime.polarimetric_output, state, "polarimetric swap output");

    runtime_invalidation_bindings.push_back({
        "radar.generation_local_runtime", &runtime,
        invalidate_radar_runtime_after_domain_recycling});
    return true;
}

const char* radar_domain_recycling_algorithm_id() noexcept
{
    return "preserve_radar_operators_invalidate_generation_products_v1";
}

namespace
{

std::array<Field3D*, 12> derived_diagnostic_fields(
    DiagnosticFieldStorage& diagnostics) noexcept
{
    return tmv::fields::pointers(diagnostics, tmv::fields::diagnostics);
}

void invalidate_derived_diagnostics_after_domain_recycling(
    void* opaque_diagnostics) noexcept
{
    auto& diagnostics =
        *static_cast<DiagnosticFieldStorage*>(opaque_diagnostics);
    for (Field3D* field : derived_diagnostic_fields(diagnostics))
        field->fill(0.0f);
}

} // namespace

bool append_derived_diagnostic_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.diagnostics == nullptr)
        throw std::invalid_argument(
            "derived-diagnostic recycling requires bound owning storage");
    DiagnosticFieldStorage& diagnostics = *state.diagnostics;
    const auto fields = derived_diagnostic_fields(diagnostics);
    const auto validate_group = [&](std::size_t begin,
                                    std::size_t end,
                                    const char* group)
    {
        const bool any = std::any_of(
            fields.begin() + static_cast<std::ptrdiff_t>(begin),
            fields.begin() + static_cast<std::ptrdiff_t>(end),
            [](const Field3D* field) { return !field->empty(); });
        const bool all = std::all_of(
            fields.begin() + static_cast<std::ptrdiff_t>(begin),
            fields.begin() + static_cast<std::ptrdiff_t>(end),
            [](const Field3D* field) { return !field->empty(); });
        if (any != all)
            throw std::invalid_argument(
                std::string("D5 derived-diagnostic ") + group
                + " publication is partial");
        if (!all) return;
        for (std::size_t field_index = begin;
             field_index < end; ++field_index)
            require_grid_field(*fields[field_index], state, group);
    };
    validate_group(0, 1, "radar");
    validate_group(1, 7, "vorticity");
    validate_group(7, 9, "angular-momentum");
    validate_group(9, 12, "pressure");

    runtime_invalidation_bindings.push_back({
        "diagnostics.derived_generation", &diagnostics,
        invalidate_derived_diagnostics_after_domain_recycling});
    return true;
}

const char* derived_diagnostic_domain_recycling_algorithm_id() noexcept
{
    return "invalidate_lazy_derived_volume_generation_v1";
}

namespace
{

std::array<Field3D*, 8> dynamics_scratch_fields(
    DynamicsRuntimeState& runtime) noexcept
{
    return {{
        &runtime.du_dt, &runtime.dv_dt, &runtime.dw_dt,
        &runtime.drho_dt, &runtime.dp_dt,
        &runtime.du_advection_dt, &runtime.dv_advection_dt,
        &runtime.dw_advection_dt,
    }};
}

void invalidate_dynamics_runtime_after_domain_recycling(
    void* opaque_runtime) noexcept
{
    auto& runtime = *static_cast<DynamicsRuntimeState*>(opaque_runtime);
    runtime.momentum_interface_flux_register = {};
    runtime.momentum_interface_flux_capture_enabled = false;

    for (Field3D* field : dynamics_scratch_fields(runtime))
        field->fill(0.0f);
    for (auto& partial : runtime.slow_pressure_energy_columns)
        partial = {};
    std::fill(runtime.state_relaxation_profiles.begin(),
              runtime.state_relaxation_profiles.end(), 0.0f);
    std::fill(runtime.acoustic_profile_rows.begin(),
              runtime.acoustic_profile_rows.end(), 0.0f);
    runtime.transport_context
        .invalidate_generation_state_after_domain_recycling();
    runtime.scheme->invalidate_generation_state_after_domain_recycling();
}

} // namespace

bool append_dynamics_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.dynamics_runtime == nullptr
        || state.terrain_runtime == nullptr
        || state.configuration == nullptr)
        throw std::invalid_argument(
            "dynamics recycling requires bound runtime, terrain, and configuration ownership");
    if (state.grid.geometry.coord != CoordinateSystem::Cartesian)
        throw std::invalid_argument(
            "D5 dynamics recycling currently requires Cartesian coordinates");

    DynamicsRuntimeState& runtime = *state.dynamics_runtime;
    if (!runtime.scheme)
        throw std::invalid_argument(
            "D5 dynamics recycling requires an initialized scheme");
    if (!runtime.scheme->supports_domain_recycling_invalidation())
        throw std::invalid_argument(
            "D5 dynamics scheme does not implement generation-state invalidation");
    const std::string& configured_scheme =
        state.configuration->dynamics_scheme_name;
    const std::string canonical_config =
        configured_scheme == "klemp_wilhelmson_split"
            ? (state.grid.geometry.staggered
                   ? kCartesianCGridDynamicsId
                   : kCartesianAGridDynamicsId)
            : canonicalize_dynamics_scheme_id(configured_scheme);
    if (canonical_config != runtime.scheme->get_scheme_name())
        throw std::invalid_argument(
            "D5 dynamics scheme and configuration ownership disagree");
    const char* expected_coordinate = state.grid.geometry.staggered
        ? "cartesian_cgrid" : "cartesian";
    if (runtime.scheme->get_coordinate_system() != expected_coordinate)
        throw std::invalid_argument(
            "D5 dynamics scheme coordinate/stagger ownership disagrees with the grid");

    const auto fields = dynamics_scratch_fields(runtime);
    const auto validate_group = [&](std::size_t begin,
                                    std::size_t end,
                                    const char* group)
    {
        const bool any = std::any_of(
            fields.begin() + static_cast<std::ptrdiff_t>(begin),
            fields.begin() + static_cast<std::ptrdiff_t>(end),
            [](const Field3D* field) { return !field->empty(); });
        const bool all = std::all_of(
            fields.begin() + static_cast<std::ptrdiff_t>(begin),
            fields.begin() + static_cast<std::ptrdiff_t>(end),
            [](const Field3D* field) { return !field->empty(); });
        if (any != all)
            throw std::invalid_argument(
                std::string("D5 dynamics ") + group
                + " scratch is only partially allocated");
        if (!all) return;
        for (std::size_t index = begin; index < end; ++index)
            require_grid_field(*fields[index], state, group);
    };
    validate_group(0, 5, "primary tendency");
    validate_group(5, 8, "slow-advection tendency");

    if (runtime.transport_context.active()
        && !runtime.transport_context.is_complete())
        throw std::invalid_argument(
            "D5 dynamics recycling cannot interrupt an active acoustic mass-flux transaction");
    const std::array<const Field3D*, 4> transport_fields{{
        &runtime.transport_context.density_begin,
        &runtime.transport_context.mass_flux_x,
        &runtime.transport_context.mass_flux_y,
        &runtime.transport_context.mass_flux_z,
    }};
    const bool transport_any = std::any_of(
        transport_fields.begin(), transport_fields.end(),
        [](const Field3D* field) { return !field->empty(); });
    const bool transport_all = std::all_of(
        transport_fields.begin(), transport_fields.end(),
        [](const Field3D* field) { return !field->empty(); });
    if (transport_any != transport_all)
        throw std::invalid_argument(
            "D5 dynamics acoustic transport scratch is only partially allocated");
    if (transport_all)
        for (const Field3D* field : transport_fields)
            require_grid_field(*field, state, "dynamics acoustic transport");

    const std::size_t columns = static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    const auto require_empty_or_size = [](std::size_t actual,
                                          std::size_t expected,
                                          const char* owner)
    {
        if (actual != 0 && actual != expected)
            throw std::invalid_argument(
                std::string("D5 dynamics ") + owner
                + " scratch has a stale shape");
    };
    require_empty_or_size(runtime.slow_pressure_energy_columns.size(),
                          columns, "slow-pressure reduction");
    require_empty_or_size(runtime.state_relaxation_profiles.size(),
                          static_cast<std::size_t>(7 * state.grid.nz),
                          "state-relaxation profile");
    require_empty_or_size(runtime.acoustic_profile_rows.size(),
                          tmv::DynamicsExecutionWorkspace::acoustic_profile_row_count * static_cast<std::size_t>(state.grid.nz),
                          "acoustic profile");

    runtime_invalidation_bindings.push_back({
        "dynamics.generation_local_runtime", &runtime,
        invalidate_dynamics_runtime_after_domain_recycling});
    return true;
}

const char* dynamics_domain_recycling_algorithm_id() noexcept
{
    return "preserve_operator_invalidate_dynamics_generation_state_v1";
}

PreparedMicrophysicsDomainRecyclingState
prepare_flat_homogeneous_microphysics_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const HomogeneousPrimaryProfiles& primary_profiles)
{
    if (!plan_matches_grid(plan, state))
        throw std::invalid_argument(
            "microphysics recycling requires a grid-matching shift plan");
    if (!state.grid.geometry.staggered)
        throw std::invalid_argument(
            "D5 microphysics recycling currently requires the C-grid");
    if (state.microphysics_runtime == nullptr
        || state.radiation_runtime == nullptr
        || state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "microphysics recycling requires bound module runtimes");
    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    if (plan.physical_i_begin != physical.i_begin
        || plan.physical_i_end != physical.i_end
        || plan.physical_j_begin != physical.j_begin
        || plan.physical_j_end != physical.j_end)
        throw std::invalid_argument(
            "microphysics recycling plan must match the physical scalar domain");

    PreparedMicrophysicsDomainRecyclingState prepared;
    prepared.nr = plan.nr;
    prepared.nth = plan.nth;
    prepared.nz = plan.nz;
    prepared.shift_i_cells = plan.shift_i_cells;
    prepared.shift_j_cells = plan.shift_j_cells;
    prepared.zero_leading_profile.assign(
        static_cast<std::size_t>(plan.nz), 0.0f);

    MicrophysicsRuntimeState& runtime = *state.microphysics_runtime;
    if (!runtime.scheme)
        throw std::invalid_argument(
            "microphysics recycling requires an initialized scheme");
    MicrophysicsPrognosticScalarList live =
        runtime.scheme->prognostic_scalars_for_transport();
    for (std::size_t index = 0; index < live.size(); ++index)
    {
        const auto& scalar = live[index];
        if (scalar.name == nullptr || scalar.name[0] == '\0'
            || scalar.field == nullptr)
            throw std::invalid_argument(
                "microphysics private prognostic binding is incomplete");
        require_grid_field(
            *scalar.field, state, "microphysics private prognostic");
        for (std::size_t value = 0; value < scalar.field->size(); ++value)
        {
            const float stored = scalar.field->data()[value];
            if (!std::isfinite(stored)
                || (scalar.positive_definite && stored < 0.0f))
                throw std::invalid_argument(
                    "microphysics private prognostic contains an invalid value");
        }
        for (std::size_t previous = 0; previous < index; ++previous)
            if (std::string_view(live[previous].name) == scalar.name)
                throw std::invalid_argument(
                    "microphysics private prognostic names are not unique");
    }

    if (!live.empty())
    {
        const std::size_t nz = static_cast<std::size_t>(state.grid.nz);
        const auto make_column = [&](HomogeneousPrimaryOwner owner)
        {
            const auto& profile = primary_profiles[owner];
            if (profile.size() != nz)
                throw std::invalid_argument(
                    "microphysics leading profile does not match nz");
            Field3D field(1, 1, state.grid.nz, 0.0f);
            for (std::size_t k = 0; k < nz; ++k)
                field[0][0][static_cast<int>(k)] = profile[k];
            return field;
        };
        Field3D rho = make_column(HomogeneousPrimaryOwner::rho);
        Field3D qc = make_column(HomogeneousPrimaryOwner::qc);
        Field3D qr = make_column(HomogeneousPrimaryOwner::qr);
        Field3D qi = make_column(HomogeneousPrimaryOwner::qi);
        Field3D qs = make_column(HomogeneousPrimaryOwner::qs);
        Field3D qg = make_column(HomogeneousPrimaryOwner::qg);
        Field3D qh = make_column(HomogeneousPrimaryOwner::qh);
        std::unique_ptr<MicrophysicsScheme> leading_scheme =
            runtime.scheme->clone_parameters();
        if (!leading_scheme || !(leading_scheme->parameter_identity() == runtime.scheme->parameter_identity()))
            throw std::runtime_error(
                "microphysics scheme cannot preserve parameters while preparing fresh recycling prognostics");
        std::vector<double> thickness(state.grid.nz);
        for (int k = 0; k < state.grid.nz; ++k)
        {
            thickness[k] = tmv::physical_grid_metrics::layer_thickness(
                state, physical.i_begin, physical.j_begin, k);
        }
        leading_scheme->set_vertical_spacing_m(1, 1, state.grid.nz, std::move(thickness));
        leading_scheme->prepare_prognostic_state(
            rho, qc, qr, qi, qs, qg, qh);
        MicrophysicsPrognosticScalarList leading =
            leading_scheme->prognostic_scalars_for_transport();
        if (leading.size() != live.size())
            throw std::runtime_error(
                "microphysics leading-state prognostic layout changed");
        prepared.private_scalar_names.reserve(live.size());
        prepared.leading_private_scalars.reserve(live.size());
        for (std::size_t index = 0; index < live.size(); ++index)
        {
            if (leading[index].name == nullptr
                || std::string_view(leading[index].name) != live[index].name
                || leading[index].field == nullptr
                || leading[index].storage != live[index].storage
                || leading[index].field->size_r() != 1
                || leading[index].field->size_th() != 1
                || leading[index].field->size_z() != state.grid.nz)
                throw std::runtime_error(
                    "microphysics leading-state prognostic layout is inconsistent");
            prepared.private_scalar_names.emplace_back(live[index].name);
            auto& profile = prepared.leading_private_scalars.emplace_back();
            profile.resize(nz);
            for (std::size_t k = 0; k < nz; ++k)
            {
                const float value =
                    (*leading[index].field)[0][0][static_cast<int>(k)];
                if (!std::isfinite(value)
                    || (leading[index].positive_definite && value < 0.0f))
                    throw std::runtime_error(
                        "microphysics leading private prognostic is invalid");
                profile[k] = value;
            }
        }
    }

    prepared.active = !runtime.theta_storage_residual.empty()
        || !prepared.private_scalar_names.empty();
    if (state.radiation_runtime->config.scheme_id != "none"
        && runtime.theta_storage_residual.empty())
        throw std::invalid_argument(
            "active radiation recycling requires allocated theta storage residual");
    return prepared;
}

PreparedMicrophysicsDomainRecyclingState
prepare_terrain_aware_microphysics_recycling(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const TerrainAwarePrimaryFields& primary_fields,
    const TerrainMetrics3D& destination_metrics)
{
    if (!plan_matches_grid(plan, state) || state.microphysics_runtime == nullptr
        || state.radiation_runtime == nullptr
        || state.terrain_runtime == nullptr
        || state.terrain_runtime->config.horizontal_placement_id
            != "earth_fixed")
        throw std::invalid_argument(
            "terrain microphysics recycling requires Earth-fixed ownership");
    auto& runtime = *state.microphysics_runtime;
    if (!runtime.scheme)
        throw std::invalid_argument(
            "terrain microphysics recycling requires an initialized scheme");
    MicrophysicsPrognosticScalarList live =
        runtime.scheme->prognostic_scalars_for_transport();
    for (std::size_t index = 0; index < live.size(); ++index)
    {
        if (live[index].name == nullptr || live[index].field == nullptr)
            throw std::invalid_argument(
                "terrain private microphysics binding is incomplete");
        require_grid_field(
            *live[index].field, state,
            "terrain private microphysics prognostic");
        for (std::size_t value = 0; value < live[index].field->size(); ++value)
            if (!std::isfinite(live[index].field->data()[value])
                || (live[index].positive_definite
                    && live[index].field->data()[value] < 0.0f))
                throw std::invalid_argument(
                    "terrain private microphysics prognostic is invalid");
    }

    PreparedMicrophysicsDomainRecyclingState prepared;
    prepared.terrain_aware = true;
    prepared.nr = plan.nr;
    prepared.nth = plan.nth;
    prepared.nz = plan.nz;
    prepared.shift_i_cells = plan.shift_i_cells;
    prepared.shift_j_cells = plan.shift_j_cells;
    prepared.zero_leading_field.resize(
        plan.nr, plan.nth, plan.nz, 0.0f);
    if (!live.empty())
    {
        std::unique_ptr<MicrophysicsScheme> leading_scheme =
            runtime.scheme->clone_parameters();
        if (!leading_scheme || !(leading_scheme->parameter_identity() == runtime.scheme->parameter_identity()))
            throw std::runtime_error(
                "terrain microphysics scheme cannot preserve parameters in fresh prognostic preparation");
        tmv::terrain_transport::validate_cgrid_shapes(destination_metrics,
            primary_fields[HomogeneousPrimaryOwner::rho], primary_fields[HomogeneousPrimaryOwner::u],
            primary_fields[HomogeneousPrimaryOwner::v], primary_fields[HomogeneousPrimaryOwner::w]);
        tmv::bind_microphysics_grid_geometry(state, *leading_scheme, &destination_metrics);
        leading_scheme->prepare_prognostic_state(
            primary_fields[HomogeneousPrimaryOwner::rho],
            primary_fields[HomogeneousPrimaryOwner::qc],
            primary_fields[HomogeneousPrimaryOwner::qr],
            primary_fields[HomogeneousPrimaryOwner::qi],
            primary_fields[HomogeneousPrimaryOwner::qs],
            primary_fields[HomogeneousPrimaryOwner::qg],
            primary_fields[HomogeneousPrimaryOwner::qh]);
        const MicrophysicsPrognosticScalarList leading =
            leading_scheme->prognostic_scalars_for_transport();
        if (leading.size() != live.size())
            throw std::runtime_error(
                "terrain private microphysics layout changed during preparation");
        prepared.private_scalar_names.reserve(live.size());
        prepared.terrain_leading_private_scalars.reserve(live.size());
        for (std::size_t index = 0; index < live.size(); ++index)
        {
            if (leading[index].name == nullptr
                || std::string_view(leading[index].name) != live[index].name
                || leading[index].field == nullptr
                || leading[index].storage != live[index].storage)
                throw std::runtime_error(
                    "terrain private microphysics identity changed during preparation");
            require_grid_field(
                *leading[index].field, state,
                "terrain private microphysics leading field");
            prepared.private_scalar_names.emplace_back(live[index].name);
            prepared.terrain_leading_private_scalars.push_back(
                *leading[index].field);
        }
    }
    prepared.active = !runtime.theta_storage_residual.empty()
        || !prepared.private_scalar_names.empty();
    if (state.radiation_runtime->config.scheme_id != "none"
        && runtime.theta_storage_residual.empty())
        throw std::invalid_argument(
            "active terrain radiation requires theta residual ownership");
    return prepared;
}

bool append_prepared_microphysics_domain_recycling_bindings(
    const SimulationStateView& state,
    const WholeCellShiftPlan& plan,
    const PreparedMicrophysicsDomainRecyclingState& prepared,
    std::vector<HomogeneousFieldShiftBinding>& bindings)
{
    if (!prepared.active) return false;
    if (!plan_matches_grid(plan, state)
        || prepared.nr != state.grid.nr || prepared.nth != state.grid.nth
        || prepared.nz != state.grid.nz
        || prepared.shift_i_cells != plan.shift_i_cells
        || prepared.shift_j_cells != plan.shift_j_cells
        || state.microphysics_runtime == nullptr)
        throw std::invalid_argument(
            "prepared microphysics recycling state does not match the live plan");
    if (prepared.terrain_aware)
    {
        require_grid_field(
            prepared.zero_leading_field, state,
            "terrain microphysics zero leading field");
        if (prepared.private_scalar_names.size()
            != prepared.terrain_leading_private_scalars.size())
            throw std::invalid_argument(
                "terrain private microphysics preparation is incomplete");
    }
    else if (prepared.zero_leading_profile.size()
                 != static_cast<std::size_t>(state.grid.nz)
             || prepared.private_scalar_names.size()
                 != prepared.leading_private_scalars.size())
        throw std::invalid_argument(
            "flat microphysics preparation is incomplete");

    bool appended = false;
    if (prepared.terrain_aware)
    {
        Field3D& residual =
            state.microphysics_runtime->theta_storage_residual;
        if (!residual.empty())
        {
            require_grid_field(
                residual, state, "terrain microphysics theta residual");
            bindings.push_back({
                "microphysics.theta_storage_residual", &residual, nullptr,
                HorizontalFieldLocation::ScalarCenter,
                &prepared.zero_leading_field});
            appended = true;
        }
        else if (state.radiation_runtime != nullptr
                 && state.radiation_runtime->config.scheme_id != "none")
            throw std::invalid_argument(
                "active terrain radiation lacks theta residual storage");
    }
    else
        appended = append_microphysics_domain_recycling_binding(
            state, prepared.zero_leading_profile, bindings);
    MicrophysicsPrognosticScalarList live =
        state.microphysics_runtime->scheme
            ? state.microphysics_runtime->scheme
                  ->prognostic_scalars_for_transport()
            : MicrophysicsPrognosticScalarList{};
    if (live.size() != prepared.private_scalar_names.size())
        throw std::invalid_argument(
            "live microphysics private prognostic layout changed after preparation");
    for (std::size_t index = 0; index < live.size(); ++index)
    {
        if (live[index].name == nullptr || live[index].field == nullptr
            || prepared.private_scalar_names[index] != live[index].name
            || (!prepared.terrain_aware
                && prepared.leading_private_scalars[index].size()
                    != static_cast<std::size_t>(state.grid.nz)))
            throw std::invalid_argument(
                "live microphysics private prognostic binding is inconsistent");
        require_grid_field(
            *live[index].field, state, "microphysics private prognostic");
        bindings.push_back({
            prepared.private_scalar_names[index], live[index].field,
            prepared.terrain_aware
                ? nullptr : &prepared.leading_private_scalars[index],
            HorizontalFieldLocation::ScalarCenter,
            prepared.terrain_aware
                ? &prepared.terrain_leading_private_scalars[index]
                : nullptr});
        appended = true;
    }
    return appended;
}

bool append_microphysics_domain_recycling_invalidation_binding(
    const SimulationStateView& state,
    std::vector<HomogeneousRuntimeInvalidationBinding>&
        runtime_invalidation_bindings)
{
    if (state.microphysics_runtime == nullptr)
        throw std::invalid_argument(
            "microphysics recycling invalidation requires a bound runtime");
    auto& runtime = *state.microphysics_runtime;
    if (runtime.scheme == nullptr)
        throw std::invalid_argument(
            "microphysics recycling invalidation requires an initialized scheme");

    Field3D* const tendencies[] = {
        &runtime.tendencies.dtheta_dt,
        &runtime.tendencies.dqv_dt,
        &runtime.tendencies.dqc_dt,
        &runtime.tendencies.dqr_dt,
        &runtime.tendencies.dqi_dt,
        &runtime.tendencies.dqs_dt,
        &runtime.tendencies.dqg_dt,
        &runtime.tendencies.dqh_dt,
    };
    const bool any = std::any_of(
        std::begin(tendencies), std::end(tendencies),
        [](const Field3D* field) { return !field->empty(); });
    const bool all = std::all_of(
        std::begin(tendencies), std::end(tendencies),
        [](const Field3D* field) { return !field->empty(); });
    if (any != all)
        throw std::invalid_argument(
            "microphysics recycling tendency products are only partially allocated");
    if (all)
        for (const Field3D* field : tendencies)
            require_grid_field(
                *field, state, "microphysics transient tendency");

    runtime_invalidation_bindings.push_back({
        "microphysics.step_local_products", &runtime,
        invalidate_microphysics_runtime_after_domain_recycling});
    return true;
}

bool append_prepared_microphysics_trailing_export_bindings(
    const SimulationStateView& state,
    std::vector<std::vector<float>>& packed_values,
    std::vector<TrailingFieldExportBinding>& bindings)
{
    if (state.microphysics_runtime == nullptr
        || !state.microphysics_runtime->scheme)
        throw std::invalid_argument(
            "microphysics trailing export requires an initialized bound runtime");
    const bool residual =
        !state.microphysics_runtime->theta_storage_residual.empty();
    MicrophysicsPrognosticScalarList live =
        state.microphysics_runtime->scheme->prognostic_scalars_for_transport();
    const std::size_t expected = live.size() + (residual ? 1U : 0U);
    if (packed_values.size() != expected)
        throw std::invalid_argument(
            "microphysics trailing export buffer count is incorrect");
    std::size_t packed = 0;
    if (residual)
    {
        (void)append_microphysics_trailing_export_binding(
            state, packed_values[packed++], bindings);
    }
    for (std::size_t index = 0; index < live.size(); ++index)
    {
        if (live[index].name == nullptr || live[index].field == nullptr)
            throw std::invalid_argument(
                "microphysics private trailing binding is incomplete");
        require_grid_field(
            *live[index].field, state, "microphysics private prognostic");
        bindings.push_back({
            live[index].name, live[index].field,
            &packed_values[packed++], HorizontalFieldLocation::ScalarCenter});
    }
    return expected != 0;
}

const char* microphysics_domain_recycling_algorithm_id() noexcept
{
    return "persistent_private_moments_flat_or_mapped_environment_shift_v2";
}

bool append_surface_precipitation_domain_recycling_bindings(
    const SimulationStateView& state,
    std::vector<HomogeneousSurfaceShiftBinding>& surface_bindings)
{
    const auto* rate = state.surface.precipitation_rate_mm_h;
    const auto* accumulated = state.surface.accumulated_rainfall_mm;
    const auto* residual = state.surface.accumulated_rainfall_residual_mm;
    const double* clock = state.surface.accumulated_rainfall_last_update_s;
    const bool any = rate != nullptr || accumulated != nullptr
        || residual != nullptr || clock != nullptr;
    if (!any) return false;
    if (rate == nullptr || accumulated == nullptr || residual == nullptr
        || clock == nullptr)
        throw std::invalid_argument(
            "surface precipitation recycling state is only partially bound");
    const std::size_t surface_size =
        static_cast<std::size_t>(state.grid.nr)
        * static_cast<std::size_t>(state.grid.nth);
    if (rate->size() != surface_size || accumulated->size() != surface_size
        || residual->size() != surface_size)
        throw std::invalid_argument(
            "surface precipitation recycling state shape does not match the grid");
    if (!std::isfinite(*clock) || *clock < 0.0
        || !std::isfinite(state.time_s) || state.time_s < 0.0
        || *clock > state.time_s
            + 1.0e-9 * std::max(1.0, std::abs(state.time_s)))
        throw std::invalid_argument(
            "surface precipitation recycling clock is invalid");
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
    surface_bindings.push_back({
        "surface.precipitation_rate_mm_h",
        state.surface.precipitation_rate_mm_h, 0.0f});
    surface_bindings.push_back({
        "surface.accumulated_rainfall_mm",
        state.surface.accumulated_rainfall_mm, 0.0f});
    surface_bindings.push_back({
        "surface.accumulated_rainfall_residual_mm",
        state.surface.accumulated_rainfall_residual_mm, 0.0f});
    return true;
}

bool append_surface_precipitation_trailing_export_bindings(
    const SimulationStateView& state,
    std::vector<float>& precipitation_rate_values,
    std::vector<float>& accumulated_rainfall_values,
    std::vector<float>& accumulated_rainfall_residual_values,
    std::vector<TrailingSurfaceExportBinding>& bindings)
{
    if (state.surface.precipitation_rate_mm_h == nullptr
        || state.surface.accumulated_rainfall_mm == nullptr
        || state.surface.accumulated_rainfall_residual_mm == nullptr)
        throw std::invalid_argument(
            "surface precipitation trailing export requires bound state");
    bindings.push_back({
        "surface.precipitation_rate_mm_h",
        state.surface.precipitation_rate_mm_h,
        &precipitation_rate_values});
    bindings.push_back({
        "surface.accumulated_rainfall_mm",
        state.surface.accumulated_rainfall_mm,
        &accumulated_rainfall_values});
    bindings.push_back({
        "surface.accumulated_rainfall_residual_mm",
        state.surface.accumulated_rainfall_residual_mm,
        &accumulated_rainfall_residual_values});
    return true;
}

const char* surface_precipitation_domain_recycling_algorithm_id() noexcept
{
    return "exact_fallout_surface_history_shift_v1";
}

bool append_microphysics_domain_recycling_binding(
    const SimulationStateView& state,
    const std::vector<float>& zero_leading_profile,
    std::vector<HomogeneousFieldShiftBinding>& bindings)
{
    if (state.microphysics_runtime == nullptr ||
        state.radiation_runtime == nullptr)
        throw std::invalid_argument(
            "microphysics recycling binding requires bound module runtimes");
    const bool compensation_active =
        state.radiation_runtime->config.scheme_id != "none";
    Field3D& residual =
        state.microphysics_runtime->theta_storage_residual;
    if (residual.empty())
    {
        if (compensation_active)
            throw std::invalid_argument(
                "active radiation recycling requires allocated theta storage residual");
        return false;
    }
    if (residual.size_r() != state.grid.nr ||
        residual.size_th() != state.grid.nth ||
        residual.size_z() != state.grid.nz)
        throw std::invalid_argument(
            "microphysics theta storage residual shape does not match the recycling grid");
    if (zero_leading_profile.size() !=
        static_cast<std::size_t>(state.grid.nz))
        throw std::invalid_argument(
            "microphysics residual leading profile length does not match nz");
    for (float value : zero_leading_profile)
        if (value != 0.0f || !std::isfinite(value))
            throw std::invalid_argument(
                "microphysics residual leading profile must be exact finite zero");

    bindings.push_back({
        "microphysics.theta_storage_residual",
        &residual,
        &zero_leading_profile,
        HorizontalFieldLocation::ScalarCenter,
    });
    return true;
}

bool append_microphysics_trailing_export_binding(
    const SimulationStateView& state,
    std::vector<float>& packed_values,
    std::vector<TrailingFieldExportBinding>& bindings)
{
    if (state.microphysics_runtime == nullptr)
        throw std::invalid_argument(
            "microphysics trailing binding requires a bound runtime");
    const Field3D& residual =
        state.microphysics_runtime->theta_storage_residual;
    if (residual.empty())
        return false;
    if (residual.size_r() != state.grid.nr ||
        residual.size_th() != state.grid.nth ||
        residual.size_z() != state.grid.nz)
        throw std::invalid_argument(
            "microphysics theta storage residual shape does not match the recycling grid");
    bindings.push_back({
        "microphysics.theta_storage_residual",
        &residual,
        &packed_values,
        HorizontalFieldLocation::ScalarCenter,
    });
    return true;
}

void commit_prepared_earth_fixed_terrain_recycling(void* opaque) noexcept
{
    auto& prepared =
        *static_cast<PreparedFlatHomogeneousDomainRecycling*>(opaque);
    if (!commit_earth_fixed_terrain(
            prepared.terrain_commit, prepared.storage->runtime.terrain))
        std::terminate();
}

std::unique_ptr<PreparedFlatHomogeneousDomainRecycling>
prepare_flat_homogeneous_domain_recycling(
    SimulationStorageState& storage,
    const StormWindowMovementDecision& decision,
    const std::vector<double>& environmental_tke_profile)
{
    auto prepared =
        std::make_unique<PreparedFlatHomogeneousDomainRecycling>();
    prepared->storage = &storage;
    SimulationStateView state = make_simulation_state_view(storage);
    if (state.grid.geometry.coord != CoordinateSystem::Cartesian
        || !state.grid.geometry.staggered)
        throw std::invalid_argument(
            "live D5 recycling requires a Cartesian C-grid");
    if (state.terrain_runtime == nullptr)
        throw std::invalid_argument(
            "live D5 recycling requires terrain runtime ownership");
    const bool active_terrain =
        state.terrain_runtime->config.scheme_id != "none";
    if (active_terrain
        && state.terrain_runtime->config.horizontal_placement_id
            != "earth_fixed")
        throw std::invalid_argument(
            "live D5 terrain recycling requires Earth-fixed placement");
    if (storage.nested.config.enabled)
        throw std::invalid_argument(
            "live D5 recycling does not support legacy nested storage");
    if (state.compute_runtime == nullptr)
        throw std::invalid_argument(
            "live D5 recycling requires a bound compute runtime");
    prepared->authoritative_backend = state.compute_runtime->active_kind;
    const bool cpu_authoritative =
        prepared->authoritative_backend == ComputeBackendKind::Cpu;
    const bool accelerator_authoritative =
        prepared->authoritative_backend != ComputeBackendKind::Cpu
        && state.compute_runtime->backend != nullptr
        && state.compute_runtime->backend->supports_homogeneous_domain_field_shift();
    if (!cpu_authoritative && !accelerator_authoritative)
        throw std::invalid_argument(
            "live D5 recycling requires CPU or a native accelerator shift pipeline");
    const bool ready_reason =
        decision.reason == StormWindowMovementReason::RequestReady ||
        decision.reason == StormWindowMovementReason::ProjectedRelaxationRisk ||
        decision.reason == StormWindowMovementReason::TrustedInteriorContainment ||
        decision.reason == StormWindowMovementReason::EmergencyContainment;
    if (decision.status != StormWindowMovementStatus::Requested
        || !ready_reason
        || (decision.requested_i_shift_cells == 0
            && decision.requested_j_shift_cells == 0))
        throw std::invalid_argument(
            "live D5 recycling requires a ready nonzero movement request");

    const PhysicalCellDomain physical = physical_scalar_domain(state.grid);
    prepared->plan = plan_whole_cell_shift_for_physical_domain(
        state.grid.nr, state.grid.nth, state.grid.nz,
        decision.requested_i_shift_cells,
        decision.requested_j_shift_cells,
        physical.i_begin, physical.i_end,
        physical.j_begin, physical.j_end);
    prepared->pre_shift_frame = storage.configuration.domain_frame;
    prepared->post_shift_frame = prepared->pre_shift_frame;
    (void)commit_whole_cell_shift_to_domain_frame(
        prepared->post_shift_frame, prepared->plan,
        state.grid.dr, state.grid.dtheta);

    prepared->earth_fixed_terrain = active_terrain;
    if (active_terrain)
    {
        if ((state.boundary_layer_runtime->config.scheme_id != "none"
                && state.boundary_layer_runtime->config.scheme_id != "slab")
            || state.turbulence_runtime->config.scheme_id != "none")
            throw std::invalid_argument(
                "Earth-fixed terrain D5 has an incomplete active column-physics owner");
        const auto pre_origin = prepared->pre_shift_frame.earth_origin_at(
            state.time_s);
        const auto& placement =
            state.terrain_runtime->earth_fixed_placement;
        if (!placement.active
            || placement.grid_origin_earth_m.x_m != pre_origin.x_m
            || placement.grid_origin_earth_m.y_m != pre_origin.y_m)
            throw std::invalid_argument(
                "Earth-fixed terrain placement does not match the live domain frame");
        EarthFixedTerrainBuildRequest request;
        request.nr = state.grid.nr;
        request.nth = state.grid.nth;
        request.nz = state.grid.nz;
        request.dx_m = state.grid.dr;
        request.dy_m = state.grid.dtheta;
        request.grid_origin_earth_m =
            prepared->post_shift_frame.earth_origin_at(state.time_s);
        request.feature_center_earth_m = {
            state.terrain_runtime->config.feature_center_earth_x_m,
            state.terrain_runtime->config.feature_center_earth_y_m};
        request.sampling = placement.sampling;
        prepared->terrain_commit = prepare_earth_fixed_terrain_commit(
            *state.terrain_runtime, request);
        prepared->terrain_primary_fields =
            prepare_terrain_aware_primary_fields(
                storage, prepared->terrain_commit.geometry.metrics,
                environmental_tke_profile);
        prepared->microphysics = prepare_terrain_aware_microphysics_recycling(
            state, prepared->plan, prepared->terrain_primary_fields, prepared->terrain_commit.geometry.metrics);
        prepared->exchange =
            compute_terrain_aware_domain_recycling_control_volume_exchange_ledger(
                state, prepared->plan,
                prepared->terrain_primary_fields,
                prepared->terrain_commit.geometry.metrics,
                prepared->pre_shift_frame, &prepared->microphysics);
    }
    else
    {
        prepared->primary_profiles =
            prepare_flat_homogeneous_primary_profiles(
                storage, environmental_tke_profile);
        prepared->microphysics = prepare_flat_homogeneous_microphysics_recycling(
            state, prepared->plan, prepared->primary_profiles);
        prepared->exchange =
            compute_domain_recycling_control_volume_exchange_ledger(
                state, prepared->plan, prepared->primary_profiles,
                prepared->pre_shift_frame, &prepared->microphysics);
    }
    if (storage.nested_execution.relationship_ready)
        prepared->nested_grid = prepare_nested_grid_domain_recycling(
            storage, prepared->plan, prepared->post_shift_frame, &prepared->exchange);

    prepared->radiation = active_terrain
        ? prepare_terrain_aware_radiation_recycling(
            state, prepared->plan, prepared->terrain_primary_fields,
            prepared->terrain_commit.geometry.metrics)
        : prepare_flat_homogeneous_radiation_recycling(
            state, prepared->plan, prepared->primary_profiles);
    prepared->boundary_layer = active_terrain
        ? prepare_terrain_aware_boundary_layer_recycling(
            state, prepared->plan, prepared->terrain_primary_fields,
            prepared->terrain_commit.geometry.metrics)
        : prepare_flat_homogeneous_boundary_layer_recycling(
            state, prepared->plan, prepared->primary_profiles);
    prepared->storm_tracking = prepare_storm_tracking_domain_recycling(
        state, prepared->plan, decision);

    const std::string chaos_scheme =
        state.chaos_runtime != nullptr && state.chaos_runtime->scheme != nullptr
            ? state.chaos_runtime->scheme->name() : std::string{};
    const bool evolving_chaos = chaos_scheme == "boundary_layer"
        || chaos_scheme == "full_stochastic";
    if (evolving_chaos)
        prepared->evolving_chaos =
            prepare_evolving_chaos_domain_recycling(
                state, prepared->plan);

    const auto primary_bindings = active_terrain
        ? bind_terrain_aware_primary_fields(
            storage, prepared->terrain_primary_fields)
        : bind_homogeneous_primary_fields(
            storage, prepared->primary_profiles);
    prepared->volume_bindings = primary_bindings;
    prepared->primary_trailing.resize(primary_bindings.size());
    for (std::size_t index = 0; index < primary_bindings.size(); ++index)
    {
        const auto owner_plan =
            plan_whole_cell_shift_for_horizontal_location(
                prepared->plan,
                primary_bindings[index].horizontal_location);
        prepared->primary_trailing[index].assign(
            owner_plan.trailing_cells, 0.0f);
        prepared->trailing_field_bindings.push_back({
            primary_bindings[index].field_name,
            primary_bindings[index].field,
            &prepared->primary_trailing[index],
            primary_bindings[index].horizontal_location});
    }

    if (prepared->radiation.active)
    {
        for (auto& values : prepared->radiation_trailing)
            values.assign(prepared->plan.trailing_cells, 0.0f);
        (void)append_radiation_trailing_export_bindings(
            state, prepared->radiation_trailing[0],
            prepared->radiation_trailing[1],
            prepared->trailing_field_bindings);
    }
    if (prepared->boundary_layer.active)
    {
        for (auto& values : prepared->boundary_layer_trailing)
            values.assign(prepared->plan.trailing_cells, 0.0f);
        (void)append_boundary_layer_trailing_export_bindings(
            state, prepared->boundary_layer_trailing,
            prepared->trailing_field_bindings);
    }
    const bool microphysics_residual = state.microphysics_runtime != nullptr
        && !state.microphysics_runtime->theta_storage_residual.empty();
    const std::size_t microphysics_capture_count =
        prepared->microphysics.private_scalar_names.size()
        + (microphysics_residual ? 1U : 0U);
    prepared->microphysics_trailing.resize(microphysics_capture_count);
    for (auto& values : prepared->microphysics_trailing)
        values.assign(prepared->plan.trailing_cells, 0.0f);
    (void)append_prepared_microphysics_trailing_export_bindings(
        state, prepared->microphysics_trailing,
        prepared->trailing_field_bindings);

    const std::size_t trailing_columns = prepared->plan.trailing_cells
        / static_cast<std::size_t>(prepared->plan.nz);
    for (auto& values : prepared->precipitation_trailing)
        values.assign(trailing_columns, 0.0f);
    (void)append_surface_precipitation_trailing_export_bindings(
        state, prepared->precipitation_trailing[0],
        prepared->precipitation_trailing[1],
        prepared->precipitation_trailing[2],
        prepared->trailing_surface_bindings);

    (void)append_radiation_domain_recycling_bindings(
        state, prepared->plan, prepared->radiation,
        prepared->volume_bindings, prepared->scalar_bindings);
    (void)append_boundary_layer_domain_recycling_bindings(
        state, prepared->plan, prepared->boundary_layer,
        prepared->volume_bindings, prepared->surface_bindings,
        prepared->record_bindings, prepared->flag_bindings);
    (void)append_prepared_microphysics_domain_recycling_bindings(
        state, prepared->plan, prepared->microphysics,
        prepared->volume_bindings);
    (void)append_surface_precipitation_domain_recycling_bindings(
        state, prepared->surface_bindings);
    (void)append_microphysics_domain_recycling_invalidation_binding(
        state, prepared->runtime_invalidation_bindings);
    (void)append_turbulence_domain_recycling_bindings(
        state, prepared->scalar_bindings,
        prepared->runtime_invalidation_bindings);
    if (evolving_chaos)
    {
        (void)append_evolving_chaos_pbl_cache_domain_recycling_bindings(
            state, prepared->boundary_layer, prepared->evolving_chaos,
            prepared->volume_bindings, prepared->surface_bindings);
        (void)append_evolving_chaos_domain_recycling_bindings(
            state, prepared->evolving_chaos,
            prepared->runtime_commit_bindings);
    }
    else
        (void)append_non_evolving_chaos_domain_recycling_bindings(
            state, prepared->runtime_invalidation_bindings);
    (void)append_rayleigh_domain_recycling_bindings(
        state, prepared->flag_bindings);
    (void)append_advection_domain_recycling_bindings(
        state, prepared->runtime_invalidation_bindings);
    (void)append_diffusion_domain_recycling_bindings(
        state, prepared->runtime_invalidation_bindings);
    (void)append_time_stepping_domain_recycling_bindings(
        state, prepared->runtime_invalidation_bindings);
    (void)append_radar_domain_recycling_bindings(
        state, prepared->runtime_invalidation_bindings);
    (void)append_derived_diagnostic_domain_recycling_bindings(
        state, prepared->runtime_invalidation_bindings);
    (void)append_dynamics_domain_recycling_bindings(
        state, prepared->runtime_invalidation_bindings);
    (void)append_storm_tracking_domain_recycling_binding(
        state, prepared->storm_tracking,
        prepared->runtime_commit_bindings);
    if (prepared->nested_grid.active)
        (void)append_nested_grid_domain_recycling_binding(
            storage, prepared->nested_grid,
            prepared->runtime_commit_bindings);
    if (active_terrain)
        prepared->runtime_commit_bindings.push_back({
            "terrain.earth_fixed_geometry_and_placement", prepared.get(),
            commit_prepared_earth_fixed_terrain_recycling});

    prepared->scalar_bindings.push_back({
        "domain_frame.origin_x",
        &storage.configuration.domain_frame.earth_origin_at_t0_m.x_m,
        prepared->post_shift_frame.earth_origin_at_t0_m.x_m});
    prepared->scalar_bindings.push_back({
        "domain_frame.origin_y",
        &storage.configuration.domain_frame.earth_origin_at_t0_m.y_m,
        prepared->post_shift_frame.earth_origin_at_t0_m.y_m});

    prepared->trailing_field_report = export_trailing_fields(
        prepared->plan, prepared->trailing_field_bindings);
    prepared->trailing_surface_report = export_trailing_surfaces(
        prepared->plan, prepared->trailing_surface_bindings);
    return prepared;
}

std::unique_ptr<PreparedFlatHomogeneousCpuDomainRecycling>
prepare_flat_homogeneous_cpu_domain_recycling(
    SimulationStorageState& storage,
    const StormWindowMovementDecision& decision,
    const std::vector<double>& environmental_tke_profile)
{
    if (storage.runtime.compute.active_kind != ComputeBackendKind::Cpu)
        throw std::invalid_argument(
            "live CPU D5 recycling requires CPU-authoritative model state");
    return prepare_flat_homogeneous_domain_recycling(
        storage, decision, environmental_tke_profile);
}

void append_flat_homogeneous_cpu_domain_recycling_provenance(
    PreparedFlatHomogeneousCpuDomainRecycling& prepared,
    std::string_view transaction_id,
    std::string_view publication_generation,
    std::string_view storage_format,
    std::string_view swath_algorithm,
    double simulation_time_s,
    std::uint64_t transaction_sequence,
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64)
{
    if (prepared.authoritative_backend != ComputeBackendKind::Cpu)
        throw std::invalid_argument(
            "D5 CPU provenance requires CPU-authoritative prepared state");
    append_flat_homogeneous_domain_recycling_provenance(
        prepared, transaction_id, publication_generation, storage_format,
        swath_algorithm, simulation_time_s, transaction_sequence,
        previous_chain_fnv1a64, event_manifest_bytes,
        event_manifest_fnv1a64);
}

void append_flat_homogeneous_domain_recycling_provenance(
    PreparedFlatHomogeneousDomainRecycling& prepared,
    std::string_view transaction_id,
    std::string_view publication_generation,
    std::string_view storage_format,
    std::string_view swath_algorithm,
    double simulation_time_s,
    std::uint64_t transaction_sequence,
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64)
{
    if (prepared.storage == nullptr || prepared.committed
        || prepared.provenance_appended)
        throw std::invalid_argument(
            "D5 provenance cannot be appended to this prepared transaction");
    SimulationStateView state =
        make_simulation_state_view(*prepared.storage);
    prepared.provenance = prepare_domain_recycling_transaction_provenance(
        state, prepared.plan, prepared.pre_shift_frame,
        prepared.exchange, transaction_id, publication_generation,
        storage_format, swath_algorithm, simulation_time_s,
        transaction_sequence, previous_chain_fnv1a64,
        event_manifest_bytes, event_manifest_fnv1a64);
    (void)append_domain_recycling_transaction_provenance_binding(
        state, prepared.provenance, prepared.runtime_commit_bindings);
    prepared.provenance_appended = true;
}

HomogeneousWindowShiftReport commit_flat_homogeneous_cpu_domain_recycling(
    PreparedFlatHomogeneousCpuDomainRecycling& prepared)
{
    if (prepared.authoritative_backend != ComputeBackendKind::Cpu)
        throw std::invalid_argument(
            "D5 CPU commit requires CPU-authoritative prepared state");
    return commit_flat_homogeneous_domain_recycling(prepared);
}

HomogeneousWindowShiftReport commit_flat_homogeneous_domain_recycling(
    PreparedFlatHomogeneousDomainRecycling& prepared)
{
    if (prepared.storage == nullptr || prepared.committed
        || !prepared.provenance_appended)
        throw std::invalid_argument(
            "D5 commit requires one unconsumed provenance-bound transaction");
    const auto& live_frame = prepared.storage->configuration.domain_frame;
    if (live_frame.earth_origin_at_t0_m.x_m
            != prepared.pre_shift_frame.earth_origin_at_t0_m.x_m
        || live_frame.earth_origin_at_t0_m.y_m
            != prepared.pre_shift_frame.earth_origin_at_t0_m.y_m
        || live_frame.velocity_ms.u_ms
            != prepared.pre_shift_frame.velocity_ms.u_ms
        || live_frame.velocity_ms.v_ms
            != prepared.pre_shift_frame.velocity_ms.v_ms)
        throw std::invalid_argument(
            "D5 commit pre-shift frame changed after preparation");
    SimulationStateView state = make_simulation_state_view(*prepared.storage);
    if (state.compute_runtime == nullptr
        || state.compute_runtime->active_kind != prepared.authoritative_backend)
        throw std::invalid_argument(
            "D5 authoritative backend changed after preparation");
    if (prepared.earth_fixed_terrain
        && !can_commit_earth_fixed_terrain(
            prepared.terrain_commit, prepared.storage->runtime.terrain))
        throw std::invalid_argument(
            "D5 Earth-fixed terrain changed after preparation");
    if (prepared.nested_grid.active
        && !can_commit_nested_grid_domain_recycling(
            *prepared.storage, prepared.nested_grid))
        throw std::invalid_argument(
            "D5 nested-grid owner changed after preparation");

    HomogeneousWindowShiftReport report;
    if (prepared.authoritative_backend == ComputeBackendKind::Cpu)
    {
        report = execute_homogeneous_window_shift(
            prepared.plan, prepared.volume_bindings,
            prepared.surface_bindings, prepared.scalar_bindings,
            prepared.record_bindings, prepared.flag_bindings,
            prepared.runtime_invalidation_bindings,
            prepared.runtime_commit_bindings);
    }
    else
    {
        ComputeBackend* backend = state.compute_runtime->backend.get();
        if (backend == nullptr
            || !backend->supports_homogeneous_domain_field_shift())
            throw std::invalid_argument(
                "D5 accelerator commit requires a native shift pipeline");

        // Finish every throwing owner validation and report calculation before
        // the device can modify a resident allocation.
        report = preflight_homogeneous_window_shift(
            prepared.plan, prepared.volume_bindings,
            prepared.surface_bindings, prepared.scalar_bindings,
            prepared.record_bindings, prepared.flag_bindings,
            prepared.runtime_invalidation_bindings,
            prepared.runtime_commit_bindings);

        std::vector<HomogeneousDomainFieldShiftDispatch> dispatches;
        std::vector<std::string> resident_names;
        std::vector<const char*> resident_ids;
        dispatches.reserve(prepared.volume_bindings.size());
        resident_names.reserve(prepared.volume_bindings.size());
        resident_ids.reserve(prepared.volume_bindings.size());
        for (const auto& binding : prepared.volume_bindings)
        {
            if ((binding.leading_profile == nullptr) == (binding.leading_field == nullptr))
                throw std::invalid_argument(
                    "D5 accelerator shift requires one leading profile or field");
            const WholeCellShiftPlan owner_plan =
                plan_whole_cell_shift_for_horizontal_location(
                    prepared.plan, binding.horizontal_location);
            HomogeneousDomainFieldShiftDispatch dispatch;
            dispatch.source = binding.field->data();
            dispatch.leading_profile = binding.leading_profile ? binding.leading_profile->data() : nullptr;
            dispatch.leading_field = binding.leading_field ? binding.leading_field->data() : nullptr;
            dispatch.destination = binding.field->data();
            dispatch.nr = owner_plan.nr;
            dispatch.nth = owner_plan.nth;
            dispatch.nz = owner_plan.nz;
            dispatch.shift_i_cells = owner_plan.shift_i_cells;
            dispatch.shift_j_cells = owner_plan.shift_j_cells;
            dispatch.destination_i_begin = owner_plan.destination_i_begin;
            dispatch.destination_i_end = owner_plan.destination_i_end;
            dispatch.destination_j_begin = owner_plan.destination_j_begin;
            dispatch.destination_j_end = owner_plan.destination_j_end;
            dispatch.source_i_begin = owner_plan.source_i_begin;
            dispatch.source_i_end = owner_plan.source_i_end;
            dispatch.source_j_begin = owner_plan.source_j_begin;
            dispatch.source_j_end = owner_plan.source_j_end;
            dispatches.push_back(dispatch);
            resident_names.emplace_back(binding.field_name);
        }
        for (const std::string& name : resident_names)
            resident_ids.push_back(name.c_str());
        const HomogeneousDomainFieldShiftBatchDispatch batch{
            dispatches.data(), resident_ids.data(),
            static_cast<int>(dispatches.size())};
        if (!backend->dispatch_homogeneous_domain_field_shift_batch(batch))
            throw std::runtime_error(
                "D5 accelerator resident volume transaction failed before host commit");

        // The fence is complete and all volume host mirrors are now current.
        // These remaining bindings were preflighted above and are noexcept.
        commit_preflighted_homogeneous_window_nonvolume(
            prepared.plan, prepared.surface_bindings,
            prepared.scalar_bindings, prepared.record_bindings,
            prepared.flag_bindings,
            prepared.runtime_invalidation_bindings,
            prepared.runtime_commit_bindings);
    }
    prepared.committed = true;
    return report;
}

void verify_flat_homogeneous_cpu_domain_recycling_after_commit(
    const PreparedFlatHomogeneousCpuDomainRecycling& prepared)
{
    if (prepared.authoritative_backend != ComputeBackendKind::Cpu)
        throw std::invalid_argument(
            "D5 CPU verification requires CPU-authoritative prepared state");
    verify_flat_homogeneous_domain_recycling_after_commit(prepared);
}

void verify_flat_homogeneous_domain_recycling_after_commit(
    const PreparedFlatHomogeneousDomainRecycling& prepared)
{
    if (prepared.storage == nullptr || !prepared.committed
        || !prepared.provenance_appended)
        throw std::invalid_argument(
            "D5 verification requires a committed prepared transaction");
    const SimulationStateView state =
        make_simulation_state_view(*prepared.storage);
    verify_domain_recycling_control_volume_exchange_after_commit(
        state, prepared.plan,
        prepared.storage->configuration.domain_frame,
        prepared.exchange);
    const auto& frame = prepared.storage->configuration.domain_frame;
    if (frame.earth_origin_at_t0_m.x_m
            != prepared.post_shift_frame.earth_origin_at_t0_m.x_m
        || frame.earth_origin_at_t0_m.y_m
            != prepared.post_shift_frame.earth_origin_at_t0_m.y_m
        || frame.velocity_ms.u_ms != prepared.post_shift_frame.velocity_ms.u_ms
        || frame.velocity_ms.v_ms != prepared.post_shift_frame.velocity_ms.v_ms)
        throw std::runtime_error(
            "D5 post-commit Earth frame does not match preparation");
    if (prepared.earth_fixed_terrain)
    {
        const auto& terrain = prepared.storage->runtime.terrain;
        const auto expected_origin = frame.earth_origin_at(
            prepared.storage->time_s);
        if (!prepared.terrain_commit.committed
            || !terrain.earth_fixed_placement.active
            || terrain.earth_fixed_placement.grid_origin_earth_m.x_m
                != expected_origin.x_m
            || terrain.earth_fixed_placement.grid_origin_earth_m.y_m
                != expected_origin.y_m
            || terrain.earth_fixed_placement.generation
                != prepared.terrain_commit.next_placement.generation)
            throw std::runtime_error(
                "D5 post-commit Earth-fixed terrain placement is inconsistent");
    }
    if (prepared.nested_grid.active)
    {
        if (nested_preserved_state_fingerprint(
                prepared.storage->nested_execution)
                != prepared.nested_grid.preserved_state_fingerprint
            || nested_scheme_identity(prepared.storage->nested_execution)
                != prepared.nested_grid.scheme_identity)
            throw std::runtime_error(
                "D5 post-commit nested state changed outside its parent-index rebase");
    }
}

const char* flat_homogeneous_cpu_domain_recycling_algorithm_id() noexcept
{
    return "prepared_all_owner_flat_cgrid_cpu_transaction_v1";
}

const char* flat_homogeneous_domain_recycling_algorithm_id() noexcept
{
    return "prepared_all_owner_flat_cgrid_transaction_v2";
}

} // namespace tmv::domain
