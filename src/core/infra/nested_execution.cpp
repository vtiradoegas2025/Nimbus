#include "microphysics/grid_geometry.hpp"
#include "core/field/field_inventory.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "core/infra/nested_feedback_interval.hpp"
#include "diagnostics/level_budget_csv.hpp"
#include "core/runtime/atmospheric_level.hpp"
#include "core/infra/nested_execution.hpp"
#include "terrain/conservative_transport.hpp"

#include "core/field/field_bounds.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/infra/physical_constants.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/storm_tracker.hpp"
#include "dynamics/coriolis.hpp"
#include "dynamics/eos_pressure.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

namespace
{

[[noreturn]] void reject_nested_execution(const char* reason)
{
    throw std::invalid_argument(std::string("invalid nested execution relationship: ") + reason);
}

double expected_child_coriolis_f(const SimulationStorageState& storage, const NestedExecutionStorage& nested)
{
    if (storage.configuration.coriolis_mode != tmv::init::config::CoriolisMode::BetaPlane)
        return storage.coriolis_f;
    return tmv::make_cartesian_coriolis_profile(storage.configuration.coriolis_enabled,
                                                true,
                                                storage.configuration.coriolis_latitude_deg,
                                                nested.child.domain_frame,
                                                nested.child.time_s)
        .f_at_grid_y0;
}

std::int64_t checked_shifted_earth_origin(std::int64_t origin, int shift)
{
    const auto shift64 = static_cast<std::int64_t>(shift);
    if ((shift64 > 0 && origin > std::numeric_limits<std::int64_t>::max() - shift64) ||
        (shift64 < 0 && origin < std::numeric_limits<std::int64_t>::min() - shift64))
        reject_nested_execution("stochastic Earth origin overflows");
    return origin + shift64;
}

std::array<char, NestedPrivateScalarRelaxationReportV1::maximum_name_bytes>
bounded_private_scalar_name(const char* name)
{
    if (name == nullptr || *name == '\0')
        reject_nested_execution("private microphysics scalar name is empty");
    std::array<char, NestedPrivateScalarRelaxationReportV1::maximum_name_bytes> stored{};
    const std::size_t length = std::char_traits<char>::length(name);
    if (length >= stored.size())
        reject_nested_execution("private microphysics scalar name exceeds the durable limit");
    std::copy_n(name, length, stored.begin());
    return stored;
}

void require_same_grid_contract(const GridStorageState&                      parent,
                                const tmv::domain::OneWayNestRelationshipV1& relationship)
{
    const auto physical = physical_scalar_domain(
        parent.nr, parent.nth, parent.nz, parent.geometry.staggered ? StaggerType::CGrid : StaggerType::Collocated);
    if (!parent.geometry.staggered || physical.nx() != relationship.parent_cells_x ||
        physical.ny() != relationship.parent_cells_y || parent.nz != relationship.levels_z ||
        parent.dx != relationship.parent_dx_m || parent.dy != relationship.parent_dy_m || !std::isfinite(parent.dt) ||
        parent.dt <= 0.0 || parent.dt > relationship.parent_dt_s)
    {
        reject_nested_execution("parent grid does not match the stored relationship");
    }
}

bool valid_flat_child_process_grid_metrics(const AtmosphericLevelStorage& child) noexcept
{
    const auto& metrics = child.process_grid_metrics;
    if (metrics.dx != child.grid.dx || metrics.dy != child.grid.dy || metrics.terrain_metrics_active ||
        metrics.terrain_metrics != nullptr || metrics.terrain_topography != nullptr ||
        metrics.dz.size() != static_cast<std::size_t>(child.grid.nz) || metrics.z_int != child.grid.geometry.z_faces)
        return false;
    if (child.grid.geometry.z_faces.size() != static_cast<std::size_t>(child.grid.nz + 1))
        return false;
    for (int k = 0; k < child.grid.nz; ++k)
    {
        const double thickness = child.grid.geometry.z_faces[static_cast<std::size_t>(k + 1)] -
                                 child.grid.geometry.z_faces[static_cast<std::size_t>(k)];
        if (!std::isfinite(thickness) || thickness <= 0.0 || metrics.dz[static_cast<std::size_t>(k)] != thickness)
            return false;
    }
    return true;
}

bool nested_active_terrain_geometry_ready(const NestedExecutionStorage& nested) noexcept
{
    return nested.active_terrain.geometry.generation != 0;
}

bool valid_active_child_process_grid_metrics(const NestedExecutionStorage& nested) noexcept
{
    if (!nested_active_terrain_geometry_ready(nested))
        return false;
    const auto& child   = nested.child;
    const auto& process = child.process_grid_metrics;
    const auto& terrain = child.runtime.terrain;
    const auto& mapped  = nested.active_terrain.geometry.child_storage;
    return (terrain.config.scheme_id == "bell" || terrain.config.scheme_id == "schar") && process.dx == child.grid.dx &&
           process.dy == child.grid.dy && process.terrain_metrics_active && process.terrain_metrics != nullptr &&
           process.terrain_topography != nullptr && process.dz.size() == static_cast<std::size_t>(child.grid.nz) &&
           process.z_int == child.grid.geometry.z_faces && terrain.metrics.J.size_r() == child.grid.nr &&
           terrain.metrics.J.size_th() == child.grid.nth && terrain.metrics.J.size_z() == child.grid.nz &&
           terrain.topography.h.size() == static_cast<std::size_t>(child.grid.nr) &&
           terrain.metrics.J.values == mapped.metrics.J.values &&
           terrain.metrics.z_face.values == mapped.metrics.z_face.values && terrain.topography.h == mapped.topography.h;
}

bool valid_child_process_grid_metrics(const NestedExecutionStorage& nested) noexcept
{
    return nested_active_terrain_geometry_ready(nested) ? valid_active_child_process_grid_metrics(nested)
                                                        : valid_flat_child_process_grid_metrics(nested.child);
}

void require_valid_nested_execution_relationship(const SimulationStorageState& storage,
                                                 const NestedExecutionStorage& nested,
                                                 double                        parent_time_s)
{
    if (!nested.relationship_ready)
    {
        if (nested.reference_state_ready || nested.physical_primary_state_ready || nested.parent_boundary_state_ready)
            reject_nested_execution("child readiness cannot exist without a relationship");
        return;
    }

    const auto& relationship = nested.relationship;
    require_same_grid_contract(storage.grid, relationship);
    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto child_physical =
        physical_scalar_domain(nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz, StaggerType::CGrid);
    const double child_time = tmv::domain::nested_child_time_s(relationship, nested.parent_time_bracket);
    const double realized_child_dt =
        (nested.parent_time_bracket.current_parent_time_s - nested.parent_time_bracket.previous_parent_time_s) /
        relationship.parent_time_step_ratio;
    if (!std::isfinite(parent_time_s) || parent_time_s != nested.parent_time_bracket.current_parent_time_s)
    {
        reject_nested_execution("parent clock does not own the current time-bracket endpoint");
    }
    if (!nested.child.grid.geometry.staggered || child_physical.nx() != relationship.child_cells_x ||
        child_physical.ny() != relationship.child_cells_y || nested.child.grid.nz != relationship.levels_z ||
        nested.child.grid.dx != relationship.child_dx_m || nested.child.grid.dy != relationship.child_dy_m ||
        nested.child.grid.dz != storage.grid.dz || nested.child.grid.geometry.z != storage.grid.geometry.z ||
        nested.child.grid.geometry.z_faces != storage.grid.geometry.z_faces ||
        nested.child.grid.dt != realized_child_dt || nested.child.time_s != child_time)
    {
        reject_nested_execution("child grid or clock does not match the stored relationship");
    }
    if (!valid_child_process_grid_metrics(nested))
        reject_nested_execution("child process grid metrics do not match its exact geometry");
    if (nested_active_terrain_geometry_ready(nested))
    {
        const auto& active = nested.active_terrain;
        const auto& mapped = active.geometry;
        if (mapped.relationship.child_cells_x != relationship.child_cells_x ||
            mapped.relationship.child_cells_y != relationship.child_cells_y ||
            mapped.relationship.child_origin_parent_cell_i != relationship.child_origin_parent_cell_i ||
            mapped.relationship.child_origin_parent_cell_j != relationship.child_origin_parent_cell_j ||
            (mapped.config.scheme_id != "bell" && mapped.config.scheme_id != "schar") ||
            nested.child.runtime.terrain.config.scheme_id != mapped.config.scheme_id)
            reject_nested_execution("active child terrain identity differs from the nesting relationship");
        if (active.ready)
        {
            const auto shape = [&](const tmv::TerrainField3D& field, int nx, int ny, int nz)
            {
                return field.size_r() == nx && field.size_th() == ny && field.size_z() == nz &&
                       std::all_of(
                           field.values.begin(), field.values.end(), [](double value) { return std::isfinite(value); });
            };
            if (active.generation == 0 || active.publication_count == 0 || active.child_time_s != nested.child.time_s ||
                active.ledger.child_time_s != nested.child.time_s ||
                !shape(active.child_state.scalar,
                       relationship.child_cells_x,
                       relationship.child_cells_y,
                       relationship.levels_z) ||
                !shape(active.child_state.x_flux,
                       relationship.child_cells_x + 1,
                       relationship.child_cells_y,
                       relationship.levels_z) ||
                !shape(active.child_state.y_flux,
                       relationship.child_cells_x,
                       relationship.child_cells_y + 1,
                       relationship.levels_z) ||
                !shape(active.child_state.zeta_flux,
                       relationship.child_cells_x,
                       relationship.child_cells_y,
                       relationship.levels_z + 1))
                reject_nested_execution("active child terrain state, clock, or ledger is incomplete");
        }
    }
    else if (nested.active_terrain.ready || nested.active_terrain.generation != 0 ||
             nested.active_terrain.publication_count != 0)
        reject_nested_execution("active child terrain state exists without mapped geometry");
    if (nested.reference_state_ready &&
        (nested.child.base.rho0 != storage.base.rho0 || nested.child.base.p0 != storage.base.p0 ||
         nested.child.base.qv0 != storage.base.qv0 || nested.child.base.theta0 != storage.base.theta0 ||
         nested.child.base.u0 != storage.base.u0 || nested.child.base.v0 != storage.base.v0 ||
         nested.child.coriolis_f != expected_child_coriolis_f(storage, nested)))
    {
        reject_nested_execution("child shared-level base state or Coriolis Earth-reference is invalid");
    }
    if (nested.physical_primary_state_ready)
    {
        const auto shape_matches = [&](const Field3D& field)
        {
            return field.size_r() == nested.child.grid.nr && field.size_th() == nested.child.grid.nth &&
                   field.size_z() == nested.child.grid.nz;
        };
        const auto& fields = nested.child.fields;
        if (!nested.reference_state_ready || !shape_matches(fields.rho) || !shape_matches(fields.p) ||
            !shape_matches(fields.u) || !shape_matches(fields.v) || !shape_matches(fields.w) ||
            !shape_matches(fields.theta) || !shape_matches(fields.qv) || !shape_matches(fields.qc) ||
            !shape_matches(fields.qr) || !shape_matches(fields.qi) || !shape_matches(fields.qs) ||
            !shape_matches(fields.qg) || !shape_matches(fields.qh) || !shape_matches(fields.tracer) ||
            !shape_matches(fields.tke))
        {
            reject_nested_execution("physical child primary state is incomplete or mis-shaped");
        }
    }
    if (nested.parent_boundary_state_ready && !nested.physical_primary_state_ready)
        reject_nested_execution("parent boundary readiness requires physical primary state");
    if (nested.parent_boundary_state_ready &&
        (!nested.interface_source.available || nested.interface_source.application_count == 0))
        reject_nested_execution("parent boundary readiness requires interface source provenance");
    if (nested.parent_boundary_state_ready && nested_active_terrain_geometry_ready(nested))
    {
        for (const auto [field, name] : {std::pair<const Field3D*, const char*>{&nested.child.fields.rho, "rho"},
                                         {&nested.child.fields.u, "u"},
                                         {&nested.child.fields.v, "v"},
                                         {&nested.child.fields.w, "w"}})
            for (std::size_t index = 0; index < field->size(); ++index)
                if (!std::isfinite(field->data()[index]) || (name[0] == 'r' && field->data()[index] <= 0.0f))
                    throw std::invalid_argument(std::string("invalid nested execution relationship: mapped child ") +
                                                name + " closure is non-finite or non-positive at value " +
                                                std::to_string(index));
    }
    if (!nested.parent_boundary_state_ready &&
        (nested.interface_source.available || nested.interface_source.application_count != 0))
        reject_nested_execution("interface source provenance requires parent boundary readiness");
    const auto provenance_i = static_cast<std::int64_t>(nested.initial_parent_origin_i) +
                              static_cast<std::int64_t>(nested.cumulative_parent_shift_i) -
                              static_cast<std::int64_t>(nested.cumulative_parent_recycling_shift_i);
    const auto provenance_j = static_cast<std::int64_t>(nested.initial_parent_origin_j) +
                              static_cast<std::int64_t>(nested.cumulative_parent_shift_j) -
                              static_cast<std::int64_t>(nested.cumulative_parent_recycling_shift_j);
    if (provenance_i != relationship.child_origin_parent_cell_i ||
        provenance_j != relationship.child_origin_parent_cell_j ||
        (nested.movement_sequence == 0 &&
         (nested.cumulative_parent_shift_i != 0 || nested.cumulative_parent_shift_j != 0)))
    {
        reject_nested_execution("child placement disagrees with its movement provenance");
    }

    storage.configuration.domain_frame.validate();
    nested.child.domain_frame.validate();
    const auto&  parent_frame = storage.configuration.domain_frame;
    const auto&  child_frame  = nested.child.domain_frame;
    const double expected_x =
        parent_frame.earth_origin_at_t0_m.x_m +
        (parent_physical.i_begin + relationship.child_origin_parent_cell_i) * relationship.parent_dx_m -
        child_physical.i_begin * relationship.child_dx_m;
    const double expected_y =
        parent_frame.earth_origin_at_t0_m.y_m +
        (parent_physical.j_begin + relationship.child_origin_parent_cell_j) * relationship.parent_dy_m -
        child_physical.j_begin * relationship.child_dy_m;
    if (child_frame.earth_origin_at_t0_m.x_m != expected_x || child_frame.earth_origin_at_t0_m.y_m != expected_y ||
        child_frame.velocity_ms.u_ms != parent_frame.velocity_ms.u_ms ||
        child_frame.velocity_ms.v_ms != parent_frame.velocity_ms.v_ms)
    {
        reject_nested_execution("child Earth frame does not match its parent-cell placement");
    }
}

void add_scalar_source(tmv::domain::ConservedScalarRelaxationReport&       destination,
                       const tmv::domain::ConservedScalarRelaxationReport& source)
{
    destination.target_dry_mass_change_kg += source.target_dry_mass_change_kg;
    destination.committed_dry_mass_change_kg += source.committed_dry_mass_change_kg;
    destination.dry_storage_residual_kg =
        destination.target_dry_mass_change_kg - destination.committed_dry_mass_change_kg;
    destination.target_carrier_change += source.target_carrier_change;
    destination.committed_carrier_change += source.committed_carrier_change;
    destination.carrier_storage_residual = destination.target_carrier_change - destination.committed_carrier_change;
    destination.forced_cells += source.forced_cells;
}

void add_momentum_source(tmv::domain::ConservedMomentumRelaxationReport&       destination,
                         const tmv::domain::ConservedMomentumRelaxationReport& source)
{
    destination.target_x_change_kgms += source.target_x_change_kgms;
    destination.committed_x_change_kgms += source.committed_x_change_kgms;
    destination.x_storage_residual_kgms = destination.target_x_change_kgms - destination.committed_x_change_kgms;
    destination.target_y_change_kgms += source.target_y_change_kgms;
    destination.committed_y_change_kgms += source.committed_y_change_kgms;
    destination.y_storage_residual_kgms = destination.target_y_change_kgms - destination.committed_y_change_kgms;
    destination.target_z_change_kgms += source.target_z_change_kgms;
    destination.committed_z_change_kgms += source.committed_z_change_kgms;
    destination.z_storage_residual_kgms = destination.target_z_change_kgms - destination.committed_z_change_kgms;
}

void add_energy_change(MoistMixtureEnergyBudgetV2& destination, const MoistMixtureEnergyBudgetV2& source)
{
    destination.dry_internal += source.dry_internal;
    destination.water_sensible += source.water_sensible;
    destination.phase_reference += source.phase_reference;
    destination.resolved_kinetic += source.resolved_kinetic;
    destination.gravitational += source.gravitational;
    destination.subgrid_tke += source.subgrid_tke;
    destination.total_water_mass += source.total_water_mass;
    destination.total += source.total;
}

void accumulate_interface_source(NestedInterfaceSourceLedgerV2& ledger, const NestedChildRelaxationReport& source)
{
    if (!ledger.available)
    {
        ledger.available         = true;
        ledger.application_count = 1;
        ledger.last              = source;
        ledger.cumulative        = source;
        return;
    }
    NestedChildRelaxationReport*                        cumulative     = &ledger.cumulative;
    tmv::domain::ConservedScalarRelaxationReport*       destinations[] = {&cumulative->theta,
                                                                          &cumulative->qv,
                                                                          &cumulative->qc,
                                                                          &cumulative->qr,
                                                                          &cumulative->qi,
                                                                          &cumulative->qs,
                                                                          &cumulative->qg,
                                                                          &cumulative->qh,
                                                                          &cumulative->tracer,
                                                                          &cumulative->tke};
    const tmv::domain::ConservedScalarRelaxationReport* sources[]      = {&source.theta,
                                                                          &source.qv,
                                                                          &source.qc,
                                                                          &source.qr,
                                                                          &source.qi,
                                                                          &source.qs,
                                                                          &source.qg,
                                                                          &source.qh,
                                                                          &source.tracer,
                                                                          &source.tke};
    for (std::size_t index = 0; index < std::size(destinations); ++index)
        add_scalar_source(*destinations[index], *sources[index]);
    if (cumulative->microphysics_private.scalar_count != source.microphysics_private.scalar_count)
        reject_nested_execution("microphysics private interface ledgers have different layouts");
    for (std::size_t index = 0; index < source.microphysics_private.scalar_count; ++index)
    {
        auto&       destination = cumulative->microphysics_private;
        const auto& addition    = source.microphysics_private;
        if (destination.name[index] != addition.name[index] ||
            destination.storage_semantic[index] != addition.storage_semantic[index] ||
            destination.positive_definite[index] != addition.positive_definite[index])
            reject_nested_execution("microphysics private interface ledger identity changed");
        destination.target_carrier_change[index] += addition.target_carrier_change[index];
        destination.committed_carrier_change[index] += addition.committed_carrier_change[index];
        destination.carrier_storage_residual[index] =
            destination.target_carrier_change[index] - destination.committed_carrier_change[index];
        if (addition.forced_cells[index] > std::numeric_limits<std::size_t>::max() - destination.forced_cells[index])
            reject_nested_execution("microphysics private interface forced-cell count overflows");
        destination.forced_cells[index] += addition.forced_cells[index];
    }
    add_momentum_source(cumulative->momentum, source.momentum);
    cumulative->pressure     = source.pressure;
    cumulative->energy.after = source.energy.after;
    add_energy_change(cumulative->energy.committed_change, source.energy.committed_change);
    const auto& energy = cumulative->energy.committed_change;
    cumulative->energy.reservoir_sum_closure_residual_j =
        energy.total - (energy.dry_internal + energy.water_sensible + energy.phase_reference + energy.resolved_kinetic +
                        energy.gravitational + energy.subgrid_tke);
    ledger.last = source;
    if (ledger.application_count == std::numeric_limits<std::uint64_t>::max())
        reject_nested_execution("nested interface application count overflows");
    ++ledger.application_count;
}

struct NestedLivePrivateMicrophysicsBindings
{
    tmv::MicrophysicsPrognosticScalarList current_parent;
    tmv::MicrophysicsPrognosticScalarList child;
};

NestedLivePrivateMicrophysicsBindings
require_private_microphysics_bindings(SimulationStorageState&                   storage,
                                      const NestedParentMicrophysicsSnapshotV1& previous)
{
    auto& nested          = storage.nested_execution;
    auto& current_runtime = storage.runtime.microphysics;
    auto& child_runtime   = nested.child.runtime.microphysics;
    if (!previous.available || !current_runtime.scheme || !child_runtime.scheme ||
        previous.sample_time_s != nested.parent_time_bracket.previous_parent_time_s)
        reject_nested_execution("private microphysics forcing is missing an exact previous endpoint");
    const auto current_name = bounded_private_scalar_name(current_runtime.scheme->get_scheme_name().c_str());
    const auto child_name   = bounded_private_scalar_name(child_runtime.scheme->get_scheme_name().c_str());
    if (previous.scheme_name != current_name || child_name != current_name)
        reject_nested_execution("private microphysics forcing requires one scheme selection");

    NestedLivePrivateMicrophysicsBindings bindings;
    bindings.current_parent = current_runtime.scheme->prognostic_scalars_for_transport();
    bindings.child          = child_runtime.scheme->prognostic_scalars_for_transport();
    if (previous.scalar_count != bindings.current_parent.size() ||
        bindings.child.size() != bindings.current_parent.size() ||
        bindings.child.size() > NestedPrivateScalarRelaxationReportV1::maximum_scalars)
        reject_nested_execution("private microphysics forcing layouts differ");

    const auto parent_shape = [&](const Field3D& field)
    {
        return field.size_r() == storage.grid.nr && field.size_th() == storage.grid.nth &&
               field.size_z() == storage.grid.nz;
    };
    const auto child_shape = [&](const Field3D& field)
    {
        return field.size_r() == nested.child.grid.nr && field.size_th() == nested.child.grid.nth &&
               field.size_z() == nested.child.grid.nz;
    };
    for (std::size_t index = 0; index < bindings.child.size(); ++index)
    {
        const auto& current       = bindings.current_parent[index];
        const auto& child         = bindings.child[index];
        const auto  expected_name = bounded_private_scalar_name(current.name);
        if (!current.field || !child.field || previous.name[index] != expected_name ||
            bounded_private_scalar_name(child.name) != expected_name ||
            previous.storage_semantic[index] != static_cast<std::uint8_t>(current.storage) ||
            child.storage != current.storage || previous.positive_definite[index] != current.positive_definite ||
            child.positive_definite != current.positive_definite || !parent_shape(previous.fields[index]) ||
            !parent_shape(*current.field) || !child_shape(*child.field))
            reject_nested_execution("private microphysics forcing binding is inconsistent");
        if (current.storage != tmv::MicrophysicsScalarStorage::DryAirMixingRatio)
            reject_nested_execution(
                "volume-concentration private forcing requires its density-independent conserved operator");
        for (std::size_t prior = 0; prior < index; ++prior)
            if (previous.name[prior] == expected_name)
                reject_nested_execution("private microphysics forcing names are not unique");
    }
    return bindings;
}

NestedLivePrivateMicrophysicsBindings
require_simultaneous_private_microphysics_bindings(SimulationStorageState&  storage,
                                                   tmv::MicrophysicsScheme& prepared_child_scheme)
{
    auto& parent_runtime = storage.runtime.microphysics;
    if (!parent_runtime.scheme || parent_runtime.scheme->get_scheme_name() != prepared_child_scheme.get_scheme_name())
        reject_nested_execution("simultaneous private microphysics forcing requires one scheme selection");

    NestedLivePrivateMicrophysicsBindings bindings;
    bindings.current_parent = parent_runtime.scheme->prognostic_scalars_for_transport();
    bindings.child          = prepared_child_scheme.prognostic_scalars_for_transport();
    if (bindings.current_parent.size() != bindings.child.size() ||
        bindings.child.size() > NestedPrivateScalarRelaxationReportV1::maximum_scalars)
        reject_nested_execution("simultaneous private microphysics forcing layouts differ");

    const auto parent_shape = [&](const Field3D& field)
    {
        return field.size_r() == storage.grid.nr && field.size_th() == storage.grid.nth &&
               field.size_z() == storage.grid.nz;
    };
    const auto child_shape = [&](const Field3D& field)
    {
        const auto& grid = storage.nested_execution.child.grid;
        return field.size_r() == grid.nr && field.size_th() == grid.nth && field.size_z() == grid.nz;
    };
    for (std::size_t index = 0; index < bindings.child.size(); ++index)
    {
        const auto& parent        = bindings.current_parent[index];
        const auto& child         = bindings.child[index];
        const auto  expected_name = bounded_private_scalar_name(parent.name);
        if (!parent.field || !child.field || bounded_private_scalar_name(child.name) != expected_name ||
            child.storage != parent.storage || child.positive_definite != parent.positive_definite ||
            !parent_shape(*parent.field) || !child_shape(*child.field))
            reject_nested_execution("simultaneous private microphysics binding is inconsistent");
        if (parent.storage != tmv::MicrophysicsScalarStorage::DryAirMixingRatio)
            reject_nested_execution(
                "volume-concentration private forcing requires its density-independent conserved operator");
        for (std::size_t prior = 0; prior < index; ++prior)
            if (bounded_private_scalar_name(bindings.current_parent[prior].name) == expected_name)
                reject_nested_execution("simultaneous private microphysics names are not unique");
    }
    return bindings;
}

void fill_private_mixing_ratio_closure(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                       const tmv::domain::ParentTimeBracketV1&      bracket,
                                       const Field3D&                               previous_parent_rho,
                                       const Field3D&                               current_parent_rho,
                                       const Field3D&                               previous_parent_specific,
                                       const Field3D&                               current_parent_specific,
                                       bool                                         positive_definite,
                                       Field3D&                                     child_specific)
{
    const auto weights   = tmv::domain::parent_temporal_weights(relationship, bracket);
    const auto floor_div = [](int value, int divisor)
    { return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor); };
    for (int si = 0; si < relationship.child_cells_x + 2; ++si)
        for (int sj = 0; sj < relationship.child_cells_y + 2; ++sj)
        {
            if (si >= 1 && si <= relationship.child_cells_x && sj >= 1 && sj <= relationship.child_cells_y)
                continue;
            const int pi = relationship.child_origin_parent_cell_i + floor_div(si - 1, relationship.parent_ratio_x);
            const int pj = relationship.child_origin_parent_cell_j + floor_div(sj - 1, relationship.parent_ratio_y);
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                double rho     = 0.0;
                double carrier = 0.0;
                if (weights.current == 0.0)
                {
                    rho     = previous_parent_rho[pi + 1][pj + 1][k];
                    carrier = rho * previous_parent_specific[pi + 1][pj + 1][k];
                }
                else if (weights.previous == 0.0)
                {
                    rho     = current_parent_rho[pi + 1][pj + 1][k];
                    carrier = rho * current_parent_specific[pi + 1][pj + 1][k];
                }
                else
                {
                    const double previous_rho = previous_parent_rho[pi + 1][pj + 1][k];
                    const double current_rho  = current_parent_rho[pi + 1][pj + 1][k];
                    rho                       = weights.previous * previous_rho + weights.current * current_rho;
                    carrier = weights.previous * previous_rho * previous_parent_specific[pi + 1][pj + 1][k] +
                              weights.current * current_rho * current_parent_specific[pi + 1][pj + 1][k];
                }
                const double specific = carrier / rho;
                const float  stored   = static_cast<float>(specific);
                if (!std::isfinite(rho) || rho <= 0.0 || !std::isfinite(carrier) || !std::isfinite(specific) ||
                    !std::isfinite(stored) || (positive_definite && (carrier < 0.0 || specific < 0.0)))
                    reject_nested_execution("private microphysics closure target is invalid");
                child_specific[si][sj][k] = stored;
            }
        }
}

tmv::domain::OneWayNestRelationshipV1
shifted_relationship(const tmv::domain::OneWayNestRelationshipV1& relationship, int origin_i, int origin_j)
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = relationship.parent_cells_x;
    input.parent_cells_y                = relationship.parent_cells_y;
    input.parent_levels_z               = relationship.levels_z;
    input.parent_dx_m                   = relationship.parent_dx_m;
    input.parent_dy_m                   = relationship.parent_dy_m;
    input.parent_dt_s                   = relationship.parent_dt_s;
    input.parent_ratio_x                = relationship.parent_ratio_x;
    input.parent_ratio_y                = relationship.parent_ratio_y;
    input.parent_time_step_ratio        = relationship.parent_time_step_ratio;
    input.child_origin_parent_cell_i    = origin_i;
    input.child_origin_parent_cell_j    = origin_j;
    input.child_cells_x                 = relationship.child_cells_x;
    input.child_cells_y                 = relationship.child_cells_y;
    input.child_levels_z                = relationship.levels_z;
    input.relaxation_width_child_cells  = relationship.relaxation_width_child_cells;
    input.boundary_stencil_parent_cells = relationship.boundary_stencil_parent_cells;
    return tmv::domain::prepare_one_way_nest_relationship(input);
}

void place_child_frame_from_relationship(const SimulationStorageState& storage, NestedExecutionStorage& nested)
{
    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto child_physical =
        physical_scalar_domain(nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz, StaggerType::CGrid);
    const auto& relationship                       = nested.relationship;
    nested.child.domain_frame.velocity_ms          = storage.configuration.domain_frame.velocity_ms;
    nested.child.domain_frame.earth_origin_at_t0_m = {
        storage.configuration.domain_frame.earth_origin_at_t0_m.x_m +
            (parent_physical.i_begin + relationship.child_origin_parent_cell_i) * relationship.parent_dx_m -
            child_physical.i_begin * relationship.child_dx_m,
        storage.configuration.domain_frame.earth_origin_at_t0_m.y_m +
            (parent_physical.j_begin + relationship.child_origin_parent_cell_j) * relationship.parent_dy_m -
            child_physical.j_begin * relationship.child_dy_m};
    nested.child.coriolis_f = expected_child_coriolis_f(storage, nested);
}

void prepare_stationary_nested_active_terrain_geometry(const SimulationStorageState& storage,
                                                       NestedExecutionStorage&       nested,
                                                       double                        child_time_s)
{
    const auto& parent = storage.runtime.terrain;
    if (parent.config.scheme_id == "none")
        return;
    if (!parent.scheme || (parent.config.scheme_id != "bell" && parent.config.scheme_id != "schar") ||
        parent.scheme->name() != parent.config.scheme_id ||
        (parent.config.horizontal_placement_id != "grid_attached" &&
         parent.config.horizontal_placement_id != "earth_fixed") ||
        storage.configuration.domain_frame.velocity_ms.u_ms != 0.0 ||
        storage.configuration.domain_frame.velocity_ms.v_ms != 0.0)
        reject_nested_execution("active-terrain child requires stationary Bell or Schar parent ownership");
    if (parent.config.horizontal_placement_id == "earth_fixed" &&
        (!parent.earth_fixed_placement.active ||
         parent.earth_fixed_placement.sampling != tmv::EarthFixedTerrainSampling::Point))
    {
        reject_nested_execution("nested Earth-fixed terrain requires initialized point-sampled geometry");
    }
    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto                         grid_origin = storage.configuration.domain_frame.earth_origin_at(child_time_s);
    const tmv::domain::HorizontalPoint physical_origin{
        grid_origin.x_m + static_cast<double>(parent_physical.i_begin) * storage.grid.dx,
        grid_origin.y_m + static_cast<double>(parent_physical.j_begin) * storage.grid.dy};
    auto geometry = tmv::domain::build_stationary_nested_terrain_geometry(
        parent.config,
        nested.relationship,
        physical_origin,
        1,
        storage.configuration.domain_motion_mode == tmv::init::config::DomainMotionMode::MovingNest ? 2 : 1);

    for (int i = 0; i < nested.relationship.parent_cells_x; ++i)
        for (int j = 0; j < nested.relationship.parent_cells_y; ++j)
        {
            const int storage_i = i + parent_physical.i_begin;
            const int storage_j = j + parent_physical.j_begin;
            if (geometry.parent.topography.h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] !=
                parent.topography.h[static_cast<std::size_t>(storage_i)][static_cast<std::size_t>(storage_j)])
                reject_nested_execution("nested parent terrain geometry differs from the live terrain owner");
            for (int k = 0; k < nested.relationship.levels_z; ++k)
                if (geometry.parent.metrics.J(i, j, k) != parent.metrics.J(storage_i, storage_j, k) ||
                    geometry.parent.metrics.z(i, j, k) != parent.metrics.z(storage_i, storage_j, k))
                    reject_nested_execution("nested parent mapped cell geometry differs from the live terrain owner");
        }

    nested.child.runtime.terrain.config     = parent.config;
    nested.child.runtime.terrain.topography = geometry.child_storage.topography;
    nested.child.runtime.terrain.metrics    = geometry.child_storage.metrics;
    if (parent.config.horizontal_placement_id == "earth_fixed")
    {
        auto& placement                  = nested.child.runtime.terrain.earth_fixed_placement;
        placement.active                 = true;
        placement.nr                     = nested.child.grid.nr;
        placement.nth                    = nested.child.grid.nth;
        placement.nz                     = nested.child.grid.nz;
        placement.dx_m                   = nested.child.grid.dx;
        placement.dy_m                   = nested.child.grid.dy;
        placement.grid_origin_earth_m    = nested.child.domain_frame.earth_origin_at(child_time_s);
        placement.feature_center_earth_m = geometry.feature_center_earth_m;
        placement.generation             = geometry.generation;
    }
    nested.active_terrain.geometry      = std::move(geometry);
    nested.active_terrain.parent_time_s = child_time_s;
    nested.active_terrain.child_time_s  = child_time_s;
}

std::size_t owner_count(int nx, int ny, int nz)
{
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
}

enum class HorizontalOwnerKind
{
    Scalar,
    XFace,
    YFace,
};

constexpr std::size_t scalar_horizontal_primary_field_count = 13;
constexpr std::size_t complete_primary_field_count          = 15;

struct FieldDiscardJournal
{
    std::vector<float>* values = nullptr;
};

struct SurfaceDiscardJournal
{
    std::vector<float>* values = nullptr;
};

struct PrimaryDiscardJournal
{
    std::array<FieldDiscardJournal, scalar_horizontal_primary_field_count> scalar_fields;
    FieldDiscardJournal                                                    u;
    FieldDiscardJournal                                                    v;
};

struct PreparedNestedRadiationMovement
{
    bool                                 active                          = false;
    std::vector<float>*                  newly_admitted_theta_tendency_s = nullptr;
    std::vector<float>*                  newly_admitted_cell_power_w     = nullptr;
    NestedChildRadiationMovementReportV1 report;
};

struct PreparedNestedBoundaryLayerMovement
{
    bool                                     active = false;
    std::array<std::vector<float>*, 7>       newly_admitted_cells{};
    std::array<std::vector<float>*, 9>       newly_admitted_columns{};
    std::vector<SlabColumnState>*            newly_admitted_slab_columns = nullptr;
    NestedChildBoundaryLayerMovementReportV1 report;
};

struct PreparedNestedChaosMovement
{
    bool                                      active                = false;
    bool                                      pbl_base_cache_active = false;
    std::unique_ptr<tmv::chaos::ChaosScheme>* translation_workspace = nullptr;
    std::array<std::vector<float>*, 5>        newly_admitted_cache_fields{};
    std::array<std::vector<float>*, 6>        newly_admitted_cache_surfaces{};
    NestedChildChaosMovementReportV1          report;
};

struct PreparedNestedMicrophysicsMovement
{
    bool        active                        = false;
    bool        theta_storage_residual_active = false;
    std::size_t private_scalar_count          = 0;
    std::array<Field3D*, NestedChildMicrophysicsMovementReportV1::maximum_private_scalars> child_private_fields{};
    Field3D* child_theta_storage_residual = nullptr;
    std::array<std::vector<float>*, NestedChildMicrophysicsMovementReportV1::maximum_private_scalars>
        newly_admitted_private_values{};
    std::array<FieldDiscardJournal, NestedChildMicrophysicsMovementReportV1::maximum_private_scalars>
                                            private_discard_journals;
    std::vector<float>*                     newly_admitted_theta_residual_values = nullptr;
    FieldDiscardJournal                     theta_residual_discard_journal;
    NestedChildMicrophysicsMovementReportV1 report;
};

struct PreparedNestedPrecipitationMovement
{
    bool                                     active                       = false;
    double                                   previous_child_sample_time_s = 0.0;
    std::array<std::vector<float>*, 3>       child_fields{};
    std::array<std::vector<float>*, 3>       newly_admitted_values{};
    std::array<SurfaceDiscardJournal, 3>     discard_journals{};
    NestedChildPrecipitationMovementReportV1 report;
};

struct OwnerLayout
{
    int count_i          = 0;
    int count_j          = 0;
    int storage_offset_i = 0;
    int storage_offset_j = 0;
};

OwnerLayout owner_layout(HorizontalOwnerKind kind, const tmv::domain::OneWayNestRelationshipV1& relationship)
{
    if (kind == HorizontalOwnerKind::XFace)
        return {relationship.child_cells_x + 1, relationship.child_cells_y, 0, 1};
    if (kind == HorizontalOwnerKind::YFace)
        return {relationship.child_cells_x, relationship.child_cells_y + 1, 1, 0};
    return {relationship.child_cells_x, relationship.child_cells_y, 1, 1};
}

template <typename Operation>
void for_each_discarded_owner(const OwnerLayout& layout, int levels, int shift_i, int shift_j, Operation&& operation)
{
    for (int source_i = 0; source_i < layout.count_i; ++source_i)
        for (int source_j = 0; source_j < layout.count_j; ++source_j)
        {
            const int destination_i = source_i - shift_i;
            const int destination_j = source_j - shift_j;
            if (destination_i >= 0 && destination_i < layout.count_i && destination_j >= 0 &&
                destination_j < layout.count_j)
                continue;
            for (int k = 0; k < levels; ++k)
                operation(source_i + layout.storage_offset_i, source_j + layout.storage_offset_j, k);
        }
}

FieldDiscardJournal journal_discarded_field(const Field3D&      field,
                                            const OwnerLayout&  layout,
                                            int                 levels,
                                            int                 shift_i,
                                            int                 shift_j,
                                            std::vector<float>& retained_values)
{
    FieldDiscardJournal journal{&retained_values};
    const auto          overlap_i = layout.count_i - static_cast<int>(std::abs(static_cast<std::int64_t>(shift_i)));
    const auto          overlap_j = layout.count_j - static_cast<int>(std::abs(static_cast<std::int64_t>(shift_j)));
    const std::size_t   discarded =
        owner_count(layout.count_i, layout.count_j, levels) - owner_count(overlap_i, overlap_j, levels);
    retained_values.clear();
    retained_values.reserve(discarded);
    for_each_discarded_owner(
        layout, levels, shift_i, shift_j, [&](int i, int j, int k) { retained_values.push_back(field[i][j][k]); });
    if (retained_values.size() != discarded)
        reject_nested_execution("movement discard journal is incomplete");
    return journal;
}

SurfaceDiscardJournal journal_discarded_surface(const std::vector<float>& field,
                                                int                       storage_nth,
                                                const OwnerLayout&        layout,
                                                int                       shift_i,
                                                int                       shift_j,
                                                std::vector<float>&       discarded_values)
{
    SurfaceDiscardJournal journal{&discarded_values};
    const auto            overlap_i = layout.count_i - static_cast<int>(std::abs(static_cast<std::int64_t>(shift_i)));
    const auto            overlap_j = layout.count_j - static_cast<int>(std::abs(static_cast<std::int64_t>(shift_j)));
    const std::size_t discarded = owner_count(layout.count_i, layout.count_j, 1) - owner_count(overlap_i, overlap_j, 1);
    discarded_values.clear();
    discarded_values.reserve(discarded);
    for_each_discarded_owner(layout,
                             1,
                             shift_i,
                             shift_j,
                             [&](int i, int j, int)
                             {
                                 discarded_values.push_back(
                                     field[static_cast<std::size_t>(i) * static_cast<std::size_t>(storage_nth) +
                                           static_cast<std::size_t>(j)]);
                             });
    if (discarded_values.size() != discarded)
        reject_nested_execution("surface movement discard journal is incomplete");
    return journal;
}

void shift_field_in_place(Field3D& field, const OwnerLayout& layout, int levels, int shift_i, int shift_j)
{
    const int minimum_i   = std::max(0, -shift_i);
    const int maximum_i   = std::min(layout.count_i - 1, layout.count_i - 1 - shift_i);
    const int minimum_j   = std::max(0, -shift_j);
    const int maximum_j   = std::min(layout.count_j - 1, layout.count_j - 1 - shift_j);
    const int begin_i     = shift_i > 0 ? minimum_i : maximum_i;
    const int end_i       = shift_i > 0 ? maximum_i : minimum_i;
    const int increment_i = shift_i > 0 ? 1 : -1;
    const int begin_j     = shift_j > 0 ? minimum_j : maximum_j;
    const int end_j       = shift_j > 0 ? maximum_j : minimum_j;
    const int increment_j = shift_j > 0 ? 1 : -1;
    for (int destination_i = begin_i;; destination_i += increment_i)
    {
        for (int destination_j = begin_j;; destination_j += increment_j)
        {
            for (int k = 0; k < levels; ++k)
                field[destination_i + layout.storage_offset_i][destination_j + layout.storage_offset_j][k] =
                    field[destination_i + shift_i + layout.storage_offset_i]
                         [destination_j + shift_j + layout.storage_offset_j][k];
            if (destination_j == end_j)
                break;
        }
        if (destination_i == end_i)
            break;
    }
}

void fill_newly_admitted_field(
    Field3D& destination, const Field3D& target, const OwnerLayout& layout, int levels, int shift_i, int shift_j)
{
    for (int destination_i = 0; destination_i < layout.count_i; ++destination_i)
        for (int destination_j = 0; destination_j < layout.count_j; ++destination_j)
        {
            const int source_i = destination_i + shift_i;
            const int source_j = destination_j + shift_j;
            if (source_i >= 0 && source_i < layout.count_i && source_j >= 0 && source_j < layout.count_j)
                continue;
            for (int k = 0; k < levels; ++k)
                destination[destination_i + layout.storage_offset_i][destination_j + layout.storage_offset_j][k] =
                    target[destination_i + layout.storage_offset_i][destination_j + layout.storage_offset_j][k];
        }
}

void fill_newly_admitted_field(Field3D&                  destination,
                               const std::vector<float>& values,
                               const OwnerLayout&        layout,
                               int                       levels,
                               int                       shift_i,
                               int                       shift_j) noexcept
{
    std::size_t index = 0;
    for (int destination_i = 0; destination_i < layout.count_i; ++destination_i)
        for (int destination_j = 0; destination_j < layout.count_j; ++destination_j)
        {
            const int source_i = destination_i + shift_i;
            const int source_j = destination_j + shift_j;
            if (source_i >= 0 && source_i < layout.count_i && source_j >= 0 && source_j < layout.count_j)
                continue;
            for (int k = 0; k < levels; ++k)
                destination[destination_i + layout.storage_offset_i][destination_j + layout.storage_offset_j][k] =
                    values[index++];
        }
    if (index != values.size())
        std::terminate();
}

template <typename Value>
void shift_surface_in_place(
    std::vector<Value>& values, int storage_nth, const OwnerLayout& layout, int shift_i, int shift_j) noexcept
{
    const int begin_i     = shift_i >= 0 ? 0 : layout.count_i - 1;
    const int end_i       = shift_i >= 0 ? layout.count_i - 1 : 0;
    const int increment_i = shift_i >= 0 ? 1 : -1;
    const int begin_j     = shift_j >= 0 ? 0 : layout.count_j - 1;
    const int end_j       = shift_j >= 0 ? layout.count_j - 1 : 0;
    const int increment_j = shift_j >= 0 ? 1 : -1;
    for (int destination_i = begin_i;; destination_i += increment_i)
    {
        for (int destination_j = begin_j;; destination_j += increment_j)
        {
            const int source_i = destination_i + shift_i;
            const int source_j = destination_j + shift_j;
            if (source_i >= 0 && source_i < layout.count_i && source_j >= 0 && source_j < layout.count_j)
            {
                const auto destination = static_cast<std::size_t>(destination_i + layout.storage_offset_i) *
                                             static_cast<std::size_t>(storage_nth) +
                                         static_cast<std::size_t>(destination_j + layout.storage_offset_j);
                const auto source      = static_cast<std::size_t>(source_i + layout.storage_offset_i) *
                                             static_cast<std::size_t>(storage_nth) +
                                         static_cast<std::size_t>(source_j + layout.storage_offset_j);
                values[destination]    = values[source];
            }
            if (destination_j == end_j)
                break;
        }
        if (destination_i == end_i)
            break;
    }
}

template <typename Value>
void fill_newly_admitted_surface(std::vector<Value>&       destination,
                                 const std::vector<Value>& values,
                                 int                       storage_nth,
                                 const OwnerLayout&        layout,
                                 int                       shift_i,
                                 int                       shift_j) noexcept
{
    std::size_t index = 0;
    for (int destination_i = 0; destination_i < layout.count_i; ++destination_i)
        for (int destination_j = 0; destination_j < layout.count_j; ++destination_j)
        {
            const int source_i = destination_i + shift_i;
            const int source_j = destination_j + shift_j;
            if (source_i >= 0 && source_i < layout.count_i && source_j >= 0 && source_j < layout.count_j)
                continue;
            const auto destination_index   = static_cast<std::size_t>(destination_i + layout.storage_offset_i) *
                                                 static_cast<std::size_t>(storage_nth) +
                                             static_cast<std::size_t>(destination_j + layout.storage_offset_j);
            destination[destination_index] = values[index++];
        }
    if (index != values.size())
        std::terminate();
}

void restore_discarded_surface(std::vector<float>&          field,
                               int                          storage_nth,
                               const SurfaceDiscardJournal& journal,
                               const OwnerLayout&           layout,
                               int                          shift_i,
                               int                          shift_j) noexcept
{
    if (journal.values == nullptr)
        std::terminate();
    std::size_t index = 0;
    for_each_discarded_owner(
        layout,
        1,
        shift_i,
        shift_j,
        [&](int i, int j, int)
        {
            field[static_cast<std::size_t>(i) * static_cast<std::size_t>(storage_nth) + static_cast<std::size_t>(j)] =
                (*journal.values)[index++];
        });
    if (index != journal.values->size())
        std::terminate();
}

bool same_flat_radiation_config(const RadiationConfig& left, const RadiationConfig& right) noexcept
{
    return left.scheme_id == right.scheme_id && left.do_lw == right.do_lw && left.do_sw == right.do_sw &&
           left.dt_radiation == right.dt_radiation && left.clear_sky_only == right.clear_sky_only &&
           left.use_subcycling == right.use_subcycling && left.subcycle_count == right.subcycle_count &&
           left.enable_diagnostics == right.enable_diagnostics &&
           left.simple_grey_optical_model == right.simple_grey_optical_model &&
           left.simple_grey_lw_vapor_absorption_m2kg == right.simple_grey_lw_vapor_absorption_m2kg &&
           left.simple_grey_lw_liquid_absorption_m2kg == right.simple_grey_lw_liquid_absorption_m2kg &&
           left.simple_grey_lw_ice_absorption_m2kg == right.simple_grey_lw_ice_absorption_m2kg &&
           left.simple_grey_sw_vapor_absorption_m2kg == right.simple_grey_sw_vapor_absorption_m2kg &&
           left.simple_grey_sw_liquid_absorption_m2kg == right.simple_grey_sw_liquid_absorption_m2kg &&
           left.simple_grey_sw_ice_absorption_m2kg == right.simple_grey_sw_ice_absorption_m2kg &&
           left.solar_forcing == right.solar_forcing && left.latitude_deg == right.latitude_deg &&
           left.declination_deg == right.declination_deg &&
           left.local_solar_time_at_start_h == right.local_solar_time_at_start_h &&
           left.tau_lw_ref == right.tau_lw_ref && left.tau_sw_ref == right.tau_sw_ref && left.n_lw == right.n_lw &&
           left.n_sw == right.n_sw && left.terrain_sw_cosine_nodes == right.terrain_sw_cosine_nodes &&
           left.terrain_sw_azimuth_nodes == right.terrain_sw_azimuth_nodes &&
           left.terrain_sw_maximum_bounces == right.terrain_sw_maximum_bounces &&
           left.terrain_sw_relative_residual_tolerance == right.terrain_sw_relative_residual_tolerance &&
           left.terrain_sw_absolute_residual_tolerance_w == right.terrain_sw_absolute_residual_tolerance_w;
}

bool same_boundary_layer_config(const BoundaryLayerConfig& left, const BoundaryLayerConfig& right) noexcept
{
    return left.scheme_id == right.scheme_id && left.dt_pbl == right.dt_pbl &&
           left.apply_surface_fluxes == right.apply_surface_fluxes && left.surface_layer_id == right.surface_layer_id &&
           left.clear_sky_only == right.clear_sky_only && left.enable_tke == right.enable_tke &&
           left.pbl_top_method == right.pbl_top_method && left.pbl_max_height == right.pbl_max_height &&
           left.min_ustar == right.min_ustar;
}

bool same_surface_config(const SurfaceConfig& left, const SurfaceConfig& right) noexcept
{
    return left.z0m == right.z0m && left.z0h == right.z0h && left.albedo == right.albedo &&
           left.emissivity == right.emissivity && left.Tsfc == right.Tsfc && left.qsfc == right.qsfc;
}

bool same_chaos_config(const tmv::chaos::ChaosConfig& left, const tmv::chaos::ChaosConfig& right) noexcept
{
    return left.scheme_id == right.scheme_id && left.seed == right.seed && left.member_id == right.member_id &&
           left.dt_chaos == right.dt_chaos && left.tau_t == right.tau_t && left.Lx == right.Lx && left.Ly == right.Ly &&
           left.filter_id == right.filter_id && left.xi_max == right.xi_max && left.taper_id == right.taper_id &&
           left.taper_z1 == right.taper_z1 && left.taper_z2 == right.taper_z2 &&
           left.apply_to_ic == right.apply_to_ic && left.apply_to_tendencies == right.apply_to_tendencies &&
           left.sigma_ic == right.sigma_ic && left.alpha_tend == right.alpha_tend;
}

bool valid_slab_column(const SlabColumnState& column) noexcept
{
    return !column.initialized ||
           (std::isfinite(column.h_m) && column.h_m > 0.0 && std::isfinite(column.theta_m_k) &&
            column.theta_m_k > 0.0 && std::isfinite(column.qv_m_kgkg) && column.qv_m_kgkg >= 0.0);
}

double require_flat_radiation_held_state(const AtmosphericLevelRuntimeState& level_runtime,
                                         const PhysicsTendencyFieldStorage&  tendencies,
                                         const GridStorageState&             grid,
                                         const char*                         owner)
{
    const auto& radiation     = level_runtime.radiation;
    const auto  shape_matches = [&](const Field3D& field)
    { return field.size_r() == grid.nr && field.size_th() == grid.nth && field.size_z() == grid.nz; };
    if (!shape_matches(tendencies.dtheta_dt_rad) || !shape_matches(radiation.held_cell_power_w))
        reject_nested_execution(owner);
    const auto physical  = physical_scalar_domain(grid.nr, grid.nth, grid.nz, StaggerType::CGrid);
    double     aggregate = 0.0;
    for (int i = physical.i_begin; i < physical.i_end; ++i)
        for (int j = physical.j_begin; j < physical.j_end; ++j)
            for (int k = physical.k_begin; k < physical.k_end; ++k)
            {
                const float tendency = tendencies.dtheta_dt_rad[i][j][k];
                const float power    = radiation.held_cell_power_w[i][j][k];
                if (!std::isfinite(tendency) || !std::isfinite(power))
                    reject_nested_execution(owner);
                aggregate += static_cast<double>(power);
            }
    const double scale = std::max({1.0, std::abs(aggregate), std::abs(radiation.held_atmospheric_power_w)});
    if (!std::isfinite(radiation.held_atmospheric_power_w) ||
        std::abs(aggregate - radiation.held_atmospheric_power_w) > 1.0e-12 * scale)
        reject_nested_execution(owner);
    return aggregate;
}

PreparedNestedRadiationMovement
prepare_nested_radiation_movement(SimulationStorageState&                      storage,
                                  const tmv::domain::OneWayNestRelationshipV1& next_relationship,
                                  int                                          child_shift_i,
                                  int                                          child_shift_j)
{
    const auto&                     parent      = storage.runtime.radiation;
    const auto&                     child_level = storage.nested_execution.child;
    const auto&                     child       = child_level.runtime.radiation;
    PreparedNestedRadiationMovement prepared;
    if (parent.config.scheme_id == "none" && child.config.scheme_id == "none")
        return prepared;
    if (parent.config.scheme_id != "simple_grey" || child.config.scheme_id != "simple_grey" || !parent.scheme ||
        !child.scheme || parent.scheme->name() != "simple_grey" || child.scheme->name() != "simple_grey" ||
        !same_flat_radiation_config(parent.config, child.config) ||
        parent.last_update_time_s != child.last_update_time_s ||
        parent.next_update_time_s != child.next_update_time_s || !std::isfinite(child.last_update_time_s) ||
        child.last_update_time_s < 0.0 || !std::isfinite(child.next_update_time_s) ||
        child.next_update_time_s <= child.last_update_time_s)
        reject_nested_execution("active radiation movement requires matching sampled flat simple-grey state");
    const double flat_aggregates[] = {parent.held_shortwave_incoming_power_w,
                                      parent.held_shortwave_atmospheric_absorbed_power_w,
                                      parent.held_surface_absorbed_power_w,
                                      parent.held_top_escape_power_w,
                                      parent.held_lateral_escape_power_w,
                                      parent.held_unresolved_reflected_power_w,
                                      child.held_shortwave_incoming_power_w,
                                      child.held_shortwave_atmospheric_absorbed_power_w,
                                      child.held_surface_absorbed_power_w,
                                      child.held_top_escape_power_w,
                                      child.held_lateral_escape_power_w,
                                      child.held_unresolved_reflected_power_w};
    if (!parent.held_reflection_converged || !child.held_reflection_converged || parent.terrain_shortwave_workspace ||
        child.terrain_shortwave_workspace ||
        std::any_of(std::begin(flat_aggregates), std::end(flat_aggregates), [](double value) { return value != 0.0; }))
        reject_nested_execution("flat radiation movement cannot carry terrain shortwave state");

    (void)require_flat_radiation_held_state(
        storage.runtime, storage.tendencies, storage.grid, "parent radiation held state is incomplete");
    const double previous_child_aggregate = require_flat_radiation_held_state(
        child_level.runtime, child_level.tendencies, child_level.grid, "child radiation held state is incomplete");

    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto   scalar              = owner_layout(HorizontalOwnerKind::Scalar, next_relationship);
    const double expected_area_ratio = 1.0 / (static_cast<double>(next_relationship.parent_ratio_x) *
                                              static_cast<double>(next_relationship.parent_ratio_y));
    const double area_ratio          = expected_area_ratio;

    const std::size_t new_cells =
        owner_count(scalar.count_i, scalar.count_j, next_relationship.levels_z) -
        owner_count(scalar.count_i - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_i))),
                    scalar.count_j - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_j))),
                    next_relationship.levels_z);
    auto& scratch = storage.nested_execution.boundary_scratch.movement_radiation_newly_admitted;
    prepared.newly_admitted_theta_tendency_s = &scratch[0];
    prepared.newly_admitted_cell_power_w     = &scratch[1];
    for (auto& values : scratch)
    {
        values.clear();
        values.reserve(new_cells);
    }

    std::size_t new_index                      = 0;
    double      committed_aggregate            = 0.0;
    double      newly_admitted_target_power    = 0.0;
    double      newly_admitted_committed_power = 0.0;
    for (int ci = 0; ci < scalar.count_i; ++ci)
        for (int cj = 0; cj < scalar.count_j; ++cj)
        {
            const int  old_ci   = ci + child_shift_i;
            const int  old_cj   = cj + child_shift_j;
            const bool retained = old_ci >= 0 && old_ci < scalar.count_i && old_cj >= 0 && old_cj < scalar.count_j;
            const int  parent_i = parent_physical.i_begin + next_relationship.child_origin_parent_cell_i +
                                  ci / next_relationship.parent_ratio_x;
            const int  parent_j = parent_physical.j_begin + next_relationship.child_origin_parent_cell_j +
                                  cj / next_relationship.parent_ratio_y;
            for (int k = 0; k < next_relationship.levels_z; ++k)
            {
                float committed_power = 0.0f;
                if (retained)
                {
                    committed_power =
                        child.held_cell_power_w[old_ci + scalar.storage_offset_i][old_cj + scalar.storage_offset_j][k];
                }
                else
                {
                    const float  tendency = storage.tendencies.dtheta_dt_rad[parent_i][parent_j][k];
                    const double target_power =
                        static_cast<double>(parent.held_cell_power_w[parent_i][parent_j][k]) * area_ratio;
                    committed_power = static_cast<float>(target_power);
                    if (!std::isfinite(tendency) || !std::isfinite(committed_power))
                        reject_nested_execution("radiation movement target overflows storage");
                    prepared.newly_admitted_theta_tendency_s->push_back(tendency);
                    prepared.newly_admitted_cell_power_w->push_back(committed_power);
                    newly_admitted_target_power += target_power;
                    newly_admitted_committed_power += static_cast<double>(committed_power);
                    ++new_index;
                }
                committed_aggregate += static_cast<double>(committed_power);
            }
        }
    if (new_index != new_cells || !std::isfinite(committed_aggregate))
        reject_nested_execution("radiation movement target is incomplete");
    prepared.active                                   = true;
    prepared.report.active                            = true;
    prepared.report.sample_time_s                     = child.last_update_time_s;
    prepared.report.newly_admitted_cells              = new_cells;
    prepared.report.previous_child_held_power_w       = previous_child_aggregate;
    prepared.report.newly_admitted_target_power_w     = newly_admitted_target_power;
    prepared.report.newly_admitted_committed_power_w  = newly_admitted_committed_power;
    prepared.report.newly_admitted_storage_rounding_w = newly_admitted_target_power - newly_admitted_committed_power;
    prepared.report.committed_child_held_power_w      = committed_aggregate;
    return prepared;
}

PreparedNestedBoundaryLayerMovement
prepare_nested_boundary_layer_movement(SimulationStorageState&                      storage,
                                       const tmv::domain::OneWayNestRelationshipV1& next_relationship,
                                       int                                          child_shift_i,
                                       int                                          child_shift_j)
{
    const auto&                         parent      = storage.runtime.boundary_layer;
    const auto&                         child_level = storage.nested_execution.child;
    const auto&                         child       = child_level.runtime.boundary_layer;
    PreparedNestedBoundaryLayerMovement prepared;
    if (parent.config.scheme_id == "none" && child.config.scheme_id == "none")
        return prepared;
    const auto supported = [](const std::string& name) { return name == "slab" || name == "ysu" || name == "mynn"; };
    if (!parent.scheme || !child.scheme || parent.scheme->name() != parent.config.scheme_id ||
        child.scheme->name() != child.config.scheme_id || parent.scheme->name() != child.scheme->name() ||
        !supported(parent.scheme->name()) || !same_boundary_layer_config(parent.config, child.config) ||
        !same_surface_config(parent.surface_config, child.surface_config) ||
        parent.last_update_time_s != child.last_update_time_s ||
        parent.next_update_time_s != child.next_update_time_s || !std::isfinite(child.last_update_time_s) ||
        child.last_update_time_s < 0.0 || !std::isfinite(child.next_update_time_s) ||
        child.next_update_time_s <= child.last_update_time_s)
    {
        reject_nested_execution("active boundary-layer movement requires matching sampled flat state");
    }

    const auto field_shape_matches = [](const Field3D& field, const GridStorageState& grid)
    { return field.size_r() == grid.nr && field.size_th() == grid.nth && field.size_z() == grid.nz; };
    const Field3D* const parent_fields[] = {&parent.held_u_force_n,
                                            &parent.held_v_force_n,
                                            &parent.held_theta_inventory_rate_kgk_s,
                                            &parent.held_vapor_mass_rate_kg_s,
                                            &parent.held_cloud_liquid_mass_rate_kg_s,
                                            &parent.held_cloud_ice_mass_rate_kg_s,
                                            &storage.tendencies.dtke_dt_pbl};
    const Field3D* const child_fields[]  = {&child.held_u_force_n,
                                            &child.held_v_force_n,
                                            &child.held_theta_inventory_rate_kgk_s,
                                            &child.held_vapor_mass_rate_kg_s,
                                            &child.held_cloud_liquid_mass_rate_kg_s,
                                            &child.held_cloud_ice_mass_rate_kg_s,
                                            &child_level.tendencies.dtke_dt_pbl};
    for (const Field3D* field : parent_fields)
    {
        if (!field_shape_matches(*field, storage.grid) ||
            std::any_of(
                field->data(), field->data() + field->size(), [](float value) { return !std::isfinite(value); }))
            reject_nested_execution("parent boundary-layer held state is incomplete");
    }
    for (const Field3D* field : child_fields)
    {
        if (!field_shape_matches(*field, child_level.grid) ||
            std::any_of(
                field->data(), field->data() + field->size(), [](float value) { return !std::isfinite(value); }))
            reject_nested_execution("child boundary-layer held state is incomplete");
    }

    const std::size_t parent_surface_size =
        static_cast<std::size_t>(storage.grid.nr) * static_cast<std::size_t>(storage.grid.nth);
    const std::size_t child_surface_size =
        static_cast<std::size_t>(child_level.grid.nr) * static_cast<std::size_t>(child_level.grid.nth);
    const std::array<const std::vector<float>*, 9> parent_surface = {{&parent.skin_theta_2d,
                                                                      &parent.skin_qv_2d,
                                                                      &storage.surface.ustar,
                                                                      &storage.surface.tau_u,
                                                                      &storage.surface.tau_v,
                                                                      &storage.surface.sensible_heat_flux,
                                                                      &storage.surface.latent_heat_flux,
                                                                      &storage.surface.moisture_flux,
                                                                      &storage.surface.pbl_height}};
    const std::array<const std::vector<float>*, 9> child_surface  = {{&child.skin_theta_2d,
                                                                      &child.skin_qv_2d,
                                                                      &child_level.surface.ustar,
                                                                      &child_level.surface.tau_u,
                                                                      &child_level.surface.tau_v,
                                                                      &child_level.surface.sensible_heat_flux,
                                                                      &child_level.surface.latent_heat_flux,
                                                                      &child_level.surface.moisture_flux,
                                                                      &child_level.surface.pbl_height}};
    const auto                                     valid_surface  = [](const auto& fields, std::size_t size)
    {
        return std::all_of(
            fields.begin(),
            fields.end(),
            [size](const std::vector<float>* field)
            {
                return field->size() == size &&
                       std::all_of(field->begin(), field->end(), [](float value) { return std::isfinite(value); });
            });
    };
    if (!parent.skin_captured || !child.skin_captured || parent.skin_nr != storage.grid.nr ||
        parent.skin_nth != storage.grid.nth || child.skin_nr != child_level.grid.nr ||
        child.skin_nth != child_level.grid.nth || parent.slab_nr != storage.grid.nr ||
        parent.slab_nth != storage.grid.nth || child.slab_nr != child_level.grid.nr ||
        child.slab_nth != child_level.grid.nth || parent.slab_columns.size() != parent_surface_size ||
        child.slab_columns.size() != child_surface_size || !valid_surface(parent_surface, parent_surface_size) ||
        !valid_surface(child_surface, child_surface_size) ||
        !std::all_of(parent.slab_columns.begin(), parent.slab_columns.end(), valid_slab_column) ||
        !std::all_of(child.slab_columns.begin(), child.slab_columns.end(), valid_slab_column))
    {
        reject_nested_execution("active boundary-layer movement requires complete column memory");
    }

    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto        scalar     = owner_layout(HorizontalOwnerKind::Scalar, next_relationship);
    const double      area_ratio = 1.0 / (static_cast<double>(next_relationship.parent_ratio_x) *
                                          static_cast<double>(next_relationship.parent_ratio_y));
    const std::size_t newly_admitted_columns =
        owner_count(scalar.count_i, scalar.count_j, 1) -
        owner_count(scalar.count_i - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_i))),
                    scalar.count_j - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_j))),
                    1);
    const std::size_t newly_admitted_cells =
        newly_admitted_columns * static_cast<std::size_t>(next_relationship.levels_z);
    auto& boundary_scratch = storage.nested_execution.boundary_scratch;
    for (std::size_t field = 0; field < prepared.newly_admitted_cells.size(); ++field)
    {
        auto& values = boundary_scratch.movement_boundary_layer_cell_newly_admitted[field];
        values.clear();
        values.reserve(newly_admitted_cells);
        prepared.newly_admitted_cells[field] = &values;
    }
    for (std::size_t field = 0; field < prepared.newly_admitted_columns.size(); ++field)
    {
        auto& values = boundary_scratch.movement_boundary_layer_column_newly_admitted[field];
        values.clear();
        values.reserve(newly_admitted_columns);
        prepared.newly_admitted_columns[field] = &values;
    }
    auto& slab_values = boundary_scratch.movement_boundary_layer_slab_newly_admitted;
    slab_values.clear();
    slab_values.reserve(newly_admitted_columns);
    prepared.newly_admitted_slab_columns = &slab_values;

    std::array<double, 6> new_target{};
    std::array<double, 6> new_committed{};
    std::array<double, 6> committed_child{};
    std::size_t           committed_columns = 0;
    std::size_t           committed_cells   = 0;
    for (int ci = 0; ci < scalar.count_i; ++ci)
        for (int cj = 0; cj < scalar.count_j; ++cj)
        {
            const int  old_ci   = ci + child_shift_i;
            const int  old_cj   = cj + child_shift_j;
            const bool retained = old_ci >= 0 && old_ci < scalar.count_i && old_cj >= 0 && old_cj < scalar.count_j;
            const int  parent_i = parent_physical.i_begin + next_relationship.child_origin_parent_cell_i +
                                  ci / next_relationship.parent_ratio_x;
            const int  parent_j = parent_physical.j_begin + next_relationship.child_origin_parent_cell_j +
                                  cj / next_relationship.parent_ratio_y;
            if (!retained)
            {
                const auto parent_surface_index =
                    static_cast<std::size_t>(parent_i) * static_cast<std::size_t>(storage.grid.nth) +
                    static_cast<std::size_t>(parent_j);
                prepared.newly_admitted_columns[0]->push_back(parent.skin_theta_2d[parent_surface_index]);
                prepared.newly_admitted_columns[1]->push_back(parent.skin_qv_2d[parent_surface_index]);
                prepared.newly_admitted_slab_columns->push_back(parent.slab_columns[parent_surface_index]);
                for (std::size_t field = 0; field < 7; ++field)
                    prepared.newly_admitted_columns[field + 2]->push_back(
                        (*parent_surface[field + 2])[parent_surface_index]);
                ++committed_columns;
            }
            for (int k = 0; k < next_relationship.levels_z; ++k)
            {
                std::array<float, 6> values{};
                if (retained)
                {
                    const int i = old_ci + scalar.storage_offset_i;
                    const int j = old_cj + scalar.storage_offset_j;
                    values      = {{child.held_u_force_n[i][j][k],
                                    child.held_v_force_n[i][j][k],
                                    child.held_theta_inventory_rate_kgk_s[i][j][k],
                                    child.held_vapor_mass_rate_kg_s[i][j][k],
                                    child.held_cloud_liquid_mass_rate_kg_s[i][j][k],
                                    child.held_cloud_ice_mass_rate_kg_s[i][j][k]}};
                }
                else
                {
                    const std::array<float, 6> parent_values = {
                        {parent.held_u_force_n[parent_i][parent_j][k],
                         parent.held_v_force_n[parent_i][parent_j][k],
                         parent.held_theta_inventory_rate_kgk_s[parent_i][parent_j][k],
                         parent.held_vapor_mass_rate_kg_s[parent_i][parent_j][k],
                         parent.held_cloud_liquid_mass_rate_kg_s[parent_i][parent_j][k],
                         parent.held_cloud_ice_mass_rate_kg_s[parent_i][parent_j][k]}};
                    for (std::size_t field = 0; field < values.size(); ++field)
                    {
                        const double target = static_cast<double>(parent_values[field]) * area_ratio;
                        values[field]       = static_cast<float>(target);
                        if (!std::isfinite(values[field]))
                            reject_nested_execution("boundary-layer movement target overflows storage");
                        new_target[field] += target;
                        new_committed[field] += static_cast<double>(values[field]);
                    }
                    for (std::size_t field = 0; field < 6; ++field)
                        prepared.newly_admitted_cells[field]->push_back(values[field]);
                    prepared.newly_admitted_cells[6]->push_back(storage.tendencies.dtke_dt_pbl[parent_i][parent_j][k]);
                    ++committed_cells;
                }
                for (std::size_t field = 0; field < values.size(); ++field)
                    committed_child[field] += static_cast<double>(values[field]);
            }
        }
    if (committed_columns != newly_admitted_columns || committed_cells != newly_admitted_cells ||
        !std::all_of(committed_child.begin(), committed_child.end(), [](double value) { return std::isfinite(value); }))
    {
        reject_nested_execution("boundary-layer movement target is incomplete");
    }

    prepared.active                                             = true;
    auto& report                                                = prepared.report;
    report.active                                               = true;
    report.sample_time_s                                        = child.last_update_time_s;
    report.newly_admitted_cells                                 = newly_admitted_cells;
    report.newly_admitted_columns                               = newly_admitted_columns;
    report.newly_admitted_target_x_force_n                      = new_target[0];
    report.newly_admitted_committed_x_force_n                   = new_committed[0];
    report.newly_admitted_x_force_storage_rounding_n            = new_target[0] - new_committed[0];
    report.newly_admitted_target_y_force_n                      = new_target[1];
    report.newly_admitted_committed_y_force_n                   = new_committed[1];
    report.newly_admitted_y_force_storage_rounding_n            = new_target[1] - new_committed[1];
    report.newly_admitted_target_theta_inventory_rate_kgk_s     = new_target[2];
    report.newly_admitted_committed_theta_inventory_rate_kgk_s  = new_committed[2];
    report.newly_admitted_theta_storage_rounding_kgk_s          = new_target[2] - new_committed[2];
    report.newly_admitted_target_vapor_mass_rate_kg_s           = new_target[3];
    report.newly_admitted_committed_vapor_mass_rate_kg_s        = new_committed[3];
    report.newly_admitted_vapor_storage_rounding_kg_s           = new_target[3] - new_committed[3];
    report.newly_admitted_target_cloud_liquid_mass_rate_kg_s    = new_target[4];
    report.newly_admitted_committed_cloud_liquid_mass_rate_kg_s = new_committed[4];
    report.newly_admitted_cloud_liquid_storage_rounding_kg_s    = new_target[4] - new_committed[4];
    report.newly_admitted_target_cloud_ice_mass_rate_kg_s       = new_target[5];
    report.newly_admitted_committed_cloud_ice_mass_rate_kg_s    = new_committed[5];
    report.newly_admitted_cloud_ice_storage_rounding_kg_s       = new_target[5] - new_committed[5];
    report.committed_child_x_force_n                            = committed_child[0];
    report.committed_child_y_force_n                            = committed_child[1];
    report.committed_child_theta_inventory_rate_kgk_s           = committed_child[2];
    report.committed_child_vapor_mass_rate_kg_s                 = committed_child[3];
    report.committed_child_cloud_liquid_mass_rate_kg_s          = committed_child[4];
    report.committed_child_cloud_ice_mass_rate_kg_s             = committed_child[5];
    return prepared;
}

PreparedNestedChaosMovement
prepare_nested_chaos_movement(SimulationStorageState&                      storage,
                              const tmv::domain::OneWayNestRelationshipV1& next_relationship,
                              int                                          child_shift_i,
                              int                                          child_shift_j)
{
    const auto&                 parent      = storage.runtime.chaos;
    const auto&                 child_level = storage.nested_execution.child;
    const auto&                 child       = child_level.runtime.chaos;
    PreparedNestedChaosMovement prepared;
    if (parent.config.scheme_id == "none" && child.config.scheme_id == "none")
        return prepared;
    const auto evolving = [](const std::string& name) { return name == "boundary_layer" || name == "full_stochastic"; };
    if (!parent.scheme || !child.scheme || parent.scheme->name() != parent.config.scheme_id ||
        child.scheme->name() != child.config.scheme_id || parent.scheme->name() != child.scheme->name() ||
        !evolving(parent.scheme->name()) || !same_chaos_config(parent.config, child.config) ||
        !std::isfinite(parent.noise_elapsed_s) || parent.noise_elapsed_s != child.noise_elapsed_s ||
        parent.noise_elapsed_s < 0.0 || parent.noise_elapsed_s >= parent.config.dt_chaos + 1.0e-12)
        reject_nested_execution("active chaos movement requires one synchronized Earth process");

    tmv::chaos::HorizontalStochasticStatePhase parent_phase;
    tmv::chaos::HorizontalStochasticStatePhase child_phase;
    std::string                                error;
    if (!parent.scheme->horizontal_stochastic_state_phase(parent_phase, error) ||
        !child.scheme->horizontal_stochastic_state_phase(child_phase, error) ||
        parent_phase.initialized != child_phase.initialized ||
        parent_phase.current_epoch != child_phase.current_epoch ||
        parent_phase.cadence_dt_s != child_phase.cadence_dt_s)
        reject_nested_execution("active chaos movement requires synchronized stochastic epochs");
    auto& translation_workspace = storage.nested_execution.boundary_scratch.movement_chaos_translation_workspace;
    tmv::chaos::HorizontalStochasticStatePhase translated_phase;
    if (!child.scheme->prepare_horizontal_stochastic_state_translation(child_shift_i,
                                                                       child_shift_j,
                                                                       child.config,
                                                                       child_level.process_grid_metrics,
                                                                       translation_workspace,
                                                                       translated_phase,
                                                                       error))
        throw std::invalid_argument("invalid nested execution relationship: child stochastic "
                                    "Earth-window translation failed: " +
                                    error);
    prepared.translation_workspace = &translation_workspace;
    const auto expected_origin_i   = checked_shifted_earth_origin(child_phase.earth_origin_i_cells, child_shift_i);
    const auto expected_origin_j   = checked_shifted_earth_origin(child_phase.earth_origin_j_cells, child_shift_j);
    if (translated_phase.current_epoch != child_phase.current_epoch ||
        translated_phase.cadence_dt_s != child_phase.cadence_dt_s ||
        translated_phase.initialized != child_phase.initialized ||
        translated_phase.earth_origin_i_cells != expected_origin_i ||
        translated_phase.earth_origin_j_cells != expected_origin_j)
        reject_nested_execution("translated stochastic phase is not the same Earth realization");

    if (parent.pbl_base_tendencies_valid != child.pbl_base_tendencies_valid)
        reject_nested_execution("parent and child stochastic PBL cache validity differs");
    prepared.active                      = true;
    prepared.report.active               = true;
    prepared.report.stochastic_epoch     = translated_phase.current_epoch;
    prepared.report.noise_elapsed_s      = child.noise_elapsed_s;
    prepared.report.earth_origin_i_cells = translated_phase.earth_origin_i_cells;
    prepared.report.earth_origin_j_cells = translated_phase.earth_origin_j_cells;
    if (!child.pbl_base_tendencies_valid)
        return prepared;

    const auto shape_matches = [](const Field3D& field, const GridStorageState& grid)
    { return field.size_r() == grid.nr && field.size_th() == grid.nth && field.size_z() == grid.nz; };
    const std::array<const Field3D*, 5> parent_fields = {{&parent.base_du_dt_pbl,
                                                          &parent.base_dv_dt_pbl,
                                                          &parent.base_dtheta_dt_pbl,
                                                          &parent.base_dqv_dt_pbl,
                                                          &parent.base_dtke_dt_pbl}};
    const std::array<const Field3D*, 5> child_fields  = {{&child.base_du_dt_pbl,
                                                          &child.base_dv_dt_pbl,
                                                          &child.base_dtheta_dt_pbl,
                                                          &child.base_dqv_dt_pbl,
                                                          &child.base_dtke_dt_pbl}};
    for (const Field3D* field : parent_fields)
        if (!shape_matches(*field, storage.grid) || std::any_of(field->data(),
                                                                field->data() + field->size(),
                                                                [](float value) { return !std::isfinite(value); }))
            reject_nested_execution("parent stochastic PBL cache is incomplete");
    for (const Field3D* field : child_fields)
        if (!shape_matches(*field, child_level.grid) || std::any_of(field->data(),
                                                                    field->data() + field->size(),
                                                                    [](float value) { return !std::isfinite(value); }))
            reject_nested_execution("child stochastic PBL cache is incomplete");

    const std::array<const std::vector<float>*, 6> parent_surfaces = {{&parent.base_sfc_ustar,
                                                                       &parent.base_sfc_tau_u,
                                                                       &parent.base_sfc_tau_v,
                                                                       &parent.base_sfc_sensible_heat_flux,
                                                                       &parent.base_sfc_latent_heat_flux,
                                                                       &parent.base_sfc_moisture_flux}};
    const std::array<const std::vector<float>*, 6> child_surfaces  = {{&child.base_sfc_ustar,
                                                                       &child.base_sfc_tau_u,
                                                                       &child.base_sfc_tau_v,
                                                                       &child.base_sfc_sensible_heat_flux,
                                                                       &child.base_sfc_latent_heat_flux,
                                                                       &child.base_sfc_moisture_flux}};
    const std::size_t parent_columns = static_cast<std::size_t>(storage.grid.nr) * storage.grid.nth;
    const std::size_t child_columns  = static_cast<std::size_t>(child_level.grid.nr) * child_level.grid.nth;
    for (const auto* values : parent_surfaces)
        if (values->size() != parent_columns ||
            !std::all_of(values->begin(), values->end(), [](float value) { return std::isfinite(value); }))
            reject_nested_execution("parent stochastic PBL surface cache is incomplete");
    for (const auto* values : child_surfaces)
        if (values->size() != child_columns ||
            !std::all_of(values->begin(), values->end(), [](float value) { return std::isfinite(value); }))
            reject_nested_execution("child stochastic PBL surface cache is incomplete");

    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto        scalar     = owner_layout(HorizontalOwnerKind::Scalar, next_relationship);
    const double      area_ratio = 1.0 / (static_cast<double>(next_relationship.parent_ratio_x) *
                                          static_cast<double>(next_relationship.parent_ratio_y));
    const std::size_t new_columns =
        owner_count(scalar.count_i, scalar.count_j, 1) -
        owner_count(scalar.count_i - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_i))),
                    scalar.count_j - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_j))),
                    1);
    const std::size_t new_cells        = new_columns * static_cast<std::size_t>(next_relationship.levels_z);
    auto&             boundary_scratch = storage.nested_execution.boundary_scratch;
    for (std::size_t field = 0; field < prepared.newly_admitted_cache_fields.size(); ++field)
    {
        auto& values = boundary_scratch.movement_chaos_cache_cell_newly_admitted[field];
        values.clear();
        values.reserve(new_cells);
        prepared.newly_admitted_cache_fields[field] = &values;
    }
    for (std::size_t field = 0; field < prepared.newly_admitted_cache_surfaces.size(); ++field)
    {
        auto& values = boundary_scratch.movement_chaos_cache_column_newly_admitted[field];
        values.clear();
        values.reserve(new_columns);
        prepared.newly_admitted_cache_surfaces[field] = &values;
    }
    std::array<double, 4> target{};
    std::array<double, 4> committed{};
    std::size_t           committed_columns = 0;
    std::size_t           committed_cells   = 0;
    for (int ci = 0; ci < scalar.count_i; ++ci)
        for (int cj = 0; cj < scalar.count_j; ++cj)
        {
            const int old_ci = ci + child_shift_i;
            const int old_cj = cj + child_shift_j;
            if (old_ci >= 0 && old_ci < scalar.count_i && old_cj >= 0 && old_cj < scalar.count_j)
                continue;
            const int  parent_i = parent_physical.i_begin + next_relationship.child_origin_parent_cell_i +
                                  ci / next_relationship.parent_ratio_x;
            const int  parent_j = parent_physical.j_begin + next_relationship.child_origin_parent_cell_j +
                                  cj / next_relationship.parent_ratio_y;
            const auto parent_surface_index =
                static_cast<std::size_t>(parent_i) * storage.grid.nth + static_cast<std::size_t>(parent_j);
            for (std::size_t field = 0; field < parent_surfaces.size(); ++field)
                prepared.newly_admitted_cache_surfaces[field]->push_back(
                    (*parent_surfaces[field])[parent_surface_index]);
            ++committed_columns;
            for (int k = 0; k < next_relationship.levels_z; ++k)
            {
                for (std::size_t field = 0; field < 4; ++field)
                {
                    const double raw = static_cast<double>((*parent_fields[field])[parent_i][parent_j][k]) * area_ratio;
                    const float  stored = static_cast<float>(raw);
                    if (!std::isfinite(stored))
                        reject_nested_execution("stochastic PBL cache target overflows storage");
                    prepared.newly_admitted_cache_fields[field]->push_back(stored);
                    target[field] += raw;
                    committed[field] += static_cast<double>(stored);
                }
                prepared.newly_admitted_cache_fields[4]->push_back((*parent_fields[4])[parent_i][parent_j][k]);
                ++committed_cells;
            }
        }
    if (committed_columns != new_columns || committed_cells != new_cells)
        reject_nested_execution("stochastic PBL cache movement target is incomplete");
    prepared.pbl_base_cache_active                             = true;
    auto& report                                               = prepared.report;
    report.pbl_base_cache_active                               = true;
    report.newly_admitted_cache_cells                          = new_cells;
    report.newly_admitted_cache_columns                        = new_columns;
    report.newly_admitted_target_x_force_n                     = target[0];
    report.newly_admitted_committed_x_force_n                  = committed[0];
    report.newly_admitted_x_force_storage_rounding_n           = target[0] - committed[0];
    report.newly_admitted_target_y_force_n                     = target[1];
    report.newly_admitted_committed_y_force_n                  = committed[1];
    report.newly_admitted_y_force_storage_rounding_n           = target[1] - committed[1];
    report.newly_admitted_target_theta_inventory_rate_kgk_s    = target[2];
    report.newly_admitted_committed_theta_inventory_rate_kgk_s = committed[2];
    report.newly_admitted_theta_storage_rounding_kgk_s         = target[2] - committed[2];
    report.newly_admitted_target_vapor_mass_rate_kg_s          = target[3];
    report.newly_admitted_committed_vapor_mass_rate_kg_s       = committed[3];
    report.newly_admitted_vapor_storage_rounding_kg_s          = target[3] - committed[3];
    return prepared;
}

void prepare_mapped_private_movement_values(SimulationStorageState&             storage,
                                            const Field3D&                      admitted_density,
                                            int                                 child_shift_i,
                                            int                                 child_shift_j,
                                            PreparedNestedMicrophysicsMovement& prepared);

PreparedNestedMicrophysicsMovement
prepare_nested_microphysics_movement(SimulationStorageState&                      storage,
                                     const tmv::domain::OneWayNestRelationshipV1& next_relationship,
                                     int                                          child_shift_i,
                                     int                                          child_shift_j,
                                     const Field3D&                               admitted_density)
{
    const auto& parent_runtime = storage.runtime.microphysics;
    auto&       child_level    = storage.nested_execution.child;
    auto&       child_runtime  = child_level.runtime.microphysics;
    if (!parent_runtime.scheme || !child_runtime.scheme ||
        parent_runtime.scheme->get_scheme_name() != child_runtime.scheme->get_scheme_name())
        reject_nested_execution("microphysics movement requires one configured scheme");

    PreparedNestedMicrophysicsMovement prepared;
    auto&                              movement_scratch  = storage.nested_execution.boundary_scratch;
    const auto                         parent_parameters = parent_runtime.scheme->parameter_identity();
    const auto                         child_parameters  = child_runtime.scheme->parameter_identity();
    if (!(parent_parameters == child_parameters))
        reject_nested_execution("parent and child microphysics parameters differ");

    tmv::MicrophysicsPrognosticScalarList parent_scalars = parent_runtime.scheme->prognostic_scalars_for_transport();
    tmv::MicrophysicsPrognosticScalarList child_scalars  = child_runtime.scheme->prognostic_scalars_for_transport();
    if (parent_scalars.size() != child_scalars.size() ||
        child_scalars.size() > NestedChildMicrophysicsMovementReportV1::maximum_private_scalars)
        reject_nested_execution("microphysics private-state layouts differ");

    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto scalar        = owner_layout(HorizontalOwnerKind::Scalar, next_relationship);
    const auto field_matches = [](const Field3D& field, const GridStorageState& grid)
    { return field.size_r() == grid.nr && field.size_th() == grid.nth && field.size_z() == grid.nz; };
    const auto valid_scalar = [](float value, bool positive)
    { return std::isfinite(value) && (!positive || value >= 0.0f); };

    for (std::size_t index = 0; index < child_scalars.size(); ++index)
    {
        const auto& parent = parent_scalars[index];
        const auto& child  = child_scalars[index];
        if (!parent.name || !child.name || !parent.field || !child.field || std::string(parent.name) != child.name ||
            parent.storage != child.storage || parent.positive_definite != child.positive_definite ||
            !field_matches(*parent.field, storage.grid) || !field_matches(*child.field, child_level.grid))
            reject_nested_execution("microphysics private-state binding is inconsistent");
        for (std::size_t previous = 0; previous < index; ++previous)
            if (std::string(parent_scalars[previous].name) == parent.name)
                reject_nested_execution("microphysics private-state names are not unique");
        for (int pi = parent_physical.i_begin; pi < parent_physical.i_end; ++pi)
            for (int pj = parent_physical.j_begin; pj < parent_physical.j_end; ++pj)
                for (int k = 0; k < next_relationship.levels_z; ++k)
                    if (!valid_scalar((*parent.field)[pi][pj][k], parent.positive_definite))
                        reject_nested_execution("parent microphysics private state is invalid");
        for (int ci = 0; ci < scalar.count_i; ++ci)
            for (int cj = 0; cj < scalar.count_j; ++cj)
                for (int k = 0; k < next_relationship.levels_z; ++k)
                    if (!valid_scalar((*child.field)[ci + scalar.storage_offset_i][cj + scalar.storage_offset_j][k],
                                      child.positive_definite))
                        reject_nested_execution("child microphysics private state is invalid");
    }

    const bool residual_active = !child_runtime.theta_storage_residual.empty();
    if (residual_active)
    {
        if (!field_matches(child_runtime.theta_storage_residual, child_level.grid))
            reject_nested_execution("child theta compensation residual has the wrong shape");
        for (int ci = 0; ci < scalar.count_i; ++ci)
            for (int cj = 0; cj < scalar.count_j; ++cj)
                for (int k = 0; k < next_relationship.levels_z; ++k)
                    if (!std::isfinite(child_runtime.theta_storage_residual[ci + scalar.storage_offset_i]
                                                                           [cj + scalar.storage_offset_j][k]))
                        reject_nested_execution("child theta compensation residual is invalid");
    }

    const auto        abs_i = static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_i)));
    const auto        abs_j = static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_j)));
    const std::size_t new_cells =
        owner_count(scalar.count_i, scalar.count_j, next_relationship.levels_z) -
        owner_count(scalar.count_i - abs_i, scalar.count_j - abs_j, next_relationship.levels_z);
    prepared.private_scalar_count = child_scalars.size();
    for (std::size_t index = 0; index < child_scalars.size(); ++index)
    {
        auto&       child                             = child_scalars[index];
        const auto& parent                            = parent_scalars[index];
        prepared.child_private_fields[index]          = child.field;
        prepared.newly_admitted_private_values[index] = &movement_scratch.movement_microphysics_newly_admitted[index];
        prepared.private_discard_journals[index] =
            journal_discarded_field(*child.field,
                                    scalar,
                                    next_relationship.levels_z,
                                    child_shift_i,
                                    child_shift_j,
                                    movement_scratch.movement_microphysics_discard[index]);
        auto& newly_admitted = *prepared.newly_admitted_private_values[index];
        newly_admitted.clear();
        newly_admitted.reserve(new_cells);
        if (storage.nested_execution.active_terrain.ready)
            continue;
        std::size_t committed_values = 0;
        for (int ci = 0; ci < scalar.count_i; ++ci)
            for (int cj = 0; cj < scalar.count_j; ++cj)
            {
                const int old_ci = ci + child_shift_i;
                const int old_cj = cj + child_shift_j;
                if (old_ci >= 0 && old_ci < scalar.count_i && old_cj >= 0 && old_cj < scalar.count_j)
                    continue;
                const int parent_i = parent_physical.i_begin + next_relationship.child_origin_parent_cell_i +
                                     ci / next_relationship.parent_ratio_x;
                const int parent_j = parent_physical.j_begin + next_relationship.child_origin_parent_cell_j +
                                     cj / next_relationship.parent_ratio_y;
                for (int k = 0; k < next_relationship.levels_z; ++k)
                {
                    const float stored = (*parent.field)[parent_i][parent_j][k];
                    newly_admitted.push_back(stored);
                    const double dz        = child_level.grid.geometry.z_faces[static_cast<std::size_t>(k + 1)] -
                                             child_level.grid.geometry.z_faces[static_cast<std::size_t>(k)];
                    const double volume    = next_relationship.child_dx_m * next_relationship.child_dy_m * dz;
                    double       target    = static_cast<double>(stored) * volume;
                    double       committed = static_cast<double>(stored) * volume;
                    if (child.storage == tmv::MicrophysicsScalarStorage::DryAirMixingRatio)
                    {
                        const double dry_density = storage.fields.rho[parent_i][parent_j][k];
                        target *= dry_density;
                        committed *= dry_density;
                    }
                    if (!std::isfinite(target) || !std::isfinite(committed))
                        reject_nested_execution("microphysics private inventory is invalid");
                    prepared.report.newly_admitted_target_inventory[index] += target;
                    prepared.report.newly_admitted_committed_inventory[index] += committed;
                    ++committed_values;
                }
            }
        if (committed_values != new_cells)
            reject_nested_execution("microphysics private movement target is incomplete");
        prepared.report.newly_admitted_storage_rounding[index] =
            prepared.report.newly_admitted_target_inventory[index] -
            prepared.report.newly_admitted_committed_inventory[index];
    }

    if (storage.nested_execution.active_terrain.ready && !child_scalars.empty())
        prepare_mapped_private_movement_values(storage, admitted_density, child_shift_i, child_shift_j, prepared);

    if (residual_active)
    {
        prepared.child_theta_storage_residual         = &child_runtime.theta_storage_residual;
        prepared.newly_admitted_theta_residual_values = &movement_scratch.movement_theta_residual_newly_admitted;
        prepared.theta_residual_discard_journal =
            journal_discarded_field(child_runtime.theta_storage_residual,
                                    scalar,
                                    next_relationship.levels_z,
                                    child_shift_i,
                                    child_shift_j,
                                    movement_scratch.movement_theta_residual_discard);
        prepared.newly_admitted_theta_residual_values->assign(new_cells, 0.0f);
    }
    prepared.active                                      = !child_scalars.empty() || residual_active;
    prepared.theta_storage_residual_active               = residual_active;
    prepared.report.active                               = prepared.active;
    prepared.report.theta_storage_residual_active        = residual_active;
    prepared.report.private_scalar_count                 = child_scalars.size();
    prepared.report.newly_admitted_private_values        = new_cells * child_scalars.size();
    prepared.report.newly_admitted_theta_residual_values = residual_active ? new_cells : 0;
    return prepared;
}

PreparedNestedPrecipitationMovement
prepare_nested_precipitation_movement(SimulationStorageState&                      storage,
                                      const tmv::domain::OneWayNestRelationshipV1& next_relationship,
                                      int                                          child_shift_i,
                                      int                                          child_shift_j)
{
    PreparedNestedPrecipitationMovement      prepared;
    auto&                                    child_level   = storage.nested_execution.child;
    const std::array<std::vector<float>*, 3> parent_fields = {{&storage.surface.precipitation_rate_mm_h,
                                                               &storage.surface.accumulated_rainfall_mm,
                                                               &storage.surface.accumulated_rainfall_residual_mm}};
    prepared.child_fields                                  = {{&child_level.surface.precipitation_rate_mm_h,
                                                               &child_level.surface.accumulated_rainfall_mm,
                                                               &child_level.surface.accumulated_rainfall_residual_mm}};
    const bool parent_present =
        std::any_of(parent_fields.begin(), parent_fields.end(), [](const auto* field) { return !field->empty(); });
    const bool child_present = std::any_of(
        prepared.child_fields.begin(), prepared.child_fields.end(), [](const auto* field) { return !field->empty(); });
    if (!parent_present && !child_present)
        return prepared;

    const std::size_t parent_size =
        static_cast<std::size_t>(storage.grid.nr) * static_cast<std::size_t>(storage.grid.nth);
    const std::size_t child_size =
        static_cast<std::size_t>(child_level.grid.nr) * static_cast<std::size_t>(child_level.grid.nth);
    const bool complete_parent = std::all_of(parent_fields.begin(),
                                             parent_fields.end(),
                                             [parent_size](const auto* field) { return field->size() == parent_size; });
    const bool complete_child  = std::all_of(prepared.child_fields.begin(),
                                             prepared.child_fields.end(),
                                             [child_size](const auto* field) { return field->size() == child_size; });
    const bool empty_parent =
        std::all_of(parent_fields.begin(), parent_fields.end(), [](const auto* field) { return field->empty(); });
    const bool authoritative_empty_parent = empty_parent && storage.runtime.microphysics.scheme != nullptr &&
                                            storage.runtime.microphysics.scheme->get_scheme_name() == "none";
    const auto same_sample_time           = [](double left, double right)
    {
        const double scale = std::max({1.0, std::abs(left), std::abs(right)});
        return std::isfinite(left) && std::isfinite(right) &&
               std::abs(left - right) <= 64.0 * std::numeric_limits<double>::epsilon() * scale;
    };
    if ((!complete_parent && !authoritative_empty_parent) || !complete_child ||
        (complete_parent && !same_sample_time(storage.surface.accumulated_rainfall_last_update_s,
                                              child_level.surface.accumulated_rainfall_last_update_s)) ||
        !same_sample_time(child_level.surface.accumulated_rainfall_last_update_s, child_level.time_s))
    {
        reject_nested_execution("precipitation movement requires synchronized complete surface history");
    }

    const auto valid_surface = [](const auto& fields)
    {
        for (std::size_t index = 0; index < fields[0]->size(); ++index)
        {
            const float rate        = (*fields[0])[index];
            const float accumulated = (*fields[1])[index];
            const float residual    = (*fields[2])[index];
            if (!std::isfinite(rate) || rate < 0.0f || !std::isfinite(accumulated) || accumulated < 0.0f ||
                !std::isfinite(residual) || static_cast<double>(accumulated) + static_cast<double>(residual) < 0.0)
                return false;
        }
        return true;
    };
    if ((complete_parent && !valid_surface(parent_fields)) || !valid_surface(prepared.child_fields))
    {
        reject_nested_execution("precipitation movement surface history is invalid");
    }

    const auto parent_physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    const auto        scalar = owner_layout(HorizontalOwnerKind::Scalar, next_relationship);
    const std::size_t new_columns =
        owner_count(scalar.count_i, scalar.count_j, 1) -
        owner_count(scalar.count_i - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_i))),
                    scalar.count_j - static_cast<int>(std::abs(static_cast<std::int64_t>(child_shift_j))),
                    1);
    auto& scratch = storage.nested_execution.boundary_scratch;
    for (std::size_t field = 0; field < prepared.child_fields.size(); ++field)
    {
        prepared.newly_admitted_values[field] = &scratch.movement_precipitation_newly_admitted[field];
        auto& newly_admitted                  = *prepared.newly_admitted_values[field];
        newly_admitted.clear();
        newly_admitted.reserve(new_columns);
        prepared.discard_journals[field] = journal_discarded_surface(*prepared.child_fields[field],
                                                                     child_level.grid.nth,
                                                                     scalar,
                                                                     child_shift_i,
                                                                     child_shift_j,
                                                                     scratch.movement_precipitation_discard[field]);
    }

    const double child_area_m2             = next_relationship.child_dx_m * next_relationship.child_dy_m;
    const double depth_to_volume           = child_area_m2 * 1.0e-3;
    const double rate_to_volume_per_second = depth_to_volume / 3600.0;
    auto         accumulated_volume        = [&](const auto& fields, std::size_t index)
    { return (static_cast<double>((*fields[1])[index]) + static_cast<double>((*fields[2])[index])) * depth_to_volume; };
    auto precipitation_volume_rate = [&](const auto& fields, std::size_t index)
    { return static_cast<double>((*fields[0])[index]) * rate_to_volume_per_second; };

    auto&       report                = prepared.report;
    std::size_t committed_new_columns = 0;
    for (int ci = 0; ci < scalar.count_i; ++ci)
        for (int cj = 0; cj < scalar.count_j; ++cj)
        {
            const auto   child_index     = static_cast<std::size_t>(ci + scalar.storage_offset_i) *
                                               static_cast<std::size_t>(child_level.grid.nth) +
                                           static_cast<std::size_t>(cj + scalar.storage_offset_j);
            const double old_accumulated = accumulated_volume(prepared.child_fields, child_index);
            const double old_rate        = precipitation_volume_rate(prepared.child_fields, child_index);
            report.previous_child_accumulated_water_m3 += old_accumulated;
            report.previous_child_precipitation_volume_rate_m3s += old_rate;

            const int  old_ci   = ci + child_shift_i;
            const int  old_cj   = cj + child_shift_j;
            const bool retained = old_ci >= 0 && old_ci < scalar.count_i && old_cj >= 0 && old_cj < scalar.count_j;
            if (retained)
                continue;

            const int  parent_i     = parent_physical.i_begin + next_relationship.child_origin_parent_cell_i +
                                      ci / next_relationship.parent_ratio_x;
            const int  parent_j     = parent_physical.j_begin + next_relationship.child_origin_parent_cell_j +
                                      cj / next_relationship.parent_ratio_y;
            const auto parent_index = static_cast<std::size_t>(parent_i) * static_cast<std::size_t>(storage.grid.nth) +
                                      static_cast<std::size_t>(parent_j);
            for (std::size_t field = 0; field < parent_fields.size(); ++field)
                prepared.newly_admitted_values[field]->push_back(complete_parent ? (*parent_fields[field])[parent_index]
                                                                                 : 0.0f);
            if (complete_parent)
            {
                report.newly_admitted_parent_accumulated_water_m3 += accumulated_volume(parent_fields, parent_index);
                report.newly_admitted_parent_precipitation_volume_rate_m3s +=
                    precipitation_volume_rate(parent_fields, parent_index);
            }
            ++committed_new_columns;
        }
    if (committed_new_columns != new_columns)
        reject_nested_execution("precipitation movement target is incomplete");

    for_each_discarded_owner(scalar,
                             1,
                             child_shift_i,
                             child_shift_j,
                             [&](int i, int j, int)
                             {
                                 const auto index =
                                     static_cast<std::size_t>(i) * static_cast<std::size_t>(child_level.grid.nth) +
                                     static_cast<std::size_t>(j);
                                 report.discarded_child_accumulated_water_m3 +=
                                     accumulated_volume(prepared.child_fields, index);
                                 report.discarded_child_precipitation_volume_rate_m3s +=
                                     precipitation_volume_rate(prepared.child_fields, index);
                             });
    report.retained_child_accumulated_water_m3 =
        report.previous_child_accumulated_water_m3 - report.discarded_child_accumulated_water_m3;
    report.committed_child_accumulated_water_m3 =
        report.retained_child_accumulated_water_m3 + report.newly_admitted_parent_accumulated_water_m3;
    report.retained_child_precipitation_volume_rate_m3s =
        report.previous_child_precipitation_volume_rate_m3s - report.discarded_child_precipitation_volume_rate_m3s;
    report.committed_child_precipitation_volume_rate_m3s = report.retained_child_precipitation_volume_rate_m3s +
                                                           report.newly_admitted_parent_precipitation_volume_rate_m3s;
    report.active                                        = true;
    report.sample_time_s                                 = child_level.time_s;
    report.newly_admitted_columns                        = new_columns;
    prepared.previous_child_sample_time_s                = child_level.surface.accumulated_rainfall_last_update_s;
    prepared.active                                      = true;
    return prepared;
}

void commit_nested_radiation_movement(AtmosphericLevelStorage&                     child,
                                      const tmv::domain::OneWayNestRelationshipV1& relationship,
                                      int                                          child_shift_i,
                                      int                                          child_shift_j,
                                      const PreparedNestedRadiationMovement&       prepared) noexcept
{
    if (!prepared.active)
        return;
    const auto scalar   = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    Field3D&   tendency = child.tendencies.dtheta_dt_rad;
    Field3D&   power    = child.runtime.radiation.held_cell_power_w;
    shift_field_in_place(tendency, scalar, relationship.levels_z, child_shift_i, child_shift_j);
    shift_field_in_place(power, scalar, relationship.levels_z, child_shift_i, child_shift_j);
    fill_newly_admitted_field(tendency,
                              *prepared.newly_admitted_theta_tendency_s,
                              scalar,
                              relationship.levels_z,
                              child_shift_i,
                              child_shift_j);
    fill_newly_admitted_field(
        power, *prepared.newly_admitted_cell_power_w, scalar, relationship.levels_z, child_shift_i, child_shift_j);
    child.runtime.radiation.held_atmospheric_power_w = prepared.report.committed_child_held_power_w;
}

void commit_nested_boundary_layer_movement(AtmosphericLevelStorage&                     child,
                                           const tmv::domain::OneWayNestRelationshipV1& relationship,
                                           int                                          child_shift_i,
                                           int                                          child_shift_j,
                                           const PreparedNestedBoundaryLayerMovement&   prepared) noexcept
{
    if (!prepared.active)
        return;
    const auto                                     scalar  = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    auto&                                          runtime = child.runtime.boundary_layer;
    const std::array<Field3D*, 7>                  fields  = {{&runtime.held_u_force_n,
                                                               &runtime.held_v_force_n,
                                                               &runtime.held_theta_inventory_rate_kgk_s,
                                                               &runtime.held_vapor_mass_rate_kg_s,
                                                               &runtime.held_cloud_liquid_mass_rate_kg_s,
                                                               &runtime.held_cloud_ice_mass_rate_kg_s,
                                                               &child.tendencies.dtke_dt_pbl}};
    const std::array<const std::vector<float>*, 7> values  = {{prepared.newly_admitted_cells[0],
                                                               prepared.newly_admitted_cells[1],
                                                               prepared.newly_admitted_cells[2],
                                                               prepared.newly_admitted_cells[3],
                                                               prepared.newly_admitted_cells[4],
                                                               prepared.newly_admitted_cells[5],
                                                               prepared.newly_admitted_cells[6]}};
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        shift_field_in_place(*fields[index], scalar, relationship.levels_z, child_shift_i, child_shift_j);
        fill_newly_admitted_field(
            *fields[index], *values[index], scalar, relationship.levels_z, child_shift_i, child_shift_j);
    }

    const std::array<std::vector<float>*, 9>       surfaces       = {{&runtime.skin_theta_2d,
                                                                      &runtime.skin_qv_2d,
                                                                      &child.surface.ustar,
                                                                      &child.surface.tau_u,
                                                                      &child.surface.tau_v,
                                                                      &child.surface.sensible_heat_flux,
                                                                      &child.surface.latent_heat_flux,
                                                                      &child.surface.moisture_flux,
                                                                      &child.surface.pbl_height}};
    const std::array<const std::vector<float>*, 9> surface_values = {{prepared.newly_admitted_columns[0],
                                                                      prepared.newly_admitted_columns[1],
                                                                      prepared.newly_admitted_columns[2],
                                                                      prepared.newly_admitted_columns[3],
                                                                      prepared.newly_admitted_columns[4],
                                                                      prepared.newly_admitted_columns[5],
                                                                      prepared.newly_admitted_columns[6],
                                                                      prepared.newly_admitted_columns[7],
                                                                      prepared.newly_admitted_columns[8]}};
    for (std::size_t index = 0; index < surfaces.size(); ++index)
    {
        shift_surface_in_place(*surfaces[index], child.grid.nth, scalar, child_shift_i, child_shift_j);
        fill_newly_admitted_surface(
            *surfaces[index], *surface_values[index], child.grid.nth, scalar, child_shift_i, child_shift_j);
    }
    shift_surface_in_place(runtime.slab_columns, child.grid.nth, scalar, child_shift_i, child_shift_j);
    fill_newly_admitted_surface(runtime.slab_columns,
                                *prepared.newly_admitted_slab_columns,
                                child.grid.nth,
                                scalar,
                                child_shift_i,
                                child_shift_j);
    runtime.updated_this_step = false;
}

void commit_nested_chaos_movement(AtmosphericLevelStorage&                     child,
                                  const tmv::domain::OneWayNestRelationshipV1& relationship,
                                  int                                          child_shift_i,
                                  int                                          child_shift_j,
                                  PreparedNestedChaosMovement&                 prepared) noexcept
{
    if (!prepared.active)
        return;
    if (!prepared.pbl_base_cache_active)
        return;
    const auto                    scalar  = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    auto&                         runtime = child.runtime.chaos;
    const std::array<Field3D*, 5> fields  = {{&runtime.base_du_dt_pbl,
                                              &runtime.base_dv_dt_pbl,
                                              &runtime.base_dtheta_dt_pbl,
                                              &runtime.base_dqv_dt_pbl,
                                              &runtime.base_dtke_dt_pbl}};
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        shift_field_in_place(*fields[index], scalar, relationship.levels_z, child_shift_i, child_shift_j);
        fill_newly_admitted_field(*fields[index],
                                  *prepared.newly_admitted_cache_fields[index],
                                  scalar,
                                  relationship.levels_z,
                                  child_shift_i,
                                  child_shift_j);
    }
    const std::array<std::vector<float>*, 6> surfaces = {{&runtime.base_sfc_ustar,
                                                          &runtime.base_sfc_tau_u,
                                                          &runtime.base_sfc_tau_v,
                                                          &runtime.base_sfc_sensible_heat_flux,
                                                          &runtime.base_sfc_latent_heat_flux,
                                                          &runtime.base_sfc_moisture_flux}};
    for (std::size_t index = 0; index < surfaces.size(); ++index)
    {
        shift_surface_in_place(*surfaces[index], child.grid.nth, scalar, child_shift_i, child_shift_j);
        fill_newly_admitted_surface(*surfaces[index],
                                    *prepared.newly_admitted_cache_surfaces[index],
                                    child.grid.nth,
                                    scalar,
                                    child_shift_i,
                                    child_shift_j);
    }
}

void exchange_nested_chaos_scheme_translation(AtmosphericLevelStorage&     child,
                                              PreparedNestedChaosMovement& prepared) noexcept
{
    if (!prepared.active)
        return;
    child.runtime.chaos.scheme->exchange_prepared_horizontal_stochastic_state_translation(
        *prepared.translation_workspace);
}

void restore_discarded_field(Field3D&                   field,
                             const FieldDiscardJournal& journal,
                             const OwnerLayout&         layout,
                             int                        levels,
                             int                        shift_i,
                             int                        shift_j) noexcept;

void apply_nested_microphysics_movement(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                        int                                          child_shift_i,
                                        int                                          child_shift_j,
                                        PreparedNestedMicrophysicsMovement&          prepared) noexcept
{
    if (!prepared.active)
        return;
    const auto scalar = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    for (std::size_t index = 0; index < prepared.private_scalar_count; ++index)
    {
        Field3D& field = *prepared.child_private_fields[index];
        shift_field_in_place(field, scalar, relationship.levels_z, child_shift_i, child_shift_j);
        fill_newly_admitted_field(field,
                                  *prepared.newly_admitted_private_values[index],
                                  scalar,
                                  relationship.levels_z,
                                  child_shift_i,
                                  child_shift_j);
    }
    if (prepared.theta_storage_residual_active)
    {
        Field3D& residual = *prepared.child_theta_storage_residual;
        shift_field_in_place(residual, scalar, relationship.levels_z, child_shift_i, child_shift_j);
        fill_newly_admitted_field(residual,
                                  *prepared.newly_admitted_theta_residual_values,
                                  scalar,
                                  relationship.levels_z,
                                  child_shift_i,
                                  child_shift_j);
    }
}

void apply_nested_precipitation_movement(AtmosphericLevelStorage&                     child,
                                         const tmv::domain::OneWayNestRelationshipV1& relationship,
                                         int                                          child_shift_i,
                                         int                                          child_shift_j,
                                         const PreparedNestedPrecipitationMovement&   prepared) noexcept
{
    if (!prepared.active)
        return;
    const auto scalar = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    for (std::size_t field = 0; field < prepared.child_fields.size(); ++field)
    {
        shift_surface_in_place(*prepared.child_fields[field], child.grid.nth, scalar, child_shift_i, child_shift_j);
        fill_newly_admitted_surface(*prepared.child_fields[field],
                                    *prepared.newly_admitted_values[field],
                                    child.grid.nth,
                                    scalar,
                                    child_shift_i,
                                    child_shift_j);
    }
    child.surface.accumulated_rainfall_last_update_s = child.time_s;
}

void rollback_nested_microphysics_movement(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                           int                                          child_shift_i,
                                           int                                          child_shift_j,
                                           const PreparedNestedMicrophysicsMovement&    prepared) noexcept
{
    if (!prepared.active)
        return;
    const auto scalar = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    for (std::size_t index = 0; index < prepared.private_scalar_count; ++index)
    {
        Field3D& field = *prepared.child_private_fields[index];
        shift_field_in_place(field, scalar, relationship.levels_z, -child_shift_i, -child_shift_j);
        restore_discarded_field(field,
                                prepared.private_discard_journals[index],
                                scalar,
                                relationship.levels_z,
                                child_shift_i,
                                child_shift_j);
    }
    if (prepared.theta_storage_residual_active)
    {
        Field3D& residual = *prepared.child_theta_storage_residual;
        shift_field_in_place(residual, scalar, relationship.levels_z, -child_shift_i, -child_shift_j);
        restore_discarded_field(residual,
                                prepared.theta_residual_discard_journal,
                                scalar,
                                relationship.levels_z,
                                child_shift_i,
                                child_shift_j);
    }
}

void rollback_nested_precipitation_movement(AtmosphericLevelStorage&                     child,
                                            const tmv::domain::OneWayNestRelationshipV1& relationship,
                                            int                                          child_shift_i,
                                            int                                          child_shift_j,
                                            const PreparedNestedPrecipitationMovement&   prepared) noexcept
{
    if (!prepared.active)
        return;
    const auto scalar = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    for (std::size_t field = 0; field < prepared.child_fields.size(); ++field)
    {
        shift_surface_in_place(*prepared.child_fields[field], child.grid.nth, scalar, -child_shift_i, -child_shift_j);
        restore_discarded_surface(*prepared.child_fields[field],
                                  child.grid.nth,
                                  prepared.discard_journals[field],
                                  scalar,
                                  child_shift_i,
                                  child_shift_j);
    }
    child.surface.accumulated_rainfall_last_update_s = prepared.previous_child_sample_time_s;
}

void restore_discarded_field(Field3D&                   field,
                             const FieldDiscardJournal& journal,
                             const OwnerLayout&         layout,
                             int                        levels,
                             int                        shift_i,
                             int                        shift_j) noexcept
{
    if (journal.values == nullptr)
        std::terminate();
    std::size_t index = 0;
    for_each_discarded_owner(
        layout, levels, shift_i, shift_j, [&](int i, int j, int k) { field[i][j][k] = (*journal.values)[index++]; });
    if (index != journal.values->size())
        std::terminate();
}

std::array<Field3D*, scalar_horizontal_primary_field_count>
scalar_horizontal_primary_fields(PrognosticFieldStorage& fields)
{
    return {&fields.rho,
            &fields.p,
            &fields.w,
            &fields.theta,
            &fields.qv,
            &fields.qc,
            &fields.qr,
            &fields.qi,
            &fields.qs,
            &fields.qg,
            &fields.qh,
            &fields.tracer,
            &fields.tke};
}

std::array<const Field3D*, scalar_horizontal_primary_field_count>
scalar_horizontal_primary_fields(const PrognosticFieldStorage& fields)
{
    return {&fields.rho,
            &fields.p,
            &fields.w,
            &fields.theta,
            &fields.qv,
            &fields.qc,
            &fields.qr,
            &fields.qi,
            &fields.qs,
            &fields.qg,
            &fields.qh,
            &fields.tracer,
            &fields.tke};
}

std::array<Field3D*, complete_primary_field_count> complete_primary_fields(PrognosticFieldStorage& fields)
{
    return {&fields.rho,
            &fields.p,
            &fields.u,
            &fields.v,
            &fields.w,
            &fields.theta,
            &fields.qv,
            &fields.qc,
            &fields.qr,
            &fields.qi,
            &fields.qs,
            &fields.qg,
            &fields.qh,
            &fields.tracer,
            &fields.tke};
}

std::array<const Field3D*, complete_primary_field_count> complete_primary_fields(const PrognosticFieldStorage& fields)
{
    return {&fields.rho,
            &fields.p,
            &fields.u,
            &fields.v,
            &fields.w,
            &fields.theta,
            &fields.qv,
            &fields.qc,
            &fields.qr,
            &fields.qi,
            &fields.qs,
            &fields.qg,
            &fields.qh,
            &fields.tracer,
            &fields.tke};
}

void copy_primary_reusing_storage(PrognosticFieldStorage& destination, const PrognosticFieldStorage& source)
{
    const auto destination_fields = complete_primary_fields(destination);
    const auto source_fields      = complete_primary_fields(source);
    for (std::size_t index = 0; index < destination_fields.size(); ++index)
    {
        Field3D&       to   = *destination_fields[index];
        const Field3D& from = *source_fields[index];
        if (to.size_r() != from.size_r() || to.size_th() != from.size_th() || to.size_z() != from.size_z())
            to.resize(from.size_r(), from.size_th(), from.size_z());
        std::copy(from.data(), from.data() + from.size(), to.data());
    }
}

PrimaryDiscardJournal prepare_primary_discard_journal(const PrognosticFieldStorage&                fields,
                                                      const tmv::domain::OneWayNestRelationshipV1& relationship,
                                                      int                                          shift_i,
                                                      int                                          shift_j,
                                                      NestedBoundaryScratchStorage&                scratch)
{
    PrimaryDiscardJournal result;
    const auto            scalar        = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    const auto            x_face        = owner_layout(HorizontalOwnerKind::XFace, relationship);
    const auto            y_face        = owner_layout(HorizontalOwnerKind::YFace, relationship);
    const auto            scalar_fields = scalar_horizontal_primary_fields(fields);
    for (std::size_t index = 0; index < scalar_fields.size(); ++index)
        result.scalar_fields[index] = journal_discarded_field(*scalar_fields[index],
                                                              scalar,
                                                              relationship.levels_z,
                                                              shift_i,
                                                              shift_j,
                                                              scratch.movement_primary_discard[index]);
    result.u = journal_discarded_field(fields.u,
                                       x_face,
                                       relationship.levels_z,
                                       shift_i,
                                       shift_j,
                                       scratch.movement_primary_discard[scalar_horizontal_primary_field_count]);
    result.v = journal_discarded_field(fields.v,
                                       y_face,
                                       relationship.levels_z,
                                       shift_i,
                                       shift_j,
                                       scratch.movement_primary_discard[scalar_horizontal_primary_field_count + 1]);
    return result;
}

void shift_primary_in_place_and_fill(PrognosticFieldStorage&                      fields,
                                     const PrognosticFieldStorage&                target,
                                     const tmv::domain::OneWayNestRelationshipV1& relationship,
                                     int                                          shift_i,
                                     int                                          shift_j)
{
    const auto scalar        = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    const auto x_face        = owner_layout(HorizontalOwnerKind::XFace, relationship);
    const auto y_face        = owner_layout(HorizontalOwnerKind::YFace, relationship);
    const auto fields_scalar = scalar_horizontal_primary_fields(fields);
    const auto target_scalar = scalar_horizontal_primary_fields(target);
    for (std::size_t index = 0; index < fields_scalar.size(); ++index)
    {
        shift_field_in_place(*fields_scalar[index], scalar, relationship.levels_z, shift_i, shift_j);
        fill_newly_admitted_field(
            *fields_scalar[index], *target_scalar[index], scalar, relationship.levels_z, shift_i, shift_j);
    }
    shift_field_in_place(fields.u, x_face, relationship.levels_z, shift_i, shift_j);
    fill_newly_admitted_field(fields.u, target.u, x_face, relationship.levels_z, shift_i, shift_j);
    shift_field_in_place(fields.v, y_face, relationship.levels_z, shift_i, shift_j);
    fill_newly_admitted_field(fields.v, target.v, y_face, relationship.levels_z, shift_i, shift_j);
}

void rollback_primary_shift(PrognosticFieldStorage&                      fields,
                            const PrimaryDiscardJournal&                 journal,
                            const tmv::domain::OneWayNestRelationshipV1& relationship,
                            int                                          shift_i,
                            int                                          shift_j) noexcept
{
    const auto scalar        = owner_layout(HorizontalOwnerKind::Scalar, relationship);
    const auto x_face        = owner_layout(HorizontalOwnerKind::XFace, relationship);
    const auto y_face        = owner_layout(HorizontalOwnerKind::YFace, relationship);
    const auto scalar_fields = scalar_horizontal_primary_fields(fields);
    for (Field3D* field : scalar_fields)
        shift_field_in_place(*field, scalar, relationship.levels_z, -shift_i, -shift_j);
    shift_field_in_place(fields.u, x_face, relationship.levels_z, -shift_i, -shift_j);
    shift_field_in_place(fields.v, y_face, relationship.levels_z, -shift_i, -shift_j);
    for (std::size_t index = 0; index < scalar_fields.size(); ++index)
        restore_discarded_field(
            *scalar_fields[index], journal.scalar_fields[index], scalar, relationship.levels_z, shift_i, shift_j);
    restore_discarded_field(fields.u, journal.u, x_face, relationship.levels_z, shift_i, shift_j);
    restore_discarded_field(fields.v, journal.v, y_face, relationship.levels_z, shift_i, shift_j);
}

std::size_t journal_value_count(const PrimaryDiscardJournal& journal) noexcept
{
    if (journal.u.values == nullptr || journal.v.values == nullptr)
        std::terminate();
    std::size_t result = journal.u.values->size() + journal.v.values->size();
    for (const auto& scalar : journal.scalar_fields)
    {
        if (scalar.values == nullptr)
            std::terminate();
        result += scalar.values->size();
    }
    return result;
}

std::array<Field3D*, 6> nested_turbulence_tendency_fields(TurbulenceTendencies& tendencies) noexcept
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

std::array<Field3D*, 5> nested_diffusion_generation_fields(DiffusionRuntimeState& runtime) noexcept
{
    return {{
        &runtime.tendencies.dudt_diff,
        &runtime.tendencies.dvdt_diff,
        &runtime.tendencies.dwdt_diff,
        &runtime.tendencies.dthetadt_diff,
        &runtime.tendencies.dqvdt_diff,
    }};
}

std::array<Field3D*, 5> nested_diffusion_candidate_fields(DiffusionRuntimeState& runtime) noexcept
{
    return {{
        &runtime.scratch_u,
        &runtime.scratch_v,
        &runtime.scratch_w,
        &runtime.scratch_theta,
        &runtime.scratch_qv,
    }};
}

std::array<Field3D*, 11> nested_time_stepping_generation_fields(tmv::TimeSteppingSlowRk2Workspace& workspace) noexcept
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

std::array<Field3D*, 8> nested_dynamics_generation_fields(tmv::DynamicsRuntimeState& runtime) noexcept
{
    return {{
        &runtime.du_dt,
        &runtime.dv_dt,
        &runtime.dw_dt,
        &runtime.drho_dt,
        &runtime.dp_dt,
        &runtime.du_advection_dt,
        &runtime.dv_advection_dt,
        &runtime.dw_advection_dt,
    }};
}

void invalidate_nested_child_generation_state(AtmosphericLevelStorage& child) noexcept
{
    auto& boundary = child.runtime.boundary_conditions;
    boundary.radiative_u_west.clear();
    boundary.radiative_u_east.clear();
    boundary.radiative_v_south.clear();
    boundary.radiative_v_north.clear();

    auto& radiation = child.runtime.radiation;
    radiation.scratch_theta_tendency.fill(0.0f);
    radiation.scratch_cell_power_w.fill(0.0f);
    std::fill(radiation.scratch_column_power_w.begin(), radiation.scratch_column_power_w.end(), 0.0);
    std::fill(radiation.scratch_column_errors.begin(), radiation.scratch_column_errors.end(), 0);
    for (auto& worker : radiation.worker_contexts)
    {
        auto zero = [](auto& values) { std::fill(values.begin(), values.end(), 0); };
        zero(worker.workspace.p);
        zero(worker.workspace.p_interface);
        zero(worker.workspace.T);
        zero(worker.workspace.rho);
        zero(worker.workspace.z_center);
        zero(worker.workspace.z_interface);
        zero(worker.workspace.layer_thickness);
        zero(worker.workspace.theta_tendency);
        zero(worker.workspace.cell_power_w);
        zero(worker.workspace.tau_profile);
        zero(worker.workspace.tau_layers);
        zero(worker.workspace.flux_up);
        zero(worker.workspace.flux_down);
        zero(worker.workspace.flux_net);
        zero(worker.tendencies.dTdt_rad);
        zero(worker.tendencies.dTdt_lw);
        zero(worker.tendencies.dTdt_sw);
        worker.tendencies.atmospheric_power_lw_wm2    = 0.0;
        worker.tendencies.atmospheric_power_sw_wm2    = 0.0;
        worker.tendencies.atmospheric_power_total_wm2 = 0.0;
    }

    auto& boundary_layer = child.runtime.boundary_layer;
    child.tendencies.dtheta_dt_pbl.fill(0.0f);
    child.tendencies.dqv_dt_pbl.fill(0.0f);
    child.tendencies.du_dt_pbl.fill(0.0f);
    child.tendencies.dv_dt_pbl.fill(0.0f);
    auto& pbl_scratch = boundary_layer.refresh_scratch;
    for (Field3D* field : {&pbl_scratch.dtheta,
                           &pbl_scratch.dqv,
                           &pbl_scratch.du,
                           &pbl_scratch.dv,
                           &pbl_scratch.dtke,
                           &pbl_scratch.u_force,
                           &pbl_scratch.v_force,
                           &pbl_scratch.theta_inventory_rate,
                           &pbl_scratch.vapor_mass_rate})
        field->fill(0.0f);
    for (std::vector<float>* values : {&pbl_scratch.ustar,
                                       &pbl_scratch.tau_u,
                                       &pbl_scratch.tau_v,
                                       &pbl_scratch.sensible_heat_flux,
                                       &pbl_scratch.latent_heat_flux,
                                       &pbl_scratch.moisture_flux,
                                       &pbl_scratch.pbl_height,
                                       &pbl_scratch.skin_theta,
                                       &pbl_scratch.skin_qv})
        std::fill(values->begin(), values->end(), 0.0f);
    std::fill(pbl_scratch.slab_columns.begin(), pbl_scratch.slab_columns.end(), SlabColumnState{});
    for (auto& worker : boundary_layer.worker_contexts)
    {
        auto  zero      = [](auto& values) { std::fill(values.begin(), values.end(), 0); };
        auto& workspace = worker.workspace;
        zero(workspace.z_center);
        zero(workspace.z_interface);
        zero(workspace.layer_thickness);
        zero(workspace.centered_u);
        zero(workspace.centered_v);
        zero(workspace.tke);
        zero(workspace.mixing_length);
        zero(workspace.K_m);
        zero(workspace.K_h);
        zero(workspace.K_e);
        zero(workspace.stored_dtheta);
        zero(workspace.stored_dqv);
        zero(workspace.stored_du);
        zero(workspace.stored_dv);
        zero(workspace.stored_dtke);
        zero(workspace.stored_u_force_n);
        zero(workspace.stored_v_force_n);
        zero(workspace.stored_theta_inventory_rate_kgk_s);
        zero(workspace.stored_vapor_mass_rate_kg_s);
        worker.tendencies.clear_retain_capacity();
        worker.diagnostics.reset_retain_capacity(false);
    }

    auto& turbulence = child.runtime.turbulence;
    for (Field3D* field : nested_turbulence_tendency_fields(turbulence.held_tendencies))
        field->fill(0.0f);
    for (Field3D* field : nested_turbulence_tendency_fields(turbulence.candidate_tendencies))
        field->fill(0.0f);
    turbulence.held_tendencies.resident_source_unchanged      = false;
    turbulence.candidate_tendencies.resident_source_unchanged = false;
    turbulence.realized_transport_tendency_current            = false;
    turbulence.stochastic_transport_applied_to_current        = false;
    turbulence.last_explicit_stability_limit_s                = std::numeric_limits<double>::infinity();
    turbulence.last_explicit_substep_count                    = 0;
    turbulence.grid_metrics_scratch.dx                        = std::numeric_limits<double>::quiet_NaN();
    turbulence.grid_metrics_scratch.dy                        = std::numeric_limits<double>::quiet_NaN();
    std::fill(turbulence.grid_metrics_scratch.dz.begin(),
              turbulence.grid_metrics_scratch.dz.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(turbulence.grid_metrics_scratch.z_int.begin(),
              turbulence.grid_metrics_scratch.z_int.end(),
              std::numeric_limits<double>::quiet_NaN());
    turbulence.grid_metrics_scratch.terrain_metrics_active = false;
    turbulence.grid_metrics_scratch.terrain_metrics        = nullptr;
    turbulence.grid_metrics_scratch.terrain_topography     = nullptr;
    if (turbulence.scheme)
        turbulence.scheme->invalidate_spatial_generation();

    auto& chaos                   = child.runtime.chaos;
    chaos.grid_metrics_scratch.dx = std::numeric_limits<double>::quiet_NaN();
    chaos.grid_metrics_scratch.dy = std::numeric_limits<double>::quiet_NaN();
    std::fill(chaos.grid_metrics_scratch.dz.begin(),
              chaos.grid_metrics_scratch.dz.end(),
              std::numeric_limits<double>::quiet_NaN());
    std::fill(chaos.grid_metrics_scratch.z_int.begin(),
              chaos.grid_metrics_scratch.z_int.end(),
              std::numeric_limits<double>::quiet_NaN());
    chaos.grid_metrics_scratch.terrain_metrics_active = false;
    chaos.grid_metrics_scratch.terrain_metrics        = nullptr;
    chaos.grid_metrics_scratch.terrain_topography     = nullptr;

    auto& advection = child.runtime.advection;
    advection.scalar_scratch_a.fill(0.0f);
    advection.scalar_scratch_b.fill(0.0f);
    advection.vertical_tendencies.dqdt_adv.fill(0.0f);
    advection.vertical_tendencies.fused_dst       = nullptr;
    advection.vertical_tendencies.fused_src       = nullptr;
    advection.vertical_tendencies.fused_dt        = 0.0f;
    advection.vertical_tendencies.fused_completed = false;
    advection.tracer_advected_in_batch            = false;
    advection.hydrometeor_active.fill(false);
    advection.activity_current                             = false;
    advection.thermodynamic_boundary_reports               = {};
    advection.thermodynamic_boundary_reports_complete      = false;
    advection.thermodynamic_boundary_reports_requested     = false;
    advection.thermodynamic_moist_energy_reports           = {};
    advection.thermodynamic_moist_energy_reports_complete  = false;
    advection.thermodynamic_moist_energy_reports_requested = false;
    invalidate_nested_feedback_registers(child.runtime);

    auto& diffusion = child.runtime.diffusion;
    for (Field3D* field : nested_diffusion_generation_fields(diffusion))
        field->fill(0.0f);
    for (Field3D* field : nested_diffusion_candidate_fields(diffusion))
        field->fill(0.0f);
    diffusion.diagnostics.max_diffusion_number = 0.0;
    diffusion.diagnostics.K_effective.fill(0.0f);
    diffusion.last_momentum_work = {};

    auto& stepping = child.runtime.time_stepping;
    for (Field3D* field : nested_time_stepping_generation_fields(stepping.slow_rk2))
        field->fill(0.0f);
    stepping.config.max_acoustic_speed_ms  = 0.0;
    stepping.config.min_acoustic_spacing_m = 0.0;
    stepping.last_boundary_mass            = {};
    stepping.last_acoustic_energy          = {};
    stepping.last_slow_momentum_energy     = {};
    stepping.last_eos_pressure_energy      = {};
    stepping.last_sponge_energy            = {};
    stepping.last_final_boundary_energy    = {};
    stepping.last_completed_step_energy    = {};
    if (stepping.scheme)
        stepping.scheme->invalidate_generation_state_after_domain_recycling();

    auto& dynamics = child.runtime.dynamics;
    for (Field3D* field : nested_dynamics_generation_fields(dynamics))
        field->fill(0.0f);
    for (auto& partial : dynamics.slow_pressure_energy_columns)
        partial = {};
    std::fill(dynamics.state_relaxation_profiles.begin(), dynamics.state_relaxation_profiles.end(), 0.0f);
    std::fill(dynamics.acoustic_profile_rows.begin(), dynamics.acoustic_profile_rows.end(), 0.0f);
    dynamics.transport_context.invalidate_generation_state_after_domain_recycling();
    if (dynamics.scheme)
        dynamics.scheme->invalidate_generation_state_after_domain_recycling();

    auto& microphysics = child.runtime.microphysics;
    for (Field3D* field : {&microphysics.tendencies.dtheta_dt,
                           &microphysics.tendencies.dqv_dt,
                           &microphysics.tendencies.dqc_dt,
                           &microphysics.tendencies.dqr_dt,
                           &microphysics.tendencies.dqi_dt,
                           &microphysics.tendencies.dqs_dt,
                           &microphysics.tendencies.dqg_dt,
                           &microphysics.tendencies.dqh_dt})
        field->fill(0.0f);
    if (microphysics.scheme)
        microphysics.scheme->invalidate_transient_state_after_domain_recycling();

    auto& radar = child.runtime.radar;
    radar.selected_output.invalidate_generation_values();
    radar.polarimetric_output.invalidate_generation_values();
    for (RadarSchemeBase* scheme : {radar.selected_scheme.get(),
                                    radar.reflectivity_operator.get(),
                                    radar.velocity_operator.get(),
                                    radar.zdr_operator.get()})
        if (scheme)
            scheme->invalidate_generation_state_after_domain_recycling();
}

void require_relocatable_child_runtime(const SimulationStorageState& storage)
{
    const auto& child   = storage.nested_execution.child;
    const auto& runtime = storage.nested_execution.child.runtime;
    if (!runtime.radiation.scheme || !runtime.turbulence.scheme || !runtime.boundary_layer.scheme ||
        !runtime.chaos.scheme || !runtime.terrain.scheme || !runtime.advection.scheme || !runtime.diffusion.scheme ||
        !runtime.time_stepping.scheme || !runtime.dynamics.scheme || !storage.runtime.microphysics.scheme ||
        !runtime.microphysics.scheme)
    {
        reject_nested_execution("integer movement requires supported persistent physics ownership");
    }
    if (runtime.terrain.config.scheme_id != "none" &&
        (!nested_active_terrain_geometry_ready(storage.nested_execution) ||
         storage.nested_execution.active_terrain.geometry.sampling_version != 2 ||
         runtime.radiation.config.scheme_id != "none" || runtime.boundary_layer.config.scheme_id != "none" ||
         runtime.turbulence.config.scheme_id != "none" || runtime.chaos.config.scheme_id != "none"))
        reject_nested_execution("mapped movement requires anchored geometry and supported persistent physics owners");
    const std::string microphysics_name = runtime.microphysics.scheme->get_scheme_name();
    if (microphysics_name != storage.runtime.microphysics.scheme->get_scheme_name() ||
        (microphysics_name != "none" && microphysics_name != "kessler" && microphysics_name != "lin" &&
         microphysics_name != "thompson" && microphysics_name != "milbrandt"))
        reject_nested_execution("integer movement requires supported microphysics ownership");
    const std::string chaos_name = runtime.chaos.scheme->name();
    if (chaos_name != runtime.chaos.config.scheme_id ||
        (chaos_name != "none" && chaos_name != "boundary_layer" && chaos_name != "full_stochastic"))
        reject_nested_execution("integer movement requires Earth-addressed stochastic ownership");
    const std::string boundary_layer_name = runtime.boundary_layer.scheme->name();
    if (boundary_layer_name != runtime.boundary_layer.config.scheme_id ||
        (boundary_layer_name != "none" && boundary_layer_name != "slab" && boundary_layer_name != "ysu" &&
         boundary_layer_name != "mynn"))
        reject_nested_execution("integer movement requires supported boundary-layer ownership");
    if (runtime.radiation.scheme->name() != runtime.radiation.config.scheme_id ||
        (runtime.radiation.config.scheme_id != "none" && runtime.radiation.config.scheme_id != "simple_grey"))
        reject_nested_execution("integer movement requires supported radiation ownership");
    if (runtime.turbulence.scheme->name() != runtime.turbulence.config.scheme_id ||
        (runtime.turbulence.config.scheme_id != "none" &&
         !runtime.turbulence.scheme->supports_spatial_generation_invalidation()))
        reject_nested_execution("integer movement requires turbulence generation invalidation");
    if (!runtime.time_stepping.scheme->supports_domain_recycling_invalidation() ||
        !runtime.dynamics.scheme->supports_domain_recycling_invalidation())
        reject_nested_execution("integer movement requires solver generation invalidation");
    const auto& radar = runtime.radar;
    if (radar.selected_scheme &&
        (radar.selected_config.scheme_id != radar.selected_scheme->name() || radar.nr != child.grid.nr ||
         radar.nth != child.grid.nth || radar.nz != child.grid.nz))
        reject_nested_execution("integer movement requires child-grid radar ownership");
    for (const RadarSchemeBase* scheme : {radar.selected_scheme.get(),
                                          radar.reflectivity_operator.get(),
                                          radar.velocity_operator.get(),
                                          radar.zdr_operator.get()})
        if (scheme && !scheme->supports_domain_recycling_invalidation())
            reject_nested_execution("integer movement requires radar generation invalidation");
    const auto field_matches = [&](const Field3D& field)
    { return field.size_r() == child.grid.nr && field.size_th() == child.grid.nth && field.size_z() == child.grid.nz; };
    const Field3D* tendencies[] = {&child.tendencies.dtheta_dt_rad,
                                   &child.tendencies.dtheta_dt_pbl,
                                   &child.tendencies.dqv_dt_pbl,
                                   &child.tendencies.du_dt_pbl,
                                   &child.tendencies.dv_dt_pbl,
                                   &child.tendencies.dtke_dt_pbl};
    if (!std::all_of(
            std::begin(tendencies), std::end(tendencies), [&](const Field3D* field) { return field_matches(*field); }))
        reject_nested_execution("integer movement requires complete child source storage");
}

} // namespace

namespace
{
void prepare_nested_child_physical_primary_state(SimulationStorageState&                  storage,
                                                 const PrognosticFieldStorage&            previous_parent_fields,
                                                 PrognosticFieldStorage&                  candidate_fields,
                                                 NestedChildInitializationReport&         report,
                                                 tmv::domain::NestedActiveTerrainStateV1* active_candidate);

std::array<double, 2> nested_horizontal_ghost_impulse(const tmv::domain::OneWayNestRelationshipV1& r,
                                                      const std::vector<double>&                   thickness,
                                                      const PrognosticFieldStorage&                fields,
                                                      bool                                         magnitude)
{
    std::array<tmv::diagnostics::CompensatedSum, 2> result;
    const auto term = [magnitude](double value) { return magnitude ? std::abs(value) : value; };
    for (int j = 1; j <= r.child_cells_y; ++j)
        for (int k = 0; k < r.levels_z; ++k)
        {
            const double half_volume = .5 * r.child_dx_m * r.child_dy_m * thickness[k];
            result[0].add(term(half_volume * fields.rho(0, j, k) * fields.u(0, j, k)));
            result[0].add(term(half_volume * fields.rho(r.child_cells_x + 1, j, k) * fields.u(r.child_cells_x, j, k)));
        }
    for (int i = 1; i <= r.child_cells_x; ++i)
        for (int k = 0; k < r.levels_z; ++k)
        {
            const double half_volume = .5 * r.child_dx_m * r.child_dy_m * thickness[k];
            result[1].add(term(half_volume * fields.rho(i, 0, k) * fields.v(i, 0, k)));
            result[1].add(term(half_volume * fields.rho(i, r.child_cells_y + 1, k) * fields.v(i, r.child_cells_y, k)));
        }
    return {result[0].value(), result[1].value()};
}

struct NestedRelocationEnergy
{
    std::array<double, 7> values{}, magnitudes{};
};

std::array<double, 7> relocation_energy_values(const MoistMixtureEnergyBudgetV2& value)
{
    return {value.dry_internal,
            value.water_sensible,
            value.phase_reference,
            value.resolved_kinetic,
            value.gravitational,
            value.subgrid_tke,
            value.total};
}

NestedRelocationEnergy nested_relocation_energy(SimulationStorageState& storage,
                                                PrognosticFieldStorage& fields,
                                                int                     shift_i = 0,
                                                int                     shift_j = 0,
                                                bool                    outside = false)
{
    auto&      child = storage.nested_execution.child;
    const auto original =
        make_simulation_state_view(child, storage.configuration, &storage.runtime.compute, &storage.runtime.parity);
    SimulationStateView view{original.grid,
                             std::apply([&](const auto&... member)
                                        { return PrognosticFieldsView{member.get(fields)...}; },
                                        tmv::fields::prognostic),
                             original.base,
                             original.tendencies,
                             original.coriolis_f,
                             original.time_s,
                             original.surface};
    child.runtime.bind(view);
    view.process_grid_metrics               = original.process_grid_metrics;
    view.configuration                      = original.configuration;
    view.domain_frame                       = original.domain_frame;
    const auto&                           r = storage.nested_execution.relationship;
    const HorizontalEnergyBudgetExclusion retained{std::max(1, 1 - shift_i),
                                                   std::min(r.child_cells_x + 1, r.child_cells_x + 1 - shift_i),
                                                   std::max(1, 1 - shift_j),
                                                   std::min(r.child_cells_y + 1, r.child_cells_y + 1 - shift_j)};
    const auto                            energy =
        outside ? compute_moist_mixture_energy_v2_excluding_horizontal_rectangle_deterministic(view, retained)
                : compute_moist_mixture_energy_v2_deterministic(view);
    return {relocation_energy_values(energy), moist_mixture_energy_v2_magnitudes(view, outside ? &retained : nullptr)};
}

using NestedRelaxationCandidateValidator = void (*)(PrognosticFieldStorage&, const NestedChildRelaxationReport&, void*);

double nested_ground_vertical_impulse(const tmv::domain::OneWayNestRelationshipV1& relationship,
                                      const TerrainMetrics3D&                      metrics,
                                      const PrognosticFieldStorage&                fields)
{
    tmv::diagnostics::CompensatedSum total;
    for (int i = 1; i <= relationship.child_cells_x; ++i)
        for (int j = 1; j <= relationship.child_cells_y; ++j)
        {
            const double volume = metrics.J(i, j, 0) * metrics.dx * metrics.dy * metrics.dzeta;
            const double mass   = volume * static_cast<double>(fields.rho(i, j, 0));
            const double ground =
                tmv::terrain_transport::cgrid_ground_vertical_velocity(metrics, fields.u, fields.v, i, j);
            total.add(.5 * mass * ground);
        }
    if (!std::isfinite(total.value()))
        reject_nested_execution("mapped ground momentum is not finite");
    return total.value();
}

void apply_nested_child_physical_relaxation_implementation(
    SimulationStorageState&                      storage,
    const PrognosticFieldStorage&                previous_parent_fields,
    const tmv::domain::NestedRelaxationConfigV1& config,
    double                                       child_step_s,
    NestedChildRelaxationReport&                 report,
    const NestedParentMicrophysicsSnapshotV1*    previous_parent_microphysics,
    tmv::MicrophysicsScheme*                     prepared_child_microphysics,
    Field3D*                                     prepared_theta_storage_residual,
    NestedRelaxationCandidateValidator           validator         = nullptr,
    void*                                        validator_context = nullptr);
} // namespace

void capture_nested_parent_microphysics_snapshot(SimulationStorageState&             storage,
                                                 NestedParentMicrophysicsSnapshotV1& snapshot)
{
    auto& runtime = storage.runtime.microphysics;
    if (!runtime.scheme || !std::isfinite(storage.time_s))
        reject_nested_execution("parent microphysics snapshot requires initialized finite state");
    tmv::prepare_microphysics_prognostics(make_simulation_state_view(storage), *runtime.scheme);
    auto scalars = runtime.scheme->prognostic_scalars_for_transport();
    if (scalars.size() > NestedParentMicrophysicsSnapshotV1::maximum_scalars)
        reject_nested_execution("parent microphysics snapshot exceeds its scalar capacity");

    const auto physical =
        physical_scalar_domain(storage.grid.nr, storage.grid.nth, storage.grid.nz, StaggerType::CGrid);
    for (std::size_t index = 0; index < scalars.size(); ++index)
    {
        const auto& scalar = scalars[index];
        if (!scalar.field || scalar.field->size_r() != storage.grid.nr || scalar.field->size_th() != storage.grid.nth ||
            scalar.field->size_z() != storage.grid.nz)
            reject_nested_execution("parent microphysics snapshot field has the wrong shape");
        static_cast<void>(bounded_private_scalar_name(scalar.name));
        for (int i = physical.i_begin; i < physical.i_end; ++i)
            for (int j = physical.j_begin; j < physical.j_end; ++j)
                for (int k = physical.k_begin; k < physical.k_end; ++k)
                {
                    const float value = (*scalar.field)[i][j][k];
                    if (!std::isfinite(value) || (scalar.positive_definite && value < 0.0f))
                        reject_nested_execution("parent microphysics snapshot field is invalid");
                }
    }
    snapshot.available     = false;
    snapshot.sample_time_s = storage.time_s;
    snapshot.scheme_name   = bounded_private_scalar_name(runtime.scheme->get_scheme_name().c_str());
    snapshot.scalar_count  = scalars.size();
    for (std::size_t index = 0; index < scalars.size(); ++index)
    {
        const auto& scalar                = scalars[index];
        snapshot.name[index]              = bounded_private_scalar_name(scalar.name);
        snapshot.storage_semantic[index]  = static_cast<std::uint8_t>(scalar.storage);
        snapshot.positive_definite[index] = scalar.positive_definite;
        snapshot.fields[index]            = *scalar.field;
    }
    snapshot.available = true;
}

void prepare_nested_child_process_grid_metrics(NestedExecutionStorage& nested)
{
    if (!nested.relationship_ready)
        reject_nested_execution("child process metrics require a prepared relationship");
    const bool active = nested_active_terrain_geometry_ready(nested);
    if ((nested.child.runtime.terrain.config.scheme_id != "none") != active)
        reject_nested_execution("child terrain selection and mapped geometry readiness differ");
    const auto& faces = nested.child.grid.geometry.z_faces;
    if (faces.size() != static_cast<std::size_t>(nested.child.grid.nz + 1))
        reject_nested_execution("child process metrics require complete vertical interfaces");

    GridMetrics candidate;
    candidate.dx                     = nested.child.grid.dx;
    candidate.dy                     = nested.child.grid.dy;
    candidate.z_int                  = faces;
    candidate.terrain_metrics_active = active;
    candidate.staggering = nested.child.grid.geometry.staggered
        ? StaggerType::CGrid : StaggerType::Collocated;
    if (active)
    {
        candidate.terrain_metrics = &nested.child.runtime.terrain.metrics;
        candidate.terrain_topography = &nested.child.runtime.terrain.topography;
        tmv::terrain_transport::validate_geometry(*candidate.terrain_metrics);
    }
    candidate.dz.resize(static_cast<std::size_t>(nested.child.grid.nz));
    for (int k = 0; k < nested.child.grid.nz; ++k)
    {
        const double thickness = faces[static_cast<std::size_t>(k + 1)] - faces[static_cast<std::size_t>(k)];
        if (!std::isfinite(thickness) || thickness <= 0.0)
            reject_nested_execution("child vertical layer thickness must be positive and finite");
        candidate.dz[static_cast<std::size_t>(k)] = thickness;
        if (active)
        {
            double minimum = std::numeric_limits<double>::infinity();
            for (int i = 0; i < nested.child.grid.nr; ++i)
                for (int j = 0; j < nested.child.grid.nth; ++j)
                    minimum = std::min(minimum,
                        candidate.terrain_metrics->z_face(i, j, k + 1)
                        - candidate.terrain_metrics->z_face(i, j, k));
            if (!std::isfinite(minimum) || minimum <= 0.0)
                reject_nested_execution("mapped child thickness must be positive and finite");
            candidate.dz[static_cast<std::size_t>(k)] = minimum;
        }
    }
    nested.child.process_grid_metrics = std::move(candidate);
}

SimulationStateView nested_child_execution_state(SimulationStorageState& storage)
{
    return make_simulation_state_view(storage.nested_execution.child, storage.configuration,
        &storage.runtime.compute, &storage.runtime.parity);
}

ScopedNestedChildResidency::ScopedNestedChildResidency(SimulationStorageState& storage) : storage_(storage)
{
    if (storage_.nested_execution.child_execution_active)
        reject_nested_execution("child execution is already active");
    storage_.nested_execution.child_execution_active = true;
    if (storage_.runtime.compute.backend)
        storage_.runtime.compute.backend->select_residency_scope(ComputeResidencyScope::Child);
}

ScopedNestedChildResidency::~ScopedNestedChildResidency() noexcept
{
    if (storage_.runtime.compute.backend)
    {
        storage_.runtime.compute.backend->invalidate_residency_scope(ComputeResidencyScope::Child);
        storage_.runtime.compute.backend->select_residency_scope(ComputeResidencyScope::Root);
    }
    storage_.nested_execution.child_execution_active = false;
}

void advance_nested_child_substep(SimulationStorageState&                      storage,
                                  const PrognosticFieldStorage&                previous_parent_fields,
                                  const tmv::domain::NestedRelaxationConfigV1& relaxation,
                                  NestedChildStepKernel                        kernel,
                                  void*                                        kernel_context,
                                  NestedChildSubstepReport&                    report,
                                  const NestedParentMicrophysicsSnapshotV1*    previous_parent_microphysics)
{
    if (kernel == nullptr)
        reject_nested_execution("child substep requires an execution kernel");
    require_valid_nested_execution_relationship(storage);
    auto& nested = storage.nested_execution;
    if (nested.child_execution_active)
        reject_nested_execution("child execution is already active");
    if (!nested.parent_boundary_state_ready)
        reject_nested_execution("child substep requires complete parent boundary state");
    if (nested.parent_time_bracket.completed_child_substeps >= nested.relationship.parent_time_step_ratio)
        reject_nested_execution("child subcycle is already synchronized");

    const double start_time = nested.child.time_s;
    const double child_step =
        (nested.parent_time_bracket.current_parent_time_s - nested.parent_time_bracket.previous_parent_time_s) /
        nested.relationship.parent_time_step_ratio;
    if (!std::isfinite(child_step) || child_step <= 0.0 || nested.child.grid.dt != child_step)
        reject_nested_execution("child substep duration disagrees with the realized bracket");

    auto& prior_child = nested.substep_rollback.get();
    tmv::state_ownership::capture_level(prior_child, nested.child);
    const auto prior_interface_source = nested.interface_source;
    const auto   prior_bracket                   = nested.parent_time_bracket;
    const double prior_active_parent_time        = nested.active_terrain.parent_time_s;
    const double prior_active_child_time         = nested.active_terrain.child_time_s;
    const double prior_active_ledger_parent_time = nested.active_terrain.ledger.parent_time_s;
    const double prior_active_ledger_child_time  = nested.active_terrain.ledger.child_time_s;
    tmv::BudgetPublicationTransaction publication;
    try
    {
        {
            ScopedNestedChildResidency residency(storage);
            const auto child_state = nested_child_execution_state(storage);
            kernel(child_state, child_step, start_time, kernel_context);
            if (nested.child.time_s != start_time)
            {
                nested.child.time_s = start_time;
                reject_nested_execution("child execution kernel must not mutate the level clock");
            }
        }
        require_valid_nested_execution_relationship(storage);

        auto next_bracket = nested.parent_time_bracket;
        tmv::domain::advance_child_substep(nested.relationship, next_bracket);
        const double end_time                        = tmv::domain::nested_child_time_s(nested.relationship, next_bracket);
        nested.parent_time_bracket                   = next_bracket;
        nested.child.time_s                          = end_time;
        nested.child.coriolis_f                      = expected_child_coriolis_f(storage, nested);
        if (nested.active_terrain.ready)
        {
            nested.active_terrain.parent_time_s        = end_time;
            nested.active_terrain.child_time_s         = end_time;
            nested.active_terrain.ledger.parent_time_s = end_time;
            nested.active_terrain.ledger.child_time_s  = end_time;
        }

        NestedChildRelaxationReport interface_source;
        apply_nested_child_physical_relaxation(
            storage, previous_parent_fields, relaxation, child_step, interface_source, previous_parent_microphysics);
        NestedChildSubstepReport candidate;
        candidate.start_time_s             = start_time;
        candidate.end_time_s               = end_time;
        candidate.completed_child_substeps = nested.parent_time_bracket.completed_child_substeps;
        candidate.interface_source         = interface_source;
        publication.commit();
        report                             = std::move(candidate);
    }
    catch (...)
    {
        tmv::state_ownership::exchange_levels(nested.child, prior_child);
        nested.interface_source = prior_interface_source;
        nested.parent_time_bracket                 = prior_bracket;
        nested.active_terrain.parent_time_s        = prior_active_parent_time;
        nested.active_terrain.child_time_s         = prior_active_child_time;
        nested.active_terrain.ledger.parent_time_s = prior_active_ledger_parent_time;
        nested.active_terrain.ledger.child_time_s  = prior_active_ledger_child_time;
        throw;
    }
}

void prepare_nested_execution_relationship(SimulationStorageState&                      storage,
                                           const tmv::domain::OneWayNestRelationshipV1& relationship,
                                           double                                       previous_parent_time_s)
{
    require_same_grid_contract(storage.grid, relationship);
    storage.configuration.domain_frame.validate();

    NestedExecutionStorage candidate;
    candidate.relationship = relationship;
    candidate.parent_time_bracket =
        tmv::domain::prepare_parent_time_bracket(relationship, previous_parent_time_s, storage.time_s);
    const double realized_child_dt = (storage.time_s - previous_parent_time_s) / relationship.parent_time_step_ratio;
    candidate.child                = AtmosphericLevelStorage({relationship.child_cells_x + 2,
                                                              relationship.child_cells_y + 2,
                                                              relationship.levels_z,
                                                              relationship.child_dx_m,
                                                              storage.grid.dz,
                                                              realized_child_dt,
                                                              relationship.child_dy_m});
    candidate.child.grid.geometry.initialize(candidate.child.grid.nr,
                                             candidate.child.grid.nth,
                                             candidate.child.grid.nz,
                                             candidate.child.grid.dx,
                                             candidate.child.grid.dz,
                                             candidate.child.grid.dy,
                                             storage.grid.geometry.coord,
                                             storage.grid.geometry.staggered ? StaggerType::CGrid
                                                                             : StaggerType::Collocated);
    candidate.child.grid.geometry.z       = storage.grid.geometry.z;
    candidate.child.grid.geometry.z_faces = storage.grid.geometry.z_faces;
    candidate.child.time_s                = previous_parent_time_s;
    place_child_frame_from_relationship(storage, candidate);
    candidate.initial_parent_origin_i = relationship.child_origin_parent_cell_i;
    candidate.initial_parent_origin_j = relationship.child_origin_parent_cell_j;
    candidate.relationship_ready      = true;

    prepare_stationary_nested_active_terrain_geometry(storage, candidate, previous_parent_time_s);

    prepare_nested_child_process_grid_metrics(candidate);

    require_valid_nested_execution_relationship(storage, candidate, storage.time_s);
    storage.nested_execution = std::move(candidate);
    prepare_nested_child_process_grid_metrics(storage.nested_execution);
}

void require_valid_nested_execution_relationship(const SimulationStorageState& storage)
{
    require_valid_nested_execution_relationship(storage, storage.nested_execution, storage.time_s);
}

void roll_nested_parent_interval(SimulationStorageState& storage)
{
    if (storage.nested_execution.child_execution_active)
        reject_nested_execution("cannot open a parent interval while the child is process-bound");

    auto& nested = storage.nested_execution;
    if (!nested.relationship_ready ||
        nested.parent_time_bracket.completed_child_substeps != nested.relationship.parent_time_step_ratio ||
        nested.child.time_s != nested.parent_time_bracket.current_parent_time_s)
    {
        reject_nested_execution("the previous parent interval must be synchronized before rolling");
    }

    const double previous_parent_time_s = nested.parent_time_bracket.current_parent_time_s;
    require_valid_nested_execution_relationship(storage, nested, previous_parent_time_s);
    if (!std::isfinite(storage.time_s) || storage.time_s <= previous_parent_time_s)
        reject_nested_execution("the parent clock must advance before opening its next interval");

    const auto   prior_bracket    = nested.parent_time_bracket;
    const double prior_child_dt_s = nested.child.grid.dt;
    try
    {
        tmv::domain::roll_parent_time_bracket(nested.relationship, nested.parent_time_bracket, storage.time_s);
        nested.child.grid.dt =
            (nested.parent_time_bracket.current_parent_time_s - nested.parent_time_bracket.previous_parent_time_s) /
            nested.relationship.parent_time_step_ratio;
        require_valid_nested_execution_relationship(storage, nested, storage.time_s);
    }
    catch (...)
    {
        nested.parent_time_bracket = prior_bracket;
        nested.child.grid.dt       = prior_child_dt_s;
        throw;
    }
}

void move_nested_child_integer_parent_cells(SimulationStorageState&                      storage,
                                            int                                          parent_shift_i,
                                            int                                          parent_shift_j,
                                            const tmv::domain::NestedRelaxationConfigV1& relaxation,
                                            NestedChildMovementReportV5&                 report)
{
    if (storage.nested_execution.child_execution_active)
        reject_nested_execution("cannot move a child while it is process-bound");
    require_valid_nested_execution_relationship(storage);
    require_relocatable_child_runtime(storage);

    const bool two_way = storage.configuration.nesting.feedback == tmv::init::config::NestingFeedback::TwoWay;
    if (!nested_two_way_feedback_interval_closed(storage))
        reject_nested_execution("integer movement requires fully consumed two-way feedback registers");
    const auto& current = storage.nested_execution;
    if (!current.relationship_ready || !current.reference_state_ready || !current.physical_primary_state_ready ||
        !current.parent_boundary_state_ready)
        reject_nested_execution("integer movement requires complete step-ready child state");
    if (parent_shift_i == 0 && parent_shift_j == 0)
        reject_nested_execution("integer movement requires a nonzero shift");
    if (current.parent_time_bracket.completed_child_substeps != current.relationship.parent_time_step_ratio ||
        current.child.time_s != storage.time_s)
        reject_nested_execution("integer movement requires a synchronized parent/child endpoint");
    if (current.movement_sequence == std::numeric_limits<std::uint64_t>::max())
        reject_nested_execution("movement sequence is exhausted");

    const auto child_shift =
        tmv::domain::child_shift_for_parent_move(current.relationship, parent_shift_i, parent_shift_j);
    const auto abs_i = std::abs(static_cast<std::int64_t>(child_shift.i));
    const auto abs_j = std::abs(static_cast<std::int64_t>(child_shift.j));
    if (abs_i >= current.relationship.child_cells_x || abs_j >= current.relationship.child_cells_y)
        reject_nested_execution("integer movement must retain a non-empty child overlap");

    auto&      partition_runtime = storage.runtime;
    const bool child_partition_active =
        partition_runtime.storm_selected_partition_available &&
        partition_runtime.storm_selected_partition_nx == static_cast<std::size_t>(current.relationship.child_cells_x) &&
        partition_runtime.storm_selected_partition_ny == static_cast<std::size_t>(current.relationship.child_cells_y);
    auto& shifted_partition = storage.nested_execution.boundary_scratch.movement_storm_partition_candidate;
    if (child_partition_active)
    {
        const std::size_t nx = partition_runtime.storm_selected_partition_nx;
        const std::size_t ny = partition_runtime.storm_selected_partition_ny;
        if (partition_runtime.storm_selected_partition.size() != nx * ny)
            reject_nested_execution("child storm partition shape is inconsistent");
        shifted_partition.assign(nx * ny, 0U);
        std::size_t retained_selected_cells = 0;
        for (std::size_t destination_i = 0; destination_i < nx; ++destination_i)
            for (std::size_t destination_j = 0; destination_j < ny; ++destination_j)
            {
                const auto source_i =
                    static_cast<std::int64_t>(destination_i) + static_cast<std::int64_t>(child_shift.i);
                const auto source_j =
                    static_cast<std::int64_t>(destination_j) + static_cast<std::int64_t>(child_shift.j);
                if (source_i < 0 || source_j < 0 || source_i >= static_cast<std::int64_t>(nx) ||
                    source_j >= static_cast<std::int64_t>(ny))
                    continue;
                const std::uint8_t value =
                    partition_runtime.storm_selected_partition[static_cast<std::size_t>(source_i) * ny +
                                                               static_cast<std::size_t>(source_j)];
                if (value > 1U)
                    reject_nested_execution("child storm partition contains a non-binary value");
                shifted_partition[destination_i * ny + destination_j] = value;
                retained_selected_cells += value;
            }
        if (retained_selected_cells == 0)
            reject_nested_execution("integer movement would retire the selected child storm partition");
    }

    const auto next_origin_i =
        static_cast<std::int64_t>(current.relationship.child_origin_parent_cell_i) + parent_shift_i;
    const auto next_origin_j =
        static_cast<std::int64_t>(current.relationship.child_origin_parent_cell_j) + parent_shift_j;
    const auto next_cumulative_i = static_cast<std::int64_t>(current.cumulative_parent_shift_i) + parent_shift_i;
    const auto next_cumulative_j = static_cast<std::int64_t>(current.cumulative_parent_shift_j) + parent_shift_j;
    const auto in_int            = [](std::int64_t value)
    { return value >= std::numeric_limits<int>::min() && value <= std::numeric_limits<int>::max(); };
    if (!in_int(next_origin_i) || !in_int(next_origin_j) || !in_int(next_cumulative_i) || !in_int(next_cumulative_j))
        reject_nested_execution("integer movement provenance overflows");

    const auto next_relationship =
        shifted_relationship(current.relationship, static_cast<int>(next_origin_i), static_cast<int>(next_origin_j));
    const bool                              mapped = nested_active_terrain_geometry_ready(current);
    tmv::domain::NestedActiveTerrainStateV1 terrain_candidate;
    TerrainMetrics3D                        metric_candidate;
    Topography2D                            topography_candidate;
    GridMetrics                             original_process_metrics;
    if (mapped)
    {
        terrain_candidate = current.active_terrain;
        terrain_candidate.geometry =
            tmv::domain::build_stationary_nested_terrain_geometry(current.active_terrain.geometry.config,
                                                                  next_relationship,
                                                                  current.active_terrain.geometry.parent_origin_earth_m,
                                                                  current.active_terrain.generation,
                                                                  2);
        const auto& before = current.child.runtime.terrain.metrics;
        const auto& after  = terrain_candidate.geometry.child_storage.metrics;
        for (int i = 1; i <= next_relationship.child_cells_x; ++i)
            for (int j = 1; j <= next_relationship.child_cells_y; ++j)
            {
                const int si = i + child_shift.i, sj = j + child_shift.j;
                if (si < 1 || si > next_relationship.child_cells_x || sj < 1 || sj > next_relationship.child_cells_y)
                    continue;
                for (int k = 0; k < next_relationship.levels_z; ++k)
                    if (before.J(si, sj, k) != after.J(i, j, k) || before.z(si, sj, k) != after.z(i, j, k) ||
                        before.z_face(si, sj, k) != after.z_face(i, j, k) ||
                        before.z_face(si, sj, k + 1) != after.z_face(i, j, k + 1))
                        reject_nested_execution("mapped movement changes retained physical cell geometry");
            }
        metric_candidate         = after;
        topography_candidate     = terrain_candidate.geometry.child_storage.topography;
        original_process_metrics = current.child.process_grid_metrics;
    }
    const auto* movement_metrics = mapped ? &current.child.runtime.terrain.metrics : nullptr;

    const PreparedNestedRadiationMovement prepared_radiation =
        prepare_nested_radiation_movement(storage, next_relationship, child_shift.i, child_shift.j);
    const PreparedNestedBoundaryLayerMovement prepared_boundary_layer =
        prepare_nested_boundary_layer_movement(storage, next_relationship, child_shift.i, child_shift.j);
    PreparedNestedChaosMovement prepared_chaos =
        prepare_nested_chaos_movement(storage, next_relationship, child_shift.i, child_shift.j);
    PreparedNestedMicrophysicsMovement        prepared_microphysics;
    const PreparedNestedPrecipitationMovement prepared_precipitation =
        prepare_nested_precipitation_movement(storage, next_relationship, child_shift.i, child_shift.j);
    const auto                  original_relationship      = current.relationship;
    const auto                  original_frame             = current.child.domain_frame;
    const double                original_coriolis_f        = current.child.coriolis_f;
    const auto                  original_movement_sequence = current.movement_sequence;
    const auto                  original_cumulative_i      = current.cumulative_parent_shift_i;
    const auto                  original_cumulative_j      = current.cumulative_parent_shift_j;
    const auto                  original_interface_source  = current.interface_source;
    const bool                  original_physical_ready    = current.physical_primary_state_ready;
    const bool                  original_boundary_ready    = current.parent_boundary_state_ready;
    NestedChildMovementReportV5 candidate_report;
    candidate_report.radiation                 = prepared_radiation.report;
    candidate_report.boundary_layer            = prepared_boundary_layer.report;
    candidate_report.chaos                     = prepared_chaos.report;
    candidate_report.precipitation             = prepared_precipitation.report;
    candidate_report.movement_sequence         = current.movement_sequence + 1;
    candidate_report.parent_shift_i            = parent_shift_i;
    candidate_report.parent_shift_j            = parent_shift_j;
    candidate_report.child_shift_i             = child_shift.i;
    candidate_report.child_shift_j             = child_shift.j;
    candidate_report.previous_parent_origin_i  = current.relationship.child_origin_parent_cell_i;
    candidate_report.previous_parent_origin_j  = current.relationship.child_origin_parent_cell_j;
    candidate_report.committed_parent_origin_i = next_relationship.child_origin_parent_cell_i;
    candidate_report.committed_parent_origin_j = next_relationship.child_origin_parent_cell_j;

    const int nx                                 = current.relationship.child_cells_x;
    const int ny                                 = current.relationship.child_cells_y;
    const int nz                                 = current.relationship.levels_z;
    const int retained_x                         = nx - static_cast<int>(abs_i);
    const int retained_y                         = ny - static_cast<int>(abs_j);
    candidate_report.overlap_scalar_cells        = owner_count(retained_x, retained_y, nz);
    candidate_report.newly_admitted_scalar_cells = owner_count(nx, ny, nz) - candidate_report.overlap_scalar_cells;
    candidate_report.overlap_x_faces             = owner_count(nx + 1 - static_cast<int>(abs_i), retained_y, nz);
    candidate_report.newly_admitted_x_faces      = owner_count(nx + 1, ny, nz) - candidate_report.overlap_x_faces;
    candidate_report.overlap_y_faces             = owner_count(retained_x, ny + 1 - static_cast<int>(abs_j), nz);
    candidate_report.newly_admitted_y_faces      = owner_count(nx, ny + 1, nz) - candidate_report.overlap_y_faces;
    candidate_report.overlap_z_faces             = candidate_report.overlap_scalar_cells;
    candidate_report.newly_admitted_z_faces      = candidate_report.newly_admitted_scalar_cells;
    candidate_report.full_primary_owner_values   = scalar_horizontal_primary_field_count * owner_count(nx, ny, nz) +
                                                   owner_count(nx + 1, ny, nz) + owner_count(nx, ny + 1, nz);

    struct RelocationValidation
    {
        SimulationStorageState*                      storage;
        NestedTwoWayRelocationReportV1*              report;
        const tmv::domain::OneWayNestRelationshipV1* relationship;
        const std::vector<double>*                   thickness;
        std::array<double, 5>                        next_inventory{};
        std::array<double, 2>                        ghost_before{};
        double                                       ground_before = 0.0;
        const TerrainMetrics3D*                      metrics       = nullptr;
        NestedTwoWayRelocationReportV1               committed_report;
    } validation{
        &storage, &candidate_report.two_way_inventory, &next_relationship, &current.child.process_grid_metrics.dz};
    validation.metrics                                     = movement_metrics;
    NestedRelaxationCandidateValidator validate_relocation = nullptr;
    if (two_way)
    {
        auto& inventory          = candidate_report.two_way_inventory;
        inventory                = current.last_two_way_feedback.relocation;
        const auto energy_before = nested_relocation_energy(storage, storage.nested_execution.child.fields);
        if (!inventory.complete)
        {
            const auto before = nested_primary_inventory(current.relationship,
                                                         *validation.thickness,
                                                         current.child.fields,
                                                         0,
                                                         0,
                                                         false,
                                                         false,
                                                         movement_metrics);
            std::copy(before.begin(), before.end(), inventory.before.begin());
            std::copy(energy_before.values.begin(), energy_before.values.end(), inventory.before.begin() + 12);
        }
        const auto  private_fields   = current.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
        const auto& feedback_private = current.last_two_way_feedback.private_scalars;
        if (private_fields.size() != feedback_private.size())
            reject_nested_execution("private relocation requires a complete preceding feedback inventory");
        if (!inventory.complete)
            inventory.private_scalars.resize(private_fields.size());
        if (inventory.private_scalars.size() != private_fields.size())
            reject_nested_execution("private relocation inventory layout changed");
        for (std::size_t f = 0; f < private_fields.size(); ++f)
        {
            const auto& field  = private_fields[f];
            auto&       scalar = inventory.private_scalars[f];
            if (!field.name || !field.field || !field.positive_definite ||
                field.storage != tmv::MicrophysicsScalarStorage::DryAirMixingRatio ||
                feedback_private[f].name != field.name)
                reject_nested_execution("private relocation requires positive dry-specific carrier identity");
            const double before = nested_private_carrier_inventory(current.relationship,
                                                                   *validation.thickness,
                                                                   current.child.fields.rho,
                                                                   *field.field,
                                                                   0,
                                                                   0,
                                                                   false,
                                                                   movement_metrics);
            if (!inventory.complete)
            {
                scalar.name   = field.name;
                scalar.before = before;
            }
            const double synchronized = feedback_private[f].carrier.parent.child_synchronized_inventory;
            if (scalar.name != field.name ||
                std::abs(scalar.before - synchronized) >
                    128 * std::numeric_limits<double>::epsilon() * std::max(1.0, synchronized) ||
                (inventory.complete && std::abs(before - scalar.after) >
                                           128 * std::numeric_limits<double>::epsilon() * std::max(1.0, before)))
                reject_nested_execution("private relocation does not start from its synchronized inventory");
            scalar.retired += nested_private_carrier_inventory(current.relationship,
                                                               *validation.thickness,
                                                               current.child.fields.rho,
                                                               *field.field,
                                                               -child_shift.i,
                                                               -child_shift.j,
                                                               true,
                                                               movement_metrics);
            scalar.arithmetic_scale += before;
        }
        const auto energy_retired = nested_relocation_energy(
            storage, storage.nested_execution.child.fields, -child_shift.i, -child_shift.j, true);
        for (std::size_t f = 0; f < 7; ++f)
        {
            inventory.retired[f + 12] += energy_retired.values[f];
            inventory.arithmetic_scale[f + 12] += energy_before.magnitudes[f];
        }
        const auto magnitude = nested_primary_inventory(
            current.relationship, *validation.thickness, current.child.fields, 0, 0, false, true, movement_metrics);
        for (std::size_t f = 0; f < magnitude.size(); ++f)
            inventory.arithmetic_scale[f] += magnitude[f];
        inventory.complete                   = false;
        inventory.movement_sequence          = candidate_report.movement_sequence;
        inventory.feedback_application_count = current.two_way_feedback_application_count;
        const auto retired                   = nested_primary_inventory(current.relationship,
                                                                        *validation.thickness,
                                                                        current.child.fields,
                                                                        -child_shift.i,
                                                                        -child_shift.j,
                                                                        true,
                                                                        false,
                                                                        movement_metrics);
        for (std::size_t f = 0; f < retired.size(); ++f)
            inventory.retired[f] += retired[f];
        if (current.last_two_way_feedback.moist)
            for (std::size_t f = 0; f < 7; ++f)
            {
                const double expected = current.last_two_way_feedback.water[f].parent.child_synchronized_inventory;
                if (std::abs(inventory.before[f + 2] - expected) >
                    128 * std::numeric_limits<double>::epsilon() * std::max(1.0, expected))
                    reject_nested_execution("two-way relocation does not start from its synchronized water state");
            }
        validate_relocation =
            [](PrognosticFieldStorage& candidate, const NestedChildRelaxationReport& boundary, void* opaque)
        {
            auto&                        validation = *static_cast<RelocationValidation*>(opaque);
            auto&                        inventory  = *validation.report;
            const std::array<double, 12> source{boundary.theta.committed_dry_mass_change_kg,
                                                boundary.theta.committed_carrier_change,
                                                boundary.qv.committed_carrier_change,
                                                boundary.qc.committed_carrier_change,
                                                boundary.qr.committed_carrier_change,
                                                boundary.qi.committed_carrier_change,
                                                boundary.qs.committed_carrier_change,
                                                boundary.qg.committed_carrier_change,
                                                boundary.qh.committed_carrier_change,
                                                boundary.momentum.committed_x_change_kgms,
                                                boundary.momentum.committed_y_change_kgms,
                                                boundary.momentum.committed_z_change_kgms};
            for (std::size_t f = 0; f < source.size(); ++f)
                inventory.boundary_source[f] += source[f];
            if (validation.metrics)
                inventory.boundary_source[11] +=
                    nested_ground_vertical_impulse(*validation.relationship, *validation.metrics, candidate) -
                    validation.ground_before;
            const auto after = nested_primary_inventory(
                *validation.relationship, *validation.thickness, candidate, 0, 0, false, false, validation.metrics);
            std::copy(after.begin(), after.end(), inventory.after.begin());
            const auto energy_after  = nested_relocation_energy(*validation.storage, candidate);
            const auto energy_source = relocation_energy_values(boundary.energy.committed_change);
            for (std::size_t f = 0; f < 7; ++f)
            {
                inventory.after[f + 12] = energy_after.values[f];
                inventory.boundary_source[f + 12] += energy_source[f];
                inventory.arithmetic_scale[f + 12] += energy_after.magnitudes[f];
            }
            const auto ghost_after = validation.metrics
                                         ? std::array<double, 2>{}
                                         : nested_horizontal_ghost_impulse(
                                               *validation.relationship, *validation.thickness, candidate, false);
            const auto ghost_magnitude =
                validation.metrics
                    ? std::array<double, 2>{}
                    : nested_horizontal_ghost_impulse(*validation.relationship, *validation.thickness, candidate, true);
            for (std::size_t component = 0; component < 2; ++component)
            {
                inventory.boundary_source[9 + component] -= ghost_after[component] - validation.ghost_before[component];
                inventory.arithmetic_scale[9 + component] += ghost_magnitude[component];
            }
            const auto magnitude = nested_primary_inventory(
                *validation.relationship, *validation.thickness, candidate, 0, 0, false, true, validation.metrics);
            for (std::size_t f = 0; f < magnitude.size(); ++f)
                inventory.arithmetic_scale[f] += magnitude[f];
            const auto& private_source = boundary.microphysics_private;
            if (private_source.scalar_count != inventory.private_scalars.size())
                reject_nested_execution("private relocation boundary inventory is incomplete");
            for (std::size_t f = 0; f < inventory.private_scalars.size(); ++f)
            {
                auto& scalar = inventory.private_scalars[f];
                if (scalar.name != private_source.name[f].data() || !private_source.positive_definite[f] ||
                    private_source.storage_semantic[f] !=
                        std::uint8_t(tmv::MicrophysicsScalarStorage::DryAirMixingRatio))
                    reject_nested_execution("private relocation boundary identity changed");
                scalar.boundary_source += private_source.committed_carrier_change[f];
                scalar.after = nested_private_carrier_inventory(
                    *validation.relationship,
                    *validation.thickness,
                    candidate.rho,
                    validation.storage->nested_execution.boundary_scratch.microphysics_private_candidate[f],
                    0,
                    0,
                    false,
                    validation.metrics);
                scalar.arithmetic_scale += scalar.after + std::abs(private_source.committed_carrier_change[f]);
            }
            inventory.complete = true;
            if (!valid_nested_two_way_relocation_report(inventory))
            {
                std::ostringstream message;
                message << "two-way relocation inventory does not close" << std::setprecision(17);
                for (std::size_t f = 0; f < inventory.before.size(); ++f)
                {
                    const double residual = inventory.after[f] - inventory.before[f] + inventory.retired[f] -
                                            inventory.admitted[f] - inventory.boundary_source[f];
                    message << "; field=" << f << " before=" << inventory.before[f] << " after=" << inventory.after[f]
                            << " retired=" << inventory.retired[f] << " admitted=" << inventory.admitted[f]
                            << " boundary=" << inventory.boundary_source[f] << " residual=" << residual;
                }
                reject_nested_execution(message.str().c_str());
            }
            // Allocate the durable private ledger before boundary publication.
            validation.committed_report = inventory;
            validation.next_inventory =
                validation.metrics
                    ? nested_two_way_child_inventory_mapped(
                          validation.storage->nested_execution.active_terrain.geometry, candidate)
                    : nested_two_way_child_inventory_flat(*validation.relationship, *validation.thickness, candidate);
        };
    }

    PrimaryDiscardJournal discard_journal;
    bool                  primary_state_shifted       = false;
    bool                  microphysics_state_shifted  = false;
    bool                  precipitation_state_shifted = false;
    bool                  chaos_scheme_shifted        = false;
    bool                  terrain_shifted             = false;
    try
    {
        auto& nested = storage.nested_execution;
        exchange_nested_chaos_scheme_translation(nested.child, prepared_chaos);
        chaos_scheme_shifted             = prepared_chaos.active;
        nested.relationship              = next_relationship;
        nested.movement_sequence         = candidate_report.movement_sequence;
        nested.cumulative_parent_shift_i = static_cast<int>(next_cumulative_i);
        nested.cumulative_parent_shift_j = static_cast<int>(next_cumulative_j);
        place_child_frame_from_relationship(storage, nested);
        if (mapped)
        {
            using std::swap;
            swap(nested.active_terrain, terrain_candidate);
            swap(nested.child.runtime.terrain.metrics, metric_candidate);
            swap(nested.child.runtime.terrain.topography, topography_candidate);
            terrain_shifted = true;
            prepare_nested_child_process_grid_metrics(nested);
        }

        // The ordinary relaxation candidate is idle until the primary shift
        // completes, so reuse it for the simultaneous full parent target.
        // This removes one full child-state allocation from relocation peak
        // memory without weakening rollback of the live child.
        auto&                                   parent_target = nested.boundary_scratch.candidate;
        tmv::domain::NestedActiveTerrainStateV1 target_terrain;
        prepare_nested_child_physical_primary_state(
            storage, storage.fields, parent_target, candidate_report.parent_target, mapped ? &target_terrain : nullptr);
        if (mapped)
            nested.active_terrain.ledger = target_terrain.ledger;
        prepared_microphysics = prepare_nested_microphysics_movement(
            storage, next_relationship, child_shift.i, child_shift.j, parent_target.rho);
        candidate_report.microphysics = prepared_microphysics.report;
        {
            // Only owners leaving the footprint are journaled. The retained
            // C-grid owners shift in place with direction-aware traversal; a
            // failed forcing transaction reverses that shift and restores
            // these discarded strips exactly.
            discard_journal = prepare_primary_discard_journal(
                nested.child.fields, nested.relationship, child_shift.i, child_shift.j, nested.boundary_scratch);
            candidate_report.rollback_journal_values = journal_value_count(discard_journal);
            candidate_report.rollback_journal_bytes  = candidate_report.rollback_journal_values * sizeof(float);
            shift_primary_in_place_and_fill(
                nested.child.fields, parent_target, nested.relationship, child_shift.i, child_shift.j);
            primary_state_shifted = true;
        }
        if (two_way)
        {
            const auto energy_admitted =
                nested_relocation_energy(storage, nested.child.fields, child_shift.i, child_shift.j, true);
            for (std::size_t f = 0; f < 7; ++f)
            {
                candidate_report.two_way_inventory.admitted[f + 12] += energy_admitted.values[f];
                candidate_report.two_way_inventory.arithmetic_scale[f + 12] += energy_admitted.magnitudes[f];
            }
            const auto admitted  = nested_primary_inventory(next_relationship,
                                                            *validation.thickness,
                                                            nested.child.fields,
                                                            child_shift.i,
                                                            child_shift.j,
                                                            true,
                                                            false,
                                                            movement_metrics);
            const auto magnitude = nested_primary_inventory(next_relationship,
                                                            *validation.thickness,
                                                            nested.child.fields,
                                                            child_shift.i,
                                                            child_shift.j,
                                                            true,
                                                            true,
                                                            movement_metrics);
            for (std::size_t f = 0; f < admitted.size(); ++f)
            {
                candidate_report.two_way_inventory.admitted[f] += admitted[f];
                candidate_report.two_way_inventory.arithmetic_scale[f] += magnitude[f];
            }
        }
        require_valid_nested_execution_relationship(storage);
        require_relocatable_child_runtime(storage);
        apply_nested_microphysics_movement(nested.relationship, child_shift.i, child_shift.j, prepared_microphysics);
        microphysics_state_shifted = prepared_microphysics.active;
        if (two_way)
            for (std::size_t f = 0; f < candidate_report.two_way_inventory.private_scalars.size(); ++f)
            {
                auto&        scalar   = candidate_report.two_way_inventory.private_scalars[f];
                const double admitted = nested_private_carrier_inventory(next_relationship,
                                                                         *validation.thickness,
                                                                         nested.child.fields.rho,
                                                                         *prepared_microphysics.child_private_fields[f],
                                                                         child_shift.i,
                                                                         child_shift.j,
                                                                         true,
                                                                         movement_metrics);
                scalar.admitted += admitted;
                scalar.arithmetic_scale += admitted;
            }
        apply_nested_precipitation_movement(
            nested.child, nested.relationship, child_shift.i, child_shift.j, prepared_precipitation);
        precipitation_state_shifted = prepared_precipitation.active;

        if (two_way)
        {
            if (mapped)
                validation.ground_before =
                    nested_ground_vertical_impulse(next_relationship, *movement_metrics, nested.child.fields);
            validation.ghost_before = mapped
                                          ? std::array<double, 2>{}
                                          : nested_horizontal_ghost_impulse(
                                                next_relationship, *validation.thickness, nested.child.fields, false);
            const auto magnitude    = mapped ? std::array<double, 2>{}
                                             : nested_horizontal_ghost_impulse(
                                                   next_relationship, *validation.thickness, nested.child.fields, true);
            for (std::size_t component = 0; component < 2; ++component)
                candidate_report.two_way_inventory.arithmetic_scale[9 + component] += magnitude[component];
        }

        // A relocation has no elapsed model time. The zero-duration profile
        // prescribes only the exact outer ring; every interior overlap stays
        // bit-for-bit retained and every newly admitted owner keeps its
        // simultaneous conservative parent target.
        apply_nested_child_physical_relaxation_implementation(storage,
                                                              storage.fields,
                                                              relaxation,
                                                              0.0,
                                                              candidate_report.boundary_source,
                                                              nullptr,
                                                              nested.child.runtime.microphysics.scheme.get(),
                                                              prepared_microphysics.theta_storage_residual_active
                                                                  ? prepared_microphysics.child_theta_storage_residual
                                                                  : nullptr,
                                                              validate_relocation,
                                                              &validation);
        primary_state_shifted       = false;
        microphysics_state_shifted  = false;
        precipitation_state_shifted = false;
        commit_nested_radiation_movement(
            nested.child, nested.relationship, child_shift.i, child_shift.j, prepared_radiation);
        commit_nested_boundary_layer_movement(
            nested.child, nested.relationship, child_shift.i, child_shift.j, prepared_boundary_layer);
        commit_nested_chaos_movement(nested.child, nested.relationship, child_shift.i, child_shift.j, prepared_chaos);
        invalidate_nested_child_generation_state(nested.child);
        nested.child.diagnostics = {};
        if (two_way)
        {
            nested.last_two_way_feedback.relocation       = std::move(validation.committed_report);
            nested.two_way_child_interval_begin_inventory = validation.next_inventory;
            nested.two_way_interval_source_begin          = nested.interface_source.cumulative;
            invalidate_nested_feedback_registers(storage.runtime);
        }
        if (child_partition_active)
            partition_runtime.storm_selected_partition.swap(shifted_partition);
    }
    catch (...)
    {
        auto& nested = storage.nested_execution;
        if (precipitation_state_shifted)
            rollback_nested_precipitation_movement(
                nested.child, next_relationship, child_shift.i, child_shift.j, prepared_precipitation);
        if (microphysics_state_shifted)
            rollback_nested_microphysics_movement(
                next_relationship, child_shift.i, child_shift.j, prepared_microphysics);
        if (primary_state_shifted)
            rollback_primary_shift(
                nested.child.fields, discard_journal, next_relationship, child_shift.i, child_shift.j);
        if (chaos_scheme_shifted)
            exchange_nested_chaos_scheme_translation(nested.child, prepared_chaos);
        if (terrain_shifted)
        {
            using std::swap;
            swap(nested.active_terrain, terrain_candidate);
            swap(nested.child.runtime.terrain.metrics, metric_candidate);
            swap(nested.child.runtime.terrain.topography, topography_candidate);
            nested.child.process_grid_metrics = std::move(original_process_metrics);
        }
        nested.relationship                 = original_relationship;
        nested.child.domain_frame           = original_frame;
        nested.child.coriolis_f             = original_coriolis_f;
        nested.movement_sequence            = original_movement_sequence;
        nested.cumulative_parent_shift_i    = original_cumulative_i;
        nested.cumulative_parent_shift_j    = original_cumulative_j;
        nested.interface_source             = original_interface_source;
        nested.physical_primary_state_ready = original_physical_ready;
        nested.parent_boundary_state_ready  = original_boundary_ready;
        throw;
    }
    report = std::move(candidate_report);
}

NestedTrackedEnvelopeV1 prepare_nested_tracked_envelope(const tmv::diagnostics::StormFeatureDetection&   detection,
                                                        const tmv::diagnostics::StormContinuityState&    continuity,
                                                        const tmv::diagnostics::StormEnvelopeFaceMotion& face_motion,
                                                        double                                           sample_time_s,
                                                        double earth_origin_x_m,
                                                        double earth_origin_y_m,
                                                        double frame_velocity_x_ms,
                                                        double frame_velocity_y_ms)
{
    const double frame_values[] = {
        sample_time_s, earth_origin_x_m, earth_origin_y_m, frame_velocity_x_ms, frame_velocity_y_ms};
    if (!std::all_of(
            std::begin(frame_values), std::end(frame_values), [](double value) { return std::isfinite(value); }))
        reject_nested_execution("tracked storm frame contains invalid numeric evidence");
    const double face_motion_values[] = {face_motion.interval_s,
                                         face_motion.west_coordinate_velocity_ms,
                                         face_motion.east_coordinate_velocity_ms,
                                         face_motion.south_coordinate_velocity_ms,
                                         face_motion.north_coordinate_velocity_ms};
    if (!std::all_of(std::begin(face_motion_values),
                     std::end(face_motion_values),
                     [](double value) { return std::isfinite(value); }) ||
        (face_motion.available && !(face_motion.interval_s > 0.0)) ||
        (!face_motion.available &&
         (face_motion.interval_s != 0.0 || face_motion.west_coordinate_velocity_ms != 0.0 ||
          face_motion.east_coordinate_velocity_ms != 0.0 || face_motion.south_coordinate_velocity_ms != 0.0 ||
          face_motion.north_coordinate_velocity_ms != 0.0)))
    {
        reject_nested_execution("tracked storm envelope face motion is invalid");
    }
    const bool valid_status = [&]()
    {
        using Status = tmv::diagnostics::StormContinuityStatus;
        switch (continuity.status)
        {
        case Status::Uninitialized:
        case Status::Tentative:
        case Status::Tracking:
        case Status::Coasting:
        case Status::Ambiguous:
        case Status::Lost:
            return true;
        }
        return false;
    }();
    const bool valid_reason = [&]()
    {
        using Reason = tmv::diagnostics::StormContinuityReason;
        switch (continuity.reason)
        {
        case Reason::None:
        case Reason::WeakDetection:
        case Reason::SpatialAmbiguity:
        case Reason::BranchCandidateOverflow:
        case Reason::SpeedGate:
        case Reason::AccelerationGate:
            return true;
        }
        return false;
    }();
    if (!valid_status || !valid_reason)
        reject_nested_execution("tracked storm continuity enum is invalid");
    if (detection.reported_branch_candidate_count > detection.branch_candidates.size())
        reject_nested_execution("tracked storm branch count exceeds fixed capacity");
    if (continuity.selected_candidate_available &&
        continuity.selected_candidate_rank >= detection.reported_branch_candidate_count)
        reject_nested_execution("tracked storm selected branch is absent from the detector report");

    const bool selected_reported_candidate =
        continuity.selected_candidate_available &&
        continuity.selected_candidate_rank < detection.reported_branch_candidate_count;
    const auto* selected_branch =
        selected_reported_candidate ? &detection.branch_candidates[continuity.selected_candidate_rank] : nullptr;

    const double envelope_west_m =
        selected_branch != nullptr ? selected_branch->envelope_west_m : detection.envelope_west_m;
    const double envelope_east_m =
        selected_branch != nullptr ? selected_branch->envelope_east_m : detection.envelope_east_m;
    const double envelope_south_m =
        selected_branch != nullptr ? selected_branch->envelope_south_m : detection.envelope_south_m;
    const double envelope_north_m =
        selected_branch != nullptr ? selected_branch->envelope_north_m : detection.envelope_north_m;
    const double values[] = {continuity.continuity_confidence,
                             envelope_west_m,
                             envelope_east_m,
                             envelope_south_m,
                             envelope_north_m,
                             continuity.velocity_x_ms,
                             continuity.velocity_y_ms};
    if (!std::all_of(std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) ||
        continuity.continuity_confidence < 0.0 || continuity.continuity_confidence > 1.0 ||
        (selected_branch != nullptr && (envelope_east_m < envelope_west_m || envelope_north_m < envelope_south_m)))
        reject_nested_execution("tracked storm selected branch contains invalid evidence");

    NestedTrackedEnvelopeV1 result;
    result.sample_time_s              = sample_time_s;
    result.selected_feature_available = selected_branch != nullptr;
    result.continuity_authoritative   = continuity.status == tmv::diagnostics::StormContinuityStatus::Tracking;
    result.motion_available           = continuity.motion_available;
    result.ambiguous                  = continuity.status == tmv::diagnostics::StormContinuityStatus::Ambiguous ||
                                        continuity.reason == tmv::diagnostics::StormContinuityReason::SpatialAmbiguity;
    result.branch_candidate_overflow =
        detection.branch_candidate_overflow ||
        continuity.reason == tmv::diagnostics::StormContinuityReason::BranchCandidateOverflow;
    result.topology_reorganization         = continuity.topology_reorganization;
    result.confidence                      = continuity.continuity_confidence;
    result.envelope_west_earth_m           = envelope_west_m + earth_origin_x_m;
    result.envelope_east_earth_m           = envelope_east_m + earth_origin_x_m;
    result.envelope_south_earth_m          = envelope_south_m + earth_origin_y_m;
    result.envelope_north_earth_m          = envelope_north_m + earth_origin_y_m;
    result.velocity_x_earth_ms             = continuity.velocity_x_ms + frame_velocity_x_ms;
    result.velocity_y_earth_ms             = continuity.velocity_y_ms + frame_velocity_y_ms;
    result.envelope_face_motion_available  = face_motion.available;
    result.envelope_face_motion_interval_s = face_motion.interval_s;
    if (face_motion.available)
    {
        result.envelope_west_earth_velocity_ms  = face_motion.west_coordinate_velocity_ms + frame_velocity_x_ms;
        result.envelope_east_earth_velocity_ms  = face_motion.east_coordinate_velocity_ms + frame_velocity_x_ms;
        result.envelope_south_earth_velocity_ms = face_motion.south_coordinate_velocity_ms + frame_velocity_y_ms;
        result.envelope_north_earth_velocity_ms = face_motion.north_coordinate_velocity_ms + frame_velocity_y_ms;
    }
    return result;
}

NestedTrackedEnvelopeV1 prepare_nested_tracked_envelope(const tmv::diagnostics::StormFeatureDetection& detection,
                                                        const tmv::diagnostics::StormContinuityState&  continuity,
                                                        double                                         sample_time_s,
                                                        double                                         earth_origin_x_m,
                                                        double                                         earth_origin_y_m,
                                                        double frame_velocity_x_ms,
                                                        double frame_velocity_y_ms)
{
    return prepare_nested_tracked_envelope(detection,
                                           continuity,
                                           {},
                                           sample_time_s,
                                           earth_origin_x_m,
                                           earth_origin_y_m,
                                           frame_velocity_x_ms,
                                           frame_velocity_y_ms);
}

NestedTrackedEnvelopeV1
prepare_nested_tracked_envelope_earth(const tmv::diagnostics::StormFeatureDetection&   detection,
                                      const tmv::diagnostics::StormContinuityState&    continuity,
                                      const tmv::diagnostics::StormEnvelopeFaceMotion& face_motion,
                                      double                                           sample_time_s)
{
    return prepare_nested_tracked_envelope(detection, continuity, face_motion, sample_time_s, 0.0, 0.0, 0.0, 0.0);
}

NestedTrackedEnvelopeV1 prepare_nested_tracked_envelope_earth(const tmv::diagnostics::StormFeatureDetection& detection,
                                                              const tmv::diagnostics::StormContinuityState&  continuity,
                                                              double sample_time_s)
{
    return prepare_nested_tracked_envelope_earth(detection, continuity, {}, sample_time_s);
}

tmv::domain::StormWindowMovementEvidence prepare_nested_child_movement_evidence(const SimulationStorageState&  storage,
                                                                                const NestedTrackedEnvelopeV1& envelope)
{
    require_valid_nested_execution_relationship(storage);
    const double values[] = {envelope.sample_time_s,
                             envelope.confidence,
                             envelope.envelope_west_earth_m,
                             envelope.envelope_east_earth_m,
                             envelope.envelope_south_earth_m,
                             envelope.envelope_north_earth_m,
                             envelope.velocity_x_earth_ms,
                             envelope.velocity_y_earth_ms,
                             envelope.envelope_face_motion_interval_s,
                             envelope.envelope_west_earth_velocity_ms,
                             envelope.envelope_east_earth_velocity_ms,
                             envelope.envelope_south_earth_velocity_ms,
                             envelope.envelope_north_earth_velocity_ms};
    if (!std::all_of(std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) ||
        envelope.confidence < 0.0 || envelope.confidence > 1.0)
        reject_nested_execution("tracked storm envelope contains invalid numeric evidence");
    if ((envelope.envelope_face_motion_available && !(envelope.envelope_face_motion_interval_s > 0.0)) ||
        (!envelope.envelope_face_motion_available &&
         (envelope.envelope_face_motion_interval_s != 0.0 || envelope.envelope_west_earth_velocity_ms != 0.0 ||
          envelope.envelope_east_earth_velocity_ms != 0.0 || envelope.envelope_south_earth_velocity_ms != 0.0 ||
          envelope.envelope_north_earth_velocity_ms != 0.0)))
    {
        reject_nested_execution("tracked storm envelope face motion is invalid");
    }
    if (envelope.selected_feature_available && (envelope.envelope_east_earth_m < envelope.envelope_west_earth_m ||
                                                envelope.envelope_north_earth_m < envelope.envelope_south_earth_m))
        reject_nested_execution("tracked storm envelope is geometrically inverted");

    const auto& nested = storage.nested_execution;
    const auto  physical =
        physical_scalar_domain(nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz, StaggerType::CGrid);
    const auto   earth_origin        = nested.child.domain_frame.earth_origin_at(envelope.sample_time_s);
    const double child_west_earth_m  = earth_origin.x_m + physical.i_begin * nested.child.grid.dx;
    const double child_south_earth_m = earth_origin.y_m + physical.j_begin * nested.child.grid.dy;
    const double child_east_earth_m  = child_west_earth_m + physical.nx() * nested.child.grid.dx;
    const double child_north_earth_m = child_south_earth_m + physical.ny() * nested.child.grid.dy;

    tmv::domain::StormWindowMovementEvidence evidence;
    evidence.sample_time_s                  = envelope.sample_time_s;
    evidence.selected_feature_available     = envelope.selected_feature_available;
    evidence.continuity_authoritative       = envelope.continuity_authoritative;
    evidence.motion_available               = envelope.motion_available;
    evidence.ambiguous                      = envelope.ambiguous;
    evidence.branch_candidate_overflow      = envelope.branch_candidate_overflow;
    evidence.topology_reorganization        = envelope.topology_reorganization;
    evidence.confidence                     = envelope.confidence;
    evidence.distance_west_m                = envelope.envelope_west_earth_m - child_west_earth_m;
    evidence.distance_east_m                = child_east_earth_m - envelope.envelope_east_earth_m;
    evidence.distance_south_m               = envelope.envelope_south_earth_m - child_south_earth_m;
    evidence.distance_north_m               = child_north_earth_m - envelope.envelope_north_earth_m;
    evidence.velocity_x_ms                  = envelope.velocity_x_earth_ms - nested.child.domain_frame.velocity_ms.u_ms;
    evidence.velocity_y_ms                  = envelope.velocity_y_earth_ms - nested.child.domain_frame.velocity_ms.v_ms;
    evidence.envelope_face_motion_available = envelope.envelope_face_motion_available;
    evidence.envelope_face_motion_interval_s = envelope.envelope_face_motion_interval_s;
    if (envelope.envelope_face_motion_available)
    {
        evidence.envelope_west_coordinate_velocity_ms =
            envelope.envelope_west_earth_velocity_ms - nested.child.domain_frame.velocity_ms.u_ms;
        evidence.envelope_east_coordinate_velocity_ms =
            envelope.envelope_east_earth_velocity_ms - nested.child.domain_frame.velocity_ms.u_ms;
        evidence.envelope_south_coordinate_velocity_ms =
            envelope.envelope_south_earth_velocity_ms - nested.child.domain_frame.velocity_ms.v_ms;
        evidence.envelope_north_coordinate_velocity_ms =
            envelope.envelope_north_earth_velocity_ms - nested.child.domain_frame.velocity_ms.v_ms;
    }
    evidence.legal_shift_bounds_available = true;
    evidence.minimum_i_shift_cells =
        nested.relationship.boundary_stencil_parent_cells - nested.relationship.child_origin_parent_cell_i;
    evidence.maximum_i_shift_cells =
        nested.relationship.parent_cells_x - nested.relationship.boundary_stencil_parent_cells -
        nested.relationship.child_parent_span_x - nested.relationship.child_origin_parent_cell_i;
    evidence.minimum_j_shift_cells =
        nested.relationship.boundary_stencil_parent_cells - nested.relationship.child_origin_parent_cell_j;
    evidence.maximum_j_shift_cells =
        nested.relationship.parent_cells_y - nested.relationship.boundary_stencil_parent_cells -
        nested.relationship.child_parent_span_y - nested.relationship.child_origin_parent_cell_j;
    return evidence;
}

void evaluate_and_move_nested_child_from_tracker(SimulationStorageState&                       storage,
                                                 const NestedTrackedEnvelopeV1&                envelope,
                                                 const tmv::domain::StormWindowMovementPolicy& policy,
                                                 const tmv::domain::NestedRelaxationConfigV1&  relaxation,
                                                 NestedChildMovementAuthorityReportV1&         report)
{
    if (storage.nested_execution.child_execution_active)
        reject_nested_execution("cannot evaluate child movement while it is process-bound");
    require_valid_nested_execution_relationship(storage);
    const auto& nested = storage.nested_execution;
    if (!nested.relationship_ready ||
        nested.parent_time_bracket.completed_child_substeps != nested.relationship.parent_time_step_ratio ||
        nested.child.time_s != storage.time_s)
        reject_nested_execution("tracker movement requires a synchronized parent/child endpoint");
    if (envelope.sample_time_s != storage.time_s)
        reject_nested_execution("tracker sample time must equal the synchronized endpoint");
    if (policy.dx_m != nested.relationship.parent_dx_m || policy.dy_m != nested.relationship.parent_dy_m)
        reject_nested_execution("tracker movement policy must use parent-cell spacing");
    const auto& live_authority    = storage.runtime.storm_window_movement;
    const auto  child_authority_i = static_cast<std::int64_t>(live_authority.cumulative_i_shift_cells) -
                                    static_cast<std::int64_t>(nested.cumulative_parent_recycling_shift_i);
    const auto  child_authority_j = static_cast<std::int64_t>(live_authority.cumulative_j_shift_cells) -
                                    static_cast<std::int64_t>(nested.cumulative_parent_recycling_shift_j);
    if (child_authority_i != nested.cumulative_parent_shift_i || child_authority_j != nested.cumulative_parent_shift_j)
        reject_nested_execution("tracker movement authority does not match child placement provenance");

    const auto                           evidence = prepare_nested_child_movement_evidence(storage, envelope);
    auto                                 authority_candidate = live_authority;
    NestedChildMovementAuthorityReportV1 candidate_report;
    candidate_report.decision = tmv::domain::evaluate_storm_window_movement(authority_candidate, evidence, policy);

    if (candidate_report.decision.status == tmv::domain::StormWindowMovementStatus::Requested)
    {
        auto committed_authority = authority_candidate;
        tmv::domain::commit_storm_window_movement(committed_authority, candidate_report.decision);
        move_nested_child_integer_parent_cells(storage,
                                               candidate_report.decision.requested_i_shift_cells,
                                               candidate_report.decision.requested_j_shift_cells,
                                               relaxation,
                                               candidate_report.movement);
        const auto post_shift_evidence = prepare_nested_child_movement_evidence(storage, envelope);
        committed_authority.armed =
            tmv::domain::storm_window_release_margin_recovered(post_shift_evidence, policy) ||
            candidate_report.decision.reason == tmv::domain::StormWindowMovementReason::TrustedInteriorContainment;
        authority_candidate                 = std::move(committed_authority);
        candidate_report.movement_committed = true;
    }

    storage.runtime.storm_window_movement = std::move(authority_candidate);
    report                                = std::move(candidate_report);
}

void initialize_nested_shared_level_reference_state(SimulationStorageState& storage)
{
    require_valid_nested_execution_relationship(storage);
    auto candidate = storage.nested_execution;
    if (!candidate.relationship_ready)
        reject_nested_execution("relationship must be ready before reference state");
    const auto valid_profile = [&](const std::vector<double>& profile)
    {
        return profile.size() == static_cast<std::size_t>(candidate.relationship.levels_z) &&
               std::all_of(profile.begin(),
                           profile.end(),
                           [](double value)
                           { return std::isfinite(value) && std::isfinite(static_cast<float>(value)); });
    };
    if (!valid_profile(storage.base.rho0) || !valid_profile(storage.base.p0) || !valid_profile(storage.base.qv0) ||
        !valid_profile(storage.base.theta0) || !valid_profile(storage.base.u0) || !valid_profile(storage.base.v0) ||
        !std::isfinite(storage.coriolis_f))
    {
        reject_nested_execution("shared-level parent reference state is incomplete or nonfinite");
    }
    constexpr double kappa = physical_constants::gas_constant_dry_air_jkgk / physical_constants::specific_heat_cp_jkgk;
    for (int k = 0; k < candidate.relationship.levels_z; ++k)
    {
        const auto   level    = static_cast<std::size_t>(k);
        const double rho      = storage.base.rho0[level];
        const double pressure = storage.base.p0[level];
        const double theta    = storage.base.theta0[level];
        const double qv       = storage.base.qv0[level];
        if (rho <= 0.0 || pressure <= 0.0 || theta <= 0.0 || qv < 0.0 ||
            (k > 0 && pressure >= storage.base.p0[level - 1]))
        {
            reject_nested_execution("shared-level thermodynamic reference profile is not physically ordered");
        }
        const double temperature    = theta * std::pow(pressure / physical_constants::reference_pressure_pa, kappa);
        const double diagnosed_rho  = pressure / (physical_constants::gas_constant_dry_air_jkgk * temperature);
        const double relative_error = std::abs(rho - diagnosed_rho) / std::max(diagnosed_rho, 0.01);
        if (!std::isfinite(diagnosed_rho) || diagnosed_rho <= 0.0 || relative_error >= 5.0e-3)
        {
            reject_nested_execution("shared-level thermodynamic reference profile violates the dry EOS");
        }
    }
    candidate.child.base            = storage.base;
    candidate.child.coriolis_f      = expected_child_coriolis_f(storage, candidate);
    candidate.reference_state_ready = true;
    prepare_nested_child_process_grid_metrics(candidate);
    require_valid_nested_execution_relationship(storage, candidate, storage.time_s);
    storage.nested_execution = std::move(candidate);
    prepare_nested_child_process_grid_metrics(storage.nested_execution);
}

namespace
{
tmv::domain::NestedMappedConservativeStateV1
mapped_parent_conservative_state(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                 const tmv::domain::ParentTimeBracketV1&           bracket,
                                 const PrognosticFieldStorage&                     previous,
                                 const PrognosticFieldStorage&                     current,
                                 const Field3D*                                    previous_specific,
                                 const Field3D*                                    current_specific)
{
    const auto& r       = geometry.relationship;
    const auto  weights = tmv::domain::parent_temporal_weights(r, bracket);
    const auto  blend   = [&](double previous_value, double current_value)
    {
        const double value = weights.previous * previous_value + weights.current * current_value;
        if (!std::isfinite(value))
            reject_nested_execution("mapped parent temporal target is non-finite");
        return value;
    };
    tmv::domain::NestedMappedConservativeStateV1 result;
    result.scalar.resize(r.parent_cells_x, r.parent_cells_y, r.levels_z, 0.0);
    result.x_flux.resize(r.parent_cells_x + 1, r.parent_cells_y, r.levels_z, 0.0);
    result.y_flux.resize(r.parent_cells_x, r.parent_cells_y + 1, r.levels_z, 0.0);
    result.zeta_flux.resize(r.parent_cells_x, r.parent_cells_y, r.levels_z + 1, 0.0);
    const auto& metrics = geometry.parent.metrics;

    for (int i = 0; i < r.parent_cells_x; ++i)
        for (int j = 0; j < r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double previous_rho = previous.rho[i + 1][j + 1][k];
                const double current_rho  = current.rho[i + 1][j + 1][k];
                const double value        = previous_specific != nullptr
                                                ? blend(previous_rho * (*previous_specific)[i + 1][j + 1][k],
                                                        current_rho * (*current_specific)[i + 1][j + 1][k])
                                                : blend(previous_rho, current_rho);
                if (!std::isfinite(value) || (previous_specific == nullptr && value <= 0.0))
                    reject_nested_execution("mapped parent scalar target is invalid");
                result.scalar(i, j, k) = value;
            }

    for (int face = 0; face <= r.parent_cells_x; ++face)
        for (int j = 0; j < r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                long double momentum = 0.0L;
                for (int side : {face - 1, face})
                    if (side >= 0 && side < r.parent_cells_x)
                    {
                        const double endpoint_previous = previous.rho[side + 1][j + 1][k] * previous.u[face][j + 1][k];
                        const double endpoint_current  = current.rho[side + 1][j + 1][k] * current.u[face][j + 1][k];
                        momentum += 0.5L * metrics.J(side, j, k) * blend(endpoint_previous, endpoint_current);
                    }
                result.x_flux(face, j, k) = static_cast<double>(momentum / metrics.Jx_face(face, j, k));
            }
    for (int i = 0; i < r.parent_cells_x; ++i)
        for (int face = 0; face <= r.parent_cells_y; ++face)
            for (int k = 0; k < r.levels_z; ++k)
            {
                long double momentum = 0.0L;
                for (int side : {face - 1, face})
                    if (side >= 0 && side < r.parent_cells_y)
                    {
                        const double endpoint_previous = previous.rho[i + 1][side + 1][k] * previous.v[i + 1][face][k];
                        const double endpoint_current  = current.rho[i + 1][side + 1][k] * current.v[i + 1][face][k];
                        momentum += 0.5L * metrics.J(i, side, k) * blend(endpoint_previous, endpoint_current);
                    }
                result.y_flux(i, face, k) = static_cast<double>(momentum / metrics.Jy_face(i, face, k));
            }
    for (int i = 0; i < r.parent_cells_x; ++i)
        for (int j = 0; j < r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                long double momentum_per_area = 0.5L * metrics.J(i, j, k) * metrics.dzeta *
                                                blend(previous.rho[i + 1][j + 1][k] * previous.w[i + 1][j + 1][k],
                                                      current.rho[i + 1][j + 1][k] * current.w[i + 1][j + 1][k]);
                if (k + 1 < r.levels_z)
                    momentum_per_area += 0.5L * metrics.J(i, j, k + 1) * metrics.dzeta *
                                         blend(previous.rho[i + 1][j + 1][k + 1] * previous.w[i + 1][j + 1][k],
                                               current.rho[i + 1][j + 1][k + 1] * current.w[i + 1][j + 1][k]);
                result.zeta_flux(i, j, k + 1) = static_cast<double>(momentum_per_area);
            }
    return result;
}

double mapped_child_x_effective_density(
    const tmv::domain::NestedMappedTerrainGeometryV1& geometry, const Field3D& rho, int face, int j, int k)
{
    const auto& r       = geometry.relationship;
    const auto& metrics = geometry.child_storage.metrics;
    long double mass    = 0.0L;
    for (int side : {face - 1, face})
        if (side >= 0 && side < r.child_cells_x)
            mass += 0.5L * rho[side + 1][j + 1][k] * metrics.J(side + 1, j + 1, k);
    const double fraction = face == 0 || face == r.child_cells_x ? .5 : 1.0;
    return static_cast<double>(mass / (fraction * metrics.Jx_face(face + 1, j + 1, k)));
}

double mapped_child_y_effective_density(
    const tmv::domain::NestedMappedTerrainGeometryV1& geometry, const Field3D& rho, int i, int face, int k)
{
    const auto& r       = geometry.relationship;
    const auto& metrics = geometry.child_storage.metrics;
    long double mass    = 0.0L;
    for (int side : {face - 1, face})
        if (side >= 0 && side < r.child_cells_y)
            mass += 0.5L * rho[i + 1][side + 1][k] * metrics.J(i + 1, side + 1, k);
    const double fraction = face == 0 || face == r.child_cells_y ? .5 : 1.0;
    return static_cast<double>(mass / (fraction * metrics.Jy_face(i + 1, face + 1, k)));
}

double mapped_child_z_dual_mass_per_area(
    const tmv::domain::NestedMappedTerrainGeometryV1& geometry, const Field3D& rho, int i, int j, int k)
{
    const auto& r       = geometry.relationship;
    const auto& metrics = geometry.child_storage.metrics;
    long double mass    = 0.5L * rho[i + 1][j + 1][k] * metrics.J(i + 1, j + 1, k) * metrics.dzeta;
    if (k + 1 < r.levels_z)
        mass += 0.5L * rho[i + 1][j + 1][k + 1] * metrics.J(i + 1, j + 1, k + 1) * metrics.dzeta;
    return static_cast<double>(mass);
}

tmv::domain::NestedMappedConservativeStateV1
mapped_conservative_state_from_child_fields(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                            const PrognosticFieldStorage&                     fields)
{
    const auto&                                  r = geometry.relationship;
    tmv::domain::NestedMappedConservativeStateV1 result;
    result.scalar.resize(r.child_cells_x, r.child_cells_y, r.levels_z, 0.0);
    result.x_flux.resize(r.child_cells_x + 1, r.child_cells_y, r.levels_z, 0.0);
    result.y_flux.resize(r.child_cells_x, r.child_cells_y + 1, r.levels_z, 0.0);
    result.zeta_flux.resize(r.child_cells_x, r.child_cells_y, r.levels_z + 1, 0.0);
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                result.scalar(i, j, k) = fields.rho[i + 1][j + 1][k];
                result.zeta_flux(i, j, k + 1) =
                    mapped_child_z_dual_mass_per_area(geometry, fields.rho, i, j, k) * fields.w[i + 1][j + 1][k];
            }
    for (int face = 0; face <= r.child_cells_x; ++face)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
                result.x_flux(face, j, k) =
                    mapped_child_x_effective_density(geometry, fields.rho, face, j, k) * fields.u[face][j + 1][k];
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int face = 0; face <= r.child_cells_y; ++face)
            for (int k = 0; k < r.levels_z; ++k)
                result.y_flux(i, face, k) =
                    mapped_child_y_effective_density(geometry, fields.rho, i, face, k) * fields.v[i + 1][face][k];
    return result;
}

void prolong_mapped_specific(const tmv::domain::NestedMappedTerrainGeometryV1&   geometry,
                             const tmv::domain::ParentTimeBracketV1&             bracket,
                             const PrognosticFieldStorage&                       previous,
                             const PrognosticFieldStorage&                       current,
                             const tmv::domain::NestedMappedConservativeStateV1& child_dry,
                             const tmv::domain::NestedTerrainTransferLedgerV1&   dry_ledger,
                             const Field3D&                                      stored_density,
                             const Field3D&                                      previous_specific,
                             const Field3D&                                      current_specific,
                             bool                                                nonnegative,
                             Field3D&                                            destination,
                             tmv::domain::ConservedScalarProlongationReport&     report)
{
    const auto& r = geometry.relationship;
    const auto  parent_carrier =
        mapped_parent_conservative_state(geometry, bracket, previous, current, &previous_specific, &current_specific);
    tmv::domain::NestedTerrainTransferLedgerV1 carrier_ledger;
    const auto                                 child_carrier =
        tmv::domain::prolong_nested_mapped_conservative_state(geometry, parent_carrier, &carrier_ledger);
    long double committed     = 0.0L;
    long double committed_dry = 0.0L;
    const auto& metrics       = geometry.child_storage.metrics;
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double density = stored_density[i + 1][j + 1][k];
                if (density != static_cast<float>(child_dry.scalar(i, j, k)))
                    reject_nested_execution("mapped carrier does not share the primary dry-mass authority");
                const float stored = static_cast<float>(child_carrier.scalar(i, j, k) / child_dry.scalar(i, j, k));
                if (!std::isfinite(stored) || (nonnegative && stored < 0.0f))
                    reject_nested_execution("mapped child carrier is not representable");
                destination[i + 1][j + 1][k] = stored;
                const long double mass = static_cast<long double>(density) * metrics.J(i + 1, j + 1, k) * metrics.dx *
                                         metrics.dy * metrics.dzeta;
                committed_dry += mass;
                committed += static_cast<long double>(stored) * density * metrics.J(i + 1, j + 1, k) * metrics.dx *
                             metrics.dy * metrics.dzeta;
            }
    report.target_dry_mass_kg          = dry_ledger.parent_scalar_inventory;
    report.committed_dry_mass_kg       = static_cast<double>(committed_dry);
    report.dry_storage_residual_kg     = report.target_dry_mass_kg - report.committed_dry_mass_kg;
    report.target_carrier_inventory    = carrier_ledger.parent_scalar_inventory;
    report.committed_carrier_inventory = static_cast<double>(committed);
    report.carrier_storage_residual    = report.target_carrier_inventory - report.committed_carrier_inventory;
    report.parent_cells = static_cast<std::size_t>(r.child_parent_span_x) * r.child_parent_span_y * r.levels_z;
    report.child_cells  = static_cast<std::size_t>(r.child_cells_x) * r.child_cells_y * r.levels_z;
}

void prepare_mapped_private_movement_values(SimulationStorageState&             storage,
                                            const Field3D&                      admitted_density,
                                            int                                 child_shift_i,
                                            int                                 child_shift_j,
                                            PreparedNestedMicrophysicsMovement& prepared)
{
    const auto& nested     = storage.nested_execution;
    const auto& geometry   = nested.active_terrain.geometry;
    const auto& r          = nested.relationship;
    const auto& metrics    = geometry.child_storage.metrics;
    const auto  parent     = storage.runtime.microphysics.scheme->prognostic_scalars_for_transport();
    const auto  parent_dry = mapped_parent_conservative_state(
        geometry, nested.parent_time_bracket, storage.fields, storage.fields, nullptr, nullptr);
    const auto child_dry = tmv::domain::prolong_nested_mapped_conservative_state(geometry, parent_dry);
    for (std::size_t f = 0; f < parent.size(); ++f)
    {
        if (parent[f].storage != tmv::MicrophysicsScalarStorage::DryAirMixingRatio)
            reject_nested_execution("mapped private movement requires dry-specific carriers");
        const auto parent_carrier = mapped_parent_conservative_state(
            geometry, nested.parent_time_bracket, storage.fields, storage.fields, parent[f].field, parent[f].field);
        const auto child_carrier = tmv::domain::prolong_nested_mapped_conservative_state(geometry, parent_carrier);
        auto&      values        = *prepared.newly_admitted_private_values[f];
        tmv::diagnostics::CompensatedSum target, committed;
        for (int i = 0; i < r.child_cells_x; ++i)
            for (int j = 0; j < r.child_cells_y; ++j)
            {
                const auto old_i = std::int64_t(i) + child_shift_i, old_j = std::int64_t(j) + child_shift_j;
                if (old_i >= 0 && old_i < r.child_cells_x && old_j >= 0 && old_j < r.child_cells_y)
                    continue;
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double rho = admitted_density(i + 1, j + 1, k);
                    if (rho != float(child_dry.scalar(i, j, k)))
                        reject_nested_execution("mapped private admission differs from primary dry mass");
                    const float q = float(child_carrier.scalar(i, j, k) / child_dry.scalar(i, j, k));
                    if (!std::isfinite(q) || (parent[f].positive_definite && q < 0))
                        reject_nested_execution("mapped private admission is not representable");
                    values.push_back(q);
                    const double volume = metrics.J(i + 1, j + 1, k) * metrics.dx * metrics.dy * metrics.dzeta;
                    target.add(child_carrier.scalar(i, j, k) * volume);
                    committed.add(double(q) * rho * volume);
                }
            }
        if (!std::isfinite(target.value()) || !std::isfinite(committed.value()))
            reject_nested_execution("mapped private admission inventory overflowed");
        prepared.report.newly_admitted_target_inventory[f]    = target.value();
        prepared.report.newly_admitted_committed_inventory[f] = committed.value();
        prepared.report.newly_admitted_storage_rounding[f]    = target.value() - committed.value();
    }
}

void fill_mapped_private_closure(const tmv::domain::OneWayNestRelationshipV1& r, Field3D& field)
{
    for (int i = 0; i < r.child_cells_x + 2; ++i)
        for (int j = 0; j < r.child_cells_y + 2; ++j)
            if (i == 0 || j == 0 || i == r.child_cells_x + 1 || j == r.child_cells_y + 1)
                for (int k = 0; k < r.levels_z; ++k)
                    field[i][j][k] = field[std::clamp(i, 1, r.child_cells_x)][std::clamp(j, 1, r.child_cells_y)][k];
}

void prepare_nested_child_mapped_primary_state(SimulationStorageState&                  storage,
                                               const PrognosticFieldStorage&            previous_parent_fields,
                                               PrognosticFieldStorage&                  candidate_fields,
                                               NestedChildInitializationReport&         report,
                                               tmv::domain::NestedActiveTerrainStateV1& active_candidate)
{
    auto&       nested   = storage.nested_execution;
    const auto& geometry = nested.active_terrain.geometry;
    const auto& r        = geometry.relationship;
    const float unowned  = std::numeric_limits<float>::quiet_NaN();
    for (Field3D* field : {&candidate_fields.rho,
                           &candidate_fields.p,
                           &candidate_fields.u,
                           &candidate_fields.v,
                           &candidate_fields.w,
                           &candidate_fields.theta,
                           &candidate_fields.qv,
                           &candidate_fields.qc,
                           &candidate_fields.qr,
                           &candidate_fields.qi,
                           &candidate_fields.qs,
                           &candidate_fields.qg,
                           &candidate_fields.qh,
                           &candidate_fields.tracer,
                           &candidate_fields.tke})
        field->resize(nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz, unowned);

    const auto parent_dry = mapped_parent_conservative_state(
        geometry, nested.parent_time_bracket, previous_parent_fields, storage.fields, nullptr, nullptr);
    tmv::domain::NestedTerrainTransferLedgerV1 dry_ledger;
    const auto child_dry = tmv::domain::prolong_nested_mapped_conservative_state(geometry, parent_dry, &dry_ledger);
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const float rho = static_cast<float>(child_dry.scalar(i, j, k));
                if (!std::isfinite(rho) || rho <= 0.0f)
                    reject_nested_execution("mapped child dry density is not representable");
                candidate_fields.rho[i + 1][j + 1][k] = rho;
            }

    const auto prolong_specific = [&](const Field3D&                                  previous_specific,
                                      const Field3D&                                  current_specific,
                                      bool                                            nonnegative,
                                      Field3D&                                        destination,
                                      tmv::domain::ConservedScalarProlongationReport& scalar_report)
    {
        prolong_mapped_specific(geometry,
                                nested.parent_time_bracket,
                                previous_parent_fields,
                                storage.fields,
                                child_dry,
                                dry_ledger,
                                candidate_fields.rho,
                                previous_specific,
                                current_specific,
                                nonnegative,
                                destination,
                                scalar_report);
    };

    NestedChildInitializationReport candidate_report;
    prolong_specific(
        previous_parent_fields.theta, storage.fields.theta, false, candidate_fields.theta, candidate_report.theta);
    prolong_specific(previous_parent_fields.qv, storage.fields.qv, true, candidate_fields.qv, candidate_report.qv);
    prolong_specific(previous_parent_fields.qc, storage.fields.qc, true, candidate_fields.qc, candidate_report.qc);
    prolong_specific(previous_parent_fields.qr, storage.fields.qr, true, candidate_fields.qr, candidate_report.qr);
    prolong_specific(previous_parent_fields.qi, storage.fields.qi, true, candidate_fields.qi, candidate_report.qi);
    prolong_specific(previous_parent_fields.qs, storage.fields.qs, true, candidate_fields.qs, candidate_report.qs);
    prolong_specific(previous_parent_fields.qg, storage.fields.qg, true, candidate_fields.qg, candidate_report.qg);
    prolong_specific(previous_parent_fields.qh, storage.fields.qh, true, candidate_fields.qh, candidate_report.qh);
    prolong_specific(
        previous_parent_fields.tracer, storage.fields.tracer, true, candidate_fields.tracer, candidate_report.tracer);
    prolong_specific(previous_parent_fields.tke, storage.fields.tke, true, candidate_fields.tke, candidate_report.tke);

    for (int face = 0; face <= r.child_cells_x; ++face)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double density  = mapped_child_x_effective_density(geometry, candidate_fields.rho, face, j, k);
                const float  velocity = static_cast<float>(child_dry.x_flux(face, j, k) / density);
                if (!std::isfinite(velocity))
                    reject_nested_execution("mapped child x momentum is not representable");
                candidate_fields.u[face][j + 1][k] = velocity;
            }
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int face = 0; face <= r.child_cells_y; ++face)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double density  = mapped_child_y_effective_density(geometry, candidate_fields.rho, i, face, k);
                const float  velocity = static_cast<float>(child_dry.y_flux(i, face, k) / density);
                if (!std::isfinite(velocity))
                    reject_nested_execution("mapped child y momentum is not representable");
                candidate_fields.v[i + 1][face][k] = velocity;
            }
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double mass     = mapped_child_z_dual_mass_per_area(geometry, candidate_fields.rho, i, j, k);
                const float  velocity = static_cast<float>(child_dry.zeta_flux(i, j, k + 1) / mass);
                if (!std::isfinite(velocity))
                    reject_nested_execution("mapped child vertical momentum is not representable");
                candidate_fields.w[i + 1][j + 1][k] = velocity;
                const double pressure = tmv::dynamics_eos::pressure_fp64(candidate_fields.rho[i + 1][j + 1][k],
                                                                         candidate_fields.theta[i + 1][j + 1][k]);
                const float  stored_pressure = static_cast<float>(pressure);
                if (!std::isfinite(stored_pressure) || stored_pressure <= 0.0f)
                    reject_nested_execution("mapped child pressure is not representable");
                candidate_fields.p[i + 1][j + 1][k] = stored_pressure;
            }

    auto        committed_child         = mapped_conservative_state_from_child_fields(geometry, candidate_fields);
    long double committed_dry_inventory = 0.0L;
    const auto& child_metrics           = geometry.child_storage.metrics;
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                committed_dry_inventory += static_cast<long double>(committed_child.scalar(i, j, k)) *
                                           child_metrics.J(i + 1, j + 1, k) * child_metrics.dx * child_metrics.dy *
                                           child_metrics.dzeta;
            }
    dry_ledger.child_scalar_inventory    = static_cast<double>(committed_dry_inventory);
    dry_ledger.scalar_inventory_residual = dry_ledger.child_scalar_inventory - dry_ledger.parent_scalar_inventory;
    dry_ledger.normalized_scalar_inventory_residual =
        std::abs(dry_ledger.scalar_inventory_residual) /
        std::max({1.0, std::abs(dry_ledger.parent_scalar_inventory), std::abs(dry_ledger.child_scalar_inventory)});

    active_candidate                      = nested.active_terrain;
    active_candidate.child_state          = std::move(committed_child);
    active_candidate.ledger               = dry_ledger;
    active_candidate.ledger.parent_time_s = nested.child.time_s;
    active_candidate.ledger.child_time_s  = nested.child.time_s;
    active_candidate.parent_time_s        = nested.child.time_s;
    active_candidate.child_time_s         = nested.child.time_s;
    active_candidate.generation           = active_candidate.geometry.generation;
    active_candidate.publication_count    = 1;
    active_candidate.ready                = true;
    const double ground_u                 = candidate_fields.u[0][1][0];
    const double ground_v                 = candidate_fields.v[1][0][0];
    active_candidate.boundary =
        tmv::domain::build_nested_terrain_tangent_boundary(geometry, ground_u, ground_v, &active_candidate.ledger);
    report = candidate_report;
}

void prepare_nested_child_physical_primary_state(SimulationStorageState&                  storage,
                                                 const PrognosticFieldStorage&            previous_parent_fields,
                                                 PrognosticFieldStorage&                  candidate_fields,
                                                 NestedChildInitializationReport&         report,
                                                 tmv::domain::NestedActiveTerrainStateV1* active_candidate)
{
    require_valid_nested_execution_relationship(storage);
    auto& nested = storage.nested_execution;
    if (!nested.relationship_ready || !nested.reference_state_ready)
    {
        reject_nested_execution("relationship and reference state must precede child primary initialization");
    }
    if (nested_active_terrain_geometry_ready(nested))
    {
        if (active_candidate == nullptr)
            reject_nested_execution("mapped child primary preparation requires a detached terrain owner");
        prepare_nested_child_mapped_primary_state(
            storage, previous_parent_fields, candidate_fields, report, *active_candidate);
        return;
    }
    const auto& relationship    = nested.relationship;
    const int   nr              = nested.child.grid.nr;
    const int   nth             = nested.child.grid.nth;
    const int   nz              = nested.child.grid.nz;
    const auto& layer_thickness = nested.child.process_grid_metrics.dz;

    const float unowned          = std::numeric_limits<float>::quiet_NaN();
    Field3D*    candidate_list[] = {&candidate_fields.rho,
                                    &candidate_fields.p,
                                    &candidate_fields.u,
                                    &candidate_fields.v,
                                    &candidate_fields.w,
                                    &candidate_fields.theta,
                                    &candidate_fields.qv,
                                    &candidate_fields.qc,
                                    &candidate_fields.qr,
                                    &candidate_fields.qi,
                                    &candidate_fields.qs,
                                    &candidate_fields.qg,
                                    &candidate_fields.qh,
                                    &candidate_fields.tracer,
                                    &candidate_fields.tke};
    for (Field3D* field : candidate_list)
        field->resize(nr, nth, nz, unowned);

    NestedChildInitializationReport candidate_report;
    const auto                      prolong = [&](const Field3D&                                  previous_specific,
                                                  const Field3D&                                  current_specific,
                                                  bool                                            nonnegative,
                                                  Field3D&                                        child_specific,
                                                  tmv::domain::ConservedScalarProlongationReport& scalar_report)
    {
        tmv::domain::prolong_conserved_scalar_flat_storage(relationship,
                                                           nested.parent_time_bracket,
                                                           layer_thickness,
                                                           previous_parent_fields.rho,
                                                           storage.fields.rho,
                                                           previous_specific,
                                                           current_specific,
                                                           nonnegative,
                                                           candidate_fields.rho,
                                                           child_specific,
                                                           scalar_report);
    };
    prolong(previous_parent_fields.theta, storage.fields.theta, false, candidate_fields.theta, candidate_report.theta);
    prolong(previous_parent_fields.qv, storage.fields.qv, true, candidate_fields.qv, candidate_report.qv);
    prolong(previous_parent_fields.qc, storage.fields.qc, true, candidate_fields.qc, candidate_report.qc);
    prolong(previous_parent_fields.qr, storage.fields.qr, true, candidate_fields.qr, candidate_report.qr);
    prolong(previous_parent_fields.qi, storage.fields.qi, true, candidate_fields.qi, candidate_report.qi);
    prolong(previous_parent_fields.qs, storage.fields.qs, true, candidate_fields.qs, candidate_report.qs);
    prolong(previous_parent_fields.qg, storage.fields.qg, true, candidate_fields.qg, candidate_report.qg);
    prolong(previous_parent_fields.qh, storage.fields.qh, true, candidate_fields.qh, candidate_report.qh);
    prolong(
        previous_parent_fields.tracer, storage.fields.tracer, true, candidate_fields.tracer, candidate_report.tracer);
    prolong(previous_parent_fields.tke, storage.fields.tke, true, candidate_fields.tke, candidate_report.tke);

    const auto same_dry_report = [&](const auto& value)
    {
        return value.target_dry_mass_kg == candidate_report.theta.target_dry_mass_kg &&
               value.committed_dry_mass_kg == candidate_report.theta.committed_dry_mass_kg &&
               value.dry_storage_residual_kg == candidate_report.theta.dry_storage_residual_kg;
    };
    const tmv::domain::ConservedScalarProlongationReport* reports[] = {&candidate_report.qv,
                                                                       &candidate_report.qc,
                                                                       &candidate_report.qr,
                                                                       &candidate_report.qi,
                                                                       &candidate_report.qs,
                                                                       &candidate_report.qg,
                                                                       &candidate_report.qh,
                                                                       &candidate_report.tracer,
                                                                       &candidate_report.tke};
    for (const auto* scalar_report : reports)
        if (!same_dry_report(*scalar_report))
            reject_nested_execution("child carriers do not share one dry-mass authority");

    tmv::domain::prolong_cgrid_momentum_flat_storage(relationship,
                                                     nested.parent_time_bracket,
                                                     layer_thickness,
                                                     previous_parent_fields.rho,
                                                     storage.fields.rho,
                                                     previous_parent_fields.u,
                                                     storage.fields.u,
                                                     previous_parent_fields.v,
                                                     storage.fields.v,
                                                     previous_parent_fields.w,
                                                     storage.fields.w,
                                                     candidate_fields.rho,
                                                     candidate_fields.u,
                                                     candidate_fields.v,
                                                     candidate_fields.w,
                                                     candidate_report.momentum);
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const int   i              = ci + 1;
                const int   j              = cj + 1;
                const float theta          = candidate_fields.theta[i][j][k];
                const float qv             = candidate_fields.qv[i][j][k];
                const float hydrometeors[] = {candidate_fields.qc[i][j][k],
                                              candidate_fields.qr[i][j][k],
                                              candidate_fields.qi[i][j][k],
                                              candidate_fields.qs[i][j][k],
                                              candidate_fields.qg[i][j][k],
                                              candidate_fields.qh[i][j][k]};
                if (theta < theta_min_k || theta > theta_max_k || qv < 0.0f || qv > qv_max_kgkg ||
                    !std::all_of(std::begin(hydrometeors),
                                 std::end(hydrometeors),
                                 [](float value)
                                 { return std::isfinite(value) && value >= 0.0f && value <= hydrometeor_max_kgkg; }) ||
                    !std::isfinite(candidate_fields.tracer[i][j][k]) || candidate_fields.tracer[i][j][k] < 0.0f ||
                    candidate_fields.tracer[i][j][k] > passive_tracer_max ||
                    !std::isfinite(candidate_fields.tke[i][j][k]))
                {
                    reject_nested_execution("child scalar target exceeds strict physical bounds");
                }
            }
    for (int face = 0; face <= relationship.child_cells_x; ++face)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
                if (std::abs(candidate_fields.u[face][cj + 1][k]) > wind_horizontal_abs_max_ms)
                    reject_nested_execution("child x-wind target exceeds strict physical bounds");
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int face = 0; face <= relationship.child_cells_y; ++face)
            for (int k = 0; k < relationship.levels_z; ++k)
                if (std::abs(candidate_fields.v[ci + 1][face][k]) > wind_horizontal_abs_max_ms)
                    reject_nested_execution("child y-wind target exceeds strict physical bounds");
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
                if (std::abs(candidate_fields.w[ci + 1][cj + 1][k]) > wind_vertical_abs_max_ms)
                    reject_nested_execution("child vertical-wind target exceeds strict physical bounds");
    tmv::domain::diagnose_child_pressure_flat_storage(relationship,
                                                      layer_thickness,
                                                      candidate_fields.rho,
                                                      candidate_fields.theta,
                                                      candidate_fields.p,
                                                      candidate_report.pressure);

    report = candidate_report;
}
} // namespace

void initialize_nested_child_physical_primary_state(SimulationStorageState&          storage,
                                                    const PrognosticFieldStorage&    previous_parent_fields,
                                                    NestedChildInitializationReport& report)
{
    PrognosticFieldStorage candidate_fields;
    auto                   active_candidate = storage.nested_execution.active_terrain;
    prepare_nested_child_physical_primary_state(
        storage, previous_parent_fields, candidate_fields, report, &active_candidate);
    storage.nested_execution.child.fields                 = std::move(candidate_fields);
    storage.nested_execution.active_terrain               = std::move(active_candidate);
    storage.nested_execution.physical_primary_state_ready = true;
}

void initialize_nested_child_microphysics_private_state(
    SimulationStorageState&                   storage,
    const PrognosticFieldStorage&             previous_parent_fields,
    const NestedParentMicrophysicsSnapshotV1& previous_parent_microphysics,
    NestedChildInitializationReport&          report)
{
    require_valid_nested_execution_relationship(storage);
    auto& nested = storage.nested_execution;
    if (!nested.parent_boundary_state_ready)
        reject_nested_execution("private microphysics initialization requires complete primary boundary state");

    auto bindings = require_private_microphysics_bindings(storage, previous_parent_microphysics);
    NestedChildInitializationReport candidate_report   = report;
    candidate_report.microphysics_private_scalar_count = bindings.child.size();
    auto& scratch_rho                                  = nested.boundary_scratch.target.rho;
    scratch_rho.resize(
        nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz, std::numeric_limits<float>::quiet_NaN());

    auto candidate_ledger         = nested.interface_source;
    auto initialize_ledger_layout = [&](NestedChildRelaxationReport& value)
    {
        auto& private_report        = value.microphysics_private;
        private_report.scalar_count = bindings.child.size();
        for (std::size_t index = 0; index < bindings.child.size(); ++index)
        {
            private_report.name[index]              = previous_parent_microphysics.name[index];
            private_report.storage_semantic[index]  = previous_parent_microphysics.storage_semantic[index];
            private_report.positive_definite[index] = previous_parent_microphysics.positive_definite[index];
        }
    };
    initialize_ledger_layout(candidate_ledger.last);
    initialize_ledger_layout(candidate_ledger.cumulative);

    const bool                                   mapped = nested_active_terrain_geometry_ready(nested);
    tmv::domain::NestedMappedConservativeStateV1 mapped_dry;
    tmv::domain::NestedTerrainTransferLedgerV1   mapped_dry_ledger;
    if (mapped && !bindings.child.empty())
    {
        const auto parent_dry = mapped_parent_conservative_state(nested.active_terrain.geometry,
                                                                 nested.parent_time_bracket,
                                                                 previous_parent_fields,
                                                                 storage.fields,
                                                                 nullptr,
                                                                 nullptr);
        mapped_dry            = tmv::domain::prolong_nested_mapped_conservative_state(
            nested.active_terrain.geometry, parent_dry, &mapped_dry_ledger);
    }

    tmv::domain::ConservedScalarProlongationReport private_dry_authority;
    bool                                           private_dry_authority_available = false;
    for (std::size_t index = 0; index < bindings.child.size(); ++index)
    {
        auto& candidate_private = nested.boundary_scratch.microphysics_private_candidate[index];
        candidate_private       = *bindings.child[index].field;
        tmv::domain::ConservedScalarProlongationReport scalar_report;
        if (mapped)
        {
            prolong_mapped_specific(nested.active_terrain.geometry,
                                    nested.parent_time_bracket,
                                    previous_parent_fields,
                                    storage.fields,
                                    mapped_dry,
                                    mapped_dry_ledger,
                                    nested.child.fields.rho,
                                    previous_parent_microphysics.fields[index],
                                    *bindings.current_parent[index].field,
                                    bindings.child[index].positive_definite,
                                    candidate_private,
                                    scalar_report);
            fill_mapped_private_closure(nested.relationship, candidate_private);
        }
        else
        {
            tmv::domain::prolong_conserved_scalar_flat_storage(nested.relationship,
                                                               nested.parent_time_bracket,
                                                               nested.child.process_grid_metrics.dz,
                                                               previous_parent_fields.rho,
                                                               storage.fields.rho,
                                                               previous_parent_microphysics.fields[index],
                                                               *bindings.current_parent[index].field,
                                                               bindings.child[index].positive_definite,
                                                               scratch_rho,
                                                               candidate_private,
                                                               scalar_report);
            fill_private_mixing_ratio_closure(nested.relationship,
                                              nested.parent_time_bracket,
                                              previous_parent_fields.rho,
                                              storage.fields.rho,
                                              previous_parent_microphysics.fields[index],
                                              *bindings.current_parent[index].field,
                                              bindings.child[index].positive_definite,
                                              candidate_private);
            for (int ci = 0; ci < nested.relationship.child_cells_x; ++ci)
                for (int cj = 0; cj < nested.relationship.child_cells_y; ++cj)
                    for (int k = 0; k < nested.relationship.levels_z; ++k)
                        if (scratch_rho[ci + 1][cj + 1][k] != nested.child.fields.rho[ci + 1][cj + 1][k])
                            reject_nested_execution(
                                "private microphysics initialization density differs from the committed child state");
        }
        if (!private_dry_authority_available)
        {
            private_dry_authority           = scalar_report;
            private_dry_authority_available = true;
        }
        else if (scalar_report.target_dry_mass_kg != private_dry_authority.target_dry_mass_kg ||
                 scalar_report.committed_dry_mass_kg != private_dry_authority.committed_dry_mass_kg ||
                 scalar_report.dry_storage_residual_kg != private_dry_authority.dry_storage_residual_kg)
            reject_nested_execution(
                "private microphysics initialization does not share the primary dry-mass authority");
        candidate_report.microphysics_private[index] = scalar_report;
    }

    using std::swap;
    for (std::size_t index = 0; index < bindings.child.size(); ++index)
        swap(*bindings.child[index].field, nested.boundary_scratch.microphysics_private_candidate[index]);
    nested.interface_source = std::move(candidate_ledger);
    report                  = candidate_report;
}

namespace
{
bool prepare_nested_theta_residual(NestedExecutionStorage&                      nested,
                                   const tmv::domain::NestedRelaxationConfigV1& config,
                                   double                                       child_step_s,
                                   const Field3D&                               old_rho_field,
                                   const Field3D&                               new_rho_field,
                                   const Field3D&                               live_theta_residual)
{
    const auto& relationship = nested.relationship;
    const int   nr           = nested.child.grid.nr;
    const int   nth          = nested.child.grid.nth;
    const int   nz           = nested.child.grid.nz;
    if (!live_theta_residual.empty())
    {
        if (live_theta_residual.size_r() != nr || live_theta_residual.size_th() != nth ||
            live_theta_residual.size_z() != nz)
            reject_nested_execution("child theta compensation residual has the wrong shape");
        auto& residual_candidate = nested.boundary_scratch.theta_storage_residual_candidate;
        residual_candidate.resize(nr, nth, nz, 0.0f);
        for (int ci = 0; ci < relationship.child_cells_x; ++ci)
            for (int cj = 0; cj < relationship.child_cells_y; ++cj)
                for (int k = 0; k < relationship.levels_z; ++k)
                {
                    const int    i = ci + 1;
                    const int    j = cj + 1;
                    const double fraction =
                        tmv::domain::nested_scalar_relaxation_fraction(relationship, config, ci, cj, child_step_s);
                    const double old_rho      = old_rho_field[i][j][k];
                    const double new_rho      = new_rho_field[i][j][k];
                    const double old_residual = live_theta_residual[i][j][k];
                    const double next         = (1.0 - fraction) * old_rho / new_rho * old_residual;
                    const float  stored       = static_cast<float>(next);
                    if (!std::isfinite(old_residual) || !std::isfinite(next) || !std::isfinite(stored))
                        reject_nested_execution("child theta compensation residual forcing is invalid");
                    residual_candidate[i][j][k] = stored;
                }
        return true;
    }
    return false;
}

void apply_nested_child_mapped_relaxation(SimulationStorageState&                      storage,
                                          const PrognosticFieldStorage&                previous_parent_fields,
                                          const tmv::domain::NestedRelaxationConfigV1& config,
                                          double                                       child_step_s,
                                          const MoistMixtureEnergyBudgetV2&            old_energy,
                                          NestedChildRelaxationReport&                 report,
                                          const NestedParentMicrophysicsSnapshotV1*    previous_parent_microphysics,
                                          bool                                         simultaneous_private,
                                          NestedRelaxationCandidateValidator           validator,
                                          void*                                        validator_context)
{
    auto& nested = storage.nested_execution;
    if (!nested.active_terrain.ready)
        reject_nested_execution("mapped relaxation requires published active-terrain child state");
    const auto&                     r        = nested.relationship;
    const auto&                     geometry = nested.active_terrain.geometry;
    PrognosticFieldStorage          target;
    NestedChildInitializationReport target_report;
    auto                            target_active = nested.active_terrain;
    prepare_nested_child_mapped_primary_state(storage, previous_parent_fields, target, target_report, target_active);

    const auto& old       = nested.child.fields;
    auto&       candidate = nested.boundary_scratch.candidate;
    copy_primary_reusing_storage(candidate, old);
    NestedChildRelaxationReport candidate_report;
    const auto&                 metrics     = geometry.child_storage.metrics;
    const auto                  fraction_at = [&](int i, int j)
    { return tmv::domain::nested_scalar_relaxation_fraction(r, config, i, j, child_step_s); };

    long double target_dry_change    = 0.0L;
    long double committed_dry_change = 0.0L;
    std::size_t forced_cells         = 0;
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
        {
            const double fraction = fraction_at(i, j);
            if (fraction > 0.0)
                forced_cells += static_cast<std::size_t>(r.levels_z);
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double old_rho    = old.rho[i + 1][j + 1][k];
                const double target_rho = target.rho[i + 1][j + 1][k];
                const double next_rho   = old_rho + fraction * (target_rho - old_rho);
                const float  stored     = static_cast<float>(next_rho);
                if (!std::isfinite(stored) || stored <= 0.0f)
                    reject_nested_execution("mapped relaxed dry density is not representable");
                candidate.rho[i + 1][j + 1][k] = stored;
                const long double volume       = metrics.J(i + 1, j + 1, k) * metrics.dx * metrics.dy * metrics.dzeta;
                target_dry_change += fraction * (target_rho - old_rho) * volume;
                committed_dry_change += (stored - old_rho) * volume;
            }
        }

    const auto relax_specific = [&](const Field3D&                                old_specific,
                                    const Field3D&                                target_specific,
                                    bool                                          nonnegative,
                                    Field3D&                                      destination,
                                    tmv::domain::ConservedScalarRelaxationReport& scalar_report)
    {
        long double target_change    = 0.0L;
        long double committed_change = 0.0L;
        for (int i = 0; i < r.child_cells_x; ++i)
            for (int j = 0; j < r.child_cells_y; ++j)
            {
                const double fraction = fraction_at(i, j);
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double old_rho        = old.rho[i + 1][j + 1][k];
                    const double target_rho     = target.rho[i + 1][j + 1][k];
                    const double new_rho        = candidate.rho[i + 1][j + 1][k];
                    const double old_carrier    = old_rho * old_specific[i + 1][j + 1][k];
                    const double target_carrier = target_rho * target_specific[i + 1][j + 1][k];
                    const double next_carrier   = old_carrier + fraction * (target_carrier - old_carrier);
                    const float  stored         = static_cast<float>(next_carrier / new_rho);
                    if (!std::isfinite(stored) || (nonnegative && stored < 0.0f))
                        reject_nested_execution("mapped relaxed carrier is not representable");
                    destination[i + 1][j + 1][k] = stored;
                    const long double volume     = metrics.J(i + 1, j + 1, k) * metrics.dx * metrics.dy * metrics.dzeta;
                    target_change += fraction * (target_carrier - old_carrier) * volume;
                    committed_change += (static_cast<double>(stored) * new_rho - old_carrier) * volume;
                }
            }
        scalar_report.target_dry_mass_change_kg    = static_cast<double>(target_dry_change);
        scalar_report.committed_dry_mass_change_kg = static_cast<double>(committed_dry_change);
        scalar_report.dry_storage_residual_kg =
            scalar_report.target_dry_mass_change_kg - scalar_report.committed_dry_mass_change_kg;
        scalar_report.target_carrier_change    = static_cast<double>(target_change);
        scalar_report.committed_carrier_change = static_cast<double>(committed_change);
        scalar_report.carrier_storage_residual =
            scalar_report.target_carrier_change - scalar_report.committed_carrier_change;
        scalar_report.forced_cells = forced_cells;
    };

    relax_specific(old.theta, target.theta, false, candidate.theta, candidate_report.theta);
    relax_specific(old.qv, target.qv, true, candidate.qv, candidate_report.qv);
    relax_specific(old.qc, target.qc, true, candidate.qc, candidate_report.qc);
    relax_specific(old.qr, target.qr, true, candidate.qr, candidate_report.qr);
    relax_specific(old.qi, target.qi, true, candidate.qi, candidate_report.qi);
    relax_specific(old.qs, target.qs, true, candidate.qs, candidate_report.qs);
    relax_specific(old.qg, target.qg, true, candidate.qg, candidate_report.qg);
    relax_specific(old.qh, target.qh, true, candidate.qh, candidate_report.qh);
    relax_specific(old.tracer, target.tracer, true, candidate.tracer, candidate_report.tracer);
    relax_specific(old.tke, target.tke, true, candidate.tke, candidate_report.tke);

    NestedLivePrivateMicrophysicsBindings private_bindings;
    const auto*                           private_scheme = nested.child.runtime.microphysics.scheme.get();
    if (private_scheme && !nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport().empty())
    {
        if (simultaneous_private)
            private_bindings =
                require_simultaneous_private_microphysics_bindings(storage, *nested.child.runtime.microphysics.scheme);
        else
        {
            if (previous_parent_microphysics == nullptr)
                reject_nested_execution("mapped private forcing requires the previous parent endpoint");
            private_bindings = require_private_microphysics_bindings(storage, *previous_parent_microphysics);
        }
        const auto parent_dry = mapped_parent_conservative_state(
            geometry, nested.parent_time_bracket, previous_parent_fields, storage.fields, nullptr, nullptr);
        tmv::domain::NestedTerrainTransferLedgerV1 dry_ledger;
        const auto child_dry = tmv::domain::prolong_nested_mapped_conservative_state(geometry, parent_dry, &dry_ledger);
        auto&      private_report   = candidate_report.microphysics_private;
        private_report.scalar_count = private_bindings.child.size();
        Field3D private_target(nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz, 0.0f);
        for (std::size_t index = 0; index < private_bindings.child.size(); ++index)
        {
            const auto& binding           = private_bindings.child[index];
            auto&       private_candidate = nested.boundary_scratch.microphysics_private_candidate[index];
            private_candidate             = *binding.field;
            tmv::domain::ConservedScalarProlongationReport target_private_report;
            prolong_mapped_specific(geometry,
                                    nested.parent_time_bracket,
                                    previous_parent_fields,
                                    storage.fields,
                                    child_dry,
                                    dry_ledger,
                                    target.rho,
                                    simultaneous_private ? *private_bindings.current_parent[index].field
                                                         : previous_parent_microphysics->fields[index],
                                    *private_bindings.current_parent[index].field,
                                    binding.positive_definite,
                                    private_target,
                                    target_private_report);
            tmv::domain::ConservedScalarRelaxationReport working_report;
            relax_specific(
                *binding.field, private_target, binding.positive_definite, private_candidate, working_report);
            fill_mapped_private_closure(r, private_candidate);
            private_report.name[index]                     = bounded_private_scalar_name(binding.name);
            private_report.storage_semantic[index]         = std::uint8_t(binding.storage);
            private_report.positive_definite[index]        = binding.positive_definite;
            private_report.target_carrier_change[index]    = working_report.target_carrier_change;
            private_report.committed_carrier_change[index] = working_report.committed_carrier_change;
            private_report.carrier_storage_residual[index] = working_report.carrier_storage_residual;
            private_report.forced_cells[index]             = working_report.forced_cells;
        }
    }
    auto&      theta_residual = nested.child.runtime.microphysics.theta_storage_residual;
    const bool theta_residual_active =
        prepare_nested_theta_residual(nested, config, child_step_s, old.rho, candidate.rho, theta_residual);

    long double target_x    = 0.0L;
    long double committed_x = 0.0L;
    for (int face = 0; face <= r.child_cells_x; ++face)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double fraction        = std::max(face > 0 ? fraction_at(face - 1, j) : 0.0,
                                                        face < r.child_cells_x ? fraction_at(face, j) : 0.0);
                const double old_density     = mapped_child_x_effective_density(geometry, old.rho, face, j, k);
                const double target_density  = mapped_child_x_effective_density(geometry, target.rho, face, j, k);
                const double new_density     = mapped_child_x_effective_density(geometry, candidate.rho, face, j, k);
                const double old_momentum    = old_density * old.u[face][j + 1][k];
                const double target_momentum = target_density * target.u[face][j + 1][k];
                const float  stored =
                    static_cast<float>((old_momentum + fraction * (target_momentum - old_momentum)) / new_density);
                if (!std::isfinite(stored))
                    reject_nested_execution("mapped relaxed x momentum is not representable");
                candidate.u[face][j + 1][k] = stored;
                const long double measure   = metrics.Jx_face(face + 1, j + 1, k) * metrics.dx * metrics.dy *
                                              metrics.dzeta * (face == 0 || face == r.child_cells_x ? .5 : 1.0);
                target_x += fraction * (target_momentum - old_momentum) * measure;
                committed_x += (new_density * stored - old_momentum) * measure;
            }
    long double target_y    = 0.0L;
    long double committed_y = 0.0L;
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int face = 0; face <= r.child_cells_y; ++face)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double fraction        = std::max(face > 0 ? fraction_at(i, face - 1) : 0.0,
                                                        face < r.child_cells_y ? fraction_at(i, face) : 0.0);
                const double old_density     = mapped_child_y_effective_density(geometry, old.rho, i, face, k);
                const double target_density  = mapped_child_y_effective_density(geometry, target.rho, i, face, k);
                const double new_density     = mapped_child_y_effective_density(geometry, candidate.rho, i, face, k);
                const double old_momentum    = old_density * old.v[i + 1][face][k];
                const double target_momentum = target_density * target.v[i + 1][face][k];
                const float  stored =
                    static_cast<float>((old_momentum + fraction * (target_momentum - old_momentum)) / new_density);
                if (!std::isfinite(stored))
                    reject_nested_execution("mapped relaxed y momentum is not representable");
                candidate.v[i + 1][face][k] = stored;
                const long double measure   = metrics.Jy_face(i + 1, face + 1, k) * metrics.dx * metrics.dy *
                                              metrics.dzeta * (face == 0 || face == r.child_cells_y ? .5 : 1.0);
                target_y += fraction * (target_momentum - old_momentum) * measure;
                committed_y += (new_density * stored - old_momentum) * measure;
            }
    long double target_z    = 0.0L;
    long double committed_z = 0.0L;
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double fraction        = fraction_at(i, j);
                const double old_mass        = mapped_child_z_dual_mass_per_area(geometry, old.rho, i, j, k);
                const double target_mass     = mapped_child_z_dual_mass_per_area(geometry, target.rho, i, j, k);
                const double new_mass        = mapped_child_z_dual_mass_per_area(geometry, candidate.rho, i, j, k);
                const double old_momentum    = old_mass * old.w[i + 1][j + 1][k];
                const double target_momentum = target_mass * target.w[i + 1][j + 1][k];
                const float  stored =
                    static_cast<float>((old_momentum + fraction * (target_momentum - old_momentum)) / new_mass);
                if (!std::isfinite(stored))
                    reject_nested_execution("mapped relaxed vertical momentum is not representable");
                candidate.w[i + 1][j + 1][k] = stored;
                const long double area       = metrics.dx * metrics.dy;
                target_z += fraction * (target_momentum - old_momentum) * area;
                committed_z += (new_mass * stored - old_momentum) * area;
            }
    candidate_report.momentum.target_x_change_kgms    = static_cast<double>(target_x);
    candidate_report.momentum.committed_x_change_kgms = static_cast<double>(committed_x);
    candidate_report.momentum.x_storage_residual_kgms =
        candidate_report.momentum.target_x_change_kgms - candidate_report.momentum.committed_x_change_kgms;
    candidate_report.momentum.target_y_change_kgms    = static_cast<double>(target_y);
    candidate_report.momentum.committed_y_change_kgms = static_cast<double>(committed_y);
    candidate_report.momentum.y_storage_residual_kgms =
        candidate_report.momentum.target_y_change_kgms - candidate_report.momentum.committed_y_change_kgms;
    candidate_report.momentum.target_z_change_kgms    = static_cast<double>(target_z);
    candidate_report.momentum.committed_z_change_kgms = static_cast<double>(committed_z);
    candidate_report.momentum.z_storage_residual_kgms =
        candidate_report.momentum.target_z_change_kgms - candidate_report.momentum.committed_z_change_kgms;

    const auto clamp_index = [](int value, int upper) { return std::max(1, std::min(value, upper)); };
    for (int i = 0; i < nested.child.grid.nr; ++i)
        for (int j = 0; j < nested.child.grid.nth; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const int source_i = clamp_index(i, r.child_cells_x);
                const int source_j = clamp_index(j, r.child_cells_y);
                if (i == 0 || i == nested.child.grid.nr - 1 || j == 0 || j == nested.child.grid.nth - 1)
                {
                    for (Field3D* field : {&candidate.rho,
                                           &candidate.theta,
                                           &candidate.qv,
                                           &candidate.qc,
                                           &candidate.qr,
                                           &candidate.qi,
                                           &candidate.qs,
                                           &candidate.qg,
                                           &candidate.qh,
                                           &candidate.tracer,
                                           &candidate.tke,
                                           &candidate.w})
                        (*field)[i][j][k] = (*field)[source_i][source_j][k];
                }
                if (i > r.child_cells_x || j < 1 || j > r.child_cells_y)
                    candidate.u[i][j][k] = candidate.u[std::min(i, r.child_cells_x)][source_j][k];
                if (j > r.child_cells_y || i < 1 || i > r.child_cells_x)
                    candidate.v[i][j][k] = candidate.v[source_i][std::min(j, r.child_cells_y)][k];
                bool           unchanged = child_step_s == 0.0;
                const Field3D* before[]  = {
                    &old.rho, &old.theta, &old.qv, &old.qc, &old.qr, &old.qi, &old.qs, &old.qg, &old.qh};
                const Field3D* after[] = {&candidate.rho,
                                          &candidate.theta,
                                          &candidate.qv,
                                          &candidate.qc,
                                          &candidate.qr,
                                          &candidate.qi,
                                          &candidate.qs,
                                          &candidate.qg,
                                          &candidate.qh};
                for (std::size_t f = 0; unchanged && f < std::size(before); ++f)
                    unchanged = (*before[f])[i][j][k] == (*after[f])[i][j][k];
                const float pressure = unchanged ? old.p[i][j][k]
                                                 : static_cast<float>(tmv::dynamics_eos::pressure_fp64(
                                                       candidate.rho[i][j][k], candidate.theta[i][j][k]));
                if (!std::isfinite(pressure) || pressure <= 0.0f)
                    reject_nested_execution("mapped relaxed pressure is not representable");
                candidate.p[i][j][k] = pressure;
            }

    using std::swap;
    swap(nested.child.fields, candidate);
    MoistMixtureEnergyBudgetV2 new_energy;
    try
    {
        const auto candidate_state = make_simulation_state_view(
            nested.child, storage.configuration, &storage.runtime.compute, &storage.runtime.parity);
        new_energy = compute_moist_mixture_energy_v2_deterministic(candidate_state);
    }
    catch (...)
    {
        swap(nested.child.fields, candidate);
        throw;
    }
    swap(nested.child.fields, candidate);
    const auto energy_change = [](const MoistMixtureEnergyBudgetV2& after, const MoistMixtureEnergyBudgetV2& before)
    {
        MoistMixtureEnergyBudgetV2 change;
        change.dry_internal     = after.dry_internal - before.dry_internal;
        change.water_sensible   = after.water_sensible - before.water_sensible;
        change.phase_reference  = after.phase_reference - before.phase_reference;
        change.resolved_kinetic = after.resolved_kinetic - before.resolved_kinetic;
        change.gravitational    = after.gravitational - before.gravitational;
        change.subgrid_tke      = after.subgrid_tke - before.subgrid_tke;
        change.total_water_mass = after.total_water_mass - before.total_water_mass;
        change.total            = after.total - before.total;
        return change;
    };
    candidate_report.energy.before           = old_energy;
    candidate_report.energy.after            = new_energy;
    candidate_report.energy.committed_change = energy_change(new_energy, old_energy);
    const auto& committed_energy             = candidate_report.energy.committed_change;
    candidate_report.energy.reservoir_sum_closure_residual_j =
        committed_energy.total -
        (committed_energy.dry_internal + committed_energy.water_sensible + committed_energy.phase_reference +
         committed_energy.resolved_kinetic + committed_energy.gravitational + committed_energy.subgrid_tke);

    auto active_candidate                 = nested.active_terrain;
    active_candidate.child_time_s         = nested.child.time_s;
    active_candidate.parent_time_s        = nested.child.time_s;
    active_candidate.ledger.child_time_s  = nested.child.time_s;
    active_candidate.ledger.parent_time_s = nested.child.time_s;
    active_candidate.child_state          = mapped_conservative_state_from_child_fields(geometry, candidate);
    long double child_dry_inventory       = 0.0L;
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
                child_dry_inventory += static_cast<long double>(active_candidate.child_state.scalar(i, j, k)) *
                                       metrics.J(i + 1, j + 1, k) * metrics.dx * metrics.dy * metrics.dzeta;
    active_candidate.ledger.child_scalar_inventory = static_cast<double>(child_dry_inventory);
    active_candidate.ledger.scalar_inventory_residual =
        active_candidate.ledger.child_scalar_inventory - active_candidate.ledger.parent_scalar_inventory;
    active_candidate.ledger.normalized_scalar_inventory_residual =
        std::abs(active_candidate.ledger.scalar_inventory_residual) /
        std::max({1.0,
                  std::abs(active_candidate.ledger.parent_scalar_inventory),
                  std::abs(active_candidate.ledger.child_scalar_inventory)});
    active_candidate.boundary = tmv::domain::build_nested_terrain_tangent_boundary(
        geometry, candidate.u[0][1][0], candidate.v[1][0][0], &active_candidate.ledger);
    if (active_candidate.publication_count == std::numeric_limits<std::uint64_t>::max() ||
        active_candidate.generation == std::numeric_limits<std::uint64_t>::max())
        reject_nested_execution("mapped child publication counter is exhausted");
    ++active_candidate.publication_count;
    ++active_candidate.generation;
    active_candidate.geometry.generation = active_candidate.generation;
    if (validator)
        validator(candidate, candidate_report, validator_context);

    auto interface_source = nested.interface_source;
    accumulate_interface_source(interface_source, candidate_report);
    if (nested.child.runtime.terrain.earth_fixed_placement.active)
        nested.child.runtime.terrain.earth_fixed_placement.generation = active_candidate.generation;
    swap(nested.child.fields, candidate);
    for (std::size_t index = 0; index < private_bindings.child.size(); ++index)
        swap(*private_bindings.child[index].field, nested.boundary_scratch.microphysics_private_candidate[index]);
    if (theta_residual_active)
        swap(theta_residual, nested.boundary_scratch.theta_storage_residual_candidate);
    nested.active_terrain              = std::move(active_candidate);
    nested.interface_source            = std::move(interface_source);
    nested.parent_boundary_state_ready = true;
    report                             = candidate_report;
}

void apply_nested_child_physical_relaxation_implementation(
    SimulationStorageState&                      storage,
    const PrognosticFieldStorage&                previous_parent_fields,
    const tmv::domain::NestedRelaxationConfigV1& config,
    double                                       child_step_s,
    NestedChildRelaxationReport&                 report,
    const NestedParentMicrophysicsSnapshotV1*    previous_parent_microphysics,
    tmv::MicrophysicsScheme*                     prepared_child_microphysics,
    Field3D*                                     prepared_theta_storage_residual,
    NestedRelaxationCandidateValidator           validator,
    void*                                        validator_context)
{
    require_valid_nested_execution_relationship(storage);
    auto& nested = storage.nested_execution;
    if (!nested.physical_primary_state_ready)
        reject_nested_execution("physical child initialization must precede relaxation");
    auto old_state = make_simulation_state_view(
        nested.child, storage.configuration, &storage.runtime.compute, &storage.runtime.parity);
    const auto old_energy = compute_moist_mixture_energy_v2_deterministic(old_state);
    if (nested_active_terrain_geometry_ready(nested))
    {
        if ((prepared_child_microphysics != nullptr &&
             prepared_child_microphysics != nested.child.runtime.microphysics.scheme.get()) ||
            (prepared_theta_storage_residual != nullptr &&
             prepared_theta_storage_residual != &nested.child.runtime.microphysics.theta_storage_residual))
            reject_nested_execution("mapped relocation requires owner-local private physics state");
        apply_nested_child_mapped_relaxation(storage,
                                             previous_parent_fields,
                                             config,
                                             child_step_s,
                                             old_energy,
                                             report,
                                             previous_parent_microphysics,
                                             prepared_child_microphysics != nullptr,
                                             validator,
                                             validator_context);
        return;
    }
    const auto& relationship = nested.relationship;
    const auto& dz           = nested.child.process_grid_metrics.dz;
    const int   nr           = nested.child.grid.nr;
    const int   nth          = nested.child.grid.nth;
    const int   nz           = nested.child.grid.nz;
    const float nan          = std::numeric_limits<float>::quiet_NaN();
    auto&       target       = nested.boundary_scratch.target;
    for (Field3D* field : {&target.rho, &target.theta, &target.u, &target.v, &target.w})
        field->resize(nr, nth, nz, nan);
    tmv::domain::ConservedScalarProlongationReport target_scalar_report;
    tmv::domain::prolong_conserved_scalar_flat_storage(relationship,
                                                       nested.parent_time_bracket,
                                                       dz,
                                                       previous_parent_fields.rho,
                                                       storage.fields.rho,
                                                       previous_parent_fields.theta,
                                                       storage.fields.theta,
                                                       false,
                                                       target.rho,
                                                       target.theta,
                                                       target_scalar_report);
    tmv::domain::ConservedMomentumProlongationReport target_momentum_report;
    tmv::domain::prolong_cgrid_momentum_flat_storage(relationship,
                                                     nested.parent_time_bracket,
                                                     dz,
                                                     previous_parent_fields.rho,
                                                     storage.fields.rho,
                                                     previous_parent_fields.u,
                                                     storage.fields.u,
                                                     previous_parent_fields.v,
                                                     storage.fields.v,
                                                     previous_parent_fields.w,
                                                     storage.fields.w,
                                                     target.rho,
                                                     target.u,
                                                     target.v,
                                                     target.w,
                                                     target_momentum_report);

    const auto& old       = nested.child.fields;
    auto&       candidate = nested.boundary_scratch.candidate;
    copy_primary_reusing_storage(candidate, old);
    NestedChildRelaxationReport candidate_report;
    const auto                  relax = [&](const Field3D&                                previous_specific,
                                            const Field3D&                                current_specific,
                                            const Field3D&                                old_specific,
                                            bool                                          nonnegative,
                                            Field3D&                                      candidate_specific,
                                            tmv::domain::ConservedScalarRelaxationReport& scalar_report)
    {
        tmv::domain::relax_conserved_scalar_flat_storage(relationship,
                                                         nested.parent_time_bracket,
                                                         config,
                                                         child_step_s,
                                                         dz,
                                                         previous_parent_fields.rho,
                                                         storage.fields.rho,
                                                         previous_specific,
                                                         current_specific,
                                                         old.rho,
                                                         old_specific,
                                                         nonnegative,
                                                         candidate.rho,
                                                         candidate_specific,
                                                         scalar_report);
    };
    relax(
        previous_parent_fields.theta, storage.fields.theta, old.theta, false, candidate.theta, candidate_report.theta);
    relax(previous_parent_fields.qv, storage.fields.qv, old.qv, true, candidate.qv, candidate_report.qv);
    relax(previous_parent_fields.qc, storage.fields.qc, old.qc, true, candidate.qc, candidate_report.qc);
    relax(previous_parent_fields.qr, storage.fields.qr, old.qr, true, candidate.qr, candidate_report.qr);
    relax(previous_parent_fields.qi, storage.fields.qi, old.qi, true, candidate.qi, candidate_report.qi);
    relax(previous_parent_fields.qs, storage.fields.qs, old.qs, true, candidate.qs, candidate_report.qs);
    relax(previous_parent_fields.qg, storage.fields.qg, old.qg, true, candidate.qg, candidate_report.qg);
    relax(previous_parent_fields.qh, storage.fields.qh, old.qh, true, candidate.qh, candidate_report.qh);
    relax(previous_parent_fields.tracer,
          storage.fields.tracer,
          old.tracer,
          true,
          candidate.tracer,
          candidate_report.tracer);
    relax(previous_parent_fields.tke, storage.fields.tke, old.tke, true, candidate.tke, candidate_report.tke);
    const auto same_dry = [&](const auto& value)
    {
        return value.target_dry_mass_change_kg == candidate_report.theta.target_dry_mass_change_kg &&
               value.committed_dry_mass_change_kg == candidate_report.theta.committed_dry_mass_change_kg &&
               value.dry_storage_residual_kg == candidate_report.theta.dry_storage_residual_kg;
    };
    const tmv::domain::ConservedScalarRelaxationReport* scalar_reports[] = {&candidate_report.qv,
                                                                            &candidate_report.qc,
                                                                            &candidate_report.qr,
                                                                            &candidate_report.qi,
                                                                            &candidate_report.qs,
                                                                            &candidate_report.qg,
                                                                            &candidate_report.qh,
                                                                            &candidate_report.tracer,
                                                                            &candidate_report.tke};
    for (const auto* value : scalar_reports)
        if (!same_dry(*value))
            reject_nested_execution("relaxed carriers do not share one dry-mass source");

    NestedLivePrivateMicrophysicsBindings private_bindings;
    std::size_t                           private_candidate_count = 0;
    tmv::MicrophysicsScheme*              private_child_scheme = prepared_child_microphysics != nullptr
                                                                     ? prepared_child_microphysics
                                                                     : nested.child.runtime.microphysics.scheme.get();
    const bool                            private_state_active =
        private_child_scheme && !private_child_scheme->prognostic_scalars_for_transport().empty();
    if (private_state_active)
    {
        auto&      private_report              = candidate_report.microphysics_private;
        const bool simultaneous_prepared_state = prepared_child_microphysics != nullptr;
        if (simultaneous_prepared_state)
            private_bindings =
                require_simultaneous_private_microphysics_bindings(storage, *prepared_child_microphysics);
        else
        {
            if (previous_parent_microphysics == nullptr)
                reject_nested_execution("private microphysics relaxation requires the previous parent endpoint");
            private_bindings = require_private_microphysics_bindings(storage, *previous_parent_microphysics);
        }
        private_candidate_count     = private_bindings.child.size();
        private_report.scalar_count = private_candidate_count;
        for (std::size_t index = 0; index < private_candidate_count; ++index)
        {
            const Field3D& previous_private = simultaneous_prepared_state
                                                  ? *private_bindings.current_parent[index].field
                                                  : previous_parent_microphysics->fields[index];
            const Field3D& previous_rho = simultaneous_prepared_state ? storage.fields.rho : previous_parent_fields.rho;
            auto&          private_candidate = nested.boundary_scratch.microphysics_private_candidate[index];
            private_candidate                = *private_bindings.child[index].field;
            tmv::domain::ConservedScalarRelaxationReport working_report;
            tmv::domain::relax_conserved_scalar_flat_storage(relationship,
                                                             nested.parent_time_bracket,
                                                             config,
                                                             child_step_s,
                                                             dz,
                                                             previous_rho,
                                                             storage.fields.rho,
                                                             previous_private,
                                                             *private_bindings.current_parent[index].field,
                                                             old.rho,
                                                             *private_bindings.child[index].field,
                                                             private_bindings.child[index].positive_definite,
                                                             candidate.rho,
                                                             private_candidate,
                                                             working_report);
            if (!same_dry(working_report))
                reject_nested_execution("private microphysics relaxation does not share the primary dry-mass source");
            private_report.name[index] = bounded_private_scalar_name(private_bindings.current_parent[index].name);
            private_report.storage_semantic[index] =
                static_cast<std::uint8_t>(private_bindings.current_parent[index].storage);
            private_report.positive_definite[index]        = private_bindings.current_parent[index].positive_definite;
            private_report.target_carrier_change[index]    = working_report.target_carrier_change;
            private_report.committed_carrier_change[index] = working_report.committed_carrier_change;
            private_report.carrier_storage_residual[index] = working_report.carrier_storage_residual;
            private_report.forced_cells[index]             = working_report.forced_cells;
            fill_private_mixing_ratio_closure(relationship,
                                              nested.parent_time_bracket,
                                              previous_rho,
                                              storage.fields.rho,
                                              previous_private,
                                              *private_bindings.current_parent[index].field,
                                              private_bindings.child[index].positive_definite,
                                              private_candidate);
        }
    }
    auto&      live_theta_residual = prepared_theta_storage_residual != nullptr
                                         ? *prepared_theta_storage_residual
                                         : nested.child.runtime.microphysics.theta_storage_residual;
    const bool theta_residual_candidate_active =
        prepare_nested_theta_residual(nested, config, child_step_s, old.rho, candidate.rho, live_theta_residual);
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const int   i              = ci + 1;
                const int   j              = cj + 1;
                const float rho            = candidate.rho[i][j][k];
                const float theta          = candidate.theta[i][j][k];
                const float qv             = candidate.qv[i][j][k];
                const float hydrometeors[] = {candidate.qc[i][j][k],
                                              candidate.qr[i][j][k],
                                              candidate.qi[i][j][k],
                                              candidate.qs[i][j][k],
                                              candidate.qg[i][j][k],
                                              candidate.qh[i][j][k]};
                if (!std::isfinite(rho) || rho < density_min_kgm3 || rho > density_max_kgm3 || !std::isfinite(theta) ||
                    theta < theta_min_k || theta > theta_max_k || !std::isfinite(qv) || qv < 0.0f || qv > qv_max_kgkg ||
                    !std::all_of(std::begin(hydrometeors),
                                 std::end(hydrometeors),
                                 [](float value)
                                 { return std::isfinite(value) && value >= 0.0f && value <= hydrometeor_max_kgkg; }) ||
                    !std::isfinite(candidate.tracer[i][j][k]) || candidate.tracer[i][j][k] < 0.0f ||
                    candidate.tracer[i][j][k] > passive_tracer_max || !std::isfinite(candidate.tke[i][j][k]) ||
                    candidate.tke[i][j][k] < 0.0f)
                {
                    reject_nested_execution("relaxed child scalar state exceeds strict physical bounds");
                }
            }
    tmv::domain::diagnose_child_pressure_flat_storage(
        relationship, dz, candidate.rho, candidate.theta, candidate.p, candidate_report.pressure);

    const auto weights   = tmv::domain::parent_temporal_weights(relationship, nested.parent_time_bracket);
    const auto floor_div = [](int value, int divisor)
    { return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor); };
    const auto fill_scalar_closure = [&](const Field3D& previous_specific,
                                         const Field3D& current_specific,
                                         double         minimum,
                                         double         maximum,
                                         Field3D&       destination)
    {
        for (int si = 0; si < nr; ++si)
            for (int sj = 0; sj < nth; ++sj)
            {
                if (si >= 1 && si <= relationship.child_cells_x && sj >= 1 && sj <= relationship.child_cells_y)
                    continue;
                const int ci = si - 1;
                const int cj = sj - 1;
                const int pi = relationship.child_origin_parent_cell_i + floor_div(ci, relationship.parent_ratio_x);
                const int pj = relationship.child_origin_parent_cell_j + floor_div(cj, relationship.parent_ratio_y);
                for (int k = 0; k < nz; ++k)
                {
                    double rho     = 0.0;
                    double carrier = 0.0;
                    if (weights.current == 0.0)
                    {
                        rho     = previous_parent_fields.rho[pi + 1][pj + 1][k];
                        carrier = rho * previous_specific[pi + 1][pj + 1][k];
                    }
                    else if (weights.previous == 0.0)
                    {
                        rho     = storage.fields.rho[pi + 1][pj + 1][k];
                        carrier = rho * current_specific[pi + 1][pj + 1][k];
                    }
                    else
                    {
                        const double rp = previous_parent_fields.rho[pi + 1][pj + 1][k];
                        const double rc = storage.fields.rho[pi + 1][pj + 1][k];
                        rho             = weights.previous * rp + weights.current * rc;
                        carrier         = weights.previous * rp * previous_specific[pi + 1][pj + 1][k] +
                                          weights.current * rc * current_specific[pi + 1][pj + 1][k];
                    }
                    const double specific        = carrier / rho;
                    const float  stored_rho      = static_cast<float>(rho);
                    const float  stored_specific = static_cast<float>(specific);
                    if (!std::isfinite(rho) || rho < density_min_kgm3 || rho > density_max_kgm3 ||
                        !std::isfinite(specific) || specific < minimum || specific > maximum ||
                        !std::isfinite(stored_rho) || !std::isfinite(stored_specific))
                        reject_nested_execution("parent scalar closure target is invalid");
                    candidate.rho[si][sj][k] = stored_rho;
                    destination[si][sj][k]   = stored_specific;
                }
            }
    };
    fill_scalar_closure(previous_parent_fields.theta, storage.fields.theta, theta_min_k, theta_max_k, candidate.theta);
    fill_scalar_closure(previous_parent_fields.qv, storage.fields.qv, 0.0, qv_max_kgkg, candidate.qv);
    fill_scalar_closure(previous_parent_fields.qc, storage.fields.qc, 0.0, hydrometeor_max_kgkg, candidate.qc);
    fill_scalar_closure(previous_parent_fields.qr, storage.fields.qr, 0.0, hydrometeor_max_kgkg, candidate.qr);
    fill_scalar_closure(previous_parent_fields.qi, storage.fields.qi, 0.0, hydrometeor_max_kgkg, candidate.qi);
    fill_scalar_closure(previous_parent_fields.qs, storage.fields.qs, 0.0, hydrometeor_max_kgkg, candidate.qs);
    fill_scalar_closure(previous_parent_fields.qg, storage.fields.qg, 0.0, hydrometeor_max_kgkg, candidate.qg);
    fill_scalar_closure(previous_parent_fields.qh, storage.fields.qh, 0.0, hydrometeor_max_kgkg, candidate.qh);
    fill_scalar_closure(
        previous_parent_fields.tracer, storage.fields.tracer, 0.0, passive_tracer_max, candidate.tracer);
    fill_scalar_closure(
        previous_parent_fields.tke, storage.fields.tke, 0.0, std::numeric_limits<float>::max(), candidate.tke);
    const auto blend_velocity =
        [&](double previous_mass, double previous_velocity, double current_mass, double current_velocity)
    {
        if (weights.current == 0.0)
        {
            if (!std::isfinite(previous_mass) || previous_mass <= 0.0 || !std::isfinite(previous_velocity))
                reject_nested_execution("previous parent momentum closure state is invalid");
            return previous_velocity;
        }
        if (weights.previous == 0.0)
        {
            if (!std::isfinite(current_mass) || current_mass <= 0.0 || !std::isfinite(current_velocity))
                reject_nested_execution("current parent momentum closure state is invalid");
            return current_velocity;
        }
        if (!std::isfinite(previous_mass) || previous_mass <= 0.0 || !std::isfinite(current_mass) ||
            current_mass <= 0.0 || !std::isfinite(previous_velocity) || !std::isfinite(current_velocity))
            reject_nested_execution("parent momentum closure bracket is invalid");
        const double mass = weights.previous * previous_mass + weights.current * current_mass;
        const double momentum =
            weights.previous * previous_mass * previous_velocity + weights.current * current_mass * current_velocity;
        const double velocity = momentum / mass;
        if (!std::isfinite(velocity))
            reject_nested_execution("parent momentum closure interpolation overflowed");
        return velocity;
    };
    const auto u_closure = [&](int si, int sj, int k)
    {
        const double parent_face_coordinate =
            relationship.child_origin_parent_cell_i + static_cast<double>(si) / relationship.parent_ratio_x;
        const int    face = static_cast<int>(std::floor(parent_face_coordinate + 0.5));
        const double parent_scalar_coordinate =
            relationship.child_origin_parent_cell_j + (static_cast<double>(sj - 1) + 0.5) / relationship.parent_ratio_y;
        const int cell_j = static_cast<int>(std::floor(parent_scalar_coordinate));
        if (face <= 0 || face > relationship.parent_cells_x || cell_j < 0 || cell_j >= relationship.parent_cells_y)
            reject_nested_execution("x-wind closure coordinate exceeds the admitted parent stencil");
        const auto dual_density = [&](const Field3D& rho)
        {
            const double left  = rho[face][cell_j + 1][k];
            const double right = rho[face + 1][cell_j + 1][k];
            return 0.5 * (left + right);
        };
        return blend_velocity(dual_density(previous_parent_fields.rho),
                              previous_parent_fields.u[face][cell_j + 1][k],
                              dual_density(storage.fields.rho),
                              storage.fields.u[face][cell_j + 1][k]);
    };
    const auto v_closure = [&](int si, int sj, int k)
    {
        const double parent_scalar_coordinate =
            relationship.child_origin_parent_cell_i + (static_cast<double>(si - 1) + 0.5) / relationship.parent_ratio_x;
        const int    cell_i = static_cast<int>(std::floor(parent_scalar_coordinate));
        const double parent_face_coordinate =
            relationship.child_origin_parent_cell_j + static_cast<double>(sj) / relationship.parent_ratio_y;
        const int face = static_cast<int>(std::floor(parent_face_coordinate + 0.5));
        if (cell_i < 0 || cell_i >= relationship.parent_cells_x || face <= 0 || face > relationship.parent_cells_y)
            reject_nested_execution("y-wind closure coordinate exceeds the admitted parent stencil");
        const auto dual_density = [&](const Field3D& rho)
        {
            const double south = rho[cell_i + 1][face][k];
            const double north = rho[cell_i + 1][face + 1][k];
            return 0.5 * (south + north);
        };
        return blend_velocity(dual_density(previous_parent_fields.rho),
                              previous_parent_fields.v[cell_i + 1][face][k],
                              dual_density(storage.fields.rho),
                              storage.fields.v[cell_i + 1][face][k]);
    };
    const auto w_closure = [&](int si, int sj, int k)
    {
        const double parent_x =
            relationship.child_origin_parent_cell_i + (static_cast<double>(si - 1) + 0.5) / relationship.parent_ratio_x;
        const double parent_y =
            relationship.child_origin_parent_cell_j + (static_cast<double>(sj - 1) + 0.5) / relationship.parent_ratio_y;
        const int cell_i = static_cast<int>(std::floor(parent_x));
        const int cell_j = static_cast<int>(std::floor(parent_y));
        if (cell_i < 0 || cell_i >= relationship.parent_cells_x || cell_j < 0 || cell_j >= relationship.parent_cells_y)
            reject_nested_execution("vertical-wind closure coordinate exceeds the admitted parent stencil");
        const auto dual_mass = [&](const Field3D& rho)
        {
            double value = rho[cell_i + 1][cell_j + 1][k] * dz[static_cast<std::size_t>(k)];
            if (k + 1 < nz)
                value += rho[cell_i + 1][cell_j + 1][k + 1] * dz[static_cast<std::size_t>(k + 1)];
            return 0.5 * value;
        };
        return blend_velocity(dual_mass(previous_parent_fields.rho),
                              previous_parent_fields.w[cell_i + 1][cell_j + 1][k],
                              dual_mass(storage.fields.rho),
                              storage.fields.w[cell_i + 1][cell_j + 1][k]);
    };
    for (int si = 0; si < nr; ++si)
        for (int sj = 0; sj < nth; ++sj)
            for (int k = 0; k < nz; ++k)
            {
                if (!(si >= 0 && si <= relationship.child_cells_x && sj >= 1 && sj <= relationship.child_cells_y))
                {
                    const float value = static_cast<float>(u_closure(si, sj, k));
                    if (!std::isfinite(value) || std::abs(value) > wind_horizontal_abs_max_ms)
                        reject_nested_execution("parent x-wind closure target is invalid");
                    candidate.u[si][sj][k] = value;
                }
                if (!(si >= 1 && si <= relationship.child_cells_x && sj >= 0 && sj <= relationship.child_cells_y))
                {
                    const float value = static_cast<float>(v_closure(si, sj, k));
                    if (!std::isfinite(value) || std::abs(value) > wind_horizontal_abs_max_ms)
                        reject_nested_execution("parent y-wind closure target is invalid");
                    candidate.v[si][sj][k] = value;
                }
                if (!(si >= 1 && si <= relationship.child_cells_x && sj >= 1 && sj <= relationship.child_cells_y))
                {
                    const float value = static_cast<float>(w_closure(si, sj, k));
                    if (!std::isfinite(value) || std::abs(value) > wind_vertical_abs_max_ms)
                        reject_nested_execution("parent vertical-wind closure target is invalid");
                    candidate.w[si][sj][k] = value;
                }
                if (!(si >= 1 && si <= relationship.child_cells_x && sj >= 1 && sj <= relationship.child_cells_y))
                {
                    const float p = static_cast<float>(
                        tmv::dynamics_eos::pressure_fp64(candidate.rho[si][sj][k], candidate.theta[si][sj][k]));
                    if (!std::isfinite(p) || p < pressure_min_pa || p > pressure_max_pa)
                        reject_nested_execution("parent pressure closure target is invalid");
                    candidate.p[si][sj][k] = p;
                }
            }

    tmv::domain::relax_cgrid_momentum_flat_storage(relationship,
                                                   config,
                                                   child_step_s,
                                                   dz,
                                                   old.rho,
                                                   old.u,
                                                   old.v,
                                                   old.w,
                                                   target.rho,
                                                   target.u,
                                                   target.v,
                                                   target.w,
                                                   candidate.rho,
                                                   candidate.u,
                                                   candidate.v,
                                                   candidate.w,
                                                   candidate_report.momentum);
    for (int face = 0; face <= relationship.child_cells_x; ++face)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
                if (!std::isfinite(candidate.u[face][cj + 1][k]) ||
                    std::abs(candidate.u[face][cj + 1][k]) > wind_horizontal_abs_max_ms)
                    reject_nested_execution("relaxed child x wind exceeds strict physical bounds");
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int face = 0; face <= relationship.child_cells_y; ++face)
            for (int k = 0; k < relationship.levels_z; ++k)
                if (!std::isfinite(candidate.v[ci + 1][face][k]) ||
                    std::abs(candidate.v[ci + 1][face][k]) > wind_horizontal_abs_max_ms)
                    reject_nested_execution("relaxed child y wind exceeds strict physical bounds");
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
                if (!std::isfinite(candidate.w[ci + 1][cj + 1][k]) ||
                    std::abs(candidate.w[ci + 1][cj + 1][k]) > wind_vertical_abs_max_ms)
                    reject_nested_execution("relaxed child vertical wind exceeds strict physical bounds");

    MoistMixtureEnergyBudgetV2 new_energy;
    using std::swap;
    swap(nested.child.fields, candidate);
    try
    {
        auto candidate_state = make_simulation_state_view(
            nested.child, storage.configuration, &storage.runtime.compute, &storage.runtime.parity);
        candidate_state.terrain_runtime = old_state.terrain_runtime;
        new_energy                      = compute_moist_mixture_energy_v2_deterministic(candidate_state);
    }
    catch (...)
    {
        swap(nested.child.fields, candidate);
        throw;
    }
    swap(nested.child.fields, candidate);
    const auto energy_change = [](const MoistMixtureEnergyBudgetV2& after, const MoistMixtureEnergyBudgetV2& before)
    {
        MoistMixtureEnergyBudgetV2 change;
        change.dry_internal     = after.dry_internal - before.dry_internal;
        change.water_sensible   = after.water_sensible - before.water_sensible;
        change.phase_reference  = after.phase_reference - before.phase_reference;
        change.resolved_kinetic = after.resolved_kinetic - before.resolved_kinetic;
        change.gravitational    = after.gravitational - before.gravitational;
        change.subgrid_tke      = after.subgrid_tke - before.subgrid_tke;
        change.total_water_mass = after.total_water_mass - before.total_water_mass;
        change.total            = after.total - before.total;
        return change;
    };
    candidate_report.energy.before           = old_energy;
    candidate_report.energy.after            = new_energy;
    candidate_report.energy.committed_change = energy_change(new_energy, old_energy);
    const auto&  committed                   = candidate_report.energy.committed_change;
    const double reservoir_sum = committed.dry_internal + committed.water_sensible + committed.phase_reference +
                                 committed.resolved_kinetic + committed.gravitational + committed.subgrid_tke;
    candidate_report.energy.reservoir_sum_closure_residual_j = committed.total - reservoir_sum;

    auto interface_source = nested.interface_source;
    accumulate_interface_source(interface_source, candidate_report);
    if (validator)
        validator(candidate, candidate_report, validator_context);
    swap(nested.child.fields, candidate);
    for (std::size_t index = 0; index < private_candidate_count; ++index)
        swap(*private_bindings.child[index].field, nested.boundary_scratch.microphysics_private_candidate[index]);
    if (theta_residual_candidate_active)
        swap(live_theta_residual, nested.boundary_scratch.theta_storage_residual_candidate);
    nested.interface_source            = std::move(interface_source);
    nested.parent_boundary_state_ready = true;
    report                             = candidate_report;
}
} // namespace

void apply_nested_child_physical_relaxation(SimulationStorageState&                      storage,
                                            const PrognosticFieldStorage&                previous_parent_fields,
                                            const tmv::domain::NestedRelaxationConfigV1& config,
                                            double                                       child_step_s,
                                            NestedChildRelaxationReport&                 report,
                                            const NestedParentMicrophysicsSnapshotV1*    previous_parent_microphysics)
{
    apply_nested_child_physical_relaxation_implementation(
        storage, previous_parent_fields, config, child_step_s, report, previous_parent_microphysics, nullptr, nullptr);
}
