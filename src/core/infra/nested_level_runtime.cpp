#include "microphysics/grid_geometry.hpp"
#include "core/infra/nested_execution.hpp"

#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "dynamics/factory.hpp"
#include "microphysics/runtime.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "radar/factory.hpp"
#include "radiation/runtime.hpp"
#include "terrain/factory.hpp"
#include "turbulence/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

[[noreturn]] void reject_child_runtime(const std::string& reason)
{
    throw std::invalid_argument(
        "invalid nested child level runtime: " + reason);
}

bool same_radar_product_selection(
    const RadarProductSelection& left,
    const RadarProductSelection& right) noexcept
{
    return left.reflectivity == right.reflectivity
        && left.radial_velocity == right.radial_velocity
        && left.zdr == right.zdr && left.kdp == right.kdp
        && left.rhohv == right.rhohv
        && left.reflectivity_tier == right.reflectivity_tier
        && left.velocity_sampling == right.velocity_sampling
        && left.zdr_tier == right.zdr_tier
        && left.radar_x == right.radar_x
        && left.radar_y == right.radar_y
        && left.radar_z == right.radar_z
        && left.apply_scatterer_correction
               == right.apply_scatterer_correction;
}

std::int64_t checked_add_earth_cells(
    std::int64_t left, std::int64_t right)
{
    if ((right > 0 &&
         left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 &&
         left < std::numeric_limits<std::int64_t>::min() - right))
        reject_child_runtime("nested stochastic Earth origin overflows");
    return left + right;
}

std::int64_t checked_scale_earth_cells(
    std::int64_t value, int positive_ratio)
{
    if (positive_ratio <= 0)
        reject_child_runtime("nested stochastic refinement ratio is invalid");
    const auto ratio = static_cast<std::int64_t>(positive_ratio);
    if (value > std::numeric_limits<std::int64_t>::max() / ratio ||
        value < std::numeric_limits<std::int64_t>::min() / ratio)
        reject_child_runtime("nested stochastic Earth origin overflows");
    return value * ratio;
}

template <typename Scheme>
std::string require_scheme_name(
    const std::unique_ptr<Scheme>& scheme, const char* owner)
{
    if (!scheme)
        reject_child_runtime(std::string(owner) + " is not initialized");
    return scheme->name();
}

std::string require_microphysics_name(
    const tmv::MicrophysicsRuntimeState& runtime)
{
    if (!runtime.scheme)
        reject_child_runtime("microphysics is not initialized");
    return runtime.scheme->get_scheme_name();
}

bool field_matches_child(
    const Field3D& field, const AtmosphericLevelStorage& child) noexcept
{
    return field.size_r() == child.grid.nr &&
           field.size_th() == child.grid.nth &&
           field.size_z() == child.grid.nz;
}

bool same_chaos_config(
    const tmv::chaos::ChaosConfig& left,
    const tmv::chaos::ChaosConfig& right) noexcept
{
    return left.scheme_id == right.scheme_id && left.seed == right.seed &&
           left.member_id == right.member_id &&
           left.dt_chaos == right.dt_chaos && left.tau_t == right.tau_t &&
           left.Lx == right.Lx && left.Ly == right.Ly &&
           left.filter_id == right.filter_id && left.xi_max == right.xi_max &&
           left.taper_id == right.taper_id &&
           left.taper_z1 == right.taper_z1 &&
           left.taper_z2 == right.taper_z2 &&
           left.apply_to_ic == right.apply_to_ic &&
           left.apply_to_tendencies == right.apply_to_tendencies &&
           left.sigma_ic == right.sigma_ic &&
           left.alpha_tend == right.alpha_tend;
}

struct RewoundChaosPhase
{
    std::uint64_t epoch = 0;
    double        elapsed_s    = 0.0;
    double        cadence_dt_s = 0.0;
};

double nested_clock_roundoff_s(const SimulationStorageState& storage, double child_time_s)
{
    return std::max(1.0e-12 * std::max(1.0, storage.runtime.chaos.config.dt_chaos),
                    4 * std::numeric_limits<double>::epsilon() *
                        std::max(std::abs(storage.time_s), std::abs(child_time_s)));
}

RewoundChaosPhase rewind_chaos_phase(const tmv::chaos::HorizontalStochasticStatePhase& current,
                                     double                                            current_elapsed_s,
                                     double                                            interval_s,
                                     double                                            rewind_s)
{
    if (!std::isfinite(interval_s) || interval_s <= 0.0 || !std::isfinite(current_elapsed_s) ||
        current_elapsed_s < 0.0 || current_elapsed_s >= interval_s + 1.0e-12 || !std::isfinite(rewind_s) ||
        rewind_s < 0.0)
        reject_child_runtime("stochastic cadence history is invalid");
    const double tolerance   = 1.0e-12 * std::max({1.0, interval_s, rewind_s, std::abs(current_elapsed_s)});
    const double deficit = rewind_s - current_elapsed_s;
    const double raw_updates = deficit <= tolerance
        ? 0.0 : std::ceil((deficit - tolerance) / interval_s);
    if (!std::isfinite(raw_updates) || raw_updates < 0.0 ||
        raw_updates > static_cast<double>(current.current_epoch))
        reject_child_runtime(
            "parent stochastic history does not reach the child clock");
    const auto updates = static_cast<std::uint64_t>(raw_updates);
    double elapsed = current_elapsed_s - rewind_s +
        static_cast<double>(updates) * interval_s;
    if (std::abs(elapsed) <= tolerance)
        elapsed = 0.0;
    if (std::abs(elapsed - interval_s) <= tolerance)
    {
        elapsed = 0.0;
        if (updates == 0)
            reject_child_runtime("stochastic rewind normalization underflowed");
    }
    if (elapsed < 0.0 || elapsed >= interval_s || !std::isfinite(elapsed))
        reject_child_runtime("stochastic rewind produced an invalid phase");
    const std::uint64_t epoch = current.current_epoch - updates;
    return {epoch, elapsed, epoch == 0 ? 0.0 : interval_s};
}

std::int64_t nested_chaos_origin(
    std::int64_t parent_origin,
    int ratio,
    int parent_physical_begin,
    int child_origin_parent_cell,
    int child_physical_begin)
{
    const auto parent_scaled =
        checked_scale_earth_cells(parent_origin, ratio);
    const auto footprint_offset =
        static_cast<std::int64_t>(parent_physical_begin) +
        static_cast<std::int64_t>(child_origin_parent_cell);
    const auto footprint_scaled =
        checked_scale_earth_cells(footprint_offset, ratio);
    const auto physical_origin =
        checked_add_earth_cells(parent_scaled, footprint_scaled);
    return checked_add_earth_cells(
        physical_origin, -static_cast<std::int64_t>(child_physical_begin));
}

void require_or_initialize_nested_chaos_phase(
    const SimulationStorageState& storage,
    const tmv::chaos::ChaosRuntimeState& child_runtime,
    tmv::chaos::ChaosRuntimeState* initialization_target)
{
    const auto& parent_runtime = storage.runtime.chaos;
    const auto& nested = storage.nested_execution;
    const auto scheme_name = parent_runtime.config.scheme_id;
    if (scheme_name != "boundary_layer" && scheme_name != "full_stochastic")
        return;
    if (!parent_runtime.scheme || !child_runtime.scheme ||
        !same_chaos_config(parent_runtime.config, child_runtime.config))
        reject_child_runtime(
            "evolving stochastic parent/child configuration differs");

    tmv::chaos::HorizontalStochasticStatePhase parent_phase;
    std::string error;
    if (!parent_runtime.scheme->horizontal_stochastic_state_phase(
            parent_phase, error))
        reject_child_runtime("parent stochastic phase is invalid: " + error);
    const double rewind_s = storage.time_s - nested.child.time_s;
    const RewoundChaosPhase expected = rewind_chaos_phase(
        parent_phase, parent_runtime.noise_elapsed_s,
        parent_runtime.config.dt_chaos, rewind_s);
    const auto parent_physical = physical_scalar_domain(
        storage.grid.nr, storage.grid.nth, storage.grid.nz,
        StaggerType::CGrid);
    const auto child_physical = physical_scalar_domain(
        nested.child.grid.nr, nested.child.grid.nth, nested.child.grid.nz,
        StaggerType::CGrid);
    const auto& relationship = nested.relationship;
    const std::int64_t child_origin_i = nested_chaos_origin(
        parent_phase.earth_origin_i_cells, relationship.parent_ratio_x,
        parent_physical.i_begin,
        relationship.child_origin_parent_cell_i, child_physical.i_begin);
    const std::int64_t child_origin_j = nested_chaos_origin(
        parent_phase.earth_origin_j_cells, relationship.parent_ratio_y,
        parent_physical.j_begin,
        relationship.child_origin_parent_cell_j, child_physical.j_begin);

    if (initialization_target != nullptr)
    {
        if (initialization_target != &child_runtime)
            reject_child_runtime(
                "stochastic initialization target does not own the child phase");
        if (!initialization_target->scheme->initialize_nested_horizontal_stochastic_state(
                *parent_runtime.scheme,
                child_origin_i,
                child_origin_j,
                nested.child.grid.nr,
                nested.child.grid.nth,
                nested.child.grid.nz,
                expected.epoch,
                expected.cadence_dt_s,
                initialization_target->config,
                nested.child.process_grid_metrics,
                error))
            reject_child_runtime("child stochastic reconstruction failed: " + error);
        initialization_target->noise_elapsed_s = expected.elapsed_s;
    }

    tmv::chaos::HorizontalStochasticStatePhase child_phase;
    if (!child_runtime.scheme->horizontal_stochastic_state_phase(child_phase, error) ||
        child_phase.current_epoch != expected.epoch || child_phase.cadence_dt_s != expected.cadence_dt_s ||
        !std::isfinite(child_runtime.noise_elapsed_s) ||
        std::abs(child_runtime.noise_elapsed_s - expected.elapsed_s) >
            nested_clock_roundoff_s(storage, nested.child.time_s) ||
        child_phase.earth_origin_i_cells != child_origin_i || child_phase.earth_origin_j_cells != child_origin_j ||
        child_phase.initialized != parent_phase.initialized ||
        (child_phase.initialized &&
         (child_phase.nr != nested.child.grid.nr || child_phase.nth != nested.child.grid.nth ||
          child_phase.nz != nested.child.grid.nz)))
        reject_child_runtime("child stochastic Earth window or temporal phase is not canonical");
}

} // namespace

void synchronize_nested_child_chaos_clock(SimulationStorageState& storage, double child_time_s)
{
    const auto& parent = storage.runtime.chaos;
    auto&       child  = storage.nested_execution.child.runtime.chaos;
    if (parent.config.scheme_id != "boundary_layer" && parent.config.scheme_id != "full_stochastic")
    {
        return;
    }
    tmv::chaos::HorizontalStochasticStatePhase parent_phase, child_phase;
    std::string                                error;
    if (!parent.scheme || !child.scheme || !parent.scheme->horizontal_stochastic_state_phase(parent_phase, error) ||
        !child.scheme->horizontal_stochastic_state_phase(child_phase, error))
    {
        reject_child_runtime("stochastic clock synchronization requires valid parent and child phases");
    }
    const auto expected =
        rewind_chaos_phase(parent_phase, parent.noise_elapsed_s, parent.config.dt_chaos, storage.time_s - child_time_s);
    if (child_phase.current_epoch != expected.epoch || child_phase.cadence_dt_s != expected.cadence_dt_s ||
        !std::isfinite(child.noise_elapsed_s) ||
        std::abs(child.noise_elapsed_s - expected.elapsed_s) > nested_clock_roundoff_s(storage, child_time_s))
    {
        reject_child_runtime("child stochastic clock differs beyond endpoint arithmetic roundoff");
    }
    child.noise_elapsed_s = expected.elapsed_s;
}

void require_nested_child_level_runtime_ready(const SimulationStorageState& storage)
{
    require_valid_nested_execution_relationship(storage);
    const auto& nested = storage.nested_execution;
    if (!nested.parent_boundary_state_ready)
        reject_child_runtime("complete child boundary state is required");

    const auto& parent    = static_cast<const AtmosphericLevelRuntimeState&>(storage.runtime);
    const auto& child = nested.child;
    const auto& runtime = child.runtime;
    const auto same_name = [](const auto& parent_scheme,
                              const auto& child_scheme,
                              const char* owner) {
        const std::string parent_name =
            require_scheme_name(parent_scheme, owner);
        const std::string child_name =
            require_scheme_name(child_scheme, owner);
        if (child_name != parent_name)
            reject_child_runtime(
                std::string(owner) + " selection differs from the parent");
    };

    same_name(parent.radiation.scheme, runtime.radiation.scheme,
              "radiation");
    same_name(parent.turbulence.scheme, runtime.turbulence.scheme,
              "turbulence");
    same_name(parent.boundary_layer.scheme,
              runtime.boundary_layer.scheme, "boundary layer");
    same_name(parent.chaos.scheme, runtime.chaos.scheme, "chaos");
    same_name(parent.terrain.scheme, runtime.terrain.scheme, "terrain");
    same_name(parent.advection.scheme, runtime.advection.scheme,
              "advection");
    same_name(parent.diffusion.scheme, runtime.diffusion.scheme,
              "diffusion");
    same_name(parent.time_stepping.scheme, runtime.time_stepping.scheme,
              "time stepping");
    if (!parent.dynamics.scheme || !runtime.dynamics.scheme ||
        parent.dynamics.scheme->get_scheme_name() !=
            runtime.dynamics.scheme->get_scheme_name())
        reject_child_runtime("dynamics selection differs from the parent");
    if (require_microphysics_name(parent.microphysics) !=
        require_microphysics_name(runtime.microphysics))
        reject_child_runtime("microphysics selection differs from the parent");
    if (static_cast<bool>(parent.radar.selected_scheme) !=
        static_cast<bool>(runtime.radar.selected_scheme))
        reject_child_runtime("radar selection differs from the parent");
    if (!same_radar_product_selection(
            parent.radar.product_selection,
            runtime.radar.product_selection))
        reject_child_runtime("radar product selection differs from the parent");
    if (parent.radar.selected_scheme &&
        (runtime.radar.selected_scheme->name() !=
             parent.radar.selected_scheme->name() ||
         runtime.radar.selected_config.scheme_id !=
             parent.radar.selected_config.scheme_id ||
         runtime.radar.nr != child.grid.nr ||
         runtime.radar.nth != child.grid.nth ||
         runtime.radar.nz != child.grid.nz))
        reject_child_runtime(
            "radar selection or initialized grid differs from the child");

    const bool active_terrain = nested.active_terrain.ready;
    if (active_terrain)
    {
        const auto& terrain_identity =
            nested.active_terrain.geometry.config.scheme_id;
        const auto& geometry = nested.active_terrain.geometry.child_storage;
        if (runtime.terrain.config.scheme_id != terrain_identity
            || parent.terrain.config.scheme_id != terrain_identity
            || runtime.terrain.metrics.J.values
                != geometry.metrics.J.values
            || runtime.terrain.topography.h != geometry.topography.h
            || !child.process_grid_metrics.terrain_metrics_active)
            reject_child_runtime(
                "mapped child terrain runtime differs from its published geometry");
    }
    else if (runtime.terrain.config.scheme_id != "none"
             || parent.terrain.config.scheme_id != "none")
        reject_child_runtime(
            "active parent terrain requires a published mapped child owner");
    if (runtime.radiation.config.scheme_id !=
            parent.radiation.config.scheme_id ||
        runtime.turbulence.config.scheme_id !=
            parent.turbulence.config.scheme_id ||
        runtime.boundary_layer.config.scheme_id !=
            parent.boundary_layer.config.scheme_id ||
        !same_chaos_config(runtime.chaos.config, parent.chaos.config) ||
        runtime.advection.config.scheme_id !=
            parent.advection.config.scheme_id ||
        runtime.diffusion.config.scheme_id !=
            parent.diffusion.config.scheme_id ||
        runtime.time_stepping.config.scheme_id !=
            parent.time_stepping.config.scheme_id)
        reject_child_runtime("a child module configuration is not canonical");

    require_or_initialize_nested_chaos_phase(
        storage, runtime.chaos, nullptr);

    if (!field_matches_child(child.tendencies.dtheta_dt_rad, child) ||
        !field_matches_child(child.tendencies.dtheta_dt_pbl, child) ||
        !field_matches_child(child.tendencies.dqv_dt_pbl, child) ||
        !field_matches_child(child.tendencies.du_dt_pbl, child) ||
        !field_matches_child(child.tendencies.dv_dt_pbl, child) ||
        !field_matches_child(child.tendencies.dtke_dt_pbl, child))
        reject_child_runtime(
            "initial radiation/PBL tendencies do not match the child grid");

    const std::size_t columns =
        static_cast<std::size_t>(child.grid.nr) *
        static_cast<std::size_t>(child.grid.nth);
    const auto surface_shape = [columns](const std::vector<float>& values) {
        return values.size() == columns;
    };
    if (!surface_shape(child.surface.ustar) ||
        !surface_shape(child.surface.tau_u) ||
        !surface_shape(child.surface.tau_v) ||
        !surface_shape(child.surface.sensible_heat_flux) ||
        !surface_shape(child.surface.latent_heat_flux) ||
        !surface_shape(child.surface.moisture_flux) ||
        !surface_shape(child.surface.pbl_height))
        reject_child_runtime(
            "initial surface/PBL diagnostics do not match the child grid");

    if (runtime.boundary_layer.scheme->name() != "none" &&
        (!field_matches_child(runtime.boundary_layer.held_u_force_n, child) ||
         !field_matches_child(runtime.boundary_layer.held_v_force_n, child) ||
         !field_matches_child(
             runtime.boundary_layer.held_theta_inventory_rate_kgk_s,
             child) ||
         !field_matches_child(
             runtime.boundary_layer.held_vapor_mass_rate_kg_s, child)))
        reject_child_runtime(
            "active PBL held sources do not match the child grid");
    if (!std::isfinite(runtime.radiation.last_update_time_s) ||
        runtime.radiation.last_update_time_s > child.time_s ||
        !std::isfinite(runtime.boundary_layer.last_update_time_s) ||
        runtime.boundary_layer.last_update_time_s > child.time_s)
        reject_child_runtime(
            "radiation/PBL samples are ahead of the child clock");

    auto parent_private = parent.microphysics.scheme
        ->prognostic_scalars_for_transport();
    auto child_private = runtime.microphysics.scheme
        ->prognostic_scalars_for_transport();
    const auto& last_private = nested.interface_source.last.microphysics_private;
    const auto& cumulative_private =
        nested.interface_source.cumulative.microphysics_private;
    if (parent_private.size() != child_private.size() ||
        last_private.scalar_count != parent_private.size() ||
        cumulative_private.scalar_count != parent_private.size())
        reject_child_runtime(
            "private microphysics interface layout is incomplete");
    for (std::size_t index = 0; index < child_private.size(); ++index)
    {
        const auto& parent_scalar = parent_private[index];
        const auto& child_scalar = child_private[index];
        const auto name_end = std::find(
            last_private.name[index].begin(),
            last_private.name[index].end(), '\0');
        const std::string durable_name(
            last_private.name[index].begin(), name_end);
        if (!parent_scalar.name || !child_scalar.name ||
            !parent_scalar.field || !child_scalar.field ||
            durable_name != parent_scalar.name ||
            durable_name != child_scalar.name ||
            cumulative_private.name[index] != last_private.name[index] ||
            last_private.storage_semantic[index] !=
                static_cast<std::uint8_t>(parent_scalar.storage) ||
            cumulative_private.storage_semantic[index] !=
                last_private.storage_semantic[index] ||
            child_scalar.storage != parent_scalar.storage ||
            last_private.positive_definite[index] !=
                parent_scalar.positive_definite ||
            cumulative_private.positive_definite[index] !=
                last_private.positive_definite[index] ||
            child_scalar.positive_definite !=
                parent_scalar.positive_definite ||
            !field_matches_child(*child_scalar.field, child))
            reject_child_runtime(
                "private microphysics interface identity differs");
        for (std::size_t cell = 0; cell < child_scalar.field->size(); ++cell)
            if (!std::isfinite(child_scalar.field->data()[cell]) ||
                (child_scalar.positive_definite &&
                 child_scalar.field->data()[cell] < 0.0f))
                reject_child_runtime(
                    "private microphysics child state is invalid");
    }
}

void initialize_nested_child_level_runtime(
    SimulationStorageState& storage)
{
    require_valid_nested_execution_relationship(storage);
    auto& nested = storage.nested_execution;
    if (!nested.parent_boundary_state_ready)
        reject_child_runtime(
            "complete child boundary state must precede module initialization");

    const auto& parent =
        static_cast<const AtmosphericLevelRuntimeState&>(storage.runtime);
    AtmosphericLevelRuntimeState candidate;

    tmv::initialize_radiation(
        candidate.radiation,
        require_scheme_name(parent.radiation.scheme, "radiation"),
        parent.radiation.config);
    tmv::initialize_turbulence(
        candidate.turbulence,
        require_scheme_name(parent.turbulence.scheme, "turbulence"),
        parent.turbulence.config);
    tmv::initialize_boundary_layer(
        candidate.boundary_layer,
        require_scheme_name(parent.boundary_layer.scheme, "boundary layer"),
        parent.boundary_layer.config,
        parent.boundary_layer.surface_config);
    candidate.microphysics.scheme = tmv::create_microphysics_scheme(
        require_microphysics_name(parent.microphysics),
        storage.configuration.microphysics_config);
    if (!candidate.microphysics.scheme)
        reject_child_runtime("microphysics factory returned no scheme");
    tmv::prepare_microphysics_prognostics(
        make_simulation_state_view(nested.child, storage.configuration), *candidate.microphysics.scheme);
    tmv::initialize_advection(
        candidate.advection,
        require_scheme_name(parent.advection.scheme, "advection"),
        parent.advection.config);
    tmv::initialize_diffusion(
        candidate.diffusion,
        require_scheme_name(parent.diffusion.scheme, "diffusion"),
        parent.diffusion.config);
    tmv::initialize_time_stepping(
        candidate.time_stepping,
        require_scheme_name(parent.time_stepping.scheme, "time stepping"),
        parent.time_stepping.config);

    if (!parent.dynamics.scheme)
        reject_child_runtime("dynamics is not initialized");
    const std::string dynamics_name =
        parent.dynamics.scheme->get_scheme_name();

    const bool active_terrain = nested.active_terrain.ready;
    const std::string active_terrain_identity = active_terrain
        ? nested.active_terrain.geometry.config.scheme_id
        : std::string{};
    if (!parent.terrain.scheme
        || (active_terrain
            ? (parent.terrain.config.scheme_id != active_terrain_identity
               || parent.terrain.scheme->name() != active_terrain_identity)
            : (parent.terrain.config.scheme_id != "none"
               || parent.terrain.scheme->name() != "none")))
        reject_child_runtime(
            "parent terrain selection differs from the prepared child geometry");
    candidate.terrain.config = parent.terrain.config;
    candidate.terrain.scheme = tmv::create_terrain_scheme(
        parent.terrain.scheme->name());
    if (!candidate.terrain.scheme)
        reject_child_runtime("terrain factory returned no child scheme");
    candidate.terrain.scheme->initialize(candidate.terrain.config);
    if (active_terrain)
    {
        candidate.terrain.topography =
            nested.active_terrain.geometry.child_storage.topography;
        candidate.terrain.metrics =
            nested.active_terrain.geometry.child_storage.metrics;
        candidate.terrain.earth_fixed_placement =
            nested.child.runtime.terrain.earth_fixed_placement;
    }

    candidate.radar.product_selection = parent.radar.product_selection;
    if (parent.radar.selected_scheme)
    {
        candidate.radar.selected_config = parent.radar.selected_config;
        candidate.radar.selected_scheme =
            tmv::create_radar_scheme(parent.radar.selected_scheme->name());
        if (!candidate.radar.selected_scheme)
            reject_child_runtime("radar factory returned no child scheme");
        candidate.radar.selected_scheme->initialize(
            candidate.radar.selected_config,
            nested.child.grid.nr, nested.child.grid.nth,
            nested.child.grid.nz);
        candidate.radar.nr = nested.child.grid.nr;
        candidate.radar.nth = nested.child.grid.nth;
        candidate.radar.nz = nested.child.grid.nz;
    }

    const tmv::chaos::ChaosConfig chaos_config = parent.chaos.config;
    auto& child = nested.child;
    AtmosphericLevelRuntimeState prior_runtime = std::move(child.runtime);
    PhysicsTendencyFieldStorage prior_tendencies =
        std::move(child.tendencies);
    SurfaceDiagnosticFieldStorage prior_surface = std::move(child.surface);
    child.runtime = std::move(candidate);
    child.tendencies = {};
    child.surface = {};

    try
    {
        prepare_nested_child_process_grid_metrics(nested);
        {
            ScopedNestedChildResidency residency(storage);
            const auto child_state = nested_child_execution_state(storage);
            child.runtime.dynamics.scheme =
                tmv::create_dynamics_scheme(dynamics_name, child_state);
            if (!child.runtime.dynamics.scheme)
                reject_child_runtime(
                    "dynamics factory returned no child scheme");
            tmv::chaos::initialize_chaos(child_state, chaos_config);
        }
        require_or_initialize_nested_chaos_phase(
            storage, child.runtime.chaos, &child.runtime.chaos);
        {
            ScopedNestedChildResidency residency(storage);
            const auto child_state = nested_child_execution_state(storage);
            tmv::step_boundary_layer(child_state, child_state.time_s);
            tmv::step_radiation(child_state, child_state.time_s);
        }
        const auto prior_interface_source = nested.interface_source;
        try
        {
            auto private_scalars = child.runtime.microphysics.scheme
                ->prognostic_scalars_for_transport();
            for (NestedChildRelaxationReport* report : {
                     &nested.interface_source.last,
                     &nested.interface_source.cumulative})
            {
                report->microphysics_private.scalar_count =
                    private_scalars.size();
                for (std::size_t index = 0;
                     index < private_scalars.size(); ++index)
                {
                    const auto& scalar = private_scalars[index];
                    if (!scalar.name || !scalar.field)
                        reject_child_runtime(
                            "private microphysics factory binding is invalid");
                    const std::size_t length =
                        std::char_traits<char>::length(scalar.name);
                    if (length == 0 || length >=
                        NestedPrivateScalarRelaxationReportV1::
                            maximum_name_bytes)
                        reject_child_runtime(
                            "private microphysics name exceeds the durable limit");
                    std::copy_n(
                        scalar.name, length,
                        report->microphysics_private.name[index].begin());
                    report->microphysics_private.storage_semantic[index] =
                        static_cast<std::uint8_t>(scalar.storage);
                    report->microphysics_private.positive_definite[index] =
                        scalar.positive_definite;
                }
            }
            require_nested_child_level_runtime_ready(storage);
        }
        catch (...)
        {
            nested.interface_source = prior_interface_source;
            throw;
        }
        nested.interface_source = prior_interface_source;
    }
    catch (...)
    {
        child.runtime = std::move(prior_runtime);
        child.tendencies = std::move(prior_tendencies);
        child.surface = std::move(prior_surface);
        prepare_nested_child_process_grid_metrics(nested);
        throw;
    }
}

void rehydrate_nested_child_transient_operators(
    SimulationStorageState& storage)
{
    require_valid_nested_execution_relationship(storage);
    auto& child = storage.nested_execution.child;
    if (!storage.runtime.dynamics.scheme)
        reject_child_runtime(
            "parent dynamics must be reconstructed before the child");
    const std::string dynamics_name =
        storage.runtime.dynamics.scheme->get_scheme_name();
    auto prior_scheme = std::move(child.runtime.dynamics.scheme);
    try
    {
        {
            ScopedNestedChildResidency residency(storage);
            const auto child_state = nested_child_execution_state(storage);
            child.runtime.dynamics.scheme =
                tmv::create_dynamics_scheme(dynamics_name, child_state);
            if (!child.runtime.dynamics.scheme)
                reject_child_runtime(
                    "dynamics factory returned no restored child scheme");
        }
        require_nested_child_level_runtime_ready(storage);
    }
    catch (...)
    {
        child.runtime.dynamics.scheme = std::move(prior_scheme);
        throw;
    }
}
