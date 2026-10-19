#include "boundary_conditions/runtime.hpp"
#include "boundary_conditions/factory.hpp"

#include "compute/compute_backend.hpp"
#include "core/field/field_sanitization.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_cell_domain.hpp"

#include <cmath>
#include <stdexcept>

namespace tmv
{

std::unique_ptr<BoundaryConditionScheme>& bc_scheme =
    global_boundary_condition_runtime_state.execution_operator;

std::unique_ptr<BoundaryConditionScheme>& boundary_scheme_for(const SimulationStateView& state)
{
    if (!state.boundary_condition_runtime)
        throw std::invalid_argument("boundary execution requires its level runtime");
    auto& selected = state.boundary_condition_runtime->execution_operator;
    if (!selected) selected = create_boundary_condition_scheme(state);
    return selected;
}

void apply_boundary_conditions(const SimulationStateView& state)
{
    auto& bc_scheme = boundary_scheme_for(state);
    if (!bc_scheme)
    {
        throw std::runtime_error(
            "boundary conditions must be initialized before application");
    }
    bc_scheme->apply_full();
    enforce_primary_state_bounds(state, "boundary_conditions");
}

void apply_boundary_conditions()
{
    apply_boundary_conditions(global_simulation_state());
}

void rebuild_microphysics_boundary_conditions(const SimulationStateView& state)
{
    if (!state.microphysics_runtime || !state.microphysics_runtime->scheme)
    {
        return;
    }
    const auto  scalars = state.microphysics_runtime->scheme->prognostic_scalars_for_transport();
    const auto& grid    = state.grid;
    bool        active  = false;
    for (const auto& scalar : scalars)
    {
        if (!scalar.field)
        {
            throw std::invalid_argument("private scalar boundary has no field owner");
        }
        if (scalar.field->empty())
        {
            continue;
        }
        if (scalar.field->size_r() != grid.nr || scalar.field->size_th() != grid.nth ||
            scalar.field->size_z() != grid.nz ||
            (scalar.storage != MicrophysicsScalarStorage::DryAirMixingRatio &&
             scalar.storage != MicrophysicsScalarStorage::VolumeConcentration))
        {
            throw std::invalid_argument("private scalar boundary has an invalid shape or storage convention");
        }
        active = true;
    }
    if (!active)
    {
        return;
    }
    const auto physical = physical_scalar_domain(grid);
    if (grid.geometry.coord != CoordinateSystem::Cartesian || physical.empty() || !state.configuration ||
        state.fields.rho.size_r() != grid.nr || state.fields.rho.size_th() != grid.nth ||
        state.fields.rho.size_z() != grid.nz)
    {
        throw std::invalid_argument("private scalar boundary requires a complete Cartesian state");
    }
    using BC           = tmv::init::config::LateralBC;
    const auto& config = *state.configuration;
    const int   west   = config.lateral_bc_west == BC::Periodic ? grid.nr - 2 : 1;
    const int   east   = config.lateral_bc_east == BC::Periodic ? 1 : grid.nr - 2;
    const int   south  = config.lateral_bc_south == BC::Periodic ? grid.nth - 2 : 1;
    const int   north  = config.lateral_bc_north == BC::Periodic ? 1 : grid.nth - 2;
    const auto  close  = [&](int i, int j, int k, bool publish)
    {
        const int si = i == 0 ? west : (i == grid.nr - 1 ? east : i);
        const int sj = j == 0 ? south : (j == grid.nth - 1 ? north : j);
        const int sk = physical.k_begin == 0 ? k : (k == 0 ? 1 : (k == grid.nz - 1 ? grid.nz - 2 : k));
        for (const auto& scalar : scalars)
        {
            auto& field = *scalar.field;
            if (field.empty())
            {
                continue;
            }
            double value = field(si, sj, sk);
            if (scalar.storage == MicrophysicsScalarStorage::VolumeConcentration)
            {
                const double source_density = state.fields.rho(si, sj, sk);
                const double target_density = state.fields.rho(i, j, k);
                if (!std::isfinite(source_density) || source_density <= 0 || !std::isfinite(target_density) ||
                    target_density <= 0)
                {
                    throw std::invalid_argument("private concentration boundary requires positive finite density");
                }
                value = value / source_density * target_density;
            }
            const float stored = static_cast<float>(value);
            if (!std::isfinite(stored) || (scalar.positive_definite && stored < 0))
            {
                throw std::invalid_argument("private scalar boundary is not representable");
            }
            if (publish)
            {
                field(i, j, k) = stored;
            }
        }
    };
    // Every donor is physical, so validation and publication read identical state.
    for (bool publish : {false, true})
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                close(0, j, k, publish);
                close(grid.nr - 1, j, k, publish);
            }
        }
        for (int i = 1; i < grid.nr - 1; ++i)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                close(i, 0, k, publish);
                close(i, grid.nth - 1, k, publish);
            }
        }
        if (physical.k_begin != 0)
        {
            for (int i = 1; i < grid.nr - 1; ++i)
            {
                for (int j = 1; j < grid.nth - 1; ++j)
                {
                    close(i, j, 0, publish);
                    close(i, j, grid.nz - 1, publish);
                }
            }
        }
    }
}

void rebuild_boundary_conditions_after_domain_recycling(const SimulationStateView& state)
{
    auto& bc_scheme = boundary_scheme_for(state);
    if (!bc_scheme)
    {
        throw std::runtime_error("boundary conditions must be initialized before domain recycling");
    }
    // The CPU shift has already replaced every primary host field. Invalidate
    // the complete resident generation before any subsequent validation can
    // reject, so an exception can never leave a stale device copy eligible
    // for reuse.
    if (state.compute_runtime != nullptr &&
        state.compute_runtime->backend != nullptr)
        state.compute_runtime->backend->invalidate_all_resident_fields();
    bc_scheme->rebuild_after_domain_recycling();
    rebuild_microphysics_boundary_conditions(state);
    enforce_primary_state_bounds(state, "domain_recycling_boundary_rebuild");
}

void rebuild_boundary_conditions_after_domain_recycling()
{
    rebuild_boundary_conditions_after_domain_recycling(
        global_simulation_state());
}

} // namespace tmv
