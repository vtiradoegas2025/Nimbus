#include "microphysics/grid_geometry.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/simulation_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tmv
{
void bind_microphysics_grid_geometry(const SimulationStateView& state,
                                     MicrophysicsScheme&        scheme,
                                     const TerrainMetrics3D*    destination)
{
    const auto& grid    = state.grid;
    const auto* terrain = destination ? destination : physical_grid_metrics::active_terrain_metrics(state);
    if (grid.nr <= 0 || grid.nth <= 0 || grid.nz <= 0 || grid.nz == std::numeric_limits<int>::max() ||
        state.fields.rho.size_r() != grid.nr || state.fields.rho.size_th() != grid.nth ||
        state.fields.rho.size_z() != grid.nz)
    {
        throw std::invalid_argument("microphysics geometry requires a complete density grid");
    }
    const std::size_t columns = state.fields.rho.size() / grid.nz;
    if (terrain &&
        (terrain->z_face.nr != grid.nr || terrain->z_face.nth != grid.nth || terrain->z_face.nz != grid.nz + 1 ||
         columns > terrain->z_face.values.max_size() / (std::size_t(grid.nz) + 1) ||
         terrain->z_face.size() != columns * (std::size_t(grid.nz) + 1)))
    {
        throw std::invalid_argument("microphysics geometry has incomplete terrain interfaces");
    }
    if (!terrain && !grid.geometry.z_faces.empty() && grid.geometry.z_faces.size() != std::size_t(grid.nz) + 1)
    {
        throw std::invalid_argument("microphysics geometry has incomplete flat interfaces");
    }
    const auto thickness = [&](int i, int j, int k)
    {
        return terrain ? terrain->z_face(i, j, k + 1) - terrain->z_face(i, j, k)
                       : physical_grid_metrics::layer_thickness(state, i, j, k);
    };
    const double uniform  = thickness(0, 0, 0);
    bool         variable = terrain != nullptr;
    if (!variable)
    {
        for (int k = 1; k < grid.nz; ++k)
        {
            const double height = grid.geometry.z_faces.empty() ? grid.dz * (k + 1) : grid.geometry.z_faces[k + 1];
            const double tolerance =
                8 * std::numeric_limits<double>::epsilon() * std::max({1.0, std::abs(height), std::abs(uniform)});
            variable = variable || std::abs(thickness(0, 0, k) - uniform) > tolerance;
        }
    }
    if (!variable)
    {
        scheme.set_vertical_spacing_m(uniform);
        return;
    }
    std::vector<double> layers(state.fields.rho.size());
    for (int i = 0; i < grid.nr; ++i)
    {
        for (int j = 0; j < grid.nth; ++j)
        {
            for (int k = 0; k < grid.nz; ++k)
            {
                layers[(std::size_t(i) * grid.nth + j) * grid.nz + k] = thickness(i, j, k);
            }
        }
    }
    scheme.set_vertical_spacing_m(grid.nr, grid.nth, grid.nz, std::move(layers));
}

void prepare_microphysics_prognostics(const SimulationStateView& state, MicrophysicsScheme& scheme)
{
    bool initialize = false;
    for (const auto& scalar : scheme.prognostic_scalars_for_transport())
    {
        if (!scalar.field)
        {
            throw std::invalid_argument("microphysics initialization has no private field owner");
        }
        initialize = initialize || scalar.field->size_r() != state.grid.nr ||
                     scalar.field->size_th() != state.grid.nth || scalar.field->size_z() != state.grid.nz;
    }
    const auto physical = physical_scalar_domain(state.grid);
    if (initialize)
    {
        bind_microphysics_grid_geometry(state, scheme);
        scheme.set_physical_cell_domain(
            state.grid.nr, state.grid.nth, state.grid.nz, {0, state.grid.nr, 0, state.grid.nth, 0, state.grid.nz});
    }
    try
    {
        scheme.prepare_prognostic_state(state.fields.rho,
                                        state.fields.qc,
                                        state.fields.qr,
                                        state.fields.qi,
                                        state.fields.qs,
                                        state.fields.qg,
                                        state.fields.qh);
    }
    catch (...)
    {
        if (initialize)
        {
            scheme.set_physical_cell_domain(state.grid.nr, state.grid.nth, state.grid.nz, physical);
        }
        throw;
    }
    if (initialize)
    {
        scheme.set_physical_cell_domain(state.grid.nr, state.grid.nth, state.grid.nz, physical);
    }
}
} // namespace tmv
