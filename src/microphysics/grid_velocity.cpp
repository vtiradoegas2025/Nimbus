#include "microphysics/grid_geometry.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/runtime/simulation_state.hpp"
#include "terrain/conservative_transport.hpp"

#include <algorithm>
#include <stdexcept>

namespace tmv
{
void bind_microphysics_velocity(const SimulationStateView& state, MicrophysicsScheme& scheme)
{
    const auto& grid = state.grid;
    const auto& wind = state.fields.w;
    if (grid.nr <= 0 || grid.nth <= 0 || grid.nz <= 0 || wind.size_r() != grid.nr || wind.size_th() != grid.nth ||
        wind.size_z() != grid.nz)
    {
        throw std::invalid_argument("microphysics velocity requires a complete owning grid");
    }
    std::vector<double> lower_boundary;
    if (grid.geometry.staggered)
    {
        if (const auto* metrics = physical_grid_metrics::active_terrain_metrics(state))
        {
            if (grid.nr < 3 || grid.nth < 3)
            {
                throw std::invalid_argument("mapped microphysics velocity requires physical columns");
            }
            const auto surface_velocity = physical_grid_metrics::surface_velocity_ms(state);
            lower_boundary.resize(wind.size() / grid.nz, 0.0);
            for (int i = 0; i < grid.nr; ++i)
            {
                for (int j = 0; j < grid.nth; ++j)
                {
                    lower_boundary[static_cast<std::size_t>(i) * grid.nth + j] =
                        terrain_transport::cgrid_ground_vertical_velocity(*metrics,
                                                                          state.fields.u,
                                                                          state.fields.v,
                                                                          std::max(i, 1),
                                                                          std::max(j, 1),
                                                                          surface_velocity.u_ms,
                                                                          surface_velocity.v_ms);
                }
            }
        }
    }
    scheme.set_vertical_velocity_field(wind, grid.geometry.staggered, std::move(lower_boundary));
}
} // namespace tmv
