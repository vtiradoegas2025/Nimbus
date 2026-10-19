#pragma once

#include "core/runtime/simulation_state.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file physical_grid_metrics.hpp
 * @brief Exact physical heights, thicknesses, and volumes for runtime physics.
 *
 * Column physics and integral diagnostics must not reconstruct a flat column
 * from grid.dz when terrain-following coordinates are active.  These helpers
 * provide one fail-closed geometry boundary for both flat and terrain grids.
 */
namespace tmv::physical_grid_metrics
{

struct ColumnGeometry
{
    std::vector<double> z_center;
    std::vector<double> z_interface;
    std::vector<double> layer_thickness;
};

inline bool terrain_is_active(const SimulationStateView& state)
{
    if (state.terrain_runtime == nullptr)
        return false;

    const auto& runtime = *state.terrain_runtime;
    if (!runtime.config.scheme_id.empty())
        return runtime.config.scheme_id != "none";
    return runtime.scheme != nullptr && runtime.scheme->name() != "none";
}

inline const TerrainMetrics3D* active_terrain_metrics(
    const SimulationStateView& state)
{
    if (!terrain_is_active(state))
        return nullptr;
    return &state.terrain_runtime->metrics;
}

inline domain::HorizontalVelocity surface_velocity_ms(const SimulationStateView& state)
{
    // A translated coordinate window does not translate Earth-fixed ground.
    if (!terrain_is_active(state) || state.domain_frame == nullptr
        || state.terrain_runtime->config.horizontal_placement_id == "earth_fixed")
        return {};
    return state.domain_frame->velocity_ms;
}

inline void validate(const SimulationStateView& state)
{
    const auto& grid = state.grid;
    if (grid.nr <= 0 || grid.nth <= 0 || grid.nz <= 0
        || !std::isfinite(grid.dr) || grid.dr <= 0.0
        || !std::isfinite(grid.dtheta) || grid.dtheta <= 0.0
        || !std::isfinite(grid.dz) || grid.dz <= 0.0)
    {
        throw std::invalid_argument(
            "physical grid geometry requires a non-empty grid and positive finite spacing");
    }

    const TerrainMetrics3D* metrics = active_terrain_metrics(state);
    if (metrics == nullptr)
    {
        const auto& z = grid.geometry.z;
        const auto& z_face = grid.geometry.z_faces;
        if (z.empty() != z_face.empty()
            || (!z.empty() && z.size() != static_cast<std::size_t>(grid.nz))
            || (!z_face.empty()
                && z_face.size() != static_cast<std::size_t>(grid.nz + 1)))
        {
            throw std::invalid_argument(
                "flat physical grid geometry is only partially initialized");
        }
        for (int k = 0; k < grid.nz; ++k)
        {
            const double center = z.empty()
                ? (static_cast<double>(k) + 0.5) * grid.dz
                : z[static_cast<std::size_t>(k)];
            const double lower = z_face.empty()
                ? static_cast<double>(k) * grid.dz
                : z_face[static_cast<std::size_t>(k)];
            const double upper = z_face.empty()
                ? static_cast<double>(k + 1) * grid.dz
                : z_face[static_cast<std::size_t>(k + 1)];
            if (!std::isfinite(center) || !std::isfinite(lower)
                || !std::isfinite(upper) || !(upper > lower)
                || !(center > lower) || !(center < upper))
            {
                throw std::invalid_argument(
                    "flat physical interface geometry is malformed");
            }
        }
        return;
    }

    if (metrics->z.size_r() != grid.nr
        || metrics->z.size_th() != grid.nth
        || metrics->z.size_z() != grid.nz
        || metrics->z_face.size_r() != grid.nr
        || metrics->z_face.size_th() != grid.nth
        || metrics->z_face.size_z() != grid.nz + 1)
    {
        throw std::invalid_argument(
            "active terrain physical geometry does not match the runtime grid");
    }
    for (int i = 0; i < grid.nr; ++i)
        for (int j = 0; j < grid.nth; ++j)
            for (int k = 0; k < grid.nz; ++k)
            {
                const double center = metrics->z(i, j, k);
                const double lower = metrics->z_face(i, j, k);
                const double upper = metrics->z_face(i, j, k + 1);
                if (!std::isfinite(center) || !std::isfinite(lower)
                    || !std::isfinite(upper) || !(upper > lower)
                    || !(center > lower) || !(center < upper))
                {
                    throw std::invalid_argument(
                        "active terrain physical interface geometry is malformed");
                }
            }
}

inline double scalar_height(const SimulationStateView& state,
                            int i, int j, int k)
{
    const auto& grid = state.grid;
    if (i < 0 || i >= grid.nr || j < 0 || j >= grid.nth
        || k < 0 || k >= grid.nz)
        throw std::out_of_range("physical scalar-height index is outside the grid");
    if (const TerrainMetrics3D* metrics = active_terrain_metrics(state))
        return metrics->z(i, j, k);
    return grid.geometry.z.empty()
        ? (static_cast<double>(k) + 0.5) * grid.dz
        : grid.geometry.z[static_cast<std::size_t>(k)];
}

inline double interface_height(const SimulationStateView& state,
                               int i, int j, int k_face)
{
    const auto& grid = state.grid;
    if (i < 0 || i >= grid.nr || j < 0 || j >= grid.nth
        || k_face < 0 || k_face > grid.nz)
        throw std::out_of_range(
            "physical interface-height index is outside the grid: i="
            + std::to_string(i) + ", j=" + std::to_string(j)
            + ", k_face=" + std::to_string(k_face)
            + ", shape=" + std::to_string(grid.nr) + "x"
            + std::to_string(grid.nth) + "x"
            + std::to_string(grid.nz));
    if (const TerrainMetrics3D* metrics = active_terrain_metrics(state))
        return metrics->z_face(i, j, k_face);
    return grid.geometry.z_faces.empty()
        ? static_cast<double>(k_face) * grid.dz
        : grid.geometry.z_faces[static_cast<std::size_t>(k_face)];
}

inline double layer_thickness(const SimulationStateView& state,
                              int i, int j, int k)
{
    const double thickness = interface_height(state, i, j, k + 1)
        - interface_height(state, i, j, k);
    if (!std::isfinite(thickness) || thickness <= 0.0)
        throw std::invalid_argument(
            "physical layer thickness must be positive and finite");
    return thickness;
}

inline double cell_volume(const SimulationStateView& state,
                          int i, int j, int k)
{
    const double volume = state.grid.dr * state.grid.dtheta
        * layer_thickness(state, i, j, k);
    if (!std::isfinite(volume) || volume <= 0.0)
        throw std::invalid_argument(
            "physical cell volume must be positive and finite");
    return volume;
}

/**
 * Allocation-free physical control-volume and mass access for hot loops.
 *
 * Call validate(state) once before constructing this view, and validate the
 * live density field once at the owning process boundary.  The point methods
 * deliberately perform no repeated shape, index, or finiteness checks.  A
 * C-grid face owns exactly half of each adjacent scalar control volume, so
 * density and volume are multiplied before the two half-masses are added.
 */
class ControlVolumeMassView
{
public:
    explicit ControlVolumeMassView(const SimulationStateView& state) noexcept
        : grid_(state.grid),
          terrain_(active_terrain_metrics(state)),
          horizontal_area_(state.grid.dr * state.grid.dtheta)
    {
    }

    [[nodiscard]] double scalar_volume_m3(int i, int j, int k) const noexcept
    {
        const double thickness = terrain_ != nullptr
            ? terrain_->z_face(i, j, k + 1) - terrain_->z_face(i, j, k)
            : (grid_.geometry.z_faces.empty()
                ? grid_.dz
                : grid_.geometry.z_faces[static_cast<std::size_t>(k + 1)]
                    - grid_.geometry.z_faces[static_cast<std::size_t>(k)]);
        return horizontal_area_ * thickness;
    }

    [[nodiscard]] double scalar_mass_kg(const Field3D& density,
                                        int i, int j, int k) const noexcept
    {
        return static_cast<double>(density(i, j, k))
            * scalar_volume_m3(i, j, k);
    }

    [[nodiscard]] double u_dual_mass_kg(const Field3D& density,
                                        int i, int j, int k,
                                        int right) const noexcept
    {
        return 0.5 * (scalar_mass_kg(density, i, j, k)
                    + scalar_mass_kg(density, right, j, k));
    }

    [[nodiscard]] double v_dual_mass_kg(const Field3D& density,
                                        int i, int j, int k,
                                        int north) const noexcept
    {
        return 0.5 * (scalar_mass_kg(density, i, j, k)
                    + scalar_mass_kg(density, i, north, k));
    }

    [[nodiscard]] double w_dual_mass_kg(const Field3D& density,
                                        int i, int j, int k) const noexcept
    {
        return 0.5 * (scalar_mass_kg(density, i, j, k)
                    + scalar_mass_kg(density, i, j, k + 1));
    }

private:
    GridStateView grid_;
    const TerrainMetrics3D* terrain_ = nullptr;
    double horizontal_area_ = 0.0;
};

inline void column_geometry(const SimulationStateView& state,
                            int i, int j,
                            int k_begin, int k_end,
                            std::vector<double>& z_center,
                            std::vector<double>& z_interface,
                            std::vector<double>& layer_thickness,
                            bool height_above_ground = false)
{
    if (k_begin < 0 || k_end > state.grid.nz || k_begin >= k_end)
        throw std::out_of_range("physical column range is outside the grid");
    if (i < 0 || i >= state.grid.nr || j < 0 || j >= state.grid.nth)
        throw std::out_of_range(
            "physical column horizontal index is outside the grid");

    const std::size_t count = static_cast<std::size_t>(k_end - k_begin);
    z_center.resize(count);
    z_interface.resize(count + 1);
    layer_thickness.resize(count);
    const TerrainMetrics3D* const terrain = active_terrain_metrics(state);
    const auto& flat_z = state.grid.geometry.z;
    const auto& flat_z_face = state.grid.geometry.z_faces;
    const auto interface_at = [&](int k_face)
    {
        return terrain != nullptr
            ? terrain->z_face(i, j, k_face)
            : (flat_z_face.empty()
                ? static_cast<double>(k_face) * state.grid.dz
                : flat_z_face[static_cast<std::size_t>(k_face)]);
    };
    const double origin =
        height_above_ground ? interface_at(k_begin) : 0.0;
    for (std::size_t kk = 0; kk < count; ++kk)
    {
        const int k = k_begin + static_cast<int>(kk);
        const double center = terrain != nullptr
            ? terrain->z(i, j, k)
            : (flat_z.empty()
                ? (static_cast<double>(k) + 0.5) * state.grid.dz
                : flat_z[static_cast<std::size_t>(k)]);
        const double lower = interface_at(k);
        const double upper = interface_at(k + 1);
        if (!std::isfinite(center) || !std::isfinite(lower)
            || !std::isfinite(upper) || !(upper > lower)
            || !(center > lower) || !(center < upper))
        {
            throw std::invalid_argument(
                "physical column interface geometry is malformed");
        }
        z_center[kk] = center - origin;
        z_interface[kk] = lower - origin;
        layer_thickness[kk] = upper - lower;
    }
    z_interface[count] = interface_at(k_end) - origin;
}

inline ColumnGeometry column_geometry(const SimulationStateView& state,
                                      int i, int j,
                                      int k_begin, int k_end,
                                      bool height_above_ground = false)
{
    ColumnGeometry result;
    column_geometry(
        state, i, j, k_begin, k_end,
        result.z_center, result.z_interface, result.layer_thickness,
        height_above_ground);
    return result;
}

} // namespace tmv::physical_grid_metrics
