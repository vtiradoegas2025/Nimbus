#pragma once

#include <string>

/**
 * @file coordinate_system.hpp
 * @brief Coordinate-system enum and parsing helpers.
 *
 * The model is Cartesian (x, y, z). The CoordinateSystem type is retained as a
 * single-valued enum so that the `coordinate_system` configuration key, the
 * grid-metric dispatch helpers, and the CM1-style config surface continue to
 * compile against a stable surface; the cylindrical (r, theta, z) backend was
 * removed. A later cleanup pass may retire the type entirely.
 */

/**
 * @brief Identifies the horizontal coordinate system used by the model.
 *
 * Cartesian is the only supported value.
 */
enum class CoordinateSystem : int
{
    Cartesian = 0,
};

/**
 * @brief Identifies the grid staggering arrangement.
 *
 * - Collocated (Arakawa A-grid): all variables at cell centers.
 * - CGrid (Arakawa C-grid): velocities at cell faces, scalars at cell centers.
 *   Eliminates checkerboard pressure modes and provides conservative flux-form
 *   divergence. This is the production Cartesian path.
 */
enum class StaggerType : int
{
    Collocated = 0,
    CGrid      = 1,
};

/**
 * @brief Returns the canonical lowercase string label for a coordinate system.
 * @param system Coordinate system enum value.
 * @return Static string label ("cartesian").
 */
const char* coordinate_system_name(CoordinateSystem system);

/**
 * @brief Parses a coordinate-system identifier from a configuration string.
 * @param value Input identifier. Matching is case-insensitive. Accepted:
 *              - "cartesian", "cart"  -> CoordinateSystem::Cartesian
 * @param system_out Receives the parsed coordinate system on success.
 *                   Left unchanged on failure.
 * @return True on successful parse; false if the value is empty or
 *         unrecognized (e.g. the retired "cylindrical").
 */
bool parse_coordinate_system(const std::string& value, CoordinateSystem& system_out);

/**
 * @brief Representative scalar y-spacing (metres) for grid-metric consumers
 *        (the Smagorinsky filter width and strain metric, WENO/TVD spacing).
 *
 * In the Cartesian model `dtheta` is already the physical y-spacing in metres
 * (it is what the `grid.dy` config key sets), so the representative dy IS
 * `dtheta`. The `system` parameter is retained for call-site stability.
 *
 * @param system Active coordinate system (always Cartesian).
 * @param dr     x spacing in metres.
 * @param dtheta y spacing in metres.
 * @return Positive y-spacing in metres.
 * @throws std::invalid_argument if either Cartesian spacing is non-positive or
 *         non-finite.
 */
double representative_dy(CoordinateSystem system, double dr, double dtheta);

/**
 * @brief Returns the canonical lowercase string label for a stagger type.
 * @param stagger Stagger type enum value.
 * @return Static string label ("collocated" or "c_grid").
 */
const char* stagger_type_name(StaggerType stagger);

/**
 * @brief Parses a stagger-type identifier from a configuration string.
 * @param value Input identifier. Matching is case-insensitive. Accepted:
 *              - "collocated", "a_grid"  -> StaggerType::Collocated
 *              - "c_grid", "cgrid"       -> StaggerType::CGrid
 * @param stagger_out Receives the parsed stagger type on success.
 *                    Left unchanged on failure.
 * @return True on successful parse; false if the value is empty or
 *         unrecognized.
 */
bool parse_stagger_type(const std::string& value, StaggerType& stagger_out);
