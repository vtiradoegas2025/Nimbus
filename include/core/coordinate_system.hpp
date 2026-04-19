#pragma once

#include <string>

/**
 * @file coordinate_system.hpp
 * @brief Coordinate-system enum and parsing helpers.
 *
 * The model supports two horizontal coordinate systems:
 *   - Cylindrical (r, theta, z): the historical default. Suited to
 *     axisymmetric tornado-vortex modeling. Uses an antisymmetric radial-axis
 *     ghost cell at i = 0 and a periodic theta wraparound.
 *   - Cartesian (x, y, z): planned for non-axisymmetric flows such as the
 *     WK2002 supercell hodograph. The cylindrical scheme produces spurious
 *     body forces under non-axisymmetric base states (see Bug 7 in
 *     docs/Journey.md and the rationale in docs/CoordinateBackend_Plan.md).
 *
 * The active coordinate system is selected at runtime by the
 * `coordinate_system` configuration key. It defaults to Cylindrical so that
 * existing tornado configurations are unaffected by the introduction of this
 * dispatch.
 *
 * Phase A only adds the type and parsing surface; dynamics, advection, and
 * boundary-condition dispatch on this enum are wired in later sub-phases
 * (A.2 onward).
 */

/**
 * @brief Identifies the horizontal coordinate system used by the model.
 */
enum class CoordinateSystem : int
{
    Cylindrical = 0,
    Cartesian = 1,
};

/**
 * @brief Returns the canonical lowercase string label for a coordinate system.
 * @param system Coordinate system enum value.
 * @return Static string label ("cylindrical" or "cartesian"). Returns
 *         "cylindrical" for any unrecognized enum value as a safe default
 *         (the type is closed but a defensive default avoids UB if the value
 *         is ever read from raw memory).
 */
const char* coordinate_system_name(CoordinateSystem system);

/**
 * @brief Parses a coordinate-system identifier from a configuration string.
 * @param value Input identifier. Matching is case-insensitive. Accepted:
 *              - "cylindrical", "cyl"   → CoordinateSystem::Cylindrical
 *              - "cartesian",   "cart"  → CoordinateSystem::Cartesian
 * @param system_out Receives the parsed coordinate system on success.
 *                   Left unchanged on failure.
 * @return True on successful parse; false if the value is empty or
 *         unrecognized.
 *
 * Aliases are intentionally limited. We do not accept "polar" (ambiguous
 * with spherical) or "xy" (does not say anything about z handling).
 */
bool parse_coordinate_system(const std::string& value, CoordinateSystem& system_out);
