/**
 * @file field_sanitization.hpp
 * @brief Field-level NaN/Inf sanitization and physical-bounds enforcement.
 *
 * Provides utilities for clamping prognostic fields into valid physical
 * ranges and replacing non-finite values. These are safety-net operations
 * called after each physics stage in the dynamics orchestrator.
 */

#pragma once

#include "core/field3d.hpp"

/**
 * @brief Replaces non-finite values with zero and clamps to [min_value, max_value].
 * @param field     The field to sanitize in-place.
 * @param min_value Lower bound for clamping.
 * @param max_value Upper bound for clamping.
 * @return Number of corrected samples.
 */
int sanitize_field_nonfinite_and_bounds(Field3D& field, float min_value, float max_value);

/**
 * @brief Sanitizes a field using bounds from its registered field contract.
 *
 * Looks up the field contract by `field_id`. If the contract defines min/max
 * bounds, delegates to `sanitize_field_nonfinite_and_bounds`. If no bounds
 * are registered, only replaces non-finite values with zero.
 *
 * @param field    The field to sanitize in-place.
 * @param field_id Contract identifier (e.g., "vorticity_r", "p_prime").
 * @return Number of corrected samples.
 */
int sanitize_field_nonfinite_and_contract_bounds(Field3D& field, const char* field_id);

/**
 * @brief Clamps all primary prognostic fields into valid physical ranges.
 *
 * Operates on the global fields: rho, p, theta, qv, qc, qr, qi, qs, qg, qh.
 * Each field is clamped using the corresponding `clamp_*` function from
 * `simulation.hpp`. Non-finite values are replaced with physically
 * reasonable defaults.
 *
 * @param stage Label used in guard diagnostics (e.g., "dynamics", "microphysics").
 * @return Number of corrected samples across all fields.
 */
int enforce_primary_state_bounds(const char* stage);
